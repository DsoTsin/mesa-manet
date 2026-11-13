#include <stdlib.h>
#include <string.h>

#include "ac_barrier.h"
#include "ac_cmdbuf.h"
#include "ac_debug.h"
#include "ac_gpu_info.h"
#include "agc_kernels.h"
#include "nir.h"
#include "nir_builder.h"
#include "sid.h"
#include "test.h"
#include "util/macros.h"
#include "util/os_misc.h"
#include "util/u_math.h"

#define CODE_VA UINT64_C(0x123456789a00)
#define GUARD 0x5a5a5a5au
#define CODE_END_MARKER 0xbf9f0000u
#define CODE_END_MARKERS 5u
#define IB_DWORD_MULTIPLE 8u
#define BARRIER_MAX_DWORDS 16u
#define STANDALONE_IB_DWORDS 512u

struct target {
   const char *name;
   enum amd_gfx_level level;
   enum radeon_family family;
   uint32_t num_se;
   uint32_t num_cu;
};

static const struct target targets[] = {
   {"gfx10.3", GFX10_3, CHIP_NAVI21, 4, 80},
   {"gfx11", GFX11, CHIP_NAVI31, 6, 96},
   {"gfx11.5", GFX11_5, CHIP_STRIX_HALO, 2, 40},
   {"gfx12", GFX12, CHIP_GFX1201, 4, 64},
};

struct expectation {
   const char *name;
   struct nir_shader *(*build)(void);
   uint32_t user_data_dwords;
   uint32_t block[3];
   bool group_id[3];
   uint32_t thread_id_components;
   bool group_size;
};

static bool dump;

static void
describe_target(const struct target *target, struct radeon_info *info)
{
   memset(info, 0, sizeof(*info));
   snprintf(info->marketing_name, sizeof(info->marketing_name), "%s target description",
            target->name);
   info->gfx_level = target->level;
   info->family = target->family;
   info->has_graphics = true;
   info->num_se = target->num_se;
   info->max_se = target->num_se;
   info->max_sa_per_se = 2;
   info->num_cu = target->num_cu;
   info->max_good_cu_per_sa = target->num_cu / (target->num_se * 2);
   info->min_good_cu_per_sa = info->max_good_cu_per_sa;
   info->spi_cu_en = ~0u;
   info->instr_prefetch_distance = 3;
   if (target->level >= GFX12) {
      info->has_set_context_pairs = true;
      info->has_set_sh_pairs = true;
      info->has_set_uconfig_pairs = true;
   }
   ac_fill_compiler_info(info, NULL, false);
}

static nir_def *
user_address(nir_builder *b, nir_def *user)
{
   return nir_pack_64_2x32_split(b, nir_channel(b, user, 0), nir_channel(b, user, 1));
}

static nir_builder
begin(const char *name, uint16_t x, uint16_t y, uint16_t z, uint32_t user_data_dwords)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "%s", name);
   b.shader->info.workgroup_size[0] = x;
   b.shader->info.workgroup_size[1] = y;
   b.shader->info.workgroup_size[2] = z;
   b.shader->info.cs.user_data_components_amd = user_data_dwords;
   return b;
}

static struct nir_shader *
build_volume(void)
{
   nir_builder b = begin("test_volume", 4, 4, 4, 2);
   nir_def *address = user_address(&b, nir_load_user_data_amd(&b));
   nir_def *group = nir_load_workgroup_id(&b);
   nir_def *thread = nir_load_local_invocation_id(&b);
   nir_def *cell = nir_iadd(&b, nir_ishl_imm(&b, group, 2), thread);
   nir_def *index = nir_iadd(&b, nir_channel(&b, cell, 0),
                             nir_iadd(&b, nir_ishl_imm(&b, nir_channel(&b, cell, 1), 8),
                                      nir_ishl_imm(&b, nir_channel(&b, cell, 2), 16)));
   nir_def *offset = nir_u2u64(&b, nir_ishl_imm(&b, index, 2));
   nir_store_global(&b, index, nir_iadd(&b, address, offset), .align_mul = 4);
   return b.shader;
}

static struct nir_shader *
build_large_group(void)
{
   nir_builder b = begin("test_large_group", 16, 16, 1, 2);
   nir_def *address = user_address(&b, nir_load_user_data_amd(&b));
   nir_def *index = nir_load_local_invocation_index(&b);
   nir_def *offset = nir_u2u64(&b, nir_ishl_imm(&b, index, 2));
   nir_store_global(&b, index, nir_iadd(&b, address, offset), .align_mul = 4);
   return b.shader;
}

static struct nir_shader *
build_undeclared_user_data(void)
{
   nir_builder b = begin("test_undeclared", 64, 1, 1, 2);
   nir_def *user = nir_load_user_data_amd(&b);
   nir_def *offset = nir_u2u64(&b, nir_ishl_imm(&b, nir_channel(&b, user, 2), 2));
   nir_store_global(&b, nir_channel(&b, user, 2), nir_iadd(&b, user_address(&b, user), offset),
                    .align_mul = 4);
   return b.shader;
}

static struct nir_shader *
build_too_much_user_data(void)
{
   nir_builder b = begin("test_too_much", 64, 1, 1, 9);
   nir_def *user = nir_load_user_data_amd(&b);
   nir_store_global(&b, nir_channel(&b, user, 2), user_address(&b, user), .align_mul = 4);
   return b.shader;
}

static struct nir_shader *
build_too_many_threads(void)
{
   nir_builder b = begin("test_too_many_threads", 64, 64, 1, 2);
   nir_def *user = nir_load_user_data_amd(&b);
   nir_store_global(&b, nir_channel(&b, user, 0), user_address(&b, user), .align_mul = 4);
   return b.shader;
}

static struct nir_shader *
build_dispatch_size(void)
{
   nir_builder b = begin("test_dispatch_size", 64, 1, 1, 2);
   nir_def *user = nir_load_user_data_amd(&b);
   nir_def *groups = nir_load_num_workgroups(&b);
   nir_store_global(&b, nir_channel(&b, groups, 0), user_address(&b, user), .align_mul = 4);
   return b.shader;
}

static struct nir_shader *
build_fragment(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, NULL, "test_fragment");
   return b.shader;
}

static const struct expectation expectations[] = {
   {"fill32", agc_kernel_fill32, AGC_FILL32_USER_DATA_DWORDS, {64, 1, 1}, {true, false, false}, 0,
    false},
   {"copy32", agc_kernel_copy32, AGC_COPY32_USER_DATA_DWORDS, {64, 1, 1}, {true, false, false}, 0,
    false},
   {"display_convert", agc_kernel_display_convert, AGC_DISPLAY_CONVERT_USER_DATA_DWORDS, {8, 8, 1},
    {true, true, false}, 1, false},
   {"element_copy", agc_kernel_element_copy, AGC_ELEMENT_COPY_USER_DATA_DWORDS, {64, 1, 1},
    {true, true, true}, 0, false},
   {"display_overlay", agc_kernel_display_overlay, AGC_DISPLAY_OVERLAY_USER_DATA_DWORDS, {8, 8, 1},
    {true, true, false}, 1, false},
   {"volume", build_volume, 2, {4, 4, 4}, {true, true, true}, 2, false},
   {"large_group", build_large_group, 2, {16, 16, 1}, {false, false, false}, 0, true},
};

static char *
parse(const struct radeon_info *info, enum amd_ip_type ip_type, uint32_t *ib, uint32_t dwords,
      const char *name)
{
   const char *path = "agc_test_kernels_ib.txt";
   FILE *f = fopen(path, "w+");
   CHECK(f != NULL);
   if (!f)
      return NULL;
   struct ac_ib_parser parser = {
      .f = f,
      .ib = ib,
      .num_dw = dwords,
      .gfx_level = info->gfx_level,
      .family = info->family,
      .ip_type = ip_type,
   };
   ac_parse_ib(&parser, name);
   long size = ftell(f);
   CHECK(size > 0);
   char *text = size > 0 ? calloc(1, (size_t)size + 1) : NULL;
   CHECK(text != NULL);
   if (text) {
      rewind(f);
      size_t got = fread(text, 1, (size_t)size, f);
      text[got] = '\0';
      CHECK(strstr(text, "UNKNOWN") == NULL);
      CHECK(strstr(text, "Unknown packet type") == NULL);
      CHECK(strstr(text, "count in header too low") == NULL);
      CHECK(strstr(text, "unrecognized") == NULL);
      if (dump)
         fputs(text, stdout);
   }
   fclose(f);
   remove(path);
   return text;
}

static bool
find_sh_register(const uint32_t *ib, uint32_t dwords, uint32_t reg, uint32_t *value,
                 uint32_t *writes)
{
   const uint32_t wanted = (reg - SI_SH_REG_OFFSET) >> 2;
   bool found = false;
   uint32_t at = 0;
   *writes = 0;
   while (at < dwords) {
      uint32_t header = ib[at];
      if (header == PKT3_NOP_PAD) {
         at++;
         continue;
      }
      CHECK_EQ(PKT_TYPE_G(header), 3);
      if (PKT_TYPE_G(header) != 3)
         return false;
      uint32_t opcode = PKT3_IT_OPCODE_G(header);
      uint32_t body = PKT_COUNT_G(header) + 1;
      CHECK(at + 1 + body <= dwords);
      if (at + 1 + body > dwords)
         return false;
      const uint32_t *payload = ib + at + 1;
      if (opcode == PKT3_SET_SH_REG || opcode == PKT3_SET_SH_REG_INDEX) {
         uint32_t first = payload[0] & 0xffff;
         for (uint32_t i = 0; i + 1 < body; i++) {
            if (first + i == wanted) {
               *value = payload[1 + i];
               found = true;
               (*writes)++;
            }
         }
      } else if (opcode == PKT3_SET_SH_REG_PAIRS) {
         CHECK_EQ(body % 2, 0);
         for (uint32_t i = 0; i + 1 < body; i += 2) {
            if (payload[i] == wanted) {
               *value = payload[i + 1];
               found = true;
               (*writes)++;
            }
         }
      }
      at += 1 + body;
   }
   CHECK_EQ(at, dwords);
   return found;
}

static void
expect_sh_register(const uint32_t *ib, uint32_t dwords, uint32_t reg, uint32_t expected)
{
   uint32_t value = 0;
   uint32_t writes = 0;
   CHECK(find_sh_register(ib, dwords, reg, &value, &writes));
   CHECK_EQ(writes, 1);
   CHECK_EQ(value, expected);
}

static void
check_dispatch(const struct radeon_info *info, const struct expectation *expect,
               const struct agc_kernel *kernel)
{
   uint32_t user_data[AGC_KERNEL_MAX_USER_DATA];
   for (uint32_t i = 0; i < AGC_KERNEL_MAX_USER_DATA; i++)
      user_data[i] = 0xc0de0000u + i;
   const uint32_t groups[3] = {0x12345, 0x234, 0x34};

   uint32_t ib[AGC_KERNEL_MAX_DISPATCH_DWORDS + 1];
   memset(ib, 0x5a, sizeof(ib));
   uint32_t dwords = agc_kernel_dispatch_dwords(info, kernel);
   CHECK(dwords > 0 && dwords <= AGC_KERNEL_MAX_DISPATCH_DWORDS);
   CHECK_EQ(dwords, 24 + (expect->user_data_dwords ? 2 + expect->user_data_dwords : 0));
   uint32_t *end = agc_kernel_emit_dispatch(ib, info, kernel, CODE_VA, user_data, groups[0],
                                            groups[1], groups[2]);
   CHECK(end == ib + dwords);
   CHECK_EQ(ib[dwords], GUARD);
   if (end != ib + dwords)
      return;

   expect_sh_register(ib, dwords, R_00B830_COMPUTE_PGM_LO, (uint32_t)(CODE_VA >> 8));
   expect_sh_register(ib, dwords, R_00B834_COMPUTE_PGM_HI, (uint32_t)(CODE_VA >> 40));
   expect_sh_register(ib, dwords, R_00B848_COMPUTE_PGM_RSRC1, kernel->pgm_rsrc1);
   expect_sh_register(ib, dwords, R_00B84C_COMPUTE_PGM_RSRC2, kernel->pgm_rsrc2);
   expect_sh_register(ib, dwords, R_00B8A0_COMPUTE_PGM_RSRC3, kernel->pgm_rsrc3);
   expect_sh_register(ib, dwords, R_00B854_COMPUTE_RESOURCE_LIMITS, kernel->resource_limits);
   expect_sh_register(ib, dwords, R_00B81C_COMPUTE_NUM_THREAD_X, expect->block[0]);
   expect_sh_register(ib, dwords, R_00B820_COMPUTE_NUM_THREAD_Y, expect->block[1]);
   expect_sh_register(ib, dwords, R_00B824_COMPUTE_NUM_THREAD_Z, expect->block[2]);
   for (uint32_t i = 0; i < expect->user_data_dwords; i++)
      expect_sh_register(ib, dwords, R_00B900_COMPUTE_USER_DATA_0 + i * 4, user_data[i]);
   uint32_t value = 0;
   uint32_t writes = 0;
   CHECK(!find_sh_register(ib, dwords,
                           R_00B900_COMPUTE_USER_DATA_0 + expect->user_data_dwords * 4, &value,
                           &writes));

   const uint32_t *dispatch = end - 5;
   CHECK_EQ(dispatch[0], PKT3(PKT3_DISPATCH_DIRECT, 3, 0) | PKT3_SHADER_TYPE_S(1));
   CHECK_EQ(dispatch[1], groups[0]);
   CHECK_EQ(dispatch[2], groups[1]);
   CHECK_EQ(dispatch[3], groups[2]);
   CHECK_EQ(G_00B800_COMPUTE_SHADER_EN(dispatch[4]), 1);
   CHECK_EQ(G_00B800_FORCE_START_AT_000(dispatch[4]), 1);
   CHECK_EQ(G_00B800_PARTIAL_TG_EN(dispatch[4]), 0);
   CHECK_EQ(G_00B800_USE_THREAD_DIMENSIONS(dispatch[4]), 0);
   CHECK_EQ(G_00B800_ORDERED_APPEND_ENBL(dispatch[4]), 0);
   CHECK_EQ(G_00B800_CS_W32_EN(dispatch[4]), kernel->wave_size == 32);

   static const enum amd_ip_type rings[] = {AMD_IP_COMPUTE, AMD_IP_GFX};
   for (unsigned r = 0; r < ARRAY_SIZE(rings); r++) {
      char *text = parse(info, rings[r], ib, dwords, expect->name);
      if (!text)
         continue;
      CHECK(strstr(text, "COMPUTE_PGM_LO") != NULL);
      CHECK(strstr(text, "COMPUTE_PGM_HI") != NULL);
      CHECK(strstr(text, "COMPUTE_PGM_RSRC1") != NULL);
      CHECK(strstr(text, "COMPUTE_PGM_RSRC2") != NULL);
      CHECK(strstr(text, "COMPUTE_PGM_RSRC3") != NULL);
      CHECK(strstr(text, "COMPUTE_RESOURCE_LIMITS") != NULL);
      CHECK(strstr(text, "COMPUTE_NUM_THREAD_X") != NULL);
      CHECK(strstr(text, "COMPUTE_NUM_THREAD_Y") != NULL);
      CHECK(strstr(text, "COMPUTE_NUM_THREAD_Z") != NULL);
      CHECK(strstr(text, "COMPUTE_USER_DATA_0") != NULL);
      CHECK(strstr(text, "DISPATCH_DIRECT") != NULL);
      CHECK(strstr(text, "(shader_type=compute)") != NULL);
      free(text);
   }
}

static void
check_kernel(const struct radeon_info *info, const struct expectation *expect,
             struct agc_kernel *kernel)
{
   char log[256] = "unset";
   bool ok = agc_kernel_compile(info, expect->build(), kernel, log, sizeof(log));
   CHECK(ok);
   if (!ok) {
      fprintf(stderr, "%s on %s: %s\n", expect->name, info->marketing_name, log);
      return;
   }
   CHECK_EQ(log[0], 0);
   if (dump) {
      printf("=== %s on %s: %u code dwords, %u executable, rsrc1 0x%08x rsrc2 0x%08x rsrc3 0x%08x, "
             "limits 0x%08x, %u sgprs, %u vgprs\n%s\n",
             expect->name, info->marketing_name, kernel->code_dwords, kernel->exec_dwords,
             kernel->pgm_rsrc1, kernel->pgm_rsrc2, kernel->pgm_rsrc3, kernel->resource_limits,
             kernel->num_sgprs, kernel->num_vgprs, kernel->listing ? kernel->listing : "");
      printf("code %s %s:", expect->name, info->marketing_name);
      for (uint32_t i = 0; kernel->code && i < kernel->exec_dwords; i++)
         printf(" %08x", kernel->code[i]);
      printf("\n");
   }

   CHECK(kernel->code != NULL);
   CHECK(kernel->listing != NULL && kernel->listing[0] != 0);
   CHECK(kernel->exec_dwords > 0);
   CHECK(kernel->exec_dwords + CODE_END_MARKERS <= kernel->code_dwords);
   CHECK_EQ(kernel->code_dwords * 4 % AGC_KERNEL_CODE_ALIGNMENT, 0);
   CHECK(kernel->code_dwords * 4 >= kernel->exec_dwords * 4 + info->instr_prefetch_distance * 64);
   for (uint32_t i = 0; i < CODE_END_MARKERS; i++)
      CHECK_EQ(kernel->code[kernel->exec_dwords + i], CODE_END_MARKER);
   for (uint32_t i = kernel->exec_dwords + CODE_END_MARKERS; i < kernel->code_dwords; i++)
      CHECK_EQ(kernel->code[i], 0);
   CHECK_EQ(kernel->wave_size, 64);
   CHECK_EQ(kernel->user_data_dwords, expect->user_data_dwords);
   for (unsigned i = 0; i < 3; i++)
      CHECK_EQ(kernel->block[i], expect->block[i]);
   CHECK(kernel->num_vgprs > 0 && kernel->num_vgprs <= 256);
   CHECK(kernel->num_sgprs >= expect->user_data_dwords);

   CHECK_EQ(G_00B848_VGPRS(kernel->pgm_rsrc1), (kernel->num_vgprs - 1) / 4);
   CHECK_EQ(G_00B848_WGP_MODE(kernel->pgm_rsrc1), 1);
   CHECK_EQ(G_00B848_PRIV(kernel->pgm_rsrc1), 0);
   if (info->gfx_level < GFX12)
      CHECK_EQ(G_00B848_DX10_CLAMP(kernel->pgm_rsrc1), 1);
   else
      CHECK_EQ(G_00B848_WG_RR_EN(kernel->pgm_rsrc1), 0);

   CHECK_EQ(G_00B84C_USER_SGPR(kernel->pgm_rsrc2), expect->user_data_dwords);
   CHECK_EQ(G_00B84C_SCRATCH_EN(kernel->pgm_rsrc2), 0);
   CHECK_EQ(G_00B84C_TGID_X_EN(kernel->pgm_rsrc2), expect->group_id[0]);
   CHECK_EQ(G_00B84C_TGID_Y_EN(kernel->pgm_rsrc2), expect->group_id[1]);
   CHECK_EQ(G_00B84C_TGID_Z_EN(kernel->pgm_rsrc2), expect->group_id[2]);
   CHECK_EQ(G_00B84C_TIDIG_COMP_CNT(kernel->pgm_rsrc2), expect->thread_id_components);
   CHECK_EQ(G_00B84C_TG_SIZE_EN(kernel->pgm_rsrc2), expect->group_size);
   CHECK_EQ(G_00B84C_LDS_SIZE(kernel->pgm_rsrc2), 0);
   CHECK_EQ(G_00B84C_EXCP_EN(kernel->pgm_rsrc2), 0);

   uint32_t prefetch = info->gfx_level >= GFX12   ? G_00B8A0_INST_PREF_SIZE_GFX12(kernel->pgm_rsrc3)
                       : info->gfx_level >= GFX11 ? G_00B8A0_INST_PREF_SIZE_GFX11(kernel->pgm_rsrc3)
                                                  : 0;
   if (info->gfx_level >= GFX11)
      CHECK(prefetch > 0 && prefetch * 128 >= kernel->exec_dwords * 4);
   else
      CHECK_EQ(kernel->pgm_rsrc3, 0);

   uint32_t threads = expect->block[0] * expect->block[1] * expect->block[2];
   CHECK_EQ(G_00B854_CU_GROUP_COUNT(kernel->resource_limits), threads <= 64 ? 1 : 0);
   CHECK_EQ(G_00B854_WAVES_PER_SH(kernel->resource_limits), 0);

   check_dispatch(info, expect, kernel);
}

static void
check_queue_setup(const struct radeon_info *info)
{
   static const struct {
      bool compute;
      enum amd_ip_type ip_type;
   } rings[] = {
      {true, AMD_IP_COMPUTE},
      {false, AMD_IP_GFX},
   };
   for (unsigned r = 0; r < ARRAY_SIZE(rings); r++) {
      uint32_t ib[AGC_KERNEL_MAX_QUEUE_SETUP_DWORDS + 1];
      memset(ib, 0x5a, sizeof(ib));
      uint32_t dwords = agc_kernel_queue_setup_dwords(info, rings[r].compute);
      CHECK(dwords > 0 && dwords <= AGC_KERNEL_MAX_QUEUE_SETUP_DWORDS);
      uint32_t *end = agc_kernel_emit_queue_setup(ib, info, rings[r].compute);
      CHECK(end == ib + dwords);
      CHECK_EQ(ib[dwords], GUARD);
      if (end != ib + dwords)
         continue;

      uint32_t value = 0;
      uint32_t writes = 0;
      CHECK(find_sh_register(ib, dwords, R_00B858_COMPUTE_STATIC_THREAD_MGMT_SE0, &value, &writes));
      CHECK_EQ(value, 0xffffffffu);
      for (uint32_t se = 0; se < 4; se++) {
         uint32_t reg = se < 2 ? R_00B858_COMPUTE_STATIC_THREAD_MGMT_SE0 + se * 4
                               : R_00B864_COMPUTE_STATIC_THREAD_MGMT_SE2 + (se - 2) * 4;
         CHECK(find_sh_register(ib, dwords, reg, &value, &writes));
         CHECK_EQ(value, se < info->max_se ? 0xffffffffu : 0);
      }
      CHECK(find_sh_register(ib, dwords, R_00B810_COMPUTE_START_X, &value, &writes));
      CHECK_EQ(value, 0);
      CHECK(!find_sh_register(ib, dwords, R_00B848_COMPUTE_PGM_RSRC1, &value, &writes));

      char *text = parse(info, rings[r].ip_type, ib, dwords, "queue setup");
      if (text) {
         CHECK(strstr(text, "COMPUTE_START_X") != NULL);
         CHECK(strstr(text, "COMPUTE_USER_ACCUM_0") != NULL);
         if (info->gfx_level >= GFX11)
            CHECK(strstr(text, "COMPUTE_DISPATCH_INTERLEAVE") != NULL);
         free(text);
      }
   }
}

static uint32_t
emit_barrier(const struct radeon_info *info, enum amd_ip_type ip_type, struct ac_cmdbuf *cs,
             enum ac_barrier_flags flags)
{
   const struct ac_barrier_state state = {
      .flags = flags,
   };
   enum ac_rgp_flush_bits flush_bits = 0;
   bool context_roll = false;
   uint32_t before = cs->cdw;
   ac_emit_barrier(cs, info->gfx_level, ip_type, &state, &context_roll, &flush_bits);
   return cs->cdw - before;
}

static const char *
find_last(const char *text, const char *needle)
{
   const char *last = NULL;
   for (const char *at = strstr(text, needle); at; at = strstr(at + 1, needle))
      last = at;
   return last;
}

static void
check_standalone_ib(const struct radeon_info *info, const struct agc_kernel *kernel)
{
   static const struct {
      bool compute;
      enum amd_ip_type ip_type;
   } rings[] = {
      {true, AMD_IP_COMPUTE},
      {false, AMD_IP_GFX},
   };
   uint32_t user_data[AGC_KERNEL_MAX_USER_DATA] = {0};

   for (unsigned r = 0; r < ARRAY_SIZE(rings); r++) {
      uint32_t ib[STANDALONE_IB_DWORDS];
      struct ac_cmdbuf cs = {
         .buf = ib,
         .max_dw = STANDALONE_IB_DWORDS,
      };
      uint32_t *end = agc_kernel_emit_queue_setup(ib, info, rings[r].compute);
      CHECK(end != NULL);
      if (!end)
         continue;
      cs.cdw = (uint32_t)(end - ib);

      uint32_t barrier = emit_barrier(info, rings[r].ip_type, &cs, AC_BARRIER_ALL_COMPUTE);
      CHECK(barrier > 0 && barrier <= BARRIER_MAX_DWORDS);

      end = agc_kernel_emit_dispatch(ib + cs.cdw, info, kernel, CODE_VA, user_data, 3, 1, 1);
      CHECK(end != NULL);
      if (!end)
         continue;
      cs.cdw = (uint32_t)(end - ib);

      barrier = emit_barrier(info, rings[r].ip_type, &cs, AC_BARRIER_SYNC_CS | AC_BARRIER_INV_L2);
      CHECK(barrier > 0 && barrier <= BARRIER_MAX_DWORDS);

      uint32_t dwords = align(cs.cdw, IB_DWORD_MULTIPLE);
      if (dwords - cs.cdw == 1) {
         ib[cs.cdw] = PKT3_NOP_PAD;
      } else if (dwords > cs.cdw) {
         ib[cs.cdw] = PKT3(PKT3_NOP, dwords - cs.cdw - 2, 0);
         memset(ib + cs.cdw + 1, 0, (dwords - cs.cdw - 1) * sizeof(uint32_t));
      }
      CHECK(dwords <= STANDALONE_IB_DWORDS);
      CHECK_EQ(dwords % IB_DWORD_MULTIPLE, 0);

      uint32_t value = 0;
      uint32_t writes = 0;
      CHECK(find_sh_register(ib, dwords, R_00B858_COMPUTE_STATIC_THREAD_MGMT_SE0, &value, &writes));
      CHECK_EQ(value, 0xffffffffu);
      CHECK(find_sh_register(ib, dwords, R_00B830_COMPUTE_PGM_LO, &value, &writes));
      CHECK_EQ(value, (uint32_t)(CODE_VA >> 8));

      char *text = parse(info, rings[r].ip_type, ib, dwords, "standalone kernel IB");
      if (!text)
         continue;
      const char *first_acquire = strstr(text, "ACQUIRE_MEM");
      const char *last_acquire = find_last(text, "ACQUIRE_MEM");
      const char *dispatch = strstr(text, "DISPATCH_DIRECT");
      CHECK(first_acquire != NULL && dispatch != NULL && last_acquire != NULL);
      CHECK(first_acquire < dispatch && dispatch < last_acquire);
      CHECK(strstr(text, "CS_PARTIAL_FLUSH") != NULL);
      free(text);
   }
}

static void
check_refusals(const struct radeon_info *info, const struct agc_kernel *kernel)
{
   static struct nir_shader *(*const refused[])(void) = {
      build_undeclared_user_data, build_too_much_user_data, build_too_many_threads,
      build_dispatch_size,        build_fragment,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(refused); i++) {
      struct agc_kernel scratch;
      char log[256] = "";
      memset(&scratch, 0x5a, sizeof(scratch));
      CHECK(!agc_kernel_compile(info, refused[i](), &scratch, log, sizeof(log)));
      CHECK(log[0] != 0);
      CHECK(scratch.code == NULL && scratch.listing == NULL && scratch.code_dwords == 0);
      if (dump)
         printf("refusal %u: %s\n", i, log);
   }

   struct agc_kernel scratch;
   char log[256] = "";
   struct radeon_info other = *info;
   other.gfx_level = GFX9;
   CHECK(!agc_kernel_compile(&other, agc_kernel_fill32(), &scratch, log, sizeof(log)));
   CHECK(log[0] != 0);
   other = *info;
   memset(&other.compiler_info, 0, sizeof(other.compiler_info));
   CHECK(!agc_kernel_compile(&other, agc_kernel_fill32(), &scratch, log, sizeof(log)));
   CHECK(log[0] != 0);
   other = *info;
   other.num_se = 0;
   CHECK(!agc_kernel_compile(&other, agc_kernel_fill32(), &scratch, NULL, 0));
   CHECK(!agc_kernel_compile(info, NULL, &scratch, log, sizeof(log)));
   CHECK(!agc_kernel_compile(NULL, agc_kernel_fill32(), &scratch, log, sizeof(log)));

   uint32_t ib[AGC_KERNEL_MAX_DISPATCH_DWORDS];
   uint32_t user_data[AGC_KERNEL_MAX_USER_DATA] = {0};
   CHECK(agc_kernel_emit_dispatch(ib, info, kernel, CODE_VA + 4, user_data, 1, 1, 1) == NULL);
   CHECK(agc_kernel_emit_dispatch(ib, info, kernel, 0, user_data, 1, 1, 1) == NULL);
   CHECK(agc_kernel_emit_dispatch(ib, info, kernel, UINT64_C(1) << 48, user_data, 1, 1, 1) == NULL);
   CHECK(agc_kernel_emit_dispatch(ib, info, kernel, CODE_VA, NULL, 1, 1, 1) == NULL);
   CHECK(agc_kernel_emit_dispatch(ib, info, kernel, CODE_VA, user_data, 0, 1, 1) == NULL);
   CHECK(agc_kernel_emit_dispatch(ib, info, kernel, CODE_VA, user_data, 1, 0x10000, 1) == NULL);
   CHECK(agc_kernel_emit_dispatch(ib, info, kernel, CODE_VA, user_data, 1, 1, 0) == NULL);
   CHECK(agc_kernel_emit_dispatch(NULL, info, kernel, CODE_VA, user_data, 1, 1, 1) == NULL);
   memset(&scratch, 0, sizeof(scratch));
   CHECK(agc_kernel_emit_dispatch(ib, info, &scratch, CODE_VA, user_data, 1, 1, 1) == NULL);
   CHECK_EQ(agc_kernel_dispatch_dwords(info, &scratch), 0);

   other = *info;
   other.spi_cu_en = 0;
   CHECK_EQ(agc_kernel_queue_setup_dwords(&other, true), 0);
   CHECK(agc_kernel_emit_queue_setup(ib, &other, true) == NULL);
   other = *info;
   other.max_se = 0;
   CHECK_EQ(agc_kernel_queue_setup_dwords(&other, false), 0);

   agc_kernel_free(&scratch);
   agc_kernel_free(NULL);
}

static void
check_helpers(void)
{
   uint32_t fill[AGC_FILL32_USER_DATA_DWORDS];
   agc_kernel_fill32_user_data(fill, UINT64_C(0x1122334455667788), 77, 0xdeadbeef);
   CHECK_EQ(fill[AGC_FILL32_ADDRESS_LO], 0x55667788);
   CHECK_EQ(fill[AGC_FILL32_ADDRESS_HI], 0x11223344);
   CHECK_EQ(fill[AGC_FILL32_COUNT], 77);
   CHECK_EQ(fill[AGC_FILL32_VALUE], 0xdeadbeef);

   uint32_t copy[AGC_COPY32_USER_DATA_DWORDS];
   agc_kernel_copy32_user_data(copy, UINT64_C(0x0102030405060708), UINT64_C(0x1112131415161718),
                               9);
   CHECK_EQ(copy[AGC_COPY32_DST_LO], 0x05060708);
   CHECK_EQ(copy[AGC_COPY32_DST_HI], 0x01020304);
   CHECK_EQ(copy[AGC_COPY32_SRC_LO], 0x15161718);
   CHECK_EQ(copy[AGC_COPY32_SRC_HI], 0x11121314);
   CHECK_EQ(copy[AGC_COPY32_COUNT], 9);

   uint32_t convert[AGC_DISPLAY_CONVERT_USER_DATA_DWORDS];
   agc_kernel_display_convert_user_data(convert, UINT64_C(0x2122232425262728),
                                        UINT64_C(0x3132333435363738), 1920, 1080, 7680);
   CHECK_EQ(convert[AGC_DISPLAY_CONVERT_DESCRIPTOR_LO], 0x25262728);
   CHECK_EQ(convert[AGC_DISPLAY_CONVERT_DESCRIPTOR_HI], 0x21222324);
   CHECK_EQ(convert[AGC_DISPLAY_CONVERT_DST_LO], 0x35363738);
   CHECK_EQ(convert[AGC_DISPLAY_CONVERT_DST_HI], 0x31323334);
   CHECK_EQ(convert[AGC_DISPLAY_CONVERT_WIDTH], 1920);
   CHECK_EQ(convert[AGC_DISPLAY_CONVERT_HEIGHT], 1080);
   CHECK_EQ(convert[AGC_DISPLAY_CONVERT_PITCH_BYTES], 7680);

   CHECK_EQ(agc_kernel_groups(1, 64), 1);
   CHECK_EQ(agc_kernel_groups(64, 64), 1);
   CHECK_EQ(agc_kernel_groups(65, 64), 2);
   CHECK_EQ(agc_kernel_groups(1920, 8), 240);
   CHECK_EQ(agc_kernel_groups(AGC_FILL32_MAX_DWORDS, 64), 0x1000000);
}

int
main(int argc, char **argv)
{
   dump = argc > 1 && !strcmp(argv[1], "--dump");
   os_set_option("ACO_DEBUG", "validateir,validatera,validate-livevars,validateopt", true);

   struct agc_kernel fill[ARRAY_SIZE(targets)];
   memset(fill, 0, sizeof(fill));
   for (unsigned t = 0; t < ARRAY_SIZE(targets); t++) {
      struct radeon_info info;
      describe_target(&targets[t], &info);
      for (unsigned e = 0; e < ARRAY_SIZE(expectations); e++) {
         struct agc_kernel kernel;
         check_kernel(&info, &expectations[e], &kernel);
         if (e == 0) {
            struct agc_kernel again;
            char log[64];
            CHECK(agc_kernel_compile(&info, agc_kernel_fill32(), &again, log, sizeof(log)));
            CHECK_EQ(again.code_dwords, kernel.code_dwords);
            CHECK(again.code && kernel.code &&
                  !memcmp(again.code, kernel.code, kernel.code_dwords * sizeof(uint32_t)));
            agc_kernel_free(&again);
            check_refusals(&info, &kernel);
            check_standalone_ib(&info, &kernel);
            fill[t] = kernel;
         } else {
            agc_kernel_free(&kernel);
         }
      }
      check_queue_setup(&info);
   }

   unsigned different = 0;
   for (unsigned t = 1; t < ARRAY_SIZE(targets); t++) {
      if (!fill[0].code || !fill[t].code)
         continue;
      uint32_t common = MIN2(fill[0].exec_dwords, fill[t].exec_dwords);
      different += fill[0].exec_dwords != fill[t].exec_dwords ||
                   memcmp(fill[0].code, fill[t].code, common * sizeof(uint32_t)) != 0;
   }
   CHECK(different > 0);
   for (unsigned t = 0; t < ARRAY_SIZE(targets); t++)
      agc_kernel_free(&fill[t]);

   check_helpers();
   return TEST_RESULT();
}
