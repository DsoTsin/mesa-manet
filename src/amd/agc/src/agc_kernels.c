#include "agc_kernels.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac_binary.h"
#include "ac_cmdbuf.h"
#include "ac_gpu_info.h"
#include "ac_nir.h"
#include "ac_pm4.h"
#include "ac_shader_args.h"
#include "ac_shader_util.h"
#include "aco_interface.h"
#include "agc_debug.h"
#include "nir.h"
#include "nir_builder.h"
#include "sid.h"
#include "util/u_math.h"

#if defined(__clang__) && defined(_MSC_VER)
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#endif

#define AGC_KERNEL_WAVE_SIZE 64u
#define AGC_KERNEL_MAX_THREADS 1024u
#define AGC_KERNEL_DISPATCH_FIXED_DWORDS 24u
#define AGC_KERNEL_DISPATCH_INTERLEAVE 64u

struct agc_kernel_usage {
   bool workgroup_id[3];
   bool local_id[3];
   bool tg_size;
   const char *unsupported;
};

struct agc_kernel_user_data {
   const struct ac_shader_args *args;
   const struct ac_arg *slots;
   uint32_t count;
};

struct agc_kernel_binary {
   struct ac_shader_config config;
   uint32_t *code;
   uint32_t code_dwords;
   uint32_t padded_dwords;
   uint32_t exec_bytes;
   uint32_t patched_symbols;
   char *listing;
   bool out_of_memory;
   enum amd_gfx_level gfx_level;
   uint32_t prefetch_distance;
};

static bool AGC_PRINTF(3, 4)
agc_kernel_fail(char *log, size_t log_bytes, const char *format, ...)
{
   if (log && log_bytes) {
      va_list args;
      va_start(args, format);
      vsnprintf(log, log_bytes, format, args);
      va_end(args);
   }
   return false;
}

static nir_builder
agc_kernel_begin(const char *name, uint32_t x, uint32_t y, uint32_t z, uint32_t user_data_dwords)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "%s", name);
   b.shader->info.workgroup_size[0] = (uint16_t)x;
   b.shader->info.workgroup_size[1] = (uint16_t)y;
   b.shader->info.workgroup_size[2] = (uint16_t)z;
   b.shader->info.cs.user_data_components_amd = user_data_dwords;
   return b;
}

static nir_def *
agc_kernel_address(nir_builder *b, nir_def *user, unsigned lo)
{
   return nir_pack_64_2x32_split(b, nir_channel(b, user, lo), nir_channel(b, user, lo + 1));
}

static nir_def *
agc_kernel_at(nir_builder *b, nir_def *address, nir_def *byte_offset)
{
   return nir_iadd(b, address, nir_u2u64(b, byte_offset));
}

struct nir_shader *
agc_kernel_fill32(void)
{
   nir_builder b = agc_kernel_begin("agc_fill32", 64, 1, 1, AGC_FILL32_USER_DATA_DWORDS);
   nir_def *user = nir_load_user_data_amd(&b);
   nir_def *address = agc_kernel_address(&b, user, AGC_FILL32_ADDRESS_LO);
   nir_def *count = nir_channel(&b, user, AGC_FILL32_COUNT);
   nir_def *value = nir_channel(&b, user, AGC_FILL32_VALUE);
   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

   nir_push_if(&b, nir_ult(&b, index, count));
   {
      nir_def *offset = nir_ishl_imm(&b, index, 2);
      nir_store_global(&b, value, agc_kernel_at(&b, address, offset), .align_mul = 4);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

struct nir_shader *
agc_kernel_copy32(void)
{
   nir_builder b = agc_kernel_begin("agc_copy32", 64, 1, 1, AGC_COPY32_USER_DATA_DWORDS);
   nir_def *user = nir_load_user_data_amd(&b);
   nir_def *dst = agc_kernel_address(&b, user, AGC_COPY32_DST_LO);
   nir_def *src = agc_kernel_address(&b, user, AGC_COPY32_SRC_LO);
   nir_def *count = nir_channel(&b, user, AGC_COPY32_COUNT);
   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

   nir_push_if(&b, nir_ult(&b, index, count));
   {
      nir_def *offset = nir_ishl_imm(&b, index, 2);
      nir_def *value = nir_load_global(&b, 1, 32, agc_kernel_at(&b, src, offset), .align_mul = 4);
      nir_store_global(&b, value, agc_kernel_at(&b, dst, offset), .align_mul = 4);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

struct nir_shader *
agc_kernel_display_convert(void)
{
   nir_builder b = agc_kernel_begin("agc_display_convert", 8, 8, 1,
                                    AGC_DISPLAY_CONVERT_USER_DATA_DWORDS);
   nir_def *user = nir_load_user_data_amd(&b);
   nir_def *descriptor_address = agc_kernel_address(&b, user, AGC_DISPLAY_CONVERT_DESCRIPTOR_LO);
   nir_def *dst = agc_kernel_address(&b, user, AGC_DISPLAY_CONVERT_DST_LO);
   nir_def *width = nir_channel(&b, user, AGC_DISPLAY_CONVERT_WIDTH);
   nir_def *height = nir_channel(&b, user, AGC_DISPLAY_CONVERT_HEIGHT);
   nir_def *pitch = nir_channel(&b, user, AGC_DISPLAY_CONVERT_PITCH_BYTES);
   nir_def *descriptor = ac_nir_load_smem(&b, 8, descriptor_address, nir_imm_int(&b, 0), 4, 0);
   nir_def *id = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, id, 0);
   nir_def *y = nir_channel(&b, id, 1);

   nir_push_if(&b, nir_iand(&b, nir_ult(&b, x, width), nir_ult(&b, y, height)));
   {
      nir_def *texel = nir_txf(&b, nir_vec2(&b, x, y), .texture_handle = descriptor,
                               .dim = GLSL_SAMPLER_DIM_2D, .dest_type = nir_type_float32);
      nir_def *bytes =
         nir_f2u32(&b, nir_fround_even(&b, nir_fmul_imm(&b, nir_fsat(&b, texel), 255.0)));
      nir_def *low = nir_ior(&b, nir_channel(&b, bytes, 0),
                             nir_ishl_imm(&b, nir_channel(&b, bytes, 1), 8));
      nir_def *high = nir_ior(&b, nir_ishl_imm(&b, nir_channel(&b, bytes, 2), 16),
                              nir_ishl_imm(&b, nir_channel(&b, bytes, 3), 24));
      nir_def *offset = nir_iadd(&b, nir_imul(&b, y, pitch), nir_ishl_imm(&b, x, 2));
      nir_store_global(&b, nir_ior(&b, low, high), agc_kernel_at(&b, dst, offset), .align_mul = 4);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

struct nir_shader *
agc_kernel_display_overlay(void)
{
   nir_builder b = agc_kernel_begin("agc_display_overlay", 8, 8, 1, AGC_DISPLAY_OVERLAY_USER_DATA_DWORDS);
   nir_def *user = nir_load_user_data_amd(&b);
   nir_def *descriptor_address = agc_kernel_address(&b, user, AGC_DISPLAY_OVERLAY_DESCRIPTOR_LO);
   nir_def *dst = agc_kernel_address(&b, user, AGC_DISPLAY_OVERLAY_DST_LO);
   nir_def *width = nir_channel(&b, user, AGC_DISPLAY_OVERLAY_WIDTH);
   nir_def *height = nir_channel(&b, user, AGC_DISPLAY_OVERLAY_HEIGHT);
   nir_def *pitch = nir_channel(&b, user, AGC_DISPLAY_OVERLAY_PITCH_BYTES);
   nir_def *size = nir_channel(&b, user, AGC_DISPLAY_OVERLAY_SIZE);
   nir_def *overlay_width = nir_iand_imm(&b, size, 0xffff);
   nir_def *overlay_height = nir_ushr_imm(&b, size, 16);
   nir_def *descriptor = ac_nir_load_smem(&b, 8, descriptor_address, nir_imm_int(&b, 0), 4, 0);
   nir_def *id = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, id, 0);
   nir_def *y = nir_channel(&b, id, 1);

   nir_push_if(&b, nir_iand(&b, nir_ult(&b, x, width), nir_ult(&b, y, height)));
   {
      nir_def *ox = nir_udiv(&b, nir_imul(&b, x, overlay_width), width);
      nir_def *oy = nir_udiv(&b, nir_imul(&b, y, overlay_height), height);
      nir_def *overlay = nir_fsat(&b, nir_txf(&b, nir_vec2(&b, ox, oy), .texture_handle = descriptor,
                                              .dim = GLSL_SAMPLER_DIM_2D, .dest_type = nir_type_float32));
      nir_def *offset = nir_iadd(&b, nir_imul(&b, y, pitch), nir_ishl_imm(&b, x, 2));
      nir_def *address = agc_kernel_at(&b, dst, offset);
      nir_def *under = nir_unpack_unorm_4x8(&b, nir_load_global(&b, 1, 32, address, .align_mul = 4));
      nir_def *keep = nir_fsub_imm(&b, 1.0, nir_channel(&b, overlay, 3));
      nir_def *color = nir_fsat(&b, nir_ffma(&b, nir_trim_vector(&b, under, 3), nir_replicate(&b, keep, 3),
                                             nir_trim_vector(&b, overlay, 3)));
      nir_def *pixel = nir_vec4(&b, nir_channel(&b, color, 0), nir_channel(&b, color, 1), nir_channel(&b, color, 2),
                                nir_imm_float(&b, 1.0));
      nir_store_global(&b, nir_pack_unorm_4x8(&b, pixel), address, .align_mul = 4);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

static nir_def *
agc_kernel_element_address(nir_builder *b, nir_def *const *layout, nir_def *x, nir_def *y, nir_def *slice)
{
   nir_def *coordinate =
      nir_ior(b,
              nir_ior(b, nir_iand_imm(b, nir_iadd(b, x, layout[8]), 0xfff),
                      nir_ishl_imm(b, nir_iand_imm(b, nir_iadd(b, y, layout[9]), 0xfff), 12)),
              nir_ishl_imm(b, nir_iand_imm(b, slice, 0xff), 24));
   nir_def *inside = nir_imm_int(b, 0);
   for (unsigned bit = 0; bit < 16; bit++) {
      nir_def *parity = nir_iand_imm(b, nir_bit_count(b, nir_iand(b, layout[16 + bit], coordinate)), 1);
      inside = nir_ior(b, inside, nir_ishl_imm(b, parity, bit));
   }
   inside = nir_ixor(b, inside, layout[10]);
   nir_def *block = nir_iadd(b, nir_ushr(b, x, layout[5]), nir_imul(b, nir_ushr(b, y, layout[6]), layout[4]));
   nir_def *offset = nir_iadd(b, nir_ishl(b, block, layout[7]), inside);
   nir_def *start = nir_pack_64_2x32_split(b, layout[0], layout[1]);
   nir_def *stride = nir_pack_64_2x32_split(b, layout[2], layout[3]);
   nir_def *base = nir_iadd(b, start, nir_imul(b, nir_u2u64(b, slice), stride));
   return nir_iadd(b, base, nir_u2u64(b, offset));
}

static void
agc_kernel_element_move(nir_builder *b, nir_def *src, nir_def *dst, unsigned components, unsigned bit_size,
                        unsigned align)
{
   nir_def *value = nir_load_global(b, components, bit_size, src, .align_mul = align);
   nir_store_global(b, value, dst, .align_mul = align);
}

struct nir_shader *
agc_kernel_element_copy(void)
{
   nir_builder b = agc_kernel_begin("agc_element_copy", AGC_ELEMENT_COPY_GROUP_THREADS, 1, 1,
                                    AGC_ELEMENT_COPY_USER_DATA_DWORDS);
   nir_def *user = nir_load_user_data_amd(&b);
   nir_def *params = agc_kernel_address(&b, user, AGC_ELEMENT_COPY_PARAMS_LO);
   nir_def *p[AGC_ELEMENT_COPY_PARAM_DWORDS];
   for (unsigned i = 0; i < AGC_ELEMENT_COPY_PARAM_DWORDS; i += 4) {
      nir_def *words = ac_nir_load_smem(&b, 4, params, nir_imm_int(&b, (int)(i * 4)), 4, 0);
      for (unsigned k = 0; k < 4; k++)
         p[i + k] = nir_channel(&b, words, k);
   }
   nir_def *group = nir_load_workgroup_id(&b);
   nir_def *local = nir_load_local_invocation_id(&b);
   nir_def *x = nir_iadd(&b, nir_ishl_imm(&b, nir_channel(&b, group, 0), 6), nir_channel(&b, local, 0));
   nir_def *y = nir_channel(&b, group, 1);
   nir_def *slice = nir_channel(&b, group, 2);
   nir_def *log2 = p[66];

   nir_push_if(&b, nir_iand(&b, nir_ult(&b, x, p[64]), nir_ult(&b, y, p[65])));
   {
      nir_def *src = agc_kernel_element_address(&b, p, x, y, slice);
      nir_def *dst = agc_kernel_element_address(&b, p + AGC_ELEMENT_COPY_LAYOUT_DWORDS, x, y, slice);
      nir_push_if(&b, nir_ieq_imm(&b, log2, 4));
      agc_kernel_element_move(&b, src, dst, 4, 32, 16);
      nir_push_else(&b, NULL);
      nir_push_if(&b, nir_ieq_imm(&b, log2, 3));
      agc_kernel_element_move(&b, src, dst, 2, 32, 8);
      nir_push_else(&b, NULL);
      nir_push_if(&b, nir_ieq_imm(&b, log2, 2));
      agc_kernel_element_move(&b, src, dst, 1, 32, 4);
      nir_push_else(&b, NULL);
      nir_push_if(&b, nir_ieq_imm(&b, log2, 1));
      agc_kernel_element_move(&b, src, dst, 1, 16, 2);
      nir_push_else(&b, NULL);
      agc_kernel_element_move(&b, src, dst, 1, 8, 1);
      nir_pop_if(&b, NULL);
      nir_pop_if(&b, NULL);
      nir_pop_if(&b, NULL);
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

static void
agc_kernel_optimize(nir_shader *nir, enum amd_gfx_level gfx_level)
{
   const nir_opt_peephole_select_options peephole = {
      .limit = 8,
      .indirect_load_ok = true,
      .expensive_alu_ok = true,
      .discard_ok = true,
   };
   bool progress;
   NIR_PASS(_, nir, nir_lower_alu_width, ac_nir_opt_vectorize_cb, &gfx_level);
   do {
      progress = false;
      NIR_PASS(_, nir, nir_lower_phis_to_scalar, ac_nir_lower_phis_to_scalar_cb, NULL);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_peephole_select, &peephole);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_undef);
   } while (progress);
}

static void
agc_kernel_optimize_late(nir_shader *nir)
{
   const nir_opt_peephole_select_options peephole = {
      .limit = 8,
      .indirect_load_ok = true,
      .expensive_alu_ok = true,
      .discard_ok = true,
   };
   bool progress;
   do {
      progress = false;
      NIR_PASS(_, nir, nir_opt_copy_prop);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_peephole_select, &peephole);
      NIR_PASS(_, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_algebraic);
   } while (progress);

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_algebraic_late);
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_copy_prop);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_opt_cse);
   } while (progress);
}

static void
agc_kernel_gather(nir_shader *nir, struct agc_kernel_usage *usage)
{
   memset(usage, 0, sizeof(*usage));
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            switch (intrin->intrinsic) {
            case nir_intrinsic_load_workgroup_id:
            case nir_intrinsic_load_local_invocation_id: {
               bool *used = intrin->intrinsic == nir_intrinsic_load_workgroup_id
                               ? usage->workgroup_id
                               : usage->local_id;
               unsigned mask = nir_def_components_read(&intrin->def);
               for (unsigned i = 0; i < 3; i++)
                  used[i] |= (mask >> i) & 1;
               break;
            }
            case nir_intrinsic_load_local_invocation_index:
               usage->tg_size = true;
               break;
            case nir_intrinsic_load_subgroup_id:
            case nir_intrinsic_load_num_subgroups:
               usage->unsupported = "subgroup counts";
               break;
            case nir_intrinsic_load_num_workgroups:
               usage->unsupported = "the dispatch size as a system value";
               break;
            case nir_intrinsic_load_shared:
            case nir_intrinsic_store_shared:
            case nir_intrinsic_shared_atomic:
            case nir_intrinsic_shared_atomic_swap:
               usage->unsupported = "shared memory";
               break;
            case nir_intrinsic_load_scratch:
            case nir_intrinsic_store_scratch:
               usage->unsupported = "scratch memory";
               break;
            default:
               break;
            }
         }
      }
   }
}

static void
agc_kernel_declare_args(const struct radeon_info *info, const struct agc_kernel_usage *usage,
                        uint32_t user_data_dwords, struct ac_shader_args *args,
                        struct ac_arg slots[AGC_KERNEL_MAX_USER_DATA])
{
   memset(args, 0, sizeof(*args));
   for (uint32_t i = 0; i < user_data_dwords; i++)
      ac_add_arg(args, AC_ARG_SGPR, 1, AC_ARG_VALUE, &slots[i]);

   for (unsigned i = 0; i < 3; i++) {
      if (!usage->workgroup_id[i])
         continue;
      if (info->gfx_level >= GFX12)
         args->workgroup_ids[i].used = true;
      else
         ac_add_arg(args, AC_ARG_SGPR, 1, AC_ARG_VALUE, &args->workgroup_ids[i]);
   }

   if (usage->tg_size)
      ac_add_arg(args, AC_ARG_SGPR, 1, AC_ARG_VALUE, &args->tg_size);

   if (info->compiler_info.local_invocation_ids_packed) {
      ac_add_arg(args, AC_ARG_VGPR, 1, AC_ARG_VALUE, &args->local_invocation_ids_packed);
   } else {
      ac_add_arg(args, AC_ARG_VGPR, 1, AC_ARG_VALUE, &args->local_invocation_id_x);
      ac_add_arg(args, AC_ARG_VGPR, 1, AC_ARG_VALUE, &args->local_invocation_id_y);
      ac_add_arg(args, AC_ARG_VGPR, 1, AC_ARG_VALUE, &args->local_invocation_id_z);
   }
}

static bool
agc_kernel_lower_user_data(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   if (intrin->intrinsic != nir_intrinsic_load_user_data_amd)
      return false;

   const struct agc_kernel_user_data *user_data = data;
   nir_def *components[AGC_KERNEL_MAX_USER_DATA];
   b->cursor = nir_before_instr(&intrin->instr);
   for (uint32_t i = 0; i < AGC_KERNEL_MAX_USER_DATA; i++) {
      components[i] = i < user_data->count
                         ? ac_nir_load_arg(b, user_data->args, user_data->slots[i])
                         : nir_undef(b, 1, 32);
   }
   nir_def_replace(&intrin->def, nir_vec(b, components, AGC_KERNEL_MAX_USER_DATA));
   return true;
}

static uint32_t
agc_kernel_user_data_read(nir_shader *nir)
{
   uint32_t mask = 0;
   nir_foreach_function_impl (impl, nir) {
      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic == nir_intrinsic_load_user_data_amd)
               mask |= nir_def_components_read(&intrin->def);
         }
      }
   }
   return mask;
}

static void
agc_kernel_lower(const struct radeon_info *info, nir_shader *nir, const struct ac_shader_args *args,
                 uint32_t threads)
{
   enum amd_gfx_level gfx_level = info->gfx_level;
   struct ac_nir_config config = {gfx_level, true};
   bool progress = false;

   NIR_PASS(_, nir, ac_nir_flag_smem_for_loads, gfx_level, false);
   NIR_PASS(_, nir, nir_lower_memory_model);

   const nir_load_store_vectorize_options vectorize = {
      .modes = nir_var_mem_global,
      .callback = ac_nir_mem_vectorize_callback,
      .cb_data = &config,
   };
   NIR_PASS(progress, nir, nir_opt_load_store_vectorize, &vectorize);
   if (progress) {
      NIR_PASS(_, nir, nir_opt_copy_prop);
      NIR_PASS(_, nir, nir_opt_shrink_stores, true);
   }

   NIR_PASS(_, nir, ac_nir_lower_mem_access_bit_sizes, gfx_level, false);

   const ac_nir_lower_tex_coords_options tex_coords = {
      .gfx_level = gfx_level,
      .lower_array_layer_round_even = !info->compiler_info.conformant_trunc_coord,
      .max_wqm_vgprs = 64,
   };
   NIR_PASS(_, nir, ac_nir_lower_tex_coords, &tex_coords);

   const ac_nir_lower_image_tex_options image_tex = {
      .gfx_level = gfx_level,
   };
   NIR_PASS(_, nir, ac_nir_lower_image_tex, &image_tex);
   if (nir->info.uses_resource_info_query)
      NIR_PASS(_, nir, ac_nir_lower_resinfo, gfx_level);

   NIR_PASS(_, nir, nir_lower_alu_width, ac_nir_opt_vectorize_cb, &gfx_level);
   NIR_PASS(_, nir, nir_opt_idiv_const, 8);

   const nir_lower_idiv_options idiv = {
      .allow_fp16 = true,
   };
   NIR_PASS(_, nir, nir_lower_idiv, &idiv);

   const ac_nir_lower_intrinsics_to_args_options to_args = {
      .gfx_level = gfx_level,
      .hw_stage = AC_HW_COMPUTE_SHADER,
      .wave_size = AGC_KERNEL_WAVE_SIZE,
      .workgroup_size = threads,
      .load_grid_size_from_user_sgpr = true,
   };
   NIR_PASS(_, nir, ac_nir_lower_intrinsics_to_args, args, &to_args);

   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_cse);
   NIR_PASS(_, nir, nir_opt_shrink_vectors, true);

   NIR_PASS(_, nir, ac_nir_flag_smem_for_loads, gfx_level, false);
   NIR_PASS(_, nir, ac_nir_lower_mem_access_bit_sizes, gfx_level, false);
   NIR_PASS(_, nir, nir_opt_load_store_vectorize, &vectorize);
   NIR_PASS(_, nir, ac_nir_lower_mem_access_bit_sizes, gfx_level, false);
   NIR_PASS(_, nir, ac_nir_lower_global_access, gfx_level);
   NIR_PASS(_, nir, nir_lower_int64);

   agc_kernel_optimize_late(nir);

   NIR_PASS(_, nir, nir_lower_fp16_casts, nir_lower_fp16_split_fp64);
   if (ac_nir_might_lower_bit_size(nir)) {
      nir_divergence_analysis(nir);
      NIR_PASS(_, nir, nir_lower_bit_size, ac_nir_lower_bit_size_callback, &gfx_level);
   }

   NIR_PASS(_, nir, nir_lower_alu_width, ac_nir_opt_vectorize_cb, &gfx_level);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);

   const nir_move_options sink = nir_move_const_undef | nir_move_copies | nir_dont_move_byte_word_vecs |
                                 nir_move_comparisons | nir_move_alu;
   NIR_PASS(_, nir, nir_opt_sink, sink);
   NIR_PASS(_, nir, nir_opt_move, sink);
   NIR_PASS(_, nir, nir_opt_move, nir_move_comparisons);
}

static void
agc_kernel_take_binary(void **data, const aco_callback_params *params)
{
   struct agc_kernel_binary *binary = (struct agc_kernel_binary *)data;
   binary->config = params->config;
   binary->exec_bytes = params->exec_size;
   binary->code_dwords = params->code_dw;

   for (unsigned i = 0; i < params->num_symbols; i++) {
      if (params->symbols[i].id != aco_symbol_const_data_addr)
         binary->patched_symbols++;
   }

   uint32_t code_bytes = params->code_dw * (uint32_t)sizeof(uint32_t);
   uint32_t padded_bytes = ac_align_shader_binary_for_prefetch(binary->gfx_level,
                                                               binary->prefetch_distance, code_bytes);
   padded_bytes = align(padded_bytes, AGC_KERNEL_CODE_ALIGNMENT);
   binary->padded_dwords = padded_bytes / (uint32_t)sizeof(uint32_t);
   binary->code = calloc(binary->padded_dwords, sizeof(uint32_t));
   if (!binary->code) {
      binary->out_of_memory = true;
      return;
   }
   memcpy(binary->code, params->code, code_bytes);

   if (params->disasm_size) {
      binary->listing = malloc(params->disasm_size + 1);
      if (binary->listing) {
         memcpy(binary->listing, params->disasm_str, params->disasm_size);
         binary->listing[params->disasm_size] = '\0';
      }
   }
}

static bool
agc_kernel_build(const struct radeon_info *info, nir_shader *nir, struct agc_kernel *kernel,
                 char *log, size_t log_bytes)
{
   const enum amd_gfx_level gfx_level = info->gfx_level;
   const uint16_t *block = nir->info.workgroup_size;
   const uint32_t threads = (uint32_t)block[0] * block[1] * block[2];
   const uint32_t user_data_dwords = nir->info.cs.user_data_components_amd;

   if (gfx_level < GFX10_3 || gfx_level > GFX12 || !info->has_graphics ||
       !aco_is_gpu_supported(info)) {
      return agc_kernel_fail(log, log_bytes, "gfx level %d is not a supported host",
                             (int)gfx_level);
   }
   if (info->family == CHIP_UNKNOWN || info->compiler_info.gfx_level != gfx_level) {
      return agc_kernel_fail(log, log_bytes,
                             "the chip description has no family or compiler information");
   }
   if (!info->num_se || !info->num_cu) {
      return agc_kernel_fail(log, log_bytes,
                             "the chip description has no shader engine or compute unit count");
   }
   if (nir->info.workgroup_size_variable || !threads || threads > AGC_KERNEL_MAX_THREADS) {
      return agc_kernel_fail(log, log_bytes, "workgroup %ux%ux%u is outside 1..%u threads",
                             (unsigned)block[0], (unsigned)block[1], (unsigned)block[2],
                             AGC_KERNEL_MAX_THREADS);
   }
   if (user_data_dwords > AGC_KERNEL_MAX_USER_DATA) {
      return agc_kernel_fail(log, log_bytes, "%u user data dwords exceed the limit of %u",
                             user_data_dwords, AGC_KERNEL_MAX_USER_DATA);
   }
   if (nir->info.shared_size) {
      return agc_kernel_fail(log, log_bytes, "kernels cannot use shared memory");
   }

   nir_shader_compiler_options *options = rzalloc(nir, nir_shader_compiler_options);
   if (!options)
      return agc_kernel_fail(log, log_bytes, "out of memory");
   ac_nir_set_options(&info->compiler_info, false, options);
   nir->options = options;
   nir->info.use_aco_amd = true;
   nir_validate_shader(nir, "at the start of agc_kernel_compile");

   const nir_lower_compute_system_values_options system_values = {
      .lower_local_invocation_index = (block[0] == 1) + (block[1] == 1) + (block[2] == 1) == 2,
   };
   NIR_PASS(_, nir, nir_lower_continue_constructs);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_compute_system_values, &system_values);
   agc_kernel_optimize(nir, gfx_level);

   uint32_t user_data_read = agc_kernel_user_data_read(nir);
   if (user_data_read & ~BITFIELD_MASK(user_data_dwords)) {
      return agc_kernel_fail(log, log_bytes,
                             "user data mask 0x%x is read but only %u dwords are declared",
                             user_data_read, user_data_dwords);
   }

   struct agc_kernel_usage usage;
   agc_kernel_gather(nir, &usage);
   if (usage.unsupported)
      return agc_kernel_fail(log, log_bytes, "kernels cannot use %s", usage.unsupported);

   struct ac_shader_args args;
   struct ac_arg slots[AGC_KERNEL_MAX_USER_DATA];
   agc_kernel_declare_args(info, &usage, user_data_dwords, &args, slots);

   struct agc_kernel_user_data user_data = {
      .args = &args,
      .slots = slots,
      .count = user_data_dwords,
   };
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, agc_kernel_lower_user_data,
            nir_metadata_control_flow, &user_data);

   agc_kernel_lower(info, nir, &args, threads);

   const struct aco_compiler_options aco_options = {
      .compiler_info = &info->compiler_info,
      .record_asm = true,
      .wgp_mode = true,
      .family = info->family,
      .gfx_level = gfx_level,
      .address32_hi = info->address32_hi,
   };
   const struct aco_shader_info aco_info = {
      .hw_stage = AC_HW_COMPUTE_SHADER,
      .wave_size = AGC_KERNEL_WAVE_SIZE,
      .workgroup_size = threads,
   };
   struct agc_kernel_binary binary = {
      .gfx_level = gfx_level,
      .prefetch_distance = info->instr_prefetch_distance,
   };
   aco_compile_shader(&aco_options, &aco_info, 1, &nir, &args, agc_kernel_take_binary,
                      (void **)&binary);

   kernel->code = binary.code;
   kernel->listing = binary.listing;
   if (binary.out_of_memory || !binary.code)
      return agc_kernel_fail(log, log_bytes, "out of memory");
   if (!binary.code_dwords || !binary.exec_bytes)
      return agc_kernel_fail(log, log_bytes, "the compiler produced no code");
   if (binary.config.scratch_bytes_per_wave) {
      return agc_kernel_fail(log, log_bytes, "the kernel needs %u bytes of scratch per wave",
                             binary.config.scratch_bytes_per_wave);
   }
   if (binary.config.lds_size) {
      return agc_kernel_fail(log, log_bytes, "the kernel needs %u bytes of shared memory",
                             binary.config.lds_size);
   }
   if (binary.patched_symbols) {
      return agc_kernel_fail(log, log_bytes, "the kernel needs %u load-time patches",
                             binary.patched_symbols);
   }

   const uint32_t wave_size = aco_info.wave_size;
   const uint32_t num_vgprs = MAX2(binary.config.num_vgprs, args.num_vgprs_used);
   const uint32_t num_sgprs = MAX2(binary.config.num_sgprs, args.num_sgprs_used + 2u + 3u);
   const uint32_t waves_per_group = DIV_ROUND_UP(threads, wave_size);
   const uint32_t tidig_comp_cnt = usage.local_id[2] ? 2 : usage.local_id[1] ? 1 : 0;

   kernel->code_dwords = binary.padded_dwords;
   kernel->exec_dwords = binary.exec_bytes / (uint32_t)sizeof(uint32_t);
   kernel->wave_size = wave_size;
   kernel->num_sgprs = num_sgprs;
   kernel->num_vgprs = num_vgprs;
   kernel->user_data_dwords = user_data_dwords;
   for (unsigned i = 0; i < 3; i++)
      kernel->block[i] = block[i];

   kernel->pgm_rsrc1 = S_00B848_VGPRS((num_vgprs - 1) / (wave_size == 32 ? 8 : 4)) |
                       S_00B848_DX10_CLAMP(gfx_level < GFX11_7) |
                       S_00B848_FLOAT_MODE(binary.config.float_mode) |
                       S_00B848_WGP_MODE(binary.config.wgp_mode);
   if (gfx_level <= GFX11_7)
      kernel->pgm_rsrc1 |= S_00B848_MEM_ORDERED(binary.config.mem_ordered);

   kernel->pgm_rsrc2 = S_00B84C_USER_SGPR(user_data_dwords) |
                       S_00B84C_TGID_X_EN(usage.workgroup_id[0]) |
                       S_00B84C_TGID_Y_EN(usage.workgroup_id[1]) |
                       S_00B84C_TGID_Z_EN(usage.workgroup_id[2]) |
                       S_00B84C_TG_SIZE_EN(usage.tg_size) |
                       S_00B84C_TIDIG_COMP_CNT(tidig_comp_cnt);

   kernel->pgm_rsrc3 = S_00B8A0_SHARED_VGPR_CNT(binary.config.num_shared_vgprs / 8);
   if (gfx_level >= GFX11) {
      uint32_t prefetch = ac_get_instr_prefetch_size(gfx_level, info->instr_prefetch_distance,
                                                     binary.exec_bytes);
      kernel->pgm_rsrc3 |= gfx_level >= GFX12 ? S_00B8A0_INST_PREF_SIZE_GFX12(prefetch)
                                              : S_00B8A0_INST_PREF_SIZE_GFX11(prefetch);
   }

   kernel->resource_limits =
      ac_get_compute_resource_limits(info, waves_per_group, 0, waves_per_group == 1 ? 2 : 1);
   return true;
}

bool
agc_kernel_compile(const struct radeon_info *info, struct nir_shader *nir,
                   struct agc_kernel *kernel, char *log, size_t log_bytes)
{
   if (log && log_bytes)
      log[0] = '\0';
   if (!kernel) {
      ralloc_free(nir);
      return agc_kernel_fail(log, log_bytes, "no kernel to fill");
   }
   memset(kernel, 0, sizeof(*kernel));
   if (!info || !nir) {
      ralloc_free(nir);
      return agc_kernel_fail(log, log_bytes, "no chip description or no shader");
   }
   if (nir->info.stage != MESA_SHADER_COMPUTE) {
      ralloc_free(nir);
      return agc_kernel_fail(log, log_bytes, "a kernel is a compute shader");
   }

   bool ok = agc_kernel_build(info, nir, kernel, log, log_bytes);
   ralloc_free(nir);
   if (!ok)
      agc_kernel_free(kernel);
   return ok;
}

void
agc_kernel_free(struct agc_kernel *kernel)
{
   if (!kernel)
      return;
   free(kernel->code);
   free(kernel->listing);
   memset(kernel, 0, sizeof(*kernel));
}

uint32_t
agc_kernel_dispatch_dwords(const struct radeon_info *info, const struct agc_kernel *kernel)
{
   if (!info || !kernel || !kernel->code)
      return 0;
   return AGC_KERNEL_DISPATCH_FIXED_DWORDS +
          (kernel->user_data_dwords ? 2 + kernel->user_data_dwords : 0);
}

uint32_t *
agc_kernel_emit_dispatch(uint32_t *out, const struct radeon_info *info,
                         const struct agc_kernel *kernel, uint64_t code_va,
                         const uint32_t *user_data, uint32_t groups_x, uint32_t groups_y,
                         uint32_t groups_z)
{
   if (!out || !info || !kernel || !kernel->code) {
      agc_report(__func__, "no output, chip description or compiled kernel");
      return NULL;
   }
   if (!code_va || code_va % AGC_KERNEL_CODE_ALIGNMENT || code_va >= AGC_KERNEL_CODE_ADDRESS_END) {
      agc_report(__func__, "code address 0x%llx is not a %u-byte aligned 48-bit address",
                 (unsigned long long)code_va, AGC_KERNEL_CODE_ALIGNMENT);
      return NULL;
   }
   if (kernel->user_data_dwords && !user_data) {
      agc_report(__func__, "the kernel reads %u user data dwords and none were given",
                 kernel->user_data_dwords);
      return NULL;
   }
   if (!groups_x || !groups_y || !groups_z || groups_y > AGC_KERNEL_MAX_GROUPS_YZ ||
       groups_z > AGC_KERNEL_MAX_GROUPS_YZ) {
      agc_report(__func__, "dispatch of %ux%ux%u groups is empty or too large", groups_x, groups_y,
                 groups_z);
      return NULL;
   }

   uint32_t initiator = S_00B800_COMPUTE_SHADER_EN(1) | S_00B800_FORCE_START_AT_000(1) |
                        S_00B800_ORDER_MODE(1) | S_00B800_TUNNEL_ENABLE(1) |
                        S_00B800_CS_W32_EN(kernel->wave_size == 32);
   struct ac_cmdbuf cs = {
      .buf = out,
      .max_dw = agc_kernel_dispatch_dwords(info, kernel),
   };

   ac_cmdbuf_begin(&cs);
   ac_cmdbuf_set_sh_reg_seq(R_00B830_COMPUTE_PGM_LO, 2);
   ac_cmdbuf_emit((uint32_t)(code_va >> 8));
   ac_cmdbuf_emit(S_00B834_DATA(code_va >> 40));
   ac_cmdbuf_set_sh_reg_seq(R_00B848_COMPUTE_PGM_RSRC1, 2);
   ac_cmdbuf_emit(kernel->pgm_rsrc1);
   ac_cmdbuf_emit(kernel->pgm_rsrc2);
   ac_cmdbuf_set_sh_reg(R_00B8A0_COMPUTE_PGM_RSRC3, kernel->pgm_rsrc3);
   ac_cmdbuf_set_sh_reg(R_00B854_COMPUTE_RESOURCE_LIMITS, kernel->resource_limits);
   ac_cmdbuf_set_sh_reg_seq(R_00B81C_COMPUTE_NUM_THREAD_X, 3);
   if (info->gfx_level >= GFX12) {
      ac_cmdbuf_emit(S_00B81C_NUM_THREAD_FULL_GFX12(kernel->block[0]));
      ac_cmdbuf_emit(S_00B820_NUM_THREAD_FULL_GFX12(kernel->block[1]));
   } else {
      ac_cmdbuf_emit(S_00B81C_NUM_THREAD_FULL_GFX6(kernel->block[0]));
      ac_cmdbuf_emit(S_00B820_NUM_THREAD_FULL_GFX6(kernel->block[1]));
   }
   ac_cmdbuf_emit(S_00B824_NUM_THREAD_FULL(kernel->block[2]));
   if (kernel->user_data_dwords) {
      ac_cmdbuf_set_sh_reg_seq(R_00B900_COMPUTE_USER_DATA_0, kernel->user_data_dwords);
      ac_cmdbuf_emit_array(user_data, kernel->user_data_dwords);
   }
   ac_cmdbuf_emit(PKT3(PKT3_DISPATCH_DIRECT, 3, 0) | PKT3_SHADER_TYPE_S(1));
   ac_cmdbuf_emit(groups_x);
   ac_cmdbuf_emit(groups_y);
   ac_cmdbuf_emit(groups_z);
   ac_cmdbuf_emit(initiator);
   ac_cmdbuf_end();
   return out + cs.cdw;
}

static uint32_t
agc_kernel_queue_setup(const struct radeon_info *info, bool compute_ring, uint32_t *out)
{
   if (!info || info->gfx_level < GFX10_3 || info->gfx_level > GFX12 || !info->has_graphics ||
       !info->max_se || !info->spi_cu_en)
      return 0;

   struct ac_pm4_state *pm4 =
      ac_pm4_create_sized(info, false, AGC_KERNEL_MAX_QUEUE_SETUP_DWORDS, compute_ring);
   if (!pm4)
      return 0;

   const struct ac_preamble_state preamble = {
      .gfx11 = {
         .compute_dispatch_interleave = AGC_KERNEL_DISPATCH_INTERLEAVE,
      },
   };
   ac_init_compute_preamble_state(&preamble, pm4);
   ac_pm4_set_reg(pm4, R_00B810_COMPUTE_START_X, 0);
   ac_pm4_set_reg(pm4, R_00B814_COMPUTE_START_Y, 0);
   ac_pm4_set_reg(pm4, R_00B818_COMPUTE_START_Z, 0);
   if (info->gfx_level >= GFX12) {
      uint32_t interleave = S_00B8BC_INTERLEAVE_1D(AGC_KERNEL_DISPATCH_INTERLEAVE);
      if (compute_ring) {
         ac_pm4_set_reg(pm4, R_00B8BC_COMPUTE_DISPATCH_INTERLEAVE, interleave);
      } else {
         ac_pm4_set_reg_custom(pm4, R_00B8BC_COMPUTE_DISPATCH_INTERLEAVE - SI_SH_REG_OFFSET,
                               interleave, PKT3_SET_SH_REG_INDEX, 2);
      }
   }
   ac_pm4_finalize(pm4);

   uint32_t dwords = pm4->ndw;
   if (out)
      memcpy(out, pm4->pm4, dwords * sizeof(uint32_t));
   ac_pm4_free_state(pm4);
   return dwords;
}

uint32_t
agc_kernel_queue_setup_dwords(const struct radeon_info *info, bool compute_ring)
{
   return agc_kernel_queue_setup(info, compute_ring, NULL);
}

uint32_t *
agc_kernel_emit_queue_setup(uint32_t *out, const struct radeon_info *info, bool compute_ring)
{
   uint32_t dwords = out ? agc_kernel_queue_setup(info, compute_ring, out) : 0;
   if (!dwords) {
      agc_report(__func__, "no output or the chip description has no shader engines");
      return NULL;
   }
   return out + dwords;
}
