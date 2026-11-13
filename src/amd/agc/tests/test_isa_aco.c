#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agc_isa.h"
#include "agc_isa_translate.h"
#include "test.h"

#define S_ENDPGM 0xbf810000u
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))
#define LEGACY_MADS 12000

struct host_ops {
   int endpgm, waitcnt, delay_alu, branch, cbranch_scc1;
   bool gfx12;
};

static struct host_ops
host_ops_for(enum agc_isa_target target)
{
   struct host_ops o;
   memset(&o, 0, sizeof(o));
   if (agc_isa_target_is_gfx11_plus(target)) {
      o.endpgm = 0x30;
      o.waitcnt = 0x09;
      o.delay_alu = 0x07;
      o.branch = 0x20;
      o.cbranch_scc1 = 0x22;
      o.gfx12 = agc_isa_target_is_gfx12(target);
   } else {
      o.endpgm = 0x01;
      o.waitcnt = 0x0c;
      o.delay_alu = -1;
      o.branch = 0x02;
      o.cbranch_scc1 = 0x05;
   }
   return o;
}

static int
sopp_op(uint32_t w)
{
   if (((w >> 23) & 0x1ff) != 0x17f)
      return -1;
   return (int)((w >> 16) & 0x7f);
}

static unsigned
count_sopp(const struct agc_shader_artifact *artifact, int op)
{
   unsigned n = 0;
   for (uint32_t i = 0; i < artifact->program_words; i++)
      if (sopp_op(artifact->code[i]) == op)
         n++;
   return n;
}

static bool
has_any_wait(const struct agc_shader_artifact *artifact, struct host_ops o)
{
   if (o.gfx12) {
      static const int waits[] = {0x40, 0x46, 0x47, 0x48, 0x41, 0x42, 0x43, 0x44, 0x49};
      for (unsigned i = 0; i < ARRAY_SIZE(waits); i++)
         if (count_sopp(artifact, waits[i]))
            return true;
      return false;
   }
   return count_sopp(artifact, o.waitcnt) > 0;
}

static void
request_init(struct agc_shader_request *request, enum agc_isa_target target, const uint32_t *words, size_t count)
{
   memset(request, 0, sizeof(*request));
   request->options.target = (enum agc_shader_isa)target;
   request->options.wave_size = 32;
   request->options.stage = AGC_SHADER_STAGE_FRAGMENT;
   request->words = words;
   request->word_count = (uint32_t)count;
}

static bool
translate(const struct agc_shader_request *request, struct agc_shader_artifact *artifact, const char *name)
{
   struct agc_isa_error error;
   bool ok = agc_isa_translate(request, artifact, &error);
   if (!ok) {
      char text[AGC_ISA_ERROR_TEXT_SIZE];
      agc_isa_error_format(&error, text, sizeof(text));
      fprintf(stderr, "%s on %s: %s\n", name, agc_isa_target_name((enum agc_isa_target)request->options.target), text);
   }
   CHECK(ok);
   return ok;
}

static size_t
legacy_mads(uint32_t *words, unsigned count)
{
   for (unsigned i = 0; i < count; i++) {
      words[2 * i] = 0xd5400000;
      words[2 * i + 1] = 0x040e0501;
   }
   return (size_t)2 * count;
}

static void
test_straight_line(enum agc_isa_target target)
{
   static const uint32_t guest[] = {0xbe800381, 0x80010000, 0x7e000282, 0x06020100, S_ENDPGM};
   struct agc_shader_request request;
   request_init(&request, target, guest, ARRAY_SIZE(guest));
   struct agc_shader_artifact artifact;
   if (translate(&request, &artifact, "straight_line")) {
      struct host_ops o = host_ops_for(target);
      CHECK(artifact.program_words > 0);
      CHECK(count_sopp(&artifact, o.endpgm) >= 1);
      if (o.delay_alu >= 0)
         CHECK(count_sopp(&artifact, o.delay_alu) >= 1);
      CHECK(artifact.code_count > artifact.program_words);
      CHECK_EQ(artifact.code[artifact.code_count - 1], 0xbf9f0000u);
      agc_shader_artifact_free(&artifact);
   }
}

static void
test_loop(enum agc_isa_target target)
{
   static const uint32_t guest[] = {0xbe800380, 0xe0301000, 0x80010100, 0x7e040301, 0xbf068000, 0xbf85fffb, S_ENDPGM};
   struct agc_shader_request request;
   request_init(&request, target, guest, ARRAY_SIZE(guest));
   struct agc_shader_artifact artifact;
   if (translate(&request, &artifact, "loop")) {
      struct host_ops o = host_ops_for(target);
      CHECK(has_any_wait(&artifact, o));
      CHECK(count_sopp(&artifact, o.cbranch_scc1) >= 1);
      agc_shader_artifact_free(&artifact);
   }
}

static void
test_far_branch(enum agc_isa_target target)
{
   static uint32_t guest[2 * LEGACY_MADS + 3];
   guest[0] = 0xbf068000;
   guest[1] = 0xbf850000u | (2 * LEGACY_MADS);
   size_t count = 2 + legacy_mads(guest + 2, LEGACY_MADS);
   guest[count++] = S_ENDPGM;
   struct agc_shader_request request;
   request_init(&request, target, guest, count);
   struct agc_shader_artifact artifact;
   if (translate(&request, &artifact, "far_branch")) {
      struct host_ops o = host_ops_for(target);
      CHECK(artifact.program_words > 0x8000);
      CHECK(count_sopp(&artifact, o.branch) >= 1);
      agc_shader_artifact_free(&artifact);
   }
}

static void
test_stages(enum agc_isa_target target)
{
   static const uint32_t compute[] = {0xbe840302, 0x7e080302, S_ENDPGM};
   static const uint32_t pixel[] = {0xbefc0302, 0xc8080500, 0xc8090501, 0xf800080f, 0x03020100, S_ENDPGM};
   static const uint32_t vertex[] = {0x7e120302, 0x7e140305, 0x7e160308, 0xf800020f,
                                     0x03020100, 0xf80008cf, 0x07060504, S_ENDPGM};
   static const uint32_t hull[] = {0x7e120303, 0x7e140305, S_ENDPGM};
   for (unsigned wave = 32; wave <= 64; wave += 32) {
      for (unsigned s = 0; s < 5; s++) {
         struct agc_shader_request request;
         request_init(&request, target, s == 0 ? compute : s == 1 ? pixel : s == 4 ? hull : vertex,
                      s == 0 ? 3 : s == 1 ? 6 : s == 4 ? 3 : 8);
         request.options.wave_size = (uint8_t)wave;
         request.options.stage = s == 0 ? AGC_SHADER_STAGE_COMPUTE :
                                 s == 1 ? AGC_SHADER_STAGE_FRAGMENT : AGC_SHADER_STAGE_VERTEX;
         request.iface.vertex_stage = s == 4 ? AGC_SHADER_VERTEX_STAGE_LS_HS : AGC_SHADER_VERTEX_STAGE_ES_GS;
         request.iface.es_input = s == 3 ? AGC_SHADER_ES_DOMAIN : AGC_SHADER_ES_VERTEX;
         request.iface.has_user_sgpr_count = true;
         request.iface.user_sgpr_count = 16;
         request.iface.has_guest_pgm_rsrc1 = true;
         request.iface.guest_pgm_rsrc1 = (3u << 28) | (1u << 24) | (1u << 25);
         request.iface.has_guest_pgm_rsrc2 = true;
         request.iface.guest_pgm_rsrc2 = (16u << 1) | (3u << 16);
         request.iface.has_attribute_ring = true;
         request.iface.attribute_ring.address = UINT64_C(0x20000100000);
         request.iface.attribute_ring.bytes = 0x2c0000;
         struct agc_shader_artifact artifact;
         if (!translate(&request, &artifact, "stages"))
            continue;
         CHECK(artifact.program_words > 0);
         if (target == AGC_ISA_TARGET_GFX12 && s == 4) {
            CHECK_EQ((artifact.pgm_rsrc1 >> 28) & 3, 1);
            CHECK_EQ((artifact.pgm_rsrc1 >> 24) & 1, 0);
         }
         if (target == AGC_ISA_TARGET_GFX12 && s == 2)
            CHECK_EQ((artifact.pgm_rsrc2 >> 16) & 3, 1);
         if (target == AGC_ISA_TARGET_GFX12 && s == 3)
            CHECK_EQ((artifact.pgm_rsrc2 >> 16) & 3, 3);
         if (agc_isa_target_is_gfx11_plus(target) && (s == 2 || s == 3)) {
            CHECK(artifact.has_ngg);
            CHECK(artifact.ngg.has_attribute_ring_relocations);
         }
         agc_shader_artifact_free(&artifact);
      }
   }
}

static void
test_modifiers(enum agc_isa_target target)
{
   static const uint32_t guest[] = {0xd5038004, 0x600e0702, S_ENDPGM};
   struct agc_shader_request request;
   request_init(&request, target, guest, ARRAY_SIZE(guest));
   struct agc_shader_artifact artifact;
   if (!translate(&request, &artifact, "modifiers"))
      return;
   bool found = false;
   for (uint32_t w = 0; w + 1 < artifact.program_words; w++) {
      if ((artifact.code[w] & 0xfc0000ff) == 0xd4000004 || (artifact.code[w] & 0xfc0000ff) == 0xd0000004) {
         CHECK(artifact.code[w] & 0x8000);
         CHECK_EQ(artifact.code[w + 1] >> 29, 3);
         CHECK_EQ((artifact.code[w + 1] >> 27) & 3, 0);
         found = true;
         break;
      }
   }
   CHECK(found);
   agc_shader_artifact_free(&artifact);
}

static void
check_dispatch(const struct agc_shader_artifact *artifact, enum agc_isa_target target, uint32_t guest_words)
{
   struct host_ops o = host_ops_for(target);
   struct agc_isa_xlat x = agc_isa_op_translate(agc_isa_op_lookup(AGC_ISA_SOP1, AGC_ISA_SOP1_GETPC_B64), target);
   uint32_t getpc = x.kind == AGC_ISA_XLAT_OP ? x.value : AGC_ISA_SOP1_GETPC_B64;
   uint32_t anchor = 0, load = 0;
   for (uint32_t w = 0; w < artifact->program_words; w++) {
      if ((artifact->code[w] >> 23) == 0x17d && ((artifact->code[w] >> 8) & 0xff) == getpc)
         anchor = w + 1;
      if (anchor && artifact->code[w] >> 26 == 0x3d) {
         load = w + 1;
         break;
      }
   }
   CHECK(anchor && load);
   if (!anchor || !load)
      return;
   uint32_t mask = target == AGC_ISA_TARGET_GFX12 ? 0x7fffff : 0xfffff;
   CHECK_EQ(artifact->code[load] & mask, (artifact->program_words - anchor) * 4);
   CHECK(artifact->code_count >= artifact->program_words + guest_words + 1);
   uint32_t miss = 0;
   for (uint32_t i = 0; i <= guest_words; i++) {
      int64_t offset = (int64_t)(int32_t)artifact->code[artifact->program_words + i] + (int64_t)anchor * 4;
      CHECK(offset >= 0 && offset < (int64_t)artifact->program_words * 4 && offset % 4 == 0);
      if (i == guest_words)
         miss = (uint32_t)offset;
   }
   CHECK_EQ(sopp_op(artifact->code[miss / 4]), o.endpgm);
}

static void
test_dispatch(enum agc_isa_target target)
{
   static uint32_t guest[2 * LEGACY_MADS + 4];
   for (unsigned far = 0; far < 2; far++) {
      size_t count = 0;
      guest[count++] = 0xbf850001;
      guest[count++] = 0xbe802002;
      if (far)
         count += legacy_mads(guest + count, LEGACY_MADS);
      guest[count++] = 0x7e000280;
      guest[count++] = S_ENDPGM;
      struct agc_shader_request request;
      request_init(&request, target, guest, count);
      request.iface.has_user_sgpr_count = true;
      request.iface.user_sgpr_count = 4;
      request.iface.has_guest_code_address = true;
      request.iface.guest_code_address = UINT64_C(0x100000000);
      struct agc_shader_artifact artifact;
      if (!translate(&request, &artifact, far ? "far dispatch" : "dispatch"))
         continue;
      check_dispatch(&artifact, target, (uint32_t)count);
      if (far)
         CHECK(artifact.program_words > 40000);
      agc_shader_artifact_free(&artifact);
   }
}

static void
test_scalar_load_address(enum agc_isa_target target)
{
   static const uint32_t offsets[2] = {0xfa000010, 0xd6000010};
   for (unsigned i = 0; i < 2; i++) {
      const uint32_t guest[] = {0xf4000101, offsets[i], S_ENDPGM};
      struct agc_shader_request request;
      request_init(&request, target, guest, ARRAY_SIZE(guest));
      struct agc_shader_artifact artifact;
      if (!translate(&request, &artifact, "scalar_load_address"))
         continue;
      bool found = false;
      for (uint32_t w = 0; w + 1 < artifact.program_words; w++) {
         if (artifact.code[w] >> 26 == 0x3d) {
            CHECK_EQ(artifact.code[w] & 0x3f, 1);
            CHECK_EQ(artifact.code[w + 1] & 0xfffff, 16);
            if (i)
               CHECK_EQ(artifact.code[w + 1] >> 25, 107);
            found = true;
         }
      }
      CHECK(found);
      agc_shader_artifact_free(&artifact);
   }
}

static void
test_global_load_based_on_vcc(enum agc_isa_target target)
{
   static const uint32_t guest[] = {0xdc308024, 0x016a0002, S_ENDPGM};
   struct agc_shader_request request;
   request_init(&request, target, guest, ARRAY_SIZE(guest));
   struct agc_shader_artifact artifact;
   if (!translate(&request, &artifact, "global_load_based_on_vcc"))
      return;
   bool found = false;
   for (uint32_t w = 0; w + 2 < artifact.program_words; w++) {
      if (agc_isa_target_is_gfx12(target) && artifact.code[w] >> 24 == 0xee) {
         CHECK_EQ(artifact.code[w] & 0x7f, 106);
         CHECK_EQ(artifact.code[w + 1] & 0xff, 1);
         CHECK_EQ(artifact.code[w + 2] & 0xff, 2);
         found = true;
      } else if (!agc_isa_target_is_gfx12(target) && artifact.code[w] >> 26 == 0x37) {
         CHECK_EQ((artifact.code[w + 1] >> 16) & 0x7f, 106);
         CHECK_EQ(artifact.code[w + 1] & 0xff, 2);
         CHECK_EQ(artifact.code[w + 1] >> 24, 1);
         found = true;
      }
   }
   CHECK(found);
   agc_shader_artifact_free(&artifact);
}

static void
test_inline_constant_beside_an_instruction_constant(enum agc_isa_target target)
{
   static const uint32_t addend = 0x3f743d61;
   static const uint32_t guest[] = {0x5a221af8, addend, S_ENDPGM};
   struct agc_shader_request request;
   request_init(&request, target, guest, ARRAY_SIZE(guest));
   struct agc_shader_artifact artifact;
   if (!translate(&request, &artifact, "inline_constant_beside_an_instruction_constant"))
      return;
   bool found = false;
   for (uint32_t w = 0; w + 1 < artifact.program_words; w++) {
      if (artifact.code[w] >> 31 == 0 && (artifact.code[w] & 0x1ff) == 0xf8 && artifact.code[w + 1] == addend)
         found = true;
   }
   CHECK(found);
   agc_shader_artifact_free(&artifact);
}

int
main(void)
{
   static const enum agc_isa_target targets[] = {
      AGC_ISA_TARGET_GFX103,
      AGC_ISA_TARGET_GFX11,
      AGC_ISA_TARGET_GFX115,
      AGC_ISA_TARGET_GFX12,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(targets); i++) {
      test_straight_line(targets[i]);
      test_loop(targets[i]);
      test_far_branch(targets[i]);
      test_stages(targets[i]);
      test_modifiers(targets[i]);
      test_dispatch(targets[i]);
      test_scalar_load_address(targets[i]);
      test_global_load_based_on_vcc(targets[i]);
      test_inline_constant_beside_an_instruction_constant(targets[i]);
   }
   printf("%d failures\n", agc_test_failures);
   return TEST_RESULT();
}
