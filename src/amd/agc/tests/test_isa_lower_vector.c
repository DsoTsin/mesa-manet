#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agc_isa.h"
#include "agc_lower.h"
#include "agc_isa_regalloc.h"
#include "test_isa_listing.h"

#include "test_isa_lower_vector_listings.h"

#define S_ENDPGM 0xbf810000u
#define V_MAC_F32 0x3e000501u
#define WORDS(...) ((const uint32_t[]){__VA_ARGS__}), (sizeof((const uint32_t[]){__VA_ARGS__}) / sizeof(uint32_t))
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

static const uint32_t v_mad_f32[2] = {0xd5410000u, 0x040e0501u};
static const uint32_t v_add_f32_sdwa[2] = {0x060004f9u, 0x02041501u};

static struct listing_set listings;

struct lowered {
   struct agc_isa_error error;
   char *listing;
   bool ok;
};

static void
lower_full(struct lowered *out, const uint32_t *words, size_t count, enum agc_isa_target target,
           enum agc_shader_stage stage, uint8_t wave_size, enum agc_shader_precision precision,
           const struct agc_shader_interface *iface)
{
   static const struct agc_shader_interface none;
   if (!iface)
      iface = &none;
   memset(out, 0, sizeof(*out));
   struct agc_shader_options options = {(enum agc_shader_isa)target, stage, wave_size, AGC_SHADER_RAY_UNIT_HOST};
   struct agc_isa_ir ir;
   agc_isa_ir_init(&ir);
   out->ok = agc_isa_analyze(target, words, count, &options, iface, &ir, &out->error);
   if (out->ok) {
      struct agc_isa_regalloc regs;
      agc_isa_regalloc_init(&regs, &ir.metadata, &options, iface);
      struct agc_isa_lowerer lowerer;
      agc_isa_lowerer_init(&lowerer, &options, precision);
      out->ok = agc_isa_lower_listing(&lowerer, &ir, &regs, &out->listing, &out->error);
   }
   agc_isa_ir_finish(&ir);
   if (!out->ok)
      out->listing = listing_of_error(&out->error);
}

static void
lower_with(struct lowered *out, const uint32_t *words, size_t count, enum agc_isa_target target,
           enum agc_shader_precision precision, uint8_t wave_size)
{
   lower_full(out, words, count, target, AGC_SHADER_STAGE_FRAGMENT, wave_size, precision, NULL);
}

static void
lower(struct lowered *out, const uint32_t *words, size_t count, enum agc_isa_target target)
{
   lower_with(out, words, count, target, AGC_SHADER_PRECISION_FUSED, 32);
   CHECK(out->ok);
}

static void
lower_compute(struct lowered *out, const uint32_t *words, size_t count, enum agc_isa_target target,
              const struct agc_shader_interface *iface)
{
   lower_full(out, words, count, target, AGC_SHADER_STAGE_COMPUTE, 32, AGC_SHADER_PRECISION_EXACT, iface);
   CHECK(out->ok);
}

static void
finish(struct lowered *out)
{
   free(out->listing);
   memset(out, 0, sizeof(*out));
}

static void
expect(const struct lowered *out, const char *name)
{
   CHECK(out->listing && listing_matches(&listings, name, out->listing));
}

static void
expect_on(const struct lowered *out, const char *name, enum agc_isa_target target)
{
   char full[128];
   snprintf(full, sizeof(full), "%s/%s", name, agc_isa_target_name(target));
   expect(out, full);
}

static bool
has(const struct lowered *out, const char *line)
{
   return out->listing && listing_has_line(out->listing, line);
}

static unsigned
lines(const struct lowered *out, const char *needle)
{
   return out->listing ? listing_lines(out->listing, needle, false) : 0;
}

static void
mac_becomes_fmac_on_every_newer_target(void)
{
   static const enum agc_isa_target targets[] = {AGC_ISA_TARGET_GFX103, AGC_ISA_TARGET_GFX11, AGC_ISA_TARGET_GFX12};
   for (unsigned i = 0; i < ARRAY_SIZE(targets); i++) {
      struct lowered l;
      lower(&l, WORDS(V_MAC_F32, S_ENDPGM), targets[i]);
      expect_on(&l, "mac/fused", targets[i]);
      CHECK(has(&l, "v0 = v_fmac_f32 v1, v2, v0"));
      finish(&l);
   }
}

static void
mad_becomes_fma_with_modifiers_preserved(void)
{
   static const enum agc_isa_target targets[] = {AGC_ISA_TARGET_GFX103, AGC_ISA_TARGET_GFX11, AGC_ISA_TARGET_GFX12};
   for (unsigned i = 0; i < ARRAY_SIZE(targets); i++) {
      struct lowered l;
      lower(&l, WORDS(0xd5418200, 0x240e0501, S_ENDPGM), targets[i]);
      expect_on(&l, "mad/fused_modifiers", targets[i]);
      CHECK_EQ(lines(&l, "v_fma_f32"), 1);
      finish(&l);
   }
}

static void
legacy_mad_is_unfused_on_every_target(void)
{
   static const enum agc_isa_target targets[] = {AGC_ISA_TARGET_GFX103, AGC_ISA_TARGET_GFX11, AGC_ISA_TARGET_GFX115,
                                                 AGC_ISA_TARGET_GFX12};
   for (unsigned i = 0; i < ARRAY_SIZE(targets); i++) {
      struct lowered l;
      lower(&l, WORDS(0xd5400200, 0x840e0501, S_ENDPGM), targets[i]);
      expect_on(&l, "mad_legacy/unfused", targets[i]);
      CHECK_EQ(lines(&l, "v_mul_legacy_f32"), 1);
      CHECK_EQ(lines(&l, "v_add_f32"), 1);
      finish(&l);
      lower(&l, WORDS(0x0c000501, S_ENDPGM), targets[i]);
      expect_on(&l, "mac_legacy/unfused", targets[i]);
      CHECK_EQ(lines(&l, "v_mul_legacy_f32"), 1);
      CHECK_EQ(lines(&l, "v_add_f32"), 1);
      finish(&l);
   }
}

static void
legacy_mad_flushes_f32_denormals_when_the_mode_keeps_them(void)
{
   struct agc_shader_interface iface;
   memset(&iface, 0, sizeof(iface));
   iface.has_guest_pgm_rsrc1 = true;
   iface.guest_pgm_rsrc1 = 0xf0u << 12;
   struct lowered l;
   lower_full(&l, WORDS(0xd5400000, 0x040e0501, S_ENDPGM), AGC_ISA_TARGET_GFX11, AGC_SHADER_STAGE_FRAGMENT, 32,
              AGC_SHADER_PRECISION_FUSED, &iface);
   CHECK(l.ok);
   expect(&l, "mad_legacy/flushes_denormals");
   CHECK_EQ(lines(&l, "s_denorm_mode"), 2);
   finish(&l);
}

static void
exact_precision_splits_mad_into_mul_and_add(void)
{
   struct lowered l;
   lower_with(&l, WORDS(v_mad_f32[0], v_mad_f32[1], S_ENDPGM), AGC_ISA_TARGET_GFX11, AGC_SHADER_PRECISION_EXACT, 32);
   CHECK(l.ok);
   expect(&l, "mad/exact");
   CHECK(has(&l, "v4 = v_mul_f32 v1, v2 vop3"));
   CHECK(has(&l, "v0 = v_add_f32 v4, v3 vop3"));
   finish(&l);
}

static void
sdwa_expands_to_bitfield_sequences_on_gfx11(void)
{
   struct lowered l;
   lower(&l, WORDS(v_add_f32_sdwa[0], v_add_f32_sdwa[1], S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "sdwa/expanded_gfx11");
   CHECK_EQ(lines(&l, "v_bfe_u32"), 2);
   CHECK_EQ(lines(&l, "v_bfi_b32"), 1);
   finish(&l);
}

static void
sdwa_is_kept_on_rdna2(void)
{
   struct lowered l;
   lower(&l, WORDS(v_add_f32_sdwa[0], v_add_f32_sdwa[1], S_ENDPGM), AGC_ISA_TARGET_GFX103);
   expect(&l, "sdwa/kept_gfx103");
   CHECK_EQ(lines(&l, "v_add_f32"), 1);
   CHECK_EQ(lines(&l, "v_bfe_u32"), 0);
   finish(&l);
}

static void
an_sdwa_move_changes_the_sign_with_a_bit_operation(void)
{
   static const struct {
      uint32_t modifiers;
      const char *name;
      const char *line;
   } cases[] = {
      {1, "sdwa/move_neg", "v29 = v_xor_b32 0x80000000, v62"},
      {2, "sdwa/move_abs", "v29 = v_and_b32 0x7fffffff, v62"},
      {3, "sdwa/move_neg_abs", "v29 = v_or_b32 0x80000000, v62"},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(cases); i++) {
      struct lowered l;
      lower(&l, WORDS(0x7e3a02f9, 0x0006063eu | cases[i].modifiers << 20, S_ENDPGM), AGC_ISA_TARGET_GFX115);
      expect(&l, cases[i].name);
      CHECK(has(&l, cases[i].line));
      finish(&l);
   }
   struct lowered l;
   lower(&l, WORDS(0x100400f9, 0x06160601, S_ENDPGM), AGC_ISA_TARGET_GFX115);
   expect(&l, "sdwa/float_modifiers_stay_on_the_instruction");
   CHECK_EQ(lines(&l, "v_mul_f32"), 1);
   CHECK_EQ(lines(&l, "v_xor_b32"), 0);
   finish(&l);
}

static void
waitcnt_is_repacked_for_gfx11(void)
{
   struct lowered l;
   lower(&l, WORDS(0xbf8c0070, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "waitcnt/gfx11");
   CHECK_EQ(lines(&l, "s_waitcnt"), 1);
   finish(&l);
}

static void
waitcnt_splits_into_gfx12_counters(void)
{
   struct lowered l;
   lower(&l, WORDS(0xf4000100, 0xfa000000, 0xe0301000, 0x80020100, 0xbf8c0070, S_ENDPGM), AGC_ISA_TARGET_GFX12);
   expect(&l, "waitcnt/gfx12_counters");
   CHECK_EQ(lines(&l, "s_wait_"), 2);
   CHECK_EQ(lines(&l, "s_wait_loadcnt"), 1);
   CHECK_EQ(lines(&l, "s_wait_kmcnt"), 1);
   finish(&l);
}

static void
compares_are_renumbered_and_keep_exec_semantics(void)
{
   struct lowered l;
   lower(&l, WORDS(0x7c240280, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "compare/cmpx_gfx11");
   finish(&l);
   lower(&l, WORDS(0xd412007e, 0x00020280, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "compare/cmpx_vop3_gfx11");
   finish(&l);
   lower(&l, WORDS(0x7d840280, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "compare/eq_u32_gfx11");
   finish(&l);
   lower(&l, WORDS(0x7d840280, S_ENDPGM), AGC_ISA_TARGET_GFX103);
   expect(&l, "compare/eq_u32_gfx103");
   finish(&l);
}

static void
gfx12_replaces_constant_compares(void)
{
   struct lowered l;
   lower(&l, WORDS(0x7c000300, S_ENDPGM), AGC_ISA_TARGET_GFX12);
   expect(&l, "compare/constant_false_gfx12");
   CHECK(has(&l, "vcc_lo = s_mov_b32 0"));
   finish(&l);
   lower_with(&l, WORDS(0x7d0e0300, S_ENDPGM), AGC_ISA_TARGET_GFX12, AGC_SHADER_PRECISION_FUSED, 64);
   CHECK(l.ok);
   expect(&l, "compare/constant_true_gfx12");
   CHECK(has(&l, "vcc = s_mov_b64 exec"));
   finish(&l);
   lower(&l, WORDS(0x7d2e0300, S_ENDPGM), AGC_ISA_TARGET_GFX12);
   expect(&l, "compare/constant_true_cmpx_gfx12");
   CHECK(has(&l, "s_endpgm"));
   CHECK_EQ(lines(&l, "s_mov"), 0);
   finish(&l);
}

static void
removed_features_are_rejected_explicitly(void)
{
   struct lowered l;
   lower_with(&l, WORDS(0xf8000a0f, 0x03020100, S_ENDPGM), AGC_ISA_TARGET_GFX11, AGC_SHADER_PRECISION_FUSED, 32);
   CHECK(!l.ok);
   CHECK(agc_isa_error_is_unsupported(&l.error));
   CHECK(l.listing && strstr(l.listing, "attribute ring") != NULL);
   expect(&l, "export/parameter_without_ring_gfx11");
   finish(&l);

   static const enum agc_isa_target targets[] = {AGC_ISA_TARGET_GFX103, AGC_ISA_TARGET_GFX11, AGC_ISA_TARGET_GFX12};
   for (unsigned t = 0; t < ARRAY_SIZE(targets); t++) {
      lower(&l, WORDS(0xda000102, 0x00000000, S_ENDPGM), targets[t]);
      expect_on(&l, "lds/add_src2", targets[t]);
      CHECK_EQ(lines(&l, "ds_read_b32"), 1);
      CHECK_EQ(lines(&l, "ds_add_u32"), 1);
      finish(&l);
   }
}

static void
packed_thread_ids_are_unpacked_for_compute_on_gfx11(void)
{
   struct lowered l;
   lower_full(&l, WORDS(0x4a040300, S_ENDPGM), AGC_ISA_TARGET_GFX11, AGC_SHADER_STAGE_COMPUTE, 32,
              AGC_SHADER_PRECISION_FUSED, NULL);
   CHECK(l.ok);
   expect(&l, "thread_ids/unpacked_gfx11");
   CHECK(has(&l, "v1 = v_bfe_u32 v0, 10, 10"));
   finish(&l);
   lower_full(&l, WORDS(0x4a040300, S_ENDPGM), AGC_ISA_TARGET_GFX103, AGC_SHADER_STAGE_COMPUTE, 32,
              AGC_SHADER_PRECISION_FUSED, NULL);
   CHECK(l.ok);
   expect(&l, "thread_ids/separate_gfx103");
   CHECK_EQ(lines(&l, "v_bfe_u32"), 0);
   finish(&l);
   lower(&l, WORDS(0x4a040300, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "thread_ids/not_compute_gfx11");
   CHECK_EQ(lines(&l, "v_bfe_u32"), 0);
   finish(&l);
}

static void
sixteen_bit_operands_above_v127_use_vop3_on_gfx11(void)
{
   struct lowered l;
   lower(&l, WORDS(0x65900501, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "t16/high_vgpr_gfx11");
   CHECK(has(&l, "v200 = v_add_f16 v1, v2 vop3"));
   finish(&l);
   lower(&l, WORDS(0x64000501, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "t16/low_vgpr_gfx11");
   CHECK(has(&l, "v0 = v_add_f16 v1, v2"));
   finish(&l);
}

static void
compare_stores_swap_their_data_operands_on_gfx11_and_later(void)
{
   static const enum agc_isa_target targets[] = {AGC_ISA_TARGET_GFX103, AGC_ISA_TARGET_GFX11, AGC_ISA_TARGET_GFX115,
                                                 AGC_ISA_TARGET_GFX12};
   struct lowered l;
   for (unsigned t = 0; t < ARRAY_SIZE(targets); t++) {
      lower(&l, WORDS(0xd8c00000, 0x04060502, S_ENDPGM), targets[t]);
      expect_on(&l, "lds/cmpst_rtn", targets[t]);
      finish(&l);
      if (targets[t] == AGC_ISA_TARGET_GFX103)
         continue;
      lower(&l, WORDS(0xd9c00000, 0x04060802, S_ENDPGM), targets[t]);
      expect_on(&l, "lds/cmpst_rtn_b64", targets[t]);
      finish(&l);
   }
   lower(&l, WORDS(0xd8800000, 0x04000502, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "lds/add_rtn_gfx11");
   finish(&l);
}

static void
gfx12_builds_the_lds_operations_it_removed_from_compare_stores(void)
{
   static const struct {
      uint32_t word0;
      const char *name;
   } cases[] = {{0xd8c40000u, "lds/cmpst_f32"}, {0xd8d00000u, "lds/wrap_rtn"}};
   struct lowered l;
   for (unsigned c = 0; c < ARRAY_SIZE(cases); c++) {
      lower(&l, WORDS(cases[c].word0, 0x04060502, S_ENDPGM), AGC_ISA_TARGET_GFX11);
      expect_on(&l, cases[c].name, AGC_ISA_TARGET_GFX11);
      finish(&l);
      lower(&l, WORDS(cases[c].word0, 0x04060502, S_ENDPGM), AGC_ISA_TARGET_GFX12);
      expect_on(&l, cases[c].name, AGC_ISA_TARGET_GFX12);
      CHECK_EQ(lines(&l, "ds_read_b32"), 1);
      CHECK_EQ(lines(&l, "ds_cmpst_rtn_b32"), 1);
      CHECK_EQ(lines(&l, "s_cbranch_execnz -> local_1"), 1);
      finish(&l);
   }
}

static void
table_corrections_for_older_and_newer_targets(void)
{
   struct lowered l;
   lower(&l, WORDS(0xd56b0000, 0x00020501, S_ENDPGM), AGC_ISA_TARGET_GFX103);
   expect(&l, "tables/vop3_0x16b_gfx103");
   finish(&l);
   lower(&l, WORDS(0xd56b0000, 0x00020501, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "tables/vop3_0x16b_gfx11");
   finish(&l);
   static const enum agc_isa_target ends[] = {AGC_ISA_TARGET_GFX11, AGC_ISA_TARGET_GFX115, AGC_ISA_TARGET_GFX12};
   for (unsigned t = 0; t < ARRAY_SIZE(ends); t++) {
      lower_with(&l, WORDS(0xbf9e0000), ends[t], AGC_SHADER_PRECISION_FUSED, 32);
      expect_on(&l, "tables/endpgm_ordered_ps_done", ends[t]);
      finish(&l);
   }
   lower(&l, WORDS(0xbf970000, S_ENDPGM), AGC_ISA_TARGET_GFX12);
   expect(&l, "tables/sopp_0x17_gfx12");
   finish(&l);
   lower(&l, WORDS(0xbf970000, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "tables/sopp_0x17_gfx11");
   finish(&l);
   lower(&l, WORDS(0x7e003600, S_ENDPGM), AGC_ISA_TARGET_GFX12);
   expect(&l, "tables/vop1_0x1b_gfx12");
   finish(&l);
}

static void
gfx11_0_64_bit_mads_never_overlap_their_sources(void)
{
   struct lowered l;
   lower(&l, WORDS(0xd5760404, 0x04120702, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "mad64/overlap_gfx11");
   CHECK_EQ(lines(&l, "v_mov_b32"), 2);
   finish(&l);
   static const enum agc_isa_target direct[] = {AGC_ISA_TARGET_GFX115, AGC_ISA_TARGET_GFX12};
   for (unsigned t = 0; t < ARRAY_SIZE(direct); t++) {
      lower(&l, WORDS(0xd5760404, 0x04120702, S_ENDPGM), direct[t]);
      expect_on(&l, "mad64/overlap", direct[t]);
      CHECK_EQ(lines(&l, "v_mov_b32"), 0);
      finish(&l);
   }
   lower(&l, WORDS(0xd5760404, 0x041a0702, S_ENDPGM), AGC_ISA_TARGET_GFX11);
   expect(&l, "mad64/disjoint_gfx11");
   CHECK_EQ(lines(&l, "v_mov_b32"), 0);
   finish(&l);
}

static void
the_prologue_is_its_own_block_so_branches_to_pc_0_skip_it(void)
{
   const uint32_t words[] = {0x4a060300, 0xbf068000, 0xbf84fffd, S_ENDPGM};
   struct lowered l;
   lower_compute(&l, words, ARRAY_SIZE(words), AGC_ISA_TARGET_GFX115, NULL);
   expect(&l, "prologue/own_block_gfx115");
   CHECK(l.listing && listing_block(l.listing, "prologue") == l.listing);
   CHECK(has(&l, "s_cbranch_scc0 -> pc_0"));
   finish(&l);
   lower_compute(&l, words, ARRAY_SIZE(words), AGC_ISA_TARGET_GFX103, NULL);
   expect(&l, "prologue/none_gfx103");
   CHECK(l.listing && !listing_block(l.listing, "prologue"));
   finish(&l);
}

static void
packed_thread_ids_are_masked_whenever_y_or_z_is_enabled(void)
{
   static const char *const names[2] = {"thread_ids/x_only", "thread_ids/x_and_z"};
   for (uint32_t components = 0; components <= 2; components += 2) {
      struct agc_shader_interface iface;
      memset(&iface, 0, sizeof(iface));
      iface.has_guest_pgm_rsrc2 = true;
      iface.guest_pgm_rsrc2 = components << AGC_ISA_RSRC2_TIDIG_SHIFT;
      struct lowered l;
      lower_compute(&l, WORDS(0x7e080300, S_ENDPGM), AGC_ISA_TARGET_GFX115, &iface);
      expect(&l, names[components / 2]);
      bool prologue = l.listing && listing_block(l.listing, "prologue") != NULL;
      CHECK(prologue == (components == 2));
      finish(&l);
   }
}

static void
gfx10_fields_unpack(void)
{
   static const struct {
      uint16_t simm16;
      uint8_t vm, exp, lgkm;
   } cases[] = {{0x0070, 0, 7, 0}, {0xc07f, 63, 7, 0}, {0x0a25, 5, 2, 10}, {0xbf78, 40, 7, 63}};
   for (unsigned i = 0; i < ARRAY_SIZE(cases); i++) {
      uint8_t vm, exp, lgkm;
      agc_isa_waitcnt_decode_gfx10(cases[i].simm16, &vm, &exp, &lgkm);
      CHECK(vm == cases[i].vm && exp == cases[i].exp && lgkm == cases[i].lgkm);
   }
}

static void
gfx10_to_gfx11_repacks_every_counter(void)
{
   static const uint16_t cases[][2] = {{0x0070, 0x0007}, {0xc07f, 0xfc07}, {0x0a25, 0x14a2}, {0xbf78, 0xa3f7}};
   for (unsigned i = 0; i < ARRAY_SIZE(cases); i++) {
      uint8_t vm, exp, lgkm;
      agc_isa_waitcnt_decode_gfx10(cases[i][0], &vm, &exp, &lgkm);
      CHECK_EQ(agc_isa_waitcnt_encode_gfx11(vm, exp, lgkm), cases[i][1]);
      uint8_t vm11, exp11, lgkm11;
      agc_isa_waitcnt_decode_gfx11(cases[i][1], &vm11, &exp11, &lgkm11);
      CHECK(vm11 == vm && exp11 == exp && lgkm11 == lgkm);
      CHECK_EQ(agc_isa_waitcnt_encode_gfx10(vm, exp, lgkm), cases[i][0]);
   }
}

static void
gfx12_waits_at_program_entry_only_cover_issued_types(void)
{
   struct lowered l;
   lower(&l,
         WORDS(0xe0301000, 0x80020100, 0xf0800f08, 0x00820004, 0xe0301000, 0x80020200, 0xbf8c3f71, 0xbf8c3f71,
               S_ENDPGM),
         AGC_ISA_TARGET_GFX12);
   expect(&l, "waits/entry_issued_types_gfx12");
   CHECK_EQ(lines(&l, "s_wait_loadcnt"), 1);
   CHECK_EQ(lines(&l, "s_wait_samplecnt"), 1);
   CHECK_EQ(lines(&l, "s_wait_"), 2);
   finish(&l);
}

static void
gfx12_waits_in_unknown_blocks_are_conservative(void)
{
   struct lowered l;
   lower(&l, WORDS(0xbf840000, 0xf0800f08, 0x00820004, 0xbf8c0003, 0xbbfd0000, S_ENDPGM), AGC_ISA_TARGET_GFX12);
   expect(&l, "waits/unknown_block_gfx12");
   CHECK_EQ(lines(&l, "s_wait_"), 7);
   finish(&l);
}

static void
gfx12_scalar_loads_always_drain_to_zero(void)
{
   struct lowered l;
   lower(&l,
         WORDS(0xf4000100, 0xfa000000, 0xd8dc0201, 0x02000002, 0xd8dc0201, 0x04000002, 0xbf8cc27f, 0xbf8cc17f,
               0xbf8cff7f, S_ENDPGM),
         AGC_ISA_TARGET_GFX12);
   expect(&l, "waits/scalar_loads_drain_gfx12");
   CHECK_EQ(lines(&l, "s_wait_kmcnt"), 1);
   CHECK_EQ(lines(&l, "s_wait_dscnt"), 1);
   CHECK_EQ(lines(&l, "s_wait_"), 2);
   finish(&l);
}

int
main(void)
{
   listing_set_open(&listings, vector_listings, ARRAY_SIZE(vector_listings), "test_isa_lower_vector_listings.h",
                    "vector_listings");
   mac_becomes_fmac_on_every_newer_target();
   mad_becomes_fma_with_modifiers_preserved();
   legacy_mad_is_unfused_on_every_target();
   legacy_mad_flushes_f32_denormals_when_the_mode_keeps_them();
   exact_precision_splits_mad_into_mul_and_add();
   sdwa_expands_to_bitfield_sequences_on_gfx11();
   sdwa_is_kept_on_rdna2();
   an_sdwa_move_changes_the_sign_with_a_bit_operation();
   waitcnt_is_repacked_for_gfx11();
   waitcnt_splits_into_gfx12_counters();
   compares_are_renumbered_and_keep_exec_semantics();
   gfx12_replaces_constant_compares();
   removed_features_are_rejected_explicitly();
   packed_thread_ids_are_unpacked_for_compute_on_gfx11();
   sixteen_bit_operands_above_v127_use_vop3_on_gfx11();
   compare_stores_swap_their_data_operands_on_gfx11_and_later();
   gfx12_builds_the_lds_operations_it_removed_from_compare_stores();
   table_corrections_for_older_and_newer_targets();
   gfx11_0_64_bit_mads_never_overlap_their_sources();
   the_prologue_is_its_own_block_so_branches_to_pc_0_skip_it();
   packed_thread_ids_are_masked_whenever_y_or_z_is_enabled();
   gfx10_fields_unpack();
   gfx10_to_gfx11_repacks_every_counter();
   gfx12_waits_at_program_entry_only_cover_issued_types();
   gfx12_waits_in_unknown_blocks_are_conservative();
   gfx12_scalar_loads_always_drain_to_zero();
   listing_set_close(&listings);
   return TEST_RESULT();
}
