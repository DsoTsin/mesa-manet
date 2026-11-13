#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agc_isa.h"
#include "agc_lower.h"
#include "agc_isa_regalloc.h"
#include "test_isa_listing.h"

#include "test_isa_lower_stages_listings.h"

#define S_ENDPGM 0xbf810000u
#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

static struct listing_set listings;

enum setup {
   SETUP_PLAIN,
   SETUP_PIXEL,
   SETUP_INTERFACE
};

struct run {
   struct agc_shader_options options;
   struct agc_shader_interface iface;
   struct agc_isa_ir ir;
   struct agc_isa_regalloc regs;
   struct agc_isa_error error;
   char *listing;
   bool analyzed;
   bool ok;
};

static void
options_init(struct agc_shader_options *options, enum agc_isa_target target, enum agc_shader_stage stage,
             uint8_t wave_size)
{
   memset(options, 0, sizeof(*options));
   options->target = (enum agc_shader_isa)target;
   options->stage = stage;
   options->wave_size = wave_size;
   options->ray_unit = AGC_SHADER_RAY_UNIT_HOST;
}

static bool
analyze(struct run *run, const uint32_t *words, size_t count, enum agc_isa_target target,
        enum agc_shader_stage stage, uint8_t wave_size, const struct agc_shader_interface *iface)
{
   memset(run, 0, sizeof(*run));
   options_init(&run->options, target, stage, wave_size);
   if (iface)
      run->iface = *iface;
   agc_isa_ir_init(&run->ir);
   run->analyzed = agc_isa_analyze(target, words, count, &run->options, &run->iface, &run->ir, &run->error);
   CHECK(run->analyzed);
   if (run->analyzed)
      agc_isa_regalloc_init(&run->regs, &run->ir.metadata, &run->options, &run->iface);
   return run->analyzed;
}

static bool
lower(struct run *run, const uint32_t *words, size_t count, enum agc_isa_target target, enum agc_shader_stage stage,
      uint8_t wave_size, enum agc_shader_precision precision, const struct agc_shader_interface *iface,
      enum setup setup)
{
   if (!analyze(run, words, count, target, stage, wave_size, iface))
      return false;
   struct agc_isa_lowerer lowerer;
   agc_isa_lowerer_init(&lowerer, &run->options, precision);
   if (setup == SETUP_PIXEL) {
      bool known = run->iface.has_user_sgpr_count || run->iface.has_guest_pgm_rsrc2;
      agc_isa_lowerer_with_pixel_interface(&lowerer, known, run->ir.metadata.user_sgpr_count,
                                           run->iface.has_ps_col_format, run->iface.ps_col_format,
                                           run->iface.has_ps_z_format, run->iface.ps_z_format,
                                           run->iface.dual_source_blend);
   } else if (setup == SETUP_INTERFACE) {
      agc_isa_lowerer_with_interface(&lowerer, &run->ir, &run->iface);
   }
   run->ok = agc_isa_lower_listing(&lowerer, &run->ir, &run->regs, &run->listing, &run->error);
   return run->ok;
}

static void
run_finish(struct run *run)
{
   free(run->listing);
   run->listing = NULL;
   agc_isa_ir_finish(&run->ir);
}

static bool
lowered(const struct run *run, const char *test)
{
   if (!run->analyzed)
      return false;
   if (!run->ok) {
      char text[AGC_ISA_ERROR_TEXT_SIZE];
      agc_isa_error_format(&run->error, text, sizeof(text));
      fprintf(stderr, "%s: %s\n", test, text);
   }
   CHECK(run->ok);
   return run->ok;
}

static bool
refused(const struct run *run, const char *reason)
{
   CHECK(run->analyzed);
   CHECK(!run->ok);
   if (!run->analyzed || run->ok)
      return false;
   bool found = agc_isa_error_is_unsupported(&run->error) && run->error.reason &&
                strstr(run->error.reason, reason) != NULL;
   if (!found) {
      char text[AGC_ISA_ERROR_TEXT_SIZE];
      agc_isa_error_format(&run->error, text, sizeof(text));
      fprintf(stderr, "expected a refusal mentioning \"%s\": %s\n", reason, text);
   }
   CHECK(found);
   return found;
}

static void
expect(const struct run *run, const char *name)
{
   CHECK(listing_matches(&listings, name, run->listing));
}

static unsigned
outside_prologue(const char *listing, const char *needle)
{
   unsigned all = listing_lines(listing, needle, false);
   const char *prologue = listing_block(listing, "prologue");
   return prologue ? all - listing_lines(prologue, needle, true) : all;
}

static struct agc_isa_inst
decode(const uint32_t *words, size_t count)
{
   struct agc_isa_inst inst;
   struct agc_isa_error error;
   memset(&inst, 0, sizeof(inst));
   CHECK(agc_isa_decode_at(words, count, 0, &inst, &error));
   return inst;
}

static const uint32_t interp_p1_p2[] = {0xbefc0302, 0xc8080500, 0xc8090501, S_ENDPGM};

static void
test_interpolation_loads_parameters_with_every_lane_enabled_on_gfx11(void)
{
   struct run run;
   lower(&run, interp_p1_p2, ARRAY_SIZE(interp_p1_p2), AGC_ISA_TARGET_GFX11, AGC_SHADER_STAGE_FRAGMENT, 32,
         AGC_SHADER_PRECISION_FUSED, NULL, SETUP_PLAIN);
   if (lowered(&run, __func__)) {
      expect(&run, "interpolation/in_place");
      CHECK_EQ(listing_lines(run.listing, "lds_param_load", false), 2);
      CHECK_EQ(listing_lines(run.listing, "v_interp_p10_f32_inreg", false), 1);
      CHECK_EQ(listing_lines(run.listing, "v_interp_p2_f32_inreg", false), 1);
      CHECK(listing_has_line(run.listing, "exec_lo = s_mov_b32 -1"));
   }
   run_finish(&run);
}

static void
pixel_interface_with_user_sgprs(struct agc_shader_interface *iface, uint8_t user_sgprs)
{
   memset(iface, 0, sizeof(*iface));
   iface->has_user_sgpr_count = true;
   iface->user_sgpr_count = user_sgprs;
}

static void
plan_for(const uint32_t *words, size_t count, const struct agc_shader_interface *iface,
         struct agc_isa_pixel_plan *plan, struct run *run, bool *ok, uint16_t *vgpr_base)
{
   *ok = false;
   memset(plan, 0, sizeof(*plan));
   if (!analyze(run, words, count, AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_FRAGMENT, 32, iface))
      return;
   *vgpr_base = agc_isa_regalloc_scratch_base(&run->regs);
   struct agc_isa_pixel_interface pixel;
   memset(&pixel, 0, sizeof(pixel));
   pixel.has_prim_mask = iface->has_user_sgpr_count || iface->has_guest_pgm_rsrc2;
   pixel.prim_mask = run->ir.metadata.user_sgpr_count;
   *ok = agc_isa_pixel_plan_init(plan, &run->ir, &run->regs, false, &pixel, &run->error);
   CHECK(*ok);
}

static void
test_a_known_prim_mask_hoists_each_parameter_load_to_the_prologue(void)
{
   struct agc_shader_interface iface;
   pixel_interface_with_user_sgprs(&iface, 2);
   struct run run;
   struct agc_isa_pixel_plan plan;
   bool ok;
   uint16_t base = 0;
   plan_for(interp_p1_p2, ARRAY_SIZE(interp_p1_p2), &iface, &plan, &run, &ok, &base);
   if (ok) {
      CHECK_EQ(base, 3);
      CHECK_EQ(agc_isa_regalloc_scratch_base(&run.regs), base + 1);
      CHECK_EQ(plan.hoisted_count, 1);
      CHECK(plan.hoisted[0].attr == 1 && plan.hoisted[0].chan == 1 && plan.hoisted[0].vgpr == base);
      uint8_t vgpr = 0;
      CHECK(agc_isa_pixel_plan_hoisted(&plan, 4, 1, 1, &vgpr) && vgpr == base);
      CHECK(agc_isa_pixel_plan_hoisted(&plan, 8, 1, 1, &vgpr) && vgpr == base);
      CHECK(!agc_isa_pixel_plan_hoisted(&plan, 4, 1, 2, &vgpr));
      CHECK(!agc_isa_pixel_plan_hoisted(&plan, 0, 1, 1, &vgpr));
      CHECK_EQ(plan.null_target, AGC_ISA_EXP_MRT0);
   }
   agc_isa_pixel_plan_finish(&plan);
   run_finish(&run);

   lower(&run, interp_p1_p2, ARRAY_SIZE(interp_p1_p2), AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_FRAGMENT, 32,
         AGC_SHADER_PRECISION_EXACT, &iface, SETUP_PIXEL);
   if (lowered(&run, __func__)) {
      expect(&run, "interpolation/hoisted");
      const char *prologue = listing_block(run.listing, "prologue");
      CHECK(prologue == run.listing);
      if (prologue) {
         CHECK(listing_has_line(prologue, "m0 = s_mov_b32 s2"));
         CHECK_EQ(listing_lines(prologue, "lds_param_load", true), 1);
      }
      CHECK_EQ(outside_prologue(run.listing, "lds_param_load"), 0);
      CHECK_EQ(outside_prologue(run.listing, "_inreg"), 2);
   }
   run_finish(&run);

   static const uint32_t other[] = {0xbefc0303, 0xc8080500, S_ENDPGM};
   plan_for(other, ARRAY_SIZE(other), &iface, &plan, &run, &ok, &base);
   if (ok) {
      CHECK_EQ(plan.hoisted_count, 0);
      CHECK_EQ(plan.from_prologue.count, 0);
   }
   agc_isa_pixel_plan_finish(&plan);
   run_finish(&run);
   lower(&run, other, ARRAY_SIZE(other), AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_FRAGMENT, 32,
         AGC_SHADER_PRECISION_EXACT, &iface, SETUP_PIXEL);
   if (lowered(&run, __func__)) {
      expect(&run, "interpolation/another_mask");
      CHECK(!listing_block(run.listing, "prologue"));
      CHECK_EQ(listing_lines(run.listing, "lds_param_load", false), 1);
   }
   run_finish(&run);
}

static void
prim_mask_case(const uint32_t *words, size_t count, unsigned in_place, const char *name)
{
   struct agc_shader_interface iface;
   pixel_interface_with_user_sgprs(&iface, 2);
   struct run run;
   struct agc_isa_pixel_plan plan;
   bool ok;
   uint16_t base = 0;
   plan_for(words, count, &iface, &plan, &run, &ok, &base);
   if (ok)
      CHECK_EQ(plan.hoisted_count, in_place ? 0 : 1);
   agc_isa_pixel_plan_finish(&plan);
   run_finish(&run);
   lower(&run, words, count, AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_FRAGMENT, 32, AGC_SHADER_PRECISION_EXACT, &iface,
         SETUP_PIXEL);
   if (lowered(&run, name)) {
      expect(&run, name);
      CHECK_EQ(outside_prologue(run.listing, "lds_param_load"), in_place);
   }
   run_finish(&run);
}

static void
test_prim_mask_copies_and_rewrites_are_followed_through_control_flow(void)
{
   static const uint32_t copy_of_copy[] = {0xbe850302, 0xbefc0305, 0xc8080500, S_ENDPGM};
   static const uint32_t overwritten[] = {0xbefc0302, 0x80028002, 0xbefc0302, 0xc8080500, S_ENDPGM};
   static const uint32_t one_path[] = {0xbefc0302, 0xbf840001, 0xbefc0303, 0xc8080500, S_ENDPGM};
   prim_mask_case(copy_of_copy, ARRAY_SIZE(copy_of_copy), 0, "prim_mask/copy_of_copy");
   prim_mask_case(overwritten, ARRAY_SIZE(overwritten), 1, "prim_mask/overwritten");
   prim_mask_case(one_path, ARRAY_SIZE(one_path), 1, "prim_mask/one_path");
}

static const uint32_t ordered_pixel[] = {
   0xbe860304, 0xbf048003, 0xbf840002, 0xb9831019, 0x8105ef03, 0xbf900007, 0xf8001890, 0x00000000, S_ENDPGM,
};
#define ORDERED_RSRC2 ((2u << 1) | (1u << 25) | (1u << 26))

static void
ordered_interface(struct agc_shader_interface *iface)
{
   memset(iface, 0, sizeof(*iface));
   iface->has_guest_pgm_rsrc2 = true;
   iface->guest_pgm_rsrc2 = ORDERED_RSRC2;
}

static uint32_t
host_rsrc2(const struct run *run)
{
   struct agc_isa_shader_config config;
   agc_isa_regalloc_finish(&run->regs, &run->ir.metadata, &run->options, &run->iface, &config);
   return config.pgm_rsrc2;
}

static void
test_ordered_pixel_shading_waits_the_way_the_target_does(void)
{
   struct agc_shader_interface iface;
   ordered_interface(&iface);
   static const enum agc_isa_target targets[2] = {AGC_ISA_TARGET_GFX115, AGC_ISA_TARGET_GFX12};
   static const char *const names[2] = {"ordered/gfx115", "ordered/gfx12"};
   for (unsigned t = 0; t < 2; t++) {
      struct run run;
      lower(&run, ordered_pixel, ARRAY_SIZE(ordered_pixel), targets[t], AGC_SHADER_STAGE_FRAGMENT, 64,
            AGC_SHADER_PRECISION_FUSED, &iface, SETUP_INTERFACE);
      if (lowered(&run, __func__)) {
         expect(&run, names[t]);
         CHECK(listing_has_line(run.listing, "s3 = s_mov_b32 0x80000001"));
         CHECK(listing_has_line(run.listing, "s4 = s_mov_b32 0"));
         CHECK_EQ(listing_lines(run.listing, "s_wait_event", false), 1);
         CHECK(listing_has_line(run.listing, "s_wait_event wait_export_ready"));
         CHECK(listing_has_line(run.listing, "s5, scc = s_add_i32 s3, 1"));
         CHECK_EQ(listing_lines(run.listing, "reg_b32", false), 0);
         CHECK_EQ(listing_lines(run.listing, "s_sendmsg", false), 0);
         CHECK_EQ(host_rsrc2(&run) & (3u << 25), 0);
      }
      run_finish(&run);
   }
   struct run run;
   lower(&run, ordered_pixel, ARRAY_SIZE(ordered_pixel), AGC_ISA_TARGET_GFX103, AGC_SHADER_STAGE_FRAGMENT, 64,
         AGC_SHADER_PRECISION_FUSED, &iface, SETUP_INTERFACE);
   if (lowered(&run, __func__)) {
      expect(&run, "ordered/gfx103");
      CHECK_EQ(listing_lines(run.listing, "reg_b32", false), 1);
      CHECK_EQ(listing_lines(run.listing, "s_wait_event", false), 0);
      CHECK_EQ(host_rsrc2(&run) & (3u << 25), 3u << 25);
   }
   run_finish(&run);
}

static void
test_a_done_export_before_the_end_of_the_ordered_section_is_refused(void)
{
   uint32_t words[ARRAY_SIZE(ordered_pixel)];
   memcpy(words, ordered_pixel, sizeof(words));
   memmove(&words[5], &words[6], 2 * sizeof(words[0]));
   words[7] = 0xbf900007;
   struct agc_shader_interface iface;
   ordered_interface(&iface);
   struct run run;
   lower(&run, words, ARRAY_SIZE(words), AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_FRAGMENT, 64,
         AGC_SHADER_PRECISION_FUSED, &iface, SETUP_INTERFACE);
   if (refused(&run, "ordered section"))
      CHECK_EQ(run.error.pc, 20);
   run_finish(&run);
   lower(&run, words, ARRAY_SIZE(words), AGC_ISA_TARGET_GFX103, AGC_SHADER_STAGE_FRAGMENT, 64,
         AGC_SHADER_PRECISION_FUSED, &iface, SETUP_INTERFACE);
   if (lowered(&run, __func__))
      expect(&run, "ordered/early_done_gfx103");
   run_finish(&run);
}

static void
test_pops_exiting_wave_reads_become_this_wave(void)
{
   static const uint32_t add[] = {0x8105ef03};
   struct agc_isa_inst inst = decode(add, 1);
   CHECK(inst.fields == AGC_ISA_SOP2 && inst.sop2.src1.kind == AGC_ISA_OPERAND_POPS_EXITING_WAVE_ID);
   struct agc_isa_inst rewritten;
   CHECK(agc_isa_lowering_pops_without_exiting_wave(&inst, &rewritten));
   CHECK(agc_isa_operand_equal(rewritten.sop2.src0, agc_isa_operand_sgpr(3)) &&
         agc_isa_operand_equal(rewritten.sop2.src1, agc_isa_operand_inline_int(1)));
   CHECK(agc_isa_operand_equal(rewritten.sop2.sdst, agc_isa_operand_sgpr(5)));
   static const uint32_t mov[] = {0xbe860304};
   inst = decode(mov, 1);
   CHECK(!agc_isa_lowering_pops_without_exiting_wave(&inst, &rewritten));
}

static void
compute_interface(struct agc_shader_interface *iface, bool has_rsrc2, uint32_t rsrc2)
{
   memset(iface, 0, sizeof(*iface));
   iface->has_guest_pgm_rsrc2 = has_rsrc2;
   iface->guest_pgm_rsrc2 = rsrc2;
}

static void
test_flat_scratch_setup_is_left_to_gfx11_hardware(void)
{
   static const uint32_t words[] = {0x80098408, 0xb989f814, S_ENDPGM};
   struct agc_shader_interface iface;
   compute_interface(&iface, false, 0);
   static const enum agc_isa_target targets[2] = {AGC_ISA_TARGET_GFX103, AGC_ISA_TARGET_GFX115};
   static const char *const names[2] = {"flat_scratch/gfx103", "flat_scratch/gfx115"};
   static const unsigned setregs[2] = {1, 0};
   for (unsigned t = 0; t < 2; t++) {
      struct run run;
      lower(&run, words, ARRAY_SIZE(words), targets[t], AGC_SHADER_STAGE_COMPUTE, 32, AGC_SHADER_PRECISION_EXACT,
            &iface, SETUP_PLAIN);
      if (lowered(&run, __func__)) {
         expect(&run, names[t]);
         CHECK_EQ(listing_lines(run.listing, "s_setreg_b32", false), setregs[t]);
      }
      run_finish(&run);
   }
}

static void
test_other_reads_of_the_scratch_wave_offset_are_rejected_on_gfx11(void)
{
   static const uint32_t words[] = {0x80098408, 0x7e020209, S_ENDPGM};
   struct agc_shader_interface iface;
   compute_interface(&iface, true, 1 | (8 << 1));
   struct run run;
   lower(&run, words, ARRAY_SIZE(words), AGC_ISA_TARGET_GFX103, AGC_SHADER_STAGE_COMPUTE, 32,
         AGC_SHADER_PRECISION_EXACT, &iface, SETUP_PLAIN);
   if (lowered(&run, __func__))
      expect(&run, "wave_offset/read_gfx103");
   run_finish(&run);
   lower(&run, words, ARRAY_SIZE(words), AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_COMPUTE, 32,
         AGC_SHADER_PRECISION_EXACT, &iface, SETUP_PLAIN);
   if (refused(&run, "scratch wave offset"))
      CHECK_EQ(run.error.pc, 4);
   run_finish(&run);
}

static void
test_wave_offset_uses(void)
{
   struct agc_shader_interface iface;
   compute_interface(&iface, true, 1 | (8 << 1));
   static const uint32_t private_load[] = {0xe0301000, 0x08000100, S_ENDPGM};
   struct run run;
   if (analyze(&run, private_load, ARRAY_SIZE(private_load), AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_COMPUTE, 32,
               &iface)) {
      uint8_t sgpr = 0;
      CHECK(agc_isa_regalloc_scratch_wave_offset_sgpr(&run.regs, &sgpr) && sgpr == 8);
      struct agc_isa_wave_offset_uses uses;
      memset(&uses, 0, sizeof(uses));
      CHECK(agc_isa_lowering_wave_offset_find(&run.ir, false, 8, AGC_ISA_TARGET_GFX115, &uses, &run.error));
      CHECK_EQ(uses.private_pcs.count, 1);
      CHECK(agc_isa_pc_set_contains(&uses.private_pcs, 0));
      CHECK(!uses.computed);
      agc_isa_pc_set_finish(&uses.private_pcs);
   }
   run_finish(&run);
   lower(&run, private_load, ARRAY_SIZE(private_load), AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_COMPUTE, 32,
         AGC_SHADER_PRECISION_EXACT, &iface, SETUP_PLAIN);
   if (lowered(&run, __func__))
      expect(&run, "wave_offset/private_load");
   run_finish(&run);

   static const uint32_t computed[] = {0x80098408, S_ENDPGM};
   if (analyze(&run, computed, ARRAY_SIZE(computed), AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_COMPUTE, 32,
               &iface)) {
      struct agc_isa_wave_offset_uses uses;
      memset(&uses, 0, sizeof(uses));
      CHECK(agc_isa_lowering_wave_offset_find(&run.ir, false, 8, AGC_ISA_TARGET_GFX115, &uses, &run.error));
      CHECK_EQ(uses.private_pcs.count, 0);
      CHECK(uses.computed);
      agc_isa_pc_set_finish(&uses.private_pcs);
   }
   run_finish(&run);

   static const uint32_t stray[] = {0x80098408, 0x7e020209, S_ENDPGM};
   if (analyze(&run, stray, ARRAY_SIZE(stray), AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_COMPUTE, 32, &iface)) {
      struct agc_isa_wave_offset_uses uses;
      memset(&uses, 0, sizeof(uses));
      CHECK(!agc_isa_lowering_wave_offset_find(&run.ir, false, 8, AGC_ISA_TARGET_GFX115, &uses, &run.error));
      CHECK_EQ(run.error.kind, AGC_ISA_ERROR_UNSUPPORTED_INSTRUCTION);
      CHECK_EQ(run.error.pc, 4);
      agc_isa_pc_set_finish(&uses.private_pcs);
   }
   run_finish(&run);
}

static void
test_interpolated_channels(void)
{
   static const uint32_t p1[] = {0xc8080500};
   struct agc_isa_inst inst = decode(p1, 1);
   uint8_t attr = 0;
   uint8_t chan = 0;
   CHECK(agc_isa_lowering_interpolated(&inst, &attr, &chan) && attr == 1 && chan == 1);
   static const uint32_t p1ll[] = {0xd7420005, 0x00020283};
   inst = decode(p1ll, 2);
   CHECK(inst.family == AGC_ISA_VOP3 && inst.opcode == AGC_ISA_VOP3_INTERP_P1LL_F16);
   CHECK(agc_isa_lowering_interpolated(&inst, &attr, &chan) && attr == 3 && chan == 2);
   static const uint32_t mov[] = {0xbefc0302};
   inst = decode(mov, 1);
   CHECK(!agc_isa_lowering_interpolated(&inst, &attr, &chan));
   CHECK(agc_isa_pixel_is_vop3_interp(AGC_ISA_VOP3_INTERP_P1_F32));
   CHECK(agc_isa_pixel_is_vop3_interp(AGC_ISA_VOP3_INTERP_MOV_F32));
   CHECK(agc_isa_pixel_is_vop3_interp(AGC_ISA_VOP3_INTERP_P2_F16));
   CHECK(!agc_isa_pixel_is_vop3_interp(AGC_ISA_VOP3_INTERP_MOV_F32 + 1));
}

static void
test_null_exports_get_export_memory(void)
{
   uint32_t format = 0xdead;
   CHECK(agc_isa_pixel_col_format_for(true, 0, false, 0, true, &format) && format == AGC_ISA_SPI_SHADER_32_R);
   CHECK(agc_isa_pixel_col_format_for(true, 0x40, true, 0, true, &format) && format == 0x41);
   CHECK(!agc_isa_pixel_col_format_for(true, 0, true, 3, true, &format));
   CHECK(!agc_isa_pixel_col_format_for(true, 4, false, 0, true, &format));
   CHECK(!agc_isa_pixel_col_format_for(true, 0, false, 0, false, &format));
   CHECK(!agc_isa_pixel_col_format_for(false, 0, false, 0, true, &format));
}

static void
test_pixel_plan_targets_and_dual_source(void)
{
   static const uint32_t words[] = {0x7e000280, S_ENDPGM};
   struct run run;
   if (analyze(&run, words, ARRAY_SIZE(words), AGC_ISA_TARGET_GFX11, AGC_SHADER_STAGE_FRAGMENT, 32, NULL)) {
      uint16_t vgprs = agc_isa_regalloc_scratch_base(&run.regs);
      struct agc_isa_pixel_interface pixel;
      memset(&pixel, 0, sizeof(pixel));
      pixel.has_col_format = true;
      pixel.col_format = 0;
      pixel.has_z_format = true;
      pixel.z_format = 1;
      pixel.dual_source = true;
      struct agc_isa_pixel_plan plan;
      CHECK(agc_isa_pixel_plan_init(&plan, &run.ir, &run.regs, false, &pixel, &run.error));
      CHECK_EQ(plan.null_target, AGC_ISA_EXP_MRTZ);
      CHECK(plan.has_dual_source && plan.dual_source == vgprs);
      CHECK_EQ(agc_isa_regalloc_scratch_base(&run.regs), vgprs + 4);
      CHECK_EQ(plan.hoisted_count, 0);
      agc_isa_pixel_plan_finish(&plan);
   }
   run_finish(&run);
}

static void
test_dual_source_exports_are_swizzled_into_one_pair(void)
{
   static const uint32_t words[] = {0xf800000f, 0x03020100, 0xf800081f, 0x07060504, S_ENDPGM};
   struct agc_shader_interface iface;
   memset(&iface, 0, sizeof(iface));
   iface.dual_source_blend = true;
   static const uint8_t waves[2] = {32, 64};
   static const char *const names[2] = {"dual_source/wave32", "dual_source/wave64"};
   for (unsigned w = 0; w < 2; w++) {
      struct run run;
      lower(&run, words, ARRAY_SIZE(words), AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_FRAGMENT, waves[w],
            AGC_SHADER_PRECISION_FUSED, &iface, SETUP_INTERFACE);
      if (lowered(&run, __func__)) {
         expect(&run, names[w]);
         CHECK_EQ(listing_lines(run.listing, "exp ", false), 2);
      }
      run_finish(&run);
   }
   static const uint32_t twice[] = {0xf800000f, 0x03020100, 0xf800080f, 0x07060504, S_ENDPGM};
   struct run run;
   lower(&run, twice, ARRAY_SIZE(twice), AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_FRAGMENT, 32,
         AGC_SHADER_PRECISION_FUSED, &iface, SETUP_INTERFACE);
   refused(&run, "once each");
   run_finish(&run);
   static const uint32_t alone[] = {0xf800000f, 0x03020100, S_ENDPGM};
   lower(&run, alone, ARRAY_SIZE(alone), AGC_ISA_TARGET_GFX115, AGC_SHADER_STAGE_FRAGMENT, 32,
         AGC_SHADER_PRECISION_FUSED, &iface, SETUP_INTERFACE);
   refused(&run, "in one block");
   run_finish(&run);
}

static const uint32_t exp_param0[2] = {0xf800020f, 0x03020100};
static const uint32_t exp_pos0_done[2] = {0xf80008cf, 0x07060504};
static const uint32_t exp_pos1_done[2] = {0xf80008df, 0x07060504};

static size_t
program(uint32_t *words, const uint32_t *const *exports, size_t count)
{
   size_t n = 0;
   for (size_t i = 0; i < count; i++) {
      words[n++] = exports[i][0];
      words[n++] = exports[i][1];
   }
   words[n++] = S_ENDPGM;
   return n;
}

static const struct agc_shader_attribute_ring test_ring = {0x123456789000ull, 0x10000};

static bool
ngg_plan(const uint32_t *words, size_t count, bool has_ring, struct agc_isa_ngg_plan *plan, struct run *run,
         uint16_t *sgpr_base, uint16_t *vgpr_base)
{
   if (!analyze(run, words, count, AGC_ISA_TARGET_GFX11, AGC_SHADER_STAGE_VERTEX, 32, NULL))
      return false;
   *sgpr_base = run->regs.scratch_sgpr_base;
   *vgpr_base = agc_isa_regalloc_scratch_base(&run->regs);
   return agc_isa_ngg_plan_init(plan, &run->ir, &run->regs, false, has_ring, test_ring, false, &run->error);
}

static void
test_ngg_plans_ring_stores_and_deferral(void)
{
   uint32_t words[16];
   struct run run;
   struct agc_isa_ngg_plan plan;
   uint16_t sgprs;
   uint16_t vgprs;
   uint32_t ring_words[4];
   agc_isa_ngg_ring_descriptor(test_ring, 1, ring_words);

   const uint32_t *before[2] = {exp_param0, exp_pos0_done};
   size_t count = program(words, before, 2);
   CHECK(ngg_plan(words, count, true, &plan, &run, &sgprs, &vgprs));
   CHECK(plan.has_ring && !plan.has_deferred);
   CHECK(!plan.ring.descriptor.lane && plan.ring.descriptor.index == sgprs);
   CHECK(!plan.ring.offset.lane && plan.ring.offset.index == sgprs + 4);
   CHECK_EQ(plan.ring.vertex, vgprs);
   CHECK(!memcmp(plan.ring.words, ring_words, sizeof(ring_words)));
   CHECK_EQ(plan.persistent_lanes, 0);
   run_finish(&run);

   CHECK(ngg_plan(words, count, false, &plan, &run, &sgprs, &vgprs));
   CHECK(!plan.has_ring && !plan.has_deferred);
   run_finish(&run);

   const uint32_t *after[2] = {exp_pos0_done, exp_param0};
   count = program(words, after, 2);
   CHECK(ngg_plan(words, count, true, &plan, &run, &sgprs, &vgprs));
   CHECK(plan.has_ring && plan.has_deferred);
   CHECK_EQ(plan.deferred.target, AGC_ISA_EXP_POS_FIRST);
   CHECK_EQ(plan.deferred.en, 0xf);
   CHECK_EQ(plan.deferred.data, vgprs + 1);
   CHECK(!plan.deferred.exec.lane && plan.deferred.exec.index == ((sgprs + 5 + 3) & ~3u));
   run_finish(&run);

   const uint32_t *differ[3] = {exp_pos0_done, exp_param0, exp_pos1_done};
   count = program(words, differ, 3);
   CHECK(!ngg_plan(words, count, true, &plan, &run, &sgprs, &vgprs));
   CHECK(run.error.reason && strstr(run.error.reason, "different targets or channels"));
   CHECK_EQ(run.error.pc, 0);
   run_finish(&run);

   const uint32_t *positions[1] = {exp_pos0_done};
   count = program(words, positions, 1);
   CHECK(ngg_plan(words, count, true, &plan, &run, &sgprs, &vgprs));
   CHECK(!plan.has_ring && !plan.has_deferred);
   run_finish(&run);
}

static void
test_ngg_exports_go_to_the_attribute_ring_and_positions_wait_for_them(void)
{
   uint32_t words[16];
   struct agc_shader_interface iface;
   memset(&iface, 0, sizeof(iface));
   iface.has_attribute_ring = true;
   iface.attribute_ring = test_ring;
   static const uint8_t waves[2] = {32, 64};
   static const char *const stored[2] = {"ngg/ring_store_wave32", "ngg/ring_store_wave64"};
   static const char *const deferred[2] = {"ngg/deferred_position_wave32", "ngg/deferred_position_wave64"};
   for (unsigned w = 0; w < 2; w++) {
      struct run run;
      const uint32_t *before[2] = {exp_param0, exp_pos0_done};
      size_t count = program(words, before, 2);
      lower(&run, words, count, AGC_ISA_TARGET_GFX11, AGC_SHADER_STAGE_VERTEX, waves[w], AGC_SHADER_PRECISION_FUSED,
            &iface, SETUP_INTERFACE);
      if (lowered(&run, __func__)) {
         expect(&run, stored[w]);
         CHECK_EQ(listing_lines(run.listing, "buffer_store_dwordx4", false), 1);
         CHECK(!listing_block(run.listing, "epilogue"));
      }
      run_finish(&run);
      const uint32_t *after[2] = {exp_pos0_done, exp_param0};
      count = program(words, after, 2);
      lower(&run, words, count, AGC_ISA_TARGET_GFX11, AGC_SHADER_STAGE_VERTEX, waves[w], AGC_SHADER_PRECISION_FUSED,
            &iface, SETUP_INTERFACE);
      if (lowered(&run, __func__)) {
         expect(&run, deferred[w]);
         const char *epilogue = listing_block(run.listing, "epilogue");
         CHECK(epilogue != NULL);
         if (epilogue)
            CHECK_EQ(listing_lines(epilogue, "s_cbranch_scc1 -> end", true), 1);
      }
      run_finish(&run);
   }
}

int
main(void)
{
   listing_set_open(&listings, stages_listings, ARRAY_SIZE(stages_listings), "test_isa_lower_stages_listings.h",
                    "stages_listings");
   test_interpolation_loads_parameters_with_every_lane_enabled_on_gfx11();
   test_a_known_prim_mask_hoists_each_parameter_load_to_the_prologue();
   test_prim_mask_copies_and_rewrites_are_followed_through_control_flow();
   test_ordered_pixel_shading_waits_the_way_the_target_does();
   test_a_done_export_before_the_end_of_the_ordered_section_is_refused();
   test_pops_exiting_wave_reads_become_this_wave();
   test_flat_scratch_setup_is_left_to_gfx11_hardware();
   test_other_reads_of_the_scratch_wave_offset_are_rejected_on_gfx11();
   test_wave_offset_uses();
   test_interpolated_channels();
   test_null_exports_get_export_memory();
   test_pixel_plan_targets_and_dual_source();
   test_dual_source_exports_are_swizzled_into_one_pair();
   test_ngg_plans_ring_stores_and_deferral();
   test_ngg_exports_go_to_the_attribute_ring_and_positions_wait_for_them();
   listing_set_close(&listings);
   printf("%d failures\n", agc_test_failures);
   return TEST_RESULT();
}
