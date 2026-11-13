#include "agc_lower.h"

namespace agc {

static const uint32_t counter_enable_bit = 25;
static const uint32_t lod_bias_step = 0x3b800000u;
static const uint32_t abs_mask = 0x7fffffffu;

enum {
   S_EXEC = 0,
   S_SCC = 2,
   S_LANE = 3,
   S_LANE1 = 4,
   S_LANE2 = 5,
   S_COORDS = 6,
   S_WIDTH = 12,
   S_HEIGHT = 13,
   S_TMP = 14,
   S_TRACKED = 15,
   S_PIXELS = 16,
   S_CLAMPED = 17,
   S_MIP = 18,
   S_OFFSET = 19,
   S_ADDRESS = 20,
   S_SAMPLER_BIAS = 22,
   S_MAX_ANISO = 23,
   S_INSTRUCTION_BIAS = 24,
   S_COUNT = 25,
};

enum {
   V_WIDTH = 0,
   V_HEIGHT = 1,
   V_D0 = 2,
   V_D1 = 3,
   V_D2 = 4,
   V_D3 = 5,
   V_OFFSET = 6,
   V_DATA = 7,
   V_COUNT = 8,
};

struct measure {
   lower_ctx &ctx;
   Builder &bld;
   const agc_isa_inst *origin;
   unsigned s;
   unsigned v;
};

static Operand
field(uint32_t offset, uint32_t width)
{
   return lit(offset | (width << 16));
}

static void
footprint(measure &m, unsigned vdst, unsigned before, unsigned after, unsigned extent)
{
   m.bld.vop1(aco_opcode::v_mov_b32, vdef(m.v + vdst), sop(m.s + before));
   m.bld.vop2(aco_opcode::v_sub_f32, vdef(m.v + vdst), sop(m.s + after), vop(m.v + vdst));
   m.bld.vop2(aco_opcode::v_mul_f32, vdef(m.v + vdst), vop(m.v + extent), vop(m.v + vdst));
   m.bld.vop2(aco_opcode::v_and_b32, vdef(m.v + vdst), lit(abs_mask), vop(m.v + vdst));
}

static void
level_of_detail(measure &m, const agc_isa_image *image, unsigned coords, bool instruction_bias)
{
   Builder &bld = m.bld;
   unsigned s = m.s;
   unsigned v = m.v;
   unsigned width = s + S_WIDTH;
   unsigned height = s + S_HEIGHT;
   unsigned scratch = s + S_TMP;
   bld.sop2(aco_opcode::s_bfe_u32, sdef(width), scc_def(), sop(image->srsrc + 1), field(30, 2));
   bld.sop2(aco_opcode::s_bfe_u32, sdef(scratch), scc_def(), sop(image->srsrc + 2), field(0, 12));
   bld.sop2(aco_opcode::s_lshl_b32, sdef(scratch), scc_def(), sop(scratch), imm(2));
   bld.sop2(aco_opcode::s_or_b32, sdef(width), scc_def(), sop(width), sop(scratch));
   bld.sop2(aco_opcode::s_add_u32, sdef(width), scc_def(), sop(width), imm(1));
   bld.sop2(aco_opcode::s_bfe_u32, sdef(height), scc_def(), sop(image->srsrc + 2), field(14, 14));
   bld.sop2(aco_opcode::s_add_u32, sdef(height), scc_def(), sop(height), imm(1));
   bld.vop1(aco_opcode::v_cvt_f32_u32, vdef(v + V_WIDTH), sop(width));
   bld.vop1(aco_opcode::v_cvt_f32_u32, vdef(v + V_HEIGHT), sop(height));
   footprint(m, V_D0, S_COORDS, S_COORDS + 1, V_WIDTH);
   footprint(m, V_D2, S_COORDS, S_COORDS + 2, V_WIDTH);
   if (coords == 2) {
      footprint(m, V_D1, S_COORDS + 3, S_COORDS + 4, V_HEIGHT);
      footprint(m, V_D3, S_COORDS + 3, S_COORDS + 5, V_HEIGHT);
   } else {
      bld.vop1(aco_opcode::v_mov_b32, vdef(v + V_D1), imm(0));
      bld.vop1(aco_opcode::v_mov_b32, vdef(v + V_D3), imm(0));
   }
   bld.vop2(aco_opcode::v_max_f32, vdef(v + V_D0), vop(v + V_D1), vop(v + V_D0));
   bld.vop2(aco_opcode::v_max_f32, vdef(v + V_D2), vop(v + V_D3), vop(v + V_D2));
   bld.vop1(aco_opcode::v_mov_b32, vdef(v + V_D1), vop(v + V_D0));
   bld.vop2(aco_opcode::v_max_f32, vdef(v + V_D1), vop(v + V_D2), vop(v + V_D1));
   bld.vop1(aco_opcode::v_mov_b32, vdef(v + V_D3), vop(v + V_D2));
   bld.vop2(aco_opcode::v_min_f32, vdef(v + V_D3), vop(v + V_D0), vop(v + V_D3));
   bld.vop1(aco_opcode::v_log_f32, vdef(v + V_D1), vop(v + V_D1));
   bld.vop1(aco_opcode::v_log_f32, vdef(v + V_D3), vop(v + V_D3));
   unsigned aniso = s + S_MAX_ANISO;
   unsigned bias = s + S_SAMPLER_BIAS;
   bld.sop2(aco_opcode::s_bfe_u32, sdef(aniso), scc_def(), sop(image->ssamp), field(9, 3));
   bld.vop1(aco_opcode::v_cvt_f32_u32, vdef(v + V_HEIGHT), sop(aniso));
   bld.vop2(aco_opcode::v_sub_f32, vdef(v + V_D1), vop(v + V_D1), vop(v + V_HEIGHT));
   bld.vop2(aco_opcode::v_max_f32, vdef(v + V_D1), vop(v + V_D3), vop(v + V_D1));
   bld.sop2(aco_opcode::s_bfe_i32, sdef(bias), scc_def(), sop(image->ssamp + 2), field(0, 14));
   bld.vop1(aco_opcode::v_cvt_f32_i32, vdef(v + V_WIDTH), sop(bias));
   bld.vop2(aco_opcode::v_mul_f32, vdef(v + V_WIDTH), lit(lod_bias_step), vop(v + V_WIDTH));
   bld.vop2(aco_opcode::v_add_f32, vdef(v + V_D1), vop(v + V_WIDTH), vop(v + V_D1));
   if (instruction_bias)
      bld.vop2(aco_opcode::v_add_f32, vdef(v + V_D1), sop(s + S_INSTRUCTION_BIAS), vop(v + V_D1));
}

void
mip_stats_count(lower_ctx &ctx, const agc_isa_inst *inst, const agc_isa_image *image, uint32_t flags, uint64_t table)
{
   if (image->r128 || image->a16)
      return;
   uint8_t base;
   if (!agc_isa_regalloc_scratch_sgprs(ctx.regs, S_COUNT + 1, &base))
      return;
   unsigned s = (unsigned)((base + 1) & ~1);
   uint8_t v;
   agc_isa_error discarded;
   if (!agc_isa_regalloc_scratch_vgprs(ctx.regs, V_COUNT, &v, &discarded))
      return;
   Builder &bld = ctx.bld;
   unsigned lm = bld.lm.size();
   unsigned coords = 0;
   if (!image->unorm && !(flags & (AGC_ISA_IMG_LOD | AGC_ISA_IMG_LZ | AGC_ISA_IMG_DERIV))) {
      if (image->dim == 0 || image->dim == 4)
         coords = 1;
      else if (image->dim == 1 || image->dim == 5)
         coords = 2;
   }
   unsigned offset_operands = (flags & AGC_ISA_IMG_OFFSET) ? 1 : 0;
   unsigned bias_operands = (flags & AGC_ISA_IMG_BIAS) ? 1 : 0;
   unsigned compare_operands = (flags & AGC_ISA_IMG_COMPARE) ? 1 : 0;
   unsigned first_coord = offset_operands + bias_operands + compare_operands;
   uint32_t done = AGC_ISA_LOCAL_LABEL_BASE + inst->pc;

   measure m = {ctx, bld, inst, s, v};
   save_scc(ctx, s + S_SCC);
   bld.sopc(aco_opcode::s_bitcmp1_b32, scc_def(), sop(image->srsrc + 5), imm(counter_enable_bit));
   branch(ctx, aco_opcode::s_cbranch_scc0, done);
   unsigned saved = s + S_EXEC;
   unsigned lane = s + S_LANE;
   bld.sop1(Builder::s_mov, sdef(saved, lm), mask_op(ctx, exec));
   bld.sop1(Builder::s_ff1_i32, sdef(lane), mask_op(ctx, exec));
   bld.sopc(aco_opcode::s_cmp_lt_i32, scc_def(), sop(lane), imm(0));
   branch(ctx, aco_opcode::s_cbranch_scc1, done);
   bld.sop2(aco_opcode::s_andn2_b32, sdef(lane), scc_def(), sop(lane), imm(3));
   if (coords) {
      bld.sop2(aco_opcode::s_add_u32, sdef(s + S_LANE1), scc_def(), sop(lane), imm(1));
      bld.sop2(aco_opcode::s_add_u32, sdef(s + S_LANE2), scc_def(), sop(lane), imm(2));
      static const uint8_t lanes[3] = {S_LANE, S_LANE1, S_LANE2};
      for (unsigned coord = 0; coord < coords; coord++) {
         unsigned reg = agc_isa_image_address_register(image, first_coord + coord);
         for (unsigned position = 0; position < 3; position++)
            bld.vop3(aco_opcode::v_readlane_b32_e64, sdef(s + S_COORDS + coord * 3 + position), vop(reg),
                     sop(s + lanes[position]));
      }
      if (bias_operands)
         bld.vop3(aco_opcode::v_readlane_b32_e64, sdef(s + S_INSTRUCTION_BIAS),
                  vop(agc_isa_image_address_register(image, offset_operands)), sop(lane));
   }
   bld.sop2(Builder::s_lshl, mask_def(ctx, exec), scc_def(), imm(1), sop(lane));
   if (coords) {
      level_of_detail(m, image, coords, bias_operands != 0);
      bld.vop1(aco_opcode::v_cvt_flr_i32_f32, vdef(v + V_D0), vop(v + V_D1));
      bld.vop2(aco_opcode::v_max_i32, vdef(v + V_D0), imm(0), vop(v + V_D0));
   } else {
      bld.vop1(aco_opcode::v_mov_b32, vdef(v + V_D0), imm(0));
   }
   unsigned mip = s + S_MIP;
   unsigned tracked = s + S_TRACKED;
   unsigned scratch = s + S_TMP;
   unsigned pixels = s + S_PIXELS;
   unsigned clamped = s + S_CLAMPED;
   unsigned offset = s + S_OFFSET;
   bld.vop1(aco_opcode::v_readfirstlane_b32, sdef(mip), vop(v + V_D0));
   bld.sop2(aco_opcode::s_bfe_u32, sdef(tracked), scc_def(), sop(image->srsrc + 1), field(16, 4));
   bld.sop2(aco_opcode::s_bfe_u32, sdef(scratch), scc_def(), sop(image->srsrc + 3), field(12, 4));
   bld.sop2(aco_opcode::s_max_u32, sdef(tracked), scc_def(), sop(tracked), sop(scratch));
   bld.sop1(Builder::s_bcnt1_i32, sdef(pixels), scc_def(), sop(saved, lm));
   bld.sopc(aco_opcode::s_cmp_lt_u32, scc_def(), sop(mip), sop(tracked));
   bld.sop2(aco_opcode::s_cselect_b32, sdef(clamped), sop(pixels), imm(0), scc_op());
   bld.sop2(aco_opcode::s_bfe_u32, sdef(offset), scc_def(), sop(image->srsrc + 6), field(0, 8));
   bld.sop2(aco_opcode::s_lshl_b32, sdef(offset), scc_def(), sop(offset), imm(3));
   bld.sop1(aco_opcode::s_mov_b32, sdef(s + S_ADDRESS), lit((uint32_t)table));
   bld.sop1(aco_opcode::s_mov_b32, sdef(s + S_ADDRESS + 1), lit((uint32_t)(table >> 32)));
   bld.vop1(aco_opcode::v_mov_b32, vdef(v + V_OFFSET), sop(offset));
   bld.vop1(aco_opcode::v_mov_b32, vdef(v + V_DATA), sop(clamped));
   global_access(ctx, inst, AGC_ISA_VMEM_ATOMIC_ADD, v + V_OFFSET, s + S_ADDRESS, v + V_DATA, 0);
   bld.vop2(aco_opcode::v_lshlrev_b32, vdef(v + V_D0), imm(24), vop(v + V_D0));
   bld.vop2(aco_opcode::v_or_b32, vdef(v + V_D0), imm(1), vop(v + V_D0));
   global_access(ctx, inst, AGC_ISA_VMEM_ATOMIC_UMIN, v + V_OFFSET, s + S_ADDRESS, v + V_D0, 4);
   bld.sop1(Builder::s_mov, mask_def(ctx, exec), sop(saved, lm));
   label_next(ctx, done);
   restore_scc(ctx, s + S_SCC);
}

}
