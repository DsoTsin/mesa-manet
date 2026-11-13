#include "agc_lower.h"

namespace agc {

static const uint16_t hwreg_mode_f32_denorm = 1 | (4 << 6) | (1 << 11);
static const uint16_t host_image_atomic_or = 0x13;

struct sequence {
   lower_ctx &ctx;
   Builder &bld;
   unsigned s;
   unsigned v;
};

static void
seq_mov(sequence &seq, unsigned to, Operand from)
{
   seq.bld.vop1(aco_opcode::v_mov_b32, vdef(seq.v + to), from);
}

static void
seq_valu(sequence &seq, aco_opcode op, unsigned to, Operand src0, unsigned src1)
{
   seq.bld.vop2(op, vdef(seq.v + to), src0, vop(seq.v + src1));
}

static void
seq_cmp(sequence &seq, aco_opcode op, Operand src0, unsigned src1)
{
   seq.bld.vopc(op, mask_def(seq.ctx, vcc), src0, vop(seq.v + src1));
}

static void
seq_select(sequence &seq, unsigned to, unsigned no, unsigned yes)
{
   seq.bld.vop2(aco_opcode::v_cndmask_b32, vdef(seq.v + to), vop(seq.v + no), vop(seq.v + yes),
                mask_op(seq.ctx, vcc));
}

static void
seq_key(sequence &seq, unsigned raw, unsigned key)
{
   seq_valu(seq, aco_opcode::v_and_b32, key, lit(0x7fffffffu), raw);
   seq_valu(seq, aco_opcode::v_and_b32, 6, lit(0x80000000u), raw);
   seq_cmp(seq, aco_opcode::v_cmp_gt_u32, lit(0x00800000u), key);
   seq.bld.sop2(Builder::s_and, mask_def(seq.ctx, vcc), scc_def(), mask_op(seq.ctx, vcc),
                sop(seq.s + 6, seq.bld.lm.size()));
   seq_select(seq, key, raw, 6);
   seq_valu(seq, aco_opcode::v_ashrrev_i32, 6, imm(31), key);
   seq_valu(seq, aco_opcode::v_or_b32, 6, lit(0x80000000u), 6);
   seq_valu(seq, aco_opcode::v_xor_b32, key, vop(seq.v + 6), key);
}

static void
seq_nan(sequence &seq, unsigned raw, unsigned other)
{
   seq_valu(seq, aco_opcode::v_and_b32, 6, lit(0x7fffffffu), raw);
   seq_cmp(seq, aco_opcode::v_cmp_lt_u32, lit(0x7f800000u), 6);
   seq_select(seq, 1, 1, other);
}

static void
seq_signaling_nan(sequence &seq, unsigned raw)
{
   seq_valu(seq, aco_opcode::v_and_b32, 6, lit(0x7fffffffu), raw);
   seq_valu(seq, aco_opcode::v_add_u32, 6, lit(0x807fffffu), 6);
   seq_cmp(seq, aco_opcode::v_cmp_gt_u32, lit(0x003fffffu), 6);
   seq_valu(seq, aco_opcode::v_or_b32, 6, lit(0x00400000u), raw);
   seq_select(seq, 1, 1, 6);
}

static void
wait_all_vm(lower_ctx &ctx)
{
   waitcnt(ctx, 0, AGC_ISA_GFX10_EXPCNT_MAX, AGC_ISA_GFX10_LGKMCNT_MAX);
}

void
image_atomic(lower_ctx &ctx, const agc_isa_inst *origin, const agc_isa_inst *image)
{
   if (origin->fields != AGC_ISA_MIMG)
      fail(ctx, origin, "malformed float image atomic");
   const agc_isa_image *guest = &origin->mimg;
   if (guest->dmask != 1 || guest->data_dwords != 1 || guest->d16 || guest->tfe || guest->lwe)
      fail(ctx, origin, "float image atomic emulation requires a 32-bit operand without D16, TFE or LWE");
   Builder &bld = ctx.bld;
   unsigned s = transient_sgprs(ctx, origin, 8);
   unsigned v = scratch_vgprs(ctx, 7);
   sequence seq = {ctx, bld, s, v};
   bld.sop1(aco_opcode::s_mov_b64, sdef(s, 2), Operand(exec, s2));
   bld.sop1(aco_opcode::s_mov_b64, sdef(s + 2, 2), Operand(vcc, s2));
   save_scc(ctx, s + 4);
   bld.sopk(aco_opcode::s_getreg_b32, sdef(s + 5), hwreg_mode_f32_denorm);
   bld.sop2(aco_opcode::s_and_b32, sdef(s + 5), scc_def(), imm(1), sop(s + 5));
   bld.sop2(aco_opcode::s_cselect_b64, sdef(s + 6, 2), imm(0), imm(0xffffffffu), scc_op());
   seq_mov(seq, 0, vop(guest->vdata));
   seq_mov(seq, 2, imm(0));
   uint32_t loop = AGC_ISA_LOCAL_LABEL_BASE + origin->pc;
   label(ctx, loop);
   seq_key(seq, 0, 4);
   seq_key(seq, 2, 5);
   seq_cmp(seq, origin->opcode == AGC_ISA_MIMG_ATOMIC_FMAX ? aco_opcode::v_cmp_gt_u32 : aco_opcode::v_cmp_lt_u32,
           vop(v + 4), 5);
   seq_select(seq, 1, 2, 0);
   seq_nan(seq, 2, 0);
   seq_nan(seq, 0, 2);
   seq_signaling_nan(seq, 2);
   seq_signaling_nan(seq, 0);
   if (image->fields != AGC_ISA_MIMG)
      fail(ctx, origin, "malformed float image atomic");
   agc_isa_inst cas = *image;
   cas.mimg.vdata = (uint8_t)(v + 1);
   cas.mimg.dmask = 3;
   cas.mimg.data_dwords = 2;
   cas.mimg.glc = true;
   cas.flags |= AGC_ISA_FLAG_CMPSWAP;
   push(ctx, cas);
   wait_all_vm(ctx);
   seq_mov(seq, 3, vop(v + 1));
   seq_cmp(seq, aco_opcode::v_cmp_eq_u32, vop(v + 1), 2);
   seq_mov(seq, 2, vop(v + 1));
   exec_andn2(ctx, mask_op(ctx, vcc));
   branch(ctx, aco_opcode::s_cbranch_execnz, loop);
   bld.sop1(aco_opcode::s_mov_b64, Definition(exec, s2), sop(s, 2));
   if (guest->glc)
      bld.vop1(aco_opcode::v_mov_b32, vdef(guest->vdata), vop(v + 3));
   bld.sop1(aco_opcode::s_mov_b64, Definition(vcc, s2), sop(s + 2, 2));
   restore_scc(ctx, s + 4);
}

void
image_fcmpswap(lower_ctx &ctx, const agc_isa_inst *origin, const agc_isa_inst *image)
{
   if (origin->fields != AGC_ISA_MIMG)
      fail(ctx, origin, "malformed float image atomic");
   const agc_isa_image *guest = &origin->mimg;
   if (guest->dmask != 3 || guest->data_dwords != 2 || guest->d16 || guest->tfe || guest->lwe)
      fail(ctx, origin, "float image compare-and-swap emulation requires a 32-bit operand pair without "
                        "D16, TFE or LWE");
   Builder &bld = ctx.bld;
   unsigned lm = bld.lm.size();
   unsigned s = transient_sgprs(ctx, origin, 5);
   unsigned v = scratch_vgprs(ctx, 3);
   unsigned cmp = guest->vdata + 1;
   sequence seq = {ctx, bld, s, v};
   bld.sop1(Builder::s_mov, sdef(s, lm), mask_op(ctx, exec));
   bld.sop1(Builder::s_mov, sdef(s + 2, lm), mask_op(ctx, vcc));
   save_scc(ctx, s + 4);
   seq_mov(seq, 2, imm(0));
   agc_isa_inst read = *image;
   read.opcode = host_image_atomic_or;
   read.flags &= ~AGC_ISA_FLAG_CMPSWAP;
   if (read.fields == AGC_ISA_MIMG) {
      read.mimg.vdata = (uint8_t)(v + 2);
      read.mimg.dmask = 1;
      read.mimg.data_dwords = 1;
      read.mimg.glc = true;
   }
   push(ctx, read);
   wait_all_vm(ctx);
   seq_mov(seq, 1, vop(v + 2));
   uint32_t loop_label = AGC_ISA_LOCAL_LABEL_BASE + origin->pc + 1;
   uint32_t done_label = AGC_ISA_LOCAL_LABEL_BASE + origin->pc + 3;
   label(ctx, loop_label);
   bld.vopc(aco_opcode::v_cmp_eq_f32, mask_def(ctx, vcc), vop(v + 1), vop(cmp));
   exec_and(ctx, mask_op(ctx, vcc));
   branch(ctx, aco_opcode::s_cbranch_execz, done_label);
   seq_mov(seq, 0, vop(guest->vdata));
   agc_isa_inst swap = *image;
   if (swap.fields == AGC_ISA_MIMG) {
      swap.mimg.vdata = (uint8_t)v;
      swap.mimg.dmask = 3;
      swap.mimg.data_dwords = 2;
      swap.mimg.glc = true;
   }
   swap.flags |= AGC_ISA_FLAG_CMPSWAP;
   push(ctx, swap);
   wait_all_vm(ctx);
   seq_cmp(seq, aco_opcode::v_cmp_eq_u32, vop(v), 1);
   seq_mov(seq, 1, vop(v));
   exec_andn2(ctx, mask_op(ctx, vcc));
   branch(ctx, aco_opcode::s_cbranch_execnz, loop_label);
   label(ctx, done_label);
   bld.sop1(Builder::s_mov, mask_def(ctx, exec), sop(s, lm));
   bld.sop1(Builder::s_mov, mask_def(ctx, vcc), sop(s + 2, lm));
   restore_scc(ctx, s + 4);
   if (guest->glc)
      bld.vop1(aco_opcode::v_mov_b32, vdef(guest->vdata), vop(v + 1));
}

}
