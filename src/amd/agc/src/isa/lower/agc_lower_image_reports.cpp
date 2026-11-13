#include <string.h>

#include "agc_lower.h"

namespace agc {

enum {
   S_KNOWN = 0,
   S_ADDRESS = 2,
   S_HASH = 4,
   S_OFFSET = 5,
};

enum {
   V_OFFSET = 0,
   V_DATA = 1,
   V_COUNT = 2,
};

static const uint32_t known_bucket_shift = 13;
static const uint32_t done_label = AGC_ISA_LOCAL_LABEL_BASE + 2;

static Operand
report_word(const agc_isa_image_report *report, unsigned k)
{
   return k < report->dwords ? sop(report->source + k) : imm(0);
}

static void
fold(lower_ctx &ctx, const agc_isa_image_report *report, unsigned sdst)
{
   ctx.bld.sop2(aco_opcode::s_xor_b32, sdef(sdst), scc_def(), report_word(report, 1), report_word(report, 2));
   unsigned end = report->dwords < 6 ? report->dwords : 6;
   for (unsigned k = 3; k < end; k++)
      ctx.bld.sop2(aco_opcode::s_xor_b32, sdef(sdst), scc_def(), sop(sdst), report_word(report, k));
}

static void
store(lower_ctx &ctx, const agc_isa_image_report *report, const agc_isa_inst *origin, Operand value, uint32_t offset)
{
   unsigned data = report->v + V_DATA;
   ctx.bld.vop1(aco_opcode::v_mov_b32, vdef(data), value);
   global_access(ctx, origin, AGC_ISA_VMEM_STORE_DWORD, report->v + V_OFFSET, report->s + S_ADDRESS, data,
                 (int32_t)offset);
}

bool
image_report_new(lower_ctx &ctx, uint64_t table, uint8_t source, bool r128, uint8_t sgprs, bool stored,
                 agc_isa_image_report *report)
{
   agc_isa_error discarded;
   uint8_t v;
   if (!agc_isa_regalloc_scratch_vgprs(ctx.regs, V_COUNT, &v, &discarded))
      return false;
   memset(report, 0, sizeof(*report));
   report->table = table;
   report->source = source;
   report->dwords = r128 ? 4 : 8;
   report->s = sgprs;
   report->v = v;
   report->stored = stored;
   return true;
}

void
image_report_begin(lower_ctx &ctx, const agc_isa_image_report *report)
{
   Builder &bld = ctx.bld;
   unsigned hash = report->s + S_HASH;
   unsigned offset = report->s + S_OFFSET;
   fold(ctx, report, hash);
   bld.sop2(aco_opcode::s_mul_i32, sdef(offset), report_word(report, 0), lit(AGC_ISA_IMAGE_REPORTS_SPREAD));
   bld.sop2(aco_opcode::s_xor_b32, sdef(hash), scc_def(), sop(hash), sop(offset));
   bld.sop2(aco_opcode::s_mul_i32, sdef(hash), sop(hash), lit(AGC_ISA_IMAGE_REPORTS_SPREAD));
   if (report->stored)
      bld.sop2(aco_opcode::s_xor_b32, sdef(hash), scc_def(), sop(hash), lit(AGC_ISA_IMAGE_REPORTS_STORE_SALT));
   bld.sop2(aco_opcode::s_lshr_b32, sdef(offset), scc_def(), sop(hash), imm(32 - known_bucket_shift));
   bld.sop2(aco_opcode::s_lshl_b32, sdef(offset), scc_def(), sop(offset), imm(3));
   bld.sop1(aco_opcode::s_mov_b32, sdef(report->s + S_ADDRESS), lit((uint32_t)report->table));
   bld.sop1(aco_opcode::s_mov_b32, sdef(report->s + S_ADDRESS + 1), lit((uint32_t)(report->table >> 32)));
   smem_load(ctx, 2, sdef(report->s + S_KNOWN, 2), report->s + S_ADDRESS, sop(offset));
}

void
image_report_finish(lower_ctx &ctx, const agc_isa_image_report *report, const agc_isa_inst *origin)
{
   Builder &bld = ctx.bld;
   unsigned lm = bld.lm.size();
   uint32_t done = done_label + origin->pc;
   unsigned first = report->s + S_KNOWN;
   unsigned second = report->s + S_KNOWN + 1;
   unsigned hash = report->s + S_HASH;
   unsigned offset = report->s + S_OFFSET;
   bld.sop2(aco_opcode::s_xor_b32, sdef(first), scc_def(), sop(first), sop(hash));
   bld.sop2(aco_opcode::s_xor_b32, sdef(second), scc_def(), sop(second), sop(hash));
   bld.sop2(aco_opcode::s_min_u32, sdef(first), scc_def(), sop(first), sop(second));
   bld.sopc(aco_opcode::s_cmp_eq_u32, scc_def(), sop(first), imm(0));
   branch(ctx, aco_opcode::s_cbranch_scc1, done);
   bld.sop1(Builder::s_ff1_i32, sdef(offset), mask_op(ctx, exec));
   bld.sopc(aco_opcode::s_cmp_lt_i32, scc_def(), sop(offset), imm(0));
   branch(ctx, aco_opcode::s_cbranch_scc1, done);
   bld.sop1(Builder::s_mov, sdef(first, lm), mask_op(ctx, exec));
   bld.sop2(Builder::s_lshl, mask_def(ctx, exec), scc_def(), imm(1), sop(offset));
   bld.sop2(aco_opcode::s_mul_i32, sdef(offset), report_word(report, 0), lit(AGC_ISA_IMAGE_REPORTS_SPREAD));
   bld.sop2(aco_opcode::s_lshr_b32, sdef(offset), scc_def(), sop(offset), imm(14));
   bld.sop2(aco_opcode::s_and_b32, sdef(offset), scc_def(), sop(offset),
            lit((AGC_ISA_IMAGE_REPORTS_REPORT_SLOTS - 1) << 6));
   if (report->stored)
      bld.sop2(aco_opcode::s_xor_b32, sdef(offset), scc_def(), sop(offset), lit(AGC_ISA_IMAGE_REPORTS_STORE_SLOT));
   bld.sop2(aco_opcode::s_add_u32, sdef(offset), scc_def(), sop(offset), lit(AGC_ISA_IMAGE_REPORTS_REPORTS_OFFSET));
   bld.vop1(aco_opcode::v_mov_b32, vdef(report->v + V_OFFSET), sop(offset));
   for (unsigned k = 0; k < 8; k++)
      store(ctx, report, origin, report_word(report, k), k * 4);
   fold(ctx, report, hash);
   bld.sop2(aco_opcode::s_xor_b32, sdef(hash), scc_def(), sop(hash),
            lit(report->stored ? AGC_ISA_IMAGE_REPORTS_STORE_CHECK : AGC_ISA_IMAGE_REPORTS_CHECK));
   store(ctx, report, origin, sop(hash), AGC_ISA_IMAGE_REPORTS_REPORT_CHECK_OFFSET);
   bld.sop1(Builder::s_mov, mask_def(ctx, exec), sop(first, lm));
   label(ctx, done);
}

}
