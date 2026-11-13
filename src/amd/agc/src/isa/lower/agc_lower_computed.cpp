#include "agc_lower.h"

namespace agc {

static const unsigned second_half_sgpr = 6;

static agc_isa_operand
high_part(agc_isa_operand op, uint8_t k)
{
   if (k == 0)
      return op;
   switch (op.kind) {
   case AGC_ISA_OPERAND_SGPR:
      return agc_isa_operand_sgpr((uint8_t)(op.reg + k));
   case AGC_ISA_OPERAND_TTMP:
      return agc_isa_operand_ttmp((uint8_t)(op.reg + k));
   case AGC_ISA_OPERAND_VCC_LO:
      return agc_isa_operand_special(AGC_ISA_OPERAND_VCC_HI);
   case AGC_ISA_OPERAND_EXEC_LO:
      return agc_isa_operand_special(AGC_ISA_OPERAND_EXEC_HI);
   default:
      return agc_isa_operand_special(AGC_ISA_OPERAND_NULL);
   }
}

static uint64_t
return_address(const lower_ctx &ctx, const agc_isa_inst *inst)
{
   return agc_isa_code_layout_guest_address(&ctx.layout, (uint32_t)(inst->pc + 4u));
}

static void
write_address(lower_ctx &ctx, const agc_isa_inst *origin, agc_isa_operand sdst, uint64_t address)
{
   const uint32_t halves[2] = {(uint32_t)address, (uint32_t)(address >> 32)};
   for (uint8_t k = 0; k < 2; k++)
      ctx.bld.sop1(aco_opcode::s_mov_b32, dst(origin, high_part(sdst, k)), imm(halves[k]));
}

static void
getpc(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_SOP1)
      fail(ctx, inst, "malformed s_getpc_b64");
   if (!ctx.has_guest_base)
      fail(ctx, inst, "the guest code address is unknown (ShaderInterface::guest_code_address)");
   write_address(ctx, inst, inst->sop1.sdst, return_address(ctx, inst));
}

static void
static_call(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_SOPK || !inst->has_branch_target)
      fail(ctx, inst, "malformed s_call_b64");
   uint32_t callee = inst->branch_target;
   write_address(ctx, inst, inst->sopk.sdst, return_address(ctx, inst));
   lower_branch(ctx, callee);
}

static void
computed_jump(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_SOP1)
      fail(ctx, inst, "malformed computed jump");
   agc_isa_operand sdst = inst->sop1.sdst;
   agc_isa_operand src0 = inst->sop1.src0;
   uint32_t destination;
   const agc_isa_jump_target *jump = agc_isa_computed_flow_jump(&ctx.ir->flow, inst->pc);
   if (jump && jump->kind == AGC_ISA_JUMP_STATIC) {
      destination = jump->pc;
   } else {
      if (!ctx.has_dispatch)
         fail(ctx, inst, "computed jumps need SGPRs above the shader's own");
      lower_template(ctx, inst, agc_isa_tmpl_sop1(AGC_ISA_SOP1_MOV_B64, agc_isa_operand_sgpr(ctx.dispatch), src0));
      destination = AGC_ISA_DISPATCH_LABEL;
   }
   if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CALL))
      write_address(ctx, inst, sdst, return_address(ctx, inst));
   lower_branch(ctx, destination);
}

bool
lower_pc_relative(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->family == AGC_ISA_SOP1 && inst->opcode == AGC_ISA_SOP1_GETPC_B64) {
      getpc(ctx, inst);
      return true;
   }
   if (inst->family == AGC_ISA_SOP1 &&
       (inst->opcode == AGC_ISA_SOP1_SETPC_B64 || inst->opcode == AGC_ISA_SOP1_SWAPPC_B64)) {
      computed_jump(ctx, inst);
      return true;
   }
   if (inst->family == AGC_ISA_SOPK && inst->opcode == AGC_ISA_SOPK_CALL_B64) {
      static_call(ctx, inst);
      return true;
   }
   return false;
}

void
set_second_half(lower_ctx &ctx, const agc_isa_inst *origin, agc_isa_mask128 live_in)
{
   bool live = agc_isa_mask128_test(live_in, second_half_sgpr) || agc_isa_mask128_test(live_in, second_half_sgpr + 1);
   if (!ctx.layout.has_second_half || !live)
      return;
   write_address(ctx, origin, agc_isa_operand_sgpr(second_half_sgpr), ctx.layout.second_half);
}

void
lower_branch(lower_ctx &ctx, uint32_t to)
{
   branch(ctx, aco_opcode::s_branch, to);
}

static void
narrow_dispatch_stub(lower_ctx &ctx, unsigned x, uint64_t base, Operand n)
{
   Builder &bld = ctx.bld;
   bld.sop2(aco_opcode::s_cselect_b32, sdef(x + 2), imm(1), imm(0), scc_op());
   bld.sop2(aco_opcode::s_sub_u32, sdef(x), scc_def(), sop(x), imm((uint32_t)base));
   bld.sop2(aco_opcode::s_subb_u32, sdef(x + 1), scc_def(), sop(x + 1), imm((uint32_t)(base >> 32)), scc_op());
   bld.sopc(aco_opcode::s_cmp_eq_u32, scc_def(), sop(x + 1), imm(0));
   bld.sop2(aco_opcode::s_cselect_b32, sdef(x), sop(x), n, scc_op());
   bld.sop2(aco_opcode::s_and_b32, sdef(x + 1), scc_def(), sop(x), imm(3));
   bld.sop2(aco_opcode::s_cselect_b32, sdef(x), n, sop(x), scc_op());
   bld.sop2(aco_opcode::s_min_u32, sdef(x), scc_def(), sop(x), n);
   bld.sop1(aco_opcode::s_mov_b32, sdef(x + 1), sop(x + 2));
   bld.sop1(aco_opcode::s_getpc_b64, sdef(x + 2, 2));
   smem_load(ctx, 1, sdef(x), x + 2, sop(x));
   waitcnt(ctx, 63, 7, 0);
   bld.sop2(aco_opcode::s_or_b32, sdef(x + 2), scc_def(), sop(x + 2), sop(x + 1));
   bld.sop2(aco_opcode::s_ashr_i32, sdef(x + 1), scc_def(), sop(x), imm(31));
   bld.sop2(aco_opcode::s_add_u32, sdef(x + 2), scc_def(), sop(x + 2), sop(x));
   bld.sop2(aco_opcode::s_addc_u32, sdef(x + 3), scc_def(), sop(x + 3), sop(x + 1), scc_op());
   bld.sopc(aco_opcode::s_bitcmp1_b32, scc_def(), sop(x + 2), imm(0));
   bld.sop1(aco_opcode::s_bitset0_b32, sdef(x + 2), imm(0), sop(x + 2));
   setpc(ctx, x + 2);
}

void
dispatch_stub(lower_ctx &ctx, const agc_isa_inst *origin, uint32_t code_bytes)
{
   if (!ctx.has_dispatch)
      fail(ctx, origin, "the dispatch SGPRs are allocated");
   Builder &bld = ctx.bld;
   unsigned x = ctx.dispatch;
   uint64_t base = ctx.has_guest_base ? ctx.guest_base : 0;
   Operand n = imm(code_bytes);
   if (ctx.layout.region_count == 0) {
      narrow_dispatch_stub(ctx, x, base, n);
      return;
   }
   bld.sop1(aco_opcode::s_getpc_b64, sdef(x + 4, 2));
   bld.sop2(aco_opcode::s_cselect_b32, sdef(x + 6), imm(1), imm(0), scc_op());
   bld.sop1(aco_opcode::s_mov_b32, sdef(x + 2), n);
   for (size_t span = 0; span <= ctx.layout.region_count; span++) {
      uint64_t address = base;
      uint32_t start = 0;
      uint32_t bytes = ctx.layout.own_bytes;
      if (span > 0) {
         const agc_isa_linked_region *region = &ctx.layout.regions[span - 1];
         address = region->address;
         start = region->start;
         bytes = region->bytes;
      }
      bld.sop2(aco_opcode::s_sub_u32, sdef(x + 3), scc_def(), sop(x), imm((uint32_t)address));
      bld.sop2(aco_opcode::s_add_u32, sdef(x + 7), scc_def(), sop(x + 3), imm(start));
      bld.sopc(aco_opcode::s_cmp_lt_u32, scc_def(), sop(x + 3), imm(bytes));
      bld.sop2(aco_opcode::s_cselect_b32, sdef(x + 7), sop(x + 7), sop(x + 2), scc_op());
      bld.sopc(aco_opcode::s_cmp_eq_u32, scc_def(), sop(x + 1), imm((uint32_t)(address >> 32)));
      bld.sop2(aco_opcode::s_cselect_b32, sdef(x + 2), sop(x + 7), sop(x + 2), scc_op());
   }
   bld.sop2(aco_opcode::s_and_b32, sdef(x + 3), scc_def(), sop(x + 2), imm(3));
   bld.sop2(aco_opcode::s_cselect_b32, sdef(x + 2), n, sop(x + 2), scc_op());
   smem_load(ctx, 1, sdef(x + 3), x + 4, sop(x + 2));
   waitcnt(ctx, 63, 7, 0);
   bld.sop2(aco_opcode::s_ashr_i32, sdef(x + 7), scc_def(), sop(x + 3), imm(31));
   bld.sop2(aco_opcode::s_add_u32, sdef(x + 4), scc_def(), sop(x + 4), sop(x + 3));
   bld.sop2(aco_opcode::s_addc_u32, sdef(x + 5), scc_def(), sop(x + 5), sop(x + 7), scc_op());
   bld.sopc(aco_opcode::s_cmp_lg_u32, scc_def(), sop(x + 6), imm(0));
   setpc(ctx, x + 4);
}

}
