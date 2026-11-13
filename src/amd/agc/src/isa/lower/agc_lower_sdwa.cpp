#include <string.h>

#include "agc_lower.h"

namespace agc {

static const unsigned scratch_vgpr_count = 3;
static const unsigned result_slot = 2;
static const uint32_t sign_bit = 0x80000000u;

static bool
select_bits(uint8_t sel, bool *whole, uint32_t *offset, uint32_t *width)
{
   *whole = false;
   *offset = 0;
   *width = 0;
   if (sel <= 3) {
      *offset = 8u * sel;
      *width = 8;
   } else if (sel == 4) {
      *width = 16;
   } else if (sel == AGC_ISA_SDWA_SEL_WORD_1) {
      *offset = 16;
      *width = 16;
   } else if (sel == AGC_ISA_SDWA_SEL_DWORD) {
      *whole = true;
   } else {
      return false;
   }
   return true;
}

static bool
ends_with(const char *name, const char *suffix)
{
   size_t length = strlen(name);
   size_t tail = strlen(suffix);
   return length >= tail && !strcmp(name + length - tail, suffix);
}

static bool
vop3_takes_modifiers(const agc_isa_inst *inst)
{
   char scratch[AGC_ISA_MNEMONIC_SIZE];
   const char *name = agc_isa_inst_mnemonic(inst, scratch);
   return !strcmp(name, "v_cndmask_b32") || ends_with(name, "_f16") || ends_with(name, "_f32") ||
          ends_with(name, "_f64");
}

static void
change_sign(lower_ctx &ctx, const agc_isa_inst *inst, agc_isa_operand to, agc_isa_operand from, bool neg, bool abs)
{
   aco_opcode op = neg && abs ? aco_opcode::v_or_b32 : neg ? aco_opcode::v_xor_b32 : aco_opcode::v_and_b32;
   Operand mask = lit(neg ? sign_bit : ~sign_bit);
   Definition result = dst(inst, to);
   Operand value = src(inst, from);
   if (agc_isa_operand_is_vgpr(from))
      ctx.bld.vop2(op, result, mask, value);
   else
      ctx.bld.vop2_e64(op, result, mask, value);
}

static agc_isa_operand
accumulator(lower_ctx &ctx, const agc_isa_inst *inst, agc_isa_operand dest, bool bits, uint32_t offset, uint32_t width)
{
   if (!bits)
      return dest;
   unsigned slot = scratch_vgprs(ctx, scratch_vgpr_count + 1) + scratch_vgpr_count;
   ctx.bld.vop3(aco_opcode::v_bfe_u32, vdef(slot), src(inst, dest), imm(offset), imm(width));
   return agc_isa_operand_vgpr((uint8_t)slot);
}

static void
merge(lower_ctx &ctx, const agc_isa_inst *inst, const agc_isa_sdwa &sdwa, agc_isa_operand dest, unsigned result,
      uint32_t offset, uint32_t width)
{
   Builder &bld = ctx.bld;
   switch (sdwa.dst_unused) {
   case AGC_ISA_SDWA_UNUSED_PRESERVE: {
      if (offset > 0)
         bld.vop2(aco_opcode::v_lshlrev_b32, vdef(result), imm(offset), vop(result));
      uint32_t mask = (uint32_t)((UINT64_C(1) << width) - 1) << offset;
      bld.vop3(aco_opcode::v_bfi_b32, dst(inst, dest), imm(mask), vop(result), src(inst, dest));
      return;
   }
   case AGC_ISA_SDWA_UNUSED_PAD:
   case AGC_ISA_SDWA_UNUSED_SEXT: {
      aco_opcode extract = sdwa.dst_unused == AGC_ISA_SDWA_UNUSED_SEXT ? aco_opcode::v_bfe_i32 : aco_opcode::v_bfe_u32;
      bld.vop3(extract, offset == 0 ? dst(inst, dest) : vdef(result), vop(result), imm(0), imm(width));
      if (offset > 0)
         bld.vop2(aco_opcode::v_lshlrev_b32, dst(inst, dest), imm(offset), vop(result));
      return;
   }
   default:
      fail(ctx, inst, "invalid SDWA unused-bits policy");
   }
}

static agc_isa_inst
guest_core(const agc_isa_inst *inst, uint16_t opcode, agc_isa_operand vdst, const agc_isa_operand sources[3],
           uint8_t abs, uint8_t neg, bool clamp, uint8_t omod)
{
   agc_isa_vop3 v = agc_isa_vop3_new(vdst, sources[0], sources[1], sources[2]);
   if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CARRY_OUT))
      v.sdst = agc_isa_operand_special(AGC_ISA_OPERAND_VCC_LO);
   v.abs = abs;
   v.neg = neg;
   v.clamp = clamp;
   v.omod = omod;
   uint64_t flags = inst->flags & ~(AGC_ISA_FLAG_T16 | AGC_ISA_FLAG_READS_DST);
   return agc_isa_lowering_vop3_inst(inst, opcode, flags, &v);
}

void
sdwa_expand(lower_ctx &ctx, const agc_isa_inst *inst, uint16_t vop3_op)
{
   sdwa_core core = {sdwa_core::guest, vop3_op, aco_opcode::num_opcodes, false};
   if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_READS_DST)) {
      core.kind = sdwa_core::fused;
      if (inst->family == AGC_ISA_VOP2 && inst->opcode == AGC_ISA_VOP2_FMAC_F32)
         core.op = aco_opcode::v_fma_f32;
      else if (inst->family == AGC_ISA_VOP2 && inst->opcode == AGC_ISA_VOP2_FMAC_F16)
         core.op = aco_opcode::v_fma_f16;
      else
         fail(ctx, inst, "SDWA accumulation has no gfx11 equivalent");
   }
   sdwa_expand_with(ctx, inst, core);
}

void
sdwa_expand_with(lower_ctx &ctx, const agc_isa_inst *inst, const sdwa_core &core)
{
   agc_isa_operand dest;
   agc_isa_operand sources[2];
   unsigned source_count = 2;
   agc_isa_sdwa sdwa;
   bool compare = false;
   if (inst->fields == AGC_ISA_VOP1 && inst->vop1.ext.kind == AGC_ISA_VOP_EXT_SDWA) {
      dest = inst->vop1.vdst;
      sources[0] = inst->vop1.src0;
      source_count = 1;
      sdwa = inst->vop1.ext.sdwa;
   } else if (inst->fields == AGC_ISA_VOP2 && inst->vop2.ext.kind == AGC_ISA_VOP_EXT_SDWA) {
      dest = inst->vop2.vdst;
      sources[0] = inst->vop2.src0;
      sources[1] = inst->vop2.vsrc1;
      sdwa = inst->vop2.ext.sdwa;
   } else if (inst->fields == AGC_ISA_VOPC && inst->vopc.ext.kind == AGC_ISA_VOP_EXT_SDWA) {
      dest = inst->vopc.sdst;
      sources[0] = inst->vopc.src0;
      sources[1] = inst->vopc.vsrc1;
      sdwa = inst->vopc.ext.sdwa;
      compare = true;
   } else {
      fail(ctx, inst, "malformed SDWA instruction");
   }
   bool takes_modifiers = vop3_takes_modifiers(inst);
   bool whole_move = inst->family == AGC_ISA_VOP1 && inst->opcode == AGC_ISA_VOP1_MOV_B32 &&
                     sdwa.src0_sel == AGC_ISA_SDWA_SEL_DWORD && sdwa.dst_sel == AGC_ISA_SDWA_SEL_DWORD &&
                     !sdwa.clamp && sdwa.omod == 0;
   if (whole_move && (sdwa.src0_neg || sdwa.src0_abs)) {
      change_sign(ctx, inst, dest, sources[0], sdwa.src0_neg, sdwa.src0_abs);
      return;
   }
   Builder &bld = ctx.bld;
   unsigned scratch = scratch_vgprs(ctx, scratch_vgpr_count);
   const uint8_t sels[2] = {sdwa.src0_sel, sdwa.src1_sel};
   const bool sexts[2] = {sdwa.src0_sext, sdwa.src1_sext};
   const bool negs[2] = {sdwa.src0_neg, sdwa.src1_neg};
   const bool abss[2] = {sdwa.src0_abs, sdwa.src1_abs};
   const agc_isa_operand unused = agc_isa_operand_raw(0);
   agc_isa_operand operands[3] = {unused, unused, unused};
   uint8_t neg = 0;
   uint8_t abs = 0;
   for (unsigned i = 0; i < source_count; i++) {
      bool whole;
      uint32_t offset, width;
      if (!select_bits(sels[i], &whole, &offset, &width))
         fail(ctx, inst, "invalid SDWA source select");
      agc_isa_operand tmp = agc_isa_operand_vgpr((uint8_t)(scratch + i));
      if (whole) {
         operands[i] = sources[i];
      } else {
         bld.vop3(sexts[i] ? aco_opcode::v_bfe_i32 : aco_opcode::v_bfe_u32, vdef(scratch + i),
                  src(inst, sources[i]), imm(offset), imm(width));
         operands[i] = tmp;
      }
      if (takes_modifiers) {
         neg |= (uint8_t)(negs[i] << i);
         abs |= (uint8_t)(abss[i] << i);
      } else if (negs[i] || abss[i]) {
         change_sign(ctx, inst, tmp, operands[i], negs[i], abss[i]);
         operands[i] = tmp;
      }
   }
   if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CARRY_IN))
      operands[2] = agc_isa_operand_special(AGC_ISA_OPERAND_VCC_LO);
   bool dst_bits = false;
   uint32_t dst_offset = 0, dst_width = 0;
   if (!compare) {
      bool whole;
      if (!select_bits(sdwa.dst_sel, &whole, &dst_offset, &dst_width))
         fail(ctx, inst, "invalid SDWA destination select");
      dst_bits = !whole;
   }
   unsigned result = scratch + result_slot;
   agc_isa_operand vdst = dst_bits ? agc_isa_operand_vgpr((uint8_t)result) : dest;
   switch (core.kind) {
   case sdwa_core::guest:
      push(ctx, guest_core(inst, core.vop3, vdst, operands, abs, neg, sdwa.clamp, sdwa.omod));
      break;
   case sdwa_core::fused: {
      agc_isa_operand sum = accumulator(ctx, inst, dest, dst_bits, dst_offset, dst_width);
      modifiers(bld.vop3(core.op, dst(inst, vdst), src(inst, operands[0]), src(inst, operands[1]),
                         src(inst, sum)),
                abs, neg, sdwa.clamp, sdwa.omod);
      break;
   }
   default: {
      agc_isa_operand sum = accumulator(ctx, inst, dest, dst_bits, dst_offset, dst_width);
      unsigned product = scratch_vgprs(ctx, scratch_vgpr_count + 2) + scratch_vgpr_count + 1;
      agc_isa_mode_restore restore = {};
      if (core.legacy)
         enter_f32_flush(ctx, inst, &restore);
      modifiers(bld.vop2_e64(core.op, vdef(product), src(inst, operands[0]), src(inst, operands[1])), abs,
                neg, false, 0);
      modifiers(bld.vop2_e64(aco_opcode::v_add_f32, dst(inst, vdst), vop(product), src(inst, sum)), 0, 0,
                sdwa.clamp, sdwa.omod);
      if (core.legacy)
         leave_f32_flush(ctx, restore);
      break;
   }
   }
   if (dst_bits)
      merge(ctx, inst, sdwa, dest, result, dst_offset, dst_width);
}

}
