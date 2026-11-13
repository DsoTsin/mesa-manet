#include "agc_lower.h"

namespace agc {

static void
lower_dot2(lower_ctx &ctx, const agc_isa_inst *inst, bool is_signed)
{
   static const uint8_t mads[2][4] = {{6, 0, 1, 4}, {4, 2, 3, 6}};
   if (inst->fields != AGC_ISA_VOP3P)
      fail(ctx, inst, "malformed integer DOT2");
   const agc_isa_vop3p dot = inst->vop3p;
   if (dot.neg != 0 || dot.neg_hi != 0 || (dot.opsel & 4) != 0 || (dot.opsel_hi & 4) == 0)
      fail(ctx, inst, "integer DOT2 requires unmodified full-width accumulator and legal half selectors");
   Builder &bld = ctx.bld;
   unsigned base = scratch_vgprs(ctx, 8);
   aco_opcode bfe = is_signed ? aco_opcode::v_bfe_i32 : aco_opcode::v_bfe_u32;
   for (unsigned half = 0; half < 2; half++) {
      for (unsigned source = 0; source < 2; source++) {
         uint8_t selectors = half == 0 ? dot.opsel : dot.opsel_hi;
         uint32_t offset = (selectors & (1u << source)) != 0 ? 16 : 0;
         bld.vop3(bfe, vdef(base + half * 2 + source), src(inst, dot.src[source]), imm(offset), imm(16));
      }
   }
   bld.vop1(aco_opcode::v_mov_b32, vdef(base + 4), src(inst, dot.src[2]));
   if (is_signed)
      bld.vop2_e64(aco_opcode::v_ashrrev_i32, vdef(base + 5), imm(31), vop(base + 4));
   else
      bld.vop1(aco_opcode::v_mov_b32, vdef(base + 5), imm(0));
   unsigned mask = transient_sgprs(ctx, inst, 2);
   for (unsigned i = 0; i < 2; i++) {
      agc_isa_vop3 mad = agc_isa_vop3_new(agc_isa_operand_vgpr((uint8_t)(base + mads[i][0])),
                                          agc_isa_operand_vgpr((uint8_t)(base + mads[i][1])),
                                          agc_isa_operand_vgpr((uint8_t)(base + mads[i][2])),
                                          agc_isa_operand_vgpr((uint8_t)(base + mads[i][3])));
      mad.sdst = agc_isa_operand_sgpr((uint8_t)mask);
      lower_template(ctx, inst,
                     agc_isa_tmpl_vop3v(is_signed ? AGC_ISA_VOP3_MAD_I64_I32 : AGC_ISA_VOP3_MAD_U64_U32, &mad));
   }
   if (!dot.clamp) {
      bld.vop1(aco_opcode::v_mov_b32, dst(inst, dot.vdst), vop(base + 4));
      return;
   }
   Operand compare = imm(0);
   Operand saturated = imm(0xffffffffu);
   if (is_signed) {
      bld.vop2_e64(aco_opcode::v_ashrrev_i32, vdef(base), imm(31), vop(base + 4));
      bld.vop2_e64(aco_opcode::v_ashrrev_i32, vdef(base + 1), imm(31), vop(base + 5));
      bld.vop2_e64(aco_opcode::v_xor_b32, vdef(base + 1), lit(0x7fffffff), vop(base + 1));
      compare = vop(base);
      saturated = vop(base + 1);
   }
   bld.vopc_e64(aco_opcode::v_cmp_lg_u32, sdef(mask, bld.lm.size()), compare, vop(base + 5));
   bld.vop2_e64(aco_opcode::v_cndmask_b32, dst(inst, dot.vdst), vop(base + 4), saturated,
                sop(mask, bld.lm.size()));
}

static const uint8_t packed_hi_all = 0x7;
static const uint8_t signed_sources = 0x3;
static const unsigned high_vgpr_first = 128;

struct mac {
   agc_isa_operand vdst;
   agc_isa_operand src0;
   agc_isa_operand src1;
   agc_isa_operand src2;
   uint8_t abs;
   uint8_t neg;
   bool clamp;
   uint8_t omod;
};

static agc_isa_operand
vcc_lo()
{
   return agc_isa_operand_special(AGC_ISA_OPERAND_VCC_LO);
}

static uint16_t
range_base(enum agc_isa_family family)
{
   switch (family) {
   case AGC_ISA_VOP1:
      return AGC_ISA_VOP3_VOP1_BASE;
   case AGC_ISA_VOP2:
      return AGC_ISA_VOP3_VOP2_BASE;
   default:
      return 0;
   }
}

static agc_isa_xlat
translation(lower_ctx &ctx, const agc_isa_inst *inst, enum agc_isa_family family, uint16_t op)
{
   const agc_isa_op_info *info = agc_isa_op_lookup(family, op);
   if (!info)
      fail(ctx, inst, "unknown vector opcode");
   return agc_isa_op_translate(info, ctx.target);
}

static bool
high_vgpr(const agc_isa_operand *ops, unsigned count)
{
   for (unsigned i = 0; i < count; i++) {
      if (ops[i].kind == AGC_ISA_OPERAND_VGPR && ops[i].reg >= high_vgpr_first)
         return true;
   }
   return false;
}

static void
emit(lower_ctx &ctx, const agc_isa_inst *inst, enum agc_isa_family family, uint16_t target_op)
{
   uint16_t vop3_op = (uint16_t)(range_base(family) + target_op);
   agc_isa_operand ops[3];
   agc_isa_vop_ext ext;
   switch (inst->fields) {
   case AGC_ISA_VOP3:
      push_retarget(ctx, inst, vop3_op);
      return;
   case AGC_ISA_VOP1:
      ops[0] = inst->vop1.vdst;
      ops[1] = inst->vop1.src0;
      ops[2] = agc_isa_operand_raw(0);
      ext = inst->vop1.ext;
      break;
   case AGC_ISA_VOP2:
      ops[0] = inst->vop2.vdst;
      ops[1] = inst->vop2.src0;
      ops[2] = inst->vop2.vsrc1;
      ext = inst->vop2.ext;
      break;
   default:
      fail(ctx, inst, "malformed vector instruction");
   }
   if (ctx.gfx11_plus) {
      if (ext.kind == AGC_ISA_VOP_EXT_SDWA) {
         sdwa_expand(ctx, inst, vop3_op);
         return;
      }
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_T16) && high_vgpr(ops, 3)) {
         if (ext.kind != AGC_ISA_VOP_EXT_NONE || agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_LITK))
            fail(ctx, inst, "16-bit operands above v127 need a VOP3 form");
         agc_isa_operand src2 = agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CARRY_IN) ? vcc_lo() : agc_isa_operand_raw(0);
         agc_isa_vop3 v = agc_isa_vop3_new(ops[0], ops[1], ops[2], src2);
         if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CARRY_OUT))
            v.sdst = vcc_lo();
         push(ctx, agc_isa_lowering_vop3_inst(inst, vop3_op, inst->flags, &v));
         return;
      }
   }
   push_retarget(ctx, inst, target_op);
}

static mac
mac_operands(lower_ctx &ctx, const agc_isa_inst *inst)
{
   mac m = {};
   if (inst->fields == AGC_ISA_VOP2 && inst->vop2.ext.kind == AGC_ISA_VOP_EXT_NONE) {
      m.vdst = inst->vop2.vdst;
      m.src0 = inst->vop2.src0;
      m.src1 = inst->vop2.vsrc1;
      m.src2 = inst->vop2.has_constant ? agc_isa_operand_literal(inst->vop2.constant) : inst->vop2.vdst;
      return m;
   }
   if (inst->fields == AGC_ISA_VOP3) {
      const agc_isa_vop3 *v = &inst->vop3;
      m.vdst = v->vdst;
      m.src0 = v->src[0];
      m.src1 = v->src[1];
      m.src2 = v->vdst;
      m.abs = v->abs;
      m.neg = v->neg;
      m.clamp = v->clamp;
      m.omod = v->omod;
      return m;
   }
   fail(ctx, inst, "MAC with DPP or SDWA has no fused equivalent");
}

static uint8_t
vop2_ext(const agc_isa_inst *inst)
{
   return inst->fields == AGC_ISA_VOP2 ? inst->vop2.ext.kind : AGC_ISA_VOP_EXT_NONE;
}

static aco_opcode
multiply_op(bool legacy)
{
   return legacy ? aco_opcode::v_mul_legacy_f32 : aco_opcode::v_mul_f32;
}

static void
sdwa_mac(lower_ctx &ctx, const agc_isa_inst *inst, bool legacy)
{
   if (!legacy && ctx.precision == AGC_SHADER_PRECISION_FUSED)
      sdwa_expand_with(ctx, inst, {sdwa_core::fused, 0, aco_opcode::v_fma_f32, false});
   else
      sdwa_expand_with(ctx, inst, {sdwa_core::unfused, 0, multiply_op(legacy), legacy});
}

static void
dpp_mac(lower_ctx &ctx, const agc_isa_inst *inst, bool legacy)
{
   if (inst->fields != AGC_ISA_VOP2)
      fail(ctx, inst, "malformed MAC");
   Builder &bld = ctx.bld;
   const agc_isa_vop2 guest = inst->vop2;
   unsigned v = scratch_vgprs(ctx, 3);
   unsigned product = v;
   unsigned flag = v + 1;
   unsigned one = v + 2;
   unsigned s = transient_sgprs(ctx, inst, 5);
   agc_isa_vop_ext lane_ext = guest.ext;
   if (lane_ext.kind == AGC_ISA_VOP_EXT_DPP16) {
      lane_ext.dpp16.src0_neg = false;
      lane_ext.dpp16.src0_abs = false;
      lane_ext.dpp16.src1_neg = false;
      lane_ext.dpp16.src1_abs = false;
   }
   bld.vop1(aco_opcode::v_mov_b32, vdef(flag), imm(2));
   bld.vop1(aco_opcode::v_mov_b32, vdef(one), imm(1));
   agc_isa_inst lanes = agc_isa_tmpl_vop1(AGC_ISA_VOP1_MOV_B32, agc_isa_operand_vgpr((uint8_t)flag),
                                          agc_isa_operand_vgpr((uint8_t)one));
   lanes.vop1.ext = lane_ext;
   lower_template(ctx, inst, lanes);
   agc_isa_mode_restore restore = {};
   if (legacy)
      enter_f32_flush(ctx, inst, &restore);
   agc_isa_inst multiply = agc_isa_tmpl_vop2(legacy ? AGC_ISA_VOP2_MUL_LEGACY_F32 : AGC_ISA_VOP2_MUL_F32,
                                             agc_isa_operand_vgpr((uint8_t)product), guest.src0, guest.vsrc1);
   multiply.vop2.ext = guest.ext;
   lower_template(ctx, inst, multiply);
   unsigned lm = bld.lm.size();
   save_scc(ctx, s + 4);
   bld.sop1(Builder::s_mov, sdef(s + 2, lm), mask_op(ctx, vcc));
   bld.vopc(aco_opcode::v_cmp_lg_u32, mask_def(ctx, vcc), imm(2), vop(flag));
   bld.sop1(Builder::s_and_saveexec, sdef(s, lm), scc_def(), mask_def(ctx, exec), mask_op(ctx, vcc),
            mask_op(ctx, exec));
   bld.vop2(aco_opcode::v_add_f32, dst(inst, guest.vdst), vop(product), src(inst, guest.vdst));
   if (legacy)
      leave_f32_flush(ctx, restore);
   bld.sop1(Builder::s_mov, mask_def(ctx, exec), sop(s, lm));
   bld.sop1(Builder::s_mov, mask_def(ctx, vcc), sop(s + 2, lm));
   restore_scc(ctx, s + 4);
}

static void
emit_fused_vop2(lower_ctx &ctx, const agc_isa_inst *inst, uint16_t vop2)
{
   uint16_t op = vop2;
   const agc_isa_op_info *info = agc_isa_op_lookup(AGC_ISA_VOP2, vop2);
   if (info) {
      agc_isa_xlat xlat = agc_isa_op_translate(info, ctx.target);
      if (xlat.kind == AGC_ISA_XLAT_OP)
         op = xlat.value;
   }
   if (inst->fields == AGC_ISA_VOP3) {
      push_retarget(ctx, inst, (uint16_t)(AGC_ISA_VOP3_VOP2_BASE + op));
      return;
   }
   if (inst->fields == AGC_ISA_VOP2 && inst->vop2.ext.kind == AGC_ISA_VOP_EXT_SDWA)
      fail(ctx, inst, "SDWA MAC has no fused equivalent");
   push_retarget(ctx, inst, op);
}

static void
fused_mad(lower_ctx &ctx, const agc_isa_inst *inst, const mac &m)
{
   modifiers(ctx.bld.vop3(aco_opcode::v_fma_f32, dst(inst, m.vdst), src(inst, m.src0), src(inst, m.src1),
                          src(inst, m.src2)),
             m.abs, m.neg, m.clamp, m.omod);
}

static void
unfused_mad(lower_ctx &ctx, const agc_isa_inst *inst, bool legacy, const mac &m)
{
   Builder &bld = ctx.bld;
   unsigned tmp = scratch_vgprs(ctx, 1);
   modifiers(bld.vop2_e64(multiply_op(legacy), vdef(tmp), src(inst, m.src0), src(inst, m.src1)), m.abs & 0x3,
             m.neg & 0x3, false, 0);
   modifiers(bld.vop2_e64(aco_opcode::v_add_f32, dst(inst, m.vdst), vop(tmp), src(inst, m.src2)),
             (m.abs >> 1) & 0x2, (m.neg >> 1) & 0x2, m.clamp, m.omod);
}

static void
legacy_mad(lower_ctx &ctx, const agc_isa_inst *inst, const mac &m)
{
   agc_isa_mode_restore restore;
   enter_f32_flush(ctx, inst, &restore);
   unfused_mad(ctx, inst, true, m);
   leave_f32_flush(ctx, restore);
}

static void
dot_c(lower_ctx &ctx, const agc_isa_inst *inst, bool dot2)
{
   agc_isa_operand vdst, src0, src1;
   uint8_t neg = 0;
   bool clamp = false;
   if (inst->fields == AGC_ISA_VOP2 && inst->vop2.ext.kind == AGC_ISA_VOP_EXT_NONE) {
      vdst = inst->vop2.vdst;
      src0 = inst->vop2.src0;
      src1 = inst->vop2.vsrc1;
   } else if (inst->fields == AGC_ISA_VOP3 && inst->vop3.abs == 0 && inst->vop3.omod == 0 && inst->vop3.opsel == 0) {
      vdst = inst->vop3.vdst;
      src0 = inst->vop3.src[0];
      src1 = inst->vop3.src[1];
      neg = inst->vop3.neg;
      clamp = inst->vop3.clamp;
   } else {
      fail(ctx, inst, "dot product modifiers have no VOP3P form");
   }
   if (!dot2 && neg != 0)
      fail(ctx, inst, "negated integer dot sources");
   Instruction *dot = ctx.bld.vop3p(dot2 ? aco_opcode::v_dot2_f32_f16 : aco_opcode::v_dot4_i32_iu8, dst(inst, vdst),
                                    src(inst, src0), src(inst, src1), src(inst, vdst), 0, packed_hi_all);
   dot->valu().neg_lo = dot2 ? neg & 0x3 : signed_sources;
   dot->valu().neg_hi = dot2 ? neg & 0x3 : 0;
   dot->valu().clamp = clamp;
}

static void
alu_rule(lower_ctx &ctx, const agc_isa_inst *inst, agc_isa_xlat xlat)
{
   bool exact = ctx.precision == AGC_SHADER_PRECISION_EXACT;
   uint8_t ext = vop2_ext(inst);
   bool dpp = ext == AGC_ISA_VOP_EXT_DPP16 || ext == AGC_ISA_VOP_EXT_DPP8;
   switch (xlat.rule) {
   case AGC_ISA_RULE_MAC_LEGACY:
      if (ext == AGC_ISA_VOP_EXT_SDWA)
         sdwa_mac(ctx, inst, true);
      else if (dpp)
         dpp_mac(ctx, inst, true);
      else
         legacy_mad(ctx, inst, mac_operands(ctx, inst));
      return;
   case AGC_ISA_RULE_MAC:
      if (ext == AGC_ISA_VOP_EXT_SDWA)
         sdwa_mac(ctx, inst, false);
      else if (dpp && exact)
         dpp_mac(ctx, inst, false);
      else if (exact)
         unfused_mad(ctx, inst, false, mac_operands(ctx, inst));
      else
         emit_fused_vop2(ctx, inst, AGC_ISA_VOP2_FMAC_F32);
      return;
   case AGC_ISA_RULE_MADMK:
   case AGC_ISA_RULE_MADAK: {
      bool madmk = xlat.rule == AGC_ISA_RULE_MADMK;
      if (inst->fields != AGC_ISA_VOP2 || !inst->vop2.has_constant || inst->vop2.ext.kind != AGC_ISA_VOP_EXT_NONE)
         fail(ctx, inst, "malformed inline-constant MAD");
      if (!exact) {
         emit_fused_vop2(ctx, inst, madmk ? AGC_ISA_VOP2_FMAMK_F32 : AGC_ISA_VOP2_FMAAK_F32);
         return;
      }
      agc_isa_operand constant = agc_isa_operand_literal(inst->vop2.constant);
      mac m = {};
      m.vdst = inst->vop2.vdst;
      m.src0 = inst->vop2.src0;
      m.src1 = madmk ? constant : inst->vop2.vsrc1;
      m.src2 = madmk ? inst->vop2.vsrc1 : constant;
      unfused_mad(ctx, inst, false, m);
      return;
   }
   case AGC_ISA_RULE_DOT2C:
   case AGC_ISA_RULE_DOT4C:
      dot_c(ctx, inst, xlat.rule == AGC_ISA_RULE_DOT2C);
      return;
   default:
      fail(ctx, inst, "no vector lowering for this rule");
   }
}

static void
alu(lower_ctx &ctx, const agc_isa_inst *inst, enum agc_isa_family family, uint16_t op)
{
   agc_isa_xlat xlat = translation(ctx, inst, family, op);
   switch (xlat.kind) {
   case AGC_ISA_XLAT_OP:
      emit(ctx, inst, family, xlat.value);
      return;
   case AGC_ISA_XLAT_DROP:
   case AGC_ISA_XLAT_SAME:
      return;
   case AGC_ISA_XLAT_NO:
      fail(ctx, inst, "instruction was removed on the target");
   default:
      alu_rule(ctx, inst, xlat);
   }
}

static bool
needs_mad64_scratch(enum agc_isa_target target, const agc_isa_inst *inst)
{
   static const uint16_t widths[3] = {1, 1, 2};
   if (inst->fields != AGC_ISA_VOP3 || inst->vop3.vdst.kind != AGC_ISA_OPERAND_VGPR)
      return false;
   uint16_t start = (uint16_t)inst->vop3.vdst.reg;
   uint16_t end = (uint16_t)(start + 2);
   bool overlaps = false;
   for (unsigned i = 0; i < 3; i++) {
      agc_isa_operand source = inst->vop3.src[i];
      if (source.kind != AGC_ISA_OPERAND_VGPR)
         continue;
      uint16_t first = (uint16_t)source.reg;
      overlaps |= first < end && start < first + widths[i];
   }
   return target == AGC_ISA_TARGET_GFX11 &&
          (inst->opcode == AGC_ISA_VOP3_MAD_U64_U32 || inst->opcode == AGC_ISA_VOP3_MAD_I64_I32) && overlaps;
}

static void
mad64_through_scratch(lower_ctx &ctx, const agc_isa_inst *inst, uint16_t op)
{
   if (inst->fields != AGC_ISA_VOP3 || inst->vop3.vdst.kind != AGC_ISA_OPERAND_VGPR)
      fail(ctx, inst, "malformed 64-bit MAD");
   unsigned result = inst->vop3.vdst.reg;
   unsigned tmp = scratch_vgprs(ctx, 2);
   agc_isa_inst mad = agc_isa_inst_retarget(inst, op);
   mad.vop3.vdst = agc_isa_operand_vgpr((uint8_t)tmp);
   push(ctx, mad);
   for (unsigned k = 0; k < 2; k++)
      ctx.bld.vop1(aco_opcode::v_mov_b32, vdef((uint8_t)(result + k)), vop(tmp + k));
}

static void
vop3_only_op(lower_ctx &ctx, const agc_isa_inst *inst)
{
   agc_isa_xlat xlat = translation(ctx, inst, AGC_ISA_VOP3, inst->opcode);
   switch (xlat.kind) {
   case AGC_ISA_XLAT_OP:
      if (needs_mad64_scratch(ctx.target, inst))
         mad64_through_scratch(ctx, inst, xlat.value);
      else
         push_retarget(ctx, inst, xlat.value);
      return;
   case AGC_ISA_XLAT_DROP:
   case AGC_ISA_XLAT_SAME:
      return;
   case AGC_ISA_XLAT_NO:
      fail(ctx, inst, "instruction was removed on the target");
   default:
      break;
   }
   if (xlat.rule == AGC_ISA_RULE_INTERP) {
      lower_interp_vop3(ctx, inst);
      return;
   }
   if (xlat.rule != AGC_ISA_RULE_MAD && xlat.rule != AGC_ISA_RULE_MAD_LEGACY)
      fail(ctx, inst, "no VOP3 lowering for this rule");
   if (inst->fields != AGC_ISA_VOP3)
      fail(ctx, inst, "malformed MAD");
   const agc_isa_vop3 *v = &inst->vop3;
   const mac m = {v->vdst, v->src[0], v->src[1], v->src[2], v->abs, v->neg, v->clamp, v->omod};
   if (xlat.rule == AGC_ISA_RULE_MAD_LEGACY)
      legacy_mad(ctx, inst, m);
   else if (ctx.precision == AGC_SHADER_PRECISION_EXACT)
      unfused_mad(ctx, inst, false, m);
   else
      fused_mad(ctx, inst, m);
}

static void
constant_compare(lower_ctx &ctx, const agc_isa_inst *inst, bool exec_result, bool value)
{
   agc_isa_operand result;
   if (exec_result)
      result = agc_isa_operand_special(AGC_ISA_OPERAND_EXEC_LO);
   else if (inst->fields == AGC_ISA_VOPC)
      result = inst->vopc.sdst;
   else if (inst->fields == AGC_ISA_VOP3)
      result = inst->vop3.vdst;
   else
      fail(ctx, inst, "malformed compare");
   if (exec_result && value)
      return;
   Builder &bld = ctx.bld;
   bld.sop1(Builder::s_mov, dst(inst, result, bld.lm.size()), value ? mask_op(ctx, exec) : imm(0));
}

static void
compare(lower_ctx &ctx, const agc_isa_inst *inst, uint16_t op)
{
   const agc_isa_vopc_info *info = agc_isa_vopc_decode(op);
   if (!info)
      fail(ctx, inst, "unknown compare");
   bool value;
   if (ctx.gfx12 && agc_isa_vopc_is_constant_result(info, &value)) {
      constant_compare(ctx, inst, info->exec, value);
      return;
   }
   uint16_t target_op = ctx.gfx11_plus ? info->gfx11_opcode : op;
   if (inst->fields == AGC_ISA_VOPC && ctx.gfx11_plus) {
      const agc_isa_vopc *c = &inst->vopc;
      if (c->ext.kind == AGC_ISA_VOP_EXT_SDWA) {
         sdwa_expand(ctx, inst, target_op);
         return;
      }
      const agc_isa_operand ops[2] = {c->src0, c->vsrc1};
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_T16) && high_vgpr(ops, 2)) {
         if (c->ext.kind != AGC_ISA_VOP_EXT_NONE)
            fail(ctx, inst, "16-bit DPP compare above v127");
         agc_isa_vop3 v = agc_isa_vop3_new(c->sdst, c->src0, c->vsrc1, agc_isa_operand_raw(0));
         push(ctx, agc_isa_lowering_vop3_inst(inst, target_op, inst->flags, &v));
         return;
      }
   }
   push_retarget(ctx, inst, target_op);
}

static void
packed(lower_ctx &ctx, const agc_isa_inst *inst)
{
   agc_isa_xlat xlat = translation(ctx, inst, AGC_ISA_VOP3P, inst->opcode);
   switch (xlat.kind) {
   case AGC_ISA_XLAT_OP:
      push_retarget(ctx, inst, xlat.value);
      return;
   case AGC_ISA_XLAT_DROP:
   case AGC_ISA_XLAT_SAME:
      return;
   case AGC_ISA_XLAT_NO:
      fail(ctx, inst, "packed instruction was removed on the target");
   default:
      break;
   }
   if (xlat.rule == AGC_ISA_RULE_DOT2) {
      lower_dot2(ctx, inst, xlat.value != 0);
      return;
   }
   if (xlat.rule != AGC_ISA_RULE_DOT_SIGNED)
      fail(ctx, inst, "no VOP3P lowering for this rule");
   if (inst->fields != AGC_ISA_VOP3P)
      fail(ctx, inst, "malformed packed dot");
   if (inst->vop3p.neg != 0 || inst->vop3p.neg_hi != 0)
      fail(ctx, inst, "negated integer dot sources");
   agc_isa_inst out = agc_isa_inst_derive(inst, AGC_ISA_VOP3P, xlat.value, inst->flags);
   out.vop3p = inst->vop3p;
   out.vop3p.neg = signed_sources;
   push(ctx, out);
}

void
lower_vector(lower_ctx &ctx, const agc_isa_inst *inst)
{
   switch (inst->family) {
   case AGC_ISA_VOP3P:
      packed(ctx, inst);
      return;
   case AGC_ISA_VOPC:
      compare(ctx, inst, inst->opcode);
      return;
   case AGC_ISA_VOP3: {
      enum agc_isa_family family;
      uint16_t base;
      if (!agc_isa_vop3_base(inst->opcode, &family, &base))
         vop3_only_op(ctx, inst);
      else if (family == AGC_ISA_VOPC)
         compare(ctx, inst, base);
      else
         alu(ctx, inst, family, base);
      return;
   }
   default:
      alu(ctx, inst, (enum agc_isa_family)inst->family, inst->opcode);
   }
}

}
