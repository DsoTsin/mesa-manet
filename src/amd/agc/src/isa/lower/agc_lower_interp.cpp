#include <string.h>

#include "agc_lower.h"

namespace agc {

static const uint16_t p10_lane = 1;
static const uint16_t attr_mask = 0x3f;
static const unsigned chan_shift = 6;
static const uint16_t high_half = 1u << 8;
static const uint8_t no_exp_wait = 7;
static const uint8_t opsel_src2_hi = 1u << 2;
static const uint8_t opsel_dst_hi = 1u << 3;

enum kind {
   KIND_P1,
   KIND_P2,
   KIND_MOV,
   KIND_P1LL_F16,
   KIND_P1LV_F16,
   KIND_P2_F16
};

struct request {
   enum kind kind;
   uint8_t vdst;
   agc_isa_operand coord;
   agc_isa_operand addend;
   uint8_t attr;
   uint8_t chan;
   bool high;
   uint8_t neg;
   uint8_t abs;
   uint8_t omod;
   uint8_t opsel;
   bool clamp;
};

static const uint16_t param_lane[3] = {1, 2, 0};
static const uint8_t omod_factor[4] = {242, 244, 246, 240};

void
quad_broadcast(lower_ctx &ctx, const agc_isa_inst *origin, uint8_t vdst, agc_isa_operand source, uint16_t lane)
{
   uint16_t ctrl = (uint16_t)(lane | (lane << 2) | (lane << 4) | (lane << 6));
   ctx.bld.vop1_dpp(aco_opcode::v_mov_b32, vdef(vdst), src(origin, source), ctrl, 0xf, 0xf, false, true);
}

void
wait_for_parameters(lower_ctx &ctx)
{
   if (ctx.gfx12)
      ctx.bld.sopp(aco_opcode::s_wait_expcnt, 0);
   else
      ctx.bld.sopp(aco_opcode::s_waitcnt,
                   agc_isa_waitcnt_encode_gfx11(AGC_ISA_GFX10_VMCNT_MAX, 0, AGC_ISA_GFX10_LGKMCNT_MAX));
}

static agc_isa_operand
clear_sign(lower_ctx &ctx, const agc_isa_inst *inst, uint8_t to, agc_isa_operand from, uint32_t mask)
{
   ctx.bld.vop2_e64(aco_opcode::v_and_b32, vdef(to), imm(mask), src(inst, from));
   return agc_isa_operand_vgpr(to);
}

static void
vinterp(lower_ctx &ctx, const agc_isa_inst *inst, const request &req, aco_opcode op, Operand p, agc_isa_operand coord,
        agc_isa_operand addend, uint8_t opsel, bool scale, uint8_t wait_exp)
{
   Instruction *instr =
      ctx.bld.vinterp_inreg(op, vdef(req.vdst), p, src(inst, coord), src(inst, addend), opsel, wait_exp)
         .instr;
   instr->vinterp_inreg().neg = req.neg;
   instr->vinterp_inreg().clamp = req.clamp && !scale;
}

static void
emit(lower_ctx &ctx, const agc_isa_inst *inst, const request &req)
{
   Builder &bld = ctx.bld;
   uint8_t hoisted = 0;
   bool has_hoisted = ctx.pixel && agc_isa_pixel_plan_hoisted(ctx.pixel, inst->pc, req.attr, req.chan, &hoisted);
   uint8_t scratch = scratch_vgprs(ctx, 2);
   uint8_t param;
   uint8_t wait_exp;
   if (has_hoisted) {
      param = hoisted;
      wait_exp = no_exp_wait;
   } else {
      uint8_t saved = transient_sgprs(ctx, inst, 2);
      enable_all_lanes(ctx, saved);
      param_load(ctx, scratch, req.attr, req.chan);
      restore_lanes(ctx, saved);
      param = scratch;
      wait_exp = 0;
   }
   agc_isa_operand coord = req.coord;
   if ((req.abs & 0x2) && req.kind != KIND_MOV) {
      uint8_t base = scratch_vgprs(ctx, 3);
      coord = clear_sign(ctx, inst, (uint8_t)(base + 2), req.coord, 0x7fffffffu);
   }
   agc_isa_operand addend = req.addend;
   if (req.abs & 0x4) {
      uint8_t base = scratch_vgprs(ctx, 4);
      uint32_t mask = req.kind == KIND_P1LV_F16 ? 0x7fff7fffu : 0x7fffffffu;
      addend = clear_sign(ctx, inst, (uint8_t)(base + 3), req.addend, mask);
   }
   bool scale = req.omod != 0 || (req.kind == KIND_MOV && req.clamp);
   agc_isa_operand p = agc_isa_operand_vgpr(param);
   uint8_t high = req.high ? 1 : 0;
   switch (req.kind) {
   case KIND_P1:
      vinterp(ctx, inst, req, aco_opcode::v_interp_p10_f32_inreg, vop(param), coord, p, 0, scale, wait_exp);
      break;
   case KIND_P2:
      vinterp(ctx, inst, req, aco_opcode::v_interp_p2_f32_inreg, vop(param), coord, agc_isa_operand_vgpr(req.vdst), 0,
              scale, wait_exp);
      break;
   case KIND_P1LL_F16:
      vinterp(ctx, inst, req, aco_opcode::v_interp_p10_f16_f32_inreg, vop(param), coord, p, (uint8_t)(high * 0x5),
              scale, wait_exp);
      break;
   case KIND_P2_F16:
      vinterp(ctx, inst, req, aco_opcode::v_interp_p2_f16_f32_inreg, vop(param), coord, addend,
              (uint8_t)(high | (req.opsel & opsel_dst_hi)), scale, wait_exp);
      break;
   case KIND_MOV: {
      if (req.coord.kind != AGC_ISA_OPERAND_RAW)
         fail(ctx, inst, "malformed interpolation parameter");
      uint32_t slot = (uint16_t)req.coord.raw;
      if (slot >= sizeof(param_lane) / sizeof(param_lane[0]))
         fail(ctx, inst, "invalid interpolation parameter");
      if (wait_exp != no_exp_wait)
         wait_for_parameters(ctx);
      quad_broadcast(ctx, inst, req.vdst, p, param_lane[slot]);
      break;
   }
   case KIND_P1LV_F16: {
      if (wait_exp != no_exp_wait)
         wait_for_parameters(ctx);
      uint8_t p10 = (uint8_t)(scratch + 1);
      quad_broadcast(ctx, inst, p10, p, p10_lane);
      Instruction *mix = bld.vop3p(aco_opcode::v_fma_mix_f32, vdef(req.vdst), vop(p10), src(inst, coord),
                                   src(inst, addend), (uint8_t)(high | (req.opsel & opsel_src2_hi)), 0x5)
                            .instr;
      mix->valu().neg_lo = req.neg;
      mix->valu().neg_hi = 0;
      mix->valu().clamp = req.clamp;
      break;
   }
   }
   if (scale) {
      uint32_t factor = 0;
      agc_isa_inline_float_bits(omod_factor[req.omod & 3], &factor);
      Instruction *multiply = bld.vop2_e64(aco_opcode::v_mul_f32, vdef(req.vdst), vop(req.vdst), imm(factor)).instr;
      multiply->valu().clamp = req.clamp;
   }
}

void
lower_interp(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (!ctx.gfx11_plus) {
      push_retarget(ctx, inst, inst->opcode);
      return;
   }
   if (inst->fields != AGC_ISA_VINTRP)
      fail(ctx, inst, "malformed interpolation");
   const agc_isa_interp *i = &inst->vintrp;
   request req;
   memset(&req, 0, sizeof(req));
   switch (inst->opcode) {
   case AGC_ISA_VINTRP_P1:
      req.kind = KIND_P1;
      req.coord = agc_isa_operand_vgpr(i->vsrc);
      break;
   case AGC_ISA_VINTRP_P2:
      req.kind = KIND_P2;
      req.coord = agc_isa_operand_vgpr(i->vsrc);
      break;
   case AGC_ISA_VINTRP_MOV:
      req.kind = KIND_MOV;
      req.coord = agc_isa_operand_raw(i->vsrc);
      break;
   default:
      fail(ctx, inst, "unknown interpolation opcode");
   }
   req.vdst = i->vdst;
   req.addend = agc_isa_operand_special(AGC_ISA_OPERAND_NULL);
   req.attr = i->attr;
   req.chan = i->chan;
   emit(ctx, inst, req);
}

void
lower_interp_vop3(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_VOP3)
      fail(ctx, inst, "malformed interpolation");
   const agc_isa_vop3 *v = &inst->vop3;
   if (v->src[0].kind != AGC_ISA_OPERAND_RAW)
      fail(ctx, inst, "malformed interpolation attribute");
   uint16_t select = (uint16_t)v->src[0].raw;
   if (v->vdst.kind != AGC_ISA_OPERAND_VGPR)
      fail(ctx, inst, "malformed interpolation destination");
   enum kind kind;
   switch (inst->opcode) {
   case AGC_ISA_VOP3_INTERP_P1LL_F16:
      kind = KIND_P1LL_F16;
      break;
   case AGC_ISA_VOP3_INTERP_P1LV_F16:
      kind = KIND_P1LV_F16;
      break;
   case AGC_ISA_VOP3_INTERP_P2_F16:
      kind = KIND_P2_F16;
      break;
   default:
      switch ((uint16_t)(inst->opcode - AGC_ISA_VOP3_INTERP_P1_F32)) {
      case AGC_ISA_VINTRP_P1:
         kind = KIND_P1;
         break;
      case AGC_ISA_VINTRP_P2:
         kind = KIND_P2;
         break;
      default:
         kind = KIND_MOV;
         break;
      }
      break;
   }
   bool f16 = kind == KIND_P1LL_F16 || kind == KIND_P1LV_F16 || kind == KIND_P2_F16;
   uint8_t opsel_used = 0;
   if (kind == KIND_P1LV_F16)
      opsel_used = opsel_src2_hi;
   else if (kind == KIND_P2_F16)
      opsel_used = opsel_dst_hi;
   if ((f16 && v->omod != 0) || (v->opsel & (uint8_t)~opsel_used) != 0 || (!f16 && (select & high_half) != 0))
      fail(ctx, inst, "interpolation modifiers have no VINTERP form");
   uint8_t neg_used = kind == KIND_P1LV_F16 || kind == KIND_P2_F16 ? 0x6 : 0x2;
   request req;
   memset(&req, 0, sizeof(req));
   req.kind = kind;
   req.vdst = (uint8_t)v->vdst.reg;
   req.coord = v->src[1];
   req.addend = v->src[2];
   req.attr = (uint8_t)(select & attr_mask);
   req.chan = (uint8_t)((select >> chan_shift) & 3);
   req.high = (select & high_half) != 0;
   req.neg = (uint8_t)(v->neg & neg_used);
   req.abs = (uint8_t)(v->abs & neg_used);
   req.omod = v->omod;
   req.opsel = v->opsel;
   req.clamp = v->clamp;
   emit(ctx, inst, req);
}

}
