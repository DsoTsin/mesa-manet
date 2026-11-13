#include "agc_lower.h"

namespace agc {

static const uint8_t src2_base = 0x80;
static const uint8_t src2_offset_in_address = 0x80;
static const uint32_t src2_address_bits = 0x1ffffu;

void
lds_access(lower_ctx &ctx, const agc_isa_inst *origin, uint16_t opcode, uint8_t addr, uint8_t vdst, uint8_t data0,
           uint8_t data1, uint8_t offset0, uint8_t offset1)
{
   const struct agc_isa_ds_op *info = agc_isa_ds_op(opcode);
   if (!info)
      fail(ctx, origin, "unknown DS opcode");
   agc_isa_inst inst = agc_isa_tmpl(AGC_ISA_DS, opcode);
   inst.ds.vdst = vdst;
   inst.ds.addr = addr;
   inst.ds.data0 = data0;
   inst.ds.data1 = data1;
   inst.ds.offset0 = offset0;
   inst.ds.offset1 = offset1;
   inst.ds.gds = false;
   inst.ds.dst_dwords = info->dst;
   inst.ds.data0_dwords = info->data0;
   inst.ds.data1_dwords = info->data1;
   inst.ds.uses_addr = true;
   inst.ds.writes_addr = false;
   lower_template(ctx, origin, inst);
}

void
lds_compare_and_swap(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_DS)
      fail(ctx, inst, "malformed DS");
   const agc_isa_ds d = inst->ds;
   bool wrap = false;
   bool wide = false;
   bool rtn = true;
   switch (inst->opcode) {
   case AGC_ISA_DS_CMPST_F32:
      rtn = false;
      break;
   case AGC_ISA_DS_CMPST_RTN_F32:
      break;
   case AGC_ISA_DS_CMPST_F64:
      wide = true;
      rtn = false;
      break;
   case AGC_ISA_DS_CMPST_RTN_F64:
      wide = true;
      break;
   case AGC_ISA_DS_WRAP_RTN_B32:
      wrap = true;
      break;
   default:
      fail(ctx, inst, "no compare-and-swap form");
   }
   Builder &bld = ctx.bld;
   unsigned dwords = wide ? 2 : 1;
   uint8_t s = transient_sgprs(ctx, inst, 5);
   uint8_t v = scratch_vgprs(ctx, (uint8_t)(3 * dwords));
   uint8_t old = v;
   uint8_t ret = (uint8_t)(v + dwords);
   uint8_t new_value = (uint8_t)(v + 2 * dwords);
   save_masks(ctx, s);
   lds_access(ctx, inst, wide ? AGC_ISA_DS_READ_B64 : AGC_ISA_DS_READ_B32, d.addr, old, 0, 0, d.offset0, d.offset1);
   waitcnt(ctx, AGC_ISA_GFX10_VMCNT_MAX, AGC_ISA_GFX10_EXPCNT_MAX, 0);
   uint32_t loop_label = AGC_ISA_LOCAL_LABEL_BASE + inst->pc + 1;
   uint32_t done_label = AGC_ISA_LOCAL_LABEL_BASE + inst->pc + 3;
   label(ctx, loop_label);
   uint8_t store;
   if (!wrap) {
      bld.vopc(wide ? aco_opcode::v_cmp_eq_f64 : aco_opcode::v_cmp_eq_f32, mask_def(ctx, vcc), vop(old, dwords),
               vop(d.data0, dwords));
      exec_and(ctx, mask_op(ctx, vcc));
      branch(ctx, aco_opcode::s_cbranch_execz, done_label);
      store = d.data1;
   } else {
      bld.vopc(aco_opcode::v_cmp_ge_u32, mask_def(ctx, vcc), vop(old), vop(d.data0));
      bld.vop2(aco_opcode::v_sub_u32, vdef(ret), vop(old), vop(d.data0));
      bld.vop2(aco_opcode::v_add_u32, vdef(new_value), vop(old), vop(d.data1));
      bld.vop2(aco_opcode::v_cndmask_b32, vdef(new_value), vop(new_value), vop(ret), mask_op(ctx, vcc));
      store = new_value;
   }
   lds_access(ctx, inst, wide ? AGC_ISA_DS_CMPST_RTN_B64 : AGC_ISA_DS_CMPST_RTN_B32, d.addr, ret, old, store,
              d.offset0, d.offset1);
   waitcnt(ctx, AGC_ISA_GFX10_VMCNT_MAX, AGC_ISA_GFX10_EXPCNT_MAX, 0);
   bld.vopc(wide ? aco_opcode::v_cmp_eq_u64 : aco_opcode::v_cmp_eq_u32, mask_def(ctx, vcc), vop(ret, dwords),
            vop(old, dwords));
   for (unsigned k = 0; k < dwords; k++)
      bld.vop1(aco_opcode::v_mov_b32, vdef(old + k), vop(ret + k));
   exec_andn2(ctx, mask_op(ctx, vcc));
   branch(ctx, aco_opcode::s_cbranch_execnz, loop_label);
   label(ctx, done_label);
   restore_masks(ctx, s);
   if (rtn) {
      for (unsigned k = 0; k < dwords; k++)
         bld.vop1(aco_opcode::v_mov_b32, vdef(d.vdst + k), vop(old + k));
   }
}

void
lds_src2(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_DS)
      fail(ctx, inst, "malformed DS");
   const agc_isa_ds d = inst->ds;
   Builder &bld = ctx.bld;
   uint16_t plain = inst->opcode == AGC_ISA_DS_WRITE_SRC2_B32 ? (uint16_t)AGC_ISA_DS_WRITE_B32
                                                               : (uint16_t)(inst->opcode - src2_base);
   uint8_t v = scratch_vgprs(ctx, 3);
   uint8_t a = v;
   uint8_t b = (uint8_t)(v + 1);
   uint8_t value = (uint8_t)(v + 2);
   uint8_t first;
   if (d.offset1 & src2_offset_in_address) {
      bld.vop2(aco_opcode::v_and_b32, vdef(a), imm(src2_address_bits), vop(d.addr));
      bld.vop2(aco_opcode::v_ashrrev_i32, vdef(b), imm(17), vop(d.addr));
      bld.vop2(aco_opcode::v_lshlrev_b32, vdef(b), imm(2), vop(b));
      bld.vop2(aco_opcode::v_add_u32, vdef(b), vop(a), vop(b));
      first = a;
   } else {
      uint32_t field = ((uint32_t)(d.offset1 & 0x7f) << 8) | d.offset0;
      int32_t dwords = (int32_t)(field & 0x3fff) - (int32_t)(field & 0x4000);
      bld.vop2(aco_opcode::v_add_u32, vdef(b), imm((uint32_t)(dwords * 4)), vop(d.addr));
      first = d.addr;
   }
   lds_access(ctx, inst, AGC_ISA_DS_READ_B32, b, value, 0, 0, 0, 0);
   waitcnt(ctx, AGC_ISA_GFX10_VMCNT_MAX, AGC_ISA_GFX10_EXPCNT_MAX, 0);
   lds_access(ctx, inst, plain, first, 0, value, 0, 0, 0);
   waitcnt(ctx, AGC_ISA_GFX10_VMCNT_MAX, AGC_ISA_GFX10_EXPCNT_MAX, 0);
}

}
