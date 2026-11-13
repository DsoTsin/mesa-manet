#include <string.h>

#include "agc_lower.h"

namespace agc {

static const uint16_t hwreg_mode_f64_denorm = 1 | (6 << 6) | (1 << 11);
static const uint32_t class_signaling_nan = 1;
static const uint32_t class_nan = 3;
static const uint32_t f64_hi_min_normal = 0x00100000u;
static const uint32_t f64_hi_quiet = 0x00080000u;

struct order {
   unsigned mask;
   unsigned key;
   unsigned class_mask;
   unsigned temp;
};

bool
float_atomic_named(uint16_t opcode, agc_isa_float_atomic *atomic)
{
   memset(atomic, 0, sizeof(*atomic));
   switch (opcode) {
   case AGC_ISA_VMEM_ATOMIC_FCMPSWAP:
      atomic->kind = AGC_ISA_FLOAT_ATOMIC_COMPARE_SWAP;
      return true;
   case AGC_ISA_VMEM_ATOMIC_FCMPSWAP_X2:
      atomic->kind = AGC_ISA_FLOAT_ATOMIC_COMPARE_SWAP;
      atomic->wide = true;
      return true;
   case AGC_ISA_VMEM_ATOMIC_FMIN_X2:
      atomic->kind = AGC_ISA_FLOAT_ATOMIC_MIN_MAX;
      return true;
   case AGC_ISA_VMEM_ATOMIC_FMAX_X2:
      atomic->kind = AGC_ISA_FLOAT_ATOMIC_MIN_MAX;
      atomic->max = true;
      return true;
   default:
      return false;
   }
}

static uint8_t
atomic_dwords(agc_isa_float_atomic atomic)
{
   return atomic.kind == AGC_ISA_FLOAT_ATOMIC_COMPARE_SWAP && !atomic.wide ? 1 : 2;
}

static void
push_access(lower_ctx &ctx, const agc_isa_inst *tmpl, uint16_t opcode, uint8_t data, uint8_t dwords)
{
   const agc_isa_mem_op *info = tmpl->family == AGC_ISA_MUBUF ? agc_isa_mubuf_op(opcode) : agc_isa_flat_op(opcode);
   agc_isa_xlat xlat = agc_isa_xlat_of(AGC_ISA_XLAT_SAME);
   if (info)
      xlat = agc_isa_mem_op_translate(info, ctx.target);
   if (!info || xlat.kind != AGC_ISA_XLAT_OP)
      fail(ctx, tmpl, "no integer atomic to emulate a float one");
   agc_isa_inst inst = agc_isa_inst_retarget(tmpl, xlat.value);
   if (opcode == AGC_ISA_VMEM_ATOMIC_CMPSWAP || opcode == AGC_ISA_VMEM_ATOMIC_CMPSWAP_X2)
      inst.flags |= AGC_ISA_FLAG_CMPSWAP;
   else
      inst.flags &= ~AGC_ISA_FLAG_CMPSWAP;
   switch (inst.fields) {
   case AGC_ISA_MUBUF:
      inst.buffer.vdata = data;
      inst.buffer.data_dwords = dwords;
      inst.buffer.glc = true;
      break;
   case AGC_ISA_FLAT:
      inst.flat.vdata = data;
      inst.flat.vdst = data;
      inst.flat.data_dwords = dwords;
      inst.flat.glc = true;
      break;
   default:
      fail(ctx, tmpl, "a float atomic outside the buffer and global units");
   }
   push(ctx, inst);
}

static void
order_flush_mask(lower_ctx &ctx, const order &o, unsigned field)
{
   Builder &bld = ctx.bld;
   bld.sopk(aco_opcode::s_getreg_b32, sdef(field), hwreg_mode_f64_denorm);
   bld.sop2(aco_opcode::s_and_b32, sdef(field), scc_def(), imm(1), sop(field));
   bld.sop2(Builder::s_cselect, sdef(o.mask, bld.lm.size()), imm(0), imm(0xffffffffu), scc_op());
}

static void
order_key(lower_ctx &ctx, const order &o, unsigned value, unsigned key)
{
   Builder &bld = ctx.bld;
   unsigned lo = value;
   unsigned hi = value + 1;
   bld.vop2(aco_opcode::v_and_b32, vdef(key + 1), lit(0x7fffffff), vop(hi));
   bld.vop2(aco_opcode::v_and_b32, vdef(o.temp), lit(0x80000000u), vop(hi));
   bld.vopc(aco_opcode::v_cmp_gt_u32, mask_def(ctx, vcc), lit(f64_hi_min_normal), vop(key + 1));
   bld.sop2(Builder::s_and, mask_def(ctx, vcc), scc_def(), mask_op(ctx, vcc), sop(o.mask, bld.lm.size()));
   bld.vop2(aco_opcode::v_cndmask_b32, vdef(key + 1), vop(hi), vop(o.temp), mask_op(ctx, vcc));
   bld.vop1(aco_opcode::v_mov_b32, vdef(o.class_mask), imm(0));
   bld.vop2(aco_opcode::v_cndmask_b32, vdef(key), vop(lo), vop(o.class_mask), mask_op(ctx, vcc));
   bld.vop2(aco_opcode::v_ashrrev_i32, vdef(o.temp), imm(31), vop(key + 1));
   bld.vop2(aco_opcode::v_xor_b32, vdef(key), vop(o.temp), vop(key));
   bld.vop2(aco_opcode::v_or_b32, vdef(o.temp), lit(0x80000000u), vop(o.temp));
   bld.vop2(aco_opcode::v_xor_b32, vdef(key + 1), vop(o.temp), vop(key + 1));
}

static void
order_choose(lower_ctx &ctx, unsigned new_value, unsigned yes, unsigned no)
{
   for (unsigned k = 0; k < 2; k++)
      ctx.bld.vop2(aco_opcode::v_cndmask_b32, vdef(new_value + k), vop(no + k), vop(yes + k), mask_op(ctx, vcc));
}

static void
order_select(lower_ctx &ctx, const order &o, bool max, unsigned new_value, unsigned data, unsigned old)
{
   Builder &bld = ctx.bld;
   bld.vopc(max ? aco_opcode::v_cmp_gt_u64 : aco_opcode::v_cmp_lt_u64, mask_def(ctx, vcc), vop(o.key, 2),
            vop(new_value + 2, 2));
   order_choose(ctx, new_value, data, old);
   bld.vop1(aco_opcode::v_mov_b32, vdef(o.class_mask), imm(class_nan));
   bld.vopc(aco_opcode::v_cmp_class_f64, mask_def(ctx, vcc), vop(old, 2), vop(o.class_mask));
   order_choose(ctx, new_value, data, new_value);
   bld.vopc(aco_opcode::v_cmp_class_f64, mask_def(ctx, vcc), vop(data, 2), vop(o.class_mask));
   order_choose(ctx, new_value, old, new_value);
   bld.vop1(aco_opcode::v_mov_b32, vdef(o.class_mask), imm(class_signaling_nan));
   const unsigned nans[2] = {old, data};
   for (unsigned nan : nans) {
      bld.vopc(aco_opcode::v_cmp_class_f64, mask_def(ctx, vcc), vop(nan, 2), vop(o.class_mask));
      bld.vop2(aco_opcode::v_or_b32, vdef(o.temp), lit(f64_hi_quiet), vop(nan + 1));
      bld.vop2(aco_opcode::v_cndmask_b32, vdef(new_value), vop(new_value), vop(nan), mask_op(ctx, vcc));
      bld.vop2(aco_opcode::v_cndmask_b32, vdef(new_value + 1), vop(new_value + 1), vop(o.temp), mask_op(ctx, vcc));
   }
}

void
memory_atomic(lower_ctx &ctx, const agc_isa_inst *origin, const agc_isa_inst *tmpl, agc_isa_float_atomic atomic)
{
   unsigned data, result;
   bool returns;
   if (origin->fields == AGC_ISA_MUBUF && !origin->buffer.tfe && !origin->buffer.lds) {
      data = origin->buffer.vdata;
      result = origin->buffer.vdata;
      returns = origin->buffer.glc;
   } else if (origin->fields == AGC_ISA_FLAT && origin->flat.segment != AGC_ISA_SEGMENT_SCRATCH &&
              !origin->flat.lds) {
      data = origin->flat.vdata;
      result = origin->flat.vdst;
      returns = origin->flat.glc;
   } else {
      fail(ctx, origin, "float atomic emulation needs a buffer or global atomic without TFE or LDS");
   }
   Builder &bld = ctx.bld;
   agc_isa_inst access = *tmpl;
   unsigned d = atomic_dwords(atomic);
   bool min_max = atomic.kind == AGC_ISA_FLOAT_ATOMIC_MIN_MAX;
   uint8_t s = transient_sgprs(ctx, origin, min_max ? 8 : 5);
   uint8_t v = scratch_vgprs(ctx, min_max ? 10 : (uint8_t)(3 * d));
   unsigned pair = v;
   unsigned old = v + 2 * d;
   uint32_t loop_label = AGC_ISA_LOCAL_LABEL_BASE + origin->pc + 1;
   uint32_t done_label = AGC_ISA_LOCAL_LABEL_BASE + origin->pc + 3;
   save_masks(ctx, s);
   for (unsigned k = 0; k < d; k++)
      bld.vop1(aco_opcode::v_mov_b32, vdef(pair + k), imm(0));
   push_access(ctx, &access, d == 2 ? AGC_ISA_VMEM_ATOMIC_OR_X2 : AGC_ISA_VMEM_ATOMIC_OR, (uint8_t)pair, (uint8_t)d);
   waitcnt(ctx, 0, AGC_ISA_GFX10_EXPCNT_MAX, AGC_ISA_GFX10_LGKMCNT_MAX);
   for (unsigned k = 0; k < d; k++)
      bld.vop1(aco_opcode::v_mov_b32, vdef(old + k), vop(pair + k));
   order o = {(unsigned)s + 6, (unsigned)v + 6, (unsigned)v + 8, (unsigned)v + 9};
   if (min_max) {
      order_flush_mask(ctx, o, s + 5);
      order_key(ctx, o, data, o.key);
   }
   label(ctx, loop_label);
   if (!min_max) {
      bld.vopc(atomic.wide ? aco_opcode::v_cmp_eq_f64 : aco_opcode::v_cmp_eq_f32, mask_def(ctx, vcc),
               vop(old, atomic.wide ? 2 : 1), vop(data + d, atomic.wide ? 2 : 1));
      exec_and(ctx, mask_op(ctx, vcc));
      branch(ctx, aco_opcode::s_cbranch_execz, done_label);
      for (unsigned k = 0; k < d; k++)
         bld.vop1(aco_opcode::v_mov_b32, vdef(pair + k), vop(data + k));
   } else {
      order_key(ctx, o, old, pair + 2);
      order_select(ctx, o, atomic.max, pair, data, old);
   }
   for (unsigned k = 0; k < d; k++)
      bld.vop1(aco_opcode::v_mov_b32, vdef(pair + d + k), vop(old + k));
   push_access(ctx, &access, d == 2 ? AGC_ISA_VMEM_ATOMIC_CMPSWAP_X2 : AGC_ISA_VMEM_ATOMIC_CMPSWAP, (uint8_t)pair,
               (uint8_t)(2 * d));
   waitcnt(ctx, 0, AGC_ISA_GFX10_EXPCNT_MAX, AGC_ISA_GFX10_LGKMCNT_MAX);
   bld.vopc(d == 2 ? aco_opcode::v_cmp_eq_u64 : aco_opcode::v_cmp_eq_u32, mask_def(ctx, vcc), vop(pair, d), vop(old, d));
   for (unsigned k = 0; k < d; k++)
      bld.vop1(aco_opcode::v_mov_b32, vdef(old + k), vop(pair + k));
   exec_andn2(ctx, mask_op(ctx, vcc));
   branch(ctx, aco_opcode::s_cbranch_execnz, loop_label);
   label(ctx, done_label);
   restore_masks(ctx, s);
   if (returns) {
      for (unsigned k = 0; k < d; k++)
         bld.vop1(aco_opcode::v_mov_b32, vdef(result + k), vop(old + k));
   }
}

}
