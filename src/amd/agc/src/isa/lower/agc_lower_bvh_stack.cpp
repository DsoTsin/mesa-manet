#include "agc_lower.h"

namespace agc {

static const uint32_t valid_offset = 0;
static const uint32_t to_tlas_offset = 5;
static const uint32_t ring_offset = 10;
static const uint32_t valid_mask = 0x0000001fu;
static const uint32_t to_tlas_mask = 0x000003e0u;
static const uint32_t ring_mask = 0x00007c00u;
static const uint32_t has_tlas = 1u << 29;
static const uint32_t overflow = 1u << 30;
static const uint32_t bottom_to_top = 1u << 31;
static const uint32_t sentinel_first = 0xfffffff8u;
static const uint32_t terminal_node = 0xfffffffeu;
static const uint32_t instance_kind = 6;
static const uint32_t ring_stride_shift = 7;
static const uint32_t flag_walkback = 1;
static const uint32_t flag_valid = 2;
static const uint32_t flag_skip_high = 4;
static const uint32_t flag_skip_low = 8;

struct ring_stack {
   lower_ctx &ctx;
   Builder &bld;
   const agc_isa_inst *origin;
   unsigned lm;
   unsigned saved_exec;
   unsigned p;
   unsigned q;
   unsigned control;
   unsigned flags;
   unsigned address;
   unsigned tmp;
   unsigned result;
   uint32_t entries;

   void v2(aco_opcode op, unsigned vdst, Operand src0, unsigned vsrc1)
   {
      bld.vop2(op, vdef(vdst), src0, vop(vsrc1));
   }

   void compare(aco_opcode op, unsigned sdst, Operand src0, unsigned vsrc1)
   {
      bld.vopc_e64(op, sdef(sdst, lm), src0, vop(vsrc1));
   }

   void extract(unsigned vdst, unsigned source, uint32_t offset, uint32_t width)
   {
      bld.vop3(aco_opcode::v_bfe_u32, vdef(vdst), vop(source), imm(offset), imm(width));
   }

   void insert(unsigned vdst, Operand mask, Operand bits, unsigned base)
   {
      bld.vop3(aco_opcode::v_bfi_b32, vdef(vdst), mask, bits, vop(base));
   }

   void select(unsigned vdst, Operand clear, Operand set, unsigned mask)
   {
      bld.vop2_e64(aco_opcode::v_cndmask_b32, vdef(vdst), clear, set, sop(mask, lm));
   }

   void restrict_exec(unsigned mask)
   {
      bld.sop2(Builder::s_and, mask_def(ctx, exec), scc_def(), sop(saved_exec, lm), sop(mask, lm));
   }

   void restore_exec()
   {
      bld.sop1(Builder::s_mov, mask_def(ctx, exec), sop(saved_exec, lm));
   }

   void insert_field(uint32_t mask)
   {
      insert(control, imm(mask), vop(tmp), control);
   }

   void clear_by_tmp()
   {
      insert(control, vop(tmp), imm(0), control);
   }

   void or_tmp()
   {
      v2(aco_opcode::v_or_b32, control, vop(tmp), control);
   }

   void slot_address()
   {
      v2(aco_opcode::v_lshrrev_b32, address, imm(13), control);
      v2(aco_opcode::v_and_b32, address, imm(0xfffc), address);
      extract(tmp, control, ring_offset, 5);
      v2(aco_opcode::v_lshlrev_b32, tmp, imm(ring_stride_shift), tmp);
      v2(aco_opcode::v_add_u32, address, vop(address), tmp);
   }

   void is_sentinel(unsigned sdst, unsigned node)
   {
      compare(aco_opcode::v_cmp_le_u32, sdst, imm(sentinel_first), node);
   }

   void init(unsigned control_in, unsigned last, const uint8_t candidates[8])
   {
      bld.vop1(aco_opcode::v_mov_b32, vdef(control), vop(control_in));
      is_sentinel(p, last);
      select(flags, imm(flag_walkback), imm(0), p);
      select(tmp, imm(0), imm(1), p);
      for (unsigned i = 0; i < 8; i++) {
         compare(aco_opcode::v_cmp_eq_u32, p, vop(last), candidates[i]);
         select(tmp, vop(tmp), imm(1), p);
      }
      v2(aco_opcode::v_lshlrev_b32, tmp, imm(1), tmp);
      v2(aco_opcode::v_or_b32, flags, vop(tmp), flags);
   }

   void candidate(unsigned node, bool high, unsigned last)
   {
      uint32_t skip = high ? flag_skip_high : flag_skip_low;
      is_sentinel(p, node);
      v2(aco_opcode::v_and_b32, tmp, imm(flag_valid | skip), flags);
      select(tmp, imm(0), vop(tmp), p);
      compare(aco_opcode::v_cmp_eq_u32, q, imm(flag_valid), tmp);

      v2(aco_opcode::v_and_b32, address, imm(2), node);
      compare(aco_opcode::v_cmp_eq_u32, p, imm(0), address);
      select(tmp, imm(0), imm(flag_skip_low), q);
      select(tmp, imm(0), vop(tmp), p);
      v2(aco_opcode::v_or_b32, flags, vop(tmp), flags);

      if (high) {
         v2(aco_opcode::v_and_b32, address, imm(4), node);
         compare(aco_opcode::v_cmp_eq_u32, p, imm(0), address);
         select(tmp, imm(0), imm(flag_skip_high), q);
         select(tmp, imm(0), vop(tmp), p);
         v2(aco_opcode::v_or_b32, flags, vop(tmp), flags);
      }

      compare(aco_opcode::v_cmp_eq_u32, p, vop(last), node);
      select(tmp, imm(0), vop(flags), p);
      v2(aco_opcode::v_and_b32, tmp, imm(flag_walkback), tmp);
      v2(aco_opcode::v_lshlrev_b32, tmp, imm(1), tmp);
      insert(flags, vop(tmp), imm(0), flags);

      v2(aco_opcode::v_and_b32, tmp, imm(flag_valid | skip), flags);
      is_sentinel(p, node);
      select(tmp, vop(tmp), imm(0), p);
      compare(aco_opcode::v_cmp_eq_u32, q, imm(flag_valid), tmp);

      restrict_exec(q);
      push_node(node);
      restore_exec();
   }

   void push_node(unsigned node)
   {
      extract(tmp, control, valid_offset, 5);
      compare(aco_opcode::v_cmp_eq_u32, p, imm(entries), tmp);
      select(tmp, imm(0), imm(overflow), p);
      or_tmp();

      slot_address();
      lds_access(ctx, origin, AGC_ISA_DS_WRITE_B32, (uint8_t)address, 0, (uint8_t)node, 0, 0, 0);
      waitcnt(ctx, AGC_ISA_GFX10_VMCNT_MAX, AGC_ISA_GFX10_EXPCNT_MAX, 0);

      extract(tmp, control, ring_offset, 5);
      compare(aco_opcode::v_cmp_eq_u32, p, imm(entries - 1), tmp);
      v2(aco_opcode::v_add_u32, tmp, imm(1), tmp);
      select(tmp, vop(tmp), imm(0), p);
      v2(aco_opcode::v_lshlrev_b32, tmp, imm(ring_offset), tmp);
      insert_field(ring_mask);

      extract(tmp, control, to_tlas_offset, 5);
      compare(aco_opcode::v_cmp_eq_u32, p, imm(entries), tmp);
      select(tmp, imm(0), imm(has_tlas), p);
      clear_by_tmp();

      extract(tmp, control, to_tlas_offset, 5);
      v2(aco_opcode::v_add_u32, tmp, imm(1), tmp);
      v2(aco_opcode::v_min_u32, tmp, imm(entries), tmp);
      v2(aco_opcode::v_lshlrev_b32, tmp, imm(to_tlas_offset), tmp);
      insert_field(to_tlas_mask);

      extract(tmp, control, valid_offset, 5);
      v2(aco_opcode::v_add_u32, tmp, imm(1), tmp);
      v2(aco_opcode::v_min_u32, tmp, imm(entries), tmp);
      insert_field(valid_mask);
   }

   void pop()
   {
      extract(tmp, control, to_tlas_offset, 5);
      compare(aco_opcode::v_cmp_eq_u32, p, imm(0), tmp);
      select(tmp, imm(0), vop(control), p);
      v2(aco_opcode::v_and_b32, tmp, imm(has_tlas), tmp);
      v2(aco_opcode::v_lshlrev_b32, tmp, imm(2), tmp);
      insert_field(bottom_to_top);
      v2(aco_opcode::v_lshrrev_b32, tmp, imm(2), tmp);
      clear_by_tmp();

      extract(tmp, control, valid_offset, 5);
      compare(aco_opcode::v_cmp_eq_u32, p, imm(0), tmp);
      select(tmp, imm(0), imm(bottom_to_top | has_tlas), p);
      clear_by_tmp();
      extract(result, control, 30, 1);
      v2(aco_opcode::v_add_u32, result, imm(terminal_node), result);

      extract(tmp, control, valid_offset, 5);
      compare(aco_opcode::v_cmp_lg_u32, q, imm(0), tmp);
      restrict_exec(q);

      extract(tmp, control, to_tlas_offset, 5);
      v2(aco_opcode::v_add_u32, tmp, imm(0xffffffffu), tmp);
      v2(aco_opcode::v_lshlrev_b32, tmp, imm(to_tlas_offset), tmp);
      insert_field(to_tlas_mask);

      extract(tmp, control, valid_offset, 5);
      v2(aco_opcode::v_add_u32, tmp, imm(0xffffffffu), tmp);
      insert_field(valid_mask);

      extract(tmp, control, ring_offset, 5);
      compare(aco_opcode::v_cmp_eq_u32, p, imm(0), tmp);
      v2(aco_opcode::v_add_u32, tmp, imm(0xffffffffu), tmp);
      select(tmp, vop(tmp), imm(entries - 1), p);
      v2(aco_opcode::v_lshlrev_b32, tmp, imm(ring_offset), tmp);
      insert_field(ring_mask);

      slot_address();
      lds_access(ctx, origin, AGC_ISA_DS_READ_B32, (uint8_t)address, (uint8_t)result, 0, 0, 0, 0);
      waitcnt(ctx, AGC_ISA_GFX10_VMCNT_MAX, AGC_ISA_GFX10_EXPCNT_MAX, 0);

      v2(aco_opcode::v_and_b32, tmp, imm(7), result);
      compare(aco_opcode::v_cmp_eq_u32, p, imm(instance_kind), tmp);
      extract(tmp, control, 29, 1);
      compare(aco_opcode::v_cmp_lg_u32, q, imm(0), tmp);
      bld.sop2(Builder::s_and, sdef(q, lm), scc_def(), sop(q, lm), sop(p, lm));
      extract(address, control, valid_offset, 5);
      extract(tmp, control, to_tlas_offset, 5);
      select(tmp, vop(address), vop(tmp), q);
      insert_field(valid_mask);
      select(tmp, imm(0), imm(overflow), q);
      or_tmp();
      select(tmp, imm(0), imm(has_tlas), p);
      or_tmp();
      select(tmp, imm(0), imm(to_tlas_mask), p);
      clear_by_tmp();

      restore_exec();
   }
};

void
lower_bvh_stack(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_DS)
      fail(ctx, inst, "ds_bvh_stack_push8_pop1: expected DS fields");
   const agc_isa_ds d = inst->ds;
   uint32_t entries = d.offset0;
   if (entries < 8 || entries > 31 || d.offset1 != 0 || d.gds)
      fail(ctx, inst, "ds_bvh_stack_push8_pop1: unsupported stack size/encoding");
   if (d.data1 + 7u > 255 || d.addr == d.vdst)
      fail(ctx, inst, "ds_bvh_stack_push8_pop1: invalid register operands");
   if (ctx.gfx12) {
      push_retarget(ctx, inst, inst->opcode);
      return;
   }
   uint8_t candidates[8];
   for (unsigned i = 0; i < 8; i++)
      candidates[i] = (uint8_t)(d.data1 + i);
   unsigned v = scratch_vgprs(ctx, 5);
   unsigned s = transient_sgprs(ctx, inst, 8);
   Builder &bld = ctx.bld;
   ring_stack stack = {ctx, bld, inst, bld.lm.size(), s, s + 2, s + 4, v, v + 1, v + 2, v + 3, v + 4, entries};
   save_scc(ctx, s + 6);
   bld.sop1(Builder::s_mov, sdef(s, stack.lm), mask_op(ctx, exec));
   stack.init(d.addr, d.data0, candidates);
   for (unsigned i = 8; i-- > 4;)
      stack.candidate(candidates[i], true, d.data0);
   for (unsigned i = 4; i-- > 0;)
      stack.candidate(candidates[i], false, d.data0);
   stack.pop();
   bld.vop1(aco_opcode::v_mov_b32, vdef(d.addr), vop(stack.control));
   bld.vop1(aco_opcode::v_mov_b32, vdef(d.vdst), vop(stack.result));
   restore_scc(ctx, s + 6);
}

}
