#include "agc_lower.h"

namespace agc {

static const uint32_t node_triangle_pair = 1;
static const uint32_t node_big_fp32 = 2;
static const uint32_t node_big_shared_exponent = 3;
static const uint32_t node_shared_exponent = 4;
static const uint32_t node_fp32 = 5;
static const uint32_t node_instance = 6;
static const uint32_t node_user = 7;

static const uint32_t invalid = 0xffffffffu;
static const uint32_t f32_inf = 0x7f800000u;
static const uint32_t f32_one = 0x3f800000u;
static const uint32_t f32_half = 0x3f000000u;
static const uint32_t f32_grow_step = 0x33800000u;
static const uint32_t f32_abs = 0x7fffffffu;
static const uint32_t sign_bit = 0x80000000u;

static const uint8_t float_mode = 0xc0;
static const uint16_t hwreg_mode_float = 1 | (7 << 11);

static const uint32_t desc_sort_heuristic = (2u << 16) | 21;
static const uint32_t desc_grow_ulp = (8u << 16) | 23;

static const aco_opcode eq_u = aco_opcode::v_cmp_eq_u32;
static const aco_opcode ne_u = aco_opcode::v_cmp_lg_u32;
static const aco_opcode le_u = aco_opcode::v_cmp_le_u32;
static const aco_opcode gt_u = aco_opcode::v_cmp_gt_u32;
static const aco_opcode ge_u = aco_opcode::v_cmp_ge_u32;
static const aco_opcode lt_f = aco_opcode::v_cmp_lt_f32;
static const aco_opcode eq_f = aco_opcode::v_cmp_eq_f32;
static const aco_opcode gt_f = aco_opcode::v_cmp_gt_f32;
static const aco_opcode ge_f = aco_opcode::v_cmp_ge_f32;
static const aco_opcode unordered_f = aco_opcode::v_cmp_u_f32;

static const uint8_t sort_network[19][2] = {
   {0, 1}, {2, 3}, {0, 2}, {1, 3}, {1, 2}, {4, 5}, {6, 7}, {4, 6}, {5, 7}, {5, 6},
   {0, 4}, {1, 5}, {1, 4}, {2, 6}, {3, 7}, {3, 6}, {2, 4}, {3, 5}, {3, 4},
};
static const unsigned sort_group_steps = 10;

enum {
   V_NODE = 0,
   V_EXTENT = 2,
   V_INSTANCE_MASK = 3,
   V_ORIGIN = 4,
   V_DIRECTION = 7,
   V_INDEX = 10,
   V_POINTER_FLAGS = 11,
   V_KIND = 12,
   V_TRIANGLE_FLAGS = 13,
   V_ADDRESS = 14,
   V_OUT = 16,
   V_KEY = 26,
   V_DATA = 34,
   V_WORK = 66,
   V_COUNT = 86,

   V_BOX_MIN = V_WORK,
   V_BOX_MAX = V_WORK + 3,
   V_CHILD = V_WORK + 6,
   V_CHILD_FLAGS = V_WORK + 7,
   V_T = V_WORK + 8,
   V_INVERSE = V_WORK + 16,
   V_GROW = V_WORK + 19,

   V_RELATIVE = 26,
   V_PERMUTED = 35,
   V_AXIS = 44,
   V_AX = 47,
   V_AY = 48,
   V_BX = 49,
   V_BY = 50,
   V_CX = 51,
   V_CY = 52,
   V_U = 53,
   V_V = 54,
   V_W = 55,
   V_UZ = 56,
   V_VZ = 57,
   V_WZ = 58,
   V_NUMERATOR = 59,
   V_DENOMINATOR = 60,
   V_RESULT_Z = 61,
   V_RESULT_W = 62,
   V_PICK_Z = 63,
   V_PICK_W = 64,
   V_PROCEDURAL = 67,
   V_DOUBLE_SIDED = 68,
   V_OPAQUE = 69,
   V_FLIP = 70,
   V_FRONT = 72,
   V_S0 = 73,
   V_S1 = 74,
   V_S2 = 75,
};

enum {
   M_EXEC = 0,
   M_VALID = 1,
   M_INSTANCE = 2,
   M_TRIANGLE = 3,
   M_A = 4,
   M_B = 5,
   M_C = 6,
   M_D = 7,
   M_M = 8,
   M_N = 9,
   M_O = 10,
   S_SCC = 22,
   S_MODE = 23,
};

struct bvh8 {
   lower_ctx &ctx;
   Builder &bld;
   const agc_isa_inst *origin;
   agc_isa_image m;
   unsigned lm;
   unsigned v;
   unsigned s;

   Operand r(unsigned slot) const
   {
      return vop(v + slot);
   }

   Definition w(unsigned slot) const
   {
      return vdef(v + slot);
   }

   Operand d(unsigned word) const
   {
      return r(V_DATA + word);
   }

   Operand mask(unsigned slot) const
   {
      return sop(s + 2 * slot, lm);
   }

   Definition mask_out(unsigned slot) const
   {
      return sdef(s + 2 * slot, lm);
   }

   unsigned guest(unsigned index) const
   {
      return agc_isa_image_address_register(&m, index);
   }

   Operand descriptor(unsigned word) const
   {
      return sop(m.srsrc + word);
   }

   void mask_and(unsigned dst, unsigned a, unsigned b)
   {
      bld.sop2(Builder::s_and, mask_out(dst), scc_def(), mask(a), mask(b));
   }

   void mask_or(unsigned dst, unsigned a, unsigned b)
   {
      bld.sop2(Builder::s_or, mask_out(dst), scc_def(), mask(a), mask(b));
   }

   void mask_xor(unsigned dst, unsigned a, unsigned b)
   {
      bld.sop2(Builder::s_xor, mask_out(dst), scc_def(), mask(a), mask(b));
   }

   void mask_andn2(unsigned dst, unsigned a, unsigned b)
   {
      bld.sop2(Builder::s_andn2, mask_out(dst), scc_def(), mask(a), mask(b));
   }

   void set_exec(unsigned slot)
   {
      bld.sop2(Builder::s_and, mask_def(ctx, exec), scc_def(), mask(M_EXEC), mask(slot));
   }

   void uniform(unsigned dst)
   {
      bld.sop2(Builder::s_cselect, mask_out(dst), imm(invalid), imm(0), scc_op());
   }

   void descriptor_flag(unsigned dst, unsigned word, uint32_t bit)
   {
      bld.sopc(aco_opcode::s_bitcmp1_b32, scc_def(), descriptor(word), imm(bit));
      uniform(dst);
   }

   void heuristic_flag(unsigned dst, unsigned tmp, uint32_t value)
   {
      bld.sop2(aco_opcode::s_bfe_u32, sdef(s + 2 * tmp), scc_def(), descriptor(1), imm(desc_sort_heuristic));
      bld.sopc(aco_opcode::s_cmp_eq_u32, scc_def(), sop(s + 2 * tmp), imm(value));
      uniform(dst);
   }

   void mov(unsigned dst, Operand src)
   {
      bld.vop1(aco_opcode::v_mov_b32, w(dst), src);
   }

   void v2(aco_opcode op, unsigned dst, Operand a, unsigned b)
   {
      bld.vop2(op, w(dst), a, r(b));
   }

   void fadd(unsigned dst, unsigned a, unsigned b)
   {
      v2(aco_opcode::v_add_f32, dst, r(a), b);
   }

   void fsub(unsigned dst, unsigned a, unsigned b)
   {
      v2(aco_opcode::v_sub_f32, dst, r(a), b);
   }

   void fmul(unsigned dst, unsigned a, unsigned b)
   {
      v2(aco_opcode::v_mul_f32, dst, r(a), b);
   }

   void iadd(unsigned dst, Operand a, unsigned b)
   {
      v2(aco_opcode::v_add_u32, dst, a, b);
   }

   void iand(unsigned dst, Operand a, unsigned b)
   {
      v2(aco_opcode::v_and_b32, dst, a, b);
   }

   void ior(unsigned dst, Operand a, unsigned b)
   {
      v2(aco_opcode::v_or_b32, dst, a, b);
   }

   void ixor(unsigned dst, Operand a, unsigned b)
   {
      v2(aco_opcode::v_xor_b32, dst, a, b);
   }

   void shr(unsigned dst, Operand amount, unsigned b)
   {
      v2(aco_opcode::v_lshrrev_b32, dst, amount, b);
   }

   void shl(unsigned dst, Operand amount, unsigned b)
   {
      v2(aco_opcode::v_lshlrev_b32, dst, amount, b);
   }

   void bfe(unsigned dst, unsigned source, uint32_t offset, uint32_t width)
   {
      bld.vop3(aco_opcode::v_bfe_u32, w(dst), r(source), imm(offset), imm(width));
   }

   void cmp(aco_opcode op, unsigned dst, Operand a, Operand b)
   {
      bld.vopc_e64(op, mask_out(dst), a, b);
   }

   void select(Definition dst, unsigned condition, Operand yes, Operand no)
   {
      bld.vop2_e64(aco_opcode::v_cndmask_b32, dst, no, yes, mask(condition));
   }

   void sel(unsigned dst, unsigned condition, Operand yes, Operand no)
   {
      select(w(dst), condition, yes, no);
   }

   void nop()
   {
      bld.vop1(aco_opcode::v_nop);
   }

   void load(unsigned dst, unsigned dwords, int32_t offset)
   {
      for (unsigned q = 0; q < dwords; q += 4)
         global_access(ctx, origin, AGC_ISA_VMEM_LOAD_DWORDX4, v + V_ADDRESS,
                       agc_isa_operand_special(AGC_ISA_OPERAND_NULL), 0, offset + 4 * (int32_t)q, v + dst + q);
   }

   void wait_loads()
   {
      nop();
      waitcnt(ctx, 0, AGC_ISA_GFX10_EXPCNT_MAX, AGC_ISA_GFX10_LGKMCNT_MAX);
      nop();
   }

   void kind_is(unsigned dst, unsigned tmp, uint32_t first, uint32_t second)
   {
      cmp(eq_u, dst, r(V_KIND), imm(first));
      cmp(eq_u, tmp, r(V_KIND), imm(second));
      mask_or(dst, dst, tmp);
   }

   void switch_mode(uint8_t from, uint8_t to)
   {
      if ((from & 0xf) != (to & 0xf))
         bld.sopp(aco_opcode::s_round_mode, to & 0xf);
      if ((from >> 4) != (to >> 4))
         bld.sopp(aco_opcode::s_denorm_mode, to >> 4);
   }

   void enter_mode()
   {
      if (!ctx.mode.dynamic) {
         switch_mode(ctx.mode.mode, float_mode);
         return;
      }
      bld.sopk(aco_opcode::s_getreg_b32, sdef(s + S_MODE), hwreg_mode_float);
      bld.sopp(aco_opcode::s_round_mode, float_mode & 0xf);
      bld.sopp(aco_opcode::s_denorm_mode, float_mode >> 4);
   }

   void leave_mode()
   {
      if (!ctx.mode.dynamic)
         switch_mode(float_mode, ctx.mode.mode);
      else
         bld.sopk(aco_opcode::s_setreg_b32, sop(s + S_MODE), hwreg_mode_float);
   }

   void expand()
   {
      enter_mode();
      save_scc(ctx, s + S_SCC);
      bld.sop1(Builder::s_mov, mask_out(M_EXEC), mask_op(ctx, exec));
      prologue();
      classify();
      boxes();
      set_exec(M_EXEC);
      triangles();
      set_exec(M_EXEC);
      instance();
      write_back();
      restore_scc(ctx, s + S_SCC);
      leave_mode();
   }

   void prologue()
   {
      for (unsigned i = 0; i < 11; i++)
         mov(V_NODE + i, vop(guest(i)));
      for (unsigned i = 0; i < 10; i++)
         mov(V_OUT + i, imm(invalid));
      shr(V_POINTER_FLAGS, imm(22), V_NODE + 1);
      iand(V_WORK, imm(4), V_POINTER_FLAGS);
      cmp(ne_u, M_A, r(V_WORK), imm(0));
      iand(V_WORK, imm(0xffffffcfu), V_POINTER_FLAGS);
      sel(V_POINTER_FLAGS, M_A, r(V_WORK), r(V_POINTER_FLAGS));
      iand(V_WORK, imm(0x30), V_POINTER_FLAGS);
      cmp(eq_u, M_B, r(V_WORK), imm(0x30));
      ior(V_WORK, imm(0x100), V_POINTER_FLAGS);
      sel(V_POINTER_FLAGS, M_B, r(V_WORK), r(V_POINTER_FLAGS));

      mov(V_WORK, r(V_NODE));
      iand(V_WORK + 1, imm(0x003fffffu), V_NODE + 1);
      agc_isa_vop3 shift = agc_isa_vop3_new(agc_isa_operand_vgpr((uint8_t)(v + V_ADDRESS)), agc_isa_operand_constant(3),
                                            agc_isa_operand_vgpr((uint8_t)(v + V_WORK)), agc_isa_operand_raw(0));
      lower_template(ctx, origin, agc_isa_tmpl_vop3v(AGC_ISA_VOP3_LSHLREV_B64, &shift));
      iand(V_WORK, imm(0xfffffff8u), V_INDEX);
      shl(V_WORK, imm(3), V_WORK);
      shr(V_WORK + 1, imm(29), V_INDEX);
      bld.vop3(aco_opcode::v_add_co_u32_e64, w(V_ADDRESS), mask_out(M_A), r(V_ADDRESS), r(V_WORK));
      bld.vop2_e64(aco_opcode::v_addc_co_u32, w(V_ADDRESS + 1), mask_out(M_B), r(V_ADDRESS + 1), r(V_WORK + 1),
                   mask(M_A));
      iand(V_KIND, imm(7), V_INDEX);
   }

   void classify()
   {
      cmp(eq_u, M_A, r(V_KIND), imm(node_user));
      cmp(eq_u, M_B, r(V_KIND), imm(node_instance));
      descriptor_flag(M_C, 3, 22);
      mask_and(M_INSTANCE, M_B, M_C);
      mask_andn2(M_D, M_B, M_C);
      mask_or(M_A, M_A, M_D);
      cmp(le_u, M_TRIANGLE, r(V_KIND), imm(node_triangle_pair));
      iand(V_WORK, imm(0x100), V_POINTER_FLAGS);
      cmp(ne_u, M_B, r(V_WORK), imm(0));
      mask_and(M_B, M_B, M_TRIANGLE);
      mask_or(M_A, M_A, M_B);
      for (uint32_t bits : {0x41u, 0x82u, 0xc0u}) {
         iand(V_WORK, imm(bits), V_POINTER_FLAGS);
         cmp(eq_u, M_B, r(V_WORK), imm(bits));
         mask_or(M_A, M_A, M_B);
      }
      mask_andn2(M_VALID, M_EXEC, M_A);
      mask_and(M_INSTANCE, M_INSTANCE, M_VALID);
      mask_and(M_TRIANGLE, M_TRIANGLE, M_VALID);
   }

   void box_mask(unsigned dst, unsigned tmp)
   {
      cmp(ge_u, dst, r(V_KIND), imm(node_big_fp32));
      cmp(le_u, tmp, r(V_KIND), imm(node_fp32));
      mask_and(dst, dst, tmp);
      mask_and(dst, dst, M_VALID);
   }

   void layout_masks()
   {
      kind_is(M_N, M_A, node_fp32, node_big_fp32);
      mask_and(M_N, M_N, M_M);
      kind_is(M_O, M_A, node_shared_exponent, node_big_shared_exponent);
      mask_and(M_O, M_O, M_M);
   }

   void boxes()
   {
      set_exec(M_EXEC);
      box_mask(M_M, M_A);
      set_exec(M_M);
      for (unsigned axis = 0; axis < 3; axis++)
         bld.vop1(aco_opcode::v_rcp_f32, w(V_INVERSE + axis), r(V_DIRECTION + axis));
      bld.sop2(aco_opcode::s_bfe_u32, sdef(s + 2 * M_A), scc_def(), descriptor(1), imm(desc_grow_ulp));
      bld.vop1(aco_opcode::v_cvt_f32_u32, w(V_GROW), sop(s + 2 * M_A));
      v2(aco_opcode::v_mul_f32, V_GROW, imm(f32_grow_step), V_GROW);
      v2(aco_opcode::v_add_f32, V_GROW, imm(f32_one), V_GROW);
      for (unsigned child = 0; child < 8; child++)
         mov(V_KEY + child, imm(f32_inf));
      box_pass(0);
      box_pass(1);
      sort();
   }

   void box_pass(unsigned pass)
   {
      set_exec(M_EXEC);
      box_mask(M_M, M_A);
      if (pass == 1) {
         kind_is(M_B, M_A, node_big_fp32, node_big_shared_exponent);
         mask_and(M_M, M_M, M_B);
      }
      layout_masks();
      if (pass == 0) {
         set_exec(M_M);
         load(V_DATA, 16, 0);
         nop();
         set_exec(M_N);
         load(V_DATA + 16, 12, 64);
      } else {
         set_exec(M_N);
         load(V_DATA, 28, 128);
         nop();
         set_exec(M_O);
         load(V_DATA, 16, 64);
      }
      wait_loads();
      for (unsigned c = 0; c < 4; c++)
         child(pass * 4 + c, c);
      set_exec(M_EXEC);
   }

   void child(unsigned slot, unsigned c)
   {
      set_exec(M_EXEC);
      if (slot < 4) {
         box_mask(M_M, M_A);
      } else {
         kind_is(M_M, M_A, node_big_fp32, node_big_shared_exponent);
         mask_and(M_M, M_M, M_VALID);
      }
      layout_masks();
      set_exec(M_N);
      mov(V_CHILD, d(c));
      mov(V_CHILD_FLAGS, imm(0));
      for (unsigned i = 0; i < 6; i++)
         mov(V_BOX_MIN + i, d(4 + 6 * c + i));
      set_exec(M_O);
      shared_exponent_index(c);
      bfe(V_CHILD_FLAGS, V_DATA + 1, 9 + 4 * c, c < 3 ? 4 : 3);
      for (unsigned bound = 0; bound < 6; bound++)
         shared_exponent_bound(V_BOX_MIN + bound, c, bound % 3, bound >= 3);
      set_exec(M_M);
      slab(slot);
      set_exec(M_EXEC);
   }

   void packed(unsigned dst, uint32_t offset, uint32_t width)
   {
      unsigned word = offset / 32;
      uint32_t shift = offset % 32;
      shr(dst, imm(shift), V_DATA + word);
      if (shift + width > 32) {
         shl(V_T + 7, imm(32 - shift), V_DATA + word + 1);
         ior(dst, r(V_T + 7), dst);
      }
      iand(dst, imm((1u << width) - 1), dst);
   }

   void shared_exponent_index(unsigned c)
   {
      packed(V_CHILD, 0, 29);
      for (unsigned earlier = 0; earlier < c; earlier++) {
         packed(V_T, 29 + 3 * earlier, 3);
         mov(V_T + 1, imm(1));
         cmp(eq_u, M_A, r(V_T), imm(2));
         sel(V_T + 1, M_A, imm(4), r(V_T + 1));
         cmp(eq_u, M_A, r(V_T), imm(3));
         cmp(eq_u, M_B, r(V_T), imm(5));
         mask_or(M_A, M_A, M_B);
         cmp(eq_u, M_B, r(V_T), imm(6));
         mask_or(M_A, M_A, M_B);
         sel(V_T + 1, M_A, imm(2), r(V_T + 1));
         iadd(V_CHILD, r(V_CHILD), V_T + 1);
      }
      packed(V_T, 29 + 3 * c, 3);
      shl(V_CHILD, imm(3), V_CHILD);
      ior(V_CHILD, r(V_T), V_CHILD);
   }

   void shared_exponent_bound(unsigned dst, unsigned c, unsigned axis, bool max)
   {
      const unsigned exponent = V_T;
      const unsigned sign = V_T + 1;
      const unsigned mantissa = V_T + 2;
      const unsigned leading = V_T + 3;
      const unsigned shift = V_T + 4;
      const unsigned result_exponent = V_T + 5;
      const unsigned result_mantissa = V_T + 6;
      const unsigned trailing = V_T + 7;
      packed(dst, 80 + 18 * (6 * c + (max ? 3 : 0) + axis), 18);
      packed(exponent, 56 + 8 * axis, 8);
      bfe(sign, dst, 17, 1);
      shl(sign, imm(31), sign);
      iand(mantissa, imm(0x1ffff), dst);
      bld.vop1(aco_opcode::v_ffbh_u32, w(leading), r(mantissa));
      iadd(shift, imm((uint32_t)-15), leading);
      v2(aco_opcode::v_min_u32, shift, r(exponent), shift);
      v2(aco_opcode::v_sub_u32, result_exponent, r(exponent), shift);
      iadd(shift, imm(7), shift);
      shl(result_mantissa, r(shift), mantissa);
      bld.vop2_e64(aco_opcode::v_lshlrev_b32, w(trailing), r(shift), imm(1));
      iadd(trailing, imm(invalid), trailing);
      cmp(eq_u, M_A, r(sign), imm(max ? 0 : sign_bit));
      cmp(ne_u, M_B, r(result_exponent), imm(255));
      mask_and(M_B, M_A, M_B);
      sel(trailing, M_B, r(trailing), imm(0));
      ior(result_mantissa, r(trailing), result_mantissa);
      iand(result_mantissa, imm(0x7fffff), result_mantissa);
      iadd(leading, imm((uint32_t)-17), exponent);
      cmp(gt_u, M_B, r(exponent), imm(17));
      mask_and(M_B, M_A, M_B);
      sel(leading, M_B, r(leading), imm(0));
      sel(trailing, M_A, imm(0x7fffff), imm(0));
      cmp(eq_u, M_C, r(mantissa), imm(0));
      sel(result_exponent, M_C, r(leading), r(result_exponent));
      sel(result_mantissa, M_C, r(trailing), r(result_mantissa));
      shl(result_exponent, imm(23), result_exponent);
      ior(result_mantissa, r(result_exponent), result_mantissa);
      ior(dst, r(sign), result_mantissa);
   }

   void slab(unsigned slot)
   {
      const unsigned t_near = V_T;
      const unsigned t_far = V_T + 1;
      const unsigned key = V_T + 2;
      for (unsigned axis = 0; axis < 3; axis++) {
         for (unsigned plane : {V_BOX_MIN + axis, V_BOX_MAX + axis}) {
            fsub(plane, plane, V_ORIGIN + axis);
            fmul(plane, plane, V_INVERSE + axis);
         }
         cmp(ge_f, M_A, r(V_INVERSE + axis), imm(0));
         sel(t_near, M_A, r(V_BOX_MIN + axis), r(V_BOX_MAX + axis));
         sel(V_BOX_MAX + axis, M_A, r(V_BOX_MAX + axis), r(V_BOX_MIN + axis));
         mov(V_BOX_MIN + axis, r(t_near));
      }
      mov(t_near, r(V_BOX_MIN));
      mov(t_far, r(V_BOX_MAX));
      for (unsigned axis = 1; axis < 3; axis++) {
         cmp(lt_f, M_A, r(t_near), r(V_BOX_MIN + axis));
         sel(t_near, M_A, r(V_BOX_MIN + axis), r(t_near));
         cmp(lt_f, M_A, r(t_far), r(V_BOX_MAX + axis));
         sel(t_far, M_A, r(t_far), r(V_BOX_MAX + axis));
      }
      fmul(t_far, t_far, V_GROW);
      cmp(ge_f, M_A, r(t_far), r(t_near));
      cmp(ge_f, M_B, r(t_far), imm(0));
      mask_and(M_D, M_A, M_B);
      cmp(lt_f, M_A, r(t_near), r(V_EXTENT));
      mask_and(M_D, M_D, M_A);
      cmp(lt_f, M_A, r(t_near), imm(0));
      sel(V_KEY + slot, M_A, imm(0), r(t_near));
      fsub(key, t_far, t_near);
      v2(aco_opcode::v_sub_f32, key, imm(0), key);
      heuristic_flag(M_A, M_B, 1);
      sel(V_KEY + slot, M_A, r(key), r(V_KEY + slot));
      fadd(key, t_near, t_far);
      v2(aco_opcode::v_mul_f32, key, imm(f32_half), key);
      heuristic_flag(M_A, M_B, 2);
      sel(V_KEY + slot, M_A, r(key), r(V_KEY + slot));
      subtree_cull();
      sel(V_OUT + slot, M_D, r(V_CHILD), imm(invalid));
      sel(V_KEY + slot, M_D, r(V_KEY + slot), imm(f32_inf));
   }

   void flag_set(unsigned dst, uint32_t bits, unsigned flags)
   {
      iand(V_T + 2, imm(bits), flags);
      cmp(ne_u, dst, r(V_T + 2), imm(0));
   }

   void subtree_cull()
   {
      static const uint32_t opacity[2][4] = {{1, 1, 2, 64}, {2, 2, 1, 128}};
      for (const uint32_t *o : opacity) {
         iand(V_T + 2, imm(o[0]), V_CHILD_FLAGS);
         iand(V_T + 3, imm(o[1]), V_POINTER_FLAGS);
         ior(V_T + 2, r(V_T + 3), V_T + 2);
         cmp(ne_u, M_A, r(V_T + 2), imm(0));
         flag_set(M_B, o[2], V_POINTER_FLAGS);
         mask_andn2(M_A, M_A, M_B);
         flag_set(M_B, o[3], V_POINTER_FLAGS);
         mask_and(M_A, M_A, M_B);
         mask_andn2(M_D, M_D, M_A);
      }
      static const uint32_t geometry[2][2] = {{4, 256}, {8, 512}};
      for (const uint32_t *g : geometry) {
         flag_set(M_A, g[0], V_CHILD_FLAGS);
         flag_set(M_B, g[1], V_POINTER_FLAGS);
         mask_and(M_A, M_A, M_B);
         mask_andn2(M_D, M_D, M_A);
      }
   }

   void swap(unsigned base, unsigned tmp, unsigned a, unsigned b)
   {
      sel(tmp, M_A, r(base + b), r(base + a));
      sel(base + b, M_A, r(base + a), r(base + b));
      mov(base + a, r(tmp));
   }

   void sort()
   {
      set_exec(M_EXEC);
      box_mask(M_M, M_A);
      for (unsigned step = 0; step < ARRAY_SIZE(sort_network); step++) {
         unsigned a = sort_network[step][0];
         unsigned b = sort_network[step][1];
         set_exec(M_M);
         if (step >= sort_group_steps) {
            descriptor_flag(M_C, 3, 21);
            mask_and(M_C, M_C, M_M);
            set_exec(M_C);
         }
         cmp(gt_u, M_A, r(V_KEY + a), r(V_KEY + b));
         cmp(gt_f, M_B, r(V_KEY + a), r(V_KEY + b));
         heuristic_flag(M_C, M_D, 0);
         mask_and(M_A, M_A, M_C);
         mask_andn2(M_B, M_B, M_C);
         mask_or(M_A, M_A, M_B);
         descriptor_flag(M_C, 1, 20);
         iand(V_T, imm(7), V_OUT + a);
         iand(V_T + 1, imm(7), V_OUT + b);
         cmp(le_u, M_B, r(V_T), imm(node_triangle_pair));
         cmp(le_u, M_D, r(V_T + 1), imm(node_triangle_pair));
         mask_xor(M_N, M_B, M_D);
         mask_and(M_N, M_N, M_C);
         mask_andn2(M_O, M_D, M_B);
         mask_and(M_O, M_O, M_C);
         mask_andn2(M_A, M_A, M_N);
         mask_or(M_A, M_A, M_O);
         descriptor_flag(M_C, 1, 31);
         mask_and(M_A, M_A, M_C);
         swap(V_OUT, V_T, a, b);
         swap(V_KEY, V_T + 1, a, b);
      }
      set_exec(M_EXEC);
   }

   void triangles()
   {
      static const uint8_t results[4] = {V_NUMERATOR, V_DENOMINATOR, V_RESULT_Z, V_RESULT_W};
      for (unsigned triangle = 0; triangle < 2; triangle++) {
         set_exec(M_TRIANGLE);
         load(V_DATA, 16, 0);
         wait_loads();
         mov(V_TRIANGLE_FLAGS, d(15));
         if (triangle == 0) {
            mov(V_OUT + 8, d(14));
         } else {
            cmp(eq_u, M_A, r(V_KIND), imm(node_triangle_pair));
            sel(V_OUT + 9, M_A, d(14), imm(invalid));
         }
         triangle_math(triangle);
         triangle_flags(triangle);
         for (unsigned i = 0; i < 4; i++)
            mov(V_OUT + 4 * triangle + i, r(results[i]));
      }
      set_exec(M_EXEC);
   }

   void permute(unsigned source, unsigned dst, unsigned x, unsigned y)
   {
      for (unsigned axis = 0; axis < 3; axis++) {
         sel(dst + axis, y, r(source + (2 + axis) % 3), r(source + axis));
         sel(dst + axis, x, r(source + (1 + axis) % 3), r(dst + axis));
      }
   }

   void triangle_math(unsigned triangle)
   {
      static const uint8_t vertices[2][3] = {{0, 1, 2}, {1, 3, 2}};
      for (unsigned vertex = 0; vertex < 3; vertex++) {
         for (unsigned axis = 0; axis < 3; axis++)
            v2(aco_opcode::v_sub_f32, V_RELATIVE + 3 * vertex + axis, d(3 * vertices[triangle][vertex] + axis),
               V_ORIGIN + axis);
      }
      for (unsigned axis = 0; axis < 3; axis++)
         iand(V_S0 + axis, imm(f32_abs), V_DIRECTION + axis);
      cmp(ge_f, M_M, r(V_S0), r(V_S1));
      cmp(ge_f, M_N, r(V_S0), r(V_S2));
      cmp(ge_f, M_O, r(V_S1), r(V_S2));
      mask_and(M_A, M_M, M_N);
      mask_andn2(M_B, M_O, M_M);
      permute(V_DIRECTION, V_AXIS, M_A, M_B);
      for (unsigned vertex = 0; vertex < 3; vertex++)
         permute(V_RELATIVE + 3 * vertex, V_PERMUTED + 3 * vertex, M_A, M_B);
      for (unsigned vertex = 0; vertex < 3; vertex++) {
         unsigned point = V_PERMUTED + 3 * vertex;
         for (unsigned axis = 0; axis < 2; axis++) {
            unsigned projected = V_AX + 2 * vertex + axis;
            fmul(projected, point + axis, V_AXIS + 2);
            fmul(V_S0, V_AXIS + axis, point + 2);
            fsub(projected, projected, V_S0);
         }
      }
      static const uint8_t edges[3][3] = {{V_U, V_CX, V_BX}, {V_V, V_AX, V_CX}, {V_W, V_BX, V_AX}};
      for (const uint8_t *e : edges) {
         fmul(V_S0, e[1], e[2] + 1);
         fmul(V_S1, e[1] + 1, e[2]);
         fsub(e[0], V_S0, V_S1);
      }
      for (unsigned edge = 0; edge < 3; edge++)
         fmul(V_UZ + edge, V_U + edge, V_AXIS + 2);
      fadd(V_DENOMINATOR, V_UZ, V_VZ);
      fadd(V_DENOMINATOR, V_DENOMINATOR, V_WZ);
      fmul(V_NUMERATOR, V_U, V_PERMUTED + 2);
      for (unsigned edge = 1; edge < 3; edge++) {
         fmul(V_S0, V_U + edge, V_PERMUTED + 3 * edge + 2);
         fadd(V_NUMERATOR, V_NUMERATOR, V_S0);
      }

      cmp(lt_f, M_A, r(V_U), imm(0));
      cmp(gt_f, M_B, r(V_U), imm(0));
      for (unsigned edge : {V_V, V_W}) {
         cmp(lt_f, M_C, r(edge), imm(0));
         mask_or(M_A, M_A, M_C);
         cmp(gt_f, M_C, r(edge), imm(0));
         mask_or(M_B, M_B, M_C);
      }
      mask_and(M_D, M_A, M_B);
      cmp(eq_f, M_A, r(V_DENOMINATOR), imm(0));
      mask_or(M_D, M_D, M_A);
      cmp(unordered_f, M_A, r(V_NUMERATOR), r(V_NUMERATOR));
      mask_or(M_D, M_D, M_A);
      static const aco_opcode behind[2][2] = {{gt_f, lt_f}, {lt_f, gt_f}};
      for (const aco_opcode *b : behind) {
         cmp(b[0], M_A, r(V_DENOMINATOR), imm(0));
         cmp(b[1], M_B, r(V_NUMERATOR), imm(0));
         mask_and(M_A, M_A, M_B);
         mask_or(M_D, M_D, M_A);
      }
      static const uint8_t ties[3][4] = {{V_U, V_BY, V_CY, V_AY}, {V_V, V_CY, V_AY, V_BY}, {V_W, V_AY, V_BY, V_CY}};
      for (const uint8_t *t : ties) {
         cmp(eq_f, M_M, r(t[0]), imm(0));
         cmp(eq_f, M_A, r(t[1]), imm(0));
         cmp(eq_f, M_B, r(t[2]), imm(0));
         mask_and(M_N, M_A, M_B);
         cmp(gt_f, M_A, r(t[3]), imm(0));
         mask_and(M_A, M_A, M_N);
         cmp(lt_f, M_B, r(t[1]), imm(0));
         cmp(eq_f, M_C, r(t[1]), imm(0));
         cmp(gt_f, M_O, r(t[2]), imm(0));
         mask_and(M_C, M_C, M_O);
         mask_or(M_B, M_B, M_C);
         cmp(gt_f, M_O, r(V_DENOMINATOR), imm(0));
         mask_xor(M_B, M_B, M_O);
         mask_andn2(M_B, M_B, M_N);
         mask_or(M_A, M_A, M_B);
         mask_and(M_A, M_A, M_M);
         mask_or(M_D, M_D, M_A);
      }

      bfe(V_PICK_Z, V_TRIANGLE_FLAGS, 8 * triangle, 2);
      bfe(V_PICK_W, V_TRIANGLE_FLAGS, 8 * triangle + 2, 2);
      for (unsigned i = 0; i < 2; i++) {
         unsigned dst = V_RESULT_Z + i;
         unsigned pick = V_PICK_Z + i;
         mov(dst, r(V_UZ));
         cmp(eq_u, M_B, r(pick), imm(1));
         sel(dst, M_B, r(V_VZ), r(dst));
         cmp(eq_u, M_B, r(pick), imm(2));
         sel(dst, M_B, r(V_WZ), r(dst));
      }
      descriptor_flag(M_A, 3, 24);
      iadd(V_S0, imm(triangle), V_TRIANGLE_FLAGS);
      sel(V_RESULT_Z, M_A, r(V_RESULT_Z), r(V_S0));
      sel(V_S0, M_D, imm(0), imm(1));
      sel(V_RESULT_W, M_A, r(V_RESULT_W), r(V_S0));
      miss(M_D);
   }

   void miss(unsigned condition)
   {
      sel(V_NUMERATOR, condition, imm(f32_inf), r(V_NUMERATOR));
      sel(V_DENOMINATOR, condition, imm(f32_one), r(V_DENOMINATOR));
   }

   void pointer_flag(unsigned dst, uint32_t bit)
   {
      bfe(V_S0, V_POINTER_FLAGS, bit, 1);
      cmp(ne_u, dst, r(V_S0), imm(0));
   }

   void cull_if(aco_opcode compare, unsigned value, uint32_t pointer_bit)
   {
      cmp(compare, M_A, r(value), imm(0));
      pointer_flag(M_B, pointer_bit);
      mask_and(M_A, M_A, M_B);
   }

   void any_hit_bit(unsigned result)
   {
      shl(V_S0, imm(31), V_S0);
      iand(V_S1, imm(f32_abs), result);
      ior(V_S1, r(V_S0), V_S1);
      sel(result, M_A, r(V_S1), r(result));
   }

   void triangle_flags(unsigned triangle)
   {
      static const uint8_t fields[4][2] = {{V_DOUBLE_SIDED, 4}, {V_FLIP, 5}, {V_PROCEDURAL, 6}, {V_OPAQUE, 7}};
      static const uint8_t results[4] = {V_NUMERATOR, V_DENOMINATOR, V_RESULT_Z, V_RESULT_W};
      for (const uint8_t *f : fields)
         bfe(f[0], V_TRIANGLE_FLAGS, 8 * triangle + f[1], 1);
      descriptor_flag(M_A, 3, 23);
      for (const uint8_t *f : fields)
         sel(f[0], M_A, r(f[0]), imm(0));
      bfe(V_S0, V_POINTER_FLAGS, 3, 1);
      ixor(V_FLIP, r(V_S0), V_FLIP);
      bfe(V_S0, V_POINTER_FLAGS, 0, 1);
      ior(V_OPAQUE, r(V_S0), V_OPAQUE);
      bfe(V_S0, V_POINTER_FLAGS, 1, 1);
      ixor(V_S0, imm(1), V_S0);
      iand(V_OPAQUE, r(V_S0), V_OPAQUE);
      cmp(ne_u, M_A, r(V_PROCEDURAL), imm(0));
      sel(V_NUMERATOR, M_A, imm(0), r(V_NUMERATOR));
      for (unsigned i = 1; i < 4; i++)
         sel(results[i], M_A, imm(f32_one), r(results[i]));
      if (triangle == 1) {
         cmp(eq_u, M_B, r(V_KIND), imm(node_triangle_pair));
         descriptor_flag(M_A, 3, 24);
         sel(V_S0, M_A, imm(0), imm(invalid));
         sel(V_NUMERATOR, M_B, r(V_NUMERATOR), imm(f32_inf));
         sel(V_DENOMINATOR, M_B, r(V_DENOMINATOR), imm(f32_one));
         sel(V_RESULT_Z, M_B, r(V_RESULT_Z), r(V_S0));
         sel(V_RESULT_W, M_B, r(V_RESULT_W), imm(0));
      }
      cmp(eq_u, M_D, r(V_NUMERATOR), imm(f32_inf));
      cmp(eq_u, M_A, r(V_PROCEDURAL), imm(0));
      cmp(ne_u, M_B, r(V_FLIP), imm(0));
      mask_and(M_A, M_A, M_B);
      mask_andn2(M_A, M_A, M_D);
      for (unsigned result : results) {
         ixor(V_S0, imm(sign_bit), result);
         sel(result, M_A, r(V_S0), r(result));
      }
      cmp(gt_f, M_A, r(V_DENOMINATOR), imm(0));
      sel(V_FRONT, M_A, imm(1), imm(0));

      cull_if(ne_u, V_OPAQUE, 6);
      mask_or(M_D, M_D, M_A);
      cull_if(eq_u, V_OPAQUE, 7);
      mask_or(M_D, M_D, M_A);
      for (unsigned back = 0; back < 2; back++) {
         cull_if(back ? eq_u : ne_u, V_FRONT, 5 - back);
         cmp(eq_u, M_B, r(V_DOUBLE_SIDED), imm(0));
         mask_and(M_A, M_A, M_B);
         cmp(eq_u, M_B, r(V_PROCEDURAL), imm(0));
         mask_and(M_A, M_A, M_B);
         mask_or(M_D, M_D, M_A);
      }
      cull_if(eq_u, V_PROCEDURAL, 8);
      mask_or(M_D, M_D, M_A);
      cull_if(ne_u, V_PROCEDURAL, 9);
      mask_or(M_D, M_D, M_A);

      descriptor_flag(M_A, 3, 24);
      sel(V_S0, M_A, imm(0), imm(invalid));
      miss(M_D);
      sel(V_RESULT_Z, M_D, r(V_S0), r(V_RESULT_Z));
      sel(V_RESULT_W, M_D, imm(0), r(V_RESULT_W));
      descriptor_flag(M_A, 3, 23);
      mask_andn2(M_A, M_A, M_D);
      ixor(V_S0, imm(1), V_OPAQUE);
      ior(V_S0, r(V_PROCEDURAL), V_S0);
      any_hit_bit(V_RESULT_Z);
      ixor(V_S0, imm(1), V_OPAQUE);
      any_hit_bit(V_RESULT_W);
   }

   void instance()
   {
      const unsigned term = V_WORK + 16;
      const unsigned child = V_WORK + 6;
      const unsigned merged = V_WORK + 7;
      const unsigned a = V_WORK + 8;
      const unsigned b = V_WORK + 9;
      const unsigned pointer = V_WORK + 10;
      set_exec(M_INSTANCE);
      load(V_DATA, 16, 0);
      wait_loads();
      for (unsigned row = 0; row < 3; row++) {
         for (unsigned direction = 0; direction < 2; direction++) {
            unsigned input = direction ? V_DIRECTION : V_ORIGIN;
            unsigned dst = V_WORK + 3 * direction + row;
            v2(aco_opcode::v_mul_f32, dst, d(4 * row), input);
            for (unsigned axis = 1; axis < 3; axis++) {
               v2(aco_opcode::v_mul_f32, term, d(4 * row + axis), input + axis);
               fadd(dst, dst, term);
            }
            if (!direction)
               fadd(dst, dst, V_DATA + 4 * row + 3);
         }
      }
      shr(child, imm(22), V_DATA + 13);
      mov(merged, imm(0));
      for (uint32_t bit = 0; bit < 10; bit++) {
         bfe(a, child, bit, 1);
         if (bit < 2) {
            bfe(b, V_POINTER_FLAGS, 1 - bit, 1);
            ixor(b, imm(1), b);
            iand(a, r(b), a);
         }
         if (bit != 2) {
            bfe(b, V_POINTER_FLAGS, bit, 1);
            v2(bit == 3 ? aco_opcode::v_xor_b32 : aco_opcode::v_or_b32, a, r(b), a);
         }
         shl(a, imm(bit), a);
         ior(merged, r(a), merged);
      }
      mov(V_OUT + 2, d(12));
      iand(pointer, imm(0x3fffff), V_DATA + 13);
      shl(merged, imm(22), merged);
      ior(V_OUT + 3, r(pointer), merged);
      iand(V_OUT + 6, imm(0xffffff), V_DATA + 15);
      shr(a, imm(24), V_DATA + 15);
      iand(a, r(V_INSTANCE_MASK), a);
      cmp(ne_u, M_A, r(a), imm(0));
      sel(V_OUT + 7, M_A, d(14), imm(invalid));
      for (unsigned axis = 0; axis < 3; axis++) {
         bld.vop1(aco_opcode::v_mov_b32, vdef(guest(4 + axis)), r(V_WORK + axis));
         bld.vop1(aco_opcode::v_mov_b32, vdef(guest(7 + axis)), r(V_WORK + 3 + axis));
      }
      set_exec(M_EXEC);
   }

   void write_back()
   {
      set_exec(M_EXEC);
      for (unsigned i = 0; i < 10; i++)
         select(vdef(m.vdata + i), M_VALID, r(V_OUT + i), imm(invalid));
   }
};

void
lower_bvh8(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_MIMG)
      fail(ctx, inst, "malformed MIMG");
   const agc_isa_image m = inst->mimg;
   if (m.dmask != 0xf || !m.r128 || m.d16 || m.tfe || m.lwe || m.a16)
      fail(ctx, inst, "BVH8 intersection requires DMASK=0xF, R128=1 and no A16, D16, TFE or LWE");
   unsigned v = scratch_vgprs(ctx, V_COUNT);
   unsigned s = transient_sgprs(ctx, inst, AGC_ISA_BVH8_SCRATCH_SGPRS);
   bvh8 expansion = {ctx, ctx.bld, inst, m, ctx.bld.lm.size(), v, s};
   expansion.expand();
}

}
