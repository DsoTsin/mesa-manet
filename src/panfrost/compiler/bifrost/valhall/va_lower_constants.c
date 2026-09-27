/*
 * Copyright (C) 2025 Arm Ltd.
 * Copyright (C) 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "bi_builder.h"
#include "va_compiler.h"
#include "valhall.h"

static int
va_compare_constant_count(const void *a, const void *b)
{
   uint32_t va = *(const uint32_t *)a;
   uint32_t vb = *(const uint32_t *)b;
   return (va < vb) - (va > vb);
}

uint32_t
va_min_fau_count(struct hash_table_u64 *counts, unsigned capacity)
{
   unsigned count = _mesa_hash_table_u64_num_entries(counts);
   if (!capacity || !count)
      return UINT32_MAX;

   uint32_t *sorted = malloc(sizeof(*sorted) * count);
   if (!sorted)
      return UINT32_MAX;

   unsigned idx = 0;
   hash_table_u64_foreach(counts, entry) {
      sorted[idx++] = (uintptr_t)entry.data;
   }

   qsort(sorted, count, sizeof(*sorted), va_compare_constant_count);
   uint32_t threshold = sorted[MIN2(capacity, count) - 1];
   free(sorted);
   return threshold;
}

/* Only some special immediates are available, as specified in the Table of
 * Immediates in the specification. Other immediates must be lowered, either to
 * uniforms or to moves.
 */

static bi_index
va_mov_imm(bi_builder *b, uint32_t imm)
{
   bi_index zero = bi_fau(BIR_FAU_IMMEDIATE | 0, false);
   return bi_iadd_imm_i32(b, zero, imm);
}

static bi_index
va_lut_index_32(uint32_t imm)
{
   for (unsigned i = 0; i < ARRAY_SIZE(valhall_immediates); ++i) {
      if (valhall_immediates[i] == imm)
         return va_lut(i);
   }

   return bi_null();
}

static bi_index
va_lut_index_16(uint16_t imm)
{
   uint16_t *arr16 = (uint16_t *)valhall_immediates;

   for (unsigned i = 0; i < (2 * ARRAY_SIZE(valhall_immediates)); ++i) {
      if (arr16[i] == imm)
         return bi_half(va_lut(i >> 1), i & 1);
   }

   return bi_null();
}

UNUSED static bi_index
va_lut_index_8(uint8_t imm)
{
   uint8_t *arr8 = (uint8_t *)valhall_immediates;

   for (unsigned i = 0; i < (4 * ARRAY_SIZE(valhall_immediates)); ++i) {
      if (arr8[i] == imm)
         return bi_byte(va_lut(i >> 2), i & 3);
   }

   return bi_null();
}

static bi_index
va_demote_constant_fp16(uint32_t value)
{
   uint16_t fp16 = _mesa_float_to_half(uif(value));

   /* Only convert if it is exact */
   if (fui(_mesa_half_to_float(fp16)) == value)
      return va_lut_index_16(fp16);
   else
      return bi_null();
}

/*
 * Test if a 32-bit word arises as a sign or zero extension of some 8/16-bit
 * value.
 */
static bool
is_extension_of_8(uint32_t x, bool is_signed)
{
   if (is_signed)
      return (x <= INT8_MAX) || ((x >> 7) == BITFIELD_MASK(24 + 1));
   else
      return (x <= UINT8_MAX);
}

static bool
is_extension_of_16(uint32_t x, bool is_signed)
{
   if (is_signed)
      return (x <= INT16_MAX) || ((x >> 15) == BITFIELD_MASK(16 + 1));
   else
      return (x <= UINT16_MAX);
}

static bi_index
va_move_const_to_fau(bi_builder *b, uint32_t value)
{
   struct pan_fau_layout *fau = b->shader->info.fau;

   int idx = pan_lookup_pushed_imm(fau, value);
   if (idx >= 0) {
      return bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | (idx >> 1)), idx & 1);
   }

   if (pan_fau_available(fau) > 0) {
      const unsigned int idx = pan_fau_emit_const(fau, value);
      return bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | (idx >> 1)), idx & 1);
   }

   return bi_null();
}

static bi_index
va_lookup_constant(uint32_t value, struct va_src_info info, bool is_signed)
{
   /* Try the constant as-is */
   {
      bi_index lut = va_lut_index_32(value);
      if (!bi_is_null(lut))
         return lut;

      /* ...or negated as a FP32 constant */
      if (info.absneg && info.size == VA_SIZE_32) {
         lut = bi_neg(va_lut_index_32(fui(-uif(value))));
         if (!bi_is_null(lut))
            return lut;
      }

      /* ...or negated as a FP16 constant */
      if (info.absneg && info.size == VA_SIZE_16) {
         lut = bi_neg(va_lut_index_32(value ^ 0x80008000));
         if (!bi_is_null(lut))
            return lut;
      }
   }

   /* Try using a single half of a FP16 constant */
   bool replicated_halves = (value & 0xFFFF) == (value >> 16);
   if (info.swizzle && info.size == VA_SIZE_16 && replicated_halves) {
      bi_index lut = va_lut_index_16(value & 0xFFFF);
      if (!bi_is_null(lut))
         return lut;

      /* ...possibly negated */
      if (info.absneg) {
         lut = bi_neg(va_lut_index_16((value & 0xFFFF) ^ 0x8000));
         if (!bi_is_null(lut))
            return lut;
      }
   }

   /* The widening swizzle returned below selects one sub-word of a LUT entry.
    * On a 32-bit instruction the hardware zero/sign-extends that sub-word into
    * the full 32-bit lane. The same encoded swizzle has a different meaning on
    * an 8/16-bit vector instruction: it replicates the sub-word across all
    * lanes (for example .h0 is BI_SWIZZLE_H00, half 0 broadcast to both lanes),
    * so a constant like (0x3f80, 0x0000) would turn into (0x3f80, 0x3f80).
    * Only widen for 32-bit instructions. The replicated-halves case for 16-bit
    * is handled separately via info.swizzle above.
    */
   bool can_widen = info.widen && info.size == VA_SIZE_32;

   /* Try extending a byte */
   if ((can_widen || info.lanes || info.lane) &&
       is_extension_of_8(value, is_signed)) {

      bi_index lut = va_lut_index_8(value & 0xFF);
      if (!bi_is_null(lut))
         return lut;
   }

   /* Try extending a halfword */
   if (can_widen && is_extension_of_16(value, is_signed)) {

      bi_index lut = va_lut_index_16(value & 0xFFFF);
      if (!bi_is_null(lut))
         return lut;
   }

   /* Try demoting the constant to FP16 */
   if (info.swizzle && info.size == VA_SIZE_32) {
      bi_index lut = va_demote_constant_fp16(value);
      if (!bi_is_null(lut))
         return lut;

      if (info.absneg) {
         bi_index lut = bi_neg(va_demote_constant_fp16(fui(-uif(value))));
         if (!bi_is_null(lut))
            return lut;
      }
   }

   return bi_null();
}

static bi_index
va_resolve_constant(bi_builder *b, uint32_t value, struct va_src_info info,
                    bool is_signed, bool staging, bool try_move_to_fau)
{
   if (!staging) {
      bi_index lut = va_lookup_constant(value, info, is_signed);
      if (!bi_is_null(lut))
         return lut;
   }

   if (!staging && try_move_to_fau) {
      bi_index c = va_move_const_to_fau(b, value);
      if (!bi_is_null(c))
         return c;
   }

   return va_mov_imm(b, value);
}

static uint32_t
va_resolve_swizzles(bi_context *ctx, bi_instr *I, unsigned s)
{
   struct va_src_info info = va_src_info(I->op, s);
   uint32_t value = I->src[s].value;
   enum bi_swizzle swz = I->src[s].swizzle;

   /* Resolve any swizzle, keeping in mind the different interpretations
    * swizzles in different contexts.
    */
   if (info.size == VA_SIZE_32) {
      /* Extracting a half from the 32-bit value */
      if (swz == BI_SWIZZLE_H00)
         value = (value & 0xFFFF);
      else if (swz == BI_SWIZZLE_H11)
         value = (value >> 16);
      else
         assert(swz == BI_SWIZZLE_H01);

      /* FP16 -> FP32 */
      if (info.swizzle && swz != BI_SWIZZLE_H01)
         value = fui(_mesa_half_to_float(value));
   } else if (info.size == VA_SIZE_16) {
      assert(swz >= BI_SWIZZLE_H00 && swz <= BI_SWIZZLE_H11);
      value = bi_apply_swizzle(value, swz);
   } else if (info.size == VA_SIZE_8 && (info.lane || info.lanes)) {
      /* 8-bit extract */
      unsigned chan = (swz - BI_SWIZZLE_B0000);
      assert(chan < 4);

      value = (value >> (8 * chan)) & 0xFF;
   } else {
      /* TODO: Any other special handling? */
      value = bi_apply_swizzle(value, swz);
   }

   return value;
}

/* A split 64-bit source is encoded with one aligned FAU slot. Looking up its
 * words independently can put them in different slots, forcing register moves
 * even when both words are constants. Keep the pair together, including words
 * that could otherwise use the immediate LUT.
 */
static bool
va_lower_constant_pair(bi_context *ctx, bi_instr *I, unsigned s,
                       struct hash_table_u64 *counts, uint32_t min_fau_count)
{
   if (!ctx->inputs->fau.promote_immediates || s >= 4 || s + 1 >= I->nr_srcs ||
       va_src_info(I->op, s).size != VA_SIZE_64 ||
       bi_count_read_registers(I, s) != 1)
      return false;

   uint32_t lo = I->src[s].value, hi = I->src[s + 1].value;
   if (!bi_is_equiv(I->src[s], bi_imm_u32(lo)) ||
       !bi_is_equiv(I->src[s + 1], bi_imm_u32(hi)))
      return false;

   struct pan_fau_layout *fau = ctx->info.fau;
   unsigned idx;
   for (idx = ALIGN_POT(fau->reserved, 2); idx + 1 < fau->count; idx += 2) {
      if (BITSET_TEST(fau->is_const, idx) &&
          BITSET_TEST(fau->is_const, idx + 1) &&
          fau->words[idx].constant == lo &&
          fau->words[idx + 1].constant == hi)
         break;
   }

   if (idx + 1 >= fau->count) {
      uint32_t lo_count = (uintptr_t)_mesa_hash_table_u64_search(counts, lo);
      uint32_t hi_count = (uintptr_t)_mesa_hash_table_u64_search(counts, hi);
      if (MAX2(lo_count, hi_count) < min_fau_count ||
          pan_fau_available(fau) < 2 + (fau->count & 1))
         return false;

      if (fau->count & 1)
         pan_fau_emit_const(fau, 0);

      idx = pan_fau_emit_const(fau, lo);
      pan_fau_emit_const(fau, hi);
   }

   I->src[s] = bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | (idx >> 1)), false);
   I->src[s + 1] = bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | (idx >> 1)), true);
   return true;
}

void
va_lower_constants(bi_context *ctx, bi_instr *I, struct hash_table_u64 *counts, uint32_t min_fau_count)
{
   bi_builder b = bi_init_builder(ctx, bi_before_instr(I));

   bi_foreach_src(I, s) {
      if (va_lower_constant_pair(ctx, I, s, counts, min_fau_count)) {
         ++s;
         continue;
      }

      if (I->src[s].type == BI_INDEX_CONSTANT) {
         /* abs(#c) is pointless, but -#c occurs in transcendental sequences */
         assert(!I->src[s].abs && "redundant .abs modifier");

         bool is_signed = valhall_opcodes[I->op].is_signed;
         bool staging = (s < valhall_opcodes[I->op].nr_staging_srcs);
         struct va_src_info info = va_src_info(I->op, s);
         const uint32_t value = va_resolve_swizzles(ctx, I, s);

         const uint32_t count = (uintptr_t)_mesa_hash_table_u64_search(counts, value);
         const bool move_to_fau = count >= min_fau_count;

         bi_index cons =
            va_resolve_constant(&b, value, info, is_signed, staging, move_to_fau);
         cons.neg ^= I->src[s].neg;
         I->src[s] = cons;

         /* If we're selecting a single lane, we should return a single lane
          * to ensure the result is encodeable. By convention, applying the
          * lane select puts the desired constant (at least) in the bottom
          * byte/half, so we can always select the bottom byte/half.
          */
         if ((info.lane || info.lanes || info.halfswizzle) &&
             I->src[s].swizzle == BI_SWIZZLE_H01) {
            assert(info.size == VA_SIZE_8 || info.size == VA_SIZE_16);
            if (info.size == VA_SIZE_8)
               I->src[s] = bi_byte(I->src[s], 0);
            else if (info.size == VA_SIZE_16)
               I->src[s] = bi_half(I->src[s], false);
         }
      }
   }
}

void
va_count_constants(bi_context *ctx, bi_instr *I, struct hash_table_u64 *counts)
{
   bi_foreach_src(I, s) {
      if (I->src[s].type != BI_INDEX_CONSTANT)
         continue;

      const bool staging = (s < valhall_opcodes[I->op].nr_staging_srcs);
      if (staging)
         continue;

      bool is_signed = valhall_opcodes[I->op].is_signed;
      struct va_src_info info = va_src_info(I->op, s);
      uint32_t value = va_resolve_swizzles(ctx, I, s);

      bi_index cons = va_lookup_constant(value, info, is_signed);

      const bool can_lut = !bi_is_null(cons);

      /* We want to move constants that can't be created from built-in
       * constants into the FAU if they are not staging register sources. */
      if (!can_lut) {
         uint32_t count = (uintptr_t)_mesa_hash_table_u64_search(counts, value);
         _mesa_hash_table_u64_insert(counts, value, (void*)(uintptr_t)(count + 1));
      }
   }
}
