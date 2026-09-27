/*
 * Copyright (C) 2021 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "bi_builder.h"
#include "bi_test.h"
#include "va_compiler.h"

#include <gtest/gtest.h>

class LowerConstantPair : public testing::Test {
 protected:
   void SetUp() override
   {
      mem_ctx = ralloc_context(NULL);
      b = bit_builder(mem_ctx, 10);
      inputs.fau.promote_immediates = true;
      b->shader->inputs = &inputs;
      fau.max = PAN_MAX_PUSH;
      b->shader->info.fau = &fau;
      counts = _mesa_hash_table_u64_create(mem_ctx);
      _mesa_hash_table_u64_insert(counts, 0x12345678, (void *)(uintptr_t)10);
   }

   void TearDown() override { ralloc_free(mem_ctx); }

   bi_instr *texture(uint32_t hi = 0)
   {
      return bi_tex_single_to(
         b, bi_temp(b->shader), bi_register(0), bi_imm_u32(0x12345678),
         bi_imm_u32(hi), false, BI_DIMENSION_2D, BI_REGISTER_FORMAT_F32,
         false, true, BI_VA_LOD_MODE_ZERO_LOD, false, BI_WRITE_MASK_RGBA, 2);
   }

   void lower(bi_instr *I)
   {
      va_lower_constants(b->shader, I, counts, 1);
      bi_builder before = bi_init_builder(b->shader, bi_before_instr(I));
      va_repair_fau(&before, I);
      EXPECT_TRUE(va_validate_fau(I));
   }

   void expect_pair(bi_instr *I, unsigned word, uint32_t hi = 0)
   {
      EXPECT_TRUE(bi_is_equiv(
         I->src[1], bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | (word >> 1)), false)));
      EXPECT_TRUE(bi_is_equiv(
         I->src[2], bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | (word >> 1)), true)));
      EXPECT_EQ(fau.words[word].constant, 0x12345678u);
      EXPECT_EQ(fau.words[word + 1].constant, hi);
   }

   void *mem_ctx;
   bi_builder *b;
   pan_compile_inputs inputs = {};
   pan_fau_layout fau = {};
   hash_table_u64 *counts;
};

TEST_F(LowerConstantPair, KeepZeroHighWordAndReuseWholePair)
{
   bi_instr *first = texture();
   lower(first);
   expect_pair(first, 0);
   bi_instr *second = texture();
   lower(second);
   expect_pair(second, 0);
   EXPECT_EQ(fau.count, 2u);
}

TEST_F(LowerConstantPair, DistinctOffsetsShareNoHalfSlot)
{
   bi_instr *first = texture(0x123);
   lower(first);
   bi_instr *second = texture(0x456);
   lower(second);
   expect_pair(first, 0, 0x123);
   expect_pair(second, 2, 0x456);
   EXPECT_EQ(fau.count, 4u);
}

TEST_F(LowerConstantPair, AlignAfterReservedWords)
{
   fau.reserved = fau.count = 3;
   fau.words[2].constant = 0xfeed;
   bi_instr *I = texture();
   lower(I);
   expect_pair(I, 4);
   EXPECT_EQ(fau.words[2].constant, 0xfeedu);
   EXPECT_EQ(fau.count, 6u);
}

TEST_F(LowerConstantPair, DoNotInterpretUboRelocationsAsConstants)
{
   fau.count = 2;
   fau.words[0].constant = 0x12345678;
   fau.words[1].constant = 0;
   bi_instr *I = texture();
   lower(I);
   expect_pair(I, 2);
   EXPECT_EQ(fau.count, 4u);
}

TEST_F(LowerConstantPair, FullFauUsesExistingPair)
{
   pan_fau_emit_const(&fau, 0x12345678);
   pan_fau_emit_const(&fau, 0);
   fau.count = fau.max;
   bi_instr *I = texture();
   lower(I);
   expect_pair(I, 0);
   EXPECT_EQ(fau.count, fau.max);
}

TEST_F(LowerConstantPair, NoRoomFallsBackWithoutOverflow)
{
   fau.reserved = fau.count = fau.max - 1;
   bi_instr *I = texture();
   lower(I);
   EXPECT_LE(fau.count, fau.max);
   EXPECT_FALSE(bi_is_word_equiv(I->src[2], bi_fau(BIR_FAU_UNIFORM, true)));
}

TEST_F(LowerConstantPair, DisabledPromotionAllocatesNothing)
{
   inputs.fau.promote_immediates = false;
   bi_instr *I = texture();
   /* Match the disabled-promotion threshold supplied by the compiler. */
   va_lower_constants(b->shader, I, counts, UINT32_MAX);
   EXPECT_EQ(fau.count, 0u);
   EXPECT_EQ(I->src[1].type, BI_INDEX_NORMAL);
}

TEST_F(LowerConstantPair, DynamicHighWordIsNotPromotedAsAConstant)
{
   bi_instr *I = texture();
   I->src[2] = bi_register(5);
   lower(I);
   EXPECT_TRUE(bi_is_equiv(I->src[2], bi_register(5)));
   EXPECT_EQ(fau.count, 1u);
}

static inline void
add_imm(bi_context *ctx)
{
   struct hash_table_u64 *stats = _mesa_hash_table_u64_create(ctx);
   bi_foreach_instr_global(ctx, I) {
      va_lower_constants(ctx, I, stats, UINT32_MAX);
   }
   _mesa_hash_table_u64_destroy(stats);
}

#define CASE(instr, expected) INSTRUCTION_CASE(instr, expected, add_imm)

class LowerConstants : public testing::Test {
 protected:
   LowerConstants()
   {
      mem_ctx = ralloc_context(NULL);
      arch = 9;
   }

   ~LowerConstants()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
   unsigned arch;
};

TEST_F(LowerConstants, Float32)
{
   CASE(bi_fadd_f32_to(b, bi_register(0), bi_register(0), bi_imm_f32(0.0)),
        bi_fadd_f32_to(b, bi_register(0), bi_register(0), va_lut(0)));

   CASE(bi_fadd_f32_to(b, bi_register(0), bi_register(0), bi_imm_f32(1.0)),
        bi_fadd_f32_to(b, bi_register(0), bi_register(0), va_lut(16)));

   CASE(bi_fadd_f32_to(b, bi_register(0), bi_register(0), bi_imm_f32(0.1)),
        bi_fadd_f32_to(b, bi_register(0), bi_register(0), va_lut(17)));
}

TEST_F(LowerConstants, WidenFloat16)
{
   CASE(bi_fadd_f32_to(b, bi_register(0), bi_register(0), bi_imm_f32(0.5)),
        bi_fadd_f32_to(b, bi_register(0), bi_register(0),
                       bi_half(va_lut(26), 1)));

   CASE(bi_fadd_f32_to(b, bi_register(0), bi_register(0), bi_imm_f32(255.0)),
        bi_fadd_f32_to(b, bi_register(0), bi_register(0),
                       bi_half(va_lut(23), 0)));

   CASE(bi_fadd_f32_to(b, bi_register(0), bi_register(0), bi_imm_f32(256.0)),
        bi_fadd_f32_to(b, bi_register(0), bi_register(0),
                       bi_half(va_lut(23), 1)));

   CASE(bi_fadd_f32_to(b, bi_register(0), bi_register(0), bi_imm_f32(8.0)),
        bi_fadd_f32_to(b, bi_register(0), bi_register(0),
                       bi_half(va_lut(30), 1)));
}

TEST_F(LowerConstants, ReplicateFloat16)
{
   CASE(bi_fadd_v2f16_to(b, bi_register(0), bi_register(0), bi_imm_f16(255.0)),
        bi_fadd_v2f16_to(b, bi_register(0), bi_register(0),
                         bi_half(va_lut(23), 0)));

   CASE(bi_fadd_v2f16_to(b, bi_register(0), bi_register(0), bi_imm_f16(4.0)),
        bi_fadd_v2f16_to(b, bi_register(0), bi_register(0),
                         bi_half(va_lut(29), 1)));
}

TEST_F(LowerConstants, NegateFloat32)
{
   CASE(bi_fadd_f32_to(b, bi_register(0), bi_register(0), bi_imm_f32(-1.0)),
        bi_fadd_f32_to(b, bi_register(0), bi_register(0), bi_neg(va_lut(16))));

   CASE(bi_fadd_f32_to(b, bi_register(0), bi_register(0), bi_imm_f32(-255.0)),
        bi_fadd_f32_to(b, bi_register(0), bi_register(0),
                       bi_neg(bi_half(va_lut(23), 0))));
}

TEST_F(LowerConstants, NegateReplicateFloat16)
{
   CASE(bi_fadd_v2f16_to(b, bi_register(0), bi_register(0), bi_imm_f16(-255.0)),
        bi_fadd_v2f16_to(b, bi_register(0), bi_register(0),
                         bi_neg(bi_half(va_lut(23), 0))));
}

TEST_F(LowerConstants, NegateVec2Float16)
{
   CASE(
      bi_fadd_v2f16_to(b, bi_register(0), bi_register(0),
                       bi_imm_u32(0xBC008000)),
      bi_fadd_v2f16_to(b, bi_register(0), bi_register(0), bi_neg(va_lut(27))));
}

TEST_F(LowerConstants, Int8InInt32)
{
   CASE(bi_lshift_or_i32(b, bi_register(0), bi_imm_u32(0), bi_imm_u8(6)),
        bi_lshift_or_i32(b, bi_register(0), va_lut(0), bi_byte(va_lut(9), 2)));

   CASE(bi_lshift_or_i32(b, bi_register(0), bi_imm_u32(0), bi_imm_u8(-2)),
        bi_lshift_or_i32(b, bi_register(0), va_lut(0), bi_byte(va_lut(3), 0)));
}

TEST_F(LowerConstants, ZeroExtendForUnsigned)
{
   CASE(bi_icmp_and_u32_to(b, bi_register(0), bi_register(0), bi_imm_u32(0xFF),
                           bi_register(0), BI_CMPF_LT, BI_RESULT_TYPE_I1),
        bi_icmp_and_u32_to(b, bi_register(0), bi_register(0),
                           bi_byte(va_lut(1), 0), bi_register(0), BI_CMPF_LT,
                           BI_RESULT_TYPE_I1));

   CASE(
      bi_icmp_and_u32_to(b, bi_register(0), bi_register(0), bi_imm_u32(0xFFFF),
                         bi_register(0), BI_CMPF_LT, BI_RESULT_TYPE_I1),
      bi_icmp_and_u32_to(b, bi_register(0), bi_register(0),
                         bi_half(va_lut(1), 0), bi_register(0), BI_CMPF_LT,
                         BI_RESULT_TYPE_I1));
}

TEST_F(LowerConstants, SignExtendPositiveForSigned)
{
   CASE(bi_icmp_and_s32_to(b, bi_register(0), bi_register(0), bi_imm_u32(0x7F),
                           bi_register(0), BI_CMPF_LT, BI_RESULT_TYPE_I1),
        bi_icmp_and_s32_to(b, bi_register(0), bi_register(0),
                           bi_byte(va_lut(2), 3), bi_register(0), BI_CMPF_LT,
                           BI_RESULT_TYPE_I1));

   CASE(
      bi_icmp_and_s32_to(b, bi_register(0), bi_register(0), bi_imm_u32(0x7FFF),
                         bi_register(0), BI_CMPF_LT, BI_RESULT_TYPE_I1),
      bi_icmp_and_s32_to(b, bi_register(0), bi_register(0),
                         bi_half(va_lut(2), 1), bi_register(0), BI_CMPF_LT,
                         BI_RESULT_TYPE_I1));
}

TEST_F(LowerConstants, SignExtendNegativeForSigned)
{
   CASE(bi_icmp_and_s32_to(b, bi_register(0), bi_register(0),
                           bi_imm_u32(0xFFFFFFF8), bi_register(0), BI_CMPF_LT,
                           BI_RESULT_TYPE_I1),
        bi_icmp_and_s32_to(b, bi_register(0), bi_register(0),
                           bi_byte(va_lut(23), 0), bi_register(0), BI_CMPF_LT,
                           BI_RESULT_TYPE_I1));

   CASE(bi_icmp_and_s32_to(b, bi_register(0), bi_register(0),
                           bi_imm_u32(0xFFFFFAFC), bi_register(0), BI_CMPF_LT,
                           BI_RESULT_TYPE_I1),
        bi_icmp_and_s32_to(b, bi_register(0), bi_register(0),
                           bi_half(va_lut(3), 1), bi_register(0), BI_CMPF_LT,
                           BI_RESULT_TYPE_I1));
}

TEST_F(LowerConstants, DontZeroExtendForSigned)
{
   CASE(bi_icmp_and_s32_to(b, bi_register(0), bi_register(0), bi_imm_u32(0xFF),
                           bi_register(0), BI_CMPF_LT, BI_RESULT_TYPE_I1),
        bi_icmp_and_s32_to(b, bi_register(0), bi_register(0),
                           bi_iadd_imm_i32(b, va_lut(0), 0xFF), bi_register(0),
                           BI_CMPF_LT, BI_RESULT_TYPE_I1));

   CASE(
      bi_icmp_and_s32_to(b, bi_register(0), bi_register(0), bi_imm_u32(0xFFFF),
                         bi_register(0), BI_CMPF_LT, BI_RESULT_TYPE_I1),
      bi_icmp_and_s32_to(b, bi_register(0), bi_register(0),
                         bi_iadd_imm_i32(b, va_lut(0), 0xFFFF), bi_register(0),
                         BI_CMPF_LT, BI_RESULT_TYPE_I1));
}

TEST_F(LowerConstants, DontZeroExtendNegative)
{
   CASE(bi_icmp_and_u32_to(b, bi_register(0), bi_register(0),
                           bi_imm_u32(0xFFFFFFF8), bi_register(0), BI_CMPF_LT,
                           BI_RESULT_TYPE_I1),
        bi_icmp_and_u32_to(b, bi_register(0), bi_register(0),
                           bi_iadd_imm_i32(b, va_lut(0), 0xFFFFFFF8),
                           bi_register(0), BI_CMPF_LT, BI_RESULT_TYPE_I1));

   CASE(bi_icmp_and_u32_to(b, bi_register(0), bi_register(0),
                           bi_imm_u32(0xFFFFFAFC), bi_register(0), BI_CMPF_LT,
                           BI_RESULT_TYPE_I1),
        bi_icmp_and_u32_to(b, bi_register(0), bi_register(0),
                           bi_iadd_imm_i32(b, va_lut(0), 0xFFFFFAFC),
                           bi_register(0), BI_CMPF_LT, BI_RESULT_TYPE_I1));
}

TEST_F(LowerConstants, HandleTrickyNegativesFP16)
{
   CASE(
      bi_fadd_v2f16_to(b, bi_register(0), bi_register(0), bi_imm_f16(-57216.0)),
      bi_fadd_v2f16_to(b, bi_register(0), bi_register(0),
                       bi_half(va_lut(3), 1)));

   CASE(
      bi_fadd_v2f16_to(b, bi_register(0), bi_register(0), bi_imm_f16(57216.0)),
      bi_fadd_v2f16_to(b, bi_register(0), bi_register(0),
                       bi_neg(bi_half(va_lut(3), 1))));
}

TEST_F(LowerConstants, DontWidenConstantsOfVectorOps)
{
   CASE(bi_icmp_or_v2u16_to(b, bi_register(0), bi_register(0),
                            bi_imm_u32(0x00003F80), bi_register(0), BI_CMPF_EQ,
                            BI_RESULT_TYPE_M1),
        bi_icmp_or_v2u16_to(b, bi_register(0), bi_register(0),
                            bi_iadd_imm_i32(b, va_lut(0), 0x00003F80),
                            bi_register(0), BI_CMPF_EQ, BI_RESULT_TYPE_M1));

   CASE(bi_icmp_or_v4u8_to(b, bi_register(0), bi_register(0),
                           bi_imm_u32(0x00000005), bi_register(0), BI_CMPF_EQ,
                           BI_RESULT_TYPE_M1),
        bi_icmp_or_v4u8_to(b, bi_register(0), bi_register(0),
                           bi_iadd_imm_i32(b, va_lut(0), 0x00000005),
                           bi_register(0), BI_CMPF_EQ, BI_RESULT_TYPE_M1));
}

TEST_F(LowerConstants, MaintainMkvecRestrictedSwizzles)
{
   CASE(bi_mkvec_v2i8_to(b, bi_register(0), bi_register(0), bi_imm_u8(0),
                         bi_imm_u32(0)),
        bi_mkvec_v2i8_to(b, bi_register(0), bi_register(0),
                         bi_byte(va_lut(0), 0), va_lut(0)));

   CASE(bi_mkvec_v2i8_to(b, bi_register(0), bi_register(0), bi_imm_u8(14),
                         bi_imm_u32(0)),
        bi_mkvec_v2i8_to(b, bi_register(0), bi_register(0),
                         bi_byte(va_lut(11), 2), va_lut(0)));
}
