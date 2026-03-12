/*
 * Copyright (C) 2021 Collabora, Ltd.
 * Copyright (C) 2026 Arm Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "bi_builder.h"
#include "bi_test.h"
#include "va_compiler.h"

#include <gtest/gtest.h>

#define CASE_ARCH(instr, arch, expected)                                       \
   do {                                                                        \
      uint64_t _value = va_pack_instr(instr, arch);                            \
      if (_value != expected) {                                                \
         fprintf(stderr, "Got %" PRIx64 ", expected %" PRIx64 "\n", _value,    \
                 (uint64_t)expected);                                          \
         bi_print_instr(instr, stderr);                                        \
         fprintf(stderr, "\n");                                                \
         ADD_FAILURE();                                                        \
      }                                                                        \
   } while (0)

class ValhallPacking : public testing::Test {
 protected:
   ValhallPacking()
   {
      mem_ctx = ralloc_context(NULL);
      b = bit_builder(mem_ctx);

      zero = bi_fau((enum bir_fau)(BIR_FAU_IMMEDIATE | 0), false);
      one = bi_fau((enum bir_fau)(BIR_FAU_IMMEDIATE | 8), false);
      n4567 = bi_fau((enum bir_fau)(BIR_FAU_IMMEDIATE | 4), true);
   }

   ~ValhallPacking()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
   bi_builder *b;
   bi_index zero, one, n4567;
};

TEST_F(ValhallPacking, Moves)
{
   bi_instr *I = bi_mov_i32_to(b, bi_register(1), bi_register(2));
   CASE_ARCH(I, 10, 0x0091c10000000002ULL);
   CASE_ARCH(I, 15, 0x0060010000200002ULL);

   I = bi_mov_i32_to(b, bi_register(1),
                     bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | 5), false));
   CASE_ARCH(I, 10, 0x0091c1000000008aULL);
   CASE_ARCH(I, 15, 0x006101000020000aULL);
}

TEST_F(ValhallPacking, Fadd)
{
   bi_instr *I =
      bi_fadd_f32_to(b, bi_register(0), bi_register(1), bi_register(2));
   CASE_ARCH(I, 10, 0x00a4c00000000201ULL);
   CASE_ARCH(I, 15, 0x00f0000000000201ULL);

   I =
      bi_fadd_f32_to(b, bi_register(0), bi_register(1), bi_abs(bi_register(2)));
   CASE_ARCH(I, 10, 0x00a4c02000000201ULL);
   CASE_ARCH(I, 15, 0x00f0002000000201ULL);

   I =
      bi_fadd_f32_to(b, bi_register(0), bi_register(1), bi_neg(bi_register(2)));
   CASE_ARCH(I, 10, 0x00a4c01000000201ULL);
   CASE_ARCH(I, 15, 0x00f0001000000201ULL);

   I = bi_fadd_v2f16_to(b, bi_register(0),
                        bi_swz_16(bi_register(1), false, false),
                        bi_swz_16(bi_register(0), true, true));
   CASE_ARCH(I, 10, 0x00a5c0000c000001ULL);
   CASE_ARCH(I, 15, 0x00f400000c000001ULL);

   I = bi_fadd_v2f16_to(b, bi_register(0), bi_register(1), bi_register(0));
   CASE_ARCH(I, 10, 0x00a5c00028000001ULL);
   CASE_ARCH(I, 15, 0x00f4000028000001ULL);

   I = bi_fadd_v2f16_to(b, bi_register(0), bi_register(1),
                        bi_swz_16(bi_register(0), true, false));
   CASE_ARCH(I, 10, 0x00a5c00024000001ULL);
   CASE_ARCH(I, 15, 0x00f4000024000001ULL);

   I = bi_fadd_v2f16_to(b, bi_register(0), bi_discard(bi_abs(bi_register(0))),
                        bi_neg(zero));
   CASE_ARCH(I, 10, 0x00a5c0902800c040ULL);
   CASE_ARCH(I, 15, 0x00f600902800c080ULL);

   I = bi_fadd_f32_to(b, bi_register(0), bi_register(1), zero);
   CASE_ARCH(I, 10, 0x00a4c0000000c001ULL);
   CASE_ARCH(I, 15, 0x00f200000000c001ULL);

   I = bi_fadd_f32_to(b, bi_register(0), bi_register(1), bi_neg(zero));
   CASE_ARCH(I, 10, 0x00a4c0100000c001ULL);
   CASE_ARCH(I, 15, 0x00f200100000c001ULL);

   I = bi_fadd_f32_to(b, bi_register(0), bi_register(1),
                      bi_half(bi_register(0), true));
   CASE_ARCH(I, 10, 0x00a4c00008000001ULL);
   CASE_ARCH(I, 15, 0x00f0000008000001ULL);

   I = bi_fadd_f32_to(b, bi_register(0), bi_register(1),
                      bi_half(bi_register(0), false));
   CASE_ARCH(I, 10, 0x00a4c00004000001ULL);
   CASE_ARCH(I, 15, 0x00f0000004000001ULL);
}

TEST_F(ValhallPacking, Clper)
{
   bi_instr *I = bi_clper_i32_to(b, bi_register(0), bi_register(0),
                                 bi_byte(n4567, 0), BI_INACTIVE_RESULT_F1,
                                 BI_LANE_OP_NONE, BI_SUBGROUP_SUBGROUP16);
   CASE_ARCH(I, 10, 0x00a0c030128fc900);
   CASE_ARCH(I, 15, 0x00e20030028fc900);
}

TEST_F(ValhallPacking, Clamps)
{
   bi_instr *I = bi_fadd_f32_to(b, bi_register(0), bi_register(1),
                                bi_neg(bi_abs(bi_register(2))));
   CASE_ARCH(I, 10, 0x00a4c03000000201ULL);
   CASE_ARCH(I, 15, 0x00f0003000000201ULL);

   I->clamp = BI_CLAMP_CLAMP_M1_1;
   CASE_ARCH(I, 10, 0x00a4c03200000201ULL);
   CASE_ARCH(I, 15, 0x00f0003080000201ULL);
}

TEST_F(ValhallPacking, Misc)
{
   bi_instr *I = bi_fma_f32_to(
      b, bi_register(1), bi_discard(bi_register(1)),
      bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | 4), false), bi_neg(zero));
   CASE_ARCH(I, 10, 0x00b2c10400c08841ULL);
   CASE_ARCH(I, 15, 0x0166010400c00881ULL);

   I = bi_fround_f32_to(b, bi_register(2), bi_discard(bi_neg(bi_register(2))),
                        BI_ROUND_RTN);
   CASE_ARCH(I, 10, 0x0090c240800d0042ULL);
   CASE_ARCH(I, 15, 0x00600242004d0082ULL);

   I = bi_fround_v2f16_to(b, bi_half(bi_register(0), false), bi_register(0),
                          BI_ROUND_RTN);
   CASE_ARCH(I, 10, 0x00904000a00f0000ULL);
   /* Removed on v11 */

   I = bi_fround_v2f16_to(b, bi_half(bi_register(0), false),
                          bi_swz_16(bi_register(1), true, false), BI_ROUND_RTN);
   CASE_ARCH(I, 10, 0x00904000900f0001ULL);
   /* Removed on v11 */
}

TEST_F(ValhallPacking, FaddImm)
{
   bi_instr *I = bi_fadd_imm_f32_to(b, bi_register(2),
                                    bi_discard(bi_register(2)), 0x4847C6C0);
   CASE_ARCH(I, 10, 0x0114C24847C6C042ULL);
   CASE_ARCH(I, 15, 0x0064024847c6c082ULL);

   I = bi_fadd_imm_v2f16_to(b, bi_register(2), bi_discard(bi_register(2)),
                            0x70AC6784);
   CASE_ARCH(I, 10, 0x0115C270AC678442ULL);
   CASE_ARCH(I, 15, 0x00620270ac678482ULL);
}

TEST_F(ValhallPacking, Comparions)
{
   bi_instr *I = bi_icmp_or_v2s16_to(
      b, bi_register(2), bi_discard(bi_swz_16(bi_register(3), true, false)),
      bi_discard(bi_swz_16(bi_register(2), true, false)), zero, BI_CMPF_GT,
      BI_RESULT_TYPE_M1);
   CASE_ARCH(I, 10, 0x00f9c21184c04243);
   CASE_ARCH(I, 15, 0x01e40212c6c08283);

   I = bi_fcmp_or_v2f16_to(b, bi_register(2),
                           bi_discard(bi_swz_16(bi_register(3), true, false)),
                           bi_discard(bi_swz_16(bi_register(2), false, false)),
                           zero, BI_CMPF_GT, BI_RESULT_TYPE_M1);
   CASE_ARCH(I, 10, 0x00f5c20190c04243);
   CASE_ARCH(I, 15, 0x01e4020352c08283);
}

TEST_F(ValhallPacking, Conversions)
{
   bi_instr *I =
      bi_v2s16_to_v2f16_to(b, bi_register(2), bi_discard(bi_register(2)));
   CASE_ARCH(I, 10, 0x0090c22000070042);
   /* Removed on v11 */
}

TEST_F(ValhallPacking, BranchzI16)
{
   bi_instr *I =
      bi_branchz_i16(b, bi_half(bi_register(2), false), bi_null(), BI_CMPF_EQ);
   I->branch_offset = 1;
   CASE_ARCH(I, 10, 0x001fc03000000102);
   CASE_ARCH(I, 15, 0x02b8003000000102);
}

TEST_F(ValhallPacking, BranchzI16Backwards)
{
   bi_instr *I = bi_branchz_i16(b, zero, bi_null(), BI_CMPF_EQ);
   I->branch_offset = -8;
   CASE_ARCH(I, 10, 0x001fc017fffff8c0);
   CASE_ARCH(I, 15, 0x02b90017fffff8c0);
}

TEST_F(ValhallPacking, Blend)
{
   bi_instr *I =
      bi_blend_to(b, bi_null(), bi_register(0), bi_register(60),
                  bi_fau(BIR_FAU_BLEND_0, false), bi_fau(BIR_FAU_BLEND_0, true),
                  bi_null(), BI_REGISTER_FORMAT_F16, 2, 0);
   CASE_ARCH(I, 10, 0x007f4004333c00f0);
   CASE_ARCH(I, 15, 0x031b0082333c00f0);
}

TEST_F(ValhallPacking, Mux)
{
   bi_instr *I = bi_mux_i32_to(
      b, bi_register(0), bi_discard(bi_register(0)), bi_discard(bi_register(4)),
      bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | 0), false), BI_MUX_BIT);
   CASE_ARCH(I, 10, 0x00b8c00300804440ull);
   CASE_ARCH(I, 15, 0x017c000c80008480ull);
}

TEST_F(ValhallPacking, AtestFP16)
{
   bi_instr *I = bi_atest_to(b, bi_register(60), bi_register(60),
                             bi_half(bi_register(1), true),
                             bi_fau(BIR_FAU_ATEST_PARAM, false));
   CASE_ARCH(I, 10, 0x007dbc0208ea013c);
   CASE_ARCH(I, 15, 0x03d43c0108ea013c);
}

TEST_F(ValhallPacking, AtestFP32)
{
   bi_instr *I = bi_atest_to(b, bi_register(60), bi_register(60), one,
                             bi_fau(BIR_FAU_ATEST_PARAM, false));
   CASE_ARCH(I, 10, 0x007dbc0200ead03c);
   CASE_ARCH(I, 15, 0x03d63c0100ead03c);
}

TEST_F(ValhallPacking, Transcendentals)
{
   bi_instr *I =
      bi_frexpm_f32_to(b, bi_register(1), bi_register(0), false, true);
   CASE_ARCH(I, 10, 0x0099c10001000000);
   CASE_ARCH(I, 15, 0x0060010041200000);

   I = bi_frexpe_f32_to(b, bi_register(0), bi_discard(bi_register(0)), false,
                        true);
   CASE_ARCH(I, 10, 0x0099c00001020040);
   CASE_ARCH(I, 15, 0x0060000041220080);

   I = bi_frsq_f32_to(b, bi_register(2), bi_register(1));
   CASE_ARCH(I, 10, 0x009cc20000020001);
   CASE_ARCH(I, 15, 0x0060020001820001);

   I = bi_fma_rscale_f32_to(b, bi_register(0), bi_discard(bi_register(1)),
                            bi_discard(bi_register(2)), bi_neg(zero),
                            bi_discard(bi_register(0)), BI_SPECIAL_LEFT);
   CASE_ARCH(I, 10, 0x0162c00440c04241);
   CASE_ARCH(I, 15, 0x0264000e80c08281);

   I = bi_fma_rscale_f32_to(b, bi_register(0), bi_register(1), bi_register(2),
                            bi_neg(zero), bi_discard(bi_register(0)),
                            BI_SPECIAL_N);
   CASE_ARCH(I, 10, 0x0161c00440c00201);
   CASE_ARCH(I, 15, 0x0264000d80c00201);
}

TEST_F(ValhallPacking, Csel)
{
   bi_instr *I = bi_csel_u32_to(
      b, bi_register(1), bi_discard(bi_register(2)), bi_discard(bi_register(3)),
      bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | 2), false),
      bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | 2), true), BI_CMPF_EQ);
   CASE_ARCH(I, 10, 0x0150c10085844342);
   CASE_ARCH(I, 15, 0x027c010005048382);

   I = bi_csel_u32_to(
      b, bi_register(1), bi_discard(bi_register(2)), bi_discard(bi_register(3)),
      bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | 2), false),
      bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | 2), true), BI_CMPF_LT);
   CASE_ARCH(I, 10, 0x0150c10485844342);
   CASE_ARCH(I, 15, 0x027c010805048382);

   I = bi_csel_s32_to(
      b, bi_register(1), bi_discard(bi_register(2)), bi_discard(bi_register(3)),
      bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | 2), false),
      bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | 2), true), BI_CMPF_LT);
   CASE_ARCH(I, 10, 0x0158c10485844342);
   CASE_ARCH(I, 15, 0x027c014805048382);
}

TEST_F(ValhallPacking, LdAttrImm)
{
   bi_instr *I = bi_ld_attr_imm_to(
      b, bi_register(0), bi_discard(bi_register(60)),
      bi_discard(bi_register(61)), BI_REGISTER_FORMAT_F16, BI_VECSIZE_V4, 1);
   I->table = 1;

   CASE_ARCH(I, 10, 0x0066800433117d7c);
   CASE_ARCH(I, 15, 0x038400023311bdbc);
}

TEST_F(ValhallPacking, LdVarBufImmF16)
{
   bi_instr *I = bi_ld_var_buf_imm_f16_to(
      b, bi_register(2), bi_register(61), BI_REGISTER_FORMAT_F16,
      BI_SAMPLE_CENTER, BI_SOURCE_FORMAT_F16, BI_UPDATE_RETRIEVE, BI_VECSIZE_V4,
      0);
   CASE_ARCH(I, 10, 0x005d82143300003d);
   CASE_ARCH(I, 15, 0x0310020a3f00003d);

   I = bi_ld_var_buf_imm_f16_to(b, bi_register(0), bi_register(61),
                                BI_REGISTER_FORMAT_F16, BI_SAMPLE_SAMPLE,
                                BI_SOURCE_FORMAT_F16, BI_UPDATE_STORE,
                                BI_VECSIZE_V4, 0);
   CASE_ARCH(I, 10, 0x005d80843300003d);
   CASE_ARCH(I, 15, 0x031000423f00003d);

   I = bi_ld_var_buf_imm_f16_to(b, bi_register(0), bi_register(61),
                                BI_REGISTER_FORMAT_F16, BI_SAMPLE_CENTROID,
                                BI_SOURCE_FORMAT_F16, BI_UPDATE_STORE,
                                BI_VECSIZE_V4, 8);
   CASE_ARCH(I, 10, 0x005d80443308003d);
   CASE_ARCH(I, 11, 0x005d80443300083d);
   CASE_ARCH(I, 15, 0x031000223f00083d);
}

TEST_F(ValhallPacking, LdVarBufFlatImmFormat)
{
   bi_instr *I = bi_ld_var_buf_flat_imm_to(
      b, bi_register(0), BI_REGISTER_FORMAT_F32, BI_VECSIZE_V4, 0x12);
   CASE_ARCH(I, 14, 0x0040800832001200);
   CASE_ARCH(I, 15, 0x033900043a0012c0);

   I = bi_ld_var_buf_flat_imm_to(b, bi_register(0), BI_REGISTER_FORMAT_F16,
                                 BI_VECSIZE_V4, 0x12);
   CASE_ARCH(I, 14, 0x0040800433001200);
   CASE_ARCH(I, 15, 0x033900023b0012c0);
}

TEST_F(ValhallPacking, LdVarBufFlat)
{
   bi_instr *I = bi_ld_var_buf_flat_to(b, bi_register(0), bi_register(61),
                                       BI_REGISTER_FORMAT_F32, BI_VECSIZE_V4);
   CASE_ARCH(I, 14, 0x005f80083200003d);
   CASE_ARCH(I, 15, 0x031400043a00003d);

   I = bi_ld_var_buf_flat_to(b, bi_register(0), bi_register(61),
                             BI_REGISTER_FORMAT_F16, BI_VECSIZE_V4);
   CASE_ARCH(I, 14, 0x005f80043300003d);
   CASE_ARCH(I, 15, 0x031400023b00003d);
}

TEST_F(ValhallPacking, LeaBufImm)
{
   bi_instr *I =
      bi_lea_buf_imm_to(b, bi_register(4), bi_discard(bi_register(59)));
   CASE_ARCH(I, 10, 0x005e84040000007b);
   CASE_ARCH(I, 15, 0x03080402000000bb);
}

TEST_F(ValhallPacking, StoreMemoryAccess)
{
   bi_instr *I = bi_store_i96(b, bi_register(0), bi_discard(bi_register(4)),
                              bi_discard(bi_register(5)), BI_SEG_NONE, 0);
   I->mem_access = VA_MEMORY_ACCESS_ESTREAM;
   CASE_ARCH(I, 10, 0x0061400632000044);
   CASE_ARCH(I, 15, 0x0320009302000084);
}

TEST_F(ValhallPacking, Convert16To32)
{
   bi_instr *I = bi_u16_to_u32_to(b, bi_register(2),
                                  bi_discard(bi_half(bi_register(55), false)));
   CASE_ARCH(I, 10, 0x0090c20000140077);
   CASE_ARCH(I, 15, 0x00600200005400b7);

   I = bi_u16_to_u32_to(b, bi_register(2),
                        bi_discard(bi_half(bi_register(55), true)));
   CASE_ARCH(I, 10, 0x0090c20010140077);
   CASE_ARCH(I, 15, 0x00600200105400b7);

   I = bi_u16_to_f32_to(b, bi_register(2),
                        bi_discard(bi_half(bi_register(55), false)));
   CASE_ARCH(I, 10, 0x0090c20000150077);
   /* Removed on v11 */

   I = bi_u16_to_f32_to(b, bi_register(2),
                        bi_discard(bi_half(bi_register(55), true)));
   CASE_ARCH(I, 10, 0x0090c20010150077);
   /* Removed on v11 */

   I = bi_s16_to_s32_to(b, bi_register(2),
                        bi_discard(bi_half(bi_register(55), false)));
   CASE_ARCH(I, 10, 0x0090c20000040077);
   CASE_ARCH(I, 15, 0x00600200004400b7);

   I = bi_s16_to_s32_to(b, bi_register(2),
                        bi_discard(bi_half(bi_register(55), true)));
   CASE_ARCH(I, 10, 0x0090c20010040077);
   CASE_ARCH(I, 15, 0x00600200104400b7);
}

TEST_F(ValhallPacking, Swizzle8)
{
   bi_instr *I =
      bi_icmp_or_v4u8_to(b, bi_register(1), bi_byte(bi_register(0), 0), zero,
                         zero, BI_CMPF_NE, BI_RESULT_TYPE_I1);
   CASE_ARCH(I, 10, 0x00f2c14300c0c000);
   /* Removed on v11 */
}

TEST_F(ValhallPacking, FauPage1)
{
   bi_instr *I = bi_mov_i32_to(
      b, bi_register(1), bi_fau((enum bir_fau)(BIR_FAU_UNIFORM | 32), false));
   CASE_ARCH(I, 10, 0x0291c10000000080ULL);
   CASE_ARCH(I, 15, 0x0061010000200040ULL);
}

TEST_F(ValhallPacking, LdTileV3F16)
{
   bi_instr *I = bi_ld_tile_to(b, bi_register(4), bi_discard(bi_register(0)),
                               bi_register(60), bi_register(3),
                               BI_REGISTER_FORMAT_F16, BI_VECSIZE_V3);
   CASE_ARCH(I, 10, 0x0078840423033c40);
   CASE_ARCH(I, 15, 0x03c0040223033c80);
}

TEST_F(ValhallPacking, Rhadd8)
{
   bi_instr *I = bi_hadd_v4s8_to(b, bi_register(0), bi_discard(bi_register(1)),
                                 bi_discard(bi_register(0)), BI_ROUND_RTP);
   CASE_ARCH(I, 10, 0x00aac000400b4041);
   /* Removed on v11 */
}

TEST_F(ValhallPacking, Atomics)
{

   bi_instr *I =
      bi_atom1_return_i64_to(b, bi_register(0), bi_discard(bi_register(2)),
                             bi_register(3), BI_ATOM_OPC_AINC, 2);
   CASE_ARCH(I, 10, 0x0069800428000042);
   CASE_ARCH(I, 15, 0x0328000220000082);

   I = bi_atom_return_i32_to(b, bi_register(0), bi_discard(bi_register(1)),
                             bi_register(2), bi_register(3), BI_ATOM_OPC_AXCHG,
                             1);
   CASE_ARCH(I, 10, 0x0120c1021bc00002);
   CASE_ARCH(I, 15, 0x032401c10f000002);

   I = bi_atom_return_i64_to(b, bi_register(0), bi_register(2), bi_register(6),
                             bi_register(7), BI_ATOM_OPC_ACMPXCHG, 2);
   CASE_ARCH(I, 10, 0x0120c2182fc00006);
   CASE_ARCH(I, 15, 0x032802cc2f000006);
}
