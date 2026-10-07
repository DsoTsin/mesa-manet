#include "compiler/nir/tests/nir_test.h"
extern "C" {
#include "panfrost/compiler/pan_nir.h"
}

class shadd_nir_test : public nir_test {
 protected:
   shadd_nir_test(): nir_test("pan shadd")
   {
      b->shader->options =
         pan_get_nir_shader_compiler_options(15, MESA_SHADER_COMPUTE, false);
   }

   nir_def *input(unsigned bits, unsigned offset = 0)
   {
      return nir_load_push_constant(b, 1, bits, nir_imm_int(b, offset),
                                    .align_mul = bits / 8);
   }

   nir_intrinsic_instr *sink(nir_def *value)
   {
      return nir_store_global(b, value, nir_imm_int64(b, 4096), .align_mul = 8);
   }

   unsigned count(nir_op op)
   {
      unsigned result = 0;
      nir_foreach_block(block, b->impl) nir_foreach_instr(instr, block)
         result += instr->type == nir_instr_type_alu &&
                   nir_instr_as_alu(instr)->op == op;
      return result;
   }

   bool run(unsigned arch = 15)
   {
      nir_validate_shader(b->shader, "shadd input");
      bool progress = pan_nir_opt_shadd(b->shader, arch);
      nir_validate_shader(b->shader, "shadd output");
      return progress;
   }
};

TEST_F(shadd_nir_test, unsigned_and_commuted)
{
   auto shifted = nir_ishl_imm(b, nir_u2u64(b, input(32)), 2);
   sink(nir_iadd(b, input(64, 8), shifted));
   sink(nir_iadd(b, shifted, input(64, 16)));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(nir_op_shadd_u32_pan), 2u);
   EXPECT_FALSE(run());
}

TEST_F(shadd_nir_test, signed_extension)
{
   sink(nir_iadd(b, input(64, 8),
                nir_ishl_imm(b, nir_i2i64(b, input(32)), 7)));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(nir_op_shadd_s32_pan), 1u);
}

TEST_F(shadd_nir_test, narrow_extensions)
{
   sink(nir_iadd(b, input(64, 8),
                nir_ishl_imm(b, nir_i2i64(b, input(16)), 1)));
   sink(nir_iadd(b, input(64, 8),
                nir_ishl_imm(b, nir_u2u64(b, input(16)), 1)));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(nir_op_shadd_s32_pan), 1u);
   EXPECT_EQ(count(nir_op_shadd_u32_pan), 1u);
}

TEST_F(shadd_nir_test, full_width_source)
{
   sink(nir_iadd(b, input(64), nir_ishl_imm(b, input(64, 8), 3)));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(nir_op_shadd_pan), 1u);
}

TEST_F(shadd_nir_test, out_of_range_and_dynamic_shift)
{
   for (unsigned shift : {0u, 8u, 31u, 63u})
      sink(nir_iadd(b, input(64), nir_ishl_imm(b, input(64, 8), shift)));
   sink(nir_iadd(b, input(64), nir_ishl(b, input(64, 8), input(32, 16))));
   EXPECT_FALSE(run());
}

TEST_F(shadd_nir_test, preserve_32bit_wrap)
{
   auto shifted = nir_ishl_imm(b, input(32), 2);
   sink(nir_iadd(b, input(64, 8), nir_u2u64(b, shifted)));
   sink(nir_iadd(b, input(64, 8), nir_i2i64(b, shifted)));
   EXPECT_FALSE(run());
}

TEST_F(shadd_nir_test, preserve_other_architectures)
{
   sink(nir_iadd(b, input(64), nir_ishl_imm(b, input(64, 8), 2)));
   EXPECT_FALSE(run(10));
   EXPECT_FALSE(run(12));
   EXPECT_FALSE(run(14));
}

TEST_F(shadd_nir_test, proven_no_unsigned_wrap)
{
   auto bounded = nir_iand_imm(b, input(32), 0x3fffffff);
   sink(nir_iadd(b, input(64, 8), nir_u2u64(b, nir_ishl_imm(b, bounded, 2))));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(nir_op_shadd_u32_pan), 1u);
}

TEST_F(shadd_nir_test, unsigned_wrap_boundary_is_rejected)
{
   auto bounded = nir_iand_imm(b, input(32), 0x40000000);
   sink(nir_iadd(b, input(64, 8), nir_u2u64(b, nir_ishl_imm(b, bounded, 2))));
   EXPECT_FALSE(run());
}

TEST_F(shadd_nir_test, explicit_no_unsigned_wrap)
{
   auto shifted = nir_ishl_imm(b, input(32), 2);
   nir_def_as_alu(shifted)->no_unsigned_wrap = true;
   sink(nir_iadd(b, input(64, 8), nir_u2u64(b, shifted)));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(nir_op_shadd_u32_pan), 1u);
}

TEST_F(shadd_nir_test, shared_address_after_int64_lowering)
{
   b->shader->info.workgroup_size[0] = 256;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;
   auto id = nir_channel(b, nir_load_local_invocation_id(b), 0);
   auto offset = nir_lshift_or_pan(b, id, nir_imm_intN_t(b, 2, 8), nir_imm_int(b, 0));
   sink(nir_iadd(b, nir_load_shared_base_ptr(b, 1, 64),
                nir_pack_64_2x32_split(b, offset, nir_imm_int(b, 0))));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(nir_op_shadd_u32_pan), 1u);
}

TEST_F(shadd_nir_test, shift_or_nonzero_is_rejected)
{
   auto bounded = nir_iand_imm(b, input(32), 255);
   auto offset = nir_lshift_or_pan(b, bounded, nir_imm_intN_t(b, 2, 8), nir_imm_int(b, 1));
   sink(nir_iadd(b, input(64, 8), nir_u2u64(b, offset)));
   EXPECT_FALSE(run());
}

TEST_F(shadd_nir_test, packed_shift_wrap_is_rejected)
{
   auto offset = nir_lshift_or_pan(b, input(32), nir_imm_intN_t(b, 2, 8), nir_imm_int(b, 0));
   sink(nir_iadd(b, input(64, 8),
                nir_pack_64_2x32_split(b, offset, nir_imm_int(b, 0))));
   EXPECT_FALSE(run());
}

TEST_F(shadd_nir_test, survives_int64_lowering)
{
   sink(nir_iadd(b, input(64), nir_ishl_imm(b, nir_i2i64(b, input(32, 8)), 2)));
   ASSERT_TRUE(run());
   nir_lower_int64(b->shader);
   nir_opt_dce(b->shader);
   nir_validate_shader(b->shader, "shadd after int64 lowering");
   EXPECT_EQ(count(nir_op_shadd_s32_pan), 1u);
   EXPECT_EQ(count(nir_op_ishl), 0u);
}

TEST_F(shadd_nir_test, differential_constants)
{
   uint64_t state = 0x9e3779b97f4a7c15ull;
   const uint64_t edges[] = {0, 1, 0x7fff, 0x8000, 0xffff, 0x7fffffff,
                             0x80000000, 0xffffffff, 0x100000000,
                             0x8000000000000000ull, UINT64_MAX};
   for (unsigned n = 0; n < 128; n++) {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      uint64_t x = n < ARRAY_SIZE(edges) ? edges[n] : state;
      uint64_t base = state ^ 0xa5a5a5a5a5a5a5a5ull;
      for (unsigned bits : {16u, 32u, 64u}) {
         for (bool sign : {false, true}) {
            for (unsigned shift = 1; shift <= 7; shift++) {
               uint64_t extended = bits == 64 ? x :
                  sign ? (bits == 16 ? uint64_t(int64_t(int16_t(x))) :
                                      uint64_t(int64_t(int32_t(x)))) :
                         (bits == 16 ? uint64_t(uint16_t(x)) : uint64_t(uint32_t(x)));
               auto value = nir_imm_intN_t(b, x, bits);
               value = sign ? nir_i2i64(b, value) : nir_u2u64(b, value);
               auto store = sink(nir_iadd(b, nir_imm_int64(b, base),
                                         nir_ishl_imm(b, value, shift)));
               ASSERT_TRUE(run());
               nir_opt_constant_folding(b->shader);
               ASSERT_TRUE(nir_src_is_const(store->src[0]));
               EXPECT_EQ(nir_src_as_uint(store->src[0]), base + (extended << shift));
               nir_instr_remove(&store->instr);
               nir_opt_dce(b->shader);
               b->cursor = nir_after_block(nir_start_block(b->impl));
            }
         }
      }
   }
}

TEST_F(shadd_nir_test, differential_narrow_shift_boundaries)
{
   for (unsigned shift = 1; shift <= 7; shift++) {
      uint32_t limit = UINT32_MAX >> shift;
      for (uint32_t x : {0u, 1u, limit - 1, limit, limit + 1, UINT32_MAX}) {
         for (bool packed : {false, true}) {
            auto value = nir_imm_int(b, x);
            auto offset = packed ? nir_lshift_or_pan(b, value,
               nir_imm_intN_t(b, shift, 8), nir_imm_int(b, 0)) :
               nir_ishl_imm(b, value, shift);
            auto extended = packed ? nir_pack_64_2x32_split(b, offset,
               nir_imm_int(b, 0)) : nir_u2u64(b, offset);
            uint64_t base = UINT64_MAX - 16;
            auto store = sink(nir_iadd(b, nir_imm_int64(b, base), extended));
            EXPECT_EQ(run(), x <= limit);
            nir_opt_constant_folding(b->shader);
            ASSERT_TRUE(nir_src_is_const(store->src[0]));
            EXPECT_EQ(nir_src_as_uint(store->src[0]), base + uint32_t(x << shift));
            nir_instr_remove(&store->instr);
            nir_opt_dce(b->shader);
            b->cursor = nir_after_block(nir_start_block(b->impl));
         }
      }
   }
}
