#include "compiler/nir/tests/nir_test.h"
extern "C" {
#include "panfrost/compiler/bifrost/bifrost_nir.h"
}

class bool_conversion_test : public nir_test,
                             public ::testing::WithParamInterface<unsigned> {
 protected:
   bool_conversion_test() : nir_test("pan boolean conversions") {}

   void check(bool floating)
   {
      for (unsigned src_bits : {8u, 16u, 32u, 64u}) {
         for (unsigned dst_bits : {8u, 16u, 32u, 64u}) {
            if (floating && dst_bits != 16 && dst_bits != 32)
               continue;
            for (bool invert : {false, true}) {
               for (bool value : {false, true}) {
                  SCOPED_TRACE(::testing::Message()
                               << src_bits << "->" << dst_bits
                               << " invert=" << invert << " value=" << value);
                  b->cursor = nir_after_impl(b->impl);
                  auto condition = nir_imm_intN_t(b, value ? -1 : 0, src_bits);
                  if (invert)
                     condition = nir_inot(b, condition);
                  auto op = floating ? nir_op_b2f16 : nir_op_b2i8;
                  if (floating)
                     op = dst_bits == 16 ? nir_op_b2f16 : nir_op_b2f32;
                  else {
                     switch (dst_bits) {
                     case 8: op = nir_op_b2i8; break;
                     case 16: op = nir_op_b2i16; break;
                     case 32: op = nir_op_b2i32; break;
                     case 64: op = nir_op_b2i64; break;
                     }
                  }
                  auto converted = nir_build_alu1(b, op, condition);
                  auto store = nir_store_global(b, converted,
                     nir_imm_int64(b, 4096), .align_mul = dst_bits / 8);
                  ASSERT_TRUE(bifrost_nir_lower_algebraic_late(
                     b->shader, GetParam(), true));
                  nir_validate_shader(b->shader, "lowered boolean conversion");
                  while (nir_opt_constant_folding(b->shader)) {}
                  ASSERT_TRUE(nir_src_is_const(store->src[0]));
                  uint64_t expected = value != invert ?
                     (floating ? (dst_bits == 16 ? 0x3c00u : 0x3f800000u) : 1) : 0;
                  EXPECT_EQ(nir_src_as_uint(store->src[0]), expected);
                  nir_instr_remove(&store->instr);
                  nir_opt_dce(b->shader);
               }
            }
         }
      }
   }
};

TEST_P(bool_conversion_test, float_values)
{
   check(true);
}

TEST_P(bool_conversion_test, integer_values)
{
   check(false);
}

INSTANTIATE_TEST_SUITE_P(Architectures, bool_conversion_test,
                        ::testing::Values(10u, 15u));
