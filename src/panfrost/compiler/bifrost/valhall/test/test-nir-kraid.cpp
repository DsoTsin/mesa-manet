#include "compiler/nir/tests/nir_test.h"
extern "C" {
#include "panfrost/compiler/pan_nir.h"
}

class kraid_nir_test : public nir_test {
 protected:
   kraid_nir_test(): nir_test("kraid NIR")
   {
      inputs.gpu_id = 0xa8670000;
      inputs.disable_preamble = true;
      b->shader->options =
         pan_get_nir_shader_compiler_options(10, MESA_SHADER_COMPUTE, false);
      b->shader->info.workgroup_size[0] = 32;
      b->shader->info.workgroup_size[1] = 1;
      b->shader->info.workgroup_size[2] = 1;
   }

   pan_compile_inputs inputs = {};
   pan_shader_info info = {};

   nir_def *input(unsigned offset)
   {
      return nir_load_push_constant(b, 1, 32, nir_imm_int(b, offset),
                                    .align_mul = 4);
   }

   void sink(nir_def *value)
   {
      nir_store_global(b, value, nir_imm_int64(b, 4096), .align_mul = 4);
   }

   void postprocess()
   {
      pan_postprocess_nir(b->shader, &inputs, &info);
      nir_validate_shader(b->shader, "kraid test");
      util_dynarray binary = {};
      pan_shader_compile(b->shader, &inputs, &binary, &info);
      EXPECT_GT(binary.size, 0u);
      util_dynarray_fini(&binary);
   }

   unsigned count(nir_op op)
   {
      unsigned count = 0;
      nir_foreach_block(block, b->impl) nir_foreach_instr(instr, block)
         count += instr->type == nir_instr_type_alu &&
                  nir_instr_as_alu(instr)->op == op;
      return count;
   }

   unsigned count(nir_intrinsic_op op)
   {
      unsigned count = 0;
      nir_foreach_block(block, b->impl) nir_foreach_instr(instr, block)
         count += instr->type == nir_instr_type_intrinsic &&
                  nir_instr_as_intrinsic(instr)->intrinsic == op;
      return count;
   }

   void add_tree()
   {
      auto a = nir_fmul(b, input(0), input(4));
      auto c = nir_fmul(b, input(8), input(12));
      auto e = nir_fmul(b, input(16), input(20));
      sink(nir_fadd(b, nir_fadd(b, nir_fadd(b, a, c), e), input(24)));
   }
};

TEST_F(kraid_nir_test, reassociate_add_tree_for_fma)
{
   if (!pan_use_kraid(10, MESA_SHADER_COMPUTE, false))
      GTEST_SKIP();
   add_tree();
   postprocess();
   EXPECT_EQ(count(nir_op_ffma), 3u);
   EXPECT_EQ(count(nir_op_fmul), 0u);
}

TEST_F(kraid_nir_test, preserve_no_contraction)
{
   b->fp_math_ctrl = nir_fp_exact;
   add_tree();
   postprocess();
   EXPECT_EQ(count(nir_op_ffma), 0u);
   EXPECT_EQ(count(nir_op_fmul), 3u);
   EXPECT_EQ(count(nir_op_fadd), 3u);
}

TEST_F(kraid_nir_test, scalar_lane_operations_remain_native)
{
   if (!pan_use_kraid(10, MESA_SHADER_COMPUTE, false))
      GTEST_SKIP();
   auto lane = nir_load_subgroup_invocation(b);
   auto value = nir_iadd(b, input(0), lane);
   auto shuffled = nir_shuffle(b, value, nir_iand_imm(b, input(4), 15));
   auto up = nir_shuffle_up(b, shuffled, nir_imm_int(b, 2));
   sink(nir_shuffle_down(b, up, nir_imm_int(b, 3)));
   postprocess();
   EXPECT_EQ(count(nir_intrinsic_shuffle), 1u);
   EXPECT_EQ(count(nir_intrinsic_shuffle_up), 1u);
   EXPECT_EQ(count(nir_intrinsic_shuffle_down), 1u);
   EXPECT_EQ(count(nir_intrinsic_ballot), 0u);
}

TEST_F(kraid_nir_test, divergent_lane_index_uses_uniform_reads)
{
   if (!pan_use_kraid(10, MESA_SHADER_COMPUTE, false))
      GTEST_SKIP();
   auto lane = nir_load_subgroup_invocation(b);
   auto value = nir_iadd(b, input(0), lane);
   auto index = nir_ixor(b, lane, nir_imm_int(b, 3));
   sink(nir_shuffle(b, value, index));
   postprocess();
   EXPECT_EQ(count(nir_intrinsic_shuffle), 0u);
   EXPECT_GT(count(nir_intrinsic_ballot), 0u);
   EXPECT_GT(count(nir_intrinsic_read_invocation), 0u);
}
