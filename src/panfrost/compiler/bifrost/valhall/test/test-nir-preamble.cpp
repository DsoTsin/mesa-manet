#include "compiler/nir/tests/nir_test.h"
extern "C" {
#include "panfrost/compiler/pan_nir.h"
}

class preamble_test : public nir_test {
 protected:
   preamble_test(): nir_test("pan preamble")
   {
      inputs.fau.reserved = 8;
   }

   ~preamble_test()
   {
      ralloc_free(main);
      ralloc_free(pilot);
   }

   pan_compile_inputs inputs = {};
   nir_shader *main = nullptr;
   nir_shader *pilot = nullptr;
   unsigned words = 0;

   nir_def *uniform()
   {
      return nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0), .range = 4,
                                    .align_mul = 4);
   }

   nir_def *chain(nir_def *x)
   {
      for (unsigned i = 0; i < 10; i++)
         x = nir_fsin(b, nir_fadd_imm(b, x, 0.1f));
      return x;
   }

   void sink(nir_def *x)
   {
      nir_store_global(b, x, nir_imm_int64(b, 4096), .align_mul = 4);
   }

   bool run()
   {
      nir_validate_shader(b->shader, "test input");
      main = pan_nir_opt_preamble(b->shader, &inputs, &pilot, &words);
      return main != nullptr;
   }

   unsigned count(nir_shader *shader, nir_intrinsic_op op)
   {
      unsigned n = 0;
      nir_foreach_function_impl(impl, shader)
         nir_foreach_block(block, impl) nir_foreach_instr(instr, block)
         {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               n++;
         }
      return n;
   }
};

TEST_F(preamble_test, uniform_chain_uses_reserved_result_space)
{
   sink(chain(uniform()));
   ASSERT_TRUE(run());
   EXPECT_EQ(words, 10u);
   EXPECT_EQ(pilot->info.stage, MESA_SHADER_COMPUTE);
   EXPECT_EQ(count(pilot, nir_intrinsic_load_global_constant), 1u);
   EXPECT_EQ(count(pilot, nir_intrinsic_store_global), 1u);
   EXPECT_EQ(count(main, nir_intrinsic_load_push_constant), 1u);
   EXPECT_EQ(count(main, nir_intrinsic_load_preamble), 0u);
   EXPECT_EQ(count(pilot, nir_intrinsic_store_preamble), 0u);
   EXPECT_EQ(count(b->shader, nir_intrinsic_load_push_constant), 1u);
   nir_foreach_block(block, nir_shader_get_entrypoint(main))
      nir_foreach_instr(instr, block)
   {
      if (instr->type == nir_instr_type_intrinsic) {
         auto intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_push_constant) {
            EXPECT_EQ(nir_src_as_uint(intr->src[0]), 32u);
         }
      }
   }
}

TEST_F(preamble_test, invocation_values_stay_in_main)
{
   auto x = nir_u2f32(b, nir_channel(b, nir_load_local_invocation_id(b), 0));
   sink(chain(x));
   EXPECT_FALSE(run());
}

TEST_F(preamble_test, workgroup_values_stay_in_main)
{
   auto x = nir_u2f32(b, nir_channel(b, nir_load_workgroup_id(b), 0));
   sink(chain(x));
   EXPECT_FALSE(run());
}

TEST_F(preamble_test, writable_memory_stays_in_main)
{
   auto x = nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, 0),
                          .align_mul = 4);
   sink(chain(x));
   EXPECT_FALSE(run());
}

TEST_F(preamble_test, no_fau_space_leaves_input_untouched)
{
   inputs.fau.reserved = PAN_MAX_PUSH;
   sink(chain(uniform()));
   EXPECT_FALSE(run());
   EXPECT_EQ(words, PAN_MAX_PUSH);
   EXPECT_EQ(count(b->shader, nir_intrinsic_load_push_constant), 1u);
}

TEST_F(preamble_test, mixed_uniform_and_varying)
{
   auto x = chain(uniform());
   auto id = nir_u2f32(b, nir_channel(b, nir_load_local_invocation_id(b), 0));
   sink(nir_fadd(b, x, id));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(main, nir_intrinsic_load_local_invocation_id), 1u);
   EXPECT_EQ(count(pilot, nir_intrinsic_load_local_invocation_id), 0u);
}

TEST_F(preamble_test, cheap_push_load_is_not_extracted)
{
   sink(uniform());
   EXPECT_FALSE(run());
}

TEST_F(preamble_test, aligned_64bit_results)
{
   inputs.fau.reserved = 3;
   auto x = nir_load_push_constant(b, 1, 64, nir_imm_int(b, 0), .range = 8,
                                   .align_mul = 8);
   for (unsigned i = 0; i < 12; i++)
      x = nir_ixor(b, nir_imul_imm(b, x, 17), nir_ushr_imm(b, x, 9));
   sink(x);
   ASSERT_TRUE(run());
   EXPECT_EQ(words, 6u);
}

TEST_F(preamble_test, half_vector_results)
{
   auto x = nir_load_push_constant(b, 4, 16, nir_imm_int(b, 0), .range = 8,
                                   .align_mul = 8);
   for (unsigned i = 0; i < 10; i++)
      x = nir_fadd(b, nir_fmul_imm(b, x, 1.25), nir_frcp(b, x));
   sink(x);
   ASSERT_TRUE(run());
   EXPECT_EQ(words, 10u);
}

TEST_F(preamble_test, unsupported_control_dependency)
{
   auto condition =
      nir_ine_imm(b, nir_channel(b, nir_load_num_workgroups(b), 0), 0);
   nir_push_if(b, condition);
   auto then_value = chain(uniform());
   nir_push_else(b, NULL);
   auto else_value = chain(nir_fadd_imm(b, uniform(), 1.0f));
   nir_pop_if(b, NULL);
   sink(nir_if_phi(b, then_value, else_value));
   if (run()) {
      EXPECT_EQ(count(pilot, nir_intrinsic_load_num_workgroups), 0u);
   }
}

TEST_F(preamble_test, compile_main_with_control_flow)
{
   if (!pan_use_kraid(10, MESA_SHADER_COMPUTE, false)) {
      GTEST_SKIP();
   }
   inputs.gpu_id = 0xa8670000;
   b->shader->options =
      pan_get_nir_shader_compiler_options(10, MESA_SHADER_COMPUTE, false);
   b->shader->info.workgroup_size[0] = 32;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;
   auto value = chain(uniform());
   auto id = nir_channel(b, nir_load_local_invocation_id(b), 0);
   nir_push_if(b, nir_ine_imm(b, id, 0));
   sink(value);
   nir_push_else(b, nullptr);
   sink(nir_fadd_imm(b, value, 1.0f));
   nir_pop_if(b, nullptr);
   pan_compile_preamble compiled = {};
   util_dynarray binary = {};
   pan_shader_info info = {};
   inputs.preamble = &compiled;
   pan_postprocess_nir(b->shader, &inputs, &info);
   pan_shader_compile(b->shader, &inputs, &binary, &info);
   EXPECT_GT(binary.size, 0u);
   EXPECT_GT(compiled.binary.size, 0u);
   util_dynarray_fini(&binary);
   util_dynarray_fini(&compiled.binary);
}

TEST_F(preamble_test, single_uniform_alu_is_extracted)
{
   sink(nir_fadd_imm(b, uniform(), 1.0f));
   ASSERT_TRUE(run());
   EXPECT_EQ(words, 10u);
   EXPECT_EQ(count(main, nir_intrinsic_load_push_constant), 1u);
}

TEST_F(preamble_test, source_modifier_is_not_extracted)
{
   sink(nir_fneg(b, uniform()));
   EXPECT_FALSE(run());
}

TEST_F(preamble_test, uniform_load_in_divergent_branch_is_extracted)
{
   auto id = nir_channel(b, nir_load_local_invocation_id(b), 0);
   nir_push_if(b, nir_ine_imm(b, id, 0));
   sink(nir_fadd_imm(b, uniform(), 1.0f));
   nir_pop_if(b, nullptr);
   ASSERT_TRUE(run());
   EXPECT_EQ(words, 10u);
   EXPECT_EQ(count(main, nir_intrinsic_load_local_invocation_id), 1u);
}

TEST_F(preamble_test, fau_pressure_keeps_cheap_uniform_alu)
{
   inputs.fau.pushable_ubos = 1;
   auto acc = nir_u2f32(b, nir_channel(b, nir_load_local_invocation_id(b), 0));
   for (unsigned i = 0; i < 100; i++) {
      auto x = nir_load_ubo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, i * 4),
                            .align_mul = 4, .range = 4096);
      acc = nir_fadd(b, acc, x);
   }
   sink(acc);
   sink(nir_fadd_imm(b, uniform(), 1.0f));
   EXPECT_FALSE(run());
   EXPECT_EQ(words, 8u);
}

TEST_F(preamble_test, fau_pressure_extracts_expensive_uniform_alu)
{
   inputs.fau.pushable_ubos = 1;
   auto acc = nir_u2f32(b, nir_channel(b, nir_load_local_invocation_id(b), 0));
   for (unsigned i = 0; i < 100; i++) {
      auto x = nir_load_ubo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, i * 4),
                            .align_mul = 4, .range = 4096);
      acc = nir_fadd(b, acc, x);
   }
   sink(acc);
   sink(nir_fsin(b, uniform()));
   ASSERT_TRUE(run());
   EXPECT_EQ(words, 10u);
   EXPECT_EQ(count(main, nir_intrinsic_load_ubo), 100u);
}

class preamble_arch_test
   : public preamble_test,
     public ::testing::WithParamInterface<std::pair<unsigned, uint64_t>> {};

TEST_P(preamble_arch_test, compile_main_and_pilot)
{
   const unsigned arch = GetParam().first;
   if (!pan_use_kraid(arch, MESA_SHADER_COMPUTE, false)) {
      GTEST_SKIP();
   }
   inputs.gpu_id = GetParam().second;
   b->shader->options =
      pan_get_nir_shader_compiler_options(arch, MESA_SHADER_COMPUTE, false);
   b->shader->info.workgroup_size[0] = 32;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;
   auto value = chain(uniform());
   auto id = nir_channel(b, nir_load_local_invocation_id(b), 0);
   nir_push_if(b, nir_ine_imm(b, id, 0));
   sink(value);
   nir_push_else(b, nullptr);
   sink(nir_fadd_imm(b, value, 1.0f));
   nir_pop_if(b, nullptr);
   pan_compile_preamble compiled = {};
   util_dynarray binary = {};
   pan_shader_info info = {};
   inputs.preamble = &compiled;
   pan_postprocess_nir(b->shader, &inputs, &info);
   pan_shader_compile(b->shader, &inputs, &binary, &info);
   EXPECT_GT(binary.size, 0u);
   EXPECT_GT(compiled.binary.size, 0u);
   util_dynarray_fini(&binary);
   util_dynarray_fini(&compiled.binary);
}

INSTANTIATE_TEST_SUITE_P(
   csf, preamble_arch_test,
   ::testing::Values(std::make_pair(10u, 0xa8670000ull),
                     std::make_pair(11u, 0xb8020000ull),
                     std::make_pair(12u, 0xc8000000ull),
                     std::make_pair(13u, 0xd8000000ull),
                     std::make_pair(14u, 0xe8000000ull),
                     std::make_pair(15u, 0x0f080000f0000000ull)));
