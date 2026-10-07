#include "compiler/nir/tests/nir_test.h"
#include <vector>
extern "C" {
#include "panfrost/compiler/pan_nir.h"
}

class preamble_test : public nir_test {
 protected:
   preamble_test(): nir_test("pan preamble")
   {
      inputs.fau.reserved = 8;
      memset(virt.map, 0xff, sizeof(virt.map));
   }

   ~preamble_test()
   {
      ralloc_free(main);
      ralloc_free(pilot);
   }

   pan_compile_inputs inputs = {};
   nir_shader *main = nullptr;
   nir_shader *pilot = nullptr;
   pan_fau_virtual virt = {};

   unsigned words()
   {
      return pan_fau_virtual_words(&virt);
   }

   void place()
   {
      unsigned next = ALIGN_POT(inputs.fau.reserved, 2);
      for (unsigned i = 0; i < virt.value_count; i++) {
         const pan_fau_value *value = &virt.values[i];
         unsigned at;
         if (value->alias >= 0) {
            at = virt.map[virt.values[value->alias].word];
         } else {
            at = ALIGN_POT(next, MAX2(value->align, 1u));
            next = at + value->size;
         }
         for (unsigned w = 0; w < value->size; w++)
            virt.map[value->word + w] = at + w;
      }
      virt.placed = true;
      pan_nir_pilot_place_results(pilot, &virt);
   }

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
      main = pan_nir_opt_preamble(b->shader, &inputs, &pilot, &virt);
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
   EXPECT_EQ(words(), 1u);
   EXPECT_EQ(pilot->info.stage, MESA_SHADER_COMPUTE);
   EXPECT_EQ(count(pilot, nir_intrinsic_load_global_constant), 1u);
   EXPECT_EQ(count(pilot, nir_intrinsic_store_preamble), 1u);
   EXPECT_EQ(count(main, nir_intrinsic_load_push_constant), 0u);
   EXPECT_EQ(count(main, nir_intrinsic_load_preamble), 1u);
   EXPECT_EQ(count(b->shader, nir_intrinsic_load_push_constant), 1u);
   place();
   EXPECT_EQ(virt.map[virt.values[0].word], 8);
   EXPECT_EQ(count(pilot, nir_intrinsic_store_preamble), 0u);
   EXPECT_EQ(count(pilot, nir_intrinsic_store_global), 1u);
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
   EXPECT_EQ(virt.value_count, 0u);
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

TEST_F(preamble_test, volatile_push_word_stays_in_main)
{
   BITSET_SET(inputs.fau.pilot_volatile, 0);
   sink(chain(uniform()));
   EXPECT_FALSE(run());
}

TEST_F(preamble_test, volatile_push_word_only_blocks_its_loads)
{
   BITSET_SET(inputs.fau.pilot_volatile, 1);
   sink(chain(uniform()));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(main, nir_intrinsic_load_push_constant), 0u);
   EXPECT_EQ(count(pilot, nir_intrinsic_load_global_constant), 1u);
}

TEST_F(preamble_test, volatile_push_word_inside_wide_load)
{
   BITSET_SET(inputs.fau.pilot_volatile, 1);
   auto x = nir_load_push_constant(b, 1, 64, nir_imm_int(b, 0), .range = 8,
                                   .align_mul = 8);
   for (unsigned i = 0; i < 12; i++)
      x = nir_ixor(b, nir_imul_imm(b, x, 17), nir_ushr_imm(b, x, 9));
   sink(x);
   EXPECT_FALSE(run());
}

TEST_F(preamble_test, volatile_push_word_keeps_its_use_in_main)
{
   BITSET_SET(inputs.fau.pilot_volatile, 1);
   auto stable = chain(uniform());
   auto patched = nir_load_push_constant(b, 1, 32, nir_imm_int(b, 4),
                                         .range = 4, .align_mul = 4);
   sink(nir_fadd(b, stable, nir_fmul(b, patched, patched)));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(main, nir_intrinsic_load_push_constant), 1u);
   EXPECT_EQ(count(pilot, nir_intrinsic_load_global_constant), 1u);
   EXPECT_EQ(count(main, nir_intrinsic_load_preamble), 1u);
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
   ASSERT_EQ(virt.value_count, 1u);
   EXPECT_EQ(virt.values[0].size, 2u);
   EXPECT_EQ(virt.values[0].align, 2u);
   EXPECT_EQ(virt.values[0].word % 2, 0u);
}

TEST_F(preamble_test, half_vector_results)
{
   auto x = nir_load_push_constant(b, 4, 16, nir_imm_int(b, 0), .range = 8,
                                   .align_mul = 8);
   for (unsigned i = 0; i < 10; i++)
      x = nir_fadd(b, nir_fmul_imm(b, x, 1.25), nir_frcp(b, x));
   sink(x);
   ASSERT_TRUE(run());
   ASSERT_EQ(virt.value_count, 1u);
   EXPECT_EQ(virt.values[0].size, 2u);
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
   EXPECT_EQ(words(), 1u);
   EXPECT_EQ(count(main, nir_intrinsic_load_preamble), 1u);
   EXPECT_EQ(count(main, nir_intrinsic_load_push_constant), 0u);
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
   EXPECT_EQ(words(), 1u);
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
   EXPECT_EQ(virt.value_count, 0u);
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
   EXPECT_EQ(words(), 1u);
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

class preamble_buffer_test : public preamble_test {
 protected:
   preamble_buffer_test()
   {
      inputs.gpu_id = 0x0f080000f0000000ull;
      inputs.gpu_variant = 4;
   }

   nir_def *lane()
   {
      return nir_channel(b, nir_load_local_invocation_id(b), 0);
   }

   nir_def *ssbo_load(unsigned table, unsigned index, nir_def *offset,
                      unsigned align_mul, unsigned align_offset)
   {
      return nir_load_ssbo(b, 4, 32, nir_imm_int(b, pan_res_handle(table, index)),
                           offset, .access = ACCESS_CAN_REORDER,
                           .align_mul = align_mul, .align_offset = align_offset);
   }

   std::vector<nir_intrinsic_instr *> intrinsics(nir_shader *shader,
                                                 nir_intrinsic_op op)
   {
      std::vector<nir_intrinsic_instr *> list;
      nir_foreach_function_impl(impl, shader)
         nir_foreach_block(block, impl) nir_foreach_instr(instr, block)
         {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               list.push_back(nir_instr_as_intrinsic(instr));
         }
      return list;
   }

   unsigned count_alu(nir_shader *shader, nir_op op)
   {
      unsigned n = 0;
      nir_foreach_function_impl(impl, shader)
         nir_foreach_block(block, impl) nir_foreach_instr(instr, block)
         {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               n++;
         }
      return n;
   }

   static nir_def *split_add_imm(nir_def *addr, uint64_t *imm)
   {
      *imm = 0;
      nir_instr *instr = nir_def_instr(addr);
      if (instr->type != nir_instr_type_alu)
         return addr;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op != nir_op_iadd)
         return addr;
      for (unsigned i = 0; i < 2; i++) {
         if (nir_src_is_const(alu->src[i].src)) {
            *imm = nir_src_as_uint(alu->src[i].src);
            return alu->src[1 - i].src.ssa;
         }
      }
      return addr;
   }
};

TEST_F(preamble_buffer_test, ssbo_loads_share_pilot_base_address)
{
   auto offset = nir_imul_imm(b, lane(), 192);
   for (unsigned i = 0; i < 3; i++)
      sink(ssbo_load(2, 1, nir_iadd_imm(b, offset, 16 * i), 64, 16 * i));
   ASSERT_TRUE(run());
   EXPECT_EQ(words(), 2u);
   EXPECT_EQ(count(main, nir_intrinsic_load_ssbo), 0u);
   EXPECT_EQ(count(main, nir_intrinsic_load_ssbo_address), 0u);
   EXPECT_EQ(count(main, nir_intrinsic_load_preamble), 1u);
   EXPECT_EQ(count(pilot, nir_intrinsic_load_ssbo_address), 0u);
   EXPECT_EQ(count(pilot, nir_intrinsic_store_preamble), 1u);

   auto loads = intrinsics(main, nir_intrinsic_load_global);
   ASSERT_EQ(loads.size(), 3u);
   nir_def *shared = nullptr;
   for (unsigned i = 0; i < 3; i++) {
      uint64_t imm;
      nir_def *base = split_add_imm(loads[i]->src[0].ssa, &imm);
      EXPECT_EQ(imm, 16u * i);
      EXPECT_EQ(nir_intrinsic_access(loads[i]), ACCESS_CAN_REORDER);
      EXPECT_EQ(nir_intrinsic_align_mul(loads[i]), 64u);
      EXPECT_EQ(nir_intrinsic_align_offset(loads[i]), 16u * i);
      if (!shared)
         shared = base;
      EXPECT_EQ(base, shared);
   }

   place();
   EXPECT_EQ(count(pilot, nir_intrinsic_store_preamble), 0u);
   EXPECT_EQ(count(pilot, nir_intrinsic_store_global), 1u);
}

TEST_F(preamble_buffer_test, offset_constant_outside_alignment_is_not_folded)
{
   sink(ssbo_load(2, 1, nir_iadd_imm(b, nir_ishl_imm(b, lane(), 4), 16), 16,
                  0));
   ASSERT_TRUE(run());
   auto loads = intrinsics(main, nir_intrinsic_load_global);
   ASSERT_EQ(loads.size(), 1u);
   uint64_t imm;
   split_add_imm(loads[0]->src[0].ssa, &imm);
   EXPECT_EQ(imm, 0u);
}

TEST_F(preamble_buffer_test, misaligned_constant_keeps_aligned_remainder)
{
   sink(ssbo_load(2, 1, nir_iadd_imm(b, nir_imul_imm(b, lane(), 192), 0x50),
                  64, 16));
   ASSERT_TRUE(run());
   auto loads = intrinsics(main, nir_intrinsic_load_global);
   ASSERT_EQ(loads.size(), 1u);
   uint64_t imm;
   nir_def *base = split_add_imm(loads[0]->src[0].ssa, &imm);
   EXPECT_EQ(imm, 16u);
   nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(base));
   ASSERT_EQ(add->op, nir_op_iadd);
   nir_alu_instr *zext = nullptr;
   for (unsigned i = 0; i < 2; i++) {
      nir_instr *src = nir_def_instr(add->src[i].src.ssa);
      if (src->type == nir_instr_type_alu &&
          nir_instr_as_alu(src)->op == nir_op_pack_64_2x32_split)
         zext = nir_instr_as_alu(src);
   }
   ASSERT_NE(zext, nullptr);
   uint64_t rest;
   split_add_imm(zext->src[0].src.ssa, &rest);
   EXPECT_EQ(rest, 0x40u);
}

TEST_F(preamble_buffer_test, uniform_address_is_computed_in_pilot)
{
   sink(nir_load_ssbo(b, 1, 32, nir_imm_int(b, pan_res_handle(1, 0)),
                      nir_iadd_imm(b, nir_ishl_imm(b, uniform(), 2), 0x30),
                      .align_mul = 4));
   ASSERT_TRUE(run());
   EXPECT_EQ(words(), 2u);
   EXPECT_EQ(count(main, nir_intrinsic_load_ssbo), 0u);
   EXPECT_EQ(count(main, nir_intrinsic_load_push_constant), 0u);
   auto loads = intrinsics(main, nir_intrinsic_load_global);
   ASSERT_EQ(loads.size(), 1u);
   nir_instr *instr = nir_def_instr(loads[0]->src[0].ssa);
   ASSERT_EQ(instr->type, nir_instr_type_intrinsic);
   EXPECT_EQ(nir_instr_as_intrinsic(instr)->intrinsic,
             nir_intrinsic_load_preamble);
   EXPECT_EQ(count(pilot, nir_intrinsic_load_ssbo_address), 0u);
   EXPECT_EQ(count(pilot, nir_intrinsic_load_push_constant), 0u);
}

TEST_F(preamble_buffer_test, reorderable_uniform_load_is_pushed_as_data)
{
   sink(chain(nir_load_ssbo(b, 1, 32, nir_imm_int(b, pan_res_handle(1, 0)),
                            nir_imm_int(b, 0x30),
                            .access = ACCESS_CAN_REORDER, .align_mul = 4)));
   ASSERT_TRUE(run());
   EXPECT_EQ(words(), 1u);
   EXPECT_EQ(count(main, nir_intrinsic_load_global), 0u);
   EXPECT_EQ(count(main, nir_intrinsic_load_ssbo), 0u);
}

TEST_F(preamble_buffer_test, buffer_without_room_keeps_descriptor_load)
{
   inputs.fau.reserved = 254;
   sink(chain(uniform()));
   sink(nir_load_ssbo(b, 4, 32, nir_imm_int(b, pan_res_handle(1, 0)),
                      nir_ishl_imm(b, lane(), 4), .align_mul = 16));
   ASSERT_TRUE(run());
   EXPECT_EQ(words(), 1u);
   EXPECT_EQ(count(main, nir_intrinsic_load_ssbo), 1u);
   EXPECT_EQ(count(main, nir_intrinsic_load_ssbo_address), 0u);
   EXPECT_EQ(count(main, nir_intrinsic_load_global), 0u);
}

TEST_F(preamble_buffer_test, ssbo_store_uses_pilot_base_address)
{
   auto addr = nir_load_ssbo_address(
      b, 1, 64, nir_imm_int(b, pan_res_handle(1, 0)),
      nir_iadd_imm(b, nir_ishl_imm(b, lane(), 5), 8));
   nir_store_global(b, nir_imm_int(b, 7), addr, .align_mul = 32,
                    .align_offset = 8);
   ASSERT_TRUE(run());
   EXPECT_EQ(count(main, nir_intrinsic_load_ssbo_address), 0u);
   auto stores = intrinsics(main, nir_intrinsic_store_global);
   ASSERT_EQ(stores.size(), 1u);
   uint64_t imm;
   split_add_imm(stores[0]->src[1].ssa, &imm);
   EXPECT_EQ(imm, 8u);
}

TEST_F(preamble_buffer_test, atomic_uses_pilot_base_address)
{
   auto addr = nir_load_ssbo_address(
      b, 1, 64, nir_imm_int(b, pan_res_handle(3, 2)), nir_ishl_imm(b, lane(), 2));
   sink(nir_global_atomic(b, 32, addr, nir_imm_int(b, 1),
                          .atomic_op = nir_atomic_op_iadd));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(main, nir_intrinsic_load_ssbo_address), 0u);
   EXPECT_EQ(count(main, nir_intrinsic_global_atomic), 1u);
   EXPECT_EQ(words(), 2u);
}

TEST_F(preamble_buffer_test, conditional_access_reads_descriptor_in_pilot)
{
   auto id = lane();
   nir_push_if(b, nir_ine_imm(b, id, 0));
   sink(ssbo_load(1, 0, nir_ishl_imm(b, id, 4), 16, 0));
   nir_pop_if(b, nullptr);
   ASSERT_TRUE(run());
   EXPECT_EQ(count(main, nir_intrinsic_load_ssbo), 0u);
   EXPECT_EQ(count(main, nir_intrinsic_load_global), 1u);
}

TEST_F(preamble_buffer_test, pilot_reads_descriptor_address_inside_rack)
{
   sink(ssbo_load(5, 19, nir_ishl_imm(b, lane(), 4), 16, 0));
   ASSERT_TRUE(run());
   auto loads = intrinsics(pilot, nir_intrinsic_load_global_constant);
   ASSERT_EQ(loads.size(), 2u);
   EXPECT_EQ(loads[0]->def.num_components, 3u);
   EXPECT_EQ(loads[1]->def.num_components, 2u);
   EXPECT_EQ(nir_intrinsic_align_offset(loads[1]), 8u);
   EXPECT_EQ(count_alu(pilot, nir_op_bcsel), 1u);

   bool rack_check = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(pilot)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op == nir_op_uge && nir_src_is_const(alu->src[1].src) &&
             nir_src_as_uint(alu->src[1].src) == 20 * 32)
            rack_check = true;
      }
   }
   EXPECT_TRUE(rack_check);
}

TEST_F(preamble_buffer_test, null_descriptors_get_sink_address)
{
   inputs.robust_descriptors = true;
   sink(ssbo_load(1, 0, nir_ishl_imm(b, lane(), 4), 16, 0));
   ASSERT_TRUE(run());
   EXPECT_EQ(count(main, nir_intrinsic_load_global), 1u);
   EXPECT_EQ(count_alu(pilot, nir_op_bcsel), 2u);
}

TEST_F(preamble_buffer_test, buffer_bases_respect_fau_padding)
{
   inputs.fau.reserved = 248;
   auto x = nir_load_push_constant(b, 3, 32, nir_imm_int(b, 0), .range = 12,
                                   .align_mul = 16);
   auto y = nir_load_push_constant(b, 3, 32, nir_imm_int(b, 16), .range = 12,
                                   .align_mul = 16);
   sink(chain(x));
   sink(chain(y));
   sink(ssbo_load(1, 0, nir_ishl_imm(b, lane(), 4), 16, 0));
   ASSERT_TRUE(run());
   EXPECT_EQ(words(), 6u);
   EXPECT_EQ(count(main, nir_intrinsic_load_ssbo), 1u);
   EXPECT_EQ(count(main, nir_intrinsic_load_global), 0u);
}

TEST_F(preamble_buffer_test, buffer_bases_fill_padded_fau_room)
{
   inputs.fau.reserved = 246;
   auto x = nir_load_push_constant(b, 3, 32, nir_imm_int(b, 0), .range = 12,
                                   .align_mul = 16);
   auto y = nir_load_push_constant(b, 3, 32, nir_imm_int(b, 16), .range = 12,
                                   .align_mul = 16);
   sink(chain(x));
   sink(chain(y));
   sink(ssbo_load(1, 0, nir_ishl_imm(b, lane(), 4), 16, 0));
   ASSERT_TRUE(run());
   EXPECT_EQ(words(), 8u);
   EXPECT_EQ(count(main, nir_intrinsic_load_ssbo), 0u);
   EXPECT_EQ(count(main, nir_intrinsic_load_global), 1u);
}

TEST_F(preamble_buffer_test, robust_buffer_access_keeps_descriptor_loads)
{
   inputs.robust_modes = nir_var_mem_ssbo;
   sink(ssbo_load(1, 0, nir_ishl_imm(b, lane(), 4), 16, 0));
   EXPECT_FALSE(run());
}

TEST_F(preamble_buffer_test, other_architectures_keep_descriptor_loads)
{
   inputs.gpu_id = 0xa8670000;
   sink(ssbo_load(1, 0, nir_ishl_imm(b, lane(), 4), 16, 0));
   EXPECT_FALSE(run());
}

TEST_F(preamble_buffer_test, dynamic_handle_keeps_descriptor_loads)
{
   auto handle = nir_ior_imm(b, lane(), pan_res_handle(1, 0));
   sink(nir_load_ssbo(b, 4, 32, handle, nir_ishl_imm(b, lane(), 4),
                      .align_mul = 16));
   EXPECT_FALSE(run());
}

TEST_F(preamble_buffer_test, ubo_keeps_descriptor_loads)
{
   sink(nir_load_ubo(b, 4, 32, nir_imm_int(b, pan_res_handle(1, 0)),
                     nir_ishl_imm(b, lane(), 4), .align_mul = 16,
                     .range = 4096));
   EXPECT_FALSE(run());
}

TEST_F(preamble_buffer_test, fau_budget_keeps_most_used_buffers)
{
   inputs.fau.reserved = 252;
   auto offset = nir_ishl_imm(b, lane(), 4);
   for (unsigned i = 0; i < 3; i++) {
      for (unsigned n = 0; n <= i; n++)
         sink(ssbo_load(1, i, nir_iadd_imm(b, offset, 16 * n), 16, 0));
   }
   ASSERT_TRUE(run());
   EXPECT_EQ(words(), 4u);
   auto loads = intrinsics(main, nir_intrinsic_load_ssbo);
   ASSERT_EQ(loads.size(), 1u);
   EXPECT_EQ(nir_src_as_uint(loads[0]->src[0]), pan_res_handle(1, 0));
}

TEST_F(preamble_buffer_test, buffer_bases_follow_preamble_values)
{
   sink(chain(uniform()));
   sink(ssbo_load(1, 0, nir_ishl_imm(b, lane(), 4), 16, 0));
   ASSERT_TRUE(run());
   EXPECT_EQ(virt.value_count, 2u);
   EXPECT_EQ(words(), 3u);
   EXPECT_EQ(count(pilot, nir_intrinsic_store_preamble), 2u);
   EXPECT_EQ(count(main, nir_intrinsic_load_preamble), 2u);
}

TEST_F(preamble_buffer_test, compile_main_and_pilot)
{
   if (!pan_use_kraid(15, MESA_SHADER_COMPUTE, false))
      GTEST_SKIP();
   b->shader->options =
      pan_get_nir_shader_compiler_options(15, MESA_SHADER_COMPUTE, false);
   b->shader->info.workgroup_size[0] = 32;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;
   auto id = lane();
   auto value = ssbo_load(2, 1, nir_ishl_imm(b, id, 4), 16, 0);
   auto addr = nir_load_ssbo_address(b, 1, 64,
                                     nir_imm_int(b, pan_res_handle(2, 2)),
                                     nir_ishl_imm(b, id, 4));
   nir_store_global(b, value, addr, .align_mul = 16);
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
