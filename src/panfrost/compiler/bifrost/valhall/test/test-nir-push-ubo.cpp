#include "compiler/nir/tests/nir_test.h"
extern "C" {
#include "panfrost/compiler/pan_nir.h"
}

class push_ubo_test : public nir_test {
protected:
   push_ubo_test() : nir_test("pan push ubo", MESA_SHADER_FRAGMENT)
   {
      inputs.fau.pushable_ubos = ~0u;
      inputs.fau.push_ubo_handles = true;
      memset(virt.map, 0xff, sizeof(virt.map));
   }

   pan_compile_inputs inputs = {};
   pan_fau_virtual virt = {};
   unsigned budget = PAN_MAX_PUSH - 8;
   uint32_t ubo_mask = 0;

   nir_def *load(unsigned table, unsigned index, unsigned offset,
                 unsigned components = 1, unsigned bits = 32)
   {
      nir_def *def = nir_load_ubo(b, components, bits,
                                  nir_imm_int(b, (table << 24) | index),
                                  nir_imm_int(b, offset), .align_mul = 4);
      nir_store_ssbo(b, def, nir_imm_int(b, 0), nir_imm_int(b, 0));
      return def;
   }

   bool run()
   {
      return pan_nir_opt_push_ubo(b->shader, &inputs, &virt, budget,
                                  &ubo_mask);
   }

   unsigned count(nir_intrinsic_op op)
   {
      unsigned n = 0;
      nir_foreach_block(block, b->impl)
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == op)
            n++;
      }
      return n;
   }

   unsigned words()
   {
      return pan_fau_virtual_words(&virt);
   }

   bool has(unsigned table, unsigned index, unsigned offset)
   {
      for (unsigned i = 0; i < virt.value_count; i++) {
         const pan_fau_value *value = &virt.values[i];
         if (value->source == PAN_FAU_VALUE_UBO &&
             value->ubo.ubo == pan_ubo_reloc_key(table, index) &&
             offset >= value->ubo.offset &&
             offset < value->ubo.offset + 4u * value->size)
            return true;
      }
      return false;
   }

   void resolve_pushes()
   {
      nir_foreach_block(block, b->impl)
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_preamble)
            continue;
         unsigned word = nir_intrinsic_base(intr);
         ASSERT_LT(word, virt.word_count);
         const pan_fau_value *value = &virt.values[virt.value_of[word]];
         ASSERT_EQ(value->source, PAN_FAU_VALUE_UBO);
         uint32_t offset = value->ubo.offset + 4 * (word - value->word);
         uint32_t result = (uint32_t(value->ubo.ubo) << 16) | offset;
         b->cursor = nir_before_instr(instr);
         nir_def_replace(&intr->def, nir_imm_int(b, result));
      }
      while (nir_opt_constant_folding(b->shader)) {}
   }
};

TEST_F(push_ubo_test, distinct_tables_and_driver_table)
{
   load(0, 7, 0);
   load(1, 7, 4);
   load(2, 7, 8);
   load(63, 1023, 12);
   ASSERT_TRUE(run());
   EXPECT_EQ(words(), 4u);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 0);
   EXPECT_TRUE(has(0, 7, 0));
   EXPECT_TRUE(has(1, 7, 4));
   EXPECT_TRUE(has(2, 7, 8));
   EXPECT_TRUE(has(63, 1023, 12));
}

TEST_F(push_ubo_test, existing_push_constant_is_kept)
{
   nir_def *push = nir_load_push_constant(b, 1, 32, nir_imm_int(b, 4),
                                         .align_mul = 4);
   auto store = nir_store_ssbo(b, push, nir_imm_int(b, 0), nir_imm_int(b, 0));
   load(1, 1, 0);
   ASSERT_TRUE(run());
   EXPECT_EQ(store->src[0].ssa, push);
   EXPECT_EQ(words(), 1u);
   EXPECT_EQ(ubo_mask, 0);
   EXPECT_TRUE(has(1, 1, 0));
}

TEST_F(push_ubo_test, unsupported_accesses_stay_as_ubo)
{
   load(1, 1024, 0);
   load(64, 1, 0);
   load(1, 1, 16380, 2);
   load(1, 1, 0xfffffffc);
   load(1, 1, 2);
   nir_def *dynamic = nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0), .align_mul = 4);
   nir_load_ubo(b, 1, 32, dynamic, nir_imm_int(b, 0), .align_mul = 4);
   nir_load_ubo(b, 1, 32, nir_imm_int(b, 1 << 24), dynamic, .align_mul = 4);
   EXPECT_FALSE(run());
   EXPECT_EQ(words(), 0u);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 7);
}

TEST_F(push_ubo_test, overlapping_vectors_preserve_values)
{
   budget = 3;
   auto a = load(2, 5, 0, 2);
   auto c = load(2, 5, 4, 2);
   auto sink_a = nir_store_ssbo(b, a, nir_imm_int(b, 0), nir_imm_int(b, 0));
   auto sink_c = nir_store_ssbo(b, c, nir_imm_int(b, 0), nir_imm_int(b, 8));
   ASSERT_TRUE(run());
   EXPECT_EQ(words(), 3u);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 0);
   resolve_pushes();
   ASSERT_TRUE(nir_src_is_const(sink_a->src[0]));
   ASSERT_TRUE(nir_src_is_const(sink_c->src[0]));
   const nir_const_value *av = nir_src_as_const_value(sink_a->src[0]);
   const nir_const_value *cv = nir_src_as_const_value(sink_c->src[0]);
   uint32_t key = pan_ubo_reloc_key(2, 5) << 16;
   EXPECT_EQ(av[0].u32, key);
   EXPECT_EQ(av[1].u32, key | 4);
   EXPECT_EQ(cv[0].u32, key | 4);
   EXPECT_EQ(cv[1].u32, key | 8);
}

TEST_F(push_ubo_test, loop_use_precedes_cold_range)
{
   budget = 2;
   load(2, 10, 0, 2);
   nir_loop *loop = nir_push_loop(b);
   load(1, 2, 8);
   nir_jump(b, nir_jump_break);
   nir_pop_loop(b, loop);
   ASSERT_TRUE(run());
   EXPECT_TRUE(has(1, 2, 8));
   EXPECT_FALSE(has(2, 10, 0));
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 1);
}

TEST_F(push_ubo_test, skips_unfittable_candidate)
{
   budget = 1;
   load(1, 2, 0);
   nir_loop *loop = nir_push_loop(b);
   load(2, 10, 0, 4);
   nir_jump(b, nir_jump_break);
   nir_pop_loop(b, loop);
   ASSERT_TRUE(run());
   EXPECT_TRUE(has(1, 2, 0));
   EXPECT_EQ(words(), 1u);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 1);
}

TEST_F(push_ubo_test, no_budget)
{
   budget = 0;
   load(1, 2, 0);
   EXPECT_FALSE(run());
   EXPECT_EQ(words(), 0u);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 1);
}

TEST_F(push_ubo_test, full_v15_fau_component)
{
   budget = pan_max_push(0x0f080000f0000000ull, 4);
   nir_def *words_[256];
   for (unsigned i = 0; i < 256; i++)
      words_[i] = load(1, 2, i * 4);
   nir_def *value = nir_imm_int(b, 0);
   for (unsigned i = 0; i < 256; i++)
      value = nir_iadd(b, value, nir_iadd(b, words_[i], words_[(i + 1) % 256]));
   auto sink = nir_store_ssbo(b, value, nir_imm_int(b, 0), nir_imm_int(b, 0));
   ASSERT_TRUE(run());
   EXPECT_EQ(words(), 256u);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 0);
   uint32_t expected = 0;
   for (unsigned i = 0; i < 256; i++) {
      EXPECT_TRUE(has(1, 2, i * 4));
      expected += 2 * ((pan_ubo_reloc_key(1, 2) << 16) | (i * 4));
   }
   resolve_pushes();
   ASSERT_TRUE(nir_src_is_const(sink->src[0]));
   EXPECT_EQ(nir_src_as_uint(sink->src[0]), expected);
}

TEST_F(push_ubo_test, model_fau_capacity)
{
   EXPECT_EQ(pan_max_push(0x0f080000f0000000ull, 4), 256);
   EXPECT_EQ(pan_max_push(0x0f080000f0000000ull, 0), 128);
   EXPECT_EQ(pan_max_push(0x0f080001f0000000ull, 4), 128);
   EXPECT_EQ(pan_max_push(0x0f080003f0000000ull, 0), 128);
   EXPECT_EQ(pan_max_push(0x0f080003f0000000ull, 4), 128);
   EXPECT_EQ(pan_max_push(0xc8000000, 4), 128);
   EXPECT_EQ(pan_max_push(0xa8670000, 0), 128);
   EXPECT_EQ(pan_max_push(0, 0), 128);
}

TEST_F(push_ubo_test, zero_key_and_last_word)
{
   load(0, 0, 16380);
   ASSERT_TRUE(run());
   EXPECT_TRUE(has(0, 0, 16380));
   EXPECT_EQ(words(), 1u);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 0);
}

TEST_F(push_ubo_test, gallium_ubo_mask)
{
   inputs.fau.push_ubo_handles = false;
   inputs.fau.pushable_ubos = BITFIELD_BIT(1);
   b->shader->info.num_ubos = 3;
   load(0, 2, 0);
   EXPECT_FALSE(run());
   EXPECT_EQ(ubo_mask, BITFIELD_BIT(2));
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 1);
}

TEST_F(push_ubo_test, vec4_is_one_aligned_chunk)
{
   inputs.gpu_id = 0x0f080000f0000000ull;
   inputs.gpu_variant = 4;
   load(1, 4, 16, 4);
   ASSERT_TRUE(run());
   ASSERT_EQ(virt.value_count, 1u);
   EXPECT_EQ(virt.values[0].size, 4u);
   EXPECT_EQ(virt.values[0].align, 4u);
   EXPECT_EQ(virt.values[0].ubo.offset, 16u);
}

TEST_F(push_ubo_test, chunks_stop_at_16_byte_boundaries)
{
   inputs.gpu_id = 0x0f080000f0000000ull;
   inputs.gpu_variant = 4;
   load(1, 4, 8, 4);
   ASSERT_TRUE(run());
   ASSERT_EQ(virt.value_count, 2u);
   EXPECT_EQ(virt.values[0].ubo.offset, 8u);
   EXPECT_EQ(virt.values[0].size, 2u);
   EXPECT_EQ(virt.values[1].ubo.offset, 16u);
   EXPECT_EQ(virt.values[1].size, 2u);
}

TEST_F(push_ubo_test, values_never_share_a_virtual_pair)
{
   load(1, 4, 0);
   load(1, 4, 8);
   ASSERT_TRUE(run());
   ASSERT_EQ(virt.value_count, 2u);
   EXPECT_EQ(virt.values[0].word % 2, 0u);
   EXPECT_EQ(virt.values[1].word % 2, 0u);
}

TEST_F(push_ubo_test, sixty_four_bit_values)
{
   auto a = load(1, 3, 0, 1, 64);
   auto c = load(2, 3, 0, 1, 64);
   auto sum = nir_iadd(b, a, c);
   auto sink = nir_store_ssbo(b, sum, nir_imm_int(b, 0), nir_imm_int(b, 0));
   ASSERT_TRUE(run());
   resolve_pushes();
   ASSERT_TRUE(nir_src_is_const(sink->src[0]));
   uint64_t av = uint64_t(pan_ubo_reloc_key(1, 3)) << 16;
   uint64_t cv = uint64_t(pan_ubo_reloc_key(2, 3)) << 16;
   EXPECT_EQ(nir_src_as_uint(sink->src[0]), (av | ((av | 4) << 32)) +
                                          (cv | ((cv | 4) << 32)));
}

TEST_F(push_ubo_test, unread_components_are_not_pushed)
{
   nir_def *v = nir_load_ubo(b, 4, 32, nir_imm_int(b, (1 << 24) | 4),
                             nir_imm_int(b, 16), .align_mul = 4);
   nir_def *sum = nir_iadd(b, nir_channel(b, v, 0), nir_channel(b, v, 2));
   nir_store_ssbo(b, sum, nir_imm_int(b, 0), nir_imm_int(b, 0));
   ASSERT_TRUE(run());
   EXPECT_EQ(words(), 2u);
   EXPECT_TRUE(has(1, 4, 16));
   EXPECT_FALSE(has(1, 4, 20));
   EXPECT_TRUE(has(1, 4, 24));
   EXPECT_FALSE(has(1, 4, 28));
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 0);
}

TEST_F(push_ubo_test, values_carry_their_handle)
{
   load(3, 9, 32);
   ASSERT_TRUE(run());
   ASSERT_EQ(virt.value_count, 1u);
   EXPECT_TRUE(virt.values[0].loadable);
   EXPECT_EQ(virt.values[0].handle, (3u << 24) | 9);
}

TEST_F(push_ubo_test, single_fau_index_pushes_words)
{
   inputs.gpu_id = 0xa8670000;
   load(1, 4, 16, 4);
   ASSERT_TRUE(run());
   ASSERT_EQ(virt.value_count, 4u);
   for (unsigned i = 0; i < 4; i++) {
      EXPECT_EQ(virt.values[i].size, 1u);
      EXPECT_EQ(virt.values[i].ubo.offset, 16u + 4 * i);
   }
}
