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
      fau.reserved = fau.count = 8;
      fau.max = PAN_MAX_PUSH;
      for (unsigned i = 0; i < fau.reserved; i++)
         fau.words[i].constant = 0xcafe0000 | i;
   }

   pan_compile_inputs inputs = {};
   pan_fau_layout fau = {};
   uint32_t ubo_mask = 0;

   nir_def *load(unsigned table, unsigned index, unsigned offset,
                 unsigned components = 1, unsigned bits = 32)
   {
      return nir_load_ubo(b, components, bits,
                          nir_imm_int(b, (table << 24) | index),
                          nir_imm_int(b, offset), .align_mul = 4);
   }

   bool run()
   {
      return pan_nir_opt_push_ubo(b->shader, &inputs, &fau, &ubo_mask);
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

   bool has(unsigned table, unsigned index, unsigned offset)
   {
      pan_fau_foreach_reloc(&fau, i) {
         auto reloc = fau.words[i].relocation;
         if (reloc.ubo == pan_ubo_reloc_key(table, index) &&
             reloc.offset == offset)
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
         if (intr->intrinsic != nir_intrinsic_load_push_constant)
            continue;
         unsigned word = nir_src_as_uint(intr->src[0]) / 4;
         ASSERT_LT(word, fau.count);
         ASSERT_GE(word, fau.reserved);
         auto reloc = fau.words[word].relocation;
         uint32_t value = (uint32_t(reloc.ubo) << 16) | reloc.offset;
         b->cursor = nir_before_instr(instr);
         nir_def_replace(&intr->def, nir_imm_int(b, value));
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
   EXPECT_EQ(fau.count, 12);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 0);
   EXPECT_TRUE(has(0, 7, 0));
   EXPECT_TRUE(has(1, 7, 4));
   EXPECT_TRUE(has(2, 7, 8));
   EXPECT_TRUE(has(63, 1023, 12));
   for (unsigned i = 0; i < fau.reserved; i++)
      EXPECT_EQ(fau.words[i].constant, 0xcafe0000 | i);
}

TEST_F(push_ubo_test, existing_push_constant_and_reserved_prefix)
{
   nir_def *push = nir_load_push_constant(b, 1, 32, nir_imm_int(b, 4),
                                         .align_mul = 4);
   auto store = nir_store_ssbo(b, push, nir_imm_int(b, 0), nir_imm_int(b, 0));
   load(1, 1, 0);
   ASSERT_TRUE(run());
   EXPECT_EQ(store->src[0].ssa, push);
   EXPECT_EQ(fau.count, 9);
   EXPECT_EQ(ubo_mask, 0);
   EXPECT_TRUE(has(1, 1, 0));
   EXPECT_EQ(fau.words[1].constant, 0xcafe0001u);
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
   EXPECT_EQ(fau.count, fau.reserved);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 7);
}

TEST_F(push_ubo_test, overlapping_vectors_preserve_values)
{
   fau.max = fau.reserved + 3;
   auto a = load(2, 5, 0, 2);
   auto c = load(2, 5, 4, 2);
   auto sink_a = nir_store_ssbo(b, a, nir_imm_int(b, 0), nir_imm_int(b, 0));
   auto sink_c = nir_store_ssbo(b, c, nir_imm_int(b, 0), nir_imm_int(b, 8));
   ASSERT_TRUE(run());
   EXPECT_EQ(fau.count, 11);
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
   fau.max = fau.reserved + 2;
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
   fau.max = fau.reserved + 1;
   load(1, 2, 0);
   nir_loop *loop = nir_push_loop(b);
   load(2, 10, 0, 4);
   nir_jump(b, nir_jump_break);
   nir_pop_loop(b, loop);
   ASSERT_TRUE(run());
   EXPECT_TRUE(has(1, 2, 0));
   EXPECT_EQ(fau.count, fau.max);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 1);
}

TEST_F(push_ubo_test, full_reserved_fau)
{
   fau.reserved = fau.count = fau.max;
   load(1, 2, 0);
   EXPECT_FALSE(run());
   EXPECT_EQ(fau.count, PAN_MAX_PUSH);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 1);
}

TEST_F(push_ubo_test, zero_key_and_last_word)
{
   load(0, 0, 16380);
   ASSERT_TRUE(run());
   EXPECT_TRUE(has(0, 0, 16380));
   EXPECT_EQ(fau.count, 9);
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 0);
}

TEST_F(push_ubo_test, gallium_ubo_mask)
{
   inputs.fau.push_ubo_handles = false;
   inputs.fau.pushable_ubos = BITFIELD_BIT(1);
   fau.reserved = fau.count = 0;
   b->shader->info.num_ubos = 3;
   load(0, 2, 0);
   EXPECT_FALSE(run());
   EXPECT_EQ(ubo_mask, BITFIELD_BIT(2));
   EXPECT_EQ(count(nir_intrinsic_load_ubo), 1);
}

TEST_F(push_ubo_test, gallium_blend_constants)
{
   inputs.fau.push_ubo_handles = false;
   fau.reserved = fau.count = 0;
   b->shader->info.num_ubos = 2;
   load(0, 1, 0, 4);
   ASSERT_TRUE(run());
   EXPECT_EQ(fau.count, 4);
   EXPECT_EQ(ubo_mask, 0);
   for (unsigned i = 0; i < 4; i++) {
      EXPECT_EQ(fau.words[i].relocation.ubo, 1);
      EXPECT_EQ(fau.words[i].relocation.offset, i * 4);
   }
}

TEST_F(push_ubo_test, odd_prefix_and_64bit_values)
{
   fau.reserved = fau.count = 5;
   auto a = load(1, 3, 0, 1, 64);
   auto c = load(2, 3, 0, 1, 64);
   auto sum = nir_iadd(b, a, c);
   auto sink = nir_store_ssbo(b, sum, nir_imm_int(b, 0), nir_imm_int(b, 0));
   ASSERT_TRUE(run());
   EXPECT_EQ(fau.words[4].constant, 0xcafe0004u);
   resolve_pushes();
   ASSERT_TRUE(nir_src_is_const(sink->src[0]));
   uint64_t av = uint64_t(pan_ubo_reloc_key(1, 3)) << 16;
   uint64_t cv = uint64_t(pan_ubo_reloc_key(2, 3)) << 16;
   EXPECT_EQ(nir_src_as_uint(sink->src[0]), (av | ((av | 4) << 32)) +
                                          (cv | ((cv | 4) << 32)));
}
