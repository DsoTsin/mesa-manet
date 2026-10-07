#include "compiler/nir/tests/nir_test.h"
extern "C" {
#include "panfrost/compiler/pan_nir.h"
}

class pan_link_varyings_test : public nir_test {
protected:
   pan_link_varyings_test() : nir_test("linked VS", MESA_SHADER_VERTEX)
   {
      fs = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &options,
                                          "linked FS");
   }

   ~pan_link_varyings_test() override { ralloc_free(fs.shader); }

   nir_builder fs;

   nir_variable *var(nir_builder *builder, nir_variable_mode mode,
                     const glsl_type *type, unsigned location,
                     unsigned interpolation = INTERP_MODE_SMOOTH)
   {
      nir_variable *v = nir_variable_create(builder->shader, mode, type, "io");
      v->data.location = location;
      v->data.interpolation = interpolation;
      return v;
   }

   nir_variable *out(unsigned location = VARYING_SLOT_VAR0)
   {
      return var(b, nir_var_shader_out, glsl_float_type(), location);
   }

   nir_variable *in(unsigned location = VARYING_SLOT_VAR0,
                    unsigned interpolation = INTERP_MODE_SMOOTH)
   {
      return var(&fs, nir_var_shader_in, glsl_float_type(), location,
                  interpolation);
   }

   void sink(nir_def *value)
   {
      nir_variable *color = var(&fs, nir_var_shader_out,
                                glsl_float_type(), FRAG_RESULT_DATA0);
      nir_store_var(&fs, color, value, 1);
   }

   nir_def *vertex_value() { return nir_u2f32(b, nir_load_vertex_id(b)); }

   unsigned io_count(nir_shader *nir, nir_variable_mode mode)
   {
      unsigned count = 0;
      nir_foreach_variable_with_modes(v, nir, mode)
         count++;
      return count;
   }

   unsigned intrinsic_count(nir_shader *nir, nir_intrinsic_op op)
   {
      unsigned count = 0;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               count += instr->type == nir_instr_type_intrinsic &&
                        nir_instr_as_intrinsic(instr)->intrinsic == op;
            }
         }
      }
      return count;
   }

   bool link()
   {
      bool progress = pan_nir_link_varyings(b->shader, fs.shader);
      nir_validate_shader(b->shader, "linked producer");
      nir_validate_shader(fs.shader, "linked consumer");
      return progress;
   }
};

TEST_F(pan_link_varyings_test, remove_unused_output)
{
   nir_store_var(b, out(), vertex_value(), 1);
   sink(nir_imm_float(&fs, 0.5f));
   EXPECT_TRUE(link());
   EXPECT_EQ(io_count(b->shader, nir_var_shader_out), 0u);
   EXPECT_EQ(intrinsic_count(b->shader, nir_intrinsic_load_vertex_id), 0u);
}

TEST_F(pan_link_varyings_test, propagate_constant)
{
   nir_store_var(b, out(), nir_imm_float(b, 0.25f), 1);
   sink(nir_load_var(&fs, in()));
   EXPECT_TRUE(link());
   EXPECT_EQ(io_count(b->shader, nir_var_shader_out), 0u);
   EXPECT_EQ(io_count(fs.shader, nir_var_shader_in), 0u);
   nir_foreach_block(block, fs.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_deref) {
            ASSERT_TRUE(nir_src_is_const(intr->src[1]));
            EXPECT_FLOAT_EQ(nir_src_as_float(intr->src[1]), 0.25f);
         }
      }
   }
}

TEST_F(pan_link_varyings_test, deduplicate_outputs)
{
   auto value = vertex_value();
   nir_store_var(b, out(), value, 1);
   nir_store_var(b, out(VARYING_SLOT_VAR0 + 1), value, 1);
   sink(nir_fadd(&fs, nir_load_var(&fs, in()),
                 nir_load_var(&fs, in(VARYING_SLOT_VAR0 + 1))));
   EXPECT_TRUE(link());
   EXPECT_EQ(io_count(b->shader, nir_var_shader_out), 1u);
   EXPECT_EQ(io_count(fs.shader, nir_var_shader_in), 1u);
}

TEST_F(pan_link_varyings_test, keep_distinct_interpolation)
{
   auto value = vertex_value();
   nir_store_var(b, out(), value, 1);
   nir_store_var(b, out(VARYING_SLOT_VAR0 + 1), value, 1);
   sink(nir_fadd(&fs, nir_load_var(&fs, in()),
                 nir_load_var(&fs, in(VARYING_SLOT_VAR0 + 1, INTERP_MODE_FLAT))));
   link();
   EXPECT_EQ(io_count(b->shader, nir_var_shader_out), 2u);
   EXPECT_EQ(io_count(fs.shader, nir_var_shader_in), 2u);
}

TEST_F(pan_link_varyings_test, keep_distinct_centroid)
{
   auto value = vertex_value();
   nir_store_var(b, out(), value, 1);
   nir_store_var(b, out(VARYING_SLOT_VAR0 + 1), value, 1);
   auto centroid = in(VARYING_SLOT_VAR0 + 1);
   centroid->data.centroid = true;
   sink(nir_fadd(&fs, nir_load_var(&fs, in()), nir_load_var(&fs, centroid)));
   link();
   EXPECT_EQ(io_count(fs.shader, nir_var_shader_in), 2u);
}

TEST_F(pan_link_varyings_test, prune_unused_vector_components)
{
   auto output = var(b, nir_var_shader_out, glsl_vec4_type(), VARYING_SLOT_VAR0);
   auto input = var(&fs, nir_var_shader_in, glsl_vec4_type(), VARYING_SLOT_VAR0);
   auto value = vertex_value();
   nir_store_var(b, output, nir_vec4(b, value, value, value, value), 15);
   sink(nir_channel(&fs, nir_load_var(&fs, input), 0));
   EXPECT_TRUE(link());
   EXPECT_EQ(io_count(b->shader, nir_var_shader_out), 1u);
   EXPECT_EQ(io_count(fs.shader, nir_var_shader_in), 1u);
}

TEST_F(pan_link_varyings_test, keep_position_output)
{
   auto position = var(b, nir_var_shader_out, glsl_vec4_type(), VARYING_SLOT_POS);
   nir_store_var(b, position, nir_imm_vec4(b, 0, 0, 0, 1), 15);
   nir_store_var(b, out(), vertex_value(), 1);
   sink(nir_imm_float(&fs, 1));
   link();
   EXPECT_TRUE(b->shader->info.outputs_written & VARYING_BIT_POS);
   EXPECT_EQ(io_count(b->shader, nir_var_shader_out), 1u);
}

TEST_F(pan_link_varyings_test, preserve_side_effects)
{
   auto value = vertex_value();
   nir_store_global(b, value, nir_imm_int64(b, 4096), .align_mul = 4);
   nir_store_var(b, out(), value, 1);
   sink(nir_imm_float(&fs, 1));
   link();
   EXPECT_EQ(io_count(b->shader, nir_var_shader_out), 0u);
   EXPECT_EQ(intrinsic_count(b->shader, nir_intrinsic_store_global), 1u);
}

TEST_F(pan_link_varyings_test, conditional_values_are_not_constant)
{
   auto output = out();
   nir_push_if(b, nir_ieq_imm(b, nir_load_vertex_id(b), 0));
   nir_store_var(b, output, nir_imm_float(b, 0.25f), 1);
   nir_push_else(b, NULL);
   nir_store_var(b, output, nir_imm_float(b, 0.75f), 1);
   nir_pop_if(b, NULL);
   sink(nir_load_var(&fs, in()));
   link();
   EXPECT_EQ(io_count(fs.shader, nir_var_shader_in), 1u);
}

TEST_F(pan_link_varyings_test, skip_transform_feedback)
{
   b->shader->info.has_transform_feedback_varyings = true;
   nir_store_var(b, out(), vertex_value(), 1);
   sink(nir_imm_float(&fs, 1));
   EXPECT_FALSE(link());
   EXPECT_EQ(io_count(b->shader, nir_var_shader_out), 1u);
}

TEST_F(pan_link_varyings_test, skip_always_active_output)
{
   auto output = out();
   output->data.always_active_io = true;
   nir_store_var(b, output, vertex_value(), 1);
   sink(nir_imm_float(&fs, 1));
   EXPECT_FALSE(link());
   EXPECT_EQ(io_count(b->shader, nir_var_shader_out), 1u);
}

TEST_F(pan_link_varyings_test, skip_non_vertex_producer)
{
   b->shader->info.stage = MESA_SHADER_TESS_EVAL;
   EXPECT_FALSE(pan_nir_link_varyings(b->shader, fs.shader));
}

TEST_F(pan_link_varyings_test, preserve_negative_zero_bits)
{
   nir_store_var(b, out(), nir_imm_int(b, 0x80000000u), 1);
   sink(nir_load_var(&fs, in()));
   EXPECT_TRUE(link());
   nir_foreach_block(block, fs.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_deref) {
            ASSERT_TRUE(nir_src_is_const(intr->src[1]));
            EXPECT_EQ(nir_src_as_uint(intr->src[1]), 0x80000000u);
         }
      }
   }
}

TEST_F(pan_link_varyings_test, preserve_nan_payload)
{
   nir_store_var(b, out(), nir_imm_int(b, 0x7fc12345u), 1);
   sink(nir_load_var(&fs, in()));
   link();
   nir_foreach_block(block, fs.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         auto intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_deref) {
            ASSERT_TRUE(nir_src_is_const(intr->src[1]));
            EXPECT_EQ(nir_src_as_uint(intr->src[1]), 0x7fc12345u);
         }
      }
   }
}

TEST_F(pan_link_varyings_test, do_not_clone_uniform_resources)
{
   options.max_varying_expression_cost = 2;
   auto uniform = var(b, nir_var_uniform, glsl_float_type(), 0);
   nir_store_var(b, out(), nir_load_var(b, uniform), 1);
   sink(nir_load_var(&fs, in()));
   const auto original = fs.shader->options;
   link();
   EXPECT_EQ(io_count(fs.shader, nir_var_shader_in), 1u);
   EXPECT_EQ(io_count(fs.shader, nir_var_uniform), 0u);
   EXPECT_EQ(fs.shader->options, original);
}
