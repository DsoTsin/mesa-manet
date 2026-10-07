#include "compiler/nir/tests/nir_test.h"
extern "C" {
#include "panfrost/compiler/pan_nir.h"
}

class robust_image_test : public nir_test {
 protected:
   robust_image_test(): nir_test("pan robust image") {}

   nir_tex_instr *fetch(glsl_sampler_dim dim, bool array, nir_def *coord,
                        int level = 0, nir_def *offset = nullptr)
   {
      auto tex = nir_tex_instr_create(b->shader,
         1 + (dim != GLSL_SAMPLER_DIM_BUF) + (offset != nullptr));
      tex->op = dim == GLSL_SAMPLER_DIM_MS ? nir_texop_txf_ms : nir_texop_txf;
      tex->sampler_dim = dim;
      tex->is_array = array;
      tex->coord_components = coord->num_components;
      tex->dest_type = nir_type_uint32;
      tex->texture_index = 7;
      tex->src[0].src_type = nir_tex_src_coord;
      tex->src[0].src = nir_src_for_ssa(coord);
      unsigned i = 1;
      if (dim != GLSL_SAMPLER_DIM_BUF) {
         tex->src[i].src_type = dim == GLSL_SAMPLER_DIM_MS ?
                               nir_tex_src_ms_index : nir_tex_src_lod;
         tex->src[i++].src = nir_src_for_ssa(nir_imm_int(b, level));
      }
      if (offset) {
         tex->src[i].src_type = nir_tex_src_offset;
         tex->src[i].src = nir_src_for_ssa(offset);
      }
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(b, &tex->instr);
      nir_store_global(b, &tex->def, nir_imm_int64(b, 4096), .align_mul = 4);
      return tex;
   }

   void lower(nir_tex_instr *fetch, unsigned layers = 2)
   {
      nir_validate_shader(b->shader, "before robust image");
      ASSERT_TRUE(pan_nir_lower_robust_image_access2(b->shader));
      nir_validate_shader(b->shader, "after robust image");
      nir_foreach_block(block, b->impl) nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_tex || instr == &fetch->instr)
            continue;
         auto query = nir_instr_as_tex(instr);
         EXPECT_EQ(query->texture_index, 7u);
         b->cursor = nir_before_instr(instr);
         nir_def *value;
         switch (query->op) {
         case nir_texop_txs: {
            nir_def *dims[4];
            for (unsigned i = 0; i < query->def.num_components; i++)
               dims[i] = nir_imm_int(b, query->is_array &&
                  i + 1 == query->def.num_components ? layers : 4);
            value = nir_vec(b, dims, query->def.num_components);
            break;
         }
         case nir_texop_query_levels: value = nir_imm_int(b, 3); break;
         case nir_texop_texture_samples: value = nir_imm_int(b, 4); break;
         default: FAIL(); return;
         }
         nir_def_replace(&query->def, value);
      }
      while (nir_opt_constant_folding(b->shader)) {}
      nir_validate_shader(b->shader, "folded robust image");
   }

   uint32_t src(nir_tex_instr *tex, nir_tex_src_type type, unsigned c = 0)
   {
      int i = nir_tex_instr_src_index(tex, type);
      EXPECT_GE(i, 0);
      auto s = nir_scalar_resolved(tex->src[i].src.ssa, c);
      EXPECT_TRUE(nir_scalar_is_const(s));
      return nir_scalar_as_uint(s);
   }
};

TEST_F(robust_image_test, mip_and_coordinate_bounds)
{
   auto tex = fetch(GLSL_SAMPLER_DIM_2D, false, nir_imm_ivec2(b, 1, 1), 1);
   lower(tex);
   EXPECT_EQ(src(tex, nir_tex_src_coord), 1u);
   EXPECT_EQ(src(tex, nir_tex_src_lod), 1u);
}

TEST_F(robust_image_test, mip_coordinate_out_of_bounds)
{
   auto tex = fetch(GLSL_SAMPLER_DIM_2D, false, nir_imm_ivec2(b, 2, 1), 1);
   lower(tex);
   EXPECT_EQ(src(tex, nir_tex_src_coord), uint32_t(INT32_MAX));
}

TEST_F(robust_image_test, invalid_lods_cannot_clamp_or_wrap)
{
   auto tex = fetch(GLSL_SAMPLER_DIM_2D, false, nir_imm_ivec2(b, 0, 0), 256);
   lower(tex);
   EXPECT_EQ(src(tex, nir_tex_src_coord), uint32_t(INT32_MAX));
   EXPECT_EQ(src(tex, nir_tex_src_lod), 0u);
}

TEST_F(robust_image_test, negative_lod)
{
   auto tex = fetch(GLSL_SAMPLER_DIM_2D, false, nir_imm_ivec2(b, 0, 0), -1);
   lower(tex);
   EXPECT_EQ(src(tex, nir_tex_src_coord), uint32_t(INT32_MAX));
   EXPECT_EQ(src(tex, nir_tex_src_lod), 0u);
}

TEST_F(robust_image_test, first_invalid_lod)
{
   auto tex = fetch(GLSL_SAMPLER_DIM_2D, false, nir_imm_ivec2(b, 0, 0), 3);
   lower(tex);
   EXPECT_EQ(src(tex, nir_tex_src_coord), uint32_t(INT32_MAX));
   EXPECT_EQ(src(tex, nir_tex_src_lod), 0u);
}

TEST_F(robust_image_test, array_layers_do_not_shrink_with_mips)
{
   auto tex = fetch(GLSL_SAMPLER_DIM_2D, true, nir_imm_ivec3(b, 0, 0, 1), 2);
   lower(tex);
   EXPECT_EQ(src(tex, nir_tex_src_coord), 0u);
   EXPECT_EQ(src(tex, nir_tex_src_coord, 2), 1u);
}

TEST_F(robust_image_test, negative_array_layer)
{
   auto tex = fetch(GLSL_SAMPLER_DIM_2D, true, nir_imm_ivec3(b, 0, 0, -1));
   lower(tex);
   EXPECT_EQ(src(tex, nir_tex_src_coord), uint32_t(INT32_MAX));
}

TEST_F(robust_image_test, offset_crosses_view_bounds)
{
   auto tex = fetch(GLSL_SAMPLER_DIM_2D, true, nir_imm_ivec3(b, 3, 0, 1), 0,
                    nir_imm_ivec2(b, 1, 0));
   lower(tex);
   EXPECT_EQ(src(tex, nir_tex_src_coord), uint32_t(INT32_MAX));
   EXPECT_EQ(nir_tex_instr_src_index(tex, nir_tex_src_offset), -1);
}

TEST_F(robust_image_test, sample_index_cannot_wrap)
{
   auto tex = fetch(GLSL_SAMPLER_DIM_MS, true, nir_imm_ivec3(b, 0, 0, 1), 256);
   lower(tex);
   EXPECT_EQ(src(tex, nir_tex_src_coord), uint32_t(INT32_MAX));
   EXPECT_EQ(src(tex, nir_tex_src_ms_index), 0u);
}

TEST_F(robust_image_test, texel_buffer_negative_index)
{
   auto tex = fetch(GLSL_SAMPLER_DIM_BUF, false, nir_imm_int(b, -1));
   lower(tex);
   EXPECT_EQ(src(tex, nir_tex_src_coord), uint32_t(INT32_MAX));
}
