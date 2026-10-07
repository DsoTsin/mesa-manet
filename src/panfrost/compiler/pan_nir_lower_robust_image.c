#include "pan_nir.h"

static nir_def *
texture_query(nir_builder *b, const nir_tex_instr *tex, nir_texop op,
              unsigned components)
{
   nir_tex_src sources[8];
   unsigned count = 0;
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_deref:
      case nir_tex_src_texture_handle:
      case nir_tex_src_texture_offset:
         sources[count++] = (nir_tex_src) {
            .src_type = tex->src[i].src_type,
            .src = nir_src_for_ssa(tex->src[i].src.ssa),
         };
         break;
      default:
         break;
      }
   }
   if (op == nir_texop_txs && tex->sampler_dim != GLSL_SAMPLER_DIM_BUF &&
       tex->sampler_dim != GLSL_SAMPLER_DIM_MS)
      sources[count++] = (nir_tex_src) {
         .src_type = nir_tex_src_lod,
         .src = nir_src_for_ssa(nir_imm_int(b, 0)),
      };
   nir_tex_instr *query = nir_tex_instr_create(b->shader, count);
   query->op = op;
   query->sampler_dim = tex->sampler_dim;
   query->is_array = tex->is_array;
   query->texture_index = tex->texture_index;
   query->texture_non_uniform = tex->texture_non_uniform;
   query->dest_type = nir_type_uint32;
   for (unsigned i = 0; i < count; i++)
      query->src[i] = sources[i];
   nir_def_init(&query->instr, &query->def, components, 32);
   nir_builder_instr_insert(b, &query->instr);
   return &query->def;
}

static bool
lower_robust_fetch(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_txf && tex->op != nir_texop_txf_ms)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   int coord_src = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *coord = tex->src[coord_src].src.ssa;
   unsigned components = tex->coord_components;
   nir_def *size = texture_query(b, tex, nir_texop_txs, components);
   nir_def *valid = nir_imm_true(b);
   int lod_src = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (lod_src >= 0) {
      nir_def *lod = tex->src[lod_src].src.ssa;
      nir_def *levels = texture_query(b, tex, nir_texop_query_levels, 1);
      valid = nir_ult(b, lod, levels);
      nir_def *dims[4];
      for (unsigned i = 0; i < components; i++) {
         dims[i] = nir_channel(b, size, i);
         if (!tex->is_array || i + 1 < components)
            dims[i] = nir_umax_imm(b, nir_ushr(b, dims[i], nir_umin_imm(b, lod, 31)), 1);
      }
      size = nir_vec(b, dims, components);
      nir_src_rewrite(&tex->src[lod_src].src,
                      nir_bcsel(b, valid, lod, nir_imm_int(b, 0)));
   }
   int offset_src = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   if (offset_src >= 0) {
      nir_def *offset = tex->src[offset_src].src.ssa;
      if (tex->is_array)
         offset = nir_pad_vector_imm_int(b, offset, 0, components);
      coord = nir_iadd(b, coord, offset);
      nir_src_rewrite(&tex->src[offset_src].src,
                      nir_imm_zero(b, offset->num_components - tex->is_array, 32));
   }
   valid = nir_iand(b, valid, nir_ball(b, nir_ult(b, coord, size)));
   int sample_src = nir_tex_instr_src_index(tex, nir_tex_src_ms_index);
   if (sample_src >= 0) {
      nir_def *sample = tex->src[sample_src].src.ssa;
      nir_def *samples = texture_query(b, tex, nir_texop_texture_samples, 1);
      nir_def *sample_valid = nir_ult(b, sample, samples);
      valid = nir_iand(b, valid, sample_valid);
      nir_src_rewrite(&tex->src[sample_src].src,
                      nir_bcsel(b, sample_valid, sample, nir_imm_int(b, 0)));
   }
   nir_def *x = nir_bcsel(b, valid, nir_channel(b, coord, 0),
                          nir_imm_int(b, INT32_MAX));
   nir_src_rewrite(&tex->src[coord_src].src, nir_vector_insert_imm(b, coord, x, 0));
   return true;
}

bool
pan_nir_lower_robust_image_access2(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower_robust_fetch, nir_metadata_none, NULL);
}
