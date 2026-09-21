/*
 * Copyright © 2016 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "nir.h"
#include "nir_builder.h"

static nir_def *
load_layer_id(nir_builder *b, const nir_input_attachment_options *options)
{
   if (options->use_view_id_for_layer)
      return nir_load_view_index(b);
   else
      return nir_load_layer_id(b);
}

static nir_def *
load_coord(nir_builder *b, nir_tex_instr *tex, nir_def *offset,
           const nir_input_attachment_options *options)
{
   if (options->use_ia_coord_intrin) {
      if (tex->input_attachment_depth) {
         return nir_load_depth_input_attachment_coord(b);
      } else if (tex->input_attachment_stencil) {
         return nir_load_stencil_input_attachment_coord(b);
      } else {
         return nir_load_input_attachment_coord(b, offset, .base =
                                                tex->texture_index);
      }
   } else {
      nir_def *pos = nir_f2i32(b, nir_build_frag_coord(b, 2));
      nir_def *layer = load_layer_id(b, options);
      return nir_vec3(b, nir_channel(b, pos, 0), nir_channel(b, pos, 1), layer);
   }
}

static bool
try_lower_input_load(nir_builder *b, nir_intrinsic_instr *load,
                     bool is_deref,
                     const nir_input_attachment_options *options)
{
   nir_tex_src_type handle_src_type;
   enum glsl_sampler_dim image_dim;
   nir_alu_type dest_type;
   nir_def *handle;

   b->cursor = nir_after_instr(&load->instr);

   if (is_deref) {
      nir_deref_instr *deref = nir_src_as_deref(load->src[0]);
      assert(glsl_type_is_image(deref->type));

      image_dim = glsl_get_sampler_dim(deref->type);

      handle = &deref->def;
      handle_src_type = nir_tex_src_texture_deref;

      dest_type = nir_get_nir_type_for_glsl_base_type(
         glsl_get_sampler_result_type(deref->type));
   } else {
      image_dim = nir_intrinsic_image_dim(load);
      if (image_dim != GLSL_SAMPLER_DIM_SUBPASS &&
          image_dim != GLSL_SAMPLER_DIM_SUBPASS_MS)
         return false;

      handle = load->src[0].ssa;
      handle_src_type = nir_tex_src_texture_heap_offset;

      dest_type = nir_intrinsic_dest_type(load);
   }

   bool input_attachment_index =
      load->intrinsic == nir_intrinsic_image_deref_input_attachment_load ||
      load->intrinsic == nir_intrinsic_image_heap_input_attachment_load;

   const bool multisampled = (image_dim == GLSL_SAMPLER_DIM_SUBPASS_MS);
   /* A range of 0 means unknown. A range of 1 means the offset must be 0. */
   const bool arrayed = input_attachment_index &&
      (nir_intrinsic_range(load) == 0 || nir_intrinsic_range(load) > 1);

   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 3 + arrayed + multisampled);

   tex->op = nir_texop_txf;
   tex->sampler_dim = image_dim;

   tex->dest_type = dest_type;
   tex->is_array = true;
   tex->is_shadow = false;
   tex->is_sparse = false;

   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->can_speculate = true;

   unsigned src = 0;

   if (arrayed) {
      tex->src[src++] = nir_tex_src_for_ssa(nir_tex_src_texture_offset,
                                            load->src[3].ssa);
      tex->texture_array_size = nir_intrinsic_range(load);
      tex->texture_non_uniform = nir_intrinsic_access(load) & ACCESS_NON_UNIFORM;
   }

   tex->input_attachment_depth =
      load->intrinsic == nir_intrinsic_image_deref_depth_input_attachment_load ||
      load->intrinsic == nir_intrinsic_image_heap_depth_input_attachment_load;
   tex->input_attachment_stencil =
      load->intrinsic == nir_intrinsic_image_deref_stencil_input_attachment_load ||
      load->intrinsic == nir_intrinsic_image_heap_stencil_input_attachment_load;
   tex->input_attachment_index = input_attachment_index;
   if (input_attachment_index)
      tex->texture_index = nir_intrinsic_base(load);

   nir_def *offset = nir_vec3(b, nir_channel(b, load->src[1].ssa, 0),
                                 nir_channel(b, load->src[1].ssa, 1),
                                 nir_imm_int(b, 0));
   nir_def *coord = nir_iadd(b, load_coord(b, tex, load->src[3].ssa, options), offset);

   tex->src[src++] = nir_tex_src_for_ssa(handle_src_type, handle);
   tex->src[src++] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->coord_components = 3;

   tex->src[src++] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));

   if (image_dim == GLSL_SAMPLER_DIM_SUBPASS_MS) {
      tex->op = nir_texop_txf_ms;
      tex->src[src++] = nir_tex_src_for_ssa(nir_tex_src_ms_index,
                                            load->src[2].ssa);
   }

   tex->texture_non_uniform = nir_intrinsic_access(load) & ACCESS_NON_UNIFORM;

   nir_def_init(&tex->instr, &tex->def, nir_tex_instr_dest_size(tex), 32);
   nir_builder_instr_insert(b, &tex->instr);

   nir_def_replace(&load->def, &tex->def);
   return true;
}

static bool
try_lower_input_texop(nir_builder *b, nir_tex_instr *tex,
                      const nir_input_attachment_options *options)
{
   const int texture_src_idx =
      nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (texture_src_idx < 0)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(tex->src[texture_src_idx].src);

   if (glsl_get_sampler_dim(deref->type) != GLSL_SAMPLER_DIM_SUBPASS_MS)
      return false;

   const int coord_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   assert(coord_src_idx >= 0);

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *offset = tex->src[coord_src_idx].src.ssa;
   offset = nir_vec3(b, nir_channel(b, offset, 0),
                        nir_channel(b, offset, 1),
                        nir_imm_int(b, 0));

   int texture_offset_idx =
      nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
   nir_def *texture_offset =
      texture_offset_idx >= 0 ? tex->src[texture_offset_idx].src.ssa :
      nir_imm_int(b, 0);
   nir_def *coord = nir_iadd(b, load_coord(b, tex, texture_offset, options), offset);

   tex->coord_components = 3;

   nir_src_rewrite(&tex->src[coord_src_idx].src, coord);

   return true;
}

static bool
lower_input_attachments_instr(nir_builder *b, nir_instr *instr, void *_data)
{
   const nir_input_attachment_options *options = _data;

   switch (instr->type) {
   case nir_instr_type_tex: {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      if (tex->op == nir_texop_fragment_mask_fetch_amd ||
          tex->op == nir_texop_fragment_fetch_amd)
         return try_lower_input_texop(b, tex, options);

      return false;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);

      switch (load->intrinsic) {
      case nir_intrinsic_image_deref_input_attachment_load:
      case nir_intrinsic_image_deref_depth_input_attachment_load:
      case nir_intrinsic_image_deref_stencil_input_attachment_load:
         return try_lower_input_load(
            b, load, true /* is_deref */, options);
      case nir_intrinsic_image_heap_input_attachment_load:
      case nir_intrinsic_image_heap_depth_input_attachment_load:
      case nir_intrinsic_image_heap_stencil_input_attachment_load:
         return try_lower_input_load(
            b, load, false /* is_deref */, options);
      default:
         return false;
      }
   }

   default:
      return false;
   }
}

bool
nir_lower_input_attachments(nir_shader *shader,
                            const nir_input_attachment_options *options)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);

   return nir_shader_instructions_pass(shader, lower_input_attachments_instr,
                                       nir_metadata_control_flow,
                                       (void *)options);
}
