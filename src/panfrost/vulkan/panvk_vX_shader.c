/*
 * Copyright © 2025 Arm Ltd.
 * Copyright © 2021 Collabora Ltd.
 * Copyright © 2019-2026 Google LLC
 *
 * Also derived from anv_pipeline.c which is
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "genxml/gen_macros.h"

#ifdef PANVK_OFFLINE_ONLY
#include "tools/kraidoc_context.h"
#include "tools/kraidoc_report.h"
#else
#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_instance.h"
#include "panvk_mempool.h"
#include "panvk_physical_device.h"
#endif
#include "panvk_blend_state.h"
#include "panvk_descriptor_set_layout.h"
#include "panvk_image_formats.h"
#include "panvk_nir.h"
#if PAN_ARCH >= 15
#include "panvk_ray_tracing.h"
#include "panvk_rt_pipeline.h"
#endif
#include "panvk_sampler.h"
#include "panvk_shader.h"

#include "spirv/nir_spirv.h"
#include "util/cache_ops.h"
#include "util/memstream.h"
#include "util/mesa-blake3.h"
#include "util/shader_stats.h"
#include "util/u_cpu_detect.h"
#include "util/u_dynarray.h"
#include "util/u_hexdump.h"
#include "util/u_memory.h"
#include "nir_builder.h"
#include "nir_control_flow.h"
#include "nir_conversion_builder.h"
#include "nir_deref.h"
#include "nir_xfb_info.h"

#include "shader_enums.h"
#include "vk_graphics_state.h"
#include "vk_nir_convert_ycbcr.h"
#include "vk_shader_module.h"
#include "vk_ycbcr_conversion.h"

#include "compiler/bifrost/bifrost_nir.h"
#include "compiler/bifrost/bifrost_compile.h"
#include "compiler/pan_compiler.h"
#include "compiler/pan_nir.h"
#include "pan_shader.h"

#include "poly/nir/poly_nir.h"
#include "poly/geometry.h"

#include "vk_log.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"
#include "vk_shader.h"
#include "vk_util.h"

#define FAU_WORD_COUNT 64

struct panvk_lower_sysvals_context {
   struct panvk_shader_variant *shader;
   const struct vk_graphics_pipeline_state *state;
};

static bool
panvk_lower_sysvals(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const struct panvk_lower_sysvals_context *ctx = data;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   unsigned bit_size = intr->def.bit_size;
   nir_def *val = NULL;
   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_base_workgroup_id:
      val = load_sysval(b, compute, bit_size, base);
      break;
   case nir_intrinsic_load_num_workgroups:
      val = load_sysval(b, compute, bit_size, num_work_groups);
      break;
   case nir_intrinsic_load_workgroup_size:
      val = load_sysval(b, compute, bit_size, local_group_size);
      break;
   case nir_intrinsic_load_viewport_scale:
      val = load_sysval(b, graphics, bit_size, viewport.scale);
      break;
   case nir_intrinsic_load_viewport_offset:
      val = load_sysval(b, graphics, bit_size, viewport.offset);
      break;
   case nir_intrinsic_load_first_vertex:
      val = load_sysval(b, graphics, bit_size, vs.first_vertex);
      break;
   case nir_intrinsic_load_base_instance:
      val = load_sysval(b, graphics, bit_size, vs.base_instance);
      break;
   case nir_intrinsic_load_num_vertices:
      val = load_sysval(b, graphics, bit_size, xfb.num_vertices);
      break;
   case nir_intrinsic_load_xfb_address: {
      const unsigned idx = nir_intrinsic_base(intr);
      nir_def *base = load_sysval_entry(
         b, graphics, 64, xfb.base, nir_imm_int(b, idx));
      nir_def *offset_ptr = load_sysval_entry(
         b, graphics, 64, xfb.offset_ptr, nir_imm_int(b, idx));
      nir_def *offset = nir_load_global(b, 1, 32, offset_ptr, 4, 0);
      val = nir_iadd(b, base, nir_u2u64(b, offset));
      break;
   }
   case nir_intrinsic_load_noperspective_varyings_pan:
      /* TODO: use a VS epilog specialized on constant noperspective_varyings
       * with VK_EXT_graphics_pipeline_libraries and VK_EXT_shader_object */
      assert(b->shader->info.stage == MESA_SHADER_VERTEX);
      val = load_sysval(b, graphics, bit_size, vs.noperspective_varyings);
      break;
   case nir_intrinsic_load_clip_cull_count_pan:
      assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      val = load_sysval(b, graphics, bit_size, fs.clip_cull);
      break;

#if PAN_ARCH < 9
   case nir_intrinsic_load_raw_vertex_offset:
      val = load_sysval(b, graphics, bit_size, vs.raw_vertex_offset);
      break;
   case nir_intrinsic_load_layer_id:
      assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      val = load_sysval(b, graphics, bit_size, layer_id);
      break;
   case nir_intrinsic_load_view_index:
      assert(b->shader->info.stage != MESA_SHADER_COMPUTE);
      if (!ctx->state || ctx->state->mv->view_mask == 0)
         val = nir_imm_zero(b, 1, 32);
      else
         val = load_sysval(b, graphics, bit_size, layer_id);
      break;
#else
   case nir_intrinsic_load_view_index:
      if (!ctx->state || ctx->state->mv->view_mask == 0) {
         val = nir_imm_zero(b, 1, 32);
         break;
      } else if (b->shader->info.stage == MESA_SHADER_VERTEX) {
         /* On v14+ we have real multiview and view_index comes from a preload
          * in the vertex stage.  On earlier generations, view_index in vertex
          * shaders gets lowered away by nir_lower_multiview() so we should
          * never see it here.
          */
         assert(PAN_ARCH >= 14);
         return false;
      }

      /* For fragment shaders, it's the same as layer_id */
      assert(b->shader->info.stage == MESA_SHADER_FRAGMENT);
      FALLTHROUGH;

   case nir_intrinsic_load_layer_id:
      val = nir_load_frame_arg_pan(b);
      val = nir_extract_u8_imm(b, nir_u2u32(b, val), 0);
      break;
#endif

   case nir_intrinsic_load_draw_id:
      /* Multidraw is supported on v10. */
      if (PAN_ARCH >= 10)
         return false;

      /* TODO: We only implement single-draw direct and indirect draws, so this
       * is sufficient. We'll revisit this when we get around to implementing
       * multidraw. */
      assert(b->shader->info.stage == MESA_SHADER_VERTEX);
      val = nir_imm_int(b, 0);
      break;

   case nir_intrinsic_load_printf_buffer_address:
      val = load_sysval(b, common, bit_size, printf_buffer_address);
      break;

   case nir_intrinsic_load_blend_descriptor_pan: {
      uint32_t loc = nir_intrinsic_base(intr);
      val = load_sysval(b, graphics, bit_size, fs.blend_descs[loc]);
      break;
   }

   case nir_intrinsic_load_input_attachment_target_pan: {
      const struct vk_input_attachment_location_state *ial =
         ctx->state ? ctx->state->ial : NULL;

      if (ial && nir_src_is_const(intr->src[0])) {
         uint32_t index = nir_src_as_uint(intr->src[0]);
         uint32_t depth_idx = ial->depth_att == MESA_VK_ATTACHMENT_NO_INDEX
                                 ? 0
                                 : ial->depth_att + 1;
         uint32_t stencil_idx = ial->stencil_att == MESA_VK_ATTACHMENT_NO_INDEX
                                   ? 0
                                   : ial->stencil_att + 1;
         uint32_t target = ~0;

         if (depth_idx == index || stencil_idx == index) {
            target = PANVK_ZS_ATTACHMENT;
         } else {
            for (unsigned i = 0; i < ial->color_attachment_count; i++) {
               if (ial->color_map[i] == MESA_VK_ATTACHMENT_UNUSED)
                  continue;

               if (ial->color_map[i] + 1 == index) {
                  target = PANVK_COLOR_ATTACHMENT(i);
                  break;
               }
            }
         }

         val = nir_imm_int(b, target);
      } else {
         nir_def *ia_info =
            load_sysval_entry(b, graphics, bit_size, iam, intr->src[0].ssa);

         val = nir_channel(b, ia_info, 0);
      }
      break;
   }

   case nir_intrinsic_load_input_attachment_conv_pan: {
      nir_def *ia_info =
         load_sysval_entry(b, graphics, bit_size, iam, intr->src[0].ssa);

      val = nir_channel(b, ia_info, 1);
      break;
   }

   case nir_intrinsic_load_vertex_param_buffer_poly:
      /*
       * Software VS/TCS run through the compute path.  A future graphics
       * consumer may use the graphics copy of the same ABI.
       */
      if (b->shader->info.stage == MESA_SHADER_COMPUTE ||
          b->shader->info.stage == MESA_SHADER_KERNEL) {
         val = load_sysval(b, compute, bit_size,
                           poly.vertex_param_buffer);
      } else {
         assert(b->shader->info.stage == MESA_SHADER_VERTEX);
         val = load_sysval(b, graphics, bit_size,
                           poly.vertex_param_buffer);
      }
      break;

   case nir_intrinsic_load_tess_param_buffer_poly:
      /*
       * TCS is lowered to compute; TES is lowered by libpoly to a hardware
       * vertex shader.  Select the appropriate FAU block after that lowering.
       */
      if (b->shader->info.stage == MESA_SHADER_COMPUTE ||
          b->shader->info.stage == MESA_SHADER_KERNEL) {
         val = load_sysval(b, compute, bit_size,
                           poly.tess_param_buffer);
      } else {
         assert(b->shader->info.stage == MESA_SHADER_VERTEX);
         val = load_sysval(b, graphics, bit_size,
                           poly.tess_param_buffer);
      }
      break;

   case nir_intrinsic_load_geometry_param_buffer_poly:
      if (b->shader->info.stage == MESA_SHADER_COMPUTE ||
          b->shader->info.stage == MESA_SHADER_KERNEL) {
         val = load_sysval(b, compute, bit_size,
                           poly.geometry_param_buffer);
      } else {
         assert(b->shader->info.stage == MESA_SHADER_VERTEX);
         val = load_sysval(b, graphics, bit_size,
                           poly.geometry_param_buffer);
      }
      break;

   case nir_intrinsic_load_stat_query_address_poly:
      val = nir_imm_intN_t(b, 0, bit_size);
      break;

   case nir_intrinsic_load_provoking_last:
      val = nir_imm_intN_t(b, 0, bit_size);
      break;

   case nir_intrinsic_load_ro_sink_address_poly:
      val = nir_imm_int64(b, PAN_SHADER_OOB_ADDRESS);
      break;

   default:
      return false;
   }

   assert(val->num_components == intr->def.num_components);

   b->cursor = nir_after_instr(instr);
   nir_def_rewrite_uses(&intr->def, val);
   return true;
}

static bool
panvk_lower_sw_vs_vertex_ids_instr(nir_builder *b,
                                      nir_intrinsic_instr *intrin,
                                      void *data)
{
   if (intrin->intrinsic != nir_intrinsic_load_vertex_id_zero_base &&
       intrin->intrinsic != nir_intrinsic_load_raw_vertex_id)
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   nir_def *id =
      nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
   if (intrin->intrinsic == nir_intrinsic_load_raw_vertex_id)
      id = poly_nir_load_vertex_id(b, id);
   nir_def_rewrite_uses(&intrin->def, id);

   return true;
}

static bool
panvk_lower_sw_vs_vertex_ids(nir_shader *nir)
{
   bool progress =
      nir_shader_intrinsics_pass(nir,
                                 panvk_lower_sw_vs_vertex_ids_instr,
                                 nir_metadata_control_flow,
                                 NULL);

   if (progress)
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   return progress;
}

/*
 * CSF launches complete compute workgroups for the software VS.  For a
 * non-multiple-of-64 vertex count, the final workgroup therefore contains
 * physical invocations which do not correspond to Vulkan vertex invocations.
 *
 * Keep generic libpoly unchanged.  At the PanVK execution boundary, wrap the
 * complete original SW-VS body in a bounds check.  This prevents padded lanes
 * from executing output stores or any other shader side effect.
 *
 * Do not use nir_jump_return here: Panfrost's divergence analysis does not
 * support return instructions at this point in the compiler pipeline.
 */
static bool
panvk_lower_sw_vs_padded_invocations(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);

   /*
    * Save the complete original shader body.  Mesa uses the same NIR CF
    * extraction/reinsertion machinery when surrounding an existing shader
    * body with newly generated control flow.
    */
   nir_cf_list body;
   nir_cf_list_extract(&body, &impl->body);

   nir_builder b = nir_builder_at(nir_after_impl(impl));

   nir_def *global_x =
      nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

   nir_def *vp = nir_load_vertex_param_buffer_poly(&b);
   nir_def *vertex_count_addr =
      nir_iadd_imm(&b, vp,
                   offsetof(struct poly_vertex_params, verts_per_instance));

   nir_def *vertex_count =
      nir_load_global_constant(&b, 1, 32, vertex_count_addr,
                               .align_mul = 4);

   /*
    * Only real Vulkan vertex invocations enter the original shader body.
    * Y remains the Vulkan instance dimension.
    */
   nir_if *active =
      nir_push_if(&b, nir_ult(&b, global_x, vertex_count));

   nir_cursor body_cursor = b.cursor;

   nir_pop_if(&b, active);

   /* Insert the complete original VS into the true side of the condition. */
   nir_cf_reinsert(&body, body_cursor);

   return nir_progress(true, impl, nir_metadata_none);
}

static bool
collect_gs_vertex_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct pan_varying_layout *layout = data;

   if (intr->intrinsic != nir_intrinsic_store_per_vertex_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (BITFIELD64_BIT(sem.location) & PAN_HARDWARE_VARYING_BITS)
      return false;

   const unsigned ncomps = util_last_bit(nir_intrinsic_write_mask(intr));
   const nir_alu_type type = nir_intrinsic_src_type(intr);

   for (unsigned i = 0; i < layout->count; i++) {
      struct pan_varying_slot *slot = &layout->slots[i];
      if (slot->location == sem.location) {
         assert(slot->alu_type == type);
         slot->ncomps = MAX2(slot->ncomps, ncomps);
         return false;
      }
   }

   assert(layout->count < ARRAY_SIZE(layout->slots));
   layout->slots[layout->count++] = (struct pan_varying_slot){
      .location = sem.location,
      .alu_type = type,
      .ncomps = ncomps,
      .section = PAN_VARYING_SECTION_GENERIC,
      .offset = -1,
   };
   return false;
}

static int
cmp_varying_slot_location(const void *a, const void *b)
{
   const struct pan_varying_slot *sa = a, *sb = b;
   return (int)sa->location - (int)sb->location;
}

static void
panvk_gs_build_varying_layout(nir_shader *gs, struct pan_varying_layout *layout)
{
   memset(layout, 0, sizeof(*layout));
   nir_shader_intrinsics_pass(gs, collect_gs_vertex_output, nir_metadata_all,
                              layout);

   qsort(layout->slots, layout->count, sizeof(layout->slots[0]),
         cmp_varying_slot_location);

   unsigned size_B = 0;
   for (unsigned i = 0; i < layout->count; i++) {
      struct pan_varying_slot *slot = &layout->slots[i];
      const unsigned slot_size =
         slot->ncomps * (nir_alu_type_get_type_size(slot->alu_type) / 8);
      const unsigned offset = align(size_B, util_next_power_of_two(slot_size));

      slot->offset = offset;
      size_B = offset + slot_size;
   }

   layout->generic_size_B = size_B;
   layout->known = PAN_VARYING_FORMAT_KNOWN | PAN_VARYING_LAYOUT_KNOWN;
}

static nir_def *
load_gs_raster_param(nir_builder *b, unsigned offset, unsigned num_components,
                     unsigned bit_size)
{
   nir_def *addr = nir_iadd_imm(b, nir_load_geometry_param_buffer_poly(b),
                                PANVK_GS_RASTER_PARAMS_OFFSET + offset);

   return nir_load_global_constant(b, num_components, bit_size, addr,
                                   .align_mul = 4);
}

static bool
lower_gs_vertex_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct pan_varying_layout *layout = data;

   if (intr->intrinsic != nir_intrinsic_store_per_vertex_output)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);

   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_def *value = intr->src[0].ssa;
   nir_def *vertex = nir_u2u64(b, intr->src[1].ssa);

   if (sem.location == VARYING_SLOT_POS) {
      value = nir_pad_vector_imm_int(b, value, 0, 4);
      nir_def *base = load_gs_raster_param(
         b, offsetof(struct panvk_gs_raster_params, position_buffer), 1, 64);
      nir_def *scale = load_gs_raster_param(
         b, offsetof(struct panvk_gs_raster_params, viewport_scale), 3, 32);
      nir_def *offset = load_gs_raster_param(
         b, offsetof(struct panvk_gs_raster_params, viewport_offset), 3, 32);

      nir_def *w_recip = nir_frcp(b, nir_channel(b, value, 3));
      w_recip = nir_fclamp(b, w_recip, nir_imm_float(b, -32768.0f),
                           nir_imm_float(b, 32768.0f));

      nir_def *ndc = nir_fmul(b, nir_trim_vector(b, value, 3), w_recip);
      nir_def *screen = nir_fadd(b, nir_fmul(b, ndc, scale), offset);
      nir_def *pos =
         nir_vec4(b, nir_channel(b, screen, 0), nir_channel(b, screen, 1),
                  nir_channel(b, screen, 2), w_recip);

      nir_store_global(b, pos, nir_iadd(b, base, nir_imul_imm(b, vertex, 16)),
                       .align_mul = 16);
      return true;
   }

   const struct pan_varying_slot *slot =
      pan_varying_layout_find_slot(layout, sem.location);
   if (!slot)
      return true;

   nir_def *base = load_gs_raster_param(
      b, offsetof(struct panvk_gs_raster_params, varying_buffer), 1, 64);
   nir_def *addr =
      nir_iadd(b, base, nir_imul_imm(b, vertex, layout->generic_size_B));

   nir_store_global(b, nir_trim_vector(b, value, slot->ncomps),
                    nir_iadd_imm(b, addr, slot->offset), .align_mul = 4);
   return true;
}

static void
panvk_lower_gs_padded_invocations(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_list_extract(&body, &impl->body);

   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_def *prim = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *prims_addr =
      nir_iadd_imm(&b, nir_load_geometry_param_buffer_poly(&b),
                   offsetof(struct poly_geometry_params, grid));
   nir_def *prims =
      nir_load_global_constant(&b, 1, 32, prims_addr, .align_mul = 4);

   nir_if *active = nir_push_if(&b, nir_ult(&b, prim, prims));
   nir_cursor body_cursor = b.cursor;
   nir_pop_if(&b, active);

   nir_cf_reinsert(&body, body_cursor);
   nir_progress(true, impl, nir_metadata_none);
}

static nir_shader *
panvk_create_gs_rast_vs(const struct pan_varying_layout *varyings)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX,
      pan_get_nir_shader_compiler_options(PAN_ARCH, MESA_SHADER_VERTEX, false),
      "panvk_gs_rast");

   nir_def *base = load_gs_raster_param(
      &b, offsetof(struct panvk_gs_raster_params, position_buffer), 1, 64);
   nir_def *vertex = nir_u2u64(&b, nir_load_raw_vertex_id(&b));
   nir_def *pos =
      nir_load_global(&b, 4, 32, nir_iadd(&b, base, nir_imul_imm(&b, vertex, 16)),
                      .align_mul = 16);

   nir_store_output(&b, pos, nir_imm_int(&b, 0), .base = 0, .range = 1,
                    .write_mask = 0xf, .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS,
                    .io_semantics.num_slots = 1);

   uint64_t outputs_written = VARYING_BIT_POS;
   if (varyings->count) {
      nir_def *record = nir_iadd(
         &b,
         load_gs_raster_param(
            &b, offsetof(struct panvk_gs_raster_params, varying_buffer), 1, 64),
         nir_imul_imm(&b, vertex, varyings->generic_size_B));

      for (unsigned i = 0; i < varyings->count; i++) {
         const struct pan_varying_slot *slot = &varyings->slots[i];
         nir_def *value = nir_load_global(
            &b, slot->ncomps, nir_alu_type_get_type_size(slot->alu_type),
            nir_iadd_imm(&b, record, slot->offset), .align_mul = 4);

         nir_store_output(&b, value, nir_imm_int(&b, 0), .base = i + 1,
                          .range = 1,
                          .write_mask = nir_component_mask(slot->ncomps),
                          .src_type = slot->alu_type,
                          .io_semantics.location = slot->location,
                          .io_semantics.num_slots = 1);
         outputs_written |= BITFIELD64_BIT(slot->location);
      }
   }

   b.shader->info.outputs_written = outputs_written;
   return b.shader;
}

#if PAN_ARCH < 9
static bool
lower_gl_pos_layer_writes(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   if (intr->intrinsic != nir_intrinsic_copy_deref)
      return false;

   nir_variable *dst_var = nir_intrinsic_get_var(intr, 0);
   nir_variable *src_var = nir_intrinsic_get_var(intr, 1);

   if (!dst_var || dst_var->data.mode != nir_var_shader_out || !src_var ||
       src_var->data.mode != nir_var_shader_temp)
      return false;

   if (dst_var->data.location == VARYING_SLOT_LAYER) {
      /* We don't really write the layer, we just make sure primitives are
       * discarded if gl_Layer doesn't match the layer passed to the draw.
       */
      b->cursor = nir_instr_remove(instr);
      return true;
   }

   if (dst_var->data.location == VARYING_SLOT_POS) {
      nir_variable *temp_layer_var = data;
      nir_variable *temp_pos_var = src_var;

      b->cursor = nir_before_instr(instr);
      nir_def *layer = nir_load_var(b, temp_layer_var);
      nir_def *pos = nir_load_var(b, temp_pos_var);
      nir_def *inf_pos = nir_imm_vec4(b, INFINITY, INFINITY, INFINITY, 1.0f);
      nir_def *ref_layer = load_sysval(b, graphics, 32, layer_id);

      nir_store_var(b, temp_pos_var,
                    nir_bcsel(b, nir_ieq(b, layer, ref_layer), pos, inf_pos),
                    0xf);
      return true;
   }

   return false;
}

static bool
lower_layer_writes(nir_shader *nir)
{
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      return false;

   nir_variable *temp_layer_var = NULL;
   bool has_layer_var = false;

   nir_foreach_variable_with_modes(var, nir,
                                   nir_var_shader_out | nir_var_shader_temp) {
      if (var->data.mode == nir_var_shader_out &&
          var->data.location == VARYING_SLOT_LAYER)
         has_layer_var = true;

      if (var->data.mode == nir_var_shader_temp &&
          var->data.location == VARYING_SLOT_LAYER)
         temp_layer_var = var;
   }

   if (!has_layer_var)
      return false;

   assert(temp_layer_var);

   return nir_shader_instructions_pass(nir, lower_gl_pos_layer_writes,
                                       nir_metadata_control_flow,
                                       temp_layer_var);
}
#endif

#if PAN_ARCH >= 10
static bool
remove_viewport_write(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_deref &&
       intr->intrinsic != nir_intrinsic_copy_deref)
      return false;

   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.mode != nir_var_shader_out ||
       var->data.location != VARYING_SLOT_VIEWPORT)
      return false;

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
remove_viewport_writes(nir_shader *nir)
{
   if (nir->info.stage == MESA_SHADER_FRAGMENT ||
       !nir_find_variable_with_location(nir, nir_var_shader_out,
                                        VARYING_SLOT_VIEWPORT))
      return false;

   bool progress = nir_shader_intrinsics_pass(
      nir, remove_viewport_write, nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_out, NULL);
   nir->info.outputs_written &= ~VARYING_BIT_VIEWPORT;
   return progress;
}

static bool
mark_all_access_non_uniform(nir_builder *b, nir_instr *instr, void *data)
{
   switch (instr->type) {
   case nir_instr_type_tex: {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      for (unsigned i = 0; i < tex->num_srcs; i++) {
         switch (tex->src[i].src_type) {
         case nir_tex_src_texture_offset:
         case nir_tex_src_texture_handle:
            tex->texture_non_uniform = true;
            break;

         case nir_tex_src_sampler_offset:
         case nir_tex_src_sampler_handle:
            tex->sampler_non_uniform = true;
            break;

         default:
            break;
         }
      }

      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

      switch (intrin->intrinsic) {
      case nir_intrinsic_load_ubo:
      case nir_intrinsic_load_ssbo:
      case nir_intrinsic_store_ssbo:
      case nir_intrinsic_ssbo_atomic:
      case nir_intrinsic_ssbo_atomic_swap:
      case nir_intrinsic_image_load:
      case nir_intrinsic_image_sparse_load:
      case nir_intrinsic_image_store:
      case nir_intrinsic_image_atomic:
      case nir_intrinsic_image_atomic_swap:
      case nir_intrinsic_image_size:
      case nir_intrinsic_image_samples:
      case nir_intrinsic_image_deref_load:
      case nir_intrinsic_image_deref_sparse_load:
      case nir_intrinsic_image_deref_store:
      case nir_intrinsic_image_deref_atomic:
      case nir_intrinsic_image_deref_atomic_swap:
      case nir_intrinsic_image_deref_levels:
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_image_deref_samples:
      case nir_intrinsic_image_deref_samples_identical:
         nir_intrinsic_set_access(
            intrin, nir_intrinsic_access(intrin) | ACCESS_NON_UNIFORM);
         return true;
      default:
         return false;
      }
   }
   default:
      return false;
   }
}
#endif

static bool
lower_cull_distance_store(nir_builder *b, nir_intrinsic_instr *intr,
                          UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_CLIP_DIST0 &&
       sem.location != VARYING_SLOT_CLIP_DIST1)
      return false;

   nir_src *offset = nir_get_io_offset_src(intr);
   if (!nir_src_is_const(*offset))
      return false;

   const unsigned clip_count = b->shader->info.clip_distance_array_size;
   const unsigned total =
      clip_count + b->shader->info.cull_distance_array_size;
   const unsigned first =
      (sem.location - VARYING_SLOT_CLIP_DIST0 + nir_src_as_uint(*offset)) *
         4 +
      nir_intrinsic_component(intr);
   const unsigned write_mask = nir_intrinsic_write_mask(intr);
   nir_def *value = intr->src[0].ssa;

   bool any_cull = false;
   for (unsigned c = 0; c < value->num_components; c++) {
      if ((write_mask & BITFIELD_BIT(c)) && first + c >= clip_count &&
          first + c < total)
         any_cull = true;
   }

   if (!any_cull)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < value->num_components; c++) {
      comps[c] = nir_channel(b, value, c);
      if ((write_mask & BITFIELD_BIT(c)) && first + c >= clip_count &&
          first + c < total)
         comps[c] = nir_b2f32(b, nir_fge_imm(b, comps[c], 0.0));
   }

   nir_src_rewrite(&intr->src[0], nir_vec(b, comps, value->num_components));
   return true;
}

static bool
panvk_nir_lower_cull_distance_vs(nir_shader *nir)
{
   if (!nir->info.cull_distance_array_size)
      return false;

   NIR_PASS(_, nir, nir_opt_constant_folding);
   return nir_shader_intrinsics_pass(nir, lower_cull_distance_store,
                                     nir_metadata_control_flow, NULL);
}

static uint32_t
panvk_clip_cull_count(const nir_shader *nir)
{
   return MIN2(nir->info.clip_distance_array_size, 8) |
          (MIN2(nir->info.cull_distance_array_size, 8) << 4);
}

static nir_def *
lower_clip_cull_fs_slot(nir_builder *b, nir_variable *var, unsigned first,
                        nir_def *clip_n, nir_def *total)
{
   nir_deref_instr *deref = nir_build_deref_var(b, var);
   nir_def *kill = nir_imm_false(b);

   for (unsigned i = first; i < first + 4; i++) {
      nir_def *d = nir_load_deref(b, nir_build_deref_array_imm(b, deref, i));
      nir_def *idx = nir_imm_int(b, i);
      nir_def *is_clip = nir_ilt(b, idx, clip_n);
      nir_def *is_cull = nir_iand(b, nir_inot(b, is_clip), nir_ilt(b, idx, total));
      nir_def *flat = nir_iand(b, nir_feq_imm(b, nir_ddx(b, d), 0.0),
                               nir_feq_imm(b, nir_ddy(b, d), 0.0));
      nir_def *culled =
         nir_iand(b, is_cull, nir_iand(b, nir_feq_imm(b, d, 0.0), flat));
      nir_def *clipped = nir_iand(b, is_clip, nir_flt_imm(b, d, 0.0));
      kill = nir_ior(b, kill, nir_ior(b, clipped, culled));
   }

   return kill;
}

static bool
panvk_nir_lower_clip_cull_fs(nir_shader *nir, int producer_clip_cull)
{
   if (producer_clip_cull == 0)
      return false;

   nir_variable *var = NULL;
   nir_foreach_shader_in_variable(v, nir) {
      if (v->data.location == VARYING_SLOT_CLIP_DIST0 && v->data.compact &&
          v->data.location_frac == 0 && glsl_get_length(v->type) == 8) {
         var = v;
         break;
      }
   }

   if (!var) {
      var = nir_variable_create(
         nir, nir_var_shader_in,
         glsl_array_type(glsl_float_type(), 8, sizeof(float)), "clip_cull_pan");
      var->data.location = VARYING_SLOT_CLIP_DIST0;
      var->data.compact = true;
      var->data.interpolation = INTERP_MODE_SMOOTH;
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *counts = producer_clip_cull > 0
                        ? nir_imm_int(&b, producer_clip_cull)
                        : nir_load_clip_cull_count_pan(&b);
   nir_def *clip_n = nir_iand_imm(&b, counts, 0xf);
   nir_def *total = nir_iadd(&b, clip_n, nir_ushr_imm(&b, counts, 4));

   nir_push_if(&b, nir_ine_imm(&b, counts, 0));
   nir_def *kill_lo = lower_clip_cull_fs_slot(&b, var, 0, clip_n, total);
   nir_def *no_kill = nir_imm_false(&b);
   nir_push_if(&b, nir_ugt_imm(&b, total, 4));
   nir_def *kill_hi = lower_clip_cull_fs_slot(&b, var, 4, clip_n, total);
   nir_pop_if(&b, NULL);
   kill_hi = nir_if_phi(&b, kill_hi, no_kill);
   nir_demote_if(&b, nir_ior(&b, kill_lo, kill_hi));
   nir_pop_if(&b, NULL);

   nir->info.inputs_read |= VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1;
   nir->info.fs.uses_discard = true;
   return nir_progress(true, impl, nir_metadata_none);
}

static void
shared_type_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size =
      glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length, *align = comp_size * (length == 3 ? 4 : length);
}

static inline nir_address_format
panvk_buffer_ubo_addr_format(VkPipelineRobustnessBufferBehaviorEXT robustness)
{
   switch (robustness) {
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT:
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT:
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT:
      return PAN_ARCH < 9 ? nir_address_format_32bit_index_offset
                           : nir_address_format_vec2_index_32bit_offset;
   default:
      UNREACHABLE("Invalid robust buffer access behavior");
   }
}

static inline nir_address_format
panvk_buffer_ssbo_addr_format(VkPipelineRobustnessBufferBehaviorEXT robustness)
{
   switch (robustness) {
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT:
      return PAN_ARCH < 9 ? nir_address_format_64bit_global_32bit_offset
                           : nir_address_format_vec2_index_32bit_offset;
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_EXT:
   case VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT:
      return PAN_ARCH < 9 ? nir_address_format_64bit_bounded_global
                           : nir_address_format_vec2_index_32bit_offset;
   default:
      UNREACHABLE("Invalid robust buffer access behavior");
   }
}

static const nir_shader_compiler_options *
panvk_get_nir_options(UNUSED struct vk_physical_device *vk_pdev,
                      mesa_shader_stage stage,
                      UNUSED const struct vk_pipeline_robustness_state *rs)
{
   return pan_get_nir_shader_compiler_options(PAN_ARCH, stage, false);
}

static struct spirv_to_nir_options
panvk_get_spirv_options(UNUSED struct vk_physical_device *vk_pdev,
                        mesa_shader_stage stage,
                        const struct vk_pipeline_robustness_state *rs)
{
   return (struct spirv_to_nir_options){
      .mediump_16bit_alu = pan_use_kraid(PAN_ARCH, stage, false),
      .ubo_addr_format = panvk_buffer_ubo_addr_format(rs->uniform_buffers),
      .ssbo_addr_format = panvk_buffer_ssbo_addr_format(rs->storage_buffers),
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .shared_addr_format = nir_address_format_32bit_offset,
      .tensor_addr_format = nir_address_format_vec2_index_32bit_offset,
      .min_ubo_alignment = 16,
      .min_ssbo_alignment = 16,
      .debug_info = pan_want_debug_info(PAN_ARCH),
   };
}

static void
panvk_preprocess_nir(struct vk_physical_device *vk_pdev,
                     nir_shader *nir,
                     UNUSED const struct vk_pipeline_robustness_state *rs)
{
   struct panvk_physical_device *pdev = to_panvk_physical_device(vk_pdev);

   /* Ensure to regroup output variables at the same location */
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_opt_vectorize_io_vars, nir_var_shader_out);

   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
            nir_shader_get_entrypoint(nir), nir_var_shader_out);

#if PAN_ARCH < 9
   /* This needs to be done just after the io_to_temporaries pass, because we
    * rely on out temporaries to collect the final layer_id value.
    */
   NIR_PASS(_, nir, lower_layer_writes);
#endif

#if PAN_ARCH >= 10
   NIR_PASS(_, nir, remove_viewport_writes);
#endif

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);

   NIR_PASS(_, nir, nir_opt_copy_prop_vars);
   NIR_PASS(_, nir, nir_opt_combine_stores, nir_var_all);
   NIR_PASS(_, nir, nir_opt_loop);

   NIR_PASS(_, nir, nir_opt_barrier_modes);
   NIR_PASS(_, nir, nir_opt_acquire_release_barriers, SCOPE_DEVICE);

   /* Do texture lowering here. We need to lower texture stuff
    * now, before we call panvk_per_arch(nir_lower_descriptors)() because some
    * of the texture lowering generates nir_texop_txs which we handle as part
    * of descriptor lowering.
    *
    * TODO: We really should be doing this in common code, not duplicated in
    * panvk. In order to do that, we need to rework the panfrost compile
    * flow to look more like the Intel flow:
    *
    *  1. Compile SPIR-V to NIR and maybe do a tiny bit of lowering that needs
    *     to be done really early.
    *
    *  2. pan_preprocess_nir: Does common lowering and runs the optimization
    *     loop.  Nothing here should be API-specific.
    *
    *  3. Do additional lowering in panvk
    *
    *  4. pan_postprocess_nir: Does final lowering and runs the optimization
    *     loop again.  This can happen as part of the final compile.
    *
    * This would give us a better place to do panvk-specific lowering.
    */
   NIR_PASS(_, nir, nir_lower_system_values);

   nir_lower_compute_system_values_options options = {
      .has_base_workgroup_id = true,
      .shuffle_local_ids_for_quad_derivatives = true,
   };

   NIR_PASS(_, nir, nir_lower_compute_system_values, &options);

   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_lower_wpos_center);

   assert(pdev->kmod.dev->props.shader_present != 0);
   uint64_t core_max_id =
      util_bitcount64(pdev->kmod.dev->props.shader_present) - 1;
   NIR_PASS(_, nir, nir_inline_sysval, nir_intrinsic_load_core_max_id_arm,
            core_max_id);

   pan_preprocess_nir(nir, pdev->kmod.dev->props.gpu_id);
}

static bool
fs_alpha_to_coverage_disabled(const struct vk_graphics_pipeline_state *state)
{
   return state != NULL && state->ms != NULL &&
          !BITSET_TEST(state->dynamic,
                       MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE) &&
          !state->ms->alpha_to_coverage_enable;
}

#if PAN_ARCH >= 14
static bool
fs_may_use_vrs(const struct vk_features *features,
               const struct vk_graphics_pipeline_state *state)
{
   if (!features->pipelineFragmentShadingRate &&
       !features->primitiveFragmentShadingRate &&
       !features->attachmentFragmentShadingRate)
      return false;

   if (state == NULL || BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_FSR))
      return true;

   return state->fsr && !vk_fragment_shading_rate_is_disabled(state->fsr);
}

static bool
lower_vrs_frag_center(nir_builder *b, nir_intrinsic_instr *intr,
                      UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_sample_pos_or_center)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *rate = nir_load_frag_shading_rate(b);
   BITSET_SET(b->shader->info.system_values_read,
              SYSTEM_VALUE_FRAG_SHADING_RATE);
   nir_def *log2_size =
      nir_vec2(b, nir_iand_imm(b, nir_ushr_imm(b, rate, 2), 3),
               nir_iand_imm(b, rate, 3));
   nir_def *center = nir_ldexp(b, nir_imm_vec2(b, 0.5f, 0.5f), log2_size);
   if (center->bit_size != intr->def.bit_size)
      center = nir_f2fN(b, center, intr->def.bit_size);

   nir_def_replace(&intr->def, center);
   return true;
}
#endif

static void
panvk_hash_state(struct vk_physical_device *device,
                 const struct vk_graphics_pipeline_state *state,
                 const struct vk_features *enabled_features,
                 VkShaderStageFlags stages, blake3_hash blake3_out)
{
   struct mesa_blake3 blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);

   if (state != NULL) {
      /* This doesn't impact the shader compile but it does go in the
       * panvk_shader and gets [de]serialized along with the binary so
       * we need to hash it.
       */
      bool sample_shading_enable =
         state->ms && state->ms->sample_shading_enable;
      _mesa_blake3_update(&blake3_ctx, &sample_shading_enable,
                          sizeof(sample_shading_enable));

      _mesa_blake3_update(&blake3_ctx, &state->mv->view_mask,
                          sizeof(state->mv->view_mask));

      if (state->ial)
         _mesa_blake3_update(&blake3_ctx, state->ial, sizeof(*state->ial));

      if (state->ial && state->cal)
         _mesa_blake3_update(&blake3_ctx, state->cal, sizeof(*state->cal));

      if (stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
         struct panvk_blend_static_key blend_key;
         panvk_per_arch(blend_static_key_init)(&blend_key, state);
         _mesa_blake3_update(&blake3_ctx, &blend_key, sizeof(blend_key));
         bool a2c_off = fs_alpha_to_coverage_disabled(state);
         _mesa_blake3_update(&blake3_ctx, &a2c_off, sizeof(a2c_off));
#if PAN_ARCH >= 14
         bool vrs = fs_may_use_vrs(enabled_features, state);
         _mesa_blake3_update(&blake3_ctx, &vrs, sizeof(vrs));
#endif
      }
   }

   _mesa_blake3_final(&blake3_ctx, blake3_out);
}

#if PAN_ARCH >= 9
static bool
valhall_pack_buf_idx(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   unsigned index_src;

   switch (intrin->intrinsic) {
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      index_src = 0;
      break;

   case nir_intrinsic_store_ssbo:
      index_src = 1;
      break;

   default:
      return false;
   }

   nir_def *index = intrin->src[index_src].ssa;

   /* The descriptor lowering pass can add UBO loads, and those already have the
    * right index format. */
   if (index->num_components == 1)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);

   /* The valhall backend expects nir_address_format_32bit_index_offset,
    * but address mode is nir_address_format_vec2_index_32bit_offset to allow
    * us to store the array size, set and index without losing information
    * while walking the descriptor deref chain (needed to do a bound check on
    * the array index when we reach the end of the chain).
    * Turn it back to nir_address_format_32bit_index_offset after IOs
    * have been lowered. */
   nir_def *packed_index =
      nir_iadd(b, nir_channel(b, index, 0), nir_channel(b, index, 1));
   nir_src_rewrite(&intrin->src[index_src], packed_index);
   return true;
}
#endif

static bool
is_robust_ssbo_intr(const nir_intrinsic_instr *intr, UNUSED const void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      return true;
   default:
      return false;
   }
}

static bool
valhall_lower_get_ssbo_size(struct nir_builder *b,
                            nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_get_ssbo_size)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *res_handle = nir_channel(b, intr->src[0].ssa, 0);
   nir_def *table_idx = nir_ushr_imm(b, res_handle, 24);
   nir_def *res_idx = nir_iand_imm(b, res_handle, BITFIELD_MASK(24));
   nir_def *res_table = nir_ior_imm(b, table_idx, pan_res_handle(62, 0));
   nir_def *buf_idx = nir_iadd(b, res_idx, nir_channel(b, intr->src[0].ssa, 1));
   nir_def *desc_offset = nir_imul_imm(b, buf_idx, PANVK_DESCRIPTOR_SIZE);
   nir_def *size = nir_load_ubo(
      b, 1, 32, res_table, nir_iadd_imm(b, desc_offset, 4), .range = ~0u,
      .align_mul = PANVK_DESCRIPTOR_SIZE, .align_offset = 4);

   nir_def_replace(&intr->def, size);
   return true;
}

static bool
collect_push_constant(struct nir_builder *b, nir_intrinsic_instr *intr,
                      void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_push_constant)
      return false;

   struct panvk_shader_variant *shader = data;
   uint32_t base = nir_intrinsic_base(intr);
   bool is_sysval = base >= SYSVALS_PUSH_CONST_BASE;
   uint32_t offset, size;

   if (is_sysval)
      base -= SYSVALS_PUSH_CONST_BASE;

   /* If the offset is dynamic, we need to flag [base:base+range] as used, to
    * allow global mem access. */
   if (!nir_src_is_const(intr->src[0])) {
      offset = base;
      size = nir_intrinsic_range(intr);

      /* Flag the push_uniforms sysval as needed if we have an indirect offset.
       */
      shader_use_sysval(shader, common, push_uniforms);
   } else {
      offset = base + nir_src_as_uint(intr->src[0]);
      size = (intr->def.bit_size / 8) * intr->def.num_components;
   }

   if (is_sysval)
      shader_use_sysval_range(shader, offset, size);
   else
      shader_use_push_const_range(shader, offset, size);

   return true;
}

static bool
move_push_constant(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_push_constant)
      return false;

   struct panvk_shader_variant *shader = data;
   unsigned base = nir_intrinsic_base(intr);
   bool is_sysval = base >= SYSVALS_PUSH_CONST_BASE;

   if (is_sysval)
      base -= SYSVALS_PUSH_CONST_BASE;

   b->cursor = nir_before_instr(&intr->instr);

   if (nir_src_is_const(intr->src[0])) {
      unsigned offset = base + nir_src_as_uint(intr->src[0]);

      /* We place the sysvals first, and then comes the user push constants.
       * We do that so we always have the blend constants at offset 0 for
       * blend shaders. */
      if (is_sysval)
         offset = shader_remapped_sysval_offset(shader, offset);
      else
         offset = shader_remapped_push_const_offset(shader, offset);

      nir_src_rewrite(&intr->src[0], nir_imm_int(b, offset));

      /* We always set the range/base to zero, to make sure no pass is using it
       * after that point. */
      nir_intrinsic_set_base(intr, 0);
      nir_intrinsic_set_range(intr, 0);
   } else {
      /* We don't use load_sysval() on purpose, because it would set
       * .base=SYSVALS_PUSH_CONST_BASE, and we're supposed to force a base of
       * zero in this pass. */
      unsigned push_const_buf_offset = shader_remapped_sysval_offset(
         shader, sysval_offset(common, push_uniforms));
      nir_def *push_const_buf = nir_load_push_constant(
         b, 1, 64, nir_imm_int(b, push_const_buf_offset));
      unsigned push_const_offset = is_sysval ?
         shader_remapped_sysval_offset(shader, base) :
         shader_remapped_push_const_offset(shader, base);
      nir_def *offset = nir_iadd_imm(b, intr->src[0].ssa, push_const_offset);
      unsigned align = nir_combined_align(nir_intrinsic_align_mul(intr),
                                          nir_intrinsic_align_offset(intr));

      /* We assume an alignment of 64-bit max for packed push-constants. */
      align = MIN2(align, FAU_WORD_SIZE);
      nir_def *value = nir_load_global(
         b, intr->def.num_components, intr->def.bit_size,
         nir_iadd(b, push_const_buf, nir_u2u64(b, offset)), .align_mul = align);

      nir_def_replace(&intr->def, value);
   }

   return true;
}

static void
lower_load_push_consts(nir_shader *nir, struct panvk_shader_variant *shader)
{
   /* Before we lower load_push_constant()s with a dynamic offset to global
    * loads, we want to run a few optimization passes to get rid of offset
    * calculation involving only constant values. */
   bool progress = false;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 64,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select, &peephole_select_options);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
   } while (progress);

   /* We always reserve the 4 blend constant words for fragment shaders,
    * because we don't know the blend configuration at this point, and
    * we might end up with a blend shader reading those blend constants. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* We rely on blend constants being placed first and covering 4 words. */
      STATIC_ASSERT(
         offsetof(struct panvk_graphics_sysvals, blend.constants) == 0 &&
         sizeof(((struct panvk_graphics_sysvals *)NULL)->blend.constants) ==
            16);

      shader_use_sysval(shader, graphics, blend.constants);
   }

   progress = false;
   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, collect_push_constant,
            nir_metadata_all, shader);

   /* Some load_push_constant instructions might be eliminated after
    * scalarization+dead-code-elimination. Since these pass happen in
    * bifrost_compile(), we can't run the push_constant packing after the
    * optimization took place, so let's just have our own FAU count instead
    * of using info.fau.end to make it consistent with the
    * used_{sysvals,push_consts} bitmaps, even if it sometimes implies loading
    * more than we really need. Doing that also takes into account the fact
    * blend constants are never loaded from the fragment shader, but might be
    * needed in the blend shader. */
   shader->fau.sysval_count = BITSET_COUNT(shader->fau.used_sysvals);
   /* 32 FAUs (256 bytes) are reserved for API push constants */
   assert(shader->fau.sysval_count <= FAU_WORD_COUNT - 32 &&
          "too many sysval FAUs");
   shader->fau.total_count =
      shader->fau.sysval_count + BITSET_COUNT(shader->fau.used_push_consts);
   assert(shader->fau.total_count <= FAU_WORD_COUNT &&
          "asking for more FAUs than the hardware has to offer");

   if (!progress)
      return;

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, move_push_constant,
            nir_metadata_control_flow, shader);
}

struct lower_ycbcr_state {
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *const *set_layouts;
};

static const struct vk_ycbcr_conversion_state *
lookup_ycbcr_conversion(const void *_state, uint32_t set,
                        uint32_t binding, uint32_t array_index)
{
   const struct lower_ycbcr_state *state = _state;
   assert(set < state->set_layout_count);
   assert(state->set_layouts[set] != NULL);
   const struct panvk_descriptor_set_layout *set_layout =
      to_panvk_descriptor_set_layout(state->set_layouts[set]);
   assert(binding < set_layout->binding_count);

   const struct panvk_descriptor_set_binding_layout *bind_layout =
      &set_layout->bindings[binding];

   if (bind_layout->immutable_samplers == NULL)
      return NULL;

   array_index = MIN2(array_index, bind_layout->desc_count - 1);

   const struct panvk_sampler *sampler =
      bind_layout->immutable_samplers[array_index];

   if (!sampler || !sampler->vk.ycbcr_conversion)
      return NULL;

   const struct vk_ycbcr_conversion_state *conversion =
      &sampler->vk.ycbcr_conversion->state;
   if (panvk_image_use_yuv_tex(PAN_ARCH, conversion->format))
      return NULL;

   return conversion;
}

static unsigned
glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static void
panvk_lower_nir(struct panvk_device *dev, nir_shader *nir,
                uint32_t set_layout_count,
                struct vk_descriptor_set_layout *const *set_layouts,
                const struct vk_pipeline_robustness_state *rs,
                const struct vk_graphics_pipeline_state *state,
                struct panvk_shader_desc_info *desc_info,
                bool allow_merging_workgroups)
{
   mesa_shader_stage stage = nir->info.stage;

   NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);

   /* Run before descriptor and explicit-IO lowering so the memory derefs this
    * pass emits get lowered by them.
    */
   NIR_PASS(_, nir, panvk_nir_lower_cooperative_matrix,
            pan_subgroup_size(PAN_ARCH));
   NIR_PASS(_, nir, pan_nir_lower_bf16);

   const nir_opt_access_options access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &access_options);

   const struct lower_ycbcr_state ycbcr_state = {
      .set_layout_count = set_layout_count,
      .set_layouts = set_layouts,
   };
   NIR_PASS(_, nir, nir_vk_lower_ycbcr_tex, lookup_ycbcr_conversion,
            &ycbcr_state);

   /* We need to do this before nir_lower_descriptors so any image_deref_size
    * intrinsics generated can be lowered there.
    */
   if (PAN_ARCH < 9)
      NIR_PASS(_, nir, pan_nir_lower_image_ms);

   if (PAN_ARCH >= 15 && rs->images ==
       VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2_EXT)
      NIR_PASS(_, nir, pan_nir_lower_robust_image_access2);

   panvk_per_arch(nir_lower_descriptors)(nir, dev, rs, set_layout_count,
                                         set_layouts, state, desc_info);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_memcpy);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            panvk_buffer_ubo_addr_format(rs->uniform_buffers));
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            panvk_buffer_ssbo_addr_format(rs->storage_buffers));
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);

   /* nir_lower_ssbo lowers SSBO writes to unbounded store_global, so the
    * descriptor-level bounds check Mali HW does for native buffer
    * loads/stores is bypassed. Insert software bounds checks here for SSBO
    * accesses when robust storage buffer access is requested. */
   if (rs->storage_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT) {
      NIR_PASS(_, nir, nir_lower_robust_access, is_robust_ssbo_intr, NULL);
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_dce);
   }

#if PAN_ARCH >= 10
   if (allow_merging_workgroups) {
      /* Accesses that were uniform in the source shader may now be
       * nonuniform. To handle this, we just flag everything as nonuniform and
       * then let nir_opt_non_uniform_access figure out which ones can really
       * be nonuniform based on divergence analysis */
      NIR_PASS(_, nir, nir_shader_instructions_pass,
               mark_all_access_non_uniform, nir_metadata_all, NULL);
   }
#endif

   /* nir_lower_non_uniform_access needs to run after lowering UBO and SSBO
    * IO. This means we run it after nir_lower_descriptors, which reads the
    * array indices, but it's okay because lower_descriptors treats all
    * dynamic indices the same. */
   enum nir_lower_non_uniform_access_type lower_non_uniform_access_types =
      nir_lower_non_uniform_ubo_access |
      nir_lower_non_uniform_ssbo_access |
      nir_lower_non_uniform_texture_access |
      nir_lower_non_uniform_texture_query |
      nir_lower_non_uniform_image_access |
      nir_lower_non_uniform_image_query |
      nir_lower_non_uniform_get_ssbo_size;
#if PAN_ARCH < 9
   lower_non_uniform_access_types |=
      nir_lower_non_uniform_texture_offset_access;
#endif

   /* In practice, most shaders do not have non-uniform-qualified accesses
    * thus a cheaper and likely to fail check is run first. */
   if (allow_merging_workgroups ||
       nir_has_non_uniform_access(nir, lower_non_uniform_access_types)) {
      NIR_PASS(_, nir, nir_opt_cse);
      NIR_PASS(_, nir, nir_opt_non_uniform_access);
      struct nir_lower_non_uniform_access_options opts = {
         .types = lower_non_uniform_access_types,
      };
      NIR_PASS(_, nir, nir_lower_non_uniform_access, &opts);
   }

#if PAN_ARCH >= 9
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, valhall_lower_get_ssbo_size,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_shader_instructions_pass, valhall_pack_buf_idx,
            nir_metadata_control_flow, NULL);
#endif

   if (mesa_shader_stage_uses_workgroup(stage)) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
               shared_type_info);

      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
               nir_address_format_32bit_offset);
   }

   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      /* Align everything up to 16 bytes to take advantage of load store
       * vectorization. */
      nir->info.shared_size = align(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size,
               16);

      /* We need to call lower_compute_system_values again because
       * nir_zero_initialize_shared_memory generates load_invocation_id which
       * has to be lowered to load_invocation_index.
       */
      NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   }

   /* Needed to turn shader_temp into function_temp since the backend only
    * handles the latter for now.
    */
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   if (PANVK_DEBUG(NIR)) {
      mesa_logi("translated nir:");
      nir_log_shaderi(nir);
   }
}

static void
panvk_lower_nir_io(nir_shader *nir)
{
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            glsl_type_size, nir_lower_io_use_interpolated_input_intrinsics);

   /* nir_lower_io just computes offsets based on the original deref and
    * lower_indirect_derefs ensures that the array derefs have a constant
    * index.  Constant-fold to get us actual constants in in load/store
    * instructions.
    */
   NIR_PASS(_, nir, nir_opt_constant_folding);

   pan_nir_lower_mediump_io(nir);
}

/*
 * Lower tessellation shader IO to NIR IO intrinsics without forcing
 * indirect per-vertex accesses through if/else trees.
 *
 * libpoly consumes load/store_per_vertex_{input,output} directly.
 */
static void
panvk_lower_tess_nir_io(nir_shader *nir)
{
   NIR_PASS(_, nir, nir_lower_var_copies);

   NIR_PASS(_, nir, nir_lower_io,
            nir_var_shader_in | nir_var_shader_out,
            glsl_type_size,
            nir_lower_io_use_interpolated_input_intrinsics);

   /*
    * Fold array/location arithmetic so poly sees the simplest possible
    * IO addressing while preserving dynamic per-vertex indexing.
    */
   NIR_PASS(_, nir, nir_opt_constant_folding);
}

static bool
panvk_pilots_supported(const nir_shader *nir,
                       const struct pan_compile_inputs *input,
                       VkShaderCreateFlagsEXT shader_flags)
{
   const mesa_shader_stage stage = nir->info.stage;

   return (PAN_ARCH == 10 || PAN_ARCH >= 15) && !input->disable_preamble &&
          !(shader_flags & VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT) &&
          !nir->info.internal &&
          (stage == MESA_SHADER_FRAGMENT || stage == MESA_SHADER_COMPUTE ||
           (stage == MESA_SHADER_VERTEX && !input->no_idvs)) &&
          pan_use_kraid(PAN_ARCH, stage, false) &&
          pan_use_kraid(PAN_ARCH, MESA_SHADER_COMPUTE, true);
}

static void
panvk_mark_pilot_volatile(struct pan_compile_inputs *input,
                          const struct panvk_shader_variant *shader,
                          unsigned offset, unsigned size)
{
   unsigned remapped = shader_remapped_sysval_offset(shader, offset);

   BITSET_SET_RANGE(input->fau.pilot_volatile, remapped / 4,
                    (remapped + size - 1) / 4);
}

static void
panvk_mark_pilot_volatile_sysvals(mesa_shader_stage stage,
                                  const struct panvk_shader_variant *shader,
                                  struct pan_compile_inputs *input)
{
   if (stage == MESA_SHADER_VERTEX) {
      if (shader_uses_sysval(shader, graphics, vs.first_vertex)) {
         panvk_mark_pilot_volatile(input, shader,
                                   sysval_offset(graphics, vs.first_vertex),
                                   sysval_size(graphics, vs.first_vertex));
      }
      if (shader_uses_sysval(shader, graphics, vs.base_instance)) {
         panvk_mark_pilot_volatile(input, shader,
                                   sysval_offset(graphics, vs.base_instance),
                                   sysval_size(graphics, vs.base_instance));
      }
   } else if (stage == MESA_SHADER_COMPUTE) {
      if (shader_uses_sysval(shader, compute, num_work_groups.x)) {
         panvk_mark_pilot_volatile(input, shader,
                                   sysval_offset(compute, num_work_groups.x),
                                   sysval_size(compute, num_work_groups.x));
      }
      if (shader_uses_sysval(shader, compute, num_work_groups.y)) {
         panvk_mark_pilot_volatile(input, shader,
                                   sysval_offset(compute, num_work_groups.y),
                                   sysval_size(compute, num_work_groups.y));
      }
      if (shader_uses_sysval(shader, compute, num_work_groups.z)) {
         panvk_mark_pilot_volatile(input, shader,
                                   sysval_offset(compute, num_work_groups.z),
                                   sysval_size(compute, num_work_groups.z));
      }
   }
}

static VkResult
panvk_compile_nir(struct panvk_device *dev, nir_shader *nir,
                  VkShaderCreateFlagsEXT shader_flags,
                  const struct pan_compile_inputs *compile_input,
                  const struct vk_graphics_pipeline_state *state,
                  const uint32_t *noperspective_varyings,
                  struct panvk_shader_desc_info *desc_info,
                  struct panvk_shader_variant *shader)
{
   const bool dump_asm =
      shader_flags & VK_SHADER_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_MESA;

   /* We're going to modify this so make our own copy to be nicer to callers */
   struct pan_compile_inputs input = *compile_input;

#if PAN_ARCH >= 15
   uint32_t ray_query_slots = panvk_per_arch(nir_lower_ray_queries)(nir);
#ifdef PANVK_OFFLINE_ONLY
   (void)ray_query_slots;
#else
   if (ray_query_slots) {
      VkResult result =
         panvk_per_arch(device_reserve_ray_query_slots)(dev, ray_query_slots);
      if (result != VK_SUCCESS)
         return result;
   }
#endif
#endif

   pan_postprocess_nir(nir, &input, &shader->info);

   if (noperspective_varyings && nir->info.stage == MESA_SHADER_VERTEX) {
      NIR_PASS(_, nir, nir_inline_sysval,
               nir_intrinsic_load_noperspective_varyings_pan,
               *noperspective_varyings);
   }

   struct panvk_lower_sysvals_context lower_sysvals_ctx = {
      .shader = shader,
      .state = state,
   };

   NIR_PASS(_, nir, nir_shader_instructions_pass, panvk_lower_sysvals,
            nir_metadata_control_flow, &lower_sysvals_ctx);

   input.instrument =
      (shader_flags & VK_SHADER_CREATE_INSTRUMENT_SHADER_BIT_ARM) &&
      (nir->info.stage == MESA_SHADER_VERTEX ||
       nir->info.stage == MESA_SHADER_FRAGMENT ||
       nir->info.stage == MESA_SHADER_COMPUTE);
   if (input.instrument)
      shader_use_sysval(shader, common, instr_counters);

   lower_load_push_consts(nir, shader);

   if (input.instrument)
      input.instrument_fau = shader_remapped_sysval_offset(
         shader, common_sysval_offset(instr_counters));

   /* Reserve sysvals/push-const, the compiler may fill the remaining space with
    * promoted constants. */
   input.fau.reserved = shader->fau.total_count * 2;
   input.fau.promote_immediates = true;

   struct pan_compile_preamble preamble = {
      .binary = UTIL_DYNARRAY_INIT,
   };
   if (panvk_pilots_supported(nir, &input, shader_flags)) {
      static int ubo_push = -1;
      if (ubo_push < 0)
         ubo_push = !debug_get_bool_option("PANVK_NO_UBO_PUSH", false);

      input.preamble = &preamble;
      if (ubo_push) {
         input.fau.pushable_ubos = ~0u;
         input.fau.push_ubo_handles = true;
      }
      panvk_mark_pilot_volatile_sysvals(nir->info.stage, shader, &input);
   }
   struct util_dynarray binary = UTIL_DYNARRAY_INIT;
   pan_shader_compile(nir, &input, &binary, &shader->info);

   if (preamble.binary.size) {
      shader->preamble = calloc(1, sizeof(*shader->preamble));
      if (!shader->preamble) {
         util_dynarray_fini(&preamble.binary);
         util_dynarray_fini(&binary);
         return panvk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      struct panvk_shader_variant *pilot = shader->preamble;
      pilot->info = preamble.info;
      pilot->cs.local_size = (struct pan_compute_dim){1, 1, 1};
      pilot->fau.total_count = 1;
      pilot->bin_ptr = mem_dup(preamble.binary.data, preamble.binary.size);
      pilot->bin_size = preamble.binary.size;
      pilot->own_bin = true;
      if (!pilot->bin_ptr) {
         util_dynarray_fini(&preamble.binary);
         util_dynarray_fini(&binary);
         return panvk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      if (dump_asm) {
         char *text = NULL;
         size_t size = 0;
         struct u_memstream mem;
         if (u_memstream_open(&mem, &text, &size)) {
            pan_disassemble(u_memstream_get(&mem), pilot->bin_ptr,
                            pilot->bin_size, input.gpu_id, false);
            u_memstream_close(&mem);
            pilot->asm_str = text;
         }
      }
   }
   util_dynarray_fini(&preamble.binary);

   /* Propagate potential additional FAU values into the panvk info struct. */
   /* FAU consts are pushed as 32bit values, but total_count is for 64bit
    * ones. */
   shader->fau.total_count = DIV_ROUND_UP(shader->info.fau.count, 2);

   void *bin_ptr = util_dynarray_element(&binary, uint8_t, 0);
   unsigned bin_size = util_dynarray_num_elements(&binary, uint8_t);

   shader->bin_size = 0;
   shader->bin_ptr = NULL;

   if (bin_size) {
      void *data = malloc(bin_size);

      if (data == NULL)
         return panvk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

      memcpy(data, bin_ptr, bin_size);
      shader->bin_size = bin_size;
      shader->bin_ptr = data;
   }
   util_dynarray_fini(&binary);

#if PAN_ARCH < 9
   if (nir->constant_data_size) {
      shader->data_ptr = mem_dup(nir->constant_data, nir->constant_data_size);

      if (shader->data_ptr == NULL)
         return panvk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

      shader->data_size = nir->constant_data_size;
   }
#endif

   if (dump_asm) {
      shader->nir_str = nir_shader_as_str(nir, NULL);

      char *data = NULL;
      size_t disasm_size = 0;

      if (shader->bin_size) {
         struct u_memstream mem;
         if (u_memstream_open(&mem, &data, &disasm_size)) {
            FILE *const stream = u_memstream_get(&mem);
            pan_disassemble(stream, shader->bin_ptr, shader->bin_size,
                            compile_input->gpu_id, false);
            if (nir->constant_data_size) {
               fprintf(stream, "constant data (%u bytes):\n",
                       nir->constant_data_size);
               u_hexdump_words(stream, nir->constant_data, nir->constant_data_size);
            }
            u_memstream_close(&mem);
         }
      }

      char *asm_str = malloc(disasm_size + 1);
      memcpy(asm_str, data, disasm_size);
      asm_str[disasm_size] = '\0';
      free(data);

      shader->asm_str = asm_str;
   }

   /* Pad the total to the 64-bit-aligned FAU count; it's used to initialize the
    * RSD in pan_shader_prepare_rsd().
    */
   shader->info.fau.count = shader->fau.total_count * 2;

#if PAN_ARCH < 9
   /* Patch the descriptor count */
   shader->info.ubo_count =
      desc_info->others.count[PANVK_BIFROST_DESC_TABLE_UBO] +
      desc_info->dyn_ubos.count;
   shader->info.texture_count =
      desc_info->others.count[PANVK_BIFROST_DESC_TABLE_TEXTURE];
   shader->info.sampler_count =
      desc_info->others.count[PANVK_BIFROST_DESC_TABLE_SAMPLER];

   /* Dummy sampler. */
   if (!shader->info.sampler_count && shader->info.texture_count)
      shader->info.sampler_count++;

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      /* We leave holes in the attribute locations, but pan_shader.c assumes the
       * opposite. Patch attribute_count accordingly, so
       * pan_shader_prepare_rsd() does what we expect.
       */
      uint32_t gen_attribs =
         (shader->info.attributes_read & VERT_BIT_GENERIC_ALL) >>
         VERT_ATTRIB_GENERIC0;

      shader->info.attribute_count = util_last_bit(gen_attribs);

      /* NULL IDVS shaders are not allowed. */
      if (!bin_size)
         shader->info.vs.idvs = false;
   }

   /* Image attributes start at MAX_VS_ATTRIBS in the VS attribute table,
    * and zero in other stages.
    */
   if (desc_info->others.count[PANVK_BIFROST_DESC_TABLE_IMG] > 0)
      shader->info.attribute_count =
         desc_info->others.count[PANVK_BIFROST_DESC_TABLE_IMG] +
         (nir->info.stage == MESA_SHADER_VERTEX ? MAX_VS_ATTRIBS : 0);
#endif

   switch (nir->info.stage) {
   case MESA_SHADER_COMPUTE:
   case MESA_SHADER_KERNEL:
      shader->cs.local_size.x = nir->info.workgroup_size[0];
      shader->cs.local_size.y = nir->info.workgroup_size[1];
      shader->cs.local_size.z = nir->info.workgroup_size[2];
      break;

   case MESA_SHADER_FRAGMENT:
      shader->fs.earlyzs_lut = pan_earlyzs_analyze(&shader->info, PAN_ARCH);
      break;

   default:
      break;
   }

   return VK_SUCCESS;
}

#if PAN_ARCH >= 9
static enum mali_flush_to_zero_mode
shader_ftz_mode(struct panvk_shader_variant *shader)
{
   if (shader->info.ftz_fp32) {
      if (shader->info.ftz_fp16)
         return MALI_FLUSH_TO_ZERO_MODE_ALWAYS;
      else
         return MALI_FLUSH_TO_ZERO_MODE_DX11;
   } else {
      /* We don't have a "flush FP16, preserve FP32" mode, but APIs
       * should not be able to generate that.
       */
      assert(!shader->info.ftz_fp16 && !shader->info.ftz_fp32);
      return MALI_FLUSH_TO_ZERO_MODE_PRESERVE_SUBNORMALS;
   }
}
#endif

#ifndef PANVK_OFFLINE_ONLY
static VkResult
panvk_shader_upload(struct panvk_device *dev,
                    struct panvk_shader_variant *shader,
                    const VkAllocationCallbacks *pAllocator)
{
   if (shader->preamble) {
      VkResult result = panvk_shader_upload(dev, shader->preamble, pAllocator);
      if (result != VK_SUCCESS)
         return result;
   }
   shader->code_mem = (struct panvk_priv_mem){0};

#if PAN_ARCH < 9
   shader->data_mem = (struct panvk_priv_mem){0};
   shader->rsd = (struct panvk_priv_mem){0};
#else
   shader->spd = (struct panvk_priv_mem){0};
#endif

   if (!shader->bin_size)
      return VK_SUCCESS;

   shader->code_mem = panvk_pool_upload_aligned(
      &dev->mempools.exec, shader->bin_ptr, shader->bin_size, 128);
   if (!panvk_priv_mem_check_alloc(shader->code_mem))
      return panvk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

#if PAN_ARCH >= 9
   /* The inline constant pool is addressed as PC plus a 32-bit offset */
   ASSERTED uint64_t code_dev_addr = panvk_priv_mem_dev_addr(shader->code_mem);
   assert(code_dev_addr >> 32 == (code_dev_addr + shader->bin_size - 1) >> 32);
#endif

#if PAN_ARCH < 9
   if (shader->data_size) {
      shader->data_mem = panvk_pool_upload_aligned(
         &dev->mempools.rw, shader->data_ptr, shader->data_size, 64);
      if (!panvk_priv_mem_check_alloc(shader->data_mem))
         return panvk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   if (shader->info.stage == MESA_SHADER_FRAGMENT)
      return VK_SUCCESS;

   shader->rsd = panvk_pool_alloc_desc(&dev->mempools.rw, RENDERER_STATE);
   if (!panvk_priv_mem_check_alloc(shader->rsd))
      return panvk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   panvk_priv_mem_write_desc(shader->rsd, 0, RENDERER_STATE, cfg) {
      pan_shader_prepare_rsd(&shader->info,
                             panvk_shader_variant_get_dev_addr(shader), &cfg);
   }
#else
   if (shader->info.stage != MESA_SHADER_VERTEX) {
      shader->spd = panvk_pool_alloc_desc(&dev->mempools.rw, SHADER_PROGRAM);
      if (!panvk_priv_mem_check_alloc(shader->spd))
         return panvk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

      panvk_priv_mem_write_desc(shader->spd, 0, SHADER_PROGRAM, cfg) {
         cfg.stage = pan_shader_stage(&shader->info);

         if (cfg.stage == MALI_SHADER_STAGE_FRAGMENT)
            cfg.fragment_coverage_bitmask_type = MALI_COVERAGE_BITMASK_TYPE_GL;
#if PAN_ARCH < 12
         else if (cfg.stage == MALI_SHADER_STAGE_VERTEX)
            cfg.vertex_warp_limit = MALI_WARP_LIMIT_HALF;
#endif

#if PAN_ARCH >= 15
         cfg.register_count = shader->info.work_reg_count;
         cfg.preload.r0_r15 = shader->info.preload;
#else
         cfg.register_allocation =
            pan_register_allocation(shader->info.work_reg_count);
         cfg.preload.r48_r63 = (shader->info.preload >> 48);
#endif
         cfg.binary = panvk_shader_variant_get_dev_addr(shader);
         cfg.flush_to_zero_mode = shader_ftz_mode(shader);

         if (cfg.stage == MALI_SHADER_STAGE_FRAGMENT)
            cfg.requires_helper_threads = shader->info.contains_barrier;
      }
   } else {
#if PAN_ARCH >= 12
      shader->spds.all_points =
         panvk_pool_alloc_desc(&dev->mempools.rw, SHADER_PROGRAM);
      if (!panvk_priv_mem_check_alloc(shader->spds.all_points))
         return panvk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

      panvk_priv_mem_write_desc(shader->spds.all_points, 0, SHADER_PROGRAM,
                                cfg) {
         cfg.stage = pan_shader_stage(&shader->info);
#if PAN_ARCH >= 15
         cfg.register_count = shader->info.work_reg_count;
         cfg.preload.r0_r15 = shader->info.preload;
#else
         cfg.register_allocation =
            pan_register_allocation(shader->info.work_reg_count);
         cfg.preload.r48_r63 = (shader->info.preload >> 48);
#endif
         cfg.binary = panvk_shader_variant_get_dev_addr(shader);
         cfg.flush_to_zero_mode = shader_ftz_mode(shader);
      }

      shader->spds.all_triangles =
         panvk_pool_alloc_desc(&dev->mempools.rw, SHADER_PROGRAM);
      if (!panvk_priv_mem_check_alloc(shader->spds.all_triangles))
         return panvk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

      panvk_priv_mem_write_desc(shader->spds.all_triangles, 0, SHADER_PROGRAM,
                                cfg) {
         cfg.stage = pan_shader_stage(&shader->info);
#if PAN_ARCH >= 15
         cfg.register_count = shader->info.work_reg_count;
         cfg.preload.r0_r15 = shader->info.preload;
#else
         cfg.register_allocation =
            pan_register_allocation(shader->info.work_reg_count);
         cfg.preload.r48_r63 = (shader->info.preload >> 48);
#endif
         cfg.binary = panvk_shader_variant_get_dev_addr(shader) +
                      shader->info.vs.no_psiz_offset;
         cfg.flush_to_zero_mode = shader_ftz_mode(shader);
      }
#else
      shader->spds.pos_points =
         panvk_pool_alloc_desc(&dev->mempools.rw, SHADER_PROGRAM);
      if (!panvk_priv_mem_check_alloc(shader->spds.pos_points))
         return panvk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

      panvk_priv_mem_write_desc(shader->spds.pos_points, 0, SHADER_PROGRAM,
                                cfg) {
         cfg.stage = pan_shader_stage(&shader->info);
         cfg.vertex_warp_limit = MALI_WARP_LIMIT_HALF;
         cfg.register_allocation =
            pan_register_allocation(shader->info.work_reg_count);
         cfg.binary = panvk_shader_variant_get_dev_addr(shader);
         cfg.preload.r48_r63 = (shader->info.preload >> 48);
         cfg.flush_to_zero_mode = shader_ftz_mode(shader);
      }

      shader->spds.pos_triangles =
         panvk_pool_alloc_desc(&dev->mempools.rw, SHADER_PROGRAM);
      if (!panvk_priv_mem_check_alloc(shader->spds.pos_triangles))
         return panvk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

      panvk_priv_mem_write_desc(shader->spds.pos_triangles, 0, SHADER_PROGRAM,
                                cfg) {
         cfg.stage = pan_shader_stage(&shader->info);
         cfg.vertex_warp_limit = MALI_WARP_LIMIT_HALF;
         cfg.register_allocation =
            pan_register_allocation(shader->info.work_reg_count);
         cfg.binary = panvk_shader_variant_get_dev_addr(shader) +
                      shader->info.vs.no_psiz_offset;
         cfg.preload.r48_r63 = (shader->info.preload >> 48);
         cfg.flush_to_zero_mode = shader_ftz_mode(shader);
      }

      if (shader->info.vs.secondary_enable) {
         shader->spds.var =
            panvk_pool_alloc_desc(&dev->mempools.rw, SHADER_PROGRAM);
         if (!panvk_priv_mem_check_alloc(shader->spds.var))
            return panvk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

         panvk_priv_mem_write_desc(shader->spds.var, 0, SHADER_PROGRAM, cfg) {
            unsigned work_count = shader->info.vs.secondary_work_reg_count;

            cfg.stage = pan_shader_stage(&shader->info);
            cfg.vertex_warp_limit = MALI_WARP_LIMIT_FULL;
            cfg.register_allocation = pan_register_allocation(work_count);
            cfg.binary = panvk_shader_variant_get_dev_addr(shader) +
                         shader->info.vs.secondary_offset;
            cfg.preload.r48_r63 = (shader->info.vs.secondary_preload >> 48);
            cfg.flush_to_zero_mode = shader_ftz_mode(shader);
         }
      }
#endif
   }
#endif

   return VK_SUCCESS;
}

#endif

static void
panvk_shader_variant_destroy(struct panvk_shader_variant *shader)
{
   if (shader->preamble) {
      panvk_shader_variant_destroy(shader->preamble);
      free(shader->preamble);
   }
   free((void *)shader->asm_str);
   ralloc_free((void *)shader->nir_str);

#ifndef PANVK_OFFLINE_ONLY
   panvk_pool_free_mem(&shader->code_mem);

#if PAN_ARCH < 9
   panvk_pool_free_mem(&shader->data_mem);
   panvk_pool_free_mem(&shader->rsd);
#else
   if (shader->info.stage != MESA_SHADER_VERTEX) {
      panvk_pool_free_mem(&shader->spd);
   } else {
#if PAN_ARCH >= 12
      panvk_pool_free_mem(&shader->spds.all_points);
      panvk_pool_free_mem(&shader->spds.all_triangles);
#else
      panvk_pool_free_mem(&shader->spds.var);
      panvk_pool_free_mem(&shader->spds.pos_points);
      panvk_pool_free_mem(&shader->spds.pos_triangles);
#endif
   }
#endif

#endif

#if PAN_ARCH < 9
   free((void *)shader->data_ptr);
#endif

   if (shader->own_bin)
      free((void *)shader->bin_ptr);
}

static void
panvk_shader_fallback_destroy(struct panvk_device *dev,
                              struct panvk_shader_fallback *job);

static void
panvk_shader_destroy(struct vk_device *vk_dev, struct vk_shader *vk_shader,
                     const VkAllocationCallbacks *pAllocator)
{
   struct panvk_shader *shader =
      container_of(vk_shader, struct panvk_shader, vk);

   if (shader->bg_no_preamble)
      panvk_shader_fallback_destroy(
         container_of(vk_dev, struct panvk_device, vk),
         shader->bg_no_preamble);

   if (shader->no_preamble)
      panvk_shader_destroy(vk_dev, &shader->no_preamble->vk, pAllocator);

   if (shader->tess_vs)
      panvk_shader_destroy(vk_dev, &shader->tess_vs->vk, pAllocator);

   panvk_shader_foreach_variant(shader, variant) {
      panvk_shader_variant_destroy(variant);
   }

#if PAN_ARCH < 9
   panvk_pool_free_mem(&shader->desc_info.others.map);
#endif

   vk_shader_free(vk_dev, pAllocator, &shader->vk);
}

static const struct vk_shader_ops panvk_shader_ops;

static VkResult
panvk_compile_shader_impl(struct panvk_device *dev,
                     struct vk_shader_compile_info *info,
                     const struct vk_graphics_pipeline_state *state,
                     const struct pan_varying_layout *vs_varying_layout,
                     int vs_clip_cull,
                     const uint32_t *noperspective_varyings,
                     const VkAllocationCallbacks *pAllocator,
                     struct vk_shader **shader_out, bool allow_preamble,
                     bool upload)
{
   struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);

   struct panvk_shader *shader;
   VkResult result;

   size_t size =
      sizeof(struct panvk_shader) + sizeof(struct panvk_shader_variant) *
                                       panvk_shader_num_variants(info->stage);
   shader = vk_shader_zalloc(&dev->vk, &panvk_shader_ops, info->stage,
                             pAllocator, size);
   if (shader == NULL)
      return panvk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   nir_variable_mode robust_modes = 0;
   if (info->robustness->uniform_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT)
      robust_modes |= nir_var_mem_ubo;
   if (info->robustness->storage_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT)
      robust_modes |= nir_var_mem_ssbo;

   struct pan_compile_inputs inputs = {
      .gpu_id = phys_dev->kmod.dev->props.gpu_id,
      .gpu_variant = phys_dev->kmod.dev->props.gpu_variant,
      .view_mask = (state && state->rp) ? state->mv->view_mask : 0,
      .robust_modes = robust_modes,
      .robust_descriptors = dev->vk.enabled_features.nullDescriptor,
      .image_access_in_bounds = info->robustness->images ==
                                VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .disable_preamble = !allow_preamble,
   };

   switch (info->stage) {
   case MESA_SHADER_VERTEX: {
#ifndef PANVK_OFFLINE_ONLY
      /*
       * Software VS feeding libpoly TCS.
       *
       * Vertex attributes must be lowered while this is still a real
       * VERTEX shader.  After that, libpoly converts the VS outputs to
       * memory stores and the shader itself is executed as compute.
       */
      const bool sw_tess_vs =
         info->next_stage_mask &
         (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
          VK_SHADER_STAGE_GEOMETRY_BIT);

      if (sw_tess_vs) {
         struct panvk_shader_variant *variant =
            &shader->variants[PANVK_VS_VARIANT_HW];

         nir_shader *nir = info->nir;

         /*
          * Use the normal graphics descriptor/Vulkan lowering while the
          * original stage is still visible as VERTEX.
          */
         inputs.no_idvs = false;

         panvk_lower_nir(dev, nir,
                         info->set_layout_count,
                         info->set_layouts,
                         info->robustness,
                         state,
                         &shader->desc_info,
                         false);

         /*
          * Preserve Vulkan vertex attribute numbering exactly as in the
          * ordinary hardware VS path.
          */
         nir_foreach_shader_in_variable(var, nir) {
            assert(var->data.location >= VERT_ATTRIB_GENERIC0 &&
                   var->data.location <= VERT_ATTRIB_GENERIC15);

            var->data.driver_location =
               var->data.location - VERT_ATTRIB_GENERIC0;
         }

         /*
          * Turn VS input/output variables into IO intrinsics before libpoly.
          */
         nir_assign_io_var_locations(nir, nir_var_shader_out);
         panvk_lower_nir_io(nir);

         NIR_PASS(_, nir, pan_nir_lower_vs_inputs, inputs.gpu_id);

         /*
          * Capture exactly the output mask libpoly is about to consume.
          *
          * panvk_lower_nir() gathers shader info, and the IO lowering above
          * may further transform the shader. Re-gather here so the CPU-side
          * poly_vertex_params allocation and poly_nir_lower_vs_before_gs()
          * agree on the same outputs_written mask.
          */
         nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
         shader->tess.vs_outputs = nir->info.outputs_written;

         /*
          * Write VS outputs into libpoly's intermediate vertex buffer.
          */
         NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);

         /*
          * A software VS has no hardware zero-based vertex ID.  Its raw
          * invocation index is global_invocation_id.x; poly_nir_lower_sw_vs()
          * subsequently reconstructs the Vulkan vertex ID/indexed draw.
          */
         NIR_PASS(_, nir, panvk_lower_sw_vs_vertex_ids);

         /*
          * The physical execution stage is now compute.
          */
         nir->info.stage = MESA_SHADER_COMPUTE;
         memset(&nir->info.cs, 0, sizeof(nir->info.cs));

         /*
          * Match libpoly users: software VS is dispatched in groups of 64.
          * Unlike TCS, VS has no per-patch synchronization requirement.
          */
         nir->info.workgroup_size[0] = 64;
         nir->info.workgroup_size[1] = 1;
         nir->info.workgroup_size[2] = 1;
         nir->info.workgroup_size_variable = false;

         nir->xfb_info = NULL;

         /*
          * Reinterpret vertex/instance IDs using the libpoly draw parameter
          * buffer and indexed-draw input assembly.
          */
         NIR_PASS(_, nir, poly_nir_lower_sw_vs);

         /*
          * CSF rounds SW-VS execution up to complete workgroups.
          * Predicate the original shader body so padded lanes are inert.
          */
         NIR_PASS(_, nir, panvk_lower_sw_vs_padded_invocations);

         NIR_PASS(_, nir, poly_nir_lower_sysvals);

         nir->options =
            pan_get_nir_shader_compiler_options(
               PAN_ARCH, MESA_SHADER_COMPUTE, false);

         nir_shader_gather_info(
            nir, nir_shader_get_entrypoint(nir));

         /*
          * Keep the first implementation conservative.  Workgroup merging
          * can be enabled later after the complete tessellation path works.
          */
         variant->info.cs.allow_merging_workgroups = false;

         NIR_PASS(_, nir, nir_opt_constant_folding);

         variant->own_bin = true;

         result = panvk_compile_nir(dev, nir,
                                    info->flags,
                                    &inputs,
                                    state,
                                    noperspective_varyings,
                                    &shader->desc_info,
                                    variant);

         if (result != VK_SUCCESS) {
            panvk_shader_destroy(&dev->vk, &shader->vk, pAllocator);
            return result;
         }

         break;
      }

#endif
      const enum panvk_vs_variant compile_order[] = {
         PANVK_VS_VARIANT_XFB,
         PANVK_VS_VARIANT_HW,
      };

      for (unsigned pass = 0; pass < ARRAY_SIZE(compile_order); pass++) {
         const enum panvk_vs_variant v = compile_order[pass];
         struct panvk_shader_variant *variant = &shader->variants[v];

         if (v == PANVK_VS_VARIANT_XFB &&
             (PAN_ARCH < 10 || info->nir->xfb_info == NULL))
            continue;

         /* The hardware variant consumes the original NIR. */
         const bool clone_nir = v != PANVK_VS_VARIANT_HW;
         nir_shader *nir =
            clone_nir ? nir_shader_clone(NULL, info->nir) : info->nir;

         if (v == PANVK_VS_VARIANT_XFB) {
            for (uint32_t i = 0; i < MAX_XFB_BUFFERS; i++)
               variant->xfb_stride[i] = nir->xfb_info->buffers[i].stride;
            inputs.no_idvs = true;
         } else {
            inputs.no_idvs = false;
         }

         panvk_lower_nir(dev, nir, info->set_layout_count,
                         info->set_layouts, info->robustness,
                         state, &shader->desc_info, false);

#if PAN_ARCH >= 10 && PAN_ARCH < 14
         if (inputs.view_mask) {
            nir_lower_multiview_options options = {
               .view_mask = inputs.view_mask,
               .allowed_per_view_outputs = ~0
            };

            /* The only case where this should fail is with memory/image
             * writes, which we don't support in vertex shaders
             */
            assert(nir_can_lower_multiview(nir, options));
            NIR_PASS(_, nir, nir_lower_multiview, options);

            /* Pull output writes out of the loop and give them constant
             * offsets for pan_lower_store_components
             */
            NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
                     nir_shader_get_entrypoint(nir), nir_var_shader_out);
            NIR_PASS(_, nir, nir_lower_global_vars_to_local);
            NIR_PASS(_, nir, nir_split_var_copies);
         }
#endif

         /* We need the driver_location to match the vertex attribute
          * location, so we can use the attribute layout described by
          * vk_vertex_input_state where there are holes in the attribute
          * locations.
          */
         nir_foreach_shader_in_variable(var, nir) {
            assert(var->data.location >= VERT_ATTRIB_GENERIC0 &&
                   var->data.location <= VERT_ATTRIB_GENERIC15);
            var->data.driver_location =
               var->data.location - VERT_ATTRIB_GENERIC0;
         }
         nir_assign_io_var_locations(nir, nir_var_shader_out);
         panvk_lower_nir_io(nir);

         const uint32_t clip_cull = panvk_clip_cull_count(nir);
         if (v == PANVK_VS_VARIANT_HW)
            NIR_PASS(_, nir, panvk_nir_lower_cull_distance_vs);

         if (v == PANVK_VS_VARIANT_XFB)
            NIR_PASS(_, nir, nir_io_add_intrinsic_xfb_info);

         struct pan_varying_layout varying_layout;
         if (v == PANVK_VS_VARIANT_HW) {
            pan_varying_collect_formats(&varying_layout, nir, inputs.gpu_id);
            pan_build_varying_layout_compact(&varying_layout, nir,
                                             inputs.gpu_id);
            inputs.varying_layout = &varying_layout;
         }

         if (v == PANVK_VS_VARIANT_XFB) {
            static const nir_lower_xfb_to_stores_options xfb_options = {
               .address_format = nir_address_format_64bit_global,
            };
            NIR_PASS(_, nir, nir_lower_xfb_to_stores, &xfb_options);

            /* nir_lower_xfb_to_stores() replaces varying stores with global memory
             * writes.  Refresh shader info so the backend does not treat this
             * no-IDVS variant as an empty vertex shader.
             */
            nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
         }

         /* This somehow folds the location for multi-slot nir_load/nir_store */
         NIR_PASS(_, nir, nir_opt_constant_folding);

         variant->own_bin = true;

         result = panvk_compile_nir(dev, nir, info->flags, &inputs, state,
                                    noperspective_varyings,
                                    &shader->desc_info, variant);

         /* If we cloned, it's our job to clean up */
         if (clone_nir)
            ralloc_free(nir);

         if (result != VK_SUCCESS) {
            panvk_shader_destroy(&dev->vk, &shader->vk, pAllocator);
            return result;
         }

         if (v == PANVK_VS_VARIANT_HW) {
            variant->info.vs.clip_distance_count = clip_cull & 0xf;
            variant->info.vs.cull_distance_count = clip_cull >> 4;
         }
      }
      break;
   }

#ifndef PANVK_OFFLINE_ONLY
   case MESA_SHADER_TESS_CTRL: {
      struct panvk_shader_variant *variant =
         (struct panvk_shader_variant *)panvk_shader_only_variant(shader);

      nir_shader *nir = info->nir;

      /*
       * libpoly expects regular NIR IO intrinsics, including the
       * per-vertex TCS forms.  Do this while the shader is still
       * MESA_SHADER_TESS_CTRL.
       */
      nir_assign_io_var_locations(nir, nir_var_shader_in);
      nir_assign_io_var_locations(nir, nir_var_shader_out);
      panvk_lower_tess_nir_io(nir);

      /*
       * Refresh outputs_written/patch_outputs_written after lowering the
       * Vulkan TCS variables to the IO intrinsics consumed by libpoly.
       *
       * Capture the ABI only after this gather so the CPU allocation and
       * poly_nir_lower_tcs() agree on exactly the same output layout.
       */
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

      shader->tess.mode = nir->info.tess._primitive_mode;
      shader->tess.spacing = nir->info.tess.spacing;
      shader->tess.points = nir->info.tess.point_mode;
      shader->tess.ccw = nir->info.tess.ccw;
      shader->tess.tcs_output_patch_size =
         nir->info.tess.tcs_vertices_out;
      shader->tess.tcs_per_vertex_outputs =
         poly_tcs_per_vertex_outputs(nir);
      shader->tess.tcs_nr_patch_outputs =
         util_last_bit(nir->info.patch_outputs_written);
      shader->tess.tcs_output_stride =
         poly_tcs_output_stride(nir);

      /*
       * Save this before destroying the tessellation stage information.
       * One compute workgroup represents one TCS output patch.
       */
      const uint32_t output_patch_size =
         shader->tess.tcs_output_patch_size;

      assert(output_patch_size > 0);

      /*
       * Mali-G720 reports subgroupSize=16, while Vulkan tessellation must
       * be able to handle patches larger than a single such subgroup.
       *
       * Preserve shader-output barriers as global-memory WORKGROUP
       * barriers so all invocations belonging to the patch synchronize.
       */
      NIR_PASS(_, nir, poly_nir_lower_tcs_with_output_scope,
               false, SCOPE_WORKGROUP);

      /*
       * Resolve libpoly pseudo-sysvals such as index_size/vs_outputs into
       * accesses through the poly parameter buffers where applicable.
       */
      NIR_PASS(_, nir, poly_nir_lower_sysvals);

      /*
       * From this point onward the hardware sees a compute shader.
       *
       * TCS invocation_id maps to local_invocation_id.x and one patch maps
       * to one workgroup, therefore X exactly matches OutputVertices.
       */
      nir->info.stage = MESA_SHADER_COMPUTE;
      memset(&nir->info.cs, 0, sizeof(nir->info.cs));

      nir->info.workgroup_size[0] = output_patch_size;
      nir->info.workgroup_size[1] = 1;
      nir->info.workgroup_size[2] = 1;
      nir->info.workgroup_size_variable = false;

      nir->xfb_info = NULL;

      /*
       * All subsequent Panfrost passes/backend compilation must use
       * compute-stage compiler assumptions.
       */
      nir->options =
         pan_get_nir_shader_compiler_options(
            PAN_ARCH, MESA_SHADER_COMPUTE, false);

      /*
       * Refresh info after changing stage and replacing TCS IO by global
       * memory operations / compute system values.
       */
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

      /*
       * Never merge software-TCS workgroups.  One Vulkan patch is one
       * synchronization domain.
       */
      variant->info.cs.allow_merging_workgroups = false;

      /*
       * Descriptor lowering is intentionally done only now, after the
       * shader became COMPUTE.  PanVK's descriptor implementation has no
       * native TESS_CTRL descriptor stage.
       */
      panvk_lower_nir(dev, nir,
                      info->set_layout_count,
                      info->set_layouts,
                      info->robustness,
                      state,
                      &shader->desc_info,
                      false);

      variant->own_bin = true;

      result = panvk_compile_nir(dev, nir,
                                 info->flags,
                                 &inputs,
                                 state,
                                 noperspective_varyings,
                                 &shader->desc_info,
                                 variant);

      if (result != VK_SUCCESS) {
         panvk_shader_destroy(&dev->vk, &shader->vk, pAllocator);
         return result;
      }

      break;
   }

   case MESA_SHADER_GEOMETRY: {
      nir_shader *nir = info->nir;
      result = VK_SUCCESS;

      nir_assign_io_var_locations(nir, nir_var_shader_in);
      nir_assign_io_var_locations(nir, nir_var_shader_out);
      panvk_lower_tess_nir_io(nir);
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

      nir_shader *count = NULL, *pre_gs = NULL;
      struct poly_gs_info gs_info;
      NIR_PASS(_, nir, poly_nir_lower_gs_vertex_output, &count, &pre_gs,
               &gs_info);
      ralloc_free(pre_gs);

      const unsigned streams = util_last_bit(nir->info.gs.active_stream_mask);
      const unsigned vertices = gs_info.static_vertices >= 0
                                   ? gs_info.static_vertices
                                   : nir->info.gs.vertices_out;
      shader->gs = (struct panvk_gs_info){
         .present = 1,
         .mode = gs_info.mode,
         .shape = gs_info.shape,
         .xfb = gs_info.xfb,
         .prefix_sum = gs_info.prefix_sum,
         .count_words = gs_info.count_words,
         .max_indices = gs_info.max_indices,
         .vertex_stride =
            vertices * util_next_power_of_two(MAX2(streams, 1)),
         .dynamic_vertices = count != NULL,
      };

      panvk_gs_build_varying_layout(nir, &shader->gs.varyings);
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_gs_vertex_output,
               nir_metadata_control_flow, &shader->gs.varyings);
      panvk_lower_gs_padded_invocations(nir);
      NIR_PASS(_, nir, poly_nir_lower_sysvals);

      nir->info.stage = MESA_SHADER_COMPUTE;
      memset(&nir->info.cs, 0, sizeof(nir->info.cs));
      nir->info.workgroup_size[0] = 64;
      nir->info.workgroup_size[1] = 1;
      nir->info.workgroup_size[2] = 1;
      nir->info.workgroup_size_variable = false;
      nir->xfb_info = NULL;
      nir->options = pan_get_nir_shader_compiler_options(
         PAN_ARCH, MESA_SHADER_COMPUTE, false);
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

      panvk_lower_nir(dev, nir, info->set_layout_count, info->set_layouts,
                      info->robustness, state, &shader->desc_info, false);

      struct panvk_shader_variant *main_variant =
         &shader->variants[PANVK_GS_VARIANT_MAIN];
      main_variant->own_bin = true;
      result = panvk_compile_nir(dev, nir, info->flags, &inputs, state,
                                 noperspective_varyings, &shader->desc_info,
                                 main_variant);

      if (result == VK_SUCCESS && count) {
         struct panvk_shader_variant *variant =
            &shader->variants[PANVK_GS_VARIANT_COUNT];

         panvk_lower_gs_padded_invocations(count);
         NIR_PASS(_, count, poly_nir_lower_sysvals);

         count->info.stage = MESA_SHADER_COMPUTE;
         memset(&count->info.cs, 0, sizeof(count->info.cs));
         count->info.workgroup_size[0] = 64;
         count->info.workgroup_size[1] = 1;
         count->info.workgroup_size[2] = 1;
         count->info.workgroup_size_variable = false;
         count->xfb_info = NULL;
         count->options = pan_get_nir_shader_compiler_options(
            PAN_ARCH, MESA_SHADER_COMPUTE, false);
         nir_shader_gather_info(count, nir_shader_get_entrypoint(count));

         panvk_lower_nir(dev, count, info->set_layout_count, info->set_layouts,
                         info->robustness, state, &shader->gs.count_desc,
                         false);

         variant->own_bin = true;
         result = panvk_compile_nir(dev, count, info->flags, &inputs, state,
                                    noperspective_varyings,
                                    &shader->gs.count_desc, variant);
      }
      ralloc_free(count);

      if (result == VK_SUCCESS) {
         struct panvk_shader_variant *variant =
            &shader->variants[PANVK_GS_VARIANT_RAST];
         nir_shader *rast = panvk_create_gs_rast_vs(&shader->gs.varyings);

         inputs.no_idvs = false;
         inputs.screen_space_position = true;
         panvk_lower_nir(dev, rast, info->set_layout_count, info->set_layouts,
                         info->robustness, state, &shader->desc_info, false);

         struct pan_varying_layout varying_layout;
         pan_varying_collect_formats(&varying_layout, rast, inputs.gpu_id);
         pan_build_varying_layout_compact(&varying_layout, rast,
                                          inputs.gpu_id);
         inputs.varying_layout = &varying_layout;

         variant->own_bin = true;
         result = panvk_compile_nir(dev, rast, info->flags, &inputs, state,
                                    noperspective_varyings, &shader->desc_info,
                                    variant);
         inputs.screen_space_position = false;
         ralloc_free(rast);
      }

      if (result != VK_SUCCESS) {
         panvk_shader_destroy(&dev->vk, &shader->vk, pAllocator);
         return result;
      }

      break;
   }

   case MESA_SHADER_TESS_EVAL: {
      struct panvk_shader_variant *variant =
         (struct panvk_shader_variant *)panvk_shader_only_variant(shader);

      nir_shader *nir = info->nir;

      /*
       * Retain the original TES topology before libpoly changes the
       * physical execution stage to MESA_SHADER_VERTEX.
       */
      shader->tess.mode = nir->info.tess._primitive_mode;
      shader->tess.spacing = nir->info.tess.spacing;
      shader->tess.points = nir->info.tess.point_mode;
      shader->tess.ccw = nir->info.tess.ccw;

      /*
       * libpoly consumes regular lowered tessellation IO.
       * Keep the shader as TESS_EVAL until poly has rewritten all TES
       * inputs and tessellation-specific system values.
       */
      nir_assign_io_var_locations(nir, nir_var_shader_in);
      nir_assign_io_var_locations(nir, nir_var_shader_out);
      panvk_lower_tess_nir_io(nir);

      /*
       * Execute TES as the real hardware vertex stage.
       *
       * poly_nir_lower_tes(..., true) rewrites TES input addressing and
       * changes nir->info.stage from TESS_EVAL to VERTEX.
       */
      NIR_PASS(_, nir, poly_nir_lower_tes, true);

      /*
       * poly_nir_lower_tes() may add a default point-size output after
       * TES IO has already been lowered.  Recompute lowered output bases so
       * newly introduced built-ins cannot alias an existing output base.
       */
      NIR_PASS(_, nir, nir_recompute_io_bases, nir_var_shader_out);

      NIR_PASS(_, nir, poly_nir_lower_sysvals);

      assert(nir->info.stage == MESA_SHADER_VERTEX);

      /*
       * From here on use the Valhall vertex compiler assumptions.
       */
      nir->options =
         pan_get_nir_shader_compiler_options(
            PAN_ARCH, MESA_SHADER_VERTEX, false);

      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

      const uint32_t clip_cull = panvk_clip_cull_count(nir);
      NIR_PASS(_, nir, panvk_nir_lower_cull_distance_vs);

      /*
       * TES inputs have already been replaced by libpoly accesses, so this
       * is not a normal Vulkan vertex-input/VBO path.
       */
      inputs.no_idvs = false;

      panvk_lower_nir(dev, nir,
                      info->set_layout_count,
                      info->set_layouts,
                      info->robustness,
                      state,
                      &shader->desc_info,
                      false);

      /*
       * TES outputs now behave exactly like VS outputs from the Panfrost
       * backend's point of view.
       */
      struct pan_varying_layout varying_layout;
      pan_varying_collect_formats(&varying_layout, nir, inputs.gpu_id);
      pan_build_varying_layout_compact(&varying_layout, nir,
                                       inputs.gpu_id);
      inputs.varying_layout = &varying_layout;

      NIR_PASS(_, nir, nir_opt_constant_folding);

      variant->own_bin = true;

      result = panvk_compile_nir(dev, nir,
                                 info->flags,
                                 &inputs,
                                 state,
                                 noperspective_varyings,
                                 &shader->desc_info,
                                 variant);

      if (result != VK_SUCCESS) {
         panvk_shader_destroy(&dev->vk, &shader->vk, pAllocator);
         return result;
      }

      variant->info.vs.clip_distance_count = clip_cull & 0xf;
      variant->info.vs.cull_distance_count = clip_cull >> 4;

      break;
   }

#endif

   case MESA_SHADER_FRAGMENT: {
      struct panvk_shader_variant *variant =
         (struct panvk_shader_variant *)panvk_shader_only_variant(shader);

      nir_shader *nir = info->nir;

      if (state && state->ms && state->ms->sample_shading_enable)
         nir->info.fs.uses_sample_shading = true;

#if PAN_ARCH >= 14
      if (!nir->info.fs.uses_sample_shading &&
          fs_may_use_vrs(&dev->vk.enabled_features, state))
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_vrs_frag_center,
                  nir_metadata_control_flow, NULL);
#endif

      /* We need to lower input attachments before we lower descriptors */
      NIR_PASS(_, nir, panvk_per_arch(nir_lower_input_attachment_loads),
               state, &variant->fs.input_attachment_read);

      NIR_PASS(_, nir, panvk_nir_lower_tile_image,
               &variant->fs.tile_image_color_read,
               &variant->fs.tile_image_z_read,
               &variant->fs.tile_image_s_read);

      NIR_PASS(_, nir, panvk_nir_lower_clip_cull_fs, vs_clip_cull);

      /* Lower input intrinsics for fragment shaders early to get the max
       * number of varying loads, as this number is required during descriptor
       * lowering for v9+.
       */
      nir_assign_io_var_locations(nir, nir_var_shader_in);

      /* VS (if known) decides the memory layout */
      inputs.varying_layout = vs_varying_layout;

      struct panvk_blend_static_key blend_key;
      panvk_per_arch(blend_static_key_init)(&blend_key, state);
      inputs.fixed_function_blend =
         panvk_per_arch(blend_fixed_function_locations)(&blend_key);

      panvk_lower_nir(dev, nir, info->set_layout_count, info->set_layouts,
                      info->robustness, state, &shader->desc_info, false);

      nir_assign_io_var_locations(nir, nir_var_shader_out);
      panvk_lower_nir_io(nir);

      /* Lower FS outputs now so that we can lower load_blend_descriptor_pan
       * to a driver-provided FAU instead of using the blend descriptors
       * uploaded by the hardware.  See panvk_vX_blend.c for details.
       */
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      const uint64_t atest_outputs = BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK) |
                                     BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                                     BITFIELD64_BIT(FRAG_RESULT_STENCIL);
      const bool skip_atest = fs_alpha_to_coverage_disabled(state) &&
                              !nir->info.fs.uses_discard &&
                              !(nir->info.outputs_written & atest_outputs);
      NIR_PASS(_, nir, pan_nir_lower_fs_outputs, skip_atest,
               0 /* fragcolor_nr_cbufs */,
               PAN_ARCH >= 15 ? inputs.fixed_function_blend : 0,
               PAN_ARCH >= 15 && fs_alpha_to_coverage_disabled(state));

      variant->own_bin = true;

      result = panvk_compile_nir(dev, nir, info->flags, &inputs, state,
                                 noperspective_varyings,
                                 &shader->desc_info, variant);
      if (result != VK_SUCCESS) {
         panvk_shader_destroy(&dev->vk, &shader->vk, pAllocator);
         return result;
      }

      #if PAN_ARCH >= 9
      /* LD_VAR_BUF[_IMM] has a fixed-size offset, when shaders overflow
       * that they fall back to LD_VAR[_IMM] and require descriptors.
       * TODO: We could only emit descriptors that overflow the offset,
       *       saving a bit of space.
       */
      shader->desc_info.fs_varying_attr_desc_count =
         variant->info.bifrost.uses_ld_var ? nir->num_inputs : 0;
      #endif

      break;
   }

   case MESA_SHADER_COMPUTE: {
      struct panvk_shader_variant *variant =
         (struct panvk_shader_variant *)panvk_shader_only_variant(shader);

      nir_shader *nir = info->nir;

#if PAN_ARCH >= 9
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      variant->info.cs.allow_merging_workgroups =
         valhall_can_merge_workgroups(nir);

      /* With merged workgroups, we need to use different divergence analysis
       * options to take into account that threads from different workgroups
       * may be in the same subgroup */
      if (variant->info.cs.allow_merging_workgroups) {
         nir->options = pan_get_nir_shader_compiler_options(
            PAN_ARCH, MESA_SHADER_COMPUTE, true);
         /* Invalidate the old divergence analysis */
         nir_foreach_function_impl(impl, nir)
            nir_progress(true, impl, ~nir_metadata_divergence);
      }
#endif

      panvk_lower_nir(dev, nir, info->set_layout_count, info->set_layouts,
                      info->robustness, state, &shader->desc_info,
                      variant->info.cs.allow_merging_workgroups);

      variant->own_bin = true;

      result = panvk_compile_nir(dev, nir, info->flags, &inputs, state,
                                 noperspective_varyings,
                                 &shader->desc_info, variant);
      if (result != VK_SUCCESS) {
         panvk_shader_destroy(&dev->vk, &shader->vk, pAllocator);
         return result;
      }
      break;
   }

   default:
      UNREACHABLE("Unknown shader stage");
   }

#ifndef PANVK_OFFLINE_ONLY
   if (upload) {
      panvk_shader_foreach_variant(shader, variant) {
         result = panvk_shader_upload(dev, variant, pAllocator);
         if (result != VK_SUCCESS) {
            panvk_shader_destroy(&dev->vk, &shader->vk, pAllocator);
            return result;
         }
      }
   }

#endif

   *shader_out = &shader->vk;

   return result;
}

struct panvk_shader_fallback {
   struct util_queue_fence queued;
   struct util_queue_fence done;
   uint32_t claimed;
   struct panvk_device *dev;
   const VkAllocationCallbacks *alloc;
   bool upload;
   struct vk_shader_compile_info info;
   struct vk_pipeline_robustness_state robustness;
   struct vk_descriptor_set_layout *set_layouts[MESA_VK_MAX_DESCRIPTOR_SETS];
   struct vk_graphics_pipeline_state state;
   void *state_mem;
   bool has_state;
   struct pan_varying_layout vs_varying_layout;
   bool has_vs_varying_layout;
   int vs_clip_cull;
   uint32_t noperspective_varyings;
   bool has_noperspective_varyings;
   VkResult result;
   struct vk_shader *shader;
};

#define PANVK_BG_COMPILE_MAX_PENDING 8

static void
panvk_bg_compile_init(const void *data)
{
   struct panvk_device *dev = (struct panvk_device *)data;
   const unsigned threads = CLAMP(util_get_cpu_caps()->nr_cpus / 2, 1, 4);

   if (!util_queue_init(&dev->bg_compile.queue, "panvk_bgc", 8, threads,
                        UTIL_QUEUE_INIT_RESIZE_IF_FULL |
                           UTIL_QUEUE_INIT_SET_FULL_THREAD_AFFINITY,
                        NULL)) {
      memset(&dev->bg_compile.queue, 0, sizeof(dev->bg_compile.queue));
      return;
   }

   p_atomic_set(&dev->bg_compile.ready, true);

   if (PANVK_DEBUG(STARTUP))
      mesa_logi("panvk: %u background shader compile threads", threads);
}

static struct util_queue *
panvk_bg_compile_queue(struct panvk_device *dev)
{
   util_call_once_data(&dev->bg_compile.once, panvk_bg_compile_init, dev);

   return p_atomic_read(&dev->bg_compile.ready) ? &dev->bg_compile.queue
                                                : NULL;
}

static bool
panvk_alloc_is_default(const VkAllocationCallbacks *alloc)
{
   const VkAllocationCallbacks *def = vk_default_allocator();

   return alloc->pfnAllocation == def->pfnAllocation &&
          alloc->pfnReallocation == def->pfnReallocation &&
          alloc->pfnFree == def->pfnFree;
}

static void
panvk_shader_fallback_release(struct panvk_shader_fallback *job)
{
   ralloc_free(job->info.nir);
   job->info.nir = NULL;

   vk_free(&job->dev->vk.alloc, job->state_mem);
   job->state_mem = NULL;

   for (uint32_t i = 0; i < job->info.set_layout_count; i++) {
      if (job->set_layouts[i] != NULL)
         vk_descriptor_set_layout_unref(&job->dev->vk, job->set_layouts[i]);
      job->set_layouts[i] = NULL;
   }

   p_atomic_dec(&job->dev->bg_compile.pending);
}

static void
panvk_shader_fallback_run(struct panvk_shader_fallback *job)
{
   job->result = panvk_compile_shader_impl(
      job->dev, &job->info, job->has_state ? &job->state : NULL,
      job->has_vs_varying_layout ? &job->vs_varying_layout : NULL,
      job->vs_clip_cull,
      job->has_noperspective_varyings ? &job->noperspective_varyings : NULL,
      job->alloc, &job->shader, false, job->upload);

   panvk_shader_fallback_release(job);

   if (job->upload)
      util_post_flush_inval_fence();

   util_queue_fence_signal(&job->done);
}

static void
panvk_shader_fallback_execute(void *data, UNUSED void *gdata,
                              UNUSED int thread_index)
{
   struct panvk_shader_fallback *job = data;

   if (p_atomic_cmpxchg(&job->claimed, 0, 1) == 0)
      panvk_shader_fallback_run(job);
}

static UNUSED VkResult
panvk_shader_fallback_wait(struct panvk_shader_fallback *job)
{
   if (p_atomic_read(&job->claimed) == 0 &&
       p_atomic_cmpxchg(&job->claimed, 0, 1) == 0)
      panvk_shader_fallback_run(job);
   else
      util_queue_fence_wait(&job->done);

   return job->result;
}

static void
panvk_shader_fallback_destroy(struct panvk_device *dev,
                              struct panvk_shader_fallback *job)
{
   if (p_atomic_cmpxchg(&job->claimed, 0, 1) == 0) {
      panvk_shader_fallback_release(job);
      util_queue_fence_signal(&job->done);
   } else {
      util_queue_fence_wait(&job->done);
   }

   util_queue_drop_job(&dev->bg_compile.queue, &job->queued);

   if (job->shader != NULL)
      panvk_shader_destroy(&dev->vk, job->shader, job->alloc);

   util_queue_fence_destroy(&job->queued);
   util_queue_fence_destroy(&job->done);
   free(job);
}

static bool
panvk_shader_defer_fallback(struct panvk_device *dev,
                            struct panvk_shader *shader,
                            const struct vk_shader_compile_info *info,
                            const struct vk_graphics_pipeline_state *state,
                            const struct pan_varying_layout *vs_varying_layout,
                            int vs_clip_cull,
                            const uint32_t *noperspective_varyings,
                            const VkAllocationCallbacks *pAllocator,
                            bool upload)
{
   if (pan_will_dump_shaders(PAN_ARCH) || PANVK_DEBUG(NIR))
      return false;

   if (!panvk_alloc_is_default(&dev->vk.alloc) ||
       (pAllocator != NULL && pAllocator != &dev->vk.alloc))
      return false;

   assert(info->set_layout_count <= MESA_VK_MAX_DESCRIPTOR_SETS);

   struct util_queue *queue = panvk_bg_compile_queue(dev);
   if (queue == NULL)
      return false;

   if (p_atomic_inc_return(&dev->bg_compile.pending) >
       PANVK_BG_COMPILE_MAX_PENDING) {
      p_atomic_dec(&dev->bg_compile.pending);
      return false;
   }

   struct panvk_shader_fallback *job = calloc(1, sizeof(*job));
   if (job == NULL) {
      p_atomic_dec(&dev->bg_compile.pending);
      return false;
   }

   if (state != NULL) {
      VkResult result = vk_graphics_pipeline_state_copy(
         &dev->vk, &job->state, state, NULL,
         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, &job->state_mem);
      if (result != VK_SUCCESS) {
         free(job);
         p_atomic_dec(&dev->bg_compile.pending);
         return false;
      }
      job->has_state = true;
   }

   job->dev = dev;
   job->alloc = pAllocator;
   job->upload = upload;
   job->info = *info;
   job->robustness = *info->robustness;
   job->info.robustness = &job->robustness;
   for (uint32_t i = 0; i < info->set_layout_count; i++) {
      if (info->set_layouts[i] != NULL)
         job->set_layouts[i] =
            vk_descriptor_set_layout_ref(info->set_layouts[i]);
   }
   job->info.set_layouts = job->set_layouts;
   job->info.embedded_sampler_count = 0;
   job->info.embedded_samplers = NULL;
   job->info.push_constant_range_count = 0;
   job->info.push_constant_ranges = NULL;

   if (vs_varying_layout != NULL) {
      job->vs_varying_layout = *vs_varying_layout;
      job->has_vs_varying_layout = true;
   }
   job->vs_clip_cull = vs_clip_cull;
   if (noperspective_varyings != NULL) {
      job->noperspective_varyings = *noperspective_varyings;
      job->has_noperspective_varyings = true;
   }

   util_queue_fence_init(&job->queued);
   util_queue_fence_init(&job->done);
   util_queue_fence_reset(&job->done);

   shader->bg_no_preamble = job;
   util_queue_add_job(queue, job, &job->queued, panvk_shader_fallback_execute,
                      NULL, 0);
   return true;
}

static VkResult
panvk_compile_shader_with_upload(struct panvk_device *dev,
                     struct vk_shader_compile_info *info,
                     const struct vk_graphics_pipeline_state *state,
                     const struct pan_varying_layout *vs_varying_layout,
                     int vs_clip_cull,
                     const uint32_t *noperspective_varyings,
                     const VkAllocationCallbacks *pAllocator,
                     struct vk_shader **shader_out, bool upload,
                     bool enable_preamble)
{
   bool eligible = enable_preamble && PAN_ARCH >= 10 &&
      !info->nir->info.internal &&
      !(info->flags & VK_SHADER_CREATE_INDIRECT_BINDABLE_BIT_EXT) &&
      (info->stage == MESA_SHADER_COMPUTE ||
       info->stage == MESA_SHADER_FRAGMENT ||
       (info->stage == MESA_SHADER_VERTEX &&
        !(info->next_stage_mask &
          (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
           VK_SHADER_STAGE_GEOMETRY_BIT))));
   struct vk_shader_compile_info fallback_info = *info;
   fallback_info.nir = eligible ? nir_shader_clone(NULL, info->nir) : NULL;
   VkResult result = panvk_compile_shader_impl(
      dev, info, state, vs_varying_layout, vs_clip_cull, noperspective_varyings,
      pAllocator, shader_out, eligible, upload);
   if (result == VK_SUCCESS && eligible) {
      struct panvk_shader *shader =
         container_of(*shader_out, struct panvk_shader, vk);
      bool has_preamble = false;
      panvk_shader_foreach_variant(shader, variant)
         has_preamble |= variant->preamble != NULL;
      if (has_preamble &&
          panvk_shader_defer_fallback(dev, shader, &fallback_info, state,
                                      vs_varying_layout, vs_clip_cull,
                                      noperspective_varyings, pAllocator,
                                      upload)) {
         fallback_info.nir = NULL;
      } else if (has_preamble) {
         struct vk_shader *fallback = NULL;
         result = panvk_compile_shader_impl(
            dev, &fallback_info, state, vs_varying_layout, vs_clip_cull,
            noperspective_varyings, pAllocator, &fallback, false, upload);
         if (result == VK_SUCCESS) {
            shader->no_preamble =
               container_of(fallback, struct panvk_shader, vk);
         } else {
            panvk_shader_destroy(&dev->vk, *shader_out, pAllocator);
            *shader_out = NULL;
         }
      }
   }
   ralloc_free(fallback_info.nir);
   return result;
}

#ifndef PANVK_OFFLINE_ONLY
static VkResult
panvk_compile_shader(struct panvk_device *dev,
                     struct vk_shader_compile_info *info,
                     const struct vk_graphics_pipeline_state *state,
                     const struct pan_varying_layout *vs_varying_layout,
                     int vs_clip_cull,
                     const uint32_t *noperspective_varyings,
                     const VkAllocationCallbacks *pAllocator,
                     struct vk_shader **shader_out)
{
   const VkShaderStageFlags tcs_bit = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                                      VK_SHADER_STAGE_GEOMETRY_BIT;

   if (info->stage != MESA_SHADER_VERTEX ||
       !(info->next_stage_mask & tcs_bit) ||
       !(info->next_stage_mask & ~tcs_bit))
      return panvk_compile_shader_with_upload(
         dev, info, state, vs_varying_layout, vs_clip_cull,
         noperspective_varyings, pAllocator, shader_out, true, true);

   struct vk_shader_compile_info hw_info = *info;
   hw_info.next_stage_mask &= ~tcs_bit;

   struct vk_shader_compile_info tess_info = *info;
   tess_info.next_stage_mask = tcs_bit;
   tess_info.nir = nir_shader_clone(NULL, info->nir);
   if (tess_info.nir == NULL)
      return panvk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result = panvk_compile_shader_with_upload(
      dev, &hw_info, state, vs_varying_layout, vs_clip_cull,
      noperspective_varyings, pAllocator, shader_out, true, true);
   if (result == VK_SUCCESS) {
      struct vk_shader *tess_vs = NULL;

      result = panvk_compile_shader_with_upload(
         dev, &tess_info, state, NULL, 0, noperspective_varyings, pAllocator,
         &tess_vs, true, true);
      if (result == VK_SUCCESS) {
         container_of(*shader_out, struct panvk_shader, vk)->tess_vs =
            container_of(tess_vs, struct panvk_shader, vk);
      } else {
         panvk_shader_destroy(&dev->vk, *shader_out, pAllocator);
         *shader_out = NULL;
      }
   }

   ralloc_free(tess_info.nir);
   return result;
}

#endif

VkResult
panvk_per_arch(compile_shader_offline)(
   struct panvk_device *dev, struct vk_shader_compile_info *info,
   const struct vk_graphics_pipeline_state *state,
   const uint32_t *noperspective_varyings, bool enable_preamble,
   struct vk_shader **shader_out)
{
   return panvk_compile_shader_with_upload(dev, info, state, NULL, 0,
                                           noperspective_varyings, NULL,
                                           shader_out, false, enable_preamble);
}

#ifndef PANVK_OFFLINE_ONLY
VkResult
panvk_per_arch(create_shader_from_binary)(struct panvk_device *dev,
                                          const struct pan_shader_info *info,
                                          struct pan_compute_dim local_size,
                                          const void *bin_ptr, size_t bin_size,
                                          struct panvk_shader **shader_out)
{
   struct panvk_shader *shader;
   VkResult result;

   size_t size =
      sizeof(struct panvk_shader) + sizeof(struct panvk_shader_variant) *
                                       panvk_shader_num_variants(info->stage);
   shader = vk_shader_zalloc(&dev->vk, &panvk_shader_ops, info->stage,
                             &dev->vk.alloc, size);
   if (shader == NULL)
      return panvk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   assert(panvk_shader_num_variants(info->stage) == 1);

   struct panvk_shader_variant *variant = &shader->variants[0];
   variant->info = *info;
   variant->cs.local_size = local_size;
   variant->bin_ptr = bin_ptr;
   variant->bin_size = bin_size;
   variant->own_bin = false;
   variant->nir_str = NULL;
   variant->asm_str = NULL;

   result = panvk_shader_upload(dev, variant, &dev->vk.alloc);

   if (result != VK_SUCCESS) {
      panvk_shader_destroy(&dev->vk, &shader->vk, &dev->vk.alloc);
      return result;
   }

   *shader_out = shader;

   return result;
}

static VkResult
panvk_create_folded_gs(struct panvk_device *dev,
                       const VkAllocationCallbacks *pAllocator,
                       struct vk_shader **shader_out)
{
   size_t size = sizeof(struct panvk_shader) +
                 sizeof(struct panvk_shader_variant) *
                    panvk_shader_num_variants(MESA_SHADER_GEOMETRY);
   struct panvk_shader *shader = vk_shader_zalloc(
      &dev->vk, &panvk_shader_ops, MESA_SHADER_GEOMETRY, pAllocator, size);
   if (shader == NULL)
      return panvk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   shader->variants[0].info.stage = MESA_SHADER_GEOMETRY;
   *shader_out = &shader->vk;
   return VK_SUCCESS;
}

static VkResult
compile_shaders(struct vk_device *vk_dev, uint32_t shader_count,
                struct vk_shader_compile_info *infos,
                const struct vk_graphics_pipeline_state *state,
                const struct vk_features *enabled_features,
                const VkAllocationCallbacks *pAllocator,
                struct vk_shader **shaders_out)
{
   struct panvk_device *dev = to_panvk_device(vk_dev);
   bool use_static_noperspective = false;
   uint32_t noperspective_varyings = 0;
   int32_t folded_gs = -1;
   VkResult result;
   int32_t i;

   for (i = 0; i < shader_count; i++) {
      if (infos[i].stage != MESA_SHADER_GEOMETRY)
         continue;

      if (PAN_ARCH < 10) {
         result = panvk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                               "unsupported geometry shader");
         i = 0;
         goto err_cleanup;
      }

      if (i > 0 &&
          panvk_nir_fold_passthrough_gs(infos[i - 1].nir, infos[i].nir)) {
         infos[i - 1].next_stage_mask = infos[i].next_stage_mask;
         folded_gs = i;
      } else if (PAN_ARCH < 12) {
         result = panvk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                               "unsupported geometry shader");
         i = 0;
         goto err_cleanup;
      }
   }

   if (shader_count == 2 && infos[0].stage == MESA_SHADER_VERTEX &&
       infos[1].stage == MESA_SHADER_FRAGMENT &&
       infos[0].next_stage_mask == VK_SHADER_STAGE_FRAGMENT_BIT &&
       (infos[0].flags & VK_SHADER_CREATE_LINK_STAGE_BIT_EXT) &&
       (infos[1].flags & VK_SHADER_CREATE_LINK_STAGE_BIT_EXT))
      pan_nir_link_varyings(infos[0].nir, infos[1].nir);

   /* If we are linking VS and FS, we can use the static interpolation
    * qualifiers from the FS in the VS.  Vulkan runtime passes us shaders in
    * stage order, so the FS will always be last if it exists.
    */
   if (infos[shader_count - 1].nir->info.stage == MESA_SHADER_FRAGMENT) {
      nir_shader *nir = infos[shader_count - 1].nir;
      noperspective_varyings = pan_nir_collect_noperspective_varyings_fs(nir);
      use_static_noperspective = true;
   }

   /* Vulkan runtime passes us shaders in stage order */
   for (i = 0; i < shader_count; i++) {
      if (i == folded_gs) {
         result = panvk_create_folded_gs(dev, pAllocator, &shaders_out[i]);
         if (result != VK_SUCCESS)
            goto err_cleanup;

         ralloc_free(infos[i].nir);
         continue;
      }

      const uint32_t *noperspective_varyings_ptr =
         use_static_noperspective ? &noperspective_varyings : NULL;
      const int32_t prev = i - 1 == folded_gs ? i - 2 : i - 1;

      /* For fragment shaders, Look at the previous stage to see if we can
       * find a varying layout we can use instead of making up our own.
       */
      const struct pan_varying_layout *vs_varying_layout = NULL;
      int vs_clip_cull = -1;
      if (infos[i].stage == MESA_SHADER_FRAGMENT && prev >= 0 &&
          infos[prev].next_stage_mask == VK_SHADER_STAGE_FRAGMENT_BIT) {
         struct panvk_shader *prev_shader =
            container_of(shaders_out[prev], struct panvk_shader, vk);

         /* The last geometry stage before the FS will always have a HW vertex
          * shader variant.
          */
         panvk_shader_foreach_variant(prev_shader, variant) {
            if (variant->info.stage == MESA_SHADER_VERTEX) {
               vs_varying_layout = &variant->info.varyings.formats;
               pan_varying_layout_require_layout(vs_varying_layout);
               vs_clip_cull = variant->info.vs.clip_distance_count |
                              (variant->info.vs.cull_distance_count << 4);
               break;
            }
         }
      }

      result = panvk_compile_shader(dev, &infos[i], state, vs_varying_layout,
                                    vs_clip_cull, noperspective_varyings_ptr,
                                    pAllocator, &shaders_out[i]);

      if (result != VK_SUCCESS)
         goto err_cleanup;

      /* Clean up NIR for the current shader */
      ralloc_free(infos[i].nir);
   }

   return VK_SUCCESS;

err_cleanup:
   /* Clean up all the shaders before this point */
   for (int32_t j = 0; j < i; j++)
      panvk_shader_destroy(&dev->vk, shaders_out[j], pAllocator);

   /* Clean up all the NIR from this point */
   for (int32_t j = i; j < shader_count; j++)
      ralloc_free(infos[j].nir);

   /* Memset the output array */
   memset(shaders_out, 0, shader_count * sizeof(*shaders_out));

   return result;
}

static simple_mtx_t compiler_mutex = SIMPLE_MTX_INITIALIZER;

void
panvk_per_arch(compiler_lock)(void)
{
   if (pan_will_dump_shaders(PAN_ARCH) || PANVK_DEBUG(NIR))
      simple_mtx_lock(&compiler_mutex);
}

void
panvk_per_arch(compiler_unlock)(void)
{
   if (pan_will_dump_shaders(PAN_ARCH) || PANVK_DEBUG(NIR))
      simple_mtx_unlock(&compiler_mutex);
}

static VkResult
panvk_compile_shaders(struct vk_device *vk_dev, uint32_t shader_count,
                      struct vk_shader_compile_info *infos,
                      const struct vk_graphics_pipeline_state *state,
                      const struct vk_features *enabled_features,
                      const VkAllocationCallbacks *pAllocator,
                      struct vk_shader **shaders_out)
{
   panvk_per_arch(compiler_lock)();

   VkResult result = compile_shaders(vk_dev, shader_count, infos, state,
                                     enabled_features, pAllocator,
                                     shaders_out);

   panvk_per_arch(compiler_unlock)();

   return result;
}

static VkResult
shader_desc_info_deserialize(struct panvk_device *dev,
                             struct blob_reader *blob,
                             struct panvk_shader *shader)
{
   shader->desc_info.used_set_mask = blob_read_uint32(blob);

#if PAN_ARCH < 9
   shader->desc_info.dyn_ubos.count = blob_read_uint32(blob);
   if (shader->desc_info.dyn_ubos.count >
       ARRAY_SIZE(shader->desc_info.dyn_ubos.map))
      return panvk_error(shader, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   blob_copy_bytes(blob, shader->desc_info.dyn_ubos.map,
                   sizeof(*shader->desc_info.dyn_ubos.map) *
                      shader->desc_info.dyn_ubos.count);
   shader->desc_info.dyn_ssbos.count = blob_read_uint32(blob);
   if (shader->desc_info.dyn_ssbos.count >
       ARRAY_SIZE(shader->desc_info.dyn_ssbos.map))
      return panvk_error(shader, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   blob_copy_bytes(blob, shader->desc_info.dyn_ssbos.map,
                   sizeof(*shader->desc_info.dyn_ssbos.map) *
                      shader->desc_info.dyn_ssbos.count);

   uint32_t others_count = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(shader->desc_info.others.count); i++) {
      shader->desc_info.others.count[i] = blob_read_uint32(blob);
      if (shader->desc_info.others.count[i] > UINT32_MAX - others_count)
         return panvk_error(shader, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      others_count += shader->desc_info.others.count[i];
   }

   if (others_count) {
      struct panvk_pool_alloc_info alloc_info = {
         .size = others_count * sizeof(uint32_t),
         .alignment = sizeof(uint32_t),
      };
      shader->desc_info.others.map =
         panvk_pool_alloc_mem(&dev->mempools.rw, alloc_info);
      if (!panvk_priv_mem_check_alloc(shader->desc_info.others.map))
         return panvk_error(shader, VK_ERROR_OUT_OF_DEVICE_MEMORY);

      panvk_priv_mem_write_array(shader->desc_info.others.map, 0, uint32_t,
                                 others_count, copy_table) {
         blob_copy_bytes(blob, copy_table, others_count * sizeof(*copy_table));
      }
   }
#else
   shader->desc_info.dyn_bufs.count = blob_read_uint32(blob);
   if (shader->desc_info.dyn_bufs.count >
       ARRAY_SIZE(shader->desc_info.dyn_bufs.map))
      return panvk_error(shader, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   blob_copy_bytes(blob, shader->desc_info.dyn_bufs.map,
                   sizeof(*shader->desc_info.dyn_bufs.map) *
                      shader->desc_info.dyn_bufs.count);
   shader->desc_info.fs_varying_attr_desc_count = blob_read_uint32(blob);
#endif

   return VK_SUCCESS;
}

static VkResult
panvk_deserialize_shader_variant(struct vk_device *vk_dev,
                                 struct blob_reader *blob,
                                 const VkAllocationCallbacks *pAllocator,
                                 struct panvk_shader_variant *shader,
                                 bool is_preamble)
{
   struct panvk_device *device = to_panvk_device(vk_dev);
   struct pan_shader_info info;
   VkResult result;

   blob_copy_bytes(blob, &info, sizeof(info));
   if (blob->overrun)
      return panvk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   shader->info = info;
   blob_copy_bytes(blob, &shader->fau, sizeof(shader->fau));
   if (is_preamble &&
       (info.stage != MESA_SHADER_COMPUTE || shader->fau.total_count != 1))
      return panvk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   switch (shader->info.stage) {
   case MESA_SHADER_COMPUTE:
   case MESA_SHADER_KERNEL:
      blob_copy_bytes(blob, &shader->cs.local_size,
                      sizeof(shader->cs.local_size));
      break;

   case MESA_SHADER_FRAGMENT:
      shader->fs.earlyzs_lut = pan_earlyzs_analyze(&shader->info, PAN_ARCH);
      blob_copy_bytes(blob, &shader->fs.input_attachment_read,
                      sizeof(shader->fs.input_attachment_read));
      blob_copy_bytes(blob, &shader->fs.tile_image_color_read,
                      sizeof(shader->fs.tile_image_color_read));
      blob_copy_bytes(blob, &shader->fs.tile_image_z_read,
                      sizeof(shader->fs.tile_image_z_read));
      blob_copy_bytes(blob, &shader->fs.tile_image_s_read,
                      sizeof(shader->fs.tile_image_s_read));
      break;

   default:
      break;
   }

   shader->bin_size = blob_read_uint32(blob);

   if (blob->overrun)
      return panvk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   shader->bin_ptr = malloc(shader->bin_size);
   if (shader->bin_ptr == NULL)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   shader->own_bin = true;
   blob_copy_bytes(blob, (void *)shader->bin_ptr, shader->bin_size);

#if PAN_ARCH < 9
   shader->data_size = blob_read_uint32(blob);

   if (shader->data_size) {
      shader->data_ptr = malloc(shader->data_size);

      if (shader->data_ptr == NULL)
         return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

      blob_copy_bytes(blob, (void *)shader->data_ptr, shader->data_size);
   }
#endif

   uint32_t nir_str_size = blob_read_uint32(blob);
   uint32_t asm_str_size = blob_read_uint32(blob);
   const char *nir_str = blob_read_bytes(blob, nir_str_size);
   const char *asm_str = blob_read_bytes(blob, asm_str_size);

   if (blob->overrun)
      return panvk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   if (nir_str_size > 0) {
      shader->nir_str = ralloc_strndup(NULL, nir_str, nir_str_size);
      if (shader->nir_str == NULL)
         return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   if (asm_str_size > 0) {
      shader->asm_str = strndup(asm_str, asm_str_size);
      if (shader->asm_str == NULL)
         return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   uint8_t has_preamble = blob_read_uint8(blob);
   if (blob->overrun || has_preamble > 1 || (is_preamble && has_preamble))
      return panvk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   if (has_preamble) {
      shader->preamble = calloc(1, sizeof(*shader->preamble));
      if (!shader->preamble)
         return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
      result = panvk_deserialize_shader_variant(vk_dev, blob, pAllocator,
                                                shader->preamble, true);
      if (result != VK_SUCCESS)
         return result;
   }

   result = is_preamble ? VK_SUCCESS :
      panvk_shader_upload(device, shader, pAllocator);

   if (result != VK_SUCCESS)
      return result;

   return result;
}

static VkResult
panvk_deserialize_shader_impl(struct vk_device *vk_dev, struct blob_reader *blob,
                         const VkAllocationCallbacks *pAllocator,
                         struct vk_shader **shader_out, bool is_fallback)
{
   struct panvk_device *device = to_panvk_device(vk_dev);
   struct panvk_shader *shader;
   VkResult result;

   uint32_t version = blob_read_uint32(blob);
   mesa_shader_stage stage = blob_read_uint8(blob);
   if (blob->overrun || version != 0x50414e02 ||
       stage > MESA_SHADER_COMPUTE)
      return vk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   size_t size =
      sizeof(struct panvk_shader) +
      sizeof(struct panvk_shader_variant) * panvk_shader_num_variants(stage);
   shader =
      vk_shader_zalloc(vk_dev, &panvk_shader_ops, stage, pAllocator, size);
   if (shader == NULL)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = shader_desc_info_deserialize(device, blob, shader);
   if (result != VK_SUCCESS) {
      panvk_shader_destroy(vk_dev, &shader->vk, pAllocator);
      return result;
   }

   blob_copy_bytes(blob, &shader->tess, sizeof(shader->tess));
   blob_copy_bytes(blob, &shader->gs, sizeof(shader->gs));
   if (blob->overrun) {
      panvk_shader_destroy(vk_dev, &shader->vk, pAllocator);
      return panvk_error(device,
                         VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }

   panvk_shader_foreach_variant(shader, variant) {
      result = panvk_deserialize_shader_variant(vk_dev, blob, pAllocator,
                                                variant, false);
      if (result != VK_SUCCESS) {
         panvk_shader_destroy(vk_dev, &shader->vk, pAllocator);
         return result;
      }
   }

   uint8_t has_fallback = blob_read_uint8(blob);
   if (blob->overrun || has_fallback > 1 || (is_fallback && has_fallback)) {
      panvk_shader_destroy(vk_dev, &shader->vk, pAllocator);
      return panvk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }
   if (has_fallback) {
      struct vk_shader *fallback = NULL;
      result = panvk_deserialize_shader_impl(vk_dev, blob, pAllocator,
                                             &fallback, true);
      if (result != VK_SUCCESS) {
         panvk_shader_destroy(vk_dev, &shader->vk, pAllocator);
         return result;
      }
      shader->no_preamble = container_of(fallback, struct panvk_shader, vk);
      if (fallback->stage != stage) {
         panvk_shader_destroy(vk_dev, &shader->vk, pAllocator);
         return panvk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      }
   }

   uint8_t has_tess_vs = blob_read_uint8(blob);
   if (blob->overrun || has_tess_vs > 1 ||
       (has_tess_vs && (is_fallback || stage != MESA_SHADER_VERTEX))) {
      panvk_shader_destroy(vk_dev, &shader->vk, pAllocator);
      return panvk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }
   if (has_tess_vs) {
      struct vk_shader *tess_vs = NULL;
      result = panvk_deserialize_shader_impl(vk_dev, blob, pAllocator,
                                             &tess_vs, true);
      if (result != VK_SUCCESS) {
         panvk_shader_destroy(vk_dev, &shader->vk, pAllocator);
         return result;
      }
      shader->tess_vs = container_of(tess_vs, struct panvk_shader, vk);
      if (tess_vs->stage != MESA_SHADER_VERTEX) {
         panvk_shader_destroy(vk_dev, &shader->vk, pAllocator);
         return panvk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      }
   }

   *shader_out = &shader->vk;

   return VK_SUCCESS;
}

static VkResult
panvk_deserialize_shader(struct vk_device *vk_dev, struct blob_reader *blob,
                         uint32_t binary_version,
                         const VkAllocationCallbacks *pAllocator,
                         struct vk_shader **shader_out)
{
   return panvk_deserialize_shader_impl(vk_dev, blob, pAllocator,
                                        shader_out, false);
}

static void
shader_desc_info_serialize(struct blob *blob,
                           const struct panvk_shader *shader)
{
   blob_write_uint32(blob, shader->desc_info.used_set_mask);

#if PAN_ARCH < 9
   blob_write_uint32(blob, shader->desc_info.dyn_ubos.count);
   blob_write_bytes(blob, shader->desc_info.dyn_ubos.map,
                    sizeof(*shader->desc_info.dyn_ubos.map) *
                       shader->desc_info.dyn_ubos.count);
   blob_write_uint32(blob, shader->desc_info.dyn_ssbos.count);
   blob_write_bytes(blob, shader->desc_info.dyn_ssbos.map,
                    sizeof(*shader->desc_info.dyn_ssbos.map) *
                       shader->desc_info.dyn_ssbos.count);

   unsigned others_count = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(shader->desc_info.others.count); i++) {
      blob_write_uint32(blob, shader->desc_info.others.count[i]);
      others_count += shader->desc_info.others.count[i];
   }

   /* No need to wrap this one in panvk_priv_mem_readback(), because the
    * GPU is not supposed to touch it. */
   blob_write_bytes(blob,
                    panvk_priv_mem_host_addr(shader->desc_info.others.map),
                    sizeof(uint32_t) * others_count);
#else
   blob_write_uint32(blob, shader->desc_info.dyn_bufs.count);
   blob_write_bytes(blob, shader->desc_info.dyn_bufs.map,
                    sizeof(*shader->desc_info.dyn_bufs.map) *
                       shader->desc_info.dyn_bufs.count);
   blob_write_uint32(blob, shader->desc_info.fs_varying_attr_desc_count);
#endif
}

static bool
panvk_shader_serialize_variant(struct vk_device *vk_dev,
                               const struct panvk_shader_variant *shader,
                               struct blob *blob)
{
   blob_write_bytes(blob, &shader->info, sizeof(shader->info));
   blob_write_bytes(blob, &shader->fau, sizeof(shader->fau));

   switch (shader->info.stage) {
   case MESA_SHADER_COMPUTE:
   case MESA_SHADER_KERNEL:
      blob_write_bytes(blob, &shader->cs.local_size,
                       sizeof(shader->cs.local_size));
      break;

   case MESA_SHADER_FRAGMENT:
      blob_write_bytes(blob, &shader->fs.input_attachment_read,
                       sizeof(shader->fs.input_attachment_read));
      blob_write_bytes(blob, &shader->fs.tile_image_color_read,
                       sizeof(shader->fs.tile_image_color_read));
      blob_write_bytes(blob, &shader->fs.tile_image_z_read,
                       sizeof(shader->fs.tile_image_z_read));
      blob_write_bytes(blob, &shader->fs.tile_image_s_read,
                       sizeof(shader->fs.tile_image_s_read));
      break;

   default:
      break;
   }

   blob_write_uint32(blob, shader->bin_size);
   blob_write_bytes(blob, shader->bin_ptr, shader->bin_size);

#if PAN_ARCH < 9
   blob_write_uint32(blob, shader->data_size);
   blob_write_bytes(blob, shader->data_ptr, shader->data_size);
#endif

   /* Include the terminating NULL in the serialization */
   uint32_t nir_str_size = shader->nir_str ? strlen(shader->nir_str) + 1 : 0;
   uint32_t asm_str_size = shader->asm_str ? strlen(shader->asm_str) + 1 : 0;
   blob_write_uint32(blob, nir_str_size);
   blob_write_uint32(blob, asm_str_size);
   blob_write_bytes(blob, shader->nir_str, nir_str_size);
   blob_write_bytes(blob, shader->asm_str, asm_str_size);

   blob_write_uint8(blob, shader->preamble != NULL);
   if (shader->preamble &&
       !panvk_shader_serialize_variant(vk_dev, shader->preamble, blob))
      return false;

   return !blob->out_of_memory;
}

static bool
panvk_shader_serialize(struct vk_device *vk_dev,
                       const struct vk_shader *vk_shader, struct blob *blob)
{
   struct panvk_shader *shader =
      container_of(vk_shader, struct panvk_shader, vk);

   blob_write_uint32(blob, 0x50414e02);
   blob_write_uint8(blob, vk_shader->stage);

   shader_desc_info_serialize(blob, shader);
   blob_write_bytes(blob, &shader->tess, sizeof(shader->tess));
   blob_write_bytes(blob, &shader->gs, sizeof(shader->gs));

   panvk_shader_foreach_variant(shader, variant) {
      panvk_shader_serialize_variant(vk_dev, variant, blob);
   }

   const struct panvk_shader *no_preamble = shader->no_preamble;
   if (shader->bg_no_preamble != NULL) {
      if (panvk_shader_fallback_wait(shader->bg_no_preamble) != VK_SUCCESS ||
          shader->bg_no_preamble->shader == NULL)
         return false;
      no_preamble = container_of(shader->bg_no_preamble->shader,
                                 struct panvk_shader, vk);
   }

   blob_write_uint8(blob, no_preamble != NULL);
   if (no_preamble)
      panvk_shader_serialize(vk_dev, &no_preamble->vk, blob);

   blob_write_uint8(blob, shader->tess_vs != NULL);
   if (shader->tess_vs)
      panvk_shader_serialize(vk_dev, &shader->tess_vs->vk, blob);

   return !blob->out_of_memory;
}

static VkResult
panvk_shader_get_executable_properties(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t *executable_count, VkPipelineExecutablePropertiesKHR *properties)
{
   struct panvk_shader *shader =
      container_of(vk_shader, struct panvk_shader, vk);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, properties,
                          executable_count);

   panvk_shader_foreach_variant(shader, variant) {
      /* Ignore absent variants */
      if (variant->bin_size == 0)
         continue;

      const char *variant_name = panvk_shader_variant_name(shader, variant);
      const char *stage_name = _mesa_shader_stage_to_string(shader->vk.stage);

      vk_outarray_append_typed(VkPipelineExecutablePropertiesKHR, &out, props)
      {
         props->stages = mesa_to_vk_shader_stage(shader->vk.stage);
         props->subgroupSize = pan_subgroup_size(PAN_ARCH);

         if (variant_name != NULL) {
            VK_PRINT_STR(props->name, "%s %s", variant_name, stage_name);
            VK_PRINT_STR(props->description, "%s %s shader", variant_name,
                         stage_name);
         } else {
            VK_COPY_STR(props->name, stage_name);
            VK_PRINT_STR(props->description, "%s shader", stage_name);
         }
      }

      if (variant->info.stage == MESA_SHADER_VERTEX &&
          variant->info.vs.secondary_offset) {
         vk_outarray_append_typed(VkPipelineExecutablePropertiesKHR, &out,
                                  props)
         {
            props->stages = mesa_to_vk_shader_stage(shader->vk.stage);
            props->subgroupSize = pan_subgroup_size(PAN_ARCH);

            if (variant_name != NULL) {
               VK_PRINT_STR(props->name, "%s %s varying", variant_name,
                            stage_name);
               VK_PRINT_STR(props->description, "%s %s varying shader",
                            variant_name, stage_name);
            } else {
               VK_PRINT_STR(props->name, "%s varying", stage_name);
               VK_PRINT_STR(props->description, "%s varying shader",
                            stage_name);
            }
         }
      }
      if (variant->preamble) {
         vk_outarray_append_typed(VkPipelineExecutablePropertiesKHR, &out,
                                   props) {
            props->stages = mesa_to_vk_shader_stage(shader->vk.stage);
            props->subgroupSize = pan_subgroup_size(PAN_ARCH);
            VK_PRINT_STR(props->name, "%s preamble", stage_name);
            VK_PRINT_STR(props->description, "%s preamble", stage_name);
         }
      }
   }

   return vk_outarray_status(&out);
}

static const struct panvk_shader_variant *
get_variant_from_executable_index(struct panvk_shader *shader,
                                  uint32_t executable_index,
                                  bool *idvs_varying)
{
   *idvs_varying = false;
   uint32_t i = 0;

   panvk_shader_foreach_variant(shader, variant) {
      /* Ignore absent variants */
      if (variant->bin_size == 0)
         continue;

      if (i == executable_index)
         return variant;

      i++;

      /* VS variants that have a separate IDVS varying shader get two indices,
       * the first for the position shader and the second for the varying
       * shader
       */
      if (variant->info.stage == MESA_SHADER_VERTEX &&
          variant->info.vs.secondary_offset) {
         if (i == executable_index) {
            *idvs_varying = true;
            return variant;
         }

         i++;
      }
      if (variant->preamble) {
         if (i == executable_index)
            return variant->preamble;
         i++;
      }
   }

   return NULL;
}

static VkResult
panvk_shader_get_executable_statistics(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *statistic_count,
   VkPipelineExecutableStatisticKHR *statistics)
{
   struct panvk_shader *shader =
      container_of(vk_shader, struct panvk_shader, vk);

   bool needs_vary = false;

   const struct panvk_shader_variant *variant =
      get_variant_from_executable_index(shader, executable_index, &needs_vary);
   assert(variant != NULL);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableStatisticKHR, out, statistics,
                          statistic_count);

   const struct pan_stats *stats =
      needs_vary ? &variant->info.stats_idvs_varying : &variant->info.stats;

   vk_add_pan_stats(out, stats);
   return vk_outarray_status(&out);
}

static bool
write_ir_text(VkPipelineExecutableInternalRepresentationKHR *ir,
              const char *data)
{
   ir->isText = VK_TRUE;

   size_t data_len = strlen(data) + 1;

   if (ir->pData == NULL) {
      ir->dataSize = data_len;
      return true;
   }

   strncpy(ir->pData, data, ir->dataSize);
   if (ir->dataSize < data_len)
      return false;

   ir->dataSize = data_len;
   return true;
}

static VkResult
panvk_shader_get_executable_internal_representations(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *internal_representation_count,
   VkPipelineExecutableInternalRepresentationKHR *internal_representations)
{
   struct panvk_shader *shader =
      container_of(vk_shader, struct panvk_shader, vk);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableInternalRepresentationKHR, out,
                          internal_representations,
                          internal_representation_count);

   bool needs_vary = false;

   const struct panvk_shader_variant *variant =
      get_variant_from_executable_index(shader, executable_index, &needs_vary);
   assert(variant != NULL);

   /* XXX: Varying shader assembly */
   if (needs_vary)
      return vk_outarray_status(&out);

   bool incomplete_text = false;

   if (variant->nir_str != NULL) {
      vk_outarray_append_typed(VkPipelineExecutableInternalRepresentationKHR,
                               &out, ir)
      {
         VK_COPY_STR(ir->name, "NIR shader");
         VK_COPY_STR(ir->description,
                     "NIR shader before sending to the back-end compiler");
         if (!write_ir_text(ir, variant->nir_str))
            incomplete_text = true;
      }
   }

   if (variant->asm_str != NULL) {
      vk_outarray_append_typed(VkPipelineExecutableInternalRepresentationKHR,
                               &out, ir)
      {
         VK_COPY_STR(ir->name, "Assembly");
         VK_COPY_STR(ir->description, "Final Assembly");
         if (!write_ir_text(ir, variant->asm_str))
            incomplete_text = true;
      }
   }

   return incomplete_text ? VK_INCOMPLETE : vk_outarray_status(&out);
}

#if PAN_ARCH < 9
static mali_pixel_format
get_varying_format(mesa_shader_stage stage, gl_varying_slot loc,
                   enum pipe_format pfmt)
{
   switch (loc) {
   case VARYING_SLOT_PNTC:
   case VARYING_SLOT_PSIZ:
#if PAN_ARCH <= 6
      return (MALI_R16F << 12) | pan_get_default_swizzle(1);
#else
      return (MALI_R16F << 12) | MALI_RGB_COMPONENT_ORDER_R000;
#endif
   case VARYING_SLOT_POS:
#if PAN_ARCH <= 6
      return (MALI_SNAP_4 << 12) | pan_get_default_swizzle(4);
#else
      return (MALI_SNAP_4 << 12) | MALI_RGB_COMPONENT_ORDER_RGBA;
#endif
   default:
      assert(pfmt != PIPE_FORMAT_NONE);
      return GENX(pan_format_from_pipe_format)(pfmt)->hw;
   }
}

static inline enum panvk_varying_buf_id
varying_buf_id(gl_varying_slot loc)
{
   switch (loc) {
   case VARYING_SLOT_POS:
      return PANVK_VARY_BUF_POSITION;
   case VARYING_SLOT_PSIZ:
      return PANVK_VARY_BUF_PSIZ;
   default:
      return PANVK_VARY_BUF_GENERAL;
   }
}

static mali_pixel_format
varying_format(gl_varying_slot loc, enum pipe_format pfmt)
{
   switch (loc) {
   case VARYING_SLOT_PNTC:
   case VARYING_SLOT_PSIZ:
#if PAN_ARCH <= 6
      return (MALI_R16F << 12) | pan_get_default_swizzle(1);
#else
      return (MALI_R16F << 12) | MALI_RGB_COMPONENT_ORDER_R000;
#endif
   case VARYING_SLOT_POS:
#if PAN_ARCH <= 6
      return (MALI_SNAP_4 << 12) | pan_get_default_swizzle(4);
#else
      return (MALI_SNAP_4 << 12) | MALI_RGB_COMPONENT_ORDER_RGBA;
#endif
   default:
      return GENX(pan_format_from_pipe_format)(pfmt)->hw;
   }
}

static VkResult
emit_varying_attrs(struct panvk_pool *desc_pool,
                   const struct pan_varying_layout *varyings,
                   const unsigned *bit_sizes, const unsigned *buf_offsets,
                   struct panvk_priv_mem *mem)
{
   const unsigned varying_count = varyings->count;
   if (!varying_count) {
      *mem = (struct panvk_priv_mem){0};
      return VK_SUCCESS;
   }

   *mem = panvk_pool_alloc_desc_array(desc_pool, varying_count, ATTRIBUTE);
   if (!panvk_priv_mem_check_alloc(*mem))
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   panvk_priv_mem_write_array(*mem, 0, struct mali_attribute_packed,
                              varying_count, attrs) {
      unsigned attr_idx = 0;

      for (unsigned i = 0; i < varying_count; i++) {
         pan_pack(&attrs[attr_idx++], ATTRIBUTE, cfg) {
            const struct pan_varying_slot *slot =
               pan_varying_layout_slot_at(varyings, i);
            if (!slot)
               continue;

            gl_varying_slot loc = slot->location;

            enum pipe_format pfmt = PIPE_FORMAT_NONE;
            bool is_hw = loc < VARYING_SLOT_VAR0;
            bool is_present = is_hw || bit_sizes[loc] != 0;

            if (is_present && !is_hw) {
               nir_alu_type base_type =
                  nir_alu_type_get_base_type(slot->alu_type);
               pfmt =
                  pan_varying_format(base_type | bit_sizes[loc], slot->ncomps);
            }

            if (!is_present) {
#if PAN_ARCH >= 7
               cfg.format =
                  (MALI_CONSTANT << 12) | MALI_RGB_COMPONENT_ORDER_0000;
#else
               cfg.format = (MALI_CONSTANT << 12) | PAN_V6_SWIZZLE(0, 0, 0, 0);
#endif
            } else {
               cfg.buffer_index = varying_buf_id(loc);
               cfg.offset = buf_offsets[loc];
               cfg.format = varying_format(loc, pfmt);
            }
            cfg.offset_enable = false;
         }
      }
   }

   return VK_SUCCESS;
}

VkResult
panvk_per_arch(link_shaders)(struct panvk_pool *desc_pool,
                             const struct panvk_shader_variant *vs,
                             const struct panvk_shader_variant *fs,
                             struct panvk_shader_link *link)
{
   unsigned buf_strides[PANVK_VARY_BUF_MAX] = {0};
   unsigned buf_offsets[VARYING_SLOT_MAX] = {0};
   unsigned buf_sizes[VARYING_SLOT_MAX] = {0};

   assert(vs && vs->info.stage == MESA_SHADER_VERTEX);
   assert(!fs || fs->info.stage == MESA_SHADER_FRAGMENT);
   const struct pan_varying_layout *vs_layout = &vs->info.varyings.formats;
   pan_varying_layout_require_layout(vs_layout);

   /* Handle the position and point size buffers explicitly, as they are
    * passed through separate buffer pointers to the tiler job.
    */
   const struct pan_varying_slot *vs_pos_slot =
      pan_varying_layout_find_slot(vs_layout, VARYING_SLOT_POS);
   const struct pan_varying_slot *vs_psiz_slot =
      pan_varying_layout_find_slot(vs_layout, VARYING_SLOT_PSIZ);

   if (vs_pos_slot) {
      buf_strides[PANVK_VARY_BUF_POSITION] = sizeof(uint32_t) * 4;
      buf_sizes[VARYING_SLOT_POS] =
         nir_alu_type_get_type_size(vs_pos_slot->alu_type);
   }
   if (vs_psiz_slot) {
      buf_strides[PANVK_VARY_BUF_PSIZ] = sizeof(uint16_t);
      buf_sizes[VARYING_SLOT_PSIZ] =
         nir_alu_type_get_type_size(vs_psiz_slot->alu_type);
   }
   buf_strides[PANVK_VARY_BUF_GENERAL] = vs_layout->generic_size_B;

   /* When no fragment shader is present we can just ignore all
    * generic varyings writes.
    */
   const uint32_t fs_var_count = fs ? fs->info.varyings.formats.count : 0;
   for (uint32_t i = 0; i < fs_var_count; i++) {
      const struct pan_varying_slot *fs_var =
         pan_varying_layout_slot_at(&fs->info.varyings.formats, i);
      if (!fs_var)
         continue;

      const gl_varying_slot pos = fs_var->location;
      /* Skip special varyings. */
      if (pos < VARYING_SLOT_VAR0)
         continue;

      /* Special buffers are handled explicitly before this loop, everything
       * else should be laid out in the general varying buffer.
       */
      assert(varying_buf_id(pos) == PANVK_VARY_BUF_GENERAL);

      const struct pan_varying_slot *vs_slot =
         pan_varying_layout_find_slot(vs_layout, fs_var->location);
      if (vs_slot) {
         buf_offsets[pos] = vs_slot->offset;
         buf_sizes[pos] = nir_alu_type_get_type_size(vs_slot->alu_type);
      }
   }

   panvk_shader_link_cleanup(link);
   VkResult result = emit_varying_attrs(desc_pool, vs_layout, buf_sizes,
                                        buf_offsets, &link->vs.attribs);
   if (result != VK_SUCCESS)
      return result;

   if (fs) {
      pan_varying_layout_require_format(&fs->info.varyings.formats);
      result = emit_varying_attrs(desc_pool, &fs->info.varyings.formats,
                                  buf_sizes, buf_offsets, &link->fs.attribs);
      if (result != VK_SUCCESS)
         return result;
   }

   memcpy(link->buf_strides, buf_strides, sizeof(link->buf_strides));
   return VK_SUCCESS;
}
#endif

static const struct vk_shader_ops panvk_shader_ops = {
   .destroy = panvk_shader_destroy,
   .serialize = panvk_shader_serialize,
   .get_executable_properties = panvk_shader_get_executable_properties,
   .get_executable_statistics = panvk_shader_get_executable_statistics,
   .get_executable_internal_representations =
      panvk_shader_get_executable_internal_representations,
};

static struct panvk_shader *
panvk_cmd_get_no_preamble(struct panvk_cmd_buffer *cmd,
                          const struct panvk_shader *shader)
{
   struct panvk_shader_fallback *job = shader->bg_no_preamble;

   if (job == NULL)
      return shader->no_preamble;

   VkResult result = panvk_shader_fallback_wait(job);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return NULL;
   }

   return job->shader != NULL
             ? container_of(job->shader, struct panvk_shader, vk)
             : NULL;
}

static void
panvk_cmd_update_vs(struct panvk_cmd_buffer *cmd)
{
   const struct panvk_shader *vs = cmd->state.gfx.vs.bound;

   if (vs && vs->tess_vs &&
       (cmd->state.gfx.tess.tcs.shader || cmd->state.gfx.gs.shader))
      vs = vs->tess_vs;
#if PAN_ARCH >= 10
   else if (vs && (cmd->flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)) {
      const struct panvk_shader *no_preamble =
         panvk_cmd_get_no_preamble(cmd, vs);
      if (no_preamble)
         vs = no_preamble;
   }
#endif

   if (cmd->state.gfx.vs.shader != vs) {
      cmd->state.gfx.vs.shader = vs;
      gfx_state_set_dirty(cmd, VS);
      gfx_state_set_dirty(cmd, VS_PUSH_UNIFORMS);
   }
}

static void
panvk_cmd_bind_shader(struct panvk_cmd_buffer *cmd, const mesa_shader_stage stage,
                      struct panvk_shader *shader)
{
   const struct panvk_shader *bound = shader;

#if PAN_ARCH >= 10
   if (shader && (cmd->flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)) {
      struct panvk_shader *no_preamble = panvk_cmd_get_no_preamble(cmd, shader);
      if (no_preamble)
         shader = no_preamble;
   }
#endif

   switch (stage) {
   case MESA_SHADER_COMPUTE:
      if (cmd->state.compute.shader != shader) {
         cmd->state.compute.shader = shader;
         compute_state_set_dirty(cmd, CS);
         compute_state_set_dirty(cmd, PUSH_UNIFORMS);
      }
      break;
   case MESA_SHADER_VERTEX:
      cmd->state.gfx.vs.bound = bound;
      break;
   case MESA_SHADER_TESS_CTRL:
      if (cmd->state.gfx.tess.tcs.shader != shader) {
         cmd->state.gfx.tess.tcs.shader = shader;
         gfx_state_set_dirty(cmd, TCS);
         gfx_state_set_dirty(cmd, TCS_PUSH_UNIFORMS);
      }
      break;
   case MESA_SHADER_TESS_EVAL:
      if (cmd->state.gfx.tess.tes.shader != shader) {
         cmd->state.gfx.tess.tes.shader = shader;
         gfx_state_set_dirty(cmd, TES);
         gfx_state_set_dirty(cmd, TES_PUSH_UNIFORMS);
      }
      break;
   case MESA_SHADER_FRAGMENT:
      if (cmd->state.gfx.fs.shader != shader) {
         cmd->state.gfx.fs.shader = shader;
         gfx_state_set_dirty(cmd, FS);
         gfx_state_set_dirty(cmd, FS_PUSH_UNIFORMS);
      }
      break;
   case MESA_SHADER_GEOMETRY:
      if (shader && !shader->gs.present)
         shader = NULL;
      if (cmd->state.gfx.gs.shader != shader) {
         cmd->state.gfx.gs.shader = shader;
         gfx_state_set_dirty(cmd, GS);
      }
      break;
   default:
      assert(!"Unsupported stage");
      break;
   }
}

static void
panvk_cmd_bind_shaders(struct vk_command_buffer *vk_cmd, uint32_t stage_count,
                       const mesa_shader_stage *stages,
                       struct vk_shader **const shaders)
{
   struct panvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct panvk_cmd_buffer, vk);

   bool gfx = false;
   for (uint32_t i = 0; i < stage_count; i++) {
      struct panvk_shader *shader =
         container_of(shaders[i], struct panvk_shader, vk);

      panvk_cmd_bind_shader(cmd, stages[i], shader);
      gfx |= stages[i] != MESA_SHADER_COMPUTE;
   }

   if (gfx)
      panvk_cmd_update_vs(cmd);
}

const struct vk_device_shader_ops panvk_per_arch(offline_shader_ops) = {
   .get_nir_options = panvk_get_nir_options,
   .get_spirv_options = panvk_get_spirv_options,
   .preprocess_nir = panvk_preprocess_nir,
};

#ifndef PANVK_OFFLINE_ONLY
const struct vk_device_shader_ops panvk_per_arch(device_shader_ops) = {
   .get_nir_options = panvk_get_nir_options,
   .get_spirv_options = panvk_get_spirv_options,
   .preprocess_nir = panvk_preprocess_nir,
   .hash_state = panvk_hash_state,
   .compile = panvk_compile_shaders,
   .deserialize = panvk_deserialize_shader,
   .cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
   .cmd_bind_shaders = panvk_cmd_bind_shaders,
};
#endif

static void
panvk_internal_shader_destroy(struct vk_device *vk_dev,
                              struct vk_shader *vk_shader,
                              const VkAllocationCallbacks *pAllocator)
{
   struct panvk_device *dev = to_panvk_device(vk_dev);
   struct panvk_internal_shader *shader =
      container_of(vk_shader, struct panvk_internal_shader, vk);

   panvk_pool_free_mem(&shader->code_mem);

#if PAN_ARCH < 9
   panvk_pool_free_mem(&shader->rsd);
#else
   panvk_pool_free_mem(&shader->spd);
#endif

   vk_shader_free(&dev->vk, pAllocator, &shader->vk);
}

static const struct vk_shader_ops panvk_internal_shader_ops = {
   .destroy = panvk_internal_shader_destroy,
};

VkResult
panvk_per_arch(create_internal_shader)(
   struct panvk_device *dev, nir_shader *nir,
   struct pan_compile_inputs *compiler_inputs,
   struct panvk_internal_shader **shader_out)
{
   struct panvk_internal_shader *shader =
      vk_shader_zalloc(&dev->vk, &panvk_internal_shader_ops, nir->info.stage,
                       NULL, sizeof(*shader));
   if (shader == NULL)
      return panvk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result;
   struct util_dynarray binary;

   panvk_per_arch(compiler_lock)();

   pan_postprocess_nir(nir, compiler_inputs, &shader->info);

   util_dynarray_init(&binary, nir);
   pan_shader_compile(nir, compiler_inputs, &binary, &shader->info);

   panvk_per_arch(compiler_unlock)();

   unsigned bin_size = util_dynarray_num_elements(&binary, uint8_t);
   if (bin_size) {
      shader->code_mem = panvk_pool_upload_aligned(&dev->mempools.exec,
                                                   binary.data, bin_size, 128);
      if (!panvk_priv_mem_check_alloc(shader->code_mem)) {
         result = panvk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         goto err_free_shader;
      }
   }

   *shader_out = shader;
   return VK_SUCCESS;

err_free_shader:
   vk_shader_free(&dev->vk, NULL, &shader->vk);
   return result;
}

VkResult panvk_per_arch(create_shader)(
   struct panvk_device *dev, nir_shader *nir, struct panvk_shader **shader_out)
{
   const struct vk_pipeline_robustness_state rs = {
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
   };

   struct vk_shader_compile_info info = {
      .stage = nir->info.stage,
      .nir = nir,
      .robustness = &rs,
   };

   panvk_preprocess_nir(dev->vk.physical, nir, &rs);

   struct vk_shader *vk_shader;
   VkResult result = panvk_compile_shader(dev, &info, NULL, NULL, 0, NULL,
                                          NULL, &vk_shader);
   if (result != VK_SUCCESS)
      return result;

   *shader_out = container_of(vk_shader, struct panvk_shader, vk);

   return VK_SUCCESS;
}

#if PAN_ARCH >= 15
VkResult
panvk_per_arch(rt_compile_nir)(
   struct panvk_device *dev, nir_shader *nir, struct vk_pipeline_layout *layout,
   const struct vk_pipeline_robustness_state *robustness,
   VkPipelineCreateFlags2KHR flags, const VkAllocationCallbacks *alloc,
   struct panvk_shader **shader_out)
{
   *shader_out = NULL;
   struct vk_shader_compile_info info = {
      .stage = MESA_SHADER_COMPUTE,
      .nir = nir,
      .robustness = robustness,
      .set_layout_count = layout->set_count,
      .set_layouts = layout->set_layouts,
      .push_constant_range_count = layout->push_range_count,
      .push_constant_ranges = layout->push_ranges,
   };
   if (flags & VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR)
      info.flags |= VK_SHADER_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_MESA;


   panvk_per_arch(compiler_lock)();
   panvk_preprocess_nir(dev->vk.physical, nir, robustness);
   struct vk_shader *shader;
   VkResult result = panvk_compile_shader(dev, &info, NULL, NULL, 0, NULL,
                                          alloc, &shader);
   panvk_per_arch(compiler_unlock)();
   ralloc_free(nir);
   if (result == VK_SUCCESS)
      *shader_out = container_of(shader, struct panvk_shader, vk);
   return result;
}
#endif

#else
static const struct vk_shader_ops panvk_shader_ops = {
   .destroy = panvk_shader_destroy,
};

const struct vk_device_shader_ops panvk_per_arch(offline_shader_ops) = {
   .get_nir_options = panvk_get_nir_options,
   .get_spirv_options = panvk_get_spirv_options,
   .preprocess_nir = panvk_preprocess_nir,
};

unsigned
panvk_per_arch(offline_shader_variants)(const struct vk_shader *vk_shader,
                                        struct kraidoc_variant *variants,
                                        unsigned max)
{
   const struct panvk_shader *shader =
      container_of(vk_shader, const struct panvk_shader, vk);
   unsigned count = 0;

   panvk_shader_foreach_variant_const(shader, variant) {
      if (!variant->bin_size || count == max)
         continue;

      const struct panvk_shader_variant *pilot = variant->preamble;
      variants[count++] = (struct kraidoc_variant){
         .index = variant - shader->variants,
         .main = {&variant->info, variant->bin_ptr, variant->bin_size},
         .pilot = pilot ? (struct kraidoc_program){&pilot->info, pilot->bin_ptr,
                                                   pilot->bin_size}
                        : (struct kraidoc_program){0},
         .nir_str = variant->nir_str,
      };
   }

   return count;
}
#endif
