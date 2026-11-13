/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "nvk_buffer.h"
#include "nvk_cmd_buffer.h"
#include "nvk_descriptor_set.h"
#include "nvk_device.h"
#include "nvk_entrypoints.h"
#include "nvk_image.h"
#include "nvk_physical_device.h"

#include "vk_format.h"
#include "vk_meta_private.h"
#include "vk_pipeline.h"

#include "nir_builder.h"

#include "nv_push_cl9097.h"
#include "nv_push_cl90c0.h"
#include "nv_push_clb197.h"

static VkResult
nvk_cmd_bind_map_buffer(struct vk_command_buffer *vk_cmd,
                        struct vk_meta_device *meta,
                        VkBuffer _buffer, void **map_out)
{
   struct nvk_cmd_buffer *cmd =
      container_of(vk_cmd, struct nvk_cmd_buffer, vk);
   VK_FROM_HANDLE(nvk_buffer, buffer, _buffer);
   VkResult result;

   uint64_t addr;
   assert(buffer->vk.size < UINT_MAX);
   result = nvk_cmd_buffer_upload_alloc(cmd, buffer->vk.size, 16,
                                        &addr, map_out);
   if (unlikely(result != VK_SUCCESS))
      return result;

   assert(buffer->vk.device_address == 0);
   buffer->vk.device_address = addr;

   return VK_SUCCESS;
}

VkResult
nvk_device_init_meta(struct nvk_device *dev)
{
   const struct nvk_physical_device *pdev = nvk_device_physical(dev);

   VkResult result = vk_meta_device_init(&dev->vk, &dev->meta);
   if (result != VK_SUCCESS)
      return result;

   dev->meta.use_gs_for_layer = pdev->info.cls_eng3d < MAXWELL_B;
   dev->meta.use_rect_list_pipeline = true;
   dev->meta.cmd_bind_map_buffer = nvk_cmd_bind_map_buffer;
   dev->meta.max_bind_map_buffer_size_B = 64 * 1024; /* TODO */

   for (unsigned i = 0; i < VK_META_BUFFER_CHUNK_SIZE_COUNT; ++i) {
      dev->meta.buffer_access.optimal_wg_size[i] = 128;
   }

   return VK_SUCCESS;
}

void
nvk_device_finish_meta(struct nvk_device *dev)
{
   vk_meta_device_finish(&dev->vk, &dev->meta);
}

struct nvk_meta_save_gfx {
   struct vk_vertex_input_state _dynamic_vi;
   struct vk_sample_locations_state _dynamic_sl;
   struct vk_dynamic_graphics_state dynamic;
   struct nvk_shader *shaders[MESA_SHADER_MESH + 1];
   VkDeviceAddressRangeKHR vb0;
   struct nvk_descriptor_set_binding desc0;
   struct nvk_buffer_address desc0_set_addr;
   struct nvk_push_descriptor_set push_desc0;
   uint8_t set_dynamic_buffer_start[NVK_MAX_SETS];
   uint8_t push[NVK_MAX_PUSH_SIZE];
};

struct nvk_meta_save_compute {
   struct nvk_compute_state state;
};

union nvk_meta_save_generic {
   struct nvk_meta_save_gfx gfx;
   struct nvk_meta_save_compute compute;
};

static void
nvk_meta_begin_gfx(struct nvk_cmd_buffer *cmd,
                   struct nvk_meta_save_gfx *save)
{
   const struct nvk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   struct nv_push *p = nvk_cmd_buffer_push(cmd, 10);
   P_IMMD(p, NV9097, SET_MME_SHADOW_RAM_CONTROL, MODE_METHOD_PASSTHROUGH);
   P_IMMD(p, NV9097, SET_RENDER_ENABLE_OVERRIDE, MODE_ALWAYS_RENDER);
   P_IMMD(p, NV9097, SET_STATISTICS_COUNTER, {
      .da_vertices_generated_enable = false,
      .da_primitives_generated_enable = false,
      .vs_invocations_enable = false,
      .gs_invocations_enable = false,
      .gs_primitives_generated_enable = false,
      .streaming_primitives_succeeded_enable = false,
      .streaming_primitives_needed_enable = false,
      .clipper_invocations_enable = false,
      .clipper_primitives_generated_enable = false,
      .ps_invocations_enable = false,
      .ti_invocations_enable = false,
      .ts_invocations_enable = false,
      .ts_primitives_generated_enable = false,
      .total_streaming_primitives_needed_succeeded_enable = false,
      .vtg_primitives_out_enable = false,
   });
   P_IMMD(p, NV9097, SET_ZPASS_PIXEL_COUNT, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_MME_SHADOW_RAM_CONTROL, MODE_METHOD_TRACK_WITH_FILTER);

   save->dynamic = cmd->vk.dynamic_graphics_state;
   save->_dynamic_vi = cmd->state.gfx._dynamic_vi;
   save->_dynamic_sl = cmd->state.gfx._dynamic_sl;

   STATIC_ASSERT(sizeof(cmd->state.gfx.shaders) == sizeof(save->shaders));
   memcpy(save->shaders, cmd->state.gfx.shaders, sizeof(save->shaders));

   save->vb0 = cmd->state.gfx.vb0;

   save->desc0 = desc->sets[0];
   nvk_descriptor_state_get_root(desc, sets[0], &save->desc0_set_addr);
   if (desc->sets[0].push != NULL)
      save->push_desc0 = *desc->sets[0].push;

   nvk_descriptor_state_get_root_array(desc, set_dynamic_buffer_start,
                                       0, NVK_MAX_SETS,
                                       save->set_dynamic_buffer_start);
   nvk_descriptor_state_get_root_array(desc, push, 0, NVK_MAX_PUSH_SIZE,
                                       save->push);
}

static void
nvk_meta_end_gfx(struct nvk_cmd_buffer *cmd,
                 struct nvk_meta_save_gfx *save)
{
   struct nvk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   switch (save->desc0.type) {
   case NVK_DESCRIPTOR_SET_TYPE_NONE:
      desc->sets[0].type = NVK_DESCRIPTOR_SET_TYPE_NONE;
      break;

   case NVK_DESCRIPTOR_SET_TYPE_SET: {
      desc->sets[0].type = NVK_DESCRIPTOR_SET_TYPE_SET;
      desc->sets[0].set = save->desc0.set;
      struct nvk_buffer_address addr = nvk_descriptor_set_addr(save->desc0.set);
      nvk_descriptor_state_set_root(cmd, desc, sets[0], addr);
      break;
   }

   case NVK_DESCRIPTOR_SET_TYPE_PUSH:
      desc->sets[0].type = NVK_DESCRIPTOR_SET_TYPE_PUSH;
      desc->sets[0].set = NULL;
      *desc->sets[0].push = save->push_desc0;
      desc->push_dirty |= BITFIELD_BIT(0);
      break;

   case NVK_DESCRIPTOR_SET_TYPE_BUFFER:
      desc->sets[0].type = NVK_DESCRIPTOR_SET_TYPE_BUFFER;
      desc->sets[0].set = NULL;
      nvk_descriptor_state_set_root(cmd, desc, sets[0], save->desc0_set_addr);
      break;

   default:
      UNREACHABLE("Unknown descriptor set type");
   }
   nvk_cmd_dirty_cbufs_for_descriptors(cmd, ~0, 0, 1);

   /* Restore set_dynaic_buffer_start because meta binding set 0 can disturb
    * all dynamic buffers starts for all sets.
    */
   nvk_descriptor_state_set_root_array(cmd, desc, set_dynamic_buffer_start,
                                       0, NVK_MAX_SETS,
                                       save->set_dynamic_buffer_start);

   /* Restore the dynamic state */
   assert(save->dynamic.vi == &cmd->state.gfx._dynamic_vi);
   assert(save->dynamic.ms.sample_locations == &cmd->state.gfx._dynamic_sl);
   cmd->vk.dynamic_graphics_state = save->dynamic;
   cmd->state.gfx._dynamic_vi = save->_dynamic_vi;
   cmd->state.gfx._dynamic_sl = save->_dynamic_sl;
   memcpy(cmd->vk.dynamic_graphics_state.dirty,
          cmd->vk.dynamic_graphics_state.set,
          sizeof(cmd->vk.dynamic_graphics_state.set));

   for (uint32_t stage = 0; stage < ARRAY_SIZE(save->shaders); stage++) {
      if (stage == MESA_SHADER_COMPUTE)
         continue;

      nvk_cmd_bind_graphics_shader(cmd, stage, save->shaders[stage]);
   }

   nvk_cmd_bind_vertex_buffer(cmd, 0, save->vb0);

   nvk_descriptor_state_set_root_array(cmd, desc, push, 0, sizeof(save->push),
                                       save->push);

   /* Replay the previous state from shadow RAM */
   struct nv_push *p = nvk_cmd_buffer_push(cmd, 10);
   P_IMMD(p, NV9097, SET_MME_SHADOW_RAM_CONTROL, MODE_METHOD_REPLAY);
   P_IMMD(p, NV9097, SET_ZPASS_PIXEL_COUNT, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_STATISTICS_COUNTER, {});
   P_IMMD(p, NV9097, SET_RENDER_ENABLE_OVERRIDE, MODE_USE_RENDER_ENABLE);
   P_IMMD(p, NV9097, SET_MME_SHADOW_RAM_CONTROL, MODE_METHOD_TRACK_WITH_FILTER);
}

static void
nvk_meta_begin_compute(struct nvk_cmd_buffer *cmd,
                       struct nvk_meta_save_compute *save)
{
   assert(cmd->state.cs.descriptors.flush_root == NULL);
   save->state = cmd->state.cs;
   nvk_cmd_invalidate_compute_state(cmd);

   struct nv_push *p = nvk_cmd_buffer_push(cmd, 1);
   P_IMMD_WORD(p, NV90C0, SET_RENDER_ENABLE_OVERRIDE, MODE_ALWAYS_RENDER);
}

static void
nvk_meta_end_compute(struct nvk_cmd_buffer *cmd,
                     struct nvk_meta_save_compute *save)
{
   struct nv_push *p = nvk_cmd_buffer_push(cmd, 1);
   P_IMMD_WORD(p, NV90C0, SET_RENDER_ENABLE_OVERRIDE, MODE_USE_RENDER_ENABLE);

   nvk_descriptor_state_fini(cmd, &cmd->state.cs.descriptors);
   cmd->state.cs = save->state;
}

static void
nvk_meta_begin_generic(struct nvk_cmd_buffer *cmd,
                       union nvk_meta_save_generic *save,
                       VkPipelineBindPoint engine)
{
   if (engine == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      nvk_meta_begin_gfx(cmd, &save->gfx);
   } else {
      assert(engine == VK_PIPELINE_BIND_POINT_COMPUTE);
      nvk_meta_begin_compute(cmd, &save->compute);
   }
}

static void
nvk_meta_end_generic(struct nvk_cmd_buffer *cmd,
                     union nvk_meta_save_generic *save,
                     VkPipelineBindPoint engine)
{
   if (engine == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      nvk_meta_end_gfx(cmd, &save->gfx);
   } else {
      assert(engine == VK_PIPELINE_BIND_POINT_COMPUTE);
      nvk_meta_end_compute(cmd, &save->compute);
   }
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdBlitImage2(VkCommandBuffer commandBuffer,
                  const VkBlitImageInfo2 *pBlitImageInfo)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, commandBuffer);
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);

   struct nvk_meta_save_gfx save;
   nvk_meta_begin_gfx(cmd, &save);

   vk_meta_blit_image2(&cmd->vk, &dev->meta, pBlitImageInfo);

   nvk_meta_end_gfx(cmd, &save);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdResolveImage2(VkCommandBuffer commandBuffer,
                     const VkResolveImageInfo2 *pResolveImageInfo)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, commandBuffer);
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);

   struct nvk_meta_save_gfx save;
   nvk_meta_begin_gfx(cmd, &save);

   vk_meta_resolve_image2(&cmd->vk, &dev->meta, pResolveImageInfo);

   nvk_meta_end_gfx(cmd, &save);
}

void
nvk_meta_resolve_rendering(struct nvk_cmd_buffer *cmd,
                           const VkRenderingInfo *pRenderingInfo)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);

   struct nvk_meta_save_gfx save;
   nvk_meta_begin_gfx(cmd, &save);

   vk_meta_resolve_rendering(&cmd->vk, &dev->meta, pRenderingInfo);

   nvk_meta_end_gfx(cmd, &save);
}

static bool
nvk_meta_image_copy_gfx_supported(struct nvk_image *img)
{
   if (vk_format_is_depth_or_stencil(img->vk.format))
      return false;
   if (vk_format_is_compressed(img->vk.format))
      return false;
   if (vk_format_get_ycbcr_info(img->vk.format))
      return false;

   assert(img->plane_count == 1);
   const struct nvk_image_plane *plane = &img->planes[0];
   const struct nil_image *nil_image = &plane->nil;

   for (int l = 0; l < nil_image->num_levels; l++) {
      const struct nil_image_level *level = &nil_image->levels[l];
      if (level->tiling.z_log2 != 0)
         return false;

      if (level->tiling.gob_type == NIL_GOB_TYPE_LINEAR &&
          !nvk_image_plane_aligned_for_linear_attachment(plane, level))
         return false;
   }
   return true;
}

static bool
nvk_meta_image_copy_compute_supported(struct nvk_image *img)
{
   if (vk_format_is_depth_or_stencil(img->vk.format))
      return false;
   if (vk_format_is_compressed(img->vk.format))
      return false;
   if (vk_format_get_ycbcr_info(img->vk.format))
      return false;

   return true;
}

static bool
nvk_meta_copy_requires_compute(struct nvk_cmd_buffer *cmd)
{
   const struct nvk_physical_device *pdev =
      nvk_device_physical(nvk_cmd_buffer_device(cmd));
   const enum nvkmd_engines engines = nvk_queue_engines_from_queue_flags(
      pdev, nvk_cmd_buffer_queue_flags(cmd));

   return pdev->nvkmd->kmd_info.has_compute_queue &&
          (engines & NVKMD_ENGINE_COMPUTE) && !(engines & NVKMD_ENGINE_3D);
}

static void
nvk_meta_copy_aspect_barrier(struct nvk_cmd_buffer *cmd)
{
   const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
   };
   const VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
   };
   nvk_CmdPipelineBarrier2(nvk_cmd_buffer_to_handle(cmd), &dep);
}

static inline
uint32_t extent_size(VkExtent3D e) {
   return e.width * e.height * e.depth;
}

static struct vk_meta_copy_image_properties
nvk_meta_copy_get_image_properties(struct nvk_image *img,
                                   bool is_destination)
{
   struct vk_meta_copy_image_properties props = {};

   const struct vk_format_ycbcr_info *ycbcr =
      vk_format_get_ycbcr_info(img->vk.format);
   if (vk_format_is_depth_or_stencil(img->vk.format)) {
      switch (img->vk.format) {
      case VK_FORMAT_S8_UINT:
         break;
      case VK_FORMAT_D16_UNORM:
      case VK_FORMAT_D16_UNORM_S8_UINT:
         props.depth.view_format = VK_FORMAT_R16_UINT;
         props.depth.component_mask = BITFIELD_BIT(0);
         break;
      case VK_FORMAT_X8_D24_UNORM_PACK32:
      case VK_FORMAT_D24_UNORM_S8_UINT:
         props.depth.view_format = VK_FORMAT_R8G8B8A8_UINT;
         props.depth.component_mask = BITFIELD_MASK(3);
         break;
      case VK_FORMAT_D32_SFLOAT:
      case VK_FORMAT_D32_SFLOAT_S8_UINT:
         props.depth.view_format = VK_FORMAT_R32_UINT;
         props.depth.component_mask = BITFIELD_BIT(0);
         break;
      default:
         UNREACHABLE("Invalid depth/stencil format");
      }

      if (img->vk.aspects & VK_IMAGE_ASPECT_STENCIL_BIT) {
         props.stencil.view_format = VK_FORMAT_R8_UINT;
         props.stencil.component_mask = BITFIELD_BIT(0);
         if (!img->separate_zs &&
             (img->vk.aspects & VK_IMAGE_ASPECT_DEPTH_BIT)) {
            if (img->vk.format == VK_FORMAT_D24_UNORM_S8_UINT) {
               props.stencil.view_format = VK_FORMAT_R8G8B8A8_UINT;
               props.stencil.component_mask = BITFIELD_BIT(3);
            } else {
               assert(img->vk.format == VK_FORMAT_D32_SFLOAT_S8_UINT);
               props.depth.view_format = VK_FORMAT_R32G32_UINT;
               props.stencil.view_format = VK_FORMAT_R32G32_UINT;
               props.stencil.component_mask = BITFIELD_BIT(1);
            }
         }
      }
   } else if (ycbcr) {
      for (uint32_t p = 0; p < ycbcr->n_planes; p++) {
         props.plane[p].view_format = vk_meta_get_uint_format_for_blk_size(
            vk_format_get_blocksize(ycbcr->planes[p].format));
      }
   } else {
      props.color.view_format = vk_meta_get_uint_format_for_blk_size(
         vk_format_get_blocksize(img->vk.format));
   }

   const struct nvk_image_plane *plane = &img->planes[0];
   const struct nil_image *nil_image = &plane->nil;
   const unsigned blk_sz = util_format_get_blocksize(nil_image->format.p_format);

   const struct nil_image_level *level =
      &nil_image->levels[0];

   if (level->tiling.gob_type == NIL_GOB_TYPE_LINEAR) {
      props.tile_size.width = 128;
      props.tile_size.height = 1;
      props.tile_size.depth = 1;
   } else {
      struct nil_Extent4D_Bytes gob_extent =
         nil_gob_type_extent_B(level->tiling.gob_type);
      props.tile_size.width = gob_extent.width / blk_sz;
      props.tile_size.height = gob_extent.height;
      props.tile_size.depth = gob_extent.depth;

      const uint32_t MIN_SIZE = 128;

      if (extent_size(props.tile_size) < MIN_SIZE)
         props.tile_size.height *= MIN2(1 << level->tiling.y_log2,
                                        MIN_SIZE / extent_size(props.tile_size));

      if (extent_size(props.tile_size) < MIN_SIZE)
         props.tile_size.depth *= MIN2(1 << level->tiling.z_log2,
                                       MIN_SIZE / extent_size(props.tile_size));

      if (extent_size(props.tile_size) < MIN_SIZE)
         props.tile_size.width *= MIN_SIZE / extent_size(props.tile_size);
   }
   ASSERTED uint32_t wg_size = extent_size(props.tile_size);
   /* We want the workgroup size to be in this range for occupancy */
   assert(wg_size >= 128 && wg_size <= 512 &&
          util_is_power_of_two_nonzero(wg_size));

   return props;
}

struct nvk_meta_raw_depth_key {
   uint32_t type;
   uint8_t src_dim;
   uint8_t dst_dim;
   bool preserve_stencil;
   nil_sample_layout sample_layout;
   uint32_t unorm_depth;
};

struct nvk_meta_raw_depth_push {
   uint64_t src_addr;
   uint64_t dst_addr;
   uint64_t image_stride;
   uint32_t row_stride;
   uint32_t src_offset[3];
   uint32_t dst_offset[3];
   uint32_t extent[3];
};

static bool
nvk_meta_copy_raw_depth(struct nvk_cmd_buffer *cmd, const struct nvk_image *image)
{
   return nvk_meta_copy_requires_compute(cmd) &&
          (image->vk.format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
           image->vk.format == VK_FORMAT_D24_UNORM_S8_UINT);
}

static nir_def *
nvk_meta_raw_depth_load_push(nir_builder *b, unsigned components,
                              unsigned bits, unsigned offset)
{
   return nir_load_push_constant(b, components, bits, nir_imm_int(b, offset));
}

static nir_deref_instr *
nvk_meta_raw_depth_image(nir_builder *b, unsigned binding, unsigned dim,
                           bool unorm_depth)
{
   const enum glsl_sampler_dim sampler_dim =
      dim == 1 ? GLSL_SAMPLER_DIM_1D :
      dim == 2 ? GLSL_SAMPLER_DIM_2D : GLSL_SAMPLER_DIM_3D;
   const struct glsl_type *type =
      glsl_image_type(sampler_dim, dim != 3,
                      unorm_depth ? GLSL_TYPE_FLOAT : GLSL_TYPE_UINT);
   nir_variable *var =
      nir_variable_create(b->shader, nir_var_uniform, type, "raw_depth");
   var->data.descriptor_set = 0;
   var->data.binding = binding;
   var->data.image.format = unorm_depth ? PIPE_FORMAT_R32_FLOAT :
                                         PIPE_FORMAT_R32_UINT;
   return nir_build_deref_var(b, var);
}

static nir_def *
nvk_meta_raw_depth_coords(nir_builder *b, nir_def *id, unsigned offset,
                           unsigned dim, nil_sample_layout layout,
                           unsigned sample)
{
   nir_def *origin = nvk_meta_raw_depth_load_push(b, 3, 32, offset);
   nir_def *pos = nir_iadd(b, id, origin);
   const struct nil_Extent4D_Samples grid = nil_px_extent_sa(layout);
   const struct nil_sample_offset loc = nil_sample_offset(layout, sample);
   nir_def *x = nir_imul_imm(b, nir_channel(b, pos, 0), grid.width);
   x = nir_iadd_imm(b, x, loc.x);
   nir_def *y = nir_imul_imm(b, nir_channel(b, pos, 1), grid.height);
   y = nir_iadd_imm(b, y, loc.y);
   nir_def *z = nir_channel(b, pos, 2);
   return dim == 1 ? nir_vec4(b, x, z, nir_imm_int(b, 0), nir_imm_int(b, 0)) :
                     nir_vec4(b, x, y, z, nir_imm_int(b, 0));
}

static nir_def *
nvk_meta_raw_depth_address(nir_builder *b, nir_def *id, unsigned offset)
{
   nir_def *address = nvk_meta_raw_depth_load_push(b, 1, 64, offset);
   nir_def *row_stride = nvk_meta_raw_depth_load_push(
      b, 1, 32, offsetof(struct nvk_meta_raw_depth_push, row_stride));
   nir_def *image_stride = nvk_meta_raw_depth_load_push(
      b, 1, 64, offsetof(struct nvk_meta_raw_depth_push, image_stride));
   nir_def *x = nir_imul_imm(b, nir_u2u64(b, nir_channel(b, id, 0)), 4);
   nir_def *y = nir_imul(b, nir_u2u64(b, nir_channel(b, id, 1)),
                           nir_u2u64(b, row_stride));
   nir_def *z = nir_imul(b, nir_u2u64(b, nir_channel(b, id, 2)), image_stride);
   return nir_iadd(b, address, nir_iadd(b, x, nir_iadd(b, y, z)));
}

static nir_shader *
nvk_meta_raw_depth_shader(const struct nvk_meta_raw_depth_key *key)
{
   nir_builder builder = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL, "nvk-meta-copy-raw-depth");
   nir_builder *b = &builder;
   b->fp_math_ctrl = nir_fp_exact;
   b->shader->info.workgroup_size[0] = 32;
   b->shader->info.workgroup_size[1] = 4;
   b->shader->info.workgroup_size[2] = 1;

   nir_deref_instr *src = key->src_dim ?
      nvk_meta_raw_depth_image(b, 0, key->src_dim, key->unorm_depth) : NULL;
   nir_deref_instr *dst = key->dst_dim ?
      nvk_meta_raw_depth_image(b, 1, key->dst_dim, false) : NULL;
   nir_def *id = nir_load_global_invocation_id(b, 32);
   nir_def *extent = nvk_meta_raw_depth_load_push(
      b, 3, 32, offsetof(struct nvk_meta_raw_depth_push, extent));
   nir_push_if(b, nir_ball(b, nir_ult(b, id, extent)));

   const enum glsl_sampler_dim src_dim =
      key->src_dim == 1 ? GLSL_SAMPLER_DIM_1D :
      key->src_dim == 2 ? GLSL_SAMPLER_DIM_2D : GLSL_SAMPLER_DIM_3D;
   const enum glsl_sampler_dim dst_dim =
      key->dst_dim == 1 ? GLSL_SAMPLER_DIM_1D :
      key->dst_dim == 2 ? GLSL_SAMPLER_DIM_2D : GLSL_SAMPLER_DIM_3D;
   for (unsigned sample = 0;
        sample < nil_sample_layout_samples(key->sample_layout); sample++) {
      nir_def *value;
      if (src) {
         nir_def *coords = nvk_meta_raw_depth_coords(
            b, id, offsetof(struct nvk_meta_raw_depth_push, src_offset),
            key->src_dim, key->sample_layout, sample);
         if (key->unorm_depth) {
            value = nir_image_deref_load(
               b, 1, 32, &src->def, coords, nir_imm_int(b, 0), nir_imm_int(b, 0),
               .image_dim = src_dim, .image_array = key->src_dim != 3,
               .format = PIPE_FORMAT_R32_FLOAT, .dest_type = nir_type_float32);
            value = nir_fround_even(b, nir_fmul_imm(b, value, 16777215.0));
            value = nir_f2u32(b, value);
         } else {
            value = nir_image_deref_load_raw_nv(
               b, 1, &src->def, coords, nir_imm_int(b, 0), nir_imm_int(b, 0),
               .image_dim = src_dim, .image_array = key->src_dim != 3,
               .format = PIPE_FORMAT_R32_UINT, .dest_type = nir_type_uint32);
         }
      } else {
         nir_def *address = nvk_meta_raw_depth_address(
            b, id, offsetof(struct nvk_meta_raw_depth_push, src_addr));
         value = nir_load_global(b, 1, 32, address, .align_mul = 4);
      }

      if (dst) {
         nir_def *coords = nvk_meta_raw_depth_coords(
            b, id, offsetof(struct nvk_meta_raw_depth_push, dst_offset),
            key->dst_dim, key->sample_layout, sample);
         if (key->unorm_depth || key->preserve_stencil) {
            for (unsigned byte = 0; byte < 3; byte++) {
               nir_def *byte_x = nir_imul_imm(b, nir_channel(b, coords, 0), 4);
               nir_def *byte_coords = nir_vector_insert_imm(
                  b, coords, nir_iadd_imm(b, byte_x, byte), 0);
               nir_def *byte_value = nir_ushr_imm(b, value, byte * 8);
               nir_image_deref_store_raw_nv(
                  b, &dst->def, byte_coords, nir_imm_int(b, 0), byte_value,
                  nir_imm_int(b, 0), .image_dim = dst_dim,
                  .image_array = key->dst_dim != 3,
                  .format = PIPE_FORMAT_R8_UINT, .src_type = nir_type_uint32);
            }
         } else {
            nir_image_deref_store_raw_nv(
               b, &dst->def, coords, nir_imm_int(b, 0), value,
               nir_imm_int(b, 0), .image_dim = dst_dim,
               .image_array = key->dst_dim != 3,
               .format = PIPE_FORMAT_R32_UINT, .src_type = nir_type_uint32);
         }
      } else {
         nir_def *address = nvk_meta_raw_depth_address(
            b, id, offsetof(struct nvk_meta_raw_depth_push, dst_addr));
         nir_store_global(b, value, address, .align_mul = 4);
      }
   }
   nir_pop_if(b, NULL);
   return b->shader;
}

static VkResult
nvk_meta_raw_depth_pipeline(struct nvk_device *dev,
                              const struct nvk_meta_raw_depth_key *key,
                              VkPipelineLayout *layout, VkPipeline *pipeline)
{
   const uint32_t layout_key = VK_META_OBJECT_KEY_DRIVER_OFFSET + 40;
   const VkDescriptorSetLayoutBinding bindings[] = {
      { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
      { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
   };
   const VkDescriptorSetLayoutCreateInfo set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
      .bindingCount = ARRAY_SIZE(bindings),
      .pBindings = bindings,
   };
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct nvk_meta_raw_depth_push),
   };
   VkResult result = vk_meta_get_pipeline_layout(
      &dev->vk, &dev->meta, &set_info, &push_range,
      &layout_key, sizeof(layout_key), layout);
   if (result != VK_SUCCESS)
      return result;

   *pipeline = vk_meta_lookup_pipeline(&dev->meta, key, sizeof(*key));
   if (*pipeline != VK_NULL_HANDLE)
      return VK_SUCCESS;

   const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
      .nir = nvk_meta_raw_depth_shader(key),
   };
   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = &nir_info,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .pName = "main",
      },
      .layout = *layout,
   };
   result = vk_meta_create_compute_pipeline(
      &dev->vk, &dev->meta, &pipeline_info, key, sizeof(*key), pipeline);
   ralloc_free(nir_info.nir);
   return result;
}

static VkResult
nvk_meta_raw_depth_view(struct nvk_cmd_buffer *cmd, struct nvk_image *image,
                          const VkImageSubresourceLayers *subres,
                          bool raw_store, VkImageView *view)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const VkImageViewUsageCreateInfo usage = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT,
   };
   const VkImageViewCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .pNext = &usage,
      .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
      .image = nvk_image_to_handle(image),
      .viewType = vk_image_sampled_view_type(&image->vk),
      .format = raw_store ? VK_FORMAT_R32_UINT : image->vk.format,
      .subresourceRange = {
         .aspectMask = raw_store ? VK_IMAGE_ASPECT_COLOR_BIT :
                                  VK_IMAGE_ASPECT_DEPTH_BIT,
         .baseMipLevel = subres->mipLevel,
         .levelCount = 1,
         .layerCount = image->vk.array_layers,
      },
   };
   return vk_meta_create_image_view(&cmd->vk, &dev->meta, &info, view);
}

static void
nvk_meta_raw_depth_dispatch(struct nvk_cmd_buffer *cmd,
                              struct nvk_image *src,
                              const VkImageSubresourceLayers *src_subres,
                              VkImageLayout src_layout, struct nvk_image *dst,
                              const VkImageSubresourceLayers *dst_subres,
                              VkImageLayout dst_layout,
                              const struct nvk_meta_raw_depth_push *push)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   struct nvk_image *image = src ? src : dst;
   const struct nvk_meta_raw_depth_key key = {
      .type = VK_META_OBJECT_KEY_DRIVER_OFFSET + 41,
      .src_dim = src ? src->vk.image_type + 1 : 0,
      .dst_dim = dst ? dst->vk.image_type + 1 : 0,
      .preserve_stencil = dst && !dst->separate_zs &&
                         (dst->vk.aspects & VK_IMAGE_ASPECT_STENCIL_BIT),
      .sample_layout = image->planes[0].nil.sample_layout,
      .unorm_depth = image->vk.format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
                     image->vk.format == VK_FORMAT_D24_UNORM_S8_UINT,
   };
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result = nvk_meta_raw_depth_pipeline(dev, &key, &layout, &pipeline);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   struct nvk_image *images[] = { src, dst };
   const VkImageSubresourceLayers *subres[] = { src_subres, dst_subres };
   const VkImageLayout layouts[] = { src_layout, dst_layout };
   VkDescriptorImageInfo image_info[2] = {};
   VkWriteDescriptorSet writes[2] = {};
   uint32_t write_count = 0;
   for (unsigned i = 0; i < 2; i++) {
      if (!images[i])
         continue;
      result = nvk_meta_raw_depth_view(cmd, images[i], subres[i], i == 1,
                                       &image_info[i].imageView);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return;
      }
      image_info[i].imageLayout = layouts[i];
      writes[write_count++] = (VkWriteDescriptorSet) {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstBinding = i,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .pImageInfo = &image_info[i],
      };
   }

   const VkCommandBuffer command_buffer = nvk_cmd_buffer_to_handle(cmd);
   disp->CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   disp->CmdPushDescriptorSetKHR(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 layout, 0, write_count, writes);
   disp->CmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(*push), push);
   nvk_CmdDispatchBase(command_buffer, 0, 0, 0,
                       DIV_ROUND_UP(push->extent[0], 32),
                       DIV_ROUND_UP(push->extent[1], 4), push->extent[2]);
}

static void
nvk_meta_raw_depth_set_offset(uint32_t out[3], const struct nvk_image *image,
                               const VkImageSubresourceLayers *subres,
                               VkOffset3D offset)
{
   out[0] = offset.x;
   out[1] = offset.y;
   out[2] = image->vk.image_type == VK_IMAGE_TYPE_3D ?
               offset.z : subres->baseArrayLayer;
}

static void
nvk_meta_raw_depth_memory_region(struct nvk_cmd_buffer *cmd,
                                  struct nvk_image *image,
                                  const VkDeviceMemoryImageCopyKHR *region,
                                  bool to_image)
{
   const struct vk_image_buffer_layout buffer_layout =
      vk_image_memory_copy_layout(&image->vk, region);
   const VkExtent3D extent =
      vk_image_sanitize_extent(&image->vk, region->imageExtent);
   struct nvk_meta_raw_depth_push push = {
      .src_addr = region->addressRange.address,
      .dst_addr = region->addressRange.address,
      .image_stride = buffer_layout.image_stride_B,
      .row_stride = buffer_layout.row_stride_B,
      .extent = {
         extent.width, extent.height,
         image->vk.image_type == VK_IMAGE_TYPE_3D ? extent.depth :
            vk_image_subresource_layer_count(&image->vk,
                                             &region->imageSubresource),
      },
   };
   nvk_meta_raw_depth_set_offset(to_image ? push.dst_offset : push.src_offset,
      image, &region->imageSubresource, region->imageOffset);
   nvk_meta_raw_depth_dispatch(cmd,
      to_image ? NULL : image, &region->imageSubresource, region->imageLayout,
      to_image ? image : NULL, &region->imageSubresource, region->imageLayout,
      &push);
}

static void
nvk_cmd_copy_image_to_memory_meta(struct nvk_cmd_buffer *cmd,
                                  const VkCopyDeviceMemoryImageInfoKHR *pCopyMemoryInfo)
{
   VK_FROM_HANDLE(nvk_image, src, pCopyMemoryInfo->image);
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);

   struct vk_meta_copy_image_properties src_img_props =
      nvk_meta_copy_get_image_properties(src, true);

   struct nvk_meta_save_compute save;
   nvk_meta_begin_compute(cmd, &save);
   if (nvk_meta_copy_raw_depth(cmd, src)) {
      VkCopyDeviceMemoryImageInfoKHR info = *pCopyMemoryInfo;
      info.regionCount = 1;
      for (uint32_t i = 0; i < pCopyMemoryInfo->regionCount; i++) {
         info.pRegions = &pCopyMemoryInfo->pRegions[i];
         if (info.pRegions->imageSubresource.aspectMask ==
             VK_IMAGE_ASPECT_DEPTH_BIT) {
            nvk_meta_raw_depth_memory_region(cmd, src, info.pRegions, false);
         } else {
            vk_meta_copy_image_to_memory(&cmd->vk, &dev->meta, &info,
                                         &src_img_props);
         }
      }
   } else {
      vk_meta_copy_image_to_memory(&cmd->vk, &dev->meta, pCopyMemoryInfo,
                                   &src_img_props);
   }
   nvk_meta_end_compute(cmd, &save);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdCopyImageToMemoryKHR(VkCommandBuffer commandBuffer,
                            const VkCopyDeviceMemoryImageInfoKHR *pCopyMemoryInfo)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(nvk_image, src, pCopyMemoryInfo->image);

   VkQueueFlags queue_flags = nvk_cmd_buffer_queue_flags(cmd);
   if (nvk_meta_copy_requires_compute(cmd) ||
       ((queue_flags & VK_QUEUE_COMPUTE_BIT) &&
        nvk_meta_image_copy_compute_supported(src))) {
      nvk_cmd_copy_image_to_memory_meta(cmd, pCopyMemoryInfo);
   } else {
      nvk_cmd_copy_image_to_memory_ce(cmd, pCopyMemoryInfo);
   }
}

static void
nvk_cmd_copy_memory_to_image_meta(struct nvk_cmd_buffer *cmd,
                                  const VkCopyDeviceMemoryImageInfoKHR *pCopyMemoryInfo,
                                  VkPipelineBindPoint engine)
{
   VK_FROM_HANDLE(nvk_image, dst, pCopyMemoryInfo->image);
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);

   struct vk_meta_copy_image_properties dst_img_props =
      nvk_meta_copy_get_image_properties(dst, true);

   union nvk_meta_save_generic save;
   nvk_meta_begin_generic(cmd, &save, engine);
   const bool interleaved_zs = !dst->separate_zs &&
      dst->vk.aspects == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
   if (nvk_meta_copy_raw_depth(cmd, dst)) {
      VkCopyDeviceMemoryImageInfoKHR info = *pCopyMemoryInfo;
      info.regionCount = 1;
      for (uint32_t i = 0; i < pCopyMemoryInfo->regionCount; i++) {
         info.pRegions = &pCopyMemoryInfo->pRegions[i];
         if (interleaved_zs && i > 0 &&
             info.pRegions->imageSubresource.aspectMask !=
             pCopyMemoryInfo->pRegions[i - 1].imageSubresource.aspectMask)
            nvk_meta_copy_aspect_barrier(cmd);
         if (info.pRegions->imageSubresource.aspectMask ==
             VK_IMAGE_ASPECT_DEPTH_BIT) {
            nvk_meta_raw_depth_memory_region(cmd, dst, info.pRegions, true);
         } else {
            vk_meta_copy_memory_to_image(&cmd->vk, &dev->meta, &info,
                                         &dst_img_props, engine);
         }
      }
   } else if (interleaved_zs && engine == VK_PIPELINE_BIND_POINT_COMPUTE) {
      VkCopyDeviceMemoryImageInfoKHR info = *pCopyMemoryInfo;
      info.regionCount = 1;
      for (uint32_t i = 0; i < pCopyMemoryInfo->regionCount; i++) {
         info.pRegions = &pCopyMemoryInfo->pRegions[i];
         if (i > 0 && info.pRegions->imageSubresource.aspectMask !=
                      pCopyMemoryInfo->pRegions[i - 1].imageSubresource.aspectMask)
            nvk_meta_copy_aspect_barrier(cmd);
         vk_meta_copy_memory_to_image(&cmd->vk, &dev->meta, &info,
                                      &dst_img_props, engine);
      }
   } else {
      vk_meta_copy_memory_to_image(&cmd->vk, &dev->meta, pCopyMemoryInfo,
                                   &dst_img_props, engine);
   }
   nvk_meta_end_generic(cmd, &save, engine);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdCopyMemoryToImageKHR(VkCommandBuffer commandBuffer,
                            const VkCopyDeviceMemoryImageInfoKHR *pCopyMemoryInfo)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(nvk_image, dst, pCopyMemoryInfo->image);

   VkQueueFlags queue_flags = nvk_cmd_buffer_queue_flags(cmd);
   if (nvk_meta_copy_requires_compute(cmd)) {
      nvk_cmd_copy_memory_to_image_meta(cmd, pCopyMemoryInfo,
                                        VK_PIPELINE_BIND_POINT_COMPUTE);
   } else if ((queue_flags & VK_QUEUE_GRAPHICS_BIT) &&
              nvk_meta_image_copy_gfx_supported(dst)) {
      nvk_cmd_copy_memory_to_image_meta(cmd, pCopyMemoryInfo,
                                        VK_PIPELINE_BIND_POINT_GRAPHICS);
   } else if ((queue_flags & VK_QUEUE_COMPUTE_BIT) &&
       nvk_meta_image_copy_compute_supported(dst)) {
      nvk_cmd_copy_memory_to_image_meta(cmd, pCopyMemoryInfo,
                                        VK_PIPELINE_BIND_POINT_COMPUTE);
   } else {
      nvk_cmd_copy_memory_to_image_ce(cmd, pCopyMemoryInfo);
   }
}

static void
nvk_cmd_copy_image_meta(struct nvk_cmd_buffer *cmd,
                        const VkCopyImageInfo2 *pCopyImageInfo,
                        VkPipelineBindPoint engine)
{
   VK_FROM_HANDLE(nvk_image, src, pCopyImageInfo->srcImage);
   VK_FROM_HANDLE(nvk_image, dst, pCopyImageInfo->dstImage);

   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);

   struct vk_meta_copy_image_properties src_img_props =
      nvk_meta_copy_get_image_properties(src, false);
   struct vk_meta_copy_image_properties dst_img_props =
      nvk_meta_copy_get_image_properties(dst, true);

   union nvk_meta_save_generic save;
   nvk_meta_begin_generic(cmd, &save, engine);
   const bool interleaved_zs = !dst->separate_zs &&
      dst->vk.aspects == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
   if (nvk_meta_copy_raw_depth(cmd, src) && nvk_meta_copy_raw_depth(cmd, dst)) {
      VkCopyImageInfo2 info = *pCopyImageInfo;
      info.regionCount = 1;
      for (uint32_t i = 0; i < pCopyImageInfo->regionCount; i++) {
         VkImageCopy2 region = pCopyImageInfo->pRegions[i];
         info.pRegions = &region;
         if (region.srcSubresource.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) {
            const VkExtent3D extent =
               vk_image_sanitize_extent(&src->vk, region.extent);
            struct nvk_meta_raw_depth_push push = {
               .extent = {
                  extent.width, extent.height,
                  src->vk.image_type == VK_IMAGE_TYPE_3D ? extent.depth :
                     vk_image_subresource_layer_count(&src->vk,
                                                      &region.srcSubresource),
               },
            };
            nvk_meta_raw_depth_set_offset(push.src_offset, src,
               &region.srcSubresource, region.srcOffset);
            nvk_meta_raw_depth_set_offset(push.dst_offset, dst,
               &region.dstSubresource, region.dstOffset);
            if (interleaved_zs && i > 0)
               nvk_meta_copy_aspect_barrier(cmd);
            nvk_meta_raw_depth_dispatch(cmd, src, &region.srcSubresource,
               info.srcImageLayout, dst, &region.dstSubresource,
               info.dstImageLayout, &push);
            region.srcSubresource.aspectMask &= ~VK_IMAGE_ASPECT_DEPTH_BIT;
            region.dstSubresource.aspectMask &= ~VK_IMAGE_ASPECT_DEPTH_BIT;
            if (interleaved_zs && region.srcSubresource.aspectMask)
               nvk_meta_copy_aspect_barrier(cmd);
         }
         if (region.srcSubresource.aspectMask) {
            vk_meta_copy_image(&cmd->vk, &dev->meta, &info,
                               &src_img_props, &dst_img_props, engine);
         }
      }
   } else if (interleaved_zs && engine == VK_PIPELINE_BIND_POINT_COMPUTE) {
      VkCopyImageInfo2 info = *pCopyImageInfo;
      info.regionCount = 1;
      for (uint32_t i = 0; i < pCopyImageInfo->regionCount; i++) {
         info.pRegions = &pCopyImageInfo->pRegions[i];
         if (i > 0 && info.pRegions->dstSubresource.aspectMask !=
                      pCopyImageInfo->pRegions[i - 1].dstSubresource.aspectMask)
            nvk_meta_copy_aspect_barrier(cmd);
         vk_meta_copy_image(&cmd->vk, &dev->meta, &info,
                            &src_img_props, &dst_img_props, engine);
      }
   } else {
      vk_meta_copy_image(&cmd->vk, &dev->meta, pCopyImageInfo,
                         &src_img_props, &dst_img_props, engine);
   }
   nvk_meta_end_generic(cmd, &save, engine);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdCopyImage2(VkCommandBuffer commandBuffer,
                  const VkCopyImageInfo2 *pCopyImageInfo)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(nvk_image, src, pCopyImageInfo->srcImage);
   VK_FROM_HANDLE(nvk_image, dst, pCopyImageInfo->dstImage);

   VkQueueFlags queue_flags = nvk_cmd_buffer_queue_flags(cmd);
   if (nvk_meta_copy_requires_compute(cmd)) {
      nvk_cmd_copy_image_meta(cmd, pCopyImageInfo,
                              VK_PIPELINE_BIND_POINT_COMPUTE);
   } else if ((queue_flags & VK_QUEUE_GRAPHICS_BIT) &&
              nvk_meta_image_copy_gfx_supported(src) &&
              nvk_meta_image_copy_gfx_supported(dst)) {
      nvk_cmd_copy_image_meta(cmd, pCopyImageInfo,
                              VK_PIPELINE_BIND_POINT_GRAPHICS);
   } else if ((queue_flags & VK_QUEUE_COMPUTE_BIT) &&
       nvk_meta_image_copy_compute_supported(src) &&
       nvk_meta_image_copy_compute_supported(dst)) {
      nvk_cmd_copy_image_meta(cmd, pCopyImageInfo,
                              VK_PIPELINE_BIND_POINT_COMPUTE);
   } else {
      nvk_cmd_copy_image_ce(cmd, pCopyImageInfo);
   }
}

static void
nvk_cmd_copy_memory_meta(struct nvk_cmd_buffer *cmd,
                         const VkCopyDeviceMemoryInfoKHR *pCopyMemoryInfo)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);

   struct nvk_meta_save_compute save;
   nvk_meta_begin_compute(cmd, &save);
   vk_meta_copy_memory(&cmd->vk, &dev->meta, pCopyMemoryInfo);
   nvk_meta_end_compute(cmd, &save);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdCopyMemoryKHR(VkCommandBuffer commandBuffer,
                     const VkCopyDeviceMemoryInfoKHR *pCopyMemoryInfo)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, commandBuffer);

   VkQueueFlags queue_flags = nvk_cmd_buffer_queue_flags(cmd);
   if (queue_flags & VK_QUEUE_COMPUTE_BIT) {
      nvk_cmd_copy_memory_meta(cmd, pCopyMemoryInfo);
   } else {
      nvk_cmd_copy_memory_ce(cmd, pCopyMemoryInfo);
   }
}

static void
nvk_cmd_fill_memory_meta(struct nvk_cmd_buffer *cmd,
                         const VkDeviceAddressRangeKHR* pDstRange,
                         VkAddressCommandFlagsKHR dstFlags,
                         uint32_t data)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);

   struct nvk_meta_save_compute save;
   nvk_meta_begin_compute(cmd, &save);
   vk_meta_fill_memory(&cmd->vk, &dev->meta, pDstRange, dstFlags, data);
   nvk_meta_end_compute(cmd, &save);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdFillMemoryKHR(VkCommandBuffer commandBuffer,
                     const VkDeviceAddressRangeKHR* pDstRange,
                     VkAddressCommandFlagsKHR dstFlags,
                     uint32_t data)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, commandBuffer);

   VkQueueFlags queue_flags = nvk_cmd_buffer_queue_flags(cmd);
   if (queue_flags & VK_QUEUE_COMPUTE_BIT) {
      nvk_cmd_fill_memory_meta(cmd, pDstRange, dstFlags, data);
   } else {
      nvk_cmd_fill_memory_ce(cmd, pDstRange->address, pDstRange->size, data);
   }
}
