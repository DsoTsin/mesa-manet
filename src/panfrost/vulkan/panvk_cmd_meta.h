/*
 * Copyright © 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_CMD_META_H
#define PANVK_CMD_META_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined"
#endif

#include "panvk_cmd_buffer.h"
#include "panvk_cmd_desc_state.h"
#include "panvk_cmd_push_constant.h"
#include "panvk_descriptor_set.h"

struct panvk_shader;

struct panvk_cmd_meta_compute_save_ctx {
   struct {
      const struct panvk_shader *shader;
      struct panvk_shader_desc_state desc;
   } cs;
   const struct panvk_descriptor_set *set0;
   struct {
      struct panvk_opaque_desc desc_storage[MAX_PUSH_DESCS];
      uint64_t descs_dev_addr;
      uint32_t desc_count;
      bool dirty;
   } push_set0;
   struct panvk_push_constant_state push_constants;
   bool cond_render_enabled;
   bool cond_render_inherited;
#if PAN_ARCH >= 10
   struct panvk_shader_instrumentation *shader_instr;
#endif
};

struct panvk_cmd_meta_graphics_save_ctx {
   const struct panvk_graphics_pipeline *pipeline;
   const struct panvk_descriptor_set *set0;
   struct {
      struct panvk_opaque_desc desc_storage[MAX_PUSH_DESCS];
      uint64_t descs_dev_addr;
      uint32_t desc_count;
      bool dirty;
   } push_set0;
   struct panvk_push_constant_state push_constants;
   struct panvk_attrib_buf vb0;

   struct {
      struct vk_dynamic_graphics_state all;
      struct vk_vertex_input_state vi;
      struct vk_sample_locations_state sl;
   } dyn_state;

   struct {
      const struct panvk_shader *shader;
      struct panvk_shader_desc_state desc;
   } fs;

   struct {
      const struct panvk_shader *shader;
      const struct panvk_shader *bound;
      struct panvk_shader_desc_state desc;
   } vs;

   const struct panvk_shader *tcs;
   const struct panvk_shader *tes;

   struct panvk_occlusion_query_state occlusion_query;
   bool cond_render_enabled;
   bool cond_render_inherited;
#if PAN_ARCH >= 10
   struct panvk_shader_instrumentation *shader_instr;
#endif
};

void panvk_per_arch(cmd_meta_resolve_attachments)(
   struct panvk_cmd_buffer *cmdbuf);

void panvk_per_arch(cmd_meta_compute_start)(
   struct panvk_cmd_buffer *cmdbuf,
   struct panvk_cmd_meta_compute_save_ctx *save_ctx);

void panvk_per_arch(cmd_meta_compute_end)(
   struct panvk_cmd_buffer *cmdbuf,
   const struct panvk_cmd_meta_compute_save_ctx *save_ctx);

void panvk_per_arch(cmd_fill_buffer_addr)(VkCommandBuffer commandBuffer,
                                          VkDeviceAddress addr,
                                          VkDeviceSize size, uint32_t data);

#if PAN_ARCH >= 10
struct panvk_image;

void panvk_per_arch(cmd_bc_emu_decode)(struct panvk_cmd_buffer *cmdbuf,
                                       struct panvk_image *img,
                                       const VkImageSubresourceLayers *sub,
                                       VkOffset3D offset, VkExtent3D extent);

void panvk_per_arch(cmd_copy_blocks_to_image)(
   struct panvk_cmd_buffer *cmdbuf, const VkCopyDeviceMemoryImageInfoKHR *info);

void panvk_per_arch(cmd_copy_blocks_from_image)(
   struct panvk_cmd_buffer *cmdbuf, const VkCopyDeviceMemoryImageInfoKHR *info);

void panvk_per_arch(cmd_copy_blocks_image)(struct panvk_cmd_buffer *cmdbuf,
                                           const VkCopyImageInfo2 *info);
#endif

#endif
