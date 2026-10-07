/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "panvk_bc_emu.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_meta.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_image.h"
#include "panvk_meta.h"

#include "util/u_math.h"

#include "vk_meta.h"

static const uint32_t bc_decode_spv[] = {
#include "bcn/decode.spv.h"
};

static const uint32_t bc_decode_3d_spv[] = {
#include "bcn/decode_3d.spv.h"
};

static const uint32_t bc_encode_astc_spv[] = {
#include "bcn/encode_astc.spv.h"
};

struct panvk_bc_decode_push {
   uint64_t src;
   uint32_t row_stride;
   uint32_t layer_stride;
   int32_t block_offset[2];
   int32_t block_extent[2];
   int32_t size[2];
   int32_t layer_offset;
   uint32_t dst_row_stride;
   uint64_t dst;
   uint64_t dst_layer_stride;
   uint32_t dst_tiled;
};

enum panvk_bc_decode_variant {
   PANVK_BC_DECODE_2D,
   PANVK_BC_DECODE_3D,
   PANVK_BC_ENCODE_ASTC,
};

struct panvk_bc_decode_key {
   enum panvk_meta_object_key_type type;
   uint32_t kind;
   uint32_t variant;
};

static VkResult
get_bc_decode_pipeline(struct panvk_device *dev, uint32_t kind,
                       enum panvk_bc_decode_variant variant,
                       VkPipelineLayout *layout, VkPipeline *pipeline)
{
   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo set_layout = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct panvk_bc_decode_push),
   };
   const struct panvk_bc_decode_key key = {
      .type = PANVK_META_OBJECT_KEY_BC_DECODE,
      .kind = kind,
      .variant = variant,
   };

   VkResult result =
      vk_meta_get_pipeline_layout(&dev->vk, &dev->meta, &set_layout,
                                  &push_range, &key.type, sizeof(key.type),
                                  layout);
   if (result != VK_SUCCESS)
      return result;

   *pipeline = vk_meta_lookup_pipeline(&dev->meta, &key, sizeof(key));
   if (*pipeline != VK_NULL_HANDLE)
      return VK_SUCCESS;

   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = variant == PANVK_BC_ENCODE_ASTC ? sizeof(bc_encode_astc_spv)
                  : variant == PANVK_BC_DECODE_3D   ? sizeof(bc_decode_3d_spv)
                                                    : sizeof(bc_decode_spv),
      .pCode = variant == PANVK_BC_ENCODE_ASTC ? bc_encode_astc_spv
               : variant == PANVK_BC_DECODE_3D ? bc_decode_3d_spv
                                               : bc_decode_spv,
   };
   const VkSpecializationMapEntry spec_entry = {
      .constantID = 0,
      .offset = 0,
      .size = sizeof(uint32_t),
   };
   const VkSpecializationInfo spec = {
      .mapEntryCount = 1,
      .pMapEntries = &spec_entry,
      .dataSize = sizeof(kind),
      .pData = &kind,
   };
   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = &module_info,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .pName = "main",
         .pSpecializationInfo = &spec,
      },
      .layout = *layout,
   };

   return vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &info, &key,
                                          sizeof(key), pipeline);
}

static void
bc_decode_barrier(struct panvk_cmd_buffer *cmdbuf,
                  VkPipelineStageFlags2 src_stages, VkAccessFlags2 src_access,
                  VkPipelineStageFlags2 dst_stages, VkAccessFlags2 dst_access)
{
   const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = src_stages,
      .srcAccessMask = src_access,
      .dstStageMask = dst_stages,
      .dstAccessMask = dst_access,
   };
   const VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
   };

   panvk_per_arch(CmdPipelineBarrier2)(panvk_cmd_buffer_to_handle(cmdbuf),
                                       &dep);
}

void
panvk_per_arch(cmd_bc_emu_decode)(struct panvk_cmd_buffer *cmdbuf,
                                  struct panvk_image *img,
                                  const VkImageSubresourceLayers *sub,
                                  VkOffset3D offset, VkExtent3D extent)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkCommandBuffer cmd = panvk_cmd_buffer_to_handle(cmdbuf);

   const uint32_t level = sub->mipLevel;
   const bool is_3d = img->vk.image_type == VK_IMAGE_TYPE_3D;
   const uint32_t width = u_minify(img->vk.extent.width, level);
   const uint32_t height = u_minify(img->vk.extent.height, level);
   const uint32_t depth = u_minify(img->vk.extent.depth, level);
   const uint32_t z0 = is_3d ? MIN2((uint32_t)MAX2(offset.z, 0), depth) : 0;
   const uint32_t layer_count =
      is_3d ? MIN2(z0 + extent.depth, depth) - z0
      : sub->layerCount == VK_REMAINING_ARRAY_LAYERS
         ? img->vk.array_layers - sub->baseArrayLayer
         : sub->layerCount;

   const int32_t bx0 = offset.x / 4;
   const int32_t by0 = offset.y / 4;
   const int32_t bx1 =
      MIN2(DIV_ROUND_UP(offset.x + extent.width, 4), DIV_ROUND_UP(width, 4));
   const int32_t by1 =
      MIN2(DIV_ROUND_UP(offset.y + extent.height, 4), DIV_ROUND_UP(height, 4));
   if (bx1 <= bx0 || by1 <= by0 || layer_count == 0)
      return;

   const bool astc = panvk_bc_emu_astc(img->vk.format);
   const enum panvk_bc_decode_variant variant =
      astc ? PANVK_BC_ENCODE_ASTC
      : is_3d ? PANVK_BC_DECODE_3D
              : PANVK_BC_DECODE_2D;

   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result = get_bc_decode_pipeline(
      dev, panvk_bc_emu_kind(img->vk.format), variant, &layout, &pipeline);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmdbuf->vk, result);
      return;
   }

   const VkImageViewUsageCreateInfo view_usage = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT,
   };
   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .pNext = &view_usage,
      .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
      .image = panvk_image_to_handle(img),
      .viewType = is_3d ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D_ARRAY,
      .format = panvk_bc_emu_storage_format(img->vk.format),
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .baseMipLevel = level,
         .levelCount = 1,
         .baseArrayLayer = is_3d ? 0 : sub->baseArrayLayer,
         .layerCount = is_3d ? 1 : layer_count,
      },
   };
   VkImageView view = VK_NULL_HANDLE;
   if (!astc) {
      result = vk_meta_create_image_view(&cmdbuf->vk, &dev->meta, &view_info,
                                         &view);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmdbuf->vk, result);
         return;
      }
   }

   const struct panvk_image_plane *src_plane = &img->planes[0];
   const struct pan_image_slice_layout *slice =
      &src_plane->plane.layout.slices[level];
   const uint64_t z_stride = is_3d ? slice->tiled_or_linear.surface_stride_B
                                   : src_plane->plane.layout.array_stride_B;
   const uint32_t z_first = is_3d ? z0 : sub->baseArrayLayer;
   const struct panvk_image_plane *dst_plane = &img->planes[1];
   const struct pan_image_slice_layout *dst_slice =
      &dst_plane->plane.layout.slices[level];
   const uint64_t dst_z_stride =
      is_3d ? dst_slice->tiled_or_linear.surface_stride_B
            : dst_plane->plane.layout.array_stride_B;
   const struct panvk_bc_decode_push push = {
      .src = src_plane->plane.base + slice->offset_B + z_first * z_stride,
      .row_stride = slice->tiled_or_linear.row_stride_B,
      .layer_stride = z_stride,
      .block_offset = {bx0, by0},
      .block_extent = {bx1 - bx0, by1 - by0},
      .size = {width, height},
      .layer_offset = is_3d ? z0 : 0,
      .dst_row_stride = dst_slice->tiled_or_linear.row_stride_B,
      .dst = dst_plane->plane.base + dst_slice->offset_B + z_first * dst_z_stride,
      .dst_layer_stride = dst_z_stride,
      .dst_tiled = dst_plane->image.props.modifier ==
                   DRM_FORMAT_MOD_ARM_16X16_BLOCK_U_INTERLEAVED,
   };

   bc_decode_barrier(cmdbuf,
                     VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT |
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT |
                        VK_ACCESS_2_SHADER_WRITE_BIT |
                        VK_ACCESS_2_SHADER_READ_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);

   struct panvk_cmd_meta_compute_save_ctx save = {0};
   panvk_per_arch(cmd_meta_compute_start)(cmdbuf, &save);

   const VkDescriptorImageInfo image_info = {
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      .pImageInfo = &image_info,
   };

   disp->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   if (!astc)
      disp->CmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                                    0, 1, &write);
   disp->CmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                          sizeof(push), &push);
   disp->CmdDispatch(cmd, DIV_ROUND_UP(bx1 - bx0, 8),
                     DIV_ROUND_UP(by1 - by0, 8), layer_count);

   panvk_per_arch(cmd_meta_compute_end)(cmdbuf, &save);

   bc_decode_barrier(cmdbuf, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}
