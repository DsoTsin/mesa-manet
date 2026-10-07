/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_meta.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_image.h"
#include "panvk_meta.h"

#include "util/format/u_format.h"
#include "util/u_math.h"

#include "vk_meta.h"

static const uint32_t copy_blocks_spv[] = {
#include "bcn/copy_blocks.spv.h"
};

struct panvk_copy_blocks_push {
   uint64_t src;
   uint64_t dst;
   uint64_t src_layer_stride;
   uint64_t dst_layer_stride;
   int32_t src_block_offset[2];
   int32_t dst_block_offset[2];
   int32_t block_extent[2];
   uint32_t src_row_stride;
   uint32_t dst_row_stride;
   uint32_t src_tile_shift;
   uint32_t dst_tile_shift;
   uint32_t block_size;
};

struct panvk_block_surface {
   uint64_t addr;
   uint64_t layer_stride;
   uint32_t row_stride;
   uint32_t tile_shift;
   int32_t block_offset[2];
};

struct panvk_copy_blocks_ctx {
   struct panvk_cmd_meta_compute_save_ctx save;
   VkPipelineLayout layout;
   bool active;
};

static VkResult
get_copy_blocks_pipeline(struct panvk_device *dev, VkPipelineLayout *layout,
                         VkPipeline *pipeline)
{
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct panvk_copy_blocks_push),
   };
   const enum panvk_meta_object_key_type key =
      PANVK_META_OBJECT_KEY_COPY_BLOCKS;

   VkResult result =
      vk_meta_get_pipeline_layout(&dev->vk, &dev->meta, NULL, &push_range,
                                  &key, sizeof(key), layout);
   if (result != VK_SUCCESS)
      return result;

   *pipeline = vk_meta_lookup_pipeline(&dev->meta, &key, sizeof(key));
   if (*pipeline != VK_NULL_HANDLE)
      return VK_SUCCESS;

   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(copy_blocks_spv),
      .pCode = copy_blocks_spv,
   };
   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = &module_info,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .pName = "main",
      },
      .layout = *layout,
   };

   return vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &info, &key,
                                          sizeof(key), pipeline);
}

static bool
copy_blocks_begin(struct panvk_cmd_buffer *cmdbuf,
                  struct panvk_copy_blocks_ctx *ctx)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   VkPipeline pipeline;

   VkResult result = get_copy_blocks_pipeline(dev, &ctx->layout, &pipeline);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmdbuf->vk, result);
      return false;
   }

   panvk_per_arch(cmd_meta_compute_start)(cmdbuf, &ctx->save);
   dev->vk.dispatch_table.CmdBindPipeline(panvk_cmd_buffer_to_handle(cmdbuf),
                                          VK_PIPELINE_BIND_POINT_COMPUTE,
                                          pipeline);
   ctx->active = true;
   return true;
}

static void
copy_blocks_end(struct panvk_cmd_buffer *cmdbuf,
                struct panvk_copy_blocks_ctx *ctx)
{
   if (!ctx->active)
      return;

   panvk_per_arch(cmd_meta_compute_end)(cmdbuf, &ctx->save);
   ctx->active = false;
}

static void
copy_blocks_dispatch(struct panvk_cmd_buffer *cmdbuf,
                     struct panvk_copy_blocks_ctx *ctx,
                     const struct panvk_block_surface *src,
                     const struct panvk_block_surface *dst,
                     uint32_t block_size, uint32_t width, uint32_t height,
                     uint32_t layers)
{
   if (!width || !height || !layers)
      return;

   if (!ctx->active && !copy_blocks_begin(cmdbuf, ctx))
      return;

   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkCommandBuffer cmd = panvk_cmd_buffer_to_handle(cmdbuf);
   const struct panvk_copy_blocks_push push = {
      .src = src->addr,
      .dst = dst->addr,
      .src_layer_stride = src->layer_stride,
      .dst_layer_stride = dst->layer_stride,
      .src_block_offset = {src->block_offset[0], src->block_offset[1]},
      .dst_block_offset = {dst->block_offset[0], dst->block_offset[1]},
      .block_extent = {width, height},
      .src_row_stride = src->row_stride,
      .dst_row_stride = dst->row_stride,
      .src_tile_shift = src->tile_shift,
      .dst_tile_shift = dst->tile_shift,
      .block_size = block_size,
   };

   disp->CmdPushConstants(cmd, ctx->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                          sizeof(push), &push);
   disp->CmdDispatch(cmd, DIV_ROUND_UP(width, 8), DIV_ROUND_UP(height, 8),
                     layers);
}

static enum pipe_format
image_block_format(const struct panvk_image *img)
{
   return img->planes[0].image.props.format;
}

static bool
image_block_surface(const struct panvk_image *img,
                    const VkImageSubresourceLayers *sub, VkOffset3D offset,
                    struct panvk_block_surface *surf)
{
   if (img->vk.samples > 1 ||
       (img->plane_count > 1 && img->bc_emu_format == VK_FORMAT_UNDEFINED))
      return false;

   const struct panvk_image_plane *plane = &img->planes[0];
   const uint64_t mod = plane->image.props.modifier;
   if (mod != DRM_FORMAT_MOD_LINEAR &&
       mod != DRM_FORMAT_MOD_ARM_16X16_BLOCK_U_INTERLEAVED)
      return false;

   const enum pipe_format pfmt = plane->image.props.format;
   const struct pan_image_slice_layout *slice =
      &plane->plane.layout.slices[sub->mipLevel];
   const bool is_3d = img->vk.image_type == VK_IMAGE_TYPE_3D;
   const uint64_t z_stride = is_3d ? slice->tiled_or_linear.surface_stride_B
                                   : plane->plane.layout.array_stride_B;
   const uint32_t z = is_3d ? offset.z : sub->baseArrayLayer;

   *surf = (struct panvk_block_surface){
      .addr = plane->plane.base + slice->offset_B + z * z_stride,
      .layer_stride = z_stride,
      .row_stride = slice->tiled_or_linear.row_stride_B,
      .tile_shift = mod == DRM_FORMAT_MOD_LINEAR       ? 0
                    : util_format_is_compressed(pfmt) ? 2
                                                      : 4,
      .block_offset = {
         offset.x / (int32_t)util_format_get_blockwidth(pfmt),
         offset.y / (int32_t)util_format_get_blockheight(pfmt),
      },
   };
   return true;
}

static struct panvk_block_surface
memory_block_surface(uint64_t addr, uint32_t row_length, uint32_t image_height,
                     VkExtent3D extent, enum pipe_format pfmt)
{
   const uint32_t row_texels = row_length ? row_length : extent.width;
   const uint32_t rows = image_height ? image_height : extent.height;
   const uint32_t row_stride =
      DIV_ROUND_UP(row_texels, util_format_get_blockwidth(pfmt)) *
      util_format_get_blocksize(pfmt);

   return (struct panvk_block_surface){
      .addr = addr,
      .layer_stride = (uint64_t)DIV_ROUND_UP(
                         rows, util_format_get_blockheight(pfmt)) *
                      row_stride,
      .row_stride = row_stride,
   };
}

static uint32_t
image_copy_layers(const struct panvk_image *img,
                  const VkImageSubresourceLayers *sub, uint32_t depth)
{
   return img->vk.image_type == VK_IMAGE_TYPE_3D
             ? depth
             : vk_image_subresource_layer_count(&img->vk, sub);
}

static void
copy_blocks_memory_image(struct panvk_cmd_buffer *cmdbuf,
                         const VkCopyDeviceMemoryImageInfoKHR *info,
                         bool to_image)
{
   VK_FROM_HANDLE(panvk_image, img, info->image);
   const enum pipe_format pfmt = image_block_format(img);
   const uint32_t bw = util_format_get_blockwidth(pfmt);
   const uint32_t bh = util_format_get_blockheight(pfmt);
   struct panvk_copy_blocks_ctx ctx = {0};

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkDeviceMemoryImageCopyKHR *r = &info->pRegions[i];
      struct panvk_block_surface img_surf;

      ASSERTED bool ok = image_block_surface(img, &r->imageSubresource,
                                             r->imageOffset, &img_surf);
      assert(ok);

      const struct panvk_block_surface mem_surf = memory_block_surface(
         r->addressRange.address, r->addressRowLength, r->addressImageHeight,
         r->imageExtent, pfmt);

      copy_blocks_dispatch(
         cmdbuf, &ctx, to_image ? &mem_surf : &img_surf,
         to_image ? &img_surf : &mem_surf, util_format_get_blocksize(pfmt),
         DIV_ROUND_UP(r->imageExtent.width, bw),
         DIV_ROUND_UP(r->imageExtent.height, bh),
         image_copy_layers(img, &r->imageSubresource, r->imageExtent.depth));
   }

   copy_blocks_end(cmdbuf, &ctx);
}

void
panvk_per_arch(cmd_copy_blocks_to_image)(
   struct panvk_cmd_buffer *cmdbuf, const VkCopyDeviceMemoryImageInfoKHR *info)
{
   copy_blocks_memory_image(cmdbuf, info, true);
}

void
panvk_per_arch(cmd_copy_blocks_from_image)(
   struct panvk_cmd_buffer *cmdbuf, const VkCopyDeviceMemoryImageInfoKHR *info)
{
   copy_blocks_memory_image(cmdbuf, info, false);
}

static void
copy_blocks_stage_barrier(struct panvk_cmd_buffer *cmdbuf)
{
   const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
   };
   const VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
   };

   panvk_per_arch(CmdPipelineBarrier2)(panvk_cmd_buffer_to_handle(cmdbuf),
                                       &dep);
}

static VkExtent3D
blocks_to_texels(const struct panvk_image *img, uint32_t level,
                 VkOffset3D offset, uint32_t width, uint32_t height,
                 uint32_t layers)
{
   const enum pipe_format pfmt = image_block_format(img);
   const VkExtent3D mip = vk_image_mip_level_extent(&img->vk, level);

   return (VkExtent3D){
      .width = MIN2(width * util_format_get_blockwidth(pfmt),
                    mip.width - offset.x),
      .height = MIN2(height * util_format_get_blockheight(pfmt),
                     mip.height - offset.y),
      .depth = img->vk.image_type == VK_IMAGE_TYPE_3D ? layers : 1,
   };
}

static void
copy_blocks_image_staged(struct panvk_cmd_buffer *cmdbuf,
                         const VkCopyImageInfo2 *info, const VkImageCopy2 *r,
                         const struct panvk_block_surface *src_surf,
                         const struct panvk_block_surface *dst_surf,
                         uint32_t block_size, uint32_t width, uint32_t height,
                         uint32_t layers)
{
   VK_FROM_HANDLE(panvk_image, src_img, info->srcImage);
   VK_FROM_HANDLE(panvk_image, dst_img, info->dstImage);
   VkCommandBuffer cmd = panvk_cmd_buffer_to_handle(cmdbuf);
   const uint64_t size = (uint64_t)width * height * layers * block_size;

   struct pan_ptr tmp = panvk_cmd_alloc_dev_mem(cmdbuf, desc, size, 64);
   if (!tmp.gpu)
      return;

   const struct panvk_block_surface tmp_surf = {
      .addr = tmp.gpu,
      .layer_stride = (uint64_t)width * height * block_size,
      .row_stride = width * block_size,
   };

   if (src_surf) {
      struct panvk_copy_blocks_ctx ctx = {0};
      const VkExtent3D extent = blocks_to_texels(
         dst_img, r->dstSubresource.mipLevel, r->dstOffset, width, height,
         layers);
      const VkDeviceMemoryImageCopyKHR region = {
         .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_IMAGE_COPY_KHR,
         .addressRange = {.address = tmp.gpu, .size = size},
         .addressRowLength =
            width * util_format_get_blockwidth(image_block_format(dst_img)),
         .addressImageHeight =
            height * util_format_get_blockheight(image_block_format(dst_img)),
         .imageSubresource = r->dstSubresource,
         .imageLayout = info->dstImageLayout,
         .imageOffset = r->dstOffset,
         .imageExtent = extent,
      };
      const VkCopyDeviceMemoryImageInfoKHR copy = {
         .sType = VK_STRUCTURE_TYPE_COPY_DEVICE_MEMORY_IMAGE_INFO_KHR,
         .image = info->dstImage,
         .regionCount = 1,
         .pRegions = &region,
      };

      copy_blocks_dispatch(cmdbuf, &ctx, src_surf, &tmp_surf, block_size,
                           width, height, layers);
      copy_blocks_end(cmdbuf, &ctx);
      copy_blocks_stage_barrier(cmdbuf);
      panvk_per_arch(CmdCopyMemoryToImageKHR)(cmd, &copy);
   } else {
      struct panvk_copy_blocks_ctx ctx = {0};
      const VkDeviceMemoryImageCopyKHR region = {
         .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_IMAGE_COPY_KHR,
         .addressRange = {.address = tmp.gpu, .size = size},
         .addressRowLength =
            width * util_format_get_blockwidth(image_block_format(src_img)),
         .addressImageHeight =
            height * util_format_get_blockheight(image_block_format(src_img)),
         .imageSubresource = r->srcSubresource,
         .imageLayout = info->srcImageLayout,
         .imageOffset = r->srcOffset,
         .imageExtent = {
            .width = r->extent.width,
            .height = r->extent.height,
            .depth = src_img->vk.image_type == VK_IMAGE_TYPE_3D
                        ? r->extent.depth
                        : 1,
         },
      };
      const VkCopyDeviceMemoryImageInfoKHR copy = {
         .sType = VK_STRUCTURE_TYPE_COPY_DEVICE_MEMORY_IMAGE_INFO_KHR,
         .image = info->srcImage,
         .regionCount = 1,
         .pRegions = &region,
      };

      panvk_per_arch(CmdCopyImageToMemoryKHR)(cmd, &copy);
      copy_blocks_stage_barrier(cmdbuf);
      copy_blocks_dispatch(cmdbuf, &ctx, &tmp_surf, dst_surf, block_size,
                           width, height, layers);
      copy_blocks_end(cmdbuf, &ctx);
   }
}

void
panvk_per_arch(cmd_copy_blocks_image)(struct panvk_cmd_buffer *cmdbuf,
                                      const VkCopyImageInfo2 *info)
{
   VK_FROM_HANDLE(panvk_image, src_img, info->srcImage);
   VK_FROM_HANDLE(panvk_image, dst_img, info->dstImage);
   const enum pipe_format src_fmt = image_block_format(src_img);
   const uint32_t block_size = util_format_get_blocksize(src_fmt);
   struct panvk_copy_blocks_ctx ctx = {0};

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageCopy2 *r = &info->pRegions[i];
      const uint32_t width =
         DIV_ROUND_UP(r->extent.width, util_format_get_blockwidth(src_fmt));
      const uint32_t height =
         DIV_ROUND_UP(r->extent.height, util_format_get_blockheight(src_fmt));
      const uint32_t layers =
         image_copy_layers(src_img, &r->srcSubresource, r->extent.depth);
      struct panvk_block_surface src_surf, dst_surf;
      const bool src_ok =
         image_block_surface(src_img, &r->srcSubresource, r->srcOffset,
                             &src_surf);
      const bool dst_ok =
         image_block_surface(dst_img, &r->dstSubresource, r->dstOffset,
                             &dst_surf);

      assert(src_ok || dst_ok);
      if (src_ok && dst_ok) {
         copy_blocks_dispatch(cmdbuf, &ctx, &src_surf, &dst_surf, block_size,
                              width, height, layers);
      } else {
         copy_blocks_end(cmdbuf, &ctx);
         copy_blocks_image_staged(cmdbuf, info, r, src_ok ? &src_surf : NULL,
                                  dst_ok ? &dst_surf : NULL, block_size, width,
                                  height, layers);
      }
   }

   copy_blocks_end(cmdbuf, &ctx);

   if (dst_img->bc_emu_format == VK_FORMAT_UNDEFINED)
      return;

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageCopy2 *r = &info->pRegions[i];
      const VkExtent3D extent = blocks_to_texels(
         dst_img, r->dstSubresource.mipLevel, r->dstOffset,
         DIV_ROUND_UP(r->extent.width, util_format_get_blockwidth(src_fmt)),
         DIV_ROUND_UP(r->extent.height, util_format_get_blockheight(src_fmt)),
         r->extent.depth);

      panvk_per_arch(cmd_bc_emu_decode)(cmdbuf, dst_img, &r->dstSubresource,
                                        r->dstOffset, extent);
   }
}
