/*
 * Copyright 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_TENSOR_H
#define PANVK_TENSOR_H

#include <stdint.h>

#include "panvk_macros.h"

#include "vk_object.h"

#define PANVK_MAX_TENSOR_DIMS        4
#define PANVK_TENSOR_DESCRIPTOR_SIZE 64

struct panvk_cmd_buffer;
struct panvk_physical_device;

enum panvk_tensor_tiling {
   PANVK_TENSOR_TILING_LINEAR = 0,
   PANVK_TENSOR_TILING_BRICK_16 = 1,
   PANVK_TENSOR_TILING_BRICK_8 = 2,
   PANVK_TENSOR_TILING_BRICK_4 = 3,
   PANVK_TENSOR_TILING_BLOCK_U = 4,
   PANVK_TENSOR_TILING_BLOCK_U_64K = 5,
};

struct panvk_tensor {
   struct vk_object_base base;

   VkTensorCreateFlagsARM flags;
   VkTensorUsageFlagsARM usage;
   VkFormat format;
   enum panvk_tensor_tiling tiling;

   uint32_t dim_count;
   uint32_t dims[PANVK_MAX_TENSOR_DIMS];
   uint32_t strides[PANVK_MAX_TENSOR_DIMS];

   bool rolling;
   uint32_t wraps[PANVK_MAX_TENSOR_DIMS];

   uint64_t size;
   uint64_t alignment;
   uint64_t dev_addr;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_tensor, base, VkTensorARM,
                               VK_OBJECT_TYPE_TENSOR_ARM)

struct panvk_tensor_view {
   struct vk_object_base base;

   struct panvk_tensor *tensor;
   VkFormat format;
   uint32_t desc[PANVK_TENSOR_DESCRIPTOR_SIZE / 4];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_tensor_view, base, VkTensorViewARM,
                               VK_OBJECT_TYPE_TENSOR_VIEW_ARM)

unsigned panvk_tensor_format_size(VkFormat format);

VkFormatFeatureFlags2
panvk_get_tensor_format_features(const struct panvk_physical_device *pdev,
                                 VkFormat format, VkTensorTilingARM tiling);

bool panvk_tensor_image_aliasing_supported(VkFormat format);

#ifdef PAN_ARCH
void panvk_per_arch(meta_copy_tensor)(struct panvk_cmd_buffer *cmdbuf,
                                      const VkCopyTensorInfoARM *info);
#endif

#endif
