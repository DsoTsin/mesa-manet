/*
 * Copyright 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "panvk_tensor.h"
#include "panvk_device.h"
#include "panvk_device_memory.h"
#include "panvk_entrypoints.h"
#include "panvk_physical_device.h"

#include "util/format/u_format.h"
#include "util/u_math.h"
#include "vk_format.h"
#include "vk_log.h"
#include "vk_util.h"

struct panvk_tensor_format {
   VkFormat format;
   uint8_t size;
   bool block_u;
};

static const struct panvk_tensor_format tensor_formats[] = {
   {VK_FORMAT_R8_UINT, 1, true},     {VK_FORMAT_R8_SINT, 1, true},
   {VK_FORMAT_R8_BOOL_ARM, 1, true}, {VK_FORMAT_R16_UINT, 2, true},
   {VK_FORMAT_R16_SINT, 2, true},    {VK_FORMAT_R16_SFLOAT, 2, false},
   {VK_FORMAT_R32_UINT, 4, true},    {VK_FORMAT_R32_SINT, 4, true},
   {VK_FORMAT_R32_SFLOAT, 4, false}, {VK_FORMAT_R64_UINT, 8, false},
   {VK_FORMAT_R64_SINT, 8, false},
};

static const struct panvk_tensor_format *
find_tensor_format(VkFormat format)
{
   for (unsigned i = 0; i < ARRAY_SIZE(tensor_formats); i++) {
      if (tensor_formats[i].format == format)
         return &tensor_formats[i];
   }

   return NULL;
}

unsigned
panvk_tensor_format_size(VkFormat format)
{
   const struct panvk_tensor_format *fmt = find_tensor_format(format);
   return fmt ? fmt->size : 0;
}

static enum panvk_tensor_tiling
get_tensor_tiling(VkTensorTilingARM tiling)
{
   switch (tiling) {
   case VK_TENSOR_TILING_BRICK_16_WIDE_ARM:
      return PANVK_TENSOR_TILING_BRICK_16;
   case VK_TENSOR_TILING_BRICK_8_WIDE_ARM:
      return PANVK_TENSOR_TILING_BRICK_8;
   case VK_TENSOR_TILING_BRICK_4_WIDE_ARM:
      return PANVK_TENSOR_TILING_BRICK_4;
   case VK_TENSOR_TILING_BLOCK_U_INTERLEAVED_ARM:
      return PANVK_TENSOR_TILING_BLOCK_U;
   case VK_TENSOR_TILING_BLOCK_U_INTERLEAVED_64K_ARM:
      return PANVK_TENSOR_TILING_BLOCK_U_64K;
   default:
      return PANVK_TENSOR_TILING_LINEAR;
   }
}

static unsigned
brick_width(enum panvk_tensor_tiling tiling)
{
   switch (tiling) {
   case PANVK_TENSOR_TILING_BRICK_16:
      return 16;
   case PANVK_TENSOR_TILING_BRICK_8:
      return 8;
   case PANVK_TENSOR_TILING_BRICK_4:
      return 4;
   default:
      UNREACHABLE("Not a brick tiling");
   }
}

VkFormatFeatureFlags2
panvk_get_tensor_format_features(const struct panvk_physical_device *pdev,
                                 VkFormat format, VkTensorTilingARM tiling)
{
   const struct panvk_tensor_format *fmt = find_tensor_format(format);
   if (!fmt || !pdev->vk.supported_extensions.ARM_tensors)
      return 0;

   const VkFormatFeatureFlags2 transfer =
      VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
      VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;

   enum panvk_tensor_tiling ptiling = get_tensor_tiling(tiling);
   switch (ptiling) {
   case PANVK_TENSOR_TILING_LINEAR:
      return transfer | VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_ARM |
             VK_FORMAT_FEATURE_2_TENSOR_IMAGE_ALIASING_BIT_ARM;
   case PANVK_TENSOR_TILING_BRICK_16:
   case PANVK_TENSOR_TILING_BRICK_8:
   case PANVK_TENSOR_TILING_BRICK_4:
      return 64 / brick_width(ptiling) / fmt->size
                ? VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_ARM
                : 0;
   case PANVK_TENSOR_TILING_BLOCK_U:
   case PANVK_TENSOR_TILING_BLOCK_U_64K:
      return fmt->block_u ? transfer | VK_FORMAT_FEATURE_2_TENSOR_SHADER_BIT_ARM
                          : 0;
   }

   return 0;
}

bool
panvk_tensor_image_aliasing_supported(VkFormat format)
{
   const VkImageAspectFlags aspects = vk_format_aspects(format);
   if (aspects == VK_IMAGE_ASPECT_DEPTH_BIT ||
       aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
      return true;

   if (aspects != VK_IMAGE_ASPECT_COLOR_BIT)
      return false;

   const enum pipe_format pfmt = vk_format_to_pipe_format(format);
   const struct util_format_description *desc = util_format_description(pfmt);
   if (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN || util_format_is_srgb(pfmt) ||
       util_format_is_scaled(pfmt))
      return false;

   const unsigned size = desc->channel[0].size;
   if (size < 8 || size > 64 || !util_is_power_of_two_nonzero(size))
      return false;

   for (unsigned i = 1; i < desc->nr_channels; i++) {
      if (desc->channel[i].size != size)
         return false;
   }

   return true;
}

static void
block_u_64k_granularity(unsigned bpp, unsigned *width, unsigned *height)
{
   switch (bpp) {
   case 1:
      *width = 256;
      *height = 256;
      break;
   case 2:
      *width = 256;
      *height = 128;
      break;
   case 4:
      *width = 128;
      *height = 128;
      break;
   case 8:
      *width = 128;
      *height = 64;
      break;
   case 16:
      *width = 64;
      *height = 64;
      break;
   default:
      *width = 1;
      *height = 1;
      break;
   }
}

static void
init_image_aliasing_strides(struct panvk_tensor *tensor, unsigned elem)
{
   const unsigned n = tensor->dim_count;
   const uint32_t *d = tensor->dims;
   uint32_t *s = tensor->strides;

   const uint32_t bpp = elem * d[n - 1];
   const uint32_t width = n >= 2 ? d[n - 2] : 1;
   const uint32_t height = n >= 3 ? d[n - 3] : 1;
   const uint32_t depth = n >= 4 ? d[n - 4] : 1;
   const uint32_t row_stride = ALIGN_POT(width * bpp, 64);

   s[n - 1] = elem;
   if (n >= 2)
      s[n - 2] = bpp;
   if (n >= 3)
      s[n - 3] = height > 1 ? row_stride : bpp * width;
   if (n >= 4)
      s[n - 4] = depth > 1 ? row_stride * height : s[n - 3] * height;
}

static void
init_tensor_strides(struct panvk_tensor *tensor,
                    const VkTensorDescriptionARM *desc)
{
   const unsigned n = tensor->dim_count;
   const unsigned elem = panvk_tensor_format_size(tensor->format);
   const uint32_t *d = tensor->rolling ? tensor->wraps : tensor->dims;
   uint32_t *s = tensor->strides;

   if (desc->pStrides) {
      for (unsigned i = 0; i < n; i++)
         s[i] = desc->pStrides[i];
      return;
   }

   if (desc->tiling == VK_TENSOR_TILING_OPTIMAL_ARM &&
       (desc->usage & VK_TENSOR_USAGE_IMAGE_ALIASING_BIT_ARM)) {
      init_image_aliasing_strides(tensor, elem);
      return;
   }

   s[n - 1] = elem;

   switch (tensor->tiling) {
   case PANVK_TENSOR_TILING_LINEAR:
      for (unsigned i = n - 1; i > 0; i--)
         s[i - 1] = s[i] * d[i];
      break;

   case PANVK_TENSOR_TILING_BRICK_16:
   case PANVK_TENSOR_TILING_BRICK_8:
   case PANVK_TENSOR_TILING_BRICK_4: {
      const unsigned width = brick_width(tensor->tiling);
      if (n >= 2)
         s[n - 2] = DIV_ROUND_UP(d[n - 1], 64 / width / elem) * 64;
      if (n >= 3)
         s[n - 3] = s[n - 2] * DIV_ROUND_UP(d[n - 2], width);
      for (unsigned i = n >= 3 ? n - 3 : 0; i > 0; i--)
         s[i - 1] = s[i] * d[i];
      break;
   }

   case PANVK_TENSOR_TILING_BLOCK_U:
      if (n >= 2)
         s[n - 2] = (elem * d[n - 1]) << 8;
      if (n >= 3)
         s[n - 3] = s[n - 2] * DIV_ROUND_UP(d[n - 2], 16);
      if (n >= 4)
         s[n - 4] = s[n - 3] * DIV_ROUND_UP(d[n - 3], 16);
      break;

   case PANVK_TENSOR_TILING_BLOCK_U_64K: {
      unsigned width, height;
      block_u_64k_granularity(elem * (d[n - 1] == 3 ? 4 : d[n - 1]), &width,
                              &height);
      if (n >= 2)
         s[n - 2] = 0x10000;
      if (n >= 3)
         s[n - 3] = DIV_ROUND_UP(d[n - 2], width) << 16;
      if (n >= 4)
         s[n - 4] = s[n - 3] * DIV_ROUND_UP(d[n - 3], height);
      break;
   }
   }
}

static void
init_tensor_size(struct panvk_tensor *tensor)
{
   const unsigned n = tensor->dim_count;
   const uint32_t *d = tensor->rolling ? tensor->wraps : tensor->dims;
   uint64_t outer = d[0];

   tensor->alignment = 64;

   switch (tensor->tiling) {
   case PANVK_TENSOR_TILING_LINEAR:
      break;

   case PANVK_TENSOR_TILING_BRICK_16:
   case PANVK_TENSOR_TILING_BRICK_8:
   case PANVK_TENSOR_TILING_BRICK_4:
      if (n < 3)
         outer = DIV_ROUND_UP(d[0], brick_width(tensor->tiling));
      break;

   case PANVK_TENSOR_TILING_BLOCK_U:
      if (n <= 2)
         outer = DIV_ROUND_UP(d[0], 16);
      break;

   case PANVK_TENSOR_TILING_BLOCK_U_64K:
      tensor->alignment = 0x10000;
      if (n < 4) {
         unsigned width, height;
         block_u_64k_granularity(
            panvk_tensor_format_size(tensor->format) * d[n - 1], &width,
            &height);
         outer = DIV_ROUND_UP(d[0], height);
      }
      break;
   }

   tensor->size = tensor->strides[0] * outer;
}

static void
init_tensor(struct panvk_tensor *tensor, const VkTensorCreateInfoARM *info)
{
   const VkTensorDescriptionARM *desc = info->pDescription;

   tensor->flags = info->flags;
   tensor->usage = desc->usage;
   tensor->format = desc->format;
   tensor->tiling = get_tensor_tiling(desc->tiling);
   tensor->dim_count = desc->dimensionCount;
   for (unsigned i = 0; i < desc->dimensionCount; i++)
      tensor->dims[i] = desc->pDimensions[i];

   const VkTensorRollingBackingCreateInfoARM *rolling =
      vk_find_struct_const(info->pNext, TENSOR_ROLLING_BACKING_CREATE_INFO_ARM);
   if (rolling) {
      tensor->rolling = true;
      memcpy(tensor->wraps, rolling->wraps,
             desc->dimensionCount * sizeof(uint32_t));
   }

   init_tensor_strides(tensor, desc);
   init_tensor_size(tensor);
}

static void
get_tensor_memory_requirements(const struct panvk_device *device,
                               const struct panvk_tensor *tensor,
                               VkMemoryRequirements2 *reqs)
{
   const struct panvk_physical_device *pdev =
      to_panvk_physical_device(device->vk.physical);
   const bool dedicated = tensor->tiling == PANVK_TENSOR_TILING_BLOCK_U_64K;

   reqs->memoryRequirements = (VkMemoryRequirements){
      .size = tensor->size,
      .alignment = tensor->alignment,
      .memoryTypeBits = BITFIELD_MASK(pdev->memory.type_count),
   };

   vk_foreach_struct(sType, ext, reqs->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS: {
         VkMemoryDedicatedRequirements *ded = (void *)ext;
         ded->prefersDedicatedAllocation = dedicated;
         ded->requiresDedicatedAllocation = dedicated;
         break;
      }
      default:
         vk_debug_ignored_stype(sType);
         break;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_CreateTensorARM(VkDevice _device, const VkTensorCreateInfoARM *pCreateInfo,
                      const VkAllocationCallbacks *pAllocator,
                      VkTensorARM *pTensor)
{
   VK_FROM_HANDLE(panvk_device, device, _device);

   struct panvk_tensor *tensor = vk_object_zalloc(
      &device->vk, pAllocator, sizeof(*tensor), VK_OBJECT_TYPE_TENSOR_ARM);
   if (!tensor)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   init_tensor(tensor, pCreateInfo);

   *pTensor = panvk_tensor_to_handle(tensor);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_DestroyTensorARM(VkDevice _device, VkTensorARM _tensor,
                       const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_tensor, tensor, _tensor);

   if (!tensor)
      return;

   if (tensor->dev_addr)
      panvk_address_binding_report(device, &tensor->base, tensor->dev_addr,
                                   tensor->size,
                                   VK_DEVICE_ADDRESS_BINDING_TYPE_UNBIND_EXT);

   vk_object_free(&device->vk, pAllocator, tensor);
}

VKAPI_ATTR void VKAPI_CALL
panvk_GetTensorMemoryRequirementsARM(
   VkDevice _device, const VkTensorMemoryRequirementsInfoARM *pInfo,
   VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_tensor, tensor, pInfo->tensor);

   get_tensor_memory_requirements(device, tensor, pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL
panvk_GetDeviceTensorMemoryRequirementsARM(
   VkDevice _device, const VkDeviceTensorMemoryRequirementsARM *pInfo,
   VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   struct panvk_tensor tensor = {0};

   init_tensor(&tensor, pInfo->pCreateInfo);
   get_tensor_memory_requirements(device, &tensor, pMemoryRequirements);
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_BindTensorMemoryARM(VkDevice _device, uint32_t bindInfoCount,
                          const VkBindTensorMemoryInfoARM *pBindInfos)
{
   VK_FROM_HANDLE(panvk_device, device, _device);

   for (uint32_t i = 0; i < bindInfoCount; i++) {
      VK_FROM_HANDLE(panvk_tensor, tensor, pBindInfos[i].tensor);
      VK_FROM_HANDLE(panvk_device_memory, mem, pBindInfos[i].memory);

      tensor->dev_addr = mem->addr.dev + pBindInfos[i].memoryOffset;
      panvk_address_binding_report(device, &tensor->base, tensor->dev_addr,
                                   tensor->size,
                                   VK_DEVICE_ADDRESS_BINDING_TYPE_BIND_EXT);
   }

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_DestroyTensorViewARM(VkDevice _device, VkTensorViewARM tensorView,
                           const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_tensor_view, view, tensorView);

   if (view)
      vk_object_free(&device->vk, pAllocator, view);
}

VKAPI_ATTR void VKAPI_CALL
panvk_GetPhysicalDeviceExternalTensorPropertiesARM(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceExternalTensorInfoARM *pExternalTensorInfo,
   VkExternalTensorPropertiesARM *pExternalTensorProperties)
{
   VK_FROM_HANDLE(panvk_physical_device, pdev, physicalDevice);
   const VkExternalMemoryHandleTypeFlagBits handle_type =
      pExternalTensorInfo->handleType;
   VkExternalMemoryProperties props = {0};

   switch (handle_type) {
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:
      props.externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
      props.compatibleHandleTypes = handle_type;
      break;
   case VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID:
      if (pdev->vk.supported_extensions
             .ANDROID_external_memory_android_hardware_buffer) {
         props.externalMemoryFeatures =
            VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT |
            VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
         props.exportFromImportedHandleTypes = handle_type;
         props.compatibleHandleTypes = handle_type;
      }
      break;
   default:
      break;
   }

   pExternalTensorProperties->externalMemoryProperties = props;
}
