/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_BC_EMU_H
#define PANVK_BC_EMU_H

#include "util/bitset.h"
#include "util/format/u_format.h"

#include "vk_format.h"

#include "pan_format.h"
#include "pan_props.h"

#include "panvk_physical_device.h"

enum panvk_bc_emu_kind {
   PANVK_BC_EMU_BC1_RGB = 1,
   PANVK_BC_EMU_BC1_RGBA = 2,
   PANVK_BC_EMU_BC2 = 3,
   PANVK_BC_EMU_BC3 = 4,
   PANVK_BC_EMU_BC4_UNORM = 5,
   PANVK_BC_EMU_BC4_SNORM = 6,
   PANVK_BC_EMU_BC5_UNORM = 7,
   PANVK_BC_EMU_BC5_SNORM = 8,
   PANVK_BC_EMU_BC6H_UF16 = 9,
   PANVK_BC_EMU_BC6H_SF16 = 10,
   PANVK_BC_EMU_BC7 = 11,
};

static inline bool
panvk_format_is_bc(VkFormat format)
{
   return format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && format <= VK_FORMAT_BC7_SRGB_BLOCK;
}

static inline bool
panvk_bc_emulation(const struct panvk_physical_device *pdev)
{
   if (pan_arch(pdev->kmod.dev->props.gpu_id) < 10)
      return false;

   const struct pan_format fmt = pdev->formats.all[PIPE_FORMAT_DXT1_RGB];
   return !(BITFIELD_BIT(fmt.texfeat_bit) &
            pan_query_compressed_formats(&pdev->kmod.dev->props));
}

static inline bool
panvk_bc_emulated(const struct panvk_physical_device *pdev, VkFormat format)
{
   return panvk_format_is_bc(format) && panvk_bc_emulation(pdev);
}

static inline VkFormat
panvk_bc_emu_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
   case VK_FORMAT_BC2_UNORM_BLOCK:
   case VK_FORMAT_BC3_UNORM_BLOCK:
   case VK_FORMAT_BC7_UNORM_BLOCK:
      return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
   case VK_FORMAT_BC2_SRGB_BLOCK:
   case VK_FORMAT_BC3_SRGB_BLOCK:
   case VK_FORMAT_BC7_SRGB_BLOCK:
      return VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
   case VK_FORMAT_BC4_UNORM_BLOCK:
      return VK_FORMAT_R8_UNORM;
   case VK_FORMAT_BC4_SNORM_BLOCK:
      return VK_FORMAT_R8_SNORM;
   case VK_FORMAT_BC5_UNORM_BLOCK:
      return VK_FORMAT_R8G8_UNORM;
   case VK_FORMAT_BC5_SNORM_BLOCK:
      return VK_FORMAT_R8G8_SNORM;
   case VK_FORMAT_BC6H_UFLOAT_BLOCK:
   case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
   default:
      return VK_FORMAT_UNDEFINED;
   }
}

static inline bool
panvk_bc_emu_astc(VkFormat format)
{
   return vk_format_is_compressed(panvk_bc_emu_format(format));
}

static inline VkFormat
panvk_bc_emu_storage_format(VkFormat format)
{
   switch (vk_format_get_blocksize(panvk_bc_emu_format(format))) {
   case 1:
      return VK_FORMAT_R8_UINT;
   case 2:
      return VK_FORMAT_R8G8_UINT;
   case 4:
      return VK_FORMAT_R8G8B8A8_UINT;
   default:
      return VK_FORMAT_R16G16B16A16_UINT;
   }
}

static inline enum panvk_bc_emu_kind
panvk_bc_emu_kind(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
      return PANVK_BC_EMU_BC1_RGB;
   case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
   case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
      return PANVK_BC_EMU_BC1_RGBA;
   case VK_FORMAT_BC2_UNORM_BLOCK:
   case VK_FORMAT_BC2_SRGB_BLOCK:
      return PANVK_BC_EMU_BC2;
   case VK_FORMAT_BC3_UNORM_BLOCK:
   case VK_FORMAT_BC3_SRGB_BLOCK:
      return PANVK_BC_EMU_BC3;
   case VK_FORMAT_BC4_UNORM_BLOCK:
      return PANVK_BC_EMU_BC4_UNORM;
   case VK_FORMAT_BC4_SNORM_BLOCK:
      return PANVK_BC_EMU_BC4_SNORM;
   case VK_FORMAT_BC5_UNORM_BLOCK:
      return PANVK_BC_EMU_BC5_UNORM;
   case VK_FORMAT_BC5_SNORM_BLOCK:
      return PANVK_BC_EMU_BC5_SNORM;
   case VK_FORMAT_BC6H_UFLOAT_BLOCK:
      return PANVK_BC_EMU_BC6H_UF16;
   case VK_FORMAT_BC6H_SFLOAT_BLOCK:
      return PANVK_BC_EMU_BC6H_SF16;
   default:
      return PANVK_BC_EMU_BC7;
   }
}

#endif
