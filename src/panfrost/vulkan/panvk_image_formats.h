#ifndef PANVK_IMAGE_FORMATS_H
#define PANVK_IMAGE_FORMATS_H

#include "vk_format.h"

static inline bool
panvk_image_use_yuv_tex(unsigned arch, VkFormat format)
{
   if (arch < 9 || arch >= 14)
      return false;
   switch (format) {
   case VK_FORMAT_G8_B8R8_2PLANE_420_UNORM:
   case VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM:
   case VK_FORMAT_G8_B8R8_2PLANE_422_UNORM:
   case VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM:
   case VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16:
      return true;
   default:
      return false;
   }
}

static inline unsigned
panvk_image_get_tex_count(unsigned arch, VkFormat format)
{
   return panvk_image_use_yuv_tex(arch, format) ?
          1 : vk_format_get_plane_count(format);
}

#endif
