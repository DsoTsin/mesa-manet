#ifndef PANVK_MICROMAP_H
#define PANVK_MICROMAP_H

#include "vk_buffer.h"

struct panvk_micromap {
   struct vk_object_base base;
   struct vk_buffer *buffer;
   VkDeviceSize offset;
   VkDeviceSize size;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_micromap, base, VkMicromapEXT,
                               VK_OBJECT_TYPE_MICROMAP_EXT)

static inline VkDeviceAddress
panvk_micromap_address(const struct panvk_micromap *micromap)
{
   return vk_buffer_address(micromap->buffer, micromap->offset);
}

#endif
