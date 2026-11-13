#ifndef NVK_MICROMAP_H
#define NVK_MICROMAP_H

#include "nvk_private.h"
#include "vk_buffer.h"

struct nvk_micromap {
   struct vk_object_base base;
   struct vk_buffer *buffer;
   uint64_t offset;
   uint64_t size;
};

static inline VkDeviceAddress
nvk_micromap_get_va(const struct nvk_micromap *micromap)
{
   return vk_buffer_address(micromap->buffer, micromap->offset);
}

VK_DEFINE_NONDISP_HANDLE_CASTS(nvk_micromap, base, VkMicromapEXT,
                               VK_OBJECT_TYPE_MICROMAP_EXT)

#endif
