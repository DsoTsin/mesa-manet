#ifndef KRAIDOC_CONTEXT_H
#define KRAIDOC_CONTEXT_H

#include <stdint.h>
#include <errno.h>
#include "pan_model.h"
#include "vk_device.h"
#include "vk_alloc.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "genxml/gen_macros.h"
#include "panvk_macros.h"

VkResult panvk_per_arch(CreateDescriptorSetLayout)(
   VkDevice, const VkDescriptorSetLayoutCreateInfo *,
   const VkAllocationCallbacks *, VkDescriptorSetLayout *);
void panvk_per_arch(GetDescriptorSetLayoutSupport)(
   VkDevice, const VkDescriptorSetLayoutCreateInfo *,
   VkDescriptorSetLayoutSupport *);

struct kraidoc_target {
   struct {
      uint64_t gpu_id, shader_present;
      uint32_t gpu_variant;
   } props;
};

struct panvk_instance {
   struct vk_instance vk;
};

struct panvk_physical_device {
   struct vk_physical_device vk;
   struct { struct kraidoc_target *dev; } kmod;
   const struct pan_model *model;
};

struct panvk_device {
   struct vk_device vk;
};

VK_DEFINE_HANDLE_CASTS(panvk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct panvk_physical_device *
to_panvk_physical_device(struct vk_physical_device *physical)
{
   return container_of(physical, struct panvk_physical_device, vk);
}

#define PANVK_DEBUG(category) false

#endif
