#include "../../nvk_instance.h"
#include "../../nvk_physical_device.h"
#include "nvkmd_wddm_ops.h"

#include <stdlib.h>

VkResult
nvk_enumerate_wddm_physical_devices(struct vk_instance *instance)
{
   struct nvk_instance *nvk = (struct nvk_instance *)instance;
   uint32_t count = 0;
   VkResult result = nvkmd_wddm_enumerate(&count, NULL);
   if (result != VK_SUCCESS || count == 0)
      return result;
   struct nvkmd_wddm_adapter_info *adapters = calloc(count, sizeof(*adapters));
   if (!adapters)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   result = nvkmd_wddm_enumerate(&count, adapters);
   if (result != VK_SUCCESS)
      goto done;
   for (uint32_t i = 0; i < count; i++) {
      if (adapters[i].vendor_id != NVIDIA_VENDOR_ID)
         continue;
      struct nvkmd_pdev *nvkmd;
      result = nvkmd_wddm_create_pdev(&adapters[i], &instance->base,
                                      nvk->debug_flags, &nvkmd);
      if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
         result = VK_SUCCESS;
         continue;
      }
      if (result != VK_SUCCESS)
         break;
      struct vk_physical_device *pdev;
      result = nvk_create_physical_device_for_nvkmd(instance, nvkmd, &pdev);
      if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
         result = VK_SUCCESS;
         continue;
      }
      if (result != VK_SUCCESS)
         break;
      list_addtail(&pdev->link, &instance->physical_devices.list);
   }
done:
   free(adapters);
   return result;
}
