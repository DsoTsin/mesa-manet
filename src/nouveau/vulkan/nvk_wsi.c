/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "nvk_wsi.h"
#include "nvk_instance.h"
#include "nvkmd/nvkmd.h"
#include "wsi_common.h"

#ifdef _WIN32
#include "nvk_device.h"
#include "nvk_device_memory.h"
#include "nvkmd/wddm/nvkmd_wddm_ops.h"

static VkResult
nvk_wsi_create_native_swapchain(VkDevice device, uintptr_t window, void **out)
{
   VK_FROM_HANDLE(nvk_device, dev, device);
   return nvkmd_wddm_presenter_create(nvkmd_wddm_native_device(dev->nvkmd),
      window, (struct nvkmd_wddm_presenter **)out);
}

static void
nvk_wsi_destroy_native_swapchain(void *swapchain)
{
   nvkmd_wddm_presenter_destroy(swapchain);
}

static VkResult
nvk_wsi_create_native_image_memory(VkDevice device, void *swapchain,
                                    VkFormat format, uint32_t width,
                                    uint32_t height,
                                    const VkMemoryAllocateInfo *info,
                                    const VkAllocationCallbacks *alloc,
                                    void **image_out, uint32_t *pitch,
                                    VkDeviceMemory *out)
{
   VK_FROM_HANDLE(nvk_device, dev, device);
   struct nvk_physical_device *pdev = nvk_device_physical_mut(dev);
   struct nvkmd_wddm_present_image *image;
   VkResult result = nvkmd_wddm_present_image_create(swapchain, width, height,
                                                    format, &image);
   if (result != VK_SUCCESS)
      return result;
   struct nvkmd_wddm_memory *native;
   uint64_t size;
   result = nvkmd_wddm_present_image_memory(image, &native, &size, pitch);
   if (result != VK_SUCCESS)
      goto fail_image;
   if (size < info->allocationSize) {
      result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
      nvkmd_wddm_memory_destroy(native);
      goto fail_image;
   }
   struct nvk_device_memory *mem = vk_device_memory_create(&dev->vk, info,
                                                           alloc, sizeof(*mem));
   if (!mem) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      nvkmd_wddm_memory_destroy(native);
      goto fail_image;
   }
   result = nvkmd_wddm_wrap_memory(dev->nvkmd, native, size, &mem->mem);
   if (result != VK_SUCCESS) {
      vk_device_memory_destroy(&dev->vk, alloc, &mem->vk);
      goto fail_image;
   }
   const VkMemoryType *type = &pdev->mem_types[info->memoryTypeIndex];
   p_atomic_add(&pdev->mem_heaps[type->heapIndex].used, size);
   *out = nvk_device_memory_to_handle(mem);
   *image_out = image;
   return VK_SUCCESS;
fail_image:
   nvkmd_wddm_present_image_destroy(image);
   return result;
}

static void
nvk_wsi_destroy_native_image(void *image)
{
   nvkmd_wddm_present_image_destroy(image);
}

static VkResult
nvk_wsi_present_native_image(void *swapchain, void *image)
{
   VkResult result = nvkmd_wddm_presenter_present(swapchain, image, 5000000000ull);
   if (result == VK_TIMEOUT)
      return VK_ERROR_DEVICE_LOST;
   if (result == VK_NOT_READY)
      return VK_ERROR_OUT_OF_DATE_KHR;
   return result;
}
#endif

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
nvk_wsi_proc_addr(VkPhysicalDevice physicalDevice, const char *pName)
{
   VK_FROM_HANDLE(nvk_physical_device, pdev, physicalDevice);
   return vk_instance_get_proc_addr_unchecked(pdev->vk.instance, pName);
}

VkResult
nvk_init_wsi(struct nvk_physical_device *pdev)
{
   VkResult result;

   int display_fd = -1;
   if (pdev->vk.instance->enabled_extensions.KHR_display)
      display_fd = nvkmd_pdev_get_drm_primary_fd(pdev->nvkmd);
   struct wsi_device_options wsi_options = {
      .sw_device = false
#ifdef _WIN32
      , .wait_present_before_queue = true
#endif
   };
   result = wsi_device_init(&pdev->wsi_device,
                            nvk_physical_device_to_handle(pdev),
                            nvk_wsi_proc_addr, &pdev->vk.instance->alloc,
                            display_fd,
                            &nvk_physical_device_instance(pdev)->drirc.options,
                            &wsi_options);
   if (result != VK_SUCCESS)
      return result;

#ifdef _WIN32
   pdev->wsi_device.win32.create_native_swapchain =
      nvk_wsi_create_native_swapchain;
   pdev->wsi_device.win32.destroy_native_swapchain =
      nvk_wsi_destroy_native_swapchain;
   pdev->wsi_device.win32.create_native_image_memory =
      nvk_wsi_create_native_image_memory;
   pdev->wsi_device.win32.destroy_native_image = nvk_wsi_destroy_native_image;
   pdev->wsi_device.win32.present_native_image = nvk_wsi_present_native_image;
#endif

   pdev->wsi_device.supports_scanout = false;
   pdev->wsi_device.supports_modifiers =
      pdev->vk.supported_extensions.table.EXT_image_drm_format_modifier;

   pdev->vk.wsi_device = &pdev->wsi_device;

   return result;
}

void
nvk_finish_wsi(struct nvk_physical_device *pdev)
{
   pdev->vk.wsi_device = NULL;
   wsi_device_finish(&pdev->wsi_device, &pdev->vk.instance->alloc);
}
