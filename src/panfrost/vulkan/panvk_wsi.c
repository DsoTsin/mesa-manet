/*
 * Copyright © 2021 Collabora Ltd.
 * Copyright © 2025 Arm Ltd.
 * Copyright © 2026 Pix Philosophy (HK) Limited
 *
 * Derived from tu_wsi.c:
 * Copyright © 2016 Red Hat
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "panvk_wsi.h"
#include "panvk_instance.h"
#include "panvk_physical_device.h"

#include <stdlib.h>
#include <string.h>

#if defined(HAVE_PAN_KMOD_KBASE)
#include "lib/kmod/kbase_kmod.h"
#include "util/detect_os.h"
#endif

#include "vk_util.h"
#include "wsi_common.h"

static VKAPI_PTR PFN_vkVoidFunction
panvk_wsi_proc_addr(VkPhysicalDevice physicalDevice, const char *pName)
{
   VK_FROM_HANDLE(panvk_physical_device, pdevice, physicalDevice);
   struct panvk_instance *instance = to_panvk_instance(pdevice->vk.instance);

   return vk_instance_get_proc_addr_unchecked(&instance->vk, pName);
}

static bool
panvk_can_present_on_device(VkPhysicalDevice pdevice, int fd)
{
   drmDevicePtr device;
   if (drmGetDevice2(fd, 0, &device) != 0)
      return false;
   /* Allow on-device presentation for all devices with bus type PLATFORM.
    * Other device types such as PCI or USB should use the PRIME blit path. */
   bool match = device->bustype == DRM_BUS_PLATFORM;

   drmFreeDevice(&device);

   return match;
}

VkResult
panvk_wsi_init(struct panvk_physical_device *physical_device)
{
   struct panvk_instance *instance =
      to_panvk_instance(physical_device->vk.instance);
   const bool uses_kbase = physical_device->kbase_node_path[0] != '\0';
   const char *dri3_option = getenv("PANVK_KBASE_DRI3");
   /* The proprietary Termux:X11 layer uses WSI_X11_TERMUX=1 as its
    * process-wide switch.  Accept the same switch here when no explicit
    * PanVK override is present, so both ICDs exercise the same raw-FD DRI3
    * transport instead of silently falling back to the slower SHM path. */
   const char *termux_wsi = getenv("WSI_X11_TERMUX");
   const bool termux_raw_dri3 =
      uses_kbase && !dri3_option && termux_wsi && strcmp(termux_wsi, "0") != 0;
   const bool kbase_raw_dri3 =
      termux_raw_dri3 ||
      (uses_kbase && dri3_option &&
       (!strcmp(dri3_option, "raw") || !strcmp(dri3_option, "termux")));
   /* dma-buf backed swapchains (zwp_linux_dmabuf on Wayland, DRI3
    * PixmapFromBuffers on X11) instead of the CPU-copy wl_shm/MIT-SHM
    * fallback.  Default on for Linux userspaces; Android/Termux keeps the
    * explicit opt-in because some Android X servers reject standard
    * PixmapFromBuffers.  PANVK_KBASE_DMABUF=0/1 overrides either way. */
   const char *dmabuf_option = getenv("PANVK_KBASE_DMABUF");
   const bool dmabuf_default =
#if DETECT_OS_ANDROID
      (!dri3_option && termux_raw_dri3) ||
      (dri3_option && strcmp(dri3_option, "0") != 0);
#else
      !(dri3_option && !strcmp(dri3_option, "0"));
#endif
   const bool want_dmabuf =
      dmabuf_option ? strcmp(dmabuf_option, "0") != 0 : dmabuf_default;
   const bool kbase_dmabuf =
#if defined(HAVE_PAN_KMOD_KBASE)
      uses_kbase && want_dmabuf &&
      kbase_kmod_supports_dmabuf(physical_device->kmod.dev);
#else
      false;
#endif
   /* Without a KCPU queue, the frame's completion cannot be exported as a
    * sync_file and attached to the dma-buf, so the fence is waited on the CPU
    * before the buffer is handed over.  WSI skips that wait for images that
    * carry the exported fence (wait_present_before_queue).
    * PANVK_KBASE_WAIT_PRESENT=0 never waits. */
   const char *wait_present_option = getenv("PANVK_KBASE_WAIT_PRESENT");
   const bool wait_present =
      kbase_dmabuf &&
      !(wait_present_option && !strcmp(wait_present_option, "0"));
   VkResult result;

   result = wsi_device_init(&physical_device->wsi_device,
                            panvk_physical_device_to_handle(physical_device),
                            panvk_wsi_proc_addr, &instance->vk.alloc, -1,
                            &instance->drirc.options,
                            &(struct wsi_device_options){
                               .sw_device = uses_kbase && !kbase_dmabuf,
                               .wait_present_before_queue = wait_present,
                               .x11_use_raw_fd_modifier =
                                  kbase_dmabuf && kbase_raw_dri3,
                               /* dma-heap images are CPU-mappable, so X
                                * servers without DRI3 (software Xwayland)
                                * can still be driven through MIT-SHM. */
                               .x11_sw_without_dri3 = kbase_dmabuf,
                            });
   if (result != VK_SUCCESS)
      return result;

#if defined(HAVE_PAN_KMOD_KBASE)
   /* sync_file export on kbase is backed by KCPU fences.  Without a KCPU
    * queue the only export path is sw_sync (absent on most Linux systems),
    * so hide SYNC_FD from WSI: it then skips the dma-buf implicit-sync
    * semaphore and relies on wait_present_before_queue. */
   if (uses_kbase && !kbase_kmod_csf_has_kcpu(physical_device->kmod.dev)) {
      physical_device->wsi_device.semaphore_export_handle_types &=
         ~VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
   }
#endif

   /* kbase syncs carry GPU seqno state in userspace and cannot be copied via
    * DRM syncobj fd payloads.  Keep even empty WSI submits on the real queue
    * so present fences are backed by the kbase submission timeline.
    */
   physical_device->wsi_device.disable_unordered_submits = uses_kbase;

   /* kbase is not a DRM fd.  The default CPU WSI path presents through
    * MIT-SHM.  Android dma-heaps can also provide shareable
    * allocations, but some Android X servers advertise DRI3 while rejecting
    * standard PixmapFromBuffer.  PANVK_KBASE_DRI3=raw (or termux) uses the
    * Termux:X11/Winlator private raw-FD modifier instead of DRM modifiers.
    */
   physical_device->wsi_device.supports_modifiers =
      !uses_kbase || (kbase_dmabuf && !kbase_raw_dri3);
   physical_device->wsi_device.can_present_on_device =
      panvk_can_present_on_device;

   physical_device->vk.wsi_device = &physical_device->wsi_device;

   return VK_SUCCESS;
}

void
panvk_wsi_finish(struct panvk_physical_device *physical_device)
{
   struct panvk_instance *instance =
      to_panvk_instance(physical_device->vk.instance);

   physical_device->vk.wsi_device = NULL;
   wsi_device_finish(&physical_device->wsi_device, &instance->vk.alloc);
}
