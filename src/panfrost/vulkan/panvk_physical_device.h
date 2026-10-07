/*
 * Copyright © 2021 Collabora Ltd.
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_PHYSICAL_DEVICE_H
#define PANVK_PHYSICAL_DEVICE_H

#include <stdint.h>
#include <sys/types.h>

#include "panvk_instance.h"
#include "panvk_macros.h"

#include "vk_physical_device.h"
#include "vk_sync.h"
#include "vk_sync_timeline.h"
#include "vk_util.h"
#include "wsi_common.h"

#include "lib/kmod/pan_kmod.h"

struct pan_model;
struct pan_blendable_format;
struct pan_format;
struct panvk_instance;

struct panvk_physical_device {
   struct vk_physical_device vk;

   struct {
      struct pan_kmod_dev *dev;
   } kmod;

   const struct pan_model *model;

   union {
      struct {
         struct {
            uint32_t chunk_size;
            uint32_t initial_chunks;
            uint32_t max_chunks;
         } tiler;
      } csf;
   };

   struct {
      dev_t primary_rdev;
      dev_t render_rdev;
   } drm;

   /* Device node path for kbase (non-DRM) devices, e.g. "/dev/mali0";
    * empty for DRM devices.  Logical devices must open() a fresh fd from
    * this path: dup()ing the physical device fd would share the kbase
    * context, whose version handshake can only be performed once. */
   char kbase_node_path[32];

   /* kbase submission model knobs, fixed at physical-device init.
    * single_csg: one persistent CSG per Vulkan queue with one CSI per PanVK
    * subqueue (the panthor layout) instead of one CSG per subqueue.
    * gpu_heap_ops: let the command streams drive the tiler heap
    * (VERTEX_TILER_COMPLETED/FRAGMENT_COMPLETED, firmware chunk recycling)
    * instead of the started-only + periodic heap renewal workaround; only
    * valid with single_csg, where all heap statistics belong to one group. */
   struct {
      bool single_csg;
      bool gpu_heap_ops;
   } kbase;

   struct {
      const struct pan_blendable_format *blendable;
      const struct pan_format *all;
   } formats;

   char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
   uint8_t cache_uuid[VK_UUID_SIZE];

   struct {
      VkMemoryHeap heaps[1];
      uint32_t heap_count;

      VkMemoryType types[4];
      uint32_t type_count;

      uint64_t max_supported_va;
      alignas(8) uint64_t heap_used;
   } memory;

   struct vk_sync_type drm_syncobj_type;
   struct vk_sync_timeline_type sync_timeline_type;
   const struct vk_sync_type *sync_types[3];

   struct wsi_device wsi_device;

   uint64_t compute_core_mask;
   uint64_t fragment_core_mask;
};

VK_DEFINE_HANDLE_CASTS(panvk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

static inline struct panvk_physical_device *
to_panvk_physical_device(struct vk_physical_device *phys_dev)
{
   return container_of(phys_dev, struct panvk_physical_device, vk);
}

float panvk_get_gpu_system_timestamp_period(
   const struct panvk_physical_device *device);

VkResult panvk_physical_device_init(struct panvk_physical_device *device,
                                    struct panvk_instance *instance,
                                    drmDevicePtr drm_device);

#if defined(HAVE_PAN_KMOD_KBASE)
VkResult panvk_physical_device_init_kbase(struct panvk_physical_device *device,
                                          struct panvk_instance *instance,
                                          const char *path);

#define PANVK_KBASE_SYNC_TARGET_COUNT 3

typedef VkResult (*panvk_kbase_sync_wait_func)(
   void *data,
   const uint64_t targets[PANVK_KBASE_SYNC_TARGET_COUNT],
   uint64_t abs_timeout_ns);

/* GPU-visible view of a pending kbase sync: a 64-bit seqno cell (in a
 * BASE_MEM_CSF_EVENT BO) that must reach `value`.  Command streams wait on
 * it with SYNC_WAIT64 (GREATER, value - 1) and the CPU with a KCPU CQS wait
 * or by polling, so cross-queue dependencies never round-trip through a CPU
 * wait before submission. */
struct panvk_kbase_gpu_wait {
   uint64_t addr;
   uint64_t value;
};

/* addrs[i] is the GPU address of the seqno cell backing targets[i], or 0
 * when the target can only be resolved on the CPU (sync_file imports). */
/* owner: the VkDevice whose kbase context the cells in addrs[] belong to
 * (NULL for CPU-only payloads); GPU waits are only handed to queues of that
 * device, every other device falls back to a CPU wait. */
void panvk_kbase_sync_set_pending(
   struct vk_sync *sync, void *data, panvk_kbase_sync_wait_func wait,
   const uint64_t targets[PANVK_KBASE_SYNC_TARGET_COUNT],
   const uint64_t addrs[PANVK_KBASE_SYNC_TARGET_COUNT],
   const struct vk_device *owner);

/* Returns true when the sync can be consumed by the GPU without a CPU wait:
 * either it is already signaled (*count == 0) or every pending target is
 * backed by a seqno cell (filled into waits[]).  Returns false when the
 * caller has to fall back to vk_sync_wait(). */
bool panvk_kbase_sync_get_gpu_waits(
   struct vk_sync *sync, const struct vk_device *device,
   struct panvk_kbase_gpu_wait waits[PANVK_KBASE_SYNC_TARGET_COUNT],
   uint32_t *count);
#endif

void panvk_physical_device_finish(struct panvk_physical_device *device);


VkSampleCountFlags panvk_get_sample_counts(unsigned arch,
                                           unsigned max_tib_size,
                                           unsigned max_cbuf_atts,
                                           unsigned format_size);

VkDeviceSize
panvk_get_max_resource_size(const struct panvk_physical_device *device);

VkDeviceSize
panvk_get_max_buffer_size(const struct panvk_physical_device *device);

#ifdef PAN_ARCH
void panvk_per_arch(get_physical_device_extensions)(
   const struct panvk_physical_device *device,
   const struct panvk_instance *instance,
   struct vk_device_extension_table *ext);

void panvk_per_arch(get_physical_device_features)(
   const struct panvk_instance *instance,
   const struct panvk_physical_device *device, struct vk_features *features);

void panvk_per_arch(get_physical_device_properties)(
   const struct panvk_instance *instance,
   const struct panvk_physical_device *device,
   struct vk_properties *properties);
#endif

#endif
