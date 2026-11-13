#include "nvkmd_wddm_ops.h"
#include "util/os_time.h"
#include "vk_sync.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(struct nvkmd_wddm_device_info) == 168, "device info ABI");

VkResult nvkmd_wddm_probe_wait(struct nvkmd_ctx *ctx,
                              const struct nvkmd_wddm_point *point);

static VkResult
probe_sync(const struct vk_sync_type *type, enum vk_sync_flags flags,
            struct vk_sync **out)
{
   struct vk_sync *sync = calloc(1, type->size);
   if (!sync)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = vk_sync_init(NULL, sync, type, flags, 0);
   if (result != VK_SUCCESS) {
      free(sync);
      return result;
   }
   *out = sync;
   return VK_SUCCESS;
}

static int
probe_timeline(struct nvkmd_dev *dev, const struct vk_sync_type *type)
{
   struct nvkmd_ctx *producer = NULL, *consumer = NULL;
   struct vk_sync *timeline = NULL, *producer_done = NULL, *consumer_done = NULL;
   struct nvkmd_wddm_fence *gate = NULL;
   struct nvkmd_wddm_point *gate_point = NULL;
   VkResult result = probe_sync(type, VK_SYNC_IS_TIMELINE, &timeline);
   uint64_t middle = 0, final = 0;
   bool blocked = false;
   bool gate_released = false;
   if (result != VK_SUCCESS)
      goto cleanup;
   result = probe_sync(type, 0, &producer_done);
   if (result != VK_SUCCESS)
      goto cleanup;
   result = probe_sync(type, 0, &consumer_done);
   if (result != VK_SUCCESS)
      goto cleanup;
   result = nvkmd_wddm_fence_create(nvkmd_wddm_native_device(dev), 0, &gate);
   if (result != VK_SUCCESS)
      goto cleanup;
   result = nvkmd_wddm_fence_point(gate, 1, &gate_point);
   if (result != VK_SUCCESS)
      goto cleanup;
   result = nvkmd_dev_create_ctx(dev, NULL,
      NVKMD_ENGINE_3D | NVKMD_ENGINE_COMPUTE, &producer);
   if (result != VK_SUCCESS)
      goto cleanup;
   result = nvkmd_dev_create_ctx(dev, NULL,
      NVKMD_ENGINE_3D | NVKMD_ENGINE_COMPUTE, &consumer);
   if (result != VK_SUCCESS)
      goto cleanup;
   result = nvkmd_wddm_probe_wait(producer, gate_point);
   if (result != VK_SUCCESS)
      goto cleanup;
   const struct vk_sync_signal producer_signals[] = {
      {.sync = timeline, .stage_mask = ~0ull, .signal_value = 10},
      {.sync = producer_done, .stage_mask = ~0ull},
   };
   result = nvkmd_ctx_signal(producer, NULL, 2, producer_signals);
   if (result != VK_SUCCESS)
      goto cleanup;
   const struct vk_sync_wait wait = {
      .sync = timeline, .stage_mask = ~0ull, .wait_value = 5,
   };
   result = nvkmd_ctx_wait(consumer, NULL, 1, &wait);
   if (result != VK_SUCCESS)
      goto cleanup;
   const struct vk_sync_signal consumer_signal = {
      .sync = consumer_done, .stage_mask = ~0ull,
   };
   result = nvkmd_ctx_signal(consumer, NULL, 1, &consumer_signal);
   if (result != VK_SUCCESS)
      goto cleanup;
   result = vk_sync_signal(NULL, timeline, 5);
   if (result != VK_SUCCESS)
      goto cleanup;
   result = vk_sync_wait(NULL, consumer_done, 0, VK_SYNC_WAIT_COMPLETE,
                         os_time_get_nano() + UINT64_C(500000000));
   if (result != VK_SUCCESS)
      goto cleanup;
   result = vk_sync_get_value(NULL, timeline, &middle);
   if (result != VK_SUCCESS)
      goto cleanup;
   blocked = vk_sync_wait(NULL, producer_done, 0, VK_SYNC_WAIT_COMPLETE,
                          os_time_get_nano()) == VK_TIMEOUT;
   result = nvkmd_wddm_fence_signal_cpu_value(gate, 1);
   if (result != VK_SUCCESS)
      goto cleanup;
   gate_released = true;
   result = vk_sync_wait(NULL, producer_done, 0, VK_SYNC_WAIT_COMPLETE,
                         os_time_get_nano() + UINT64_C(2000000000));
   if (result != VK_SUCCESS)
      goto cleanup;
   result = vk_sync_get_value(NULL, timeline, &final);
cleanup:
   if (gate && !gate_released)
      nvkmd_wddm_fence_signal_cpu_value(gate, 1);
   if (consumer)
      nvkmd_ctx_destroy(consumer);
   if (producer)
      nvkmd_ctx_destroy(producer);
   nvkmd_wddm_point_destroy(gate_point);
   nvkmd_wddm_fence_destroy(gate);
   struct vk_sync *syncs[] = {timeline, producer_done, consumer_done};
   for (unsigned i = 0; i < 3; i++) {
      if (syncs[i]) {
         vk_sync_finish(NULL, syncs[i]);
         free(syncs[i]);
      }
   }
   printf("scope=c_nvkmd_vk_sync_bridge result=%d host_value=%" PRIu64
          " producer_blocked=%u final_value=%" PRIu64 " passed=%u\n",
          result, middle, blocked, final,
          result == VK_SUCCESS && middle == 5 && blocked && final == 10);
   return result != VK_SUCCESS || middle != 5 || !blocked || final != 10;
}

int
main(int argc, char **argv)
{
   setvbuf(stdout, NULL, _IONBF, 0);
   setvbuf(stderr, NULL, _IONBF, 0);
   const bool timeline = argc == 2 && strcmp(argv[1], "--timeline") == 0;
   uint32_t count = 0;
   VkResult result = nvkmd_wddm_enumerate(&count, NULL);
   if (result != VK_SUCCESS)
      return 1;
   struct nvkmd_wddm_adapter_info *adapters = calloc(count, sizeof(*adapters));
   if (count && !adapters)
      return 2;
   result = nvkmd_wddm_enumerate(&count, adapters);
   if (result != VK_SUCCESS)
      return 3;
   for (uint32_t i = 0; i < count; i++) {
      if (adapters[i].vendor_id != NVIDIA_VENDOR_ID)
         continue;
      struct nvkmd_pdev *pdev;
      result = nvkmd_wddm_create_pdev(&adapters[i], NULL, 0, &pdev);
      if (result != VK_SUCCESS) {
         printf("pdev=%d\n", result);
         return 4;
      }
      printf("pdev SM%u chipset=%x gpc=%u tpc=%u name=%s\n",
             pdev->dev_info.sm, pdev->dev_info.chipset,
             pdev->dev_info.gpc_count, pdev->dev_info.tpc_count,
             pdev->dev_info.device_name);
      struct nvkmd_dev *dev;
      result = nvkmd_pdev_create_dev(pdev, NULL, &dev);
      if (result != VK_SUCCESS)
         return 5;
      struct nvkmd_mem *mem;
      result = nvkmd_dev_alloc_mem(dev, NULL, 65536, 65536,
         NVKMD_MEM_GART | NVKMD_MEM_CAN_MAP, &mem);
      if (result != VK_SUCCESS) {
         printf("memory=%d\n", result);
         return 6;
      }
      void *map;
      result = nvkmd_mem_map(mem, NULL, NVKMD_MEM_MAP_RDWR, NULL, &map);
      if (result != VK_SUCCESS)
         return 7;
      *(volatile uint32_t *)map = UINT32_C(0x4e564b32);
      if (*(volatile uint32_t *)map != UINT32_C(0x4e564b32))
         return 8;
      printf("nvkmd memory=%" PRIu64 " va=%" PRIx64 " map=ok\n",
             mem->size_B, mem->va->addr);
      nvkmd_mem_unmap(mem, 0);
      nvkmd_mem_unref(mem);
      const enum nvkmd_engines engines[] = {
         NVKMD_ENGINE_COPY,
         NVKMD_ENGINE_COPY | NVKMD_ENGINE_3D | NVKMD_ENGINE_COMPUTE,
      };
      for (unsigned n = 0; n < 2; n++) {
         struct nvkmd_ctx *ctx;
         result = dev->ops->create_ctx(dev, NULL, engines[n], &ctx);
         if (result != VK_SUCCESS) {
            printf("context engines=%u result=%d\n", engines[n], result);
            return 9;
         }
         result = nvkmd_ctx_sync(ctx, NULL);
         if (result != VK_SUCCESS)
            return 10;
         nvkmd_ctx_destroy(ctx);
         printf("context engines=%u hardware=1 sync=ok\n", engines[n]);
      }
      if (timeline && probe_timeline(dev, pdev->sync_types[0]))
         return 11;
      nvkmd_dev_destroy(dev);
      nvkmd_pdev_destroy(pdev);
   }
   free(adapters);
   return 0;
}
