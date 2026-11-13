#include "nvkmd_wddm_ops.h"
#include "util/cnd_monotonic.h"
#include "util/os_time.h"
#include "util/u_math.h"
#include "vk_sync.h"

#include <stdlib.h>
#include <string.h>

struct wddm_pdev {
   struct nvkmd_pdev base;
   struct nvkmd_wddm_luid luid;
   const struct vk_sync_type *sync_types[2];
};

struct wddm_dev {
   struct nvkmd_dev base;
   struct nvkmd_wddm_device *native;
   struct list_head vas;
};

struct wddm_mem {
   struct nvkmd_mem base;
   struct nvkmd_wddm_memory *native;
};

struct wddm_mapping {
   struct list_head link;
   uint64_t offset;
   uint64_t size;
   struct nvkmd_wddm_binding *native;
};

struct wddm_va {
   struct nvkmd_va base;
   struct nvkmd_wddm_va *native;
   struct list_head link;
   struct list_head mappings;
};

struct wddm_completion {
   uint32_t refs;
   struct nvkmd_wddm_point *native;
};

struct wddm_ctx {
   struct nvkmd_ctx base;
   struct nvkmd_wddm_context *native;
   struct wddm_completion *last;
   bool pending;
};

struct wddm_sync {
   struct vk_sync base;
   mtx_t mutex;
   struct u_cnd_monotonic cond;
   bool signaled;
   struct wddm_completion *completion;
   struct nvkmd_wddm_fence *fence;
   uint64_t initial_value;
   uint64_t pending_value;
};

static const struct nvkmd_pdev_ops wddm_pdev_ops;
static const struct nvkmd_dev_ops wddm_dev_ops;
static const struct nvkmd_mem_ops wddm_mem_ops;
static const struct nvkmd_va_ops wddm_va_ops;
static const struct nvkmd_ctx_ops wddm_ctx_ops;
static const struct vk_sync_type wddm_sync_type;

const struct nvkmd_wddm_device *
nvkmd_wddm_native_device(struct nvkmd_dev *dev)
{
   if (!dev || dev->ops != &wddm_dev_ops)
      return NULL;
   return ((struct wddm_dev *)dev)->native;
}

VkResult
nvkmd_wddm_ctx_get_reflected_sked_address(struct nvkmd_ctx *base, uint64_t *out)
{
   if (!out)
      return VK_ERROR_UNKNOWN;
   *out = 0;
   if (!base || base->ops != &wddm_ctx_ops)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   return nvkmd_wddm_context_reflected_sked_address(
      ((struct wddm_ctx *)base)->native, out);
}

static struct wddm_completion *
completion_ref(struct wddm_completion *completion)
{
   if (completion)
      p_atomic_inc(&completion->refs);
   return completion;
}

static void
completion_unref(struct wddm_completion *completion)
{
   if (completion && p_atomic_dec_zero(&completion->refs)) {
      nvkmd_wddm_point_destroy(completion->native);
      free(completion);
   }
}

static uint64_t
remaining_timeout(uint64_t absolute)
{
   if (absolute == UINT64_MAX)
      return UINT64_MAX;
   const uint64_t now = os_time_get_nano();
   return absolute > now ? absolute - now : 0;
}

static VkResult
wddm_sync_init(struct vk_device *device, struct vk_sync *base, uint64_t initial)
{
   struct wddm_sync *sync = (struct wddm_sync *)base;
   if (mtx_init(&sync->mutex, mtx_plain) != thrd_success)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   if (u_cnd_monotonic_init(&sync->cond) != thrd_success) {
      mtx_destroy(&sync->mutex);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   sync->signaled = initial != 0;
   sync->initial_value = initial;
   sync->pending_value = initial;
   return VK_SUCCESS;
}

static void
wddm_sync_finish(struct vk_device *device, struct vk_sync *base)
{
   struct wddm_sync *sync = (struct wddm_sync *)base;
   completion_unref(sync->completion);
   nvkmd_wddm_fence_destroy(sync->fence);
   u_cnd_monotonic_destroy(&sync->cond);
   mtx_destroy(&sync->mutex);
}

static VkResult
wddm_sync_signal(struct vk_device *device, struct vk_sync *base, uint64_t value)
{
   struct wddm_sync *sync = (struct wddm_sync *)base;
   mtx_lock(&sync->mutex);
   if (base->flags & VK_SYNC_IS_TIMELINE) {
      VkResult result = VK_SUCCESS;
      if (sync->fence)
         result = nvkmd_wddm_fence_signal_cpu_value(sync->fence, value);
      else
         sync->initial_value = MAX2(sync->initial_value, value);
      if (result == VK_SUCCESS)
         u_cnd_monotonic_broadcast(&sync->cond);
      mtx_unlock(&sync->mutex);
      return result;
   }
   completion_unref(sync->completion);
   sync->completion = NULL;
   sync->signaled = true;
   u_cnd_monotonic_broadcast(&sync->cond);
   mtx_unlock(&sync->mutex);
   return VK_SUCCESS;
}

static VkResult
wddm_sync_reset(struct vk_device *device, struct vk_sync *base)
{
   struct wddm_sync *sync = (struct wddm_sync *)base;
   mtx_lock(&sync->mutex);
   completion_unref(sync->completion);
   sync->completion = NULL;
   sync->signaled = false;
   mtx_unlock(&sync->mutex);
   return VK_SUCCESS;
}

static VkResult
wddm_sync_move(struct vk_device *device, struct vk_sync *dst_base,
               struct vk_sync *src_base)
{
   struct wddm_sync *src = (struct wddm_sync *)src_base;
   struct wddm_sync *dst = (struct wddm_sync *)dst_base;
   mtx_lock(&src->mutex);
   struct wddm_completion *completion = src->completion;
   const bool signaled = src->signaled;
   src->completion = NULL;
   src->signaled = false;
   mtx_unlock(&src->mutex);
   mtx_lock(&dst->mutex);
   completion_unref(dst->completion);
   dst->completion = completion;
   dst->signaled = signaled;
   u_cnd_monotonic_broadcast(&dst->cond);
   mtx_unlock(&dst->mutex);
   return VK_SUCCESS;
}

static VkResult
wddm_timeline_native_locked(struct wddm_sync *sync,
                            const struct nvkmd_wddm_device *device)
{
   if (sync->fence)
      return VK_SUCCESS;
   return nvkmd_wddm_fence_create(device, sync->initial_value, &sync->fence);
}

static VkResult
wddm_timeline_value_locked(struct wddm_sync *sync, uint64_t *value)
{
   if (sync->fence)
      return nvkmd_wddm_fence_get_value(sync->fence, value);
   *value = sync->initial_value;
   return VK_SUCCESS;
}

static VkResult
wddm_sync_get_value(struct vk_device *device, struct vk_sync *base, uint64_t *value)
{
   struct wddm_sync *sync = (struct wddm_sync *)base;
   mtx_lock(&sync->mutex);
   VkResult result = wddm_timeline_value_locked(sync, value);
   mtx_unlock(&sync->mutex);
   return result;
}

static VkResult
wddm_timeline_wait(struct wddm_sync *sync, uint64_t value,
                   enum vk_sync_wait_flags flags, uint64_t timeout)
{
   for (;;) {
      mtx_lock(&sync->mutex);
      uint64_t current;
      VkResult result = wddm_timeline_value_locked(sync, &current);
      if (result != VK_SUCCESS || current >= value ||
          ((flags & VK_SYNC_WAIT_PENDING) && sync->pending_value >= value)) {
         mtx_unlock(&sync->mutex);
         return result;
      }
      if (sync->fence && !(flags & VK_SYNC_WAIT_PENDING)) {
         struct nvkmd_wddm_point *point;
         result = nvkmd_wddm_fence_point(sync->fence, value, &point);
         mtx_unlock(&sync->mutex);
         if (result != VK_SUCCESS)
            return result;
         result = nvkmd_wddm_point_wait(point, remaining_timeout(timeout));
         nvkmd_wddm_point_destroy(point);
         return result;
      } else {
         int wait_result;
         if (timeout == UINT64_MAX) {
            wait_result = u_cnd_monotonic_wait(&sync->cond, &sync->mutex);
         } else {
            struct timespec deadline = {
               .tv_sec = timeout / 1000000000,
               .tv_nsec = timeout % 1000000000,
            };
            wait_result = u_cnd_monotonic_timedwait(&sync->cond, &sync->mutex, &deadline);
         }
         mtx_unlock(&sync->mutex);
         if (wait_result != thrd_success)
            return wait_result == thrd_timedout ? VK_TIMEOUT : VK_ERROR_UNKNOWN;
      }
   }
}

static VkResult
wddm_sync_wait(struct vk_device *device, struct vk_sync *base, uint64_t value,
               enum vk_sync_wait_flags flags, uint64_t timeout)
{
   struct wddm_sync *sync = (struct wddm_sync *)base;
   if (base->flags & VK_SYNC_IS_TIMELINE)
      return wddm_timeline_wait(sync, value, flags, timeout);
   mtx_lock(&sync->mutex);
   while (!sync->signaled && !sync->completion) {
      int result;
      if (timeout == UINT64_MAX) {
         result = u_cnd_monotonic_wait(&sync->cond, &sync->mutex);
      } else {
         struct timespec deadline = {
            .tv_sec = timeout / 1000000000,
            .tv_nsec = timeout % 1000000000,
         };
         result = u_cnd_monotonic_timedwait(&sync->cond, &sync->mutex, &deadline);
      }
      if (result != thrd_success) {
         mtx_unlock(&sync->mutex);
         return result == thrd_timedout ? VK_TIMEOUT : VK_ERROR_UNKNOWN;
      }
   }
   struct wddm_completion *completion = completion_ref(sync->completion);
   mtx_unlock(&sync->mutex);
   if (completion == NULL || (flags & VK_SYNC_WAIT_PENDING)) {
      completion_unref(completion);
      return VK_SUCCESS;
   }
   VkResult result = nvkmd_wddm_point_wait(completion->native,
                                          remaining_timeout(timeout));
   completion_unref(completion);
   return result;
}

static const struct vk_sync_type wddm_sync_type = {
   .size = sizeof(struct wddm_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_TIMELINE |
               VK_SYNC_FEATURE_GPU_MULTI_WAIT | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_RESET |
               VK_SYNC_FEATURE_CPU_SIGNAL | VK_SYNC_FEATURE_WAIT_PENDING,
   .init = wddm_sync_init,
   .finish = wddm_sync_finish,
   .signal = wddm_sync_signal,
   .get_value = wddm_sync_get_value,
   .reset = wddm_sync_reset,
   .move = wddm_sync_move,
   .wait = wddm_sync_wait,
};

static void
wddm_va_free(struct nvkmd_va *base)
{
   struct wddm_va *va = (struct wddm_va *)base;
   simple_mtx_lock(&base->dev->mems_mutex);
   list_del(&va->link);
   list_for_each_entry_safe(struct wddm_mapping, map, &va->mappings, link) {
      list_del(&map->link);
      nvkmd_wddm_binding_destroy(map->native);
      free(map);
   }
   simple_mtx_unlock(&base->dev->mems_mutex);
   nvkmd_wddm_va_destroy(va->native);
   free(va);
}

static VkResult
wddm_va_unbind(struct nvkmd_va *base, struct vk_object_base *log_obj,
               uint64_t offset, uint64_t size)
{
   struct wddm_va *va = (struct wddm_va *)base;
   simple_mtx_lock(&base->dev->mems_mutex);
   list_for_each_entry(struct wddm_mapping, map, &va->mappings, link) {
      if (map->offset < offset + size && offset < map->offset + map->size &&
          (map->offset < offset || map->offset + map->size > offset + size)) {
         simple_mtx_unlock(&base->dev->mems_mutex);
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
   }
   list_for_each_entry_safe(struct wddm_mapping, map, &va->mappings, link) {
      if (map->offset >= offset && map->offset + map->size <= offset + size) {
         list_del(&map->link);
         nvkmd_wddm_binding_destroy(map->native);
         free(map);
      }
   }
   simple_mtx_unlock(&base->dev->mems_mutex);
   return VK_SUCCESS;
}

static VkResult
wddm_va_bind(struct nvkmd_va *base, struct vk_object_base *log_obj,
             uint64_t offset, struct nvkmd_mem *mem_base,
             uint64_t mem_offset, uint64_t size)
{
   struct wddm_va *va = (struct wddm_va *)base;
   struct wddm_mem *mem = (struct wddm_mem *)mem_base;
   struct wddm_mapping *map = calloc(1, sizeof(*map));
   if (!map)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = wddm_va_unbind(base, log_obj, offset, size);
   if (result != VK_SUCCESS) {
      free(map);
      return result;
   }
   result = nvkmd_wddm_va_bind_executable(va->native, mem->native, offset,
                                         mem_offset, size, &map->native);
   if (result != VK_SUCCESS) {
      free(map);
      return result;
   }
   map->offset = offset;
   map->size = size;
   simple_mtx_lock(&base->dev->mems_mutex);
   list_addtail(&map->link, &va->mappings);
   simple_mtx_unlock(&base->dev->mems_mutex);
   return VK_SUCCESS;
}

static const struct nvkmd_va_ops wddm_va_ops = {
   .free = wddm_va_free,
   .bind_mem = wddm_va_bind,
   .unbind = wddm_va_unbind,
};

static VkResult
wddm_alloc_va(struct nvkmd_dev *base, struct vk_object_base *log_obj,
              enum nvkmd_va_flags flags, uint8_t kind, uint64_t size,
              uint64_t alignment, uint64_t fixed, struct nvkmd_va **out)
{
   if ((kind != 0 && kind != 6) ||
       (flags & (NVKMD_VA_SPARSE | NVKMD_VA_REPLAY)))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct wddm_dev *dev = (struct wddm_dev *)base;
   struct wddm_va *va = calloc(1, sizeof(*va));
   if (!va)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   alignment = MAX2(alignment, base->pdev->bind_align_B);
   size = align64(size, base->pdev->bind_align_B);
   VkResult result = nvkmd_wddm_va_reserve(dev->native, size, alignment,
      (flags & NVKMD_VA_ALLOC_FIXED) ? fixed : 0, &va->native);
   if (result != VK_SUCCESS)
      goto fail;
   va->base = (struct nvkmd_va) {
      .ops = &wddm_va_ops, .dev = base, .flags = flags,
      .size_B = size, .pte_kind = kind,
   };
   result = nvkmd_wddm_va_address(va->native, &va->base.addr);
   if (result != VK_SUCCESS) {
      nvkmd_wddm_va_destroy(va->native);
      goto fail;
   }
   list_inithead(&va->mappings);
   simple_mtx_lock(&base->mems_mutex);
   list_addtail(&va->link, &dev->vas);
   simple_mtx_unlock(&base->mems_mutex);
   *out = &va->base;
   return VK_SUCCESS;
fail:
   free(va);
   return result;
}

static void
wddm_mem_free(struct nvkmd_mem *base)
{
   struct wddm_mem *mem = (struct wddm_mem *)base;
   if (base->va)
      nvkmd_va_free(base->va);
   nvkmd_wddm_memory_destroy(mem->native);
   simple_mtx_destroy(&base->map_mutex);
   free(mem);
}

static VkResult
wddm_mem_map(struct nvkmd_mem *base, struct vk_object_base *log_obj,
             enum nvkmd_mem_map_flags flags, void *fixed, void **out)
{
   if (fixed || (flags & NVKMD_MEM_MAP_FIXED))
      return VK_ERROR_MEMORY_MAP_FAILED;
   return nvkmd_wddm_memory_map(((struct wddm_mem *)base)->native, out);
}

static void
wddm_mem_unmap(struct nvkmd_mem *base, enum nvkmd_mem_map_flags flags, void *map)
{
}

static uint32_t
wddm_mem_handle(struct nvkmd_mem *base)
{
   return (uint32_t)(uintptr_t)((struct wddm_mem *)base)->native;
}

static const struct nvkmd_mem_ops wddm_mem_ops = {
   .free = wddm_mem_free, .map = wddm_mem_map, .unmap = wddm_mem_unmap,
   .log_handle = wddm_mem_handle,
};

static VkResult
wddm_wrap_memory(struct nvkmd_dev *base, struct vk_object_base *log_obj,
                 struct nvkmd_wddm_memory *owned, uint64_t size,
                 uint64_t alignment, enum nvkmd_mem_flags flags,
                 struct nvkmd_mem **out)
{
   struct wddm_mem *mem = calloc(1, sizeof(*mem));
   if (!mem) {
      nvkmd_wddm_memory_destroy(owned);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   mem->native = owned;
   nvkmd_mem_init(base, &mem->base, &wddm_mem_ops,
      flags, size, base->pdev->bind_align_B);
   const enum nvkmd_va_flags va_flags =
      (flags & NVKMD_MEM_GART) ? NVKMD_VA_GART : 0;
   VkResult result = wddm_alloc_va(base, log_obj, va_flags, 0, size,
                                  alignment, 0, &mem->base.va);
   if (result != VK_SUCCESS)
      goto fail_native;
   result = wddm_va_bind(mem->base.va, log_obj, 0, &mem->base, 0, size);
   if (result != VK_SUCCESS) {
      wddm_va_free(mem->base.va);
      goto fail_native;
   }
   *out = &mem->base;
   return VK_SUCCESS;
fail_native:
   simple_mtx_destroy(&mem->base.map_mutex);
   nvkmd_wddm_memory_destroy(mem->native);
   free(mem);
   return result;
}

VkResult
nvkmd_wddm_wrap_memory(struct nvkmd_dev *base,
                       struct nvkmd_wddm_memory *owned,
                       uint64_t size, struct nvkmd_mem **out)
{
   VkResult result = wddm_wrap_memory(base, NULL, owned, size,
      base->pdev->bind_align_B,
      NVKMD_MEM_GART | NVKMD_MEM_CAN_MAP | NVKMD_MEM_COHERENT, out);
   if (result == VK_SUCCESS) {
      simple_mtx_lock(&base->mems_mutex);
      list_addtail(&(*out)->link, &base->mems);
      simple_mtx_unlock(&base->mems_mutex);
   }
   return result;
}

static VkResult
wddm_alloc_mem(struct nvkmd_dev *base, struct vk_object_base *log_obj,
               uint64_t size, uint64_t alignment, enum nvkmd_mem_flags flags,
               struct nvkmd_mem **out)
{
   if ((flags & NVKMD_MEM_SHARED) ||
       ((flags & NVKMD_MEM_VRAM) && ((flags & NVKMD_MEM_CAN_MAP) ||
                                   base->pdev->dev_info.vram_size_B == 0)))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct nvkmd_wddm_memory *native;
   uint32_t kind;
   const bool local = (flags & (NVKMD_MEM_LOCAL | NVKMD_MEM_VRAM)) &&
                      !(flags & NVKMD_MEM_CAN_MAP) &&
                      base->pdev->dev_info.vram_size_B > 0;
   if (local)
      alignment = MAX2(alignment, 65536);
   size = align64(size, local ? 65536 : base->pdev->bind_align_B);
   VkResult result;
   if (local) {
      result = nvkmd_wddm_nvidia_local_memory_create(
         ((struct wddm_dev *)base)->native, size, &native, &kind);
      flags = (flags & ~NVKMD_MEM_PLACEMENT_FLAGS) | NVKMD_MEM_VRAM;
   } else {
      result = nvkmd_wddm_nvidia_host_memory_create(
         ((struct wddm_dev *)base)->native, size, &native, &kind);
      flags |= NVKMD_MEM_GART | NVKMD_MEM_CAN_MAP | NVKMD_MEM_COHERENT;
   }
   if (result != VK_SUCCESS)
      return result;
   if (kind != 6) {
      nvkmd_wddm_memory_destroy(native);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   return wddm_wrap_memory(base, log_obj, native, size, alignment, flags, out);
}

static VkResult
wddm_ctx_flush(struct nvkmd_ctx *base, struct vk_object_base *log_obj)
{
   struct wddm_ctx *ctx = (struct wddm_ctx *)base;
   VkResult collected = nvkmd_wddm_context_collect(ctx->native);
   if (collected != VK_SUCCESS)
      return collected;
   if (!ctx->pending)
      return VK_SUCCESS;
   struct wddm_completion *completion = calloc(1, sizeof(*completion));
   if (!completion)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct wddm_dev *dev = (struct wddm_dev *)base->dev;
   VkResult result = VK_SUCCESS;
   simple_mtx_lock(&base->dev->mems_mutex);
   list_for_each_entry(struct wddm_va, va, &dev->vas, link) {
      list_for_each_entry(struct wddm_mapping, map, &va->mappings, link) {
         result = nvkmd_wddm_context_retain(ctx->native, map->native);
         if (result != VK_SUCCESS)
            goto unlock;
      }
   }
unlock:
   simple_mtx_unlock(&base->dev->mems_mutex);
   if (result == VK_SUCCESS)
      result = nvkmd_wddm_context_flush(ctx->native, &completion->native);
   if (result != VK_SUCCESS) {
      free(completion);
      return result;
   }
   completion->refs = 1;
   completion_unref(ctx->last);
   ctx->last = completion;
   ctx->pending = false;
   return VK_SUCCESS;
}

static VkResult
wddm_ctx_wait(struct nvkmd_ctx *base, struct vk_object_base *log_obj,
              uint32_t count, const struct vk_sync_wait *waits)
{
   struct wddm_ctx *ctx = (struct wddm_ctx *)base;
   VkResult result = wddm_ctx_flush(base, log_obj);
   if (result != VK_SUCCESS)
      return result;
   for (uint32_t i = 0; i < count; i++) {
      if (waits[i].sync->type != &wddm_sync_type)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      result = wddm_sync_wait(NULL, waits[i].sync, waits[i].wait_value,
                              VK_SYNC_WAIT_PENDING, UINT64_MAX);
      if (result != VK_SUCCESS)
         return result;
      struct wddm_sync *sync = (struct wddm_sync *)waits[i].sync;
      mtx_lock(&sync->mutex);
      if (waits[i].sync->flags & VK_SYNC_IS_TIMELINE) {
         uint64_t current;
         result = wddm_timeline_value_locked(sync, &current);
         struct nvkmd_wddm_point *point = NULL;
         if (result == VK_SUCCESS &&
             (sync->fence || current < waits[i].wait_value)) {
            result = wddm_timeline_native_locked(sync,
               ((struct wddm_dev *)base->dev)->native);
            if (result == VK_SUCCESS)
               result = nvkmd_wddm_fence_point(sync->fence,
                  waits[i].wait_value, &point);
         }
         mtx_unlock(&sync->mutex);
         if (result != VK_SUCCESS)
            return result;
         if (point) {
            result = nvkmd_wddm_context_wait(ctx->native, point);
            nvkmd_wddm_point_destroy(point);
            if (result != VK_SUCCESS)
               return result;
            ctx->pending = true;
         }
         continue;
      }
      struct wddm_completion *completion = completion_ref(sync->completion);
      mtx_unlock(&sync->mutex);
      if (completion) {
         result = nvkmd_wddm_context_wait(ctx->native, completion->native);
         completion_unref(completion);
         if (result != VK_SUCCESS)
            return result;
         ctx->pending = true;
      }
   }
   return VK_SUCCESS;
}

static VkResult
wddm_ctx_exec(struct nvkmd_ctx *base, struct vk_object_base *log_obj,
              uint32_t count, const struct nvkmd_ctx_exec *execs)
{
   struct wddm_ctx *ctx = (struct wddm_ctx *)base;
   for (uint32_t i = 0; i < count; i++) {
      struct nvkmd_wddm_push push = {
         .address = execs[i].addr, .size = execs[i].size_B,
         .incomplete = execs[i].incomplete, .no_prefetch = execs[i].no_prefetch,
      };
      VkResult result = nvkmd_wddm_context_push(ctx->native, push);
      if (result != VK_SUCCESS)
         return result;
      ctx->pending = true;
   }
   return VK_SUCCESS;
}

static VkResult
wddm_ctx_bind(struct nvkmd_ctx *base, struct vk_object_base *log_obj,
              uint32_t count, const struct nvkmd_ctx_bind *binds)
{
   for (uint32_t i = 0; i < count; i++) {
      const struct nvkmd_ctx_bind *b = &binds[i];
      VkResult result = b->op == NVKMD_BIND_OP_BIND ?
         wddm_va_bind(b->va, log_obj, b->va_offset_B, b->mem,
                      b->mem_offset_B, b->range_B) :
         wddm_va_unbind(b->va, log_obj, b->va_offset_B, b->range_B);
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

static VkResult
wddm_ctx_signal(struct nvkmd_ctx *base, struct vk_object_base *log_obj,
                uint32_t count, const struct vk_sync_signal *signals)
{
   struct wddm_ctx *ctx = (struct wddm_ctx *)base;
   for (uint32_t i = 0; i < count; i++) {
      if (signals[i].sync->type != &wddm_sync_type)
         return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   for (uint32_t i = 0; i < count; i++) {
      if (!(signals[i].sync->flags & VK_SYNC_IS_TIMELINE))
         continue;
      struct wddm_sync *sync = (struct wddm_sync *)signals[i].sync;
      mtx_lock(&sync->mutex);
      VkResult result = wddm_timeline_native_locked(sync,
         ((struct wddm_dev *)base->dev)->native);
      struct nvkmd_wddm_point *point = NULL;
      if (result == VK_SUCCESS)
         result = nvkmd_wddm_fence_point(sync->fence,
            signals[i].signal_value, &point);
      mtx_unlock(&sync->mutex);
      if (result != VK_SUCCESS)
         return result;
      result = nvkmd_wddm_context_signal(ctx->native, point);
      nvkmd_wddm_point_destroy(point);
      if (result != VK_SUCCESS)
         return result;
      ctx->pending = true;
   }
   VkResult result = wddm_ctx_flush(base, log_obj);
   if (result != VK_SUCCESS)
      return result;
   for (uint32_t i = 0; i < count; i++) {
      struct wddm_sync *sync = (struct wddm_sync *)signals[i].sync;
      mtx_lock(&sync->mutex);
      if (signals[i].sync->flags & VK_SYNC_IS_TIMELINE) {
         sync->pending_value = MAX2(sync->pending_value, signals[i].signal_value);
         u_cnd_monotonic_broadcast(&sync->cond);
         mtx_unlock(&sync->mutex);
         continue;
      }
      completion_unref(sync->completion);
      sync->completion = completion_ref(ctx->last);
      sync->signaled = ctx->last == NULL;
      u_cnd_monotonic_broadcast(&sync->cond);
      mtx_unlock(&sync->mutex);
   }
   return VK_SUCCESS;
}

static VkResult
wddm_ctx_sync(struct nvkmd_ctx *base, struct vk_object_base *log_obj)
{
   struct wddm_ctx *ctx = (struct wddm_ctx *)base;
   VkResult result = wddm_ctx_flush(base, log_obj);
   if (result == VK_SUCCESS && ctx->last)
      result = nvkmd_wddm_point_wait(ctx->last->native, UINT64_MAX);
   if (result == VK_SUCCESS)
      result = nvkmd_wddm_context_collect(ctx->native);
   return result;
}

static void
wddm_ctx_destroy(struct nvkmd_ctx *base)
{
   struct wddm_ctx *ctx = (struct wddm_ctx *)base;
   wddm_ctx_sync(base, NULL);
   completion_unref(ctx->last);
   nvkmd_wddm_context_destroy(ctx->native);
   free(ctx);
}

static const struct nvkmd_ctx_ops wddm_ctx_ops = {
   .destroy = wddm_ctx_destroy, .wait = wddm_ctx_wait, .exec = wddm_ctx_exec,
   .bind = wddm_ctx_bind, .signal = wddm_ctx_signal,
   .flush = wddm_ctx_flush, .sync = wddm_ctx_sync,
};

static VkResult
wddm_create_ctx(struct nvkmd_dev *base, struct vk_object_base *log_obj,
                enum nvkmd_engines engines, struct nvkmd_ctx **out)
{
   if (engines & (NVKMD_ENGINE_2D | NVKMD_ENGINE_M2MF | NVKMD_ENGINE_VDEC))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct wddm_ctx *ctx = calloc(1, sizeof(*ctx));
   if (!ctx)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   ctx->base.ops = &wddm_ctx_ops;
   ctx->base.dev = base;
   VkResult result = nvkmd_wddm_context_create(((struct wddm_dev *)base)->native,
      engines == NVKMD_ENGINE_BIND ? NVKMD_ENGINE_3D | NVKMD_ENGINE_COMPUTE : engines,
      1, &ctx->native);
   if (result != VK_SUCCESS) {
      free(ctx);
      return result;
   }
   *out = &ctx->base;
   return VK_SUCCESS;
}

static void
wddm_dev_destroy(struct nvkmd_dev *base)
{
   struct wddm_dev *dev = (struct wddm_dev *)base;
   assert(list_is_empty(&base->mems));
   assert(list_is_empty(&dev->vas));
   nvkmd_wddm_device_destroy(dev->native);
   simple_mtx_destroy(&base->mems_mutex);
   free(dev);
}

static uint64_t
wddm_dev_get_gpu_timestamp(struct nvkmd_dev *base)
{
   uint64_t timestamp = 0;
   nvkmd_wddm_device_timestamp(((struct wddm_dev *)base)->native, &timestamp);
   return timestamp;
}

static const struct nvkmd_dev_ops wddm_dev_ops = {
   .destroy = wddm_dev_destroy,
   .get_gpu_timestamp = wddm_dev_get_gpu_timestamp,
   .alloc_mem = wddm_alloc_mem, .alloc_va = wddm_alloc_va,
   .create_ctx = wddm_create_ctx,
};

static VkResult
wddm_create_dev(struct nvkmd_pdev *base, struct vk_object_base *log_obj,
                struct nvkmd_dev **out)
{
   struct wddm_pdev *pdev = (struct wddm_pdev *)base;
   struct wddm_dev *dev = calloc(1, sizeof(*dev));
   if (!dev)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   dev->base = (struct nvkmd_dev) {
      .ops = &wddm_dev_ops, .pdev = base,
      .va_start = 0x10000, .va_end = UINT64_C(1) << 40,
   };
   list_inithead(&dev->base.mems);
   list_inithead(&dev->vas);
   simple_mtx_init(&dev->base.mems_mutex, mtx_plain);
   VkResult result = nvkmd_wddm_nvidia_device_create(pdev->luid, 0, 1,
                                                   &dev->native);
   if (result != VK_SUCCESS) {
      simple_mtx_destroy(&dev->base.mems_mutex);
      free(dev);
      return result;
   }
   *out = &dev->base;
   return VK_SUCCESS;
}

static void
wddm_pdev_destroy(struct nvkmd_pdev *base)
{
   free(base);
}

static const struct nvkmd_pdev_ops wddm_pdev_ops = {
   .destroy = wddm_pdev_destroy, .create_dev = wddm_create_dev,
};

VkResult
nvkmd_wddm_create_pdev(const struct nvkmd_wddm_adapter_info *adapter,
                      struct vk_object_base *log_obj, enum nvk_debug flags,
                      struct nvkmd_pdev **out)
{
   struct nvkmd_wddm_device *probe;
   VkResult result = nvkmd_wddm_nvidia_device_create(adapter->luid, 0, 1, &probe);
   if (result != VK_SUCCESS)
      return result;
   struct nvkmd_wddm_device_info info = {0};
   result = nvkmd_wddm_device_info(probe, &info);
   uint64_t timestamp;
   const bool has_gpu_timestamp =
      nvkmd_wddm_device_timestamp(probe, &timestamp) == VK_SUCCESS;
   struct nvkmd_wddm_memory *local_probe = NULL;
   uint32_t local_kind;
   const bool has_local_memory =
      nvkmd_wddm_nvidia_local_memory_create(probe, 65536, &local_probe,
                                            &local_kind) == VK_SUCCESS;
   nvkmd_wddm_memory_destroy(local_probe);
   nvkmd_wddm_device_destroy(probe);
   if (result != VK_SUCCESS)
      return result;
   const uint64_t required = NVKMD_WDDM_INFO_IDENTITY |
      NVKMD_WDDM_INFO_CLASSES | NVKMD_WDDM_INFO_TOPOLOGY;
   if ((info.valid_fields & required) != required || !info.sm ||
       !info.cls_eng3d || !info.cls_compute || !info.gpc_count ||
       !info.tpc_count || !info.mp_per_tpc || !info.max_warps_per_mp)
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   struct wddm_pdev *pdev = calloc(1, sizeof(*pdev));
   if (!pdev)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   pdev->base.ops = &wddm_pdev_ops;
   pdev->base.debug_flags = flags;
   pdev->base.bind_align_B = 4096;
   pdev->base.kmd_info.has_prebound_engines = true;
   pdev->base.kmd_info.has_gpu_timestamp = has_gpu_timestamp;
   pdev->base.kmd_info.has_compute_queue = true;
   pdev->luid = adapter->luid;
   struct nv_device_info *nv = &pdev->base.dev_info;
   nv->type = NV_DEVICE_TYPE_DIS;
   nv->vram_size_B = has_local_memory ? info.vram_size : 0;
   nv->device_id = info.device_id;
   nv->chipset = info.chipset;
   nv->sm = info.sm;
   nv->gpc_count = info.gpc_count;
   nv->tpc_count = info.tpc_count;
   nv->mp_per_tpc = info.mp_per_tpc;
   nv->max_warps_per_mp = info.max_warps_per_mp;
   nv->max_blocks_per_mp = info.max_blocks_per_mp;
   nv->cls_copy = info.cls_copy;
   nv->cls_eng3d = info.cls_eng3d;
   nv->cls_compute = info.cls_compute;
   nv->cls_gpfifo = info.cls_gpfifo;
   nv->nc_atom_size_B = 64;
   memcpy(nv->device_name, info.device_name, sizeof(nv->device_name));
   memcpy(nv->chipset_name, info.chipset_name, sizeof(nv->chipset_name));
   nv->device_name[sizeof(nv->device_name) - 1] = 0;
   nv->chipset_name[sizeof(nv->chipset_name) - 1] = 0;
   if (nv->sm == 120) {
      const uint16_t splits[] = {0, 8, 16, 32, 64, 100};
      nv->max_smem_per_wg_kB = 99;
      nv->sm_smem_size_count = ARRAY_SIZE(splits);
      memcpy(nv->sm_smem_sizes_kB, splits, sizeof(splits));
   } else {
      free(pdev);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }
   pdev->sync_types[0] = &wddm_sync_type;
   pdev->base.sync_types = pdev->sync_types;
   *out = &pdev->base;
   return VK_SUCCESS;
}

#ifdef NVKMD_WDDM_SYNC_PROBE
VkResult
nvkmd_wddm_probe_wait(struct nvkmd_ctx *base,
                       const struct nvkmd_wddm_point *point)
{
   struct wddm_ctx *ctx = (struct wddm_ctx *)base;
   VkResult result = nvkmd_wddm_context_wait(ctx->native, point);
   if (result == VK_SUCCESS)
      ctx->pending = true;
   return result;
}
#endif
