#include "panvk_dynamic_cache.h"

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#if UINTPTR_MAX == UINT64_MAX
_Static_assert(sizeof(VkDynamicCacheMemoryInfoMTK) == 32, "MTK info ABI");
_Static_assert(offsetof(VkDynamicCacheMemoryInfoMTK, pCaches) == 24, "MTK array ABI");
_Static_assert(sizeof(VkDynamicCacheMemoryRegionMTK) == 48, "MTK record ABI");
_Static_assert(offsetof(VkDynamicCacheMemoryRegionMTK, groupMask) == 40, "MTK mask ABI");
#endif
_Static_assert(sizeof(struct panvk_pdma_resources) == 104, "PDMA ioctl ABI");
_Static_assert(offsetof(struct panvk_pdma_resources, version) == 68, "PDMA version ABI");
_Static_assert(offsetof(struct panvk_pdma_resources, hw_mode) == 76, "PDMA mode ABI");
_Static_assert(offsetof(struct panvk_pdma_resources, shared_offset) == 80,
               "PDMA shared ABI");

void
panvk_cache_policy_add(struct panvk_cache_policy *policy,
                       const VkDynamicCacheMemoryInfoMTK *info)
{
   if (!info || !info->pCaches)
      return;

   for (uint32_t i = 0; i < info->cacheCount && policy->count < 4; i++) {
      const VkDynamicCacheMemoryRegionMTK *src = &info->pCaches[i];
      const uint32_t masks[2][3] = {
         {0x10003000, 0x00070000, 0x00700000},
         {0x10000000, 0x01000000, 0x00800000},
      };
      if (src->cacheKind > 1)
         continue;

      uint32_t mask = 0;
      for (unsigned bit = 0; bit < 3; bit++)
         if (src->groupMask & (1u << bit))
            mask |= masks[src->cacheKind][bit];

      for (unsigned j = 0; j < policy->count; j++)
         if (policy->groups[j].kind == src->cacheKind &&
             (policy->groups[j].mask & mask))
            mask = 0;
      if (!mask)
         continue;

      static const uint8_t indices[2][3] = {{6, 5, 4}, {2, 1, 3}};
      const unsigned p = src->cacheKind == 0 ? (src->policy < 2 ? src->policy : 2)
                                             : src->policy;
      policy->groups[policy->count++] = (struct panvk_cache_group){
         .mask = mask,
         .kind = src->cacheKind,
         .policy_index = p < 3 ? indices[src->cacheKind][p] : 0,
      };
   }
}

void
panvk_cache_policy_encode(const struct panvk_cache_policy *policy,
                          uint32_t group_uid, uint8_t frame_id,
                          uint16_t pass_id, bool compute,
                          struct panvk_pdma_entry *entry)
{
   memset(entry, 0, sizeof(*entry));
   entry->words[0] = 1 | (compute ? 8 : 0);
   entry->words[1] = group_uid;
   entry->words[2] = pass_id | 0x100000 | ((uint32_t)frame_id << 24);
   for (unsigned i = 0; i < policy->count; i++) {
      entry->words[3 + 2 * i] = policy->groups[i].mask & 0xffffff00;
      entry->words[4 + 2 * i] = 0x70 | policy->groups[i].policy_index;
   }
}

uint64_t
panvk_cache_control_word(bool compute, bool enabled)
{
   return (compute ? UINT64_C(0x103e3e0600020214)
                   : UINT64_C(0x1000000600000214)) |
          ((uint64_t)enabled << 16);
}

uint64_t
panvk_cache_pass_word(bool compute, uint16_t pass_id)
{
   return (compute ? UINT64_C(0x103e3e0600001000)
                   : UINT64_C(0x1000000600001000)) |
          ((uint64_t)pass_id << 16);
}

uint64_t
panvk_cache_frame_word(uint8_t frame_id)
{
   return UINT64_C(0x103f3f0600000800) | ((uint64_t)frame_id << 16);
}

int
panvk_pdma_available(uint32_t read, uint32_t write)
{
   if (read == 0xdeadbeef)
      return -EOWNERDEAD;
   if (read > 511 || write > 511)
      return -EINVAL;
   unsigned used = (write - read) & 511;
   return used > PANVK_PDMA_CAPACITY ? -EINVAL : PANVK_PDMA_CAPACITY - used;
}

static int
pdma_open(void *data, const char *path, int flags)
{
   return open(path, flags);
}

static int
pdma_close(void *data, int fd)
{
   return close(fd);
}

static int
pdma_ioctl(void *data, int fd, unsigned long request, void *arg)
{
   return ioctl(fd, request, arg);
}

static void *
pdma_map(void *data, int fd, size_t size, uint64_t offset)
{
   return mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
}

static int
pdma_unmap(void *data, void *addr, size_t size)
{
   return munmap(addr, size);
}

void
panvk_dynamic_cache_finish(struct panvk_dynamic_cache *cache)
{
   if (cache->shared && cache->shared != cache->ring)
      cache->ops.unmap(cache->ops.data, cache->shared, cache->shared_size);
   if (cache->ring)
      cache->ops.unmap(cache->ops.data, cache->ring, cache->resources.ring_size);
   if (cache->regs)
      cache->ops.unmap(cache->ops.data, cache->regs, cache->resources.reg_size);
   if (cache->locked)
      cache->ops.ioctl(cache->ops.data, cache->fd, 0xc0688002,
                       &cache->resources);
   if (cache->fd >= 0)
      cache->ops.close(cache->ops.data, cache->fd);
   cache->shared = cache->ring = cache->regs = NULL;
   cache->locked = false;
   cache->fd = -1;
}

int
panvk_dynamic_cache_init(struct panvk_dynamic_cache *cache, int kbase_fd,
                         const struct panvk_pdma_ops *ops)
{
   memset(cache, 0, sizeof(*cache));
   cache->fd = -1;
   cache->ops = ops ? *ops : (struct panvk_pdma_ops){
      .open = pdma_open, .close = pdma_close, .ioctl = pdma_ioctl,
      .map = pdma_map, .unmap = pdma_unmap,
   };
   if (cache->ops.ioctl(cache->ops.data, kbase_fd, 0x80048011,
                        &cache->resources.context_id))
      return -EIO;
   cache->resources.fixed_policy = 15;
   cache->fd = cache->ops.open(cache->ops.data, "/dev/gpu_pdma",
                                O_RDWR | O_NONBLOCK | O_CLOEXEC);
   if (cache->fd < 0)
      return -EIO;
   int ret = -EIO;
   cache->locked = true;
   if (cache->ops.ioctl(cache->ops.data, cache->fd, 0xc0688001,
                        &cache->resources) || cache->resources.status != 1)
      goto fail;
   struct panvk_pdma_resources *r = &cache->resources;
   ret = -EINVAL;
   if ((r->ring_size >> 6) != PANVK_PDMA_SLOTS ||
       r->reg_size < 8 || (r->hw_mode == 1 &&
                          (r->cid >= 4 || r->reg_size < 200 + 8 * r->cid)))
      goto fail;

   cache->shared_size = r->shared_offset == r->ring_offset ? r->ring_size : 4096;
   uint64_t ro = r->read_offset & UINT64_C(0x3fffffffc);
   uint64_t wo = r->write_offset & UINT64_C(0x3fffffffc);
   if (wo + 4 > cache->shared_size ||
       (r->hw_mode != 1 && ro + 4 > cache->shared_size))
      goto fail;

   void **maps[] = {&cache->regs, &cache->ring, &cache->shared};
   const size_t sizes[] = {r->reg_size, r->ring_size, cache->shared_size};
   const uint64_t offsets[] = {r->reg_offset, r->ring_offset, r->shared_offset};
   for (unsigned i = 0; i < 3; i++) {
      if (i == 2 && r->shared_offset == r->ring_offset) {
         cache->shared = cache->ring;
         break;
      }
      *maps[i] = cache->ops.map(cache->ops.data, cache->fd, sizes[i], offsets[i]);
      if (*maps[i] == MAP_FAILED || !*maps[i]) {
         *maps[i] = NULL;
         ret = -ENOMEM;
         goto fail;
      }
   }

   size_t reg = r->hw_mode == 1 ? 192 + 8 * r->cid : 0;
   cache->hw_read = (uint32_t *)((char *)cache->regs + reg);
   cache->hw_write = (uint32_t *)((char *)cache->regs + reg + 4);
   cache->read = r->hw_mode == 1 ? cache->hw_read
                                : (uint32_t *)((char *)cache->shared + ro);
   cache->shared_write = (uint32_t *)((char *)cache->shared + wo);
   cache->reset = (uint32_t *)((char *)cache->shared + 120);
   if (*cache->read || *cache->hw_read || *cache->hw_write || *cache->shared_write) {
      ret = -EINVAL;
      goto fail;
   }
   return 0;
fail:
   panvk_dynamic_cache_finish(cache);
   return ret;
}

int
panvk_dynamic_cache_write(struct panvk_dynamic_cache *cache,
                          const struct panvk_pdma_entry *entry)
{
   if (!cache->locked)
      return -ENODEV;
   if (cache->preempted || *cache->read == 0xdeadbeef) {
      cache->preempted = true;
      return -EOWNERDEAD;
   }
   if (*cache->reset)
      return -EBUSY;
   int available = panvk_pdma_available(*cache->read, cache->write);
   if (available <= 0)
      return available ? available : -ENOSPC;
   memcpy((char *)cache->ring + (cache->write & 255) * 64, entry, 56);
   cache->write = (cache->write + 1) & 511;
   return 0;
}

void
panvk_dynamic_cache_publish(struct panvk_dynamic_cache *cache)
{
   if (!cache->locked)
      return;
   if (cache->preempted || *cache->read == 0xdeadbeef) {
      cache->preempted = true;
      return;
   }
   if (*cache->reset)
      return;
#if defined(__aarch64__)
   __asm__ volatile("dsb sy" ::: "memory");
#else
   __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
   *cache->hw_write = cache->write;
   *cache->shared_write = cache->write;
}
