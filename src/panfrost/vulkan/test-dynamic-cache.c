#include "panvk_dynamic_cache.h"
#include "vk_command_buffer.h"
#include "vk_dispatch_table.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #x); abort(); } } while (0)

struct mock {
   uint32_t regs[1024], ring[4096], shared[1024];
   unsigned maps, unmaps, closes, unlocks;
   int fail_map;
   unsigned mode, cid;
   bool alias, invalid_size, invalid_status;
};

static int
mock_open(void *data, const char *path, int flags)
{
   CHECK(strcmp(path, "/dev/gpu_pdma") == 0);
   return 99;
}

static int
mock_close(void *data, int fd)
{
   struct mock *m = data;
   CHECK(fd == 99);
   m->closes++;
   return 0;
}

static int
mock_ioctl(void *data, int fd, unsigned long request, void *arg)
{
   struct mock *m = data;
   if (request == 0x80048011) {
      CHECK(fd == 11);
      *(uint32_t *)arg = 42;
   } else if (request == 0xc0688001) {
      struct panvk_pdma_resources *r = arg;
      CHECK(fd == 99 && r->context_id == 42 && r->fixed_policy == 15 && !r->mode);
      r->status = !m->invalid_status;
      r->reg_offset = 0x1000;
      r->reg_size = 4096;
      r->ring_offset = 0x2000;
      r->ring_size = m->invalid_size ? 8192 : 16384;
      r->shared_offset = m->alias ? r->ring_offset : 0x3000;
      r->read_offset = 128;
      r->write_offset = 132;
      r->hw_mode = m->mode;
      r->cid = m->cid;
   } else {
      CHECK(request == 0xc0688002 && fd == 99);
      m->unlocks++;
   }
   return 0;
}

static void *
mock_map(void *data, int fd, size_t size, uint64_t offset)
{
   struct mock *m = data;
   CHECK(fd == 99);
   if ((int)m->maps++ == m->fail_map)
      return MAP_FAILED;
   if (offset == 0x1000) return m->regs;
   if (offset == 0x2000) return m->ring;
   CHECK(offset == 0x3000);
   return m->shared;
}

static int
mock_unmap(void *data, void *addr, size_t size)
{
   struct mock *m = data;
   CHECK(addr == m->regs || addr == m->ring || addr == m->shared);
   m->unmaps++;
   return 0;
}

static struct panvk_pdma_ops
mock_ops(struct mock *m)
{
   return (struct panvk_pdma_ops){
      .data = m, .open = mock_open, .close = mock_close,
      .ioctl = mock_ioctl, .map = mock_map, .unmap = mock_unmap,
   };
}

static void
test_policy(void)
{
   VkDynamicCacheMemoryRegionMTK regions[] = {
      {.cacheKind = 9, .groupMask = 7},
      {.cacheKind = 0, .groupMask = 0},
      {.cacheKind = 0, .groupMask = 1, .policy = 0},
      {.cacheKind = 0, .groupMask = 3, .policy = 1},
      {.cacheKind = 1, .groupMask = 1, .policy = 2},
      {.cacheKind = 0, .groupMask = 2, .policy = 1},
      {.cacheKind = 0, .groupMask = 4, .policy = 99},
      {.cacheKind = 1, .groupMask = 2, .policy = 0},
   };
   VkDynamicCacheMemoryInfoMTK info = {.cacheCount = 8, .pCaches = regions};
   struct panvk_cache_policy p = {0};
   panvk_cache_policy_add(&p, &info);
   CHECK(p.count == 4);
   const struct panvk_cache_group expected[] = {
      {0x10003000, 0, 6}, {0x10000000, 1, 3},
      {0x00070000, 0, 5}, {0x00700000, 0, 4},
   };
   for (unsigned i = 0; i < 4; i++) {
      CHECK(p.groups[i].mask == expected[i].mask);
      CHECK(p.groups[i].kind == expected[i].kind);
      CHECK(p.groups[i].policy_index == expected[i].policy_index);
   }
   memset(regions, 0xff, sizeof(regions));
   CHECK(p.groups[0].mask == 0x10003000);
   panvk_cache_policy_add(&p, NULL);
   CHECK(p.count == 4);
}

static void
test_encoding(void)
{
   const uint8_t expected[2][4] = {{6, 5, 4, 4}, {2, 1, 3, 0}};
   for (unsigned kind = 0; kind < 2; kind++) {
      for (unsigned policy = 0; policy < 4; policy++) {
         VkDynamicCacheMemoryRegionMTK region = {
            .cacheKind = kind, .groupMask = 7, .policy = policy,
         };
         VkDynamicCacheMemoryInfoMTK info = {.cacheCount = 1, .pCaches = &region};
         struct panvk_cache_policy p = {0};
         panvk_cache_policy_add(&p, &info);
         struct panvk_pdma_entry entry;
         panvk_cache_policy_encode(&p, 0x12345678, 0xab, 0x3456, false, &entry);
         uint32_t golden[16] = {
            1, 0x12345678, 0xab103456,
            kind ? 0x11800000 : 0x10773000, 0x70 | expected[kind][policy],
         };
         CHECK(memcmp(entry.words, golden, sizeof(golden)) == 0);
         panvk_cache_policy_encode(&p, 0x12345678, 0xab, 0x3456, true, &entry);
         golden[0] = 9;
         CHECK(memcmp(entry.words, golden, sizeof(golden)) == 0);
      }
   }
   CHECK(panvk_cache_control_word(false, true) == UINT64_C(0x1000000600010214));
   CHECK(panvk_cache_control_word(false, false) == UINT64_C(0x1000000600000214));
   CHECK(panvk_cache_control_word(true, true) == UINT64_C(0x103e3e0600030214));
   CHECK(panvk_cache_control_word(true, false) == UINT64_C(0x103e3e0600020214));
   CHECK(panvk_cache_pass_word(true, 0x3456) == UINT64_C(0x103e3e0634561000));
   CHECK(panvk_cache_pass_word(false, 0x3456) == UINT64_C(0x1000000634561000));
   CHECK(panvk_cache_frame_word(0xab) == UINT64_C(0x103f3f0600ab0800));
}

static void
test_ring(void)
{
   CHECK(panvk_pdma_available(0, 192) == 0);
   CHECK(panvk_pdma_available(0, 193) == -EINVAL);
   CHECK(panvk_pdma_available(511, 0) == 191);
   CHECK(panvk_pdma_available(0xdeadbeef, 0) == -EOWNERDEAD);
   CHECK(panvk_pdma_available(512, 0) == -EINVAL);
   for (unsigned r = 0; r < 512; r++)
      for (unsigned used = 0; used < 512; used++)
         CHECK(panvk_pdma_available(r, (r + used) & 511) ==
               (used > 192 ? -EINVAL : 192 - (int)used));

   struct mock m = {.fail_map = -1};
   struct panvk_pdma_ops ops = mock_ops(&m);
   struct panvk_dynamic_cache cache;
   CHECK(panvk_dynamic_cache_init(&cache, 11, &ops) == 0);
   struct panvk_pdma_entry entry = {.words = {0x1234}};
   for (unsigned i = 0; i < 192; i++)
      CHECK(panvk_dynamic_cache_write(&cache, &entry) == 0);
   CHECK(panvk_dynamic_cache_write(&cache, &entry) == -ENOSPC);
   panvk_dynamic_cache_publish(&cache);
   CHECK(m.regs[1] == 192 && m.shared[33] == 192);
   *cache.read = 192;
   for (unsigned i = 192; i < 600; i++) {
      entry.words[0] = i;
      CHECK(panvk_dynamic_cache_write(&cache, &entry) == 0);
      CHECK(m.ring[(i & 255) * 16] == i);
      *cache.read = cache.write;
   }
   CHECK(cache.write == 88);
   *cache.reset = 1;
   CHECK(panvk_dynamic_cache_write(&cache, &entry) == -EBUSY);
   panvk_dynamic_cache_publish(&cache);
   CHECK(m.regs[1] == 192);
   *cache.reset = 0;
   *cache.read = 0xdeadbeef;
   CHECK(panvk_dynamic_cache_write(&cache, &entry) == -EOWNERDEAD);
   panvk_dynamic_cache_publish(&cache);
   CHECK(m.regs[1] == 192);
   panvk_dynamic_cache_finish(&cache);
   CHECK(m.closes == 1 && m.unlocks == 1 && m.unmaps == 3);
   panvk_dynamic_cache_finish(&cache);
   CHECK(m.closes == 1 && m.unlocks == 1 && m.unmaps == 3);
}

static void
test_lifetime(void)
{
   for (int fail = 0; fail < 6; fail++) {
      struct mock m = {.fail_map = fail < 3 ? fail : -1,
                       .invalid_size = fail == 3,
                       .invalid_status = fail == 4,
                       .mode = fail == 5, .cid = 4};
      struct panvk_pdma_ops ops = mock_ops(&m);
      struct panvk_dynamic_cache cache;
      CHECK(panvk_dynamic_cache_init(&cache, 11, &ops) < 0);
      CHECK(m.closes == 1 && m.unlocks == 1);
      CHECK(m.unmaps == (fail < 3 ? (unsigned)fail : 0));
   }
   for (unsigned mode = 0; mode < 2; mode++) {
      struct mock m = {.fail_map = -1, .mode = mode, .cid = 3, .alias = mode == 0};
      struct panvk_pdma_ops ops = mock_ops(&m);
      struct panvk_dynamic_cache cache;
      CHECK(panvk_dynamic_cache_init(&cache, 11, &ops) == 0);
      CHECK(cache.read == (mode ? &m.regs[54] : &m.ring[32]));
      CHECK(cache.hw_write == (mode ? &m.regs[55] : &m.regs[1]));
      panvk_dynamic_cache_finish(&cache);
      CHECK(m.unmaps == (mode ? 3 : 2) && m.closes == 1 && m.unlocks == 1);
   }
}

static unsigned replayed;

static void VKAPI_CALL
replay_memory(VkCommandBuffer commandBuffer, const VkDynamicCacheMemoryInfoMTK *info)
{
   CHECK(info->cacheCount == 1 && info->pCaches->cacheKind == 1);
   CHECK(info->pCaches->groupMask == 7 && info->pCaches->policy == 2);
   CHECK(!info->pNext && !info->pCaches->pNext);
   replayed++;
}

static void VKAPI_CALL
replay_access(VkCommandBuffer commandBuffer)
{
   replayed++;
}

static void
test_dispatch_and_copy(void)
{
   struct vk_device_dispatch_table dispatch = {
      .CmdSetDynamicCacheMemoryMTK = replay_memory,
      .CmdSetDynamicCacheAccessMTK = replay_access,
   };
   struct vk_instance_extension_table iext = {0};
   struct vk_device_extension_table dext = {0};
   const char *names[] = {"vkCmdSetDynamicCacheMemoryMTK", "vkCmdSetDynamicCacheAccessMTK"};
   for (unsigned i = 0; i < 2; i++) {
      CHECK(!vk_device_dispatch_table_get_if_supported(
         &dispatch, names[i], VK_API_VERSION_1_4, &iext, &dext));
      dext.MTK_dynamic_cache_memory = true;
      CHECK(vk_device_dispatch_table_get_if_supported(
         &dispatch, names[i], VK_API_VERSION_1_4, &iext, &dext));
      dext.MTK_dynamic_cache_memory = false;
   }

   struct vk_command_buffer buffer = {0};
   vk_cmd_queue_init(&buffer.cmd_queue);
   VkDynamicCacheMemoryRegionMTK region = {.cacheKind = 1, .groupMask = 7, .policy = 2};
   VkDynamicCacheMemoryInfoMTK info = {.cacheCount = 1, .pCaches = &region};
   struct vk_cmd_queue_entry *cmd = vk_enqueue_cmd_set_dynamic_cache_memory_mtk(
      &buffer.cmd_queue, &info);
   CHECK(cmd);
   list_addtail(&cmd->cmd_link, &buffer.cmd_queue.cmds);
   cmd = vk_enqueue_cmd_set_dynamic_cache_access_mtk(&buffer.cmd_queue);
   CHECK(cmd);
   list_addtail(&cmd->cmd_link, &buffer.cmd_queue.cmds);
   memset(&info, 0xff, sizeof(info));
   memset(&region, 0xff, sizeof(region));
   vk_cmd_queue_execute(&buffer.cmd_queue, VK_NULL_HANDLE, &dispatch);
   vk_cmd_queue_execute(&buffer.cmd_queue, VK_NULL_HANDLE, &dispatch);
   CHECK(replayed == 4);
   vk_cmd_queue_finish(&buffer.cmd_queue);
}

int
main(void)
{
   test_policy();
   test_encoding();
   test_ring();
   test_lifetime();
   test_dispatch_and_copy();
   puts("dynamic cache: 5 groups passed (262144 cursor vectors, ABI, encoding, lifetime, dispatch, secondary copy)");
   return 0;
}
