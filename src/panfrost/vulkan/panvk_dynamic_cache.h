#ifndef PANVK_DYNAMIC_CACHE_H
#define PANVK_DYNAMIC_CACHE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "vulkan/vulkan_mtk.h"
#include "util/simple_mtx.h"

#define PANVK_DYNAMIC_CACHE_GROUPS 4
#define PANVK_PDMA_SLOTS 256
#define PANVK_PDMA_CAPACITY 192
#define PANVK_PDMA_SLOT_SIZE 64

struct panvk_cache_group {
   uint32_t mask;
   uint8_t kind;
   uint8_t policy_index;
};

struct panvk_cache_policy {
   uint32_t count;
   struct panvk_cache_group groups[PANVK_DYNAMIC_CACHE_GROUPS];
};

struct panvk_pdma_entry {
   uint32_t words[16];
};

struct panvk_pdma_resources {
   uint32_t context_id, mode, fixed_policy, reserved0;
   uint32_t status, reserved1;
   uint64_t reg_offset;
   uint32_t reg_size, reserved2;
   uint64_t ring_offset;
   uint32_t ring_size, reserved3;
   uint32_t reserved4[3], version, cid;
   uint8_t hw_mode, reserved5[3];
   uint64_t shared_offset, read_offset, write_offset;
};

struct panvk_pdma_ops {
   void *data;
   int (*open)(void *data, const char *path, int flags);
   int (*close)(void *data, int fd);
   int (*ioctl)(void *data, int fd, unsigned long request, void *arg);
   void *(*map)(void *data, int fd, size_t size, uint64_t offset);
   int (*unmap)(void *data, void *addr, size_t size);
};

struct panvk_dynamic_cache {
   struct panvk_pdma_ops ops;
   struct panvk_pdma_resources resources;
   int fd;
   bool locked, preempted;
   void *regs, *ring, *shared;
   size_t shared_size;
   volatile uint32_t *read, *hw_read, *hw_write, *shared_write, *reset;
   uint32_t write;
   simple_mtx_t lock;
};

void panvk_cache_policy_add(struct panvk_cache_policy *policy,
                           const VkDynamicCacheMemoryInfoMTK *info);
void panvk_cache_policy_encode(const struct panvk_cache_policy *policy,
                              uint32_t group_uid, uint8_t frame_id,
                              uint16_t pass_id, bool compute,
                              struct panvk_pdma_entry *entry);
uint64_t panvk_cache_control_word(bool compute, bool enabled);
uint64_t panvk_cache_pass_word(bool compute, uint16_t pass_id);
uint64_t panvk_cache_frame_word(uint8_t frame_id);
int panvk_pdma_available(uint32_t read, uint32_t write);
int panvk_dynamic_cache_init(struct panvk_dynamic_cache *cache, int kbase_fd,
                            const struct panvk_pdma_ops *ops);
void panvk_dynamic_cache_finish(struct panvk_dynamic_cache *cache);
int panvk_dynamic_cache_write(struct panvk_dynamic_cache *cache,
                             const struct panvk_pdma_entry *entry);
void panvk_dynamic_cache_publish(struct panvk_dynamic_cache *cache);

#endif
