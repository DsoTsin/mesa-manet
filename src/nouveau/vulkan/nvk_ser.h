#ifndef NVK_SER_H
#define NVK_SER_H

#include <stdint.h>
#include <stddef.h>

#define NVK_SER_INVALID_OWNER UINT32_MAX
#define NVK_SER_SHADE_THREADS 32u
#define NVK_SER_SORT_THREADS 256u
#define NVK_SER_SORT_ITEMS 46u
#define NVK_SER_SORT_CAPACITY (NVK_SER_SORT_THREADS * NVK_SER_SORT_ITEMS)
#define NVK_SER_DEBUG_STOP_BEFORE_PCAS 1u
#define NVK_SER_DEBUG_STOP_BEFORE_SHADE 2u
#define NVK_SER_DEBUG_PCAS_TAIL 4u

struct nvk_ser_native_queue {
   uint16_t flags;
   uint16_t config;
   uint32_t defer_threshold;
   uint32_t thread_count;
   uint16_t shading_warps;
   uint16_t batch_width;
   uint64_t backing;
   uint64_t entry;
   uint32_t fill_warps;
   uint32_t sort_delay;
   uint16_t affinity_min;
   uint16_t affinity_max;
   uint32_t shared_sync_offset;
   uint64_t reflected_sked;
   uint32_t sort_qmd;
   uint32_t tail_qmd;
   uint32_t shade_qmd;
   uint32_t affinity_stride;
   uint32_t dispatch_config;
   uint32_t deferred_mask;
   uint64_t sorter;
};

struct nvk_ser_packet {
   uint32_t owner;
   uint32_t frame;
   uint32_t pc;
   uint32_t hit_object_key;
   uint64_t ray_index;
   uint32_t hint;
   uint32_t hint_bits;
};

struct nvk_ser_queue {
   struct nvk_ser_native_queue native;
   uint64_t packets;
   uint64_t resume_owner_ids;
   uint64_t counters;
   uint32_t owner_capacity;
   uint32_t resume_record_mask;
   uint32_t flags;
   uint32_t reserved0;
   uint64_t reserved1;
   uint64_t reserved2;
};

struct nvk_ser_counters {
   uint32_t alive;
   uint32_t arrived;
   uint32_t scheduled;
   uint32_t sort_completed;
   uint32_t epoch;
   uint32_t completed;
   uint32_t error;
   uint32_t remaining;
};

struct nvk_ser_kernel_args {
   uint64_t queue;
};

#ifdef __cplusplus
static_assert(sizeof(nvk_ser_native_queue) == 88, "native queue size");
static_assert(sizeof(nvk_ser_packet) == 32, "packet size");
static_assert(sizeof(nvk_ser_queue) == 144, "queue size");
static_assert(sizeof(nvk_ser_counters) == 32, "counter size");
static_assert(offsetof(nvk_ser_queue, packets) == 88, "packet address");
static_assert(offsetof(nvk_ser_queue, resume_owner_ids) == 96, "resume address");
static_assert(offsetof(nvk_ser_queue, counters) == 104, "counter address");
static_assert(offsetof(nvk_ser_counters, completed) == 20, "completion address");
#endif

#endif
