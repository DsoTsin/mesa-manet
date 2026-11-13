#ifndef AGC_DRIVER_OPS_H
#define AGC_DRIVER_OPS_H

#include "agc_types.h"

struct radeon_info;
struct agc_backend;
struct agc_bo;
struct agc_hwq;

enum agc_queue_kind {
   AGC_QUEUE_GFX,
   AGC_QUEUE_COMPUTE,
   AGC_QUEUE_KIND_COUNT
};

enum agc_device_state {
   AGC_DEVICE_ACTIVE,
   AGC_DEVICE_HUNG,
   AGC_DEVICE_PAGE_FAULT,
   AGC_DEVICE_REMOVED,
   AGC_DEVICE_UNKNOWN
};

enum agc_va_fill {
   AGC_VA_NO_ACCESS,
   AGC_VA_ZERO
};

#define AGC_BO_GL2_BYPASS (UINT32_C(1) << 0)
#define AGC_BO_EXECUTE (UINT32_C(1) << 1)

#define AGC_IB_MAIN UINT32_C(0)
#define AGC_IB_PREAMBLE UINT32_C(0xc)

#define AGC_SUBMIT_MAX_IBS 64
#define AGC_SUBMIT_MAX_IN_FLIGHT 64
#define AGC_TIMEOUT_INFINITE INT64_C(-1)

struct agc_ib {
   uint64_t va;
   uint32_t dwords;
   uint32_t flags;
};

struct agc_driver_ops {
   void (*destroy)(struct agc_backend *be);
   bool (*query_info)(struct agc_backend *be, struct radeon_info *info);

   bool (*va_reserve)(struct agc_backend *be, uint64_t va, uint64_t size);
   bool (*va_reserve_zero)(struct agc_backend *be, uint64_t va, uint64_t size);
   bool (*va_clear)(struct agc_backend *be, uint64_t va, uint64_t size, enum agc_va_fill fill);
   bool (*va_release)(struct agc_backend *be, uint64_t va, uint64_t size);

   struct agc_bo *(*bo_create)(struct agc_backend *be, uint64_t size, uint32_t flags, uint64_t va);
   struct agc_bo *(*bo_import)(struct agc_backend *be, void *cpu, uint64_t size, uint32_t flags,
                               uint64_t va);
   bool (*bo_alias)(struct agc_backend *be, struct agc_bo *bo, uint64_t offset, uint64_t size,
                    uint64_t va);
   void *(*bo_cpu)(struct agc_bo *bo);
   uint64_t (*bo_va)(struct agc_bo *bo);
   uint64_t (*bo_size)(struct agc_bo *bo);
   void (*bo_destroy)(struct agc_backend *be, struct agc_bo *bo);
   bool (*paging_settle)(struct agc_backend *be);

   struct agc_hwq *(*queue_create)(struct agc_backend *be, enum agc_queue_kind kind);
   void (*queue_destroy)(struct agc_hwq *queue);
   uint64_t (*submit)(struct agc_hwq *queue, const struct agc_ib *ibs, uint32_t count);
   bool (*fence_wait)(struct agc_hwq *queue, uint64_t fence, int64_t timeout_ns);
   uint64_t (*fence_completed)(struct agc_hwq *queue);
   bool (*queue_wait)(struct agc_hwq *queue, struct agc_hwq *producer, uint64_t fence);

   enum agc_device_state (*device_state)(struct agc_backend *be, uint64_t *fault_va);
};

struct agc_backend {
   const struct agc_driver_ops *ops;
};

#define AGC_WDDM_NODE_NONE UINT32_MAX

enum agc_wddm_chip_source {
   AGC_WDDM_CHIP_NOT_QUERIED,
   AGC_WDDM_CHIP_PROFILE_ONLY,
   AGC_WDDM_CHIP_KMD_QUERY
};

struct agc_wddm_adapter {
   uint32_t vendor_id;
   uint32_t device_id;
   uint32_t sub_vendor_id;
   uint32_t subsystem_id;
   uint32_t revision;
   uint32_t luid_low;
   int32_t luid_high;
   uint64_t driver_version;
   const char *profile;
};

struct agc_wddm_node {
   uint32_t ordinal;
   int32_t engine_type;
   char name[100];
};

struct agc_wddm_description {
   struct agc_wddm_adapter adapter;
   uint32_t va_bits;
   uint64_t va_end;
   uint64_t driver_va_base;
   uint32_t node_count;
   uint32_t gfx_node;
   uint32_t compute_node;
   uint32_t live_allocations;
   bool lost;
   enum agc_wddm_chip_source chip_source;
   int32_t chip_basic_status;
   int32_t chip_gpu_status;
};

struct agc_wddm_queue_description {
   uint32_t node;
   uint32_t queue_id;
   uint64_t fence_gpu_va;
   uint64_t submitted;
};

struct agc_wddm_device_state {
   int32_t execution_status;
   int32_t execution;
   int32_t page_fault_status;
   uint32_t fault_flags;
   uint32_t fault_error_code;
   uint32_t fault_stage;
   uint32_t fault_bind_table_entry;
   uint64_t fault_va;
   uint64_t fault_sequence;
};

struct agc_wddm_video_memory {
   uint64_t local_budget;
   uint64_t local_usage;
   uint64_t non_local_budget;
   uint64_t non_local_usage;
};

uint32_t agc_wddm_enumerate(struct agc_wddm_adapter *adapters, uint32_t capacity);
struct agc_backend *agc_wddm_open(void);
void agc_wddm_describe(struct agc_backend *be, struct agc_wddm_description *description);
uint32_t agc_wddm_nodes(struct agc_backend *be, struct agc_wddm_node *nodes, uint32_t capacity);
void agc_wddm_queue_describe(struct agc_hwq *queue, struct agc_wddm_queue_description *description);
void agc_wddm_device_state_detail(struct agc_backend *be, struct agc_wddm_device_state *state);
size_t agc_wddm_device_state_format(const struct agc_wddm_device_state *state, char *text,
                                    size_t size);
bool agc_wddm_video_memory(struct agc_backend *be, struct agc_wddm_video_memory *memory);
int32_t agc_wddm_query_umd_private(struct agc_backend *be, void *data, uint32_t bytes);
size_t agc_wddm_abi_fixture(void *out, size_t capacity);

struct agc_wddm_fence;

struct agc_wddm_fence *agc_wddm_fence_create_shared(struct agc_backend *be, uint64_t initial_value,
                                                    void **nt_handle);
struct agc_wddm_fence *agc_wddm_fence_open_shared(struct agc_backend *be, void *nt_handle);
void agc_wddm_fence_destroy(struct agc_wddm_fence *fence);
bool agc_wddm_fence_value(struct agc_wddm_fence *fence, uint64_t *value);
bool agc_wddm_queue_wait_fence(struct agc_hwq *queue, struct agc_wddm_fence *fence, uint64_t value);
bool agc_wddm_queue_signal_fence(struct agc_hwq *queue, struct agc_wddm_fence *fence,
                                 uint64_t value);
struct agc_bo *agc_wddm_bo_open_shared(struct agc_backend *be, void *nt_handle, uint64_t size);

#endif
