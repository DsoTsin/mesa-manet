#ifndef NVKMD_WDDM_H
#define NVKMD_WDDM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nvkmd_wddm_device;
struct nvkmd_wddm_memory;
struct nvkmd_wddm_va;
struct nvkmd_wddm_binding;
struct nvkmd_wddm_fence;
struct nvkmd_wddm_point;
struct nvkmd_wddm_context;
struct nvkmd_wddm_presenter;
struct nvkmd_wddm_present_image;

struct nvkmd_wddm_luid { uint32_t low; int32_t high; };

struct nvkmd_wddm_push {
   uint64_t address;
   uint32_t size;
   uint8_t incomplete;
   uint8_t no_prefetch;
   uint16_t reserved;
};

struct nvkmd_wddm_adapter_info {
   struct nvkmd_wddm_luid luid;
   uint32_t vendor_id;
   uint32_t device_id;
   uint64_t driver_version;
   uint32_t wddm_version;
   uint32_t scheduling_caps;
};

struct nvkmd_wddm_device_info {
   uint64_t valid_fields;
   uint64_t vram_size;
   uint64_t bar_size;
   uint32_t vendor_id;
   uint32_t device_id;
   uint32_t chipset;
   uint32_t sm;
   uint32_t gpc_count;
   uint32_t tpc_count;
   uint32_t mp_per_tpc;
   uint32_t max_warps_per_mp;
   uint32_t max_blocks_per_mp;
   uint32_t cls_copy;
   uint32_t cls_eng2d;
   uint32_t cls_eng3d;
   uint32_t cls_m2mf;
   uint32_t cls_compute;
   uint32_t cls_gpfifo;
   char device_name[64];
   char chipset_name[16];
};

#define NVKMD_WDDM_INFO_IDENTITY UINT64_C(1)
#define NVKMD_WDDM_INFO_CLASSES UINT64_C(2)
#define NVKMD_WDDM_INFO_TOPOLOGY UINT64_C(4)
#define NVKMD_WDDM_INFO_MEMORY UINT64_C(8)

int32_t nvkmd_wddm_enumerate(uint32_t *count, struct nvkmd_wddm_adapter_info *adapters);
int32_t nvkmd_wddm_device_create(struct nvkmd_wddm_luid luid, struct nvkmd_wddm_device **out);
int32_t nvkmd_wddm_device_check_state(const struct nvkmd_wddm_device *device);
int32_t nvkmd_wddm_device_timestamp(const struct nvkmd_wddm_device *device, uint64_t *out);
int32_t nvkmd_wddm_presenter_create(const struct nvkmd_wddm_device *device, uintptr_t hwnd, struct nvkmd_wddm_presenter **out);
void nvkmd_wddm_presenter_destroy(struct nvkmd_wddm_presenter *presenter);
int32_t nvkmd_wddm_present_image_create(const struct nvkmd_wddm_presenter *presenter, uint32_t width, uint32_t height, uint32_t vk_format, struct nvkmd_wddm_present_image **out);
int32_t nvkmd_wddm_present_image_memory(const struct nvkmd_wddm_present_image *image, struct nvkmd_wddm_memory **memory, uint64_t *size, uint32_t *pitch);
void nvkmd_wddm_present_image_destroy(struct nvkmd_wddm_present_image *image);
int32_t nvkmd_wddm_presenter_present(struct nvkmd_wddm_presenter *presenter, const struct nvkmd_wddm_present_image *image, uint64_t timeout_ns);
int32_t nvkmd_wddm_memory_map(const struct nvkmd_wddm_memory *memory, void **out);
int32_t nvkmd_wddm_va_address(const struct nvkmd_wddm_va *va, uint64_t *out);
int32_t nvkmd_wddm_va_bind_executable(struct nvkmd_wddm_va *va, const struct nvkmd_wddm_memory *memory, uint64_t va_offset, uint64_t memory_offset, uint64_t size, struct nvkmd_wddm_binding **out);
void nvkmd_wddm_device_destroy(struct nvkmd_wddm_device *device);
int32_t nvkmd_wddm_device_info(const struct nvkmd_wddm_device *device, struct nvkmd_wddm_device_info *out);
int32_t nvkmd_wddm_host_memory_create(struct nvkmd_wddm_device *device, uint64_t size, struct nvkmd_wddm_memory **out);
int32_t nvkmd_wddm_nvidia_host_memory_create(const struct nvkmd_wddm_device *device, uint64_t size, struct nvkmd_wddm_memory **out, uint32_t *page_kind);
int32_t nvkmd_wddm_nvidia_local_memory_create(const struct nvkmd_wddm_device *device, uint64_t size, struct nvkmd_wddm_memory **out, uint32_t *page_kind);
void nvkmd_wddm_memory_destroy(struct nvkmd_wddm_memory *memory);
int32_t nvkmd_wddm_memory_make_resident(const struct nvkmd_wddm_memory *memory, struct nvkmd_wddm_point **out);
int32_t nvkmd_wddm_memory_evict(struct nvkmd_wddm_memory *memory);
int32_t nvkmd_wddm_memory_write(struct nvkmd_wddm_memory *memory, uint64_t offset, const uint8_t *bytes, size_t size);
int32_t nvkmd_wddm_memory_read(struct nvkmd_wddm_memory *memory, uint64_t offset, uint8_t *bytes, size_t size);
int32_t nvkmd_wddm_va_reserve(struct nvkmd_wddm_device *device, uint64_t size, uint64_t alignment, uint64_t fixed, struct nvkmd_wddm_va **out);
void nvkmd_wddm_va_destroy(struct nvkmd_wddm_va *va);
int32_t nvkmd_wddm_va_bind(struct nvkmd_wddm_va *va, const struct nvkmd_wddm_memory *memory, uint64_t va_offset, uint64_t memory_offset, uint64_t size, struct nvkmd_wddm_binding **out);
int32_t nvkmd_wddm_binding_address(const struct nvkmd_wddm_binding *binding, uint64_t *out);
void nvkmd_wddm_binding_destroy(struct nvkmd_wddm_binding *binding);
int32_t nvkmd_wddm_fence_create(const struct nvkmd_wddm_device *device, uint64_t initial, struct nvkmd_wddm_fence **out);
int32_t nvkmd_wddm_fence_point(const struct nvkmd_wddm_fence *fence, uint64_t value, struct nvkmd_wddm_point **out);
int32_t nvkmd_wddm_fence_get_value(const struct nvkmd_wddm_fence *fence, uint64_t *out);
int32_t nvkmd_wddm_fence_signal_cpu_value(struct nvkmd_wddm_fence *fence, uint64_t value);
void nvkmd_wddm_fence_destroy(struct nvkmd_wddm_fence *fence);
int32_t nvkmd_wddm_fence_signal_cpu(struct nvkmd_wddm_fence *fence, struct nvkmd_wddm_point **out);
int32_t nvkmd_wddm_point_wait(const struct nvkmd_wddm_point *point, uint64_t timeout_ns);
void nvkmd_wddm_point_destroy(struct nvkmd_wddm_point *point);
int32_t nvkmd_wddm_nvidia_device_create(struct nvkmd_wddm_luid luid, uint32_t node, uint32_t context_id, struct nvkmd_wddm_device **out);
int32_t nvkmd_wddm_context_create(const struct nvkmd_wddm_device *device, uint32_t engines, uint32_t hardware_queue, struct nvkmd_wddm_context **out);
void nvkmd_wddm_context_destroy(struct nvkmd_wddm_context *context);
int32_t nvkmd_wddm_context_retain(struct nvkmd_wddm_context *context, const struct nvkmd_wddm_binding *binding);
int32_t nvkmd_wddm_context_push(struct nvkmd_wddm_context *context, struct nvkmd_wddm_push push);
int32_t nvkmd_wddm_context_wait(struct nvkmd_wddm_context *context, const struct nvkmd_wddm_point *point);
int32_t nvkmd_wddm_context_signal(struct nvkmd_wddm_context *context, const struct nvkmd_wddm_point *point);
int32_t nvkmd_wddm_context_flush(struct nvkmd_wddm_context *context, struct nvkmd_wddm_point **out);
int32_t nvkmd_wddm_context_collect(struct nvkmd_wddm_context *context);
int32_t nvkmd_wddm_context_reflected_sked_address(const struct nvkmd_wddm_context *context, uint64_t *out);

#ifdef __cplusplus
}
#endif

#endif
