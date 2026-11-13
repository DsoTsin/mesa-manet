#include "nvk_cmd_dispatch.h"
#include "nvk_cmd_buffer.h"
#include "nvk_cubin.h"
#include "nvk_device.h"
#include "nvk_entrypoints.h"
#include "nvk_physical_device.h"
#include "nvk_queue.h"
#include "nvk_ser.h"
#include "nvk_shader.h"
#include "vk_meta.h"
#include "util/u_math.h"
#include "util/u_debug.h"
#include "nv_push_cl906f.h"
#include "clcdc0.h"
#include "clcdc0qmd.h"
#include "drf.h"
#include "nak.h"
#include <stdio.h>

static_assert(sizeof(struct nvk_ser_native_queue) == 88, "native queue size");
static_assert(sizeof(struct nvk_ser_queue) == 144, "queue size");
static_assert(sizeof(struct nvk_ser_packet) == 32, "packet size");
static_assert(sizeof(struct nvk_ser_counters) == 32, "counter size");
static_assert(offsetof(struct nvk_ser_native_queue, reflected_sked) == 48,
              "reflected sked offset");
static_assert(offsetof(struct nvk_ser_counters, completed) == 20,
              "completion offset");
static_assert(offsetof(struct nvk_ser_counters, remaining) == 28,
              "remaining offset");

#ifdef _WIN32
#include "nvkmd/wddm/nvkmd_wddm_ops.h"
#endif

#ifdef NVK_SER_CUDA_ENABLED
static void
nvk_ser_dump(const char *name, const void *data, size_t size)
{
   const char *directory = debug_get_option("NVK_WDDM_SER_DUMP", NULL);
   if (!directory)
      return;
   char path[4096];
   snprintf(path, sizeof(path), "%s/%s", directory, name);
   FILE *file = fopen(path, "wb");
   if (file) {
      fwrite(data, 1, size, file);
      fclose(file);
   }
}

static const uint8_t ser_cubin[] = {
#include "nvk_ser_queue.cubin.h"
};

static VkResult
nvk_ser_prepare_kernel(struct nvk_cmd_buffer *cmd, const char *name,
                       uint64_t queue_addr, uint32_t groups, uint32_t threads,
                       struct nvk_prepared_dispatch *out)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const struct {
      enum vk_meta_object_key_type type;
      uint32_t module;
   } module_key = { VK_META_OBJECT_KEY_DRIVER_OFFSET + 34, 0 };
   struct {
      enum vk_meta_object_key_type type;
      uint32_t module;
      char name[64];
   } function_key = { VK_META_OBJECT_KEY_DRIVER_OFFSET + 35, 0, {0} };
   assert(strlen(name) < sizeof(function_key.name));
   memcpy(function_key.name, name, strlen(name));
   VkCuFunctionNVX function_handle = (VkCuFunctionNVX)vk_meta_lookup_object(
      &dev->meta, VK_OBJECT_TYPE_CU_FUNCTION_NVX,
      &function_key, sizeof(function_key));
   if (function_handle == VK_NULL_HANDLE) {
      VkCuModuleNVX module_handle = (VkCuModuleNVX)vk_meta_lookup_object(
         &dev->meta, VK_OBJECT_TYPE_CU_MODULE_NVX,
         &module_key, sizeof(module_key));
      if (module_handle == VK_NULL_HANDLE) {
         struct nvk_cubin_module *module;
         VkResult result = nvk_cubin_module_create(dev, ser_cubin,
            sizeof(ser_cubin), NULL, &module);
         if (result != VK_SUCCESS)
            return result;
         module_handle = (VkCuModuleNVX)vk_meta_cache_object(&dev->vk, &dev->meta,
            &module_key, sizeof(module_key), VK_OBJECT_TYPE_CU_MODULE_NVX,
            (uint64_t)nvk_cubin_module_to_handle(module));
      }
      struct nvk_cubin_function *function;
      VkResult result = nvk_cubin_function_create(dev,
         nvk_cubin_module_from_handle(module_handle), name, NULL, &function);
      if (result != VK_SUCCESS)
         return result;
      function_handle = (VkCuFunctionNVX)vk_meta_cache_object(&dev->vk, &dev->meta,
         &function_key, sizeof(function_key), VK_OBJECT_TYPE_CU_FUNCTION_NVX,
         (uint64_t)nvk_cubin_function_to_handle(function));
   }
   const struct nvk_ser_kernel_args args = { .queue = queue_addr };
   const struct nvk_cubin_launch_info launch = {
      .grid_dim = { groups, 1, 1 },
      .block_dim = { threads, 1, 1 },
      .param_data = &args,
      .param_size = sizeof(args),
   };
   VkResult result = nvk_cmd_cubin_prepare(cmd,
      nvk_cubin_function_from_handle(function_handle), &launch, out);
   if (result == VK_SUCCESS)
      nvk_ser_dump(name, out->qmd_map,
         nak_qmd_size_B(&nvk_device_physical(dev)->info));
   return result;
}

static VkResult
nvk_ser_reflected_sked_address(struct nvk_device *dev, uint64_t *out)
{
   *out = 0;
#ifdef _WIN32
   vk_foreach_queue(vk_queue, &dev->vk) {
      struct nvk_queue *queue = container_of(vk_queue, struct nvk_queue, vk);
      if (!(queue->engines & NVKMD_ENGINE_COMPUTE))
         continue;
      uint64_t address = 0;
      VkResult result = nvkmd_wddm_ctx_get_reflected_sked_address(
         queue->exec_ctx, &address);
      if (result != VK_SUCCESS)
         return result;
      if (!address || (*out && *out != address))
         return VK_ERROR_FEATURE_NOT_PRESENT;
      *out = address;
   }
#endif
   return *out ? VK_SUCCESS : VK_ERROR_FEATURE_NOT_PRESENT;
}

static VkResult
nvk_ser_pack_qmd(const struct nvk_prepared_dispatch *dispatch, uint32_t *out)
{
   if (!dispatch->qmd_addr || (dispatch->qmd_addr & 255) ||
       (dispatch->qmd_addr >> 8) > UINT32_MAX)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   *out = dispatch->qmd_addr >> 8;
   return VK_SUCCESS;
}
#endif

VkResult
nvk_cmd_prepare_ser_queue(struct nvk_cmd_buffer *cmd, uint32_t owner_capacity,
                          struct nvk_ser_dispatch *out)
{
   *out = (struct nvk_ser_dispatch) {0};
#ifdef NVK_SER_CUDA_ENABLED
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const struct nvk_physical_device *pdev = nvk_device_physical(dev);
   if (pdev->info.cls_compute < BLACKWELL_COMPUTE_A ||
       owner_capacity < NVK_SER_SHADE_THREADS ||
       !util_is_power_of_two_nonzero(owner_capacity) ||
       owner_capacity / NVK_SER_SHADE_THREADS > 0x40000u)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   uint64_t reflected_sked;
   VkResult result = nvk_ser_reflected_sked_address(dev, &reflected_sked);
   if (result != VK_SUCCESS)
      return result;
   const uint64_t packets_size =
      (uint64_t)owner_capacity * sizeof(struct nvk_ser_packet);
   const uint64_t owner_ids_size = (uint64_t)owner_capacity * sizeof(uint32_t);
   const uint64_t counters_offset = align64(packets_size + owner_ids_size, 64);
   uint64_t scratch_addr;
   result = nvk_cmd_buffer_alloc_scratch(cmd,
      counters_offset + sizeof(struct nvk_ser_counters), &scratch_addr);
   if (result != VK_SUCCESS)
      return result;
   void *queue_map;
   result = nvk_cmd_buffer_upload_alloc(cmd, sizeof(struct nvk_ser_queue), 16,
      &out->queue_addr, &queue_map);
   if (result != VK_SUCCESS)
      return result;
   out->queue_map = queue_map;
   out->owner_capacity = owner_capacity;
   *out->queue_map = (struct nvk_ser_queue) {
      .native = {
         .thread_count = owner_capacity,
         .reflected_sked = reflected_sked,
      },
      .packets = scratch_addr,
      .resume_owner_ids = scratch_addr + packets_size,
      .counters = scratch_addr + counters_offset,
      .owner_capacity = owner_capacity,
      .resume_record_mask = owner_capacity - 1,
      .flags = debug_get_num_option("NVK_WDDM_SER_DEBUG_STOP", 0),
   };
   out->completion_addr = out->queue_map->counters +
      offsetof(struct nvk_ser_counters, completed);
   struct nvk_prepared_dispatch sort, tail;
   result = nvk_ser_prepare_kernel(cmd, "nvk_ser_init_native", out->queue_addr,
      DIV_ROUND_UP(owner_capacity, 256), 256, &out->init);
   if (result != VK_SUCCESS)
      return result;
   uint32_t init_qmd;
   result = nvk_ser_pack_qmd(&out->init, &init_qmd);
   if (result != VK_SUCCESS)
      return result;
   result = nvk_ser_prepare_kernel(cmd, "nvk_ser_sort_native", out->queue_addr,
      DIV_ROUND_UP(owner_capacity, NVK_SER_SORT_CAPACITY),
      NVK_SER_SORT_THREADS, &sort);
   if (result != VK_SUCCESS)
      return result;
   result = nvk_ser_prepare_kernel(cmd, "nvk_ser_tail_native", out->queue_addr,
      1, NVK_SER_SHADE_THREADS, &tail);
   if (result != VK_SUCCESS)
      return result;
   result = nvk_ser_pack_qmd(&sort, &out->queue_map->native.sort_qmd);
   if (result != VK_SUCCESS)
      return result;
   return nvk_ser_pack_qmd(&tail, &out->queue_map->native.tail_qmd);
#else
   return VK_ERROR_FEATURE_NOT_PRESENT;
#endif
}

VkResult
nvk_cmd_prepare_ser_shade(struct nvk_cmd_buffer *cmd,
                          struct nvk_ser_dispatch *dispatch)
{
#ifdef NVK_SER_CUDA_ENABLED
   const uint32_t groups = dispatch->owner_capacity / NVK_SER_SHADE_THREADS;
   const uint32_t group_count[3] = { groups, 1, 1 };
   VkResult result = nvk_cmd_prepare_cs_dispatch(cmd, group_count,
      &dispatch->shade);
   if (result != VK_SUCCESS)
      return result;
   result = nvk_ser_pack_qmd(&dispatch->shade,
      &dispatch->queue_map->native.shade_qmd);
   if (result != VK_SUCCESS)
      return result;
   uint32_t *qmd = dispatch->shade.qmd_map;
   NVVAL_MW_SET(qmd, NVCDC0, QMDV05_00, QMD_TYPE,
      NVCDC0_QMDV05_00_QMD_TYPE_QUEUE);
   NVVAL_MW_SET(qmd, NVCDC0, QMDV05_00, REQUIRE_SCHEDULING_PCAS, 0);
   NVVAL_MW_SET(qmd, NVCDC0, QMDV05_00, INNER_GET, 0);
   NVVAL_MW_SET(qmd, NVCDC0, QMDV05_00, INNER_PUT, groups);
   NVVAL_MW_SET(qmd, NVCDC0, QMDV05_00, CIRCULAR_QUEUE_SIZE, MAX2(groups, 1024));
   NVVAL_MW_SET(qmd, NVCDC0, QMDV05_00, QUEUE_ENTRIES_PER_CTA_LOG2, 0);
   NVVAL_MW_SET(qmd, NVCDC0, QMDV05_00, GRID_WIDTH, 0);
   NVVAL_MW_SET(qmd, NVCDC0, QMDV05_00, GRID_HEIGHT, 0);
   NVVAL_MW_SET(qmd, NVCDC0, QMDV05_00, GRID_DEPTH, 0);
   NVVAL_MW_SET(qmd, NVCDC0, QMDV05_00, HW_ONLY_INNER_GET, 0);
   NVVAL_MW_SET(qmd, NVCDC0, QMDV05_00, HW_ONLY_INNER_PUT, 0);
   nvk_ser_dump("shade.qmd", qmd,
      nak_qmd_size_B(&nvk_device_physical(nvk_cmd_buffer_device(cmd))->info));
   nvk_ser_dump("shade.code", cmd->state.cs.shader->code_ptr,
      cmd->state.cs.shader->code_size);
   nvk_ser_dump("queue.bin", dispatch->queue_map, sizeof(*dispatch->queue_map));
   return VK_SUCCESS;
#else
   return VK_ERROR_FEATURE_NOT_PRESENT;
#endif
}

void
nvk_cmd_launch_ser_dispatch(struct nvk_cmd_buffer *cmd,
                            const struct nvk_ser_dispatch *dispatch)
{
   nvk_cmd_launch_prepared_dispatch(cmd, &dispatch->init, true);
   const VkMemoryBarrier2 initialize_barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
   };
   const VkDependencyInfo initialize_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &initialize_barrier,
   };
   nvk_CmdPipelineBarrier2(nvk_cmd_buffer_to_handle(cmd), &initialize_dep);
   nvk_cmd_launch_prepared_dispatch(cmd, &dispatch->shade, true);
   struct nv_push *p = nvk_cmd_buffer_push(cmd, 5);
   __push_mthd(p, nvk_cmd_buffer_last_subchannel(cmd), NV906F_SEMAPHOREA);
   P_NV906F_SEMAPHOREA(p, dispatch->completion_addr >> 32);
   P_NV906F_SEMAPHOREB(p, (dispatch->completion_addr & UINT32_MAX) >> 2);
   P_NV906F_SEMAPHOREC(p, 1);
   P_NV906F_SEMAPHORED(p, {
      .operation = OPERATION_ACQUIRE,
      .acquire_switch = ACQUIRE_SWITCH_ENABLED,
      .release_size = RELEASE_SIZE_4BYTE,
   });
   const VkMemoryBarrier2 complete_barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
   };
   const VkDependencyInfo complete_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &complete_barrier,
   };
   nvk_CmdPipelineBarrier2(nvk_cmd_buffer_to_handle(cmd), &complete_dep);
}
