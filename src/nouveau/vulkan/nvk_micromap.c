#include "nvk_micromap.h"
#include "nvk_bvh_cuda.h"
#include "nvk_cmd_buffer.h"
#include "nvk_device.h"
#include "nvk_entrypoints.h"
#include "nvk_query_pool.h"
#include "nvkmd/nvkmd.h"
#include "bvh/nvk_bvh_omm.h"
#include "vk_acceleration_structure.h"
#include "vk_command_buffer.h"

static const uint32_t nvk_micromap_uuid[8] = {
   0x4e564b4du, 0x30323101u, 1, 0,
   0x4e564b4du, 0x30323101u, 1, 0,
};

struct nvk_micromap_sizes {
   uint32_t count;
   uint64_t data;
   uint64_t size;
   uint64_t scratch;
};

static struct nvk_micromap_sizes
nvk_micromap_sizes(const VkMicromapBuildInfoEXT *info)
{
   assert(info->type == VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT);
   struct nvk_micromap_sizes sizes = {0};
   for (uint32_t index = 0; index < info->usageCountsCount; index++) {
      const VkMicromapUsageEXT *usage = info->pUsageCounts ?
         &info->pUsageCounts[index] : info->ppUsageCounts[index];
      assert(usage->subdivisionLevel <= 12);
      assert(usage->format == VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT ||
             usage->format == VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT);
      sizes.count += usage->count;
      sizes.data += (uint64_t)usage->count *
                    nvk_omm_map_size(usage->subdivisionLevel, usage->format);
   }
   sizes.size = nvk_omm_array_size(sizes.count, sizes.data);
   sizes.scratch = nvk_omm_scratch_size(sizes.count);
   sizes.data += (uint64_t)nvk_omm_group_count(sizes.count) * 128;
   return sizes;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_CreateMicromapEXT(VkDevice device, const VkMicromapCreateInfoEXT *info,
                      const VkAllocationCallbacks *allocator, VkMicromapEXT *handle)
{
   VK_FROM_HANDLE(nvk_device, dev, device);
   VK_FROM_HANDLE(vk_buffer, buffer, info->buffer);
   struct nvk_micromap *micromap = vk_object_zalloc(&dev->vk, allocator,
      sizeof(*micromap), VK_OBJECT_TYPE_MICROMAP_EXT);
   if (!micromap)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   micromap->buffer = buffer;
   micromap->offset = info->offset;
   micromap->size = info->size;
   if (info->deviceAddress && info->deviceAddress != nvk_micromap_get_va(micromap)) {
      vk_object_free(&dev->vk, allocator, micromap);
      return vk_error(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
   }
   *handle = nvk_micromap_to_handle(micromap);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
nvk_DestroyMicromapEXT(VkDevice device, VkMicromapEXT handle,
                       const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(nvk_device, dev, device);
   VK_FROM_HANDLE(nvk_micromap, micromap, handle);
   if (micromap)
      vk_object_free(&dev->vk, allocator, micromap);
}

VKAPI_ATTR void VKAPI_CALL
nvk_GetMicromapBuildSizesEXT(VkDevice device, VkAccelerationStructureBuildTypeKHR type,
                            const VkMicromapBuildInfoEXT *info,
                            VkMicromapBuildSizesInfoEXT *output)
{
   const struct nvk_micromap_sizes sizes = nvk_micromap_sizes(info);
   output->micromapSize = sizes.size;
   output->buildScratchSize = sizes.scratch;
   output->discardable = VK_FALSE;
}

static VkResult
nvk_micromap_launch_build(struct nvk_cmd_buffer *cmd,
                          const VkMicromapBuildInfoEXT *info)
{
   VK_FROM_HANDLE(nvk_micromap, micromap, info->dstMicromap);
   const struct nvk_micromap_sizes sizes = nvk_micromap_sizes(info);
   struct nvk_bvh_omm_build_args args = {
      .dst = nvk_micromap_get_va(micromap),
      .data = info->data.deviceAddress,
      .triangles = info->triangleArray.deviceAddress,
      .scratch = info->scratchData.deviceAddress,
      .size = sizes.size,
      .data_size = sizes.data,
      .stride = info->triangleArrayStride,
      .count = sizes.count,
      .flags = info->flags,
   };
   memcpy(args.uuid, nvk_micromap_uuid, sizeof(args.uuid));
   const char *stages[] = {
      "nvk_bvh_omm_init_native", "nvk_bvh_omm_scan_tiles_native",
      "nvk_bvh_omm_scan_sums_native", "nvk_bvh_omm_scan_add_native",
      "nvk_bvh_omm_transcode_native", "nvk_bvh_omm_hull_native",
   };
   for (uint32_t stage = 0; stage < ARRAY_SIZE(stages); stage++) {
      vk_bvh_build_barrier_compute_to_compute(nvk_cmd_buffer_to_handle(cmd), false);
      uint32_t block = stage == 1 || stage == 2 ? NVK_OMM_SCAN_THREADS : NVK_OMM_BUILD_THREADS;
      uint32_t grid = stage == 2 ? 1 : stage == 5 ? sizes.count :
         stage >= 1 && stage <= 3 ? DIV_ROUND_UP(nvk_omm_group_count(sizes.count), 256) :
         stage == 4 ? nvk_omm_group_count(sizes.count) : DIV_ROUND_UP(sizes.count, 128);
      VkResult result = nvk_bvh_cuda_launch(cmd, stages[stage], &args, sizeof(args),
                                             CLAMP(grid, 1, 256), block);
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdBuildMicromapsEXT(VkCommandBuffer command_buffer, uint32_t count,
                        const VkMicromapBuildInfoEXT *infos)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   struct nvk_compute_state saved;
   nvk_bvh_meta_begin(cmd, &saved);
   for (uint32_t index = 0; index < count; index++) {
      VkResult result = nvk_micromap_launch_build(cmd, &infos[index]);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         break;
      }
   }
   nvk_bvh_meta_end(cmd, &saved);
}

static void
nvk_micromap_copy(VkCommandBuffer command_buffer, struct nvk_bvh_omm_copy_args args)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   struct nvk_compute_state saved;
   nvk_bvh_meta_begin(cmd, &saved);
   vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
   VkResult result = nvk_bvh_cuda_launch(cmd, "nvk_bvh_omm_copy_native", &args, sizeof(args),
                                          64, NVK_OMM_COPY_THREADS);
   if (result != VK_SUCCESS)
      vk_command_buffer_set_error(&cmd->vk, result);
   nvk_bvh_meta_end(cmd, &saved);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdCopyMicromapEXT(VkCommandBuffer command_buffer, const VkCopyMicromapInfoEXT *info)
{
   VK_FROM_HANDLE(nvk_micromap, src, info->src);
   VK_FROM_HANDLE(nvk_micromap, dst, info->dst);
   nvk_micromap_copy(command_buffer, (struct nvk_bvh_omm_copy_args) {
      .src = nvk_micromap_get_va(src), .dst = nvk_micromap_get_va(dst),
      .size = src->size, .mode = NVK_OMM_COPY_CLONE,
   });
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdCopyMicromapToMemoryEXT(VkCommandBuffer command_buffer,
                             const VkCopyMicromapToMemoryInfoEXT *info)
{
   VK_FROM_HANDLE(nvk_micromap, src, info->src);
   nvk_micromap_copy(command_buffer, (struct nvk_bvh_omm_copy_args) {
      .src = nvk_micromap_get_va(src), .dst = info->dst.deviceAddress,
      .size = src->size, .mode = NVK_OMM_COPY_SERIALIZE,
   });
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdCopyMemoryToMicromapEXT(VkCommandBuffer command_buffer,
                             const VkCopyMemoryToMicromapInfoEXT *info)
{
   VK_FROM_HANDLE(nvk_micromap, dst, info->dst);
   nvk_micromap_copy(command_buffer, (struct nvk_bvh_omm_copy_args) {
      .src = info->src.deviceAddress, .dst = nvk_micromap_get_va(dst),
      .size = dst->size, .mode = NVK_OMM_COPY_DESERIALIZE,
   });
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdWriteMicromapsPropertiesEXT(VkCommandBuffer command_buffer, uint32_t count,
                                  const VkMicromapEXT *handles, VkQueryType type,
                                  VkQueryPool query_pool, uint32_t first_query)
{
   VK_FROM_HANDLE(nvk_query_pool, pool, query_pool);
   const uint64_t base = pool->mem->va->addr;
   const uint64_t available_stride = pool->layout == NVK_QUERY_POOL_LAYOUT_SEPARATE ?
                                      sizeof(uint32_t) : pool->query_stride;
   for (uint32_t index = 0; index < count; index++) {
      VK_FROM_HANDLE(nvk_micromap, micromap, handles[index]);
      const uint32_t query = first_query + index;
      nvk_micromap_copy(command_buffer, (struct nvk_bvh_omm_copy_args) {
         .src = nvk_micromap_get_va(micromap),
         .dst = base + pool->reports_start + (uint64_t)query * pool->query_stride,
         .available = base + (uint64_t)query * available_stride,
         .mode = NVK_OMM_COPY_PROPERTY,
         .field = type == VK_QUERY_TYPE_MICROMAP_SERIALIZATION_SIZE_EXT ?
                     NVK_OMM_PROPERTY_SERIALIZED_SIZE : NVK_OMM_PROPERTY_COMPACTED_SIZE,
      });
   }
}

VKAPI_ATTR void VKAPI_CALL
nvk_GetDeviceMicromapCompatibilityEXT(VkDevice device,
                                     const VkMicromapVersionInfoEXT *info,
                                     VkAccelerationStructureCompatibilityKHR *output)
{
   *output = memcmp(info->pVersionData, nvk_micromap_uuid, sizeof(nvk_micromap_uuid)) == 0 ?
      VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR :
      VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_BuildMicromapsEXT(VkDevice device, VkDeferredOperationKHR operation,
                      uint32_t count, const VkMicromapBuildInfoEXT *infos)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_CopyMicromapEXT(VkDevice device, VkDeferredOperationKHR operation,
                    const VkCopyMicromapInfoEXT *info)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_CopyMicromapToMemoryEXT(VkDevice device, VkDeferredOperationKHR operation,
                            const VkCopyMicromapToMemoryInfoEXT *info)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_CopyMemoryToMicromapEXT(VkDevice device, VkDeferredOperationKHR operation,
                            const VkCopyMemoryToMicromapInfoEXT *info)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_WriteMicromapsPropertiesEXT(VkDevice device, uint32_t count,
                                const VkMicromapEXT *micromaps, VkQueryType type,
                                size_t size, void *data, size_t stride)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}
