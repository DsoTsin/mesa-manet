#include "panvk_micromap.h"

#include "bvh/panvk_micromap.h"
#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_meta.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_instance.h"
#include "panvk_meta.h"
#include "panvk_physical_device.h"
#include "vk_acceleration_structure.h"

static const uint32_t micromap_spv[] = {
#include "bvh/micromap.spv.h"
};

static const uint32_t micromap_copy_spv[] = {
#include "bvh/micromap_copy.spv.h"
};

struct micromap_layout {
   struct panvk_micromap_bucket buckets[PANVK_MICROMAP_BUCKETS];
   uint32_t count;
   uint32_t usage_count;
   uint32_t summary_offset;
   VkDeviceSize size;
   bool discardable;
};

static struct micromap_layout
micromap_layout(const VkMicromapBuildInfoEXT *info)
{
   struct micromap_layout layout = {.discardable = true};
   uint32_t counts[PANVK_MICROMAP_BUCKETS] = {0};
   uint32_t order[PANVK_MICROMAP_BUCKETS];
   VkDeviceSize high_size = 0;

   for (uint32_t i = 0; i < info->usageCountsCount; i++) {
      const VkMicromapUsageEXT *usage = info->pUsageCounts ?
         &info->pUsageCounts[i] : info->ppUsageCounts[i];
      uint32_t level = MAX2(usage->subdivisionLevel, 3);
      uint32_t format = usage->subdivisionLevel <= 3 ? 2 : usage->format;
      assert(level <= PANVK_MICROMAP_MAX_LEVEL && format >= 1 && format <= 2);
      uint32_t bucket = level + 8 * format - 11;
      if (level > 3)
         layout.discardable = false;
      if (!usage->count)
         continue;
      if (!counts[bucket])
         order[layout.usage_count++] = bucket;
      counts[bucket] += usage->count;
      layout.count += usage->count;
      if (level > 3) {
         high_size += (VkDeviceSize)usage->count *
            ALIGN_POT(((format << (2 * level)) + 7) / 8, 64);
      }
   }

   VkDeviceSize low = ALIGN_POT(PANVK_MICROMAP_HEADER_SIZE +
                              (VkDeviceSize)layout.count * 8, 16);
   VkDeviceSize low_size = (VkDeviceSize)counts[8] * 16;
   if (info->flags & VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT) {
      layout.summary_offset = layout.count ? low : 0;
      low_size = (VkDeviceSize)layout.count * 16;
   }
   VkDeviceSize high = ALIGN_POT(low + low_size, 64);
   layout.size = high_size ? high + high_size : low_size ? low + low_size :
      PANVK_MICROMAP_HEADER_SIZE + (VkDeviceSize)layout.count * 8;

   for (uint32_t i = 0; i < layout.usage_count; i++) {
      uint32_t bucket = order[i];
      uint32_t format = bucket < 8 ? 1 : 2;
      uint32_t level = bucket + 11 - 8 * format;
      uint32_t stride = level == 3 ? 16 :
         ALIGN_POT(((format << (2 * level)) + 7) / 8, 64);
      layout.buckets[bucket] = (struct panvk_micromap_bucket){
         .offset = level == 3 ? low : high,
         .stride = stride,
         .counter = i,
      };
      if (level > 3)
         high += (VkDeviceSize)counts[bucket] * stride;
   }
   return layout;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CreateMicromapEXT)(
   VkDevice _device, const VkMicromapCreateInfoEXT *info,
   const VkAllocationCallbacks *allocator, VkMicromapEXT *handle)
{
   VK_FROM_HANDLE(panvk_device, dev, _device);
   VK_FROM_HANDLE(vk_buffer, buffer, info->buffer);
   struct panvk_micromap *micromap = vk_object_zalloc(
      &dev->vk, allocator, sizeof(*micromap), VK_OBJECT_TYPE_MICROMAP_EXT);
   if (!micromap)
      return panvk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   micromap->buffer = buffer;
   micromap->offset = info->offset;
   micromap->size = info->size;
   if (info->deviceAddress && panvk_micromap_address(micromap) != info->deviceAddress) {
      vk_object_free(&dev->vk, allocator, micromap);
      return panvk_error(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
   }
   *handle = panvk_micromap_to_handle(micromap);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(DestroyMicromapEXT)(
   VkDevice _device, VkMicromapEXT handle,
   const VkAllocationCallbacks *allocator)
{
   VK_FROM_HANDLE(panvk_device, dev, _device);
   VK_FROM_HANDLE(panvk_micromap, micromap, handle);
   if (micromap)
      vk_object_free(&dev->vk, allocator, micromap);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(GetMicromapBuildSizesEXT)(
   VkDevice device, VkAccelerationStructureBuildTypeKHR type,
   const VkMicromapBuildInfoEXT *info, VkMicromapBuildSizesInfoEXT *sizes)
{
   struct micromap_layout layout = micromap_layout(info);
   sizes->micromapSize = layout.size;
   sizes->buildScratchSize = type == VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR ?
      0 : (VkDeviceSize)info->usageCountsCount * sizeof(VkMicromapUsageEXT);
   if (type != VK_ACCELERATION_STRUCTURE_BUILD_TYPE_HOST_KHR)
      sizes->buildScratchSize = MAX2(sizes->buildScratchSize,
                                     (VkDeviceSize)MAX2(layout.count, 1) * 16);
   sizes->discardable = layout.discardable;
}

static VkResult
bind_micromap_pipeline(struct panvk_cmd_buffer *cmd,
                       enum panvk_meta_object_key_type key,
                       const uint32_t *spv, uint32_t size, uint32_t push_size,
                       VkPipelineLayout *layout)
{
   struct panvk_device *dev = to_panvk_device(cmd->vk.base.device);
   const struct vk_acceleration_structure_build_args args = {
      .subgroup_size = pan_subgroup_size(PAN_ARCH),
   };
   VkPipeline pipeline;
   VkResult result = vk_get_bvh_build_pipeline_spv(
      &dev->vk, &dev->meta, (enum vk_meta_object_key_type)key, spv, size,
      push_size, &args, 0, &pipeline, false);
   if (result == VK_SUCCESS)
      result = vk_get_bvh_build_pipeline_layout(&dev->vk, &dev->meta,
                                                push_size, layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return result;
   }
   dev->vk.dispatch_table.CmdBindPipeline(panvk_cmd_buffer_to_handle(cmd),
                                          VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBuildMicromapsEXT)(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkMicromapBuildInfoEXT *infos)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   struct panvk_device *dev = to_panvk_device(cmd->vk.base.device);
   struct panvk_cmd_meta_compute_save_ctx save = {0};
   VkPipelineLayout pipeline_layout;
   panvk_per_arch(cmd_meta_compute_start)(cmd, &save);
   if (bind_micromap_pipeline(cmd, PANVK_META_OBJECT_KEY_MICROMAP_BUILD,
      micromap_spv, sizeof(micromap_spv), sizeof(struct panvk_micromap_build_args),
      &pipeline_layout) != VK_SUCCESS)
      goto end;

   for (uint32_t i = 0; i < infoCount; i++) {
      VK_FROM_HANDLE(panvk_micromap, micromap, infos[i].dstMicromap);
      const struct micromap_layout layout = micromap_layout(&infos[i]);
      const uint64_t address = panvk_micromap_address(micromap);
      struct panvk_micromap_header header = {
         .size = layout.size,
         .flags = 0x10000,
         .version = 3,
         .triangle_count = layout.count,
         .active_count = layout.count,
         .usage_count = layout.usage_count,
         .summary_offset = layout.summary_offset,
      };
      dev->vk.write_buffer_cp(commandBuffer, address, &header, sizeof(header));
      if (!layout.count)
         continue;
      struct pan_ptr buckets = panvk_cmd_alloc_dev_mem(cmd, desc,
         sizeof(layout.buckets), 8);
      if (!buckets.gpu)
         break;
      memcpy(buckets.cpu, layout.buckets, sizeof(layout.buckets));
      dev->vk.cmd_fill_buffer_addr(commandBuffer,
         infos[i].scratchData.deviceAddress, MAX2(layout.usage_count, 1) * 4, 0);
      dev->vk.flush_buffer_write_cp(commandBuffer);
      vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
      struct panvk_micromap_build_args push = {
         .src = infos[i].data.deviceAddress,
         .triangles = infos[i].triangleArray.deviceAddress,
         .dst = address,
         .buckets = buckets.gpu,
         .counters = infos[i].scratchData.deviceAddress,
         .triangle_stride = infos[i].triangleArrayStride,
         .summary_offset = layout.summary_offset,
      };
      dev->vk.dispatch_table.CmdPushConstants(commandBuffer, pipeline_layout,
         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
      uint32_t width = MIN2(layout.count, 65535);
      dev->vk.dispatch_table.CmdDispatch(commandBuffer, width,
                                          DIV_ROUND_UP(layout.count, width), 1);
   }
   dev->vk.flush_buffer_write_cp(commandBuffer);
end:
   panvk_per_arch(cmd_meta_compute_end)(cmd, &save);
}

static void
micromap_uuid(struct panvk_device *dev, uint32_t uuid[8])
{
   const struct panvk_physical_device *phys = to_panvk_physical_device(dev->vk.physical);
   const struct panvk_instance *instance = to_panvk_instance(phys->vk.instance);
   memcpy(uuid, instance->driver_build_sha, VK_UUID_SIZE);
   memcpy(uuid + VK_UUID_SIZE / 4, phys->cache_uuid, VK_UUID_SIZE);
}

static void
cmd_copy_micromap(VkCommandBuffer commandBuffer, uint64_t src, uint64_t dst,
                  uint32_t mode)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmd, commandBuffer);
   struct panvk_device *dev = to_panvk_device(cmd->vk.base.device);
   struct panvk_cmd_meta_compute_save_ctx save = {0};
   struct panvk_micromap_copy_args push = {.src = src, .dst = dst, .mode = mode};
   VkPipelineLayout layout;
   micromap_uuid(dev, push.uuid);
   panvk_per_arch(cmd_meta_compute_start)(cmd, &save);
   if (bind_micromap_pipeline(cmd, PANVK_META_OBJECT_KEY_MICROMAP_COPY,
      micromap_copy_spv, sizeof(micromap_copy_spv), sizeof(push), &layout) == VK_SUCCESS) {
      dev->vk.dispatch_table.CmdPushConstants(commandBuffer, layout,
         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
      dev->vk.dispatch_table.CmdDispatch(commandBuffer, 64, 1, 1);
   }
   panvk_per_arch(cmd_meta_compute_end)(cmd, &save);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdCopyMicromapEXT)(
   VkCommandBuffer commandBuffer, const VkCopyMicromapInfoEXT *info)
{
   VK_FROM_HANDLE(panvk_micromap, src, info->src);
   VK_FROM_HANDLE(panvk_micromap, dst, info->dst);
   cmd_copy_micromap(commandBuffer, panvk_micromap_address(src),
                     panvk_micromap_address(dst), 0);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdCopyMicromapToMemoryEXT)(
   VkCommandBuffer commandBuffer, const VkCopyMicromapToMemoryInfoEXT *info)
{
   VK_FROM_HANDLE(panvk_micromap, src, info->src);
   cmd_copy_micromap(commandBuffer, panvk_micromap_address(src),
                     info->dst.deviceAddress, 1);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdCopyMemoryToMicromapEXT)(
   VkCommandBuffer commandBuffer, const VkCopyMemoryToMicromapInfoEXT *info)
{
   VK_FROM_HANDLE(panvk_micromap, dst, info->dst);
   cmd_copy_micromap(commandBuffer, info->src.deviceAddress,
                     panvk_micromap_address(dst), 2);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(GetDeviceMicromapCompatibilityEXT)(
   VkDevice _device, const VkMicromapVersionInfoEXT *info,
   VkAccelerationStructureCompatibilityKHR *compatibility)
{
   VK_FROM_HANDLE(panvk_device, dev, _device);
   uint32_t uuid[8];
   micromap_uuid(dev, uuid);
   *compatibility = !memcmp(info->pVersionData, uuid, sizeof(uuid)) ?
      VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR :
      VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(BuildMicromapsEXT)(VkDevice device,
                                  VkDeferredOperationKHR deferredOperation,
                                  uint32_t infoCount,
                                  const VkMicromapBuildInfoEXT *infos)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CopyMicromapEXT)(VkDevice device,
                                VkDeferredOperationKHR deferredOperation,
                                const VkCopyMicromapInfoEXT *info)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CopyMicromapToMemoryEXT)(
   VkDevice device, VkDeferredOperationKHR deferredOperation,
   const VkCopyMicromapToMemoryInfoEXT *info)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CopyMemoryToMicromapEXT)(
   VkDevice device, VkDeferredOperationKHR deferredOperation,
   const VkCopyMemoryToMicromapInfoEXT *info)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(WriteMicromapsPropertiesEXT)(
   VkDevice device, uint32_t micromapCount, const VkMicromapEXT *micromaps,
   VkQueryType queryType, size_t dataSize, void *data, VkDeviceSize stride)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}
