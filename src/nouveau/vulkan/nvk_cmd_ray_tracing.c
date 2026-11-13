#include "nvk_rt_pipeline.h"
#include "nvk_cmd_buffer.h"
#include "nvk_cmd_dispatch.h"
#include "nvk_device.h"
#include "nvk_entrypoints.h"
#include "nvk_physical_device.h"
#include "vk_command_buffer.h"
#include "util/u_math.h"

static_assert(offsetof(struct nvk_rt_dispatch, stack_stride) == 104, "stack stride offset");
static_assert(offsetof(struct nvk_rt_dispatch, push_addr) == 112, "push address offset");
static_assert(offsetof(struct nvk_rt_dispatch, indirect_addr) == 128, "indirect address offset");

static VkResult
nvk_cmd_alloc_rt_stack(struct nvk_cmd_buffer *cmd,
                       struct nvk_rt_pipeline *pipeline,
                       struct nvk_rt_dispatch *dispatch, uint32_t *groups)
{
   uint64_t group_size = (uint64_t)pipeline->storage_size * NVK_RT_WORKGROUP_SIZE;
   if (!group_size)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   uint32_t max_groups = MIN2(UINT32_MAX, (256ull * 1024 * 1024) / group_size);
   if (pipeline->uses_ser_queue) {
      if (!max_groups)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      max_groups = 1u << util_logbase2(max_groups);
   }
   *groups = MIN2(*groups, MAX2(1, max_groups));
   for (;;) {
      VkResult result = nvk_cmd_buffer_alloc_scratch(cmd,
         group_size * *groups, &dispatch->stack_addr);
      if (result == VK_SUCCESS)
         return VK_SUCCESS;
      if (result != VK_ERROR_OUT_OF_DEVICE_MEMORY || *groups == 1)
         return result;
      *groups = MAX2(*groups / 2, 1);
   }
}

static void
nvk_cmd_trace_rays(struct nvk_cmd_buffer *cmd, struct nvk_rt_dispatch *dispatch)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const struct nvk_physical_device *pdev = nvk_device_physical(dev);
   struct nvk_rt_pipeline *pipeline = cmd->state.rt.pipeline;
   if (!pipeline || !pipeline->shader) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
      return;
   }
   if (pipeline->uses_ser_queue) {
      ASSERTED const uint32_t queue_family = cmd->vk.pool->queue_family_index;
      assert(pdev->queue_families[queue_family].queue_count == 1);
   }
   uint64_t rays = (uint64_t)dispatch->command.width * dispatch->command.height;
   if (dispatch->command.depth && rays > UINT64_MAX / dispatch->command.depth) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   rays *= dispatch->command.depth;
   if (!dispatch->indirect_mode && !rays)
      return;
   uint32_t sms = MAX2((uint32_t)pdev->info.tpc_count * pdev->info.mp_per_tpc, 1);
   uint32_t groups = sms * 4;
   if (!dispatch->indirect_mode)
      groups = MIN2(groups, DIV_ROUND_UP(rays, NVK_RT_WORKGROUP_SIZE));
   if (pipeline->uses_ser_queue)
      groups = util_next_power_of_two(groups);
   dispatch->stack_stride = pipeline->dynamic_stack_size ?
      MIN2(cmd->state.rt.stack_size, pipeline->stack_size) : pipeline->stack_size;
   VkResult result = nvk_device_ensure_slm(dev, pipeline->shader->info.slm_size,
                                          pipeline->shader->info.crs_size);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   if (pipeline->storage_size) {
      result = nvk_cmd_alloc_rt_stack(cmd, pipeline, dispatch, &groups);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return;
      }
   }
   dispatch->worker_count = groups * NVK_RT_WORKGROUP_SIZE;
   struct nvk_ser_dispatch ser = {0};
   if (pipeline->uses_ser_queue) {
      result = nvk_cmd_prepare_ser_queue(cmd, dispatch->worker_count, &ser);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return;
      }
      dispatch->ser_queue_addr = ser.queue_addr;
   }
   result = nvk_cmd_buffer_upload_data(cmd, cmd->state.rt.push,
                                       sizeof(cmd->state.rt.push),
                                       16, &dispatch->push_addr);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   uint64_t dispatch_addr;
   result = nvk_cmd_buffer_upload_data(cmd, dispatch, sizeof(*dispatch), 16,
                                       &dispatch_addr);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   nvk_cmd_buffer_flush_push_descriptors(cmd, &cmd->state.rt.descriptors);
   struct nvk_compute_state saved = cmd->state.cs;
   VkQueryPipelineStatisticFlags saved_statistics =
      cmd->state.inherited_pipeline_statistics;
   cmd->state.cs.descriptors = cmd->state.rt.descriptors;
   cmd->state.cs.shader = pipeline->shader;
   cmd->state.cs.active_compute_invocations_query = false;
   cmd->state.inherited_pipeline_statistics &=
      ~VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT;
   nvk_descriptor_state_set_root(cmd, &cmd->state.cs.descriptors,
                                  rt_dispatch_addr, dispatch_addr);
   if (pipeline->uses_ser_queue) {
      result = nvk_cmd_prepare_ser_shade(cmd, &ser);
      if (result == VK_SUCCESS)
         nvk_cmd_launch_ser_dispatch(cmd, &ser);
      else
         vk_command_buffer_set_error(&cmd->vk, result);
   } else {
      nvk_CmdDispatchBase(nvk_cmd_buffer_to_handle(cmd), 0, 0, 0, groups, 1, 1);
   }
   if (pipeline->storage_size &&
       (cmd->usage_flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT)) {
      VkMemoryBarrier2 barrier = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      };
      VkDependencyInfo dep = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &barrier,
      };
      nvk_CmdPipelineBarrier2(nvk_cmd_buffer_to_handle(cmd), &dep);
   }
   cmd->state.cs = saved;
   cmd->state.inherited_pipeline_statistics = saved_statistics;
}

static struct nvk_rt_dispatch
nvk_rt_dispatch_init(const VkStridedDeviceAddressRegionKHR *raygen,
                       const VkStridedDeviceAddressRegionKHR *miss,
                       const VkStridedDeviceAddressRegionKHR *hit,
                       const VkStridedDeviceAddressRegionKHR *callable)
{
   return (struct nvk_rt_dispatch) {
      .command = {
         .raygenShaderRecordAddress = raygen->deviceAddress,
         .raygenShaderRecordSize = raygen->size,
         .missShaderBindingTableAddress = miss->deviceAddress,
         .missShaderBindingTableSize = miss->size,
         .missShaderBindingTableStride = miss->stride,
         .hitShaderBindingTableAddress = hit->deviceAddress,
         .hitShaderBindingTableSize = hit->size,
         .hitShaderBindingTableStride = hit->stride,
         .callableShaderBindingTableAddress = callable->deviceAddress,
         .callableShaderBindingTableSize = callable->size,
         .callableShaderBindingTableStride = callable->stride,
      },
   };
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdTraceRaysKHR(VkCommandBuffer command_buffer,
                    const VkStridedDeviceAddressRegionKHR *raygen,
                    const VkStridedDeviceAddressRegionKHR *miss,
                    const VkStridedDeviceAddressRegionKHR *hit,
                    const VkStridedDeviceAddressRegionKHR *callable,
                    uint32_t width, uint32_t height, uint32_t depth)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   struct nvk_rt_dispatch dispatch = nvk_rt_dispatch_init(raygen, miss, hit, callable);
   dispatch.command.width = width;
   dispatch.command.height = height;
   dispatch.command.depth = depth;
   nvk_cmd_trace_rays(cmd, &dispatch);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdTraceRaysIndirectKHR(VkCommandBuffer command_buffer,
                            const VkStridedDeviceAddressRegionKHR *raygen,
                            const VkStridedDeviceAddressRegionKHR *miss,
                            const VkStridedDeviceAddressRegionKHR *hit,
                            const VkStridedDeviceAddressRegionKHR *callable,
                            VkDeviceAddress indirect)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   struct nvk_rt_dispatch dispatch = nvk_rt_dispatch_init(raygen, miss, hit, callable);
   dispatch.indirect_mode = 1;
   dispatch.indirect_addr = indirect;
   nvk_cmd_trace_rays(cmd, &dispatch);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdTraceRaysIndirect2KHR(VkCommandBuffer command_buffer,
                             VkDeviceAddress indirect)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   struct nvk_rt_dispatch dispatch = {
      .indirect_mode = 2,
      .indirect_addr = indirect,
   };
   nvk_cmd_trace_rays(cmd, &dispatch);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdSetRayTracingPipelineStackSizeKHR(VkCommandBuffer command_buffer,
                                         uint32_t stack_size)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   cmd->state.rt.stack_size = stack_size;
}
