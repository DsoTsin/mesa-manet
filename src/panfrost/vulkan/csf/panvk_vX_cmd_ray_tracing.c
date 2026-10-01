#include "panvk_cmd_alloc.h"
#include "panvk_cmd_ray_tracing.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_physical_device.h"
#include "panvk_rt_pipeline.h"
#include "panvk_nir_rt_pipeline.h"
#include "pan_props.h"
#include "vk_log.h"
#include "util/u_math.h"

static VkResult
get_stack(struct panvk_cmd_buffer *cmdbuf, uint64_t size,
          uint64_t *base, uint64_t *stride)
{
   const struct panvk_physical_device *pdev =
      to_panvk_physical_device(cmdbuf->vk.base.device->physical);
   const struct pan_kmod_dev_props *props = &pdev->kmod.dev->props;
   uint64_t lane_count = (uint64_t)pan_query_core_id_range(props) *
                         props->max_threads_per_core;
   if (!size || size > UINT64_MAX - 127)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   size = ALIGN_POT(size, 128);
   if (!lane_count || size > SIZE_MAX / lane_count)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct panvk_cmd_ray_tracing_state *state = &cmdbuf->state.ray_tracing;
   if (size > state->stack_stride) {
      struct pan_ptr stack = panvk_cmd_alloc_dev_mem(cmdbuf, tls,
                                                     size * lane_count, 4096);
      if (!stack.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      state->stack_base = stack.gpu;
      state->stack_stride = size;
   }
   *base = state->stack_base;
   *stride = state->stack_stride;
   return VK_SUCCESS;
}

void
panvk_per_arch(cmd_bind_rt_pipeline)(struct vk_command_buffer *vk_cmdbuf,
                                     struct panvk_rt_pipeline *pipeline)
{
   struct panvk_cmd_buffer *cmdbuf =
      container_of(vk_cmdbuf, struct panvk_cmd_buffer, vk);
   cmdbuf->state.ray_tracing.pipeline = pipeline;
}

static void
copy_launch(struct cs_builder *b, uint64_t dest, uint64_t src)
{
   cs_move64_to(b, cs_scratch_reg64(b, 0), src);
   cs_move64_to(b, cs_scratch_reg64(b, 2), dest);
   for (unsigned offset = 0; offset < sizeof(VkTraceRaysIndirectCommand2KHR);
        offset += 12 * sizeof(uint32_t)) {
      unsigned count = MIN2(12, (sizeof(VkTraceRaysIndirectCommand2KHR) - offset) / 4);
      struct cs_index words = cs_scratch_reg_tuple(b, 4, count);
      cs_load_to(b, words, cs_scratch_reg64(b, 0), BITFIELD_MASK(count), offset);
      cs_store(b, words, cs_scratch_reg64(b, 2), BITFIELD_MASK(count), offset);
   }
   cs_flush_stores(b);
}

static uint64_t
prepare_indirect_groups(struct panvk_cmd_buffer *cmdbuf,
                        uint64_t launch, uint64_t dimensions, bool copy_dimensions)
{
   struct pan_ptr groups = panvk_cmd_alloc_dev_mem(cmdbuf, desc, 12, 4);
   if (!groups.gpu)
      return 0;
   memset(groups.cpu, 0, 12);
   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, PANVK_SUBQUEUE_COMPUTE);
   struct cs_index dims = cs_scratch_reg_tuple(b, 4, 3);
   cs_move64_to(b, cs_scratch_reg64(b, 0), dimensions);
   cs_load_to(b, dims, cs_scratch_reg64(b, 0), BITFIELD_MASK(3), 0);
   if (copy_dimensions) {
      cs_move64_to(b, cs_scratch_reg64(b, 2), launch);
      cs_store(b, dims, cs_scratch_reg64(b, 2), BITFIELD_MASK(3), 88);
   }
   cs_add_imm32(b, cs_scratch_reg32(b, 4), cs_scratch_reg32(b, 4), 15);
   cs_rshift_imm_u32(b, cs_scratch_reg32(b, 4), cs_scratch_reg32(b, 4), 4);
   cs_move64_to(b, cs_scratch_reg64(b, 2), groups.gpu);
   cs_store(b, dims, cs_scratch_reg64(b, 2), BITFIELD_MASK(3), 0);
   cs_flush_stores(b);
   return groups.gpu;
}

static void
trace_rays(struct panvk_cmd_buffer *cmdbuf,
           const VkStridedDeviceAddressRegionKHR *raygen,
           const VkStridedDeviceAddressRegionKHR *miss,
           const VkStridedDeviceAddressRegionKHR *hit,
           const VkStridedDeviceAddressRegionKHR *callable,
           uint32_t width, uint32_t height, uint32_t depth,
           uint64_t indirect, bool indirect2)
{
   const struct panvk_rt_pipeline *pipeline = cmdbuf->state.ray_tracing.pipeline;
   if (!pipeline || !pipeline->shader) {
      vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_UNKNOWN);
      return;
   }
   if (!indirect && (!width || !height || !depth))
      return;

   struct panvk_rt_dispatch_record record = {0};
   if (!indirect2) {
      record.launch = (VkTraceRaysIndirectCommand2KHR){
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
         .width = width,
         .height = height,
         .depth = depth,
      };
   }

   uint64_t stack_size = pipeline->compiled_stack_size;
   if (cmdbuf->state.ray_tracing.stack_size_set)
      stack_size = MAX2(stack_size, cmdbuf->state.ray_tracing.stack_size);
   VkResult result = get_stack(cmdbuf, stack_size, &record.stack_base,
                               &record.stack_stride);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmdbuf->vk, result);
      return;
   }
   struct pan_ptr push = panvk_cmd_upload_dev_mem(
      cmdbuf, desc, cmdbuf->state.ray_tracing.push_constants,
      sizeof(cmdbuf->state.ray_tracing.push_constants), 8);
   if (!push.gpu)
      return;
   record.push_constants = push.gpu;
   struct pan_ptr ids = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, pipeline->stage_count * sizeof(uint32_t), 4);
   if (!ids.gpu)
      return;
   for (uint32_t i = 0; i < pipeline->stage_count; i++)
      ((uint32_t *)ids.cpu)[i] = pipeline->stages[i].id;
   record.stage_ids = ids.gpu;
   struct pan_ptr launch = panvk_cmd_upload_dev_mem(
      cmdbuf, desc, &record, sizeof(record), 8);
   if (!launch.gpu)
      return;

   struct panvk_dispatch_info info = {
      .direct.wg_count = {DIV_ROUND_UP(width, 16), height, depth},
      .barrier = PANVK_CSF_BARRIER_SYNC,
   };
   if (indirect) {
      struct cs_builder *b = panvk_get_cs_builder(cmdbuf, PANVK_SUBQUEUE_COMPUTE);
      if (indirect2)
         copy_launch(b, launch.gpu, indirect);
      info.indirect.buffer_dev_addr = prepare_indirect_groups(
         cmdbuf, launch.gpu, indirect2 ? launch.gpu + 88 : indirect, !indirect2);
      if (!info.indirect.buffer_dev_addr)
         return;
   }
   panvk_per_arch(cmd_dispatch_rt)(cmdbuf, pipeline->shader, &info, launch.gpu);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdTraceRaysKHR)(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   uint32_t width, uint32_t height, uint32_t depth)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   trace_rays(cmdbuf, pRaygenShaderBindingTable, pMissShaderBindingTable,
              pHitShaderBindingTable, pCallableShaderBindingTable,
              width, height, depth, 0, false);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdTraceRaysIndirectKHR)(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   VkDeviceAddress indirectDeviceAddress)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   trace_rays(cmdbuf, pRaygenShaderBindingTable, pMissShaderBindingTable,
              pHitShaderBindingTable, pCallableShaderBindingTable,
              0, 0, 0, indirectDeviceAddress, false);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdTraceRaysIndirect2KHR)(VkCommandBuffer commandBuffer,
                                        VkDeviceAddress indirectDeviceAddress)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   trace_rays(cmdbuf, NULL, NULL, NULL, NULL, 0, 0, 0,
              indirectDeviceAddress, true);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdSetRayTracingPipelineStackSizeKHR)(
   VkCommandBuffer commandBuffer, uint32_t pipelineStackSize)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   cmdbuf->state.ray_tracing.stack_size = pipelineStackSize;
   cmdbuf->state.ray_tracing.stack_size_set = true;
}
