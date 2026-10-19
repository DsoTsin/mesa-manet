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
#include "util/u_debug.h"

struct rt_native_entry {
   uint32_t id;
   uint32_t stage;
   uint32_t address;
   uint32_t frame_size;
};

static int
compare_native_entry(const void *a, const void *b)
{
   const struct rt_native_entry *x = a, *y = b;
   return (x->id > y->id) - (x->id < y->id);
}

static VkResult
get_stack(struct panvk_cmd_buffer *cmdbuf, uint64_t size, uint64_t *base)
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
   if (!stack_size || stack_size > UINT32_MAX - 127) {
      vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   record.params.stack_stride = ALIGN_POT(stack_size, 128);
   bool native_tls = pipeline->native_frame_stride && pipeline->native_tls;
   uint64_t external_size = native_tls ? 0 : record.params.stack_stride;
   uint32_t tls_size = 0;
   record.params.stack_tls_offset = UINT64_MAX;
   record.params.external_state_scale = 1;
   if (pipeline->native_frame_stride) {
      external_size += pipeline->external_frame_size;
      const struct panvk_physical_device *pdev =
         to_panvk_physical_device(cmdbuf->vk.base.device->physical);
      const struct pan_kmod_dev_props *props = &pdev->kmod.dev->props;
      record.params.external_state_scale = (uint64_t)pan_query_core_id_range(props) *
                                    props->max_threads_per_core;
   }
   if (native_tls) {
      if (record.params.stack_stride > UINT32_MAX - pipeline->native_tls_offset) {
         vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
      tls_size = pipeline->native_tls_offset + record.params.stack_stride;
      const struct panvk_physical_device *pdev =
         to_panvk_physical_device(cmdbuf->vk.base.device->physical);
      const struct pan_kmod_dev_props *props = &pdev->kmod.dev->props;
      if (!panvk_rt_tls_allocation_fits(tls_size,
             pan_query_thread_tls_alloc(props), pan_query_core_id_range(props))) {
         vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
      record.params.stack_tls_offset = pipeline->native_tls_offset;
   }
   VkResult result = get_stack(cmdbuf, external_size, &record.params.stack_base);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmdbuf->vk, result);
      return;
   }
   struct pan_ptr push = panvk_cmd_upload_dev_mem(
      cmdbuf, desc, cmdbuf->state.ray_tracing.push_constants,
      sizeof(cmdbuf->state.ray_tracing.push_constants), 8);
   if (!push.gpu)
      return;
   record.params.push_constants = push.gpu;
   if (!pipeline->native_direct_sbt) {
      struct pan_ptr ids = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, pipeline->stage_count *
            (pipeline->native_frame_stride ? sizeof(struct rt_native_entry) : sizeof(uint32_t)), 8);
      if (!ids.gpu)
         return;
      if (pipeline->native_frame_stride) {
         uint64_t base = panvk_shader_variant_get_dev_addr(&pipeline->shader->variants[0]);
         struct rt_native_entry *entries = ids.cpu;
         for (uint32_t i = 0; i < pipeline->stage_count; i++) {
            entries[i] = (struct rt_native_entry){
               .id = pipeline->stages[i].id,
               .stage = pipeline->stages[i].stage,
               .address = base + pipeline->stage_entry_offsets[i],
               .frame_size = pipeline->stage_stack_sizes[i],
            };
         }
         qsort(entries, pipeline->stage_count, sizeof(*entries), compare_native_entry);
      } else {
         for (uint32_t i = 0; i < pipeline->stage_count; i++)
            ((uint32_t *)ids.cpu)[i] = pipeline->stages[i].id;
      }
      record.params.stage_ids = ids.gpu;
   }
   record.params.native_frame_stride = pipeline->native_frame_stride;
   struct pan_ptr launch = panvk_cmd_upload_dev_mem(
      cmdbuf, desc, &record, sizeof(record), 8);
   if (!launch.gpu)
      return;

   struct panvk_dispatch_info info = {
      .direct.wg_count = {DIV_ROUND_UP(width, 16), height, depth},
      .barrier = PANVK_CSF_BARRIER_SYNC,
      .tls_size = tls_size,
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
   panvk_per_arch(cmd_dispatch_rt)(cmdbuf, pipeline->shader, &info, launch.gpu, &record.params);
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
