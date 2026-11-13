#include "nvk_rt_pipeline.h"
#include "nvk_nir_ser_queue.h"
#include "nvk_cmd_buffer.h"
#include "nvk_device.h"
#include "nvk_entrypoints.h"
#include "nvk_physical_device.h"
#include "nvk_shader.h"
#include "vk_pipeline_layout.h"
#include "vk_shader.h"
#include "util/u_atomic.h"
#include "util/os_time.h"
#include "compiler/spirv/nir_spirv.h"

static uint32_t next_stage_token;

static bool
nvk_rt_pipeline_slm_fits(const struct nvk_physical_device *pdev,
                         const struct nvk_shader *shader)
{
   if (shader->info.slm_size >= (1u << 24) ||
       shader->info.crs_size > (1u << 20))
      return false;
   struct nvk_slm_layout layout = nvk_slm_layout_calculate(&pdev->info,
      shader->info.slm_size, shader->info.crs_size);
   if (layout.bytes_per_warp > UINT32_MAX || layout.bytes_per_tpc > UINT32_MAX)
      return false;
#ifdef _WIN32
   if (layout.size > UINT32_MAX)
      return false;
#endif
   return true;
}

static void
nvk_rt_pipeline_destroy(struct vk_device *device, struct vk_pipeline *base,
                         const VkAllocationCallbacks *alloc)
{
   struct nvk_rt_pipeline *pipeline = container_of(base, struct nvk_rt_pipeline, vk);
   if (pipeline->shader)
      vk_shader_destroy(device, &pipeline->shader->vk, alloc);
   if (pipeline->layout)
      vk_pipeline_layout_unref(device, pipeline->layout);
   ralloc_free(pipeline->mem_ctx);
   vk_pipeline_free(device, alloc, base);
}

void
nvk_cmd_bind_rt_pipeline(struct nvk_cmd_buffer *cmd, struct nvk_rt_pipeline *pipeline)
{
   cmd->state.rt.pipeline = pipeline;
   if (!pipeline->dynamic_stack_size)
      cmd->state.rt.stack_size = pipeline->stack_size;
}

static void
nvk_rt_pipeline_bind(struct vk_command_buffer *vk_cmd, struct vk_pipeline *base)
{
   struct nvk_cmd_buffer *cmd = container_of(vk_cmd, struct nvk_cmd_buffer, vk);
   struct nvk_rt_pipeline *pipeline = container_of(base, struct nvk_rt_pipeline, vk);
   nvk_cmd_bind_rt_pipeline(cmd, pipeline);
   if (pipeline->layout) {
      nvk_descriptor_state_set_root_array(cmd, &cmd->state.rt.descriptors,
         set_dynamic_buffer_start, 0, pipeline->layout->set_count,
         pipeline->layout->dynamic_descriptor_offset);
   }
}

static struct vk_shader *
nvk_rt_pipeline_get_shader(struct vk_pipeline *base, mesa_shader_stage stage)
{
   struct nvk_rt_pipeline *pipeline = container_of(base, struct nvk_rt_pipeline, vk);
   return pipeline->shader ? &pipeline->shader->vk : NULL;
}

static VkResult
nvk_rt_pipeline_executable_properties(struct vk_device *device,
   struct vk_pipeline *base, uint32_t *count,
   VkPipelineExecutablePropertiesKHR *properties)
{
   struct nvk_rt_pipeline *pipeline = container_of(base, struct nvk_rt_pipeline, vk);
   if (!pipeline->shader) {
      *count = 0;
      return VK_SUCCESS;
   }
   struct vk_shader *shader = &pipeline->shader->vk;
   VkResult result = shader->ops->get_executable_properties(device, shader, count,
                                                           properties);
   if (properties) {
      for (unsigned i = 0; i < *count; i++)
         properties[i].stages = pipeline->vk.stages;
   }
   return result;
}

static VkResult
nvk_rt_pipeline_executable_statistics(struct vk_device *device,
   struct vk_pipeline *base, uint32_t index, uint32_t *count,
   VkPipelineExecutableStatisticKHR *statistics)
{
   struct nvk_rt_pipeline *pipeline = container_of(base, struct nvk_rt_pipeline, vk);
   struct vk_shader *shader = &pipeline->shader->vk;
   return shader->ops->get_executable_statistics(device, shader, index,
                                                 count, statistics);
}

static VkResult
nvk_rt_pipeline_executable_representations(struct vk_device *device,
   struct vk_pipeline *base, uint32_t index, uint32_t *count,
   VkPipelineExecutableInternalRepresentationKHR *representations)
{
   struct nvk_rt_pipeline *pipeline = container_of(base, struct nvk_rt_pipeline, vk);
   struct vk_shader *shader = &pipeline->shader->vk;
   return shader->ops->get_executable_internal_representations(device, shader,
      index, count, representations);
}

static const struct vk_pipeline_ops nvk_rt_pipeline_ops = {
   .destroy = nvk_rt_pipeline_destroy,
   .get_executable_properties = nvk_rt_pipeline_executable_properties,
   .get_executable_statistics = nvk_rt_pipeline_executable_statistics,
   .get_internal_representations = nvk_rt_pipeline_executable_representations,
   .cmd_bind = nvk_rt_pipeline_bind,
   .get_shader = nvk_rt_pipeline_get_shader,
};

static VkResult
nvk_rt_pipeline_create(struct nvk_device *dev,
                        const VkRayTracingPipelineCreateInfoKHR *info,
                        const VkAllocationCallbacks *alloc, VkPipeline *out)
{
   const uint64_t start_time = os_time_get_nano();
   const struct vk_device_shader_ops *shader_ops = dev->vk.shader_ops;
   VkPipelineCreateFlags2KHR flags = vk_rt_pipeline_create_flags(info);
   if (flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR)
      return VK_PIPELINE_COMPILE_REQUIRED;
   struct nvk_rt_pipeline *pipeline = vk_pipeline_zalloc(&dev->vk,
      &nvk_rt_pipeline_ops, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
      flags, alloc, sizeof(*pipeline));
   if (!pipeline)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   pipeline->mem_ctx = ralloc_context(NULL);
   if (!pipeline->mem_ctx) {
      nvk_rt_pipeline_destroy(&dev->vk, &pipeline->vk, alloc);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   pipeline->stage_count = info->stageCount;
   pipeline->group_count = info->groupCount;
   const VkPipelineLibraryCreateInfoKHR *libraries = info->pLibraryInfo;
   if (libraries) {
      for (unsigned i = 0; i < libraries->libraryCount; i++) {
         VK_FROM_HANDLE(vk_pipeline, base, libraries->pLibraries[i]);
         struct nvk_rt_pipeline *lib = container_of(base, struct nvk_rt_pipeline, vk);
         pipeline->stage_count += lib->stage_count;
         pipeline->group_count += lib->group_count;
      }
   }
   pipeline->stages = rzalloc_array(pipeline->mem_ctx,
      struct nvk_nir_rt_pipeline_stage, pipeline->stage_count);
   pipeline->groups = rzalloc_array(pipeline->mem_ctx,
      struct nvk_rt_group, pipeline->group_count);
   if ((pipeline->stage_count && !pipeline->stages) ||
       (pipeline->group_count && !pipeline->groups)) {
      nvk_rt_pipeline_destroy(&dev->vk, &pipeline->vk, alloc);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   VK_FROM_HANDLE(vk_pipeline_layout, layout, info->layout);
   if (layout)
      pipeline->layout = vk_pipeline_layout_ref(layout);
   struct vk_pipeline_robustness_state robustness;
   vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &robustness,
                                     info->pNext, NULL);
   const nir_shader_compiler_options *options = shader_ops->get_nir_options(
      dev->vk.physical, MESA_SHADER_COMPUTE, &robustness);
   VkResult result = VK_SUCCESS;
   for (unsigned i = 0; i < info->stageCount; i++) {
      mesa_shader_stage stage = vk_to_mesa_shader_stage(info->pStages[i].stage);
      struct spirv_to_nir_options spirv = shader_ops->get_spirv_options(
         dev->vk.physical, stage, &robustness);
      nir_shader *nir;
      result = vk_pipeline_shader_stage_to_nir(&dev->vk, flags, &info->pStages[i],
         &spirv, options, pipeline->mem_ctx, &nir);
      if (result != VK_SUCCESS)
         goto fail;
      pipeline->stages[i] = (struct nvk_nir_rt_pipeline_stage) {
         .nir = nir, .stage = stage,
         .id = p_atomic_inc_return(&next_stage_token),
      };
      pipeline->vk.stages |= info->pStages[i].stage;
   }
   for (unsigned g = 0; g < info->groupCount; g++) {
      const VkRayTracingShaderGroupCreateInfoKHR *group = &info->pGroups[g];
      struct nvk_rt_group *dst = &pipeline->groups[g];
      dst->stages[0] = group->type == VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR ?
         group->generalShader : group->intersectionShader;
      dst->stages[1] = group->anyHitShader;
      dst->stages[2] = group->closestHitShader;
      for (unsigned s = 0; s < 3; s++) {
         if (dst->stages[s] != VK_SHADER_UNUSED_KHR) {
            uint32_t token = pipeline->stages[dst->stages[s]].id;
            memcpy(dst->handle + s * 8, &token, sizeof(token));
         }
      }
   }
   uint32_t stage_offset = info->stageCount;
   uint32_t group_offset = info->groupCount;
   if (libraries) {
      for (unsigned l = 0; l < libraries->libraryCount; l++) {
         VK_FROM_HANDLE(vk_pipeline, base, libraries->pLibraries[l]);
         struct nvk_rt_pipeline *lib = container_of(base, struct nvk_rt_pipeline, vk);
         for (unsigned i = 0; i < lib->stage_count; i++) {
            pipeline->stages[stage_offset + i] = lib->stages[i];
            pipeline->stages[stage_offset + i].nir =
               nir_shader_clone(pipeline->mem_ctx, lib->stages[i].nir);
         }
         for (unsigned g = 0; g < lib->group_count; g++) {
            struct nvk_rt_group *dst = &pipeline->groups[group_offset + g];
            *dst = lib->groups[g];
            for (unsigned s = 0; s < 3; s++) {
               if (dst->stages[s] != VK_SHADER_UNUSED_KHR)
                  dst->stages[s] += stage_offset;
            }
         }
         pipeline->vk.stages |= lib->vk.stages;
         stage_offset += lib->stage_count;
         group_offset += lib->group_count;
      }
   }
   if (info->pDynamicState) {
      for (unsigned i = 0; i < info->pDynamicState->dynamicStateCount; i++)
         pipeline->dynamic_stack_size |= info->pDynamicState->pDynamicStates[i] ==
            VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR;
   }
   struct nvk_nir_rt_pipeline_info link = {
      .ray_query_opacity_micromap = dev->vk.enabled_features.micromapEXT,
      .stages = pipeline->stages,
      .stage_count = pipeline->stage_count,
      .options = options,
      .max_recursion_depth = info->maxPipelineRayRecursionDepth,
      .max_hit_attribute_size = info->pLibraryInterface ?
         info->pLibraryInterface->maxPipelineRayHitAttributeSize : 32,
      .flags = flags,
      .ser = &nvk_nir_ser_queue_options,
   };
   struct nvk_nir_rt_pipeline_result linked;
   for (;;) {
      result = nvk_nir_rt_pipeline_create(pipeline->mem_ctx, &link, &linked);
      if (result != VK_SUCCESS)
         goto fail;
      if (flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)
         break;
      if (shader_ops->preprocess_nir)
         shader_ops->preprocess_nir(dev->vk.physical, linked.nir, &robustness);
      VkShaderCreateFlagsEXT shader_flags = 0;
      if (flags & VK_PIPELINE_CREATE_2_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR)
         shader_flags |= VK_SHADER_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_MESA;
      result = nvk_compile_nir_shader_with_layout(dev, linked.nir, shader_flags,
         &robustness, layout, alloc, &pipeline->shader);
      if (result != VK_SUCCESS)
         goto fail;
      if (nvk_rt_pipeline_slm_fits(nvk_device_physical(dev), pipeline->shader))
         break;
      if (linked.storage_size) {
         result = vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         goto fail;
      }
      vk_shader_destroy(&dev->vk, &pipeline->shader->vk, alloc);
      pipeline->shader = NULL;
      ralloc_free(linked.per_stage_stack_size);
      link.force_global_stack = true;
   }
   pipeline->frame_size = linked.frame_size;
   pipeline->stack_size = linked.stack_size;
   pipeline->storage_size = linked.storage_size;
   pipeline->uses_ser_queue = linked.uses_ser_queue;
   pipeline->per_stage_stack_size = linked.per_stage_stack_size;
   VkPipelineCreationFeedbackCreateInfo *feedback = (void *)vk_find_struct_const(
      info->pNext, PIPELINE_CREATION_FEEDBACK_CREATE_INFO);
   if (feedback) {
      if (feedback->pPipelineCreationFeedback) {
         *feedback->pPipelineCreationFeedback = (VkPipelineCreationFeedback) {
            .flags = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT,
            .duration = os_time_get_nano() - start_time,
         };
      }
      for (unsigned i = 0; i < feedback->pipelineStageCreationFeedbackCount; i++)
         feedback->pPipelineStageCreationFeedbacks[i] = (VkPipelineCreationFeedback){0};
   }
   *out = vk_pipeline_to_handle(&pipeline->vk);
   return VK_SUCCESS;

fail:
   nvk_rt_pipeline_destroy(&dev->vk, &pipeline->vk, alloc);
   return result;
}

VKAPI_ATTR uint32_t VKAPI_CALL
nvk_GetDeferredOperationMaxConcurrencyKHR(UNUSED VkDevice device,
                                        UNUSED VkDeferredOperationKHR operation)
{
   return 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_CreateRayTracingPipelinesKHR(VkDevice device, VkDeferredOperationKHR deferred,
                                VkPipelineCache cache, uint32_t count,
                                const VkRayTracingPipelineCreateInfoKHR *infos,
                                const VkAllocationCallbacks *alloc,
                                VkPipeline *pipelines)
{
   VK_FROM_HANDLE(nvk_device, dev, device);
   VkResult result = VK_SUCCESS;
   memset(pipelines, 0, count * sizeof(*pipelines));
   for (unsigned i = 0; i < count; i++) {
      VkResult current = nvk_rt_pipeline_create(dev, &infos[i], alloc, &pipelines[i]);
      if (current != VK_SUCCESS) {
         result = current;
         if (vk_rt_pipeline_create_flags(&infos[i]) &
             VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR)
            break;
      }
   }
   if (result == VK_SUCCESS && deferred != VK_NULL_HANDLE)
      return VK_OPERATION_DEFERRED_KHR;
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_GetRayTracingShaderGroupHandlesKHR(VkDevice device, VkPipeline handle,
                                      uint32_t first, uint32_t count,
                                      size_t size, void *data)
{
   VK_FROM_HANDLE(vk_pipeline, base, handle);
   struct nvk_rt_pipeline *pipeline = container_of(base, struct nvk_rt_pipeline, vk);
   for (unsigned i = 0; i < count; i++)
      memcpy((uint8_t *)data + i * NVK_RT_GROUP_HANDLE_SIZE,
         pipeline->groups[first + i].handle, NVK_RT_GROUP_HANDLE_SIZE);
   return VK_SUCCESS;
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL
nvk_GetRayTracingShaderGroupStackSizeKHR(VkDevice device, VkPipeline handle,
                                        uint32_t group,
                                        VkShaderGroupShaderKHR which)
{
   VK_FROM_HANDLE(vk_pipeline, base, handle);
   struct nvk_rt_pipeline *pipeline = container_of(base, struct nvk_rt_pipeline, vk);
   unsigned slot = which == VK_SHADER_GROUP_SHADER_ANY_HIT_KHR ? 1 :
                   which == VK_SHADER_GROUP_SHADER_CLOSEST_HIT_KHR ? 2 : 0;
   uint32_t stage = pipeline->groups[group].stages[slot];
   return stage == VK_SHADER_UNUSED_KHR ? 0 : pipeline->per_stage_stack_size[stage];
}
