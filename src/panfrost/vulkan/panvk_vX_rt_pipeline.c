#include "panvk_rt_pipeline.h"

#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_nir_rt_pipeline.h"
#include "panvk_shader.h"

#include "spirv/nir_spirv.h"
#include "nir_serialize.h"
#include "util/blob.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"
#include "vk_alloc.h"
#include "vk_log.h"

#include <limits.h>
#include <string.h>

static const struct vk_pipeline_ops rt_pipeline_ops;
static uint32_t next_stage_id;
static simple_mtx_t stage_ids_mutex = SIMPLE_MTX_INITIALIZER;
static struct rt_stage_identity {
   struct rt_stage_identity *next;
   blake3_hash hash;
   uint32_t id;
   uint32_t refs;
} *stage_ids;

VkResult
panvk_per_arch(rt_stage_register)(nir_shader *nir, uint32_t id)
{
   if (!id || id > INT32_MAX)
      return VK_ERROR_UNKNOWN;
   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, nir, false);
   if (blob.out_of_memory) {
      blob_finish(&blob);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   blake3_hash hash;
   _mesa_blake3_compute(blob.data, blob.size, hash);
   blob_finish(&blob);

   simple_mtx_lock(&stage_ids_mutex);
   for (struct rt_stage_identity *entry = stage_ids; entry; entry = entry->next) {
      if (entry->id != id)
         continue;
      VkResult result = VK_SUCCESS;
      if (memcmp(entry->hash, hash, sizeof(hash)) || entry->refs == UINT32_MAX)
         result = VK_ERROR_UNKNOWN;
      else
         entry->refs++;
      simple_mtx_unlock(&stage_ids_mutex);
      return result;
   }
   struct rt_stage_identity *entry = malloc(sizeof(*entry));
   if (!entry) {
      simple_mtx_unlock(&stage_ids_mutex);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   *entry = (struct rt_stage_identity){.next = stage_ids, .id = id, .refs = 1};
   memcpy(entry->hash, hash, sizeof(hash));
   stage_ids = entry;
   uint32_t previous = p_atomic_read(&next_stage_id);
   while (previous < id) {
      uint32_t current = p_atomic_cmpxchg(&next_stage_id, previous, id);
      if (current == previous)
         break;
      previous = current;
   }
   simple_mtx_unlock(&stage_ids_mutex);
   return VK_SUCCESS;
}

static void
rt_stage_unregister(uint32_t id)
{
   simple_mtx_lock(&stage_ids_mutex);
   struct rt_stage_identity **link = &stage_ids;
   while (*link && (*link)->id != id)
      link = &(*link)->next;
   assert(*link && (*link)->refs);
   struct rt_stage_identity *entry = *link;
   if (!--entry->refs) {
      *link = entry->next;
      free(entry);
   }
   simple_mtx_unlock(&stage_ids_mutex);
}

bool
panvk_per_arch(is_rt_pipeline)(const struct vk_pipeline *pipeline)
{
   return pipeline && pipeline->ops == &rt_pipeline_ops;
}

static uint32_t
rt_stage_id_alloc(void)
{
   uint32_t previous = p_atomic_read(&next_stage_id);
   while (previous < INT32_MAX) {
      uint32_t current =
         p_atomic_cmpxchg(&next_stage_id, previous, previous + 1);
      if (current == previous)
         return previous + 1;
      previous = current;
   }
   return 0;
}

VkResult
panvk_per_arch(rt_stage_import)(nir_shader *nir, uint32_t *id)
{
   VkResult result = panvk_per_arch(rt_stage_register)(nir, *id);
   while (result == VK_ERROR_UNKNOWN) {
      *id = rt_stage_id_alloc();
      if (!*id)
         return VK_ERROR_TOO_MANY_OBJECTS;
      result = panvk_per_arch(rt_stage_register)(nir, *id);
   }
   return result;
}

static void
rt_pipeline_destroy(struct vk_device *device, struct vk_pipeline *vk_pipeline,
                    const VkAllocationCallbacks *alloc)
{
   struct panvk_rt_pipeline *pipeline =
      container_of(vk_pipeline, struct panvk_rt_pipeline, vk);

   if (pipeline->shader)
      vk_shader_destroy(device, &pipeline->shader->vk, alloc);
   for (uint32_t i = 0; i < pipeline->registered_stage_count; i++)
      rt_stage_unregister(pipeline->stages[i].id);
   for (uint32_t i = 0; i < pipeline->stage_count; i++)
      ralloc_free(pipeline->stages[i].nir);
   vk_free2(&device->alloc, alloc, pipeline->stage_stack_sizes);
   vk_free2(&device->alloc, alloc, pipeline->stages);
   vk_free2(&device->alloc, alloc, pipeline->groups);
   if (pipeline->layout)
      vk_pipeline_layout_unref(device, pipeline->layout);
   vk_pipeline_free(device, alloc, vk_pipeline);
}

static VkResult
rt_pipeline_properties(struct vk_device *device, struct vk_pipeline *vk_pipeline,
                       uint32_t *count,
                       VkPipelineExecutablePropertiesKHR *properties)
{
   struct panvk_rt_pipeline *pipeline =
      container_of(vk_pipeline, struct panvk_rt_pipeline, vk);
   if (!pipeline->shader) {
      *count = 0;
      return VK_SUCCESS;
   }

   VkResult result = pipeline->shader->vk.ops->get_executable_properties(
      device, &pipeline->shader->vk, count, properties);
   if (properties) {
      for (uint32_t i = 0; i < *count; i++)
         properties[i].stages = pipeline->vk.stages;
   }
   return result;
}

static VkResult
rt_pipeline_statistics(struct vk_device *device, struct vk_pipeline *vk_pipeline,
                       uint32_t executable, uint32_t *count,
                       VkPipelineExecutableStatisticKHR *statistics)
{
   struct panvk_rt_pipeline *pipeline =
      container_of(vk_pipeline, struct panvk_rt_pipeline, vk);
   if (!pipeline->shader) {
      *count = 0;
      return VK_SUCCESS;
   }
   return pipeline->shader->vk.ops->get_executable_statistics(
      device, &pipeline->shader->vk, executable, count, statistics);
}

static VkResult
rt_pipeline_representations(
   struct vk_device *device, struct vk_pipeline *vk_pipeline,
   uint32_t executable, uint32_t *count,
   VkPipelineExecutableInternalRepresentationKHR *representations)
{
   struct panvk_rt_pipeline *pipeline =
      container_of(vk_pipeline, struct panvk_rt_pipeline, vk);
   if (!pipeline->shader) {
      *count = 0;
      return VK_SUCCESS;
   }
   return pipeline->shader->vk.ops->get_executable_internal_representations(
      device, &pipeline->shader->vk, executable, count, representations);
}

static void
rt_pipeline_bind(struct vk_command_buffer *cmd, struct vk_pipeline *vk_pipeline)
{
   struct panvk_rt_pipeline *pipeline =
      container_of(vk_pipeline, struct panvk_rt_pipeline, vk);
   panvk_per_arch(cmd_bind_rt_pipeline)(cmd, pipeline);
}

static struct vk_shader *
rt_pipeline_get_shader(struct vk_pipeline *vk_pipeline, mesa_shader_stage stage)
{
   struct panvk_rt_pipeline *pipeline =
      container_of(vk_pipeline, struct panvk_rt_pipeline, vk);
   return stage == MESA_SHADER_COMPUTE && pipeline->shader
             ? &pipeline->shader->vk
             : NULL;
}

static const struct vk_pipeline_ops rt_pipeline_ops = {
   .destroy = rt_pipeline_destroy,
   .get_executable_properties = rt_pipeline_properties,
   .get_executable_statistics = rt_pipeline_statistics,
   .get_internal_representations = rt_pipeline_representations,
   .cmd_bind = rt_pipeline_bind,
   .get_shader = rt_pipeline_get_shader,
};

static void
rt_robustness_merge(struct vk_pipeline_robustness_state *dst,
                    const struct vk_pipeline_robustness_state *src)
{
   dst->storage_buffers = MAX2(dst->storage_buffers, src->storage_buffers);
   dst->uniform_buffers = MAX2(dst->uniform_buffers, src->uniform_buffers);
   dst->vertex_inputs = MAX2(dst->vertex_inputs, src->vertex_inputs);
   dst->images = MAX2(dst->images, src->images);
   dst->null_uniform_buffer_descriptor |= src->null_uniform_buffer_descriptor;
   dst->null_storage_buffer_descriptor |= src->null_storage_buffer_descriptor;
}

static bool
rt_stage_is_valid(mesa_shader_stage stage)
{
   return stage == MESA_SHADER_RAYGEN || stage == MESA_SHADER_MISS ||
          stage == MESA_SHADER_CALLABLE || stage == MESA_SHADER_CLOSEST_HIT ||
          stage == MESA_SHADER_ANY_HIT || stage == MESA_SHADER_INTERSECTION;
}

static bool
rt_group_stage_matches(const struct panvk_rt_pipeline *pipeline, uint32_t index,
                       mesa_shader_stage stage)
{
   return index == VK_SHADER_UNUSED_KHR ||
          (index < pipeline->stage_count &&
           pipeline->stages[index].stage == stage);
}

VkResult
panvk_per_arch(rt_pipeline_group_init)(struct panvk_rt_pipeline *pipeline,
              struct panvk_rt_pipeline_group *group,
              const VkRayTracingShaderGroupCreateInfoKHR *info)
{
   group->info = *info;
   group->info.pNext = NULL;
   group->info.pShaderGroupCaptureReplayHandle = NULL;

   if (info->pShaderGroupCaptureReplayHandle)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   if (info->type == VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR) {
      if (info->closestHitShader != VK_SHADER_UNUSED_KHR ||
          info->anyHitShader != VK_SHADER_UNUSED_KHR ||
          info->intersectionShader != VK_SHADER_UNUSED_KHR)
         return VK_ERROR_UNKNOWN;
      uint64_t id = 0;
      uint32_t stack = 0;
      if (info->generalShader != VK_SHADER_UNUSED_KHR) {
         if (info->generalShader >= pipeline->stage_count)
            return VK_ERROR_UNKNOWN;
         const struct panvk_nir_rt_pipeline_stage *stage =
            &pipeline->stages[info->generalShader];
         if (stage->stage != MESA_SHADER_RAYGEN &&
             stage->stage != MESA_SHADER_MISS &&
             stage->stage != MESA_SHADER_CALLABLE)
            return VK_ERROR_UNKNOWN;
         id = stage->id;
         stack = pipeline->stage_stack_sizes[info->generalShader];
      }
      panvk_rt_shader_group_init_general(&group->handle, &group->stack_sizes,
                                         id, 0, stack);
      return VK_SUCCESS;
   }

   if (info->type != VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR &&
       info->type != VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR)
      return VK_ERROR_UNKNOWN;
   if (info->generalShader != VK_SHADER_UNUSED_KHR ||
       !rt_group_stage_matches(pipeline, info->closestHitShader,
                               MESA_SHADER_CLOSEST_HIT) ||
       !rt_group_stage_matches(pipeline, info->anyHitShader,
                               MESA_SHADER_ANY_HIT) ||
       !rt_group_stage_matches(pipeline, info->intersectionShader,
                               MESA_SHADER_INTERSECTION))
      return VK_ERROR_UNKNOWN;
   if (info->type == VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR &&
       info->intersectionShader != VK_SHADER_UNUSED_KHR)
      return VK_ERROR_UNKNOWN;

   uint32_t indices[3] = {info->intersectionShader, info->anyHitShader,
                          info->closestHitShader};
   uint64_t ids[3] = {0};
   uint32_t stacks[3] = {0};
   for (unsigned i = 0; i < 3; i++) {
      if (indices[i] != VK_SHADER_UNUSED_KHR) {
         ids[i] = pipeline->stages[indices[i]].id;
         stacks[i] = pipeline->stage_stack_sizes[indices[i]];
      }
   }
   panvk_rt_shader_group_init_hit(&group->handle, &group->stack_sizes,
                                 ids[0], stacks[0], ids[1], stacks[1],
                                 ids[2], stacks[2]);
   return VK_SUCCESS;
}

static uint32_t
rt_rebase_stage(uint32_t index, uint32_t base)
{
   return index == VK_SHADER_UNUSED_KHR ? index : index + base;
}

VkDeviceSize
panvk_per_arch(rt_pipeline_default_stack_size)(const struct panvk_rt_pipeline *pipeline)
{
   VkDeviceSize raygen = 0, closest = 0, miss = 0, intersection = 0;
   VkDeviceSize any_hit = 0, callable = 0;
   for (uint32_t i = 0; i < pipeline->group_count; i++) {
      const struct panvk_rt_pipeline_group *group = &pipeline->groups[i];
      if (group->info.type == VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR &&
          group->info.generalShader != VK_SHADER_UNUSED_KHR) {
         switch (pipeline->stages[group->info.generalShader].stage) {
         case MESA_SHADER_RAYGEN: raygen = MAX2(raygen, group->stack_sizes.general); break;
         case MESA_SHADER_MISS: miss = MAX2(miss, group->stack_sizes.general); break;
         case MESA_SHADER_CALLABLE: callable = MAX2(callable, group->stack_sizes.general); break;
         default: UNREACHABLE("Invalid general RT stage");
         }
      }
      closest = MAX2(closest, group->stack_sizes.closest_hit);
      any_hit = MAX2(any_hit, group->stack_sizes.any_hit);
      intersection = MAX2(intersection, group->stack_sizes.intersection);
   }
   uint32_t depth = pipeline->max_recursion_depth;
   return raygen + MIN2(1, depth) * MAX3(closest, miss, intersection + any_hit) +
          (depth ? depth - 1 : 0) * MAX2(closest, miss) + 2 * callable;
}

static bool
rt_stack_is_dynamic(const VkRayTracingPipelineCreateInfoKHR *info)
{
   if (!info->pDynamicState)
      return false;
   for (uint32_t i = 0; i < info->pDynamicState->dynamicStateCount; i++) {
      if (info->pDynamicState->pDynamicStates[i] ==
          VK_DYNAMIC_STATE_RAY_TRACING_PIPELINE_STACK_SIZE_KHR)
         return true;
   }
   return false;
}

static VkResult
rt_pipeline_create(struct panvk_device *dev,
                   const VkRayTracingPipelineCreateInfoKHR *info,
                   const VkAllocationCallbacks *alloc, VkPipeline *handle)
{
   VkPipelineCreateFlags2KHR flags = vk_rt_pipeline_create_flags(info);
   if (flags & (VK_PIPELINE_CREATE_2_RAY_TRACING_SHADER_GROUP_HANDLE_CAPTURE_REPLAY_BIT_KHR |
                VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT))
      return VK_ERROR_FEATURE_NOT_PRESENT;

   const VkPipelineBinaryInfoKHR *binary_info =
      vk_find_struct_const(info->pNext, PIPELINE_BINARY_INFO_KHR);
   if ((!binary_info || !binary_info->binaryCount) &&
       (flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR))
      return VK_PIPELINE_COMPILE_REQUIRED;

   uint64_t stage_count = info->stageCount;
   uint64_t group_count = info->groupCount;
   if (info->pLibraryInfo) {
      for (uint32_t i = 0; i < info->pLibraryInfo->libraryCount; i++) {
         VK_FROM_HANDLE(panvk_rt_pipeline, library,
                         info->pLibraryInfo->pLibraries[i]);
         if (!library || !panvk_per_arch(is_rt_pipeline)(&library->vk) ||
             !(library->vk.flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR))
            return VK_ERROR_UNKNOWN;
         stage_count += library->stage_count;
         group_count += library->group_count;
      }
   }
   if (stage_count > UINT32_MAX || group_count > UINT32_MAX ||
       stage_count > SIZE_MAX / sizeof(struct panvk_nir_rt_pipeline_stage) ||
       group_count > SIZE_MAX / sizeof(struct panvk_rt_pipeline_group))
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   struct panvk_rt_pipeline *pipeline = vk_pipeline_zalloc(
      &dev->vk, &rt_pipeline_ops, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
      flags, alloc, sizeof(*pipeline));
   if (!pipeline)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   VK_FROM_HANDLE(vk_pipeline_layout, layout, info->layout);
   if (layout)
      pipeline->layout = vk_pipeline_layout_ref(layout);
   pipeline->max_recursion_depth = info->maxPipelineRayRecursionDepth;
   pipeline->max_hit_attribute_size =
      dev->vk.physical->properties.maxRayHitAttributeSize;
   if (!pipeline->max_hit_attribute_size)
      pipeline->max_hit_attribute_size = 32;
   pipeline->stack_size_dynamic = rt_stack_is_dynamic(info);
   vk_pipeline_robustness_state_fill(&dev->vk.robustness_state,
                                    &pipeline->robustness, info->pNext, NULL);

   if (info->pLibraryInterface) {
      pipeline->max_payload_size = info->pLibraryInterface->maxPipelineRayPayloadSize;
      pipeline->max_hit_attribute_size =
         info->pLibraryInterface->maxPipelineRayHitAttributeSize;
   }

   VkResult result;
   if (binary_info && binary_info->binaryCount) {
      result = panvk_per_arch(rt_pipeline_import)(dev, pipeline, info,
                                                  binary_info, alloc);
      if (result != VK_SUCCESS)
         goto fail;
      *handle = panvk_rt_pipeline_to_handle(pipeline);
      return VK_SUCCESS;
   }
   result = panvk_per_arch(rt_pipeline_key)(dev, info, pipeline->key);
   if (result != VK_SUCCESS)
      goto fail;

   pipeline->stages = vk_zalloc2(
      &dev->vk.alloc, alloc, stage_count * sizeof(*pipeline->stages), 8,
      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   pipeline->groups = vk_zalloc2(
      &dev->vk.alloc, alloc, group_count * sizeof(*pipeline->groups), 8,
      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   pipeline->stage_stack_sizes = vk_zalloc2(
      &dev->vk.alloc, alloc, stage_count * sizeof(*pipeline->stage_stack_sizes), 8,
      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   result = VK_SUCCESS;
   if ((stage_count && (!pipeline->stages || !pipeline->stage_stack_sizes)) ||
       (group_count && !pipeline->groups)) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail;
   }

   const struct vk_device_shader_ops *ops = dev->vk.shader_ops;
   const VkPipelineCreationFeedbackCreateInfo *feedback_info =
      vk_find_struct_const(info->pNext, PIPELINE_CREATION_FEEDBACK_CREATE_INFO);
   for (uint32_t i = 0; i < info->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *stage_info = &info->pStages[i];
      int64_t start = os_time_get_nano();
      mesa_shader_stage stage = vk_to_mesa_shader_stage(stage_info->stage);
      if (!rt_stage_is_valid(stage)) {
         result = VK_ERROR_UNKNOWN;
         goto fail;
      }
      if (vk_pipeline_shader_stage_has_identifier(stage_info) &&
          vk_pipeline_shader_stage_is_null(stage_info)) {
         result = VK_PIPELINE_COMPILE_REQUIRED;
         goto fail;
      }
      struct vk_pipeline_robustness_state robustness;
      vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &robustness,
                                       info->pNext, stage_info->pNext);
      rt_robustness_merge(&pipeline->robustness, &robustness);
      struct spirv_to_nir_options spirv_options =
         ops->get_spirv_options(dev->vk.physical, stage, &robustness);
      const nir_shader_compiler_options *nir_options =
         ops->get_nir_options(dev->vk.physical, MESA_SHADER_COMPUTE, &robustness);
      nir_shader *nir = NULL;
      result = vk_pipeline_shader_stage_to_nir(
         &dev->vk, flags, stage_info, &spirv_options, nir_options, NULL, &nir);
      if (result != VK_SUCCESS)
         goto fail;
      uint32_t id = rt_stage_id_alloc();
      if (!id) {
         ralloc_free(nir);
         result = VK_ERROR_TOO_MANY_OBJECTS;
         goto fail;
      }
      pipeline->stages[pipeline->stage_count++] =
         (struct panvk_nir_rt_pipeline_stage){
            .nir = nir, .stage = stage, .id = id,
         };
      result = panvk_per_arch(rt_stage_import)(
         nir, &pipeline->stages[pipeline->stage_count - 1].id);
      if (result != VK_SUCCESS)
         goto fail;
      pipeline->registered_stage_count++;
      pipeline->vk.stages |= stage_info->stage;
      if (feedback_info && i < feedback_info->pipelineStageCreationFeedbackCount)
         feedback_info->pPipelineStageCreationFeedbacks[i] =
            (VkPipelineCreationFeedback){
               .flags = VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT,
               .duration = os_time_get_nano() - start,
            };
   }

   pipeline->group_count = info->groupCount;
   for (uint32_t i = 0; i < info->groupCount; i++) {
      result = panvk_per_arch(rt_pipeline_group_init)(
         pipeline, &pipeline->groups[i], &info->pGroups[i]);
      if (result != VK_SUCCESS)
         goto fail;
   }

   if (info->pLibraryInfo) {
      for (uint32_t l = 0; l < info->pLibraryInfo->libraryCount; l++) {
         VK_FROM_HANDLE(panvk_rt_pipeline, library,
                         info->pLibraryInfo->pLibraries[l]);
         uint32_t base = pipeline->stage_count;
         for (uint32_t i = 0; i < library->stage_count; i++) {
            pipeline->stages[pipeline->stage_count] = library->stages[i];
            pipeline->stages[pipeline->stage_count].nir =
               nir_shader_clone(NULL, library->stages[i].nir);
            if (!pipeline->stages[pipeline->stage_count].nir) {
               result = VK_ERROR_OUT_OF_HOST_MEMORY;
               goto fail;
            }
            pipeline->stage_stack_sizes[pipeline->stage_count++] =
               library->stage_stack_sizes[i];
            result = panvk_per_arch(rt_stage_register)(
               pipeline->stages[pipeline->stage_count - 1].nir,
               pipeline->stages[pipeline->stage_count - 1].id);
            if (result != VK_SUCCESS)
               goto fail;
            pipeline->registered_stage_count++;
         }
         for (uint32_t i = 0; i < library->group_count; i++) {
            struct panvk_rt_pipeline_group *group =
               &pipeline->groups[pipeline->group_count++];
            *group = library->groups[i];
            group->info.generalShader =
               rt_rebase_stage(group->info.generalShader, base);
            group->info.intersectionShader =
               rt_rebase_stage(group->info.intersectionShader, base);
            group->info.anyHitShader =
               rt_rebase_stage(group->info.anyHitShader, base);
            group->info.closestHitShader =
               rt_rebase_stage(group->info.closestHitShader, base);
         }
         pipeline->vk.stages |= library->vk.stages;
         rt_robustness_merge(&pipeline->robustness, &library->robustness);
      }
   }

   if (!pipeline->stage_count) {
      if (!(flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)) {
         result = VK_ERROR_UNKNOWN;
         goto fail;
      }
      *handle = panvk_rt_pipeline_to_handle(pipeline);
      return VK_SUCCESS;
   }

   void *link_ctx = ralloc_context(NULL);
   if (!link_ctx) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail;
   }
   struct panvk_nir_rt_pipeline_result linked = {0};
   struct panvk_nir_rt_pipeline_info link_info = {
      .stages = pipeline->stages,
      .stage_count = pipeline->stage_count,
      .options = ops->get_nir_options(dev->vk.physical, MESA_SHADER_COMPUTE,
                                     &pipeline->robustness),
      .dispatch_sysval_offset =
         SYSVALS_PUSH_CONST_BASE + sysval_offset(compute, rt_dispatch),
      .max_recursion_depth = pipeline->max_recursion_depth,
      .max_hit_attribute_size = pipeline->max_hit_attribute_size,
      .flags = flags,
   };
   result = panvk_nir_rt_pipeline_create(link_ctx, &link_info, &linked);
   if (result != VK_SUCCESS) {
      ralloc_free(link_ctx);
      goto fail;
   }
   pipeline->frame_size = linked.frame_size;
   pipeline->compiled_stack_size = linked.stack_size;
   memcpy(pipeline->stage_stack_sizes, linked.per_stage_stack_size,
          pipeline->stage_count * sizeof(*pipeline->stage_stack_sizes));
   for (uint32_t i = 0; i < pipeline->group_count; i++) {
      VkRayTracingShaderGroupCreateInfoKHR group_info = pipeline->groups[i].info;
      result = panvk_per_arch(rt_pipeline_group_init)(
         pipeline, &pipeline->groups[i], &group_info);
      if (result != VK_SUCCESS) {
         ralloc_free(link_ctx);
         goto fail;
      }
   }
   pipeline->stack_size = pipeline->stack_size_dynamic ? 0 :
      panvk_per_arch(rt_pipeline_default_stack_size)(pipeline);

   if (!(flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)) {
      nir_shader *nir = linked.nir;
      ralloc_steal(NULL, nir);
      result = panvk_per_arch(rt_compile_nir)(
         dev, nir, pipeline->layout, &pipeline->robustness, flags, alloc,
         &pipeline->shader);
   }
   ralloc_free(link_ctx);
   if (result != VK_SUCCESS)
      goto fail;

   *handle = panvk_rt_pipeline_to_handle(pipeline);
   return VK_SUCCESS;

fail:
   rt_pipeline_destroy(&dev->vk, &pipeline->vk, alloc);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CreateRayTracingPipelinesKHR)(
   VkDevice device, VkDeferredOperationKHR deferredOperation,
   VkPipelineCache pipelineCache, uint32_t createInfoCount,
   const VkRayTracingPipelineCreateInfoKHR *pCreateInfos,
   const VkAllocationCallbacks *pAllocator, VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(panvk_device, dev, device);
   (void)pipelineCache;
   memset(pPipelines, 0, createInfoCount * sizeof(*pPipelines));
   VkResult first_result = VK_SUCCESS;
   for (uint32_t i = 0; i < createInfoCount; i++) {
      int64_t start = os_time_get_nano();
      const VkPipelineCreationFeedbackCreateInfo *feedback_info =
         vk_find_struct_const(pCreateInfos[i].pNext,
                              PIPELINE_CREATION_FEEDBACK_CREATE_INFO);
      if (feedback_info && feedback_info->pipelineStageCreationFeedbackCount)
         memset(feedback_info->pPipelineStageCreationFeedbacks, 0,
                feedback_info->pipelineStageCreationFeedbackCount *
                   sizeof(VkPipelineCreationFeedback));
      VkResult result =
         rt_pipeline_create(dev, &pCreateInfos[i], pAllocator, &pPipelines[i]);
      if (feedback_info && feedback_info->pPipelineCreationFeedback) {
         *feedback_info->pPipelineCreationFeedback =
            (VkPipelineCreationFeedback){
               .flags = result == VK_SUCCESS
                           ? VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT
                           : 0,
               .duration = os_time_get_nano() - start,
            };
      }
      if (result == VK_SUCCESS)
         continue;
      if (first_result == VK_SUCCESS || first_result == VK_PIPELINE_COMPILE_REQUIRED)
         first_result = result;
      if (vk_rt_pipeline_create_flags(&pCreateInfos[i]) &
          VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR)
         break;
   }
   return first_result == VK_SUCCESS && deferredOperation != VK_NULL_HANDLE
             ? VK_OPERATION_NOT_DEFERRED_KHR
             : first_result;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(GetRayTracingShaderGroupHandlesKHR)(
   VkDevice device, VkPipeline pipelineHandle, uint32_t firstGroup,
   uint32_t groupCount, size_t dataSize, void *pData)
{
   (void)device;
   VK_FROM_HANDLE(panvk_rt_pipeline, pipeline, pipelineHandle);
   if (!pipeline || firstGroup > pipeline->group_count ||
       groupCount > pipeline->group_count - firstGroup ||
       dataSize < (uint64_t)groupCount * PANVK_RT_SHADER_GROUP_HANDLE_SIZE)
      return VK_ERROR_UNKNOWN;
   for (uint32_t i = 0; i < groupCount; i++) {
      memcpy((uint8_t *)pData + (size_t)i * PANVK_RT_SHADER_GROUP_HANDLE_SIZE,
             &pipeline->groups[firstGroup + i].handle,
             PANVK_RT_SHADER_GROUP_HANDLE_SIZE);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL
panvk_per_arch(GetRayTracingShaderGroupStackSizeKHR)(
   VkDevice device, VkPipeline pipelineHandle, uint32_t group,
   VkShaderGroupShaderKHR groupShader)
{
   (void)device;
   VK_FROM_HANDLE(panvk_rt_pipeline, pipeline, pipelineHandle);
   if (!pipeline || group >= pipeline->group_count)
      return 0;
   return panvk_rt_shader_group_stack_size(&pipeline->groups[group].stack_sizes,
                                          groupShader);
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(GetRayTracingCaptureReplayShaderGroupHandlesKHR)(
   VkDevice device, VkPipeline pipelineHandle, uint32_t firstGroup,
   uint32_t groupCount, size_t dataSize, void *pData)
{
   (void)device;
   (void)pipelineHandle;
   (void)firstGroup;
   (void)groupCount;
   (void)dataSize;
   (void)pData;
   return VK_ERROR_FEATURE_NOT_PRESENT;
}
