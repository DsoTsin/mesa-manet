#include "panvk_rt_pipeline.h"

#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_nir_rt_pipeline.h"
#include "panvk_shader.h"

#include "nir_serialize.h"
#include "util/blob.h"
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_descriptor_set_layout.h"

#include <limits.h>
#include <string.h>

#define PANVK_RT_BINARY_MAGIC UINT64_C(0x314254524b564e50)
#define PANVK_RT_BINARY_VERSION 1

struct rt_binary_data {
   void *data;
   size_t size;
   VkPipelineBinaryKeyKHR key;
};

static VkPipelineCreateFlags2KHR
rt_binary_flags(VkPipelineCreateFlags2KHR flags)
{
   return flags & ~(VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR |
                    VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR |
                    VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR |
                    VK_PIPELINE_CREATE_2_DERIVATIVE_BIT_KHR |
                    VK_PIPELINE_CREATE_2_ALLOW_DERIVATIVES_BIT_KHR);
}

static void
rt_layout_key(const struct vk_pipeline_layout *layout, blake3_hash key)
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   uint32_t present = layout != NULL;
   _mesa_blake3_update(&ctx, &present, sizeof(present));
   if (layout) {
      _mesa_blake3_update(&ctx, &layout->create_flags, sizeof(layout->create_flags));
      _mesa_blake3_update(&ctx, &layout->set_count, sizeof(layout->set_count));
      for (uint32_t i = 0; i < layout->set_count; i++) {
         uint32_t set_present = layout->set_layouts[i] != NULL;
         _mesa_blake3_update(&ctx, &set_present, sizeof(set_present));
         if (set_present)
            _mesa_blake3_update(&ctx, layout->set_layouts[i]->blake3,
                               sizeof(blake3_hash));
      }
      _mesa_blake3_update(&ctx, &layout->push_range_count,
                          sizeof(layout->push_range_count));
      for (uint32_t i = 0; i < layout->push_range_count; i++) {
         const VkPushConstantRange *range = &layout->push_ranges[i];
         _mesa_blake3_update(&ctx, &range->stageFlags, sizeof(range->stageFlags));
         _mesa_blake3_update(&ctx, &range->offset, sizeof(range->offset));
         _mesa_blake3_update(&ctx, &range->size, sizeof(range->size));
      }
   }
   _mesa_blake3_final(&ctx, key);
}

static VkResult
rt_binary_read(struct panvk_device *dev, VkPipelineBinaryKHR binary,
               struct rt_binary_data *data)
{
   VkPipelineBinaryDataInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR,
      .pipelineBinary = binary,
   };
   data->key.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
   VkResult result = vk_common_GetPipelineBinaryDataKHR(
      panvk_device_to_handle(dev), &info, &data->key, &data->size, NULL);
   if (result != VK_SUCCESS)
      return result;
   if (data->size < 64 || data->key.keySize != sizeof(blake3_hash))
      return VK_ERROR_UNKNOWN;
   data->data = malloc(data->size);
   if (!data->data)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   result = vk_common_GetPipelineBinaryDataKHR(
      panvk_device_to_handle(dev), &info, &data->key, &data->size, data->data);
   if (result != VK_SUCCESS)
      return result;
   blake3_hash key;
   _mesa_blake3_compute(data->data, data->size, key);
   return memcmp(key, data->key.key, sizeof(key)) ? VK_ERROR_UNKNOWN : VK_SUCCESS;
}

static bool
rt_binary_header(struct panvk_device *dev, struct blob_reader *reader,
                 uint32_t *shader_version)
{
   uint64_t magic = blob_read_uint64(reader);
   uint32_t version = blob_read_uint32(reader);
   uint32_t arch = blob_read_uint32(reader);
   uint8_t uuid[VK_UUID_SIZE];
   blob_copy_bytes(reader, uuid, sizeof(uuid));
   *shader_version = blob_read_uint32(reader);
   return !reader->overrun && magic == PANVK_RT_BINARY_MAGIC &&
          version == PANVK_RT_BINARY_VERSION && arch == PAN_ARCH &&
          *shader_version == dev->vk.physical->properties.shaderBinaryVersion &&
          !memcmp(uuid, dev->vk.physical->properties.shaderBinaryUUID,
                  sizeof(uuid));
}

VkResult
panvk_per_arch(rt_pipeline_key)(struct panvk_device *dev,
                                const VkRayTracingPipelineCreateInfoKHR *info,
                                blake3_hash key)
{
   const VkPipelineBinaryInfoKHR *binaries =
      vk_find_struct_const(info->pNext, PIPELINE_BINARY_INFO_KHR);
   if (binaries && binaries->binaryCount) {
      if (binaries->binaryCount != 1)
         return VK_ERROR_UNKNOWN;
      struct rt_binary_data data = {0};
      VkResult result = rt_binary_read(dev, binaries->pPipelineBinaries[0], &data);
      if (result == VK_SUCCESS) {
         struct blob_reader reader;
         blob_reader_init(&reader, data.data, data.size);
         uint32_t shader_version;
         if (!rt_binary_header(dev, &reader, &shader_version))
            result = VK_ERROR_UNKNOWN;
         else {
            blob_copy_bytes(&reader, key, sizeof(blake3_hash));
            if (reader.overrun)
               result = VK_ERROR_UNKNOWN;
         }
      }
      free(data.data);
      return result;
   }

   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   uint32_t version = PANVK_RT_BINARY_VERSION;
   _mesa_blake3_update(&ctx, &version, sizeof(version));
   _mesa_blake3_update(&ctx, dev->vk.physical->properties.shaderBinaryUUID,
                       VK_UUID_SIZE);
   VkPipelineCreateFlags2KHR flags = rt_binary_flags(vk_rt_pipeline_create_flags(info));
   _mesa_blake3_update(&ctx, &flags, sizeof(flags));
   VK_FROM_HANDLE(vk_pipeline_layout, layout, info->layout);
   blake3_hash layout_key;
   rt_layout_key(layout, layout_key);
   _mesa_blake3_update(&ctx, layout_key, sizeof(layout_key));
   _mesa_blake3_update(&ctx, &info->maxPipelineRayRecursionDepth,
                       sizeof(info->maxPipelineRayRecursionDepth));
   uint32_t interface_sizes[2] = {0,
      dev->vk.physical->properties.maxRayHitAttributeSize ?: 32};
   if (info->pLibraryInterface) {
      interface_sizes[0] = info->pLibraryInterface->maxPipelineRayPayloadSize;
      interface_sizes[1] = info->pLibraryInterface->maxPipelineRayHitAttributeSize;
   }
   _mesa_blake3_update(&ctx, interface_sizes, sizeof(interface_sizes));
   _mesa_blake3_update(&ctx, &info->stageCount, sizeof(info->stageCount));
   for (uint32_t i = 0; i < info->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *stage = &info->pStages[i];
      if (vk_pipeline_shader_stage_is_null(stage) &&
          !vk_pipeline_shader_stage_has_identifier(stage))
         return VK_PIPELINE_COMPILE_REQUIRED;
      struct vk_pipeline_robustness_state robustness;
      vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &robustness,
                                       info->pNext, stage->pNext);
      blake3_hash stage_key;
      vk_pipeline_hash_shader_stage(flags, stage, &robustness, stage_key);
      _mesa_blake3_update(&ctx, stage_key, sizeof(stage_key));
   }
   _mesa_blake3_update(&ctx, &info->groupCount, sizeof(info->groupCount));
   for (uint32_t i = 0; i < info->groupCount; i++) {
      const VkRayTracingShaderGroupCreateInfoKHR *group = &info->pGroups[i];
      uint32_t values[] = {group->type, group->generalShader,
         group->closestHitShader, group->anyHitShader, group->intersectionShader};
      _mesa_blake3_update(&ctx, values, sizeof(values));
   }
   uint32_t library_count = info->pLibraryInfo ? info->pLibraryInfo->libraryCount : 0;
   _mesa_blake3_update(&ctx, &library_count, sizeof(library_count));
   for (uint32_t i = 0; i < library_count; i++) {
      VK_FROM_HANDLE(panvk_rt_pipeline, library, info->pLibraryInfo->pLibraries[i]);
      if (!library || !panvk_per_arch(is_rt_pipeline)(&library->vk))
         return VK_ERROR_UNKNOWN;
      _mesa_blake3_update(&ctx, library->key, sizeof(library->key));
   }
   uint32_t dynamic_count = info->pDynamicState ? info->pDynamicState->dynamicStateCount : 0;
   _mesa_blake3_update(&ctx, &dynamic_count, sizeof(dynamic_count));
   if (dynamic_count)
      _mesa_blake3_update(&ctx, info->pDynamicState->pDynamicStates,
                          dynamic_count * sizeof(VkDynamicState));
   _mesa_blake3_final(&ctx, key);
   return VK_SUCCESS;
}

static VkResult
rt_pipeline_serialize(struct panvk_device *dev,
                      const struct panvk_rt_pipeline *pipeline,
                      struct blob *blob)
{
   blob_write_uint64(blob, PANVK_RT_BINARY_MAGIC);
   blob_write_uint32(blob, PANVK_RT_BINARY_VERSION);
   blob_write_uint32(blob, PAN_ARCH);
   blob_write_bytes(blob, dev->vk.physical->properties.shaderBinaryUUID, VK_UUID_SIZE);
   blob_write_uint32(blob, dev->vk.physical->properties.shaderBinaryVersion);
   blob_write_bytes(blob, pipeline->key, sizeof(pipeline->key));
   blake3_hash layout_key;
   rt_layout_key(pipeline->layout, layout_key);
   blob_write_bytes(blob, layout_key, sizeof(layout_key));
   blob_write_uint64(blob, rt_binary_flags(pipeline->vk.flags));
   blob_write_uint32(blob, pipeline->vk.stages);
   blob_write_uint32(blob, pipeline->stage_count);
   blob_write_uint32(blob, pipeline->group_count);
   blob_write_uint32(blob, pipeline->max_recursion_depth);
   blob_write_uint32(blob, pipeline->max_payload_size);
   blob_write_uint32(blob, pipeline->max_hit_attribute_size);
   blob_write_uint32(blob, pipeline->frame_size);
   blob_write_uint64(blob, pipeline->compiled_stack_size);
   blob_write_uint32(blob, pipeline->stack_size_dynamic);
   blob_write_bytes(blob, &pipeline->robustness, sizeof(pipeline->robustness));
   for (uint32_t i = 0; i < pipeline->stage_count; i++) {
      blob_write_uint32(blob, pipeline->stages[i].stage);
      blob_write_uint32(blob, pipeline->stages[i].id);
      blob_write_uint32(blob, pipeline->stage_stack_sizes[i]);
      struct blob nir;
      blob_init(&nir);
      nir_serialize(&nir, pipeline->stages[i].nir, false);
      if (nir.out_of_memory) {
         blob_finish(&nir);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      blob_write_uint64(blob, nir.size);
      blob_write_bytes(blob, nir.data, nir.size);
      blob_finish(&nir);
   }
   for (uint32_t i = 0; i < pipeline->group_count; i++) {
      const struct panvk_rt_pipeline_group *group = &pipeline->groups[i];
      blob_write_uint32(blob, group->info.type);
      blob_write_uint32(blob, group->info.generalShader);
      blob_write_uint32(blob, group->info.closestHitShader);
      blob_write_uint32(blob, group->info.anyHitShader);
      blob_write_uint32(blob, group->info.intersectionShader);
      blob_write_bytes(blob, &group->handle, sizeof(group->handle));
      blob_write_bytes(blob, &group->stack_sizes, sizeof(group->stack_sizes));
   }
   blob_write_uint32(blob, pipeline->shader != NULL);
   if (pipeline->shader) {
      const struct vk_shader *shader = &pipeline->shader->vk;
      blob_write_uint64(blob, shader->stack_size);
      blob_write_uint64(blob, shader->scratch_size);
      blob_write_uint32(blob, shader->ray_queries);
      struct blob code;
      blob_init(&code);
      bool serialized = shader->ops->serialize(&dev->vk, shader, &code);
      if (!serialized || code.out_of_memory) {
         blob_finish(&code);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      blob_write_uint64(blob, code.size);
      blob_write_bytes(blob, code.data, code.size);
      blob_finish(&code);
   }
   return blob->out_of_memory ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}

static bool
rt_read_chunk(struct blob_reader *reader, struct blob_reader *chunk)
{
   uint64_t size = blob_read_uint64(reader);
   if (reader->overrun || size > (uint64_t)(reader->end - reader->current))
      return false;
   const void *bytes = blob_read_bytes(reader, size);
   blob_reader_init(chunk, bytes, size);
   return !reader->overrun;
}

VkResult
panvk_per_arch(rt_pipeline_import)(
   struct panvk_device *dev, struct panvk_rt_pipeline *pipeline,
   const VkRayTracingPipelineCreateInfoKHR *info,
   const VkPipelineBinaryInfoKHR *binaries, const VkAllocationCallbacks *alloc)
{
   if (binaries->binaryCount != 1)
      return VK_ERROR_UNKNOWN;
   struct rt_binary_data data = {0};
   uint32_t *original_ids = NULL;
   VkResult result = rt_binary_read(dev, binaries->pPipelineBinaries[0], &data);
   if (result != VK_SUCCESS)
      goto done;
   struct blob_reader reader;
   blob_reader_init(&reader, data.data, data.size);
   uint32_t shader_version;
   result = VK_ERROR_UNKNOWN;
   if (!rt_binary_header(dev, &reader, &shader_version))
      goto done;
   blob_copy_bytes(&reader, pipeline->key, sizeof(pipeline->key));
   blake3_hash layout_key, expected_layout_key;
   blob_copy_bytes(&reader, layout_key, sizeof(layout_key));
   rt_layout_key(pipeline->layout, expected_layout_key);
   uint64_t flags = blob_read_uint64(&reader);
   uint32_t stages = blob_read_uint32(&reader);
   uint32_t stage_count = blob_read_uint32(&reader);
   uint32_t group_count = blob_read_uint32(&reader);
   uint32_t max_depth = blob_read_uint32(&reader);
   uint32_t max_payload = blob_read_uint32(&reader);
   uint32_t max_attribute = blob_read_uint32(&reader);
   uint32_t frame_size = blob_read_uint32(&reader);
   uint64_t stack_size = blob_read_uint64(&reader);
   uint32_t dynamic = blob_read_uint32(&reader);
   struct vk_pipeline_robustness_state robustness;
   blob_copy_bytes(&reader, &robustness, sizeof(robustness));
   uint64_t expected_stages = info->stageCount;
   uint64_t expected_groups = info->groupCount;
   uint64_t stage_bytes = (uint64_t)stage_count * sizeof(*pipeline->stages);
   uint64_t group_bytes = (uint64_t)group_count * sizeof(*pipeline->groups);
   if (info->pLibraryInfo) {
      for (uint32_t i = 0; i < info->pLibraryInfo->libraryCount; i++) {
         VK_FROM_HANDLE(panvk_rt_pipeline, library, info->pLibraryInfo->pLibraries[i]);
         expected_stages += library->stage_count;
         expected_groups += library->group_count;
      }
   }
   if (reader.overrun || flags != rt_binary_flags(pipeline->vk.flags) ||
       memcmp(layout_key, expected_layout_key, sizeof(layout_key)) ||
       stage_count != expected_stages || group_count != expected_groups ||
       max_depth != pipeline->max_recursion_depth ||
       max_payload != pipeline->max_payload_size ||
       max_attribute != pipeline->max_hit_attribute_size ||
       dynamic != pipeline->stack_size_dynamic ||
       frame_size % 128 || (stage_count && frame_size < 384) ||
       stack_size > UINT32_MAX ||
       stage_count > (reader.end - reader.current) / 24 ||
       stage_bytes > SIZE_MAX || group_bytes > SIZE_MAX)
      goto done;
   pipeline->robustness = robustness;
   pipeline->vk.stages = stages;
   pipeline->frame_size = frame_size;
   pipeline->compiled_stack_size = stack_size;
   pipeline->stages = vk_zalloc2(&dev->vk.alloc, alloc,
      stage_bytes, 8,
      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   pipeline->stage_stack_sizes = vk_zalloc2(&dev->vk.alloc, alloc,
      (size_t)stage_count * sizeof(*pipeline->stage_stack_sizes), 8,
      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   pipeline->groups = vk_zalloc2(&dev->vk.alloc, alloc,
      group_bytes, 8,
      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   original_ids = malloc((size_t)stage_count * sizeof(*original_ids));
   if ((stage_count && (!pipeline->stages || !pipeline->stage_stack_sizes)) ||
       (stage_count && !original_ids) || (group_count && !pipeline->groups)) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto done;
   }
   const nir_shader_compiler_options *options = dev->vk.shader_ops->get_nir_options(
      dev->vk.physical, MESA_SHADER_COMPUTE, &pipeline->robustness);
   VkShaderStageFlags actual_stages = 0;
   for (uint32_t i = 0; i < stage_count; i++) {
      uint32_t stage = blob_read_uint32(&reader);
      uint32_t id = blob_read_uint32(&reader);
      uint32_t stage_stack = blob_read_uint32(&reader);
      struct blob_reader nir;
      if (stage < MESA_SHADER_RAYGEN || stage > MESA_SHADER_CALLABLE ||
          !id || id > INT32_MAX || !rt_read_chunk(&reader, &nir) ||
          nir.end - nir.current < 8)
         goto done;
      nir_shader *shader = nir_deserialize(NULL, options, &nir);
      pipeline->stages[pipeline->stage_count++] =
         (struct panvk_nir_rt_pipeline_stage){.nir = shader, .stage = stage, .id = id};
      pipeline->stage_stack_sizes[i] = stage_stack;
      original_ids[i] = id;
      if (nir.overrun || nir.current != nir.end || shader->info.stage != stage)
         goto done;
      if (i < info->stageCount) {
         if (info->pStages && vk_to_mesa_shader_stage(info->pStages[i].stage) != stage)
            goto done;
         result = panvk_per_arch(rt_stage_import)(shader, &pipeline->stages[i].id);
      } else {
         uint32_t index = i - info->stageCount;
         for (uint32_t l = 0; l < info->pLibraryInfo->libraryCount; l++) {
            VK_FROM_HANDLE(panvk_rt_pipeline, library,
                            info->pLibraryInfo->pLibraries[l]);
            if (index >= library->stage_count) {
               index -= library->stage_count;
               continue;
            }
            pipeline->stages[i].id = library->stages[index].id;
            result = panvk_per_arch(rt_stage_register)(shader, pipeline->stages[i].id);
            break;
         }
      }
      if (result != VK_SUCCESS)
         goto done;
      pipeline->registered_stage_count++;
      result = VK_ERROR_UNKNOWN;
      actual_stages |= mesa_to_vk_shader_stage(stage);
   }
   if (actual_stages != stages)
      goto done;
   for (uint32_t i = 0; i < group_count; i++) {
      VkRayTracingShaderGroupCreateInfoKHR group_info = {
         .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
         .type = blob_read_uint32(&reader),
         .generalShader = blob_read_uint32(&reader),
         .closestHitShader = blob_read_uint32(&reader),
         .anyHitShader = blob_read_uint32(&reader),
         .intersectionShader = blob_read_uint32(&reader),
      };
      struct panvk_rt_shader_group_handle handle;
      struct panvk_rt_shader_group_stack_sizes stacks;
      blob_copy_bytes(&reader, &handle, sizeof(handle));
      blob_copy_bytes(&reader, &stacks, sizeof(stacks));
      VkRayTracingShaderGroupCreateInfoKHR expected_info;
      if (i < info->groupCount) {
         expected_info = info->pGroups[i];
      } else {
         uint32_t index = i - info->groupCount;
         uint32_t stage_base = info->stageCount;
         bool found = false;
         for (uint32_t l = 0; l < info->pLibraryInfo->libraryCount; l++) {
            VK_FROM_HANDLE(panvk_rt_pipeline, library,
                            info->pLibraryInfo->pLibraries[l]);
            if (index >= library->group_count) {
               index -= library->group_count;
               stage_base += library->stage_count;
               continue;
            }
            expected_info = library->groups[index].info;
            uint32_t *indices[] = {&expected_info.generalShader,
               &expected_info.closestHitShader, &expected_info.anyHitShader,
               &expected_info.intersectionShader};
            for (unsigned s = 0; s < ARRAY_SIZE(indices); s++) {
               if (*indices[s] != VK_SHADER_UNUSED_KHR)
                  *indices[s] += stage_base;
            }
            found = true;
            break;
         }
         if (!found)
            goto done;
      }
      if (group_info.type != expected_info.type ||
          group_info.generalShader != expected_info.generalShader ||
          group_info.closestHitShader != expected_info.closestHitShader ||
          group_info.anyHitShader != expected_info.anyHitShader ||
          group_info.intersectionShader != expected_info.intersectionShader)
         goto done;
      if (reader.overrun || panvk_per_arch(rt_pipeline_group_init)(
             pipeline, &pipeline->groups[i], &group_info) != VK_SUCCESS ||
          memcmp(&stacks, &pipeline->groups[i].stack_sizes, sizeof(stacks)))
         goto done;
      uint32_t primary = group_info.type == VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR
         ? group_info.generalShader : group_info.intersectionShader;
      struct panvk_rt_shader_group_handle expected_handle = {
         .general_or_intersection = primary == VK_SHADER_UNUSED_KHR
            ? 0 : original_ids[primary],
         .any_hit = group_info.anyHitShader == VK_SHADER_UNUSED_KHR
            ? 0 : original_ids[group_info.anyHitShader],
         .closest_hit = group_info.closestHitShader == VK_SHADER_UNUSED_KHR
            ? 0 : original_ids[group_info.closestHitShader],
      };
      if (memcmp(&handle, &expected_handle, sizeof(handle)))
         goto done;
      pipeline->group_count++;
   }
   uint32_t has_shader = blob_read_uint32(&reader);
   if (reader.overrun || has_shader > 1 ||
       has_shader == !!(pipeline->vk.flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR))
      goto done;
   if (has_shader) {
      uint64_t shader_stack = blob_read_uint64(&reader);
      uint64_t scratch = blob_read_uint64(&reader);
      uint32_t queries = blob_read_uint32(&reader);
      struct blob_reader code;
      if (!rt_read_chunk(&reader, &code) || code.current == code.end ||
          code.current[0] != MESA_SHADER_COMPUTE)
         goto done;
      struct vk_shader *shader = NULL;
      result = dev->vk.shader_ops->deserialize(&dev->vk, &code, shader_version,
                                               alloc, &shader);
      if (result != VK_SUCCESS)
         goto done;
      pipeline->shader = container_of(shader, struct panvk_shader, vk);
      shader->stack_size = shader_stack;
      shader->scratch_size = scratch;
      shader->ray_queries = queries;
      result = VK_ERROR_UNKNOWN;
      if (code.overrun || code.current != code.end)
         goto done;
   }
   if (!reader.overrun && reader.current == reader.end)
      result = VK_SUCCESS;
   if (result == VK_SUCCESS)
      pipeline->stack_size = dynamic ? 0 :
         panvk_per_arch(rt_pipeline_default_stack_size)(pipeline);
done:
   free(original_ids);
   free(data.data);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CreatePipelineBinariesKHR)(
   VkDevice device, const VkPipelineBinaryCreateInfoKHR *info,
   const VkAllocationCallbacks *alloc, VkPipelineBinaryHandlesInfoKHR *binaries)
{
   if (info->pPipelineCreateInfo &&
       vk_find_struct_const(info->pPipelineCreateInfo->pNext,
                            RAY_TRACING_PIPELINE_CREATE_INFO_KHR)) {
      binaries->pipelineBinaryCount = 0;
      return VK_PIPELINE_BINARY_MISSING_KHR;
   }
   VK_FROM_HANDLE(vk_pipeline, vk_pipeline, info->pipeline);
   if (!panvk_per_arch(is_rt_pipeline)(vk_pipeline))
      return vk_common_CreatePipelineBinariesKHR(device, info, alloc, binaries);
   VK_FROM_HANDLE(panvk_device, dev, device);
   struct panvk_rt_pipeline *pipeline =
      container_of(vk_pipeline, struct panvk_rt_pipeline, vk);
   if (!binaries->pPipelineBinaries) {
      binaries->pipelineBinaryCount = 1;
      return VK_SUCCESS;
   }
   struct blob blob;
   blob_init(&blob);
   VkResult result = rt_pipeline_serialize(dev, pipeline, &blob);
   if (result == VK_SUCCESS) {
      VkPipelineBinaryKeyKHR key = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR,
         .keySize = sizeof(blake3_hash),
      };
      _mesa_blake3_compute(blob.data, blob.size, key.key);
      VkPipelineBinaryDataKHR data = {.dataSize = blob.size, .pData = blob.data};
      VkPipelineBinaryKeysAndDataKHR keys = {
         .binaryCount = 1, .pPipelineBinaryKeys = &key, .pPipelineBinaryData = &data,
      };
      VkPipelineBinaryCreateInfoKHR create = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR,
         .pKeysAndDataInfo = &keys,
      };
      result = vk_common_CreatePipelineBinariesKHR(device, &create, alloc, binaries);
   }
   blob_finish(&blob);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(GetPipelineKeyKHR)(VkDevice device,
                                const VkPipelineCreateInfoKHR *create,
                                VkPipelineBinaryKeyKHR *key)
{
   const VkRayTracingPipelineCreateInfoKHR *info = create ?
      vk_find_struct_const(create->pNext, RAY_TRACING_PIPELINE_CREATE_INFO_KHR) : NULL;
   if (!info)
      return vk_common_GetPipelineKeyKHR(device, create, key);
   VK_FROM_HANDLE(panvk_device, dev, device);
   VkResult result = panvk_per_arch(rt_pipeline_key)(dev, info, key->key);
   if (result == VK_SUCCESS)
      key->keySize = sizeof(blake3_hash);
   return result;
}
