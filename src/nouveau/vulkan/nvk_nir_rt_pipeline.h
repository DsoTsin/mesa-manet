#ifndef NVK_NIR_RT_PIPELINE_H
#define NVK_NIR_RT_PIPELINE_H

#include "nir.h"
#include <vulkan/vulkan.h>

#define NVK_RT_GROUP_HANDLE_SIZE 32
#define NVK_RT_WORKGROUP_SIZE 32

struct nvk_nir_rt_pipeline_stage {
   nir_shader *nir;
   mesa_shader_stage stage;
   uint32_t id;
};

struct nir_builder;

struct nvk_nir_rt_ser_options {
   void (*emit_suspend)(struct nir_builder *b, nir_def *queue,
                        nir_def *owner, void *data);
   void (*emit_complete)(struct nir_builder *b, nir_def *queue,
                         nir_def *owner, void *data);
   void *data;
};

struct nvk_nir_rt_pipeline_info {
   const struct nvk_nir_rt_pipeline_stage *stages;
   uint32_t stage_count;
   const nir_shader_compiler_options *options;
   uint32_t max_recursion_depth;
   uint32_t max_hit_attribute_size;
   VkPipelineCreateFlags2KHR flags;
   bool ray_query_opacity_micromap;
   bool force_global_stack;
   const struct nvk_nir_rt_ser_options *ser;
};

struct nvk_nir_rt_pipeline_result {
   nir_shader *nir;
   uint32_t frame_size;
   uint32_t stack_size;
   uint32_t storage_size;
   uint32_t *per_stage_stack_size;
   bool uses_ser_queue;
};

VkResult nvk_nir_rt_pipeline_create(
   void *mem_ctx, const struct nvk_nir_rt_pipeline_info *info,
   struct nvk_nir_rt_pipeline_result *result);

#endif
