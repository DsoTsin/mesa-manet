#ifndef PANVK_NIR_RT_PIPELINE_H
#define PANVK_NIR_RT_PIPELINE_H

#include <vulkan/vulkan.h>
#include "nir.h"

struct panvk_nir_rt_pipeline_stage {
   nir_shader *nir;
   mesa_shader_stage stage;
   uint32_t id;
};

struct panvk_nir_rt_pipeline_info {
   const struct panvk_nir_rt_pipeline_stage *stages;
   uint32_t stage_count;
   const nir_shader_compiler_options *options;
   uint32_t dispatch_sysval_offset;
   uint32_t max_recursion_depth;
   uint32_t max_hit_attribute_size;
   VkPipelineCreateFlags2KHR flags;
};

struct panvk_nir_rt_pipeline_result {
   nir_shader *nir;
   uint32_t frame_size;
   uint32_t stack_size;
   uint32_t *per_stage_stack_size;
};

VkResult panvk_nir_rt_pipeline_create(
   void *mem_ctx, const struct panvk_nir_rt_pipeline_info *info,
   struct panvk_nir_rt_pipeline_result *result);

#endif
