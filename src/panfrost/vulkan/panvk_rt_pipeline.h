#ifndef PANVK_RT_PIPELINE_H
#define PANVK_RT_PIPELINE_H

#include "nir.h"
#include "panvk_macros.h"
#include "panvk_rt_shader_group.h"
#include "util/mesa-blake3.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"

struct panvk_device;
struct panvk_shader;
struct panvk_nir_rt_pipeline_stage;

struct panvk_rt_pipeline_group {
   VkRayTracingShaderGroupCreateInfoKHR info;
   struct panvk_rt_shader_group_handle handle;
   struct panvk_rt_shader_group_stack_sizes stack_sizes;
};

struct panvk_rt_pipeline {
   struct vk_pipeline vk;
   struct vk_pipeline_layout *layout;
   struct panvk_shader *shader;
   struct panvk_nir_rt_pipeline_stage *stages;
   struct panvk_rt_pipeline_group *groups;
   struct vk_pipeline_robustness_state robustness;
   blake3_hash key;
   uint32_t *stage_stack_sizes;
   uint32_t stage_count;
   uint32_t registered_stage_count;
   uint32_t group_count;
   uint32_t max_recursion_depth;
   uint32_t max_payload_size;
   uint32_t max_hit_attribute_size;
   uint32_t frame_size;
   VkDeviceSize stack_size;
   VkDeviceSize compiled_stack_size;
   bool stack_size_dynamic;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_rt_pipeline, vk.base, VkPipeline,
                               VK_OBJECT_TYPE_PIPELINE);

bool panvk_per_arch(is_rt_pipeline)(const struct vk_pipeline *pipeline);

VkResult panvk_per_arch(rt_stage_register)(nir_shader *nir, uint32_t id);
VkResult panvk_per_arch(rt_stage_import)(nir_shader *nir, uint32_t *id);
VkResult panvk_per_arch(rt_pipeline_group_init)(
   struct panvk_rt_pipeline *pipeline, struct panvk_rt_pipeline_group *group,
   const VkRayTracingShaderGroupCreateInfoKHR *info);
VkDeviceSize panvk_per_arch(rt_pipeline_default_stack_size)(
   const struct panvk_rt_pipeline *pipeline);
VkResult panvk_per_arch(rt_pipeline_key)(
   struct panvk_device *dev, const VkRayTracingPipelineCreateInfoKHR *info,
   blake3_hash key);
VkResult panvk_per_arch(rt_pipeline_import)(
   struct panvk_device *dev, struct panvk_rt_pipeline *pipeline,
   const VkRayTracingPipelineCreateInfoKHR *info,
   const VkPipelineBinaryInfoKHR *binaries,
   const VkAllocationCallbacks *alloc);

VkResult panvk_per_arch(rt_compile_nir)(
   struct panvk_device *dev, nir_shader *nir, struct vk_pipeline_layout *layout,
   const struct vk_pipeline_robustness_state *robustness,
   VkPipelineCreateFlags2KHR flags, const VkAllocationCallbacks *alloc,
   struct panvk_shader **shader_out);

void panvk_per_arch(cmd_bind_rt_pipeline)(struct vk_command_buffer *cmd,
                                         struct panvk_rt_pipeline *pipeline);

#endif
