#include "panvk_rt_pipeline.h"
#include "panvk_nir_rt_pipeline.h"
#include "nir_builder.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
   glsl_type_singleton_init_or_ref();
   const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MISS, &options, NULL);
   assert(panvk_per_arch(rt_stage_register)(b.shader, 23) == VK_SUCCESS);
   struct panvk_nir_rt_pipeline_stage stage = {
      .nir = b.shader, .stage = MESA_SHADER_MISS, .id = 23,
   };
   uint32_t stack_size = 512;
   struct panvk_rt_pipeline pipeline = {
      .stages = &stage, .stage_count = 1, .stage_stack_sizes = &stack_size,
   };
   VkRayTracingShaderGroupCreateInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
      .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
      .generalShader = 0,
      .closestHitShader = VK_SHADER_UNUSED_KHR,
      .anyHitShader = VK_SHADER_UNUSED_KHR,
      .intersectionShader = VK_SHADER_UNUSED_KHR,
   };
   struct panvk_rt_pipeline_group group, replayed;
   assert(panvk_per_arch(rt_pipeline_group_init)(&pipeline, &group, &info) == VK_SUCCESS);
   assert(sizeof(group.capture_replay) == 64);
   assert(group.capture_replay.handle.general_or_intersection == 23);
   info.pShaderGroupCaptureReplayHandle = &group.capture_replay;
   assert(panvk_per_arch(rt_pipeline_group_init)(&pipeline, &replayed, &info) == VK_SUCCESS);
   assert(!memcmp(&group.capture_replay, &replayed.capture_replay, sizeof(group.capture_replay)));
   group.capture_replay.hash[0] ^= 1;
   assert(panvk_per_arch(rt_pipeline_group_init)(&pipeline, &replayed, &info) == VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
   group.capture_replay.hash[0] ^= 1;
   group.capture_replay.handle.general_or_intersection++;
   assert(panvk_per_arch(rt_pipeline_group_init)(&pipeline, &replayed, &info) == VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
   nir_builder other = nir_builder_init_simple_shader(MESA_SHADER_CALLABLE, &options, NULL);
   assert(panvk_per_arch(rt_stage_register)(other.shader, 23) == VK_ERROR_UNKNOWN);
   ralloc_free(other.shader);
   ralloc_free(b.shader);
   glsl_type_singleton_decref();
   puts("capture replay identity and rejection cases passed");
   return 0;
}
