#ifndef PANVK_CMD_RAY_TRACING_H
#define PANVK_CMD_RAY_TRACING_H

#include "panvk_cmd_desc_state.h"
#include "panvk_cmd_push_constant.h"

#define PANVK_RT_STAGE_FLAGS (VK_SHADER_STAGE_RAYGEN_BIT_KHR | \
   VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | \
   VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR | \
   VK_SHADER_STAGE_CALLABLE_BIT_KHR)

struct panvk_rt_pipeline;
struct panvk_device;
struct panvk_cmd_buffer;
struct panvk_dispatch_info;

struct panvk_cmd_ray_tracing_state {
   const struct panvk_rt_pipeline *pipeline;
   struct panvk_descriptor_state desc_state;
   uint64_t push_constants[6][MAX_PUSH_CONSTANTS_SIZE / sizeof(uint64_t)];
   uint64_t stack_base;
   uint64_t stack_stride;
   uint32_t stack_size;
   bool stack_size_set;
};

struct panvk_rt_dispatch_record {
   VkTraceRaysIndirectCommand2KHR launch;
   uint64_t stack_base;
   uint64_t stack_stride;
   uint64_t push_constants;
   uint64_t stage_ids;
};

_Static_assert(sizeof(VkTraceRaysIndirectCommand2KHR) == 104,
               "indirect trace rays size");
_Static_assert(offsetof(struct panvk_rt_dispatch_record, stack_base) == 104,
               "ray tracing stack base offset");
_Static_assert(offsetof(struct panvk_rt_dispatch_record, stack_stride) == 112,
               "ray tracing stack stride offset");
_Static_assert(offsetof(struct panvk_rt_dispatch_record, push_constants) == 120,
               "ray tracing push constants offset");
_Static_assert(offsetof(struct panvk_rt_dispatch_record, stage_ids) == 128,
               "ray tracing stage IDs offset");
_Static_assert(sizeof(struct panvk_rt_dispatch_record) == 136,
               "ray tracing dispatch size");

void panvk_per_arch(cmd_dispatch_rt)(
   struct panvk_cmd_buffer *cmdbuf, const struct panvk_shader *shader,
   const struct panvk_dispatch_info *info, uint64_t dispatch_addr);

#endif
