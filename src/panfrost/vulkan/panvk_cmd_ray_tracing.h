#ifndef PANVK_CMD_RAY_TRACING_H
#define PANVK_CMD_RAY_TRACING_H

#include "panvk_cmd_desc_state.h"
#include "panvk_cmd_push_constant.h"
#include "util/u_math.h"

#define PANVK_RT_STAGE_FLAGS (VK_SHADER_STAGE_RAYGEN_BIT_KHR | \
   VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | \
   VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR | \
   VK_SHADER_STAGE_CALLABLE_BIT_KHR)

struct panvk_rt_pipeline;
struct panvk_device;
struct panvk_cmd_buffer;
struct panvk_dispatch_info;

static inline bool
panvk_rt_tls_allocation_fits(uint32_t size, uint32_t threads, uint32_t cores)
{
   if (!size)
      return true;
   uint64_t lanes = (uint64_t)threads * cores;
   uint64_t aligned = util_next_power_of_two64(MAX2(size, 16));
   return lanes && aligned <= UINT32_MAX / lanes;
}

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
   struct panvk_rt_dispatch_params params;
};

_Static_assert(sizeof(VkTraceRaysIndirectCommand2KHR) == 104,
               "indirect trace rays size");
_Static_assert(offsetof(struct panvk_rt_dispatch_record, params.stack_base) == 104,
               "ray tracing stack base offset");
_Static_assert(offsetof(struct panvk_rt_dispatch_record, params.stack_stride) == 112,
               "ray tracing stack stride offset");
_Static_assert(offsetof(struct panvk_rt_dispatch_record, params.push_constants) == 120,
               "ray tracing push constants offset");
_Static_assert(offsetof(struct panvk_rt_dispatch_record, params.stage_ids) == 128,
               "ray tracing stage IDs offset");
_Static_assert(offsetof(struct panvk_rt_dispatch_record, params.native_frame_stride) == 136,
               "ray tracing native frame stride offset");
_Static_assert(offsetof(struct panvk_rt_dispatch_record, params.stack_tls_offset) == 144,
               "ray tracing TLS stack offset");
_Static_assert(offsetof(struct panvk_rt_dispatch_record, params.external_state_scale) == 152,
               "ray tracing external state scale offset");
_Static_assert(sizeof(struct panvk_rt_dispatch_record) == 160,
               "ray tracing dispatch size");

void panvk_per_arch(cmd_dispatch_rt)(
   struct panvk_cmd_buffer *cmdbuf, const struct panvk_shader *shader,
   const struct panvk_dispatch_info *info, uint64_t dispatch_addr,
   const struct panvk_rt_dispatch_params *params);

#endif
