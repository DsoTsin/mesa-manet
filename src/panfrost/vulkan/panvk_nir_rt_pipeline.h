#ifndef PANVK_NIR_RT_PIPELINE_H
#define PANVK_NIR_RT_PIPELINE_H

#include <vulkan/vulkan.h>
#include "nir.h"

struct nir_builder;

#define PANVK_RT_DISPATCH_PARAMS_OFFSET 104
#define PANVK_RT_DISPATCH_PARAMS_SIZE 56
#define PANVK_RT_NATIVE_SHADER_INDEX_BITS 21
#define PANVK_RT_NATIVE_SHADER_INDEX_MASK ((1u << PANVK_RT_NATIVE_SHADER_INDEX_BITS) - 1)

static inline uint64_t
panvk_rt_native_shader_indices(const uint32_t stages[3], uint32_t stage_count)
{
   if (stage_count > PANVK_RT_NATIVE_SHADER_INDEX_MASK)
      return 0;
   uint64_t indices = 0;
   for (unsigned i = 0; i < 3; i++) {
      if (stages[i] < stage_count)
         indices |= (uint64_t)(stages[i] + 1) << (i * PANVK_RT_NATIVE_SHADER_INDEX_BITS);
   }
   return indices;
}

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
   bool native;
   bool direct_handles;
   bool disable_lto;
   bool tls;
};

struct panvk_nir_rt_pipeline_result {
   nir_shader *nir;
   nir_shader **functions;
   uint32_t frame_size;
   uint32_t stack_size;
   uint32_t ray_query_slots;
   uint32_t external_frame_size;
   uint32_t *per_stage_stack_size;
};

static inline bool
panvk_nir_is_native_rt(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            if (op == nir_intrinsic_rt_call_pan || op == nir_intrinsic_rt_return_pan ||
                op == nir_intrinsic_rt_call_result_pan || op == nir_intrinsic_rt_return_value_pan)
               return true;
         }
      }
   }
   return false;
}

VkResult panvk_nir_rt_pipeline_create(
   void *mem_ctx, const struct panvk_nir_rt_pipeline_info *info,
   struct panvk_nir_rt_pipeline_result *result);

bool panvk_nir_opt_rt_calls(const struct panvk_nir_rt_pipeline_info *info,
                          struct panvk_nir_rt_pipeline_result *result);

bool panvk_nir_opt_rt_lto(const struct panvk_nir_rt_pipeline_info *info,
                         struct panvk_nir_rt_pipeline_result *result);

bool panvk_nir_lower_rt_tls(nir_shader *nir);
bool panvk_nir_opt_rt_private_io(nir_shader *nir);
bool panvk_nir_lower_rt_call_barriers(nir_shader *nir);
bool panvk_nir_lower_rt_launch_id(
   const struct panvk_nir_rt_pipeline_info *info,
   struct panvk_nir_rt_pipeline_result *result);

nir_def *panvk_nir_load_rt_dispatch(
   struct nir_builder *b, nir_def *dispatch, unsigned offset, unsigned comps,
   unsigned bits, unsigned params_sysval_offset);

#endif
