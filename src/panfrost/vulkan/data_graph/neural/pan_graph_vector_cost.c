#include "pan_graph_vector_cost.h"

#include <limits.h>
#include <math.h>

bool
pan_graph_vector_cost_estimate_program_add_i32(
   const struct pan_graph_memory_cost_target *target,
   const struct pan_graph_memory_cost_program *program,
   struct pan_graph_vector_cost_result *result)
{
   if (!target || !program || !result || program->workload.rank != 3 ||
       !program->sections || !program->input_views ||
       program->section_count > INT32_MAX ||
       program->input_view_count > INT32_MAX)
      return false;
   uint32_t reads = 0, vectors = 0, writes = 0;
   for (uint32_t i = 0; i < program->section_count; i++) {
      const struct pan_graph_memory_cost_section *section =
         &program->sections[i];
      switch (section->type) {
      case PAN_GRAPH_V15_MEMORY_INPUT:
         if (vectors || writes || section->bits != 32)
            return false;
         reads++;
         break;
      case PAN_GRAPH_V15_VE:
         if (!reads || writes)
            return false;
         vectors++;
         break;
      case PAN_GRAPH_V15_MEMORY_OUTPUT:
         if (!vectors || section->bits != 32)
            return false;
         writes++;
         break;
      default:
         return false;
      }
   }
   if (!reads || !vectors || !writes || reads > program->input_view_count ||
       writes != program->output_view_count ||
       reads + vectors != program->resource_count)
      return false;
   for (uint32_t i = 0; i < program->input_view_count; i++) {
      if (program->input_views[i] != PAN_GRAPH_MEMORY_COST_TENSOR_VIEW)
         return false;
   }
   struct pan_graph_vector_cost_result value = {0};
   if (!pan_graph_memory_cost_estimate(target, program, value.memory))
      return false;
   const struct pan_graph_cost_workload *workload = &program->workload;
   uint64_t n = (workload->transfer[2] + 1) / 2 * 2;
   uint64_t m = (workload->transfer[1] + 3) / 4 * 4;
   uint64_t volume = n * m * workload->transfer[0];
   volatile float sum = (float)(int64_t)volume + 8.0f;
   volatile float adjusted = sum - 1.0f;
   volatile float base = adjusted / 8.0f;
   uint64_t cycles = (uint64_t)base;
   for (unsigned i = 0; i < 3; i++) {
      uint64_t extent = workload->extent[i];
      for (unsigned j = 0; j < 2; j++) {
         if (workload->step[j] && workload->axis[j] == i &&
             extent > workload->step[j])
            extent = workload->step[j];
      }
      volatile float ratio = (float)extent / (float)workload->transfer[i];
      volatile float product = (float)(int64_t)cycles * ratio;
      float rounded = ceilf(product);
      cycles = rounded >= 0x1p63f ? INT64_MAX : (uint64_t)rounded;
   }
   if (cycles > INT64_MAX / vectors)
      return false;
   cycles *= vectors;
   value.engine[0].engine_cycles[3] = cycles;
   value.engine[0].critical_cycles = cycles;
   value.engine[0].critical_kind = 3;
   value.engine[1].critical_kind = 3;
   if (!pan_graph_cost_combine(value.memory, value.engine, &value.total) ||
       !pan_graph_cost_apply_workload(workload, value.total.cycles,
                                      &value.total.cycles))
      return false;
   *result = value;
   return true;
}

bool
pan_graph_vector_cost_estimate_add_i32(
   const struct pan_graph_memory_cost_target *target,
   const struct pan_graph_memory_cost_program *program,
   struct pan_graph_vector_cost_result *result)
{
   if (!program || program->section_count != 3 ||
       program->input_view_count != 2 || program->output_view_count != 1 ||
       program->resource_count != 2)
      return false;
   return pan_graph_vector_cost_estimate_program_add_i32(target, program,
                                                         result);
}
