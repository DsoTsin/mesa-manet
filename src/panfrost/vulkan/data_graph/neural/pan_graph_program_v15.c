#include "pan_graph_program_v15.h"
#include "pan_graph_vector_v15.h"

#include <string.h>

static bool
same_memory(const struct pan_graph_v15_memory *a,
            const struct pan_graph_v15_memory *b)
{
   return a->domain == b->domain && a->domain_bias == b->domain_bias &&
          a->offset == b->offset && a->level == b->level &&
          a->override == b->override && a->flags == b->flags &&
          a->value == b->value && a->extra == b->extra;
}

static uint32_t
local_value(const struct pan_graph_program_cost_plan *plan, uint32_t value,
            uint32_t vectors, bool ce)
{
   if (!ce) {
      for (uint32_t i = 0; i < plan->read_count; i++) {
         if (plan->read_tensors[i] == value)
            return i;
      }
   }
   for (uint32_t i = 0; i < vectors; i++) {
      if (plan->vector_groups[i] - 1 == value)
         return i + (ce ? 0 : plan->read_count);
   }
   return UINT32_MAX;
}

enum pan_graph_tsu_status
pan_graph_program_v15_compile_candidate(
   const struct pan_graph_program_cost_plan *plan,
   const struct pan_graph_program_selection *selection,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const struct pan_graph_program_bindings *bindings,
   const struct pan_graph_nrt_target *target,
   struct pan_graph_program_kernel *kernel)
{
   if (!plan || !selection || !bindings || !target || !kernel ||
       (add_count && !adds) || !plan->read_tensors || !plan->vector_groups ||
       !plan->boundary.outputs || !bindings->reads || !bindings->writes ||
       bindings->read_count != plan->read_count ||
       bindings->write_count != plan->boundary.output_count)
      return PAN_GRAPH_TSU_INVALID;
   if (!plan->read_count || plan->read_count > 7 || !plan->vector_count ||
       plan->vector_count > 7 || !bindings->write_count ||
       bindings->write_count > 7)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   bool ce = plan->program.workload.rank == 7;
   if ((!ce && plan->program.workload.rank != 3) ||
       (ce && (plan->read_count != 2 || plan->vector_groups[0] != 1)))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   for (uint32_t i = ce; i < plan->vector_count; i++) {
      uint32_t group = plan->vector_groups[i];
      if (group < 2 || group - 2 >= add_count ||
          (i && group <= plan->vector_groups[i - 1]))
         return PAN_GRAPH_TSU_INVALID;
   }
   struct pan_graph_program_kernel next = {0};
   if (ce) {
      if (add_count > UINT32_MAX - 3 ||
          plan->read_tensors[0] != add_count + 3 ||
          plan->read_tensors[1] != add_count + 2)
         return PAN_GRAPH_TSU_INVALID;
      uint32_t sink = plan->vector_groups[plan->vector_count - 1] - 1;
      for (uint32_t i = 0; i < bindings->write_count; i++) {
         if (plan->boundary.outputs[i].tensor != sink ||
             !same_memory(bindings->writes, bindings->writes + i))
            return PAN_GRAPH_TSU_UNSUPPORTED;
      }
      struct pan_graph_matmul_add epilogue_adds[6];
      for (uint32_t i = 1; i < plan->vector_count; i++) {
         uint32_t group = plan->vector_groups[i];
         for (unsigned j = 0; j < 2; j++) {
            uint32_t input =
               local_value(plan, adds[group - 2].input[j], i, true);
            if (input == UINT32_MAX)
               return PAN_GRAPH_TSU_INVALID;
            epilogue_adds[i - 1].input[j] = input;
         }
      }
      struct pan_graph_matmul_plan matmul;
      if (!pan_graph_matmul_i8_init(&matmul, &plan->info))
         return PAN_GRAPH_TSU_INVALID;
      const struct pan_graph_matmul_epilogue epilogue = {
         .adds = epilogue_adds,
         .count = plan->vector_count - 1,
         .output_view_count = bindings->write_count,
      };
      const struct pan_graph_matmul_memory memory = {
         .inputs = {bindings->reads[1], bindings->reads[0]},
         .output = bindings->writes[0],
      };
      struct pan_graph_matmul_kernel result;
      enum pan_graph_tsu_status status =
         pan_graph_matmul_v15_compile_bound_program_candidate(
            &matmul, target, selection->transfer, selection->loops, &epilogue,
            &memory, &result);
      if (status != PAN_GRAPH_TSU_SUCCESS)
         return status;
      next.program = result.program;
      memcpy(next.extent, result.extent, sizeof(next.extent));
      memcpy(next.loops, result.loops, sizeof(next.loops));
   } else {
      struct pan_graph_vector_add vector_adds[7];
      struct pan_graph_vector_write writes[7];
      for (uint32_t i = 0; i < plan->vector_count; i++) {
         uint32_t group = plan->vector_groups[i];
         for (unsigned j = 0; j < 2; j++) {
            uint32_t input =
               local_value(plan, adds[group - 2].input[j], i, false);
            if (input == UINT32_MAX)
               return PAN_GRAPH_TSU_INVALID;
            vector_adds[i].input[j] = input;
         }
      }
      for (uint32_t i = 0; i < bindings->write_count; i++) {
         uint32_t value = local_value(plan, plan->boundary.outputs[i].tensor,
                                      plan->vector_count, false);
         if (value == UINT32_MAX)
            return PAN_GRAPH_TSU_INVALID;
         writes[i] = (struct pan_graph_vector_write){
            .value = value,
            .memory = bindings->writes[i],
         };
      }
      struct pan_graph_vector_program program = {
         .full = {plan->info.batch, plan->info.rows, plan->info.columns},
         .reads = bindings->reads,
         .adds = vector_adds,
         .writes = writes,
         .read_count = plan->read_count,
         .add_count = plan->vector_count,
         .write_count = bindings->write_count,
      };
      memcpy(program.transfer, selection->transfer, sizeof(program.transfer));
      memcpy(program.loops, selection->loops, sizeof(program.loops));
      struct pan_graph_vector_kernel result;
      enum pan_graph_tsu_status status =
         pan_graph_vector_v15_compile_candidate(&program, target, &result);
      if (status != PAN_GRAPH_TSU_SUCCESS)
         return status;
      next.program = result.program;
      memcpy(next.extent, result.extent, sizeof(next.extent));
      memcpy(next.loops, result.loops, sizeof(next.loops));
   }
   *kernel = next;
   return PAN_GRAPH_TSU_SUCCESS;
}
