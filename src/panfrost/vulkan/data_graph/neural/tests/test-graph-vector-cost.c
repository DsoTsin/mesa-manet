#include "pan_graph_vector_cost.h"

#include <assert.h>
#include <limits.h>
#include <string.h>

int
main(void)
{
   struct pan_graph_memory_cost_target target = {
      .memory =
         {
            [2] = {.bandwidth = {32, 16},
                   .latency = {12, 12},
                   .parallel = 1,
                   .line_size = 64,
                   .size = 32768},
            [3] = {.bandwidth = {32, 32},
                   .latency = {60, 60},
                   .parallel = 8,
                   .line_size = 64,
                   .size = 262144},
            [4] = {.bandwidth = {32, 32},
                   .latency = {100, 100},
                   .parallel = 1,
                   .line_size = 64,
                   .size = 8388608},
            [5] = {.bandwidth = {104.30635833740234f, 104.30635833740234f},
                   .latency = {170, 100},
                   .parallel = 1,
                   .line_size = 64,
                   .size = 268435456},
         },
      .descriptor_bytes = {36, 64, 32, 12},
      .section_bytes = {0, 8, 20, 32, 8, 8, 8, 20},
      .alignment = 32,
      .issue_width = 32,
      .parallel_budget = 6,
   };
   struct pan_graph_memory_cost_section sections[] = {
      {.type = PAN_GRAPH_V15_MEMORY_INPUT,
       .bits = 32,
       .replication_rank = 3,
       .layout = PAN_GRAPH_MEMORY_COST_BLOCKED_260,
       .block = {1, 16, 1, 1},
       .full = {1, 1, 1},
       .transfer = {1, 1, 1},
       .retain_read_peak = true},
      {.type = PAN_GRAPH_V15_VE},
      {.type = PAN_GRAPH_V15_MEMORY_OUTPUT,
       .bits = 32,
       .replication_rank = 3,
       .full = {1, 1, 1},
       .transfer = {1, 1, 1}},
   };
   const enum pan_graph_memory_cost_view views[] = {
      PAN_GRAPH_MEMORY_COST_TENSOR_VIEW,
      PAN_GRAPH_MEMORY_COST_TENSOR_VIEW,
   };
   struct pan_graph_memory_cost_program program = {
      .sections = sections,
      .section_count = 3,
      .resource_count = 2,
      .input_views = views,
      .input_view_count = 2,
      .output_view_count = 1,
      .workload = {.rank = 3,
                   .extent = {1, 1, 1},
                   .transfer = {1, 1, 1},
                   .repeats = 1,
                   .minimum = .5f,
                   .latency = 170},
   };
   struct pan_graph_vector_cost_result result;
   assert(pan_graph_vector_cost_estimate_add_i32(&target, &program, &result));
   assert(result.engine[0].engine_cycles[3] == 1 && result.total.cycles == 85);
   program.workload.extent[0] = 3;
   program.workload.step[0] = 2;
   sections[0].full[2] = sections[2].full[2] = 3;
   assert(pan_graph_vector_cost_estimate_add_i32(&target, &program, &result));
   assert(result.engine[0].engine_cycles[3] == 2);
   struct pan_graph_memory_cost_section multiple[] = {
      sections[0], sections[0], sections[1],
      sections[1], sections[2], sections[2],
   };
   const enum pan_graph_memory_cost_view repeated[] = {
      PAN_GRAPH_MEMORY_COST_TENSOR_VIEW,
      PAN_GRAPH_MEMORY_COST_TENSOR_VIEW,
      PAN_GRAPH_MEMORY_COST_TENSOR_VIEW,
   };
   struct pan_graph_memory_cost_program combined = program;
   combined.sections = multiple;
   combined.section_count = 6;
   combined.resource_count = 4;
   combined.input_views = repeated;
   combined.input_view_count = 3;
   combined.output_view_count = 2;
   assert(pan_graph_vector_cost_estimate_program_add_i32(&target, &combined,
                                                         &result));
   assert(result.engine[0].engine_cycles[3] == 4);
   assert(result.memory[0].peak_resource[1] == 0);
   struct pan_graph_vector_cost_result saved = result;
   combined.output_view_count = 1;
   assert(!pan_graph_vector_cost_estimate_program_add_i32(&target, &combined,
                                                          &result));
   combined.output_view_count = 2;
   combined.resource_count = 3;
   assert(!pan_graph_vector_cost_estimate_program_add_i32(&target, &combined,
                                                          &result));
   combined.resource_count = 4;
   multiple[2].type = PAN_GRAPH_V15_CE;
   assert(!pan_graph_vector_cost_estimate_program_add_i32(&target, &combined,
                                                          &result));
   multiple[2].type = PAN_GRAPH_V15_MEMORY_INPUT;
   assert(!pan_graph_vector_cost_estimate_program_add_i32(&target, &combined,
                                                          &result));
   multiple[2].type = PAN_GRAPH_V15_VE;
   multiple[4].type = PAN_GRAPH_V15_VE;
   assert(!pan_graph_vector_cost_estimate_program_add_i32(&target, &combined,
                                                          &result));
   assert(!memcmp(&result, &saved, sizeof(result)));
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_vector_cost_result before = result;
   program.workload.axis[0] = 3;
   assert(!pan_graph_vector_cost_estimate_add_i32(&target, &program, &result));
   program.workload.axis[0] = 0;
   program.input_view_count = 1;
   assert(!pan_graph_vector_cost_estimate_add_i32(&target, &program, &result));
   program.input_view_count = 2;
   program.workload.rank = 7;
   assert(!pan_graph_vector_cost_estimate_add_i32(&target, &program, &result));
   program.workload.rank = 3;
   program.resource_count = 3;
   assert(!pan_graph_vector_cost_estimate_add_i32(&target, &program, &result));
   program.resource_count = 2;
   sections[0].bits = 8;
   assert(!pan_graph_vector_cost_estimate_add_i32(&target, &program, &result));
   sections[0].bits = 32;
   sections[0].layout = 263;
   assert(!pan_graph_vector_cost_estimate_add_i32(&target, &program, &result));
   sections[0].layout = PAN_GRAPH_MEMORY_COST_BLOCKED_260;
   program.workload.transfer[0] = 4096;
   assert(!pan_graph_vector_cost_estimate_add_i32(&target, &program, &result));
   program.workload.transfer[0] = 1;
   program.workload.extent[0] = UINT64_MAX;
   assert(!pan_graph_vector_cost_estimate_add_i32(&target, &program, &result));
   assert(!pan_graph_vector_cost_estimate_add_i32(NULL, &program, &result));
   assert(!pan_graph_vector_cost_estimate_add_i32(&target, NULL, &result));
   assert(!pan_graph_vector_cost_estimate_add_i32(&target, &program, NULL));
   assert(!memcmp(&result, &before, sizeof(result)));
   return 0;
}
