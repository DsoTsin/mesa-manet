#include "pan_graph_program_cost.h"

#include <assert.h>
#include <string.h>

int
main(void)
{
   const struct pan_graph_memory_cost_target target = {
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
   struct pan_graph_memory_cost_target initialized;
   assert(pan_graph_program_cost_target_v15_init(8, 64, 262144, 6,
                                                &initialized));
   assert(!memcmp(&initialized, &target, sizeof(target)));
   struct pan_graph_memory_cost_target before = initialized;
   assert(!pan_graph_program_cost_target_v15_init(8, 64, 262144, 0,
                                                 &initialized));
   assert(!memcmp(&initialized, &before, sizeof(before)));
   assert(!pan_graph_program_cost_target_v15_init(8, 63, 262144, 6,
                                                 &initialized));
   assert(!memcmp(&initialized, &before, sizeof(before)));
   const struct pan_graph_matmul_info info = {
      .batch = 2,
      .rows = 3,
      .inner = 5,
      .columns = 4,
   };
   struct pan_graph_matmul_add adds[70] = {{{0, 0}}, {{1, 1}}};
   uint32_t transfer[7] = {2, 3, 4}, loops[4] = {0};
   uint64_t selected[2] = {12, 0};
   struct pan_graph_program_cost_result result;
   assert(pan_graph_program_cost_matmul_add_v15(
             &info, adds, 2, selected, transfer, loops, &target, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.input_view_count == 4 && result.output_view_count == 1);
   assert(result.read_count == 1 && result.vector_count == 2 &&
          result.resource_count == 3 && result.total.cycles == 85);
   assert(result.memory[0].peak_resource[0] == 190);
   assert(result.memory[0].resource_bytes[0][5] == 128 &&
          result.memory[0].resource_bytes[1][5] == 96 &&
          result.memory[0].resource_bytes[1][2] == 0);
   struct pan_graph_program_cost_plan plan = {0};
   assert(pan_graph_program_cost_plan_matmul_add_v15(
             &info, adds, 2, selected, &plan) == PAN_GRAPH_TSU_SUCCESS);
   assert(plan.boundary.input_count == 4 && plan.boundary.output_count == 1);
   assert(plan.read_tensors[0] == 0 && plan.read_count == 1);
   assert(plan.vector_groups[0] == 2 && plan.vector_groups[1] == 3 &&
          plan.vector_count == 2);
   assert(pan_graph_program_cost_plan_estimate(&plan, transfer, loops, &target,
                                               &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.total.cycles == 85 &&
          result.memory[0].peak_resource[0] == 190);
   struct pan_graph_program_cost_result saved_plan_cost = result;
   transfer[0] = 0;
   assert(pan_graph_program_cost_plan_estimate(&plan, transfer, loops, &target,
                                               &result) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&result, &saved_plan_cost, sizeof(result)));
   transfer[0] = 2;
   assert(pan_graph_program_cost_plan_estimate(&plan, transfer, loops, &target,
                                               &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.total.cycles == 85);
   pan_graph_program_cost_plan_finish(&plan);
   assert(!plan.sections && !plan.input_views && !plan.vector_ops &&
          !plan.read_tensors && !plan.vector_groups && !plan.read_count &&
          !plan.boundary.inputs && !plan.boundary.outputs);
   pan_graph_program_cost_plan_finish(NULL);
   adds[1].input[1] = 0;
   assert(pan_graph_program_cost_matmul_add_v15(
             &info, adds, 2, selected, transfer, loops, &target, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.input_view_count == 3 &&
          result.memory[0].peak_resource[0] == 0);
   selected[0] = 8;
   assert(pan_graph_program_cost_matmul_add_v15(
             &info, adds, 2, selected, transfer, loops, &target, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.read_count == 2 && result.resource_count == 3 &&
          result.memory[0].resource_bytes[0][5] == 256);
   adds[1].input[1] = 1;
   selected[0] = 7;
   transfer[0] = 4;
   transfer[1] = 1;
   transfer[2] = 1;
   transfer[3] = 3;
   transfer[4] = 5;
   transfer[5] = transfer[6] = 1;
   loops[0] = 2;
   loops[2] = 1;
   assert(pan_graph_program_cost_matmul_add_v15(
             &info, adds, 2, selected, transfer, loops, &target, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.rank == 7 && result.output_view_count == 2 &&
          result.total.cycles == 170 && result.memory[0].peak_resource[1] == 0);
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_program_cost_result saved = result;
   selected[0] = 0;
   assert(pan_graph_program_cost_matmul_add_v15(
             &info, adds, 2, selected, transfer, loops, &target, &result) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   struct pan_graph_program_cost_plan rejected, original;
   memset(&rejected, 0xa5, sizeof(rejected));
   original = rejected;
   assert(pan_graph_program_cost_plan_matmul_add_v15(
             &info, adds, 2, selected, &rejected) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&rejected, &original, sizeof(rejected)));
   selected[0] = 2;
   assert(pan_graph_program_cost_matmul_add_v15(
             &info, adds, 2, selected, transfer, loops, &target, &result) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   selected[0] = 12;
   adds[0].input[1] = 1;
   assert(pan_graph_program_cost_matmul_add_v15(
             &info, adds, 2, selected, transfer, loops, &target, &result) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&result, &saved, sizeof(result)));
   for (uint32_t i = 0; i < 70; i++)
      adds[i] = (struct pan_graph_matmul_add){{i, i}};
   selected[0] = 0;
   selected[1] = UINT64_C(1) << 7;
   transfer[0] = 2;
   transfer[1] = 3;
   transfer[2] = 4;
   memset(loops, 0, sizeof(loops));
   assert(pan_graph_program_cost_matmul_add_v15(
             &info, adds, 70, selected, transfer, loops, &target, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.rank == 3 && result.read_count == 1 &&
          result.total.cycles == 85);
   return 0;
}
