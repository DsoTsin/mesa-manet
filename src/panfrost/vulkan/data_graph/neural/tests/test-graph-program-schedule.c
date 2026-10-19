#include "pan_graph_program_schedule.h"

#include <assert.h>
#include <string.h>

int
main(void)
{
   const struct pan_graph_memory_cost_target cost = {
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
   const struct pan_graph_nrt_target nrt = pan_graph_nrt_target_v15();
   const struct pan_graph_matmul_add add = {{0, 0}};
   const struct {
      struct pan_graph_matmul_info info;
      uint64_t cycles;
      size_t count;
   } cases[] = {
      {{.batch = 2, .rows = 3, .inner = 5, .columns = 4}, 170, 1},
      {{.batch = 1, .rows = 32, .inner = 32, .columns = 32}, 494, 1},
      {{.batch = 1, .rows = 4096, .inner = 1, .columns = 1}, 4278, 2},
      {{.batch = 1, .rows = 1, .inner = 1, .columns = 4096}, 4150, 2},
   };
   for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
      struct pan_graph_program_schedule result = {0};
      assert(pan_graph_program_schedule_matmul_add_v15(
                &cases[c].info, &add, 1, &nrt, &cost, 6, 8, &result) ==
             PAN_GRAPH_TSU_SUCCESS);
      assert(result.cycles == cases[c].cycles &&
             result.count == cases[c].count);
      assert(result.group_count == 3 && result.candidate_count >= result.count);
      uint64_t groups = 0, cycles = 0;
      for (size_t i = 0; i < result.count; i++) {
         const struct pan_graph_scheduled_program *p = result.programs + i;
         assert(p->groups == result.groups + i);
         groups |= *p->groups;
         cycles += p->selection.cost.total.cycles;
         assert(p->plan.sections == p->plan.program.sections);
         assert(p->plan.input_views == p->plan.program.input_views);
         struct pan_graph_program_cost_result repeated;
         assert(pan_graph_program_cost_plan_estimate(
                   &result.programs[i].plan, p->selection.transfer,
                   p->selection.loops, &cost,
                   &repeated) == PAN_GRAPH_TSU_SUCCESS);
         assert(repeated.total.cycles == p->selection.cost.total.cycles);
      }
      assert(groups == 7 && cycles == result.cycles);
      pan_graph_program_schedule_finish(&result);
      assert(!result.programs && !result.groups && !result.count);
      pan_graph_program_schedule_finish(&result);
   }
   struct pan_graph_program_schedule rejected, before;
   memset(&rejected, 0xa5, sizeof(rejected));
   before = rejected;
   assert(pan_graph_program_schedule_matmul_add_v15(
             &cases[0].info, &add, 1, &nrt, &cost, 0, 8, &rejected) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&rejected, &before, sizeof(rejected)));
   const struct pan_graph_matmul_add invalid = {{1, 0}};
   assert(pan_graph_program_schedule_matmul_add_v15(
             &cases[0].info, &invalid, 1, &nrt, &cost, 6, 8, &rejected) !=
          PAN_GRAPH_TSU_SUCCESS);
   assert(!memcmp(&rejected, &before, sizeof(rejected)));
   pan_graph_program_schedule_finish(NULL);
   return 0;
}
