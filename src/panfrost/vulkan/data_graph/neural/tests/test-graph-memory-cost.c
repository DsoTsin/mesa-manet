#include "pan_graph_memory_cost.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <string.h>

static struct pan_graph_memory_cost_target
target(void)
{
   struct pan_graph_memory_cost_target result = {
      .memory = {
         [2] = {.bandwidth = {32, 16}, .latency = {12, 12},
                .parallel = 1, .line_size = 64, .size = 32768},
         [3] = {.bandwidth = {32, 32}, .latency = {60, 60},
                .parallel = 8, .line_size = 64, .size = 262144},
         [4] = {.bandwidth = {32, 32}, .latency = {100, 100},
                .parallel = 1, .line_size = 64, .size = 8388608},
         [5] = {.bandwidth = {104.30635833740234f, 104.30635833740234f},
                .latency = {170, 100}, .parallel = 1, .line_size = 64,
                .size = 268435456},
      },
      .descriptor_bytes = {36, 64, 32, 12},
      .section_bytes = {0, 8, 20, 32, 8, 8, 8, 20},
      .alignment = 32,
      .issue_width = 32,
      .parallel_budget = 6,
   };
   return result;
}

static void
reject(const struct pan_graph_memory_cost_target *target,
       const struct pan_graph_memory_cost_program *program)
{
   struct pan_graph_cost_report actual[3], saved[3];
   memset(actual, 0xa5, sizeof(actual));
   memcpy(saved, actual, sizeof(saved));
   assert(!pan_graph_memory_cost_estimate(target, program, actual));
   assert(!memcmp(saved, actual, sizeof(saved)));
}

static void
standalone(const struct pan_graph_memory_cost_target *profile)
{
   struct pan_graph_memory_cost_section sections[] = {
      {.type = PAN_GRAPH_V15_MEMORY_INPUT,
       .layout = PAN_GRAPH_MEMORY_COST_BLOCKED_262,
       .full = {4, 3, 17}, .transfer = {4, 3, 3}, .block = {4, 4, 1, 1},
       .bits = 32, .replication_rank = 3, .retain_read_peak = true},
      {.type = PAN_GRAPH_V15_VE},
      {.type = PAN_GRAPH_V15_MEMORY_OUTPUT, .full = {4, 3, 17},
       .transfer = {4, 3, 3}, .bits = 32, .replication_rank = 3},
   };
   enum pan_graph_memory_cost_view views[] = {
      PAN_GRAPH_MEMORY_COST_TENSOR_VIEW, PAN_GRAPH_MEMORY_COST_TENSOR_VIEW,
   };
   struct pan_graph_memory_cost_program program = {
      .sections = sections, .section_count = 3,
      .input_views = views, .input_view_count = 2,
      .output_view_count = 1, .resource_count = 2,
      .workload = {.rank = 3, .extent = {17, 3, 4}, .transfer = {3, 3, 4},
                   .axis = {0, 0}, .step = {3, 0}},
   };
   struct pan_graph_cost_report reports[3], cleared[3];
   assert(pan_graph_memory_cost_estimate(profile, &program, reports));
   assert(reports[0].resource_bytes[0][5] == 1088 &&
          reports[0].resource_bytes[1][5] == 816 &&
          reports[0].resource_bytes[1][2] == 816 &&
          reports[0].engine_cycles[1] == 24 &&
          reports[0].engine_cycles[6] == 48 &&
          reports[0].peak_resource[0] == 205 &&
          reports[0].peak_resource[1] == 65 &&
          reports[0].memory_cycles[5] == 18 && reports[2].cycles == 3);
   assert(!memcmp(reports[0].peak_resource, reports[1].peak_resource,
                  sizeof(reports[0].peak_resource)));
   sections[0].retain_read_peak = false;
   assert(pan_graph_memory_cost_estimate(profile, &program, cleared));
   assert(cleared[0].peak_resource[0] == 0 &&
          cleared[1].peak_resource[0] == 0);
   assert(!memcmp(reports[0].resource_bytes, cleared[0].resource_bytes,
                  sizeof(reports[0].resource_bytes)));
   sections[0].retain_read_peak = true;
   sections[2].retain_read_peak = true;
   reject(profile, &program);
   sections[2].retain_read_peak = false;
   sections[1].retain_read_peak = true;
   reject(profile, &program);
   sections[1].retain_read_peak = false;
   sections[0].full[0]++;
   reject(profile, &program);
   sections[0].full[0]--;
   sections[2].transfer[2]++;
   reject(profile, &program);
   sections[2].transfer[2]--;
   sections[0].replication_rank = 5;
   reject(profile, &program);
   sections[0].replication_rank = 3;
   sections[0].bits = 8;
   reject(profile, &program);
   sections[0].bits = 32;
   sections[2].replication_rank = 4;
   reject(profile, &program);
   sections[2].replication_rank = 3;
   sections[2].bits = 8;
   reject(profile, &program);
   sections[2].bits = 32;
   sections[0].layout = 263;
   reject(profile, &program);
   sections[0].layout = PAN_GRAPH_MEMORY_COST_BLOCKED_262;
   program.workload.axis[0] = 3;
   reject(profile, &program);
   program.workload.axis[0] = 0;
   program.workload.rank = 4;
   reject(profile, &program);
   program.workload.rank = 3;
   program.workload.extent[0] = 0;
   reject(profile, &program);
   program.workload.extent[0] = 17;
   sections[0].block[0]++;
   reject(profile, &program);
   sections[0].block[0]--;
   assert(pan_graph_memory_cost_estimate(profile, &program, reports));
}

int
main(void)
{
   struct pan_graph_memory_cost_target profile = target();
   struct pan_graph_memory_cost_section sections[] = {
      {.type = PAN_GRAPH_V15_MEMORY_INPUT, .bits = 8,
       .full = {4, 5, 2}, .transfer = {4, 5, 1}, .replication_rank = 5},
      {.type = PAN_GRAPH_V15_MEMORY_INPUT, .bits = 8,
       .full = {5, 3, 2}, .transfer = {5, 3, 1}, .replication_rank = 5},
      {.type = PAN_GRAPH_V15_TRANSFORM},
      {.type = PAN_GRAPH_V15_CE},
      {.type = PAN_GRAPH_V15_VE},
      {.type = PAN_GRAPH_V15_VE},
      {.type = PAN_GRAPH_V15_MEMORY_OUTPUT, .bits = 32,
       .full = {4, 3, 2}, .transfer = {4, 3, 1}, .replication_rank = 4},
      {.type = PAN_GRAPH_V15_MEMORY_OUTPUT, .bits = 32,
       .full = {4, 3, 2}, .transfer = {4, 3, 1}, .replication_rank = 4},
   };
   enum pan_graph_memory_cost_view views[] = {
      PAN_GRAPH_MEMORY_COST_TENSOR_VIEW, PAN_GRAPH_MEMORY_COST_WEIGHT_VIEW,
   };
   struct pan_graph_memory_cost_program program = {
      .sections = sections, .section_count = 8,
      .input_views = views, .input_view_count = 2,
      .output_view_count = 2, .resource_count = 6,
      .workload = {.extent = {4, 1, 2, 3, 5, 1, 1},
                   .transfer = {4, 1, 1, 3, 5, 1, 1}, .rank = 7},
   };
   struct pan_graph_cost_report reports[3], single[3];
   assert(pan_graph_memory_cost_estimate(&profile, &program, reports));
   sections[0].retain_read_peak = sections[1].retain_read_peak = true;
   assert(pan_graph_memory_cost_estimate(&profile, &program, single));
   assert(single[0].peak_resource[0] > 0 && single[1].peak_resource[0] > 0);
   sections[1].retain_read_peak = false;
   assert(pan_graph_memory_cost_estimate(&profile, &program, single));
   assert(single[0].peak_resource[0] == 0);
   sections[0].retain_read_peak = false;
   assert(reports[0].peak_resource[1] == 0);
   assert(reports[2].cycles == 4);
   program.section_count = 7;
   program.output_view_count = 1;
   assert(pan_graph_memory_cost_estimate(&profile, &program, single));
   assert(single[0].peak_resource[1] > 0);
   assert(reports[0].resource_bytes[1][5] ==
          2 * single[0].resource_bytes[1][5]);
   assert(reports[0].engine_cycles[6] == 2 * single[0].engine_cycles[6]);
   program.section_count = 8;
   program.output_view_count = 2;
   assert(!pan_graph_memory_cost_estimate(NULL, &program, reports));
   reject(&profile, NULL);
   assert(!pan_graph_memory_cost_estimate(&profile, &program, NULL));
   struct pan_graph_memory_cost_program bad = program;
   bad.sections = NULL;
   reject(&profile, &bad);
   bad = program;
   bad.input_views = NULL;
   reject(&profile, &bad);
   bad = program;
   bad.workload.rank = 8;
   reject(&profile, &bad);
   bad = program;
   bad.workload.step[1] = 1;
   reject(&profile, &bad);
   bad = program;
   bad.workload.step[0] = bad.workload.step[1] = 1;
   reject(&profile, &bad);
   bad = program;
   bad.workload.extent[0] = 0;
   reject(&profile, &bad);
   bad = program;
   bad.workload.transfer[0] = 4096;
   reject(&profile, &bad);
   bad = program;
   bad.workload.extent[0] = bad.workload.extent[3] = INT32_MAX;
   bad.workload.step[0] = bad.workload.step[1] = 1;
   bad.workload.axis[0] = 0;
   bad.workload.axis[1] = 3;
   reject(&profile, &bad);
   enum pan_graph_memory_cost_view saved = views[1];
   views[1] = 0;
   reject(&profile, &program);
   views[1] = saved;
   sections[0].type = PAN_GRAPH_V15_MEMORY_CONSTANT;
   reject(&profile, &program);
   sections[0].type = PAN_GRAPH_V15_MEMORY_INPUT;
   sections[0].layout = 1;
   reject(&profile, &program);
   sections[0].layout = PAN_GRAPH_MEMORY_COST_COMPACT_LINEAR;
   sections[0].bits = 16;
   reject(&profile, &program);
   sections[0].bits = 8;
   sections[0].replication_rank = 4;
   reject(&profile, &program);
   sections[0].replication_rank = 5;
   sections[0].transfer[0] = 0;
   reject(&profile, &program);
   sections[0].transfer[0] = 4;
   struct pan_graph_memory_cost_target broken = profile;
   broken.memory[5].bandwidth[0] = NAN;
   reject(&broken, &program);
   broken = profile;
   broken.memory[2].bandwidth[1] = 0;
   reject(&broken, &program);
   broken = profile;
   broken.memory[5].line_size = 0;
   reject(&broken, &program);
   broken = profile;
   broken.alignment = 0;
   reject(&broken, &program);
   assert(pan_graph_memory_cost_estimate(&profile, &program, reports));
   struct pan_graph_memory_cost_section blocked_sections[] = {
      {.type = PAN_GRAPH_V15_MEMORY_INPUT, .bits = 8,
       .full = {1, 1, 1}, .transfer = {1, 1, 1}, .replication_rank = 5},
      {.type = PAN_GRAPH_V15_MEMORY_INPUT, .bits = 8,
       .full = {1, 1, 1}, .transfer = {1, 1, 1}, .replication_rank = 5},
      {.type = PAN_GRAPH_V15_TRANSFORM},
      {.type = PAN_GRAPH_V15_CE},
      {.type = PAN_GRAPH_V15_VE},
      {.type = PAN_GRAPH_V15_MEMORY_OUTPUT, .bits = 32,
       .full = {1, 1, 1}, .transfer = {1, 1, 1}, .replication_rank = 4},
   };
   struct pan_graph_memory_cost_program blocked_program = program;
   blocked_program.sections = blocked_sections;
   blocked_program.section_count = 6;
   blocked_program.output_view_count = 1;
   blocked_program.resource_count = 5;
   blocked_program.workload = (struct pan_graph_cost_workload){
      .rank = 7, .extent = {1, 1, 1, 1, 1, 1, 1},
      .transfer = {1, 1, 1, 1, 1, 1, 1},
   };
   assert(pan_graph_memory_cost_estimate(&profile, &blocked_program, reports));
   assert(reports[0].engine_cycles[6] == 16 &&
          reports[0].resource_bytes[1][2] == 0);
   for (unsigned layout = 260; layout <= 262; layout++) {
      blocked_sections[5].layout = layout;
      blocked_sections[5].block[0] = 512 / (layout == 260 ? 16 : layout == 261 ? 8 : 4) / 32;
      blocked_sections[5].block[1] = layout == 260 ? 16 : layout == 261 ? 8 : 4;
      blocked_sections[5].block[2] = blocked_sections[5].block[3] = 1;
      assert(pan_graph_memory_cost_estimate(&profile, &blocked_program, reports));
      assert(reports[0].engine_cycles[6] == 4 &&
             reports[0].resource_bytes[1][5] == 4 &&
             reports[0].resource_bytes[1][2] == 4);
      blocked_sections[5].block[0]++;
      reject(&profile, &blocked_program);
      blocked_sections[5].block[0]--;
   }
   blocked_sections[5].layout = 263;
   reject(&profile, &blocked_program);
   blocked_sections[5].layout = PAN_GRAPH_MEMORY_COST_BLOCKED_262;
   blocked_sections[0].layout = PAN_GRAPH_MEMORY_COST_BLOCKED_262;
   blocked_sections[0].block[0] = 16;
   blocked_sections[0].block[1] = 4;
   blocked_sections[0].block[2] = blocked_sections[0].block[3] = 1;
   assert(pan_graph_memory_cost_estimate(&profile, &blocked_program, reports));
   blocked_sections[0].bits = 32;
   blocked_sections[0].replication_rank = 3;
   reject(&profile, &blocked_program);
   blocked_sections[0].bits = 8;
   blocked_sections[0].replication_rank = 5;
   blocked_sections[0].full[2] = blocked_sections[0].transfer[2] = 2;
   reject(&profile, &blocked_program);
   standalone(&profile);
   return 0;
}
