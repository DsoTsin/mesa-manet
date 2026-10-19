#include "pan_graph_cost.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int
main(void)
{
   uint32_t extent[4] = {INT32_MAX, 1, 1, 1};
   uint32_t transfer[4] = {1, 1, 1, 1};
   uint64_t cycles;
   assert(pan_graph_cost_v15_ve_i32(PAN_GRAPH_COST_VE_I32_IDENTITY, extent,
                                    transfer, &cycles));
   assert(cycles == UINT64_C(2147483648));
   extent[0] = 3;
   extent[3] = 4;
   transfer[0] = 3;
   transfer[3] = 4;
   assert(pan_graph_cost_v15_ve_i32(PAN_GRAPH_COST_VE_I32_IDENTITY, extent,
                                    transfer, &cycles) && cycles == 2);
   assert(pan_graph_cost_v15_ve_i32(PAN_GRAPH_COST_VE_I32_ADD, extent,
                                    transfer, &cycles) && cycles == 2);
   extent[0] = transfer[0] = 1;
   assert(pan_graph_cost_v15_ve_i32(PAN_GRAPH_COST_VE_I32_IDENTITY, extent,
                                    transfer, &cycles) && cycles == 2);
   assert(pan_graph_cost_v15_ve_i32(PAN_GRAPH_COST_VE_I32_ADD, extent,
                                    transfer, &cycles) && cycles == 1);
   cycles = 0xabcdef;
   transfer[1] = 2;
   assert(!pan_graph_cost_v15_ve_i32(PAN_GRAPH_COST_VE_I32_ADD, extent,
                                     transfer, &cycles) && cycles == 0xabcdef);
   transfer[1] = 1;
   transfer[0] = 4096;
   assert(!pan_graph_cost_v15_ve_i32(PAN_GRAPH_COST_VE_I32_ADD, extent,
                                     transfer, &cycles) && cycles == 0xabcdef);
   struct pan_graph_cost_report a[3] = {
      {.resource_bytes = {{[5] = 5120}, {[5] = 4096}},
       .peak_resource = {0, 295}, .memory_cycles = {[5] = 88},
       .engine_cycles = {[1] = 240, [6] = 256},
       .critical_kind = 6, .critical_cycles = 256},
      {.peak_resource = {0, 295}, .critical_kind = 6},
      {.cycles = 3},
   };
   const struct pan_graph_cost_report b[3] = {
      {.engine_cycles = {[2] = 64, [3] = 32, [4] = 128},
       .critical_kind = 4, .critical_cycles = 128},
      {.critical_kind = 4},
      {0},
   };
   struct pan_graph_cost_report report;
   assert(pan_graph_cost_combine(a, b, &report));
   assert(report.cycles == 259 && report.critical_cycles == 128 &&
          report.critical_kind == 4);
   assert(report.resource_bytes[0][5] == 5120 &&
          report.resource_bytes[1][5] == 4096 && report.peak_resource[1] == 295);
   assert(report.engine_cycles[1] == 240 && report.engine_cycles[2] == 64 &&
          report.engine_cycles[3] == 32 && report.engine_cycles[4] == 128 &&
          report.engine_cycles[6] == 256);
   struct pan_graph_cost_workload w = {
      .extent = {32, 1, 1, 32, 32, 1, 1},
      .transfer = {32, 1, 1, 8, 32, 1, 1},
      .rank = 7, .step = {8}, .axis = {3},
      .repeats = 1, .latency = 170, .minimum = 0.5f,
   };
   uint64_t result;
   assert(pan_graph_cost_apply_workload(&w, report.cycles, &result) && result == 340);
   w.transfer[3] = 4;
   assert(pan_graph_cost_apply_workload(&w, 435, &result) && result == 680);
   w.transfer[3] = 2;
   w.step[0] = 6;
   assert(pan_graph_cost_apply_workload(&w, 651, &result) && result == 1360);
   w.transfer[0] = 11;
   w.transfer[3] = 16;
   w.axis[0] = 0;
   w.axis[1] = 3;
   w.step[0] = 11;
   w.step[1] = 16;
   assert(pan_graph_cost_apply_workload(&w, 195, &result) && result == 494);
   w = (struct pan_graph_cost_workload){
      .extent = {17, 1, 1, 17, 17, 1, 1},
      .transfer = {3, 1, 1, 17, 17, 1, 1},
      .rank = 7, .step = {3}, .axis = {0},
      .repeats = 1, .latency = 170, .minimum = 0.5f,
   };
   assert(pan_graph_cost_apply_workload(&w, 83, &result) && result == 481);
   result = 0xabcdef;
   w.transfer[0] = 0;
   assert(!pan_graph_cost_apply_workload(&w, 83, &result) && result == 0xabcdef);
   w.transfer[0] = 3;
   w.step[1] = 1;
   assert(!pan_graph_cost_apply_workload(&w, 83, &result) && result == 0xabcdef);
   w.step[1] = 0;
   w.minimum = NAN;
   assert(!pan_graph_cost_apply_workload(&w, 83, &result) && result == 0xabcdef);
   w.minimum = 0.0f;
   assert(pan_graph_cost_apply_workload(&w, INT64_MAX, &result) && result == INT64_MAX);
   assert(!pan_graph_cost_apply_workload(&w, UINT64_MAX, &result));
   w.minimum = 0.5f;
   assert(!pan_graph_cost_apply_workload(&w, INT64_MAX, &result));
   memset(&report, 0xa5, sizeof(report));
   struct pan_graph_cost_report saved = report;
   a[2].cycles = INT64_MAX;
   assert(!pan_graph_cost_combine(a, b, &report));
   assert(!memcmp(&saved, &report, sizeof(report)));
   puts("graph cost original-candidate goldens and rejection guards passed");
   return 0;
}
