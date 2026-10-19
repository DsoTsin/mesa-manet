#include "pan_graph_matmul_cost.h"
#include "pan_graph_memory_cost.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void
reject(const struct pan_graph_matmul_info *info, const uint32_t transfer[7],
       const uint32_t loops[4],
       const struct pan_graph_matmul_cost_target *target, uint32_t budget)
{
   struct pan_graph_matmul_cost_result result, original;
   memset(&result, 0xa5, sizeof(result));
   original = result;
   assert(!pan_graph_matmul_cost_estimate(info, transfer, loops, target,
                                        budget, &result));
   assert(!memcmp(&result, &original, sizeof(result)));
}

static void
test_program(const struct pan_graph_matmul_cost_target *normal)
{
   struct pan_graph_memory_cost_target target = {
      .descriptor_bytes = {36, 64, 32, 12},
      .section_bytes = {0, 8, 20, 32, 8, 8, 8, 20},
      .alignment = 32, .issue_width = 32, .parallel_budget = 6,
   };
   for (unsigned level = 0; level < 6; level++) {
      const struct pan_graph_matmul_cost_memory *source = &normal->memory[level];
      target.memory[level] = (struct pan_graph_memory_cost_resource){
         .bandwidth = {source->bandwidth[0], source->bandwidth[1]},
         .latency = {source->latency[0], source->latency[1]},
         .parallel = source->parallel, .line_size = source->line_size,
         .size = source->size,
      };
   }
   const enum pan_graph_memory_cost_view views[2] = {
      PAN_GRAPH_MEMORY_COST_TENSOR_VIEW, PAN_GRAPH_MEMORY_COST_WEIGHT_VIEW,
   };
   enum pan_graph_cost_ve_i32_op ops[2] = {
      PAN_GRAPH_COST_VE_I32_IDENTITY, PAN_GRAPH_COST_VE_I32_ADD,
   };
   static const struct {
      struct pan_graph_matmul_info info;
      uint32_t transfer[7], loops[4];
      uint64_t cycles;
   } cases[] = {
      {{1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0}, 85},
      {{1, 17, 17, 17, 0, 0}, {17, 1, 1, 17, 17, 1, 1}, {0, 0, 0, 0}, 724},
      {{1, 32, 32, 32, 0, 0}, {32, 1, 1, 32, 32, 1, 1}, {0, 0, 0, 0}, 2052},
      {{1, 32, 32, 32, 0, 0}, {11, 1, 1, 16, 32, 1, 1}, {0, 3, 11, 16}, 494},
      {{2, 3, 5, 4, 0, 0}, {4, 1, 1, 3, 5, 1, 1}, {2, 0, 1, 0}, 170},
      {{1, 1, 1025, 1, 0, 0}, {1, 1, 1, 1, 1024, 1, 1}, {0, 0, 0, 0}, 1797},
      {{3, 17, 4096, 32, -128, 127}, {16, 1, 1, 16, 512, 1, 1}, {0, 2, 16, 1}, 26116},
      {{4096, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 1, 1, 1}, {2, 0, 683, 0}, 348160},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
      const struct pan_graph_matmul_info *info = &cases[i].info;
      const uint32_t *transfer = cases[i].transfer, *loops = cases[i].loops;
      struct pan_graph_memory_cost_section sections[8] = {
         {.type = PAN_GRAPH_V15_MEMORY_INPUT, .bits = 8, .replication_rank = 5,
          .full = {info->columns, info->inner, info->batch},
          .transfer = {transfer[0], transfer[4], 1}},
         {.type = PAN_GRAPH_V15_MEMORY_INPUT, .bits = 8, .replication_rank = 5,
          .full = {info->inner, info->rows, info->batch},
          .transfer = {transfer[4], transfer[3], 1}},
         {.type = PAN_GRAPH_V15_TRANSFORM}, {.type = PAN_GRAPH_V15_CE},
         {.type = PAN_GRAPH_V15_VE}, {.type = PAN_GRAPH_V15_VE},
         {.type = PAN_GRAPH_V15_MEMORY_OUTPUT, .bits = 32, .replication_rank = 4,
          .full = {info->columns, info->rows, info->batch},
          .transfer = {transfer[0], transfer[3], 1}},
      };
      sections[7] = sections[6];
      struct pan_graph_memory_cost_program program = {
         .sections = sections, .section_count = 8,
         .input_views = views, .input_view_count = 2,
         .output_view_count = 2, .resource_count = 6,
         .workload = {.rank = 7, .repeats = 1, .latency = 170, .minimum = 0.5f,
                      .axis = {loops[0], loops[1]},
                      .step = {loops[2], loops[3]},
                      .extent = {info->columns, 1, info->batch, info->rows,
                                 info->inner, 1, 1}},
      };
      for (unsigned axis = 0; axis < 7; axis++)
         program.workload.transfer[axis] = transfer[axis];
      struct pan_graph_matmul_cost_result result, saved;
      assert(pan_graph_matmul_cost_estimate_program(info, &target, &program,
                                                   ops, 2, &result));
      assert(result.total.cycles == cases[i].cycles);
      assert(result.memory[0].peak_resource[1] == 0);
      assert(result.memory[1].peak_resource[1] == 0);
      saved = result;
      for (unsigned invalid = 0; invalid < 9; invalid++) {
         struct pan_graph_memory_cost_program bad = program;
         struct pan_graph_memory_cost_target bad_target = target;
         uint32_t count = 2;
         switch (invalid) {
         case 0: count = 0; break;
         case 1: count = 9; break;
         case 2: bad.resource_count--; break;
         case 3: bad.output_view_count--; break;
         case 4: bad.sections = NULL; break;
         case 5: bad.workload.repeats = 0; break;
         case 6: bad.workload.extent[0]++; break;
         case 7: bad_target.issue_width = 0; break;
         case 8: ops[1] = 99; break;
         }
         assert(!pan_graph_matmul_cost_estimate_program(info, &bad_target,
                                                        &bad, ops, count, &result));
         assert(!memcmp(&result, &saved, sizeof(result)));
         ops[1] = PAN_GRAPH_COST_VE_I32_ADD;
      }
   }
}

int
main(void)
{
   struct pan_graph_matmul_cost_target target;
   assert(pan_graph_matmul_cost_target_v15_init(8, 64, 262144, &target));
   uint32_t bits;
   memcpy(&bits, &target.memory[5].bandwidth[0], sizeof(bits));
   assert(bits == 1120967899);
   assert(target.memory[3].parallel == 8);
   assert(target.memory[3].line_size == 64);
   assert(target.memory[3].size == 262144);
   assert(target.memory[2].bandwidth[1] == 16.0f);
   assert(target.memory[5].latency[0] == 170);
   test_program(&target);

   static const struct {
      struct pan_graph_matmul_info info;
      uint32_t transfer[7];
      uint32_t loops[4];
      uint32_t cycles;
   } cases[] = {
      {{1, 17, 17, 17, 0, 0}, {17, 1, 1, 17, 17, 1, 1}, {0, 0, 0, 0}, 363},
      {{1, 17, 17, 17, 0, 0}, {3, 1, 1, 17, 17, 1, 1}, {0, 0, 3, 0}, 481},
      {{1, 32, 32, 32, 0, 0}, {32, 1, 1, 32, 32, 1, 1}, {0, 0, 0, 0}, 1027},
      {{1, 32, 32, 32, 0, 0}, {32, 1, 1, 16, 32, 1, 1}, {3, 0, 16, 0}, 515},
      {{1, 32, 32, 32, 0, 0}, {32, 1, 1, 8, 32, 1, 1}, {3, 0, 8, 0}, 340},
      {{1, 32, 32, 32, 0, 0}, {32, 1, 1, 4, 32, 1, 1}, {3, 0, 8, 0}, 680},
      {{1, 32, 32, 32, 0, 0}, {32, 1, 1, 2, 32, 1, 1}, {3, 0, 6, 0}, 1360},
      {{1, 32, 32, 32, 0, 0}, {11, 1, 1, 32, 32, 1, 1}, {0, 0, 11, 0}, 387},
      {{1, 32, 32, 32, 0, 0}, {11, 1, 1, 16, 32, 1, 1}, {0, 3, 11, 16}, 494},
      {{2, 3, 5, 4, 0, 0}, {4, 1, 1, 3, 5, 1, 1}, {2, 0, 1, 0}, 170},
      {{2, 3, 5, 4, 0, 0}, {2, 1, 1, 3, 5, 1, 1}, {0, 2, 2, 1}, 340},
      {{1, 1, 1025, 1, 0, 0}, {1, 1, 1, 1, 512, 1, 1}, {0, 0, 0, 0}, 1796},
      {{3, 17, 4096, 32, 0, 0}, {32, 1, 1, 17, 16, 1, 1}, {2, 0, 1, 0}, 65280},
   };
   struct pan_graph_matmul_cost_result result;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
      assert(pan_graph_matmul_cost_estimate(&cases[i].info, cases[i].transfer,
                                           cases[i].loops, &target, 6, &result));
      assert(result.total.cycles == cases[i].cycles);
      if (i == 0) {
         assert(result.memory[0].resource_bytes[0][5] == 680);
         assert(result.memory[0].resource_bytes[1][2] == 1156);
         assert(result.memory[0].resource_bytes[1][5] == 1156);
         assert(result.memory[0].peak_resource[1] == 468);
         assert(result.memory[0].engine_cycles[1] == 90);
         assert(result.memory[0].engine_cycles[6] == 360);
         assert(result.memory[1].peak_resource[1] == 468);
         assert(result.memory[2].cycles == 3);
         assert(result.engine[0].engine_cycles[2] == 64);
         assert(result.engine[0].engine_cycles[3] == 45);
         assert(result.engine[0].engine_cycles[4] == 51);
      }
      if (i == 4) {
         assert(result.memory[0].resource_bytes[0][5] == 5120);
         assert(result.memory[0].resource_bytes[1][2] == 0);
         assert(result.memory[0].resource_bytes[1][5] == 4096);
         assert(result.memory[0].engine_cycles[6] == 256);
         assert(result.total.critical_cycles == 128);
      }
      if (i == 11) {
         assert(result.engine[0].engine_cycles[2] == 1025);
         assert(result.engine[0].engine_cycles[3] == 1);
         assert(result.engine[0].engine_cycles[4] == 128);
      }
      if (i == 12) {
         assert(result.engine[0].engine_cycles[2] == 16384);
         assert(result.engine[0].engine_cycles[3] == 72);
         assert(result.total.critical_kind == 2);
      }
   }

   struct pan_graph_matmul_info info = {2, 3, 5, 4, -128, 127};
   uint32_t transfer[7] = {4, 1, 1, 3, 5, 1, 1};
   uint32_t loops[4] = {2, 0, 1, 0};
   assert(pan_graph_matmul_cost_estimate(&info, transfer, loops,
                                        &target, 6, &result));
   assert(result.total.cycles == 170);
   info.a_zero_point = 128;
   reject(&info, transfer, loops, &target, 6);
   info.a_zero_point = -129;
   reject(&info, transfer, loops, &target, 6);
   info.a_zero_point = 0;
   info.batch = 0;
   reject(&info, transfer, loops, &target, 6);
   info.batch = UINT32_MAX;
   reject(&info, transfer, loops, &target, 6);
   info.batch = 2;
   for (unsigned i = 0; i < 7; i++) {
      uint32_t saved = transfer[i];
      transfer[i] = 0;
      reject(&info, transfer, loops, &target, 6);
      transfer[i] = UINT32_MAX;
      reject(&info, transfer, loops, &target, 6);
      transfer[i] = saved;
   }
   transfer[2] = 2;
   reject(&info, transfer, loops, &target, 6);
   transfer[2] = 1;
   reject(&info, transfer, loops, &target, 0);
   reject(&info, transfer, loops, &target, UINT32_MAX);
   loops[0] = 7;
   reject(&info, transfer, loops, &target, 6);
   loops[0] = 2;
   loops[1] = 2;
   loops[3] = 1;
   reject(&info, transfer, loops, &target, 6);
   loops[1] = 0;
   loops[2] = 0;
   reject(&info, transfer, loops, &target, 6);
   loops[2] = 1;
   loops[3] = 0;
   struct pan_graph_matmul_cost_target other = target;
   other.alignment = 64;
   reject(&info, transfer, loops, &other, 6);
   for (unsigned i = 2; i < 6; i++) {
      for (unsigned j = 0; j < 2; j++) {
         other = target;
         other.memory[i].bandwidth[j] = NAN;
         reject(&info, transfer, loops, &other, 6);
         other.memory[i].bandwidth[j] = 0.0f;
         reject(&info, transfer, loops, &other, 6);
      }
      other = target;
      other.memory[i].parallel = 0;
      reject(&info, transfer, loops, &other, 6);
   }
   reject(NULL, transfer, loops, &target, 6);
   reject(&info, NULL, loops, &target, 6);
   reject(&info, transfer, NULL, &target, 6);
   reject(&info, transfer, loops, NULL, 6);
   assert(!pan_graph_matmul_cost_estimate(&info, transfer, loops,
                                         &target, 6, NULL));
   other = target;
   assert(!pan_graph_matmul_cost_target_v15_init(0, 64, 262144, &other));
   assert(!memcmp(&other, &target, sizeof(target)));
   assert(!pan_graph_matmul_cost_target_v15_init(8, 63, 262144, &other));
   assert(!memcmp(&other, &target, sizeof(target)));
   assert(!pan_graph_matmul_cost_target_v15_init(8, 64, 32, &other));
   assert(!memcmp(&other, &target, sizeof(target)));
   assert(!pan_graph_matmul_cost_target_v15_init(8, 64, 262144, NULL));
   assert(pan_graph_matmul_cost_target_v15_init(4, 128, 524288, &other));
   assert(other.memory[3].parallel == 4 && other.memory[3].line_size == 128 &&
          other.memory[3].size == 524288);
   puts("MatMul cost: full and partial reduction candidate vectors, resource and engine reports, "
        "runtime cache profile and transactional rejection passed");
   return 0;
}
