#include "pan_graph_matmul_cost.h"
#include "pan_graph_memory_cost.h"

#include <limits.h>
#include <math.h>
#include <string.h>

static float
multiply(float a, float b)
{
   volatile float result = a * b;
   return result;
}

static uint64_t
divide_up(uint64_t a, uint64_t b)
{
   return a / b + (a % b != 0);
}

static bool
integer(float value, uint64_t *result)
{
   if (!isfinite(value) || value < 0.0f || value >= 0x1p63f)
      return false;
   *result = (uint64_t)value;
   return true;
}

static uint64_t
maximum(uint64_t a, uint64_t b)
{
   return a > b ? a : b;
}

bool
pan_graph_matmul_cost_target_v15_init(
   uint32_t l2_slices, uint32_t l2_line_size, uint32_t l2_size_per_slice,
   struct pan_graph_matmul_cost_target *target)
{
   if (!target || !l2_slices || !l2_line_size || !l2_size_per_slice ||
       (l2_line_size & (l2_line_size - 1)) ||
       l2_size_per_slice < l2_line_size)
      return false;
   struct pan_graph_matmul_cost_target value = {
      .memory = {
         [2] = {.bandwidth = {32.0f, 16.0f}, .latency = {12, 12},
                .parallel = 1, .line_size = 64, .size = 32768},
         [3] = {.bandwidth = {32.0f, 32.0f}, .latency = {60, 60},
                .parallel = l2_slices, .line_size = l2_line_size,
                .size = l2_size_per_slice},
         [4] = {.bandwidth = {32.0f, 32.0f}, .latency = {100, 100},
                .parallel = 1, .line_size = 64, .size = 8388608},
         [5] = {.latency = {170, 100}, .parallel = 1,
                .line_size = 64, .size = 268435456},
      },
      .alignment = 32,
   };
   float dram = multiply(multiply(multiply(68.0f, 1024.0f), 1.024f), 1.024f);
   dram /= multiply(350.0f, 2.0f);
   value.memory[5].bandwidth[0] = dram;
   value.memory[5].bandwidth[1] = dram;
   *target = value;
   return true;
}

static void
critical(struct pan_graph_cost_report *report)
{
   for (unsigned i = 0; i < 8; i++) {
      if (report->engine_cycles[i] > report->critical_cycles) {
         report->critical_cycles = report->engine_cycles[i];
         report->critical_kind = i;
      }
   }
}

static unsigned
schedule(const uint32_t transfer[7])
{
   static const unsigned order[4] = {0, 2, 1, 3};
   static const unsigned blocks[4][3] = {
      {32, 2, 2}, {8, 4, 4}, {32, 4, 1}, {8, 16, 1},
   };
   uint64_t best_count = UINT64_MAX;
   bool best_small = true;
   float best_waste = 1.0f;
   unsigned best = 0;
   uint64_t volume = (uint64_t)transfer[0] * transfer[3] * transfer[2];
   for (unsigned i = 0; i < 4; i++) {
      unsigned type = order[i];
      uint64_t count = divide_up(transfer[0], blocks[type][0]) *
                       divide_up(transfer[3], blocks[type][1]) *
                       divide_up(transfer[2], blocks[type][2]);
      bool small = type == 1 || type == 3;
      float waste = 1.0f - (float)volume / (float)(count * 128);
      if (count < best_count ||
          (count == best_count &&
           (small < best_small ||
            (small == best_small && waste < best_waste)))) {
         best = type;
         best_count = count;
         best_small = small;
         best_waste = waste;
      }
   }
   return best;
}

bool
pan_graph_matmul_cost_estimate(
   const struct pan_graph_matmul_info *info, const uint32_t transfer[7],
   const uint32_t loops[4], const struct pan_graph_matmul_cost_target *target,
   uint32_t parallel_budget, struct pan_graph_matmul_cost_result *result)
{
   struct pan_graph_matmul_plan plan;
   if (!info || !transfer || !loops || !target || !result ||
       !pan_graph_matmul_i8_init(&plan, info) || !parallel_budget ||
       parallel_budget > INT32_MAX || target->alignment != 32 ||
       transfer[2] != 1 || transfer[1] != 1 ||
       transfer[5] != 1 || transfer[6] != 1 ||
       (loops[3] && (!loops[2] || loops[0] == loops[1])) ||
       (loops[2] && (loops[0] >= 4 || loops[2] > INT32_MAX)) ||
       (loops[3] && (loops[1] >= 4 || loops[3] > INT32_MAX)))
      return false;
   for (unsigned i = 2; i < 6; i++) {
      if (!target->memory[i].parallel || !target->memory[i].line_size ||
          !target->memory[i].latency[0])
         return false;
      for (unsigned j = 0; j < 2; j++) {
         if (!isfinite(target->memory[i].bandwidth[j]) ||
             target->memory[i].bandwidth[j] <= 0.0f)
            return false;
      }
   }
   uint32_t full[7];
   for (unsigned i = 0; i < 7; i++) {
      full[i] = plan.transfer_high[i] + 1;
      if (i == 2)
         full[i] = info->batch;
      if (!transfer[i] || transfer[i] > full[i] || transfer[i] > 4095)
         return false;
   }
   unsigned a = loops[0], b = loops[1];
   struct pan_graph_memory_cost_target memory_target = {
      .descriptor_bytes = {36, 64, 32, 12},
      .section_bytes = {0, 8, 20, 32, 8, 8, 8, 20},
      .alignment = target->alignment,
      .issue_width = 32,
      .parallel_budget = parallel_budget,
   };
   for (unsigned level = 0; level < 6; level++) {
      const struct pan_graph_matmul_cost_memory *memory = &target->memory[level];
      memory_target.memory[level] = (struct pan_graph_memory_cost_resource){
         .bandwidth = {memory->bandwidth[0], memory->bandwidth[1]},
         .latency = {memory->latency[0], memory->latency[1]},
         .parallel = memory->parallel,
         .line_size = memory->line_size,
         .size = memory->size,
      };
   }
   const struct pan_graph_memory_cost_section sections[6] = {
      {.type = PAN_GRAPH_V15_MEMORY_INPUT, .bits = 8, .replication_rank = 5,
       .full = {info->columns, info->inner, info->batch},
       .transfer = {transfer[0], transfer[4], 1}},
      {.type = PAN_GRAPH_V15_MEMORY_INPUT, .bits = 8, .replication_rank = 5,
       .full = {info->inner, info->rows, info->batch},
       .transfer = {transfer[4], transfer[3], 1}},
      {.type = PAN_GRAPH_V15_TRANSFORM},
      {.type = PAN_GRAPH_V15_CE},
      {.type = PAN_GRAPH_V15_VE},
      {.type = PAN_GRAPH_V15_MEMORY_OUTPUT, .bits = 32, .replication_rank = 4,
       .full = {info->columns, info->rows, info->batch},
       .transfer = {transfer[0], transfer[3], 1}},
   };
   const enum pan_graph_memory_cost_view views[2] = {
      PAN_GRAPH_MEMORY_COST_TENSOR_VIEW, PAN_GRAPH_MEMORY_COST_WEIGHT_VIEW,
   };
   struct pan_graph_memory_cost_program program = {
      .sections = sections, .section_count = 6,
      .input_views = views, .input_view_count = 2,
      .output_view_count = 1, .resource_count = 5,
      .workload = {.rank = 7, .axis = {a, b},
                   .step = {loops[2], loops[3]}},
   };
   program.workload.repeats = 1;
   program.workload.latency = target->memory[5].latency[0];
   program.workload.minimum = 0.5f;
   for (unsigned i = 0; i < 7; i++) {
      program.workload.extent[i] = full[i];
      program.workload.transfer[i] = transfer[i];
   }
   const enum pan_graph_cost_ve_i32_op vector_op = PAN_GRAPH_COST_VE_I32_IDENTITY;
   return pan_graph_matmul_cost_estimate_program(
      info, &memory_target, &program, &vector_op, 1, result);
}

bool
pan_graph_matmul_cost_estimate_program(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_memory_cost_target *target,
   const struct pan_graph_memory_cost_program *program,
   const enum pan_graph_cost_ve_i32_op *vector_ops, uint32_t vector_count,
   struct pan_graph_matmul_cost_result *result)
{
   struct pan_graph_matmul_plan plan;
   if (!info || !target || !program || !result || !vector_ops ||
       !vector_count || vector_count > 8 || target->issue_width != 32 ||
       !pan_graph_matmul_i8_init(&plan, info) ||
       program->workload.rank != 7 || !program->sections ||
       program->input_view_count != 2 || !program->input_views ||
       program->input_views[0] != PAN_GRAPH_MEMORY_COST_TENSOR_VIEW ||
       program->input_views[1] != PAN_GRAPH_MEMORY_COST_WEIGHT_VIEW ||
       !program->output_view_count || program->output_view_count > 15 ||
       program->resource_count != 4 + vector_count ||
       program->section_count != 4 + vector_count + program->output_view_count)
      return false;
   uint32_t transfer[7], full[7];
   const struct pan_graph_cost_workload *workload = &program->workload;
   for (unsigned i = 0; i < 7; i++) {
      full[i] = i == 2 ? info->batch : plan.transfer_high[i] + 1;
      if (workload->extent[i] != full[i] ||
          !workload->transfer[i] || workload->transfer[i] > full[i] ||
          workload->transfer[i] > 4095)
         return false;
      transfer[i] = workload->transfer[i];
   }
   const uint32_t sources[3][3] = {
      {info->columns, info->inner, info->batch},
      {info->inner, info->rows, info->batch},
      {info->columns, info->rows, info->batch},
   };
   const uint32_t slices[3][3] = {
      {transfer[0], transfer[4], 1},
      {transfer[4], transfer[3], 1},
      {transfer[0], transfer[3], 1},
   };
   for (unsigned i = 0; i < program->section_count; i++) {
      enum pan_graph_v15_section_type type =
         i < 2 ? PAN_GRAPH_V15_MEMORY_INPUT :
         i == 2 ? PAN_GRAPH_V15_TRANSFORM :
         i == 3 ? PAN_GRAPH_V15_CE :
         i < 4 + vector_count ? PAN_GRAPH_V15_VE :
         PAN_GRAPH_V15_MEMORY_OUTPUT;
      const struct pan_graph_memory_cost_section *section = &program->sections[i];
      if (section->type != type)
         return false;
      if (type != PAN_GRAPH_V15_MEMORY_INPUT &&
          type != PAN_GRAPH_V15_MEMORY_OUTPUT)
         continue;
      unsigned role = i < 2 ? i : 2;
      if (memcmp(section->full, sources[role], sizeof(section->full)) ||
          memcmp(section->transfer, slices[role], sizeof(section->transfer)))
         return false;
   }
   struct pan_graph_matmul_cost_result value = {0};
   if (!pan_graph_memory_cost_estimate(target, program, value.memory))
      return false;
   float first[7];
   uint32_t first_extent[4];
   for (unsigned i = 0; i < 7; i++) {
      uint32_t extent = full[i];
      for (unsigned loop = 0; loop < 2; loop++) {
         if (workload->step[loop] && workload->axis[loop] == i &&
             extent > workload->step[loop])
            extent = workload->step[loop];
      }
      first[i] = (float)extent / (float)transfer[i];
      if (i < 4)
         first_extent[i] = extent;
   }
   static const unsigned blocks[4][3] = {
      {32, 2, 2}, {8, 4, 4}, {32, 4, 1}, {8, 16, 1},
   };
   const unsigned *block = blocks[schedule(transfer)];
   uint64_t tn = divide_up(transfer[0], block[0]);
   uint64_t tm = divide_up(transfer[3], block[1]);
   uint64_t tb = divide_up(transfer[2], block[2]);
   uint64_t ce = maximum(maximum(8 * tm * tb * tn, 2 * block[0] * tn),
                          divide_up(transfer[3], 4) * 8 * transfer[2]) *
                 divide_up(transfer[4], 64);
   float replication = 1.0f;
   for (unsigned i = 0; i < 7; i++)
      replication = multiply(replication, first[i]);
   if (!integer(ceilf(multiply(replication, (float)ce)),
                &value.engine[0].engine_cycles[2]) ||
       !integer(multiply(replication, divide_up(transfer[4], 8) * transfer[0]),
                &value.engine[0].engine_cycles[4]))
      return false;
   for (unsigned i = 0; i < vector_count; i++) {
      uint64_t cycles;
      if (!pan_graph_cost_v15_ve_i32(vector_ops[i], first_extent, transfer,
                                     &cycles) ||
          cycles > INT64_MAX - value.engine[0].engine_cycles[3])
         return false;
      value.engine[0].engine_cycles[3] += cycles;
   }
   critical(&value.engine[0]);
   value.engine[1].critical_kind = value.engine[0].critical_kind;
   if (!pan_graph_cost_combine(value.memory, value.engine, &value.total))
      return false;
   if (!pan_graph_cost_apply_workload(workload, value.total.cycles,
                                      &value.total.cycles))
      return false;
   *result = value;
   return true;
}
