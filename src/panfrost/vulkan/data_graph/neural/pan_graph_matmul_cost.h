#ifndef PAN_GRAPH_MATMUL_COST_H
#define PAN_GRAPH_MATMUL_COST_H

#include "pan_graph_cost.h"
#include "pan_graph_matmul.h"

struct pan_graph_matmul_cost_memory {
   float bandwidth[2];
   uint32_t latency[2];
   uint32_t parallel;
   uint32_t line_size;
   uint64_t size;
};

struct pan_graph_matmul_cost_target {
   struct pan_graph_matmul_cost_memory memory[6];
   uint32_t alignment;
};

struct pan_graph_matmul_cost_result {
   struct pan_graph_cost_report memory[3];
   struct pan_graph_cost_report engine[3];
   struct pan_graph_cost_report total;
};

struct pan_graph_memory_cost_target;
struct pan_graph_memory_cost_program;

bool pan_graph_matmul_cost_estimate_program(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_memory_cost_target *target,
   const struct pan_graph_memory_cost_program *program,
   const enum pan_graph_cost_ve_i32_op *vector_ops, uint32_t vector_count,
   struct pan_graph_matmul_cost_result *result);

bool pan_graph_matmul_cost_target_v15_init(
   uint32_t l2_slices, uint32_t l2_line_size, uint32_t l2_size_per_slice,
   struct pan_graph_matmul_cost_target *target);

bool pan_graph_matmul_cost_estimate(
   const struct pan_graph_matmul_info *info,
   const uint32_t transfer[PAN_GRAPH_MATMUL_RANK], const uint32_t loops[4],
   const struct pan_graph_matmul_cost_target *target, uint32_t parallel_budget,
   struct pan_graph_matmul_cost_result *result);

#endif
