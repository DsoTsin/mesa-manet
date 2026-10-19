#ifndef PAN_GRAPH_PROGRAM_COST_H
#define PAN_GRAPH_PROGRAM_COST_H

#include "pan_graph_matmul_v15.h"
#include "pan_graph_memory_cost.h"
#include "pan_graph_program_boundary.h"

struct pan_graph_program_cost_plan {
   struct pan_graph_matmul_info info;
   struct pan_graph_program_boundary boundary;
   struct pan_graph_memory_cost_program program;
   struct pan_graph_memory_cost_section *sections;
   enum pan_graph_memory_cost_view *input_views;
   enum pan_graph_cost_ve_i32_op *vector_ops;
   uint32_t *read_tensors;
   uint32_t *vector_groups;
   uint32_t read_count;
   uint32_t vector_count;
};

struct pan_graph_program_cost_result {
   struct pan_graph_cost_report memory[3];
   struct pan_graph_cost_report engine[3];
   struct pan_graph_cost_report total;
   size_t input_view_count;
   size_t output_view_count;
   uint32_t rank;
   uint32_t read_count;
   uint32_t vector_count;
   uint32_t resource_count;
};

bool pan_graph_program_cost_target_v15_init(
   uint32_t l2_slices, uint32_t l2_line_size, uint32_t l2_size_per_slice,
   uint32_t core_budget, struct pan_graph_memory_cost_target *target);

enum pan_graph_tsu_status pan_graph_program_cost_plan_matmul_add_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const uint64_t *selected, struct pan_graph_program_cost_plan *plan);

enum pan_graph_tsu_status pan_graph_program_cost_plan_estimate(
   struct pan_graph_program_cost_plan *plan,
   const uint32_t transfer[PAN_GRAPH_MATMUL_RANK], const uint32_t loops[4],
   const struct pan_graph_memory_cost_target *target,
   struct pan_graph_program_cost_result *result);

void
pan_graph_program_cost_plan_finish(struct pan_graph_program_cost_plan *plan);

enum pan_graph_tsu_status pan_graph_program_cost_matmul_add_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const uint64_t *selected, const uint32_t transfer[7],
   const uint32_t loops[4], const struct pan_graph_memory_cost_target *target,
   struct pan_graph_program_cost_result *result);

#endif
