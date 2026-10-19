#ifndef PAN_GRAPH_MATMUL_SELECT_H
#define PAN_GRAPH_MATMUL_SELECT_H

#include "pan_graph_matmul_cost.h"
#include "pan_graph_matmul_schedule.h"
#include "pan_graph_matmul_search.h"

struct pan_graph_matmul_selection {
   struct pan_graph_matmul_candidate candidate;
   uint32_t loops[4];
   struct pan_graph_cost_report cost;
   size_t candidate_count;
};

enum pan_graph_tsu_status pan_graph_matmul_parallel_candidates(
   const struct pan_graph_matmul_info *info,
   const uint32_t initial[PAN_GRAPH_MATMUL_RANK],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   bool exhaustive, const struct pan_graph_matmul_parallel_options *options,
   struct pan_graph_matmul_candidate_list *candidates);

enum pan_graph_tsu_status pan_graph_matmul_program_parallel_candidates(
   const struct pan_graph_matmul_info *info,
   const uint32_t initial[PAN_GRAPH_MATMUL_RANK],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   bool exhaustive, const struct pan_graph_matmul_parallel_options *options,
   const uint32_t granule[PAN_GRAPH_MATMUL_RANK], uint32_t vector_count,
   const struct pan_graph_matmul_candidate_list *seed,
   struct pan_graph_matmul_candidate_list *candidates);

enum pan_graph_tsu_status pan_graph_matmul_select_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_nrt_target *target,
   const struct pan_graph_matmul_cost_target *cost_target, uint32_t core_budget,
   struct pan_graph_matmul_selection *selection);

#endif
