#ifndef PAN_GRAPH_MATMUL_SEARCH_H
#define PAN_GRAPH_MATMUL_SEARCH_H

#include "pan_graph_matmul_candidate.h"
#include "pan_graph_tsu.h"

struct pan_graph_matmul_candidate_list {
   size_t count;
   struct pan_graph_matmul_candidate *candidates;
};

enum pan_graph_tsu_status pan_graph_matmul_base_candidates(
   const struct pan_graph_matmul_info *info,
   const uint32_t caps[PAN_GRAPH_MATMUL_RANK],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   uint32_t column_divisor, bool exhaustive,
   struct pan_graph_matmul_candidate_list *candidates);

enum pan_graph_tsu_status pan_graph_matmul_program_base_candidates(
   const struct pan_graph_matmul_info *info,
   const uint32_t caps[PAN_GRAPH_MATMUL_RANK],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   uint32_t column_divisor, bool exhaustive, uint32_t vector_count,
   struct pan_graph_matmul_candidate_list *candidates);

void pan_graph_matmul_candidate_list_finish(
   struct pan_graph_matmul_candidate_list *candidates);

#endif
