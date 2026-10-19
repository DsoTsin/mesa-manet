#ifndef PAN_GRAPH_MATMUL_CANDIDATE_H
#define PAN_GRAPH_MATMUL_CANDIDATE_H

#include "pan_graph_matmul.h"
#include "pan_graph_nrt.h"

struct pan_graph_matmul_candidate {
   uint32_t transfer[PAN_GRAPH_MATMUL_RANK];
   uint32_t schedule_type;
};

struct pan_graph_matmul_candidate_set {
   struct pan_graph_matmul_candidate candidates[2];
   uint32_t count;
   bool root_flag;
};

bool pan_graph_matmul_candidate_filter(
   const struct pan_graph_matmul_info *info,
   const uint32_t proposed[PAN_GRAPH_MATMUL_RANK],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   uint32_t column_divisor, struct pan_graph_matmul_candidate_set *candidates);

bool pan_graph_matmul_program_candidate_filter(
   const struct pan_graph_matmul_info *info,
   const uint32_t proposed[PAN_GRAPH_MATMUL_RANK],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   uint32_t column_divisor, uint32_t vector_count,
   struct pan_graph_matmul_candidate_set *candidates);

#endif
