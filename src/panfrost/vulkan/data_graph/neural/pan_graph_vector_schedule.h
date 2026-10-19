#ifndef PAN_GRAPH_VECTOR_SCHEDULE_H
#define PAN_GRAPH_VECTOR_SCHEDULE_H

#include <stddef.h>

#include "pan_graph_nrt.h"
#include "pan_graph_tsu.h"

struct pan_graph_vector_schedule_info {
   uint32_t full[3];
   uint32_t granule[3];
   uint32_t read_count;
   uint32_t vector_count;
   uint32_t export_count;
   uint32_t write_count;
};

struct pan_graph_vector_candidate {
   uint32_t transfer[3];
};

struct pan_graph_vector_candidate_list {
   size_t count;
   struct pan_graph_vector_candidate *candidates;
};

enum pan_graph_tsu_status pan_graph_vector_base_candidates(
   const struct pan_graph_vector_schedule_info *info, const uint32_t caps[3],
   const struct pan_graph_nrt_target *target,
   struct pan_graph_vector_candidate_list *candidates);

enum pan_graph_tsu_status pan_graph_vector_parallel_candidates(
   const struct pan_graph_vector_schedule_info *info,
   const struct pan_graph_nrt_target *target, uint32_t core_budget,
   struct pan_graph_vector_candidate_list *candidates);

bool pan_graph_vector_assign_loops(
   const struct pan_graph_vector_schedule_info *info,
   const uint32_t transfer[3], uint32_t core_budget, uint32_t loops[4]);

void pan_graph_vector_candidates_finish(
   struct pan_graph_vector_candidate_list *candidates);

#endif
