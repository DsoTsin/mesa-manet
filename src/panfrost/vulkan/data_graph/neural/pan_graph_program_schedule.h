#ifndef PAN_GRAPH_PROGRAM_SCHEDULE_H
#define PAN_GRAPH_PROGRAM_SCHEDULE_H

#include "pan_graph_program_cost.h"

struct pan_graph_program_selection {
   struct pan_graph_program_cost_result cost;
   uint32_t transfer[PAN_GRAPH_MATMUL_RANK];
   uint32_t granule[PAN_GRAPH_MATMUL_RANK];
   uint32_t loops[4];
   size_t candidate_count;
   uint32_t schedule_type;
};

struct pan_graph_scheduled_program {
   struct pan_graph_program_cost_plan plan;
   struct pan_graph_program_selection selection;
   const uint64_t *groups;
};

struct pan_graph_program_schedule {
   struct pan_graph_scheduled_program *programs;
   uint64_t *groups;
   size_t count;
   size_t candidate_count;
   uint64_t cycles;
   uint32_t group_count;
};

enum pan_graph_tsu_status pan_graph_program_schedule_v15(
   struct pan_graph_program_cost_plan *plan,
   const struct pan_graph_nrt_target *target,
   const struct pan_graph_memory_cost_target *cost_target, uint32_t core_budget,
   struct pan_graph_program_selection *selection);

enum pan_graph_tsu_status pan_graph_program_schedule_matmul_add_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const struct pan_graph_nrt_target *target,
   const struct pan_graph_memory_cost_target *cost_target, uint32_t core_budget,
   uint32_t io_limit, struct pan_graph_program_schedule *schedule);

void
pan_graph_program_schedule_finish(struct pan_graph_program_schedule *schedule);

#endif
