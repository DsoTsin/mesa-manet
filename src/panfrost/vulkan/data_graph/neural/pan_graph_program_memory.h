#ifndef PAN_GRAPH_PROGRAM_MEMORY_H
#define PAN_GRAPH_PROGRAM_MEMORY_H

#include "pan_graph_memory_lifetime.h"
#include "pan_graph_memory_resource.h"
#include "pan_graph_program_schedule.h"
#include "pan_graph_program_view.h"
#include "pan_graph_program_v15.h"

struct pan_graph_program_memory_binding {
   uint32_t binding[3];
   uint32_t ordinal;
};

struct pan_graph_program_memory {
   struct pan_graph_program_views *views;
   struct pan_graph_memory_resources resources;
   struct pan_graph_memory_lifetimes lifetimes;
   struct pan_graph_program_bindings *program_bindings;
   struct pan_graph_v15_memory *memories;
   size_t program_count;
};

enum pan_graph_tsu_status pan_graph_program_memory_matmul_add_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const struct pan_graph_program_schedule *schedule,
   const struct pan_graph_memory_cost_target *cost_target,
   const struct pan_graph_program_memory_binding bindings[3],
   struct pan_graph_program_memory *result);

void pan_graph_program_memory_finish(struct pan_graph_program_memory *memory);

#endif
