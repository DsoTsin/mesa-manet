#ifndef PAN_GRAPH_VECTOR_COST_H
#define PAN_GRAPH_VECTOR_COST_H

#include "pan_graph_memory_cost.h"

struct pan_graph_vector_cost_result {
   struct pan_graph_cost_report memory[3];
   struct pan_graph_cost_report engine[3];
   struct pan_graph_cost_report total;
};

bool pan_graph_vector_cost_estimate_program_add_i32(
   const struct pan_graph_memory_cost_target *target,
   const struct pan_graph_memory_cost_program *program,
   struct pan_graph_vector_cost_result *result);

bool pan_graph_vector_cost_estimate_add_i32(
   const struct pan_graph_memory_cost_target *target,
   const struct pan_graph_memory_cost_program *program,
   struct pan_graph_vector_cost_result *result);

#endif
