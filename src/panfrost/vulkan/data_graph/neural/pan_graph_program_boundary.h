#ifndef PAN_GRAPH_PROGRAM_BOUNDARY_H
#define PAN_GRAPH_PROGRAM_BOUNDARY_H

#include "pan_graph_tsu.h"

enum pan_graph_program_boundary_kind {
   PAN_GRAPH_PROGRAM_BOUNDARY_WEIGHT,
   PAN_GRAPH_PROGRAM_BOUNDARY_CE,
   PAN_GRAPH_PROGRAM_BOUNDARY_ADD,
};

struct pan_graph_program_boundary_view {
   uint32_t tensor;
   uint32_t key;
   uint32_t origin;
};

struct pan_graph_program_boundary_group {
   enum pan_graph_program_boundary_kind kind;
   struct pan_graph_program_boundary_view inputs[2];
   struct pan_graph_program_boundary_view output;
};

struct pan_graph_program_boundary_input {
   const uint32_t *tensor_kinds;
   const struct pan_graph_program_boundary_group *groups;
   const uint64_t *selected;
   uint32_t tensor_count;
   uint32_t group_count;
};

struct pan_graph_program_boundary {
   struct pan_graph_program_boundary_view *inputs;
   struct pan_graph_program_boundary_view *outputs;
   size_t input_count;
   size_t output_count;
   uint32_t root;
};

enum pan_graph_tsu_status pan_graph_program_boundary_matmul_add_v15(
   const struct pan_graph_program_boundary_input *input,
   struct pan_graph_program_boundary *result);

void pan_graph_program_boundary_finish(struct pan_graph_program_boundary *result);

#endif
