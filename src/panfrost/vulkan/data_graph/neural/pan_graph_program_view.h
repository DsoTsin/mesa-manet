#ifndef PAN_GRAPH_PROGRAM_VIEW_H
#define PAN_GRAPH_PROGRAM_VIEW_H

#include "pan_graph_matmul_v15.h"

#define PAN_GRAPH_PROGRAM_VIEW_MAX_RANK 7

struct pan_graph_program_view_bounds {
   uint32_t rank;
   int32_t low[PAN_GRAPH_PROGRAM_VIEW_MAX_RANK];
   int32_t high[PAN_GRAPH_PROGRAM_VIEW_MAX_RANK];
};

struct pan_graph_program_view_origin {
   uint32_t group;
   uint32_t index;
   bool output;
};

struct pan_graph_program_view {
   uint32_t tensor;
   uint32_t key;
   uint32_t kind;
   struct pan_graph_program_view_origin origin;
   struct pan_graph_program_view_bounds bounds;
};

struct pan_graph_program_views {
   struct pan_graph_program_view *inputs;
   struct pan_graph_program_view *outputs;
   size_t input_count;
   size_t output_count;
   uint32_t root;
};

enum pan_graph_tsu_status pan_graph_program_view_bounds_init(
   uint32_t rank, const uint32_t *shape,
   const struct pan_graph_program_view_bounds *explicit_bounds,
   struct pan_graph_program_view_bounds *result);

enum pan_graph_tsu_status pan_graph_program_view_matmul_add_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const uint64_t *selected, struct pan_graph_program_views *result);

void pan_graph_program_views_finish(struct pan_graph_program_views *result);

#endif
