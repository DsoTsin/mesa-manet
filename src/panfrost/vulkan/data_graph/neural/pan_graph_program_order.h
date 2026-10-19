#ifndef PAN_GRAPH_PROGRAM_ORDER_H
#define PAN_GRAPH_PROGRAM_ORDER_H

#include "pan_graph_tsu.h"

struct pan_graph_program_order_node {
   uint64_t uid;
   const uint64_t *dependencies;
   size_t dependency_count;
};

struct pan_graph_program_order {
   uint32_t *indices;
   size_t count;
};

enum pan_graph_tsu_status pan_graph_program_order_build_dependencies(
   const struct pan_graph_program_order_node *nodes, size_t count,
   struct pan_graph_program_order *result);

enum pan_graph_tsu_status pan_graph_program_order_build_selected(
   const struct pan_graph_program_order_node *nodes, size_t count,
   struct pan_graph_program_order *result);

void pan_graph_program_order_finish(struct pan_graph_program_order *result);

#endif
