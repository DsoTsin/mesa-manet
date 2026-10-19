#ifndef PAN_GRAPH_PROGRAM_EXPAND_H
#define PAN_GRAPH_PROGRAM_EXPAND_H

#include "pan_graph_matmul_v15.h"

struct pan_graph_program_subsets {
   uint64_t *groups;
   size_t count;
   uint32_t group_count;
};

enum pan_graph_tsu_status pan_graph_program_expand_matmul_add_v15(
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   uint32_t io_limit, struct pan_graph_program_subsets *result);

enum pan_graph_tsu_status pan_graph_program_candidates_matmul_add_v15(
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   uint32_t io_limit, struct pan_graph_program_subsets *result);

enum pan_graph_tsu_status pan_graph_program_unique_candidates_matmul_add_v15(
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   uint32_t io_limit, struct pan_graph_program_subsets *result);

void pan_graph_program_subsets_finish(struct pan_graph_program_subsets *result);

#endif
