#ifndef PAN_GRAPH_CONV_V15_H
#define PAN_GRAPH_CONV_V15_H

#include "pan_graph_conv.h"
#include "pan_graph_matmul_v15.h"

struct pan_graph_conv_rescale {
   struct pan_graph_v15_rescale parameters;
   struct pan_graph_v15_memory coefficients;
   struct pan_graph_v15_memory table;
   bool has_table;
};

bool pan_graph_conv_v15_assign_loops(
   const struct pan_graph_conv_plan *plan, const uint32_t transfer[7],
   uint32_t core_budget, uint32_t loops[4]);

bool pan_graph_conv_v15_ce_cycles(
   const struct pan_graph_conv_plan *plan, const uint32_t transfer[7],
   const uint32_t loops[4], uint64_t *cycles);

enum pan_graph_tsu_status pan_graph_conv_v15_compile_candidate(
   const struct pan_graph_conv_plan *plan,
   const struct pan_graph_nrt_target *target, const uint32_t transfer[7],
   const uint32_t loops[4], const struct pan_graph_matmul_memory *memory,
   struct pan_graph_matmul_kernel *kernel);

enum pan_graph_tsu_status pan_graph_conv_v15_compile_rescale_candidate(
   const struct pan_graph_conv_plan *plan,
   const struct pan_graph_nrt_target *target, const uint32_t transfer[7],
   const uint32_t loops[4], const struct pan_graph_matmul_memory *memory,
   const struct pan_graph_conv_rescale *rescale,
   struct pan_graph_matmul_kernel *kernel);

#endif
