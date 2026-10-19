#ifndef PAN_GRAPH_MATMUL_V15_H
#define PAN_GRAPH_MATMUL_V15_H

#include "pan_graph_matmul.h"
#include "pan_graph_nrt.h"
#include "pan_graph_v15.h"

struct pan_graph_matmul_kernel {
   struct pan_graph_v15_program program;
   uint32_t extent[4];
   uint32_t loops[6];
};

struct pan_graph_matmul_add {
   uint32_t input[2];
};

struct pan_graph_matmul_epilogue {
   const struct pan_graph_matmul_add *adds;
   uint32_t count;
   uint32_t output_view_count;
};

struct pan_graph_matmul_memory {
   struct pan_graph_v15_memory inputs[2];
   struct pan_graph_v15_memory output;
};

enum pan_graph_tsu_status pan_graph_matmul_v15_compile_bound_program_candidate(
   const struct pan_graph_matmul_plan *plan,
   const struct pan_graph_nrt_target *target,
   const uint32_t transfer[PAN_GRAPH_MATMUL_RANK], const uint32_t loops[4],
   const struct pan_graph_matmul_epilogue *epilogue,
   const struct pan_graph_matmul_memory *memory,
   struct pan_graph_matmul_kernel *kernel);

enum pan_graph_tsu_status pan_graph_matmul_v15_compile_program_candidate(
   const struct pan_graph_matmul_plan *plan,
   const struct pan_graph_nrt_target *target,
   const uint32_t transfer[PAN_GRAPH_MATMUL_RANK], const uint32_t loops[4],
   const struct pan_graph_matmul_epilogue *epilogue,
   struct pan_graph_matmul_kernel *kernel);

enum pan_graph_tsu_status
pan_graph_matmul_v15_compile(const struct pan_graph_matmul_plan *plan,
                             const struct pan_graph_nrt_target *target,
                             struct pan_graph_matmul_kernel *kernel);

enum pan_graph_tsu_status pan_graph_matmul_v15_compile_candidate(
   const struct pan_graph_matmul_plan *plan,
   const struct pan_graph_nrt_target *target,
   const uint32_t transfer[PAN_GRAPH_MATMUL_RANK], const uint32_t loops[4],
   struct pan_graph_matmul_kernel *kernel);

#endif
