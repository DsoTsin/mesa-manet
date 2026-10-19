#ifndef PAN_GRAPH_PROGRAM_V15_H
#define PAN_GRAPH_PROGRAM_V15_H

#include "pan_graph_program_schedule.h"

struct pan_graph_program_bindings {
   const struct pan_graph_v15_memory *reads;
   const struct pan_graph_v15_memory *writes;
   uint32_t read_count;
   uint32_t write_count;
};

struct pan_graph_program_kernel {
   struct pan_graph_v15_program program;
   uint32_t extent[4];
   uint32_t loops[6];
};

enum pan_graph_tsu_status pan_graph_program_v15_compile_candidate(
   const struct pan_graph_program_cost_plan *plan,
   const struct pan_graph_program_selection *selection,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const struct pan_graph_program_bindings *bindings,
   const struct pan_graph_nrt_target *target,
   struct pan_graph_program_kernel *kernel);

#endif
