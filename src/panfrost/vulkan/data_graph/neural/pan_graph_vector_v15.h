#ifndef PAN_GRAPH_VECTOR_V15_H
#define PAN_GRAPH_VECTOR_V15_H

#include "pan_graph_nrt.h"
#include "pan_graph_v15.h"

struct pan_graph_vector_add {
   uint32_t input[2];
};

struct pan_graph_vector_write {
   uint32_t value;
   struct pan_graph_v15_memory memory;
};

struct pan_graph_vector_program {
   uint32_t full[3];
   uint32_t transfer[3];
   uint32_t loops[4];
   const struct pan_graph_v15_memory *reads;
   const struct pan_graph_vector_add *adds;
   const struct pan_graph_vector_write *writes;
   uint32_t read_count;
   uint32_t add_count;
   uint32_t write_count;
};

struct pan_graph_vector_kernel {
   struct pan_graph_v15_program program;
   uint32_t extent[4];
   uint32_t loops[6];
};

enum pan_graph_tsu_status pan_graph_vector_v15_compile_candidate(
   const struct pan_graph_vector_program *info,
   const struct pan_graph_nrt_target *target,
   struct pan_graph_vector_kernel *kernel);

#endif
