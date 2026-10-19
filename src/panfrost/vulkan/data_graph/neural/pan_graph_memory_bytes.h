#ifndef PAN_GRAPH_MEMORY_BYTES_H
#define PAN_GRAPH_MEMORY_BYTES_H

#include "pan_graph_memory_layout.h"

struct pan_graph_memory_bytes_result {
   uint32_t padded[3];
   uint64_t local_bytes;
   uint64_t total_bytes;
   uint64_t cache_bytes;
};

bool pan_graph_memory_bytes_estimate(
   const struct pan_graph_memory_layout_target *target,
   const struct pan_graph_memory_layout_input *input, uint8_t level,
   struct pan_graph_memory_bytes_result *result);

#endif
