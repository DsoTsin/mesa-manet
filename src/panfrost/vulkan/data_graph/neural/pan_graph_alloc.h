#ifndef PAN_GRAPH_ALLOC_H
#define PAN_GRAPH_ALLOC_H

#include <stdint.h>

struct pan_graph_alloc_request {
   uint32_t first;
   uint32_t last;
   uint64_t size;
};

struct pan_graph_allocation {
   uint64_t offset;
   uint32_t position;
   int32_t predecessor;
};

struct pan_graph_alloc_stats {
   uint64_t peak;
   uint64_t lower_bound;
   uint32_t iterations;
};

enum pan_graph_alloc_status {
   PAN_GRAPH_ALLOC_SUCCESS,
   PAN_GRAPH_ALLOC_INVALID,
   PAN_GRAPH_ALLOC_NO_MEMORY,
};

enum pan_graph_alloc_status pan_graph_allocate(
   const struct pan_graph_alloc_request *requests, uint32_t count,
   uint64_t alignment, uint32_t max_trials,
   struct pan_graph_allocation *allocations, struct pan_graph_alloc_stats *stats);

#endif
