#ifndef PAN_GRAPH_MEMORY_LIFETIME_H
#define PAN_GRAPH_MEMORY_LIFETIME_H

#include "pan_graph_alloc.h"

enum pan_graph_memory_lifetime_status {
   PAN_GRAPH_MEMORY_LIFETIME_SUCCESS,
   PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED,
   PAN_GRAPH_MEMORY_LIFETIME_NO_MEMORY,
};

struct pan_graph_memory_lifetime_tensor {
   uint64_t uid;
   uint64_t storage_size;
   uint32_t kind;
};

struct pan_graph_memory_lifetime_program {
   uint32_t outer_group;
   uint32_t first_reference;
   uint32_t reference_count;
};

struct pan_graph_memory_lifetime_input {
   const struct pan_graph_memory_lifetime_tensor *tensors;
   const struct pan_graph_memory_lifetime_program *programs;
   const uint32_t *references;
   uint32_t tensor_count;
   uint32_t program_count;
   uint32_t reference_count;
   uint32_t outer_group_count;
};

struct pan_graph_memory_lifetime_group {
   uint64_t uid;
   uint32_t first_identity;
   uint32_t alias_count;
};

struct pan_graph_memory_lifetime_identity {
   uint64_t offset;
   uint32_t group;
};

struct pan_graph_memory_lifetimes {
   struct pan_graph_memory_lifetime_group *groups;
   struct pan_graph_alloc_request *requests;
   struct pan_graph_allocation *allocations;
   struct pan_graph_memory_lifetime_identity *identities;
   struct pan_graph_alloc_stats stats;
   uint32_t group_count;
   uint32_t tensor_count;
   uint32_t end_clock;
};

enum pan_graph_memory_lifetime_status pan_graph_memory_lifetimes_create(
   const struct pan_graph_memory_lifetime_input *input,
   struct pan_graph_memory_lifetimes *result);

void pan_graph_memory_lifetimes_cleanup(struct pan_graph_memory_lifetimes *lifetimes);

#endif
