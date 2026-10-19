#ifndef PAN_GRAPH_MEMORY_RESOURCE_H
#define PAN_GRAPH_MEMORY_RESOURCE_H

#include "pan_graph_memory_descriptor.h"

enum pan_graph_memory_resource_domain {
   PAN_GRAPH_MEMORY_RESOURCE_INPUT,
   PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT,
   PAN_GRAPH_MEMORY_RESOURCE_CONSTANT,
   PAN_GRAPH_MEMORY_RESOURCE_OUTPUT,
};

enum pan_graph_memory_resource_status {
   PAN_GRAPH_MEMORY_RESOURCE_SUCCESS,
   PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED,
   PAN_GRAPH_MEMORY_RESOURCE_NO_MEMORY,
};

struct pan_graph_memory_resource_tensor {
   struct pan_graph_memory_descriptor_input descriptor;
   uint64_t uid;
   uint64_t allocation_offset;
   uint32_t binding[3];
   uint32_t ordinal;
   enum pan_graph_memory_resource_domain domain;
};

struct pan_graph_memory_resource_input {
   const struct pan_graph_memory_resource_tensor *tensors;
   const uint32_t *references;
   uint32_t tensor_count;
   uint32_t reference_count;
   uint32_t input_count;
   uint32_t output_count;
};

struct pan_graph_memory_resource_reference {
   uint32_t slot32;
   enum pan_graph_memory_resource_domain domain;
};

struct pan_graph_memory_resource_entry {
   struct pan_graph_memory_descriptor descriptor;
   uint64_t uid;
   uint32_t binding[3];
   uint32_t slot32;
};

struct pan_graph_memory_resource_table {
   struct pan_graph_memory_resource_entry *entries;
   uint32_t count;
};

struct pan_graph_memory_resources {
   struct pan_graph_memory_resource_table domains[4];
   struct pan_graph_memory_resource_reference *references;
   uint32_t tensor_count;
   void *storage;
};

enum pan_graph_memory_resource_status pan_graph_memory_resources_create(
   const struct pan_graph_memory_resource_input *input,
   struct pan_graph_memory_resources *result);

void pan_graph_memory_resources_cleanup(struct pan_graph_memory_resources *resources);

#endif
