#ifndef PAN_GRAPH_MEMORY_REGISTRY_H
#define PAN_GRAPH_MEMORY_REGISTRY_H

#include "pan_graph_tsu.h"

#define PAN_GRAPH_MEMORY_REGISTRY_MAX_RANK 8
#define PAN_GRAPH_MEMORY_REGISTRY_NO_PAIR UINT32_MAX

enum pan_graph_memory_registry_opcode {
   PAN_GRAPH_MEMORY_REGISTRY_FUNCTION = 512,
   PAN_GRAPH_MEMORY_REGISTRY_MATMUL = 0x8001,
   PAN_GRAPH_MEMORY_REGISTRY_SCATTER = 0x8401,
   PAN_GRAPH_MEMORY_REGISTRY_GATHER = 0x9401,
   PAN_GRAPH_MEMORY_REGISTRY_RESHAPE = 0xa403,
   PAN_GRAPH_MEMORY_REGISTRY_ADD = 0xd001,
   PAN_GRAPH_MEMORY_REGISTRY_CE_GROUP = 0x10001,
   PAN_GRAPH_MEMORY_REGISTRY_OP_18002 = 0x18002,
   PAN_GRAPH_MEMORY_REGISTRY_WEIGHT_CONVERT = 0x18008,
};

struct pan_graph_memory_registry_shape {
   uint32_t rank;
   uint32_t dims[PAN_GRAPH_MEMORY_REGISTRY_MAX_RANK];
};

struct pan_graph_memory_registry_tensor {
   struct pan_graph_memory_registry_shape shape;
   uint32_t block[4];
   uint16_t layout;
   uint8_t bits;
   bool producer_flag;
};

struct pan_graph_memory_registry_view {
   struct pan_graph_memory_registry_shape shape;
   uint32_t tensor;
   uint32_t key;
   uint32_t attributes;
};

struct pan_graph_memory_registry_operation {
   uint32_t opcode;
   uint32_t input_count;
   uint32_t output_count;
   const struct pan_graph_memory_registry_view *inputs;
   const struct pan_graph_memory_registry_view *outputs;
};

struct pan_graph_memory_registry_graph {
   uint32_t tensor_count;
   uint32_t operation_count;
   const struct pan_graph_memory_registry_tensor *tensors;
   const struct pan_graph_memory_registry_operation *operations;
};

struct pan_graph_memory_registry_result {
   uint32_t block[4];
   uint16_t layout;
   bool producer_allowed;
   bool consumer_allowed;
   bool changed;
   bool producer_flag;
};

enum pan_graph_tsu_status pan_graph_memory_registry_analyze(
   const struct pan_graph_memory_registry_graph *graph, uint32_t tensor,
   uint32_t paired, struct pan_graph_memory_registry_result *result);

#endif
