#ifndef PANVK_DATA_GRAPH_NEURAL_H
#define PANVK_DATA_GRAPH_NEURAL_H

#include "neural/pan_graph_matmul_v15.h"
#include "vulkan/vulkan_core.h"

struct panvk_data_graph;

struct panvk_data_graph_matmul_add {
   struct pan_graph_matmul_info info;
   struct pan_graph_matmul_add *adds;
   uint32_t *values;
   uint32_t add_count;
   uint32_t inputs[2];
   uint32_t input_indices[2];
   uint32_t output;
};

VkResult
panvk_data_graph_matmul_add_extract(const struct panvk_data_graph *graph,
                                    struct panvk_data_graph_matmul_add *result);

void
panvk_data_graph_matmul_add_finish(struct panvk_data_graph_matmul_add *graph);

#endif
