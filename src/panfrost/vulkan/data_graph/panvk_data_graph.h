#ifndef PANVK_DATA_GRAPH_H
#define PANVK_DATA_GRAPH_H

#include <stdbool.h>
#include <stdint.h>
#include "neural/pan_graph_matmul.h"
#include "neural/pan_graph_conv.h"
#include "vulkan/vulkan_core.h"
#include "nir.h"

enum panvk_graph_value_kind {
   PANVK_GRAPH_INPUT,
   PANVK_GRAPH_CONSTANT,
   PANVK_GRAPH_OPERATION,
   PANVK_GRAPH_OUTPUT,
};

enum panvk_graph_op {
   PANVK_GRAPH_OP_AVG_POOL2D = 1,
   PANVK_GRAPH_OP_CONV2D = 2,
   PANVK_GRAPH_OP_MATMUL = 6,
   PANVK_GRAPH_OP_MAX_POOL2D = 7,
   PANVK_GRAPH_OP_CLAMP = 10,
   PANVK_GRAPH_OP_SIGMOID = 12,
   PANVK_GRAPH_OP_TANH = 13,
   PANVK_GRAPH_OP_ADD = 14,
   PANVK_GRAPH_OP_MUL = 27,
   PANVK_GRAPH_OP_SUB = 29,
   PANVK_GRAPH_OP_TABLE = 30,
   PANVK_GRAPH_OP_EXP = 36,
   PANVK_GRAPH_OP_RECIPROCAL = 41,
   PANVK_GRAPH_OP_REDUCE_MAX = 50,
   PANVK_GRAPH_OP_REDUCE_MIN = 51,
   PANVK_GRAPH_OP_REDUCE_PRODUCT = 52,
   PANVK_GRAPH_OP_REDUCE_SUM = 53,
   PANVK_GRAPH_OP_CONCAT = 54,
   PANVK_GRAPH_OP_RESHAPE = 56,
   PANVK_GRAPH_OP_TRANSPOSE = 60,
   PANVK_GRAPH_OP_RESIZE = 63,
   PANVK_GRAPH_OP_RESCALE = 65,
   PANVK_GRAPH_OP_IM2COL = 0x10000,
};

struct panvk_graph_value {
   enum panvk_graph_value_kind kind;
   VkFormat format;
   VkFormat storage_format;
   uint32_t rank;
   uint32_t dims[4];
   uint32_t elements;
   enum panvk_graph_op op;
   uint32_t inputs[6];
   uint32_t input_count;
   uint32_t attribute;
   uint32_t axis;
   uint32_t kernel[2];
   uint32_t stride[2];
   uint32_t dilation[2];
   uint32_t padding[4];
   uint8_t permutation[4];
   int32_t resize_scale[4];
   int32_t resize_offset[2];
   int32_t resize_border[2];
   bool live;
   uint32_t set;
   uint32_t binding;
   uint32_t array_element;
   uint32_t resource_index;
   uint32_t descriptor_offset;
   uint32_t last_use;
   uint32_t fused_into;
   uint64_t offset;
   uint64_t size;
   struct panvk_graph_value *storage;
   uint32_t view_offset;
   bool alias_concat;
   void *data;
   struct pan_graph_matmul_plan *matmul;
   struct pan_graph_conv_plan *conv;
   struct panvk_graph_value *conv_rescale;
};

struct panvk_data_graph {
   uint32_t count;
   uint32_t pass_count;
   uint32_t *passes;
   struct panvk_graph_value *values;
   uint64_t transient_size;
   uint64_t constant_size;
};

VkResult panvk_data_graph_parse(
   void *mem_ctx, const uint32_t *code, size_t size,
   const VkDataGraphPipelineCreateInfoARM *info,
   const VkDataGraphPipelineShaderModuleCreateInfoARM *module,
   struct panvk_data_graph **result);

nir_shader *panvk_data_graph_shader(const struct panvk_data_graph *graph,
                                    uint32_t pass);

#endif
