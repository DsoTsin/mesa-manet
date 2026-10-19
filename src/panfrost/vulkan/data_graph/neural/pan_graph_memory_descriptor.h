#ifndef PAN_GRAPH_MEMORY_DESCRIPTOR_H
#define PAN_GRAPH_MEMORY_DESCRIPTOR_H

#include "pan_graph_memory_layout.h"

enum pan_graph_memory_descriptor_format {
   PAN_GRAPH_MEMORY_DESCRIPTOR_I8 = 1,
   PAN_GRAPH_MEMORY_DESCRIPTOR_I32,
};

enum pan_graph_memory_descriptor_flags {
   PAN_GRAPH_MEMORY_DESCRIPTOR_EXPLICIT_STRIDES = 1,
   PAN_GRAPH_MEMORY_DESCRIPTOR_SHAPE_OVERRIDE = 2,
};

struct pan_graph_memory_descriptor_input {
   enum pan_graph_memory_descriptor_format format;
   enum pan_graph_memory_layout layout;
   uint32_t dimensions[4];
   uint32_t block[4];
   uint32_t flags;
   uint8_t rank;
};

struct pan_graph_memory_descriptor {
   uint32_t words[16];
   uint64_t size;
};

bool pan_graph_memory_descriptor_template(
   const struct pan_graph_memory_descriptor_input *input,
   struct pan_graph_memory_descriptor *result);

#endif
