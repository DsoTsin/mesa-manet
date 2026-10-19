#ifndef PAN_GRAPH_MEMORY_LAYOUT_H
#define PAN_GRAPH_MEMORY_LAYOUT_H

#include <stdbool.h>
#include <stdint.h>

enum pan_graph_memory_layout {
   PAN_GRAPH_MEMORY_LAYOUT_LINEAR = 3,
   PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_260 = 260,
   PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_261 = 261,
   PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_262 = 262,
   PAN_GRAPH_MEMORY_LAYOUT_INTERNAL_263 = 263,
};

struct pan_graph_memory_layout_target {
   uint32_t alignment;
   uint32_t block[4];
};

struct pan_graph_memory_layout_input {
   enum pan_graph_memory_layout layout;
   uint32_t full[3];
   uint32_t transfer[3];
   float ratios[7];
   uint32_t skip[2];
   uint8_t bits;
   uint8_t replication_rank;
   bool read;
   bool skip_axes;
};

struct pan_graph_memory_layout_result {
   uint64_t base;
   uint64_t cycles;
};

bool pan_graph_memory_layout_v15_block(enum pan_graph_memory_layout layout,
                                     uint8_t bits, uint32_t block[4]);

bool pan_graph_memory_layout_valid_block(enum pan_graph_memory_layout layout,
                                        uint8_t bits, const uint32_t block[4]);

bool pan_graph_memory_layout_units(
   const struct pan_graph_memory_layout_target *target,
   const struct pan_graph_memory_layout_input *input,
   struct pan_graph_memory_layout_result *result);

#endif
