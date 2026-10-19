#ifndef PAN_GRAPH_NRT_H
#define PAN_GRAPH_NRT_H

#include <stdbool.h>
#include <stdint.h>

enum pan_graph_nrt_status {
   PAN_GRAPH_NRT_SUCCESS,
   PAN_GRAPH_NRT_INVALID,
   PAN_GRAPH_NRT_OUT_OF_MEMORY,
};

enum pan_graph_nrt_type {
   PAN_GRAPH_NRT_NORMAL,
   PAN_GRAPH_NRT_WEIGHT,
};

struct pan_graph_nrt_target {
   uint32_t shared_size;
   uint32_t alignment;
   uint32_t banks;
   uint32_t bank_parts;
};

struct pan_graph_nrt_allocator {
   struct pan_graph_nrt_target target;
   uint32_t used;
   uint32_t front;
   uint32_t back;
};

struct pan_graph_nrt_request {
   enum pan_graph_nrt_type type;
   uint32_t dimensions[4];
   uint32_t element_bits;
   uint32_t initial_format;
   uint32_t weight_format;
   uint32_t buffers;
   bool front;
   bool wide;
   bool narrow_weight;
};

struct pan_graph_nrt_layout {
   uint32_t kind;
   uint32_t buffers;
   uint32_t size;
   uint32_t base;
   uint32_t banks;
   uint32_t dimensions[4];
   uint32_t format;
};

struct pan_graph_nrt_accumulator_request {
   uint32_t dimensions[4];
   uint32_t schedule_type;
   uint32_t initial_format;
   uint32_t producer_mode;
   uint32_t input_bits[2];
   bool paired_axes;
};

struct pan_graph_nrt_target pan_graph_nrt_target_v15(void);

bool pan_graph_nrt_allocator_init(struct pan_graph_nrt_allocator *allocator,
                                  const struct pan_graph_nrt_target *target);
enum pan_graph_nrt_status
pan_graph_nrt_allocate(struct pan_graph_nrt_allocator *allocator,
                       const struct pan_graph_nrt_request *request,
                       struct pan_graph_nrt_layout *layout);
bool pan_graph_nrt_accumulator_layout(
   const struct pan_graph_nrt_accumulator_request *request,
   struct pan_graph_nrt_layout *layout);
bool pan_graph_nrt_kind2_layout(bool flag, struct pan_graph_nrt_layout *layout);
bool pan_graph_nrt_kind3_layout(uint32_t element_bits, uint32_t initial_format,
                               struct pan_graph_nrt_layout *layout);

#endif
