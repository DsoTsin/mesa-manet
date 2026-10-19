#ifndef PAN_GRAPH_CONV_H
#define PAN_GRAPH_CONV_H

#include "pan_graph_tsu.h"

#include <stdbool.h>
#include <stdint.h>

#define PAN_GRAPH_CONV_RANK 7

struct pan_graph_conv_info {
   uint32_t batch;
   uint32_t input_height;
   uint32_t input_width;
   uint32_t input_channels;
   uint32_t output_channels;
   uint32_t kernel[2];
   uint32_t stride[2];
   uint32_t dilation[2];
   uint32_t padding[4];
   int32_t input_zero_point;
   int32_t weight_zero_point;
   bool input_nearest_2x;
};

struct pan_graph_conv_plan {
   struct pan_graph_conv_info info;
   uint32_t output_height;
   uint32_t output_width;
   int64_t high[PAN_GRAPH_CONV_RANK];
};

enum pan_graph_conv_view {
   PAN_GRAPH_CONV_INPUT,
   PAN_GRAPH_CONV_WEIGHT,
   PAN_GRAPH_CONV_OUTPUT,
};

struct pan_graph_conv_tsu {
   int64_t low[PAN_GRAPH_CONV_RANK];
   int64_t high[PAN_GRAPH_CONV_RANK];
   int64_t matrix[5 * (PAN_GRAPH_CONV_RANK + 1)];
   int64_t resize_matrix[25], resize_low[25], resize_high[25], divisors[4];
   struct pan_graph_tsu_transform transforms[8];
   struct pan_graph_tsu_info info;
};

bool pan_graph_conv_i8_init(struct pan_graph_conv_plan *plan,
                            const struct pan_graph_conv_info *info);
bool pan_graph_conv_i8_view(const struct pan_graph_conv_plan *plan,
                            enum pan_graph_conv_view view,
                            struct pan_graph_conv_tsu *tsu);

#endif
