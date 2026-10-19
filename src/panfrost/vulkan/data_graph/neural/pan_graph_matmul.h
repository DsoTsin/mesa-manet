#ifndef PAN_GRAPH_MATMUL_H
#define PAN_GRAPH_MATMUL_H

#include <stdbool.h>
#include <stdint.h>

#define PAN_GRAPH_MATMUL_RANK 7
#define PAN_GRAPH_MATMUL_MATRIX_SIZE 64

enum pan_graph_matmul_view {
   PAN_GRAPH_MATMUL_CONV_A,
   PAN_GRAPH_MATMUL_CONV_WEIGHT,
   PAN_GRAPH_MATMUL_CONV_OUTPUT,
   PAN_GRAPH_MATMUL_CONVERT_INPUT,
   PAN_GRAPH_MATMUL_CONVERT_OUTPUT,
   PAN_GRAPH_MATMUL_READ_B,
   PAN_GRAPH_MATMUL_VIEW_COUNT,
};

struct pan_graph_matmul_info {
   uint32_t batch;
   uint32_t rows;
   uint32_t inner;
   uint32_t columns;
   int32_t a_zero_point;
   int32_t b_zero_point;
};

struct pan_graph_matmul_map {
   uint8_t input_rank;
   uint8_t output_rank;
   uint8_t range_rank;
   bool identity;
   int8_t forward[PAN_GRAPH_MATMUL_RANK];
   int8_t inverse[PAN_GRAPH_MATMUL_RANK];
   int64_t inverse_high[PAN_GRAPH_MATMUL_RANK];
   int64_t range_high[4];
};

struct pan_graph_matmul_plan {
   struct pan_graph_matmul_info info;
   int64_t conv_high[PAN_GRAPH_MATMUL_RANK];
   int64_t transfer_high[PAN_GRAPH_MATMUL_RANK];
   int64_t weight_high[4];
   struct pan_graph_matmul_map views[PAN_GRAPH_MATMUL_VIEW_COUNT];
};

struct pan_graph_matmul_affine {
   uint32_t rows;
   uint32_t columns;
   int64_t low[PAN_GRAPH_MATMUL_MATRIX_SIZE];
   int64_t high[PAN_GRAPH_MATMUL_MATRIX_SIZE];
};

bool pan_graph_matmul_i8_init(struct pan_graph_matmul_plan *plan,
                             const struct pan_graph_matmul_info *info);
bool pan_graph_matmul_affine(const struct pan_graph_matmul_plan *plan,
                            enum pan_graph_matmul_view view, bool inverse,
                            struct pan_graph_matmul_affine *affine);
bool pan_graph_matmul_section_affine(const struct pan_graph_matmul_plan *plan,
                                    enum pan_graph_matmul_view view,
                                    struct pan_graph_matmul_affine *affine);

#endif
