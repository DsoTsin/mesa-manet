#include "pan_graph_matmul.h"

#include <limits.h>
#include <string.h>

bool
pan_graph_matmul_i8_init(struct pan_graph_matmul_plan *plan,
                        const struct pan_graph_matmul_info *info)
{
   if (!plan || !info || !info->batch || !info->rows || !info->inner ||
       !info->columns || info->batch > INT32_MAX || info->rows > INT32_MAX ||
       info->inner > INT32_MAX || info->columns > INT32_MAX ||
       info->a_zero_point < INT8_MIN || info->a_zero_point > INT8_MAX ||
       info->b_zero_point < INT8_MIN || info->b_zero_point > INT8_MAX)
      return false;

   struct pan_graph_matmul_plan result = {
      .info = *info,
      .conv_high = {info->columns - 1, 0, info->batch - 1,
                    info->rows - 1, info->inner - 1, 0, 0},
      .transfer_high = {info->columns - 1, 0, 0,
                        info->rows - 1, info->inner - 1, 0, 0},
      .weight_high = {info->columns - 1, 0, 0, info->inner - 1},
      .views = {
         [PAN_GRAPH_MATMUL_CONV_A] = {
            .input_rank = 7, .output_rank = 3, .range_rank = 3,
            .forward = {2, 3, 4},
            .inverse = {-1, -1, 0, 1, 2, -1, -1},
            .inverse_high = {info->columns - 1},
            .range_high = {info->batch - 1, info->rows - 1, info->inner - 1},
         },
         [PAN_GRAPH_MATMUL_CONV_WEIGHT] = {
            .input_rank = 7, .output_rank = 4, .range_rank = 4,
            .forward = {0, -1, 1, 4},
            .inverse = {0, 2, -1, -1, 3, -1, -1},
            .inverse_high = {0, 0, info->batch - 1, info->rows - 1},
            .range_high = {info->columns - 1, 0, 0, info->inner - 1},
         },
         [PAN_GRAPH_MATMUL_CONV_OUTPUT] = {
            .input_rank = 7, .output_rank = 3, .range_rank = 3,
            .forward = {2, 3, 0},
            .inverse = {2, -1, 0, 1, -1, -1, -1},
            .inverse_high = {0, 0, 0, 0, info->inner - 1},
            .range_high = {info->batch - 1, info->rows - 1, info->columns - 1},
         },
         [PAN_GRAPH_MATMUL_CONVERT_INPUT] = {
            .input_rank = 4, .output_rank = 4, .range_rank = 3,
            .forward = {-1, -1, 3, 0},
            .inverse = {3, -1, -1, 2},
            .range_high = {info->batch - 1, info->inner - 1, info->columns - 1},
         },
         [PAN_GRAPH_MATMUL_CONVERT_OUTPUT] = {
            .input_rank = 4, .output_rank = 4, .range_rank = 4,
            .identity = true,
            .forward = {0, 1, 2, 3},
            .inverse = {0, 1, 2, 3},
            .range_high = {info->columns - 1, 0, 0, info->inner - 1},
         },
         [PAN_GRAPH_MATMUL_READ_B] = {
            .input_rank = 7, .output_rank = 3, .range_rank = 3,
            .forward = {2, 4, 0},
            .inverse = {2, -1, 0, -1, 1, -1, -1},
            .inverse_high = {0, 0, 0, info->rows - 1},
            .range_high = {info->batch - 1, info->inner - 1, info->columns - 1},
         },
      },
   };
   *plan = result;
   return true;
}

bool
pan_graph_matmul_affine(const struct pan_graph_matmul_plan *plan,
                       enum pan_graph_matmul_view view, bool inverse,
                       struct pan_graph_matmul_affine *affine)
{
   if (!plan || !affine || (unsigned)view >= PAN_GRAPH_MATMUL_VIEW_COUNT)
      return false;

   const struct pan_graph_matmul_map *map = &plan->views[view];
   uint32_t rows = inverse ? map->input_rank : map->output_rank;
   uint32_t columns = inverse ? map->output_rank : map->input_rank;
   if (rows > PAN_GRAPH_MATMUL_RANK || columns > PAN_GRAPH_MATMUL_RANK)
      return false;

   struct pan_graph_matmul_affine result = {
      .rows = rows + 1,
      .columns = columns + 1,
   };
   const int8_t *selectors = inverse ? map->inverse : map->forward;
   for (unsigned r = 0; r < rows; r++) {
      int32_t c = selectors[r];
      if (c < -1 || c >= (int32_t)columns)
         return false;
      if (c >= 0) {
         result.low[r * result.columns + c] = 1;
         result.high[r * result.columns + c] = 1;
      }
      if (inverse)
         result.high[r * result.columns + columns] = map->inverse_high[r];
   }
   result.low[rows * result.columns + columns] = 1;
   result.high[rows * result.columns + columns] = 1;
   *affine = result;
   return true;
}

bool
pan_graph_matmul_section_affine(const struct pan_graph_matmul_plan *plan,
                                enum pan_graph_matmul_view view,
                                struct pan_graph_matmul_affine *affine)
{
   if (!affine || view == PAN_GRAPH_MATMUL_CONVERT_INPUT ||
       view == PAN_GRAPH_MATMUL_CONVERT_OUTPUT)
      return false;

   struct pan_graph_matmul_affine mapped;
   if (!pan_graph_matmul_affine(plan, view, false, &mapped))
      return false;
   if (mapped.rows == 5) {
      *affine = mapped;
      return true;
   }
   if (mapped.rows != 4)
      return false;

   struct pan_graph_matmul_affine result = {
      .rows = 5,
      .columns = mapped.columns,
   };
   size_t size = mapped.rows * mapped.columns * sizeof(int64_t);
   memcpy(result.low + mapped.columns, mapped.low, size);
   memcpy(result.high + mapped.columns, mapped.high, size);
   *affine = result;
   return true;
}
