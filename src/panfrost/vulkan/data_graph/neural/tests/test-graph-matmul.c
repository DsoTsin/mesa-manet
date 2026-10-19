#include "pan_graph_matmul.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

static void
evaluate(const struct pan_graph_matmul_affine *affine, bool high,
         const int64_t *input, int64_t *output)
{
   const int64_t *matrix = high ? affine->high : affine->low;
   for (unsigned r = 0; r + 1 < affine->rows; r++) {
      output[r] = matrix[r * affine->columns + affine->columns - 1];
      for (unsigned c = 0; c + 1 < affine->columns; c++)
         output[r] += matrix[r * affine->columns + c] * input[c];
   }
}

int
main(void)
{
   const struct pan_graph_matmul_info info = {3, 5, 7, 4, -128, 127};
   struct pan_graph_matmul_plan plan;
   assert(pan_graph_matmul_i8_init(&plan, &info));
   assert(plan.info.a_zero_point == -128 && plan.info.b_zero_point == 127);
   unsigned checked = 0;
   for (unsigned b = 0; b < info.batch; b++) {
      for (unsigned m = 0; m < info.rows; m++) {
         for (unsigned k = 0; k < info.inner; k++) {
            for (unsigned n = 0; n < info.columns; n++) {
               int64_t point[7] = {n, 0, b, m, k, 0, 0};
               int64_t transfer[7] = {n, 0, 0, m, k, 0, 0};
               for (unsigned d = 0; d < 7; d++) {
                  assert(transfer[d] <= plan.transfer_high[d]);
                  assert(point[d] <= plan.conv_high[d]);
               }
               const enum pan_graph_matmul_view roles[] = {
                  PAN_GRAPH_MATMUL_CONV_A, PAN_GRAPH_MATMUL_CONV_WEIGHT,
                  PAN_GRAPH_MATMUL_CONV_OUTPUT, PAN_GRAPH_MATMUL_READ_B,
               };
               for (unsigned i = 0; i < 4; i++) {
                  enum pan_graph_matmul_view role = roles[i];
                  struct pan_graph_matmul_affine forward, inverse;
                  int64_t mapped[7], low[7], high[7];
                  assert(pan_graph_matmul_affine(&plan, role, false, &forward));
                  assert(pan_graph_matmul_affine(&plan, role, true, &inverse));
                  evaluate(&forward, false, point, mapped);
                  struct pan_graph_matmul_affine section;
                  int64_t padded[4];
                  assert(pan_graph_matmul_section_affine(&plan, role, &section));
                  evaluate(&section, false, point, padded);
                  unsigned pad = role == PAN_GRAPH_MATMUL_CONV_WEIGHT ? 0 : 1;
                  if (pad)
                     assert(padded[0] == 0);
                  for (unsigned d = 0; d < 4 - pad; d++)
                     assert(padded[d + pad] == mapped[d]);
                  if (role == PAN_GRAPH_MATMUL_CONV_A)
                     assert(mapped[0] == b && mapped[1] == m && mapped[2] == k);
                  else if (role == PAN_GRAPH_MATMUL_CONV_WEIGHT)
                     assert(mapped[0] == n && mapped[1] == 0 &&
                            mapped[2] == 0 && mapped[3] == k);
                  else if (role == PAN_GRAPH_MATMUL_CONV_OUTPUT)
                     assert(mapped[0] == b && mapped[1] == m && mapped[2] == n);
                  else
                     assert(mapped[0] == b && mapped[1] == k && mapped[2] == n);
                  evaluate(&inverse, false, mapped, low);
                  evaluate(&inverse, true, mapped, high);
                  for (unsigned d = 0; d < 7; d++)
                     assert(low[d] <= point[d] && point[d] <= high[d]);
                  checked++;
               }
            }
         }
      }
   }
   struct pan_graph_matmul_affine convert;
   assert(pan_graph_matmul_affine(&plan, PAN_GRAPH_MATMUL_CONVERT_INPUT,
                                 false, &convert));
   for (unsigned n = 0; n < info.columns; n++) {
      for (unsigned k = 0; k < info.inner; k++) {
         int64_t point[4] = {n, 0, 0, k}, mapped[4];
         evaluate(&convert, false, point, mapped);
         assert(mapped[0] == 0 && mapped[1] == 0 &&
                mapped[2] == k && mapped[3] == n);
         checked++;
      }
   }
   struct pan_graph_matmul_plan saved = plan;
   for (unsigned i = 0; i < 6; i++) {
      struct pan_graph_matmul_info invalid = info;
      if (i == 0) invalid.batch = 0;
      if (i == 1) invalid.rows = UINT32_MAX;
      if (i == 2) invalid.inner = 0;
      if (i == 3) invalid.columns = UINT32_MAX;
      if (i == 4) invalid.a_zero_point = -129;
      if (i == 5) invalid.b_zero_point = 128;
      assert(!pan_graph_matmul_i8_init(&plan, &invalid));
      assert(!memcmp(&saved, &plan, sizeof(plan)));
   }
   struct pan_graph_matmul_info limit = {INT32_MAX, 1, INT32_MAX, 1, 0, 0};
   assert(pan_graph_matmul_i8_init(&plan, &limit));
   assert(plan.conv_high[2] == INT32_MAX - 1 &&
          plan.transfer_high[2] == 0 &&
          plan.weight_high[3] == INT32_MAX - 1);
   assert(!pan_graph_matmul_affine(&plan, PAN_GRAPH_MATMUL_VIEW_COUNT,
                                  false, &convert));
   struct pan_graph_matmul_affine saved_affine = convert;
   assert(!pan_graph_matmul_section_affine(&plan, PAN_GRAPH_MATMUL_CONVERT_INPUT,
                                          &convert));
   assert(!memcmp(&saved_affine, &convert, sizeof(convert)));
   puts("panfrost graph MatMul affine, inverse bounds, quantization and limits passed");
   printf("coordinate checks=%u\n", checked);
   return 0;
}
