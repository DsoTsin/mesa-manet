#include "pan_graph_conv.h"

#include <limits.h>
#include <string.h>

bool
pan_graph_conv_i8_init(struct pan_graph_conv_plan *plan,
                       const struct pan_graph_conv_info *info)
{
   if (!plan || !info || !info->batch || !info->input_height ||
       !info->input_width || !info->input_channels || !info->output_channels ||
       info->batch > INT32_MAX || info->input_height > INT32_MAX ||
       info->input_width > INT32_MAX || info->input_channels > INT32_MAX ||
       info->output_channels > INT32_MAX ||
       info->input_zero_point < INT8_MIN || info->input_zero_point > INT8_MAX ||
       info->weight_zero_point < INT8_MIN || info->weight_zero_point > INT8_MAX)
      return false;
   uint32_t output[2];
   const uint32_t input[2] = {info->input_height, info->input_width};
   for (unsigned i = 0; i < 2; i++) {
      if (info->input_nearest_2x &&
          ((input[i] & 1) || info->kernel[i] != 3 || info->stride[i] != 1 ||
           info->dilation[i] != 1 || info->padding[2 * i] != 1 ||
           info->padding[2 * i + 1] != 1))
         return false;
      if (!info->kernel[i] || !info->stride[i] || !info->dilation[i] ||
          info->kernel[i] > INT32_MAX || info->stride[i] > INT32_MAX ||
          info->dilation[i] > INT32_MAX ||
          info->padding[i * 2] > INT32_MAX ||
          info->padding[i * 2 + 1] > INT32_MAX)
         return false;
      uint64_t padded = (uint64_t)input[i] + info->padding[i * 2] +
                        info->padding[i * 2 + 1];
      uint64_t window = (uint64_t)(info->kernel[i] - 1) * info->dilation[i] + 1;
      if (padded < window || padded > INT32_MAX)
         return false;
      uint64_t size = (padded - window) / info->stride[i] + 1;
      if (size > INT32_MAX)
         return false;
      output[i] = size;
   }
   *plan = (struct pan_graph_conv_plan){
      .info = *info,
      .output_height = output[0],
      .output_width = output[1],
      .high = {info->output_channels - 1, info->batch - 1,
               output[0] - 1, output[1] - 1, info->input_channels - 1,
               info->kernel[0] - 1, info->kernel[1] - 1},
   };
   return true;
}

bool
pan_graph_conv_i8_view(const struct pan_graph_conv_plan *plan,
                       enum pan_graph_conv_view view,
                       struct pan_graph_conv_tsu *tsu)
{
   if (!plan || !tsu || (unsigned)view > PAN_GRAPH_CONV_OUTPUT)
      return false;
   struct pan_graph_conv_plan canonical;
   if (!pan_graph_conv_i8_init(&canonical, &plan->info) ||
       memcmp(canonical.high, plan->high, sizeof(plan->high)))
      return false;
   memset(tsu, 0, sizeof(*tsu));
   memcpy(tsu->high, canonical.high, sizeof(tsu->high));
   int64_t *m = tsu->matrix;
   if (view == PAN_GRAPH_CONV_INPUT) {
      m[0 * 8 + 1] = 1;
      m[1 * 8 + 2] = plan->info.stride[0];
      m[1 * 8 + 5] = plan->info.dilation[0];
      m[1 * 8 + 7] = -(int64_t)plan->info.padding[0];
      m[2 * 8 + 3] = plan->info.stride[1];
      m[2 * 8 + 6] = plan->info.dilation[1];
      m[2 * 8 + 7] = -(int64_t)plan->info.padding[2];
      m[3 * 8 + 4] = 1;
      tsu->transforms[1] = (struct pan_graph_tsu_transform){
         .kind = PAN_GRAPH_TSU_CLAMP, .index = 1,
         .value = plan->info.input_height - 1,
      };
      tsu->transforms[2] = (struct pan_graph_tsu_transform){
         .kind = PAN_GRAPH_TSU_CLAMP, .index = 2,
         .value = plan->info.input_width - 1,
      };
      if (plan->info.input_nearest_2x) {
         for (unsigned i = 0; i < 5; i++) {
            tsu->resize_matrix[i * 5 + i] = i == 1 || i == 2 ? 2 : 1;
            tsu->resize_low[i * 5 + i] = 1;
            tsu->resize_high[i * 5 + i] = 1;
         }
         for (unsigned i = 1; i <= 2; i++) {
            tsu->resize_matrix[i * 5 + 4] = 1;
            tsu->resize_high[i * 5 + 4] = 1;
         }
         tsu->divisors[0] = tsu->divisors[3] = 1;
         tsu->divisors[1] = tsu->divisors[2] = 4;
         tsu->transforms[3] = (struct pan_graph_tsu_transform){
            .kind = PAN_GRAPH_TSU_AFFINE, .rows = 5, .columns = 5,
            .low = tsu->resize_matrix, .high = tsu->resize_matrix,
         };
         tsu->transforms[4] = (struct pan_graph_tsu_transform){
            .kind = PAN_GRAPH_TSU_DIVIDE, .divisors = tsu->divisors,
         };
         tsu->transforms[5] = (struct pan_graph_tsu_transform){
            .kind = PAN_GRAPH_TSU_AFFINE, .rows = 5, .columns = 5,
            .low = tsu->resize_low, .high = tsu->resize_high,
         };
         for (unsigned i = 0; i < 2; i++)
            tsu->transforms[6 + i] = (struct pan_graph_tsu_transform){
               .kind = PAN_GRAPH_TSU_CLAMP, .index = i + 1,
               .value = (i ? plan->info.input_width : plan->info.input_height) / 2 - 1,
            };
      }
   } else {
      const unsigned weight[4] = {0, 5, 6, 4};
      const unsigned output[4] = {1, 2, 3, 0};
      const unsigned *axes = view == PAN_GRAPH_CONV_WEIGHT ? weight : output;
      for (unsigned i = 0; i < 4; i++)
         m[i * 8 + axes[i]] = 1;
   }
   m[4 * 8 + 7] = 1;
   tsu->transforms[0] = (struct pan_graph_tsu_transform){
      .kind = PAN_GRAPH_TSU_AFFINE, .rows = 5, .columns = 8,
      .low = m, .high = m,
   };
   tsu->info = (struct pan_graph_tsu_info){
      .rank = PAN_GRAPH_CONV_RANK, .low = tsu->low, .high = tsu->high,
      .count = view == PAN_GRAPH_CONV_INPUT ? plan->info.input_nearest_2x ? 8 : 3 : 1,
      .transforms = tsu->transforms,
   };
   return true;
}
