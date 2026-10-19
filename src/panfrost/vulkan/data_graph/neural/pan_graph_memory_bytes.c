#include "pan_graph_memory_bytes.h"

#include <limits.h>
#include <math.h>
#include <string.h>

static float
multiply(float a, float b)
{
   volatile float result = a * b;
   return result;
}

static bool
product(uint64_t *value, uint64_t factor)
{
   if (factor > INT64_MAX || (factor && *value > INT64_MAX / factor))
      return false;
   *value *= factor;
   return true;
}

bool
pan_graph_memory_bytes_estimate(
   const struct pan_graph_memory_layout_target *target,
   const struct pan_graph_memory_layout_input *input, uint8_t level,
   struct pan_graph_memory_bytes_result *result)
{
   if (!target || !input || !result || target->alignment != 32 || level < 2 ||
       level > 5 || input->replication_rank > 7 ||
       !pan_graph_memory_layout_valid_block(input->layout, input->bits,
                                            target->block))
      return false;
   const uint32_t *block = target->block;
   struct pan_graph_memory_bytes_result value = {
      .local_bytes = input->bits / 8,
   };
   bool padding = false;
   for (unsigned i = 0; i < 3; i++) {
      if (!input->full[i] || input->full[i] > INT32_MAX ||
          !input->transfer[i] || input->transfer[i] > input->full[i] ||
          input->transfer[i] > 4095)
         return false;
      uint32_t local = input->transfer[i];
      value.padded[i] = (local / block[i] + (local % block[i] != 0)) * block[i];
      padding |= value.padded[i] != local;
      if (!product(&value.local_bytes, input->read ? value.padded[i] : local))
         return false;
   }
   value.total_bytes = value.local_bytes;
   for (unsigned i = 0; i < input->replication_rank; i++) {
      if (!isfinite(input->ratios[i]) || input->ratios[i] <= 0.0f)
         return false;
      if (input->skip_axes && (i == input->skip[0] || i == input->skip[1]))
         continue;
      float scaled = multiply((float)value.total_bytes, input->ratios[i]);
      if (!isfinite(scaled) || scaled < 0.0f || scaled >= 0x1p63f)
         return false;
      value.total_bytes = scaled;
   }
   value.cache_bytes = level == 3 || (!input->read && padding) ?
                          value.total_bytes : 0;
   *result = value;
   return true;
}
