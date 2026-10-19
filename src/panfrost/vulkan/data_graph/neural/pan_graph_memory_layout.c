#include "pan_graph_memory_layout.h"

#include <limits.h>
#include <math.h>
#include <string.h>

static float
multiply(float a, float b)
{
   volatile float result = a * b;
   return result;
}

static uint64_t
divide_up(uint64_t a, uint64_t b)
{
   return a / b + (a % b != 0);
}

static bool
product(uint64_t *a, uint64_t b)
{
   if (b > INT64_MAX || (b && *a > INT64_MAX / b))
      return false;
   *a *= b;
   return true;
}

bool
pan_graph_memory_layout_v15_block(enum pan_graph_memory_layout layout,
                                 uint8_t bits, uint32_t block[4])
{
   if (!block || (bits != 8 && bits != 32))
      return false;
   uint32_t height;
   switch (layout) {
   case PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_260:
      height = 16;
      break;
   case PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_261:
      height = 8;
      break;
   case PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_262:
      height = 4;
      break;
   default:
      return false;
   }
   const uint32_t value[4] = {512 / height / bits, height, 1, 1};
   memcpy(block, value, sizeof(value));
   return true;
}

bool
pan_graph_memory_layout_valid_block(enum pan_graph_memory_layout layout,
                                    uint8_t bits, const uint32_t block[4])
{
   if (!block)
      return false;
   uint32_t expected[4];
   if (layout != PAN_GRAPH_MEMORY_LAYOUT_INTERNAL_263)
      return pan_graph_memory_layout_v15_block(layout, bits, expected) &&
             !memcmp(block, expected, sizeof(expected));
   for (unsigned source = PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_260;
        source <= PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_262; source++) {
      if (pan_graph_memory_layout_v15_block(source, bits, expected) &&
          !memcmp(block, expected, sizeof(expected)))
         return true;
   }
   return false;
}

bool
pan_graph_memory_layout_units(
   const struct pan_graph_memory_layout_target *target,
   const struct pan_graph_memory_layout_input *input,
   struct pan_graph_memory_layout_result *result)
{
   if (!target || !input || !result || target->alignment != 32 ||
       !pan_graph_memory_layout_valid_block(input->layout, input->bits,
                                            target->block) ||
       input->replication_rank > 7)
      return false;
   for (unsigned i = 0; i < 3; i++) {
      if (!input->full[i] || input->full[i] > INT32_MAX ||
          !input->transfer[i] || input->transfer[i] > input->full[i] ||
          input->transfer[i] > 4095)
         return false;
   }
   for (unsigned i = 0; i < input->replication_rank; i++) {
      if (!isfinite(input->ratios[i]) || input->ratios[i] <= 0.0f)
         return false;
   }
   const uint32_t *block = target->block;
   const uint32_t *slice = input->transfer;
   bool width_partial = slice[0] % block[0];
   bool height_partial = slice[1] % block[1];
   uint32_t factor;
   if (input->read) {
      factor = width_partial ?
                  input->layout == PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_260 ? 4 : 2 :
                  1;
      if (height_partial)
         factor += 3;
   } else {
      factor = width_partial && height_partial &&
               input->layout == PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_260 ? 5 : 4;
   }
   struct pan_graph_memory_layout_result value = {
      .base = divide_up(slice[0], target->alignment / (input->bits / 2)),
   };
   if (!product(&value.base, divide_up(slice[1], 4)) ||
       !product(&value.base, slice[2]) || !product(&value.base, factor))
      return false;
   float ratio = 1.0f;
   for (unsigned i = 0; i < input->replication_rank; i++) {
      if (!input->skip_axes || (i != input->skip[0] && i != input->skip[1]))
         ratio = multiply(ratio, input->ratios[i]);
   }
   float scaled = multiply(ratio, (float)value.base);
   if (!isfinite(scaled) || scaled < 0.0f || scaled >= 0x1p63f)
      return false;
   value.cycles = scaled;
   *result = value;
   return true;
}
