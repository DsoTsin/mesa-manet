#include "pan_graph_memory_bytes.h"

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

static void
reject(const struct pan_graph_memory_layout_target *target,
       const struct pan_graph_memory_layout_input *input, uint8_t level)
{
   struct pan_graph_memory_bytes_result result, saved;
   memset(&result, 0xa5, sizeof(result));
   saved = result;
   assert(!pan_graph_memory_bytes_estimate(target, input, level, &result));
   assert(!memcmp(&result, &saved, sizeof(result)));
}

int
main(void)
{
   struct pan_graph_memory_layout_target target = {.alignment = 32};
   struct pan_graph_memory_layout_input input = {
      .layout = PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_260,
      .bits = 8,
      .full = {1, 1, 1},
      .transfer = {1, 1, 1},
      .ratios = {1, 1, 1, 1, 1, 1, 1},
   };
   assert(pan_graph_memory_layout_v15_block(input.layout, input.bits,
                                           target.block));
   struct pan_graph_memory_bytes_result result;
   assert(pan_graph_memory_bytes_estimate(&target, &input, 5, &result));
   assert(result.padded[0] == 4 && result.padded[1] == 16 &&
          result.padded[2] == 1 && result.local_bytes == 1 &&
          result.total_bytes == 1 && result.cache_bytes == 1);
   input.read = true;
   assert(pan_graph_memory_bytes_estimate(&target, &input, 5, &result));
   assert(result.local_bytes == 64 && result.total_bytes == 64 &&
          result.cache_bytes == 0);
   assert(pan_graph_memory_bytes_estimate(&target, &input, 3, &result));
   assert(result.cache_bytes == 64);
   input.read = false;
   input.replication_rank = 2;
   input.ratios[0] = input.ratios[1] = 1.5f;
   assert(pan_graph_memory_bytes_estimate(&target, &input, 5, &result));
   assert(result.total_bytes == 1 && result.cache_bytes == 1);
   input.skip_axes = true;
   input.skip[0] = 0;
   input.skip[1] = 1;
   input.ratios[0] = input.ratios[1] = 10;
   assert(pan_graph_memory_bytes_estimate(&target, &input, 5, &result));
   assert(result.total_bytes == 1);
   reject(NULL, &input, 5);
   reject(&target, NULL, 5);
   assert(!pan_graph_memory_bytes_estimate(&target, &input, 5, NULL));
   reject(&target, &input, 1);
   reject(&target, &input, 6);
   struct pan_graph_memory_layout_input bad = input;
   bad.layout = 264;
   reject(&target, &bad, 5);
   bad = input;
   bad.bits = 16;
   reject(&target, &bad, 5);
   bad = input;
   bad.full[0] = 0;
   reject(&target, &bad, 5);
   bad = input;
   bad.full[0] = UINT32_MAX;
   reject(&target, &bad, 5);
   bad = input;
   bad.transfer[1] = 2;
   reject(&target, &bad, 5);
   bad = input;
   bad.transfer[0] = 0;
   reject(&target, &bad, 5);
   bad = input;
   bad.full[0] = bad.transfer[0] = 4096;
   reject(&target, &bad, 5);
   bad = input;
   bad.replication_rank = 8;
   reject(&target, &bad, 5);
   bad = input;
   bad.ratios[0] = NAN;
   reject(&target, &bad, 5);
   bad = input;
   bad.ratios[0] = 0;
   reject(&target, &bad, 5);
   bad = input;
   bad.skip_axes = false;
   bad.ratios[0] = FLT_MAX;
   reject(&target, &bad, 5);
   struct pan_graph_memory_layout_target broken = target;
   broken.alignment = 0;
   reject(&broken, &input, 5);
   broken = target;
   broken.block[1] = 8;
   reject(&broken, &input, 5);
   input.read = true;
   input.full[0] = INT32_MAX;
   input.transfer[0] = 4095;
   assert(pan_graph_memory_bytes_estimate(&target, &input, 5, &result));
   assert(result.padded[0] == 4096);
   struct pan_graph_memory_layout_input internal = {
      .layout = PAN_GRAPH_MEMORY_LAYOUT_INTERNAL_263, .bits = 8,
      .full = {3, 5, 2}, .transfer = {1, 3, 1},
      .ratios = {1, 1, 1, 1, 1, 1, 1}, .replication_rank = 3,
   };
   for (unsigned source = 260; source <= 262; source++) {
      assert(pan_graph_memory_layout_v15_block(source, 8, target.block));
      internal.read = true;
      assert(pan_graph_memory_bytes_estimate(&target, &internal, 5, &result));
      assert(result.local_bytes == 64 && result.total_bytes == 64 &&
             result.cache_bytes == 0);
      assert(pan_graph_memory_bytes_estimate(&target, &internal, 3, &result));
      assert(result.cache_bytes == 64);
      internal.read = false;
      assert(pan_graph_memory_bytes_estimate(&target, &internal, 5, &result));
      assert(result.local_bytes == 3 && result.total_bytes == 3 &&
             result.cache_bytes == 3);
   }
   target.block[3] = 2;
   reject(&target, &internal, 5);
   return 0;
}
