#include "pan_graph_memory_layout.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

static void
reject(const struct pan_graph_memory_layout_target *target,
       const struct pan_graph_memory_layout_input *input)
{
   struct pan_graph_memory_layout_result value, saved;
   memset(&value, 0xa5, sizeof(value));
   saved = value;
   assert(!pan_graph_memory_layout_units(target, input, &value));
   assert(!memcmp(&value, &saved, sizeof(value)));
}

int
main(void)
{
   struct pan_graph_memory_layout_target target = {.alignment = 32};
   struct pan_graph_memory_layout_input input = {
      .layout = PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_260,
      .bits = 32,
      .full = {1, 1, 1},
      .transfer = {1, 1, 1},
      .ratios = {1, 1, 1, 1, 1, 1, 1},
      .replication_rank = 4,
   };
   assert(pan_graph_memory_layout_v15_block(input.layout, input.bits,
                                           target.block));
   assert(target.block[0] == 1 && target.block[1] == 16 &&
          target.block[2] == 1 && target.block[3] == 1);
   struct pan_graph_memory_layout_result output;
   assert(pan_graph_memory_layout_units(&target, &input, &output));
   assert(output.base == 4 && output.cycles == 4);
   input.read = true;
   input.replication_rank = 5;
   assert(pan_graph_memory_layout_units(&target, &input, &output));
   assert(output.base == 4 && output.cycles == 4);
   input.ratios[0] = 1.5f;
   input.ratios[1] = 0.75f;
   input.skip_axes = true;
   input.skip[0] = 1;
   input.skip[1] = 3;
   assert(pan_graph_memory_layout_units(&target, &input, &output));
   assert(output.base == 4 && output.cycles == 6);
   reject(NULL, &input);
   reject(&target, NULL);
   assert(!pan_graph_memory_layout_units(&target, &input, NULL));
   struct pan_graph_memory_layout_input bad = input;
   bad.layout = 3;
   reject(&target, &bad);
   bad = input;
   bad.bits = 16;
   reject(&target, &bad);
   bad = input;
   bad.transfer[0] = 0;
   reject(&target, &bad);
   bad = input;
   bad.full[0] = bad.transfer[0] = 4096;
   reject(&target, &bad);
   bad = input;
   bad.transfer[1] = 2;
   reject(&target, &bad);
   bad = input;
   bad.replication_rank = 8;
   reject(&target, &bad);
   bad = input;
   bad.ratios[0] = NAN;
   reject(&target, &bad);
   bad = input;
   bad.ratios[0] = 0;
   reject(&target, &bad);
   bad = input;
   bad.ratios[0] = FLT_MAX;
   reject(&target, &bad);
   struct pan_graph_memory_layout_target broken = target;
   broken.alignment = 0;
   reject(&broken, &input);
   broken = target;
   broken.block[1] = 8;
   reject(&broken, &input);
   uint32_t block[4], saved[4];
   memset(block, 0xa5, sizeof(block));
   memcpy(saved, block, sizeof(block));
   assert(!pan_graph_memory_layout_v15_block(3, 32, block));
   assert(!memcmp(saved, block, sizeof(block)));
   assert(!pan_graph_memory_layout_v15_block(263, 32, block));
   assert(!memcmp(saved, block, sizeof(block)));
   assert(!pan_graph_memory_layout_v15_block(input.layout, 16, block));
   assert(!memcmp(saved, block, sizeof(block)));
   assert(!pan_graph_memory_layout_v15_block(input.layout, 32, NULL));
   for (unsigned layout = 260; layout <= 262; layout++) {
      for (unsigned bits = 8; bits <= 32; bits += 24) {
         assert(pan_graph_memory_layout_v15_block(layout, bits, block));
         assert(block[0] * block[1] * bits == 512);
         assert(pan_graph_memory_layout_valid_block(263, bits, block));
      }
   }
   struct pan_graph_memory_layout_input internal = {
      .layout = PAN_GRAPH_MEMORY_LAYOUT_INTERNAL_263, .bits = 8,
      .full = {3, 5, 2}, .transfer = {1, 3, 1},
      .ratios = {1, 1, 1, 1, 1, 1, 1}, .replication_rank = 3,
   };
   for (unsigned source = 260; source <= 262; source++) {
      assert(pan_graph_memory_layout_v15_block(source, 8, target.block));
      internal.read = true;
      assert(pan_graph_memory_layout_units(&target, &internal, &output));
      assert(output.base == 5 && output.cycles == 5);
      internal.read = false;
      assert(pan_graph_memory_layout_units(&target, &internal, &output));
      assert(output.base == 4 && output.cycles == 4);
   }
   target.block[3] = 2;
   reject(&target, &internal);
   target.block[3] = 1;
   internal.bits = 32;
   reject(&target, &internal);
   assert(!pan_graph_memory_layout_valid_block(263, 16, target.block));
   assert(!pan_graph_memory_layout_valid_block(263, 8, NULL));
   assert(!pan_graph_memory_layout_valid_block(264, 8, target.block));
   return 0;
}
