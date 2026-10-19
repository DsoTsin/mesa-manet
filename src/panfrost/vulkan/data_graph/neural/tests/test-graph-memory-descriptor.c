#include "pan_graph_memory_descriptor.h"

#include <assert.h>
#include <limits.h>
#include <string.h>

static void
reject(const struct pan_graph_memory_descriptor_input *input)
{
   struct pan_graph_memory_descriptor result, saved;
   memset(&result, 0xa5, sizeof(result));
   saved = result;
   assert(!pan_graph_memory_descriptor_template(input, &result));
   assert(!memcmp(&result, &saved, sizeof(result)));
}

int
main(void)
{
   struct pan_graph_memory_descriptor_input input = {
      .format = PAN_GRAPH_MEMORY_DESCRIPTOR_I32,
      .layout = PAN_GRAPH_MEMORY_LAYOUT_LINEAR,
      .dimensions = {3, 5, 2}, .rank = 3,
   };
   struct pan_graph_memory_descriptor result;
   assert(pan_graph_memory_descriptor_template(&input, &result));
   const uint32_t linear[16] = {
      0x18b, 0x20001, 0x30005, 120, 60, 12, UINT32_MAX, UINT32_MAX,
   };
   assert(result.size == 120 && !memcmp(result.words, linear, sizeof(linear)));
   for (unsigned layout = 260; layout <= 262; layout++) {
      assert(pan_graph_memory_layout_v15_block(layout, 32, input.block));
      input.layout = layout;
      struct pan_graph_memory_descriptor blocked;
      assert(pan_graph_memory_descriptor_template(&input, &blocked));
      input.layout = PAN_GRAPH_MEMORY_LAYOUT_INTERNAL_263;
      assert(pan_graph_memory_descriptor_template(&input, &result));
      assert(result.size == blocked.size && result.words[0] == 0x18b);
      blocked.words[0] = result.words[0];
      assert(!memcmp(&blocked, &result, sizeof(result)));
   }
   input.dimensions[2] = 1;
   input.dimensions[3] = 2;
   input.rank = 4;
   assert(pan_graph_memory_descriptor_template(&input, &result));
   assert(result.words[1] == 0x10002 && result.size == 256);
   input.format = PAN_GRAPH_MEMORY_DESCRIPTOR_I8;
   assert(pan_graph_memory_layout_v15_block(262, 8, input.block));
   assert(pan_graph_memory_descriptor_template(&input, &result));
   assert(result.words[0] == 0x12b && result.size == 256);
   reject(NULL);
   assert(!pan_graph_memory_descriptor_template(&input, NULL));
   struct pan_graph_memory_descriptor_input bad = input;
   bad.format = 0;
   reject(&bad);
   bad = input;
   bad.flags = PAN_GRAPH_MEMORY_DESCRIPTOR_EXPLICIT_STRIDES;
   reject(&bad);
   bad.flags = PAN_GRAPH_MEMORY_DESCRIPTOR_SHAPE_OVERRIDE;
   reject(&bad);
   bad.flags = 4;
   reject(&bad);
   bad = input;
   bad.rank = 2;
   reject(&bad);
   bad.rank = 5;
   reject(&bad);
   bad = input;
   bad.dimensions[0] = 0;
   reject(&bad);
   bad.dimensions[0] = UINT16_MAX + 1u;
   reject(&bad);
   bad = input;
   bad.layout = 259;
   reject(&bad);
   bad = input;
   bad.block[0]++;
   reject(&bad);
   bad = input;
   bad.layout = PAN_GRAPH_MEMORY_LAYOUT_LINEAR;
   reject(&bad);
   bad = input;
   bad.dimensions[0] = bad.dimensions[1] = UINT16_MAX;
   bad.dimensions[2] = 2;
   reject(&bad);
   bad = input;
   bad.dimensions[0] = bad.dimensions[1] = bad.dimensions[3] = 4095;
   reject(&bad);
   memset(input.block, 0, sizeof(input.block));
   input.layout = PAN_GRAPH_MEMORY_LAYOUT_LINEAR;
   input.dimensions[0] = UINT16_MAX;
   input.dimensions[1] = input.dimensions[2] = input.dimensions[3] = 1;
   assert(pan_graph_memory_descriptor_template(&input, &result));
   assert(result.size == UINT16_MAX);
   return 0;
}
