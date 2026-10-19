#include "pan_graph_memory_descriptor.h"

#include <limits.h>
#include <string.h>

bool
pan_graph_memory_descriptor_template(
   const struct pan_graph_memory_descriptor_input *input,
   struct pan_graph_memory_descriptor *result)
{
   if (!input || !result || input->flags ||
       (input->rank != 3 && input->rank != 4))
      return false;
   unsigned bits;
   uint32_t header;
   switch (input->format) {
   case PAN_GRAPH_MEMORY_DESCRIPTOR_I8:
      bits = 8;
      header = 11 | 288;
      break;
   case PAN_GRAPH_MEMORY_DESCRIPTOR_I32:
      bits = 32;
      header = 11 | 384;
      break;
   default:
      return false;
   }
   uint32_t dimensions[4] = {1, 1, 1, 1};
   for (unsigned i = 0; i < input->rank; i++) {
      if (!input->dimensions[i] || input->dimensions[i] > UINT16_MAX)
         return false;
      dimensions[i] = input->dimensions[i];
   }
   uint64_t pitch, plane;
   if (input->layout == PAN_GRAPH_MEMORY_LAYOUT_LINEAR) {
      const uint32_t zero[4] = {0};
      if (memcmp(input->block, zero, sizeof(zero)))
         return false;
      pitch = (uint64_t)dimensions[0] * (bits / 8);
      plane = pitch * dimensions[1];
   } else {
      if (!pan_graph_memory_layout_valid_block(input->layout, bits,
                                               input->block))
         return false;
      pitch = 64 * ((uint64_t)dimensions[0] / input->block[0] +
                    (dimensions[0] % input->block[0] != 0));
      plane = pitch * ((uint64_t)dimensions[1] / input->block[1] +
                       (dimensions[1] % input->block[1] != 0));
      if (input->layout != PAN_GRAPH_MEMORY_LAYOUT_INTERNAL_263)
         header |= (input->layout - 259) << 10;
   }
   uint64_t stride = plane * dimensions[2];
   if (pitch > INT32_MAX || plane > INT32_MAX || stride > INT32_MAX)
      return false;
   uint64_t size = stride * dimensions[3];
   if (size > INT32_MAX)
      return false;
   struct pan_graph_memory_descriptor value = {
      .words = {header,
                dimensions[3] | (dimensions[2] << 16),
                dimensions[1] | (dimensions[0] << 16),
                stride, plane, pitch, UINT32_MAX, UINT32_MAX},
      .size = size,
   };
   *result = value;
   return true;
}
