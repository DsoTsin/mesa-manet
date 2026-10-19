#include "pan_graph_memory_resource.h"

#include <assert.h>
#include <string.h>

int
main(void)
{
   struct pan_graph_memory_resource_tensor tensors[6] = {0};
   for (unsigned i = 0; i < 6; i++) {
      tensors[i].descriptor = (struct pan_graph_memory_descriptor_input){
         .format = PAN_GRAPH_MEMORY_DESCRIPTOR_I32,
         .layout = PAN_GRAPH_MEMORY_LAYOUT_INTERNAL_263,
         .dimensions = {33, 17, 2, 1},
         .block = {4, 4, 1, 1},
         .rank = 3,
      };
      tensors[i].uid = i + 11;
      tensors[i].allocation_offset = i * 64;
      tensors[i].ordinal = 1 - i % 2;
      tensors[i].domain = i < 2 ? PAN_GRAPH_MEMORY_RESOURCE_INPUT :
                          i < 4 ? PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT :
                                  PAN_GRAPH_MEMORY_RESOURCE_OUTPUT;
      for (unsigned j = 0; j < 3; j++)
         tensors[i].binding[j] = (i + 1) * 16 + j;
   }
   uint32_t references[] = {0, 1, 2, 2, 3, 4, 4, 2, 3, 0, 5, 4};
   struct pan_graph_memory_resource_input input = {
      tensors, references, 6, 12, 2, 2,
   };
   struct pan_graph_memory_resources result = {0};
   assert(pan_graph_memory_resources_create(&input, &result) == PAN_GRAPH_MEMORY_RESOURCE_SUCCESS);
   assert(result.tensor_count == 6 && result.domains[0].count == 2 &&
          result.domains[1].count == 2 && result.domains[2].count == 0 &&
          result.domains[3].count == 2);
   assert(result.references[0].slot32 == 2 && result.references[1].slot32 == 0);
   assert(result.references[2].slot32 == 0 && result.references[3].slot32 == 2);
   assert(result.domains[1].entries[0].descriptor.words[6] == 128);
   assert(result.domains[1].entries[1].descriptor.words[6] == 192);
   assert(result.domains[0].entries[0].uid == 12);
   assert(result.domains[3].entries[0].uid == 16);
   assert(result.domains[1].entries[0].binding[0] == UINT32_MAX);
   struct pan_graph_memory_resources keep = result;
   tensors[1].ordinal = 1;
   assert(pan_graph_memory_resources_create(&input, &result) == PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED);
   assert(!memcmp(&result, &keep, sizeof(result)));
   tensors[1].ordinal = 0;
   tensors[2].allocation_offset++;
   assert(pan_graph_memory_resources_create(&input, &result) == PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED);
   assert(!memcmp(&result, &keep, sizeof(result)));
   tensors[2].allocation_offset--;
   tensors[2].allocation_offset = UINT64_MAX - 63;
   assert(pan_graph_memory_resources_create(&input, &result) == PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED);
   tensors[2].allocation_offset = 128;
   tensors[2].domain = PAN_GRAPH_MEMORY_RESOURCE_CONSTANT;
   assert(pan_graph_memory_resources_create(&input, &result) == PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED);
   tensors[2].domain = PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT;
   references[1] = 6;
   assert(pan_graph_memory_resources_create(&input, &result) == PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED);
   references[1] = 1;
   tensors[0].descriptor.flags = PAN_GRAPH_MEMORY_DESCRIPTOR_EXPLICIT_STRIDES;
   assert(pan_graph_memory_resources_create(&input, &result) == PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED);
   tensors[0].descriptor.flags = 0;
   input.input_count = 1;
   assert(pan_graph_memory_resources_create(&input, &result) == PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED);
   assert(!memcmp(&result, &keep, sizeof(result)));
   pan_graph_memory_resources_cleanup(&result);
   pan_graph_memory_resources_cleanup(&result);
   assert(!result.storage && !result.references && !result.tensor_count);
   input.reference_count = input.input_count = input.output_count = 0;
   assert(pan_graph_memory_resources_create(&input, &result) == PAN_GRAPH_MEMORY_RESOURCE_SUCCESS);
   assert(!result.storage && result.references && result.references[5].slot32 == UINT32_MAX);
   pan_graph_memory_resources_cleanup(&result);
   input = (struct pan_graph_memory_resource_input){0};
   assert(pan_graph_memory_resources_create(&input, &result) == PAN_GRAPH_MEMORY_RESOURCE_SUCCESS);
   assert(pan_graph_memory_resources_create(NULL, &result) == PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED);
   assert(pan_graph_memory_resources_create(&input, NULL) == PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED);
   return 0;
}
