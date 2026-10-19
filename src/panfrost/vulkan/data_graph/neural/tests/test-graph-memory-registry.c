#include "pan_graph_memory_registry.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static void
reject(const struct pan_graph_memory_registry_graph *graph, uint32_t tensor,
       uint32_t paired)
{
   struct pan_graph_memory_registry_result result, saved;
   memset(&result, 0xa5, sizeof(result));
   saved = result;
   assert(pan_graph_memory_registry_analyze(graph, tensor, paired, &result) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&saved, &result, sizeof(result)));
}

int
main(void)
{
   const struct pan_graph_memory_registry_shape shape = {3, {32, 17, 1}};
   struct pan_graph_memory_registry_tensor tensors[3];
   for (unsigned i = 0; i < 3; i++)
      tensors[i] = (struct pan_graph_memory_registry_tensor){
         .shape = shape, .bits = 32, .layout = 3,
      };
   struct pan_graph_memory_registry_view inputs[3] = {
      {.tensor = 0, .key = 1, .shape = shape},
      {.tensor = 1, .key = 1, .shape = shape},
      {.tensor = 1, .key = 257, .shape = shape},
   };
   struct pan_graph_memory_registry_view outputs[2] = {
      {.tensor = 1, .key = 2, .shape = shape},
      {.tensor = 2, .key = 2, .shape = shape},
   };
   struct pan_graph_memory_registry_operation ops[2] = {
      {PAN_GRAPH_MEMORY_REGISTRY_MATMUL, 1, 1, inputs, outputs},
      {PAN_GRAPH_MEMORY_REGISTRY_ADD, 2, 1, inputs + 1, outputs + 1},
   };
   struct pan_graph_memory_registry_graph graph = {3, 2, tensors, ops};
   struct pan_graph_memory_registry_result result;
   assert(pan_graph_memory_registry_analyze(&graph, 1, UINT32_MAX, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.producer_allowed && result.consumer_allowed && result.changed);
   assert(result.layout == 262 && result.block[0] == 4 && result.block[1] == 4);
   assert(!result.producer_flag);
   assert(tensors[1].layout == 3 && tensors[1].block[0] == 0);
   reject(NULL, 1, UINT32_MAX);
   reject(&graph, 3, UINT32_MAX);
   reject(&graph, 1, 3);
   reject(&graph, 1, 2);
   assert(pan_graph_memory_registry_analyze(&graph, 1, UINT32_MAX, NULL) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   ops[0].opcode = UINT32_MAX;
   reject(&graph, 1, UINT32_MAX);
   ops[0].opcode = PAN_GRAPH_MEMORY_REGISTRY_MATMUL;
   inputs[0].tensor = 3;
   reject(&graph, 1, UINT32_MAX);
   inputs[0].tensor = 0;
   inputs[2].key = 1;
   reject(&graph, 1, UINT32_MAX);
   inputs[2].key = 257;
   tensors[0].shape.rank = 9;
   reject(&graph, 1, UINT32_MAX);
   tensors[0].shape = shape;
   inputs[0].shape.dims[0] = 0;
   reject(&graph, 1, UINT32_MAX);
   inputs[0].shape = shape;
   tensors[0].bits = 16;
   reject(&graph, 1, UINT32_MAX);
   tensors[0].bits = 32;
   ops[0].opcode = PAN_GRAPH_MEMORY_REGISTRY_FUNCTION;
   outputs[0].attributes = 1;
   assert(pan_graph_memory_registry_analyze(&graph, 1, UINT32_MAX, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(!result.producer_allowed && result.consumer_allowed && !result.changed);
   assert(result.layout == 3);
   outputs[0].attributes = 0;
   ops[0].opcode = PAN_GRAPH_MEMORY_REGISTRY_CE_GROUP;
   assert(pan_graph_memory_registry_analyze(&graph, 1, UINT32_MAX, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.producer_flag);
   ops[0].opcode = PAN_GRAPH_MEMORY_REGISTRY_MATMUL;
   tensors[1].layout = 263;
   tensors[1].producer_flag = true;
   tensors[1].block[0] = 17;
   assert(pan_graph_memory_registry_analyze(&graph, 1, UINT32_MAX, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(!result.changed && result.layout == 263 && result.block[0] == 17 &&
          result.producer_flag);
   tensors[1].layout = 3;
   tensors[1].shape.rank = 5;
   tensors[1].shape.dims[3] = tensors[1].shape.dims[4] = 1;
   assert(pan_graph_memory_registry_analyze(&graph, 1, UINT32_MAX, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(!result.changed && result.layout == 3 && result.producer_flag);
   tensors[1].shape = shape;
   ops[0].opcode = ops[1].opcode = PAN_GRAPH_MEMORY_REGISTRY_RESHAPE;
   ops[1].input_count = 1;
   outputs[1].tensor = 0;
   reject(&graph, 1, UINT32_MAX);
   inputs[0].key = 257;
   reject(&graph, 1, UINT32_MAX);

   const unsigned length = 256;
   struct pan_graph_memory_registry_tensor *chain = calloc(length + 1, sizeof(*chain));
   struct pan_graph_memory_registry_operation *nodes = calloc(length, sizeof(*nodes));
   struct pan_graph_memory_registry_view *edges = calloc(2 * length, sizeof(*edges));
   assert(chain && nodes && edges);
   for (unsigned i = 0; i <= length; i++)
      chain[i] = (struct pan_graph_memory_registry_tensor){
         .shape = shape, .bits = 32, .layout = 3,
      };
   for (unsigned i = 0; i < length; i++) {
      edges[2 * i] = (struct pan_graph_memory_registry_view){shape, i, 1, 0};
      edges[2 * i + 1] = (struct pan_graph_memory_registry_view){shape, i + 1, 2, 0};
      nodes[i] = (struct pan_graph_memory_registry_operation){
         PAN_GRAPH_MEMORY_REGISTRY_RESHAPE, 1, 1, edges + 2 * i, edges + 2 * i + 1,
      };
   }
   struct pan_graph_memory_registry_graph long_graph = {length + 1, length, chain, nodes};
   assert(pan_graph_memory_registry_analyze(&long_graph, length, UINT32_MAX, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.producer_allowed && result.consumer_allowed && result.changed);
   free(edges);
   free(nodes);
   free(chain);
   return 0;
}
