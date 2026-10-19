#include "panvk_data_graph.h"
#include "panvk_data_graph_neural.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#ifdef PANVK_GRAPH_TEST_ALLOC_FAILURE
static size_t allocations, fail_at = SIZE_MAX;
void *__real_malloc(size_t size);
void *
__wrap_malloc(size_t size)
{
   return allocations++ == fail_at ? NULL : __real_malloc(size);
}
#endif

static void
reject(const struct panvk_data_graph *graph)
{
   struct panvk_data_graph_matmul_add result, before;
   memset(&result, 0xa5, sizeof(result));
   before = result;
   assert(panvk_data_graph_matmul_add_extract(graph, &result) ==
          VK_ERROR_FEATURE_NOT_PRESENT);
   assert(!memcmp(&result, &before, sizeof(result)));
}

int
main(void)
{
   int8_t zeros[2] = {3, -2};
   struct pan_graph_matmul_plan plan = {
      .info = {2, 3, 5, 4, 3, -2},
   };
   struct panvk_graph_value values[8] = {
      {.kind = PANVK_GRAPH_INPUT,
       .format = VK_FORMAT_R8_SINT,
       .rank = 3,
       .dims = {2, 3, 5}},
      {.kind = PANVK_GRAPH_INPUT,
       .format = VK_FORMAT_R8_SINT,
       .rank = 3,
       .resource_index = 1,
       .dims = {2, 5, 4}},
      {.kind = PANVK_GRAPH_CONSTANT,
       .format = VK_FORMAT_R8_SINT,
       .rank = 1,
       .dims = {1},
       .elements = 1,
       .size = 1,
       .data = zeros},
      {.kind = PANVK_GRAPH_CONSTANT,
       .format = VK_FORMAT_R8_SINT,
       .rank = 1,
       .dims = {1},
       .elements = 1,
       .size = 1,
       .data = zeros + 1},
      {.kind = PANVK_GRAPH_OPERATION,
       .format = VK_FORMAT_R32_SINT,
       .rank = 3,
       .dims = {2, 3, 4},
       .op = PANVK_GRAPH_OP_MATMUL,
       .inputs = {0, 1, 2, 3},
       .input_count = 4,
       .matmul = &plan},
      {.kind = PANVK_GRAPH_OPERATION,
       .format = VK_FORMAT_R32_SINT,
       .rank = 3,
       .dims = {2, 3, 4},
       .op = PANVK_GRAPH_OP_ADD,
       .inputs = {4, 4},
       .input_count = 2,
       .fused_into = 7},
      {.kind = PANVK_GRAPH_OPERATION,
       .format = VK_FORMAT_R32_SINT,
       .rank = 3,
       .dims = {2, 3, 4},
       .op = PANVK_GRAPH_OP_ADD,
       .inputs = {5, 4},
       .input_count = 2,
       .fused_into = 7},
      {.kind = PANVK_GRAPH_OUTPUT,
       .format = VK_FORMAT_R32_SINT,
       .rank = 3,
       .dims = {2, 3, 4},
       .inputs = {6},
       .input_count = 1},
   };
   uint32_t passes[2] = {4, 7};
   for (unsigned i = 0; i < 8; i++)
      values[i].live = true;
   struct panvk_data_graph graph = {.count = 8,
                                    .pass_count = 2,
                                    .passes = passes,
                                    .values = values};
   struct panvk_data_graph_matmul_add result = {0};
#ifdef PANVK_GRAPH_TEST_ALLOC_FAILURE
   allocations = 0;
#endif
   assert(panvk_data_graph_matmul_add_extract(&graph, &result) == VK_SUCCESS);
   assert(!memcmp(&result.info, &plan.info, sizeof(plan.info)));
   assert(result.add_count == 2 && result.output == 7);
   assert(result.inputs[0] == 0 && result.inputs[1] == 1);
   assert(result.input_indices[0] == 0 && result.input_indices[1] == 1);
   assert(result.values[0] == 4 && result.values[1] == 5 &&
          result.values[2] == 6);
   assert(result.adds[0].input[0] == 0 && result.adds[0].input[1] == 0);
   assert(result.adds[1].input[0] == 1 && result.adds[1].input[1] == 0);
#ifdef PANVK_GRAPH_TEST_ALLOC_FAILURE
   size_t total = allocations;
   assert(total == 3);
   for (size_t i = 0; i < total; i++) {
      struct panvk_data_graph_matmul_add rejected, before;
      memset(&rejected, 0xa5, sizeof(rejected));
      before = rejected;
      allocations = 0;
      fail_at = i;
      assert(panvk_data_graph_matmul_add_extract(&graph, &rejected) ==
             VK_ERROR_OUT_OF_HOST_MEMORY);
      assert(!memcmp(&rejected, &before, sizeof(before)));
   }
   fail_at = SIZE_MAX;
#endif
   panvk_data_graph_matmul_add_finish(&result);
   assert(!result.values && !result.adds && !result.add_count);
   struct panvk_graph_value original[8];
   memcpy(original, values, sizeof(values));
   const uint32_t bad_inputs[] = {0, 2, 6, 7, 8, UINT32_MAX};
   for (unsigned i = 0; i < sizeof(bad_inputs) / sizeof(bad_inputs[0]); i++) {
      values[5].inputs[0] = bad_inputs[i];
      reject(&graph);
      memcpy(values, original, sizeof(values));
   }
   values[5].op = PANVK_GRAPH_OP_SUB;
   reject(&graph);
   memcpy(values, original, sizeof(values));
   values[6].op = PANVK_GRAPH_OP_MATMUL;
   reject(&graph);
   memcpy(values, original, sizeof(values));
   values[4].matmul = NULL;
   reject(&graph);
   memcpy(values, original, sizeof(values));
   values[5].format = VK_FORMAT_R32_SFLOAT;
   reject(&graph);
   memcpy(values, original, sizeof(values));
   values[5].dims[2] = 1;
   reject(&graph);
   memcpy(values, original, sizeof(values));
   values[7].inputs[0] = 5;
   reject(&graph);
   memcpy(values, original, sizeof(values));
   values[6].inputs[0] = 4;
   reject(&graph);
   memcpy(values, original, sizeof(values));
   values[5].kind = PANVK_GRAPH_OUTPUT;
   reject(&graph);
   memcpy(values, original, sizeof(values));
   values[3].kind = PANVK_GRAPH_INPUT;
   reject(&graph);
   memcpy(values, original, sizeof(values));
   values[3].kind = 99;
   reject(&graph);
   memcpy(values, original, sizeof(values));
   zeros[0] = 4;
   reject(&graph);
   zeros[0] = 3;
   reject(NULL);
   struct panvk_data_graph empty = {0};
   reject(&empty);
   graph.count = 6;
   values[5] = original[7];
   values[5].inputs[0] = 4;
   assert(panvk_data_graph_matmul_add_extract(&graph, &result) == VK_SUCCESS);
   assert(!result.add_count && !result.adds && result.values[0] == 4);
   assert(result.output == 5);
   panvk_data_graph_matmul_add_finish(&result);
   struct panvk_graph_value with_dead[10];
   memcpy(with_dead, original, sizeof(original));
   with_dead[8] = (struct panvk_graph_value){.kind = PANVK_GRAPH_INPUT};
   with_dead[9] = (struct panvk_graph_value){.kind = PANVK_GRAPH_OPERATION,
                                             .op = PANVK_GRAPH_OP_ADD};
   graph.values = with_dead;
   graph.count = ARRAY_SIZE(with_dead);
   assert(panvk_data_graph_matmul_add_extract(&graph, &result) == VK_SUCCESS);
   assert(result.add_count == 2 && result.output == 7);
   panvk_data_graph_matmul_add_finish(&result);
   return 0;
}
