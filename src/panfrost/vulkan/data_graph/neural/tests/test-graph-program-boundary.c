#include "pan_graph_program_boundary.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TEST_BOUNDARY_ALLOC_FAILURES
static size_t allocation_count, allocation_failure;
void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void *__wrap_malloc(size_t size)
{
   if (++allocation_count == allocation_failure)
      return NULL;
   return __real_malloc(size);
}
void *__wrap_calloc(size_t count, size_t size)
{
   if (++allocation_count == allocation_failure)
      return NULL;
   return __real_calloc(count, size);
}
#endif

static void
chain(struct pan_graph_program_boundary_group *groups, uint32_t *kinds,
      uint32_t count)
{
   kinds[0] = kinds[1] = 3;
   for (uint32_t i = 0; i < count; i++) {
      kinds[i + 2] = i + 1 == count ? 4 : 1;
      groups[i] = (struct pan_graph_program_boundary_group){
         .kind = i == 0 ? PAN_GRAPH_PROGRAM_BOUNDARY_WEIGHT :
                 i == 1 ? PAN_GRAPH_PROGRAM_BOUNDARY_CE : PAN_GRAPH_PROGRAM_BOUNDARY_ADD,
         .inputs = {{i == 0 ? 0 : i == 1 ? 1 : i + 1, 1, 3 * i},
                    {i == 1 ? 2 : i + 1, i == 1 ? 3 : 257, 3 * i + 1}},
         .output = {i + 2, 2, 3 * i + 2},
      };
   }
}

static void
reject(const struct pan_graph_program_boundary_input *input)
{
   struct pan_graph_program_boundary result, before;
   memset(&result, 0xa5, sizeof(result));
   before = result;
   assert(pan_graph_program_boundary_matmul_add_v15(input, &result) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&result, &before, sizeof(result)));
}

int main(void)
{
   struct pan_graph_program_boundary_group groups[72];
   uint32_t kinds[74];
   chain(groups, kinds, 4);
   uint64_t selected[2] = {3, 0};
   struct pan_graph_program_boundary_input input = {kinds, groups, selected, 6, 4};
   struct pan_graph_program_boundary result = {0};
   assert(pan_graph_program_boundary_matmul_add_v15(&input, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.root == 1 && result.input_count == 2 && result.output_count == 1);
   assert(result.inputs[0].tensor == 1 && result.inputs[0].key == 1 && result.inputs[0].origin == 3);
   assert(result.inputs[1].tensor == 0 && result.inputs[1].key == 3 && result.inputs[1].origin == 0);
   assert(result.outputs[0].tensor == 3 && result.outputs[0].key == 2);
   pan_graph_program_boundary_finish(&result);
   assert(!result.inputs && !result.outputs && !result.input_count && !result.output_count);
   selected[0] = 7;
   for (unsigned i = 0; i < 16; i++) {
      assert(pan_graph_program_boundary_matmul_add_v15(&input, &result) == PAN_GRAPH_TSU_SUCCESS);
      assert(result.root == 1 && result.input_count == 2 && result.output_count == 2);
      assert(result.inputs[0].tensor == 1 && result.inputs[1].tensor == 0);
      assert(result.outputs[0].key == 2 && result.outputs[1].key == 258);
      assert(result.outputs[0].origin == 8 && result.outputs[1].origin == 8);
      pan_graph_program_boundary_finish(&result);
   }
   selected[0] = 4;
   assert(pan_graph_program_boundary_matmul_add_v15(&input, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.root == 2 && result.input_count == 2 && result.output_count == 1);
   assert(result.inputs[0].key == 1 && result.inputs[1].key == 257);
   pan_graph_program_boundary_finish(&result);
   selected[0] = 12;
   assert(pan_graph_program_boundary_matmul_add_v15(&input, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.root == 3 && result.input_count == 4 && result.output_count == 1);
   for (unsigned i = 0; i < 4; i++) {
      assert(result.inputs[i].key == 1 + 256 * i && result.inputs[i].tensor == 3);
      assert(result.inputs[i].origin == 6 + i % 2);
   }
   pan_graph_program_boundary_finish(&result);
   selected[0] = 15;
   assert(pan_graph_program_boundary_matmul_add_v15(&input, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.root == 1 && result.input_count == 2 && result.output_count == 4);
   pan_graph_program_boundary_finish(&result);

   selected[0] = 1;
   reject(&input);
   selected[0] = 2;
   reject(&input);
   selected[0] = 0;
   reject(&input);
   selected[0] = 16;
   reject(&input);
   selected[0] = 3;
   groups[2].kind = PAN_GRAPH_PROGRAM_BOUNDARY_CE;
   reject(&input);
   chain(groups, kinds, 4);
   groups[2].inputs[1].key = 1;
   reject(&input);
   chain(groups, kinds, 4);
   groups[2].inputs[0].tensor = 6;
   reject(&input);
   chain(groups, kinds, 4);
   groups[2].inputs[0].tensor = 5;
   reject(&input);
   chain(groups, kinds, 4);
   groups[2].output.tensor = 3;
   reject(&input);
   chain(groups, kinds, 4);
   kinds[3] = 0;
   reject(&input);
   chain(groups, kinds, 4);
   kinds[0] = 2;
   reject(&input);
   chain(groups, kinds, 4);
   kinds[3] = 3;
   reject(&input);
   reject(NULL);

   chain(groups, kinds, 6);
   input.tensor_count = 8;
   input.group_count = 6;
   selected[0] = 20;
   reject(&input);
   chain(groups, kinds, 72);
   input.tensor_count = 74;
   input.group_count = 72;
   selected[0] = 0;
   selected[1] = UINT64_C(1) << 7;
   assert(pan_graph_program_boundary_matmul_add_v15(&input, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.root == 71 && result.input_count == 2 && result.output_count == 1);
   assert(result.inputs[0].origin == 213 && result.inputs[1].origin == 214);
   pan_graph_program_boundary_finish(&result);
   selected[1] |= UINT64_C(1) << 8;
   reject(&input);

#ifdef TEST_BOUNDARY_ALLOC_FAILURES
   chain(groups, kinds, 7);
   input.tensor_count = 9;
   input.group_count = 7;
   selected[0] = 127;
   allocation_count = 0;
   assert(pan_graph_program_boundary_matmul_add_v15(&input, &result) == PAN_GRAPH_TSU_SUCCESS);
   size_t total = allocation_count;
   assert(result.input_count == 2 && result.output_count == 32 && total >= 8);
   pan_graph_program_boundary_finish(&result);
   for (size_t failure = 1; failure <= total; failure++) {
      allocation_count = 0;
      allocation_failure = failure;
      memset(&result, 0xa5, sizeof(result));
      struct pan_graph_program_boundary before = result;
      assert(pan_graph_program_boundary_matmul_add_v15(&input, &result) == PAN_GRAPH_TSU_OUT_OF_MEMORY);
      assert(!memcmp(&result, &before, sizeof(result)));
   }
   allocation_failure = 0;
   printf("allocation_failures=%zu\n", total);
#endif
   pan_graph_program_boundary_finish(NULL);
   return 0;
}
