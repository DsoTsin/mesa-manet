#include "pan_graph_program_view.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TEST_PROGRAM_VIEW_OOM
static size_t allocation_count, fail_allocation;
void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void *__wrap_malloc(size_t size)
{
   return ++allocation_count == fail_allocation ? NULL : __real_malloc(size);
}
void *__wrap_calloc(size_t count, size_t size)
{
   return ++allocation_count == fail_allocation ? NULL
                                               : __real_calloc(count, size);
}
#endif

static void
reject(const struct pan_graph_matmul_info *info,
       const struct pan_graph_matmul_add *adds, uint32_t count,
       const uint64_t *selected)
{
   struct pan_graph_program_views result;
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_program_views before = result;
   assert(pan_graph_program_view_matmul_add_v15(info, adds, count, selected,
                                             &result) != PAN_GRAPH_TSU_SUCCESS);
   assert(!memcmp(&result, &before, sizeof(result)));
}

int
main(void)
{
   const uint32_t shape[] = {4, 3, 2};
   struct pan_graph_program_view_bounds bounds;
   assert(pan_graph_program_view_bounds_init(3, shape, NULL, &bounds) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(bounds.rank == 3);
   for (unsigned i = 0; i < 7; i++)
      assert(bounds.low[i] == 0 &&
             bounds.high[i] == (i < 3 ? (int32_t)shape[i] : 0));
   struct pan_graph_program_view_bounds explicit_bounds = {
      .rank = 7,
      .low = {INT32_MIN, -2, 0, 1, 2, 3, 4},
      .high = {INT32_MAX, 0, 1, 2, 3, 4, 5},
   };
   assert(pan_graph_program_view_bounds_init(0, NULL, &explicit_bounds, &bounds) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(!memcmp(&bounds, &explicit_bounds, sizeof(bounds)));
   struct pan_graph_program_view_bounds before = bounds;
   explicit_bounds.low[1] = 1;
   assert(pan_graph_program_view_bounds_init(0, NULL, &explicit_bounds, &bounds) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&bounds, &before, sizeof(bounds)));
   const uint32_t invalid[] = {0, UINT32_MAX};
   for (unsigned i = 0; i < 2; i++) {
      assert(pan_graph_program_view_bounds_init(1, invalid + i, NULL, &bounds) ==
             PAN_GRAPH_TSU_UNSUPPORTED);
      assert(!memcmp(&bounds, &before, sizeof(bounds)));
   }
   assert(pan_graph_program_view_bounds_init(8, shape, NULL, &bounds) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(pan_graph_program_view_bounds_init(3, NULL, NULL, &bounds) ==
          PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_program_view_bounds_init(3, shape, NULL, NULL) ==
          PAN_GRAPH_TSU_INVALID);

   struct pan_graph_matmul_info info = {2, 3, 5, 4, 0, 0};
   struct pan_graph_matmul_add adds[] = {{{0, 0}}, {{1, 1}}};
   uint64_t selected = 3;
   struct pan_graph_program_views result = {0};
   assert(pan_graph_program_view_matmul_add_v15(&info, adds, 2, &selected,
                                              &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.root == 1 && result.input_count == 2 && result.output_count == 1);
   assert(result.inputs[0].tensor == 4 && result.inputs[0].key == 1);
   assert(result.inputs[0].origin.group == 1 && !result.inputs[0].origin.output &&
          result.inputs[0].origin.index == 0);
   assert(result.inputs[0].bounds.high[0] == 5);
   assert(result.inputs[1].tensor == 5 && result.inputs[1].key == 3);
   assert(result.inputs[1].origin.group == 0);
   assert(result.outputs[0].tensor == 0 && result.outputs[0].origin.group == 1 &&
          result.outputs[0].origin.output);
   pan_graph_program_views_finish(&result);
   selected = 7;
   assert(pan_graph_program_view_matmul_add_v15(&info, adds, 2, &selected,
                                              &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.output_count == 2);
   for (size_t i = 0; i < result.output_count; i++) {
      assert(result.outputs[i].tensor == 1 && result.outputs[i].key == 2 + 256*i);
      assert(result.outputs[i].origin.group == 2 && result.outputs[i].origin.output);
      assert(result.outputs[i].origin.index == 0);
   }
   pan_graph_program_views_finish(&result);
   selected = 12;
   assert(pan_graph_program_view_matmul_add_v15(&info, adds, 2, &selected,
                                              &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.root == 3 && result.input_count == 4);
   for (size_t i = 0; i < result.input_count; i++) {
      assert(result.inputs[i].tensor == 0 && result.inputs[i].key == 1 + 256*i);
      assert(result.inputs[i].origin.group == 2 && !result.inputs[i].origin.output);
   }
   pan_graph_program_views_finish(&result);
   pan_graph_program_views_finish(&result);
   pan_graph_program_views_finish(NULL);
   selected = 0;
   reject(&info, adds, 2, &selected);
   selected = 2;
   reject(&info, adds, 2, &selected);
   selected = 16;
   reject(&info, adds, 2, &selected);
   selected = 7;
   reject(NULL, adds, 2, &selected);
   reject(&info, NULL, 2, &selected);
   reject(&info, adds, UINT32_MAX, &selected);
   reject(&info, adds, 2, NULL);
   adds[0].input[1] = 1;
   reject(&info, adds, 2, &selected);
   adds[0].input[1] = 0;
   struct pan_graph_matmul_info invalid_info = info;
   invalid_info.batch = 0;
   reject(&invalid_info, adds, 2, &selected);
   invalid_info = info;
   invalid_info.rows = UINT32_MAX;
   reject(&invalid_info, adds, 2, &selected);
   assert(pan_graph_program_view_matmul_add_v15(&info, adds, 2, &selected, NULL) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   struct pan_graph_matmul_add wide_adds[70] = {0};
   const uint64_t wide_selected[] = {0, UINT64_C(1) << 7};
   assert(pan_graph_program_view_matmul_add_v15(&info, wide_adds, 70,
                                              wide_selected, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.inputs[0].origin.group == 71 && result.inputs[0].tensor == 0);
   assert(result.outputs[0].tensor == 70 && result.outputs[0].kind == 4);
   pan_graph_program_views_finish(&result);
#ifdef TEST_PROGRAM_VIEW_OOM
   allocation_count = fail_allocation = 0;
   assert(pan_graph_program_view_matmul_add_v15(&info, adds, 2, &selected,
                                              &result) == PAN_GRAPH_TSU_SUCCESS);
   size_t allocations = allocation_count;
   assert(allocations > 0);
   pan_graph_program_views_finish(&result);
   for (size_t i = 1; i <= allocations; i++) {
      allocation_count = 0;
      fail_allocation = i;
      struct pan_graph_program_views failed;
      memset(&failed, 0xa5, sizeof(failed));
      struct pan_graph_program_views original = failed;
      assert(pan_graph_program_view_matmul_add_v15(&info, adds, 2, &selected,
                                                 &failed) ==
             PAN_GRAPH_TSU_OUT_OF_MEMORY);
      assert(!memcmp(&failed, &original, sizeof(failed)));
   }
   fail_allocation = 0;
   printf("allocation failures=%zu\n", allocations);
#endif
   return 0;
}
