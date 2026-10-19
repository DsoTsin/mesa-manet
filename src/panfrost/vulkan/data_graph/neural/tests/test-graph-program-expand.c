#include "pan_graph_program_expand.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#ifdef TEST_EXPAND_ALLOC_FAILURES
static size_t allocation_count, allocation_failure;
void *__real_calloc(size_t count, size_t size);
void *__wrap_calloc(size_t count, size_t size)
{
   if (++allocation_count == allocation_failure)
      return NULL;
   return __real_calloc(count, size);
}
#endif

int main(void)
{
   const struct pan_graph_matmul_add adds[] = {{{0, 0}}, {{1, 1}}};
   struct pan_graph_program_subsets result = {0};
   assert(pan_graph_program_expand_matmul_add_v15(NULL, 0, 8, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.group_count == 2 && result.count == 1 && result.groups[0] == 3);
   pan_graph_program_subsets_finish(&result);
   assert(pan_graph_program_expand_matmul_add_v15(adds, 2, 8, &result) == PAN_GRAPH_TSU_SUCCESS);
   const uint64_t expected[] = {3, 7, 4, 12, 8};
   assert(result.group_count == 4 && result.count == 5);
   assert(!memcmp(result.groups, expected, sizeof(expected)));
   pan_graph_program_subsets_finish(&result);
   const struct pan_graph_matmul_add mixed[] = {{{0, 0}}, {{0, 1}}};
   assert(pan_graph_program_candidates_matmul_add_v15(mixed, 2, 8, &result) == PAN_GRAPH_TSU_SUCCESS);
   const uint64_t filtered[] = {3, 4, 12, 8};
   assert(result.count == 4 && !memcmp(result.groups, filtered, sizeof(filtered)));
   pan_graph_program_subsets_finish(&result);
   const struct pan_graph_matmul_add diamond[] = {{{0, 0}}, {{0, 0}}, {{1, 2}}};
   assert(pan_graph_program_candidates_matmul_add_v15(diamond, 3, 8, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.count == 7);
   for (size_t i = 0; i < result.count; i++)
      assert(result.groups[i] != 7 && result.groups[i] != 11);
   pan_graph_program_subsets_finish(&result);
   struct pan_graph_matmul_add chain[70];
   for (unsigned i = 0; i < 70; i++)
      chain[i] = (struct pan_graph_matmul_add){{i, i}};
#ifdef TEST_EXPAND_ALLOC_FAILURES
   allocation_count = 0;
   assert(pan_graph_program_unique_candidates_matmul_add_v15(chain, 70, 8, &result) == PAN_GRAPH_TSU_SUCCESS);
   size_t total = allocation_count;
   pan_graph_program_subsets_finish(&result);
   assert(total >= 9);
   for (size_t failure = 1; failure <= total; failure++) {
      allocation_count = 0;
      allocation_failure = failure;
      memset(&result, 0xa5, sizeof(result));
      struct pan_graph_program_subsets before = result;
      assert(pan_graph_program_unique_candidates_matmul_add_v15(chain, 70, 8, &result) == PAN_GRAPH_TSU_OUT_OF_MEMORY);
      assert(!memcmp(&result, &before, sizeof(result)));
   }
   allocation_failure = 0;
#endif
   assert(pan_graph_program_expand_matmul_add_v15(chain, 70, 8, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.group_count == 72 && result.count > 70);
   bool wide = false;
   for (size_t i = 0; i < result.count; i++)
      wide |= result.groups[2 * i + 1] != 0;
   assert(wide);
   pan_graph_program_subsets_finish(&result);
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_program_subsets before = result;
   chain[0].input[0] = 1;
   assert(pan_graph_program_expand_matmul_add_v15(chain, 70, 8, &result) == PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_program_expand_matmul_add_v15(NULL, 1, 8, &result) == PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_program_expand_matmul_add_v15(NULL, 0, 0, &result) == PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_program_expand_matmul_add_v15(NULL, UINT32_MAX, 8, &result) == PAN_GRAPH_TSU_INVALID);
   const struct pan_graph_matmul_add dead[] = {{{0, 0}}, {{0, 0}}};
   assert(pan_graph_program_expand_matmul_add_v15(dead, 2, 8, &result) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&result, &before, sizeof(result)));
   return 0;
}
