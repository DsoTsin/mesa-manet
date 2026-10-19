#include "pan_graph_program_order.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TEST_PROGRAM_ORDER_ALLOC_FAILURES
static unsigned allocation_count, fail_at;
void *__real_malloc(size_t size);
void *__wrap_malloc(size_t size)
{
   if (++allocation_count == fail_at)
      return NULL;
   return __real_malloc(size);
}
#endif

static void
reject(const struct pan_graph_program_order_node *nodes, size_t count)
{
   struct pan_graph_program_order result;
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_program_order before = result;
   assert(pan_graph_program_order_build_dependencies(nodes, count, &result) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&result, &before, sizeof(result)));
   assert(pan_graph_program_order_build_selected(nodes, count, &result) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&result, &before, sizeof(result)));
}

int
main(void)
{
   const uint64_t dependencies[] = {9000, 30, 30};
   struct pan_graph_program_order_node nodes[] = {
      {10, dependencies, 3}, {20, NULL, 0}, {30, NULL, 0}, {40, NULL, 0},
   };
   struct pan_graph_program_order result = {0};
   assert(pan_graph_program_order_build_dependencies(nodes, 4, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   const uint32_t wanted[] = {1, 2, 0, 3};
   assert(result.count == 4 &&
          !memcmp(result.indices, wanted, sizeof(wanted)));
   pan_graph_program_order_finish(&result);
   assert(!result.indices && !result.count);
   assert(pan_graph_program_order_build_selected(nodes, 4, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   const uint32_t selected[] = {3, 2, 1, 0};
   assert(result.count == 4 &&
          !memcmp(result.indices, selected, sizeof(selected)));
   pan_graph_program_order_finish(&result);
   assert(pan_graph_program_order_build_selected(NULL, 0, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(!result.indices && !result.count);
   assert(pan_graph_program_order_build_dependencies(NULL, 0, NULL) ==
          PAN_GRAPH_TSU_INVALID);
   reject(NULL, 1);
   reject(nodes, UINT32_MAX);
   nodes[1].uid = nodes[0].uid;
   reject(nodes, 4);
   nodes[1].uid = 20;
   nodes[1].dependency_count = 1;
   reject(nodes, 4);
   nodes[1].dependency_count = (size_t)INT32_MAX + 1;
   reject(nodes, 4);
   nodes[1].dependencies = &nodes[0].uid;
   nodes[1].dependency_count = 1;
   nodes[0].dependencies = &nodes[1].uid;
   nodes[0].dependency_count = 1;
   reject(nodes, 4);
   nodes[0].dependencies = &nodes[0].uid;
   reject(nodes, 1);
   nodes[0] = (struct pan_graph_program_order_node){UINT64_MAX, NULL, 0};
   assert(pan_graph_program_order_build_selected(nodes, 1, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.count == 1 && result.indices[0] == 0);
   pan_graph_program_order_finish(&result);
   pan_graph_program_order_finish(NULL);
#ifdef TEST_PROGRAM_ORDER_ALLOC_FAILURES
   unsigned tested = 0;
   for (unsigned mode = 0; mode < 2; mode++) {
      for (unsigned failure = 1; failure <= 2; failure++) {
         allocation_count = 0;
         fail_at = failure;
         memset(&result, 0xa5, sizeof(result));
         struct pan_graph_program_order before = result;
         enum pan_graph_tsu_status status =
            mode ? pan_graph_program_order_build_selected(nodes, 1, &result)
                 : pan_graph_program_order_build_dependencies(nodes, 1, &result);
         assert(status == PAN_GRAPH_TSU_OUT_OF_MEMORY);
         assert(!memcmp(&result, &before, sizeof(result)));
         tested++;
      }
   }
   fail_at = 0;
   printf("allocation_failures=%u\n", tested);
#endif
   return 0;
}
