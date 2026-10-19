#include "pan_graph_alloc.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

int
main(void)
{
   const struct pan_graph_alloc_request requests[] = {
      {5, 13, 156}, {1, 5, 336}, {12, 21, 84}, {11, 19, 316},
   };
   const uint64_t offsets[] = {0, 192, 512, 192};
   const uint32_t positions[] = {0, 1, 3, 2};
   struct pan_graph_allocation allocations[4];
   struct pan_graph_alloc_stats stats;
   assert(pan_graph_allocate(requests, 4, 64, 500, allocations, &stats) == PAN_GRAPH_ALLOC_SUCCESS);
   assert(stats.peak == 640 && stats.lower_bound == 640 && stats.iterations == 3);
   for (unsigned i = 0; i < 4; i++) {
      assert(allocations[i].offset == offsets[i] && allocations[i].position == positions[i]);
      if (allocations[i].predecessor >= 0)
         assert(allocations[allocations[i].predecessor].position < allocations[i].position);
   }
   assert(pan_graph_allocate(requests, 4, 64, 0, allocations, &stats) == PAN_GRAPH_ALLOC_SUCCESS);
   assert(stats.peak > 640 && stats.lower_bound == 640 && stats.iterations == 1);
   const struct pan_graph_alloc_request adjacent[] = {{0, 1, 1}, {1, 2, 65}};
   assert(pan_graph_allocate(adjacent, 2, 64, 500, allocations, &stats) == PAN_GRAPH_ALLOC_SUCCESS);
   assert(stats.peak == 192 && allocations[0].offset == 0 && allocations[1].offset == 64);
   const struct pan_graph_alloc_request disjoint[] = {{0, 1, 1}, {2, 3, 65}};
   assert(pan_graph_allocate(disjoint, 2, 64, 500, allocations, &stats) == PAN_GRAPH_ALLOC_SUCCESS);
   assert(stats.peak == 128 && allocations[0].offset == 0 && allocations[1].offset == 0);
   const struct pan_graph_alloc_request zero = {0, 0, 0};
   assert(pan_graph_allocate(&zero, 1, 64, 500, allocations, &stats) == PAN_GRAPH_ALLOC_SUCCESS);
   assert(stats.peak == 0 && stats.lower_bound == 0 && stats.iterations == 1);
   assert(pan_graph_allocate(NULL, 0, 64, 500, NULL, &stats) == PAN_GRAPH_ALLOC_SUCCESS);
   assert(stats.peak == 0 && stats.lower_bound == 0 && stats.iterations == 1);
   unsigned char allocation_guard[sizeof(allocations)], stats_guard[sizeof(stats)];
   memset(allocations, 0xa5, sizeof(allocations));
   memset(&stats, 0x5a, sizeof(stats));
   memcpy(allocation_guard, allocations, sizeof(allocations));
   memcpy(stats_guard, &stats, sizeof(stats));
   const struct pan_graph_alloc_request invalid[] = {{1, 0, 64}, {0, 1, UINT64_MAX}};
   for (unsigned i = 0; i < 2; i++) {
      assert(pan_graph_allocate(&invalid[i], 1, 64, 500, allocations, &stats) == PAN_GRAPH_ALLOC_INVALID);
      assert(!memcmp(allocation_guard, allocations, sizeof(allocations)) &&
             !memcmp(stats_guard, &stats, sizeof(stats)));
   }
   const struct pan_graph_alloc_request overflow[] = {{0, 1, UINT64_C(1) << 63}, {0, 1, UINT64_C(1) << 63}};
   assert(pan_graph_allocate(overflow, 2, 64, 500, allocations, &stats) == PAN_GRAPH_ALLOC_INVALID);
   assert(pan_graph_allocate(requests, 4, 3, 500, allocations, &stats) == PAN_GRAPH_ALLOC_INVALID);
   assert(pan_graph_allocate(requests, 4, 0, 500, allocations, &stats) == PAN_GRAPH_ALLOC_INVALID);
   assert(!memcmp(allocation_guard, allocations, sizeof(allocations)) &&
          !memcmp(stats_guard, &stats, sizeof(stats)));
   return 0;
}
