#include "pan_graph_memory_lifetime.h"

#include <assert.h>
#include <string.h>

int
main(void)
{
   struct pan_graph_memory_lifetime_tensor tensors[] = {
      {11, 65, 1}, {11, 1000, 1}, {17, 33, 1}, {3, 0, 3}, {5, 0, 4}, {21, 100, 2},
   };
   uint32_t references[] = {3, 0, 0, 1, 5, 2, 4, 1, 2, 1};
   struct pan_graph_memory_lifetime_program programs[] = {{0, 0, 7}, {2, 7, 3}};
   struct pan_graph_memory_lifetime_input input = {tensors, programs, references, 6, 2, 10, 3};
   struct pan_graph_memory_lifetimes result = {0};
   assert(pan_graph_memory_lifetimes_create(&input, &result) == PAN_GRAPH_MEMORY_LIFETIME_SUCCESS);
   assert(result.group_count == 2 && result.groups[0].uid == 11 && result.groups[0].first_identity == 0);
   assert(result.groups[0].alias_count == 2 && result.requests[0].size == 65);
   assert(result.requests[0].first == 0 && result.requests[0].last == 5);
   assert(result.requests[1].first == 0 && result.requests[1].last == 5);
   assert(result.identities[0].offset == result.identities[1].offset);
   assert(result.identities[2].offset != result.identities[0].offset);
   assert(result.identities[3].group == UINT32_MAX && result.identities[5].offset == UINT64_MAX);
   assert(result.stats.peak == 192);
   assert(result.end_clock == 6);
   struct pan_graph_memory_lifetimes keep = result;
   programs[1].outer_group = 3;
   assert(pan_graph_memory_lifetimes_create(&input, &result) == PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED);
   assert(!memcmp(&result, &keep, sizeof(result)));
   programs[1].outer_group = 2;
   programs[0].reference_count = UINT32_MAX;
   assert(pan_graph_memory_lifetimes_create(&input, &result) == PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED);
   programs[0].reference_count = 7;
   references[0] = 6;
   assert(pan_graph_memory_lifetimes_create(&input, &result) == PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED);
   references[0] = 3;
   tensors[0].storage_size = UINT64_MAX;
   assert(pan_graph_memory_lifetimes_create(&input, &result) == PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED);
   tensors[0].storage_size = 65;
   tensors[1].storage_size = 0;
   struct pan_graph_memory_lifetimes second = {0};
   assert(pan_graph_memory_lifetimes_create(&input, &second) == PAN_GRAPH_MEMORY_LIFETIME_SUCCESS);
   assert(second.requests[0].size == 65);
   pan_graph_memory_lifetimes_cleanup(&second);
   programs[1].outer_group = 0;
   assert(pan_graph_memory_lifetimes_create(&input, &second) == PAN_GRAPH_MEMORY_LIFETIME_SUCCESS);
   assert(second.requests[0].last == 1);
   pan_graph_memory_lifetimes_cleanup(&second);
   programs[0].outer_group = 2;
   assert(pan_graph_memory_lifetimes_create(&input, &second) == PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED);
   programs[0].outer_group = 0;
   programs[1].outer_group = 2;
   tensors[1].kind = 0;
   assert(pan_graph_memory_lifetimes_create(&input, &result) == PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED);
   tensors[1].kind = 1;
   input.outer_group_count = UINT32_MAX;
   assert(pan_graph_memory_lifetimes_create(&input, &result) == PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED);
   assert(!memcmp(&result, &keep, sizeof(result)));
   pan_graph_memory_lifetimes_cleanup(&result);
   pan_graph_memory_lifetimes_cleanup(&result);
   input = (struct pan_graph_memory_lifetime_input){0};
   assert(pan_graph_memory_lifetimes_create(&input, &result) == PAN_GRAPH_MEMORY_LIFETIME_SUCCESS);
   assert(result.stats.peak == 0 && result.stats.iterations == 1);
   assert(pan_graph_memory_lifetimes_create(NULL, &result) == PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED);
   assert(pan_graph_memory_lifetimes_create(&input, NULL) == PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED);
   return 0;
}
