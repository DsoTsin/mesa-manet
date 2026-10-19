#include "pan_graph_nrt_plan.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void
requests_init(struct pan_graph_nrt_plan_request requests[5], uint32_t n,
              uint32_t m, uint32_t k)
{
   const struct pan_graph_nrt_plan_request values[5] = {
      {
         .allocation = {.dimensions = {n, k, 1, 1}, .element_bits = 8,
                        .buffers = 2, .front = true},
         .buffers = PAN_GRAPH_NRT_BUFFER_INPUT,
      },
      {
         .allocation = {.dimensions = {k, m, 1, 1}, .element_bits = 8,
                        .buffers = 2, .front = true},
         .buffers = PAN_GRAPH_NRT_BUFFER_INPUT,
      },
      {
         .allocation = {.dimensions = {n, m, 1, 1}, .element_bits = 32,
                        .buffers = 2},
         .buffers = PAN_GRAPH_NRT_BUFFER_OUTPUT,
      },
      {
         .allocation = {.type = PAN_GRAPH_NRT_WEIGHT,
                        .dimensions = {k, 1, 1, n}, .element_bits = 8,
                        .initial_format = 1, .weight_format = 1,
                        .buffers = 2, .narrow_weight = true},
      },
      {
         .allocation = {.dimensions = {n, m, 1, 1}, .element_bits = 32,
                        .buffers = 2},
      },
   };
   memcpy(requests, values, sizeof(values));
}

int
main(void)
{
   struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   struct pan_graph_nrt_plan_request requests[5];
   requests_init(requests, 4, 3, 5);
   struct pan_graph_nrt_plan plan;
   assert(pan_graph_nrt_plan_search(&target, requests, 5, 16, &plan) ==
          PAN_GRAPH_NRT_SUCCESS);
   const uint32_t full[] = {16, 16, 16};
   assert(!memcmp(plan.buffers, full, sizeof(full)));
   const uint32_t bases[] = {0, 1, 7, 6, 5};
   const uint32_t buffers[] = {16, 16, 16, 2, 2};
   for (unsigned i = 0; i < 5; i++) {
      assert(plan.layouts[i].base == bases[i]);
      assert(plan.layouts[i].buffers == buffers[i]);
   }
   requests_init(requests, 16, 16, 512);
   assert(pan_graph_nrt_plan_search(&target, requests, 5, 16, &plan) ==
          PAN_GRAPH_NRT_SUCCESS);
   const uint32_t limited[] = {2, 16, 3};
   assert(!memcmp(plan.buffers, limited, sizeof(limited)));
   const uint32_t limited_bases[] = {0, 2, 7, 5, 4};
   for (unsigned i = 0; i < 5; i++)
      assert(plan.layouts[i].base == limited_bases[i]);
   struct pan_graph_nrt_plan unchanged;
   memset(&unchanged, 0xa5, sizeof(unchanged));
   plan = unchanged;
   const uint32_t overflow[] = {3, 16, 3};
   assert(pan_graph_nrt_plan_trial(&target, requests, 5, overflow, &plan) ==
          PAN_GRAPH_NRT_OUT_OF_MEMORY);
   assert(!memcmp(&plan, &unchanged, sizeof(plan)));
   assert(pan_graph_nrt_plan_search(&target, requests, 16, 16, &plan) ==
          PAN_GRAPH_NRT_INVALID);
   assert(!memcmp(&plan, &unchanged, sizeof(plan)));
   requests[4].buffers = (enum pan_graph_nrt_buffer_group)4;
   assert(pan_graph_nrt_plan_search(&target, requests, 5, 16, &plan) ==
          PAN_GRAPH_NRT_INVALID);
   assert(!memcmp(&plan, &unchanged, sizeof(plan)));
   assert(pan_graph_nrt_plan_search(&target, NULL, 5, 16, &plan) ==
          PAN_GRAPH_NRT_INVALID);
   assert(pan_graph_nrt_plan_search(&target, requests, 5, 17, &plan) ==
          PAN_GRAPH_NRT_INVALID);
   assert(pan_graph_nrt_plan_trial(&target, requests, 5, NULL, &plan) ==
          PAN_GRAPH_NRT_INVALID);
   assert(!memcmp(&plan, &unchanged, sizeof(plan)));
   puts("NRT graph buffer planning: original external/internal order, fixed "
        "intermediate buffers, first failure stop and transactional errors passed");
   return 0;
}
