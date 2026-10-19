#include "pan_graph_nrt_plan.h"

#include <limits.h>
#include <string.h>

enum pan_graph_nrt_status
pan_graph_nrt_plan_trial(const struct pan_graph_nrt_target *target,
                         const struct pan_graph_nrt_plan_request *requests,
                         size_t count, const uint32_t buffers[3],
                         struct pan_graph_nrt_plan *plan)
{
   struct pan_graph_nrt_allocator allocator;
   if (!buffers || !plan || (count && !requests) ||
       count > PAN_GRAPH_NRT_PLAN_MAX_RESOURCES ||
       !pan_graph_nrt_allocator_init(&allocator, target))
      return PAN_GRAPH_NRT_INVALID;
   for (unsigned i = 0; i < 3; i++) {
      if (!buffers[i] || buffers[i] > INT32_MAX)
         return PAN_GRAPH_NRT_INVALID;
   }
   struct pan_graph_nrt_plan result = {0};
   memcpy(result.buffers, buffers, sizeof(result.buffers));
   for (size_t i = 0; i < count; i++) {
      if ((unsigned)requests[i].buffers > PAN_GRAPH_NRT_BUFFER_OUTPUT)
         return PAN_GRAPH_NRT_INVALID;
      struct pan_graph_nrt_request request = requests[i].allocation;
      if (requests[i].buffers != PAN_GRAPH_NRT_BUFFER_FIXED)
         request.buffers = buffers[requests[i].buffers - 1];
      enum pan_graph_nrt_status status =
         pan_graph_nrt_allocate(&allocator, &request, &result.layouts[i]);
      if (status != PAN_GRAPH_NRT_SUCCESS)
         return status;
   }
   *plan = result;
   return PAN_GRAPH_NRT_SUCCESS;
}

enum pan_graph_nrt_status
pan_graph_nrt_plan_search(const struct pan_graph_nrt_target *target,
                          const struct pan_graph_nrt_plan_request *requests,
                          size_t count, uint32_t buffer_limit,
                          struct pan_graph_nrt_plan *plan)
{
   if (!plan || buffer_limit < 2 || buffer_limit > 16)
      return PAN_GRAPH_NRT_INVALID;
   uint32_t candidates[43][3] = {{2, 2, 2}};
   unsigned candidate_count = 1;
   for (uint32_t i = 3; i <= buffer_limit; i++) {
      candidates[candidate_count][0] = 2;
      candidates[candidate_count][1] = i;
      candidates[candidate_count++][2] = 2;
   }
   for (uint32_t i = 2; i < buffer_limit; i++) {
      candidates[candidate_count][0] = i;
      candidates[candidate_count][1] = buffer_limit;
      candidates[candidate_count++][2] = i + 1;
      candidates[candidate_count][0] = i + 1;
      candidates[candidate_count][1] = buffer_limit;
      candidates[candidate_count++][2] = i + 1;
   }
   struct pan_graph_nrt_plan result;
   bool found = false;
   for (unsigned i = 0; i < candidate_count; i++) {
      struct pan_graph_nrt_plan trial;
      enum pan_graph_nrt_status status = pan_graph_nrt_plan_trial(
         target, requests, count, candidates[i], &trial);
      if (status == PAN_GRAPH_NRT_INVALID)
         return status;
      if (status != PAN_GRAPH_NRT_SUCCESS)
         break;
      result = trial;
      found = true;
   }
   if (!found)
      return PAN_GRAPH_NRT_OUT_OF_MEMORY;
   *plan = result;
   return PAN_GRAPH_NRT_SUCCESS;
}
