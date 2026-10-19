#include "pan_graph_matmul_select.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static enum pan_graph_tsu_status
merge_candidates(struct pan_graph_matmul_candidate_list *destination,
                 const struct pan_graph_matmul_candidate_list *source)
{
   if (source->count >
       SIZE_MAX / sizeof(*source->candidates) - destination->count)
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!source->count)
      return PAN_GRAPH_TSU_SUCCESS;
   size_t maximum = source->count + destination->count;
   struct pan_graph_matmul_candidate *candidates =
      realloc(destination->candidates, maximum * sizeof(*candidates));
   if (!candidates)
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   destination->candidates = candidates;
   for (size_t i = 0; i < source->count; i++) {
      const struct pan_graph_matmul_candidate *candidate =
         &source->candidates[i];
      size_t j = 0;
      for (; j < destination->count; j++) {
         if (!memcmp(candidates[j].transfer, candidate->transfer,
                     sizeof(candidate->transfer)))
            break;
      }
      if (j == destination->count)
         candidates[destination->count++] = *candidate;
   }
   return PAN_GRAPH_TSU_SUCCESS;
}

enum pan_graph_tsu_status
pan_graph_matmul_program_parallel_candidates(
   const struct pan_graph_matmul_info *info, const uint32_t initial[7],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   bool exhaustive, const struct pan_graph_matmul_parallel_options *options,
   const uint32_t granule[7], uint32_t vector_count,
   const struct pan_graph_matmul_candidate_list *seed,
   struct pan_graph_matmul_candidate_list *candidates)
{
   struct pan_graph_matmul_plan plan;
   if (!initial || !target || !options || !granule || !candidates ||
       !vector_count || vector_count > 8 ||
       (seed && (seed == candidates ||
                 seed->count > SIZE_MAX / sizeof(*seed->candidates) ||
                 (seed->count && !seed->candidates))) ||
       schedule_override < -1 || schedule_override > 3 ||
       !pan_graph_matmul_i8_init(&plan, info))
      return PAN_GRAPH_TSU_INVALID;
   for (unsigned i = 0; i < 7; i++) {
      if (!initial[i] || initial[i] > (uint64_t)plan.conv_high[i] + 1)
         return PAN_GRAPH_TSU_INVALID;
   }
   struct pan_graph_matmul_parallel_caps caps;
   if (!pan_graph_matmul_parallel_caps(info, granule, options, &caps))
      return PAN_GRAPH_TSU_UNSUPPORTED;

   struct pan_graph_matmul_candidate_list result = {0};
   if (seed) {
      for (size_t i = 0; i < seed->count; i++) {
         if (seed->candidates[i].schedule_type > 3)
            return PAN_GRAPH_TSU_INVALID;
         for (unsigned j = 0; j < 7; j++) {
            if (!seed->candidates[i].transfer[j] ||
                seed->candidates[i].transfer[j] > 4095 ||
                seed->candidates[i].transfer[j] >
                   (uint64_t)plan.conv_high[j] + 1)
               return PAN_GRAPH_TSU_INVALID;
         }
      }
      if (seed->count) {
         result.candidates =
            realloc(NULL, seed->count * sizeof(*seed->candidates));
         if (!result.candidates)
            return PAN_GRAPH_TSU_OUT_OF_MEMORY;
         memcpy(result.candidates, seed->candidates,
                seed->count * sizeof(*seed->candidates));
         result.count = seed->count;
      }
   }
   for (unsigned i = 0; i < caps.count; i++) {
      uint32_t limit[7];
      for (unsigned j = 0; j < 7; j++)
         limit[j] =
            caps.extents[i][j] < initial[j] ? caps.extents[i][j] : initial[j];
      struct pan_graph_matmul_candidate_list base = {0};
      enum pan_graph_tsu_status status =
         pan_graph_matmul_program_base_candidates(
            info, limit, target, schedule_override, options->column_divisor,
            exhaustive, vector_count, &base);
      if (status == PAN_GRAPH_TSU_SUCCESS)
         status = merge_candidates(&result, &base);
      pan_graph_matmul_candidate_list_finish(&base);
      if (status != PAN_GRAPH_TSU_SUCCESS) {
         pan_graph_matmul_candidate_list_finish(&result);
         return status;
      }
   }
   *candidates = result;
   return PAN_GRAPH_TSU_SUCCESS;
}

enum pan_graph_tsu_status
pan_graph_matmul_parallel_candidates(
   const struct pan_graph_matmul_info *info, const uint32_t initial[7],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   bool exhaustive, const struct pan_graph_matmul_parallel_options *options,
   struct pan_graph_matmul_candidate_list *candidates)
{
   const uint32_t granule[7] = {1, 1, 1, 1, 1, 1, 1};
   return pan_graph_matmul_program_parallel_candidates(
      info, initial, target, schedule_override, exhaustive, options, granule, 1,
      NULL, candidates);
}

enum pan_graph_tsu_status
pan_graph_matmul_select_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_nrt_target *target,
   const struct pan_graph_matmul_cost_target *cost_target, uint32_t core_budget,
   struct pan_graph_matmul_selection *selection)
{
   struct pan_graph_matmul_plan plan;
   if (!target || !cost_target || !selection || !core_budget ||
       core_budget > INT32_MAX || !pan_graph_matmul_i8_init(&plan, info))
      return PAN_GRAPH_TSU_INVALID;
   uint32_t initial[7];
   for (unsigned i = 0; i < 7; i++)
      initial[i] = plan.conv_high[i] + 1;
   struct pan_graph_matmul_candidate_list candidates = {0};
   enum pan_graph_tsu_status status = pan_graph_matmul_base_candidates(
      info, initial, target, -1, 0, false, &candidates);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      return status;
   const struct pan_graph_matmul_parallel_options options = {
      .core_budget = core_budget,
      .output_axis_mask = 15,
      .multiple_axes = true,
   };
   struct pan_graph_matmul_candidate_list parallel = {0};
   status = pan_graph_matmul_parallel_candidates(info, initial, target, -1,
                                                 false, &options, &parallel);
   if (status == PAN_GRAPH_TSU_SUCCESS)
      status = merge_candidates(&candidates, &parallel);
   pan_graph_matmul_candidate_list_finish(&parallel);
   if (status != PAN_GRAPH_TSU_SUCCESS) {
      pan_graph_matmul_candidate_list_finish(&candidates);
      return status;
   }

   const uint32_t granule[7] = {1, 1, 1, 1, 1, 1, 1};
   struct pan_graph_matmul_selection result = {.candidate_count =
                                                  candidates.count};
   uint64_t best = INT64_MAX;
   for (size_t i = 0; i < candidates.count; i++) {
      uint32_t loops[4];
      struct pan_graph_matmul_cost_result cost;
      const struct pan_graph_matmul_candidate *candidate =
         &candidates.candidates[i];
      if (!pan_graph_matmul_assign_loops(info, candidate->transfer, granule,
                                         &options, loops) ||
          !pan_graph_matmul_cost_estimate(info, candidate->transfer, loops,
                                          cost_target, core_budget, &cost) ||
          cost.total.cycles >= best)
         continue;
      best = cost.total.cycles;
      result.candidate = *candidate;
      memcpy(result.loops, loops, sizeof(result.loops));
      result.cost = cost.total;
   }
   pan_graph_matmul_candidate_list_finish(&candidates);
   if (best == INT64_MAX)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   *selection = result;
   return PAN_GRAPH_TSU_SUCCESS;
}
