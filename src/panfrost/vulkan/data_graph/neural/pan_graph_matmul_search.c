#include "pan_graph_matmul_search.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct base_search {
   const struct pan_graph_matmul_info *info;
   const struct pan_graph_nrt_target *target;
   int32_t schedule_override;
   uint32_t column_divisor;
   uint32_t vector_count;
   uint32_t values[3][13];
   unsigned value_count[3];
   int32_t levels[14];
   unsigned level_count;
   unsigned depth;
   size_t grid_count;
   uint8_t *visited;
   bool ready;
   enum pan_graph_tsu_status status;
   struct pan_graph_matmul_candidate_list result;
};

static size_t
grid_index(const struct base_search *search, const uint32_t transfer[7])
{
   static const unsigned axes[] = {0, 3, 4};
   size_t result = 0;
   for (unsigned i = 0; i < 3; i++) {
      unsigned at = 0;
      while (at < search->value_count[i] &&
             search->values[i][at] != transfer[axes[i]])
         at++;
      if (at == search->value_count[i])
         return search->grid_count;
      result = result * search->value_count[i] + at;
   }
   return result;
}

static void
append_candidate(struct base_search *search, const uint32_t transfer[7])
{
   size_t index = grid_index(search, transfer);
   if (index == search->grid_count) {
      search->status = PAN_GRAPH_TSU_UNSUPPORTED;
      return;
   }
   if (search->visited[index] & 1)
      return;
   search->visited[index] |= 1;

   struct pan_graph_matmul_candidate_set filtered;
   if (!pan_graph_matmul_program_candidate_filter(
          search->info, transfer, search->target, search->schedule_override,
          search->column_divisor, search->vector_count, &filtered) ||
       filtered.count > 1 ||
       (filtered.count && memcmp(filtered.candidates[0].transfer, transfer,
                                 sizeof(filtered.candidates[0].transfer)))) {
      search->status = PAN_GRAPH_TSU_UNSUPPORTED;
      return;
   }
   search->ready |= filtered.root_flag;
   if (filtered.count)
      search->result.candidates[search->result.count++] =
         filtered.candidates[0];
}

static void
enumerate_regions(struct base_search *search, const uint32_t input[7],
                  unsigned axis)
{
   if (search->status != PAN_GRAPH_TSU_SUCCESS)
      return;
   size_t index = grid_index(search, input);
   if (index == search->grid_count) {
      search->status = PAN_GRAPH_TSU_UNSUPPORTED;
      return;
   }
   uint8_t bit = 1u << (axis + 1);
   if (search->visited[index] & bit)
      return;
   search->visited[index] |= bit;

   uint32_t current[7];
   memcpy(current, input, sizeof(current));
   if (!search->depth) {
      append_candidate(search, current);
      return;
   }
   for (unsigned i = 0; i < search->level_count; i++) {
      int32_t level = search->levels[i];
      if (level < 0) {
         append_candidate(search, current);
      } else if (current[axis] - 1 > (uint32_t)level) {
         current[axis] = level + 1;
         append_candidate(search, current);
      }
      if (axis + 1 < search->depth)
         enumerate_regions(search, current, axis + 1);
      if (search->status != PAN_GRAPH_TSU_SUCCESS)
         return;
   }
}

enum pan_graph_tsu_status
pan_graph_matmul_program_base_candidates(
   const struct pan_graph_matmul_info *info, const uint32_t caps[7],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   uint32_t column_divisor, bool exhaustive, uint32_t vector_count,
   struct pan_graph_matmul_candidate_list *candidates)
{
   struct pan_graph_matmul_plan plan;
   struct pan_graph_nrt_allocator allocator;
   if (!candidates || !caps || !target ||
       !pan_graph_matmul_i8_init(&plan, info) ||
       !pan_graph_nrt_allocator_init(&allocator, target) ||
       schedule_override < -1 || schedule_override > 3 ||
       column_divisor > INT32_MAX || !vector_count || vector_count > 8)
      return PAN_GRAPH_TSU_INVALID;
   if (target->alignment != 32 || target->banks != 8 || target->bank_parts != 1)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   for (unsigned i = 0; i < 7; i++) {
      if (!caps[i] || caps[i] > (uint64_t)plan.conv_high[i] + 1)
         return PAN_GRAPH_TSU_INVALID;
   }

   uint32_t base[7];
   for (unsigned i = 0; i < 7; i++)
      base[i] = caps[i] < 4095 ? caps[i] : 4095;
   base[2] = 1;
   uint32_t maximum = 1;
   for (unsigned i = 0; i < 7; i++) {
      uint32_t extent = plan.conv_high[i] + 1;
      if (extent > maximum)
         maximum = extent;
   }
   unsigned size = 4096;
   while (size > maximum)
      size /= 2;
   int32_t levels[14] = {-1};
   unsigned level_count = 1;
   for (; size >= 2; size /= 2)
      levels[level_count++] = size - 1;
   levels[level_count++] = 0;

   struct base_search search = {
      .info = info,
      .target = target,
      .schedule_override = schedule_override,
      .column_divisor = column_divisor,
      .vector_count = vector_count,
      .grid_count = 1,
   };
   static const unsigned axes[] = {0, 3, 4};
   for (unsigned i = 0; i < 3; i++) {
      search.values[i][search.value_count[i]++] = base[axes[i]];
      for (unsigned j = 1; j < level_count; j++) {
         uint32_t value = levels[j] + 1;
         if (value < base[axes[i]])
            search.values[i][search.value_count[i]++] = value;
      }
      search.grid_count *= search.value_count[i];
   }
   search.visited = calloc(search.grid_count, sizeof(*search.visited));
   search.result.candidates =
      calloc(search.grid_count, sizeof(*search.result.candidates));
   if (!search.visited || !search.result.candidates) {
      free(search.visited);
      free(search.result.candidates);
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   }

   static const int32_t ranges[][2] = {
      {INT32_MAX, 15},
      {15, 7},
      {7, 3},
      {3, 0},
   };
   for (unsigned stage = 0; stage < 4; stage++) {
      if (!exhaustive && search.result.count && search.ready)
         break;
      search.level_count = 0;
      for (unsigned i = 0; i < level_count; i++) {
         if (levels[i] < 0 ||
             (levels[i] >= ranges[stage][1] && levels[i] <= ranges[stage][0]))
            search.levels[search.level_count++] = levels[i];
      }
      size_t initial_count = search.result.count;
      for (search.depth = 0; search.depth <= 7; search.depth++) {
         for (size_t i = 0; i < search.grid_count; i++)
            search.visited[i] &= 1;
         enumerate_regions(&search, base, 0);
         if (search.status != PAN_GRAPH_TSU_SUCCESS ||
             (search.result.count != initial_count && search.ready))
            break;
      }
      if (search.status != PAN_GRAPH_TSU_SUCCESS)
         break;
   }
   free(search.visited);
   if (search.status != PAN_GRAPH_TSU_SUCCESS) {
      free(search.result.candidates);
      return search.status;
   }
   if (!search.result.count) {
      free(search.result.candidates);
      search.result.candidates = NULL;
   }
   *candidates = search.result;
   return PAN_GRAPH_TSU_SUCCESS;
}

enum pan_graph_tsu_status
pan_graph_matmul_base_candidates(
   const struct pan_graph_matmul_info *info, const uint32_t caps[7],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   uint32_t column_divisor, bool exhaustive,
   struct pan_graph_matmul_candidate_list *candidates)
{
   return pan_graph_matmul_program_base_candidates(
      info, caps, target, schedule_override, column_divisor, exhaustive, 1,
      candidates);
}

void
pan_graph_matmul_candidate_list_finish(
   struct pan_graph_matmul_candidate_list *candidates)
{
   if (candidates) {
      free(candidates->candidates);
      memset(candidates, 0, sizeof(*candidates));
   }
}
