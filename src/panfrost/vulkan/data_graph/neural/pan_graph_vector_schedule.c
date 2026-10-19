#include "pan_graph_vector_schedule.h"
#include "pan_graph_matmul_schedule.h"
#include "pan_graph_nrt_plan.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct base_search {
   const struct pan_graph_vector_schedule_info *info;
   const struct pan_graph_nrt_target *target;
   uint32_t base[3];
   uint32_t values[3][13];
   unsigned value_count[3];
   int32_t levels[14];
   unsigned level_count;
   unsigned depth;
   size_t grid_count;
   uint8_t *visited;
   struct pan_graph_vector_candidate_list result;
};

static bool
valid_info(const struct pan_graph_vector_schedule_info *info)
{
   if (!info || !info->read_count || info->read_count > 7 ||
       !info->vector_count || info->vector_count > 7 ||
       !info->export_count || info->export_count > info->vector_count ||
       info->write_count < info->export_count || info->write_count > 7)
      return false;
   for (unsigned i = 0; i < 3; i++) {
      if (!info->full[i] || info->full[i] > INT32_MAX ||
          !info->granule[i] || info->granule[i] > INT32_MAX)
         return false;
   }
   return true;
}

static bool
fits(const struct base_search *search, const uint32_t transfer[3])
{
   const struct pan_graph_vector_schedule_info *info = search->info;
   struct pan_graph_nrt_plan_request requests[14] = {0};
   unsigned count = info->read_count + info->vector_count;
   for (unsigned i = 0; i < count; i++) {
      requests[i].allocation = (struct pan_graph_nrt_request){
         .dimensions = {transfer[2], transfer[1], transfer[0], 1},
         .element_bits = 32, .buffers = 2, .front = i < info->read_count,
      };
      requests[i].buffers = i < info->read_count
                              ? PAN_GRAPH_NRT_BUFFER_INPUT
                           : i < info->read_count + info->export_count
                              ? PAN_GRAPH_NRT_BUFFER_OUTPUT
                              : PAN_GRAPH_NRT_BUFFER_FIXED;
   }
   struct pan_graph_nrt_plan plan;
   return pan_graph_nrt_plan_search(search->target, requests, count, 16, &plan) ==
          PAN_GRAPH_NRT_SUCCESS;
}

static size_t
grid_index(const struct base_search *search, const uint32_t transfer[3])
{
   size_t result = 0;
   for (unsigned axis = 0; axis < 3; axis++) {
      unsigned i = 0;
      while (search->values[axis][i] != transfer[axis])
         i++;
      result = result * search->value_count[axis] + i;
   }
   return result;
}

static void
append(struct base_search *search, const uint32_t transfer[3])
{
   size_t index = grid_index(search, transfer);
   if (search->visited[index] & 1)
      return;
   search->visited[index] |= 1;
   if (!fits(search, transfer))
      return;
   memcpy(search->result.candidates[search->result.count++].transfer, transfer,
          sizeof(search->base));
}

static void
enumerate(struct base_search *search, const uint32_t transfer[3], unsigned axis)
{
   size_t index = grid_index(search, transfer);
   uint8_t bit = 2u << axis;
   if (search->visited[index] & bit)
      return;
   search->visited[index] |= bit;
   uint32_t current[3];
   memcpy(current, transfer, sizeof(current));
   if (!search->depth) {
      append(search, current);
      return;
   }
   for (unsigned i = 0; i < search->level_count; i++) {
      int32_t level = search->levels[i];
      if (level < 0) {
         append(search, current);
      } else if (current[axis] - 1 > (uint32_t)level) {
         current[axis] = level + 1;
         append(search, current);
      }
      if (axis + 1 < search->depth)
         enumerate(search, current, axis + 1);
   }
}

enum pan_graph_tsu_status
pan_graph_vector_base_candidates(
   const struct pan_graph_vector_schedule_info *info, const uint32_t caps[3],
   const struct pan_graph_nrt_target *target,
   struct pan_graph_vector_candidate_list *candidates)
{
   if (!valid_info(info) || !caps || !target || !candidates)
      return PAN_GRAPH_TSU_INVALID;
   if (target->shared_size != 65536 || target->alignment != 32 ||
       target->banks != 8 || target->bank_parts != 1)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct base_search search = {.info = info, .target = target, .grid_count = 1};
   uint32_t maximum = 1;
   for (unsigned i = 0; i < 3; i++) {
      if (!caps[i] || caps[i] > info->full[i])
         return PAN_GRAPH_TSU_INVALID;
      search.base[i] = caps[i] < 4095 ? caps[i] : 4095;
      search.values[i][search.value_count[i]++] = search.base[i];
      for (uint32_t size = 2048; size; size /= 2) {
         if (size < search.base[i])
            search.values[i][search.value_count[i]++] = size;
      }
      search.grid_count *= search.value_count[i];
      if (info->full[i] > maximum)
         maximum = info->full[i];
   }
   search.visited = calloc(search.grid_count, sizeof(*search.visited));
   search.result.candidates =
      malloc(search.grid_count * sizeof(*search.result.candidates));
   if (!search.visited || !search.result.candidates) {
      free(search.visited);
      free(search.result.candidates);
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   }
   uint32_t size = 4096;
   while (size > maximum)
      size /= 2;
   int32_t levels[14] = {-1};
   unsigned count = 1;
   for (; size >= 2; size /= 2)
      levels[count++] = size - 1;
   levels[count++] = 0;
   const int32_t ranges[4][2] = {{INT32_MAX, 15}, {15, 7}, {7, 3}, {3, 0}};
   for (unsigned stage = 0; stage < 4 && !search.result.count; stage++) {
      search.level_count = 0;
      for (unsigned i = 0; i < count; i++) {
         int32_t level = levels[i];
         if (level < 0 || (level >= ranges[stage][1] && level <= ranges[stage][0]))
            search.levels[search.level_count++] = level;
      }
      for (search.depth = 0; search.depth <= 3; search.depth++) {
         for (size_t i = 0; i < search.grid_count; i++)
            search.visited[i] &= 1;
         enumerate(&search, search.base, 0);
         if (search.result.count)
            break;
      }
   }
   free(search.visited);
   *candidates = search.result;
   return PAN_GRAPH_TSU_SUCCESS;
}

static void
ordered_axes(const uint32_t full[7], const uint32_t alignment[7], int32_t axes[4])
{
   uint8_t available = 15;
   for (unsigned pass = 0; pass < 4; pass++) {
      uint64_t largest = 0;
      axes[pass] = -1;
      for (unsigned i = 0; i < 4; i++) {
         if (!(available & (1u << i)) || full[i] <= alignment[i])
            continue;
         uint64_t groups = ((uint64_t)full[i] + alignment[i] - 1) / alignment[i];
         if (groups > largest) {
            largest = groups;
            axes[pass] = i;
         }
      }
      if (axes[pass] >= 0)
         available &= ~(1u << axes[pass]);
   }
}

static uint32_t
greatest_common_divisor(uint32_t a, uint32_t b)
{
   while (b) {
      uint32_t remainder = a % b;
      a = b;
      b = remainder;
   }
   return a;
}

bool
pan_graph_vector_assign_loops(
   const struct pan_graph_vector_schedule_info *info,
   const uint32_t transfer[3], uint32_t core_budget, uint32_t loops[4])
{
   if (!valid_info(info) || !transfer || !loops || core_budget > 64)
      return false;
   uint32_t full[7] = {1, 1, 1, 1, 1, 1, 1};
   uint32_t alignment[7] = {1, 1, 1, 1, 1, 1, 1};
   for (unsigned i = 0; i < 3; i++) {
      if (!transfer[i] || transfer[i] > info->full[i])
         return false;
      full[i + 1] = info->full[i];
      uint64_t common = (uint64_t)transfer[i] * info->granule[i] /
                        greatest_common_divisor(transfer[i], info->granule[i]);
      alignment[i + 1] = common < full[i + 1] ? common : full[i + 1];
   }
   int32_t axes[4];
   ordered_axes(full, alignment, axes);
   int32_t a = axes[0], b = axes[1];
   struct pan_graph_matmul_parallel_score best;
   if (!pan_graph_matmul_parallel_score(full, alignment, axes, core_budget, 0, &best))
      return false;
   for (unsigned i = 0; i < 3; i++) {
      for (unsigned j = i + 1; j < 4; j++) {
         if ((!i && j == 1) || axes[j] < 0)
            continue;
         int32_t pair[2] = {axes[i], axes[j]};
         struct pan_graph_matmul_parallel_score trial;
         if (!pan_graph_matmul_parallel_score(full, alignment, pair, core_budget,
                                               0, &trial))
            return false;
         float difference = trial.score - 0.01f;
         if (difference > best.score) {
            best = trial;
            a = pair[0];
            b = pair[1];
         }
      }
   }
   uint32_t result[4] = {0};
   if (best.total_groups >= 2) {
      uint32_t x = best.tile[0], y = best.tile[1];
      if (best.groups[1] == 1)
         b = -1;
      if (a >= 0 && x >= full[a])
         a = -1;
      if (b >= 0 && (a < 0 || b < a)) {
         int32_t axis = a;
         a = b;
         b = axis;
         uint32_t tile = x;
         x = y;
         y = tile;
      }
      result[0] = a;
      result[1] = b >= 0 ? b : 0;
      result[2] = x;
      result[3] = b >= 0 ? y : 0;
   }
   memcpy(loops, result, sizeof(result));
   return true;
}

static bool
insert_cap(uint32_t caps[23][4], unsigned *count, const uint32_t shape[4])
{
   unsigned at = 0;
   for (; at < *count; at++) {
      unsigned i = 0;
      while (i < 4 && caps[at][i] == shape[i])
         i++;
      if (i == 4)
         return true;
      if (shape[i] > caps[at][i])
         break;
   }
   if (*count == 23)
      return false;
   memmove(caps[at + 1], caps[at], (*count - at) * sizeof(caps[0]));
   memcpy(caps[at], shape, sizeof(caps[0]));
   (*count)++;
   return true;
}

enum pan_graph_tsu_status
pan_graph_vector_parallel_candidates(
   const struct pan_graph_vector_schedule_info *info,
   const struct pan_graph_nrt_target *target, uint32_t core_budget,
   struct pan_graph_vector_candidate_list *candidates)
{
   if (!valid_info(info) || !candidates || core_budget > 64)
      return PAN_GRAPH_TSU_INVALID;
   struct pan_graph_vector_candidate_list result = {0};
   enum pan_graph_tsu_status status =
      pan_graph_vector_base_candidates(info, info->full, target, &result);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      return status;
   if (core_budget < 2) {
      *candidates = result;
      return PAN_GRAPH_TSU_SUCCESS;
   }
   uint32_t full[7] = {1, 1, 1, 1, 1, 1, 1};
   uint32_t granule[7] = {1, 1, 1, 1, 1, 1, 1};
   memcpy(full + 1, info->full, sizeof(info->full));
   memcpy(granule + 1, info->granule, sizeof(info->granule));
   int32_t axes[4];
   ordered_axes(full, granule, axes);
   struct pan_graph_matmul_parallel_score score;
   if (!pan_graph_matmul_parallel_score(full, granule, axes, core_budget, 0, &score)) {
      status = PAN_GRAPH_TSU_INVALID;
      goto out;
   }
   int32_t a = axes[0], b = score.groups[1] == 1 ? -1 : axes[1];
   uint32_t x = score.tile[0], y = score.tile[1];
   if (a >= 0 && x >= full[a])
      a = -1;
   if (b >= 0 && (a < 0 || b < a)) {
      int32_t axis = a;
      a = b;
      b = axis;
      uint32_t tile = x;
      x = y;
      y = tile;
   }
   uint32_t caps[23][4];
   unsigned cap_count = 0;
   if (a >= 0) {
      uint32_t minimum_a = granule[a] > 2 ? granule[a] : 2;
      uint32_t minimum_b = b >= 0 && granule[b] > 2 ? granule[b] : 2;
      for (uint32_t ta = x; ta >= minimum_a; ta -= granule[a]) {
         uint32_t shape[4];
         memcpy(shape, full, sizeof(shape));
         if (!(x % ta)) {
            shape[a] = ta;
            if (!insert_cap(caps, &cap_count, shape)) {
               status = PAN_GRAPH_TSU_UNSUPPORTED;
               goto out;
            }
         }
         if (b >= 0) {
            for (uint32_t tb = y; tb >= minimum_b; tb -= granule[b]) {
               if (!(y % tb)) {
                  shape[b] = tb;
                  if (!insert_cap(caps, &cap_count, shape)) {
                     status = PAN_GRAPH_TSU_UNSUPPORTED;
                     goto out;
                  }
                  memcpy(shape, full, sizeof(shape));
                  shape[b] = tb;
                  if (!insert_cap(caps, &cap_count, shape)) {
                     status = PAN_GRAPH_TSU_UNSUPPORTED;
                     goto out;
                  }
               }
               if (cap_count > 20)
                  break;
            }
         }
         if (cap_count > 20)
            break;
      }
   }
   size_t capacity = result.count;
   for (unsigned cap = 0; cap < cap_count; cap++) {
      struct pan_graph_vector_candidate_list extra = {0};
      status = pan_graph_vector_base_candidates(info, caps[cap] + 1, target, &extra);
      if (status != PAN_GRAPH_TSU_SUCCESS)
         goto out;
      for (size_t i = 0; i < extra.count; i++) {
         size_t j = 0;
         while (j < result.count && memcmp(result.candidates[j].transfer,
                                           extra.candidates[i].transfer,
                                           sizeof(extra.candidates[i].transfer)))
            j++;
         if (j != result.count)
            continue;
         if (result.count == capacity) {
            if (capacity > SIZE_MAX / (2 * sizeof(*result.candidates))) {
               status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
               break;
            }
            size_t next = capacity ? capacity * 2 : 1;
            void *data = realloc(result.candidates, next * sizeof(*result.candidates));
            if (!data) {
               status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
               break;
            }
            result.candidates = data;
            capacity = next;
         }
         result.candidates[result.count++] = extra.candidates[i];
      }
      pan_graph_vector_candidates_finish(&extra);
      if (status != PAN_GRAPH_TSU_SUCCESS)
         goto out;
   }
   *candidates = result;
   return PAN_GRAPH_TSU_SUCCESS;
out:
   pan_graph_vector_candidates_finish(&result);
   return status;
}

void
pan_graph_vector_candidates_finish(struct pan_graph_vector_candidate_list *candidates)
{
   if (!candidates)
      return;
   free(candidates->candidates);
   *candidates = (struct pan_graph_vector_candidate_list){0};
}
