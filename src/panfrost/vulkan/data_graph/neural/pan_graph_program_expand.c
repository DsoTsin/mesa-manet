#include "pan_graph_program_expand.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct expand_frame {
   const uint64_t *included;
   uint64_t *excluded;
   uint32_t node;
   uint32_t budget;
   uint32_t neighbor;
   uint32_t member;
   uint8_t phase;
   bool entered;
   bool missing;
   bool called;
};

static bool
contains(const uint64_t *mask, uint32_t index)
{
   return (mask[index / 64] >> (index % 64)) & 1;
}

static void
include(uint64_t *mask, uint32_t index)
{
   mask[index / 64] |= UINT64_C(1) << (index % 64);
}

static bool
intersects(const uint64_t *left, const uint64_t *right, size_t words)
{
   for (size_t i = 0; i < words; i++) {
      if (left[i] & right[i])
         return true;
   }
   return false;
}

void
pan_graph_program_subsets_finish(struct pan_graph_program_subsets *result)
{
   if (!result)
      return;
   free(result->groups);
   *result = (struct pan_graph_program_subsets){0};
}

static enum pan_graph_tsu_status
expand_matmul_add(
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   uint32_t io_limit, bool filter_fanout,
   struct pan_graph_program_subsets *result)
{
   if (!result || (add_count && !adds) || add_count > INT32_MAX - 2 ||
       !io_limit || io_limit > INT32_MAX)
      return PAN_GRAPH_TSU_INVALID;
   for (uint32_t i = 0; i < add_count; i++) {
      if (adds[i].input[0] > i || adds[i].input[1] > i)
         return PAN_GRAPH_TSU_INVALID;
   }
   uint32_t count = add_count + 2;
   size_t group_count = count;
   size_t words = ((size_t)count + 63) / 64;
   if (group_count > (SIZE_MAX - 4) / 4 ||
       words > SIZE_MAX / sizeof(uint64_t) ||
       (4 * (size_t)count + 4) > SIZE_MAX / (words * sizeof(uint64_t)) ||
       group_count > SIZE_MAX / sizeof(uint32_t) ||
       (size_t)count + 1 > SIZE_MAX / sizeof(struct expand_frame))
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   uint64_t *storage = calloc((4 * (size_t)count + 4) * words, sizeof(*storage));
   uint32_t *contributions = calloc(count, sizeof(*contributions));
   struct expand_frame *frames = calloc((size_t)count + 1, sizeof(*frames));
   uint64_t *subsets = NULL;
   size_t subset_count = 0, capacity = 0;
   enum pan_graph_tsu_status status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!storage || !contributions || !frames)
      goto cleanup;
   uint64_t *successors = storage;
   uint64_t *descendants = successors + count * words;
   uint64_t *sets = descendants + count * words;
   uint64_t *zero = sets + 2 * ((size_t)count + 1) * words;
   uint64_t *excluded = zero + words;
   include(successors, 1);
   for (uint32_t i = 0; i < add_count; i++) {
      for (unsigned side = 0; side < 2; side++) {
         uint32_t source = adds[i].input[side] + 1;
         include(successors + source * words, i + 2);
         contributions[source]++;
      }
   }
   contributions[0] = 1;
   for (uint32_t i = 1; i < count; i++)
      contributions[i] = 2 + (contributions[i] > 1);
   for (uint32_t i = count; i-- > 0;) {
      uint64_t *desc = descendants + i * words;
      for (uint32_t j = i + 1; j < count; j++) {
         if (contains(successors + i * words, j)) {
            include(desc, j);
            for (size_t word = 0; word < words; word++)
               desc[word] |= descendants[j * words + word];
         }
      }
   }
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   for (uint32_t i = 0; i + 1 < count; i++) {
      if (!contains(descendants + i * words, count - 1))
         goto cleanup;
   }
   for (uint32_t root = 0; root < count; root++) {
      if (root < 2 && (contains(excluded, 0) || contains(excluded, 1)))
         continue;
      size_t depth = 0;
      frames[0] = (struct expand_frame){
         .included = zero, .excluded = excluded, .node = root, .budget = 1,
      };
      for (;;) {
         struct expand_frame *frame = frames + depth;
         uint64_t *current = sets + 2 * depth * words;
         uint64_t *local = current + words;
         if (!frame->entered) {
            if (contains(frame->included, frame->node) ||
                contains(frame->excluded, frame->node))
               goto returned;
            uint32_t proposed = frame->budget + contributions[frame->node];
            if (depth && proposed > io_limit) {
               include(frame->excluded, frame->node);
               goto returned;
            }
            frame->budget = proposed;
            memcpy(current, frame->included, words * sizeof(*current));
            include(current, frame->node);
            frame->missing = frame->node < 2 &&
                             (!contains(current, 0) || !contains(current, 1));
            bool convex = true;
            for (uint32_t i = 0; i < count && convex; i++) {
               if (!contains(current, i))
                  continue;
               for (uint32_t j = 0; j < count; j++) {
                  if (!contains(current, j) && contains(successors + i * words, j) &&
                      intersects(descendants + j * words, current, words)) {
                     convex = false;
                     break;
                  }
               }
            }
            if (!frame->missing && convex) {
               if (subset_count == capacity) {
                  status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
                  if (capacity > SIZE_MAX / 2)
                     goto cleanup;
                  size_t next_capacity = capacity ? capacity * 2 : 16;
                  if (next_capacity > SIZE_MAX / (words * sizeof(*subsets)))
                     goto cleanup;
                  uint64_t *next = calloc(next_capacity * words, sizeof(*next));
                  if (!next)
                     goto cleanup;
                  if (subsets)
                     memcpy(next, subsets, subset_count * words * sizeof(*next));
                  free(subsets);
                  subsets = next;
                  capacity = next_capacity;
               }
               memcpy(subsets + subset_count++ * words, current,
                      words * sizeof(*current));
            }
            memcpy(local, frame->excluded, words * sizeof(*local));
            frame->entered = true;
         }
         bool descend = false;
         while (frame->phase < 2) {
            while (frame->neighbor < count) {
               uint32_t neighbor = frame->neighbor++;
               const uint64_t *included = frame->phase ? frame->included : current;
               uint32_t member = frame->phase ? frame->member : frame->node;
               if (contains(included, neighbor) || contains(local, neighbor))
                  continue;
               if (!frame->phase && frame->missing && neighbor >= 2)
                  continue;
               if (!contains(successors + member * words, neighbor) &&
                   !contains(successors + neighbor * words, member))
                  continue;
               frame->called = true;
               frames[depth + 1] = (struct expand_frame){
                  .included = current, .excluded = local, .node = neighbor,
                  .budget = frame->budget,
               };
               depth++;
               descend = true;
               break;
            }
            if (descend)
               break;
            frame->neighbor = 0;
            if (!frame->phase) {
               if (frame->missing || frame->called)
                  frame->phase = 2;
               else
                  frame->phase = 1;
            } else {
               frame->member++;
            }
            if (frame->phase == 1) {
               while (frame->member < count &&
                      !contains(frame->included, frame->member))
                  frame->member++;
               if (frame->member == count)
                  frame->phase = 2;
            }
         }
         if (descend)
            continue;
returned:
         if (!depth)
            break;
         depth--;
      }
      include(excluded, root);
   }
   if (filter_fanout) {
      size_t retained = 0;
      for (size_t i = 0; i < subset_count; i++) {
         const uint64_t *mask = subsets + i * words;
         bool reject = false;
         for (uint32_t member = 0; member < count && !reject; member++) {
            if (!contains(mask, member))
               continue;
            bool inside = false, outside = false;
            for (size_t word = 0; word < words; word++) {
               uint64_t consumers = successors[member * words + word];
               inside |= (consumers & mask[word]) != 0;
               outside |= (consumers & ~mask[word]) != 0;
            }
            reject = inside && outside;
         }
         if (!reject) {
            if (retained != i)
               memcpy(subsets + retained * words, mask, words * sizeof(*mask));
            for (size_t word = 0; word < words; word++)
               zero[word] |= mask[word];
            retained++;
         }
      }
      subset_count = retained;
      status = PAN_GRAPH_TSU_UNSUPPORTED;
      for (uint32_t i = 0; i < count; i++) {
         if (!contains(zero, i))
            goto cleanup;
      }
   }
   *result = (struct pan_graph_program_subsets){subsets, subset_count, count};
   subsets = NULL;
   status = PAN_GRAPH_TSU_SUCCESS;
cleanup:
   free(subsets);
   free(frames);
   free(contributions);
   free(storage);
   return status;
}

enum pan_graph_tsu_status
pan_graph_program_expand_matmul_add_v15(
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   uint32_t io_limit, struct pan_graph_program_subsets *result)
{
   return expand_matmul_add(adds, add_count, io_limit, false, result);
}

enum pan_graph_tsu_status
pan_graph_program_candidates_matmul_add_v15(
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   uint32_t io_limit, struct pan_graph_program_subsets *result)
{
   return expand_matmul_add(adds, add_count, io_limit, true, result);
}

enum pan_graph_tsu_status
pan_graph_program_unique_candidates_matmul_add_v15(
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   uint32_t io_limit, struct pan_graph_program_subsets *result)
{
   if (!result)
      return PAN_GRAPH_TSU_INVALID;
   struct pan_graph_program_subsets next = {0};
   enum pan_graph_tsu_status status =
      expand_matmul_add(adds, add_count, io_limit, true, &next);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      return status;
   status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   size_t capacity = 2;
   if (next.count > SIZE_MAX / 2)
      goto cleanup;
   while (capacity < 2 * next.count) {
      if (capacity > SIZE_MAX / 2)
         goto cleanup;
      capacity *= 2;
   }
   if (capacity > SIZE_MAX / sizeof(size_t))
      goto cleanup;
   size_t *map = calloc(capacity, sizeof(*map));
   if (!map)
      goto cleanup;
   size_t words = ((size_t)next.group_count + 63) / 64;
   size_t retained = 0;
   for (size_t i = 0; i < next.count; i++) {
      const uint64_t *mask = next.groups + i * words;
      uint64_t hash = 0;
      for (size_t word = 0; word < words; word++)
         hash = hash * 13 ^ mask[word];
      size_t bucket = hash & (capacity - 1);
      while (map[bucket] &&
             memcmp(next.groups + (map[bucket] - 1) * words, mask,
                    words * sizeof(*mask)))
         bucket = (bucket + 1) & (capacity - 1);
      if (map[bucket])
         continue;
      if (retained != i)
         memcpy(next.groups + retained * words, mask, words * sizeof(*mask));
      map[bucket] = ++retained;
   }
   free(map);
   next.count = retained;
   *result = next;
   return PAN_GRAPH_TSU_SUCCESS;
cleanup:
   pan_graph_program_subsets_finish(&next);
   return status;
}
