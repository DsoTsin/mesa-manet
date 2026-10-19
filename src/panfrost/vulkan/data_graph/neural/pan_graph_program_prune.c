#include "pan_graph_program_prune.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct candidate_order {
   const uint64_t *groups;
   size_t index;
   uint32_t words;
   uint32_t anchor;
   uint32_t size;
};

struct cache_entry {
   struct pan_graph_cost_report cost;
   uint64_t hash;
   bool used;
};

struct cache {
   uint8_t *data;
   size_t count;
   size_t capacity;
   size_t stride;
   size_t key_bytes;
   size_t selection_bytes;
   bool minimize_count;
};

struct frame {
   struct pan_graph_cost_report best;
   size_t next;
   size_t pending;
   uint32_t anchor;
   bool entered;
};

static int
compare_groups(const void *left, const void *right)
{
   const struct candidate_order *a = left, *b = right;
   if (a->anchor != b->anchor)
      return a->anchor < b->anchor ? -1 : 1;
   for (uint32_t i = a->words; i-- > 0;) {
      if (a->groups[i] != b->groups[i])
         return a->groups[i] < b->groups[i] ? -1 : 1;
   }
   return 0;
}

static int
compare_roots(const void *left, const void *right)
{
   const struct candidate_order *a = left, *b = right;
   if (a->size != b->size)
      return a->size < b->size ? -1 : 1;
   return a->index < b->index ? -1 : a->index > b->index;
}

static uint32_t
first_group(const uint64_t *groups, uint32_t words)
{
   for (uint32_t i = 0; i < words; i++) {
      if (groups[i])
         return i * 64 + __builtin_ctzll(groups[i]);
   }
   return UINT32_MAX;
}

static bool
subset(const uint64_t *a, const uint64_t *b, uint32_t words)
{
   for (uint32_t i = 0; i < words; i++) {
      if ((a[i] & b[i]) != a[i])
         return false;
   }
   return true;
}

static uint64_t
key_hash(const uint64_t *groups, size_t bytes)
{
   uint64_t hash = 0;
   for (size_t i = 0; i < bytes / sizeof(*groups); i++)
      hash = hash * 13 ^ groups[i];
   return hash;
}

static struct cache_entry *
cache_find(struct cache *cache, const uint64_t *groups, uint64_t hash)
{
   size_t slot = hash & (cache->capacity - 1);
   for (;;) {
      struct cache_entry *entry = (void *)(cache->data + slot * cache->stride);
      if (!entry->used ||
          (entry->hash == hash &&
           !memcmp(entry + 1, groups, cache->key_bytes)))
         return entry;
      slot = (slot + 1) & (cache->capacity - 1);
   }
}

static bool
cache_grow(struct cache *cache)
{
   if (cache->capacity > SIZE_MAX / 2)
      return false;
   size_t capacity = cache->capacity ? cache->capacity * 2 : 16;
   if (capacity > SIZE_MAX / cache->stride)
      return false;
   struct cache next = *cache;
   next.data = calloc(capacity, cache->stride);
   if (!next.data)
      return false;
   next.capacity = capacity;
   for (size_t i = 0; i < cache->capacity; i++) {
      struct cache_entry *entry = (void *)(cache->data + i * cache->stride);
      if (entry->used)
         memcpy(cache_find(&next, (void *)(entry + 1), entry->hash), entry,
                cache->stride);
   }
   free(cache->data);
   *cache = next;
   return true;
}

static bool
cache_store(struct cache *cache, const uint64_t *groups,
            const struct pan_graph_cost_report *cost, const uint64_t *selected)
{
   if (cache->count >= cache->capacity / 2 && !cache_grow(cache))
      return false;
   uint64_t hash = key_hash(groups, cache->key_bytes);
   struct cache_entry *entry = cache_find(cache, groups, hash);
   if (!entry->used)
      cache->count++;
   entry->cost = *cost;
   entry->hash = hash;
   entry->used = true;
   memcpy(entry + 1, groups, cache->key_bytes);
   if (cache->selection_bytes)
      memcpy((uint8_t *)(entry + 1) + cache->key_bytes, selected,
             cache->selection_bytes);
   return true;
}

static enum pan_graph_tsu_status
solve(const struct pan_graph_program_cost *candidates,
      const struct candidate_order *order, size_t count, uint32_t words,
      uint64_t *requests, struct frame *frames, struct cache *cache,
      struct pan_graph_program_prune_entry *entries,
      uint64_t *selections,
      struct pan_graph_cost_report *result)
{
   size_t depth = 0;
   bool returning = false;
   struct pan_graph_cost_report returned;
   const uint64_t *returned_selection = NULL;
   size_t selection_words = cache->selection_bytes / sizeof(uint64_t);
   frames[0].entered = false;
   for (;;) {
      struct frame *frame = frames + depth;
      uint64_t *request = requests + depth * words;
      uint64_t *selected = selection_words ? selections + depth * selection_words : NULL;
      if (returning) {
         struct pan_graph_cost_report candidate;
         if (!pan_graph_cost_accumulate_serial(&candidates[frame->pending].cost,
                                               &returned, &candidate))
            return PAN_GRAPH_TSU_UNSUPPORTED;
         bool better = candidate.cycles < frame->best.cycles ||
                       (!selection_words && candidate.cycles == frame->best.cycles);
         if (cache->minimize_count) {
            size_t current_count = 0, trial_count = 1;
            for (size_t i = 0; i < selection_words; i++) {
               current_count += __builtin_popcountll(selected[i]);
               trial_count += __builtin_popcountll(returned_selection[i]);
            }
            better = !current_count || trial_count <= current_count;
         }
         if (better) {
            frame->best = candidate;
            if (selection_words) {
               memcpy(selected, returned_selection, cache->selection_bytes);
               selected[frame->pending / 64] |= UINT64_C(1) << (frame->pending % 64);
            }
         }
         returning = false;
      }
      if (!frame->entered) {
         uint64_t hash = key_hash(request, cache->key_bytes);
         if (cache->capacity) {
            struct cache_entry *entry = cache_find(cache, request, hash);
            if (entry->used) {
               returned = entry->cost;
               returned_selection = (void *)((uint8_t *)(entry + 1) + cache->key_bytes);
               goto complete;
            }
         }
         frame->best = (struct pan_graph_cost_report){.cycles = INT64_MAX};
         frame->anchor = first_group(request, words);
         frame->next = 0;
         frame->entered = true;
         if (selection_words)
            memset(selected, 0, cache->selection_bytes);
      }
      bool descend = false;
      for (; frame->next < count;) {
         const struct candidate_order *candidate = order + frame->next++;
         size_t index = candidate->index;
         if (candidate->anchor > frame->anchor)
            break;
         if (candidate->anchor != frame->anchor || !entries[index].retained ||
             !memcmp(candidate->groups, request, cache->key_bytes) ||
             !subset(candidate->groups, request, words))
            continue;
         for (uint32_t i = 0; i < words; i++)
            request[words + i] = request[i] ^ candidate->groups[i];
         frame->pending = index;
         frames[++depth].entered = false;
         descend = true;
         break;
      }
      if (descend)
         continue;
      for (size_t i = 0; i < count; i++) {
         const struct candidate_order *candidate = order + i;
         size_t index = candidate->index;
         if (candidate->anchor == frame->anchor && entries[index].retained &&
             !memcmp(candidate->groups, request, cache->key_bytes)) {
            if (cache->minimize_count ||
                candidates[index].cost.cycles < frame->best.cycles) {
               frame->best = candidates[index].cost;
               if (selection_words) {
                  memset(selected, 0, cache->selection_bytes);
                  selected[index / 64] |= UINT64_C(1) << (index % 64);
               }
            } else if (!selection_words) {
               entries[index].retained = false;
            }
            break;
         }
      }
      if (frame->best.cycles == INT64_MAX)
         return PAN_GRAPH_TSU_UNSUPPORTED;
      if (!cache_store(cache, request, &frame->best, selected))
         return PAN_GRAPH_TSU_OUT_OF_MEMORY;
      returned = frame->best;
      returned_selection = selected;
complete:
      if (!depth) {
         *result = returned;
         if (selection_words && returned_selection != selections)
            memcpy(selections, returned_selection, cache->selection_bytes);
         return PAN_GRAPH_TSU_SUCCESS;
      }
      depth--;
      returning = true;
   }
}

static enum pan_graph_tsu_status
program_search(
   uint32_t group_count, const struct pan_graph_program_cost *candidates,
   size_t count, const uint64_t *request,
   enum pan_graph_program_choice_policy policy,
   struct pan_graph_program_prune_result *pruned,
   struct pan_graph_program_choice *chosen)
{
   if (!group_count || !candidates || !count)
      return PAN_GRAPH_TSU_INVALID;
   size_t words = ((size_t)group_count + 63) / 64;
   size_t depth = count < group_count ? count : group_count;
   size_t selection_words = request ? count / 64 + (count % 64 != 0) : 0;
   if (count > SIZE_MAX / (2 * sizeof(struct candidate_order)) ||
       count > SIZE_MAX / sizeof(struct pan_graph_program_prune_entry) ||
       depth == SIZE_MAX || depth + 1 > SIZE_MAX / sizeof(struct frame) ||
       words > (SIZE_MAX - sizeof(struct cache_entry)) / sizeof(uint64_t) ||
       depth + 1 > SIZE_MAX / (words * sizeof(uint64_t)) ||
       selection_words > (SIZE_MAX - sizeof(struct cache_entry) - words * sizeof(uint64_t)) / sizeof(uint64_t) ||
       (selection_words && depth + 1 > SIZE_MAX / (selection_words * sizeof(uint64_t))))
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (request &&
       (((group_count & 63) && request[words - 1] >> (group_count & 63)) ||
        first_group(request, words) == UINT32_MAX))
      return PAN_GRAPH_TSU_INVALID;
   struct pan_graph_program_prune_entry *entries = calloc(count, sizeof(*entries));
   struct candidate_order *order = calloc(2 * count, sizeof(*order));
   struct frame *frames = calloc(depth + 1, sizeof(*frames));
   uint64_t *requests = calloc((depth + 1) * words, sizeof(*requests));
   uint64_t *selections = request ? calloc((depth + 1) * selection_words, sizeof(*selections)) : NULL;
   enum pan_graph_tsu_status status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   struct cache cache = {
      .stride = sizeof(struct cache_entry) + (words + selection_words) * sizeof(uint64_t),
      .key_bytes = words * sizeof(uint64_t),
      .selection_bytes = selection_words * sizeof(uint64_t),
      .minimize_count = policy == PAN_GRAPH_PROGRAM_CHOICE_COUNT,
   };
   if (!entries || !order || !frames || !requests || (request && !selections))
      goto cleanup;
   struct candidate_order *roots = order + count;
   for (size_t i = 0; i < count; i++) {
      const uint64_t *groups = candidates[i].groups;
      status = PAN_GRAPH_TSU_INVALID;
      if (!groups || ((group_count & 63) &&
                      groups[words - 1] >> (group_count & 63)))
         goto cleanup;
      uint32_t anchor = first_group(groups, words);
      if (anchor == UINT32_MAX)
         goto cleanup;
      uint32_t size = 0;
      for (size_t j = 0; j < words; j++)
         size += __builtin_popcountll(groups[j]);
      struct pan_graph_cost_report checked;
      struct pan_graph_cost_report zero = {0};
      status = PAN_GRAPH_TSU_UNSUPPORTED;
      if (candidates[i].cost.cycles == INT64_MAX ||
          !pan_graph_cost_accumulate_serial(&candidates[i].cost, &zero, &checked))
         goto cleanup;
      entries[i] = (struct pan_graph_program_prune_entry){true, candidates[i].cost};
      roots[i] = order[i] = (struct candidate_order){groups, i, words, anchor, size};
   }
   qsort(order, count, sizeof(*order), compare_groups);
   qsort(roots, count, sizeof(*roots), compare_roots);
   for (size_t i = 1; i < count; i++) {
      if (!compare_groups(order + i - 1, order + i)) {
         status = PAN_GRAPH_TSU_INVALID;
         goto cleanup;
      }
   }
   if (request) {
      memcpy(requests, request, cache.key_bytes);
      struct pan_graph_cost_report cost;
      status = solve(candidates, order, count, words, requests, frames, &cache,
                     entries, selections, &cost);
      if (status != PAN_GRAPH_TSU_SUCCESS)
         goto cleanup;
      uint64_t *selected = calloc(selection_words, sizeof(*selected));
      if (!selected) {
         status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
         goto cleanup;
      }
      memcpy(selected, selections, cache.selection_bytes);
      *chosen = (struct pan_graph_program_choice){count, selected, cost.cycles};
   } else {
      for (size_t i = 0; i < count; i++) {
         if (roots[i].size < 2)
            continue;
         memcpy(requests, roots[i].groups, cache.key_bytes);
         status = solve(candidates, order, count, words, requests, frames, &cache,
                        entries, NULL, &entries[roots[i].index].cost);
         if (status != PAN_GRAPH_TSU_SUCCESS)
            goto cleanup;
      }
      *pruned = (struct pan_graph_program_prune_result){count, entries};
      entries = NULL;
   }
   status = PAN_GRAPH_TSU_SUCCESS;
cleanup:
   free(cache.data);
   free(selections);
   free(requests);
   free(frames);
   free(order);
   free(entries);
   return status;
}

enum pan_graph_tsu_status
pan_graph_program_prune_costs(
   uint32_t group_count, const struct pan_graph_program_cost *candidates,
   size_t count, struct pan_graph_program_prune_result *result)
{
   if (!result)
      return PAN_GRAPH_TSU_INVALID;
   return program_search(group_count, candidates, count, NULL,
                         PAN_GRAPH_PROGRAM_CHOICE_CYCLES, result, NULL);
}

enum pan_graph_tsu_status
pan_graph_program_choose_costs(
   uint32_t group_count, const uint64_t *request,
   const struct pan_graph_program_cost *candidates, size_t count,
   struct pan_graph_program_choice *result)
{
   return pan_graph_program_choose(group_count, request, candidates, count,
                                    PAN_GRAPH_PROGRAM_CHOICE_CYCLES, result);
}

enum pan_graph_tsu_status
pan_graph_program_choose(
   uint32_t group_count, const uint64_t *request,
   const struct pan_graph_program_cost *candidates, size_t count,
   enum pan_graph_program_choice_policy policy,
   struct pan_graph_program_choice *result)
{
   if (!request || !result ||
       (policy != PAN_GRAPH_PROGRAM_CHOICE_CYCLES &&
        policy != PAN_GRAPH_PROGRAM_CHOICE_COUNT))
      return PAN_GRAPH_TSU_INVALID;
   return program_search(group_count, candidates, count, request, policy, NULL,
                         result);
}

void
pan_graph_program_choice_finish(struct pan_graph_program_choice *result)
{
   if (!result)
      return;
   free(result->selected);
   *result = (struct pan_graph_program_choice){0};
}

enum pan_graph_tsu_status
pan_graph_program_select_costs(
   uint32_t group_count, const struct pan_graph_program_cost *candidates,
   size_t count, struct pan_graph_program_choice *result)
{
   if (!result)
      return PAN_GRAPH_TSU_INVALID;
   struct pan_graph_program_prune_result pruned = {0};
   enum pan_graph_tsu_status status =
      pan_graph_program_prune_costs(group_count, candidates, count, &pruned);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      return status;
   size_t words = ((size_t)group_count + 63) / 64;
   size_t selected_words = count / 64 + (count % 64 != 0);
   if (count > SIZE_MAX / sizeof(struct pan_graph_program_cost) ||
       count > SIZE_MAX / sizeof(size_t) ||
       count > SIZE_MAX / (words * sizeof(uint64_t))) {
      pan_graph_program_prune_finish(&pruned);
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   }
   struct candidate_order *order = calloc(count, sizeof(*order));
   struct pan_graph_program_cost *covers = calloc(count, sizeof(*covers));
   size_t *queue = calloc(count, sizeof(*queue));
   uint8_t *known = calloc(count, sizeof(*known));
   uint64_t *seen = calloc(words, sizeof(*seen));
   uint64_t *component = calloc(words, sizeof(*component));
   uint64_t *keys = calloc(count * words, sizeof(*keys));
   uint64_t *selected = calloc(selected_words, sizeof(*selected));
   uint64_t total = 0;
   status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!order || !covers || !queue || !known || !seen || !component ||
       !keys || !selected)
      goto cleanup;
   size_t active_count = 0;
   for (size_t i = 0; i < count; i++) {
      if (pruned.entries[i].retained) {
         order[active_count++] = (struct candidate_order){
            .groups = candidates[i].groups,
            .index = i,
            .words = words,
            .anchor = first_group(candidates[i].groups, words),
         };
      }
   }
   qsort(order, active_count, sizeof(*order), compare_groups);
   for (uint32_t atom = 0; atom < group_count; atom++) {
      if (seen[atom / 64] & (UINT64_C(1) << (atom % 64)))
         continue;
      memset(known, 0, count);
      memset(component, 0, words * sizeof(*component));
      size_t queue_count = 0;
      for (size_t i = 0; i < active_count; i++) {
         if (order[i].anchor == atom) {
            queue[queue_count++] = order[i].index;
            known[order[i].index] = 1;
         }
      }
      status = PAN_GRAPH_TSU_UNSUPPORTED;
      if (!queue_count)
         goto cleanup;
      for (size_t at = 0; at < queue_count; at++) {
         const uint64_t *groups = candidates[queue[at]].groups;
         for (size_t word = 0; word < words; word++)
            component[word] |= groups[word];
         for (size_t i = 0; i < active_count; i++) {
            size_t index = order[i].index;
            if (known[index])
               continue;
            bool overlaps = false;
            for (size_t word = 0; word < words; word++)
               overlaps |= (groups[word] & candidates[index].groups[word]) != 0;
            if (overlaps) {
               known[index] = 1;
               queue[queue_count++] = index;
            }
         }
      }
      for (size_t word = 0; word < words; word++)
         seen[word] |= component[word];
      uint64_t cycles;
      if (queue_count == 1) {
         selected[queue[0] / 64] |= UINT64_C(1) << (queue[0] % 64);
         cycles = candidates[queue[0]].cost.cycles;
      } else {
         memset(keys, 0, queue_count * words * sizeof(*keys));
         uint32_t compact = 0;
         for (size_t word = 0; word < words; word++) {
            uint64_t pending = component[word];
            while (pending) {
               unsigned bit = __builtin_ctzll(pending);
               for (size_t i = 0; i < queue_count; i++) {
                  if (candidates[queue[i]].groups[word] & (UINT64_C(1) << bit))
                     keys[i * words + compact / 64] |= UINT64_C(1) << (compact % 64);
               }
               compact++;
               pending &= pending - 1;
            }
         }
         for (size_t i = 0; i < queue_count; i++) {
            covers[i] = (struct pan_graph_program_cost){
               .groups = keys + i * words,
               .cost = candidates[queue[i]].cost,
            };
         }
         memset(component, 0, words * sizeof(*component));
         for (uint32_t i = 0; i < compact; i++)
            component[i / 64] |= UINT64_C(1) << (i % 64);
         struct pan_graph_program_choice choice = {0};
         status = pan_graph_program_choose_costs(compact, component, covers,
                                                 queue_count, &choice);
         if (status != PAN_GRAPH_TSU_SUCCESS)
            goto cleanup;
         for (size_t i = 0; i < queue_count; i++) {
            if (choice.selected[i / 64] & (UINT64_C(1) << (i % 64)))
               selected[queue[i] / 64] |= UINT64_C(1) << (queue[i] % 64);
         }
         cycles = choice.cycles;
         pan_graph_program_choice_finish(&choice);
      }
      status = PAN_GRAPH_TSU_UNSUPPORTED;
      if (cycles > INT64_MAX - total)
         goto cleanup;
      total += cycles;
   }
   *result = (struct pan_graph_program_choice){count, selected, total};
   selected = NULL;
   status = PAN_GRAPH_TSU_SUCCESS;
cleanup:
   free(selected);
   free(keys);
   free(component);
   free(seen);
   free(known);
   free(queue);
   free(covers);
   free(order);
   pan_graph_program_prune_finish(&pruned);
   return status;
}

void
pan_graph_program_prune_finish(struct pan_graph_program_prune_result *result)
{
   if (!result)
      return;
   free(result->entries);
   *result = (struct pan_graph_program_prune_result){0};
}
