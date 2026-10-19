#include "pan_graph_memory_lifetime.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static uint64_t
hash_uid(uint64_t uid)
{
   uid ^= uid >> 30;
   uid *= UINT64_C(0xbf58476d1ce4e5b9);
   uid ^= uid >> 27;
   uid *= UINT64_C(0x94d049bb133111eb);
   return uid ^ (uid >> 31);
}

void
pan_graph_memory_lifetimes_cleanup(struct pan_graph_memory_lifetimes *lifetimes)
{
   if (!lifetimes)
      return;
   free(lifetimes->groups);
   free(lifetimes->requests);
   free(lifetimes->allocations);
   free(lifetimes->identities);
   memset(lifetimes, 0, sizeof(*lifetimes));
}

enum pan_graph_memory_lifetime_status
pan_graph_memory_lifetimes_create(const struct pan_graph_memory_lifetime_input *input,
                                 struct pan_graph_memory_lifetimes *result)
{
   if (!input || !result || (input->tensor_count && !input->tensors) ||
       (input->program_count && !input->programs) ||
       (input->reference_count && !input->references) ||
       input->tensor_count > INT32_MAX || input->program_count > INT32_MAX ||
       input->reference_count > INT32_MAX || input->outer_group_count > INT32_MAX / 2)
      return PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED;
   for (uint32_t i = 0; i < input->tensor_count; i++) {
      if (input->tensors[i].kind < 1 || input->tensors[i].kind > 4)
         return PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED;
   }
   for (uint32_t i = 0; i < input->program_count; i++) {
      const struct pan_graph_memory_lifetime_program *program = &input->programs[i];
      if (program->outer_group >= input->outer_group_count ||
          (i && program->outer_group < input->programs[i - 1].outer_group) ||
          program->first_reference > input->reference_count ||
          program->reference_count > input->reference_count - program->first_reference)
         return PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED;
      for (uint32_t j = 0; j < program->reference_count; j++) {
         if (input->references[program->first_reference + j] >= input->tensor_count)
            return PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED;
      }
   }

   struct pan_graph_memory_lifetimes next = {
      .tensor_count = input->tensor_count,
      .end_clock = 2 * input->outer_group_count,
   };
   uint32_t *map = NULL;
   size_t capacity = 2;
   if (input->tensor_count) {
      if (SIZE_MAX / input->tensor_count < sizeof(*next.groups) ||
          SIZE_MAX / input->tensor_count < sizeof(*next.requests) ||
          SIZE_MAX / input->tensor_count < sizeof(*next.allocations) ||
          SIZE_MAX / input->tensor_count < sizeof(*next.identities) ||
          SIZE_MAX / input->tensor_count < 2)
         return PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED;
      while (capacity < (size_t)input->tensor_count * 2) {
         if (capacity > SIZE_MAX / 2)
            return PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED;
         capacity *= 2;
      }
      if (SIZE_MAX / capacity < sizeof(*map))
         return PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED;
      next.groups = calloc(input->tensor_count, sizeof(*next.groups));
      next.requests = calloc(input->tensor_count, sizeof(*next.requests));
      next.allocations = calloc(input->tensor_count, sizeof(*next.allocations));
      next.identities = calloc(input->tensor_count, sizeof(*next.identities));
      map = calloc(capacity, sizeof(*map));
      if (!next.groups || !next.requests || !next.allocations || !next.identities || !map)
         goto no_memory;
      for (uint32_t i = 0; i < input->tensor_count; i++)
         next.identities[i] = (struct pan_graph_memory_lifetime_identity){UINT64_MAX, UINT32_MAX};
   }

   for (uint32_t i = 0; i < input->program_count; i++) {
      const struct pan_graph_memory_lifetime_program *program = &input->programs[i];
      uint32_t time = 2 * program->outer_group;
      for (uint32_t j = 0; j < program->reference_count; j++) {
         uint32_t id = input->references[program->first_reference + j];
         const struct pan_graph_memory_lifetime_tensor *tensor = &input->tensors[id];
         if (tensor->kind != 1)
            continue;
         uint32_t group = next.identities[id].group;
         if (group == UINT32_MAX) {
            size_t bucket = hash_uid(tensor->uid) & (capacity - 1);
            while (map[bucket] && next.groups[map[bucket] - 1].uid != tensor->uid)
               bucket = (bucket + 1) & (capacity - 1);
            if (!map[bucket]) {
               if (!tensor->storage_size || tensor->storage_size > UINT64_MAX - 63)
                  goto unsupported;
               group = next.group_count++;
               map[bucket] = group + 1;
               next.groups[group] = (struct pan_graph_memory_lifetime_group){tensor->uid, id, 0};
               next.requests[group] = (struct pan_graph_alloc_request){time, time + 1, tensor->storage_size};
            } else {
               group = map[bucket] - 1;
            }
            next.identities[id].group = group;
            next.groups[group].alias_count++;
         }
         struct pan_graph_alloc_request *request = &next.requests[group];
         if (time < request->first)
            request->first = time;
         if (time + 1 > request->last)
            request->last = time + 1;
      }
   }
   free(map);
   map = NULL;
   enum pan_graph_alloc_status status =
      pan_graph_allocate(next.requests, next.group_count, 64, 500,
                         next.allocations, &next.stats);
   if (status == PAN_GRAPH_ALLOC_NO_MEMORY)
      goto no_memory;
   if (status != PAN_GRAPH_ALLOC_SUCCESS)
      goto unsupported;
   for (uint32_t i = 0; i < input->tensor_count; i++) {
      uint32_t group = next.identities[i].group;
      if (group != UINT32_MAX)
         next.identities[i].offset = next.allocations[group].offset;
   }
   *result = next;
   return PAN_GRAPH_MEMORY_LIFETIME_SUCCESS;

unsupported:
   free(map);
   pan_graph_memory_lifetimes_cleanup(&next);
   return PAN_GRAPH_MEMORY_LIFETIME_UNSUPPORTED;
no_memory:
   free(map);
   pan_graph_memory_lifetimes_cleanup(&next);
   return PAN_GRAPH_MEMORY_LIFETIME_NO_MEMORY;
}
