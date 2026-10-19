#include "pan_graph_program_deps.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>

void
pan_graph_program_deps_finish(struct pan_graph_program_deps *deps)
{
   if (!deps)
      return;
   free(deps->offsets);
   free(deps->indices);
   *deps = (struct pan_graph_program_deps){0};
}

static bool
full_view(const struct pan_graph_matmul_info *info, uint32_t add_count,
          const struct pan_graph_program_view *view, bool output)
{
   uint32_t tensor = view->tensor;
   if (tensor >= add_count + 4 || tensor == add_count + 1 ||
       view->bounds.rank != 3 || (output && tensor > add_count) ||
       view->kind != (tensor > add_count + 1 ? 3 : tensor == add_count ? 4 : 1))
      return false;
   const uint32_t shape[3] = {
      tensor == add_count + 2 ? info->inner : info->columns,
      tensor == add_count + 3 ? info->inner : info->rows,
      info->batch,
   };
   for (unsigned i = 0; i < 3; i++) {
      if (view->bounds.low[i] || view->bounds.high[i] != (int32_t)shape[i])
         return false;
   }
   return true;
}

static bool
valid_storage(const struct pan_graph_program_memory *memory,
              uint32_t tensor_count, uint32_t output, uint64_t output_size)
{
   const struct pan_graph_memory_lifetimes *lifetimes = &memory->lifetimes;
   const struct pan_graph_memory_resources *resources = &memory->resources;
   if (lifetimes->tensor_count != tensor_count ||
       lifetimes->group_count > tensor_count || !lifetimes->identities ||
       resources->tensor_count != tensor_count || !resources->references ||
       (lifetimes->group_count &&
        (!lifetimes->groups || !lifetimes->requests || !lifetimes->allocations)))
      return false;
   for (uint32_t i = 0; i < lifetimes->group_count; i++) {
      const struct pan_graph_memory_lifetime_group *group = lifetimes->groups + i;
      uint64_t begin = lifetimes->allocations[i].offset;
      uint64_t size = lifetimes->requests[i].size;
      if (group->alias_count != 1 || group->first_identity >= tensor_count ||
          !size || begin > INT32_MAX || size - 1 > INT32_MAX - begin ||
          lifetimes->requests[i].first > lifetimes->requests[i].last ||
          lifetimes->identities[group->first_identity].group != i ||
          lifetimes->identities[group->first_identity].offset != begin)
         return false;
      const struct pan_graph_memory_resource_reference *ref =
         resources->references + group->first_identity;
      const struct pan_graph_memory_resource_table *table =
         resources->domains + PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT;
      if (ref->domain != PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT ||
          (ref->slot32 & 1) || ref->slot32 / 2 >= table->count || !table->entries ||
          table->entries[ref->slot32 / 2].uid != group->uid ||
          table->entries[ref->slot32 / 2].descriptor.size != size)
         return false;
      for (uint32_t j = 0; j < i; j++) {
         uint64_t other = lifetimes->allocations[j].offset;
         if (group->uid == lifetimes->groups[j].uid ||
             (begin < other + lifetimes->requests[j].size && other < begin + size &&
              lifetimes->requests[i].first <= lifetimes->requests[j].last &&
              lifetimes->requests[j].first <= lifetimes->requests[i].last))
            return false;
      }
   }
   for (uint32_t t = 0; t < tensor_count; t++) {
      uint32_t group = lifetimes->identities[t].group;
      if (group != UINT32_MAX &&
          (group >= lifetimes->group_count ||
           lifetimes->groups[group].first_identity != t))
         return false;
   }
   const struct pan_graph_memory_resource_reference *ref =
      resources->references + output;
   const struct pan_graph_memory_resource_table *table =
      resources->domains + PAN_GRAPH_MEMORY_RESOURCE_OUTPUT;
   if (ref->domain != PAN_GRAPH_MEMORY_RESOURCE_OUTPUT || (ref->slot32 & 1) ||
       ref->slot32 / 2 >= table->count || !table->entries ||
       table->entries[ref->slot32 / 2].descriptor.size != output_size)
      return false;
   return true;
}

static bool
append_dependency(uint32_t dependency, uint32_t program, uint32_t *seen,
                  uint32_t *indices, uint32_t *count)
{
   if (seen[dependency] == program)
      return true;
   if (*count == UINT32_MAX)
      return false;
   seen[dependency] = program;
   if (indices)
      indices[*count] = dependency;
   (*count)++;
   return true;
}

static bool
collect_dependencies(const struct pan_graph_program_memory *memory,
                     uint32_t output, uint64_t output_size, uint32_t program,
                     const uint32_t *writers, uint32_t *seen,
                     uint32_t *indices, uint32_t *count)
{
   const struct pan_graph_program_views *views = memory->views + program;
   for (size_t i = 0; i < views->input_count; i++) {
      uint32_t tensor = views->inputs[i].tensor;
      if (tensor > output + 1)
         continue;
      uint32_t writer = writers[tensor];
      if (writer >= program ||
          !append_dependency(writer, program, seen, indices, count))
         return false;
   }
   const struct pan_graph_memory_lifetimes *lifetimes = &memory->lifetimes;
   for (uint32_t q = 0; q < program; q++) {
      const struct pan_graph_program_views *prior = memory->views + q;
      for (size_t r = 0; r < prior->input_count; r++) {
         uint32_t tensor = prior->inputs[r].tensor;
         if (prior->inputs[r].kind != 1)
            continue;
         uint32_t group = lifetimes->identities[tensor].group;
         uint64_t begin = lifetimes->allocations[group].offset;
         uint64_t end = begin + lifetimes->requests[group].size;
         for (size_t w = 0; w < views->output_count; w++) {
            uint32_t written = views->outputs[w].tensor;
            if (written == tensor)
               continue;
            uint64_t other = 0, size = output_size;
            if (written != output) {
               uint32_t g = lifetimes->identities[written].group;
               other = lifetimes->allocations[g].offset;
               size = lifetimes->requests[g].size;
            }
            if (begin < other + size && other < end &&
                !append_dependency(q, program, seen, indices, count))
               return false;
         }
      }
   }
   return true;
}

enum pan_graph_tsu_status
pan_graph_program_deps_matmul_add_v15(
   const struct pan_graph_matmul_info *info, uint32_t add_count,
   const struct pan_graph_program_memory *memory,
   struct pan_graph_program_deps *result)
{
   if (!info || !memory || !result || !memory->views || !memory->program_count ||
       memory->program_count >= UINT32_MAX || add_count > INT32_MAX - 4 ||
       !info->batch || !info->rows || !info->inner || !info->columns ||
       info->batch > UINT16_MAX || info->rows > UINT16_MAX ||
       info->inner > UINT16_MAX || info->columns > UINT16_MAX)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   uint32_t tensor_count = add_count + 4;
   uint64_t output_size = (uint64_t)info->batch * info->rows * info->columns * 4;
   if (output_size - 1 > INT32_MAX ||
       !valid_storage(memory, tensor_count, add_count, output_size))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   uint32_t count = memory->program_count;
   size_t capacity = 0;
   for (uint32_t p = 0; p < count; p++) {
      const struct pan_graph_program_views *views = memory->views + p;
      if (!views->input_count || !views->inputs || !views->output_count ||
          !views->outputs || views->input_count > UINT32_MAX - capacity)
         return PAN_GRAPH_TSU_UNSUPPORTED;
      capacity += views->input_count;
      for (unsigned side = 0; side < 2; side++) {
         size_t n = side ? views->output_count : views->input_count;
         const struct pan_graph_program_view *v =
            side ? views->outputs : views->inputs;
         for (size_t i = 0; i < n; i++) {
            if (!full_view(info, add_count, v + i, side))
               return PAN_GRAPH_TSU_UNSUPPORTED;
            if (v[i].kind == 1 &&
                memory->lifetimes.identities[v[i].tensor].group == UINT32_MAX)
               return PAN_GRAPH_TSU_UNSUPPORTED;
         }
      }
   }
   size_t metadata_count = (size_t)tensor_count + count;
   if (SIZE_MAX / ((size_t)count + 1) < sizeof(uint32_t) ||
       metadata_count < tensor_count ||
       SIZE_MAX / metadata_count < sizeof(uint32_t))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct pan_graph_program_deps next = {.program_count = count};
   uint32_t *writers = malloc(metadata_count * sizeof(*writers));
   next.offsets = calloc((size_t)count + 1, sizeof(*next.offsets));
   enum pan_graph_tsu_status status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!writers || !next.offsets)
      goto cleanup;
   for (uint32_t t = 0; t < tensor_count; t++)
      writers[t] = UINT32_MAX;
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   for (uint32_t p = 0; p < count; p++) {
      const struct pan_graph_program_views *views = memory->views + p;
      for (size_t i = 0; i < views->output_count; i++) {
         uint32_t tensor = views->outputs[i].tensor;
         if (writers[tensor] != UINT32_MAX && writers[tensor] != p)
            goto cleanup;
         writers[tensor] = p;
      }
   }
   uint32_t *seen = writers + tensor_count;
   for (unsigned pass = 0; pass < 2; pass++) {
      for (uint32_t p = 0; p < count; p++)
         seen[p] = UINT32_MAX;
      next.index_count = 0;
      for (uint32_t p = 0; p < count; p++) {
         next.offsets[p] = next.index_count;
         if (!collect_dependencies(memory, add_count, output_size, p, writers,
                                   seen, next.indices, &next.index_count))
            goto cleanup;
      }
      if (!pass) {
         size_t n = next.index_count ? next.index_count : 1;
         if (SIZE_MAX / n < sizeof(*next.indices))
            goto cleanup;
         next.indices = malloc(n * sizeof(*next.indices));
         if (!next.indices) {
            status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
            goto cleanup;
         }
      }
   }
   next.offsets[count] = next.index_count;
   *result = next;
   next = (struct pan_graph_program_deps){0};
   status = PAN_GRAPH_TSU_SUCCESS;
cleanup:
   free(writers);
   pan_graph_program_deps_finish(&next);
   return status;
}

uint16_t
pan_graph_program_deps_wait_mask(const struct pan_graph_program_deps *deps,
                                uint32_t program, unsigned first_slot,
                                unsigned slot_count, uint8_t *waited)
{
   assert(program < deps->program_count && slot_count && first_slot < 16 &&
          slot_count <= 16 - first_slot);
   uint16_t mask = 0;
   for (uint32_t i = deps->offsets[program]; i < deps->offsets[program + 1]; i++) {
      uint32_t d = deps->indices[i];
      assert(d < program);
      if (!(waited[d] & 1)) {
         waited[d] = 1;
         mask |= 1u << (first_slot + d % slot_count);
      }
   }
   return mask;
}
