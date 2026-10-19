#include "pan_graph_program_memory.h"
#include "pan_graph_memory_policy.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

void
pan_graph_program_memory_finish(struct pan_graph_program_memory *memory)
{
   if (!memory)
      return;
   for (size_t i = 0; memory->views && i < memory->program_count; i++)
      pan_graph_program_views_finish(memory->views + i);
   free(memory->views);
   free(memory->program_bindings);
   free(memory->memories);
   pan_graph_memory_resources_cleanup(&memory->resources);
   pan_graph_memory_lifetimes_cleanup(&memory->lifetimes);
   *memory = (struct pan_graph_program_memory){0};
}

static bool
pinning_capacities(const struct pan_graph_memory_cost_target *target,
                   uint64_t *half, uint64_t *full)
{
   if (!target || !target->memory[3].parallel || !target->memory[4].parallel ||
       !target->memory[3].size || !target->memory[4].size ||
       target->memory[3].size > INT64_MAX / target->memory[3].parallel ||
       target->memory[4].size > INT64_MAX / target->memory[4].parallel)
      return false;
   int64_t capacity = target->memory[3].size * target->memory[3].parallel;
   float scaled = (float)capacity * 0.5f;
   *half = (int64_t)scaled;
   *full = target->memory[4].size * target->memory[4].parallel;
   return true;
}

static bool
same_boundary(const struct pan_graph_program_views *views,
              const struct pan_graph_program_boundary *boundary)
{
   if (views->input_count != boundary->input_count ||
       views->output_count != boundary->output_count ||
       views->root != boundary->root)
      return false;
   for (unsigned side = 0; side < 2; side++) {
      size_t count = side ? views->output_count : views->input_count;
      const struct pan_graph_program_view *v =
         side ? views->outputs : views->inputs;
      const struct pan_graph_program_boundary_view *b =
         side ? boundary->outputs : boundary->inputs;
      if (count && !b)
         return false;
      for (size_t i = 0; i < count; i++) {
         if (v[i].tensor != b[i].tensor || v[i].key != b[i].key)
            return false;
      }
   }
   return true;
}

enum pan_graph_tsu_status
pan_graph_program_memory_matmul_add_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const struct pan_graph_program_schedule *schedule,
   const struct pan_graph_memory_cost_target *cost_target,
   const struct pan_graph_program_memory_binding bindings[3],
   struct pan_graph_program_memory *result)
{
   struct pan_graph_matmul_plan matmul;
   uint64_t half_capacity, full_capacity;
   if (!info || !schedule || !bindings || !result || (add_count && !adds) ||
       !pinning_capacities(cost_target, &half_capacity, &full_capacity) ||
       add_count > INT32_MAX - 4 || !pan_graph_matmul_i8_init(&matmul, info) ||
       info->batch > UINT16_MAX || info->rows > UINT16_MAX ||
       info->inner > UINT16_MAX || info->columns > UINT16_MAX ||
       !schedule->count || !schedule->programs || schedule->count > INT32_MAX ||
       schedule->group_count != add_count + 2 || bindings[0].ordinal > 1 ||
       bindings[1].ordinal > 1 || bindings[0].ordinal == bindings[1].ordinal ||
       bindings[2].ordinal)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   uint32_t tensor_count = add_count + 4;
   size_t program_count = schedule->count;
   if (SIZE_MAX / program_count < sizeof(struct pan_graph_program_views) ||
       SIZE_MAX / program_count <
          sizeof(struct pan_graph_memory_lifetime_program) ||
       SIZE_MAX / tensor_count <
          sizeof(struct pan_graph_memory_resource_tensor) ||
       SIZE_MAX / tensor_count <
          sizeof(struct pan_graph_memory_lifetime_tensor))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   for (uint32_t g = 0; g < schedule->group_count; g++) {
      size_t owners = 0;
      for (size_t p = 0; p < program_count; p++) {
         if (!schedule->programs[p].groups)
            return PAN_GRAPH_TSU_UNSUPPORTED;
         owners += (schedule->programs[p].groups[g / 64] >> (g % 64)) & 1;
      }
      if (owners != 1)
         return PAN_GRAPH_TSU_UNSUPPORTED;
   }
   struct pan_graph_program_memory next = {.program_count = program_count};
   struct pan_graph_memory_lifetime_program *programs =
      calloc(program_count, sizeof(*programs));
   struct pan_graph_memory_resource_tensor *tensors =
      calloc(tensor_count, sizeof(*tensors));
   struct pan_graph_memory_lifetime_tensor *lifetime_tensors =
      calloc(tensor_count, sizeof(*lifetime_tensors));
   uint32_t *references = NULL;
   struct pan_graph_memory_policy_source_view *policy_views = NULL;
   struct pan_graph_memory_policy_snapshot *snapshots = NULL;
   struct pan_graph_memory_policy_view *lookup_views = NULL;
   next.views = calloc(program_count, sizeof(*next.views));
   enum pan_graph_tsu_status status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!programs || !tensors || !lifetime_tensors || !next.views)
      goto cleanup;
   size_t reference_count = 0;
   size_t memory_count = 0, max_views = 0;
   for (size_t p = 0; p < program_count; p++) {
      status = pan_graph_program_view_matmul_add_v15(
         info, adds, add_count, schedule->programs[p].groups, next.views + p);
      if (status != PAN_GRAPH_TSU_SUCCESS)
         goto cleanup;
      const struct pan_graph_program_views *views = next.views + p;
      status = PAN_GRAPH_TSU_UNSUPPORTED;
      if (memcmp(info, &schedule->programs[p].plan.info, sizeof(*info)) ||
          !same_boundary(views, &schedule->programs[p].plan.boundary) ||
          views->input_count > INT32_MAX || views->output_count > INT32_MAX ||
          views->input_count > INT32_MAX - views->output_count)
         goto cleanup;
      size_t count = views->input_count + views->output_count;
      if (reference_count > INT32_MAX - count)
         goto cleanup;
      const struct pan_graph_program_cost_plan *plan =
         &schedule->programs[p].plan;
      if (!plan->read_count || !plan->read_tensors ||
          plan->read_count > INT32_MAX - views->output_count ||
          memory_count > INT32_MAX - plan->read_count - views->output_count)
         goto cleanup;
      memory_count += plan->read_count + views->output_count;
      if (max_views < count)
         max_views = count;
      programs[p] =
         (struct pan_graph_memory_lifetime_program){p, reference_count, count};
      reference_count += count;
   }
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   if (reference_count > SIZE_MAX / sizeof(*references))
      goto cleanup;
   references = malloc(reference_count * sizeof(*references));
   status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (reference_count && !references)
      goto cleanup;
   for (uint32_t t = 0; t < tensor_count; t++) {
      bool input = t >= add_count + 2, output = t == add_count;
      tensors[t].uid = t + 1;
      tensors[t].domain = input    ? PAN_GRAPH_MEMORY_RESOURCE_INPUT
                          : output ? PAN_GRAPH_MEMORY_RESOURCE_OUTPUT
                                   : PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT;
      lifetime_tensors[t].uid = t + 1;
      lifetime_tensors[t].kind = input ? 3 : output ? 4 : 1;
      if (input || output) {
         unsigned b = output ? 2 : t - add_count - 2;
         memcpy(tensors[t].binding, bindings[b].binding,
                sizeof(tensors[t].binding));
         tensors[t].ordinal = bindings[b].ordinal;
      }
   }
   for (size_t p = 0; p < program_count; p++) {
      const struct pan_graph_program_views *views = next.views + p;
      const struct pan_graph_program_cost_plan *plan =
         &schedule->programs[p].plan;
      if (plan->program.section_count < views->output_count ||
          !plan->sections) {
         status = PAN_GRAPH_TSU_UNSUPPORTED;
         goto cleanup;
      }
      uint32_t output_begin = plan->program.section_count - views->output_count;
      for (unsigned side = 0; side < 2; side++) {
         size_t count = side ? views->output_count : views->input_count;
         const struct pan_graph_program_view *v =
            side ? views->outputs : views->inputs;
         size_t begin =
            programs[p].first_reference + (side ? views->input_count : 0);
         for (size_t i = 0; i < count; i++) {
            uint32_t t = v[i].tensor;
            references[begin + i] = t;
            if (t == add_count + 1 || t >= tensor_count ||
                v[i].bounds.rank != 3) {
               status = PAN_GRAPH_TSU_UNSUPPORTED;
               goto cleanup;
            }
            struct pan_graph_memory_descriptor_input *descriptor =
               &tensors[t].descriptor;
            if (!descriptor->rank) {
               descriptor->rank = 3;
               descriptor->format = t >= add_count + 2
                                       ? PAN_GRAPH_MEMORY_DESCRIPTOR_I8
                                       : PAN_GRAPH_MEMORY_DESCRIPTOR_I32;
               descriptor->layout = PAN_GRAPH_MEMORY_LAYOUT_LINEAR;
               for (unsigned axis = 0; axis < 3; axis++) {
                  if (v[i].bounds.low[axis] || v[i].bounds.high[axis] <= 0) {
                     status = PAN_GRAPH_TSU_UNSUPPORTED;
                     goto cleanup;
                  }
                  descriptor->dimensions[axis] = v[i].bounds.high[axis];
               }
            }
            if (side && t != add_count) {
               const struct pan_graph_memory_cost_section *section =
                  plan->sections + output_begin + i;
               if (section->type != PAN_GRAPH_V15_MEMORY_OUTPUT ||
                   section->layout < PAN_GRAPH_MEMORY_COST_BLOCKED_260 ||
                   section->layout > PAN_GRAPH_MEMORY_COST_BLOCKED_262) {
                  status = PAN_GRAPH_TSU_UNSUPPORTED;
                  goto cleanup;
               }
               descriptor->layout =
                  (enum pan_graph_memory_layout)section->layout;
               memcpy(descriptor->block, section->block,
                      sizeof(descriptor->block));
            }
         }
      }
   }
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   for (uint32_t t = 0; t < tensor_count; t++) {
      if (!tensors[t].descriptor.rank)
         continue;
      struct pan_graph_memory_descriptor descriptor;
      if (!pan_graph_memory_descriptor_template(&tensors[t].descriptor,
                                                &descriptor))
         goto cleanup;
      if (lifetime_tensors[t].kind == 1) {
         if (tensors[t].descriptor.layout == PAN_GRAPH_MEMORY_LAYOUT_LINEAR)
            goto cleanup;
         lifetime_tensors[t].storage_size = descriptor.size;
      }
   }
   const struct pan_graph_memory_lifetime_input life = {
      .tensors = lifetime_tensors,
      .programs = programs,
      .references = references,
      .tensor_count = tensor_count,
      .program_count = program_count,
      .reference_count = reference_count,
      .outer_group_count = program_count,
   };
   enum pan_graph_memory_lifetime_status life_status =
      pan_graph_memory_lifetimes_create(&life, &next.lifetimes);
   if (life_status != PAN_GRAPH_MEMORY_LIFETIME_SUCCESS) {
      status = life_status == PAN_GRAPH_MEMORY_LIFETIME_NO_MEMORY
                  ? PAN_GRAPH_TSU_OUT_OF_MEMORY
                  : PAN_GRAPH_TSU_UNSUPPORTED;
      goto cleanup;
   }
   for (uint32_t t = 0; t < tensor_count; t++) {
      if (next.lifetimes.identities[t].group != UINT32_MAX)
         tensors[t].allocation_offset = next.lifetimes.identities[t].offset;
   }
   const struct pan_graph_memory_resource_input input = {
      .tensors = tensors,
      .references = references,
      .tensor_count = tensor_count,
      .reference_count = reference_count,
      .input_count = 2,
      .output_count = 1,
   };
   enum pan_graph_memory_resource_status resource_status =
      pan_graph_memory_resources_create(&input, &next.resources);
   if (resource_status != PAN_GRAPH_MEMORY_RESOURCE_SUCCESS) {
      status = resource_status == PAN_GRAPH_MEMORY_RESOURCE_NO_MEMORY
                  ? PAN_GRAPH_TSU_OUT_OF_MEMORY
                  : PAN_GRAPH_TSU_UNSUPPORTED;
      goto cleanup;
   }
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   if (SIZE_MAX / program_count < sizeof(*next.program_bindings) ||
       SIZE_MAX / memory_count < sizeof(*next.memories) ||
       SIZE_MAX / max_views < sizeof(*policy_views) ||
       SIZE_MAX / max_views < sizeof(*snapshots) ||
       SIZE_MAX / max_views < sizeof(*lookup_views))
      goto cleanup;
   next.program_bindings = calloc(program_count, sizeof(*next.program_bindings));
   next.memories = calloc(memory_count, sizeof(*next.memories));
   policy_views = calloc(max_views, sizeof(*policy_views));
   snapshots = calloc(max_views, sizeof(*snapshots));
   lookup_views = calloc(max_views, sizeof(*lookup_views));
   status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!next.program_bindings || !next.memories || !policy_views ||
       !snapshots || !lookup_views)
      goto cleanup;
   size_t memory_cursor = 0;
   for (size_t p = 0; p < program_count; p++) {
      const struct pan_graph_program_views *views = next.views + p;
      const struct pan_graph_program_cost_plan *plan = &schedule->programs[p].plan;
      size_t view_count = views->input_count + views->output_count;
      for (size_t side = 0; side < 2; side++) {
         size_t count = side ? views->output_count : views->input_count;
         size_t base = side ? views->input_count : 0;
         const struct pan_graph_program_view *source =
            side ? views->outputs : views->inputs;
         for (size_t i = 0; i < count; i++) {
            uint32_t tensor = source[i].tensor, level = 5;
            if (source[i].kind == 1) {
               uint32_t group = next.lifetimes.identities[tensor].group;
               if (group >= next.lifetimes.group_count) {
                  status = PAN_GRAPH_TSU_UNSUPPORTED;
                  goto cleanup;
               }
               uint64_t bytes = next.lifetimes.requests[group].size;
               if (bytes > INT64_MAX) {
                  status = PAN_GRAPH_TSU_UNSUPPORTED;
                  goto cleanup;
               }
               level = bytes <= half_capacity ? 3 : bytes <= full_capacity ? 4 : 5;
            }
            policy_views[base + i] = (struct pan_graph_memory_policy_source_view){
               .uid = tensor + 1,
               .key = source[i].key,
               .kind = source[i].kind,
               .rank = source[i].bounds.rank,
               .policy = {.level = level},
            };
            memcpy(policy_views[base + i].low, source[i].bounds.low,
                   sizeof(policy_views[base + i].low));
            memcpy(policy_views[base + i].high, source[i].bounds.high,
                   sizeof(policy_views[base + i].high));
            snapshots[base + i] =
               (struct pan_graph_memory_policy_snapshot){source[i].key, 5};
         }
      }
      const struct pan_graph_memory_policy_finalize_input finalize = {
         policy_views, views->input_count, policy_views + views->input_count,
         views->output_count, snapshots, views->input_count,
         snapshots + views->input_count, views->output_count, 1, 1};
      struct pan_graph_memory_policy_finalized finalized = {0};
      status = pan_graph_memory_policy_finalize(&finalize, &finalized);
      if (status != PAN_GRAPH_TSU_SUCCESS)
         goto cleanup;
      for (size_t i = 0; i < view_count; i++) {
         lookup_views[i].uid = policy_views[i].uid;
         lookup_views[i].policy = finalized.policies[i];
      }
      pan_graph_memory_policy_finalized_finish(&finalized);
      const struct pan_graph_memory_policy_input lookup = {
         lookup_views, views->input_count, lookup_views + views->input_count,
         views->output_count};
      next.program_bindings[p] = (struct pan_graph_program_bindings){
         .reads = next.memories + memory_cursor,
         .writes = next.memories + memory_cursor + plan->read_count,
         .read_count = plan->read_count,
         .write_count = views->output_count,
      };
      for (size_t side = 0; side < 2; side++) {
         size_t count = side ? views->output_count : plan->read_count;
         for (size_t i = 0; i < count; i++) {
            uint32_t tensor = side ? views->outputs[i].tensor
                                   : plan->read_tensors[i];
            struct pan_graph_memory_policy policy;
            if (tensor >= tensor_count ||
                next.resources.references[tensor].slot32 == UINT32_MAX ||
                !pan_graph_memory_policy_lookup(&lookup, tensor + 1, &policy)) {
               status = PAN_GRAPH_TSU_UNSUPPORTED;
               goto cleanup;
            }
            const struct pan_graph_memory_resource_reference *reference =
               next.resources.references + tensor;
            next.memories[memory_cursor++] = (struct pan_graph_v15_memory){
               .domain = reference->domain,
               .offset = reference->slot32,
               .level = policy.level,
               .override = policy.override,
               .flags = policy.flags,
               .value = policy.value,
               .extra = side && lifetime_tensors[tensor].kind == 1,
            };
         }
      }
   }
   *result = next;
   next = (struct pan_graph_program_memory){0};
   status = PAN_GRAPH_TSU_SUCCESS;
cleanup:
   free(programs);
   free(tensors);
   free(lifetime_tensors);
   free(references);
   free(policy_views);
   free(snapshots);
   free(lookup_views);
   pan_graph_program_memory_finish(&next);
   return status;
}
