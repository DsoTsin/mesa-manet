#include "pan_graph_memory_policy.h"

#include <limits.h>
#include <stdlib.h>

bool
pan_graph_memory_policy_lookup(
   const struct pan_graph_memory_policy_input *input, uint64_t uid,
   struct pan_graph_memory_policy *policy)
{
   if (!input || !policy || (input->input_count && !input->inputs) ||
       (input->output_count && !input->outputs) ||
       input->input_count > INT32_MAX || input->output_count > INT32_MAX)
      return false;

   struct pan_graph_memory_policy next = {.level = 5};
   for (size_t i = 0; i < input->input_count; i++) {
      if (input->inputs[i].uid == uid)
         next = input->inputs[i].policy;
   }
   for (size_t i = 0; i < input->output_count; i++) {
      if (input->outputs[i].uid == uid)
         next = input->outputs[i].policy;
   }
   *policy = next;
   return true;
}

static bool
same_geometry(const struct pan_graph_memory_policy_source_view *a,
              const struct pan_graph_memory_policy_source_view *b)
{
   if (a->uid != b->uid || a->rank != b->rank)
      return false;
   for (uint32_t i = 0; i < a->rank; i++) {
      if (a->low[i] != b->low[i] || a->high[i] != b->high[i])
         return false;
   }
   return true;
}

static enum pan_graph_tsu_status
validate_views(const struct pan_graph_memory_policy_source_view *views,
               size_t count)
{
   for (size_t i = 0; i < count; i++) {
      if (!views[i].rank)
         return PAN_GRAPH_TSU_INVALID;
      if (views[i].rank > PAN_GRAPH_MEMORY_POLICY_MAX_RANK)
         return PAN_GRAPH_TSU_UNSUPPORTED;
      for (uint32_t axis = 0; axis < views[i].rank; axis++) {
         if (views[i].low[axis] > views[i].high[axis])
            return PAN_GRAPH_TSU_INVALID;
      }
      for (size_t j = 0; j < i; j++) {
         if (views[i].key == views[j].key)
            return PAN_GRAPH_TSU_INVALID;
         if (views[i].uid == views[j].uid && views[i].kind != views[j].kind)
            return PAN_GRAPH_TSU_UNSUPPORTED;
      }
   }
   return PAN_GRAPH_TSU_SUCCESS;
}

enum pan_graph_tsu_status
pan_graph_memory_policy_finalize(
   const struct pan_graph_memory_policy_finalize_input *input,
   struct pan_graph_memory_policy_finalized *output)
{
   if (!input || !output || (input->input_count && !input->inputs) ||
       (input->output_count && !input->outputs) ||
       (input->input_snapshot_count && !input->input_snapshot) ||
       (input->output_snapshot_count && !input->output_snapshot) ||
       input->input_count > INT32_MAX || input->output_count > INT32_MAX ||
       input->input_snapshot_count > INT32_MAX ||
       input->output_snapshot_count > INT32_MAX ||
       input->input_count > SIZE_MAX - input->output_count)
      return PAN_GRAPH_TSU_INVALID;

   size_t count = input->input_count + input->output_count;
   if (count > SIZE_MAX / sizeof(struct pan_graph_memory_policy))
      return PAN_GRAPH_TSU_INVALID;

   enum pan_graph_tsu_status status =
      validate_views(input->inputs, input->input_count);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      return status;
   status = validate_views(input->outputs, input->output_count);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      return status;
   for (size_t i = 0; i < input->input_count; i++) {
      for (size_t j = 0; j < input->output_count; j++) {
         if (input->inputs[i].uid == input->outputs[j].uid &&
             input->inputs[i].kind != input->outputs[j].kind)
            return PAN_GRAPH_TSU_UNSUPPORTED;
      }
   }

   struct pan_graph_memory_policy *policies =
      count ? malloc(count * sizeof(*policies)) : NULL;
   if (count && !policies)
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;

   for (size_t side = 0; side < 2; side++) {
      const struct pan_graph_memory_policy_source_view *views =
         side ? input->outputs : input->inputs;
      size_t view_count = side ? input->output_count : input->input_count;
      const struct pan_graph_memory_policy_snapshot *snapshot =
         side ? input->output_snapshot : input->input_snapshot;
      size_t snapshot_count =
         side ? input->output_snapshot_count : input->input_snapshot_count;
      size_t base = side ? input->input_count : 0;
      for (size_t i = 0; i < view_count; i++) {
         struct pan_graph_memory_policy next = views[i].policy;
         bool seen = false;
         if (!side) {
            for (size_t j = 0; j < i; j++) {
               if (same_geometry(&views[i], &views[j])) {
                  seen = true;
                  break;
               }
            }
         }
         if (!seen) {
            for (size_t j = 0; j < snapshot_count; j++) {
               if (snapshot[j].key == views[i].key)
                  next.level = snapshot[j].level;
            }
         }
         bool eligible = views[i].kind >= 1 && views[i].kind <= 3;
         next.flags = 0;
         next.value = eligible && input->load_speculative == 1;
         next.override = eligible && input->store_speculative == 1;
         policies[base + i] = next;
      }
   }

   *output = (struct pan_graph_memory_policy_finalized){
      policies, input->input_count, input->output_count};
   return PAN_GRAPH_TSU_SUCCESS;
}

void
pan_graph_memory_policy_finalized_finish(
   struct pan_graph_memory_policy_finalized *output)
{
   if (!output)
      return;
   free(output->policies);
   *output = (struct pan_graph_memory_policy_finalized){0};
}
