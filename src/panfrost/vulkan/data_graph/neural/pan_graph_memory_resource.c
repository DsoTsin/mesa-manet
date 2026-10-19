#include "pan_graph_memory_resource.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

void
pan_graph_memory_resources_cleanup(struct pan_graph_memory_resources *resources)
{
   if (!resources)
      return;
   free(resources->storage);
   free(resources->references);
   memset(resources, 0, sizeof(*resources));
}

enum pan_graph_memory_resource_status
pan_graph_memory_resources_create(const struct pan_graph_memory_resource_input *input,
                                 struct pan_graph_memory_resources *result)
{
   if (!input || !result || (input->tensor_count && !input->tensors) ||
       (input->reference_count && !input->references) ||
       input->tensor_count > INT32_MAX || input->input_count > input->tensor_count ||
       input->output_count > input->tensor_count)
      return PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED;

   struct pan_graph_memory_resources next = {.tensor_count = input->tensor_count};
   if (!input->tensor_count) {
      if (input->reference_count || input->input_count || input->output_count)
         return PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED;
      *result = next;
      return PAN_GRAPH_MEMORY_RESOURCE_SUCCESS;
   }
   if (SIZE_MAX / input->tensor_count < sizeof(*next.references))
      return PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED;
   next.references = calloc(input->tensor_count, sizeof(*next.references));
   if (!next.references)
      return PAN_GRAPH_MEMORY_RESOURCE_NO_MEMORY;
   for (uint32_t i = 0; i < input->tensor_count; i++) {
      const struct pan_graph_memory_resource_tensor *tensor = &input->tensors[i];
      if (tensor->domain != PAN_GRAPH_MEMORY_RESOURCE_INPUT &&
          tensor->domain != PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT &&
          tensor->domain != PAN_GRAPH_MEMORY_RESOURCE_OUTPUT)
         goto unsupported;
      next.references[i].slot32 = UINT32_MAX;
      next.references[i].domain = tensor->domain;
   }

   uint32_t total = 0;
   for (uint32_t i = 0; i < input->reference_count; i++) {
      uint32_t id = input->references[i];
      if (id >= input->tensor_count)
         goto unsupported;
      if (next.references[id].slot32 != UINT32_MAX)
         continue;
      const struct pan_graph_memory_resource_tensor *tensor = &input->tensors[id];
      uint32_t domain = tensor->domain;
      uint32_t count = next.domains[domain].count;
      uint32_t ordinal = domain == PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT ? count : tensor->ordinal;
      uint32_t limit = domain == PAN_GRAPH_MEMORY_RESOURCE_INPUT ? input->input_count : input->output_count;
      if ((domain != PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT && ordinal >= limit) ||
          ordinal > UINT32_MAX / 2)
         goto unsupported;
      next.references[id].slot32 = 2 * ordinal;
      next.domains[domain].count++;
      total++;
   }
   if (next.domains[PAN_GRAPH_MEMORY_RESOURCE_INPUT].count != input->input_count ||
       next.domains[PAN_GRAPH_MEMORY_RESOURCE_OUTPUT].count != input->output_count ||
       (total && SIZE_MAX / total < sizeof(struct pan_graph_memory_resource_entry)))
      goto unsupported;
   if (total) {
      next.storage = calloc(total, sizeof(struct pan_graph_memory_resource_entry));
      if (!next.storage) {
         pan_graph_memory_resources_cleanup(&next);
         return PAN_GRAPH_MEMORY_RESOURCE_NO_MEMORY;
      }
   }
   struct pan_graph_memory_resource_entry *entries = next.storage;
   for (unsigned d = 0; d < 4; d++) {
      if (next.domains[d].count) {
         next.domains[d].entries = entries;
         entries += next.domains[d].count;
      }
   }

   for (uint32_t i = 0; i < input->tensor_count; i++) {
      if (next.references[i].slot32 == UINT32_MAX)
         continue;
      const struct pan_graph_memory_resource_tensor *tensor = &input->tensors[i];
      uint32_t ordinal = next.references[i].slot32 / 2;
      struct pan_graph_memory_resource_entry *entry = &next.domains[tensor->domain].entries[ordinal];
      if (entry->descriptor.size ||
          !pan_graph_memory_descriptor_template(&tensor->descriptor, &entry->descriptor))
         goto unsupported;
      uint64_t address = 0;
      if (tensor->domain == PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT) {
         address = tensor->allocation_offset;
         if ((address & 63) || address > UINT64_MAX - entry->descriptor.size)
            goto unsupported;
      }
      entry->descriptor.words[6] = address;
      entry->descriptor.words[7] = address >> 32;
      entry->uid = tensor->uid;
      entry->slot32 = next.references[i].slot32;
      for (unsigned j = 0; j < 3; j++)
         entry->binding[j] = tensor->domain == PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT ? UINT32_MAX : tensor->binding[j];
   }
   *result = next;
   return PAN_GRAPH_MEMORY_RESOURCE_SUCCESS;

unsupported:
   pan_graph_memory_resources_cleanup(&next);
   return PAN_GRAPH_MEMORY_RESOURCE_UNSUPPORTED;
}
