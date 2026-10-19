#include "pan_graph_vector_v15.h"
#include "pan_graph_nrt_plan.h"

#include <limits.h>
#include <string.h>

static bool
valid_memory(const struct pan_graph_v15_memory *memory)
{
   return memory->domain <= 3 && memory->domain_bias == 0 &&
          memory->level >= 2 && memory->level <= 5 &&
          memory->override <= 1 && memory->flags <= 1 &&
          memory->value <= 3 && memory->extra <= 1;
}

static bool
same_memory(const struct pan_graph_v15_memory *a,
            const struct pan_graph_v15_memory *b)
{
   return a->domain == b->domain && a->offset == b->offset &&
          a->level == b->level && a->override == b->override &&
          a->flags == b->flags && a->value == b->value && a->extra == b->extra;
}

enum pan_graph_tsu_status
pan_graph_vector_v15_compile_candidate(
   const struct pan_graph_vector_program *info,
   const struct pan_graph_nrt_target *target,
   struct pan_graph_vector_kernel *kernel)
{
   if (!info || !target || !kernel || !info->reads || !info->adds ||
       !info->writes || !info->read_count || !info->add_count ||
       !info->write_count)
      return PAN_GRAPH_TSU_INVALID;
   if (info->read_count > 7 || info->add_count > 7 || info->write_count > 7)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   if (target->shared_size != 65536 || target->alignment != 32 ||
       target->banks != 8 || target->bank_parts != 1)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   for (unsigned i = 0; i < 3; i++) {
      if (!info->full[i] || !info->transfer[i] ||
          info->transfer[i] > info->full[i])
         return PAN_GRAPH_TSU_INVALID;
      if (info->transfer[i] > 4095)
         return PAN_GRAPH_TSU_UNSUPPORTED;
   }
   const uint32_t *loops = info->loops;
   if ((loops[3] && (!loops[2] || loops[0] == loops[1])) ||
       (loops[2] && (loops[0] >= 4 || loops[2] > INT32_MAX)) ||
       (loops[3] && (loops[1] >= 4 || loops[3] > INT32_MAX)))
      return PAN_GRAPH_TSU_INVALID;
   for (unsigned i = 0; i < info->read_count; i++) {
      if (!valid_memory(&info->reads[i]))
         return PAN_GRAPH_TSU_INVALID;
   }
   unsigned value_count = info->read_count + info->add_count;
   bool live[14] = {false};
   for (unsigned i = 0; i < info->add_count; i++) {
      for (unsigned j = 0; j < 2; j++) {
         if (info->adds[i].input[j] >= info->read_count + i)
            return PAN_GRAPH_TSU_INVALID;
      }
   }
   for (unsigned i = 0; i < info->write_count; i++) {
      const struct pan_graph_vector_write *write = &info->writes[i];
      if (write->value < info->read_count || write->value >= value_count ||
          !valid_memory(&write->memory))
         return PAN_GRAPH_TSU_INVALID;
      for (unsigned j = 0; j < i; j++) {
         if (write->value == info->writes[j].value &&
             !same_memory(&write->memory, &info->writes[j].memory))
            return PAN_GRAPH_TSU_UNSUPPORTED;
      }
      live[write->value] = true;
   }
   for (unsigned i = info->add_count; i-- > 0;) {
      if (!live[info->read_count + i])
         return PAN_GRAPH_TSU_INVALID;
      live[info->adds[i].input[0]] = true;
      live[info->adds[i].input[1]] = true;
   }
   for (unsigned i = 0; i < info->read_count; i++) {
      if (!live[i])
         return PAN_GRAPH_TSU_INVALID;
   }

   struct pan_graph_v15_section raw[21] = {0};
   unsigned resource_ids[14], producer[14], resource_values[14];
   unsigned resource_count = 0, section_count = 0;
   for (unsigned i = 0; i < value_count; i++)
      resource_ids[i] = UINT32_MAX;
   for (unsigned i = 0; i < info->add_count; i++) {
      unsigned section = section_count++, value = info->read_count + i;
      raw[section].type = PAN_GRAPH_V15_VE;
      raw[section].ve_family = PAN_GRAPH_V15_VE_BINARY;
      producer[value] = section;
      for (unsigned j = 0; j < 2; j++) {
         unsigned input = info->adds[i].input[j];
         if (resource_ids[input] == UINT32_MAX) {
            resource_ids[input] = resource_count++;
            producer[input] = section_count;
            raw[section_count].type = PAN_GRAPH_V15_MEMORY_INPUT;
            raw[section_count].common.output = resource_ids[input];
            raw[section_count++].memory = info->reads[input];
         }
         raw[section].common.input[j] = resource_ids[input];
         if (!j)
            resource_ids[value] = resource_count++;
      }
      raw[section].common.output = resource_ids[value];
   }
   for (unsigned i = 0; i < info->write_count; i++) {
      struct pan_graph_v15_section *section = &raw[section_count++];
      section->type = PAN_GRAPH_V15_MEMORY_OUTPUT;
      section->common.output = UINT32_MAX;
      section->common.input[0] = resource_ids[info->writes[i].value];
      section->memory = info->writes[i].memory;
   }
   for (unsigned i = 0; i < section_count; i++) {
      struct pan_graph_v15_section *section = &raw[i];
      unsigned inputs = section->type == PAN_GRAPH_V15_VE ? 2 :
                        section->type == PAN_GRAPH_V15_MEMORY_OUTPUT ? 1 : 0;
      section->common.registers = 4;
      for (unsigned j = inputs; j < 6; j++)
         section->common.input[j] = UINT32_MAX;
   }
   for (unsigned i = 0; i < value_count; i++)
      resource_values[resource_ids[i]] = i;
   struct pan_graph_v15_section sections[21] = {0};
   unsigned instances[8] = {0}, sorted = 0;
   bool visited[21] = {false};
   while (sorted < section_count) {
      bool progress = false;
      for (unsigned i = 0; i < section_count; i++) {
         if (visited[i])
            continue;
         bool ready = true;
         for (unsigned j = 0; j < 6; j++) {
            unsigned input = raw[i].common.input[j];
            if (input != UINT32_MAX &&
                !visited[producer[resource_values[input]]])
               ready = false;
         }
         if (!ready)
            continue;
         sections[sorted] = raw[i];
         sections[sorted++].common.instance = instances[raw[i].type]++;
         visited[i] = true;
         progress = true;
      }
      if (!progress)
         return PAN_GRAPH_TSU_INVALID;
   }

   struct pan_graph_nrt_plan_request requests[14] = {0};
   unsigned ids[14], request_count = 0;
   bool external[14] = {false};
   for (unsigned i = 0; i < info->read_count; i++) {
      ids[request_count] = resource_ids[i];
      external[i] = true;
      requests[request_count].allocation.front = true;
      requests[request_count++].buffers = PAN_GRAPH_NRT_BUFFER_INPUT;
   }
   for (unsigned i = 0; i < info->write_count; i++) {
      unsigned value = info->writes[i].value;
      if (external[value])
         continue;
      external[value] = true;
      ids[request_count] = resource_ids[value];
      requests[request_count++].buffers = PAN_GRAPH_NRT_BUFFER_OUTPUT;
   }
   for (unsigned i = 0; i < section_count; i++) {
      if (sections[i].type != PAN_GRAPH_V15_VE)
         continue;
      unsigned id = sections[i].common.output;
      if (!external[resource_values[id]])
         ids[request_count++] = id;
   }
   for (unsigned i = 0; i < request_count; i++) {
      struct pan_graph_nrt_request *request = &requests[i].allocation;
      request->dimensions[0] = info->transfer[2];
      request->dimensions[1] = info->transfer[1];
      request->dimensions[2] = info->transfer[0];
      request->dimensions[3] = 1;
      request->element_bits = 32;
      request->buffers = 2;
   }
   struct pan_graph_nrt_plan plan;
   if (pan_graph_nrt_plan_search(target, requests, request_count, 16, &plan) !=
       PAN_GRAPH_NRT_SUCCESS)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct pan_graph_v15_nrt resources[14] = {0};
   for (unsigned i = 0; i < request_count; i++) {
      const struct pan_graph_nrt_layout *layout = &plan.layouts[i];
      struct pan_graph_v15_nrt *resource = &resources[ids[i]];
      *resource = (struct pan_graph_v15_nrt){
         .kind = layout->kind, .width = layout->buffers,
         .size = layout->size, .flags = layout->base,
         .tile = layout->banks, .format = layout->format, .binding = 4,
      };
      for (unsigned j = 0; j < 4; j++) {
         if (layout->dimensions[j] > 2047)
            return PAN_GRAPH_TSU_UNSUPPORTED;
         resource->dimensions[j] = layout->dimensions[j];
      }
   }

   const int64_t low[4] = {0};
   const int64_t high[4] = {
      0, (int64_t)info->full[0] - 1, (int64_t)info->full[1] - 1,
      (int64_t)info->full[2] - 1,
   };
   int64_t affine[25] = {0};
   for (unsigned i = 1; i < 5; i++)
      affine[5 * i + i] = 1;
   const struct pan_graph_tsu_transform transform = {
      .kind = PAN_GRAPH_TSU_AFFINE, .rows = 5, .columns = 5,
      .low = affine, .high = affine,
   };
   const struct pan_graph_tsu_info tsu = {
      .rank = 4, .low = low, .high = high,
      .count = 1, .transforms = &transform,
   };
   uint32_t section_tsu[21] = {0};
   struct pan_graph_v15_program_info program = {
      .section_count = section_count, .sections = sections,
      .section_tsu = section_tsu, .resource_count = resource_count,
      .resources = resources, .tsu_count = 1, .tsu = &tsu,
   };
   for (unsigned i = 0; i < 8; i++) {
      program.header.output[i] = i >= 1 && i <= 3 ? info->full[i - 1] : 1;
      program.header.transfer[i] = i >= 1 && i <= 3 ? info->transfer[i - 1] : 1;
   }
   struct pan_graph_vector_kernel result = {.extent = {1}};
   enum pan_graph_tsu_status status =
      pan_graph_v15_compile(&program, &result.program);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      return status;
   memcpy(result.extent + 1, info->full, sizeof(info->full));
   memcpy(result.loops, info->loops, sizeof(info->loops));
   *kernel = result;
   return PAN_GRAPH_TSU_SUCCESS;
}
