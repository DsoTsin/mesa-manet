#include "pan_graph_memory_cost.h"
#include "pan_graph_memory_bytes.h"
#include "pan_graph_memory_layout.h"

#include <limits.h>
#include <math.h>
#include <string.h>

static float
multiply(float a, float b)
{
   volatile float result = a * b;
   return result;
}

static uint64_t
divide_up(uint64_t a, uint64_t b)
{
   return a / b + (a % b != 0);
}

static bool
add(uint64_t *a, uint64_t b)
{
   if (b > INT64_MAX || *a > INT64_MAX - b)
      return false;
   *a += b;
   return true;
}

static bool
product(uint64_t *a, uint64_t b)
{
   if (b > INT64_MAX || (b && *a > INT64_MAX / b))
      return false;
   *a *= b;
   return true;
}

static bool
integer(float value, uint64_t *result)
{
   if (!isfinite(value) || value < 0.0f || value >= 0x1p63f)
      return false;
   *result = value;
   return true;
}

static unsigned
stream_group(unsigned budget, uint64_t index, uint64_t count)
{
   unsigned cutoff, group;
   if (budget >= 9 && budget <= 15) {
      cutoff = 16 - budget;
      group = index < cutoff ? 1 : 2;
   } else if (budget >= 16 && budget <= 24) {
      cutoff = 48 - 2 * budget;
      group = index < cutoff ? 2 : 3;
   } else {
      cutoff = budget == 32 ? 32 : budget;
      group = budget == 32 ? 2 : 1;
   }
   if (group > count - index) {
      if (index < cutoff)
         cutoff = 0;
      unsigned remainder = (count - cutoff) % group;
      if (remainder)
         group = remainder;
   }
   return group;
}

static bool
units(const struct pan_graph_memory_cost_section *section,
      const struct pan_graph_memory_cost_target *target, uint64_t *result)
{
   const uint32_t *full = section->full, *slice = section->transfer;
   unsigned bytes = section->bits / 8;
   bool aligned = !((((uint64_t)full[0] * bytes) |
                     ((uint64_t)full[1] * bytes) |
                     ((uint64_t)full[2] * bytes) |
                     ((uint64_t)slice[0] * bytes)) & 7);
   uint64_t width = (uint64_t)slice[0] * bytes;
   unsigned factor;
   if (section->type == PAN_GRAPH_V15_MEMORY_INPUT) {
      if (aligned)
         factor = width < 33 ? 1 : 2;
      else
         factor = (uint64_t)full[0] * bytes >= 32 ? 6 :
                  (uint64_t)full[0] * bytes >= 9 ? 3 : 4;
   } else {
      factor = aligned ? 4 : width > 7 ? 8 : 64 / width;
   }
   uint64_t value = divide_up(slice[0],
                              target->alignment / (section->bits / 2));
   if (!product(&value, divide_up(slice[1], 4)) ||
       !product(&value, slice[2]) || !product(&value, factor))
      return false;
   *result = value;
   return true;
}

static void
ratios(const struct pan_graph_cost_workload *workload, uint64_t a, uint64_t b,
       float result[7])
{
   const uint64_t index[2] = {a, b};
   for (unsigned axis = 0; axis < 7; axis++) {
      if (axis >= workload->rank) {
         result[axis] = 1.0f;
         continue;
      }
      uint64_t extent = workload->extent[axis];
      for (unsigned i = 0; i < 2; i++) {
         if (workload->step[i] && workload->axis[i] == axis) {
            extent -= index[i] * workload->step[i];
            if (extent > workload->step[i])
               extent = workload->step[i];
         }
      }
      result[axis] = (float)extent / (float)workload->transfer[axis];
   }
}

static bool
section_cost(const struct pan_graph_memory_cost_section *section,
             const struct pan_graph_memory_cost_target *target,
             const struct pan_graph_cost_workload *workload,
             const uint64_t count[2], struct pan_graph_cost_report *report,
             uint64_t *section_peak)
{
   bool read = section->type == PAN_GRAPH_V15_MEMORY_INPUT;
   bool blocked = section->layout != PAN_GRAPH_MEMORY_COST_COMPACT_LINEAR;
   struct pan_graph_memory_layout_target layout_target = {
      .alignment = target->alignment,
   };
   struct pan_graph_memory_layout_input layout_input = {
      .layout = (enum pan_graph_memory_layout)section->layout,
      .bits = section->bits,
      .replication_rank = section->replication_rank,
      .read = read,
   };
   memcpy(layout_target.block, section->block, sizeof(layout_target.block));
   memcpy(layout_input.full, section->full, sizeof(layout_input.full));
   memcpy(layout_input.transfer, section->transfer, sizeof(layout_input.transfer));
   uint64_t streams = count[0] * count[1], stride = section->bits / 8;
   uint64_t bytes = 1;
   bool same = true;
   for (unsigned i = 0; !blocked && i < 3; i++) {
      uint64_t granule = divide_up(target->memory[5].line_size, stride);
      if (granule > section->full[i])
         granule = section->full[i];
      uint64_t padded = divide_up(section->transfer[i], granule) * granule;
      same &= padded == section->transfer[i];
      if (!product(&bytes, read ? padded : section->transfer[i]) ||
          !product(&stride, section->full[i]))
         return false;
   }
   uint64_t base = 0;
   if (!blocked &&
       (!product(&bytes, section->bits / 8) || !units(section, target, &base)))
      return false;
   uint64_t peak = 0, maximum_units = 0;
   for (uint64_t a = 0; a < count[0]; a++) {
      for (uint64_t b = 0; b < count[1]; b++) {
         float ratio[7], multiplier = 1.0f;
         ratios(workload, a, b, ratio);
         uint64_t total = bytes;
         uint64_t scaled, cache_bytes;
         if (blocked) {
            memcpy(layout_input.ratios, ratio, sizeof(layout_input.ratios));
            struct pan_graph_memory_bytes_result traffic;
            struct pan_graph_memory_layout_result transfer;
            if (!pan_graph_memory_bytes_estimate(&layout_target, &layout_input,
                                                5, &traffic) ||
                !pan_graph_memory_layout_units(&layout_target, &layout_input,
                                               &transfer))
               return false;
            bytes = traffic.local_bytes;
            total = traffic.total_bytes;
            cache_bytes = traffic.cache_bytes;
            base = transfer.base;
            scaled = transfer.cycles;
            same = !memcmp(traffic.padded, section->transfer,
                           sizeof(traffic.padded));
         } else {
            for (unsigned i = 0; i < section->replication_rank; i++) {
               if (!integer(multiply((float)total, ratio[i]), &total))
                  return false;
               multiplier = multiply(multiplier, ratio[i]);
            }
            cache_bytes = !read && !same ? total : 0;
            if (!integer(multiply(multiplier, (float)base), &scaled))
               return false;
         }
         if (!add(&report->resource_bytes[read ? 0 : 1][5], total) ||
             !add(&report->resource_bytes[read ? 0 : 1][2], cache_bytes))
            return false;
         unsigned group = stream_group(target->parallel_budget,
                                        a * count[1] + b, streams);
         float rate = multiply(target->memory[5].bandwidth[read ? 0 : 1],
                               (float)target->memory[5].parallel) /
                      (float)streams;
         float limit = (float)target->issue_width / (float)group;
         if (rate > limit)
            rate = limit;
         uint64_t latency;
         if (!integer((float)bytes / rate, &latency) ||
             !add(&latency, read ? target->memory[5].latency[0] : 0) ||
             !add(&latency, base))
            return false;
         if (!read && !same) {
            uint64_t extra;
            if (!integer((float)bytes / target->memory[2].bandwidth[1],
                         &extra) || !add(&latency, extra))
               return false;
         }
         if (latency > peak)
            peak = latency;
         if (scaled > maximum_units)
            maximum_units = scaled;
      }
   }
   *section_peak = peak;
   return add(&report->engine_cycles[read ? 1 : 6], maximum_units);
}

bool
pan_graph_memory_cost_estimate(
   const struct pan_graph_memory_cost_target *target,
   const struct pan_graph_memory_cost_program *program,
   struct pan_graph_cost_report reports[3])
{
   if (!target || !program || !reports ||
       (program->section_count && !program->sections) ||
       (program->input_view_count && !program->input_views) ||
       target->alignment != 32 || !target->issue_width ||
       !target->parallel_budget || target->parallel_budget > INT32_MAX ||
       program->section_count > INT32_MAX ||
       program->input_view_count > INT32_MAX ||
       program->output_view_count > INT32_MAX ||
       program->resource_count > INT32_MAX)
      return false;
   for (unsigned level = 2; level < 6; level++) {
      if (!target->memory[level].parallel || !target->memory[level].line_size)
         return false;
      for (unsigned direction = 0; direction < 2; direction++) {
         if (!isfinite(target->memory[level].bandwidth[direction]) ||
             target->memory[level].bandwidth[direction] <= 0.0f)
            return false;
      }
   }
   const struct pan_graph_cost_workload *workload = &program->workload;
   bool rank3 = workload->rank == 3;
   if ((!rank3 && workload->rank != 7) ||
       (workload->step[1] && !workload->step[0]))
      return false;
   uint64_t count[2] = {1, 1};
   for (unsigned i = 0; i < workload->rank; i++) {
      if (!workload->extent[i] || workload->extent[i] > INT32_MAX ||
          !workload->transfer[i] ||
          workload->transfer[i] > workload->extent[i] ||
          workload->transfer[i] > 4095)
         return false;
   }
   if (!rank3 &&
       (workload->extent[1] != 1 || workload->extent[5] != 1 ||
        workload->extent[6] != 1 || workload->transfer[2] != 1))
      return false;
   for (unsigned i = 0; i < 2; i++) {
      if (workload->step[i]) {
         if (workload->axis[i] >= (rank3 ? 3 : 4) ||
             workload->step[i] > INT32_MAX ||
             (i && workload->axis[0] == workload->axis[1]))
            return false;
         count[i] = divide_up(workload->extent[workload->axis[i]],
                              workload->step[i]);
      }
   }
   uint64_t streams = count[0];
   if (!product(&streams, count[1]) || streams > INT32_MAX)
      return false;
   for (unsigned i = 0; i < program->section_count; i++) {
      const struct pan_graph_memory_cost_section *section = &program->sections[i];
      if (section->type < PAN_GRAPH_V15_MEMORY_INPUT ||
          section->type >= PAN_GRAPH_V15_SECTION_TYPES ||
          section->type == PAN_GRAPH_V15_MEMORY_CONSTANT)
         return false;
      bool read = section->type == PAN_GRAPH_V15_MEMORY_INPUT;
      bool blocked = section->layout != PAN_GRAPH_MEMORY_COST_COMPACT_LINEAR;
      if (section->retain_read_peak && !read)
         return false;
      if (!read && section->type != PAN_GRAPH_V15_MEMORY_OUTPUT) {
         if (blocked)
            return false;
         continue;
      }
      if (section->bits != (rank3 ? 32 : read ? 8 : 32) ||
          section->replication_rank != (rank3 ? 3 : read ? 5 : 4))
         return false;
      if (blocked) {
         uint32_t expected[4];
         if ((!rank3 && section->transfer[2] != 1) ||
             !pan_graph_memory_layout_v15_block(
                (enum pan_graph_memory_layout)section->layout,
                section->bits, expected) ||
             memcmp(expected, section->block, sizeof(expected)))
            return false;
      }
      for (unsigned axis = 0; axis < 3; axis++) {
         if (!section->full[axis] || section->full[axis] > INT32_MAX ||
             !section->transfer[axis] ||
             section->transfer[axis] > section->full[axis] ||
             section->transfer[axis] > 4095)
            return false;
         if (rank3 &&
             (section->full[axis] != workload->extent[2 - axis] ||
              section->transfer[axis] != workload->transfer[2 - axis]))
            return false;
      }
   }
   struct pan_graph_cost_report value[3] = {0};
   unsigned writes = 0;
   uint64_t metadata = target->descriptor_bytes[0];
   for (unsigned i = 0; i < program->input_view_count; i++) {
      enum pan_graph_memory_cost_view view = program->input_views[i];
      if ((view != PAN_GRAPH_MEMORY_COST_TENSOR_VIEW &&
           view != PAN_GRAPH_MEMORY_COST_WEIGHT_VIEW) ||
          !add(&metadata, target->descriptor_bytes[view]))
         return false;
   }
   uint64_t output_bytes = program->output_view_count;
   uint64_t resource_bytes = program->resource_count;
   if (!product(&output_bytes, target->descriptor_bytes[1]) ||
       !product(&resource_bytes, target->descriptor_bytes[3]) ||
       !add(&metadata, output_bytes) || !add(&metadata, resource_bytes))
      return false;
   for (unsigned i = 0; i < program->section_count; i++) {
      const struct pan_graph_memory_cost_section *section = &program->sections[i];
      if (!add(&metadata, target->section_bytes[section->type]))
         return false;
      if (section->type != PAN_GRAPH_V15_MEMORY_INPUT &&
          section->type != PAN_GRAPH_V15_MEMORY_OUTPUT)
         continue;
      uint64_t peak;
      if (!section_cost(section, target, workload, count, &value[0], &peak))
         return false;
      if (section->type == PAN_GRAPH_V15_MEMORY_OUTPUT)
         value[0].peak_resource[1] = writes++ ? 0 : peak;
      else if (!section->retain_read_peak)
         value[0].peak_resource[0] = 0;
      else if (peak > value[0].peak_resource[0])
         value[0].peak_resource[0] = peak;
   }
   float bandwidth[6] = {0};
   for (uint64_t index = 0; index < streams; index++) {
      unsigned group = stream_group(target->parallel_budget, index, streams);
      for (unsigned level = 2; level < 6; level++) {
         float rate = target->memory[level].bandwidth[0];
         if (level == 3 || level == 5) {
            rate = multiply(rate, (float)target->memory[level].parallel) /
                   (float)streams;
            float limit = (float)target->issue_width / (float)group;
            if (rate > limit)
               rate = limit;
         }
         bandwidth[level] += rate;
      }
   }
   for (unsigned level = 2; level < 6; level++) {
      uint64_t bytes = value[0].resource_bytes[0][level];
      if (!add(&bytes, value[0].resource_bytes[1][level]) ||
          !integer((float)bytes / bandwidth[level],
                   &value[0].memory_cycles[level]))
         return false;
   }
   for (unsigned kind = 0; kind < 8; kind++) {
      if (value[0].engine_cycles[kind] > value[0].critical_cycles) {
         value[0].critical_cycles = value[0].engine_cycles[kind];
         value[0].critical_kind = kind;
      }
   }
   memcpy(value[1].peak_resource, value[0].peak_resource,
          sizeof(value[1].peak_resource));
   value[1].critical_kind = value[0].critical_kind;
   uint64_t rounded = divide_up(metadata, target->memory[5].line_size);
   if (!product(&rounded, target->memory[5].line_size) ||
       !integer((float)rounded / target->memory[5].bandwidth[0],
                &value[2].cycles))
      return false;
   memcpy(reports, value, sizeof(value));
   return true;
}
