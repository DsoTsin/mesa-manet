#include "pan_graph_cost.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

static bool
add(uint64_t *sum, uint64_t value)
{
   if (value > INT64_MAX || *sum > INT64_MAX - value)
      return false;
   *sum += value;
   return true;
}

static uint64_t
maximum(uint64_t a, uint64_t b)
{
   return a > b ? a : b;
}

bool
pan_graph_cost_accumulate_serial(const struct pan_graph_cost_report *left,
                                 const struct pan_graph_cost_report *right,
                                 struct pan_graph_cost_report *result)
{
   if (!left || !right || !result)
      return false;
   struct pan_graph_cost_report report = *left;
   for (unsigned direction = 0; direction < 2; direction++) {
      if (left->peak_resource[direction] > INT64_MAX ||
          right->peak_resource[direction] > INT64_MAX)
         return false;
      for (unsigned i = 0; i < 6; i++) {
         if (!add(&report.resource_bytes[direction][i],
                  right->resource_bytes[direction][i]))
            return false;
      }
   }
   for (unsigned i = 0; i < 6; i++) {
      if (!add(&report.memory_cycles[i], right->memory_cycles[i]))
         return false;
   }
   for (unsigned i = 0; i < 8; i++) {
      if (!add(&report.engine_cycles[i], right->engine_cycles[i]))
         return false;
   }
   if (!add(&report.critical_cycles, right->critical_cycles) ||
       !add(&report.cycles, right->cycles))
      return false;
   *result = report;
   return true;
}

bool
pan_graph_cost_v15_ve_i32(enum pan_graph_cost_ve_i32_op op,
                         const uint32_t extent[4], const uint32_t transfer[4],
                         uint64_t *cycles)
{
   if (!extent || !transfer || !cycles ||
       (op != PAN_GRAPH_COST_VE_I32_IDENTITY &&
        op != PAN_GRAPH_COST_VE_I32_ADD))
      return false;
   for (unsigned i = 0; i < 4; i++) {
      if (!extent[i] || extent[i] > INT32_MAX || !transfer[i] ||
          transfer[i] > 4095)
         return false;
   }
   if (extent[1] != 1 || transfer[1] != 1)
      return false;
   uint32_t block_n = op == PAN_GRAPH_COST_VE_I32_ADD ? 2 : 4;
   uint32_t block_m = op == PAN_GRAPH_COST_VE_I32_ADD ? 4 : 2;
   uint64_t n = ((transfer[0] + block_n - 1) / block_n) * block_n;
   uint64_t m = ((transfer[3] + block_m - 1) / block_m) * block_m;
   uint64_t volume = n * m * transfer[2];
   volatile float sum = (float)(int64_t)volume + 8.0f;
   volatile float adjusted = sum - 1.0f;
   volatile float base = adjusted / 8.0f;
   uint64_t value = (uint64_t)base;
   for (unsigned i = 0; i < 4; i++) {
      volatile float ratio = (float)extent[i] / (float)transfer[i];
      volatile float product = (float)(int64_t)value * ratio;
      float rounded = ceilf(product);
      value = rounded >= 0x1p63f ? INT64_MAX : (uint64_t)rounded;
   }
   *cycles = value;
   return true;
}

bool
pan_graph_cost_combine(const struct pan_graph_cost_report a[3],
                       const struct pan_graph_cost_report b[3],
                       struct pan_graph_cost_report *result)
{
   if (!a || !b || !result)
      return false;
   struct pan_graph_cost_report report = {0};
   for (unsigned block = 0; block < 2; block++) {
      uint64_t critical = maximum(a[block].critical_cycles, b[block].critical_cycles);
      for (unsigned i = 0; i < 6; i++) {
         critical = maximum(critical, a[block].memory_cycles[i]);
         if (!add(&report.memory_cycles[i], a[block].memory_cycles[i]))
            return false;
         for (unsigned direction = 0; direction < 2; direction++) {
            if (!add(&report.resource_bytes[direction][i],
                     a[block].resource_bytes[direction][i]))
               return false;
         }
      }
      for (unsigned i = 0; i < 2; i++) {
         if (a[block].peak_resource[i] > INT64_MAX)
            return false;
         report.peak_resource[i] = maximum(report.peak_resource[i], a[block].peak_resource[i]);
      }
      for (unsigned i = 0; i < 8; i++) {
         if (!add(&report.engine_cycles[i], a[block].engine_cycles[i]) ||
             !add(&report.engine_cycles[i], b[block].engine_cycles[i]))
            return false;
      }
      if (!report.critical_kind)
         report.critical_kind = b[block].critical_kind;
      if (!add(&report.critical_cycles, b[block].critical_cycles) ||
          !add(&report.cycles, critical))
         return false;
   }
   if (!add(&report.cycles, a[2].cycles) || !add(&report.cycles, b[2].cycles))
      return false;
   *result = report;
   return true;
}

static float
chunk_weight(const struct pan_graph_cost_workload *workload,
             const uint64_t extent[PAN_GRAPH_COST_MAX_RANK])
{
   float product = 1.0f;
   for (unsigned i = 0; i < workload->rank; i++) {
      float ratio = (float)extent[i] / (float)workload->transfer[i];
      volatile float next = product * ratio;
      product = next;
   }
   return product;
}

bool
pan_graph_cost_apply_workload(const struct pan_graph_cost_workload *workload,
                              uint64_t cycles, uint64_t *result)
{
   if (!workload || !result || cycles > INT64_MAX ||
       !isfinite(workload->minimum) || workload->minimum < 0.0f)
      return false;
   if (workload->minimum == 0.0f) {
      *result = cycles;
      return true;
   }
   if (!workload->rank || workload->rank > PAN_GRAPH_COST_MAX_RANK ||
       !workload->repeats || workload->repeats > INT32_MAX ||
       !workload->latency || workload->latency > INT32_MAX ||
       (workload->step[1] && !workload->step[0]))
      return false;
   for (unsigned i = 0; i < workload->rank; i++) {
      if (!workload->extent[i] || workload->extent[i] > INT32_MAX ||
          !workload->transfer[i] || workload->transfer[i] > INT32_MAX)
         return false;
   }
   if (workload->step[0] && workload->axis[0] >= workload->rank)
      return false;
   if (workload->step[1] && (workload->axis[1] >= workload->rank ||
                           workload->axis[0] == workload->axis[1]))
      return false;

   float sum = 0.0f;
   uint64_t extent[PAN_GRAPH_COST_MAX_RANK];
   memcpy(extent, workload->extent, sizeof(extent));
   if (!workload->step[0]) {
      sum = chunk_weight(workload, extent);
   } else {
      unsigned a = workload->axis[0], b = workload->axis[1];
      for (uint64_t i = 0; i < workload->extent[a]; i += workload->step[0]) {
         uint64_t remaining = workload->extent[a] - i;
         extent[a] = remaining < workload->step[0] ? remaining : workload->step[0];
         if (!workload->step[1]) {
            sum += chunk_weight(workload, extent);
         } else {
            for (uint64_t j = 0; j < workload->extent[b]; j += workload->step[1]) {
               remaining = workload->extent[b] - j;
               extent[b] = remaining < workload->step[1] ? remaining : workload->step[1];
               sum += chunk_weight(workload, extent);
            }
         }
      }
   }
   float factor = sum * (float)workload->repeats;
   float rate = (float)cycles / factor;
   float minimum = workload->minimum * (float)workload->latency;
   volatile float scaled = factor * (rate < minimum ? minimum : rate);
   if (!isfinite(factor) || factor <= 0.0f || !isfinite(scaled) ||
       scaled < 0.0f || scaled >= 0x1p63f)
      return false;
   *result = (uint64_t)scaled;
   return true;
}
