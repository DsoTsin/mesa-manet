#include "pan_graph_matmul_schedule.h"

#include <limits.h>
#include <string.h>

static bool
schedule_policy_init(
   const uint32_t dimensions[7], const uint32_t alignment[7],
   uint8_t allowed_axes, bool multiple_axes,
   struct pan_graph_matmul_schedule_policy *policy)
{
   if (!dimensions || !alignment || !policy || allowed_axes > 15)
      return false;
   for (unsigned i = 0; i < 7; i++) {
      if (!dimensions[i] || dimensions[i] > INT32_MAX ||
          !alignment[i] || alignment[i] > INT32_MAX)
         return false;
   }

   struct pan_graph_matmul_schedule_policy result = {
      .axes = {-1, -1, -1, -1}, .allowed_axes = allowed_axes,
   };
   for (unsigned i = 0; i < 7; i++)
      result.granule[i] = 1;

   uint8_t available = result.allowed_axes;
   for (unsigned pass = 0; pass < (multiple_axes ? 4u : 1u); pass++) {
      uint64_t largest = 0;
      for (unsigned axis = 0; axis < 4; axis++) {
         uint64_t extent = dimensions[axis];
         if (!(available & (1u << axis)) || extent <= alignment[axis])
            continue;
         uint64_t groups = (extent + alignment[axis] - 1) / alignment[axis];
         if (groups > largest) {
            largest = groups;
            result.axes[pass] = axis;
         }
      }
      if (result.axes[pass] >= 0)
         available &= ~(1u << result.axes[pass]);
   }
   *policy = result;
   return true;
}

static uint8_t
output_axes(const struct pan_graph_matmul_plan *plan, uint8_t mask)
{
   uint8_t axes = 0;
   const struct pan_graph_matmul_map *output =
      &plan->views[PAN_GRAPH_MATMUL_CONV_OUTPUT];
   for (unsigned r = 0; r < output->output_rank; r++) {
      int axis = output->forward[r];
      if (axis >= 0 && axis < 4 &&
          (mask & (1u << (4 - output->output_rank + r))))
         axes |= 1u << axis;
   }
   return axes;
}

bool
pan_graph_matmul_schedule_policy_init(
   const struct pan_graph_matmul_info *info, const uint32_t alignment[7],
   uint8_t output_axis_mask, bool multiple_axes,
   struct pan_graph_matmul_schedule_policy *policy)
{
   struct pan_graph_matmul_plan plan;
   if (output_axis_mask > 15 || !pan_graph_matmul_i8_init(&plan, info))
      return false;
   uint32_t dimensions[7];
   for (unsigned i = 0; i < 7; i++)
      dimensions[i] = plan.conv_high[i] + 1;
   return schedule_policy_init(dimensions, alignment,
                               output_axes(&plan, output_axis_mask),
                               multiple_axes, policy);
}

static uint32_t
parallel_tile(uint32_t extent, uint32_t alignment, uint32_t groups)
{
   uint64_t tile = ((uint64_t)extent + groups - 1) / groups;
   tile = (tile + alignment - 1) / alignment * alignment;
   return tile < extent ? tile : extent;
}

bool
pan_graph_matmul_parallel_score(const uint32_t dimensions[7],
                                const uint32_t alignment[7],
                                const int32_t axes[2], uint32_t core_budget,
                                uint32_t column_divisor,
                                struct pan_graph_matmul_parallel_score *score)
{
   if (!dimensions || !alignment || !axes || !score ||
       core_budget > INT32_MAX || column_divisor > INT32_MAX ||
       axes[0] < -1 || axes[0] >= 4 || axes[1] < -1 || axes[1] >= 4 ||
       (axes[1] >= 0 && (axes[0] < 0 || axes[0] == axes[1])))
      return false;
   for (unsigned axis = 0; axis < 7; axis++) {
      if (!dimensions[axis] || dimensions[axis] > INT32_MAX ||
          !alignment[axis] || alignment[axis] > INT32_MAX)
         return false;
   }

   struct pan_graph_matmul_parallel_score result = {
      .tile = {1, 1}, .groups = {1, 1}, .total_groups = 1,
   };
   if (axes[0] >= 0) {
      uint32_t a = axes[0];
      int32_t b = axes[1];
      for (uint32_t i = core_budget; i > 0; i--) {
         for (uint32_t j = 1; j <= core_budget / i; j++) {
            uint32_t x = parallel_tile(dimensions[a], alignment[a], i);
            uint32_t y = b >= 0 ? parallel_tile(dimensions[b], alignment[b], j)
                                : 1;
            if (column_divisor && ((a == 0 && column_divisor % x) ||
                                   (b == 0 && column_divisor % y)))
               continue;

            float coverage_a = (float)dimensions[a] / (float)x;
            float coverage_b = b >= 0 ? (float)dimensions[b] / (float)y : 1.0f;
            float value = coverage_a * coverage_b;
            if (value <= result.score + 0.01f)
               continue;

            uint32_t groups_a = ((uint64_t)dimensions[a] + x - 1) / x;
            uint32_t groups_b = b >= 0 ? ((uint64_t)dimensions[b] + y - 1) / y
                                       : 1;
            uint64_t total = (uint64_t)groups_a * groups_b;
            if (total > UINT32_MAX)
               return false;
            result = (struct pan_graph_matmul_parallel_score){
               .tile = {x, y}, .groups = {groups_a, groups_b},
               .total_groups = total, .coverage = {coverage_a, coverage_b},
               .score = value,
            };
         }
      }
   }
   *score = result;
   return true;
}

static uint32_t
greatest_common_divisor(uint32_t a, uint32_t b)
{
   while (b) {
      uint32_t remainder = a % b;
      a = b;
      b = remainder;
   }
   return a;
}

bool
pan_graph_assign_loops(
   const uint32_t dimensions[7], const uint32_t transfer[7],
   const uint32_t granule[7], uint8_t allowed_axes,
   const struct pan_graph_matmul_parallel_options *options, uint32_t loops[4])
{
   if (!dimensions || !transfer || !granule || !options || !loops ||
       options->output_axis_mask > 15)
      return false;

   uint32_t alignment[7];
   for (unsigned i = 0; i < 7; i++) {
      if (!dimensions[i] || dimensions[i] > INT32_MAX ||
          !transfer[i] || transfer[i] > dimensions[i] ||
          !granule[i] || granule[i] > INT32_MAX)
         return false;
      uint64_t common = (uint64_t)transfer[i] * granule[i] /
                        greatest_common_divisor(transfer[i], granule[i]);
      alignment[i] = common < dimensions[i] ? common : dimensions[i];
   }
   struct pan_graph_matmul_schedule_policy policy;
   if (!schedule_policy_init(dimensions, alignment, allowed_axes,
                              options->multiple_axes, &policy))
      return false;

   int32_t axes[2] = {policy.axes[0], policy.axes[1]};
   struct pan_graph_matmul_parallel_score score;
   if (!pan_graph_matmul_parallel_score(dimensions, alignment, axes,
                                        options->core_budget,
                                        options->column_divisor, &score))
      return false;
   if (!options->preferred_pair_only && policy.axes[2] >= 0) {
      for (unsigned i = 2; i < 4 && policy.axes[i] >= 0; i++) {
         for (unsigned j = 0; j < i; j++) {
            const int32_t pair[2] = {policy.axes[j], policy.axes[i]};
            struct pan_graph_matmul_parallel_score trial;
            if (!pan_graph_matmul_parallel_score(
                   dimensions, alignment, pair, options->core_budget,
                   options->column_divisor, &trial))
               return false;
            if (trial.score - 0.01f > score.score) {
               axes[0] = pair[0];
               axes[1] = pair[1];
               score = trial;
            }
         }
      }
   }

   if (score.groups[1] == 1)
      axes[1] = -1;
   if (axes[0] >= 0 && score.tile[0] >= dimensions[axes[0]])
      axes[0] = -1;
   if (axes[1] >= 0 && (axes[0] < 0 || axes[1] < axes[0])) {
      int32_t axis = axes[0];
      uint32_t tile = score.tile[0];
      axes[0] = axes[1];
      axes[1] = axis;
      score.tile[0] = score.tile[1];
      score.tile[1] = tile;
   }

   uint32_t result[4] = {0};
   if (score.total_groups >= 2) {
      if (axes[0] < 0)
         return false;
      result[0] = axes[0];
      result[2] = score.tile[0];
      if (axes[1] >= 0) {
         result[1] = axes[1];
         result[3] = score.tile[1];
      }
   }
   memcpy(loops, result, sizeof(result));
   return true;
}

bool
pan_graph_matmul_assign_loops(
   const struct pan_graph_matmul_info *info, const uint32_t transfer[7],
   const uint32_t granule[7],
   const struct pan_graph_matmul_parallel_options *options, uint32_t loops[4])
{
   struct pan_graph_matmul_plan plan;
   if (!options || !pan_graph_matmul_i8_init(&plan, info))
      return false;
   uint32_t dimensions[7];
   for (unsigned i = 0; i < 7; i++)
      dimensions[i] = plan.conv_high[i] + 1;
   return pan_graph_assign_loops(dimensions, transfer, granule,
                                 output_axes(&plan, options->output_axis_mask),
                                 options, loops);
}

static bool
insert_parallel_cap(struct pan_graph_matmul_parallel_caps *caps,
                    const uint32_t extent[7])
{
   unsigned at = 0;
   for (; at < caps->count; at++) {
      int order = 0;
      for (unsigned axis = 0; axis < 7 && !order; axis++) {
         if (extent[axis] != caps->extents[at][axis])
            order = extent[axis] > caps->extents[at][axis] ? 1 : -1;
      }
      if (!order)
         return true;
      if (order > 0)
         break;
   }
   if (caps->count == PAN_GRAPH_MATMUL_PARALLEL_CAPS_MAX)
      return false;
   memmove(&caps->extents[at + 1], &caps->extents[at],
           (caps->count - at) * sizeof(caps->extents[0]));
   memcpy(caps->extents[at], extent, sizeof(caps->extents[0]));
   caps->count++;
   return true;
}

bool
pan_graph_matmul_parallel_caps(
   const struct pan_graph_matmul_info *info, const uint32_t granule[7],
   const struct pan_graph_matmul_parallel_options *options,
   struct pan_graph_matmul_parallel_caps *caps)
{
   struct pan_graph_matmul_plan plan;
   struct pan_graph_matmul_schedule_policy policy;
   if (!options || !caps || !pan_graph_matmul_i8_init(&plan, info) ||
       !pan_graph_matmul_schedule_policy_init(
          info, granule, options->output_axis_mask, options->multiple_axes,
          &policy))
      return false;
   uint32_t dimensions[7];
   for (unsigned i = 0; i < 7; i++)
      dimensions[i] = plan.conv_high[i] + 1;
   int32_t a = policy.axes[0], b = policy.axes[1];
   struct pan_graph_matmul_parallel_score score;
   if (!pan_graph_matmul_parallel_score(dimensions, granule, policy.axes,
                                        options->core_budget,
                                        options->column_divisor, &score))
      return false;
   if (score.groups[1] == 1)
      b = -1;
   if (a >= 0 && score.tile[0] >= dimensions[a])
      a = -1;
   if (b >= 0 && (a < 0 || b < a)) {
      int32_t axis = a;
      uint32_t tile = score.tile[0];
      a = b;
      b = axis;
      score.tile[0] = score.tile[1];
      score.tile[1] = tile;
   }

   struct pan_graph_matmul_parallel_caps result = {0};
   if (a >= 0 && options->core_budget >= 2) {
      uint32_t minimum_a = granule[a] > 2 ? granule[a] : 2;
      uint32_t divisor_a = a == 0 ? options->column_divisor : 0;
      uint32_t divisor_b = b == 0 ? options->column_divisor : 0;
      for (int64_t x = score.tile[0]; x >= minimum_a; x -= granule[a]) {
         uint32_t extent[7];
         memcpy(extent, dimensions, sizeof(extent));
         if (score.tile[0] % x == 0 && divisor_a % x == 0) {
            extent[a] = x;
            if (!insert_parallel_cap(&result, extent))
               return false;
         }
         if (b >= 0) {
            uint32_t minimum_b = granule[b] > 2 ? granule[b] : 2;
            for (int64_t y = score.tile[1]; y >= minimum_b; y -= granule[b]) {
               if (score.tile[1] % y == 0 && divisor_b % y == 0) {
                  extent[b] = y;
                  if (!insert_parallel_cap(&result, extent))
                     return false;
                  memcpy(extent, dimensions, sizeof(extent));
                  extent[b] = y;
                  if (!insert_parallel_cap(&result, extent))
                     return false;
               }
               if (result.count > 20)
                  break;
            }
         }
         if (result.count > 20)
            break;
      }
   }
   *caps = result;
   return true;
}
