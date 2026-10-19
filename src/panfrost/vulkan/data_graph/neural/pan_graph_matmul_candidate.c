#include "pan_graph_matmul_candidate.h"

#include <limits.h>
#include <string.h>

static const uint32_t blocks[4][3] = {
   {32, 2, 2},
   {8, 4, 4},
   {32, 4, 1},
   {8, 16, 1},
};

struct schedule_score {
   uint32_t tiles;
   uint32_t type;
   float waste;
   bool small;
};

static uint64_t
ceil_div(uint64_t value, uint32_t divisor)
{
   return (value + divisor - 1) / divisor;
}

static bool
score_before(const struct schedule_score *a, const struct schedule_score *b)
{
   if (a->tiles != b->tiles)
      return a->tiles < b->tiles;
   if (a->small != b->small)
      return !a->small;
   return a->waste < b->waste;
}

static bool
choose_schedules(const uint32_t proposed[7], uint32_t types[2], uint32_t *count)
{
   const uint32_t dimensions[3] = {proposed[0], proposed[3], proposed[2]};
   uint64_t volume = 1;
   for (unsigned i = 0; i < 3; i++) {
      if (volume > INT32_MAX / dimensions[i])
         return false;
      volume *= dimensions[i];
   }

   static const uint32_t order[] = {0, 2, 1, 3};
   struct schedule_score scores[4];
   for (unsigned i = 0; i < 4; i++) {
      uint32_t type = order[i];
      uint64_t tiles = 1;
      for (unsigned j = 0; j < 3; j++)
         tiles *= ceil_div(dimensions[j], blocks[type][j]);
      if (tiles > INT32_MAX / 128)
         return false;
      struct schedule_score score = {
         .tiles = tiles,
         .type = type,
         .waste = 1.0f - (float)volume / (float)(tiles * 128),
         .small = type == 1 || type == 3,
      };
      unsigned at = i;
      while (at && score_before(&score, &scores[at - 1])) {
         scores[at] = scores[at - 1];
         at--;
      }
      scores[at] = score;
   }

   types[0] = scores[0].type;
   *count = 1;
   if (types[0] == 3 && proposed[2] > 1) {
      types[1] = scores[1].type;
      *count = 2;
   }
   return true;
}

static bool
resources_fit(const struct pan_graph_nrt_target *target,
              const uint32_t transfer[7], uint32_t schedule, uint32_t vectors)
{
   const uint32_t n = transfer[0], b = transfer[2], m = transfer[3];
   const uint32_t k = transfer[4];
   struct pan_graph_nrt_allocator allocator;
   if (!pan_graph_nrt_allocator_init(&allocator, target))
      return false;

   const struct pan_graph_nrt_request requests[] = {
      {
         .type = PAN_GRAPH_NRT_NORMAL,
         .dimensions = {n, k, b, 1},
         .element_bits = 8,
         .buffers = 2,
         .front = true,
      },
      {
         .type = PAN_GRAPH_NRT_NORMAL,
         .dimensions = {k, m, b, 1},
         .element_bits = 8,
         .buffers = 2,
         .front = true,
      },
      {
         .type = PAN_GRAPH_NRT_NORMAL,
         .dimensions = {n, m, b, 1},
         .element_bits = 32,
         .buffers = 2,
      },
      {
         .type = PAN_GRAPH_NRT_WEIGHT,
         .dimensions = {k, 1, 1, n},
         .element_bits = 8,
         .buffers = 2,
         .weight_format = 1,
         .initial_format = 1,
         .narrow_weight = true,
      },
   };
   struct pan_graph_nrt_layout layout;
   for (unsigned i = 0; i < 4; i++) {
      if (pan_graph_nrt_allocate(&allocator, &requests[i], &layout) !=
          PAN_GRAPH_NRT_SUCCESS)
         return false;
   }
   for (uint32_t i = 1; i < vectors; i++) {
      const struct pan_graph_nrt_request request = {
         .type = PAN_GRAPH_NRT_NORMAL,
         .dimensions = {n, m, b, 1},
         .element_bits = 32,
         .buffers = 2,
      };
      if (pan_graph_nrt_allocate(&allocator, &request, &layout) !=
          PAN_GRAPH_NRT_SUCCESS)
         return false;
   }

   const struct pan_graph_nrt_accumulator_request accumulator = {
      .dimensions = {n, m, b, 1},
      .schedule_type = schedule,
      .producer_mode = 3,
      .input_bits = {8, 8},
   };
   return pan_graph_nrt_accumulator_layout(&accumulator, &layout) &&
          (uint64_t)layout.dimensions[0] * 1024 <= 32768;
}

bool
pan_graph_matmul_program_candidate_filter(
   const struct pan_graph_matmul_info *info, const uint32_t proposed[7],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   uint32_t column_divisor, uint32_t vector_count,
   struct pan_graph_matmul_candidate_set *candidates)
{
   struct pan_graph_matmul_plan plan;
   struct pan_graph_nrt_allocator allocator;
   if (!candidates || !proposed || !target || !vector_count ||
       vector_count > 8 || !pan_graph_matmul_i8_init(&plan, info) ||
       !pan_graph_nrt_allocator_init(&allocator, target) ||
       target->alignment != 32 || target->banks != 8 ||
       target->bank_parts != 1 || schedule_override < -1 ||
       schedule_override > 3 || column_divisor > INT32_MAX)
      return false;
   for (unsigned i = 0; i < 7; i++) {
      if (!proposed[i] || proposed[i] > (uint64_t)plan.conv_high[i] + 1)
         return false;
   }

   uint32_t types[2], count;
   if (schedule_override >= 0) {
      types[0] = schedule_override;
      count = 1;
   } else if (!choose_schedules(proposed, types, &count)) {
      return false;
   }

   struct pan_graph_matmul_candidate_set result = {0};
   for (unsigned i = 0; i < count; i++) {
      struct pan_graph_matmul_candidate candidate = {.schedule_type = types[i]};
      memcpy(candidate.transfer, proposed, sizeof(candidate.transfer));
      if (types[i] == 3)
         candidate.transfer[2] = 1;
      if (column_divisor % candidate.transfer[0] ||
          candidate.transfer[4] > 4080 ||
          !resources_fit(target, candidate.transfer, types[i], vector_count))
         continue;
      result.candidates[result.count++] = candidate;
      if (types[i] != 3 || count == 1)
         result.root_flag = true;
   }
   *candidates = result;
   return true;
}

bool
pan_graph_matmul_candidate_filter(
   const struct pan_graph_matmul_info *info, const uint32_t proposed[7],
   const struct pan_graph_nrt_target *target, int32_t schedule_override,
   uint32_t column_divisor, struct pan_graph_matmul_candidate_set *candidates)
{
   return pan_graph_matmul_program_candidate_filter(
      info, proposed, target, schedule_override, column_divisor, 1, candidates);
}
