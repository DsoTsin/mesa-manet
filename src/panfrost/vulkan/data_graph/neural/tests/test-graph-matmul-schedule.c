#include "pan_graph_matmul_schedule.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

static void
reject(const struct pan_graph_matmul_info *info, const uint32_t alignment[7],
       uint8_t mask)
{
   struct pan_graph_matmul_schedule_policy policy, original;
   memset(&policy, 0xa5, sizeof(policy));
   original = policy;
   assert(!pan_graph_matmul_schedule_policy_init(info, alignment, mask,
                                                 true, &policy));
   assert(!memcmp(&policy, &original, sizeof(policy)));
}

static void
parallel_score(void)
{
   uint32_t dimensions[7] = {32, 1, 3, 65, 17, 1, 1};
   uint32_t alignment[7] = {1, 1, 1, 4, 1, 1, 1};
   int32_t axes[2] = {0, 3};
   struct pan_graph_matmul_parallel_score score;
   assert(sizeof(score) == 32);
   assert(pan_graph_matmul_parallel_score(dimensions, alignment, axes, 6, 0,
                                         &score));
   assert(score.tile[0] == 16 && score.tile[1] == 24);
   assert(score.groups[0] == 2 && score.groups[1] == 3);
   assert(score.total_groups == 6);
   assert(score.coverage[0] == 2.0f && score.coverage[1] == 65.0f / 24.0f);
   assert(score.score == 2.0f * (65.0f / 24.0f));
   assert(pan_graph_matmul_parallel_score(dimensions, alignment, axes, 6, 16,
                                         &score));
   assert(16 % score.tile[0] == 0);
   axes[1] = -1;
   assert(pan_graph_matmul_parallel_score(dimensions, alignment, axes, 6, 0,
                                         &score));
   assert(score.tile[0] == 6 && score.tile[1] == 1);
   assert(score.groups[0] == 6 && score.groups[1] == 1);
   assert(score.coverage[0] == 32.0f / 6.0f && score.coverage[1] == 1.0f);
   const struct pan_graph_matmul_parallel_score initial = {
      .tile = {1, 1}, .groups = {1, 1}, .total_groups = 1,
   };
   assert(pan_graph_matmul_parallel_score(dimensions, alignment, axes, 0, 0,
                                         &score));
   assert(!memcmp(&score, &initial, sizeof(score)));
   axes[0] = -1;
   assert(pan_graph_matmul_parallel_score(dimensions, alignment, axes, 6, 0,
                                         &score));
   assert(!memcmp(&score, &initial, sizeof(score)));
   for (unsigned axis = 0; axis < 7; axis++) {
      uint32_t saved = alignment[axis];
      alignment[axis] = 0;
      assert(!pan_graph_matmul_parallel_score(dimensions, alignment, axes, 6, 0,
                                             &score));
      assert(!memcmp(&score, &initial, sizeof(score)));
      alignment[axis] = saved;
      saved = dimensions[axis];
      dimensions[axis] = UINT32_MAX;
      assert(!pan_graph_matmul_parallel_score(dimensions, alignment, axes, 6, 0,
                                             &score));
      assert(!memcmp(&score, &initial, sizeof(score)));
      dimensions[axis] = saved;
   }
   axes[0] = axes[1] = 0;
   assert(!pan_graph_matmul_parallel_score(dimensions, alignment, axes, 6, 0,
                                          &score));
   axes[1] = -1;
   assert(!pan_graph_matmul_parallel_score(dimensions, alignment, axes,
                                          UINT32_MAX, 0, &score));
   assert(!pan_graph_matmul_parallel_score(dimensions, alignment, axes, 6,
                                          UINT32_MAX, &score));
   axes[0] = 4;
   assert(!pan_graph_matmul_parallel_score(dimensions, alignment, axes, 6, 0,
                                          &score));
   assert(!pan_graph_matmul_parallel_score(NULL, alignment, axes, 6, 0, &score));
   assert(!pan_graph_matmul_parallel_score(dimensions, NULL, axes, 6, 0, &score));
   assert(!pan_graph_matmul_parallel_score(dimensions, alignment, NULL, 6, 0,
                                          &score));
   assert(!pan_graph_matmul_parallel_score(dimensions, alignment, axes, 6, 0,
                                          NULL));
}

static void
assign_loops(void)
{
   struct pan_graph_matmul_info info = {3, 17, 4096, 32, -128, 127};
   uint32_t transfer[7] = {32, 1, 1, 6, 256, 1, 1};
   uint32_t granule[7] = {1, 1, 1, 1, 1, 1, 1};
   struct pan_graph_matmul_parallel_options options = {
      .core_budget = 6, .output_axis_mask = 15, .multiple_axes = true,
   };
   uint32_t loops[4];
   const uint32_t tie[4] = {2, 3, 1, 12};
   assert(pan_graph_matmul_assign_loops(&info, transfer, granule, &options,
                                       loops));
   assert(!memcmp(loops, tie, sizeof(loops)));
   transfer[0] = transfer[3] = 16;
   const uint32_t pair[4] = {0, 2, 16, 1};
   assert(pan_graph_matmul_assign_loops(&info, transfer, granule, &options,
                                       loops));
   assert(!memcmp(loops, pair, sizeof(loops)));
   info = (struct pan_graph_matmul_info){1, 32, 32, 32, 0, 0};
   transfer[0] = transfer[4] = 32;
   transfer[3] = 8;
   const uint32_t row[4] = {3, 0, 8, 0};
   assert(pan_graph_matmul_assign_loops(&info, transfer, granule, &options,
                                       loops));
   assert(!memcmp(loops, row, sizeof(loops)));
   options.core_budget = 1;
   const uint32_t empty[4] = {0};
   assert(pan_graph_matmul_assign_loops(&info, transfer, granule, &options,
                                       loops));
   assert(!memcmp(loops, empty, sizeof(loops)));
   transfer[3] = 0;
   assert(!pan_graph_matmul_assign_loops(&info, transfer, granule, &options,
                                        loops));
   assert(!memcmp(loops, empty, sizeof(loops)));
   transfer[3] = 8;
   granule[0] = 0;
   assert(!pan_graph_matmul_assign_loops(&info, transfer, granule, &options,
                                        loops));
   assert(!memcmp(loops, empty, sizeof(loops)));
   granule[0] = 1;
   options.output_axis_mask = 16;
   assert(!pan_graph_matmul_assign_loops(&info, transfer, granule, &options,
                                        loops));
   assert(!memcmp(loops, empty, sizeof(loops)));
   assert(!pan_graph_matmul_assign_loops(NULL, transfer, granule, &options,
                                        loops));
   assert(!pan_graph_matmul_assign_loops(&info, NULL, granule, &options, loops));
   assert(!pan_graph_matmul_assign_loops(&info, transfer, NULL, &options, loops));
   assert(!pan_graph_matmul_assign_loops(&info, transfer, granule, NULL, loops));
   assert(!pan_graph_matmul_assign_loops(&info, transfer, granule, &options,
                                        NULL));
}

static void
parallel_caps(void)
{
   struct pan_graph_matmul_info info = {1, 32, 32, 32, 0, 0};
   const uint32_t granule[7] = {1, 1, 1, 1, 1, 1, 1};
   struct pan_graph_matmul_parallel_options options = {
      .core_budget = 6, .output_axis_mask = 15, .multiple_axes = true,
   };
   struct pan_graph_matmul_parallel_caps caps;
   const uint32_t expected[6][7] = {
      {32, 1, 1, 16, 32, 1, 1}, {32, 1, 1, 8, 32, 1, 1},
      {32, 1, 1, 4, 32, 1, 1}, {32, 1, 1, 2, 32, 1, 1},
      {11, 1, 1, 32, 32, 1, 1}, {11, 1, 1, 16, 32, 1, 1},
   };
   assert(pan_graph_matmul_parallel_caps(&info, granule, &options, &caps));
   assert(caps.count == 6);
   assert(!memcmp(caps.extents, expected, sizeof(expected)));
   info.rows = info.inner = info.columns = 17;
   assert(pan_graph_matmul_parallel_caps(&info, granule, &options, &caps));
   const uint32_t single[7] = {3, 1, 1, 17, 17, 1, 1};
   assert(caps.count == 1);
   assert(!memcmp(caps.extents, single, sizeof(single)));
   options.core_budget = 1;
   assert(pan_graph_matmul_parallel_caps(&info, granule, &options, &caps));
   assert(!caps.count);
   struct pan_graph_matmul_parallel_caps original = caps;
   options.core_budget = UINT32_MAX;
   assert(!pan_graph_matmul_parallel_caps(&info, granule, &options, &caps));
   assert(!memcmp(&caps, &original, sizeof(caps)));
   assert(!pan_graph_matmul_parallel_caps(NULL, granule, &options, &caps));
   assert(!pan_graph_matmul_parallel_caps(&info, NULL, &options, &caps));
   assert(!pan_graph_matmul_parallel_caps(&info, granule, NULL, &caps));
   assert(!pan_graph_matmul_parallel_caps(&info, granule, &options, NULL));
}

int
main(void)
{
   parallel_score();
   assign_loops();
   parallel_caps();
   struct pan_graph_matmul_info info = {3, 32, 17, 17, -128, 127};
   uint32_t alignment[7] = {1, 1, 1, 1, 1, 1, 1};
   struct pan_graph_matmul_schedule_policy policy;
   assert(pan_graph_matmul_schedule_policy_init(&info, alignment, 15,
                                                 true, &policy));
   assert(policy.allowed_axes == 13);
   assert(policy.axes[0] == 3 && policy.axes[1] == 0);
   assert(policy.axes[2] == 2 && policy.axes[3] == -1);
   for (unsigned i = 0; i < 7; i++)
      assert(policy.granule[i] == 1);
   info.batch = info.rows = info.columns = 17;
   assert(pan_graph_matmul_schedule_policy_init(&info, alignment, 15,
                                                 true, &policy));
   assert(policy.axes[0] == 0 && policy.axes[1] == 2);
   assert(policy.axes[2] == 3 && policy.axes[3] == -1);
   assert(pan_graph_matmul_schedule_policy_init(&info, alignment, 15,
                                                 false, &policy));
   assert(policy.axes[0] == 0 && policy.axes[1] == -1);
   assert(policy.axes[2] == -1 && policy.axes[3] == -1);
   for (unsigned bit = 0; bit < 4; bit++) {
      static const int32_t expected[] = {-1, 2, 3, 0};
      assert(pan_graph_matmul_schedule_policy_init(&info, alignment, 1u << bit,
                                                    true, &policy));
      assert(policy.axes[0] == expected[bit] && policy.axes[1] == -1);
   }
   info.batch = info.rows = info.columns = 1;
   assert(pan_graph_matmul_schedule_policy_init(&info, alignment, 15,
                                                 true, &policy));
   assert(policy.allowed_axes == 13);
   assert(policy.axes[0] == -1 && policy.axes[1] == -1);
   info = (struct pan_graph_matmul_info){3, 32, 17, 17, 0, 0};
   alignment[0] = 8;
   alignment[3] = 4;
   assert(pan_graph_matmul_schedule_policy_init(&info, alignment, 15,
                                                 true, &policy));
   assert(policy.axes[0] == 3 && policy.axes[1] == 0);
   for (unsigned i = 0; i < 7; i++) {
      uint32_t saved = alignment[i];
      alignment[i] = 0;
      reject(&info, alignment, 15);
      alignment[i] = UINT32_MAX;
      reject(&info, alignment, 15);
      alignment[i] = saved;
   }
   reject(&info, alignment, 16);
   info.batch = 0;
   reject(&info, alignment, 15);
   info.batch = 1;
   info.a_zero_point = 128;
   reject(&info, alignment, 15);
   info.a_zero_point = 0;
   reject(NULL, alignment, 15);
   reject(&info, NULL, 15);
   assert(!pan_graph_matmul_schedule_policy_init(&info, alignment, 15,
                                                  true, NULL));
   puts("MatMul schedule policy and parallel score: affine axes, stable ties, "
        "float32 threshold, divisor and transactional rejection passed");
   return 0;
}
