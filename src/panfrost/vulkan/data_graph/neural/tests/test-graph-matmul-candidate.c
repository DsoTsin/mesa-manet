#include "pan_graph_matmul_candidate.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

static void
reject(const struct pan_graph_matmul_info *info, const uint32_t transfer[7],
       const struct pan_graph_nrt_target *target, int32_t override,
       uint32_t divisor)
{
   struct pan_graph_matmul_candidate_set result, saved;
   memset(&result, 0xa5, sizeof(result));
   saved = result;
   assert(!pan_graph_matmul_candidate_filter(info, transfer, target, override,
                                             divisor, &result));
   assert(!memcmp(&result, &saved, sizeof(result)));
}

int
main(void)
{
   struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   struct pan_graph_matmul_info info = {2, 64, 1, 1, -128, 127};
   uint32_t transfer[7] = {1, 1, 2, 64, 1, 1, 1};
   struct pan_graph_matmul_candidate_set result;
   assert(pan_graph_matmul_candidate_filter(&info, transfer, &target, -1, 0,
                                            &result));
   assert(result.count == 2 && result.root_flag);
   assert(result.candidates[0].schedule_type == 3);
   assert(result.candidates[0].transfer[2] == 1);
   assert(result.candidates[1].schedule_type == 1);
   assert(result.candidates[1].transfer[2] == 2);

   info.batch = transfer[2] = 3;
   info.rows = transfer[3] = 129;
   assert(pan_graph_matmul_candidate_filter(&info, transfer, &target, -1, 0,
                                            &result));
   assert(result.count == 1 && !result.root_flag);
   assert(result.candidates[0].schedule_type == 3);
   assert(result.candidates[0].transfer[2] == 1);

   info = (struct pan_graph_matmul_info){1, 65, 1, 32, 0, 0};
   transfer[0] = 32;
   transfer[2] = 1;
   transfer[3] = 64;
   assert(pan_graph_matmul_candidate_filter(&info, transfer, &target, 0, 0,
                                            &result));
   assert(result.count == 1);
   transfer[3] = 65;
   assert(pan_graph_matmul_candidate_filter(&info, transfer, &target, 0, 0,
                                            &result));
   assert(!result.count && !result.root_flag);

   info = (struct pan_graph_matmul_info){1, 1, 1, 4096, 0, 0};
   const uint32_t program_trial[7] = {1024, 1, 1, 1, 1, 1, 1};
   assert(pan_graph_matmul_candidate_filter(&info, program_trial, &target, -1,
                                            0, &result));
   assert(result.count == 1);
   assert(pan_graph_matmul_program_candidate_filter(
      &info, program_trial, &target, -1, 0, 2, &result));
   assert(!result.count && !result.root_flag);
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_matmul_candidate_set saved = result;
   assert(!pan_graph_matmul_program_candidate_filter(
      &info, program_trial, &target, -1, 0, 0, &result));
   assert(!pan_graph_matmul_program_candidate_filter(
      &info, program_trial, &target, -1, 0, 9, &result));
   assert(!memcmp(&result, &saved, sizeof(result)));

   info = (struct pan_graph_matmul_info){1, 1, 4096, 1, 0, 0};
   transfer[0] = transfer[2] = transfer[3] = 1;
   transfer[4] = 1024;
   assert(pan_graph_matmul_candidate_filter(&info, transfer, &target, 0, 0,
                                            &result));
   assert(result.count == 1 && result.candidates[0].transfer[4] == 1024);
   transfer[4] = 4081;
   assert(pan_graph_matmul_candidate_filter(&info, transfer, &target, 0, 0,
                                            &result));
   assert(!result.count);
   transfer[4] = 1;

   reject(&info, transfer, &target, -2, 0);
   reject(&info, transfer, &target, 4, 0);
   reject(&info, transfer, &target, -1, UINT32_MAX);
   reject(NULL, transfer, &target, -1, 0);
   reject(&info, NULL, &target, -1, 0);
   reject(&info, transfer, NULL, -1, 0);
   assert(!pan_graph_matmul_candidate_filter(&info, transfer, &target, -1, 0,
                                             NULL));
   for (unsigned i = 0; i < 7; i++) {
      uint32_t saved = transfer[i];
      transfer[i] = 0;
      reject(&info, transfer, &target, -1, 0);
      transfer[i] = UINT32_MAX;
      reject(&info, transfer, &target, -1, 0);
      transfer[i] = saved;
   }
   target.alignment = 64;
   reject(&info, transfer, &target, -1, 0);
   target = pan_graph_nrt_target_v15();
   target.bank_parts = 2;
   reject(&info, transfer, &target, -1, 0);
   info.batch = info.rows = info.columns = INT32_MAX;
   transfer[0] = transfer[2] = transfer[3] = INT32_MAX;
   target = pan_graph_nrt_target_v15();
   reject(&info, transfer, &target, -1, 0);
   puts("MatMul candidate root: schedule order, batch contraction, accumulator "
        "boundary, partial K and transactional failure passed");
   return 0;
}
