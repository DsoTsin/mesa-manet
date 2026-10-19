#include "pan_graph_matmul_search.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void
reject(const struct pan_graph_matmul_info *info, const uint32_t caps[7],
       const struct pan_graph_nrt_target *target, int32_t schedule,
       enum pan_graph_tsu_status status)
{
   struct pan_graph_matmul_candidate_list result, saved;
   memset(&result, 0xa5, sizeof(result));
   saved = result;
   assert(pan_graph_matmul_base_candidates(info, caps, target, schedule, 0,
                                           false, &result) == status);
   assert(!memcmp(&result, &saved, sizeof(result)));
}

int
main(void)
{
   struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   struct pan_graph_matmul_info info = {1, 32, 32, 32, 0, 0};
   uint32_t caps[7] = {32, 1, 1, 32, 32, 1, 1};
   struct pan_graph_matmul_candidate_list result = {0};
   assert(pan_graph_matmul_base_candidates(&info, caps, &target, -1, 0, false,
                                           &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.count == 1 && result.candidates[0].schedule_type == 2);
   assert(!memcmp(result.candidates[0].transfer, caps, sizeof(caps)));
   pan_graph_matmul_candidate_list_finish(&result);
   assert(!result.count && !result.candidates);

   info = (struct pan_graph_matmul_info){17, 1, 4096, 1, -128, 127};
   caps[0] = caps[3] = 1;
   caps[2] = 17;
   caps[4] = 4096;
   assert(pan_graph_matmul_base_candidates(&info, caps, &target, 2, 0, false,
                                           &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.count == 7);
   for (unsigned i = 0; i < result.count; i++) {
      assert(result.candidates[i].transfer[2] == 1);
      assert(result.candidates[i].transfer[4] == (1024u >> i));
      assert(result.candidates[i].schedule_type == 2);
   }
   pan_graph_matmul_candidate_list_finish(&result);

   info = (struct pan_graph_matmul_info){1, 1, 1, 4096, 0, 0};
   const uint32_t program_caps[7] = {4096, 1, 1, 1, 1, 1, 1};
   assert(pan_graph_matmul_program_base_candidates(&info, program_caps, &target,
                                                   -1, 0, false, 2, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.count == 6);
   assert(result.candidates[0].transfer[0] == 512);
   pan_graph_matmul_candidate_list_finish(&result);
   struct pan_graph_matmul_candidate_list invalid, saved;
   memset(&invalid, 0xa5, sizeof(invalid));
   saved = invalid;
   for (unsigned vectors = 0; vectors <= 9; vectors += 9) {
      assert(pan_graph_matmul_program_base_candidates(
                &info, program_caps, &target, -1, 0, false, vectors,
                &invalid) == PAN_GRAPH_TSU_INVALID);
      assert(!memcmp(&invalid, &saved, sizeof(saved)));
   }
   info = (struct pan_graph_matmul_info){1, 17, 17, 17, 0, 0};

   info = (struct pan_graph_matmul_info){1, 17, 17, 17, 0, 0};
   caps[0] = caps[3] = caps[4] = 17;
   caps[2] = 1;
   assert(pan_graph_matmul_base_candidates(&info, caps, &target, 3, 0, true,
                                           &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.count > 1);
   for (unsigned i = 0; i < result.count; i++) {
      assert(result.candidates[i].schedule_type == 3);
      for (unsigned j = 0; j < i; j++)
         assert(memcmp(result.candidates[i].transfer,
                       result.candidates[j].transfer, sizeof(caps)));
   }
   pan_graph_matmul_candidate_list_finish(&result);

   reject(NULL, caps, &target, -1, PAN_GRAPH_TSU_INVALID);
   reject(&info, NULL, &target, -1, PAN_GRAPH_TSU_INVALID);
   reject(&info, caps, NULL, -1, PAN_GRAPH_TSU_INVALID);
   reject(&info, caps, &target, -2, PAN_GRAPH_TSU_INVALID);
   reject(&info, caps, &target, 4, PAN_GRAPH_TSU_INVALID);
   for (unsigned i = 0; i < 7; i++) {
      uint32_t saved = caps[i];
      caps[i] = 0;
      reject(&info, caps, &target, -1, PAN_GRAPH_TSU_INVALID);
      caps[i] = UINT32_MAX;
      reject(&info, caps, &target, -1, PAN_GRAPH_TSU_INVALID);
      caps[i] = saved;
   }
   target.alignment = 64;
   reject(&info, caps, &target, -1, PAN_GRAPH_TSU_UNSUPPORTED);
   pan_graph_matmul_candidate_list_finish(NULL);
   puts("MatMul staged base search: early exit, partial K, batch contraction, "
        "schedule tags, deduplication and transactional failure passed");
   return 0;
}
