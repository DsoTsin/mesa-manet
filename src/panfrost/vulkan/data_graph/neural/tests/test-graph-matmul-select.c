#include "pan_graph_matmul_select.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

int
main(void)
{
   struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   struct pan_graph_matmul_cost_target cost;
   assert(pan_graph_matmul_cost_target_v15_init(8, 64, 262144, &cost));
   struct pan_graph_matmul_info info = {1, 32, 32, 32, 0, 0};
   struct pan_graph_matmul_selection selected;
   assert(pan_graph_matmul_select_v15(&info, &target, &cost, 6, &selected) ==
          PAN_GRAPH_TSU_SUCCESS);
   const uint32_t row_tile[7] = {32, 1, 1, 8, 32, 1, 1};
   const uint32_t row_loops[4] = {3, 0, 8, 0};
   assert(!memcmp(selected.candidate.transfer, row_tile, sizeof(row_tile)));
   assert(!memcmp(selected.loops, row_loops, sizeof(row_loops)));
   assert(selected.candidate.schedule_type == 2);
   assert(selected.cost.cycles == 340 && selected.candidate_count == 7);
   info = (struct pan_graph_matmul_info){1, 1, 8193, 1, 0, 0};
   assert(pan_graph_matmul_select_v15(&info, &target, &cost, 6, &selected) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(selected.candidate.transfer[4] == 1024);
   assert(selected.cost.cycles == 14340 && selected.candidate_count == 7);
   for (unsigned i = 0; i < 4; i++)
      assert(!selected.loops[i]);

   info = (struct pan_graph_matmul_info){1, 32, 32, 32, 0, 0};
   const uint32_t full[7] = {32, 1, 1, 32, 32, 1, 1};
   uint32_t granule[7] = {4, 1, 1, 4, 1, 1, 1};
   const struct pan_graph_matmul_parallel_options options = {
      .core_budget = 6,
      .output_axis_mask = 15,
      .multiple_axes = true,
   };
   struct pan_graph_matmul_candidate_list seed = {0}, parallel = {0};
   assert(pan_graph_matmul_program_base_candidates(&info, full, &target, -1, 0,
                                                   false, 1, &seed) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(seed.count == 1);
   struct pan_graph_matmul_candidate saved_seed = seed.candidates[0];
   assert(pan_graph_matmul_program_parallel_candidates(
             &info, full, &target, -1, false, &options, granule, 1, &seed,
             &parallel) == PAN_GRAPH_TSU_SUCCESS);
   assert(parallel.count == 8);
   assert(!memcmp(&parallel.candidates[0], &saved_seed, sizeof(saved_seed)));
   assert(parallel.candidates[4].transfer[0] == 12);
   assert(parallel.candidates[4].schedule_type == 3);
   assert(parallel.candidates[7].transfer[0] == 4);
   assert(parallel.candidates[7].transfer[3] == 16);
   assert(seed.count == 1 &&
          !memcmp(seed.candidates, &saved_seed, sizeof(saved_seed)));
   pan_graph_matmul_candidate_list_finish(&parallel);
   struct pan_graph_matmul_candidate repeated[2] = {saved_seed, saved_seed};
   struct pan_graph_matmul_candidate_list duplicate_seed = {
      .count = 2,
      .candidates = repeated,
   };
   assert(pan_graph_matmul_program_parallel_candidates(
             &info, full, &target, -1, false, &options, granule, 1,
             &duplicate_seed, &parallel) == PAN_GRAPH_TSU_SUCCESS);
   assert(parallel.count == 9);
   assert(!memcmp(parallel.candidates, repeated, sizeof(repeated)));
   assert(duplicate_seed.count == 2 &&
          !memcmp(repeated, &saved_seed, sizeof(saved_seed)) &&
          !memcmp(repeated + 1, &saved_seed, sizeof(saved_seed)));
   pan_graph_matmul_candidate_list_finish(&parallel);
   memset(&parallel, 0xa5, sizeof(parallel));
   struct pan_graph_matmul_candidate_list saved_parallel = parallel;
   granule[0] = 0;
   assert(pan_graph_matmul_program_parallel_candidates(
             &info, full, &target, -1, false, &options, granule, 1, &seed,
             &parallel) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&parallel, &saved_parallel, sizeof(parallel)));
   granule[0] = 4;
   seed.candidates[0].schedule_type = 4;
   assert(pan_graph_matmul_program_parallel_candidates(
             &info, full, &target, -1, false, &options, granule, 1, &seed,
             &parallel) == PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&parallel, &saved_parallel, sizeof(parallel)));
   pan_graph_matmul_candidate_list_finish(&seed);
   info = (struct pan_graph_matmul_info){3, 17, 4096, 32, -128, 127};
   assert(pan_graph_matmul_select_v15(&info, &target, &cost, 6, &selected) ==
          PAN_GRAPH_TSU_SUCCESS);
   const uint32_t partial[7] = {16, 1, 1, 16, 512, 1, 1};
   const uint32_t partial_loops[4] = {0, 2, 16, 1};
   assert(!memcmp(selected.candidate.transfer, partial, sizeof(partial)));
   assert(!memcmp(selected.loops, partial_loops, sizeof(partial_loops)));
   assert(selected.cost.cycles == 26115 && selected.candidate_count == 112);

   const struct pan_graph_matmul_info large[] = {
      {8193, 1, 1, 1, 0, 0},
      {1, 8193, 1, 1, 0, 0},
      {1, 1, 1, 8193, 0, 0},
   };
   const uint32_t large_transfers[][7] = {
      {1, 1, 1, 1, 1, 1, 1},
      {1, 1, 1, 128, 1, 1, 1},
      {683, 1, 1, 1, 1, 1, 1},
   };
   const uint32_t large_loops[][4] = {
      {2, 0, 1366, 0},
      {3, 0, 1408, 0},
      {0, 0, 1366, 0},
   };
   const uint64_t large_cycles[] = {696405, 5635, 5475};
   for (unsigned i = 0; i < 3; i++) {
      assert(pan_graph_matmul_select_v15(&large[i], &target, &cost, 6,
                                         &selected) == PAN_GRAPH_TSU_SUCCESS);
      assert(!memcmp(selected.candidate.transfer, large_transfers[i],
                     sizeof(large_transfers[i])));
      assert(!memcmp(selected.loops, large_loops[i], sizeof(large_loops[i])));
      assert(selected.cost.cycles == large_cycles[i]);
   }

   memset(&selected, 0xa5, sizeof(selected));
   struct pan_graph_matmul_selection original = selected;
   assert(pan_graph_matmul_select_v15(&info, &target, &cost, 0, &selected) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&selected, &original, sizeof(selected)));
   info.batch = 0;
   assert(pan_graph_matmul_select_v15(&info, &target, &cost, 6, &selected) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&selected, &original, sizeof(selected)));
   assert(pan_graph_matmul_select_v15(NULL, &target, &cost, 6, &selected) ==
          PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_matmul_select_v15(&info, NULL, &cost, 6, &selected) ==
          PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_matmul_select_v15(&info, &target, NULL, 6, &selected) ==
          PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_matmul_select_v15(&info, &target, &cost, 6, NULL) ==
          PAN_GRAPH_TSU_INVALID);
   puts("MatMul automatic selection: original costs, stable minima, row tile, "
        "partial K and transactional rejection passed");
   return 0;
}
