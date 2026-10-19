#include "pan_graph_vector_schedule.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#ifdef TEST_VECTOR_SCHEDULE_ALLOC_FAILURES
static size_t allocation_count, allocation_failure;
void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void *__wrap_malloc(size_t size)
{
   return ++allocation_count == allocation_failure ? NULL : __real_malloc(size);
}
void *__wrap_calloc(size_t count, size_t size)
{
   return ++allocation_count == allocation_failure ? NULL : __real_calloc(count, size);
}
void *__wrap_realloc(void *pointer, size_t size)
{
   return ++allocation_count == allocation_failure ? NULL : __real_realloc(pointer, size);
}
#endif

static void
reject(const struct pan_graph_vector_schedule_info *info,
       const struct pan_graph_nrt_target *target, const uint32_t caps[3])
{
   struct pan_graph_vector_candidate_list result;
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_vector_candidate_list before = result;
   assert(pan_graph_vector_base_candidates(info, caps, target, &result) !=
          PAN_GRAPH_TSU_SUCCESS);
   assert(!memcmp(&before, &result, sizeof(result)));
}

int
main(void)
{
   struct pan_graph_vector_schedule_info info = {
      .full = {1, 32, 32}, .granule = {1, 1, 1},
      .read_count = 1, .vector_count = 1, .export_count = 1, .write_count = 1,
   };
   const struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   struct pan_graph_vector_candidate_list candidates = {0};
   assert(pan_graph_vector_base_candidates(&info, info.full, &target, &candidates) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(candidates.count == 1 &&
          !memcmp(candidates.candidates[0].transfer, info.full, sizeof(info.full)));
   pan_graph_vector_candidates_finish(&candidates);
   const uint32_t expected[7][3] = {
      {1, 32, 32}, {1, 32, 16}, {1, 32, 8}, {1, 32, 4},
      {1, 32, 2}, {1, 11, 32}, {1, 11, 16},
   };
   assert(pan_graph_vector_parallel_candidates(&info, &target, 6, &candidates) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(candidates.count == 7 &&
          !memcmp(candidates.candidates, expected, sizeof(expected)));
   pan_graph_vector_candidates_finish(&candidates);
   info.granule[1] = info.granule[2] = 4;
   const uint32_t blocked[8][3] = {
      {1, 32, 32}, {1, 32, 16}, {1, 32, 8}, {1, 32, 4},
      {1, 12, 32}, {1, 12, 16}, {1, 4, 32}, {1, 4, 16},
   };
   assert(pan_graph_vector_parallel_candidates(&info, &target, 6, &candidates) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(candidates.count == 8 &&
          !memcmp(candidates.candidates, blocked, sizeof(blocked)));
   pan_graph_vector_candidates_finish(&candidates);
   uint32_t loops[4];
   assert(pan_graph_vector_assign_loops(&info, (uint32_t[]){1, 4, 16}, 6, loops));
   assert(!memcmp(loops, (uint32_t[]){2, 3, 12, 16}, sizeof(loops)));
   info.granule[1] = info.granule[2] = 1;
   assert(pan_graph_vector_assign_loops(&info, (uint32_t[]){1, 32, 2}, 6, loops));
   assert(!memcmp(loops, (uint32_t[]){3, 0, 6, 0}, sizeof(loops)));
   for (unsigned axis = 0; axis < 3; axis++) {
      struct pan_graph_vector_schedule_info large = info;
      large.full[0] = large.full[1] = large.full[2] = 1;
      large.full[axis] = 4096;
      assert(pan_graph_vector_base_candidates(&large, large.full, &target,
                                                &candidates) == PAN_GRAPH_TSU_SUCCESS);
      unsigned count = axis == 0 ? 6 : axis == 1 ? 8 : 7;
      assert(candidates.count == count);
      for (size_t i = 0; i < candidates.count; i++)
         assert(candidates.candidates[i].transfer[axis] == (16u << (count - i - 1)));
      pan_graph_vector_candidates_finish(&candidates);
   }
   for (unsigned budget = 0; budget < 2; budget++) {
      assert(pan_graph_vector_parallel_candidates(&info, &target, budget,
                                                    &candidates) == PAN_GRAPH_TSU_SUCCESS);
      assert(candidates.count == 1);
      pan_graph_vector_candidates_finish(&candidates);
   }
   for (unsigned i = 0; i < 11; i++) {
      struct pan_graph_vector_schedule_info bad = info;
      struct pan_graph_nrt_target bad_target = target;
      uint32_t caps[3];
      memcpy(caps, info.full, sizeof(caps));
      switch (i) {
      case 0: bad.full[0] = 0; break;
      case 1: bad.full[0] = UINT32_MAX; break;
      case 2: bad.granule[2] = 0; break;
      case 3: bad.read_count = 8; break;
      case 4: bad.vector_count = 8; break;
      case 5: bad.export_count = 2; break;
      case 6: bad.write_count = 0; break;
      case 7: bad.write_count = 8; break;
      case 8: caps[1] = 0; break;
      case 9: caps[2]++; break;
      case 10: bad_target.shared_size = 32768; break;
      }
      reject(&bad, &bad_target, caps);
   }
   reject(NULL, &target, info.full);
   reject(&info, NULL, info.full);
   reject(&info, &target, NULL);
   assert(pan_graph_vector_base_candidates(&info, info.full, &target, NULL) ==
          PAN_GRAPH_TSU_INVALID);
   memset(&candidates, 0xa5, sizeof(candidates));
   struct pan_graph_vector_candidate_list before = candidates;
   assert(pan_graph_vector_parallel_candidates(&info, &target, 65, &candidates) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&candidates, &before, sizeof(candidates)));
   memset(loops, 0xa5, sizeof(loops));
   uint32_t previous[4];
   memcpy(previous, loops, sizeof(loops));
   assert(!pan_graph_vector_assign_loops(&info, (uint32_t[]){1, 32, 33}, 6, loops));
   assert(!memcmp(previous, loops, sizeof(loops)));
   assert(!pan_graph_vector_assign_loops(&info, info.full, 65, loops));
   assert(!memcmp(previous, loops, sizeof(loops)));

#ifdef TEST_VECTOR_SCHEDULE_ALLOC_FAILURES
   size_t failures = 0;
   for (unsigned test = 0; test < 4; test++) {
      struct pan_graph_vector_schedule_info sample = info;
      if (test == 1)
         sample.granule[1] = sample.granule[2] = 4;
      if (test >= 2) {
         sample.full[0] = sample.full[1] = sample.full[2] = 1;
         sample.full[test - 1] = 4096;
      }
      allocation_count = 0;
      assert(pan_graph_vector_parallel_candidates(&sample, &target, 6, &candidates) ==
             PAN_GRAPH_TSU_SUCCESS);
      size_t total = allocation_count;
      pan_graph_vector_candidates_finish(&candidates);
      assert(total >= 2);
      for (size_t failure = 1; failure <= total; failure++) {
         allocation_count = 0;
         allocation_failure = failure;
         memset(&candidates, 0xa5, sizeof(candidates));
         before = candidates;
         assert(pan_graph_vector_parallel_candidates(&sample, &target, 6, &candidates) ==
                PAN_GRAPH_TSU_OUT_OF_MEMORY);
         assert(!memcmp(&candidates, &before, sizeof(candidates)));
         failures++;
      }
      allocation_failure = 0;
   }
   printf("allocation_failures=%zu\n", failures);
#endif
   pan_graph_vector_candidates_finish(NULL);
   return 0;
}
