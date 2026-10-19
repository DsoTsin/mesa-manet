#include "pan_graph_program_prune.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#ifdef TEST_PRUNE_ALLOC_FAILURES
static size_t allocation_count, allocation_failure;
void *__real_calloc(size_t count, size_t size);
void *__wrap_calloc(size_t count, size_t size)
{
   allocation_count++;
   return allocation_count == allocation_failure ? NULL : __real_calloc(count, size);
}
#endif

static void
test_serial(void)
{
   struct pan_graph_cost_report a = {
      .resource_bytes = {{3, 5}, {7, 11}},
      .peak_resource = {13, 17},
      .memory_cycles = {19, 23},
      .engine_cycles = {29, 31},
      .critical_kind = 37,
      .critical_cycles = 41,
      .cycles = 43,
   };
   struct pan_graph_cost_report b = a, result;
   b.peak_resource[0] = 1000;
   b.critical_kind = 2000;
   assert(pan_graph_cost_accumulate_serial(&a, &b, &result));
   assert(result.resource_bytes[0][1] == 10);
   assert(result.resource_bytes[1][1] == 22);
   assert(result.peak_resource[0] == 13 && result.peak_resource[1] == 17);
   assert(result.memory_cycles[1] == 46 && result.engine_cycles[1] == 62);
   assert(result.critical_kind == 37 && result.critical_cycles == 82 && result.cycles == 86);
   assert(pan_graph_cost_accumulate_serial(&a, &b, &a));
   assert(!memcmp(&a, &result, sizeof(a)));
   struct pan_graph_cost_report before = result;
   a.cycles = INT64_MAX;
   assert(!pan_graph_cost_accumulate_serial(&a, &b, &result));
   assert(!memcmp(&before, &result, sizeof(result)));
   assert(!pan_graph_cost_accumulate_serial(NULL, &b, &result));
}

static void
test_ties(void)
{
   uint64_t groups[] = {7, 3, 1, 2, 4, 6};
   uint64_t cycles[] = {50, 19, 10, 10, 10, 19};
   struct pan_graph_program_cost candidates[6] = {0};
   for (unsigned i = 0; i < 6; i++) {
      candidates[i].groups = groups + i;
      candidates[i].cost.cycles = cycles[i];
      candidates[i].cost.peak_resource[0] = (i + 1) * 100;
      candidates[i].cost.critical_kind = i + 1;
   }
   struct pan_graph_program_prune_result result = {0};
   assert(pan_graph_program_prune_costs(3, candidates, 6, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.count == 6 && !result.entries[0].retained);
   assert(result.entries[0].cost.cycles == 29);
   assert(result.entries[0].cost.peak_resource[0] == 200);
   assert(result.entries[0].cost.critical_kind == 2);
   for (unsigned i = 1; i < 6; i++)
      assert(result.entries[i].retained);
   pan_graph_program_prune_finish(&result);
   assert(!result.count && !result.entries);
   candidates[0].cost.cycles = 29;
   assert(pan_graph_program_prune_costs(3, candidates, 6, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(!result.entries[0].retained && result.entries[0].cost.critical_kind == 2);
   pan_graph_program_prune_finish(&result);
   pan_graph_program_prune_finish(NULL);
   struct pan_graph_program_choice choice = {0};
   assert(pan_graph_program_choose_costs(3, groups, candidates, 6, &choice) == PAN_GRAPH_TSU_SUCCESS);
   assert(choice.count == 6 && choice.cycles == 29 && choice.selected[0] == 0x24);
   pan_graph_program_choice_finish(&choice);
   assert(!choice.count && !choice.selected && !choice.cycles);
   pan_graph_program_choice_finish(NULL);
}

static void
test_wide_and_transaction(void)
{
   const uint64_t groups[][2] = {{1, 2}, {1, 0}, {0, 2}};
   struct pan_graph_program_cost candidates[] = {
      {groups[0], {.cycles = 20}},
      {groups[1], {.cycles = 10}},
      {groups[2], {.cycles = 10}},
   };
   struct pan_graph_program_prune_result result = {0};
   assert(pan_graph_program_prune_costs(66, candidates, 3, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(!result.entries[0].retained && result.entries[0].cost.cycles == 20);
   pan_graph_program_prune_finish(&result);
   struct pan_graph_program_choice choice = {0};
   assert(pan_graph_program_choose_costs(66, groups[0], candidates, 3, &choice) == PAN_GRAPH_TSU_SUCCESS);
   assert(choice.count == 3 && choice.cycles == 20 && choice.selected[0] == 6);
   pan_graph_program_choice_finish(&choice);
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_program_prune_result before = result;
   assert(pan_graph_program_prune_costs(65, candidates, 3, &result) == PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&result, &before, sizeof(result)));
   candidates[2].groups = groups[1];
   assert(pan_graph_program_prune_costs(66, candidates, 3, &result) == PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&result, &before, sizeof(result)));
   candidates[2].groups = groups[2];
   candidates[0].cost.cycles = INT64_MAX;
   assert(pan_graph_program_prune_costs(66, candidates, 3, &result) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&result, &before, sizeof(result)));
   candidates[0].cost.cycles = 20;
   candidates[1].cost.cycles = INT64_MAX - 1;
   assert(pan_graph_program_prune_costs(66, candidates, 3, &result) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&result, &before, sizeof(result)));
   candidates[1].groups = NULL;
   assert(pan_graph_program_prune_costs(66, candidates, 3, &result) == PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&result, &before, sizeof(result)));
   assert(pan_graph_program_prune_costs(0, candidates, 3, &result) == PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_program_prune_costs(66, NULL, 3, &result) == PAN_GRAPH_TSU_INVALID);
}

static void
test_wide_selection(void)
{
   uint64_t groups[70][2] = {{0}};
   struct pan_graph_program_cost candidates[70] = {0};
   for (unsigned i = 0; i < 70; i++) {
      groups[i][i / 64] = UINT64_C(1) << (i % 64);
      candidates[i].groups = groups[i];
      candidates[i].cost.cycles = 10;
   }
   uint64_t request[] = {UINT64_MAX, 63};
   struct pan_graph_program_choice result = {0};
   assert(pan_graph_program_choose_costs(70, request, candidates, 70, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.count == 70 && result.cycles == 700);
   assert(result.selected[0] == UINT64_MAX && result.selected[1] == 63);
   pan_graph_program_choice_finish(&result);
#ifdef TEST_PRUNE_ALLOC_FAILURES
   allocation_count = 0;
   assert(pan_graph_program_choose_costs(70, request, candidates, 70, &result) == PAN_GRAPH_TSU_SUCCESS);
   size_t total = allocation_count;
   pan_graph_program_choice_finish(&result);
   assert(total >= 11);
   for (size_t failure = 1; failure <= total; failure++) {
      allocation_count = 0;
      allocation_failure = failure;
      memset(&result, 0xa5, sizeof(result));
      struct pan_graph_program_choice before = result;
      assert(pan_graph_program_choose_costs(70, request, candidates, 70, &result) == PAN_GRAPH_TSU_OUT_OF_MEMORY);
      assert(!memcmp(&result, &before, sizeof(result)));
   }
   allocation_failure = 0;
#endif
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_program_choice before = result;
   assert(pan_graph_program_choose_costs(70, request, candidates, 69, &result) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&result, &before, sizeof(result)));
   assert(pan_graph_program_choose_costs(69, request, candidates, 69, &result) == PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&result, &before, sizeof(result)));
   assert(pan_graph_program_choose_costs(70, NULL, candidates, 70, &result) == PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&result, &before, sizeof(result)));
}

static void
test_count_policy(void)
{
   uint64_t groups[] = {1, 2, 4, 3, 6, 7};
   const uint64_t costs[] = {1, 1, 9, 9, 1, 1000};
   struct pan_graph_program_cost candidates[6] = {0};
   for (unsigned i = 0; i < 6; i++) {
      candidates[i].groups = groups + i;
      candidates[i].cost.cycles = costs[i];
   }
   uint64_t request = 7;
   struct pan_graph_program_choice result = {0};
   assert(pan_graph_program_choose(3, &request, candidates, 5,
                                    PAN_GRAPH_PROGRAM_CHOICE_COUNT, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.selected[0] == 12 && result.cycles == 18);
   pan_graph_program_choice_finish(&result);
   assert(pan_graph_program_choose_costs(3, &request, candidates, 5, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.selected[0] == 17 && result.cycles == 2);
   pan_graph_program_choice_finish(&result);
   assert(pan_graph_program_choose(3, &request, candidates, 6,
                                    PAN_GRAPH_PROGRAM_CHOICE_COUNT, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.selected[0] == 32 && result.cycles == 1000);
   pan_graph_program_choice_finish(&result);
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_program_choice before = result;
   assert(pan_graph_program_choose(3, &request, candidates, 6, -1, &result) == PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&result, &before, sizeof(result)));
}

static void
test_components(void)
{
   uint64_t groups[] = {3, 1, 2, 12, 4, 8};
   struct pan_graph_program_cost candidates[6] = {0};
   for (unsigned i = 0; i < 6; i++) {
      candidates[i].groups = groups + i;
      candidates[i].cost.cycles = i % 3 == 0 ? 15 : 10;
   }
   struct pan_graph_program_choice result = {0};
   assert(pan_graph_program_select_costs(4, candidates, 6, &result) == PAN_GRAPH_TSU_SUCCESS);
   assert(result.cycles == 30 && result.selected[0] == 9);
   pan_graph_program_choice_finish(&result);
#ifdef TEST_PRUNE_ALLOC_FAILURES
   allocation_count = 0;
   assert(pan_graph_program_select_costs(4, candidates, 6, &result) == PAN_GRAPH_TSU_SUCCESS);
   size_t total = allocation_count;
   pan_graph_program_choice_finish(&result);
   assert(total >= 20);
   for (size_t failure = 1; failure <= total; failure++) {
      allocation_count = 0;
      allocation_failure = failure;
      memset(&result, 0xa5, sizeof(result));
      struct pan_graph_program_choice before = result;
      assert(pan_graph_program_select_costs(4, candidates, 6, &result) == PAN_GRAPH_TSU_OUT_OF_MEMORY);
      assert(!memcmp(&result, &before, sizeof(result)));
   }
   allocation_failure = 0;
#endif
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_program_choice before = result;
   assert(pan_graph_program_select_costs(5, candidates, 6, &result) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&result, &before, sizeof(result)));
   assert(pan_graph_program_select_costs(4, candidates, 6, NULL) == PAN_GRAPH_TSU_INVALID);
   struct pan_graph_program_cost disconnected[] = {
      {groups + 1, {.cycles = INT64_MAX - 1}},
      {groups + 2, {.cycles = 10}},
   };
   assert(pan_graph_program_select_costs(2, disconnected, 2, &result) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&result, &before, sizeof(result)));
}

static void
test_cache_growth(void)
{
   uint64_t groups[21];
   struct pan_graph_program_cost candidates[21] = {0};
   unsigned count = 0;
   for (unsigned i = 0; i < 6; i++)
      groups[count++] = 1ull << i;
   for (unsigned i = 0; i < 6; i++) {
      for (unsigned j = i + 1; j < 6; j++)
         groups[count++] = (1ull << i) | (1ull << j);
   }
   for (unsigned i = 0; i < count; i++) {
      candidates[i].groups = groups + i;
      candidates[i].cost.cycles = i < 6 ? 10 : 20;
   }
   struct pan_graph_program_prune_result result = {0};
   assert(pan_graph_program_prune_costs(6, candidates, count, &result) == PAN_GRAPH_TSU_SUCCESS);
   for (unsigned i = 0; i < count; i++)
      assert(result.entries[i].retained == (i < 6));
   pan_graph_program_prune_finish(&result);
#ifdef TEST_PRUNE_ALLOC_FAILURES
   allocation_count = 0;
   assert(pan_graph_program_prune_costs(6, candidates, count, &result) == PAN_GRAPH_TSU_SUCCESS);
   size_t total = allocation_count;
   pan_graph_program_prune_finish(&result);
   assert(total >= 7);
   for (size_t failure = 1; failure <= total; failure++) {
      allocation_count = 0;
      allocation_failure = failure;
      memset(&result, 0xa5, sizeof(result));
      struct pan_graph_program_prune_result before = result;
      assert(pan_graph_program_prune_costs(6, candidates, count, &result) == PAN_GRAPH_TSU_OUT_OF_MEMORY);
      assert(!memcmp(&result, &before, sizeof(result)));
   }
   allocation_failure = 0;
#endif
}

int main(void)
{
   test_serial();
   test_ties();
   test_wide_and_transaction();
   test_cache_growth();
   test_wide_selection();
   test_count_policy();
   test_components();
   return 0;
}
