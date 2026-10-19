#include "pan_graph_program_memory.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#ifdef TEST_PROGRAM_MEMORY_OOM
static size_t allocations, fail_at = SIZE_MAX;
void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void *
__wrap_malloc(size_t a)
{
   return allocations++ == fail_at ? NULL : __real_malloc(a);
}
void *
__wrap_calloc(size_t a, size_t b)
{
   return allocations++ == fail_at ? NULL : __real_calloc(a, b);
}
void *
__wrap_realloc(void *a, size_t b)
{
   return allocations++ == fail_at ? NULL : __real_realloc(a, b);
}
#endif

int
main(void)
{
   struct pan_graph_matmul_info info = {2, 3, 5, 4, 0, 0};
   const struct pan_graph_matmul_add adds[] = {{{0, 0}}, {{1, 1}}};
   const struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   struct pan_graph_memory_cost_target cost;
   assert(pan_graph_program_cost_target_v15_init(8, 64, 262144, 6, &cost));
   struct pan_graph_program_schedule schedule = {0};
   assert(pan_graph_program_schedule_matmul_add_v15(&info, adds, 2, &target,
                                                    &cost, 6, 8, &schedule) ==
          PAN_GRAPH_TSU_SUCCESS);
   struct pan_graph_program_memory_binding bindings[] = {{{3, 7, 2}, 0},
                                                         {{2, 5, 1}, 1},
                                                         {{1, 9, 3}, 0}};
   struct pan_graph_program_memory memory = {0};
#ifdef TEST_PROGRAM_MEMORY_OOM
   allocations = 0;
#endif
   assert(pan_graph_program_memory_matmul_add_v15(&info, adds, 2, &schedule,
                                                  &cost, bindings, &memory) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(memory.program_count == 2);
   assert(memory.program_bindings && memory.memories);
   assert(memory.program_bindings[0].read_count == 2 &&
          memory.program_bindings[0].write_count == 1);
   const struct pan_graph_v15_memory *ce_write =
      memory.program_bindings[0].writes;
   const struct pan_graph_v15_memory *ve_read =
      memory.program_bindings[1].reads;
   const struct pan_graph_v15_memory *public_write =
      memory.program_bindings[1].writes;
   assert(ce_write->domain == 1 && ce_write->offset == 0 &&
          ce_write->level == 5 && ce_write->override == 1 &&
          ce_write->value == 1 && !ce_write->flags && ce_write->extra == 1);
   assert(ve_read->domain == 1 && ve_read->offset == 0 && ve_read->level == 3 &&
          ve_read->override == 1 && ve_read->value == 1 && !ve_read->flags);
   assert(public_write->domain == 3 && !public_write->offset &&
          public_write->level == 5 && !public_write->override &&
          !public_write->flags && !public_write->value && !public_write->extra);
   assert(memory.views[1].input_count == 4);
   for (unsigned i = 0; i < 4; i++) {
      const struct pan_graph_program_view *view = memory.views[1].inputs + i;
      assert(view->tensor == 0 && view->key == 1 + 256 * i);
      assert(view->bounds.high[0] == 4 && view->bounds.high[1] == 3 &&
             view->bounds.high[2] == 2);
   }
   assert(memory.resources.domains[0].count == 2 &&
          memory.resources.domains[1].count == 1 &&
          !memory.resources.domains[2].count &&
          memory.resources.domains[3].count == 1);
   const struct pan_graph_memory_resource_reference *references =
      memory.resources.references;
   assert(references[0].domain == PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT &&
          references[0].slot32 == 0);
   assert(references[4].domain == PAN_GRAPH_MEMORY_RESOURCE_INPUT &&
          references[4].slot32 == 0);
   assert(references[5].domain == PAN_GRAPH_MEMORY_RESOURCE_INPUT &&
          references[5].slot32 == 2);
   assert(references[2].domain == PAN_GRAPH_MEMORY_RESOURCE_OUTPUT &&
          references[2].slot32 == 0);
   assert(references[1].slot32 == UINT32_MAX &&
          references[3].slot32 == UINT32_MAX);
   const struct pan_graph_memory_resource_entry *entry =
      memory.resources.domains[1].entries;
   const uint32_t expected[] = {3467, 131073, 262147, 128, 64, 64, 0, 0};
   assert(!memcmp(entry->descriptor.words, expected, sizeof(expected)));
   assert(entry->descriptor.size == 128 && entry->uid == 1);
   assert(memory.lifetimes.group_count == 1 &&
          memory.lifetimes.requests[0].first == 0 &&
          memory.lifetimes.requests[0].last == 3 &&
          memory.lifetimes.requests[0].size == 128 &&
          memory.lifetimes.stats.peak == 128 &&
          memory.lifetimes.end_clock == 4);
   assert(memory.lifetimes.identities[0].offset == 0 &&
          memory.lifetimes.identities[1].group == UINT32_MAX);
   assert(!memcmp(memory.resources.domains[0].entries[0].binding,
                  bindings[0].binding, sizeof(bindings[0].binding)));
#ifdef TEST_PROGRAM_MEMORY_OOM
   size_t total = allocations;
   for (size_t i = 0; i < total; i++) {
      struct pan_graph_program_memory rejected, before;
      memset(&rejected, 0xa5, sizeof(rejected));
      before = rejected;
      allocations = 0;
      fail_at = i;
      assert(pan_graph_program_memory_matmul_add_v15(&info, adds, 2, &schedule,
                                                     &cost, bindings, &rejected) ==
             PAN_GRAPH_TSU_OUT_OF_MEMORY);
      assert(!memcmp(&rejected, &before, sizeof(before)));
   }
   fail_at = SIZE_MAX;
#endif
   pan_graph_program_memory_finish(&memory);
   assert(!memory.views && !memory.program_bindings && !memory.memories &&
          !memory.program_count && !memory.resources.storage &&
          !memory.lifetimes.groups);
   struct pan_graph_memory_cost_target alternate = cost;
   alternate.memory[3].parallel = alternate.memory[4].parallel = 1;
   alternate.memory[3].size = 255;
   alternate.memory[4].size = 128;
   assert(pan_graph_program_memory_matmul_add_v15(
             &info, adds, 2, &schedule, &alternate, bindings, &memory) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(memory.program_bindings[1].reads[0].level == 4);
   assert(memory.program_bindings[0].writes[0].level == 5);
   pan_graph_program_memory_finish(&memory);
   alternate.memory[4].size = 127;
   assert(pan_graph_program_memory_matmul_add_v15(
             &info, adds, 2, &schedule, &alternate, bindings, &memory) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(memory.program_bindings[1].reads[0].level == 5);
   pan_graph_program_memory_finish(&memory);
   alternate.memory[3].size = 257;
   assert(pan_graph_program_memory_matmul_add_v15(
             &info, adds, 2, &schedule, &alternate, bindings, &memory) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(memory.program_bindings[1].reads[0].level == 3);
   pan_graph_program_memory_finish(&memory);
   bindings[0].ordinal = 1;
   bindings[1].ordinal = 0;
   assert(pan_graph_program_memory_matmul_add_v15(&info, adds, 2, &schedule,
                                                  &cost, bindings, &memory) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(memory.resources.domains[0].entries[0].uid == 6 &&
          memory.resources.domains[0].entries[1].uid == 5);
   pan_graph_program_memory_finish(&memory);
   struct pan_graph_program_memory before;
   memset(&memory, 0xa5, sizeof(memory));
   before = memory;
   assert(pan_graph_program_memory_matmul_add_v15(
             &info, adds, 2, &schedule, NULL, bindings, &memory) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&before, &memory, sizeof(before)));
   struct pan_graph_matmul_info wide_info = info;
   wide_info.rows = UINT16_MAX + 1u;
   assert(pan_graph_program_memory_matmul_add_v15(
             &wide_info, adds, 2, &schedule, &cost, bindings, &memory) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&before, &memory, sizeof(before)));
   alternate = cost;
   alternate.memory[3].size = UINT64_MAX;
   assert(pan_graph_program_memory_matmul_add_v15(
             &info, adds, 2, &schedule, &alternate, bindings, &memory) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&before, &memory, sizeof(before)));
   bindings[1].ordinal = 1;
   assert(pan_graph_program_memory_matmul_add_v15(&info, adds, 2, &schedule,
                                                  &cost, bindings, &memory) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&before, &memory, sizeof(before)));
   bindings[1].ordinal = 0;
   info.rows++;
   assert(pan_graph_program_memory_matmul_add_v15(&info, adds, 2, &schedule,
                                                  &cost, bindings, &memory) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&before, &memory, sizeof(before)));
   info.rows--;
   schedule.groups[0] &= ~UINT64_C(1);
   assert(pan_graph_program_memory_matmul_add_v15(&info, adds, 2, &schedule,
                                                  &cost, bindings, &memory) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&before, &memory, sizeof(before)));
   schedule.groups[0] |= 1;
   pan_graph_program_schedule_finish(&schedule);
   info = (struct pan_graph_matmul_info){1, 32, 32, 32, 0, 0};
   assert(pan_graph_program_schedule_matmul_add_v15(&info, adds, 1, &target,
                                                    &cost, 6, 8, &schedule) ==
          PAN_GRAPH_TSU_SUCCESS);
   memory = (struct pan_graph_program_memory){0};
   assert(pan_graph_program_memory_matmul_add_v15(&info, adds, 1, &schedule,
                                                  &cost, bindings, &memory) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(memory.program_count == 1 && !memory.lifetimes.group_count &&
          !memory.lifetimes.stats.peak && !memory.resources.domains[1].count);
   pan_graph_program_memory_finish(&memory);
   pan_graph_program_schedule_finish(&schedule);
   pan_graph_program_memory_finish(NULL);
   return 0;
}
