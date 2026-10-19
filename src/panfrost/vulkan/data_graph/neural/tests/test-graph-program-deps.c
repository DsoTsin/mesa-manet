#include "pan_graph_program_deps.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#ifdef TEST_PROGRAM_DEPS_OOM
static size_t allocations, fail_at = SIZE_MAX;
void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *
__wrap_malloc(size_t size)
{
   return allocations++ == fail_at ? NULL : __real_malloc(size);
}
void *
__wrap_calloc(size_t n, size_t size)
{
   return allocations++ == fail_at ? NULL : __real_calloc(n, size);
}
#endif

static struct pan_graph_program_view
view(uint32_t tensor)
{
   return (struct pan_graph_program_view){
      .tensor = tensor,
      .kind = tensor >= 5 ? 3 : tensor == 3 ? 4 : 1,
      .bounds = {.rank = 3, .high = {tensor == 5 ? 5 : 4,
                                    tensor == 6 ? 5 : 3, 2}},
   };
}

static void
reject(const struct pan_graph_matmul_info *info,
       const struct pan_graph_program_memory *memory)
{
   struct pan_graph_program_deps result, before;
   memset(&result, 0xa5, sizeof(result));
   before = result;
   assert(pan_graph_program_deps_matmul_add_v15(info, 3, memory, &result) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&result, &before, sizeof(result)));
}

int
main(void)
{
   const struct pan_graph_matmul_info info = {2, 3, 5, 4, 0, 0};
   struct pan_graph_program_view inputs[][3] = {
      {view(5), view(6)}, {view(0), view(0)},
      {view(0)}, {view(2), view(1), view(2)},
   };
   struct pan_graph_program_view outputs[][2] = {
      {view(0), view(0)}, {view(1)}, {view(2)}, {view(3)},
   };
   struct pan_graph_program_views views[] = {
      {inputs[0], outputs[0], 2, 2, 0},
      {inputs[1], outputs[1], 2, 1, 0},
      {inputs[2], outputs[2], 1, 1, 0},
      {inputs[3], outputs[3], 3, 1, 0},
   };
   struct pan_graph_memory_lifetime_group groups[] = {
      {1, 0, 1}, {2, 1, 1}, {3, 2, 1},
   };
   struct pan_graph_alloc_request requests[] = {
      {0, 5, 128}, {2, 7, 128}, {4, 7, 128},
   };
   struct pan_graph_allocation allocated[] = {{0, 0, -1}, {128, 1, -1},
                                              {256, 2, -1}};
   struct pan_graph_memory_lifetime_identity identities[7];
   for (unsigned i = 0; i < 7; i++)
      identities[i] = (struct pan_graph_memory_lifetime_identity){
         i < 3 ? i * 128 : UINT64_MAX, i < 3 ? i : UINT32_MAX};
   struct pan_graph_memory_resource_entry entries[4] = {0};
   struct pan_graph_memory_resource_reference references[7] = {0};
   for (unsigned i = 0; i < 3; i++) {
      entries[i].uid = i + 1;
      entries[i].descriptor.size = 128;
      references[i] = (struct pan_graph_memory_resource_reference){
         i * 2, PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT};
   }
   entries[3].descriptor.size = 96;
   references[3] = (struct pan_graph_memory_resource_reference){
      0, PAN_GRAPH_MEMORY_RESOURCE_OUTPUT};
   struct pan_graph_program_memory memory = {
      .views = views,
      .program_count = 4,
      .lifetimes = {
         .groups = groups, .requests = requests, .allocations = allocated,
         .identities = identities, .group_count = 3, .tensor_count = 7,
      },
      .resources = {
         .domains = {
            [PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT] = {entries, 3},
            [PAN_GRAPH_MEMORY_RESOURCE_OUTPUT] = {entries + 3, 1},
         },
         .references = references, .tensor_count = 7,
      },
   };
   struct pan_graph_program_deps result = {0};
   assert(pan_graph_program_deps_matmul_add_v15(&info, 3, &memory, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   const uint32_t offsets[] = {0, 0, 1, 2, 4};
   const uint32_t indices[] = {0, 0, 2, 1};
   assert(result.program_count == 4 && result.index_count == 4);
   assert(!memcmp(result.offsets, offsets, sizeof(offsets)));
   assert(!memcmp(result.indices, indices, sizeof(indices)));
   pan_graph_program_deps_finish(&result);
   pan_graph_program_deps_finish(&result);
   assert(!result.offsets && !result.indices && !result.program_count);
   inputs[1][0] = view(2);
   reject(&info, &memory);
   inputs[1][0] = view(0);
   outputs[2][0] = view(1);
   reject(&info, &memory);
   outputs[2][0] = view(2);
   inputs[1][0].bounds.high[0]--;
   reject(&info, &memory);
   inputs[1][0] = view(0);
   inputs[1][0].bounds.low[1] = 1;
   reject(&info, &memory);
   inputs[1][0] = view(0);
   groups[1].alias_count = 2;
   reject(&info, &memory);
   groups[1].alias_count = 0;
   reject(&info, &memory);
   groups[1].alias_count = 1;
   allocated[1].offset = identities[1].offset = 64;
   reject(&info, &memory);
   allocated[1].offset = identities[1].offset = UINT64_MAX - 63;
   reject(&info, &memory);
   allocated[1].offset = identities[1].offset = 128;
   identities[2].group = 1;
   reject(&info, &memory);
   identities[2].group = 2;
   requests[0].size = 0;
   reject(&info, &memory);
   requests[0].size = 128;
   struct pan_graph_matmul_info large = info;
   large.columns = UINT16_MAX + 1;
   reject(&large, &memory);
   outputs[3][0] = view(5);
   reject(&info, &memory);
   outputs[3][0] = view(3);
   entries[0].descriptor.size--;
   reject(&info, &memory);
   entries[0].descriptor.size++;
   requests[0].last = 3;
   inputs[2][0] = view(1);
   views[3].input_count = 1;
   allocated[2].offset = identities[2].offset = 0;
   assert(pan_graph_program_deps_matmul_add_v15(&info, 3, &memory, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   const uint32_t reuse_indices[] = {0, 1, 2, 1};
   assert(result.index_count == 4);
   assert(!memcmp(result.offsets, offsets, sizeof(offsets)));
   assert(!memcmp(result.indices, reuse_indices, sizeof(reuse_indices)));
   pan_graph_program_deps_finish(&result);
   allocated[2].offset = identities[2].offset = INT32_MAX - 63;
   reject(&info, &memory);
   allocated[2].offset = identities[2].offset = 256;
   inputs[2][0] = view(0);
   views[3].input_count = 3;
   requests[0].last = 5;
   uint32_t ring_offsets[31], ring_indices[29];
   uint8_t waited[30] = {0};
   for (unsigned i = 0; i < 31; i++)
      ring_offsets[i] = i ? i - 1 : 0;
   for (unsigned i = 0; i < 29; i++)
      ring_indices[i] = i;
   struct pan_graph_program_deps ring = {ring_offsets, ring_indices, 30, 29};
   for (unsigned i = 0; i < 30; i++)
      assert(pan_graph_program_deps_wait_mask(&ring, i, 3, 13, waited) ==
             (i ? 1u << (3 + (i - 1) % 13) : 0));
   assert(waited[0] && waited[13] && waited[26] && !waited[29]);
#ifdef TEST_PROGRAM_DEPS_OOM
   allocations = 0;
   assert(pan_graph_program_deps_matmul_add_v15(&info, 3, &memory, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   size_t total = allocations;
   pan_graph_program_deps_finish(&result);
   assert(total == 3);
   for (size_t i = 0; i < total; i++) {
      struct pan_graph_program_deps rejected, before;
      memset(&rejected, 0xa5, sizeof(rejected));
      before = rejected;
      allocations = 0;
      fail_at = i;
      assert(pan_graph_program_deps_matmul_add_v15(&info, 3, &memory, &rejected) ==
             PAN_GRAPH_TSU_OUT_OF_MEMORY);
      assert(!memcmp(&rejected, &before, sizeof(before)));
   }
   fail_at = SIZE_MAX;
#endif
   return 0;
}
