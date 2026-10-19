#include "pan_graph_memory_policy.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#ifdef TEST_MEMORY_POLICY_NO_ALLOC
void *__wrap_malloc(size_t size) { abort(); }
void *__wrap_calloc(size_t count, size_t size) { abort(); }
void *__wrap_realloc(void *pointer, size_t size) { abort(); }
#endif

#ifdef TEST_MEMORY_POLICY_FAIL_ALLOC
static size_t allocation_count;
static size_t fail_allocation;
void *__real_malloc(size_t size);
void *__wrap_malloc(size_t size)
{
   allocation_count++;
   return allocation_count == fail_allocation ? NULL : __real_malloc(size);
}
#endif

static void
expect(const struct pan_graph_memory_policy_input *input, uint64_t uid,
       struct pan_graph_memory_policy wanted)
{
   struct pan_graph_memory_policy result;
   memset(&result, 0xa5, sizeof(result));
   assert(pan_graph_memory_policy_lookup(input, uid, &result));
   assert(result.level == wanted.level && result.override == wanted.override &&
          result.flags == wanted.flags && result.value == wanted.value);
}

static void
reject(const struct pan_graph_memory_policy_input *input)
{
   struct pan_graph_memory_policy result;
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_memory_policy before = result;
   assert(!pan_graph_memory_policy_lookup(input, 42, &result));
   assert(!memcmp(&result, &before, sizeof(result)));
}

#ifndef TEST_MEMORY_POLICY_NO_ALLOC
static struct pan_graph_memory_policy_source_view
view(uint64_t uid, uint32_t key, uint32_t kind)
{
   return (struct pan_graph_memory_policy_source_view){
      .uid = uid,
      .key = key,
      .kind = kind,
      .rank = 2,
      .low = {-1, 2},
      .high = {7, 8},
      .policy = {3, 0xa2, 0xb3, 0xc4},
   };
}

static void
expect_policy(const struct pan_graph_memory_policy *policy, uint32_t level,
              uint8_t override, uint8_t value)
{
   assert(policy->level == level && policy->override == override &&
          policy->flags == 0 && policy->value == value);
}

static void
finalize_reject(const struct pan_graph_memory_policy_finalize_input *input,
                enum pan_graph_tsu_status status)
{
   struct pan_graph_memory_policy_finalized result;
   memset(&result, 0xa5, sizeof(result));
   struct pan_graph_memory_policy_finalized before = result;
   assert(pan_graph_memory_policy_finalize(input, &result) == status);
   assert(!memcmp(&result, &before, sizeof(result)));
}

static void
test_finalize(void)
{
   struct pan_graph_memory_policy_source_view inputs[] = {
      view(42, 1, 1), view(42, 257, 1), view(42, 513, 1), view(43, 769, 4),
   };
   inputs[1].low[6] = 123;
   inputs[1].high[6] = -123;
   inputs[2].high[1]++;
   struct pan_graph_memory_policy_source_view outputs[] = {
      view(42, 2, 1), view(42, 258, 1),
   };
   const struct pan_graph_memory_policy_snapshot input_snapshot[] = {
      {257, 5}, {513, 4}, {513, 6}, {769, 2},
   };
   const struct pan_graph_memory_policy_snapshot output_snapshot[] = {
      {2, 5}, {258, 6}, {258, 7},
   };
   struct pan_graph_memory_policy_finalize_input input = {
      inputs, 4, outputs, 2, input_snapshot, 4, output_snapshot, 3, 1, 1};
   struct pan_graph_memory_policy_finalized result = {0};
   assert(pan_graph_memory_policy_finalize(&input, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(result.input_count == 4 && result.output_count == 2);
   const uint32_t expected[] = {3, 3, 6, 2, 5, 7};
   for (size_t i = 0; i < 6; i++)
      expect_policy(&result.policies[i], expected[i], i != 3, i != 3);
   assert(inputs[0].policy.override == 0xa2 &&
          inputs[0].policy.flags == 0xb3 && inputs[0].policy.value == 0xc4);
   pan_graph_memory_policy_finalized_finish(&result);
   assert(!result.policies && !result.input_count && !result.output_count);
   pan_graph_memory_policy_finalized_finish(&result);
   pan_graph_memory_policy_finalized_finish(NULL);

   const uint32_t kinds[] = {0, 1, 2, 3, 4, UINT32_MAX};
   const uint8_t configs[] = {0, 1, 2, 255};
   for (size_t k = 0; k < sizeof(kinds) / sizeof(*kinds); k++) {
      for (size_t load = 0; load < sizeof(configs); load++) {
         for (size_t store = 0; store < sizeof(configs); store++) {
            inputs[0] = view(UINT64_MAX, UINT32_MAX, kinds[k]);
            inputs[0].policy.level = UINT32_MAX;
            input = (struct pan_graph_memory_policy_finalize_input){
               .inputs = inputs,
               .input_count = 1,
               .load_speculative = configs[load],
               .store_speculative = configs[store],
            };
            assert(pan_graph_memory_policy_finalize(&input, &result) ==
                   PAN_GRAPH_TSU_SUCCESS);
            bool eligible = kinds[k] >= 1 && kinds[k] <= 3;
            expect_policy(&result.policies[0], UINT32_MAX,
                          eligible && configs[store] == 1,
                          eligible && configs[load] == 1);
            pan_graph_memory_policy_finalized_finish(&result);
         }
      }
   }

   input = (struct pan_graph_memory_policy_finalize_input){0};
   assert(pan_graph_memory_policy_finalize(&input, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(!result.policies && !result.input_count && !result.output_count);
   finalize_reject(NULL, PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_memory_policy_finalize(&input, NULL) == PAN_GRAPH_TSU_INVALID);
   input.input_count = 1;
   finalize_reject(&input, PAN_GRAPH_TSU_INVALID);
   input.input_count = 0;
   input.output_count = 1;
   finalize_reject(&input, PAN_GRAPH_TSU_INVALID);
   input.output_count = 0;
   input.input_snapshot_count = 1;
   finalize_reject(&input, PAN_GRAPH_TSU_INVALID);
   input.input_snapshot_count = 0;
   input.output_snapshot_count = 1;
   finalize_reject(&input, PAN_GRAPH_TSU_INVALID);
   input.output_snapshot_count = 0;
   input.inputs = inputs;
   input.input_count = (size_t)INT32_MAX + 1;
   finalize_reject(&input, PAN_GRAPH_TSU_INVALID);
   input.input_count = 0;
   input.outputs = outputs;
   input.output_count = SIZE_MAX;
   finalize_reject(&input, PAN_GRAPH_TSU_INVALID);
   input.output_count = 0;
   input.input_snapshot = input_snapshot;
   input.input_snapshot_count = (size_t)INT32_MAX + 1;
   finalize_reject(&input, PAN_GRAPH_TSU_INVALID);
   input.input_snapshot_count = 0;
   input.output_snapshot = output_snapshot;
   input.output_snapshot_count = SIZE_MAX;
   finalize_reject(&input, PAN_GRAPH_TSU_INVALID);

   inputs[0] = view(42, 1, 1);
   input = (struct pan_graph_memory_policy_finalize_input){
      .inputs = inputs, .input_count = 1};
   inputs[0].rank = 0;
   finalize_reject(&input, PAN_GRAPH_TSU_INVALID);
   inputs[0].rank = 8;
   finalize_reject(&input, PAN_GRAPH_TSU_UNSUPPORTED);
   inputs[0].rank = 2;
   inputs[0].low[0] = 10;
   finalize_reject(&input, PAN_GRAPH_TSU_INVALID);
   inputs[0] = view(42, 1, 1);
   inputs[1] = view(43, 1, 1);
   input.input_count = 2;
   finalize_reject(&input, PAN_GRAPH_TSU_INVALID);
   inputs[1] = view(42, 257, 4);
   finalize_reject(&input, PAN_GRAPH_TSU_UNSUPPORTED);
   input.input_count = 1;
   outputs[0] = view(42, 2, 4);
   input.outputs = outputs;
   input.output_count = 1;
   finalize_reject(&input, PAN_GRAPH_TSU_UNSUPPORTED);
   input.output_count = 0;
   inputs[0].rank = 7;
   for (size_t axis = 0; axis < 7; axis++) {
      inputs[0].low[axis] = INT32_MIN;
      inputs[0].high[axis] = INT32_MAX;
   }
   assert(pan_graph_memory_policy_finalize(&input, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   pan_graph_memory_policy_finalized_finish(&result);
#ifdef TEST_MEMORY_POLICY_FAIL_ALLOC
   allocation_count = 0;
   fail_allocation = 1;
   finalize_reject(&input, PAN_GRAPH_TSU_OUT_OF_MEMORY);
   assert(allocation_count == 1);
   fail_allocation = 0;
   allocation_count = 0;
   assert(pan_graph_memory_policy_finalize(&input, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(allocation_count == 1);
   pan_graph_memory_policy_finalized_finish(&result);
   input = (struct pan_graph_memory_policy_finalize_input){0};
   allocation_count = 0;
   fail_allocation = 1;
   assert(pan_graph_memory_policy_finalize(&input, &result) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(allocation_count == 0);
#endif
}
#endif

int
main(void)
{
   const struct pan_graph_memory_policy_view inputs[] = {
      {42, {3, 1, 0, 1}}, {43, {2, 0, 1, 0}}, {42, {4, 0, 1, 2}},
   };
   const struct pan_graph_memory_policy_view outputs[] = {
      {42, {5, 1, 1, 1}}, {43, {3, 0, 0, 0}}, {42, {2, 0, 0, 0}},
   };
   struct pan_graph_memory_policy_input input = {inputs, 3, outputs, 3};
   expect(&input, 42, outputs[2].policy);
   expect(&input, 43, outputs[1].policy);
   expect(&input, 0, (struct pan_graph_memory_policy){.level = 5});
   input.output_count = 0;
   expect(&input, 42, inputs[2].policy);
   expect(&input, 43, inputs[1].policy);
   input.input_count = 0;
   expect(&input, 42, (struct pan_graph_memory_policy){.level = 5});
   input.outputs = NULL;
   input.inputs = NULL;
   expect(&input, UINT64_MAX, (struct pan_graph_memory_policy){.level = 5});
   input.input_count = 1;
   reject(&input);
   input.input_count = 0;
   input.output_count = 1;
   reject(&input);
   input.outputs = outputs;
   input.output_count = (size_t)INT32_MAX + 1;
   reject(&input);
   input.output_count = 0;
   input.inputs = inputs;
   input.input_count = (size_t)INT32_MAX + 1;
   reject(&input);
   reject(NULL);
   assert(!pan_graph_memory_policy_lookup(&input, 42, NULL));
   const struct pan_graph_memory_policy_view all_bytes = {
      UINT64_MAX, {UINT32_MAX, 255, 127, 63},
   };
   input = (struct pan_graph_memory_policy_input){&all_bytes, 1, NULL, 0};
   expect(&input, UINT64_MAX, all_bytes.policy);
#ifndef TEST_MEMORY_POLICY_NO_ALLOC
   test_finalize();
#endif
   return 0;
}
