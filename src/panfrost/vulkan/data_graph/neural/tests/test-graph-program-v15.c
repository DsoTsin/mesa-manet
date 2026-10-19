#include "pan_graph_program_v15.h"

#include <assert.h>
#include <string.h>

static uint32_t
word(const struct pan_graph_v15_program *program, unsigned offset)
{
   assert(offset + 4 <= program->size);
   const uint8_t *p = (const uint8_t *)program->data + offset;
   return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
          (uint32_t)p[3] << 24;
}

int
main(void)
{
   const struct pan_graph_matmul_info info = {
      .batch = 2,
      .rows = 3,
      .inner = 5,
      .columns = 4,
   };
   const struct pan_graph_matmul_add add = {{0, 0}};
   const struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   struct pan_graph_program_cost_plan plan = {0};
   uint64_t selected = 7;
   assert(pan_graph_program_cost_plan_matmul_add_v15(
             &info, &add, 1, &selected, &plan) == PAN_GRAPH_TSU_SUCCESS);
   struct pan_graph_program_selection selection = {
      .transfer = {4, 1, 1, 3, 5, 1, 1},
      .loops = {2, 0, 1, 0},
   };
   struct pan_graph_v15_memory reads[2] = {
      {.offset = 2, .level = 5, .override = 1, .value = 1},
      {.level = 5, .override = 1, .value = 1},
   };
   struct pan_graph_v15_memory writes[2] = {
      {.domain = 3, .level = 5},
      {.domain = 3, .level = 5},
   };
   struct pan_graph_program_bindings bindings = {
      .reads = reads,
      .writes = writes,
      .read_count = 2,
      .write_count = 2,
   };
   struct pan_graph_program_kernel kernel;
   assert(pan_graph_program_v15_compile_candidate(
             &plan, &selection, &add, 1, &bindings, &target, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(kernel.program.size == 276);
   assert(word(&kernel.program, 0) == 0x30021488);
   assert(word(&kernel.program, 40) == 2 && word(&kernel.program, 48) == 0);
   assert(word(&kernel.program, 52) == 0xf005b204);
   assert(word(&kernel.program, 60) == 0xf015b204);
   pan_graph_v15_program_cleanup(&kernel.program);
   struct pan_graph_program_kernel before;
   memset(&before, 0xa5, sizeof(before));
   kernel = before;
   writes[1].offset = 2;
   assert(pan_graph_program_v15_compile_candidate(
             &plan, &selection, &add, 1, &bindings, &target, &kernel) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&kernel, &before, sizeof(kernel)));
   writes[1].offset = 0;
   bindings.read_count = 1;
   assert(pan_graph_program_v15_compile_candidate(
             &plan, &selection, &add, 1, &bindings, &target, &kernel) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&kernel, &before, sizeof(kernel)));
   pan_graph_program_cost_plan_finish(&plan);
   selected = 3;
   assert(pan_graph_program_cost_plan_matmul_add_v15(
             &info, &add, 1, &selected, &plan) == PAN_GRAPH_TSU_SUCCESS);
   writes[0] = (struct pan_graph_v15_memory){
      .domain = 1,
      .level = 5,
      .override = 1,
      .value = 1,
      .extra = 1,
   };
   bindings.read_count = 2;
   bindings.write_count = 1;
   assert(pan_graph_program_v15_compile_candidate(
             &plan, &selection, &add, 1, &bindings, &target, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(kernel.program.size == 224);
   assert(word(&kernel.program, 52) == 0xf3045204);
   pan_graph_v15_program_cleanup(&kernel.program);
   pan_graph_program_cost_plan_finish(&plan);
   selected = 4;
   assert(pan_graph_program_cost_plan_matmul_add_v15(
             &info, &add, 1, &selected, &plan) == PAN_GRAPH_TSU_SUCCESS);
   selection = (struct pan_graph_program_selection){.transfer = {2, 3, 4}};
   reads[0] = (struct pan_graph_v15_memory){
      .domain = 1,
      .level = 5,
      .override = 1,
      .value = 1,
   };
   writes[0] = (struct pan_graph_v15_memory){.domain = 3, .level = 5};
   bindings.read_count = 1;
   assert(pan_graph_program_v15_compile_candidate(
             &plan, &selection, &add, 1, &bindings, &target, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(kernel.program.size == 112 && kernel.extent[0] == 1 &&
          kernel.extent[1] == 2 && kernel.extent[2] == 3 &&
          kernel.extent[3] == 4);
   pan_graph_v15_program_cleanup(&kernel.program);
   kernel = before;
   const struct pan_graph_matmul_add invalid = {{1, 0}};
   assert(pan_graph_program_v15_compile_candidate(
             &plan, &selection, &invalid, 1, &bindings, &target, &kernel) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&kernel, &before, sizeof(kernel)));
   pan_graph_program_cost_plan_finish(&plan);
   return 0;
}
