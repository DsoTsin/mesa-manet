#include "pan_graph_vector_v15.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#ifdef TEST_VECTOR_ALLOC_FAILURES
static size_t allocation_count, allocation_failure;
void *__real_malloc(size_t);
void *__real_realloc(void *, size_t);
void *__wrap_malloc(size_t size)
{
   return ++allocation_count == allocation_failure ? NULL : __real_malloc(size);
}
void *__wrap_realloc(void *pointer, size_t size)
{
   return ++allocation_count == allocation_failure ? NULL :
          __real_realloc(pointer, size);
}
#endif

static uint32_t
word(const struct pan_graph_vector_kernel *kernel, size_t offset)
{
   assert(offset + 4 <= kernel->program.size);
   const uint8_t *p = (const uint8_t *)kernel->program.data + offset;
   return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
          (uint32_t)p[3] << 24;
}

static void
reject(const struct pan_graph_vector_program *info,
       const struct pan_graph_nrt_target *target,
       enum pan_graph_tsu_status status)
{
   struct pan_graph_vector_kernel kernel;
   memset(&kernel, 0xa5, sizeof(kernel));
   struct pan_graph_vector_kernel before = kernel;
   assert(pan_graph_vector_v15_compile_candidate(info, target, &kernel) == status);
   assert(!memcmp(&kernel, &before, sizeof(kernel)));
}

int
main(void)
{
   struct pan_graph_v15_memory reads[2] = {
      {.domain = 1, .level = 5, .override = 1, .value = 1},
      {.domain = 1, .offset = 2, .level = 5, .override = 1, .value = 1},
   };
   struct pan_graph_vector_add adds[7] = {
      {{0, 0}}, {{0, 0}}, {{1, 2}},
   };
   struct pan_graph_vector_write writes[7] = {
      {.value = 3, .memory = {.domain = 3, .level = 5}},
   };
   struct pan_graph_vector_program info = {
      .full = {2, 3, 4}, .transfer = {2, 3, 4},
      .reads = reads, .adds = adds, .writes = writes,
      .read_count = 1, .add_count = 3, .write_count = 1,
   };
   struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   struct pan_graph_vector_kernel kernel = {0};
   assert(pan_graph_vector_v15_compile_candidate(&info, &target, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(kernel.program.size == 200 && word(&kernel, 0) == 0x20030044);
   assert(word(&kernel, 52) == 0xf2002004);
   assert(word(&kernel, 84) == 0xf2101004);
   assert(word(&kernel, 116) == 0xf2213004);
   assert(word(&kernel, 160) == 0x03001414);
   assert(word(&kernel, 172) == 0x03001814);
   assert(word(&kernel, 196) == 0x63636362);
   assert(!memcmp(kernel.extent, (uint32_t[]){1, 2, 3, 4}, sizeof(kernel.extent)));
   assert(!memcmp(kernel.loops, (uint32_t[6]){0}, sizeof(kernel.loops)));
   pan_graph_v15_program_cleanup(&kernel.program);

   writes[1] = writes[0];
   info.write_count = 2;
   assert(pan_graph_vector_v15_compile_candidate(&info, &target, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(kernel.program.size == 208);
   assert(word(&kernel, 44) == word(&kernel, 52) - (1u << 20));
   pan_graph_v15_program_cleanup(&kernel.program);
   writes[1].memory.offset = 2;
   reject(&info, &target, PAN_GRAPH_TSU_UNSUPPORTED);
   info.write_count = 1;

   for (unsigned axis = 0; axis < 3; axis++) {
      info.full[axis] = 8193;
      info.loops[0] = axis + 1;
      info.loops[2] = info.transfer[axis];
      assert(pan_graph_vector_v15_compile_candidate(&info, &target, &kernel) ==
             PAN_GRAPH_TSU_SUCCESS);
      assert(kernel.extent[axis + 1] == 8193);
      assert(kernel.loops[0] == axis + 1 &&
             kernel.loops[2] == info.transfer[axis]);
      pan_graph_v15_program_cleanup(&kernel.program);
      info.full[axis] = info.transfer[axis];
   }
   memset(info.loops, 0, sizeof(info.loops));
   for (unsigned i = 0; i < 18; i++) {
      struct pan_graph_vector_program bad = info;
      struct pan_graph_nrt_target bad_target = target;
      enum pan_graph_tsu_status status = PAN_GRAPH_TSU_INVALID;
      switch (i) {
      case 0: bad.reads = NULL; break;
      case 1: bad.adds = NULL; break;
      case 2: bad.writes = NULL; break;
      case 3: bad.read_count = 0; break;
      case 4: bad.read_count = 8; status = PAN_GRAPH_TSU_UNSUPPORTED; break;
      case 5: bad.add_count = 8; status = PAN_GRAPH_TSU_UNSUPPORTED; break;
      case 6: bad.write_count = 8; status = PAN_GRAPH_TSU_UNSUPPORTED; break;
      case 7: bad.full[1] = 0; break;
      case 8: bad.transfer[0] = 0; break;
      case 9: bad.transfer[0] = 3; break;
      case 10: bad.full[0] = bad.transfer[0] = 4096;
               status = PAN_GRAPH_TSU_UNSUPPORTED; break;
      case 11: bad_target.banks = 4; status = PAN_GRAPH_TSU_UNSUPPORTED; break;
      case 12: bad.loops[0] = 4; bad.loops[2] = 1; break;
      case 13: bad.loops[3] = 1; break;
      case 14: adds[0].input[0] = 1; break;
      case 15: writes[0].value = 0; break;
      case 16: reads[0].domain_bias = 1; break;
      case 17: reads[0].level = 1; break;
      }
      reject(&bad, &bad_target, status);
      adds[0].input[0] = 0;
      writes[0].value = 3;
      reads[0].domain_bias = 0;
      reads[0].level = 5;
   }
   reject(NULL, &target, PAN_GRAPH_TSU_INVALID);
   reject(&info, NULL, PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_vector_v15_compile_candidate(&info, &target, NULL) ==
          PAN_GRAPH_TSU_INVALID);
   info.read_count = 2;
   adds[0] = (struct pan_graph_vector_add){{0, 0}};
   adds[1] = (struct pan_graph_vector_add){{2, 2}};
   adds[2] = (struct pan_graph_vector_add){{3, 3}};
   writes[0].value = 4;
   reject(&info, &target, PAN_GRAPH_TSU_INVALID);
   info.read_count = 1;
   adds[1] = (struct pan_graph_vector_add){{0, 0}};
   adds[2] = (struct pan_graph_vector_add){{1, 1}};
   writes[0].value = 3;
   reject(&info, &target, PAN_GRAPH_TSU_INVALID);
   adds[2] = (struct pan_graph_vector_add){{1, 2}};
   info.full[0] = info.transfer[0] = 4095;
   info.full[1] = info.transfer[1] = 4095;
   info.full[2] = info.transfer[2] = 4095;
   reject(&info, &target, PAN_GRAPH_TSU_UNSUPPORTED);
   memcpy(info.full, (uint32_t[]){2, 3, 4}, sizeof(info.full));
   memcpy(info.transfer, info.full, sizeof(info.full));

   info.add_count = info.write_count = 7;
   for (unsigned i = 0; i < 7; i++) {
      adds[i] = (struct pan_graph_vector_add){{i, i}};
      writes[i] = (struct pan_graph_vector_write){
         .value = 7, .memory = {.domain = 3, .level = 5},
      };
   }
   assert(pan_graph_vector_v15_compile_candidate(&info, &target, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(kernel.program.size == 424);
   pan_graph_v15_program_cleanup(&kernel.program);

#ifdef TEST_VECTOR_ALLOC_FAILURES
   allocation_count = 0;
   assert(pan_graph_vector_v15_compile_candidate(&info, &target, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   size_t total = allocation_count;
   assert(total >= 6);
   pan_graph_v15_program_cleanup(&kernel.program);
   for (size_t failure = 1; failure <= total; failure++) {
      allocation_count = 0;
      allocation_failure = failure;
      reject(&info, &target, PAN_GRAPH_TSU_OUT_OF_MEMORY);
   }
   allocation_failure = 0;
   printf("allocation_failures=%zu\n", total);
#endif
   return 0;
}
