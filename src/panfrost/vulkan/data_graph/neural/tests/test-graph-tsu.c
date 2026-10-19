#include "pan_graph_tsu.h"
#include "pan_graph_v15.h"

#include <assert.h>
#include <limits.h>
#include <string.h>

static void
test_identity(void)
{
   const int64_t low[] = {0, 0}, high[] = {15, 31};
   const struct pan_graph_tsu_info info = {.rank = 2, .low = low, .high = high};
   struct pan_graph_tsu_program program = {0};
   assert(pan_graph_tsu_compile(&info, &program) == PAN_GRAPH_TSU_SUCCESS);
   assert(program.count == 1 && program.instructions[0].opcode == 21);
   pan_graph_tsu_cleanup(&program);
   assert(!program.count && !program.instructions);
}

static void
test_permutation(void)
{
   const int64_t low[] = {0, 0}, high[] = {15, 31};
   const int64_t matrix[] = {0, 1, 0, 1, 0, 0, 0, 0, 1};
   struct pan_graph_tsu_transform transform = {.kind = PAN_GRAPH_TSU_AFFINE,
      .rows = 3, .columns = 3, .low = matrix, .high = matrix};
   struct pan_graph_tsu_info info = {.rank = 2, .low = low, .high = high,
      .count = 1, .transforms = &transform};
   struct pan_graph_tsu_program program = {0};
   assert(pan_graph_tsu_compile(&info, &program) == PAN_GRAPH_TSU_SUCCESS);
   assert(program.count == 2);
   assert(program.instructions[0].opcode == 1 && program.instructions[0].reg == 1);
   assert(program.instructions[1].opcode == 21);
   pan_graph_tsu_cleanup(&program);
}

static void
test_invalid(void)
{
   int64_t low[] = {0, 0}, high[] = {15, 31};
   int64_t matrix[] = {1, 1, 0, 1, 1, 0, 0, 0, 1};
   struct pan_graph_tsu_transform transform = {.kind = PAN_GRAPH_TSU_AFFINE,
      .rows = 3, .columns = 3, .low = matrix, .high = matrix};
   struct pan_graph_tsu_info info = {.rank = 2, .low = low, .high = high,
      .count = 1, .transforms = &transform};
   struct pan_graph_tsu_program program = {.count = SIZE_MAX};
   assert(pan_graph_tsu_compile(&info, &program) == PAN_GRAPH_TSU_INVALID);
   assert(program.count == SIZE_MAX && !program.instructions);
   matrix[0] = -1;
   assert(pan_graph_tsu_compile(&info, &program) == PAN_GRAPH_TSU_INVALID);
   transform = (struct pan_graph_tsu_transform){.kind = PAN_GRAPH_TSU_SPLIT};
   assert(pan_graph_tsu_compile(&info, &program) == PAN_GRAPH_TSU_INVALID);
   transform = (struct pan_graph_tsu_transform){.kind = PAN_GRAPH_TSU_NEGATE};
   low[0] = INT64_MIN;
   assert(pan_graph_tsu_compile(&info, &program) == PAN_GRAPH_TSU_INVALID);
   info.count = 0;
   info.rank = PAN_GRAPH_TSU_MAX_RANK + 1;
   assert(pan_graph_tsu_compile(&info, &program) == PAN_GRAPH_TSU_INVALID);
}

static void
test_link(void)
{
   const int64_t low[] = {0, 0}, high[] = {15, 31};
   const int64_t matrix[] = {0, 1, 0, 1, 0, 0, 0, 0, 1};
   const struct pan_graph_tsu_transform transform = {.kind = PAN_GRAPH_TSU_AFFINE,
      .rows = 3, .columns = 3, .low = matrix, .high = matrix};
   const struct pan_graph_tsu_info infos[] = {
      {.rank = 2, .low = low, .high = high, .count = 1, .transforms = &transform},
      {.rank = 2, .low = low, .high = high},
      {.rank = 2, .low = low, .high = high, .count = 1, .transforms = &transform},
      {.rank = 2, .low = low, .high = high},
   };
   uint32_t offsets[4];
   struct pan_graph_tsu_program program = {0};
   assert(pan_graph_tsu_link(infos, 4, &program, offsets) == PAN_GRAPH_TSU_SUCCESS);
   assert(program.count == 8);
   assert(offsets[0] == 0 && offsets[1] == 4 && offsets[2] == 0 && offsets[3] == 1);
   pan_graph_tsu_cleanup(&program);
   uint8_t bytes[8];
   memset(bytes, 0xa5, sizeof(bytes));
   size_t written;
   assert(pan_graph_v15_compile_tsu(bytes, 3, &infos[0], &written) == PAN_GRAPH_TSU_INVALID);
   assert(!written && bytes[0] == 0xa5);
   assert(pan_graph_v15_compile_tsu(bytes, sizeof(bytes), &infos[0], &written) == PAN_GRAPH_TSU_SUCCESS);
   const uint8_t expected[] = {0x85, 0x63, 0x63, 0x63};
   assert(written == sizeof(expected) && !memcmp(bytes, expected, sizeof(expected)));
   assert(bytes[4] == 0xa5);
}

static void
test_program_failure(void)
{
   const int64_t low[] = {0}, high[] = {31};
   const struct pan_graph_tsu_info tsu = {.rank = 1, .low = low, .high = high};
   struct pan_graph_v15_section section = {.type = PAN_GRAPH_V15_MEMORY_INPUT,
      .memory = {.level = 1}};
   uint32_t index = 0;
   struct pan_graph_v15_program_info info = {
      .tsu_count = 1, .tsu = &tsu, .section_count = 1,
      .sections = &section, .section_tsu = &index,
   };
   struct pan_graph_v15_program program = {.size = SIZE_MAX};
   assert(pan_graph_v15_compile(&info, &program) == PAN_GRAPH_TSU_INVALID);
   assert(program.size == SIZE_MAX && !program.data);
   section.memory.level = 5;
   index = 1;
   assert(pan_graph_v15_compile(&info, &program) == PAN_GRAPH_TSU_INVALID);
   assert(program.size == SIZE_MAX && !program.data);
   index = 0;
   info.resource_count = 1;
   assert(pan_graph_v15_compile(&info, &program) == PAN_GRAPH_TSU_INVALID);
   info.resource_count = 0;
   assert(pan_graph_v15_compile(&info, &program) == PAN_GRAPH_TSU_SUCCESS);
   assert(program.size == 48 && program.data);
   pan_graph_v15_program_cleanup(&program);
}

static void
test_encoding_limits(void)
{
   int64_t low[PAN_GRAPH_TSU_MAX_RANK] = {0};
   int64_t high[PAN_GRAPH_TSU_MAX_RANK] = {0};
   struct pan_graph_tsu_transform transforms[130];
   for (unsigned i = 0; i < 130; i++)
      transforms[i] = (struct pan_graph_tsu_transform){
         .kind = PAN_GRAPH_TSU_CLAMP, .value = 31};
   struct pan_graph_tsu_info info = {.rank = 1, .low = low, .high = high,
      .count = 130, .transforms = transforms};
   uint8_t bytes[256];
   memset(bytes, 0xa5, sizeof(bytes));
   size_t written;
   assert(pan_graph_v15_compile_tsu(bytes, sizeof(bytes), &info, &written) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!written);
   for (unsigned i = 0; i < sizeof(bytes); i++)
      assert(bytes[i] == 0xa5);
   info.rank = 10;
   info.count = 1;
   transforms[0] = (struct pan_graph_tsu_transform){
      .kind = PAN_GRAPH_TSU_NEGATE, .index = 9};
   assert(pan_graph_v15_compile_tsu(bytes, sizeof(bytes), &info, &written) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!written && bytes[0] == 0xa5);
}

int
main(void)
{
   test_identity();
   test_permutation();
   test_invalid();
   test_link();
   test_program_failure();
   test_encoding_limits();
   return 0;
}
