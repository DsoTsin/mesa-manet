#include "pan_graph_v15.h"

#include <assert.h>
#include <stdlib.h>

static unsigned attempt, fail_at, live;
static void *allocations[1024];

void *__real_malloc(size_t size);
void *__real_realloc(void *ptr, size_t size);
void __real_free(void *ptr);
void *__wrap_malloc(size_t size);
void *__wrap_realloc(void *ptr, size_t size);
void __wrap_free(void *ptr);

static unsigned
allocation_index(void *ptr)
{
   for (unsigned i = 0; i < live; i++) {
      if (allocations[i] == ptr)
         return i;
   }
   assert(!ptr);
   return live;
}

void *
__wrap_malloc(size_t size)
{
   if (++attempt == fail_at)
      return NULL;
   void *ptr = __real_malloc(size);
   if (ptr) {
      assert(live < 1024);
      allocations[live++] = ptr;
   }
   return ptr;
}

void *
__wrap_realloc(void *old, size_t size)
{
   if (++attempt == fail_at)
      return NULL;
   unsigned index = allocation_index(old);
   void *ptr = __real_realloc(old, size);
   if (ptr) {
      if (!old) {
         assert(live < 1024);
         live++;
      }
      allocations[index] = ptr;
   }
   return ptr;
}

void
__wrap_free(void *ptr)
{
   if (ptr) {
      unsigned index = allocation_index(ptr);
      allocations[index] = allocations[--live];
   }
   __real_free(ptr);
}

int
main(void)
{
   const int64_t low[] = {0, 0}, high[] = {15, 31};
   const int64_t matrix[] = {0, 1, 0, 1, 0, 0, 0, 0, 1};
   const struct pan_graph_tsu_transform transform = {.kind = PAN_GRAPH_TSU_AFFINE,
      .rows = 3, .columns = 3, .low = matrix, .high = matrix};
   const struct pan_graph_tsu_info tsu[] = {
      {.rank = 2, .low = low, .high = high, .count = 1, .transforms = &transform},
      {.rank = 2, .low = low, .high = high},
   };
   const struct pan_graph_v15_program_info info = {.tsu_count = 2, .tsu = tsu};
   unsigned total = 0;
   for (unsigned failure = 0; failure <= total; failure++) {
      attempt = 0;
      fail_at = failure;
      struct pan_graph_v15_program program = {.size = SIZE_MAX};
      enum pan_graph_tsu_status status = pan_graph_v15_compile(&info, &program);
      if (!failure) {
         assert(status == PAN_GRAPH_TSU_SUCCESS);
         total = attempt;
         pan_graph_v15_program_cleanup(&program);
      } else {
         assert(status == PAN_GRAPH_TSU_OUT_OF_MEMORY);
         assert(program.size == SIZE_MAX && !program.data);
      }
      assert(!live);
   }
   return 0;
}
