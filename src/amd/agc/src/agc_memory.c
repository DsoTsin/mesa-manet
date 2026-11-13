#include "agc_memory.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c11/threads.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/u_atomic.h"

#include "agc_debug.h"

#ifdef _WIN32
#include <windows.h>
#endif

#define AGC_MEMORY_POOL_COUNT 4
#define AGC_MEMORY_POOL_FLAGS (AGC_BO_GL2_BYPASS | AGC_BO_EXECUTE)
#define AGC_MEMORY_QUERY_ATTEMPTS 8
#define AGC_MEMORY_HOST_WRITE_WAIT_NS (INT64_C(1000) * 1000000)
#define AGC_MEMORY_HOST_WRITE_POLL_US 1000
#define AGC_WRITE_BLOCK_BYTES (AGC_WRITE_PAGE_BYTES * AGC_WRITE_BLOCK_PAGES)

_Static_assert(AGC_MEMORY_POOL_FLAGS == AGC_MEMORY_POOL_COUNT - 1, "one pool per flag combination");
_Static_assert(AGC_SUBALLOC_SLAB_BYTES % AGC_DRIVER_BLOCK_ALIGNMENT == 0, "suballocation slab");
_Static_assert(AGC_MIRROR_SLAB_BYTES % AGC_DRIVER_BLOCK_ALIGNMENT == 0, "mirror slab");

static _Thread_local char agc_memory_error_text[256];

static bool agc_memory_fail(char *error, size_t error_bytes, const char *format, ...) AGC_PRINTF(3, 4);

static bool
agc_memory_fail(char *error, size_t error_bytes, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   vsnprintf(agc_memory_error_text, sizeof(agc_memory_error_text), format, args);
   va_end(args);
   if (error && error_bytes)
      snprintf(error, error_bytes, "%s", agc_memory_error_text);
   return false;
}

static void
agc_memory_copy_error(char *error, size_t error_bytes)
{
   if (error && error_bytes)
      snprintf(error, error_bytes, "%s", agc_memory_error_text);
}

const char *
agc_memory_last_error(void)
{
   return agc_memory_error_text;
}

static uint64_t
agc_align_down(uint64_t value, uint64_t alignment)
{
   return value & ~(alignment - 1);
}

static uint64_t
agc_align_up(uint64_t value, uint64_t alignment)
{
   uint64_t rounded = (value + (alignment - 1)) & ~(alignment - 1);
   return rounded < value ? ~(alignment - 1) : rounded;
}

static uint64_t
agc_range_end(uint64_t address, uint64_t bytes)
{
   return bytes > UINT64_MAX - address ? UINT64_MAX : address + bytes;
}

static bool
agc_span_list_reserve(struct agc_span_list *list, uint32_t count)
{
   if (count <= list->capacity)
      return true;
   uint32_t capacity = list->capacity ? list->capacity : 16;
   while (capacity < count) {
      if (capacity > UINT32_MAX / 2)
         return false;
      capacity *= 2;
   }
   struct agc_span *spans = realloc(list->spans, (size_t)capacity * sizeof(*spans));
   if (!spans)
      return false;
   list->spans = spans;
   list->capacity = capacity;
   return true;
}

static bool
agc_span_list_insert(struct agc_span_list *list, uint32_t index, uint64_t start, uint64_t end)
{
   if (list->count == UINT32_MAX || !agc_span_list_reserve(list, list->count + 1))
      return false;
   memmove(&list->spans[index + 1], &list->spans[index],
           (size_t)(list->count - index) * sizeof(list->spans[0]));
   list->spans[index] = (struct agc_span){start, end};
   list->count++;
   return true;
}

static void
agc_span_list_remove(struct agc_span_list *list, uint32_t index)
{
   memmove(&list->spans[index], &list->spans[index + 1],
           (size_t)(list->count - index - 1) * sizeof(list->spans[0]));
   list->count--;
}

void
agc_span_list_fini(struct agc_span_list *list)
{
   free(list->spans);
   memset(list, 0, sizeof(*list));
}

bool
agc_span_list_append(struct agc_span_list *list, uint64_t start, uint64_t end)
{
   if (start >= end)
      return true;
   if (list->count == UINT32_MAX || !agc_span_list_reserve(list, list->count + 1))
      return false;
   list->spans[list->count++] = (struct agc_span){start, end};
   return true;
}

bool
agc_span_list_extend(struct agc_span_list *list, uint64_t start, uint64_t end)
{
   if (start >= end)
      return true;
   if (list->count && start <= list->spans[list->count - 1].end) {
      struct agc_span *last = &list->spans[list->count - 1];
      last->start = MIN2(last->start, start);
      last->end = MAX2(last->end, end);
      return true;
   }
   return agc_span_list_append(list, start, end);
}

bool
agc_span_list_subtract(const struct agc_span_list *from, const struct agc_span_list *cut,
                       struct agc_span_list *out)
{
   out->count = 0;
   uint32_t skipped = 0;
   for (uint32_t i = 0; i < from->count; i++) {
      uint64_t cursor = from->spans[i].start;
      const uint64_t end = from->spans[i].end;
      while (skipped < cut->count && cut->spans[skipped].end <= cursor)
         skipped++;
      for (uint32_t k = skipped; cursor < end; k++) {
         if (k == cut->count || cut->spans[k].start >= end) {
            if (!agc_span_list_append(out, cursor, end))
               return false;
            break;
         }
         if (!agc_span_list_append(out, cursor, cut->spans[k].start))
            return false;
         cursor = MAX2(cursor, cut->spans[k].end);
      }
   }
   return true;
}

bool
agc_span_list_union(const struct agc_span_list *a, const struct agc_span_list *b,
                    struct agc_span_list *out)
{
   out->count = 0;
   uint32_t i = 0;
   uint32_t j = 0;
   while (i < a->count || j < b->count) {
      const struct agc_span *next;
      if (j == b->count || (i < a->count && a->spans[i].start <= b->spans[j].start))
         next = &a->spans[i++];
      else
         next = &b->spans[j++];
      if (!agc_span_list_extend(out, next->start, next->end))
         return false;
   }
   return true;
}

bool
agc_span_list_clip(const struct agc_span_list *list, uint64_t start, uint64_t end,
                   struct agc_span_list *out)
{
   out->count = 0;
   for (uint32_t i = 0; i < list->count && list->spans[i].start < end; i++) {
      if (!agc_span_list_append(out, MAX2(list->spans[i].start, start), MIN2(list->spans[i].end, end)))
         return false;
   }
   return true;
}

uint64_t
agc_span_list_bytes(const struct agc_span_list *list)
{
   uint64_t bytes = 0;
   for (uint32_t i = 0; i < list->count; i++)
      bytes += list->spans[i].end - list->spans[i].start;
   return bytes;
}

void
agc_range_map_fini(struct agc_range_map *map)
{
   free(map->ranges);
   memset(map, 0, sizeof(*map));
}

uint32_t
agc_range_map_lower_bound(const struct agc_range_map *map, uint64_t address)
{
   uint32_t lo = 0;
   uint32_t hi = map->count;
   while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      if (map->ranges[mid].end <= address)
         lo = mid + 1;
      else
         hi = mid;
   }
   return lo;
}

struct agc_range *
agc_range_map_find(const struct agc_range_map *map, uint64_t address)
{
   uint32_t index = agc_range_map_lower_bound(map, address);
   if (index < map->count && map->ranges[index].start <= address)
      return &map->ranges[index];
   return NULL;
}

bool
agc_range_map_overlaps(const struct agc_range_map *map, uint64_t start, uint64_t end)
{
   if (start >= end)
      return false;
   uint32_t index = agc_range_map_lower_bound(map, start);
   return index < map->count && map->ranges[index].start < end;
}

bool
agc_range_map_covers(const struct agc_range_map *map, uint64_t start, uint64_t end)
{
   if (start >= end)
      return false;
   uint64_t cursor = start;
   for (uint32_t index = agc_range_map_lower_bound(map, start); index < map->count; index++) {
      if (map->ranges[index].start > cursor)
         return false;
      cursor = map->ranges[index].end;
      if (cursor >= end)
         return true;
   }
   return false;
}

bool
agc_range_map_insert(struct agc_range_map *map, uint64_t start, uint64_t end, void *data)
{
   if (start >= end || map->count == UINT32_MAX)
      return false;
   uint32_t index = agc_range_map_lower_bound(map, start);
   if (index < map->count && map->ranges[index].start < end)
      return false;
   if (map->count == map->capacity) {
      if (map->capacity > UINT32_MAX / 2)
         return false;
      uint32_t capacity = map->capacity ? map->capacity * 2 : 64;
      struct agc_range *ranges = realloc(map->ranges, (size_t)capacity * sizeof(*ranges));
      if (!ranges)
         return false;
      map->ranges = ranges;
      map->capacity = capacity;
   }
   memmove(&map->ranges[index + 1], &map->ranges[index],
           (size_t)(map->count - index) * sizeof(map->ranges[0]));
   map->ranges[index] = (struct agc_range){start, end, data};
   map->count++;
   return true;
}

bool
agc_range_map_remove(struct agc_range_map *map, uint64_t start, void **data)
{
   uint32_t index = agc_range_map_lower_bound(map, start);
   if (index == map->count || map->ranges[index].start != start)
      return false;
   if (data)
      *data = map->ranges[index].data;
   memmove(&map->ranges[index], &map->ranges[index + 1],
           (size_t)(map->count - index - 1) * sizeof(map->ranges[0]));
   map->count--;
   return true;
}

bool
agc_range_map_gaps(const struct agc_range_map *map, uint64_t start, uint64_t end,
                   struct agc_span_list *gaps)
{
   uint64_t cursor = start;
   for (uint32_t index = agc_range_map_lower_bound(map, start); cursor < end; index++) {
      if (index == map->count || map->ranges[index].start >= end)
         return agc_span_list_append(gaps, cursor, end);
      if (!agc_span_list_append(gaps, cursor, map->ranges[index].start))
         return false;
      cursor = MAX2(cursor, map->ranges[index].end);
   }
   return true;
}

bool
agc_import_plan(const struct agc_guest_range *visible, size_t count, struct agc_span_list *wanted)
{
   wanted->count = 0;
   uint64_t previous_end = 0;
   size_t i = 0;
   while (i < count) {
      if (visible[i].bytes == 0) {
         i++;
         continue;
      }
      const size_t first = i;
      const uint64_t run_start = visible[i].address;
      uint64_t run_end = agc_range_end(run_start, visible[i].bytes);
      if (run_start < previous_end)
         return false;
      for (i++; i < count && visible[i].address == run_end; i++)
         run_end = agc_range_end(run_end, visible[i].bytes);
      previous_end = run_end;

      const uint64_t start = agc_align_up(MAX2(run_start, AGC_GUEST_WINDOW_BASE), AGC_GUEST_PAGE_BYTES);
      const uint64_t end = agc_align_down(MIN2(run_end, AGC_GUEST_WINDOW_END), AGC_GUEST_PAGE_BYTES);
      size_t piece = first;
      for (uint64_t cursor = start; cursor < end;) {
         uint64_t stop = MIN2(end, (cursor | (AGC_IMPORT_WINDOW_BYTES - 1)) + 1);
         for (; piece + 1 < i; piece++) {
            uint64_t boundary = agc_range_end(visible[piece].address, visible[piece].bytes);
            if (boundary > cursor && boundary % AGC_GUEST_PAGE_BYTES == 0) {
               stop = MIN2(stop, boundary);
               break;
            }
         }
         if (!agc_span_list_append(wanted, cursor, stop))
            return false;
         cursor = stop;
      }
   }
   return true;
}

bool
agc_import_diff(const struct agc_span_list *wanted, const struct agc_range_map *imports,
                struct agc_span_list *missing, struct agc_span_list *stale)
{
   missing->count = 0;
   stale->count = 0;
   for (uint32_t i = 0; i < wanted->count; i++) {
      if (!agc_range_map_gaps(imports, wanted->spans[i].start, wanted->spans[i].end, missing))
         return false;
   }
   uint32_t skipped = 0;
   for (uint32_t i = 0; i < imports->count; i++) {
      uint64_t cursor = imports->ranges[i].start;
      const uint64_t end = imports->ranges[i].end;
      while (skipped < wanted->count && wanted->spans[skipped].end <= cursor)
         skipped++;
      for (uint32_t k = skipped; cursor < end && k < wanted->count && wanted->spans[k].start <= cursor; k++)
         cursor = wanted->spans[k].end;
      if (cursor < end && !agc_span_list_append(stale, imports->ranges[i].start, end))
         return false;
   }
   return true;
}

bool
agc_heap_init(struct agc_heap *heap, uint64_t start, uint64_t bytes)
{
   memset(heap, 0, sizeof(*heap));
   heap->start = start;
   heap->end = agc_range_end(start, bytes);
   heap->free_bytes = heap->end - start;
   return agc_span_list_append(&heap->holes, start, heap->end);
}

void
agc_heap_fini(struct agc_heap *heap)
{
   agc_span_list_fini(&heap->holes);
   memset(heap, 0, sizeof(*heap));
}

bool
agc_heap_alloc(struct agc_heap *heap, uint64_t bytes, uint64_t alignment, uint64_t *address)
{
   if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)))
      return false;
   for (uint32_t i = 0; i < heap->holes.count; i++) {
      const struct agc_span hole = heap->holes.spans[i];
      uint64_t aligned = agc_align_up(hole.start, alignment);
      if (aligned < hole.start || aligned >= hole.end || hole.end - aligned < bytes)
         continue;
      uint64_t tail = aligned + bytes;
      if (aligned != hole.start && tail != hole.end) {
         if (!agc_span_list_insert(&heap->holes, i + 1, tail, hole.end))
            return false;
         heap->holes.spans[i].end = aligned;
      } else if (aligned != hole.start) {
         heap->holes.spans[i].end = aligned;
      } else if (tail != hole.end) {
         heap->holes.spans[i].start = tail;
      } else {
         agc_span_list_remove(&heap->holes, i);
      }
      heap->free_bytes -= bytes;
      *address = aligned;
      return true;
   }
   return false;
}

bool
agc_heap_alloc_at(struct agc_heap *heap, uint64_t address, uint64_t bytes)
{
   if (bytes == 0 || bytes > UINT64_MAX - address)
      return false;
   const uint64_t end = address + bytes;
   struct agc_span *holes = heap->holes.spans;
   uint32_t lo = 0;
   uint32_t hi = heap->holes.count;
   while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      if (holes[mid].end <= address)
         lo = mid + 1;
      else
         hi = mid;
   }
   if (lo == heap->holes.count || holes[lo].start > address || holes[lo].end < end)
      return false;
   const struct agc_span hole = holes[lo];
   if (hole.start != address && hole.end != end) {
      if (!agc_span_list_insert(&heap->holes, lo + 1, end, hole.end))
         return false;
      heap->holes.spans[lo].end = address;
   } else if (hole.start != address) {
      holes[lo].end = address;
   } else if (hole.end != end) {
      holes[lo].start = end;
   } else {
      agc_span_list_remove(&heap->holes, lo);
   }
   heap->free_bytes -= bytes;
   return true;
}

bool
agc_heap_free(struct agc_heap *heap, uint64_t address, uint64_t bytes)
{
   const uint64_t end = agc_range_end(address, bytes);
   if (bytes == 0 || address < heap->start || end > heap->end)
      return false;
   struct agc_span *holes = heap->holes.spans;
   uint32_t lo = 0;
   uint32_t hi = heap->holes.count;
   while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      if (holes[mid].start < address)
         lo = mid + 1;
      else
         hi = mid;
   }
   if (lo > 0 && holes[lo - 1].end > address)
      return false;
   if (lo < heap->holes.count && holes[lo].start < end)
      return false;
   bool joins_previous = lo > 0 && holes[lo - 1].end == address;
   bool joins_next = lo < heap->holes.count && holes[lo].start == end;
   if (joins_previous && joins_next) {
      holes[lo - 1].end = holes[lo].end;
      agc_span_list_remove(&heap->holes, lo);
   } else if (joins_previous) {
      holes[lo - 1].end = end;
   } else if (joins_next) {
      holes[lo].start = address;
   } else if (!agc_span_list_insert(&heap->holes, lo, address, end)) {
      return false;
   }
   heap->free_bytes += bytes;
   return true;
}

bool
agc_heap_unused(const struct agc_heap *heap)
{
   return heap->free_bytes == heap->end - heap->start;
}

void
agc_subpool_init(struct agc_subpool *pool, uint64_t slab_bytes, uint64_t slab_alignment)
{
   memset(pool, 0, sizeof(*pool));
   pool->slab_bytes = slab_bytes;
   pool->slab_alignment = slab_alignment;
}

void
agc_subpool_fini(struct agc_subpool *pool)
{
   for (uint32_t i = 0; i < pool->count; i++) {
      if (!pool->slabs[i])
         continue;
      agc_heap_fini(&pool->slabs[i]->heap);
      free(pool->slabs[i]);
   }
   free(pool->slabs);
   memset(pool, 0, sizeof(*pool));
}

struct agc_subpool_slab *
agc_subpool_alloc(struct agc_subpool *pool, uint64_t bytes, uint64_t alignment, uint64_t *offset)
{
   for (uint32_t i = 0; i < pool->count; i++) {
      if (pool->slabs[i] && agc_heap_alloc(&pool->slabs[i]->heap, bytes, alignment, offset))
         return pool->slabs[i];
   }
   return NULL;
}

uint64_t
agc_subpool_growth(const struct agc_subpool *pool, uint64_t bytes)
{
   return MAX2(agc_align_up(bytes, pool->slab_alignment), pool->slab_bytes);
}

struct agc_subpool_slab *
agc_subpool_grow(struct agc_subpool *pool, uint64_t bytes, void *data)
{
   uint32_t slot = 0;
   while (slot < pool->count && pool->slabs[slot])
      slot++;
   if (slot == pool->capacity) {
      if (pool->capacity > UINT32_MAX / 2)
         return NULL;
      uint32_t capacity = pool->capacity ? pool->capacity * 2 : 8;
      struct agc_subpool_slab **slabs = realloc(pool->slabs, (size_t)capacity * sizeof(*slabs));
      if (!slabs)
         return NULL;
      pool->slabs = slabs;
      pool->capacity = capacity;
   }
   struct agc_subpool_slab *slab = calloc(1, sizeof(*slab));
   if (!slab)
      return NULL;
   if (!agc_heap_init(&slab->heap, 0, bytes)) {
      agc_heap_fini(&slab->heap);
      free(slab);
      return NULL;
   }
   slab->bytes = bytes;
   slab->data = data;
   pool->slabs[slot] = slab;
   if (slot == pool->count)
      pool->count++;
   return slab;
}

bool
agc_subpool_owns(const struct agc_subpool *pool, const struct agc_subpool_slab *slab)
{
   for (uint32_t i = 0; slab && i < pool->count; i++) {
      if (pool->slabs[i] == slab)
         return true;
   }
   return false;
}

bool
agc_subpool_free(struct agc_subpool *pool, struct agc_subpool_slab *slab, uint64_t offset, uint64_t bytes)
{
   return agc_subpool_owns(pool, slab) && agc_heap_free(&slab->heap, offset, bytes);
}

uint32_t
agc_subpool_slab_count(const struct agc_subpool *pool)
{
   uint32_t live = 0;
   for (uint32_t i = 0; i < pool->count; i++)
      live += pool->slabs[i] != NULL;
   return live;
}

bool
agc_subpool_shrink(struct agc_subpool *pool, struct agc_subpool_slab *slab, void **data)
{
   for (uint32_t i = 0; slab && i < pool->count; i++) {
      if (pool->slabs[i] != slab)
         continue;
      if (!agc_heap_unused(&slab->heap))
         return false;
      if (data)
         *data = slab->data;
      agc_heap_fini(&slab->heap);
      free(slab);
      pool->slabs[i] = NULL;
      while (pool->count && !pool->slabs[pool->count - 1])
         pool->count--;
      return true;
   }
   return false;
}

static uint32_t
agc_write_tracker_slot(uint64_t key, uint32_t capacity)
{
   return (uint32_t)((key * UINT64_C(0x9e3779b97f4a7c15)) >> 32) & (capacity - 1);
}

static struct agc_write_block *
agc_write_tracker_block(const struct agc_write_tracker *tracker, uint64_t number)
{
   if (!tracker->block_capacity)
      return NULL;
   const uint64_t key = number + 1;
   const uint32_t mask = tracker->block_capacity - 1;
   for (uint32_t slot = agc_write_tracker_slot(key, tracker->block_capacity);; slot = (slot + 1) & mask) {
      struct agc_write_block *block = &tracker->blocks[slot];
      if (block->key == key)
         return block;
      if (block->key == 0)
         return NULL;
   }
}

static struct agc_write_block *
agc_write_tracker_place(struct agc_write_block *blocks, uint32_t capacity, uint64_t key)
{
   uint32_t slot = agc_write_tracker_slot(key, capacity);
   while (blocks[slot].key != 0)
      slot = (slot + 1) & (capacity - 1);
   blocks[slot].key = key;
   return &blocks[slot];
}

static bool
agc_write_tracker_grow(struct agc_write_tracker *tracker)
{
   if (tracker->block_capacity > UINT32_MAX / 2)
      return false;
   uint32_t capacity = tracker->block_capacity ? tracker->block_capacity * 2 : 64;
   struct agc_write_block *blocks = calloc(capacity, sizeof(*blocks));
   if (!blocks)
      return false;
   for (uint32_t i = 0; i < tracker->block_capacity; i++) {
      const struct agc_write_block *old = &tracker->blocks[i];
      if (old->key != 0)
         *agc_write_tracker_place(blocks, capacity, old->key) = *old;
   }
   free(tracker->blocks);
   tracker->blocks = blocks;
   tracker->block_capacity = capacity;
   return true;
}

static void
agc_write_tracker_stamp(struct agc_write_tracker *tracker, uint64_t page, uint64_t generation)
{
   struct agc_write_block *block = agc_write_tracker_block(tracker, page / AGC_WRITE_BLOCK_BYTES);
   if (!block) {
      if ((uint64_t)(tracker->block_count + 1) * 2 > tracker->block_capacity &&
          !agc_write_tracker_grow(tracker)) {
         tracker->overflowed = true;
         return;
      }
      block = agc_write_tracker_place(tracker->blocks, tracker->block_capacity,
                                      page / AGC_WRITE_BLOCK_BYTES + 1);
      tracker->block_count++;
   }
   block->pages[(page % AGC_WRITE_BLOCK_BYTES) / AGC_WRITE_PAGE_BYTES] = generation;
   block->newest = MAX2(block->newest, generation);
}

static void
agc_write_tracker_stamp_range(struct agc_write_tracker *tracker, uint64_t first, uint64_t stop,
                              uint64_t generation)
{
   for (uint64_t page = first; page < stop; page += AGC_WRITE_PAGE_BYTES)
      agc_write_tracker_stamp(tracker, page, generation);
}

static bool
agc_write_tracker_watched(const struct agc_write_tracker *tracker, uint64_t address, uint64_t bytes)
{
   return tracker->collect_writes && tracker->pages && tracker->arena_bytes != 0 &&
          address >= tracker->arena_base && bytes <= UINT64_MAX - address &&
          address + bytes <= agc_range_end(tracker->arena_base, tracker->arena_bytes);
}

void
agc_write_tracker_init(struct agc_write_tracker *tracker, agc_collect_writes collect_writes,
                       uint64_t arena_base, uint64_t arena_bytes)
{
   memset(tracker, 0, sizeof(*tracker));
   tracker->collect_writes = collect_writes;
   tracker->arena_base = arena_base;
   tracker->arena_bytes = arena_bytes;
   tracker->generation = 1;
   tracker->pages = calloc(AGC_WRITE_COLLECT_PAGES, sizeof(*tracker->pages));
}

void
agc_write_tracker_fini(struct agc_write_tracker *tracker)
{
   free(tracker->blocks);
   free(tracker->pages);
   memset(tracker, 0, sizeof(*tracker));
}

uint64_t
agc_write_tracker_collect(struct agc_write_tracker *tracker, uint64_t address, uint64_t bytes)
{
   if (bytes == 0 || bytes > UINT64_MAX - address)
      return 0;
   const uint64_t first = agc_align_down(address, AGC_WRITE_PAGE_BYTES);
   const uint64_t stop = agc_align_up(address + bytes, AGC_WRITE_PAGE_BYTES);
   const uint64_t generation = ++tracker->generation;
   if (!agc_write_tracker_watched(tracker, first, stop - first)) {
      agc_write_tracker_stamp_range(tracker, first, stop, generation);
      return 0;
   }
   for (uint64_t cursor = first; cursor < stop;) {
      size_t count = AGC_WRITE_COLLECT_PAGES;
      if (!tracker->collect_writes((uintptr_t)cursor, (size_t)(stop - cursor), tracker->pages, &count,
                                   true)) {
         agc_write_tracker_stamp_range(tracker, cursor, stop, generation);
         return 0;
      }
      count = MIN2(count, (size_t)AGC_WRITE_COLLECT_PAGES);
      for (size_t i = 0; i < count; i++) {
         uint64_t page = agc_align_down((uint64_t)(uintptr_t)tracker->pages[i], AGC_WRITE_PAGE_BYTES);
         agc_write_tracker_stamp(tracker, page, generation);
      }
      if (count < AGC_WRITE_COLLECT_PAGES)
         break;
      uint64_t next = agc_align_down((uint64_t)(uintptr_t)tracker->pages[count - 1], AGC_WRITE_PAGE_BYTES) +
                      AGC_WRITE_PAGE_BYTES;
      if (next <= cursor)
         break;
      cursor = next;
   }
   return tracker->overflowed ? 0 : generation;
}

bool
agc_write_tracker_unchanged_since(struct agc_write_tracker *tracker, uint64_t address, uint64_t bytes,
                                  uint64_t generation)
{
   if (generation == 0 || agc_write_tracker_collect(tracker, address, bytes) == 0)
      return false;
   const uint64_t stop = agc_align_up(address + bytes, AGC_WRITE_PAGE_BYTES);
   for (uint64_t page = agc_align_down(address, AGC_WRITE_PAGE_BYTES); page < stop;) {
      const struct agc_write_block *block = agc_write_tracker_block(tracker, page / AGC_WRITE_BLOCK_BYTES);
      if (!block || block->newest <= generation) {
         page = (page / AGC_WRITE_BLOCK_BYTES + 1) * AGC_WRITE_BLOCK_BYTES;
         continue;
      }
      if (block->pages[(page % AGC_WRITE_BLOCK_BYTES) / AGC_WRITE_PAGE_BYTES] > generation)
         return false;
      page += AGC_WRITE_PAGE_BYTES;
   }
   return true;
}

void
agc_write_tracker_absorb(struct agc_write_tracker *tracker, uint64_t address, uint64_t bytes)
{
   const uint64_t generation = agc_write_tracker_collect(tracker, address, bytes);
   if (generation == 0)
      return;
   const uint64_t stop = agc_align_up(address + bytes, AGC_WRITE_PAGE_BYTES);
   for (uint64_t page = agc_align_down(address, AGC_WRITE_PAGE_BYTES); page < stop;) {
      struct agc_write_block *block = agc_write_tracker_block(tracker, page / AGC_WRITE_BLOCK_BYTES);
      if (!block) {
         page = (page / AGC_WRITE_BLOCK_BYTES + 1) * AGC_WRITE_BLOCK_BYTES;
         continue;
      }
      uint64_t *stamp = &block->pages[(page % AGC_WRITE_BLOCK_BYTES) / AGC_WRITE_PAGE_BYTES];
      if (*stamp == generation)
         *stamp = 0;
      page += AGC_WRITE_PAGE_BYTES;
   }
}

static bool
agc_span_list_join(struct agc_span_list *list, uint64_t start, uint64_t end)
{
   if (list->count && list->spans[list->count - 1].end == start) {
      list->spans[list->count - 1].end = end;
      return true;
   }
   return agc_span_list_append(list, start, end);
}

uint64_t
agc_write_tracker_written_since(struct agc_write_tracker *tracker, const struct agc_span *ranges,
                                size_t count, uint64_t since, struct agc_span_list *written)
{
   written->count = 0;
   for (size_t i = 0; i < count; i++) {
      const uint64_t start = ranges[i].start;
      const uint64_t end = ranges[i].end;
      if (start >= end)
         continue;
      if (agc_write_tracker_collect(tracker, start, end - start) == 0) {
         agc_span_list_join(written, start, end);
         continue;
      }
      for (uint64_t page = agc_align_down(start, AGC_WRITE_PAGE_BYTES); page < end;) {
         const struct agc_write_block *block = agc_write_tracker_block(tracker, page / AGC_WRITE_BLOCK_BYTES);
         if (!block || block->newest <= since) {
            page = (page / AGC_WRITE_BLOCK_BYTES + 1) * AGC_WRITE_BLOCK_BYTES;
            continue;
         }
         if (block->pages[(page % AGC_WRITE_BLOCK_BYTES) / AGC_WRITE_PAGE_BYTES] > since)
            agc_span_list_join(written, MAX2(page, start), MIN2(page + AGC_WRITE_PAGE_BYTES, end));
         page += AGC_WRITE_PAGE_BYTES;
      }
   }
   return tracker->generation;
}

struct agc_import {
   uint64_t va;
   uint64_t bytes;
   void *alias;
   struct agc_bo *bo;
   bool stale;
};

struct agc_detached {
   uint64_t va;
   uint64_t bytes;
   struct agc_subpool_slab *slab;
   uint64_t offset;
   uint32_t *pins;
};

struct agc_block_memory {
   struct agc_bo *bo;
   void *cpu;
   uint64_t va;
   uint64_t bytes;
   uint32_t flags;
   bool write_combined;
};

struct agc_block {
   struct agc_block *next;
   struct agc_block *previous;
   struct agc_block_memory memory;
};

struct agc_pending_free {
   struct agc_subpool_slab *slab;
   uint64_t offset;
   uint64_t bytes;
};

struct agc_memory {
   int32_t ready;
   int32_t dirty;
   int32_t lost;
   uint64_t synced_generation;
   uint64_t synced_sparse_generation;
   struct agc_anyps5 host;

   struct agc_backend *be;
   const struct agc_driver_ops *ops;
   void (*quiesce)(void *user);
   void (*resume)(void *user);
   void (*released)(void *user, uint64_t guest_va, uint64_t bytes);
   void *user;

   struct agc_range_map imports;
   struct agc_range_map detached;
   struct agc_span_list zero_filled;
   struct agc_span_list sparse_target;
   uint64_t imported_bytes;
   uint64_t detached_bytes;
   uint32_t stale_count;

   struct agc_heap driver_va;
   struct agc_heap runtime_va;
   struct agc_subpool pools[AGC_MEMORY_POOL_COUNT];
   struct agc_subpool mirror_pool;
   struct agc_block *blocks;
   uint64_t block_count;
   uint64_t block_bytes;
   uint64_t runtime_bytes;
   bool host_write_waited;
   uint64_t host_write_generation;

   struct agc_write_tracker writes;

   struct agc_guest_range *visible;
   size_t visible_capacity;
   struct agc_guest_sparse_range *sparse;
   size_t sparse_capacity;
   struct agc_span_list wanted;
   struct agc_span_list missing;
   struct agc_span_list stale;
   struct agc_span_list retry;
   struct agc_span_list zero_part;
   struct agc_span_list other_part;
   struct agc_span_list gaps;
   struct agc_span_list changed;
   struct agc_pending_free *pending_frees;
   uint32_t pending_free_count;
   uint32_t pending_free_capacity;

   struct agc_memory_stats stats;
};

static struct agc_memory agc_memory;
static mtx_t agc_memory_mutex;
static once_flag agc_memory_once = ONCE_FLAG_INIT;

static void
agc_memory_mutex_init(void)
{
   mtx_init(&agc_memory_mutex, mtx_plain);
}

static void
agc_memory_lock(void)
{
   call_once(&agc_memory_once, agc_memory_mutex_init);
   mtx_lock(&agc_memory_mutex);
}

static void
agc_memory_unlock(void)
{
   mtx_unlock(&agc_memory_mutex);
}

static void
agc_memory_clear(void)
{
   struct agc_memory *m = &agc_memory;
   memset(&m->be, 0, sizeof(*m) - offsetof(struct agc_memory, be));
   p_atomic_set(&m->synced_generation, 0);
   p_atomic_set(&m->synced_sparse_generation, 0);
   p_atomic_set(&m->lost, 0);
   p_atomic_set(&m->dirty, 1);
}

static bool
agc_memory_usable(void)
{
   struct agc_memory *m = &agc_memory;
   if (!m->ready)
      return agc_memory_fail(NULL, 0, "the memory module is not initialised");
   if (m->lost)
      return agc_memory_fail(NULL, 0, "the GPU device is lost");
   return true;
}

static void
agc_memory_set_lost(const char *function, const char *what)
{
   struct agc_memory *m = &agc_memory;
   if (p_atomic_xchg(&m->lost, 1))
      return;
   agc_report(function, "%s; guest imports stay pinned until the process exits", what);
}

static void
agc_memory_check_device(const char *function, const char *what)
{
   struct agc_memory *m = &agc_memory;
   if (m->lost)
      return;
   if (m->ops->device_state(m->be, NULL) != AGC_DEVICE_ACTIVE)
      agc_memory_set_lost(function, what);
   else
      agc_report(function, "%s on an active device", what);
}

static bool
agc_memory_settle(const char *function)
{
   struct agc_memory *m = &agc_memory;
   if (m->lost)
      return false;
   if (m->ops->paging_settle(m->be))
      return true;
   agc_memory_set_lost(function, "the backend no longer settles paging work");
   return false;
}

static bool
agc_memory_cpu_write_combined(const void *cpu)
{
#ifdef _WIN32
   MEMORY_BASIC_INFORMATION info;
   if (cpu && VirtualQuery(cpu, &info, sizeof(info)) == sizeof(info))
      return (info.Protect & PAGE_WRITECOMBINE) != 0;
#else
   (void)cpu;
#endif
   return true;
}

static struct agc_heap *
agc_memory_heap_of(uint64_t va)
{
   struct agc_memory *m = &agc_memory;
   return agc_va_in_runtime_range(va) ? &m->runtime_va : &m->driver_va;
}

static bool
agc_memory_block_create(uint64_t bytes, uint32_t flags, uint64_t fixed_va,
                        struct agc_block_memory *memory)
{
   struct agc_memory *m = &agc_memory;
   if (bytes == 0 || bytes > AGC_DRIVER_AREA_BYTES)
      return agc_memory_fail(NULL, 0, "a driver buffer of 0x%" PRIx64 " bytes is not possible", bytes);
   const uint64_t size = agc_align_up(bytes, AGC_DRIVER_BLOCK_ALIGNMENT);
   uint64_t va = fixed_va;
   if (fixed_va) {
      if (!agc_heap_alloc_at(&m->runtime_va, fixed_va, size))
         return agc_memory_fail(NULL, 0,
                                "runtime memory 0x%" PRIx64 "+0x%" PRIx64 " overlaps an earlier fixed "
                                "allocation",
                                fixed_va, size);
   } else if (!agc_heap_alloc(&m->driver_va, size, AGC_DRIVER_BLOCK_ALIGNMENT, &va)) {
      return agc_memory_fail(NULL, 0, "the driver area has no room for 0x%" PRIx64 " bytes", size);
   }
   struct agc_bo *bo = m->ops->bo_create(m->be, size, flags, va);
   if (!bo) {
      agc_heap_free(agc_memory_heap_of(va), va, size);
      return agc_memory_fail(NULL, 0,
                             "the backend made no buffer of 0x%" PRIx64 " bytes (flags 0x%x) at 0x%" PRIx64,
                             size, flags, va);
   }
   memory->bo = bo;
   memory->cpu = m->ops->bo_cpu(bo);
   memory->va = m->ops->bo_va(bo);
   memory->bytes = m->ops->bo_size(bo);
   memory->flags = flags;
   if (!memory->cpu || memory->va != va || memory->bytes != size) {
      agc_report(__func__, "the backend placed 0x%" PRIx64 "+0x%" PRIx64 " (CPU %p) instead of 0x%" PRIx64
                           "+0x%" PRIx64,
                 memory->va, memory->bytes, memory->cpu, va, size);
      m->ops->bo_destroy(m->be, bo);
      agc_heap_free(agc_memory_heap_of(va), va, size);
      memset(memory, 0, sizeof(*memory));
      return agc_memory_fail(NULL, 0, "the backend misplaced a driver buffer");
   }
   memory->write_combined = agc_memory_cpu_write_combined(memory->cpu);
   return true;
}

static void
agc_memory_block_destroy(struct agc_block_memory *memory)
{
   struct agc_memory *m = &agc_memory;
   if (!m->lost)
      m->ops->bo_destroy(m->be, memory->bo);
   if (!agc_heap_free(agc_memory_heap_of(memory->va), memory->va, memory->bytes))
      agc_report(__func__, "driver area extent 0x%" PRIx64 "+0x%" PRIx64 " was not allocated", memory->va,
                 memory->bytes);
   memset(memory, 0, sizeof(*memory));
}

static struct agc_subpool_slab *
agc_memory_pool_alloc(struct agc_subpool *pool, uint64_t bytes, uint64_t alignment, uint32_t flags,
                      uint64_t *offset)
{
   struct agc_subpool_slab *slab = agc_subpool_alloc(pool, bytes, alignment, offset);
   if (slab)
      return slab;
   struct agc_block_memory *memory = calloc(1, sizeof(*memory));
   if (!memory) {
      agc_memory_fail(NULL, 0, "out of memory");
      return NULL;
   }
   if (!agc_memory_block_create(agc_subpool_growth(pool, bytes), flags, 0, memory)) {
      free(memory);
      return NULL;
   }
   slab = agc_subpool_grow(pool, memory->bytes, memory);
   if (!slab || !agc_heap_alloc(&slab->heap, bytes, alignment, offset)) {
      void *data = NULL;
      if (slab)
         agc_subpool_shrink(pool, slab, &data);
      agc_memory_block_destroy(memory);
      free(memory);
      agc_memory_fail(NULL, 0, "out of memory");
      return NULL;
   }
   return slab;
}

static void
agc_memory_pool_free(struct agc_subpool *pool, struct agc_subpool_slab *slab, uint64_t offset, uint64_t bytes)
{
   if (!agc_subpool_free(pool, slab, offset, bytes)) {
      agc_report(__func__, "0x%" PRIx64 "+0x%" PRIx64 " is not an allocation of this pool", offset, bytes);
      return;
   }
   if (agc_memory.lost || !agc_heap_unused(&slab->heap) || agc_subpool_slab_count(pool) < 2)
      return;
   void *data = NULL;
   if (agc_subpool_shrink(pool, slab, &data)) {
      agc_memory_block_destroy(data);
      free(data);
   }
}

static void
agc_memory_pool_destroy(struct agc_subpool *pool)
{
   for (uint32_t i = 0; i < pool->count; i++) {
      if (!pool->slabs[i])
         continue;
      struct agc_block_memory *memory = pool->slabs[i]->data;
      agc_memory_block_destroy(memory);
      free(memory);
   }
   agc_subpool_fini(pool);
}

static uint64_t
agc_memory_pool_bytes(const struct agc_subpool *pool)
{
   uint64_t bytes = 0;
   for (uint32_t i = 0; i < pool->count; i++) {
      if (pool->slabs[i])
         bytes += pool->slabs[i]->bytes;
   }
   return bytes;
}

static void
agc_memory_defer_free(struct agc_subpool_slab *slab, uint64_t offset, uint64_t bytes)
{
   struct agc_memory *m = &agc_memory;
   if (m->pending_free_count == m->pending_free_capacity) {
      uint32_t capacity = m->pending_free_capacity ? m->pending_free_capacity * 2 : 16;
      struct agc_pending_free *frees = realloc(m->pending_frees, (size_t)capacity * sizeof(*frees));
      if (!frees) {
         agc_report(__func__, "out of memory; 0x%" PRIx64 " bytes of mirror memory stay allocated", bytes);
         return;
      }
      m->pending_frees = frees;
      m->pending_free_capacity = capacity;
   }
   m->pending_frees[m->pending_free_count++] = (struct agc_pending_free){slab, offset, bytes};
}

static void
agc_memory_flush_frees(void)
{
   struct agc_memory *m = &agc_memory;
   for (uint32_t i = 0; i < m->pending_free_count; i++) {
      const struct agc_pending_free *pending = &m->pending_frees[i];
      agc_memory_pool_free(&m->mirror_pool, pending->slab, pending->offset, pending->bytes);
   }
   m->pending_free_count = 0;
}

static bool
agc_memory_fill(uint64_t start, uint64_t end, enum agc_va_fill fill)
{
   struct agc_memory *m = &agc_memory;
   return start >= end || m->ops->va_clear(m->be, start, end - start, fill);
}

static bool
agc_memory_restore_background(uint64_t start, uint64_t end)
{
   struct agc_memory *m = &agc_memory;
   struct agc_span whole_span = {start, end};
   const struct agc_span_list whole = {&whole_span, 1, 1};
   if (!agc_span_list_clip(&m->zero_filled, start, end, &m->zero_part) ||
       !agc_span_list_subtract(&whole, &m->zero_part, &m->other_part)) {
      agc_report(__func__, "out of memory; 0x%" PRIx64 "+0x%" PRIx64 " becomes inaccessible", start,
                 end - start);
      m->zero_part.count = 0;
      bool cleared = agc_memory_fill(agc_mirror_va(start), agc_mirror_va(end), AGC_VA_NO_ACCESS);
      return agc_memory_fill(start, end, AGC_VA_NO_ACCESS) && cleared;
   }
   bool ok = true;
   for (unsigned pass = 0; pass < 2; pass++) {
      const uint64_t offset = pass == 0 ? AGC_MIRROR_OFFSET : 0;
      for (uint32_t i = 0; i < m->zero_part.count; i++) {
         ok = agc_memory_fill(m->zero_part.spans[i].start + offset, m->zero_part.spans[i].end + offset,
                              AGC_VA_ZERO) && ok;
      }
      for (uint32_t i = 0; i < m->other_part.count; i++) {
         ok = agc_memory_fill(m->other_part.spans[i].start + offset, m->other_part.spans[i].end + offset,
                              AGC_VA_NO_ACCESS) && ok;
      }
   }
   return ok;
}

static void
agc_memory_drop_detached(uint64_t start, uint64_t end)
{
   struct agc_memory *m = &agc_memory;
   for (;;) {
      uint32_t index = agc_range_map_lower_bound(&m->detached, start);
      if (index == m->detached.count || m->detached.ranges[index].start >= end)
         return;
      struct agc_detached *entry = m->detached.ranges[index].data;
      agc_range_map_remove(&m->detached, entry->va, NULL);
      agc_memory_defer_free(entry->slab, entry->offset, entry->bytes);
      m->detached_bytes -= entry->bytes;
      free(entry->pins);
      free(entry);
   }
}

static void
agc_memory_destroy_import(struct agc_import *import)
{
   struct agc_memory *m = &agc_memory;
   const uint64_t end = import->va + import->bytes;
   agc_memory_drop_detached(import->va, end);
   if (!m->lost && !agc_memory_restore_background(import->va, end))
      agc_memory_check_device(__func__, "a released guest range could not be unmapped");
   if (import->bo && !m->lost)
      m->ops->bo_destroy(m->be, import->bo);
   if (agc_memory_settle(__func__)) {
      m->host.unmap_import_alias(import->alias);
   } else {
      m->host.forget_import((uintptr_t)import->va, (size_t)import->bytes);
      m->stats.imports_abandoned++;
   }
   agc_memory_flush_frees();
   if (import->stale)
      m->stale_count--;
   free(import);
}

static void
agc_memory_release_import(struct agc_import *import)
{
   struct agc_memory *m = &agc_memory;
   agc_range_map_remove(&m->imports, import->va, NULL);
   m->imported_bytes -= import->bytes;
   m->stats.imports_released++;
   agc_memory_destroy_import(import);
}

enum agc_import_outcome {
   AGC_IMPORT_DONE,
   AGC_IMPORT_REFUSED,
   AGC_IMPORT_FAILED
};

static enum agc_import_outcome
agc_memory_import(uint64_t start, uint64_t end, uint32_t *refused)
{
   struct agc_memory *m = &agc_memory;
   const uint64_t bytes = end - start;
   *refused = AGC_GUEST_IMPORT_MADE;
   if (agc_range_map_overlaps(&m->imports, start, end))
      return AGC_IMPORT_DONE;
   struct agc_import *import = calloc(1, sizeof(*import));
   if (!import) {
      agc_memory_fail(NULL, 0, "out of memory");
      return AGC_IMPORT_FAILED;
   }
   import->va = start;
   import->bytes = bytes;
   import->alias = m->host.map_import_alias((uintptr_t)start, (size_t)bytes, refused);
   if (!import->alias) {
      free(import);
      if (*refused == AGC_GUEST_IMPORT_MADE)
         *refused = AGC_GUEST_IMPORT_NOT_GPU_VISIBLE;
      return AGC_IMPORT_REFUSED;
   }
   import->bo = m->ops->bo_import(m->be, import->alias, bytes, agc_memory_guest_flags(), start);
   const char *failure = NULL;
   if (!import->bo)
      failure = "imported by the backend";
   else if (!m->ops->bo_alias(m->be, import->bo, 0, bytes, agc_mirror_va(start)))
      failure = "aliased into the mirror window";
   else if (!agc_range_map_insert(&m->imports, start, end, import))
      failure = "recorded";
   if (failure) {
      agc_memory_destroy_import(import);
      agc_memory_fail(NULL, 0, "guest memory 0x%" PRIx64 "+0x%" PRIx64 " could not be %s", start, bytes,
                      failure);
      return AGC_IMPORT_FAILED;
   }
   m->imported_bytes += bytes;
   m->stats.imports_made++;
   return AGC_IMPORT_DONE;
}

static int
agc_memory_compare_sparse(const void *a, const void *b)
{
   const struct agc_guest_sparse_range *left = a;
   const struct agc_guest_sparse_range *right = b;
   return left->address < right->address ? -1 : left->address > right->address;
}

static bool
agc_memory_fill_gaps(const struct agc_span_list *spans, enum agc_va_fill fill)
{
   struct agc_memory *m = &agc_memory;
   bool ok = true;
   for (uint32_t i = 0; i < spans->count && ok; i++) {
      m->gaps.count = 0;
      if (!agc_range_map_gaps(&m->imports, spans->spans[i].start, spans->spans[i].end, &m->gaps))
         return agc_memory_fail(NULL, 0, "out of memory");
      for (uint32_t k = 0; k < m->gaps.count && ok; k++) {
         const struct agc_span gap = m->gaps.spans[k];
         ok = agc_memory_fill(agc_mirror_va(gap.start), agc_mirror_va(gap.end), fill) &&
              agc_memory_fill(gap.start, gap.end, fill);
      }
   }
   if (!ok) {
      agc_memory_check_device(__func__, "the background of a sparse range could not be mapped");
      return agc_memory_fail(NULL, 0, "the backend could not map the background of a sparse range");
   }
   return true;
}

static bool
agc_memory_sync_sparse(uint64_t *generation)
{
   struct agc_memory *m = &agc_memory;
   *generation = 0;
   if (!m->host.sparse_ranges)
      return true;
   size_t count = 0;
   for (unsigned attempt = 0;; attempt++) {
      count = m->host.sparse_ranges(generation, m->sparse, m->sparse_capacity);
      if (count <= m->sparse_capacity)
         break;
      if (attempt == AGC_MEMORY_QUERY_ATTEMPTS)
         return agc_memory_fail(NULL, 0, "the sparse ranges kept growing while they were read");
      struct agc_guest_sparse_range *sparse = realloc(m->sparse, (count + 64) * sizeof(*sparse));
      if (!sparse)
         return agc_memory_fail(NULL, 0, "out of memory");
      m->sparse = sparse;
      m->sparse_capacity = count + 64;
   }
   if (count > 1)
      qsort(m->sparse, count, sizeof(m->sparse[0]), agc_memory_compare_sparse);

   m->sparse_target.count = 0;
   for (size_t i = 0; i < count; i++) {
      const uint64_t start = MAX2((uint64_t)m->sparse[i].address, AGC_GUEST_WINDOW_BASE);
      const uint64_t end = MIN2(agc_range_end(m->sparse[i].address, m->sparse[i].bytes), AGC_GUEST_WINDOW_END);
      if (!agc_span_list_extend(&m->sparse_target, agc_align_up(start, AGC_GPU_PAGE_BYTES),
                                agc_align_down(end, AGC_GPU_PAGE_BYTES)))
         return agc_memory_fail(NULL, 0, "out of memory");
   }
   if (!agc_span_list_subtract(&m->sparse_target, &m->zero_filled, &m->changed))
      return agc_memory_fail(NULL, 0, "out of memory");
   if (m->changed.count == 0)
      return true;
   if (!agc_memory_fill_gaps(&m->changed, AGC_VA_ZERO))
      return false;
   if (!agc_span_list_union(&m->zero_filled, &m->changed, &m->zero_part))
      return agc_memory_fail(NULL, 0, "out of memory");
   struct agc_span_list merged = m->zero_part;
   m->zero_part = m->zero_filled;
   m->zero_filled = merged;
   return true;
}

static void
agc_memory_retire_sparse(void)
{
   struct agc_memory *m = &agc_memory;
   if (m->lost || !agc_span_list_subtract(&m->zero_filled, &m->sparse_target, &m->changed) ||
       m->changed.count == 0)
      return;
   if (!agc_memory_fill_gaps(&m->changed, AGC_VA_NO_ACCESS))
      return;
   if (!agc_span_list_subtract(&m->zero_filled, &m->changed, &m->zero_part)) {
      agc_report(__func__, "out of memory");
      return;
   }
   struct agc_span_list kept = m->zero_part;
   m->zero_part = m->zero_filled;
   m->zero_filled = kept;
}

static bool
agc_memory_query_visible(size_t *count)
{
   struct agc_memory *m = &agc_memory;
   for (unsigned attempt = 0;; attempt++) {
      *count = m->host.gpu_visible((uintptr_t)AGC_GUEST_WINDOW_BASE, (size_t)AGC_GUEST_WINDOW_BYTES,
                                   m->visible, m->visible_capacity);
      if (*count <= m->visible_capacity)
         return true;
      if (attempt == AGC_MEMORY_QUERY_ATTEMPTS)
         return agc_memory_fail(NULL, 0, "the GPU-visible pieces kept growing while they were read");
      size_t capacity = *count + *count / 2 + 64;
      struct agc_guest_range *visible = realloc(m->visible, capacity * sizeof(*visible));
      if (!visible)
         return agc_memory_fail(NULL, 0, "out of memory");
      m->visible = visible;
      m->visible_capacity = capacity;
   }
}

static void
agc_memory_note_refusal(const struct agc_span *span, uint32_t refused)
{
   struct agc_memory *m = &agc_memory;
   m->stats.last_sync_unimported_bytes += span->end - span->start;
   switch (refused) {
   case AGC_GUEST_IMPORT_NOT_GPU_VISIBLE:
   case AGC_GUEST_IMPORT_HOST_WRITE_OPEN:
      m->stats.last_sync_refused_transient++;
      break;
   case AGC_GUEST_IMPORT_NOT_SECTION_MEMORY:
      m->stats.last_sync_refused_unimportable++;
      break;
   default:
      m->stats.last_sync_refused_unimportable++;
      agc_report(__func__, "AnyPS5 refused the import of 0x%" PRIx64 "+0x%" PRIx64 " with code %u",
                 span->start, span->end - span->start, refused);
      break;
   }
}

static void
agc_memory_mark_stale(void)
{
   struct agc_memory *m = &agc_memory;
   uint32_t listed = 0;
   for (uint32_t i = 0; i < m->imports.count; i++) {
      struct agc_import *import = m->imports.ranges[i].data;
      while (listed < m->stale.count && m->stale.spans[listed].start < import->va)
         listed++;
      const bool stale = listed < m->stale.count && m->stale.spans[listed].start == import->va;
      if (stale && !import->stale) {
         m->stale_count++;
         agc_report(__func__, "the import 0x%" PRIx64 "+0x%" PRIx64 " is no longer GPU-visible and AnyPS5 "
                              "asked for no release; it goes at the next release",
                    import->va, import->bytes);
      } else if (!stale && import->stale) {
         m->stale_count--;
      }
      import->stale = stale;
   }
}

static bool
agc_memory_import_missing(uint64_t generation, bool *incomplete)
{
   struct agc_memory *m = &agc_memory;
   uint32_t refused = AGC_GUEST_IMPORT_MADE;
   m->retry.count = 0;
   for (uint32_t i = 0; i < m->missing.count; i++) {
      const struct agc_span span = m->missing.spans[i];
      enum agc_import_outcome outcome = agc_memory_import(span.start, span.end, &refused);
      if (outcome == AGC_IMPORT_FAILED)
         return false;
      if (outcome == AGC_IMPORT_DONE)
         continue;
      if (refused == AGC_GUEST_IMPORT_HOST_WRITE_OPEN && agc_span_list_append(&m->retry, span.start, span.end))
         continue;
      agc_memory_note_refusal(&span, refused);
      *incomplete |= refused == AGC_GUEST_IMPORT_HOST_WRITE_OPEN;
   }

   const bool wait = !m->host_write_waited || m->host_write_generation != generation;
   const int64_t deadline = os_time_get_nano() + (wait ? AGC_MEMORY_HOST_WRITE_WAIT_NS : 0);
   while (m->retry.count && os_time_get_nano() < deadline) {
      os_time_sleep(AGC_MEMORY_HOST_WRITE_POLL_US);
      uint32_t kept = 0;
      for (uint32_t i = 0; i < m->retry.count; i++) {
         const struct agc_span span = m->retry.spans[i];
         enum agc_import_outcome outcome = agc_memory_import(span.start, span.end, &refused);
         if (outcome == AGC_IMPORT_FAILED)
            return false;
         if (outcome == AGC_IMPORT_DONE)
            continue;
         if (refused == AGC_GUEST_IMPORT_HOST_WRITE_OPEN)
            m->retry.spans[kept++] = span;
         else
            agc_memory_note_refusal(&span, refused);
      }
      m->retry.count = kept;
   }
   if (m->retry.count) {
      *incomplete = true;
      m->host_write_waited = true;
      m->host_write_generation = generation;
   }
   for (uint32_t i = 0; i < m->retry.count; i++) {
      agc_memory_note_refusal(&m->retry.spans[i], AGC_GUEST_IMPORT_HOST_WRITE_OPEN);
      if (wait) {
         agc_report(__func__, "a host write stayed open on guest memory 0x%" PRIx64 "+0x%" PRIx64 " for %" PRId64
                              " ms; it is not imported yet",
                    m->retry.spans[i].start, m->retry.spans[i].end - m->retry.spans[i].start,
                    AGC_MEMORY_HOST_WRITE_WAIT_NS / 1000000);
      }
   }
   return true;
}

static bool
agc_memory_synced(void)
{
   struct agc_memory *m = &agc_memory;
   if (p_atomic_read(&m->dirty) || p_atomic_read(&m->lost))
      return false;
   uint64_t sparse_generation = 0;
   if (m->host.sparse_ranges)
      m->host.sparse_ranges(&sparse_generation, NULL, 0);
   return p_atomic_read(&m->synced_generation) == m->host.gpu_generation() &&
          p_atomic_read(&m->synced_sparse_generation) == sparse_generation;
}

static bool
agc_memory_sync_locked(void)
{
   struct agc_memory *m = &agc_memory;
   if (!agc_memory_usable())
      return false;
   if (agc_memory_synced())
      return true;
   const uint64_t generation = m->host.gpu_generation();
   uint64_t sparse_generation = 0;
   p_atomic_set(&m->dirty, 1);
   m->stats.sync_passes++;
   m->stats.last_sync_refused_transient = 0;
   m->stats.last_sync_refused_unimportable = 0;
   m->stats.last_sync_unimported_bytes = 0;

   if (!agc_memory_sync_sparse(&sparse_generation))
      return false;

   size_t count = 0;
   if (!agc_memory_query_visible(&count))
      return false;
   if (!agc_import_plan(m->visible, count, &m->wanted))
      return agc_memory_fail(NULL, 0, "AnyPS5's GPU-visible pieces are out of order, or memory ran out");
   if (!agc_import_diff(&m->wanted, &m->imports, &m->missing, &m->stale))
      return agc_memory_fail(NULL, 0, "out of memory");
   if (m->stale.count || m->stale_count)
      agc_memory_mark_stale();

   bool incomplete = false;
   bool imported = agc_memory_import_missing(generation, &incomplete);
   bool settled = agc_memory_settle(__func__);
   if (!imported)
      return false;
   if (!settled)
      return agc_memory_fail(NULL, 0, "the GPU device is lost");
   if (!incomplete) {
      p_atomic_set(&m->synced_generation, generation);
      p_atomic_set(&m->synced_sparse_generation, sparse_generation);
      p_atomic_set(&m->dirty, 0);
   }
   return true;
}

bool
agc_memory_sync(char *error, size_t error_bytes)
{
   struct agc_memory *m = &agc_memory;
   if (!p_atomic_read(&m->ready))
      return agc_memory_fail(error, error_bytes, "the memory module is not initialised");
   if (agc_memory_synced())
      return true;
   agc_memory_lock();
   bool ok = agc_memory_sync_locked();
   agc_memory_unlock();
   if (!ok)
      agc_memory_copy_error(error, error_bytes);
   return ok;
}

static bool
agc_memory_narrowing_pending(void)
{
   struct agc_memory *m = &agc_memory;
   if (m->stale_count)
      return true;
   return !agc_span_list_subtract(&m->zero_filled, &m->sparse_target, &m->changed) || m->changed.count;
}

static uint32_t
agc_memory_release_locked(uint64_t start, uint64_t end, struct agc_span_list *gone)
{
   struct agc_memory *m = &agc_memory;
   uint32_t released = 0;
   if (!m->lost && m->ops->device_state(m->be, NULL) != AGC_DEVICE_ACTIVE)
      agc_memory_set_lost(__func__, "the GPU device left the active state");
   while (start < end) {
      uint32_t index = agc_range_map_lower_bound(&m->imports, start);
      if (index == m->imports.count || m->imports.ranges[index].start >= end)
         break;
      struct agc_import *import = m->imports.ranges[index].data;
      agc_span_list_append(gone, import->va, import->va + import->bytes);
      agc_memory_release_import(import);
      released++;
   }
   for (uint32_t i = 0; m->stale_count && i < m->imports.count;) {
      struct agc_import *import = m->imports.ranges[i].data;
      if (!import->stale) {
         i++;
         continue;
      }
      agc_span_list_append(gone, import->va, import->va + import->bytes);
      agc_memory_release_import(import);
      released++;
   }
   agc_memory_retire_sparse();
   if (released)
      p_atomic_set(&m->dirty, 1);
   return released;
}

static uint32_t
agc_memory_release_range(uint64_t start, uint64_t end, struct agc_span_list *gone)
{
   struct agc_memory *m = &agc_memory;
   gone->count = 0;
   agc_memory_lock();
   const bool pending = m->ready && (start < end ? agc_range_map_overlaps(&m->imports, start, end)
                                                 : agc_memory_narrowing_pending());
   void (*quiesce)(void *) = m->quiesce;
   void (*resume)(void *) = m->resume;
   void (*notify)(void *, uint64_t, uint64_t) = m->released;
   void *core = m->user;
   agc_memory_unlock();
   if (!pending)
      return 0;

   if (quiesce)
      quiesce(core);

   uint32_t released = 0;
   agc_memory_lock();
   if (m->ready)
      released = agc_memory_release_locked(start, end, gone);
   agc_memory_unlock();

   for (uint32_t i = 0; notify && i < gone->count; i++)
      notify(core, gone->spans[i].start, gone->spans[i].end - gone->spans[i].start);
   if (resume)
      resume(core);
   return released;
}

uint32_t
agc_memory_release_stale(struct agc_span_list *gone)
{
   struct agc_memory *m = &agc_memory;
   struct agc_span_list local = {0};
   if (gone)
      gone->count = 0;
   if (!p_atomic_read(&m->ready))
      return 0;
   uint32_t released = agc_memory_release_range(0, 0, gone ? gone : &local);
   agc_span_list_fini(&local);
   return released;
}

static bool
agc_memory_release(void *user, uintptr_t address, size_t bytes)
{
   struct agc_memory *m = &agc_memory;
   struct agc_span_list gone = {0};
   (void)user;
   if (!p_atomic_read(&m->ready) || bytes == 0)
      return false;
   uint32_t released = agc_memory_release_range(address, agc_range_end(address, bytes), &gone);
   agc_span_list_fini(&gone);
   return released != 0;
}

static bool
agc_memory_reserve(uint64_t base, uint64_t bytes, const char *name, bool *reserved)
{
   struct agc_memory *m = &agc_memory;
   *reserved = m->ops->va_reserve(m->be, base, bytes);
   if (*reserved)
      return true;
   return agc_memory_fail(NULL, 0,
                          "the %s 0x%" PRIx64 "..0x%" PRIx64 " could not be reserved in the GPU address "
                          "space; it must be reserved before anything else in the process maps GPU memory",
                          name, base, base + bytes);
}

bool
agc_memory_init(const struct agc_memory_config *config, char *error, size_t error_bytes)
{
   struct agc_memory *m = &agc_memory;
   if (!config || !config->backend || !config->backend->ops || !config->anyps5)
      return agc_memory_fail(error, error_bytes, "the memory module needs a backend and AnyPS5's functions");
   const struct agc_driver_ops *ops = config->backend->ops;
   const struct agc_anyps5 *host = config->anyps5;
   if (!ops->va_reserve || !ops->va_clear || !ops->va_release || !ops->bo_create || !ops->bo_import ||
       !ops->bo_alias || !ops->bo_cpu || !ops->bo_va || !ops->bo_size || !ops->bo_destroy ||
       !ops->paging_settle || !ops->device_state)
      return agc_memory_fail(error, error_bytes, "the backend lacks address space or buffer functions");
   if (!host->map_import_alias || !host->unmap_import_alias || !host->gpu_visible || !host->forget_import ||
       !host->gpu_generation || !host->set_import_release)
      return agc_memory_fail(error, error_bytes, "AnyPS5's guest import functions are incomplete");

   agc_memory_lock();
   if (m->ready) {
      agc_memory_unlock();
      return agc_memory_fail(error, error_bytes, "the memory module is already initialised");
   }
   agc_memory_clear();
   m->host = *host;
   m->be = config->backend;
   m->ops = ops;
   m->quiesce = config->quiesce;
   m->resume = config->resume;
   m->released = config->released;
   m->user = config->user;

   bool guest = false;
   bool mirror = false;
   bool area = false;
   bool ok = agc_heap_init(&m->driver_va, AGC_DRIVER_AREA_BASE, AGC_DRIVER_AREA_BYTES) &&
             agc_heap_alloc_at(&m->driver_va, AGC_RUNTIME_BASE, AGC_RUNTIME_BYTES) &&
             agc_heap_init(&m->runtime_va, AGC_RUNTIME_BASE, AGC_RUNTIME_BYTES);
   if (!ok)
      agc_memory_fail(NULL, 0, "out of memory");
   ok = ok && agc_memory_reserve(AGC_GUEST_WINDOW_BASE, AGC_GUEST_WINDOW_BYTES, "guest window", &guest) &&
        agc_memory_reserve(AGC_MIRROR_WINDOW_BASE, AGC_GUEST_WINDOW_BYTES, "mirror window", &mirror) &&
        agc_memory_reserve(AGC_DRIVER_AREA_BASE, AGC_DRIVER_AREA_BYTES, "driver area", &area);
   if (!ok) {
      if (area)
         ops->va_release(m->be, AGC_DRIVER_AREA_BASE, AGC_DRIVER_AREA_BYTES);
      if (mirror)
         ops->va_release(m->be, AGC_MIRROR_WINDOW_BASE, AGC_GUEST_WINDOW_BYTES);
      if (guest)
         ops->va_release(m->be, AGC_GUEST_WINDOW_BASE, AGC_GUEST_WINDOW_BYTES);
      agc_heap_fini(&m->driver_va);
      agc_heap_fini(&m->runtime_va);
      agc_memory_clear();
      agc_memory_unlock();
      agc_memory_copy_error(error, error_bytes);
      return false;
   }

   for (unsigned i = 0; i < AGC_MEMORY_POOL_COUNT; i++)
      agc_subpool_init(&m->pools[i], AGC_SUBALLOC_SLAB_BYTES, AGC_DRIVER_BLOCK_ALIGNMENT);
   agc_subpool_init(&m->mirror_pool, AGC_MIRROR_SLAB_BYTES, AGC_DRIVER_BLOCK_ALIGNMENT);

   uintptr_t arena_base = 0;
   size_t arena_bytes = 0;
   if (m->host.collect_writes && m->host.write_watched && m->host.arena_range && m->host.write_watched())
      m->host.arena_range(&arena_base, &arena_bytes);
   agc_write_tracker_init(&m->writes, m->host.collect_writes, arena_base, arena_bytes);

   p_atomic_set(&m->dirty, 1);
   p_atomic_set(&m->ready, 1);
   m->host.set_import_release(agc_memory_release, NULL);
   agc_memory_unlock();
   return true;
}

void
agc_memory_fini(void)
{
   struct agc_memory *m = &agc_memory;
   agc_memory_lock();
   if (!m->ready) {
      agc_memory_unlock();
      return;
   }
   p_atomic_set(&m->ready, 0);
   m->host.set_import_release(NULL, NULL);

   if (!m->lost && m->ops->device_state(m->be, NULL) != AGC_DEVICE_ACTIVE)
      agc_memory_set_lost(__func__, "the GPU device left the active state");
   if (!m->lost && !(m->ops->va_release(m->be, AGC_MIRROR_WINDOW_BASE, AGC_GUEST_WINDOW_BYTES) &&
                     m->ops->va_release(m->be, AGC_GUEST_WINDOW_BASE, AGC_GUEST_WINDOW_BYTES)))
      agc_memory_set_lost(__func__, "the guest and mirror windows could not be released");

   for (uint32_t i = 0; i < m->imports.count && !m->lost; i++) {
      struct agc_import *import = m->imports.ranges[i].data;
      m->ops->bo_destroy(m->be, import->bo);
   }
   bool settled = agc_memory_settle(__func__);
   for (uint32_t i = 0; i < m->imports.count; i++) {
      struct agc_import *import = m->imports.ranges[i].data;
      if (settled)
         m->host.unmap_import_alias(import->alias);
      else
         m->host.forget_import((uintptr_t)import->va, (size_t)import->bytes);
      free(import);
   }
   for (uint32_t i = 0; i < m->detached.count; i++) {
      struct agc_detached *entry = m->detached.ranges[i].data;
      free(entry->pins);
      free(entry);
   }

   if (m->block_count)
      agc_report(__func__, "%" PRIu64 " driver buffers of 0x%" PRIx64 " bytes were never freed",
                 m->block_count, m->block_bytes);
   while (m->blocks) {
      struct agc_block *block = m->blocks;
      m->blocks = block->next;
      agc_memory_block_destroy(&block->memory);
      free(block);
   }
   for (unsigned i = 0; i < AGC_MEMORY_POOL_COUNT; i++)
      agc_memory_pool_destroy(&m->pools[i]);
   agc_memory_pool_destroy(&m->mirror_pool);
   if (!m->lost && !m->ops->va_release(m->be, AGC_DRIVER_AREA_BASE, AGC_DRIVER_AREA_BYTES))
      agc_memory_set_lost(__func__, "the driver area could not be released");

   agc_range_map_fini(&m->imports);
   agc_range_map_fini(&m->detached);
   agc_span_list_fini(&m->zero_filled);
   agc_span_list_fini(&m->sparse_target);
   agc_span_list_fini(&m->wanted);
   agc_span_list_fini(&m->missing);
   agc_span_list_fini(&m->stale);
   agc_span_list_fini(&m->retry);
   agc_span_list_fini(&m->zero_part);
   agc_span_list_fini(&m->other_part);
   agc_span_list_fini(&m->gaps);
   agc_span_list_fini(&m->changed);
   agc_heap_fini(&m->driver_va);
   agc_heap_fini(&m->runtime_va);
   agc_write_tracker_fini(&m->writes);
   free(m->visible);
   free(m->sparse);
   free(m->pending_frees);
   agc_memory_clear();
   agc_memory_unlock();
}

bool
agc_memory_guest_cached(void)
{
   static int cached = -1;
   if (cached < 0) {
      const char *value = getenv("AGC_GUEST_CACHED");
      cached = !value || !*value || *value != '0';
   }
   return cached != 0;
}

uint32_t
agc_memory_guest_flags(void)
{
   return agc_memory_guest_cached() ? 0 : AGC_BO_GL2_BYPASS;
}

bool
agc_memory_imported(uint64_t va, uint64_t bytes)
{
   struct agc_memory *m = &agc_memory;
   if (!p_atomic_read(&m->ready))
      return false;
   if (agc_va_in_mirror_window(va))
      va = agc_guest_va(va);
   const uint64_t end = agc_range_end(va, bytes ? bytes : 1);
   agc_memory_lock();
   bool covered = m->ready && agc_range_map_covers(&m->imports, va, end);
   agc_memory_unlock();
   return covered;
}

static void *
block_cpu(const struct agc_block_memory *memory, uint64_t va, uint64_t bytes)
{
   if (!memory || !memory->cpu || va < memory->va || va - memory->va > memory->bytes ||
       bytes > memory->bytes - (va - memory->va))
      return NULL;
   return (uint8_t *)memory->cpu + (va - memory->va);
}

static void *
pool_cpu(const struct agc_subpool *pool, uint64_t va, uint64_t bytes)
{
   for (uint32_t i = 0; i < pool->count; i++) {
      void *cpu = block_cpu(pool->slabs[i] ? pool->slabs[i]->data : NULL, va, bytes);
      if (cpu)
         return cpu;
   }
   return NULL;
}

void *
agc_memory_cpu_of(uint64_t va, uint64_t bytes)
{
   struct agc_memory *m = &agc_memory;
   if (!p_atomic_read(&m->ready))
      return NULL;
   if (agc_va_in_guest_window(va))
      return agc_memory_imported(va, bytes) ? (void *)(uintptr_t)va : NULL;
   void *found = NULL;
   agc_memory_lock();
   for (struct agc_block *block = m->blocks; block && !found; block = block->next)
      found = block_cpu(&block->memory, va, bytes);
   for (unsigned i = 0; i < AGC_MEMORY_POOL_COUNT && !found; i++)
      found = pool_cpu(&m->pools[i], va, bytes);
   if (!found)
      found = pool_cpu(&m->mirror_pool, va, bytes);
   agc_memory_unlock();
   return found;
}

void
agc_memory_get_stats(struct agc_memory_stats *stats)
{
   struct agc_memory *m = &agc_memory;
   agc_memory_lock();
   *stats = m->stats;
   if (m->ready) {
      stats->imports = m->imports.count;
      stats->imported_bytes = m->imported_bytes;
      stats->stale_imports = m->stale_count;
      stats->detached_bytes = m->detached_bytes;
      stats->mirror_private_bytes = agc_memory_pool_bytes(&m->mirror_pool);
      stats->zero_filled_bytes = agc_span_list_bytes(&m->zero_filled);
      stats->driver_blocks = m->block_count;
      stats->driver_block_bytes = m->block_bytes;
      stats->runtime_bytes = m->runtime_bytes;
      for (unsigned i = 0; i < AGC_MEMORY_POOL_COUNT; i++) {
         stats->suballoc_slabs += agc_subpool_slab_count(&m->pools[i]);
         stats->suballoc_slab_bytes += agc_memory_pool_bytes(&m->pools[i]);
      }
      stats->lost = m->lost != 0;
   }
   agc_memory_unlock();
}

static bool
agc_memory_alloc_block(uint64_t bytes, uint32_t flags, uint64_t fixed_va, struct agc_mem *mem)
{
   struct agc_memory *m = &agc_memory;
   struct agc_block *block = calloc(1, sizeof(*block));
   if (!block)
      return agc_memory_fail(NULL, 0, "out of memory");
   agc_memory_lock();
   bool ok = agc_memory_usable() && agc_memory_block_create(bytes, flags, fixed_va, &block->memory);
   if (ok) {
      block->next = m->blocks;
      if (m->blocks)
         m->blocks->previous = block;
      m->blocks = block;
      m->block_count++;
      m->block_bytes += block->memory.bytes;
      if (fixed_va)
         m->runtime_bytes += block->memory.bytes;
      mem->cpu = block->memory.cpu;
      mem->va = block->memory.va;
      mem->bytes = block->memory.bytes;
      mem->flags = flags;
      mem->write_combined = block->memory.write_combined;
      mem->owner = block;
      mem->extent = block->memory.bytes;
   }
   agc_memory_unlock();
   if (!ok)
      free(block);
   return ok;
}

bool
agc_mem_alloc(uint64_t bytes, uint32_t flags, struct agc_mem *mem)
{
   memset(mem, 0, sizeof(*mem));
   if (bytes == 0 || (flags & ~AGC_MEMORY_POOL_FLAGS))
      return agc_memory_fail(NULL, 0, "a driver buffer of 0x%" PRIx64 " bytes with flags 0x%x is not possible",
                             bytes, flags);
   return agc_memory_alloc_block(bytes, flags, 0, mem);
}

bool
agc_mem_alloc_at(uint64_t va, uint64_t bytes, uint32_t flags, struct agc_mem *mem)
{
   memset(mem, 0, sizeof(*mem));
   if (bytes == 0 || (flags & ~AGC_MEMORY_POOL_FLAGS))
      return agc_memory_fail(NULL, 0, "a runtime buffer of 0x%" PRIx64 " bytes with flags 0x%x is not possible",
                             bytes, flags);
   if (va % AGC_DRIVER_BLOCK_ALIGNMENT != 0)
      return agc_memory_fail(NULL, 0, "runtime address 0x%" PRIx64 " is not 64 KiB aligned", va);
   const uint64_t extent = agc_align_up(bytes, AGC_DRIVER_BLOCK_ALIGNMENT);
   if (!agc_va_in_runtime_range(va) || extent < bytes || extent > AGC_RUNTIME_BYTES - (va - AGC_RUNTIME_BASE))
      return agc_memory_fail(NULL, 0,
                             "runtime memory 0x%" PRIx64 "+0x%" PRIx64 " is outside the runtime range 0x%" PRIx64
                             "..0x%" PRIx64,
                             va, bytes, (uint64_t)AGC_RUNTIME_BASE, (uint64_t)(AGC_RUNTIME_BASE + AGC_RUNTIME_BYTES));
   return agc_memory_alloc_block(bytes, flags, va, mem);
}

void
agc_mem_free(struct agc_mem *mem)
{
   struct agc_memory *m = &agc_memory;
   if (!mem || !mem->owner)
      return;
   agc_memory_lock();
   struct agc_block *block = m->ready ? m->blocks : NULL;
   while (block && block != mem->owner)
      block = block->next;
   if (block) {
      if (block->previous)
         block->previous->next = block->next;
      else
         m->blocks = block->next;
      if (block->next)
         block->next->previous = block->previous;
      m->block_count--;
      m->block_bytes -= block->memory.bytes;
      if (agc_va_in_runtime_range(block->memory.va))
         m->runtime_bytes -= block->memory.bytes;
      agc_memory_block_destroy(&block->memory);
      free(block);
   } else if (m->ready) {
      agc_report(__func__, "0x%" PRIx64 "+0x%" PRIx64 " is not a live buffer of agc_mem_alloc", mem->va,
                 mem->bytes);
   }
   agc_memory_unlock();
   memset(mem, 0, sizeof(*mem));
}

bool
agc_mem_suballoc(uint64_t bytes, uint64_t alignment, uint32_t flags, struct agc_mem *mem)
{
   struct agc_memory *m = &agc_memory;
   memset(mem, 0, sizeof(*mem));
   if (alignment == 0)
      alignment = AGC_SUBALLOC_GRANULE_BYTES;
   if (bytes == 0 || bytes > AGC_DRIVER_AREA_BYTES || (flags & ~AGC_MEMORY_POOL_FLAGS) ||
       (alignment & (alignment - 1)) || alignment > AGC_DRIVER_BLOCK_ALIGNMENT)
      return agc_memory_fail(NULL, 0,
                             "a suballocation of 0x%" PRIx64 " bytes aligned to 0x%" PRIx64 " with flags 0x%x "
                             "is not possible",
                             bytes, alignment, flags);
   const uint64_t extent = agc_align_up(bytes, AGC_SUBALLOC_GRANULE_BYTES);
   alignment = MAX2(alignment, AGC_SUBALLOC_GRANULE_BYTES);
   agc_memory_lock();
   uint64_t offset = 0;
   struct agc_subpool_slab *slab = NULL;
   if (agc_memory_usable())
      slab = agc_memory_pool_alloc(&m->pools[flags], extent, alignment, flags, &offset);
   if (slab) {
      const struct agc_block_memory *memory = slab->data;
      mem->cpu = (uint8_t *)memory->cpu + offset;
      mem->va = memory->va + offset;
      mem->bytes = bytes;
      mem->flags = flags;
      mem->write_combined = memory->write_combined;
      mem->owner = slab;
      mem->extent = extent;
   }
   agc_memory_unlock();
   return slab != NULL;
}

void
agc_mem_subfree(struct agc_mem *mem)
{
   struct agc_memory *m = &agc_memory;
   if (!mem || !mem->owner)
      return;
   agc_memory_lock();
   struct agc_subpool *pool = &m->pools[mem->flags & AGC_MEMORY_POOL_FLAGS];
   struct agc_subpool_slab *slab = mem->owner;
   if (m->ready && agc_subpool_owns(pool, slab)) {
      const struct agc_block_memory *memory = slab->data;
      agc_memory_pool_free(pool, slab, mem->va - memory->va, mem->extent);
   } else if (m->ready) {
      agc_report(__func__, "0x%" PRIx64 "+0x%" PRIx64 " is not a live suballocation", mem->va, mem->bytes);
   }
   agc_memory_unlock();
   memset(mem, 0, sizeof(*mem));
}

static struct agc_detached *
agc_memory_detach_piece(const struct agc_import *import, uint64_t start, uint64_t end)
{
   struct agc_memory *m = &agc_memory;
   const uint64_t bytes = end - start;
   const uint64_t pages = bytes / AGC_GPU_PAGE_BYTES;
   struct agc_detached *entry = calloc(1, sizeof(*entry));
   uint32_t *pins = calloc((size_t)pages, sizeof(*pins));
   uint64_t offset = 0;
   struct agc_subpool_slab *slab = NULL;
   if (entry && pins)
      slab = agc_memory_pool_alloc(&m->mirror_pool, bytes, AGC_GPU_PAGE_BYTES, agc_memory_guest_flags(), &offset);
   else
      agc_memory_fail(NULL, 0, "out of memory");
   if (!slab) {
      free(pins);
      free(entry);
      return NULL;
   }
   for (uint64_t i = 0; i < pages; i++)
      pins[i] = 1;
   entry->va = start;
   entry->bytes = bytes;
   entry->slab = slab;
   entry->offset = offset;
   entry->pins = pins;

   const struct agc_block_memory *memory = slab->data;
   memcpy((uint8_t *)memory->cpu + offset, (const uint8_t *)import->alias + (start - import->va), (size_t)bytes);
   agc_mem_flush_write_combining();
   if (!agc_range_map_insert(&m->detached, start, end, entry)) {
      agc_memory_fail(NULL, 0, "out of memory");
   } else if (!m->ops->bo_alias(m->be, memory->bo, offset, bytes, agc_mirror_va(start))) {
      agc_range_map_remove(&m->detached, start, NULL);
      agc_memory_fail(NULL, 0, "the backend could not map private memory at mirror address 0x%" PRIx64,
                      agc_mirror_va(start));
   } else {
      m->detached_bytes += bytes;
      return entry;
   }
   agc_memory_pool_free(&m->mirror_pool, slab, offset, bytes);
   free(pins);
   free(entry);
   return NULL;
}

static bool
agc_memory_reattach_unpinned(struct agc_detached *entry)
{
   struct agc_memory *m = &agc_memory;
   const uint64_t pages = entry->bytes / AGC_GPU_PAGE_BYTES;
   struct agc_range *source = agc_range_map_find(&m->imports, entry->va);
   const struct agc_import *import = source ? source->data : NULL;
   bool ok = true;
   agc_range_map_remove(&m->detached, entry->va, NULL);
   for (uint64_t first = 0; first < pages;) {
      const bool pinned = entry->pins[first] != 0;
      uint64_t last = first + 1;
      while (last < pages && (entry->pins[last] != 0) == pinned)
         last++;
      const uint64_t run_start = entry->va + first * AGC_GPU_PAGE_BYTES;
      const uint64_t run_bytes = (last - first) * AGC_GPU_PAGE_BYTES;
      const uint64_t run_offset = entry->offset + first * AGC_GPU_PAGE_BYTES;
      bool reattached = false;
      if (!pinned) {
         reattached = m->lost || (import && m->ops->bo_alias(m->be, import->bo, run_start - import->va,
                                                             run_bytes, agc_mirror_va(run_start)));
         if (reattached) {
            agc_memory_defer_free(entry->slab, run_offset, run_bytes);
            m->detached_bytes -= run_bytes;
         } else {
            ok = agc_memory_fail(NULL, 0, "the backend could not map the alias back at mirror address 0x%" PRIx64,
                                 agc_mirror_va(run_start));
         }
      }
      if (!reattached) {
         struct agc_detached *piece = calloc(1, sizeof(*piece));
         uint32_t *pins = malloc((size_t)(last - first) * sizeof(*pins));
         if (piece && pins) {
            memcpy(pins, &entry->pins[first], (size_t)(last - first) * sizeof(*pins));
            piece->va = run_start;
            piece->bytes = run_bytes;
            piece->slab = entry->slab;
            piece->offset = run_offset;
            piece->pins = pins;
         }
         if (!piece || !pins || !agc_range_map_insert(&m->detached, run_start, run_start + run_bytes, piece)) {
            agc_report(__func__, "out of memory; mirror pages 0x%" PRIx64 "+0x%" PRIx64 " stay private and "
                                 "untracked",
                       agc_mirror_va(run_start), run_bytes);
            m->detached_bytes -= run_bytes;
            free(pins);
            free(piece);
            ok = agc_memory_fail(NULL, 0, "out of memory");
         }
      }
      first = last;
   }
   free(entry->pins);
   free(entry);
   return ok;
}

static bool
agc_memory_unpin(uint64_t start, uint64_t end)
{
   struct agc_memory *m = &agc_memory;
   bool ok = true;
   uint64_t cursor = start;
   for (;;) {
      uint32_t index = agc_range_map_lower_bound(&m->detached, cursor);
      if (index == m->detached.count || m->detached.ranges[index].start >= end)
         break;
      struct agc_detached *entry = m->detached.ranges[index].data;
      const uint64_t entry_end = entry->va + entry->bytes;
      const uint64_t stop = MIN2(end, entry_end);
      bool unpinned = false;
      for (uint64_t page = MAX2(cursor, entry->va); page < stop; page += AGC_GPU_PAGE_BYTES) {
         uint32_t *pin = &entry->pins[(page - entry->va) / AGC_GPU_PAGE_BYTES];
         if (*pin)
            (*pin)--;
         unpinned |= *pin == 0;
      }
      cursor = entry_end;
      if (unpinned)
         ok = agc_memory_reattach_unpinned(entry) && ok;
   }
   return ok;
}

static bool
agc_memory_detach(uint64_t start, uint64_t end)
{
   struct agc_memory *m = &agc_memory;
   if (!agc_range_map_covers(&m->imports, start, end))
      return agc_memory_fail(NULL, 0, "guest memory 0x%" PRIx64 "+0x%" PRIx64 " is not imported", start,
                             end - start);
   uint64_t cursor = start;
   while (cursor < end) {
      struct agc_range *held = agc_range_map_find(&m->detached, cursor);
      if (held) {
         struct agc_detached *entry = held->data;
         const uint64_t stop = MIN2(end, held->end);
         for (uint64_t page = cursor; page < stop; page += AGC_GPU_PAGE_BYTES)
            entry->pins[(page - entry->va) / AGC_GPU_PAGE_BYTES]++;
         cursor = stop;
         continue;
      }
      uint32_t next = agc_range_map_lower_bound(&m->detached, cursor);
      uint64_t stop = next < m->detached.count ? MIN2(end, m->detached.ranges[next].start) : end;
      const struct agc_range *source = agc_range_map_find(&m->imports, cursor);
      stop = MIN2(stop, source->end);
      if (!agc_memory_detach_piece(source->data, cursor, stop))
         break;
      cursor = stop;
   }
   bool ok = cursor == end;
   if (!ok) {
      char text[sizeof(agc_memory_error_text)];
      snprintf(text, sizeof(text), "%s", agc_memory_error_text);
      agc_memory_unpin(start, cursor);
      agc_memory_fail(NULL, 0, "%s", text);
   }
   if (!agc_memory_settle(__func__))
      ok = agc_memory_fail(NULL, 0, "the GPU device is lost");
   agc_memory_flush_frees();
   return ok;
}

static bool
agc_memory_mirror_range(uint64_t guest_va, uint64_t bytes, uint64_t *start, uint64_t *end)
{
   const uint64_t last = agc_range_end(guest_va, bytes);
   if (bytes == 0 || !agc_va_in_guest_window(guest_va) || last > AGC_GUEST_WINDOW_END)
      return agc_memory_fail(NULL, 0, "0x%" PRIx64 "+0x%" PRIx64 " is not a range of the guest window",
                             guest_va, bytes);
   *start = agc_align_down(guest_va, AGC_GPU_PAGE_BYTES);
   *end = agc_align_up(last, AGC_GPU_PAGE_BYTES);
   return true;
}

bool
agc_mirror_detach(uint64_t guest_va, uint64_t bytes, struct agc_mem *mem)
{
   struct agc_memory *m = &agc_memory;
   uint64_t start = 0;
   uint64_t end = 0;
   if (mem)
      memset(mem, 0, sizeof(*mem));
   if (!agc_memory_mirror_range(guest_va, bytes, &start, &end))
      return false;
   agc_memory_lock();
   bool ok = agc_memory_usable() && agc_memory_detach(start, end);
   if (ok && mem) {
      const struct agc_range *held = agc_range_map_find(&m->detached, guest_va);
      mem->va = agc_mirror_va(guest_va);
      mem->bytes = bytes;
      mem->flags = agc_memory_guest_flags();
      mem->write_combined = true;
      if (held && held->end >= guest_va + bytes) {
         const struct agc_detached *entry = held->data;
         const struct agc_block_memory *memory = entry->slab->data;
         mem->cpu = (uint8_t *)memory->cpu + entry->offset + (guest_va - entry->va);
         mem->write_combined = memory->write_combined;
      }
   }
   agc_memory_unlock();
   return ok;
}

bool
agc_mirror_attach(uint64_t guest_va, uint64_t bytes)
{
   struct agc_memory *m = &agc_memory;
   uint64_t start = 0;
   uint64_t end = 0;
   if (!agc_memory_mirror_range(guest_va, bytes, &start, &end))
      return false;
   agc_memory_lock();
   bool ok = m->ready ? true : agc_memory_fail(NULL, 0, "the memory module is not initialised");
   if (ok) {
      ok = agc_memory_unpin(start, end);
      if (!m->lost && !agc_memory_settle(__func__))
         ok = agc_memory_fail(NULL, 0, "the GPU device is lost");
      agc_memory_flush_frees();
   }
   agc_memory_unlock();
   return ok;
}

bool
agc_mirror_detached(uint64_t guest_va, uint64_t bytes)
{
   struct agc_memory *m = &agc_memory;
   if (!p_atomic_read(&m->ready))
      return false;
   if (agc_va_in_mirror_window(guest_va))
      guest_va = agc_guest_va(guest_va);
   agc_memory_lock();
   bool detached = m->ready && agc_range_map_covers(&m->detached, guest_va, agc_range_end(guest_va, bytes ? bytes : 1));
   agc_memory_unlock();
   return detached;
}

bool
agc_mirror_write(uint64_t guest_va, const void *data, uint64_t bytes)
{
   struct agc_memory *m = &agc_memory;
   if (bytes == 0)
      return true;
   const uint64_t end = agc_range_end(guest_va, bytes);
   agc_memory_lock();
   bool ok = agc_memory_usable();
   if (ok && !agc_range_map_covers(&m->detached, guest_va, end))
      ok = agc_memory_fail(NULL, 0, "mirror memory 0x%" PRIx64 "+0x%" PRIx64 " is not driver-private",
                           agc_mirror_va(guest_va), bytes);
   for (uint64_t cursor = guest_va; ok && cursor < end;) {
      const struct agc_range *held = agc_range_map_find(&m->detached, cursor);
      const struct agc_detached *entry = held->data;
      const struct agc_block_memory *memory = entry->slab->data;
      const uint64_t stop = MIN2(end, held->end);
      memcpy((uint8_t *)memory->cpu + entry->offset + (cursor - entry->va),
             (const uint8_t *)data + (cursor - guest_va), (size_t)(stop - cursor));
      cursor = stop;
   }
   agc_mem_flush_write_combining();
   agc_memory_unlock();
   return ok;
}

uint64_t
agc_writes_collect(uint64_t address, uint64_t bytes)
{
   struct agc_memory *m = &agc_memory;
   agc_memory_lock();
   uint64_t generation = m->ready ? agc_write_tracker_collect(&m->writes, address, bytes) : 0;
   agc_memory_unlock();
   return generation;
}

bool
agc_writes_unchanged_since(uint64_t address, uint64_t bytes, uint64_t generation)
{
   struct agc_memory *m = &agc_memory;
   agc_memory_lock();
   bool unchanged = m->ready && agc_write_tracker_unchanged_since(&m->writes, address, bytes, generation);
   agc_memory_unlock();
   return unchanged;
}

void
agc_writes_absorb(uint64_t address, uint64_t bytes)
{
   struct agc_memory *m = &agc_memory;
   agc_memory_lock();
   if (m->ready)
      agc_write_tracker_absorb(&m->writes, address, bytes);
   agc_memory_unlock();
}

bool
agc_writes_watched(uint64_t address, uint64_t bytes)
{
   struct agc_memory *m = &agc_memory;
   agc_memory_lock();
   const bool watched = m->ready && agc_write_tracker_watched(&m->writes, address, bytes);
   agc_memory_unlock();
   return watched;
}

bool
agc_writes_collectable(uint64_t address, uint64_t bytes)
{
   struct agc_memory *m = &agc_memory;
   agc_memory_lock();
   const bool collectable = m->ready && agc_write_tracker_watched(&m->writes, address, bytes) &&
                            agc_write_tracker_collect(&m->writes, address, bytes) != 0;
   agc_memory_unlock();
   return collectable;
}

uint64_t
agc_writes_written_since(const struct agc_span *ranges, size_t count, uint64_t since,
                         struct agc_span_list *written)
{
   struct agc_memory *m = &agc_memory;
   uint64_t generation = 0;
   agc_memory_lock();
   if (m->ready) {
      generation = agc_write_tracker_written_since(&m->writes, ranges, count, since, written);
   } else {
      written->count = 0;
      for (size_t i = 0; i < count; i++)
         agc_span_list_append(written, ranges[i].start, ranges[i].end);
   }
   agc_memory_unlock();
   return generation;
}
