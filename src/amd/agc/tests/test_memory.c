#include <stdlib.h>
#include <string.h>

#include "agc_memory.h"
#include "test.h"

#define KIB UINT64_C(1024)
#define MIB (UINT64_C(1024) * KIB)
#define PAGE AGC_GUEST_PAGE_BYTES
#define WINDOW AGC_IMPORT_WINDOW_BYTES
#define BASE AGC_GUEST_WINDOW_BASE

static bool
spans_equal(const struct agc_span_list *list, const struct agc_span *expected, uint32_t count)
{
   if (list->count != count)
      return false;
   for (uint32_t i = 0; i < count; i++) {
      if (list->spans[i].start != expected[i].start || list->spans[i].end != expected[i].end)
         return false;
   }
   return true;
}

static void
fill_spans(struct agc_span_list *list, const struct agc_span *spans, uint32_t count)
{
   list->count = 0;
   for (uint32_t i = 0; i < count; i++)
      CHECK(agc_span_list_append(list, spans[i].start, spans[i].end));
}

static void
test_layout(void)
{
   CHECK_EQ(AGC_GUEST_WINDOW_BASE, UINT64_C(0x200000000));
   CHECK_EQ(AGC_GUEST_WINDOW_END, UINT64_C(0xfc00000000));
   CHECK_EQ(AGC_MIRROR_WINDOW_BASE, UINT64_C(0x10200000000));
   CHECK_EQ(AGC_MIRROR_WINDOW_END, UINT64_C(0x1fc00000000));
   CHECK_EQ(AGC_DRIVER_AREA_BASE, UINT64_C(0x20000000000));
   CHECK_EQ(agc_mirror_va(UINT64_C(0x245678000)), UINT64_C(0x10245678000));
   CHECK_EQ(agc_guest_va(UINT64_C(0x10245678000)), UINT64_C(0x245678000));
   CHECK(agc_va_in_guest_window(AGC_GUEST_WINDOW_BASE) && !agc_va_in_guest_window(AGC_GUEST_WINDOW_END));
   CHECK(agc_va_in_mirror_window(agc_mirror_va(AGC_GUEST_WINDOW_END - 1)));
   CHECK(!agc_va_in_mirror_window(AGC_GUEST_WINDOW_BASE) && !agc_va_in_mirror_window(AGC_DRIVER_AREA_BASE));
   CHECK(agc_va_in_driver_area(AGC_DRIVER_AREA_BASE) && !agc_va_in_driver_area(AGC_DRIVER_AREA_END));
   CHECK_EQ(agc_memory_guest_flags(), agc_memory_guest_cached() ? 0u : AGC_BO_GL2_BYPASS);
}

static void
test_span_lists(void)
{
   struct agc_span_list a = {0};
   struct agc_span_list b = {0};
   struct agc_span_list out = {0};

   CHECK(agc_span_list_append(&a, 10, 10) && a.count == 0);
   CHECK(agc_span_list_append(&a, 10, 20) && agc_span_list_append(&a, 20, 30));
   CHECK_EQ(a.count, 2);
   CHECK(agc_span_list_extend(&a, 30, 40) && a.count == 2 && a.spans[1].end == 40);
   CHECK(agc_span_list_extend(&a, 35, 38) && a.count == 2 && a.spans[1].end == 40);
   CHECK(agc_span_list_extend(&a, 50, 60) && a.count == 3);
   CHECK_EQ(agc_span_list_bytes(&a), 40);

   fill_spans(&a, (struct agc_span[]){{0, 100}, {200, 300}, {400, 500}}, 3);
   fill_spans(&b, (struct agc_span[]){{10, 20}, {90, 210}, {250, 260}, {290, 450}}, 4);
   CHECK(agc_span_list_subtract(&a, &b, &out));
   CHECK(spans_equal(&out, (struct agc_span[]){{0, 10}, {20, 90}, {210, 250}, {260, 290}, {450, 500}}, 5));
   CHECK(agc_span_list_subtract(&b, &a, &out));
   CHECK(spans_equal(&out, (struct agc_span[]){{100, 200}, {300, 400}}, 2));
   b.count = 0;
   CHECK(agc_span_list_subtract(&a, &b, &out) && spans_equal(&out, a.spans, a.count));
   CHECK(agc_span_list_subtract(&b, &a, &out) && out.count == 0);
   fill_spans(&b, (struct agc_span[]){{0, 1000}}, 1);
   CHECK(agc_span_list_subtract(&a, &b, &out) && out.count == 0);

   fill_spans(&a, (struct agc_span[]){{0, 10}, {20, 30}, {100, 110}}, 3);
   fill_spans(&b, (struct agc_span[]){{5, 20}, {40, 50}, {110, 120}}, 3);
   CHECK(agc_span_list_union(&a, &b, &out));
   CHECK(spans_equal(&out, (struct agc_span[]){{0, 30}, {40, 50}, {100, 120}}, 3));
   CHECK(agc_span_list_union(&b, &a, &out));
   CHECK(spans_equal(&out, (struct agc_span[]){{0, 30}, {40, 50}, {100, 120}}, 3));

   CHECK(agc_span_list_clip(&out, 25, 105, &a));
   CHECK(spans_equal(&a, (struct agc_span[]){{25, 30}, {40, 50}, {100, 105}}, 3));
   CHECK(agc_span_list_clip(&out, 30, 40, &a) && a.count == 0);

   agc_span_list_fini(&a);
   agc_span_list_fini(&b);
   agc_span_list_fini(&out);
   CHECK(a.spans == NULL && a.capacity == 0);
}

static void
test_range_map(void)
{
   struct agc_range_map map = {0};
   struct agc_span_list gaps = {0};
   int tags[4];
   void *data = NULL;

   CHECK(!agc_range_map_insert(&map, 100, 100, &tags[0]));
   CHECK(agc_range_map_insert(&map, 300, 400, &tags[2]));
   CHECK(agc_range_map_insert(&map, 100, 200, &tags[0]));
   CHECK(agc_range_map_insert(&map, 200, 300, &tags[1]));
   CHECK(agc_range_map_insert(&map, 500, 600, &tags[3]));
   CHECK_EQ(map.count, 4);
   for (uint32_t i = 1; i < map.count; i++)
      CHECK(map.ranges[i - 1].end <= map.ranges[i].start);

   CHECK(!agc_range_map_insert(&map, 150, 160, NULL));
   CHECK(!agc_range_map_insert(&map, 50, 101, NULL));
   CHECK(!agc_range_map_insert(&map, 399, 500, NULL));
   CHECK(!agc_range_map_insert(&map, 0, 1000, NULL));
   CHECK_EQ(map.count, 4);

   CHECK(agc_range_map_find(&map, 99) == NULL);
   CHECK(agc_range_map_find(&map, 100)->data == &tags[0]);
   CHECK(agc_range_map_find(&map, 199)->data == &tags[0]);
   CHECK(agc_range_map_find(&map, 200)->data == &tags[1]);
   CHECK(agc_range_map_find(&map, 400) == NULL);
   CHECK(agc_range_map_find(&map, 599)->data == &tags[3]);
   CHECK(agc_range_map_find(&map, 600) == NULL);
   CHECK_EQ(agc_range_map_lower_bound(&map, 0), 0);
   CHECK_EQ(agc_range_map_lower_bound(&map, 400), 3);
   CHECK_EQ(agc_range_map_lower_bound(&map, 600), 4);

   CHECK(agc_range_map_covers(&map, 100, 400));
   CHECK(agc_range_map_covers(&map, 150, 350));
   CHECK(agc_range_map_covers(&map, 399, 400));
   CHECK(!agc_range_map_covers(&map, 100, 401));
   CHECK(!agc_range_map_covers(&map, 99, 200));
   CHECK(!agc_range_map_covers(&map, 350, 550));
   CHECK(!agc_range_map_covers(&map, 400, 500));
   CHECK(!agc_range_map_covers(&map, 150, 150));
   CHECK(agc_range_map_overlaps(&map, 0, 101) && !agc_range_map_overlaps(&map, 0, 100));
   CHECK(agc_range_map_overlaps(&map, 399, 500) && !agc_range_map_overlaps(&map, 400, 500));
   CHECK(!agc_range_map_overlaps(&map, 600, 700) && !agc_range_map_overlaps(&map, 150, 150));

   CHECK(agc_range_map_gaps(&map, 0, 700, &gaps));
   CHECK(spans_equal(&gaps, (struct agc_span[]){{0, 100}, {400, 500}, {600, 700}}, 3));
   gaps.count = 0;
   CHECK(agc_range_map_gaps(&map, 150, 350, &gaps) && gaps.count == 0);
   CHECK(agc_range_map_gaps(&map, 350, 450, &gaps));
   CHECK(spans_equal(&gaps, (struct agc_span[]){{400, 450}}, 1));

   CHECK(!agc_range_map_remove(&map, 150, &data));
   CHECK(agc_range_map_remove(&map, 200, &data) && data == &tags[1]);
   CHECK(!agc_range_map_covers(&map, 100, 400));
   CHECK(agc_range_map_find(&map, 250) == NULL);
   CHECK(agc_range_map_insert(&map, 200, 250, &tags[1]));
   CHECK(agc_range_map_insert(&map, 250, 300, &tags[1]));
   CHECK(agc_range_map_covers(&map, 100, 400));
   agc_range_map_fini(&map);

   const uint32_t many = 6000;
   for (uint32_t i = 0; i < many; i++) {
      uint32_t k = (i * 2654435761u) % many;
      while (agc_range_map_find(&map, BASE + (uint64_t)k * 2 * PAGE))
         k = (k + 1) % many;
      CHECK(agc_range_map_insert(&map, BASE + (uint64_t)k * 2 * PAGE, BASE + (uint64_t)k * 2 * PAGE + PAGE,
                                 (void *)(uintptr_t)(k + 1)));
   }
   CHECK_EQ(map.count, many);
   bool sorted = true;
   bool found = true;
   for (uint32_t i = 0; i < many; i++) {
      const struct agc_range *range = agc_range_map_find(&map, BASE + (uint64_t)i * 2 * PAGE + PAGE - 1);
      found &= range && range->data == (void *)(uintptr_t)(i + 1);
      found &= agc_range_map_find(&map, BASE + (uint64_t)i * 2 * PAGE + PAGE) == NULL;
      sorted &= map.ranges[i].start == BASE + (uint64_t)i * 2 * PAGE;
   }
   CHECK(sorted);
   CHECK(found);
   CHECK(!agc_range_map_covers(&map, BASE, BASE + 2 * PAGE));
   for (uint32_t i = 0; i < many; i += 2)
      CHECK(agc_range_map_remove(&map, BASE + (uint64_t)i * 2 * PAGE, NULL));
   CHECK_EQ(map.count, many / 2);
   CHECK(agc_range_map_find(&map, BASE) == NULL && agc_range_map_find(&map, BASE + 2 * PAGE) != NULL);
   agc_range_map_fini(&map);
   agc_span_list_fini(&gaps);
}

static bool
importable(const struct agc_span *span)
{
   return span->start < span->end && span->start % PAGE == 0 && span->end % PAGE == 0 &&
          span->start / WINDOW == (span->end - 1) / WINDOW && span->start >= AGC_GUEST_WINDOW_BASE &&
          span->end <= AGC_GUEST_WINDOW_END;
}

static void
test_import_plan(void)
{
   struct agc_span_list wanted = {0};
   const uint64_t w = BASE + 16 * WINDOW;

   struct agc_guest_range inside[] = {{w + 4 * PAGE, 8 * PAGE}};
   CHECK(agc_import_plan(inside, 1, &wanted));
   CHECK(spans_equal(&wanted, (struct agc_span[]){{w + 4 * PAGE, w + 12 * PAGE}}, 1));

   struct agc_guest_range straddling[] = {{w + WINDOW - 2 * PAGE, 5 * PAGE}};
   CHECK(agc_import_plan(straddling, 1, &wanted));
   CHECK(spans_equal(&wanted,
                     (struct agc_span[]){{w + WINDOW - 2 * PAGE, w + WINDOW}, {w + WINDOW, w + WINDOW + 3 * PAGE}},
                     2));

   struct agc_guest_range large[] = {{w + PAGE, 3 * WINDOW}};
   CHECK(agc_import_plan(large, 1, &wanted));
   CHECK(spans_equal(&wanted,
                     (struct agc_span[]){{w + PAGE, w + WINDOW},
                                         {w + WINDOW, w + 2 * WINDOW},
                                         {w + 2 * WINDOW, w + 3 * WINDOW},
                                         {w + 3 * WINDOW, w + 3 * WINDOW + PAGE}},
                     4));

   struct agc_guest_range exact[] = {{w, WINDOW}};
   CHECK(agc_import_plan(exact, 1, &wanted));
   CHECK(spans_equal(&wanted, (struct agc_span[]){{w, w + WINDOW}}, 1));

   struct agc_guest_range ragged[] = {{w + 0x1000, 3 * PAGE}};
   CHECK(agc_import_plan(ragged, 1, &wanted));
   CHECK(spans_equal(&wanted, (struct agc_span[]){{w + PAGE, w + 3 * PAGE}}, 1));

   struct agc_guest_range small[] = {{w + 0x1000, PAGE}, {w + 8 * PAGE + 1, PAGE - 2}};
   CHECK(agc_import_plan(small, 2, &wanted) && wanted.count == 0);

   struct agc_guest_range neighbours[] = {{w, 2 * PAGE}, {w + 2 * PAGE, 3 * PAGE}, {w + 8 * PAGE, PAGE}};
   CHECK(agc_import_plan(neighbours, 3, &wanted));
   CHECK(spans_equal(&wanted,
                     (struct agc_span[]){{w, w + 2 * PAGE}, {w + 2 * PAGE, w + 5 * PAGE}, {w + 8 * PAGE, w + 9 * PAGE}},
                     3));

   struct agc_guest_range sharing[] = {{w, PAGE + 0x2000}, {w + PAGE + 0x2000, 2 * PAGE - 0x2000}, {w + 3 * PAGE, PAGE}};
   CHECK(agc_import_plan(sharing, 3, &wanted));
   CHECK(spans_equal(&wanted, (struct agc_span[]){{w, w + 3 * PAGE}, {w + 3 * PAGE, w + 4 * PAGE}}, 2));

   struct agc_guest_range crossing[] = {{w + WINDOW - PAGE, PAGE + 0x1000}, {w + WINDOW + 0x1000, 2 * PAGE - 0x1000}};
   CHECK(agc_import_plan(crossing, 2, &wanted));
   CHECK(spans_equal(&wanted,
                     (struct agc_span[]){{w + WINDOW - PAGE, w + WINDOW}, {w + WINDOW, w + WINDOW + 2 * PAGE}}, 2));

   struct agc_guest_range outside[] = {
      {AGC_GUEST_WINDOW_BASE - 4 * PAGE, 6 * PAGE},
      {AGC_GUEST_WINDOW_END - 2 * PAGE, 8 * PAGE},
      {AGC_GUEST_WINDOW_END + WINDOW, PAGE},
   };
   CHECK(agc_import_plan(outside, 3, &wanted));
   CHECK(spans_equal(&wanted,
                     (struct agc_span[]){{AGC_GUEST_WINDOW_BASE, AGC_GUEST_WINDOW_BASE + 2 * PAGE},
                                         {AGC_GUEST_WINDOW_END - 2 * PAGE, AGC_GUEST_WINDOW_END}},
                     2));

   struct agc_guest_range empty[] = {{w, 0}, {w + PAGE, PAGE}, {w + 4 * PAGE, 0}};
   CHECK(agc_import_plan(empty, 3, &wanted));
   CHECK(spans_equal(&wanted, (struct agc_span[]){{w + PAGE, w + 2 * PAGE}}, 1));
   CHECK(agc_import_plan(NULL, 0, &wanted) && wanted.count == 0);

   struct agc_guest_range unordered[] = {{w + 8 * PAGE, PAGE}, {w, PAGE}};
   CHECK(!agc_import_plan(unordered, 2, &wanted));
   struct agc_guest_range overlapping[] = {{w, 4 * PAGE}, {w + 2 * PAGE, 4 * PAGE}};
   CHECK(!agc_import_plan(overlapping, 2, &wanted));

   enum { pieces = 3000 };
   struct agc_guest_range *title = calloc(pieces, sizeof(*title));
   CHECK(title != NULL);
   uint64_t cursor = BASE + 3 * PAGE;
   uint64_t visible_pages = 0;
   for (unsigned i = 0; title && i < pieces; i++) {
      uint64_t bytes = (1 + (i * 7919u) % 1500) * PAGE;
      title[i] = (struct agc_guest_range){(uintptr_t)cursor, (size_t)bytes};
      visible_pages += bytes / PAGE;
      cursor += bytes + ((i % 3) ? 0 : (1 + i % 5) * PAGE);
   }
   CHECK(title && agc_import_plan(title, pieces, &wanted));
   bool rules = true;
   bool ordered = true;
   uint64_t planned_pages = 0;
   for (uint32_t i = 0; i < wanted.count; i++) {
      rules &= importable(&wanted.spans[i]);
      ordered &= i == 0 || wanted.spans[i - 1].end <= wanted.spans[i].start;
      planned_pages += (wanted.spans[i].end - wanted.spans[i].start) / PAGE;
   }
   CHECK(rules);
   CHECK(ordered);
   CHECK_EQ(planned_pages, visible_pages);
   CHECK(wanted.count >= pieces);
   free(title);
   agc_span_list_fini(&wanted);
}

static void
test_import_diff(void)
{
   struct agc_span_list wanted = {0};
   struct agc_span_list missing = {0};
   struct agc_span_list stale = {0};
   struct agc_range_map imports = {0};
   const uint64_t w = BASE + 64 * WINDOW;

   fill_spans(&wanted, (struct agc_span[]){{w, w + 4 * PAGE}, {w + 8 * PAGE, w + 12 * PAGE}}, 2);
   CHECK(agc_import_diff(&wanted, &imports, &missing, &stale));
   CHECK(spans_equal(&missing, wanted.spans, wanted.count) && stale.count == 0);

   for (uint32_t i = 0; i < missing.count; i++)
      CHECK(agc_range_map_insert(&imports, missing.spans[i].start, missing.spans[i].end, NULL));
   CHECK(agc_import_diff(&wanted, &imports, &missing, &stale));
   CHECK(missing.count == 0 && stale.count == 0);

   fill_spans(&wanted, (struct agc_span[]){{w, w + 12 * PAGE}}, 1);
   CHECK(agc_import_diff(&wanted, &imports, &missing, &stale));
   CHECK(spans_equal(&missing, (struct agc_span[]){{w + 4 * PAGE, w + 8 * PAGE}}, 1) && stale.count == 0);
   CHECK(agc_range_map_insert(&imports, w + 4 * PAGE, w + 8 * PAGE, NULL));

   fill_spans(&wanted, (struct agc_span[]){{w, w + 3 * PAGE}, {w + 3 * PAGE, w + 6 * PAGE}, {w + 6 * PAGE, w + 12 * PAGE}}, 3);
   CHECK(agc_import_diff(&wanted, &imports, &missing, &stale));
   CHECK(missing.count == 0 && stale.count == 0);

   fill_spans(&wanted, (struct agc_span[]){{w, w + 4 * PAGE}, {w + 4 * PAGE, w + 7 * PAGE}, {w + 16 * PAGE, w + 20 * PAGE}}, 3);
   CHECK(agc_import_diff(&wanted, &imports, &missing, &stale));
   CHECK(spans_equal(&missing, (struct agc_span[]){{w + 16 * PAGE, w + 20 * PAGE}}, 1));
   CHECK(spans_equal(&stale, (struct agc_span[]){{w + 4 * PAGE, w + 8 * PAGE}, {w + 8 * PAGE, w + 12 * PAGE}}, 2));

   wanted.count = 0;
   CHECK(agc_import_diff(&wanted, &imports, &missing, &stale));
   CHECK(missing.count == 0 && stale.count == 3);

   agc_range_map_fini(&imports);
   CHECK(agc_range_map_insert(&imports, w + 2 * PAGE, w + 3 * PAGE, NULL));
   CHECK(agc_range_map_insert(&imports, w + 5 * PAGE, w + 6 * PAGE, NULL));
   fill_spans(&wanted, (struct agc_span[]){{w, w + 4 * PAGE}, {w + 4 * PAGE, w + 8 * PAGE}}, 2);
   CHECK(agc_import_diff(&wanted, &imports, &missing, &stale));
   CHECK(spans_equal(&missing,
                     (struct agc_span[]){{w, w + 2 * PAGE},
                                         {w + 3 * PAGE, w + 4 * PAGE},
                                         {w + 4 * PAGE, w + 5 * PAGE},
                                         {w + 6 * PAGE, w + 8 * PAGE}},
                     4));
   CHECK(stale.count == 0);

   agc_range_map_fini(&imports);
   agc_span_list_fini(&wanted);
   agc_span_list_fini(&missing);
   agc_span_list_fini(&stale);
}

static void
test_heap(void)
{
   struct agc_heap heap;
   uint64_t a = 0;
   uint64_t b = 0;
   uint64_t c = 0;
   uint64_t d = 0;
   const uint64_t start = AGC_DRIVER_AREA_BASE;
   const uint64_t unit = AGC_DRIVER_BLOCK_ALIGNMENT;

   CHECK(agc_heap_init(&heap, start, 16 * unit));
   CHECK(agc_heap_unused(&heap));
   CHECK(!agc_heap_alloc(&heap, 0, unit, &a));
   CHECK(!agc_heap_alloc(&heap, unit, 3, &a));
   CHECK(!agc_heap_alloc(&heap, 17 * unit, unit, &a));

   CHECK(agc_heap_alloc(&heap, unit, unit, &a) && a == start);
   CHECK(agc_heap_alloc(&heap, 3 * unit, unit, &b) && b == start + unit);
   CHECK(agc_heap_alloc(&heap, unit, 4 * unit, &c) && c == start + 4 * unit);
   CHECK(agc_heap_alloc(&heap, unit, 8 * unit, &d) && d == start + 8 * unit);
   CHECK(!agc_heap_unused(&heap));
   CHECK_EQ(heap.free_bytes, 10 * unit);
   CHECK_EQ(heap.holes.count, 2);

   uint64_t e = 0;
   CHECK(agc_heap_alloc(&heap, 2 * unit, unit, &e) && e == start + 5 * unit);
   CHECK(agc_heap_alloc(&heap, unit, unit, &e) && e == start + 7 * unit);
   CHECK(!agc_heap_alloc(&heap, 8 * unit, unit, &e));
   CHECK(agc_heap_alloc(&heap, 7 * unit, unit, &e) && e == start + 9 * unit);
   CHECK_EQ(heap.free_bytes, 0);
   CHECK_EQ(heap.holes.count, 0);
   CHECK(!agc_heap_alloc(&heap, unit, unit, &e));

   CHECK(!agc_heap_free(&heap, start - unit, unit));
   CHECK(!agc_heap_free(&heap, start + 15 * unit, 2 * unit));
   CHECK(!agc_heap_free(&heap, a, 0));
   CHECK(agc_heap_free(&heap, b, 3 * unit));
   CHECK(!agc_heap_free(&heap, b, 3 * unit));
   CHECK(!agc_heap_free(&heap, b + unit, unit));
   CHECK(!agc_heap_free(&heap, a, 2 * unit));
   CHECK(agc_heap_free(&heap, d, unit));
   CHECK_EQ(heap.holes.count, 2);
   CHECK(agc_heap_free(&heap, a, unit));
   CHECK_EQ(heap.holes.count, 2);
   CHECK(agc_heap_free(&heap, c, unit));
   CHECK_EQ(heap.holes.count, 2);
   CHECK(agc_heap_free(&heap, start + 5 * unit, 2 * unit));
   CHECK(agc_heap_free(&heap, start + 7 * unit, unit));
   CHECK_EQ(heap.holes.count, 1);
   CHECK(agc_heap_free(&heap, start + 9 * unit, 7 * unit));
   CHECK(agc_heap_unused(&heap));
   CHECK(heap.holes.count == 1 && heap.holes.spans[0].start == start && heap.holes.spans[0].end == start + 16 * unit);

   CHECK(agc_heap_alloc(&heap, 16 * unit, unit, &a) && a == start);
   CHECK(agc_heap_free(&heap, start + 4 * unit, unit));
   CHECK(agc_heap_alloc(&heap, unit, unit, &a) && a == start + 4 * unit);
   agc_heap_fini(&heap);

   CHECK(agc_heap_init(&heap, 0, 4096));
   uint64_t offsets[64];
   for (unsigned i = 0; i < 64; i++)
      CHECK(agc_heap_alloc(&heap, 64, 64, &offsets[i]) && offsets[i] == i * 64u);
   for (unsigned i = 0; i < 64; i += 2)
      CHECK(agc_heap_free(&heap, offsets[i], 64));
   CHECK_EQ(heap.holes.count, 32);
   CHECK(!agc_heap_alloc(&heap, 128, 64, &a));
   for (unsigned i = 1; i < 64; i += 2)
      CHECK(agc_heap_free(&heap, offsets[i], 64));
   CHECK(agc_heap_unused(&heap) && heap.holes.count == 1);
   CHECK(agc_heap_alloc(&heap, 100, 256, &a) && a == 0);
   CHECK(agc_heap_alloc(&heap, 100, 256, &b) && b == 256);
   CHECK(agc_heap_alloc(&heap, 28, 4, &c) && c == 100);
   agc_heap_fini(&heap);
}

static void
test_heap_at(void)
{
   struct agc_heap heap;
   const uint64_t start = AGC_RUNTIME_BASE;
   const uint64_t unit = AGC_DRIVER_BLOCK_ALIGNMENT;
   uint64_t a = 0;

   CHECK(agc_heap_init(&heap, start, 16 * unit));
   CHECK(!agc_heap_alloc_at(&heap, start, 0));
   CHECK(!agc_heap_alloc_at(&heap, start - unit, unit));
   CHECK(!agc_heap_alloc_at(&heap, start + 15 * unit, 2 * unit));
   CHECK(!agc_heap_alloc_at(&heap, UINT64_MAX - unit + 2, unit));

   CHECK(agc_heap_alloc_at(&heap, start + 4 * unit, 2 * unit));
   CHECK_EQ(heap.holes.count, 2);
   CHECK_EQ(heap.free_bytes, 14 * unit);
   CHECK(!agc_heap_alloc_at(&heap, start + 4 * unit, unit));
   CHECK(!agc_heap_alloc_at(&heap, start + 5 * unit, unit));
   CHECK(!agc_heap_alloc_at(&heap, start + 3 * unit, 2 * unit));
   CHECK(!agc_heap_alloc_at(&heap, start + 5 * unit, 2 * unit));
   CHECK(!agc_heap_alloc_at(&heap, start + 2 * unit, 6 * unit));
   CHECK_EQ(heap.free_bytes, 14 * unit);

   CHECK(agc_heap_alloc_at(&heap, start, unit));
   CHECK(agc_heap_alloc_at(&heap, start + 3 * unit, unit));
   CHECK(agc_heap_alloc_at(&heap, start + 15 * unit, unit));
   CHECK(agc_heap_alloc_at(&heap, start + 6 * unit, unit));
   CHECK(heap.holes.count == 2 && heap.holes.spans[0].start == start + unit &&
         heap.holes.spans[0].end == start + 3 * unit && heap.holes.spans[1].start == start + 7 * unit &&
         heap.holes.spans[1].end == start + 15 * unit);
   CHECK(agc_heap_alloc_at(&heap, start + unit, 2 * unit));
   CHECK_EQ(heap.holes.count, 1);
   CHECK(agc_heap_alloc(&heap, unit, unit, &a) && a == start + 7 * unit);

   CHECK(agc_heap_free(&heap, start + 4 * unit, 2 * unit));
   CHECK(agc_heap_alloc_at(&heap, start + 4 * unit, 2 * unit));
   CHECK(!agc_heap_free(&heap, start + 7 * unit, 2 * unit));
   CHECK(agc_heap_free(&heap, start + 3 * unit, unit));
   CHECK(agc_heap_free(&heap, start + 6 * unit, unit));
   CHECK(!agc_heap_alloc_at(&heap, start + 3 * unit, 4 * unit));
   CHECK(agc_heap_free(&heap, start + 4 * unit, 2 * unit));
   CHECK(agc_heap_alloc_at(&heap, start + 3 * unit, 4 * unit));
   agc_heap_fini(&heap);

   struct agc_heap area;
   struct agc_heap runtime;
   CHECK(agc_heap_init(&area, AGC_DRIVER_AREA_BASE, AGC_DRIVER_AREA_BYTES));
   CHECK(agc_heap_alloc_at(&area, AGC_RUNTIME_BASE, AGC_RUNTIME_BYTES));
   CHECK(agc_heap_init(&runtime, AGC_RUNTIME_BASE, AGC_RUNTIME_BYTES));
   CHECK_EQ(area.free_bytes, AGC_DRIVER_AREA_BYTES - AGC_RUNTIME_BYTES);
   bool outside = true;
   uint64_t blocks[64];
   for (unsigned i = 0; i < 64; i++) {
      CHECK(agc_heap_alloc(&area, unit, unit, &blocks[i]));
      outside &= !agc_va_in_runtime_range(blocks[i]) && !agc_va_in_runtime_range(blocks[i] + unit - 1) &&
                 agc_va_in_driver_area(blocks[i]);
   }
   CHECK(outside);
   CHECK(agc_heap_alloc(&area, 3 * AGC_RUNTIME_BYTES, unit, &a) && !agc_va_in_runtime_range(a));
   CHECK(!agc_heap_alloc_at(&area, AGC_RUNTIME_BASE, unit));
   CHECK(!agc_heap_free(&runtime, AGC_RUNTIME_BASE + AGC_RUNTIME_BYTES, unit));
   CHECK(!agc_heap_free(&runtime, AGC_RUNTIME_BASE, unit));

   CHECK(agc_heap_alloc_at(&runtime, AGC_RUNTIME_DESCRIPTOR_LUT_VA,
                           (AGC_RUNTIME_DESCRIPTOR_LUT_BYTES + unit - 1) & ~(unit - 1)));
   CHECK(agc_heap_alloc_at(&runtime, AGC_RUNTIME_MIP_STATS_VA, unit));
   CHECK(agc_heap_alloc_at(&runtime, AGC_RUNTIME_GDS_VA, 2 * unit));
   CHECK(!agc_heap_alloc_at(&runtime, AGC_RUNTIME_GDS_VA + unit, unit));
   CHECK(agc_heap_alloc_at(&runtime, AGC_RUNTIME_IMAGE_REPORTS_VA, AGC_RUNTIME_IMAGE_REPORTS_BYTES));
   CHECK(agc_heap_alloc_at(&runtime, AGC_RUNTIME_ATTRIBUTE_RING_VA, AGC_RUNTIME_ATTRIBUTE_RING_MAX_BYTES));
   CHECK(agc_heap_alloc_at(&runtime, AGC_RUNTIME_TABLES_END, AGC_RUNTIME_BASE + AGC_RUNTIME_BYTES -
                                                                 AGC_RUNTIME_TABLES_END));
   CHECK(!agc_heap_alloc_at(&runtime, AGC_RUNTIME_BASE + AGC_RUNTIME_BYTES, unit));
   CHECK(agc_heap_free(&runtime, AGC_RUNTIME_MIP_STATS_VA, unit));
   CHECK(agc_heap_alloc_at(&runtime, AGC_RUNTIME_MIP_STATS_VA, unit));
   agc_heap_fini(&runtime);
   agc_heap_fini(&area);
}

static void
test_subpool(void)
{
   struct agc_subpool data_pool;
   struct agc_subpool code_pool;
   int first_slab;
   int second_slab;
   int large_slab;
   int code_slab;
   uint64_t offset = 0;
   void *data = NULL;

   agc_subpool_init(&data_pool, AGC_SUBALLOC_SLAB_BYTES, AGC_DRIVER_BLOCK_ALIGNMENT);
   agc_subpool_init(&code_pool, AGC_SUBALLOC_SLAB_BYTES, AGC_DRIVER_BLOCK_ALIGNMENT);
   CHECK(agc_subpool_alloc(&data_pool, 256, 64, &offset) == NULL);
   CHECK_EQ(agc_subpool_growth(&data_pool, 256), AGC_SUBALLOC_SLAB_BYTES);
   CHECK_EQ(agc_subpool_growth(&data_pool, AGC_SUBALLOC_SLAB_BYTES + 1),
            AGC_SUBALLOC_SLAB_BYTES + AGC_DRIVER_BLOCK_ALIGNMENT);

   struct agc_subpool_slab *first = agc_subpool_grow(&data_pool, AGC_SUBALLOC_SLAB_BYTES, &first_slab);
   CHECK(first != NULL && first->data == &first_slab && first->bytes == AGC_SUBALLOC_SLAB_BYTES);
   CHECK(agc_subpool_owns(&data_pool, first) && !agc_subpool_owns(&code_pool, first));
   CHECK_EQ(agc_subpool_slab_count(&data_pool), 1);

   CHECK(agc_subpool_alloc(&data_pool, 256, 64, &offset) == first && offset == 0);
   CHECK(agc_subpool_alloc(&data_pool, 64, 64, &offset) == first && offset == 256);
   CHECK(agc_subpool_alloc(&data_pool, 4096, 4096, &offset) == first && offset == 4096);
   CHECK(agc_subpool_alloc(&data_pool, 128, 256, &offset) == first && offset == 512);
   CHECK(agc_subpool_alloc(&code_pool, 256, 256, &offset) == NULL);
   struct agc_subpool_slab *code = agc_subpool_grow(&code_pool, AGC_SUBALLOC_SLAB_BYTES, &code_slab);
   CHECK(code != NULL && code != first);
   CHECK(agc_subpool_alloc(&code_pool, 256, 256, &offset) == code && offset == 0);

   CHECK(agc_subpool_alloc(&data_pool, AGC_SUBALLOC_SLAB_BYTES, 64, &offset) == NULL);
   struct agc_subpool_slab *second = agc_subpool_grow(&data_pool, AGC_SUBALLOC_SLAB_BYTES, &second_slab);
   CHECK(second != NULL && agc_subpool_slab_count(&data_pool) == 2);
   CHECK(agc_subpool_alloc(&data_pool, AGC_SUBALLOC_SLAB_BYTES, 64, &offset) == second && offset == 0);
   CHECK(agc_subpool_alloc(&data_pool, 64, 64, &offset) == first && offset == 320);

   const uint64_t large_bytes = agc_subpool_growth(&data_pool, 3 * AGC_SUBALLOC_SLAB_BYTES + 5);
   struct agc_subpool_slab *large = agc_subpool_grow(&data_pool, large_bytes, &large_slab);
   CHECK(large != NULL && large->bytes == 3 * AGC_SUBALLOC_SLAB_BYTES + AGC_DRIVER_BLOCK_ALIGNMENT);
   CHECK(agc_heap_alloc(&large->heap, 3 * AGC_SUBALLOC_SLAB_BYTES + 64, 64, &offset) && offset == 0);

   CHECK(!agc_subpool_free(&code_pool, first, 0, 256));
   CHECK(!agc_subpool_free(&data_pool, first, 320, 256));
   CHECK(!agc_subpool_free(&data_pool, first, AGC_SUBALLOC_SLAB_BYTES - 64, 128));
   CHECK(agc_subpool_free(&data_pool, first, 0, 256));
   CHECK(!agc_subpool_free(&data_pool, first, 0, 256));
   CHECK(agc_subpool_alloc(&data_pool, 192, 64, &offset) == first && offset == 0);
   CHECK(agc_subpool_alloc(&data_pool, 64, 64, &offset) == first && offset == 192);

   CHECK(!agc_subpool_shrink(&data_pool, second, &data));
   CHECK(agc_subpool_free(&data_pool, second, 0, AGC_SUBALLOC_SLAB_BYTES));
   CHECK(agc_subpool_shrink(&data_pool, second, &data) && data == &second_slab);
   CHECK(!agc_subpool_owns(&data_pool, second) && agc_subpool_slab_count(&data_pool) == 2);
   CHECK(!agc_subpool_shrink(&data_pool, second, &data));

   struct agc_subpool_slab *reused = agc_subpool_grow(&data_pool, AGC_SUBALLOC_SLAB_BYTES, &second_slab);
   CHECK(reused != NULL && agc_subpool_slab_count(&data_pool) == 3 && data_pool.count == 3);
   CHECK(data_pool.slabs[1] == reused);

   CHECK(agc_subpool_free(&data_pool, large, 0, 3 * AGC_SUBALLOC_SLAB_BYTES + 64));
   CHECK(agc_subpool_shrink(&data_pool, large, &data) && data == &large_slab);
   CHECK_EQ(data_pool.count, 2);

   agc_subpool_fini(&data_pool);
   agc_subpool_fini(&code_pool);
   CHECK(data_pool.slabs == NULL && data_pool.count == 0);
}

#define ARENA_BASE (BASE + 256 * MIB)
#define ARENA_PAGES 12000u
#define ARENA_BYTES ((uint64_t)ARENA_PAGES * AGC_WRITE_PAGE_BYTES)

static bool arena_written[ARENA_PAGES];
static uint64_t arena_unwatchable_from = UINT64_MAX;
static unsigned arena_collect_calls;

static bool
arena_collect_writes(uintptr_t address, size_t bytes, void **pages, size_t *count, bool clear)
{
   arena_collect_calls++;
   const uint64_t end = (uint64_t)address + bytes;
   if (address < ARENA_BASE || end > ARENA_BASE + ARENA_BYTES || end > arena_unwatchable_from)
      return false;
   const size_t capacity = *count;
   size_t stored = 0;
   for (uint64_t page = address & ~(AGC_WRITE_PAGE_BYTES - 1); page < end && stored < capacity;
        page += AGC_WRITE_PAGE_BYTES) {
      size_t index = (size_t)((page - ARENA_BASE) / AGC_WRITE_PAGE_BYTES);
      if (!arena_written[index])
         continue;
      pages[stored++] = (void *)(uintptr_t)page;
      if (clear)
         arena_written[index] = false;
   }
   *count = stored;
   return true;
}

static void
arena_write(uint64_t offset, uint64_t bytes)
{
   for (uint64_t page = offset / AGC_WRITE_PAGE_BYTES; page <= (offset + bytes - 1) / AGC_WRITE_PAGE_BYTES; page++)
      arena_written[page] = true;
}

static void
test_write_tracker(void)
{
   struct agc_write_tracker tracker;
   struct agc_span_list written = {0};
   const uint64_t page = AGC_WRITE_PAGE_BYTES;
   const uint64_t texture = ARENA_BASE + 40 * page + 0x80;
   const uint64_t texture_bytes = 20 * page;

   memset(arena_written, 0, sizeof(arena_written));
   agc_write_tracker_init(&tracker, arena_collect_writes, ARENA_BASE, ARENA_BYTES);

   CHECK_EQ(agc_write_tracker_collect(&tracker, texture, 0), 0);
   uint64_t first = agc_write_tracker_collect(&tracker, texture, texture_bytes);
   CHECK(first > 1);
   CHECK(agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, first));
   CHECK(!agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, 0));

   arena_write(45 * page + 12, 4);
   CHECK(agc_write_tracker_unchanged_since(&tracker, ARENA_BASE + 46 * page, 4 * page, first));
   CHECK(!agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, first));
   CHECK(!agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, first));
   CHECK(!agc_write_tracker_unchanged_since(&tracker, ARENA_BASE + 45 * page + 100, 8, first));
   uint64_t second = agc_write_tracker_collect(&tracker, texture, texture_bytes);
   CHECK(second > first);
   CHECK(agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, second));
   CHECK(!agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, first));

   arena_write(40 * page, 1);
   arena_write(50 * page, 2 * page);
   arena_write(59 * page + page - 1, 2);
   arena_write(70 * page, page);
   const struct agc_span ranges[] = {
      {texture, texture + texture_bytes},
      {ARENA_BASE + 64 * page, ARENA_BASE + 64 * page},
      {ARENA_BASE + 68 * page + 8, ARENA_BASE + 70 * page + 16},
      {ARENA_BASE + 80 * page, ARENA_BASE + 84 * page},
   };
   uint64_t third = agc_write_tracker_written_since(&tracker, ranges, 4, second, &written);
   CHECK(third > second);
   CHECK(spans_equal(&written,
                     (struct agc_span[]){{texture, ARENA_BASE + 41 * page},
                                         {ARENA_BASE + 50 * page, ARENA_BASE + 52 * page},
                                         {ARENA_BASE + 59 * page, texture + texture_bytes},
                                         {ARENA_BASE + 70 * page, ARENA_BASE + 70 * page + 16}},
                     4));
   CHECK(agc_write_tracker_written_since(&tracker, ranges, 4, third, &written) > third);
   CHECK(written.count == 0);
   agc_write_tracker_written_since(&tracker, ranges, 4, first, &written);
   CHECK(spans_equal(&written,
                     (struct agc_span[]){{texture, ARENA_BASE + 41 * page},
                                         {ARENA_BASE + 45 * page, ARENA_BASE + 46 * page},
                                         {ARENA_BASE + 50 * page, ARENA_BASE + 52 * page},
                                         {ARENA_BASE + 59 * page, texture + texture_bytes},
                                         {ARENA_BASE + 70 * page, ARENA_BASE + 70 * page + 16}},
                     5));

   uint64_t before = agc_write_tracker_collect(&tracker, texture, texture_bytes);
   arena_write(47 * page, page);
   agc_write_tracker_absorb(&tracker, texture, texture_bytes);
   CHECK(agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, before));
   arena_write(48 * page, page);
   CHECK(!agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, before));

   CHECK_EQ(agc_write_tracker_collect(&tracker, ARENA_BASE - page, 2 * page), 0);
   CHECK_EQ(agc_write_tracker_collect(&tracker, ARENA_BASE + ARENA_BYTES - page, 2 * page), 0);
   CHECK(!agc_write_tracker_unchanged_since(&tracker, ARENA_BASE - page, page, before));
   const struct agc_span unwatched[] = {{ARENA_BASE - 3 * page + 5, ARENA_BASE - page - 7}};
   agc_write_tracker_written_since(&tracker, unwatched, 1, before, &written);
   CHECK(spans_equal(&written, unwatched, 1));
   CHECK(!agc_write_tracker_unchanged_since(&tracker, ARENA_BASE, page, before));

   memset(arena_written, 0, sizeof(arena_written));
   uint64_t clean = agc_write_tracker_collect(&tracker, ARENA_BASE, ARENA_BYTES);
   CHECK(clean != 0);
   for (unsigned i = 0; i < ARENA_PAGES; i++)
      arena_written[i] = true;
   arena_collect_calls = 0;
   const struct agc_span whole[] = {{ARENA_BASE, ARENA_BASE + ARENA_BYTES}};
   agc_write_tracker_written_since(&tracker, whole, 1, clean, &written);
   CHECK(spans_equal(&written, whole, 1));
   CHECK_EQ(arena_collect_calls, ARENA_PAGES / AGC_WRITE_COLLECT_PAGES + 1);
   bool collected = true;
   for (unsigned i = 0; i < ARENA_PAGES; i++)
      collected &= !arena_written[i];
   CHECK(collected);

   uint64_t settled = agc_write_tracker_collect(&tracker, ARENA_BASE, ARENA_BYTES);
   for (unsigned i = 0; i < ARENA_PAGES; i += 3)
      arena_written[i] = true;
   agc_write_tracker_written_since(&tracker, whole, 1, settled, &written);
   CHECK_EQ(written.count, ARENA_PAGES / 3);
   CHECK(written.count && written.spans[1].start == ARENA_BASE + 3 * page && written.spans[1].end == ARENA_BASE + 4 * page);

   uint64_t watched = agc_write_tracker_collect(&tracker, texture, texture_bytes);
   CHECK(watched != 0);
   arena_unwatchable_from = ARENA_BASE + 50 * page;
   CHECK_EQ(agc_write_tracker_collect(&tracker, texture, texture_bytes), 0);
   CHECK(!agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, watched));
   arena_unwatchable_from = UINT64_MAX;
   CHECK(!agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, watched));
   CHECK(!agc_write_tracker_unchanged_since(&tracker, ARENA_BASE + 52 * page, page, watched));
   uint64_t again = agc_write_tracker_collect(&tracker, texture, texture_bytes);
   CHECK(again > watched && agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, again));

   agc_write_tracker_fini(&tracker);

   agc_write_tracker_init(&tracker, NULL, 0, 0);
   CHECK_EQ(agc_write_tracker_collect(&tracker, texture, texture_bytes), 0);
   CHECK(!agc_write_tracker_unchanged_since(&tracker, texture, texture_bytes, 1));
   agc_write_tracker_written_since(&tracker, ranges, 1, 0, &written);
   CHECK(spans_equal(&written, ranges, 1));
   agc_write_tracker_fini(&tracker);
   agc_span_list_fini(&written);
}

static void
quiesce(void *user)
{
   (void)user;
}

static void
test_uninitialised(void)
{
   char error[128] = "";
   struct agc_mem mem;
   struct agc_memory_stats stats;
   struct agc_span_list written = {0};
   const struct agc_span range = {BASE, BASE + PAGE};

   CHECK(!agc_memory_sync(error, sizeof(error)) && strstr(error, "not initialised") != NULL);
   CHECK(strstr(agc_memory_last_error(), "not initialised") != NULL);
   CHECK(!agc_memory_imported(BASE, PAGE));
   CHECK(agc_span_list_append(&written, 1, 2));
   CHECK(agc_memory_release_stale(&written) == 0 && written.count == 0);
   CHECK_EQ(agc_memory_release_stale(NULL), 0);
   CHECK(!agc_mem_alloc(0x10000, 0, &mem) && mem.cpu == NULL && mem.va == 0);
   CHECK(!agc_mem_alloc(0, 0, &mem) && !agc_mem_alloc(0x10000, 0x80, &mem));
   CHECK(!agc_mem_suballoc(256, 64, AGC_BO_EXECUTE, &mem) && mem.owner == NULL);
   CHECK(!agc_mem_suballoc(256, 48, 0, &mem) && !agc_mem_suballoc(256, 0x20000, 0, &mem));
   CHECK(!agc_mem_alloc_at(AGC_RUNTIME_BASE + 0x1000, 0x1000, 0, &mem));
   CHECK(strstr(agc_memory_last_error(), "64 KiB aligned") != NULL);
   CHECK(!agc_mem_alloc_at(AGC_RUNTIME_BASE + AGC_RUNTIME_BYTES, 0x10000, 0, &mem));
   CHECK(strstr(agc_memory_last_error(), "outside the runtime range") != NULL);
   CHECK(!agc_mem_alloc_at(AGC_RUNTIME_BASE + AGC_RUNTIME_BYTES - 0x10000, 0x10001, 0, &mem));
   CHECK(strstr(agc_memory_last_error(), "outside the runtime range") != NULL);
   CHECK(!agc_mem_alloc_at(AGC_RUNTIME_BASE - 0x10000, 0x10000, 0, &mem));
   CHECK(strstr(agc_memory_last_error(), "outside the runtime range") != NULL);
   CHECK(!agc_mem_alloc_at(0, 0x10000, 0, &mem) && !agc_mem_alloc_at(AGC_RUNTIME_BASE, 0, 0, &mem));
   CHECK(!agc_mem_alloc_at(AGC_RUNTIME_BASE, 0x10000, 0x80, &mem));
   CHECK(!agc_mem_alloc_at(AGC_RUNTIME_BASE + AGC_RUNTIME_BYTES - 0x10000, UINT64_MAX, 0, &mem));
   CHECK(!agc_mem_alloc_at(AGC_RUNTIME_BASE, 0x10000, 0, &mem));
   CHECK(strstr(agc_memory_last_error(), "not initialised") != NULL && mem.owner == NULL);
   agc_mem_free(&mem);
   agc_mem_subfree(&mem);
   CHECK(!agc_mirror_detach(BASE, PAGE, &mem) && !agc_mirror_attach(BASE, PAGE));
   CHECK(!agc_mirror_detach(AGC_MIRROR_WINDOW_BASE, PAGE, &mem));
   CHECK(strstr(agc_memory_last_error(), "guest window") != NULL);
   CHECK(!agc_mirror_detached(BASE, PAGE) && !agc_mirror_write(BASE, error, 4));
   CHECK_EQ(agc_writes_collect(BASE, PAGE), 0);
   CHECK(!agc_writes_unchanged_since(BASE, PAGE, 1));
   agc_writes_absorb(BASE, PAGE);
   CHECK_EQ(agc_writes_written_since(&range, 1, 0, &written), 0);
   CHECK(spans_equal(&written, &range, 1));
   agc_span_list_fini(&written);
   agc_memory_get_stats(&stats);
   CHECK(stats.imports == 0 && stats.driver_blocks == 0 && !stats.lost);

   error[0] = 0;
   CHECK(!agc_memory_init(NULL, error, sizeof(error)) && error[0] != 0);
   struct agc_memory_config config = {.quiesce = quiesce};
   error[0] = 0;
   CHECK(!agc_memory_init(&config, error, sizeof(error)) && strstr(error, "backend") != NULL);
   agc_memory_fini();
   CHECK(!agc_memory_sync(NULL, 0));
}

int
main(void)
{
   test_layout();
   test_span_lists();
   test_range_map();
   test_import_plan();
   test_import_diff();
   test_heap();
   test_heap_at();
   test_subpool();
   test_write_tracker();
   test_uninitialised();
   return TEST_RESULT();
}
