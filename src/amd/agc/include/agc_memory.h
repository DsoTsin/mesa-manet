#ifndef AGC_MEMORY_H
#define AGC_MEMORY_H

#include <xmmintrin.h>

#include "agc_anyps5.h"
#include "agc_driver_ops.h"
#include "agc_runtime.h"
#include "agc_types.h"

#define AGC_GUEST_WINDOW_BASE UINT64_C(0x200000000)
#define AGC_GUEST_WINDOW_END UINT64_C(0xfc00000000)
#define AGC_GUEST_WINDOW_BYTES (AGC_GUEST_WINDOW_END - AGC_GUEST_WINDOW_BASE)
#define AGC_MIRROR_BIT 40
#define AGC_MIRROR_OFFSET (UINT64_C(1) << AGC_MIRROR_BIT)
#define AGC_MIRROR_WINDOW_BASE (AGC_GUEST_WINDOW_BASE + AGC_MIRROR_OFFSET)
#define AGC_MIRROR_WINDOW_END (AGC_GUEST_WINDOW_END + AGC_MIRROR_OFFSET)
#define AGC_DRIVER_AREA_BASE UINT64_C(0x20000000000)
#define AGC_DRIVER_AREA_BYTES (UINT64_C(1) << 38)
#define AGC_DRIVER_AREA_END (AGC_DRIVER_AREA_BASE + AGC_DRIVER_AREA_BYTES)

#define AGC_GUEST_PAGE_BYTES UINT64_C(0x4000)
#define AGC_GPU_PAGE_BYTES UINT64_C(0x1000)
#define AGC_IMPORT_WINDOW_BYTES (UINT64_C(8) << 20)
#define AGC_DRIVER_BLOCK_ALIGNMENT UINT64_C(0x10000)
#define AGC_SUBALLOC_GRANULE_BYTES UINT64_C(64)
#define AGC_SUBALLOC_SLAB_BYTES (UINT64_C(4) << 20)
#define AGC_MIRROR_SLAB_BYTES (UINT64_C(2) << 20)


#define AGC_WRITE_PAGE_BYTES UINT64_C(0x1000)
#define AGC_WRITE_BLOCK_PAGES 16
#define AGC_WRITE_COLLECT_PAGES 4096

_Static_assert(AGC_GUEST_WINDOW_END <= AGC_MIRROR_OFFSET, "the mirror bit is clear in guest addresses");
_Static_assert(AGC_MIRROR_WINDOW_BASE == UINT64_C(0x10200000000), "mirror window base");
_Static_assert(AGC_MIRROR_WINDOW_END == UINT64_C(0x1fc00000000), "mirror window end");
_Static_assert(AGC_MIRROR_WINDOW_END <= AGC_DRIVER_AREA_BASE, "the driver area follows the mirror window");
_Static_assert(AGC_GUEST_WINDOW_BASE % AGC_IMPORT_WINDOW_BYTES == 0, "guest window base");
_Static_assert(AGC_GUEST_WINDOW_END % AGC_IMPORT_WINDOW_BYTES == 0, "guest window end");
_Static_assert(AGC_IMPORT_WINDOW_BYTES % AGC_GUEST_PAGE_BYTES == 0, "import window");
_Static_assert(AGC_GUEST_PAGE_BYTES % AGC_GPU_PAGE_BYTES == 0, "guest page");
_Static_assert(AGC_DRIVER_AREA_BASE % AGC_DRIVER_BLOCK_ALIGNMENT == 0, "driver area base");
_Static_assert(AGC_DRIVER_AREA_BYTES % AGC_DRIVER_BLOCK_ALIGNMENT == 0, "driver area size");
_Static_assert(AGC_RUNTIME_BASE >= AGC_DRIVER_AREA_BASE && AGC_RUNTIME_BYTES != 0 &&
                  AGC_RUNTIME_BYTES <= AGC_DRIVER_AREA_END - AGC_RUNTIME_BASE,
               "the runtime tables lie in the driver area");
_Static_assert(AGC_RUNTIME_BASE % AGC_DRIVER_BLOCK_ALIGNMENT == 0 &&
                  AGC_RUNTIME_BYTES % AGC_DRIVER_BLOCK_ALIGNMENT == 0,
               "the runtime tables are whole driver blocks");
_Static_assert(AGC_RUNTIME_TABLES_END <= AGC_RUNTIME_BASE + AGC_RUNTIME_BYTES, "runtime tables");

struct agc_span {
   uint64_t start;
   uint64_t end;
};

struct agc_span_list {
   struct agc_span *spans;
   uint32_t count;
   uint32_t capacity;
};

struct agc_range {
   uint64_t start;
   uint64_t end;
   void *data;
};

struct agc_range_map {
   struct agc_range *ranges;
   uint32_t count;
   uint32_t capacity;
};

struct agc_heap {
   struct agc_span_list holes;
   uint64_t start;
   uint64_t end;
   uint64_t free_bytes;
};

struct agc_subpool_slab {
   struct agc_heap heap;
   uint64_t bytes;
   void *data;
};

struct agc_subpool {
   struct agc_subpool_slab **slabs;
   uint32_t count;
   uint32_t capacity;
   uint64_t slab_bytes;
   uint64_t slab_alignment;
};

typedef bool (*agc_collect_writes)(uintptr_t address, size_t bytes, void **pages, size_t *count, bool clear);

struct agc_write_block {
   uint64_t key;
   uint64_t newest;
   uint64_t pages[AGC_WRITE_BLOCK_PAGES];
};

struct agc_write_tracker {
   agc_collect_writes collect_writes;
   uint64_t arena_base;
   uint64_t arena_bytes;
   uint64_t generation;
   struct agc_write_block *blocks;
   uint32_t block_count;
   uint32_t block_capacity;
   void **pages;
   bool overflowed;
};

struct agc_mem {
   void *cpu;
   uint64_t va;
   uint64_t bytes;
   uint32_t flags;
   bool write_combined;
   void *owner;
   uint64_t extent;
};

struct agc_memory_config {
   struct agc_backend *backend;
   const struct agc_anyps5 *anyps5;
   void (*quiesce)(void *user);
   void *user;
   void (*resume)(void *user);
   void (*released)(void *user, uint64_t guest_va, uint64_t bytes);
};

struct agc_memory_stats {
   uint64_t imports;
   uint64_t imported_bytes;
   uint64_t stale_imports;
   uint64_t detached_bytes;
   uint64_t mirror_private_bytes;
   uint64_t zero_filled_bytes;
   uint64_t driver_blocks;
   uint64_t driver_block_bytes;
   uint64_t runtime_bytes;
   uint64_t suballoc_slabs;
   uint64_t suballoc_slab_bytes;
   uint64_t sync_passes;
   uint64_t imports_made;
   uint64_t imports_released;
   uint64_t imports_abandoned;
   uint64_t last_sync_refused_transient;
   uint64_t last_sync_refused_unimportable;
   uint64_t last_sync_unimported_bytes;
   bool lost;
};

static inline bool
agc_va_in_guest_window(uint64_t va)
{
   return va >= AGC_GUEST_WINDOW_BASE && va < AGC_GUEST_WINDOW_END;
}

static inline bool
agc_va_in_mirror_window(uint64_t va)
{
   return va >= AGC_MIRROR_WINDOW_BASE && va < AGC_MIRROR_WINDOW_END;
}

static inline bool
agc_va_in_driver_area(uint64_t va)
{
   return va >= AGC_DRIVER_AREA_BASE && va < AGC_DRIVER_AREA_END;
}

static inline bool
agc_va_in_runtime_range(uint64_t va)
{
   return va >= AGC_RUNTIME_BASE && va - AGC_RUNTIME_BASE < AGC_RUNTIME_BYTES;
}

static inline uint64_t
agc_mirror_va(uint64_t guest_va)
{
   return guest_va | AGC_MIRROR_OFFSET;
}

static inline uint64_t
agc_guest_va(uint64_t mirror_va)
{
   return mirror_va & ~AGC_MIRROR_OFFSET;
}

static inline void
agc_mem_flush_write_combining(void)
{
   _mm_sfence();
}

void agc_span_list_fini(struct agc_span_list *list);
bool agc_span_list_append(struct agc_span_list *list, uint64_t start, uint64_t end);
bool agc_span_list_extend(struct agc_span_list *list, uint64_t start, uint64_t end);
bool agc_span_list_subtract(const struct agc_span_list *from, const struct agc_span_list *cut,
                            struct agc_span_list *out);
bool agc_span_list_union(const struct agc_span_list *a, const struct agc_span_list *b,
                         struct agc_span_list *out);
bool agc_span_list_clip(const struct agc_span_list *list, uint64_t start, uint64_t end,
                        struct agc_span_list *out);
uint64_t agc_span_list_bytes(const struct agc_span_list *list);

void agc_range_map_fini(struct agc_range_map *map);
uint32_t agc_range_map_lower_bound(const struct agc_range_map *map, uint64_t address);
struct agc_range *agc_range_map_find(const struct agc_range_map *map, uint64_t address);
bool agc_range_map_overlaps(const struct agc_range_map *map, uint64_t start, uint64_t end);
bool agc_range_map_covers(const struct agc_range_map *map, uint64_t start, uint64_t end);
bool agc_range_map_insert(struct agc_range_map *map, uint64_t start, uint64_t end, void *data);
bool agc_range_map_remove(struct agc_range_map *map, uint64_t start, void **data);
bool agc_range_map_gaps(const struct agc_range_map *map, uint64_t start, uint64_t end,
                        struct agc_span_list *gaps);

bool agc_import_plan(const struct agc_guest_range *visible, size_t count, struct agc_span_list *wanted);
bool agc_import_diff(const struct agc_span_list *wanted, const struct agc_range_map *imports,
                     struct agc_span_list *missing, struct agc_span_list *stale);

bool agc_heap_init(struct agc_heap *heap, uint64_t start, uint64_t bytes);
void agc_heap_fini(struct agc_heap *heap);
bool agc_heap_alloc(struct agc_heap *heap, uint64_t bytes, uint64_t alignment, uint64_t *address);
bool agc_heap_alloc_at(struct agc_heap *heap, uint64_t address, uint64_t bytes);
bool agc_heap_free(struct agc_heap *heap, uint64_t address, uint64_t bytes);
bool agc_heap_unused(const struct agc_heap *heap);

void agc_subpool_init(struct agc_subpool *pool, uint64_t slab_bytes, uint64_t slab_alignment);
void agc_subpool_fini(struct agc_subpool *pool);
struct agc_subpool_slab *agc_subpool_alloc(struct agc_subpool *pool, uint64_t bytes, uint64_t alignment,
                                           uint64_t *offset);
uint64_t agc_subpool_growth(const struct agc_subpool *pool, uint64_t bytes);
struct agc_subpool_slab *agc_subpool_grow(struct agc_subpool *pool, uint64_t bytes, void *data);
bool agc_subpool_owns(const struct agc_subpool *pool, const struct agc_subpool_slab *slab);
bool agc_subpool_free(struct agc_subpool *pool, struct agc_subpool_slab *slab, uint64_t offset,
                      uint64_t bytes);
uint32_t agc_subpool_slab_count(const struct agc_subpool *pool);
bool agc_subpool_shrink(struct agc_subpool *pool, struct agc_subpool_slab *slab, void **data);

void agc_write_tracker_init(struct agc_write_tracker *tracker, agc_collect_writes collect_writes,
                            uint64_t arena_base, uint64_t arena_bytes);
void agc_write_tracker_fini(struct agc_write_tracker *tracker);
uint64_t agc_write_tracker_collect(struct agc_write_tracker *tracker, uint64_t address, uint64_t bytes);
bool agc_write_tracker_unchanged_since(struct agc_write_tracker *tracker, uint64_t address,
                                       uint64_t bytes, uint64_t generation);
void agc_write_tracker_absorb(struct agc_write_tracker *tracker, uint64_t address, uint64_t bytes);
uint64_t agc_write_tracker_written_since(struct agc_write_tracker *tracker, const struct agc_span *ranges,
                                         size_t count, uint64_t since, struct agc_span_list *written);

bool agc_memory_init(const struct agc_memory_config *config, char *error, size_t error_bytes);
void agc_memory_fini(void);
bool agc_memory_sync(char *error, size_t error_bytes);
bool agc_memory_guest_cached(void);
uint32_t agc_memory_guest_flags(void);
bool agc_memory_imported(uint64_t va, uint64_t bytes);
void *agc_memory_cpu_of(uint64_t va, uint64_t bytes);
uint32_t agc_memory_release_stale(struct agc_span_list *gone);
void agc_memory_get_stats(struct agc_memory_stats *stats);
const char *agc_memory_last_error(void);

bool agc_mem_alloc(uint64_t bytes, uint32_t flags, struct agc_mem *mem);
bool agc_mem_alloc_at(uint64_t va, uint64_t bytes, uint32_t flags, struct agc_mem *mem);
void agc_mem_free(struct agc_mem *mem);
bool agc_mem_suballoc(uint64_t bytes, uint64_t alignment, uint32_t flags, struct agc_mem *mem);
void agc_mem_subfree(struct agc_mem *mem);

bool agc_mirror_detach(uint64_t guest_va, uint64_t bytes, struct agc_mem *mem);
bool agc_mirror_attach(uint64_t guest_va, uint64_t bytes);
bool agc_mirror_detached(uint64_t guest_va, uint64_t bytes);
bool agc_mirror_write(uint64_t guest_va, const void *data, uint64_t bytes);

uint64_t agc_writes_collect(uint64_t address, uint64_t bytes);
bool agc_writes_watched(uint64_t address, uint64_t bytes);
bool agc_writes_collectable(uint64_t address, uint64_t bytes);
bool agc_writes_unchanged_since(uint64_t address, uint64_t bytes, uint64_t generation);
void agc_writes_absorb(uint64_t address, uint64_t bytes);
uint64_t agc_writes_written_since(const struct agc_span *ranges, size_t count, uint64_t since,
                                  struct agc_span_list *written);

#endif
