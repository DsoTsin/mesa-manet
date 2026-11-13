#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sid.h"
#include "util/macros.h"
#include "util/os_time.h"

#include "agc_anyps5.h"
#include "agc_driver_ops.h"
#include "agc_memory.h"

#include <windows.h>

#ifndef MEM_RESERVE_PLACEHOLDER
#define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_PRESERVE_PLACEHOLDER
#define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif

#define PROBE_OK 0
#define PROBE_FAILED 1
#define PROBE_UNCONFIRMED 2

#define PROBE_PAGE AGC_GUEST_PAGE_BYTES
#define PROBE_WINDOW AGC_IMPORT_WINDOW_BYTES
#define PROBE_GRANULARITY UINT64_C(0x10000)
#define PROBE_ARENA_WINDOWS 4
#define PROBE_ARENA_BYTES (PROBE_ARENA_WINDOWS * PROBE_WINDOW)
#define PROBE_ARENA_FIRST (AGC_GUEST_WINDOW_BASE + (UINT64_C(1) << 30))
#define PROBE_ARENA_STEP (UINT64_C(1) << 30)
#define PROBE_ARENA_ATTEMPTS 64u
#define PROBE_SECTION_BYTES (UINT64_C(1) << 20)

#define PROBE_STRADDLE_BYTES UINT64_C(0x40000)
#define PROBE_STRADDLE_SECTION_OFFSET UINT64_C(0)
#define PROBE_RELEASED_GUEST_OFFSET (2 * PROBE_WINDOW + 3 * PROBE_PAGE)
#define PROBE_RELEASED_BYTES (5 * PROBE_PAGE)
#define PROBE_RELEASED_SECTION_OFFSET UINT64_C(0x44000)
#define PROBE_SPARSE_GUEST_OFFSET (3 * PROBE_WINDOW)
#define PROBE_SPARSE_BYTES UINT64_C(0x10000)

#define PROBE_BLOCK_BYTES UINT64_C(0x10000)
#define PROBE_GUARD 0xa5a5a5a5u
#define PROBE_PATTERN_DWORDS 16u
#define PROBE_PATTERN_BYTES (PROBE_PATTERN_DWORDS * 4u)
#define PROBE_IMPORT_RECORDS 16u
#define PROBE_RELEASE_ROUNDS 8u

enum probe_copy {
   PROBE_COPY_BOTH,
   PROBE_COPY_DMA_DATA,
   PROBE_COPY_COPY_DATA
};

struct probe_mapping {
   const char *name;
   uint64_t address;
   uint64_t bytes;
   uint64_t section_offset;
   bool mapped;
};

struct probe_import {
   void *alias;
   void *view;
   uint64_t address;
   uint64_t bytes;
   bool live;
};

typedef PVOID(WINAPI *probe_virtual_alloc2)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void *, ULONG);
typedef PVOID(WINAPI *probe_map_view_of_file3)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void *,
                                               ULONG);
typedef BOOL(WINAPI *probe_unmap_view_of_file2)(HANDLE, PVOID, ULONG);

struct probe_guest {
   probe_virtual_alloc2 allocate;
   probe_map_view_of_file3 map;
   probe_unmap_view_of_file2 unmap;
   HANDLE section;
   uint64_t arena;
   struct probe_mapping straddle;
   struct probe_mapping released;
   struct probe_import imports[PROBE_IMPORT_RECORDS];
   uint64_t generation;
   uint64_t retiring_start;
   uint64_t retiring_end;
   agc_guest_import_release release;
   void *release_user;
   struct agc_guest_sparse_range sparse;
   bool sparse_registered;
   uint64_t sparse_generation;
   unsigned aliases_mapped;
   unsigned aliases_unmapped;
   unsigned imports_forgotten;
};

struct probe {
   struct agc_backend *be;
   const struct agc_driver_ops *ops;
   struct agc_hwq *queue;
   enum agc_queue_kind kind;
   enum probe_copy copy;
   bool sparse;
   int64_t timeout_ns;
   uint64_t last_fence;
   struct agc_mem ib;
   struct agc_mem readback;
   uint8_t *expected;
   unsigned quiesce_calls;
   unsigned resume_calls;
   unsigned released_calls;
   uint64_t released_va;
   uint64_t released_bytes;
};

static struct probe_guest guest;

static void probe_note(const char *name, const char *format, ...) AGC_PRINTF(2, 3);
static int probe_pass(const char *name, const char *format, ...) AGC_PRINTF(2, 3);
static int probe_fail(const char *name, const char *format, ...) AGC_PRINTF(2, 3);

static void
probe_print(const char *prefix, const char *name, const char *format, va_list args)
{
   printf("%s%s: ", prefix, name);
   vprintf(format, args);
   putchar('\n');
   fflush(stdout);
}

static void
probe_note(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   probe_print("", name, format, args);
   va_end(args);
}

static int
probe_pass(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   probe_print("PASS ", name, format, args);
   va_end(args);
   return PROBE_OK;
}

static int
probe_fail(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   probe_print("FAIL ", name, format, args);
   va_end(args);
   return PROBE_FAILED;
}

static enum agc_device_state
probe_device_state(struct probe *p)
{
   static const char *const names[] = {
      [AGC_DEVICE_ACTIVE] = "active",         [AGC_DEVICE_HUNG] = "hung",
      [AGC_DEVICE_PAGE_FAULT] = "page-fault", [AGC_DEVICE_REMOVED] = "removed",
      [AGC_DEVICE_UNKNOWN] = "unknown",
   };
   struct agc_wddm_device_state detail;
   char text[320];
   uint64_t fault_va = 0;
   enum agc_device_state state = p->ops->device_state(p->be, &fault_va);
   agc_wddm_device_state_detail(p->be, &detail);
   agc_wddm_device_state_format(&detail, text, sizeof(text));
   probe_note("device state", "%s, fault va 0x%" PRIx64 "; %s", names[state], fault_va, text);
   return state;
}

static bool
probe_guest_overlaps(const struct probe_mapping *mapping, uint64_t address, uint64_t bytes)
{
   return mapping->mapped && address < mapping->address + mapping->bytes && mapping->address < address + bytes;
}

static const struct probe_mapping *
probe_guest_containing(uint64_t address, uint64_t bytes)
{
   const struct probe_mapping *mappings[] = {&guest.straddle, &guest.released};
   for (unsigned i = 0; i < ARRAY_SIZE(mappings); i++) {
      const struct probe_mapping *mapping = mappings[i];
      if (mapping->mapped && address >= mapping->address &&
          address + bytes <= mapping->address + mapping->bytes)
         return mapping;
   }
   return NULL;
}

static unsigned
probe_guest_live_imports(uint64_t address, uint64_t bytes)
{
   unsigned live = 0;
   for (unsigned i = 0; i < PROBE_IMPORT_RECORDS; i++) {
      const struct probe_import *import = &guest.imports[i];
      live += import->live && address < import->address + import->bytes && import->address < address + bytes;
   }
   return live;
}

static void *
probe_map_import_alias(uintptr_t address, size_t bytes, uint32_t *refused)
{
   *refused = AGC_GUEST_IMPORT_MADE;
   if (bytes == 0 || address % PROBE_PAGE != 0 || bytes % PROBE_PAGE != 0) {
      *refused = AGC_GUEST_IMPORT_NOT_WHOLE_PAGES;
      return NULL;
   }
   if (address / PROBE_WINDOW != (address + bytes - 1) / PROBE_WINDOW) {
      *refused = AGC_GUEST_IMPORT_CROSSES_WINDOW;
      return NULL;
   }
   const struct probe_mapping *mapping = probe_guest_containing(address, bytes);
   if (!mapping || (address < guest.retiring_end && guest.retiring_start < address + bytes)) {
      *refused = AGC_GUEST_IMPORT_NOT_GPU_VISIBLE;
      return NULL;
   }
   struct probe_import *record = NULL;
   for (unsigned i = 0; i < PROBE_IMPORT_RECORDS && !record; i++) {
      if (!guest.imports[i].live)
         record = &guest.imports[i];
   }
   const uint64_t offset = mapping->section_offset + (address - mapping->address);
   const uint64_t lead = offset % PROBE_GRANULARITY;
   void *view = record ? guest.map(guest.section, GetCurrentProcess(), NULL, offset - lead, (SIZE_T)(lead + bytes), 0,
                                   PAGE_READWRITE, NULL, 0)
                       : NULL;
   if (!view) {
      *refused = AGC_GUEST_IMPORT_NOT_SECTION_MEMORY;
      return NULL;
   }
   *record = (struct probe_import){
      .alias = (uint8_t *)view + lead,
      .view = view,
      .address = address,
      .bytes = bytes,
      .live = true,
   };
   guest.aliases_mapped++;
   return record->alias;
}

static void
probe_unmap_import_alias(void *alias)
{
   for (unsigned i = 0; i < PROBE_IMPORT_RECORDS; i++) {
      struct probe_import *import = &guest.imports[i];
      if (!import->live || import->alias != alias)
         continue;
      guest.unmap(GetCurrentProcess(), import->view, 0);
      import->live = false;
      guest.aliases_unmapped++;
      return;
   }
}

static size_t
probe_import_alias_pages(const void *alias, uint64_t *pages, size_t capacity)
{
   (void)alias;
   (void)pages;
   (void)capacity;
   return 0;
}

static void *
probe_map_backing_alias(uintptr_t address, size_t bytes, bool *bounds_refused)
{
   (void)address;
   (void)bytes;
   *bounds_refused = false;
   return NULL;
}

static void
probe_unmap_backing_alias(void *alias)
{
   (void)alias;
}

static size_t
probe_gpu_visible(uintptr_t address, size_t bytes, struct agc_guest_range *pieces, size_t capacity)
{
   const struct probe_mapping *mappings[] = {&guest.straddle, &guest.released};
   size_t count = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(mappings); i++) {
      const struct probe_mapping *mapping = mappings[i];
      if (!probe_guest_overlaps(mapping, address, bytes))
         continue;
      const uint64_t start = MAX2(mapping->address, (uint64_t)address);
      const uint64_t end = MIN2(mapping->address + mapping->bytes, (uint64_t)address + bytes);
      if (count < capacity)
         pieces[count] = (struct agc_guest_range){(uintptr_t)start, (size_t)(end - start)};
      count++;
   }
   return count;
}

static void
probe_forget_import(uintptr_t address, size_t bytes)
{
   for (unsigned i = 0; i < PROBE_IMPORT_RECORDS; i++) {
      struct probe_import *import = &guest.imports[i];
      if (import->live && import->address == address && import->bytes == bytes) {
         import->live = false;
         guest.imports_forgotten++;
         return;
      }
   }
}

static uint64_t
probe_generation(void)
{
   return guest.generation;
}

static void
probe_arena_range(uintptr_t *base, size_t *bytes)
{
   *base = (uintptr_t)guest.arena;
   *bytes = (size_t)PROBE_ARENA_BYTES;
}

static bool
probe_write_watched(void)
{
   return false;
}

static bool
probe_collect_writes(uintptr_t address, size_t bytes, void **pages, size_t *count, bool clear)
{
   (void)address;
   (void)bytes;
   (void)pages;
   (void)count;
   (void)clear;
   return false;
}

static void
probe_set_import_release(agc_guest_import_release release, void *user)
{
   guest.release = release;
   guest.release_user = user;
}

static size_t
probe_sparse_ranges(uint64_t *generation, struct agc_guest_sparse_range *ranges, size_t capacity)
{
   if (generation)
      *generation = guest.sparse_generation;
   if (!guest.sparse_registered)
      return 0;
   if (capacity)
      ranges[0] = guest.sparse;
   return 1;
}

static const struct agc_anyps5 probe_host = {
   .map_import_alias = probe_map_import_alias,
   .unmap_import_alias = probe_unmap_import_alias,
   .import_alias_pages = probe_import_alias_pages,
   .map_backing_alias = probe_map_backing_alias,
   .unmap_backing_alias = probe_unmap_backing_alias,
   .gpu_visible = probe_gpu_visible,
   .forget_import = probe_forget_import,
   .generation = probe_generation,
   .gpu_generation = probe_generation,
   .arena_range = probe_arena_range,
   .write_watched = probe_write_watched,
   .collect_writes = probe_collect_writes,
   .set_import_release = probe_set_import_release,
   .sparse_ranges = probe_sparse_ranges,
};

static bool
probe_guest_split(uint64_t address, uint64_t bytes)
{
   MEMORY_BASIC_INFORMATION info;
   if (!VirtualQuery((void *)(uintptr_t)address, &info, sizeof(info)) ||
       !VirtualQuery(info.AllocationBase, &info, sizeof(info)) || info.State != MEM_RESERVE)
      return false;
   const uint64_t base = (uint64_t)(uintptr_t)info.BaseAddress;
   if (address != base) {
      if (!VirtualFree(info.BaseAddress, (SIZE_T)(address - base), MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) ||
          !VirtualQuery((void *)(uintptr_t)address, &info, sizeof(info)))
         return false;
   }
   if (info.State != MEM_RESERVE || info.RegionSize < bytes)
      return false;
   return info.RegionSize == bytes ||
          VirtualFree((void *)(uintptr_t)address, (SIZE_T)bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
}

static int
probe_guest_map(struct probe_mapping *mapping)
{
   void *address = (void *)(uintptr_t)mapping->address;
   if (!probe_guest_split(mapping->address, mapping->bytes))
      return probe_fail("guest", "%s: the placeholder at %p could not be split (error %lu)", mapping->name,
                        address, GetLastError());
   if (guest.map(guest.section, GetCurrentProcess(), address, mapping->section_offset, (SIZE_T)mapping->bytes,
                 MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, NULL, 0) != address)
      return probe_fail("guest", "%s: no section view at %p (error %lu)", mapping->name, address,
                        GetLastError());
   mapping->mapped = true;
   guest.generation++;
   probe_note("guest", "%s: 0x%" PRIx64 " bytes of section memory (offset 0x%" PRIx64 ") at CPU %p, "
                       "generation %" PRIu64,
              mapping->name, mapping->bytes, mapping->section_offset, address, guest.generation);
   return PROBE_OK;
}

static int
probe_guest_unmap(struct probe_mapping *mapping)
{
   guest.retiring_start = mapping->address;
   guest.retiring_end = mapping->address + mapping->bytes;
   unsigned rounds = 0;
   while (probe_guest_live_imports(mapping->address, mapping->bytes)) {
      bool progressed = guest.release &&
                        guest.release(guest.release_user, (uintptr_t)mapping->address, (size_t)mapping->bytes);
      if (!progressed || ++rounds == PROBE_RELEASE_ROUNDS) {
         guest.retiring_start = guest.retiring_end = 0;
         return probe_fail("release", "%s: %u imports still overlap after %u calls of the release callback",
                           mapping->name, probe_guest_live_imports(mapping->address, mapping->bytes), rounds + 1);
      }
   }
   if (!guest.unmap(GetCurrentProcess(), (void *)(uintptr_t)mapping->address, MEM_PRESERVE_PLACEHOLDER)) {
      guest.retiring_start = guest.retiring_end = 0;
      return probe_fail("release", "%s: the guest view could not be unmapped (error %lu)", mapping->name,
                        GetLastError());
   }
   mapping->mapped = false;
   guest.retiring_start = guest.retiring_end = 0;
   guest.generation++;
   probe_note("guest", "%s: unmapped after %u release calls, generation %" PRIu64, mapping->name, rounds,
              guest.generation);
   return PROBE_OK;
}

static int
probe_guest_open(void)
{
   HMODULE kernel = GetModuleHandleW(L"KernelBase.dll");
   guest.allocate = kernel ? (probe_virtual_alloc2)(void *)GetProcAddress(kernel, "VirtualAlloc2") : NULL;
   guest.map = kernel ? (probe_map_view_of_file3)(void *)GetProcAddress(kernel, "MapViewOfFile3") : NULL;
   guest.unmap = kernel ? (probe_unmap_view_of_file2)(void *)GetProcAddress(kernel, "UnmapViewOfFile2") : NULL;
   if (!guest.allocate || !guest.map || !guest.unmap)
      return probe_fail("guest", "VirtualAlloc2, MapViewOfFile3 or UnmapViewOfFile2 is missing");

   for (unsigned i = 0; i < PROBE_ARENA_ATTEMPTS && !guest.arena; i++) {
      uint64_t address = PROBE_ARENA_FIRST + i * PROBE_ARENA_STEP;
      if (address + PROBE_ARENA_BYTES > AGC_GUEST_WINDOW_END)
         break;
      if (guest.allocate(GetCurrentProcess(), (void *)(uintptr_t)address, (SIZE_T)PROBE_ARENA_BYTES,
                         MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, NULL, 0))
         guest.arena = address;
   }
   if (!guest.arena)
      return probe_fail("guest", "no placeholder of 0x%" PRIx64 " bytes inside the guest window (error %lu)",
                        (uint64_t)PROBE_ARENA_BYTES, GetLastError());

   guest.section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)PROBE_SECTION_BYTES,
                                      NULL);
   if (!guest.section)
      return probe_fail("guest", "no pagefile section (error %lu)", GetLastError());

   guest.generation = 1;
   guest.sparse_generation = 1;
   guest.straddle = (struct probe_mapping){
      .name = "straddling allocation",
      .address = guest.arena + PROBE_WINDOW - PROBE_STRADDLE_BYTES / 2,
      .bytes = PROBE_STRADDLE_BYTES,
      .section_offset = PROBE_STRADDLE_SECTION_OFFSET,
   };
   guest.released = (struct probe_mapping){
      .name = "released allocation",
      .address = guest.arena + PROBE_RELEASED_GUEST_OFFSET,
      .bytes = PROBE_RELEASED_BYTES,
      .section_offset = PROBE_RELEASED_SECTION_OFFSET,
   };
   probe_note("guest", "placeholder arena 0x%" PRIx64 "..0x%" PRIx64, guest.arena,
              guest.arena + PROBE_ARENA_BYTES);
   int result = probe_guest_map(&guest.straddle);
   return result == PROBE_OK ? probe_guest_map(&guest.released) : result;
}

static void
probe_quiesce(void *user)
{
   struct probe *p = user;
   p->quiesce_calls++;
   if (!p->queue || !p->last_fence || p->ops->fence_wait(p->queue, p->last_fence, p->timeout_ns))
      return;
   probe_fail("quiesce", "fence %" PRIu64 " not reached within %" PRId64 " ms, completed %" PRIu64,
              p->last_fence, p->timeout_ns / 1000000, p->ops->fence_completed(p->queue));
   probe_device_state(p);
   probe_note("teardown", "skipped: GPU completion is unconfirmed, the process exit releases everything");
   exit(PROBE_UNCONFIRMED);
}

static void
probe_resume(void *user)
{
   struct probe *p = user;
   p->resume_calls++;
}

static void
probe_released(void *user, uint64_t guest_va, uint64_t bytes)
{
   struct probe *p = user;
   p->released_calls++;
   p->released_va = guest_va;
   p->released_bytes = bytes;
}

static uint32_t
probe_pattern(uint32_t seed, uint32_t index)
{
   return seed + index * 0x01010101u;
}

static void
probe_fill_pattern(void *memory, uint32_t seed)
{
   uint32_t words[PROBE_PATTERN_DWORDS];
   for (uint32_t i = 0; i < PROBE_PATTERN_DWORDS; i++)
      words[i] = probe_pattern(seed, i);
   memcpy(memory, words, sizeof(words));
}

static void
probe_fill_guard(void *memory, uint64_t bytes)
{
   uint32_t *words = memory;
   for (uint64_t i = 0; i < bytes / sizeof(uint32_t); i++)
      words[i] = PROBE_GUARD;
}

static int
probe_compare(const char *name, const char *what, const void *memory, const void *expected, uint64_t bytes)
{
   const volatile uint8_t *seen = memory;
   const uint8_t *wanted = expected;
   for (uint64_t i = 0; i < bytes; i++) {
      uint8_t value = seen[i];
      if (value != wanted[i])
         return probe_fail(name, "%s: byte +0x%" PRIx64 " is 0x%02x, not 0x%02x", what, i, value, wanted[i]);
   }
   return PROBE_OK;
}

static int
probe_compare_pattern(const char *name, const char *what, const void *memory, uint32_t seed)
{
   uint8_t expected[PROBE_PATTERN_BYTES];
   probe_fill_pattern(expected, seed);
   return probe_compare(name, what, memory, expected, sizeof(expected));
}

static uint32_t *
probe_emit_write(uint32_t *out, uint64_t va, uint32_t seed)
{
   *out++ = PKT3(PKT3_WRITE_DATA, 2 + PROBE_PATTERN_DWORDS, 0);
   *out++ = S_371_DST_SEL(V_371_MEMORY) | S_371_WR_CONFIRM(V_371_WAIT_FOR_WRITE_CONFIRMATION) |
            S_371_ENGINE_SEL(V_371_MICRO_ENGINE);
   *out++ = (uint32_t)va;
   *out++ = (uint32_t)(va >> 32);
   for (uint32_t i = 0; i < PROBE_PATTERN_DWORDS; i++)
      *out++ = probe_pattern(seed, i);
   return out;
}

static uint32_t *
probe_emit_dma_data(struct probe *p, uint32_t *out, uint64_t source, uint64_t destination, uint32_t bytes)
{
   *out++ = PKT3(PKT3_DMA_DATA, 5, 0);
   *out++ = p->kind == AGC_QUEUE_GFX ? S_501_CP_SYNC(1) : 0;
   *out++ = (uint32_t)source;
   *out++ = (uint32_t)(source >> 32);
   *out++ = (uint32_t)destination;
   *out++ = (uint32_t)(destination >> 32);
   *out++ = S_506_BYTE_COUNT(bytes);
   return out;
}

static uint32_t *
probe_emit_copy_data(uint32_t *out, uint64_t source, uint64_t destination, uint32_t bytes)
{
   for (uint32_t done = 0; done < bytes; done += 8) {
      *out++ = PKT3(PKT3_COPY_DATA, 4, 0);
      *out++ = COPY_DATA_SRC_SEL(COPY_DATA_SRC_MEM) | COPY_DATA_DST_SEL(COPY_DATA_DST_MEM) | COPY_DATA_COUNT_SEL |
               COPY_DATA_WR_CONFIRM;
      *out++ = (uint32_t)(source + done);
      *out++ = (uint32_t)((source + done) >> 32);
      *out++ = (uint32_t)(destination + done);
      *out++ = (uint32_t)((destination + done) >> 32);
   }
   return out;
}

static uint32_t *
probe_emit_copy(struct probe *p, uint32_t *out, uint64_t source, uint64_t destination)
{
   if (p->copy == PROBE_COPY_COPY_DATA)
      return probe_emit_copy_data(out, source, destination, PROBE_PATTERN_BYTES);
   return probe_emit_dma_data(p, out, source, destination, PROBE_PATTERN_BYTES);
}

static int
probe_submit(struct probe *p, uint32_t *end, const char *name, const char *what, bool wait)
{
   uint32_t *words = p->ib.cpu;
   uint32_t dwords = (uint32_t)(end - words);
   uint32_t padded = (dwords + 7) & ~7u;
   while (dwords < padded) {
      uint32_t chunk = padded - dwords;
      if (chunk == 1) {
         words[dwords++] = PKT3_NOP_PAD;
      } else {
         words[dwords] = PKT3(PKT3_NOP, chunk - 2, 0);
         memset(&words[dwords + 1], 0, (chunk - 1) * sizeof(uint32_t));
         dwords += chunk;
      }
   }
   agc_mem_flush_write_combining();

   const struct agc_ib entry = {.va = p->ib.va, .dwords = padded, .flags = AGC_IB_MAIN};
   uint64_t fence = p->ops->submit(p->queue, &entry, 1);
   if (!fence) {
      probe_fail(name, "%s: the submission of %u dwords was refused", what, padded);
      return probe_device_state(p) == AGC_DEVICE_ACTIVE ? PROBE_FAILED : PROBE_UNCONFIRMED;
   }
   p->last_fence = fence;
   if (!wait) {
      probe_note(name, "%s: fence %" PRIu64 " submitted, %u dwords, not waited for", what, fence, padded);
      return PROBE_OK;
   }
   int64_t start = os_time_get_nano();
   if (!p->ops->fence_wait(p->queue, fence, p->timeout_ns)) {
      probe_fail(name, "%s: fence %" PRIu64 " not reached within %" PRId64 " ms, completed %" PRIu64, what, fence,
                 p->timeout_ns / 1000000, p->ops->fence_completed(p->queue));
      probe_device_state(p);
      return PROBE_UNCONFIRMED;
   }
   probe_note(name, "%s: fence %" PRIu64 " reached in %.3f ms, %u dwords", what, fence,
              (double)(os_time_get_nano() - start) / 1e6, padded);
   return PROBE_OK;
}

static int
probe_open(struct probe *p)
{
   p->be = agc_wddm_open();
   if (!p->be)
      return probe_fail("open", "no AMD adapter with a verified profile could be opened");
   p->ops = p->be->ops;
   struct agc_wddm_description description;
   agc_wddm_describe(p->be, &description);
   probe_note("open", "device %04x revision %02x, %u GPU VA bits, end 0x%" PRIx64 ", profile %s",
              description.adapter.device_id, description.adapter.revision, description.va_bits,
              description.va_end, description.adapter.profile ? description.adapter.profile : "none");
   return PROBE_OK;
}

static int
probe_init(struct probe *p)
{
   char error[256];
   const struct agc_memory_config config = {
      .backend = p->be,
      .anyps5 = &probe_host,
      .quiesce = probe_quiesce,
      .user = p,
      .resume = probe_resume,
      .released = probe_released,
   };
   agc_anyps5_set(&probe_host);
   if (!agc_memory_init(&config, error, sizeof(error)))
      return probe_fail("init", "%s", error);
   if (guest.release == NULL)
      return probe_fail("init", "the import release callback was not installed");
   probe_pass("init", "guest window 0x%" PRIx64 "..0x%" PRIx64 ", mirror window 0x%" PRIx64 "..0x%" PRIx64
                      " and driver area 0x%" PRIx64 "..0x%" PRIx64 " reserved, release callback installed",
              (uint64_t)AGC_GUEST_WINDOW_BASE, (uint64_t)AGC_GUEST_WINDOW_END, (uint64_t)AGC_MIRROR_WINDOW_BASE,
              (uint64_t)AGC_MIRROR_WINDOW_END, (uint64_t)AGC_DRIVER_AREA_BASE, (uint64_t)AGC_DRIVER_AREA_END);

   p->queue = p->ops->queue_create(p->be, p->kind);
   if (!p->queue)
      return probe_fail("queue", "no %s hardware queue", p->kind == AGC_QUEUE_GFX ? "graphics" : "compute");
   struct agc_wddm_queue_description queue;
   agc_wddm_queue_describe(p->queue, &queue);
   probe_note("queue", "%s on node %u, id %u, progress fence at GPU va 0x%" PRIx64,
              p->kind == AGC_QUEUE_GFX ? "graphics" : "compute", queue.node, queue.queue_id, queue.fence_gpu_va);
   if (agc_va_in_guest_window(queue.fence_gpu_va) || agc_va_in_mirror_window(queue.fence_gpu_va) ||
       agc_va_in_driver_area(queue.fence_gpu_va))
      return probe_fail("queue", "the progress fence was placed inside a reserved window");

   if (!agc_mem_alloc(PROBE_BLOCK_BYTES, AGC_BO_GL2_BYPASS, &p->ib) ||
       !agc_mem_alloc(PROBE_BLOCK_BYTES, AGC_BO_GL2_BYPASS, &p->readback))
      return probe_fail("driver memory", "%s", agc_memory_last_error());
   return PROBE_OK;
}

static int
probe_driver_memory(struct probe *p)
{
   static const char name[] = "(e) driver memory";
   struct agc_mem target;
   struct agc_mem small;
   struct agc_mem table;
   struct agc_mem code;
   if (!agc_mem_alloc(PROBE_BLOCK_BYTES, AGC_BO_GL2_BYPASS, &target))
      return probe_fail(name, "%s", agc_memory_last_error());
   if (!agc_mem_suballoc(0x180, 0x100, AGC_BO_GL2_BYPASS, &small) ||
       !agc_mem_suballoc(0x1000, 0, AGC_BO_GL2_BYPASS, &table) ||
       !agc_mem_suballoc(0x2340, 0x100, AGC_BO_EXECUTE, &code))
      return probe_fail(name, "suballocation: %s", agc_memory_last_error());
   probe_note(name, "IB 0x%" PRIx64 " (CPU %p, %s), target 0x%" PRIx64 " (%s), suballocations 0x%" PRIx64
                    " and 0x%" PRIx64 " (%s), code 0x%" PRIx64 " (%s)",
              p->ib.va, p->ib.cpu, p->ib.write_combined ? "write-combined" : "cached", target.va,
              target.write_combined ? "write-combined" : "cached", small.va, table.va,
              small.write_combined ? "write-combined" : "cached", code.va,
              code.write_combined ? "write-combined" : "cached");

   const struct agc_mem *all[] = {&p->ib, &p->readback, &target, &small, &table, &code};
   for (unsigned i = 0; i < ARRAY_SIZE(all); i++) {
      if (!all[i]->cpu || !agc_va_in_driver_area(all[i]->va) || !agc_va_in_driver_area(all[i]->va + all[i]->bytes - 1))
         return probe_fail(name, "0x%" PRIx64 "+0x%" PRIx64 " is not CPU-mapped driver area memory", all[i]->va,
                           all[i]->bytes);
      for (unsigned k = 0; k < i; k++) {
         if (all[i]->va < all[k]->va + all[k]->bytes && all[k]->va < all[i]->va + all[i]->bytes)
            return probe_fail(name, "0x%" PRIx64 " and 0x%" PRIx64 " overlap", all[i]->va, all[k]->va);
      }
   }
   if (small.va % 0x100 != 0 || code.va % 0x100 != 0 || table.va < small.va + 0x180 ||
       table.va - small.va >= AGC_SUBALLOC_SLAB_BYTES)
      return probe_fail(name, "suballocations 0x%" PRIx64 " and 0x%" PRIx64 " are misplaced", small.va, table.va);
   if (code.va < small.va + AGC_SUBALLOC_SLAB_BYTES && small.va < code.va + AGC_SUBALLOC_SLAB_BYTES)
      return probe_fail(name, "code 0x%" PRIx64 " shares a slab with data 0x%" PRIx64, code.va, small.va);

   uint8_t *expected = malloc((size_t)PROBE_BLOCK_BYTES);
   if (!expected)
      return probe_fail(name, "out of memory");
   probe_fill_guard(target.cpu, target.bytes);
   probe_fill_guard(expected, PROBE_BLOCK_BYTES);
   probe_fill_guard(small.cpu, 0x180);
   probe_fill_guard(table.cpu, 0x1000);
   memset(code.cpu, 0x5a, 0x2340);
   agc_mem_flush_write_combining();

   uint32_t *out = p->ib.cpu;
   out = probe_emit_write(out, target.va + 0x100, 0xe0000001u);
   out = probe_emit_write(out, target.va + target.bytes - PROBE_PATTERN_BYTES, 0xe0000002u);
   out = probe_emit_write(out, small.va + 0x40, 0xe0000003u);
   out = probe_emit_write(out, table.va + 0x1000 - PROBE_PATTERN_BYTES, 0xe0000004u);
   int result = probe_submit(p, out, name, "WRITE_DATA into a buffer of its own and two suballocations", true);
   if (result == PROBE_OK) {
      probe_fill_pattern(expected + 0x100, 0xe0000001u);
      probe_fill_pattern(expected + PROBE_BLOCK_BYTES - PROBE_PATTERN_BYTES, 0xe0000002u);
      result = probe_compare(name, "buffer of its own", target.cpu, expected, PROBE_BLOCK_BYTES);
   }
   if (result == PROBE_OK) {
      probe_fill_guard(expected, PROBE_BLOCK_BYTES);
      probe_fill_pattern(expected + 0x40, 0xe0000003u);
      result = probe_compare(name, "first suballocation", small.cpu, expected, 0x180);
   }
   if (result == PROBE_OK) {
      probe_fill_guard(expected, PROBE_BLOCK_BYTES);
      probe_fill_pattern(expected + 0x1000 - PROBE_PATTERN_BYTES, 0xe0000004u);
      result = probe_compare(name, "second suballocation", table.cpu, expected, 0x1000);
   }
   if (result == PROBE_OK) {
      memset(expected, 0x5a, 0x2340);
      result = probe_compare(name, "code suballocation", code.cpu, expected, 0x2340);
   }
   free(expected);
   if (result != PROBE_OK)
      return result;

   struct agc_memory_stats stats;
   agc_memory_get_stats(&stats);
   if (stats.driver_blocks != 3 || stats.suballoc_slabs != 2)
      return probe_fail(name, "%" PRIu64 " buffers and %" PRIu64 " slabs are live, not 3 and 2", stats.driver_blocks,
                        stats.suballoc_slabs);
   agc_mem_subfree(&small);
   agc_mem_subfree(&table);
   agc_mem_subfree(&code);
   agc_mem_free(&target);
   agc_memory_get_stats(&stats);
   if (stats.driver_blocks != 2)
      return probe_fail(name, "%" PRIu64 " buffers stay after the free, not 2", stats.driver_blocks);
   return probe_pass(name, "the GPU wrote 4 patterns into agc_mem_alloc and agc_mem_suballoc memory, guards "
                           "intact, code and data in separate slabs");
}

static int
probe_runtime_memory(struct probe *p)
{
   static const char name[] = "(e) runtime memory";
   struct agc_mem gds;
   struct agc_mem reports;
   struct agc_mem refused;
   struct agc_memory_stats stats;

   if (agc_va_in_runtime_range(p->ib.va) || agc_va_in_runtime_range(p->readback.va))
      return probe_fail(name, "agc_mem_alloc placed 0x%" PRIx64 " or 0x%" PRIx64 " in the runtime range", p->ib.va,
                        p->readback.va);
   if (!agc_mem_alloc_at(AGC_RUNTIME_GDS_VA, AGC_RUNTIME_GDS_BUFFER_BYTES, 0, &gds) ||
       !agc_mem_alloc_at(AGC_RUNTIME_IMAGE_REPORTS_VA, AGC_RUNTIME_IMAGE_REPORTS_BYTES, AGC_BO_GL2_BYPASS, &reports))
      return probe_fail(name, "%s", agc_memory_last_error());
   if (gds.va != AGC_RUNTIME_GDS_VA || reports.va != AGC_RUNTIME_IMAGE_REPORTS_VA || !gds.cpu || !reports.cpu ||
       gds.bytes < AGC_RUNTIME_GDS_BUFFER_BYTES || reports.bytes != AGC_RUNTIME_IMAGE_REPORTS_BYTES)
      return probe_fail(name, "fixed buffers at 0x%" PRIx64 "+0x%" PRIx64 " and 0x%" PRIx64 "+0x%" PRIx64, gds.va,
                        gds.bytes, reports.va, reports.bytes);
   if (agc_mem_alloc_at(AGC_RUNTIME_GDS_VA + 0x10000, 0x10000, 0, &refused))
      return probe_fail(name, "an allocation overlapping the GDS buffer was granted at 0x%" PRIx64, refused.va);
   probe_note(name, "overlap refused: %s", agc_memory_last_error());
   agc_memory_get_stats(&stats);
   if (stats.runtime_bytes != gds.bytes + reports.bytes)
      return probe_fail(name, "0x%" PRIx64 " runtime bytes are live, not 0x%" PRIx64, stats.runtime_bytes,
                        gds.bytes + reports.bytes);

   probe_fill_guard(gds.cpu, gds.bytes);
   probe_fill_guard(reports.cpu, reports.bytes);
   agc_mem_flush_write_combining();
   uint32_t *out = probe_emit_write(p->ib.cpu, AGC_RUNTIME_GDS_VA + 0x10000, 0xe1000001u);
   out = probe_emit_write(out, AGC_RUNTIME_IMAGE_REPORTS_VA + AGC_RUNTIME_IMAGE_REPORTS_BYTES - PROBE_PATTERN_BYTES,
                          0xe1000002u);
   int result = probe_submit(p, out, name, "WRITE_DATA to the fixed runtime addresses", true);
   if (result == PROBE_OK)
      result = probe_compare_pattern(name, "GDS buffer", (uint8_t *)gds.cpu + 0x10000, 0xe1000001u);
   if (result == PROBE_OK)
      result = probe_compare_pattern(name, "image reports",
                                     (uint8_t *)reports.cpu + AGC_RUNTIME_IMAGE_REPORTS_BYTES - PROBE_PATTERN_BYTES,
                                     0xe1000002u);
   if (result != PROBE_OK)
      return result;

   const bool write_combined = reports.write_combined;
   agc_mem_free(&gds);
   if (!agc_mem_alloc_at(AGC_RUNTIME_GDS_VA, AGC_RUNTIME_GDS_BUFFER_BYTES, 0, &gds) || gds.va != AGC_RUNTIME_GDS_VA)
      return probe_fail(name, "the GDS address was not available again after the free: %s", agc_memory_last_error());
   agc_mem_free(&gds);
   agc_mem_free(&reports);
   agc_memory_get_stats(&stats);
   if (stats.runtime_bytes != 0)
      return probe_fail(name, "0x%" PRIx64 " runtime bytes stay after the free", stats.runtime_bytes);
   return probe_pass(name, "buffers at 0x%" PRIx64 " and 0x%" PRIx64 " are exactly where asked (%s), an overlap is "
                           "refused, the GPU wrote both, and a freed address can be allocated again",
                     (uint64_t)AGC_RUNTIME_GDS_VA, (uint64_t)AGC_RUNTIME_IMAGE_REPORTS_VA,
                     write_combined ? "write-combined" : "cached");
}

static int
probe_sync(struct probe *p)
{
   static const char name[] = "sync";
   char error[256];
   struct agc_memory_stats stats;
   (void)p;
   if (agc_memory_imported(guest.straddle.address, 1))
      return probe_fail(name, "guest memory counts as imported before the first sync");
   if (!agc_memory_sync(error, sizeof(error)))
      return probe_fail(name, "%s", error);
   agc_memory_get_stats(&stats);
   const uint64_t bytes = guest.straddle.bytes + guest.released.bytes;
   if (stats.imports != 3 || stats.imported_bytes != bytes || stats.last_sync_unimported_bytes != 0 ||
       guest.aliases_mapped != 3)
      return probe_fail(name, "%" PRIu64 " imports of 0x%" PRIx64 " bytes from %u aliases, 0x%" PRIx64
                              " bytes refused; 3 imports of 0x%" PRIx64 " bytes were expected",
                        stats.imports, stats.imported_bytes, guest.aliases_mapped, stats.last_sync_unimported_bytes,
                        bytes);
   if (!agc_memory_imported(guest.straddle.address, guest.straddle.bytes) ||
       !agc_memory_imported(guest.released.address, guest.released.bytes) ||
       !agc_memory_imported(agc_mirror_va(guest.straddle.address), guest.straddle.bytes) ||
       agc_memory_imported(guest.straddle.address - PROBE_PAGE, PROBE_PAGE) ||
       agc_memory_imported(guest.released.address, guest.released.bytes + 1))
      return probe_fail(name, "agc_memory_imported disagrees with the mapped guest memory");
   const uint64_t passes = stats.sync_passes;
   if (!agc_memory_sync(error, sizeof(error)))
      return probe_fail(name, "second call: %s", error);
   agc_memory_get_stats(&stats);
   if (stats.sync_passes != passes)
      return probe_fail(name, "an unchanged generation was synchronised again");
   return probe_pass(name, "3 imports of 0x%" PRIx64 " bytes at their guest addresses, the allocation crossing "
                           "0x%" PRIx64 " cut there, paging settled",
                     bytes, guest.arena + PROBE_WINDOW);
}

static int
probe_straddle_matches(struct probe *p, const char *name, const char *what)
{
   return probe_compare(name, what, (void *)(uintptr_t)guest.straddle.address, p->expected, guest.straddle.bytes);
}

static void
probe_straddle_write(struct probe *p, uint64_t offset, uint32_t seed)
{
   probe_fill_pattern((uint8_t *)(uintptr_t)guest.straddle.address + offset, seed);
   probe_fill_pattern(p->expected + offset, seed);
}

static int
probe_guest_write(struct probe *p)
{
   static const char name[] = "(a) write to guest address";
   const uint64_t boundary = guest.arena + PROBE_WINDOW;
   const uint64_t offset = boundary - PROBE_PATTERN_BYTES / 2 - guest.straddle.address;
   probe_fill_guard((void *)(uintptr_t)guest.straddle.address, guest.straddle.bytes);
   probe_fill_guard(p->expected, guest.straddle.bytes);

   uint32_t *out = probe_emit_write(p->ib.cpu, guest.straddle.address + offset, 0xa0000001u);
   out = probe_emit_write(out, guest.straddle.address, 0xa0000002u);
   out = probe_emit_write(out, guest.straddle.address + guest.straddle.bytes - PROBE_PATTERN_BYTES, 0xa0000003u);
   int result = probe_submit(p, out, name, "WRITE_DATA across the import window boundary and at both ends", true);
   if (result != PROBE_OK)
      return result;
   probe_fill_pattern(p->expected + offset, 0xa0000001u);
   probe_fill_pattern(p->expected, 0xa0000002u);
   probe_fill_pattern(p->expected + guest.straddle.bytes - PROBE_PATTERN_BYTES, 0xa0000003u);
   result = probe_straddle_matches(p, name, "guest memory read by the CPU at the same address");
   if (result != PROBE_OK)
      return result;
   return probe_pass(name, "3 GPU writes, one across 0x%" PRIx64 ", are visible to the CPU at the same addresses; "
                           "0x%" PRIx64 " guard bytes intact",
                     boundary, guest.straddle.bytes - 3 * PROBE_PATTERN_BYTES);
}

static int
probe_alias_with(struct probe *p, bool dma_data, uint64_t base)
{
   const char *name = dma_data ? "(b) mirror alias, DMA_DATA" : "(b) mirror alias, COPY_DATA";
   const uint32_t bytes = dma_data ? PROBE_PATTERN_BYTES : 16;
   const uint32_t seed = dma_data ? 0xb0000001u : 0xb0000002u;
   const uint64_t source = guest.straddle.address + base;
   const uint64_t through_mirror = source + 0x2000;
   const uint64_t back = source + 0x2800;
   probe_straddle_write(p, base, seed);

   uint32_t *out = p->ib.cpu;
   if (dma_data) {
      out = probe_emit_dma_data(p, out, source, agc_mirror_va(through_mirror), bytes);
      out = probe_emit_dma_data(p, out, agc_mirror_va(through_mirror), back, bytes);
   } else {
      out = probe_emit_copy_data(out, source, agc_mirror_va(through_mirror), bytes);
      out = probe_emit_copy_data(out, agc_mirror_va(through_mirror), back, bytes);
   }
   int result = probe_submit(p, out, name, "guest to mirror + 0x2000, then mirror back to guest + 0x2800", true);
   if (result != PROBE_OK)
      return result;
   memcpy(p->expected + base + 0x2000, p->expected + base, bytes);
   memcpy(p->expected + base + 0x2800, p->expected + base, bytes);
   result = probe_straddle_matches(p, name, "guest memory after copies through the mirror window");
   if (result != PROBE_OK)
      return result;
   return probe_pass(name, "%u bytes written at mirror 0x%" PRIx64 " appear at guest 0x%" PRIx64
                           " and read back through the mirror",
                     bytes, agc_mirror_va(through_mirror), through_mirror);
}

static int
probe_alias(struct probe *p)
{
   int result = PROBE_OK;
   if (p->copy != PROBE_COPY_COPY_DATA)
      result = probe_alias_with(p, true, 0x1000);
   if (result == PROBE_OK && p->copy != PROBE_COPY_DMA_DATA)
      result = probe_alias_with(p, false, 0x1100);
   return result;
}

static int
probe_mirror_reads(struct probe *p, const char *name, const char *what, const uint64_t *offsets,
                   const uint32_t *seeds, unsigned count)
{
   probe_fill_guard(p->readback.cpu, PROBE_BLOCK_BYTES);
   agc_mem_flush_write_combining();
   uint32_t *out = p->ib.cpu;
   for (unsigned i = 0; i < count; i++) {
      out = probe_emit_copy(p, out, agc_mirror_va(guest.straddle.address + offsets[i]),
                            p->readback.va + i * PROBE_PATTERN_BYTES);
   }
   int result = probe_submit(p, out, name, what, true);
   for (unsigned i = 0; i < count && result == PROBE_OK; i++) {
      char where[96];
      snprintf(where, sizeof(where), "%s: mirror page +0x%" PRIx64, what, offsets[i] & ~UINT64_C(0xfff));
      result = probe_compare_pattern(name, where, (uint8_t *)p->readback.cpu + i * PROBE_PATTERN_BYTES, seeds[i]);
   }
   return result;
}

static int
probe_detach(struct probe *p)
{
   static const char name[] = "(c) mirror detach and attach";
   const uint64_t before = 0x4000;
   const uint64_t page = 0x5000;
   const uint64_t after = 0x6000;
   const uint64_t offsets[] = {page + 0x40, before + 0x40, after + 0x40};
   const uint64_t detached_va = guest.straddle.address + page;
   struct agc_memory_stats stats;
   struct agc_mem mem;

   probe_straddle_write(p, offsets[0], 0xc0000001u);
   probe_straddle_write(p, offsets[1], 0xc0000002u);
   probe_straddle_write(p, offsets[2], 0xc0000003u);
   if (!agc_mirror_detach(detached_va + 0x40, PROBE_PATTERN_BYTES, &mem))
      return probe_fail(name, "detach: %s", agc_memory_last_error());
   agc_memory_get_stats(&stats);
   if (!mem.cpu || mem.va != agc_mirror_va(detached_va + 0x40) || stats.detached_bytes != AGC_GPU_PAGE_BYTES ||
       !agc_mirror_detached(detached_va, AGC_GPU_PAGE_BYTES) || agc_mirror_detached(detached_va - 1, 1) ||
       agc_mirror_detached(detached_va + AGC_GPU_PAGE_BYTES, 1))
      return probe_fail(name, "detach made 0x%" PRIx64 " private bytes, CPU %p, GPU 0x%" PRIx64, stats.detached_bytes,
                        mem.cpu, mem.va);
   int result = probe_compare_pattern(name, "private copy at detach", mem.cpu, 0xc0000001u);
   if (result != PROBE_OK)
      return result;

   probe_straddle_write(p, offsets[0], 0xc0000011u);
   probe_straddle_write(p, offsets[1], 0xc0000012u);
   probe_straddle_write(p, offsets[2], 0xc0000013u);
   result = probe_mirror_reads(p, name, "after detach and CPU writes to the guest pages", offsets,
                               (const uint32_t[]){0xc0000001u, 0xc0000012u, 0xc0000013u}, 3);
   if (result != PROBE_OK)
      return result;
   probe_note(name, "the detached mirror page kept its contents, both neighbours followed the CPU writes");

   uint8_t stored[PROBE_PATTERN_BYTES];
   probe_fill_pattern(stored, 0xc0000021u);
   if (!agc_mirror_write(detached_va + 0x80, stored, sizeof(stored)))
      return probe_fail(name, "agc_mirror_write: %s", agc_memory_last_error());
   uint32_t *out = probe_emit_write(p->ib.cpu, agc_mirror_va(detached_va + 0x100), 0xc0000022u);
   result = probe_submit(p, out, name, "WRITE_DATA to the detached mirror page", true);
   if (result == PROBE_OK)
      result = probe_mirror_reads(p, name, "driver and GPU writes to the detached page", (const uint64_t[]){page + 0x80},
                                  (const uint32_t[]){0xc0000021u}, 1);
   if (result == PROBE_OK)
      result = probe_compare_pattern(name, "GPU write seen in private memory", (uint8_t *)mem.cpu + 0xc0, 0xc0000022u);
   if (result == PROBE_OK)
      result = probe_straddle_matches(p, name, "guest memory while its mirror page is private");
   if (result != PROBE_OK)
      return result;

   if (!agc_mirror_attach(detached_va + 0x40, PROBE_PATTERN_BYTES))
      return probe_fail(name, "attach: %s", agc_memory_last_error());
   agc_memory_get_stats(&stats);
   if (stats.detached_bytes != 0 || agc_mirror_detached(detached_va, 1))
      return probe_fail(name, "0x%" PRIx64 " bytes stay private after attach", stats.detached_bytes);
   result = probe_mirror_reads(p, name, "after attach", offsets,
                               (const uint32_t[]){0xc0000011u, 0xc0000012u, 0xc0000013u}, 3);
   if (result != PROBE_OK)
      return result;
   probe_straddle_write(p, offsets[0], 0xc0000031u);
   result = probe_mirror_reads(p, name, "after attach and another CPU write", offsets,
                               (const uint32_t[]){0xc0000031u, 0xc0000012u, 0xc0000013u}, 3);
   if (result != PROBE_OK)
      return result;
   return probe_pass(name, "mirror page 0x%" PRIx64 " held its copy while detached, took driver and GPU writes "
                           "without touching guest memory, and follows the guest page again after attach",
                     agc_mirror_va(detached_va));
}

static int
probe_release(struct probe *p)
{
   static const char name[] = "(d) release on unmap";
   char error[256];
   struct agc_memory_stats stats;
   uint8_t *memory = (uint8_t *)(uintptr_t)guest.released.address;
   const uint64_t address = guest.released.address;
   const uint64_t bytes = guest.released.bytes;
   const unsigned unmapped = guest.aliases_unmapped;

   probe_fill_guard(memory, bytes);
   uint32_t *out = probe_emit_write(p->ib.cpu, address + 0x100, 0xd0000001u);
   out = probe_emit_write(out, address + bytes - PROBE_PATTERN_BYTES, 0xd0000002u);
   int result = probe_submit(p, out, name, "WRITE_DATA to the allocation about to be unmapped", false);
   if (result != PROBE_OK)
      return result;
   result = probe_guest_unmap(&guest.released);
   if (result != PROBE_OK)
      return result;
   if (p->quiesce_calls != 1 || p->resume_calls != 1 || p->released_calls != 1 || p->released_va != address ||
       p->released_bytes != bytes || guest.aliases_unmapped != unmapped + 1 || guest.imports_forgotten != 0)
      return probe_fail(name, "quiesce %u, resume %u, released %u (0x%" PRIx64 "+0x%" PRIx64 "), %u aliases "
                              "unmapped, %u forgotten",
                        p->quiesce_calls, p->resume_calls, p->released_calls, p->released_va, p->released_bytes,
                        guest.aliases_unmapped - unmapped, guest.imports_forgotten);
   if (agc_memory_imported(address, 1) || !agc_memory_imported(guest.straddle.address, guest.straddle.bytes))
      return probe_fail(name, "the released allocation is still imported, or the other one is not");

   const uint64_t lead = guest.released.section_offset % PROBE_GRANULARITY;
   uint8_t *view = MapViewOfFile(guest.section, FILE_MAP_READ, 0, (DWORD)(guest.released.section_offset - lead),
                                 (SIZE_T)(lead + bytes));
   if (!view)
      return probe_fail(name, "no view of the unmapped pages (error %lu)", GetLastError());
   result = probe_compare_pattern(name, "write queued before the unmap", view + lead + 0x100, 0xd0000001u);
   if (result == PROBE_OK)
      result = probe_compare_pattern(name, "write queued before the unmap", view + lead + bytes - PROBE_PATTERN_BYTES,
                                     0xd0000002u);
   UnmapViewOfFile(view);
   if (result != PROBE_OK)
      return result;

   if (!agc_memory_sync(error, sizeof(error)))
      return probe_fail(name, "sync after the unmap: %s", error);
   agc_memory_get_stats(&stats);
   if (stats.imports != 2 || stats.imports_released != 1 || stats.imports_abandoned != 0)
      return probe_fail(name, "%" PRIu64 " imports, %" PRIu64 " released, %" PRIu64 " abandoned after the unmap",
                        stats.imports, stats.imports_released, stats.imports_abandoned);

   probe_straddle_write(p, 0x8000, 0xd0000003u);
   out = probe_emit_write(p->ib.cpu, guest.straddle.address + 0x8100, 0xd0000004u);
   out = probe_emit_copy(p, out, guest.straddle.address + 0x8000, agc_mirror_va(guest.straddle.address + 0x8200));
   result = probe_submit(p, out, name, "GPU work on the remaining imports", true);
   if (result != PROBE_OK)
      return result;
   probe_fill_pattern(p->expected + 0x8100, 0xd0000004u);
   probe_fill_pattern(p->expected + 0x8200, 0xd0000003u);
   result = probe_straddle_matches(p, name, "guest memory after the release");
   if (result != PROBE_OK)
      return result;
   if (probe_device_state(p) != AGC_DEVICE_ACTIVE)
      return PROBE_UNCONFIRMED;

   result = probe_guest_map(&guest.released);
   if (result != PROBE_OK)
      return result;
   if (!agc_memory_sync(error, sizeof(error)))
      return probe_fail(name, "sync after mapping again: %s", error);
   agc_memory_get_stats(&stats);
   if (stats.imports != 3 || !agc_memory_imported(address, bytes))
      return probe_fail(name, "%" PRIu64 " imports after mapping the allocation again", stats.imports);
   probe_fill_guard(memory, bytes);
   out = probe_emit_write(p->ib.cpu, address + 0x200, 0xd0000005u);
   result = probe_submit(p, out, name, "WRITE_DATA to the allocation imported again", true);
   if (result == PROBE_OK)
      result = probe_compare_pattern(name, "allocation imported again", memory + 0x200, 0xd0000005u);
   if (result != PROBE_OK)
      return result;
   return probe_pass(name, "the release callback waited for the queued write, released 1 import and its alias; "
                           "the GPU stayed usable and the address was imported again");
}

static int
probe_sparse(struct probe *p)
{
   static const char name[] = "(f) sparse range";
   char error[256];
   struct agc_memory_stats stats;
   const uint64_t address = guest.arena + PROBE_SPARSE_GUEST_OFFSET;
   guest.sparse = (struct agc_guest_sparse_range){(uintptr_t)address, (size_t)PROBE_SPARSE_BYTES, 0, 0};
   guest.sparse_registered = true;
   guest.sparse_generation++;
   if (!agc_memory_sync(error, sizeof(error)))
      return probe_fail(name, "sync: %s", error);
   agc_memory_get_stats(&stats);
   if (stats.zero_filled_bytes != PROBE_SPARSE_BYTES)
      return probe_fail(name, "0x%" PRIx64 " bytes read zero, not 0x%" PRIx64, stats.zero_filled_bytes,
                        (uint64_t)PROBE_SPARSE_BYTES);

   probe_fill_guard(p->readback.cpu, PROBE_BLOCK_BYTES);
   agc_mem_flush_write_combining();
   uint32_t *out = probe_emit_copy(p, p->ib.cpu, address + 0x3000, p->readback.va);
   out = probe_emit_copy(p, out, agc_mirror_va(address + 0x7000), p->readback.va + PROBE_PATTERN_BYTES);
   int result = probe_submit(p, out, name, "reads of an unmapped sparse range through both windows", true);
   if (result != PROBE_OK)
      return result;
   uint8_t expected[2 * PROBE_PATTERN_BYTES] = {0};
   result = probe_compare(name, "sparse hole", p->readback.cpu, expected, sizeof(expected));
   if (result != PROBE_OK)
      return result;
   if (probe_device_state(p) != AGC_DEVICE_ACTIVE)
      return PROBE_UNCONFIRMED;

   guest.sparse_registered = false;
   guest.sparse_generation++;
   if (!agc_memory_sync(error, sizeof(error)))
      return probe_fail(name, "sync after unregistering: %s", error);
   agc_memory_get_stats(&stats);
   const uint64_t deferred = stats.zero_filled_bytes;
   if (agc_memory_release_stale(NULL) != 0)
      return probe_fail(name, "an import counted as stale");
   agc_memory_get_stats(&stats);
   if (deferred != PROBE_SPARSE_BYTES || stats.zero_filled_bytes != 0 || stats.lost)
      return probe_fail(name, "0x%" PRIx64 " bytes read zero after unregistering and 0x%" PRIx64
                              " after the idle release",
                        deferred, stats.zero_filled_bytes);
   if (probe_device_state(p) != AGC_DEVICE_ACTIVE)
      return PROBE_UNCONFIRMED;
   return probe_pass(name, "an unmapped registered range read zero at 0x%" PRIx64 " and 0x%" PRIx64 "; after "
                           "unregistering it kept reading zero until the idle release made it inaccessible",
                     address, agc_mirror_va(address));
}

static int
probe_teardown(struct probe *p)
{
   static const char name[] = "teardown";
   agc_mem_free(&p->readback);
   agc_mem_free(&p->ib);
   p->ops->queue_destroy(p->queue);
   p->queue = NULL;
   agc_memory_fini();

   struct agc_wddm_description description;
   agc_wddm_describe(p->be, &description);
   unsigned live = probe_guest_live_imports(guest.arena, PROBE_ARENA_BYTES);
   p->ops->destroy(p->be);
   p->be = NULL;
   if (description.lost)
      return probe_fail(name, "the device was lost");
   if (live != 0 || guest.imports_forgotten != 0 || guest.release != NULL)
      return probe_fail(name, "%u import aliases stay mapped, %u were forgotten, release callback %s", live,
                        guest.imports_forgotten, guest.release ? "still installed" : "removed");
   if (description.live_allocations != 0)
      return probe_fail(name, "%u kernel allocations were still live", description.live_allocations);
   return probe_pass(name, "%u aliases mapped and unmapped, no kernel allocation left, windows released",
                     guest.aliases_mapped);
}

static int
probe_usage(void)
{
   fprintf(stderr, "usage: agc_memory_probe [--timeout-ms N] [--queue compute|gfx] [--copy both|dma|copy_data] "
                   "[--sparse yes|no]\n");
   return PROBE_FAILED;
}

int
main(int argc, char **argv)
{
   struct probe p = {
      .kind = AGC_QUEUE_COMPUTE,
      .copy = PROBE_COPY_BOTH,
      .timeout_ns = INT64_C(5000) * 1000000,
   };
   if (argc % 2 != 1)
      return probe_usage();
   for (int i = 1; i < argc; i += 2) {
      const char *value = argv[i + 1];
      if (!strcmp(argv[i], "--timeout-ms")) {
         char *end = NULL;
         unsigned long long ms = strtoull(value, &end, 0);
         if (*end || ms == 0 || ms > 600000)
            return probe_usage();
         p.timeout_ns = (int64_t)ms * 1000000;
      } else if (!strcmp(argv[i], "--queue") && !strcmp(value, "compute")) {
         p.kind = AGC_QUEUE_COMPUTE;
      } else if (!strcmp(argv[i], "--queue") && !strcmp(value, "gfx")) {
         p.kind = AGC_QUEUE_GFX;
      } else if (!strcmp(argv[i], "--copy") && !strcmp(value, "both")) {
         p.copy = PROBE_COPY_BOTH;
      } else if (!strcmp(argv[i], "--copy") && !strcmp(value, "dma")) {
         p.copy = PROBE_COPY_DMA_DATA;
      } else if (!strcmp(argv[i], "--copy") && !strcmp(value, "copy_data")) {
         p.copy = PROBE_COPY_COPY_DATA;
      } else if (!strcmp(argv[i], "--sparse") && !strcmp(value, "yes")) {
         p.sparse = true;
      } else if (!strcmp(argv[i], "--sparse") && !strcmp(value, "no")) {
         p.sparse = false;
      } else {
         return probe_usage();
      }
   }

   int result = probe_open(&p);
   if (result != PROBE_OK)
      return result;
   result = probe_guest_open();
   if (result == PROBE_OK) {
      p.expected = malloc((size_t)guest.straddle.bytes);
      if (!p.expected)
         result = probe_fail("guest", "out of memory");
   }
   if (result == PROBE_OK)
      result = probe_init(&p);
   if (result == PROBE_OK)
      result = probe_driver_memory(&p);
   if (result == PROBE_OK)
      result = probe_runtime_memory(&p);
   if (result == PROBE_OK)
      result = probe_sync(&p);
   if (result == PROBE_OK)
      result = probe_guest_write(&p);
   if (result == PROBE_OK)
      result = probe_alias(&p);
   if (result == PROBE_OK)
      result = probe_detach(&p);
   if (result == PROBE_OK)
      result = probe_release(&p);
   if (result == PROBE_OK && p.sparse)
      result = probe_sparse(&p);

   if (result != PROBE_OK) {
      probe_note("failure", "memory module: %s", agc_memory_last_error());
      if (result == PROBE_UNCONFIRMED || probe_device_state(&p) != AGC_DEVICE_ACTIVE) {
         probe_note("teardown", "skipped: the GPU state is unconfirmed, the process exit releases everything");
         return PROBE_UNCONFIRMED;
      }
      probe_note("teardown", "skipped after a failure, the process exit releases everything");
      return result;
   }
   if (probe_device_state(&p) != AGC_DEVICE_ACTIVE)
      return PROBE_UNCONFIRMED;
   return probe_teardown(&p);
}
