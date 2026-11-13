#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac_gpu_info.h"
#include "amd_family.h"
#include "sid.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/u_math.h"

#define XXH_INLINE_ALL
#include "util/xxhash.h"

#include "agc_cb_builders.h"
#include "agc_driver_ops.h"

#include <windows.h>

#define PROBE_OK 0
#define PROBE_FAILED 1
#define PROBE_UNCONFIRMED 2

#define PROBE_BUFFER_BYTES UINT64_C(0x10000)
#define PROBE_GUARD 0xa5a5a5a5u
#define PROBE_PATTERN_DWORDS 16u
#define PROBE_WRITE_OFFSET UINT64_C(0x100)
#define PROBE_GUEST_WINDOW_BASE UINT64_C(0x200000000)
#define PROBE_GUEST_WINDOW_END UINT64_C(0xfc00000000)
#define PROBE_VIEW_FIRST_OFFSET UINT64_C(0x10000000)
#define PROBE_VIEW_STEP UINT64_C(0x40000000)
#define PROBE_VIEW_ATTEMPTS 64u
#define PROBE_ALIAS_WINDOW_BYTES (UINT64_C(8) << 20)
#define PROBE_ALIAS_OFFSET UINT64_C(0x4000)
#define PROBE_ALIAS_BYTES UINT64_C(0x8000)
#define PROBE_IMPORT_CHUNK_BYTES (UINT64_C(128) << 20)
#define PROBE_ABI_RUST_FIXTURE_BYTES 1160u
#define PROBE_ABI_RUST_FIXTURE_XXH64 UINT64_C(0x0cd03582c2d2b4d7)
#define PROBE_ABI_FIXTURE_BYTES 4784u
#define PROBE_ABI_FIXTURE_XXH64 UINT64_C(0x38cd98ba7fa028c1)
#define PROBE_UMD_BASIC_BYTES 3008u
#define PROBE_UMD_GPU_BYTES 17248u

struct probe {
   struct agc_backend *be;
   const struct agc_driver_ops *ops;
   int64_t timeout_ns;
   uint64_t import_bytes;
   const char *dump_path;
};

static void probe_step(const char *name, const char *format, ...) AGC_PRINTF(2, 3);
static int probe_fail(const char *name, const char *format, ...) AGC_PRINTF(2, 3);

static void
probe_step(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   printf("%s: ", name);
   vprintf(format, args);
   putchar('\n');
   va_end(args);
   fflush(stdout);
}

static int
probe_fail(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   printf("FAIL %s: ", name);
   vprintf(format, args);
   putchar('\n');
   va_end(args);
   fflush(stdout);
   return PROBE_FAILED;
}

static void
probe_version(uint64_t version, char *text, size_t size)
{
   snprintf(text, size, "%u.%u.%u.%u", (unsigned)(version >> 48),
            (unsigned)(version >> 32) & 0xffff, (unsigned)(version >> 16) & 0xffff,
            (unsigned)version & 0xffff);
}

static void
probe_print_adapter(const char *name, const struct agc_wddm_adapter *adapter)
{
   char version[32];
   probe_version(adapter->driver_version, version, sizeof(version));
   probe_step(name,
              "vendor %04x device %04x revision %02x subsystem %04x:%04x driver %s luid "
              "%08x:%08x profile %s",
              adapter->vendor_id, adapter->device_id, adapter->revision, adapter->sub_vendor_id,
              adapter->subsystem_id, version, (unsigned)adapter->luid_high, adapter->luid_low,
              adapter->profile ? adapter->profile : "none");
}

static enum agc_device_state
probe_print_state(struct probe *p)
{
   static const char *const names[] = {
      [AGC_DEVICE_ACTIVE] = "active",         [AGC_DEVICE_HUNG] = "hung",
      [AGC_DEVICE_PAGE_FAULT] = "page-fault", [AGC_DEVICE_REMOVED] = "removed",
      [AGC_DEVICE_UNKNOWN] = "unknown",
   };
   struct agc_wddm_device_state detail;
   struct agc_wddm_description description;
   char text[320];
   uint64_t fault_va = 0;
   enum agc_device_state state = p->ops->device_state(p->be, &fault_va);
   agc_wddm_device_state_detail(p->be, &detail);
   agc_wddm_device_state_format(&detail, text, sizeof(text));
   agc_wddm_describe(p->be, &description);
   probe_step("device state", "%s, fault va 0x%" PRIx64 ", transport %s; %s", names[state],
              fault_va, description.lost ? "refuses further work" : "usable", text);
   return state;
}

static int
probe_open(struct probe *p)
{
   struct agc_wddm_adapter adapters[16];
   uint32_t count = agc_wddm_enumerate(adapters, ARRAY_SIZE(adapters));
   probe_step("adapters", "%u enumerated", count);
   for (uint32_t i = 0; i < MIN2(count, ARRAY_SIZE(adapters)); i++) {
      char name[24];
      snprintf(name, sizeof(name), "adapter %u", i);
      probe_print_adapter(name, &adapters[i]);
   }

   p->be = agc_wddm_open();
   if (!p->be)
      return probe_fail("open", "no AMD adapter with a verified profile could be opened");
   p->ops = p->be->ops;

   struct agc_wddm_description description;
   agc_wddm_describe(p->be, &description);
   probe_print_adapter("open", &description.adapter);
   probe_step("gpu va", "%u bits, end 0x%" PRIx64 ", driver window from 0x%" PRIx64,
              description.va_bits, description.va_end, description.driver_va_base);
   return PROBE_OK;
}

static int
probe_close(struct probe *p)
{
   struct agc_wddm_description description;
   agc_wddm_describe(p->be, &description);
   p->ops->destroy(p->be);
   p->be = NULL;
   if (description.lost)
      return probe_fail("close", "the device was lost");
   if (description.live_allocations != 0)
      return probe_fail("close", "%u allocations were still live", description.live_allocations);
   probe_step("close", "device, paging queue and adapter released");
   return PROBE_OK;
}

static uint32_t
probe_pattern(uint32_t seed, uint32_t index)
{
   return seed + index * 0x01010101u;
}

static void
probe_fill(void *memory, uint64_t bytes)
{
   uint32_t *words = memory;
   for (uint64_t i = 0; i < bytes / sizeof(uint32_t); i++)
      words[i] = PROBE_GUARD;
}

static int
probe_verify(const char *what, const void *memory, uint64_t bytes, uint64_t offset, uint32_t seed)
{
   const volatile uint32_t *words = memory;
   uint64_t first = offset / sizeof(uint32_t);
   for (uint32_t i = 0; i < PROBE_PATTERN_DWORDS; i++) {
      uint32_t value = words[first + i];
      if (value != probe_pattern(seed, i)) {
         return probe_fail("verify", "%s: dword %u at +0x%" PRIx64 " is 0x%08x, not 0x%08x", what,
                           i, offset + i * sizeof(uint32_t), value, probe_pattern(seed, i));
      }
   }
   for (uint64_t i = 0; i < bytes / sizeof(uint32_t); i++) {
      if (i >= first && i < first + PROBE_PATTERN_DWORDS)
         continue;
      uint32_t value = words[i];
      if (value != PROBE_GUARD) {
         return probe_fail("verify", "%s: guard dword at +0x%" PRIx64 " changed to 0x%08x", what,
                           i * sizeof(uint32_t), value);
      }
   }
   probe_step("verify", "%s: %u dwords at +0x%" PRIx64 " match, %" PRIu64 " guard dwords intact",
              what, PROBE_PATTERN_DWORDS, offset,
              bytes / sizeof(uint32_t) - PROBE_PATTERN_DWORDS);
   return PROBE_OK;
}

static int
probe_gpu_write(struct probe *p, struct agc_hwq *queue, struct agc_bo *ib, uint64_t va,
                uint32_t seed, const char *what)
{
   uint32_t *words = p->ops->bo_cpu(ib);
   uint32_t used = 0;
   words[used++] = PKT3(PKT3_WRITE_DATA, 2 + PROBE_PATTERN_DWORDS, 0);
   words[used++] = S_371_DST_SEL(V_371_MEMORY) |
                   S_371_WR_CONFIRM(V_371_WAIT_FOR_WRITE_CONFIRMATION) |
                   S_371_ENGINE_SEL(V_371_MICRO_ENGINE);
   words[used++] = (uint32_t)va;
   words[used++] = (uint32_t)(va >> 32);
   for (uint32_t i = 0; i < PROBE_PATTERN_DWORDS; i++)
      words[used++] = probe_pattern(seed, i);
   uint32_t dwords = align(used, 8);
   agc_fill_nops(words + used, dwords - used);

   const struct agc_ib entry = {
      .va = p->ops->bo_va(ib),
      .dwords = dwords,
      .flags = AGC_IB_MAIN,
   };
   uint64_t fence = p->ops->submit(queue, &entry, 1);
   if (!fence) {
      struct agc_wddm_description description;
      agc_wddm_describe(p->be, &description);
      probe_fail("submit", "%s: the submission was refused", what);
      if (!description.lost)
         return PROBE_FAILED;
      probe_print_state(p);
      return PROBE_UNCONFIRMED;
   }
   probe_step("submit", "%s: fence %" PRIu64 ", IB 0x%" PRIx64 " of %u dwords, WRITE_DATA of %u "
                        "dwords to 0x%" PRIx64,
              what, fence, entry.va, dwords, PROBE_PATTERN_DWORDS, va);

   int64_t start = os_time_get_nano();
   if (!p->ops->fence_wait(queue, fence, p->timeout_ns)) {
      probe_fail("wait", "%s: fence %" PRIu64 " not reached within %" PRId64 " ms, completed %"
                         PRIu64,
                 what, fence, p->timeout_ns / 1000000, p->ops->fence_completed(queue));
      probe_print_state(p);
      return PROBE_UNCONFIRMED;
   }
   probe_step("wait", "%s: fence %" PRIu64 " reached in %.3f ms, completed %" PRIu64, what, fence,
              (double)(os_time_get_nano() - start) / 1e6, p->ops->fence_completed(queue));
   return PROBE_OK;
}

static struct agc_hwq *
probe_queue(struct probe *p, enum agc_queue_kind kind)
{
   struct agc_hwq *queue = p->ops->queue_create(p->be, kind);
   if (!queue) {
      probe_fail("queue", "no %s hardware queue", kind == AGC_QUEUE_GFX ? "graphics" : "compute");
      return NULL;
   }
   struct agc_wddm_queue_description description;
   agc_wddm_queue_describe(queue, &description);
   probe_step("queue", "%s on node %u, id %u, progress fence at GPU va 0x%" PRIx64 " reads %" PRIu64,
              kind == AGC_QUEUE_GFX ? "graphics" : "compute", description.node,
              description.queue_id, description.fence_gpu_va, description.submitted);
   return queue;
}

static struct agc_bo *
probe_buffer(struct probe *p, const char *name)
{
   struct agc_bo *bo = p->ops->bo_create(p->be, PROBE_BUFFER_BYTES, AGC_BO_GL2_BYPASS, 0);
   if (!bo) {
      probe_fail(name, "allocation of 0x%" PRIx64 " bytes failed", PROBE_BUFFER_BYTES);
      return NULL;
   }
   probe_step(name, "GPU va 0x%" PRIx64 ", CPU %p, 0x%" PRIx64 " bytes", p->ops->bo_va(bo),
              p->ops->bo_cpu(bo), p->ops->bo_size(bo));
   return bo;
}

static int
probe_finish(struct probe *p, int result)
{
   if (result == PROBE_UNCONFIRMED) {
      probe_step("teardown", "skipped: GPU completion is unconfirmed, the process exit releases "
                             "everything");
      return result;
   }
   enum agc_device_state state = probe_print_state(p);
   if (state != AGC_DEVICE_ACTIVE) {
      probe_step("teardown", "skipped: the device is not active");
      return PROBE_UNCONFIRMED;
   }
   return result;
}

static int
cmd_info(struct probe *p)
{
   int result = probe_open(p);
   if (result != PROBE_OK)
      return result;

   struct agc_wddm_description description;
   agc_wddm_describe(p->be, &description);
   struct agc_wddm_node *nodes = calloc(MAX2(description.node_count, 1), sizeof(*nodes));
   if (!nodes)
      return probe_fail("nodes", "out of memory");
   uint32_t node_count = agc_wddm_nodes(p->be, nodes, description.node_count);
   for (uint32_t i = 0; i < node_count; i++) {
      char name[24];
      snprintf(name, sizeof(name), "node %u", nodes[i].ordinal);
      probe_step(name, "engine type %d, \"%s\"%s%s", nodes[i].engine_type, nodes[i].name,
                 nodes[i].ordinal == description.gfx_node ? ", graphics queues" : "",
                 nodes[i].ordinal == description.compute_node ? ", compute queues" : "");
   }
   free(nodes);
   probe_step("nodes", "%u reported, graphics queues %s, compute queues %s", node_count,
              description.gfx_node == AGC_WDDM_NODE_NONE ? "unavailable" : "available",
              description.compute_node == AGC_WDDM_NODE_NONE ? "unavailable" : "available");

   struct radeon_info info;
   if (!p->ops->query_info(p->be, &info)) {
      probe_fail("chip", "the chip information could not be filled");
      probe_close(p);
      return PROBE_FAILED;
   }
   agc_wddm_describe(p->be, &description);
   probe_step("chip query", "KMD basic NTSTATUS 0x%08x, KMD gpu NTSTATUS 0x%08x, source: %s",
              (unsigned)description.chip_basic_status, (unsigned)description.chip_gpu_status,
              description.chip_source == AGC_WDDM_CHIP_KMD_QUERY
                 ? "KMD query checked against the profile table"
                 : "profile table only");
   probe_step("chip", "\"%s\", family %s, gfx level %d, family id 0x%x, external rev 0x%x, chip "
                      "rev %u, pci %04x rev %02x",
              info.marketing_name, ac_get_family_name(info.family), (int)info.gfx_level,
              info.family_id, info.chip_external_rev, info.chip_rev, info.pci_id, info.pci_rev_id);
   probe_step("pci", "bus %u device %u function %u%s", info.pci.bus, info.pci.dev, info.pci.func,
              info.pci.valid ? "" : " (not reported)");
   probe_step("shader engines", "%u of %u enabled, %u arrays each, %u compute units, %u..%u good "
                                "per array, cu mask 0x%x",
              info.num_se, info.max_se, info.max_sa_per_se, info.num_cu, info.min_good_cu_per_sa,
              info.max_good_cu_per_sa, info.cu_mask[0][0]);
   probe_step("render backends", "%u of %u enabled, mask 0x%" PRIx64 ", gb_addr_config 0x%x, %u "
                                 "tile pipes",
              info.num_rb, info.max_render_backends, info.enabled_rb_mask, info.gb_addr_config,
              info.num_tile_pipes);
   probe_step("command processor", "gfx ip %u.%u.%u, IB alignment %u bytes, pad mask 0x%x, up to "
                                   "%u IBs per submission, me %u mec %u pfp %u",
              info.ip[AMD_IP_GFX].ver_major, info.ip[AMD_IP_GFX].ver_minor,
              info.ip[AMD_IP_GFX].ver_rev, info.ip[AMD_IP_GFX].ib_alignment,
              info.ip[AMD_IP_GFX].ib_pad_dw_mask, info.max_submitted_ibs[AMD_IP_GFX],
              info.me_fw_version, info.mec_fw_version, info.pfp_fw_version);
   probe_step("memory", "vram %u KiB, gart %u KiB, va max 0x%" PRIx64 ", pte fragment %u, gart "
                        "page %u, l2 %u bytes, l3 %u MiB, tcc blocks %u",
              info.vram_size_kb, info.gart_size_kb, info.virtual_address_max,
              info.pte_fragment_size, info.gart_page_size, info.l2_cache_size,
              info.l3_cache_size_mb, info.num_tcc_blocks);
   probe_step("rings", "attribute ring %u bytes per engine, scratch waves %u, tess offchip %u "
                       "bytes, hs offchip param 0x%x",
              info.attribute_ring_size_per_se, info.max_scratch_waves, info.tess_offchip_ring_size,
              info.hs_offchip_param);

   struct agc_wddm_video_memory memory;
   if (agc_wddm_video_memory(p->be, &memory)) {
      probe_step("video memory", "local %" PRIu64 " of %" PRIu64 " MiB used, non-local %" PRIu64
                                 " of %" PRIu64 " MiB used",
                 memory.local_usage >> 20, memory.local_budget >> 20, memory.non_local_usage >> 20,
                 memory.non_local_budget >> 20);
   } else {
      result = probe_fail("video memory", "the budget query failed");
   }

   if (p->dump_path) {
      uint8_t *raw = calloc(1, PROBE_UMD_BASIC_BYTES + PROBE_UMD_GPU_BYTES);
      FILE *file = raw ? fopen(p->dump_path, "wb") : NULL;
      if (!file) {
         result = probe_fail("dump", "cannot write %s", p->dump_path);
      } else {
         int32_t basic = agc_wddm_query_umd_private(p->be, raw, PROBE_UMD_BASIC_BYTES);
         int32_t gpu =
            agc_wddm_query_umd_private(p->be, raw + PROBE_UMD_BASIC_BYTES, PROBE_UMD_GPU_BYTES);
         size_t written = fwrite(raw, 1, PROBE_UMD_BASIC_BYTES + PROBE_UMD_GPU_BYTES, file);
         fclose(file);
         probe_step("dump", "%zu bytes to %s: %u-byte query (NTSTATUS 0x%08x), then %u-byte query "
                            "(NTSTATUS 0x%08x)",
                    written, p->dump_path, PROBE_UMD_BASIC_BYTES, (unsigned)basic,
                    PROBE_UMD_GPU_BYTES, (unsigned)gpu);
      }
      free(raw);
   }

   if (probe_print_state(p) != AGC_DEVICE_ACTIVE)
      return PROBE_UNCONFIRMED;
   int closed = probe_close(p);
   return result != PROBE_OK ? result : closed;
}

static int
cmd_write(struct probe *p, enum agc_queue_kind kind)
{
   int result = probe_open(p);
   if (result != PROBE_OK)
      return result;

   struct agc_hwq *queue = probe_queue(p, kind);
   struct agc_bo *ib = queue ? probe_buffer(p, "ib buffer") : NULL;
   struct agc_bo *target = ib ? probe_buffer(p, "target buffer") : NULL;
   result = target ? PROBE_OK : PROBE_FAILED;

   if (result == PROBE_OK) {
      const uint32_t seed = kind == AGC_QUEUE_GFX ? 0x600d0000u : 0xc0de0000u;
      probe_fill(p->ops->bo_cpu(target), p->ops->bo_size(target));
      result = probe_gpu_write(p, queue, ib, p->ops->bo_va(target) + PROBE_WRITE_OFFSET, seed,
                               "target buffer");
      if (result == PROBE_OK) {
         result = probe_verify("target buffer", p->ops->bo_cpu(target), p->ops->bo_size(target),
                               PROBE_WRITE_OFFSET, seed);
      }
   }

   result = probe_finish(p, result);
   if (result == PROBE_UNCONFIRMED)
      return result;
   p->ops->bo_destroy(p->be, target);
   p->ops->bo_destroy(p->be, ib);
   p->ops->queue_destroy(queue);
   probe_step("teardown", "buffers and queue released");
   int closed = probe_close(p);
   return result != PROBE_OK ? result : closed;
}

static void *
probe_map_view(HANDLE section, uint64_t bytes, uint64_t alias_span)
{
   uint64_t limit = PROBE_GUEST_WINDOW_END - bytes - alias_span;
   for (uint32_t i = 0; i < PROBE_VIEW_ATTEMPTS; i++) {
      uint64_t address = PROBE_GUEST_WINDOW_BASE + PROBE_VIEW_FIRST_OFFSET + i * PROBE_VIEW_STEP;
      if (address > limit)
         break;
      void *view = MapViewOfFileEx(section, FILE_MAP_ALL_ACCESS, 0, 0, (SIZE_T)bytes,
                                   (void *)(uintptr_t)address);
      if (view)
         return view;
   }
   return NULL;
}

static int
cmd_import(struct probe *p)
{
   int result = probe_open(p);
   if (result != PROBE_OK)
      return result;

   const uint64_t bytes = p->import_bytes;
   const uint64_t alias_span = align64(bytes, PROBE_ALIAS_WINDOW_BYTES) + PROBE_ALIAS_WINDOW_BYTES;
   struct agc_hwq *queue = probe_queue(p, AGC_QUEUE_GFX);
   struct agc_bo *ib = NULL;
   struct agc_bo *import = NULL;
   HANDLE section = NULL;
   void *view = NULL;
   bool reserved = false;
   result = queue ? PROBE_OK : PROBE_FAILED;

   if (result == PROBE_OK) {
      reserved = p->ops->va_reserve(p->be, PROBE_GUEST_WINDOW_BASE,
                                    PROBE_GUEST_WINDOW_END - PROBE_GUEST_WINDOW_BASE);
      if (reserved) {
         probe_step("reserve", "GPU va 0x%" PRIx64 "..0x%" PRIx64, PROBE_GUEST_WINDOW_BASE,
                    PROBE_GUEST_WINDOW_END);
      } else {
         result = probe_fail("reserve", "GPU va 0x%" PRIx64 "..0x%" PRIx64 " was refused",
                             PROBE_GUEST_WINDOW_BASE, PROBE_GUEST_WINDOW_END);
      }
   }

   if (result == PROBE_OK) {
      section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
                                   (DWORD)(bytes >> 32), (DWORD)bytes, NULL);
      view = section ? probe_map_view(section, bytes, alias_span) : NULL;
      if (view) {
         probe_fill(view, bytes);
         probe_step("section", "0x%" PRIx64 " bytes of pagefile section memory mapped at CPU %p",
                    bytes, view);
      } else {
         result = probe_fail("section", "no view of 0x%" PRIx64 " bytes inside the reserved range "
                                        "(error %lu)",
                             bytes, GetLastError());
      }
   }

   if (result == PROBE_OK) {
      ib = probe_buffer(p, "ib buffer");
      if (!ib)
         result = PROBE_FAILED;
   }

   const uint64_t view_va = (uint64_t)(uintptr_t)view;
   if (result == PROBE_OK) {
      import = p->ops->bo_import(p->be, view, bytes, AGC_BO_GL2_BYPASS, view_va);
      if (!import) {
         result = probe_fail("import", "0x%" PRIx64 " bytes at %p were refused", bytes, view);
      } else if (p->ops->bo_va(import) != view_va || p->ops->bo_cpu(import) != view) {
         result = probe_fail("import", "GPU va 0x%" PRIx64 " is not the CPU address %p",
                             p->ops->bo_va(import), view);
      } else if (!p->ops->paging_settle(p->be)) {
         probe_fail("import", "the paging work did not settle");
         probe_print_state(p);
         result = PROBE_UNCONFIRMED;
      } else {
         probe_step("import", "GPU va 0x%" PRIx64 " equals the CPU address, 0x%" PRIx64 " bytes in "
                              "%" PRIu64 " allocations, paging settled",
                    view_va, bytes, DIV_ROUND_UP(bytes, PROBE_IMPORT_CHUNK_BYTES));
      }
   }

   if (result == PROBE_OK) {
      const uint32_t seed = 0x1a900000u;
      result = probe_gpu_write(p, queue, ib, view_va + PROBE_WRITE_OFFSET, seed, "import");
      if (result == PROBE_OK)
         result = probe_verify("import", view, bytes, PROBE_WRITE_OFFSET, seed);
   }

   if (result == PROBE_OK && bytes > PROBE_IMPORT_CHUNK_BYTES) {
      const uint32_t seed = 0x2b900000u;
      const uint64_t offset = PROBE_IMPORT_CHUNK_BYTES - PROBE_PATTERN_DWORDS * 2;
      probe_fill(view, bytes);
      result = probe_gpu_write(p, queue, ib, view_va + offset, seed, "allocation boundary");
      if (result == PROBE_OK)
         result = probe_verify("allocation boundary", view, bytes, offset, seed);
   }

   if (result == PROBE_OK) {
      const uint64_t alias_va = view_va + align64(bytes, PROBE_ALIAS_WINDOW_BYTES);
      const uint32_t seed = 0x3c900000u;
      probe_fill(view, bytes);
      if (!p->ops->bo_alias(p->be, import, PROBE_ALIAS_OFFSET, PROBE_ALIAS_BYTES, alias_va)) {
         result = probe_fail("alias", "pages +0x%" PRIx64 "..+0x%" PRIx64 " were not mapped at 0x%"
                                      PRIx64,
                             PROBE_ALIAS_OFFSET, PROBE_ALIAS_OFFSET + PROBE_ALIAS_BYTES, alias_va);
      } else if (!p->ops->paging_settle(p->be)) {
         probe_fail("alias", "the paging work did not settle");
         probe_print_state(p);
         result = PROBE_UNCONFIRMED;
      } else {
         probe_step("alias", "pages +0x%" PRIx64 "..+0x%" PRIx64 " also mapped at GPU va 0x%" PRIx64
                             ", paging settled",
                    PROBE_ALIAS_OFFSET, PROBE_ALIAS_OFFSET + PROBE_ALIAS_BYTES, alias_va);
         result = probe_gpu_write(p, queue, ib, alias_va + PROBE_WRITE_OFFSET, seed, "alias");
         if (result == PROBE_OK) {
            result = probe_verify("alias", view, bytes, PROBE_ALIAS_OFFSET + PROBE_WRITE_OFFSET,
                                  seed);
         }
      }
   }

   result = probe_finish(p, result);
   if (result == PROBE_UNCONFIRMED)
      return result;
   p->ops->bo_destroy(p->be, import);
   if (reserved && !p->ops->va_release(p->be, PROBE_GUEST_WINDOW_BASE,
                                       PROBE_GUEST_WINDOW_END - PROBE_GUEST_WINDOW_BASE))
      result = probe_fail("teardown", "the reservation was not released");
   if (view)
      UnmapViewOfFile(view);
   if (section)
      CloseHandle(section);
   p->ops->bo_destroy(p->be, ib);
   p->ops->queue_destroy(queue);
   probe_step("teardown", "import, reservation, view, buffer and queue released");
   int closed = probe_close(p);
   return result != PROBE_OK ? result : closed;
}

static int
cmd_abi(struct probe *p)
{
   uint8_t fixture[8192];
   size_t bytes = agc_wddm_abi_fixture(fixture, sizeof(fixture));
   if (bytes > sizeof(fixture) || bytes < PROBE_ABI_RUST_FIXTURE_BYTES)
      return probe_fail("abi", "the fixture is %zu bytes", bytes);
   uint64_t rust_hash = XXH64(fixture, PROBE_ABI_RUST_FIXTURE_BYTES, 0);
   uint64_t hash = XXH64(fixture, bytes, 0);
   if (p->dump_path) {
      FILE *file = fopen(p->dump_path, "wb");
      size_t written = file ? fwrite(fixture, 1, bytes, file) : 0;
      if (file)
         fclose(file);
      if (written != bytes)
         return probe_fail("dump", "cannot write %s", p->dump_path);
      probe_step("dump", "%zu bytes to %s", bytes, p->dump_path);
   }
   if (rust_hash != PROBE_ABI_RUST_FIXTURE_XXH64) {
      return probe_fail("abi", "the first %u bytes have xxh64 %016" PRIx64 ", the Rust transport's "
                               "fixture has %016" PRIx64,
                        PROBE_ABI_RUST_FIXTURE_BYTES, rust_hash, PROBE_ABI_RUST_FIXTURE_XXH64);
   }
   probe_step("abi", "graphics context, queue, device allocation and one-IB submission records: %u "
                     "bytes, xxh64 %016" PRIx64 ", identical to the Rust transport's fixture",
              PROBE_ABI_RUST_FIXTURE_BYTES, rust_hash);
   if (bytes != PROBE_ABI_FIXTURE_BYTES || hash != PROBE_ABI_FIXTURE_XXH64) {
      return probe_fail("abi", "all records are %zu bytes with xxh64 %016" PRIx64 ", the checked records "
                               "are %u bytes with %016" PRIx64,
                        bytes, hash, PROBE_ABI_FIXTURE_BYTES, PROBE_ABI_FIXTURE_XXH64);
   }
   probe_step("abi", "with the compute context, cached, fixed-address and host-memory allocation "
                     "and three-IB preamble submission records: %zu bytes, xxh64 %016" PRIx64
                     ", the checked records; the cached allocations keep the last-level cache off, which the "
                     "Rust transport leaves on",
              bytes, hash);
   return PROBE_OK;
}

static int
probe_usage(void)
{
   fprintf(stderr,
           "usage: agc_wddm_probe <info|write|import|compute|abi> [--timeout-ms N] "
           "[--import-mib N] [--dump FILE]\n");
   return PROBE_FAILED;
}

int
main(int argc, char **argv)
{
   struct probe p = {
      .timeout_ns = INT64_C(5000) * 1000000,
      .import_bytes = PROBE_BUFFER_BYTES,
   };
   if (argc < 2 || argc % 2 != 0)
      return probe_usage();

   for (int i = 2; i < argc; i += 2) {
      const char *value = argv[i + 1];
      char *end = NULL;
      if (!strcmp(argv[i], "--timeout-ms")) {
         unsigned long long ms = strtoull(value, &end, 0);
         if (*end || ms == 0 || ms > 600000)
            return probe_usage();
         p.timeout_ns = (int64_t)ms * 1000000;
      } else if (!strcmp(argv[i], "--import-mib")) {
         unsigned long long mib = strtoull(value, &end, 0);
         if (*end || mib == 0 || mib > 4096)
            return probe_usage();
         p.import_bytes = (uint64_t)mib << 20;
      } else if (!strcmp(argv[i], "--dump")) {
         p.dump_path = value;
      } else {
         return probe_usage();
      }
   }

   if (!strcmp(argv[1], "info"))
      return cmd_info(&p);
   if (!strcmp(argv[1], "write"))
      return cmd_write(&p, AGC_QUEUE_GFX);
   if (!strcmp(argv[1], "compute"))
      return cmd_write(&p, AGC_QUEUE_COMPUTE);
   if (!strcmp(argv[1], "import"))
      return cmd_import(&p);
   if (!strcmp(argv[1], "abi"))
      return cmd_abi(&p);
   return probe_usage();
}
