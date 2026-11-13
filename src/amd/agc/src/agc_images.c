#include "agc_images.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac_barrier.h"
#include "ac_cmdbuf.h"
#include "ac_gpu_info.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"
#include "util/u_math.h"

#include "agc_cb_builders.h"
#include "agc_debug.h"
#include "agc_driver_ops.h"
#include "agc_kernels.h"
#include "agc_memory.h"
#include "agc_runtime.h"
#include "agc_surface.h"

#define IMAGES_REPORTS_MAX 1024u
#define IMAGES_BATCHES 8u
#define IMAGES_BATCH_LEVELS 512u
#define IMAGES_PARAM_BYTES (AGC_ELEMENT_COPY_PARAM_DWORDS * 4u)
#define IMAGES_PARAMS_REGION 0x22000u
#define IMAGES_BATCH_BYTES 0x40000u
#define IMAGES_BARRIER_DWORDS 64u
#define IMAGES_TIMEOUT_NS (INT64_C(2000) * 1000000)
#define IMAGES_REPORT_LINES 32u
#define IMAGES_VERDICT_LINES 160u
#define IMAGES_RELEASES_MAX 256u
#define IMAGES_TARGETS_INITIAL 256u
#define IMAGES_HOT_FRAMES 8u
#define IMAGES_COLD_SLICES 16u

_Static_assert(IMAGES_BATCH_LEVELS * IMAGES_PARAM_BYTES <= IMAGES_PARAMS_REGION, "image parameter region");

enum images_kind {
   IMAGES_DIRECT,
   IMAGES_RENDERED,
   IMAGES_CONVERT,
   IMAGES_REFUSED,
   IMAGES_GONE,
};

struct images_batch {
   struct agc_mem memory;
   uint64_t fence;
   uint32_t levels;
   uint32_t *ib;
   uint32_t *cursor;
};

struct images_entry {
   uint64_t base;
   struct agc_surface_desc desc;
   struct agc_surface_layout guest;
   struct agc_surface_layout host;
   uint32_t converted;
   bool refused;
   bool rendered;
   bool gone;
   uint64_t written;
   uint32_t hot;
};

struct images_target {
   uint64_t base;
};

struct images_release {
   uint64_t start;
   uint64_t end;
};

static struct {
   simple_mtx_t lock;
   bool tried;
   bool ready;
   struct agc_backend *backend;
   const struct agc_driver_ops *ops;
   const struct radeon_info *info;
   struct agc_hwq *queue;
   bool queue_set_up;
   struct agc_kernel copy;
   struct agc_mem code;
   struct images_batch batches[IMAGES_BATCHES];
   uint32_t next_batch;
   struct images_batch *open;
   struct images_entry *entries;
   uint32_t entry_count;
   uint32_t entry_capacity;
   simple_mtx_t release_lock;
   struct images_release releases[IMAGES_RELEASES_MAX];
   uint32_t release_count;
   bool releases_overflowed;
   uint32_t report_lines;
   uint32_t verdict_lines;
   struct agc_images_stats stats;
   uint32_t reported[IMAGES_REPORTS_MAX][8];
   bool stored[IMAGES_REPORTS_MAX];
   simple_mtx_t target_lock;
   struct images_target *targets;
   uint32_t target_count;
   uint32_t target_capacity;
   uint32_t frame;
} images = {.lock = SIMPLE_MTX_INITIALIZER,
            .release_lock = SIMPLE_MTX_INITIALIZER,
            .target_lock = SIMPLE_MTX_INITIALIZER};

static void AGC_PRINTF(1, 2)
images_note(const char *format, ...)
{
   if (images.report_lines >= IMAGES_REPORT_LINES)
      return;
   images.report_lines++;
   char text[512];
   va_list args;
   va_start(args, format);
   vsnprintf(text, sizeof(text), format, args);
   va_end(args);
   agc_stat("agc_images", "%s", text);
}

static void AGC_PRINTF(1, 2)
images_verdict(const char *format, ...)
{
   if (images.verdict_lines >= IMAGES_VERDICT_LINES)
      return;
   images.verdict_lines++;
   char text[512];
   va_list args;
   va_start(args, format);
   vsnprintf(text, sizeof(text), format, args);
   va_end(args);
   agc_stat("agc_images", "%s", text);
}

static uint32_t
images_target_slot(uint64_t base, uint32_t capacity)
{
   return (uint32_t)(((base >> 8) * UINT64_C(0x9e3779b97f4a7c15)) >> 32) & (capacity - 1);
}

static struct images_target *
images_target_find(uint64_t base)
{
   if (!images.target_capacity)
      return NULL;
   const uint32_t mask = images.target_capacity - 1;
   for (uint32_t slot = images_target_slot(base, images.target_capacity);; slot = (slot + 1) & mask) {
      struct images_target *target = &images.targets[slot];
      if (target->base == base)
         return target;
      if (!target->base)
         return NULL;
   }
}

static struct images_target *
images_target_add(uint64_t base)
{
   if ((uint64_t)(images.target_count + 1) * 2 > images.target_capacity) {
      const uint32_t capacity = images.target_capacity ? images.target_capacity * 2 : IMAGES_TARGETS_INITIAL;
      struct images_target *grown = calloc(capacity, sizeof(*grown));
      if (!grown)
         return NULL;
      for (uint32_t i = 0; i < images.target_capacity; i++) {
         const struct images_target *old = &images.targets[i];
         if (!old->base)
            continue;
         uint32_t slot = images_target_slot(old->base, capacity);
         while (grown[slot].base)
            slot = (slot + 1) & (capacity - 1);
         grown[slot] = *old;
      }
      free(images.targets);
      images.targets = grown;
      images.target_capacity = capacity;
   }
   uint32_t slot = images_target_slot(base, images.target_capacity);
   while (images.targets[slot].base)
      slot = (slot + 1) & (images.target_capacity - 1);
   images.targets[slot] = (struct images_target){.base = base};
   images.target_count++;
   return &images.targets[slot];
}

void
agc_images_note_target(uint64_t base)
{
   if (!base)
      return;
   simple_mtx_lock(&images.target_lock);
   if (!images_target_find(base))
      images_target_add(base);
   simple_mtx_unlock(&images.target_lock);
}

static bool
images_target_get(uint64_t base, struct images_target *out)
{
   simple_mtx_lock(&images.target_lock);
   const struct images_target *target = images_target_find(base);
   const bool found = target != NULL;
   if (found)
      *out = *target;
   simple_mtx_unlock(&images.target_lock);
   return found;
}

static bool
images_cpu_wrote(uint64_t start, uint64_t bytes, uint64_t since, uint64_t *generation, struct agc_span_list *written)
{
   const struct agc_span range = {.start = start, .end = start + bytes};
   *generation = agc_writes_written_since(&range, 1, since, written);
   return written->count != 0;
}

static bool
images_setup(struct agc_backend *backend, const struct radeon_info *info)
{
   if (images.tried)
      return images.ready;
   images.tried = true;
   if (!backend || !backend->ops || !info)
      return false;
   if (info->gfx_level < GFX11 || info->gb_addr_config != AGC_SURFACE_HOST_GB_ADDR_CONFIG) {
      images_note("texture conversion is implemented for gfx11 hosts with GB_ADDR_CONFIG 0x%x, not gfx level %d "
                  "with 0x%x; S-mode textures are read unconverted",
                  AGC_SURFACE_HOST_GB_ADDR_CONFIG, (int)info->gfx_level, info->gb_addr_config);
      return false;
   }
   images.backend = backend;
   images.ops = backend->ops;
   images.info = info;
   char text[256];
   if (!agc_kernel_compile(info, agc_kernel_element_copy(), &images.copy, text, sizeof(text))) {
      images_note("the element-copy kernel does not compile: %s", text);
      return false;
   }
   const uint64_t code_bytes = align64((uint64_t)images.copy.code_dwords * 4, AGC_KERNEL_CODE_ALIGNMENT);
   if (!agc_mem_alloc(code_bytes, AGC_BO_EXECUTE, &images.code) || images.code.va % AGC_KERNEL_CODE_ALIGNMENT) {
      images_note("no driver memory for the element-copy kernel: %s", agc_memory_last_error());
      return false;
   }
   memcpy(images.code.cpu, images.copy.code, (size_t)images.copy.code_dwords * 4);
   agc_mem_flush_write_combining();
   images.queue = images.ops->queue_create(backend, AGC_QUEUE_COMPUTE);
   if (!images.queue) {
      images_note("the kernel refused the texture conversion queue");
      return false;
   }
   images.ready = true;
   return true;
}

static bool
images_tsharp_null(const uint32_t tsharp[8])
{
   for (unsigned i = 0; i < 8; i++) {
      if (tsharp[i])
         return false;
   }
   return true;
}

static bool images_dispatch_level(const struct agc_surface_layout *guest, uint64_t guest_base,
                                  const struct agc_surface_layout *host, uint64_t host_base, uint32_t level);

static bool
images_same_surface(const struct agc_surface_desc *a, const struct agc_surface_desc *b)
{
   return a->swizzle_mode == b->swizzle_mode && a->width == b->width && a->height == b->height &&
          a->depth == b->depth && a->bytes_per_element == b->bytes_per_element && a->volume == b->volume;
}

static struct images_entry *
images_entry_find(uint64_t base, const struct agc_surface_desc *desc)
{
   for (uint32_t i = 0; i < images.entry_count; i++) {
      struct images_entry *entry = &images.entries[i];
      if (!entry->gone && entry->base == base && !memcmp(&entry->desc, desc, sizeof(*desc)))
         return entry;
   }
   return NULL;
}

static struct images_entry *
images_entry_add(uint64_t base, const struct agc_surface_desc *desc, const struct agc_surface_layout *guest,
                 const struct agc_surface_layout *host)
{
   for (uint32_t i = 0; i < images.entry_count; i++) {
      struct images_entry *other = &images.entries[i];
      if (!other->gone && other->base == base && !images_same_surface(&other->desc, desc))
         other->gone = true;
   }
   if (images.entry_count == images.entry_capacity) {
      uint32_t capacity = images.entry_capacity ? images.entry_capacity * 2 : 64;
      struct images_entry *grown = realloc(images.entries, capacity * sizeof(*grown));
      if (!grown)
         return NULL;
      images.entries = grown;
      images.entry_capacity = capacity;
   }
   struct images_entry *entry = &images.entries[images.entry_count++];
   memset(entry, 0, sizeof(*entry));
   entry->base = base;
   entry->desc = *desc;
   entry->guest = *guest;
   entry->host = *host;
   entry->hot = images.frame + IMAGES_HOT_FRAMES;
   return entry;
}

static bool
images_cpu_mode(uint32_t swizzle_mode)
{
   switch (swizzle_mode) {
   case AGC_SW_256B_S:
   case AGC_SW_256B_D:
   case AGC_SW_4KB_S:
   case AGC_SW_4KB_D:
   case AGC_SW_64KB_S:
   case AGC_SW_64KB_D:
      return true;
   default:
      return false;
   }
}

static bool
images_guest_wrote(uint64_t base, uint64_t bytes, uint32_t swizzle_mode, uint64_t *generation, const char **writer,
                   bool *watch)
{
   struct images_target target;
   *generation = 0;
   *watch = false;
   if (!images_cpu_mode(swizzle_mode)) {
      *writer = "its layout is one the GPU renders in";
      return false;
   }
   if (images_target_get(base, &target)) {
      *writer = "the GPU renders or stores to it";
      images.stats.targets++;
      return false;
   }
   if (!agc_writes_collectable(base, bytes)) {
      *writer = "its memory cannot be write-watched and nothing marks it as written by the GPU";
      images.stats.unwatched++;
      return true;
   }
   struct agc_span_list spans = {0};
   const bool wrote = images_cpu_wrote(base, bytes, 0, generation, &spans);
   agc_span_list_fini(&spans);
   *writer = wrote ? "the CPU wrote it" : "nothing the driver saw wrote it";
   *watch = !wrote;
   return wrote;
}

static enum images_kind
images_classify(const uint32_t tsharp[8], struct agc_image_view *view, struct images_entry **found)
{
   char text[256];
   *found = NULL;
   if (images_tsharp_null(tsharp) || !agc_image_view_decode(tsharp, view, text, sizeof(text)))
      return IMAGES_DIRECT;
   if (agc_image_view_samples_in_place(view))
      return IMAGES_DIRECT;
   struct agc_texel_format texel;
   if (!agc_texel_format_from_gfx10(view->format, &texel, text, sizeof(text)))
      return IMAGES_DIRECT;
   const bool blocks = texel.block_width != 1 || texel.block_height != 1;
   struct agc_surface_desc desc;
   if (!agc_image_view_surface_desc(view, &desc, text, sizeof(text)))
      return blocks ? IMAGES_DIRECT : IMAGES_RENDERED;
   struct agc_surface_desc host_desc = desc;
   host_desc.swizzle_mode = agc_image_view_host_swizzle_mode(view);
   host_desc.pipe_bank_xor = 0;
   host_desc.pitch_elements = 0;
   struct agc_surface_layout guest;
   struct agc_surface_layout host;
   if (!agc_surface_layout_init(&desc, AGC_SURFACE_GUEST, &guest, text, sizeof(text)) ||
       !agc_surface_layout_init(&host_desc, AGC_SURFACE_HOST, &host, text, sizeof(text))) {
      if (!blocks) {
         images_verdict("image 0x%" PRIx64 " type %u format %u %ux%ux%u mode %u has no layouts (%s); it is left as "
                        "the GPU wrote it",
                        view->base, view->type, view->format, view->width, view->height, view->depth,
                        view->swizzle_mode, text);
         return IMAGES_RENDERED;
      }
      images_note("image 0x%" PRIx64 " type %u format %u mode %u has no conversion: %s", view->base, view->type,
                  view->format, view->swizzle_mode, text);
      return IMAGES_REFUSED;
   }
   if (agc_surface_same_addressing(&guest, &host))
      return IMAGES_DIRECT;
   struct images_entry *entry = images_entry_find(view->base, &desc);
   if (entry) {
      *found = entry;
      return entry->rendered ? IMAGES_RENDERED : IMAGES_CONVERT;
   }
   const char *problem = NULL;
   if (!agc_surface_same_placement(&guest, &host))
      problem = "the host layout does not keep the guest placement";
   uint64_t written = 0;
   const char *writer = "block-compressed data comes from the CPU";
   bool by_guest = true;
   bool watch = false;
   if (blocks) {
      struct agc_span_list spans = {0};
      if (agc_writes_collectable(view->base, guest.size))
         images_cpu_wrote(view->base, guest.size, 0, &written, &spans);
      agc_span_list_fini(&spans);
   } else {
      by_guest = images_guest_wrote(view->base, guest.size, view->swizzle_mode, &written, &writer, &watch);
      images_verdict("image 0x%" PRIx64 " type %u format %u %ux%ux%u, %u levels, mode %u -> %u, %" PRIu64
                     " KiB: %s; %s",
                     view->base, view->type, view->format, view->width, view->height, view->depth, guest.mip_count,
                     view->swizzle_mode, host_desc.swizzle_mode, guest.size / 1024, writer,
                     !by_guest ? "left as the GPU wrote it" : problem ? problem : "converted");
   }
   if (!by_guest) {
      if (!problem && watch) {
         entry = images_entry_add(view->base, &desc, &guest, &host);
         if (entry) {
            entry->rendered = true;
            entry->written = written;
            *found = entry;
         }
      }
      return IMAGES_RENDERED;
   }
   if (problem) {
      images_note("image 0x%" PRIx64 " mode %u (host mode %u): %s; it is read unconverted", view->base,
                  view->swizzle_mode, host_desc.swizzle_mode, problem);
      return IMAGES_REFUSED;
   }
   entry = images_entry_add(view->base, &desc, &guest, &host);
   if (!entry)
      return IMAGES_REFUSED;
   entry->written = written;
   if (!blocks)
      images.stats.cpu_converted++;
   *found = entry;
   return IMAGES_CONVERT;
}

static uint32_t
images_layers(const struct agc_surface_layout *layout)
{
   return layout->desc.volume ? 1 : MAX2(layout->desc.layers, 1);
}

static void
images_unit(const struct agc_surface_layout *layout, uint32_t level, uint32_t *first, uint32_t *last)
{
   *first = level;
   *last = level;
   if (!layout->mips[level].in_tail)
      return;
   while (*first > 0 && layout->mips[*first - 1].in_tail)
      (*first)--;
   *last = layout->mip_count - 1;
}

static uint64_t
images_level_start(const struct images_entry *entry, uint32_t level, uint32_t layer)
{
   return entry->base + (uint64_t)layer * entry->guest.slice_stride + entry->guest.mips[level].offset;
}

static bool
images_unit_imported(const struct images_entry *entry, uint32_t level)
{
   for (uint32_t layer = 0; layer < images_layers(&entry->guest); layer++) {
      if (!agc_memory_imported(images_level_start(entry, level, layer), entry->guest.mips[level].size))
         return false;
   }
   return true;
}

static uint32_t
images_unit_mask(uint32_t first, uint32_t last)
{
   return (last >= 31 ? UINT32_MAX : (UINT32_C(1) << (last + 1)) - 1) & ~((UINT32_C(1) << first) - 1);
}

static bool
images_unit_convert(struct images_entry *entry, uint32_t first, uint32_t last)
{
   for (uint32_t layer = 0; layer < images_layers(&entry->guest); layer++) {
      if (!agc_mirror_detach(images_level_start(entry, first, layer), entry->guest.mips[first].size, NULL)) {
         images_note("the mirror of levels %u-%u of image 0x%" PRIx64 " cannot be detached: %s", first, last,
                     entry->base, agc_memory_last_error());
         return false;
      }
   }
   for (uint32_t level = first; level <= last; level++) {
      if (!images_dispatch_level(&entry->guest, entry->base, &entry->host, agc_mirror_va(entry->base), level))
         return false;
   }
   entry->converted |= images_unit_mask(first, last);
   images.stats.bytes += entry->guest.mips[first].size * images_layers(&entry->guest);
   return true;
}

static bool
images_update(struct images_entry *entry, const struct agc_image_view *view)
{
   if (entry->refused)
      return true;
   const uint32_t top = MIN2(view->last_level, entry->guest.mip_count - 1);
   const uint32_t converted = entry->converted;
   bool complete = true;
   for (uint32_t level = agc_image_view_first_reachable_level(view); level <= top; level++) {
      if (entry->converted & (UINT32_C(1) << level))
         continue;
      uint32_t first;
      uint32_t last;
      images_unit(&entry->guest, level, &first, &last);
      if (!images_unit_imported(entry, first)) {
         complete = false;
         continue;
      }
      if (!images_unit_convert(entry, first, last)) {
         entry->refused = true;
         images.stats.refused++;
         return true;
      }
   }
   if (entry->converted != converted) {
      if (!converted)
         images.stats.converted++;
      images_note("image 0x%" PRIx64 " format %u %ux%u, mode %u -> %u: levels 0x%x of 0-%u converted into its mirror",
                  view->base, view->format, view->width, view->height, view->swizzle_mode,
                  entry->host.desc.swizzle_mode, entry->converted, entry->guest.mip_count - 1);
   }
   return complete;
}

static bool
images_convert_missing(struct images_entry *entry)
{
   for (uint32_t level = 0; level < entry->guest.mip_count; level++) {
      if (entry->converted & (UINT32_C(1) << level))
         continue;
      uint32_t first;
      uint32_t last;
      images_unit(&entry->guest, level, &first, &last);
      if (!images_unit_imported(entry, first))
         continue;
      if (!images_unit_convert(entry, first, last)) {
         entry->refused = true;
         images.stats.refused++;
         return false;
      }
   }
   return true;
}

static bool
images_refresh(struct images_entry *entry, const struct agc_span_list *written)
{
   for (uint32_t level = 0; level < entry->guest.mip_count; level++) {
      uint32_t first;
      uint32_t last;
      images_unit(&entry->guest, level, &first, &last);
      if (level != first || !(entry->converted & (UINT32_C(1) << level)))
         continue;
      bool hit = false;
      for (uint32_t layer = 0; !hit && layer < images_layers(&entry->guest); layer++) {
         const uint64_t start = images_level_start(entry, first, layer);
         const uint64_t end = start + entry->guest.mips[first].size;
         for (uint32_t i = 0; !hit && i < written->count; i++)
            hit = written->spans[i].start < end && start < written->spans[i].end;
      }
      if (!hit || !images_unit_imported(entry, first))
         continue;
      for (uint32_t each = first; each <= last; each++) {
         if (!images_dispatch_level(&entry->guest, entry->base, &entry->host, agc_mirror_va(entry->base), each))
            return false;
      }
      images.stats.refreshed++;
   }
   return true;
}

static void
images_check_writes(void)
{
   const uint32_t frame = images.frame;
   for (uint32_t i = 0; i < images.entry_count; i++) {
      struct images_entry *entry = &images.entries[i];
      if (entry->gone || entry->refused)
         continue;
      const bool due = (int32_t)(entry->hot - frame) >= 0 ||
                       (entry->base >> 16) % IMAGES_COLD_SLICES == frame % IMAGES_COLD_SLICES;
      if (!due)
         continue;
      if (!entry->rendered && !images_convert_missing(entry))
         continue;
      images.stats.checks++;
      if (!agc_writes_collectable(entry->base, entry->guest.size)) {
         images.stats.blind++;
         continue;
      }
      struct agc_span_list written = {0};
      uint64_t generation = 0;
      const bool wrote = images_cpu_wrote(entry->base, entry->guest.size, entry->written, &generation, &written);
      entry->written = generation;
      if (wrote) {
         images.stats.wrote++;
         entry->hot = frame + IMAGES_HOT_FRAMES;
         struct images_target target;
         if (!entry->rendered) {
            images_refresh(entry, &written);
         } else if (images_target_get(entry->base, &target)) {
            entry->gone = true;
            images.stats.kept++;
         } else {
            entry->rendered = false;
            images.stats.turned++;
            images_verdict("image 0x%" PRIx64 " %ux%ux%u mode %u: the CPU wrote it after it was first used; it is "
                           "converted from now on",
                           entry->base, entry->desc.width, entry->desc.height, entry->desc.depth,
                           entry->desc.swizzle_mode);
            images_convert_missing(entry);
         }
      }
      agc_span_list_fini(&written);
   }
}

static void
images_release_pending(void)
{
   struct images_release releases[IMAGES_RELEASES_MAX];
   simple_mtx_lock(&images.release_lock);
   const uint32_t count = images.release_count;
   const bool overflowed = images.releases_overflowed;
   memcpy(releases, images.releases, count * sizeof(releases[0]));
   images.release_count = 0;
   images.releases_overflowed = false;
   simple_mtx_unlock(&images.release_lock);
   if (!count && !overflowed)
      return;
   for (uint32_t i = 0; i < images.entry_count; i++) {
      struct images_entry *entry = &images.entries[i];
      for (uint32_t level = 0; level < entry->guest.mip_count; level++) {
         if (!(entry->converted & (UINT32_C(1) << level)))
            continue;
         uint32_t first;
         uint32_t last;
         images_unit(&entry->guest, level, &first, &last);
         bool hit = overflowed;
         for (uint32_t layer = 0; !hit && layer < images_layers(&entry->guest); layer++) {
            const uint64_t start = images_level_start(entry, first, layer);
            const uint64_t end = start + entry->guest.mips[first].size;
            for (uint32_t r = 0; !hit && r < count; r++)
               hit = releases[r].start < end && start < releases[r].end;
         }
         if (!hit)
            continue;
         for (uint32_t layer = 0; layer < images_layers(&entry->guest); layer++)
            agc_mirror_attach(images_level_start(entry, first, layer), entry->guest.mips[first].size);
         entry->converted &= ~images_unit_mask(first, last);
         images.stats.released++;
      }
   }
}

void
agc_images_released(void *user, uint64_t guest_va, uint64_t bytes)
{
   (void)user;
   simple_mtx_lock(&images.release_lock);
   if (images.release_count < IMAGES_RELEASES_MAX)
      images.releases[images.release_count++] = (struct images_release){.start = guest_va, .end = guest_va + bytes};
   else
      images.releases_overflowed = true;
   simple_mtx_unlock(&images.release_lock);
}

static uint32_t
images_emit_barrier(uint32_t *out, enum ac_barrier_flags flags)
{
   struct ac_cmdbuf cs = {.buf = out, .max_dw = IMAGES_BARRIER_DWORDS};
   const struct ac_barrier_state state = {.flags = flags};
   bool context_roll = false;
   enum ac_rgp_flush_bits rgp = 0;
   ac_emit_barrier(&cs, images.info->gfx_level, AMD_IP_COMPUTE, &state, &context_roll, &rgp);
   return cs.cdw;
}

static struct images_batch *
images_batch_open(void)
{
   if (images.open)
      return images.open;
   struct images_batch *batch = &images.batches[images.next_batch];
   if (batch->fence && !images.ops->fence_wait(images.queue, batch->fence, IMAGES_TIMEOUT_NS)) {
      images_note("an earlier texture conversion did not finish");
      return NULL;
   }
   if (!batch->memory.va && !agc_mem_alloc(IMAGES_BATCH_BYTES, AGC_BO_GL2_BYPASS, &batch->memory)) {
      images_note("no driver memory for texture conversions: %s", agc_memory_last_error());
      return NULL;
   }
   batch->fence = 0;
   batch->levels = 0;
   batch->ib = (uint32_t *)((uint8_t *)batch->memory.cpu + IMAGES_PARAMS_REGION);
   uint32_t *p = batch->ib;
   if (!images.queue_set_up) {
      p = agc_kernel_emit_queue_setup(p, images.info, true);
      if (!p) {
         images_note("the conversion queue setup could not be emitted");
         return NULL;
      }
   }
   p += images_emit_barrier(p, AC_BARRIER_INV_ICACHE | AC_BARRIER_INV_SMEM | AC_BARRIER_INV_VMEM |
                                   AC_BARRIER_INV_L2);
   batch->cursor = p;
   images.open = batch;
   return batch;
}

static bool
images_batch_submit(void)
{
   struct images_batch *batch = images.open;
   if (!batch)
      return true;
   images.open = NULL;
   if (!batch->levels)
      return true;
   uint32_t *p = batch->cursor;
   p += images_emit_barrier(p, AC_BARRIER_SYNC_CS | AC_BARRIER_WB_L2);
   const uint32_t multiple = MAX2(images.info->ip[AMD_IP_COMPUTE].ib_pad_dw_mask + 1, 8u);
   uint32_t dwords = (uint32_t)(p - batch->ib);
   uint32_t padded = (dwords + multiple - 1) / multiple * multiple;
   if (padded - dwords == 1)
      padded += multiple;
   agc_fill_nops(p, padded - dwords);
   agc_mem_flush_write_combining();
   const struct agc_ib ib = {
      .va = batch->memory.va + IMAGES_PARAMS_REGION,
      .dwords = padded,
      .flags = AGC_IB_MAIN,
   };
   const uint64_t fence = images.ops->submit(images.queue, &ib, 1);
   if (!fence) {
      images_note("the kernel refused a texture conversion");
      return false;
   }
   images.queue_set_up = true;
   batch->fence = fence;
   images.next_batch = (images.next_batch + 1) % IMAGES_BATCHES;
   images.stats.batches++;
   if (!images.ops->fence_wait(images.queue, fence, IMAGES_TIMEOUT_NS)) {
      uint64_t fault = 0;
      const enum agc_device_state state = images.ops->device_state(images.backend, &fault);
      agc_stat("agc_images", "a conversion batch of %u levels did not finish within %" PRId64 " ms (device state %d, "
               "fault address 0x%" PRIx64 "); texture conversion stops",
               batch->levels, IMAGES_TIMEOUT_NS / 1000000, (int)state, fault);
      images.ready = false;
      return false;
   }
   return true;
}

static bool
images_dispatch_level(const struct agc_surface_layout *guest, uint64_t guest_base,
                      const struct agc_surface_layout *host, uint64_t host_base, uint32_t level)
{
   struct images_batch *batch = images_batch_open();
   if (!batch)
      return false;
   const uint32_t room = agc_kernel_dispatch_dwords(images.info, &images.copy);
   const uint32_t *end = (const uint32_t *)((uint8_t *)batch->memory.cpu + IMAGES_BATCH_BYTES);
   if (batch->levels == IMAGES_BATCH_LEVELS || batch->cursor + room + 2 * IMAGES_BARRIER_DWORDS + 16 > end) {
      if (!images_batch_submit())
         return false;
      batch = images_batch_open();
      if (!batch)
         return false;
   }
   uint32_t *params = (uint32_t *)((uint8_t *)batch->memory.cpu + batch->levels * IMAGES_PARAM_BYTES);
   uint32_t groups[3];
   char text[256];
   if (!agc_surface_copy_params(guest, guest_base, host, host_base, level, params, groups, text, sizeof(text))) {
      images_note("level %u of image 0x%" PRIx64 " has no copy: %s", level, guest_base, text);
      return false;
   }
   if (!groups[0] || !groups[1] || !groups[2])
      return true;
   const uint64_t params_va = batch->memory.va + batch->levels * IMAGES_PARAM_BYTES;
   const uint32_t user_data[AGC_ELEMENT_COPY_USER_DATA_DWORDS] = {(uint32_t)params_va, (uint32_t)(params_va >> 32)};
   uint32_t *p = agc_kernel_emit_dispatch(batch->cursor, images.info, &images.copy, images.code.va, user_data,
                                          groups[0], groups[1], groups[2]);
   if (!p) {
      images_note("the element-copy dispatch could not be emitted");
      return false;
   }
   batch->cursor = p;
   batch->levels++;
   images.stats.levels++;
   return true;
}

void
agc_images_frame(struct agc_backend *backend, const struct radeon_info *info)
{
   simple_mtx_lock(&images.lock);
   p_atomic_inc(&images.frame);
   const uint32_t count = agc_runtime_image_reports_take(images.reported, images.stored, IMAGES_REPORTS_MAX);
   for (uint32_t i = 0; i < count; i++) {
      if (!images.stored[i])
         continue;
      struct agc_image_view view;
      char text[256];
      if (!images_tsharp_null(images.reported[i]) &&
          agc_image_view_decode(images.reported[i], &view, text, sizeof(text)))
         agc_images_note_target(view.base);
      images.stats.stores++;
      agc_runtime_image_mark(images.reported[i], true);
   }
   if ((!count && !images.entry_count) || !images_setup(backend, info)) {
      simple_mtx_unlock(&images.lock);
      return;
   }
   images_release_pending();
   for (uint32_t i = 0; i < count; i++) {
      if (images.stored[i])
         continue;
      const uint32_t *tsharp = images.reported[i];
      struct agc_image_view view;
      struct images_entry *entry;
      images.stats.reports++;
      const enum images_kind kind = images_classify(tsharp, &view, &entry);
      switch (kind) {
      case IMAGES_GONE:
         continue;
      case IMAGES_CONVERT:
         if (entry && !images_update(entry, &view)) {
            images.stats.waiting++;
            continue;
         }
         break;
      case IMAGES_RENDERED:
         images.stats.rendered++;
         break;
      case IMAGES_REFUSED:
         images.stats.refused++;
         break;
      case IMAGES_DIRECT:
         images.stats.direct++;
         break;
      }
      agc_runtime_image_mark(tsharp, false);
   }
   images_check_writes();
   images_batch_submit();
   simple_mtx_unlock(&images.lock);
}

void
agc_images_stats(struct agc_images_stats *stats)
{
   simple_mtx_lock(&images.lock);
   *stats = images.stats;
   simple_mtx_unlock(&images.lock);
}

void
agc_images_fini(void)
{
   simple_mtx_lock(&images.lock);
   if (images.ready) {
      for (uint32_t i = 0; i < IMAGES_BATCHES; i++) {
         if (images.batches[i].fence)
            images.ops->fence_wait(images.queue, images.batches[i].fence, IMAGES_TIMEOUT_NS);
      }
      images.ops->queue_destroy(images.queue);
      for (uint32_t i = 0; i < IMAGES_BATCHES; i++)
         agc_mem_free(&images.batches[i].memory);
      agc_mem_free(&images.code);
   }
   agc_kernel_free(&images.copy);
   free(images.entries);
   memset(&images.stats, 0, sizeof(images.stats));
   images.entries = NULL;
   images.entry_count = 0;
   images.entry_capacity = 0;
   images.ready = false;
   images.tried = false;
   simple_mtx_lock(&images.target_lock);
   free(images.targets);
   images.targets = NULL;
   images.target_count = 0;
   images.target_capacity = 0;
   simple_mtx_unlock(&images.target_lock);
   simple_mtx_unlock(&images.lock);
}
