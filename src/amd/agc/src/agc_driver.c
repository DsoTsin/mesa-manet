#include "agc_driver.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include "c11/threads.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"
#include "util/u_thread.h"

#include "ac_barrier.h"
#include "ac_cmdbuf.h"
#include "ac_gpu_info.h"
#include "sid.h"

#include "agc_anyps5.h"
#include "agc_cb_builders.h"
#include "agc_debug.h"
#include "agc_driver_ops.h"
#include "agc_graphics.h"
#include "agc_images.h"
#include "agc_memory.h"
#include "agc_queue_state.h"
#include "agc_record.h"
#include "agc_register_defaults.h"
#include "agc_runtime.h"
#include "agc_videoout.h"

#define QUEUE_SLOTS AGC_QUEUE_COMPUTE_END
#define IB_DWORDS_MAX UINT32_C(0xfffff)
#define ARENA_CHUNK_BYTES (UINT64_C(4) << 20)
#define ARENA_ALIGNMENT UINT64_C(16)
#define CHAIN_DWORDS 4u
#define TESS_REGISTERS 4u
#define PROLOGUE_DWORDS (AGC_DRAW_STATE_COUNT * AGC_DRAW_STATE_MAX_DWORDS + 3u * TESS_REGISTERS)
#define FENCE_POLL_NS INT64_C(2000000000)
#define LABEL_YIELD_NS INT64_C(4000000)
#define RENDERING_WAIT_POLL_MS 100u
#define FLIP_HANDLES 64u
#define FLIP_INDICES 17u
#define LABEL_LOG_ENTRIES 1024u
#define LABEL_RELEASE_REPORTS 8u
#define FOLLOW_TRACE_MAX 512u
#define SUBMIT_LOG_ENTRIES 256u
#define SUBMIT_LOG_REPORTED 64u
#define FLIP_MISSING_REPORTS 4u

enum follow_end {
   FOLLOW_END_NONE,
   FOLLOW_END_UNRESOLVED,
   FOLLOW_END_NO_CHAIN,
   FOLLOW_END_SCAN_STOPPED,
   FOLLOW_END_BUDGET,
};

struct agc_step {
   struct agc_step *next;
   uint64_t va;
   uint64_t sequence;
   uint64_t fence;
   uint64_t ib_va;
   uint32_t ib_count;
   uint32_t dwords;
   uint32_t wait_handle;
   uint32_t wait_index;
   uint32_t event_count;
   uint32_t event_capacity;
   bool wait;
   uint32_t prologue_dwords;
   uint32_t prologue[PROLOGUE_DWORDS];
   bool label;
   uint8_t label_function;
   bool label_64;
   uint64_t label_va;
   uint64_t label_reference;
   uint64_t label_mask;
   uint64_t label_since;
   uint64_t stamp;
   uint32_t label_skip;
   uint32_t write_count;
   uint32_t write_capacity;
   struct label_write *writes;
   struct agc_event *events;
};

struct agc_chunk {
   struct agc_mem mem;
   uint64_t used;
   uint64_t last_step;
};

struct label_write {
   uint64_t va;
   uint64_t value;
   bool known;
   bool wide;
};

struct label_done {
   uint64_t va;
   uint64_t value;
   uint64_t stamp;
   bool wide;
};

static struct {
   simple_mtx_t lock;
   uint64_t serial;
   uint64_t stamp;
   struct label_done entries[LABEL_LOG_ENTRIES];
} label_log = {.lock = SIMPLE_MTX_INITIALIZER};

struct draw_state {
   bool valid[AGC_DRAW_STATE_COUNT];
   uint32_t words[AGC_DRAW_STATE_COUNT][AGC_DRAW_STATE_MAX_DWORDS];
};

struct agc_queue {
   uint32_t id;
   struct draw_state draw;
   struct label_write *writes;
   uint32_t write_count;
   uint32_t write_capacity;
   struct agc_hwq *hwq;
   mtx_t lock;
   cnd_t submit_wake;
   cnd_t complete_wake;
   cnd_t idle_wake;
   thrd_t submit_thread;
   thrd_t complete_thread;
   struct agc_step *ready_head;
   struct agc_step *ready_tail;
   struct agc_step *pending_head;
   struct agc_step *pending_tail;
   uint64_t enqueued;
   uint64_t completed;
   uint64_t last_fence;
   struct agc_chunk *chunks;
   uint32_t chunk_count;
   uint32_t chunk_capacity;
   uint32_t chunk_current;
   uint32_t ib_start_bytes;
   uint32_t ib_dwords;
   bool submitting;
   bool stop;
   bool shadowed;
   struct agc_queue_state state;
   struct agc_mem cache_ibs;
   uint32_t head_dwords;
   uint32_t tail_dwords;
   uint32_t cache_ib_stride;
   bool context_known;
   struct agc_reg_shadow context;
   struct agc_reg_shadow context_saved;
   uint32_t tess_applied;
   struct agc_step *active;
   uint32_t flips_settled;
   uint32_t flip_records_seen;
   uint32_t trace_count;
   uint64_t trace_va[FOLLOW_TRACE_MAX];
   uint32_t trace_first[FOLLOW_TRACE_MAX];
   uint32_t trace_limit[FOLLOW_TRACE_MAX];
   uint8_t trace_depth[FOLLOW_TRACE_MAX];
   uint64_t trace_end;
   enum follow_end trace_end_reason;
};

static struct {
   mtx_t lock;
   mtx_t flip_lock;
   cnd_t flip_wake;
   bool started;
   bool start_failed;
   bool device_open;
   bool device_failed;
   bool stopping;
   char start_error[256];
   struct agc_queue *graphics;
   struct agc_backend *backend;
   const struct agc_anyps5 *anyps5;
   struct radeon_info info;
   struct agc_queue *queues[QUEUE_SLOTS];
   uint32_t flip_pending[FLIP_HANDLES][FLIP_INDICES];
   struct agc_driver_counters counters;
   const struct agc_host_defaults *defaults;
   struct {
      simple_mtx_t lock;
      uint32_t generation;
      bool has_ring;
      bool has_offchip;
      uint64_t base;
      uint32_t bytes;
      uint32_t offchip;
      struct agc_mem fallback;
   } tess;
} driver = {.tess = {.lock = SIMPLE_MTX_INITIALIZER, .generation = 1}};

static once_flag driver_once = ONCE_FLAG_INIT;

static void
driver_init_locks(void)
{
   mtx_init(&driver.lock, mtx_plain);
   mtx_init(&driver.flip_lock, mtx_plain);
   cnd_init(&driver.flip_wake);
}

void
agc_runtime_anchor(void)
{
}

static void
count(uint64_t *counter, uint64_t amount)
{
   p_atomic_add(counter, amount);
}

void
agc_driver_get_counters(struct agc_driver_counters *counters)
{
   uint64_t *out = (uint64_t *)counters;
   uint64_t *in = (uint64_t *)&driver.counters;
   for (size_t i = 0; i < sizeof(*counters) / sizeof(uint64_t); i++)
      out[i] = p_atomic_read(&in[i]);
}

static const char *
device_state_name(enum agc_device_state state)
{
   switch (state) {
   case AGC_DEVICE_ACTIVE:
      return "active";
   case AGC_DEVICE_HUNG:
      return "hung";
   case AGC_DEVICE_PAGE_FAULT:
      return "page fault";
   case AGC_DEVICE_REMOVED:
      return "removed";
   default:
      return "unknown";
   }
}

static enum agc_device_state
describe_device(char *out, size_t bytes)
{
   uint64_t fault = 0;
   enum agc_device_state state = driver.backend->ops->device_state(driver.backend, &fault);
   if (state == AGC_DEVICE_PAGE_FAULT)
      snprintf(out, bytes, "device state: page fault at 0x%llx", (unsigned long long)fault);
   else
      snprintf(out, bytes, "device state: %s", device_state_name(state));
   return state;
}

void
agc_driver_fail(const char *format, ...)
{
   char message[512];
   va_list args;
   va_start(args, format);
   vsnprintf(message, sizeof(message), format, args);
   va_end(args);
   agc_report("agc_driver", "%s", message);
   agc_flips_fail(message);
}

static bool
failed(void)
{
   return agc_flips_failure() != NULL || p_atomic_read(&driver.stopping);
}

static bool
flip_slot(uint32_t handle, int32_t index, uint32_t **slot)
{
   if (handle >= FLIP_HANDLES || index < -1 || index >= (int32_t)FLIP_INDICES - 1)
      return false;
   *slot = &driver.flip_pending[handle][index + 1];
   return true;
}

static void
flips_submitted(const struct agc_step *step, int direction)
{
   bool any = false;
   mtx_lock(&driver.flip_lock);
   for (uint32_t i = 0; i < step->event_count; i++) {
      uint32_t *slot;
      if (step->events[i].kind != AGC_EVENT_FLIP)
         continue;
      if (!flip_slot(step->events[i].flip.handle, step->events[i].flip.index, &slot))
         continue;
      *slot += direction;
      any = true;
   }
   if (any && direction < 0)
      cnd_broadcast(&driver.flip_wake);
   mtx_unlock(&driver.flip_lock);
}

static void
step_free(struct agc_step *step)
{
   free(step->writes);
   free(step->events);
   free(step);
}

static bool
step_add_event(struct agc_step *step, const struct agc_event *event)
{
   if (step->event_count == step->event_capacity) {
      uint32_t capacity = step->event_capacity ? step->event_capacity * 2 : 4;
      struct agc_event *events = realloc(step->events, capacity * sizeof(*events));
      if (!events)
         return false;
      step->events = events;
      step->event_capacity = capacity;
   }
   step->events[step->event_count++] = *event;
   return true;
}

static struct agc_chunk *
arena_next_chunk(struct agc_queue *queue, uint64_t bytes)
{
   for (uint32_t i = 0; i < queue->chunk_count; i++) {
      struct agc_chunk *chunk = &queue->chunks[i];
      if (i == queue->chunk_current || chunk->last_step > queue->completed || chunk->mem.bytes < bytes)
         continue;
      chunk->used = 0;
      queue->chunk_current = i;
      return chunk;
   }
   if (queue->chunk_count == queue->chunk_capacity) {
      uint32_t capacity = queue->chunk_capacity ? queue->chunk_capacity * 2 : 8;
      struct agc_chunk *chunks = realloc(queue->chunks, capacity * sizeof(*chunks));
      if (!chunks)
         return NULL;
      queue->chunks = chunks;
      queue->chunk_capacity = capacity;
   }
   struct agc_chunk *chunk = &queue->chunks[queue->chunk_count];
   uint64_t size = bytes > ARENA_CHUNK_BYTES ? (bytes + 0xffff) & ~UINT64_C(0xffff) : ARENA_CHUNK_BYTES;
   memset(chunk, 0, sizeof(*chunk));
   if (!agc_mem_alloc(size, 0, &chunk->mem))
      return NULL;
   count(&driver.counters.arena_bytes, size);
   queue->chunk_current = queue->chunk_count++;
   return chunk;
}

static void *
arena_alloc(struct agc_queue *queue, uint64_t bytes, uint64_t alignment, uint64_t sequence, uint64_t *va)
{
   bytes = (bytes + ARENA_ALIGNMENT - 1) & ~(ARENA_ALIGNMENT - 1);
   if (alignment < ARENA_ALIGNMENT)
      alignment = ARENA_ALIGNMENT;
   struct agc_chunk *chunk = queue->chunk_count ? &queue->chunks[queue->chunk_current] : NULL;
   uint64_t start = chunk ? (chunk->used + alignment - 1) & ~(alignment - 1) : 0;
   if (!chunk || start > chunk->mem.bytes || chunk->mem.bytes - start < bytes) {
      chunk = arena_next_chunk(queue, bytes);
      if (!chunk)
         return NULL;
      start = 0;
   }
   void *cpu = (char *)chunk->mem.cpu + start;
   *va = chunk->mem.va + start;
   chunk->used = start + bytes;
   chunk->last_step = sequence;
   return cpu;
}

static void
arena_trim(struct agc_queue *queue, uint64_t bytes)
{
   bytes &= ~(ARENA_ALIGNMENT - 1);
   queue->chunks[queue->chunk_current].used -= bytes;
}

static bool condition_met(uint32_t function, uint64_t reference, uint64_t mask, uint64_t value);

static bool
label_met(const struct agc_step *step, uint64_t value)
{
   return condition_met(step->label_function, step->label_reference, step->label_mask, value);
}

static const volatile uint32_t *
label_cpu(const struct agc_step *step)
{
   if (agc_va_in_guest_window(step->label_va))
      return (const volatile uint32_t *)(uintptr_t)step->label_va;
   return agc_memory_cpu_of(step->label_va, step->label_64 ? 8 : 4);
}

static uint64_t
label_read(const struct agc_step *step)
{
   const volatile uint32_t *word = label_cpu(step);
   if (!word)
      return UINT64_MAX;
   uint64_t value = word[0];
   if (step->label_64)
      value |= (uint64_t)word[1] << 32;
   return value;
}

static void
label_log_note(const struct agc_step *step)
{
   if (!step->write_count)
      return;
   simple_mtx_lock(&label_log.lock);
   for (uint32_t i = 0; i < step->write_count; i++) {
      label_log.entries[label_log.serial % LABEL_LOG_ENTRIES] = (struct label_done){
         .va = step->writes[i].va,
         .value = step->writes[i].value,
         .stamp = step->stamp,
         .wide = step->writes[i].wide,
      };
      p_atomic_set(&label_log.serial, label_log.serial + 1);
   }
   simple_mtx_unlock(&label_log.lock);
}

static bool
label_done(struct agc_queue *queue, struct agc_step *step)
{
   if (p_atomic_read(&label_log.serial) == step->label_since)
      return false;
   bool met = false;
   simple_mtx_lock(&label_log.lock);
   const uint64_t last = label_log.serial;
   uint64_t at = step->label_since;
   if (last - at > LABEL_LOG_ENTRIES)
      at = last - LABEL_LOG_ENTRIES;
   for (; at < last && !met; at++) {
      const struct label_done *done = &label_log.entries[at % LABEL_LOG_ENTRIES];
      met = done->stamp > step->stamp && done->va == step->label_va && (done->wide || !step->label_64) &&
            label_met(step, done->value);
   }
   simple_mtx_unlock(&label_log.lock);
   step->label_since = last;
   if (!met)
      return false;
   uint64_t releases = p_atomic_inc_return(&driver.counters.label_releases);
   if (releases <= LABEL_RELEASE_REPORTS)
      agc_stat("agc_driver", "queue 0x%x: step %llu goes on because the step that writes label 0x%llx has finished; "
                 "the label reads 0x%llx (function %u, reference 0x%llx)%s",
                 queue->id, (unsigned long long)step->sequence, (unsigned long long)step->label_va,
                 (unsigned long long)label_read(step), step->label_function,
                 (unsigned long long)step->label_reference,
                 releases == LABEL_RELEASE_REPORTS ? "; later cases are only counted" : "");
   return true;
}

static bool
label_wait(struct agc_queue *queue, struct agc_step *step)
{
   if (!label_cpu(step)) {
      agc_report("agc_driver", "queue 0x%x: label 0x%llx is not in readable memory; the GPU waits for it", queue->id,
                 (unsigned long long)step->label_va);
      return true;
   }
   count(&driver.counters.label_waits, 1);
   if (label_met(step, label_read(step)) || label_done(queue, step))
      return true;
   int64_t start = os_time_get_nano();
   int64_t reported = start;
   for (uint32_t spins = 0;; spins++) {
      if (label_met(step, label_read(step)) || label_done(queue, step))
         break;
      if (failed() || queue->stop)
         return false;
      if (spins < 4096) {
         _mm_pause();
         continue;
      }
      int64_t now = os_time_get_nano();
      if (now - reported > INT64_C(2000000000)) {
         reported = now;
         agc_stat("agc_driver", "queue 0x%x: step %llu waits for label 0x%llx (now 0x%llx, function %u, reference 0x%llx, "
                    "mask 0x%llx) for %lld ms",
                    queue->id, (unsigned long long)step->sequence, (unsigned long long)step->label_va,
                    (unsigned long long)label_read(step), step->label_function,
                    (unsigned long long)step->label_reference, (unsigned long long)step->label_mask,
                    (long long)((now - start) / 1000000));
      }
      if (now - start < LABEL_YIELD_NS)
         thrd_yield();
      else
         os_time_sleep(1000);
   }
   count(&driver.counters.label_wait_ns, (uint64_t)(os_time_get_nano() - start));
   return true;
}

static bool
rendering_wait(struct agc_queue *queue, const struct agc_step *step)
{
   struct agc_driver_error error = {0};
   struct agc_rendering_wait *wait = NULL;
   uint32_t *slot;

   count(&driver.counters.rendering_waits, 1);
   if (flip_slot(step->wait_handle, (int32_t)step->wait_index, &slot)) {
      mtx_lock(&driver.flip_lock);
      while (*slot != 0 && !failed() && !queue->stop) {
         struct timespec until;
         timespec_get(&until, TIME_UTC);
         until.tv_nsec += RENDERING_WAIT_POLL_MS * 1000000;
         if (until.tv_nsec >= 1000000000) {
            until.tv_sec++;
            until.tv_nsec -= 1000000000;
         }
         cnd_timedwait(&driver.flip_wake, &driver.flip_lock, &until);
      }
      mtx_unlock(&driver.flip_lock);
   }
   if (failed() || queue->stop)
      return false;
   const int captured = agc_flips_rendering_wait(step->wait_handle, step->wait_index, &wait, &error);
   if (captured == AGC_DRIVER_SHUTDOWN)
      return false;
   if (captured != AGC_DRIVER_OK) {
      agc_driver_fail("rendering wait for buffer %u of output %u: %s", step->wait_index, step->wait_handle,
                      error.message);
      return false;
   }
   for (;;) {
      int result = agc_rendering_wait_for(wait, RENDERING_WAIT_POLL_MS, &error);
      if (result == 1)
         break;
      if (result == AGC_DRIVER_SHUTDOWN) {
         agc_rendering_wait_release(wait);
         return false;
      }
      if (result != 0) {
         agc_rendering_wait_release(wait);
         agc_driver_fail("rendering wait for buffer %u of output %u: %s", step->wait_index,
                         step->wait_handle, error.message);
         return false;
      }
      if (failed() || queue->stop) {
         agc_rendering_wait_release(wait);
         return false;
      }
   }
   agc_rendering_wait_release(wait);
   return true;
}

static uint32_t
packet_dwords(uint32_t header)
{
   if (header == PKT3_NOP_PAD)
      return 1;
   switch (header >> 30) {
   case 3:
      return ((header >> 16) & 0x3fff) + 2;
   case 2:
      return 1;
   default:
      return 0;
   }
}

static uint32_t
split_range(uint64_t va, uint32_t dwords, struct agc_ib *pieces)
{
   const uint32_t *words = (const uint32_t *)(uintptr_t)va;
   uint32_t count = 0;
   uint32_t start = 0;
   uint32_t at = 0;
   if (dwords <= IB_DWORDS_MAX) {
      pieces[0] = (struct agc_ib){.va = va, .dwords = dwords};
      return 1;
   }
   while (at < dwords) {
      uint32_t size = packet_dwords(words[at]);
      if (!size || size > dwords - at || size > IB_DWORDS_MAX)
         return 0;
      if (at + size - start > IB_DWORDS_MAX) {
         if (count == AGC_SUBMIT_MAX_IBS - 4)
            return 0;
         pieces[count++] = (struct agc_ib){.va = va + (uint64_t)start * 4, .dwords = at - start};
         start = at;
      }
      at += size;
   }
   pieces[count++] = (struct agc_ib){.va = va + (uint64_t)start * 4, .dwords = dwords - start};
   return count;
}

static uint32_t
ib_stride(const struct agc_queue *queue)
{
   uint32_t bytes = queue->ib_dwords * 4;
   return (bytes + queue->ib_start_bytes - 1) / queue->ib_start_bytes * queue->ib_start_bytes;
}

static bool
prepare_ibs(struct agc_queue *queue, struct agc_step *step, uint64_t sequence, const char *function)
{
   struct agc_ib pieces[AGC_SUBMIT_MAX_IBS];
   uint32_t count = split_range(step->va, step->dwords, pieces);
   if (!count) {
      agc_report(function, "a command range of %u dwords at 0x%llx cannot be split into indirect buffers",
                 step->dwords, (unsigned long long)step->va);
      return false;
   }
   uint32_t stride = ib_stride(queue);
   uint64_t va;
   uint32_t *cpu = arena_alloc(queue, (uint64_t)count * stride, queue->ib_start_bytes, sequence, &va);
   if (!cpu) {
      agc_report(function, "out of driver memory for %u indirect buffers", count);
      return false;
   }
   for (uint32_t i = 0; i < count; i++) {
      uint32_t *ib = cpu + (size_t)i * stride / 4;
      uint32_t *chain = ib + queue->ib_dwords - CHAIN_DWORDS;
      uint32_t prologue = i == 0 ? step->prologue_dwords : 0;
      memcpy(ib, step->prologue, prologue * sizeof(uint32_t));
      agc_fill_nops(ib + prologue, queue->ib_dwords - CHAIN_DWORDS - prologue);
      chain[0] = PKT3(PKT3_INDIRECT_BUFFER, 2, 0);
      chain[1] = (uint32_t)pieces[i].va;
      chain[2] = (uint32_t)(pieces[i].va >> 32);
      chain[3] = S_3F3_IB_SIZE(pieces[i].dwords) | S_3F3_CHAIN(1) | S_3F3_VALID(1);
   }
   step->ib_va = va;
   step->ib_count = count;
   return true;
}

static void note_submit(const struct agc_queue *queue, const struct agc_step *step);
static void dump_step(struct agc_queue *queue, const struct agc_step *step, const char *tag);
static bool dump_step_wanted(const struct agc_queue *queue, const struct agc_step *step);

#define CACHE_IB_BYTES 256u

static uint32_t
cache_ib_emit(struct agc_queue *queue, uint32_t *out, enum ac_barrier_flags flags)
{
   const bool graphics = queue->id == AGC_QUEUE_GRAPHICS;
   const enum amd_ip_type ip = graphics ? AMD_IP_GFX : AMD_IP_COMPUTE;
   const uint32_t multiple = driver.info.ip[ip].ib_pad_dw_mask + 1;
   struct ac_cmdbuf cs = {.buf = out, .max_dw = CACHE_IB_BYTES / 4};
   const struct ac_barrier_state state = {.flags = flags, .pws_acquire_point = AC_PWS_ACQUIRE_POINT_ME};
   bool context_roll = false;
   enum ac_rgp_flush_bits rgp = 0;
   ac_emit_barrier(&cs, driver.info.gfx_level, ip, &state, &context_roll, &rgp);
   uint32_t dwords = (cs.cdw + multiple - 1) / multiple * multiple;
   if (dwords - cs.cdw == 1)
      dwords += multiple;
   agc_fill_nops(out + cs.cdw, dwords - cs.cdw);
   return dwords;
}

static bool
cache_ibs_ready(struct agc_queue *queue)
{
   if (queue->cache_ibs.va)
      return true;
   queue->cache_ib_stride = MAX2(CACHE_IB_BYTES, queue->ib_start_bytes);
   if (!agc_mem_alloc(2 * queue->cache_ib_stride, AGC_BO_GL2_BYPASS, &queue->cache_ibs)) {
      agc_driver_fail("queue 0x%x: no driver memory for the cache maintenance IBs: %s", queue->id,
                      agc_memory_last_error());
      return false;
   }
   const bool graphics = queue->id == AGC_QUEUE_GRAPHICS;
   uint32_t *head = queue->cache_ibs.cpu;
   uint32_t *tail = (uint32_t *)((uint8_t *)queue->cache_ibs.cpu + queue->cache_ib_stride);
   queue->head_dwords = cache_ib_emit(queue, head,
                                      AC_BARRIER_INV_ICACHE | AC_BARRIER_INV_SMEM | AC_BARRIER_INV_VMEM |
                                         AC_BARRIER_INV_L2 | (graphics ? AC_BARRIER_PFP_SYNC_ME : 0));
   queue->tail_dwords = cache_ib_emit(queue, tail,
                                      (graphics ? AC_BARRIER_SYNC_AND_INV_CB | AC_BARRIER_SYNC_AND_INV_DB
                                                : AC_BARRIER_SYNC_CS) |
                                         AC_BARRIER_WB_L2);
   agc_mem_flush_write_combining();
   agc_report("agc_driver", "queue 0x%x: guest memory is GL2-cached; each submission starts with a %u-dword L2 "
                            "write-back and invalidate and ends with a %u-dword write-back",
              queue->id, queue->head_dwords, queue->tail_dwords);
   return true;
}

static bool
kernel_submit(struct agc_queue *queue, struct agc_step *step)
{
   char text[256];
   struct agc_ib ibs[AGC_SUBMIT_MAX_IBS];
   if (!agc_memory_sync(text, sizeof(text))) {
      agc_driver_fail("guest memory import: %s", text);
      return false;
   }
   const bool cached = agc_memory_guest_cached();
   if (cached && !cache_ibs_ready(queue))
      return false;
   uint32_t stride = ib_stride(queue);
   uint32_t ib_count = 0;
   if (cached) {
      ibs[ib_count++] = (struct agc_ib){
         .va = queue->cache_ibs.va,
         .dwords = queue->head_dwords,
         .flags = AGC_IB_MAIN,
      };
   }
   if (queue->shadowed) {
      ibs[ib_count++] = (struct agc_ib){
         .va = queue->state.preamble_va,
         .dwords = queue->state.preamble_dwords,
         .flags = AGC_IB_MAIN,
      };
   }
   for (uint32_t i = 0; i < step->ib_count; i++) {
      ibs[ib_count++] = (struct agc_ib){
         .va = step->ib_va + (uint64_t)i * stride,
         .dwords = queue->ib_dwords,
         .flags = AGC_IB_MAIN,
      };
   }
   if (cached) {
      ibs[ib_count++] = (struct agc_ib){
         .va = queue->cache_ibs.va + queue->cache_ib_stride,
         .dwords = queue->tail_dwords,
         .flags = AGC_IB_MAIN,
      };
   }
   uint64_t fence = driver.backend->ops->submit(queue->hwq, ibs, ib_count);
   if (!fence) {
      describe_device(text, sizeof(text));
      agc_driver_fail("queue 0x%x: the kernel refused a submission of %u dwords at 0x%llx (%s)", queue->id,
                      step->dwords, (unsigned long long)step->va, text);
      return false;
   }
   step->fence = fence;
   queue->last_fence = fence;
   note_submit(queue, step);
   count(&driver.counters.kernel_submits, 1);
   count(&driver.counters.submitted_dwords, step->dwords);
   return true;
}

static int
submit_main(void *data)
{
   struct agc_queue *queue = data;
   u_thread_setname("agc:submit");
   mtx_lock(&queue->lock);
   for (;;) {
      while (!queue->ready_head && !queue->stop)
         cnd_wait(&queue->submit_wake, &queue->lock);
      if (!queue->ready_head)
         break;
      struct agc_step *step = queue->ready_head;
      queue->ready_head = step->next;
      if (!queue->ready_head)
         queue->ready_tail = NULL;
      step->next = NULL;
      queue->submitting = true;
      queue->active = step;
      mtx_unlock(&queue->lock);

      bool ok = !failed() && !queue->stop;
      if (ok && step->label)
         ok = label_wait(queue, step);
      if (ok && step->wait)
         ok = rendering_wait(queue, step);
      if (ok && step->dwords) {
         ok = kernel_submit(queue, step);
         if (ok && dump_step_wanted(queue, step))
            dump_step(queue, step, "step");
      }
      else if (ok)
         step->fence = queue->last_fence;
      if (ok)
         flips_submitted(step, 1);
      else
         step->event_count = 0;

      mtx_lock(&queue->lock);
      queue->submitting = false;
      queue->active = NULL;
      if (!ok)
         step->fence = 0;
      if (queue->pending_tail)
         queue->pending_tail->next = step;
      else
         queue->pending_head = step;
      queue->pending_tail = step;
      cnd_signal(&queue->complete_wake);
   }
   mtx_unlock(&queue->lock);
   return 0;
}

#define HANG_DUMP_BLOCKS 2048u
#define HANG_DUMP_DEPTH 64u
#define HANG_DUMP_RELEASES 32u
#define HANG_DUMP_UNMET 256u
#define HANG_DUMP_MAGIC UINT64_C(0x31474e4148434741)
#define SUBMIT_HISTORY 32u

struct submit_record {
   uint64_t sequence;
   uint64_t va;
   uint64_t fence;
   uint32_t queue;
   uint32_t dwords;
};

static struct submit_record submit_history[SUBMIT_HISTORY];
static uint32_t submit_history_next;

static void
note_submit(const struct agc_queue *queue, const struct agc_step *step)
{
   uint32_t slot = p_atomic_inc_return(&submit_history_next) % SUBMIT_HISTORY;
   submit_history[slot] = (struct submit_record){
      .sequence = step->sequence,
      .va = step->va,
      .fence = step->fence,
      .queue = queue->id,
      .dwords = step->dwords,
   };
}

struct submit_log_entry {
   uint64_t ns;
   uint64_t va;
   uint64_t sequence;
   uint32_t queue;
   uint32_t dwords;
   uint32_t steps;
   uint32_t held;
   uint32_t followed;
   uint32_t flips;
   int32_t result;
   bool recorded;
};

static struct submit_log_entry submit_log[SUBMIT_LOG_ENTRIES];
static uint32_t submit_log_next;

static void
submit_log_note(uint32_t queue, const uint32_t *address, uint32_t dwords, bool recorded, const struct agc_step *head,
                uint32_t followed, uint32_t flips, int32_t result)
{
   struct submit_log_entry entry = {
      .ns = os_time_get_nano(),
      .va = (uintptr_t)address,
      .queue = queue,
      .dwords = dwords,
      .followed = followed,
      .flips = flips,
      .result = result,
      .recorded = recorded,
   };
   for (const struct agc_step *each = head; each; each = each->next) {
      if (!entry.steps)
         entry.sequence = each->sequence;
      entry.steps++;
      entry.held += each->label || each->wait;
   }
   submit_log[p_atomic_inc_return(&submit_log_next) % SUBMIT_LOG_ENTRIES] = entry;
}

static void
submit_log_report(void)
{
   uint32_t last = p_atomic_read(&submit_log_next);
   uint64_t now = os_time_get_nano();
   for (uint32_t i = 0; i < SUBMIT_LOG_REPORTED && i < last; i++) {
      const struct submit_log_entry *entry = &submit_log[(last - i) % SUBMIT_LOG_ENTRIES];
      agc_stat("agc_driver",
               "  title submit %.1f ms ago: queue 0x%x guest 0x%llx, %u dwords%s, %u steps from %llu (%u held on the "
               "CPU), %u buffers followed, %u flips, result %d",
               (double)(now - entry->ns) / 1e6, entry->queue, (unsigned long long)entry->va, entry->dwords,
               entry->recorded ? "" : " (not a recorded segment)", entry->steps, (unsigned long long)entry->sequence,
               entry->held, entry->followed, entry->flips, entry->result);
   }
}

static void
queue_states_report(void)
{
   for (uint32_t i = 0; i < QUEUE_SLOTS; i++) {
      struct agc_queue *queue = p_atomic_read(&driver.queues[i]);
      if (!queue)
         continue;
      if (mtx_trylock(&queue->lock) != thrd_success) {
         agc_stat("agc_driver", "  queue 0x%x: its lock is held", queue->id);
         continue;
      }
      uint32_t ready = 0;
      uint32_t gpu = 0;
      for (const struct agc_step *each = queue->ready_head; each; each = each->next)
         ready++;
      for (const struct agc_step *each = queue->pending_head; each; each = each->next)
         gpu++;
      char active[320] = "idle";
      const struct agc_step *step = queue->active;
      if (step && step->label) {
         snprintf(active, sizeof(active),
                  "step %llu (0x%llx, %u dwords) waits on the CPU for label 0x%llx, now 0x%llx, function %u, "
                  "reference 0x%llx",
                  (unsigned long long)step->sequence, (unsigned long long)step->va, step->dwords,
                  (unsigned long long)step->label_va, (unsigned long long)label_read(step), step->label_function,
                  (unsigned long long)step->label_reference);
      } else if (step && step->wait) {
         snprintf(active, sizeof(active), "step %llu waits until buffer %u of video output %u is safe for rendering",
                  (unsigned long long)step->sequence, step->wait_index, step->wait_handle);
      } else if (step) {
         snprintf(active, sizeof(active), "step %llu (0x%llx, %u dwords) is being submitted",
                  (unsigned long long)step->sequence, (unsigned long long)step->va, step->dwords);
      }
      char oldest[200] = "";
      if (queue->pending_head) {
         const struct agc_step *head = queue->pending_head;
         snprintf(oldest, sizeof(oldest), "; oldest on the GPU: step %llu, fence %llu, 0x%llx, %u dwords",
                  (unsigned long long)head->sequence, (unsigned long long)head->fence,
                  (unsigned long long)head->va, head->dwords);
      }
      agc_stat("agc_driver",
               "  queue 0x%x: %llu steps enqueued, %llu completed, last fence %llu, %u ready, %u on the GPU; submit "
               "thread: %s%s",
               queue->id, (unsigned long long)queue->enqueued, (unsigned long long)queue->completed,
               (unsigned long long)queue->last_fence, ready, gpu, active, oldest);
      mtx_unlock(&queue->lock);
   }
}

struct hang_dump {
   FILE *file;
   uint64_t va[HANG_DUMP_BLOCKS];
   uint32_t dwords[HANG_DUMP_BLOCKS];
   uint32_t count;
   uint32_t missing;
   uint32_t releases;
   uint32_t waits_met;
   uint32_t waits_unmet;
   uint32_t waits_other;
};

static const volatile uint32_t *
hang_label_cpu(uint64_t label)
{
   if (!label)
      return NULL;
   if (agc_va_in_guest_window(label))
      return (const volatile uint32_t *)(uintptr_t)label;
   return agc_memory_cpu_of(label, 8);
}

static void
hang_dump_wait(struct hang_dump *dump, uint64_t va, uint32_t at, const uint32_t *packet, bool wide)
{
   uint64_t label = (packet[2] & ~UINT32_C(3)) | (uint64_t)(packet[3] & 0xffff) << 32;
   const volatile uint32_t *value = (packet[1] & 0x30) != 0 ? hang_label_cpu(label) : NULL;
   if (!value) {
      dump->waits_other++;
      return;
   }
   uint64_t reference = wide ? packet[4] | (uint64_t)packet[5] << 32 : packet[4];
   uint64_t mask = wide ? packet[6] | (uint64_t)packet[7] << 32 : packet[5];
   uint64_t now = value[0] | (wide ? (uint64_t)value[1] << 32 : 0);
   uint32_t function = packet[1] & 7;
   if (condition_met(function, reference, mask, now)) {
      dump->waits_met++;
      return;
   }
   if (++dump->waits_unmet > HANG_DUMP_UNMET)
      return;
   agc_stat("agc_driver", "  unmet wait at 0x%llx+%u: label 0x%llx is 0x%llx, function %u, reference 0x%llx, mask 0x%llx, %s",
            (unsigned long long)va, at, (unsigned long long)label, (unsigned long long)now, function,
            (unsigned long long)reference, (unsigned long long)mask, (packet[1] >> 8) & 1 ? "PFP" : "ME");
}

static void
hang_dump_block(struct hang_dump *dump, uint64_t va, uint32_t dwords, uint32_t depth)
{
   if (!dwords || depth > HANG_DUMP_DEPTH || dump->count >= HANG_DUMP_BLOCKS)
      return;
   for (uint32_t i = 0; i < dump->count; i++) {
      if (dump->va[i] == va && dump->dwords[i] >= dwords)
         return;
   }
   const uint32_t *cpu = agc_memory_cpu_of(va, (uint64_t)dwords * 4);
   if (!cpu) {
      dump->missing++;
      return;
   }
   dump->va[dump->count] = va;
   dump->dwords[dump->count] = dwords;
   dump->count++;
   uint32_t pad = 0;
   fwrite(&va, sizeof(va), 1, dump->file);
   fwrite(&dwords, sizeof(dwords), 1, dump->file);
   fwrite(&pad, sizeof(pad), 1, dump->file);
   fwrite(cpu, sizeof(uint32_t), dwords, dump->file);
   for (uint32_t at = 0; at < dwords;) {
      uint32_t header = cpu[at];
      if (header == PKT3_NOP_PAD || PKT_TYPE_G(header) == 2) {
         at++;
         continue;
      }
      if (PKT_TYPE_G(header) != 3)
         break;
      uint32_t length = PKT_COUNT_G(header) + 2;
      uint32_t opcode = PKT3_IT_OPCODE_G(header);
      if ((opcode == PKT3_WAIT_REG_MEM && length >= 7) || (opcode == PKT3_WAIT_REG_MEM64 && length >= 9)) {
         if (at + length <= dwords)
            hang_dump_wait(dump, va, at, cpu + at, opcode == PKT3_WAIT_REG_MEM64);
      } else if (opcode == PKT3_RELEASE_MEM && at + length <= dwords && dump->releases < HANG_DUMP_RELEASES &&
                 G_492_DATA_SEL(cpu[at + 2]) != 0) {
         uint64_t label = cpu[at + 3] | (uint64_t)(cpu[at + 4] & 0xffff) << 32;
         const volatile uint32_t *value = hang_label_cpu(label);
         if (value) {
            dump->releases++;
            agc_stat("agc_driver", "  release at 0x%llx+%u: label 0x%llx now 0x%08x%08x, data 0x%08x%08x",
                     (unsigned long long)va, at, (unsigned long long)label, value[1], value[0],
                     length > 6 ? cpu[at + 6] : 0, cpu[at + 5]);
         }
      }
      if (PKT3_IT_OPCODE_G(header) == PKT3_INDIRECT_BUFFER && length == 4 && at + 4 <= dwords) {
         uint64_t target = cpu[at + 1] | (uint64_t)(cpu[at + 2] & 0xffff) << 32;
         hang_dump_block(dump, target, G_3F3_IB_SIZE(cpu[at + 3]), depth + ((cpu[at + 3] & S_3F3_CHAIN(1)) ? 0 : 1));
      }
      at += length;
   }
}

static void state_report(bool force);

static void
dump_step(struct agc_queue *queue, const struct agc_step *step, const char *tag)
{
   const char *dir = getenv("AGC_HANG_DUMP");
   if (!dir || !*dir)
      return;
   char path[600];
   snprintf(path, sizeof(path), "%s\\agc-%s-q%02x-s%llu.bin", dir, tag, queue->id,
            (unsigned long long)step->sequence);
   struct hang_dump *dump = calloc(1, sizeof(*dump));
   if (!dump)
      return;
   dump->file = fopen(path, "wb");
   if (!dump->file) {
      agc_report("agc_driver", "cannot write %s", path);
      free(dump);
      return;
   }
   uint64_t magic = HANG_DUMP_MAGIC;
   uint32_t head[6] = {
      (uint32_t)driver.info.gfx_level, (uint32_t)driver.info.family, queue->id == AGC_QUEUE_GRAPHICS ? 0u : 1u,
      step->ib_count + (queue->shadowed ? 1u : 0u), 0, 0,
   };
   fwrite(&magic, sizeof(magic), 1, dump->file);
   fwrite(head, sizeof(head), 1, dump->file);
   uint32_t stride = ib_stride(queue);
   uint64_t roots[AGC_SUBMIT_MAX_IBS + 1];
   uint32_t root_dwords[AGC_SUBMIT_MAX_IBS + 1];
   uint32_t root_count = 0;
   if (queue->shadowed) {
      roots[root_count] = queue->state.preamble_va;
      root_dwords[root_count++] = queue->state.preamble_dwords;
   }
   for (uint32_t i = 0; i < step->ib_count; i++) {
      roots[root_count] = step->ib_va + (uint64_t)i * stride;
      root_dwords[root_count++] = queue->ib_dwords;
   }
   for (uint32_t i = 0; i < root_count; i++) {
      fwrite(&roots[i], sizeof(roots[i]), 1, dump->file);
      fwrite(&root_dwords[i], sizeof(root_dwords[i]), 1, dump->file);
   }
   for (uint32_t i = 0; i < root_count; i++)
      hang_dump_block(dump, roots[i], root_dwords[i], 0);
   fclose(dump->file);
   agc_stat("agc_driver", "dumped %u command blocks (%u not readable) to %s; memory waits: %u met, %u unmet, %u not readable",
            dump->count, dump->missing, path, dump->waits_met, dump->waits_unmet, dump->waits_other);
   free(dump);
}

static bool
dump_step_wanted(const struct agc_queue *queue, const struct agc_step *step)
{
   static int parsed;
   static uint64_t first;
   static uint64_t last;
   if (!parsed) {
      const char *spec = getenv("AGC_DUMP_STEPS");
      if (spec && *spec) {
         char *end = NULL;
         first = strtoull(spec, &end, 10);
         last = end && *end == ':' ? strtoull(end + 1, NULL, 10) : first;
      }
      parsed = 1;
   }
   return queue->id == AGC_QUEUE_GRAPHICS && first && step->sequence >= first && step->sequence <= last;
}

static void follow_report(void);

static void
hang_report(struct agc_queue *queue, const struct agc_step *step, const char *device)
{
   if (queue->id == AGC_QUEUE_GRAPHICS)
      state_report(true);
   follow_report();
   agc_stat("agc_driver", "queue 0x%x: step %llu (guest 0x%llx, %u dwords, fence %llu) has not finished after 2 "
              "seconds; device %s",
              queue->id, (unsigned long long)step->sequence, (unsigned long long)step->va, step->dwords,
              (unsigned long long)step->fence, device);
   uint32_t last = p_atomic_read(&submit_history_next);
   for (uint32_t i = 0; i < SUBMIT_HISTORY && i < last; i++) {
      const struct submit_record *r = &submit_history[(last - i) % SUBMIT_HISTORY];
      agc_stat("agc_driver", "  recent submit: queue 0x%x step %llu guest 0x%llx %u dwords fence %llu", r->queue,
                 (unsigned long long)r->sequence, (unsigned long long)r->va, r->dwords,
                 (unsigned long long)r->fence);
   }
   queue_states_report();
   submit_log_report();
   dump_step(queue, step, "hang");
}

static bool
wait_fence(struct agc_queue *queue, const struct agc_step *step)
{
   const struct agc_driver_ops *ops = driver.backend->ops;
   unsigned polls = 0;
   uint64_t fence = step->fence;
   if (!fence)
      return true;
   while (!ops->fence_wait(queue->hwq, fence, FENCE_POLL_NS)) {
      char text[256];
      enum agc_device_state state = describe_device(text, sizeof(text));
      if (++polls == 1)
         hang_report(queue, step, text);
      if (state != AGC_DEVICE_ACTIVE) {
         agc_driver_fail("queue 0x%x: fence %llu did not complete (%s)", queue->id, (unsigned long long)fence,
                         text);
         return false;
      }
      if (failed())
         return false;
      if (polls == 5) {
         agc_stat("agc_driver", "queue 0x%x: fence %llu is still pending after %u seconds", queue->id,
                    (unsigned long long)fence, polls * 2);
      }
   }
   return true;
}

#define STATE_REPORT_INTERVAL 120u

struct state_register {
   const char *name;
   uint32_t reg;
};

static const struct state_register state_registers[] = {
   {"CB_COLOR0_BASE", R_028C60_CB_COLOR0_BASE},
   {"CB_COLOR0_BASE_EXT", R_028E40_CB_COLOR0_BASE_EXT},
   {"CB_COLOR0_VIEW", R_028C6C_CB_COLOR0_VIEW},
   {"CB_COLOR0_INFO", R_028C70_CB_COLOR0_INFO},
   {"CB_COLOR0_ATTRIB", R_028C74_CB_COLOR0_ATTRIB},
   {"CB_COLOR0_ATTRIB2", R_028EC0_CB_COLOR0_ATTRIB2},
   {"CB_COLOR0_ATTRIB3", R_028EE0_CB_COLOR0_ATTRIB3},
   {"CB_COLOR1_BASE", R_028C60_CB_COLOR0_BASE + 0x3c},
   {"CB_COLOR1_INFO", R_028C70_CB_COLOR0_INFO + 0x3c},
   {"CB_TARGET_MASK", R_028238_CB_TARGET_MASK},
   {"CB_SHADER_MASK", R_02823C_CB_SHADER_MASK},
   {"CB_COLOR_CONTROL", R_028808_CB_COLOR_CONTROL},
   {"SPI_SHADER_COL_FORMAT", R_028714_SPI_SHADER_COL_FORMAT},
   {"SPI_PS_INPUT_ENA", R_0286CC_SPI_PS_INPUT_ENA},
   {"DB_SHADER_CONTROL", R_02880C_DB_SHADER_CONTROL},
   {"DB_RENDER_CONTROL", R_028000_DB_RENDER_CONTROL},
   {"DB_DEPTH_CONTROL", R_028800_DB_DEPTH_CONTROL},
   {"PA_SC_WINDOW_SCISSOR_TL", R_028204_PA_SC_WINDOW_SCISSOR_TL},
   {"PA_SC_WINDOW_SCISSOR_BR", R_028208_PA_SC_WINDOW_SCISSOR_BR},
   {"PA_SC_VPORT_SCISSOR_0_TL", R_028250_PA_SC_VPORT_SCISSOR_0_TL},
   {"PA_SC_VPORT_SCISSOR_0_BR", R_028254_PA_SC_VPORT_SCISSOR_0_BR},
   {"PA_CL_VPORT_XSCALE", R_02843C_PA_CL_VPORT_XSCALE},
   {"PA_CL_VPORT_YSCALE", R_028444_PA_CL_VPORT_YSCALE},
   {"PA_CL_VTE_CNTL", R_028818_PA_CL_VTE_CNTL},
   {"PA_SU_SC_MODE_CNTL", R_028814_PA_SU_SC_MODE_CNTL},
   {"VGT_SHADER_STAGES_EN", R_028B54_VGT_SHADER_STAGES_EN},
   {"SPI_SHADER_PGM_LO_PS", R_00B020_SPI_SHADER_PGM_LO_PS},
   {"SPI_SHADER_PGM_HI_PS", R_00B024_SPI_SHADER_PGM_HI_PS},
   {"SPI_SHADER_PGM_LO_GS", R_00B220_SPI_SHADER_PGM_LO_GS},
   {"GE_CNTL", R_03096C_GE_CNTL},
   {"VGT_PRIMITIVE_TYPE", R_030908_VGT_PRIMITIVE_TYPE},
};

static void
probe_ranges(void)
{
   const char *spec = getenv("AGC_PROBE_RANGES");
   while (spec && *spec) {
      char *end = NULL;
      uint64_t va = strtoull(spec, &end, 16);
      if (!end || *end != ':')
         return;
      uint64_t bytes = strtoull(end + 1, &end, 16);
      if (va && bytes && agc_memory_imported(va, bytes)) {
         const volatile uint64_t *words = (const volatile uint64_t *)(uintptr_t)va;
         uint64_t count = bytes / 8;
         uint64_t nonzero = 0;
         char first[256];
         size_t used = 0;
         first[0] = 0;
         for (uint64_t i = 0; i < count; i++) {
            uint64_t value = words[i];
            if (!value)
               continue;
            nonzero++;
            if (used < sizeof(first) - 40) {
               int n = snprintf(first + used, sizeof(first) - used, " +0x%llx=0x%llx", (unsigned long long)(i * 8),
                                (unsigned long long)value);
               if (n > 0)
                  used += (size_t)n;
            }
         }
         agc_report("agc_probe", "0x%llx+0x%llx: %llu of %llu qwords nonzero;%s", (unsigned long long)va,
                    (unsigned long long)bytes, (unsigned long long)nonzero, (unsigned long long)count, first);
      } else if (va && bytes) {
         agc_report("agc_probe", "0x%llx+0x%llx is not imported", (unsigned long long)va, (unsigned long long)bytes);
      }
      if (!end || *end != ',')
         return;
      spec = end + 1;
   }
}

static void census_report(void);

static void
images_report(void)
{
   struct agc_images_stats stats;
   agc_images_stats(&stats);
   agc_stat("agc_images", "%" PRIu64 " reports: %" PRIu64 " converted (%" PRIu64 " levels, %.1f MiB, %" PRIu64
            " batches), %" PRIu64 " waiting for levels, %" PRIu64 " levels released, %" PRIu64 " read in place, %" PRIu64
            " GPU-written (%" PRIu64 " known as targets or stored to; %" PRIu64 " store reports), %" PRIu64
            " refused; %" PRIu64 " uncompressed images converted as the CPU's (%" PRIu64
            " in memory that cannot be write-watched), %" PRIu64 " more after a later CPU write, %" PRIu64
            " left alone because the GPU wrote them meanwhile, %" PRIu64 " levels converted again; %" PRIu64
            " CPU-write checks (%" PRIu64 " found writes, %" PRIu64 " could not be made)",
            stats.reports, stats.converted, stats.levels, (double)stats.bytes / (1024.0 * 1024.0), stats.batches,
            stats.waiting, stats.released, stats.direct, stats.rendered, stats.targets, stats.stores, stats.refused,
            stats.cpu_converted, stats.unwatched, stats.turned, stats.kept, stats.refreshed, stats.checks,
            stats.wrote, stats.blind);
}

static void
state_report(bool force)
{
   static int enabled = -1;
   static uint32_t flips;
   if (enabled < 0) {
      const char *value = getenv("AGC_PRESENT_STATS");
      enabled = value && *value && *value != '0';
   }
   if (!force && (!enabled || p_atomic_inc_return(&flips) % STATE_REPORT_INTERVAL != 1))
      return;
   char line[2048];
   size_t used = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(state_registers); i++) {
      uint32_t reg = state_registers[i].reg;
      unsigned bank;
      uint32_t offset;
      if (reg >= CIK_UCONFIG_REG_OFFSET) {
         bank = 2;
         offset = (reg - CIK_UCONFIG_REG_OFFSET) / 4;
      } else if (reg >= SI_CONTEXT_REG_OFFSET) {
         bank = 0;
         offset = (reg - SI_CONTEXT_REG_OFFSET) / 4;
      } else {
         bank = 1;
         offset = (reg - SI_SH_REG_OFFSET) / 4;
      }
      uint32_t value = 0;
      if (!agc_driver_read_register(bank, offset, &value))
         return;
      int n = snprintf(line + used, sizeof(line) - used, "%s%s=0x%08x", i ? " " : "", state_registers[i].name, value);
      if (n < 0 || (size_t)n >= sizeof(line) - used)
         break;
      used += (size_t)n;
   }
   agc_report("agc_state", "%s", line);
   probe_ranges();
   census_report();
   images_report();
}

static void
deliver(struct agc_queue *queue, struct agc_step *step)
{
   struct agc_driver_error error = {0};
   for (uint32_t i = 0; i < step->event_count; i++) {
      if (step->events[i].kind == AGC_EVENT_FLIP) {
         agc_images_frame(driver.backend, &driver.info);
         state_report(false);
      }
   }
   for (uint32_t i = 0; i < step->event_count; i++) {
      if (step->events[i].kind == AGC_EVENT_FLIP)
         count(&driver.counters.flips, 1);
      else
         count(&driver.counters.interrupts, 1);
   }
   int result = agc_flips_deliver(step->events, step->event_count, &error);
   flips_submitted(step, -1);
   if (result != 0 && result != AGC_DRIVER_SHUTDOWN && !failed())
      agc_driver_fail("queue 0x%x: delivering events: %s", queue->id, error.message);
}

static int
complete_main(void *data)
{
   struct agc_queue *queue = data;
   u_thread_setname("agc:complete");
   mtx_lock(&queue->lock);
   for (;;) {
      while (!queue->pending_head && !(queue->stop && !queue->ready_head && !queue->submitting))
         cnd_wait(&queue->complete_wake, &queue->lock);
      if (!queue->pending_head)
         break;
      struct agc_step *step = queue->pending_head;
      mtx_unlock(&queue->lock);

      bool ok = wait_fence(queue, step);
      if (ok)
         label_log_note(step);
      if (step->event_count) {
         if (ok)
            deliver(queue, step);
         else
            flips_submitted(step, -1);
      }

      mtx_lock(&queue->lock);
      queue->pending_head = step->next;
      if (!queue->pending_head)
         queue->pending_tail = NULL;
      queue->completed = step->sequence;
      cnd_broadcast(&queue->idle_wake);
      step_free(step);
   }
   mtx_unlock(&queue->lock);
   return 0;
}

static struct agc_queue *
queue_new(uint32_t id, const char *function)
{
   struct agc_queue *queue = calloc(1, sizeof(*queue));
   if (!queue)
      return NULL;
   enum agc_queue_kind kind = id == AGC_QUEUE_GRAPHICS ? AGC_QUEUE_GFX : AGC_QUEUE_COMPUTE;
   const struct amd_ip_info *ip = &driver.info.ip[id == AGC_QUEUE_GRAPHICS ? AMD_IP_GFX : AMD_IP_COMPUTE];
   uint32_t multiple = ip->ib_pad_dw_mask + 1;
   queue->id = id;
   queue->ib_start_bytes = ip->ib_alignment ? ip->ib_alignment : 256;
   queue->ib_dwords = (2 * CHAIN_DWORDS + PROLOGUE_DWORDS + multiple - 1) / multiple * multiple;
   queue->hwq = driver.backend->ops->queue_create(driver.backend, kind);
   if (!queue->hwq) {
      agc_report(function, "the kernel refused to create queue 0x%x", id);
      free(queue);
      return NULL;
   }
   return queue;
}

static void
queue_run(struct agc_queue *queue)
{
   mtx_init(&queue->lock, mtx_plain);
   cnd_init(&queue->submit_wake);
   cnd_init(&queue->complete_wake);
   cnd_init(&queue->idle_wake);
   thrd_create(&queue->submit_thread, submit_main, queue);
   thrd_create(&queue->complete_thread, complete_main, queue);
   p_atomic_set(&driver.queues[queue->id], queue);
}

static struct agc_queue *
queue_get(uint32_t id, const char *function)
{
   if (id != AGC_QUEUE_GRAPHICS && (id < AGC_QUEUE_COMPUTE_FIRST || id >= AGC_QUEUE_COMPUTE_END)) {
      agc_report(function, "unsupported queue 0x%x", id);
      return NULL;
   }
   struct agc_queue *queue = p_atomic_read(&driver.queues[id]);
   if (AGC_LIKELY(queue))
      return queue;

   mtx_lock(&driver.lock);
   queue = driver.queues[id];
   if (!queue && !driver.stopping && id != AGC_QUEUE_GRAPHICS) {
      queue = queue_new(id, function);
      if (queue)
         queue_run(queue);
   }
   mtx_unlock(&driver.lock);
   return queue;
}

static uint32_t
first_reloc(const struct agc_record *record, uint32_t offset)
{
   uint32_t low = 0;
   uint32_t high = record->reloc_count;
   while (low < high) {
      uint32_t mid = low + (high - low) / 2;
      if (record->relocs[mid].offset < offset)
         low = mid + 1;
      else
         high = mid;
   }
   return low;
}

#define CENSUS_TARGETS 32u
#define CX_REG(reg) (((reg) - SI_CONTEXT_REG_OFFSET) / 4)

struct census_target {
   uint64_t base;
   uint32_t info;
   uint32_t attrib2;
   uint32_t uses;
};

struct census_depth {
   uint64_t base;
   uint32_t info;
   uint32_t clear;
   uint32_t render;
   uint32_t size_xy;
   uint32_t uses;
};

static struct {
   simple_mtx_t lock;
   bool enabled;
   bool checked;
   struct census_target targets[CENSUS_TARGETS];
   uint32_t count;
   struct census_depth depths[CENSUS_TARGETS];
   uint32_t depth_count;
} census = {.lock = SIMPLE_MTX_INITIALIZER};

static bool
census_find(const struct agc_register *host, uint32_t count, uint32_t offset, uint32_t *value)
{
   for (uint32_t j = 0; j < count; j++) {
      if (host[j].offset == offset) {
         *value = host[j].value;
         return true;
      }
   }
   return false;
}

static void
census_note_depth(const struct agc_register *host, uint32_t count)
{
   uint32_t lo = 0;
   if (!census_find(host, count, CX_REG(R_028048_DB_Z_READ_BASE), &lo) || !lo)
      return;
   uint32_t hi = 0;
   uint32_t info = 0;
   uint32_t clear = 0;
   uint32_t render = 0;
   uint32_t size_xy = 0;
   census_find(host, count, CX_REG(R_028068_DB_Z_READ_BASE_HI), &hi);
   census_find(host, count, CX_REG(R_028040_DB_Z_INFO), &info);
   census_find(host, count, CX_REG(R_02802C_DB_DEPTH_CLEAR), &clear);
   census_find(host, count, CX_REG(R_028000_DB_RENDER_CONTROL), &render);
   census_find(host, count, CX_REG(R_02801C_DB_DEPTH_SIZE_XY), &size_xy);
   uint64_t base = lo | (uint64_t)hi << 32;
   simple_mtx_lock(&census.lock);
   uint32_t k = 0;
   while (k < census.depth_count && census.depths[k].base != base)
      k++;
   if (k == census.depth_count && census.depth_count < CENSUS_TARGETS)
      census.depths[census.depth_count++] = (struct census_depth){.base = base};
   if (k < census.depth_count) {
      struct census_depth *d = &census.depths[k];
      d->uses++;
      if (info)
         d->info = info;
      if (clear)
         d->clear = clear;
      if (render)
         d->render = render;
      if (size_xy)
         d->size_xy = size_xy;
   }
   simple_mtx_unlock(&census.lock);
}

static bool
census_enabled(void)
{
   if (!census.checked) {
      const char *value = getenv("AGC_PRESENT_STATS");
      census.enabled = value && *value && *value != '0';
      census.checked = true;
   }
   return census.enabled;
}

static void
census_guest_depth(const struct agc_register *guest, uint32_t count)
{
   static uint32_t reported;
   uint32_t z_info = 0;
   if (reported >= 6 || !census_find(guest, count, CX_REG(R_028040_DB_Z_INFO), &z_info))
      return;
   uint32_t stencil_info = 0;
   uint32_t htile_surface = 0;
   uint32_t htile_base = 0;
   uint32_t htile_hi = 0;
   uint32_t render = 0;
   uint32_t z_base = 0;
   census_find(guest, count, CX_REG(R_028044_DB_STENCIL_INFO), &stencil_info);
   census_find(guest, count, CX_REG(R_028ABC_DB_HTILE_SURFACE), &htile_surface);
   census_find(guest, count, CX_REG(R_028014_DB_HTILE_DATA_BASE), &htile_base);
   census_find(guest, count, CX_REG(R_028078_DB_HTILE_DATA_BASE_HI), &htile_hi);
   census_find(guest, count, CX_REG(R_028000_DB_RENDER_CONTROL), &render);
   census_find(guest, count, CX_REG(R_028048_DB_Z_READ_BASE), &z_base);
   reported++;
   agc_report("agc_census", "guest depth: DB_Z_INFO 0x%08x DB_STENCIL_INFO 0x%08x DB_HTILE_SURFACE 0x%08x HTILE base 0x%02x%08x "
              "DB_RENDER_CONTROL 0x%08x DB_Z_READ_BASE 0x%08x (%u registers in the array)",
              z_info, stencil_info, htile_surface, htile_hi, htile_base, render, z_base, count);
}

static void
census_guest_color(const struct agc_register *guest, uint32_t count)
{
   static uint64_t seen[16];
   static uint32_t seen_count;
   for (uint32_t rt = 0; rt < 8; rt++) {
      uint32_t base_lo = 0;
      if (!census_find(guest, count, CX_REG(R_028C60_CB_COLOR0_BASE) + rt * 15, &base_lo) || !base_lo)
         continue;
      uint32_t base_hi = 0;
      uint32_t info = 0;
      uint32_t dcc_control = 0;
      uint32_t dcc_lo = 0;
      uint32_t dcc_hi = 0;
      uint32_t attrib3 = 0;
      uint32_t clear0 = 0;
      uint32_t clear1 = 0;
      uint32_t cmask = 0;
      census_find(guest, count, CX_REG(R_028E40_CB_COLOR0_BASE_EXT) + rt, &base_hi);
      census_find(guest, count, CX_REG(R_028C70_CB_COLOR0_INFO) + rt * 15, &info);
      census_find(guest, count, CX_REG(R_028C78_CB_COLOR0_DCC_CONTROL) + rt * 15, &dcc_control);
      census_find(guest, count, CX_REG(R_028C94_CB_COLOR0_DCC_BASE) + rt * 15, &dcc_lo);
      census_find(guest, count, CX_REG(R_028EA0_CB_COLOR0_DCC_BASE_EXT) + rt, &dcc_hi);
      census_find(guest, count, CX_REG(R_028EE0_CB_COLOR0_ATTRIB3) + rt, &attrib3);
      census_find(guest, count, CX_REG(R_028C8C_CB_COLOR0_CLEAR_WORD0) + rt * 15, &clear0);
      census_find(guest, count, CX_REG(R_028C90_CB_COLOR0_CLEAR_WORD1) + rt * 15, &clear1);
      census_find(guest, count, CX_REG(R_028C7C_CB_COLOR0_CMASK) + rt * 15, &cmask);
      const uint64_t base = ((uint64_t)base_lo | (uint64_t)(base_hi & 0xff) << 32) << 8;
      bool known = false;
      for (uint32_t k = 0; k < seen_count; k++)
         known |= seen[k] == base;
      if (known || seen_count == ARRAY_SIZE(seen))
         continue;
      seen[seen_count++] = base;
      const uint64_t dcc = ((uint64_t)dcc_lo | (uint64_t)(dcc_hi & 0xff) << 32) << 8;
      uint32_t first_dcc = 0;
      if (dcc && agc_memory_imported(dcc & ~UINT64_C(0xff), 4))
         first_dcc = *(const volatile uint32_t *)(uintptr_t)(dcc & ~UINT64_C(0xff));
      agc_stat("agc_census",
               "guest color target %u at 0x%llx: INFO 0x%08x (DCC_ENABLE %u, FAST_CLEAR %u, COMPRESSION %u) DCC_CONTROL "
               "0x%08x DCC base 0x%llx (first dword 0x%08x) ATTRIB3 0x%08x CLEAR 0x%08x %08x CMASK 0x%08x",
               rt, (unsigned long long)base, info, (info >> 28) & 1, (info >> 13) & 1, (info >> 14) & 1, dcc_control,
               (unsigned long long)dcc, first_dcc, attrib3, clear1, clear0, cmask);
   }
}

static void
census_note(const struct agc_register *host, uint32_t count)
{
   census_note_depth(host, count);
   for (uint32_t i = 0; i < count; i++) {
      for (uint32_t rt = 0; rt < 8; rt++) {
         if (host[i].offset != CX_REG(R_028C60_CB_COLOR0_BASE) + rt * 15)
            continue;
         uint64_t base = host[i].value;
         uint32_t info = 0;
         uint32_t attrib2 = 0;
         for (uint32_t j = 0; j < count; j++) {
            if (host[j].offset == CX_REG(R_028E40_CB_COLOR0_BASE_EXT) + rt)
               base |= (uint64_t)host[j].value << 32;
            else if (host[j].offset == CX_REG(R_028C70_CB_COLOR0_INFO) + rt * 15)
               info = host[j].value;
            else if (host[j].offset == CX_REG(R_028EC0_CB_COLOR0_ATTRIB2) + rt)
               attrib2 = host[j].value;
         }
         if (!base)
            continue;
         simple_mtx_lock(&census.lock);
         uint32_t k = 0;
         while (k < census.count && census.targets[k].base != base)
            k++;
         if (k == census.count && census.count < CENSUS_TARGETS)
            census.targets[census.count++] = (struct census_target){.base = base, .info = info, .attrib2 = attrib2};
         if (k < census.count) {
            census.targets[k].uses++;
            if (info)
               census.targets[k].info = info;
            if (attrib2)
               census.targets[k].attrib2 = attrib2;
         }
         simple_mtx_unlock(&census.lock);
      }
   }
}

static void
census_report(void)
{
   simple_mtx_lock(&census.lock);
   for (uint32_t k = 0; k < census.count; k++) {
      const struct census_target *t = &census.targets[k];
      uint64_t va = t->base << 8;
      uint64_t guest = agc_va_in_mirror_window(va) ? agc_guest_va(va) : va;
      uint32_t width = ((t->attrib2 >> 14) & 0x3fff) + 1;
      uint32_t height = (t->attrib2 & 0x3fff) + 1;
      uint64_t bytes = (uint64_t)width * height * 4;
      uint32_t nonzero = 0;
      uint32_t samples = 0;
      uint32_t first = 0;
      if (t->attrib2 && bytes && agc_memory_imported(guest, bytes)) {
         const volatile uint32_t *words = (const volatile uint32_t *)(uintptr_t)guest;
         uint64_t count = bytes / 4;
         for (uint32_t i = 0; i < 2048; i++, samples++) {
            uint32_t word = words[(count / 2048) * i];
            if (word) {
               nonzero++;
               if (!first)
                  first = word;
            }
         }
      }
      agc_report("agc_census", "target 0x%llx (guest 0x%llx) %ux%u info 0x%08x uses %u: %u of %u sampled dwords nonzero, first "
                 "0x%08x",
                 (unsigned long long)va, (unsigned long long)guest, width, height, t->info, t->uses, nonzero, samples,
                 first);
   }
   for (uint32_t k = 0; k < census.depth_count; k++) {
      const struct census_depth *d = &census.depths[k];
      uint64_t va = d->base << 8;
      uint64_t guest = agc_va_in_mirror_window(va) ? agc_guest_va(va) : va;
      uint32_t width = (d->size_xy & 0x3fff) + 1;
      uint32_t height = ((d->size_xy >> 16) & 0x3fff) + 1;
      uint64_t bytes = (uint64_t)width * height * 4;
      uint32_t nonzero = 0;
      uint32_t ones = 0;
      uint32_t first = 0;
      if (d->size_xy && agc_memory_imported(guest, bytes)) {
         const volatile uint32_t *words = (const volatile uint32_t *)(uintptr_t)guest;
         uint64_t count = bytes / 4;
         for (uint32_t i = 0; i < 2048; i++) {
            uint32_t word = words[(count / 2048) * i];
            if (word)
               nonzero++;
            if (word == 0x3f800000)
               ones++;
            if (word && !first)
               first = word;
         }
      }
      agc_report("agc_census", "depth 0x%llx (guest 0x%llx) %ux%u z_info 0x%08x clear 0x%08x render 0x%08x uses %u: %u of "
                 "2048 sampled dwords nonzero, %u are 1.0, first 0x%08x",
                 (unsigned long long)va, (unsigned long long)guest, width, height, d->info, d->clear, d->render, d->uses,
                 nonzero, ones, first);
   }
   census.count = 0;
   census.depth_count = 0;
   simple_mtx_unlock(&census.lock);
}

static struct agc_reg_shadow *
queue_context(struct agc_queue *queue)
{
   if (!queue->context_known) {
      agc_reg_shadow_reset(&queue->context);
      queue->context_saved = queue->context;
      queue->context_known = true;
   }
   return &queue->context;
}

static void
context_op(struct agc_queue *queue, uint32_t operation)
{
   struct agc_reg_shadow *context = queue_context(queue);
   if (operation == 1 || operation == 3)
      queue->context_saved = *context;
   if (operation == 0 || operation == 3)
      agc_reg_shadow_reset(context);
   if (operation == 2)
      *context = queue->context_saved;
}

static bool
resolve_register_array(struct agc_queue *queue, struct agc_record *record, struct agc_reloc *reloc,
                       uint64_t sequence, uint32_t depth, const char *function)
{
   const struct agc_register *guest = agc_reloc_registers(reloc);
   uint32_t guest_count = reloc->arg32;
   uint32_t bound = agc_reg_convert_pairs_bound(guest_count);
   uint64_t va = 0;
   uint32_t host_count = 0;
   struct agc_register uc[AGC_REGISTERS_UC_MAX];
   uint32_t uc_count = 0;
   if (guest_count) {
      uint64_t bytes = (uint64_t)bound * sizeof(struct agc_register);
      struct agc_register *host = arena_alloc(queue, bytes, 0, sequence, &va);
      if (!host) {
         agc_report(function, "out of driver memory for a register array of %u registers", guest_count);
         return false;
      }
      host_count = agc_reg_convert_pairs_state(reloc->bank, guest, guest_count, host, queue_context(queue), uc,
                                               AGC_REGISTERS_UC_MAX, &uc_count, function);
      arena_trim(queue, (uint64_t)(bound - host_count) * sizeof(struct agc_register));
      if (reloc->bank == AGC_REG_BANK_CX && census_enabled()) {
         census_note(host, host_count);
         census_guest_depth(guest, guest_count);
         census_guest_color(guest, guest_count);
      }
   }
   if (uc_count) {
      if (depth) {
         agc_fatal(function,
                   "a register array in a called command buffer needs %u uconfig writes: the call that would make "
                   "them nests a third command buffer level",
                   uc_count);
      }
      uint32_t most = AGC_INDIRECT_REGISTERS_DWORDS + 3 * uc_count + AGC_REGISTERS_IB_PAD;
      uint64_t ib_va = 0;
      uint32_t *ib = arena_alloc(queue, (uint64_t)most * 4, 256, sequence, &ib_va);
      if (!ib) {
         agc_report(function, "out of driver memory for the uconfig writes of a register array");
         return false;
      }
      uint32_t ib_dwords = agc_registers_indirect_ib(ib, reloc->bank, va, host_count, uc, uc_count);
      agc_resolve_registers_call(record, reloc, ib_va, ib_dwords);
   } else {
      agc_resolve_registers_indirect(record, reloc, va, host_count);
   }
   count(&driver.counters.register_arrays, 1);
   count(&driver.counters.converted_registers, guest_count);
   return true;
}

static struct agc_step *
step_new(uint64_t va)
{
   struct agc_step *step = calloc(1, sizeof(*step));
   if (step) {
      step->va = va;
      step->stamp = p_atomic_inc_return(&label_log.stamp);
   }
   return step;
}

static uint32_t
count_steps(const struct agc_record *record, uint32_t first, uint32_t limit)
{
   uint32_t steps = 1;
   for (uint32_t i = first_reloc(record, first); i < record->reloc_count; i++) {
      const struct agc_reloc *reloc = &record->relocs[i];
      if (reloc->offset >= limit)
         break;
      if ((reloc->kind == AGC_RELOC_WAIT_RENDERING || reloc->kind == AGC_RELOC_LABEL_WAIT) &&
          agc_reloc_live(record, reloc, limit))
         steps++;
   }
   return steps;
}

static bool
condition_met(uint32_t function, uint64_t reference, uint64_t mask, uint64_t value)
{
   uint64_t masked = value & mask;
   switch (function) {
   case V_3C1_ALWAYS_PASS:
      return true;
   case V_3C1_LESS_THAN_REF_VALUE:
      return masked < reference;
   case V_3C1_LESS_THAN_EQUAL_TO_THE_REF_VALUE:
      return masked <= reference;
   case V_3C1_EQUAL_TO_THE_REFERENCE_VALUE:
      return masked == reference;
   case V_3C1_NOT_EQUAL_REFERENCE_VALUE:
      return masked != reference;
   case V_3C1_GREATER_THAN_OR_EQUAL_REFERENCE_VALUE:
      return masked >= reference;
   case V_3C1_GREATER_THAN_REFERENCE_VALUE:
      return masked > reference;
   default:
      return true;
   }
}

static bool
label_write_value(const struct agc_record *record, const struct agc_reloc *reloc, uint64_t *value, bool *wide)
{
   *wide = false;
   if (reloc->kind == AGC_RELOC_LABEL_WRITE && reloc->flags == AGC_LABEL_WRITE_ARG) {
      *value = reloc->arg64[1];
      return (reloc->arg32 & AGC_LABEL_WRITE_VALUE_KNOWN) != 0;
   }
   const uint32_t *packet = record->bottom + reloc->offset;
   switch (PKT3_IT_OPCODE_G(packet[0])) {
   case PKT3_RELEASE_MEM:
      if (G_492_DATA_SEL(packet[2]) == V_492_SEND_32_BIT_LOW) {
         *value = packet[5];
         return true;
      }
      if (G_492_DATA_SEL(packet[2]) == V_492_SEND_64_BIT_DATA) {
         *value = packet[5] | (uint64_t)packet[6] << 32;
         *wide = true;
         return true;
      }
      return false;
   case PKT3_WRITE_DATA:
      *wide = PKT_COUNT_G(packet[0]) >= 4;
      *value = packet[4] | (*wide ? (uint64_t)packet[5] << 32 : 0);
      return true;
   case PKT3_DMA_DATA:
      if (G_501_SRC_SEL(packet[1]) == V_501_DATA) {
         *wide = G_506_BYTE_COUNT(packet[6]) >= 8;
         *value = packet[2] | (*wide ? (uint64_t)packet[2] << 32 : 0);
         return true;
      }
      return false;
   default:
      return false;
   }
}

static bool
step_note_write(struct agc_step *step, const struct agc_record *record, const struct agc_reloc *reloc, uint64_t va)
{
   uint64_t value;
   bool wide;
   if (!va || !label_write_value(record, reloc, &value, &wide))
      return true;
   uint32_t slot = step->write_count;
   for (uint32_t i = 0; i < step->write_count; i++) {
      if (step->writes[i].va == va)
         slot = i;
   }
   if (slot == step->write_capacity) {
      uint32_t capacity = step->write_capacity ? step->write_capacity * 2 : 8;
      struct label_write *writes = realloc(step->writes, capacity * sizeof(*writes));
      if (!writes)
         return false;
      step->writes = writes;
      step->write_capacity = capacity;
   }
   step->writes[slot] = (struct label_write){.va = va, .value = value, .known = true, .wide = wide};
   if (slot == step->write_count)
      step->write_count++;
   return true;
}

static void
note_label_write(struct agc_queue *queue, const struct agc_record *record, const struct agc_reloc *reloc,
                 uint64_t va)
{
   if (!va)
      return;
   if (queue->write_count == queue->write_capacity) {
      uint32_t capacity = queue->write_capacity ? queue->write_capacity * 2 : 64;
      struct label_write *writes = realloc(queue->writes, capacity * sizeof(*writes));
      if (!writes)
         return;
      queue->writes = writes;
      queue->write_capacity = capacity;
   }
   struct label_write *write = &queue->writes[queue->write_count++];
   write->va = va;
   write->known = label_write_value(record, reloc, &write->value, &write->wide);
}

static bool
label_written_before(const struct agc_queue *queue, uint64_t va, uint32_t function, uint64_t reference,
                     uint64_t mask)
{
   for (uint32_t i = queue->write_count; i-- > 0;) {
      const struct label_write *write = &queue->writes[i];
      if ((write->va & ~UINT64_C(7)) != (va & ~UINT64_C(7)))
         continue;
      return !write->known || condition_met(function, reference, mask, write->value);
   }
   return false;
}

static uint64_t
label_write_address(const struct agc_record *record, const struct agc_reloc *reloc)
{
   if (reloc->flags == AGC_LABEL_WRITE_ARG)
      return reloc->arg64[0];
   const uint32_t *packet = record->bottom + reloc->offset;
   return packet[reloc->flags] | (uint64_t)packet[reloc->flags + 1] << 32;
}

static bool
label_split_disabled(void)
{
   static int disabled = -1;
   if (disabled < 0) {
      const char *value = getenv("AGC_NO_LABEL_SPLIT");
      disabled = value && *value && *value != '0';
   }
   return disabled;
}

static void
draw_state_note(struct agc_queue *queue, const struct agc_record *record, const struct agc_reloc *reloc)
{
   if (reloc->flags >= AGC_DRAW_STATE_COUNT)
      return;
   const uint32_t *packet = record->bottom + reloc->offset;
   uint32_t dwords = PKT_COUNT_G(packet[0]) + 2;
   if (dwords > AGC_DRAW_STATE_MAX_DWORDS)
      return;
   memset(queue->draw.words[reloc->flags], 0, sizeof(queue->draw.words[reloc->flags]));
   memcpy(queue->draw.words[reloc->flags], packet, dwords * sizeof(uint32_t));
   queue->draw.valid[reloc->flags] = true;
}

static void
draw_state_prologue(const struct agc_queue *queue, struct agc_step *step)
{
   step->prologue_dwords = 0;
   for (unsigned kind = 0; kind < AGC_DRAW_STATE_COUNT; kind++) {
      if (!queue->draw.valid[kind]) {
         if (kind == AGC_DRAW_STATE_NUM_INSTANCES) {
            step->prologue[step->prologue_dwords++] = PKT3(PKT3_NUM_INSTANCES, 0, 0);
            step->prologue[step->prologue_dwords++] = 1;
         }
         continue;
      }
      uint32_t dwords = PKT_COUNT_G(queue->draw.words[kind][0]) + 2;
      memcpy(step->prologue + step->prologue_dwords, queue->draw.words[kind], dwords * sizeof(uint32_t));
      step->prologue_dwords += dwords;
   }
}

void
agc_driver_set_tess_factor_ring(uint64_t base, uint32_t bytes)
{
   simple_mtx_lock(&driver.tess.lock);
   driver.tess.base = base;
   driver.tess.bytes = bytes;
   driver.tess.has_ring = true;
   driver.tess.generation++;
   simple_mtx_unlock(&driver.tess.lock);
}

void
agc_driver_set_hs_offchip(uint32_t param)
{
   simple_mtx_lock(&driver.tess.lock);
   driver.tess.offchip = param;
   driver.tess.has_offchip = true;
   driver.tess.generation++;
   simple_mtx_unlock(&driver.tess.lock);
}

#define UC_OFFSET(reg) (((reg) - CIK_UCONFIG_REG_OFFSET) / 4)

static uint32_t
tess_registers(struct agc_register *host, uint32_t *generation, const char *function)
{
   simple_mtx_lock(&driver.tess.lock);
   if (!driver.tess.has_ring && !driver.tess.fallback.cpu &&
       !agc_mem_alloc(driver.info.tess_factor_ring_size, 0, &driver.tess.fallback)) {
      simple_mtx_unlock(&driver.tess.lock);
      agc_fatal(function, "out of driver memory for a tessellation factor ring of %u bytes",
                driver.info.tess_factor_ring_size);
   }
   uint64_t base = driver.tess.has_ring ? driver.tess.base : driver.tess.fallback.va;
   uint32_t bytes = driver.tess.has_ring ? driver.tess.bytes : driver.info.tess_factor_ring_size;
   const struct agc_register guest[] = {
      {UC_OFFSET(R_030938_VGT_TF_RING_SIZE), bytes / 4},
      {UC_OFFSET(R_030940_VGT_TF_MEMORY_BASE), (uint32_t)(base >> 8)},
      {UC_OFFSET(R_030984_VGT_TF_MEMORY_BASE_HI), (uint32_t)(base >> 40) & 0xff},
      {UC_OFFSET(R_03093C_VGT_HS_OFFCHIP_PARAM), driver.tess.offchip},
   };
   uint32_t count = driver.tess.has_offchip ? TESS_REGISTERS : TESS_REGISTERS - 1;
   uint32_t written = 0;
   for (uint32_t i = 0; i < count; i++) {
      if (agc_reg_convert(AGC_REG_BANK_UC, guest[i].offset, guest[i].value, &host[written].offset,
                          &host[written].value, function))
         written++;
   }
   if (!driver.tess.has_offchip) {
      host[written].offset = UC_OFFSET(R_03093C_VGT_HS_OFFCHIP_PARAM);
      host[written].value = driver.info.hs_offchip_param;
      written++;
   }
   *generation = driver.tess.generation;
   simple_mtx_unlock(&driver.tess.lock);
   return written;
}

static void
tess_prologue(struct agc_queue *queue, struct agc_step *step, const char *function)
{
   if (queue->id != AGC_QUEUE_GRAPHICS || !queue->shadowed ||
       p_atomic_read(&driver.tess.generation) == queue->tess_applied)
      return;
   struct agc_register host[TESS_REGISTERS];
   uint32_t generation;
   uint32_t count = tess_registers(host, &generation, function);
   uint32_t *defaults = queue->state.defaults.cpu;
   for (uint32_t i = 0; i < count; i++) {
      step->prologue[step->prologue_dwords++] = PKT3(PKT3_SET_UCONFIG_REG, 1, 0);
      step->prologue[step->prologue_dwords++] = host[i].offset;
      step->prologue[step->prologue_dwords++] = host[i].value;
      if (defaults && (uint64_t)host[i].offset * 4 < SI_UCONFIG_REG_SPACE_SIZE)
         defaults[(SI_SHADOWED_UCONFIG_REG_OFFSET + (uint64_t)host[i].offset * 4) / 4] = host[i].value;
   }
   queue->tess_applied = generation;
}

static void
label_condition(const struct agc_record *record, const struct agc_reloc *reloc, struct agc_step *step)
{
   const uint32_t *packet = record->bottom + reloc->offset;
   step->label = true;
   step->label_function = (uint8_t)AGC_LABEL_WAIT_FUNCTION(reloc->flags);
   step->label_64 = (reloc->flags & AGC_LABEL_WAIT_64) != 0;
   if (!(reloc->flags & AGC_LABEL_WAIT_FROM_PACKET)) {
      step->label_va = reloc->arg64[0];
      step->label_reference = reloc->arg64[1];
      step->label_mask = step->label_64 ? UINT64_MAX : UINT32_MAX;
      return;
   }
   step->label_va = packet[2] | (uint64_t)packet[3] << 32;
   if (step->label_64) {
      step->label_reference = packet[4] | (uint64_t)packet[5] << 32;
      step->label_mask = packet[6] | (uint64_t)packet[7] << 32;
   } else {
      step->label_reference = packet[4];
      step->label_mask = packet[5];
   }
}

static void
free_steps(struct agc_step *head)
{
   while (head) {
      struct agc_step *next = head->next;
      step_free(head);
      head = next;
   }
}

#define FOLLOW_BUFFERS_MAX 4096u
#define FOLLOW_CALL_DEPTH 4u

static struct {
   uint64_t buffers;
   uint64_t unrecorded;
   uint64_t stale;
   uint64_t packets;
   uint64_t submits;
   uint64_t ns;
   uint64_t flips;
   uint64_t stopped;
   uint64_t kinds[AGC_RELOC_KIND_COUNT];
} follow_stats;

static bool
settle_reloc(struct agc_queue *queue, uint32_t queue_id, struct agc_record *record, struct agc_reloc *reloc,
             struct agc_step *step, uint64_t sequence, uint32_t depth, const char *function)
{
   struct agc_event event = {0};
   switch (reloc->kind) {
   case AGC_RELOC_REG_ARRAY:
      return resolve_register_array(queue, record, reloc, sequence, depth, function);
   case AGC_RELOC_FLIP:
      queue->flips_settled++;
      count(&follow_stats.flips, 1);
      event.kind = AGC_EVENT_FLIP;
      event.queue = queue_id;
      event.flip.handle = reloc->arg32;
      event.flip.index = (int32_t)(uint32_t)reloc->arg64[0];
      event.flip.mode = (uint32_t)(reloc->arg64[0] >> 32);
      event.flip.argument = (int64_t)reloc->arg64[1];
      return step_add_event(step, &event);
   case AGC_RELOC_NOTIFY: {
      const uint32_t *packet = record->bottom + reloc->offset;
      bool ok = true;
      if (G_492_DATA_SEL(packet[2]) != V_492_NONE) {
         note_label_write(queue, record, reloc, packet[3] | (uint64_t)packet[4] << 32);
         ok = step_note_write(step, record, reloc, packet[3] | (uint64_t)packet[4] << 32);
      }
      event.kind = AGC_EVENT_EOP;
      event.queue = queue_id;
      event.context = reloc->arg32;
      return ok && step_add_event(step, &event);
   }
   case AGC_RELOC_LABEL_WRITE:
      note_label_write(queue, record, reloc, label_write_address(record, reloc));
      return step_note_write(step, record, reloc, label_write_address(record, reloc));
   case AGC_RELOC_DRAW_STATE:
      draw_state_note(queue, record, reloc);
      return true;
   case AGC_RELOC_CONTEXT_OP:
      context_op(queue, reloc->arg32);
      return true;
   default:
      agc_fatal(function, "relocation kind %u has no submit handling", reloc->kind);
   }
}

static void
follow_kinds_text(const uint64_t *kinds, char *text, size_t bytes)
{
   size_t used = 0;
   for (unsigned kind = 0; kind < AGC_RELOC_KIND_COUNT && used < bytes; kind++) {
      int n = snprintf(text + used, bytes - used, "%s%" PRIu64, kind ? " " : "", kinds[kind]);
      if (n < 0)
         break;
      used += (size_t)n;
   }
}

static void
follow_report(void)
{
   char followed[256];
   follow_kinds_text(follow_stats.kinds, followed, sizeof(followed));
   uint64_t submits = MAX2(follow_stats.submits, 1);
   agc_stat("agc_driver",
            "followed %" PRIu64 " command buffers (%" PRIu64 " jump targets outside any record, %" PRIu64
            " stale relocations), their relocations by kind [%s] (arrays, flips, rendering waits, notifies, label "
            "waits, label writes, cond exec, draw state, context ops); scanned %" PRIu64 " packets, %.2f us and %" PRIu64
            " packets a submission; register arrays %" PRIu64 ", flips %" PRIu64 ", interrupts %" PRIu64
            ", submits %" PRIu64,
            follow_stats.buffers, follow_stats.unrecorded, follow_stats.stale, followed, follow_stats.packets,
            (double)follow_stats.ns / 1000.0 / (double)submits, follow_stats.packets / submits,
            driver.counters.register_arrays, driver.counters.flips, driver.counters.interrupts,
            driver.counters.submits);
   agc_stat("agc_driver",
            "the title recorded %u flips and %" PRIu64 " were found at submit; %" PRIu64
            " submissions stopped at a packet the follow could not step over",
            p_atomic_read(&agc_flip_records), follow_stats.flips, follow_stats.stopped);
}

static uint64_t
jump_address(const uint32_t *packet)
{
   return (packet[1] & ~UINT32_C(3)) | (uint64_t)(packet[2] & 0xffff) << 32;
}

static void
trace_visit(struct agc_queue *queue, const struct agc_record *record, uint32_t first, uint32_t limit, uint32_t depth)
{
   uint32_t at = queue->trace_count++;
   if (at >= FOLLOW_TRACE_MAX)
      return;
   queue->trace_va[at] = (uintptr_t)record->bottom;
   queue->trace_first[at] = first;
   queue->trace_limit[at] = limit;
   queue->trace_depth[at] = (uint8_t)MIN2(depth, 255u);
}

static void
trace_end(struct agc_queue *queue, enum follow_end reason, uint64_t va)
{
   queue->trace_end_reason = reason;
   queue->trace_end = va;
   if (reason == FOLLOW_END_SCAN_STOPPED)
      count(&follow_stats.stopped, 1);
}

static uint32_t flip_missing_count;

static void
flip_missing_report(const struct agc_queue *queue, const uint32_t *address, uint32_t dwords, uint32_t recorded,
                    uint32_t followed)
{
   if (p_atomic_inc_return(&flip_missing_count) > FLIP_MISSING_REPORTS)
      return;
   static const char *const ends[] = {
      [FOLLOW_END_NONE] = "the end of the submission",
      [FOLLOW_END_UNRESOLVED] = "a jump to a target outside every recorded segment,",
      [FOLLOW_END_NO_CHAIN] = "a segment without a closing jump, at",
      [FOLLOW_END_SCAN_STOPPED] = "a packet it could not step over, at",
      [FOLLOW_END_BUDGET] = "the follow budget, at",
   };
   char header[48] = "";
   if (queue->trace_end_reason == FOLLOW_END_SCAN_STOPPED && agc_va_in_guest_window(queue->trace_end))
      snprintf(header, sizeof(header), " (word 0x%08x)", *(const volatile uint32_t *)(uintptr_t)queue->trace_end);
   agc_stat("agc_driver",
            "graphics submission 0x%llx (%u dwords) found no flip although the title recorded %u since the previous "
            "one; it followed %u buffers and ended at %s 0x%llx%s",
            (unsigned long long)(uintptr_t)address, dwords, recorded, followed, ends[queue->trace_end_reason],
            (unsigned long long)queue->trace_end, header);
   const uint32_t *flip = (const uint32_t *)(uintptr_t)p_atomic_read(&agc_flip_last);
   const struct agc_record *holder = flip ? agc_record_containing(flip) : NULL;
   uint32_t traced = MIN2(queue->trace_count, FOLLOW_TRACE_MAX);
   if (!holder) {
      agc_stat("agc_driver", "  the last flip packet 0x%llx is in no recorded segment",
               (unsigned long long)(uintptr_t)flip);
   } else {
      uint32_t offset = (uint32_t)(flip - holder->bottom);
      bool kept = false;
      for (uint32_t i = 0; i < holder->reloc_count; i++)
         kept |= holder->relocs[i].kind == AGC_RELOC_FLIP && holder->relocs[i].offset == offset;
      int visit = -1;
      for (uint32_t i = 0; i < traced && visit < 0; i++) {
         if (queue->trace_va[i] == (uintptr_t)holder->bottom)
            visit = (int)i;
      }
      agc_stat("agc_driver",
               "  the last flip packet is dword %u of segment 0x%llx (%u dwords recorded, %u relocations, the flip "
               "relocation is %s); the follow %s",
               offset, (unsigned long long)(uintptr_t)holder->bottom, (uint32_t)(holder->end - holder->bottom),
               holder->reloc_count, kept ? "there" : "gone",
               visit < 0 ? "never entered that segment" : "entered that segment");
      if (visit >= 0)
         agc_stat("agc_driver", "  visit %d covered dwords %u to %u of it", visit, queue->trace_first[visit],
                  queue->trace_limit[visit]);
   }
   for (uint32_t i = 0; i < traced; i++) {
      if (i < 4 || i + 12 >= traced)
         agc_stat("agc_driver", "  followed %u: 0x%llx dwords %u to %u, call depth %u", i,
                  (unsigned long long)queue->trace_va[i], queue->trace_first[i], queue->trace_limit[i],
                  queue->trace_depth[i]);
   }
}

static struct agc_record *
jump_target(const uint32_t *packet, uint32_t *first, uint32_t *limit)
{
   uint32_t dwords = G_3F3_IB_SIZE(packet[3]);
   uint64_t target = (packet[1] & ~UINT32_C(3)) | (uint64_t)(packet[2] & 0xffff) << 32;
   if (agc_va_in_mirror_window(target))
      target -= AGC_MIRROR_OFFSET;
   if (!dwords || !agc_va_in_guest_window(target))
      return NULL;
   const uint32_t *entry = (const uint32_t *)(uintptr_t)target;
   struct agc_record *record = agc_record_find(entry);
   if (!record)
      record = agc_record_containing(entry);
   if (!record) {
      count(&follow_stats.unrecorded, 1);
      return NULL;
   }
   *first = (uint32_t)(entry - record->bottom);
   *limit = *first + dwords;
   return record;
}

static bool follow_range(struct agc_queue *queue, uint32_t queue_id, struct agc_record *record, uint32_t first,
                         uint32_t limit, struct agc_step *step, uint64_t sequence, uint32_t depth, uint32_t *budget,
                         const char *function);

static bool
follow_packets(struct agc_queue *queue, uint32_t queue_id, const uint32_t *base, uint32_t *cursor, uint32_t until,
               struct agc_step *step, uint64_t sequence, uint32_t depth, uint32_t *budget, const uint32_t **tail,
               const char *function)
{
   uint32_t at = *cursor;
   uint64_t packets = 0;
   while (at < until && !*tail) {
      const uint32_t *packet = base + at;
      uint32_t dwords = packet_dwords(packet[0]);
      if (!dwords || dwords > until - at)
         break;
      packets++;
      if (dwords == AGC_DWORDS_JUMP && PKT_TYPE_G(packet[0]) == 3 &&
          PKT3_IT_OPCODE_G(packet[0]) == PKT3_INDIRECT_BUFFER) {
         uint32_t first;
         uint32_t limit;
         if (packet[3] & S_3F3_CHAIN(1)) {
            *tail = packet;
         } else if (depth < FOLLOW_CALL_DEPTH) {
            struct agc_record *callee = jump_target(packet, &first, &limit);
            if (callee &&
                !follow_range(queue, queue_id, callee, first, limit, step, sequence, depth + 1, budget, function))
               return false;
         }
      }
      at += dwords;
   }
   *cursor = at;
   count(&follow_stats.packets, packets);
   return true;
}

static bool
follow_range(struct agc_queue *queue, uint32_t queue_id, struct agc_record *record, uint32_t first, uint32_t limit,
             struct agc_step *step, uint64_t sequence, uint32_t depth, uint32_t *budget, const char *function)
{
   while (record) {
      if (!*budget) {
         agc_report(function, "a submission runs through more than %u command buffers; the rest is not followed",
                    FOLLOW_BUFFERS_MAX);
         if (!depth)
            trace_end(queue, FOLLOW_END_BUDGET, (uintptr_t)(record->bottom + first));
         return true;
      }
      (*budget)--;
      count(&follow_stats.buffers, 1);
      trace_visit(queue, record, first, limit, depth);
      const uint32_t *tail = NULL;
      uint32_t cursor = first;
      for (uint32_t i = first_reloc(record, first); !tail && i < record->reloc_count; i++) {
         struct agc_reloc *reloc = &record->relocs[i];
         if (reloc->offset >= limit)
            break;
         if (!follow_packets(queue, queue_id, record->bottom, &cursor, reloc->offset, step, sequence, depth, budget,
                             &tail, function))
            return false;
         if (tail)
            break;
         if (!agc_reloc_live(record, reloc, limit)) {
            count(&follow_stats.stale, 1);
            continue;
         }
         count(&follow_stats.kinds[reloc->kind], 1);
         switch (reloc->kind) {
         case AGC_RELOC_WAIT_RENDERING:
            if (!step->wait) {
               step->wait = true;
               step->wait_handle = reloc->arg32;
               step->wait_index = (uint32_t)reloc->arg64[0];
            }
            break;
         case AGC_RELOC_COND_EXEC:
         case AGC_RELOC_LABEL_WAIT:
            break;
         default:
            if (!settle_reloc(queue, queue_id, record, reloc, step, sequence, depth, function))
               return false;
         }
      }
      if (!tail && !follow_packets(queue, queue_id, record->bottom, &cursor, limit, step, sequence, depth, budget,
                                   &tail, function))
         return false;
      if (tail) {
         const uint32_t *chain = tail;
         record = jump_target(chain, &first, &limit);
         if (!record && !depth)
            trace_end(queue, FOLLOW_END_UNRESOLVED, jump_address(chain));
      } else {
         if (!depth)
            trace_end(queue, cursor < limit ? FOLLOW_END_SCAN_STOPPED : FOLLOW_END_NO_CHAIN,
                      (uintptr_t)(record->bottom + cursor));
         record = NULL;
      }
   }
   return true;
}

int32_t
agc_driver_submit(uint32_t queue_id, const uint32_t *address, uint32_t dwords, const char *function)
{
   char text[256];
   if (dwords && (!address || (uintptr_t)address % 4 != 0)) {
      agc_report(function, "null or misaligned command buffer");
      return AGC_DRIVER_ERROR_INVALID;
   }
   if (!agc_driver_start(text, sizeof(text))) {
      agc_report(function, "%s", text);
      return AGC_DRIVER_ERROR_INVALID;
   }
   driver.anyps5->note_gpu_submit();
   const char *failure = agc_flips_failure();
   if (failure) {
      agc_report(function, "%s", failure);
      submit_log_note(queue_id, address, dwords, false, NULL, 0, 0, AGC_DRIVER_ERROR_INVALID);
      return AGC_DRIVER_ERROR_INVALID;
   }
   if (!dwords) {
      submit_log_note(queue_id, address, 0, false, NULL, 0, 0, 0);
      return 0;
   }
   struct agc_queue *queue = queue_get(queue_id, function);
   if (!queue)
      return AGC_DRIVER_ERROR_INVALID;

   struct agc_record *record = agc_record_containing(address);
   uint32_t first = 0;
   uint32_t limit = 0;
   if (record) {
      first = (uint32_t)(address - record->bottom);
      limit = first + dwords;
   }

   mtx_lock(&queue->lock);
   uint32_t planned = record ? count_steps(record, first, limit) : 1;
   uint64_t sequence = queue->enqueued + planned;
   struct agc_step *head = step_new((uintptr_t)address);
   struct agc_step *step = head;
   uint32_t cursor = first;
   uint32_t protected_until = 0;
   uint32_t budget = FOLLOW_BUFFERS_MAX;
   uint32_t scanned = first;
   const uint32_t *tail = NULL;
   const uint64_t planning = os_time_get_nano();
   bool ok = head != NULL;
   queue->write_count = 0;
   queue->trace_count = 0;
   queue->trace_end = 0;
   queue->trace_end_reason = FOLLOW_END_NONE;
   queue->flips_settled = 0;
   if (record)
      trace_visit(queue, record, first, limit, 0);
   if (head && queue->id == AGC_QUEUE_GRAPHICS) {
      draw_state_prologue(queue, head);
      tess_prologue(queue, head, function);
   }

   for (uint32_t i = record && ok ? first_reloc(record, first) : 0; ok && record && i < record->reloc_count; i++) {
      struct agc_reloc *reloc = &record->relocs[i];
      if (reloc->offset >= limit)
         break;
      if (!agc_reloc_live(record, reloc, limit))
         continue;
      ok = follow_packets(queue, queue_id, record->bottom, &scanned, reloc->offset, step, sequence, 0, &budget, &tail,
                          function);
      if (!ok || tail)
         break;
      switch (reloc->kind) {
      case AGC_RELOC_COND_EXEC:
         protected_until = MAX2(protected_until, reloc->offset + AGC_DWORDS_COND_EXEC +
                                                    G_224_EXEC_COUNT(record->bottom[reloc->offset + 4]));
         break;
      case AGC_RELOC_LABEL_WAIT: {
         if (reloc->offset < protected_until || label_split_disabled())
            break;
         struct agc_step probe = {0};
         label_condition(record, reloc, &probe);
         if (label_written_before(queue, probe.label_va, probe.label_function, probe.label_reference,
                                  probe.label_mask))
            break;
         if (reloc->offset > cursor || step->event_count || step->wait || step->label) {
            struct agc_step *next = step_new((uintptr_t)(record->bottom + reloc->offset));
            if (!next) {
               ok = false;
               break;
            }
            step->dwords = reloc->offset - cursor;
            step->next = next;
            step = next;
            cursor = reloc->offset;
            if (queue->id == AGC_QUEUE_GRAPHICS)
               draw_state_prologue(queue, step);
         }
         label_condition(record, reloc, step);
         step->label_since = p_atomic_read(&label_log.serial);
         if ((reloc->flags & AGC_LABEL_WAIT_FROM_PACKET) && label_cpu(step))
            step->label_skip = agc_wait_dwords(step->label_64 ? 1 : 0);
         break;
      }
      case AGC_RELOC_WAIT_RENDERING:
         if (reloc->offset > cursor || step->event_count || step->wait || step->label) {
            struct agc_step *next = step_new((uintptr_t)(record->bottom + reloc->offset));
            if (!next) {
               ok = false;
               break;
            }
            step->dwords = reloc->offset - cursor;
            step->next = next;
            step = next;
            cursor = reloc->offset;
            if (queue->id == AGC_QUEUE_GRAPHICS)
               draw_state_prologue(queue, step);
         }
         step->wait = true;
         step->wait_handle = reloc->arg32;
         step->wait_index = (uint32_t)reloc->arg64[0];
         break;
      default:
         ok = settle_reloc(queue, queue_id, record, reloc, step, sequence, 0, function);
      }
   }
   if (ok && !tail)
      ok = follow_packets(queue, queue_id, record ? record->bottom : address, &scanned, record ? limit : dwords, step,
                          sequence, 0, &budget, &tail, function);
   if (ok && tail) {
      uint32_t tail_first;
      uint32_t tail_limit;
      struct agc_record *next = jump_target(tail, &tail_first, &tail_limit);
      if (next)
         ok = follow_range(queue, queue_id, next, tail_first, tail_limit, step, sequence, 0, &budget, function);
      else
         trace_end(queue, FOLLOW_END_UNRESOLVED, jump_address(tail));
   } else if (ok) {
      uint32_t end = record ? limit : dwords;
      trace_end(queue, scanned < end ? FOLLOW_END_SCAN_STOPPED : FOLLOW_END_NO_CHAIN,
                (uintptr_t)((record ? record->bottom : address) + scanned));
   }
   count(&follow_stats.ns, os_time_get_nano() - planning);
   count(&follow_stats.submits, 1);
   if (!ok) {
      mtx_unlock(&queue->lock);
      free_steps(head);
      agc_report(function, "out of memory while preparing a submission");
      return AGC_DRIVER_ERROR_INVALID;
   }
   step->dwords = (record ? limit : dwords) - cursor;

   for (struct agc_step *each = head; ok && each; each = each->next) {
      if (each->label_skip && each->dwords >= each->label_skip) {
         each->va += (uint64_t)each->label_skip * 4;
         each->dwords -= each->label_skip;
      }
      if (each->dwords)
         ok = prepare_ibs(queue, each, sequence, function);
   }
   if (!ok) {
      mtx_unlock(&queue->lock);
      free_steps(head);
      return AGC_DRIVER_ERROR_INVALID;
   }
   _mm_sfence();

   uint64_t number = queue->enqueued;
   for (struct agc_step *each = head; each; each = each->next)
      each->sequence = ++number;
   for (uint32_t i = 0; i < queue->chunk_count; i++) {
      if (queue->chunks[i].last_step == sequence)
         queue->chunks[i].last_step = number;
   }
   queue->enqueued = number;
   uint32_t followed = FOLLOW_BUFFERS_MAX - budget;
   if (queue->id == AGC_QUEUE_GRAPHICS) {
      uint32_t flips = p_atomic_read(&agc_flip_records);
      if (!queue->flips_settled && flips != queue->flip_records_seen)
         flip_missing_report(queue, address, dwords, flips - queue->flip_records_seen, followed);
      queue->flip_records_seen = flips;
   }
   submit_log_note(queue_id, address, dwords, record != NULL, head, followed, queue->flips_settled, 0);
   if (queue->ready_tail)
      queue->ready_tail->next = head;
   else
      queue->ready_head = head;
   queue->ready_tail = step;
   cnd_signal(&queue->submit_wake);
   mtx_unlock(&queue->lock);
   count(&driver.counters.submits, 1);
   static uint32_t graphics_submits;
   if (queue_id == AGC_QUEUE_GRAPHICS && p_atomic_inc_return(&graphics_submits) % 500 == 1)
      follow_report();
   return 0;
}

static void
queue_wait_idle(struct agc_queue *queue)
{
   mtx_lock(&queue->lock);
   while (queue->ready_head || queue->pending_head || queue->submitting)
      cnd_wait(&queue->idle_wake, &queue->lock);
   mtx_unlock(&queue->lock);
}

void
agc_driver_wait_idle(void)
{
   if (!agc_driver_started())
      return;
   for (uint32_t i = 0; i < QUEUE_SLOTS; i++) {
      struct agc_queue *queue = p_atomic_read(&driver.queues[i]);
      if (queue)
         queue_wait_idle(queue);
   }
}

void
agc_driver_shutdown(void)
{
   call_once(&driver_once, driver_init_locks);
   mtx_lock(&driver.lock);
   bool was = driver.stopping;
   p_atomic_set(&driver.stopping, true);
   mtx_unlock(&driver.lock);
   if (was)
      return;
   agc_flips_shutdown();
   mtx_lock(&driver.flip_lock);
   cnd_broadcast(&driver.flip_wake);
   mtx_unlock(&driver.flip_lock);
   for (uint32_t i = 0; i < QUEUE_SLOTS; i++) {
      struct agc_queue *queue = p_atomic_read(&driver.queues[i]);
      if (!queue)
         continue;
      mtx_lock(&queue->lock);
      queue->stop = true;
      cnd_broadcast(&queue->submit_wake);
      cnd_broadcast(&queue->complete_wake);
      mtx_unlock(&queue->lock);
      thrd_join(queue->submit_thread, NULL);
      mtx_lock(&queue->lock);
      cnd_broadcast(&queue->complete_wake);
      mtx_unlock(&queue->lock);
      thrd_join(queue->complete_thread, NULL);
   }
}

static void
quiesce(void *user)
{
   (void)user;
   agc_driver_wait_idle();
}

static bool
select_host(const struct radeon_info *info)
{
   switch (info->gfx_level) {
   case GFX10_3:
      return agc_host_select(AGC_HOST_GFX103);
   case GFX11:
      return agc_host_select(AGC_HOST_GFX11);
   case GFX11_5:
      return agc_host_select(AGC_HOST_GFX115);
   case GFX12:
      return agc_host_select(AGC_HOST_GFX12);
   default:
      return false;
   }
}

static bool
stub_alloc(uint64_t bytes, struct agc_stub_block *block)
{
   struct agc_mem *mem = malloc(sizeof(*mem));
   if (!mem)
      return false;
   if (!agc_mem_suballoc(bytes, AGC_STUB_ALIGNMENT, 0, mem)) {
      free(mem);
      return false;
   }
   block->cpu = mem->cpu;
   block->va = mem->va;
   block->bytes = mem->bytes;
   block->handle = mem;
   return true;
}

static void
stub_release(struct agc_stub_block *block)
{
   struct agc_mem *mem = block->handle;
   agc_mem_subfree(mem);
   free(mem);
}

static const struct agc_stub_source stub_source = {
   .alloc = stub_alloc,
   .release = stub_release,
};

static bool
open_device(char *error, size_t bytes)
{
   driver.anyps5 = agc_anyps5_get();
   if (!driver.anyps5) {
      snprintf(error, bytes, "AnyPS5's libc and libkernel exports are missing");
      return false;
   }
   driver.backend = agc_backend_open();
   if (!driver.backend) {
      snprintf(error, bytes, "no supported AMD GPU could be opened");
      return false;
   }
   if (!driver.backend->ops->query_info(driver.backend, &driver.info)) {
      snprintf(error, bytes, "the GPU did not describe itself");
      return false;
   }
   if (!select_host(&driver.info)) {
      snprintf(error, bytes, "GPU generation %d is not supported", (int)driver.info.gfx_level);
      return false;
   }
   agc_host_caps.shader_engines = driver.info.max_se ? driver.info.max_se : 1;
   driver.graphics = queue_new(AGC_QUEUE_GRAPHICS, "agc_driver_start");
   if (!driver.graphics) {
      snprintf(error, bytes, "the kernel refused to create the graphics queue");
      return false;
   }
   struct agc_memory_config config = {
      .backend = driver.backend,
      .anyps5 = driver.anyps5,
      .quiesce = quiesce,
      .released = agc_images_released,
   };
   if (!agc_memory_init(&config, error, bytes) || !agc_runtime_init(error, bytes))
      return false;
   agc_record_set_stub_source(&stub_source);
   agc_reg_set_target_note(agc_images_note_target);
   return true;
}

static bool
device_locked(char *error, size_t bytes)
{
   if (!driver.device_open && !driver.device_failed) {
      if (open_device(driver.start_error, sizeof(driver.start_error)))
         p_atomic_set(&driver.device_open, true);
      else
         driver.device_failed = true;
   }
   if (!driver.device_open && error && bytes)
      snprintf(error, bytes, "%s", driver.start_error);
   return driver.device_open;
}

static bool
start(char *error, size_t bytes)
{
   static const struct agc_videoout_core core = {
      .wait_idle = agc_driver_wait_idle,
      .shutdown = agc_driver_shutdown,
   };
   if (!device_locked(error, bytes))
      return false;
   static struct agc_host_defaults defaults;
   if (!driver.defaults) {
      uint32_t version = agc_register_defaults_init_version();
      for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++)
         defaults.registers[bank] = agc_default_state_host(version, (enum agc_reg_bank)bank, &defaults.count[bank]);
      driver.defaults = &defaults;
   }
   struct agc_queue *graphics = driver.graphics;
   if (!agc_queue_state_init(driver.backend, &driver.info, graphics->hwq, driver.defaults, &graphics->state,
                             error, bytes))
      return false;
   graphics->shadowed = true;
   queue_run(graphics);
   agc_videoout_set_core(&core);
   return true;
}

bool
agc_driver_open_device(char *error, size_t bytes)
{
   if (AGC_LIKELY(p_atomic_read(&driver.device_open)))
      return true;
   call_once(&driver_once, driver_init_locks);
   mtx_lock(&driver.lock);
   bool ok = device_locked(error, bytes);
   mtx_unlock(&driver.lock);
   return ok;
}

bool
agc_driver_start(char *error, size_t bytes)
{
   if (AGC_LIKELY(p_atomic_read(&driver.started)))
      return true;
   call_once(&driver_once, driver_init_locks);
   mtx_lock(&driver.lock);
   if (!driver.started && !driver.start_failed) {
      if (start(driver.start_error, sizeof(driver.start_error)))
         p_atomic_set(&driver.started, true);
      else
         driver.start_failed = true;
   }
   bool ok = driver.started;
   if (!ok && error && bytes)
      snprintf(error, bytes, "%s", driver.start_error);
   mtx_unlock(&driver.lock);
   return ok;
}

bool
agc_driver_started(void)
{
   return p_atomic_read(&driver.started);
}

struct agc_backend *
agc_driver_backend(void)
{
   return driver.backend;
}

const struct radeon_info *
agc_driver_info(void)
{
   return p_atomic_read(&driver.device_open) ? &driver.info : NULL;
}

void
agc_driver_set_defaults(const struct agc_host_defaults *defaults)
{
   driver.defaults = defaults;
}

bool
agc_driver_reset_sequence(uint64_t *va, uint32_t *dwords)
{
   struct agc_queue *queue = p_atomic_read(&driver.queues[AGC_QUEUE_GRAPHICS]);
   if (!queue || !queue->shadowed)
      return false;
   *va = queue->state.reset_va;
   *dwords = queue->state.reset_dwords;
   return true;
}

bool
agc_driver_context_sequence(uint32_t operation, uint64_t *va, uint32_t *dwords)
{
   struct agc_queue *queue = p_atomic_read(&driver.queues[AGC_QUEUE_GRAPHICS]);
   if (!queue || !queue->shadowed || operation >= AGC_CONTEXT_OPS)
      return false;
   *va = queue->state.context_va[operation];
   *dwords = queue->state.context_dwords[operation];
   return true;
}

bool
agc_driver_read_register(unsigned bank, uint32_t offset, uint32_t *value)
{
   struct agc_queue *queue = p_atomic_read(&driver.queues[AGC_QUEUE_GRAPHICS]);
   return queue && queue->shadowed && agc_queue_state_read(&queue->state, bank, offset, value);
}
