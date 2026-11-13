#include "agc_videoout.h"

#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/simple_mtx.h"
#include "util/u_atomic.h"

#include "agc_anyps5.h"
#include "agc_debug.h"

#ifdef _WIN32
#include <windows.h>
#endif

#define GRAPHICS_QUEUE UINT32_C(0)
#define FAILURE_BYTES 1024
#define DELIVERY_INLINE_FLIPS 8
#define GUEST_REGION_SLOTS 64

enum eop_state {
   EOP_EMPTY = 0,
   EOP_CLAIMED,
   EOP_READY
};

struct registered_output {
   struct agc_video_output table;
   int32_t references;
};

struct reserved_flip {
   struct registered_output *output;
   struct agc_videoout_request *request;
};

struct eop_slot {
   enum eop_state state;
   uint32_t token;
   struct agc_flip_info info;
   struct reserved_flip flip;
};

struct agc_rendering_wait {
   struct registered_output *output;
   struct agc_videoout_wait *captured;
};

struct event_registration {
   bool used;
   uint32_t gpu_queue;
   int64_t eq;
   uint64_t ident;
};

struct event_target {
   int64_t eq;
   uint64_t ident;
};

struct guest_region {
   uintptr_t base;
   uintptr_t end;
   bool writable;
};

static struct {
   simple_mtx_t lock;
   struct registered_output *outputs[AGC_VIDEO_OUTPUT_SLOTS];
   struct eop_slot eop[AGC_EOP_FLIP_SLOTS];
   uint32_t next_token;
   int stopped;
   int failed;
   uint64_t frames;
   char failure[FAILURE_BYTES];
} flips = {
   .lock = SIMPLE_MTX_INITIALIZER,
   .next_token = AGC_DRIVER_EOP_FLIP_TOKEN_BASE,
};

static struct {
   simple_mtx_t lock;
   struct event_registration slots[AGC_EVENT_REGISTRATION_SLOTS];
} registrations = {
   .lock = SIMPLE_MTX_INITIALIZER,
};

static struct {
   simple_mtx_t lock;
   struct agc_host_hdr_status status;
} hdr = {
   .lock = SIMPLE_MTX_INITIALIZER,
};

static struct {
   simple_mtx_t lock;
   uint64_t generation;
   uint32_t count;
   uint32_t oldest;
   struct guest_region regions[GUEST_REGION_SLOTS];
} guest_memory = {
   .lock = SIMPLE_MTX_INITIALIZER,
};

static const struct agc_presenter *installed_presenter;
static const struct agc_videoout_core *installed_core;
static uint64_t frames_presented;

static int set_error(struct agc_driver_error *error, int code, const char *format, ...) AGC_PRINTF(3, 4);

static int
set_error(struct agc_driver_error *error, int code, const char *format, ...)
{
   if (error) {
      va_list args;
      va_start(args, format);
      vsnprintf(error->message, sizeof(error->message), format, args);
      va_end(args);
   }
   return code;
}

static int
callback_error(struct agc_driver_error *error, int code, const char *what, const struct agc_driver_error *said)
{
   if (code > AGC_DRIVER_INVALID || code < AGC_DRIVER_ALREADY_RESERVED)
      code = AGC_DRIVER_FAILED;
   return set_error(error, code, "%s: %.*s", what, (int)sizeof(said->message), said->message);
}

static int
check_running(struct agc_driver_error *error)
{
   if (p_atomic_read(&flips.stopped))
      return set_error(error, AGC_DRIVER_SHUTDOWN, "the video output driver has stopped");
   if (p_atomic_read(&flips.failed))
      return set_error(error, AGC_DRIVER_FAILED, "%s", flips.failure);
   return AGC_DRIVER_OK;
}

static bool
table_complete(const struct agc_video_output *table)
{
   return table->self && table->reserve && table->wait_for_flip_room && table->capture_rendering_wait &&
          table->describe_buffer && table->fail && table->release && table->flip_ready && table->flip_fail &&
          table->flip_release && table->rendering_wait_for && table->rendering_wait_release;
}

static void
output_release(struct registered_output *output)
{
   if (!p_atomic_dec_zero(&output->references))
      return;
   output->table.release(output->table.self);
   free(output);
}

static int
output_acquire(uint32_t handle, struct registered_output **output, struct agc_driver_error *error)
{
   int code = check_running(error);
   if (code != AGC_DRIVER_OK)
      return code;
   if (handle >= AGC_VIDEO_OUTPUT_SLOTS)
      return set_error(error, AGC_DRIVER_INVALID, "video output handle %u is out of range", handle);
   simple_mtx_lock(&flips.lock);
   struct registered_output *found = flips.outputs[handle];
   if (found)
      p_atomic_inc(&found->references);
   simple_mtx_unlock(&flips.lock);
   if (!found)
      return set_error(error, AGC_DRIVER_INVALID, "video output %u is not registered", handle);
   *output = found;
   return AGC_DRIVER_OK;
}

static int
output_wait_for_flip_room(struct registered_output *output, struct agc_driver_error *error)
{
   struct agc_driver_error said = {0};
   int code = output->table.wait_for_flip_room(output->table.self, &said);
   if (code != AGC_DRIVER_OK)
      return callback_error(error, code, "VideoOut's flip queue", &said);
   return AGC_DRIVER_OK;
}

static int
output_reserve(struct registered_output *output, const struct agc_flip_info *info, bool limited,
               struct agc_videoout_request **request, struct agc_driver_error *error)
{
   struct agc_driver_error said = {0};
   struct agc_videoout_request *reserved = NULL;
   int code = output->table.reserve(output->table.self, info, limited, &reserved, &said);
   if (code == AGC_DRIVER_FULL && limited)
      return set_error(error, AGC_DRIVER_FULL, "the flip queue is full");
   if (code != AGC_DRIVER_OK)
      return callback_error(error, code, "VideoOut refused the flip", &said);
   if (!reserved)
      return set_error(error, AGC_DRIVER_FAILED, "VideoOut returned a null flip reservation");
   *request = reserved;
   return AGC_DRIVER_OK;
}

static int
reserve_flip(const struct agc_flip_info *info, bool limited, struct reserved_flip *flip,
             struct agc_driver_error *error)
{
   int code = output_acquire(info->handle, &flip->output, error);
   if (code != AGC_DRIVER_OK)
      return code;
   code = output_reserve(flip->output, info, limited, &flip->request, error);
   if (code != AGC_DRIVER_OK)
      output_release(flip->output);
   return code;
}

static void
flip_drop(const struct reserved_flip *flip)
{
   flip->output->table.flip_release(flip->request);
   output_release(flip->output);
}

static int
flip_ready(const struct reserved_flip *flip, struct agc_driver_error *error)
{
   struct agc_driver_error said = {0};
   uint64_t frame = (uint64_t)p_atomic_inc_return(&flips.frames);
   int code = flip->output->table.flip_ready(flip->request, frame, &said);
   flip_drop(flip);
   if (code != AGC_DRIVER_OK)
      return callback_error(error, code, "VideoOut's flip", &said);
   return AGC_DRIVER_OK;
}

static bool
eop_token_valid(uint32_t token)
{
   return token >= AGC_DRIVER_EOP_FLIP_TOKEN_BASE && token <= AGC_DRIVER_EOP_FLIP_TOKEN_LIMIT;
}

static struct eop_slot *
eop_slot_of(uint32_t token)
{
   return &flips.eop[(token - AGC_DRIVER_EOP_FLIP_TOKEN_BASE) % AGC_EOP_FLIP_SLOTS];
}

static bool
eop_reserved_locked(uint32_t token)
{
   if (!eop_token_valid(token))
      return false;
   const struct eop_slot *slot = eop_slot_of(token);
   return slot->state == EOP_READY && slot->token == token;
}

static struct reserved_flip
eop_take_locked(struct eop_slot *slot)
{
   struct reserved_flip flip = slot->flip;
   memset(slot, 0, sizeof(*slot));
   return flip;
}

static uint32_t
eop_take_all_locked(const struct registered_output *only, struct reserved_flip *taken)
{
   uint32_t count = 0;
   for (uint32_t i = 0; i < AGC_EOP_FLIP_SLOTS; i++) {
      struct eop_slot *slot = &flips.eop[i];
      if (slot->state == EOP_READY && (!only || slot->flip.output == only))
         taken[count++] = eop_take_locked(slot);
   }
   return count;
}

static bool
eop_take(uint32_t token, struct reserved_flip *flip)
{
   simple_mtx_lock(&flips.lock);
   bool reserved = eop_reserved_locked(token);
   if (reserved)
      *flip = eop_take_locked(eop_slot_of(token));
   simple_mtx_unlock(&flips.lock);
   return reserved;
}

static void
eop_abandon(struct eop_slot *slot)
{
   simple_mtx_lock(&flips.lock);
   slot->state = EOP_EMPTY;
   simple_mtx_unlock(&flips.lock);
}

static bool
same_flip(const struct agc_flip_info *a, const struct agc_flip_info *b)
{
   return a->handle == b->handle && a->index == b->index && a->mode == b->mode && a->argument == b->argument;
}

static int
reserve_eop_flip(const struct agc_flip_info *info, const uint32_t *chosen, uint32_t *reserved_token,
                 struct agc_driver_error *error)
{
   int code = check_running(error);
   if (code != AGC_DRIVER_OK)
      return code;

   simple_mtx_lock(&flips.lock);
   uint32_t token;
   if (chosen) {
      token = *chosen;
   } else {
      if (!eop_token_valid(flips.next_token))
         flips.next_token = AGC_DRIVER_EOP_FLIP_TOKEN_BASE;
      token = flips.next_token++;
   }
   if (!eop_token_valid(token)) {
      simple_mtx_unlock(&flips.lock);
      return set_error(error, AGC_DRIVER_INVALID, "EOP flip token 0x%x is out of range", token);
   }
   struct eop_slot *slot = eop_slot_of(token);
   if (slot->state != EOP_EMPTY) {
      bool token_reserved = slot->state == EOP_READY && slot->token == token;
      bool flip_reserved = token_reserved && same_flip(&slot->info, info);
      simple_mtx_unlock(&flips.lock);
      if (flip_reserved) {
         return set_error(error, AGC_DRIVER_ALREADY_RESERVED,
                          "the EOP flip token is already reserved for this flip");
      }
      if (token_reserved)
         return set_error(error, AGC_DRIVER_INVALID, "the EOP flip token is reserved for a different flip");
      return set_error(error, AGC_DRIVER_FAILED, "more than %d EOP flips are outstanding", AGC_EOP_FLIP_SLOTS);
   }
   slot->state = EOP_CLAIMED;
   simple_mtx_unlock(&flips.lock);

   struct reserved_flip flip;
   code = reserve_flip(info, true, &flip, error);
   if (code != AGC_DRIVER_OK) {
      eop_abandon(slot);
      return code;
   }

   simple_mtx_lock(&flips.lock);
   bool stopped = flips.stopped;
   bool failed = flips.failed;
   bool published = !stopped && !failed && flips.outputs[info->handle] == flip.output;
   if (published) {
      slot->state = EOP_READY;
      slot->token = token;
      slot->info = *info;
      slot->flip = flip;
      if (chosen && token >= flips.next_token)
         flips.next_token = token + 1;
   } else {
      slot->state = EOP_EMPTY;
   }
   simple_mtx_unlock(&flips.lock);
   if (published) {
      *reserved_token = token;
      return AGC_DRIVER_OK;
   }

   flip_drop(&flip);
   if (stopped)
      return set_error(error, AGC_DRIVER_SHUTDOWN, "the video output driver has stopped");
   if (failed)
      return set_error(error, AGC_DRIVER_FAILED, "%s", flips.failure);
   return set_error(error, AGC_DRIVER_INVALID, "video output %u is not registered", info->handle);
}

static bool
readies_reserved_flip(const struct agc_event *events, uint32_t count)
{
   bool reserved = false;
   simple_mtx_lock(&flips.lock);
   for (uint32_t i = 0; i < count && !reserved; i++)
      reserved = events[i].kind == AGC_EVENT_EOP && eop_reserved_locked(events[i].context);
   simple_mtx_unlock(&flips.lock);
   return reserved;
}

static int
wait_for_flip_rooms(const struct agc_event *events, uint32_t count, struct agc_driver_error *error)
{
   for (uint32_t i = 0; i < count; i++) {
      if (events[i].kind != AGC_EVENT_FLIP)
         continue;
      struct registered_output *output;
      int code = output_acquire(events[i].flip.handle, &output, error);
      if (code != AGC_DRIVER_OK)
         return code;
      code = output_wait_for_flip_room(output, error);
      output_release(output);
      if (code != AGC_DRIVER_OK)
         return code;
   }
   return AGC_DRIVER_OK;
}

static int
reserve_flips(const struct agc_event *events, uint32_t count, struct reserved_flip *pending,
              struct agc_driver_error *error)
{
   uint32_t reserved = 0;
   for (uint32_t i = 0; i < count; i++) {
      if (events[i].kind != AGC_EVENT_FLIP)
         continue;
      int code = reserve_flip(&events[i].flip, false, &pending[reserved], error);
      if (code != AGC_DRIVER_OK) {
         for (uint32_t dropped = 0; dropped < reserved; dropped++)
            flip_drop(&pending[dropped]);
         return code;
      }
      reserved++;
   }
   return AGC_DRIVER_OK;
}

static int
ready_events(const struct agc_event *events, uint32_t count, const struct reserved_flip *pending,
             uint32_t pending_count, struct agc_driver_error *error)
{
   uint32_t next = 0;
   int code = AGC_DRIVER_OK;
   for (uint32_t i = 0; i < count && code == AGC_DRIVER_OK; i++) {
      struct reserved_flip reserved;
      if (events[i].kind == AGC_EVENT_FLIP) {
         code = flip_ready(&pending[next++], error);
      } else if (!eop_take(events[i].context, &reserved)) {
         agc_events_interrupt(events[i].queue, events[i].context);
      } else if (events[i].queue != GRAPHICS_QUEUE) {
         flip_drop(&reserved);
         code = set_error(error, AGC_DRIVER_FAILED, "EOP flips on async compute queues are not implemented");
      } else {
         code = flip_ready(&reserved, error);
      }
   }
   while (next < pending_count)
      flip_drop(&pending[next++]);
   return code;
}

int
agc_flips_deliver(const struct agc_event *events, uint32_t count, struct agc_driver_error *error)
{
   if (count && !events)
      return set_error(error, AGC_DRIVER_INVALID, "null events");
   uint32_t flip_count = 0;
   for (uint32_t i = 0; i < count; i++) {
      if (events[i].kind == AGC_EVENT_FLIP)
         flip_count++;
      else if (events[i].kind != AGC_EVENT_EOP)
         return set_error(error, AGC_DRIVER_INVALID, "unknown event kind %u", events[i].kind);
   }

   if (!readies_reserved_flip(events, count)) {
      int code = wait_for_flip_rooms(events, count, error);
      if (code != AGC_DRIVER_OK)
         return code;
   }

   struct reserved_flip few[DELIVERY_INLINE_FLIPS];
   struct reserved_flip *pending = few;
   if (flip_count > DELIVERY_INLINE_FLIPS) {
      pending = malloc(flip_count * sizeof(*pending));
      if (!pending)
         return set_error(error, AGC_DRIVER_FAILED, "out of memory for %u flips", flip_count);
   }
   int code = reserve_flips(events, count, pending, error);
   if (code == AGC_DRIVER_OK)
      code = ready_events(events, count, pending, flip_count, error);
   if (pending != few)
      free(pending);
   return code;
}

int
agc_flips_rendering_wait(uint32_t handle, uint32_t index, struct agc_rendering_wait **wait,
                         struct agc_driver_error *error)
{
   if (!wait)
      return set_error(error, AGC_DRIVER_INVALID, "null rendering wait");
   struct registered_output *output;
   int code = output_acquire(handle, &output, error);
   if (code != AGC_DRIVER_OK)
      return code;

   struct agc_driver_error said = {0};
   struct agc_videoout_wait *captured = NULL;
   code = output->table.capture_rendering_wait(output->table.self, index, &captured, &said);
   if (code != AGC_DRIVER_OK) {
      output_release(output);
      return callback_error(error, code, "VideoOut's rendering wait", &said);
   }
   if (!captured) {
      output_release(output);
      return set_error(error, AGC_DRIVER_FAILED, "VideoOut returned a null rendering wait");
   }

   struct agc_rendering_wait *made = malloc(sizeof(*made));
   if (!made) {
      output->table.rendering_wait_release(captured);
      output_release(output);
      return set_error(error, AGC_DRIVER_FAILED, "out of memory for a rendering wait");
   }
   made->output = output;
   made->captured = captured;
   *wait = made;
   return AGC_DRIVER_OK;
}

int
agc_rendering_wait_for(struct agc_rendering_wait *wait, uint32_t milliseconds, struct agc_driver_error *error)
{
   if (!wait)
      return set_error(error, AGC_DRIVER_INVALID, "null rendering wait");
   struct agc_driver_error said = {0};
   int code = wait->output->table.rendering_wait_for(wait->captured, milliseconds, &said);
   if (code == 0 || code == 1)
      return code;
   return callback_error(error, code, "VideoOut's rendering wait", &said);
}

void
agc_rendering_wait_release(struct agc_rendering_wait *wait)
{
   if (!wait)
      return;
   wait->output->table.rendering_wait_release(wait->captured);
   output_release(wait->output);
   free(wait);
}

bool
agc_flips_describe_buffer(uint32_t handle, int32_t index, struct agc_display_buffer *buffer)
{
   struct registered_output *output;
   if (!buffer || output_acquire(handle, &output, NULL) != AGC_DRIVER_OK)
      return false;
   struct agc_display_buffer described = {0};
   bool registered = output->table.describe_buffer(output->table.self, index, &described);
   output_release(output);
   if (registered)
      *buffer = described;
   return registered;
}

void
agc_flips_fail(const char *message)
{
   struct registered_output *failing[AGC_VIDEO_OUTPUT_SLOTS];
   struct reserved_flip pending[AGC_EOP_FLIP_SLOTS];
   uint32_t failing_count = 0;
   if (!message)
      message = "(no message)";

   simple_mtx_lock(&flips.lock);
   bool first = !flips.failed;
   if (first) {
      snprintf(flips.failure, sizeof(flips.failure), "%s", message);
      p_atomic_set(&flips.failed, 1);
   }
   for (uint32_t handle = 0; handle < AGC_VIDEO_OUTPUT_SLOTS; handle++) {
      if (!flips.outputs[handle])
         continue;
      p_atomic_inc(&flips.outputs[handle]->references);
      failing[failing_count++] = flips.outputs[handle];
   }
   uint32_t pending_count = eop_take_all_locked(NULL, pending);
   simple_mtx_unlock(&flips.lock);

   for (uint32_t i = 0; i < failing_count; i++) {
      failing[i]->table.fail(failing[i]->table.self, message);
      output_release(failing[i]);
   }
   for (uint32_t i = 0; i < pending_count; i++) {
      pending[i].output->table.flip_fail(pending[i].request, message);
      flip_drop(&pending[i]);
   }
   if (first)
      fprintf(stderr, "[agc] driver failure: %s\n", message);
}

const char *
agc_flips_failure(void)
{
   return p_atomic_read(&flips.failed) ? flips.failure : NULL;
}

void
agc_flips_shutdown(void)
{
   struct reserved_flip cancelled[AGC_EOP_FLIP_SLOTS];
   simple_mtx_lock(&flips.lock);
   p_atomic_set(&flips.stopped, 1);
   uint32_t count = eop_take_all_locked(NULL, cancelled);
   simple_mtx_unlock(&flips.lock);
   for (uint32_t i = 0; i < count; i++)
      flip_drop(&cancelled[i]);
}

int
agc_events_add(int64_t eq, uint32_t gpu_queue, uint64_t ident, void *udata, struct agc_driver_error *error)
{
   if (eq == 0)
      return set_error(error, AGC_DRIVER_INVALID, "invalid event queue");
   const struct agc_anyps5 *host = agc_anyps5_get();
   if (!host || !host->add_data_event)
      return set_error(error, AGC_DRIVER_FAILED, "AnyPS5's event queues are not available");
   if (host->add_data_event(eq, (uintptr_t)ident, AGC_EVFILT_GRAPHICS_CORE, AGC_EV_ADD | AGC_EV_CLEAR,
                            (intptr_t)gpu_queue, udata) != 0)
      return set_error(error, AGC_DRIVER_INVALID, "the event queue rejected the event");

   struct event_registration *vacant = NULL;
   bool registered = false;
   simple_mtx_lock(&registrations.lock);
   for (uint32_t i = 0; i < AGC_EVENT_REGISTRATION_SLOTS && !registered; i++) {
      struct event_registration *slot = &registrations.slots[i];
      if (slot->used)
         registered = slot->eq == eq && slot->ident == ident;
      else if (!vacant)
         vacant = slot;
   }
   if (!registered && vacant) {
      *vacant = (struct event_registration){
         .used = true,
         .gpu_queue = gpu_queue,
         .eq = eq,
         .ident = ident,
      };
      registered = true;
   }
   simple_mtx_unlock(&registrations.lock);
   if (!registered) {
      return set_error(error, AGC_DRIVER_FAILED, "more than %d graphics events are registered",
                       AGC_EVENT_REGISTRATION_SLOTS);
   }
   return AGC_DRIVER_OK;
}

int
agc_events_delete(int64_t eq, uint64_t ident, struct agc_driver_error *error)
{
   if (eq == 0)
      return set_error(error, AGC_DRIVER_INVALID, "invalid event queue");
   const struct agc_anyps5 *host = agc_anyps5_get();
   if (!host || !host->delete_event)
      return set_error(error, AGC_DRIVER_FAILED, "AnyPS5's event queues are not available");
   if (host->delete_event(eq, (uintptr_t)ident, AGC_EVFILT_GRAPHICS_CORE) != 0)
      return set_error(error, AGC_DRIVER_INVALID, "the event is not registered");

   simple_mtx_lock(&registrations.lock);
   for (uint32_t i = 0; i < AGC_EVENT_REGISTRATION_SLOTS; i++) {
      struct event_registration *slot = &registrations.slots[i];
      if (slot->used && slot->eq == eq && slot->ident == ident)
         slot->used = false;
   }
   simple_mtx_unlock(&registrations.lock);
   return AGC_DRIVER_OK;
}

void
agc_events_interrupt(uint32_t gpu_queue, uint32_t context)
{
   struct event_target targets[AGC_EVENT_REGISTRATION_SLOTS];
   uint32_t count = 0;
   simple_mtx_lock(&registrations.lock);
   for (uint32_t i = 0; i < AGC_EVENT_REGISTRATION_SLOTS; i++) {
      const struct event_registration *slot = &registrations.slots[i];
      if (slot->used && slot->gpu_queue == gpu_queue)
         targets[count++] = (struct event_target){.eq = slot->eq, .ident = slot->ident};
   }
   simple_mtx_unlock(&registrations.lock);
   if (!count)
      return;

   const struct agc_anyps5 *host = agc_anyps5_get();
   if (!host || !host->trigger_event)
      return;
   for (uint32_t i = 0; i < count; i++) {
      uint64_t data = (uint64_t)(uint8_t)gpu_queue | (uint64_t)context << 16;
      host->trigger_event(targets[i].eq, (uintptr_t)targets[i].ident, AGC_EVFILT_GRAPHICS_CORE, &data);
   }
}

static bool
same_display(const struct agc_host_display_hdr *a, const struct agc_host_display_hdr *b)
{
   return a->hdr == b->hdr && a->max_luminance == b->max_luminance && a->min_luminance == b->min_luminance &&
          a->max_full_frame_luminance == b->max_full_frame_luminance;
}

static void
hdr_publish(const struct agc_host_display_hdr *display, const bool *surface_hdr10)
{
   simple_mtx_lock(&hdr.lock);
   struct agc_host_hdr_status *status = &hdr.status;
   struct agc_host_display_hdr shown = display ? *display : status->display;
   bool offered = surface_hdr10 ? *surface_hdr10 : status->surface_hdr10;
   bool display_changed = !same_display(&status->display, &shown);
   bool available = shown.hdr && offered && !getenv("APS5_FORCE_SDR");
   bool availability_changed = available != status->available;
   status->display.hdr = shown.hdr;
   status->display.max_luminance = shown.max_luminance;
   status->display.min_luminance = shown.min_luminance;
   status->display.max_full_frame_luminance = shown.max_full_frame_luminance;
   status->surface_hdr10 = offered;
   status->available = available;
   if (display_changed)
      status->display_generation++;
   if (availability_changed || (available && display_changed))
      status->generation++;
   simple_mtx_unlock(&hdr.lock);
}

static bool
guest_region_query(uintptr_t address, bool writable, struct guest_region *region)
{
#ifdef _WIN32
   MEMORY_BASIC_INFORMATION memory;
   if (VirtualQuery((const void *)address, &memory, sizeof(memory)) != sizeof(memory))
      return false;
   DWORD protect = memory.Protect;
   bool committed = memory.State == MEM_COMMIT && !(protect & (PAGE_GUARD | PAGE_NOACCESS));
   bool can_write =
      (protect & (PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;
   bool can_read = can_write || (protect & (PAGE_READONLY | PAGE_EXECUTE_READ)) != 0;
   region->base = (uintptr_t)memory.BaseAddress;
   region->end = region->base + memory.RegionSize;
   region->writable = can_write;
   return committed && can_read && (can_write || !writable) && region->end > address;
#else
   (void)address;
   (void)writable;
   (void)region;
   return false;
#endif
}

static void
guest_region_keep(const struct guest_region *region)
{
   uint32_t slot = guest_memory.count;
   for (uint32_t i = 0; i < guest_memory.count; i++) {
      if (guest_memory.regions[i].base == region->base)
         slot = i;
   }
   if (slot == GUEST_REGION_SLOTS)
      slot = guest_memory.oldest++ % GUEST_REGION_SLOTS;
   else if (slot == guest_memory.count)
      guest_memory.count++;
   guest_memory.regions[slot] = *region;
}

static bool
guest_region_end(uintptr_t address, bool writable, uintptr_t *end)
{
   for (uint32_t i = 0; i < guest_memory.count; i++) {
      const struct guest_region *known = &guest_memory.regions[i];
      if (known->base <= address && address < known->end && (known->writable || !writable)) {
         *end = known->end;
         return true;
      }
   }
   struct guest_region queried;
   if (!guest_region_query(address, writable, &queried))
      return false;
   guest_region_keep(&queried);
   *end = queried.end;
   return true;
}

static bool
guest_accessible(uintptr_t address, size_t bytes, bool writable, uint64_t generation)
{
   uintptr_t end = address + bytes;
   if (end < address)
      return false;
   bool accessible = true;
   simple_mtx_lock(&guest_memory.lock);
   if (guest_memory.generation != generation) {
      guest_memory.generation = generation;
      guest_memory.count = 0;
      guest_memory.oldest = 0;
   }
   for (uintptr_t cursor = address; accessible && cursor < end;)
      accessible = guest_region_end(cursor, writable, &cursor);
   simple_mtx_unlock(&guest_memory.lock);
   return accessible;
}

static uint32_t
round_extent(double value)
{
   double rounded = round(value);
   if (!(rounded >= 1.0))
      return 1;
   if (rounded >= (double)UINT32_MAX)
      return UINT32_MAX;
   return (uint32_t)rounded;
}

static bool
aspect_source_valid(uint32_t source_width, uint32_t source_height)
{
   if (source_width && source_height)
      return true;
   agc_report("AspectFit", "source extent must be non-zero");
   return false;
}

static uint32_t
aspect_extent(uint32_t length, uint32_t numerator, uint32_t denominator)
{
   if (!aspect_source_valid(numerator, denominator))
      return 1;
   return round_extent((double)length * (double)numerator / (double)denominator);
}

static int
presenter_missing(struct agc_driver_error *error)
{
   return set_error(error, AGC_DRIVER_FAILED, "no presenter is installed");
}

static int
presenter_result(struct agc_driver_error *error, int code, const struct agc_driver_error *said)
{
   if (code == AGC_DRIVER_OK)
      return AGC_DRIVER_OK;
   return set_error(error, code, "%.*s", (int)sizeof(said->message), said->message);
}

static int
present(const struct agc_presentation_window *window, const struct agc_display_buffer *buffer,
        const struct agc_display_buffer *plane, void (*gpu_ready)(void *), void *context,
        struct agc_driver_error *error)
{
   if (!gpu_ready || !context)
      return set_error(error, AGC_DRIVER_INVALID, "missing GPU completion callback");
   const char *failure = agc_flips_failure();
   if (failure)
      return set_error(error, AGC_DRIVER_FAILED, "%s", failure);
   const struct agc_presenter *presenter = p_atomic_read(&installed_presenter);
   if (!presenter || !presenter->present)
      return presenter_missing(error);

   struct agc_driver_error said = {0};
   bool presented = false;
   int code = presenter->present(window, buffer, plane, &presented, &said);
   if (code != AGC_DRIVER_OK)
      return presenter_result(error, code, &said);
   if (presented)
      p_atomic_inc(&frames_presented);
   gpu_ready(context);
   return AGC_DRIVER_OK;
}

void
agc_videoout_set_presenter(const struct agc_presenter *presenter)
{
   p_atomic_set(&installed_presenter, presenter);
}

void
agc_videoout_set_core(const struct agc_videoout_core *core)
{
   p_atomic_set(&installed_core, core);
}

uint64_t
agc_videoout_frames_presented(void)
{
   return p_atomic_read(&frames_presented);
}

void
agc_videoout_reset(void)
{
   struct registered_output *registered[AGC_VIDEO_OUTPUT_SLOTS];
   struct reserved_flip cancelled[AGC_EOP_FLIP_SLOTS];
   simple_mtx_lock(&flips.lock);
   uint32_t cancelled_count = eop_take_all_locked(NULL, cancelled);
   memcpy(registered, flips.outputs, sizeof(registered));
   memset(flips.outputs, 0, sizeof(flips.outputs));
   memset(flips.eop, 0, sizeof(flips.eop));
   flips.next_token = AGC_DRIVER_EOP_FLIP_TOKEN_BASE;
   flips.frames = 0;
   p_atomic_set(&flips.failed, 0);
   p_atomic_set(&flips.stopped, 0);
   flips.failure[0] = 0;
   simple_mtx_unlock(&flips.lock);
   for (uint32_t i = 0; i < cancelled_count; i++)
      flip_drop(&cancelled[i]);
   for (uint32_t handle = 0; handle < AGC_VIDEO_OUTPUT_SLOTS; handle++) {
      if (registered[handle])
         output_release(registered[handle]);
   }

   simple_mtx_lock(&registrations.lock);
   memset(registrations.slots, 0, sizeof(registrations.slots));
   simple_mtx_unlock(&registrations.lock);

   simple_mtx_lock(&hdr.lock);
   memset(&hdr.status, 0, sizeof(hdr.status));
   simple_mtx_unlock(&hdr.lock);

   simple_mtx_lock(&guest_memory.lock);
   guest_memory.generation = 0;
   guest_memory.count = 0;
   guest_memory.oldest = 0;
   simple_mtx_unlock(&guest_memory.lock);

   p_atomic_set(&installed_presenter, NULL);
   p_atomic_set(&installed_core, NULL);
   p_atomic_set(&frames_presented, 0);
}

int
AgcDriverRegisterVideoOutput(uint32_t handle, const struct agc_video_output *output,
                             struct agc_driver_error *error)
{
   if (!output)
      return set_error(error, AGC_DRIVER_INVALID, "null video output");
   int code = check_running(error);
   if (code != AGC_DRIVER_OK)
      return code;
   if (!table_complete(output))
      return set_error(error, AGC_DRIVER_INVALID, "the video output's table is incomplete");
   if (handle >= AGC_VIDEO_OUTPUT_SLOTS)
      return set_error(error, AGC_DRIVER_INVALID, "video output handle %u is out of range", handle);

   struct registered_output *registered = malloc(sizeof(*registered));
   if (!registered)
      return set_error(error, AGC_DRIVER_FAILED, "out of memory for video output %u", handle);
   registered->table = *output;
   registered->references = 1;
   simple_mtx_lock(&flips.lock);
   code = check_running(error);
   bool vacant = code == AGC_DRIVER_OK && !flips.outputs[handle];
   if (vacant)
      flips.outputs[handle] = registered;
   simple_mtx_unlock(&flips.lock);
   if (vacant)
      return AGC_DRIVER_OK;
   free(registered);
   if (code != AGC_DRIVER_OK)
      return code;
   return set_error(error, AGC_DRIVER_INVALID, "video output %u is already registered", handle);
}

int
AgcDriverUnregisterVideoOutput(uint32_t handle, void *self, struct agc_driver_error *error)
{
   if (handle >= AGC_VIDEO_OUTPUT_SLOTS)
      return set_error(error, AGC_DRIVER_INVALID, "video output handle %u is out of range", handle);

   struct reserved_flip cancelled[AGC_EOP_FLIP_SLOTS];
   uint32_t cancelled_count = 0;
   simple_mtx_lock(&flips.lock);
   struct registered_output *output = flips.outputs[handle];
   bool registered = output && output->table.self == self;
   if (registered) {
      flips.outputs[handle] = NULL;
      cancelled_count = eop_take_all_locked(output, cancelled);
   }
   simple_mtx_unlock(&flips.lock);
   if (!registered)
      return set_error(error, AGC_DRIVER_INVALID, "video output %u is registered for another output", handle);

   for (uint32_t i = 0; i < cancelled_count; i++)
      flip_drop(&cancelled[i]);
   output_release(output);
   return AGC_DRIVER_OK;
}

int
AgcDriverSubmitFlip(const struct agc_flip_info *info, struct agc_driver_error *error)
{
   if (!info)
      return set_error(error, AGC_DRIVER_INVALID, "null flip");
   struct reserved_flip flip;
   int code = output_acquire(info->handle, &flip.output, error);
   if (code != AGC_DRIVER_OK)
      return code;
   code = output_wait_for_flip_room(flip.output, error);
   if (code == AGC_DRIVER_OK)
      code = output_reserve(flip.output, info, false, &flip.request, error);
   if (code != AGC_DRIVER_OK) {
      output_release(flip.output);
      return code;
   }
   return flip_ready(&flip, error);
}

int
AgcDriverReserveEopFlip(const struct agc_flip_info *info, uint32_t *token, struct agc_driver_error *error)
{
   if (!info)
      return set_error(error, AGC_DRIVER_INVALID, "null flip");
   if (!token)
      return set_error(error, AGC_DRIVER_INVALID, "null token");
   const struct agc_flip_info flip = *info;
   return reserve_eop_flip(&flip, NULL, token, error);
}

int
AgcDriverReserveEopFlipAt(const struct agc_flip_info *info, uint32_t token, struct agc_driver_error *error)
{
   if (!info)
      return set_error(error, AGC_DRIVER_INVALID, "null flip");
   const struct agc_flip_info flip = *info;
   uint32_t reserved;
   return reserve_eop_flip(&flip, &token, &reserved, error);
}

uint32_t
AgcDriverPresentationApi(void)
{
   return AGC_DRIVER_PRESENTATION_NATIVE_WINDOW;
}

int
AgcDriverAttachWindow(const struct agc_presentation_window *window, struct agc_driver_error *error)
{
   if (!window)
      return set_error(error, AGC_DRIVER_INVALID, "null window");
   const struct agc_presenter *presenter = p_atomic_read(&installed_presenter);
   if (!presenter || !presenter->attach)
      return presenter_missing(error);
   struct agc_driver_error said = {0};
   int code = presenter->attach(window, &said);
   if (code != AGC_DRIVER_OK)
      return presenter_result(error, code, &said);
   AgcDriverSetSurfaceHdr10(false);
   return AGC_DRIVER_OK;
}

void
AgcDriverReleaseWindow(void *context)
{
   const struct agc_presenter *presenter = p_atomic_read(&installed_presenter);
   if (!presenter || !presenter->release)
      agc_report("AgcDriverReleaseWindow", "no presenter is installed");
   else if (!presenter->release(context))
      agc_report("AgcDriverReleaseWindow", "the window belongs to another thread, or is not attached");
}

int
AgcDriverCaptureOverlay(const struct agc_display_buffer *buffer, struct agc_driver_error *error)
{
   const struct agc_presenter *presenter = p_atomic_read(&installed_presenter);
   if (!presenter || !presenter->capture_overlay)
      return presenter_missing(error);
   struct agc_driver_error said = {0};
   return presenter_result(error, presenter->capture_overlay(buffer, &said), &said);
}

int
AgcDriverPresentBuffer(const struct agc_presentation_window *window, const struct agc_display_buffer *buffer,
                       const struct agc_display_buffer *plane, void (*gpu_ready)(void *), void *context,
                       struct agc_driver_error *error)
{
   if (!window)
      return set_error(error, AGC_DRIVER_INVALID, "null window");
   if (!buffer)
      return set_error(error, AGC_DRIVER_INVALID, "null buffer");
   return present(window, buffer, plane, gpu_ready, context, error);
}

int
AgcDriverPresentClear(const struct agc_presentation_window *window, bool opaque,
                      const struct agc_display_buffer *plane, void (*gpu_ready)(void *), void *context,
                      struct agc_driver_error *error)
{
   (void)opaque;
   if (!window)
      return set_error(error, AGC_DRIVER_INVALID, "null window");
   return present(window, NULL, plane, gpu_ready, context, error);
}

size_t
AgcDriverDisplayBufferSize(const struct agc_display_buffer *buffer)
{
   const struct agc_presenter *presenter = p_atomic_read(&installed_presenter);
   if (!buffer || !presenter || !presenter->display_buffer_size)
      return 0;
   return presenter->display_buffer_size(buffer);
}

void
AgcDriverSetHostDisplayHdr(const struct agc_host_display_hdr *display)
{
   if (display)
      hdr_publish(display, NULL);
}

void
AgcDriverSetSurfaceHdr10(bool offered)
{
   hdr_publish(NULL, &offered);
}

void
AgcDriverNoteReportedToneMapPeak(float nits)
{
   simple_mtx_lock(&hdr.lock);
   hdr.status.reported_tone_map_peak = nits;
   simple_mtx_unlock(&hdr.lock);
}

void
AgcDriverHostHdr(struct agc_host_hdr_status *status)
{
   if (!status)
      return;
   simple_mtx_lock(&hdr.lock);
   *status = hdr.status;
   simple_mtx_unlock(&hdr.lock);
}

bool
AgcDriverPresentingHdr10(void)
{
   return false;
}

void
AgcDriverSetSystemOverlay(const struct ImeKeyboardView *view)
{
   (void)view;
}

void
AgcDriverSetSystemDialog(const struct ModalDialogView *view)
{
   (void)view;
}

void
AgcDriverPadStatusPoll(const struct AgcDriverPadStatusProvider *provider)
{
   (void)provider;
}

void
AgcDriverPadStatusShow(void)
{
}

void
AgcDriverTogglePerformanceOverlay(void)
{
}

void
AgcDriverRequestScreenshot(void)
{
}

int
AgcDriverCheckGuestMemory(const void *pointer, size_t bytes, size_t alignment, bool writable,
                          struct agc_driver_error *error)
{
   uintptr_t address = (uintptr_t)pointer;
   if (!address || !alignment || address % alignment)
      return set_error(error, AGC_DRIVER_INVALID, "null or misaligned guest address 0x%" PRIxPTR, address);
   const struct agc_anyps5 *host = agc_anyps5_get();
   if (!host || !host->generation)
      return set_error(error, AGC_DRIVER_FAILED, "AnyPS5's guest memory is not available");
   if (!guest_accessible(address, bytes, writable, host->generation())) {
      return set_error(error, AGC_DRIVER_INVALID,
                       "guest memory 0x%" PRIxPTR "+0x%zx is not mapped with the access asked for", address,
                       bytes);
   }
   return AGC_DRIVER_OK;
}

void
AgcDriverStoreGuestLabel(uint64_t *label, uint64_t value)
{
   if (!label || (uintptr_t)label % sizeof(*label)) {
      agc_report("AgcDriverStoreGuestLabel", "null or misaligned label");
      return;
   }
   p_atomic_set(label, value);
}

void
AgcDriverWaitIdle(void)
{
   const struct agc_videoout_core *core = p_atomic_read(&installed_core);
   if (core && core->wait_idle)
      core->wait_idle();
}

void
AgcDriverRaiseWorkerThreadPriority(const char *role)
{
   (void)role;
}

void
AgcDriverShutdown(void)
{
   const struct agc_videoout_core *core = p_atomic_read(&installed_core);
   if (core && core->shutdown)
      core->shutdown();
   agc_flips_shutdown();
}

void
AgcDriverLockVulkanLoader(void)
{
}

void
AgcDriverUnlockVulkanLoader(void)
{
}

bool
AgcDriverTraceRecording(void)
{
   return false;
}

void *
AgcDriverTraceThread(void)
{
   return NULL;
}

void
AgcDriverTraceCount(const char *scope)
{
   (void)scope;
}

double
ComputeContainScale(uint32_t source_width, uint32_t source_height, uint32_t bounds_width, uint32_t bounds_height,
                    bool allow_upscale)
{
   if (!aspect_source_valid(source_width, source_height) || !bounds_width || !bounds_height)
      return 1.0;
   double horizontal = (double)bounds_width / (double)source_width;
   double vertical = (double)bounds_height / (double)source_height;
   double scale = horizontal < vertical ? horizontal : vertical;
   return allow_upscale || scale < 1.0 ? scale : 1.0;
}

struct agc_aspect_fit_size
ComputeContainSize(uint32_t source_width, uint32_t source_height, uint32_t bounds_width, uint32_t bounds_height,
                   bool allow_upscale)
{
   double scale = ComputeContainScale(source_width, source_height, bounds_width, bounds_height, allow_upscale);
   return (struct agc_aspect_fit_size){
      .width = round_extent((double)source_width * scale),
      .height = round_extent((double)source_height * scale),
   };
}

struct agc_aspect_fit_rect
ComputeContainRect(uint32_t source_width, uint32_t source_height, uint32_t destination_width,
                   uint32_t destination_height)
{
   struct agc_aspect_fit_size size =
      ComputeContainSize(source_width, source_height, destination_width, destination_height, true);
   return (struct agc_aspect_fit_rect){
      .x = (int32_t)(((int64_t)destination_width - (int64_t)size.width) / 2),
      .y = (int32_t)(((int64_t)destination_height - (int64_t)size.height) / 2),
      .width = size.width,
      .height = size.height,
   };
}

uint32_t
ComputeWidthForHeight(uint32_t source_width, uint32_t source_height, uint32_t height)
{
   return aspect_extent(height, source_width, source_height);
}

uint32_t
ComputeHeightForWidth(uint32_t source_width, uint32_t source_height, uint32_t width)
{
   return aspect_extent(width, source_height, source_width);
}
