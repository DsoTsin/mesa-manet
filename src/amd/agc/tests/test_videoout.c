#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "c11/threads.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"

#include "agc_anyps5.h"
#include "agc_debug.h"
#include "agc_videoout.h"
#include "test.h"

#ifdef _WIN32
#include <windows.h>
#endif

#define TOKEN_BASE AGC_DRIVER_EOP_FLIP_TOKEN_BASE
#define TOKEN_LIMIT AGC_DRIVER_EOP_FLIP_TOKEN_LIMIT
#define CALL_LOG_MAX 4096
#define OUTPUTS 4
#define REQUESTS 32
#define WAITS 8
#define STRESS_ROUNDS 20000

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CALLED(at, kind, who, what, detail) CHECK(called(at, kind, who, what, detail))
#define EXPECT(code, expected, text) \
   do { \
      CHECK_EQ(code, expected); \
      CHECK(said(&error, text)); \
   } while (0)

enum call_kind {
   CALL_ROOM,
   CALL_RESERVE,
   CALL_CAPTURE,
   CALL_DESCRIBE,
   CALL_FAIL,
   CALL_RELEASE,
   CALL_FLIP_READY,
   CALL_FLIP_FAIL,
   CALL_FLIP_RELEASE,
   CALL_WAIT_FOR,
   CALL_WAIT_RELEASE,
   CALL_EVENT_ADD,
   CALL_EVENT_TRIGGER,
   CALL_EVENT_DELETE,
   CALL_ATTACH,
   CALL_DETACH,
   CALL_OVERLAY,
   CALL_PRESENT,
   CALL_GPU_READY,
   CALL_BUFFER_SIZE,
   CALL_WAIT_IDLE,
   CALL_CORE_SHUTDOWN
};

struct call {
   enum call_kind kind;
   uint32_t who;
   int64_t what;
   uint64_t detail;
};

struct gate {
   int entered;
   int open;
};

struct test_output;

struct test_request {
   struct test_output *output;
   struct agc_flip_info info;
   bool limited;
   bool readied;
   uint32_t fails;
   uint32_t releases;
   uint64_t frame;
   char failure[64];
};

struct test_wait {
   struct test_output *output;
   uint32_t index;
   uint32_t releases;
};

struct test_output {
   uint32_t id;
   uint32_t capacity;
   uint32_t queued;
   int room_code;
   int reserve_code;
   int ready_code;
   int capture_code;
   int wait_code;
   bool null_reservation;
   bool null_wait;
   bool describes;
   uint32_t releases;
   uint32_t fails;
   char failure[64];
   uint32_t request_count;
   struct test_request requests[REQUESTS];
   uint32_t wait_count;
   struct test_wait waits[WAITS];
   struct gate *room_gate;
   struct gate *reserve_gate;
   void (*in_room)(struct test_output *output);
};

static struct call call_log[CALL_LOG_MAX];
static uint32_t call_count;
static simple_mtx_t call_lock = SIMPLE_MTX_INITIALIZER;
static struct test_output outputs[OUTPUTS];

static struct {
   int32_t add_code;
   int32_t delete_code;
   uint64_t generation;
   int16_t add_filter;
   uint16_t add_flags;
   void *add_udata;
   int16_t trigger_filter;
   int16_t delete_filter;
} libkernel;

static struct {
   int attach_code;
   int overlay_code;
   int present_code;
   bool shows;
   bool detaches;
   size_t buffer_bytes;
   const struct agc_presentation_window *window;
   const struct agc_display_buffer *buffer;
   const struct agc_display_buffer *plane;
   const struct agc_display_buffer *overlay;
} presenting;

static struct {
   int code;
   uint32_t releases_then;
} closed_in_room;

static struct {
   int reserved;
   int released;
   int registered;
   int unregistered;
   int unexpected;
   int stop;
} stress;

static char stress_identities[STRESS_ROUNDS];

static void
note(enum call_kind kind, uint32_t who, int64_t what, uint64_t detail)
{
   simple_mtx_lock(&call_lock);
   if (call_count < CALL_LOG_MAX)
      call_log[call_count] = (struct call){.kind = kind, .who = who, .what = what, .detail = detail};
   call_count++;
   simple_mtx_unlock(&call_lock);
}

static bool
called(uint32_t at, enum call_kind kind, uint32_t who, int64_t what, uint64_t detail)
{
   if (at >= call_count || at >= CALL_LOG_MAX) {
      fprintf(stderr, "   call %u was not made (%u calls)\n", at, call_count);
      return false;
   }
   const struct call *made = &call_log[at];
   if (made->kind == kind && made->who == who && made->what == what && made->detail == detail)
      return true;
   fprintf(stderr, "   call %u is kind %d who %u what %lld detail %#llx\n", at, (int)made->kind, made->who,
           (long long)made->what, (unsigned long long)made->detail);
   return false;
}

static uint32_t
calls_of(enum call_kind kind)
{
   uint32_t count = 0;
   for (uint32_t i = 0; i < call_count && i < CALL_LOG_MAX; i++)
      count += call_log[i].kind == kind;
   return count;
}

static bool
said(const struct agc_driver_error *error, const char *text)
{
   if (strcmp(error->message, text) == 0)
      return true;
   fprintf(stderr, "   said \"%s\"\n", error->message);
   return false;
}

static int
refuse(struct agc_driver_error *error, const char *what, int code)
{
   snprintf(error->message, sizeof(error->message), "%s refused %d", what, code);
   return code;
}

static void
gate_hold(struct gate *gate)
{
   p_atomic_set(&gate->entered, 1);
   while (!p_atomic_read(&gate->open))
      thrd_yield();
}

static void
gate_await(struct gate *gate)
{
   while (!p_atomic_read(&gate->entered))
      thrd_yield();
}

static int
out_reserve(void *self, const struct agc_flip_info *info, bool limited, struct agc_videoout_request **request,
            struct agc_driver_error *error)
{
   struct test_output *output = self;
   note(CALL_RESERVE, output->id, info->argument, limited);
   if (output->reserve_gate)
      gate_hold(output->reserve_gate);
   if (output->reserve_code != AGC_DRIVER_OK)
      return refuse(error, "reserve", output->reserve_code);
   if (limited && output->queued >= output->capacity)
      return AGC_DRIVER_FULL;
   if (output->null_reservation)
      return AGC_DRIVER_OK;
   struct test_request *made = &output->requests[output->request_count++];
   *made = (struct test_request){.output = output, .info = *info, .limited = limited};
   output->queued++;
   *request = (struct agc_videoout_request *)made;
   return AGC_DRIVER_OK;
}

static int
out_wait_for_flip_room(void *self, struct agc_driver_error *error)
{
   struct test_output *output = self;
   note(CALL_ROOM, output->id, 0, 0);
   if (output->room_gate)
      gate_hold(output->room_gate);
   if (output->in_room)
      output->in_room(output);
   if (output->room_code != AGC_DRIVER_OK)
      return refuse(error, "room", output->room_code);
   return AGC_DRIVER_OK;
}

static int
out_capture_rendering_wait(void *self, uint32_t index, struct agc_videoout_wait **wait,
                           struct agc_driver_error *error)
{
   struct test_output *output = self;
   note(CALL_CAPTURE, output->id, index, 0);
   if (output->capture_code != AGC_DRIVER_OK)
      return refuse(error, "capture", output->capture_code);
   if (output->null_wait)
      return AGC_DRIVER_OK;
   struct test_wait *made = &output->waits[output->wait_count++];
   *made = (struct test_wait){.output = output, .index = index};
   *wait = (struct agc_videoout_wait *)made;
   return AGC_DRIVER_OK;
}

static bool
out_describe_buffer(void *self, int32_t index, struct agc_display_buffer *buffer)
{
   struct test_output *output = self;
   note(CALL_DESCRIBE, output->id, index, 0);
   *buffer = (struct agc_display_buffer){
      .address = UINT64_C(0x100000000) + (uint32_t)index,
      .pixel_format = 0x8100,
      .width = 3840,
      .height = 2160,
      .tiling_mode = 2,
      .pitch_in_pixel = 3840,
      .dcc_address = UINT64_C(0x200000000),
      .dcc_clear_color = 7,
      .alignment = 0x10000,
   };
   return output->describes;
}

static void
out_fail(void *self, const char *message)
{
   struct test_output *output = self;
   note(CALL_FAIL, output->id, 0, 0);
   output->fails++;
   snprintf(output->failure, sizeof(output->failure), "%s", message);
}

static void
out_release(void *self)
{
   struct test_output *output = self;
   note(CALL_RELEASE, output->id, 0, 0);
   output->releases++;
}

static int
out_flip_ready(struct agc_videoout_request *request, uint64_t frame, struct agc_driver_error *error)
{
   struct test_request *made = (struct test_request *)request;
   note(CALL_FLIP_READY, made->output->id, made->info.argument, frame);
   made->readied = true;
   made->frame = frame;
   if (made->output->ready_code != AGC_DRIVER_OK)
      return refuse(error, "ready", made->output->ready_code);
   return AGC_DRIVER_OK;
}

static void
out_flip_fail(struct agc_videoout_request *request, const char *message)
{
   struct test_request *made = (struct test_request *)request;
   note(CALL_FLIP_FAIL, made->output->id, made->info.argument, 0);
   made->fails++;
   snprintf(made->failure, sizeof(made->failure), "%s", message);
}

static void
out_flip_release(struct agc_videoout_request *request)
{
   struct test_request *made = (struct test_request *)request;
   note(CALL_FLIP_RELEASE, made->output->id, made->info.argument, 0);
   made->releases++;
   if (!made->readied)
      made->output->queued--;
}

static int
out_rendering_wait_for(struct agc_videoout_wait *wait, uint32_t milliseconds, struct agc_driver_error *error)
{
   struct test_wait *made = (struct test_wait *)wait;
   note(CALL_WAIT_FOR, made->output->id, made->index, milliseconds);
   if (made->output->wait_code < 0 || made->output->wait_code > 1)
      return refuse(error, "wait", made->output->wait_code);
   return made->output->wait_code;
}

static void
out_rendering_wait_release(struct agc_videoout_wait *wait)
{
   struct test_wait *made = (struct test_wait *)wait;
   note(CALL_WAIT_RELEASE, made->output->id, made->index, 0);
   made->releases++;
}

static struct agc_video_output
table_of(struct test_output *output)
{
   return (struct agc_video_output){
      .self = output,
      .reserve = out_reserve,
      .wait_for_flip_room = out_wait_for_flip_room,
      .capture_rendering_wait = out_capture_rendering_wait,
      .describe_buffer = out_describe_buffer,
      .fail = out_fail,
      .release = out_release,
      .flip_ready = out_flip_ready,
      .flip_fail = out_flip_fail,
      .flip_release = out_flip_release,
      .rendering_wait_for = out_rendering_wait_for,
      .rendering_wait_release = out_rendering_wait_release,
   };
}

static void
open_output(uint32_t handle, struct test_output *output)
{
   struct agc_video_output table = table_of(output);
   CHECK_EQ(AgcDriverRegisterVideoOutput(handle, &table, NULL), AGC_DRIVER_OK);
}

static struct agc_flip_info
flip_of(uint32_t handle, int64_t argument)
{
   return (struct agc_flip_info){.handle = handle, .index = 0, .mode = 1, .argument = argument};
}

static struct agc_event
flip_event(uint32_t handle, int64_t argument)
{
   return (struct agc_event){.kind = AGC_EVENT_FLIP, .flip = flip_of(handle, argument)};
}

static struct agc_event
eop_event(uint32_t queue, uint32_t context)
{
   return (struct agc_event){.kind = AGC_EVENT_EOP, .queue = queue, .context = context};
}

static int32_t AGC_SYSV
libkernel_add_data_event(int64_t queue, uintptr_t ident, int16_t filter, uint16_t flags, intptr_t data,
                         void *udata)
{
   libkernel.add_filter = filter;
   libkernel.add_flags = flags;
   libkernel.add_udata = udata;
   note(CALL_EVENT_ADD, (uint32_t)ident, queue, (uint64_t)data);
   return libkernel.add_code;
}

static int32_t AGC_SYSV
libkernel_trigger_event(int64_t queue, uintptr_t ident, int16_t filter, void *data)
{
   uint64_t value;
   memcpy(&value, data, sizeof(value));
   libkernel.trigger_filter = filter;
   note(CALL_EVENT_TRIGGER, (uint32_t)ident, queue, value);
   return 0;
}

static int32_t AGC_SYSV
libkernel_delete_event(int64_t queue, uintptr_t ident, int16_t filter)
{
   libkernel.delete_filter = filter;
   note(CALL_EVENT_DELETE, (uint32_t)ident, queue, 0);
   return libkernel.delete_code;
}

static uint64_t
libc_generation(void)
{
   return libkernel.generation;
}

static const struct agc_anyps5 anyps5 = {
   .generation = libc_generation,
   .add_data_event = libkernel_add_data_event,
   .trigger_event = libkernel_trigger_event,
   .delete_event = libkernel_delete_event,
};

static int
recorded_attach(const struct agc_presentation_window *window, struct agc_driver_error *error)
{
   note(CALL_ATTACH, 0, (int64_t)window->native_window, 0);
   if (presenting.attach_code != AGC_DRIVER_OK)
      return refuse(error, "attach", presenting.attach_code);
   return AGC_DRIVER_OK;
}

static bool
recorded_detach(void *context)
{
   note(CALL_DETACH, 0, (int64_t)(intptr_t)context, 0);
   return presenting.detaches;
}

static int
recorded_overlay(const struct agc_display_buffer *buffer, struct agc_driver_error *error)
{
   note(CALL_OVERLAY, 0, 0, 0);
   presenting.overlay = buffer;
   if (presenting.overlay_code != AGC_DRIVER_OK)
      return refuse(error, "overlay", presenting.overlay_code);
   return AGC_DRIVER_OK;
}

static int
recorded_present(const struct agc_presentation_window *window, const struct agc_display_buffer *buffer,
                 const struct agc_display_buffer *plane, bool *presented, struct agc_driver_error *error)
{
   note(CALL_PRESENT, 0, 0, 0);
   presenting.window = window;
   presenting.buffer = buffer;
   presenting.plane = plane;
   if (presenting.present_code != AGC_DRIVER_OK)
      return refuse(error, "present", presenting.present_code);
   *presented = presenting.shows;
   return AGC_DRIVER_OK;
}

static size_t
recorded_buffer_size(const struct agc_display_buffer *buffer)
{
   note(CALL_BUFFER_SIZE, 0, (int64_t)buffer->width, buffer->height);
   return presenting.buffer_bytes;
}

static const struct agc_presenter recording_presenter = {
   .attach = recorded_attach,
   .release = recorded_detach,
   .capture_overlay = recorded_overlay,
   .present = recorded_present,
   .display_buffer_size = recorded_buffer_size,
};

static void
recorded_wait_idle(void)
{
   note(CALL_WAIT_IDLE, 0, 0, 0);
}

static void
recorded_core_shutdown(void)
{
   note(CALL_CORE_SHUTDOWN, 0, 0, 0);
}

static const struct agc_videoout_core recording_core = {
   .wait_idle = recorded_wait_idle,
   .shutdown = recorded_core_shutdown,
};

static void
gpu_ready(void *context)
{
   note(CALL_GPU_READY, 0, *(const int *)context, 0);
}

static void
force_sdr(bool forced)
{
#ifdef _WIN32
   _putenv_s("APS5_FORCE_SDR", forced ? "1" : "");
#else
   if (forced)
      setenv("APS5_FORCE_SDR", "1", 1);
   else
      unsetenv("APS5_FORCE_SDR");
#endif
}

static void
fresh(void)
{
   agc_videoout_reset();
   memset(outputs, 0, sizeof(outputs));
   for (uint32_t i = 0; i < OUTPUTS; i++) {
      outputs[i].id = i + 1;
      outputs[i].capacity = 16;
      outputs[i].wait_code = 1;
      outputs[i].describes = true;
   }
   memset(&libkernel, 0, sizeof(libkernel));
   memset(&presenting, 0, sizeof(presenting));
   call_count = 0;
   agc_anyps5_set(&anyps5);
   force_sdr(false);
}

static void
close_output(uint32_t handle, struct test_output *output)
{
   CHECK_EQ(output->releases, 0);
   CHECK_EQ(AgcDriverUnregisterVideoOutput(handle, output, NULL), AGC_DRIVER_OK);
   CHECK_EQ(output->releases, 1);
}

static void
use_unrelated_state(void)
{
   struct agc_host_hdr_status status;
   struct agc_display_buffer buffer;
   CHECK_EQ(agc_events_add(0x77, 0x33, 0x33, NULL, NULL), AGC_DRIVER_OK);
   agc_events_interrupt(0x33, 1);
   CHECK_EQ(agc_events_delete(0x77, 0x33, NULL), AGC_DRIVER_OK);
   AgcDriverSetSurfaceHdr10(false);
   AgcDriverHostHdr(&status);
   CHECK(!agc_flips_describe_buffer(AGC_VIDEO_OUTPUT_SLOTS - 1, 0, &buffer));
}

static void
test_layout(void)
{
   CHECK_EQ(sizeof(struct agc_driver_error), 256);
   CHECK_EQ(sizeof(struct agc_flip_info), 24);
   CHECK_EQ(sizeof(struct agc_display_buffer), 56);
   CHECK_EQ(sizeof(struct agc_video_output), 12 * sizeof(void *));
   CHECK_EQ(sizeof(struct agc_presentation_window), 32);
   CHECK_EQ(sizeof(struct agc_host_display_hdr), 16);
   CHECK_EQ(sizeof(struct agc_host_hdr_status), 48);
   CHECK_EQ(sizeof(struct agc_aspect_fit_size), 8);
   CHECK_EQ(sizeof(struct agc_aspect_fit_rect), 16);
   CHECK_EQ(offsetof(struct agc_flip_info, argument), 16);
   CHECK_EQ(offsetof(struct agc_display_buffer, dcc_address), 32);
   CHECK_EQ(offsetof(struct agc_display_buffer, alignment), 48);
   CHECK_EQ(offsetof(struct agc_video_output, release), 6 * sizeof(void *));
   CHECK_EQ(offsetof(struct agc_video_output, flip_ready), 7 * sizeof(void *));
   CHECK_EQ(offsetof(struct agc_video_output, rendering_wait_release), 11 * sizeof(void *));
   CHECK_EQ(offsetof(struct agc_presentation_window, native_window), 24);
   CHECK_EQ(offsetof(struct agc_host_display_hdr, max_luminance), 4);
   CHECK_EQ(offsetof(struct agc_host_hdr_status, display), 4);
   CHECK_EQ(offsetof(struct agc_host_hdr_status, generation), 24);
   CHECK_EQ(offsetof(struct agc_host_hdr_status, display_generation), 32);
   CHECK_EQ(offsetof(struct agc_host_hdr_status, reported_tone_map_peak), 40);

   CHECK_EQ(AGC_DRIVER_OK, 0);
   CHECK_EQ(AGC_DRIVER_INVALID, -1);
   CHECK_EQ(AGC_DRIVER_FAILED, -2);
   CHECK_EQ(AGC_DRIVER_SHUTDOWN, -3);
   CHECK_EQ(AGC_DRIVER_FULL, -4);
   CHECK_EQ(AGC_DRIVER_ALREADY_RESERVED, -5);
   CHECK_EQ(AGC_DRIVER_EOP_FLIP_TOKEN_BASE, 0x04000000u);
   CHECK_EQ(AGC_DRIVER_EOP_FLIP_TOKEN_LIMIT, 0x07ffffffu);
   CHECK_EQ(AGC_DRIVER_PRESENTATION_NATIVE_WINDOW, 1);
   CHECK_EQ(AgcDriverPresentationApi(), 1);
   CHECK_EQ(AGC_VIDEO_OUTPUT_SLOTS, 64);
   CHECK_EQ(AGC_EOP_FLIP_SLOTS, 256);
   CHECK_EQ(AGC_EVENT_REGISTRATION_SLOTS, 128);
}

static void
test_registration(void)
{
   fresh();
   struct test_output *first = &outputs[0];
   struct test_output *second = &outputs[1];
   struct agc_video_output table = table_of(first);
   struct agc_video_output other = table_of(second);
   struct agc_driver_error error = {0};

   EXPECT(AgcDriverRegisterVideoOutput(1, NULL, &error), AGC_DRIVER_INVALID, "null video output");
   for (unsigned member = 0; member < 12; member++) {
      void *members[12];
      struct agc_video_output partial;
      memcpy(members, &table, sizeof(members));
      members[member] = NULL;
      memcpy(&partial, members, sizeof(partial));
      EXPECT(AgcDriverRegisterVideoOutput(1, &partial, &error), AGC_DRIVER_INVALID,
             "the video output's table is incomplete");
      EXPECT(AgcDriverRegisterVideoOutput(AGC_VIDEO_OUTPUT_SLOTS, &partial, &error), AGC_DRIVER_INVALID,
             "the video output's table is incomplete");
   }
   EXPECT(AgcDriverRegisterVideoOutput(AGC_VIDEO_OUTPUT_SLOTS, &table, &error), AGC_DRIVER_INVALID,
          "video output handle 64 is out of range");
   EXPECT(AgcDriverRegisterVideoOutput(UINT32_MAX, &table, &error), AGC_DRIVER_INVALID,
          "video output handle 4294967295 is out of range");

   CHECK_EQ(AgcDriverRegisterVideoOutput(1, &table, &error), AGC_DRIVER_OK);
   EXPECT(AgcDriverRegisterVideoOutput(1, &other, &error), AGC_DRIVER_INVALID,
          "video output 1 is already registered");
   CHECK_EQ(AgcDriverRegisterVideoOutput(1, &table, NULL), AGC_DRIVER_INVALID);
   CHECK_EQ(second->releases, 0);
   CHECK_EQ(first->releases, 0);
   CHECK_EQ(AgcDriverRegisterVideoOutput(0, &other, &error), AGC_DRIVER_OK);
   CHECK_EQ(call_count, 0);

   memset(&table, 0, sizeof(table));
   struct agc_flip_info flip = flip_of(1, 1);
   CHECK_EQ(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_OK);
   CHECK_EQ(first->request_count, 1);

   EXPECT(AgcDriverUnregisterVideoOutput(1, second, &error), AGC_DRIVER_INVALID,
          "video output 1 is registered for another output");
   EXPECT(AgcDriverUnregisterVideoOutput(1, NULL, &error), AGC_DRIVER_INVALID,
          "video output 1 is registered for another output");
   EXPECT(AgcDriverUnregisterVideoOutput(2, first, &error), AGC_DRIVER_INVALID,
          "video output 2 is registered for another output");
   EXPECT(AgcDriverUnregisterVideoOutput(AGC_VIDEO_OUTPUT_SLOTS, first, &error), AGC_DRIVER_INVALID,
          "video output handle 64 is out of range");
   CHECK_EQ(first->releases, 0);

   uint32_t at = call_count;
   CHECK_EQ(AgcDriverUnregisterVideoOutput(1, first, &error), AGC_DRIVER_OK);
   CALLED(at, CALL_RELEASE, 1, 0, 0);
   CHECK_EQ(call_count, at + 1);
   CHECK_EQ(first->releases, 1);
   EXPECT(AgcDriverUnregisterVideoOutput(1, first, &error), AGC_DRIVER_INVALID,
          "video output 1 is registered for another output");
   CHECK_EQ(first->releases, 1);
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_INVALID, "video output 1 is not registered");

   struct test_output *third = &outputs[2];
   open_output(1, third);
   open_output(AGC_VIDEO_OUTPUT_SLOTS - 1, &outputs[3]);
   CHECK_EQ(AgcDriverUnregisterVideoOutput(1, third, NULL), AGC_DRIVER_OK);
   CHECK_EQ(AgcDriverUnregisterVideoOutput(0, second, NULL), AGC_DRIVER_OK);
   CHECK_EQ(AgcDriverUnregisterVideoOutput(AGC_VIDEO_OUTPUT_SLOTS - 1, &outputs[3], NULL), AGC_DRIVER_OK);
   CHECK_EQ(first->releases, 1);
   CHECK_EQ(second->releases, 1);
   CHECK_EQ(third->releases, 1);
   CHECK_EQ(outputs[3].releases, 1);
}

static void
close_in_room(struct test_output *output)
{
   closed_in_room.code = AgcDriverUnregisterVideoOutput(2, output, NULL);
   closed_in_room.releases_then = output->releases;
}

static void
test_release_waits_for_users(void)
{
   fresh();
   struct test_output *output = &outputs[0];
   struct agc_driver_error error = {0};
   struct agc_rendering_wait *wait = NULL;
   open_output(3, output);

   CHECK_EQ(agc_flips_rendering_wait(3, 5, &wait, &error), AGC_DRIVER_OK);
   CHECK(wait != NULL);
   CHECK_EQ(AgcDriverUnregisterVideoOutput(3, output, &error), AGC_DRIVER_OK);
   CHECK_EQ(output->releases, 0);
   CHECK_EQ(agc_rendering_wait_for(wait, 7, &error), 1);
   CHECK_EQ(output->releases, 0);
   uint32_t at = call_count;
   agc_rendering_wait_release(wait);
   CALLED(at, CALL_WAIT_RELEASE, 1, 5, 0);
   CALLED(at + 1, CALL_RELEASE, 1, 0, 0);
   CHECK_EQ(call_count, at + 2);
   CHECK_EQ(output->releases, 1);
   EXPECT(agc_flips_rendering_wait(3, 5, &wait, &error), AGC_DRIVER_INVALID, "video output 3 is not registered");

   fresh();
   memset(&closed_in_room, 0xff, sizeof(closed_in_room));
   output->in_room = close_in_room;
   open_output(2, output);
   struct agc_flip_info flip = flip_of(2, 9);
   CHECK_EQ(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_OK);
   CHECK_EQ(closed_in_room.code, AGC_DRIVER_OK);
   CHECK_EQ(closed_in_room.releases_then, 0);
   CALLED(0, CALL_ROOM, 1, 0, 0);
   CALLED(1, CALL_RESERVE, 1, 9, false);
   CALLED(2, CALL_FLIP_READY, 1, 9, 1);
   CALLED(3, CALL_FLIP_RELEASE, 1, 9, 0);
   CALLED(4, CALL_RELEASE, 1, 0, 0);
   CHECK_EQ(call_count, 5);
   CHECK_EQ(output->releases, 1);
}

static void
test_cpu_flip(void)
{
   fresh();
   struct test_output *output = &outputs[0];
   struct agc_driver_error error = {0};
   open_output(1, output);

   struct agc_flip_info flip = {.handle = 1, .index = 2, .mode = 1, .reserved = 0xabcd, .argument = 77};
   CHECK_EQ(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_OK);
   CALLED(0, CALL_ROOM, 1, 0, 0);
   CALLED(1, CALL_RESERVE, 1, 77, false);
   CALLED(2, CALL_FLIP_READY, 1, 77, 1);
   CALLED(3, CALL_FLIP_RELEASE, 1, 77, 0);
   CHECK_EQ(call_count, 4);
   CHECK(memcmp(&output->requests[0].info, &flip, sizeof(flip)) == 0);
   CHECK(!output->requests[0].limited);
   CHECK_EQ(output->requests[0].releases, 1);

   flip.argument = 78;
   CHECK_EQ(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_OK);
   CALLED(6, CALL_FLIP_READY, 1, 78, 2);
   CHECK_EQ(call_count, 8);

   EXPECT(AgcDriverSubmitFlip(NULL, &error), AGC_DRIVER_INVALID, "null flip");
   flip.handle = 5;
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_INVALID, "video output 5 is not registered");
   flip.handle = AGC_VIDEO_OUTPUT_SLOTS;
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_INVALID, "video output handle 64 is out of range");
   flip.handle = 1;
   CHECK_EQ(call_count, 8);

   output->room_code = AGC_DRIVER_SHUTDOWN;
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_SHUTDOWN, "VideoOut's flip queue: room refused -3");
   CHECK_EQ(call_count, 9);
   output->room_code = 7;
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_FAILED, "VideoOut's flip queue: room refused 7");
   output->room_code = AGC_DRIVER_OK;

   output->reserve_code = AGC_DRIVER_FAILED;
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_FAILED, "VideoOut refused the flip: reserve refused -2");
   output->reserve_code = AGC_DRIVER_FULL;
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_FULL, "VideoOut refused the flip: reserve refused -4");
   output->reserve_code = AGC_DRIVER_OK;
   output->null_reservation = true;
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_FAILED, "VideoOut returned a null flip reservation");
   output->null_reservation = false;
   CHECK_EQ(calls_of(CALL_FLIP_READY), 2);
   CHECK_EQ(calls_of(CALL_FLIP_RELEASE), 2);

   uint32_t at = call_count;
   output->ready_code = AGC_DRIVER_FAILED;
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_FAILED, "VideoOut's flip: ready refused -2");
   CALLED(at + 2, CALL_FLIP_READY, 1, 78, 3);
   CALLED(at + 3, CALL_FLIP_RELEASE, 1, 78, 0);
   CHECK_EQ(call_count, at + 4);
   output->ready_code = AGC_DRIVER_OK;

   at = call_count;
   CHECK_EQ(AgcDriverSubmitFlip(&flip, NULL), AGC_DRIVER_OK);
   CALLED(at + 2, CALL_FLIP_READY, 1, 78, 4);
   CHECK_EQ(output->queued, 4);
   close_output(1, output);
}

static void
test_eop_reservation(void)
{
   fresh();
   struct test_output *output = &outputs[0];
   struct agc_driver_error error = {0};
   open_output(1, output);

   struct agc_flip_info flip = flip_of(1, 100);
   uint32_t token = 0;
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   CHECK_EQ(token, TOKEN_BASE);
   CALLED(0, CALL_RESERVE, 1, 100, true);
   CHECK_EQ(call_count, 1);
   CHECK(output->requests[0].limited);

   flip.argument = 101;
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   CHECK_EQ(token, TOKEN_BASE + 1);
   flip.argument = 102;
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 10, &error), AGC_DRIVER_OK);
   flip.argument = 103;
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   CHECK_EQ(token, TOKEN_BASE + 11);
   flip.argument = 104;
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 5, &error), AGC_DRIVER_OK);
   flip.argument = 105;
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   CHECK_EQ(token, TOKEN_BASE + 12);
   CHECK_EQ(call_count, 6);

   flip.argument = 102;
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 10, &error), AGC_DRIVER_ALREADY_RESERVED,
          "the EOP flip token is already reserved for this flip");
   flip.reserved = 0xffff;
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 10, &error), AGC_DRIVER_ALREADY_RESERVED);
   flip.reserved = 0;
   flip.argument = 999;
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 10, &error), AGC_DRIVER_INVALID,
          "the EOP flip token is reserved for a different flip");
   flip.argument = 102;
   flip.index = 1;
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 10, &error), AGC_DRIVER_INVALID);
   flip.index = 0;
   flip.mode = 2;
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 10, &error), AGC_DRIVER_INVALID);
   flip.mode = 1;
   flip.handle = 2;
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 10, &error), AGC_DRIVER_INVALID,
          "the EOP flip token is reserved for a different flip");
   flip.handle = 1;
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 10 + AGC_EOP_FLIP_SLOTS, &error), AGC_DRIVER_FAILED,
          "more than 256 EOP flips are outstanding");
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE - 1, &error), AGC_DRIVER_INVALID,
          "EOP flip token 0x3ffffff is out of range");
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_LIMIT + 1, &error), AGC_DRIVER_INVALID,
          "EOP flip token 0x8000000 is out of range");
   EXPECT(AgcDriverReserveEopFlipAt(&flip, 0, &error), AGC_DRIVER_INVALID, "EOP flip token 0x0 is out of range");
   EXPECT(AgcDriverReserveEopFlip(NULL, &token, &error), AGC_DRIVER_INVALID, "null flip");
   EXPECT(AgcDriverReserveEopFlipAt(NULL, TOKEN_BASE + 40, &error), AGC_DRIVER_INVALID, "null flip");
   EXPECT(AgcDriverReserveEopFlip(&flip, NULL, &error), AGC_DRIVER_INVALID, "null token");
   CHECK_EQ(call_count, 6);

   flip = flip_of(9, 200);
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 40, &error), AGC_DRIVER_INVALID,
          "video output 9 is not registered");
   flip.handle = AGC_VIDEO_OUTPUT_SLOTS;
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 40, &error), AGC_DRIVER_INVALID,
          "video output handle 64 is out of range");
   flip.handle = 1;
   output->reserve_code = AGC_DRIVER_FAILED;
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 40, &error), AGC_DRIVER_FAILED,
          "VideoOut refused the flip: reserve refused -2");
   output->reserve_code = AGC_DRIVER_OK;
   output->null_reservation = true;
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 40, &error), AGC_DRIVER_FAILED,
          "VideoOut returned a null flip reservation");
   output->null_reservation = false;
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 40, &error), AGC_DRIVER_OK);

   flip.handle = 9;
   token = 0xdead;
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_INVALID);
   CHECK_EQ(token, 0xdead);
   flip = flip_of(1, 201);
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   CHECK_EQ(token, TOKEN_BASE + 42);
   close_output(1, output);
   CHECK_EQ(calls_of(CALL_FLIP_RELEASE), 8);
   CHECK_EQ(calls_of(CALL_FLIP_READY), 0);

   fresh();
   output->capacity = 1;
   open_output(1, output);
   flip = flip_of(1, 300);
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   CHECK_EQ(token, TOKEN_BASE);
   flip.argument = 301;
   token = 0xdead;
   EXPECT(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_FULL, "the flip queue is full");
   CHECK_EQ(token, 0xdead);
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 7, &error), AGC_DRIVER_FULL, "the flip queue is full");
   CHECK_EQ(calls_of(CALL_FLIP_RELEASE), 0);
   output->capacity = 4;
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 7, &error), AGC_DRIVER_OK);
   flip.argument = 302;
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 1, &error), AGC_DRIVER_OK);
   flip.argument = 303;
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   CHECK_EQ(token, TOKEN_BASE + 8);
   close_output(1, output);

   fresh();
   open_output(1, output);
   flip = flip_of(1, 400);
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_LIMIT, &error), AGC_DRIVER_OK);
   flip.argument = 401;
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   CHECK_EQ(token, TOKEN_BASE);
   close_output(1, output);

   fresh();
   open_output(1, output);
   flip = flip_of(1, 500);
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 20, &error), AGC_DRIVER_OK);
   flip.argument = 501;
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 20 + AGC_EOP_FLIP_SLOTS, &error), AGC_DRIVER_FAILED,
          "more than 256 EOP flips are outstanding");
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 20 + 2 * AGC_EOP_FLIP_SLOTS, &error), AGC_DRIVER_FAILED,
          "more than 256 EOP flips are outstanding");
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 20 + AGC_EOP_FLIP_SLOTS - 1, &error), AGC_DRIVER_OK);
   flip.argument = 502;
   CHECK_EQ(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 20 + AGC_EOP_FLIP_SLOTS + 1, &error), AGC_DRIVER_OK);
   struct agc_event aliased = eop_event(0, TOKEN_BASE + 20 + AGC_EOP_FLIP_SLOTS);
   CHECK_EQ(agc_flips_deliver(&aliased, 1, &error), AGC_DRIVER_OK);
   aliased = eop_event(0, 20);
   CHECK_EQ(agc_flips_deliver(&aliased, 1, &error), AGC_DRIVER_OK);
   CHECK_EQ(calls_of(CALL_FLIP_READY), 0);
   aliased = eop_event(0, TOKEN_BASE + 20);
   CHECK_EQ(agc_flips_deliver(&aliased, 1, &error), AGC_DRIVER_OK);
   CHECK_EQ(calls_of(CALL_FLIP_READY), 1);
   CHECK(output->requests[0].readied);
   close_output(1, output);
}

static void
test_unregister_drops_reservations(void)
{
   fresh();
   struct test_output *first = &outputs[0];
   struct test_output *second = &outputs[1];
   struct agc_driver_error error = {0};
   uint32_t token = 0;
   open_output(1, first);
   open_output(2, second);

   struct agc_flip_info flip = flip_of(1, 10);
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   flip.argument = 11;
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   struct agc_flip_info kept = flip_of(2, 20);
   CHECK_EQ(AgcDriverReserveEopFlip(&kept, &token, &error), AGC_DRIVER_OK);
   CHECK_EQ(token, TOKEN_BASE + 2);

   uint32_t at = call_count;
   CHECK_EQ(AgcDriverUnregisterVideoOutput(1, first, &error), AGC_DRIVER_OK);
   CALLED(at, CALL_FLIP_RELEASE, 1, 10, 0);
   CALLED(at + 1, CALL_FLIP_RELEASE, 1, 11, 0);
   CALLED(at + 2, CALL_RELEASE, 1, 0, 0);
   CHECK_EQ(call_count, at + 3);
   CHECK_EQ(first->requests[0].releases, 1);
   CHECK_EQ(first->requests[1].releases, 1);
   CHECK_EQ(first->requests[0].fails, 0);
   CHECK(!first->requests[0].readied);
   CHECK_EQ(first->queued, 0);
   CHECK_EQ(second->queued, 1);

   CHECK_EQ(AgcDriverReserveEopFlipAt(&kept, TOKEN_BASE + 2, &error), AGC_DRIVER_ALREADY_RESERVED);
   struct agc_event dropped[2] = {eop_event(0, TOKEN_BASE), eop_event(0, TOKEN_BASE + 1)};
   CHECK_EQ(agc_flips_deliver(dropped, 2, &error), AGC_DRIVER_OK);
   CHECK_EQ(calls_of(CALL_FLIP_READY), 0);
   kept.argument = 21;
   CHECK_EQ(AgcDriverReserveEopFlipAt(&kept, TOKEN_BASE, &error), AGC_DRIVER_OK);
   CHECK_EQ(first->releases, 1);
   close_output(2, second);
   CHECK_EQ(second->requests[0].releases, 1);
   CHECK_EQ(second->requests[1].releases, 1);
}

static void
test_delivery_order(void)
{
   fresh();
   struct agc_driver_error error = {0};
   open_output(1, &outputs[0]);
   open_output(2, &outputs[1]);
   CHECK_EQ(agc_events_add(0x51, 0, 0, NULL, &error), AGC_DRIVER_OK);
   CHECK_EQ(agc_events_add(0x52, 0x21, 0x21, NULL, &error), AGC_DRIVER_OK);

   struct agc_event events[] = {
      flip_event(1, 1), eop_event(0, 0x1234), flip_event(2, 2), eop_event(0x21, 0xdeadbeef), flip_event(1, 3),
   };
   uint32_t at = call_count;
   CHECK_EQ(agc_flips_deliver(events, ARRAY_COUNT(events), &error), AGC_DRIVER_OK);
   CALLED(at, CALL_ROOM, 1, 0, 0);
   CALLED(at + 1, CALL_ROOM, 2, 0, 0);
   CALLED(at + 2, CALL_ROOM, 1, 0, 0);
   CALLED(at + 3, CALL_RESERVE, 1, 1, false);
   CALLED(at + 4, CALL_RESERVE, 2, 2, false);
   CALLED(at + 5, CALL_RESERVE, 1, 3, false);
   CALLED(at + 6, CALL_FLIP_READY, 1, 1, 1);
   CALLED(at + 7, CALL_FLIP_RELEASE, 1, 1, 0);
   CALLED(at + 8, CALL_EVENT_TRIGGER, 0, 0x51, UINT64_C(0x12340000));
   CALLED(at + 9, CALL_FLIP_READY, 2, 2, 2);
   CALLED(at + 10, CALL_FLIP_RELEASE, 2, 2, 0);
   CALLED(at + 11, CALL_EVENT_TRIGGER, 0x21, 0x52, UINT64_C(0xdeadbeef0021));
   CALLED(at + 12, CALL_FLIP_READY, 1, 3, 3);
   CALLED(at + 13, CALL_FLIP_RELEASE, 1, 3, 0);
   CHECK_EQ(call_count, at + 14);

   CHECK_EQ(agc_flips_deliver(NULL, 0, &error), AGC_DRIVER_OK);
   CHECK_EQ(agc_flips_deliver(events, 0, &error), AGC_DRIVER_OK);
   CHECK_EQ(call_count, at + 14);
   close_output(1, &outputs[0]);
   close_output(2, &outputs[1]);

   fresh();
   struct test_output *output = &outputs[0];
   struct agc_event many[12];
   open_output(1, output);
   for (unsigned i = 0; i < ARRAY_COUNT(many); i++)
      many[i] = flip_event(1, 500 + i);
   CHECK_EQ(agc_flips_deliver(many, ARRAY_COUNT(many), &error), AGC_DRIVER_OK);
   CHECK_EQ(call_count, 48);
   for (unsigned i = 0; i < ARRAY_COUNT(many); i++) {
      CALLED(i, CALL_ROOM, 1, 0, 0);
      CALLED(12 + i, CALL_RESERVE, 1, 500 + i, false);
      CALLED(24 + 2 * i, CALL_FLIP_READY, 1, 500 + i, i + 1);
      CALLED(25 + 2 * i, CALL_FLIP_RELEASE, 1, 500 + i, 0);
   }
   close_output(1, output);
}

static void
test_delivery_of_reserved_flips(void)
{
   fresh();
   struct test_output *output = &outputs[0];
   struct agc_driver_error error = {0};
   uint32_t token = 0;
   open_output(1, output);

   struct agc_flip_info reserved = flip_of(1, 50);
   CHECK_EQ(AgcDriverReserveEopFlip(&reserved, &token, &error), AGC_DRIVER_OK);
   struct agc_event events[] = {flip_event(1, 51), eop_event(0, token), flip_event(1, 52)};
   uint32_t at = call_count;
   CHECK_EQ(agc_flips_deliver(events, ARRAY_COUNT(events), &error), AGC_DRIVER_OK);
   CALLED(at, CALL_RESERVE, 1, 51, false);
   CALLED(at + 1, CALL_RESERVE, 1, 52, false);
   CALLED(at + 2, CALL_FLIP_READY, 1, 51, 1);
   CALLED(at + 3, CALL_FLIP_RELEASE, 1, 51, 0);
   CALLED(at + 4, CALL_FLIP_READY, 1, 50, 2);
   CALLED(at + 5, CALL_FLIP_RELEASE, 1, 50, 0);
   CALLED(at + 6, CALL_FLIP_READY, 1, 52, 3);
   CALLED(at + 7, CALL_FLIP_RELEASE, 1, 52, 0);
   CHECK_EQ(call_count, at + 8);
   CHECK_EQ(calls_of(CALL_ROOM), 0);
   CHECK(output->requests[0].readied);
   CHECK_EQ(output->requests[0].frame, 2);
   CHECK_EQ(output->requests[0].releases, 1);

   at = call_count;
   CHECK_EQ(agc_flips_deliver(events, ARRAY_COUNT(events), &error), AGC_DRIVER_OK);
   CALLED(at, CALL_ROOM, 1, 0, 0);
   CALLED(at + 1, CALL_ROOM, 1, 0, 0);
   CALLED(at + 2, CALL_RESERVE, 1, 51, false);
   CALLED(at + 3, CALL_RESERVE, 1, 52, false);
   CALLED(at + 4, CALL_FLIP_READY, 1, 51, 4);
   CALLED(at + 5, CALL_FLIP_RELEASE, 1, 51, 0);
   CALLED(at + 6, CALL_FLIP_READY, 1, 52, 5);
   CALLED(at + 7, CALL_FLIP_RELEASE, 1, 52, 0);
   CHECK_EQ(call_count, at + 8);
   close_output(1, output);

   fresh();
   open_output(1, output);
   reserved = flip_of(1, 60);
   CHECK_EQ(AgcDriverReserveEopFlip(&reserved, &token, &error), AGC_DRIVER_OK);
   struct agc_event compute[] = {flip_event(1, 61), eop_event(0x20, token), flip_event(1, 62)};
   at = call_count;
   EXPECT(agc_flips_deliver(compute, ARRAY_COUNT(compute), &error), AGC_DRIVER_FAILED,
          "EOP flips on async compute queues are not implemented");
   CALLED(at, CALL_RESERVE, 1, 61, false);
   CALLED(at + 1, CALL_RESERVE, 1, 62, false);
   CALLED(at + 2, CALL_FLIP_READY, 1, 61, 1);
   CALLED(at + 3, CALL_FLIP_RELEASE, 1, 61, 0);
   CALLED(at + 4, CALL_FLIP_RELEASE, 1, 60, 0);
   CALLED(at + 5, CALL_FLIP_RELEASE, 1, 62, 0);
   CHECK_EQ(call_count, at + 6);
   CHECK_EQ(calls_of(CALL_ROOM), 0);
   CHECK(!output->requests[0].readied);
   CHECK_EQ(output->requests[0].releases, 1);
   reserved.argument = 63;
   CHECK_EQ(AgcDriverReserveEopFlipAt(&reserved, token, &error), AGC_DRIVER_OK);
   close_output(1, output);

   fresh();
   open_output(1, output);
   reserved = flip_of(1, 64);
   CHECK_EQ(AgcDriverReserveEopFlip(&reserved, &token, &error), AGC_DRIVER_OK);
   struct agc_event alone = eop_event(0x20, token);
   at = call_count;
   EXPECT(agc_flips_deliver(&alone, 1, &error), AGC_DRIVER_FAILED,
          "EOP flips on async compute queues are not implemented");
   CALLED(at, CALL_FLIP_RELEASE, 1, 64, 0);
   CHECK_EQ(call_count, at + 1);
   alone = eop_event(0, token);
   CHECK_EQ(agc_flips_deliver(&alone, 1, &error), AGC_DRIVER_OK);
   CHECK_EQ(call_count, at + 1);
   close_output(1, output);
}

static void
test_delivery_failures(void)
{
   fresh();
   struct test_output *first = &outputs[0];
   struct test_output *second = &outputs[1];
   struct agc_driver_error error = {0};
   open_output(1, first);
   open_output(2, second);

   struct agc_event unknown[] = {flip_event(1, 1), {.kind = 2}};
   EXPECT(agc_flips_deliver(unknown, 2, &error), AGC_DRIVER_INVALID, "unknown event kind 2");
   EXPECT(agc_flips_deliver(NULL, 1, &error), AGC_DRIVER_INVALID, "null events");
   CHECK_EQ(call_count, 0);

   struct agc_event pair[] = {flip_event(1, 1), flip_event(2, 2)};
   second->room_code = AGC_DRIVER_FAILED;
   EXPECT(agc_flips_deliver(pair, 2, &error), AGC_DRIVER_FAILED, "VideoOut's flip queue: room refused -2");
   CALLED(0, CALL_ROOM, 1, 0, 0);
   CALLED(1, CALL_ROOM, 2, 0, 0);
   CHECK_EQ(call_count, 2);
   second->room_code = AGC_DRIVER_OK;

   struct agc_event absent[] = {flip_event(1, 1), flip_event(9, 2)};
   EXPECT(agc_flips_deliver(absent, 2, &error), AGC_DRIVER_INVALID, "video output 9 is not registered");
   CALLED(2, CALL_ROOM, 1, 0, 0);
   CHECK_EQ(call_count, 3);

   struct agc_event three[] = {flip_event(1, 1), flip_event(2, 2), flip_event(1, 3)};
   uint32_t at = call_count;
   second->reserve_code = AGC_DRIVER_FAILED;
   EXPECT(agc_flips_deliver(three, 3, &error), AGC_DRIVER_FAILED, "VideoOut refused the flip: reserve refused -2");
   CALLED(at + 3, CALL_RESERVE, 1, 1, false);
   CALLED(at + 4, CALL_RESERVE, 2, 2, false);
   CALLED(at + 5, CALL_FLIP_RELEASE, 1, 1, 0);
   CHECK_EQ(call_count, at + 6);
   CHECK_EQ(calls_of(CALL_FLIP_READY), 0);
   CHECK_EQ(first->queued, 0);
   second->reserve_code = AGC_DRIVER_OK;

   at = call_count;
   first->ready_code = AGC_DRIVER_SHUTDOWN;
   EXPECT(agc_flips_deliver(pair, 2, &error), AGC_DRIVER_SHUTDOWN, "VideoOut's flip: ready refused -3");
   CALLED(at, CALL_ROOM, 1, 0, 0);
   CALLED(at + 1, CALL_ROOM, 2, 0, 0);
   CALLED(at + 2, CALL_RESERVE, 1, 1, false);
   CALLED(at + 3, CALL_RESERVE, 2, 2, false);
   CALLED(at + 4, CALL_FLIP_READY, 1, 1, 1);
   CALLED(at + 5, CALL_FLIP_RELEASE, 1, 1, 0);
   CALLED(at + 6, CALL_FLIP_RELEASE, 2, 2, 0);
   CHECK_EQ(call_count, at + 7);
   CHECK_EQ(second->queued, 0);
   close_output(1, first);
   close_output(2, second);
}

static void
test_eop_interrupts(void)
{
   fresh();
   struct agc_driver_error error = {0};
   static const uint32_t queues[] = {0, 0x21, 0x121, 0xff, 0x20};
   for (unsigned i = 0; i < ARRAY_COUNT(queues); i++)
      CHECK_EQ(agc_events_add(0x100 + queues[i], queues[i], queues[i], NULL, &error), AGC_DRIVER_OK);
   CHECK_EQ(agc_events_add(0x300, 0x21, 0x21, NULL, &error), AGC_DRIVER_OK);

   uint32_t at = call_count;
   struct agc_event event = eop_event(0x21, 0xdeadbeef);
   CHECK_EQ(agc_flips_deliver(&event, 1, &error), AGC_DRIVER_OK);
   CALLED(at, CALL_EVENT_TRIGGER, 0x21, 0x121, UINT64_C(0xdeadbeef0021));
   CALLED(at + 1, CALL_EVENT_TRIGGER, 0x21, 0x300, UINT64_C(0xdeadbeef0021));
   CHECK_EQ(call_count, at + 2);
   CHECK_EQ(libkernel.trigger_filter, AGC_EVFILT_GRAPHICS_CORE);

   event = eop_event(0, TOKEN_BASE + 3);
   CHECK_EQ(agc_flips_deliver(&event, 1, &error), AGC_DRIVER_OK);
   CALLED(at + 2, CALL_EVENT_TRIGGER, 0, 0x100, (uint64_t)(TOKEN_BASE + 3) << 16);
   CHECK_EQ(call_count, at + 3);

   at = call_count;
   agc_events_interrupt(0, 0);
   agc_events_interrupt(0, 0x1234);
   agc_events_interrupt(0x121, 7);
   agc_events_interrupt(0xff, 0xffffffff);
   agc_events_interrupt(0x20, 0x04000000);
   agc_events_interrupt(0x22, 5);
   CALLED(at, CALL_EVENT_TRIGGER, 0, 0x100, 0);
   CALLED(at + 1, CALL_EVENT_TRIGGER, 0, 0x100, UINT64_C(0x12340000));
   CALLED(at + 2, CALL_EVENT_TRIGGER, 0x121, 0x221, UINT64_C(0x70021));
   CALLED(at + 3, CALL_EVENT_TRIGGER, 0xff, 0x1ff, UINT64_C(0xffffffff00ff));
   CALLED(at + 4, CALL_EVENT_TRIGGER, 0x20, 0x120, UINT64_C(0x40000000020));
   CHECK_EQ(call_count, at + 5);
}

static void
test_sticky_failure(void)
{
   fresh();
   struct test_output *first = &outputs[0];
   struct test_output *second = &outputs[1];
   struct agc_driver_error error = {0};
   struct agc_rendering_wait *wait = NULL;
   struct agc_display_buffer buffer = {0};
   uint32_t first_token = 0;
   uint32_t second_token = 0;
   int completion = 3;
   open_output(1, first);
   open_output(2, second);
   agc_videoout_set_presenter(&recording_presenter);
   CHECK_EQ(agc_events_add(0x60, 0, 0, NULL, &error), AGC_DRIVER_OK);

   struct agc_flip_info flip = flip_of(1, 5);
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &first_token, &error), AGC_DRIVER_OK);
   flip = flip_of(2, 6);
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &second_token, &error), AGC_DRIVER_OK);
   CHECK(agc_flips_failure() == NULL);

   uint32_t at = call_count;
   agc_flips_fail("the device was lost");
   CALLED(at, CALL_FAIL, 1, 0, 0);
   CALLED(at + 1, CALL_FAIL, 2, 0, 0);
   CALLED(at + 2, CALL_FLIP_FAIL, 1, 5, 0);
   CALLED(at + 3, CALL_FLIP_RELEASE, 1, 5, 0);
   CALLED(at + 4, CALL_FLIP_FAIL, 2, 6, 0);
   CALLED(at + 5, CALL_FLIP_RELEASE, 2, 6, 0);
   CHECK_EQ(call_count, at + 6);
   CHECK(strcmp(first->failure, "the device was lost") == 0);
   CHECK(strcmp(second->requests[0].failure, "the device was lost") == 0);
   CHECK(agc_flips_failure() != NULL && strcmp(agc_flips_failure(), "the device was lost") == 0);

   agc_flips_fail("second");
   CALLED(at + 6, CALL_FAIL, 1, 0, 0);
   CALLED(at + 7, CALL_FAIL, 2, 0, 0);
   CHECK_EQ(call_count, at + 8);
   CHECK(strcmp(second->failure, "second") == 0);
   CHECK(strcmp(agc_flips_failure(), "the device was lost") == 0);

   at = call_count;
   struct agc_video_output third = table_of(&outputs[2]);
   struct agc_presentation_window window = {.context = &window, .width = 1280, .height = 720, .native_window = 1};
   struct agc_event flipped = flip_event(1, 7);
   flip = flip_of(1, 8);
   EXPECT(AgcDriverRegisterVideoOutput(3, &third, &error), AGC_DRIVER_FAILED, "the device was lost");
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_FAILED, "the device was lost");
   EXPECT(AgcDriverReserveEopFlip(&flip, &first_token, &error), AGC_DRIVER_FAILED, "the device was lost");
   EXPECT(AgcDriverReserveEopFlipAt(&flip, TOKEN_BASE + 9, &error), AGC_DRIVER_FAILED, "the device was lost");
   EXPECT(agc_flips_deliver(&flipped, 1, &error), AGC_DRIVER_FAILED, "the device was lost");
   EXPECT(agc_flips_rendering_wait(1, 0, &wait, &error), AGC_DRIVER_FAILED, "the device was lost");
   CHECK(!agc_flips_describe_buffer(1, 0, &buffer));
   EXPECT(AgcDriverPresentBuffer(&window, &buffer, NULL, gpu_ready, &completion, &error), AGC_DRIVER_FAILED,
          "the device was lost");
   EXPECT(AgcDriverPresentClear(&window, true, NULL, gpu_ready, &completion, &error), AGC_DRIVER_FAILED,
          "the device was lost");
   EXPECT(AgcDriverPresentBuffer(&window, &buffer, NULL, NULL, &completion, &error), AGC_DRIVER_INVALID,
          "missing GPU completion callback");
   CHECK_EQ(call_count, at);

   struct agc_event interrupted = eop_event(0, first_token);
   CHECK_EQ(agc_flips_deliver(&interrupted, 1, &error), AGC_DRIVER_OK);
   CALLED(at, CALL_EVENT_TRIGGER, 0, 0x60, (uint64_t)first_token << 16);
   CHECK_EQ(AgcDriverUnregisterVideoOutput(1, first, &error), AGC_DRIVER_OK);
   CALLED(at + 1, CALL_RELEASE, 1, 0, 0);
   CHECK_EQ(call_count, at + 2);
   agc_flips_fail(NULL);
   CHECK(strcmp(second->failure, "(no message)") == 0);
   CHECK_EQ(first->fails, 2);
   CHECK_EQ(second->fails, 3);
   close_output(2, second);

   fresh();
   char lengthy[1101];
   memset(lengthy, 'x', sizeof(lengthy) - 1);
   lengthy[sizeof(lengthy) - 1] = 0;
   agc_flips_fail(lengthy);
   CHECK_EQ(strlen(agc_flips_failure()), 1023);
   flip = flip_of(1, 1);
   CHECK_EQ(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_FAILED);
   CHECK_EQ(strlen(error.message), 255);
}

static void
test_shutdown(void)
{
   fresh();
   struct test_output *output = &outputs[0];
   struct agc_driver_error error = {0};
   struct agc_rendering_wait *reader = NULL;
   struct agc_rendering_wait *late = NULL;
   open_output(1, output);
   CHECK_EQ(agc_flips_rendering_wait(1, 0, &reader, &error), AGC_DRIVER_OK);
   agc_flips_shutdown();
   agc_flips_shutdown();
   EXPECT(agc_flips_rendering_wait(1, 0, &late, &error), AGC_DRIVER_SHUTDOWN,
          "the video output driver has stopped");
   CHECK_EQ(AgcDriverUnregisterVideoOutput(1, NULL, &error), AGC_DRIVER_INVALID);
   CHECK_EQ(AgcDriverUnregisterVideoOutput(1, output, &error), AGC_DRIVER_OK);
   CHECK_EQ(AgcDriverUnregisterVideoOutput(1, output, &error), AGC_DRIVER_INVALID);
   CHECK_EQ(output->releases, 0);
   agc_rendering_wait_release(reader);
   CHECK_EQ(output->releases, 1);

   fresh();
   struct agc_display_buffer buffer = {0};
   uint32_t token = 0;
   open_output(1, output);
   CHECK_EQ(agc_events_add(0x70, 0, 0, NULL, &error), AGC_DRIVER_OK);
   struct agc_flip_info flip = flip_of(1, 30);
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   flip.argument = 31;
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);

   uint32_t at = call_count;
   agc_flips_shutdown();
   CALLED(at, CALL_FLIP_RELEASE, 1, 30, 0);
   CALLED(at + 1, CALL_FLIP_RELEASE, 1, 31, 0);
   CHECK_EQ(call_count, at + 2);
   CHECK_EQ(output->releases, 0);
   CHECK_EQ(output->fails, 0);

   struct agc_video_output other = table_of(&outputs[1]);
   struct agc_event flipped = flip_event(1, 32);
   EXPECT(AgcDriverRegisterVideoOutput(2, &other, &error), AGC_DRIVER_SHUTDOWN,
          "the video output driver has stopped");
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_SHUTDOWN, "the video output driver has stopped");
   EXPECT(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_SHUTDOWN,
          "the video output driver has stopped");
   EXPECT(agc_flips_deliver(&flipped, 1, &error), AGC_DRIVER_SHUTDOWN, "the video output driver has stopped");
   CHECK(!agc_flips_describe_buffer(1, 0, &buffer));
   agc_flips_fail("after the stop");
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_SHUTDOWN, "the video output driver has stopped");
   CHECK_EQ(output->fails, 1);

   at = call_count;
   struct agc_event interrupted = eop_event(0, token);
   CHECK_EQ(agc_flips_deliver(&interrupted, 1, &error), AGC_DRIVER_OK);
   CALLED(at, CALL_EVENT_TRIGGER, 0, 0x70, (uint64_t)token << 16);
   CHECK_EQ(AgcDriverUnregisterVideoOutput(1, output, &error), AGC_DRIVER_OK);
   CHECK_EQ(output->releases, 1);

   fresh();
   agc_videoout_set_core(&recording_core);
   AgcDriverWaitIdle();
   AgcDriverShutdown();
   CALLED(0, CALL_WAIT_IDLE, 0, 0, 0);
   CALLED(1, CALL_CORE_SHUTDOWN, 0, 0, 0);
   CHECK_EQ(call_count, 2);
   EXPECT(AgcDriverRegisterVideoOutput(2, &other, &error), AGC_DRIVER_SHUTDOWN,
          "the video output driver has stopped");

   fresh();
   AgcDriverWaitIdle();
   AgcDriverShutdown();
   CHECK_EQ(call_count, 0);
   EXPECT(AgcDriverRegisterVideoOutput(2, &other, &error), AGC_DRIVER_SHUTDOWN,
          "the video output driver has stopped");
}

struct reserving {
   thrd_t thread;
   struct agc_flip_info flip;
   uint32_t token;
   int code;
};

struct delivering {
   thrd_t thread;
   struct agc_event event;
   int code;
};

static int
reserving_main(void *data)
{
   struct reserving *self = data;
   self->code = AgcDriverReserveEopFlip(&self->flip, &self->token, NULL);
   return 0;
}

static int
delivering_main(void *data)
{
   struct delivering *self = data;
   self->code = agc_flips_deliver(&self->event, 1, NULL);
   return 0;
}

static int
stopping_main(void *data)
{
   (void)data;
   agc_flips_shutdown();
   return 0;
}

static void
test_shutdown_races(void)
{
   for (unsigned paused = 0; paused < 2; paused++) {
      fresh();
      struct test_output *output = &outputs[0];
      struct gate gate = {0};
      struct reserving reserving = {.flip = flip_of(1, 70), .code = 1};
      thrd_t stopping;
      output->reserve_gate = paused ? &gate : NULL;
      open_output(1, output);
      if (paused) {
         CHECK(thrd_create(&reserving.thread, reserving_main, &reserving) == thrd_success);
         gate_await(&gate);
         agc_flips_shutdown();
         CHECK_EQ(AgcDriverUnregisterVideoOutput(1, output, NULL), AGC_DRIVER_OK);
         CHECK_EQ(output->releases, 0);
         p_atomic_set(&gate.open, 1);
         CHECK(thrd_join(reserving.thread, NULL) == thrd_success);
         CHECK_EQ(reserving.code, AGC_DRIVER_SHUTDOWN);
      } else {
         CHECK_EQ(AgcDriverReserveEopFlip(&reserving.flip, &reserving.token, NULL), AGC_DRIVER_OK);
         CHECK(thrd_create(&stopping, stopping_main, NULL) == thrd_success);
         CHECK_EQ(AgcDriverUnregisterVideoOutput(1, output, NULL), AGC_DRIVER_OK);
         CHECK(thrd_join(stopping, NULL) == thrd_success);
      }
      CHECK_EQ(output->request_count, 1);
      CHECK_EQ(output->requests[0].releases, 1);
      CHECK_EQ(output->requests[0].fails, 0);
      CHECK(!output->requests[0].readied);
      CHECK_EQ(output->releases, 1);
   }

   fresh();
   struct test_output *output = &outputs[0];
   struct gate gate = {0};
   struct reserving reserving = {.flip = flip_of(1, 71), .code = 1};
   output->reserve_gate = &gate;
   open_output(1, output);
   CHECK(thrd_create(&reserving.thread, reserving_main, &reserving) == thrd_success);
   gate_await(&gate);
   use_unrelated_state();
   CHECK_EQ(AgcDriverUnregisterVideoOutput(1, output, NULL), AGC_DRIVER_OK);
   CHECK_EQ(output->releases, 0);
   p_atomic_set(&gate.open, 1);
   CHECK(thrd_join(reserving.thread, NULL) == thrd_success);
   CHECK_EQ(reserving.code, AGC_DRIVER_INVALID);
   CHECK_EQ(output->requests[0].releases, 1);
   CHECK_EQ(output->releases, 1);
   struct agc_event event = eop_event(0, TOKEN_BASE);
   CHECK_EQ(agc_flips_deliver(&event, 1, NULL), AGC_DRIVER_OK);
   CHECK_EQ(calls_of(CALL_FLIP_READY), 0);

   fresh();
   memset(&gate, 0, sizeof(gate));
   reserving = (struct reserving){.flip = flip_of(1, 72), .code = 1};
   output->reserve_gate = &gate;
   open_output(1, output);
   CHECK(thrd_create(&reserving.thread, reserving_main, &reserving) == thrd_success);
   gate_await(&gate);
   agc_flips_fail("lost while reserving");
   p_atomic_set(&gate.open, 1);
   CHECK(thrd_join(reserving.thread, NULL) == thrd_success);
   CHECK_EQ(reserving.code, AGC_DRIVER_FAILED);
   CHECK_EQ(output->requests[0].releases, 1);
   CHECK_EQ(output->requests[0].fails, 0);
   close_output(1, output);

   fresh();
   memset(&gate, 0, sizeof(gate));
   reserving = (struct reserving){.flip = flip_of(1, 73), .code = 1};
   output->reserve_gate = &gate;
   open_output(1, output);
   CHECK(thrd_create(&reserving.thread, reserving_main, &reserving) == thrd_success);
   gate_await(&gate);
   agc_flips_shutdown();
   p_atomic_set(&gate.open, 1);
   CHECK(thrd_join(reserving.thread, NULL) == thrd_success);
   CHECK_EQ(reserving.code, AGC_DRIVER_SHUTDOWN);
   CHECK_EQ(output->requests[0].releases, 1);
   CHECK(!output->requests[0].readied);
   close_output(1, output);
   CHECK_EQ(calls_of(CALL_FLIP_RELEASE), 1);
}

static void
test_reservation_during_room_wait(void)
{
   fresh();
   struct test_output *output = &outputs[0];
   struct gate gate = {0};
   struct delivering delivering = {.event = flip_event(1, 80), .code = 1};
   struct agc_display_buffer buffer = {0};
   uint32_t token = 0;
   output->room_gate = &gate;
   open_output(1, output);
   CHECK(thrd_create(&delivering.thread, delivering_main, &delivering) == thrd_success);
   gate_await(&gate);

   struct agc_flip_info flip = flip_of(1, 81);
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, NULL), AGC_DRIVER_OK);
   CHECK_EQ(token, TOKEN_BASE);
   CHECK(agc_flips_describe_buffer(1, 0, &buffer));
   use_unrelated_state();
   p_atomic_set(&gate.open, 1);
   CHECK(thrd_join(delivering.thread, NULL) == thrd_success);
   CHECK_EQ(delivering.code, AGC_DRIVER_OK);
   CALLED(0, CALL_ROOM, 1, 0, 0);
   CALLED(1, CALL_RESERVE, 1, 81, true);
   CALLED(2, CALL_DESCRIBE, 1, 0, 0);
   CALLED(3, CALL_EVENT_ADD, 0x33, 0x77, 0x33);
   CALLED(4, CALL_EVENT_TRIGGER, 0x33, 0x77, UINT64_C(0x10033));
   CALLED(5, CALL_EVENT_DELETE, 0x33, 0x77, 0);
   CALLED(6, CALL_RESERVE, 1, 80, false);
   CALLED(7, CALL_FLIP_READY, 1, 80, 1);
   CALLED(8, CALL_FLIP_RELEASE, 1, 80, 0);
   CHECK_EQ(call_count, 9);
   close_output(1, output);
   CHECK_EQ(calls_of(CALL_FLIP_RELEASE), 2);
}

static void
test_events(void)
{
   fresh();
   struct agc_driver_error error = {0};
   int udata = 0;

   EXPECT(agc_events_add(0, 0, 0, NULL, &error), AGC_DRIVER_INVALID, "invalid event queue");
   EXPECT(agc_events_delete(0, 0, &error), AGC_DRIVER_INVALID, "invalid event queue");
   CHECK_EQ(call_count, 0);

   libkernel.add_code = 1;
   EXPECT(agc_events_add(0x40, 0x21, 0x21, &udata, &error), AGC_DRIVER_INVALID,
          "the event queue rejected the event");
   CALLED(0, CALL_EVENT_ADD, 0x21, 0x40, 0x21);
   agc_events_interrupt(0x21, 1);
   CHECK_EQ(call_count, 1);
   libkernel.add_code = 0;

   CHECK_EQ(agc_events_add(0x40, 0x21, 0x21, &udata, &error), AGC_DRIVER_OK);
   CALLED(1, CALL_EVENT_ADD, 0x21, 0x40, 0x21);
   CHECK_EQ(libkernel.add_filter, AGC_EVFILT_GRAPHICS_CORE);
   CHECK_EQ(libkernel.add_filter, -14);
   CHECK_EQ(libkernel.add_flags, 0x21);
   CHECK(libkernel.add_udata == &udata);
   CHECK_EQ(agc_events_add(0x40, 0x21, 0x21, NULL, &error), AGC_DRIVER_OK);
   CHECK(libkernel.add_udata == NULL);
   CHECK_EQ(agc_events_add(0x40, 0x22, 0x21, NULL, NULL), AGC_DRIVER_OK);
   CHECK_EQ(calls_of(CALL_EVENT_ADD), 4);

   uint32_t at = call_count;
   agc_events_interrupt(0x22, 1);
   CHECK_EQ(call_count, at);
   agc_events_interrupt(0x21, 1);
   CALLED(at, CALL_EVENT_TRIGGER, 0x21, 0x40, UINT64_C(0x10021));
   CHECK_EQ(call_count, at + 1);

   libkernel.delete_code = 1;
   EXPECT(agc_events_delete(0x40, 0x21, &error), AGC_DRIVER_INVALID, "the event is not registered");
   CALLED(at + 1, CALL_EVENT_DELETE, 0x21, 0x40, 0);
   CHECK_EQ(libkernel.delete_filter, AGC_EVFILT_GRAPHICS_CORE);
   agc_events_interrupt(0x21, 2);
   CALLED(at + 2, CALL_EVENT_TRIGGER, 0x21, 0x40, UINT64_C(0x20021));
   libkernel.delete_code = 0;
   CHECK_EQ(agc_events_delete(0x40, 0x21, &error), AGC_DRIVER_OK);
   at = call_count;
   agc_events_interrupt(0x21, 3);
   CHECK_EQ(call_count, at);

   fresh();
   for (unsigned i = 0; i < AGC_EVENT_REGISTRATION_SLOTS; i++)
      CHECK_EQ(agc_events_add(0x1000 + i, 5, i, NULL, &error), AGC_DRIVER_OK);
   EXPECT(agc_events_add(0x2000, 5, 0, NULL, &error), AGC_DRIVER_FAILED,
          "more than 128 graphics events are registered");
   CHECK_EQ(agc_events_add(0x1000 + 17, 5, 17, NULL, &error), AGC_DRIVER_OK);
   at = call_count;
   agc_events_interrupt(5, 9);
   CHECK_EQ(call_count, at + AGC_EVENT_REGISTRATION_SLOTS);
   CHECK_EQ(agc_events_delete(0x1000 + 17, 17, &error), AGC_DRIVER_OK);
   CHECK_EQ(agc_events_add(0x2000, 5, 0, NULL, &error), AGC_DRIVER_OK);
   at = call_count;
   agc_events_interrupt(5, 9);
   CHECK_EQ(call_count, at + AGC_EVENT_REGISTRATION_SLOTS);
   CHECK_EQ(calls_of(CALL_EVENT_TRIGGER), 2 * AGC_EVENT_REGISTRATION_SLOTS);

   agc_anyps5_set(NULL);
   EXPECT(agc_events_add(0x40, 0, 0, NULL, &error), AGC_DRIVER_FAILED, "AnyPS5's event queues are not available");
   EXPECT(agc_events_delete(0x40, 0, &error), AGC_DRIVER_FAILED, "AnyPS5's event queues are not available");
   agc_events_interrupt(5, 9);
   CHECK_EQ(call_count, at + AGC_EVENT_REGISTRATION_SLOTS);
}

static void
test_rendering_wait(void)
{
   fresh();
   struct test_output *output = &outputs[0];
   struct agc_driver_error error = {0};
   struct agc_rendering_wait *wait = NULL;
   open_output(1, output);

   output->capture_code = AGC_DRIVER_INVALID;
   EXPECT(agc_flips_rendering_wait(1, 3, &wait, &error), AGC_DRIVER_INVALID,
          "VideoOut's rendering wait: capture refused -1");
   output->capture_code = AGC_DRIVER_OK;
   output->null_wait = true;
   EXPECT(agc_flips_rendering_wait(1, 3, &wait, &error), AGC_DRIVER_FAILED,
          "VideoOut returned a null rendering wait");
   output->null_wait = false;
   CHECK(wait == NULL);
   EXPECT(agc_flips_rendering_wait(1, 3, NULL, &error), AGC_DRIVER_INVALID, "null rendering wait");
   EXPECT(agc_flips_rendering_wait(4, 3, &wait, &error), AGC_DRIVER_INVALID, "video output 4 is not registered");
   EXPECT(agc_flips_rendering_wait(AGC_VIDEO_OUTPUT_SLOTS, 3, &wait, &error), AGC_DRIVER_INVALID,
          "video output handle 64 is out of range");

   uint32_t at = call_count;
   CHECK_EQ(agc_flips_rendering_wait(1, 3, &wait, &error), AGC_DRIVER_OK);
   CALLED(at, CALL_CAPTURE, 1, 3, 0);
   CHECK(wait != NULL);
   CHECK_EQ(agc_rendering_wait_for(wait, 16, &error), 1);
   CALLED(at + 1, CALL_WAIT_FOR, 1, 3, 16);
   output->wait_code = 0;
   CHECK_EQ(agc_rendering_wait_for(wait, 0, &error), 0);
   output->wait_code = AGC_DRIVER_SHUTDOWN;
   EXPECT(agc_rendering_wait_for(wait, 1, &error), AGC_DRIVER_SHUTDOWN,
          "VideoOut's rendering wait: wait refused -3");
   output->wait_code = 5;
   EXPECT(agc_rendering_wait_for(wait, 1, &error), AGC_DRIVER_FAILED, "VideoOut's rendering wait: wait refused 5");
   EXPECT(agc_rendering_wait_for(NULL, 1, &error), AGC_DRIVER_INVALID, "null rendering wait");
   CHECK_EQ(calls_of(CALL_WAIT_FOR), 4);

   at = call_count;
   agc_rendering_wait_release(wait);
   agc_rendering_wait_release(NULL);
   CALLED(at, CALL_WAIT_RELEASE, 1, 3, 0);
   CHECK_EQ(call_count, at + 1);
   CHECK_EQ(output->waits[0].releases, 1);
   close_output(1, output);
}

static void
test_describe_buffer(void)
{
   fresh();
   struct test_output *output = &outputs[0];
   struct agc_display_buffer buffer = {0};
   open_output(1, output);

   CHECK(agc_flips_describe_buffer(1, 2, &buffer));
   CALLED(0, CALL_DESCRIBE, 1, 2, 0);
   CHECK_EQ(buffer.address, UINT64_C(0x100000002));
   CHECK_EQ(buffer.pixel_format, 0x8100);
   CHECK_EQ(buffer.width, 3840);
   CHECK_EQ(buffer.height, 2160);
   CHECK_EQ(buffer.tiling_mode, 2);
   CHECK_EQ(buffer.pitch_in_pixel, 3840);
   CHECK_EQ(buffer.dcc_address, UINT64_C(0x200000000));
   CHECK_EQ(buffer.dcc_clear_color, 7);
   CHECK_EQ(buffer.alignment, 0x10000);

   struct agc_display_buffer untouched;
   memset(&untouched, 0x5a, sizeof(untouched));
   buffer = untouched;
   output->describes = false;
   CHECK(!agc_flips_describe_buffer(1, -1, &buffer));
   CALLED(1, CALL_DESCRIBE, 1, -1, 0);
   CHECK(memcmp(&buffer, &untouched, sizeof(buffer)) == 0);
   output->describes = true;
   CHECK(!agc_flips_describe_buffer(9, 0, &buffer));
   CHECK(!agc_flips_describe_buffer(AGC_VIDEO_OUTPUT_SLOTS, 0, &buffer));
   CHECK(!agc_flips_describe_buffer(1, 0, NULL));
   CHECK_EQ(call_count, 2);
   close_output(1, output);
}

struct contain_case {
   uint32_t source_width;
   uint32_t source_height;
   uint32_t bounds_width;
   uint32_t bounds_height;
   bool allow_upscale;
   uint64_t scale_bits;
   uint32_t width;
   uint32_t height;
};

struct rect_case {
   uint32_t source_width;
   uint32_t source_height;
   uint32_t destination_width;
   uint32_t destination_height;
   int32_t x;
   int32_t y;
   uint32_t width;
   uint32_t height;
};

struct extent_case {
   uint32_t source_width;
   uint32_t source_height;
   uint32_t length;
   uint32_t width_for_height;
   uint32_t height_for_width;
};

static const struct contain_case contain_cases[] = {
   {1920, 1080, 1280, 720, true, UINT64_C(0x3fe5555555555555), 1280, 720},
   {1920, 1080, 3840, 2160, true, UINT64_C(0x4000000000000000), 3840, 2160},
   {1920, 1080, 3840, 2160, false, UINT64_C(0x3ff0000000000000), 1920, 1080},
   {1920, 1080, 1000, 1000, true, UINT64_C(0x3fe0aaaaaaaaaaab), 1000, 563},
   {1920, 1080, 1000, 1000, false, UINT64_C(0x3fe0aaaaaaaaaaab), 1000, 563},
   {3840, 2160, 1366, 768, true, UINT64_C(0x3fd6c16c16c16c17), 1365, 768},
   {1280, 720, 1920, 1200, true, UINT64_C(0x3ff8000000000000), 1920, 1080},
   {1280, 720, 1920, 1200, false, UINT64_C(0x3ff0000000000000), 1280, 720},
   {3, 7, 1000, 1000, true, UINT64_C(0x4061db6db6db6db7), 429, 1000},
   {7, 3, 10, 10, true, UINT64_C(0x3ff6db6db6db6db7), 10, 4},
   {4000, 3000, 1, 1, true, UINT64_C(0x3f30624dd2f1a9fc), 1, 1},
   {1, 1, UINT32_MAX, UINT32_MAX, true, UINT64_C(0x41efffffffe00000), UINT32_MAX, UINT32_MAX},
   {3, 1, UINT32_MAX, UINT32_MAX, true, UINT64_C(0x41d5555555400000), UINT32_MAX, 1431655765},
   {UINT32_MAX, 1, UINT32_MAX, UINT32_MAX, true, UINT64_C(0x3ff0000000000000), UINT32_MAX, 1},
   {UINT32_MAX, UINT32_MAX, 1, UINT32_MAX, false, UINT64_C(0x3df0000000100000), 1, 1},
   {0, 1080, 1280, 720, true, UINT64_C(0x3ff0000000000000), 1, 1080},
   {1920, 0, 1280, 720, true, UINT64_C(0x3ff0000000000000), 1920, 1},
   {1920, 1080, 0, 720, true, UINT64_C(0x3ff0000000000000), 1920, 1080},
   {1920, 1080, 1280, 0, true, UINT64_C(0x3ff0000000000000), 1920, 1080},
   {0, 0, 0, 0, false, UINT64_C(0x3ff0000000000000), 1, 1},
   {1921, 1081, 1279, 721, true, UINT64_C(0x3fe54e39d627b670), 1279, 720},
   {333, 777, 555, 999, true, UINT64_C(0x3ff4924924924925), 428, 999},
   {5, 3, 2, 2, true, UINT64_C(0x3fd999999999999a), 2, 1},
   {2, 3, 5, 5, true, UINT64_C(0x3ffaaaaaaaaaaaab), 3, 5},
};

static const struct rect_case rect_cases[] = {
   {1920, 1080, 1280, 720, 0, 0, 1280, 720},
   {1920, 1080, 1000, 1000, 0, 218, 1000, 563},
   {1080, 1920, 1000, 1000, 218, 0, 563, 1000},
   {1280, 720, 1920, 1200, 0, 60, 1920, 1080},
   {3840, 2160, 1366, 768, 0, 0, 1365, 768},
   {4, 3, UINT32_MAX, UINT32_MAX, 0, 536870912, UINT32_MAX, 3221225471u},
   {1, UINT32_MAX, UINT32_MAX, 1, 2147483647, 0, 1, 1},
   {0, 0, 640, 480, 319, 239, 1, 1},
   {1920, 1080, 0, 0, -960, -540, 1920, 1080},
   {333, 777, 555, 999, 63, 0, 428, 999},
   {16, 9, 1, 1, 0, 0, 1, 1},
};

static const struct extent_case extent_cases[] = {
   {1920, 1080, 720, 1280, 405},
   {1920, 1080, 1, 2, 1},
   {1920, 1080, 0, 1, 1},
   {16, 9, 1000, 1778, 563},
   {9, 16, 1000, 563, 1778},
   {3, 7, 5, 2, 12},
   {7, 3, 5, 12, 2},
   {UINT32_MAX, 1, UINT32_MAX, UINT32_MAX, 1},
   {1, UINT32_MAX, UINT32_MAX, 1, UINT32_MAX},
   {1, UINT32_MAX, 1, 1, UINT32_MAX},
   {0, 1080, 720, 1, 1},
   {1920, 0, 720, 1, 1},
   {2, 4, 3, 2, 6},
   {2, 4, 1, 1, 2},
   {333, 777, 555, 238, 1295},
};

static void
test_aspect_fit(void)
{
   for (unsigned i = 0; i < ARRAY_COUNT(contain_cases); i++) {
      const struct contain_case *c = &contain_cases[i];
      double scale = ComputeContainScale(c->source_width, c->source_height, c->bounds_width, c->bounds_height,
                                         c->allow_upscale);
      uint64_t bits;
      memcpy(&bits, &scale, sizeof(bits));
      CHECK_EQ(bits, c->scale_bits);
      struct agc_aspect_fit_size size =
         ComputeContainSize(c->source_width, c->source_height, c->bounds_width, c->bounds_height, c->allow_upscale);
      CHECK_EQ(size.width, c->width);
      CHECK_EQ(size.height, c->height);
   }
   for (unsigned i = 0; i < ARRAY_COUNT(rect_cases); i++) {
      const struct rect_case *c = &rect_cases[i];
      struct agc_aspect_fit_rect rect =
         ComputeContainRect(c->source_width, c->source_height, c->destination_width, c->destination_height);
      CHECK_EQ(rect.x, c->x);
      CHECK_EQ(rect.y, c->y);
      CHECK_EQ(rect.width, c->width);
      CHECK_EQ(rect.height, c->height);
   }
   for (unsigned i = 0; i < ARRAY_COUNT(extent_cases); i++) {
      const struct extent_case *c = &extent_cases[i];
      CHECK_EQ(ComputeWidthForHeight(c->source_width, c->source_height, c->length), c->width_for_height);
      CHECK_EQ(ComputeHeightForWidth(c->source_width, c->source_height, c->length), c->height_for_width);
   }

   uint32_t reports = agc_report_count();
   ComputeContainScale(1920, 1080, 0, 0, true);
   ComputeContainSize(1920, 1080, 0, 720, false);
   ComputeWidthForHeight(1920, 1080, 0);
   CHECK_EQ(agc_report_count(), reports);
   ComputeContainScale(0, 1080, 1280, 720, true);
   CHECK_EQ(agc_report_count(), reports + 1);
   ComputeContainSize(1920, 0, 1280, 720, true);
   CHECK_EQ(agc_report_count(), reports + 2);
   ComputeContainRect(0, 0, 640, 480);
   CHECK_EQ(agc_report_count(), reports + 3);
   ComputeWidthForHeight(0, 1080, 720);
   ComputeHeightForWidth(1920, 0, 720);
   CHECK_EQ(agc_report_count(), reports + 5);
}

static bool
hdr_is(bool available, bool surface_hdr10, bool display_hdr, float max_luminance, uint64_t generation,
       uint64_t display_generation)
{
   struct agc_host_hdr_status status;
   memset(&status, 0xff, sizeof(status));
   AgcDriverHostHdr(&status);
   if (status.available == available && status.surface_hdr10 == surface_hdr10 && status.display.hdr == display_hdr &&
       (status.display.max_luminance == max_luminance || max_luminance != max_luminance) &&
       status.generation == generation && status.display_generation == display_generation)
      return true;
   fprintf(stderr, "   hdr is available %d surface %d display %d max %g generation %llu display generation %llu\n",
           status.available, status.surface_hdr10, status.display.hdr, status.display.max_luminance,
           (unsigned long long)status.generation, (unsigned long long)status.display_generation);
   return false;
}

static void
test_hdr(void)
{
   fresh();
   struct agc_host_hdr_status status;
   struct agc_host_hdr_status zero;
   memset(&status, 0xff, sizeof(status));
   memset(&zero, 0, sizeof(zero));
   AgcDriverHostHdr(&status);
   CHECK(memcmp(&status, &zero, sizeof(status)) == 0);
   CHECK(!AgcDriverPresentingHdr10());

   struct agc_host_display_hdr display = {
      .hdr = true, .max_luminance = 1000.0f, .min_luminance = 0.01f, .max_full_frame_luminance = 600.0f};
   AgcDriverSetHostDisplayHdr(&display);
   CHECK(hdr_is(false, false, true, 1000.0f, 0, 1));
   AgcDriverSetSurfaceHdr10(true);
   CHECK(hdr_is(true, true, true, 1000.0f, 1, 1));
   AgcDriverSetSurfaceHdr10(true);
   AgcDriverSetHostDisplayHdr(&display);
   AgcDriverSetHostDisplayHdr(NULL);
   AgcDriverHostHdr(NULL);
   CHECK(hdr_is(true, true, true, 1000.0f, 1, 1));
   display.max_luminance = 800.0f;
   AgcDriverSetHostDisplayHdr(&display);
   CHECK(hdr_is(true, true, true, 800.0f, 2, 2));
   AgcDriverSetSurfaceHdr10(false);
   CHECK(hdr_is(false, false, true, 800.0f, 3, 2));
   display.max_luminance = 900.0f;
   AgcDriverSetHostDisplayHdr(&display);
   CHECK(hdr_is(false, false, true, 900.0f, 3, 3));
   display = (struct agc_host_display_hdr){0};
   AgcDriverSetHostDisplayHdr(&display);
   CHECK(hdr_is(false, false, false, 0.0f, 3, 4));
   AgcDriverNoteReportedToneMapPeak(450.5f);
   CHECK(hdr_is(false, false, false, 0.0f, 3, 4));
   AgcDriverHostHdr(&status);
   CHECK(status.reported_tone_map_peak == 450.5f);
   CHECK(status.display.min_luminance == 0.0f && status.display.max_full_frame_luminance == 0.0f);
   display = (struct agc_host_display_hdr){
      .hdr = true, .max_luminance = 1000.0f, .min_luminance = 0.01f, .max_full_frame_luminance = 600.0f};
   AgcDriverSetHostDisplayHdr(&display);
   AgcDriverHostHdr(&status);
   CHECK(status.display.min_luminance == 0.01f && status.display.max_full_frame_luminance == 600.0f);

   fresh();
   force_sdr(true);
   display = (struct agc_host_display_hdr){.hdr = true, .max_luminance = 1000.0f, .max_full_frame_luminance = 1000.0f};
   AgcDriverSetSurfaceHdr10(true);
   AgcDriverSetHostDisplayHdr(&display);
   CHECK(hdr_is(false, true, true, 1000.0f, 0, 1));
   force_sdr(false);
   AgcDriverSetSurfaceHdr10(true);
   CHECK(hdr_is(true, true, true, 1000.0f, 1, 1));
   force_sdr(true);
   AgcDriverSetSurfaceHdr10(true);
   CHECK(hdr_is(false, true, true, 1000.0f, 2, 1));
   force_sdr(false);

   fresh();
   display = (struct agc_host_display_hdr){.hdr = true, .max_luminance = NAN};
   AgcDriverSetSurfaceHdr10(true);
   AgcDriverSetHostDisplayHdr(&display);
   CHECK(hdr_is(true, true, true, NAN, 1, 1));
   AgcDriverSetHostDisplayHdr(&display);
   CHECK(hdr_is(true, true, true, NAN, 2, 2));

   fresh();
   struct agc_driver_error error = {0};
   struct agc_presentation_window window = {.context = &window, .width = 1280, .height = 720, .native_window = 2};
   agc_videoout_set_presenter(&recording_presenter);
   AgcDriverSetSurfaceHdr10(true);
   presenting.attach_code = AGC_DRIVER_FAILED;
   CHECK_EQ(AgcDriverAttachWindow(&window, &error), AGC_DRIVER_FAILED);
   CHECK(hdr_is(false, true, false, 0.0f, 0, 0));
   presenting.attach_code = AGC_DRIVER_OK;
   CHECK_EQ(AgcDriverAttachWindow(&window, &error), AGC_DRIVER_OK);
   CHECK(hdr_is(false, false, false, 0.0f, 0, 0));
}

static void
test_presentation(void)
{
   fresh();
   struct agc_driver_error error = {0};
   struct agc_presentation_window window = {
      .context = &window, .width = 1920, .height = 1080, .native_window = 0x1234};
   struct agc_display_buffer buffer = {.address = 0x1000, .width = 1920, .height = 1080};
   struct agc_display_buffer plane = {.address = 0x2000, .width = 640, .height = 360};
   int completion = 41;

   EXPECT(AgcDriverAttachWindow(&window, &error), AGC_DRIVER_FAILED, "no presenter is installed");
   EXPECT(AgcDriverCaptureOverlay(&buffer, &error), AGC_DRIVER_FAILED, "no presenter is installed");
   EXPECT(AgcDriverPresentBuffer(&window, &buffer, &plane, gpu_ready, &completion, &error), AGC_DRIVER_FAILED,
          "no presenter is installed");
   EXPECT(AgcDriverPresentClear(&window, true, NULL, gpu_ready, &completion, &error), AGC_DRIVER_FAILED,
          "no presenter is installed");
   CHECK_EQ(AgcDriverDisplayBufferSize(&buffer), 0);
   uint32_t reports = agc_report_count();
   AgcDriverReleaseWindow(window.context);
   CHECK_EQ(agc_report_count(), reports + 1);

   EXPECT(AgcDriverAttachWindow(NULL, &error), AGC_DRIVER_INVALID, "null window");
   EXPECT(AgcDriverPresentBuffer(NULL, &buffer, NULL, gpu_ready, &completion, &error), AGC_DRIVER_INVALID,
          "null window");
   EXPECT(AgcDriverPresentBuffer(&window, NULL, NULL, gpu_ready, &completion, &error), AGC_DRIVER_INVALID,
          "null buffer");
   EXPECT(AgcDriverPresentBuffer(&window, &buffer, NULL, NULL, &completion, &error), AGC_DRIVER_INVALID,
          "missing GPU completion callback");
   EXPECT(AgcDriverPresentBuffer(&window, &buffer, NULL, gpu_ready, NULL, &error), AGC_DRIVER_INVALID,
          "missing GPU completion callback");
   EXPECT(AgcDriverPresentClear(NULL, false, NULL, gpu_ready, &completion, &error), AGC_DRIVER_INVALID,
          "null window");
   EXPECT(AgcDriverPresentClear(&window, false, NULL, NULL, &completion, &error), AGC_DRIVER_INVALID,
          "missing GPU completion callback");
   CHECK_EQ(call_count, 0);
   CHECK_EQ(agc_videoout_frames_presented(), 0);

   agc_videoout_set_presenter(&recording_presenter);
   CHECK_EQ(AgcDriverAttachWindow(&window, &error), AGC_DRIVER_OK);
   CALLED(0, CALL_ATTACH, 0, 0x1234, 0);
   presenting.attach_code = AGC_DRIVER_FAILED;
   EXPECT(AgcDriverAttachWindow(&window, &error), AGC_DRIVER_FAILED, "attach refused -2");
   CHECK_EQ(AgcDriverAttachWindow(&window, NULL), AGC_DRIVER_FAILED);

   uint32_t at = call_count;
   presenting.shows = true;
   CHECK_EQ(AgcDriverPresentBuffer(&window, &buffer, NULL, gpu_ready, &completion, &error), AGC_DRIVER_OK);
   CALLED(at, CALL_PRESENT, 0, 0, 0);
   CALLED(at + 1, CALL_GPU_READY, 0, 41, 0);
   CHECK_EQ(call_count, at + 2);
   CHECK(presenting.window == &window && presenting.buffer == &buffer && presenting.plane == NULL);
   CHECK_EQ(agc_videoout_frames_presented(), 1);

   completion = 42;
   CHECK_EQ(AgcDriverPresentBuffer(&window, &buffer, &plane, gpu_ready, &completion, &error), AGC_DRIVER_OK);
   CALLED(at + 3, CALL_GPU_READY, 0, 42, 0);
   CHECK(presenting.plane == &plane);
   CHECK_EQ(agc_videoout_frames_presented(), 2);

   presenting.shows = false;
   CHECK_EQ(AgcDriverPresentBuffer(&window, &buffer, NULL, gpu_ready, &completion, &error), AGC_DRIVER_OK);
   CALLED(at + 4, CALL_PRESENT, 0, 0, 0);
   CALLED(at + 5, CALL_GPU_READY, 0, 42, 0);
   CHECK_EQ(agc_videoout_frames_presented(), 2);

   presenting.present_code = AGC_DRIVER_SHUTDOWN;
   EXPECT(AgcDriverPresentBuffer(&window, &buffer, NULL, gpu_ready, &completion, &error), AGC_DRIVER_SHUTDOWN,
          "present refused -3");
   CHECK_EQ(AgcDriverPresentClear(&window, true, NULL, gpu_ready, &completion, NULL), AGC_DRIVER_SHUTDOWN);
   CHECK_EQ(call_count, at + 8);
   CHECK_EQ(calls_of(CALL_GPU_READY), 3);
   CHECK_EQ(agc_videoout_frames_presented(), 2);
   presenting.present_code = AGC_DRIVER_OK;

   presenting.shows = true;
   completion = 43;
   CHECK_EQ(AgcDriverPresentClear(&window, true, &plane, gpu_ready, &completion, &error), AGC_DRIVER_OK);
   CALLED(at + 8, CALL_PRESENT, 0, 0, 0);
   CALLED(at + 9, CALL_GPU_READY, 0, 43, 0);
   CHECK(presenting.window == &window && presenting.buffer == NULL && presenting.plane == &plane);
   CHECK_EQ(AgcDriverPresentClear(&window, false, NULL, gpu_ready, &completion, &error), AGC_DRIVER_OK);
   CHECK(presenting.buffer == NULL && presenting.plane == NULL);
   CHECK_EQ(agc_videoout_frames_presented(), 4);

   at = call_count;
   CHECK_EQ(AgcDriverCaptureOverlay(&buffer, &error), AGC_DRIVER_OK);
   CHECK(presenting.overlay == &buffer);
   CHECK_EQ(AgcDriverCaptureOverlay(NULL, &error), AGC_DRIVER_OK);
   CHECK(presenting.overlay == NULL);
   presenting.overlay_code = AGC_DRIVER_FAILED;
   EXPECT(AgcDriverCaptureOverlay(&buffer, &error), AGC_DRIVER_FAILED, "overlay refused -2");
   CHECK_EQ(call_count, at + 3);

   reports = agc_report_count();
   presenting.detaches = true;
   AgcDriverReleaseWindow(window.context);
   CALLED(at + 3, CALL_DETACH, 0, (int64_t)(intptr_t)window.context, 0);
   CHECK_EQ(agc_report_count(), reports);
   presenting.detaches = false;
   AgcDriverReleaseWindow(window.context);
   CHECK_EQ(agc_report_count(), reports + 1);

   at = call_count;
   presenting.buffer_bytes = 8294400;
   CHECK_EQ(AgcDriverDisplayBufferSize(&buffer), 8294400);
   CALLED(at, CALL_BUFFER_SIZE, 0, 1920, 1080);
   CHECK_EQ(AgcDriverDisplayBufferSize(NULL), 0);
   CHECK_EQ(call_count, at + 1);
}

#ifdef _WIN32
static bool
guest_memory_is(const void *pointer, size_t bytes, size_t alignment, bool writable, int expected)
{
   struct agc_driver_error error = {0};
   char refusal[256];
   int code = AgcDriverCheckGuestMemory(pointer, bytes, alignment, writable, &error);
   snprintf(refusal, sizeof(refusal), "guest memory 0x%llx+0x%zx is not mapped with the access asked for",
            (unsigned long long)(uintptr_t)pointer, bytes);
   if (code == expected && (code != AGC_DRIVER_INVALID || strcmp(error.message, refusal) == 0))
      return true;
   fprintf(stderr, "   guest memory check answered %d \"%s\"\n", code, error.message);
   return false;
}

static void
test_guest_memory(void)
{
   fresh();
   struct agc_driver_error error = {0};
   SYSTEM_INFO host;
   DWORD previous = 0;
   char refusal[256];
   GetSystemInfo(&host);
   size_t page = host.dwPageSize;
   uint8_t *block = VirtualAlloc(NULL, 4 * page, MEM_RESERVE, PAGE_NOACCESS);
   CHECK(block != NULL);
   if (!block)
      return;

   EXPECT(AgcDriverCheckGuestMemory(NULL, 16, 8, false, &error), AGC_DRIVER_INVALID,
          "null or misaligned guest address 0x0");
   snprintf(refusal, sizeof(refusal), "null or misaligned guest address 0x%llx",
            (unsigned long long)(uintptr_t)(block + 1));
   EXPECT(AgcDriverCheckGuestMemory(block + 1, 16, 8, false, &error), AGC_DRIVER_INVALID, refusal);
   snprintf(refusal, sizeof(refusal), "null or misaligned guest address 0x%llx",
            (unsigned long long)(uintptr_t)block);
   EXPECT(AgcDriverCheckGuestMemory(block, 16, 0, false, &error), AGC_DRIVER_INVALID, refusal);

   CHECK(guest_memory_is(block, 16, 8, false, AGC_DRIVER_INVALID));
   CHECK(guest_memory_is(block, 0, 8, true, AGC_DRIVER_OK));
   CHECK(guest_memory_is(block, SIZE_MAX, 8, false, AGC_DRIVER_INVALID));

   CHECK(VirtualAlloc(block, page, MEM_COMMIT, PAGE_READONLY) == block);
   CHECK(guest_memory_is(block, 16, 8, false, AGC_DRIVER_OK));
   CHECK(guest_memory_is(block + page - 16, 16, 16, false, AGC_DRIVER_OK));
   CHECK(guest_memory_is(block, 16, 8, true, AGC_DRIVER_INVALID));
   CHECK(VirtualProtect(block, page, PAGE_READWRITE, &previous));
   CHECK(guest_memory_is(block, 16, 8, true, AGC_DRIVER_OK));
   CHECK(guest_memory_is(block, page + 1, 8, false, AGC_DRIVER_INVALID));
   CHECK(VirtualAlloc(block + page, page, MEM_COMMIT, PAGE_READWRITE) == block + page);
   CHECK(guest_memory_is(block, 2 * page, 8, true, AGC_DRIVER_OK));
   CHECK(guest_memory_is(block + 8, 2 * page - 8, 8, true, AGC_DRIVER_OK));
   CHECK(guest_memory_is(block, 4 * page, 8, false, AGC_DRIVER_INVALID));

   CHECK(VirtualProtect(block, page, PAGE_NOACCESS, &previous));
   CHECK(guest_memory_is(block, 16, 8, false, AGC_DRIVER_OK));
   libkernel.generation++;
   CHECK(guest_memory_is(block, 16, 8, false, AGC_DRIVER_INVALID));
   CHECK(guest_memory_is(block + page, 16, 8, true, AGC_DRIVER_OK));
   CHECK(VirtualProtect(block, page, PAGE_READWRITE, &previous));
   CHECK(guest_memory_is(block, 2 * page, 8, true, AGC_DRIVER_OK));

   CHECK(VirtualProtect(block + page, page, PAGE_READWRITE | PAGE_GUARD, &previous));
   libkernel.generation++;
   CHECK(guest_memory_is(block + page, 16, 8, false, AGC_DRIVER_INVALID));
   CHECK(guest_memory_is(block, 16, 8, true, AGC_DRIVER_OK));
   CHECK(VirtualFree(block + page, page, MEM_DECOMMIT));
   libkernel.generation++;
   CHECK(guest_memory_is(block, page, 8, true, AGC_DRIVER_OK));
   CHECK(guest_memory_is(block, page + 1, 8, true, AGC_DRIVER_INVALID));

   agc_anyps5_set(NULL);
   EXPECT(AgcDriverCheckGuestMemory(block, 16, 8, false, &error), AGC_DRIVER_FAILED,
          "AnyPS5's guest memory is not available");
   CHECK_EQ(AgcDriverCheckGuestMemory(block, 16, 8, false, NULL), AGC_DRIVER_FAILED);
   CHECK(VirtualFree(block, 0, MEM_RELEASE));
}
#endif

static void
test_guest_label(void)
{
   _Alignas(8) uint64_t labels[2] = {1, 2};
   uint32_t reports = agc_report_count();
   AgcDriverStoreGuestLabel(&labels[0], UINT64_C(0x1122334455667788));
   CHECK_EQ(labels[0], UINT64_C(0x1122334455667788));
   CHECK_EQ(labels[1], 2);
   AgcDriverStoreGuestLabel(&labels[1], 0);
   CHECK_EQ(labels[1], 0);
   CHECK_EQ(agc_report_count(), reports);

   AgcDriverStoreGuestLabel(NULL, 5);
   CHECK_EQ(agc_report_count(), reports + 1);
   AgcDriverStoreGuestLabel((uint64_t *)((uintptr_t)labels + 4), 9);
   CHECK_EQ(agc_report_count(), reports + 2);
   CHECK_EQ(labels[0], UINT64_C(0x1122334455667788));
   CHECK_EQ(labels[1], 0);
}

static void
test_idle_exports(void)
{
   fresh();
   AgcDriverSetSystemOverlay(NULL);
   AgcDriverSetSystemDialog(NULL);
   AgcDriverPadStatusPoll(NULL);
   AgcDriverPadStatusShow();
   AgcDriverTogglePerformanceOverlay();
   AgcDriverRequestScreenshot();
   AgcDriverRaiseWorkerThreadPriority("presenter");
   AgcDriverRaiseWorkerThreadPriority(NULL);
   AgcDriverLockVulkanLoader();
   AgcDriverUnlockVulkanLoader();
   AgcDriverTraceCount("scope");
   AgcDriverTraceCount(NULL);
   CHECK(!AgcDriverTraceRecording());
   CHECK(AgcDriverTraceThread() == NULL);
   CHECK(!AgcDriverPresentingHdr10());
   CHECK_EQ(call_count, 0);
}

static void
test_reset(void)
{
   fresh();
   struct test_output *first = &outputs[0];
   struct test_output *second = &outputs[1];
   struct agc_driver_error error = {0};
   uint32_t token = 0;
   open_output(1, first);
   open_output(2, second);
   struct agc_flip_info flip = flip_of(1, 5);
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   CHECK_EQ(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_OK);
   CHECK_EQ(agc_events_add(0x90, 0, 0, NULL, &error), AGC_DRIVER_OK);
   AgcDriverSetSurfaceHdr10(true);

   uint32_t at = call_count;
   agc_videoout_reset();
   CALLED(at, CALL_FLIP_RELEASE, 1, 5, 0);
   CALLED(at + 1, CALL_RELEASE, 1, 0, 0);
   CALLED(at + 2, CALL_RELEASE, 2, 0, 0);
   CHECK_EQ(call_count, at + 3);
   CHECK(hdr_is(false, false, false, 0.0f, 0, 0));
   agc_events_interrupt(0, 1);
   CHECK_EQ(call_count, at + 3);
   EXPECT(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_INVALID, "video output 1 is not registered");

   open_output(1, &outputs[2]);
   agc_flips_fail("broken");
   agc_flips_shutdown();
   CHECK_EQ(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_SHUTDOWN);
   agc_videoout_reset();
   CHECK(agc_flips_failure() == NULL);
   CHECK_EQ(outputs[2].releases, 1);

   open_output(1, &outputs[3]);
   at = call_count;
   CHECK_EQ(AgcDriverReserveEopFlip(&flip, &token, &error), AGC_DRIVER_OK);
   CHECK_EQ(token, TOKEN_BASE);
   CHECK_EQ(AgcDriverSubmitFlip(&flip, &error), AGC_DRIVER_OK);
   CALLED(at + 3, CALL_FLIP_READY, 4, 5, 1);
}

static int
stress_reserve(void *self, const struct agc_flip_info *info, bool limited, struct agc_videoout_request **request,
               struct agc_driver_error *error)
{
   (void)self;
   (void)info;
   (void)limited;
   (void)error;
   *request = (struct agc_videoout_request *)(uintptr_t)p_atomic_inc_return(&stress.reserved);
   return AGC_DRIVER_OK;
}

static int
stress_room(void *self, struct agc_driver_error *error)
{
   (void)self;
   (void)error;
   return AGC_DRIVER_OK;
}

static int
stress_capture(void *self, uint32_t index, struct agc_videoout_wait **wait, struct agc_driver_error *error)
{
   (void)self;
   (void)index;
   (void)wait;
   (void)error;
   return AGC_DRIVER_FAILED;
}

static bool
stress_describe(void *self, int32_t index, struct agc_display_buffer *buffer)
{
   (void)self;
   (void)index;
   (void)buffer;
   return false;
}

static void
stress_fail(void *self, const char *message)
{
   (void)self;
   (void)message;
}

static void
stress_release(void *self)
{
   (void)self;
   p_atomic_inc(&stress.unregistered);
}

static int
stress_flip_ready(struct agc_videoout_request *request, uint64_t frame, struct agc_driver_error *error)
{
   (void)request;
   (void)frame;
   (void)error;
   return AGC_DRIVER_OK;
}

static void
stress_flip_fail(struct agc_videoout_request *request, const char *message)
{
   (void)request;
   (void)message;
}

static void
stress_flip_release(struct agc_videoout_request *request)
{
   (void)request;
   p_atomic_inc(&stress.released);
}

static int
stress_wait_for(struct agc_videoout_wait *wait, uint32_t milliseconds, struct agc_driver_error *error)
{
   (void)wait;
   (void)milliseconds;
   (void)error;
   return 1;
}

static void
stress_wait_release(struct agc_videoout_wait *wait)
{
   (void)wait;
}

static int
stress_registrar_main(void *data)
{
   (void)data;
   for (unsigned round = 0; round < STRESS_ROUNDS; round++) {
      struct agc_video_output table = {
         .self = &stress_identities[round],
         .reserve = stress_reserve,
         .wait_for_flip_room = stress_room,
         .capture_rendering_wait = stress_capture,
         .describe_buffer = stress_describe,
         .fail = stress_fail,
         .release = stress_release,
         .flip_ready = stress_flip_ready,
         .flip_fail = stress_flip_fail,
         .flip_release = stress_flip_release,
         .rendering_wait_for = stress_wait_for,
         .rendering_wait_release = stress_wait_release,
      };
      struct agc_flip_info flip = flip_of(1, round);
      uint32_t token = 0;
      if (AgcDriverRegisterVideoOutput(1, &table, NULL) != AGC_DRIVER_OK) {
         p_atomic_inc(&stress.unexpected);
         continue;
      }
      p_atomic_inc(&stress.registered);
      int code = AgcDriverReserveEopFlip(&flip, &token, NULL);
      if (code != AGC_DRIVER_OK && code != AGC_DRIVER_FAILED)
         p_atomic_inc(&stress.unexpected);
      thrd_yield();
      if (AgcDriverUnregisterVideoOutput(1, table.self, NULL) != AGC_DRIVER_OK)
         p_atomic_inc(&stress.unexpected);
   }
   return 0;
}

static int
stress_reserver_main(void *data)
{
   (void)data;
   while (!p_atomic_read(&stress.stop)) {
      struct agc_flip_info flip = flip_of(1, -1);
      uint32_t token = 0;
      int code = AgcDriverReserveEopFlip(&flip, &token, NULL);
      if (code == AGC_DRIVER_OK) {
         struct agc_event event = eop_event(0, token);
         if (agc_flips_deliver(&event, 1, NULL) != AGC_DRIVER_OK)
            p_atomic_inc(&stress.unexpected);
      } else if (code != AGC_DRIVER_INVALID && code != AGC_DRIVER_FAILED) {
         p_atomic_inc(&stress.unexpected);
      }
   }
   return 0;
}

static int
stress_flipper_main(void *data)
{
   (void)data;
   while (!p_atomic_read(&stress.stop)) {
      struct agc_event event = flip_event(1, -2);
      int code = agc_flips_deliver(&event, 1, NULL);
      if (code != AGC_DRIVER_OK && code != AGC_DRIVER_INVALID)
         p_atomic_inc(&stress.unexpected);
   }
   return 0;
}

static void
test_concurrent_use(void)
{
   fresh();
   thrd_t registrar;
   thrd_t reserver;
   thrd_t flipper;
   memset(&stress, 0, sizeof(stress));
   CHECK(thrd_create(&reserver, stress_reserver_main, NULL) == thrd_success);
   CHECK(thrd_create(&flipper, stress_flipper_main, NULL) == thrd_success);
   CHECK(thrd_create(&registrar, stress_registrar_main, NULL) == thrd_success);
   CHECK(thrd_join(registrar, NULL) == thrd_success);
   p_atomic_set(&stress.stop, 1);
   CHECK(thrd_join(reserver, NULL) == thrd_success);
   CHECK(thrd_join(flipper, NULL) == thrd_success);

   CHECK_EQ(stress.unexpected, 0);
   CHECK_EQ(stress.registered, STRESS_ROUNDS);
   CHECK_EQ(stress.unregistered, STRESS_ROUNDS);
   CHECK(stress.reserved > 0);
   CHECK_EQ(stress.released, stress.reserved);
}

int
main(void)
{
   test_layout();
   test_registration();
   test_release_waits_for_users();
   test_cpu_flip();
   test_eop_reservation();
   test_unregister_drops_reservations();
   test_delivery_order();
   test_delivery_of_reserved_flips();
   test_delivery_failures();
   test_eop_interrupts();
   test_sticky_failure();
   test_shutdown();
   test_shutdown_races();
   test_reservation_during_room_wait();
   test_events();
   test_rendering_wait();
   test_describe_buffer();
   test_aspect_fit();
   test_hdr();
   test_presentation();
#ifdef _WIN32
   test_guest_memory();
#endif
   test_guest_label();
   test_idle_exports();
   test_reset();
   test_concurrent_use();
   agc_videoout_reset();
   agc_anyps5_set(NULL);
   return TEST_RESULT();
}
