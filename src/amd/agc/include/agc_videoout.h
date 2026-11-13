#ifndef AGC_VIDEOOUT_H
#define AGC_VIDEOOUT_H

#include "agc_types.h"

enum {
   AGC_DRIVER_OK = 0,
   AGC_DRIVER_INVALID = -1,
   AGC_DRIVER_FAILED = -2,
   AGC_DRIVER_SHUTDOWN = -3,
   AGC_DRIVER_FULL = -4,
   AGC_DRIVER_ALREADY_RESERVED = -5,
};

#define AGC_DRIVER_EOP_FLIP_TOKEN_BASE UINT32_C(0x04000000)
#define AGC_DRIVER_EOP_FLIP_TOKEN_LIMIT UINT32_C(0x07ffffff)
#define AGC_DRIVER_PRESENTATION_NATIVE_WINDOW UINT32_C(1)

#define AGC_VIDEO_OUTPUT_SLOTS 64
#define AGC_EOP_FLIP_SLOTS 256
#define AGC_EVENT_REGISTRATION_SLOTS 128

struct agc_driver_error {
   char message[256];
};

struct agc_flip_info {
   uint32_t handle;
   int32_t index;
   uint32_t mode;
   uint32_t reserved;
   int64_t argument;
};

struct agc_display_buffer {
   uint64_t address;
   uint64_t pixel_format;
   uint32_t width;
   uint32_t height;
   uint32_t tiling_mode;
   uint32_t pitch_in_pixel;
   uint64_t dcc_address;
   uint64_t dcc_clear_color;
   uint32_t alignment;
   uint32_t reserved;
};

struct agc_videoout_request;
struct agc_videoout_wait;

struct agc_video_output {
   void *self;
   int (*reserve)(void *self, const struct agc_flip_info *info, bool limited,
                  struct agc_videoout_request **request, struct agc_driver_error *error);
   int (*wait_for_flip_room)(void *self, struct agc_driver_error *error);
   int (*capture_rendering_wait)(void *self, uint32_t index, struct agc_videoout_wait **wait,
                                 struct agc_driver_error *error);
   bool (*describe_buffer)(void *self, int32_t index, struct agc_display_buffer *buffer);
   void (*fail)(void *self, const char *message);
   void (*release)(void *self);
   int (*flip_ready)(struct agc_videoout_request *request, uint64_t frame, struct agc_driver_error *error);
   void (*flip_fail)(struct agc_videoout_request *request, const char *message);
   void (*flip_release)(struct agc_videoout_request *request);
   int (*rendering_wait_for)(struct agc_videoout_wait *wait, uint32_t milliseconds,
                             struct agc_driver_error *error);
   void (*rendering_wait_release)(struct agc_videoout_wait *wait);
};

struct agc_presentation_window {
   void *context;
   void (*drawable_size)(void *context, uint32_t *width, uint32_t *height);
   uint32_t width;
   uint32_t height;
   uintptr_t native_window;
};

struct agc_host_display_hdr {
   bool hdr;
   float max_luminance;
   float min_luminance;
   float max_full_frame_luminance;
};

struct agc_host_hdr_status {
   bool available;
   bool surface_hdr10;
   struct agc_host_display_hdr display;
   uint64_t generation;
   uint64_t display_generation;
   float reported_tone_map_peak;
};

struct agc_aspect_fit_size {
   uint32_t width;
   uint32_t height;
};

struct agc_aspect_fit_rect {
   int32_t x;
   int32_t y;
   uint32_t width;
   uint32_t height;
};

_Static_assert(sizeof(struct agc_driver_error) == 256, "agc_driver_error");
_Static_assert(sizeof(struct agc_flip_info) == 24, "agc_flip_info");
_Static_assert(offsetof(struct agc_flip_info, index) == 4, "agc_flip_info.index");
_Static_assert(offsetof(struct agc_flip_info, mode) == 8, "agc_flip_info.mode");
_Static_assert(offsetof(struct agc_flip_info, reserved) == 12, "agc_flip_info.reserved");
_Static_assert(offsetof(struct agc_flip_info, argument) == 16, "agc_flip_info.argument");
_Static_assert(sizeof(struct agc_display_buffer) == 56, "agc_display_buffer");
_Static_assert(offsetof(struct agc_display_buffer, pixel_format) == 8, "agc_display_buffer.pixel_format");
_Static_assert(offsetof(struct agc_display_buffer, width) == 16, "agc_display_buffer.width");
_Static_assert(offsetof(struct agc_display_buffer, height) == 20, "agc_display_buffer.height");
_Static_assert(offsetof(struct agc_display_buffer, tiling_mode) == 24, "agc_display_buffer.tiling_mode");
_Static_assert(offsetof(struct agc_display_buffer, pitch_in_pixel) == 28, "agc_display_buffer.pitch_in_pixel");
_Static_assert(offsetof(struct agc_display_buffer, dcc_address) == 32, "agc_display_buffer.dcc_address");
_Static_assert(offsetof(struct agc_display_buffer, dcc_clear_color) == 40, "agc_display_buffer.dcc_clear_color");
_Static_assert(offsetof(struct agc_display_buffer, alignment) == 48, "agc_display_buffer.alignment");
_Static_assert(offsetof(struct agc_display_buffer, reserved) == 52, "agc_display_buffer.reserved");
_Static_assert(sizeof(struct agc_video_output) == 12 * sizeof(void *), "agc_video_output");
_Static_assert(offsetof(struct agc_video_output, reserve) == 1 * sizeof(void *), "agc_video_output.reserve");
_Static_assert(offsetof(struct agc_video_output, wait_for_flip_room) == 2 * sizeof(void *),
               "agc_video_output.wait_for_flip_room");
_Static_assert(offsetof(struct agc_video_output, capture_rendering_wait) == 3 * sizeof(void *),
               "agc_video_output.capture_rendering_wait");
_Static_assert(offsetof(struct agc_video_output, describe_buffer) == 4 * sizeof(void *),
               "agc_video_output.describe_buffer");
_Static_assert(offsetof(struct agc_video_output, fail) == 5 * sizeof(void *), "agc_video_output.fail");
_Static_assert(offsetof(struct agc_video_output, release) == 6 * sizeof(void *), "agc_video_output.release");
_Static_assert(offsetof(struct agc_video_output, flip_ready) == 7 * sizeof(void *),
               "agc_video_output.flip_ready");
_Static_assert(offsetof(struct agc_video_output, flip_fail) == 8 * sizeof(void *), "agc_video_output.flip_fail");
_Static_assert(offsetof(struct agc_video_output, flip_release) == 9 * sizeof(void *),
               "agc_video_output.flip_release");
_Static_assert(offsetof(struct agc_video_output, rendering_wait_for) == 10 * sizeof(void *),
               "agc_video_output.rendering_wait_for");
_Static_assert(offsetof(struct agc_video_output, rendering_wait_release) == 11 * sizeof(void *),
               "agc_video_output.rendering_wait_release");
_Static_assert(sizeof(struct agc_presentation_window) == 32, "agc_presentation_window");
_Static_assert(offsetof(struct agc_presentation_window, drawable_size) == 8,
               "agc_presentation_window.drawable_size");
_Static_assert(offsetof(struct agc_presentation_window, width) == 16, "agc_presentation_window.width");
_Static_assert(offsetof(struct agc_presentation_window, height) == 20, "agc_presentation_window.height");
_Static_assert(offsetof(struct agc_presentation_window, native_window) == 24,
               "agc_presentation_window.native_window");
_Static_assert(sizeof(struct agc_host_display_hdr) == 16, "agc_host_display_hdr");
_Static_assert(offsetof(struct agc_host_display_hdr, max_luminance) == 4, "agc_host_display_hdr.max_luminance");
_Static_assert(offsetof(struct agc_host_display_hdr, min_luminance) == 8, "agc_host_display_hdr.min_luminance");
_Static_assert(offsetof(struct agc_host_display_hdr, max_full_frame_luminance) == 12,
               "agc_host_display_hdr.max_full_frame_luminance");
_Static_assert(sizeof(struct agc_host_hdr_status) == 48, "agc_host_hdr_status");
_Static_assert(offsetof(struct agc_host_hdr_status, surface_hdr10) == 1, "agc_host_hdr_status.surface_hdr10");
_Static_assert(offsetof(struct agc_host_hdr_status, display) == 4, "agc_host_hdr_status.display");
_Static_assert(offsetof(struct agc_host_hdr_status, generation) == 24, "agc_host_hdr_status.generation");
_Static_assert(offsetof(struct agc_host_hdr_status, display_generation) == 32,
               "agc_host_hdr_status.display_generation");
_Static_assert(offsetof(struct agc_host_hdr_status, reported_tone_map_peak) == 40,
               "agc_host_hdr_status.reported_tone_map_peak");
_Static_assert(sizeof(struct agc_aspect_fit_size) == 8, "agc_aspect_fit_size");
_Static_assert(sizeof(struct agc_aspect_fit_rect) == 16, "agc_aspect_fit_rect");

enum agc_event_kind {
   AGC_EVENT_FLIP,
   AGC_EVENT_EOP
};

struct agc_event {
   uint32_t kind;
   uint32_t queue;
   uint32_t context;
   struct agc_flip_info flip;
};

struct agc_rendering_wait;

int agc_flips_deliver(const struct agc_event *events, uint32_t count, struct agc_driver_error *error);
int agc_flips_rendering_wait(uint32_t handle, uint32_t index, struct agc_rendering_wait **wait,
                             struct agc_driver_error *error);
int agc_rendering_wait_for(struct agc_rendering_wait *wait, uint32_t milliseconds,
                           struct agc_driver_error *error);
void agc_rendering_wait_release(struct agc_rendering_wait *wait);
bool agc_flips_describe_buffer(uint32_t handle, int32_t index, struct agc_display_buffer *buffer);
void agc_flips_fail(const char *message);
const char *agc_flips_failure(void);
void agc_flips_shutdown(void);
int agc_events_add(int64_t eq, uint32_t gpu_queue, uint64_t ident, void *udata, struct agc_driver_error *error);
int agc_events_delete(int64_t eq, uint64_t ident, struct agc_driver_error *error);
void agc_events_interrupt(uint32_t gpu_queue, uint32_t context);
void agc_videoout_reset(void);

struct agc_presenter {
   int (*attach)(const struct agc_presentation_window *window, struct agc_driver_error *error);
   bool (*release)(void *context);
   int (*capture_overlay)(const struct agc_display_buffer *buffer, struct agc_driver_error *error);
   int (*present)(const struct agc_presentation_window *window, const struct agc_display_buffer *buffer,
                  const struct agc_display_buffer *plane, bool *presented, struct agc_driver_error *error);
   size_t (*display_buffer_size)(const struct agc_display_buffer *buffer);
};

struct agc_videoout_core {
   void (*wait_idle)(void);
   void (*shutdown)(void);
};

void agc_videoout_set_presenter(const struct agc_presenter *presenter);
void agc_videoout_set_core(const struct agc_videoout_core *core);
uint64_t agc_videoout_frames_presented(void);

struct ImeKeyboardView;
struct ModalDialogView;
struct AgcDriverPadStatusProvider;

int AgcDriverRegisterVideoOutput(uint32_t handle, const struct agc_video_output *output,
                                 struct agc_driver_error *error);
int AgcDriverUnregisterVideoOutput(uint32_t handle, void *self, struct agc_driver_error *error);
int AgcDriverSubmitFlip(const struct agc_flip_info *info, struct agc_driver_error *error);
int AgcDriverReserveEopFlip(const struct agc_flip_info *info, uint32_t *token, struct agc_driver_error *error);
int AgcDriverReserveEopFlipAt(const struct agc_flip_info *info, uint32_t token, struct agc_driver_error *error);

uint32_t AgcDriverPresentationApi(void);
int AgcDriverAttachWindow(const struct agc_presentation_window *window, struct agc_driver_error *error);
void AgcDriverReleaseWindow(void *context);
int AgcDriverCaptureOverlay(const struct agc_display_buffer *buffer, struct agc_driver_error *error);
int AgcDriverPresentBuffer(const struct agc_presentation_window *window, const struct agc_display_buffer *buffer,
                           const struct agc_display_buffer *plane, void (*gpu_ready)(void *), void *context,
                           struct agc_driver_error *error);
int AgcDriverPresentClear(const struct agc_presentation_window *window, bool opaque,
                          const struct agc_display_buffer *plane, void (*gpu_ready)(void *), void *context,
                          struct agc_driver_error *error);
size_t AgcDriverDisplayBufferSize(const struct agc_display_buffer *buffer);

void AgcDriverSetHostDisplayHdr(const struct agc_host_display_hdr *display);
void AgcDriverSetSurfaceHdr10(bool offered);
void AgcDriverNoteReportedToneMapPeak(float nits);
void AgcDriverHostHdr(struct agc_host_hdr_status *status);
bool AgcDriverPresentingHdr10(void);

void AgcDriverSetSystemOverlay(const struct ImeKeyboardView *view);
void AgcDriverSetSystemDialog(const struct ModalDialogView *view);
void AgcDriverPadStatusPoll(const struct AgcDriverPadStatusProvider *provider);
void AgcDriverPadStatusShow(void);
void AgcDriverTogglePerformanceOverlay(void);
void AgcDriverRequestScreenshot(void);

int AgcDriverCheckGuestMemory(const void *pointer, size_t bytes, size_t alignment, bool writable,
                              struct agc_driver_error *error);
void AgcDriverStoreGuestLabel(uint64_t *label, uint64_t value);
void AgcDriverWaitIdle(void);
void AgcDriverRaiseWorkerThreadPriority(const char *role);
void AgcDriverShutdown(void);
void AgcDriverLockVulkanLoader(void);
void AgcDriverUnlockVulkanLoader(void);
bool AgcDriverTraceRecording(void);
void *AgcDriverTraceThread(void);
void AgcDriverTraceCount(const char *scope);

double ComputeContainScale(uint32_t source_width, uint32_t source_height, uint32_t bounds_width,
                           uint32_t bounds_height, bool allow_upscale);
struct agc_aspect_fit_size ComputeContainSize(uint32_t source_width, uint32_t source_height,
                                              uint32_t bounds_width, uint32_t bounds_height,
                                              bool allow_upscale);
struct agc_aspect_fit_rect ComputeContainRect(uint32_t source_width, uint32_t source_height,
                                              uint32_t destination_width, uint32_t destination_height);
uint32_t ComputeWidthForHeight(uint32_t source_width, uint32_t source_height, uint32_t height);
uint32_t ComputeHeightForWidth(uint32_t source_width, uint32_t source_height, uint32_t width);

#endif
