#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac_gpu_info.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/streaming-load-memcpy.h"
#include "util/u_math.h"

#include "agc_anyps5.h"
#include "agc_driver_ops.h"
#include "agc_kernels.h"
#include "agc_memory.h"
#include "agc_present.h"

#include <windows.h>

#define PROBE_OK 0
#define PROBE_FAILED 1
#define PROBE_UNCONFIRMED 2

#define PROBE_CLIENT_WIDTH 1280u
#define PROBE_CLIENT_HEIGHT 720u
#define PROBE_TIMEOUT_NS (INT64_C(2000) * 1000000)
#define PROBE_BAR_PIXELS 24u
#define PROBE_TILE_PIXELS 128u
#define PROBE_TILE_DWORDS (AGC_PRESENT_BLOCK_64KB / 4)

enum probe_layout {
   PROBE_LINEAR,
   PROBE_TILED,
};

struct probe_format {
   const char *name;
   uint64_t pixel_format;
   bool ten;
   bool red_low;
};

static const struct probe_format probe_formats[] = {
   {"bgra", AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, false, false},
   {"rgba", AGC_PRESENT_PIXEL_FORMAT_R8G8B8A8, false, true},
   {"bgr10", AGC_PRESENT_PIXEL_FORMAT_B10G10R10A2, true, false},
   {"rgb10", AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2, true, true},
};

struct probe_stats {
   uint64_t presented;
   uint64_t skipped;
   int64_t total_ns;
   int64_t worst_total_ns;
   int64_t work_ns;
   int64_t worst_work_ns;
   int64_t wait_ns;
   int64_t convert_ns;
   int64_t present_ns;
};

struct probe {
   HWND hwnd;
   bool quit;
   struct agc_backend *be;
   const struct agc_driver_ops *ops;
   struct radeon_info info;
   bool memory;
   const struct agc_presenter *presenter;
   bool attached;
   struct agc_presentation_window window;
   struct agc_hwq *queue;
   bool queue_ready;
   struct agc_kernel fill;
   struct agc_mem code;
   struct agc_mem ib;
   struct agc_mem display;
   struct agc_display_buffer buffer;
   const struct probe_format *format;
   uint32_t frames;
   uint32_t fps;
   uint32_t width;
   uint32_t height;
   bool linear;
   bool tiled;
   agc_guest_import_release release;
};

static struct probe probe;

static void probe_say(const char *name, const char *format, ...) AGC_PRINTF(2, 3);
static int probe_fail(const char *name, const char *format, ...) AGC_PRINTF(2, 3);

static void
probe_line(const char *prefix, const char *name, const char *format, va_list args)
{
   printf("%s%s: ", prefix, name);
   vprintf(format, args);
   putchar('\n');
   fflush(stdout);
}

static void
probe_say(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   probe_line("", name, format, args);
   va_end(args);
}

static int
probe_fail(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   probe_line("FAIL ", name, format, args);
   va_end(args);
   return PROBE_FAILED;
}

static enum agc_device_state
probe_device_state(void)
{
   static const char *const names[] = {
      [AGC_DEVICE_ACTIVE] = "active",         [AGC_DEVICE_HUNG] = "hung",
      [AGC_DEVICE_PAGE_FAULT] = "page-fault", [AGC_DEVICE_REMOVED] = "removed",
      [AGC_DEVICE_UNKNOWN] = "unknown",
   };
   struct agc_wddm_device_state detail;
   char text[320];
   uint64_t fault_va = 0;
   enum agc_device_state state = probe.ops->device_state(probe.be, &fault_va);
   agc_wddm_device_state_detail(probe.be, &detail);
   agc_wddm_device_state_format(&detail, text, sizeof(text));
   probe_say("device state", "%s, fault va 0x%" PRIx64 "; %s",
             (unsigned)state < ARRAY_SIZE(names) ? names[state] : "unknown", fault_va, text);
   return state;
}

static void *
probe_map_import_alias(uintptr_t address, size_t bytes, uint32_t *refused)
{
   (void)address;
   (void)bytes;
   *refused = AGC_GUEST_IMPORT_NOT_GPU_VISIBLE;
   return NULL;
}

static void
probe_unmap_import_alias(void *alias)
{
   (void)alias;
}

static size_t
probe_gpu_visible(uintptr_t address, size_t bytes, struct agc_guest_range *pieces, size_t capacity)
{
   (void)address;
   (void)bytes;
   (void)pieces;
   (void)capacity;
   return 0;
}

static void
probe_forget_import(uintptr_t address, size_t bytes)
{
   (void)address;
   (void)bytes;
}

static uint64_t
probe_generation(void)
{
   return 1;
}

static void
probe_set_import_release(agc_guest_import_release release, void *user)
{
   (void)user;
   probe.release = release;
}

static const struct agc_anyps5 probe_host = {
   .map_import_alias = probe_map_import_alias,
   .unmap_import_alias = probe_unmap_import_alias,
   .gpu_visible = probe_gpu_visible,
   .forget_import = probe_forget_import,
   .generation = probe_generation,
   .gpu_generation = probe_generation,
   .set_import_release = probe_set_import_release,
};

static bool
probe_open_device(struct agc_present_device *device, char *error, size_t error_bytes)
{
   if (!probe.be) {
      snprintf(error, error_bytes, "the probe opened no transport");
      return false;
   }
   device->backend = probe.be;
   device->info = &probe.info;
   device->sync = NULL;
   device->resident = NULL;
   return true;
}

static LRESULT CALLBACK
probe_window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
   switch (message) {
   case WM_CLOSE:
      probe.quit = true;
      return 0;
   case WM_ERASEBKGND:
      return 1;
   case WM_PAINT: {
      PAINTSTRUCT paint;
      BeginPaint(hwnd, &paint);
      EndPaint(hwnd, &paint);
      return 0;
   }
   default:
      return DefWindowProcW(hwnd, message, wparam, lparam);
   }
}

static void
probe_pump(void)
{
   MSG message;
   while (PeekMessageW(&message, NULL, 0, 0, PM_REMOVE)) {
      if (message.message == WM_QUIT)
         probe.quit = true;
      TranslateMessage(&message);
      DispatchMessageW(&message);
   }
}

static void
probe_drawable_size(void *context, uint32_t *width, uint32_t *height)
{
   (void)context;
   RECT client;
   if (GetClientRect(probe.hwnd, &client)) {
      *width = (uint32_t)MAX2(client.right - client.left, 0);
      *height = (uint32_t)MAX2(client.bottom - client.top, 0);
   }
}

static int
probe_create_window(void)
{
   const HINSTANCE instance = GetModuleHandleW(NULL);
   WNDCLASSEXW window_class = {
      .cbSize = sizeof(window_class),
      .lpfnWndProc = probe_window_proc,
      .hInstance = instance,
      .hCursor = LoadCursorW(NULL, (LPCWSTR)IDC_ARROW),
      .lpszClassName = L"agc_present_probe",
   };
   if (!RegisterClassExW(&window_class))
      return probe_fail("window", "RegisterClassExW failed (error %lu)", GetLastError());
   RECT frame = {0, 0, PROBE_CLIENT_WIDTH, PROBE_CLIENT_HEIGHT};
   const DWORD style = WS_OVERLAPPEDWINDOW;
   AdjustWindowRect(&frame, style, FALSE);
   probe.hwnd = CreateWindowExW(0, window_class.lpszClassName, L"agc_present_probe", style, CW_USEDEFAULT,
                                CW_USEDEFAULT, frame.right - frame.left, frame.bottom - frame.top, NULL, NULL,
                                instance, NULL);
   if (!probe.hwnd)
      return probe_fail("window", "CreateWindowExW failed (error %lu)", GetLastError());
   ShowWindow(probe.hwnd, SW_SHOWNORMAL);
   UpdateWindow(probe.hwnd);
   probe_pump();
   uint32_t width = 0;
   uint32_t height = 0;
   probe_drawable_size(NULL, &width, &height);
   probe_say("window", "HWND %p, client area %ux%u", (void *)probe.hwnd, width, height);
   probe.window = (struct agc_presentation_window){
      .context = &probe,
      .drawable_size = probe_drawable_size,
      .width = PROBE_CLIENT_WIDTH,
      .height = PROBE_CLIENT_HEIGHT,
      .native_window = (uintptr_t)probe.hwnd,
   };
   return PROBE_OK;
}

static int
probe_open(void)
{
   probe.be = agc_wddm_open();
   if (!probe.be)
      return probe_fail("open", "no AMD adapter with a verified profile could be opened");
   probe.ops = probe.be->ops;
   if (!probe.ops->query_info(probe.be, &probe.info))
      return probe_fail("chip", "the GPU did not describe itself");
   probe_say("chip", "\"%s\", %s, gfx level %d, %u compute units", probe.info.marketing_name,
             ac_get_family_name(probe.info.family), (int)probe.info.gfx_level, probe.info.num_cu);

   char error[256];
   const struct agc_memory_config config = {
      .backend = probe.be,
      .anyps5 = &probe_host,
   };
   if (!agc_memory_init(&config, error, sizeof(error)))
      return probe_fail("memory", "%s", error);
   probe.memory = true;

   probe.queue = probe.ops->queue_create(probe.be, AGC_QUEUE_COMPUTE);
   if (!probe.queue)
      return probe_fail("queue", "the kernel refused the probe's compute queue");
   if (!agc_kernel_compile(&probe.info, agc_kernel_fill32(), &probe.fill, error, sizeof(error)))
      return probe_fail("fill32", "%s", error);
   const uint64_t code_bytes = align64((uint64_t)probe.fill.code_dwords * 4, AGC_KERNEL_CODE_ALIGNMENT);
   if (!agc_mem_alloc(code_bytes, AGC_BO_EXECUTE, &probe.code))
      return probe_fail("fill32", "no code memory: %s", agc_memory_last_error());
   memcpy(probe.code.cpu, probe.fill.code, (size_t)probe.fill.code_dwords * 4);
   agc_mem_flush_write_combining();
   return PROBE_OK;
}

static uint32_t
probe_ten(uint32_t c)
{
   return (c << 2) | (c >> 6);
}

static uint32_t
probe_eight(uint32_t c10)
{
   return (c10 * 510 + 1023) / 2046;
}

static uint32_t
probe_encode(uint32_t r, uint32_t g, uint32_t b)
{
   const struct probe_format *f = probe.format;
   if (!f->ten) {
      uint32_t low = f->red_low ? r : b;
      uint32_t high = f->red_low ? b : r;
      return low | g << 8 | high << 16 | UINT32_C(0xff000000);
   }
   uint32_t low = probe_ten(f->red_low ? r : b);
   uint32_t high = probe_ten(f->red_low ? b : r);
   return low | probe_ten(g) << 10 | high << 20 | UINT32_C(3) << 30;
}

static uint32_t
probe_expected(uint32_t r, uint32_t g, uint32_t b)
{
   if (probe.format->ten) {
      r = probe_eight(probe_ten(r));
      g = probe_eight(probe_ten(g));
      b = probe_eight(probe_ten(b));
   }
   return b | g << 8 | r << 16 | UINT32_C(0xff000000);
}

static void
probe_linear_color(uint32_t x, uint32_t y, uint32_t n, uint32_t rgb[3])
{
   const uint32_t bar = (n * 8) % probe.width;
   if (x >= bar && x < bar + PROBE_BAR_PIXELS) {
      rgb[0] = rgb[1] = rgb[2] = 0xff;
      return;
   }
   rgb[0] = ((x ^ y) + 5 * n) & 0xff;
   rgb[1] = (y + n) & 0xff;
   rgb[2] = (x + 3 * n) & 0xff;
}

static void
probe_tile_color(uint32_t tx, uint32_t ty, uint32_t n, uint32_t rgb[3])
{
   const bool on = ((tx + ty + n / 15) & 1) != 0;
   rgb[0] = (tx * 40 + n * 3) & 0xff;
   rgb[1] = (ty * 60 + n * 2) & 0xff;
   rgb[2] = on ? 0xf0 : 0x20;
}

static void
probe_expected_pixel(enum probe_layout layout, uint32_t x, uint32_t y, uint32_t n, uint32_t *value)
{
   uint32_t rgb[3];
   if (layout == PROBE_LINEAR)
      probe_linear_color(x, y, n, rgb);
   else
      probe_tile_color(x / PROBE_TILE_PIXELS, y / PROBE_TILE_PIXELS, n, rgb);
   *value = probe_expected(rgb[0], rgb[1], rgb[2]);
}

static int
probe_display_buffer(enum probe_layout layout)
{
   agc_mem_free(&probe.display);
   const uint32_t pitch = layout == PROBE_LINEAR ? align(probe.width, 64) : 0;
   probe.buffer = (struct agc_display_buffer){
      .address = AGC_PRESENT_ALIGNMENT_64KB,
      .pixel_format = probe.format->pixel_format,
      .width = probe.width,
      .height = probe.height,
      .tiling_mode = layout == PROBE_LINEAR ? AGC_PRESENT_TILING_LINEAR : AGC_PRESENT_TILING_64KB,
      .pitch_in_pixel = pitch,
   };
   const size_t bytes = agc_present_display_buffer_size(&probe.buffer);
   if (!bytes)
      return probe_fail("display buffer", "%ux%u %s is not a valid display buffer", probe.width, probe.height,
                        probe.format->name);
   if (!agc_mem_alloc(bytes, AGC_BO_GL2_BYPASS, &probe.display))
      return probe_fail("display buffer", "no driver memory for %zu bytes: %s", bytes, agc_memory_last_error());
   if (probe.display.va % AGC_PRESENT_ALIGNMENT_64KB)
      return probe_fail("display buffer", "driver memory 0x%" PRIx64 " is not 64 KiB aligned", probe.display.va);
   probe.buffer.address = probe.display.va;
   probe_say("display buffer", "%s %ux%u %s at GPU 0x%" PRIx64 ", %zu bytes%s",
             layout == PROBE_LINEAR ? "linear" : "64KB_R_X", probe.width, probe.height, probe.format->name,
             probe.display.va, bytes, probe.display.write_combined ? ", write-combined" : "");
   return PROBE_OK;
}

static void
probe_write_linear(uint32_t n)
{
   uint32_t *words = probe.display.cpu;
   const uint32_t pitch = probe.buffer.pitch_in_pixel;
   for (uint32_t y = 0; y < probe.height; y++) {
      uint32_t *row = words + (size_t)y * pitch;
      for (uint32_t x = 0; x < probe.width; x++) {
         uint32_t rgb[3];
         probe_linear_color(x, y, n, rgb);
         row[x] = probe_encode(rgb[0], rgb[1], rgb[2]);
      }
   }
   agc_mem_flush_write_combining();
}

static int
probe_fill_tiles(uint32_t n)
{
   const uint32_t across = DIV_ROUND_UP(probe.width, PROBE_TILE_PIXELS);
   const uint32_t down = DIV_ROUND_UP(probe.height, PROBE_TILE_PIXELS);
   const uint32_t dispatch = agc_kernel_dispatch_dwords(&probe.info, &probe.fill);
   const uint64_t dwords = AGC_KERNEL_MAX_QUEUE_SETUP_DWORDS + 2 * AGC_PRESENT_ACQUIRE_DWORDS +
                           AGC_PRESENT_PARTIAL_FLUSH_DWORDS + (uint64_t)across * down * dispatch + 32;
   if (!probe.ib.owner || probe.ib.bytes < dwords * 4) {
      agc_mem_free(&probe.ib);
      if (!agc_mem_alloc(align64(dwords * 4, 0x10000), AGC_BO_GL2_BYPASS, &probe.ib))
         return probe_fail("fill", "no IB memory: %s", agc_memory_last_error());
   }
   uint32_t *begin = probe.ib.cpu;
   uint32_t *p = begin;
   if (!probe.queue_ready) {
      p = agc_kernel_emit_queue_setup(p, &probe.info, true);
      if (!p)
         return probe_fail("fill", "no compute queue setup");
   }
   p = agc_present_emit_acquire(p, agc_present_gcr_before());
   for (uint32_t ty = 0; ty < down; ty++) {
      for (uint32_t tx = 0; tx < across; tx++) {
         uint32_t rgb[3];
         uint32_t user_data[AGC_FILL32_USER_DATA_DWORDS];
         probe_tile_color(tx, ty, n, rgb);
         const uint64_t block = probe.display.va + ((uint64_t)ty * across + tx) * AGC_PRESENT_BLOCK_64KB;
         agc_kernel_fill32_user_data(user_data, block, PROBE_TILE_DWORDS, probe_encode(rgb[0], rgb[1], rgb[2]));
         p = agc_kernel_emit_dispatch(p, &probe.info, &probe.fill, probe.code.va, user_data,
                                      agc_kernel_groups(PROBE_TILE_DWORDS, probe.fill.block[0]), 1, 1);
         if (!p)
            return probe_fail("fill", "the fill dispatch could not be emitted");
      }
   }
   p = agc_present_emit_partial_flush(p);
   p = agc_present_emit_acquire(p, agc_present_gcr_after());
   const uint32_t multiple = MAX2(probe.info.ip[AMD_IP_COMPUTE].ib_pad_dw_mask + 1, 8u);
   p = agc_present_emit_padding(p, agc_present_padding((uint32_t)(p - begin), multiple));
   agc_mem_flush_write_combining();
   const struct agc_ib ib = {.va = probe.ib.va, .dwords = (uint32_t)(p - begin), .flags = AGC_IB_MAIN};
   const uint64_t fence = probe.ops->submit(probe.queue, &ib, 1);
   if (!fence) {
      probe_device_state();
      return probe_fail("fill", "the kernel refused the tile fill");
   }
   probe.queue_ready = true;
   if (!probe.ops->fence_wait(probe.queue, fence, PROBE_TIMEOUT_NS)) {
      probe_device_state();
      probe_fail("fill", "the tile fill did not complete within %" PRId64 " ms", PROBE_TIMEOUT_NS / 1000000);
      return PROBE_UNCONFIRMED;
   }
   return PROBE_OK;
}

static int
probe_verify(enum probe_layout layout, uint32_t n, const char *name)
{
   struct agc_present_frame frame;
   if (!agc_present_last_frame(&frame) || frame.cleared || !frame.cpu)
      return probe_fail(name, "the presenter reports no converted frame");
   if (frame.width != probe.width || frame.height != probe.height || frame.row_pitch < probe.width * 4u)
      return probe_fail(name, "the converted frame is %ux%u with %" PRIu64 "-byte rows", frame.width, frame.height,
                        frame.row_pitch);
   const size_t bytes = (size_t)(frame.row_pitch * frame.height);
   uint8_t *copy = malloc(bytes);
   if (!copy)
      return probe_fail(name, "out of memory");
   util_streaming_load_memcpy(copy, (void *)frame.cpu, bytes);
   uint64_t wrong = 0;
   uint64_t close_count = 0;
   uint32_t first_x = 0;
   uint32_t first_y = 0;
   uint32_t first_got = 0;
   uint32_t first_expected = 0;
   for (uint32_t y = 0; y < frame.height; y++) {
      const uint32_t *row = (const uint32_t *)(copy + (size_t)y * frame.row_pitch);
      for (uint32_t x = 0; x < frame.width; x++) {
         uint32_t expected;
         probe_expected_pixel(layout, x, y, n, &expected);
         if (row[x] == expected)
            continue;
         bool close = probe.format->ten && (row[x] >> 24) == (expected >> 24);
         for (unsigned c = 0; close && c < 3; c++) {
            int diff = (int)((row[x] >> (8 * c)) & 0xff) - (int)((expected >> (8 * c)) & 0xff);
            close = diff >= -1 && diff <= 1;
         }
         if (close) {
            close_count++;
            continue;
         }
         if (!wrong) {
            first_x = x;
            first_y = y;
            first_got = row[x];
            first_expected = expected;
         }
         wrong++;
      }
   }
   free(copy);
   if (wrong)
      return probe_fail(name, "frame %u: %" PRIu64 " of %u pixels are wrong; the first is (%u, %u): 0x%08x, expected "
                              "0x%08x",
                        n, wrong, frame.width * frame.height, first_x, first_y, first_got, first_expected);
   probe_say(name, "frame %u: all %u converted pixels match (%" PRIu64 " within one 8-bit step), drawn at %d,%d %ux%u",
             n, frame.width * frame.height, close_count, frame.rect.x, frame.rect.y, frame.rect.width, frame.rect.height);
   return PROBE_OK;
}

static void
probe_note_timing(struct probe_stats *stats)
{
   struct agc_present_timing timing;
   if (!agc_present_last_timing(&timing))
      return;
   const int64_t work = timing.total_ns - timing.convert_ns - timing.wait_ns;
   stats->total_ns += timing.total_ns;
   stats->worst_total_ns = MAX2(stats->worst_total_ns, timing.total_ns);
   stats->work_ns += work;
   stats->worst_work_ns = MAX2(stats->worst_work_ns, work);
   stats->wait_ns += timing.wait_ns;
   stats->convert_ns += timing.convert_ns;
   stats->present_ns += timing.present_ns;
}

static int64_t
probe_thread_cpu_ns(void)
{
   FILETIME created;
   FILETIME exited;
   FILETIME kernel;
   FILETIME user;
   if (!GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user))
      return 0;
   const uint64_t k = (uint64_t)kernel.dwHighDateTime << 32 | kernel.dwLowDateTime;
   const uint64_t u = (uint64_t)user.dwHighDateTime << 32 | user.dwLowDateTime;
   return (int64_t)((k + u) * 100);
}

static int
probe_run(enum probe_layout layout)
{
   const char *name = layout == PROBE_LINEAR ? "linear" : "tiled";
   int result = probe_display_buffer(layout);
   if (result != PROBE_OK)
      return result;

   struct probe_stats stats = {0};
   const uint64_t attempts = (uint64_t)probe.frames * 4;
   const int64_t interval_ns = probe.fps ? INT64_C(1000000000) / probe.fps : 0;
   uint32_t last = 0;
   int64_t content_ns = 0;
   const int64_t cpu_start = probe_thread_cpu_ns();
   const int64_t start = os_time_get_nano();
   int64_t next = start;
   for (uint64_t attempt = 0; attempt < attempts && stats.presented < probe.frames; attempt++) {
      probe_pump();
      if (probe.quit)
         return probe_fail(name, "the window was closed after %" PRIu64 " frames", stats.presented);
      const uint32_t n = (uint32_t)attempt;
      const int64_t content_start = os_time_get_nano();
      if (layout == PROBE_LINEAR) {
         probe_write_linear(n);
      } else {
         result = probe_fill_tiles(n);
         if (result != PROBE_OK)
            return result;
      }
      content_ns += os_time_get_nano() - content_start;

      struct agc_driver_error error = {0};
      bool presented = false;
      const int code = probe.presenter->present(&probe.window, &probe.buffer, NULL, &presented, &error);
      if (code != AGC_DRIVER_OK) {
         probe_device_state();
         return probe_fail(name, "present %" PRIu64 " returned %d: %s", attempt + 1, code, error.message);
      }
      if (presented) {
         stats.presented++;
         last = n;
         probe_note_timing(&stats);
      } else {
         stats.skipped++;
      }
      if (interval_ns) {
         next += interval_ns;
         const int64_t now = os_time_get_nano();
         if (next > now)
            Sleep((DWORD)((next - now) / 1000000));
         else
            next = now;
      }
   }
   const int64_t elapsed = os_time_get_nano() - start;
   const int64_t cpu = probe_thread_cpu_ns() - cpu_start;
   const double count = stats.presented ? (double)stats.presented : 1.0;
   probe_say(name, "%" PRIu64 " frames presented, %" PRIu64 " skipped, %.1f FPS over %.2f s (pacing %u FPS)",
             stats.presented, stats.skipped, stats.presented * 1e9 / (double)MAX2(elapsed, 1), elapsed / 1e9,
             probe.fps);
   probe_say(name, "per present: %.3f ms average, %.3f ms worst; without the GPU and swap chain waits %.3f ms "
                   "average, %.3f ms worst",
             stats.total_ns / count / 1e6, stats.worst_total_ns / 1e6, stats.work_ns / count / 1e6,
             stats.worst_work_ns / 1e6);
   probe_say(name, "per present: swap chain wait %.3f ms, GPU conversion wait %.3f ms, D3D12 copy and Present %.3f "
                   "ms; content %.3f ms per frame; thread CPU time %.3f ms per frame",
             stats.wait_ns / count / 1e6, stats.convert_ns / count / 1e6, stats.present_ns / count / 1e6,
             content_ns / count / 1e6, cpu / count / 1e6);
   if (stats.presented < probe.frames)
      return probe_fail(name, "only %" PRIu64 " of %u frames were presented", stats.presented, probe.frames);
   return probe_verify(layout, last, name);
}

static int
probe_clear(void)
{
   struct agc_driver_error error = {0};
   bool presented = false;
   const int code = probe.presenter->present(&probe.window, NULL, NULL, &presented, &error);
   if (code != AGC_DRIVER_OK)
      return probe_fail("clear", "returned %d: %s", code, error.message);
   struct agc_present_frame frame;
   if (!presented || !agc_present_last_frame(&frame) || !frame.cleared)
      return probe_fail("clear", "the clear was not presented");
   probe_say("clear", "presented black");
   return PROBE_OK;
}

static void
probe_close(void)
{
   if (probe.attached && !probe.presenter->release(&probe))
      probe_fail("release", "the presenter refused to release the window");
   probe.attached = false;
   if (probe.queue)
      probe.ops->queue_destroy(probe.queue);
   probe.queue = NULL;
   agc_mem_free(&probe.display);
   agc_mem_free(&probe.ib);
   agc_mem_free(&probe.code);
   agc_kernel_free(&probe.fill);
   if (probe.memory)
      agc_memory_fini();
   probe.memory = false;
   if (probe.be)
      probe.ops->destroy(probe.be);
   probe.be = NULL;
   if (probe.hwnd)
      DestroyWindow(probe.hwnd);
   probe.hwnd = NULL;
}

static int
probe_usage(void)
{
   fprintf(stderr,
           "usage: agc_present_probe [--layout linear|tiled|both] [--format bgra|rgba|bgr10|rgb10] [--frames N]\n"
           "                         [--size WxH] [--fps N]\n"
           "  Presents a moving pattern into a 1280x720 window through the D3D12 presenter: a linear display\n"
           "  buffer written by the CPU, then a 64KB_R_X one filled by the GPU. The probe's presenter also copies\n"
           "  each presented back buffer into a readback buffer, and the probe checks those pixels.\n"
           "  --fps 0 presents as fast as possible. Exit code 0: PASS, 1: FAIL, 2: GPU completion unconfirmed.\n");
   return PROBE_FAILED;
}

static int
probe_arguments(int argc, char **argv)
{
   probe.format = &probe_formats[0];
   probe.frames = 300;
   probe.fps = 60;
   probe.width = PROBE_CLIENT_WIDTH;
   probe.height = PROBE_CLIENT_HEIGHT;
   probe.linear = true;
   probe.tiled = true;
   for (int i = 1; i < argc; i += 2) {
      const char *value = i + 1 < argc ? argv[i + 1] : NULL;
      if (!value)
         return probe_usage();
      if (!strcmp(argv[i], "--layout")) {
         probe.linear = !strcmp(value, "linear") || !strcmp(value, "both");
         probe.tiled = !strcmp(value, "tiled") || !strcmp(value, "both");
         if (!probe.linear && !probe.tiled)
            return probe_usage();
      } else if (!strcmp(argv[i], "--format")) {
         const struct probe_format *found = NULL;
         for (unsigned f = 0; f < ARRAY_SIZE(probe_formats); f++) {
            if (!strcmp(value, probe_formats[f].name))
               found = &probe_formats[f];
         }
         if (!found)
            return probe_usage();
         probe.format = found;
      } else if (!strcmp(argv[i], "--frames")) {
         probe.frames = (uint32_t)strtoul(value, NULL, 10);
         if (!probe.frames || probe.frames > 100000)
            return probe_usage();
      } else if (!strcmp(argv[i], "--fps")) {
         probe.fps = (uint32_t)strtoul(value, NULL, 10);
         if (probe.fps > 1000)
            return probe_usage();
      } else if (!strcmp(argv[i], "--size")) {
         if (sscanf(value, "%ux%u", &probe.width, &probe.height) != 2 || !probe.width || !probe.height ||
             probe.width > 8192 || probe.height > 8192)
            return probe_usage();
      } else {
         return probe_usage();
      }
   }
   return PROBE_OK;
}

int
main(int argc, char **argv)
{
   int result = probe_arguments(argc, argv);
   if (result != PROBE_OK)
      return result;
   result = probe_create_window();
   if (result == PROBE_OK)
      result = probe_open();
   if (result == PROBE_OK) {
      probe.presenter = agc_present_d3d12_verified(probe_open_device);
      struct agc_driver_error error = {0};
      const int code = probe.presenter->attach(&probe.window, &error);
      if (code != AGC_DRIVER_OK)
         result = probe_fail("attach", "returned %d: %s", code, error.message);
      else
         probe.attached = true;
   }
   if (result == PROBE_OK && probe.linear)
      result = probe_run(PROBE_LINEAR);
   if (result == PROBE_OK && probe.tiled)
      result = probe_run(PROBE_TILED);
   if (result == PROBE_OK)
      result = probe_clear();
   if (probe.be && result != PROBE_UNCONFIRMED && probe_device_state() != AGC_DEVICE_ACTIVE)
      result = PROBE_UNCONFIRMED;
   if (result == PROBE_UNCONFIRMED) {
      probe_say("teardown", "skipped: GPU completion is unconfirmed, the process exit releases everything");
   } else {
      probe_close();
   }
   probe_say("result", "%s", result == PROBE_OK ? "PASS" : result == PROBE_FAILED ? "FAIL" : "UNCONFIRMED");
   return result;
}
