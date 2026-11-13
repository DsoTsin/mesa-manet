#include "agc_present.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac_gpu_info.h"
#include "sid.h"
#include "util/format/u_formats.h"
#include "util/os_time.h"
#include "util/u_math.h"

#include "agc_debug.h"
#include "agc_driver_ops.h"
#include "agc_kernels.h"
#include "agc_memory.h"

#define COBJMACROS
#include <initguid.h>
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#define AGC_PRESENT_D3D12_TIMEOUT_NS (INT64_C(2000) * 1000000)
#define AGC_PRESENT_D3D12_TIMEOUT_MS 2000u
#define AGC_PRESENT_D3D12_FRAME_WAIT_MS 16u
#define AGC_PRESENT_D3D12_COMMAND_BYTES UINT64_C(0x10000)
#define AGC_PRESENT_D3D12_IB_STRIDE 0x2000u
#define AGC_PRESENT_D3D12_DESCRIPTOR_OFFSET 0x8000u
#define AGC_PRESENT_D3D12_DESCRIPTOR_STRIDE 0x100u
#define AGC_PRESENT_D3D12_IB_MULTIPLE 8u
#define AGC_PRESENT_D3D12_CAPTURE_IB_OFFSET (AGC_PRESENT_SLOTS * AGC_PRESENT_D3D12_IB_STRIDE)
#define AGC_PRESENT_D3D12_OVERLAY_DESCRIPTOR 0x40u
#define AGC_PRESENT_D3D12_CAPTURE_DESCRIPTOR \
   (AGC_PRESENT_D3D12_DESCRIPTOR_OFFSET + AGC_PRESENT_SLOTS * AGC_PRESENT_D3D12_DESCRIPTOR_STRIDE)
#define AGC_PRESENT_D3D12_META_CHECK_BYTES 0x1000u
#define AGC_PRESENT_D3D12_BUFFERS 3u
#define AGC_PRESENT_D3D12_LATENCY 2u
#define AGC_PRESENT_D3D12_STATS_INTERVAL 120u

enum agc_present_d3d12_sync {
   AGC_PRESENT_D3D12_FIFO,
   AGC_PRESENT_D3D12_MAILBOX,
   AGC_PRESENT_D3D12_IMMEDIATE,
};

struct agc_present_d3d12_slot {
   ID3D12Resource *resource;
   struct agc_bo *bo;
   uint64_t va;
   ID3D12Resource *readback;
   void *readback_cpu;
};

struct agc_present_d3d12_stats {
   uint64_t first;
   uint32_t frames;
   uint32_t skipped;
   int64_t total_ns;
   int64_t wait_ns;
   int64_t convert_ns;
   int64_t present_ns;
};

struct agc_present_d3d12_window {
   void *context;
   uintptr_t hwnd;
   struct agc_present_device device;
   const struct agc_driver_ops *ops;
   struct agc_hwq *queue;
   bool queue_ready;
   uint64_t last_fence;
   struct agc_kernel convert;
   struct agc_mem code;
   struct agc_kernel overlay;
   struct agc_mem overlay_code;
   struct agc_mem commands;
   struct agc_mem snapshot;
   struct agc_present_source snapshot_source;
   bool snapshot_valid;

   HMODULE d3d12_module;
   HMODULE dxgi_module;
   IDXGIFactory4 *factory;
   ID3D12Device *d3d;
   ID3D12CommandQueue *d3d_queue;
   ID3D12CommandAllocator *allocators[AGC_PRESENT_SLOTS];
   ID3D12GraphicsCommandList *list;
   ID3D12Fence *ready;
   ID3D12Fence *copied;
   struct agc_wddm_fence *ready_fence;
   struct agc_wddm_fence *copied_fence;
   HANDLE event;

   IDXGISwapChain3 *swapchain;
   HANDLE waitable;
   enum agc_present_d3d12_sync sync;
   bool tearing;
   struct agc_present_canvas canvas;
   struct agc_present_footprint footprint;
   struct agc_present_d3d12_slot slots[AGC_PRESENT_SLOTS];
   ID3D12Resource *black;
   bool verify;

   uint64_t frames;
   uint64_t fence_base;
   bool failed;
   char failure[320];
   bool lost;
   char lost_reason[320];
   struct agc_present_timing timing;
   struct agc_present_frame frame;
   struct agc_present_d3d12_stats stats;
   bool stats_enabled;
};

static agc_present_open_device agc_present_d3d12_open;
static bool agc_present_d3d12_verify;
static _Thread_local struct agc_present_d3d12_window *agc_present_d3d12_attached;

static int AGC_PRINTF(3, 4)
agc_present_d3d12_error(struct agc_driver_error *error, int code, const char *format, ...)
{
   if (error) {
      va_list args;
      va_start(args, format);
      vsnprintf(error->message, sizeof(error->message), format, args);
      va_end(args);
   }
   return code;
}

static int AGC_PRINTF(3, 4)
agc_present_d3d12_latch(struct agc_present_d3d12_window *w, struct agc_driver_error *error, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   vsnprintf(w->failure, sizeof(w->failure), format, args);
   va_end(args);
   w->failed = true;
   agc_report("agc_present", "%s", w->failure);
   return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "%s", w->failure);
}

static void
agc_present_d3d12_device_text(struct agc_present_d3d12_window *w, char *text, size_t bytes)
{
   struct agc_wddm_device_state state;
   agc_wddm_device_state_detail(w->device.backend, &state);
   agc_wddm_device_state_format(&state, text, bytes);
}

static int
agc_present_d3d12_failed(struct agc_present_d3d12_window *w, struct agc_driver_error *error, HRESULT hr,
                         const char *what)
{
   HRESULT removed = w->d3d ? ID3D12Device_GetDeviceRemovedReason(w->d3d) : S_OK;
   if (removed != S_OK || hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
      snprintf(w->lost_reason, sizeof(w->lost_reason), "%s failed (HRESULT 0x%08lx, removal reason 0x%08lx)", what,
               (unsigned long)hr, (unsigned long)removed);
      w->lost = true;
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "%s", w->lost_reason);
   }
   return agc_present_d3d12_latch(w, error, "%s failed (HRESULT 0x%08lx)", what, (unsigned long)hr);
}

static enum agc_present_d3d12_sync
agc_present_d3d12_sync_mode(void)
{
   const char *mode = getenv("ANYPS5_PRESENT_MODE");
   if (mode && !strcmp(mode, "immediate"))
      return AGC_PRESENT_D3D12_IMMEDIATE;
   if (mode && !strcmp(mode, "mailbox"))
      return AGC_PRESENT_D3D12_MAILBOX;
   return AGC_PRESENT_D3D12_FIFO;
}

static bool
agc_present_d3d12_wait_copied(struct agc_present_d3d12_window *w, uint64_t value)
{
   if (value <= w->fence_base || !w->copied || ID3D12Fence_GetCompletedValue(w->copied) >= value)
      return true;
   if (FAILED(ID3D12Fence_SetEventOnCompletion(w->copied, value, w->event)))
      return false;
   if (WaitForSingleObject(w->event, AGC_PRESENT_D3D12_TIMEOUT_MS) != WAIT_OBJECT_0)
      return ID3D12Fence_GetCompletedValue(w->copied) >= value;
   return true;
}

static bool
agc_present_d3d12_drain(struct agc_present_d3d12_window *w)
{
   bool idle = true;
   if (w->queue && w->last_fence && !w->ops->fence_wait(w->queue, w->last_fence, AGC_PRESENT_D3D12_TIMEOUT_NS))
      idle = false;
   if (!agc_present_d3d12_wait_copied(w, w->frames))
      idle = false;
   return idle;
}

static void
agc_present_d3d12_free_slots(struct agc_present_d3d12_window *w)
{
   for (uint32_t i = 0; i < AGC_PRESENT_SLOTS; i++) {
      struct agc_present_d3d12_slot *slot = &w->slots[i];
      if (slot->bo)
         w->ops->bo_destroy(w->device.backend, slot->bo);
      if (slot->resource)
         ID3D12Resource_Release(slot->resource);
      if (slot->readback)
         ID3D12Resource_Release(slot->readback);
      memset(slot, 0, sizeof(*slot));
   }
   if (w->black)
      ID3D12Resource_Release(w->black);
   w->black = NULL;
}

static void
agc_present_d3d12_release_d3d12(struct agc_present_d3d12_window *w, bool idle)
{
   if (idle)
      agc_present_d3d12_free_slots(w);
   if (w->swapchain)
      IDXGISwapChain3_Release(w->swapchain);
   if (w->waitable)
      CloseHandle(w->waitable);
   if (w->list)
      ID3D12GraphicsCommandList_Release(w->list);
   for (uint32_t i = 0; i < AGC_PRESENT_SLOTS; i++) {
      if (w->allocators[i])
         ID3D12CommandAllocator_Release(w->allocators[i]);
      w->allocators[i] = NULL;
   }
   if (w->ready)
      ID3D12Fence_Release(w->ready);
   if (w->copied)
      ID3D12Fence_Release(w->copied);
   if (w->d3d_queue)
      ID3D12CommandQueue_Release(w->d3d_queue);
   if (w->d3d)
      ID3D12Device_Release(w->d3d);
   if (w->factory)
      IDXGIFactory4_Release(w->factory);
   if (w->event)
      CloseHandle(w->event);
   agc_wddm_fence_destroy(w->copied_fence);
   agc_wddm_fence_destroy(w->ready_fence);
   w->swapchain = NULL;
   w->waitable = NULL;
   w->list = NULL;
   w->ready = NULL;
   w->copied = NULL;
   w->d3d_queue = NULL;
   w->d3d = NULL;
   w->factory = NULL;
   w->event = NULL;
   w->copied_fence = NULL;
   w->ready_fence = NULL;
   memset(&w->canvas, 0, sizeof(w->canvas));
   memset(&w->footprint, 0, sizeof(w->footprint));
}

static void
agc_present_d3d12_destroy(struct agc_present_d3d12_window *w)
{
   bool idle = agc_present_d3d12_drain(w);
   if (!idle) {
      char text[320];
      agc_present_d3d12_device_text(w, text, sizeof(text));
      agc_report("agc_present", "presented frames did not finish; the presenter's memory stays allocated (%s)", text);
   }
   agc_present_d3d12_release_d3d12(w, idle);
   if (w->queue)
      w->ops->queue_destroy(w->queue);
   if (idle) {
      agc_mem_free(&w->commands);
      agc_mem_free(&w->code);
      agc_mem_free(&w->overlay_code);
      agc_mem_free(&w->snapshot);
   }
   agc_kernel_free(&w->convert);
   agc_kernel_free(&w->overlay);
   free(w);
}

static bool
agc_present_d3d12_kernel(const struct radeon_info *info, struct nir_shader *nir, struct agc_kernel *kernel,
                         struct agc_mem *code, char *text, size_t text_bytes)
{
   char log[256];
   if (!agc_kernel_compile(info, nir, kernel, log, sizeof(log))) {
      snprintf(text, text_bytes, "a presenter kernel does not compile: %s", log);
      return false;
   }
   const uint64_t bytes = align64((uint64_t)kernel->code_dwords * 4, AGC_KERNEL_CODE_ALIGNMENT);
   if (!agc_mem_alloc(bytes, AGC_BO_EXECUTE, code) || code->va % AGC_KERNEL_CODE_ALIGNMENT) {
      snprintf(text, text_bytes, "no driver memory for a presenter kernel: %s", agc_memory_last_error());
      return false;
   }
   memcpy(code->cpu, kernel->code, (size_t)kernel->code_dwords * 4);
   return true;
}

static int
agc_present_d3d12_setup_native(struct agc_present_d3d12_window *w, struct agc_driver_error *error)
{
   char text[256];
   if (!agc_present_d3d12_open(&w->device, text, sizeof(text)))
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "the driver did not start: %s", text);
   const struct radeon_info *info = w->device.info;
   if (!w->device.backend || !w->device.backend->ops || !info)
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "the driver started without a device");
   if (info->gfx_level < GFX10_3 || info->gfx_level > GFX11_5)
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED,
                                     "display conversion is implemented for gfx10.3 to gfx11.5 hosts, not gfx level %d",
                                     (int)info->gfx_level);
   if (info->ip[AMD_IP_COMPUTE].ib_alignment > AGC_PRESENT_D3D12_IB_STRIDE)
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "compute IBs need %u-byte alignment",
                                     info->ip[AMD_IP_COMPUTE].ib_alignment);
   w->ops = w->device.backend->ops;
   w->queue = w->ops->queue_create(w->device.backend, AGC_QUEUE_COMPUTE);
   if (!w->queue)
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "the kernel refused the presenter's compute queue");
   if (!agc_kernel_compile(info, agc_kernel_display_convert(), &w->convert, text, sizeof(text)))
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "display_convert does not compile: %s", text);
   const uint64_t code_bytes = align64((uint64_t)w->convert.code_dwords * 4, AGC_KERNEL_CODE_ALIGNMENT);
   if (!agc_mem_alloc(code_bytes, AGC_BO_EXECUTE, &w->code) ||
       !agc_mem_alloc(AGC_PRESENT_D3D12_COMMAND_BYTES, AGC_BO_GL2_BYPASS, &w->commands))
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "no driver memory for the presenter: %s",
                                     agc_memory_last_error());
   if (w->code.va % AGC_KERNEL_CODE_ALIGNMENT || w->commands.va % AGC_PRESENT_D3D12_IB_STRIDE)
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "driver memory at 0x%" PRIx64 " and 0x%" PRIx64
                                     " is not aligned for code and IBs",
                                     w->code.va, w->commands.va);
   memcpy(w->code.cpu, w->convert.code, (size_t)w->convert.code_dwords * 4);
   if (!agc_present_d3d12_kernel(info, agc_kernel_display_overlay(), &w->overlay, &w->overlay_code, text,
                                 sizeof(text)))
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "%s", text);
   agc_mem_flush_write_combining();
   return AGC_DRIVER_OK;
}

typedef HRESULT(WINAPI *agc_present_create_factory)(UINT flags, REFIID riid, void **factory);

static int
agc_present_d3d12_setup_d3d12(struct agc_present_d3d12_window *w, struct agc_driver_error *error)
{
   w->d3d12_module = LoadLibraryW(L"d3d12.dll");
   w->dxgi_module = LoadLibraryW(L"dxgi.dll");
   if (!w->d3d12_module || !w->dxgi_module)
      return agc_present_d3d12_latch(w, error, "d3d12.dll or dxgi.dll does not load (error %lu)", GetLastError());
   PFN_D3D12_CREATE_DEVICE create_device =
      (PFN_D3D12_CREATE_DEVICE)(uintptr_t)GetProcAddress(w->d3d12_module, "D3D12CreateDevice");
   agc_present_create_factory create_factory =
      (agc_present_create_factory)(uintptr_t)GetProcAddress(w->dxgi_module, "CreateDXGIFactory2");
   if (!create_device || !create_factory)
      return agc_present_d3d12_latch(w, error, "D3D12CreateDevice or CreateDXGIFactory2 is missing");

   HRESULT hr = create_factory(0, &IID_IDXGIFactory4, (void **)&w->factory);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "CreateDXGIFactory2");
   struct agc_wddm_description description;
   agc_wddm_describe(w->device.backend, &description);
   const LUID luid = {.LowPart = description.adapter.luid_low, .HighPart = description.adapter.luid_high};
   IDXGIAdapter1 *adapter = NULL;
   hr = IDXGIFactory4_EnumAdapterByLuid(w->factory, luid, &IID_IDXGIAdapter1, (void **)&adapter);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "EnumAdapterByLuid");
   hr = create_device((IUnknown *)adapter, D3D_FEATURE_LEVEL_12_0, &IID_ID3D12Device, (void **)&w->d3d);
   IDXGIAdapter1_Release(adapter);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "D3D12CreateDevice");

   const D3D12_COMMAND_QUEUE_DESC queue = {.Type = D3D12_COMMAND_LIST_TYPE_DIRECT};
   hr = ID3D12Device_CreateCommandQueue(w->d3d, &queue, &IID_ID3D12CommandQueue, (void **)&w->d3d_queue);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "CreateCommandQueue");

   hr = ID3D12Device_CreateFence(w->d3d, 0, D3D12_FENCE_FLAG_SHARED, &IID_ID3D12Fence, (void **)&w->copied);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "CreateFence(F_copy)");
   HANDLE copied = NULL;
   hr = ID3D12Device_CreateSharedHandle(w->d3d, (ID3D12DeviceChild *)w->copied, NULL, GENERIC_ALL, NULL, &copied);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "CreateSharedHandle(F_copy)");
   w->copied_fence = agc_wddm_fence_open_shared(w->device.backend, copied);
   CloseHandle(copied);
   if (!w->copied_fence)
      return agc_present_d3d12_latch(w, error, "the native device does not open F_copy");

   void *ready = NULL;
   w->ready_fence = agc_wddm_fence_create_shared(w->device.backend, 0, &ready);
   if (!w->ready_fence)
      return agc_present_d3d12_latch(w, error, "the native device does not create F_ready");
   hr = ID3D12Device_OpenSharedHandle(w->d3d, ready, &IID_ID3D12Fence, (void **)&w->ready);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "OpenSharedHandle(F_ready)");

   for (uint32_t i = 0; i < AGC_PRESENT_SLOTS; i++) {
      hr = ID3D12Device_CreateCommandAllocator(w->d3d, D3D12_COMMAND_LIST_TYPE_DIRECT, &IID_ID3D12CommandAllocator,
                                               (void **)&w->allocators[i]);
      if (FAILED(hr))
         return agc_present_d3d12_failed(w, error, hr, "CreateCommandAllocator");
   }
   hr = ID3D12Device_CreateCommandList(w->d3d, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, w->allocators[0], NULL,
                                       &IID_ID3D12GraphicsCommandList, (void **)&w->list);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "CreateCommandList");
   hr = ID3D12GraphicsCommandList_Close(w->list);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "CommandList::Close");
   w->event = CreateEventW(NULL, FALSE, FALSE, NULL);
   if (!w->event)
      return agc_present_d3d12_latch(w, error, "CreateEventW failed (error %lu)", GetLastError());

   w->sync = agc_present_d3d12_sync_mode();
   if (w->sync == AGC_PRESENT_D3D12_IMMEDIATE) {
      IDXGIFactory5 *factory5 = NULL;
      BOOL allowed = FALSE;
      if (SUCCEEDED(IDXGIFactory4_QueryInterface(w->factory, &IID_IDXGIFactory5, (void **)&factory5))) {
         if (FAILED(IDXGIFactory5_CheckFeatureSupport(factory5, DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowed,
                                                      sizeof(allowed))))
            allowed = FALSE;
         IDXGIFactory5_Release(factory5);
      }
      w->tearing = allowed != FALSE;
   }
   return AGC_DRIVER_OK;
}

static UINT
agc_present_d3d12_swapchain_flags(const struct agc_present_d3d12_window *w)
{
   UINT flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
   if (w->tearing)
      flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
   return flags;
}

static int
agc_present_d3d12_create_swapchain(struct agc_present_d3d12_window *w, uint32_t width, uint32_t height,
                                   struct agc_driver_error *error)
{
   const DXGI_SWAP_CHAIN_DESC1 desc = {
      .Width = width,
      .Height = height,
      .Format = DXGI_FORMAT_B8G8R8A8_UNORM,
      .SampleDesc = {.Count = 1},
      .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
      .BufferCount = AGC_PRESENT_D3D12_BUFFERS,
      .Scaling = DXGI_SCALING_STRETCH,
      .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
      .AlphaMode = DXGI_ALPHA_MODE_IGNORE,
      .Flags = agc_present_d3d12_swapchain_flags(w),
   };
   IDXGISwapChain1 *swapchain = NULL;
   HRESULT hr = IDXGIFactory4_CreateSwapChainForHwnd(w->factory, (IUnknown *)w->d3d_queue, (HWND)w->hwnd, &desc,
                                                     NULL, NULL, &swapchain);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "CreateSwapChainForHwnd");
   hr = IDXGIFactory4_MakeWindowAssociation(w->factory, (HWND)w->hwnd,
                                            DXGI_MWA_NO_WINDOW_CHANGES | DXGI_MWA_NO_ALT_ENTER);
   if (FAILED(hr)) {
      IDXGISwapChain1_Release(swapchain);
      return agc_present_d3d12_failed(w, error, hr, "MakeWindowAssociation");
   }
   hr = IDXGISwapChain1_QueryInterface(swapchain, &IID_IDXGISwapChain3, (void **)&w->swapchain);
   IDXGISwapChain1_Release(swapchain);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "IDXGISwapChain3");
   hr = IDXGISwapChain3_SetMaximumFrameLatency(w->swapchain, AGC_PRESENT_D3D12_LATENCY);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "SetMaximumFrameLatency");
   w->waitable = IDXGISwapChain3_GetFrameLatencyWaitableObject(w->swapchain);
   if (!w->waitable)
      return agc_present_d3d12_latch(w, error, "the swap chain has no frame latency waitable object");
   return AGC_DRIVER_OK;
}

static int
agc_present_d3d12_committed(struct agc_present_d3d12_window *w, D3D12_HEAP_TYPE type, D3D12_HEAP_FLAGS flags,
                            uint64_t bytes, D3D12_RESOURCE_STATES state, ID3D12Resource **resource,
                            struct agc_driver_error *error, const char *what)
{
   const D3D12_HEAP_PROPERTIES heap = {.Type = type};
   const D3D12_RESOURCE_DESC desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = bytes,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_UNKNOWN,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
   };
   HRESULT hr = ID3D12Device_CreateCommittedResource(w->d3d, &heap, flags, &desc, state, NULL, &IID_ID3D12Resource,
                                                     (void **)resource);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, what);
   return AGC_DRIVER_OK;
}

static int
agc_present_d3d12_check_footprint(struct agc_present_d3d12_window *w, const struct agc_present_footprint *footprint,
                                  struct agc_driver_error *error)
{
   const D3D12_RESOURCE_DESC desc = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
      .Width = footprint->width,
      .Height = footprint->height,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .Format = DXGI_FORMAT_B8G8R8A8_UNORM,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
   };
   D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;
   UINT64 total = 0;
   memset(&layout, 0, sizeof(layout));
   ID3D12Device_GetCopyableFootprints(w->d3d, &desc, 0, 1, 0, &layout, NULL, NULL, &total);
   if (layout.Footprint.RowPitch != footprint->row_pitch || total != footprint->required_bytes ||
       layout.Footprint.Width != footprint->width || layout.Footprint.Height != footprint->height)
      return agc_present_d3d12_latch(w, error,
                                     "GetCopyableFootprints gives %ux%u pitch %u and %" PRIu64
                                     " bytes, the frame layout %ux%u pitch %" PRIu64 " and %" PRIu64 " bytes",
                                     layout.Footprint.Width, layout.Footprint.Height, layout.Footprint.RowPitch,
                                     (uint64_t)total, footprint->width, footprint->height, footprint->row_pitch,
                                     footprint->required_bytes);
   return AGC_DRIVER_OK;
}

static int
agc_present_d3d12_create_slots(struct agc_present_d3d12_window *w, struct agc_driver_error *error)
{
   const uint64_t bytes = w->footprint.slot_bytes;
   for (uint32_t i = 0; i < AGC_PRESENT_SLOTS; i++) {
      struct agc_present_d3d12_slot *slot = &w->slots[i];
      int code = agc_present_d3d12_committed(w, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED, bytes,
                                             D3D12_RESOURCE_STATE_COMMON, &slot->resource, error,
                                             "CreateCommittedResource(shared frame slot)");
      if (code != AGC_DRIVER_OK)
         return code;
      HANDLE shared = NULL;
      HRESULT hr = ID3D12Device_CreateSharedHandle(w->d3d, (ID3D12DeviceChild *)slot->resource, NULL, GENERIC_ALL,
                                                   NULL, &shared);
      if (FAILED(hr))
         return agc_present_d3d12_failed(w, error, hr, "CreateSharedHandle(frame slot)");
      slot->bo = agc_wddm_bo_open_shared(w->device.backend, shared, bytes);
      CloseHandle(shared);
      if (!slot->bo)
         return agc_present_d3d12_latch(w, error, "the native device does not open a %" PRIu64 "-byte frame slot",
                                        bytes);
      slot->va = w->ops->bo_va(slot->bo);
      if (w->verify) {
         code = agc_present_d3d12_committed(w, D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_NONE, bytes,
                                            D3D12_RESOURCE_STATE_COPY_DEST, &slot->readback, error,
                                            "CreateCommittedResource(verification readback)");
         if (code != AGC_DRIVER_OK)
            return code;
         hr = ID3D12Resource_Map(slot->readback, 0, NULL, &slot->readback_cpu);
         if (FAILED(hr))
            return agc_present_d3d12_failed(w, error, hr, "Map(verification readback)");
      }
   }
   return agc_present_d3d12_committed(w, D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, bytes,
                                      D3D12_RESOURCE_STATE_COMMON, &w->black, error,
                                      "CreateCommittedResource(black frame)");
}

static int
agc_present_d3d12_fit(struct agc_present_d3d12_window *w, const struct agc_present_canvas *canvas,
                      struct agc_driver_error *error)
{
   if (w->swapchain && agc_present_canvas_equal(&w->canvas, canvas))
      return AGC_DRIVER_OK;
   struct agc_present_footprint footprint;
   if (!agc_present_footprint_of(canvas->width, canvas->height, &footprint))
      return agc_present_d3d12_latch(w, error, "no frame layout for a %ux%u canvas", canvas->width, canvas->height);
   if (!agc_present_d3d12_drain(w)) {
      char text[320];
      agc_present_d3d12_device_text(w, text, sizeof(text));
      return agc_present_d3d12_latch(w, error, "presented frames did not finish before a resize (%s)", text);
   }
   agc_present_d3d12_free_slots(w);
   int code = AGC_DRIVER_OK;
   if (!w->swapchain) {
      code = agc_present_d3d12_create_swapchain(w, canvas->width, canvas->height, error);
   } else if (w->canvas.width != canvas->width || w->canvas.height != canvas->height) {
      HRESULT hr = IDXGISwapChain3_ResizeBuffers(w->swapchain, AGC_PRESENT_D3D12_BUFFERS, canvas->width,
                                                 canvas->height, DXGI_FORMAT_B8G8R8A8_UNORM,
                                                 agc_present_d3d12_swapchain_flags(w));
      if (FAILED(hr))
         code = agc_present_d3d12_failed(w, error, hr, "ResizeBuffers");
   }
   if (code != AGC_DRIVER_OK)
      return code;
   code = agc_present_d3d12_check_footprint(w, &footprint, error);
   if (code != AGC_DRIVER_OK)
      return code;
   w->footprint = footprint;
   w->canvas = *canvas;
   code = agc_present_d3d12_create_slots(w, error);
   if (code != AGC_DRIVER_OK)
      return code;
   agc_report("agc_present", "D3D12 swap chain %ux%u for a %ux%u image at %u,%u (%s%s)", canvas->width,
              canvas->height, canvas->image_width, canvas->image_height, canvas->image_x, canvas->image_y,
              w->sync == AGC_PRESENT_D3D12_FIFO ? "fifo" : w->sync == AGC_PRESENT_D3D12_MAILBOX ? "mailbox" : "immediate",
              w->tearing ? ", tearing" : "");
   return AGC_DRIVER_OK;
}

static int
agc_present_d3d12_convert(struct agc_present_d3d12_window *w, const struct agc_present_source *source, uint32_t slot,
                          uint64_t frame, bool overlay, struct agc_driver_error *error)
{
   const struct radeon_info *info = w->device.info;
   uint32_t descriptor[AGC_PRESENT_DESCRIPTOR_DWORDS];
   char text[320];
   if (!agc_present_descriptor(info, source, descriptor, text, sizeof(text)))
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "display buffer descriptor: %s", text);

   const uint64_t descriptor_offset = AGC_PRESENT_D3D12_DESCRIPTOR_OFFSET + slot * AGC_PRESENT_D3D12_DESCRIPTOR_STRIDE;
   memcpy((uint8_t *)w->commands.cpu + descriptor_offset, descriptor, sizeof(descriptor));
   uint32_t *begin = (uint32_t *)((uint8_t *)w->commands.cpu + slot * AGC_PRESENT_D3D12_IB_STRIDE);
   uint32_t *p = begin;
   if (!w->queue_ready) {
      p = agc_kernel_emit_queue_setup(p, info, true);
      if (!p)
         return agc_present_d3d12_latch(w, error, "the presenter's compute queue setup could not be emitted");
   }
   p = agc_present_emit_acquire(p, agc_present_gcr_before());
   uint32_t user_data[AGC_DISPLAY_CONVERT_USER_DATA_DWORDS];
   const uint64_t dst = w->slots[slot].va + agc_present_image_offset(&w->canvas, &w->footprint);
   agc_kernel_display_convert_user_data(user_data, w->commands.va + descriptor_offset, dst, source->width,
                                        source->height, (uint32_t)w->footprint.row_pitch);
   p = agc_kernel_emit_dispatch(p, info, &w->convert, w->code.va, user_data,
                                agc_kernel_groups(source->width, w->convert.block[0]),
                                agc_kernel_groups(source->height, w->convert.block[1]), 1);
   if (!p)
      return agc_present_d3d12_latch(w, error, "the display conversion dispatch could not be emitted");
   if (overlay) {
      uint32_t overlay_descriptor[AGC_PRESENT_DESCRIPTOR_DWORDS];
      if (!agc_present_descriptor(info, &w->snapshot_source, overlay_descriptor, text, sizeof(text)))
         return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "overlay descriptor: %s", text);
      const uint64_t overlay_offset = descriptor_offset + AGC_PRESENT_D3D12_OVERLAY_DESCRIPTOR;
      memcpy((uint8_t *)w->commands.cpu + overlay_offset, overlay_descriptor, sizeof(overlay_descriptor));
      p = agc_present_emit_partial_flush(p);
      p = agc_present_emit_acquire(p, S_587_GLV_INV(1) | S_587_GL1_INV(1));
      const uint32_t overlay_user_data[AGC_DISPLAY_OVERLAY_USER_DATA_DWORDS] = {
         [AGC_DISPLAY_OVERLAY_DESCRIPTOR_LO] = (uint32_t)(w->commands.va + overlay_offset),
         [AGC_DISPLAY_OVERLAY_DESCRIPTOR_HI] = (uint32_t)((w->commands.va + overlay_offset) >> 32),
         [AGC_DISPLAY_OVERLAY_DST_LO] = (uint32_t)dst,
         [AGC_DISPLAY_OVERLAY_DST_HI] = (uint32_t)(dst >> 32),
         [AGC_DISPLAY_OVERLAY_WIDTH] = source->width,
         [AGC_DISPLAY_OVERLAY_HEIGHT] = source->height,
         [AGC_DISPLAY_OVERLAY_PITCH_BYTES] = (uint32_t)w->footprint.row_pitch,
         [AGC_DISPLAY_OVERLAY_SIZE] = w->snapshot_source.width | (w->snapshot_source.height << 16),
      };
      p = agc_kernel_emit_dispatch(p, info, &w->overlay, w->overlay_code.va, overlay_user_data,
                                   agc_kernel_groups(source->width, w->overlay.block[0]),
                                   agc_kernel_groups(source->height, w->overlay.block[1]), 1);
      if (!p)
         return agc_present_d3d12_latch(w, error, "the overlay dispatch could not be emitted");
   }
   p = agc_present_emit_partial_flush(p);
   p = agc_present_emit_acquire(p, agc_present_gcr_after());
   const uint32_t multiple = MAX2(info->ip[AMD_IP_COMPUTE].ib_pad_dw_mask + 1, AGC_PRESENT_D3D12_IB_MULTIPLE);
   p = agc_present_emit_padding(p, agc_present_padding((uint32_t)(p - begin), multiple));
   const struct agc_ib ib = {
      .va = w->commands.va + (uint64_t)slot * AGC_PRESENT_D3D12_IB_STRIDE,
      .dwords = (uint32_t)(p - begin),
      .flags = AGC_IB_MAIN,
   };
   agc_mem_flush_write_combining();

   if (frame > w->fence_base + AGC_PRESENT_SLOTS &&
       !agc_wddm_queue_wait_fence(w->queue, w->copied_fence, frame - AGC_PRESENT_SLOTS)) {
      agc_present_d3d12_device_text(w, text, sizeof(text));
      return agc_present_d3d12_latch(w, error, "the presenter's queue cannot wait for F_copy (%s)", text);
   }
   const uint64_t fence = w->ops->submit(w->queue, &ib, 1);
   if (!fence) {
      agc_present_d3d12_device_text(w, text, sizeof(text));
      return agc_present_d3d12_latch(w, error, "the kernel refused the display conversion (%s)", text);
   }
   w->queue_ready = true;
   w->last_fence = fence;
   return AGC_DRIVER_OK;
}

static D3D12_RESOURCE_BARRIER
agc_present_d3d12_transition(ID3D12Resource *resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
   D3D12_RESOURCE_BARRIER barrier;
   memset(&barrier, 0, sizeof(barrier));
   barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
   barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
   barrier.Transition.pResource = resource;
   barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
   barrier.Transition.StateBefore = before;
   barrier.Transition.StateAfter = after;
   return barrier;
}

static D3D12_TEXTURE_COPY_LOCATION
agc_present_d3d12_placed(ID3D12Resource *buffer, const struct agc_present_footprint *footprint)
{
   D3D12_TEXTURE_COPY_LOCATION location;
   memset(&location, 0, sizeof(location));
   location.pResource = buffer;
   location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
   location.PlacedFootprint.Offset = 0;
   location.PlacedFootprint.Footprint.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
   location.PlacedFootprint.Footprint.Width = footprint->width;
   location.PlacedFootprint.Footprint.Height = footprint->height;
   location.PlacedFootprint.Footprint.Depth = 1;
   location.PlacedFootprint.Footprint.RowPitch = (UINT)footprint->row_pitch;
   return location;
}

static D3D12_TEXTURE_COPY_LOCATION
agc_present_d3d12_subresource(ID3D12Resource *texture)
{
   D3D12_TEXTURE_COPY_LOCATION location;
   memset(&location, 0, sizeof(location));
   location.pResource = texture;
   location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
   location.SubresourceIndex = 0;
   return location;
}

static int
agc_present_d3d12_show(struct agc_present_d3d12_window *w, uint32_t slot, uint64_t frame, ID3D12Resource *source,
                       struct agc_driver_error *error)
{
   if (frame > w->fence_base + AGC_PRESENT_SLOTS && !agc_present_d3d12_wait_copied(w, frame - AGC_PRESENT_SLOTS))
      return agc_present_d3d12_latch(w, error, "frame %" PRIu64 " was not copied within %u ms",
                                     frame - AGC_PRESENT_SLOTS, AGC_PRESENT_D3D12_TIMEOUT_MS);
   ID3D12CommandAllocator *allocator = w->allocators[slot];
   HRESULT hr = ID3D12CommandAllocator_Reset(allocator);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "CommandAllocator::Reset");
   hr = ID3D12GraphicsCommandList_Reset(w->list, allocator, NULL);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "CommandList::Reset");
   ID3D12Resource *back = NULL;
   hr = IDXGISwapChain3_GetBuffer(w->swapchain, IDXGISwapChain3_GetCurrentBackBufferIndex(w->swapchain),
                                  &IID_ID3D12Resource, (void **)&back);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "GetBuffer");

   D3D12_RESOURCE_BARRIER barrier =
      agc_present_d3d12_transition(back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST);
   ID3D12GraphicsCommandList_ResourceBarrier(w->list, 1, &barrier);
   D3D12_TEXTURE_COPY_LOCATION dst = agc_present_d3d12_subresource(back);
   D3D12_TEXTURE_COPY_LOCATION src = agc_present_d3d12_placed(source, &w->footprint);
   ID3D12GraphicsCommandList_CopyTextureRegion(w->list, &dst, 0, 0, 0, &src, NULL);
   if (w->verify) {
      barrier = agc_present_d3d12_transition(back, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
      ID3D12GraphicsCommandList_ResourceBarrier(w->list, 1, &barrier);
      dst = agc_present_d3d12_placed(w->slots[slot].readback, &w->footprint);
      src = agc_present_d3d12_subresource(back);
      ID3D12GraphicsCommandList_CopyTextureRegion(w->list, &dst, 0, 0, 0, &src, NULL);
      barrier = agc_present_d3d12_transition(back, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
   } else {
      barrier = agc_present_d3d12_transition(back, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PRESENT);
   }
   ID3D12GraphicsCommandList_ResourceBarrier(w->list, 1, &barrier);
   hr = ID3D12GraphicsCommandList_Close(w->list);
   if (FAILED(hr)) {
      ID3D12Resource_Release(back);
      return agc_present_d3d12_failed(w, error, hr, "CommandList::Close");
   }

   hr = ID3D12CommandQueue_Wait(w->d3d_queue, w->ready, frame);
   if (FAILED(hr)) {
      ID3D12Resource_Release(back);
      return agc_present_d3d12_failed(w, error, hr, "CommandQueue::Wait(F_ready)");
   }
   ID3D12CommandList *lists[] = {(ID3D12CommandList *)w->list};
   ID3D12CommandQueue_ExecuteCommandLists(w->d3d_queue, 1, lists);
   hr = ID3D12CommandQueue_Signal(w->d3d_queue, w->copied, frame);
   if (FAILED(hr)) {
      ID3D12Resource_Release(back);
      return agc_present_d3d12_failed(w, error, hr, "CommandQueue::Signal(F_copy)");
   }
   UINT interval = w->sync == AGC_PRESENT_D3D12_FIFO ? 1 : 0;
   UINT flags = w->sync == AGC_PRESENT_D3D12_IMMEDIATE && w->tearing ? DXGI_PRESENT_ALLOW_TEARING : 0;
   hr = IDXGISwapChain3_Present(w->swapchain, interval, flags);
   ID3D12Resource_Release(back);
   if (FAILED(hr))
      return agc_present_d3d12_failed(w, error, hr, "Present");
   return AGC_DRIVER_OK;
}

static void agc_present_d3d12_note(struct agc_present_d3d12_window *w, bool skipped);
static int agc_present_d3d12_setup_d3d12(struct agc_present_d3d12_window *w, struct agc_driver_error *error);

static int
agc_present_d3d12_recover(struct agc_present_d3d12_window *w, int64_t start, struct agc_driver_error *error)
{
   agc_stat("agc_present", "the D3D12 side is gone (%s); it is rebuilt and frame %" PRIu64 " is dropped",
            w->lost_reason, w->frames + 1);
   char text[320];
   if (w->queue && w->last_fence && !w->ops->fence_wait(w->queue, w->last_fence, AGC_PRESENT_D3D12_TIMEOUT_NS)) {
      agc_present_d3d12_device_text(w, text, sizeof(text));
      return agc_present_d3d12_latch(w, error, "%s, and the presenter's own queue did not go idle (%s)",
                                     w->lost_reason, text);
   }
   agc_present_d3d12_release_d3d12(w, true);
   w->lost = false;
   w->fence_base = w->frames;
   const int code = agc_present_d3d12_setup_d3d12(w, error);
   if (code != AGC_DRIVER_OK) {
      if (!w->failed)
         return agc_present_d3d12_latch(w, error, "the D3D12 side could not be rebuilt after %s", w->lost_reason);
      return code;
   }
   w->timing.total_ns = os_time_get_nano() - start;
   agc_present_d3d12_note(w, true);
   return AGC_DRIVER_OK;
}

static void
agc_present_d3d12_note(struct agc_present_d3d12_window *w, bool skipped)
{
   if (!w->stats_enabled)
      return;
   struct agc_present_d3d12_stats *s = &w->stats;
   if (!s->frames && !s->skipped)
      s->first = w->timing.frame;
   if (skipped) {
      s->skipped++;
   } else {
      s->frames++;
      s->total_ns += w->timing.total_ns;
      s->wait_ns += w->timing.wait_ns;
      s->convert_ns += w->timing.convert_ns;
      s->present_ns += w->timing.present_ns;
   }
   if (s->frames + s->skipped < AGC_PRESENT_D3D12_STATS_INTERVAL)
      return;
   const double frames = MAX2(s->frames, 1);
   agc_stat("agc_present",
              "frames %" PRIu64 "-%" PRIu64 ": %.2f ms in the presenter a frame (swap chain wait %.2f, conversion "
              "%.2f, D3D12 copy and Present %.2f); %u frames not taken by the swap chain",
              s->first, w->timing.frame, (double)s->total_ns / 1e6 / frames, (double)s->wait_ns / 1e6 / frames,
              (double)s->convert_ns / 1e6 / frames, (double)s->present_ns / 1e6 / frames, s->skipped);
   memset(s, 0, sizeof(*s));
}

static int
agc_present_d3d12_attach(const struct agc_presentation_window *window, struct agc_driver_error *error)
{
   if (!window || !window->context || !window->native_window)
      return agc_present_d3d12_error(error, AGC_DRIVER_INVALID, "attach needs a native window");
   if (!agc_present_d3d12_open)
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "the presenter has no device to open");
   if (agc_present_d3d12_attached) {
      agc_present_d3d12_destroy(agc_present_d3d12_attached);
      agc_present_d3d12_attached = NULL;
   }
   struct agc_present_d3d12_window *w = calloc(1, sizeof(*w));
   if (!w)
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "out of memory for the presenter");
   w->context = window->context;
   w->hwnd = window->native_window;
   w->verify = agc_present_d3d12_verify;
   const char *stats = getenv("AGC_PRESENT_STATS");
   w->stats_enabled = stats && *stats && *stats != '0';
   int code = agc_present_d3d12_setup_native(w, error);
   if (code == AGC_DRIVER_OK)
      code = agc_present_d3d12_setup_d3d12(w, error);
   if (code != AGC_DRIVER_OK) {
      agc_present_d3d12_destroy(w);
      return code;
   }
   agc_present_d3d12_attached = w;
   return AGC_DRIVER_OK;
}

static bool
agc_present_d3d12_release(void *context)
{
   struct agc_present_d3d12_window *w = agc_present_d3d12_attached;
   if (!w || w->context != context)
      return false;
   agc_present_d3d12_attached = NULL;
   agc_present_d3d12_destroy(w);
   return true;
}

static bool
agc_present_d3d12_source_resident(struct agc_present_d3d12_window *w, const struct agc_present_source *source)
{
   if (!w->device.resident)
      return true;
   return w->device.resident(source->va, source->bytes) &&
          (!source->dcc || w->device.resident(source->meta_va, AGC_PRESENT_D3D12_META_CHECK_BYTES));
}

static int
agc_present_d3d12_capture_overlay(const struct agc_display_buffer *buffer, struct agc_driver_error *error)
{
   struct agc_present_d3d12_window *w = agc_present_d3d12_attached;
   if (!w || w->failed)
      return AGC_DRIVER_OK;
   if (!buffer) {
      w->snapshot_valid = false;
      return AGC_DRIVER_OK;
   }
   char text[320];
   struct agc_present_source source;
   if (!agc_present_source_of(buffer, w->device.info->gfx_level, &source, text, sizeof(text)))
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "overlay: %s", text);
   if (w->device.sync && !w->device.sync(text, sizeof(text)))
      return agc_present_d3d12_latch(w, error, "guest memory import: %s", text);
   if (!agc_present_d3d12_source_resident(w, &source))
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED,
                                     "overlay buffer 0x%" PRIx64 "+0x%" PRIx64 " or its metadata is not mapped for the "
                                     "GPU",
                                     source.va, source.bytes);
   const uint32_t pitch_bytes = align(source.width * 4, AGC_PRESENT_ROW_ALIGNMENT);
   const uint64_t bytes = (uint64_t)pitch_bytes * source.height;
   if (w->snapshot.bytes < bytes) {
      if (w->last_fence && !w->ops->fence_wait(w->queue, w->last_fence, AGC_PRESENT_D3D12_TIMEOUT_NS))
         return agc_present_d3d12_latch(w, error, "the presenter's queue did not go idle before the overlay grew");
      agc_mem_free(&w->snapshot);
      w->snapshot_valid = false;
      if (!agc_mem_alloc(align64(bytes, AGC_PRESENT_SLOT_GRANULARITY), 0, &w->snapshot))
         return agc_present_d3d12_latch(w, error, "no driver memory for the overlay snapshot: %s",
                                        agc_memory_last_error());
   }

   const struct radeon_info *info = w->device.info;
   uint32_t descriptor[AGC_PRESENT_DESCRIPTOR_DWORDS];
   if (!agc_present_descriptor(info, &source, descriptor, text, sizeof(text)))
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "overlay descriptor: %s", text);
   memcpy((uint8_t *)w->commands.cpu + AGC_PRESENT_D3D12_CAPTURE_DESCRIPTOR, descriptor, sizeof(descriptor));
   uint32_t *begin = (uint32_t *)((uint8_t *)w->commands.cpu + AGC_PRESENT_D3D12_CAPTURE_IB_OFFSET);
   uint32_t *p = begin;
   if (!w->queue_ready) {
      p = agc_kernel_emit_queue_setup(p, info, true);
      if (!p)
         return agc_present_d3d12_latch(w, error, "the presenter's compute queue setup could not be emitted");
   }
   p = agc_present_emit_acquire(p, agc_present_gcr_before());
   uint32_t user_data[AGC_DISPLAY_CONVERT_USER_DATA_DWORDS];
   agc_kernel_display_convert_user_data(user_data, w->commands.va + AGC_PRESENT_D3D12_CAPTURE_DESCRIPTOR,
                                        w->snapshot.va, source.width, source.height, pitch_bytes);
   p = agc_kernel_emit_dispatch(p, info, &w->convert, w->code.va, user_data,
                                agc_kernel_groups(source.width, w->convert.block[0]),
                                agc_kernel_groups(source.height, w->convert.block[1]), 1);
   if (!p)
      return agc_present_d3d12_latch(w, error, "the overlay capture dispatch could not be emitted");
   p = agc_present_emit_partial_flush(p);
   p = agc_present_emit_acquire(p, agc_present_gcr_after());
   const uint32_t multiple = MAX2(info->ip[AMD_IP_COMPUTE].ib_pad_dw_mask + 1, AGC_PRESENT_D3D12_IB_MULTIPLE);
   p = agc_present_emit_padding(p, agc_present_padding((uint32_t)(p - begin), multiple));
   const struct agc_ib ib = {
      .va = w->commands.va + AGC_PRESENT_D3D12_CAPTURE_IB_OFFSET,
      .dwords = (uint32_t)(p - begin),
      .flags = AGC_IB_MAIN,
   };
   agc_mem_flush_write_combining();
   const uint64_t fence = w->ops->submit(w->queue, &ib, 1);
   if (!fence) {
      agc_present_d3d12_device_text(w, text, sizeof(text));
      return agc_present_d3d12_latch(w, error, "the kernel refused the overlay capture (%s)", text);
   }
   w->queue_ready = true;
   w->last_fence = fence;
   if (!w->ops->fence_wait(w->queue, fence, AGC_PRESENT_D3D12_TIMEOUT_NS)) {
      agc_present_d3d12_device_text(w, text, sizeof(text));
      return agc_present_d3d12_latch(w, error, "the overlay capture did not complete (%s)", text);
   }
   if (!w->snapshot_valid)
      agc_stat("agc_present", "overlay %ux%u (mode %u%s) captured for composition", source.width, source.height,
               source.swizzle_mode, source.dcc ? ", DCC" : "");
   w->snapshot_source = (struct agc_present_source){
      .va = w->snapshot.va,
      .bytes = bytes,
      .width = source.width,
      .height = source.height,
      .pitch = pitch_bytes / 4,
      .custom_pitch = pitch_bytes / 4 != align(source.width, AGC_PRESENT_ROW_ALIGNMENT / 4),
      .texel_bytes = 4,
      .swizzle_mode = AGC_PRESENT_SW_LINEAR,
      .pipe_format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .swizzle = {PIPE_SWIZZLE_X, PIPE_SWIZZLE_Y, PIPE_SWIZZLE_Z, PIPE_SWIZZLE_W},
   };
   w->snapshot_valid = true;
   return AGC_DRIVER_OK;
}

static int
agc_present_d3d12_present(const struct agc_presentation_window *window, const struct agc_display_buffer *buffer,
                          const struct agc_display_buffer *plane, bool *presented, struct agc_driver_error *error)
{
   const int64_t start = os_time_get_nano();
   if (presented)
      *presented = false;
   struct agc_present_d3d12_window *w = agc_present_d3d12_attached;
   if (!window || !presented)
      return agc_present_d3d12_error(error, AGC_DRIVER_INVALID, "null window or result");
   if (!w || w->context != window->context || w->hwnd != window->native_window)
      return agc_present_d3d12_error(error, AGC_DRIVER_INVALID, "presentation came from a different window or thread");
   if (w->failed)
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "%s", w->failure);

   memset(&w->timing, 0, sizeof(w->timing));
   const uint64_t frame = w->frames + 1;
   w->timing.frame = frame;
   char text[320];
   struct agc_present_source source;
   if (buffer) {
      if (!agc_present_source_of(buffer, w->device.info->gfx_level, &source, text, sizeof(text)))
         return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "%s", text);
      if (w->device.sync && !w->device.sync(text, sizeof(text)))
         return agc_present_d3d12_latch(w, error, "guest memory import: %s", text);
      if (!agc_present_d3d12_source_resident(w, &source))
         return agc_present_d3d12_error(error, AGC_DRIVER_FAILED,
                                        "display buffer 0x%" PRIx64 "+0x%" PRIx64 " or its metadata is not mapped for "
                                        "the GPU",
                                        source.va, source.bytes);
   }

   RECT client;
   if (!GetClientRect((HWND)w->hwnd, &client))
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "GetClientRect failed (error %lu)", GetLastError());
   const uint32_t client_width = (uint32_t)MAX2(client.right - client.left, 0);
   const uint32_t client_height = (uint32_t)MAX2(client.bottom - client.top, 0);
   if (!client_width || !client_height) {
      w->timing.total_ns = os_time_get_nano() - start;
      agc_present_d3d12_note(w, true);
      return AGC_DRIVER_OK;
   }
   struct agc_present_canvas canvas;
   const uint32_t image_width = buffer ? source.width : client_width;
   const uint32_t image_height = buffer ? source.height : client_height;
   if (!agc_present_canvas_fit(image_width, image_height, client_width, client_height, &canvas, text, sizeof(text)))
      return agc_present_d3d12_error(error, AGC_DRIVER_FAILED, "%s", text);
   int code = agc_present_d3d12_fit(w, &canvas, error);
   if (code != AGC_DRIVER_OK)
      return w->lost ? agc_present_d3d12_recover(w, start, error) : code;

   const int64_t wait_start = os_time_get_nano();
   const DWORD waited = WaitForSingleObjectEx(w->waitable, AGC_PRESENT_D3D12_FRAME_WAIT_MS, FALSE);
   w->timing.wait_ns = os_time_get_nano() - wait_start;
   if (waited != WAIT_OBJECT_0) {
      w->timing.total_ns = os_time_get_nano() - start;
      agc_present_d3d12_note(w, true);
      return AGC_DRIVER_OK;
   }

   const uint32_t slot = agc_present_frame_slot(frame);
   const int64_t convert_start = os_time_get_nano();
   if (buffer) {
      code = agc_present_d3d12_convert(w, &source, slot, frame, plane && w->snapshot_valid, error);
      if (code != AGC_DRIVER_OK)
         return code;
   }
   if (!agc_wddm_queue_signal_fence(w->queue, w->ready_fence, frame)) {
      agc_present_d3d12_device_text(w, text, sizeof(text));
      return agc_present_d3d12_latch(w, error, "the presenter's queue cannot signal F_ready (%s)", text);
   }

   const int64_t present_start = os_time_get_nano();
   code = agc_present_d3d12_show(w, slot, frame, buffer ? w->slots[slot].resource : w->black, error);
   if (code != AGC_DRIVER_OK)
      return w->lost ? agc_present_d3d12_recover(w, start, error) : code;
   w->timing.present_ns = os_time_get_nano() - present_start;

   if (buffer && !w->ops->fence_wait(w->queue, w->last_fence, AGC_PRESENT_D3D12_TIMEOUT_NS)) {
      agc_present_d3d12_device_text(w, text, sizeof(text));
      return agc_present_d3d12_latch(w, error, "the display conversion did not complete within %" PRId64 " ms (%s)",
                                     AGC_PRESENT_D3D12_TIMEOUT_NS / 1000000, text);
   }
   w->timing.convert_ns = os_time_get_nano() - convert_start - w->timing.present_ns;

   w->frames = frame;
   w->frame.number = frame;
   w->frame.cleared = !buffer;
   w->frame.slot = slot;
   w->frame.va = buffer ? w->slots[slot].va : 0;
   w->frame.width = image_width;
   w->frame.height = image_height;
   w->frame.row_pitch = w->footprint.row_pitch;
   w->frame.image_offset = agc_present_image_offset(&w->canvas, &w->footprint);
   w->frame.rect = (struct agc_present_rect){
      .x = (int32_t)canvas.image_x,
      .y = (int32_t)canvas.image_y,
      .width = canvas.image_width,
      .height = canvas.image_height,
   };
   w->timing.presented = true;
   w->timing.total_ns = os_time_get_nano() - start;
   agc_present_d3d12_note(w, false);
   *presented = true;
   return AGC_DRIVER_OK;
}

static const struct agc_presenter agc_present_d3d12_table = {
   .attach = agc_present_d3d12_attach,
   .release = agc_present_d3d12_release,
   .capture_overlay = agc_present_d3d12_capture_overlay,
   .present = agc_present_d3d12_present,
   .display_buffer_size = agc_present_display_buffer_size,
};

const struct agc_presenter *
agc_present_d3d12(agc_present_open_device open)
{
   agc_present_d3d12_open = open;
   agc_present_d3d12_verify = false;
   return &agc_present_d3d12_table;
}

const struct agc_presenter *
agc_present_d3d12_verified(agc_present_open_device open)
{
   agc_present_d3d12_open = open;
   agc_present_d3d12_verify = true;
   return &agc_present_d3d12_table;
}

bool
agc_present_last_timing(struct agc_present_timing *timing)
{
   const struct agc_present_d3d12_window *w = agc_present_d3d12_attached;
   if (!w || !timing)
      return false;
   *timing = w->timing;
   return true;
}

bool
agc_present_last_frame(struct agc_present_frame *frame)
{
   struct agc_present_d3d12_window *w = agc_present_d3d12_attached;
   if (!w || !frame || !w->frames)
      return false;
   *frame = w->frame;
   frame->cpu = NULL;
   if (w->verify && !w->frame.cleared && w->slots[w->frame.slot].readback_cpu &&
       agc_present_d3d12_wait_copied(w, w->frame.number))
      frame->cpu = (const uint8_t *)w->slots[w->frame.slot].readback_cpu + w->frame.image_offset;
   return true;
}
