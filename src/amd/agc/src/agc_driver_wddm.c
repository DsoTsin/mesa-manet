#include "agc_driver_ops.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ac_gpu_info.h"
#include "ac_linux_drm.h"
#include "addrlib/src/amdgpu_asic_addr.h"
#include "amdgfxregs.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_atomic.h"
#include "util/u_math.h"

#include "agc_debug.h"
#include "agc_driver_wddm_private.h"

#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>

#define AGC_WDDM_VENDOR_AMD 0x1002u
#define AGC_WDDM_ADAPTER_LIMIT 256u
#define AGC_WDDM_NODE_LIMIT 256u
#define AGC_WDDM_GPU_PAGE_BYTES UINT64_C(4096)
#define AGC_WDDM_RESERVATION_ALIGNMENT UINT64_C(0x10000)
#define AGC_WDDM_BUFFER_ALIGNMENT UINT64_C(0x10000)
#define AGC_WDDM_HOST_ALIGNMENT UINT64_C(4096)
#define AGC_WDDM_IMPORT_CHUNK_BYTES (UINT64_C(128) << 20)
#define AGC_WDDM_DRIVER_VA_BASE UINT64_C(0x20000000000)
#define AGC_WDDM_FENCE_TIMEOUT_NS (INT64_C(5) * 1000000000)
#define AGC_WDDM_PAGING_TIMEOUT_NS (INT64_C(60) * 1000000000)
#define AGC_WDDM_WAIT_SLICE_MS UINT64_C(1000)
#define AGC_WDDM_POLL_INTERVAL_MS 1u
#define AGC_WDDM_MAX_IB_DWORDS 0xfffffu
#define AGC_WDDM_ATTRIBUTE_RING_ALIGNMENT (64u * 1024u)

#define AGC_WDDM_DRIVER_VERSION(product, version, build, revision)                                 \
   (((uint64_t)(product) << 48) | ((uint64_t)(version) << 32) | ((uint64_t)(build) << 16) |       \
    (uint64_t)(revision))
#define AGC_WDDM_IP_VERSION(major, minor, revision) (((major) << 16) | ((minor) << 8) | (revision))

_Static_assert(sizeof(D3DKMT_ENUMADAPTERS2) == 16, "D3DKMT_ENUMADAPTERS2");
_Static_assert(sizeof(D3DKMT_ADAPTERINFO) == 20, "D3DKMT_ADAPTERINFO");
_Static_assert(sizeof(D3DKMT_QUERYADAPTERINFO) == 24, "D3DKMT_QUERYADAPTERINFO");
_Static_assert(sizeof(D3DKMT_QUERY_DEVICE_IDS) == 28, "D3DKMT_QUERY_DEVICE_IDS");
_Static_assert(sizeof(D3DKMT_KMD_DRIVER_VERSION) == 8, "D3DKMT_KMD_DRIVER_VERSION");
_Static_assert(sizeof(D3DKMT_QUERY_GPUMMU_CAPS) == 12, "D3DKMT_QUERY_GPUMMU_CAPS");
_Static_assert(sizeof(D3DKMT_QUERYSTATISTICS) == 808, "D3DKMT_QUERYSTATISTICS");
_Static_assert(sizeof(D3DKMT_NODEMETADATA) == 78, "D3DKMT_NODEMETADATA");
_Static_assert(sizeof(D3DKMT_CREATEDEVICE) == 64, "D3DKMT_CREATEDEVICE");
_Static_assert(sizeof(D3DKMT_CREATEPAGINGQUEUE) == 32, "D3DKMT_CREATEPAGINGQUEUE");
_Static_assert(sizeof(D3DKMT_CREATECONTEXTVIRTUAL) == 40, "D3DKMT_CREATECONTEXTVIRTUAL");
_Static_assert(sizeof(D3DKMT_CREATEHWQUEUE) == 48, "D3DKMT_CREATEHWQUEUE");
_Static_assert(sizeof(D3DKMT_SUBMITCOMMANDTOHWQUEUE) == 56, "D3DKMT_SUBMITCOMMANDTOHWQUEUE");
_Static_assert(sizeof(D3DKMT_CREATEALLOCATION) == 72, "D3DKMT_CREATEALLOCATION");
_Static_assert(sizeof(D3DDDI_ALLOCATIONINFO2) == 96, "D3DDDI_ALLOCATIONINFO2");
_Static_assert(sizeof(D3DKMT_DESTROYALLOCATION2) == 24, "D3DKMT_DESTROYALLOCATION2");
_Static_assert(sizeof(D3DDDI_MAKERESIDENT) == 48, "D3DDDI_MAKERESIDENT");
_Static_assert(sizeof(D3DDDI_MAPGPUVIRTUALADDRESS) == 104, "D3DDDI_MAPGPUVIRTUALADDRESS");
_Static_assert(sizeof(D3DDDI_RESERVEGPUVIRTUALADDRESS) == 72, "D3DDDI_RESERVEGPUVIRTUALADDRESS");
_Static_assert(sizeof(D3DKMT_FREEGPUVIRTUALADDRESS) == 24, "D3DKMT_FREEGPUVIRTUALADDRESS");
_Static_assert(sizeof(D3DKMT_LOCK2) == 24, "D3DKMT_LOCK2");
_Static_assert(sizeof(D3DKMT_UNLOCK2) == 8, "D3DKMT_UNLOCK2");
_Static_assert(sizeof(D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU) == 40,
               "D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU");
_Static_assert(sizeof(D3DKMT_GETDEVICESTATE) == 56, "D3DKMT_GETDEVICESTATE");
_Static_assert(sizeof(D3DKMT_CREATESYNCHRONIZATIONOBJECT2) == 96,
               "D3DKMT_CREATESYNCHRONIZATIONOBJECT2");
_Static_assert(sizeof(D3DKMT_OPENSYNCOBJECTFROMNTHANDLE2) == 88,
               "D3DKMT_OPENSYNCOBJECTFROMNTHANDLE2");
_Static_assert(sizeof(D3DKMT_QUERYRESOURCEINFOFROMNTHANDLE) == 40,
               "D3DKMT_QUERYRESOURCEINFOFROMNTHANDLE");
_Static_assert(sizeof(D3DKMT_OPENRESOURCEFROMNTHANDLE) == 104, "D3DKMT_OPENRESOURCEFROMNTHANDLE");
_Static_assert(sizeof(D3DDDI_OPENALLOCATIONINFO2) == 80, "D3DDDI_OPENALLOCATIONINFO2");
_Static_assert(sizeof(D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE) == 24,
               "D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE");
_Static_assert(sizeof(D3DKMT_SUBMITSIGNALSYNCOBJECTSTOHWQUEUE) == 40,
               "D3DKMT_SUBMITSIGNALSYNCOBJECTSTOHWQUEUE");
_Static_assert(offsetof(D3DKMT_CREATEHWQUEUE, HwQueueProgressFenceCPUVirtualAddress) == 32,
               "hardware queue progress fence mapping");
_Static_assert(offsetof(D3DKMT_SUBMITCOMMANDTOHWQUEUE, pPrivateDriverData) == 32,
               "submission private data");
_Static_assert(offsetof(D3DDDI_MAPGPUVIRTUALADDRESS, Protection) == 56, "mapping protection");
_Static_assert(offsetof(D3DDDI_MAPGPUVIRTUALADDRESS, VirtualAddress) == 88, "mapped address");
_Static_assert(offsetof(D3DDDI_MAPGPUVIRTUALADDRESS, PagingFenceValue) == 96,
               "mapping paging fence");
_Static_assert(sizeof(DXGK_FAULT_ERROR_CODE) == sizeof(uint32_t), "fault error code");
_Static_assert(AGC_WDDM_DRIVER_VA_BASE % AGC_WDDM_RESERVATION_ALIGNMENT == 0, "driver window base");
_Static_assert(AGC_WDDM_IMPORT_CHUNK_BYTES % AGC_WDDM_GPU_PAGE_BYTES == 0, "import chunk");
_Static_assert(AGC_WDDM_IMPORT_CHUNK_BYTES <= UINT32_MAX, "import chunk record size");

struct agc_wddm_profile {
   const char *name;
   uint32_t device_id;
   uint32_t revision;
   uint64_t driver_version;
   uint32_t abi_version;
   uint8_t device_heaps[2];
   uint8_t host_heaps[2];
   uint64_t va_end;
   uint32_t ib_start_alignment;
   uint32_t ib_size_alignment;
   enum radeon_family chip;
   enum amd_gfx_level gfx_level;
   uint32_t family_id;
   uint32_t external_rev;
   uint32_t gfx_ip_version;
   uint32_t gb_addr_config;
   uint32_t num_shader_engines;
   uint32_t num_shader_arrays_per_engine;
   uint32_t num_cu_per_shader_array;
   uint32_t num_rb_pipes;
   uint32_t mall_size_mb;
   bool integrated;
};

static const struct agc_wddm_profile agc_wddm_profiles[] = {
   {
      .name = "amd-v9-1586-32.0.32015.7029",
      .device_id = 0x1586,
      .revision = 0xc1,
      .driver_version = AGC_WDDM_DRIVER_VERSION(32, 0, 32015, 7029),
      .abi_version = AGC_WDDM_ABI_VERSION,
      .device_heaps = {4, 3},
      .host_heaps = {4, 0},
      .va_end = UINT64_C(0x7fffffffffff),
      .ib_start_alignment = 256,
      .ib_size_alignment = 32,
      .chip = CHIP_STRIX_HALO,
      .gfx_level = GFX11_5,
      .family_id = FAMILY_STX,
      .external_rev = 0xc1,
      .gfx_ip_version = AGC_WDDM_IP_VERSION(11, 5, 1),
      .gb_addr_config = 0x343,
      .num_shader_engines = 2,
      .num_shader_arrays_per_engine = 2,
      .num_cu_per_shader_array = 10,
      .num_rb_pipes = 8,
      .mall_size_mb = 32,
      .integrated = true,
   },
};

enum agc_wddm_wait {
   AGC_WDDM_WAIT_REACHED,
   AGC_WDDM_WAIT_TIMEOUT,
   AGC_WDDM_WAIT_FAILED
};

struct agc_wddm_monitored_fence {
   D3DKMT_HANDLE object;
   const volatile uint64_t *value;
};

struct agc_wddm_reservation {
   uint64_t base;
   uint64_t size;
};

struct agc_wddm_adapter_entry {
   D3DKMT_HANDLE handle;
   struct agc_wddm_adapter info;
};

struct agc_wddm_backend {
   struct agc_backend base;
   const struct agc_wddm_profile *profile;
   struct agc_wddm_adapter identity;
   D3DKMT_HANDLE adapter;
   D3DKMT_HANDLE device;
   D3DKMT_HANDLE paging;
   struct agc_wddm_monitored_fence paging_fence;
   uint32_t va_bits;
   uint64_t va_end;
   struct agc_wddm_node *nodes;
   uint32_t node_count;
   uint32_t gfx_node;
   uint32_t compute_node;
   simple_mtx_t lock;
   struct agc_wddm_reservation *reservations;
   uint32_t reservation_count;
   uint32_t reservation_capacity;
   uint64_t paging_pending;
   int32_t lost;
   int32_t live_allocations;
   int32_t live_queues;
   enum agc_wddm_chip_source chip_source;
   int32_t chip_basic_status;
   int32_t chip_gpu_status;
};

struct agc_wddm_chunk {
   D3DKMT_HANDLE handle;
   D3DKMT_HANDLE resource;
};

struct agc_bo {
   struct agc_wddm_backend *be;
   uint64_t va;
   uint64_t size;
   uint64_t chunk_bytes;
   void *cpu;
   bool locked;
   bool reserved;
   bool execute;
   uint32_t chunk_count;
   struct agc_wddm_chunk chunks[];
};

struct agc_hwq {
   struct agc_wddm_backend *be;
   enum agc_queue_kind kind;
   uint32_t node;
   D3DKMT_HANDLE context;
   D3DKMT_HANDLE handle;
   uint32_t id;
   struct agc_wddm_monitored_fence progress;
   uint64_t progress_gpu_va;
   simple_mtx_t lock;
   uint64_t submitted;
};

struct agc_wddm_fence {
   struct agc_wddm_backend *be;
   struct agc_wddm_monitored_fence fence;
   HANDLE nt_handle;
};

static const struct agc_driver_ops agc_wddm_ops;

static struct agc_wddm_backend *
agc_wddm_backend(struct agc_backend *base)
{
   return (struct agc_wddm_backend *)base;
}

static bool
agc_wddm_failed(NTSTATUS status, const char *function, const char *operation)
{
   if (status >= 0)
      return false;
   agc_report(function, "%s: NTSTATUS=0x%08x", operation, (unsigned)status);
   return true;
}

static NTSTATUS
agc_wddm_query_adapter(D3DKMT_HANDLE adapter, KMTQUERYADAPTERINFOTYPE type, void *data,
                       uint32_t bytes)
{
   D3DKMT_QUERYADAPTERINFO query = {
      .hAdapter = adapter,
      .Type = type,
      .pPrivateDriverData = data,
      .PrivateDriverDataSize = bytes,
   };
   return D3DKMTQueryAdapterInfo(&query);
}

static const char *
agc_wddm_execution_name(int32_t execution)
{
   switch (execution) {
   case D3DKMT_DEVICEEXECUTION_ACTIVE:
      return "active";
   case D3DKMT_DEVICEEXECUTION_RESET:
      return "reset";
   case D3DKMT_DEVICEEXECUTION_HUNG:
      return "hung";
   case D3DKMT_DEVICEEXECUTION_STOPPED:
      return "stopped";
   case D3DKMT_DEVICEEXECUTION_ERROR_OUTOFMEMORY:
      return "out-of-memory";
   case D3DKMT_DEVICEEXECUTION_ERROR_DMAFAULT:
      return "dma-fault";
   case D3DKMT_DEVICEEXECUTION_ERROR_DMAPAGEFAULT:
      return "dma-page-fault";
   default:
      return "other";
   }
}

void
agc_wddm_device_state_detail(struct agc_backend *base, struct agc_wddm_device_state *state)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   memset(state, 0, sizeof(*state));

   D3DKMT_GETDEVICESTATE execution = {
      .hDevice = be->device,
      .StateType = D3DKMT_DEVICESTATE_EXECUTION,
   };
   state->execution_status = D3DKMTGetDeviceState(&execution);
   if (state->execution_status >= 0)
      state->execution = execution.ExecutionState;

   D3DKMT_GETDEVICESTATE fault = {
      .hDevice = be->device,
      .StateType = D3DKMT_DEVICESTATE_PAGE_FAULT,
   };
   state->page_fault_status = D3DKMTGetDeviceState(&fault);
   if (state->page_fault_status >= 0) {
      state->fault_va = fault.PageFaultState.FaultedVirtualAddress;
      state->fault_sequence = fault.PageFaultState.FaultedPrimitiveAPISequenceNumber;
      state->fault_stage = (uint32_t)fault.PageFaultState.FaultedPipelineStage;
      state->fault_bind_table_entry = fault.PageFaultState.FaultedBindTableEntry;
      state->fault_flags = (uint32_t)fault.PageFaultState.PageFaultFlags;
      memcpy(&state->fault_error_code, &fault.PageFaultState.FaultErrorCode,
             sizeof(state->fault_error_code));
   }
}

size_t
agc_wddm_device_state_format(const struct agc_wddm_device_state *state, char *text, size_t size)
{
   int used;
   if (state->execution_status >= 0) {
      used = snprintf(text, size, "execution=%s (%d, 0x%08x)",
                      agc_wddm_execution_name(state->execution), state->execution,
                      (unsigned)state->execution_status);
   } else {
      used = snprintf(text, size, "execution=query-failed (0x%08x)",
                      (unsigned)state->execution_status);
   }
   if (used < 0 || (size_t)used >= size)
      return size ? size - 1 : 0;

   int more;
   if (state->page_fault_status >= 0) {
      more = snprintf(text + used, size - (size_t)used,
                      " page_fault_va=0x%" PRIx64 " flags=0x%x code=0x%x device_specific=%u"
                      " stage=%u bind_table_entry=%u sequence=%" PRIu64 " (0x%08x)",
                      state->fault_va, state->fault_flags, state->fault_error_code >> 1,
                      state->fault_error_code & 1, state->fault_stage,
                      state->fault_bind_table_entry, state->fault_sequence,
                      (unsigned)state->page_fault_status);
   } else {
      more = snprintf(text + used, size - (size_t)used, " page_fault=query-failed (0x%08x)",
                      (unsigned)state->page_fault_status);
   }
   if (more < 0 || (size_t)more >= size - (size_t)used)
      return size - 1;
   return (size_t)(used + more);
}

static void
agc_wddm_poison(struct agc_wddm_backend *be, const char *function, const char *reason)
{
   if (p_atomic_xchg(&be->lost, 1))
      return;
   struct agc_wddm_device_state state;
   char text[320];
   agc_wddm_device_state_detail(&be->base, &state);
   agc_wddm_device_state_format(&state, text, sizeof(text));
   agc_report(function, "device lost: %s; %s", reason, text);
}

static void
agc_wddm_poison_status(struct agc_wddm_backend *be, const char *function, const char *operation,
                       NTSTATUS status)
{
   char reason[160];
   snprintf(reason, sizeof(reason), "%s NTSTATUS=0x%08x", operation, (unsigned)status);
   agc_wddm_poison(be, function, reason);
}

static bool
agc_wddm_healthy(struct agc_wddm_backend *be, const char *function)
{
   if (!p_atomic_read(&be->lost))
      return true;
   agc_report(function, "the device is lost; no further GPU work is accepted");
   return false;
}

static bool
agc_wddm_fence_current(const struct agc_wddm_monitored_fence *fence, uint64_t *value)
{
   if (!fence->value)
      return false;
   *value = *fence->value;
   return true;
}

static bool
agc_wddm_fence_reached(const struct agc_wddm_monitored_fence *fence, uint64_t target)
{
   uint64_t value;
   return agc_wddm_fence_current(fence, &value) && value >= target;
}

static enum agc_wddm_wait
agc_wddm_fence_reach(struct agc_wddm_backend *be, const struct agc_wddm_monitored_fence *fence,
                     uint64_t target, int64_t timeout_ns)
{
   if (agc_wddm_fence_reached(fence, target))
      return AGC_WDDM_WAIT_REACHED;
   if (timeout_ns == 0)
      return AGC_WDDM_WAIT_TIMEOUT;

   int64_t start = os_time_get_nano();
   int64_t deadline = INT64_MAX;
   if (timeout_ns > 0 && timeout_ns <= INT64_MAX - start)
      deadline = start + timeout_ns;

   bool armed = false;
   HANDLE event = CreateEventW(NULL, FALSE, FALSE, NULL);
   if (event) {
      D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU wait = {
         .hDevice = be->device,
         .ObjectCount = 1,
         .ObjectHandleArray = &fence->object,
         .FenceValueArray = &target,
         .hAsyncEvent = event,
      };
      armed = !agc_wddm_failed(D3DKMTWaitForSynchronizationObjectFromCpu(&wait), __func__,
                               "WaitForSynchronizationObjectFromCpu");
   } else {
      agc_report(__func__, "CreateEventW failed: %lu", GetLastError());
   }

   enum agc_wddm_wait result = AGC_WDDM_WAIT_TIMEOUT;
   for (;;) {
      if (agc_wddm_fence_reached(fence, target)) {
         result = AGC_WDDM_WAIT_REACHED;
         break;
      }
      if (p_atomic_read(&be->lost)) {
         result = AGC_WDDM_WAIT_FAILED;
         break;
      }
      int64_t now = os_time_get_nano();
      if (now >= deadline)
         break;
      uint64_t remaining_ms = ((uint64_t)(deadline - now) + 999999) / 1000000;
      if (armed) {
         bool signaled =
            WaitForSingleObject(event, (DWORD)MIN2(remaining_ms, AGC_WDDM_WAIT_SLICE_MS)) ==
            WAIT_OBJECT_0;
         if (signaled && !fence->value) {
            result = AGC_WDDM_WAIT_REACHED;
            break;
         }
      } else if (fence->value) {
         Sleep(AGC_WDDM_POLL_INTERVAL_MS);
      } else {
         agc_report(__func__, "the fence has no CPU-visible value to poll");
         result = AGC_WDDM_WAIT_FAILED;
         break;
      }
   }

   if (event)
      CloseHandle(event);
   return result;
}

static void
agc_wddm_paging_note(struct agc_wddm_backend *be, uint64_t value)
{
   uint64_t seen = p_atomic_read(&be->paging_pending);
   while (value > seen) {
      uint64_t old = (uint64_t)p_atomic_cmpxchg(&be->paging_pending, seen, value);
      if (old == seen)
         break;
      seen = old;
   }
}

static bool
agc_wddm_paging_wait(struct agc_wddm_backend *be, uint64_t value, const char *function)
{
   if (agc_wddm_fence_reach(be, &be->paging_fence, value, AGC_WDDM_PAGING_TIMEOUT_NS) ==
       AGC_WDDM_WAIT_REACHED)
      return true;

   char reason[96];
   snprintf(reason, sizeof(reason), "the paging fence did not reach %" PRIu64 " in %" PRId64 " ms",
            value, AGC_WDDM_PAGING_TIMEOUT_NS / 1000000);
   agc_wddm_poison(be, function, reason);
   return false;
}

static bool
agc_wddm_paging_settle(struct agc_backend *base)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   if (!agc_wddm_healthy(be, __func__))
      return false;
   return agc_wddm_paging_wait(be, p_atomic_read(&be->paging_pending), __func__);
}

static void
agc_wddm_close_adapter(D3DKMT_HANDLE adapter)
{
   if (!adapter)
      return;
   D3DKMT_CLOSEADAPTER close = {.hAdapter = adapter};
   D3DKMTCloseAdapter(&close);
}

static const struct agc_wddm_profile *
agc_wddm_find_profile(const struct agc_wddm_adapter *adapter)
{
   if (adapter->vendor_id != AGC_WDDM_VENDOR_AMD)
      return NULL;
   for (size_t i = 0; i < ARRAY_SIZE(agc_wddm_profiles); i++) {
      const struct agc_wddm_profile *profile = &agc_wddm_profiles[i];
      if (adapter->device_id == profile->device_id && adapter->revision == profile->revision &&
          adapter->driver_version == profile->driver_version &&
          profile->abi_version == AGC_WDDM_ABI_VERSION)
         return profile;
   }
   return NULL;
}

static bool
agc_wddm_enumerate_open(struct agc_wddm_adapter_entry **out, uint32_t *out_count)
{
   *out = NULL;
   *out_count = 0;

   D3DKMT_ENUMADAPTERS2 enumerate = {0};
   if (agc_wddm_failed(D3DKMTEnumAdapters2(&enumerate), __func__, "EnumAdapters2(count)"))
      return false;
   if (enumerate.NumAdapters > AGC_WDDM_ADAPTER_LIMIT) {
      agc_report(__func__, "unreasonable adapter count %lu", enumerate.NumAdapters);
      return false;
   }
   if (enumerate.NumAdapters == 0)
      return true;

   D3DKMT_ADAPTERINFO *infos = calloc(enumerate.NumAdapters, sizeof(*infos));
   struct agc_wddm_adapter_entry *entries = calloc(enumerate.NumAdapters, sizeof(*entries));
   if (!infos || !entries) {
      free(infos);
      free(entries);
      agc_report(__func__, "out of memory");
      return false;
   }

   uint32_t capacity = enumerate.NumAdapters;
   enumerate.pAdapters = infos;
   if (agc_wddm_failed(D3DKMTEnumAdapters2(&enumerate), __func__, "EnumAdapters2")) {
      free(infos);
      free(entries);
      return false;
   }

   uint32_t count = 0;
   for (uint32_t i = 0; i < MIN2(enumerate.NumAdapters, capacity); i++) {
      D3DKMT_QUERY_DEVICE_IDS ids = {0};
      if (agc_wddm_query_adapter(infos[i].hAdapter, KMTQAITYPE_PHYSICALADAPTERDEVICEIDS, &ids,
                                 sizeof(ids)) < 0) {
         agc_wddm_close_adapter(infos[i].hAdapter);
         continue;
      }
      D3DKMT_KMD_DRIVER_VERSION version = {0};
      NTSTATUS status = agc_wddm_query_adapter(infos[i].hAdapter, KMTQAITYPE_KMD_DRIVER_VERSION,
                                               &version, sizeof(version));
      if (status < 0) {
         agc_report(__func__, "adapter %04x:%04x gives no kernel driver version: NTSTATUS=0x%08x",
                    ids.DeviceIds.VendorID, ids.DeviceIds.DeviceID, (unsigned)status);
         agc_wddm_close_adapter(infos[i].hAdapter);
         continue;
      }

      struct agc_wddm_adapter_entry *entry = &entries[count++];
      entry->handle = infos[i].hAdapter;
      entry->info = (struct agc_wddm_adapter){
         .vendor_id = ids.DeviceIds.VendorID,
         .device_id = ids.DeviceIds.DeviceID,
         .sub_vendor_id = ids.DeviceIds.SubVendorID,
         .subsystem_id = ids.DeviceIds.SubSystemID,
         .revision = ids.DeviceIds.RevisionID,
         .luid_low = infos[i].AdapterLuid.LowPart,
         .luid_high = infos[i].AdapterLuid.HighPart,
         .driver_version = (uint64_t)version.DriverVersion.QuadPart,
      };
      const struct agc_wddm_profile *profile = agc_wddm_find_profile(&entry->info);
      entry->info.profile = profile ? profile->name : NULL;
   }

   free(infos);
   *out = entries;
   *out_count = count;
   return true;
}

uint32_t
agc_wddm_enumerate(struct agc_wddm_adapter *adapters, uint32_t capacity)
{
   struct agc_wddm_adapter_entry *entries;
   uint32_t count;
   if (!agc_wddm_enumerate_open(&entries, &count))
      return 0;
   for (uint32_t i = 0; i < count; i++) {
      if (i < capacity)
         adapters[i] = entries[i].info;
      agc_wddm_close_adapter(entries[i].handle);
   }
   free(entries);
   return count;
}

static bool
agc_wddm_name_has_compute(const char *name)
{
   static const char word[] = "compute";
   for (const char *at = name; *at; at++) {
      size_t i = 0;
      while (word[i] && at[i] && (at[i] | 0x20) == word[i])
         i++;
      if (!word[i])
         return true;
   }
   return false;
}

static bool
agc_wddm_query_nodes(struct agc_wddm_backend *be)
{
   D3DKMT_QUERYSTATISTICS statistics = {
      .Type = D3DKMT_QUERYSTATISTICS_ADAPTER,
      .AdapterLuid = {.LowPart = be->identity.luid_low, .HighPart = be->identity.luid_high},
   };
   if (agc_wddm_failed(D3DKMTQueryStatistics(&statistics), __func__, "QueryStatistics"))
      return false;

   uint32_t count = MIN2(statistics.QueryResult.AdapterInformation.NodeCount, AGC_WDDM_NODE_LIMIT);
   be->nodes = calloc(MAX2(count, 1), sizeof(*be->nodes));
   if (!be->nodes) {
      agc_report(__func__, "out of memory");
      return false;
   }

   be->gfx_node = AGC_WDDM_NODE_NONE;
   be->compute_node = AGC_WDDM_NODE_NONE;
   for (uint32_t ordinal = 0; ordinal < count; ordinal++) {
      D3DKMT_NODEMETADATA metadata = {.NodeOrdinalAndAdapterIndex = ordinal};
      if (agc_wddm_query_adapter(be->adapter, KMTQAITYPE_NODEMETADATA, &metadata,
                                 sizeof(metadata)) < 0)
         continue;

      struct agc_wddm_node *node = &be->nodes[be->node_count++];
      WCHAR wide[ARRAY_SIZE(metadata.NodeData.FriendlyName) + 1] = {0};
      memcpy(wide, metadata.NodeData.FriendlyName, sizeof(metadata.NodeData.FriendlyName));
      node->ordinal = ordinal;
      node->engine_type = metadata.NodeData.EngineType;
      if (!WideCharToMultiByte(CP_UTF8, 0, wide, -1, node->name, sizeof(node->name), NULL, NULL))
         node->name[0] = '\0';

      if (be->gfx_node == AGC_WDDM_NODE_NONE && node->engine_type == DXGK_ENGINE_TYPE_3D)
         be->gfx_node = ordinal;
      if (be->compute_node == AGC_WDDM_NODE_NONE && agc_wddm_name_has_compute(node->name))
         be->compute_node = ordinal;
   }
   return true;
}

static void
agc_wddm_close(struct agc_wddm_backend *be)
{
   if (be->paging) {
      D3DDDI_DESTROYPAGINGQUEUE destroy = {.hPagingQueue = be->paging};
      D3DKMTDestroyPagingQueue(&destroy);
   }
   if (be->device) {
      D3DKMT_DESTROYDEVICE destroy = {.hDevice = be->device};
      D3DKMTDestroyDevice(&destroy);
   }
   agc_wddm_close_adapter(be->adapter);
}

static void
agc_wddm_free(struct agc_wddm_backend *be)
{
   simple_mtx_destroy(&be->lock);
   free(be->reservations);
   free(be->nodes);
   free(be);
}

static struct agc_backend *
agc_wddm_open_adapter(const struct agc_wddm_adapter_entry *entry,
                      const struct agc_wddm_profile *profile)
{
   struct agc_wddm_backend *be = calloc(1, sizeof(*be));
   if (!be) {
      agc_report(__func__, "out of memory");
      agc_wddm_close_adapter(entry->handle);
      return NULL;
   }
   simple_mtx_init(&be->lock, mtx_plain);
   be->base.ops = &agc_wddm_ops;
   be->profile = profile;
   be->identity = entry->info;
   be->adapter = entry->handle;
   be->gfx_node = AGC_WDDM_NODE_NONE;
   be->compute_node = AGC_WDDM_NODE_NONE;

   D3DKMT_QUERY_GPUMMU_CAPS caps = {0};
   if (agc_wddm_query_adapter(be->adapter, KMTQAITYPE_QUERY_GPUMMU_CAPS, &caps, sizeof(caps)) >= 0)
      be->va_bits = caps.Caps.VirtualAddressBitCount;
   be->va_end = be->profile->va_end;
   if (be->va_bits >= 1 && be->va_bits <= 63)
      be->va_end = MIN2((UINT64_C(1) << be->va_bits) - 1, be->profile->va_end);
   if (be->va_end <= AGC_WDDM_DRIVER_VA_BASE) {
      agc_report(__func__, "%u GPU VA bits leave no room above the driver window base 0x%" PRIx64,
                 be->va_bits, AGC_WDDM_DRIVER_VA_BASE);
      goto fail;
   }

   D3DKMT_CREATEDEVICE create = {.hAdapter = be->adapter};
   if (agc_wddm_failed(D3DKMTCreateDevice(&create), __func__, "CreateDevice"))
      goto fail;
   be->device = create.hDevice;

   D3DKMT_CREATEPAGINGQUEUE paging = {.hDevice = be->device};
   if (agc_wddm_failed(D3DKMTCreatePagingQueue(&paging), __func__, "CreatePagingQueue"))
      goto fail;
   be->paging = paging.hPagingQueue;
   be->paging_fence.object = paging.hSyncObject;
   be->paging_fence.value = paging.FenceValueCPUVirtualAddress;

   if (!agc_wddm_query_nodes(be))
      goto fail;
   if (be->gfx_node == AGC_WDDM_NODE_NONE) {
      agc_report(__func__, "the adapter reports no graphics node");
      goto fail;
   }
   return &be->base;

fail:
   agc_wddm_close(be);
   agc_wddm_free(be);
   return NULL;
}

struct agc_backend *
agc_wddm_open(void)
{
   struct agc_wddm_adapter_entry *entries;
   uint32_t count;
   if (!agc_wddm_enumerate_open(&entries, &count))
      return NULL;

   struct agc_backend *be = NULL;
   bool found = false;
   for (uint32_t i = 0; i < count; i++) {
      const struct agc_wddm_profile *profile = agc_wddm_find_profile(&entries[i].info);
      if (profile && !found) {
         found = true;
         be = agc_wddm_open_adapter(&entries[i], profile);
      } else {
         agc_wddm_close_adapter(entries[i].handle);
      }
   }
   free(entries);
   if (!found)
      agc_report(__func__, "none of the %u adapters matches a verified profile", count);
   return be;
}

static void
agc_wddm_destroy(struct agc_backend *base)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   if (p_atomic_read(&be->live_queues) || p_atomic_read(&be->live_allocations)) {
      agc_report(__func__, "%d queues and %d allocations are still open",
                 (int)p_atomic_read(&be->live_queues), (int)p_atomic_read(&be->live_allocations));
   }
   if (p_atomic_read(&be->lost)) {
      agc_report(__func__, "the device is lost; handles and allocations stay until process exit");
      agc_wddm_free(be);
      return;
   }

   for (uint32_t i = be->reservation_count; i > 0; i--) {
      D3DKMT_FREEGPUVIRTUALADDRESS release = {
         .hAdapter = be->adapter,
         .BaseAddress = be->reservations[i - 1].base,
         .Size = be->reservations[i - 1].size,
      };
      agc_wddm_failed(D3DKMTFreeGpuVirtualAddress(&release), __func__,
                      "FreeGpuVirtualAddress(reservation)");
   }
   agc_wddm_close(be);
   agc_wddm_free(be);
}

void
agc_wddm_describe(struct agc_backend *base, struct agc_wddm_description *description)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   *description = (struct agc_wddm_description){
      .adapter = be->identity,
      .va_bits = be->va_bits,
      .va_end = be->va_end,
      .driver_va_base = AGC_WDDM_DRIVER_VA_BASE,
      .node_count = be->node_count,
      .gfx_node = be->gfx_node,
      .compute_node = be->compute_node,
      .live_allocations = (uint32_t)p_atomic_read(&be->live_allocations),
      .lost = p_atomic_read(&be->lost) != 0,
      .chip_source = be->chip_source,
      .chip_basic_status = be->chip_basic_status,
      .chip_gpu_status = be->chip_gpu_status,
   };
}

uint32_t
agc_wddm_nodes(struct agc_backend *base, struct agc_wddm_node *nodes, uint32_t capacity)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   for (uint32_t i = 0; i < MIN2(be->node_count, capacity); i++)
      nodes[i] = be->nodes[i];
   return be->node_count;
}

int32_t
agc_wddm_query_umd_private(struct agc_backend *base, void *data, uint32_t bytes)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   return agc_wddm_query_adapter(be->adapter, KMTQAITYPE_UMDRIVERPRIVATE, data, bytes);
}

bool
agc_wddm_video_memory(struct agc_backend *base, struct agc_wddm_video_memory *memory)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   D3DKMT_QUERYVIDEOMEMORYINFO local = {
      .hAdapter = be->adapter,
      .MemorySegmentGroup = D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL,
   };
   D3DKMT_QUERYVIDEOMEMORYINFO non_local = {
      .hAdapter = be->adapter,
      .MemorySegmentGroup = D3DKMT_MEMORY_SEGMENT_GROUP_NON_LOCAL,
   };
   memset(memory, 0, sizeof(*memory));
   if (agc_wddm_failed(D3DKMTQueryVideoMemoryInfo(&local), __func__,
                       "QueryVideoMemoryInfo(local)") ||
       agc_wddm_failed(D3DKMTQueryVideoMemoryInfo(&non_local), __func__,
                       "QueryVideoMemoryInfo(non-local)"))
      return false;
   memory->local_budget = local.Budget;
   memory->local_usage = local.CurrentUsage;
   memory->non_local_budget = non_local.Budget;
   memory->non_local_usage = non_local.CurrentUsage;
   return true;
}

static bool
agc_wddm_inside_reservation(struct agc_wddm_backend *be, uint64_t va, uint64_t size)
{
   bool inside = false;
   simple_mtx_lock(&be->lock);
   for (uint32_t i = 0; i < be->reservation_count && !inside; i++) {
      const struct agc_wddm_reservation *r = &be->reservations[i];
      inside = va >= r->base && size <= r->size && va - r->base <= r->size - size;
   }
   simple_mtx_unlock(&be->lock);
   return inside;
}

static bool
agc_wddm_check_fixed(struct agc_wddm_backend *be, uint64_t va, uint64_t size, uint64_t alignment,
                     const char *function)
{
   if (va != 0 && va % alignment == 0 && agc_wddm_inside_reservation(be, va, size))
      return true;
   agc_report(function, "0x%" PRIx64 "+0x%" PRIx64 " is not %" PRIu64
                        "-byte aligned inside a reservation",
              va, size, alignment);
   return false;
}

static bool
agc_wddm_free_va(struct agc_wddm_backend *be, uint64_t va, uint64_t size, const char *function,
                 const char *operation)
{
   D3DKMT_FREEGPUVIRTUALADDRESS release = {
      .hAdapter = be->adapter,
      .BaseAddress = va,
      .Size = size,
   };
   NTSTATUS status = D3DKMTFreeGpuVirtualAddress(&release);
   if (status >= 0)
      return true;
   agc_wddm_poison_status(be, function, operation, status);
   return false;
}

static bool
agc_wddm_reserve(struct agc_wddm_backend *be, uint64_t va, uint64_t size, bool zero,
                 const char *function)
{
   if (!agc_wddm_healthy(be, function))
      return false;
   if (va == 0 || size == 0 || va % AGC_WDDM_RESERVATION_ALIGNMENT != 0 ||
       size % AGC_WDDM_RESERVATION_ALIGNMENT != 0 || size - 1 > be->va_end ||
       va > be->va_end - (size - 1)) {
      agc_report(function, "VA window 0x%" PRIx64 "+0x%" PRIx64
                           " is not 64 KiB aligned below va_end 0x%" PRIx64,
                 va, size, be->va_end);
      return false;
   }

   simple_mtx_lock(&be->lock);
   if (be->reservation_count == be->reservation_capacity) {
      uint32_t capacity = MAX2(be->reservation_capacity * 2, 8);
      struct agc_wddm_reservation *grown =
         realloc(be->reservations, capacity * sizeof(*be->reservations));
      if (grown) {
         be->reservations = grown;
         be->reservation_capacity = capacity;
      }
   }
   bool room = be->reservation_count < be->reservation_capacity;
   simple_mtx_unlock(&be->lock);
   if (!room) {
      agc_report(function, "out of memory");
      return false;
   }

   D3DDDI_RESERVEGPUVIRTUALADDRESS reserve = {
      .BaseAddress = va,
      .Size = size,
      .ReservationType = zero ? D3DDDIGPUVIRTUALADDRESS_RESERVE_ZERO
                              : D3DDDIGPUVIRTUALADDRESS_RESERVE_NO_ACCESS,
   };
   if (zero)
      reserve.hPagingQueue = be->paging;
   else
      reserve.hAdapter = be->adapter;
   if (agc_wddm_failed(D3DKMTReserveGpuVirtualAddress(&reserve), function,
                       zero ? "ReserveGpuVirtualAddress(zero)" : "ReserveGpuVirtualAddress"))
      return false;

   if (reserve.VirtualAddress != va) {
      agc_report(function, "ReserveGpuVirtualAddress placed 0x%" PRIx64 " instead of 0x%" PRIx64,
                 (uint64_t)reserve.VirtualAddress, va);
      agc_wddm_free_va(be, reserve.VirtualAddress, size, function,
                       "FreeGpuVirtualAddress(misplaced reservation)");
      return false;
   }

   simple_mtx_lock(&be->lock);
   room = be->reservation_count < be->reservation_capacity;
   if (room)
      be->reservations[be->reservation_count++] = (struct agc_wddm_reservation){va, size};
   simple_mtx_unlock(&be->lock);
   if (!room) {
      agc_report(function, "out of memory");
      agc_wddm_free_va(be, va, size, function, "FreeGpuVirtualAddress(reservation)");
      return false;
   }

   if (reserve.PagingFenceValue != 0)
      return agc_wddm_paging_wait(be, reserve.PagingFenceValue, function);
   return true;
}

static bool
agc_wddm_va_reserve(struct agc_backend *base, uint64_t va, uint64_t size)
{
   return agc_wddm_reserve(agc_wddm_backend(base), va, size, false, __func__);
}

static bool
agc_wddm_va_reserve_zero(struct agc_backend *base, uint64_t va, uint64_t size)
{
   return agc_wddm_reserve(agc_wddm_backend(base), va, size, true, __func__);
}

static bool
agc_wddm_va_release(struct agc_backend *base, uint64_t va, uint64_t size)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   if (!agc_wddm_healthy(be, __func__))
      return false;

   bool found = false;
   simple_mtx_lock(&be->lock);
   for (uint32_t i = 0; i < be->reservation_count && !found; i++) {
      if (be->reservations[i].base == va && be->reservations[i].size == size) {
         be->reservations[i] = be->reservations[--be->reservation_count];
         found = true;
      }
   }
   simple_mtx_unlock(&be->lock);
   if (!found) {
      agc_report(__func__, "no reservation is 0x%" PRIx64 "+0x%" PRIx64, va, size);
      return false;
   }

   return agc_wddm_free_va(be, va, size, __func__, "FreeGpuVirtualAddress(reservation)");
}

static bool
agc_wddm_va_clear(struct agc_backend *base, uint64_t va, uint64_t size, enum agc_va_fill fill)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   if (!agc_wddm_healthy(be, __func__))
      return false;
   if (size == 0 || size % AGC_WDDM_GPU_PAGE_BYTES != 0 ||
       (fill != AGC_VA_NO_ACCESS && fill != AGC_VA_ZERO)) {
      agc_report(__func__, "0x%" PRIx64 "+0x%" PRIx64 " is not whole pages to clear", va, size);
      return false;
   }
   if (!agc_wddm_check_fixed(be, va, size, AGC_WDDM_GPU_PAGE_BYTES, __func__))
      return false;

   D3DDDI_MAPGPUVIRTUALADDRESS map = {
      .hPagingQueue = be->paging,
      .BaseAddress = va,
      .SizeInPages = size / AGC_WDDM_GPU_PAGE_BYTES,
   };
   if (fill == AGC_VA_ZERO)
      map.Protection.Zero = 1;
   else
      map.Protection.NoAccess = 1;
   if (agc_wddm_failed(D3DKMTMapGpuVirtualAddress(&map), __func__,
                       fill == AGC_VA_ZERO ? "MapGpuVirtualAddress(zero)"
                                           : "MapGpuVirtualAddress(no access)"))
      return false;
   if (map.VirtualAddress != va) {
      agc_wddm_poison(be, __func__, "a cleared range was placed at another virtual address");
      return false;
   }
   return agc_wddm_paging_wait(be, map.PagingFenceValue, __func__);
}

static uint32_t
agc_wddm_heap_bits(const uint8_t heaps[2])
{
   uint32_t bits = 0;
   for (unsigned i = 0; i < 2; i++) {
      if (heaps[i])
         bits |= AGC_WDDM_BO_HEAP_BIT(heaps[i]);
   }
   return bits;
}

static struct agc_wddm_alloc_params
agc_wddm_device_params(const struct agc_wddm_profile *profile, uint32_t size, bool gl2_bypass,
                       uint64_t va)
{
   return (struct agc_wddm_alloc_params){
      .size = size,
      .flags = AGC_WDDM_BO_CPU_VISIBLE | AGC_WDDM_BO_NOT_VIRTUAL |
               agc_wddm_heap_bits(profile->device_heaps),
      .alignment = (uint32_t)AGC_WDDM_BUFFER_ALIGNMENT,
      .priority = AGC_WDDM_BO_PRIORITY,
      .heaps = {profile->device_heaps[0], profile->device_heaps[1]},
      .mtype = gl2_bypass ? AGC_WDDM_MTYPE_GL2_BYPASS : AGC_WDDM_MTYPE_GL2_CACHED,
      .mall_policy = AGC_WDDM_MALL_POLICY_NEVER,
      .va_address = va,
   };
}

static struct agc_wddm_alloc_params
agc_wddm_host_params(const struct agc_wddm_profile *profile, uint32_t size, bool gl2_bypass,
                     uint64_t va)
{
   return (struct agc_wddm_alloc_params){
      .size = size,
      .flags = AGC_WDDM_BO_CPU_VISIBLE | AGC_WDDM_BO_HOST_ALLOCATED | AGC_WDDM_BO_NOT_VIRTUAL |
               agc_wddm_heap_bits(profile->host_heaps),
      .alignment = (uint32_t)AGC_WDDM_HOST_ALIGNMENT,
      .priority = AGC_WDDM_BO_PRIORITY | AGC_WDDM_BO_PRIORITY_HOST,
      .heaps = {profile->host_heaps[0], profile->host_heaps[1]},
      .mtype = gl2_bypass ? AGC_WDDM_MTYPE_GL2_BYPASS : AGC_WDDM_MTYPE_GL2_CACHED,
      .mall_policy = AGC_WDDM_MALL_POLICY_NEVER,
      .va_address = va,
   };
}

struct agc_wddm_fixture {
   uint8_t *out;
   size_t capacity;
   size_t bytes;
};

static void
agc_wddm_fixture_add(struct agc_wddm_fixture *fixture, const void *data, size_t bytes)
{
   if (fixture->out && fixture->bytes + bytes <= fixture->capacity)
      memcpy(fixture->out + fixture->bytes, data, bytes);
   fixture->bytes += bytes;
}

static void
agc_wddm_fixture_add_alloc(struct agc_wddm_fixture *fixture, struct agc_wddm_alloc_params params)
{
   struct agc_wddm_alloc_create_data create;
   struct agc_wddm_alloc_records records;
   agc_wddm_fill_alloc(&create, &records, &params);
   agc_wddm_fixture_add(fixture, &create, sizeof(create));
   agc_wddm_fixture_add(fixture, &records, sizeof(records));
}

size_t
agc_wddm_abi_fixture(void *out, size_t capacity)
{
   const struct agc_wddm_profile *profile = &agc_wddm_profiles[0];
   struct agc_wddm_fixture fixture = {.out = out, .capacity = capacity};
   struct agc_wddm_context_data context;
   struct agc_wddm_queue_data queue;
   struct agc_wddm_submit_data submit;
   const struct agc_ib single = {.va = UINT64_C(0x123456789000), .dwords = 64};
   const struct agc_ib several[] = {
      {.va = UINT64_C(0x20000001000), .dwords = 16, .flags = AGC_IB_PREAMBLE},
      {.va = UINT64_C(0x200004100), .dwords = 0xffff8, .flags = AGC_IB_MAIN},
      {.va = UINT64_C(0x7fffffffff00), .dwords = 8, .flags = AGC_IB_MAIN},
   };

   agc_wddm_fill_context(&context, true);
   agc_wddm_fixture_add(&fixture, &context, sizeof(context));
   agc_wddm_fill_queue(&queue);
   agc_wddm_fixture_add(&fixture, &queue, sizeof(queue));
   agc_wddm_fixture_add_alloc(&fixture, agc_wddm_device_params(profile, 0x10000, true, 0));
   agc_wddm_fixture_add(&fixture, &submit, agc_wddm_fill_submit(&submit, &single, 1));

   agc_wddm_fill_context(&context, false);
   agc_wddm_fixture_add(&fixture, &context, sizeof(context));
   agc_wddm_fixture_add_alloc(&fixture, agc_wddm_device_params(profile, 0x30000, false, 0));
   agc_wddm_fixture_add_alloc(
      &fixture, agc_wddm_device_params(profile, 0x20000, false, UINT64_C(0x10203040000)));
   agc_wddm_fixture_add_alloc(
      &fixture, agc_wddm_device_params(profile, 0x40000, true, UINT64_C(0x20000010000)));
   agc_wddm_fixture_add_alloc(
      &fixture, agc_wddm_host_params(profile, 0x8000000, false, UINT64_C(0x200000000)));
   agc_wddm_fixture_add_alloc(
      &fixture, agc_wddm_host_params(profile, 0x10000, true, UINT64_C(0x245673000)));
   agc_wddm_fixture_add(&fixture, &submit,
                        agc_wddm_fill_submit(&submit, several, ARRAY_SIZE(several)));
   return fixture.bytes;
}

static bool
agc_wddm_create_allocation(struct agc_wddm_backend *be, const struct agc_wddm_alloc_params *params,
                           const void *system_memory, struct agc_wddm_chunk *chunk,
                           const char *function)
{
   struct agc_wddm_alloc_create_data create_data;
   struct agc_wddm_alloc_records records;
   agc_wddm_fill_alloc(&create_data, &records, params);

   D3DDDI_ALLOCATIONINFO2 info = {
      .pSystemMem = system_memory,
      .pPrivateDriverData = &records,
      .PrivateDriverDataSize = sizeof(records),
      .VidPnSourceId = UINT32_MAX,
      .Priority = D3DDDI_ALLOCATIONPRIORITY_NORMAL,
   };
   D3DKMT_CREATEALLOCATION request = {
      .hDevice = be->device,
      .pPrivateDriverData = &create_data,
      .PrivateDriverDataSize = sizeof(create_data),
      .NumAllocations = 1,
      .pAllocationInfo2 = &info,
   };
   request.Flags.CreateResource = 1;
   request.Flags.NonSecure = 1;
   if (agc_wddm_failed(D3DKMTCreateAllocation2(&request), function,
                       system_memory ? "CreateAllocation2(host memory)" : "CreateAllocation2"))
      return false;

   chunk->handle = info.hAllocation;
   chunk->resource = request.hResource;
   p_atomic_inc(&be->live_allocations);
   return true;
}

static bool
agc_wddm_make_resident(struct agc_wddm_backend *be, D3DKMT_HANDLE allocation, uint64_t *paging,
                       const char *function)
{
   D3DDDI_MAKERESIDENT resident = {
      .hPagingQueue = be->paging,
      .NumAllocations = 1,
      .AllocationList = &allocation,
   };
   resident.Flags.MustSucceed = 1;
   NTSTATUS status = D3DKMTMakeResident(&resident);
   if (status < 0) {
      agc_report(function, "MakeResident: NTSTATUS=0x%08x, %" PRIu64 " bytes to trim",
                 (unsigned)status, (uint64_t)resident.NumBytesToTrim);
      return false;
   }
   *paging = MAX2(*paging, resident.PagingFenceValue);
   return true;
}

static bool
agc_wddm_map(struct agc_wddm_backend *be, D3DKMT_HANDLE allocation, uint64_t fixed_va,
             uint64_t offset, uint64_t size, bool execute, uint64_t *placed, uint64_t *paging,
             const char *function)
{
   D3DDDI_MAPGPUVIRTUALADDRESS map = {
      .hPagingQueue = be->paging,
      .hAllocation = allocation,
      .OffsetInPages = offset / AGC_WDDM_GPU_PAGE_BYTES,
      .SizeInPages = size / AGC_WDDM_GPU_PAGE_BYTES,
   };
   if (fixed_va) {
      map.BaseAddress = fixed_va;
   } else {
      map.MinimumAddress = AGC_WDDM_DRIVER_VA_BASE;
      map.MaximumAddress = be->va_end;
   }
   map.Protection.Write = 1;
   map.Protection.Execute = execute ? 1 : 0;
   if (agc_wddm_failed(D3DKMTMapGpuVirtualAddress(&map), function, "MapGpuVirtualAddress"))
      return false;
   *placed = map.VirtualAddress;
   *paging = MAX2(*paging, map.PagingFenceValue);
   return true;
}

static struct agc_bo *
agc_wddm_bo_alloc(struct agc_wddm_backend *be, uint32_t chunks, const char *function)
{
   struct agc_bo *bo = calloc(1, sizeof(*bo) + (size_t)chunks * sizeof(bo->chunks[0]));
   if (!bo) {
      agc_report(function, "out of memory");
      return NULL;
   }
   bo->be = be;
   return bo;
}

static void
agc_wddm_bo_release(struct agc_wddm_backend *be, struct agc_bo *bo, const char *function)
{
   if (bo->locked && !p_atomic_read(&be->lost)) {
      D3DKMT_UNLOCK2 unlock = {
         .hDevice = be->device,
         .hAllocation = bo->chunks[0].handle,
      };
      NTSTATUS status = D3DKMTUnlock2(&unlock);
      if (status < 0)
         agc_wddm_poison_status(be, function, "cleanup Unlock2", status);
   }
   if (bo->va && !bo->reserved && !p_atomic_read(&be->lost))
      agc_wddm_free_va(be, bo->va, bo->size, function, "cleanup FreeGpuVirtualAddress");

   for (uint32_t i = 0; i < bo->chunk_count && !p_atomic_read(&be->lost); i++) {
      D3DKMT_DESTROYALLOCATION2 destroy = {
         .hDevice = be->device,
         .hResource = bo->chunks[i].resource,
         .phAllocationList = bo->chunks[i].resource ? NULL : &bo->chunks[i].handle,
         .AllocationCount = bo->chunks[i].resource ? 0 : 1,
      };
      NTSTATUS status = D3DKMTDestroyAllocation2(&destroy);
      if (status < 0)
         agc_wddm_poison_status(be, function, "cleanup DestroyAllocation2", status);
      else
         p_atomic_dec(&be->live_allocations);
   }
   free(bo);
}

static struct agc_bo *
agc_wddm_bo_create(struct agc_backend *base, uint64_t size, uint32_t flags, uint64_t va)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   if (!agc_wddm_healthy(be, __func__))
      return NULL;
   if (flags & ~(AGC_BO_GL2_BYPASS | AGC_BO_EXECUTE)) {
      agc_report(__func__, "unknown buffer flags 0x%x", flags);
      return NULL;
   }
   if (size == 0 || size > UINT32_MAX - (AGC_WDDM_BUFFER_ALIGNMENT - 1)) {
      agc_report(__func__, "buffer size 0x%" PRIx64 " is zero or too large", size);
      return NULL;
   }
   uint64_t bytes = align64(size, AGC_WDDM_BUFFER_ALIGNMENT);
   if (va && !agc_wddm_check_fixed(be, va, bytes, AGC_WDDM_RESERVATION_ALIGNMENT, __func__))
      return NULL;

   struct agc_bo *bo = agc_wddm_bo_alloc(be, 1, __func__);
   if (!bo)
      return NULL;
   bo->size = bytes;
   bo->chunk_bytes = bytes;
   bo->reserved = va != 0;
   bo->execute = (flags & AGC_BO_EXECUTE) != 0;

   const struct agc_wddm_alloc_params params =
      agc_wddm_device_params(be->profile, (uint32_t)bytes, (flags & AGC_BO_GL2_BYPASS) != 0, va);
   uint64_t paging = 0;
   if (!agc_wddm_create_allocation(be, &params, NULL, &bo->chunks[0], __func__))
      goto fail;
   bo->chunk_count = 1;
   if (!agc_wddm_make_resident(be, bo->chunks[0].handle, &paging, __func__) ||
       !agc_wddm_map(be, bo->chunks[0].handle, va, 0, bytes, bo->execute, &bo->va, &paging,
                     __func__))
      goto fail;
   if (va && bo->va != va) {
      agc_report(__func__, "MapGpuVirtualAddress placed 0x%" PRIx64 " instead of 0x%" PRIx64,
                 bo->va, va);
      goto fail;
   }
   if (!agc_wddm_paging_wait(be, paging, __func__))
      goto fail;

   D3DKMT_LOCK2 lock = {
      .hDevice = be->device,
      .hAllocation = bo->chunks[0].handle,
   };
   if (agc_wddm_failed(D3DKMTLock2(&lock), __func__, "Lock2"))
      goto fail;
   if (!lock.pData) {
      agc_report(__func__, "Lock2 returned null");
      goto fail;
   }
   bo->cpu = lock.pData;
   bo->locked = true;
   return bo;

fail:
   agc_wddm_bo_release(be, bo, __func__);
   return NULL;
}

static bool
agc_wddm_check_source(const void *cpu, uint64_t size, const char *function)
{
   uintptr_t start = (uintptr_t)cpu;
   if (size > UINTPTR_MAX - start) {
      agc_report(function, "import source range overflows");
      return false;
   }
   uintptr_t end = start + (uintptr_t)size;
   for (uintptr_t at = start; at < end;) {
      MEMORY_BASIC_INFORMATION info;
      if (!VirtualQuery((const void *)at, &info, sizeof(info))) {
         agc_report(function, "import source %p: VirtualQuery failed at 0x%" PRIxPTR, cpu, at);
         return false;
      }
      if (info.State != MEM_COMMIT || info.Type != MEM_MAPPED ||
          (info.Protect != PAGE_READWRITE && info.Protect != PAGE_EXECUTE_READWRITE)) {
         agc_report(function,
                    "import source %p+0x%" PRIx64 ": memory at %p is not committed read-write "
                    "section memory (state 0x%lx, type 0x%lx, protection 0x%lx)",
                    cpu, size, info.BaseAddress, info.State, info.Type, info.Protect);
         return false;
      }
      at = (uintptr_t)info.BaseAddress + info.RegionSize;
   }
   return true;
}

static struct agc_bo *
agc_wddm_bo_import(struct agc_backend *base, void *cpu, uint64_t size, uint32_t flags, uint64_t va)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   if (!agc_wddm_healthy(be, __func__))
      return NULL;
   if (flags & ~AGC_BO_GL2_BYPASS) {
      agc_report(__func__, "host memory imports take no flags but AGC_BO_GL2_BYPASS (0x%x)", flags);
      return NULL;
   }
   if (!cpu || (uintptr_t)cpu % AGC_WDDM_HOST_ALIGNMENT != 0 || size == 0 ||
       size % AGC_WDDM_HOST_ALIGNMENT != 0) {
      agc_report(__func__, "host import of 0x%" PRIx64 " bytes at %p is not whole pages", size,
                 cpu);
      return NULL;
   }
   if (!agc_wddm_check_fixed(be, va, size, AGC_WDDM_HOST_ALIGNMENT, __func__) ||
       !agc_wddm_check_source(cpu, size, __func__))
      return NULL;

   uint64_t chunks = DIV_ROUND_UP(size, AGC_WDDM_IMPORT_CHUNK_BYTES);
   struct agc_bo *bo = agc_wddm_bo_alloc(be, (uint32_t)chunks, __func__);
   if (!bo)
      return NULL;
   bo->va = va;
   bo->size = size;
   bo->chunk_bytes = AGC_WDDM_IMPORT_CHUNK_BYTES;
   bo->cpu = cpu;
   bo->reserved = true;

   uint64_t paging = 0;
   for (uint64_t offset = 0; offset < size; offset += AGC_WDDM_IMPORT_CHUNK_BYTES) {
      uint64_t bytes = MIN2(size - offset, AGC_WDDM_IMPORT_CHUNK_BYTES);
      struct agc_wddm_chunk *chunk = &bo->chunks[bo->chunk_count];
      const struct agc_wddm_alloc_params params = agc_wddm_host_params(
         be->profile, (uint32_t)bytes, (flags & AGC_BO_GL2_BYPASS) != 0, va + offset);
      if (!agc_wddm_create_allocation(be, &params, (const uint8_t *)cpu + offset, chunk, __func__))
         goto fail;
      bo->chunk_count++;

      uint64_t placed = 0;
      if (!agc_wddm_make_resident(be, chunk->handle, &paging, __func__) ||
          !agc_wddm_map(be, chunk->handle, va + offset, 0, bytes, false, &placed, &paging,
                        __func__))
         goto fail;
      if (placed != va + offset) {
         agc_report(__func__, "MapGpuVirtualAddress placed 0x%" PRIx64 " instead of 0x%" PRIx64,
                    placed, va + offset);
         goto fail;
      }
   }
   agc_wddm_paging_note(be, paging);
   return bo;

fail:
   agc_wddm_paging_note(be, paging);
   agc_wddm_bo_release(be, bo, __func__);
   return NULL;
}

static bool
agc_wddm_bo_alias(struct agc_backend *base, struct agc_bo *bo, uint64_t offset, uint64_t size,
                  uint64_t va)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   if (!agc_wddm_healthy(be, __func__))
      return false;
   if (bo->be != be) {
      agc_report(__func__, "the buffer belongs to another device");
      return false;
   }
   if (size == 0 || offset % AGC_WDDM_GPU_PAGE_BYTES != 0 || size % AGC_WDDM_GPU_PAGE_BYTES != 0 ||
       offset > bo->size || size > bo->size - offset) {
      agc_report(__func__, "alias 0x%" PRIx64 "+0x%" PRIx64 " is not whole pages of a 0x%" PRIx64
                           "-byte buffer",
                 offset, size, bo->size);
      return false;
   }
   if (!agc_wddm_check_fixed(be, va, size, AGC_WDDM_GPU_PAGE_BYTES, __func__))
      return false;

   uint64_t paging = 0;
   bool ok = true;
   for (uint64_t done = 0; done < size && ok;) {
      uint64_t at = offset + done;
      uint64_t within = at % bo->chunk_bytes;
      uint64_t span = MIN2(size - done, bo->chunk_bytes - within);
      uint64_t placed = 0;
      ok = agc_wddm_map(be, bo->chunks[at / bo->chunk_bytes].handle, va + done, within, span,
                        bo->execute, &placed, &paging, __func__);
      if (ok && placed != va + done) {
         agc_wddm_poison(be, __func__, "an alias was placed at another virtual address");
         ok = false;
      }
      done += span;
   }
   agc_wddm_paging_note(be, paging);
   return ok;
}

static void *
agc_wddm_bo_cpu(struct agc_bo *bo)
{
   return bo->cpu;
}

static uint64_t
agc_wddm_bo_va(struct agc_bo *bo)
{
   return bo->va;
}

static uint64_t
agc_wddm_bo_size(struct agc_bo *bo)
{
   return bo->size;
}

static void
agc_wddm_bo_destroy(struct agc_backend *base, struct agc_bo *bo)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   if (!bo)
      return;
   if (bo->be != be) {
      agc_report(__func__, "the buffer belongs to another device");
      return;
   }
   agc_wddm_bo_release(be, bo, __func__);
}

static struct agc_hwq *
agc_wddm_queue_create(struct agc_backend *base, enum agc_queue_kind kind)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   if (!agc_wddm_healthy(be, __func__))
      return NULL;
   if (kind != AGC_QUEUE_GFX && kind != AGC_QUEUE_COMPUTE) {
      agc_report(__func__, "unknown queue kind %d", (int)kind);
      return NULL;
   }
   uint32_t node = kind == AGC_QUEUE_GFX ? be->gfx_node : be->compute_node;
   if (node == AGC_WDDM_NODE_NONE) {
      agc_report(__func__, "the adapter reports no %s node",
                 kind == AGC_QUEUE_GFX ? "graphics" : "compute");
      return NULL;
   }

   struct agc_hwq *queue = calloc(1, sizeof(*queue));
   if (!queue) {
      agc_report(__func__, "out of memory");
      return NULL;
   }
   queue->be = be;
   queue->kind = kind;
   queue->node = node;

   struct agc_wddm_context_data context_data;
   agc_wddm_fill_context(&context_data, kind == AGC_QUEUE_GFX);
   D3DKMT_CREATECONTEXTVIRTUAL context = {
      .hDevice = be->device,
      .NodeOrdinal = node,
      .EngineAffinity = 1,
      .pPrivateDriverData = &context_data,
      .PrivateDriverDataSize = sizeof(context_data),
      .ClientHint = D3DKMT_CLIENTHINT_VULKAN,
   };
   context.Flags.HwQueueSupported = 1;
   if (agc_wddm_failed(D3DKMTCreateContextVirtual(&context), __func__, "CreateContextVirtual")) {
      free(queue);
      return NULL;
   }
   queue->context = context.hContext;

   struct agc_wddm_queue_data queue_data;
   agc_wddm_fill_queue(&queue_data);
   D3DKMT_CREATEHWQUEUE create = {
      .hHwContext = queue->context,
      .PrivateDriverDataSize = sizeof(queue_data),
      .pPrivateDriverData = &queue_data,
   };
   if (agc_wddm_failed(D3DKMTCreateHwQueue(&create), __func__, "CreateHwQueue")) {
      D3DKMT_DESTROYCONTEXT destroy = {.hContext = queue->context};
      D3DKMTDestroyContext(&destroy);
      free(queue);
      return NULL;
   }
   queue->handle = create.hHwQueue;
   queue->id = queue_data.queue_id;
   queue->progress.object = create.hHwQueueProgressFence;
   queue->progress.value = create.HwQueueProgressFenceCPUVirtualAddress;
   queue->progress_gpu_va = create.HwQueueProgressFenceGPUVirtualAddress;
   agc_wddm_fence_current(&queue->progress, &queue->submitted);
   simple_mtx_init(&queue->lock, mtx_plain);
   p_atomic_inc(&be->live_queues);
   return queue;
}

static void
agc_wddm_queue_destroy(struct agc_hwq *queue)
{
   if (!queue)
      return;
   struct agc_wddm_backend *be = queue->be;
   uint64_t submitted = p_atomic_read(&queue->submitted);
   if (!p_atomic_read(&be->lost) &&
       agc_wddm_fence_reach(be, &queue->progress, submitted, AGC_WDDM_FENCE_TIMEOUT_NS) !=
          AGC_WDDM_WAIT_REACHED) {
      char reason[96];
      snprintf(reason, sizeof(reason),
               "queue work up to fence %" PRIu64 " did not complete at close", submitted);
      agc_wddm_poison(be, __func__, reason);
   }

   if (!p_atomic_read(&be->lost)) {
      D3DKMT_DESTROYHWQUEUE destroy_queue = {.hHwQueue = queue->handle};
      D3DKMTDestroyHwQueue(&destroy_queue);
      D3DKMT_DESTROYCONTEXT destroy_context = {.hContext = queue->context};
      D3DKMTDestroyContext(&destroy_context);
   }
   p_atomic_dec(&be->live_queues);
   simple_mtx_destroy(&queue->lock);
   free(queue);
}

void
agc_wddm_queue_describe(struct agc_hwq *queue, struct agc_wddm_queue_description *description)
{
   *description = (struct agc_wddm_queue_description){
      .node = queue->node,
      .queue_id = queue->id,
      .fence_gpu_va = queue->progress_gpu_va,
      .submitted = p_atomic_read(&queue->submitted),
   };
}

static bool
agc_wddm_check_ibs(const struct agc_wddm_backend *be, const struct agc_ib *ibs, uint32_t count,
                   const char *function)
{
   if (!ibs || count == 0 || count > AGC_SUBMIT_MAX_IBS) {
      agc_report(function, "a submission lists 1 to %d IBs, not %u", AGC_SUBMIT_MAX_IBS, count);
      return false;
   }
   uint32_t size_dwords = be->profile->ib_size_alignment / (uint32_t)sizeof(uint32_t);
   for (uint32_t i = 0; i < count; i++) {
      if (ibs[i].dwords == 0 || ibs[i].dwords > AGC_WDDM_MAX_IB_DWORDS ||
          ibs[i].dwords % size_dwords != 0 || ibs[i].va == 0 ||
          ibs[i].va % be->profile->ib_start_alignment != 0 ||
          (ibs[i].flags != AGC_IB_MAIN && ibs[i].flags != AGC_IB_PREAMBLE)) {
         agc_report(function, "IB %u (0x%" PRIx64 ", %u dwords, flags 0x%x) breaks the %u-byte / "
                              "%u-dword / 0x%x-dword rules",
                    i, ibs[i].va, ibs[i].dwords, ibs[i].flags, be->profile->ib_start_alignment,
                    size_dwords, AGC_WDDM_MAX_IB_DWORDS);
         return false;
      }
   }
   return true;
}

static uint64_t
agc_wddm_submit(struct agc_hwq *queue, const struct agc_ib *ibs, uint32_t count)
{
   struct agc_wddm_backend *be = queue->be;
   if (!agc_wddm_healthy(be, __func__) || !agc_wddm_check_ibs(be, ibs, count, __func__))
      return 0;
   if (!agc_wddm_paging_wait(be, p_atomic_read(&be->paging_pending), __func__))
      return 0;

   struct agc_wddm_submit_data data;
   uint32_t data_bytes = agc_wddm_fill_submit(&data, ibs, count);

   simple_mtx_lock(&queue->lock);
   uint64_t fence = queue->submitted + 1;
   if (fence > AGC_SUBMIT_MAX_IN_FLIGHT &&
       agc_wddm_fence_reach(be, &queue->progress, fence - AGC_SUBMIT_MAX_IN_FLIGHT,
                            AGC_WDDM_FENCE_TIMEOUT_NS) != AGC_WDDM_WAIT_REACHED) {
      simple_mtx_unlock(&queue->lock);
      agc_report(__func__, "the queue stayed full: fence %" PRIu64 " did not complete in %" PRId64
                           " ms",
                 fence - AGC_SUBMIT_MAX_IN_FLIGHT, AGC_WDDM_FENCE_TIMEOUT_NS / 1000000);
      return 0;
   }

   D3DKMT_SUBMITCOMMANDTOHWQUEUE submit = {
      .hHwQueue = queue->handle,
      .HwQueueProgressFenceId = fence,
      .CommandBuffer = ibs[0].va,
      .CommandLength = ibs[0].dwords * (uint32_t)sizeof(uint32_t),
      .PrivateDriverDataSize = data_bytes,
      .pPrivateDriverData = &data,
   };
   p_atomic_set(&queue->submitted, fence);
   MemoryBarrier();
   NTSTATUS status = D3DKMTSubmitCommandToHwQueue(&submit);
   if (status < 0)
      agc_wddm_poison_status(be, __func__, "SubmitCommandToHwQueue", status);
   simple_mtx_unlock(&queue->lock);
   return status < 0 ? 0 : fence;
}

static bool
agc_wddm_fence_wait(struct agc_hwq *queue, uint64_t fence, int64_t timeout_ns)
{
   struct agc_wddm_backend *be = queue->be;
   uint64_t submitted = p_atomic_read(&queue->submitted);
   if (fence > submitted) {
      agc_report(__func__, "fence %" PRIu64 " was never submitted on this queue", fence);
      return false;
   }
   if (agc_wddm_fence_reach(be, &queue->progress, fence, timeout_ns) != AGC_WDDM_WAIT_REACHED)
      return false;

   uint64_t value;
   if (agc_wddm_fence_current(&queue->progress, &value) &&
       value > p_atomic_read(&queue->submitted)) {
      char reason[128];
      snprintf(reason, sizeof(reason),
               "the progress fence reads %" PRIu64 ", past the last submission %" PRIu64, value,
               p_atomic_read(&queue->submitted));
      agc_wddm_poison(be, __func__, reason);
      return false;
   }
   MemoryBarrier();
   return true;
}

static uint64_t
agc_wddm_fence_completed(struct agc_hwq *queue)
{
   uint64_t value;
   if (!agc_wddm_fence_current(&queue->progress, &value))
      return 0;
   return MIN2(value, p_atomic_read(&queue->submitted));
}

static bool
agc_wddm_queue_wait(struct agc_hwq *queue, struct agc_hwq *producer, uint64_t fence)
{
   struct agc_wddm_backend *be = queue->be;
   if (!agc_wddm_healthy(be, __func__))
      return false;
   if (producer->be != be || producer == queue) {
      agc_report(__func__, "a queue waits for another queue of the same device");
      return false;
   }
   if (fence > p_atomic_read(&producer->submitted)) {
      agc_report(__func__, "fence %" PRIu64 " was never submitted on the producer queue", fence);
      return false;
   }

   D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE wait = {
      .hHwQueue = queue->handle,
      .ObjectCount = 1,
      .ObjectHandleArray = &producer->progress.object,
      .FenceValueArray = &fence,
   };
   simple_mtx_lock(&queue->lock);
   NTSTATUS status = D3DKMTSubmitWaitForSyncObjectsToHwQueue(&wait);
   simple_mtx_unlock(&queue->lock);
   return !agc_wddm_failed(status, __func__, "SubmitWaitForSyncObjectsToHwQueue(progress fence)");
}

struct agc_wddm_fence *
agc_wddm_fence_create_shared(struct agc_backend *base, uint64_t initial_value, void **nt_handle)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   *nt_handle = NULL;
   if (!agc_wddm_healthy(be, __func__))
      return NULL;
   struct agc_wddm_fence *fence = calloc(1, sizeof(*fence));
   if (!fence) {
      agc_report(__func__, "out of memory");
      return NULL;
   }
   fence->be = be;

   D3DKMT_CREATESYNCHRONIZATIONOBJECT2 create = {
      .hDevice = be->device,
      .Info = {.Type = D3DDDI_MONITORED_FENCE},
   };
   create.Info.Flags.Shared = 1;
   create.Info.Flags.NtSecuritySharing = 1;
   create.Info.MonitoredFence.InitialFenceValue = initial_value;
   create.Info.MonitoredFence.EngineAffinity = 1;
   if (agc_wddm_failed(D3DKMTCreateSynchronizationObject2(&create), __func__,
                       "CreateSynchronizationObject2(shared monitored fence)")) {
      free(fence);
      return NULL;
   }
   fence->fence.object = create.hSyncObject;
   fence->fence.value = create.Info.MonitoredFence.FenceValueCPUVirtualAddress;

   OBJECT_ATTRIBUTES attributes = {.Length = sizeof(attributes)};
   if (agc_wddm_failed(D3DKMTShareObjects(1, &fence->fence.object, &attributes,
                                          D3DDDI_SYNC_OBJECT_ALL_ACCESS, &fence->nt_handle),
                       __func__, "ShareObjects(monitored fence)")) {
      fence->nt_handle = NULL;
      agc_wddm_fence_destroy(fence);
      return NULL;
   }
   *nt_handle = fence->nt_handle;
   return fence;
}

struct agc_wddm_fence *
agc_wddm_fence_open_shared(struct agc_backend *base, void *nt_handle)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   if (!agc_wddm_healthy(be, __func__))
      return NULL;
   if (!nt_handle) {
      agc_report(__func__, "null fence NT handle");
      return NULL;
   }
   struct agc_wddm_fence *fence = calloc(1, sizeof(*fence));
   if (!fence) {
      agc_report(__func__, "out of memory");
      return NULL;
   }
   fence->be = be;

   D3DKMT_OPENSYNCOBJECTFROMNTHANDLE2 open = {
      .hNtHandle = nt_handle,
      .hDevice = be->device,
   };
   open.Flags.Shared = 1;
   open.Flags.NtSecuritySharing = 1;
   open.Flags.NoGPUAccess = 1;
   if (agc_wddm_failed(D3DKMTOpenSyncObjectFromNtHandle2(&open), __func__,
                       "OpenSyncObjectFromNtHandle2")) {
      free(fence);
      return NULL;
   }
   fence->fence.object = open.hSyncObject;
   fence->fence.value = open.MonitoredFence.FenceValueCPUVirtualAddress;
   return fence;
}

void
agc_wddm_fence_destroy(struct agc_wddm_fence *fence)
{
   if (!fence)
      return;
   if (fence->nt_handle)
      CloseHandle(fence->nt_handle);
   if (!p_atomic_read(&fence->be->lost)) {
      D3DKMT_DESTROYSYNCHRONIZATIONOBJECT destroy = {.hSyncObject = fence->fence.object};
      D3DKMTDestroySynchronizationObject(&destroy);
   }
   free(fence);
}

bool
agc_wddm_fence_value(struct agc_wddm_fence *fence, uint64_t *value)
{
   return agc_wddm_fence_current(&fence->fence, value);
}

bool
agc_wddm_queue_wait_fence(struct agc_hwq *queue, struct agc_wddm_fence *fence, uint64_t value)
{
   struct agc_wddm_backend *be = queue->be;
   if (!agc_wddm_healthy(be, __func__))
      return false;
   if (fence->be != be) {
      agc_report(__func__, "the fence belongs to another device");
      return false;
   }

   D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE wait = {
      .hHwQueue = queue->handle,
      .ObjectCount = 1,
      .ObjectHandleArray = &fence->fence.object,
      .FenceValueArray = &value,
   };
   simple_mtx_lock(&queue->lock);
   NTSTATUS status = D3DKMTSubmitWaitForSyncObjectsToHwQueue(&wait);
   simple_mtx_unlock(&queue->lock);
   if (status < 0) {
      agc_wddm_poison_status(be, __func__, "SubmitWaitForSyncObjectsToHwQueue", status);
      return false;
   }
   return true;
}

bool
agc_wddm_queue_signal_fence(struct agc_hwq *queue, struct agc_wddm_fence *fence, uint64_t value)
{
   struct agc_wddm_backend *be = queue->be;
   if (!agc_wddm_healthy(be, __func__))
      return false;
   if (fence->be != be) {
      agc_report(__func__, "the fence belongs to another device");
      return false;
   }

   D3DKMT_SUBMITSIGNALSYNCOBJECTSTOHWQUEUE signal = {
      .BroadcastHwQueueCount = 1,
      .BroadcastHwQueueArray = &queue->handle,
      .ObjectCount = 1,
      .ObjectHandleArray = &fence->fence.object,
      .FenceValueArray = &value,
   };
   simple_mtx_lock(&queue->lock);
   NTSTATUS status = D3DKMTSubmitSignalSyncObjectsToHwQueue(&signal);
   simple_mtx_unlock(&queue->lock);
   if (status < 0) {
      agc_wddm_poison_status(be, __func__, "SubmitSignalSyncObjectsToHwQueue", status);
      return false;
   }
   return true;
}

static bool
agc_wddm_shared_records(const uint8_t *data, size_t bytes, uint64_t *physical_size)
{
   for (size_t offset = 0; offset + sizeof(struct agc_wddm_alloc_header) <= bytes;
        offset += sizeof(uint32_t)) {
      struct agc_wddm_alloc_header header;
      struct agc_wddm_alloc_entry entry;
      memcpy(&header, data + offset, sizeof(header));
      size_t start = offset + sizeof(header);
      size_t entry_size = header.entry_size[0];
      if (header.entry_count == 0 || header.entry_count > ARRAY_SIZE(header.entry_size) ||
          entry_size < sizeof(entry) || entry_size > bytes - start)
         continue;
      memcpy(&entry, data + start, sizeof(entry));
      if (entry.entry_size != entry_size ||
          (entry.version != AGC_WDDM_ABI_VERSION && entry.version != AGC_WDDM_ABI_VERSION_D3D12) ||
          entry.bo_info.section_size != sizeof(entry.bo_info))
         continue;
      *physical_size = entry.bo_info.physical_size;
      return true;
   }
   return false;
}

struct agc_bo *
agc_wddm_bo_open_shared(struct agc_backend *base, void *nt_handle, uint64_t size)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   if (!agc_wddm_healthy(be, __func__))
      return NULL;
   if (!nt_handle) {
      agc_report(__func__, "null resource NT handle");
      return NULL;
   }

   D3DKMT_QUERYRESOURCEINFOFROMNTHANDLE query = {
      .hDevice = be->device,
      .hNtHandle = nt_handle,
   };
   if (agc_wddm_failed(D3DKMTQueryResourceInfoFromNtHandle(&query), __func__,
                       "QueryResourceInfoFromNtHandle"))
      return NULL;
   if (query.NumAllocations != 1) {
      agc_report(__func__, "shared resource has %u allocations; a buffer has one",
                 query.NumAllocations);
      return NULL;
   }

   size_t total_bytes = query.TotalPrivateDriverDataSize;
   size_t resource_bytes = query.ResourcePrivateDriverDataSize;
   size_t runtime_bytes = query.PrivateRuntimeDataSize;
   uint8_t *total = calloc(1, total_bytes + resource_bytes + runtime_bytes + 1);
   struct agc_bo *bo = agc_wddm_bo_alloc(be, 1, __func__);
   if (!total || !bo) {
      free(total);
      free(bo);
      return NULL;
   }

   D3DDDI_OPENALLOCATIONINFO2 info = {0};
   D3DKMT_OPENRESOURCEFROMNTHANDLE open = {
      .hDevice = be->device,
      .hNtHandle = nt_handle,
      .NumAllocations = 1,
      .pOpenAllocationInfo2 = &info,
      .PrivateRuntimeDataSize = (UINT)runtime_bytes,
      .pPrivateRuntimeData = total + total_bytes + resource_bytes,
      .ResourcePrivateDriverDataSize = (UINT)resource_bytes,
      .pResourcePrivateDriverData = total + total_bytes,
      .TotalPrivateDriverDataBufferSize = (UINT)total_bytes,
      .pTotalPrivateDriverDataBuffer = total,
   };
   if (agc_wddm_failed(D3DKMTOpenResourceFromNtHandle(&open), __func__,
                       "OpenResourceFromNtHandle")) {
      free(total);
      free(bo);
      return NULL;
   }
   bo->chunks[0].handle = info.hAllocation;
   bo->chunks[0].resource = open.hResource;
   bo->chunk_count = 1;
   p_atomic_inc(&be->live_allocations);

   const uint8_t *records = total;
   size_t records_bytes = total_bytes;
   uintptr_t records_offset = (uintptr_t)info.pPrivateDriverData - (uintptr_t)total;
   if (info.pPrivateDriverData && info.PrivateDriverDataSize != 0 &&
       records_offset <= total_bytes &&
       info.PrivateDriverDataSize <= total_bytes - records_offset) {
      records = total + records_offset;
      records_bytes = info.PrivateDriverDataSize;
   }
   uint64_t physical_size = 0;
   bool described = agc_wddm_shared_records(records, records_bytes, &physical_size);
   free(total);

   uint64_t bytes;
   if (size != 0 && described && size > physical_size) {
      agc_report(__func__, "the shared allocation's AMD records give 0x%" PRIx64
                           " bytes, less than the 0x%" PRIx64 " its creator gave",
                 physical_size, size);
      goto fail;
   } else if (size != 0) {
      bytes = size > UINT32_MAX ? UINT64_MAX : align64(size, AGC_WDDM_GPU_PAGE_BYTES);
   } else if (described) {
      bytes = physical_size;
   } else {
      agc_report(__func__, "the shared resource's size is unknown: no v8 or v9 AMD allocation "
                           "record in 0x%zx bytes of private data; pass the size its creator "
                           "allocated",
                 records_bytes);
      goto fail;
   }
   if (bytes == 0 || bytes % AGC_WDDM_GPU_PAGE_BYTES != 0 || bytes > UINT32_MAX) {
      agc_report(__func__, "shared allocation size 0x%" PRIx64 " is not whole 4 KiB pages below "
                           "4 GiB",
                 bytes);
      goto fail;
   }
   bo->size = bytes;
   bo->chunk_bytes = bytes;

   uint64_t paging = 0;
   if (!agc_wddm_make_resident(be, bo->chunks[0].handle, &paging, __func__) ||
       !agc_wddm_map(be, bo->chunks[0].handle, 0, 0, bytes, false, &bo->va, &paging, __func__) ||
       !agc_wddm_paging_wait(be, paging, __func__))
      goto fail;
   return bo;

fail:
   agc_wddm_bo_release(be, bo, __func__);
   return NULL;
}

static enum agc_device_state
agc_wddm_device_state(struct agc_backend *base, uint64_t *fault_va)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   struct agc_wddm_device_state state;
   agc_wddm_device_state_detail(base, &state);
   if (fault_va)
      *fault_va = state.page_fault_status >= 0 ? state.fault_va : 0;
   if (state.execution_status < 0)
      return AGC_DEVICE_UNKNOWN;
   if (state.execution == D3DKMT_DEVICEEXECUTION_ACTIVE)
      return p_atomic_read(&be->lost) ? AGC_DEVICE_UNKNOWN : AGC_DEVICE_ACTIVE;

   agc_wddm_poison(be, __func__, "the device left the active execution state");
   switch (state.execution) {
   case D3DKMT_DEVICEEXECUTION_HUNG:
      return AGC_DEVICE_HUNG;
   case D3DKMT_DEVICEEXECUTION_ERROR_DMAPAGEFAULT:
      return AGC_DEVICE_PAGE_FAULT;
   case D3DKMT_DEVICEEXECUTION_RESET:
   case D3DKMT_DEVICEEXECUTION_STOPPED:
      return AGC_DEVICE_REMOVED;
   default:
      return AGC_DEVICE_UNKNOWN;
   }
}

static const char agc_wddm_chip_function[] = "agc_wddm_query_info";

static bool
agc_wddm_chip_expect(const char *what, uint64_t queried, uint64_t expected)
{
   if (queried == expected)
      return true;
   agc_report(agc_wddm_chip_function,
              "KMD chip query: %s is 0x%" PRIx64 ", the profile has 0x%" PRIx64, what, queried,
              expected);
   return false;
}

static bool
agc_wddm_chip_within(const char *what, uint64_t queried, uint64_t low, uint64_t high)
{
   if (queried >= low && queried <= high)
      return true;
   agc_report(agc_wddm_chip_function,
              "KMD chip query: %s is %" PRIu64 ", outside %" PRIu64 "..%" PRIu64, what, queried,
              low, high);
   return false;
}

static bool
agc_wddm_chip_query(struct agc_wddm_backend *be, struct agc_wddm_umd_basic *basic,
                    struct agc_wddm_umd_gpu *gpu)
{
   const struct agc_wddm_profile *profile = be->profile;
   memset(basic, 0, sizeof(*basic));
   memset(gpu, 0, sizeof(*gpu));
   be->chip_basic_status = agc_wddm_query_adapter(be->adapter, KMTQAITYPE_UMDRIVERPRIVATE, basic,
                                                  AGC_WDDM_UMD_BASIC_QUERY_BYTES);
   be->chip_gpu_status = agc_wddm_query_adapter(be->adapter, KMTQAITYPE_UMDRIVERPRIVATE, gpu,
                                                AGC_WDDM_UMD_GPU_QUERY_BYTES);
   if (be->chip_basic_status < 0 || be->chip_gpu_status < 0) {
      agc_report(agc_wddm_chip_function,
                 "KMD chip query failed: basic NTSTATUS=0x%08x, gpu NTSTATUS=0x%08x",
                 (unsigned)be->chip_basic_status, (unsigned)be->chip_gpu_status);
      return false;
   }

   bool ok = true;
   ok &= agc_wddm_chip_expect("basic device_id", basic->device_id, be->identity.device_id);
   ok &= agc_wddm_chip_expect("basic family_id", basic->family_id, profile->family_id);
   ok &= agc_wddm_chip_expect("basic num_rb_pipes", basic->num_rb_pipes, profile->num_rb_pipes);
   ok &= agc_wddm_chip_expect("gpu device_id", gpu->device_id, be->identity.device_id);
   ok &= agc_wddm_chip_expect("gpu family_id", gpu->family_id, profile->family_id);
   ok &= agc_wddm_chip_expect("gpu gb_addr_cfg", gpu->gb_addr_cfg, profile->gb_addr_config);
   ok &= agc_wddm_chip_expect("gpu num_shader_engines", gpu->num_shader_engines,
                              profile->num_shader_engines);
   ok &= agc_wddm_chip_expect("gpu va_end", gpu->va_end, profile->va_end);
   ok &= agc_wddm_chip_within("gpu num_shader_engines", gpu->num_shader_engines, 1,
                              ARRAY_SIZE(gpu->wgp_bitmap));
   ok &= agc_wddm_chip_within("gpu num_shader_arrays_per_engine",
                              gpu->num_shader_arrays_per_engine, 1, AMD_MAX_SA_PER_SE);
   ok &= agc_wddm_chip_within("gpu num_cu_per_wgp", gpu->num_cu_per_wgp, 1, 8);
   ok &= agc_wddm_chip_within("gpu large_page", gpu->large_page & 0x3f, 0, 18);
   ok &= agc_wddm_chip_within("gpu IB start alignment", gpu->ib_alignments[0].start, 1, 4096);
   ok &= agc_wddm_chip_within("gpu IB size alignment", gpu->ib_alignments[0].size, 1, 4096);
   if (!ok)
      return false;

   for (uint32_t se = 0; se < gpu->num_shader_engines; se++) {
      for (uint32_t sa = 0; sa < gpu->num_shader_arrays_per_engine; sa++) {
         uint32_t units = util_bitcount(gpu->wgp_bitmap[se][sa]) * gpu->num_cu_per_wgp;
         uint32_t last = util_last_bit(gpu->wgp_bitmap[se][sa]) * gpu->num_cu_per_wgp;
         if (units != profile->num_cu_per_shader_array || last > 16) {
            agc_report(agc_wddm_chip_function,
                       "KMD chip query: shader array %u.%u has %u compute units up to bit %u, the "
                       "profile has %u",
                       se, sa, units, last, profile->num_cu_per_shader_array);
            return false;
         }
      }
   }
   return true;
}

static void
agc_wddm_chip_apply(const struct agc_wddm_umd_basic *basic, const struct agc_wddm_umd_gpu *gpu,
                    struct radeon_info *info, struct drm_amdgpu_info_device *dev,
                    struct drm_amdgpu_memory_info *mem, struct drm_amdgpu_info_hw_ip *gfx_ip,
                    struct drm_amdgpu_info_hw_ip *compute_ip)
{
   info->me_fw_version = gpu->me_fw_version;
   info->mec_fw_version = gpu->mec_fw_version;
   info->pfp_fw_version = gpu->pfp_fw_version;

   dev->chip_rev = gpu->chip_rev;
   dev->external_rev = gpu->chip_external_rev;
   dev->family = gpu->family_id;

   dev->num_shader_engines = gpu->num_shader_engines;
   dev->num_shader_arrays_per_engine = gpu->num_shader_arrays_per_engine;
   dev->gpu_counter_freq = gpu->clock_crystal_freq_hz / 1000;
   dev->wave_front_size = gpu->wave_front_size;
   dev->num_shader_visible_vgprs = gpu->num_shader_visible_vgprs;
   dev->num_tcc_blocks = gpu->num_tcc_blocks;
   dev->num_rb_pipes = basic->num_rb_pipes;
   dev->enabled_rb_pipes_mask = basic->enabled_rb_pipes_mask;
   dev->enabled_rb_pipes_mask_hi = basic->enabled_rb_pipes_mask_hi;

   dev->max_gs_waves_per_vgt = gpu->max_gs_waves_per_vgt;
   dev->gs_vgt_table_depth = gpu->gs_vgt_table_depth;
   dev->gs_prim_buffer_depth = gpu->gs_prim_buffer_depth;

   dev->num_sqc_per_wgp = gpu->num_sqc_per_wgp;
   dev->sqc_inst_cache_size = gpu->sqc_inst_cache_size;
   dev->sqc_data_cache_size = gpu->sqc_data_cache_size;
   dev->gl1c_cache_size = gpu->gl1c_cache_size;
   dev->gl2c_cache_size = gpu->gl2c_cache_size;
   dev->mall_size = (uint64_t)gpu->mall_size_mb * 1024 * 1024;

   dev->pte_fragment_size =
      (uint32_t)((UINT64_C(1) << (gpu->large_page & 0x3f)) * AGC_WDDM_GPU_PAGE_BYTES);
   dev->vram_type = basic->vram_type == 6 ? AMDGPU_VRAM_TYPE_GDDR6 : AMDGPU_VRAM_TYPE_UNKNOWN;
   dev->vram_bit_width = basic->vram_bit_width;
   dev->virtual_address_offset = gpu->va_start;
   dev->virtual_address_max = gpu->va_end;
   mem->vram.total_heap_size = gpu->vram_vis_size + gpu->vram_inv_size;
   mem->cpu_accessible_vram.total_heap_size = gpu->vram_vis_size;

   memset(dev->cu_bitmap, 0, sizeof(dev->cu_bitmap));
   memset(dev->cu_ao_bitmap, 0, sizeof(dev->cu_ao_bitmap));
   uint32_t width = gpu->num_shader_arrays_per_engine;
   for (uint32_t se = 0; se < gpu->num_shader_engines; se++) {
      for (uint32_t sa = 0; sa < width; sa++) {
         dev->cu_bitmap[se % 4][se / 4 * width + sa] =
            util_widen_mask(gpu->wgp_bitmap[se][sa], gpu->num_cu_per_wgp);
         dev->cu_ao_bitmap[se % 4][se / 4 * width + sa] =
            util_widen_mask(gpu->wgp_ao_bitmap[se][sa], gpu->num_cu_per_wgp);
      }
   }

   gfx_ip->ib_start_alignment = gpu->ib_alignments[0].start;
   gfx_ip->ib_size_alignment = gpu->ib_alignments[0].size;
   compute_ip->ib_start_alignment = gpu->ib_alignments[1].start;
   compute_ip->ib_size_alignment = gpu->ib_alignments[1].size;
}

static void
agc_wddm_fill_rings(struct radeon_info *info)
{
   info->scratch_wavesize_granularity_shift = info->gfx_level >= GFX11 ? 8 : 10;
   info->scratch_wavesize_granularity = BITFIELD_BIT(info->scratch_wavesize_granularity_shift);
   info->max_scratch_waves =
      MAX2(32u * info->max_good_cu_per_sa * info->max_sa_per_se * info->num_se, 32u);
   info->has_scratch_base_registers = info->gfx_level >= GFX11;
   info->instr_prefetch_distance = info->gfx_level >= GFX10 ? 3 : 0;
   info->se_tile_repeat = 32 * info->max_se;

   for (unsigned ip = 0; ip < AMD_NUM_IP_TYPES; ip++)
      info->max_submitted_ibs[ip] = info->ip[ip].num_queues ? AGC_SUBMIT_MAX_IBS : 0;

   if (info->gfx_level < GFX11)
      return;
   if (info->gfx_level >= GFX12 || info->l3_cache_size_mb)
      info->attribute_ring_size_per_se = 1400 * 1024;
   else if (info->l2_cache_size >= 2 * 1024 * 1024)
      info->attribute_ring_size_per_se = 768 * 1024;
   else
      info->attribute_ring_size_per_se = info->l2_cache_size / 2;
   info->attribute_ring_size_per_se =
      align(info->attribute_ring_size_per_se, AGC_WDDM_ATTRIBUTE_RING_ALIGNMENT);

   unsigned prim_exports = info->gfx_level >= GFX12 ? 16368 : 0;
   unsigned pos_exports = info->gfx_level >= GFX12 ? 16384 : 0;
   info->pos_ring_size_per_se = align(pos_exports * 16, 32);
   info->prim_ring_size_per_se = align(prim_exports * 4, 32);
   uint32_t se_squared = info->max_se * info->max_se;
   info->pos_ring_offset = info->attribute_ring_size_per_se * info->max_se;
   info->prim_ring_offset = info->pos_ring_offset +
                            align(info->pos_ring_size_per_se * se_squared,
                                  AGC_WDDM_ATTRIBUTE_RING_ALIGNMENT);
   info->total_attribute_pos_prim_ring_size =
      info->prim_ring_offset +
      align(info->prim_ring_size_per_se * se_squared, AGC_WDDM_ATTRIBUTE_RING_ALIGNMENT);
}

static bool
agc_wddm_fill_info(struct agc_wddm_backend *be, struct radeon_info *info,
                   const struct agc_wddm_umd_basic *basic, const struct agc_wddm_umd_gpu *gpu)
{
   const struct agc_wddm_profile *profile = be->profile;
   memset(info, 0, sizeof(*info));

   struct drm_amdgpu_info_device dev = {
      .device_id = be->identity.device_id,
      .pci_rev = be->identity.revision,
      .family = profile->family_id,
      .external_rev = profile->external_rev,
      .num_shader_engines = profile->num_shader_engines,
      .num_shader_arrays_per_engine = profile->num_shader_arrays_per_engine,
      .num_cu_per_sh = profile->num_cu_per_shader_array,
      .num_rb_pipes = profile->num_rb_pipes,
      .enabled_rb_pipes_mask = BITFIELD_MASK(profile->num_rb_pipes),
      .ids_flags = profile->integrated ? AMDGPU_IDS_FLAGS_FUSION : 0,
      .virtual_address_max = be->va_end,
      .virtual_address_alignment = (uint32_t)AGC_WDDM_GPU_PAGE_BYTES,
      .pte_fragment_size = (uint32_t)AGC_WDDM_GPU_PAGE_BYTES,
      .gart_page_size = (uint32_t)AGC_WDDM_GPU_PAGE_BYTES,
      .mall_size = (uint64_t)profile->mall_size_mb * 1024 * 1024,
   };
   for (uint32_t se = 0; se < MIN2(profile->num_shader_engines, 4); se++) {
      for (uint32_t sa = 0; sa < MIN2(profile->num_shader_arrays_per_engine, 4); sa++)
         dev.cu_bitmap[se][sa] = BITFIELD_MASK(profile->num_cu_per_shader_array);
   }
   struct drm_amdgpu_info_hw_ip gfx_ip = {
      .hw_ip_version_major = profile->gfx_ip_version >> 16,
      .hw_ip_version_minor = (profile->gfx_ip_version >> 8) & 0xff,
      .ib_start_alignment = profile->ib_start_alignment,
      .ib_size_alignment = profile->ib_size_alignment,
      .available_rings = 0x1,
      .ip_discovery_version = profile->gfx_ip_version,
   };
   struct drm_amdgpu_info_hw_ip compute_ip = gfx_ip;
   compute_ip.available_rings = 0xf;
   struct drm_amdgpu_memory_info mem = {0};
   uint32_t gb_addr_config = profile->gb_addr_config;

   D3DKMT_SEGMENTSIZEINFO segment = {0};
   if (agc_wddm_query_adapter(be->adapter, KMTQAITYPE_GETSEGMENTSIZE, &segment, sizeof(segment)) >=
       0) {
      mem.gtt.total_heap_size = segment.SharedSystemMemorySize;
      mem.vram.total_heap_size = segment.DedicatedVideoMemorySize;
      mem.cpu_accessible_vram.total_heap_size = segment.DedicatedVideoMemorySize;
   }

   if (gpu) {
      agc_wddm_chip_apply(basic, gpu, info, &dev, &mem, &gfx_ip, &compute_ip);
      gb_addr_config = gpu->gb_addr_cfg;
   }

   if (!ac_fill_hw_ip_info(info, &dev, AMD_IP_GFX, &gfx_ip) ||
       !ac_fill_hw_ip_info(info, &dev, AMD_IP_COMPUTE, &compute_ip)) {
      agc_report(agc_wddm_chip_function, "the graphics and compute engines are not usable");
      return false;
   }
   info->ip[AMD_IP_GFX].num_instances = 1;
   info->ip[AMD_IP_COMPUTE].num_instances = 1;
   if (!ac_identify_chip(info, &dev) || info->family != profile->chip ||
       info->gfx_level != profile->gfx_level) {
      agc_report(agc_wddm_chip_function,
                 "family 0x%x revision 0x%x is %s at gfx level %d, the profile has %s at %d",
                 dev.family, dev.external_rev, ac_get_family_name(info->family),
                 (int)info->gfx_level, ac_get_family_name(profile->chip), (int)profile->gfx_level);
      return false;
   }

   D3DKMT_ADAPTERREGISTRYINFO registry = {0};
   if (agc_wddm_query_adapter(be->adapter, KMTQAITYPE_ADAPTERREGISTRYINFO_RENDER, &registry,
                              sizeof(registry)) < 0 ||
       !WideCharToMultiByte(CP_UTF8, 0, registry.AdapterString, -1, info->marketing_name,
                            sizeof(info->marketing_name), NULL, NULL)) {
      snprintf(info->marketing_name, sizeof(info->marketing_name), "AMD %04x:%04x",
               be->identity.vendor_id, be->identity.device_id);
   }
   info->marketing_name[sizeof(info->marketing_name) - 1] = '\0';

   D3DKMT_ADAPTERADDRESS address = {0};
   if (agc_wddm_query_adapter(be->adapter, KMTQAITYPE_ADAPTERADDRESS, &address, sizeof(address)) >=
       0) {
      info->pci.bus = address.BusNumber;
      info->pci.dev = address.DeviceNumber;
      info->pci.func = address.FunctionNumber;
      info->pci.valid = true;
   }

   ac_fill_memory_info(info, &dev, &mem);
   ac_fill_hw_info(info, &dev);
   info->gb_addr_config = gb_addr_config;
   info->num_tile_pipes = 1u << G_0098F8_NUM_PIPES(gb_addr_config);
   ac_fill_feature_info(info, &dev);
   ac_fill_bug_info(info);
   ac_fill_tess_info(info);
   ac_fill_compiler_info(info, &dev, false);

   info->spi_cu_en = ~0u;
   info->has_gpuvm_fault_query = true;
   info->has_vm_always_valid = true;
   info->has_timeline_syncobj = true;
   agc_wddm_fill_rings(info);
   return true;
}

static bool
agc_wddm_query_info(struct agc_backend *base, struct radeon_info *info)
{
   struct agc_wddm_backend *be = agc_wddm_backend(base);
   struct agc_wddm_umd_basic *basic = malloc(sizeof(*basic));
   struct agc_wddm_umd_gpu *gpu = malloc(sizeof(*gpu));

   bool filled = basic && gpu && agc_wddm_chip_query(be, basic, gpu) &&
                 agc_wddm_fill_info(be, info, basic, gpu);
   be->chip_source = filled ? AGC_WDDM_CHIP_KMD_QUERY : AGC_WDDM_CHIP_PROFILE_ONLY;
   if (!filled) {
      agc_report(__func__, "chip information comes from the profile table %s only",
                 be->profile->name);
      filled = agc_wddm_fill_info(be, info, NULL, NULL);
   }
   free(basic);
   free(gpu);
   return filled;
}

static const struct agc_driver_ops agc_wddm_ops = {
   .destroy = agc_wddm_destroy,
   .query_info = agc_wddm_query_info,
   .va_reserve = agc_wddm_va_reserve,
   .va_reserve_zero = agc_wddm_va_reserve_zero,
   .va_clear = agc_wddm_va_clear,
   .va_release = agc_wddm_va_release,
   .bo_create = agc_wddm_bo_create,
   .bo_import = agc_wddm_bo_import,
   .bo_alias = agc_wddm_bo_alias,
   .bo_cpu = agc_wddm_bo_cpu,
   .bo_va = agc_wddm_bo_va,
   .bo_size = agc_wddm_bo_size,
   .bo_destroy = agc_wddm_bo_destroy,
   .paging_settle = agc_wddm_paging_settle,
   .queue_create = agc_wddm_queue_create,
   .queue_destroy = agc_wddm_queue_destroy,
   .submit = agc_wddm_submit,
   .fence_wait = agc_wddm_fence_wait,
   .fence_completed = agc_wddm_fence_completed,
   .queue_wait = agc_wddm_queue_wait,
   .device_state = agc_wddm_device_state,
};
