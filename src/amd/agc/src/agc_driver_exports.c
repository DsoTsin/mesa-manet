#include <string.h>

#include "util/u_atomic.h"

#include "sid.h"

#include "agc_cb_builders.h"
#include "agc_debug.h"
#include "agc_driver.h"
#include "agc_exports.h"
#include "agc_record.h"
#include "agc_videoout.h"

#define FEATURE_UNAVAILABLE ((int32_t)0x8a6d0001u)
#define NULL_OUTPUT ((int32_t)0x8a6d0003u)
#define WORKLOAD_STREAM_INVALID_ID ((int32_t)0x8a6c0033u)
#define WORKLOAD_STREAM_NULL ((int32_t)0x8a6c0035u)
#define WORKLOAD_STREAM_NOT_REGISTERED ((int32_t)0x8a6c003au)
#define EINVAL_KERNEL ((int32_t)0x80020065u)
#define EINVAL_POSIX 22
#define MULTI_SUBMIT_MAX (UINT32_MAX / 8)
#define TOOL_MEMORY_RANGE_BYTES 12
#define FLIP_PACKET_DWORDS 6u
#define WAIT_RENDERING_PACKET_DWORDS 4u
#define GRAPHICS_EVENT_FILTER (-14)

struct agc_packet {
   const uint32_t *address;
   uint32_t dwords;
   uint8_t flags;
   uint8_t reserved[3];
};

struct agc_kernel_event {
   uintptr_t ident;
   int16_t filter;
   uint16_t flags;
   uint32_t fflags;
   intptr_t data;
   void *udata;
};

_Static_assert(sizeof(struct agc_packet) == 16, "agc_packet");
_Static_assert(sizeof(struct agc_kernel_event) == 32, "agc_kernel_event");

static uint32_t workload_streams;
static uint64_t tf_ring_base;
static uint32_t tf_ring_bytes;
static uint32_t hs_offchip_param;

static int32_t
submit_packet(uint32_t queue, const struct agc_packet *packet, const char *function)
{
   if (!packet) {
      agc_report(function, "null packet");
      return AGC_DRIVER_ERROR_INVALID;
   }
   if (packet->flags) {
      agc_report(function, "submission flags 0x%x are not implemented", packet->flags);
      return AGC_DRIVER_ERROR_INVALID;
   }
   return agc_driver_submit(queue, packet->address, packet->dwords, function);
}

static int32_t
submit_many(uint32_t queue, const uint32_t *const *addresses, const uint32_t *sizes, uint32_t count,
            const char *function)
{
   if (count == 0)
      return 0;
   if (count > MULTI_SUBMIT_MAX || !addresses || !sizes) {
      agc_report(function, "invalid multi-submit arrays");
      return AGC_DRIVER_ERROR_INVALID;
   }
   for (uint32_t i = 0; i < count; i++) {
      if (sizes[i] && !addresses[i]) {
         agc_report(function, "null command buffer %u of a multi-submit", i);
         return AGC_DRIVER_ERROR_INVALID;
      }
      int32_t result = agc_driver_submit(queue, addresses[i], sizes[i], function);
      if (result)
         return result;
   }
   return 0;
}

static bool
compute_queue(uint32_t queue, uint32_t end, const char *function)
{
   if (queue >= AGC_QUEUE_COMPUTE_FIRST && queue < end)
      return true;
   agc_report(function, "unsupported compute queue 0x%x", queue);
   return false;
}

static bool
context_queue(const void *context, uint32_t *queue, const char *function)
{
   uint32_t id;
   memcpy(&id, (const uint8_t *)context + 4, sizeof(id));
   if (id == AGC_QUEUE_GRAPHICS || compute_queue(id, AGC_QUEUE_COMPUTE_MULTI_END, function)) {
      *queue = id;
      return true;
   }
   return false;
}

int32_t AGC_SYSV
sceAgcDriverSubmitDcb(const struct agc_packet *packet)
{
   return submit_packet(AGC_QUEUE_GRAPHICS, packet, "sceAgcDriverSubmitDcb");
}

int32_t AGC_SYSV
sceAgcDriverAgrSubmitDcb(const struct agc_packet *packet)
{
   return submit_packet(AGC_QUEUE_GRAPHICS, packet, "sceAgcDriverAgrSubmitDcb");
}

int32_t AGC_SYSV
sceAgcDriverSubmitMultiDcbs(const uint32_t *const *addresses, const uint32_t *sizes, uint32_t count)
{
   return submit_many(AGC_QUEUE_GRAPHICS, addresses, sizes, count, "sceAgcDriverSubmitMultiDcbs");
}

int32_t AGC_SYSV
sceAgcDriverAgrSubmitMultiDcbs(const uint32_t *const *addresses, const uint32_t *sizes, uint32_t count)
{
   return submit_many(AGC_QUEUE_GRAPHICS, addresses, sizes, count, "sceAgcDriverAgrSubmitMultiDcbs");
}

int32_t AGC_SYSV
sceAgcDriverSubmitAcb(uint32_t queue, const struct agc_packet *packet)
{
   static const char function[] = "sceAgcDriverSubmitAcb";
   if (!compute_queue(queue, AGC_QUEUE_COMPUTE_END, function))
      return AGC_DRIVER_ERROR_INVALID;
   return submit_packet(queue, packet, function);
}

int32_t AGC_SYSV
sceAgcDriverSubmitMultiAcbs(uint32_t queue, const uint32_t *const *addresses, const uint32_t *sizes,
                            uint32_t count)
{
   static const char function[] = "sceAgcDriverSubmitMultiAcbs";
   if (count == 0)
      return 0;
   if (!compute_queue(queue, AGC_QUEUE_COMPUTE_MULTI_END, function))
      return AGC_DRIVER_ERROR_INVALID;
   return submit_many(queue, addresses, sizes, count, function);
}

int32_t AGC_SYSV
sceAgcDriverSubmitCommandBuffer(const void *context, const struct agc_packet *packet)
{
   static const char function[] = "sceAgcDriverSubmitCommandBuffer";
   uint32_t queue;
   if (!context || !packet)
      return EINVAL_POSIX;
   if (!context_queue(context, &queue, function))
      return AGC_DRIVER_ERROR_INVALID;
   return submit_packet(queue, packet, function);
}

int32_t AGC_SYSV
sceAgcDriverSubmitMultiCommandBuffers(const void *context, const uint32_t *const *addresses, const uint32_t *sizes,
                                      uint32_t count)
{
   static const char function[] = "sceAgcDriverSubmitMultiCommandBuffers";
   uint32_t queue;
   if (count == 0)
      return 0;
   if (!context || !addresses || !sizes)
      return EINVAL_POSIX;
   if (!context_queue(context, &queue, function))
      return AGC_DRIVER_ERROR_INVALID;
   return submit_many(queue, addresses, sizes, count, function);
}

int32_t AGC_SYSV
sceAgcDriverAddEqEvent(int64_t queue, int32_t id, void *udata)
{
   static const char function[] = "sceAgcDriverAddEqEvent";
   struct agc_driver_error error = {0};
   if (id < 0) {
      agc_report(function, "negative event id");
      return AGC_DRIVER_ERROR_INVALID;
   }
   if (agc_events_add(queue, (uint32_t)id, (uint64_t)id, udata, &error)) {
      agc_report(function, "%s", error.message);
      return AGC_DRIVER_ERROR_INVALID;
   }
   return 0;
}

int32_t AGC_SYSV
sceAgcDriverDeleteEqEvent(int64_t queue, int32_t id)
{
   static const char function[] = "sceAgcDriverDeleteEqEvent";
   struct agc_driver_error error = {0};
   if (id < 0) {
      agc_report(function, "negative event id");
      return AGC_DRIVER_ERROR_INVALID;
   }
   if (agc_events_delete(queue, (uint64_t)id, &error)) {
      agc_report(function, "%s", error.message);
      return AGC_DRIVER_ERROR_INVALID;
   }
   return 0;
}

static bool
event_valid(const struct agc_kernel_event *event, const char *function)
{
   if (event && (uintptr_t)event % _Alignof(struct agc_kernel_event) == 0)
      return true;
   agc_report(function, "null or misaligned event");
   return false;
}

uint32_t AGC_SYSV
sceAgcDriverGetEqContextId(const struct agc_kernel_event *event)
{
   static const char function[] = "sceAgcDriverGetEqContextId";
   if (!event_valid(event, function))
      return 0;
   if (event->filter != GRAPHICS_EVENT_FILTER) {
      agc_report(function, "not a graphics event");
      return 0;
   }
   return (uint32_t)((uint64_t)event->data >> 16);
}

int32_t AGC_SYSV
sceAgcDriverGetEqEventType(const struct agc_kernel_event *event)
{
   if (!event_valid(event, "sceAgcDriverGetEqEventType"))
      return AGC_DRIVER_ERROR_INVALID;
   return (int32_t)event->data;
}

int32_t AGC_SYSV
sceAgcDriverRegisterWorkloadStream(uint32_t id, const void *stream)
{
   if (id < 1 || id >= 32 || (p_atomic_read(&workload_streams) & (UINT32_C(1) << id)))
      return WORKLOAD_STREAM_INVALID_ID;
   if (!stream)
      return WORKLOAD_STREAM_NULL;
   uint32_t old = p_atomic_read(&workload_streams);
   for (;;) {
      if (old & (UINT32_C(1) << id))
         return WORKLOAD_STREAM_INVALID_ID;
      uint32_t seen = p_atomic_cmpxchg(&workload_streams, old, old | (UINT32_C(1) << id));
      if (seen == old)
         return 0;
      old = seen;
   }
}

int32_t AGC_SYSV
sceAgcDriverUnregisterWorkloadStream(uint32_t id)
{
   if (id < 1 || id >= 32)
      return WORKLOAD_STREAM_INVALID_ID;
   uint32_t old = p_atomic_read(&workload_streams);
   for (;;) {
      if (!(old & (UINT32_C(1) << id)))
         return WORKLOAD_STREAM_NOT_REGISTERED;
      uint32_t seen = p_atomic_cmpxchg(&workload_streams, old, old & ~(UINT32_C(1) << id));
      if (seen == old)
         return 0;
      old = seen;
   }
}

int32_t AGC_SYSV
sceAgcDriverSetTFRing(const void *base, uint32_t bytes)
{
   if (!base || !bytes) {
      agc_report("sceAgcDriverSetTFRing", "empty ring");
      return AGC_DRIVER_ERROR_INVALID;
   }
   p_atomic_set(&tf_ring_base, (uint64_t)(uintptr_t)base);
   p_atomic_set(&tf_ring_bytes, bytes);
   agc_driver_set_tess_factor_ring((uint64_t)(uintptr_t)base, bytes);
   return 0;
}

int32_t AGC_SYSV
sceAgcDriverSetHsOffchipParam(uint32_t granularity, uint32_t buffering)
{
   p_atomic_set(&hs_offchip_param, (buffering & 0x1ff) | (granularity & 3) << 9);
   agc_driver_set_hs_offchip((buffering & 0x1ff) | (granularity & 3) << 9);
   return 0;
}

int32_t AGC_SYSV
sceAgcDriverGetTFRing(uint64_t *base, uint32_t *bytes)
{
   if (!base || !bytes) {
      agc_report("sceAgcDriverGetTFRing", "null output");
      return AGC_DRIVER_ERROR_INVALID;
   }
   *base = p_atomic_read(&tf_ring_base);
   *bytes = p_atomic_read(&tf_ring_bytes);
   return 0;
}

int32_t AGC_SYSV
sceAgcDriverGetResourceRegistrationMaxNameLength(uint32_t *length)
{
   if (!length) {
      agc_report("sceAgcDriverGetResourceRegistrationMaxNameLength", "null output");
      return AGC_DRIVER_ERROR_INVALID;
   }
   *length = 0xfc;
   return 0;
}

int32_t AGC_SYSV
sceAgcDriverQueryResourceRegistrationUserMemoryRequirements(uint64_t *bytes)
{
   if (!bytes)
      return EINVAL_KERNEL;
   *bytes = 0x4000;
   return 0;
}

void *AGC_SYSV
sceAgcDriverGetCoredumpAddress(void)
{
   return NULL;
}

int32_t AGC_SYSV
sceAgcDriverGetCoredumpMode(uint32_t *mode)
{
   if (!mode)
      return NULL_OUTPUT;
   *mode = 0;
   return 0;
}

int32_t AGC_SYSV
sceAgcDriverAllocateToolMemoryForGpuReset(uint64_t first, uint64_t second, uint64_t third, uint8_t *range)
{
   (void)first;
   (void)second;
   (void)third;
   if (range)
      memset(range, 0, TOOL_MEMORY_RANGE_BYTES);
   return FEATURE_UNAVAILABLE;
}

int32_t AGC_SYSV
sceAgcDriverGetAllocatedToolMemoryForGpuReset(uint64_t first, uint8_t *range)
{
   (void)first;
   if (range)
      memset(range, 0, TOOL_MEMORY_RANGE_BYTES);
   return FEATURE_UNAVAILABLE;
}

static uint32_t
user_data_dwords(uint32_t bytes)
{
   if (bytes == 0)
      return 3;
   uint32_t payload = (uint32_t)(((uint64_t)bytes + 3) >> 2);
   return payload == 1 ? 4 : payload + 7;
}

uint32_t AGC_SYSV
sceAgcDriverGetSetFlipPacketSizeInDwords(void)
{
   return FLIP_PACKET_DWORDS;
}

uint32_t AGC_SYSV
sceAgcDriverGetWaitRenderingPacketSizeInDwords(void)
{
   return WAIT_RENDERING_PACKET_DWORDS;
}

uint32_t AGC_SYSV
sceAgcDriverGetSetWorkloadsActivePacketSize(void)
{
   return user_data_dwords(8) + 9;
}

uint32_t AGC_SYSV
sceAgcDriverGetSetWorkloadCompletePacketSize(void)
{
   return user_data_dwords(0) + 9;
}

uint32_t AGC_SYSV
sceAgcDriverGetSetWorkloadStreamInactivePacketSize(void)
{
   return user_data_dwords(8);
}

uint32_t AGC_SYSV
sceAgcDriverNopDataGetPacketSize(uint32_t bytes)
{
   return (uint32_t)(((uint64_t)bytes + 3) >> 2) + 2;
}

uint32_t AGC_SYSV
sceAgcDriverUserDataGetPacketSize(uint32_t bytes)
{
   return user_data_dwords(bytes);
}

static uint32_t
user_data_nops(uint32_t *commands, const void *data, uint32_t bytes)
{
   uint32_t dwords = !data || !bytes ? 3 : user_data_dwords(bytes);
   if (commands)
      agc_fill_nops(commands, dwords);
   return dwords;
}

uint32_t AGC_SYSV
sceAgcDriverUserDataWritePacket(uint32_t *commands, uint32_t async_compute, uint32_t value, const void *data,
                                uint32_t bytes)
{
   (void)async_compute;
   (void)value;
   return user_data_nops(commands, data, bytes);
}

uint32_t AGC_SYSV
sceAgcDriverUserDataWritePopMarker(uint32_t *commands, uint32_t async_compute)
{
   (void)async_compute;
   return user_data_nops(commands, NULL, 0);
}

uint32_t AGC_SYSV
sceAgcDriverUserDataWritePushMarker(uint32_t *commands, uint32_t async_compute, const char *name, uint32_t bytes,
                                    uint32_t color)
{
   (void)async_compute;
   (void)color;
   return user_data_nops(commands, name, bytes);
}

uint32_t AGC_SYSV
sceAgcDriverUserDataWriteSetMarker(uint32_t *commands, uint32_t async_compute, const char *name, uint32_t bytes,
                                   uint32_t color)
{
   (void)async_compute;
   (void)color;
   return user_data_nops(commands, name, bytes);
}

int32_t AGC_SYSV
sceAgcDriverNopDataWritePacket(uint32_t *destination, uint32_t address, const uint8_t *data, uint32_t bytes)
{
   if (!destination) {
      agc_report("sceAgcDriverNopDataWritePacket", "null destination");
      return 0;
   }
   bool empty = !data || !bytes;
   uint32_t total = empty ? 2 : (uint32_t)(((uint64_t)bytes + 3) >> 2) + 2;
   destination[0] = PKT3(PKT3_NOP, total - 2, 0);
   destination[1] = address;
   if (!empty) {
      destination[total - 1] = 0;
      memcpy(destination + 2, data, bytes);
   }
   return (int32_t)total;
}

uint32_t AGC_SYSV
sceAgcDriverWaitUntilSafeForRendering(uint32_t **command, uint32_t capacity, uint32_t mode, uint32_t handle,
                                      int32_t index)
{
   static const char function[] = "sceAgcDriverWaitUntilSafeForRendering";
   if (!command || !*command || capacity < WAIT_RENDERING_PACKET_DWORDS || mode != 0 || index < 0) {
      agc_report(function, "invalid rendering wait arguments");
      return 0;
   }
   uint32_t *packet = *command;
   agc_fill_nops(packet, WAIT_RENDERING_PACKET_DWORDS);
   if (agc_record_claim(packet, WAIT_RENDERING_PACKET_DWORDS)) {
      struct agc_reloc *reloc = agc_record_add_reloc(packet, AGC_RELOC_WAIT_RENDERING);
      reloc->arg32 = handle;
      reloc->arg64[0] = (uint32_t)index;
   } else {
      agc_report(function, "the packet is outside the command buffer being recorded; the wait is dropped");
   }
   *command = packet + WAIT_RENDERING_PACKET_DWORDS;
   return 0;
}
