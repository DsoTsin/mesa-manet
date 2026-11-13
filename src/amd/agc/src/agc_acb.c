#include "agc_cb_builders.h"
#include "agc_exports.h"
#include "sid.h"

uint32_t *AGC_SYSV
sceAgcAcbDispatchIndirect(struct agc_cb *cb, uint64_t address, uint32_t modifier)
{
   if (!cb || !agc_address_valid(address, 4) || !agc_dispatch_modifier_valid(modifier)) {
      agc_report(__func__, "invalid dispatch arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_DISPATCH_INDIRECT_ACB);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_DISPATCH_INDIRECT, 2, 0) | PKT3_SHADER_TYPE_S(1);
   out[1] = (uint32_t)address;
   out[2] = (uint32_t)(address >> 32);
   out[3] = agc_dispatch_initiator(modifier);
   return out;
}

uint32_t AGC_SYSV
sceAgcAcbDispatchIndirectGetSize(void)
{
   return AGC_DWORDS_DISPATCH_INDIRECT_ACB * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbWriteData(struct agc_cb *cb, uint8_t dst, uint8_t cache_policy, uint64_t address,
                   const uint32_t *data, uint32_t count, uint8_t increment, uint8_t confirm)
{
   return agc_emit_write_data(cb, true, dst, cache_policy, address, data, count, increment, confirm,
                              __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbCopyData(struct agc_cb *cb, uint8_t dst, uint8_t dst_policy, uint64_t destination,
                  uint8_t src, uint8_t src_policy, uint64_t source, uint8_t item, uint8_t confirm)
{
   return agc_emit_copy_data(cb, true, dst, dst_policy, destination, src, src_policy, source, item,
                             confirm, __func__);
}

uint64_t AGC_SYSV
sceAgcAcbCopyDataGetSize(void)
{
   return AGC_DWORDS_COPY_DATA * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbDmaData(struct agc_cb *cb, uint8_t dst, uint8_t dst_policy, uint64_t destination,
                 uint8_t src, uint8_t src_policy, uint64_t source, uint32_t bytes, uint8_t wait,
                 uint8_t confirm)
{
   return agc_emit_dma_data(cb, true, 0, dst, dst_policy, destination, src, src_policy, source,
                            bytes, wait, confirm, 0, __func__);
}

uint32_t AGC_SYSV
sceAgcAcbDmaDataGetSize(void)
{
   return AGC_DWORDS_DMA_DATA * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbAtomicMem(struct agc_cb *cb, uint8_t op, uint8_t command, uint64_t address,
                   uint8_t cache_policy, uint64_t source, uint64_t compare, uint32_t interval)
{
   return agc_emit_atomic_mem(cb, true, 0, op, command, cache_policy, address, source, compare,
                              interval, __func__);
}

uint32_t AGC_SYSV
sceAgcAcbAtomicMemGetSize(void)
{
   return AGC_DWORDS_ATOMIC_MEM * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbWaitRegMem(struct agc_cb *cb, uint8_t size, uint8_t function, uint8_t cache_policy,
                    uint64_t address, uint64_t reference, uint64_t mask, uint32_t poll)
{
   return agc_emit_wait_reg_mem(cb, true, size, function, 0, cache_policy, address, reference, mask,
                                poll, __func__);
}

uint64_t AGC_SYSV
sceAgcAcbWaitOnAddressGetSize(uint32_t size)
{
   return agc_wait_dwords(size) * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbAcquireMem(struct agc_cb *cb, uint32_t gcr, uint64_t base, uint64_t size, uint32_t poll)
{
   return agc_emit_acquire_mem(cb, true, 1, 0, gcr, base, size, poll, __func__);
}

uint32_t AGC_SYSV
sceAgcAcbAcquireMemGetSize(void)
{
   return AGC_DWORDS_ACQUIRE_MEM * 4;
}

uint32_t AGC_SYSV
sceAgcAcbQueueEndOfShaderActionGetSize(void)
{
   return AGC_DWORDS_RELEASE_MEM * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbEventWrite(struct agc_cb *cb, uint8_t event, uint64_t address)
{
   return agc_emit_event_write(cb, true, event, address, __func__);
}

uint64_t AGC_SYSV
sceAgcAcbEventWriteGetSize(uint8_t event)
{
   (void)event;
   return AGC_DWORDS_EVENT_WRITE * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbPrimeUtcl2(struct agc_cb *cb, uint32_t permission, uint32_t mode, uint64_t address,
                    uint32_t pages)
{
   (void)permission;
   (void)mode;
   (void)address;
   (void)pages;
   return agc_emit_nops(cb, AGC_DWORDS_PRIME_UTCL2, __func__);
}

uint32_t AGC_SYSV
sceAgcAcbPrimeUtcl2GetSize(void)
{
   return AGC_DWORDS_PRIME_UTCL2 * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbJump(struct agc_cb *cb, uint8_t cache_policy, uint64_t target, uint32_t dwords)
{
   return agc_emit_jump(cb, true, 1, cache_policy, target, dwords, __func__);
}

uint32_t AGC_SYSV
sceAgcAcbJumpGetSize(void)
{
   return AGC_DWORDS_JUMP * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbCondExec(struct agc_cb *cb, uint64_t address, uint32_t dwords)
{
   return agc_emit_cond_exec(cb, address, dwords, __func__);
}

uint32_t AGC_SYSV
sceAgcAcbCondExecGetSize(void)
{
   return AGC_DWORDS_COND_EXEC * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbRewind(struct agc_cb *cb, uint32_t state, uint32_t offload)
{
   (void)offload;
   return agc_emit_rewind(cb, state, __func__);
}

uint32_t AGC_SYSV
sceAgcAcbRewindGetSize(void)
{
   return AGC_DWORDS_REWIND * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbPushMarker(struct agc_cb *cb, const char *text, uint32_t color)
{
   (void)color;
   return agc_emit_marker(cb, text, false, __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbPushMarkerSpan(struct agc_cb *cb, const char *text, uint32_t length, uint32_t color)
{
   (void)color;
   return agc_emit_marker_span(cb, text, length, false, __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbSetMarker(struct agc_cb *cb, const char *text, uint32_t color)
{
   (void)color;
   return agc_emit_marker(cb, text, true, __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbSetMarkerSpan(struct agc_cb *cb, const char *text, uint32_t length, uint32_t color)
{
   (void)color;
   return agc_emit_marker_span(cb, text, length, true, __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbPopMarker(struct agc_cb *cb)
{
   return agc_emit_nops(cb, AGC_DWORDS_MARKER_POP, __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbSetFlip(struct agc_cb *cb, uint32_t video_out, int32_t index, uint32_t mode,
                 int64_t argument)
{
   return agc_emit_flip(cb, video_out, index, mode, argument, __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbWaitUntilSafeForRendering(struct agc_cb *cb, uint32_t video_out, int32_t index)
{
   return agc_emit_rendering_wait(cb, video_out, index, __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbSetWorkloadsActive(struct agc_cb *cb, uint32_t stream, const uint32_t *ids,
                            uint32_t count)
{
   return agc_emit_workloads_active(cb, stream, ids, count, __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbSetWorkloadComplete(struct agc_cb *cb, uint32_t stream, uint32_t id)
{
   return agc_emit_workload_complete(cb, stream, id, __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbSetWorkloadStreamInactive(struct agc_cb *cb, uint32_t stream)
{
   return agc_emit_workload_stream_inactive(cb, stream, __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbResetQueue(struct agc_cb *cb, uint32_t operation, uint32_t value)
{
   (void)value;
   if (!cb || operation > AGC_RESET_QUEUE_OPS_MAX) {
      agc_report(__func__, "null command buffer or invalid reset operations");
      return NULL;
   }
   return agc_emit_nops(cb, AGC_DWORDS_RESET_QUEUE_ACB, __func__);
}

uint32_t *AGC_SYSV
sceAgcAcbAtomicGds_0900(struct agc_cb *cb, uint16_t op, uint8_t command, uint8_t float_mode,
                        uint32_t increment, uint32_t base, uint32_t size, uint32_t offset,
                        uint32_t dst, uint64_t source0, uint64_t source1)
{
   (void)size;
   return agc_emit_atomic_gds(cb, true, 0, op, command, float_mode, increment, base, offset, dst,
                              source0, source1, __func__);
}

uint32_t AGC_SYSV
sceAgcAcbAtomicGdsGetSize(void)
{
   return AGC_DWORDS_ATOMIC_GDS * 4;
}

uint32_t *AGC_SYSV
sceAgcAcbMemSemaphore(struct agc_cb *cb, uint64_t address, uint8_t mailbox, uint8_t signal_type,
                      uint8_t operation)
{
   return agc_emit_mem_semaphore(cb, address, mailbox, signal_type, operation, __func__);
}
