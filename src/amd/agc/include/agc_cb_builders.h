#ifndef AGC_CB_BUILDERS_H
#define AGC_CB_BUILDERS_H

#include "agc_cb.h"
#include "agc_debug.h"
#include "agc_record.h"
#include "agc_registers.h"
#include "agc_runtime.h"

#define AGC_SET_REGISTERS_MAX 0x3fffu
#define AGC_REGISTER_OFFSET_MAX 0xffffu

#define AGC_PATCH_INVALID ((int32_t)0x8a6c0001)
#define AGC_PATCH_NOT_ACQUIRE ((int32_t)0x8a6c000c)
#define AGC_ERROR_INVALID_ADDRESS ((int32_t)0x8a6c000a)

#define AGC_NOP_DWORDS_MAX 0x4001u
#define AGC_WRITE_DATA_DWORDS_MAX 0x3ffdu
#define AGC_IB_DWORDS_MAX 0xfffffu
#define AGC_COND_EXEC_DWORDS_MAX 0x3fffu
#define AGC_ADDRESS_MAX UINT64_C(0xffffffffffff)
#define AGC_MARKER_TEXT_MAX 0x10000u
#define AGC_MARKER_PAYLOAD_MAX 0x4000u
#define AGC_RESET_QUEUE_OPS_MAX 0xfffu

#define AGC_SEMAPHORE_SIGNAL 6u
#define AGC_SEMAPHORE_WAIT 7u
#define AGC_MEMSET_ALIGNMENT 256u
#define AGC_MEMSET_CHUNK_BYTES 0x7f00u
#define AGC_MEMSET_CHUNKS_MAX 0x100000u

#define AGC_GDS_OP_ADD 0x00u
#define AGC_GDS_OP_SUB 0x01u
#define AGC_GDS_OP_INC 0x03u
#define AGC_GDS_OP_DEC 0x04u
#define AGC_GDS_OP_MIN_I 0x05u
#define AGC_GDS_OP_MAX_I 0x06u
#define AGC_GDS_OP_MIN_U 0x07u
#define AGC_GDS_OP_MAX_U 0x08u
#define AGC_GDS_OP_AND 0x09u
#define AGC_GDS_OP_OR 0x0au
#define AGC_GDS_OP_XOR 0x0bu
#define AGC_GDS_OP_STORE 0x0du
#define AGC_GDS_OP_CMPSTORE 0x10u
#define AGC_GDS_OP_CMPSTORE_F 0x11u
#define AGC_GDS_OP_READ 0x16u
#define AGC_GDS_OP_KIND 0x1fu
#define AGC_GDS_OP_RETURN 0x20u
#define AGC_GDS_OP_64BIT 0x40u
#define AGC_GDS_OP_SRC2 0x80u
#define AGC_GDS_OP_LOOP 0x200u
#define AGC_GDS_OP_BITS 0x2ffu
#define AGC_GDS_LOOP_INTERVAL 10u
#define AGC_GL2_ATOMIC_64BIT 32u
#define AGC_GL2_ATOMIC_NO_RETURN 64u

enum agc_footprint {
   AGC_DWORDS_ACQUIRE_MEM = 8,
   AGC_DWORDS_ATOMIC_GDS = 11,
   AGC_DWORDS_ATOMIC_MEM = 9,
   AGC_DWORDS_BRANCH = 14,
   AGC_DWORDS_COND_EXEC = 5,
   AGC_DWORDS_COND_WRITE = 9,
   AGC_DWORDS_COPY_DATA = 6,
   AGC_DWORDS_DISPATCH = 5,
   AGC_DWORDS_DISPATCH_INDIRECT_DCB = 3,
   AGC_DWORDS_DISPATCH_INDIRECT_ACB = 4,
   AGC_DWORDS_DMA_DATA = 7,
   AGC_DWORDS_DRAW_INDEX = 6,
   AGC_DWORDS_DRAW_INDEX_AUTO = 3,
   AGC_DWORDS_DRAW_INDEX_OFFSET = 5,
   AGC_DWORDS_DRAW_INDIRECT = 5,
   AGC_DWORDS_DRAW_INDIRECT_MULTI = 16,
   AGC_DWORDS_DRAW_INDEX_MULTI_INSTANCED = 15,
   AGC_DWORDS_EVENT_WRITE = 2,
   AGC_DWORDS_EVENT_WRITE_ADDRESS = 4,
   AGC_DWORDS_FLIP = 6,
   AGC_DWORDS_INDEX_BASE = 3,
   AGC_DWORDS_INDEX_COUNT = 2,
   AGC_DWORDS_INDEX_INDIRECT_ARGS = 4,
   AGC_DWORDS_INDEX_SIZE = 3,
   AGC_DWORDS_JUMP = 4,
   AGC_DWORDS_LOD_STATS = 5,
   AGC_DWORDS_MARKER_POP = 2,
   AGC_DWORDS_MEM_SEMAPHORE = 4,
   AGC_DWORDS_NUM_INSTANCES = 2,
   AGC_DWORDS_PREDICATION = 4,
   AGC_DWORDS_PRIME_UTCL2 = 5,
   AGC_DWORDS_RELEASE_MEM = 8,
   AGC_DWORDS_RENDERING_WAIT = 4,
   AGC_DWORDS_RESET_QUEUE_ACB = 3,
   AGC_DWORDS_RESET_QUEUE_DCB = 4,
   AGC_DWORDS_REWIND = 2,
   AGC_DWORDS_SET_BASE = 4,
   AGC_DWORDS_STALL_PARSER = 2,
   AGC_DWORDS_WAIT_32 = 14,
   AGC_DWORDS_WAIT_64 = 16,
   AGC_DWORDS_WORKLOADS_ACTIVE = 18,
   AGC_DWORDS_WORKLOAD_COMPLETE = 12,
   AGC_DWORDS_WORKLOAD_STREAM_INACTIVE = 9,
   AGC_DWORDS_WRITE_DATA_HEAD = 4,
   AGC_DWORDS_OCCLUSION_RESULTS = 64,
};

struct agc_memory_range {
   uint32_t *base;
   uint64_t size;
};

void agc_fill_nops(uint32_t *out, uint32_t dwords);

uint32_t *agc_emit_set_registers(struct agc_cb *cb, enum agc_reg_bank bank, uint32_t offset,
                                 const uint32_t *values, uint32_t count, const char *function);
uint32_t *agc_emit_set_register_pairs(struct agc_cb *cb, enum agc_reg_bank bank,
                                      const struct agc_register *pairs, uint32_t count,
                                      const char *function);
uint32_t *agc_emit_set_register_range(struct agc_cb *cb, enum agc_reg_bank bank, uint32_t offset,
                                      const uint32_t *values, uint32_t count, const char *function);
uint32_t *agc_emit_set_register_list(struct agc_cb *cb, enum agc_reg_bank bank,
                                     const struct agc_register *pairs, uint32_t count,
                                     const char *function);

uint32_t agc_dispatch_initiator(uint32_t modifier);
bool agc_dispatch_modifier_valid(uint32_t modifier);

uint32_t *agc_emit_nops(struct agc_cb *cb, uint32_t dwords, const char *function);
uint32_t *agc_emit_trampoline(struct agc_cb *cb, uint32_t footprint, const uint32_t *host, uint32_t dwords,
                              const char *function);
uint32_t *agc_trampoline_stub(uint32_t *packet, uint32_t *dwords);
uint32_t agc_event_write_host_dwords(bool compute, uint8_t event);
uint32_t *agc_emit_write_data(struct agc_cb *cb, bool compute, uint8_t dst, uint8_t cache_policy,
                              uint64_t address, const uint32_t *data, uint32_t count,
                              uint8_t increment, uint8_t confirm, const char *function);
uint32_t *agc_emit_copy_data(struct agc_cb *cb, bool compute, uint8_t dst, uint8_t dst_policy,
                             uint64_t destination, uint8_t src, uint8_t src_policy, uint64_t source,
                             uint8_t item, uint8_t confirm, const char *function);
uint32_t *agc_emit_dma_data(struct agc_cb *cb, bool compute, uint8_t engine, uint8_t dst,
                            uint8_t dst_policy, uint64_t destination, uint8_t src,
                            uint8_t src_policy, uint64_t source, uint32_t bytes, uint8_t wait,
                            uint8_t confirm, uint8_t block, const char *function);
uint32_t *agc_emit_atomic_mem(struct agc_cb *cb, bool compute, uint8_t engine, uint8_t op,
                              uint8_t command, uint8_t cache_policy, uint64_t address,
                              uint64_t source, uint64_t compare, uint32_t interval,
                              const char *function);
uint32_t *agc_emit_atomic_gds(struct agc_cb *cb, bool compute, uint8_t engine, uint16_t op,
                              uint8_t command, uint8_t float_mode, uint32_t increment,
                              uint32_t base, uint32_t offset, uint32_t dst, uint64_t source0,
                              uint64_t source1, const char *function);
uint32_t *agc_emit_wait_reg_mem(struct agc_cb *cb, bool compute, uint8_t size, uint8_t compare,
                                uint8_t operation, uint8_t cache_policy, uint64_t address,
                                uint64_t reference, uint64_t mask, uint32_t poll,
                                const char *function);
uint32_t *agc_emit_acquire_mem(struct agc_cb *cb, bool compute, uint8_t engine, uint32_t cb_db,
                               uint32_t gcr, uint64_t base, uint64_t size, uint32_t poll,
                               const char *function);
uint32_t *agc_emit_event_write(struct agc_cb *cb, bool compute, uint8_t event, uint64_t address,
                               const char *function);
uint32_t *agc_emit_jump(struct agc_cb *cb, bool compute, uint8_t chain, uint8_t cache_policy,
                        uint64_t target, uint32_t dwords, const char *function);
uint32_t *agc_emit_cond_exec(struct agc_cb *cb, uint64_t address, uint32_t dwords,
                             const char *function);
uint32_t *agc_emit_rewind(struct agc_cb *cb, uint32_t state, const char *function);
uint32_t *agc_emit_marker(struct agc_cb *cb, const char *text, bool pop, const char *function);
uint32_t *agc_emit_marker_span(struct agc_cb *cb, const char *text, uint32_t length, bool pop,
                               const char *function);
uint32_t *agc_emit_mem_semaphore(struct agc_cb *cb, uint64_t address, uint8_t mailbox,
                                 uint8_t signal_type, uint8_t operation, const char *function);
uint32_t *agc_emit_flip(struct agc_cb *cb, uint32_t video_out, int32_t index, uint32_t mode,
                        int64_t argument, const char *function);
uint32_t *agc_emit_rendering_wait(struct agc_cb *cb, uint32_t video_out, int32_t index,
                                  const char *function);
uint32_t *agc_emit_workloads_active(struct agc_cb *cb, uint32_t stream, const uint32_t *ids,
                                    uint32_t count, const char *function);
uint32_t *agc_emit_workload_complete(struct agc_cb *cb, uint32_t stream, uint32_t id,
                                     const char *function);
uint32_t *agc_emit_workload_stream_inactive(struct agc_cb *cb, uint32_t stream,
                                            const char *function);

static inline uint32_t
agc_set_registers_bytes(uint32_t count)
{
   return (count + 2) * 4;
}

static inline uint32_t
agc_set_register_pairs_bytes(uint32_t count)
{
   return count > UINT32_MAX / 12 ? 0 : count * 12;
}

static inline uint32_t *
agc_packet_alloc(struct agc_cb *cb, uint32_t dwords, const char *function)
{
   if (AGC_UNLIKELY(!cb)) {
      agc_report(function, "null command buffer");
      return NULL;
   }
   return agc_record_alloc(cb, dwords);
}

static inline bool
agc_address_valid(uint64_t address, uint64_t alignment)
{
   return address != 0 && (address & (alignment - 1)) == 0;
}

static inline bool
agc_host_is_gfx12(void)
{
   return agc_host_current() == AGC_HOST_GFX12;
}

static inline uint32_t
agc_host_cache_policy(uint32_t policy)
{
   return agc_host_is_gfx12() ? 0 : policy;
}

static inline uint32_t
agc_event_write_dwords(uint8_t event)
{
   return (event & 0xfe) == 0x38 ? AGC_DWORDS_EVENT_WRITE_ADDRESS : AGC_DWORDS_EVENT_WRITE;
}

static inline uint32_t
agc_wait_dwords(uint32_t size)
{
   return size == 0 ? AGC_DWORDS_WAIT_32 : size == 1 ? AGC_DWORDS_WAIT_64 : 0;
}

#endif
