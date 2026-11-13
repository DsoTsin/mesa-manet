#include "agc_cb_builders.h"

#include <stdlib.h>
#include <string.h>

#include "ac_cmdbuf.h"
#include "ac_rgp.h"
#include "ac_cmdbuf_cp.h"
#include "agc_exports.h"
#include "agc_graphics.h"
#include "agc_memory.h"
#include "sid.h"
#include "util/u_atomic.h"

#define AGC_NOP_MAX_DWORDS 0x4000u
#define AGC_DISPATCH_MODIFIER_MASK 0xa079u
#define AGC_DMA_BYTES_MAX 0x3ffffffu
#define AGC_DMA_BYTES_MAX_GFX11 32767u
#define AGC_DMA_CHUNK_GFX11 (AGC_DMA_BYTES_MAX_GFX11 & ~31u)
#define AGC_ACQUIRE_POLL_DIVISOR 40u
#define AGC_NOP_PAD_MASK 0x3fffff00u
#define AGC_COPY_GDS_ATOMIC_RETURN_0 7u

uint32_t agc_flip_records;
uintptr_t agc_flip_last;

enum agc_event_form {
   AGC_EVENT_NONE,
   AGC_EVENT_FORWARD,
   AGC_EVENT_PARTIAL_FLUSH,
   AGC_EVENT_DROP,
   AGC_EVENT_TIMESTAMP,
   AGC_EVENT_DB_META,
   AGC_EVENT_CB_META,
   AGC_EVENT_OCCLUSION_DUMP,
   AGC_EVENT_OCCLUSION_CONTROL,
};

static const uint8_t set_reg_opcode[AGC_REG_BANK_COUNT] = {
   [AGC_REG_BANK_CX] = PKT3_SET_CONTEXT_REG,
   [AGC_REG_BANK_SH] = PKT3_SET_SH_REG,
   [AGC_REG_BANK_UC] = PKT3_SET_UCONFIG_REG,
};

static const uint8_t dcb_event_form[64] = {
   [V_028A90_CS_PARTIAL_FLUSH] = AGC_EVENT_PARTIAL_FLUSH,
   [V_028A90_RST_PIX_CNT] = AGC_EVENT_FORWARD,
   [V_028A90_BREAK_BATCH] = AGC_EVENT_FORWARD,
   [V_028A90_VS_PARTIAL_FLUSH] = AGC_EVENT_PARTIAL_FLUSH,
   [V_028A90_PS_PARTIAL_FLUSH] = AGC_EVENT_PARTIAL_FLUSH,
   [V_028A90_FLUSH_DFSM] = AGC_EVENT_FORWARD,
   [V_028A90_CACHE_FLUSH_AND_INV_EVENT] = AGC_EVENT_TIMESTAMP,
   [V_028A90_PERFCOUNTER_START] = AGC_EVENT_DROP,
   [V_028A90_PERFCOUNTER_STOP] = AGC_EVENT_DROP,
   [V_028A90_PERFCOUNTER_SAMPLE] = AGC_EVENT_DROP,
   [V_028A90_DB_CACHE_FLUSH_AND_INV] = AGC_EVENT_TIMESTAMP,
   [V_028A90_FLUSH_AND_INV_DB_META] = AGC_EVENT_DB_META,
   [V_028A90_FLUSH_AND_INV_CB_META] = AGC_EVENT_CB_META,
   [V_028A90_FLUSH_AND_INV_CB_PIXEL_DATA] = AGC_EVENT_TIMESTAMP,
   [V_028A90_PIXEL_PIPE_STAT_CONTROL] = AGC_EVENT_OCCLUSION_CONTROL,
   [V_028A90_PIXEL_PIPE_STAT_DUMP] = AGC_EVENT_OCCLUSION_DUMP,
   [V_028A90_PIXEL_PIPE_STAT_RESET] = AGC_EVENT_OCCLUSION_CONTROL,
};

static const uint8_t acb_event_form[64] = {
   [V_028A90_CS_PARTIAL_FLUSH] = AGC_EVENT_PARTIAL_FLUSH,
   [V_028A90_PERFCOUNTER_START] = AGC_EVENT_DROP,
   [V_028A90_PERFCOUNTER_STOP] = AGC_EVENT_DROP,
   [V_028A90_PERFCOUNTER_SAMPLE] = AGC_EVENT_DROP,
};

static inline void
put_address(uint32_t *out, uint64_t address)
{
   out[0] = (uint32_t)address;
   out[1] = (uint32_t)(address >> 32);
}

static inline bool
host_before_gfx11(void)
{
   return agc_host_current() == AGC_HOST_GFX103;
}

void
agc_fill_nops(uint32_t *out, uint32_t dwords)
{
   while (dwords) {
      if (dwords == 1) {
         out[0] = PKT3_NOP_PAD;
         return;
      }
      uint32_t chunk = dwords;
      if (chunk > AGC_NOP_MAX_DWORDS)
         chunk = dwords - AGC_NOP_MAX_DWORDS == 1 ? AGC_NOP_MAX_DWORDS - 1 : AGC_NOP_MAX_DWORDS;
      out[0] = PKT3(PKT3_NOP, chunk - 2, 0);
      memset(out + 1, 0, (chunk - 1) * sizeof(uint32_t));
      out += chunk;
      dwords -= chunk;
   }
}

static uint32_t
host_register_dwords(const struct agc_register *host, uint32_t count)
{
   uint32_t dwords = 0;
   for (uint32_t i = 0; i < count; i++) {
      if (i == 0 || host[i].offset != host[i - 1].offset + 1)
         dwords += 2;
      dwords++;
   }
   return dwords;
}

static void
put_host_registers(uint32_t *out, enum agc_reg_bank bank, const struct agc_register *host, uint32_t count)
{
   uint32_t *header = NULL;
   uint32_t run = 0;
   for (uint32_t i = 0; i < count; i++) {
      if (i == 0 || host[i].offset != host[i - 1].offset + 1) {
         if (header)
            header[0] = PKT3(set_reg_opcode[bank], run, 0);
         header = out;
         out[1] = host[i].offset;
         out += 2;
         run = 0;
      }
      *out++ = host[i].value;
      run++;
   }
   if (header)
      header[0] = PKT3(set_reg_opcode[bank], run, 0);
}

static uint32_t *
emit_host_registers(struct agc_cb *cb, enum agc_reg_bank bank, uint32_t footprint, const struct agc_register *guest,
                    uint32_t count, const char *function)
{
   uint32_t bound = agc_reg_convert_pairs_bound(count);
   struct agc_register *host = bound ? malloc(bound * sizeof(*host)) : NULL;
   if (!host)
      agc_fatal(function, "out of memory");
   uint32_t written = agc_reg_convert_pairs(bank, guest, count, host, function);
   uint32_t dwords = host_register_dwords(host, written);
   uint32_t *packets = malloc((dwords ? dwords : 1) * sizeof(uint32_t));
   if (!packets)
      agc_fatal(function, "out of memory");
   put_host_registers(packets, bank, host, written);
   uint32_t *out;
   if (dwords > footprint && footprint < AGC_DWORDS_JUMP) {
      out = agc_record_alloc(cb, dwords);
      if (out)
         memcpy(out, packets, dwords * sizeof(uint32_t));
   } else {
      out = agc_emit_trampoline(cb, footprint, packets, dwords, function);
   }
   free(packets);
   free(host);
   return out;
}

uint32_t *
agc_emit_set_registers(struct agc_cb *cb, enum agc_reg_bank bank, uint32_t offset,
                       const uint32_t *values, uint32_t count, const char *function)
{
   if (AGC_LIKELY(agc_reg_run_passes(agc_host_regs, bank, offset, count))) {
      uint32_t *out = agc_record_alloc(cb, count + 2);
      if (AGC_UNLIKELY(!out))
         return NULL;
      out[0] = PKT3(set_reg_opcode[bank], count, 0);
      out[1] = offset;
      if (values)
         memcpy(out + 2, values, count * sizeof(uint32_t));
      return out;
   }

   if (!values) {
      agc_fatal(function, "registers %s:0x%03x+%u are filled in later and need converting",
                agc_reg_bank_name(bank), offset, count);
   }

   struct agc_register *guest = malloc(count * sizeof(*guest));
   if (!guest)
      agc_fatal(function, "out of memory");
   for (uint32_t i = 0; i < count; i++) {
      guest[i].offset = offset + i;
      guest[i].value = values[i];
   }
   uint32_t *out = emit_host_registers(cb, bank, count + 2, guest, count, function);
   free(guest);
   return out;
}

uint32_t *
agc_emit_set_register_pairs(struct agc_cb *cb, enum agc_reg_bank bank,
                            const struct agc_register *pairs, uint32_t count, const char *function)
{
   return emit_host_registers(cb, bank, count * 3, pairs, count, function);
}

uint32_t *
agc_emit_set_register_range(struct agc_cb *cb, enum agc_reg_bank bank, uint32_t offset,
                            const uint32_t *values, uint32_t count, const char *function)
{
   if (!cb || count == 0 || count > AGC_SET_REGISTERS_MAX || offset > AGC_REGISTER_OFFSET_MAX ||
       count > AGC_REGISTER_OFFSET_MAX + 1 - offset) {
      agc_report(function, "invalid register range");
      return NULL;
   }
   return agc_emit_set_registers(cb, bank, offset, values, count, function);
}

uint32_t *
agc_emit_set_register_list(struct agc_cb *cb, enum agc_reg_bank bank,
                           const struct agc_register *pairs, uint32_t count, const char *function)
{
   if (!cb) {
      agc_report(function, "null command buffer");
      return NULL;
   }
   if (count == 0)
      return cb->up;
   if (!pairs || count > UINT32_MAX / 12) {
      agc_report(function, "null or oversized register list");
      return NULL;
   }
   return agc_emit_set_register_pairs(cb, bank, pairs, count, function);
}

bool
agc_dispatch_modifier_valid(uint32_t modifier)
{
   return (modifier & ~AGC_DISPATCH_MODIFIER_MASK) == 0;
}

uint32_t
agc_dispatch_initiator(uint32_t modifier)
{
   const uint32_t passed = S_00B800_ORDERED_APPEND_ENBL(1) | S_00B800_ORDERED_APPEND_MODE(1) |
                           S_00B800_USE_THREAD_DIMENSIONS(1) | S_00B800_TUNNEL_ENABLE(1) |
                           S_00B800_CS_W32_EN(1);
   return S_00B800_COMPUTE_SHADER_EN(1) | S_00B800_ORDER_MODE(1) | (modifier & passed);
}

uint32_t *
agc_emit_nops(struct agc_cb *cb, uint32_t dwords, const char *function)
{
   uint32_t *out = agc_packet_alloc(cb, dwords, function);
   if (AGC_UNLIKELY(!out))
      return NULL;
   agc_fill_nops(out, dwords);
   return out;
}

#define AGC_TRAMPOLINE_PAD 8u

uint32_t *
agc_emit_trampoline(struct agc_cb *cb, uint32_t footprint, const uint32_t *host, uint32_t dwords,
                    const char *function)
{
   if (dwords > footprint && footprint < AGC_DWORDS_JUMP)
      agc_fatal(function, "a %u-dword packet cannot hold a call to %u host dwords", footprint, dwords);
   uint32_t *out = agc_record_alloc(cb, footprint);
   if (AGC_UNLIKELY(!out))
      return NULL;
   if (dwords <= footprint) {
      memcpy(out, host, dwords * sizeof(uint32_t));
      agc_fill_nops(out + dwords, footprint - dwords);
      return out;
   }
   uint32_t padded = (dwords + AGC_TRAMPOLINE_PAD - 1) & ~(AGC_TRAMPOLINE_PAD - 1);
   uint64_t va;
   uint32_t *stub = agc_record_stub(out, padded, &va);
   memcpy(stub, host, dwords * sizeof(uint32_t));
   agc_fill_nops(stub + dwords, padded - dwords);
   agc_mem_flush_write_combining();
   out[0] = PKT3(PKT3_INDIRECT_BUFFER, 2, 0);
   out[1] = (uint32_t)va;
   out[2] = (uint32_t)(va >> 32);
   out[3] = S_3F3_IB_SIZE(padded) | S_3F3_VALID(1);
   agc_fill_nops(out + AGC_DWORDS_JUMP, footprint - AGC_DWORDS_JUMP);
   return out;
}

uint32_t *
agc_trampoline_stub(uint32_t *packet, uint32_t *dwords)
{
   if (!packet || ((uintptr_t)packet & 3) || (packet[0] & ~UINT32_C(0xff)) != PKT3(PKT3_INDIRECT_BUFFER, 2, 0) ||
       (packet[3] & S_3F3_CHAIN(1)))
      return NULL;
   uint64_t va = packet[1] | (uint64_t)packet[2] << 32;
   *dwords = G_3F3_IB_SIZE(packet[3]);
   return agc_record_stub_cpu(packet, va, *dwords);
}

static enum amd_gfx_level
host_gfx_level(void)
{
   switch (agc_host_current()) {
   case AGC_HOST_GFX103:
      return GFX10_3;
   case AGC_HOST_GFX11:
      return GFX11;
   case AGC_HOST_GFX115:
      return GFX11_5;
   default:
      return GFX12;
   }
}

uint32_t *
agc_emit_write_data(struct agc_cb *cb, bool compute, uint8_t dst, uint8_t cache_policy,
                    uint64_t address, const uint32_t *data, uint32_t count, uint8_t increment,
                    uint8_t confirm, const char *function)
{
   if (!cb || !data || count == 0 || count > AGC_WRITE_DATA_DWORDS_MAX ||
       dst > (compute ? 0xf : 0x1f) || cache_policy > 3 || increment > 1 || confirm > 1 ||
       (!compute && (address & 3)) || (dst == 0 && confirm)) {
      agc_report(function, "invalid write data arguments");
      return NULL;
   }
   uint32_t select = compute ? dst : dst >> 1;
   uint32_t engine = compute ? 0 : dst & 1;
   if (select != V_371_TC_L2 && select != V_371_MEMORY)
      agc_fatal(function, "write data destination %u has no host form", select);

   uint32_t *out = agc_record_alloc(cb, count + AGC_DWORDS_WRITE_DATA_HEAD);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_WRITE_DATA, count + 2, 0);
   out[1] = S_371_DST_SEL(select) | S_371_ADDR_INCR(increment) | S_371_WR_CONFIRM(confirm) |
            S_371_CACHE_POLICY(agc_host_cache_policy(cache_policy)) | S_371_ENGINE_SEL(engine);
   put_address(out + 2, address);
   memcpy(out + 4, data, count * sizeof(uint32_t));
   agc_record_note_write(out, 2, address);
   return out;
}

uint32_t *
agc_emit_copy_data(struct agc_cb *cb, bool compute, uint8_t dst, uint8_t dst_policy,
                   uint64_t destination, uint8_t src, uint8_t src_policy, uint64_t source,
                   uint8_t item, uint8_t confirm, const char *function)
{
   uint32_t src_select = compute ? src & 0xf : (src >> 1) & 0xf;
   uint32_t dst_select = compute ? dst & 0xf : (dst >> 1) & 0xf;
   uint32_t engine = compute ? 0 : src & 1;
   if (!cb || dst_policy > 3 || src_policy > 3 || item > 1 || confirm > 1 ||
       (src_select == V_401_IMMEDIATE_DATA && item == 0 && (source >> 32))) {
      agc_report(function, "invalid copy data arguments");
      return NULL;
   }
   bool micro_engine = engine == V_401_MICRO_ENGINE;
   bool src_ok = src_select == V_401_TC_L2 || src_select == V_401_IMMEDIATE_DATA ||
                 (micro_engine &&
                  (src_select == V_401_ATOMIC_RETURN_DATA || src_select == V_401_GPU_CLOCK_COUNT ||
                   src_select == AGC_COPY_GDS_ATOMIC_RETURN_0));
   if (!src_ok || dst_select != V_401_TC_L2) {
      agc_fatal(function, "copy data from source %u to destination %u has no host form",
                src_select, dst_select);
   }
   if (src_select == AGC_COPY_GDS_ATOMIC_RETURN_0)
      src_select = V_401_ATOMIC_RETURN_DATA;

   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_COPY_DATA);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_COPY_DATA, 4, 0);
   out[1] = S_401_SRC_SEL(src_select) | S_401_DST_SEL(dst_select) |
            S_401_SRC_CACHE_POLICY(agc_host_cache_policy(src_policy)) | S_401_COUNT_SEL(item) |
            S_401_WR_CONFIRM(confirm) | S_401_DST_CACHE_POLICY(agc_host_cache_policy(dst_policy)) |
            S_401_ENGINE_SEL(engine);
   put_address(out + 2, source);
   put_address(out + 4, destination);
   agc_record_note_write(out, 4, destination);
   return out;
}

static void
put_dma_data(uint32_t *out, uint32_t header, uint64_t source, uint64_t destination, uint32_t control)
{
   out[0] = PKT3(PKT3_DMA_DATA, 5, 0);
   out[1] = header;
   put_address(out + 2, source);
   put_address(out + 4, destination);
   out[6] = control;
}

static bool
gds_place(uint64_t *address, uint32_t bytes, const char *function)
{
   uint32_t offset = (uint32_t)*address;
   if ((uint64_t)offset + bytes > AGC_RUNTIME_GDS_BYTES) {
      agc_report(function, "a DMA of %u bytes at GDS byte 0x%x leaves the GDS", bytes, offset);
      return false;
   }
   *address = AGC_RUNTIME_GDS_VA + offset;
   return true;
}

uint32_t *
agc_emit_dma_data(struct agc_cb *cb, bool compute, uint8_t engine, uint8_t dst, uint8_t dst_policy,
                  uint64_t destination, uint8_t src, uint8_t src_policy, uint64_t source,
                  uint32_t bytes, uint8_t wait, uint8_t confirm, uint8_t block,
                  const char *function)
{
   bool console_source = src == 0x14 || src == 0x24 || src == (compute ? 0x25 : 0x64);
   if (!cb || engine > 1 || dst > 0xf || dst_policy > 3 || src_policy > 3 || wait > 1 ||
       confirm > 1 || block > 1 || bytes > AGC_DMA_BYTES_MAX || (!console_source && src > 0xf)) {
      agc_report(function, "invalid DMA arguments");
      return NULL;
   }
   uint32_t src_select = src & 3;
   uint32_t dst_select = dst & 3;
   if (console_source)
      agc_fatal(function, "DMA source 0x%02x reads console registers and has no host form", src);
   if ((src & 4) || (dst & 4))
      agc_fatal(function, "DMA through the register address space has no host form");
   if (src_select == V_501_GDS) {
      if (!gds_place(&source, bytes, function))
         return NULL;
      src_select = V_501_SRC_ADDR_USING_L2;
      src_policy = 0;
   }
   if (dst_select == V_501_GDS) {
      if (!gds_place(&destination, bytes, function))
         return NULL;
      dst_select = V_501_DST_ADDR_USING_L2;
      dst_policy = 0;
   }
   uint32_t header = S_501_SRC_CACHE_POLICY(agc_host_cache_policy(src_policy)) | S_501_DST_SEL(dst_select) |
                     S_501_DST_CACHE_POLICY(agc_host_cache_policy(dst_policy)) | S_501_SRC_SEL(src_select);
   uint32_t sync = 0;
   if (!compute) {
      header |= S_501_ENGINE_SEL(engine);
      sync = S_501_CP_SYNC(block);
   }
   uint32_t control = S_506_SAIC((src >> 3) & 1) | S_506_DAIC((dst >> 3) & 1) | S_506_DISABLE_WR_CONFIRM(confirm);

   if (host_before_gfx11() || bytes <= AGC_DMA_BYTES_MAX_GFX11) {
      uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_DMA_DATA);
      if (AGC_UNLIKELY(!out))
         return NULL;
      put_dma_data(out, header | sync, source, destination, control | S_506_RAW_WAIT(wait) | S_506_BYTE_COUNT(bytes));
      if (dst_select == V_501_DST_ADDR_USING_L2 || dst_select == V_501_DST_ADDR_USING_DAS)
         agc_record_note_write(out, 4, destination);
      return out;
   }

   uint32_t chunks = (bytes + AGC_DMA_CHUNK_GFX11 - 1) / AGC_DMA_CHUNK_GFX11;
   uint32_t *host = malloc((size_t)chunks * AGC_DWORDS_DMA_DATA * sizeof(uint32_t));
   if (!host)
      agc_fatal(function, "out of memory");
   bool source_moves = src_select != V_501_DATA && !((src >> 3) & 1);
   bool destination_moves = !((dst >> 3) & 1);
   for (uint32_t i = 0; i < chunks; i++) {
      uint32_t offset = i * AGC_DMA_CHUNK_GFX11;
      uint32_t count = bytes - offset < AGC_DMA_CHUNK_GFX11 ? bytes - offset : AGC_DMA_CHUNK_GFX11;
      bool last = i + 1 == chunks;
      put_dma_data(host + i * AGC_DWORDS_DMA_DATA, header | (last ? sync : 0),
                   source + (source_moves ? offset : 0), destination + (destination_moves ? offset : 0),
                   control | S_506_RAW_WAIT(i == 0 ? wait : 0) | S_506_BYTE_COUNT(count));
   }
   uint32_t *out = agc_emit_trampoline(cb, AGC_DWORDS_DMA_DATA, host, chunks * AGC_DWORDS_DMA_DATA, function);
   free(host);
   if (out && (dst_select == V_501_DST_ADDR_USING_L2 || dst_select == V_501_DST_ADDR_USING_DAS))
      agc_record_note_write(out, AGC_LABEL_WRITE_ARG, destination);
   return out;
}

static void
put_atomic_mem(uint32_t *out, bool compute, uint32_t engine, uint32_t op, uint32_t command,
               uint32_t cache_policy, uint64_t address, uint64_t source, uint64_t compare,
               uint32_t interval)
{
   out[0] = PKT3(PKT3_ATOMIC_MEM, 7, 0);
   out[1] = S_1E1_ATOMIC(op) | S_1E1_COMMAND(command) |
            S_1E1_CACHE_POLICY(agc_host_cache_policy(cache_policy));
   if (!compute)
      out[1] |= S_1E1_ENGINE_SEL(engine);
   put_address(out + 2, address);
   put_address(out + 4, source);
   put_address(out + 6, compare);
   out[8] = S_1E8_LOOP_INTERVAL(interval);
}

uint32_t *
agc_emit_atomic_mem(struct agc_cb *cb, bool compute, uint8_t engine, uint8_t op, uint8_t command,
                    uint8_t cache_policy, uint64_t address, uint64_t source, uint64_t compare,
                    uint32_t interval, const char *function)
{
   if (!cb || engine > 1 || op > 0x7f || command > V_1E1_SEND_AND_CONTINUE || cache_policy > 3 ||
       interval > 0x1fff || !agc_address_valid(address, (op & 0x60) == 0x60 ? 8 : 4)) {
      agc_report(function, "invalid atomic arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_ATOMIC_MEM);
   if (AGC_UNLIKELY(!out))
      return NULL;
   put_atomic_mem(out, compute, engine, op, command, cache_policy, address, source, compare,
                  interval);
   agc_record_note_write(out, 2, address);
   return out;
}

static const uint8_t gds_atomic_op[32] = {
   [AGC_GDS_OP_ADD] = V_1E1_GL2_OP_ATOMIC_ADD_RTN_32,
   [AGC_GDS_OP_SUB] = V_1E1_GL2_OP_ATOMIC_SUB_RTN_32,
   [AGC_GDS_OP_INC] = V_1E1_GL2_OP_ATOMIC_INC_RTN_32,
   [AGC_GDS_OP_DEC] = V_1E1_GL2_OP_ATOMIC_DEC_RTN_32,
   [AGC_GDS_OP_MIN_I] = V_1E1_GL2_OP_ATOMIC_SMIN_RTN_32,
   [AGC_GDS_OP_MAX_I] = V_1E1_GL2_OP_ATOMIC_SMAX_RTN_32,
   [AGC_GDS_OP_MIN_U] = V_1E1_GL2_OP_ATOMIC_UMIN_RTN_32,
   [AGC_GDS_OP_MAX_U] = V_1E1_GL2_OP_ATOMIC_UMAX_RTN_32,
   [AGC_GDS_OP_AND] = V_1E1_GL2_OP_ATOMIC_AND_RTN_32,
   [AGC_GDS_OP_OR] = V_1E1_GL2_OP_ATOMIC_OR_RTN_32,
   [AGC_GDS_OP_XOR] = V_1E1_GL2_OP_ATOMIC_XOR_RTN_32,
   [AGC_GDS_OP_STORE] = V_1E1_GL2_OP_ATOMIC_SWAP_RTN_32,
   [AGC_GDS_OP_CMPSTORE] = V_1E1_GL2_OP_ATOMIC_CMPSWAP_RTN_32,
   [AGC_GDS_OP_READ] = V_1E1_GL2_OP_ATOMIC_ADD_RTN_32,
};

uint32_t *
agc_emit_atomic_gds(struct agc_cb *cb, bool compute, uint8_t engine, uint16_t op, uint8_t command,
                    uint8_t float_mode, uint32_t increment, uint32_t base, uint32_t offset,
                    uint32_t dst, uint64_t source0, uint64_t source1, const char *function)
{
   uint32_t kind = op & AGC_GDS_OP_KIND;
   bool loop = op & AGC_GDS_OP_LOOP;
   bool wide = op & AGC_GDS_OP_64BIT;
   bool ret = op & AGC_GDS_OP_RETURN;
   if (!cb || engine > 1 || command > 1 || float_mode > 1 || (op & ~AGC_GDS_OP_BITS) ||
       (loop && kind != AGC_GDS_OP_CMPSTORE && kind != AGC_GDS_OP_CMPSTORE_F)) {
      agc_report(function, "invalid GDS atomic arguments");
      return NULL;
   }
   if (!gds_atomic_op[kind] || (op & AGC_GDS_OP_SRC2) || (kind == AGC_GDS_OP_READ && !ret))
      agc_fatal(function, "GDS operation 0x%03x has no host form", op);
   if (increment)
      agc_fatal(function, "GDS operations that increment their address have no host form");

   uint32_t host_op = gds_atomic_op[kind] + (wide ? AGC_GL2_ATOMIC_64BIT : 0) +
                      (ret ? 0 : AGC_GL2_ATOMIC_NO_RETURN);
   uint32_t host_command = loop ? V_1E1_LOOP_UNTIL_COMPARE_SATISFIED
                           : command && !ret ? V_1E1_WAIT_FOR_WRITE_CONFIRMATION
                                             : V_1E1_SINGLE_PASS_ATOMIC;
   uint32_t byte = (base & 0xffff) + dst + (offset & 0xff);
   byte &= (uint32_t)(AGC_RUNTIME_GDS_BYTES - 1) & (wide ? ~UINT32_C(7) : ~UINT32_C(3));
   uint64_t source = kind == AGC_GDS_OP_CMPSTORE ? source1 : kind == AGC_GDS_OP_READ ? 0 : source0;
   uint64_t compare = kind == AGC_GDS_OP_CMPSTORE ? source0 : 0;

   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_ATOMIC_GDS);
   if (AGC_UNLIKELY(!out))
      return NULL;
   put_atomic_mem(out, compute, engine, host_op, host_command, 0, AGC_RUNTIME_GDS_VA + byte, source,
                  compare, AGC_GDS_LOOP_INTERVAL);
   agc_fill_nops(out + AGC_DWORDS_ATOMIC_MEM, AGC_DWORDS_ATOMIC_GDS - AGC_DWORDS_ATOMIC_MEM);
   return out;
}

uint32_t *
agc_emit_wait_reg_mem(struct agc_cb *cb, bool compute, uint8_t size, uint8_t compare,
                      uint8_t operation, uint8_t cache_policy, uint64_t address, uint64_t reference,
                      uint64_t mask, uint32_t poll, const char *function)
{
   if (!cb || size > 1 || compare > V_3C1_GREATER_THAN_REFERENCE_VALUE ||
       operation > (size == 0 ? 0xf : 7) || cache_policy > 3 || (poll >> 4) > 0xffff ||
       (address & (size == 0 ? 3 : 7)) || address > AGC_ADDRESS_MAX) {
      agc_report(function, "invalid wait arguments");
      return NULL;
   }
   uint32_t engine = size == 0 ? operation & 3 : operation & 1;
   uint32_t wait_operation = size == 0 ? operation >> 2 : operation >> 1;
   if (engine > V_3C1_PREFETCH_PARSER || wait_operation != V_3C1_WAIT_REG_MEM)
      agc_fatal(function, "wait operation %u has no host form", operation);

   uint32_t total = agc_wait_dwords(size);
   uint32_t *out = agc_record_alloc(cb, total);
   if (AGC_UNLIKELY(!out))
      return NULL;
   uint32_t control = S_3C1_FUNCTION(compare) | S_3C1_MEM_SPACE(V_3C1_MEMORY_SPACE) |
                      S_3C1_CACHE_POLICY(agc_host_cache_policy(cache_policy));
   if (compute)
      control |= S_3C1_OPERATION(V_3C1_WAIT_MEM_PREEMPTABLE);
   else
      control |= S_3C1_ENGINE_SEL(engine);

   uint32_t used;
   if (size == 0) {
      out[0] = PKT3(PKT3_WAIT_REG_MEM, 5, 0);
      out[1] = control;
      put_address(out + 2, address);
      out[4] = (uint32_t)reference;
      out[5] = (uint32_t)mask;
      out[6] = S_3C6_POLL_INTERVAL(poll >> 4);
      used = 7;
   } else {
      out[0] = PKT3(PKT3_WAIT_REG_MEM64, 7, 0);
      out[1] = control;
      put_address(out + 2, address);
      put_address(out + 4, reference);
      put_address(out + 6, mask);
      out[8] = S_938_POLL_INTERVAL(poll >> 4);
      used = 9;
   }
   agc_fill_nops(out + used, total - used);
   struct agc_reloc *reloc = agc_record_add_reloc(out, AGC_RELOC_LABEL_WAIT);
   reloc->flags = (uint16_t)(compare | (size ? AGC_LABEL_WAIT_64 : 0) | AGC_LABEL_WAIT_FROM_PACKET);
   return out;
}

static uint32_t
acquire_gcr(uint32_t gcr)
{
   const uint32_t kept = S_587_GLM_WB(1) | S_587_GLM_INV(1) | S_587_GLK_WB(1) | S_587_GLK_INV(1) |
                         S_587_GLV_INV(1) | S_587_GL1_INV(1) | S_587_GL2_INV(1) | S_587_GL2_WB(1) |
                         S_587_SEQ(3);
   uint32_t host = gcr & kept;
   if (G_587_GLI_INV(gcr))
      host |= S_587_GLI_INV(V_587_GLI_ALL);
   if (G_587_GL2_DISCARD(gcr))
      host |= S_587_GL2_INV(1) | S_587_GL2_WB(1);
   if (host & (S_587_GLM_WB(1) | S_587_GLM_INV(1)))
      host |= S_587_GLM_WB(1) | S_587_GLM_INV(1);
   if (agc_host_is_gfx12())
      host &= ~(S_587_GL1_INV(1) | S_587_GLM_WB(1) | S_587_GLM_INV(1));
   return host & ~S_587_SEQ(3) ? host : 0;
}

#define ACQUIRE_CB_BITS (S_0085F0_CB_ACTION_ENA(1) | (UINT32_C(0xff) << 6))
#define ACQUIRE_DB_BITS (S_0085F0_DB_ACTION_ENA(1) | S_0085F0_DB_DEST_BASE_ENA(1))
#define ACQUIRE_CB_DB_HOST_DWORDS 32u

static uint32_t *
acquire_mem_cb_db(struct agc_cb *cb, uint8_t engine, uint32_t cb_db, uint32_t host_gcr, const char *function)
{
   enum amd_gfx_level level = host_gfx_level();
   if (level < GFX11)
      agc_fatal(function, "a color or depth block flush in an acquire has no gfx10.3 form yet");
   bool flush_cb = cb_db & ACQUIRE_CB_BITS;
   bool flush_db = cb_db & ACQUIRE_DB_BITS;
   if (!flush_cb && !flush_db) {
      flush_cb = true;
      flush_db = true;
   }
   uint32_t event;
   if ((flush_cb && flush_db) || (level == GFX11 && flush_db))
      event = V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT;
   else if (flush_cb)
      event = V_028A90_FLUSH_AND_INV_CB_DATA_TS;
   else
      event = V_028A90_FLUSH_AND_INV_DB_DATA_TS;

   uint32_t host[ACQUIRE_CB_DB_HOST_DWORDS];
   struct ac_cmdbuf cs = {.buf = host, .max_dw = ACQUIRE_CB_DB_HOST_DWORDS};
   if (level < GFX12 && event != V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT) {
      uint32_t meta = event == V_028A90_FLUSH_AND_INV_CB_DATA_TS ? V_028A90_FLUSH_AND_INV_CB_META
                                                                 : V_028A90_FLUSH_AND_INV_DB_META;
      host[cs.cdw++] = PKT3(PKT3_EVENT_WRITE, 0, 0);
      host[cs.cdw++] = EVENT_TYPE(meta) | EVENT_INDEX(0);
   }
   uint32_t gcr = host_gcr | S_587_SEQ(V_587_SEQ_FORWARD);
   ac_emit_cp_release_mem_pws(&cs, level, AMD_IP_GFX, event, gcr & C_587_GLI_INV);
   ac_emit_cp_acquire_mem_pws(&cs, level, AMD_IP_GFX, event,
                              engine == V_581A_PREFETCH_PARSER ? V_581B_CP_PFP : V_581B_CP_ME, 0,
                              gcr & ~C_587_GLI_INV);
   return agc_emit_trampoline(cb, AGC_DWORDS_ACQUIRE_MEM, host, cs.cdw, function);
}

uint32_t *
agc_emit_acquire_mem(struct agc_cb *cb, bool compute, uint8_t engine, uint32_t cb_db, uint32_t gcr,
                     uint64_t base, uint64_t size, uint32_t poll, const char *function)
{
   (void)base;
   (void)size;
   if (!cb || engine > 1 || cb_db > 0x7fffffff || gcr > 0x7ffff ||
       poll / AGC_ACQUIRE_POLL_DIVISOR > 0xffff) {
      agc_report(function, "invalid acquire arguments");
      return NULL;
   }
   if (cb_db && !compute)
      return acquire_mem_cb_db(cb, engine, cb_db, acquire_gcr(gcr), function);

   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_ACQUIRE_MEM);
   if (AGC_UNLIKELY(!out))
      return NULL;
   uint32_t host = acquire_gcr(gcr);
   if (!host) {
      uint32_t used = 0;
      if (!compute && engine == V_581A_PREFETCH_PARSER) {
         out[0] = PKT3(PKT3_PFP_SYNC_ME, 0, 0);
         out[1] = 0;
         used = 2;
      }
      agc_fill_nops(out + used, AGC_DWORDS_ACQUIRE_MEM - used);
      return out;
   }
   out[0] = PKT3(PKT3_ACQUIRE_MEM, 6, 0);
   out[1] = compute ? 0 : S_581A_ENGINE_SEL(engine);
   out[2] = 0xffffffff;
   out[3] = compute || host_before_gfx11() ? S_583_COHER_SIZE_HI_MEC(0xff)
                                           : S_583A_COHER_SIZE_HI_PFP(0xffffff);
   out[4] = 0;
   out[5] = 0;
   out[6] = S_586A_POLL_INTERVAL(poll / AGC_ACQUIRE_POLL_DIVISOR);
   out[7] = S_587_GCR_CNTL(host);
   return out;
}

static uint32_t
event_flush_ts(uint8_t event)
{
   switch (event) {
   case V_028A90_FLUSH_AND_INV_CB_PIXEL_DATA:
      return V_028A90_FLUSH_AND_INV_CB_DATA_TS;
   default:
      return V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT;
   }
}

uint32_t
agc_event_write_host_dwords(bool compute, uint8_t event)
{
   if (compute)
      return AGC_DWORDS_EVENT_WRITE;
   if (event < 64 && !host_before_gfx11() &&
       (dcb_event_form[event] == AGC_EVENT_TIMESTAMP ||
        (dcb_event_form[event] == AGC_EVENT_DB_META && agc_host_current() == AGC_HOST_GFX11)))
      return AGC_DWORDS_RELEASE_MEM;
   return agc_event_write_dwords(event);
}

uint32_t *
agc_emit_event_write(struct agc_cb *cb, bool compute, uint8_t event, uint64_t address,
                     const char *function)
{
   bool with_address = !compute && (event & 0xfe) == 0x38;
   bool address_valid = with_address ? agc_address_valid(address, 8) : !compute || !address;
   if (!cb || event > 0x3f || !address_valid) {
      agc_report(function, "invalid event arguments");
      return NULL;
   }
   enum agc_event_form form = (compute ? acb_event_form : dcb_event_form)[event];
   uint32_t flush = 0;
   switch (form) {
   case AGC_EVENT_NONE:
      agc_fatal(function, "event 0x%02x has no host form", event);
   case AGC_EVENT_OCCLUSION_CONTROL:
      agc_fatal(function, "occlusion counter event 0x%02x has no host form yet", event);
   case AGC_EVENT_OCCLUSION_DUMP:
      agc_fatal(function, "occlusion counter dump 0x%02x has no host form yet", event);
   case AGC_EVENT_TIMESTAMP:
      flush = host_before_gfx11() ? 0 : event_flush_ts(event);
      break;
   case AGC_EVENT_DB_META:
      if (agc_host_current() == AGC_HOST_GFX11)
         flush = V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT;
      form = agc_host_is_gfx12() ? AGC_EVENT_DROP : AGC_EVENT_FORWARD;
      break;
   case AGC_EVENT_CB_META:
      form = agc_host_is_gfx12() ? AGC_EVENT_DROP : AGC_EVENT_FORWARD;
      break;
   default:
      break;
   }

   uint32_t dwords = agc_event_write_host_dwords(compute, event);
   uint32_t *out = agc_record_alloc(cb, dwords);
   if (AGC_UNLIKELY(!out))
      return NULL;
   if (flush) {
      struct ac_cmdbuf cs = {.buf = out, .max_dw = dwords};
      ac_emit_cp_release_mem(&cs, host_gfx_level(), AMD_IP_GFX, flush, 0, EOP_DST_SEL_MEM, EOP_INT_SEL_NONE,
                             EOP_DATA_SEL_DISCARD, 0, 0, 0);
      agc_fill_nops(out + cs.cdw, dwords - cs.cdw);
      return out;
   }
   if (form == AGC_EVENT_DROP) {
      agc_fill_nops(out, dwords);
      return out;
   }
   uint32_t index = form == AGC_EVENT_PARTIAL_FLUSH ? V_461_CS_VS_PS_PARTIAL_FLUSH : V_461_OTHER;
   out[0] = PKT3(PKT3_EVENT_WRITE, 0, 0);
   out[1] = S_461_EVENT_TYPE(event) | S_461_EVENT_INDEX(index);
   return out;
}

uint32_t *
agc_emit_jump(struct agc_cb *cb, bool compute, uint8_t chain, uint8_t cache_policy, uint64_t target,
              uint32_t dwords, const char *function)
{
   if (!cb || chain > 1 || cache_policy > 3 || dwords > AGC_IB_DWORDS_MAX || (target & 3)) {
      agc_report(function, "invalid jump arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_JUMP);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_INDIRECT_BUFFER, 2, 0);
   put_address(out + 1, target);
   out[3] = S_3F3_IB_SIZE(dwords) | S_3F3_CHAIN(chain) | S_3F3_VALID(chain || compute) |
            S_3F3_CACHE_POLICY(agc_host_cache_policy(cache_policy));
   return out;
}

uint32_t *
agc_emit_cond_exec(struct agc_cb *cb, uint64_t address, uint32_t dwords, const char *function)
{
   if (!cb || !agc_address_valid(address, 4) || dwords > AGC_COND_EXEC_DWORDS_MAX) {
      agc_report(function, "invalid conditional execution arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_COND_EXEC);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_COND_EXEC, 3, 0);
   put_address(out + 1, address);
   out[3] = 0;
   out[4] = S_224_EXEC_COUNT(dwords);
   agc_record_add_reloc(out, AGC_RELOC_COND_EXEC);
   return out;
}

uint32_t *
agc_emit_rewind(struct agc_cb *cb, uint32_t state, const char *function)
{
   uint32_t *out = agc_packet_alloc(cb, AGC_DWORDS_REWIND, function);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_REWIND, 0, 0);
   out[1] = S_591_VALID(state);
   return out;
}

uint32_t *
agc_emit_marker_span(struct agc_cb *cb, const char *text, uint32_t length, bool pop,
                     const char *function)
{
   uint64_t payload = ((uint64_t)length + 4) / 4;
   if (!cb || (!text && length) || payload > AGC_MARKER_PAYLOAD_MAX) {
      agc_report(function, "null command buffer, null marker text, or marker text too long");
      return NULL;
   }
   uint32_t dwords = (uint32_t)payload + 1;
   uint32_t *out = agc_record_alloc(cb, dwords);
   if (AGC_UNLIKELY(!out))
      return NULL;
   agc_fill_nops(out, dwords);
   if (pop) {
      uint32_t *end = agc_record_alloc(cb, AGC_DWORDS_MARKER_POP);
      if (end)
         agc_fill_nops(end, AGC_DWORDS_MARKER_POP);
   }
   return out;
}

uint32_t *
agc_emit_marker(struct agc_cb *cb, const char *text, bool pop, const char *function)
{
   size_t length = text ? strlen(text) : 0;
   if (length >= AGC_MARKER_TEXT_MAX) {
      agc_report(function, "marker text too long");
      return NULL;
   }
   return agc_emit_marker_span(cb, text ? text : "", (uint32_t)length, pop, function);
}

#define AGC_SEMAPHORE_HOST_DWORDS 32u

uint32_t *
agc_emit_mem_semaphore(struct agc_cb *cb, uint64_t address, uint8_t mailbox, uint8_t signal_type,
                       uint8_t operation, const char *function)
{
   if (!cb || mailbox > 1 || signal_type > 1 ||
       (operation != AGC_SEMAPHORE_SIGNAL && operation != AGC_SEMAPHORE_WAIT) ||
       !agc_address_valid(address, 8)) {
      agc_report(function, "invalid memory semaphore arguments");
      return NULL;
   }
   uint32_t host[AGC_SEMAPHORE_HOST_DWORDS];
   struct ac_cmdbuf cs = {.buf = host, .max_dw = AGC_SEMAPHORE_HOST_DWORDS};
   if (operation == AGC_SEMAPHORE_WAIT) {
      ac_emit_cp_wait_mem(&cs, address, 0, 0xffffffff, WAIT_REG_MEM_NOT_EQUAL);
      ac_emit_cp_atomic_mem(&cs, V_1E1_GL2_OP_ATOMIC_SUB_64, V_1E1_SINGLE_PASS_ATOMIC, address, 1, 0);
   } else {
      ac_emit_cp_atomic_mem(&cs, signal_type ? V_1E1_GL2_OP_ATOMIC_SWAP_64 : V_1E1_GL2_OP_ATOMIC_ADD_64,
                            V_1E1_SINGLE_PASS_ATOMIC, address, 1, 0);
   }
   uint32_t *out = agc_emit_trampoline(cb, AGC_DWORDS_MEM_SEMAPHORE, host, cs.cdw, function);
   if (out && operation == AGC_SEMAPHORE_WAIT) {
      struct agc_reloc *reloc = agc_record_add_reloc(out, AGC_RELOC_LABEL_WAIT);
      reloc->flags = (uint16_t)(V_3C1_NOT_EQUAL_REFERENCE_VALUE | AGC_LABEL_WAIT_64);
      reloc->arg32 = 0;
      reloc->arg64[0] = address;
      reloc->arg64[1] = 0;
   } else if (out) {
      agc_record_note_write(out, AGC_LABEL_WRITE_ARG, address);
   }
   return out;
}

uint32_t *
agc_emit_flip(struct agc_cb *cb, uint32_t video_out, int32_t index, uint32_t mode, int64_t argument,
              const char *function)
{
   uint32_t *out = agc_emit_nops(cb, AGC_DWORDS_FLIP, function);
   if (AGC_UNLIKELY(!out))
      return NULL;
   struct agc_reloc *reloc = agc_record_add_reloc(out, AGC_RELOC_FLIP);
   reloc->arg32 = video_out;
   reloc->arg64[0] = (uint32_t)index | (uint64_t)mode << 32;
   reloc->arg64[1] = (uint64_t)argument;
   p_atomic_set(&agc_flip_last, (uintptr_t)out);
   p_atomic_inc(&agc_flip_records);
   return out;
}

uint32_t *
agc_emit_rendering_wait(struct agc_cb *cb, uint32_t video_out, int32_t index, const char *function)
{
   uint32_t *out = agc_emit_nops(cb, AGC_DWORDS_RENDERING_WAIT, function);
   if (AGC_UNLIKELY(!out))
      return NULL;
   struct agc_reloc *reloc = agc_record_add_reloc(out, AGC_RELOC_WAIT_RENDERING);
   reloc->arg32 = video_out;
   reloc->arg64[0] = (uint32_t)index;
   return out;
}

static inline bool
workload_stream_valid(uint32_t stream)
{
   return stream >= 1 && stream < 32;
}

uint32_t *
agc_emit_workloads_active(struct agc_cb *cb, uint32_t stream, const uint32_t *ids, uint32_t count,
                          const char *function)
{
   bool valid = cb && ids && count && count < 64 && workload_stream_valid(stream);
   uint64_t mask = 0;
   for (uint32_t i = 0; valid && i < count; i++) {
      valid = ids[i] < 64 && !(mask & (UINT64_C(1) << ids[i]));
      if (valid)
         mask |= UINT64_C(1) << ids[i];
   }
   if (!valid) {
      agc_report(function, "invalid workload stream or workload ids");
      return NULL;
   }
   return agc_emit_nops(cb, AGC_DWORDS_WORKLOADS_ACTIVE, function);
}

uint32_t *
agc_emit_workload_complete(struct agc_cb *cb, uint32_t stream, uint32_t id, const char *function)
{
   if (!cb || id >= 64 || !workload_stream_valid(stream)) {
      agc_report(function, "invalid workload stream or workload id");
      return NULL;
   }
   return agc_emit_nops(cb, AGC_DWORDS_WORKLOAD_COMPLETE, function);
}

uint32_t *
agc_emit_workload_stream_inactive(struct agc_cb *cb, uint32_t stream, const char *function)
{
   if (!cb || !workload_stream_valid(stream)) {
      agc_report(function, "invalid workload stream");
      return NULL;
   }
   return agc_emit_nops(cb, AGC_DWORDS_WORKLOAD_STREAM_INACTIVE, function);
}

uint32_t *AGC_SYSV
sceAgcCbSetShRegisterRangeDirect(struct agc_cb *cb, uint32_t offset, const uint32_t *values,
                                 uint32_t count)
{
   return agc_emit_set_register_range(cb, AGC_REG_BANK_SH, offset, values, count, __func__);
}

uint32_t *AGC_SYSV
sceAgcCbSetUcRegisterRangeDirect(struct agc_cb *cb, uint32_t offset, const uint32_t *values,
                                 uint32_t count)
{
   return agc_emit_set_register_range(cb, AGC_REG_BANK_UC, offset, values, count, __func__);
}

uint32_t AGC_SYSV
sceAgcCbSetShRegisterRangeDirectGetSize(uint32_t count)
{
   return agc_set_registers_bytes(count);
}

uint32_t AGC_SYSV
sceAgcCbSetUcRegisterRangeDirectGetSize(uint32_t count)
{
   return agc_set_registers_bytes(count);
}

uint32_t *AGC_SYSV
sceAgcCbSetShRegistersDirect(struct agc_cb *cb, const struct agc_register *registers, uint32_t count)
{
   return agc_emit_set_register_list(cb, AGC_REG_BANK_SH, registers, count, __func__);
}

uint32_t *AGC_SYSV
sceAgcCbSetUcRegistersDirect(struct agc_cb *cb, const struct agc_register *registers, uint32_t count)
{
   return agc_emit_set_register_list(cb, AGC_REG_BANK_UC, registers, count, __func__);
}

uint32_t AGC_SYSV
sceAgcCbSetShRegistersDirectGetSize(uint32_t count)
{
   return agc_set_register_pairs_bytes(count);
}

uint32_t AGC_SYSV
sceAgcCbSetUcRegistersDirectGetSize(uint32_t count)
{
   return agc_set_register_pairs_bytes(count);
}

uint32_t *AGC_SYSV
sceAgcCbNop(struct agc_cb *cb, uint32_t dwords)
{
   if (!cb || dwords == 0 || dwords > AGC_NOP_DWORDS_MAX) {
      agc_report(__func__, "invalid NOP size");
      return NULL;
   }
   return agc_emit_nops(cb, dwords, __func__);
}

uint32_t AGC_SYSV
sceAgcCbNopGetSize(uint32_t dwords)
{
   return dwords * 4;
}

uint32_t *AGC_SYSV
sceAgcCbDispatch(struct agc_cb *cb, uint32_t x, uint32_t y, uint32_t z, uint32_t modifier)
{
   if (!cb || !agc_dispatch_modifier_valid(modifier)) {
      agc_report(__func__, "invalid dispatch arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_DISPATCH);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_DISPATCH_DIRECT, 3, 0) | PKT3_SHADER_TYPE_S(1);
   out[1] = x;
   out[2] = y;
   out[3] = z;
   out[4] = agc_dispatch_initiator(modifier);
   return out;
}

uint32_t AGC_SYSV
sceAgcCbDispatchGetSize(void)
{
   return AGC_DWORDS_DISPATCH * 4;
}

uint32_t *AGC_SYSV
sceAgcCbCondWrite(struct agc_cb *cb, uint32_t function, uint32_t space, uint64_t address,
                  uint32_t value, uint64_t poll, uint32_t reference, uint32_t mask)
{
   if (!cb || function > V_451_GREATER_THAN_REFERENCE_VALUE ||
       (space != V_451_MEMORY && space != V_451_SCRATCH)) {
      agc_report(__func__, "invalid conditional write arguments");
      return NULL;
   }
   if (space == V_451_SCRATCH)
      agc_fatal(__func__, "conditional writes to the scratch space have no host form yet");
   if (!agc_address_valid(address, 4) || !agc_address_valid(poll, 4) || address > AGC_ADDRESS_MAX ||
       poll > AGC_ADDRESS_MAX) {
      agc_report(__func__, "invalid conditional write addresses");
      return NULL;
   }

   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_COND_WRITE);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_COND_WRITE, 7, 0);
   out[1] = S_451_FUNCTION(function) | S_451_POLL_SPACE(V_451_MEMORY) | S_451_WRITE_SPACE(space);
   put_address(out + 2, poll);
   out[4] = reference;
   out[5] = mask;
   put_address(out + 6, address);
   out[8] = value;
   return out;
}

uint32_t AGC_SYSV
sceAgcCbCondWriteGetSize(void)
{
   return AGC_DWORDS_COND_WRITE * 4;
}

static bool
release_action_known(uint32_t action)
{
   switch (action) {
   case V_028A90_CACHE_FLUSH_TS:
   case V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT:
   case V_028A90_BOTTOM_OF_PIPE_TS:
   case V_028A90_FLUSH_AND_INV_DB_DATA_TS:
   case V_028A90_FLUSH_AND_INV_CB_DATA_TS:
   case V_028A90_CS_DONE:
   case V_028A90_PS_DONE:
      return true;
   default:
      return false;
   }
}

static bool
release_action_is_shader_done(uint32_t action)
{
   return action == V_028A90_CS_DONE || action == V_028A90_PS_DONE;
}

static uint32_t
release_gcr(uint32_t gcr)
{
   const uint32_t kept = S_491_GLM_WB(1) | S_491_GLM_INV(1) | S_491_GLV_INV(1) | S_491_GL1_INV(1) |
                         S_491_GL2_INV(1) | S_491_GL2_WB(1) | S_491_SEQ(3);
   uint32_t guest = S_491_GCR_CNTL(gcr);
   uint32_t host = guest & kept;
   if ((guest & S_491_GL2_INV(1)) || (guest & S_491_GL2_DISCARD(1)))
      host |= S_491_GL2_INV(1) | S_491_GL2_WB(1);
   if (host & (S_491_GLM_WB(1) | S_491_GLM_INV(1)))
      host |= S_491_GLM_WB(1) | S_491_GLM_INV(1);
   if (agc_host_is_gfx12())
      host &= ~(S_491_GLM_WB(1) | S_491_GLM_INV(1) | S_491_GL1_INV(1));
   return host & ~S_491_SEQ(3) ? host : 0;
}

static void
release_check_host(uint32_t action, const char *function)
{
   if (!release_action_known(action))
      agc_fatal(function, "end-of-pipe action 0x%02x has no host form", action);
}

static uint32_t
release_event(uint32_t action, uint32_t host_gcr)
{
   if (release_action_is_shader_done(action) && host_gcr && !agc_host_is_gfx12())
      action = V_028A90_BOTTOM_OF_PIPE_TS;
   if (action == V_028A90_FLUSH_AND_INV_DB_DATA_TS && !host_before_gfx11() && !agc_host_is_gfx12())
      action = V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT;
   return S_491_EVENT_TYPE(action) |
          S_491_EVENT_INDEX(release_action_is_shader_done(action) ? V_491_SHADER_DONE
                                                                  : V_491_END_OF_PIPE);
}

uint32_t *AGC_SYSV
sceAgcCbReleaseMem(struct agc_cb *cb, uint8_t action, uint16_t gcr, uint8_t dst,
                   uint8_t cache_policy, uint64_t address, uint8_t select, uint64_t data,
                   uint16_t gds_offset, uint16_t gds_size, uint8_t interrupt, uint32_t context)
{
   bool valid = cb && action <= 0x3f && gcr <= 0xfff && dst <= 1 && cache_policy <= 3 &&
                (select <= 3 || select == 5) && interrupt <= 4 && context <= 0x7ffffff;
   if (valid) {
      if (interrupt == 4)
         valid = address == 0 && data == 0;
      else if (select == 5)
         valid = data == 0;
      else
         valid = gds_offset == 0 && gds_size <= 1;
   }
   if (valid && select != 0 && interrupt != 4 && address != 0)
      valid = (address & (select == 2 || select == 3 ? 7 : 3)) == 0;
   if (valid && select == 1)
      valid = (data >> 32) == 0;
   if (!valid) {
      agc_report(__func__, "invalid release arguments");
      return NULL;
   }
   if (select == 5)
      agc_fatal(__func__, "a release of GDS data has no host form");
   uint32_t host_gcr = release_gcr(gcr);
   release_check_host(action, __func__);

   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_RELEASE_MEM);
   if (AGC_UNLIKELY(!out))
      return NULL;
   uint32_t host_select = interrupt == 4 ? V_492_NONE : select;
   uint32_t host_interrupt = interrupt == 2 || interrupt == 3 ? V_492_SEND_DATA_AND_WRITE_CONFIRM
                                                              : V_492_NONE;
   out[0] = PKT3(PKT3_RELEASE_MEM, 6, 0);
   out[1] = release_event(action, host_gcr) | host_gcr |
            S_491_CACHE_POLICY(agc_host_cache_policy(cache_policy));
   out[2] = S_492_DST_SEL(dst) | S_492_INT_SEL(host_interrupt) | S_492_DATA_SEL(host_select);
   put_address(out + 3, address);
   put_address(out + 5, data);
   out[7] = 0;
   if (interrupt == 1 || interrupt == 2 || interrupt == 4) {
      struct agc_reloc *reloc = agc_record_add_reloc(out, AGC_RELOC_NOTIFY);
      reloc->flags = interrupt;
      reloc->arg32 = context;
   } else if (host_select != V_492_NONE && address) {
      agc_record_note_write(out, 3, address);
   }
   return out;
}

uint32_t AGC_SYSV
sceAgcCbQueueEndOfPipeActionGetSize(void)
{
   return AGC_DWORDS_RELEASE_MEM * 4;
}

uint32_t *AGC_SYSV
sceAgcCbBranch(struct agc_cb *cb, uint8_t mode, uint8_t function, uint64_t address, uint64_t mask,
               uint64_t reference, uint8_t policy1, uint64_t target1, uint32_t dwords1,
               uint8_t policy2, uint64_t target2, uint32_t dwords2)
{
   bool valid = cb && (mode == V_3F1_IF_THEN || mode == V_3F1_IF_THEN_ELSE) &&
                function <= V_3F1_GREATER_THAN_REFERENCE_VALUE && policy1 <= 3 && policy2 <= 3 &&
                dwords1 <= AGC_IB_DWORDS_MAX && dwords2 <= AGC_IB_DWORDS_MAX;
   if (valid && function != V_3F1_ALWAYS_PASS)
      valid = agc_address_valid(address, 8);
   if (valid && (target1 || dwords1))
      valid = agc_address_valid(target1, 4);
   if (valid && (target2 || dwords2))
      valid = agc_address_valid(target2, 4);
   if (!valid) {
      agc_report(__func__, "invalid branch arguments");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_DWORDS_BRANCH);
   if (AGC_UNLIKELY(!out))
      return NULL;
   out[0] = PKT3(PKT3_COND_INDIRECT_BUFFER, 12, 0);
   out[1] = S_3F1_MODE(mode) | S_3F1_FUNCTION(function);
   put_address(out + 2, address & ~UINT64_C(7));
   put_address(out + 4, mask);
   put_address(out + 6, reference);
   put_address(out + 8, target1);
   out[10] = S_3F10_IB_SIZE1(dwords1) | S_3F10_CACHE_POLICY1(agc_host_cache_policy(policy1));
   put_address(out + 11, target2);
   out[13] = S_3F13_IB_SIZE2(dwords2) | S_3F13_CACHE_POLICY2(agc_host_cache_policy(policy2));
   return out;
}

uint32_t AGC_SYSV
sceAgcCbBranchGetSize(void)
{
   return AGC_DWORDS_BRANCH * 4;
}

static bool
packet_pointer_valid(const uint32_t *command)
{
   return command && ((uintptr_t)command & 3) == 0;
}

static bool
packet_is(const uint32_t *command, uint32_t opcode, uint32_t count)
{
   return packet_pointer_valid(command) && (command[0] & ~UINT32_C(0xff)) == PKT3(opcode, count, 0);
}

static bool
packet_has_opcode(const uint32_t *command, uint32_t opcode)
{
   return packet_pointer_valid(command) && PKT_TYPE_G(command[0]) == 3 &&
          PKT3_IT_OPCODE_G(command[0]) == opcode;
}

static bool
packet_is_nop_pad(uint32_t header)
{
   return (header & AGC_NOP_PAD_MASK) == (PKT3_NOP_PAD & AGC_NOP_PAD_MASK);
}

static int32_t
patch_refused(const char *function, const char *reason)
{
   agc_report(function, "%s", reason);
   return AGC_PATCH_INVALID;
}

static int32_t
cond_exec_set_address(uint32_t *command, const uint32_t *address, const char *function)
{
   if (!packet_is(command, PKT3_COND_EXEC, 3) || !agc_address_valid((uintptr_t)address, 4))
      return patch_refused(function, "not a conditional execution packet, or an invalid address");
   put_address(command + 1, (uintptr_t)address);
   return 0;
}

static int32_t
cond_exec_set_end(uint32_t *command, const uint32_t *end, const char *function)
{
   if (!packet_is(command, PKT3_COND_EXEC, 3) || !agc_address_valid((uintptr_t)end, 4))
      return patch_refused(function, "not a conditional execution packet, or an invalid end");
   uintptr_t dwords = ((uintptr_t)end - (uintptr_t)(command + AGC_DWORDS_COND_EXEC)) / 4;
   if (dwords > AGC_COND_EXEC_DWORDS_MAX)
      return patch_refused(function, "conditional execution range is too large");
   command[4] = (command[4] & ~S_224_EXEC_COUNT(~0u)) | S_224_EXEC_COUNT((uint32_t)dwords);
   return 0;
}

int32_t AGC_SYSV
sceAgcCondExecPatchSetCommandAddress(uint32_t *command, const uint32_t *address)
{
   return cond_exec_set_address(command, address, __func__);
}

int32_t AGC_SYSV
sceAgcCondExecPatchSetEnd(uint32_t *command, const uint32_t *end)
{
   return cond_exec_set_end(command, end, __func__);
}

int32_t AGC_SYSV
sceAgcAsyncCondExecPatchSetCommandAddress(uint32_t *command, const uint32_t *address)
{
   return cond_exec_set_address(command, address, __func__);
}

int32_t AGC_SYSV
sceAgcAsyncCondExecPatchSetEnd(uint32_t *command, const uint32_t *end)
{
   return cond_exec_set_end(command, end, __func__);
}

static int32_t
dma_patch_address(uint32_t *command, uint32_t word, uint64_t address, const char *function)
{
   if (packet_is(command, PKT3_DMA_DATA, 5)) {
      put_address(command + word, address);
      return 0;
   }
   uint32_t dwords = 0;
   uint32_t *stub = agc_trampoline_stub(command, &dwords);
   if (!stub || dwords < AGC_DWORDS_DMA_DATA || !packet_is(stub, PKT3_DMA_DATA, 5))
      return patch_refused(function, "not a DMA packet");
   uint64_t first = stub[word] | (uint64_t)stub[word + 1] << 32;
   for (uint32_t at = 0; at + AGC_DWORDS_DMA_DATA <= dwords && packet_is(stub + at, PKT3_DMA_DATA, 5);
        at += AGC_DWORDS_DMA_DATA) {
      uint64_t old = stub[at + word] | (uint64_t)stub[at + word + 1] << 32;
      put_address(stub + at + word, address + (old - first));
   }
   agc_mem_flush_write_combining();
   return 0;
}

int32_t AGC_SYSV
sceAgcDmaDataPatchSetDstAddressOrOffset(uint32_t *command, uint64_t address)
{
   return dma_patch_address(command, 4, address, __func__);
}

int32_t AGC_SYSV
sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate(uint32_t *command, uint64_t address)
{
   return dma_patch_address(command, 2, address, __func__);
}

static int32_t
indirect_buffer_set_target(uint32_t *words, const uint32_t *target, uint32_t dwords,
                           const char *function)
{
   if (!agc_address_valid((uintptr_t)target, 4) || dwords > AGC_IB_DWORDS_MAX)
      return patch_refused(function, "invalid target or size");
   put_address(words, (uintptr_t)target);
   words[2] = (words[2] & ~S_3F3_IB_SIZE(~0u)) | S_3F3_IB_SIZE(dwords);
   return 0;
}

static int32_t
indirect_buffer_set_target_policy(uint32_t *words, uint64_t cache_policy, const uint32_t *target,
                                  uint32_t dwords, const char *function)
{
   if (cache_policy > 3)
      return patch_refused(function, "invalid cache policy");
   int32_t result = indirect_buffer_set_target(words, target, dwords, function);
   if (result == 0) {
      words[2] = (words[2] & ~S_3F3_CACHE_POLICY(~0u)) |
                 S_3F3_CACHE_POLICY(agc_host_cache_policy((uint32_t)cache_policy));
   }
   return result;
}

int32_t AGC_SYSV
sceAgcJumpPatchSetTarget(uint32_t *command, const uint32_t *target, uint32_t dwords)
{
   if (!packet_is(command, PKT3_INDIRECT_BUFFER, 2))
      return patch_refused(__func__, "not a jump packet");
   return indirect_buffer_set_target(command + 1, target, dwords, __func__);
}

int32_t AGC_SYSV
sceAgcJumpPatchSetTarget_0300(uint32_t *command, uint64_t cache_policy, const uint32_t *target,
                              uint32_t dwords)
{
   if (!packet_is(command, PKT3_INDIRECT_BUFFER, 2))
      return patch_refused(__func__, "not a jump packet");
   return indirect_buffer_set_target_policy(command + 1, cache_policy, target, dwords, __func__);
}

int32_t AGC_SYSV
sceAgcBranchPatchSetCompareAddress(uint32_t *command, const uint64_t *address)
{
   if (!packet_is(command, PKT3_COND_INDIRECT_BUFFER, 12) ||
       !agc_address_valid((uintptr_t)address, 8))
      return patch_refused(__func__, "not a branch packet, or an invalid address");
   put_address(command + 2, (uintptr_t)address);
   return 0;
}

int32_t AGC_SYSV
sceAgcBranchPatchSetThenTarget(uint32_t *command, const uint32_t *target, uint32_t dwords)
{
   if (!packet_is(command, PKT3_COND_INDIRECT_BUFFER, 12))
      return patch_refused(__func__, "not a branch packet");
   return indirect_buffer_set_target(command + 8, target, dwords, __func__);
}

int32_t AGC_SYSV
sceAgcBranchPatchSetElseTarget(uint32_t *command, const uint32_t *target, uint32_t dwords)
{
   if (!packet_is(command, PKT3_COND_INDIRECT_BUFFER, 12))
      return patch_refused(__func__, "not a branch packet");
   return indirect_buffer_set_target(command + 11, target, dwords, __func__);
}

int32_t AGC_SYSV
sceAgcBranchPatchSetThenTarget_0300(uint32_t *command, uint8_t cache_policy, const uint32_t *target,
                                    uint32_t dwords)
{
   if (!packet_is(command, PKT3_COND_INDIRECT_BUFFER, 12))
      return patch_refused(__func__, "not a branch packet");
   return indirect_buffer_set_target_policy(command + 8, cache_policy, target, dwords, __func__);
}

int32_t AGC_SYSV
sceAgcBranchPatchSetElseTarget_0300(uint32_t *command, uint8_t cache_policy, const uint32_t *target,
                                    uint32_t dwords)
{
   if (!packet_is(command, PKT3_COND_INDIRECT_BUFFER, 12))
      return patch_refused(__func__, "not a branch packet");
   return indirect_buffer_set_target_policy(command + 11, cache_policy, target, dwords, __func__);
}

int32_t AGC_SYSV
sceAgcQueueEndOfPipeActionPatchAddress(uint32_t *command, const uint64_t *label)
{
   if (!packet_is(command, PKT3_RELEASE_MEM, 6))
      return patch_refused(__func__, "not a release packet");
   uint32_t select = G_492_DATA_SEL(command[2]);
   if (!agc_address_valid((uintptr_t)label, select == 2 || select == 3 ? 8 : 4))
      return patch_refused(__func__, "null or misaligned label");
   put_address(command + 3, (uintptr_t)label);
   return 0;
}

int32_t AGC_SYSV
sceAgcQueueEndOfPipeActionPatchData(uint32_t *command, uint64_t data)
{
   if (!packet_is(command, PKT3_RELEASE_MEM, 6))
      return patch_refused(__func__, "not a release packet");
   if (G_492_DATA_SEL(command[2]) == V_492_SEND_32_BIT_LOW && (data >> 32))
      return patch_refused(__func__, "data exceeds 32 bits");
   put_address(command + 5, data);
   return 0;
}

int32_t AGC_SYSV
sceAgcQueueEndOfPipeActionPatchGcrCntl(uint32_t *command, uint16_t gcr)
{
   if (!packet_is(command, PKT3_RELEASE_MEM, 6) || gcr > 0xfff)
      return patch_refused(__func__, "not a release packet, or invalid cache controls");
   uint32_t host_gcr = release_gcr(gcr);
   uint32_t action = G_491_EVENT_TYPE(command[1]);
   release_check_host(action, __func__);
   command[1] = (command[1] & ~(S_491_GCR_CNTL(~0u) | S_491_EVENT_TYPE(~0u) | S_491_EVENT_INDEX(~0u))) |
                release_event(action, host_gcr) | host_gcr;
   return 0;
}

int32_t AGC_SYSV
sceAgcQueueEndOfPipeActionPatchType(uint32_t *command, uint8_t action)
{
   if (!packet_is(command, PKT3_RELEASE_MEM, 6) || action > 0x3f)
      return patch_refused(__func__, "not a release packet, or an invalid action");
   release_check_host(action, __func__);
   command[1] = (command[1] & ~(S_491_EVENT_TYPE(~0u) | S_491_EVENT_INDEX(~0u))) |
                release_event(action, command[1] & S_491_GCR_CNTL(~0u));
   return 0;
}

static int32_t
rewind_set_state(uint32_t *command, uint8_t state, const char *function)
{
   if (!packet_is(command, PKT3_REWIND, 0) || state > 1)
      return patch_refused(function, "not a rewind packet, or an invalid state");
   p_atomic_set(&command[1], S_591_VALID(state));
   return 0;
}

int32_t AGC_SYSV
sceAgcRewindPatchSetRewindState(uint32_t *command, uint8_t state)
{
   return rewind_set_state(command, state, __func__);
}

int32_t AGC_SYSV
sceAgcAsyncRewindPatchSetRewindState(uint32_t *command, uint8_t state)
{
   return rewind_set_state(command, state, __func__);
}

static uint32_t
wait_packet_bits(const uint32_t *command)
{
   if (packet_is(command, PKT3_WAIT_REG_MEM, 5))
      return 32;
   if (packet_is(command, PKT3_WAIT_REG_MEM64, 7))
      return 64;
   return 0;
}

int32_t AGC_SYSV
sceAgcWaitRegMemPatchAddress(uint32_t *command, const void *address)
{
   uint32_t bits = wait_packet_bits(command);
   if (!bits)
      return patch_refused(__func__, "not a wait packet");
   put_address(command + 2, (uintptr_t)address & ~(uintptr_t)(bits == 32 ? 3 : 7));
   return 0;
}

int32_t AGC_SYSV
sceAgcWaitRegMemPatchReference(uint32_t *command, uint64_t reference)
{
   uint32_t bits = wait_packet_bits(command);
   if (!bits)
      return patch_refused(__func__, "not a wait packet");
   command[4] = (uint32_t)reference;
   if (bits == 64)
      command[5] = (uint32_t)(reference >> 32);
   return 0;
}

int32_t AGC_SYSV
sceAgcWaitRegMemPatchMask(uint32_t *command, uint32_t mask)
{
   uint32_t bits = wait_packet_bits(command);
   if (!bits)
      return patch_refused(__func__, "not a wait packet");
   command[bits == 32 ? 5 : 6] = mask;
   return 0;
}

int32_t AGC_SYSV
sceAgcWaitRegMemPatchCompareFunction(uint32_t *command, uint8_t function)
{
   if (!wait_packet_bits(command) || function > V_3C1_GREATER_THAN_REFERENCE_VALUE)
      return patch_refused(__func__, "not a wait packet, or an invalid compare function");
   command[1] = (command[1] & ~S_3C1_FUNCTION(~0u)) | S_3C1_FUNCTION(function);
   return 0;
}

static int32_t
write_data_set_address(uint32_t *command, uint64_t address, const char *function)
{
   if (!packet_has_opcode(command, PKT3_WRITE_DATA))
      return patch_refused(function, "not a write data packet");
   put_address(command + 2, address);
   return 0;
}

static int32_t
write_data_set_cache_policy(uint32_t *command, uint8_t policy, const char *function)
{
   if (!packet_has_opcode(command, PKT3_WRITE_DATA) || policy > 3)
      return patch_refused(function, "not a write data packet, or an invalid cache policy");
   command[1] = (command[1] & ~S_371_CACHE_POLICY(~0u)) |
                S_371_CACHE_POLICY(agc_host_cache_policy(policy));
   return 0;
}

static int32_t
write_data_set_destination(uint32_t *command, uint32_t select, uint32_t engine, uint8_t dst,
                           const char *function)
{
   if (dst == 0 && G_371_WR_CONFIRM(command[1]))
      return patch_refused(function, "register writes do not support write confirmation");
   if (select != V_371_TC_L2 && select != V_371_MEMORY)
      agc_fatal(function, "write data destination %u has no host form", select);
   command[1] = (command[1] & ~(S_371_DST_SEL(~0u) | S_371_ENGINE_SEL(~0u))) |
                S_371_DST_SEL(select) | S_371_ENGINE_SEL(engine);
   return 0;
}

int32_t AGC_SYSV
sceAgcWriteDataPatchSetAddressOrOffset(uint32_t *command, uint64_t address)
{
   return write_data_set_address(command, address, __func__);
}

int32_t AGC_SYSV
sceAgcAsyncWriteDataPatchSetAddressOrOffset(uint32_t *command, uint64_t address)
{
   return write_data_set_address(command, address, __func__);
}

int32_t AGC_SYSV
sceAgcWriteDataPatchSetCachePolicy(uint32_t *command, uint8_t policy)
{
   return write_data_set_cache_policy(command, policy, __func__);
}

int32_t AGC_SYSV
sceAgcAsyncWriteDataPatchSetCachePolicy(uint32_t *command, uint8_t policy)
{
   return write_data_set_cache_policy(command, policy, __func__);
}

int32_t AGC_SYSV
sceAgcWriteDataPatchSetDst(uint32_t *command, uint8_t dst)
{
   if (!packet_has_opcode(command, PKT3_WRITE_DATA) || dst > 0x1f)
      return patch_refused(__func__, "not a write data packet, or an invalid destination");
   return write_data_set_destination(command, dst >> 1, dst & 1, dst, __func__);
}

int32_t AGC_SYSV
sceAgcAsyncWriteDataPatchSetDst(uint32_t *command, uint8_t dst)
{
   if (!packet_has_opcode(command, PKT3_WRITE_DATA) || dst > 0xf)
      return patch_refused(__func__, "not a write data packet, or an invalid destination");
   return write_data_set_destination(command, dst, 0, dst, __func__);
}

int32_t AGC_SYSV
sceAgcAcquireMemSetEngine(uint32_t *command, uint8_t engine)
{
   if (!packet_pointer_valid(command))
      return patch_refused(__func__, "null or misaligned packet");
   if (!packet_has_opcode(command, PKT3_ACQUIRE_MEM))
      return AGC_PATCH_NOT_ACQUIRE;
   command[1] = (command[1] & ~S_581A_ENGINE_SEL(~0u)) | S_581A_ENGINE_SEL(engine);
   return 0;
}

int32_t AGC_SYSV
sceAgcSetNop(uint32_t *command)
{
   if (!packet_pointer_valid(command) || PKT_TYPE_G(command[0]) != 3)
      return patch_refused(__func__, "not a type-3 packet");
   if (!packet_is_nop_pad(command[0]))
      agc_fill_nops(command, PKT_COUNT_G(command[0]) + 2);
   return 0;
}

int32_t AGC_SYSV
sceAgcSetPacketPredication(uint32_t *command, uint32_t predication)
{
   if (!command || predication > 1 || PKT_TYPE_G(command[0]) != 3)
      return patch_refused(__func__, "not a type-3 packet, or an invalid predication");
   if (!packet_is_nop_pad(command[0]))
      command[0] = (command[0] & ~UINT32_C(1)) | predication;
   return 0;
}

int32_t AGC_SYSV
sceAgcSetRangePredication(uint32_t *start, const uint32_t *end, uint32_t predication)
{
   if (!start || !end || start > end || predication > 1)
      return patch_refused(__func__, "invalid packet range or predication");
   uint32_t *packet = start;
   while (packet < end) {
      uint32_t header = packet[0];
      if (PKT_TYPE_G(header) == 2 || packet_is_nop_pad(header)) {
         packet++;
         continue;
      }
      uint32_t dwords = PKT_COUNT_G(header) + 2;
      if (PKT_TYPE_G(header) != 3 || (uintptr_t)(end - packet) < dwords)
         return patch_refused(__func__, "the range does not hold whole packets");
      packet[0] = (header & ~UINT32_C(1)) | predication;
      packet += dwords;
   }
   return 0;
}

uint32_t *AGC_SYSV
sceAgcCbMemsetExclusive(struct agc_cb *cb, uint64_t address, const uint32_t *pattern,
                        uint64_t bytes)
{
   uint64_t chunks = (bytes + AGC_MEMSET_CHUNK_BYTES - 1) / AGC_MEMSET_CHUNK_BYTES;
   if (!cb || !pattern || !agc_address_valid(address, AGC_MEMSET_ALIGNMENT) || bytes == 0 ||
       bytes % AGC_MEMSET_ALIGNMENT || chunks > AGC_MEMSET_CHUNKS_MAX) {
      agc_report(__func__, "invalid memset arguments");
      return NULL;
   }
   if (pattern[1] != pattern[0] || pattern[2] != pattern[0] || pattern[3] != pattern[0])
      agc_fatal(__func__, "a pattern of different dwords has no host form yet");

   uint32_t *out = agc_record_alloc(cb, (uint32_t)chunks * AGC_DWORDS_DMA_DATA);
   if (AGC_UNLIKELY(!out))
      return NULL;
   uint32_t *packet = out;
   for (uint64_t at = 0; at < bytes; at += AGC_MEMSET_CHUNK_BYTES) {
      uint64_t left = bytes - at;
      uint32_t count = left < AGC_MEMSET_CHUNK_BYTES ? (uint32_t)left : AGC_MEMSET_CHUNK_BYTES;
      packet[0] = PKT3(PKT3_DMA_DATA, 5, 0);
      packet[1] = S_501_SRC_SEL(V_501_DATA) | S_501_DST_SEL(V_501_DST_ADDR_USING_L2);
      packet[2] = pattern[0];
      packet[3] = 0;
      put_address(packet + 4, address + at);
      packet[6] = S_506_BYTE_COUNT(count);
      packet += AGC_DWORDS_DMA_DATA;
   }
   return out;
}

uint32_t AGC_SYSV
sceAgcGetPacketSize(const uint32_t *packet)
{
   if (!packet_pointer_valid(packet) || PKT_TYPE_G(packet[0]) != 3) {
      agc_report(__func__, "null or misaligned pointer, or not a PM4 packet");
      return 0;
   }
   return packet_is_nop_pad(packet[0]) ? 1 : PKT_COUNT_G(packet[0]) + 2;
}

int32_t AGC_SYSV
sceAgcGetDataPacketPayloadAddress_0090(uint32_t **address, uint32_t *packet, int32_t kind)
{
   if (!agc_address_valid((uintptr_t)address, 8) || !packet_pointer_valid(packet)) {
      agc_report(__func__, "null or misaligned argument");
      return AGC_ERROR_INVALID_ADDRESS;
   }
   if (kind)
      *address = packet + 2;
   else
      *address = PKT_COUNT_G(packet[0]) == PKT_COUNT_G(PKT3_NOP_PAD) ? NULL : packet + 1;
   return 0;
}

int32_t AGC_SYSV
sceAgcGetDataPacketPayloadRange(struct agc_memory_range *range, uint32_t *packet, int32_t kind)
{
   if (!agc_address_valid((uintptr_t)range, 8) || !packet_pointer_valid(packet)) {
      agc_report(__func__, "null or misaligned argument");
      return AGC_ERROR_INVALID_ADDRESS;
   }
   uint64_t bytes = (uint64_t)PKT_COUNT_G(packet[0]) * 4;
   if (kind) {
      range->base = packet + 2;
      range->size = bytes;
   } else if (PKT_COUNT_G(packet[0]) == PKT_COUNT_G(PKT3_NOP_PAD)) {
      range->base = NULL;
      range->size = 0;
   } else {
      range->base = packet + 1;
      range->size = bytes + 4;
   }
   return 0;
}
