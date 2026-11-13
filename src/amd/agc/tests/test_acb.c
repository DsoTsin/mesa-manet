#include <assert.h>
#include <setjmp.h>
#include <string.h>

#include "ac_barrier.h"
#include "ac_cmdbuf.h"
#include "ac_cmdbuf_cp.h"
#include "ac_debug.h"
#include "agc_cb_builders.h"
#include "agc_exports.h"
#include "sid.h"
#include "test.h"

#define ACB_GL2 2
#define ACB_GDS 3
#define ACB_PERFCOUNTER 4
#define ACB_IMMEDIATE 5
#define ACB_ATOMIC_RETURN 6
#define ACB_REF_CLOCK 9
#define DMA_GDS 1
#define DMA_IMMEDIATE 2
#define DMA_GL2 3

#define GCR_ICACHE 0x1u
#define GCR_METADATA 0x20u
#define GCR_SCALAR 0x80u
#define GCR_VECTOR 0x100u
#define GCR_GL1 0x200u
#define GCR_GL2_WRITEBACK_INVALIDATE 0xc000u
#define GCR_ORDER_012 0x10000u

static jmp_buf fatal_jump;
static char fatal_text[512];

static void
on_fatal(const char *message)
{
   strncpy(fatal_text, message, sizeof(fatal_text) - 1);
   longjmp(fatal_jump, 1);
}

static const struct {
   enum amd_gfx_level level;
   enum radeon_family family;
} host_chips[AGC_HOST_COUNT] = {
   [AGC_HOST_GFX103] = {GFX10_3, CHIP_NAVI21},
   [AGC_HOST_GFX11] = {GFX11, CHIP_NAVI31},
   [AGC_HOST_GFX115] = {GFX11_5, CHIP_STRIX_HALO},
   [AGC_HOST_GFX12] = {GFX12, CHIP_GFX1201},
};

struct reference {
   struct ac_cmdbuf cs;
   uint32_t words[96];
};

static void
reference_begin(struct reference *ref)
{
   memset(ref, 0, sizeof(*ref));
   ref->cs.buf = ref->words;
   ref->cs.max_dw = 96;
}

static void
reference_event(struct reference *ref, unsigned event)
{
   reference_begin(ref);
   ac_cmdbuf_begin(&ref->cs);
   ac_cmdbuf_event_write(event);
   CHECK(ac_cmdbuf_packets_added());
   ac_cmdbuf_end();
}

static void
expect_reference(const uint32_t *out, const struct reference *ref, unsigned dwords, int line)
{
   agc_test_check_eq(ref->cs.cdw, dwords, __FILE__, line, "length of the Mesa packets");
   for (unsigned i = 0; i < ref->cs.cdw; i++)
      agc_test_check_eq(out[i], ref->words[i], __FILE__, line, "word equals Mesa's");
}

static void
expect_words(const uint32_t *out, const uint32_t *expected, unsigned count, int line)
{
   for (unsigned i = 0; i < count; i++)
      agc_test_check_eq(out[i], expected[i], __FILE__, line, "packet word");
}

static void
expect_nops(const uint32_t *out, unsigned dwords, int line)
{
   unsigned at = 0;
   while (at < dwords) {
      uint32_t header = out[at];
      unsigned length = header == PKT3_NOP_PAD ? 1 : PKT_COUNT_G(header) + 2;
      agc_test_check(PKT_TYPE_G(header) == 3 && PKT3_IT_OPCODE_G(header) == PKT3_NOP &&
                        (header & 0xff) == 0,
                     __FILE__, line, "padding is a NOP packet");
      for (unsigned i = 1; i < length && at + i < dwords; i++)
         agc_test_check(out[at + i] == 0, __FILE__, line, "NOP body is zero");
      at += length;
   }
   agc_test_check(at == dwords, __FILE__, line, "padding ends with the footprint");
}

static void
expect_parse(enum agc_host host, uint32_t *ib, unsigned dwords, const char *name, bool nameless,
             int line)
{
   const char *path = "agc_test_acb_ib.txt";
   FILE *f = fopen(path, "w+");
   agc_test_check(f != NULL, __FILE__, line, "parser output file");
   if (!f)
      return;
   struct ac_ib_parser parser = {
      .f = f,
      .ib = ib,
      .num_dw = dwords,
      .gfx_level = host_chips[host].level,
      .family = host_chips[host].family,
      .ip_type = AMD_IP_COMPUTE,
   };
   ac_parse_ib(&parser, "agc");
   rewind(f);
   char text[1024];
   bool clean = true;
   bool named = false;
   while (fgets(text, sizeof(text), f)) {
      if (strstr(text, "UNKNOWN") && (!nameless || !strstr(text, "UNKNOWN(0x93)")))
         clean = false;
      if (strstr(text, "unrecognized") || strstr(text, "????????") ||
          strstr(text, "count in header too low") || strstr(text, "Unknown packet type"))
         clean = false;
      if (strstr(text, name))
         named = true;
   }
   agc_test_check(clean, __FILE__, line, "Mesa's parser reads every dword");
   agc_test_check(named, __FILE__, line, name);
   fclose(f);
   remove(path);
}

#define EXPECT_WORDS(out, ...)                                                                \
   do {                                                                                       \
      const uint32_t expected_words[] = {__VA_ARGS__};                                        \
      expect_words((out), expected_words, sizeof(expected_words) / sizeof(uint32_t),          \
                   __LINE__);                                                                 \
   } while (0)

#define EXPECT_REFERENCE(out, ref, dwords) expect_reference((out), &(ref), (dwords), __LINE__)
#define EXPECT_NOPS(out, dwords) expect_nops((out), (dwords), __LINE__)
#define EXPECT_PARSE(out, dwords, name) expect_parse(host, (out), (dwords), (name), false, __LINE__)
#define EXPECT_PARSE_NAMELESS(out, dwords, name) \
   expect_parse(host, (out), (dwords), (name), true, __LINE__)

static _Alignas(4096) uint8_t stub_pool[1 << 15];
static uint64_t stub_pool_used;

static bool
test_stub_alloc(uint64_t bytes, struct agc_stub_block *block)
{
   if (bytes > sizeof(stub_pool) - stub_pool_used)
      return false;
   block->cpu = stub_pool + stub_pool_used;
   block->va = (uint64_t)(uintptr_t)block->cpu;
   block->bytes = bytes;
   block->handle = NULL;
   stub_pool_used += bytes;
   return true;
}

static void
test_stub_release(struct agc_stub_block *block)
{
   (void)block;
}

static const struct agc_stub_source test_stubs = {
   .alloc = test_stub_alloc,
   .release = test_stub_release,
};

#define FOOTPRINT(call, bytes)                                                \
   do {                                                                       \
      before = cb->up;                                                        \
      out = (call);                                                           \
      CHECK(out != NULL && out == before);                                    \
      CHECK_EQ((uint64_t)(cb->up - before) * 4, (uint64_t)(bytes));           \
   } while (0)

#define REJECTED(call)                                 \
   do {                                                \
      uint32_t reports = agc_report_count();           \
      before = cb->up;                                 \
      CHECK((call) == NULL);                           \
      CHECK(cb->up == before);                         \
      CHECK_EQ(agc_report_count(), reports + 1);       \
   } while (0)

#define EXPECT_FATAL(statement, text)                  \
   do {                                                \
      fatal_text[0] = 0;                               \
      before = cb->up;                                 \
      if (setjmp(fatal_jump) == 0) {                   \
         statement;                                    \
         CHECK(!"expected a fatal error");             \
      } else {                                         \
         CHECK(strstr(fatal_text, text) != NULL);      \
      }                                                \
      CHECK(cb->up == before);                         \
   } while (0)

static uint32_t
policy_bits(enum agc_host host, uint32_t policy)
{
   return host == AGC_HOST_GFX12 ? 0 : policy;
}

static void
test_memory(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   struct reference ref;
   static const uint32_t data[3] = {0x11111111, 0x22222222, 0x33333333};
   const uint64_t address = UINT64_C(0x00000034567890a0);
   const uint64_t source = UINT64_C(0x0000004567890ac0);

   FOOTPRINT(sceAgcAcbDispatchIndirect(cb, address, 0x2001), sceAgcAcbDispatchIndirectGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DISPATCH_INDIRECT, 2, 0) | PKT3_SHADER_TYPE_S(1), (uint32_t)address,
                (uint32_t)(address >> 32),
                S_00B800_COMPUTE_SHADER_EN(1) | S_00B800_ORDER_MODE(1) | S_00B800_TUNNEL_ENABLE(1));
   EXPECT_PARSE(out, 4, "DISPATCH_INDIRECT");
   REJECTED(sceAgcAcbDispatchIndirect(cb, 0, 1));
   REJECTED(sceAgcAcbDispatchIndirect(cb, address + 2, 1));
   REJECTED(sceAgcAcbDispatchIndirect(cb, address, 0x4));
   REJECTED(sceAgcAcbDispatchIndirect(NULL, address, 1));

   FOOTPRINT(sceAgcAcbWriteData(cb, ACB_GL2, 0, address, data, 3, 0, 1), 28);
   reference_begin(&ref);
   ac_emit_cp_write_data(&ref.cs, V_371_MICRO_ENGINE, V_371_TC_L2, address, 3, data, false);
   EXPECT_REFERENCE(out, ref, 7);
   EXPECT_PARSE(out, 7, "WRITE_DATA");

   FOOTPRINT(sceAgcAcbWriteData(cb, ACB_GL2, 3, address + 1, data, 1, 1, 0), 20);
   EXPECT_WORDS(out, PKT3(PKT3_WRITE_DATA, 3, 0),
                S_371_DST_SEL(V_371_TC_L2) | S_371_ADDR_INCR(V_371_DO_NOT_INCREMENT_ADDRESS) |
                   S_371_CACHE_POLICY(policy_bits(host, V_371_BYPASS)),
                (uint32_t)(address + 1), (uint32_t)(address >> 32), data[0]);
   EXPECT_PARSE(out, 5, "WRITE_DATA");
   REJECTED(sceAgcAcbWriteData(cb, 0x10, 0, address, data, 1, 0, 0));
   REJECTED(sceAgcAcbWriteData(cb, ACB_GL2, 0, address, NULL, 1, 0, 0));
   REJECTED(sceAgcAcbWriteData(cb, ACB_GL2, 0, address, data, 1, 2, 0));
   EXPECT_FATAL(sceAgcAcbWriteData(cb, ACB_GDS, 0, address, data, 1, 0, 0), "no host form");

   FOOTPRINT(sceAgcAcbCopyData(cb, ACB_GL2, 0, address, ACB_GL2, 0, source, 1, 1),
             sceAgcAcbCopyDataGetSize());
   reference_begin(&ref);
   ac_emit_cp_copy_data(&ref.cs, COPY_DATA_TC_L2, COPY_DATA_TC_L2, source, address,
                        AC_CP_COPY_DATA_WR_CONFIRM | AC_CP_COPY_DATA_COUNT_SEL, false);
   EXPECT_REFERENCE(out, ref, 6);
   EXPECT_PARSE(out, 6, "COPY_DATA");

   FOOTPRINT(sceAgcAcbCopyData(cb, ACB_GL2, 0, address, ACB_REF_CLOCK, 0, 0, 1, 1), 24);
   reference_begin(&ref);
   ac_emit_cp_copy_data(&ref.cs, COPY_DATA_TIMESTAMP, COPY_DATA_TC_L2, 0, address,
                        AC_CP_COPY_DATA_WR_CONFIRM | AC_CP_COPY_DATA_COUNT_SEL, false);
   EXPECT_REFERENCE(out, ref, 6);
   EXPECT_PARSE(out, 6, "COPY_DATA");

   FOOTPRINT(sceAgcAcbCopyData(cb, ACB_GL2, 0, address, ACB_IMMEDIATE, 0, 0x77, 0, 0), 24);
   reference_begin(&ref);
   ac_emit_cp_copy_data(&ref.cs, COPY_DATA_IMM, COPY_DATA_TC_L2, 0x77, address, 0, false);
   EXPECT_REFERENCE(out, ref, 6);

   FOOTPRINT(sceAgcAcbCopyData(cb, ACB_GL2, 0, address, ACB_ATOMIC_RETURN, 0, 0, 1, 0), 24);
   EXPECT_WORDS(out, PKT3(PKT3_COPY_DATA, 4, 0),
                S_401_SRC_SEL(V_401_ATOMIC_RETURN_DATA) | S_401_DST_SEL(V_401_TC_L2) |
                   S_401_COUNT_SEL(V_401_64_BITS_OF_DATA),
                0, 0, (uint32_t)address, (uint32_t)(address >> 32));
   EXPECT_PARSE(out, 6, "COPY_DATA");
   REJECTED(sceAgcAcbCopyData(cb, ACB_GL2, 0, address, ACB_GL2, 4, source, 0, 0));
   EXPECT_FATAL(sceAgcAcbCopyData(cb, ACB_GL2, 0, address, ACB_GDS, 0, 0, 0, 0), "no host form");
   EXPECT_FATAL(sceAgcAcbCopyData(cb, ACB_PERFCOUNTER, 0, 0, ACB_GL2, 0, source, 0, 0),
                "no host form");

   FOOTPRINT(sceAgcAcbDmaData(cb, DMA_GL2, 1, address, DMA_GL2, 2, source, 1024, 1, 0),
             sceAgcAcbDmaDataGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DMA_DATA, 5, 0),
                S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2) | S_501_DST_SEL(V_501_DST_ADDR_USING_L2) |
                   S_501_SRC_CACHE_POLICY(policy_bits(host, V_501_NOA)) |
                   S_501_DST_CACHE_POLICY(policy_bits(host, V_501_STREAM)),
                (uint32_t)source, (uint32_t)(source >> 32), (uint32_t)address,
                (uint32_t)(address >> 32), S_506_BYTE_COUNT(1024) | S_506_RAW_WAIT(1));
   EXPECT_PARSE(out, 7, "DMA_DATA");

   FOOTPRINT(sceAgcAcbDmaData(cb, DMA_GL2, 0, address, DMA_IMMEDIATE, 0, 0x5a5a5a5a, 32, 0, 1), 28);
   EXPECT_WORDS(out, PKT3(PKT3_DMA_DATA, 5, 0),
                S_501_SRC_SEL(V_501_DATA) | S_501_DST_SEL(V_501_DST_ADDR_USING_L2), 0x5a5a5a5a, 0,
                (uint32_t)address, (uint32_t)(address >> 32),
                S_506_BYTE_COUNT(32) | S_506_DISABLE_WR_CONFIRM(1));
   EXPECT_PARSE(out, 7, "DMA_DATA");
   REJECTED(sceAgcAcbDmaData(cb, DMA_GL2, 0, address, DMA_GL2, 0, source, 8, 2, 0));
   EXPECT_FATAL(sceAgcAcbDmaData(cb, DMA_GL2, 0, address, 0x25, 0, 0, 8, 0, 0),
                "console registers");
   FOOTPRINT(sceAgcAcbDmaData(cb, DMA_GL2, 0, address, DMA_GDS, 2, 0x100, 8, 0, 0), 28);
   EXPECT_WORDS(out, PKT3(PKT3_DMA_DATA, 5, 0),
                S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2) | S_501_DST_SEL(V_501_DST_ADDR_USING_L2),
                (uint32_t)(AGC_RUNTIME_GDS_VA + 0x100), (uint32_t)((AGC_RUNTIME_GDS_VA + 0x100) >> 32),
                (uint32_t)address, (uint32_t)(address >> 32), S_506_BYTE_COUNT(8));
   EXPECT_PARSE(out, 7, "DMA_DATA");
   REJECTED(sceAgcAcbDmaData(cb, DMA_GL2, 0, address, DMA_GDS, 0, 0xfffc, 8, 0, 0));
   REJECTED(sceAgcAcbDmaData(cb, DMA_GL2, 0, address, 0x64, 0, 0, 8, 0, 0));

   FOOTPRINT(sceAgcAcbAtomicMem(cb, 111, 2, address, 0, 5, 9, 10), sceAgcAcbAtomicMemGetSize());
   reference_begin(&ref);
   ac_emit_cp_atomic_mem(&ref.cs, 111, V_1E1_WAIT_FOR_WRITE_CONFIRMATION, address, 5, 9);
   EXPECT_REFERENCE(out, ref, 9);
   EXPECT_PARSE(out, 9, "ATOMIC_MEM");
   REJECTED(sceAgcAcbAtomicMem(cb, 111, 2, address + 4, 0, 5, 9, 10));
   REJECTED(sceAgcAcbAtomicMem(cb, 15, 2, address, 4, 5, 9, 10));
}

static void
test_sync(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   struct reference ref;
   enum amd_gfx_level level = host_chips[host].level;
   const uint64_t address = UINT64_C(0x00000056789abcd0);
   const uint64_t wide = UINT64_C(0x1122334455667788);

   FOOTPRINT(sceAgcAcbWaitRegMem(cb, 0, 5, 0, address, 1, 0xffffffff, 64),
             sceAgcAcbWaitOnAddressGetSize(0));
   reference_begin(&ref);
   ac_emit_cp_wait_mem(&ref.cs, address, 1, 0xffffffff,
                       S_3C1_FUNCTION(V_3C1_GREATER_THAN_OR_EQUAL_REFERENCE_VALUE) |
                          S_3C1_OPERATION(V_3C1_WAIT_MEM_PREEMPTABLE));
   EXPECT_REFERENCE(out, ref, 7);
   EXPECT_NOPS(out + 7, 7);
   EXPECT_PARSE(out, 14, "WAIT_REG_MEM");

   FOOTPRINT(sceAgcAcbWaitRegMem(cb, 1, 3, 1, address, wide, ~wide, 160),
             sceAgcAcbWaitOnAddressGetSize(1));
   EXPECT_WORDS(out, PKT3(PKT3_WAIT_REG_MEM64, 7, 0),
                S_931_FUNCTION(V_931_EQUAL_TO_THE_REFERENCE_VALUE) |
                   S_931_MEM_SPACE(V_931_MEMORY_SPACE) |
                   S_931_OPERATION(V_931_WAIT_MEM_PREEMPTABLE) |
                   S_931_CACHE_POLICY(policy_bits(host, V_931_STREAM)),
                (uint32_t)address, (uint32_t)(address >> 32), (uint32_t)wide,
                (uint32_t)(wide >> 32), (uint32_t)~wide, (uint32_t)(~wide >> 32),
                S_938_POLL_INTERVAL(10));
   EXPECT_NOPS(out + 9, 7);
   EXPECT_PARSE_NAMELESS(out, 16, "REFERENCE_HI");
   CHECK_EQ(sceAgcAcbWaitOnAddressGetSize(2), 0);
   REJECTED(sceAgcAcbWaitRegMem(cb, 2, 3, 0, address, 0, 0, 0));
   REJECTED(sceAgcAcbWaitRegMem(cb, 0, 7, 0, address, 0, 0, 0));
   REJECTED(sceAgcAcbWaitRegMem(cb, 1, 3, 0, address + 4, 0, 0, 0));

   struct ac_barrier_state barrier = {.flags = AC_BARRIER_INV_SMEM | AC_BARRIER_INV_VMEM};
   bool context_roll = false;
   enum ac_rgp_flush_bits flush_bits = 0;
   FOOTPRINT(sceAgcAcbAcquireMem(cb, GCR_SCALAR | GCR_VECTOR | GCR_GL1, address, 4096, 400),
             sceAgcAcbAcquireMemGetSize());
   reference_begin(&ref);
   ac_emit_barrier(&ref.cs, level, AMD_IP_COMPUTE, &barrier, &context_roll, &flush_bits);
   EXPECT_REFERENCE(out, ref, 8);
   EXPECT_PARSE(out, 8, "ACQUIRE_MEM");

   barrier.flags = AC_BARRIER_INV_L2 | AC_BARRIER_INV_ICACHE;
   FOOTPRINT(sceAgcAcbAcquireMem(cb, GCR_METADATA | GCR_GL2_WRITEBACK_INVALIDATE | GCR_ICACHE, 0,
                                 UINT64_MAX, 400),
             32);
   reference_begin(&ref);
   ac_emit_barrier(&ref.cs, level, AMD_IP_COMPUTE, &barrier, &context_roll, &flush_bits);
   EXPECT_REFERENCE(out, ref, 8);
   EXPECT_PARSE(out, 8, "ACQUIRE_MEM");

   FOOTPRINT(sceAgcAcbAcquireMem(cb, GCR_ORDER_012, 0, 0, 400), 32);
   EXPECT_NOPS(out, 8);
   REJECTED(sceAgcAcbAcquireMem(cb, 0x80000, 0, 0, 400));
   CHECK_EQ(sceAgcAcbQueueEndOfShaderActionGetSize(), 32);

   FOOTPRINT(sceAgcAcbEventWrite(cb, V_028A90_CS_PARTIAL_FLUSH, 0),
             sceAgcAcbEventWriteGetSize(V_028A90_CS_PARTIAL_FLUSH));
   reference_event(&ref, V_028A90_CS_PARTIAL_FLUSH);
   EXPECT_REFERENCE(out, ref, 2);
   EXPECT_PARSE(out, 2, "EVENT_WRITE");

   static const uint8_t dropped[] = {
      V_028A90_PERFCOUNTER_START,
      V_028A90_PERFCOUNTER_STOP,
      V_028A90_PERFCOUNTER_SAMPLE,
   };
   for (unsigned i = 0; i < sizeof(dropped); i++) {
      FOOTPRINT(sceAgcAcbEventWrite(cb, dropped[i], 0), 8);
      EXPECT_NOPS(out, 2);
   }
   CHECK_EQ(sceAgcAcbEventWriteGetSize(V_028A90_PIXEL_PIPE_STAT_DUMP), 8);
   EXPECT_FATAL(sceAgcAcbEventWrite(cb, V_028A90_PS_PARTIAL_FLUSH, 0), "no host form");
   EXPECT_FATAL(sceAgcAcbEventWrite(cb, V_028A90_PIXEL_PIPE_STAT_DUMP, 0), "no host form");
   REJECTED(sceAgcAcbEventWrite(cb, V_028A90_CS_PARTIAL_FLUSH, address));
   REJECTED(sceAgcAcbEventWrite(cb, 0x40, 0));

   FOOTPRINT(sceAgcAcbPrimeUtcl2(cb, 0, 0, address, 4), sceAgcAcbPrimeUtcl2GetSize());
   EXPECT_NOPS(out, 5);
   EXPECT_PARSE(out, 5, "NOP");
}

static struct agc_reloc *
reloc_of(uint32_t *packet)
{
   return agc_record_reloc_at(agc_record_containing(packet), packet);
}

static void
test_flow(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   struct reference ref;
   enum amd_gfx_level level = host_chips[host].level;
   const uint64_t target = UINT64_C(0x000000789abcdef0);

   FOOTPRINT(sceAgcAcbJump(cb, 0, target, 100), sceAgcAcbJumpGetSize());
   reference_begin(&ref);
   ac_emit_cp_indirect_buffer(&ref.cs, target, 100,
                              AC_CP_INDIRECT_BUFFER_CHAIN | AC_CP_INDIRECT_BUFFER_VALID, false);
   EXPECT_REFERENCE(out, ref, 4);
   EXPECT_PARSE(out, 4, "INDIRECT_BUFFER");
   FOOTPRINT(sceAgcAcbJump(cb, 2, 0, 0), 16);
   EXPECT_WORDS(out, PKT3(PKT3_INDIRECT_BUFFER, 2, 0), 0, 0,
                S_3F3_CHAIN(1) | S_3F3_VALID(1) | S_3F3_CACHE_POLICY(policy_bits(host, 2)));
   REJECTED(sceAgcAcbJump(cb, 4, target, 1));
   REJECTED(sceAgcAcbJump(cb, 0, target + 1, 1));
   REJECTED(sceAgcAcbJump(cb, 0, target, 0x100000));

   FOOTPRINT(sceAgcAcbCondExec(cb, target, 0x3fff), sceAgcAcbCondExecGetSize());
   reference_begin(&ref);
   ac_emit_cp_cond_exec(&ref.cs, level, target, 0x3fff);
   EXPECT_REFERENCE(out, ref, 5);
   EXPECT_PARSE(out, 5, "COND_EXEC");
   REJECTED(sceAgcAcbCondExec(cb, 0, 1));
   REJECTED(sceAgcAcbCondExec(cb, target, 0x4000));

   FOOTPRINT(sceAgcAcbRewind(cb, 1, 1), sceAgcAcbRewindGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_REWIND, 0, 0), S_591_VALID(1));
   EXPECT_PARSE(out, 2, "REWIND");
   FOOTPRINT(sceAgcAcbRewind(cb, 0, 0), 8);
   EXPECT_WORDS(out, PKT3(PKT3_REWIND, 0, 0), 0);
}

static void
test_record(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   struct reference ref;
   const uint64_t gds = AGC_RUNTIME_GDS_VA;
   const uint64_t address = UINT64_C(0x0000004400002000);

   FOOTPRINT(sceAgcAcbPushMarker(cb, "dispatch", 0), 16);
   EXPECT_NOPS(out, 4);
   EXPECT_PARSE(out, 4, "NOP");
   FOOTPRINT(sceAgcAcbPushMarkerSpan(cb, "dispatch", 2, 0), 8);
   EXPECT_NOPS(out, 2);
   FOOTPRINT(sceAgcAcbSetMarker(cb, "abc", 0), 16);
   EXPECT_NOPS(out, 4);
   FOOTPRINT(sceAgcAcbSetMarkerSpan(cb, "abcdefgh", 8, 0), 24);
   EXPECT_NOPS(out, 6);
   FOOTPRINT(sceAgcAcbPopMarker(cb), 8);
   EXPECT_NOPS(out, 2);
   REJECTED(sceAgcAcbPushMarkerSpan(cb, NULL, 4, 0));
   REJECTED(sceAgcAcbSetMarkerSpan(cb, "abc", 0x10000, 0));

   FOOTPRINT(sceAgcAcbSetFlip(cb, 3, 1, 0, 42), 24);
   EXPECT_NOPS(out, 6);
   struct agc_reloc *reloc = reloc_of(out);
   CHECK(reloc != NULL && reloc->kind == AGC_RELOC_FLIP);
   if (reloc) {
      CHECK_EQ(reloc->arg32, 3);
      CHECK_EQ(reloc->arg64[0], 1);
      CHECK_EQ(reloc->arg64[1], 42);
   }
   FOOTPRINT(sceAgcAcbWaitUntilSafeForRendering(cb, 3, 1), 16);
   EXPECT_NOPS(out, 4);
   reloc = reloc_of(out);
   CHECK(reloc != NULL && reloc->kind == AGC_RELOC_WAIT_RENDERING);
   if (reloc) {
      CHECK_EQ(reloc->arg32, 3);
      CHECK_EQ(reloc->arg64[0], 1);
   }

   const uint32_t ids[] = {1, 2};
   FOOTPRINT(sceAgcAcbSetWorkloadsActive(cb, 2, ids, 2), 72);
   EXPECT_NOPS(out, 18);
   FOOTPRINT(sceAgcAcbSetWorkloadComplete(cb, 2, 1), 48);
   EXPECT_NOPS(out, 12);
   FOOTPRINT(sceAgcAcbSetWorkloadStreamInactive(cb, 2), 36);
   EXPECT_NOPS(out, 9);
   EXPECT_PARSE(out, 9, "NOP");
   REJECTED(sceAgcAcbSetWorkloadsActive(cb, 2, ids, 0));
   REJECTED(sceAgcAcbSetWorkloadComplete(cb, 0, 1));
   REJECTED(sceAgcAcbSetWorkloadStreamInactive(cb, 40));

   FOOTPRINT(sceAgcAcbResetQueue(cb, 0x1c2, 0x12345678), 12);
   EXPECT_NOPS(out, 3);
   REJECTED(sceAgcAcbResetQueue(cb, 0x1000, 0));
   REJECTED(sceAgcAcbResetQueue(NULL, 2, 0));

   CHECK_EQ(sceAgcAcbAtomicGdsGetSize(), 44);
   FOOTPRINT(sceAgcAcbAtomicGds_0900(cb, AGC_GDS_OP_SUB | AGC_GDS_OP_RETURN, 0, 0, 0, 0x40, 0x40,
                                     2, 0x10, 3, 0),
             sceAgcAcbAtomicGdsGetSize());
   reference_begin(&ref);
   ac_emit_cp_atomic_mem(&ref.cs, V_1E1_GL2_OP_ATOMIC_SUB_RTN_32, V_1E1_SINGLE_PASS_ATOMIC,
                         gds + 0x50, 3, 0);
   EXPECT_REFERENCE(out, ref, 9);
   EXPECT_NOPS(out + 9, 2);
   EXPECT_PARSE(out, 11, "ATOMIC_MEM");
   FOOTPRINT(sceAgcAcbAtomicGds_0900(cb, AGC_GDS_OP_OR, 1, 0, 0, 0, 0, 0, 0x20, 0x80, 0), 44);
   reference_begin(&ref);
   ac_emit_cp_atomic_mem(&ref.cs, V_1E1_GL2_OP_ATOMIC_OR_32, V_1E1_WAIT_FOR_WRITE_CONFIRMATION,
                         gds + 0x20, 0x80, 0);
   EXPECT_REFERENCE(out, ref, 9);
   EXPECT_FATAL(sceAgcAcbAtomicGds_0900(cb, 0x0c, 0, 0, 0, 0, 0, 0, 0, 0, 0), "no host form");
   REJECTED(sceAgcAcbAtomicGds_0900(cb, AGC_GDS_OP_OR, 3, 0, 0, 0, 0, 0, 0, 0, 0));

   FOOTPRINT(sceAgcAcbCopyData(cb, ACB_GL2, 0, address, 7, 0, 0, 0, 0), 24);
   EXPECT_WORDS(out, PKT3(PKT3_COPY_DATA, 4, 0),
                S_401_SRC_SEL(V_401_ATOMIC_RETURN_DATA) | S_401_DST_SEL(V_401_TC_L2), 0, 0,
                (uint32_t)address, (uint32_t)(address >> 32));
   EXPECT_PARSE(out, 6, "COPY_DATA");

   FOOTPRINT(sceAgcAcbMemSemaphore(cb, address, 1, 0, 7), 16);
   CHECK_EQ(out[0], PKT3(PKT3_INDIRECT_BUFFER, 2, 0));
   CHECK_EQ(out[3], S_3F3_IB_SIZE(16) | S_3F3_VALID(1));
   reference_begin(&ref);
   ac_emit_cp_wait_mem(&ref.cs, address, 0, 0xffffffff, WAIT_REG_MEM_NOT_EQUAL);
   ac_emit_cp_atomic_mem(&ref.cs, V_1E1_GL2_OP_ATOMIC_SUB_64, V_1E1_SINGLE_PASS_ATOMIC, address, 1, 0);
   EXPECT_REFERENCE((const uint32_t *)(uintptr_t)(out[1] | (uint64_t)out[2] << 32), ref, 16);
   REJECTED(sceAgcAcbMemSemaphore(cb, address, 0, 0, 8));
   REJECTED(sceAgcAcbMemSemaphore(cb, address + 4, 0, 0, 6));
}

static void
test_host(enum agc_host host)
{
   static _Alignas(64) uint32_t buffer[2048];
   struct agc_cb cb;
   CHECK(agc_host_select(host));
   agc_cb_init(&cb, buffer, sizeof(buffer), NULL, NULL);

   test_memory(host, &cb);
   test_sync(host, &cb);
   test_flow(host, &cb);
   test_record(host, &cb);
   expect_parse(host, buffer, (unsigned)(cb.up - buffer), "DISPATCH_INDIRECT", true, __LINE__);

   _Alignas(64) uint32_t tiny[3];
   struct agc_cb small;
   agc_cb_init(&small, tiny, sizeof(tiny), NULL, NULL);
   CHECK(sceAgcAcbDispatchIndirect(&small, 0x1000, 1) == NULL);
   CHECK(sceAgcAcbAcquireMem(&small, GCR_GL1, 0, 0, 400) == NULL);
   CHECK(small.up == tiny);
}

int
main(void)
{
   agc_set_fatal_handler(on_fatal);
   agc_record_set_stub_source(&test_stubs);
   for (unsigned host = 0; host < AGC_HOST_COUNT; host++)
      test_host(host);
   agc_record_reset_all();
   return TEST_RESULT();
}
