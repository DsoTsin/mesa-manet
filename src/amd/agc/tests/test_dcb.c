#include <assert.h>
#include <setjmp.h>
#include <string.h>

#include "ac_barrier.h"
#include "ac_cmdbuf.h"
#include "ac_cmdbuf_cp.h"
#include "ac_debug.h"
#include "ac_gpu_info.h"
#include "agc_cb_builders.h"
#include "agc_driver.h"
#include "agc_exports.h"
#include "sid.h"
#include "test.h"

#define GS_USER_DATA(n) (((R_00B230_SPI_SHADER_USER_DATA_GS_0 - SI_SH_REG_OFFSET) >> 2) + (n))
#define HS_USER_DATA(n) (((R_00B430_SPI_SHADER_USER_DATA_HS_0 - SI_SH_REG_OFFSET) >> 2) + (n))

#define DRAW_VERTEX 0x1u
#define DRAW_START_INDEX 0x2u
#define DRAW_INSTANCE 0x4u
#define DRAW_DRAW_INDEX 0x8u
#define DRAW_USER_VGPRS 0x10u
#define DRAW_FUSE 0x100u

#define DST_GL2_ME 4
#define DST_GL2_PFP 5
#define DST_GDS_ME 6
#define SRC_IMMEDIATE_PFP 11
#define SRC_ATOMIC_RETURN_ME 12
#define SRC_ATOMIC_RETURN_PFP 13
#define SRC_REF_CLOCK_ME 18
#define DST_PERFCOUNTER_ME 8
#define DMA_GDS 1
#define DMA_IMMEDIATE 2
#define DMA_NOWHERE 2
#define DMA_GL2 3
#define DMA_ATOMIC_RETURN 0x14

#define GCR_ICACHE 0x1u
#define GCR_GL1_RANGE 0x208u
#define GCR_METADATA 0x20u
#define GCR_SCALAR 0x80u
#define GCR_VECTOR 0x100u
#define GCR_GL1 0x200u
#define GCR_GL2_RANGE 0x1000u
#define GCR_GL2_WRITEBACK 0x8000u
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
reference_index_type(struct reference *ref, const struct radeon_info *info, uint32_t value)
{
   reference_begin(ref);
   ac_cmdbuf_begin(&ref->cs);
   ac_cmdbuf_set_ucfg_reg_idx(info, R_03090C_VGT_INDEX_TYPE, 2, value);
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
expect_parse(enum agc_host host, enum amd_ip_type ip, uint32_t *ib, unsigned dwords,
             const char *name, bool nameless, int line)
{
   const char *path = "agc_test_dcb_ib.txt";
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
      .ip_type = ip,
   };
   ac_parse_ib(&parser, "agc");
   rewind(f);
   char text[1024];
   bool clean = true;
   bool named = false;
   while (fgets(text, sizeof(text), f)) {
      if (strstr(text, "UNKNOWN")) {
         if (!nameless || (!strstr(text, "UNKNOWN(0x93)") && !strstr(text, "UNKNOWN(0x91)")))
            clean = false;
      }
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
#define EXPECT_PARSE(out, dwords, name) \
   expect_parse(host, AMD_IP_GFX, (out), (dwords), (name), false, __LINE__)
#define EXPECT_PARSE_NAMELESS(out, dwords, name) \
   expect_parse(host, AMD_IP_GFX, (out), (dwords), (name), true, __LINE__)

static _Alignas(4096) uint8_t stub_pool[1 << 16];
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

static void
expect_trampoline(const uint32_t *out, unsigned footprint, const struct reference *ref, int line)
{
   unsigned padded = (ref->cs.cdw + 7) & ~7u;
   agc_test_check_eq(out[0], PKT3(PKT3_INDIRECT_BUFFER, 2, 0), __FILE__, line, "call header");
   agc_test_check_eq(out[3], S_3F3_IB_SIZE(padded) | S_3F3_VALID(1), __FILE__, line, "call size");
   uint64_t va = out[1] | (uint64_t)out[2] << 32;
   agc_test_check(va % 32 == 0, __FILE__, line, "host sequence is 32-byte aligned");
   expect_nops(out + 4, footprint - 4, line);
   const uint32_t *stub = (const uint32_t *)(uintptr_t)va;
   expect_reference(stub, ref, ref->cs.cdw, line);
   expect_nops(stub + ref->cs.cdw, padded - ref->cs.cdw, line);
}

#define EXPECT_TRAMPOLINE(out, footprint, ref) expect_trampoline((out), (footprint), &(ref), __LINE__)

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

static uint64_t
draw_modifier(unsigned stage, unsigned flags, unsigned vertex, unsigned start, unsigned instance,
              unsigned draw)
{
   return ((uint64_t)stage << 29) | flags | ((uint64_t)vertex << 9) | ((uint64_t)start << 14) |
          ((uint64_t)instance << 19) | ((uint64_t)draw << 24);
}

static uint32_t
policy_bits(enum agc_host host, uint32_t policy)
{
   return host == AGC_HOST_GFX12 ? 0 : policy;
}

static void
test_draws(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   const uint64_t fused = draw_modifier(2, DRAW_FUSE, 0, 0, 0, 0);
   const uint64_t address = UINT64_C(0x0000001234567890);
   const uint64_t count_address = UINT64_C(0x0000000120001000);

   FOOTPRINT(sceAgcDcbDrawIndexAuto(cb, 36, fused), sceAgcDcbDrawIndexAutoGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DRAW_INDEX_AUTO, 1, 0), 36, V_0287F0_DI_SRC_SEL_AUTO_INDEX);
   EXPECT_PARSE(out, 3, "DRAW_INDEX_AUTO");

   FOOTPRINT(sceAgcDcbDrawIndex(cb, 6, address, fused), sceAgcDcbDrawIndexGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DRAW_INDEX_2, 4, 0), 6, (uint32_t)address,
                (uint32_t)(address >> 32), 6, V_0287F0_DI_SRC_SEL_DMA);
   EXPECT_PARSE(out, 6, "DRAW_INDEX_2");
   FOOTPRINT(sceAgcDcbDrawIndex(cb, 0, address, fused), 24);
   CHECK_EQ(out[1], 1);
   CHECK_EQ(out[4], 0);

   FOOTPRINT(sceAgcDcbDrawIndexOffset(cb, 12, 30, fused), sceAgcDcbDrawIndexOffsetGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DRAW_INDEX_OFFSET_2, 3, 0), 42, 12, 30, V_0287F0_DI_SRC_SEL_DMA);
   EXPECT_PARSE(out, 5, "DRAW_INDEX_OFFSET_2");
   FOOTPRINT(sceAgcDcbDrawIndexOffset(cb, 0xfffffff0, 0x20, fused), 20);
   CHECK_EQ(out[1], UINT32_MAX);
   FOOTPRINT(sceAgcDcbDrawIndexOffset(cb, 0, 0, fused), 20);
   CHECK_EQ(out[1], 1);

   uint64_t offsets = draw_modifier(2, DRAW_VERTEX | DRAW_INSTANCE, 3, 0, 4, 0);
   FOOTPRINT(sceAgcDcbDrawIndirect(cb, 16, offsets), sceAgcDcbDrawIndirectGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DRAW_INDIRECT, 3, 0), 16, S_242_START_VTX_LOC(GS_USER_DATA(3)),
                S_243_START_INST_LOC(GS_USER_DATA(4)), V_0287F0_DI_SRC_SEL_AUTO_INDEX);
   EXPECT_PARSE(out, 5, "DRAW_INDIRECT");
   FOOTPRINT(sceAgcDcbDrawIndirect(cb, 0, fused), 20);
   EXPECT_WORDS(out, PKT3(PKT3_DRAW_INDIRECT, 3, 0), 0, 0, 0, V_0287F0_DI_SRC_SEL_AUTO_INDEX);

   uint64_t hull = draw_modifier(3, DRAW_VERTEX | DRAW_START_INDEX | DRAW_INSTANCE, 1, 2, 5, 0);
   FOOTPRINT(sceAgcDcbDrawIndexIndirect(cb, 32, hull), sceAgcDcbDrawIndexIndirectGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DRAW_INDEX_INDIRECT, 3, 0), 32,
                S_252_BASE_VTX_LOC(HS_USER_DATA(1)) | S_252_START_INDX_LOC(HS_USER_DATA(2)),
                S_253_START_INST_LOC(HS_USER_DATA(5)) | S_253_START_INDX_ENABLE(1),
                V_0287F0_DI_SRC_SEL_DMA);
   EXPECT_PARSE(out, 5, "DRAW_INDEX_INDIRECT");
   FOOTPRINT(sceAgcDcbDrawIndexIndirect(cb, 8, offsets), 20);
   EXPECT_WORDS(out, PKT3(PKT3_DRAW_INDEX_INDIRECT, 3, 0), 8, S_252_BASE_VTX_LOC(GS_USER_DATA(3)),
                S_253_START_INST_LOC(GS_USER_DATA(4)), V_0287F0_DI_SRC_SEL_DMA);

   uint64_t multi = draw_modifier(
      2, DRAW_VERTEX | DRAW_INSTANCE | DRAW_DRAW_INDEX | DRAW_USER_VGPRS, 0, 0, 1, 2);
   FOOTPRINT(sceAgcDcbDrawIndirectMulti(cb, 0, 1, 8, count_address, 16, multi),
             sceAgcDcbDrawIndirectMultiGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DRAW_INDIRECT_MULTI, 8, 0), 0, S_2C2_START_VTX_LOC(GS_USER_DATA(0)),
                S_2C3_START_INST_LOC(GS_USER_DATA(1)),
                S_2C4_DRAW_INDEX_LOC(GS_USER_DATA(2)) | S_2C4_USE_VGPRS(1) |
                   S_2C4_COUNT_INDIRECT_ENABLE(1) | S_2C4_DRAW_INDEX_ENABLE(1),
                8, (uint32_t)count_address, (uint32_t)(count_address >> 32), 16,
                V_0287F0_DI_SRC_SEL_AUTO_INDEX);
   EXPECT_NOPS(out + 10, 6);
   EXPECT_PARSE(out, 16, "DRAW_INDIRECT_MULTI");

   uint64_t indexed =
      draw_modifier(5, DRAW_VERTEX | DRAW_START_INDEX | DRAW_DRAW_INDEX, 4, 6, 0, 7);
   FOOTPRINT(sceAgcDcbDrawIndexIndirectMulti(cb, 4, 0, 5, 0, 20, indexed),
             sceAgcDcbDrawIndexIndirectMultiGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DRAW_INDEX_INDIRECT_MULTI, 8, 0), 4,
                S_382_BASE_VTX_LOC(HS_USER_DATA(4)) | S_382_START_INDX_LOC(HS_USER_DATA(6)), 0,
                S_384_DRAW_INDEX_LOC(HS_USER_DATA(7)) | S_384_START_INDX_ENABLE(1) |
                   S_384_DRAW_INDEX_ENABLE(1),
                5, 0, 0, 20, V_0287F0_DI_SRC_SEL_DMA);
   EXPECT_NOPS(out + 10, 6);
   EXPECT_PARSE(out, 16, "DRAW_INDEX_INDIRECT_MULTI");

   REJECTED(sceAgcDcbDrawIndexAuto(cb, 3, UINT64_C(1) << 33));
   REJECTED(sceAgcDcbDrawIndexAuto(NULL, 3, fused));
   REJECTED(sceAgcDcbDrawIndex(cb, 3, 0, fused));
   REJECTED(sceAgcDcbDrawIndexOffset(cb, 0, 3, UINT64_C(1) << 40));
   REJECTED(sceAgcDcbDrawIndirect(cb, 2, fused));
   REJECTED(sceAgcDcbDrawIndexIndirect(cb, 6, fused));
   REJECTED(sceAgcDcbDrawIndirectMulti(cb, 0, 2, 1, count_address, 16, fused));
   REJECTED(sceAgcDcbDrawIndirectMulti(cb, 0, 0, 1, 0, 12, fused));
   REJECTED(sceAgcDcbDrawIndirectMulti(cb, 0, 0, 1, 0, 18, fused));
   REJECTED(sceAgcDcbDrawIndexIndirectMulti(cb, 0, 0, 1, 0, 16, fused));
   REJECTED(sceAgcDcbDrawIndirectMulti(cb, 0, 0, 1, count_address, 16, fused));
   REJECTED(sceAgcDcbDrawIndirectMulti(cb, 0, 1, 1, 0, 16, fused));
   REJECTED(sceAgcDcbDrawIndirectMulti(cb, 0, 1, 1, count_address + 2, 16, fused));
   REJECTED(sceAgcDcbDrawIndexIndirectMulti(cb, 2, 0, 1, 0, 20, fused));
   CHECK_EQ(sceAgcDcbDrawIndexMultiInstancedGetSize(), 60);
}

static void
test_state(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   struct reference ref;
   struct radeon_info info = {.gfx_level = host_chips[host].level};
   const uint64_t address = UINT64_C(0x00000023456789a0);

   FOOTPRINT(sceAgcDcbSetIndexSize(cb, 1, 2), sceAgcDcbSetIndexSizeGetSize());
   reference_index_type(&ref, &info, S_028A7C_INDEX_TYPE(V_028A7C_VGT_INDEX_32));
   EXPECT_REFERENCE(out, ref, 3);
   EXPECT_PARSE(out, 3, "SET_UCONFIG_REG_INDEX");

   FOOTPRINT(sceAgcDcbSetIndexSize_0600(cb, 2, 0, 1), 12);
   reference_index_type(&ref, &info,
                        S_028A7C_INDEX_TYPE(V_028A7C_VGT_INDEX_8) |
                           S_028A7C_DISABLE_INSTANCE_PACKING(1));
   EXPECT_REFERENCE(out, ref, 3);
   REJECTED(sceAgcDcbSetIndexSize(cb, 3, 0));
   REJECTED(sceAgcDcbSetIndexSize(cb, 0, 4));
   REJECTED(sceAgcDcbSetIndexSize_0600(cb, 0, 0, 2));

   FOOTPRINT(sceAgcDcbSetIndexCount(cb, 1234), sceAgcDcbSetIndexCountGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_INDEX_BUFFER_SIZE, 0, 0), 1234);
   EXPECT_PARSE(out, 2, "INDEX_BUFFER_SIZE");
   REJECTED(sceAgcDcbSetIndexCount(NULL, 1));

   FOOTPRINT(sceAgcDcbSetIndexBuffer(cb, address), sceAgcDcbSetIndexBufferGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_INDEX_BASE, 1, 0), (uint32_t)address, (uint32_t)(address >> 32));
   EXPECT_PARSE(out, 3, "INDEX_BASE");
   FOOTPRINT(sceAgcDcbSetIndexBuffer(cb, 0), 12);
   REJECTED(sceAgcDcbSetIndexBuffer(cb, address + 1));

   FOOTPRINT(sceAgcDcbSetNumInstances(cb, 7), sceAgcDcbSetNumInstancesGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_NUM_INSTANCES, 0, 0), 7);
   EXPECT_PARSE(out, 2, "NUM_INSTANCES");

   FOOTPRINT(sceAgcDcbSetBaseIndirectArgs(cb, 0, address + 8),
             sceAgcDcbSetBaseDrawIndirectArgsGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_SET_BASE, 2, 0), S_111_BASE_INDEX(V_111_PATCH_TABLE_BASE),
                (uint32_t)(address + 8), (uint32_t)(address >> 32));
   EXPECT_PARSE(out, 4, "SET_BASE");
   FOOTPRINT(sceAgcDcbSetBaseIndirectArgs(cb, 1, address),
             sceAgcDcbSetBaseDispatchIndirectArgsGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_SET_BASE, 2, 0) | PKT3_SHADER_TYPE_S(1),
                S_111_BASE_INDEX(V_111_PATCH_TABLE_BASE), (uint32_t)address,
                (uint32_t)(address >> 32));
   EXPECT_PARSE(out, 4, "SET_BASE");
   REJECTED(sceAgcDcbSetBaseIndirectArgs(cb, 2, address));
   REJECTED(sceAgcDcbSetBaseIndirectArgs(cb, 0, address + 4));
   REJECTED(sceAgcDcbSetBaseIndirectArgs(cb, 0, UINT64_C(1) << 48));

   FOOTPRINT(sceAgcDcbSetIndexIndirectArgs(cb, address + 0x10, 3),
             sceAgcDcbSetIndexIndirectArgsGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_INDEX_ATTRIBUTES_INDIRECT, 2, 0),
                S_911_ATTRIBUTE_BASE_LO((uint32_t)(address + 0x10) >> 4),
                (uint32_t)(address >> 32), S_913_ATTRIBUTE_INDEX(3));
   EXPECT_PARSE_NAMELESS(out, 4, "ATTRIBUTE_INDEX");
   REJECTED(sceAgcDcbSetIndexIndirectArgs(cb, address + 8, 0));
   REJECTED(sceAgcDcbSetIndexIndirectArgs(cb, address, 0x10000));

   FOOTPRINT(sceAgcDcbSetPredication(cb, 1, 1, 1, address, 0),
             sceAgcDcbSetZPassPredicationEnableGetSize());
   reference_begin(&ref);
   ac_emit_cp_set_predication(&ref.cs, info.gfx_level, address,
                              S_201_PRED_OP(PREDICATION_OP_ZPASS) | PREDICATION_DRAW_VISIBLE |
                                 PREDICATION_HINT_NOWAIT_DRAW);
   EXPECT_REFERENCE(out, ref, 4);
   EXPECT_PARSE(out, 4, "SET_PREDICATION");

   FOOTPRINT(sceAgcDcbSetPredication(cb, 0, 1, 0, address, 0), 16);
   reference_begin(&ref);
   ac_emit_cp_set_predication(&ref.cs, info.gfx_level, address,
                              S_201_PRED_OP(PREDICATION_OP_ZPASS) | PREDICATION_DRAW_NOT_VISIBLE |
                                 PREDICATION_HINT_WAIT);
   EXPECT_REFERENCE(out, ref, 4);

   FOOTPRINT(sceAgcDcbSetPredication(cb, 1, 3, 1, address, 0),
             sceAgcDcbSetBoolPredicationEnableGetSize());
   reference_begin(&ref);
   ac_emit_cp_set_predication(&ref.cs, info.gfx_level, address,
                              S_201_PRED_OP(PREDICATION_OP_BOOL64) | PREDICATION_DRAW_VISIBLE);
   EXPECT_REFERENCE(out, ref, 4);
   EXPECT_PARSE(out, 4, "SET_PREDICATION");

   FOOTPRINT(sceAgcDcbSetPredication(cb, 1, 0, 1, 0, 0), sceAgcDcbSetPredicationDisableGetSize());
   reference_begin(&ref);
   ac_emit_cp_set_predication(&ref.cs, info.gfx_level, 0, 0);
   EXPECT_REFERENCE(out, ref, 4);
   EXPECT_PARSE(out, 4, "SET_PREDICATION");
   REJECTED(sceAgcDcbSetPredication(cb, 2, 1, 0, address, 0));
   REJECTED(sceAgcDcbSetPredication(cb, 1, 5, 0, address, 0));
   REJECTED(sceAgcDcbSetPredication(cb, 1, 1, 2, address, 0));
   REJECTED(sceAgcDcbSetPredication(cb, 1, 1, 0, 0, 0));
   REJECTED(sceAgcDcbSetPredication(cb, 1, 3, 0, address + 4, 0));

   FOOTPRINT(sceAgcDcbStallCommandBufferParser(cb), sceAgcDcbStallCommandBufferParserGetSize());
   reference_begin(&ref);
   ac_emit_cp_pfp_sync_me(&ref.cs, false);
   EXPECT_REFERENCE(out, ref, 2);
   EXPECT_PARSE(out, 2, "PFP_SYNC_ME");

   FOOTPRINT(sceAgcDcbDispatchIndirect(cb, 64, 0x8021), sceAgcDcbDispatchIndirectGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DISPATCH_INDIRECT, 1, 0) | PKT3_SHADER_TYPE_S(1), 64,
                S_00B800_COMPUTE_SHADER_EN(1) | S_00B800_ORDER_MODE(1) |
                   S_00B800_USE_THREAD_DIMENSIONS(1) | S_00B800_CS_W32_EN(1));
   EXPECT_PARSE(out, 3, "DISPATCH_INDIRECT");
   REJECTED(sceAgcDcbDispatchIndirect(cb, 64, 0x2));
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

   FOOTPRINT(sceAgcDcbWriteData(cb, DST_GL2_PFP, 0, address, data, 3, 0, 1),
             sceAgcDcbWriteDataGetSize(3));
   reference_begin(&ref);
   ac_emit_cp_write_data(&ref.cs, V_371_PREFETCH_PARSER, V_371_TC_L2, address, 3, data, false);
   EXPECT_REFERENCE(out, ref, 7);
   EXPECT_PARSE(out, 7, "WRITE_DATA");

   FOOTPRINT(sceAgcDcbWriteData(cb, DST_GL2_ME, 0, address, data, 2, 0, 1), 24);
   reference_begin(&ref);
   ac_emit_cp_write_data(&ref.cs, V_371_MICRO_ENGINE, V_371_TC_L2, address, 2, data, false);
   EXPECT_REFERENCE(out, ref, 6);
   EXPECT_PARSE(out, 6, "WRITE_DATA");

   FOOTPRINT(sceAgcDcbWriteData(cb, DST_GL2_ME, 1, address, data, 1, 1, 0), 20);
   EXPECT_WORDS(out, PKT3(PKT3_WRITE_DATA, 3, 0),
                S_371_DST_SEL(V_371_TC_L2) | S_371_ADDR_INCR(V_371_DO_NOT_INCREMENT_ADDRESS) |
                   S_371_CACHE_POLICY(policy_bits(host, V_371_STREAM)),
                (uint32_t)address, (uint32_t)(address >> 32), data[0]);
   EXPECT_PARSE(out, 5, "WRITE_DATA");

   REJECTED(sceAgcDcbWriteData(cb, DST_GL2_ME, 0, address, data, 0, 0, 0));
   REJECTED(sceAgcDcbWriteData(cb, DST_GL2_ME, 0, address, NULL, 1, 0, 0));
   REJECTED(sceAgcDcbWriteData(cb, DST_GL2_ME, 0, address, data, 0x3ffe, 0, 0));
   REJECTED(sceAgcDcbWriteData(cb, DST_GL2_ME, 0, address + 2, data, 1, 0, 0));
   REJECTED(sceAgcDcbWriteData(cb, 0x20, 0, address, data, 1, 0, 0));
   REJECTED(sceAgcDcbWriteData(cb, DST_GL2_ME, 4, address, data, 1, 0, 0));
   REJECTED(sceAgcDcbWriteData(cb, 0, 0, address, data, 1, 0, 1));
   EXPECT_FATAL(sceAgcDcbWriteData(cb, DST_GDS_ME, 0, address, data, 1, 0, 0), "no host form");
   EXPECT_FATAL(sceAgcDcbWriteData(cb, 0, 0, address, data, 1, 0, 0), "no host form");

   FOOTPRINT(sceAgcDcbCopyData(cb, DST_GL2_ME, 0, address, DST_GL2_ME, 0, source, 1, 1),
             sceAgcDcbCopyDataGetSize());
   reference_begin(&ref);
   ac_emit_cp_copy_data(&ref.cs, COPY_DATA_TC_L2, COPY_DATA_TC_L2, source, address,
                        AC_CP_COPY_DATA_WR_CONFIRM | AC_CP_COPY_DATA_COUNT_SEL, false);
   EXPECT_REFERENCE(out, ref, 6);
   EXPECT_PARSE(out, 6, "COPY_DATA");

   FOOTPRINT(sceAgcDcbCopyData(cb, DST_GL2_PFP, 0, address, SRC_IMMEDIATE_PFP, 0, 0x55, 0, 1), 24);
   reference_begin(&ref);
   ac_emit_cp_copy_data(&ref.cs, COPY_DATA_IMM, COPY_DATA_TC_L2, 0x55, address,
                        AC_CP_COPY_DATA_WR_CONFIRM | AC_CP_COPY_DATA_ENGINE_PFP, false);
   EXPECT_REFERENCE(out, ref, 6);
   EXPECT_PARSE(out, 6, "COPY_DATA");

   FOOTPRINT(sceAgcDcbCopyData(cb, DST_GL2_ME, 0, address, SRC_REF_CLOCK_ME, 0, 0, 1, 1), 24);
   reference_begin(&ref);
   ac_emit_cp_copy_data(&ref.cs, COPY_DATA_TIMESTAMP, COPY_DATA_TC_L2, 0, address,
                        AC_CP_COPY_DATA_WR_CONFIRM | AC_CP_COPY_DATA_COUNT_SEL, false);
   EXPECT_REFERENCE(out, ref, 6);
   EXPECT_PARSE(out, 6, "COPY_DATA");

   FOOTPRINT(sceAgcDcbCopyData(cb, DST_GL2_ME, 2, address, SRC_ATOMIC_RETURN_ME, 1, 0, 0, 0), 24);
   EXPECT_WORDS(out, PKT3(PKT3_COPY_DATA, 4, 0),
                S_401_SRC_SEL(V_401_ATOMIC_RETURN_DATA) | S_401_DST_SEL(V_401_TC_L2) |
                   S_401_SRC_CACHE_POLICY(policy_bits(host, V_401_STREAM)) |
                   S_401_DST_CACHE_POLICY(policy_bits(host, V_401_NOA)),
                0, 0, (uint32_t)address, (uint32_t)(address >> 32));
   EXPECT_PARSE(out, 6, "COPY_DATA");

   REJECTED(sceAgcDcbCopyData(cb, DST_GL2_ME, 4, address, DST_GL2_ME, 0, source, 0, 0));
   REJECTED(sceAgcDcbCopyData(cb, DST_GL2_ME, 0, address, DST_GL2_ME, 0, source, 2, 0));
   REJECTED(sceAgcDcbCopyData(cb, DST_GL2_PFP, 0, address, SRC_IMMEDIATE_PFP, 0,
                              UINT64_C(0x100000000), 0, 0));
   EXPECT_FATAL(sceAgcDcbCopyData(cb, DST_GL2_ME, 0, address, DST_GDS_ME, 0, 0, 0, 0),
                "no host form");
   EXPECT_FATAL(sceAgcDcbCopyData(cb, DST_PERFCOUNTER_ME, 0, 0, DST_GL2_ME, 0, source, 0, 0),
                "no host form");
   EXPECT_FATAL(sceAgcDcbCopyData(cb, DST_GL2_PFP, 0, address, SRC_ATOMIC_RETURN_PFP, 0, 0, 0, 0),
                "no host form");

   FOOTPRINT(sceAgcDcbDmaData(cb, 0, DMA_GL2, 0, address, DMA_GL2, 0, source, 4096, 1, 0, 1),
             sceAgcDcbDmaDataGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DMA_DATA, 5, 0),
                S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2) | S_501_DST_SEL(V_501_DST_ADDR_USING_L2) |
                   S_501_CP_SYNC(1),
                (uint32_t)source, (uint32_t)(source >> 32), (uint32_t)address,
                (uint32_t)(address >> 32), S_506_BYTE_COUNT(4096) | S_506_RAW_WAIT(1));
   EXPECT_PARSE(out, 7, "DMA_DATA");

   FOOTPRINT(sceAgcDcbDmaData(cb, 1, DMA_GL2, 3, address, DMA_IMMEDIATE, 0, 0xdeadbeef, 64, 0, 0,
                              0),
             28);
   EXPECT_WORDS(out, PKT3(PKT3_DMA_DATA, 5, 0),
                S_501_ENGINE_SEL(V_501_PREFETCH_PARSER) | S_501_SRC_SEL(V_501_DATA) |
                   S_501_DST_SEL(V_501_DST_ADDR_USING_L2) |
                   S_501_DST_CACHE_POLICY(policy_bits(host, V_501_BYPASS)),
                0xdeadbeef, 0, (uint32_t)address, (uint32_t)(address >> 32), S_506_BYTE_COUNT(64));
   EXPECT_PARSE(out, 7, "DMA_DATA");

   FOOTPRINT(sceAgcDcbDmaData(cb, 0, DMA_NOWHERE, 0, source, DMA_GL2, 0, source, 256, 0, 1, 0),
             28);
   EXPECT_WORDS(out, PKT3(PKT3_DMA_DATA, 5, 0),
                S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2) | S_501_DST_SEL(V_501_DST_NOWHERE),
                (uint32_t)source, (uint32_t)(source >> 32), (uint32_t)source,
                (uint32_t)(source >> 32), S_506_BYTE_COUNT(256) | S_506_DISABLE_WR_CONFIRM(1));
   EXPECT_PARSE(out, 7, "DMA_DATA");

   REJECTED(sceAgcDcbDmaData(cb, 2, DMA_GL2, 0, address, DMA_GL2, 0, source, 4, 0, 0, 0));
   REJECTED(sceAgcDcbDmaData(cb, 0, 0x10, 0, address, DMA_GL2, 0, source, 4, 0, 0, 0));
   REJECTED(sceAgcDcbDmaData(cb, 0, DMA_GL2, 0, address, 0x10, 0, source, 4, 0, 0, 0));
   REJECTED(sceAgcDcbDmaData(cb, 0, DMA_GL2, 0, address, DMA_GL2, 0, source, 0x4000000, 0, 0, 0));
   EXPECT_FATAL(sceAgcDcbDmaData(cb, 0, DMA_GL2, 0, address, DMA_ATOMIC_RETURN, 0, 0, 8, 0, 0, 0),
                "console registers");
   FOOTPRINT(sceAgcDcbDmaData(cb, 0, DMA_GDS, 1, 0x40, DMA_GL2, 0, source, 8, 0, 0, 1), 28);
   EXPECT_WORDS(out, PKT3(PKT3_DMA_DATA, 5, 0),
                S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2) | S_501_DST_SEL(V_501_DST_ADDR_USING_L2) |
                   S_501_CP_SYNC(1),
                (uint32_t)source, (uint32_t)(source >> 32), (uint32_t)(AGC_RUNTIME_GDS_VA + 0x40),
                (uint32_t)((AGC_RUNTIME_GDS_VA + 0x40) >> 32), S_506_BYTE_COUNT(8));
   EXPECT_PARSE(out, 7, "DMA_DATA");
   REJECTED(sceAgcDcbDmaData(cb, 0, DMA_GDS, 0, 0x10000, DMA_GL2, 0, source, 4, 0, 0, 0));
   EXPECT_FATAL(sceAgcDcbDmaData(cb, 0, DMA_GL2, 0, address, 4, 0, source, 8, 0, 0, 0),
                "register address space");
   if (host == AGC_HOST_GFX103) {
      FOOTPRINT(sceAgcDcbDmaData(cb, 0, DMA_GL2, 0, address, DMA_GL2, 0, source, 0x10000, 0, 0, 0),
                28);
      CHECK_EQ(out[6], S_506_BYTE_COUNT(0x10000));
   } else {
      FOOTPRINT(sceAgcDcbDmaData(cb, 0, DMA_GL2, 0, address, DMA_GL2, 0, source, 0x10000, 1, 0, 1),
                28);
      {
         const uint32_t chunk = 32736;
         const uint32_t sizes[3] = {chunk, chunk, 0x10000 - 2 * chunk};
         reference_begin(&ref);
         for (unsigned i = 0; i < 3; i++) {
            uint32_t *w = ref.words + ref.cs.cdw;
            w[0] = PKT3(PKT3_DMA_DATA, 5, 0);
            w[1] = S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2) | S_501_DST_SEL(V_501_DST_ADDR_USING_L2) |
                   S_501_CP_SYNC(i == 2);
            w[2] = (uint32_t)(source + i * chunk);
            w[3] = (uint32_t)((source + i * chunk) >> 32);
            w[4] = (uint32_t)(address + i * chunk);
            w[5] = (uint32_t)((address + i * chunk) >> 32);
            w[6] = S_506_BYTE_COUNT(sizes[i]) | S_506_RAW_WAIT(i == 0);
            ref.cs.cdw += 7;
         }
         EXPECT_TRAMPOLINE(out, 7, ref);
         CHECK_EQ(sceAgcDmaDataPatchSetDstAddressOrOffset(out, address + 0x100000), 0);
         CHECK_EQ(sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate(out, source + 0x200000), 0);
         for (unsigned i = 0; i < 3; i++) {
            ref.words[7 * i + 2] = (uint32_t)(source + 0x200000 + i * chunk);
            ref.words[7 * i + 3] = (uint32_t)((source + 0x200000 + i * chunk) >> 32);
            ref.words[7 * i + 4] = (uint32_t)(address + 0x100000 + i * chunk);
            ref.words[7 * i + 5] = (uint32_t)((address + 0x100000 + i * chunk) >> 32);
         }
         EXPECT_TRAMPOLINE(out, 7, ref);
      }
      FOOTPRINT(sceAgcDcbDmaData(cb, 0, DMA_GL2, 0, address, DMA_GL2, 0, source, 32767, 0, 0, 0),
                28);
   }

   FOOTPRINT(sceAgcDcbAtomicMem(cb, 0, 111, 2, 0, address, 5, 9, 10), sceAgcDcbAtomicMemGetSize());
   reference_begin(&ref);
   ac_emit_cp_atomic_mem(&ref.cs, 111, V_1E1_WAIT_FOR_WRITE_CONFIRMATION, address, 5, 9);
   EXPECT_REFERENCE(out, ref, 9);
   EXPECT_PARSE(out, 9, "ATOMIC_MEM");

   FOOTPRINT(sceAgcDcbAtomicMem(cb, 1, 15, 0, 1, address + 4, 2, 0, 0x1fff), 36);
   EXPECT_WORDS(out, PKT3(PKT3_ATOMIC_MEM, 7, 0),
                S_1E1_ATOMIC(15) | S_1E1_COMMAND(V_1E1_SINGLE_PASS_ATOMIC) |
                   S_1E1_CACHE_POLICY(policy_bits(host, V_1E1_STREAM)) |
                   S_1E1_ENGINE_SEL(V_1E1_PREFETCH_PARSER),
                (uint32_t)(address + 4), (uint32_t)(address >> 32), 2, 0, 0, 0,
                S_1E8_LOOP_INTERVAL(0x1fff));
   EXPECT_PARSE(out, 9, "ATOMIC_MEM");
   REJECTED(sceAgcDcbAtomicMem(cb, 2, 15, 0, 0, address, 0, 0, 0));
   REJECTED(sceAgcDcbAtomicMem(cb, 0, 0x80, 0, 0, address, 0, 0, 0));
   REJECTED(sceAgcDcbAtomicMem(cb, 0, 15, 4, 0, address, 0, 0, 0));
   REJECTED(sceAgcDcbAtomicMem(cb, 0, 15, 0, 0, 0, 0, 0, 0));
   REJECTED(sceAgcDcbAtomicMem(cb, 0, 111, 0, 0, address + 4, 0, 0, 0));
   REJECTED(sceAgcDcbAtomicMem(cb, 0, 15, 0, 0, address, 0, 0, 0x2000));
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

   FOOTPRINT(sceAgcDcbWaitRegMem(cb, 0, 3, 0, 0, address, 0x1234, 0xffffffff, 64),
             sceAgcDcbWaitOnAddressGetSize(0));
   reference_begin(&ref);
   ac_emit_cp_wait_mem(&ref.cs, address, 0x1234, 0xffffffff, WAIT_REG_MEM_EQUAL);
   EXPECT_REFERENCE(out, ref, 7);
   EXPECT_NOPS(out + 7, 7);
   EXPECT_PARSE(out, 14, "WAIT_REG_MEM");

   FOOTPRINT(sceAgcDcbWaitRegMem(cb, 0, 5, 1, 0, address, 1, 0xff, 64), 56);
   reference_begin(&ref);
   ac_emit_cp_wait_mem(&ref.cs, address, 1, 0xff, WAIT_REG_MEM_GREATER_OR_EQUAL | WAIT_REG_MEM_PFP);
   EXPECT_REFERENCE(out, ref, 7);
   EXPECT_PARSE(out, 14, "WAIT_REG_MEM");

   FOOTPRINT(sceAgcDcbWaitRegMem(cb, 1, 4, 1, 2, address, wide, ~wide, 0xffff0),
             sceAgcDcbWaitOnAddressGetSize(1));
   EXPECT_WORDS(out, PKT3(PKT3_WAIT_REG_MEM64, 7, 0),
                S_931_FUNCTION(V_931_NOT_EQUAL_REFERENCE_VALUE) |
                   S_931_MEM_SPACE(V_931_MEMORY_SPACE) | S_931_ENGINE_SEL(V_931_PREFETCH_PARSER) |
                   S_931_CACHE_POLICY(policy_bits(host, V_931_NOA)),
                (uint32_t)address, (uint32_t)(address >> 32), (uint32_t)wide,
                (uint32_t)(wide >> 32), (uint32_t)~wide, (uint32_t)(~wide >> 32),
                S_938_POLL_INTERVAL(0xffff));
   EXPECT_NOPS(out + 9, 7);
   EXPECT_PARSE_NAMELESS(out, 16, "REFERENCE_HI");

   FOOTPRINT(sceAgcDcbWaitRegMem(cb, 0, 0, 0, 0, 0, 0, 0, 0), 56);
   CHECK_EQ(out[2], 0);
   CHECK_EQ(sceAgcDcbWaitOnAddressGetSize(2), 0);
   REJECTED(sceAgcDcbWaitRegMem(cb, 2, 3, 0, 0, address, 0, 0, 0));
   REJECTED(sceAgcDcbWaitRegMem(cb, 0, 7, 0, 0, address, 0, 0, 0));
   REJECTED(sceAgcDcbWaitRegMem(cb, 1, 3, 8, 0, address, 0, 0, 0));
   REJECTED(sceAgcDcbWaitRegMem(cb, 0, 3, 0, 4, address, 0, 0, 0));
   REJECTED(sceAgcDcbWaitRegMem(cb, 0, 3, 0, 0, address, 0, 0, 0x100000));
   REJECTED(sceAgcDcbWaitRegMem(cb, 0, 3, 0, 0, address + 2, 0, 0, 0));
   REJECTED(sceAgcDcbWaitRegMem(cb, 1, 3, 0, 0, address + 4, 0, 0, 0));
   REJECTED(sceAgcDcbWaitRegMem(cb, 0, 3, 0, 0, UINT64_C(1) << 48, 0, 0, 0));
   EXPECT_FATAL(sceAgcDcbWaitRegMem(cb, 0, 3, 4, 0, address, 0, 0, 0), "no host form");
   EXPECT_FATAL(sceAgcDcbWaitRegMem(cb, 1, 3, 4, 0, address, 0, 0, 0), "no host form");

   struct ac_barrier_state barrier = {.flags = AC_BARRIER_INV_SMEM | AC_BARRIER_INV_VMEM};
   bool context_roll = false;
   enum ac_rgp_flush_bits flush_bits = 0;
   FOOTPRINT(sceAgcDcbAcquireMem(cb, 1, 0, GCR_SCALAR | GCR_VECTOR | GCR_GL1, address, 4096, 400),
             sceAgcDcbAcquireMemGetSize());
   reference_begin(&ref);
   ac_emit_barrier(&ref.cs, level, AMD_IP_GFX, &barrier, &context_roll, &flush_bits);
   EXPECT_REFERENCE(out, ref, 8);
   EXPECT_PARSE(out, 8, "ACQUIRE_MEM");

   barrier.flags = AC_BARRIER_INV_L2 | AC_BARRIER_INV_ICACHE | AC_BARRIER_PFP_SYNC_ME;
   FOOTPRINT(sceAgcDcbAcquireMem(cb, 0, 0, GCR_METADATA | GCR_GL2_WRITEBACK_INVALIDATE | GCR_ICACHE,
                                 0, UINT64_MAX, 400),
             32);
   reference_begin(&ref);
   ac_emit_barrier(&ref.cs, level, AMD_IP_GFX, &barrier, &context_roll, &flush_bits);
   EXPECT_REFERENCE(out, ref, 8);
   EXPECT_PARSE(out, 8, "ACQUIRE_MEM");

   uint32_t writeback = S_587_GL2_WB(1) | S_587_SEQ(V_587_SEQ_FORWARD);
   if (host != AGC_HOST_GFX12)
      writeback |= S_587_GL1_INV(1);
   FOOTPRINT(sceAgcDcbAcquireMem(cb, 1, 0,
                                 GCR_GL1_RANGE | GCR_ORDER_012 | GCR_GL2_WRITEBACK | GCR_GL2_RANGE,
                                 address, 0x10000, 800),
             32);
   reference_begin(&ref);
   ac_emit_cp_acquire_mem(&ref.cs, level, AMD_IP_GFX, V_581A_MICRO_ENGINE, writeback, NULL, NULL);
   ref.words[6] = S_586A_POLL_INTERVAL(20);
   EXPECT_REFERENCE(out, ref, 8);
   EXPECT_PARSE(out, 8, "ACQUIRE_MEM");

   FOOTPRINT(sceAgcDcbAcquireMem(cb, 0, 0, 0, 0, 0, 400), 32);
   reference_begin(&ref);
   ac_emit_cp_pfp_sync_me(&ref.cs, false);
   EXPECT_REFERENCE(out, ref, 2);
   EXPECT_NOPS(out + 2, 6);
   EXPECT_PARSE(out, 8, "PFP_SYNC_ME");
   FOOTPRINT(sceAgcDcbAcquireMem(cb, 1, 0, GCR_ORDER_012, 0, 0, 400), 32);
   EXPECT_NOPS(out, 8);
   REJECTED(sceAgcDcbAcquireMem(cb, 2, 0, GCR_GL1, 0, 0, 400));
   REJECTED(sceAgcDcbAcquireMem(cb, 0, 0x80000000, GCR_GL1, 0, 0, 400));
   REJECTED(sceAgcDcbAcquireMem(cb, 0, 0, 0x80000, 0, 0, 400));
   REJECTED(sceAgcDcbAcquireMem(cb, 0, 0, GCR_GL1, 0, 0, 40 * 0x10000));
   if (level < GFX11) {
      EXPECT_FATAL(sceAgcDcbAcquireMem(cb, 1, 1 << 25, GCR_GL1, 0, 0, 400), "no gfx10.3 form yet");
   } else {
      uint32_t gcr = (level >= GFX12 ? 0 : S_587_GL1_INV(1)) | S_587_SEQ(V_587_SEQ_FORWARD);
      FOOTPRINT(sceAgcDcbAcquireMem(cb, 1, 1 << 25, GCR_GL1, 0, 0, 400), 32);
      reference_begin(&ref);
      if (level < GFX12) {
         ref.words[ref.cs.cdw++] = PKT3(PKT3_EVENT_WRITE, 0, 0);
         ref.words[ref.cs.cdw++] = EVENT_TYPE(V_028A90_FLUSH_AND_INV_CB_META) | EVENT_INDEX(0);
      }
      ac_emit_cp_release_mem_pws(&ref.cs, level, AMD_IP_GFX, V_028A90_FLUSH_AND_INV_CB_DATA_TS,
                                 gcr & C_587_GLI_INV);
      ac_emit_cp_acquire_mem_pws(&ref.cs, level, AMD_IP_GFX, V_028A90_FLUSH_AND_INV_CB_DATA_TS,
                                 V_581B_CP_ME, 0, gcr & ~C_587_GLI_INV);
      EXPECT_TRAMPOLINE(out, 8, ref);

      FOOTPRINT(sceAgcDcbAcquireMem(cb, 0, 0x7fc0, 0, 0, 0, 400), 32);
      reference_begin(&ref);
      ac_emit_cp_release_mem_pws(&ref.cs, level, AMD_IP_GFX, V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT,
                                 S_587_SEQ(V_587_SEQ_FORWARD));
      ac_emit_cp_acquire_mem_pws(&ref.cs, level, AMD_IP_GFX, V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT,
                                 V_581B_CP_PFP, 0, 0);
      EXPECT_TRAMPOLINE(out, 8, ref);

      uint32_t db_event = level == GFX11 ? V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT : V_028A90_FLUSH_AND_INV_DB_DATA_TS;
      FOOTPRINT(sceAgcDcbAcquireMem(cb, 1, 1 << 26, 0, 0, 0, 400), 32);
      reference_begin(&ref);
      if (level < GFX12 && db_event == V_028A90_FLUSH_AND_INV_DB_DATA_TS) {
         ref.words[ref.cs.cdw++] = PKT3(PKT3_EVENT_WRITE, 0, 0);
         ref.words[ref.cs.cdw++] = EVENT_TYPE(V_028A90_FLUSH_AND_INV_DB_META) | EVENT_INDEX(0);
      }
      ac_emit_cp_release_mem_pws(&ref.cs, level, AMD_IP_GFX, db_event, S_587_SEQ(V_587_SEQ_FORWARD));
      ac_emit_cp_acquire_mem_pws(&ref.cs, level, AMD_IP_GFX, db_event, V_581B_CP_ME, 0, 0);
      EXPECT_TRAMPOLINE(out, 8, ref);
   }
   CHECK_EQ(sceAgcDcbQueueEndOfShaderActionGetSize(), 32);

   FOOTPRINT(sceAgcDcbEventWrite(cb, V_028A90_PS_PARTIAL_FLUSH, 0),
             sceAgcDcbEventWriteGetSize(V_028A90_PS_PARTIAL_FLUSH));
   reference_event(&ref, V_028A90_PS_PARTIAL_FLUSH);
   EXPECT_REFERENCE(out, ref, 2);
   EXPECT_PARSE(out, 2, "EVENT_WRITE");

   static const uint8_t forwarded[] = {
      V_028A90_CS_PARTIAL_FLUSH, V_028A90_VS_PARTIAL_FLUSH, V_028A90_BREAK_BATCH,
      V_028A90_RST_PIX_CNT,      V_028A90_FLUSH_DFSM,
   };
   for (unsigned i = 0; i < sizeof(forwarded); i++) {
      FOOTPRINT(sceAgcDcbEventWrite(cb, forwarded[i], address), 8);
      reference_event(&ref, forwarded[i]);
      EXPECT_REFERENCE(out, ref, 2);
      EXPECT_PARSE(out, 2, "EVENT_WRITE");
   }

   static const uint8_t dropped[] = {
      V_028A90_PERFCOUNTER_START,
      V_028A90_PERFCOUNTER_STOP,
      V_028A90_PERFCOUNTER_SAMPLE,
   };
   for (unsigned i = 0; i < sizeof(dropped); i++) {
      FOOTPRINT(sceAgcDcbEventWrite(cb, dropped[i], 0), 8);
      EXPECT_NOPS(out, 2);
   }

   static const uint8_t meta[] = {V_028A90_FLUSH_AND_INV_CB_META, V_028A90_FLUSH_AND_INV_DB_META};
   for (unsigned i = 0; i < sizeof(meta); i++) {
      if (host == AGC_HOST_GFX11 && meta[i] == V_028A90_FLUSH_AND_INV_DB_META) {
         FOOTPRINT(sceAgcDcbEventWrite(cb, meta[i], 0), 32);
         reference_begin(&ref);
         ac_emit_cp_release_mem(&ref.cs, level, AMD_IP_GFX, V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT, 0,
                                EOP_DST_SEL_MEM, EOP_INT_SEL_NONE, EOP_DATA_SEL_DISCARD, 0, 0, 0);
         EXPECT_REFERENCE(out, ref, 8);
         continue;
      }
      FOOTPRINT(sceAgcDcbEventWrite(cb, meta[i], 0), 8);
      if (host == AGC_HOST_GFX12) {
         EXPECT_NOPS(out, 2);
         continue;
      }
      reference_event(&ref, meta[i]);
      EXPECT_REFERENCE(out, ref, 2);
      EXPECT_PARSE(out, 2, "EVENT_WRITE");
   }

   CHECK_EQ(sceAgcDcbEventWriteGetSize(V_028A90_PIXEL_PIPE_STAT_CONTROL), 16);
   CHECK_EQ(sceAgcDcbEventWriteGetSize(V_028A90_PIXEL_PIPE_STAT_DUMP), 16);
   CHECK_EQ(sceAgcDcbEventWriteGetSize(V_028A90_PIXEL_PIPE_STAT_RESET), 8);
   CHECK_EQ(sceAgcDcbBeginOcclusionQueryGetSize(0), 16);
   CHECK_EQ(sceAgcDcbBeginOcclusionQueryGetSize(1), sceAgcDcbWriteDataGetSize(64) + 16);
   CHECK_EQ(sceAgcDcbBeginOcclusionQueryGetSize(2), 0);
   CHECK_EQ(sceAgcDcbEndOcclusionQueryGetSize(), 16);
   EXPECT_FATAL(sceAgcDcbEventWrite(cb, V_028A90_PIXEL_PIPE_STAT_DUMP, address), "no host form yet");
   EXPECT_FATAL(sceAgcDcbEventWrite(cb, V_028A90_PIXEL_PIPE_STAT_CONTROL, address), "no host form");
   EXPECT_FATAL(sceAgcDcbEventWrite(cb, V_028A90_PIXEL_PIPE_STAT_RESET, 0), "no host form");
   static const uint8_t flush_events[] = {
      V_028A90_CACHE_FLUSH_AND_INV_EVENT,
      V_028A90_DB_CACHE_FLUSH_AND_INV,
      V_028A90_FLUSH_AND_INV_CB_PIXEL_DATA,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(flush_events); i++) {
      uint8_t event = flush_events[i];
      CHECK_EQ(sceAgcDcbEventWriteGetSize(event), level < GFX11 ? 8 : 32);
      FOOTPRINT(sceAgcDcbEventWrite(cb, event, 0), sceAgcDcbEventWriteGetSize(event));
      if (level < GFX11) {
         reference_event(&ref, event);
         EXPECT_REFERENCE(out, ref, 2);
      } else {
         uint32_t ts = event == V_028A90_FLUSH_AND_INV_CB_PIXEL_DATA ? V_028A90_FLUSH_AND_INV_CB_DATA_TS
                                                                    : V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT;
         reference_begin(&ref);
         ac_emit_cp_release_mem(&ref.cs, level, AMD_IP_GFX, ts, 0, EOP_DST_SEL_MEM, EOP_INT_SEL_NONE,
                                EOP_DATA_SEL_DISCARD, 0, 0, 0);
         EXPECT_REFERENCE(out, ref, 8);
         EXPECT_PARSE(out, 8, "RELEASE_MEM");
      }
   }
   EXPECT_FATAL(sceAgcDcbEventWrite(cb, 0x15, 0), "no host form");
   REJECTED(sceAgcDcbEventWrite(cb, 0x40, 0));
   REJECTED(sceAgcDcbEventWrite(cb, V_028A90_PIXEL_PIPE_STAT_DUMP, 0));
   REJECTED(sceAgcDcbEventWrite(cb, V_028A90_PIXEL_PIPE_STAT_DUMP, address + 4));

   FOOTPRINT(sceAgcDcbPrimeUtcl2(cb, 1, 1, address, 16), sceAgcDcbPrimeUtcl2GetSize());
   EXPECT_NOPS(out, 5);
   EXPECT_PARSE(out, 5, "NOP");
}

static struct agc_reloc *
reloc_of(uint32_t *packet)
{
   return agc_record_reloc_at(agc_record_containing(packet), packet);
}

static bool
reloc_live(uint32_t *packet)
{
   struct agc_record *record = agc_record_containing(packet);
   struct agc_reloc *reloc = agc_record_reloc_at(record, packet);
   return reloc && agc_reloc_live(record, reloc, (uint32_t)(record->end - record->bottom));
}

static void
test_flow(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   struct reference ref;
   enum amd_gfx_level level = host_chips[host].level;
   const uint64_t target = UINT64_C(0x000000789abcdef0);

   FOOTPRINT(sceAgcDcbJump(cb, 1, 0, target, 100), sceAgcDcbJumpGetSize());
   reference_begin(&ref);
   ac_emit_cp_indirect_buffer(&ref.cs, target, 100,
                              AC_CP_INDIRECT_BUFFER_CHAIN | AC_CP_INDIRECT_BUFFER_VALID, false);
   EXPECT_REFERENCE(out, ref, 4);
   EXPECT_PARSE(out, 4, "INDIRECT_BUFFER");

   FOOTPRINT(sceAgcDcbJump(cb, 0, 0, target, 0xfffff), 16);
   reference_begin(&ref);
   ac_emit_cp_indirect_buffer(&ref.cs, target, 0xfffff, 0, false);
   EXPECT_REFERENCE(out, ref, 4);
   EXPECT_PARSE(out, 4, "INDIRECT_BUFFER");

   FOOTPRINT(sceAgcDcbJump(cb, 1, 1, 0, 0), 16);
   EXPECT_WORDS(out, PKT3(PKT3_INDIRECT_BUFFER, 2, 0), 0, 0,
                S_3F3_CHAIN(1) | S_3F3_VALID(1) | S_3F3_CACHE_POLICY(policy_bits(host, 1)));
   REJECTED(sceAgcDcbJump(cb, 2, 0, target, 1));
   REJECTED(sceAgcDcbJump(cb, 0, 4, target, 1));
   REJECTED(sceAgcDcbJump(cb, 0, 0, target + 2, 1));
   REJECTED(sceAgcDcbJump(cb, 0, 0, target, 0x100000));

   FOOTPRINT(sceAgcDcbCondExec(cb, target, 12), sceAgcDcbCondExecGetSize());
   reference_begin(&ref);
   ac_emit_cp_cond_exec(&ref.cs, level, target, 12);
   EXPECT_REFERENCE(out, ref, 5);
   EXPECT_PARSE(out, 5, "COND_EXEC");
   REJECTED(sceAgcDcbCondExec(cb, 0, 1));
   REJECTED(sceAgcDcbCondExec(cb, target + 1, 1));
   REJECTED(sceAgcDcbCondExec(cb, target, 0x4000));

   FOOTPRINT(sceAgcDcbRewind(cb, 1), sceAgcDcbRewindGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_REWIND, 0, 0), S_591_VALID(1));
   EXPECT_PARSE(out, 2, "REWIND");
   FOOTPRINT(sceAgcDcbRewind(cb, 0), 8);
   EXPECT_WORDS(out, PKT3(PKT3_REWIND, 0, 0), 0);
   REJECTED(sceAgcDcbRewind(NULL, 0));
}

static void
test_markers(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   static char long_text[AGC_MARKER_TEXT_MAX + 1];
   memset(long_text, 'm', AGC_MARKER_TEXT_MAX);

   FOOTPRINT(sceAgcDcbPushMarker(cb, "abc", 0xff0000), 8);
   EXPECT_NOPS(out, 2);
   EXPECT_PARSE(out, 2, "NOP");
   FOOTPRINT(sceAgcDcbPushMarker(cb, "abcd", 0), 12);
   EXPECT_NOPS(out, 3);
   FOOTPRINT(sceAgcDcbPushMarker(cb, NULL, 0), 8);
   EXPECT_NOPS(out, 2);
   FOOTPRINT(sceAgcDcbSetMarker(cb, "abcdefgh", 0), 24);
   EXPECT_NOPS(out, 6);
   EXPECT_PARSE(out, 6, "NOP");
   FOOTPRINT(sceAgcDcbPushMarkerSpan(cb, "abcdefgh", 3, 0), 8);
   EXPECT_NOPS(out, 2);
   FOOTPRINT(sceAgcDcbPushMarkerSpan(cb, NULL, 0, 0), 8);
   FOOTPRINT(sceAgcDcbPushMarkerSpan(cb, long_text, 1000, 0), 1008);
   EXPECT_NOPS(out, 252);
   FOOTPRINT(sceAgcDcbSetMarkerSpan(cb, "abcdefgh", 4, 0), 20);
   EXPECT_NOPS(out, 5);
   FOOTPRINT(sceAgcDcbPopMarker(cb), 8);
   EXPECT_NOPS(out, 2);
   REJECTED(sceAgcDcbPushMarkerSpan(cb, NULL, 1, 0));
   REJECTED(sceAgcDcbPushMarkerSpan(cb, long_text, AGC_MARKER_TEXT_MAX, 0));
   REJECTED(sceAgcDcbPushMarker(cb, long_text, 0));
   REJECTED(sceAgcDcbSetMarker(cb, long_text, 0));
   REJECTED(sceAgcDcbPushMarker(NULL, "abc", 0));
   REJECTED(sceAgcDcbPopMarker(NULL));
}

static void
test_presentation(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;

   FOOTPRINT(sceAgcDcbSetFlip(cb, 7, -1, 2, INT64_C(-3)), 24);
   EXPECT_NOPS(out, 6);
   EXPECT_PARSE(out, 6, "NOP");
   struct agc_reloc *reloc = reloc_of(out);
   CHECK(reloc != NULL && reloc->kind == AGC_RELOC_FLIP && reloc_live(out));
   if (reloc) {
      CHECK_EQ(reloc->arg32, 7);
      CHECK_EQ(reloc->arg64[0], UINT64_C(0x2ffffffff));
      CHECK_EQ(reloc->arg64[1], UINT64_C(0xfffffffffffffffd));
   }

   FOOTPRINT(sceAgcDcbWaitUntilSafeForRendering(cb, 9, 2), 16);
   EXPECT_NOPS(out, 4);
   reloc = reloc_of(out);
   CHECK(reloc != NULL && reloc->kind == AGC_RELOC_WAIT_RENDERING && reloc_live(out));
   if (reloc) {
      CHECK_EQ(reloc->arg32, 9);
      CHECK_EQ(reloc->arg64[0], 2);
   }
   REJECTED(sceAgcDcbSetFlip(NULL, 7, 0, 0, 0));
   REJECTED(sceAgcDcbWaitUntilSafeForRendering(NULL, 7, 0));

   const uint32_t ids[] = {0, 5, 63};
   const uint32_t twice[] = {4, 4};
   const uint32_t wide[] = {64};
   FOOTPRINT(sceAgcDcbSetWorkloadsActive(cb, 3, ids, 3), 72);
   EXPECT_NOPS(out, 18);
   EXPECT_PARSE(out, 18, "NOP");
   FOOTPRINT(sceAgcDcbSetWorkloadComplete(cb, 31, 63), 48);
   EXPECT_NOPS(out, 12);
   FOOTPRINT(sceAgcDcbSetWorkloadStreamInactive(cb, 1), 36);
   EXPECT_NOPS(out, 9);
   REJECTED(sceAgcDcbSetWorkloadsActive(cb, 0, ids, 3));
   REJECTED(sceAgcDcbSetWorkloadsActive(cb, 32, ids, 3));
   REJECTED(sceAgcDcbSetWorkloadsActive(cb, 3, NULL, 3));
   REJECTED(sceAgcDcbSetWorkloadsActive(cb, 3, ids, 0));
   REJECTED(sceAgcDcbSetWorkloadsActive(cb, 3, ids, 64));
   REJECTED(sceAgcDcbSetWorkloadsActive(cb, 3, twice, 2));
   REJECTED(sceAgcDcbSetWorkloadsActive(cb, 3, wide, 1));
   REJECTED(sceAgcDcbSetWorkloadComplete(cb, 3, 64));
   REJECTED(sceAgcDcbSetWorkloadComplete(cb, 0, 1));
   REJECTED(sceAgcDcbSetWorkloadStreamInactive(cb, 32));
   REJECTED(sceAgcDcbSetWorkloadStreamInactive(NULL, 1));
}

static void
expect_driver_call(enum agc_host host, struct agc_cb *cb, int32_t operation, uint32_t footprint,
                   int line)
{
   uint64_t va = 0;
   uint32_t dwords = 0;
   uint32_t reports = agc_report_count();
   uint32_t *before = cb->up;
   uint32_t *out = operation < 0 ? sceAgcDcbResetQueue(cb, 0xfff, 0x12345678)
                                 : sceAgcDcbContextStateOp_0100(cb, (uint32_t)operation);
   bool known = operation < 0 ? agc_driver_reset_sequence(&va, &dwords)
                              : agc_driver_context_sequence((uint32_t)operation, &va, &dwords);
   if (!known) {
      agc_test_check(out == NULL && cb->up == before, __FILE__, line,
                     "no driver sequence: nothing is written");
      agc_test_check(agc_report_count() > reports, __FILE__, line, "the failure is reported");
      return;
   }
   struct reference ref;
   agc_test_check(out == before, __FILE__, line, "the call starts the footprint");
   agc_test_check_eq((uint64_t)(cb->up - before), footprint, __FILE__, line, "footprint");
   reference_begin(&ref);
   ac_emit_cp_indirect_buffer(&ref.cs, va, dwords, AC_CP_INDIRECT_BUFFER_VALID, false);
   expect_reference(out, &ref, 4, line);
   expect_nops(out + 4, footprint - 4, line);
   expect_parse(host, AMD_IP_GFX, out, footprint, "INDIRECT_BUFFER", false, line);
}

static void
test_queue_state(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;

   expect_driver_call(host, cb, -1, AGC_DWORDS_RESET_QUEUE_DCB, __LINE__);
   static const uint32_t context_dwords[] = {5, 27, 27, 32};
   for (uint32_t op = 0; op < 4; op++) {
      CHECK_EQ(sceAgcDcbContextStateOpGetSize(op), context_dwords[op] * 4);
      expect_driver_call(host, cb, (int32_t)op, context_dwords[op], __LINE__);
   }
   CHECK_EQ(sceAgcDcbContextStateOpGetSize(4), 0);
   REJECTED(sceAgcDcbContextStateOp_0100(cb, 4));
   REJECTED(sceAgcDcbContextStateOp_0100(NULL, 0));
   REJECTED(sceAgcDcbResetQueue(cb, 0x1000, 0));
   REJECTED(sceAgcDcbResetQueue(NULL, 1, 0));
}

static void
test_gds(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   struct reference ref;
   const uint64_t gds = AGC_RUNTIME_GDS_VA;

   CHECK_EQ(sceAgcDcbAtomicGdsGetSize(), 44);
   FOOTPRINT(sceAgcDcbAtomicGds_0900(cb, 0, AGC_GDS_OP_ADD, 0, 0, 0, 0x100, 0x200, 4, 0x20, 5, 0),
             sceAgcDcbAtomicGdsGetSize());
   reference_begin(&ref);
   ac_emit_cp_atomic_mem(&ref.cs, V_1E1_GL2_OP_ATOMIC_ADD_32, V_1E1_SINGLE_PASS_ATOMIC,
                         gds + 0x124, 5, 0);
   EXPECT_REFERENCE(out, ref, 9);
   EXPECT_NOPS(out + 9, 2);
   EXPECT_PARSE(out, 11, "ATOMIC_MEM");

   FOOTPRINT(sceAgcDcbAtomicGds_0900(cb, 0, AGC_GDS_OP_ADD | AGC_GDS_OP_RETURN, 1, 0, 0, 0, 0, 0,
                                     0x40, 1, 0),
             44);
   reference_begin(&ref);
   ac_emit_cp_atomic_mem(&ref.cs, V_1E1_GL2_OP_ATOMIC_ADD_RTN_32, V_1E1_SINGLE_PASS_ATOMIC,
                         gds + 0x40, 1, 0);
   EXPECT_REFERENCE(out, ref, 9);

   FOOTPRINT(sceAgcDcbAtomicGds_0900(cb, 1, AGC_GDS_OP_STORE, 1, 0, 0, 0, 0, 0, 8, 0x77, 0), 44);
   EXPECT_WORDS(out, PKT3(PKT3_ATOMIC_MEM, 7, 0),
                S_1E1_ATOMIC(V_1E1_GL2_OP_ATOMIC_SWAP_32) |
                   S_1E1_COMMAND(V_1E1_WAIT_FOR_WRITE_CONFIRMATION) |
                   S_1E1_ENGINE_SEL(V_1E1_PREFETCH_PARSER),
                (uint32_t)(gds + 8), (uint32_t)(gds >> 32), 0x77, 0, 0, 0,
                S_1E8_LOOP_INTERVAL(AGC_GDS_LOOP_INTERVAL));
   EXPECT_PARSE(out, 11, "ATOMIC_MEM");

   FOOTPRINT(sceAgcDcbAtomicGds_0900(cb, 0, AGC_GDS_OP_CMPSTORE | AGC_GDS_OP_LOOP, 0, 0, 0, 0, 0, 0,
                                     0xc, 7, 9),
             44);
   reference_begin(&ref);
   ac_emit_cp_atomic_mem(&ref.cs, V_1E1_GL2_OP_ATOMIC_CMPSWAP_32,
                         V_1E1_LOOP_UNTIL_COMPARE_SATISFIED, gds + 0xc, 9, 7);
   EXPECT_REFERENCE(out, ref, 9);

   FOOTPRINT(sceAgcDcbAtomicGds_0900(cb, 0, AGC_GDS_OP_READ | AGC_GDS_OP_RETURN | AGC_GDS_OP_64BIT,
                                     0, 0, 0, 0xfff0, 0, 0, 0x24, 99, 98),
             44);
   reference_begin(&ref);
   ac_emit_cp_atomic_mem(&ref.cs, V_1E1_GL2_OP_ATOMIC_ADD_RTN_64, V_1E1_SINGLE_PASS_ATOMIC,
                         gds + 0x10, 0, 0);
   EXPECT_REFERENCE(out, ref, 9);

   FOOTPRINT(sceAgcDcbAtomicGds_0900(cb, 0, AGC_GDS_OP_MAX_U | AGC_GDS_OP_64BIT, 0, 0, 0, 0, 0, 0,
                                     0x104, UINT64_C(0x100000002), 0),
             44);
   reference_begin(&ref);
   ac_emit_cp_atomic_mem(&ref.cs, V_1E1_GL2_OP_ATOMIC_UMAX_64, V_1E1_SINGLE_PASS_ATOMIC,
                         gds + 0x100, UINT64_C(0x100000002), 0);
   EXPECT_REFERENCE(out, ref, 9);

   static const uint16_t no_host_form[] = {0x02, 0x0c, 0x0e, 0x0f, 0x11, 0x12, 0x13, 0x16, 0x1e,
                                           0x39, 0x37, 0x80};
   for (unsigned i = 0; i < sizeof(no_host_form) / sizeof(no_host_form[0]); i++) {
      EXPECT_FATAL(sceAgcDcbAtomicGds_0900(cb, 0, no_host_form[i], 0, 0, 0, 0, 0, 0, 0, 0, 0),
                   "no host form");
   }
   EXPECT_FATAL(sceAgcDcbAtomicGds_0900(cb, 0, AGC_GDS_OP_ADD, 0, 0, 4, 0, 0, 0, 0, 0, 0),
                "no host form");
   REJECTED(sceAgcDcbAtomicGds_0900(cb, 2, AGC_GDS_OP_ADD, 0, 0, 0, 0, 0, 0, 0, 0, 0));
   REJECTED(sceAgcDcbAtomicGds_0900(cb, 0, AGC_GDS_OP_ADD, 2, 0, 0, 0, 0, 0, 0, 0, 0));
   REJECTED(sceAgcDcbAtomicGds_0900(cb, 0, AGC_GDS_OP_ADD, 0, 2, 0, 0, 0, 0, 0, 0, 0));
   REJECTED(sceAgcDcbAtomicGds_0900(cb, 0, 0x100, 0, 0, 0, 0, 0, 0, 0, 0, 0));
   REJECTED(sceAgcDcbAtomicGds_0900(cb, 0, AGC_GDS_OP_ADD | AGC_GDS_OP_LOOP, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0));
   REJECTED(sceAgcDcbAtomicGds_0900(NULL, 0, AGC_GDS_OP_ADD, 0, 0, 0, 0, 0, 0, 0, 0, 0));

   const uint64_t label = UINT64_C(0x0000004400001000);
   FOOTPRINT(sceAgcDcbCopyData(cb, DST_GL2_ME, 0, label, 14, 0, 0, 0, 1), 24);
   EXPECT_WORDS(out, PKT3(PKT3_COPY_DATA, 4, 0),
                S_401_SRC_SEL(V_401_ATOMIC_RETURN_DATA) | S_401_DST_SEL(V_401_TC_L2) |
                   S_401_WR_CONFIRM(1),
                0, 0, (uint32_t)label, (uint32_t)(label >> 32));
   EXPECT_PARSE(out, 6, "COPY_DATA");
   EXPECT_FATAL(sceAgcDcbCopyData(cb, DST_GL2_ME, 0, label, 16, 0, 0, 0, 0), "no host form");
}

static void
test_unsupported(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   struct reference ref;
   const uint64_t address = UINT64_C(0x0000004400002000);
   (void)host;

   FOOTPRINT(sceAgcDcbMemSemaphore(cb, address, 1, 1, 6), 16);
   reference_begin(&ref);
   ac_emit_cp_atomic_mem(&ref.cs, V_1E1_GL2_OP_ATOMIC_SWAP_64, V_1E1_SINGLE_PASS_ATOMIC, address, 1, 0);
   EXPECT_TRAMPOLINE(out, 4, ref);
   FOOTPRINT(sceAgcDcbMemSemaphore(cb, address, 0, 0, 6), 16);
   reference_begin(&ref);
   ac_emit_cp_atomic_mem(&ref.cs, V_1E1_GL2_OP_ATOMIC_ADD_64, V_1E1_SINGLE_PASS_ATOMIC, address, 1, 0);
   EXPECT_TRAMPOLINE(out, 4, ref);
   FOOTPRINT(sceAgcDcbMemSemaphore(cb, address, 0, 0, 7), 16);
   reference_begin(&ref);
   ac_emit_cp_wait_mem(&ref.cs, address, 0, 0xffffffff, WAIT_REG_MEM_NOT_EQUAL);
   ac_emit_cp_atomic_mem(&ref.cs, V_1E1_GL2_OP_ATOMIC_SUB_64, V_1E1_SINGLE_PASS_ATOMIC, address, 1, 0);
   EXPECT_TRAMPOLINE(out, 4, ref);
   REJECTED(sceAgcDcbMemSemaphore(cb, address, 0, 0, 5));
   REJECTED(sceAgcDcbMemSemaphore(cb, address, 2, 0, 6));
   REJECTED(sceAgcDcbMemSemaphore(cb, address, 0, 2, 6));
   REJECTED(sceAgcDcbMemSemaphore(cb, address + 4, 0, 0, 6));
   REJECTED(sceAgcDcbMemSemaphore(cb, 0, 0, 0, 6));

   CHECK_EQ(sceAgcDcbGetLodStatsGetSize(), 20);
   FOOTPRINT(sceAgcDcbGetLodStats(cb, 0, 0, 0, 0, 0, 0, 0), sceAgcDcbGetLodStatsGetSize());
   EXPECT_NOPS(out, 5);
   FOOTPRINT(sceAgcDcbGetLodStats(cb, 1, address, 0x840, 0, 0, 1, 0), 20);
   {
      const uint32_t copy = S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2) | S_501_DST_SEL(V_501_DST_ADDR_USING_L2) |
                            S_501_CP_SYNC(1);
      const uint64_t moves[2][3] = {
         {AGC_RUNTIME_MIP_STATS_VA, address + 64, 2048},
         {AGC_RUNTIME_MIP_STATS_TEMPLATE_VA, AGC_RUNTIME_MIP_STATS_VA, 2048},
      };
      reference_begin(&ref);
      for (unsigned i = 0; i < 2; i++) {
         uint32_t *w = ref.words + ref.cs.cdw;
         w[0] = PKT3(PKT3_DMA_DATA, 5, 0);
         w[1] = copy;
         w[2] = (uint32_t)moves[i][0];
         w[3] = (uint32_t)(moves[i][0] >> 32);
         w[4] = (uint32_t)moves[i][1];
         w[5] = (uint32_t)(moves[i][1] >> 32);
         w[6] = S_506_BYTE_COUNT((uint32_t)moves[i][2]);
         ref.cs.cdw += 7;
      }
      ac_emit_cp_write_data_imm(&ref.cs, V_371_MICRO_ENGINE, address, 1);
      EXPECT_TRAMPOLINE(out, 5, ref);
   }
   REJECTED(sceAgcDcbGetLodStats(cb, 4, 0, 0, 0, 0, 0, 0));
   REJECTED(sceAgcDcbGetLodStats(cb, 0, 0, 0, 0x100, 0, 0, 0));
   REJECTED(sceAgcDcbGetLodStats(cb, 0, 0, 0, 0, 2, 0, 0));
   REJECTED(sceAgcDcbGetLodStats(cb, 0, 0, 0, 0, 0, 2, 0));
   REJECTED(sceAgcDcbGetLodStats(cb, 0, 0, 0, 0, 0, 0, 0x100));
   REJECTED(sceAgcDcbGetLodStats(cb, 0, address + 32, 0x840, 0, 0, 1, 0));

   CHECK_EQ(sceAgcDcbDrawIndexMultiInstancedGetSize(), 60);
   EXPECT_FATAL(sceAgcDcbDrawIndexMultiInstanced(cb, 36, address, address + 256, 4,
                                                 UINT64_C(2) << 29),
                "needs a trampoline");
   REJECTED(sceAgcDcbDrawIndexMultiInstanced(cb, 36, address, address, 4, UINT64_C(1) << 40));
}

static void
test_host(enum agc_host host)
{
   static _Alignas(64) uint32_t buffer[4096];
   struct agc_cb cb;
   CHECK(agc_host_select(host));
   agc_cb_init(&cb, buffer, sizeof(buffer), NULL, NULL);

   test_draws(host, &cb);
   test_state(host, &cb);
   test_memory(host, &cb);
   test_sync(host, &cb);
   test_flow(host, &cb);
   test_markers(host, &cb);
   test_presentation(host, &cb);
   test_queue_state(host, &cb);
   test_gds(host, &cb);
   test_unsupported(host, &cb);
   expect_parse(host, AMD_IP_GFX, buffer, (unsigned)(cb.up - buffer), "DRAW_INDEX_2", true,
                __LINE__);

   _Alignas(64) uint32_t tiny[2];
   struct agc_cb small;
   agc_cb_init(&small, tiny, sizeof(tiny), NULL, NULL);
   CHECK(sceAgcDcbDrawIndexAuto(&small, 3, UINT64_C(2) << 29) == NULL);
   CHECK(sceAgcDcbWaitRegMem(&small, 0, 3, 0, 0, 0x1000, 0, 0, 0) == NULL);
   CHECK(small.up == tiny);
}

static void
test_large_marker(void)
{
   static _Alignas(64) uint32_t buffer[0x4100];
   static char text[0xfffc];
   struct agc_cb storage;
   struct agc_cb *cb = &storage;
   uint32_t *before;
   uint32_t *out;
   agc_cb_init(cb, buffer, sizeof(buffer), NULL, NULL);
   memset(text, 'x', sizeof(text));
   FOOTPRINT(sceAgcDcbPushMarkerSpan(cb, text, sizeof(text), 0), 0x4001 * 4);
   EXPECT_NOPS(out, 0x4001);
}

int
main(void)
{
   agc_set_fatal_handler(on_fatal);
   agc_record_set_stub_source(&test_stubs);
   for (unsigned host = 0; host < AGC_HOST_COUNT; host++)
      test_host(host);
   test_large_marker();
   agc_record_reset_all();
   return TEST_RESULT();
}
