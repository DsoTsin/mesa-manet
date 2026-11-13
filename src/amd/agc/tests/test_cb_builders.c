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

#define DRAW_MODIFIER (UINT64_C(2) << 29)
#define DST_GL2_ME 4
#define DST_GL2_PFP 5
#define DST_GDS_ME 6
#define ACB_GL2 2
#define ACB_GDS 3
#define DMA_GL2 3
#define GCR_SCALAR 0x80u
#define RELEASE_FLUSH_GL2 0x302u
#define RELEASE_ORDER_012 0x400u

static jmp_buf fatal_jump;
static char fatal_text[512];

static void
on_fatal(const char *message)
{
   strncpy(fatal_text, message, sizeof(fatal_text) - 1);
   longjmp(fatal_jump, 1);
}

#define EXPECT_FATAL(statement)                \
   do {                                        \
      fatal_text[0] = 0;                       \
      if (setjmp(fatal_jump) == 0) {           \
         statement;                            \
         CHECK(!"expected a fatal error");     \
      } else {                                 \
         CHECK(fatal_text[0] != 0);            \
      }                                        \
   } while (0)

static const struct {
   enum amd_gfx_level level;
   enum radeon_family family;
} host_chips[AGC_HOST_COUNT] = {
   [AGC_HOST_GFX103] = {GFX10_3, CHIP_NAVI21},
   [AGC_HOST_GFX11] = {GFX11, CHIP_NAVI31},
   [AGC_HOST_GFX115] = {GFX11_5, CHIP_STRIX_HALO},
   [AGC_HOST_GFX12] = {GFX12, CHIP_GFX1201},
};

static const uint8_t set_opcode[AGC_REG_BANK_COUNT] = {
   PKT3_SET_CONTEXT_REG,
   PKT3_SET_SH_REG,
   PKT3_SET_UCONFIG_REG,
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
   const char *path = "agc_test_cb_builders_ib.txt";
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
#define EXPECT_PARSE(ip, out, dwords, name) \
   expect_parse(host, (ip), (out), (dwords), (name), false, __LINE__)
#define EXPECT_PARSE_BOTH(out, dwords, name)        \
   do {                                             \
      EXPECT_PARSE(AMD_IP_GFX, out, dwords, name);  \
      EXPECT_PARSE(AMD_IP_COMPUTE, out, dwords, name); \
   } while (0)
#define EXPECT_PARSE_NAMELESS(ip, out, dwords, name) \
   expect_parse(host, (ip), (out), (dwords), (name), true, __LINE__)

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

#define REFUSED(call, code)                            \
   do {                                                \
      uint32_t reports = agc_report_count();           \
      CHECK_EQ((uint32_t)(call), (uint32_t)(code));    \
      CHECK_EQ(agc_report_count(), reports + 1);       \
   } while (0)

#define EXPECT_FATAL_TEXT(statement, text)             \
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

static const uint32_t *
guest_pointer(uint64_t address)
{
   return (const uint32_t *)(uintptr_t)address;
}

static bool
find_run(enum agc_reg_bank bank, enum agc_reg_class cls, uint32_t length, uint32_t *first)
{
   const struct agc_reg_bank_table *table = &agc_host_regs->bank[bank];
   uint32_t run = 0;
   for (uint32_t offset = 0; offset < table->count; offset++) {
      run = table->entries[offset].cls == cls ? run + 1 : 0;
      if (run == length) {
         *first = offset + 1 - length;
         return true;
      }
   }
   return false;
}

static bool
find_moved_run(enum agc_reg_bank bank, uint32_t length, uint32_t *first)
{
   const struct agc_reg_bank_table *table = &agc_host_regs->bank[bank];
   for (uint32_t offset = 0; offset + length <= table->count; offset++) {
      bool ok = true;
      for (uint32_t i = 0; i < length && ok; i++) {
         const struct agc_reg_entry *entry = &table->entries[offset + i];
         ok = entry->cls == AGC_REG_MOVE && entry->host_offset == table->entries[offset].host_offset + i;
      }
      if (ok) {
         *first = offset;
         return true;
      }
   }
   return false;
}

static void
test_registers(enum agc_host host)
{
   _Alignas(64) uint32_t buffer[512];
   struct agc_cb cb;
   CHECK(agc_host_select(host));
   CHECK_EQ(agc_host_current(), host);
   agc_cb_init(&cb, buffer, sizeof(buffer), NULL, NULL);

   uint32_t first;
   static const uint32_t values[4] = {0x11111111, 0x22222222, 0x33333333, 0x44444444};
   CHECK(find_run(AGC_REG_BANK_CX, AGC_REG_PASS, 4, &first));
   uint32_t *out = agc_emit_set_register_range(&cb, AGC_REG_BANK_CX, first, values, 4, "test");
   CHECK(out == buffer);
   CHECK_EQ(out[0], PKT3(PKT3_SET_CONTEXT_REG, 4, 0));
   CHECK_EQ(out[1], first);
   CHECK(memcmp(out + 2, values, sizeof(values)) == 0);
   CHECK(cb.up == buffer + 6);

   for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++) {
      uint32_t offset;
      CHECK(find_run(bank, AGC_REG_PASS, 1, &offset));
      struct agc_register reg = {offset, 0xdeadbeef};
      uint32_t *before = cb.up;
      if (bank == AGC_REG_BANK_CX)
         out = sceAgcDcbSetCxRegisterDirect(&cb, reg);
      else if (bank == AGC_REG_BANK_SH)
         out = sceAgcDcbSetShRegisterDirect(&cb, reg);
      else
         out = sceAgcDcbSetUcRegisterDirect(&cb, reg);
      CHECK(out == before && cb.up == before + 3);
      CHECK_EQ(out[0], PKT3(set_opcode[bank], 1, 0));
      CHECK_EQ(out[1], offset);
      CHECK_EQ(out[2], 0xdeadbeef);

      if (find_run(bank, AGC_REG_MOVE, 1, &offset)) {
         const struct agc_reg_entry *entry = agc_reg_lookup(agc_host_regs, bank, offset);
         out = agc_emit_set_register_range(&cb, bank, offset, values, 1, "test");
         CHECK(out != NULL);
         CHECK_EQ(out[0], PKT3(set_opcode[bank], 1, 0));
         CHECK_EQ(out[1], entry->host_offset);
         CHECK(out[1] != offset);
         CHECK_EQ(out[2], values[0]);
      }
      if (find_moved_run(bank, 3, &offset)) {
         const struct agc_reg_entry *entry = agc_reg_lookup(agc_host_regs, bank, offset);
         out = agc_emit_set_register_range(&cb, bank, offset, values, 3, "test");
         CHECK(out != NULL);
         CHECK_EQ(out[0], PKT3(set_opcode[bank], 3, 0));
         CHECK_EQ(out[1], entry->host_offset);
         CHECK(memcmp(out + 2, values, 3 * sizeof(uint32_t)) == 0);
      }
      if (find_run(bank, AGC_REG_FIELD, 1, &offset)) {
         const struct agc_reg_entry *entry = agc_reg_lookup(agc_host_regs, bank, offset);
         out = agc_emit_set_register_range(&cb, bank, offset, values + 3, 1, "test");
         CHECK(out != NULL);
         CHECK_EQ(out[1], entry->host_offset);
         CHECK_EQ(out[2], agc_reg_field_apply(entry->extra, values[3]));
      }
      if (find_run(bank, AGC_REG_PENDING, 1, &offset)) {
         uint32_t *up = cb.up;
         EXPECT_FATAL(agc_emit_set_register_range(&cb, bank, offset, values, 1, "test"));
         CHECK(strstr(fatal_text, "no host conversion") != NULL);
         cb.up = up;
      }
   }

   uint32_t *up = cb.up;
   EXPECT_FATAL(agc_emit_set_register_range(&cb, AGC_REG_BANK_CX, 0x3ff, values, 1, "test"));
   cb.up = up;

   uint32_t a, b;
   CHECK(find_run(AGC_REG_BANK_SH, AGC_REG_PASS, 1, &a));
   CHECK(find_run(AGC_REG_BANK_SH, AGC_REG_PASS, 3, &b));
   struct agc_register pairs[3] = {{b + 2, 7}, {a, 8}, {b, 9}};
   out = sceAgcCbSetShRegistersDirect(&cb, pairs, 3);
   CHECK(out == up && cb.up == up + 9);
   for (unsigned i = 0; i < 3; i++) {
      CHECK_EQ(out[i * 3], PKT3(PKT3_SET_SH_REG, 1, 0));
      CHECK_EQ(out[i * 3 + 1], pairs[i].offset);
      CHECK_EQ(out[i * 3 + 2], pairs[i].value);
   }
   up = cb.up;
   CHECK(sceAgcCbSetShRegistersDirect(&cb, pairs, 0) == up && cb.up == up);
   CHECK(sceAgcCbSetUcRegistersDirect(&cb, NULL, 2) == NULL && cb.up == up);
   CHECK(sceAgcCbSetShRegistersDirect(NULL, pairs, 1) == NULL);

   CHECK(sceAgcCbSetShRegisterRangeDirect(&cb, a, values, 0) == NULL);
   CHECK(sceAgcCbSetShRegisterRangeDirect(&cb, 0x10000, values, 1) == NULL);
   CHECK(sceAgcCbSetShRegisterRangeDirect(&cb, a, values, 0x4000) == NULL);
   CHECK(sceAgcCbSetShRegisterRangeDirect(&cb, 0xffff, values, 2) == NULL);
   CHECK(sceAgcCbSetUcRegisterRangeDirect(NULL, a, values, 1) == NULL);
   CHECK(cb.up == up);

   out = sceAgcCbSetShRegisterRangeDirect(&cb, b, NULL, 3);
   CHECK(out == up && cb.up == up + 5);
   CHECK_EQ(out[0], PKT3(PKT3_SET_SH_REG, 3, 0));
   CHECK_EQ(out[1], b);
   memset(out + 2, 0, 3 * sizeof(uint32_t));

   expect_parse(host, AMD_IP_GFX, buffer, (unsigned)(cb.up - buffer), "SET_", false, __LINE__);

   _Alignas(64) uint32_t tiny[2];
   struct agc_cb small;
   agc_cb_init(&small, tiny, sizeof(tiny), NULL, NULL);
   CHECK(agc_emit_set_register_range(&small, AGC_REG_BANK_SH, a, values, 1, "test") == NULL);
   CHECK(small.up == tiny);
}

static void
test_nops(void)
{
   static uint32_t buffer[0x8100];
   static const uint32_t sizes[] = {1, 2, 3, 5, 0x4000, 0x4001, 0x4002, 0x8003};
   for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
      memset(buffer, 0xff, sizeof(buffer));
      agc_fill_nops(buffer, sizes[s]);
      uint32_t at = 0;
      while (at < sizes[s]) {
         uint32_t header = buffer[at];
         CHECK_EQ(header >> 30, 3);
         CHECK_EQ((header >> 8) & 0xff, PKT3_NOP);
         uint32_t count = (header >> 16) & 0x3fff;
         uint32_t length = count == 0x3fff ? 1 : count + 2;
         for (uint32_t i = 1; i < length && at + i < sizes[s]; i++)
            CHECK_EQ(buffer[at + i], 0);
         at += length;
      }
      CHECK_EQ(at, sizes[s]);
      CHECK_EQ(buffer[sizes[s]], 0xffffffffu);
   }
}

static void
test_sizes(void)
{
   CHECK_EQ(sceAgcDcbSetCxRegisterDirectGetSize(), 12);
   CHECK_EQ(sceAgcDcbSetShRegisterDirectGetSize(), 12);
   CHECK_EQ(sceAgcDcbSetUcRegisterDirectGetSize(), 12);
   CHECK_EQ(sceAgcCbSetShRegisterRangeDirectGetSize(5), 28);
   CHECK_EQ(sceAgcCbSetUcRegisterRangeDirectGetSize(1), 12);
   CHECK_EQ(sceAgcCbSetShRegistersDirectGetSize(3), 36);
   CHECK_EQ(sceAgcCbSetUcRegistersDirectGetSize(0), 0);
   CHECK_EQ(sceAgcCbSetShRegistersDirectGetSize(UINT32_MAX), 0);
   CHECK_EQ(sceAgcCbNopGetSize(7), 28);
   CHECK_EQ(sceAgcCbDispatchGetSize(), 20);
   CHECK_EQ(sceAgcCbCondWriteGetSize(), 36);
   CHECK_EQ(sceAgcCbQueueEndOfPipeActionGetSize(), 32);
   CHECK_EQ(sceAgcCbBranchGetSize(), 56);
}

static struct agc_reloc *
reloc_of(uint32_t *packet)
{
   return agc_record_reloc_at(agc_record_containing(packet), packet);
}

static void
test_packets(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   struct reference ref;
   enum amd_gfx_level level = host_chips[host].level;
   const uint64_t address = UINT64_C(0x00000034567890a0);
   const uint64_t label = UINT64_C(0x0000004567890ac0);
   const uint64_t mask = UINT64_C(0xff00ff00ff00ff00);
   const uint64_t wide = UINT64_C(0x1122334455667788);

   FOOTPRINT(sceAgcCbNop(cb, 1), sceAgcCbNopGetSize(1));
   EXPECT_WORDS(out, PKT3_NOP_PAD);
   FOOTPRINT(sceAgcCbNop(cb, 2), sceAgcCbNopGetSize(2));
   reference_begin(&ref);
   ac_emit_cp_nop(&ref.cs, 0);
   EXPECT_REFERENCE(out, ref, 2);
   FOOTPRINT(sceAgcCbNop(cb, 5), sceAgcCbNopGetSize(5));
   EXPECT_WORDS(out, PKT3(PKT3_NOP, 3, 0), 0, 0, 0, 0);
   EXPECT_PARSE_BOTH(out - 3, 8, "NOP");
   REJECTED(sceAgcCbNop(cb, 0));
   REJECTED(sceAgcCbNop(cb, 0x4002));
   REJECTED(sceAgcCbNop(NULL, 1));

   FOOTPRINT(sceAgcCbDispatch(cb, 8, 4, 2, 0x8001), sceAgcCbDispatchGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_DISPATCH_DIRECT, 3, 0) | PKT3_SHADER_TYPE_S(1), 8, 4, 2,
                S_00B800_COMPUTE_SHADER_EN(1) | S_00B800_ORDER_MODE(1) | S_00B800_CS_W32_EN(1));
   EXPECT_PARSE_BOTH(out, 5, "DISPATCH_DIRECT");
   FOOTPRINT(sceAgcCbDispatch(cb, 1, 1, 1, 0xa079), 20);
   CHECK_EQ(out[4], S_00B800_COMPUTE_SHADER_EN(1) | S_00B800_ORDER_MODE(1) |
                       S_00B800_ORDERED_APPEND_ENBL(1) | S_00B800_ORDERED_APPEND_MODE(1) |
                       S_00B800_USE_THREAD_DIMENSIONS(1) | S_00B800_TUNNEL_ENABLE(1) |
                       S_00B800_CS_W32_EN(1));
   REJECTED(sceAgcCbDispatch(cb, 1, 1, 1, 0x2));
   REJECTED(sceAgcCbDispatch(NULL, 1, 1, 1, 1));

   FOOTPRINT(sceAgcCbCondWrite(cb, 3, 1, address, 0xabcd, label, 7, 0xff),
             sceAgcCbCondWriteGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_COND_WRITE, 7, 0),
                S_451_FUNCTION(V_451_EQUAL_TO_THE_REFERENCE_VALUE) |
                   S_451_POLL_SPACE(V_451_MEMORY) | S_451_WRITE_SPACE(V_451_MEMORY),
                (uint32_t)label, (uint32_t)(label >> 32), 7, 0xff, (uint32_t)address,
                (uint32_t)(address >> 32), 0xabcd);
   EXPECT_PARSE_BOTH(out, 9, "COND_WRITE");
   REJECTED(sceAgcCbCondWrite(cb, 7, 1, address, 0, label, 0, 0));
   REJECTED(sceAgcCbCondWrite(cb, 3, 0, address, 0, label, 0, 0));
   REJECTED(sceAgcCbCondWrite(cb, 3, 3, address, 0, label, 0, 0));
   REJECTED(sceAgcCbCondWrite(cb, 3, 1, 0, 0, label, 0, 0));
   REJECTED(sceAgcCbCondWrite(cb, 3, 1, address, 0, label + 2, 0, 0));
   REJECTED(sceAgcCbCondWrite(cb, 3, 1, UINT64_C(1) << 48, 0, label, 0, 0));
   EXPECT_FATAL_TEXT(sceAgcCbCondWrite(cb, 3, 2, 4, 0, label, 0, 0), "scratch");

   FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_BOTTOM_OF_PIPE_TS, 0, 0, 0, label, 1, 0x1234, 0, 0, 3,
                                0),
             sceAgcCbQueueEndOfPipeActionGetSize());
   reference_begin(&ref);
   ac_emit_cp_release_mem(&ref.cs, level, AMD_IP_GFX, V_028A90_BOTTOM_OF_PIPE_TS, 0,
                          EOP_DST_SEL_MEM, EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM,
                          EOP_DATA_SEL_VALUE_32BIT, label, 0x1234, 0);
   EXPECT_REFERENCE(out, ref, 8);
   reference_begin(&ref);
   ac_emit_cp_release_mem(&ref.cs, level, AMD_IP_COMPUTE, V_028A90_BOTTOM_OF_PIPE_TS, 0,
                          EOP_DST_SEL_MEM, EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM,
                          EOP_DATA_SEL_VALUE_32BIT, label, 0x1234, 0);
   EXPECT_REFERENCE(out, ref, 8);
   EXPECT_PARSE_BOTH(out, 8, "RELEASE_MEM");
   CHECK(reloc_of(out) != NULL && reloc_of(out)->kind == AGC_RELOC_LABEL_WRITE && reloc_of(out)->flags == 3);

   FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_BOTTOM_OF_PIPE_TS, 0, 1, 3, label, 2, 1, 0, 1, 0, 0),
             32);
   EXPECT_WORDS(out, PKT3(PKT3_RELEASE_MEM, 6, 0),
                S_491_EVENT_TYPE(V_028A90_BOTTOM_OF_PIPE_TS) |
                   S_491_EVENT_INDEX(V_491_END_OF_PIPE) |
                   S_491_CACHE_POLICY(policy_bits(host, V_491_BYPASS)),
                S_492_DST_SEL(V_492_TC_L2) | S_492_DATA_SEL(V_492_SEND_64_BIT_DATA),
                (uint32_t)label, (uint32_t)(label >> 32), 1, 0, 0);
   EXPECT_PARSE_BOTH(out, 8, "RELEASE_MEM");

   uint32_t flush_flags = S_491_GL2_INV(1) | S_491_GL2_WB(1) | S_491_SEQ(V_587_SEQ_FORWARD);
   if (host != AGC_HOST_GFX12)
      flush_flags |= S_491_GLM_WB(1) | S_491_GLM_INV(1);
   FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT,
                                RELEASE_FLUSH_GL2 | RELEASE_ORDER_012, 0, 0, label, 1, 9, 0, 0, 3,
                                0),
             32);
   reference_begin(&ref);
   ac_emit_cp_release_mem(&ref.cs, level, AMD_IP_GFX, V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT,
                          flush_flags, EOP_DST_SEL_MEM, EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM,
                          EOP_DATA_SEL_VALUE_32BIT, label, 9, 0);
   EXPECT_REFERENCE(out, ref, 8);
   EXPECT_PARSE_BOTH(out, 8, "RELEASE_MEM");

   FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_CACHE_FLUSH_TS, 0x10c, 0, 0, label, 0, 0, 0, 0, 0, 0),
             32);
   CHECK_EQ(out[1], S_491_EVENT_TYPE(V_028A90_CACHE_FLUSH_TS) |
                       S_491_EVENT_INDEX(V_491_END_OF_PIPE) | S_491_GLV_INV(1) |
                       S_491_GL1_INV(host != AGC_HOST_GFX12) | S_491_GL2_INV(1) | S_491_GL2_WB(1));
   EXPECT_PARSE_BOTH(out, 8, "RELEASE_MEM");

   FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_BOTTOM_OF_PIPE_TS, 0, 0, 0, label, 3, 0, 0, 0, 0, 0),
             32);
   reference_begin(&ref);
   ac_emit_cp_release_mem(&ref.cs, level, AMD_IP_GFX, V_028A90_BOTTOM_OF_PIPE_TS, 0,
                          EOP_DST_SEL_MEM, EOP_INT_SEL_NONE, EOP_DATA_SEL_TIMESTAMP, label, 0, 0);
   EXPECT_REFERENCE(out, ref, 8);
   EXPECT_PARSE_BOTH(out, 8, "RELEASE_MEM");

   FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_CS_DONE, 0, 0, 0, label, 1, 5, 0, 0, 3, 0), 32);
   reference_begin(&ref);
   ac_emit_cp_release_mem(&ref.cs, level, AMD_IP_COMPUTE, V_028A90_CS_DONE, 0, EOP_DST_SEL_MEM,
                          EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM, EOP_DATA_SEL_VALUE_32BIT, label,
                          5, 0);
   EXPECT_REFERENCE(out, ref, 8);
   EXPECT_PARSE_BOTH(out, 8, "RELEASE_MEM");
   FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_PS_DONE, 0x400, 0, 0, label, 1, 5, 0, 0, 3, 0), 32);
   CHECK_EQ(out[1], S_491_EVENT_TYPE(V_028A90_PS_DONE) | S_491_EVENT_INDEX(V_491_SHADER_DONE));
   if (host == AGC_HOST_GFX12) {
      FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_CS_DONE, 0x200, 0, 0, label, 1, 5, 0, 0, 3, 0), 32);
      reference_begin(&ref);
      ac_emit_cp_release_mem(&ref.cs, level, AMD_IP_COMPUTE, V_028A90_CS_DONE, S_491_GL2_WB(1),
                             EOP_DST_SEL_MEM, EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM,
                             EOP_DATA_SEL_VALUE_32BIT, label, 5, 0);
      EXPECT_REFERENCE(out, ref, 8);
   } else {
      FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_CS_DONE, 0x200, 0, 0, label, 1, 5, 0, 0, 3, 0), 32);
      reference_begin(&ref);
      ac_emit_cp_release_mem(&ref.cs, level, AMD_IP_COMPUTE, V_028A90_BOTTOM_OF_PIPE_TS, S_491_GL2_WB(1),
                             EOP_DST_SEL_MEM, EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM,
                             EOP_DATA_SEL_VALUE_32BIT, label, 5, 0);
      EXPECT_REFERENCE(out, ref, 8);
      FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_PS_DONE, 0x2, 0, 0, label, 1, 5, 0, 0, 3, 0), 32);
      CHECK_EQ(G_491_EVENT_TYPE(out[1]), V_028A90_BOTTOM_OF_PIPE_TS);
      CHECK_EQ(out[1] & S_491_EVENT_INDEX(~0u), S_491_EVENT_INDEX(V_491_END_OF_PIPE));
   }

   FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_BOTTOM_OF_PIPE_TS, 0, 1, 0, label, 2, 0x55, 0, 1, 2,
                                0x1234),
             32);
   EXPECT_WORDS(out, PKT3(PKT3_RELEASE_MEM, 6, 0),
                S_491_EVENT_TYPE(V_028A90_BOTTOM_OF_PIPE_TS) | S_491_EVENT_INDEX(V_491_END_OF_PIPE),
                S_492_DST_SEL(V_492_TC_L2) | S_492_INT_SEL(V_492_SEND_DATA_AND_WRITE_CONFIRM) |
                   S_492_DATA_SEL(V_492_SEND_64_BIT_DATA),
                (uint32_t)label, (uint32_t)(label >> 32), 0x55, 0, 0);
   EXPECT_PARSE_BOTH(out, 8, "RELEASE_MEM");
   struct agc_reloc *reloc = reloc_of(out);
   CHECK(reloc && reloc->kind == AGC_RELOC_NOTIFY && reloc->flags == 2 && reloc->arg32 == 0x1234);
   CHECK(reloc && reloc->header == out[0]);

   FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_BOTTOM_OF_PIPE_TS, 0, 1, 0, 0, 0, 0, 0, 1, 4, 0x77),
             32);
   EXPECT_WORDS(out, PKT3(PKT3_RELEASE_MEM, 6, 0),
                S_491_EVENT_TYPE(V_028A90_BOTTOM_OF_PIPE_TS) | S_491_EVENT_INDEX(V_491_END_OF_PIPE),
                S_492_DST_SEL(V_492_TC_L2), 0, 0, 0, 0, 0);
   reloc = reloc_of(out);
   CHECK(reloc && reloc->kind == AGC_RELOC_NOTIFY && reloc->flags == 4 && reloc->arg32 == 0x77);
   FOOTPRINT(sceAgcCbReleaseMem(cb, V_028A90_BOTTOM_OF_PIPE_TS, 0, 1, 0, label, 0, 0, 0, 1, 1,
                                0x10),
             32);
   CHECK_EQ(out[2], S_492_DST_SEL(V_492_TC_L2));
   reloc = reloc_of(out);
   CHECK(reloc && reloc->kind == AGC_RELOC_NOTIFY && reloc->flags == 1 && reloc->arg32 == 0x10);

   REJECTED(sceAgcCbReleaseMem(cb, 0x40, 0, 1, 0, label, 1, 0, 0, 1, 0, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0x1000, 1, 0, label, 1, 0, 0, 1, 0, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 2, 0, label, 1, 0, 0, 1, 0, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 4, label, 1, 0, 0, 1, 0, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 0, label, 4, 0, 0, 1, 0, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 0, label, 1, 0, 0, 1, 5, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 0, label, 1, 0, 0, 1, 0, 0x8000000));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 0, label, 0, 0, 0, 1, 4, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 0, label, 5, 1, 0, 1, 0, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 0, label, 1, 0, 4, 1, 0, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 0, label, 1, 0, 0, 2, 0, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 0, label + 2, 1, 0, 0, 1, 0, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 0, label + 4, 2, 0, 0, 1, 0, 0));
   REJECTED(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 0, label, 1, UINT64_C(1) << 32, 0, 1, 0, 0));
   EXPECT_FATAL_TEXT(sceAgcCbReleaseMem(cb, 0x28, 0, 1, 0, label, 5, 0, 4, 1, 0, 0), "GDS");
   EXPECT_FATAL_TEXT(sceAgcCbReleaseMem(cb, 0x29, 0, 1, 0, label, 1, 0, 0, 1, 0, 0),
                     "no host form");

   FOOTPRINT(sceAgcCbBranch(cb, 2, 3, label, mask, wide, 1, address, 64, 2, address + 0x400, 32),
             sceAgcCbBranchGetSize());
   EXPECT_WORDS(out, PKT3(PKT3_COND_INDIRECT_BUFFER, 12, 0),
                S_3F1_MODE(V_3F1_IF_THEN_ELSE) | S_3F1_FUNCTION(V_3F1_EQUAL_TO_THE_REFERENCE_VALUE),
                S_3F2_COMPARE_ADDR_LO((uint32_t)label >> 3), (uint32_t)(label >> 32),
                (uint32_t)mask, (uint32_t)(mask >> 32), (uint32_t)wide, (uint32_t)(wide >> 32),
                S_3F8_IB_BASE1_LO((uint32_t)address >> 2), (uint32_t)(address >> 32),
                S_3F10_IB_SIZE1(64) | S_3F10_CACHE_POLICY1(policy_bits(host, V_3F10_STREAM)),
                S_3F11_IB_BASE2_LO((uint32_t)(address + 0x400) >> 2), (uint32_t)(address >> 32),
                S_3F13_IB_SIZE2(32) | S_3F13_CACHE_POLICY2(policy_bits(host, V_3F13_NOA)));
   EXPECT_PARSE_BOTH(out, 14, "INDIRECT_BUFFER");
   FOOTPRINT(sceAgcCbBranch(cb, 1, 0, 0, 0, 0, 0, address, 8, 0, 0, 0), 56);
   EXPECT_WORDS(out, PKT3(PKT3_COND_INDIRECT_BUFFER, 12, 0), S_3F1_MODE(V_3F1_IF_THEN), 0, 0, 0, 0,
                0, 0, (uint32_t)address, (uint32_t)(address >> 32), S_3F10_IB_SIZE1(8), 0, 0, 0);
   REJECTED(sceAgcCbBranch(cb, 0, 3, label, 0, 0, 0, address, 8, 0, 0, 0));
   REJECTED(sceAgcCbBranch(cb, 3, 3, label, 0, 0, 0, address, 8, 0, 0, 0));
   REJECTED(sceAgcCbBranch(cb, 1, 7, label, 0, 0, 0, address, 8, 0, 0, 0));
   REJECTED(sceAgcCbBranch(cb, 1, 3, label, 0, 0, 4, address, 8, 0, 0, 0));
   REJECTED(sceAgcCbBranch(cb, 1, 3, label, 0, 0, 0, address, 0x100000, 0, 0, 0));
   REJECTED(sceAgcCbBranch(cb, 1, 3, label + 4, 0, 0, 0, address, 8, 0, 0, 0));
   REJECTED(sceAgcCbBranch(cb, 1, 3, 0, 0, 0, 0, address, 8, 0, 0, 0));
   REJECTED(sceAgcCbBranch(cb, 1, 3, label, 0, 0, 0, address + 2, 8, 0, 0, 0));
   REJECTED(sceAgcCbBranch(cb, 2, 3, label, 0, 0, 0, address, 8, 0, 0, 4));
}

static void
test_patches(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   struct reference ref;
   enum amd_gfx_level level = host_chips[host].level;
   static const uint32_t data[2] = {0xaaaaaaaa, 0xbbbbbbbb};
   const uint64_t address = UINT64_C(0x00000034567890a0);
   const uint64_t other = UINT64_C(0x0000004567890ac0);
   const uint64_t wide = UINT64_C(0x1122334455667788);

   uint32_t *jump = sceAgcDcbJump(cb, 1, 0, 0, 0);
   uint32_t *exec = sceAgcDcbCondExec(cb, address, 3);
   CHECK_EQ(sceAgcCondExecPatchSetCommandAddress(exec, guest_pointer(other)), 0);
   reference_begin(&ref);
   ac_emit_cp_cond_exec(&ref.cs, level, other, 3);
   EXPECT_REFERENCE(exec, ref, 5);
   CHECK_EQ(sceAgcCondExecPatchSetEnd(exec, exec + 5 + 40), 0);
   CHECK_EQ(sceAgcAsyncCondExecPatchSetCommandAddress(exec, guest_pointer(address)), 0);
   reference_begin(&ref);
   ac_emit_cp_cond_exec(&ref.cs, level, address, 40);
   EXPECT_REFERENCE(exec, ref, 5);
   CHECK_EQ(sceAgcAsyncCondExecPatchSetEnd(exec, exec + 5), 0);
   CHECK_EQ(exec[4], 0);
   EXPECT_PARSE_BOTH(exec, 5, "COND_EXEC");
   REFUSED(sceAgcCondExecPatchSetEnd(exec, exec + 4), AGC_PATCH_INVALID);
   REFUSED(sceAgcCondExecPatchSetEnd(exec, guest_pointer((uintptr_t)(exec + 5) + 0x10000)),
           AGC_PATCH_INVALID);
   REFUSED(sceAgcCondExecPatchSetEnd(exec, NULL), AGC_PATCH_INVALID);
   REFUSED(sceAgcCondExecPatchSetCommandAddress(exec, NULL), AGC_PATCH_INVALID);
   REFUSED(sceAgcCondExecPatchSetCommandAddress(exec, guest_pointer(other + 2)), AGC_PATCH_INVALID);
   REFUSED(sceAgcCondExecPatchSetCommandAddress(jump, guest_pointer(other)), AGC_PATCH_INVALID);
   REFUSED(sceAgcAsyncCondExecPatchSetCommandAddress(NULL, guest_pointer(other)),
           AGC_PATCH_INVALID);

   CHECK_EQ(sceAgcJumpPatchSetTarget_0300(jump, 0, guest_pointer(address), 77), 0);
   reference_begin(&ref);
   ac_emit_cp_indirect_buffer(&ref.cs, address, 77,
                              AC_CP_INDIRECT_BUFFER_CHAIN | AC_CP_INDIRECT_BUFFER_VALID, false);
   EXPECT_REFERENCE(jump, ref, 4);
   CHECK_EQ(sceAgcJumpPatchSetTarget(jump, guest_pointer(other), 5), 0);
   reference_begin(&ref);
   ac_emit_cp_indirect_buffer(&ref.cs, other, 5,
                              AC_CP_INDIRECT_BUFFER_CHAIN | AC_CP_INDIRECT_BUFFER_VALID, false);
   EXPECT_REFERENCE(jump, ref, 4);
   CHECK_EQ(sceAgcJumpPatchSetTarget_0300(jump, 2, guest_pointer(other), 6), 0);
   CHECK_EQ(jump[3], S_3F3_IB_SIZE(6) | S_3F3_CHAIN(1) | S_3F3_VALID(1) |
                        S_3F3_CACHE_POLICY(policy_bits(host, 2)));
   EXPECT_PARSE_BOTH(jump, 4, "INDIRECT_BUFFER");
   REFUSED(sceAgcJumpPatchSetTarget_0300(jump, 4, guest_pointer(other), 6), AGC_PATCH_INVALID);
   REFUSED(sceAgcJumpPatchSetTarget(jump, NULL, 6), AGC_PATCH_INVALID);
   REFUSED(sceAgcJumpPatchSetTarget(jump, guest_pointer(other + 1), 6), AGC_PATCH_INVALID);
   REFUSED(sceAgcJumpPatchSetTarget(jump, guest_pointer(other), 0x100000), AGC_PATCH_INVALID);
   REFUSED(sceAgcJumpPatchSetTarget(exec, guest_pointer(other), 6), AGC_PATCH_INVALID);

   uint32_t *branch = sceAgcCbBranch(cb, 2, 3, address, 1, 2, 0, 0, 0, 0, 0, 0);
   CHECK(branch != NULL);
   CHECK_EQ(sceAgcBranchPatchSetCompareAddress(branch, (const uint64_t *)(uintptr_t)other), 0);
   CHECK_EQ(sceAgcBranchPatchSetThenTarget_0300(branch, 3, guest_pointer(address), 10), 0);
   CHECK_EQ(sceAgcBranchPatchSetElseTarget_0300(branch, 0, guest_pointer(other), 20), 0);
   EXPECT_WORDS(branch, PKT3(PKT3_COND_INDIRECT_BUFFER, 12, 0),
                S_3F1_MODE(V_3F1_IF_THEN_ELSE) | S_3F1_FUNCTION(V_3F1_EQUAL_TO_THE_REFERENCE_VALUE),
                (uint32_t)other, (uint32_t)(other >> 32), 1, 0, 2, 0, (uint32_t)address,
                (uint32_t)(address >> 32),
                S_3F10_IB_SIZE1(10) | S_3F10_CACHE_POLICY1(policy_bits(host, V_3F10_BYPASS)),
                (uint32_t)other, (uint32_t)(other >> 32), S_3F13_IB_SIZE2(20));
   REFUSED(sceAgcBranchPatchSetCompareAddress(branch, (const uint64_t *)(uintptr_t)(other + 4)),
           AGC_PATCH_INVALID);
   REFUSED(sceAgcBranchPatchSetCompareAddress(jump, (const uint64_t *)(uintptr_t)other),
           AGC_PATCH_INVALID);
   REFUSED(sceAgcBranchPatchSetThenTarget_0300(branch, 4, guest_pointer(address), 10),
           AGC_PATCH_INVALID);
   REFUSED(sceAgcBranchPatchSetElseTarget_0300(branch, 0, NULL, 10), AGC_PATCH_INVALID);
   REFUSED(sceAgcBranchPatchSetThenTarget_0300(jump, 0, guest_pointer(address), 10),
           AGC_PATCH_INVALID);
   REFUSED(sceAgcJumpPatchSetTarget(branch, guest_pointer(other), 6), AGC_PATCH_INVALID);

   uint32_t *dma = sceAgcDcbDmaData(cb, 0, DMA_GL2, 0, 0, DMA_GL2, 0, 0, 64, 0, 0, 0);
   CHECK_EQ(sceAgcDmaDataPatchSetDstAddressOrOffset(dma, address), 0);
   CHECK_EQ(sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate(dma, other), 0);
   EXPECT_WORDS(dma, PKT3(PKT3_DMA_DATA, 5, 0),
                S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2) | S_501_DST_SEL(V_501_DST_ADDR_USING_L2),
                (uint32_t)other, (uint32_t)(other >> 32), (uint32_t)address,
                (uint32_t)(address >> 32), S_506_BYTE_COUNT(64));
   EXPECT_PARSE_BOTH(dma, 7, "DMA_DATA");
   REFUSED(sceAgcDmaDataPatchSetDstAddressOrOffset(jump, address), AGC_PATCH_INVALID);
   REFUSED(sceAgcDmaDataPatchSetSrcAddressOrOffsetOrImmediate(NULL, address), AGC_PATCH_INVALID);

   uint32_t *release = sceAgcCbReleaseMem(cb, V_028A90_BOTTOM_OF_PIPE_TS, 0, 0, 0, 0, 1, 0, 0, 0, 3,
                                          0);
   CHECK_EQ(sceAgcQueueEndOfPipeActionPatchAddress(release, (const uint64_t *)(uintptr_t)other), 0);
   CHECK_EQ(sceAgcQueueEndOfPipeActionPatchData(release, 0x4321), 0);
   reference_begin(&ref);
   ac_emit_cp_release_mem(&ref.cs, level, AMD_IP_GFX, V_028A90_BOTTOM_OF_PIPE_TS, 0,
                          EOP_DST_SEL_MEM, EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM,
                          EOP_DATA_SEL_VALUE_32BIT, other, 0x4321, 0);
   EXPECT_REFERENCE(release, ref, 8);
   uint32_t flush_flags = S_491_GL2_INV(1) | S_491_GL2_WB(1) | S_491_SEQ(V_587_SEQ_FORWARD);
   if (host != AGC_HOST_GFX12)
      flush_flags |= S_491_GLM_WB(1) | S_491_GLM_INV(1);
   CHECK_EQ(sceAgcQueueEndOfPipeActionPatchType(release, V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT), 0);
   CHECK_EQ(sceAgcQueueEndOfPipeActionPatchGcrCntl(release, RELEASE_FLUSH_GL2 | RELEASE_ORDER_012),
            0);
   reference_begin(&ref);
   ac_emit_cp_release_mem(&ref.cs, level, AMD_IP_GFX, V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT,
                          flush_flags, EOP_DST_SEL_MEM, EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM,
                          EOP_DATA_SEL_VALUE_32BIT, other, 0x4321, 0);
   EXPECT_REFERENCE(release, ref, 8);
   EXPECT_PARSE_BOTH(release, 8, "RELEASE_MEM");
   REFUSED(sceAgcQueueEndOfPipeActionPatchData(release, UINT64_C(1) << 32), AGC_PATCH_INVALID);
   REFUSED(sceAgcQueueEndOfPipeActionPatchGcrCntl(release, 0x1000), AGC_PATCH_INVALID);
   REFUSED(sceAgcQueueEndOfPipeActionPatchType(release, 0x40), AGC_PATCH_INVALID);
   REFUSED(sceAgcQueueEndOfPipeActionPatchAddress(release, NULL), AGC_PATCH_INVALID);
   REFUSED(sceAgcQueueEndOfPipeActionPatchAddress(release,
                                                  (const uint64_t *)(uintptr_t)(other + 2)),
           AGC_PATCH_INVALID);
   REFUSED(sceAgcQueueEndOfPipeActionPatchType(jump, V_028A90_BOTTOM_OF_PIPE_TS),
           AGC_PATCH_INVALID);
   if (host == AGC_HOST_GFX12) {
      CHECK_EQ(sceAgcQueueEndOfPipeActionPatchType(release, V_028A90_CS_DONE), 0);
      CHECK_EQ(G_491_EVENT_TYPE(release[1]), V_028A90_CS_DONE);
      CHECK_EQ(release[1] & S_491_EVENT_INDEX(~0u), S_491_EVENT_INDEX(V_491_SHADER_DONE));
   } else {
      CHECK_EQ(sceAgcQueueEndOfPipeActionPatchType(release, V_028A90_CS_DONE), 0);
      CHECK_EQ(G_491_EVENT_TYPE(release[1]), V_028A90_BOTTOM_OF_PIPE_TS);
      CHECK_EQ(release[1] & S_491_EVENT_INDEX(~0u), S_491_EVENT_INDEX(V_491_END_OF_PIPE));
      CHECK_EQ(release[1] & flush_flags, flush_flags);
   }
   CHECK_EQ(sceAgcQueueEndOfPipeActionPatchGcrCntl(release, 0), 0);
   CHECK_EQ(sceAgcQueueEndOfPipeActionPatchType(release, V_028A90_PS_DONE), 0);
   CHECK_EQ(release[1], S_491_EVENT_TYPE(V_028A90_PS_DONE) | S_491_EVENT_INDEX(V_491_SHADER_DONE));
   CHECK_EQ(sceAgcQueueEndOfPipeActionPatchGcrCntl(release, RELEASE_FLUSH_GL2), 0);
   if (host != AGC_HOST_GFX12) {
      CHECK_EQ(G_491_EVENT_TYPE(release[1]), V_028A90_BOTTOM_OF_PIPE_TS);
      CHECK_EQ(release[1] & S_491_EVENT_INDEX(~0u), S_491_EVENT_INDEX(V_491_END_OF_PIPE));
   } else {
      CHECK_EQ(G_491_EVENT_TYPE(release[1]), V_028A90_PS_DONE);
   }

   uint32_t *rewind_packet = sceAgcDcbRewind(cb, 0);
   CHECK_EQ(sceAgcRewindPatchSetRewindState(rewind_packet, 1), 0);
   CHECK_EQ(rewind_packet[1], S_591_VALID(1));
   CHECK_EQ(sceAgcAsyncRewindPatchSetRewindState(rewind_packet, 0), 0);
   CHECK_EQ(rewind_packet[1], 0);
   REFUSED(sceAgcRewindPatchSetRewindState(rewind_packet, 2), AGC_PATCH_INVALID);
   REFUSED(sceAgcAsyncRewindPatchSetRewindState(jump, 1), AGC_PATCH_INVALID);

   uint32_t *wait32 = sceAgcDcbWaitRegMem(cb, 0, 3, 0, 0, 0, 0, 0, 64);
   CHECK_EQ(sceAgcWaitRegMemPatchAddress(wait32, (const void *)(uintptr_t)address), 0);
   CHECK_EQ(sceAgcWaitRegMemPatchReference(wait32, 0x1234), 0);
   CHECK_EQ(sceAgcWaitRegMemPatchMask(wait32, 0xffffffff), 0);
   reference_begin(&ref);
   ac_emit_cp_wait_mem(&ref.cs, address, 0x1234, 0xffffffff, WAIT_REG_MEM_EQUAL);
   EXPECT_REFERENCE(wait32, ref, 7);
   CHECK_EQ(sceAgcWaitRegMemPatchCompareFunction(wait32, 5), 0);
   reference_begin(&ref);
   ac_emit_cp_wait_mem(&ref.cs, address, 0x1234, 0xffffffff, WAIT_REG_MEM_GREATER_OR_EQUAL);
   EXPECT_REFERENCE(wait32, ref, 7);
   EXPECT_NOPS(wait32 + 7, 7);
   EXPECT_PARSE(AMD_IP_GFX, wait32, 14, "WAIT_REG_MEM");
   REFUSED(sceAgcWaitRegMemPatchCompareFunction(wait32, 7), AGC_PATCH_INVALID);
   REFUSED(sceAgcWaitRegMemPatchAddress(jump, NULL), AGC_PATCH_INVALID);
   REFUSED(sceAgcWaitRegMemPatchReference(jump, 0), AGC_PATCH_INVALID);
   REFUSED(sceAgcWaitRegMemPatchMask(NULL, 0), AGC_PATCH_INVALID);

   uint32_t *wait64 = sceAgcAcbWaitRegMem(cb, 1, 3, 0, 0, 0, UINT64_MAX, 64);
   CHECK_EQ(sceAgcWaitRegMemPatchAddress(wait64, (const void *)(uintptr_t)(address + 4)), 0);
   CHECK_EQ(sceAgcWaitRegMemPatchReference(wait64, wide), 0);
   CHECK_EQ(sceAgcWaitRegMemPatchMask(wait64, 0x0f0f0f0f), 0);
   EXPECT_WORDS(wait64, PKT3(PKT3_WAIT_REG_MEM64, 7, 0),
                S_931_FUNCTION(V_931_EQUAL_TO_THE_REFERENCE_VALUE) |
                   S_931_MEM_SPACE(V_931_MEMORY_SPACE) |
                   S_931_OPERATION(V_931_WAIT_MEM_PREEMPTABLE),
                (uint32_t)address, (uint32_t)(address >> 32), (uint32_t)wide,
                (uint32_t)(wide >> 32), 0x0f0f0f0f, 0xffffffff, S_938_POLL_INTERVAL(4));
   EXPECT_PARSE_NAMELESS(AMD_IP_COMPUTE, wait64, 16, "REFERENCE_HI");

   uint32_t *write = sceAgcDcbWriteData(cb, DST_GL2_PFP, 0, 0, data, 2, 0, 1);
   CHECK_EQ(sceAgcWriteDataPatchSetAddressOrOffset(write, address), 0);
   reference_begin(&ref);
   ac_emit_cp_write_data(&ref.cs, V_371_PREFETCH_PARSER, V_371_TC_L2, address, 2, data, false);
   EXPECT_REFERENCE(write, ref, 6);
   CHECK_EQ(sceAgcWriteDataPatchSetDst(write, DST_GL2_ME), 0);
   reference_begin(&ref);
   ac_emit_cp_write_data(&ref.cs, V_371_MICRO_ENGINE, V_371_TC_L2, address, 2, data, false);
   EXPECT_REFERENCE(write, ref, 6);
   CHECK_EQ(sceAgcWriteDataPatchSetCachePolicy(write, 3), 0);
   CHECK_EQ(write[1], S_371_DST_SEL(V_371_TC_L2) | S_371_WR_CONFIRM(1) |
                         S_371_CACHE_POLICY(policy_bits(host, V_371_BYPASS)));
   EXPECT_PARSE(AMD_IP_GFX, write, 6, "WRITE_DATA");
   REFUSED(sceAgcWriteDataPatchSetCachePolicy(write, 4), AGC_PATCH_INVALID);
   REFUSED(sceAgcWriteDataPatchSetDst(write, 0x20), AGC_PATCH_INVALID);
   REFUSED(sceAgcWriteDataPatchSetDst(write, 0), AGC_PATCH_INVALID);
   REFUSED(sceAgcWriteDataPatchSetAddressOrOffset(jump, address), AGC_PATCH_INVALID);
   EXPECT_FATAL_TEXT(sceAgcWriteDataPatchSetDst(write, DST_GDS_ME), "no host form");

   uint32_t *async_write = sceAgcAcbWriteData(cb, ACB_GL2, 0, address, data, 1, 0, 0);
   CHECK_EQ(sceAgcAsyncWriteDataPatchSetAddressOrOffset(async_write, other), 0);
   CHECK_EQ(sceAgcAsyncWriteDataPatchSetCachePolicy(async_write, 1), 0);
   CHECK_EQ(sceAgcAsyncWriteDataPatchSetDst(async_write, V_371_MEMORY), 0);
   EXPECT_WORDS(async_write, PKT3(PKT3_WRITE_DATA, 3, 0),
                S_371_DST_SEL(V_371_MEMORY) | S_371_CACHE_POLICY(policy_bits(host, V_371_STREAM)),
                (uint32_t)other, (uint32_t)(other >> 32), data[0]);
   EXPECT_PARSE(AMD_IP_COMPUTE, async_write, 5, "WRITE_DATA");
   REFUSED(sceAgcAsyncWriteDataPatchSetDst(async_write, 0x10), AGC_PATCH_INVALID);
   REFUSED(sceAgcAsyncWriteDataPatchSetCachePolicy(jump, 1), AGC_PATCH_INVALID);
   EXPECT_FATAL_TEXT(sceAgcAsyncWriteDataPatchSetDst(async_write, ACB_GDS), "no host form");

   uint32_t *acquire = sceAgcDcbAcquireMem(cb, 0, 0, GCR_SCALAR, 0, 0, 400);
   CHECK_EQ(sceAgcAcquireMemSetEngine(acquire, 1), 0);
   reference_begin(&ref);
   ac_emit_cp_acquire_mem(&ref.cs, level, AMD_IP_GFX, V_581A_MICRO_ENGINE, S_587_GLK_INV(1), NULL,
                          NULL);
   EXPECT_REFERENCE(acquire, ref, 8);
   CHECK_EQ(sceAgcAcquireMemSetEngine(acquire, 0), 0);
   reference_begin(&ref);
   ac_emit_cp_acquire_mem(&ref.cs, level, AMD_IP_GFX, V_581A_PREFETCH_PARSER, S_587_GLK_INV(1),
                          NULL, NULL);
   EXPECT_REFERENCE(acquire, ref, 8);
   EXPECT_PARSE(AMD_IP_GFX, acquire, 8, "ACQUIRE_MEM");
   CHECK_EQ((uint32_t)sceAgcAcquireMemSetEngine(write, 1), (uint32_t)AGC_PATCH_NOT_ACQUIRE);
   REFUSED(sceAgcAcquireMemSetEngine(NULL, 1), AGC_PATCH_INVALID);

   uint32_t *draw = sceAgcDcbDrawIndex(cb, 3, address, DRAW_MODIFIER);
   CHECK_EQ(sceAgcSetPacketPredication(draw, 1), 0);
   CHECK_EQ(draw[0], PKT3(PKT3_DRAW_INDEX_2, 4, 1));
   EXPECT_PARSE(AMD_IP_GFX, draw, 6, "predicated");
   CHECK_EQ(sceAgcSetPacketPredication(draw, 0), 0);
   CHECK_EQ(draw[0], PKT3(PKT3_DRAW_INDEX_2, 4, 0));
   REFUSED(sceAgcSetPacketPredication(draw, 2), AGC_PATCH_INVALID);
   REFUSED(sceAgcSetPacketPredication(NULL, 1), AGC_PATCH_INVALID);
   CHECK_EQ(sceAgcSetNop(draw), 0);
   EXPECT_NOPS(draw, 6);
   CHECK_EQ(sceAgcSetNop(wait32), 0);
   EXPECT_NOPS(wait32, 14);
   EXPECT_PARSE(AMD_IP_GFX, wait32, 14, "NOP");
   uint32_t *pad = sceAgcCbNop(cb, 1);
   CHECK_EQ(sceAgcSetNop(pad), 0);
   CHECK_EQ(sceAgcSetPacketPredication(pad, 1), 0);
   CHECK_EQ(pad[0], PKT3_NOP_PAD);
   uint32_t type0 = 0;
   REFUSED(sceAgcSetNop(&type0), AGC_PATCH_INVALID);
   REFUSED(sceAgcSetNop(NULL), AGC_PATCH_INVALID);
   REFUSED(sceAgcSetPacketPredication(&type0, 1), AGC_PATCH_INVALID);

   uint32_t *notify = sceAgcCbReleaseMem(cb, V_028A90_BOTTOM_OF_PIPE_TS, 0, 1, 0, other, 2, 1, 0, 1,
                                         2, 9);
   struct agc_record *record = agc_record_containing(notify);
   struct agc_reloc *reloc = agc_record_reloc_at(record, notify);
   CHECK(record && reloc && agc_reloc_live(record, reloc, (uint32_t)(cb->up - cb->bottom)));
   CHECK_EQ(sceAgcSetNop(notify), 0);
   EXPECT_NOPS(notify, 8);
   CHECK(record && reloc && !agc_reloc_live(record, reloc, (uint32_t)(cb->up - cb->bottom)));

   uint32_t *start = cb->up;
   sceAgcDcbDrawIndexAuto(cb, 3, DRAW_MODIFIER);
   sceAgcCbNop(cb, 1);
   uint32_t *filler = agc_record_alloc(cb, 1);
   filler[0] = PKT2_NOP_PAD;
   uint32_t *ranged_wait = sceAgcDcbWaitRegMem(cb, 0, 3, 0, 0, address, 1, 1, 64);
   uint32_t *instances = sceAgcDcbSetNumInstances(cb, 2);
   uint32_t *end = cb->up;
   uint32_t saved[32];
   unsigned range = (unsigned)(end - start);
   CHECK(range <= 32);
   memcpy(saved, start, range * sizeof(uint32_t));
   CHECK_EQ(sceAgcSetRangePredication(start, end, 1), 0);
   CHECK_EQ(start[0], PKT3(PKT3_DRAW_INDEX_AUTO, 1, 1));
   CHECK_EQ(start[3], PKT3_NOP_PAD);
   CHECK_EQ(start[4], PKT2_NOP_PAD);
   CHECK_EQ(ranged_wait[0], PKT3(PKT3_WAIT_REG_MEM, 5, 1));
   CHECK_EQ(ranged_wait[7], PKT3(PKT3_NOP, 5, 1));
   CHECK_EQ(instances[0], PKT3(PKT3_NUM_INSTANCES, 0, 1));
   CHECK_EQ(sceAgcSetRangePredication(start, end, 0), 0);
   CHECK(memcmp(saved, start, range * sizeof(uint32_t)) == 0);
   CHECK_EQ(sceAgcSetRangePredication(start, start, 1), 0);
   CHECK(memcmp(saved, start, range * sizeof(uint32_t)) == 0);
   REFUSED(sceAgcSetRangePredication(start, end - 1, 1), AGC_PATCH_INVALID);
   CHECK_EQ(sceAgcSetRangePredication(start, end, 0), 0);
   CHECK(memcmp(saved, start, range * sizeof(uint32_t)) == 0);
   REFUSED(sceAgcSetRangePredication(end, start, 1), AGC_PATCH_INVALID);
   REFUSED(sceAgcSetRangePredication(NULL, end, 1), AGC_PATCH_INVALID);
   REFUSED(sceAgcSetRangePredication(start, end, 2), AGC_PATCH_INVALID);
}

static void
test_queries(enum agc_host host, struct agc_cb *cb)
{
   uint32_t *volatile before;
   uint32_t *out;
   const uint64_t address = UINT64_C(0x00000034567890a0);
   const uint64_t other = UINT64_C(0x0000004567890ac0);
   const uint64_t target = UINT64_C(0x0000004400010000);
   static const uint32_t zero[4] = {0, 0, 0, 0};
   static const uint32_t ones[4] = {~0u, ~0u, ~0u, ~0u};
   static const uint32_t mixed[4] = {1, 2, 3, 4};
   const uint32_t fill = S_501_SRC_SEL(V_501_DATA) | S_501_DST_SEL(V_501_DST_ADDR_USING_L2);

   uint32_t *branch = sceAgcCbBranch(cb, 1, 0, 0, 0, 0, 2, address, 8, 0, 0, 0);
   CHECK(branch != NULL);
   CHECK_EQ(sceAgcBranchPatchSetThenTarget(branch, guest_pointer(other), 11), 0);
   CHECK_EQ(sceAgcBranchPatchSetElseTarget(branch, guest_pointer(address), 21), 0);
   EXPECT_WORDS(branch + 8, (uint32_t)other, (uint32_t)(other >> 32),
                S_3F10_IB_SIZE1(11) | S_3F10_CACHE_POLICY1(policy_bits(host, 2)),
                (uint32_t)address, (uint32_t)(address >> 32), S_3F13_IB_SIZE2(21));
   REFUSED(sceAgcBranchPatchSetThenTarget(branch, NULL, 1), AGC_PATCH_INVALID);
   REFUSED(sceAgcBranchPatchSetElseTarget(branch, guest_pointer(other), 0x100000),
           AGC_PATCH_INVALID);
   REFUSED(sceAgcBranchPatchSetElseTarget(branch + 1, guest_pointer(other), 1), AGC_PATCH_INVALID);

   FOOTPRINT(sceAgcCbMemsetExclusive(cb, target, zero, 256), 28);
   EXPECT_WORDS(out, PKT3(PKT3_DMA_DATA, 5, 0), fill, 0, 0, (uint32_t)target,
                (uint32_t)(target >> 32), S_506_BYTE_COUNT(256));
   EXPECT_PARSE_BOTH(out, 7, "DMA_DATA");
   FOOTPRINT(sceAgcCbMemsetExclusive(cb, target, ones, AGC_MEMSET_CHUNK_BYTES + 512), 56);
   EXPECT_WORDS(out, PKT3(PKT3_DMA_DATA, 5, 0), fill, ~0u, 0, (uint32_t)target,
                (uint32_t)(target >> 32), S_506_BYTE_COUNT(AGC_MEMSET_CHUNK_BYTES),
                PKT3(PKT3_DMA_DATA, 5, 0), fill, ~0u, 0,
                (uint32_t)(target + AGC_MEMSET_CHUNK_BYTES),
                (uint32_t)((target + AGC_MEMSET_CHUNK_BYTES) >> 32), S_506_BYTE_COUNT(512));
   EXPECT_PARSE_BOTH(out, 14, "DMA_DATA");
   EXPECT_FATAL_TEXT(sceAgcCbMemsetExclusive(cb, target, mixed, 256), "no host form");
   REJECTED(sceAgcCbMemsetExclusive(cb, target, NULL, 256));
   REJECTED(sceAgcCbMemsetExclusive(cb, target + 64, zero, 256));
   REJECTED(sceAgcCbMemsetExclusive(cb, target, zero, 0));
   REJECTED(sceAgcCbMemsetExclusive(cb, target, zero, 300));
   const uint64_t too_large = (uint64_t)AGC_MEMSET_CHUNK_BYTES * (AGC_MEMSET_CHUNKS_MAX + 1);
   REJECTED(sceAgcCbMemsetExclusive(cb, target, zero, too_large));
   REJECTED(sceAgcCbMemsetExclusive(NULL, target, zero, 256));

   uint32_t *nop = sceAgcCbNop(cb, 6);
   uint32_t *pad = sceAgcCbNop(cb, 1);
   uint32_t type2 = PKT2_NOP_PAD;
   CHECK_EQ(sceAgcGetPacketSize(nop), 6);
   CHECK_EQ(sceAgcGetPacketSize(pad), 1);
   CHECK_EQ(sceAgcGetPacketSize(branch), 14);
   REFUSED(sceAgcGetPacketSize(&type2), 0);
   REFUSED(sceAgcGetPacketSize(NULL), 0);

   uint32_t *payload = NULL;
   CHECK_EQ(sceAgcGetDataPacketPayloadAddress_0090(&payload, nop, 0), 0);
   CHECK(payload == nop + 1);
   CHECK_EQ(sceAgcGetDataPacketPayloadAddress_0090(&payload, pad, 0), 0);
   CHECK(payload == NULL);
   CHECK_EQ(sceAgcGetDataPacketPayloadAddress_0090(&payload, nop, 1), 0);
   CHECK(payload == nop + 2);
   REFUSED(sceAgcGetDataPacketPayloadAddress_0090(NULL, nop, 0), AGC_ERROR_INVALID_ADDRESS);
   REFUSED(sceAgcGetDataPacketPayloadAddress_0090(&payload, NULL, 0), AGC_ERROR_INVALID_ADDRESS);

   struct agc_memory_range range = {0};
   CHECK_EQ(sceAgcGetDataPacketPayloadRange(&range, nop, 0), 0);
   CHECK(range.base == nop + 1 && range.size == 20);
   CHECK_EQ(sceAgcGetDataPacketPayloadRange(&range, pad, 0), 0);
   CHECK(range.base == NULL && range.size == 0);
   CHECK_EQ(sceAgcGetDataPacketPayloadRange(&range, nop, 1), 0);
   CHECK(range.base == nop + 2 && range.size == 16);
   REFUSED(sceAgcGetDataPacketPayloadRange(NULL, nop, 0), AGC_ERROR_INVALID_ADDRESS);
   REFUSED(sceAgcGetDataPacketPayloadRange(&range, (uint32_t *)((uintptr_t)nop + 2), 0),
           AGC_ERROR_INVALID_ADDRESS);
}

static void
test_host(enum agc_host host)
{
   static _Alignas(64) uint32_t buffer[2048];
   struct agc_cb cb;
   test_registers(host);
   agc_cb_init(&cb, buffer, sizeof(buffer), NULL, NULL);
   test_packets(host, &cb);
   test_patches(host, &cb);
   test_queries(host, &cb);
}

static void
test_large_nop(void)
{
   static _Alignas(64) uint32_t buffer[0x4100];
   struct agc_cb storage;
   struct agc_cb *cb = &storage;
   uint32_t *before;
   uint32_t *out;
   agc_cb_init(cb, buffer, sizeof(buffer), NULL, NULL);
   FOOTPRINT(sceAgcCbNop(cb, 0x4001), sceAgcCbNopGetSize(0x4001));
   EXPECT_NOPS(out, 0x4001);
   CHECK(sceAgcCbNop(cb, 0x4001) == NULL);
}

int
main(void)
{
   agc_set_fatal_handler(on_fatal);
   for (unsigned host = 0; host < AGC_HOST_COUNT; host++)
      test_host(host);
   test_nops();
   test_sizes();
   test_large_nop();
   agc_record_reset_all();
   return TEST_RESULT();
}
