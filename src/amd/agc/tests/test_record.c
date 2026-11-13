#include <setjmp.h>
#include <string.h>

#include "agc_cb_builders.h"
#include "agc_debug.h"
#include "agc_graphics.h"
#include "sid.h"
#include "test.h"

static jmp_buf fatal_jump;
static char fatal_text[512];

static void
on_fatal(const char *message)
{
   strncpy(fatal_text, message, sizeof(fatal_text) - 1);
   longjmp(fatal_jump, 1);
}

static bool
find_class(enum agc_reg_bank bank, enum agc_reg_class cls, uint32_t skip, uint32_t *offset)
{
   const struct agc_reg_bank_table *table = &agc_host_regs->bank[bank];
   for (uint32_t i = 0; i < table->count; i++) {
      if (table->entries[i].cls != cls)
         continue;
      if (skip--)
         continue;
      *offset = i;
      return true;
   }
   return false;
}

static void
test_segments(void)
{
   _Alignas(64) static uint32_t first[256];
   _Alignas(64) static uint32_t second[256];
   struct agc_cb a, b;
   agc_record_reset_all();
   agc_cb_init(&a, first, sizeof(first), NULL, NULL);
   agc_cb_init(&b, second, sizeof(second), NULL, NULL);

   uint32_t *p = agc_record_alloc(&a, 4);
   CHECK(p == first);
   struct agc_record *ra = agc_record_find(first);
   CHECK(ra != NULL && ra->bottom == first && ra->end == first + 4);
   CHECK_EQ(ra->generation, 0);
   CHECK_EQ(agc_record_count(), 1);
   CHECK(agc_record_cache.record == ra);

   CHECK(agc_record_alloc(&a, 6) == first + 4);
   CHECK(ra->end == first + 10);

   CHECK(agc_record_alloc(&b, 3) == second);
   struct agc_record *rb = agc_record_find(second);
   CHECK(rb != NULL && rb != ra && rb->end == second + 3);
   CHECK(agc_record_alloc(&a, 2) == first + 10);
   CHECK(ra->end == first + 12 && rb->end == second + 3);
   CHECK_EQ(agc_record_count(), 2);
   CHECK_EQ(ra->foreign_dwords, 0);

   a.up += 5;
   CHECK(agc_record_alloc(&a, 1) == first + 17);
   CHECK_EQ(ra->foreign_dwords, 5);
   CHECK(ra->end == first + 18);
   CHECK_EQ(ra->generation, 0);

   p[0] = 0xc0001000;
   struct agc_reloc *reloc = agc_record_add_reloc(first + 17, AGC_RELOC_NOTIFY);
   CHECK(reloc != NULL && ra->reloc_count == 1);

   agc_cb_reset(&a);
   CHECK(agc_record_alloc(&a, 8) == first);
   CHECK_EQ(ra->generation, 1);
   CHECK_EQ(ra->reloc_count, 0);
   CHECK_EQ(ra->foreign_dwords, 0);
   CHECK(ra->end == first + 8);

   agc_cb_reset(&a);
   a.up += 3;
   CHECK(agc_record_alloc(&a, 2) == first + 3);
   CHECK_EQ(ra->generation, 2);
   CHECK_EQ(ra->foreign_dwords, 3);

   CHECK(agc_record_containing(first + 4) == ra);
   CHECK(agc_record_containing(second + 1) == rb);
   CHECK(agc_record_containing(second + 3) == NULL);
   CHECK(agc_record_find(first + 1) == NULL);
}

static void
test_relocs(void)
{
   _Alignas(64) static uint32_t buffer[256];
   struct agc_cb cb;
   agc_record_reset_all();
   agc_cb_init(&cb, buffer, sizeof(buffer), NULL, NULL);

   uint32_t *packets[5];
   for (unsigned i = 0; i < 5; i++) {
      packets[i] = agc_record_alloc(&cb, 5);
      packets[i][0] = 0xc0030000 | i;
      struct agc_reloc *reloc = agc_record_add_reloc(packets[i], AGC_RELOC_REG_ARRAY);
      reloc->arg32 = i;
   }
   struct agc_record *record = agc_record_find(buffer);
   CHECK_EQ(record->reloc_count, 5);
   for (unsigned i = 0; i < 5; i++) {
      struct agc_reloc *reloc = agc_record_reloc_at(record, packets[i]);
      CHECK(reloc != NULL && reloc->offset == i * 5 && reloc->arg32 == i);
      CHECK(agc_reloc_live(record, reloc, 25));
      CHECK(agc_reloc_live(record, reloc, i * 5 + 1));
      CHECK(!agc_reloc_live(record, reloc, i * 5));
   }
   CHECK(agc_record_reloc_at(record, packets[2] + 1) == NULL);

   packets[3][0] = 0xffff1000;
   CHECK(!agc_reloc_live(record, agc_record_reloc_at(record, packets[3]), 25));
   packets[3][0] = 0xc0030003;

   agc_record_remove_reloc(record, packets[1]);
   CHECK_EQ(record->reloc_count, 4);
   CHECK(agc_record_reloc_at(record, packets[1]) == NULL);
   CHECK(agc_record_reloc_at(record, packets[4])->arg32 == 4);

   struct agc_reloc *again = agc_record_add_reloc(packets[2], AGC_RELOC_FLIP);
   CHECK_EQ(record->reloc_count, 2);
   CHECK(again->kind == AGC_RELOC_FLIP && again->offset == 10);
   CHECK(agc_record_reloc_at(record, packets[0])->arg32 == 0);

   for (unsigned i = 0; i < 40; i++) {
      uint32_t *packet = agc_record_alloc(&cb, 2);
      packet[0] = 0xc0001000;
      agc_record_add_reloc(packet, AGC_RELOC_NOTIFY);
   }
   CHECK_EQ(record->reloc_count, 42);
   for (unsigned i = 1; i < record->reloc_count; i++)
      CHECK(record->relocs[i - 1].offset < record->relocs[i].offset);
}

static void
test_indirect(enum agc_host host)
{
   _Alignas(64) static uint32_t buffer[256];
   _Alignas(8) static struct agc_register guest[4];
   _Alignas(8) static struct agc_register converted[4];
   static const uint8_t opcode[AGC_REG_BANK_COUNT] = {
      PKT3_LOAD_CONTEXT_REG_INDEX,
      PKT3_LOAD_SH_REG_INDEX,
      PKT3_LOAD_UCONFIG_REG_INDEX,
   };
   struct agc_cb cb;
   CHECK(agc_host_select(host));
   agc_record_reset_all();
   agc_cb_init(&cb, buffer, sizeof(buffer), NULL, NULL);

   for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++) {
      uint32_t pass[2];
      CHECK(find_class(bank, AGC_REG_PASS, 0, &pass[0]));
      CHECK(find_class(bank, AGC_REG_PASS, 5, &pass[1]));
      guest[0] = (struct agc_register){pass[1], 0x1111};
      guest[1] = (struct agc_register){pass[0], 0x2222};
      uint32_t count = 2;
      uint32_t moved;
      const struct agc_reg_entry *moved_entry = NULL;
      if (find_class(bank, AGC_REG_MOVE, 0, &moved)) {
         moved_entry = agc_reg_lookup(agc_host_regs, bank, moved);
         guest[count++] = (struct agc_register){moved, 0x3333};
      }

      uint32_t *out = agc_emit_set_registers_indirect(&cb, bank, guest, count, "test");
      CHECK(out != NULL);
      CHECK_EQ(out[0], PKT3(opcode[bank], 3, 0));
      CHECK_EQ(out[1], (uint32_t)(uintptr_t)guest);
      CHECK_EQ(out[2], (uint32_t)((uint64_t)(uintptr_t)guest >> 32));
      CHECK_EQ(out[3], 0x80000000u);
      CHECK_EQ(out[4], count);
      CHECK(cb.up == out + 5);

      struct agc_record *record = agc_record_find(buffer);
      struct agc_reloc *reloc = agc_record_reloc_at(record, out);
      CHECK(reloc != NULL && reloc->kind == AGC_RELOC_REG_ARRAY && reloc->bank == bank);
      CHECK(agc_reloc_registers(reloc) == guest && reloc->arg32 == count);
      CHECK(agc_reloc_live(record, reloc, (uint32_t)(cb.up - buffer)));

      uint32_t host_count = agc_reg_convert_pairs(bank, agc_reloc_registers(reloc), reloc->arg32, converted, "test");
      CHECK_EQ(host_count, count);
      CHECK(converted[0].offset == pass[1] && converted[0].value == 0x1111);
      CHECK(converted[1].offset == pass[0] && converted[1].value == 0x2222);
      if (moved_entry) {
         CHECK_EQ(converted[2].offset, moved_entry->host_offset);
         CHECK_EQ(converted[2].value, 0x3333);
      }

      uint64_t host_va = UINT64_C(0x123456789000);
      agc_resolve_registers_indirect(record, reloc, host_va, host_count);
      CHECK_EQ(out[0], PKT3(opcode[bank], 3, 0));
      CHECK_EQ(out[1], 0x56789000u);
      CHECK_EQ(out[2], 0x1234u);
      CHECK_EQ(out[3], 0x80000000u);
      CHECK_EQ(out[4], host_count);
      CHECK(agc_reloc_live(record, reloc, (uint32_t)(cb.up - buffer)));

      agc_resolve_registers_indirect(record, reloc, host_va, 0);
      CHECK_EQ(out[0], PKT3(PKT3_NOP, 3, 0));
      CHECK(agc_reloc_live(record, reloc, (uint32_t)(cb.up - buffer)));

      CHECK(agc_patch_registers_indirect(out, guest + 1, 1));
      CHECK_EQ(out[0], PKT3(opcode[bank], 3, 0));
      CHECK_EQ(out[1], (uint32_t)(uintptr_t)(guest + 1));
      CHECK_EQ(out[4], 1);
      CHECK(agc_reloc_registers(reloc) == guest + 1 && reloc->arg32 == 1);
      CHECK(!agc_patch_registers_indirect(out + 1, guest, 1));
      CHECK(!agc_patch_registers_indirect(out, (const struct agc_register *)((uintptr_t)guest + 2), 1));
   }

   uint32_t *up = cb.up;
   uint32_t *empty = agc_emit_set_registers_indirect(&cb, AGC_REG_BANK_CX, NULL, 0, "test");
   CHECK(empty == up && empty[4] == 0);
   up = cb.up;
   CHECK(agc_emit_set_registers_indirect(&cb, AGC_REG_BANK_CX, NULL, 1, "test") == NULL);
   CHECK(agc_emit_set_registers_indirect(&cb, AGC_REG_BANK_CX, guest, 0x4000, "test") == NULL);
   CHECK(agc_emit_set_registers_indirect(NULL, AGC_REG_BANK_CX, guest, 1, "test") == NULL);
   CHECK(cb.up == up);

   uint32_t pending;
   if (find_class(AGC_REG_BANK_CX, AGC_REG_PENDING, 0, &pending)) {
      guest[0] = (struct agc_register){pending, 1};
      fatal_text[0] = 0;
      if (setjmp(fatal_jump) == 0) {
         agc_reg_convert_pairs(AGC_REG_BANK_CX, guest, 1, converted, "test");
         CHECK(!"expected a fatal error");
      } else {
         CHECK(strstr(fatal_text, "no host conversion") != NULL);
      }
   }
}

int
main(void)
{
   agc_set_fatal_handler(on_fatal);
   CHECK(agc_host_select(AGC_HOST_GFX115));
   test_segments();
   test_relocs();
   for (unsigned host = 0; host < AGC_HOST_COUNT; host++)
      test_indirect(host);
   return TEST_RESULT();
}
