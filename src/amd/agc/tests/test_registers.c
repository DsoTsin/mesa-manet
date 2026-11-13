#include "agc_registers.h"
#include "test.h"

static bool
pass_bit(const struct agc_reg_bank_table *table, uint32_t offset)
{
   return (table->pass[offset >> 6] >> (offset & 63)) & 1;
}

static void
test_tables(enum agc_host host)
{
   const struct agc_reg_tables *tables = agc_reg_tables_get(host);
   CHECK(tables != NULL);
   if (!tables)
      return;

   for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++) {
      const struct agc_reg_bank_table *table = &tables->bank[bank];
      uint32_t counts[AGC_REG_CLASS_COUNT] = {0};
      CHECK(table->count != 0 && (table->count & 63) == 0);
      for (uint32_t offset = 0; offset < table->count; offset++) {
         const struct agc_reg_entry *entry = agc_reg_lookup(tables, bank, offset);
         CHECK(entry != NULL && entry->cls < AGC_REG_CLASS_COUNT);
         if (!entry || entry->cls >= AGC_REG_CLASS_COUNT)
            continue;
         counts[entry->cls]++;
         CHECK_EQ(pass_bit(table, offset), entry->cls == AGC_REG_PASS);
         switch (entry->cls) {
         case AGC_REG_UNDEFINED:
            CHECK(entry->host_offset == 0 && entry->extra == 0);
            break;
         case AGC_REG_PASS:
            CHECK_EQ(entry->host_offset, offset);
            break;
         case AGC_REG_MOVE:
            CHECK(entry->host_offset != offset);
            CHECK(entry->host_offset < table->count);
            break;
         case AGC_REG_FIELD:
            CHECK(entry->extra < agc_reg_field_prog_count());
            CHECK(entry->host_offset < table->count);
            break;
         default:
            CHECK(entry->host_offset < table->count);
            break;
         }
      }
      CHECK(agc_reg_lookup(tables, bank, table->count) == NULL);
      for (unsigned cls = 0; cls < AGC_REG_CLASS_COUNT; cls++) {
         uint32_t expected = cls == AGC_REG_UNDEFINED ? 0 : counts[cls];
         CHECK_EQ(agc_reg_class_count(host, bank, cls), expected);
      }
   }
}

static void
test_runs(enum agc_host host)
{
   const struct agc_reg_tables *tables = agc_reg_tables_get(host);
   for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++) {
      const struct agc_reg_bank_table *table = &tables->bank[bank];
      for (uint32_t first = 0; first < table->count; first++) {
         bool all = true;
         for (uint32_t count = 1; count <= 150 && first + count <= table->count; count++) {
            all = all && table->entries[first + count - 1].cls == AGC_REG_PASS;
            if (agc_reg_run_passes(tables, bank, first, count) != all) {
               CHECK_EQ(first, 0xffffffffu);
               CHECK_EQ(count, 0xffffffffu);
               return;
            }
         }
      }
      CHECK(!agc_reg_run_passes(tables, bank, 0, 0));
      CHECK(!agc_reg_run_passes(tables, bank, table->count, 1));
      CHECK(!agc_reg_run_passes(tables, bank, table->count - 1, 2));
      CHECK(!agc_reg_run_passes(tables, bank, 1, UINT32_MAX));
   }
}

static void
test_field_programs(void)
{
   for (uint32_t prog = 0; prog < agc_reg_field_prog_count(); prog++) {
      CHECK_EQ(agc_reg_field_apply(prog, 0), 0);
      uint32_t all = agc_reg_field_apply(prog, UINT32_MAX);
      for (unsigned bit = 0; bit < 32; bit++) {
         uint32_t one = agc_reg_field_apply(prog, UINT32_C(1) << bit);
         CHECK((one & (one - 1)) == 0);
         CHECK((one & ~all) == 0);
      }
   }
}

static void
test_known_registers(void)
{
   const struct agc_reg_tables *gfx103 = agc_reg_tables_get(AGC_HOST_GFX103);
   const struct agc_reg_entry *target_mask = agc_reg_lookup(gfx103, AGC_REG_BANK_CX, 0x08e);
   CHECK(target_mask && target_mask->cls == AGC_REG_PASS);
   const struct agc_reg_entry *hole = agc_reg_lookup(gfx103, AGC_REG_BANK_CX, 0x3ff);
   CHECK(hole && hole->cls == AGC_REG_UNDEFINED);
   CHECK(agc_reg_tables_get(AGC_HOST_COUNT) == NULL);
   CHECK_EQ(agc_reg_class_count(AGC_HOST_COUNT, AGC_REG_BANK_CX, AGC_REG_PASS), 0);

   for (unsigned host = 0; host < AGC_HOST_COUNT; host++) {
      uint32_t defined = 0;
      for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++) {
         for (unsigned cls = AGC_REG_PASS; cls < AGC_REG_CLASS_COUNT; cls++)
            defined += agc_reg_class_count(host, bank, cls);
      }
      CHECK(defined > 1800);
      CHECK(agc_reg_class_count(host, AGC_REG_BANK_CX, AGC_REG_PASS) > 100);
   }
}

int
main(void)
{
   for (unsigned host = 0; host < AGC_HOST_COUNT; host++) {
      test_tables(host);
      test_runs(host);
   }
   test_field_programs();
   test_known_registers();
   return TEST_RESULT();
}
