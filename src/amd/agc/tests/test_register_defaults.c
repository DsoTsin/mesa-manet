#include <string.h>

#include "agc_register_defaults.h"
#include "c11/threads.h"
#include "test.h"
#include "util/u_atomic.h"

#define ARRAY_COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define RACE_THREADS 8
#define RACE_VERSIONS (AGC_DEFAULTS_VERSION_MAX + 2)

struct race_thread {
   thrd_t thread;
   const void *blobs[RACE_VERSIONS][2];
};

struct set_shape {
   uint8_t version;
   uint16_t groups[2];
   uint8_t lengths[2][AGC_DEFAULTS_TABLES];
   uint32_t slot_bytes[2];
};

struct known_value {
   uint8_t version;
   bool internal;
   uint8_t bank;
   uint32_t type;
   uint16_t position;
   uint16_t offset;
   uint32_t value;
   bool repeated;
};

static const uint8_t version_tables[AGC_DEFAULTS_VERSION_MAX + 1] = {
   0, 0, 0, 0, 4, 5, 5, 7, 8, 9, 10, 11, 10,
};

static const uint32_t versions_beyond[] = {13, 14, 99, 0x7fffffffu, 0x80000000u, UINT32_MAX};

static const struct set_shape set_shapes[] = {
   {0, {150, 15}, {{80, 28, 42, 0}, {1, 12, 2, 0}}, {0x50000, 0x8000}},
   {4, {150, 19}, {{80, 28, 42, 0}, {2, 15, 2, 0}}, {0x50000, 0xa000}},
   {5, {156, 21}, {{80, 29, 47, 0}, {3, 15, 3, 0}}, {0x50000, 0xb000}},
   {7, {127, 22}, {{78, 29, 20, 0}, {4, 15, 3, 0}}, {0x41000, 0xc000}},
   {8, {127, 22}, {{78, 29, 20, 0}, {4, 15, 3, 0}}, {0x41000, 0xc000}},
   {9, {127, 22}, {{78, 29, 20, 0}, {4, 14, 1, 3}}, {0x41000, 0xc000}},
   {10, {128, 28}, {{79, 29, 20, 0}, {9, 15, 1, 3}}, {0x41000, 0xf000}},
   {11, {137, 24}, {{84, 32, 21, 0}, {8, 12, 1, 3}}, {0x46000, 0xd000}},
};

static const struct known_value known_values[] = {
   {8, false, AGC_REG_BANK_CX, 0xe24f806du, 0, 0x0202, 0x00cc0010u, false},
   {8, false, AGC_REG_BANK_CX, 0xf6c28182u, 0, 0x0109, 0x00000000u, false},
   {8, false, AGC_REG_BANK_CX, 0x9e5ad592u, 0, 0x008e, 0x0000000fu, false},
   {8, false, AGC_REG_BANK_CX, 0xbb513b98u, 0, 0x02dc, 0x0000aa00u, false},
   {8, false, AGC_REG_BANK_CX, 0xab64b23bu, 0, 0x0001, 0x00000000u, false},
   {8, false, AGC_REG_BANK_CX, 0x09afddafu, 0, 0x0206, 0x0000043fu, false},
   {8, false, AGC_REG_BANK_CX, 0x43707db8u, 0, 0x0083, 0x0000ffffu, false},
   {8, false, AGC_REG_BANK_CX, 0x1eb8d73au, 0, 0x0292, 0x00000002u, false},
   {8, false, AGC_REG_BANK_CX, 0x5cc72a74u, 0, 0x0282, 0x00000008u, false},
   {8, false, AGC_REG_BANK_CX, 0x3b77713cu, 0, 0x0281, 0xffff0000u, false},
   {8, false, AGC_REG_BANK_CX, 0x40f64410u, 0, 0x0280, 0x00080008u, false},
   {8, false, AGC_REG_BANK_CX, 0x2e418b83u, 0, 0x02de, 0x000001e9u, false},
   {8, false, AGC_REG_BANK_CX, 0xa00d0c8du, 0, 0x0205, 0x00000240u, false},
   {8, false, AGC_REG_BANK_CX, 0xb1289fb3u, 0, 0x020c, 0x00000001u, false},
   {8, false, AGC_REG_BANK_CX, 0x144832fbu, 0, 0x02f9, 0x0000002du, false},
   {8, false, AGC_REG_BANK_CX, 0x4b73ce27u, 0, 0x02ce, 0x00000400u, false},
   {8, false, AGC_REG_BANK_CX, 0x5f5a3e7bu, 0, 0x029b, 0x00000002u, false},
   {8, false, AGC_REG_BANK_CX, 0x6cf4f543u, 0, 0x0103, 0xffffffffu, false},
   {8, false, AGC_REG_BANK_CX, 0xc5831803u, 0, 0x02d4, 0x88101000u, false},
   {8, false, AGC_REG_BANK_CX, 0x5c6637cdu, 1, 0x030f, 0xffffffffu, false},
   {8, false, AGC_REG_BANK_CX, 0xcae3e690u, 0, 0x0311, 0x00000002u, false},
   {8, false, AGC_REG_BANK_CX, 0xcae3e690u, 1, 0x0312, 0x03ff0080u, false},
   {8, false, AGC_REG_BANK_CX, 0xef550356u, 7, 0x01e7, 0x20010001u, false},
   {8, false, AGC_REG_BANK_CX, 0x1f2d8149u, 1, 0x0085, 0x20002000u, false},
   {8, false, AGC_REG_BANK_CX, 0x1f2d8149u, 6, 0x008a, 0x00000000u, false},
   {8, false, AGC_REG_BANK_CX, 0x67096014u, 0, 0x0010, 0x80000000u, false},
   {8, false, AGC_REG_BANK_CX, 0x67096014u, 1, 0x0011, 0x20000180u, false},
   {8, false, AGC_REG_BANK_CX, 0x67096014u, 15, 0x000a, 0x00000000u, false},
   {8, false, AGC_REG_BANK_CX, 0x88f5e915u, 0, 0x00eb, 0xff00ff00u, false},
   {8, false, AGC_REG_BANK_CX, 0x88f5e915u, 1, 0x00ec, 0x00010000u, false},
   {8, false, AGC_REG_BANK_CX, 0x918106bbu, 1, 0x0091, 0x40004000u, false},
   {8, false, AGC_REG_BANK_CX, 0x95f0e7acu, 0, 0x02fa, 0x4e7e0000u, false},
   {8, false, AGC_REG_BANK_CX, 0x95f0e7acu, 3, 0x02fd, 0x3f800000u, false},
   {8, false, AGC_REG_BANK_CX, 0x38e92c91u, 4, 0x031e, 0x00000048u, false},
   {8, false, AGC_REG_BANK_CX, 0x38e92c91u, 15, 0x03b8, 0x08c6c000u, false},
   {8, false, AGC_REG_BANK_CX, 0x38e92c91u, 127, 0x03bf, 0x08c6c000u, false},
   {8, false, AGC_REG_BANK_CX, 0x0b177b43u, 1, 0x000d, 0x40004000u, false},
   {8, false, AGC_REG_BANK_CX, 0x7690af6fu, 0, 0x010f, 0x4e7e0000u, false},
   {8, false, AGC_REG_BANK_CX, 0x7690af6fu, 1, 0x0111, 0x3f800000u, false},
   {8, false, AGC_REG_BANK_CX, 0x7690af6fu, 6, 0x0094, 0x80000000u, false},
   {8, false, AGC_REG_BANK_CX, 0x7690af6fu, 7, 0x0095, 0x40004000u, false},
   {8, false, AGC_REG_BANK_CX, 0x7690af6fu, 157, 0x00b3, 0x40004000u, false},
   {8, false, AGC_REG_BANK_CX, 0x7690af6fu, 159, 0x00d3, 0x00000000u, false},
   {8, false, AGC_REG_BANK_CX, 0x078d7060u, 0, 0x0081, 0x80000000u, false},
   {8, false, AGC_REG_BANK_SH, 0x5d6e3ec7u, 0, 0x0212, 0x00000000u, false},
   {8, false, AGC_REG_BANK_SH, 0x9e5cfb8au, 0, 0x0107, 0xffff0000u, false},
   {8, false, AGC_REG_BANK_SH, 0x9e5cfb8au, 1, 0x0087, 0x0000ffffu, false},
   {8, false, AGC_REG_BANK_SH, 0x9e5cfb8au, 2, 0x0007, 0x0000ffffu, false},
   {8, false, AGC_REG_BANK_SH, 0xbdf02a4cu, 31, 0x002b, 0x00000000u, false},
   {8, false, AGC_REG_BANK_UC, 0x19e93e85u, 0, 0x041f, 0x00000000u, false},
   {8, false, AGC_REG_BANK_UC, 0xf6d8a76eu, 0, 0x0383, 0x40000040u, false},
   {8, false, AGC_REG_BANK_UC, 0x036ac8a6u, 2, 0x025e, 0x00000000u, false},
   {8, true, AGC_REG_BANK_CX, 0x8fb4edb5u, 0, 0x000e, 0x00000002u, false},
   {8, true, AGC_REG_BANK_CX, 0xb994ad29u, 0, 0x02af, 0x00040000u, false},
   {8, true, AGC_REG_BANK_CX, 0xd427322fu, 0, 0x0314, 0x00000202u, false},
   {8, true, AGC_REG_BANK_CX, 0xf58fea31u, 0, 0x01b5, 0x00000001u, false},
   {8, true, AGC_REG_BANK_SH, 0x6ac156efu, 0, 0x0216, 0xffffffffu, false},
   {8, true, AGC_REG_BANK_SH, 0xbe7dcd73u, 0, 0x027d, 0x000004ffu, false},
   {8, true, AGC_REG_BANK_SH, 0xd9e6d9f7u, 0, 0x0001, 0x00000003u, false},
   {8, true, AGC_REG_BANK_UC, 0x31f34b9fu, 0, 0x026c, 0x00000000u, false},
   {8, true, AGC_REG_BANK_UC, 0xac0f9e76u, 1, 0x0095, 0x00000000u, false},
   {8, true, AGC_REG_BANK_UC, 0x929fd95du, 2, 0x0262, 0x00000000u, false},
   {11, false, AGC_REG_BANK_CX, 0xe24f806du, 0, 0x0202, 0x00cc0010u, false},
   {11, false, AGC_REG_BANK_CX, 0xa6d12629u, 0, 0x01e0, 0x20010001u, false},
   {11, false, AGC_REG_BANK_CX, 0x9e5ad592u, 0, 0x008e, 0x00000000u, false},
   {11, false, AGC_REG_BANK_CX, 0xab64b23bu, 0, 0x0001, 0x11000100u, false},
   {11, false, AGC_REG_BANK_CX, 0x381c673cu, 0, 0x00db, 0x00002000u, true},
   {11, false, AGC_REG_BANK_CX, 0xf6ae26bau, 0, 0x0313, 0x00006000u, false},
   {11, false, AGC_REG_BANK_CX, 0x15051fa3u, 0, 0x0293, 0x06020000u, false},
   {11, false, AGC_REG_BANK_CX, 0x34a7d6d3u, 0, 0x0210, 0x04000000u, false},
   {11, false, AGC_REG_BANK_CX, 0xb1289fb3u, 0, 0x020c, 0x00000021u, false},
   {11, false, AGC_REG_BANK_CX, 0xd0de9ee6u, 0, 0x02d5, 0x00002000u, false},
   {11, false, AGC_REG_BANK_CX, 0xcae3e690u, 0, 0x0311, 0x01fd2002u, false},
   {11, false, AGC_REG_BANK_CX, 0xef550356u, 0, 0x01e0, 0x20010001u, false},
   {11, false, AGC_REG_BANK_CX, 0x1f2d8149u, 7, 0x008b, 0x20002000u, false},
   {11, false, AGC_REG_BANK_CX, 0x67096014u, 0, 0x0010, 0x80000180u, false},
   {11, false, AGC_REG_BANK_CX, 0x67096014u, 13, 0x02af, 0x00040000u, false},
   {11, false, AGC_REG_BANK_CX, 0x67096014u, 16, 0x000a, 0x00000000u, false},
   {11, false, AGC_REG_BANK_CX, 0x88f5e915u, 0, 0x00eb, 0xff00ff00u, false},
   {11, false, AGC_REG_BANK_SH, 0x5d6e3ec7u, 0, 0x0212, 0x00000000u, false},
   {11, false, AGC_REG_BANK_UC, 0x19e93e85u, 0, 0x041f, 0x00000000u, false},
   {11, true, AGC_REG_BANK_CX, 0x8fb4edb5u, 0, 0x000e, 0x00000002u, false},
   {11, true, AGC_REG_BANK_CX, 0x700e23c3u, 0, 0x02d3, 0x00000000u, false},
   {11, true, AGC_REG_BANK_CX, 0xd427322fu, 0, 0x0314, 0x00000202u, false},
   {11, true, AGC_REG_BANK_CX, 0xf58fea31u, 0, 0x01b5, 0x00000001u, false},
   {11, true, AGC_REG_BANK_CX, 0x1b1658dfu, 0, 0x0291, 0x00000000u, false},
   {11, true, AGC_REG_BANK_SH, 0x6ac153bau, 0, 0x021a, 0xffffffffu, false},
   {11, true, AGC_REG_BANK_SH, 0xbe7dcd73u, 0, 0x027d, 0x000004ffu, false},
   {11, true, AGC_REG_BANK_SH, 0x0c4a80efu, 0, 0x0100, 0x00ffffffu, false},
   {11, true, AGC_REG_BANK_SH, 0xe27f8fe6u, 0, 0x0081, 0x00000000u, false},
   {11, true, AGC_REG_BANK_SH, 0xd9e6d9f7u, 0, 0x0001, 0x00000003u, false},
   {11, true, AGC_REG_BANK_UC, 0x60289246u, 0, 0x0260, 0x00000000u, false},
   {11, true, AGC_REG_BANK_UC, 0x31f34b9fu, 0, 0x026c, 0x00000000u, false},
   {11, true, AGC_REG_BANK_UC, 0xac0f9e76u, 0, 0x0094, 0x00000000u, false},
   {11, true, AGC_REG_BANK_UC, 0x929fd95du, 1, 0x0278, 0x00000000u, false},
};

static int race_start;

static int
race_main(void *data)
{
   struct race_thread *self = data;
   while (!p_atomic_read(&race_start))
      thrd_yield();
   for (uint32_t version = 0; version < RACE_VERSIONS; version++) {
      for (unsigned internal = 0; internal < 2; internal++)
         self->blobs[version][internal] = agc_register_defaults_get(version, internal != 0);
   }
   return 0;
}

static void
test_threads(void)
{
   static struct race_thread threads[RACE_THREADS];
   unsigned started = 0;
   while (started < RACE_THREADS &&
          thrd_create(&threads[started].thread, race_main, &threads[started]) == thrd_success)
      started++;
   CHECK_EQ(started, RACE_THREADS);
   p_atomic_set(&race_start, 1);
   for (unsigned i = 0; i < started; i++)
      CHECK(thrd_join(threads[i].thread, NULL) == thrd_success);

   for (uint32_t version = 0; version < RACE_VERSIONS; version++) {
      for (unsigned internal = 0; internal < 2; internal++) {
         const void *blob = agc_register_defaults_get(version, internal != 0);
         CHECK(blob != NULL);
         for (unsigned i = 0; i < started; i++)
            CHECK(threads[i].blobs[version][internal] == blob);
      }
   }
}

static size_t
round_up(size_t value, size_t unit)
{
   return (value + unit - 1) / unit * unit;
}

static uint64_t
digest_word(uint64_t state, uint32_t value)
{
   for (unsigned shift = 0; shift < 32; shift += 8) {
      state ^= (value >> shift) & 0xff;
      state *= UINT64_C(0x100000001b3);
   }
   return state;
}

static const struct agc_register *
sdk_lookup(const struct agc_register_defaults *header, uint32_t type)
{
   for (uint32_t i = 0; i < header->count; i++) {
      uint32_t word = header->types[i].slot;
      if (header->types[i].type != type)
         continue;
      if (!header->tables[word & 3])
         return NULL;
      return header->tables[word & 3][(word & 0x3fc) >> 2];
   }
   return NULL;
}

static bool
is_block(uint32_t version, bool internal, const struct agc_register *block)
{
   uintptr_t base = (uintptr_t)agc_register_defaults_get(version, internal);
   uintptr_t at = (uintptr_t)block;
   size_t bytes = agc_register_defaults_bytes(version, internal);
   return at >= base && at - base <= bytes - AGC_DEFAULTS_BLOCK_BYTES;
}

static void
check_blob(uint32_t version, bool internal)
{
   int failures = agc_test_failures;
   const struct agc_register_defaults *header = agc_register_defaults_get(version, internal);
   const uint8_t *base = (const uint8_t *)header;
   uint32_t count = 0;
   const struct agc_defaults_group *groups =
      agc_register_defaults_groups(version, internal, &count);
   CHECK(header != NULL && groups != NULL && count != 0);
   if (agc_test_failures != failures)
      return;

   uint32_t lengths[AGC_DEFAULTS_TABLES] = {0};
   for (uint32_t i = 0; i < count; i++) {
      const struct agc_defaults_group *group = &groups[i];
      CHECK(group->table < AGC_DEFAULTS_TABLES && group->index < 0x100);
      CHECK(group->count >= 1 && group->count <= AGC_DEFAULTS_BLOCK_REGISTERS);
      CHECK(group->registers != NULL);
      if (agc_test_failures != failures)
         return;
      CHECK_EQ(group->bank, agc_register_defaults_table_bank(group->table));
      if (group->index >= lengths[group->table])
         lengths[group->table] = group->index + 1;
   }

   size_t table_at[AGC_DEFAULTS_TABLES];
   size_t at = 0x40;
   for (unsigned table = 0; table < AGC_DEFAULTS_TABLES; table++) {
      CHECK_EQ(agc_register_defaults_table_length(version, internal, table), lengths[table]);
      if (lengths[table])
         CHECK((const uint8_t *)header->tables[table] == base + at);
      else
         CHECK(header->tables[table] == NULL);
      table_at[table] = at;
      at += (size_t)lengths[table] * 8;
   }
   size_t types_at = round_up(at, 4);
   size_t blocks_at = round_up(types_at + (size_t)count * 12, 8);
   size_t bytes = blocks_at + (size_t)count * 2048;
   CHECK_EQ(agc_register_defaults_bytes(version, internal), bytes);
   CHECK((const uint8_t *)header->types == base + types_at);
   CHECK_EQ(header->count, count);
   CHECK_EQ(header->pad, 0);
   for (unsigned i = 0; i < 4; i++)
      CHECK_EQ(header->reserved[i], 0);
   for (size_t gap = types_at + (size_t)count * 12; gap < blocks_at; gap++)
      CHECK_EQ(base[gap], 0);
   if (agc_test_failures != failures)
      return;

   uint32_t filled = 0;
   for (unsigned table = 0; table < AGC_DEFAULTS_TABLES; table++) {
      const struct agc_register *const *entries = (const void *)(base + table_at[table]);
      for (uint32_t index = 0; index < lengths[table]; index++) {
         if (!entries[index])
            continue;
         size_t offset = (size_t)((const uint8_t *)entries[index] - base);
         CHECK(offset >= blocks_at && offset < bytes && (offset - blocks_at) % 2048 == 0);
         filled++;
      }
   }
   CHECK_EQ(filled, count);

   uint64_t digest = digest_word(UINT64_C(0xcbf29ce484222325), count);
   uint32_t padding = 0;
   for (uint32_t i = 0; i < count; i++) {
      const struct agc_defaults_group *group = &groups[i];
      const struct agc_defaults_type *entry = &header->types[i];
      const struct agc_register *block = (const void *)(base + blocks_at + (size_t)i * 2048);
      CHECK_EQ(entry->type, group->type);
      CHECK_EQ(entry->slot, agc_register_defaults_slot(group->table, group->index));
      CHECK_EQ(entry->slot, group->index * 4 + group->table);
      CHECK_EQ(entry->reserved, 0);
      CHECK(sdk_lookup(header, entry->type) == block);
      CHECK(agc_register_defaults_find(version, internal, entry->type) == group);
      CHECK(memcmp(block, group->registers, (size_t)group->count * sizeof(*block)) == 0);
      for (uint32_t r = group->count; r < AGC_DEFAULTS_BLOCK_REGISTERS; r++)
         padding |= block[r].offset | block[r].value;

      digest = digest_word(digest, agc_register_defaults_table_bank(entry->slot & 3));
      digest = digest_word(digest, entry->slot >> 2);
      digest = digest_word(digest, entry->type);
      digest = digest_word(digest, group->count);
      for (uint32_t r = 0; r < group->count; r++) {
         digest = digest_word(digest, block[r].offset);
         digest = digest_word(digest, block[r].value);
         CHECK(block[r].offset < agc_default_state_registers(group->bank));
      }
   }
   CHECK_EQ(padding, 0);
   CHECK_EQ(digest, agc_register_defaults_digest(version, internal));
}

static void
test_blobs(void)
{
   for (uint32_t version = 0; version <= AGC_DEFAULTS_VERSION_MAX; version++) {
      check_blob(version, false);
      check_blob(version, true);
   }
   for (unsigned i = 0; i < ARRAY_COUNT(versions_beyond); i++) {
      check_blob(versions_beyond[i], false);
      check_blob(versions_beyond[i], true);
   }
}

static uint32_t
count_registers(uint32_t version, bool internal)
{
   uint32_t count = 0;
   uint32_t registers = 0;
   const struct agc_defaults_group *groups =
      agc_register_defaults_groups(version, internal, &count);
   for (uint32_t i = 0; i < count; i++)
      registers += groups[i].count;
   return registers;
}

static void
test_shapes(void)
{
   for (unsigned i = 0; i < ARRAY_COUNT(set_shapes); i++) {
      const struct set_shape *shape = &set_shapes[i];
      for (unsigned kind = 0; kind < 2; kind++) {
         bool internal = kind != 0;
         const struct agc_register_defaults *header =
            agc_register_defaults_get(shape->version, internal);
         uint32_t count = 0;
         agc_register_defaults_groups(shape->version, internal, &count);
         CHECK_EQ(count, shape->groups[kind]);
         CHECK_EQ(header->count, shape->groups[kind]);
         for (unsigned table = 0; table < AGC_DEFAULTS_TABLES; table++) {
            CHECK_EQ(agc_register_defaults_table_length(shape->version, internal, table),
                     shape->lengths[kind][table]);
            CHECK_EQ(header->tables[table] != NULL, shape->lengths[kind][table] != 0);
         }
         CHECK(agc_register_defaults_bytes(shape->version, internal) <= shape->slot_bytes[kind]);
      }
   }
   CHECK_EQ(agc_register_defaults_table_length(8, false, AGC_DEFAULTS_TABLES), 0);
   CHECK_EQ(agc_register_defaults_bytes(8, false), 0x40230);
   CHECK_EQ(agc_register_defaults_bytes(8, true), 0xb1f8);
   CHECK_EQ(count_registers(8, false), 703);
   CHECK_EQ(count_registers(8, true), 25);

   uint32_t count = 0;
   const struct agc_defaults_group *groups = agc_register_defaults_groups(8, true, &count);
   CHECK_EQ(count, 22);
   CHECK_EQ(groups[0].bank, AGC_REG_BANK_CX);
   CHECK_EQ(groups[0].type, 0x8fb4edb5u);
   CHECK_EQ(groups[0].registers[0].offset, 0x00e);
   CHECK_EQ(groups[0].registers[0].value, 2);
   CHECK_EQ(groups[4].bank, AGC_REG_BANK_SH);
   CHECK_EQ(groups[4].registers[0].offset, 0x216);
   CHECK_EQ(groups[4].registers[0].value, 0xffffffffu);
}

static void
test_version_map(void)
{
   for (uint32_t version = 0; version <= AGC_DEFAULTS_VERSION_MAX; version++) {
      uint32_t table = version_tables[version];
      CHECK_EQ(agc_register_defaults_table_version(version), table);
      for (unsigned kind = 0; kind < 2; kind++) {
         const void *blob = agc_register_defaults_get(version, kind != 0);
         CHECK(blob == agc_register_defaults_get(table, kind != 0));
         CHECK(blob == agc_register_defaults_get(version, kind != 0));
      }
      CHECK(agc_register_defaults_get(version, false) != agc_register_defaults_get(version, true));
   }
   for (unsigned i = 0; i < ARRAY_COUNT(versions_beyond); i++) {
      uint32_t version = versions_beyond[i];
      uint32_t count = 0;
      CHECK_EQ(agc_register_defaults_table_version(version), AGC_DEFAULTS_VERSION_FALLBACK);
      CHECK_EQ(agc_register_defaults_table_version(version), 11);
      CHECK(agc_register_defaults_get(version, false) == agc_register_defaults_get(11, false));
      CHECK(agc_register_defaults_get(version, true) == agc_register_defaults_get(11, true));
      agc_register_defaults_groups(version, false, &count);
      CHECK_EQ(count, 137);
      agc_register_defaults_groups(version, true, &count);
      CHECK_EQ(count, 24);
   }
   for (unsigned a = 0; a < ARRAY_COUNT(set_shapes); a++) {
      for (unsigned b = a + 1; b < ARRAY_COUNT(set_shapes); b++) {
         for (unsigned kind = 0; kind < 2; kind++) {
            CHECK(agc_register_defaults_get(set_shapes[a].version, kind != 0) !=
                  agc_register_defaults_get(set_shapes[b].version, kind != 0));
         }
      }
   }
}

static void
test_known_values(void)
{
   for (unsigned i = 0; i < ARRAY_COUNT(known_values); i++) {
      const struct known_value *known = &known_values[i];
      int failures = agc_test_failures;
      const struct agc_register_defaults *header =
         agc_register_defaults_get(known->version, known->internal);
      const struct agc_defaults_group *group =
         agc_register_defaults_find(known->version, known->internal, known->type);
      const struct agc_register *block = sdk_lookup(header, known->type);
      CHECK(group != NULL && is_block(known->version, known->internal, block));
      if (agc_test_failures == failures) {
         CHECK_EQ(group->bank, known->bank);
         CHECK(known->position < group->count);
         CHECK_EQ(group->registers[known->position].offset, known->offset);
         CHECK_EQ(group->registers[known->position].value, known->value);
         CHECK_EQ(block[known->position].offset, known->offset);
         CHECK_EQ(block[known->position].value, known->value);
      }
      uint32_t value = ~known->value;
      CHECK(agc_register_defaults_value(known->version, known->internal, known->bank,
                                        known->offset, &value));
      if (!known->repeated)
         CHECK_EQ(value, known->value);
      if (agc_test_failures != failures)
         fprintf(stderr, "known value %u failed\n", i);
   }

   uint32_t value = 0x12345678;
   CHECK(agc_register_defaults_value(11, false, AGC_REG_BANK_CX, 0x00db, &value));
   CHECK_EQ(value, 0);
   value = 0x12345678;
   CHECK(!agc_register_defaults_value(8, false, AGC_REG_BANK_CX, 0x03ff, &value));
   CHECK(!agc_register_defaults_value(8, false, AGC_REG_BANK_CX, 0x01b5, &value));
   CHECK(!agc_register_defaults_value(8, false, AGC_REG_BANK_SH, 0x0202, &value));
   CHECK(!agc_register_defaults_value(8, false, AGC_REG_BANK_COUNT, 0x0202, &value));
   CHECK(!agc_register_defaults_value(8, true, AGC_REG_BANK_CX, 0x0202, &value));
   CHECK_EQ(value, 0x12345678);
   CHECK(agc_register_defaults_find(8, false, 0x8fb4edb5u) == NULL);
   CHECK(agc_register_defaults_find(8, true, 0xe24f806du) == NULL);
   CHECK(agc_register_defaults_find(8, false, 0) == NULL);

   int failures = agc_test_failures;
   const struct agc_register_defaults *internal = agc_register_defaults_get(11, true);
   const struct agc_register *first = sdk_lookup(internal, 0x60289246u);
   const struct agc_register *second = sdk_lookup(internal, 0x31f34b9fu);
   CHECK(internal->tables[AGC_REG_BANK_UC] != NULL);
   CHECK(internal->tables[AGC_DEFAULTS_TABLE_UC_EXTRA] != NULL);
   CHECK(is_block(11, true, first) && is_block(11, true, second) && first != second);
   if (agc_test_failures == failures) {
      CHECK(internal->tables[AGC_REG_BANK_UC][0] == first);
      CHECK(internal->tables[AGC_DEFAULTS_TABLE_UC_EXTRA][0] == second);
      CHECK_EQ(first->offset, 0x260);
      CHECK_EQ(second->offset, 0x26c);
   }
   CHECK_EQ(internal->types[0].type, 0x8fb4edb5u);
   CHECK_EQ(internal->types[0].slot, 0);
   CHECK_EQ(internal->types[8].slot, 1);
   CHECK_EQ(internal->types[20].slot, 2);
   CHECK_EQ(internal->types[21].slot, 3);
   CHECK_EQ(internal->types[23].slot, 11);
   CHECK(sdk_lookup(internal, 0xe24f806du) == NULL);
}

static void
test_default_state(void)
{
   static const uint32_t versions[] = {0, 8, 11, 13};
   static uint32_t state[AGC_DEFAULT_STATE_UC_REGISTERS + 2];

   CHECK_EQ(agc_default_state_registers(AGC_REG_BANK_CX), 0x400);
   CHECK_EQ(agc_default_state_registers(AGC_REG_BANK_SH), 0x400);
   CHECK_EQ(agc_default_state_registers(AGC_REG_BANK_UC), 0x2000);
   CHECK_EQ(agc_default_state_registers(AGC_REG_BANK_COUNT), 0);
   for (unsigned host = 0; host < AGC_HOST_COUNT; host++) {
      for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++)
         CHECK_EQ(agc_reg_tables_get(host)->bank[bank].count, agc_default_state_registers(bank));
   }

   for (unsigned v = 0; v < ARRAY_COUNT(versions); v++) {
      for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++) {
         uint32_t registers = agc_default_state_registers(bank);
         uint32_t wrong = 0;
         uint32_t set = 0;
         for (uint32_t i = 0; i < registers + 2; i++)
            state[i] = 0xdeadbeef;
         CHECK_EQ(agc_default_state_flat(versions[v], bank, state + 1, 0, registers), 0);
         CHECK_EQ(state[0], 0xdeadbeef);
         CHECK_EQ(state[registers + 1], 0xdeadbeef);
         for (uint32_t offset = 0; offset < registers; offset++) {
            uint32_t expected = 0;
            set += agc_register_defaults_value(versions[v], false, bank, offset, &expected);
            wrong += state[offset + 1] != expected;
         }
         CHECK_EQ(wrong, 0);
         CHECK(set > 40);
      }
   }

   uint32_t part[10];
   for (unsigned i = 0; i < ARRAY_COUNT(part); i++)
      part[i] = 0xdeadbeef;
   CHECK_EQ(agc_default_cx_state_flat(8, part + 1, 0x200, 0x208), 0);
   CHECK_EQ(part[0], 0xdeadbeef);
   CHECK_EQ(part[1], 0);
   CHECK_EQ(part[3], 0x00cc0010);
   CHECK_EQ(part[6], 0x00000240);
   CHECK_EQ(part[7], 0x0000043f);
   CHECK_EQ(part[8], 0);
   CHECK_EQ(part[9], 0xdeadbeef);
   CHECK_EQ(agc_default_cx_state_flat(8, part, 0x08e, 0x08f), 0);
   CHECK_EQ(part[0], 0x0000000f);
   CHECK_EQ(agc_default_cx_state_flat(11, part, 0x08e, 0x08f), 0);
   CHECK_EQ(part[0], 0);
   CHECK_EQ(agc_default_cx_state_flat(99, part, 0x001, 0x002), 0);
   CHECK_EQ(part[0], 0x11000100);
   CHECK_EQ(agc_default_cx_state_flat(8, part, 0x3bf, 0x3c1), 0);
   CHECK_EQ(part[0], 0x08c6c000);
   CHECK_EQ(part[1], 0);
   part[0] = 0xdeadbeef;
   CHECK_EQ(agc_default_cx_state_flat(8, part, 0x3ff, 0x400), 0);
   CHECK_EQ(part[0], 0);
   part[0] = 0xdeadbeef;
   CHECK_EQ(agc_default_cx_state_flat(8, part, 0x1b5, 0x1b6), 0);
   CHECK_EQ(part[0], 0);
   CHECK_EQ(agc_default_state_flat(8, AGC_REG_BANK_SH, part, 0x107, 0x108), 0);
   CHECK_EQ(part[0], 0xffff0000);
   CHECK_EQ(agc_default_state_flat(8, AGC_REG_BANK_UC, part, 0x383, 0x384), 0);
   CHECK_EQ(part[0], 0x40000040);
   CHECK_EQ(agc_default_state_flat(8, AGC_REG_BANK_UC, part, 0x1fff, 0x2000), 0);
   CHECK_EQ(part[0], 0);

   part[0] = 0xdeadbeef;
   CHECK_EQ(agc_default_cx_state_flat(8, NULL, 0, 0x400), AGC_DEFAULT_STATE_ERROR_ARGUMENT);
   CHECK_EQ(agc_default_cx_state_flat(8, part, 0, 0), AGC_DEFAULT_STATE_ERROR_RANGE);
   CHECK_EQ(agc_default_cx_state_flat(8, part, 5, 5), AGC_DEFAULT_STATE_ERROR_RANGE);
   CHECK_EQ(agc_default_cx_state_flat(8, part, 6, 5), AGC_DEFAULT_STATE_ERROR_RANGE);
   CHECK_EQ(agc_default_cx_state_flat(8, part, 0x400, 0x401), AGC_DEFAULT_STATE_ERROR_RANGE);
   CHECK_EQ(agc_default_cx_state_flat(8, part, 0x3ff, 0x401), AGC_DEFAULT_STATE_ERROR_RANGE);
   CHECK_EQ(agc_default_cx_state_flat(8, part, 0, UINT32_MAX), AGC_DEFAULT_STATE_ERROR_RANGE);
   CHECK_EQ(agc_default_cx_state_flat(8, part, UINT32_MAX, 0), AGC_DEFAULT_STATE_ERROR_RANGE);
   CHECK_EQ(agc_default_state_flat(8, AGC_REG_BANK_SH, part, 0x400, 0x401),
            AGC_DEFAULT_STATE_ERROR_RANGE);
   CHECK_EQ(agc_default_state_flat(8, AGC_REG_BANK_UC, part, 0x2000, 0x2001),
            AGC_DEFAULT_STATE_ERROR_RANGE);
   CHECK_EQ(agc_default_state_flat(8, AGC_REG_BANK_COUNT, part, 0, 1),
            AGC_DEFAULT_STATE_ERROR_ARGUMENT);
   CHECK_EQ(part[0], 0xdeadbeef);
   CHECK_EQ(AGC_DEFAULT_STATE_ERROR_ARGUMENT, (int32_t)0x8a6c000au);
   CHECK_EQ(AGC_DEFAULT_STATE_ERROR_RANGE, (int32_t)0x8a6c0044u);
}

int
main(void)
{
   test_threads();
   test_blobs();
   test_shapes();
   test_version_map();
   test_known_values();
   test_default_state();
   test_blobs();
   return TEST_RESULT();
}
