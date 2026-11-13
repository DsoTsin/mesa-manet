#include "agc_record.h"

#include <stdlib.h>
#include <string.h>

#include "agc_debug.h"
#include "util/u_atomic.h"

#define AGC_RECORD_SLOTS_INITIAL 8192u

struct record_table {
   uint32_t capacity;
   struct agc_record *slots[];
};

_Thread_local struct agc_record_cache agc_record_cache;

static struct record_table *record_table;
static uint32_t record_count;
static uint32_t record_lock;
static const struct agc_stub_source *stub_source;

static void
record_lock_take(void)
{
   while (p_atomic_cmpxchg(&record_lock, 0, 1) != 0)
      ;
}

static void
record_lock_drop(void)
{
   p_atomic_set(&record_lock, 0);
}

static uint32_t
record_hash(const uint32_t *bottom, uint32_t capacity)
{
   uint64_t key = (uint64_t)(uintptr_t)bottom >> 2;
   key *= UINT64_C(0x9e3779b97f4a7c15);
   return (uint32_t)(key >> 32) & (capacity - 1);
}

static struct record_table *
table_new(uint32_t capacity)
{
   struct record_table *table = calloc(1, sizeof(*table) + (size_t)capacity * sizeof(table->slots[0]));
   if (!table)
      agc_fatal("agc_record", "out of memory for a table of %u command segments", capacity);
   table->capacity = capacity;
   return table;
}

static struct record_table *
table_current(void)
{
   struct record_table *table = p_atomic_read(&record_table);
   if (table)
      return table;
   record_lock_take();
   if (!record_table)
      p_atomic_set(&record_table, table_new(AGC_RECORD_SLOTS_INITIAL));
   table = record_table;
   record_lock_drop();
   return table;
}

static struct agc_record *
table_find(struct record_table *table, const uint32_t *bottom)
{
   uint32_t mask = table->capacity - 1;
   for (uint32_t slot = record_hash(bottom, table->capacity);; slot = (slot + 1) & mask) {
      struct agc_record *record = p_atomic_read(&table->slots[slot]);
      if (!record)
         return NULL;
      if (record->bottom == bottom)
         return record;
   }
}

static void
table_put(struct record_table *table, struct agc_record *record)
{
   uint32_t mask = table->capacity - 1;
   uint32_t slot = record_hash(record->bottom, table->capacity);
   while (table->slots[slot])
      slot = (slot + 1) & mask;
   p_atomic_set(&table->slots[slot], record);
}

struct agc_record *
agc_record_find(const uint32_t *bottom)
{
   return table_find(table_current(), bottom);
}

struct agc_record *
agc_record_containing(const uint32_t *packet)
{
   struct agc_record *cached = agc_record_cache.record;
   if (cached && packet >= cached->bottom && packet < cached->end)
      return cached;
   struct record_table *table = table_current();
   for (uint32_t slot = 0; slot < table->capacity; slot++) {
      struct agc_record *record = p_atomic_read(&table->slots[slot]);
      if (record && packet >= record->bottom && packet < record->end)
         return record;
   }
   return NULL;
}

static struct agc_record *
record_get(uint32_t *bottom)
{
   struct agc_record *record = agc_record_find(bottom);
   if (record)
      return record;
   record_lock_take();
   struct record_table *table = p_atomic_read(&record_table);
   record = table_find(table, bottom);
   if (!record) {
      if ((uint64_t)(record_count + 1) * 2 > table->capacity) {
         struct record_table *grown = table_new(table->capacity * 2);
         for (uint32_t slot = 0; slot < table->capacity; slot++) {
            if (table->slots[slot])
               table_put(grown, table->slots[slot]);
         }
         p_atomic_set(&record_table, grown);
         table = grown;
      }
      record = calloc(1, sizeof(*record));
      if (!record)
         agc_fatal(__func__, "out of memory");
      record->bottom = bottom;
      record->end = bottom;
      table_put(table, record);
      p_atomic_inc(&record_count);
   }
   record_lock_drop();
   return record;
}

static void
stub_rewind(struct agc_record *record, uint32_t offset)
{
   uint32_t count = record->stub_mark_count;
   while (count && record->stub_marks[count - 1].offset >= offset)
      count--;
   if (count == record->stub_mark_count)
      return;
   record->stub_block = record->stub_marks[count].block;
   record->stub_used = record->stub_marks[count].used;
   record->stub_mark_count = count;
}

static void
stub_insert_block(struct agc_record *record, uint32_t index, uint64_t bytes)
{
   const struct agc_stub_source *source = p_atomic_read(&stub_source);
   if (!source)
      agc_fatal("agc_record_stub", "the driver has no memory for host command sequences");
   struct agc_stub_block fresh;
   memset(&fresh, 0, sizeof(fresh));
   if (!source->alloc(bytes, &fresh))
      agc_fatal("agc_record_stub", "no memory for 0x%llx bytes of host commands", (unsigned long long)bytes);
   if (record->stub_block_count == record->stub_block_capacity) {
      uint32_t capacity = record->stub_block_capacity ? record->stub_block_capacity * 2 : 4;
      struct agc_stub_block *blocks = realloc(record->stub_blocks, capacity * sizeof(*blocks));
      if (!blocks)
         agc_fatal("agc_record_stub", "out of memory");
      record->stub_blocks = blocks;
      record->stub_block_capacity = capacity;
   }
   memmove(record->stub_blocks + index + 1, record->stub_blocks + index,
           (record->stub_block_count - index) * sizeof(*record->stub_blocks));
   record->stub_blocks[index] = fresh;
   record->stub_block_count++;
}

void
agc_record_set_stub_source(const struct agc_stub_source *source)
{
   p_atomic_set(&stub_source, source);
}

uint32_t *
agc_record_stub(const uint32_t *packet, uint32_t dwords, uint64_t *va)
{
   struct agc_record *record = agc_record_cache.record;
   if (!record || packet < record->bottom || packet >= record->end)
      agc_fatal(__func__, "the packet is not in the segment being recorded on this thread");
   uint32_t offset = (uint32_t)(packet - record->bottom);
   stub_rewind(record, offset);

   uint64_t bytes = ((uint64_t)dwords * 4 + AGC_STUB_ALIGNMENT - 1) & ~(uint64_t)(AGC_STUB_ALIGNMENT - 1);
   uint32_t block = record->stub_block;
   uint64_t used = record->stub_used;
   if (block < record->stub_block_count && record->stub_blocks[block].bytes - used < bytes) {
      block++;
      used = 0;
   }
   if (block >= record->stub_block_count || record->stub_blocks[block].bytes < bytes) {
      stub_insert_block(record, block, bytes > AGC_STUB_BLOCK_BYTES ? bytes : AGC_STUB_BLOCK_BYTES);
      used = 0;
   }

   if (record->stub_mark_count == record->stub_mark_capacity) {
      uint32_t capacity = record->stub_mark_capacity ? record->stub_mark_capacity * 2 : 16;
      struct agc_stub_mark *marks = realloc(record->stub_marks, capacity * sizeof(*marks));
      if (!marks)
         agc_fatal(__func__, "out of memory");
      record->stub_marks = marks;
      record->stub_mark_capacity = capacity;
   }
   struct agc_stub_mark *mark = &record->stub_marks[record->stub_mark_count++];
   mark->offset = offset;
   mark->block = record->stub_block;
   mark->used = record->stub_used;

   record->stub_block = block;
   record->stub_used = used + bytes;
   const struct agc_stub_block *chosen = &record->stub_blocks[block];
   *va = chosen->va + used;
   return (uint32_t *)((uint8_t *)chosen->cpu + used);
}

uint32_t *
agc_record_stub_cpu(const uint32_t *packet, uint64_t va, uint32_t dwords)
{
   const struct agc_record *record = agc_record_containing(packet);
   uint64_t bytes = (uint64_t)dwords * 4;
   for (uint32_t i = 0; record && i < record->stub_block_count; i++) {
      const struct agc_stub_block *block = &record->stub_blocks[i];
      if (va >= block->va && va - block->va <= block->bytes && bytes <= block->bytes - (va - block->va))
         return (uint32_t *)((uint8_t *)block->cpu + (va - block->va));
   }
   return NULL;
}

uint32_t *
agc_record_alloc_slow(struct agc_cb *cb, uint32_t *out, uint32_t dwords)
{
   struct agc_record *record = record_get(cb->bottom);
   if (out < record->end) {
      stub_rewind(record, (uint32_t)(out - record->bottom));
      record->reloc_count = 0;
      record->foreign_dwords = 0;
      record->generation++;
      record->foreign_dwords += (uint32_t)(out - record->bottom);
   } else {
      record->foreign_dwords += (uint32_t)(out - record->end);
   }
   record->end = out + dwords;
   agc_record_cache.bottom = cb->bottom;
   agc_record_cache.record = record;
   return out;
}

bool
agc_record_claim(uint32_t *packet, uint32_t dwords)
{
   struct agc_record *record = agc_record_cache.record;
   if (!record || packet < record->bottom || packet > record->end)
      return false;
   if (packet + dwords > record->end)
      record->end = packet + dwords;
   return true;
}

struct agc_reloc *
agc_record_add_reloc(uint32_t *packet, enum agc_reloc_kind kind)
{
   struct agc_record *record = agc_record_cache.record;
   if (!record || packet < record->bottom || packet >= record->end)
      agc_fatal(__func__, "the packet is not in the segment being recorded on this thread");

   uint32_t offset = (uint32_t)(packet - record->bottom);
   while (record->reloc_count && record->relocs[record->reloc_count - 1].offset >= offset)
      record->reloc_count--;

   if (record->reloc_count == record->reloc_capacity) {
      uint32_t capacity = record->reloc_capacity ? record->reloc_capacity * 2 : 16;
      struct agc_reloc *relocs = realloc(record->relocs, capacity * sizeof(*relocs));
      if (!relocs)
         agc_fatal(__func__, "out of memory");
      record->relocs = relocs;
      record->reloc_capacity = capacity;
   }

   struct agc_reloc *reloc = &record->relocs[record->reloc_count++];
   memset(reloc, 0, sizeof(*reloc));
   reloc->offset = offset;
   reloc->header = packet[0];
   reloc->kind = (uint8_t)kind;
   return reloc;
}

struct agc_reloc *
agc_record_reloc_at(struct agc_record *record, const uint32_t *packet)
{
   if (!record || packet < record->bottom)
      return NULL;
   uint64_t offset = (uint64_t)(packet - record->bottom);
   uint32_t lo = 0;
   uint32_t hi = record->reloc_count;
   while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      if (record->relocs[mid].offset < offset)
         lo = mid + 1;
      else
         hi = mid;
   }
   if (lo < record->reloc_count && record->relocs[lo].offset == offset)
      return &record->relocs[lo];
   return NULL;
}

void
agc_record_remove_reloc(struct agc_record *record, const uint32_t *packet)
{
   struct agc_reloc *reloc = agc_record_reloc_at(record, packet);
   if (!reloc)
      return;
   uint32_t index = (uint32_t)(reloc - record->relocs);
   memmove(reloc, reloc + 1, (record->reloc_count - index - 1) * sizeof(*reloc));
   record->reloc_count--;
}

uint32_t
agc_record_count(void)
{
   return (uint32_t)p_atomic_read(&record_count);
}

void
agc_record_reset_all(void)
{
   table_current();
   record_lock_take();
   struct record_table *table = p_atomic_read(&record_table);
   for (uint32_t slot = 0; slot < table->capacity; slot++) {
      struct agc_record *record = p_atomic_read(&table->slots[slot]);
      if (record) {
         p_atomic_set(&table->slots[slot], NULL);
         const struct agc_stub_source *source = p_atomic_read(&stub_source);
         for (uint32_t i = 0; source && i < record->stub_block_count; i++)
            source->release(&record->stub_blocks[i]);
         free(record->stub_blocks);
         free(record->stub_marks);
         free(record->relocs);
         free(record);
      }
   }
   p_atomic_set(&record_count, 0);
   record_lock_drop();
   agc_record_cache.bottom = NULL;
   agc_record_cache.record = NULL;
}
