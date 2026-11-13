#ifndef AGC_RECORD_H
#define AGC_RECORD_H

#include "agc_cb.h"

enum agc_reloc_kind {
   AGC_RELOC_REG_ARRAY,
   AGC_RELOC_FLIP,
   AGC_RELOC_WAIT_RENDERING,
   AGC_RELOC_NOTIFY,
   AGC_RELOC_LABEL_WAIT,
   AGC_RELOC_LABEL_WRITE,
   AGC_RELOC_COND_EXEC,
   AGC_RELOC_DRAW_STATE,
   AGC_RELOC_CONTEXT_OP,
   AGC_RELOC_KIND_COUNT
};

enum agc_draw_state_kind {
   AGC_DRAW_STATE_NUM_INSTANCES,
   AGC_DRAW_STATE_INDEX_BASE,
   AGC_DRAW_STATE_INDEX_SIZE,
   AGC_DRAW_STATE_DRAW_BASE,
   AGC_DRAW_STATE_DISPATCH_BASE,
   AGC_DRAW_STATE_INDEX_ATTRIBUTES,
   AGC_DRAW_STATE_INDEX_TYPE,
   AGC_DRAW_STATE_COUNT
};

#define AGC_DRAW_STATE_MAX_DWORDS 4u

#define AGC_LABEL_WAIT_FUNCTION(flags) ((flags) & 7u)
#define AGC_LABEL_WAIT_64 (1u << 3)
#define AGC_LABEL_WAIT_FROM_PACKET (1u << 4)
#define AGC_LABEL_WRITE_ARG 0u

extern uint32_t agc_flip_records;
extern uintptr_t agc_flip_last;

struct agc_reloc {
   uint32_t offset;
   uint32_t header;
   uint8_t kind;
   uint8_t bank;
   uint16_t flags;
   uint32_t arg32;
   uint64_t arg64[2];
};

#define AGC_STUB_BLOCK_BYTES 4096u
#define AGC_STUB_ALIGNMENT 32u

struct agc_stub_block {
   void *cpu;
   uint64_t va;
   uint64_t bytes;
   void *handle;
};

struct agc_stub_source {
   bool (*alloc)(uint64_t bytes, struct agc_stub_block *block);
   void (*release)(struct agc_stub_block *block);
};

struct agc_stub_mark {
   uint32_t offset;
   uint32_t block;
   uint64_t used;
};

struct agc_record {
   uint32_t *bottom;
   uint32_t *end;
   uint32_t generation;
   uint32_t reloc_count;
   uint32_t reloc_capacity;
   uint32_t foreign_dwords;
   struct agc_reloc *relocs;
   struct agc_stub_block *stub_blocks;
   uint32_t stub_block_count;
   uint32_t stub_block_capacity;
   uint32_t stub_block;
   uint64_t stub_used;
   struct agc_stub_mark *stub_marks;
   uint32_t stub_mark_count;
   uint32_t stub_mark_capacity;
};

struct agc_record_cache {
   uint32_t *bottom;
   struct agc_record *record;
};

extern _Thread_local struct agc_record_cache agc_record_cache;

uint32_t *agc_record_alloc_slow(struct agc_cb *cb, uint32_t *out, uint32_t dwords);
struct agc_record *agc_record_find(const uint32_t *bottom);
struct agc_record *agc_record_containing(const uint32_t *packet);
bool agc_record_claim(uint32_t *packet, uint32_t dwords);
void agc_record_set_stub_source(const struct agc_stub_source *source);
uint32_t *agc_record_stub(const uint32_t *packet, uint32_t dwords, uint64_t *va);
uint32_t *agc_record_stub_cpu(const uint32_t *packet, uint64_t va, uint32_t dwords);
struct agc_reloc *agc_record_add_reloc(uint32_t *packet, enum agc_reloc_kind kind);
struct agc_reloc *agc_record_reloc_at(struct agc_record *record, const uint32_t *packet);
void agc_record_remove_reloc(struct agc_record *record, const uint32_t *packet);
uint32_t agc_record_count(void);
void agc_record_reset_all(void);

static inline uint32_t *
agc_record_alloc(struct agc_cb *cb, uint32_t dwords)
{
   uint32_t *out = agc_cb_alloc_up(cb, dwords);
   if (AGC_UNLIKELY(!out))
      return NULL;
   if (AGC_LIKELY(agc_record_cache.bottom == cb->bottom && agc_record_cache.record->end == out)) {
      agc_record_cache.record->end = out + dwords;
      return out;
   }
   return agc_record_alloc_slow(cb, out, dwords);
}

#define AGC_LABEL_WRITE_VALUE_KNOWN 1u

static inline void
agc_record_note_write(uint32_t *packet, uint32_t address_word, uint64_t address)
{
   struct agc_reloc *reloc = agc_record_add_reloc(packet, AGC_RELOC_LABEL_WRITE);
   reloc->flags = (uint16_t)address_word;
   reloc->arg64[0] = address;
}

static inline void
agc_record_note_write_value(uint32_t *packet, uint64_t address, uint64_t value)
{
   struct agc_reloc *reloc = agc_record_add_reloc(packet, AGC_RELOC_LABEL_WRITE);
   reloc->flags = AGC_LABEL_WRITE_ARG;
   reloc->arg32 = AGC_LABEL_WRITE_VALUE_KNOWN;
   reloc->arg64[0] = address;
   reloc->arg64[1] = value;
}

static inline void
agc_record_note_draw_state(uint32_t *packet, enum agc_draw_state_kind kind)
{
   struct agc_reloc *reloc = agc_record_add_reloc(packet, AGC_RELOC_DRAW_STATE);
   reloc->flags = (uint16_t)kind;
}

static inline bool
agc_reloc_live(const struct agc_record *record, const struct agc_reloc *reloc, uint32_t dwords)
{
   return reloc->offset < dwords && record->bottom[reloc->offset] == reloc->header;
}

#endif
