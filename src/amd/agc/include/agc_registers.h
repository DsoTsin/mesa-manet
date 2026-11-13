#ifndef AGC_REGISTERS_H
#define AGC_REGISTERS_H

#include "agc_types.h"

enum agc_reg_bank {
   AGC_REG_BANK_CX,
   AGC_REG_BANK_SH,
   AGC_REG_BANK_UC,
   AGC_REG_BANK_COUNT
};

enum agc_reg_class {
   AGC_REG_UNDEFINED,
   AGC_REG_PASS,
   AGC_REG_MOVE,
   AGC_REG_FIELD,
   AGC_REG_ADDR,
   AGC_REG_DROP,
   AGC_REG_SPLIT,
   AGC_REG_MERGE,
   AGC_REG_GUEST,
   AGC_REG_PENDING,
   AGC_REG_CLASS_COUNT
};

struct agc_reg_entry {
   uint16_t host_offset;
   uint8_t cls;
   uint16_t extra;
};

struct agc_field_move {
   uint8_t src;
   uint8_t dst;
   uint8_t width;
};

struct agc_field_prog {
   uint32_t keep;
   uint16_t first;
   uint16_t count;
};

#define AGC_REG_WRITES_MAX 8
#define AGC_REG_UC_WRITES_MAX 2
#define AGC_REG_SHADOW_REGS 0x400u

struct agc_reg_shadow {
   uint32_t value[AGC_REG_SHADOW_REGS];
   uint64_t known[AGC_REG_SHADOW_REGS / 64];
};

struct agc_reg_hook_args {
   enum agc_host host;
   enum agc_reg_bank bank;
   uint32_t offset;
   uint32_t host_offset;
   uint32_t value;
   const char *function;
   const struct agc_register *siblings;
   uint32_t sibling_count;
   const struct agc_reg_shadow *context;
};

struct agc_reg_writes {
   uint32_t count;
   struct agc_register reg[AGC_REG_WRITES_MAX];
   uint32_t uc_count;
   struct agc_register uc[AGC_REG_UC_WRITES_MAX];
};

typedef void (*agc_reg_hook_fn)(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);

static inline bool
agc_reg_sibling(const struct agc_reg_hook_args *args, uint32_t offset, uint32_t *value)
{
   for (uint32_t i = args->sibling_count; i-- > 0;) {
      if (args->siblings[i].offset == offset) {
         *value = args->siblings[i].value;
         return true;
      }
   }
   return false;
}

static inline bool
agc_reg_context(const struct agc_reg_hook_args *args, uint32_t offset, uint32_t *value)
{
   const struct agc_reg_shadow *context = args->context;
   if (!context || offset >= AGC_REG_SHADOW_REGS || !((context->known[offset / 64] >> (offset % 64)) & 1))
      return false;
   *value = context->value[offset];
   return true;
}

static inline bool
agc_reg_state(const struct agc_reg_hook_args *args, uint32_t offset, uint32_t *value)
{
   if (agc_reg_sibling(args, offset, value))
      return true;
   return args->bank == AGC_REG_BANK_CX && agc_reg_context(args, offset, value);
}

static inline void
agc_reg_write(struct agc_reg_writes *out, uint32_t offset, uint32_t value)
{
   out->reg[out->count].offset = offset;
   out->reg[out->count].value = value;
   out->count++;
}

static inline void
agc_reg_write_uc(struct agc_reg_writes *out, uint32_t offset, uint32_t value)
{
   out->uc[out->uc_count].offset = offset;
   out->uc[out->uc_count].value = value;
   out->uc_count++;
}

struct agc_reg_bank_table {
   const struct agc_reg_entry *entries;
   const uint64_t *pass;
   uint32_t count;
};

struct agc_reg_tables {
   struct agc_reg_bank_table bank[AGC_REG_BANK_COUNT];
};

struct agc_host_caps {
   uint32_t shader_engines;
};

extern const struct agc_reg_tables *agc_host_regs;
extern struct agc_host_caps agc_host_caps;

bool agc_host_select(enum agc_host host);
enum agc_host agc_host_current(void);
const char *agc_reg_bank_name(enum agc_reg_bank bank);
const char *agc_reg_class_name(enum agc_reg_class cls);
const struct agc_reg_tables *agc_reg_tables_get(enum agc_host host);
uint32_t agc_reg_class_count(enum agc_host host, enum agc_reg_bank bank, enum agc_reg_class cls);
uint32_t agc_reg_field_prog_count(void);
uint32_t agc_reg_field_apply(uint32_t prog, uint32_t value);

static inline const struct agc_reg_entry *
agc_reg_lookup(const struct agc_reg_tables *tables, enum agc_reg_bank bank, uint32_t offset)
{
   const struct agc_reg_bank_table *table = &tables->bank[bank];
   return offset < table->count ? &table->entries[offset] : NULL;
}

static inline bool
agc_reg_run_passes(const struct agc_reg_tables *tables, enum agc_reg_bank bank, uint32_t first, uint32_t count)
{
   const struct agc_reg_bank_table *table = &tables->bank[bank];
   if (count == 0 || first >= table->count || count > table->count - first)
      return false;
   uint32_t last = first + count - 1;
   uint32_t w0 = first >> 6;
   uint32_t w1 = last >> 6;
   uint64_t m0 = ~UINT64_C(0) << (first & 63);
   uint64_t m1 = ~UINT64_C(0) >> (63 - (last & 63));
   if (w0 == w1)
      return (table->pass[w0] & m0 & m1) == (m0 & m1);
   if ((table->pass[w0] & m0) != m0 || (table->pass[w1] & m1) != m1)
      return false;
   for (uint32_t w = w0 + 1; w < w1; w++) {
      if (table->pass[w] != ~UINT64_C(0))
         return false;
   }
   return true;
}

#endif
