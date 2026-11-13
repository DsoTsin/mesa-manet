#ifndef AGC_REGISTER_DEFAULTS_H
#define AGC_REGISTER_DEFAULTS_H

#include "agc_registers.h"

#define AGC_DEFAULTS_VERSION_MAX 13u
#define AGC_DEFAULTS_TABLES 4u
#define AGC_DEFAULTS_TABLE_UC_EXTRA 3u
#define AGC_DEFAULTS_BLOCK_REGISTERS 256u

#define AGC_DEFAULT_STATE_CX_REGISTERS 0x400u
#define AGC_DEFAULT_STATE_SH_REGISTERS 0x400u
#define AGC_DEFAULT_STATE_UC_REGISTERS 0x2000u

#define AGC_DEFAULT_STATE_ERROR_ARGUMENT ((int32_t)0x8a6c000au)
#define AGC_DEFAULT_STATE_ERROR_RANGE ((int32_t)0x8a6c0044u)

struct agc_defaults_type {
   uint32_t type;
   uint32_t word;
   uint32_t reserved;
};

struct agc_register_defaults {
   const struct agc_register *const *tables[AGC_DEFAULTS_TABLES];
   uint32_t counts[AGC_DEFAULTS_TABLES];
   const struct agc_defaults_type *types;
   uint32_t count;
   uint32_t pad;
};

struct agc_defaults_group {
   uint32_t type;
   uint32_t word;
   uint32_t index;
   uint32_t count;
   uint8_t bank;
   uint8_t table;
   const struct agc_register *registers;
};

static inline uint32_t
agc_defaults_word_table(uint32_t word)
{
   return word & 3;
}

static inline uint32_t
agc_defaults_word_index(uint32_t word)
{
   return (word >> 2) & 0xff;
}

static inline uint32_t
agc_defaults_word_registers(uint32_t word)
{
   return ((word >> 10) & 0xff) * (word >> 18);
}

static inline enum agc_reg_bank
agc_register_defaults_table_bank(uint32_t table)
{
   return table < AGC_REG_BANK_COUNT ? (enum agc_reg_bank)table : AGC_REG_BANK_UC;
}

uint32_t agc_register_defaults_table_version(uint32_t version, bool internal);
const void *agc_register_defaults_get(uint32_t version, bool internal);
size_t agc_register_defaults_bytes(uint32_t version, bool internal);
uint32_t agc_register_defaults_table_length(uint32_t version, bool internal, uint32_t table);
const struct agc_defaults_group *agc_register_defaults_groups(uint32_t version, bool internal,
                                                              uint32_t *count);
const struct agc_defaults_group *agc_register_defaults_find(uint32_t version, bool internal,
                                                            uint32_t type);
bool agc_register_defaults_value(uint32_t version, bool internal, enum agc_reg_bank bank,
                                 uint32_t offset, uint32_t *value);
void agc_register_defaults_init(uint32_t version);
uint32_t agc_register_defaults_init_version(void);

uint32_t agc_default_state_registers(enum agc_reg_bank bank);
int32_t agc_default_state_flat(uint32_t version, enum agc_reg_bank bank, uint32_t *destination,
                               uint32_t first, uint32_t end);
int32_t agc_default_cx_state_flat(uint32_t version, uint32_t *destination, uint32_t first,
                                  uint32_t end);
const struct agc_register *agc_default_state_host(uint32_t version, enum agc_reg_bank bank,
                                                  uint32_t *count);
uint32_t agc_default_state_host_skipped(uint32_t version, enum agc_reg_bank bank);

#endif
