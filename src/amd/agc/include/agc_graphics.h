#ifndef AGC_GRAPHICS_H
#define AGC_GRAPHICS_H

#include "agc_record.h"
#include "agc_registers.h"

#define AGC_INDIRECT_REGISTERS_MAX 0x3fffu
#define AGC_INDIRECT_REGISTERS_DWORDS 5u

bool agc_reg_convert(enum agc_reg_bank bank, uint32_t offset, uint32_t value, uint32_t *host_offset,
                     uint32_t *host_value, const char *function);
uint32_t agc_reg_convert_pairs_bound(uint32_t count);
uint32_t agc_reg_convert_pairs(enum agc_reg_bank bank, const struct agc_register *guest, uint32_t count,
                               struct agc_register *host, const char *function);
uint32_t agc_reg_convert_pairs_state(enum agc_reg_bank bank, const struct agc_register *guest, uint32_t count,
                                     struct agc_register *host, struct agc_reg_shadow *shadow, struct agc_register *uc,
                                     uint32_t uc_capacity, uint32_t *uc_count, const char *function);
void agc_reg_shadow_reset(struct agc_reg_shadow *shadow);
void agc_reg_shadow_note(struct agc_reg_shadow *shadow, const struct agc_register *guest, uint32_t count);

typedef void (*agc_reg_target_note)(uint64_t guest_base);
void agc_reg_set_target_note(agc_reg_target_note note);

uint32_t *agc_emit_set_registers_indirect(struct agc_cb *cb, enum agc_reg_bank bank,
                                          const struct agc_register *registers, uint32_t count,
                                          const char *function);
bool agc_patch_registers_indirect(uint32_t *packet, const struct agc_register *registers, uint32_t count);
bool agc_registers_indirect_get(const uint32_t *packet, const struct agc_register **registers, uint32_t *count,
                                enum agc_reg_bank *bank);
void agc_resolve_registers_indirect(const struct agc_record *record, struct agc_reloc *reloc,
                                    uint64_t host_va, uint32_t host_count);

#define AGC_REGISTERS_IB_PAD 8u
#define AGC_REGISTERS_UC_MAX 16u

uint32_t agc_registers_indirect_ib(uint32_t *ib, enum agc_reg_bank bank, uint64_t host_va, uint32_t host_count,
                                   const struct agc_register *uc, uint32_t uc_count);
void agc_resolve_registers_call(const struct agc_record *record, struct agc_reloc *reloc, uint64_t ib_va,
                                uint32_t ib_dwords);

static inline const struct agc_register *
agc_reloc_registers(const struct agc_reloc *reloc)
{
   return (const struct agc_register *)(uintptr_t)reloc->arg64[0];
}

#endif
