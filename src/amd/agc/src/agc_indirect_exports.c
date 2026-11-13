#include "agc_debug.h"
#include "agc_exports.h"
#include "agc_graphics.h"

#define PATCH_INVALID ((int32_t)0x8a6c0001u)

static uint32_t *
set_indirect(struct agc_cb *cb, enum agc_reg_bank bank, const struct agc_register *registers, uint32_t count,
             const char *function)
{
   return agc_emit_set_registers_indirect(cb, bank, registers, count, function);
}

uint32_t *AGC_SYSV
sceAgcDcbSetCxRegistersIndirect(struct agc_cb *cb, const struct agc_register *registers, uint32_t count)
{
   return set_indirect(cb, AGC_REG_BANK_CX, registers, count, "sceAgcDcbSetCxRegistersIndirect");
}

uint32_t *AGC_SYSV
sceAgcDcbSetShRegistersIndirect(struct agc_cb *cb, const struct agc_register *registers, uint32_t count)
{
   return set_indirect(cb, AGC_REG_BANK_SH, registers, count, "sceAgcDcbSetShRegistersIndirect");
}

uint32_t *AGC_SYSV
sceAgcDcbSetUcRegistersIndirect(struct agc_cb *cb, const struct agc_register *registers, uint32_t count)
{
   return set_indirect(cb, AGC_REG_BANK_UC, registers, count, "sceAgcDcbSetUcRegistersIndirect");
}

uint32_t AGC_SYSV
sceAgcDcbSetCxRegistersIndirectGetSize(uint32_t count)
{
   (void)count;
   return AGC_INDIRECT_REGISTERS_DWORDS;
}

uint32_t AGC_SYSV
sceAgcDcbSetShRegistersIndirectGetSize(uint32_t count)
{
   (void)count;
   return AGC_INDIRECT_REGISTERS_DWORDS;
}

uint32_t AGC_SYSV
sceAgcDcbSetUcRegistersIndirectGetSize(uint32_t count)
{
   (void)count;
   return AGC_INDIRECT_REGISTERS_DWORDS;
}

enum patch_kind {
   PATCH_ADD,
   PATCH_ADDRESS,
   PATCH_COUNT,
};

static int32_t
patch(uint32_t *command, enum agc_reg_bank bank, enum patch_kind kind, uint32_t count,
      const struct agc_register *registers, const char *function)
{
   const struct agc_register *current;
   uint32_t current_count;
   enum agc_reg_bank current_bank;
   if (!command || !agc_registers_indirect_get(command, &current, &current_count, &current_bank)) {
      agc_report(function, "the command is not an indirect register load this library recorded");
      return PATCH_INVALID;
   }
   if (current_bank != bank) {
      agc_report(function, "the command loads %s registers", agc_reg_bank_name(current_bank));
      return PATCH_INVALID;
   }
   switch (kind) {
   case PATCH_ADD:
      if (current_count > AGC_INDIRECT_REGISTERS_MAX || count > AGC_INDIRECT_REGISTERS_MAX - current_count) {
         agc_report(function, "%u + %u registers exceed 0x3fff", current_count, count);
         return PATCH_INVALID;
      }
      current_count += count;
      break;
   case PATCH_ADDRESS:
      if (!registers || (uintptr_t)registers % 4 != 0) {
         agc_report(function, "null or misaligned register array");
         return PATCH_INVALID;
      }
      current = registers;
      break;
   case PATCH_COUNT:
      if (count > AGC_INDIRECT_REGISTERS_MAX) {
         agc_report(function, "%u registers exceed 0x3fff", count);
         return PATCH_INVALID;
      }
      current_count = count;
      break;
   }
   if (!agc_patch_registers_indirect(command, current, current_count)) {
      agc_report(function, "the patch cannot be applied");
      return PATCH_INVALID;
   }
   return 0;
}

int32_t AGC_SYSV
sceAgcSetCxRegIndirectPatchAddRegisters(uint32_t *command, uint32_t count)
{
   return patch(command, AGC_REG_BANK_CX, PATCH_ADD, count, NULL, "sceAgcSetCxRegIndirectPatchAddRegisters");
}

int32_t AGC_SYSV
sceAgcSetCxRegIndirectPatchSetAddress(uint32_t *command, const struct agc_register *registers)
{
   return patch(command, AGC_REG_BANK_CX, PATCH_ADDRESS, 0, registers, "sceAgcSetCxRegIndirectPatchSetAddress");
}

int32_t AGC_SYSV
sceAgcSetCxRegIndirectPatchSetNumRegisters(uint32_t *command, uint32_t count)
{
   return patch(command, AGC_REG_BANK_CX, PATCH_COUNT, count, NULL, "sceAgcSetCxRegIndirectPatchSetNumRegisters");
}

int32_t AGC_SYSV
sceAgcSetShRegIndirectPatchAddRegisters(uint32_t *command, uint32_t count)
{
   return patch(command, AGC_REG_BANK_SH, PATCH_ADD, count, NULL, "sceAgcSetShRegIndirectPatchAddRegisters");
}

int32_t AGC_SYSV
sceAgcSetShRegIndirectPatchSetAddress(uint32_t *command, const struct agc_register *registers)
{
   return patch(command, AGC_REG_BANK_SH, PATCH_ADDRESS, 0, registers, "sceAgcSetShRegIndirectPatchSetAddress");
}

int32_t AGC_SYSV
sceAgcSetShRegIndirectPatchSetNumRegisters(uint32_t *command, uint32_t count)
{
   return patch(command, AGC_REG_BANK_SH, PATCH_COUNT, count, NULL, "sceAgcSetShRegIndirectPatchSetNumRegisters");
}

int32_t AGC_SYSV
sceAgcSetUcRegIndirectPatchAddRegisters(uint32_t *command, uint32_t count)
{
   return patch(command, AGC_REG_BANK_UC, PATCH_ADD, count, NULL, "sceAgcSetUcRegIndirectPatchAddRegisters");
}

int32_t AGC_SYSV
sceAgcSetUcRegIndirectPatchSetAddress(uint32_t *command, const struct agc_register *registers)
{
   return patch(command, AGC_REG_BANK_UC, PATCH_ADDRESS, 0, registers, "sceAgcSetUcRegIndirectPatchSetAddress");
}

int32_t AGC_SYSV
sceAgcSetUcRegIndirectPatchSetNumRegisters(uint32_t *command, uint32_t count)
{
   return patch(command, AGC_REG_BANK_UC, PATCH_COUNT, count, NULL, "sceAgcSetUcRegIndirectPatchSetNumRegisters");
}
