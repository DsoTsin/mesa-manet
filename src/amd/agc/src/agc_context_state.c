#include "agc_register_defaults.h"

#include <stdlib.h>
#include <string.h>

#include "agc_graphics.h"

static const uint32_t state_registers[AGC_REG_BANK_COUNT] = {
   [AGC_REG_BANK_CX] = AGC_DEFAULT_STATE_CX_REGISTERS,
   [AGC_REG_BANK_SH] = AGC_DEFAULT_STATE_SH_REGISTERS,
   [AGC_REG_BANK_UC] = AGC_DEFAULT_STATE_UC_REGISTERS,
};

uint32_t
agc_default_state_registers(enum agc_reg_bank bank)
{
   return (unsigned)bank < AGC_REG_BANK_COUNT ? state_registers[bank] : 0;
}

int32_t
agc_default_state_flat(uint32_t version, enum agc_reg_bank bank, uint32_t *destination,
                       uint32_t first, uint32_t end)
{
   if (!destination || (unsigned)bank >= AGC_REG_BANK_COUNT)
      return AGC_DEFAULT_STATE_ERROR_ARGUMENT;
   if (first >= end || end > state_registers[bank])
      return AGC_DEFAULT_STATE_ERROR_RANGE;

   memset(destination, 0, (size_t)(end - first) * sizeof(*destination));
   uint32_t count;
   const struct agc_defaults_group *groups = agc_register_defaults_groups(version, false, &count);
   for (uint32_t i = 0; i < count; i++) {
      const struct agc_defaults_group *group = &groups[i];
      if (group->bank != bank)
         continue;
      for (uint32_t r = 0; r < group->count; r++) {
         uint32_t offset = group->registers[r].offset;
         if (offset >= first && offset < end)
            destination[offset - first] = group->registers[r].value;
      }
   }
   return 0;
}

int32_t
agc_default_cx_state_flat(uint32_t version, uint32_t *destination, uint32_t first, uint32_t end)
{
   return agc_default_state_flat(version, AGC_REG_BANK_CX, destination, first, end);
}

_Static_assert(AGC_REG_SHADOW_REGS == AGC_DEFAULT_STATE_CX_REGISTERS, "the shadow holds the whole context bank");

void
agc_reg_shadow_reset(struct agc_reg_shadow *shadow)
{
   if (agc_default_cx_state_flat(agc_register_defaults_init_version(), shadow->value, 0, AGC_REG_SHADOW_REGS))
      memset(shadow->value, 0, sizeof(shadow->value));
   memset(shadow->known, 0xff, sizeof(shadow->known));
}

struct host_state {
   bool built;
   enum agc_host host;
   uint32_t version;
   struct agc_register *registers[AGC_REG_BANK_COUNT];
   uint32_t count[AGC_REG_BANK_COUNT];
   uint32_t skipped[AGC_REG_BANK_COUNT];
};

static struct host_state host_state;

static bool
convertible(enum agc_reg_bank bank, uint32_t offset)
{
   const struct agc_reg_entry *entry = agc_reg_lookup(agc_host_regs, bank, offset);
   return entry && entry->cls != AGC_REG_UNDEFINED && entry->cls != AGC_REG_PENDING;
}

static void
build_host_state(uint32_t version)
{
   for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++) {
      free(host_state.registers[bank]);
      host_state.registers[bank] = NULL;
      host_state.count[bank] = 0;
      host_state.skipped[bank] = 0;

      uint32_t slots = state_registers[bank];
      uint32_t *values = calloc(slots, sizeof(*values));
      bool *present = calloc(slots, sizeof(*present));
      struct agc_register *guest = calloc(slots, sizeof(*guest));
      if (!values || !present || !guest) {
         free(values);
         free(present);
         free(guest);
         continue;
      }
      uint32_t groups_count;
      const struct agc_defaults_group *groups = agc_register_defaults_groups(version, false, &groups_count);
      for (uint32_t g = 0; g < groups_count; g++) {
         if (groups[g].bank != bank)
            continue;
         for (uint32_t r = 0; r < groups[g].count; r++) {
            uint32_t offset = groups[g].registers[r].offset;
            if (offset >= slots)
               continue;
            values[offset] = groups[g].registers[r].value;
            present[offset] = true;
         }
      }
      uint32_t guest_count = 0;
      for (uint32_t offset = 0; offset < slots; offset++) {
         if (!present[offset])
            continue;
         if (!convertible(bank, offset)) {
            host_state.skipped[bank]++;
            continue;
         }
         guest[guest_count++] = (struct agc_register){offset, values[offset]};
      }
      uint32_t bound = agc_reg_convert_pairs_bound(guest_count);
      struct agc_register *host = calloc(bound ? bound : 1, sizeof(*host));
      if (host) {
         uint32_t written = agc_reg_convert_pairs(bank, guest, guest_count, host, "agc_default_state_host");
         memset(values, 0, slots * sizeof(*values));
         memset(present, 0, slots * sizeof(*present));
         uint32_t unique = 0;
         for (uint32_t i = 0; i < written; i++) {
            if (host[i].offset >= slots)
               continue;
            if (!present[host[i].offset])
               unique++;
            present[host[i].offset] = true;
            values[host[i].offset] = host[i].value;
         }
         uint32_t at = 0;
         for (uint32_t offset = 0; offset < slots && at < unique; offset++) {
            if (present[offset])
               host[at++] = (struct agc_register){offset, values[offset]};
         }
         host_state.registers[bank] = host;
         host_state.count[bank] = at;
      }
      free(values);
      free(present);
      free(guest);
   }
   host_state.host = agc_host_current();
   host_state.version = version;
   host_state.built = true;
}

const struct agc_register *
agc_default_state_host(uint32_t version, enum agc_reg_bank bank, uint32_t *count)
{
   if ((unsigned)bank >= AGC_REG_BANK_COUNT || agc_host_current() >= AGC_HOST_COUNT) {
      *count = 0;
      return NULL;
   }
   if (!host_state.built || host_state.version != version || host_state.host != agc_host_current())
      build_host_state(version);
   *count = host_state.count[bank];
   return host_state.registers[bank];
}

uint32_t
agc_default_state_host_skipped(uint32_t version, enum agc_reg_bank bank)
{
   uint32_t count;
   agc_default_state_host(version, bank, &count);
   return (unsigned)bank < AGC_REG_BANK_COUNT ? host_state.skipped[bank] : 0;
}
