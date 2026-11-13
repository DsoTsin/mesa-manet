#include "agc_graphics.h"

#include "agc_cb_builders.h"
#include "agc_debug.h"
#include "agc_surface.h"
#include "sid.h"

#include "agc_regclass_tables.h"

#define AGC_ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

const struct agc_reg_tables *agc_host_regs;
struct agc_host_caps agc_host_caps = {.shader_engines = 1};
static enum agc_host current_host = AGC_HOST_COUNT;

bool
agc_host_select(enum agc_host host)
{
   if ((unsigned)host >= AGC_HOST_COUNT)
      return false;
   current_host = host;
   agc_host_regs = &agc_reg_tables[host];
   return true;
}

enum agc_host
agc_host_current(void)
{
   return current_host;
}

const char *
agc_reg_bank_name(enum agc_reg_bank bank)
{
   static const char *const names[AGC_REG_BANK_COUNT] = {"cx", "sh", "uc"};
   return (unsigned)bank < AGC_REG_BANK_COUNT ? names[bank] : "?";
}

const char *
agc_reg_class_name(enum agc_reg_class cls)
{
   static const char *const names[AGC_REG_CLASS_COUNT] = {
      "undefined", "pass", "move", "field", "addr", "drop", "split", "merge", "guest", "pending",
   };
   return (unsigned)cls < AGC_REG_CLASS_COUNT ? names[cls] : "?";
}

const struct agc_reg_tables *
agc_reg_tables_get(enum agc_host host)
{
   return (unsigned)host < AGC_HOST_COUNT ? &agc_reg_tables[host] : NULL;
}

uint32_t
agc_reg_class_count(enum agc_host host, enum agc_reg_bank bank, enum agc_reg_class cls)
{
   if ((unsigned)host >= AGC_HOST_COUNT || (unsigned)bank >= AGC_REG_BANK_COUNT ||
       (unsigned)cls >= AGC_REG_CLASS_COUNT)
      return 0;
   return agc_reg_class_counts[host][bank][cls];
}

uint32_t
agc_reg_field_prog_count(void)
{
   return AGC_ARRAY_SIZE(agc_field_progs);
}

uint32_t
agc_reg_field_apply(uint32_t prog, uint32_t value)
{
   const struct agc_field_prog *p = &agc_field_progs[prog];
   uint32_t out = value & p->keep;
   for (uint32_t i = 0; i < p->count; i++) {
      const struct agc_field_move *move = &agc_field_moves[p->first + i];
      uint32_t mask = move->width >= 32 ? ~UINT32_C(0) : (UINT32_C(1) << move->width) - 1;
      out |= ((value >> move->src) & mask) << move->dst;
   }
   return out;
}

static void
run_hook(enum agc_reg_bank bank, uint32_t offset, uint32_t value, const struct agc_reg_entry *entry,
         const struct agc_register *siblings, uint32_t sibling_count, const struct agc_reg_shadow *shadow,
         struct agc_reg_writes *out, const char *function)
{
   struct agc_reg_hook_args args = {
      .host = current_host,
      .bank = bank,
      .offset = offset,
      .host_offset = entry->host_offset,
      .value = value,
      .function = function,
      .siblings = siblings,
      .sibling_count = sibling_count,
      .context = shadow,
   };
   out->count = 0;
   out->uc_count = 0;
   if (entry->extra >= AGC_REG_HOOK_COUNT)
      agc_fatal(function, "register %s:0x%03x names an unknown hook", agc_reg_bank_name(bank), offset);
   agc_reg_hooks[entry->extra](&args, out);
   if (out->count > AGC_REG_WRITES_MAX || out->uc_count > AGC_REG_UC_WRITES_MAX) {
      agc_fatal(function, "the hook of register %s:0x%03x wrote too many registers",
                agc_reg_bank_name(bank), offset);
   }
}

bool
agc_reg_convert(enum agc_reg_bank bank, uint32_t offset, uint32_t value, uint32_t *host_offset,
                uint32_t *host_value, const char *function)
{
   const struct agc_reg_entry *entry = agc_reg_lookup(agc_host_regs, bank, offset);
   enum agc_reg_class cls = entry ? (enum agc_reg_class)entry->cls : AGC_REG_UNDEFINED;
   switch (cls) {
   case AGC_REG_PASS:
   case AGC_REG_MOVE:
   case AGC_REG_ADDR:
      *host_offset = entry->host_offset;
      *host_value = value;
      return true;
   case AGC_REG_FIELD:
      *host_offset = entry->host_offset;
      *host_value = agc_reg_field_apply(entry->extra, value);
      return true;
   case AGC_REG_DROP:
      return false;
   case AGC_REG_SPLIT:
   case AGC_REG_MERGE:
   case AGC_REG_GUEST: {
      struct agc_reg_writes writes;
      run_hook(bank, offset, value, entry, NULL, 0, NULL, &writes, function);
      if (writes.uc_count) {
         agc_fatal(function, "register %s:0x%03x needs a uconfig write that a single register write cannot make",
                   agc_reg_bank_name(bank), offset);
      }
      if (writes.count > 1) {
         agc_fatal(function, "register %s:0x%03x becomes %u host writes and needs a trampoline here",
                   agc_reg_bank_name(bank), offset, writes.count);
      }
      if (writes.count == 0)
         return false;
      *host_offset = writes.reg[0].offset;
      *host_value = writes.reg[0].value;
      return true;
   }
   default:
      agc_fatal(function, "register %s:0x%03x has no host conversion yet (%s)",
                agc_reg_bank_name(bank), offset, agc_reg_class_name(cls));
   }
}

uint32_t
agc_reg_convert_pairs_bound(uint32_t count)
{
   return count > UINT32_MAX / AGC_REG_WRITES_MAX ? 0 : count * AGC_REG_WRITES_MAX;
}

void
agc_reg_shadow_note(struct agc_reg_shadow *shadow, const struct agc_register *guest, uint32_t count)
{
   for (uint32_t i = 0; i < count; i++) {
      uint32_t offset = guest[i].offset;
      if (offset >= AGC_REG_SHADOW_REGS)
         continue;
      shadow->value[offset] = guest[i].value;
      shadow->known[offset / 64] |= UINT64_C(1) << (offset % 64);
   }
}

#define TARGET_COLOR_SLOTS 8u
#define TARGET_COLOR_STRIDE 15u
#define TARGET_DEPTH_PLANES 4u
#define TARGET_CX(reg) (((reg) - SI_CONTEXT_REG_OFFSET) / 4)

static agc_reg_target_note target_note;

void
agc_reg_set_target_note(agc_reg_target_note note)
{
   target_note = note;
}

static uint32_t
target_value(const struct agc_register *guest, uint32_t count, const struct agc_reg_shadow *shadow, uint32_t offset)
{
   for (uint32_t i = count; i-- > 0;) {
      if (guest[i].offset == offset)
         return guest[i].value;
   }
   return shadow && offset < AGC_REG_SHADOW_REGS ? shadow->value[offset] : 0;
}

static void
target_report(uint32_t low, uint32_t high, uint32_t mode)
{
   uint64_t base = 0;
   agc_surface_split_pipe_bank_xor(mode, ((uint64_t)low | (uint64_t)(high & 0xff) << 32) << 8, &base, NULL);
   if (base)
      target_note(base);
}

static void
targets_report(const struct agc_register *guest, uint32_t count, const struct agc_reg_shadow *shadow)
{
   const uint32_t color_base = TARGET_CX(R_028C60_CB_COLOR0_BASE);
   const uint32_t color_ext = TARGET_CX(R_028E40_CB_COLOR0_BASE_EXT);
   const uint32_t depth_base = TARGET_CX(R_028048_DB_Z_READ_BASE);
   const uint32_t depth_hi = TARGET_CX(R_028068_DB_Z_READ_BASE_HI);
   uint32_t colors = 0;
   uint32_t planes = 0;
   for (uint32_t i = 0; i < count; i++) {
      const uint32_t offset = guest[i].offset;
      if (offset >= color_base && offset < color_base + TARGET_COLOR_SLOTS * TARGET_COLOR_STRIDE) {
         if ((offset - color_base) % TARGET_COLOR_STRIDE == 0)
            colors |= UINT32_C(1) << ((offset - color_base) / TARGET_COLOR_STRIDE);
      } else if (offset >= color_ext && offset < color_ext + TARGET_COLOR_SLOTS) {
         colors |= UINT32_C(1) << (offset - color_ext);
      } else if (offset >= depth_base && offset < depth_base + TARGET_DEPTH_PLANES) {
         planes |= UINT32_C(1) << (offset - depth_base);
      } else if (offset >= depth_hi && offset < depth_hi + TARGET_DEPTH_PLANES) {
         planes |= UINT32_C(1) << (offset - depth_hi);
      }
   }
   for (uint32_t slot = 0; slot < TARGET_COLOR_SLOTS; slot++) {
      if (!(colors & (UINT32_C(1) << slot)))
         continue;
      const uint32_t attrib3 = target_value(guest, count, shadow, TARGET_CX(R_028EE0_CB_COLOR0_ATTRIB3) + slot);
      target_report(target_value(guest, count, shadow, color_base + slot * TARGET_COLOR_STRIDE),
                    target_value(guest, count, shadow, color_ext + slot), (attrib3 >> 14) & 0x1f);
   }
   for (uint32_t plane = 0; plane < TARGET_DEPTH_PLANES; plane++) {
      if (!(planes & (UINT32_C(1) << plane)))
         continue;
      const uint32_t info = target_value(guest, count, shadow,
                                         TARGET_CX(plane & 1 ? R_028044_DB_STENCIL_INFO : R_028040_DB_Z_INFO));
      target_report(target_value(guest, count, shadow, depth_base + plane),
                    target_value(guest, count, shadow, depth_hi + plane), (info >> 4) & 0x1f);
   }
}

uint32_t
agc_reg_convert_pairs_state(enum agc_reg_bank bank, const struct agc_register *guest, uint32_t count,
                            struct agc_register *host, struct agc_reg_shadow *shadow, struct agc_register *uc,
                            uint32_t uc_capacity, uint32_t *uc_count, const char *function)
{
   uint32_t written = 0;
   if (uc_count)
      *uc_count = 0;
   for (uint32_t i = 0; i < count; i++) {
      if (guest[i].offset > AGC_REGISTER_OFFSET_MAX)
         continue;
      const struct agc_reg_entry *entry = agc_reg_lookup(agc_host_regs, bank, guest[i].offset);
      if (entry && entry->cls >= AGC_REG_SPLIT && entry->cls <= AGC_REG_GUEST) {
         struct agc_reg_writes writes;
         run_hook(bank, guest[i].offset, guest[i].value, entry, guest, count, shadow, &writes, function);
         for (uint32_t w = 0; w < writes.count; w++)
            host[written++] = writes.reg[w];
         if (writes.uc_count && (!uc || !uc_count || *uc_count + writes.uc_count > uc_capacity)) {
            agc_fatal(function,
                      "register %s:0x%03x needs a uconfig write that this register array cannot make outside a "
                      "submission",
                      agc_reg_bank_name(bank), guest[i].offset);
         }
         for (uint32_t w = 0; w < writes.uc_count; w++)
            uc[(*uc_count)++] = writes.uc[w];
         continue;
      }
      uint32_t host_offset;
      uint32_t host_value;
      if (!agc_reg_convert(bank, guest[i].offset, guest[i].value, &host_offset, &host_value, function))
         continue;
      host[written].offset = host_offset;
      host[written].value = host_value;
      written++;
   }
   if (shadow && bank == AGC_REG_BANK_CX)
      agc_reg_shadow_note(shadow, guest, count);
   if (target_note && bank == AGC_REG_BANK_CX)
      targets_report(guest, count, shadow);
   return written;
}

uint32_t
agc_reg_convert_pairs(enum agc_reg_bank bank, const struct agc_register *guest, uint32_t count,
                      struct agc_register *host, const char *function)
{
   return agc_reg_convert_pairs_state(bank, guest, count, host, NULL, NULL, 0, NULL, function);
}

static const uint8_t load_reg_opcode[AGC_REG_BANK_COUNT] = {
   [AGC_REG_BANK_CX] = PKT3_LOAD_CONTEXT_REG_INDEX,
   [AGC_REG_BANK_SH] = PKT3_LOAD_SH_REG_INDEX,
   [AGC_REG_BANK_UC] = PKT3_LOAD_UCONFIG_REG_INDEX,
};

static void
write_registers_indirect(uint32_t *out, enum agc_reg_bank bank, uint64_t address, uint32_t count)
{
   out[0] = PKT3(load_reg_opcode[bank], 3, 0);
   out[1] = (uint32_t)address & ~UINT32_C(3);
   out[2] = (uint32_t)(address >> 32);
   out[3] = S_9F3_DATA_FORMAT(V_9F3_OFFSET_AND_DATA);
   out[4] = count;
}

static bool
registers_indirect_valid(const struct agc_register *registers, uint32_t count)
{
   uintptr_t address = (uintptr_t)registers;
   if (count > AGC_INDIRECT_REGISTERS_MAX)
      return false;
   return address ? (address & 3) == 0 : count == 0;
}

uint32_t *
agc_emit_set_registers_indirect(struct agc_cb *cb, enum agc_reg_bank bank,
                                const struct agc_register *registers, uint32_t count,
                                const char *function)
{
   if (!cb || !registers_indirect_valid(registers, count)) {
      agc_report(function, "null or misaligned register list, or more than 0x3fff registers");
      return NULL;
   }
   uint32_t *out = agc_record_alloc(cb, AGC_INDIRECT_REGISTERS_DWORDS);
   if (AGC_UNLIKELY(!out))
      return NULL;
   write_registers_indirect(out, bank, (uintptr_t)registers, count);
   struct agc_reloc *reloc = agc_record_add_reloc(out, AGC_RELOC_REG_ARRAY);
   reloc->bank = (uint8_t)bank;
   reloc->arg32 = count;
   reloc->arg64[0] = (uintptr_t)registers;
   return out;
}

bool
agc_patch_registers_indirect(uint32_t *packet, const struct agc_register *registers, uint32_t count)
{
   struct agc_record *record = agc_record_containing(packet);
   struct agc_reloc *reloc = agc_record_reloc_at(record, packet);
   if (!reloc || reloc->kind != AGC_RELOC_REG_ARRAY || !registers_indirect_valid(registers, count))
      return false;
   reloc->arg32 = count;
   reloc->arg64[0] = (uintptr_t)registers;
   write_registers_indirect(packet, reloc->bank, (uintptr_t)registers, count);
   reloc->header = packet[0];
   return true;
}

bool
agc_registers_indirect_get(const uint32_t *packet, const struct agc_register **registers, uint32_t *count,
                           enum agc_reg_bank *bank)
{
   struct agc_record *record = agc_record_containing(packet);
   struct agc_reloc *reloc = agc_record_reloc_at(record, packet);
   if (!reloc || reloc->kind != AGC_RELOC_REG_ARRAY)
      return false;
   *registers = agc_reloc_registers(reloc);
   *count = reloc->arg32;
   *bank = (enum agc_reg_bank)reloc->bank;
   return true;
}

void
agc_resolve_registers_indirect(const struct agc_record *record, struct agc_reloc *reloc,
                               uint64_t host_va, uint32_t host_count)
{
   uint32_t *packet = record->bottom + reloc->offset;
   if (host_count == 0)
      agc_fill_nops(packet, AGC_INDIRECT_REGISTERS_DWORDS);
   else
      write_registers_indirect(packet, reloc->bank, host_va, host_count);
   reloc->header = packet[0];
}

uint32_t
agc_registers_indirect_ib(uint32_t *ib, enum agc_reg_bank bank, uint64_t host_va, uint32_t host_count,
                          const struct agc_register *uc, uint32_t uc_count)
{
   uint32_t dwords = 0;
   if (host_count) {
      write_registers_indirect(ib, bank, host_va, host_count);
      dwords = AGC_INDIRECT_REGISTERS_DWORDS;
   }
   for (uint32_t i = 0; i < uc_count; i++) {
      ib[dwords++] = PKT3(PKT3_SET_UCONFIG_REG, 1, 0);
      ib[dwords++] = uc[i].offset;
      ib[dwords++] = uc[i].value;
   }
   uint32_t padded = (dwords + AGC_REGISTERS_IB_PAD - 1) & ~(AGC_REGISTERS_IB_PAD - 1);
   agc_fill_nops(ib + dwords, padded - dwords);
   return padded;
}

void
agc_resolve_registers_call(const struct agc_record *record, struct agc_reloc *reloc, uint64_t ib_va,
                           uint32_t ib_dwords)
{
   uint32_t *packet = record->bottom + reloc->offset;
   packet[0] = PKT3(PKT3_INDIRECT_BUFFER, 2, 0);
   packet[1] = (uint32_t)ib_va;
   packet[2] = (uint32_t)(ib_va >> 32);
   packet[3] = S_3F3_IB_SIZE(ib_dwords) | S_3F3_VALID(1);
   agc_fill_nops(packet + AGC_DWORDS_JUMP, AGC_INDIRECT_REGISTERS_DWORDS - AGC_DWORDS_JUMP);
   reloc->header = packet[0];
}
