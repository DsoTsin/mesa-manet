#include "agc_register_defaults.h"

#include <stdlib.h>
#include <string.h>

#include "agc_debug.h"
#include "util/u_atomic.h"

struct agc_defaults_set {
   const struct agc_defaults_group *groups;
   const struct agc_register *registers;
   uint32_t group_count;
   uint32_t version;
   uint32_t register_count[AGC_DEFAULTS_TABLES];
   uint32_t pointer_count[AGC_DEFAULTS_TABLES];
};

#include "agc_register_defaults_tables.h"

#define REPORTED_VERSIONS 64u

_Static_assert(sizeof(void *) == 8, "guest pointers");
_Static_assert(sizeof(struct agc_register) == 8, "ShaderRegister size");
_Static_assert(offsetof(struct agc_register, offset) == 0, "ShaderRegister offset");
_Static_assert(offsetof(struct agc_register, value) == 4, "ShaderRegister value");
_Static_assert(sizeof(struct agc_defaults_type) == 12, "type entry size");
_Static_assert(offsetof(struct agc_defaults_type, type) == 0, "type entry key");
_Static_assert(offsetof(struct agc_defaults_type, word) == 4, "type entry word");
_Static_assert(offsetof(struct agc_defaults_type, reserved) == 8, "type entry reserved");
_Static_assert(sizeof(struct agc_register_defaults) == 0x40, "RegisterDefaults size");
_Static_assert(offsetof(struct agc_register_defaults, tables[AGC_REG_BANK_CX]) == 0x00, "cx table");
_Static_assert(offsetof(struct agc_register_defaults, tables[AGC_REG_BANK_SH]) == 0x08, "sh table");
_Static_assert(offsetof(struct agc_register_defaults, tables[AGC_REG_BANK_UC]) == 0x10, "uc table");
_Static_assert(offsetof(struct agc_register_defaults, tables[AGC_DEFAULTS_TABLE_UC_EXTRA]) == 0x18,
               "second uc table");
_Static_assert(offsetof(struct agc_register_defaults, counts) == 0x20, "register counts");
_Static_assert(offsetof(struct agc_register_defaults, counts[AGC_DEFAULTS_TABLE_UC_EXTRA]) == 0x2c,
               "second uc register count");
_Static_assert(offsetof(struct agc_register_defaults, types) == 0x30, "type table");
_Static_assert(offsetof(struct agc_register_defaults, count) == 0x38, "type count");
_Static_assert(offsetof(struct agc_register_defaults, pad) == 0x3c, "tail");
_Static_assert(AGC_DEFAULTS_TABLE_UC_EXTRA == AGC_REG_BANK_COUNT, "one table per bank and a spare");
_Static_assert(AGC_DEFAULTS_TABLE_UC_EXTRA + 1 == AGC_DEFAULTS_TABLES, "table count");
_Static_assert(AGC_DEFAULTS_MAP_VERSIONS == AGC_DEFAULTS_VERSION_MAX + 1, "version map");
_Static_assert(AGC_DEFAULTS_GROUP_REGISTERS_MAX <= AGC_DEFAULTS_BLOCK_REGISTERS, "largest group");
_Static_assert(AGC_DEFAULTS_CX_END <= AGC_DEFAULT_STATE_CX_REGISTERS, "cx flat state");
_Static_assert(AGC_DEFAULTS_SH_END <= AGC_DEFAULT_STATE_SH_REGISTERS, "sh flat state");
_Static_assert(AGC_DEFAULTS_UC_END <= AGC_DEFAULT_STATE_UC_REGISTERS, "uc flat state");

struct defaults_layout {
   size_t tables[AGC_DEFAULTS_TABLES];
   size_t types;
   size_t registers[AGC_DEFAULTS_TABLES];
   size_t bytes;
};

static void *defaults_blobs[AGC_DEFAULTS_SETS];
static uint32_t reported_versions[REPORTED_VERSIONS];
static uint32_t init_version;

static bool
first_report(uint32_t version)
{
   for (unsigned i = 0; i < REPORTED_VERSIONS; i++) {
      uint32_t seen = p_atomic_read(&reported_versions[i]);
      if (!seen) {
         seen = p_atomic_cmpxchg(&reported_versions[i], 0, version);
         if (!seen)
            return true;
      }
      if (seen == version)
         return false;
   }
   return false;
}

static uint32_t
resolve_version(uint32_t version, const char *function)
{
   if (AGC_LIKELY(version <= AGC_DEFAULTS_VERSION_MAX))
      return version;
   if (first_report(version)) {
      agc_report(function, "register defaults version %u does not exist; serving version %u", version,
                 AGC_DEFAULTS_VERSION_MAX);
   }
   return AGC_DEFAULTS_VERSION_MAX;
}

static unsigned
defaults_set_index(uint32_t version, bool internal, const char *function)
{
   return agc_defaults_version_sets[internal ? 1 : 0][resolve_version(version, function)];
}

static const struct agc_defaults_set *
defaults_set(uint32_t version, bool internal, const char *function)
{
   return &agc_defaults_sets[defaults_set_index(version, internal, function)];
}

static size_t
align_up(size_t value, size_t alignment)
{
   return (value + alignment - 1) & ~(alignment - 1);
}

static void
defaults_layout(const struct agc_defaults_set *set, struct defaults_layout *layout)
{
   size_t at = sizeof(struct agc_register_defaults);
   for (unsigned table = 0; table < AGC_DEFAULTS_TABLES; table++) {
      layout->tables[table] = at;
      at += (size_t)set->pointer_count[table] * sizeof(struct agc_register *);
   }
   layout->types = at;
   at = align_up(at + (size_t)set->group_count * sizeof(struct agc_defaults_type), 8);
   for (unsigned table = 0; table < AGC_DEFAULTS_TABLES; table++) {
      layout->registers[table] = at;
      at += (size_t)set->register_count[table] * sizeof(struct agc_register);
   }
   layout->bytes = at;
}

static void *
defaults_build(const struct agc_defaults_set *set)
{
   struct defaults_layout layout;
   defaults_layout(set, &layout);
   uint8_t *blob = calloc(1, layout.bytes);
   if (!blob)
      agc_fatal(__func__, "out of memory");

   struct agc_register_defaults *header = (struct agc_register_defaults *)blob;
   const struct agc_register **tables[AGC_DEFAULTS_TABLES];
   struct agc_register *banks[AGC_DEFAULTS_TABLES];
   const struct agc_register *sources[AGC_DEFAULTS_TABLES];
   const struct agc_register *source = set->registers;
   for (unsigned table = 0; table < AGC_DEFAULTS_TABLES; table++) {
      tables[table] = (const struct agc_register **)(blob + layout.tables[table]);
      banks[table] = (struct agc_register *)(blob + layout.registers[table]);
      sources[table] = source;
      memcpy(banks[table], source, (size_t)set->register_count[table] * sizeof(*source));
      source += set->register_count[table];
      header->tables[table] = set->pointer_count[table] ? tables[table] : NULL;
      header->counts[table] = set->register_count[table];
   }
   struct agc_defaults_type *types = (struct agc_defaults_type *)(blob + layout.types);
   header->types = types;
   header->count = set->group_count;

   for (uint32_t i = 0; i < set->group_count; i++) {
      const struct agc_defaults_group *group = &set->groups[i];
      types[i].type = group->type;
      types[i].word = group->word;
      tables[group->table][group->index] = banks[group->table] + (group->registers - sources[group->table]);
   }
   return blob;
}

uint32_t
agc_register_defaults_table_version(uint32_t version, bool internal)
{
   return defaults_set(version, internal, __func__)->version;
}

const void *
agc_register_defaults_get(uint32_t version, bool internal)
{
   unsigned index = defaults_set_index(version, internal,
                                       internal ? "sceAgcGetRegisterDefaults2Internal" : "sceAgcGetRegisterDefaults2");
   void **slot = &defaults_blobs[index];
   void *blob = p_atomic_read(slot);
   if (AGC_LIKELY(blob))
      return blob;

   void *fresh = defaults_build(&agc_defaults_sets[index]);
   blob = p_atomic_cmpxchg_ptr(slot, NULL, fresh);
   if (!blob)
      return fresh;
   free(fresh);
   return blob;
}

size_t
agc_register_defaults_bytes(uint32_t version, bool internal)
{
   struct defaults_layout layout;
   defaults_layout(defaults_set(version, internal, __func__), &layout);
   return layout.bytes;
}

uint32_t
agc_register_defaults_table_length(uint32_t version, bool internal, uint32_t table)
{
   const struct agc_defaults_set *set = defaults_set(version, internal, __func__);
   return table < AGC_DEFAULTS_TABLES ? set->pointer_count[table] : 0;
}

const struct agc_defaults_group *
agc_register_defaults_groups(uint32_t version, bool internal, uint32_t *count)
{
   const struct agc_defaults_set *set = defaults_set(version, internal, __func__);
   *count = set->group_count;
   return set->groups;
}

const struct agc_defaults_group *
agc_register_defaults_find(uint32_t version, bool internal, uint32_t type)
{
   const struct agc_defaults_set *set = defaults_set(version, internal, __func__);
   for (uint32_t i = 0; i < set->group_count; i++) {
      if (set->groups[i].type == type)
         return &set->groups[i];
   }
   return NULL;
}

bool
agc_register_defaults_value(uint32_t version, bool internal, enum agc_reg_bank bank,
                            uint32_t offset, uint32_t *value)
{
   const struct agc_defaults_set *set = defaults_set(version, internal, __func__);
   for (uint32_t i = set->group_count; i-- > 0;) {
      const struct agc_defaults_group *group = &set->groups[i];
      if (group->bank != bank)
         continue;
      for (uint32_t r = group->count; r-- > 0;) {
         if (group->registers[r].offset == offset) {
            *value = group->registers[r].value;
            return true;
         }
      }
   }
   return false;
}

void
agc_register_defaults_init(uint32_t version)
{
   p_atomic_set(&init_version, resolve_version(version, "sceAgcInit") + 1);
}

uint32_t
agc_register_defaults_init_version(void)
{
   uint32_t stored = p_atomic_read(&init_version);
   return stored ? stored - 1 : AGC_DEFAULTS_VERSION_MAX;
}
