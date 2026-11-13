#ifndef AGC_ISA_ANALYZE_PRIV_H
#define AGC_ISA_ANALYZE_PRIV_H

#include <stdlib.h>
#include <string.h>

#include "agc_isa.h"

#define AGC_ISA_RSRC2_SCRATCH_EN 0x1u
#define AGC_ISA_RSRC2_USER_SGPR_SHIFT 1
#define AGC_ISA_RSRC2_USER_SGPR_MASK (0x1fu << AGC_ISA_RSRC2_USER_SGPR_SHIFT)
#define AGC_ISA_RSRC2_TGID_X_EN (1u << 7)
#define AGC_ISA_RSRC2_USER_SGPR_MSB (1u << 27)
#define AGC_ISA_MAX_COMPUTE_USER_SGPRS 16
#define AGC_ISA_MAX_GRAPHICS_USER_SGPRS 32
#define AGC_ISA_NGG_SYSTEM_SGPRS 8

static inline bool
agc_isa_fail(struct agc_isa_error *error, enum agc_isa_error_kind kind, uint32_t pc)
{
   memset(error, 0, sizeof(*error));
   error->kind = kind;
   error->pc = pc;
   return false;
}

static inline bool
agc_isa_fail_memory(struct agc_isa_error *error)
{
   return agc_isa_fail(error, AGC_ISA_ERROR_OUT_OF_MEMORY, 0);
}

static inline bool
agc_isa_fail_branch(struct agc_isa_error *error, uint32_t pc, uint32_t target)
{
   agc_isa_fail(error, AGC_ISA_ERROR_INVALID_BRANCH, pc);
   error->branch_target = target;
   return false;
}

static inline bool
agc_isa_fail_options(struct agc_isa_error *error, const char *reason)
{
   agc_isa_fail(error, AGC_ISA_ERROR_INVALID_OPTIONS, 0);
   error->reason = reason;
   return false;
}

static inline bool
agc_isa_fail_operand(struct agc_isa_error *error, uint32_t pc, const char *reason)
{
   agc_isa_fail(error, AGC_ISA_ERROR_INVALID_OPERAND, pc);
   error->reason = reason;
   return false;
}

static inline bool
agc_isa_block_at(const struct agc_isa_ir *ir, uint32_t pc, size_t *block)
{
   size_t index;
   if (!agc_isa_ir_instruction_at(ir, pc, &index))
      return false;
   size_t lo = 0;
   size_t hi = ir->block_count;
   while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      if (ir->blocks[mid].first < index)
         lo = mid + 1;
      else
         hi = mid;
   }
   if (lo == ir->block_count || ir->blocks[lo].first != index)
      return false;
   *block = lo;
   return true;
}

struct agc_isa_work {
   uint64_t *bits;
   size_t words;
   size_t lowest;
};

static inline bool
agc_isa_work_init(struct agc_isa_work *work, size_t count)
{
   work->words = count / 64 + 1;
   work->lowest = work->words;
   work->bits = calloc(work->words, sizeof(uint64_t));
   return work->bits != NULL;
}

static inline void
agc_isa_work_insert(struct agc_isa_work *work, size_t index)
{
   size_t word = index / 64;
   work->bits[word] |= UINT64_C(1) << (index % 64);
   if (word < work->lowest)
      work->lowest = word;
}

static inline bool
agc_isa_work_pop_first(struct agc_isa_work *work, size_t *index)
{
   for (size_t word = work->lowest; word < work->words; word++) {
      uint64_t bits = work->bits[word];
      if (!bits)
         continue;
      unsigned bit = 0;
      while (!((bits >> bit) & 1))
         bit++;
      work->bits[word] = bits & (bits - 1);
      work->lowest = word;
      *index = word * 64 + bit;
      return true;
   }
   work->lowest = work->words;
   return false;
}

struct agc_isa_uses {
   struct agc_isa_descriptor_use *descriptors;
   size_t descriptor_count;
   struct agc_isa_descriptor_use *pointers;
   size_t pointer_count;
   bool has_fill;
   struct agc_isa_buffer_fill fill;
   void *storage;
};

bool agc_isa_sources_descriptor_uses(const struct agc_isa_ir *ir, bool wave64, uint16_t user_start,
                                     uint16_t user_end, bool has_group_x, uint8_t group_x,
                                     struct agc_isa_uses *uses);

#endif
