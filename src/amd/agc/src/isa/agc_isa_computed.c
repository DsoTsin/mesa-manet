#include "agc_isa_analyze_priv.h"

#define SCALARS 128
#define ENTRY_SGPRS 106
#define SECOND_HALF_SGPR 6

enum word_kind {
   WORD_UNSET,
   WORD_EXTERNAL,
   WORD_ENTRY,
   WORD_CONST,
   WORD_CODE,
   WORD_RETURN,
   WORD_CODE_RELATIVE,
   WORD_CODE_DATA,
   WORD_UNKNOWN
};

struct word {
   uint8_t kind;
   bool high;
   int64_t value;
};

enum carry_kind {
   CARRY_UNSET,
   CARRY_CODE,
   CARRY_CODE_RELATIVE,
   CARRY_UNKNOWN
};

struct carry {
   uint8_t kind;
   bool subtract;
   uint32_t amount;
   int64_t offset;
};

struct state {
   struct word words[SCALARS];
   struct carry scc;
};

struct analysis {
   const struct agc_isa_ir *ir;
   struct agc_isa_code_layout layout;
   int64_t *returns;
   size_t return_count;
   uint32_t *return_pcs;
   size_t return_pc_count;
   int64_t code_bytes;
};

static inline struct word
word_of(enum word_kind kind)
{
   struct word word = {(uint8_t)kind, false, 0};
   return word;
}

static inline struct word
word_entry(uint8_t sgpr)
{
   struct word word = {WORD_ENTRY, false, sgpr};
   return word;
}

static inline struct word
word_const(uint32_t value)
{
   struct word word = {WORD_CONST, false, value};
   return word;
}

static inline struct word
word_code(int64_t offset, bool high)
{
   struct word word = {WORD_CODE, high, offset};
   return word;
}

static inline struct word
word_half(enum word_kind kind, bool high)
{
   struct word word = {(uint8_t)kind, high, 0};
   return word;
}

static inline bool
word_equal(struct word a, struct word b)
{
   return a.kind == b.kind && a.high == b.high && a.value == b.value;
}

static inline bool
word_code_derived(struct word word)
{
   return word.kind == WORD_CODE || word.kind == WORD_RETURN || word.kind == WORD_CODE_RELATIVE ||
          word.kind == WORD_CODE_DATA;
}

static inline bool
word_outside(struct word word)
{
   return word.kind == WORD_EXTERNAL || word.kind == WORD_ENTRY || word.kind == WORD_CONST;
}

static inline struct carry
carry_of(enum carry_kind kind)
{
   struct carry carry = {(uint8_t)kind, false, 0, 0};
   return carry;
}

static inline bool
carry_equal(struct carry a, struct carry b)
{
   return a.kind == b.kind && a.subtract == b.subtract && a.amount == b.amount && a.offset == b.offset;
}

static inline bool
carry_from_code(struct carry carry)
{
   return carry.kind == CARRY_CODE || carry.kind == CARRY_CODE_RELATIVE;
}

static void
state_unset(struct state *state)
{
   for (unsigned i = 0; i < SCALARS; i++)
      state->words[i] = word_of(WORD_UNSET);
   state->scc = carry_of(CARRY_UNSET);
}

static void
state_fill(struct state *state, enum word_kind kind)
{
   for (unsigned i = 0; i < SCALARS; i++)
      state->words[i] = word_of(kind);
}

static void
state_entry(struct state *state)
{
   state_fill(state, WORD_EXTERNAL);
   for (unsigned n = 0; n < ENTRY_SGPRS; n++)
      state->words[n] = word_entry((uint8_t)n);
   state->scc = carry_of(CARRY_UNKNOWN);
}

static inline bool
state_reached(const struct state *state)
{
   return state->words[0].kind != WORD_UNSET;
}

static struct word
state_read(const struct state *state, struct agc_isa_operand op)
{
   uint32_t bits = 0;
   uint16_t index;
   switch (op.kind) {
   case AGC_ISA_OPERAND_INLINE_INT:
      return word_const((uint32_t)op.inline_int);
   case AGC_ISA_OPERAND_INLINE_FLOAT:
      if (!agc_isa_inline_float_bits((uint8_t)op.inline_float, &bits))
         bits = 0;
      return word_const(bits);
   case AGC_ISA_OPERAND_LITERAL:
      return word_const(op.literal);
   case AGC_ISA_OPERAND_NULL:
      return word_const(0);
   default:
      if (agc_isa_operand_scalar_index(op, &index) && index < SCALARS)
         return state->words[index];
      return word_of(WORD_EXTERNAL);
   }
}

static void
state_read_pair(const struct state *state, struct agc_isa_operand op, struct word pair[2])
{
   uint16_t index;
   if (agc_isa_operand_scalar_index(op, &index) && (unsigned)index + 1 < SCALARS) {
      pair[0] = state->words[index];
      pair[1] = state->words[index + 1];
      return;
   }
   pair[0] = state_read(state, op);
   pair[1] = word_const(0);
}

static void
state_write(struct state *state, struct agc_isa_operand op, struct word word)
{
   uint16_t index;
   if (agc_isa_operand_scalar_index(op, &index) && index < SCALARS)
      state->words[index] = word;
}

static void
state_write_pair(struct state *state, struct agc_isa_operand op, const struct word pair[2])
{
   uint16_t index;
   if (agc_isa_operand_scalar_index(op, &index) && (unsigned)index + 1 < SCALARS) {
      state->words[index] = pair[0];
      state->words[index + 1] = pair[1];
   }
}

static bool
contains_return(const int64_t *returns, size_t count, int64_t offset)
{
   size_t lo = 0;
   size_t hi = count;
   while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      if (returns[mid] < offset)
         lo = mid + 1;
      else
         hi = mid;
   }
   return lo < count && returns[lo] == offset;
}

static bool
is_return(struct word word, bool high, const int64_t *returns, size_t count)
{
   if (word.kind == WORD_RETURN)
      return word.high == high;
   if (word.kind == WORD_CODE)
      return word.high == high && contains_return(returns, count, word.value);
   return false;
}

static inline bool
word_has_half(struct word word)
{
   return word.kind == WORD_CODE || word.kind == WORD_RETURN || word.kind == WORD_CODE_RELATIVE;
}

static struct word
merge_word(struct word a, struct word b, const int64_t *returns, size_t count)
{
   if (a.kind == WORD_UNSET)
      return b;
   if (b.kind == WORD_UNSET)
      return a;
   if (word_equal(a, b))
      return a;
   if ((a.kind == WORD_CODE || a.kind == WORD_RETURN) && is_return(a, a.high, returns, count) &&
       is_return(b, a.high, returns, count))
      return word_half(WORD_RETURN, a.high);
   if (word_has_half(a) && word_has_half(b) && a.high == b.high)
      return word_half(WORD_CODE_RELATIVE, a.high);
   if (word_outside(a) && word_outside(b))
      return word_of(WORD_EXTERNAL);
   return word_of(WORD_UNKNOWN);
}

static bool
state_merge(struct state *state, const struct state *other, const int64_t *returns, size_t count)
{
   bool changed = false;
   for (unsigned i = 0; i < SCALARS; i++) {
      struct word merged = merge_word(state->words[i], other->words[i], returns, count);
      changed |= !word_equal(merged, state->words[i]);
      state->words[i] = merged;
   }
   struct carry scc;
   if (state->scc.kind == CARRY_UNSET)
      scc = other->scc;
   else if (other->scc.kind == CARRY_UNSET)
      scc = state->scc;
   else if (carry_equal(state->scc, other->scc))
      scc = state->scc;
   else if (carry_from_code(state->scc) && carry_from_code(other->scc))
      scc = carry_of(CARRY_CODE_RELATIVE);
   else
      scc = carry_of(CARRY_UNKNOWN);
   changed |= !carry_equal(scc, state->scc);
   state->scc = scc;
   return changed;
}

static inline bool
word_is_low_code(struct word word)
{
   return word.kind == WORD_CODE && !word.high;
}

static inline bool
word_is_low_address(struct word word)
{
   return (word.kind == WORD_CODE || word.kind == WORD_CODE_RELATIVE) && !word.high;
}

static struct word
low_dword(struct word a, struct word b, bool subtract, struct carry *carry)
{
   bool code_first = word_is_low_code(a) && b.kind == WORD_CONST;
   bool code_second = a.kind == WORD_CONST && word_is_low_code(b);
   if ((code_first || code_second) && (!subtract || a.kind == WORD_CODE)) {
      int64_t offset = code_first ? a.value : b.value;
      uint32_t amount = (uint32_t)(code_first ? b.value : a.value);
      uint64_t moved = subtract ? (uint64_t)offset - amount : (uint64_t)offset + amount;
      carry->kind = CARRY_CODE;
      carry->subtract = subtract;
      carry->amount = amount;
      carry->offset = offset;
      return word_code((int64_t)moved, false);
   }
   if (word_is_low_address(a) || word_is_low_address(b)) {
      *carry = carry_of(CARRY_CODE_RELATIVE);
      return word_half(WORD_CODE_RELATIVE, false);
   }
   *carry = carry_of(CARRY_UNKNOWN);
   if (word_outside(a) && word_outside(b))
      return word_of(WORD_EXTERNAL);
   return word_of(WORD_UNKNOWN);
}

static struct word
high_dword(struct word a, struct word b, struct carry carry, bool subtract)
{
   bool found = false;
   int64_t offset = 0;
   uint32_t high = 0;
   if (a.kind == WORD_CODE && a.high && b.kind == WORD_CONST) {
      found = true;
      offset = a.value;
      high = (uint32_t)b.value;
   } else if (a.kind == WORD_CONST && b.kind == WORD_CODE && b.high && !subtract) {
      found = true;
      offset = b.value;
      high = (uint32_t)a.value;
   }
   if (found && carry.kind == CARRY_CODE && carry.offset == offset && carry.subtract == subtract) {
      uint64_t total = ((uint64_t)high << 32) | carry.amount;
      uint64_t moved = subtract ? (uint64_t)offset - total : (uint64_t)offset + total;
      return word_code((int64_t)moved, true);
   }
   if (word_code_derived(a) || word_code_derived(b))
      return word_half(WORD_CODE_RELATIVE, true);
   if (word_outside(a) && word_outside(b) && !carry_from_code(carry))
      return word_of(WORD_EXTERNAL);
   return word_of(WORD_UNKNOWN);
}

static int64_t
analysis_offset(const struct analysis *analysis, int64_t pc)
{
   uint64_t base = analysis->layout.has_base ? analysis->layout.base : 0;
   return (int64_t)(agc_isa_code_layout_guest_address(&analysis->layout, (uint32_t)pc) - base);
}

static struct agc_isa_jump_target
code_target(const struct analysis *analysis, uint64_t address)
{
   uint32_t pc;
   if (agc_isa_code_layout_pc_of(&analysis->layout, address, &pc) && (int64_t)pc < analysis->code_bytes &&
       pc % 4 == 0)
      return agc_isa_jump_target_make(AGC_ISA_JUMP_STATIC, pc);
   return agc_isa_jump_target_make(AGC_ISA_JUMP_EXTERNAL, 0);
}

static bool
classify(const struct analysis *analysis, const struct state *state, const struct agc_isa_inst *inst,
         struct agc_isa_jump_target *target)
{
   if (inst->fields != AGC_ISA_SOP1)
      return false;
   struct word pair[2];
   state_read_pair(state, inst->sop1.src0, pair);
   struct word lo = pair[0];
   struct word hi = pair[1];
   if (lo.kind == WORD_UNSET || hi.kind == WORD_UNSET)
      return false;
   bool has_base = analysis->layout.has_base;
   uint64_t base = has_base ? analysis->layout.base : 0;
   if (lo.kind == WORD_CODE && !lo.high && hi.kind == WORD_CODE && hi.high && (uint32_t)lo.value == (uint32_t)hi.value)
      *target = code_target(analysis, base + (uint64_t)hi.value);
   else if (is_return(lo, false, analysis->returns, analysis->return_count) &&
            is_return(hi, true, analysis->returns, analysis->return_count))
      *target = agc_isa_jump_target_make(AGC_ISA_JUMP_RETURN, 0);
   else if (lo.kind == WORD_CONST && hi.kind == WORD_CONST && has_base)
      *target = code_target(analysis, ((uint64_t)(uint32_t)hi.value << 32) | (uint32_t)lo.value);
   else if (word_outside(lo) && word_outside(hi))
      *target = agc_isa_jump_target_make(AGC_ISA_JUMP_EXTERNAL, 0);
   else
      *target = agc_isa_jump_target_make(AGC_ISA_JUMP_OPEN, 0);
   return true;
}

static bool
entry_pair(const struct state *state, const struct agc_isa_inst *inst, uint8_t *sgpr)
{
   if (inst->fields != AGC_ISA_SOP1)
      return false;
   struct word pair[2];
   state_read_pair(state, inst->sop1.src0, pair);
   if (pair[0].kind != WORD_ENTRY || pair[1].kind != WORD_ENTRY || pair[1].value != pair[0].value + 1)
      return false;
   *sgpr = (uint8_t)pair[0].value;
   return true;
}

static struct agc_isa_operand
operand_offset(struct agc_isa_operand op, uint8_t k)
{
   switch (op.kind) {
   case AGC_ISA_OPERAND_SGPR:
      return agc_isa_operand_sgpr((uint8_t)(op.reg + k));
   case AGC_ISA_OPERAND_TTMP:
      return agc_isa_operand_ttmp((uint8_t)(op.reg + k));
   case AGC_ISA_OPERAND_VCC_LO:
      if (k == 1)
         return agc_isa_operand_special(AGC_ISA_OPERAND_VCC_HI);
      break;
   case AGC_ISA_OPERAND_EXEC_LO:
      if (k == 1)
         return agc_isa_operand_special(AGC_ISA_OPERAND_EXEC_HI);
      break;
   default:
      break;
   }
   if (k == 0)
      return op;
   return agc_isa_operand_special(AGC_ISA_OPERAND_NULL);
}

static bool
preserves_scc(const struct agc_isa_inst *inst)
{
   enum agc_isa_family family = (enum agc_isa_family)inst->family;
   if (!agc_isa_family_is_scalar_alu(family) || family == AGC_ISA_SOPP)
      return true;
   if (family == AGC_ISA_SOP1) {
      switch (inst->opcode) {
      case AGC_ISA_SOP1_MOV_B32:
      case AGC_ISA_SOP1_MOV_B64:
      case AGC_ISA_SOP1_CMOV_B32:
      case AGC_ISA_SOP1_CMOV_B64:
      case AGC_ISA_SOP1_GETPC_B64:
      case AGC_ISA_SOP1_SETPC_B64:
      case AGC_ISA_SOP1_SWAPPC_B64:
         return true;
      default:
         return false;
      }
   }
   return family == AGC_ISA_SOPK && inst->opcode == AGC_ISA_SOPK_CALL_B64;
}

static void
step_other(struct state *state, const struct agc_isa_inst *inst)
{
   enum agc_isa_family family = (enum agc_isa_family)inst->family;
   if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_RELATIVE) && agc_isa_family_is_scalar_alu(family))
      state_fill(state, WORD_UNKNOWN);
   struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
   bool code_input = false;
   unsigned count = agc_isa_inst_uses(inst, true, ranges);
   for (unsigned r = 0; r < count; r++) {
      if (ranges[r].file != AGC_ISA_REG_SCALAR)
         continue;
      for (unsigned i = ranges[r].first; i < (unsigned)ranges[r].first + ranges[r].count && i < SCALARS; i++)
         code_input |= word_code_derived(state->words[i]) || state->words[i].kind == WORD_UNKNOWN;
   }
   struct word result = word_of(code_input || agc_isa_family_is_vector_alu(family) ? WORD_UNKNOWN : WORD_EXTERNAL);
   count = agc_isa_inst_defs(inst, true, ranges);
   for (unsigned r = 0; r < count; r++) {
      if (ranges[r].file != AGC_ISA_REG_SCALAR)
         continue;
      for (unsigned i = ranges[r].first; i < (unsigned)ranges[r].first + ranges[r].count && i < SCALARS; i++)
         state->words[i] = result;
   }
}

static void
step(struct state *state, const struct agc_isa_inst *inst, const struct analysis *analysis)
{
   int64_t next_pc = (int64_t)inst->pc + 4 * (int64_t)inst->word_count;
   bool sop1 = inst->fields == AGC_ISA_SOP1 && inst->family == AGC_ISA_SOP1;
   bool sop2 = inst->fields == AGC_ISA_SOP2 && inst->family == AGC_ISA_SOP2;
   bool sopk = inst->fields == AGC_ISA_SOPK && inst->family == AGC_ISA_SOPK;
   bool writes_address = (sop1 && inst->opcode == AGC_ISA_SOP1_GETPC_B64) ||
                         (sop1 && inst->opcode == AGC_ISA_SOP1_SWAPPC_B64 &&
                          !agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_DST)) ||
                         (sopk && inst->opcode == AGC_ISA_SOPK_CALL_B64);
   if (writes_address) {
      int64_t offset = analysis_offset(analysis, next_pc);
      struct word pair[2] = {word_code(offset, false), word_code(offset, true)};
      state_write_pair(state, sop1 ? inst->sop1.sdst : inst->sopk.sdst, pair);
   } else if (sop1 && inst->opcode == AGC_ISA_SOP1_MOV_B32) {
      state_write(state, inst->sop1.sdst, state_read(state, inst->sop1.src0));
   } else if (sop1 && inst->opcode == AGC_ISA_SOP1_MOV_B64) {
      struct word pair[2];
      state_read_pair(state, inst->sop1.src0, pair);
      state_write_pair(state, inst->sop1.sdst, pair);
   } else if (sop1 && (inst->opcode == AGC_ISA_SOP1_CMOV_B32 || inst->opcode == AGC_ISA_SOP1_CMOV_B64)) {
      unsigned dwords = inst->opcode == AGC_ISA_SOP1_CMOV_B64 ? 2 : 1;
      for (uint8_t k = 0; k < dwords; k++) {
         struct word old = state_read(state, operand_offset(inst->sop1.sdst, k));
         struct word new = state_read(state, operand_offset(inst->sop1.src0, k));
         state_write(state, operand_offset(inst->sop1.sdst, k), merge_word(old, new, NULL, 0));
      }
   } else if (sop2 && (inst->opcode == AGC_ISA_SOP2_ADD_U32 || inst->opcode == AGC_ISA_SOP2_SUB_U32)) {
      struct carry carry;
      struct word word = low_dword(state_read(state, inst->sop2.src0), state_read(state, inst->sop2.src1),
                                   inst->opcode == AGC_ISA_SOP2_SUB_U32, &carry);
      state_write(state, inst->sop2.sdst, word);
      state->scc = carry;
      return;
   } else if (sop2 && (inst->opcode == AGC_ISA_SOP2_ADDC_U32 || inst->opcode == AGC_ISA_SOP2_SUBB_U32)) {
      struct word word = high_dword(state_read(state, inst->sop2.src0), state_read(state, inst->sop2.src1),
                                    state->scc, inst->opcode == AGC_ISA_SOP2_SUBB_U32);
      state_write(state, inst->sop2.sdst, word);
      state->scc = carry_of(CARRY_UNKNOWN);
      return;
   } else if (sop2 && (inst->opcode == AGC_ISA_SOP2_CSELECT_B32 || inst->opcode == AGC_ISA_SOP2_CSELECT_B64)) {
      unsigned dwords = inst->opcode == AGC_ISA_SOP2_CSELECT_B64 ? 2 : 1;
      for (uint8_t k = 0; k < dwords; k++) {
         struct word a = state_read(state, operand_offset(inst->sop2.src0, k));
         struct word b = state_read(state, operand_offset(inst->sop2.src1, k));
         state_write(state, operand_offset(inst->sop2.sdst, k), merge_word(a, b, NULL, 0));
      }
      return;
   } else if (inst->fields == AGC_ISA_SMEM && inst->family == AGC_ISA_SMEM && inst->smem.dwords > 0) {
      const struct agc_isa_smem *s = &inst->smem;
      struct word pair[2];
      state_read_pair(state, agc_isa_operand_sgpr(s->sbase), pair);
      bool from_code = !s->buffer && (word_code_derived(pair[0]) || word_code_derived(pair[1]));
      struct word loaded = word_of(from_code ? WORD_CODE_DATA : WORD_EXTERNAL);
      for (uint8_t k = 0; k < s->dwords; k++)
         state_write(state, operand_offset(s->sdata, k), loaded);
   } else {
      step_other(state, inst);
   }
   if (!preserves_scc(inst))
      state->scc = carry_of(CARRY_UNKNOWN);
}

static inline bool
is_computed_jump(const struct agc_isa_inst *inst)
{
   return inst->family == AGC_ISA_SOP1 &&
          (inst->opcode == AGC_ISA_SOP1_SETPC_B64 || inst->opcode == AGC_ISA_SOP1_SWAPPC_B64);
}

static inline bool
is_call(const struct agc_isa_inst *inst)
{
   return agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CALL);
}

static int
compare_u32(const void *a, const void *b)
{
   uint32_t x = *(const uint32_t *)a;
   uint32_t y = *(const uint32_t *)b;
   return x < y ? -1 : x > y;
}

static int
compare_i64(const void *a, const void *b)
{
   int64_t x = *(const int64_t *)a;
   int64_t y = *(const int64_t *)b;
   return x < y ? -1 : x > y;
}

static size_t
sort_unique_u32(uint32_t *values, size_t count)
{
   qsort(values, count, sizeof(*values), compare_u32);
   size_t kept = 0;
   for (size_t i = 0; i < count; i++) {
      if (!kept || values[kept - 1] != values[i])
         values[kept++] = values[i];
   }
   return kept;
}

static size_t
sort_unique_i64(int64_t *values, size_t count)
{
   qsort(values, count, sizeof(*values), compare_i64);
   size_t kept = 0;
   for (size_t i = 0; i < count; i++) {
      if (!kept || values[kept - 1] != values[i])
         values[kept++] = values[i];
   }
   return kept;
}

void
agc_isa_computed_flow_finish(struct agc_isa_computed_flow *flow)
{
   free(flow->jumps);
   free(flow->entries);
   free(flow->external);
   memset(flow, 0, sizeof(*flow));
}

static void *
copy_items(const void *items, size_t count, size_t size, bool *ok)
{
   if (!count)
      return NULL;
   void *copy = malloc(count * size);
   if (!copy) {
      *ok = false;
      return NULL;
   }
   memcpy(copy, items, count * size);
   return copy;
}

static bool
agc_isa_computed_flow_copy(struct agc_isa_computed_flow *dst, const struct agc_isa_computed_flow *src)
{
   bool ok = true;
   struct agc_isa_computed_flow copy = {0};
   copy.jumps = copy_items(src->jumps, src->jump_count, sizeof(*src->jumps), &ok);
   copy.entries = copy_items(src->entries, src->entry_count, sizeof(*src->entries), &ok);
   copy.external = copy_items(src->external, src->external_count, sizeof(*src->external), &ok);
   if (!ok) {
      agc_isa_computed_flow_finish(&copy);
      return false;
   }
   copy.jump_count = src->jump_count;
   copy.entry_count = src->entry_count;
   copy.external_count = src->external_count;
   agc_isa_computed_flow_finish(dst);
   *dst = copy;
   return true;
}

bool
agc_isa_computed_flow_equal(const struct agc_isa_computed_flow *a, const struct agc_isa_computed_flow *b)
{
   if (a->jump_count != b->jump_count || a->entry_count != b->entry_count ||
       a->external_count != b->external_count)
      return false;
   for (size_t i = 0; i < a->jump_count; i++) {
      if (a->jumps[i].pc != b->jumps[i].pc || !agc_isa_jump_target_equal(a->jumps[i].target, b->jumps[i].target))
         return false;
   }
   for (size_t i = 0; i < a->entry_count; i++) {
      if (a->entries[i] != b->entries[i])
         return false;
   }
   for (size_t i = 0; i < a->external_count; i++) {
      const struct agc_isa_external_target *x = &a->external[i];
      const struct agc_isa_external_target *y = &b->external[i];
      if (x->pc != y->pc || x->call != y->call || x->has_sgpr != y->has_sgpr ||
          (x->has_sgpr && x->sgpr != y->sgpr))
         return false;
   }
   return true;
}

bool
agc_isa_computed_flow_has_dynamic_jumps(const struct agc_isa_computed_flow *flow)
{
   for (size_t i = 0; i < flow->jump_count; i++) {
      if (flow->jumps[i].target.kind != AGC_ISA_JUMP_STATIC)
         return true;
   }
   return false;
}

static bool
agc_isa_computed_flow_is_open(const struct agc_isa_computed_flow *flow)
{
   for (size_t i = 0; i < flow->jump_count; i++) {
      if (flow->jumps[i].target.kind == AGC_ISA_JUMP_OPEN)
         return true;
   }
   return false;
}

const struct agc_isa_jump_target *
agc_isa_computed_flow_jump(const struct agc_isa_computed_flow *flow, uint32_t pc)
{
   size_t lo = 0;
   size_t hi = flow->jump_count;
   while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      if (flow->jumps[mid].pc < pc)
         lo = mid + 1;
      else
         hi = mid;
   }
   if (lo == flow->jump_count || flow->jumps[lo].pc != pc)
      return NULL;
   return &flow->jumps[lo].target;
}

bool
agc_isa_computed_flow_has_entry(const struct agc_isa_computed_flow *flow, uint32_t pc)
{
   size_t lo = 0;
   size_t hi = flow->entry_count;
   while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      if (flow->entries[mid] < pc)
         lo = mid + 1;
      else
         hi = mid;
   }
   return lo < flow->entry_count && flow->entries[lo] == pc;
}

bool
agc_isa_computed_decode_scope(const struct agc_isa_computed_flow *flow, uint32_t **entries, size_t *entry_count,
                              bool *through_jumps)
{
   uint32_t *merged = malloc((flow->entry_count + flow->jump_count + 1) * sizeof(*merged));
   if (!merged)
      return false;
   size_t count = 0;
   for (size_t i = 0; i < flow->entry_count; i++)
      merged[count++] = flow->entries[i];
   for (size_t i = 0; i < flow->jump_count; i++) {
      if (flow->jumps[i].target.kind == AGC_ISA_JUMP_STATIC)
         merged[count++] = flow->jumps[i].target.pc;
   }
   *entries = merged;
   *entry_count = sort_unique_u32(merged, count);
   *through_jumps = agc_isa_computed_flow_is_open(flow);
   return true;
}

bool
agc_isa_computed_apply(struct agc_isa_ir *ir, const struct agc_isa_computed_flow *flow, struct agc_isa_error *error)
{
   for (size_t i = 0; i < flow->jump_count; i++) {
      const struct agc_isa_jump *jump = &flow->jumps[i];
      if (jump->target.kind != AGC_ISA_JUMP_STATIC)
         continue;
      size_t index;
      if (!agc_isa_ir_instruction_at(ir, jump->target.pc, &index))
         return agc_isa_fail_branch(error, jump->pc, jump->target.pc);
      if (agc_isa_ir_instruction_at(ir, jump->pc, &index)) {
         ir->instructions[index].branch_target = jump->target.pc;
         ir->instructions[index].has_branch_target = true;
      }
   }
   if (!agc_isa_computed_flow_copy(&ir->flow, flow))
      return agc_isa_fail_memory(error);
   return true;
}

struct jump_map {
   struct agc_isa_jump *jumps;
   size_t jump_count;
   size_t jump_capacity;
   struct agc_isa_external_target *external;
   size_t external_count;
   size_t external_capacity;
};

static bool
insert_jump(struct jump_map *map, uint32_t pc, struct agc_isa_jump_target target)
{
   size_t at = 0;
   while (at < map->jump_count && map->jumps[at].pc < pc)
      at++;
   if (at < map->jump_count && map->jumps[at].pc == pc) {
      map->jumps[at].target = target;
      return true;
   }
   if (map->jump_count == map->jump_capacity) {
      size_t capacity = map->jump_capacity ? map->jump_capacity * 2 : 8;
      struct agc_isa_jump *jumps = realloc(map->jumps, capacity * sizeof(*jumps));
      if (!jumps)
         return false;
      map->jumps = jumps;
      map->jump_capacity = capacity;
   }
   memmove(&map->jumps[at + 1], &map->jumps[at], (map->jump_count - at) * sizeof(*map->jumps));
   map->jumps[at].pc = pc;
   map->jumps[at].target = target;
   map->jump_count++;
   return true;
}

static bool
insert_external(struct jump_map *map, struct agc_isa_external_target target)
{
   size_t at = 0;
   while (at < map->external_count && map->external[at].pc < target.pc)
      at++;
   if (at < map->external_count && map->external[at].pc == target.pc) {
      map->external[at] = target;
      return true;
   }
   if (map->external_count == map->external_capacity) {
      size_t capacity = map->external_capacity ? map->external_capacity * 2 : 8;
      struct agc_isa_external_target *external = realloc(map->external, capacity * sizeof(*external));
      if (!external)
         return false;
      map->external = external;
      map->external_capacity = capacity;
   }
   memmove(&map->external[at + 1], &map->external[at], (map->external_count - at) * sizeof(*map->external));
   map->external[at] = target;
   map->external_count++;
   return true;
}

static void
merge_into(struct state *entry, size_t successor, const struct state *outgoing, const struct analysis *analysis,
           struct agc_isa_work *work)
{
   if (state_merge(&entry[successor], outgoing, analysis->returns, analysis->return_count))
      agc_isa_work_insert(work, successor);
}

static bool
run_block(const struct analysis *analysis, size_t id, struct state *entry, struct state *scratch,
          const size_t *return_blocks, size_t return_block_count, const size_t *region_blocks,
          size_t region_block_count, struct jump_map *map, struct agc_isa_work *work)
{
   const struct agc_isa_ir *ir = analysis->ir;
   const struct agc_isa_block *block = &ir->blocks[id];
   struct state *state = &scratch[0];
   *state = entry[id];
   const struct agc_isa_inst *last = &ir->instructions[block->end - 1];
   for (size_t i = block->first; i + 1 < block->end; i++)
      step(state, &ir->instructions[i], analysis);
   if (!is_computed_jump(last) && !is_call(last)) {
      step(state, last, analysis);
      for (uint32_t s = 0; s < block->successor_count; s++)
         merge_into(entry, block->successors[s], state, analysis, work);
      return true;
   }
   struct agc_isa_jump_target target = {0, 0};
   bool has_target;
   if (last->family == AGC_ISA_SOPK) {
      has_target = last->has_branch_target;
      if (has_target)
         target = agc_isa_jump_target_make(AGC_ISA_JUMP_STATIC, last->branch_target);
   } else {
      has_target = classify(analysis, state, last, &target);
   }
   if (has_target && is_computed_jump(last) && !insert_jump(map, last->pc, target))
      return false;
   bool returns = is_call(last) && !agc_isa_inst_has_flag(last, AGC_ISA_FLAG_NO_DST);
   bool external = has_target && target.kind == AGC_ISA_JUMP_EXTERNAL;
   if (external) {
      struct agc_isa_external_target found = {.pc = last->pc, .call = returns};
      found.has_sgpr = entry_pair(state, last, &found.sgpr);
      if (!insert_external(map, found))
         return false;
   }
   step(state, last, analysis);
   if (has_target && target.kind == AGC_ISA_JUMP_STATIC) {
      size_t to;
      if (agc_isa_block_at(ir, target.pc, &to))
         merge_into(entry, to, state, analysis, work);
   } else if (has_target && target.kind == AGC_ISA_JUMP_RETURN && !returns) {
      for (size_t i = 0; i < return_block_count; i++)
         merge_into(entry, return_blocks[i], state, analysis, work);
   } else if (external) {
      for (size_t i = 0; i < region_block_count; i++)
         merge_into(entry, region_blocks[i], state, analysis, work);
   }
   bool linked = external && region_block_count != 0;
   if (returns && !linked && !(has_target && target.kind == AGC_ISA_JUMP_STATIC)) {
      struct state *clobbered = &scratch[1];
      state_fill(clobbered, external ? WORD_EXTERNAL : WORD_UNKNOWN);
      clobbered->scc = carry_of(CARRY_UNKNOWN);
      if (last->fields == AGC_ISA_SOP1 || last->fields == AGC_ISA_SOPK) {
         struct agc_isa_operand sdst = last->fields == AGC_ISA_SOP1 ? last->sop1.sdst : last->sopk.sdst;
         struct word pair[2];
         state_read_pair(state, sdst, pair);
         state_write_pair(clobbered, sdst, pair);
      }
      for (uint32_t s = 0; s < block->successor_count; s++)
         merge_into(entry, block->successors[s], clobbered, analysis, work);
   }
   return true;
}

static bool
finish_flow(const struct analysis *analysis, struct jump_map *map, struct agc_isa_computed_flow *flow)
{
   const struct agc_isa_ir *ir = analysis->ir;
   flow->jumps = map->jumps;
   flow->jump_count = map->jump_count;
   flow->external = map->external;
   flow->external_count = map->external_count;
   map->jumps = NULL;
   map->external = NULL;
   bool dynamic = agc_isa_computed_flow_has_dynamic_jumps(flow);
   bool open = agc_isa_computed_flow_is_open(flow);
   size_t capacity = analysis->layout.region_count + 1;
   if (dynamic)
      capacity += analysis->return_pc_count;
   if (open)
      capacity += ir->instruction_count;
   uint32_t *entries = malloc(capacity * sizeof(*entries));
   if (!entries)
      return false;
   size_t count = 0;
   if (dynamic) {
      for (size_t i = 0; i < analysis->return_pc_count; i++)
         entries[count++] = analysis->return_pcs[i];
   }
   for (size_t i = 0; i < analysis->layout.region_count; i++)
      entries[count++] = analysis->layout.regions[i].start;
   if (open) {
      for (size_t i = 0; i < ir->instruction_count; i++)
         entries[count++] = ir->instructions[i].pc;
   }
   flow->entries = entries;
   flow->entry_count = sort_unique_u32(entries, count);
   return true;
}

static bool
region_entries(const struct agc_isa_ir *ir, struct agc_isa_computed_flow *flow)
{
   if (!ir->layout.region_count)
      return true;
   flow->entries = malloc(ir->layout.region_count * sizeof(*flow->entries));
   if (!flow->entries)
      return false;
   for (size_t i = 0; i < ir->layout.region_count; i++)
      flow->entries[i] = ir->layout.regions[i].start;
   flow->entry_count = sort_unique_u32(flow->entries, ir->layout.region_count);
   return true;
}

bool
agc_isa_computed_analyze(const struct agc_isa_ir *ir, bool has_base, uint64_t base, size_t code_words,
                         struct agc_isa_computed_flow *flow, struct agc_isa_error *error)
{
   agc_isa_computed_flow_finish(flow);
   size_t calls = 0;
   bool any = false;
   for (size_t i = 0; i < ir->instruction_count; i++) {
      calls += is_call(&ir->instructions[i]);
      any |= is_computed_jump(&ir->instructions[i]) || is_call(&ir->instructions[i]);
   }
   if (!any || !ir->block_count) {
      if (!region_entries(ir, flow))
         return agc_isa_fail_memory(error);
      return true;
   }

   struct analysis analysis = {.ir = ir, .layout = ir->layout, .code_bytes = 4 * (int64_t)code_words};
   if (has_base) {
      analysis.layout.has_base = true;
      analysis.layout.base = base;
   }
   if (analysis.layout.own_bytes == 0)
      analysis.layout.own_bytes = 4 * (uint32_t)code_words;

   size_t count = ir->block_count;
   size_t region_count = ir->layout.region_count;
   struct jump_map map = {0};
   struct agc_isa_work work = {0};
   struct state *entry = malloc((count + 2) * sizeof(*entry));
   size_t *block_lists = malloc((calls + region_count + 1) * sizeof(*block_lists));
   analysis.return_pcs = malloc((calls + 1) * sizeof(*analysis.return_pcs));
   analysis.returns = malloc((calls + 1) * sizeof(*analysis.returns));
   bool ok = entry && block_lists && analysis.return_pcs && analysis.returns && agc_isa_work_init(&work, count);
   if (ok) {
      for (size_t i = 0; i < ir->instruction_count; i++) {
         const struct agc_isa_inst *inst = &ir->instructions[i];
         if (is_call(inst))
            analysis.return_pcs[analysis.return_pc_count++] = agc_isa_inst_end_pc(inst);
      }
      for (size_t i = 0; i < analysis.return_pc_count; i++)
         analysis.returns[i] = analysis_offset(&analysis, (int64_t)analysis.return_pcs[i]);
      analysis.return_count = sort_unique_i64(analysis.returns, analysis.return_pc_count);

      size_t *return_blocks = block_lists;
      size_t return_block_count = 0;
      for (size_t i = 0; i < analysis.return_pc_count; i++) {
         if (agc_isa_block_at(ir, analysis.return_pcs[i], &return_blocks[return_block_count]))
            return_block_count++;
      }
      size_t *region_blocks = block_lists + return_block_count;
      size_t region_block_count = 0;
      for (size_t i = 0; i < region_count; i++) {
         if (agc_isa_block_at(ir, ir->layout.regions[i].start, &region_blocks[region_block_count]))
            region_block_count++;
      }

      struct state *scratch = entry + count;
      for (size_t id = 0; id < count; id++)
         state_unset(&entry[id]);
      state_entry(&entry[0]);
      if (analysis.layout.has_second_half) {
         uint64_t address = analysis.layout.second_half;
         entry[0].words[SECOND_HALF_SGPR] = word_const((uint32_t)address);
         entry[0].words[SECOND_HALF_SGPR + 1] = word_const((uint32_t)(address >> 32));
      }
      for (size_t id = 0; id < count; id++)
         agc_isa_work_insert(&work, id);
      size_t id;
      while (ok && agc_isa_work_pop_first(&work, &id)) {
         if (!state_reached(&entry[id]))
            continue;
         ok = run_block(&analysis, id, entry, scratch, return_blocks, return_block_count, region_blocks,
                        region_block_count, &map, &work);
      }
      ok = ok && finish_flow(&analysis, &map, flow);
   }
   free(entry);
   free(block_lists);
   free(analysis.return_pcs);
   free(analysis.returns);
   free(work.bits);
   free(map.jumps);
   free(map.external);
   if (!ok) {
      agc_isa_computed_flow_finish(flow);
      return agc_isa_fail_memory(error);
   }
   return true;
}
