#include "agc_isa_analyze_priv.h"

#define SGPRS 106
#define UNTRACKED 0xff
#define FILL_GROUP_SHIFT 6
#define VECTORS 256

enum val_kind {
   VAL_UNSET,
   VAL_UNKNOWN,
   VAL_CONST,
   VAL_MASKED,
   VAL_WORD
};

struct val {
   uint8_t kind;
   uint8_t dword;
   bool load;
   uint32_t value;
   int64_t add;
};

enum carry_kind {
   CARRY_UNSET,
   CARRY_UNKNOWN,
   CARRY_POINTER
};

struct carry {
   uint8_t kind;
   uint8_t dword;
   bool load;
   bool subtract;
   uint32_t index;
   uint32_t amount;
   int64_t add;
};

struct state {
   struct val *regs;
   struct carry scc;
   bool reached;
};

struct table {
   uint8_t root;
   uint8_t root_form;
   uint32_t link_first;
   uint32_t link_count;
   uint32_t offset_mask;
   int64_t offset;
};

struct use_record {
   struct agc_isa_descriptor_use use;
   uint32_t table;
   uint8_t dword;
};

struct use_list {
   struct use_record *records;
   size_t count;
   size_t capacity;
};

struct ctx {
   const struct agc_isa_ir *ir;
   bool wave64;
   uint16_t user_start;
   uint16_t user_end;
   uint8_t of[SGPRS];
   unsigned slot_count;
   struct table *tables;
   size_t table_count;
   size_t table_capacity;
   struct agc_isa_table_link *links;
   size_t link_count;
   size_t link_capacity;
   bool failed;
};

static inline struct val
val_of(enum val_kind kind)
{
   struct val val = {(uint8_t)kind, 0, false, 0, 0};
   return val;
}

static inline struct val
val_const(uint32_t value)
{
   struct val val = {VAL_CONST, 0, false, value, 0};
   return val;
}

static inline struct val
val_masked(uint32_t mask)
{
   struct val val = {VAL_MASKED, 0, false, mask, 0};
   return val;
}

static inline struct val
val_word(bool load, uint32_t index, uint8_t dword, int64_t add)
{
   struct val val = {VAL_WORD, dword, load, load ? index : 0, add};
   return val;
}

static inline bool
val_equal(struct val a, struct val b)
{
   return a.kind == b.kind && a.dword == b.dword && a.load == b.load && a.value == b.value && a.add == b.add;
}

static inline struct carry
carry_of(enum carry_kind kind)
{
   struct carry carry = {(uint8_t)kind, 0, false, false, 0, 0, 0};
   return carry;
}

static inline bool
carry_equal(struct carry a, struct carry b)
{
   return a.kind == b.kind && a.dword == b.dword && a.load == b.load && a.subtract == b.subtract &&
          a.index == b.index && a.amount == b.amount && a.add == b.add;
}

static inline bool
is_user(const struct ctx *ctx, unsigned first, unsigned count)
{
   return ctx->user_start <= first && first + count <= ctx->user_end;
}

static inline bool
tracks(const struct ctx *ctx, unsigned n)
{
   return n < SGPRS && ctx->of[n] != UNTRACKED;
}

static inline struct val
state_reg(const struct ctx *ctx, const struct state *state, unsigned n)
{
   if (tracks(ctx, n))
      return state->regs[ctx->of[n]];
   return val_of(VAL_UNKNOWN);
}

static struct val
state_read(const struct ctx *ctx, const struct state *state, struct agc_isa_operand op)
{
   uint32_t bits;
   switch (op.kind) {
   case AGC_ISA_OPERAND_SGPR:
      return state_reg(ctx, state, op.reg);
   case AGC_ISA_OPERAND_INLINE_INT:
      return val_const((uint32_t)op.inline_int);
   case AGC_ISA_OPERAND_INLINE_FLOAT:
      if (agc_isa_inline_float_bits((uint8_t)op.inline_float, &bits))
         return val_const(bits);
      return val_of(VAL_UNKNOWN);
   case AGC_ISA_OPERAND_LITERAL:
      return val_const(op.literal);
   case AGC_ISA_OPERAND_NULL:
      return val_const(0);
   default:
      return val_of(VAL_UNKNOWN);
   }
}

static struct val
state_read_high(const struct ctx *ctx, const struct state *state, struct agc_isa_operand op)
{
   switch (op.kind) {
   case AGC_ISA_OPERAND_SGPR:
      return state_reg(ctx, state, op.reg + 1);
   case AGC_ISA_OPERAND_INLINE_INT:
      return val_const(op.inline_int < 0 ? UINT32_MAX : 0);
   case AGC_ISA_OPERAND_NULL:
      return val_const(0);
   default:
      return val_of(VAL_UNKNOWN);
   }
}

static inline void
state_write_reg(const struct ctx *ctx, struct state *state, unsigned n, struct val value)
{
   if (tracks(ctx, n))
      state->regs[ctx->of[n]] = value;
}

static inline void
state_write(const struct ctx *ctx, struct state *state, struct agc_isa_operand op, struct val value)
{
   if (op.kind == AGC_ISA_OPERAND_SGPR)
      state_write_reg(ctx, state, op.reg, value);
}

static void
state_fill(const struct ctx *ctx, struct state *state, enum val_kind kind)
{
   for (unsigned i = 0; i < ctx->slot_count; i++)
      state->regs[i] = val_of(kind);
}

static void
state_unset(const struct ctx *ctx, struct state *state)
{
   state_fill(ctx, state, VAL_UNSET);
   state->scc = carry_of(CARRY_UNSET);
   state->reached = false;
}

static void
state_unknown(const struct ctx *ctx, struct state *state)
{
   state_fill(ctx, state, VAL_UNKNOWN);
   state->scc = carry_of(CARRY_UNKNOWN);
   state->reached = true;
}

static void
state_entry(const struct ctx *ctx, struct state *state)
{
   state_unknown(ctx, state);
   for (unsigned n = 0; n < SGPRS; n++) {
      if (ctx->of[n] != UNTRACKED)
         state->regs[ctx->of[n]] = val_word(false, 0, (uint8_t)n, 0);
   }
}

static void
state_copy(const struct ctx *ctx, struct state *state, const struct state *other)
{
   if (ctx->slot_count)
      memcpy(state->regs, other->regs, ctx->slot_count * sizeof(*state->regs));
   state->scc = other->scc;
   state->reached = other->reached;
}

static inline struct val
join_val(struct val a, struct val b)
{
   if (a.kind == VAL_UNSET)
      return b;
   if (b.kind == VAL_UNSET)
      return a;
   if (val_equal(a, b))
      return a;
   return val_of(VAL_UNKNOWN);
}

static bool
state_join(const struct ctx *ctx, struct state *state, const struct state *other)
{
   bool changed = other->reached && !state->reached;
   state->reached |= other->reached;
   for (unsigned i = 0; i < ctx->slot_count; i++) {
      struct val joined = join_val(state->regs[i], other->regs[i]);
      changed |= !val_equal(joined, state->regs[i]);
      state->regs[i] = joined;
   }
   struct carry scc;
   if (state->scc.kind == CARRY_UNSET)
      scc = other->scc;
   else if (other->scc.kind == CARRY_UNSET)
      scc = state->scc;
   else if (carry_equal(state->scc, other->scc))
      scc = state->scc;
   else
      scc = carry_of(CARRY_UNKNOWN);
   changed |= !carry_equal(scc, state->scc);
   state->scc = scc;
   return changed;
}

static struct val
low_dword(struct val a, struct val b, bool subtract, struct carry *carry)
{
   if (a.kind == VAL_UNSET || b.kind == VAL_UNSET) {
      *carry = carry_of(CARRY_UNSET);
      return val_of(VAL_UNSET);
   }
   bool word_first = a.kind == VAL_WORD && b.kind == VAL_CONST;
   bool word_second = a.kind == VAL_CONST && b.kind == VAL_WORD;
   if ((word_first || word_second) && (!subtract || a.kind == VAL_WORD)) {
      struct val word = word_first ? a : b;
      uint32_t amount = word_first ? b.value : a.value;
      uint64_t moved = subtract ? (uint64_t)word.add - amount : (uint64_t)word.add + amount;
      carry->kind = CARRY_POINTER;
      carry->dword = word.dword;
      carry->load = word.load;
      carry->subtract = subtract;
      carry->index = word.value;
      carry->amount = amount;
      carry->add = word.add;
      return val_word(word.load, word.value, word.dword, (int64_t)moved);
   }
   *carry = carry_of(CARRY_UNKNOWN);
   if (a.kind == VAL_CONST && b.kind == VAL_CONST)
      return val_const(subtract ? a.value - b.value : a.value + b.value);
   return val_of(VAL_UNKNOWN);
}

static struct val
high_dword(struct val a, struct val b, struct carry carry, bool subtract)
{
   bool found = false;
   struct val word = val_of(VAL_UNKNOWN);
   uint32_t high = 0;
   if (a.kind == VAL_WORD && b.kind == VAL_CONST) {
      found = true;
      word = a;
      high = b.value;
   } else if (a.kind == VAL_CONST && b.kind == VAL_WORD && !subtract) {
      found = true;
      word = b;
      high = a.value;
   }
   if (carry.kind == CARRY_UNSET)
      return val_of(VAL_UNSET);
   if (found && carry.kind == CARRY_POINTER && word.load == carry.load && word.value == carry.index &&
       word.dword == (uint8_t)(carry.dword + 1) && word.add == carry.add && subtract == carry.subtract) {
      uint64_t total = ((uint64_t)high << 32) | carry.amount;
      uint64_t moved = subtract ? (uint64_t)word.add - total : (uint64_t)word.add + total;
      return val_word(word.load, word.value, word.dword, (int64_t)moved);
   }
   return val_of(VAL_UNKNOWN);
}

static unsigned
descriptor_operands(const struct agc_isa_inst *inst, uint8_t first[2], uint8_t count[2])
{
   switch (inst->fields) {
   case AGC_ISA_SMEM:
      if (inst->smem.buffer && inst->smem.dwords > 0 && !agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_SRC)) {
         first[0] = inst->smem.sbase;
         count[0] = 4;
         return 1;
      }
      return 0;
   case AGC_ISA_MUBUF:
   case AGC_ISA_MTBUF:
      if (inst->buffer.kind != AGC_ISA_MEM_CONTROL) {
         first[0] = inst->buffer.srsrc;
         count[0] = 4;
         return 1;
      }
      return 0;
   case AGC_ISA_MIMG:
      first[0] = inst->mimg.srsrc;
      count[0] = inst->mimg.r128 ? 4 : 8;
      if (!inst->mimg.uses_sampler)
         return 1;
      first[1] = inst->mimg.ssamp;
      count[1] = 4;
      return 2;
   default:
      return 0;
   }
}

static bool
is_table_load(const struct agc_isa_inst *inst)
{
   const struct agc_isa_smem *s = &inst->smem;
   if (s->dwords == 0 || agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_SRC))
      return false;
   if (!s->buffer && inst->opcode > AGC_ISA_SMEM_LOAD_DWORDX16)
      return false;
   const struct agc_isa_mem_op *op = agc_isa_smem_op(inst->opcode);
   return op && op->kind == AGC_ISA_MEM_LOAD;
}

static bool
mark(bool tracked[SGPRS], unsigned first, unsigned count)
{
   bool changed = false;
   for (unsigned n = first; n < first + count; n++) {
      if (n < SGPRS) {
         changed |= !tracked[n];
         tracked[n] = true;
      }
   }
   return changed;
}

static bool
mark_op(bool tracked[SGPRS], struct agc_isa_operand op, unsigned count)
{
   if (op.kind != AGC_ISA_OPERAND_SGPR)
      return false;
   return mark(tracked, op.reg, count);
}

static bool
any_tracked(const bool tracked[SGPRS], unsigned first, unsigned count)
{
   for (unsigned n = first; n < first + count; n++) {
      if (n < SGPRS && tracked[n])
         return true;
   }
   return false;
}

static bool
is_tracked_sop2(uint16_t opcode)
{
   switch (opcode) {
   case AGC_ISA_SOP2_ADD_U32:
   case AGC_ISA_SOP2_SUB_U32:
   case AGC_ISA_SOP2_ADDC_U32:
   case AGC_ISA_SOP2_SUBB_U32:
   case AGC_ISA_SOP2_AND_B32:
   case AGC_ISA_SOP2_LSHL_B32:
   case AGC_ISA_SOP2_LSHR_B32:
      return true;
   default:
      return false;
   }
}

static void
slots_init(struct ctx *ctx)
{
   const struct agc_isa_ir *ir = ctx->ir;
   bool tracked[SGPRS] = {false};
   for (size_t i = 0; i < ir->instruction_count; i++) {
      const struct agc_isa_inst *inst = &ir->instructions[i];
      uint8_t first[2];
      uint8_t count[2];
      unsigned operands = descriptor_operands(inst, first, count);
      for (unsigned k = 0; k < operands; k++)
         mark(tracked, first[k], count[k]);
      if (inst->fields == AGC_ISA_SMEM) {
         const struct agc_isa_smem *s = &inst->smem;
         const struct agc_isa_mem_op *op = agc_isa_smem_op(inst->opcode);
         if (!s->buffer && s->dwords > 0 && op && op->kind != AGC_ISA_MEM_CONTROL) {
            mark(tracked, s->sbase, 2);
            mark_op(tracked, s->soffset, 1);
         }
      }
   }
   bool carry = false;
   for (;;) {
      bool changed = false;
      for (size_t i = 0; i < ir->instruction_count; i++) {
         const struct agc_isa_inst *inst = &ir->instructions[i];
         if (inst->fields == AGC_ISA_SOP1 && inst->family == AGC_ISA_SOP1 &&
             (inst->opcode == AGC_ISA_SOP1_MOV_B32 || inst->opcode == AGC_ISA_SOP1_MOV_B64)) {
            unsigned dwords = inst->opcode == AGC_ISA_SOP1_MOV_B64 ? 2 : 1;
            if (inst->sop1.sdst.kind == AGC_ISA_OPERAND_SGPR && any_tracked(tracked, inst->sop1.sdst.reg, dwords))
               changed |= mark_op(tracked, inst->sop1.src0, dwords);
         } else if (inst->fields == AGC_ISA_SOP2 && inst->family == AGC_ISA_SOP2 && is_tracked_sop2(inst->opcode)) {
            bool with_carry = inst->opcode == AGC_ISA_SOP2_ADDC_U32 || inst->opcode == AGC_ISA_SOP2_SUBB_U32;
            bool builds = inst->sop2.sdst.kind == AGC_ISA_OPERAND_SGPR && any_tracked(tracked, inst->sop2.sdst.reg, 1);
            if (builds || (carry && !with_carry)) {
               changed |= mark_op(tracked, inst->sop2.src0, 1);
               changed |= mark_op(tracked, inst->sop2.src1, 1);
            }
            if (builds && with_carry && !carry) {
               carry = true;
               changed = true;
            }
         } else if (inst->fields == AGC_ISA_SMEM && inst->family == AGC_ISA_SMEM && is_table_load(inst)) {
            const struct agc_isa_smem *s = &inst->smem;
            if (s->sdata.kind == AGC_ISA_OPERAND_SGPR && any_tracked(tracked, s->sdata.reg, s->dwords)) {
               changed |= mark(tracked, s->sbase, s->buffer ? 4 : 2);
               changed |= mark_op(tracked, s->soffset, 1);
            }
         } else if ((inst->fields == AGC_ISA_MUBUF || inst->fields == AGC_ISA_MTBUF) &&
                    inst->buffer.kind != AGC_ISA_MEM_CONTROL) {
            changed |= mark_op(tracked, inst->buffer.soffset, 1);
         }
      }
      if (!changed)
         break;
   }
   ctx->slot_count = 0;
   for (unsigned n = 0; n < SGPRS; n++)
      ctx->of[n] = tracked[n] ? (uint8_t)ctx->slot_count++ : UNTRACKED;
}

static bool
table_equal(const struct ctx *ctx, const struct table *a, const struct table *b)
{
   if (a->root != b->root || a->root_form != b->root_form || a->offset != b->offset ||
       a->offset_mask != b->offset_mask || a->link_count != b->link_count)
      return false;
   for (uint32_t i = 0; i < a->link_count; i++) {
      const struct agc_isa_table_link *x = &ctx->links[a->link_first + i];
      const struct agc_isa_table_link *y = &ctx->links[b->link_first + i];
      if (x->offset != y->offset || x->form != y->form)
         return false;
   }
   return true;
}

static bool
reserve_links(struct ctx *ctx, size_t extra)
{
   if (ctx->link_count + extra <= ctx->link_capacity)
      return true;
   size_t capacity = ctx->link_capacity ? ctx->link_capacity * 2 : 64;
   while (capacity < ctx->link_count + extra)
      capacity *= 2;
   struct agc_isa_table_link *links = realloc(ctx->links, capacity * sizeof(*links));
   if (!links) {
      ctx->failed = true;
      return false;
   }
   ctx->links = links;
   ctx->link_capacity = capacity;
   return true;
}

static bool
intern(struct ctx *ctx, const struct table *table, uint32_t *index)
{
   for (size_t i = 0; i < ctx->table_count; i++) {
      if (table_equal(ctx, &ctx->tables[i], table)) {
         *index = (uint32_t)i;
         return true;
      }
   }
   if (ctx->table_count == ctx->table_capacity) {
      size_t capacity = ctx->table_capacity ? ctx->table_capacity * 2 : 32;
      struct table *tables = realloc(ctx->tables, capacity * sizeof(*tables));
      if (!tables) {
         ctx->failed = true;
         return false;
      }
      ctx->tables = tables;
      ctx->table_capacity = capacity;
   }
   ctx->tables[ctx->table_count] = *table;
   *index = (uint32_t)ctx->table_count++;
   return true;
}

static unsigned
count_ones(uint32_t value)
{
   unsigned count = 0;
   for (; value; value &= value - 1)
      count++;
   return count;
}

static bool
table_load(struct ctx *ctx, const struct state *state, const struct agc_isa_smem *s, uint32_t *index)
{
   struct val soffset = state_read(ctx, state, s->soffset);
   uint64_t extra = 0;
   uint32_t offset_mask = 0;
   if (soffset.kind == VAL_CONST)
      extra = soffset.value;
   else if (soffset.kind == VAL_MASKED && count_ones(soffset.value) <= 8)
      offset_mask = soffset.value;
   else
      return false;
   unsigned base = s->sbase;
   struct val low = state_reg(ctx, state, base);
   if (low.kind != VAL_WORD)
      return false;
   enum agc_isa_address_form form;
   int64_t add;
   if (s->buffer) {
      if (low.add != 0)
         return false;
      for (uint8_t k = 1; k < 4; k++) {
         struct val wanted = val_word(low.load, low.value, (uint8_t)(low.dword + k), 0);
         if (!val_equal(state_reg(ctx, state, base + k), wanted))
            return false;
      }
      add = 0;
      form = AGC_ISA_ADDRESS_BUFFER;
   } else {
      struct val wanted = val_word(low.load, low.value, (uint8_t)(low.dword + 1), low.add);
      if (!val_equal(state_reg(ctx, state, base + 1), wanted))
         return false;
      add = low.add;
      form = AGC_ISA_ADDRESS_POINTER;
   }
   struct table table = {0};
   table.offset = (int64_t)((uint64_t)add + (uint64_t)(int64_t)s->offset + extra);
   table.offset_mask = offset_mask;
   if (!low.load) {
      if (!is_user(ctx, low.dword, form == AGC_ISA_ADDRESS_BUFFER ? 4 : 2))
         return false;
      table.root = low.dword;
      table.root_form = (uint8_t)form;
      table.link_first = (uint32_t)ctx->link_count;
      table.link_count = 0;
   } else {
      struct table parent = ctx->tables[low.value];
      if (parent.link_count >= AGC_ISA_MAX_TABLE_LINKS || parent.offset_mask != 0)
         return false;
      if (!reserve_links(ctx, (size_t)parent.link_count + 1))
         return false;
      table.root = parent.root;
      table.root_form = parent.root_form;
      table.link_first = (uint32_t)ctx->link_count;
      table.link_count = parent.link_count + 1;
      for (uint32_t i = 0; i < parent.link_count; i++)
         ctx->links[table.link_first + i] = ctx->links[parent.link_first + i];
      struct agc_isa_table_link *link = &ctx->links[table.link_first + parent.link_count];
      memset(link, 0, sizeof(*link));
      link->offset = (int64_t)((uint64_t)parent.offset + 4 * (uint64_t)low.dword);
      link->form = (uint8_t)form;
   }
   size_t before = ctx->table_count;
   if (!intern(ctx, &table, index))
      return false;
   if (ctx->table_count != before)
      ctx->link_count += table.link_count;
   return true;
}

static bool
preserves_scc(const struct agc_isa_inst *inst)
{
   enum agc_isa_family family = (enum agc_isa_family)inst->family;
   if (!agc_isa_family_is_scalar_alu(family) || family == AGC_ISA_SOPP)
      return true;
   if (family == AGC_ISA_SOP1)
      return inst->opcode == AGC_ISA_SOP1_MOV_B32 || inst->opcode == AGC_ISA_SOP1_MOV_B64;
   return family == AGC_ISA_SOPK && inst->opcode == AGC_ISA_SOPK_MOVK_I32;
}

static struct val
and_values(struct val a, struct val b)
{
   if (a.kind == VAL_CONST && b.kind == VAL_CONST)
      return val_const(a.value & b.value);
   if ((a.kind == VAL_MASKED && b.kind == VAL_CONST) || (a.kind == VAL_CONST && b.kind == VAL_MASKED))
      return val_masked(a.value & b.value);
   if (b.kind == VAL_CONST)
      return val_masked(b.value);
   if (a.kind == VAL_CONST)
      return val_masked(a.value);
   return val_of(VAL_UNKNOWN);
}

static struct val
shift_values(struct val a, struct val b, bool left)
{
   if (b.kind != VAL_CONST || (a.kind != VAL_CONST && a.kind != VAL_MASKED))
      return val_of(VAL_UNKNOWN);
   unsigned amount = b.value & 31;
   uint32_t shifted = left ? a.value << amount : a.value >> amount;
   return a.kind == VAL_CONST ? val_const(shifted) : val_masked(shifted);
}

static void
step(struct ctx *ctx, struct state *state, const struct agc_isa_inst *inst)
{
   bool sop1 = inst->fields == AGC_ISA_SOP1 && inst->family == AGC_ISA_SOP1;
   bool sop2 = inst->fields == AGC_ISA_SOP2 && inst->family == AGC_ISA_SOP2;
   if (sop1 && inst->opcode == AGC_ISA_SOP1_MOV_B32) {
      state_write(ctx, state, inst->sop1.sdst, state_read(ctx, state, inst->sop1.src0));
   } else if (sop1 && inst->opcode == AGC_ISA_SOP1_MOV_B64) {
      struct val lo = state_read(ctx, state, inst->sop1.src0);
      struct val hi = state_read_high(ctx, state, inst->sop1.src0);
      if (inst->sop1.sdst.kind == AGC_ISA_OPERAND_SGPR) {
         state_write_reg(ctx, state, inst->sop1.sdst.reg, lo);
         state_write_reg(ctx, state, inst->sop1.sdst.reg + 1, hi);
      }
   } else if (inst->fields == AGC_ISA_SOPK && inst->family == AGC_ISA_SOPK &&
              inst->opcode == AGC_ISA_SOPK_MOVK_I32) {
      state_write(ctx, state, inst->sopk.sdst, val_const((uint32_t)(int32_t)(int16_t)inst->sopk.simm16));
   } else if (sop2 && (inst->opcode == AGC_ISA_SOP2_ADD_U32 || inst->opcode == AGC_ISA_SOP2_SUB_U32)) {
      struct carry carry;
      struct val value = low_dword(state_read(ctx, state, inst->sop2.src0), state_read(ctx, state, inst->sop2.src1),
                                   inst->opcode == AGC_ISA_SOP2_SUB_U32, &carry);
      state_write(ctx, state, inst->sop2.sdst, value);
      state->scc = carry;
      return;
   } else if (sop2 && (inst->opcode == AGC_ISA_SOP2_ADDC_U32 || inst->opcode == AGC_ISA_SOP2_SUBB_U32)) {
      struct val value = high_dword(state_read(ctx, state, inst->sop2.src0), state_read(ctx, state, inst->sop2.src1),
                                    state->scc, inst->opcode == AGC_ISA_SOP2_SUBB_U32);
      state_write(ctx, state, inst->sop2.sdst, value);
      state->scc = carry_of(CARRY_UNKNOWN);
      return;
   } else if (sop2 && inst->opcode == AGC_ISA_SOP2_AND_B32) {
      struct val value = and_values(state_read(ctx, state, inst->sop2.src0), state_read(ctx, state, inst->sop2.src1));
      state_write(ctx, state, inst->sop2.sdst, value);
   } else if (sop2 && (inst->opcode == AGC_ISA_SOP2_LSHL_B32 || inst->opcode == AGC_ISA_SOP2_LSHR_B32)) {
      struct val value = shift_values(state_read(ctx, state, inst->sop2.src0),
                                      state_read(ctx, state, inst->sop2.src1), inst->opcode == AGC_ISA_SOP2_LSHL_B32);
      state_write(ctx, state, inst->sop2.sdst, value);
   } else if (inst->fields == AGC_ISA_SMEM && inst->family == AGC_ISA_SMEM && is_table_load(inst)) {
      const struct agc_isa_smem *s = &inst->smem;
      if (s->sdata.kind == AGC_ISA_OPERAND_SGPR) {
         unsigned first = s->sdata.reg;
         bool wanted = false;
         for (unsigned k = 0; k < s->dwords; k++)
            wanted |= tracks(ctx, first + k);
         uint32_t index = 0;
         bool found = wanted && table_load(ctx, state, s, &index);
         for (unsigned k = 0; k < s->dwords; k++)
            state_write_reg(ctx, state, first + k, found ? val_word(true, index, (uint8_t)k, 0) : val_of(VAL_UNKNOWN));
      }
   } else {
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_RELATIVE) &&
          agc_isa_family_is_scalar_alu((enum agc_isa_family)inst->family))
         state_fill(ctx, state, VAL_UNKNOWN);
      struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
      unsigned count = agc_isa_inst_defs(inst, ctx->wave64, ranges);
      for (unsigned r = 0; r < count; r++) {
         if (ranges[r].file != AGC_ISA_REG_SCALAR)
            continue;
         for (unsigned n = ranges[r].first; n < (unsigned)ranges[r].first + ranges[r].count; n++)
            state_write_reg(ctx, state, n, val_of(VAL_UNKNOWN));
      }
   }
   if (!preserves_scc(inst))
      state->scc = carry_of(CARRY_UNKNOWN);
}

static void
source_of(const struct ctx *ctx, const struct val *vals, unsigned count, struct use_record *record)
{
   record->use.source.kind = AGC_ISA_SOURCE_DYNAMIC;
   record->use.source.first = 0;
   record->use.source.tables = NULL;
   record->use.source.table_count = 0;
   record->table = 0;
   record->dword = 0;
   struct val low = vals[0];
   if (low.kind != VAL_WORD || low.add != 0)
      return;
   for (unsigned k = 1; k < count; k++) {
      if (!val_equal(vals[k], val_word(low.load, low.value, (uint8_t)(low.dword + k), 0)))
         return;
   }
   if (!low.load) {
      if (is_user(ctx, low.dword, count)) {
         record->use.source.kind = AGC_ISA_SOURCE_USER_SGPRS;
         record->use.source.first = low.dword;
      }
      return;
   }
   record->use.source.kind = ctx->tables[low.value].offset_mask ? AGC_ISA_SOURCE_TABLE_CHOICES : AGC_ISA_SOURCE_TABLE;
   record->table = low.value;
   record->dword = low.dword;
}

static struct agc_isa_scalar_offset
scalar_offset_of(const struct ctx *ctx, const struct state *state, struct agc_isa_operand operand)
{
   struct agc_isa_scalar_offset offset = {AGC_ISA_SCALAR_OFFSET_UNKNOWN, 0, 0};
   struct val value = state_read(ctx, state, operand);
   if (value.kind == VAL_CONST || value.kind == VAL_MASKED) {
      offset.kind = AGC_ISA_SCALAR_OFFSET_BOUNDED;
      offset.bound = value.value;
   } else if (value.kind == VAL_WORD && !value.load && value.add == 0 && is_user(ctx, value.dword, 1)) {
      offset.kind = AGC_ISA_SCALAR_OFFSET_USER_SGPR;
      offset.sgpr = value.dword;
   }
   return offset;
}

static struct use_record *
push_use(struct ctx *ctx, struct use_list *list)
{
   if (list->count == list->capacity) {
      size_t capacity = list->capacity ? list->capacity * 2 : 16;
      struct use_record *records = realloc(list->records, capacity * sizeof(*records));
      if (!records) {
         ctx->failed = true;
         return NULL;
      }
      list->records = records;
      list->capacity = capacity;
   }
   struct use_record *record = &list->records[list->count++];
   memset(record, 0, sizeof(*record));
   return record;
}

static void
add_use(struct ctx *ctx, const struct state *state, const struct agc_isa_inst *inst, struct use_list *uses,
        enum agc_isa_resource_kind kind, uint8_t base_sgpr, uint8_t dword_count,
        enum agc_isa_descriptor_access access, const struct agc_isa_buffer_access *buffer_access)
{
   struct use_record *record = push_use(ctx, uses);
   if (!record)
      return;
   struct val vals[8];
   for (unsigned k = 0; k < dword_count; k++)
      vals[k] = state_reg(ctx, state, (unsigned)base_sgpr + k);
   record->use.pc = inst->pc;
   record->use.kind = (uint8_t)kind;
   record->use.base_sgpr = base_sgpr;
   record->use.dword_count = dword_count;
   record->use.access = (uint8_t)access;
   source_of(ctx, vals, dword_count, record);
   if (kind == AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR && buffer_access) {
      record->use.has_buffer_access = true;
      record->use.buffer_access = *buffer_access;
   }
}

static inline enum agc_isa_descriptor_access
written(uint8_t kind)
{
   return kind == AGC_ISA_MEM_STORE || kind == AGC_ISA_MEM_ATOMIC ? AGC_ISA_ACCESS_WRITTEN : AGC_ISA_ACCESS_LOADED;
}

static void
record(struct ctx *ctx, const struct state *state, const struct agc_isa_inst *inst, struct use_list *uses)
{
   struct agc_isa_buffer_access access;
   memset(&access, 0, sizeof(access));
   switch (inst->fields) {
   case AGC_ISA_SMEM: {
      const struct agc_isa_smem *s = &inst->smem;
      if (!s->buffer || s->dwords == 0 || agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_SRC))
         break;
      access.scalar = true;
      access.immediate_offset = (uint32_t)s->offset;
      access.scalar_offset = scalar_offset_of(ctx, state, s->soffset);
      const struct agc_isa_mem_op *op = agc_isa_smem_op(inst->opcode);
      add_use(ctx, state, inst, uses, AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR, s->sbase, 4,
              written(op ? op->kind : AGC_ISA_MEM_STORE), &access);
      break;
   }
   case AGC_ISA_MUBUF:
   case AGC_ISA_MTBUF: {
      const struct agc_isa_buffer *b = &inst->buffer;
      if (b->kind == AGC_ISA_MEM_CONTROL)
         break;
      access.indexed = b->idxen;
      access.vector_offset = b->offen;
      access.immediate_offset = b->offset;
      access.scalar_offset = scalar_offset_of(ctx, state, b->soffset);
      add_use(ctx, state, inst, uses, AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR, b->srsrc, 4, written(b->kind), &access);
      break;
   }
   case AGC_ISA_MIMG: {
      const struct agc_isa_image *m = &inst->mimg;
      const struct agc_isa_image_op *op = agc_isa_image_op(inst->opcode);
      enum agc_isa_descriptor_access image_access = written(m->kind);
      if (op && (op->flags & AGC_ISA_IMG_RESINFO))
         image_access = AGC_ISA_ACCESS_QUERIED;
      else if (m->uses_sampler)
         image_access = AGC_ISA_ACCESS_SAMPLED;
      add_use(ctx, state, inst, uses, AGC_ISA_RESOURCE_IMAGE_DESCRIPTOR, m->srsrc, m->r128 ? 4 : 8, image_access,
              NULL);
      if (m->uses_sampler)
         add_use(ctx, state, inst, uses, AGC_ISA_RESOURCE_SAMPLER_DESCRIPTOR, m->ssamp, 4, AGC_ISA_ACCESS_SAMPLED,
                 NULL);
      break;
   }
   default:
      break;
   }
}

static void
record_pointer(struct ctx *ctx, const struct state *state, const struct agc_isa_inst *inst, struct use_list *uses)
{
   if (inst->fields != AGC_ISA_SMEM)
      return;
   const struct agc_isa_smem *s = &inst->smem;
   const struct agc_isa_mem_op *op = agc_isa_smem_op(inst->opcode);
   if (!op || s->buffer || s->dwords == 0 || op->kind == AGC_ISA_MEM_CONTROL)
      return;
   struct agc_isa_scalar_offset scalar_offset = scalar_offset_of(ctx, state, s->soffset);
   struct val vals[2] = {state_reg(ctx, state, s->sbase), state_reg(ctx, state, (unsigned)s->sbase + 1)};
   int32_t immediate_offset = s->offset;
   if (vals[0].kind == VAL_WORD) {
      struct val low = vals[0];
      int64_t sum = (int64_t)s->offset + low.add;
      if (val_equal(vals[1], val_word(low.load, low.value, (uint8_t)(low.dword + 1), low.add)) &&
          low.add >= INT32_MIN && low.add <= INT32_MAX && sum >= INT32_MIN && sum <= INT32_MAX) {
         immediate_offset = (int32_t)sum;
         vals[0] = val_word(low.load, low.value, low.dword, 0);
         vals[1] = val_word(low.load, low.value, (uint8_t)(low.dword + 1), 0);
      }
   }
   struct use_record *record = push_use(ctx, uses);
   if (!record)
      return;
   record->use.pc = inst->pc;
   record->use.kind = AGC_ISA_RESOURCE_SCALAR_MEMORY_PAIR;
   record->use.base_sgpr = s->sbase;
   record->use.dword_count = 2;
   record->use.access = op->kind == AGC_ISA_MEM_LOAD ? AGC_ISA_ACCESS_LOADED : AGC_ISA_ACCESS_WRITTEN;
   source_of(ctx, vals, 2, record);
   record->use.has_buffer_access = true;
   record->use.buffer_access.scalar = true;
   record->use.buffer_access.immediate_offset = (uint32_t)immediate_offset;
   record->use.buffer_access.scalar_offset = scalar_offset;
}

enum holds_kind {
   HOLDS_OTHER,
   HOLDS_LOCAL,
   HOLDS_INDEX,
   HOLDS_USER
};

struct holds {
   uint8_t kind;
   uint8_t sgpr;
};

static inline bool
holds_equal(struct holds a, struct holds b)
{
   return a.kind == b.kind && a.sgpr == b.sgpr;
}

static inline bool
user_contains(const struct ctx *ctx, unsigned n)
{
   return ctx->user_start <= n && n < ctx->user_end;
}

static bool
buffer_fill(const struct ctx *ctx, bool has_group_x, uint8_t group_x, struct agc_isa_buffer_fill *fill)
{
   const struct agc_isa_ir *ir = ctx->ir;
   if (!has_group_x || ir->block_count != 1)
      return false;
   struct holds vector[VECTORS];
   memset(vector, 0, sizeof(vector));
   vector[0].kind = HOLDS_LOCAL;
   bool found = false;
   for (size_t i = 0; i < ir->instruction_count; i++) {
      const struct agc_isa_inst *inst = &ir->instructions[i];
      struct holds holds = {HOLDS_OTHER, 0};
      switch (inst->fields) {
      case AGC_ISA_MUBUF: {
         const struct agc_isa_buffer *b = &inst->buffer;
         if (b->kind != AGC_ISA_MEM_STORE || found)
            return false;
         unsigned data = b->vdata;
         if (vector[data].kind != HOLDS_USER)
            return false;
         uint8_t first = vector[data].sgpr;
         bool plain = b->idxen && !b->offen && b->offset == 0 &&
                      agc_isa_operand_equal(b->soffset, agc_isa_operand_inline_int(0)) && b->data_dwords == 4;
         if (!plain || vector[b->vaddr].kind != HOLDS_INDEX || data + 4 > VECTORS)
            return false;
         for (unsigned k = 0; k < 4; k++) {
            struct holds wanted = {HOLDS_USER, (uint8_t)(first + k)};
            if (!holds_equal(vector[data + k], wanted))
               return false;
            if (!user_contains(ctx, (unsigned)b->srsrc + k) || !user_contains(ctx, (unsigned)first + k))
               return false;
         }
         fill->buffer = b->srsrc;
         fill->value = first;
         found = true;
         break;
      }
      case AGC_ISA_MTBUF:
      case AGC_ISA_SMEM:
      case AGC_ISA_FLAT:
      case AGC_ISA_DS:
      case AGC_ISA_MIMG:
      case AGC_ISA_EXP:
      case AGC_ISA_LDSDIR:
         return false;
      case AGC_ISA_VOP1:
         if (inst->vop1.src0.kind == AGC_ISA_OPERAND_SGPR && inst->opcode == AGC_ISA_VOP1_MOV_B32 &&
             user_contains(ctx, inst->vop1.src0.reg)) {
            holds.kind = HOLDS_USER;
            holds.sgpr = (uint8_t)inst->vop1.src0.reg;
         }
         break;
      case AGC_ISA_VOP3: {
         const struct agc_isa_vop3 *v = &inst->vop3;
         if (inst->opcode == AGC_ISA_VOP3_LSHL_ADD_U32 &&
             agc_isa_operand_equal(v->src[0], agc_isa_operand_sgpr(group_x)) &&
             agc_isa_operand_equal(v->src[1], agc_isa_operand_inline_int(FILL_GROUP_SHIFT)) &&
             v->src[2].kind == AGC_ISA_OPERAND_VGPR && vector[v->src[2].reg & 0xff].kind == HOLDS_LOCAL)
            holds.kind = HOLDS_INDEX;
         break;
      }
      default:
         break;
      }
      struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
      unsigned count = agc_isa_inst_defs(inst, ctx->wave64, ranges);
      for (unsigned r = 0; r < count; r++) {
         for (unsigned n = ranges[r].first; n < (unsigned)ranges[r].first + ranges[r].count; n++) {
            if (ranges[r].file == AGC_ISA_REG_VECTOR) {
               if (n < VECTORS)
                  vector[n] = holds;
            } else if (user_contains(ctx, n) || n == group_x) {
               return false;
            }
         }
      }
   }
   return found;
}

static bool
uses_descriptors(const struct agc_isa_inst *inst)
{
   uint8_t first[2];
   uint8_t count[2];
   if (descriptor_operands(inst, first, count))
      return true;
   return inst->fields == AGC_ISA_SMEM && !inst->smem.buffer && inst->smem.dwords > 0;
}

struct successor_lists {
   size_t *returns;
   size_t return_count;
   size_t *regions;
   size_t region_count;
};

static void
join_into(const struct ctx *ctx, struct state *entry, size_t next, const struct state *state,
          struct agc_isa_work *work)
{
   if (state_join(ctx, &entry[next], state))
      agc_isa_work_insert(work, next);
}

static void
solve(struct ctx *ctx, struct state *entry, struct state *opened, struct state *state,
      const struct successor_lists *lists, struct agc_isa_work *work)
{
   const struct agc_isa_ir *ir = ctx->ir;
   agc_isa_work_insert(work, 0);
   size_t id;
   while (agc_isa_work_pop_first(work, &id)) {
      const struct agc_isa_block *block = &ir->blocks[id];
      state_copy(ctx, state, &entry[id]);
      for (size_t i = block->first; i < block->end; i++)
         step(ctx, state, &ir->instructions[i]);
      for (uint32_t s = 0; s < block->successor_count; s++)
         join_into(ctx, entry, block->successors[s], state, work);
      const struct agc_isa_jump_target *jump =
         agc_isa_computed_flow_jump(&ir->flow, ir->instructions[block->end - 1].pc);
      if (jump && jump->kind == AGC_ISA_JUMP_RETURN) {
         for (size_t k = 0; k < lists->return_count; k++)
            join_into(ctx, entry, lists->returns[k], state, work);
      } else if (jump && jump->kind == AGC_ISA_JUMP_EXTERNAL) {
         for (size_t k = 0; k < lists->region_count; k++)
            join_into(ctx, entry, lists->regions[k], state, work);
      }
      if (jump && jump->kind == AGC_ISA_JUMP_OPEN && state_join(ctx, opened, state)) {
         for (size_t next = 0; next < ir->block_count; next++)
            join_into(ctx, entry, next, opened, work);
      }
   }
}

static size_t
choice_count(const struct ctx *ctx, const struct use_record *record)
{
   if (record->use.source.kind == AGC_ISA_SOURCE_TABLE)
      return 1;
   if (record->use.source.kind == AGC_ISA_SOURCE_TABLE_CHOICES)
      return (size_t)1 << count_ones(ctx->tables[record->table].offset_mask);
   return 0;
}

static int
compare_u32(const void *a, const void *b)
{
   uint32_t x = *(const uint32_t *)a;
   uint32_t y = *(const uint32_t *)b;
   return x < y ? -1 : x > y;
}

static struct agc_isa_descriptor_use *
materialize(const struct ctx *ctx, const struct use_list *list, struct agc_isa_descriptor_table **next_table,
            const struct agc_isa_table_link *links)
{
   struct agc_isa_descriptor_use *uses = malloc((list->count + 1) * sizeof(*uses));
   if (!uses)
      return NULL;
   for (size_t i = 0; i < list->count; i++) {
      const struct use_record *record = &list->records[i];
      size_t at = i;
      while (at > 0 && uses[at - 1].pc > record->use.pc) {
         uses[at] = uses[at - 1];
         at--;
      }
      uses[at] = record->use;
      size_t choices = choice_count(ctx, record);
      if (!choices)
         continue;
      const struct table *table = &ctx->tables[record->table];
      uint64_t fixed = (uint64_t)table->offset + 4 * (uint64_t)record->dword;
      uint32_t offsets[AGC_ISA_MAX_TABLE_CHOICES] = {0};
      size_t offset_count = 1;
      for (unsigned bit = 0; bit < 32 && choices > 1; bit++) {
         if (!(table->offset_mask & (UINT32_C(1) << bit)))
            continue;
         for (size_t k = 0; k < offset_count; k++)
            offsets[offset_count + k] = offsets[k] | (UINT32_C(1) << bit);
         offset_count *= 2;
      }
      qsort(offsets, offset_count, sizeof(*offsets), compare_u32);
      struct agc_isa_descriptor_table *tables = *next_table;
      for (size_t k = 0; k < choices; k++) {
         tables[k].root = table->root;
         tables[k].root_form = table->root_form;
         tables[k].links = table->link_count ? links + table->link_first : NULL;
         tables[k].link_count = table->link_count;
         tables[k].offset = (int64_t)(fixed + offsets[k]);
      }
      uses[at].source.tables = tables;
      uses[at].source.table_count = (uint32_t)choices;
      *next_table += choices;
   }
   return uses;
}

static bool
pack(const struct ctx *ctx, const struct use_list *descriptors, const struct use_list *pointers,
     struct agc_isa_uses *uses)
{
   size_t table_total = 0;
   for (size_t i = 0; i < descriptors->count; i++)
      table_total += choice_count(ctx, &descriptors->records[i]);
   for (size_t i = 0; i < pointers->count; i++)
      table_total += choice_count(ctx, &pointers->records[i]);
   struct agc_isa_descriptor_table *tables = NULL;
   struct agc_isa_table_link *links = NULL;
   if (table_total) {
      size_t table_bytes = table_total * sizeof(*tables);
      uses->storage = calloc(1, table_bytes + (ctx->link_count + 1) * sizeof(*links));
      if (!uses->storage)
         return false;
      tables = uses->storage;
      links = (struct agc_isa_table_link *)((char *)uses->storage + table_bytes);
      if (ctx->link_count)
         memcpy(links, ctx->links, ctx->link_count * sizeof(*links));
   }
   uses->descriptors = materialize(ctx, descriptors, &tables, links);
   uses->pointers = materialize(ctx, pointers, &tables, links);
   if (!uses->descriptors || !uses->pointers)
      return false;
   uses->descriptor_count = descriptors->count;
   uses->pointer_count = pointers->count;
   return true;
}

bool
agc_isa_sources_descriptor_uses(const struct agc_isa_ir *ir, bool wave64, uint16_t user_start, uint16_t user_end,
                                bool has_group_x, uint8_t group_x, struct agc_isa_uses *uses)
{
   memset(uses, 0, sizeof(*uses));
   bool used = false;
   for (size_t i = 0; i < ir->instruction_count && !used; i++)
      used = uses_descriptors(&ir->instructions[i]);
   if (!ir->block_count || !used)
      return true;

   struct ctx ctx = {.ir = ir, .wave64 = wave64, .user_start = user_start, .user_end = user_end};
   slots_init(&ctx);
   uses->has_fill = buffer_fill(&ctx, has_group_x, group_x, &uses->fill);

   size_t count = ir->block_count;
   size_t calls = 0;
   for (size_t i = 0; i < ir->instruction_count; i++)
      calls += agc_isa_inst_has_flag(&ir->instructions[i], AGC_ISA_FLAG_CALL);
   struct use_list descriptors = {0};
   struct use_list pointers = {0};
   struct agc_isa_work work = {0};
   struct state *entry = malloc((count + 2) * sizeof(*entry));
   struct val *regs = malloc((count + 2) * (size_t)(ctx.slot_count + 1) * sizeof(*regs));
   size_t *block_lists = malloc((calls + ir->layout.region_count + 1) * sizeof(*block_lists));
   bool ok = entry && regs && block_lists && agc_isa_work_init(&work, count);
   if (ok) {
      struct successor_lists lists = {.returns = block_lists};
      for (size_t i = 0; i < ir->instruction_count; i++) {
         const struct agc_isa_inst *inst = &ir->instructions[i];
         if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CALL) &&
             agc_isa_block_at(ir, agc_isa_inst_end_pc(inst), &lists.returns[lists.return_count]))
            lists.return_count++;
      }
      lists.regions = block_lists + lists.return_count;
      for (size_t i = 0; i < ir->layout.region_count; i++) {
         if (agc_isa_block_at(ir, ir->layout.regions[i].start, &lists.regions[lists.region_count]))
            lists.region_count++;
      }

      for (size_t id = 0; id < count + 2; id++) {
         entry[id].regs = regs + id * (size_t)(ctx.slot_count + 1);
         state_unset(&ctx, &entry[id]);
      }
      struct state *opened = &entry[count];
      struct state *state = &entry[count + 1];
      state_entry(&ctx, &entry[0]);
      solve(&ctx, entry, opened, state, &lists, &work);

      for (size_t id = 0; id < count; id++) {
         const struct agc_isa_block *block = &ir->blocks[id];
         if (!block->reachable)
            continue;
         if (entry[id].reached)
            state_copy(&ctx, state, &entry[id]);
         else
            state_unknown(&ctx, state);
         for (size_t i = block->first; i < block->end; i++) {
            const struct agc_isa_inst *inst = &ir->instructions[i];
            record(&ctx, state, inst, &descriptors);
            record_pointer(&ctx, state, inst, &pointers);
            step(&ctx, state, inst);
         }
      }
      ok = !ctx.failed && pack(&ctx, &descriptors, &pointers, uses);
   }
   free(entry);
   free(regs);
   free(block_lists);
   free(work.bits);
   free(descriptors.records);
   free(pointers.records);
   free(ctx.tables);
   free(ctx.links);
   if (!ok) {
      free(uses->descriptors);
      free(uses->pointers);
      free(uses->storage);
      memset(uses, 0, sizeof(*uses));
   }
   return ok;
}
