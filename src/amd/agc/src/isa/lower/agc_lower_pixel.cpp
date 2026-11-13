#include "agc_lower.h"

#define SPI_SHADER_ZERO 0u

struct facts {
   bool m0;
   struct agc_isa_mask128 sgprs;
};

static bool
fail_memory(struct agc_isa_error *error)
{
   memset(error, 0, sizeof(*error));
   error->kind = AGC_ISA_ERROR_OUT_OF_MEMORY;
   return false;
}

bool
agc_isa_pixel_is_vop3_interp(uint16_t opcode)
{
   return (opcode >= AGC_ISA_VOP3_INTERP_P1_F32 && opcode <= AGC_ISA_VOP3_INTERP_MOV_F32) ||
          opcode == AGC_ISA_VOP3_INTERP_P1LL_F16 || opcode == AGC_ISA_VOP3_INTERP_P1LV_F16 ||
          opcode == AGC_ISA_VOP3_INTERP_P2_F16;
}

bool
agc_isa_lowering_interpolated(const struct agc_isa_inst *inst, uint8_t *attr, uint8_t *chan)
{
   if (inst->family == AGC_ISA_VINTRP && inst->fields == AGC_ISA_VINTRP) {
      *attr = inst->vintrp.attr;
      *chan = inst->vintrp.chan;
      return true;
   }
   if (inst->family == AGC_ISA_VOP3 && inst->fields == AGC_ISA_VOP3 && agc_isa_pixel_is_vop3_interp(inst->opcode) &&
       inst->vop3.src[0].kind == AGC_ISA_OPERAND_RAW) {
      uint16_t select = (uint16_t)inst->vop3.src[0].raw;
      *attr = (uint8_t)(select & 0x3f);
      *chan = (uint8_t)((select >> 6) & 3);
      return true;
   }
   return false;
}

static uint8_t
null_target(bool has_col_format, uint32_t col_format, bool has_z_format, uint32_t z_format)
{
   bool mrt0_zero = has_col_format && (col_format & 0xf) == SPI_SHADER_ZERO;
   bool has_depth = has_z_format && z_format != SPI_SHADER_ZERO;
   return mrt0_zero && has_depth ? AGC_ISA_EXP_MRTZ : AGC_ISA_EXP_MRT0;
}

bool
agc_isa_pixel_col_format_for(bool has_col_format, uint32_t col_format, bool has_z_format, uint32_t z_format,
                             bool null_exports, uint32_t *format)
{
   *format = 0;
   if (!has_col_format)
      return false;
   bool needs_memory = null_exports &&
                       null_target(has_col_format, col_format, has_z_format, z_format) == AGC_ISA_EXP_MRT0 &&
                       (col_format & 0xf) == SPI_SHADER_ZERO;
   if (!needs_memory)
      return false;
   *format = col_format | AGC_ISA_SPI_SHADER_32_R;
   return true;
}

static struct facts
facts_nothing(void)
{
   struct facts facts = {false, {0, 0}};
   return facts;
}

static struct facts
facts_unvisited(void)
{
   struct facts facts = {true, {UINT64_MAX, UINT64_MAX}};
   return facts;
}

static struct facts
facts_meet(struct facts a, struct facts b)
{
   struct facts facts = {a.m0 && b.m0, agc_isa_mask128_and(a.sgprs, b.sgprs)};
   return facts;
}

static bool
facts_equal(struct facts a, struct facts b)
{
   return a.m0 == b.m0 && agc_isa_mask128_equal(a.sgprs, b.sgprs);
}

static bool
facts_hold(struct facts facts, struct agc_isa_operand operand)
{
   switch (operand.kind) {
   case AGC_ISA_OPERAND_M0:
      return facts.m0;
   case AGC_ISA_OPERAND_SGPR:
      return agc_isa_mask128_test(facts.sgprs, operand.reg);
   default:
      return false;
   }
}

static void
facts_step(struct facts *facts, const struct agc_isa_inst *inst, bool wave64)
{
   if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CALL | AGC_ISA_FLAG_RELATIVE)) {
      *facts = facts_nothing();
      return;
   }
   bool copied = false;
   struct agc_isa_operand copy = agc_isa_operand_none();
   if (inst->fields == AGC_ISA_SOP1 && inst->family == AGC_ISA_SOP1 && inst->opcode == AGC_ISA_SOP1_MOV_B32 &&
       facts_hold(*facts, inst->sop1.src0)) {
      copied = true;
      copy = inst->sop1.sdst;
   }
   struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
   unsigned count = agc_isa_inst_defs(inst, wave64, ranges);
   for (unsigned i = 0; i < count; i++) {
      if (ranges[i].file != AGC_ISA_REG_SCALAR)
         continue;
      unsigned end = (unsigned)ranges[i].first + (ranges[i].count ? ranges[i].count : 1);
      for (unsigned reg = ranges[i].first; reg < end; reg++) {
         if (reg == AGC_ISA_SCALAR_M0)
            facts->m0 = false;
         else if (reg <= 127)
            facts->sgprs = agc_isa_mask128_andnot(facts->sgprs, agc_isa_mask128_bit(reg));
      }
   }
   if (!copied)
      return;
   if (copy.kind == AGC_ISA_OPERAND_M0)
      facts->m0 = true;
   else if (copy.kind == AGC_ISA_OPERAND_SGPR)
      facts->sgprs = agc_isa_mask128_or(facts->sgprs, agc_isa_mask128_bit(copy.reg));
}

static struct facts
facts_exit(const struct agc_isa_ir *ir, struct facts facts, uint32_t id, bool wave64)
{
   const struct agc_isa_block *block = &ir->blocks[id];
   for (uint32_t i = block->first; i < block->end; i++)
      facts_step(&facts, &ir->instructions[i], wave64);
   return facts;
}

struct work_list {
   uint32_t *items;
   size_t count;
   size_t capacity;
};

static bool
work_push(struct work_list *work, uint32_t id)
{
   if (work->count == work->capacity) {
      size_t capacity = work->capacity ? work->capacity * 2 : 16;
      uint32_t *items = (uint32_t *)realloc(work->items, capacity * sizeof(*items));
      if (!items)
         return false;
      work->items = items;
      work->capacity = capacity;
   }
   work->items[work->count++] = id;
   return true;
}

static bool
m0_holds_prim_mask(const struct agc_isa_ir *ir, bool wave64, uint8_t prim_mask, struct agc_isa_pc_set *found)
{
   size_t count = ir->block_count;
   struct facts *entry = (struct facts *)malloc((count ? count : 1) * sizeof(*entry));
   bool *seen = (bool *)calloc(count ? count : 1, sizeof(*seen));
   struct work_list work = {NULL, 0, 0};
   bool ok = entry && seen;
   for (size_t id = 0; ok && id < count; id++)
      entry[id] = facts_unvisited();
   for (size_t b = 0; ok && b < count; b++) {
      const struct agc_isa_block *block = &ir->blocks[b];
      if (!block->reachable)
         continue;
      if (block->id == 0) {
         struct facts start = {false, agc_isa_mask128_bit(prim_mask)};
         entry[0] = start;
         seen[0] = true;
      }
      if (agc_isa_computed_flow_has_entry(&ir->flow, block->start_pc)) {
         entry[block->id] = facts_nothing();
         seen[block->id] = true;
      }
   }
   for (size_t id = 0; ok && id < count; id++) {
      if (seen[id])
         ok = work_push(&work, (uint32_t)id);
   }
   while (ok && work.count) {
      uint32_t id = work.items[--work.count];
      struct facts out = facts_exit(ir, entry[id], id, wave64);
      const struct agc_isa_block *block = &ir->blocks[id];
      for (uint32_t s = 0; ok && s < block->successor_count; s++) {
         uint32_t next = block->successors[s];
         struct facts merged = seen[next] ? facts_meet(entry[next], out) : out;
         if (!seen[next] || !facts_equal(merged, entry[next])) {
            entry[next] = merged;
            seen[next] = true;
            ok = work_push(&work, next);
         }
      }
   }
   for (size_t b = 0; ok && b < count; b++) {
      const struct agc_isa_block *block = &ir->blocks[b];
      if (!block->reachable || !seen[block->id])
         continue;
      struct facts facts = entry[block->id];
      for (uint32_t i = block->first; ok && i < block->end; i++) {
         const struct agc_isa_inst *inst = &ir->instructions[i];
         uint8_t attr;
         uint8_t chan;
         if (facts.m0 && agc_isa_lowering_interpolated(inst, &attr, &chan))
            ok = agc_isa_pc_set_insert(found, inst->pc);
         facts_step(&facts, inst, wave64);
      }
   }
   free(entry);
   free(seen);
   free(work.items);
   return ok;
}

static int
compare_channel(const void *a, const void *b)
{
   uint16_t x = *(const uint16_t *)a;
   uint16_t y = *(const uint16_t *)b;
   return x < y ? -1 : x > y;
}

static bool
hoist_channels(struct agc_isa_pixel_plan *plan, const struct agc_isa_ir *ir, struct agc_isa_regalloc *regs,
               struct agc_isa_pc_set *at_entry, struct agc_isa_error *error)
{
   size_t capacity = ir->instruction_count ? ir->instruction_count : 1;
   uint16_t *channels = (uint16_t *)malloc(capacity * sizeof(*channels));
   if (!channels)
      return fail_memory(error);
   size_t count = 0;
   for (size_t i = 0; i < ir->instruction_count; i++) {
      const struct agc_isa_inst *inst = &ir->instructions[i];
      uint8_t attr;
      uint8_t chan;
      if (agc_isa_pc_set_contains(at_entry, inst->pc) && agc_isa_lowering_interpolated(inst, &attr, &chan))
         channels[count++] = (uint16_t)((attr << 8) | chan);
   }
   qsort(channels, count, sizeof(*channels), compare_channel);
   size_t unique = 0;
   for (size_t i = 0; i < count; i++) {
      if (unique == 0 || channels[unique - 1] != channels[i])
         channels[unique++] = channels[i];
   }
   bool ok = true;
   if (unique != 0 && unique <= AGC_ISA_HOISTED_CHANNELS) {
      uint8_t base;
      ok = agc_isa_regalloc_reserve_program_vgprs(regs, (uint8_t)unique, &base, error);
      if (ok) {
         for (size_t k = 0; k < unique; k++) {
            plan->hoisted[k].attr = (uint8_t)(channels[k] >> 8);
            plan->hoisted[k].chan = (uint8_t)(channels[k] & 0xff);
            plan->hoisted[k].vgpr = (uint8_t)(base + k);
         }
         plan->hoisted_count = (unsigned)unique;
         plan->from_prologue = *at_entry;
         memset(at_entry, 0, sizeof(*at_entry));
      }
   }
   free(channels);
   return ok;
}

bool
agc_isa_pixel_plan_init(struct agc_isa_pixel_plan *plan, const struct agc_isa_ir *ir, struct agc_isa_regalloc *regs,
                        bool wave64, const struct agc_isa_pixel_interface *iface, struct agc_isa_error *error)
{
   memset(plan, 0, sizeof(*plan));
   plan->has_prim_mask = iface->has_prim_mask;
   plan->prim_mask = iface->has_prim_mask ? iface->prim_mask : 0;
   plan->null_target = null_target(iface->has_col_format, iface->col_format, iface->has_z_format, iface->z_format);
   if (iface->has_prim_mask) {
      struct agc_isa_pc_set at_entry;
      memset(&at_entry, 0, sizeof(at_entry));
      bool ok = m0_holds_prim_mask(ir, wave64, iface->prim_mask, &at_entry);
      if (!ok)
         fail_memory(error);
      else
         ok = hoist_channels(plan, ir, regs, &at_entry, error);
      agc_isa_pc_set_finish(&at_entry);
      if (!ok)
         return false;
   }
   if (iface->dual_source) {
      uint8_t base;
      if (!agc_isa_regalloc_reserve_program_vgprs(regs, 4, &base, error))
         return false;
      plan->has_dual_source = true;
      plan->dual_source = base;
   }
   return true;
}

void
agc_isa_pixel_plan_finish(struct agc_isa_pixel_plan *plan)
{
   agc_isa_pc_set_finish(&plan->from_prologue);
}

bool
agc_isa_pixel_plan_hoisted(const struct agc_isa_pixel_plan *plan, uint32_t pc, uint8_t attr, uint8_t chan,
                           uint8_t *vgpr)
{
   if (!agc_isa_pc_set_contains(&plan->from_prologue, pc))
      return false;
   for (unsigned i = 0; i < plan->hoisted_count; i++) {
      if (plan->hoisted[i].attr == attr && plan->hoisted[i].chan == chan) {
         *vgpr = plan->hoisted[i].vgpr;
         return true;
      }
   }
   return false;
}

#define WAVE_ID 1u

static bool
reads_exiting(struct agc_isa_operand op)
{
   return op.kind == AGC_ISA_OPERAND_POPS_EXITING_WAVE_ID;
}

static bool
writes_packer(uint16_t opcode)
{
   return opcode == AGC_ISA_SOPK_SETREG_B32 || opcode == AGC_ISA_SOPK_SETREG_IMM32_B32;
}

static bool
orders(const struct agc_isa_inst *inst)
{
   switch (inst->fields) {
   case AGC_ISA_SOPK:
      return writes_packer(inst->opcode) && (inst->sopk.simm16 & AGC_ISA_HWREG_ID_MASK) == AGC_ISA_HWREG_POPS_PACKER;
   case AGC_ISA_SOPP:
      return (inst->opcode == AGC_ISA_SOPP_SENDMSG || inst->opcode == AGC_ISA_SOPP_SENDMSGHALT) &&
             (inst->sopp.simm16 & 0xf) == AGC_ISA_MSG_ORDERED_PS_DONE;
   case AGC_ISA_SOP1:
      return inst->family == AGC_ISA_SOP1 && reads_exiting(inst->sop1.src0);
   case AGC_ISA_SOP2:
      return reads_exiting(inst->sop2.src0) || reads_exiting(inst->sop2.src1);
   case AGC_ISA_SOPC:
      return reads_exiting(inst->sopc.src0) || reads_exiting(inst->sopc.src1);
   default:
      return false;
   }
}

struct ordering {
   uint32_t block;
   uint32_t pc;
};

static void
reachable_after(const struct agc_isa_ir *ir, const struct agc_isa_block *block, bool *after, uint32_t *stack)
{
   memset(after, 0, ir->block_count * sizeof(*after));
   size_t depth = 0;
   for (uint32_t s = 0; s < block->successor_count; s++)
      stack[depth++] = block->successors[s];
   while (depth) {
      uint32_t id = stack[--depth];
      if (after[id])
         continue;
      after[id] = true;
      const struct agc_isa_block *next = &ir->blocks[id];
      for (uint32_t s = 0; s < next->successor_count; s++)
         stack[depth++] = next->successors[s];
   }
}

bool
agc_isa_lowering_pops_check(const struct agc_isa_ir *ir, enum agc_isa_target target, enum agc_shader_stage stage,
                            struct agc_isa_error *error)
{
   if (stage != AGC_SHADER_STAGE_FRAGMENT || !agc_isa_target_is_gfx11_plus(target))
      return true;
   size_t count = 0;
   for (size_t b = 0; b < ir->block_count; b++) {
      const struct agc_isa_block *block = &ir->blocks[b];
      if (!block->reachable)
         continue;
      for (uint32_t i = block->first; i < block->end; i++)
         count += orders(&ir->instructions[i]) ? 1 : 0;
   }
   if (count == 0)
      return true;
   struct ordering *ordering = (struct ordering *)malloc(count * sizeof(*ordering));
   bool *after = (bool *)malloc((ir->block_count ? ir->block_count : 1) * sizeof(*after));
   uint32_t *stack = (uint32_t *)malloc((2 * ir->block_count + 2) * sizeof(*stack));
   if (!ordering || !after || !stack) {
      free(ordering);
      free(after);
      free(stack);
      return fail_memory(error);
   }
   count = 0;
   for (size_t b = 0; b < ir->block_count; b++) {
      const struct agc_isa_block *block = &ir->blocks[b];
      if (!block->reachable)
         continue;
      for (uint32_t i = block->first; i < block->end; i++) {
         if (orders(&ir->instructions[i])) {
            ordering[count].block = block->id;
            ordering[count].pc = ir->instructions[i].pc;
            count++;
         }
      }
   }
   bool ok = true;
   for (size_t b = 0; ok && b < ir->block_count; b++) {
      const struct agc_isa_block *block = &ir->blocks[b];
      if (!block->reachable)
         continue;
      for (uint32_t i = block->first; ok && i < block->end; i++) {
         const struct agc_isa_inst *done_export = &ir->instructions[i];
         if (done_export->fields != AGC_ISA_EXP || !done_export->exp.done)
            continue;
         reachable_after(ir, block, after, stack);
         bool late = false;
         for (size_t k = 0; k < count && !late; k++)
            late = after[ordering[k].block] || (ordering[k].block == block->id && ordering[k].pc > done_export->pc);
         if (late)
            ok = agc_isa_lowering_error_instruction(
               error, done_export, target,
               "the done export can run before the ordered section (POPS) ends, which on the target would end it");
      }
   }
   free(ordering);
   free(after);
   free(stack);
   return ok;
}

bool
agc_isa_lowering_pops_without_exiting_wave(const struct agc_isa_inst *inst, struct agc_isa_inst *rewritten)
{
   *rewritten = *inst;
   struct agc_isa_operand *operands[3];
   unsigned count = agc_isa_lowering_source_operands(rewritten, operands);
   bool found = false;
   for (unsigned i = 0; i < count; i++) {
      if (operands[i]->kind == AGC_ISA_OPERAND_POPS_EXITING_WAVE_ID) {
         *operands[i] = agc_isa_operand_constant(WAVE_ID);
         found = true;
      }
   }
   return found;
}

namespace agc {

void
param_load(lower_ctx &ctx, uint8_t vdst, uint8_t attr, uint8_t chan)
{
   ctx.bld.ldsdir(aco_opcode::lds_param_load, vdef(vdst), Operand(m0, s1), attr, chan,
                  memory_sync_info(storage_shared), 0, 1);
}

void
enable_all_lanes(lower_ctx &ctx, uint8_t saved)
{
   Builder &bld = ctx.bld;
   bld.sop1(Builder::s_mov, sdef(saved, bld.lm.size()), mask_op(ctx, exec));
   bld.sop1(Builder::s_mov, mask_def(ctx, exec), imm(0xffffffffu));
}

void
restore_lanes(lower_ctx &ctx, uint8_t saved)
{
   ctx.bld.sop1(Builder::s_mov, mask_def(ctx, exec), sop(saved, ctx.bld.lm.size()));
}

void
load_hoisted_parameters(lower_ctx &ctx, const agc_isa_inst *origin)
{
   const agc_isa_pixel_plan *plan = ctx.pixel;
   if (!plan || !plan->has_prim_mask || plan->hoisted_count == 0)
      return;
   ctx.bld.sop1(aco_opcode::s_mov_b32, Definition(m0, s1), sop(plan->prim_mask));
   uint8_t saved = transient_sgprs(ctx, origin, 2);
   enable_all_lanes(ctx, saved);
   for (unsigned i = 0; i < plan->hoisted_count; i++) {
      const agc_isa_hoisted_channel *channel = &plan->hoisted[i];
      param_load(ctx, channel->vgpr, channel->attr, channel->chan);
   }
   restore_lanes(ctx, saved);
   wait_for_parameters(ctx);
}


static const uint32_t collision = (1u << 31) | 1u;
static const uint16_t gfx11_wait_export_ready = 0;
static const uint16_t gfx12_wait_export_ready = 2;

void
pops_inputs(lower_ctx &ctx, agc_isa_mask128 live_in_scalar)
{
   if (ctx.stage != AGC_SHADER_STAGE_FRAGMENT || !ctx.gfx11_plus)
      return;
   uint8_t sgprs[2];
   bool present[2];
   present[0] = agc_isa_regalloc_collision_wave_sgpr(ctx.regs, &sgprs[0]);
   present[1] = agc_isa_regalloc_intrawave_collision_sgpr(ctx.regs, &sgprs[1]);
   static const uint32_t values[2] = {collision, 0};
   for (unsigned k = 0; k < 2; k++) {
      if (!present[k] || !agc_isa_mask128_test(live_in_scalar, sgprs[k]))
         continue;
      ctx.bld.sop1(aco_opcode::s_mov_b32, sdef(sgprs[k]), imm(values[k]));
   }
}

void
pops_packer(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->opcode != AGC_ISA_SOPK_SETREG_B32 && inst->opcode != AGC_ISA_SOPK_SETREG_IMM32_B32)
      fail(ctx, inst, "the POPS packer register cannot be read on the target");
   ctx.bld.sopp(aco_opcode::s_wait_event, ctx.gfx12 ? gfx12_wait_export_ready : gfx11_wait_export_ready);
}
}
