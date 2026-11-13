#include "agc_isa_analyze_priv.h"

#include <assert.h>

void
agc_isa_ir_free_analysis(struct agc_isa_ir *ir)
{
   free(ir->blocks);
   ir->blocks = NULL;
   ir->block_count = 0;
   agc_isa_metadata_finish(&ir->metadata);
   agc_isa_computed_flow_finish(&ir->flow);
   agc_isa_code_layout_finish(&ir->layout);
}

static size_t
block_of(const struct agc_isa_block *blocks, size_t block_count, size_t index)
{
   size_t lo = 0;
   size_t hi = block_count;
   while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      if (blocks[mid].first < index)
         lo = mid + 1;
      else
         hi = mid;
   }
   assert(lo < block_count && blocks[lo].first == index);
   return lo;
}

static void
tail_jump_sgpr(struct agc_isa_ir *ir)
{
   ir->metadata.has_user_data_tail_sgpr = false;
   ir->metadata.user_data_tail_sgpr = 0;
   size_t last_block = ir->block_count;
   while (last_block > 0 && !ir->blocks[last_block - 1].reachable)
      last_block--;
   if (last_block == 0)
      return;
   const struct agc_isa_inst *jump = &ir->instructions[ir->blocks[last_block - 1].end - 1];
   if (jump->family != AGC_ISA_SOP1 || !agc_isa_inst_has_flag(jump, AGC_ISA_FLAG_JUMP))
      return;
   if (jump->fields != AGC_ISA_SOP1 || jump->sop1.src0.kind != AGC_ISA_OPERAND_SGPR)
      return;
   uint16_t base = (uint16_t)jump->sop1.src0.reg;
   for (size_t id = 0; id < ir->block_count; id++) {
      const struct agc_isa_block *block = &ir->blocks[id];
      if (!block->reachable)
         continue;
      for (size_t i = block->first; i < block->end; i++) {
         struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
         unsigned count = agc_isa_inst_defs(&ir->instructions[i], true, ranges);
         for (unsigned r = 0; r < count; r++) {
            if (agc_isa_reg_range_contains(ranges[r], AGC_ISA_REG_SCALAR, base) ||
                agc_isa_reg_range_contains(ranges[r], AGC_ISA_REG_SCALAR, (uint16_t)(base + 1)))
               return;
         }
      }
   }
   ir->metadata.has_user_data_tail_sgpr = true;
   ir->metadata.user_data_tail_sgpr = (uint8_t)base;
}

bool
agc_isa_cfg_build(struct agc_isa_ir *ir, struct agc_isa_error *error)
{
   size_t count = ir->instruction_count;
   if (count == 0)
      return agc_isa_fail(error, AGC_ISA_ERROR_EMPTY_PROGRAM, 0);
   bool *leader = calloc(count, sizeof(*leader));
   size_t *entries = malloc((ir->flow.entry_count + 1) * sizeof(*entries));
   if (!leader || !entries) {
      free(leader);
      free(entries);
      return agc_isa_fail_memory(error);
   }
   leader[0] = true;
   size_t entry_count = 0;
   for (size_t i = 0; i < ir->flow.entry_count; i++) {
      size_t index;
      if (agc_isa_ir_instruction_at(ir, ir->flow.entries[i], &index)) {
         entries[entry_count++] = index;
         leader[index] = true;
      }
   }
   for (size_t i = 0; i < count; i++) {
      const struct agc_isa_inst *inst = &ir->instructions[i];
      if (inst->has_branch_target) {
         size_t index;
         if (!agc_isa_ir_instruction_at(ir, inst->branch_target, &index)) {
            free(leader);
            free(entries);
            return agc_isa_fail_branch(error, inst->pc, inst->branch_target);
         }
         leader[index] = true;
      }
      if ((agc_isa_inst_ends_block(inst) || agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CALL)) && i + 1 < count)
         leader[i + 1] = true;
   }

   size_t block_count = 0;
   for (size_t i = 0; i < count; i++)
      block_count += leader[i];
   size_t edge_capacity = block_count * AGC_ISA_MAX_BLOCK_SUCCESSORS;
   struct agc_isa_block *blocks = calloc(1, block_count * sizeof(*blocks) + edge_capacity * sizeof(uint32_t));
   uint32_t *stack = malloc((1 + entry_count + edge_capacity) * sizeof(*stack));
   if (!blocks || !stack) {
      free(leader);
      free(entries);
      free(blocks);
      free(stack);
      return agc_isa_fail_memory(error);
   }
   uint32_t *edges = (uint32_t *)(blocks + block_count);
   size_t made = 0;
   for (size_t i = 0; i < count; i++) {
      if (!leader[i])
         continue;
      if (made)
         blocks[made - 1].end = (uint32_t)i;
      blocks[made].id = (uint32_t)made;
      blocks[made].first = (uint32_t)i;
      blocks[made].end = (uint32_t)count;
      made++;
   }
   free(leader);

   for (size_t id = 0; id < block_count; id++) {
      struct agc_isa_block *block = &blocks[id];
      const struct agc_isa_inst *last = &ir->instructions[block->end - 1];
      block->start_pc = ir->instructions[block->first].pc;
      if (last->has_branch_target) {
         size_t index = 0;
         bool found = agc_isa_ir_instruction_at(ir, last->branch_target, &index);
         assert(found);
         (void)found;
         block->successors[block->successor_count++] = (uint32_t)block_of(blocks, block_count, index);
      }
      if (agc_isa_inst_falls_through(last) && id + 1 < block_count) {
         if (!block->successor_count || block->successors[block->successor_count - 1] != id + 1)
            block->successors[block->successor_count++] = (uint32_t)(id + 1);
      }
      for (uint32_t s = 0; s < block->successor_count; s++)
         blocks[block->successors[s]].predecessor_count++;
   }
   size_t offset = 0;
   for (size_t id = 0; id < block_count; id++) {
      blocks[id].predecessors = edges + offset;
      offset += blocks[id].predecessor_count;
      blocks[id].predecessor_count = 0;
   }
   for (size_t id = 0; id < block_count; id++) {
      for (uint32_t s = 0; s < blocks[id].successor_count; s++) {
         struct agc_isa_block *successor = &blocks[blocks[id].successors[s]];
         edges[(size_t)(successor->predecessors - edges) + successor->predecessor_count++] = (uint32_t)id;
      }
   }

   size_t depth = 0;
   stack[depth++] = 0;
   if (agc_isa_computed_flow_has_dynamic_jumps(&ir->flow)) {
      for (size_t i = 0; i < entry_count; i++)
         stack[depth++] = (uint32_t)block_of(blocks, block_count, entries[i]);
   }
   free(entries);
   while (depth) {
      struct agc_isa_block *block = &blocks[stack[--depth]];
      if (block->reachable)
         continue;
      block->reachable = true;
      for (uint32_t s = 0; s < block->successor_count; s++)
         stack[depth++] = block->successors[s];
   }
   free(stack);

   free(ir->blocks);
   ir->blocks = blocks;
   ir->block_count = block_count;
   tail_jump_sgpr(ir);
   return true;
}
