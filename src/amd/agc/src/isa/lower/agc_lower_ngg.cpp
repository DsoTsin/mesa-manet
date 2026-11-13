#include "agc_lower.h"

#define SPI_SHADER_PGM_LO_GS 0xb220u
#define SPI_SHADER_PGM_HI_GS 0xb224u
#define SPI_SHADER_PGM_LO_ES 0xb320u
#define SPI_SHADER_PGM_HI_ES 0xb324u
#define VS_EXPORT_COUNT_SHIFT 1
#define NO_PC_EXPORT (1u << 7)
#define RING_WORD3 0x0043ffacu
#define RING_SWIZZLE_16_BYTES (3u << 30)

const struct agc_shader_register_relocation agc_isa_ngg_program_registers[AGC_SHADER_NGG_REGISTER_COUNT] = {
   {SPI_SHADER_PGM_LO_ES, AGC_SHADER_REGISTER_CODE_ADDRESS},
   {SPI_SHADER_PGM_HI_ES, AGC_SHADER_REGISTER_CODE_ADDRESS},
   {SPI_SHADER_PGM_LO_GS, AGC_SHADER_REGISTER_GUEST_USER_DATA_ADDRESS},
   {SPI_SHADER_PGM_HI_GS, AGC_SHADER_REGISTER_GUEST_USER_DATA_ADDRESS},
};

uint32_t
agc_isa_ngg_spi_vs_out_config(uint32_t param_count)
{
   uint32_t count = (param_count > 1 ? param_count : 1) - 1;
   uint32_t none = param_count == 0 ? NO_PC_EXPORT : 0;
   return (count << VS_EXPORT_COUNT_SHIFT) | none;
}

void
agc_isa_ngg_ring_descriptor(struct agc_shader_attribute_ring ring, uint32_t param_count, uint32_t words[4])
{
   uint32_t stride = 16 * (param_count > 1 ? param_count : 1);
   words[0] = (uint32_t)ring.address;
   words[1] = ((uint32_t)(ring.address >> 32) & 0xffff) | (stride << 16) | RING_SWIZZLE_16_BYTES;
   words[2] = ring.bytes;
   words[3] = RING_WORD3;
}

#define EXP_PARAM_LAST 63

static bool
fail_kind(struct agc_isa_error *error, enum agc_isa_error_kind kind)
{
   memset(error, 0, sizeof(*error));
   error->kind = kind;
   return false;
}

static bool
is_param(uint8_t target)
{
   return target >= AGC_ISA_EXP_PARAM_FIRST && target <= EXP_PARAM_LAST;
}

static bool
is_done_position(const struct agc_isa_export *e)
{
   return e->done && e->target >= AGC_ISA_EXP_POS_FIRST && e->target <= AGC_ISA_EXP_POS_LAST;
}

static const struct agc_isa_export *
export_of(const struct agc_isa_inst *inst)
{
   return inst->fields == AGC_ISA_EXP ? &inst->exp : NULL;
}

static struct agc_isa_kept
kept_sgpr(uint8_t sgpr)
{
   struct agc_isa_kept kept = {false, sgpr};
   return kept;
}

static struct agc_isa_kept
kept_lane(uint8_t lane)
{
   struct agc_isa_kept kept = {true, lane};
   return kept;
}

static uint8_t
top_lane(bool wave64, uint8_t index)
{
   return (uint8_t)((wave64 ? 63 : 31) - index);
}

static bool
has_param(const struct agc_isa_ir *ir, uint32_t first, uint32_t end)
{
   for (uint32_t i = first; i < end; i++) {
      const struct agc_isa_export *e = export_of(&ir->instructions[i]);
      if (e && is_param(e->target))
         return true;
   }
   return false;
}

static bool
params_follow_done_position(const struct agc_isa_ir *ir, bool *follow)
{
   size_t count = ir->block_count;
   bool *seen = (bool *)calloc(count ? count : 1, sizeof(*seen));
   uint32_t *work = (uint32_t *)malloc((4 * count + 1) * sizeof(*work));
   if (!seen || !work) {
      free(seen);
      free(work);
      return false;
   }
   bool dynamic = agc_isa_computed_flow_has_dynamic_jumps(&ir->flow);
   size_t depth = 0;
   *follow = false;
   for (size_t b = 0; b < count && !*follow; b++) {
      const struct agc_isa_block *block = &ir->blocks[b];
      if (!block->reachable)
         continue;
      for (uint32_t i = block->first; i < block->end; i++) {
         const struct agc_isa_export *e = export_of(&ir->instructions[i]);
         if (!e || !is_done_position(e))
            continue;
         if (dynamic || has_param(ir, i + 1, block->end)) {
            *follow = true;
         } else {
            for (uint32_t s = 0; s < block->successor_count; s++)
               work[depth++] = block->successors[s];
         }
         break;
      }
   }
   while (!*follow && depth) {
      uint32_t id = work[--depth];
      if (seen[id])
         continue;
      seen[id] = true;
      const struct agc_isa_block *block = &ir->blocks[id];
      if (has_param(ir, block->first, block->end)) {
         *follow = true;
         break;
      }
      for (uint32_t s = 0; s < block->successor_count; s++)
         work[depth++] = block->successors[s];
   }
   free(seen);
   free(work);
   return true;
}

static bool
leaves_program(const struct agc_isa_ir *ir)
{
   for (size_t i = 0; i < ir->flow.jump_count; i++) {
      if (ir->flow.jumps[i].target.kind == AGC_ISA_JUMP_EXTERNAL)
         return true;
   }
   return false;
}

bool
agc_isa_ngg_plan_init(struct agc_isa_ngg_plan *plan, const struct agc_isa_ir *ir, struct agc_isa_regalloc *regs,
                      bool wave64, bool has_ring, struct agc_shader_attribute_ring ring, bool spill,
                      struct agc_isa_error *error)
{
   memset(plan, 0, sizeof(*plan));
   uint32_t param_count = agc_isa_export_info_param_count(&ir->metadata.exports);
   if (!has_ring || param_count == 0)
      return true;
   if (ir->instruction_count == 0)
      return fail_kind(error, AGC_ISA_ERROR_EMPTY_PROGRAM);
   const struct agc_isa_inst *first = &ir->instructions[0];
   enum agc_isa_target target = agc_isa_regalloc_target(regs);
   const char *no_sgprs = "attribute-ring stores need SGPRs above the shader's own";
   uint8_t base;
   struct agc_isa_kept descriptor;
   struct agc_isa_kept offset;
   if (agc_isa_regalloc_reserve_sgprs(regs, 5, &base)) {
      descriptor = kept_sgpr(base);
      offset = kept_sgpr((uint8_t)(base + 4));
   } else if (spill) {
      plan->persistent_lanes += 5;
      descriptor = kept_lane(top_lane(wave64, 4));
      offset = kept_lane(top_lane(wave64, 0));
   } else {
      return agc_isa_lowering_error_instruction(error, first, target, no_sgprs);
   }
   uint8_t vertex;
   if (!agc_isa_regalloc_reserve_program_vgprs(regs, 1, &vertex, error))
      return false;
   plan->has_ring = true;
   plan->ring.descriptor = descriptor;
   plan->ring.offset = offset;
   plan->ring.vertex = vertex;
   agc_isa_ngg_ring_descriptor(ring, param_count, plan->ring.words);
   bool follow;
   if (!params_follow_done_position(ir, &follow))
      return fail_kind(error, AGC_ISA_ERROR_OUT_OF_MEMORY);
   if (!follow)
      return true;
   const struct agc_isa_export *first_done = NULL;
   bool differ = false;
   for (size_t b = 0; b < ir->block_count; b++) {
      const struct agc_isa_block *block = &ir->blocks[b];
      if (!block->reachable)
         continue;
      for (uint32_t i = block->first; i < block->end; i++) {
         const struct agc_isa_export *e = export_of(&ir->instructions[i]);
         if (!e || !is_done_position(e))
            continue;
         if (!first_done)
            first_done = e;
         else if (e->target != first_done->target || e->en != first_done->en || e->compr != first_done->compr)
            differ = true;
      }
   }
   if (!first_done)
      return true;
   if (leaves_program(ir))
      return agc_isa_lowering_error_instruction(
         error, first, target, "a deferred done position export cannot follow a jump out of the program");
   if (differ)
      return agc_isa_lowering_error_instruction(
         error, first, target, "deferring done position exports with different targets or channels");
   uint8_t data;
   if (!agc_isa_regalloc_reserve_program_vgprs(regs, 4, &data, error))
      return false;
   uint8_t sgpr;
   struct agc_isa_kept exec;
   if (agc_isa_regalloc_reserve_sgprs(regs, wave64 ? 2 : 1, &sgpr)) {
      exec = kept_sgpr(sgpr);
   } else if (spill) {
      uint8_t lane = top_lane(wave64, (uint8_t)(plan->persistent_lanes + (wave64 ? 1 : 0)));
      plan->persistent_lanes += wave64 ? 2 : 1;
      exec = kept_lane(lane);
   } else {
      return agc_isa_lowering_error_instruction(error, first, target, no_sgprs);
   }
   plan->has_deferred = true;
   plan->deferred.target = first_done->target;
   plan->deferred.en = first_done->en;
   plan->deferred.data = data;
   plan->deferred.exec = exec;
   return true;
}

namespace agc {

static const uint32_t attr_offset_field = 15u << 16;
static const uint32_t wave_index_field = 24u | (4u << 16);
static const uint32_t attr_offset_unit_shift = 9;
static const unsigned gs_attr_offset_sgpr = 5;
static const unsigned merged_wave_info_sgpr = 3;

static void
ring_descriptor(lower_ctx &ctx, unsigned sgpr, const uint32_t words[4])
{
   for (unsigned k = 0; k < 4; k++)
      ctx.bld.sop1(aco_opcode::s_mov_b32, sdef(sgpr + k), lit(words[k]));
}

static void
read_mask(lower_ctx &ctx, const agc_isa_inst *origin, unsigned sgpr, unsigned lane)
{
   for (unsigned k = 0; k < ctx.bld.lm.size(); k++)
      read_lane(ctx, origin, (uint8_t)(sgpr + k), (uint8_t)(lane + k));
}

static void
write_mask(lower_ctx &ctx, const agc_isa_inst *origin, unsigned sgpr, unsigned lane)
{
   for (unsigned k = 0; k < ctx.bld.lm.size(); k++)
      write_lane(ctx, origin, sop(sgpr + k), (uint8_t)(lane + k));
}

void
ngg_prologue(lower_ctx &ctx, const agc_isa_inst *origin)
{
   if (!ctx.has_ngg)
      return;
   Builder &bld = ctx.bld;
   unsigned lm = bld.lm.size();
   const agc_isa_ngg_plan plan = ctx.ngg;
   if (plan.has_ring) {
      const agc_isa_ngg_ring *ring = &plan.ring;
      unsigned offset;
      unsigned first_lane;
      if (!ring->offset.lane) {
         offset = ring->offset.index;
         first_lane = transient_sgprs(ctx, origin, 1);
      } else {
         offset = transient_sgprs(ctx, origin, 2);
         first_lane = offset + 1;
      }
      bld.sop2(aco_opcode::s_bfe_u32, sdef(offset), scc_def(), sop(gs_attr_offset_sgpr), lit(attr_offset_field));
      bld.sop2(aco_opcode::s_lshl_b32, sdef(offset), scc_def(), sop(offset), imm(attr_offset_unit_shift));
      if (ring->offset.lane)
         write_lane(ctx, origin, sop(offset), ring->offset.index);
      bld.sop2(aco_opcode::s_bfe_u32, sdef(first_lane), scc_def(), sop(merged_wave_info_sgpr), lit(wave_index_field));
      bld.sop2(aco_opcode::s_lshl_b32, sdef(first_lane), scc_def(), sop(first_lane), imm(ctx.wave64 ? 6 : 5));
      bld.vop3(aco_opcode::v_mbcnt_lo_u32_b32, vdef(ring->vertex), imm(0xffffffffu), sop(first_lane));
      if (ctx.wave64)
         bld.vop3(aco_opcode::v_mbcnt_hi_u32_b32_e64, vdef(ring->vertex), imm(0xffffffffu), vop(ring->vertex));
      if (!ring->descriptor.lane) {
         ring_descriptor(ctx, ring->descriptor.index, ring->words);
      } else {
         unsigned descriptor = transient_sgprs(ctx, origin, 4);
         ring_descriptor(ctx, descriptor, ring->words);
         for (unsigned k = 0; k < 4; k++)
            write_lane(ctx, origin, sop(descriptor + k), (uint8_t)(ring->descriptor.index + k));
      }
   }
   if (plan.has_deferred) {
      if (!plan.deferred.exec.lane) {
         bld.sop1(Builder::s_mov, sdef(plan.deferred.exec.index, lm), imm(0));
      } else {
         for (unsigned k = 0; k < lm; k++)
            write_lane(ctx, origin, imm(0), (uint8_t)(plan.deferred.exec.index + k));
      }
   }
}

void
init_full_exec(lower_ctx &ctx)
{
   ctx.bld.sop1(Builder::s_mov, mask_def(ctx, exec), imm(0xffffffffu));
}

void
ring_store(lower_ctx &ctx, const agc_isa_inst *inst, const agc_isa_export *e, const agc_isa_ngg_ring *ring)
{
   uint8_t dwords;
   bool enabled[4];
   if (e->compr) {
      dwords = 2;
      enabled[0] = (e->en & 0x3) != 0;
      enabled[1] = (e->en & 0xc) != 0;
      enabled[2] = false;
      enabled[3] = false;
   } else {
      dwords = 4;
      for (unsigned c = 0; c < 4; c++)
         enabled[c] = (e->en & (1u << c)) != 0;
   }
   bool contiguous = true;
   for (unsigned c = 0; c < dwords && contiguous; c++)
      contiguous = enabled[c] && e->vsrc[c] == (uint8_t)(e->vsrc[0] + c);
   uint8_t data = e->vsrc[0];
   if (!contiguous) {
      data = scratch_vgprs(ctx, dwords);
      for (unsigned c = 0; c < dwords; c++)
         ctx.bld.vop1(aco_opcode::v_mov_b32, vdef(data + c), enabled[c] ? vop(e->vsrc[c]) : imm(0));
   }
   uint16_t opcode = dwords == 2 ? AGC_ISA_VMEM_STORE_DWORDX2 : AGC_ISA_VMEM_STORE_DWORDX4;
   const agc_isa_mem_op *info = agc_isa_mubuf_op(opcode);
   if (!info)
      fail(ctx, inst, "buffer stores are missing");
   agc_isa_xlat xlat = agc_isa_mem_op_translate(info, ctx.target);
   if (xlat.kind != AGC_ISA_XLAT_OP)
      fail(ctx, inst, "buffer stores are missing");
   uint8_t srsrc;
   uint8_t soffset;
   if (!ring->descriptor.lane && !ring->offset.lane) {
      srsrc = ring->descriptor.index;
      soffset = ring->offset.index;
   } else if (ring->descriptor.lane && ring->offset.lane) {
      uint8_t block = spill_block(ctx, inst, 5);
      for (unsigned k = 0; k < 4; k++)
         read_lane(ctx, inst, (uint8_t)(block + k), (uint8_t)(ring->descriptor.index + k));
      read_lane(ctx, inst, (uint8_t)(block + 4), ring->offset.index);
      srsrc = block;
      soffset = (uint8_t)(block + 4);
   } else {
      fail(ctx, inst, "an attribute ring with mixed storage");
   }
   agc_isa_inst store = agc_isa_inst_derive(inst, AGC_ISA_MUBUF, xlat.value, info->flags);
   agc_isa_buffer *b = &store.buffer;
   b->vdata = data;
   b->vaddr = ring->vertex;
   b->srsrc = srsrc;
   b->soffset = agc_isa_operand_sgpr(soffset);
   b->offset = (uint16_t)(16u * (uint8_t)(e->target - AGC_ISA_EXP_PARAM_FIRST));
   b->offen = false;
   b->idxen = true;
   b->glc = false;
   b->dlc = false;
   b->slc = false;
   b->tfe = false;
   b->lds = false;
   b->format = 0;
   b->kind = AGC_ISA_MEM_STORE;
   b->data_dwords = dwords;
   push(ctx, store);
}

void
wait_for_stores(lower_ctx &ctx)
{
   if (ctx.gfx12)
      gfx12_waits(ctx, false, 0, false, 0, false, 0, true, 0);
   else
      ctx.bld.sopk(aco_opcode::s_waitcnt_vscnt, Operand(sgpr_null, s1), 0);
}

void
defer_position(lower_ctx &ctx, const agc_isa_inst *inst, const agc_isa_export *e, const agc_isa_ngg_deferred *deferred)
{
   Builder &bld = ctx.bld;
   unsigned lm = bld.lm.size();
   for (unsigned c = 0; c < 4; c++) {
      if (e->en & (1u << c))
         bld.vop1(aco_opcode::v_mov_b32, vdef(deferred->data + c), vop(e->vsrc[c]));
   }
   unsigned t = transient_sgprs(ctx, inst, 4);
   save_scc(ctx, t);
   unsigned mask = deferred->exec.index;
   if (deferred->exec.lane) {
      mask = t + 2;
      read_mask(ctx, inst, mask, deferred->exec.index);
   }
   bld.sop2(Builder::s_or, sdef(mask, lm), scc_def(), sop(mask, lm), mask_op(ctx, exec));
   if (deferred->exec.lane)
      write_mask(ctx, inst, mask, deferred->exec.index);
   restore_scc(ctx, t);
}

void
ngg_epilogue(lower_ctx &ctx, const agc_isa_inst *origin, const agc_isa_ngg_deferred *deferred)
{
   Builder &bld = ctx.bld;
   unsigned lm = bld.lm.size();
   wait_for_stores(ctx);
   unsigned mask = deferred->exec.index;
   if (deferred->exec.lane) {
      mask = transient_sgprs(ctx, origin, 2);
      read_mask(ctx, origin, mask, deferred->exec.index);
   }
   bld.sopc(ctx.wave64 ? aco_opcode::s_cmp_eq_u64 : aco_opcode::s_cmp_eq_u32, scc_def(), sop(mask, lm), imm(0));
   branch(ctx, aco_opcode::s_cbranch_scc1, AGC_ISA_EPILOGUE_END_LABEL);
   bld.sop1(Builder::s_mov, mask_def(ctx, exec), sop(mask, lm));
   agc_isa_inst out = agc_isa_inst_derive(origin, AGC_ISA_EXP, 0, 0);
   out.exp.target = deferred->target;
   out.exp.en = deferred->en;
   out.exp.compr = false;
   out.exp.done = true;
   out.exp.vm = false;
   out.exp.row_en = false;
   for (unsigned c = 0; c < 4; c++)
      out.exp.vsrc[c] = (uint8_t)(deferred->data + c);
   push(ctx, out);
}

}
