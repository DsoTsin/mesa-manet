#include <algorithm>
#include <new>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <unordered_set>

#include "agc_lower.h"

#include "aco_interface.h"

#include "ac_gpu_info.h"
#include "util/memstream.h"

void
agc_isa_pc_set_finish(struct agc_isa_pc_set *set)
{
   free(set->items);
   memset(set, 0, sizeof(*set));
}

static size_t
pc_set_lower_bound(const struct agc_isa_pc_set *set, uint32_t pc)
{
   size_t lo = 0;
   size_t hi = set->count;
   while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      if (set->items[mid] < pc)
         lo = mid + 1;
      else
         hi = mid;
   }
   return lo;
}

bool
agc_isa_pc_set_insert(struct agc_isa_pc_set *set, uint32_t pc)
{
   size_t at = pc_set_lower_bound(set, pc);
   if (at < set->count && set->items[at] == pc)
      return true;
   if (set->count == set->capacity) {
      size_t capacity = set->capacity ? set->capacity * 2 : 16;
      uint32_t *items = (uint32_t *)realloc(set->items, capacity * sizeof(*items));
      if (!items)
         return false;
      set->items = items;
      set->capacity = capacity;
   }
   memmove(set->items + at + 1, set->items + at, (set->count - at) * sizeof(*set->items));
   set->items[at] = pc;
   set->count++;
   return true;
}

bool
agc_isa_pc_set_contains(const struct agc_isa_pc_set *set, uint32_t pc)
{
   size_t at = pc_set_lower_bound(set, pc);
   return at < set->count && set->items[at] == pc;
}

bool
agc_isa_lowering_error_instruction(struct agc_isa_error *error, const struct agc_isa_inst *inst,
                                   enum agc_isa_target target, const char *reason)
{
   memset(error, 0, sizeof(*error));
   error->kind = AGC_ISA_ERROR_UNSUPPORTED_INSTRUCTION;
   error->pc = inst->pc;
   error->family = inst->family;
   error->opcode = inst->opcode;
   error->target = (uint8_t)target;
   error->reason = reason;
   return false;
}

unsigned
agc_isa_lowering_source_operands(struct agc_isa_inst *inst, struct agc_isa_operand *operands[3])
{
   switch (inst->fields) {
   case AGC_ISA_SOP1:
      operands[0] = &inst->sop1.src0;
      return 1;
   case AGC_ISA_VOP1:
      operands[0] = &inst->vop1.src0;
      return 1;
   case AGC_ISA_SOP2:
      operands[0] = &inst->sop2.src0;
      operands[1] = &inst->sop2.src1;
      return 2;
   case AGC_ISA_SOPC:
      operands[0] = &inst->sopc.src0;
      operands[1] = &inst->sopc.src1;
      return 2;
   case AGC_ISA_VOP2:
      operands[0] = &inst->vop2.src0;
      operands[1] = &inst->vop2.vsrc1;
      return 2;
   case AGC_ISA_VOPC:
      operands[0] = &inst->vopc.src0;
      operands[1] = &inst->vopc.vsrc1;
      return 2;
   case AGC_ISA_VOP3:
      for (unsigned i = 0; i < 3; i++)
         operands[i] = &inst->vop3.src[i];
      return 3;
   case AGC_ISA_VOP3P:
      for (unsigned i = 0; i < 3; i++)
         operands[i] = &inst->vop3p.src[i];
      return 3;
   case AGC_ISA_SMEM:
      operands[0] = &inst->smem.soffset;
      return 1;
   case AGC_ISA_MUBUF:
   case AGC_ISA_MTBUF:
      operands[0] = &inst->buffer.soffset;
      return 1;
   default:
      return 0;
   }
}

void
agc_isa_lowerer_init(struct agc_isa_lowerer *lowerer, const struct agc_shader_options *options,
                     enum agc_shader_precision precision)
{
   memset(lowerer, 0, sizeof(*lowerer));
   lowerer->options = *options;
   lowerer->precision = precision;
}

static void
agc_isa_lowerer_with_vertex_stage(struct agc_isa_lowerer *lowerer, enum agc_shader_vertex_stage stage)
{
   lowerer->hull = stage == AGC_SHADER_VERTEX_STAGE_LS_HS;
}

static void
agc_isa_lowerer_with_gds_buffer(struct agc_isa_lowerer *lowerer, bool has_address, uint64_t address)
{
   lowerer->has_gds_buffer = has_address;
   lowerer->gds_buffer = has_address ? address : 0;
}

static void
agc_isa_lowerer_with_mip_stats_table(struct agc_isa_lowerer *lowerer, bool has_address, uint64_t address)
{
   lowerer->has_mip_stats_table = has_address;
   lowerer->mip_stats_table = has_address ? address : 0;
}

static void
agc_isa_lowerer_with_image_reports(struct agc_isa_lowerer *lowerer, bool has_address, uint64_t address)
{
   lowerer->has_image_reports = has_address;
   lowerer->image_reports = has_address ? address : 0;
}

static void
agc_isa_lowerer_with_attribute_ring(struct agc_isa_lowerer *lowerer, bool has_ring,
                                    struct agc_shader_attribute_ring ring)
{
   lowerer->has_attribute_ring = has_ring;
   if (has_ring)
      lowerer->attribute_ring = ring;
   else
      memset(&lowerer->attribute_ring, 0, sizeof(lowerer->attribute_ring));
}

void
agc_isa_lowerer_with_pixel_interface(struct agc_isa_lowerer *lowerer, bool has_prim_mask, uint8_t prim_mask,
                                     bool has_col_format, uint32_t col_format, bool has_z_format,
                                     uint32_t z_format, bool dual_source)
{
   lowerer->pixel.has_prim_mask = has_prim_mask;
   lowerer->pixel.prim_mask = has_prim_mask ? prim_mask : 0;
   lowerer->pixel.has_col_format = has_col_format;
   lowerer->pixel.col_format = has_col_format ? col_format : 0;
   lowerer->pixel.has_z_format = has_z_format;
   lowerer->pixel.z_format = has_z_format ? z_format : 0;
   lowerer->pixel.dual_source = dual_source;
}

static void
agc_isa_lowerer_with_guest_code_address(struct agc_isa_lowerer *lowerer, bool has_address, uint64_t address)
{
   lowerer->has_guest_code_address = has_address;
   lowerer->guest_code_address = has_address ? address : 0;
}

static void
agc_isa_lowerer_with_descriptor_conversion(struct agc_isa_lowerer *lowerer, bool has_lut, uint64_t lut,
                                           bool preconvert)
{
   lowerer->has_descriptor_lut = has_lut;
   lowerer->descriptor_lut = has_lut ? lut : 0;
   lowerer->preconvert_user_descriptors = preconvert;
}

static void
agc_isa_lowerer_with_native_dcc(struct agc_isa_lowerer *lowerer, bool native)
{
   lowerer->native_dcc = native;
}

void
agc_isa_lowerer_with_interface(struct agc_isa_lowerer *lowerer, const struct agc_isa_ir *ir,
                               const struct agc_shader_interface *iface)
{
   bool user_sgprs_known = iface->has_user_sgpr_count || iface->has_guest_pgm_rsrc2;
   agc_isa_lowerer_with_guest_code_address(lowerer, iface->has_guest_code_address, iface->guest_code_address);
   agc_isa_lowerer_with_descriptor_conversion(lowerer, iface->has_descriptor_lut, iface->descriptor_lut,
                                              iface->preconvert_user_descriptors);
   agc_isa_lowerer_with_native_dcc(lowerer, iface->native_dcc);
   agc_isa_lowerer_with_pixel_interface(lowerer, user_sgprs_known, ir->metadata.user_sgpr_count,
                                        iface->has_ps_col_format, iface->ps_col_format, iface->has_ps_z_format,
                                        iface->ps_z_format, iface->dual_source_blend);
   agc_isa_lowerer_with_attribute_ring(lowerer, iface->has_attribute_ring, iface->attribute_ring);
   agc_isa_lowerer_with_gds_buffer(lowerer, iface->has_gds_buffer, iface->gds_buffer);
   agc_isa_lowerer_with_mip_stats_table(lowerer, iface->has_mip_stats_table, iface->mip_stats_table);
   agc_isa_lowerer_with_image_reports(lowerer, iface->has_image_reports, iface->image_reports);
   agc_isa_lowerer_with_vertex_stage(lowerer, iface->vertex_stage);
   lowerer->es_input = iface->es_input;
}

enum use {
   USE_NONE,
   USE_PRIVATE,
   USE_COMPUTED,
   USE_FLAT_SCRATCH,
   USE_STRAY
};

enum scc_write {
   SCC_KEPT,
   SCC_WRITTEN,
   SCC_UNKNOWN
};

struct flow {
   struct agc_isa_mask128 derived;
   bool scc;
   bool exact;
   bool reached;
};

static struct flow
flow_unreached(void)
{
   struct flow flow = {{0, 0}, false, true, false};
   return flow;
}

static bool
flow_equal(struct flow a, struct flow b)
{
   return agc_isa_mask128_equal(a.derived, b.derived) && a.scc == b.scc && a.exact == b.exact &&
          a.reached == b.reached;
}

static struct flow
flow_join(struct flow a, struct flow b)
{
   if (!a.reached)
      return b;
   if (!b.reached)
      return a;
   struct flow flow = {agc_isa_mask128_or(a.derived, b.derived), a.scc || b.scc, a.exact && b.exact, true};
   return flow;
}

static bool
fail_memory(struct agc_isa_error *error)
{
   memset(error, 0, sizeof(*error));
   error->kind = AGC_ISA_ERROR_OUT_OF_MEMORY;
   return false;
}

static struct agc_isa_mask128
range_mask(unsigned first, unsigned count)
{
   struct agc_isa_mask128 mask = {0, 0};
   unsigned end = first + count;
   if (end > UINT16_MAX)
      end = UINT16_MAX;
   for (unsigned reg = first; reg < end && reg < 128; reg++)
      mask = agc_isa_mask128_or(mask, agc_isa_mask128_bit(reg));
   return mask;
}

static struct agc_isa_mask128
scalar_mask(const struct agc_isa_inst *inst, bool wave64, bool defs)
{
   struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
   unsigned count = defs ? agc_isa_inst_defs(inst, wave64, ranges) : agc_isa_inst_uses(inst, wave64, ranges);
   struct agc_isa_mask128 mask = {0, 0};
   for (unsigned i = 0; i < count; i++) {
      if (ranges[i].file == AGC_ISA_REG_SCALAR)
         mask = agc_isa_mask128_or(mask, range_mask(ranges[i].first, ranges[i].count ? ranges[i].count : 1));
   }
   return mask;
}

static bool
operand_is_scc(struct agc_isa_operand op)
{
   return op.kind == AGC_ISA_OPERAND_SCC;
}

static bool
reads_scc(const struct agc_isa_inst *inst)
{
   switch (inst->family) {
   case AGC_ISA_SOP2:
      if (inst->opcode == 0x04 || inst->opcode == 0x05 || inst->opcode == 0x0a || inst->opcode == 0x0b)
         return true;
      break;
   case AGC_ISA_SOP1:
      if (inst->opcode == 0x05 || inst->opcode == 0x06)
         return true;
      break;
   case AGC_ISA_SOPK:
      if (inst->opcode == 0x02)
         return true;
      break;
   default:
      break;
   }
   switch (inst->fields) {
   case AGC_ISA_SOP1:
      return operand_is_scc(inst->sop1.src0);
   case AGC_ISA_SOP2:
      return operand_is_scc(inst->sop2.src0) || operand_is_scc(inst->sop2.src1);
   case AGC_ISA_SOPC:
      return operand_is_scc(inst->sopc.src0) || operand_is_scc(inst->sopc.src1);
   default:
      return false;
   }
}

static enum scc_write
writes_scc(const struct agc_isa_inst *inst)
{
   uint16_t op = inst->opcode;
   switch (inst->family) {
   case AGC_ISA_SOPC:
      return SCC_WRITTEN;
   case AGC_ISA_SOP2:
      if (op == 0x0a || op == 0x0b || op == 0x24 || op == 0x25 || op == 0x26 || (op >= 0x32 && op <= 0x36))
         return SCC_KEPT;
      return SCC_WRITTEN;
   case AGC_ISA_SOPK:
      return op >= 0x03 && op <= 0x0f ? SCC_WRITTEN : SCC_KEPT;
   default:
      return SCC_UNKNOWN;
   }
}

static bool
is_scalar_alu(const struct agc_isa_inst *inst)
{
   if (inst->family == AGC_ISA_SOP1 || inst->family == AGC_ISA_SOP2 || inst->family == AGC_ISA_SOPC)
      return true;
   return inst->family == AGC_ISA_SOPK && inst->opcode != AGC_ISA_SOPK_SETREG_B32 &&
          inst->opcode != AGC_ISA_SOPK_SETREG_IMM32_B32;
}

static bool
writes_flat_scratch(const struct agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_SOPK || inst->family != AGC_ISA_SOPK || inst->opcode != AGC_ISA_SOPK_SETREG_B32)
      return false;
   uint16_t id = inst->sopk.simm16 & AGC_ISA_HWREG_ID_MASK;
   return id == AGC_ISA_HWREG_FLAT_SCR_LO || id == AGC_ISA_HWREG_FLAT_SCR_HI;
}

static enum use
step(const struct agc_isa_inst *inst, struct flow *flow, uint8_t sgpr, bool wave64, bool forget)
{
   struct agc_isa_mask128 offset = agc_isa_mask128_bit(sgpr);
   struct agc_isa_mask128 reads = scalar_mask(inst, wave64, false);
   struct agc_isa_mask128 defs = scalar_mask(inst, wave64, true);
   bool from_scc = flow->scc && reads_scc(inst);
   struct agc_isa_mask128 derived_reads = agc_isa_mask128_and(reads, flow->derived);
   bool derived_read = !agc_isa_mask128_is_zero(derived_reads) || from_scc;
   bool private_soffset = (inst->fields == AGC_ISA_MUBUF || inst->fields == AGC_ISA_MTBUF) &&
                          inst->buffer.soffset.kind == AGC_ISA_OPERAND_SGPR && inst->buffer.soffset.reg == sgpr;
   enum use used;
   if (!derived_read)
      used = USE_NONE;
   else if (private_soffset && flow->exact && agc_isa_mask128_equal(derived_reads, offset))
      used = USE_PRIVATE;
   else if (writes_flat_scratch(inst))
      used = USE_FLAT_SCRATCH;
   else if (is_scalar_alu(inst) && !agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_JUMP | AGC_ISA_FLAG_CALL))
      used = USE_COMPUTED;
   else
      used = USE_STRAY;
   if (used == USE_COMPUTED)
      flow->derived = agc_isa_mask128_or(flow->derived, defs);
   else if (forget)
      flow->derived = agc_isa_mask128_andnot(flow->derived, defs);
   if (!agc_isa_mask128_is_zero(agc_isa_mask128_and(defs, offset)))
      flow->exact = false;
   if (is_scalar_alu(inst)) {
      switch (writes_scc(inst)) {
      case SCC_WRITTEN:
         if (forget)
            flow->scc = derived_read;
         else
            flow->scc = flow->scc || derived_read;
         break;
      case SCC_UNKNOWN:
         flow->scc = flow->scc || derived_read;
         break;
      case SCC_KEPT:
         break;
      }
   }
   return used;
}

static bool
work_push(uint32_t **items, size_t *count, size_t *capacity, uint32_t id)
{
   if (*count == *capacity) {
      size_t grown = *capacity ? *capacity * 2 : 16;
      uint32_t *resized = (uint32_t *)realloc(*items, grown * sizeof(*resized));
      if (!resized)
         return false;
      *items = resized;
      *capacity = grown;
   }
   (*items)[(*count)++] = id;
   return true;
}

bool
agc_isa_lowering_wave_offset_find(const struct agc_isa_ir *ir, bool wave64, uint8_t sgpr,
                                  enum agc_isa_target target, struct agc_isa_wave_offset_uses *uses,
                                  struct agc_isa_error *error)
{
   struct flow entry = {agc_isa_mask128_bit(sgpr), false, true, true};
   size_t blocks = ir->block_count;
   struct flow *at_entry = (struct flow *)malloc((blocks ? blocks : 1) * sizeof(*at_entry));
   if (!at_entry)
      return fail_memory(error);
   for (size_t id = 0; id < blocks; id++)
      at_entry[id] = flow_unreached();
   bool dynamic = agc_isa_computed_flow_has_dynamic_jumps(&ir->flow);
   if (dynamic) {
      struct flow all = entry;
      for (;;) {
         struct flow before = all;
         for (size_t b = 0; b < blocks; b++) {
            const struct agc_isa_block *block = &ir->blocks[b];
            if (!block->reachable)
               continue;
            for (uint32_t i = block->first; i < block->end; i++)
               step(&ir->instructions[i], &all, sgpr, wave64, false);
         }
         if (flow_equal(all, before))
            break;
      }
      for (size_t id = 0; id < blocks; id++)
         at_entry[id] = all;
   } else if (blocks) {
      at_entry[0] = entry;
      uint32_t *work = NULL;
      size_t depth = 0;
      size_t capacity = 0;
      if (!work_push(&work, &depth, &capacity, 0)) {
         free(at_entry);
         return fail_memory(error);
      }
      while (depth) {
         uint32_t id = work[--depth];
         struct flow flow = at_entry[id];
         const struct agc_isa_block *block = &ir->blocks[id];
         for (uint32_t i = block->first; i < block->end; i++)
            step(&ir->instructions[i], &flow, sgpr, wave64, true);
         for (uint32_t s = 0; s < block->successor_count; s++) {
            uint32_t next = block->successors[s];
            struct flow joined = flow_join(at_entry[next], flow);
            if (!flow_equal(joined, at_entry[next])) {
               at_entry[next] = joined;
               if (!work_push(&work, &depth, &capacity, next)) {
                  free(work);
                  free(at_entry);
                  return fail_memory(error);
               }
            }
         }
      }
      free(work);
   }
   bool forget = !dynamic;
   bool ok = true;
   for (size_t b = 0; ok && b < blocks; b++) {
      const struct agc_isa_block *block = &ir->blocks[b];
      if (!block->reachable || !at_entry[block->id].reached)
         continue;
      struct flow flow = at_entry[block->id];
      for (uint32_t i = block->first; ok && i < block->end; i++) {
         const struct agc_isa_inst *inst = &ir->instructions[i];
         switch (step(inst, &flow, sgpr, wave64, forget)) {
         case USE_PRIVATE:
            if (!agc_isa_pc_set_insert(&uses->private_pcs, inst->pc))
               ok = fail_memory(error);
            break;
         case USE_COMPUTED:
            uses->computed = true;
            break;
         case USE_STRAY:
            ok = agc_isa_lowering_error_instruction(error, inst, target,
                                                    "a value computed from the scratch wave offset is used outside "
                                                    "the private segment and the flat scratch setup");
            break;
         default:
            break;
         }
      }
   }
   free(at_entry);
   return ok;
}

namespace agc {

static const uint16_t spill_headroom_sgprs = 8;
static const uint16_t reserved_sgpr_estimate = 8;
static const uint32_t packed_tid_bits = 10;
static const uint32_t tg_size_wave_id = 20u | (5u << 16);

struct program_state {
   agc_isa_resource_binding *preconverted = NULL;
   size_t preconverted_count = 0;
   bool has_pixel = false;
   agc_isa_pixel_plan pixel;
   agc_isa_wave_offset_uses scratch;
   bool *static_reach = NULL;

   program_state() { memset(&scratch, 0, sizeof(scratch)); }
   ~program_state()
   {
      free(preconverted);
      if (has_pixel)
         agc_isa_pixel_plan_finish(&pixel);
      agc_isa_pc_set_finish(&scratch.private_pcs);
      free(static_reach);
   }
};

static bool
target_to_gfx(enum agc_isa_target target, enum amd_gfx_level *level, enum radeon_family *family)
{
   switch (target) {
   case AGC_ISA_TARGET_GFX103:
      *level = GFX10_3;
      *family = CHIP_NAVI21;
      return true;
   case AGC_ISA_TARGET_GFX11:
      *level = GFX11;
      *family = CHIP_NAVI31;
      return true;
   case AGC_ISA_TARGET_GFX115:
      *level = GFX11_5;
      *family = CHIP_STRIX_HALO;
      return true;
   case AGC_ISA_TARGET_GFX12:
      *level = GFX12;
      *family = CHIP_GFX1201;
      return true;
   default:
      return false;
   }
}

[[noreturn]] static void
fail_first(const lower_ctx &ctx, const char *reason)
{
   if (ctx.ir->instruction_count == 0)
      fail_plain();
   fail(ctx, &ctx.ir->instructions[0], reason);
}

static void
begin_context(lower_ctx &ctx, bool program_entry)
{
   wait_tracker_init(ctx.waits, program_entry);
   memset(&ctx.descriptor_cache, 0, sizeof(ctx.descriptor_cache));
   ctx.has_pending_dual_source = false;
   memset(&ctx.pending_dual_source, 0, sizeof(ctx.pending_dual_source));
   ctx.epilogue = ctx.has_ngg && ctx.ngg.has_deferred;
   memset(&ctx.spill, 0, sizeof(ctx.spill));
   if (ctx.has_spill) {
      ctx.spill.vgpr = ctx.spill_vgpr;
      ctx.spill.persistent = ctx.spill_persistent;
   }
   ctx.pending.clear();
   ctx.staged.clear();
   ctx.staged.emplace_back();
   ctx.split = false;
   ctx.has_next_label = false;
}

static void
commit(lower_ctx &ctx, uint32_t name, bool dynamic_entry, size_t id)
{
   flush(ctx);
   ctx.staged.front().has_label = true;
   ctx.staged.front().label = name;
   ctx.staged.front().dynamic_entry = dynamic_entry;
   lowered_block block = {id, ctx.segments.size(), ctx.staged.size(), ctx.split};
   for (segment &seg : ctx.staged)
      ctx.segments.emplace_back(std::move(seg));
   ctx.staged.clear();
   ctx.blocks.push_back(block);
}

static void
reject_streamout(const lower_ctx &ctx)
{
   const agc_isa_ir *ir = ctx.ir;
   if (ctx.stage != AGC_SHADER_STAGE_VERTEX)
      return;
   for (size_t b = 0; b < ir->block_count; b++) {
      const agc_isa_block *block = &ir->blocks[b];
      if (!block->reachable)
         continue;
      for (uint32_t i = block->first; i < block->end; i++) {
         const agc_isa_inst *inst = &ir->instructions[i];
         if (inst->family == AGC_ISA_DS && inst->opcode == AGC_ISA_DS_ORDERED_COUNT)
            fail(ctx, inst, "streamout (GDS ordered counters in a vertex program) is not supported");
      }
   }
}

static bool
uses_binding(const agc_isa_ir *ir, enum agc_isa_resource_kind kind)
{
   for (size_t i = 0; i < ir->metadata.binding_count; i++) {
      if (ir->metadata.bindings[i].kind == kind)
         return true;
   }
   return false;
}

static uint16_t
descriptor_slot_sgprs(const lower_ctx &ctx)
{
   if (!ctx.lowerer->has_descriptor_lut)
      return 0;
   bool images = uses_binding(ctx.ir, AGC_ISA_RESOURCE_IMAGE_DESCRIPTOR);
   bool samplers = uses_binding(ctx.ir, AGC_ISA_RESOURCE_SAMPLER_DESCRIPTOR);
   bool buffers = uses_binding(ctx.ir, AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR);
   return (uint16_t)((images ? 8 : 0) + (buffers ? 4 : 0) + (samplers ? 4 : 0) + (images || buffers ? 4 : 0));
}

static uint8_t
reserve_slot(lower_ctx &ctx, bool wanted, uint8_t count)
{
   uint8_t slot = AGC_ISA_NO_SLOT;
   if (!wanted)
      return slot;
   if (agc_isa_regalloc_reserve_sgprs(ctx.regs, count, &slot))
      return slot;
   uint16_t sgprs = ctx.ir->metadata.sgpr_count;
   const char *reason = "descriptor conversion needs SGPRs above the shader's own: it uses at most 88";
   if (ctx.ir->metadata.uses_relative_sgprs)
      reason = "descriptor conversion needs SGPRs above the shader's own: it indexes SGPRs relatively";
   else if (sgprs > 96)
      reason = "descriptor conversion needs SGPRs above the shader's own: it uses more than 96";
   else if (sgprs > 88)
      reason = "descriptor conversion needs SGPRs above the shader's own: it uses 89 to 96";
   fail_first(ctx, reason);
}

static void
reserve_descriptor_slots(lower_ctx &ctx, bool spill)
{
   ctx.has_descriptors = false;
   if (!ctx.lowerer->has_descriptor_lut || !ctx.gfx11_plus)
      return;
   agc_isa_descriptor_slots *slots = &ctx.descriptors;
   slots->address = ctx.lowerer->descriptor_lut;
   slots->native_dcc = ctx.lowerer->native_dcc;
   ctx.has_descriptors = true;
   if (spill) {
      slots->tsharp = AGC_ISA_NO_SLOT;
      slots->vsharp = AGC_ISA_NO_SLOT;
      slots->ssharp = AGC_ISA_NO_SLOT;
      slots->table = AGC_ISA_NO_SLOT;
      return;
   }
   bool images = uses_binding(ctx.ir, AGC_ISA_RESOURCE_IMAGE_DESCRIPTOR);
   bool samplers = uses_binding(ctx.ir, AGC_ISA_RESOURCE_SAMPLER_DESCRIPTOR);
   bool buffers = uses_binding(ctx.ir, AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR);
   slots->tsharp = reserve_slot(ctx, images, 8);
   slots->vsharp = reserve_slot(ctx, buffers, 4);
   slots->ssharp = reserve_slot(ctx, samplers, 4);
   slots->table = reserve_slot(ctx, images || buffers, 2);
}

static void
architected_workgroup_ids(lower_ctx &ctx)
{
   if (ctx.stage != AGC_SHADER_STAGE_COMPUTE || !ctx.gfx12)
      return;
   Builder &bld = ctx.bld;
   const agc_isa_system_sgprs *system = &ctx.regs->system;
   if (system->has_tg_size && system->has_host_tg_size && system->tg_size != system->host_tg_size)
      bld.sop1(aco_opcode::s_mov_b32, sdef(system->tg_size), sop(system->host_tg_size));
   for (unsigned i = 0; i < 3; i++) {
      uint8_t destination = system->workgroup_ids[i];
      if (destination == UINT8_MAX)
         continue;
      if (i == 0)
         bld.sop1(aco_opcode::s_mov_b32, sdef(destination), sop(108 + 9));
      else if (i == 1)
         bld.sop2(aco_opcode::s_and_b32, sdef(destination), scc_def(), sop(108 + 7), imm(0xffff));
      else
         bld.sop2(aco_opcode::s_lshr_b32, sdef(destination), scc_def(), sop(108 + 7), imm(16));
   }
}

static void
unpack_thread_ids(lower_ctx &ctx)
{
   const agc_isa_ir *ir = ctx.ir;
   uint8_t components = agc_isa_regalloc_thread_id_components(ctx.regs);
   if (!ctx.gfx11_plus || components == 0 || ir->instruction_count == 0)
      return;
   agc_isa_mask128 live = ir->metadata.live_in_vector;
   uint8_t highest = 0;
   for (uint8_t component = 2; component >= 1; component--) {
      if (component > components || !agc_isa_mask128_test(live, component))
         continue;
      if (component > highest)
         highest = component;
      ctx.bld.vop3(aco_opcode::v_bfe_u32, vdef(component), vop(0), imm(packed_tid_bits * component),
                   imm(packed_tid_bits));
   }
   if (agc_isa_mask128_test(live, 0))
      ctx.bld.vop2(aco_opcode::v_and_b32, vdef(0), lit((1u << packed_tid_bits) - 1), vop(0));
   agc_isa_regalloc_reserve_vgprs(ctx.regs, (uint16_t)(highest + 1));
}

static void
hull_prologue(lower_ctx &ctx, const agc_isa_inst *origin, agc_isa_mask128 live_in_vector)
{
   const uint8_t local_index_vgpr = 3;
   const uint8_t tcs_wave_id_sgpr = 5;
   const uint32_t tcs_wave_id_field = 3u << 16;
   Builder &bld = ctx.bld;
   if (ctx.gfx12 && agc_isa_mask128_test(live_in_vector, 5))
      bld.vop1(aco_opcode::v_mov_b32, vdef(5), vop(3));
   if (!ctx.gfx11_plus || !agc_isa_mask128_test(live_in_vector, local_index_vgpr))
      return;
   uint8_t first_lane = transient_sgprs(ctx, origin, 1);
   bld.sop2(aco_opcode::s_bfe_u32, sdef(first_lane), scc_def(), sop(tcs_wave_id_sgpr), lit(tcs_wave_id_field));
   bld.sop2(aco_opcode::s_mul_i32, sdef(first_lane), sop(first_lane), imm(ctx.wave64 ? 64 : 32));
   bld.vop3(aco_opcode::v_mbcnt_lo_u32_b32, vdef(local_index_vgpr), imm(0xffffffffu), sop(first_lane));
   if (ctx.wave64)
      bld.vop3(aco_opcode::v_mbcnt_hi_u32_b32_e64, vdef(local_index_vgpr), imm(0xffffffffu),
               vop(local_index_vgpr));
}

static void
save_wave_index(lower_ctx &ctx, const agc_isa_inst *origin)
{
   if (!ctx.has_wave_index)
      return;
   Operand source;
   uint32_t field = tg_size_wave_id;
   if (ctx.gfx12) {
      source = sop(108 + 8);
      field = 25u | (5u << 16);
   } else {
      uint8_t tg_size;
      if (!agc_isa_regalloc_tg_size_sgpr(ctx.regs, &tg_size))
         fail(ctx, origin, "the wave index needs the compute TG_SIZE SGPR");
      source = sop(tg_size);
   }
   uint8_t destination = ctx.wave_index.index;
   if (ctx.wave_index.lane)
      destination = transient_sgprs(ctx, origin, 1);
   ctx.bld.sop2(aco_opcode::s_bfe_u32, sdef(destination), scc_def(), source, lit(field));
   if (ctx.wave_index.lane)
      write_lane(ctx, origin, sop(destination), ctx.wave_index.index);
   if (ctx.regs->system.has_host_scratch_wave_offset)
      ctx.bld.sop1(aco_opcode::s_mov_b32, sdef(ctx.regs->system.scratch_wave_offset),
                   sop(ctx.regs->system.host_scratch_wave_offset));
}

static void
ngg_input_vgprs(lower_ctx &ctx, agc_isa_mask128 live, enum agc_shader_es_input input)
{
   if (!ctx.gfx12)
      return;
   static const uint8_t vertex[][2] = {{8, 4}, {5, 3}, {2, 1}};
   static const uint8_t domain[][2] = {{8, 6}, {7, 5}, {6, 4}, {5, 3}, {2, 1}};
   const uint8_t (*moves)[2] = input == AGC_SHADER_ES_DOMAIN ? domain : vertex;
   size_t count = input == AGC_SHADER_ES_DOMAIN ? 5 : 3;
   for (size_t i = 0; i < count; i++) {
      if (agc_isa_mask128_test(live, moves[i][0]))
         ctx.bld.vop1(aco_opcode::v_mov_b32, vdef(moves[i][0]), vop(moves[i][1]));
   }
}

static void
lower_prologue(lower_ctx &ctx, bool has_wave_offset, uint8_t wave_offset)
{
   const agc_isa_ir *ir = ctx.ir;
   const agc_isa_lowerer *lowerer = ctx.lowerer;
   if (ir->instruction_count)
      architected_workgroup_ids(ctx);
   if (ctx.stage == AGC_SHADER_STAGE_COMPUTE)
      unpack_thread_ids(ctx);
   if (ir->instruction_count == 0)
      return;
   const agc_isa_inst *first = &ir->instructions[0];
   if (ctx.stage == AGC_SHADER_STAGE_VERTEX)
      init_full_exec(ctx);
   if (lowerer->hull)
      hull_prologue(ctx, first, ir->metadata.live_in_vector);
   else if (ctx.stage == AGC_SHADER_STAGE_VERTEX)
      ngg_input_vgprs(ctx, ir->metadata.live_in_vector, lowerer->es_input);
   save_wave_index(ctx, first);
   set_second_half(ctx, first, ir->metadata.live_in_scalar);
   pops_inputs(ctx, ir->metadata.live_in_scalar);
   load_descriptor_table(ctx);
   load_hoisted_parameters(ctx, first);
   ngg_prologue(ctx, first);
   if (has_wave_offset)
      ctx.bld.sop1(aco_opcode::s_mov_b32, sdef(wave_offset), imm(0));
   restore_spills(ctx);
}

static void
inline_literal(agc_isa_operand *op, bool wide)
{
   agc_isa_operand inlined;
   if (op->kind == AGC_ISA_OPERAND_LITERAL && !wide && agc_isa_operand_inline_32(op->literal, &inlined))
      *op = inlined;
}

static agc_isa_inst
inline_scalar_literals(const agc_isa_inst *inst)
{
   agc_isa_inst out = *inst;
   bool wide0 = agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_S0_2);
   bool wide1 = agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_S1_2);
   switch (inst->fields) {
   case AGC_ISA_SOP1:
      if (!agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_SRC))
         inline_literal(&out.sop1.src0, wide0);
      break;
   case AGC_ISA_SOP2:
      inline_literal(&out.sop2.src0, wide0);
      inline_literal(&out.sop2.src1, wide1);
      break;
   case AGC_ISA_SOPC:
      inline_literal(&out.sopc.src0, wide0);
      inline_literal(&out.sopc.src1, wide1);
      break;
   default:
      break;
   }
   return out;
}

static bool
indexes_private_segment(const lower_ctx &ctx, const agc_isa_pc_set *private_pcs)
{
   if (!agc_isa_target_is_gfx11(ctx.target))
      return false;
   for (size_t i = 0; i < private_pcs->count; i++) {
      size_t index;
      if (!agc_isa_ir_instruction_at(ctx.ir, private_pcs->items[i], &index))
         continue;
      const agc_isa_inst *inst = &ctx.ir->instructions[index];
      if ((inst->fields == AGC_ISA_MUBUF || inst->fields == AGC_ISA_MTBUF) && inst->buffer.idxen)
         return true;
   }
   return false;
}

static bool *
statically_reachable(const agc_isa_ir *ir)
{
   size_t count = ir->block_count;
   bool *seen = (bool *)calloc(count ? count : 1, sizeof(*seen));
   size_t *stack = (size_t *)malloc((2 * count + 1) * sizeof(*stack));
   if (!seen || !stack) {
      free(seen);
      free(stack);
      fail_memory();
   }
   size_t depth = 0;
   stack[depth++] = 0;
   while (depth) {
      size_t id = stack[--depth];
      if (id >= count || seen[id])
         continue;
      seen[id] = true;
      const agc_isa_block *block = &ir->blocks[id];
      for (uint32_t s = 0; s < block->successor_count; s++)
         stack[depth++] = block->successors[s];
   }
   free(stack);
   return seen;
}

static void
lower_blocks(lower_ctx &ctx, const program_state &state)
{
   const agc_isa_ir *ir = ctx.ir;
   for (size_t b = 0; b < ir->block_count; b++) {
      const agc_isa_block *block = &ir->blocks[b];
      if (!block->reachable)
         continue;
      bool entry = block->id == 0 && block->predecessor_count == 0;
      begin_context(ctx, entry);
      try {
         for (uint32_t i = block->first; i < block->end; i++) {
            agc_isa_inst inst = inline_scalar_literals(&ir->instructions[i]);
            ctx.pc = inst.pc;
            try {
               lower_inst(ctx, &inst);
               restore_spills(ctx);
               flush(ctx);
            } catch (const encode_error &e) {
               ctx.pending.clear();
               note_reject(ctx, e.error);
            }
         }
         if (ctx.has_pending_dual_source)
            fail(ctx, &ir->instructions[block->end - 1],
                 "dual-source blending needs MRT0 and MRT1 exported in one block");
      } catch (const lower_error &e) {
         if (e.error.kind == AGC_ISA_ERROR_OUT_OF_MEMORY || state.static_reach[block->id])
            throw;
         ctx.pending.clear();
         ctx.staged.clear();
         continue;
      }
      bool dynamic_entry = ctx.has_dispatch && agc_isa_computed_flow_has_entry(&ir->flow, block->start_pc);
      commit(ctx, block->start_pc, dynamic_entry, block->id);
   }
}

template <typename F>
static void
lower_stub(lower_ctx &ctx, uint32_t name, const agc_isa_inst *origin, F &&body)
{
   begin_context(ctx, false);
   ctx.pc = origin->pc;
   try {
      body();
      flush(ctx);
   } catch (const encode_error &e) {
      ctx.pending.clear();
      note_reject(ctx, e.error);
   }
   commit(ctx, name, false, SIZE_MAX);
}

static void
redirect_dropped_blocks(lower_ctx &ctx)
{
   const agc_isa_ir *ir = ctx.ir;
   std::unordered_set<uint32_t> kept;
   std::vector<bool> is_lowered(ir->block_count + 1, false);
   for (const lowered_block &block : ctx.blocks) {
      kept.insert(ctx.segments[block.first].label);
      if (block.id != SIZE_MAX)
         is_lowered[block.id] = true;
   }
   std::vector<std::pair<size_t, uint32_t>> appended;
   for (size_t position = 0; position < ctx.blocks.size(); position++) {
      const lowered_block &lowered = ctx.blocks[position];
      if (lowered.id == SIZE_MAX)
         continue;
      const agc_isa_block *block = &ir->blocks[lowered.id];
      const agc_isa_inst *last = &ir->instructions[block->end - 1];
      bool falls_into_dropped = false;
      if (agc_isa_inst_falls_through(last) && !is_lowered[lowered.id + 1]) {
         for (uint32_t s = 0; s < block->successor_count && !falls_into_dropped; s++)
            falls_into_dropped = block->successors[s] == lowered.id + 1;
      }
      for (size_t i = 0; i < lowered.count; i++) {
         segment &seg = ctx.segments[lowered.first + i];
         if (seg.has_target && seg.target < AGC_ISA_LOCAL_LABEL_BASE && !kept.count(seg.target))
            seg.target = AGC_ISA_MISS_LABEL;
      }
      if (falls_into_dropped)
         appended.emplace_back(position, last->pc);
   }
   for (size_t k = appended.size(); k-- > 0;) {
      const lowered_block lowered = ctx.blocks[appended[k].first];
      size_t tail = lowered.first + lowered.count - 1;
      Instruction *instr = create_instruction(aco_opcode::s_branch, Format::SOPP, 0, 0);
      size_t at = tail;
      if (lowered.split) {
         ctx.segments.emplace(ctx.segments.begin() + tail + 1);
         at = tail + 1;
         for (lowered_block &other : ctx.blocks) {
            if (other.first > tail)
               other.first++;
         }
         ctx.blocks[appended[k].first].count++;
      }
      segment &seg = ctx.segments[at];
      seg.instructions.emplace_back(instr);
      seg.has_target = true;
      seg.target = AGC_ISA_MISS_LABEL;
      seg.branch = true;
      seg.falls = false;
      seg.pc = appended[k].second;
      ctx.blocks[appended[k].first].split = true;
   }
}

static void
lower_program(lower_ctx &ctx, program_state &state, agc_isa_encoded *encoded)
{
   const agc_isa_lowerer *lowerer = ctx.lowerer;
   const agc_isa_ir *ir = ctx.ir;
   agc_isa_regalloc *regs = ctx.regs;
   agc_isa_error error;
   memset(&error, 0, sizeof(error));

   reject_streamout(ctx);
   if (!agc_isa_lowering_pops_check(ir, ctx.target, ctx.stage, &error))
      fail_error(error);
   ctx.mode.dynamic = false;
   ctx.mode.mode = agc_isa_regalloc_initial_float_mode(regs);
   for (size_t i = 0; i < ir->instruction_count && !ctx.mode.dynamic; i++)
      ctx.mode.dynamic = bvh_writes_mode(&ir->instructions[i]);
   if (ctx.mode.dynamic)
      ctx.mode.mode = 0;

   uint32_t software = ctx.ray_unit == AGC_SHADER_RAY_UNIT_HOST ? AGC_ISA_IMG_BVH8 : AGC_ISA_IMG_BVH;
   bool intersects = false;
   for (size_t i = 0; i < ir->instruction_count && !intersects; i++) {
      const agc_isa_inst *inst = &ir->instructions[i];
      if (inst->family != AGC_ISA_MIMG)
         continue;
      const struct agc_isa_image_op *op = agc_isa_image_op(inst->opcode);
      intersects = op && (op->flags & software) != 0;
   }
   uint16_t bvh_sgprs = AGC_ISA_BVH_SCRATCH_SGPRS > AGC_ISA_BVH8_SCRATCH_SGPRS ? AGC_ISA_BVH_SCRATCH_SGPRS
                                                                              : AGC_ISA_BVH8_SCRATCH_SGPRS;
   uint16_t headroom = (uint16_t)(intersects ? spill_headroom_sgprs + bvh_sgprs : spill_headroom_sgprs);
   agc_isa_error discarded;
   bool has_spill_vgpr = false;
   uint8_t spill_vgpr = 0;
   if (ctx.gfx11_plus &&
       agc_isa_regalloc_free_sgprs(regs) < descriptor_slot_sgprs(ctx) + reserved_sgpr_estimate + headroom)
      has_spill_vgpr = agc_isa_regalloc_reserve_program_vgprs(regs, 1, &spill_vgpr, &discarded);
   reserve_descriptor_slots(ctx, has_spill_vgpr);
   if (ctx.has_descriptors && lowerer->preconvert_user_descriptors) {
      if (!agc_isa_lowering_pure_user_descriptors(ir, ctx.wave64, &state.preconverted, &state.preconverted_count))
         fail_memory();
   }
   ctx.preconverted = state.preconverted;
   ctx.preconverted_count = state.preconverted_count;
   ctx.pixel = NULL;
   if (ctx.stage == AGC_SHADER_STAGE_FRAGMENT && ctx.gfx11_plus) {
      state.has_pixel = true;
      if (!agc_isa_pixel_plan_init(&state.pixel, ir, regs, ctx.wave64, &lowerer->pixel, &error))
         fail_error(error);
      ctx.pixel = &state.pixel;
   }
   ctx.has_ngg = false;
   memset(&ctx.ngg, 0, sizeof(ctx.ngg));
   if (ctx.stage == AGC_SHADER_STAGE_VERTEX && !lowerer->hull && ctx.gfx11_plus) {
      ctx.has_ngg = true;
      if (!agc_isa_ngg_plan_init(&ctx.ngg, ir, regs, ctx.wave64, lowerer->has_attribute_ring, lowerer->attribute_ring,
                                 has_spill_vgpr, &error))
         fail_error(error);
   }
   ctx.has_wave_index = false;
   memset(&ctx.wave_index, 0, sizeof(ctx.wave_index));
   if (ctx.stage == AGC_SHADER_STAGE_COMPUTE && ir->metadata.uses_wave_id) {
      uint8_t sgpr;
      ctx.has_wave_index = true;
      if (agc_isa_regalloc_reserve_sgprs(regs, 1, &sgpr)) {
         ctx.wave_index.lane = false;
         ctx.wave_index.index = sgpr;
      } else if (has_spill_vgpr) {
         ctx.wave_index.lane = true;
         ctx.wave_index.index = ctx.wave64 ? 63 : 31;
      } else {
         fail_first(ctx, "the wave index needs an SGPR or a spill VGPR");
      }
   }
   uint16_t spill_below = ctx.gfx11_plus ? headroom : intersects ? bvh_sgprs : 0;
   if (!has_spill_vgpr && agc_isa_regalloc_free_sgprs(regs) < spill_below)
      has_spill_vgpr = agc_isa_regalloc_reserve_program_vgprs(regs, 1, &spill_vgpr, &discarded);
   bool dynamic_jumps = agc_isa_computed_flow_has_dynamic_jumps(&ir->flow);
   ctx.has_dispatch = false;
   ctx.dispatch = 0;
   if (dynamic_jumps) {
      uint8_t count = ir->layout.region_count == 0 ? AGC_ISA_DISPATCH_SGPRS : AGC_ISA_LINKED_DISPATCH_SGPRS;
      ctx.has_dispatch = agc_isa_regalloc_scratch_sgprs(regs, count, &ctx.dispatch) ||
                         agc_isa_regalloc_top_sgprs(regs, count, &ctx.dispatch);
   }
   ctx.layout = ir->layout;
   if (!ctx.layout.has_base) {
      ctx.layout.has_base = lowerer->has_guest_code_address;
      ctx.layout.base = lowerer->has_guest_code_address ? lowerer->guest_code_address : 0;
   }
   if (ctx.layout.own_bytes == 0)
      ctx.layout.own_bytes = 4 * (uint32_t)ir->code_words;
   ctx.has_guest_base = ctx.layout.has_base;
   ctx.guest_base = ctx.layout.base;
   uint8_t wave_offset = 0;
   bool has_wave_offset = agc_isa_regalloc_scratch_wave_offset_sgpr(regs, &wave_offset) && ctx.gfx11_plus &&
                          agc_isa_mask128_test(ir->metadata.live_in_scalar, wave_offset);
   if (has_wave_offset &&
       !agc_isa_lowering_wave_offset_find(ir, ctx.wave64, wave_offset, ctx.target, &state.scratch, &error))
      fail_error(error);
   ctx.private_buffer = indexes_private_segment(ctx, &state.scratch.private_pcs);
   ctx.private_pcs = &state.scratch.private_pcs;
   state.static_reach = statically_reachable(ir);

   ctx.has_gds = lowerer->has_gds_buffer;
   ctx.gds = lowerer->gds_buffer;
   ctx.has_mip_stats = lowerer->has_mip_stats_table;
   ctx.mip_stats = lowerer->mip_stats_table;
   ctx.has_image_reports = lowerer->has_image_reports;
   ctx.image_reports = lowerer->image_reports;
   ctx.has_spill = has_spill_vgpr;
   ctx.spill_vgpr = spill_vgpr;
   ctx.spill_persistent = 0;
   if (has_spill_vgpr) {
      uint8_t persistent = ctx.has_ngg ? ctx.ngg.persistent_lanes : 0;
      uint8_t wave_id = ctx.has_wave_index && ctx.wave_index.lane ? 1 : 0;
      ctx.spill_persistent = persistent > wave_id ? persistent : wave_id;
   }

   begin_context(ctx, true);
   ctx.pc = ir->instruction_count ? ir->instructions[0].pc : 0;
   try {
      lower_prologue(ctx, has_wave_offset && state.scratch.computed, wave_offset);
      flush(ctx);
   } catch (const encode_error &e) {
      ctx.pending.clear();
      note_reject(ctx, e.error);
   }
   bool prologue = false;
   for (const segment &seg : ctx.staged)
      prologue |= !seg.instructions.empty() || seg.has_reject;
   if (prologue)
      commit(ctx, AGC_ISA_PROLOGUE_LABEL, false, SIZE_MAX);

   lower_blocks(ctx, state);
   if (ctx.has_ngg && ctx.ngg.has_deferred && ir->instruction_count > 0) {
      const agc_isa_inst *origin = &ir->instructions[ir->instruction_count - 1];
      const agc_isa_ngg_deferred deferred = ctx.ngg.deferred;
      lower_stub(ctx, AGC_ISA_EPILOGUE_LABEL, origin, [&] {
         ctx.epilogue = false;
         ngg_epilogue(ctx, origin, &deferred);
      });
      lower_stub(ctx, AGC_ISA_EPILOGUE_END_LABEL, origin, [&] {
         ctx.epilogue = false;
         endpgm(ctx);
      });
   }
   if (state.preconverted_count) {
      encoded->preconverted =
         (agc_shader_binding *)malloc(state.preconverted_count * sizeof(*encoded->preconverted));
      if (!encoded->preconverted)
         fail_memory();
      for (size_t i = 0; i < state.preconverted_count; i++) {
         encoded->preconverted[i].kind = (enum agc_shader_resource_kind)state.preconverted[i].kind;
         encoded->preconverted[i].base_sgpr = state.preconverted[i].base_sgpr;
         encoded->preconverted[i].dword_count = state.preconverted[i].dword_count;
         encoded->preconverted[i].writable = state.preconverted[i].writable;
      }
      encoded->preconverted_count = state.preconverted_count;
   }
   if (ctx.has_ngg && ctx.ngg.has_ring) {
      encoded->has_attribute_ring = true;
      encoded->attribute_ring.has_sgpr = !ctx.ngg.ring.descriptor.lane;
      encoded->attribute_ring.sgpr = ctx.ngg.ring.descriptor.lane ? 0 : ctx.ngg.ring.descriptor.index;
      memcpy(encoded->attribute_ring.words, ctx.ngg.ring.words, sizeof(encoded->attribute_ring.words));
   }
   if (ir->instruction_count > 0 && ctx.has_dispatch) {
      const agc_isa_inst *origin = &ir->instructions[0];
      redirect_dropped_blocks(ctx);
      lower_stub(ctx, AGC_ISA_DISPATCH_LABEL, origin, [&] { dispatch_stub(ctx, origin, 4 * (uint32_t)ir->code_words); });
      lower_stub(ctx, AGC_ISA_MISS_LABEL, origin, [&] { guest_endpgm(ctx); });
   } else if (dynamic_jumps) {
      uint32_t jump = ir->flow.jump_count ? ir->flow.jumps[0].pc : 0;
      size_t index;
      if (!agc_isa_ir_instruction_at(ir, jump, &index))
         fail_plain();
      fail(ctx, &ir->instructions[index], "computed jumps need SGPRs above the shader's own");
   }
}

static void
add_edge(Program *program, unsigned from, unsigned to)
{
   program->blocks[from].linear_succs.push_back(to);
   program->blocks[to].linear_preds.push_back(from);
   program->blocks[from].logical_succs.push_back(to);
   program->blocks[to].logical_preds.push_back(from);
}

static void
mark_loops(Program *program)
{
   std::vector<std::pair<unsigned, unsigned>> ranges;
   for (unsigned b = 0; b < program->blocks.size(); b++) {
      for (unsigned s : program->blocks[b].linear_succs) {
         if (s <= b)
            ranges.emplace_back(s, b);
      }
   }
   std::sort(ranges.begin(), ranges.end());
   std::vector<std::pair<unsigned, unsigned>> merged;
   for (auto range : ranges) {
      if (!merged.empty() && range.first <= merged.back().second)
         merged.back().second = std::max(merged.back().second, range.second);
      else
         merged.push_back(range);
   }
   if (!merged.empty() && merged.back().second + 1 == program->blocks.size())
      program->create_and_insert_block();
   for (auto [first, last] : merged) {
      program->blocks[first].kind |= block_kind_loop_header;
      program->blocks[last].kind |= block_kind_loop_latch;
      program->blocks[last + 1].kind |= block_kind_loop_exit;
      for (unsigned b = first; b <= last; b++)
         program->blocks[b].loop_nest_depth = 1;
   }
}

[[noreturn]] static void
reject_at(uint32_t pc, enum agc_isa_error_kind kind, const char *reason)
{
   encode_error e;
   memset(&e.error, 0, sizeof(e.error));
   e.error.kind = kind;
   e.error.pc = pc;
   e.error.reason = reason;
   throw e;
}

static void
build_program(lower_ctx &ctx, unsigned *dispatch_anchor, unsigned *dispatch_load)
{
   Program *program = ctx.program;
   size_t count = ctx.segments.size();
   std::unordered_map<uint32_t, uint32_t> first_label;
   std::unordered_map<uint32_t, uint32_t> last_label;
   std::vector<uint32_t> dynamic;
   for (size_t i = 0; i < count; i++) {
      const segment &seg = ctx.segments[i];
      Block *block = program->create_and_insert_block();
      block->kind = (i == 0) ? block_kind_top_level : 0;
      block->kind |= block_kind_uniform;
      if (seg.has_label) {
         first_label.emplace(seg.label, (uint32_t)i);
         last_label[seg.label] = (uint32_t)i;
      }
      if (seg.dynamic_entry || (seg.has_label && seg.label == AGC_ISA_MISS_LABEL))
         dynamic.push_back((uint32_t)i);
   }

   std::vector<std::vector<uint32_t>> successors(count);
   std::vector<size_t> incoming(count);
   for (size_t i = 0; i < count; i++) {
      const segment &seg = ctx.segments[i];
      if (seg.has_label && seg.label == AGC_ISA_DISPATCH_LABEL) {
         successors[i] = dynamic;
      } else {
         auto target = seg.has_target ? last_label.find(seg.target) : last_label.end();
         if (target != last_label.end())
            successors[i].push_back(target->second);
         if (seg.falls && i + 1 < count && !(successors[i].size() && successors[i][0] == i + 1))
            successors[i].push_back((uint32_t)(i + 1));
      }
      if (successors[i].size() > UINT16_MAX)
         reject_at(seg.label, AGC_ISA_ERROR_INVALID_OPERAND, "the control-flow fanout exceeds ACO's edge capacity");
      for (uint32_t to : successors[i]) {
         if (++incoming[to] > UINT16_MAX)
            reject_at(seg.label, AGC_ISA_ERROR_INVALID_OPERAND,
                      "the control-flow predecessors exceed ACO's edge capacity");
      }
   }
   for (size_t i = 0; i < count; i++) {
      program->blocks[i].linear_succs.reserve(successors[i].size());
      program->blocks[i].logical_succs.reserve(successors[i].size());
      program->blocks[i].linear_preds.reserve(incoming[i]);
      program->blocks[i].logical_preds.reserve(incoming[i]);
   }
   for (size_t i = 0; i < count; i++) {
      for (uint32_t to : successors[i])
         add_edge(program, (unsigned)i, to);
   }

   mark_loops(program);

   for (size_t i = 0; i < count; i++) {
      segment &seg = ctx.segments[i];
      Block *block = &program->blocks[i];
      bool stub = ctx.has_dispatch && seg.has_label && seg.label == AGC_ISA_DISPATCH_LABEL;
      if (seg.has_reject)
         throw encode_error{seg.reject};
      block->instructions.reserve(seg.instructions.size() + (stub ? 2 : 0));
      for (size_t n = 0; n < seg.instructions.size(); n++) {
         Instruction *instr = seg.instructions[n].get();
         if (seg.branch && n + 1 == seg.instructions.size()) {
            auto target = first_label.find(seg.target);
            if (target == first_label.end())
               reject_at(seg.pc, AGC_ISA_ERROR_INVALID_BRANCH, "branch target is not a block");
            instr->salu().imm = target->second;
         } else {
            for (const Operand &o : instr->operands) {
               if (o.physReg() == vcc || o.physReg() == vcc_hi)
                  program->needs_vcc = true;
            }
            for (const Definition &d : instr->definitions) {
               if (d.physReg() == vcc || d.physReg() == vcc_hi)
                  program->needs_vcc = true;
            }
         }
         block->instructions.emplace_back(std::move(seg.instructions[n]));
         if (stub) {
            unsigned *marker = NULL;
            if (instr->opcode == aco_opcode::s_getpc_b64 && *dispatch_anchor == UINT32_MAX)
               marker = dispatch_anchor;
            else if (instr->isSMEM() && *dispatch_load == UINT32_MAX)
               marker = dispatch_load;
            if (marker) {
               *marker = program->debug_info.size();
               program->debug_info.emplace_back();
               Instruction *note = create_instruction(aco_opcode::p_debug_info, Format::PSEUDO, 1, 0);
               note->operands[0] = Operand::c32(*marker);
               block->instructions.emplace_back(note);
            }
         }
      }
   }
   for (Block &block : program->blocks) {
      if (block.linear_succs.empty())
         block.kind |= block_kind_uniform;
   }
}

static void
run_passes(Program *program)
{
   insert_waitcnt(program);
   insert_NOPs(program);
   if (program->gfx_level >= GFX11)
      insert_delay_alu(program);
   if (program->gfx_level >= GFX10)
      form_hard_clauses(program);
   if (program->gfx_level >= GFX11)
      combine_delay_alu(program);
}

static void
emit(lower_ctx &ctx, unsigned dispatch_anchor, unsigned dispatch_load, agc_isa_encoded *encoded)
{
   Program *program = ctx.program;
   for (Block &block : program->blocks) {
      block.kind &=
         ~(block_kind_loop_header | block_kind_loop_latch | block_kind_loop_exit | block_kind_loop_preheader);
      block.loop_nest_depth = 0;
   }
   std::vector<uint32_t> code;
   std::vector<aco_symbol> symbols;
   aco_callback_params params = {};
   emit_program(program, code, symbols, false, &params);

   size_t program_words = code.size();
   if (ctx.has_dispatch && ctx.ir->instruction_count > 0) {
      if (dispatch_anchor == UINT32_MAX || dispatch_load == UINT32_MAX)
         reject_at(AGC_ISA_DISPATCH_LABEL, AGC_ISA_ERROR_INVALID_OPERAND,
                   "the dispatch stub lacks its PC anchor or table load");
      size_t miss = SIZE_MAX;
      for (size_t i = 0; i < ctx.segments.size() && miss == SIZE_MAX; i++) {
         if (ctx.segments[i].has_label && ctx.segments[i].label == AGC_ISA_MISS_LABEL)
            miss = i;
      }
      if (miss == SIZE_MAX)
         reject_at(AGC_ISA_DISPATCH_LABEL, AGC_ISA_ERROR_INVALID_OPERAND, "the dispatch stub has no miss target");
      size_t anchor = program->debug_info[dispatch_anchor].offset / 4;
      size_t load_end = program->debug_info[dispatch_load].offset / 4;
      if (load_end < 2 || load_end > code.size() || anchor > code.size())
         reject_at(AGC_ISA_DISPATCH_LABEL, AGC_ISA_ERROR_INVALID_OPERAND,
                   "the dispatch stub has invalid final offsets");
      int64_t table_offset = ((int64_t)code.size() - anchor) * 4;
      unsigned offset_bits = program->gfx_level >= GFX12 ? 23 : 20;
      if (table_offset < 0 || table_offset >= (1ll << offset_bits))
         reject_at(AGC_ISA_DISPATCH_LABEL, AGC_ISA_ERROR_INVALID_OPERAND,
                   "the dispatch table is outside its load's immediate range");
      uint32_t mask = (1u << offset_bits) - 1;
      code[load_end - 1] = (code[load_end - 1] & ~mask) | (uint32_t)table_offset;
      uint32_t words = (uint32_t)ctx.ir->code_words;
      size_t count = (size_t)words + 1;
      std::vector<uint32_t> offsets(count, program->blocks[miss].offset * 4);
      for (size_t i = 0; i < ctx.segments.size(); i++) {
         const segment &seg = ctx.segments[i];
         if (seg.dynamic_entry && !(seg.label & 3) && seg.label / 4 < words)
            offsets[seg.label / 4] = program->blocks[i].offset * 4;
      }
      for (uint32_t offset : offsets) {
         int64_t relative = (int64_t)offset - (int64_t)anchor * 4;
         if (relative < INT32_MIN || relative > INT32_MAX)
            reject_at(AGC_ISA_DISPATCH_LABEL, AGC_ISA_ERROR_INVALID_OPERAND,
                      "a dispatch target is outside the signed displacement range");
         code.push_back((uint32_t)relative);
      }
   }
   unsigned line = agc_isa_target_instruction_cache_line_bytes(ctx.target) / 4;
   while (code.size() % line != 0)
      code.push_back(AGC_ISA_S_CODE_END);
   for (unsigned i = 0; i < AGC_ISA_PREFETCH_PADDING_LINES * line; i++)
      code.push_back(AGC_ISA_S_CODE_END);

   encoded->code = (uint32_t *)malloc(code.size() * sizeof(uint32_t));
   if (!encoded->code && !code.empty())
      fail_memory();
   memcpy(encoded->code, code.data(), code.size() * sizeof(uint32_t));
   encoded->code_count = code.size();
   encoded->program_words = program_words;
}

}

extern "C" void
agc_isa_encoded_init(struct agc_isa_encoded *encoded)
{
   memset(encoded, 0, sizeof(*encoded));
}

extern "C" void
agc_isa_encoded_finish(struct agc_isa_encoded *encoded)
{
   free(encoded->code);
   free(encoded->preconverted);
   memset(encoded, 0, sizeof(*encoded));
}

namespace agc {

static void
print_label(FILE *out, uint32_t label)
{
   switch (label) {
   case AGC_ISA_PROLOGUE_LABEL:
      fputs("prologue", out);
      break;
   case AGC_ISA_DISPATCH_LABEL:
      fputs("dispatch", out);
      break;
   case AGC_ISA_MISS_LABEL:
      fputs("miss", out);
      break;
   case AGC_ISA_EPILOGUE_LABEL:
      fputs("epilogue", out);
      break;
   case AGC_ISA_EPILOGUE_END_LABEL:
      fputs("end", out);
      break;
   default:
      if (label >= AGC_ISA_LOCAL_LABEL_BASE)
         fprintf(out, "local_%x", label - AGC_ISA_LOCAL_LABEL_BASE);
      else
         fprintf(out, "pc_%x", label);
      break;
   }
}

static char *
listing_text(const lower_ctx &ctx)
{
   const Program *program = ctx.program;
   char *text = NULL;
   size_t size = 0;
   u_memstream mem;
   if (!u_memstream_open(&mem, &text, &size))
      fail_memory();
   FILE *out = u_memstream_get(&mem);
   for (size_t i = 0; i < ctx.segments.size(); i++) {
      const segment &seg = ctx.segments[i];
      if (seg.has_label)
         print_label(out, seg.label);
      fputs(seg.dynamic_entry ? ": entry\n" : ":\n", out);
      const std::vector<aco_ptr<Instruction>> &instructions = program->blocks[i].instructions;
      for (size_t n = 0; n < instructions.size(); n++) {
         const Instruction *instr = instructions[n].get();
         if (instr->opcode == aco_opcode::p_debug_info)
            continue;
         fputs("   ", out);
         if (seg.branch && n + 1 == instructions.size()) {
            fprintf(out, "%s -> ", instr_info.name[(int)instr->opcode]);
            print_label(out, seg.target);
         } else {
            aco_print_instr(program->gfx_level, instr, out, print_no_ssa);
            if (instr->isVOP3() && (instr->isVOP1() || instr->isVOP2() || instr->isVOPC()))
               fputs(" vop3", out);
         }
         fputc('\n', out);
      }
   }
   u_memstream_close(&mem);
   if (!text)
      text = (char *)calloc(1, 1);
   if (!text)
      fail_memory();
   return text;
}

static bool
run(const agc_isa_lowerer *lowerer, const agc_isa_ir *ir, agc_isa_regalloc *regs, agc_isa_encoded *encoded,
    char **listing, agc_isa_error *error)
{
   agc_isa_encoded_init(encoded);
   memset(error, 0, sizeof(*error));

   enum agc_isa_target target = agc_isa_target_from_shader_isa(lowerer->options.target);
   enum amd_gfx_level level;
   enum radeon_family family;
   if (!target_to_gfx(target, &level, &family)) {
      error->kind = AGC_ISA_ERROR_INVALID_OPTIONS;
      error->reason = "ACO back-end targets gfx10.3, gfx11, gfx11.5 and gfx12 only";
      return false;
   }
   bool wave64 = lowerer->options.wave_size == 64;
   unsigned wave_size = wave64 ? 64 : 32;

   try {
      init();

      radeon_info rad_info;
      memset(&rad_info, 0, sizeof(rad_info));
      rad_info.gfx_level = level;
      rad_info.family = family;
      ac_fill_compiler_info(&rad_info, NULL, false);

      ac_shader_config config;
      memset(&config, 0, sizeof(config));

      aco_shader_info info;
      memset(&info, 0, sizeof(info));
      info.wave_size = wave_size;
      Stage stage = lowerer->options.stage == AGC_SHADER_STAGE_FRAGMENT ? fragment_fs
                    : lowerer->options.stage == AGC_SHADER_STAGE_VERTEX
                         ? (lowerer->hull                                  ? vertex_tess_control_hs
                            : lowerer->es_input == AGC_SHADER_ES_DOMAIN ? tess_eval_geometry_ngg
                                                                         : vertex_geometry_ngg)
                         : compute_cs;
      info.hw_stage = stage.hw;
      info.workgroup_size = wave_size;

      aco_compiler_options options;
      memset(&options, 0, sizeof(options));
      options.compiler_info = &rad_info.compiler_info;
      options.family = family;
      options.gfx_level = level;

      Program program;
      init_program(&program, stage, &info, &options, &config);
      program.progress = CompilationProgress::after_ra;
      program.workgroup_size = wave_size;

      lower_ctx ctx(&program);
      ctx.lowerer = lowerer;
      ctx.ir = ir;
      ctx.regs = regs;
      ctx.target = target;
      ctx.stage = lowerer->options.stage;
      ctx.precision = lowerer->precision;
      ctx.ray_unit = lowerer->options.ray_unit;
      ctx.wave64 = wave64;
      ctx.gfx11_plus = agc_isa_target_is_gfx11_plus(target);
      ctx.gfx12 = agc_isa_target_is_gfx12(target);
      ctx.pc = 0;
      ctx.split = false;
      ctx.has_next_label = false;

      program_state state;
      lower_program(ctx, state, encoded);
      if (ctx.segments.empty()) {
         agc_isa_encoded_finish(encoded);
         error->kind = AGC_ISA_ERROR_EMPTY_PROGRAM;
         return false;
      }
      unsigned dispatch_anchor = UINT32_MAX;
      unsigned dispatch_load = UINT32_MAX;
      build_program(ctx, &dispatch_anchor, &dispatch_load);
      if (listing) {
         *listing = listing_text(ctx);
         return true;
      }
      calc_min_waves(&program);
      run_passes(&program);
      emit(ctx, dispatch_anchor, dispatch_load, encoded);
      return true;
   } catch (const lower_error &e) {
      agc_isa_encoded_finish(encoded);
      *error = e.error;
      return false;
   } catch (const encode_error &e) {
      agc_isa_encoded_finish(encoded);
      *error = e.error;
      return false;
   } catch (const std::bad_alloc &) {
      agc_isa_encoded_finish(encoded);
      memset(error, 0, sizeof(*error));
      error->kind = AGC_ISA_ERROR_OUT_OF_MEMORY;
      return false;
   }
}

}

extern "C" bool
agc_isa_lower_encode(const struct agc_isa_lowerer *lowerer, const struct agc_isa_ir *ir, struct agc_isa_regalloc *regs,
                     struct agc_isa_encoded *encoded, struct agc_isa_error *error)
{
   return agc::run(lowerer, ir, regs, encoded, NULL, error);
}

extern "C" bool
agc_isa_lower_listing(const struct agc_isa_lowerer *lowerer, const struct agc_isa_ir *ir, struct agc_isa_regalloc *regs,
                      char **listing, struct agc_isa_error *error)
{
   struct agc_isa_encoded encoded;
   *listing = NULL;
   bool ok = agc::run(lowerer, ir, regs, &encoded, listing, error);
   agc_isa_encoded_finish(&encoded);
   return ok;
}
