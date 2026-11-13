#include "agc_isa_analyze_priv.h"

#define SETTLE_ROUNDS 8
#define EXP_MRT_LAST 7
#define EXP_MRTZ 8
#define EXP_POS_FIRST 12
#define EXP_POS_LAST 16
#define EXP_PRIM 20
#define EXP_PARAM_FIRST 32

struct agc_isa_linked_code {
   const uint32_t *words;
   size_t word_count;
   uint32_t *storage;
   struct agc_isa_code_layout layout;
};

void
agc_isa_metadata_finish(struct agc_isa_metadata *metadata)
{
   free(metadata->bindings);
   free(metadata->descriptor_uses);
   free(metadata->pointer_uses);
   free(metadata->storage);
   memset(metadata, 0, sizeof(*metadata));
}

void
agc_isa_code_layout_finish(struct agc_isa_code_layout *layout)
{
   free(layout->regions);
   memset(layout, 0, sizeof(*layout));
}

static bool
agc_isa_code_layout_copy(struct agc_isa_code_layout *dst, const struct agc_isa_code_layout *src)
{
   struct agc_isa_linked_region *regions = NULL;
   if (src->region_count) {
      regions = malloc(src->region_count * sizeof(*regions));
      if (!regions)
         return false;
      memcpy(regions, src->regions, src->region_count * sizeof(*regions));
   }
   struct agc_isa_code_layout copy = *src;
   copy.regions = regions;
   agc_isa_code_layout_finish(dst);
   *dst = copy;
   return true;
}

uint64_t
agc_isa_code_layout_guest_address(const struct agc_isa_code_layout *layout, uint32_t pc)
{
   for (size_t i = 0; i < layout->region_count; i++) {
      const struct agc_isa_linked_region *region = &layout->regions[i];
      if (pc >= region->start && pc < region->start + region->bytes)
         return region->address + (pc - region->start);
   }
   return (layout->has_base ? layout->base : 0) + pc;
}

bool
agc_isa_code_layout_pc_of(const struct agc_isa_code_layout *layout, uint64_t address, uint32_t *pc)
{
   uint64_t base = layout->has_base ? layout->base : 0;
   if (address >= base && address - base < layout->own_bytes) {
      *pc = (uint32_t)(address - base);
      return true;
   }
   for (size_t i = 0; i < layout->region_count; i++) {
      const struct agc_isa_linked_region *region = &layout->regions[i];
      if (address >= region->address && address - region->address < region->bytes) {
         *pc = region->start + (uint32_t)(address - region->address);
         return true;
      }
   }
   return false;
}

static void
agc_isa_linked_code_finish(struct agc_isa_linked_code *linked)
{
   free(linked->storage);
   agc_isa_code_layout_finish(&linked->layout);
   memset(linked, 0, sizeof(*linked));
}

struct span {
   uint64_t address;
   uint64_t bytes;
};

static bool
link_regions(const uint32_t *words, size_t word_count, uint64_t base, const struct agc_shader_linkage *linkage,
             struct agc_isa_linked_code *linked, struct span *spans, struct agc_isa_error *error)
{
   struct agc_isa_code_layout *layout = &linked->layout;
   size_t total = word_count;
   spans[0].address = base;
   spans[0].bytes = 4 * (uint64_t)word_count;
   for (uint32_t i = 0; i < linkage->region_count; i++) {
      const struct agc_shader_code_region *region = &linkage->regions[i];
      uint64_t bytes = 4 * (uint64_t)region->word_count;
      if (region->word_count == 0 || region->address % 4 != 0 || region->address + bytes < region->address)
         return agc_isa_fail_options(error, "a linked region must be non-empty, dword aligned words");
      layout->regions[i].address = region->address;
      layout->regions[i].start = 4 * (uint32_t)total;
      layout->regions[i].bytes = (uint32_t)bytes;
      layout->region_count = i + 1;
      total += region->word_count;
      spans[i + 1].address = region->address;
      spans[i + 1].bytes = bytes;
   }
   for (uint32_t k = 0; k <= linkage->region_count; k++) {
      uint64_t address = spans[k].address;
      uint64_t bytes = spans[k].bytes;
      if (address >> 32 != (address + bytes - 1) >> 32)
         return agc_isa_fail_options(error, "linked code must not cross a 4 GiB boundary");
      for (uint32_t j = 0; j < k; j++) {
         if (address < spans[j].address + spans[j].bytes && spans[j].address < address + bytes)
            return agc_isa_fail_options(error, "linked code regions overlap");
      }
   }
   if (linkage->has_second_half) {
      bool half_linked = false;
      for (size_t i = 0; i < layout->region_count; i++)
         half_linked |= layout->regions[i].address == linkage->second_half;
      if (!half_linked)
         return agc_isa_fail_options(error, "Linkage::second_half must be the address of a linked region");
   }
   linked->storage = malloc((total + 1) * sizeof(*linked->storage));
   if (!linked->storage)
      return agc_isa_fail_memory(error);
   if (word_count)
      memcpy(linked->storage, words, word_count * sizeof(*words));
   size_t at = word_count;
   for (uint32_t i = 0; i < linkage->region_count; i++) {
      memcpy(linked->storage + at, linkage->regions[i].words, linkage->regions[i].word_count * sizeof(*words));
      at += linkage->regions[i].word_count;
   }
   linked->words = linked->storage;
   linked->word_count = total;
   return true;
}

static bool
agc_isa_link_layout(const uint32_t *words, size_t word_count, const struct agc_shader_interface *iface,
                    const struct agc_shader_linkage *linkage, struct agc_isa_linked_code *linked,
                    struct agc_isa_error *error)
{
   static const struct agc_shader_linkage no_linkage;
   if (!linkage)
      linkage = &no_linkage;
   memset(linked, 0, sizeof(*linked));
   struct agc_isa_code_layout *layout = &linked->layout;
   layout->has_base = iface && iface->has_guest_code_address;
   layout->base = layout->has_base ? iface->guest_code_address : 0;
   layout->own_bytes = (uint32_t)(4 * (uint64_t)word_count);
   layout->has_second_half = linkage->has_second_half;
   layout->second_half = linkage->has_second_half ? linkage->second_half : 0;
   if (linkage->region_count == 0 && !linkage->has_second_half) {
      linked->words = words;
      linked->word_count = word_count;
      return true;
   }
   if (!layout->has_base)
      return agc_isa_fail_options(error, "linked code needs ShaderInterface::guest_code_address");
   struct span *spans = malloc(((size_t)linkage->region_count + 1) * sizeof(*spans));
   layout->regions = calloc((size_t)linkage->region_count + 1, sizeof(*layout->regions));
   bool ok = spans && layout->regions;
   if (!ok)
      agc_isa_fail_memory(error);
   ok = ok && link_regions(words, word_count, layout->base, linkage, linked, spans, error);
   free(spans);
   if (!ok)
      agc_isa_linked_code_finish(linked);
   return ok;
}

static int
compare_u32(const void *a, const void *b)
{
   uint32_t x = *(const uint32_t *)a;
   uint32_t y = *(const uint32_t *)b;
   return x < y ? -1 : x > y;
}

static bool
settle_scope(const struct agc_isa_computed_flow *flow, const struct agc_isa_code_layout *layout,
             uint32_t **entries, struct agc_isa_decode_scope *scope)
{
   uint32_t *from_flow;
   size_t count;
   bool through_jumps;
   if (!agc_isa_computed_decode_scope(flow, &from_flow, &count, &through_jumps))
      return false;
   uint32_t *merged = realloc(from_flow, (count + layout->region_count + 1) * sizeof(*merged));
   if (!merged) {
      free(from_flow);
      return false;
   }
   for (size_t i = 0; i < layout->region_count; i++)
      merged[count++] = layout->regions[i].start;
   qsort(merged, count, sizeof(*merged), compare_u32);
   size_t kept = 0;
   for (size_t i = 0; i < count; i++) {
      if (!kept || merged[kept - 1] != merged[i])
         merged[kept++] = merged[i];
   }
   *entries = merged;
   scope->entries = merged;
   scope->entry_count = kept;
   scope->through_jumps = through_jumps;
   return true;
}

static bool
settle_round(const uint32_t *words, size_t word_count, const struct agc_isa_code_layout *layout,
             const struct agc_isa_computed_flow *flow, struct agc_isa_computed_flow *next, struct agc_isa_ir *ir,
             struct agc_isa_error *error)
{
   uint32_t *entries;
   struct agc_isa_decode_scope scope;
   agc_isa_ir_free_analysis(ir);
   if (!settle_scope(flow, layout, &entries, &scope))
      return agc_isa_fail_memory(error);
   bool ok = agc_isa_decode_scoped(words, word_count, &scope, ir, error);
   free(entries);
   if (!ok)
      return false;
   if (!agc_isa_code_layout_copy(&ir->layout, layout))
      return agc_isa_fail_memory(error);
   return agc_isa_computed_apply(ir, flow, error) && agc_isa_cfg_build(ir, error) &&
          agc_isa_computed_analyze(ir, layout->has_base, layout->base, word_count, next, error);
}

static bool
agc_isa_decode_program(const uint32_t *words, size_t word_count, const struct agc_isa_code_layout *layout,
                       struct agc_isa_ir *ir, struct agc_isa_error *error)
{
   struct agc_isa_code_layout own = {0};
   struct agc_isa_computed_flow flow = {0};
   struct agc_isa_computed_flow next = {0};
   uint32_t last_jump = 0;
   bool ok = agc_isa_code_layout_copy(&own, layout);
   bool settled = false;
   if (!ok)
      agc_isa_fail_memory(error);
   for (unsigned round = 0; ok && !settled && round < SETTLE_ROUNDS; round++) {
      ok = settle_round(words, word_count, &own, &flow, &next, ir, error);
      if (!ok)
         break;
      settled = agc_isa_computed_flow_equal(&next, &flow);
      if (settled)
         break;
      last_jump = next.jump_count ? next.jumps[next.jump_count - 1].pc : 0;
      struct agc_isa_computed_flow previous = flow;
      flow = next;
      next = previous;
   }
   if (ok && !settled)
      ok = agc_isa_fail_operand(error, last_jump, "computed jump targets do not settle");
   agc_isa_computed_flow_finish(&flow);
   agc_isa_computed_flow_finish(&next);
   agc_isa_code_layout_finish(&own);
   if (!ok) {
      agc_isa_ir_clear(ir);
      agc_isa_ir_free_analysis(ir);
   }
   return ok;
}

struct binding_list {
   struct agc_isa_resource_binding *bindings;
   size_t count;
   size_t capacity;
   bool failed;
};

static void
bind(struct binding_list *list, enum agc_isa_resource_kind kind, uint8_t base, uint8_t dword_count, bool writable)
{
   for (size_t i = 0; i < list->count; i++) {
      struct agc_isa_resource_binding *binding = &list->bindings[i];
      if (binding->kind == kind && binding->base_sgpr == base && binding->dword_count == dword_count) {
         binding->writable |= writable;
         return;
      }
   }
   if (list->count == list->capacity) {
      size_t capacity = list->capacity ? list->capacity * 2 : 16;
      struct agc_isa_resource_binding *bindings = realloc(list->bindings, capacity * sizeof(*bindings));
      if (!bindings) {
         list->failed = true;
         return;
      }
      list->bindings = bindings;
      list->capacity = capacity;
   }
   struct agc_isa_resource_binding *binding = &list->bindings[list->count++];
   binding->kind = (uint8_t)kind;
   binding->base_sgpr = base;
   binding->dword_count = dword_count;
   binding->writable = writable;
}

static void
sort_bindings(struct binding_list *list)
{
   for (size_t i = 1; i < list->count; i++) {
      struct agc_isa_resource_binding moved = list->bindings[i];
      size_t at = i;
      while (at > 0) {
         const struct agc_isa_resource_binding *before = &list->bindings[at - 1];
         if (before->base_sgpr < moved.base_sgpr ||
             (before->base_sgpr == moved.base_sgpr && before->kind <= moved.kind))
            break;
         list->bindings[at] = *before;
         at--;
      }
      list->bindings[at] = moved;
   }
}

static void
record_export(struct agc_isa_export_info *exports, const struct agc_isa_export *e)
{
   exports->compressed |= e->compr;
   exports->valid_mask |= e->vm;
   if (e->target <= EXP_MRT_LAST)
      exports->color_mask |= (uint32_t)(e->en & 0xf) << (4 * e->target);
   else if (e->target == EXP_MRTZ)
      exports->depth_mask |= e->en & 0xf;
   else if (e->target >= EXP_POS_FIRST && e->target <= EXP_POS_LAST)
      exports->position_mask |= (uint8_t)(1u << (e->target - EXP_POS_FIRST));
   else if (e->target == EXP_PRIM)
      exports->primitive = true;
   else if (e->target >= EXP_PARAM_FIRST)
      exports->param_mask |= UINT32_C(1) << ((e->target - EXP_PARAM_FIRST) & 31);
}

struct usage {
   bool has_max_sgpr;
   bool has_max_vgpr;
   uint16_t max_sgpr;
   uint16_t max_vgpr;
};

static void
track(struct usage *usage, const struct agc_isa_reg_range *ranges, unsigned count)
{
   for (unsigned r = 0; r < count; r++) {
      uint16_t last = agc_isa_reg_range_last(ranges[r]);
      if (ranges[r].file == AGC_ISA_REG_VECTOR) {
         if (last > AGC_ISA_VGPR_LIMIT - 1)
            last = AGC_ISA_VGPR_LIMIT - 1;
         if (!usage->has_max_vgpr || last > usage->max_vgpr)
            usage->max_vgpr = last;
         usage->has_max_vgpr = true;
      } else if (ranges[r].first < AGC_ISA_SGPR_LIMIT) {
         if (last > AGC_ISA_SGPR_LIMIT - 1)
            last = AGC_ISA_SGPR_LIMIT - 1;
         if (!usage->has_max_sgpr || last > usage->max_sgpr)
            usage->max_sgpr = last;
         usage->has_max_sgpr = true;
      }
   }
}

static void
scan_instruction(const struct agc_isa_inst *inst, struct agc_isa_metadata *meta, struct binding_list *bindings)
{
   bool relative = agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_RELATIVE);
   meta->uses_pc_relative |= agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_PC_RELATIVE);
   meta->uses_relative_indexing |= relative;
   meta->uses_relative_sgprs |= relative && agc_isa_family_is_scalar_alu((enum agc_isa_family)inst->family);
   switch (inst->fields) {
   case AGC_ISA_SMEM: {
      const struct agc_isa_smem *s = &inst->smem;
      if (s->dwords > 0 && !agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_SRC)) {
         if (s->buffer)
            bind(bindings, AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR, s->sbase, 4, false);
         else
            bind(bindings, AGC_ISA_RESOURCE_SCALAR_MEMORY_PAIR, s->sbase, 2, false);
         meta->uses_scratch |= inst->opcode >= AGC_ISA_SMEM_SCRATCH_LOAD_DWORD &&
                               inst->opcode <= AGC_ISA_SMEM_SCRATCH_LOAD_DWORDX4;
      } else {
         meta->uses_wave_id |= inst->opcode == AGC_ISA_SMEM_GET_WAVEID_IN_WORKGROUP;
      }
      break;
   }
   case AGC_ISA_MUBUF:
   case AGC_ISA_MTBUF: {
      const struct agc_isa_buffer *b = &inst->buffer;
      if (b->kind == AGC_ISA_MEM_CONTROL)
         break;
      bind(bindings, AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR, b->srsrc, 4, b->kind != AGC_ISA_MEM_LOAD);
      break;
   }
   case AGC_ISA_MIMG: {
      const struct agc_isa_image *m = &inst->mimg;
      bind(bindings, AGC_ISA_RESOURCE_IMAGE_DESCRIPTOR, m->srsrc, m->r128 ? 4 : 8, m->kind != AGC_ISA_MEM_LOAD);
      if (m->uses_sampler)
         bind(bindings, AGC_ISA_RESOURCE_SAMPLER_DESCRIPTOR, m->ssamp, 4, false);
      break;
   }
   case AGC_ISA_FLAT: {
      const struct agc_isa_flat *f = &inst->flat;
      if (f->saddr.kind == AGC_ISA_OPERAND_SGPR && f->segment == AGC_ISA_SEGMENT_GLOBAL)
         bind(bindings, AGC_ISA_RESOURCE_SCALAR_MEMORY_PAIR, (uint8_t)f->saddr.reg, 2, f->kind != AGC_ISA_MEM_LOAD);
      meta->uses_scratch |= f->segment == AGC_ISA_SEGMENT_SCRATCH;
      break;
   }
   case AGC_ISA_VOP3:
      for (unsigned i = 0; i < 3; i++)
         meta->uses_scratch |= inst->vop3.src[i].kind == AGC_ISA_OPERAND_PRIVATE_BASE ||
                               inst->vop3.src[i].kind == AGC_ISA_OPERAND_PRIVATE_LIMIT;
      break;
   case AGC_ISA_EXP:
      record_export(&meta->exports, &inst->exp);
      break;
   default:
      break;
   }
}

struct summary {
   struct agc_isa_mask128 uses[2];
   struct agc_isa_mask128 defs[2];
   struct agc_isa_mask128 live_in[2];
};

static bool
live_in(const struct agc_isa_ir *ir, bool wave64, struct agc_isa_metadata *meta)
{
   meta->live_in_scalar = agc_isa_mask128_bit(128);
   meta->live_in_vector = agc_isa_mask128_bit(128);
   if (!ir->block_count)
      return true;
   struct summary *summaries = calloc(ir->block_count, sizeof(*summaries));
   if (!summaries)
      return false;
   for (size_t id = 0; id < ir->block_count; id++) {
      const struct agc_isa_block *block = &ir->blocks[id];
      struct summary *summary = &summaries[id];
      for (size_t i = block->first; i < block->end; i++) {
         struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
         unsigned count = agc_isa_inst_uses(&ir->instructions[i], wave64, ranges);
         for (unsigned r = 0; r < count; r++) {
            unsigned file = ranges[r].file == AGC_ISA_REG_VECTOR;
            struct agc_isa_mask128 mask = agc_isa_mask128_range(ranges[r].first, ranges[r].count);
            summary->uses[file] =
               agc_isa_mask128_or(summary->uses[file], agc_isa_mask128_andnot(mask, summary->defs[file]));
         }
         count = agc_isa_inst_defs(&ir->instructions[i], wave64, ranges);
         for (unsigned r = 0; r < count; r++) {
            unsigned file = ranges[r].file == AGC_ISA_REG_VECTOR;
            summary->defs[file] =
               agc_isa_mask128_or(summary->defs[file], agc_isa_mask128_range(ranges[r].first, ranges[r].count));
         }
      }
   }
   bool changed = true;
   while (changed) {
      changed = false;
      for (size_t id = ir->block_count; id-- > 0;) {
         const struct agc_isa_block *block = &ir->blocks[id];
         struct summary *summary = &summaries[id];
         for (unsigned file = 0; file < 2; file++) {
            struct agc_isa_mask128 live_out = agc_isa_mask128_bit(128);
            for (uint32_t s = 0; s < block->successor_count; s++)
               live_out = agc_isa_mask128_or(live_out, summaries[block->successors[s]].live_in[file]);
            struct agc_isa_mask128 value =
               agc_isa_mask128_or(summary->uses[file], agc_isa_mask128_andnot(live_out, summary->defs[file]));
            if (!agc_isa_mask128_equal(value, summary->live_in[file])) {
               summary->live_in[file] = value;
               changed = true;
            }
         }
      }
   }
   meta->live_in_scalar = summaries[0].live_in[0];
   meta->live_in_vector = summaries[0].live_in[1];
   free(summaries);
   return true;
}

static uint8_t
user_sgpr_count(const struct agc_isa_metadata *meta, const struct agc_shader_options *options,
                const struct agc_shader_interface *iface)
{
   bool compute = options->stage == AGC_SHADER_STAGE_COMPUTE;
   unsigned limit = compute ? AGC_ISA_MAX_COMPUTE_USER_SGPRS : AGC_ISA_MAX_GRAPHICS_USER_SGPRS;
   unsigned count;
   if (iface->has_user_sgpr_count) {
      count = iface->user_sgpr_count;
   } else if (iface->has_guest_pgm_rsrc2) {
      uint32_t rsrc2 = iface->guest_pgm_rsrc2;
      count = (rsrc2 & AGC_ISA_RSRC2_USER_SGPR_MASK) >> AGC_ISA_RSRC2_USER_SGPR_SHIFT;
      if (!compute && (rsrc2 & AGC_ISA_RSRC2_USER_SGPR_MSB))
         count += 32;
   } else {
      count = agc_isa_mask128_length(agc_isa_mask128_low(meta->live_in_scalar, AGC_ISA_SGPR_LIMIT));
   }
   return (uint8_t)(count < limit ? count : limit);
}

static bool
agc_isa_analyze_metadata(struct agc_isa_ir *ir, const struct agc_shader_options *options,
                         const struct agc_shader_interface *iface, struct agc_isa_error *error)
{
   static const struct agc_shader_interface no_interface;
   if (!iface)
      iface = &no_interface;
   bool wave64 = options->wave_size == 64;
   struct agc_isa_metadata meta;
   memset(&meta, 0, sizeof(meta));
   meta.has_user_data_tail_sgpr = ir->metadata.has_user_data_tail_sgpr;
   meta.user_data_tail_sgpr = ir->metadata.user_data_tail_sgpr;
   meta.stage = (uint8_t)options->stage;
   struct usage usage = {0};
   struct binding_list bindings = {0};

   for (size_t id = 0; id < ir->block_count; id++) {
      const struct agc_isa_block *block = &ir->blocks[id];
      if (!block->reachable)
         continue;
      for (size_t i = block->first; i < block->end; i++) {
         const struct agc_isa_inst *inst = &ir->instructions[i];
         struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
         track(&usage, ranges, agc_isa_inst_defs(inst, wave64, ranges));
         track(&usage, ranges, agc_isa_inst_uses(inst, wave64, ranges));
         scan_instruction(inst, &meta, &bindings);
      }
   }

   sort_bindings(&bindings);
   meta.has_max_sgpr = usage.has_max_sgpr;
   meta.max_sgpr = (uint8_t)usage.max_sgpr;
   meta.has_max_vgpr = usage.has_max_vgpr;
   meta.max_vgpr = (uint8_t)usage.max_vgpr;
   meta.sgpr_count = usage.has_max_sgpr ? (uint16_t)(usage.max_sgpr + 1) : 0;
   meta.vgpr_count = usage.has_max_vgpr ? (uint16_t)(usage.max_vgpr + 1) : 0;
   struct agc_isa_uses uses = {0};
   bool ok = !bindings.failed && live_in(ir, wave64, &meta);
   if (ok) {
      meta.user_sgpr_count = user_sgpr_count(&meta, options, iface);
      uint16_t user_start = 0;
      uint16_t user_end = meta.user_sgpr_count;
      if (options->stage == AGC_SHADER_STAGE_VERTEX) {
         user_start = AGC_ISA_NGG_SYSTEM_SGPRS;
         user_end = AGC_ISA_NGG_SYSTEM_SGPRS + AGC_ISA_MAX_GRAPHICS_USER_SGPRS;
      }
      bool has_group_x = iface->has_guest_pgm_rsrc2 && options->stage == AGC_SHADER_STAGE_COMPUTE &&
                         (iface->guest_pgm_rsrc2 & AGC_ISA_RSRC2_TGID_X_EN) != 0 &&
                         (iface->guest_pgm_rsrc2 & AGC_ISA_RSRC2_SCRATCH_EN) == 0;
      ok = agc_isa_sources_descriptor_uses(ir, wave64, user_start, user_end, has_group_x, meta.user_sgpr_count,
                                           &uses);
   }
   if (!ok) {
      free(bindings.bindings);
      return agc_isa_fail_memory(error);
   }
   meta.bindings = bindings.bindings;
   meta.binding_count = bindings.count;
   meta.descriptor_uses = uses.descriptors;
   meta.descriptor_use_count = uses.descriptor_count;
   meta.pointer_uses = uses.pointers;
   meta.pointer_use_count = uses.pointer_count;
   meta.has_buffer_fill = uses.has_fill;
   meta.buffer_fill = uses.fill;
   meta.storage = uses.storage;
   if (iface->has_guest_pgm_rsrc2)
      meta.uses_scratch |= (iface->guest_pgm_rsrc2 & AGC_ISA_RSRC2_SCRATCH_EN) != 0;
   meta.uses_scratch |= iface->scratch_bytes_per_lane > 0;
   agc_isa_metadata_finish(&ir->metadata);
   ir->metadata = meta;
   return true;
}

bool
agc_isa_analyze_linked(enum agc_isa_target compiler, const uint32_t *words, size_t word_count,
                       const struct agc_shader_options *options, const struct agc_shader_interface *iface,
                       const struct agc_shader_linkage *linkage, struct agc_isa_ir *ir, struct agc_isa_error *error)
{
   static const struct agc_shader_interface no_interface;
   if (!iface)
      iface = &no_interface;
   agc_isa_ir_clear(ir);
   agc_isa_ir_free_analysis(ir);
   if (options->wave_size != 32 && options->wave_size != 64)
      return agc_isa_fail_options(error, "wave_size must be 32 or 64");
   enum agc_isa_target requested = agc_isa_target_from_shader_isa(options->target);
   if (requested != compiler) {
      agc_isa_fail(error, AGC_ISA_ERROR_TARGET_MISMATCH, 0);
      error->compiler = (uint8_t)compiler;
      error->requested = (uint8_t)requested;
      return false;
   }
   struct agc_isa_linked_code linked;
   if (!agc_isa_link_layout(words, word_count, iface, linkage, &linked, error))
      return false;
   bool ok = agc_isa_decode_program(linked.words, linked.word_count, &linked.layout, ir, error);
   agc_isa_linked_code_finish(&linked);
   ok = ok && agc_isa_analyze_metadata(ir, options, iface, error);
   if (!ok) {
      agc_isa_ir_clear(ir);
      agc_isa_ir_free_analysis(ir);
   }
   return ok;
}

bool
agc_isa_analyze(enum agc_isa_target compiler, const uint32_t *words, size_t word_count,
                const struct agc_shader_options *options, const struct agc_shader_interface *iface,
                struct agc_isa_ir *ir, struct agc_isa_error *error)
{
   return agc_isa_analyze_linked(compiler, words, word_count, options, iface, NULL, ir, error);
}
