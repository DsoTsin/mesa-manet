#include <string.h>

#include "agc_lower.h"

namespace agc {

static const uint16_t block_align = 4;

[[noreturn]] static void
no_spill(const lower_ctx &ctx, const agc_isa_inst *origin)
{
   fail(ctx, origin, "the translation needs SGPRs above the shader's own");
}

static bool
overlaps(const agc_isa_reg_range *touched, unsigned touched_count, uint16_t owned_start, uint16_t owned_end,
         const agc_isa_spill_lanes *spill, uint16_t base, uint16_t size)
{
   uint16_t end = (uint16_t)(base + size);
   for (unsigned i = 0; i < touched_count; i++) {
      uint16_t count = touched[i].count ? touched[i].count : 1;
      if (touched[i].first < end && base < touched[i].first + count)
         return true;
   }
   if (owned_start < end && base < owned_end)
      return true;
   for (unsigned i = 0; i < spill->saved_count; i++) {
      const agc_isa_spill_block *block = &spill->saved[i];
      if (block->base < end && base < block->base + block->size)
         return true;
   }
   return false;
}

static unsigned
scalar_ranges(const agc_isa_reg_range *ranges, unsigned count, agc_isa_reg_range *out, unsigned kept)
{
   for (unsigned i = 0; i < count; i++) {
      if (ranges[i].file == AGC_ISA_REG_SCALAR)
         out[kept++] = ranges[i];
   }
   return kept;
}

static void
write_lane_raw(lower_ctx &ctx, unsigned vgpr, Operand value, unsigned lane)
{
   ctx.bld.vop3(aco_opcode::v_writelane_b32_e64, vdef(vgpr), value, imm(lane), Operand(PhysReg{0}, s1));
}

uint8_t
spill_block(lower_ctx &ctx, const agc_isa_inst *origin, uint8_t count)
{
   if (!ctx.has_spill)
      no_spill(ctx, origin);
   agc_isa_spill_lanes *spill = &ctx.spill;
   uint16_t lanes = (uint16_t)((ctx.wave64 ? 64 : 32) - spill->persistent);
   uint16_t size = (uint16_t)((count + block_align - 1) / block_align * block_align);
   uint16_t used_lanes = 0;
   for (unsigned i = 0; i < spill->saved_count; i++)
      used_lanes = (uint16_t)(used_lanes + spill->saved[i].size);
   if (used_lanes + size > lanes || spill->saved_count == AGC_ISA_SPILL_MAX_BLOCKS)
      fail(ctx, origin, "the SGPRs one instruction spills exceed the lanes of a VGPR");
   agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
   agc_isa_reg_range touched[2 * AGC_ISA_MAX_REG_RANGES];
   unsigned touched_count = scalar_ranges(ranges, agc_isa_inst_uses(origin, ctx.wave64, ranges), touched, 0);
   touched_count = scalar_ranges(ranges, agc_isa_inst_defs(origin, ctx.wave64, ranges), touched, touched_count);
   uint16_t owned_start, owned_end;
   agc_isa_regalloc_translation_sgprs(ctx.regs, &owned_start, &owned_end);
   uint16_t first = (uint16_t)((AGC_ISA_SGPR_LIMIT - size) / block_align * block_align);
   while (overlaps(touched, touched_count, owned_start, owned_end, spill, first, size)) {
      if (first < block_align)
         fail(ctx, origin, "no SGPR block the instruction leaves alone can be spilled");
      first = (uint16_t)(first - block_align);
   }
   uint8_t first_lane = (uint8_t)used_lanes;
   waitcnt(ctx, AGC_ISA_GFX10_VMCNT_MAX, AGC_ISA_GFX10_EXPCNT_MAX, 0);
   for (uint8_t k = 0; k < size; k++)
      write_lane_raw(ctx, spill->vgpr, sop(first + k), (unsigned)(first_lane + k));
   agc_isa_spill_block *block = &spill->saved[spill->saved_count++];
   block->base = (uint8_t)first;
   block->size = (uint8_t)size;
   block->first_lane = first_lane;
   return (uint8_t)first;
}

void
read_lane(lower_ctx &ctx, const agc_isa_inst *origin, uint8_t sdst, uint8_t lane)
{
   if (!ctx.has_spill)
      no_spill(ctx, origin);
   ctx.bld.vop3(aco_opcode::v_readlane_b32_e64, sdef(sdst), vop(ctx.spill.vgpr), imm(lane));
}

void
write_lane(lower_ctx &ctx, const agc_isa_inst *origin, Operand value, uint8_t lane)
{
   if (!ctx.has_spill)
      no_spill(ctx, origin);
   write_lane_raw(ctx, ctx.spill.vgpr, value, lane);
}

void
restore_spills(lower_ctx &ctx)
{
   if (!ctx.has_spill || ctx.spill.saved_count == 0)
      return;
   uint8_t vgpr = ctx.spill.vgpr;
   unsigned count = ctx.spill.saved_count;
   agc_isa_spill_block saved[AGC_ISA_SPILL_MAX_BLOCKS];
   memcpy(saved, ctx.spill.saved, count * sizeof(saved[0]));
   ctx.spill.saved_count = 0;
   for (unsigned i = 0; i < count; i++) {
      for (uint8_t k = 0; k < saved[i].size; k++)
         ctx.bld.vop3(aco_opcode::v_readlane_b32_e64, sdef(saved[i].base + k), vop(vgpr),
                      imm((unsigned)(saved[i].first_lane + k)));
   }
}

}
