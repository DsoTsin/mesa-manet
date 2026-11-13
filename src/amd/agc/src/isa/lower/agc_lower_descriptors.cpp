#include <atomic>

#include "agc_lower.h"

static std::atomic<uint64_t> report_counts[2];

void
agc_isa_lowering_report_counts(uint64_t counts[2])
{
   counts[0] = report_counts[0].load(std::memory_order_relaxed);
   counts[1] = report_counts[1].load(std::memory_order_relaxed);
}

static bool
writes(const struct agc_isa_reg_range *defs, unsigned count, uint8_t first, uint8_t dwords)
{
   for (unsigned i = 0; i < count; i++) {
      for (unsigned k = 0; k < dwords; k++) {
         if (agc_isa_reg_range_contains(defs[i], AGC_ISA_REG_SCALAR, (uint16_t)(first + k)))
            return true;
      }
   }
   return false;
}

void
agc_isa_descriptor_cache_invalidate(struct agc_isa_descriptor_cache *cache, const struct agc_isa_inst *inst,
                                    bool wave64)
{
   if (!cache->has_vsharp && !cache->has_tsharp && !cache->has_ssharp)
      return;
   struct agc_isa_reg_range defs[AGC_ISA_MAX_REG_RANGES];
   unsigned count = agc_isa_inst_defs(inst, wave64, defs);
   if (cache->has_vsharp && writes(defs, count, cache->vsharp, 4)) {
      cache->has_vsharp = false;
      cache->vsharp = 0;
   }
   if (cache->has_tsharp && writes(defs, count, cache->tsharp, cache->tsharp_r128 ? 4 : 8)) {
      cache->has_tsharp = false;
      cache->tsharp = 0;
      cache->tsharp_r128 = false;
      cache->tsharp_stored = false;
   }
   if (cache->has_ssharp && writes(defs, count, cache->ssharp, 4)) {
      cache->has_ssharp = false;
      cache->ssharp = 0;
   }
}

static bool
touches(struct agc_isa_reg_range range, uint8_t base, uint8_t dwords)
{
   for (unsigned k = 0; k < dwords; k++) {
      if (agc_isa_reg_range_contains(range, AGC_ISA_REG_SCALAR, (uint16_t)(base + k)))
         return true;
   }
   return false;
}

static bool
exact(struct agc_isa_reg_range range, uint8_t base, uint8_t dwords)
{
   return range.file == AGC_ISA_REG_SCALAR && range.first == base && range.count == dwords;
}

static bool
used_as_descriptor(const struct agc_isa_inst *inst, const struct agc_isa_resource_binding *binding)
{
   uint8_t base = binding->base_sgpr;
   switch (inst->fields) {
   case AGC_ISA_MUBUF:
   case AGC_ISA_MTBUF:
      return binding->kind == AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR && inst->buffer.srsrc == base;
   case AGC_ISA_SMEM:
      return binding->kind == AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR && inst->smem.buffer && inst->smem.sbase == base;
   case AGC_ISA_MIMG:
      if (binding->kind == AGC_ISA_RESOURCE_IMAGE_DESCRIPTOR) {
         const struct agc_isa_image_op *info = agc_isa_image_op(inst->opcode);
         bool bvh = info && (info->flags & AGC_ISA_IMG_BVH) != 0;
         return inst->mimg.srsrc == base && !bvh && binding->dword_count == (inst->mimg.r128 ? 4 : 8);
      }
      if (binding->kind == AGC_ISA_RESOURCE_SAMPLER_DESCRIPTOR)
         return inst->mimg.uses_sampler && inst->mimg.ssamp == base;
      return false;
   default:
      return false;
   }
}

static bool
only_used_as(const struct agc_isa_ir *ir, const struct agc_isa_resource_binding *binding, bool wave64)
{
   uint8_t base = binding->base_sgpr;
   uint8_t dwords = binding->dword_count;
   for (size_t i = 0; i < ir->instruction_count; i++) {
      const struct agc_isa_inst *inst = &ir->instructions[i];
      struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
      unsigned count = agc_isa_inst_defs(inst, wave64, ranges);
      for (unsigned r = 0; r < count; r++) {
         if (touches(ranges[r], base, dwords))
            return false;
      }
      bool descriptor = used_as_descriptor(inst, binding);
      count = agc_isa_inst_uses(inst, wave64, ranges);
      for (unsigned r = 0; r < count; r++) {
         if (touches(ranges[r], base, dwords) && !(descriptor && exact(ranges[r], base, dwords)))
            return false;
      }
   }
   return true;
}

bool
agc_isa_lowering_pure_user_descriptors(const struct agc_isa_ir *ir, bool wave64,
                                       struct agc_isa_resource_binding **bindings, size_t *count)
{
   *bindings = NULL;
   *count = 0;
   uint16_t user = ir->metadata.user_sgpr_count;
   struct agc_isa_resource_binding *out =
      (struct agc_isa_resource_binding *)malloc((ir->metadata.binding_count + 1) * sizeof(*out));
   if (!out)
      return false;
   size_t found = 0;
   for (size_t i = 0; i < ir->metadata.binding_count; i++) {
      const struct agc_isa_resource_binding *binding = &ir->metadata.bindings[i];
      bool descriptor = binding->kind == AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR ||
                        binding->kind == AGC_ISA_RESOURCE_IMAGE_DESCRIPTOR ||
                        binding->kind == AGC_ISA_RESOURCE_SAMPLER_DESCRIPTOR;
      if (!descriptor || (uint16_t)binding->base_sgpr + binding->dword_count > user)
         continue;
      if (only_used_as(ir, binding, wave64))
         out[found++] = *binding;
   }
   if (found == 0) {
      free(out);
      return true;
   }
   *bindings = out;
   *count = found;
   return true;
}

namespace agc {

enum host_layout {
   HOST_GFX11,
   HOST_GFX12,
};

struct guest_descriptor {
   unsigned source;
   unsigned dwords;
};

struct tsharp_temps {
   unsigned fmt_index;
   unsigned mode_index;
   unsigned format;
   unsigned mode;
   unsigned min_lod;
   unsigned field;
   unsigned tmp;
   unsigned saved;
};

static uint32_t
trailing_zeros(uint32_t value)
{
   uint32_t count = 0;
   while (count < 32 && !(value & (1u << count)))
      count++;
   return count;
}

static Operand
guest_dword(const guest_descriptor &guest, unsigned k)
{
   return k < guest.dwords ? sop(guest.source + k) : imm(0);
}

static void
alu(Builder &bld, aco_opcode op, unsigned sdst, Operand src0, Operand src1)
{
   bld.sop2(op, sdef(sdst), scc_def(), src0, src1);
}

static void
select(Builder &bld, unsigned sdst, Operand set, Operand clear)
{
   bld.sop2(aco_opcode::s_cselect_b32, sdef(sdst), set, clear, scc_op());
}

static void
test_bit(Builder &bld, Operand value, uint32_t bit)
{
   bld.sopc(aco_opcode::s_bitcmp1_b32, scc_def(), value, imm(bit));
}

static void
mov(Builder &bld, unsigned sdst, Operand value)
{
   bld.sop1(aco_opcode::s_mov_b32, sdef(sdst), value);
}

static void
bfe(Builder &bld, unsigned sdst, Operand value, uint32_t offset, uint32_t width)
{
   alu(bld, aco_opcode::s_bfe_u32, sdst, value, imm(offset | (width << 16)));
}

static void
or_into(Builder &bld, unsigned t, unsigned value)
{
   alu(bld, aco_opcode::s_or_b32, t, sop(t), sop(value));
}

static void
move_field(Builder &bld, unsigned t, unsigned field, Operand value, uint32_t offset, uint32_t width, uint32_t to)
{
   bfe(bld, field, value, offset, width);
   alu(bld, aco_opcode::s_lshl_b32, field, sop(field), imm(to));
   or_into(bld, t, field);
}

static void
wait_for_loads(lower_ctx &ctx)
{
   waitcnt(ctx, AGC_ISA_GFX10_VMCNT_MAX, AGC_ISA_GFX10_EXPCNT_MAX, 0);
}

static void
load_table(Builder &bld, unsigned table, uint64_t address)
{
   mov(bld, table, imm((uint32_t)address));
   mov(bld, table + 1, imm((uint32_t)(address >> 32)));
}

static void
tsharp_mode(Builder &bld, unsigned t, const tsharp_temps &temps, Operand g0)
{
   bfe(bld, temps.tmp, sop(temps.mode), AGC_ISA_DESCRIPTOR_XOR_SHIFT, 8);
   alu(bld, aco_opcode::s_andn2_b32, t, g0, sop(temps.tmp));
   or_into(bld, t + 1, temps.format);
   alu(bld, aco_opcode::s_and_b32, temps.tmp, sop(temps.mode), imm(0x01f00000u));
   or_into(bld, t + 3, temps.tmp);
   alu(bld, aco_opcode::s_and_b32, temps.tmp, sop(temps.mode), imm(1));
   or_into(bld, t + 1, temps.tmp);
   test_bit(bld, sop(temps.mode), 1);
   select(bld, temps.tmp, imm(0xffffc000u), imm(0xffffffffu));
   alu(bld, aco_opcode::s_and_b32, t + 4, sop(t + 4), sop(temps.tmp));
}

static void
gfx11_tsharp(lower_ctx &ctx, unsigned t, const tsharp_temps &temps, const guest_descriptor &guest, bool native_dcc)
{
   Builder &bld = ctx.bld;
   unsigned format = temps.format;
   unsigned mode = temps.mode;
   unsigned min_lod = temps.min_lod;
   unsigned max_mip = temps.field;
   unsigned tmp = temps.tmp;
   bool dcc = native_dcc && guest.dwords == 8;
   bfe(bld, min_lod, guest_dword(guest, 1), 8, 12);
   bfe(bld, max_mip, guest_dword(guest, 5), 4, 4);
   alu(bld, aco_opcode::s_and_b32, t + 1, guest_dword(guest, 1), imm(0xc00000ffu));
   alu(bld, aco_opcode::s_lshl_b32, max_mip, sop(max_mip), imm(16));
   or_into(bld, t + 1, max_mip);
   alu(bld, aco_opcode::s_and_b32, t + 2, guest_dword(guest, 2), imm(0x0fffcfffu));
   alu(bld, aco_opcode::s_and_b32, t + 5, guest_dword(guest, 5), imm(0x04ffff0fu));
   alu(bld, aco_opcode::s_lshl_b32, tmp, sop(min_lod), imm(27));
   or_into(bld, t + 5, tmp);
   alu(bld, aco_opcode::s_lshr_b32, t + 6, sop(min_lod), imm(5));
   if (dcc) {
      test_bit(bld, guest_dword(guest, 6), trailing_zeros(AGC_ISA_DESCRIPTOR_COMPRESSION_EN));
      select(bld, t + 7, guest_dword(guest, 7), imm(0));
      select(bld, max_mip, imm(AGC_ISA_DESCRIPTOR_DCC_WORD6_FIELDS), imm(0));
      alu(bld, aco_opcode::s_and_b32, max_mip, guest_dword(guest, 6), sop(max_mip));
      or_into(bld, t + 6, max_mip);
   } else {
      mov(bld, t + 7, imm(0));
   }
   alu(bld, aco_opcode::s_and_b32, t + 3, guest_dword(guest, 3), imm(0xfe0fffffu));
   mov(bld, t + 4, guest_dword(guest, 4));
   wait_for_loads(ctx);
   if (dcc) {
      bfe(bld, tmp, sop(mode), AGC_ISA_DESCRIPTOR_XOR_SHIFT, AGC_ISA_DESCRIPTOR_DCC_XOR_BITS);
      alu(bld, aco_opcode::s_lshl_b32, tmp, sop(tmp), imm(24));
      alu(bld, aco_opcode::s_andn2_b32, t + 6, sop(t + 6), sop(tmp));
   }
   alu(bld, aco_opcode::s_and_b32, tmp, sop(mode), sop(format));
   test_bit(bld, sop(tmp), trailing_zeros(AGC_ISA_DESCRIPTOR_DIRECT_D));
   select(bld, tmp, imm(~3u), imm(0xffffffffu));
   alu(bld, aco_opcode::s_and_b32, mode, sop(mode), sop(tmp));
   alu(bld, aco_opcode::s_andn2_b32, format, sop(format), imm(AGC_ISA_DESCRIPTOR_DIRECT_D));
   tsharp_mode(bld, t, temps, guest_dword(guest, 0));
}

static void
gfx12_tsharp(lower_ctx &ctx, unsigned t, const tsharp_temps &temps, const guest_descriptor &guest)
{
   Builder &bld = ctx.bld;
   unsigned min_lod = temps.min_lod;
   unsigned field = temps.field;
   bfe(bld, min_lod, guest_dword(guest, 1), 8, 12);
   alu(bld, aco_opcode::s_and_b32, t + 1, guest_dword(guest, 1), imm(0xc00000ffu));
   move_field(bld, t + 1, field, guest_dword(guest, 5), 4, 4, 12);
   move_field(bld, t + 1, field, guest_dword(guest, 3), 12, 4, 25);
   alu(bld, aco_opcode::s_and_b32, t + 2, guest_dword(guest, 2), imm(0x0fffcfffu));
   alu(bld, aco_opcode::s_and_b32, t + 3, guest_dword(guest, 3), imm(0xfe000fffu));
   move_field(bld, t + 3, field, guest_dword(guest, 3), 16, 4, 15);
   alu(bld, aco_opcode::s_and_b32, t + 4, guest_dword(guest, 4), imm(0x1fff3fffu));
   bfe(bld, t + 5, guest_dword(guest, 5), 8, 12);
   alu(bld, aco_opcode::s_lshl_b32, t + 5, sop(t + 5), imm(5));
   alu(bld, aco_opcode::s_and_b32, field, guest_dword(guest, 5), imm(0x00f00000u));
   or_into(bld, t + 5, field);
   alu(bld, aco_opcode::s_and_b32, field, guest_dword(guest, 5), imm(0xf));
   alu(bld, aco_opcode::s_min_u32, field, sop(field), imm(1));
   alu(bld, aco_opcode::s_lshl_b32, field, sop(field), imm(4));
   or_into(bld, t + 5, field);
   alu(bld, aco_opcode::s_lshl_b32, field, sop(min_lod), imm(26));
   or_into(bld, t + 5, field);
   alu(bld, aco_opcode::s_lshr_b32, t + 6, sop(min_lod), imm(6));
   mov(bld, t + 7, imm(0));
   wait_for_loads(ctx);
   alu(bld, aco_opcode::s_lshr_b32, temps.format, sop(temps.format), imm(AGC_ISA_DESCRIPTOR_GFX12_FORMAT_SHIFT));
   tsharp_mode(bld, t, temps, guest_dword(guest, 0));
}

static void
gfx11_ssharp(Builder &bld, unsigned t, unsigned tmp, const guest_descriptor &guest)
{
   alu(bld, aco_opcode::s_lshl_b32, tmp, guest_dword(guest, 3), imm(18));
   alu(bld, aco_opcode::s_and_b32, tmp, sop(tmp), imm(0x3ffc0000u));
   alu(bld, aco_opcode::s_and_b32, t + 3, guest_dword(guest, 3), imm(0xc0000000u));
   or_into(bld, t + 3, tmp);
   mov(bld, t + 2, guest_dword(guest, 2));
   bld.sop1(aco_opcode::s_bitset0_b32, sdef(t + 2), imm(28), sop(t + 2));
   mov(bld, t, guest_dword(guest, 0));
   mov(bld, t + 1, guest_dword(guest, 1));
}

static void
gfx12_ssharp(Builder &bld, unsigned t, unsigned tmp, const guest_descriptor &guest)
{
   mov(bld, t, guest_dword(guest, 0));
   bfe(bld, t + 2, guest_dword(guest, 1), 24, 4);
   alu(bld, aco_opcode::s_lshr_b32, t + 3, sop(t + 2), imm(2));
   alu(bld, aco_opcode::s_lshl_b32, t + 2, sop(t + 2), imm(30));
   alu(bld, aco_opcode::s_and_b32, tmp, guest_dword(guest, 2), imm(0x2fffffffu));
   or_into(bld, t + 2, tmp);
   alu(bld, aco_opcode::s_lshl_b32, tmp, guest_dword(guest, 3), imm(18));
   alu(bld, aco_opcode::s_and_b32, tmp, sop(tmp), imm(0x3ffc0000u));
   or_into(bld, t + 3, tmp);
   alu(bld, aco_opcode::s_and_b32, tmp, guest_dword(guest, 3), imm(0xc0000000u));
   or_into(bld, t + 3, tmp);
   alu(bld, aco_opcode::s_and_b32, t + 1, guest_dword(guest, 1), imm(0xf0000fffu));
   move_field(bld, t + 1, tmp, guest_dword(guest, 1), 12, 12, 13);
}

static const agc_isa_descriptor_slots &
descriptor_slots(const lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (!ctx.has_descriptors)
      fail(ctx, inst, "descriptor conversion is not set up");
   return ctx.descriptors;
}

static void
check_slot(const lower_ctx &ctx, const agc_isa_inst *inst, uint8_t slot, const agc_isa_descriptor_slots &slots)
{
   if (slot == AGC_ISA_NO_SLOT || (slot != slots.ssharp && slots.table == AGC_ISA_NO_SLOT))
      fail(ctx, inst, "a descriptor the program's resource bindings do not list needs conversion");
}

static bool
spilled(const lower_ctx &ctx, uint8_t slot)
{
   return slot == AGC_ISA_NO_SLOT && ctx.has_spill;
}

static void
check_source(const lower_ctx &ctx, const agc_isa_inst *inst, unsigned first, unsigned count)
{
   if (first + count > AGC_ISA_SGPR_LIMIT)
      fail(ctx, inst, "descriptor outside the SGPRs");
}

static host_layout
host_descriptors(const lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (agc_isa_target_is_gfx11(ctx.target))
      return HOST_GFX11;
   if (agc_isa_target_is_gfx12(ctx.target))
      return HOST_GFX12;
   fail(ctx, inst, "the target reads guest descriptors as they are");
}

template <typename Allocate>
static unsigned
with_report_room(uint8_t *extra, Allocate allocate)
{
   if (*extra) {
      try {
         return allocate(*extra);
      } catch (const lower_error &) {
         *extra = 0;
      }
   }
   return allocate(0);
}

uint8_t
converted_vsharp(lower_ctx &ctx, const agc_isa_inst *inst, uint8_t source)
{
   flush(ctx);
   const agc_isa_descriptor_slots &slots = descriptor_slots(ctx, inst);
   bool is_spilled = spilled(ctx, slots.vsharp);
   if (!is_spilled) {
      check_slot(ctx, inst, slots.vsharp, slots);
      if (ctx.descriptor_cache.has_vsharp && ctx.descriptor_cache.vsharp == source)
         return slots.vsharp;
   }
   check_source(ctx, inst, source, 4);
   Builder &bld = ctx.bld;
   unsigned t;
   unsigned temps;
   unsigned table;
   if (is_spilled) {
      unsigned block = spill_block(ctx, inst, 10);
      load_table(bld, block + 8, slots.address);
      t = block;
      temps = block + 4;
      table = block + 8;
   } else {
      t = slots.vsharp;
      temps = transient_sgprs(ctx, inst, 4);
      table = slots.table;
   }
   unsigned index = temps;
   unsigned swizzle = temps + 1;
   unsigned format = temps + 2;
   unsigned saved = temps + 3;
   save_scc(ctx, saved);
   alu(bld, aco_opcode::s_lshr_b32, index, sop(source + 3), imm(10));
   alu(bld, aco_opcode::s_and_b32, index, sop(index), imm(0x1fc));
   smem_load(ctx, 1, sdef(format), table, sop(index), AGC_ISA_DESCRIPTOR_LUT_BUFFER_FORMATS);
   alu(bld, aco_opcode::s_and_b32, t + 3, sop(source + 3), imm(0xf0e00fffu));
   alu(bld, aco_opcode::s_lshr_b32, swizzle, sop(source + 1), imm(1));
   alu(bld, aco_opcode::s_and_b32, swizzle, sop(swizzle), imm(0x40000000u));
   alu(bld, aco_opcode::s_and_b32, t + 1, sop(source + 1), imm(0x3fffffffu));
   alu(bld, aco_opcode::s_or_b32, t + 1, sop(t + 1), sop(swizzle));
   mov(bld, t, sop(source));
   mov(bld, t + 2, sop(source + 2));
   wait_for_loads(ctx);
   alu(bld, aco_opcode::s_or_b32, t + 3, sop(t + 3), sop(format));
   restore_scc(ctx, saved);
   if (!is_spilled) {
      flush(ctx);
      ctx.descriptor_cache.has_vsharp = true;
      ctx.descriptor_cache.vsharp = source;
   }
   return (uint8_t)t;
}

uint8_t
converted_tsharp(lower_ctx &ctx, const agc_isa_inst *inst, uint8_t source, bool r128, bool has_report, uint64_t report,
                 bool stored)
{
   flush(ctx);
   const agc_isa_descriptor_slots &slots = descriptor_slots(ctx, inst);
   bool is_spilled = spilled(ctx, slots.tsharp);
   if (!is_spilled) {
      check_slot(ctx, inst, slots.tsharp, slots);
      const agc_isa_descriptor_cache &cache = ctx.descriptor_cache;
      if (cache.has_tsharp && cache.tsharp == source && cache.tsharp_r128 == r128 &&
          (cache.tsharp_stored || !stored || !has_report))
         return slots.tsharp;
   }
   check_source(ctx, inst, source, r128 ? 4 : 8);
   Builder &bld = ctx.bld;
   bool wanted = has_report && (ctx.wave64 || !ctx.has_spill);
   uint8_t extra = wanted ? AGC_ISA_IMAGE_REPORT_SGPRS : 0;
   unsigned t;
   unsigned first;
   unsigned table;
   unsigned spare;
   if (is_spilled) {
      unsigned block =
         with_report_room(&extra, [&](uint8_t room) { return spill_block(ctx, inst, (uint8_t)(18 + room)); });
      load_table(bld, block + 16, slots.address);
      t = block;
      first = block + 8;
      table = block + 16;
      spare = block + 18;
   } else {
      first = with_report_room(&extra, [&](uint8_t room) { return transient_sgprs(ctx, inst, (uint8_t)(8 + room)); });
      t = slots.tsharp;
      table = slots.table;
      spare = first + 8;
   }
   host_layout layout = host_descriptors(ctx, inst);
   const tsharp_temps temps = {first, first + 1, first + 2, first + 3, first + 4, first + 5, first + 6, first + 7};
   agc_isa_image_report image_report;
   bool reports = wanted && extra != 0 && spare % 2 == 0 &&
                  image_report_new(ctx, report, source, r128, (uint8_t)spare, stored, &image_report);
   if (has_report) {
      report_counts[0].fetch_add(1, std::memory_order_relaxed);
      report_counts[1].fetch_add(reports, std::memory_order_relaxed);
   }
   const guest_descriptor guest = {source, r128 ? 4u : 8u};
   save_scc(ctx, temps.saved);
   if (reports)
      image_report_begin(ctx, &image_report);
   alu(bld, aco_opcode::s_lshr_b32, temps.fmt_index, guest_dword(guest, 1), imm(18));
   alu(bld, aco_opcode::s_and_b32, temps.fmt_index, sop(temps.fmt_index), imm(0x7fc));
   alu(bld, aco_opcode::s_lshr_b32, temps.mode_index, guest_dword(guest, 3), imm(18));
   alu(bld, aco_opcode::s_and_b32, temps.mode_index, sop(temps.mode_index), imm(0x3ffc));
   smem_load(ctx, 1, sdef(temps.format), table, sop(temps.fmt_index), AGC_ISA_DESCRIPTOR_LUT_IMAGE_FORMATS);
   smem_load(ctx, 1, sdef(temps.mode), table, sop(temps.mode_index),
             layout == HOST_GFX12 ? AGC_ISA_DESCRIPTOR_LUT_GFX12_SWIZZLE_MODES : AGC_ISA_DESCRIPTOR_LUT_SWIZZLE_MODES);
   if (layout == HOST_GFX11)
      gfx11_tsharp(ctx, t, temps, guest, slots.native_dcc);
   else
      gfx12_tsharp(ctx, t, temps, guest);
   if (reports)
      image_report_finish(ctx, &image_report, inst);
   restore_scc(ctx, temps.saved);
   if (!is_spilled) {
      flush(ctx);
      ctx.descriptor_cache.has_tsharp = true;
      ctx.descriptor_cache.tsharp = source;
      ctx.descriptor_cache.tsharp_r128 = r128;
      ctx.descriptor_cache.tsharp_stored = stored;
   }
   return (uint8_t)t;
}

uint8_t
converted_ssharp(lower_ctx &ctx, const agc_isa_inst *inst, uint8_t source)
{
   flush(ctx);
   const agc_isa_descriptor_slots &slots = descriptor_slots(ctx, inst);
   bool is_spilled = spilled(ctx, slots.ssharp);
   if (!is_spilled) {
      check_slot(ctx, inst, slots.ssharp, slots);
      if (ctx.descriptor_cache.has_ssharp && ctx.descriptor_cache.ssharp == source)
         return slots.ssharp;
   }
   check_source(ctx, inst, source, 4);
   host_layout layout = host_descriptors(ctx, inst);
   Builder &bld = ctx.bld;
   unsigned t;
   unsigned temps;
   if (is_spilled) {
      t = spill_block(ctx, inst, 6);
      temps = t + 4;
   } else {
      t = slots.ssharp;
      temps = transient_sgprs(ctx, inst, 2);
   }
   unsigned tmp = temps;
   unsigned saved = temps + 1;
   const guest_descriptor guest = {source, 4};
   save_scc(ctx, saved);
   if (layout == HOST_GFX11)
      gfx11_ssharp(bld, t, tmp, guest);
   else
      gfx12_ssharp(bld, t, tmp, guest);
   restore_scc(ctx, saved);
   if (!is_spilled) {
      flush(ctx);
      ctx.descriptor_cache.has_ssharp = true;
      ctx.descriptor_cache.ssharp = source;
   }
   return (uint8_t)t;
}

void
load_descriptor_table(lower_ctx &ctx)
{
   if (!ctx.has_descriptors || ctx.descriptors.table == AGC_ISA_NO_SLOT)
      return;
   load_table(ctx.bld, ctx.descriptors.table, ctx.descriptors.address);
}

}
