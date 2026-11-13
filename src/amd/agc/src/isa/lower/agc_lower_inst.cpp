#include <string.h>

#include "agc_lower.h"

namespace agc {

static void
image_addresses_pack(lower_ctx &ctx, agc_isa_inst *image)
{
   if (image->fields != AGC_ISA_MIMG)
      return;
   agc_isa_image *m = &image->mimg;
   if (!ctx.gfx11_plus || !m->nsa)
      return;
   unsigned slots = ctx.gfx12 && agc_isa_gfx12_vsample(image->opcode, m->uses_sampler) ? 4 : 5;
   unsigned count = m->addr_dwords;
   if (count <= slots)
      return;
   unsigned tail = slots - 1;
   bool consecutive = true;
   for (unsigned i = tail; i < count && consecutive; i++)
      consecutive = (unsigned)m->addr[i] == (unsigned)m->addr[tail] + i - tail;
   if (consecutive)
      return;
   uint16_t scratch = agc_isa_regalloc_scratch_base(ctx.regs);
   uint16_t held = 0;
   for (unsigned i = 0; i < count; i++) {
      if (m->addr[i] >= scratch) {
         uint16_t above = (uint16_t)(m->addr[i] - scratch + 1);
         if (above > held)
            held = above;
      }
   }
   uint8_t base = scratch_vgprs(ctx, (uint8_t)((uint8_t)held + (count - tail)));
   base = (uint8_t)(base + (uint8_t)held);
   for (unsigned i = tail; i < count; i++) {
      uint8_t to = (uint8_t)(base + (i - tail));
      ctx.bld.vop1(aco_opcode::v_mov_b32, vdef(to), vop(m->addr[i]));
      m->addr[i] = to;
   }
}

static const uint16_t msg_interrupt = 1;
static const uint16_t msg_early_prim_dealloc = 8;
static const uint16_t msg_gs_alloc_req = 9;
static const uint16_t hwreg_gpr_alloc = 5;
static const uint16_t hwreg_mode_f32_denorm = 1 | (4 << 6) | (1 << 11);
static const uint8_t f32_denorm_mode = 0x30;
static const uint32_t shader_clocks_per_32_realtime = 715;
static const uint16_t dpp_row_xmask_1 = 0x161;

uint8_t
scratch_vgprs(lower_ctx &ctx, uint8_t count)
{
   agc_isa_error error;
   uint8_t base;
   memset(&error, 0, sizeof(error));
   if (!agc_isa_regalloc_scratch_vgprs(ctx.regs, count, &base, &error))
      fail_error(error);
   return base;
}

uint8_t
transient_sgprs(lower_ctx &ctx, const agc_isa_inst *origin, uint8_t count)
{
   uint8_t base;
   if (agc_isa_regalloc_scratch_sgprs(ctx.regs, count, &base))
      return base;
   return spill_block(ctx, origin, count);
}

void
save_scc(lower_ctx &ctx, unsigned sgpr)
{
   ctx.bld.sop2(aco_opcode::s_cselect_b32, sdef(sgpr), imm(1), imm(0), scc_op());
}

void
restore_scc(lower_ctx &ctx, unsigned sgpr)
{
   ctx.bld.sopc(aco_opcode::s_cmp_lg_u32, scc_def(), sop(sgpr), imm(0));
}

void
save_masks(lower_ctx &ctx, unsigned sgpr)
{
   Builder &bld = ctx.bld;
   bld.sop1(Builder::s_mov, sdef(sgpr, bld.lm.size()), mask_op(ctx, exec));
   bld.sop1(Builder::s_mov, sdef(sgpr + 2, bld.lm.size()), mask_op(ctx, vcc));
   save_scc(ctx, sgpr + 4);
}

void
restore_masks(lower_ctx &ctx, unsigned sgpr)
{
   Builder &bld = ctx.bld;
   bld.sop1(Builder::s_mov, mask_def(ctx, exec), sop(sgpr, bld.lm.size()));
   bld.sop1(Builder::s_mov, mask_def(ctx, vcc), sop(sgpr + 2, bld.lm.size()));
   restore_scc(ctx, sgpr + 4);
}

void
exec_and(lower_ctx &ctx, Operand mask)
{
   ctx.bld.sop2(Builder::s_and, mask_def(ctx, exec), scc_def(), mask_op(ctx, exec), mask);
}

void
exec_andn2(lower_ctx &ctx, Operand mask)
{
   ctx.bld.sop2(Builder::s_andn2, mask_def(ctx, exec), scc_def(), mask_op(ctx, exec), mask);
}

void
enter_f32_flush(lower_ctx &ctx, const agc_isa_inst *origin, agc_isa_mode_restore *restore)
{
   memset(restore, 0, sizeof(*restore));
   if (ctx.mode.dynamic) {
      uint8_t saved = transient_sgprs(ctx, origin, 1);
      ctx.bld.sopk(aco_opcode::s_getreg_b32, sdef(saved), hwreg_mode_f32_denorm);
      ctx.bld.sopk(aco_opcode::s_setreg_imm32_b32, lit(0), hwreg_mode_f32_denorm);
      restore->kind = AGC_ISA_MODE_RESTORE_SAVED;
      restore->sgpr = saved;
      return;
   }
   uint8_t mode = ctx.mode.mode;
   if ((mode & f32_denorm_mode) == 0) {
      restore->kind = AGC_ISA_MODE_RESTORE_UNCHANGED;
      return;
   }
   uint16_t denorm = (uint16_t)(mode >> 4);
   uint16_t flushed = (uint16_t)(denorm & ~(uint16_t)(f32_denorm_mode >> 4));
   ctx.bld.sopp(aco_opcode::s_denorm_mode, flushed);
   restore->kind = AGC_ISA_MODE_RESTORE_DENORM;
   restore->denorm = denorm;
}

void
leave_f32_flush(lower_ctx &ctx, agc_isa_mode_restore restore)
{
   switch (restore.kind) {
   case AGC_ISA_MODE_RESTORE_DENORM:
      ctx.bld.sopp(aco_opcode::s_denorm_mode, restore.denorm);
      break;
   case AGC_ISA_MODE_RESTORE_SAVED:
      ctx.bld.sopk(aco_opcode::s_setreg_b32, sop(restore.sgpr), hwreg_mode_f32_denorm);
      break;
   default:
      break;
   }
}

static bool
is_preconverted(const lower_ctx &ctx, enum agc_isa_resource_kind kind, uint8_t base, uint8_t dwords)
{
   for (size_t i = 0; i < ctx.preconverted_count; i++) {
      const agc_isa_resource_binding *b = &ctx.preconverted[i];
      if (b->kind == kind && b->base_sgpr == base && b->dword_count == dwords)
         return true;
   }
   return false;
}

void
lower_template(lower_ctx &ctx, const agc_isa_inst *origin, agc_isa_inst tmpl)
{
   uint64_t flags = 0;
   if (tmpl.family == AGC_ISA_VOP3) {
      if (!agc_isa_vop3_flags(tmpl.opcode, &flags))
         flags = 0;
   } else {
      const agc_isa_op_info *info = agc_isa_op_lookup((enum agc_isa_family)tmpl.family, tmpl.opcode);
      flags = info ? info->flags : 0;
   }
   tmpl.pc = origin->pc;
   tmpl.flags = flags;
   tmpl.has_branch_target = false;
   tmpl.branch_target = 0;
   tmpl.word_count = 0;
   lower_inst(ctx, &tmpl);
}

static bool
materialize_zero_flags(lower_ctx &ctx, const agc_isa_inst *inst, agc_isa_inst *rewritten)
{
   *rewritten = *inst;
   agc_isa_operand *operands[3];
   unsigned count = agc_isa_lowering_source_operands(rewritten, operands);
   bool has_vccz = false;
   bool has_execz = false;
   for (unsigned i = 0; i < count; i++) {
      has_vccz |= operands[i]->kind == AGC_ISA_OPERAND_VCC_Z;
      has_execz |= operands[i]->kind == AGC_ISA_OPERAND_EXEC_Z;
   }
   if (!has_vccz && !has_execz)
      return false;
   uint8_t base = transient_sgprs(ctx, inst, 3);
   uint8_t saved_scc = base;
   uint8_t vccz = (uint8_t)(base + 1);
   uint8_t execz = (uint8_t)(base + 2);
   aco_opcode compare = ctx.wave64 ? aco_opcode::s_cmp_eq_u64 : aco_opcode::s_cmp_eq_u32;
   save_scc(ctx, saved_scc);
   if (has_vccz) {
      ctx.bld.sopc(compare, scc_def(), mask_op(ctx, vcc), imm(0));
      save_scc(ctx, vccz);
   }
   if (has_execz) {
      ctx.bld.sopc(compare, scc_def(), mask_op(ctx, exec), imm(0));
      save_scc(ctx, execz);
   }
   restore_scc(ctx, saved_scc);
   for (unsigned i = 0; i < count; i++) {
      if (operands[i]->kind == AGC_ISA_OPERAND_VCC_Z)
         *operands[i] = agc_isa_operand_sgpr(vccz);
      else if (operands[i]->kind == AGC_ISA_OPERAND_EXEC_Z)
         *operands[i] = agc_isa_operand_sgpr(execz);
   }
   return true;
}

static void
scalar_rule(lower_ctx &ctx, const agc_isa_inst *inst, agc_isa_xlat xlat)
{
   uint16_t simm16 = 0;
   if (inst->fields == AGC_ISA_SOPK)
      simm16 = inst->sopk.simm16;
   else if (inst->fields == AGC_ISA_SOPP)
      simm16 = inst->sopp.simm16;
   switch (xlat.rule) {
   case AGC_ISA_RULE_FIND_FIRST_ZERO: {
      if (inst->fields != AGC_ISA_SOP1)
         fail(ctx, inst, "malformed first-zero instruction");
      bool wide = agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_S0_2);
      unsigned count = wide ? 2 : 1;
      uint8_t temp = transient_sgprs(ctx, inst, (uint8_t)(count + 1));
      unsigned saved_scc = temp + count;
      save_scc(ctx, saved_scc);
      ctx.bld.sop1(wide ? aco_opcode::s_not_b64 : aco_opcode::s_not_b32, sdef(temp, count), scc_def(),
                   src(inst, inst->sop1.src0, count));
      ctx.bld.sop1(wide ? aco_opcode::s_ff1_i32_b64 : aco_opcode::s_ff1_i32_b32, dst(inst, inst->sop1.sdst),
                   sop(temp, count));
      restore_scc(ctx, saved_scc);
      return;
   }
   case AGC_ISA_RULE_CMPK: {
      if (inst->fields != AGC_ISA_SOPK)
         fail(ctx, inst, "malformed SOPK compare");
      uint32_t value =
         agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_IMM_U) ? simm16 : (uint32_t)(int32_t)(int16_t)simm16;
      agc_isa_inst out = agc_isa_inst_derive(inst, AGC_ISA_SOPC, xlat.value, 0);
      out.sopc.src0 = inst->sopk.sdst;
      out.sopc.src1 = agc_isa_operand_constant(value);
      push(ctx, out);
      return;
   }
   case AGC_ISA_RULE_HWREG: {
      uint16_t id = simm16 & AGC_ISA_HWREG_ID_MASK;
      if (id == AGC_ISA_HWREG_POPS_PACKER) {
         pops_packer(ctx, inst);
         return;
      }
      bool flat_scratch = id == AGC_ISA_HWREG_FLAT_SCR_LO || id == AGC_ISA_HWREG_FLAT_SCR_HI;
      if (flat_scratch && ctx.gfx11_plus && inst->opcode != AGC_ISA_SOPK_GETREG_B32)
         return;
      bool portable = (id == 1 || id == 2 || id == 5 || id == 6 || id == 23 || id == 24) &&
                      !(id == hwreg_gpr_alloc && ctx.gfx11_plus);
      if (!portable)
         fail(ctx, inst, "hardware register layout differs on the target");
      push_retarget(ctx, inst, xlat.value);
      return;
   }
   case AGC_ISA_RULE_SENDMSG:
      switch (simm16 & 0xf) {
      case msg_interrupt:
      case msg_gs_alloc_req:
         push_retarget(ctx, inst, xlat.value);
         return;
      case msg_early_prim_dealloc:
      case AGC_ISA_MSG_ORDERED_PS_DONE:
         return;
      default:
         fail(ctx, inst, "message is not available on the target");
      }
   case AGC_ISA_RULE_WAITCNT: {
      uint8_t vm, exp, lgkm;
      agc_isa_waitcnt_decode_gfx10(simm16, &vm, &exp, &lgkm);
      if (ctx.gfx12)
         gfx12_waits(ctx, true, vm, true, exp, true, lgkm, false, 0);
      else
         ctx.bld.sopp(aco_opcode::s_waitcnt, agc_isa_waitcnt_encode_gfx11(vm, exp, lgkm));
      return;
   }
   case AGC_ISA_RULE_WAITCNT_COUNTER: {
      if (inst->fields != AGC_ISA_SOPK)
         fail(ctx, inst, "malformed counter wait");
      if (inst->sopk.sdst.kind != AGC_ISA_OPERAND_NULL)
         fail(ctx, inst, "counter waits read from an SGPR");
      uint8_t count = (uint8_t)(simm16 < 0xff ? simm16 : 0xff);
      switch (inst->opcode) {
      case AGC_ISA_SOPK_WAITCNT_VSCNT:
         gfx12_waits(ctx, false, 0, false, 0, false, 0, true, count);
         return;
      case AGC_ISA_SOPK_WAITCNT_VMCNT:
         gfx12_waits(ctx, true, count, false, 0, false, 0, false, 0);
         return;
      case AGC_ISA_SOPK_WAITCNT_EXPCNT:
         gfx12_waits(ctx, false, 0, true, count, false, 0, false, 0);
         return;
      default:
         gfx12_waits(ctx, false, 0, false, 0, true, count, false, 0);
         return;
      }
   }
   case AGC_ISA_RULE_BARRIER:
      ctx.bld.sop1(aco_opcode::s_barrier_signal, imm(0xffffffffu));
      ctx.bld.sopp(aco_opcode::s_barrier_wait, 0xffff);
      return;
   default:
      fail(ctx, inst, "no scalar lowering for this rule");
   }
}

static void
lower_scalar(lower_ctx &ctx, const agc_isa_inst *inst)
{
   const agc_isa_op_info *info = agc_isa_op_lookup((enum agc_isa_family)inst->family, inst->opcode);
   if (!info)
      fail(ctx, inst, "unknown scalar opcode");
   agc_isa_xlat xlat = agc_isa_op_translate(info, ctx.target);
   switch (xlat.kind) {
   case AGC_ISA_XLAT_OP:
      push_retarget(ctx, inst, xlat.value);
      return;
   case AGC_ISA_XLAT_NO:
      fail(ctx, inst, "instruction was removed on the target");
   case AGC_ISA_XLAT_RULE:
      scalar_rule(ctx, inst, xlat);
      return;
   default:
      return;
   }
}

static void
scale_realtime_to_shader_clock(lower_ctx &ctx, const agc_isa_inst *inst, agc_isa_operand sdata)
{
   if (sdata.kind != AGC_ISA_OPERAND_SGPR)
      fail(ctx, inst, "s_memtime into a special register");
   uint8_t base = transient_sgprs(ctx, inst, 2);
   unsigned lo = sdata.reg;
   unsigned hi = sdata.reg + 1;
   unsigned carry = base;
   unsigned saved_scc = base + 1;
   Operand factor = lit(shader_clocks_per_32_realtime);
   if (ctx.gfx12)
      gfx12_waits(ctx, false, 0, false, 0, true, 0, false, 0);
   else
      ctx.bld.sopp(aco_opcode::s_waitcnt,
                   agc_isa_waitcnt_encode_gfx11(AGC_ISA_GFX10_VMCNT_MAX, AGC_ISA_GFX10_EXPCNT_MAX, 0));
   save_scc(ctx, saved_scc);
   ctx.bld.sop2(aco_opcode::s_mul_hi_u32, sdef(carry), sop(lo), factor);
   ctx.bld.sop2(aco_opcode::s_mul_i32, sdef(lo), sop(lo), factor);
   ctx.bld.sop2(aco_opcode::s_mul_i32, sdef(hi), sop(hi), factor);
   ctx.bld.sop2(aco_opcode::s_add_u32, sdef(hi), scc_def(), sop(hi), sop(carry));
   ctx.bld.sop2(aco_opcode::s_lshr_b64, sdef(lo, 2), scc_def(), sop(lo, 2), imm(5));
   restore_scc(ctx, saved_scc);
}

static void
lower_wave_id(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_SMEM)
      fail(ctx, inst, "malformed SMEM");
   if (!ctx.has_wave_index)
      fail(ctx, inst, "the wave index needs the compute TG_SIZE SGPR");
   agc_isa_operand sdata = inst->smem.sdata;
   if (!ctx.wave_index.lane) {
      ctx.bld.sop1(aco_opcode::s_mov_b32, dst(inst, sdata), sop(ctx.wave_index.index));
      return;
   }
   if (sdata.kind != AGC_ISA_OPERAND_SGPR)
      fail(ctx, inst, "a wave index destination is not an SGPR");
   read_lane(ctx, inst, (uint8_t)sdata.reg, ctx.wave_index.index);
}

static void
lower_smem(lower_ctx &ctx, const agc_isa_inst *inst)
{
   const agc_isa_mem_op *info = agc_isa_smem_op(inst->opcode);
   if (!info)
      fail(ctx, inst, "unknown SMEM opcode");
   agc_isa_xlat xlat = agc_isa_mem_op_translate(info, ctx.target);
   switch (xlat.kind) {
   case AGC_ISA_XLAT_OP:
      push_retarget(ctx, inst, xlat.value);
      return;
   case AGC_ISA_XLAT_NO:
      fail(ctx, inst, "scalar memory operation was removed");
   case AGC_ISA_XLAT_RULE:
      break;
   default:
      return;
   }
   if (xlat.rule == AGC_ISA_RULE_MEM_TIME) {
      if (inst->fields != AGC_ISA_SMEM)
         fail(ctx, inst, "malformed SMEM");
      ctx.bld.sop1(aco_opcode::s_sendmsg_rtn_b64, dst(inst, inst->smem.sdata, 2),
                   Operand(PhysReg{AGC_ISA_MSG_RTN_GET_REALTIME}, s1));
      if (inst->opcode == AGC_ISA_SMEM_MEMTIME)
         scale_realtime_to_shader_clock(ctx, inst, inst->smem.sdata);
      return;
   }
   if (xlat.rule == AGC_ISA_RULE_WAVE_ID) {
      lower_wave_id(ctx, inst);
      return;
   }
   fail(ctx, inst, "no SMEM lowering for this rule");
}

static bool
is_buffer(const agc_isa_inst *inst)
{
   return inst->fields == AGC_ISA_MUBUF || inst->fields == AGC_ISA_MTBUF;
}

static bool
is_private(const lower_ctx &ctx, const agc_isa_inst *inst)
{
   return is_buffer(inst) && agc_isa_pc_set_contains(ctx.private_pcs, inst->pc);
}

static bool
scratch_buffer(lower_ctx &ctx, const agc_isa_inst *inst, agc_isa_inst *scratch)
{
   if (inst->fields != AGC_ISA_MUBUF || !is_private(ctx, inst))
      return false;
   const agc_isa_buffer *b = &inst->buffer;
   if (b->idxen || b->lds)
      fail(ctx, inst, "indexed and LDS-direct private-segment accesses are not supported on the target");
   const agc_isa_mem_op *info = agc_isa_flat_op(inst->opcode);
   if (!info || info->kind == AGC_ISA_MEM_CONTROL)
      fail(ctx, inst, "no flat scratch form");
   *scratch = agc_isa_inst_derive(inst, AGC_ISA_FLAT, inst->opcode, info->flags);
   agc_isa_flat *f = &scratch->flat;
   f->segment = AGC_ISA_SEGMENT_SCRATCH;
   f->vdst = b->vdata;
   f->vdata = b->vdata;
   f->vaddr = b->vaddr;
   f->saddr = b->offen ? agc_isa_operand_special(AGC_ISA_OPERAND_NULL) : agc_isa_operand_none();
   f->uses_vaddr = b->offen;
   f->offset = (int32_t)b->offset;
   f->glc = b->glc;
   f->dlc = b->dlc;
   f->slc = b->slc;
   f->lds = false;
   f->kind = info->kind;
   f->data_dwords = info->dwords;
   return true;
}

static void
buffer_access(lower_ctx &ctx, const agc_isa_inst *inst, bool convert)
{
   const agc_isa_mem_op *info =
      inst->family == AGC_ISA_MUBUF ? agc_isa_mubuf_op(inst->opcode) : agc_isa_mtbuf_op(inst->opcode);
   if (!info)
      fail(ctx, inst, "unknown buffer opcode");
   agc_isa_float_atomic atomic;
   memset(&atomic, 0, sizeof(atomic));
   bool lost = inst->family == AGC_ISA_MUBUF && convert && float_atomic_named(inst->opcode, &atomic);
   agc_isa_xlat xlat = agc_isa_mem_op_translate(info, ctx.target);
   if (xlat.kind == AGC_ISA_XLAT_DROP || xlat.kind == AGC_ISA_XLAT_SAME)
      return;
   if (xlat.kind != AGC_ISA_XLAT_OP && !lost)
      fail(ctx, inst, "buffer operation was removed");
   agc_isa_inst lowered = agc_isa_inst_retarget(inst, xlat.kind == AGC_ISA_XLAT_OP ? xlat.value : inst->opcode);
   if (is_buffer(&lowered)) {
      if (ctx.gfx11_plus && lowered.buffer.lds)
         fail(ctx, inst, "LDS-direct buffer loads use different opcodes");
      if (inst->family == AGC_ISA_MTBUF) {
         uint8_t format;
         if (!agc_isa_remap_buffer_format(lowered.buffer.format, ctx.target, &format))
            fail(ctx, inst, "buffer data format was removed");
         lowered.buffer.format = format;
      }
   }
   if (ctx.has_descriptors && is_buffer(inst) && convert &&
       !is_preconverted(ctx, AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR, inst->buffer.srsrc, 4)) {
      uint8_t converted = converted_vsharp(ctx, inst, inst->buffer.srsrc);
      if (is_buffer(&lowered))
         lowered.buffer.srsrc = converted;
   }
   if (xlat.kind == AGC_ISA_XLAT_NO && lost) {
      memory_atomic(ctx, inst, &lowered, atomic);
      return;
   }
   push(ctx, lowered);
}

static void
private_buffer_access(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (!is_buffer(inst))
      fail(ctx, inst, "malformed buffer instruction");
   uint8_t guest = inst->buffer.srsrc;
   if (ctx.has_descriptors && !is_preconverted(ctx, AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR, inst->buffer.srsrc, 4))
      guest = converted_vsharp(ctx, inst, inst->buffer.srsrc);
   uint8_t t = transient_sgprs(ctx, inst, 6);
   unsigned temp = t + 4;
   unsigned saved_scc = t + 5;
   Builder &bld = ctx.bld;
   save_scc(ctx, saved_scc);
   static const uint16_t ids[2] = {AGC_ISA_HWREG_FLAT_SCR_LO, AGC_ISA_HWREG_FLAT_SCR_HI};
   for (unsigned k = 0; k < 2; k++)
      bld.sopk(aco_opcode::s_getreg_b32, sdef(t + k), (uint16_t)(ids[k] | (31 << 11)));
   uint32_t index_stride = ctx.wave64 ? 3 : 2;
   bld.sop2(aco_opcode::s_and_b32, sdef(t + 1), scc_def(), sop(t + 1), imm(0xffff));
   bld.sop2(aco_opcode::s_and_b32, sdef(temp), scc_def(), sop(guest + 1), imm(0xffff0000u));
   bld.sop2(aco_opcode::s_or_b32, sdef(t + 1), scc_def(), sop(t + 1), sop(temp));
   bld.sop2(aco_opcode::s_or_b32, sdef(t + 2), scc_def(), sop(guest + 2), imm(0));
   bld.sop2(aco_opcode::s_and_b32, sdef(t + 3), scc_def(), sop(guest + 3), imm(~(3u << 21)));
   bld.sop2(aco_opcode::s_or_b32, sdef(t + 3), scc_def(), sop(t + 3), imm(index_stride << 21));
   restore_scc(ctx, saved_scc);
   agc_isa_inst access = *inst;
   access.buffer.srsrc = t;
   access.buffer.soffset = agc_isa_operand_inline_int(0);
   buffer_access(ctx, &access, false);
}

static void
lower_buffer(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (!is_private(ctx, inst)) {
      buffer_access(ctx, inst, true);
      return;
   }
   if (ctx.private_buffer) {
      private_buffer_access(ctx, inst);
      return;
   }
   agc_isa_inst scratch;
   if (!scratch_buffer(ctx, inst, &scratch))
      fail(ctx, inst, "a typed buffer access to the private segment has no flat scratch form");
   lower_inst(ctx, &scratch);
}

static void
lower_flat(lower_ctx &ctx, const agc_isa_inst *inst)
{
   const agc_isa_mem_op *info = agc_isa_flat_op(inst->opcode);
   if (!info)
      fail(ctx, inst, "unknown FLAT opcode");
   agc_isa_xlat xlat = agc_isa_mem_op_translate(info, ctx.target);
   if (xlat.kind != AGC_ISA_XLAT_OP) {
      agc_isa_float_atomic atomic;
      if (float_atomic_named(inst->opcode, &atomic)) {
         memory_atomic(ctx, inst, inst, atomic);
         return;
      }
      fail(ctx, inst, "flat operation was removed");
   }
   if (inst->fields == AGC_ISA_FLAT && ctx.gfx11_plus && inst->flat.lds)
      fail(ctx, inst, "LDS-direct global loads use different opcodes");
   push_retarget(ctx, inst, xlat.value);
}

static void
lower_ds(lower_ctx &ctx, const agc_isa_inst *inst)
{
   const struct agc_isa_ds_op *info = agc_isa_ds_op(inst->opcode);
   if (!info)
      fail(ctx, inst, "unknown DS opcode");
   if (info->flags & AGC_ISA_DS_FLAG_GWS)
      fail(ctx, inst, "GWS is not emulated: ds_gws_* barriers and semaphores across workgroups are rejected");
   if (ctx.has_gds && inst->fields == AGC_ISA_DS &&
       (inst->ds.gds || (info->flags & (AGC_ISA_DS_FLAG_GWS | AGC_ISA_DS_FLAG_GDS_ONLY)))) {
      lower_gds(ctx, inst, ctx.gds);
      return;
   }
   agc_isa_xlat xlat = agc_isa_ds_op_translate(info, ctx.target);
   if (xlat.kind != AGC_ISA_XLAT_OP) {
      if (info->flags & AGC_ISA_DS_FLAG_SRC2) {
         lds_src2(ctx, inst);
         return;
      }
      if (info->flags & AGC_ISA_DS_FLAG_NOT_GFX12) {
         lds_compare_and_swap(ctx, inst);
         return;
      }
      fail(ctx, inst, "GWS and GDS-only operations were removed in gfx12");
   }
   agc_isa_inst lowered = agc_isa_inst_retarget(inst, xlat.value);
   if (lowered.fields == AGC_ISA_DS) {
      if (ctx.gfx12 && lowered.ds.gds)
         fail(ctx, inst, "GDS was removed");
      if (agc_isa_ds_op_swaps_data(info, ctx.target)) {
         uint8_t data0 = lowered.ds.data0;
         lowered.ds.data0 = lowered.ds.data1;
         lowered.ds.data1 = data0;
      }
   }
   push(ctx, lowered);
}

static uint8_t
adjusted_sampler(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_MIMG)
      fail(ctx, inst, "malformed MIMG");
   uint8_t base = transient_sgprs(ctx, inst, 5);
   unsigned samp = inst->mimg.ssamp;
   unsigned saved_scc = base + 4;
   save_scc(ctx, saved_scc);
   ctx.bld.sop1(aco_opcode::s_mov_b64, sdef(base, 2), sop(samp, 2));
   ctx.bld.sop1(aco_opcode::s_mov_b32, sdef(base + 2), sop(samp + 2));
   ctx.bld.sop2(aco_opcode::s_and_b32, sdef(base + 3), scc_def(), sop(samp + 3),
                lit(~AGC_ISA_IMG_SAMPLER_ADJUST_BITS));
   restore_scc(ctx, saved_scc);
   return base;
}

static void
lower_image(lower_ctx &ctx, const agc_isa_inst *inst)
{
   const struct agc_isa_image_op *info = agc_isa_image_op(inst->opcode);
   if (!info)
      fail(ctx, inst, "unknown MIMG opcode");
   if (info->flags & AGC_ISA_IMG_BVH) {
      lower_bvh(ctx, inst);
      return;
   }
   agc_isa_xlat xlat = agc_isa_image_op_translate(info, ctx.target);
   if (xlat.kind == AGC_ISA_XLAT_NO) {
      agc_isa_inst rewritten;
      if (per_quad_derivatives(ctx, inst, info, &rewritten)) {
         lower_image(ctx, &rewritten);
         return;
      }
      if (gather_without_offset(ctx, inst, info, &rewritten)) {
         lower_image(ctx, &rewritten);
         return;
      }
   }
   bool float_atomic = agc_isa_target_is_gfx11(ctx.target) &&
                       (inst->opcode == AGC_ISA_MIMG_ATOMIC_FMIN || inst->opcode == AGC_ISA_MIMG_ATOMIC_FMAX);
   bool float_cmpswap = ctx.gfx11_plus && inst->opcode == AGC_ISA_MIMG_ATOMIC_FCMPSWAP;
   uint16_t op;
   if (xlat.kind == AGC_ISA_XLAT_OP)
      op = xlat.value;
   else if (float_atomic || float_cmpswap)
      op = 0x0b;
   else
      fail(ctx, inst, "image operation is not available on the target");
   agc_isa_inst lowered = agc_isa_inst_retarget(inst, op);
   if (inst->fields != AGC_ISA_MIMG)
      fail(ctx, inst, "malformed MIMG");
   agc_isa_image guest = inst->mimg;
   if (ctx.has_mip_stats) {
      bool sampled = (info->flags & AGC_ISA_IMG_SAMPLER) != 0 && (info->flags & AGC_ISA_IMG_GATHER) == 0;
      if (ctx.stage == AGC_SHADER_STAGE_FRAGMENT && sampled)
         mip_stats_count(ctx, inst, &guest, info->flags, ctx.mip_stats);
   }
   uint8_t image_dwords = guest.r128 ? 4 : 8;
   bool convert_image = ctx.has_descriptors &&
                        !is_preconverted(ctx, AGC_ISA_RESOURCE_IMAGE_DESCRIPTOR, guest.srsrc, image_dwords);
   bool has_report = ctx.has_image_reports;
   bool convert_sampler = ctx.has_descriptors && guest.uses_sampler &&
                          !is_preconverted(ctx, AGC_ISA_RESOURCE_SAMPLER_DESCRIPTOR, guest.ssamp, 4);
   if (convert_image) {
      bool stored = (info->flags & (AGC_ISA_IMG_STORE_OP | AGC_ISA_IMG_ATOMIC_OP)) != 0;
      lowered.mimg.srsrc =
         converted_tsharp(ctx, inst, guest.srsrc, guest.r128, has_report, ctx.image_reports, stored);
      lowered.mimg.r128 = false;
   }
   if (convert_sampler)
      lowered.mimg.ssamp = converted_ssharp(ctx, inst, guest.ssamp);
   else if (info->flags & AGC_ISA_IMG_ADJUST)
      lowered.mimg.ssamp = adjusted_sampler(ctx, inst);
   image_addresses_pack(ctx, &lowered);
   if (float_cmpswap) {
      image_fcmpswap(ctx, inst, &lowered);
      return;
   }
   if (float_atomic) {
      image_atomic(ctx, inst, &lowered);
      return;
   }
   push(ctx, lowered);
}

static void
dual_source_export(lower_ctx &ctx, const agc_isa_inst *inst, agc_isa_export exp, uint8_t held)
{
   Builder &bld = ctx.bld;
   if (!ctx.has_pending_dual_source) {
      for (unsigned c = 0; c < 4; c++) {
         if (!(exp.en & (1u << c)))
            continue;
         bld.vop1(aco_opcode::v_mov_b32, vdef(held + c), vop(exp.vsrc[c]));
      }
      ctx.pending_dual_source = exp;
      ctx.has_pending_dual_source = true;
      return;
   }
   agc_isa_export first = ctx.pending_dual_source;
   ctx.has_pending_dual_source = false;
   if (first.target == exp.target)
      fail(ctx, inst, "dual-source blending exports MRT0 and MRT1 once each");
   uint8_t copies[4] = {held, (uint8_t)(held + 1), (uint8_t)(held + 2), (uint8_t)(held + 3)};
   const uint8_t *src0 = first.target == AGC_ISA_EXP_MRT0 ? copies : exp.vsrc;
   const uint8_t *src1 = first.target == AGC_ISA_EXP_MRT0 ? exp.vsrc : copies;
   uint8_t en = first.en | exp.en;
   uint8_t swizzled = scratch_vgprs(ctx, 8);
   uint8_t saved = transient_sgprs(ctx, inst, 4);
   uint8_t saved_exec = saved;
   unsigned saved_vcc = saved + 2;
   enable_all_lanes(ctx, saved_exec);
   bld.sop1(Builder::s_mov, sdef(saved_vcc, bld.lm.size()), mask_op(ctx, vcc));
   for (unsigned pass = 0; pass < 2; pass++) {
      uint8_t half = pass ? 4 : 0;
      uint32_t mask = pass ? 0xaaaaaaaau : 0x55555555u;
      const uint8_t *others = pass ? src0 : src1;
      const uint8_t *owns = pass ? src1 : src0;
      bld.sop1(aco_opcode::s_mov_b32, Definition(vcc, s1), lit(mask));
      if (ctx.wave64)
         bld.sop1(aco_opcode::s_mov_b32, Definition(vcc_hi, s1), lit(mask));
      for (unsigned c = 0; c < 4; c++) {
         if (!(en & (1u << c)))
            continue;
         agc_isa_inst select =
            agc_isa_tmpl_vop2(AGC_ISA_VOP2_CNDMASK_B32, agc_isa_operand_vgpr((uint8_t)(swizzled + half + c)),
                              agc_isa_operand_vgpr(others[c]), agc_isa_operand_vgpr(owns[c]));
         select.vop2.ext.kind = AGC_ISA_VOP_EXT_DPP16;
         select.vop2.ext.dpp16.ctrl = dpp_row_xmask_1;
         select.vop2.ext.dpp16.fetch_inactive = true;
         select.vop2.ext.dpp16.bank_mask = 0xf;
         select.vop2.ext.dpp16.row_mask = 0xf;
         lower_template(ctx, inst, select);
      }
   }
   bld.sop1(Builder::s_mov, mask_def(ctx, vcc), sop(saved_vcc, bld.lm.size()));
   restore_lanes(ctx, saved_exec);
   static const uint8_t targets[2] = {21, 22};
   for (unsigned k = 0; k < 2; k++) {
      uint8_t base = (uint8_t)(swizzled + 4 * k);
      agc_isa_inst out = agc_isa_inst_derive(inst, AGC_ISA_EXP, inst->opcode, inst->flags);
      out.exp.target = targets[k];
      out.exp.en = en;
      out.exp.compr = false;
      out.exp.done = k == 1 && (first.done || exp.done);
      out.exp.vm = false;
      out.exp.row_en = false;
      for (unsigned c = 0; c < 4; c++)
         out.exp.vsrc[c] = (uint8_t)(base + c);
      push(ctx, out);
   }
}

static void
lower_export(lower_ctx &ctx, const agc_isa_inst *inst)
{
   agc_isa_inst lowered = agc_isa_inst_retarget(inst, inst->opcode);
   if (lowered.fields == AGC_ISA_EXP && ctx.gfx11_plus) {
      agc_isa_export *e = &lowered.exp;
      if (e->target >= AGC_ISA_EXP_PARAM_FIRST) {
         if (ctx.has_ngg && ctx.ngg.has_ring) {
            agc_isa_export copy = *e;
            ring_store(ctx, inst, &copy, &ctx.ngg.ring);
            return;
         }
         if (ctx.stage == AGC_SHADER_STAGE_VERTEX)
            return;
         fail(ctx, inst, "parameter exports require the attribute ring");
      }
      if (e->target == AGC_ISA_EXP_TARGET_NULL) {
         e->target = ctx.pixel ? ctx.pixel->null_target : AGC_ISA_EXP_MRT0;
         e->en = 0;
      }
      if (e->compr) {
         e->en = (uint8_t)(((e->en & 0x3) != 0 ? 1 : 0) | (((e->en & 0xc) != 0 ? 1 : 0) << 1));
         e->vsrc[2] = 0;
         e->vsrc[3] = 0;
         e->compr = false;
      }
      e->vm = false;
      bool done_position = e->done && e->target >= AGC_ISA_EXP_POS_FIRST && e->target <= AGC_ISA_EXP_POS_LAST;
      if (ctx.has_ngg && done_position) {
         if (ctx.ngg.has_deferred) {
            agc_isa_export copy = *e;
            defer_position(ctx, inst, &copy, &ctx.ngg.deferred);
            return;
         }
         if (ctx.ngg.has_ring)
            wait_for_stores(ctx);
      }
      if (ctx.pixel && ctx.pixel->has_dual_source && (e->target == AGC_ISA_EXP_MRT0 || e->target == AGC_ISA_EXP_MRT1)) {
         dual_source_export(ctx, inst, *e, ctx.pixel->dual_source);
         return;
      }
   }
   push(ctx, lowered);
}

void
lower_inst(lower_ctx &ctx, const agc_isa_inst *inst)
{
   flush(ctx);
   agc_isa_descriptor_cache_invalidate(&ctx.descriptor_cache, inst, ctx.wave64);
   if (ctx.epilogue && inst->family == AGC_ISA_SOPP && inst->opcode == AGC_ISA_SOPP_ENDPGM) {
      lower_branch(ctx, AGC_ISA_EPILOGUE_LABEL);
      return;
   }
   if (lower_pc_relative(ctx, inst))
      return;
   if (inst->fields == AGC_ISA_MIMG && inst->mimg.bvh8) {
      lower_bvh8(ctx, inst);
      return;
   }
   if (inst->family == AGC_ISA_DS && inst->opcode == AGC_ISA_DS_BVH_STACK_PUSH8_POP1_RTN_B32) {
      lower_bvh_stack(ctx, inst);
      return;
   }
   if (ctx.gfx11_plus) {
      agc_isa_inst rewritten;
      if (agc_isa_lowering_pops_without_exiting_wave(inst, &rewritten)) {
         lower_inst(ctx, &rewritten);
         return;
      }
      if (materialize_zero_flags(ctx, inst, &rewritten)) {
         lower_inst(ctx, &rewritten);
         return;
      }
   }
   switch (inst->family) {
   case AGC_ISA_SOP1:
   case AGC_ISA_SOP2:
   case AGC_ISA_SOPK:
   case AGC_ISA_SOPC:
   case AGC_ISA_SOPP:
      lower_scalar(ctx, inst);
      return;
   case AGC_ISA_VOP1:
   case AGC_ISA_VOP2:
   case AGC_ISA_VOPC:
   case AGC_ISA_VOP3:
   case AGC_ISA_VOP3P:
      lower_vector(ctx, inst);
      return;
   case AGC_ISA_VINTRP:
      lower_interp(ctx, inst);
      return;
   case AGC_ISA_SMEM:
      lower_smem(ctx, inst);
      return;
   case AGC_ISA_MUBUF:
   case AGC_ISA_MTBUF:
      lower_buffer(ctx, inst);
      return;
   case AGC_ISA_FLAT:
      lower_flat(ctx, inst);
      return;
   case AGC_ISA_DS:
      lower_ds(ctx, inst);
      return;
   case AGC_ISA_MIMG:
      lower_image(ctx, inst);
      return;
   case AGC_ISA_EXP:
      lower_export(ctx, inst);
      return;
   default:
      fail(ctx, inst, "not a gfx1013 instruction");
   }
}

}
