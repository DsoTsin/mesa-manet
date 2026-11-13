#include <array>
#include <string.h>
#include <unordered_map>

#include "agc_lower.h"

#include "ac_gpu_info.h"

namespace agc {

[[noreturn]] void
fail(const lower_ctx &ctx, const agc_isa_inst *inst, const char *reason)
{
   lower_error e;
   memset(&e.error, 0, sizeof(e.error));
   e.error.kind = AGC_ISA_ERROR_UNSUPPORTED_INSTRUCTION;
   e.error.pc = inst->pc;
   e.error.family = inst->family;
   e.error.opcode = inst->opcode;
   e.error.target = (uint8_t)ctx.target;
   e.error.reason = reason;
   throw e;
}

[[noreturn]] void
fail_plain()
{
   lower_error e;
   memset(&e.error, 0, sizeof(e.error));
   e.error.kind = AGC_ISA_ERROR_UNSUPPORTED;
   throw e;
}

[[noreturn]] void
fail_memory()
{
   lower_error e;
   memset(&e.error, 0, sizeof(e.error));
   e.error.kind = AGC_ISA_ERROR_OUT_OF_MEMORY;
   throw e;
}

[[noreturn]] void
fail_error(const agc_isa_error &error)
{
   throw lower_error{error};
}

[[noreturn]] static void
reject(uint32_t pc, const char *reason)
{
   encode_error e;
   memset(&e.error, 0, sizeof(e.error));
   e.error.kind = AGC_ISA_ERROR_INVALID_OPERAND;
   e.error.pc = pc;
   e.error.reason = reason;
   throw e;
}

[[noreturn]] static void
reject_unsupported(uint32_t pc, const char *reason)
{
   encode_error e;
   memset(&e.error, 0, sizeof(e.error));
   e.error.kind = AGC_ISA_ERROR_UNSUPPORTED_INSTRUCTION;
   e.error.pc = pc;
   e.error.reason = reason;
   throw e;
}

static uint32_t
opcode_key(Format format, unsigned opcode)
{
   return (uint32_t)format << 16 | opcode;
}

static Format
instruction_format(const agc_isa_inst *inst)
{
   static const Format formats[] = {
      Format::SOP1,   Format::SOP2,  Format::SOPK,  Format::SOPC, Format::SOPP,   Format::VOP1,
      Format::VOP2,   Format::VOP3,  Format::VOP3P, Format::VOPC, Format::VINTRP, Format::SMEM,
      Format::MUBUF,  Format::MTBUF, Format::FLAT,  Format::DS,   Format::MIMG,   Format::EXP,
      Format::LDSDIR, Format::VINTERP_INREG,
   };
   if (inst->family >= sizeof(formats) / sizeof(formats[0]))
      reject_unsupported(inst->pc, "unknown instruction family");
   if (inst->family == AGC_ISA_FLAT) {
      if (inst->flat.segment == AGC_ISA_SEGMENT_GLOBAL)
         return Format::GLOBAL;
      if (inst->flat.segment == AGC_ISA_SEGMENT_SCRATCH)
         return Format::SCRATCH;
   }
   return formats[inst->family];
}

static aco_opcode
host_opcode(const lower_ctx &ctx, const agc_isa_inst *inst)
{
   using opcode_map = std::unordered_map<uint32_t, aco_opcode>;
   static const std::array<opcode_map, 3> maps = [] {
      std::array<opcode_map, 3> result;
      const int16_t *tables[] = {instr_info.opcode_gfx10, instr_info.opcode_gfx11, instr_info.opcode_gfx12};
      for (unsigned generation = 0; generation < result.size(); generation++) {
         for (unsigned i = 0; i < (unsigned)aco_opcode::num_opcodes; i++) {
            Format format = instr_info.format[i];
            int opcode = tables[generation][i];
            if (opcode >= 0 && format != Format::PSEUDO && format != Format::PSEUDO_BRANCH &&
                format != Format::PSEUDO_BARRIER && format != Format::PSEUDO_REDUCTION &&
                format != Format::PSEUDO_CALL)
               result[generation].emplace(opcode_key(format, opcode), (aco_opcode)i);
         }
         for (unsigned i = 0; i < (unsigned)aco_opcode::num_opcodes; i++) {
            Format format = instr_info.format[i];
            int opcode = tables[generation][i];
            if (opcode < 0)
               continue;
            if (format == Format::VOP1)
               result[generation].emplace(opcode_key(Format::VOP3, opcode + 0x180), (aco_opcode)i);
            else if (format == Format::VOP2)
               result[generation].emplace(opcode_key(Format::VOP3, opcode + 0x100), (aco_opcode)i);
            else if (format == Format::VOPC)
               result[generation].emplace(opcode_key(Format::VOP3, opcode), (aco_opcode)i);
         }
      }
      return result;
   }();
   amd_gfx_level level = ctx.program->gfx_level;
   const auto &map = maps[level >= GFX12 ? 2 : level >= GFX11 ? 1 : 0];
   if (inst->family == AGC_ISA_EXP)
      return aco_opcode::exp;
   auto found = map.find(opcode_key(instruction_format(inst), inst->opcode));
   if (found == map.end())
      reject_unsupported(inst->pc, "no ACO opcode for this target encoding");
   return found->second;
}

static PhysReg
phys_of(const agc_isa_inst *inst, agc_isa_operand op)
{
   switch (op.kind) {
   case AGC_ISA_OPERAND_SGPR:
      return PhysReg(op.reg);
   case AGC_ISA_OPERAND_VGPR:
      return PhysReg(256u + op.reg);
   case AGC_ISA_OPERAND_VCC_LO:
      return vcc;
   case AGC_ISA_OPERAND_VCC_HI:
      return vcc_hi;
   case AGC_ISA_OPERAND_TTMP:
      return PhysReg(108u + op.reg);
   case AGC_ISA_OPERAND_M0:
      return m0;
   case AGC_ISA_OPERAND_NULL:
      return sgpr_null;
   case AGC_ISA_OPERAND_EXEC_LO:
      return exec_lo;
   case AGC_ISA_OPERAND_EXEC_HI:
      return exec_hi;
   case AGC_ISA_OPERAND_SCC:
      return scc;
   case AGC_ISA_OPERAND_RAW:
      return PhysReg(op.value);
   case AGC_ISA_OPERAND_SHARED_BASE:
      return PhysReg(235);
   case AGC_ISA_OPERAND_SHARED_LIMIT:
      return PhysReg(236);
   case AGC_ISA_OPERAND_PRIVATE_BASE:
      return PhysReg(237);
   case AGC_ISA_OPERAND_PRIVATE_LIMIT:
      return PhysReg(238);
   case AGC_ISA_OPERAND_POPS_EXITING_WAVE_ID:
      return PhysReg(239);
   case AGC_ISA_OPERAND_VCC_Z:
      return PhysReg(251);
   case AGC_ISA_OPERAND_EXEC_Z:
      return PhysReg(252);
   case AGC_ISA_OPERAND_LDS_DIRECT:
      return PhysReg(254);
   default:
      reject(inst->pc, "operand is not a physical register");
   }
}

static bool
constant_of(agc_isa_operand op, Operand *out)
{
   switch (op.kind) {
   case AGC_ISA_OPERAND_INLINE_INT:
      *out = Operand::c32((uint32_t)(int32_t)op.inline_int);
      return true;
   case AGC_ISA_OPERAND_INLINE_FLOAT: {
      uint32_t bits;
      if (!agc_isa_inline_float_bits((uint8_t)op.inline_float, &bits))
         return false;
      *out = imm(bits);
      return true;
   }
   case AGC_ISA_OPERAND_LITERAL:
      *out = Operand::literal32(op.literal);
      return true;
   default:
      return false;
   }
}

static RegClass
reg_class_of(agc_isa_operand op, unsigned dwords)
{
   bool vgpr = op.kind == AGC_ISA_OPERAND_VGPR;
   return RegClass(vgpr ? RegType::vgpr : RegType::sgpr, dwords ? dwords : 1);
}

Operand
src(const agc_isa_inst *inst, agc_isa_operand op, unsigned dwords)
{
   Operand constant;
   if (constant_of(op, &constant))
      return constant;
   return Operand(phys_of(inst, op), reg_class_of(op, dwords));
}

Definition
dst(const agc_isa_inst *inst, agc_isa_operand op, unsigned dwords)
{
   return Definition(phys_of(inst, op), reg_class_of(op, dwords));
}

Operand
mask_op(const lower_ctx &ctx, PhysReg reg)
{
   return Operand(reg, ctx.program->lane_mask);
}

Definition
mask_def(const lower_ctx &ctx, PhysReg reg)
{
   return Definition(reg, ctx.program->lane_mask);
}

static unsigned
aco_type_dwords(aco_type t)
{
   unsigned dw = t.dwords();
   return dw ? dw : 1;
}

static void
collect_alu_sources(const agc_isa_inst *inst, agc_isa_operand *sources, unsigned *count, agc_isa_operand *dest,
                    bool *has_dest)
{
   *has_dest = false;
   *count = 0;
   switch (inst->fields) {
   case AGC_ISA_SOP1:
      *dest = inst->sop1.sdst;
      *has_dest = true;
      sources[(*count)++] = inst->sop1.src0;
      break;
   case AGC_ISA_SOP2:
      *dest = inst->sop2.sdst;
      *has_dest = true;
      sources[(*count)++] = inst->sop2.src0;
      sources[(*count)++] = inst->sop2.src1;
      break;
   case AGC_ISA_SOPC:
      sources[(*count)++] = inst->sopc.src0;
      sources[(*count)++] = inst->sopc.src1;
      break;
   case AGC_ISA_SOPK:
      *dest = inst->sopk.sdst;
      *has_dest = true;
      sources[(*count)++] = inst->sopk.has_imm32 ? agc_isa_operand_literal(inst->sopk.imm32) : inst->sopk.sdst;
      break;
   case AGC_ISA_SOPP:
      break;
   case AGC_ISA_VOP1:
      *dest = inst->vop1.vdst;
      *has_dest = true;
      sources[(*count)++] = inst->vop1.src0;
      break;
   case AGC_ISA_VOP2:
      *dest = inst->vop2.vdst;
      *has_dest = true;
      sources[(*count)++] = inst->vop2.src0;
      sources[(*count)++] = inst->vop2.vsrc1;
      break;
   case AGC_ISA_VOPC:
      sources[(*count)++] = inst->vopc.src0;
      sources[(*count)++] = inst->vopc.vsrc1;
      break;
   case AGC_ISA_VOP3:
      *dest = inst->vop3.vdst;
      *has_dest = true;
      for (unsigned i = 0; i < 3; i++)
         sources[(*count)++] = inst->vop3.src[i];
      break;
   case AGC_ISA_VOP3P:
      *dest = inst->vop3p.vdst;
      *has_dest = true;
      for (unsigned i = 0; i < 3; i++)
         sources[(*count)++] = inst->vop3p.src[i];
      break;
   default:
      reject_unsupported(inst->pc, "instruction family is not an ALU family");
   }
}

static Instruction *
build_alu(const lower_ctx &ctx, const agc_isa_inst *inst, aco_opcode op)
{
   const aco_alu_opcode_info &ai = instr_info.alu_opcode_infos[(int)op];
   Format fmt = instr_info.format[(int)op];
   RegClass lane_mask = ctx.program->lane_mask;

   agc_isa_operand sources[4];
   agc_isa_operand dest;
   unsigned source_count;
   bool has_dest;
   collect_alu_sources(inst, sources, &source_count, &dest, &has_dest);

   if (inst->fields == AGC_ISA_VOP3 && fmt != Format::VOP3)
      fmt = asVOP3(fmt);
   const agc_isa_vop_ext *extension = NULL;
   if (inst->fields == AGC_ISA_VOP1)
      extension = &inst->vop1.ext;
   else if (inst->fields == AGC_ISA_VOP2)
      extension = &inst->vop2.ext;
   else if (inst->fields == AGC_ISA_VOPC)
      extension = &inst->vopc.ext;
   if (extension) {
      if (extension->kind == AGC_ISA_VOP_EXT_DPP16)
         fmt = (Format)((unsigned)fmt | (unsigned)Format::DPP16);
      else if (extension->kind == AGC_ISA_VOP_EXT_DPP8)
         fmt = (Format)((unsigned)fmt | (unsigned)Format::DPP8);
      else if (extension->kind == AGC_ISA_VOP_EXT_SDWA)
         fmt = asSDWA(fmt);
   }
   Instruction *instr = create_instruction(op, fmt, ai.num_operands, ai.num_defs);

   unsigned next_source = 0;
   bool used_dest = false;
   for (unsigned i = 0; i < ai.num_operands; i++) {
      fixed_reg fr = ai.op_fixed_reg[i];
      aco_type t = ai.op_types[i];
      if (fr == fixed_scc) {
         instr->operands[i] = Operand(scc, s1);
      } else if (fr == fixed_exec) {
         instr->operands[i] = Operand(exec, t.base_type == aco_base_type_lanemask
                                               ? lane_mask
                                               : RegClass(RegType::sgpr, aco_type_dwords(t)));
      } else if (fr == fixed_m0) {
         instr->operands[i] = Operand(m0, s1);
      } else if (fr == fixed_imm) {
         if (inst->fields == AGC_ISA_VOP2 && inst->vop2.has_constant)
            instr->operands[i] = Operand::literal32(inst->vop2.constant);
         else if (inst->fields == AGC_ISA_SOPK && inst->sopk.has_imm32)
            instr->operands[i] = Operand::literal32(inst->sopk.imm32);
         else
            reject(inst->pc, "instruction is missing its immediate operand");
      } else if (t.base_type == aco_base_type_lanemask && inst->fields != AGC_ISA_VOP3) {
         instr->operands[i] = Operand(vcc, lane_mask);
      } else {
         agc_isa_operand source;
         if (next_source < source_count && sources[next_source].kind != AGC_ISA_OPERAND_NONE)
            source = sources[next_source++];
         else if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_READS_DST))
            source = dest;
         else
            reject(inst->pc, "ALU instruction is missing a source operand");
         unsigned dwords = t.base_type == aco_base_type_lanemask ? lane_mask.size() : aco_type_dwords(t);
         instr->operands[i] = src(inst, source, dwords);
      }
   }

   for (unsigned i = 0; i < ai.num_defs; i++) {
      fixed_reg fr = ai.def_fixed_reg[i];
      aco_type t = ai.def_types[i];
      if (fr == fixed_scc) {
         instr->definitions[i] = Definition(scc, s1);
      } else if (fr == fixed_exec) {
         instr->definitions[i] = Definition(exec, t.base_type == aco_base_type_lanemask
                                                     ? lane_mask
                                                     : RegClass(RegType::sgpr, aco_type_dwords(t)));
      } else if (fr == fixed_m0) {
         instr->definitions[i] = Definition(m0, s1);
      } else if (t.base_type == aco_base_type_lanemask) {
         agc_isa_operand destination = agc_isa_operand_special(AGC_ISA_OPERAND_VCC_LO);
         if (inst->fields == AGC_ISA_VOP3)
            destination = i == 0 ? dest : inst->vop3.sdst;
         else if (inst->fields == AGC_ISA_VOPC)
            destination = inst->vopc.sdst;
         instr->definitions[i] = Definition(phys_of(inst, destination), lane_mask);
      } else {
         if (!has_dest || used_dest)
            reject(inst->pc, "ALU instruction is missing a destination");
         instr->definitions[i] = dst(inst, dest, aco_type_dwords(t));
         used_dest = true;
      }
   }

   if (inst->fields == AGC_ISA_VOP3) {
      auto &valu = instr->valu();
      valu.neg = inst->vop3.neg;
      valu.abs = inst->vop3.abs;
      valu.opsel = inst->vop3.opsel;
      valu.clamp = inst->vop3.clamp;
      valu.omod = inst->vop3.omod;
   } else if (inst->fields == AGC_ISA_VOP3P) {
      auto &valu = instr->valu();
      valu.neg_lo = inst->vop3p.neg;
      valu.neg_hi = inst->vop3p.neg_hi;
      valu.opsel_lo = inst->vop3p.opsel;
      valu.opsel_hi = inst->vop3p.opsel_hi;
      valu.clamp = inst->vop3p.clamp;
   }
   if (extension && extension->kind == AGC_ISA_VOP_EXT_DPP16) {
      const auto &source = extension->dpp16;
      auto &dpp = instr->dpp16();
      dpp.dpp_ctrl = source.ctrl;
      dpp.fetch_inactive = source.fetch_inactive;
      dpp.bound_ctrl = source.bound_ctrl;
      dpp.bank_mask = source.bank_mask;
      dpp.row_mask = source.row_mask;
      dpp.neg[0] = source.src0_neg;
      dpp.neg[1] = source.src1_neg;
      dpp.abs[0] = source.src0_abs;
      dpp.abs[1] = source.src1_abs;
   } else if (extension && extension->kind == AGC_ISA_VOP_EXT_DPP8) {
      instr->dpp8().lane_sel = extension->dpp8.selects;
      instr->dpp8().fetch_inactive = extension->dpp8.fetch_inactive;
   } else if (extension && extension->kind == AGC_ISA_VOP_EXT_SDWA) {
      const auto &source = extension->sdwa;
      auto &sdwa = instr->sdwa();
      auto select = [](uint8_t value, bool sign) {
         unsigned size = value < 4 ? 1 : value < 6 ? 2 : 4;
         unsigned offset = value < 4 ? value : value < 6 ? (value - 4) * 2 : 0;
         return SubdwordSel(size, offset, sign);
      };
      sdwa.sel[0] = select(source.src0_sel, source.src0_sext);
      sdwa.sel[1] = select(source.src1_sel, source.src1_sext);
      sdwa.dst_sel = select(source.dst_sel, source.dst_unused == AGC_ISA_SDWA_UNUSED_SEXT);
      sdwa.clamp = source.clamp;
      sdwa.omod = source.omod;
      sdwa.neg[0] = source.src0_neg;
      sdwa.neg[1] = source.src1_neg;
      sdwa.abs[0] = source.src0_abs;
      sdwa.abs[1] = source.src1_abs;
      if (source.dst_unused == AGC_ISA_SDWA_UNUSED_PRESERVE && !instr->isVOPC()) {
         unsigned size = sdwa.dst_sel.size();
         PhysReg destination = instr->definitions[0].physReg().advance(sdwa.dst_sel.offset());
         instr->definitions[0] = Definition(destination, RegClass::get(RegType::vgpr, size));
         sdwa.dst_sel = SubdwordSel(size, 0, false);
      }
   }
   if (inst->fields == AGC_ISA_SOPK)
      instr->salu().imm = inst->sopk.simm16;
   else if (inst->fields == AGC_ISA_SOPP)
      instr->salu().imm = inst->sopp.simm16;
   return instr;
}

static ac_hw_cache_flags
cache_flags(const lower_ctx &ctx, bool glc, bool slc, bool dlc, bool atomic = false)
{
   ac_hw_cache_flags flags = {};
   if (ctx.program->gfx_level >= GFX12) {
      flags.gfx12.scope = atomic || glc ? gfx12_scope_device : gfx12_scope_cu;
      flags.gfx12.temporal_hint =
         atomic ? (glc ? gfx12_atomic_return : 0) : (slc ? gfx12_load_non_temporal : gfx12_load_regular_temporal);
      return flags;
   }
   if (glc)
      flags.value |= ac_glc;
   if (slc)
      flags.value |= ac_slc;
   if (dlc)
      flags.value |= ac_dlc;
   return flags;
}

static Instruction *
build_smem(const lower_ctx &ctx, const agc_isa_inst *inst, aco_opcode op)
{
   const agc_isa_smem *s = &inst->smem;
   const char *name = instr_info.name[(unsigned)op];
   bool address = agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_SMEM_ADDR) || strstr(name, "_load") ||
                  strstr(name, "_store") || strstr(name, "_atomic") || strstr(name, "_probe");
   bool store = strstr(name, "_store");
   bool has_offset = s->soffset.kind != AGC_ISA_OPERAND_NULL && s->soffset.kind != AGC_ISA_OPERAND_NONE;
   unsigned operands = address ? 2 + has_offset : 0;
   Instruction *instr = create_instruction(op, Format::SMEM, operands + store, s->dwords && !store);
   if (address) {
      instr->operands[0] = Operand(PhysReg(s->sbase), s->buffer ? s4 : s2);
      instr->operands[1] = Operand::c32((uint32_t)s->offset);
      if (has_offset) {
         Operand offset = src(inst, s->soffset, 1);
         if (offset.isConstant() || offset.regClass().type() != RegType::sgpr || offset.physReg().reg() >= 128)
            reject(inst->pc, "SMEM offset must be a physical scalar register");
         instr->operands[store ? 3 : 2] = offset;
      }
   }
   if (s->dwords) {
      RegClass data(RegType::sgpr, s->dwords);
      if (store)
         instr->operands[2] = Operand(phys_of(inst, s->sdata), data);
      else
         instr->definitions[0] = Definition(phys_of(inst, s->sdata), data);
   }
   instr->smem().sync = memory_sync_info(storage_buffer);
   instr->smem().cache = cache_flags(ctx, s->glc, false, s->dlc);
   return instr;
}

static Instruction *
build_mubuf(const lower_ctx &ctx, const agc_isa_inst *inst, aco_opcode op)
{
   const agc_isa_buffer *b = &inst->buffer;
   bool atomic = b->kind == AGC_ISA_MEM_ATOMIC;
   bool write = b->kind == AGC_ISA_MEM_STORE || atomic;
   bool result = !b->lds && (b->kind == AGC_ISA_MEM_LOAD || (atomic && b->glc));
   bool typed = inst->fields == AGC_ISA_MTBUF;
   unsigned data_words = b->data_dwords ? b->data_dwords : 1;
   unsigned result_words =
      atomic && agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CMPSWAP) ? data_words / 2 : data_words;
   Instruction *instr = create_instruction(op, typed ? Format::MTBUF : Format::MUBUF, 3 + write + b->lds, result);
   instr->operands[0] = Operand(PhysReg(b->srsrc), s4);
   if (b->offen || b->idxen)
      instr->operands[1] = Operand(PhysReg(256u + b->vaddr), RegClass(RegType::vgpr, b->offen + b->idxen));
   else
      instr->operands[1] = Operand(v1);
   if (b->soffset.kind == AGC_ISA_OPERAND_NONE || b->soffset.kind == AGC_ISA_OPERAND_NULL)
      instr->operands[2] = Operand::c32(0);
   else
      instr->operands[2] = src(inst, b->soffset, 1);
   if (write)
      instr->operands[3] = Operand(PhysReg(256u + b->vdata), RegClass(RegType::vgpr, data_words));
   if (b->lds)
      instr->operands.back() = Operand(m0, s1);
   if (result)
      instr->definitions[0] = Definition(PhysReg(256u + b->vdata), RegClass(RegType::vgpr, result_words + b->tfe));
   auto sync = memory_sync_info(storage_buffer | (b->lds ? storage_shared : 0), atomic ? semantic_atomicrmw : 0);
   auto cache = cache_flags(ctx, b->glc, b->slc, b->dlc, atomic);
   if (typed) {
      auto &m = instr->mtbuf();
      bool found = false;
      static const uint8_t valid_nfmt[] = {0x10, 0x3f, 0xbf, 0x3f, 0xb0, 0xbf, 0xbf, 0,
                                           0,    0x3f, 0x3f, 0xb0, 0xbf, 0xb0, 0xb0};
      for (unsigned dfmt = 0; dfmt < sizeof(valid_nfmt) && !found; dfmt++) {
         for (unsigned nfmt = 0; nfmt < 8; nfmt++) {
            if (!(valid_nfmt[dfmt] & (1u << nfmt)) || (ctx.program->gfx_level >= GFX11 && dfmt == 6 && nfmt != 7))
               continue;
            if (ac_get_tbuffer_format(ctx.program->gfx_level, dfmt, nfmt) == b->format) {
               m.dfmt = dfmt;
               m.nfmt = nfmt;
               found = true;
               break;
            }
         }
      }
      if (!found)
         reject(inst->pc, "unknown typed buffer format");
      m.offset = b->offset;
      m.offen = b->offen;
      m.idxen = b->idxen;
      m.tfe = b->tfe;
      m.sync = sync;
      m.cache = cache;
   } else {
      auto &m = instr->mubuf();
      m.offset = b->offset;
      m.offen = b->offen;
      m.idxen = b->idxen;
      m.tfe = b->tfe;
      m.lds = b->lds;
      m.sync = sync;
      m.cache = cache;
   }
   return instr;
}

static Instruction *
build_flat(const lower_ctx &ctx, const agc_isa_inst *inst, aco_opcode op)
{
   const auto &f = inst->flat;
   bool atomic = f.kind == AGC_ISA_MEM_ATOMIC;
   bool write = f.kind == AGC_ISA_MEM_STORE || atomic;
   bool result = !f.lds && (f.kind == AGC_ISA_MEM_LOAD || (atomic && f.glc));
   unsigned words = f.data_dwords ? f.data_dwords : 1;
   unsigned result_words = atomic && agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CMPSWAP) ? words / 2 : words;
   Instruction *instr = create_instruction(op, instruction_format(inst), 2 + write, result);
   unsigned address_words =
      f.segment == AGC_ISA_SEGMENT_FLAT ||
            (f.segment == AGC_ISA_SEGMENT_GLOBAL && f.saddr.kind == AGC_ISA_OPERAND_NULL)
         ? 2
         : 1;
   instr->operands[0] =
      f.uses_vaddr ? Operand(PhysReg(256u + f.vaddr), RegClass(RegType::vgpr, address_words)) : Operand(v1);
   bool has_base = f.saddr.kind != AGC_ISA_OPERAND_NULL && f.saddr.kind != AGC_ISA_OPERAND_NONE;
   if (has_base && instr->format == Format::FLAT)
      reject_unsupported(inst->pc, "a flat access cannot take a scalar base");
   instr->operands[1] = has_base ? Operand(phys_of(inst, f.saddr), s2) : Operand(s2);
   if (write)
      instr->operands[2] = Operand(PhysReg(256u + f.vdata), RegClass(RegType::vgpr, words));
   if (result)
      instr->definitions[0] = Definition(PhysReg(256u + f.vdst), RegClass(RegType::vgpr, result_words));
   auto &flat = instr->flatlike();
   flat.offset = ctx.program->gfx_level < GFX11 && f.segment == AGC_ISA_SEGMENT_FLAT ? 0 : f.offset;
   flat.lds = f.lds;
   flat.sync = memory_sync_info(f.segment == AGC_ISA_SEGMENT_SCRATCH ? storage_scratch : storage_buffer,
                                atomic ? semantic_atomicrmw : 0);
   flat.cache = cache_flags(ctx, f.glc, f.slc, f.dlc, atomic);
   return instr;
}

static Instruction *
build_ds(const agc_isa_inst *inst, aco_opcode op)
{
   const auto &d = inst->ds;
   unsigned operands = 1 + !!d.data0_dwords + !!d.data1_dwords + d.gds;
   Instruction *instr = create_instruction(op, Format::DS, operands, !!d.dst_dwords);
   instr->operands[0] = d.uses_addr ? Operand(PhysReg(256u + d.addr), v1) : Operand(v1);
   unsigned next = 1;
   if (d.data0_dwords)
      instr->operands[next++] = Operand(PhysReg(256u + d.data0), RegClass(RegType::vgpr, d.data0_dwords));
   if (d.data1_dwords)
      instr->operands[next++] = Operand(PhysReg(256u + d.data1), RegClass(RegType::vgpr, d.data1_dwords));
   if (d.gds)
      instr->operands[next++] = Operand(m0, s1);
   if (d.dst_dwords)
      instr->definitions[0] = Definition(PhysReg(256u + d.vdst), RegClass(RegType::vgpr, d.dst_dwords));
   auto &ds = instr->ds();
   ds.gds = d.gds;
   ds.offset0 = d.offset0;
   ds.offset1 = d.offset1;
   ds.sync = memory_sync_info(d.gds ? storage_gds : storage_shared);
   return instr;
}

static Instruction *
build_image(const lower_ctx &ctx, const agc_isa_inst *inst, aco_opcode op)
{
   const auto &m = inst->mimg;
   bool atomic = m.kind == AGC_ISA_MEM_ATOMIC;
   bool write = m.kind == AGC_ISA_MEM_STORE || atomic;
   bool result = m.kind == AGC_ISA_MEM_LOAD || (atomic && m.glc);
   unsigned words = m.data_dwords ? m.data_dwords : 1;
   unsigned result_words = atomic && agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CMPSWAP) ? words / 2 : words;
   unsigned max_slots = ctx.program->gfx_level >= GFX11 ? 5 : 13;
   if (ctx.program->gfx_level >= GFX12 && (m.uses_sampler || op == aco_opcode::image_msaa_load))
      max_slots = 4;
   unsigned widths[AGC_ISA_MAX_NSA_ADDRESSES];
   unsigned width_count = 0;
   bool ray = op == aco_opcode::image_bvh_intersect_ray || op == aco_opcode::image_bvh64_intersect_ray;
   if (ctx.program->gfx_level >= GFX11 && ray) {
      widths[width_count++] = op == aco_opcode::image_bvh64_intersect_ray ? 2u : 1u;
      widths[width_count++] = 1;
      widths[width_count++] = 3;
      widths[width_count++] = 3;
      if (!m.a16)
         widths[width_count++] = 3;
   } else {
      unsigned addresses = m.nsa ? std::min<unsigned>(m.addr_dwords, max_slots) : 1;
      for (unsigned i = 0; i < addresses; i++)
         widths[width_count++] = 1;
      if (addresses)
         widths[width_count - 1] = m.addr_dwords - addresses + 1;
   }
   if (!m.addr_dwords || !width_count || m.addr_dwords > AGC_ISA_MAX_NSA_ADDRESSES)
      reject(inst->pc, "invalid image address operands");
   Instruction *instr = create_instruction(op, Format::MIMG, 3 + width_count, result);
   instr->operands[0] = Operand(PhysReg(m.srsrc), m.r128 ? s4 : s8);
   instr->operands[1] = m.uses_sampler ? Operand(PhysReg(m.ssamp), s4) : Operand(s4);
   instr->operands[2] = write ? Operand(PhysReg(256u + m.vdata), RegClass(RegType::vgpr, words)) : Operand(v1);
   unsigned next_address = 0;
   for (unsigned i = 0; i < width_count; i++) {
      unsigned width = widths[i];
      if (next_address + width > m.addr_dwords)
         reject(inst->pc, "an image address group exceeds its operand count");
      unsigned first = m.nsa ? m.addr[next_address] : m.addr[0] + next_address;
      if (m.nsa) {
         for (unsigned k = 1; k < width; k++) {
            if (m.addr[next_address + k] != first + k)
               reject(inst->pc, "an image address group is not in consecutive VGPRs");
         }
      }
      instr->operands[3 + i] = Operand(PhysReg(256u + first), RegClass(RegType::vgpr, width));
      next_address += width;
   }
   if (next_address != m.addr_dwords)
      reject(inst->pc, "image address groups do not cover every operand");
   if (result)
      instr->definitions[0] = Definition(PhysReg(256u + m.vdata), RegClass(RegType::vgpr, result_words + m.tfe));
   auto &image = instr->mimg();
   image.dim = m.dim;
   image.dmask = m.dmask;
   image.unrm = m.unorm;
   image.tfe = m.tfe;
   image.lwe = m.lwe;
   image.r128 = m.r128;
   image.a16 = m.a16;
   image.d16 = m.d16;
   image.sync = memory_sync_info(storage_image, atomic ? semantic_atomicrmw : 0);
   image.cache = cache_flags(ctx, m.glc, m.slc, m.dlc, atomic);
   return instr;
}

static Instruction *
build_export(const agc_isa_inst *inst)
{
   const auto &e = inst->exp;
   Instruction *instr = create_instruction(aco_opcode::exp, Format::EXP, 4, 0);
   for (unsigned i = 0; i < 4; i++)
      instr->operands[i] = Operand(PhysReg(256u + e.vsrc[i]), v1);
   auto &exp = instr->exp();
   exp.enabled_mask = e.en;
   exp.dest = e.target;
   exp.compressed = e.compr;
   exp.done = e.done;
   exp.valid_mask = e.vm;
   exp.row_en = e.row_en;
   return instr;
}

static Instruction *
build_interp(const agc_isa_inst *inst, aco_opcode op)
{
   if (inst->fields == AGC_ISA_LDSDIR) {
      Instruction *instr = create_instruction(op, Format::LDSDIR, 1, 1);
      instr->operands[0] = Operand(m0, s1);
      instr->definitions[0] = Definition(PhysReg(256u + inst->ldsdir.vdst), v1);
      auto &dir = instr->ldsdir();
      dir.attr = inst->ldsdir.attr;
      dir.attr_chan = inst->ldsdir.chan;
      dir.wait_vdst = inst->ldsdir.wait_vdst;
      dir.wait_vsrc = 1;
      dir.sync = memory_sync_info(storage_shared);
      return instr;
   }
   if (inst->fields == AGC_ISA_VINTERP) {
      Instruction *instr = create_instruction(op, Format::VINTERP_INREG, 3, 1);
      for (unsigned i = 0; i < 3; i++)
         instr->operands[i] = src(inst, inst->vinterp.src[i], 1);
      instr->definitions[0] = Definition(PhysReg(256u + inst->vinterp.vdst), v1);
      auto &interp = instr->vinterp_inreg();
      interp.neg = inst->vinterp.neg;
      interp.opsel = inst->vinterp.opsel;
      interp.clamp = inst->vinterp.clamp;
      interp.wait_exp = inst->vinterp.wait_exp;
      return instr;
   }
   bool mov = op == aco_opcode::v_interp_mov_f32;
   bool p2 = op == aco_opcode::v_interp_p2_f32;
   Instruction *instr = create_instruction(op, Format::VINTRP, 2 + p2, 1);
   instr->operands[0] = mov ? Operand::c32(inst->vintrp.vsrc) : Operand(PhysReg(256u + inst->vintrp.vsrc), v1);
   instr->operands[1] = Operand(m0, s1);
   if (p2)
      instr->operands[2] = Operand(PhysReg(256u + inst->vintrp.vdst), v1);
   instr->definitions[0] = Definition(PhysReg(256u + inst->vintrp.vdst), v1);
   instr->vintrp().attribute = inst->vintrp.attr;
   instr->vintrp().component = inst->vintrp.chan;
   return instr;
}

void
wait_tracker_init(wait_tracker &tracker, bool program_entry)
{
   memset(&tracker, 0, sizeof(tracker));
   for (bool &prefix : tracker.vm_prefix)
      prefix = !program_entry;
   for (bool &prefix : tracker.lgkm_prefix)
      prefix = !program_entry;
}

static void
push_vm(wait_tracker &tracker, enum agc_isa_wait_vm_type type)
{
   tracker.vm[tracker.vm_count++] = (uint8_t)type;
   if (tracker.vm_count > AGC_ISA_WAIT_QUEUE_LIMIT) {
      memmove(tracker.vm, tracker.vm + 1, AGC_ISA_WAIT_QUEUE_LIMIT);
      tracker.vm_count = AGC_ISA_WAIT_QUEUE_LIMIT;
      memset(tracker.vm_prefix, 0, sizeof(tracker.vm_prefix));
   }
}

static void
push_lgkm(wait_tracker &tracker, enum agc_isa_wait_lgkm_type type)
{
   tracker.lgkm[tracker.lgkm_count++] = (uint8_t)type;
   if (tracker.lgkm_count > AGC_ISA_WAIT_QUEUE_LIMIT) {
      memmove(tracker.lgkm, tracker.lgkm + 1, AGC_ISA_WAIT_QUEUE_LIMIT);
      tracker.lgkm_count = AGC_ISA_WAIT_QUEUE_LIMIT;
      memset(tracker.lgkm_prefix, 0, sizeof(tracker.lgkm_prefix));
   }
}

static void
wait_tracker_record(lower_ctx &ctx, const Instruction *instr)
{
   if (!ctx.gfx12)
      return;
   wait_tracker &tracker = ctx.waits;
   bool loads = !instr->definitions.empty();
   if (instr->isMIMG()) {
      if (!loads)
         return;
      enum agc_isa_wait_vm_type type = AGC_ISA_WAIT_VM_LOAD;
      if (instr->opcode == aco_opcode::image_bvh_intersect_ray ||
          instr->opcode == aco_opcode::image_bvh64_intersect_ray)
         type = AGC_ISA_WAIT_VM_BVH;
      else if (!instr->operands[1].isUndefined() || instr->opcode == aco_opcode::image_msaa_load)
         type = AGC_ISA_WAIT_VM_SAMPLE;
      push_vm(tracker, type);
   } else if (instr->isMUBUF() || instr->isMTBUF()) {
      if (loads)
         push_vm(tracker, AGC_ISA_WAIT_VM_LOAD);
   } else if (instr->isFlatLike()) {
      if (loads) {
         push_vm(tracker, AGC_ISA_WAIT_VM_LOAD);
         if (instr->isFlat())
            push_lgkm(tracker, AGC_ISA_WAIT_LGKM_DS);
      }
   } else if (instr->isDS()) {
      push_lgkm(tracker, AGC_ISA_WAIT_LGKM_DS);
   } else if (instr->isSMEM()) {
      if (loads || strstr(instr_info.name[(unsigned)instr->opcode], "_store"))
         push_lgkm(tracker, AGC_ISA_WAIT_LGKM_KM);
   } else if (instr->opcode == aco_opcode::s_sendmsg_rtn_b32 || instr->opcode == aco_opcode::s_sendmsg_rtn_b64) {
      push_lgkm(tracker, AGC_ISA_WAIT_LGKM_KM);
   }
}

static void
vm_waits(lower_ctx &ctx, uint8_t n)
{
   static const aco_opcode opcodes[3] = {
      aco_opcode::s_wait_loadcnt,
      aco_opcode::s_wait_samplecnt,
      aco_opcode::s_wait_bvhcnt,
   };
   wait_tracker &tracker = ctx.waits;
   unsigned total = tracker.vm_count;
   unsigned split = total - (n < total ? n : total);
   for (unsigned type = 0; type < 3; type++) {
      bool older = false;
      for (unsigned i = 0; i < split && !older; i++)
         older = tracker.vm[i] == type;
      if (!older && !tracker.vm_prefix[type])
         continue;
      unsigned younger = 0;
      for (unsigned i = split; i < total; i++)
         younger += tracker.vm[i] == type;
      ctx.bld.sopp(opcodes[type], (uint8_t)younger);
      tracker.vm_prefix[type] = false;
   }
   memmove(tracker.vm, tracker.vm + split, total - split);
   tracker.vm_count = (uint8_t)(total - split);
}

static void
lgkm_waits(lower_ctx &ctx, uint8_t n)
{
   wait_tracker &tracker = ctx.waits;
   unsigned total = tracker.lgkm_count;
   unsigned split = total - (n < total ? n : total);
   bool older_ds = false;
   bool older_km = false;
   for (unsigned i = 0; i < split; i++) {
      older_ds |= tracker.lgkm[i] == AGC_ISA_WAIT_LGKM_DS;
      older_km |= tracker.lgkm[i] == AGC_ISA_WAIT_LGKM_KM;
   }
   if (older_ds || tracker.lgkm_prefix[AGC_ISA_WAIT_LGKM_DS]) {
      unsigned younger = 0;
      for (unsigned i = split; i < total; i++)
         younger += tracker.lgkm[i] == AGC_ISA_WAIT_LGKM_DS;
      ctx.bld.sopp(aco_opcode::s_wait_dscnt, (uint8_t)younger);
      tracker.lgkm_prefix[AGC_ISA_WAIT_LGKM_DS] = false;
   }
   memmove(tracker.lgkm, tracker.lgkm + split, total - split);
   tracker.lgkm_count = (uint8_t)(total - split);
   if (older_km || tracker.lgkm_prefix[AGC_ISA_WAIT_LGKM_KM]) {
      ctx.bld.sopp(aco_opcode::s_wait_kmcnt, 0);
      tracker.lgkm_prefix[AGC_ISA_WAIT_LGKM_KM] = false;
      unsigned kept = 0;
      for (unsigned i = 0; i < tracker.lgkm_count; i++) {
         if (tracker.lgkm[i] != AGC_ISA_WAIT_LGKM_KM)
            tracker.lgkm[kept++] = tracker.lgkm[i];
      }
      tracker.lgkm_count = (uint8_t)kept;
   }
}

void
gfx12_waits(lower_ctx &ctx, bool has_vm, uint8_t vm, bool has_exp, uint8_t exp, bool has_lgkm, uint8_t lgkm,
            bool has_vs, uint8_t vs)
{
   flush(ctx);
   if (has_vm && vm < AGC_ISA_GFX10_VMCNT_MAX)
      vm_waits(ctx, vm);
   if (has_exp && exp < AGC_ISA_GFX10_EXPCNT_MAX)
      ctx.bld.sopp(aco_opcode::s_wait_expcnt, exp);
   if (has_lgkm && lgkm < AGC_ISA_GFX10_LGKMCNT_MAX)
      lgkm_waits(ctx, lgkm);
   if (has_vs && vs < AGC_ISA_GFX10_VSCNT_MAX)
      ctx.bld.sopp(aco_opcode::s_wait_storecnt, vs);
}

void
waitcnt(lower_ctx &ctx, uint8_t vm, uint8_t exp, uint8_t lgkm)
{
   if (ctx.gfx12)
      gfx12_waits(ctx, true, vm, true, exp, true, lgkm, false, 0);
   else if (ctx.gfx11_plus)
      ctx.bld.sopp(aco_opcode::s_waitcnt, agc_isa_waitcnt_encode_gfx11(vm, exp, lgkm));
   else
      ctx.bld.sopp(aco_opcode::s_waitcnt, agc_isa_waitcnt_encode_gfx10(vm, exp, lgkm));
}

static void
descriptor_cache_invalidate(lower_ctx &ctx, const Instruction *instr)
{
   agc_isa_descriptor_cache *cache = &ctx.descriptor_cache;
   if (!cache->has_vsharp && !cache->has_tsharp && !cache->has_ssharp)
      return;
   for (const Definition &def : instr->definitions) {
      unsigned first = def.physReg().reg();
      if (def.regClass().type() != RegType::sgpr || first >= AGC_ISA_SGPR_LIMIT)
         continue;
      unsigned end = first + def.size();
      auto writes = [&](unsigned base, unsigned count) { return first < base + count && base < end; };
      if (cache->has_vsharp && writes(cache->vsharp, 4)) {
         cache->has_vsharp = false;
         cache->vsharp = 0;
      }
      if (cache->has_tsharp && writes(cache->tsharp, cache->tsharp_r128 ? 4 : 8)) {
         cache->has_tsharp = false;
         cache->tsharp = 0;
         cache->tsharp_r128 = false;
         cache->tsharp_stored = false;
      }
      if (cache->has_ssharp && writes(cache->ssharp, 4)) {
         cache->has_ssharp = false;
         cache->ssharp = 0;
      }
   }
}

static void
check_alu(const lower_ctx &ctx, const Instruction *instr)
{
   if (!instr->isSALU() && !instr->isVALU())
      return;
   if (instr->isSOPP() && !instr->operands.empty())
      return;
   const aco_alu_opcode_info &ai = instr_info.alu_opcode_infos[(int)instr->opcode];
   bool ok = instr->operands.size() == ai.num_operands && instr->definitions.size() == ai.num_defs;
   RegClass lane_mask = ctx.program->lane_mask;
   for (unsigned i = 0; ok && i < ai.num_operands; i++) {
      const Operand &op = instr->operands[i];
      fixed_reg fr = ai.op_fixed_reg[i];
      aco_type t = ai.op_types[i];
      if (fr == fixed_scc)
         ok = op.physReg() == scc;
      else if (fr == fixed_exec)
         ok = op.physReg() == exec;
      else if (fr == fixed_m0)
         ok = op.physReg() == m0;
      else if (fr == fixed_imm)
         ok = op.isLiteral();
      else if (!op.isConstant())
         ok = op.size() == (t.base_type == aco_base_type_lanemask ? lane_mask.size() : aco_type_dwords(t));
   }
   for (unsigned i = 0; ok && i < ai.num_defs; i++) {
      const Definition &def = instr->definitions[i];
      fixed_reg fr = ai.def_fixed_reg[i];
      aco_type t = ai.def_types[i];
      if (fr == fixed_scc)
         ok = def.physReg() == scc;
      else if (fr == fixed_exec)
         ok = def.physReg() == exec;
      else if (fr == fixed_m0)
         ok = def.physReg() == m0;
      else if (!instr->isSDWA())
         ok = def.size() == (t.base_type == aco_base_type_lanemask ? lane_mask.size() : aco_type_dwords(t));
   }
   if (!ok) {
      encode_error e;
      memset(&e.error, 0, sizeof(e.error));
      e.error.kind = AGC_ISA_ERROR_INVALID_OPERAND;
      e.error.pc = ctx.pc;
      e.error.opcode = (uint16_t)instr->opcode;
      e.error.reason = instr_info.name[(unsigned)instr->opcode];
      throw e;
   }
}

void
note_reject(lower_ctx &ctx, const agc_isa_error &error)
{
   if (ctx.split) {
      ctx.staged.emplace_back();
      ctx.split = false;
   }
   segment &seg = ctx.staged.back();
   if (!seg.has_reject) {
      seg.has_reject = true;
      seg.reject = error;
   }
}

static void
append(lower_ctx &ctx, Instruction *instr, const inst_meta &given)
{
   inst_meta meta = given;
   if (ctx.has_next_label) {
      meta.has_label = true;
      meta.label = ctx.next_label;
      ctx.has_next_label = false;
   }
   if (meta.has_label || ctx.split) {
      ctx.staged.emplace_back();
      ctx.staged.back().has_label = meta.has_label;
      ctx.staged.back().label = meta.label;
   }
   segment &seg = ctx.staged.back();
   if (instr) {
      wait_tracker_record(ctx, instr);
      descriptor_cache_invalidate(ctx, instr);
      seg.instructions.emplace_back(instr);
   }
   seg.has_target = meta.has_target;
   seg.target = meta.target;
   seg.branch = meta.branch;
   seg.falls = meta.falls;
   seg.pc = meta.pc;
   ctx.split = meta.ends;
}

void
flush(lower_ctx &ctx)
{
   if (ctx.pending.empty())
      return;
   inst_meta plain;
   for (aco_ptr<Instruction> &instr : ctx.pending) {
      check_alu(ctx, instr.get());
      append(ctx, instr.release(), plain);
   }
   ctx.pending.clear();
}

static void
push_checked(lower_ctx &ctx, const agc_isa_inst &inst)
{
   aco_opcode op = host_opcode(ctx, &inst);
   inst_meta meta;
   meta.pc = inst.pc;
   meta.has_label = inst.pc >= AGC_ISA_LOCAL_LABEL_BASE && inst.pc < AGC_ISA_PROLOGUE_LABEL;
   meta.label = inst.pc;
   meta.ends = agc_isa_inst_ends_block(&inst);
   meta.falls = agc_isa_inst_falls_through(&inst);
   meta.has_target = inst.has_branch_target;
   meta.target = inst.branch_target;
   if (op == aco_opcode::s_clause || op == aco_opcode::s_delay_alu) {
      append(ctx, NULL, meta);
      return;
   }
   if (inst.family == AGC_ISA_SOPP && agc_isa_inst_is_branch(&inst)) {
      if (!inst.has_branch_target)
         reject(inst.pc, "branch without a target");
      bool conditional = agc_isa_inst_has_flag(&inst, AGC_ISA_FLAG_COND_BRANCH);
      Instruction *instr = create_instruction(op, Format::SOPP, conditional, 0);
      if (conditional) {
         PhysReg reg = op == aco_opcode::s_cbranch_scc0 || op == aco_opcode::s_cbranch_scc1    ? scc
                       : op == aco_opcode::s_cbranch_vccz || op == aco_opcode::s_cbranch_vccnz ? vcc
                                                                                               : exec;
         instr->operands[0] = Operand(reg, reg == scc ? s1 : ctx.program->lane_mask);
      }
      meta.branch = true;
      append(ctx, instr, meta);
      return;
   }
   Instruction *instr;
   switch (inst.family) {
   case AGC_ISA_SOP1:
   case AGC_ISA_SOP2:
   case AGC_ISA_SOPC:
   case AGC_ISA_SOPK:
   case AGC_ISA_SOPP:
   case AGC_ISA_VOP1:
   case AGC_ISA_VOP2:
   case AGC_ISA_VOPC:
   case AGC_ISA_VOP3:
   case AGC_ISA_VOP3P:
      instr = build_alu(ctx, &inst, op);
      break;
   case AGC_ISA_SMEM:
      instr = build_smem(ctx, &inst, op);
      break;
   case AGC_ISA_MUBUF:
   case AGC_ISA_MTBUF:
      instr = build_mubuf(ctx, &inst, op);
      break;
   case AGC_ISA_FLAT:
      instr = build_flat(ctx, &inst, op);
      break;
   case AGC_ISA_DS:
      instr = build_ds(&inst, op);
      break;
   case AGC_ISA_MIMG:
      instr = build_image(ctx, &inst, op);
      break;
   case AGC_ISA_EXP:
      instr = build_export(&inst);
      break;
   case AGC_ISA_VINTRP:
   case AGC_ISA_LDSDIR:
   case AGC_ISA_VINTERP:
      instr = build_interp(&inst, op);
      break;
   default:
      reject_unsupported(inst.pc, "instruction family is not supported by the ACO bridge");
   }
   append(ctx, instr, meta);
}

void
push(lower_ctx &ctx, const agc_isa_inst &inst)
{
   flush(ctx);
   try {
      push_checked(ctx, inst);
   } catch (const encode_error &e) {
      note_reject(ctx, e.error);
   }
}

void
push_retarget(lower_ctx &ctx, const agc_isa_inst *inst, uint16_t opcode)
{
   push(ctx, agc_isa_inst_retarget(inst, opcode));
}

void
label(lower_ctx &ctx, uint32_t name)
{
   flush(ctx);
   inst_meta meta;
   meta.has_label = true;
   meta.label = name;
   meta.pc = name;
   append(ctx, create_instruction(aco_opcode::s_nop, Format::SOPP, 0, 0), meta);
}

void
label_next(lower_ctx &ctx, uint32_t name)
{
   flush(ctx);
   ctx.has_next_label = true;
   ctx.next_label = name;
}

void
branch(lower_ctx &ctx, aco_opcode op, uint32_t to)
{
   flush(ctx);
   bool conditional = op != aco_opcode::s_branch;
   Instruction *instr = create_instruction(op, Format::SOPP, conditional, 0);
   if (conditional) {
      PhysReg reg = op == aco_opcode::s_cbranch_scc0 || op == aco_opcode::s_cbranch_scc1    ? scc
                    : op == aco_opcode::s_cbranch_vccz || op == aco_opcode::s_cbranch_vccnz ? vcc
                                                                                            : exec;
      instr->operands[0] = Operand(reg, reg == scc ? s1 : ctx.program->lane_mask);
   }
   inst_meta meta;
   meta.ends = true;
   meta.falls = conditional;
   meta.has_target = true;
   meta.target = to;
   meta.branch = true;
   meta.pc = ctx.pc;
   append(ctx, instr, meta);
}

void
endpgm(lower_ctx &ctx)
{
   flush(ctx);
   inst_meta meta;
   meta.ends = true;
   meta.falls = false;
   meta.pc = ctx.pc;
   append(ctx, create_instruction(aco_opcode::s_endpgm, Format::SOPP, 0, 0), meta);
}

void
guest_endpgm(lower_ctx &ctx)
{
   if (ctx.epilogue)
      branch(ctx, aco_opcode::s_branch, AGC_ISA_EPILOGUE_LABEL);
   else
      endpgm(ctx);
}

void
setpc(lower_ctx &ctx, unsigned sgpr)
{
   flush(ctx);
   Instruction *instr = create_instruction(aco_opcode::s_setpc_b64, Format::SOP1, 1, 0);
   instr->operands[0] = sop(sgpr, 2);
   inst_meta meta;
   meta.ends = true;
   meta.falls = false;
   meta.pc = ctx.pc;
   append(ctx, instr, meta);
}

void
smem_load(lower_ctx &ctx, unsigned dwords, Definition data, unsigned sbase, Operand soffset, uint32_t offset)
{
   static const aco_opcode loads[] = {
      aco_opcode::s_load_dword,
      aco_opcode::s_load_dwordx2,
      aco_opcode::s_load_dwordx4,
      aco_opcode::s_load_dwordx8,
   };
   aco_opcode op = loads[dwords == 1 ? 0 : dwords == 2 ? 1 : dwords == 4 ? 2 : 3];
   ctx.bld.smem(op, data, sop(sbase, 2), imm(offset), soffset, memory_sync_info(storage_buffer),
                cache_flags(ctx, false, false, false));
}

void
global_access(lower_ctx &ctx, const agc_isa_inst *origin, uint16_t opcode, unsigned vaddr, agc_isa_operand saddr,
              unsigned vdata, int32_t offset, unsigned vdst, bool glc, bool dlc)
{
   const agc_isa_mem_op *info = agc_isa_flat_op(opcode);
   if (!info)
      fail(ctx, origin, "a global access of the memops table");
   agc_isa_inst out = agc_isa_tmpl(AGC_ISA_FLAT, opcode);
   out.flat.segment = AGC_ISA_SEGMENT_GLOBAL;
   out.flat.vdst = (uint8_t)vdst;
   out.flat.vdata = (uint8_t)vdata;
   out.flat.vaddr = (uint8_t)vaddr;
   out.flat.saddr = saddr;
   out.flat.uses_vaddr = true;
   out.flat.offset = offset;
   out.flat.glc = glc;
   out.flat.dlc = dlc;
   out.flat.slc = false;
   out.flat.lds = false;
   out.flat.kind = info->kind;
   out.flat.data_dwords = info->dwords;
   lower_template(ctx, origin, out);
}

}
