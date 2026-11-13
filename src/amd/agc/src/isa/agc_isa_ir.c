#include "agc_isa.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *
agc_isa_target_name(enum agc_isa_target target)
{
   static const char *const names[AGC_ISA_TARGET_COUNT] = {"gfx1013", "gfx10.3", "gfx11", "gfx11.5", "gfx12"};
   return (unsigned)target < AGC_ISA_TARGET_COUNT ? names[target] : "?";
}

bool
agc_isa_target_from_gfx_ip(uint32_t major, uint32_t minor, uint32_t stepping, enum agc_isa_target *target)
{
   if (major == 10 && minor == 1 && stepping == 3)
      *target = AGC_ISA_TARGET_GFX1013;
   else if (major == 10 && minor == 3)
      *target = AGC_ISA_TARGET_GFX103;
   else if (major == 11 && minor == 5)
      *target = AGC_ISA_TARGET_GFX115;
   else if (major == 11)
      *target = AGC_ISA_TARGET_GFX11;
   else if (major == 12)
      *target = AGC_ISA_TARGET_GFX12;
   else
      return false;
   return true;
}

const char *
agc_isa_family_name(enum agc_isa_family family)
{
   static const char *const names[AGC_ISA_FAMILY_COUNT] = {
      "SOP1",   "SOP2", "SOPK",  "SOPC",  "SOPP", "VOP1", "VOP2", "VOP3", "VOP3P",  "VOPC",
      "VINTRP", "SMEM", "MUBUF", "MTBUF", "FLAT", "DS",   "MIMG", "EXP",  "LDSDIR", "VINTERP",
   };
   return (unsigned)family < AGC_ISA_FAMILY_COUNT ? names[family] : "?";
}

bool
agc_isa_operand_from_code(uint16_t code, struct agc_isa_operand *op)
{
   if (code >= AGC_ISA_CODE_VGPR_BASE && code <= 511)
      *op = agc_isa_operand_vgpr((uint8_t)(code - AGC_ISA_CODE_VGPR_BASE));
   else if (code < AGC_ISA_SGPR_LIMIT)
      *op = agc_isa_operand_sgpr((uint8_t)code);
   else if (code == AGC_ISA_SCALAR_VCC_LO)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_VCC_LO);
   else if (code == AGC_ISA_SCALAR_VCC_HI)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_VCC_HI);
   else if (code < AGC_ISA_CODE_GFX10_M0)
      *op = agc_isa_operand_ttmp((uint8_t)(code - AGC_ISA_SCALAR_TTMP_BASE));
   else if (code == AGC_ISA_CODE_GFX10_M0)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_M0);
   else if (code == AGC_ISA_CODE_GFX10_NULL)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_NULL);
   else if (code == AGC_ISA_SCALAR_EXEC_LO)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_EXEC_LO);
   else if (code == AGC_ISA_SCALAR_EXEC_HI)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_EXEC_HI);
   else if (code <= AGC_ISA_CODE_INLINE_INT_MAX)
      *op = agc_isa_operand_inline_int((int8_t)(code - AGC_ISA_CODE_INLINE_INT_ZERO));
   else if (code <= AGC_ISA_CODE_INLINE_INT_NEG_MAX)
      *op = agc_isa_operand_inline_int((int8_t)-(int)(code - AGC_ISA_CODE_INLINE_INT_MAX));
   else if (code == 235)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_SHARED_BASE);
   else if (code == 236)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_SHARED_LIMIT);
   else if (code == 237)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_PRIVATE_BASE);
   else if (code == 238)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_PRIVATE_LIMIT);
   else if (code == 239)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_POPS_EXITING_WAVE_ID);
   else if (code >= AGC_ISA_CODE_INLINE_FLOAT_FIRST && code <= AGC_ISA_CODE_INLINE_FLOAT_LAST)
      *op = agc_isa_operand_inline_float((uint8_t)code);
   else if (code == 251)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_VCC_Z);
   else if (code == 252)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_EXEC_Z);
   else if (code == 253)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_SCC);
   else if (code == 254)
      *op = agc_isa_operand_special(AGC_ISA_OPERAND_LDS_DIRECT);
   else if (code == AGC_ISA_CODE_LITERAL)
      *op = agc_isa_operand_literal(0);
   else
      return false;
   return true;
}

bool
agc_isa_operand_scalar_destination(uint16_t code, struct agc_isa_operand *op)
{
   struct agc_isa_operand found;
   if (!agc_isa_operand_from_code(code, &found))
      return false;
   switch (found.kind) {
   case AGC_ISA_OPERAND_SGPR:
   case AGC_ISA_OPERAND_VCC_LO:
   case AGC_ISA_OPERAND_VCC_HI:
   case AGC_ISA_OPERAND_TTMP:
   case AGC_ISA_OPERAND_M0:
   case AGC_ISA_OPERAND_NULL:
   case AGC_ISA_OPERAND_EXEC_LO:
   case AGC_ISA_OPERAND_EXEC_HI:
      *op = found;
      return true;
   default:
      return false;
   }
}

struct agc_isa_operand
agc_isa_operand_from_code_or_raw(uint16_t code)
{
   struct agc_isa_operand op;
   if (!agc_isa_operand_from_code(code, &op))
      op = agc_isa_operand_raw(code);
   return op;
}

bool
agc_isa_operand_code(struct agc_isa_operand op, enum agc_isa_target target, uint16_t *code)
{
   bool gfx11 = agc_isa_target_is_gfx11_plus(target);
   switch (op.kind) {
   case AGC_ISA_OPERAND_SGPR:
      if (op.reg >= AGC_ISA_SGPR_LIMIT)
         return false;
      *code = (uint16_t)op.reg;
      return true;
   case AGC_ISA_OPERAND_VGPR:
      *code = (uint16_t)(AGC_ISA_CODE_VGPR_BASE + op.reg);
      return true;
   case AGC_ISA_OPERAND_VCC_LO:
      *code = AGC_ISA_SCALAR_VCC_LO;
      return true;
   case AGC_ISA_OPERAND_VCC_HI:
      *code = AGC_ISA_SCALAR_VCC_HI;
      return true;
   case AGC_ISA_OPERAND_TTMP:
      if (op.reg >= 16)
         return false;
      *code = (uint16_t)(AGC_ISA_SCALAR_TTMP_BASE + op.reg);
      return true;
   case AGC_ISA_OPERAND_M0:
      *code = gfx11 ? AGC_ISA_CODE_GFX11_M0 : AGC_ISA_CODE_GFX10_M0;
      return true;
   case AGC_ISA_OPERAND_NULL:
      *code = gfx11 ? AGC_ISA_CODE_GFX11_NULL : AGC_ISA_CODE_GFX10_NULL;
      return true;
   case AGC_ISA_OPERAND_EXEC_LO:
      *code = AGC_ISA_SCALAR_EXEC_LO;
      return true;
   case AGC_ISA_OPERAND_EXEC_HI:
      *code = AGC_ISA_SCALAR_EXEC_HI;
      return true;
   case AGC_ISA_OPERAND_INLINE_INT:
      if (op.inline_int >= 0 && op.inline_int <= 64)
         *code = (uint16_t)(AGC_ISA_CODE_INLINE_INT_ZERO + op.inline_int);
      else if (op.inline_int >= -16 && op.inline_int <= -1)
         *code = (uint16_t)(AGC_ISA_CODE_INLINE_INT_MAX - op.inline_int);
      else
         return false;
      return true;
   case AGC_ISA_OPERAND_INLINE_FLOAT:
      if (op.inline_float < AGC_ISA_CODE_INLINE_FLOAT_FIRST || op.inline_float > AGC_ISA_CODE_INLINE_FLOAT_LAST)
         return false;
      *code = (uint16_t)op.inline_float;
      return true;
   case AGC_ISA_OPERAND_LITERAL:
      *code = AGC_ISA_CODE_LITERAL;
      return true;
   case AGC_ISA_OPERAND_SHARED_BASE:
      *code = 235;
      return true;
   case AGC_ISA_OPERAND_SHARED_LIMIT:
      *code = 236;
      return true;
   case AGC_ISA_OPERAND_PRIVATE_BASE:
      *code = 237;
      return true;
   case AGC_ISA_OPERAND_PRIVATE_LIMIT:
      *code = 238;
      return true;
   case AGC_ISA_OPERAND_POPS_EXITING_WAVE_ID:
      if (gfx11)
         return false;
      *code = 239;
      return true;
   case AGC_ISA_OPERAND_VCC_Z:
      *code = 251;
      return true;
   case AGC_ISA_OPERAND_EXEC_Z:
      *code = 252;
      return true;
   case AGC_ISA_OPERAND_SCC:
      *code = 253;
      return true;
   case AGC_ISA_OPERAND_LDS_DIRECT:
      if (gfx11)
         return false;
      *code = 254;
      return true;
   case AGC_ISA_OPERAND_RAW:
      *code = (uint16_t)op.raw;
      return true;
   default:
      return false;
   }
}

bool
agc_isa_operand_scalar_index(struct agc_isa_operand op, uint16_t *index)
{
   switch (op.kind) {
   case AGC_ISA_OPERAND_SGPR:
      *index = (uint16_t)op.reg;
      return true;
   case AGC_ISA_OPERAND_VCC_LO:
      *index = AGC_ISA_SCALAR_VCC_LO;
      return true;
   case AGC_ISA_OPERAND_VCC_HI:
      *index = AGC_ISA_SCALAR_VCC_HI;
      return true;
   case AGC_ISA_OPERAND_TTMP:
      *index = (uint16_t)(AGC_ISA_SCALAR_TTMP_BASE + op.reg);
      return true;
   case AGC_ISA_OPERAND_M0:
      *index = AGC_ISA_SCALAR_M0;
      return true;
   case AGC_ISA_OPERAND_EXEC_LO:
      *index = AGC_ISA_SCALAR_EXEC_LO;
      return true;
   case AGC_ISA_OPERAND_EXEC_HI:
      *index = AGC_ISA_SCALAR_EXEC_HI;
      return true;
   default:
      return false;
   }
}

bool
agc_isa_inline_float_bits(uint8_t code, uint32_t *bits)
{
   static const uint32_t table[9] = {
      0x3f000000, 0xbf000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000, 0x40800000, 0xc0800000, 0x3e22f983,
   };
   if (code < AGC_ISA_CODE_INLINE_FLOAT_FIRST || code > AGC_ISA_CODE_INLINE_FLOAT_LAST)
      return false;
   *bits = table[code - AGC_ISA_CODE_INLINE_FLOAT_FIRST];
   return true;
}

bool
agc_isa_operand_inline_32(uint32_t value, struct agc_isa_operand *op)
{
   int32_t signed_value = (int32_t)value;
   if (signed_value >= -16 && signed_value <= 64) {
      *op = agc_isa_operand_inline_int((int8_t)signed_value);
      return true;
   }
   for (unsigned code = AGC_ISA_CODE_INLINE_FLOAT_FIRST; code <= AGC_ISA_CODE_INLINE_FLOAT_LAST; code++) {
      uint32_t bits;
      if (agc_isa_inline_float_bits((uint8_t)code, &bits) && bits == value) {
         *op = agc_isa_operand_inline_float((uint8_t)code);
         return true;
      }
   }
   return false;
}

struct agc_isa_operand
agc_isa_operand_constant(uint32_t value)
{
   struct agc_isa_operand op;
   if (!agc_isa_operand_inline_32(value, &op))
      op = agc_isa_operand_literal(value);
   return op;
}

uint8_t
agc_isa_image_address_register(const struct agc_isa_image *image, unsigned index)
{
   if (image->bvh8) {
      static const uint8_t group[11] = {0, 0, 2, 2, 4, 4, 4, 5, 5, 5, 6};
      static const uint8_t component[11] = {0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 0};
      assert(index < 11);
      return (uint8_t)(image->addr[group[index]] + component[index]);
   }
   if (image->nsa) {
      assert(index < AGC_ISA_MAX_NSA_ADDRESSES);
      return image->addr[index];
   }
   return (uint8_t)(image->addr[0] + index);
}

struct range_list {
   struct agc_isa_reg_range *ranges;
   unsigned count;
};

static void
push_range(struct range_list *list, enum agc_isa_reg_file file, unsigned first, unsigned count)
{
   assert(list->count < AGC_ISA_MAX_REG_RANGES);
   struct agc_isa_reg_range *range = &list->ranges[list->count++];
   range->file = (uint8_t)file;
   range->first = (uint16_t)first;
   range->count = (uint16_t)count;
}

static void
push_scalar(struct range_list *list, struct agc_isa_operand op, unsigned dwords)
{
   uint16_t first;
   if (agc_isa_operand_scalar_index(op, &first))
      push_range(list, AGC_ISA_REG_SCALAR, first, dwords > 1 ? dwords : 1);
}

static void
push_register(struct range_list *list, struct agc_isa_operand op, unsigned dwords)
{
   if (op.kind == AGC_ISA_OPERAND_VGPR)
      push_range(list, AGC_ISA_REG_VECTOR, op.reg, dwords > 1 ? dwords : 1);
   else
      push_scalar(list, op, dwords);
}

static unsigned
dst_dwords(const struct agc_isa_inst *inst)
{
   return agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_D2) ? 2 : 1;
}

static unsigned
src_dwords(const struct agc_isa_inst *inst, unsigned index)
{
   static const uint64_t wide[3] = {AGC_ISA_FLAG_S0_2, AGC_ISA_FLAG_S1_2, AGC_ISA_FLAG_S2_2};
   return agc_isa_inst_has_flag(inst, wide[index < 2 ? index : 2]) ? 2 : 1;
}

static unsigned
returned_dwords(const struct agc_isa_inst *inst, uint8_t kind, unsigned data_dwords)
{
   if (kind == AGC_ISA_MEM_ATOMIC && agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CMPSWAP))
      return data_dwords / 2;
   return data_dwords;
}

unsigned
agc_isa_inst_defs(const struct agc_isa_inst *inst, bool wave64, struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES])
{
   struct range_list list = {ranges, 0};
   unsigned mask = wave64 ? 2 : 1;
   switch (inst->fields) {
   case AGC_ISA_SOP1:
   case AGC_ISA_SOP2: {
      struct agc_isa_operand sdst = inst->fields == AGC_ISA_SOP1 ? inst->sop1.sdst : inst->sop2.sdst;
      if (!agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_DST))
         push_scalar(&list, sdst, dst_dwords(inst));
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_WRITES_EXEC))
         push_range(&list, AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_EXEC_LO, mask);
      break;
   }
   case AGC_ISA_SOPK:
      if (!agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_DST))
         push_scalar(&list, inst->sopk.sdst, dst_dwords(inst));
      break;
   case AGC_ISA_VOP1:
      if (!agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_OPERANDS))
         push_register(&list, inst->vop1.vdst, dst_dwords(inst));
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_SWAP))
         push_register(&list, inst->vop1.src0, 1);
      break;
   case AGC_ISA_VOP2:
      push_register(&list, inst->vop2.vdst, dst_dwords(inst));
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CARRY_OUT))
         push_range(&list, AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_VCC_LO, mask);
      break;
   case AGC_ISA_VOPC:
      push_scalar(&list, inst->vopc.sdst, mask);
      break;
   case AGC_ISA_VOP3: {
      const struct agc_isa_vop3 *v = &inst->vop3;
      unsigned dwords = mask;
      if (agc_isa_operand_is_vgpr(v->vdst))
         dwords = dst_dwords(inst);
      else if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_SDST))
         dwords = 1;
      push_register(&list, v->vdst, dwords);
      if (agc_isa_operand_is_some(v->sdst))
         push_scalar(&list, v->sdst, mask);
      break;
   }
   case AGC_ISA_VOP3P:
      push_register(&list, inst->vop3p.vdst, dst_dwords(inst));
      break;
   case AGC_ISA_VINTRP:
      push_range(&list, AGC_ISA_REG_VECTOR, inst->vintrp.vdst, 1);
      break;
   case AGC_ISA_SMEM:
      if (inst->smem.dwords > 0)
         push_scalar(&list, inst->smem.sdata, inst->smem.dwords);
      break;
   case AGC_ISA_MUBUF:
   case AGC_ISA_MTBUF: {
      const struct agc_isa_buffer *b = &inst->buffer;
      if (b->kind == AGC_ISA_MEM_LOAD || (b->kind == AGC_ISA_MEM_ATOMIC && b->glc))
         push_range(&list, AGC_ISA_REG_VECTOR, b->vdata, returned_dwords(inst, b->kind, b->data_dwords) + b->tfe);
      break;
   }
   case AGC_ISA_FLAT: {
      const struct agc_isa_flat *f = &inst->flat;
      if (f->kind == AGC_ISA_MEM_LOAD || (f->kind == AGC_ISA_MEM_ATOMIC && f->glc))
         push_range(&list, AGC_ISA_REG_VECTOR, f->vdst, returned_dwords(inst, f->kind, f->data_dwords));
      break;
   }
   case AGC_ISA_DS:
      if (inst->ds.dst_dwords > 0)
         push_range(&list, AGC_ISA_REG_VECTOR, inst->ds.vdst, inst->ds.dst_dwords);
      if (inst->ds.writes_addr)
         push_range(&list, AGC_ISA_REG_VECTOR, inst->ds.addr, 1);
      break;
   case AGC_ISA_MIMG: {
      const struct agc_isa_image *m = &inst->mimg;
      if (m->kind == AGC_ISA_MEM_STORE || m->data_dwords == 0)
         break;
      if (m->bvh8) {
         push_range(&list, AGC_ISA_REG_VECTOR, agc_isa_image_address_register(m, 4), 3);
         push_range(&list, AGC_ISA_REG_VECTOR, agc_isa_image_address_register(m, 7), 3);
      }
      if (m->kind != AGC_ISA_MEM_ATOMIC || m->glc)
         push_range(&list, AGC_ISA_REG_VECTOR, m->vdata,
                    returned_dwords(inst, m->kind, m->data_dwords) + (m->tfe || m->lwe));
      break;
   }
   case AGC_ISA_LDSDIR:
      push_range(&list, AGC_ISA_REG_VECTOR, inst->ldsdir.vdst, 1);
      break;
   case AGC_ISA_VINTERP:
      push_range(&list, AGC_ISA_REG_VECTOR, inst->vinterp.vdst, 1);
      break;
   default:
      break;
   }
   if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_WRITES_VCC))
      push_range(&list, AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_VCC_LO, mask);
   return list.count;
}

unsigned
agc_isa_inst_uses(const struct agc_isa_inst *inst, bool wave64, struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES])
{
   struct range_list list = {ranges, 0};
   unsigned mask = wave64 ? 2 : 1;
   switch (inst->fields) {
   case AGC_ISA_SOP1:
      if (!agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_SRC))
         push_scalar(&list, inst->sop1.src0, src_dwords(inst, 0));
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_READS_DST))
         push_scalar(&list, inst->sop1.sdst, dst_dwords(inst));
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_WRITES_EXEC))
         push_range(&list, AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_EXEC_LO, mask);
      break;
   case AGC_ISA_SOP2:
      push_scalar(&list, inst->sop2.src0, src_dwords(inst, 0));
      push_scalar(&list, inst->sop2.src1, src_dwords(inst, 1));
      break;
   case AGC_ISA_SOPK:
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_READS_DST))
         push_scalar(&list, inst->sopk.sdst, 1);
      break;
   case AGC_ISA_SOPC:
      push_scalar(&list, inst->sopc.src0, src_dwords(inst, 0));
      push_scalar(&list, inst->sopc.src1, src_dwords(inst, 1));
      break;
   case AGC_ISA_VOP1:
      if (!agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_OPERANDS))
         push_register(&list, inst->vop1.src0, src_dwords(inst, 0));
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_SWAP))
         push_register(&list, inst->vop1.vdst, 1);
      break;
   case AGC_ISA_VOP2:
      push_register(&list, inst->vop2.src0, src_dwords(inst, 0));
      push_register(&list, inst->vop2.vsrc1, src_dwords(inst, 1));
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_READS_DST))
         push_register(&list, inst->vop2.vdst, dst_dwords(inst));
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CARRY_IN))
         push_range(&list, AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_VCC_LO, mask);
      break;
   case AGC_ISA_VOPC:
      push_register(&list, inst->vopc.src0, src_dwords(inst, 0));
      push_register(&list, inst->vopc.vsrc1, src_dwords(inst, 1));
      break;
   case AGC_ISA_VOP3: {
      const struct agc_isa_vop3 *v = &inst->vop3;
      unsigned count = agc_isa_source_count(inst);
      for (unsigned i = 0; i < 3 && i < count; i++) {
         bool masked = !agc_isa_operand_is_vgpr(v->src[i]) && i == 2 &&
                       agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_CARRY_IN | AGC_ISA_FLAG_MASK_SRC2);
         push_register(&list, v->src[i], masked ? mask : src_dwords(inst, i));
      }
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_READS_DST))
         push_register(&list, v->vdst, dst_dwords(inst));
      break;
   }
   case AGC_ISA_VOP3P: {
      const struct agc_isa_vop3p *v = &inst->vop3p;
      unsigned count = agc_isa_source_count(inst);
      for (unsigned i = 0; i < 3 && i < count; i++)
         push_register(&list, v->src[i], src_dwords(inst, i));
      if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_READS_DST))
         push_register(&list, v->vdst, 1);
      break;
   }
   case AGC_ISA_VINTRP:
      if (inst->opcode != AGC_ISA_VINTRP_MOV)
         push_range(&list, AGC_ISA_REG_VECTOR, inst->vintrp.vsrc, 1);
      if (inst->opcode == AGC_ISA_VINTRP_P2)
         push_range(&list, AGC_ISA_REG_VECTOR, inst->vintrp.vdst, 1);
      push_range(&list, AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_M0, 1);
      break;
   case AGC_ISA_SMEM: {
      const struct agc_isa_smem *s = &inst->smem;
      if (!agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_NO_SRC) &&
          (s->dwords > 0 || agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_SMEM_ADDR)))
         push_range(&list, AGC_ISA_REG_SCALAR, s->sbase, s->buffer ? 4 : 2);
      push_scalar(&list, s->soffset, 1);
      break;
   }
   case AGC_ISA_MUBUF:
   case AGC_ISA_MTBUF: {
      const struct agc_isa_buffer *b = &inst->buffer;
      if (b->kind == AGC_ISA_MEM_CONTROL)
         break;
      if (b->offen || b->idxen)
         push_range(&list, AGC_ISA_REG_VECTOR, b->vaddr, (unsigned)b->offen + b->idxen);
      push_range(&list, AGC_ISA_REG_SCALAR, b->srsrc, 4);
      push_scalar(&list, b->soffset, 1);
      if (b->kind != AGC_ISA_MEM_LOAD)
         push_range(&list, AGC_ISA_REG_VECTOR, b->vdata, b->data_dwords);
      break;
   }
   case AGC_ISA_FLAT: {
      const struct agc_isa_flat *f = &inst->flat;
      bool scratch = f->segment == AGC_ISA_SEGMENT_SCRATCH;
      if (f->uses_vaddr) {
         bool wide = !scratch && (f->saddr.kind == AGC_ISA_OPERAND_NONE || f->saddr.kind == AGC_ISA_OPERAND_NULL);
         push_range(&list, AGC_ISA_REG_VECTOR, f->vaddr, wide ? 2 : 1);
      }
      if (agc_isa_operand_is_some(f->saddr))
         push_scalar(&list, f->saddr, scratch ? 1 : 2);
      if (f->kind != AGC_ISA_MEM_LOAD)
         push_range(&list, AGC_ISA_REG_VECTOR, f->vdata, f->data_dwords);
      break;
   }
   case AGC_ISA_DS: {
      const struct agc_isa_ds *d = &inst->ds;
      if (d->uses_addr)
         push_range(&list, AGC_ISA_REG_VECTOR, d->addr, 1);
      if (d->data0_dwords > 0)
         push_range(&list, AGC_ISA_REG_VECTOR, d->data0, d->data0_dwords);
      if (d->data1_dwords > 0)
         push_range(&list, AGC_ISA_REG_VECTOR, d->data1, d->data1_dwords);
      if (d->gds || agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_READS_M0))
         push_range(&list, AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_M0, 1);
      break;
   }
   case AGC_ISA_MIMG: {
      const struct agc_isa_image *m = &inst->mimg;
      if (m->nsa) {
         for (unsigned index = 0; index < m->addr_dwords; index++)
            push_range(&list, AGC_ISA_REG_VECTOR, agc_isa_image_address_register(m, index), 1);
      } else {
         push_range(&list, AGC_ISA_REG_VECTOR, m->addr[0], m->addr_dwords);
      }
      push_range(&list, AGC_ISA_REG_SCALAR, m->srsrc, m->r128 ? 4 : 8);
      if (m->uses_sampler)
         push_range(&list, AGC_ISA_REG_SCALAR, m->ssamp, 4);
      if (m->kind != AGC_ISA_MEM_LOAD && m->data_dwords > 0)
         push_range(&list, AGC_ISA_REG_VECTOR, m->vdata, m->data_dwords);
      break;
   }
   case AGC_ISA_EXP: {
      const struct agc_isa_export *e = &inst->exp;
      for (unsigned i = 0; i < 4; i++) {
         bool live = e->compr ? i < 2 && (e->en & (3u << (i * 2))) != 0 : (e->en & (1u << i)) != 0;
         if (live)
            push_range(&list, AGC_ISA_REG_VECTOR, e->vsrc[i], 1);
      }
      break;
   }
   case AGC_ISA_LDSDIR:
      push_range(&list, AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_M0, 1);
      break;
   case AGC_ISA_VINTERP:
      for (unsigned i = 0; i < 3; i++)
         push_register(&list, inst->vinterp.src[i], 1);
      break;
   default:
      break;
   }
   if (agc_isa_family_is_vector_alu((enum agc_isa_family)inst->family) ||
       agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_READS_EXEC))
      push_range(&list, AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_EXEC_LO, mask);
   if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_READS_VCC))
      push_range(&list, AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_VCC_LO, mask);
   if (agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_READS_M0) && inst->fields != AGC_ISA_DS)
      push_range(&list, AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_M0, 1);
   return list.count;
}

int
agc_isa_error_format(const struct agc_isa_error *error, char *text, size_t size)
{
   char scratch[AGC_ISA_MNEMONIC_SIZE];
   switch (error->kind) {
   case AGC_ISA_ERROR_NONE:
      return snprintf(text, size, "no error");
   case AGC_ISA_ERROR_UNSUPPORTED:
      return snprintf(text, size, "unsupported shader");
   case AGC_ISA_ERROR_INVALID_OPTIONS:
      return snprintf(text, size, "invalid compile options: %s", error->reason);
   case AGC_ISA_ERROR_TARGET_MISMATCH:
      return snprintf(text, size, "compiler targets %s but %s was requested",
                      agc_isa_target_name((enum agc_isa_target)error->compiler),
                      agc_isa_target_name((enum agc_isa_target)error->requested));
   case AGC_ISA_ERROR_EMPTY_PROGRAM:
      return snprintf(text, size, "shader code is empty");
   case AGC_ISA_ERROR_TRUNCATED:
      return snprintf(text, size, "instruction at 0x%x is truncated", error->pc);
   case AGC_ISA_ERROR_INVALID_ENCODING:
      return snprintf(text, size, "invalid encoding 0x%08x at 0x%x", error->word, error->pc);
   case AGC_ISA_ERROR_UNKNOWN_OPCODE:
      return snprintf(text, size, "unknown %s opcode 0x%x at 0x%x",
                      agc_isa_family_name((enum agc_isa_family)error->family), (unsigned)error->opcode, error->pc);
   case AGC_ISA_ERROR_INVALID_OPERAND:
      return snprintf(text, size, "invalid operand at 0x%x: %s", error->pc, error->reason);
   case AGC_ISA_ERROR_INVALID_BRANCH:
      return snprintf(text, size, "branch at 0x%x targets invalid address 0x%x", error->pc, error->branch_target);
   case AGC_ISA_ERROR_MISSING_END_PROGRAM:
      return snprintf(text, size, "code ends before a reachable s_endpgm");
   case AGC_ISA_ERROR_UNSUPPORTED_INSTRUCTION:
      return snprintf(text, size, "%s at 0x%x is unsupported on %s: %s",
                      agc_isa_mnemonic((enum agc_isa_family)error->family, error->opcode, scratch), error->pc,
                      agc_isa_target_name((enum agc_isa_target)error->target), error->reason);
   case AGC_ISA_ERROR_BRANCH_OUT_OF_RANGE:
      return snprintf(text, size, "branch at 0x%x is out of range after lowering", error->pc);
   case AGC_ISA_ERROR_OUT_OF_REGISTERS:
      return snprintf(text, size, "lowering needs %u VGPRs, exceeding the register file", (unsigned)error->vgprs);
   case AGC_ISA_ERROR_OUT_OF_MEMORY:
      return snprintf(text, size, "out of memory");
   }
   return snprintf(text, size, "unknown error");
}

bool
agc_isa_error_is_unsupported(const struct agc_isa_error *error)
{
   return error->kind == AGC_ISA_ERROR_UNSUPPORTED || error->kind == AGC_ISA_ERROR_UNSUPPORTED_INSTRUCTION ||
          error->kind == AGC_ISA_ERROR_UNKNOWN_OPCODE;
}

bool
agc_isa_error_pc(const struct agc_isa_error *error, uint32_t *pc)
{
   switch (error->kind) {
   case AGC_ISA_ERROR_TRUNCATED:
   case AGC_ISA_ERROR_INVALID_ENCODING:
   case AGC_ISA_ERROR_UNKNOWN_OPCODE:
   case AGC_ISA_ERROR_INVALID_OPERAND:
   case AGC_ISA_ERROR_INVALID_BRANCH:
   case AGC_ISA_ERROR_UNSUPPORTED_INSTRUCTION:
   case AGC_ISA_ERROR_BRANCH_OUT_OF_RANGE:
      *pc = error->pc;
      return true;
   default:
      return false;
   }
}

void
agc_isa_ir_init(struct agc_isa_ir *ir)
{
   memset(ir, 0, sizeof(*ir));
}

void
agc_isa_ir_finish(struct agc_isa_ir *ir)
{
   free(ir->instructions);
   memset(ir, 0, sizeof(*ir));
}

void
agc_isa_ir_clear(struct agc_isa_ir *ir)
{
   ir->instruction_count = 0;
   ir->code_words = 0;
}

bool
agc_isa_ir_reserve(struct agc_isa_ir *ir, size_t capacity)
{
   if (capacity <= ir->instruction_capacity)
      return true;
   size_t grown = ir->instruction_capacity ? ir->instruction_capacity * 2 : 64;
   if (grown < capacity)
      grown = capacity;
   if (grown > SIZE_MAX / sizeof(*ir->instructions))
      return false;
   struct agc_isa_inst *instructions = realloc(ir->instructions, grown * sizeof(*ir->instructions));
   if (!instructions)
      return false;
   ir->instructions = instructions;
   ir->instruction_capacity = grown;
   return true;
}

bool
agc_isa_ir_instruction_at(const struct agc_isa_ir *ir, uint32_t pc, size_t *index)
{
   size_t lo = 0;
   size_t hi = ir->instruction_count;
   while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      if (ir->instructions[mid].pc < pc)
         lo = mid + 1;
      else
         hi = mid;
   }
   if (lo == ir->instruction_count || ir->instructions[lo].pc != pc)
      return false;
   *index = lo;
   return true;
}
