#include "agc_isa.h"

#include <stdlib.h>
#include <string.h>

#define FLAT_SADDR_NULL 0x7d
#define FLAT_SADDR_OFF 0x7f
#define EXP_RESERVED_BITS 0x03ffe000u

struct decode_ctx {
   const uint32_t *words;
   size_t word_count;
   size_t index;
   uint32_t pc;
   const struct agc_isa_host_tables *host;
   bool gfx11;
   struct agc_isa_error *error;
};

static inline bool
bit(uint32_t word, unsigned index)
{
   return ((word >> index) & 1u) != 0;
}

static inline uint16_t
field(uint32_t word, unsigned shift, uint32_t mask)
{
   return (uint16_t)((word >> shift) & mask);
}

static inline int32_t
sign_extend(uint32_t value, unsigned bits)
{
   unsigned shift = 32 - bits;
   return (int32_t)(value << shift) >> shift;
}

static bool
fail(struct agc_isa_error *error, enum agc_isa_error_kind kind, uint32_t pc)
{
   memset(error, 0, sizeof(*error));
   error->kind = kind;
   error->pc = pc;
   return false;
}

static bool
fail_truncated(const struct decode_ctx *ctx)
{
   return fail(ctx->error, AGC_ISA_ERROR_TRUNCATED, ctx->pc);
}

static bool
fail_encoding(const struct decode_ctx *ctx, uint32_t word)
{
   fail(ctx->error, AGC_ISA_ERROR_INVALID_ENCODING, ctx->pc);
   ctx->error->word = word;
   return false;
}

static bool
fail_unknown(const struct decode_ctx *ctx, enum agc_isa_family family, uint16_t opcode)
{
   fail(ctx->error, AGC_ISA_ERROR_UNKNOWN_OPCODE, ctx->pc);
   ctx->error->family = (uint8_t)family;
   ctx->error->opcode = opcode;
   return false;
}

static bool
fail_operand(const struct decode_ctx *ctx, const char *reason)
{
   fail(ctx->error, AGC_ISA_ERROR_INVALID_OPERAND, ctx->pc);
   ctx->error->reason = reason;
   return false;
}

static bool
fail_branch(struct agc_isa_error *error, uint32_t pc, uint32_t target)
{
   fail(error, AGC_ISA_ERROR_INVALID_BRANCH, pc);
   error->branch_target = target;
   return false;
}

static bool
read_word(const struct decode_ctx *ctx, size_t offset, uint32_t *word)
{
   if (offset >= ctx->word_count - ctx->index)
      return fail_truncated(ctx);
   *word = ctx->words[ctx->index + offset];
   return true;
}

static bool
guest_opcode(const struct decode_ctx *ctx, enum agc_isa_family family, uint16_t raw, uint16_t *op)
{
   if (!ctx->host) {
      *op = raw;
      return true;
   }
   if (!agc_isa_host_guest_opcode(ctx->host, family, raw, op))
      return fail_unknown(ctx, family, raw);
   return true;
}

static inline uint16_t
guest_code(const struct decode_ctx *ctx, uint16_t code)
{
   if (ctx->gfx11) {
      if (code == AGC_ISA_CODE_GFX11_NULL)
         return AGC_ISA_CODE_GFX10_NULL;
      if (code == AGC_ISA_CODE_GFX11_M0)
         return AGC_ISA_CODE_GFX10_M0;
   }
   return code;
}

static inline struct agc_isa_operand
raw_operand(const struct decode_ctx *ctx, uint16_t code)
{
   return agc_isa_operand_from_code_or_raw(guest_code(ctx, code));
}

static bool
source(const struct decode_ctx *ctx, uint16_t code, struct agc_isa_operand *op)
{
   if (!agc_isa_operand_from_code(guest_code(ctx, code), op))
      return fail_operand(ctx, "reserved source operand encoding");
   return true;
}

static bool
scalar_destination(const struct decode_ctx *ctx, uint16_t code, struct agc_isa_operand *op)
{
   if (!agc_isa_operand_scalar_destination(guest_code(ctx, code), op))
      return fail_operand(ctx, "invalid scalar destination");
   return true;
}

static bool
literal(const struct decode_ctx *ctx, size_t offset, struct agc_isa_operand *ops, unsigned count, unsigned *extra)
{
   bool wanted = false;
   for (unsigned i = 0; i < count; i++)
      wanted |= ops[i].kind == AGC_ISA_OPERAND_LITERAL;
   *extra = 0;
   if (!wanted)
      return true;
   uint32_t value;
   if (!read_word(ctx, offset, &value))
      return false;
   for (unsigned i = 0; i < count; i++) {
      if (ops[i].kind == AGC_ISA_OPERAND_LITERAL)
         ops[i].literal = value;
   }
   *extra = 1;
   return true;
}

static void
finish(struct agc_isa_inst *inst, enum agc_isa_family family, uint16_t opcode, uint64_t flags, unsigned words)
{
   inst->family = (uint8_t)family;
   inst->fields = (uint8_t)family;
   inst->opcode = opcode;
   inst->flags = flags;
   inst->word_count = (uint8_t)words;
}

static bool
branch(const struct decode_ctx *ctx, struct agc_isa_inst *inst, uint16_t simm16)
{
   int64_t target = (int64_t)ctx->pc + 4 + (int64_t)(int16_t)simm16 * 4;
   if (target < 0 || target > UINT32_MAX)
      return fail_branch(ctx->error, ctx->pc, (uint32_t)target);
   inst->has_branch_target = true;
   inst->branch_target = (uint32_t)target;
   return true;
}

static bool
decode_sendmsg_rtn(const struct decode_ctx *ctx, uint32_t w, struct agc_isa_inst *inst)
{
   if (field(w, 0, 0xff) != AGC_ISA_MSG_RTN_GET_REALTIME)
      return fail_unknown(ctx, AGC_ISA_SOP1, AGC_ISA_GFX11_SOP1_SENDMSG_RTN_B64);
   const struct agc_isa_mem_op *info = agc_isa_smem_op(AGC_ISA_SMEM_MEMREALTIME);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_SMEM, AGC_ISA_SMEM_MEMREALTIME);
   struct agc_isa_smem *s = &inst->smem;
   if (!scalar_destination(ctx, field(w, 16, 0x7f), &s->sdata))
      return false;
   s->soffset = agc_isa_operand_special(AGC_ISA_OPERAND_NULL);
   s->dwords = info->dwords;
   finish(inst, AGC_ISA_SMEM, info->op, info->flags, 1);
   return true;
}

static bool
decode_sop1(const struct decode_ctx *ctx, uint32_t w, struct agc_isa_inst *inst)
{
   uint16_t raw = field(w, 8, 0xff);
   if (ctx->gfx11 && raw == AGC_ISA_GFX11_SOP1_SENDMSG_RTN_B64)
      return decode_sendmsg_rtn(ctx, w, inst);
   uint16_t op;
   if (!guest_opcode(ctx, AGC_ISA_SOP1, raw, &op))
      return false;
   const struct agc_isa_op_info *info = agc_isa_op_lookup(AGC_ISA_SOP1, op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_SOP1, op);
   uint64_t flags = info->flags;
   struct agc_isa_sop1 *f = &inst->sop1;
   uint16_t sdst_code = field(w, 16, 0x7f);
   if (flags & AGC_ISA_FLAG_NO_DST)
      f->sdst = raw_operand(ctx, sdst_code);
   else if (!scalar_destination(ctx, sdst_code, &f->sdst))
      return false;
   if (flags & AGC_ISA_FLAG_NO_SRC)
      f->src0 = raw_operand(ctx, field(w, 0, 0xff));
   else if (!source(ctx, field(w, 0, 0xff), &f->src0))
      return false;
   if (op == AGC_ISA_SOP1_SWAPPC_B64 && f->sdst.kind == AGC_ISA_OPERAND_NULL) {
      flags &= ~(AGC_ISA_FLAG_CALL | AGC_ISA_FLAG_D2 | AGC_ISA_FLAG_PC_RELATIVE);
      flags |= AGC_ISA_FLAG_JUMP | AGC_ISA_FLAG_NO_DST;
   }
   unsigned extra = 0;
   if (!(flags & AGC_ISA_FLAG_NO_SRC) && !literal(ctx, 1, &f->src0, 1, &extra))
      return false;
   finish(inst, AGC_ISA_SOP1, op, flags, 1 + extra);
   return true;
}

static bool
decode_sop2(const struct decode_ctx *ctx, uint32_t w, struct agc_isa_inst *inst)
{
   uint16_t op;
   if (!guest_opcode(ctx, AGC_ISA_SOP2, field(w, 23, 0x7f), &op))
      return false;
   const struct agc_isa_op_info *info = agc_isa_op_lookup(AGC_ISA_SOP2, op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_SOP2, op);
   struct agc_isa_sop2 *f = &inst->sop2;
   struct agc_isa_operand src[2];
   unsigned extra;
   if (!scalar_destination(ctx, field(w, 16, 0x7f), &f->sdst) || !source(ctx, field(w, 0, 0xff), &src[0]) ||
       !source(ctx, field(w, 8, 0xff), &src[1]) || !literal(ctx, 1, src, 2, &extra))
      return false;
   f->src0 = src[0];
   f->src1 = src[1];
   finish(inst, AGC_ISA_SOP2, op, info->flags, 1 + extra);
   return true;
}

static bool
decode_sopk(const struct decode_ctx *ctx, uint32_t w, struct agc_isa_inst *inst)
{
   uint16_t op;
   if (!guest_opcode(ctx, AGC_ISA_SOPK, field(w, 23, 0x1f), &op))
      return false;
   const struct agc_isa_op_info *info = agc_isa_op_lookup(AGC_ISA_SOPK, op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_SOPK, op);
   struct agc_isa_sopk *f = &inst->sopk;
   f->sdst = raw_operand(ctx, field(w, 16, 0x7f));
   f->simm16 = field(w, 0, 0xffff);
   if (op == AGC_ISA_SOPK_SETREG_IMM32_B32) {
      if (!read_word(ctx, 1, &f->imm32))
         return false;
      f->has_imm32 = true;
   }
   finish(inst, AGC_ISA_SOPK, op, info->flags, f->has_imm32 ? 2 : 1);
   if (info->flags & (AGC_ISA_FLAG_CALL | AGC_ISA_FLAG_COND_BRANCH))
      return branch(ctx, inst, f->simm16);
   return true;
}

static bool
decode_sopc(const struct decode_ctx *ctx, uint32_t w, struct agc_isa_inst *inst)
{
   uint16_t op;
   if (!guest_opcode(ctx, AGC_ISA_SOPC, field(w, 16, 0x7f), &op))
      return false;
   const struct agc_isa_op_info *info = agc_isa_op_lookup(AGC_ISA_SOPC, op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_SOPC, op);
   struct agc_isa_operand src[2];
   unsigned extra;
   if (!source(ctx, field(w, 0, 0xff), &src[0]) || !source(ctx, field(w, 8, 0xff), &src[1]) ||
       !literal(ctx, 1, src, 2, &extra))
      return false;
   inst->sopc.src0 = src[0];
   inst->sopc.src1 = src[1];
   finish(inst, AGC_ISA_SOPC, op, info->flags, 1 + extra);
   return true;
}

static uint16_t
waitcnt_gfx11_to_gfx10(uint16_t simm16)
{
   unsigned vm = (simm16 >> 10) & 0x3f;
   unsigned exp = simm16 & 0x7;
   unsigned lgkm = (simm16 >> 4) & 0x3f;
   return (uint16_t)((vm & 0xf) | ((vm >> 4) << 14) | (exp << 4) | (lgkm << 8));
}

static bool
decode_sopp(const struct decode_ctx *ctx, uint32_t w, struct agc_isa_inst *inst)
{
   uint16_t op;
   if (!guest_opcode(ctx, AGC_ISA_SOPP, field(w, 16, 0x7f), &op))
      return false;
   const struct agc_isa_op_info *info = agc_isa_op_lookup(AGC_ISA_SOPP, op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_SOPP, op);
   uint16_t simm16 = field(w, 0, 0xffff);
   if (ctx->gfx11 && op == AGC_ISA_SOPP_WAITCNT)
      simm16 = waitcnt_gfx11_to_gfx10(simm16);
   inst->sopp.simm16 = simm16;
   finish(inst, AGC_ISA_SOPP, op, info->flags, 1);
   if (info->flags & (AGC_ISA_FLAG_BRANCH | AGC_ISA_FLAG_COND_BRANCH))
      return branch(ctx, inst, simm16);
   return true;
}

static struct agc_isa_sdwa
sdwa_word(uint32_t w1, bool vopc)
{
   struct agc_isa_sdwa s = {
      .dst_sel = vopc ? 0 : (uint8_t)field(w1, 8, 7),
      .dst_unused = vopc ? 0 : (uint8_t)field(w1, 11, 3),
      .clamp = !vopc && bit(w1, 13),
      .omod = vopc ? 0 : (uint8_t)field(w1, 14, 3),
      .src0_sel = (uint8_t)field(w1, 16, 7),
      .src0_sext = bit(w1, 19),
      .src0_neg = bit(w1, 20),
      .src0_abs = bit(w1, 21),
      .src1_sel = (uint8_t)field(w1, 24, 7),
      .src1_sext = bit(w1, 27),
      .src1_neg = bit(w1, 28),
      .src1_abs = bit(w1, 29),
   };
   return s;
}

static struct agc_isa_dpp16
dpp16_word(uint32_t w1)
{
   struct agc_isa_dpp16 d = {
      .ctrl = field(w1, 8, 0x1ff),
      .fetch_inactive = bit(w1, 18),
      .bound_ctrl = bit(w1, 19),
      .src0_neg = bit(w1, 20),
      .src0_abs = bit(w1, 21),
      .src1_neg = bit(w1, 22),
      .src1_abs = bit(w1, 23),
      .bank_mask = (uint8_t)field(w1, 24, 0xf),
      .row_mask = (uint8_t)field(w1, 28, 0xf),
   };
   return d;
}

static bool
sdwa_source(const struct decode_ctx *ctx, uint32_t w1, unsigned select_bit, uint16_t code, struct agc_isa_operand *op)
{
   return source(ctx, bit(w1, select_bit) ? code : (uint16_t)(code + 256), op);
}

struct vop_ext {
   struct agc_isa_operand src0;
   struct agc_isa_vop_ext ext;
   unsigned words;
   uint32_t w1;
};

static bool
decode_vop_ext(const struct decode_ctx *ctx, uint16_t src0_code, struct vop_ext *out)
{
   memset(out, 0, sizeof(*out));
   out->words = 2;
   switch (src0_code) {
   case AGC_ISA_CODE_SDWA:
      if (!read_word(ctx, 1, &out->w1) || !sdwa_source(ctx, out->w1, 23, field(out->w1, 0, 0xff), &out->src0))
         return false;
      out->ext.kind = AGC_ISA_VOP_EXT_SDWA;
      out->ext.sdwa = sdwa_word(out->w1, false);
      return true;
   case AGC_ISA_CODE_DPP16:
      if (!read_word(ctx, 1, &out->w1))
         return false;
      out->src0 = agc_isa_operand_vgpr((uint8_t)out->w1);
      out->ext.kind = AGC_ISA_VOP_EXT_DPP16;
      out->ext.dpp16 = dpp16_word(out->w1);
      return true;
   case AGC_ISA_CODE_DPP8:
   case AGC_ISA_CODE_DPP8_FI:
      if (!read_word(ctx, 1, &out->w1))
         return false;
      out->src0 = agc_isa_operand_vgpr((uint8_t)out->w1);
      out->ext.kind = AGC_ISA_VOP_EXT_DPP8;
      out->ext.dpp8.selects = out->w1 >> 8;
      out->ext.dpp8.fetch_inactive = src0_code == AGC_ISA_CODE_DPP8_FI;
      return true;
   default:
      out->words = 1;
      return source(ctx, src0_code, &out->src0);
   }
}

static bool
decode_vop1(const struct decode_ctx *ctx, uint32_t w, struct agc_isa_inst *inst)
{
   uint16_t op;
   if (!guest_opcode(ctx, AGC_ISA_VOP1, field(w, 9, 0xff), &op))
      return false;
   const struct agc_isa_op_info *info = agc_isa_op_lookup(AGC_ISA_VOP1, op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_VOP1, op);
   struct agc_isa_vop1 *f = &inst->vop1;
   uint16_t vdst_code = field(w, 17, 0xff);
   if (!(info->flags & AGC_ISA_FLAG_SDST))
      f->vdst = agc_isa_operand_vgpr((uint8_t)vdst_code);
   else if (!scalar_destination(ctx, vdst_code, &f->vdst))
      return false;
   uint16_t src0_code = field(w, 0, 0x1ff);
   if (info->flags & AGC_ISA_FLAG_NO_OPERANDS) {
      f->src0 = raw_operand(ctx, src0_code);
      finish(inst, AGC_ISA_VOP1, op, info->flags, 1);
      return true;
   }
   struct vop_ext ext;
   unsigned extra;
   if (!decode_vop_ext(ctx, src0_code, &ext) || !literal(ctx, ext.words, &ext.src0, 1, &extra))
      return false;
   f->src0 = ext.src0;
   f->ext = ext.ext;
   finish(inst, AGC_ISA_VOP1, op, info->flags, ext.words + extra);
   return true;
}

static bool
decode_vop2(const struct decode_ctx *ctx, uint32_t w, struct agc_isa_inst *inst)
{
   uint16_t op;
   if (!guest_opcode(ctx, AGC_ISA_VOP2, field(w, 25, 0x3f), &op))
      return false;
   const struct agc_isa_op_info *info = agc_isa_op_lookup(AGC_ISA_VOP2, op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_VOP2, op);
   struct agc_isa_vop2 *f = &inst->vop2;
   f->vdst = agc_isa_operand_vgpr((uint8_t)field(w, 17, 0xff));
   uint16_t vsrc1_field = field(w, 9, 0xff);
   struct vop_ext ext;
   if (!decode_vop_ext(ctx, field(w, 0, 0x1ff), &ext))
      return false;
   if (ext.ext.kind != AGC_ISA_VOP_EXT_SDWA)
      f->vsrc1 = agc_isa_operand_vgpr((uint8_t)vsrc1_field);
   else if (!sdwa_source(ctx, ext.w1, 31, vsrc1_field, &f->vsrc1))
      return false;
   unsigned extra;
   if (!literal(ctx, ext.words, &ext.src0, 1, &extra))
      return false;
   if (info->flags & AGC_ISA_FLAG_LITK) {
      if (extra != 0 || ext.ext.kind != AGC_ISA_VOP_EXT_NONE)
         return fail_operand(ctx, "inline-constant MAD forms take exactly one literal");
      extra = 1;
      if (!read_word(ctx, ext.words, &f->constant))
         return false;
      f->has_constant = true;
   }
   f->src0 = ext.src0;
   f->ext = ext.ext;
   finish(inst, AGC_ISA_VOP2, op, info->flags, ext.words + extra);
   return true;
}

static bool
decode_vopc(const struct decode_ctx *ctx, uint32_t w, struct agc_isa_inst *inst)
{
   uint16_t op;
   if (!guest_opcode(ctx, AGC_ISA_VOPC, field(w, 17, 0xff), &op))
      return false;
   const struct agc_isa_vopc_info *info = agc_isa_vopc_decode(op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_VOPC, op);
   struct agc_isa_vopc *f = &inst->vopc;
   uint16_t vsrc1_field = field(w, 9, 0xff);
   uint16_t src0_code = field(w, 0, 0x1ff);
   f->sdst = agc_isa_operand_special(info->exec ? AGC_ISA_OPERAND_EXEC_LO : AGC_ISA_OPERAND_VCC_LO);
   struct vop_ext ext;
   if (!decode_vop_ext(ctx, src0_code, &ext))
      return false;
   if (src0_code == AGC_ISA_CODE_SDWA) {
      ext.ext.sdwa = sdwa_word(ext.w1, true);
      if (bit(ext.w1, 15) && !info->exec && !scalar_destination(ctx, field(ext.w1, 8, 0x7f), &f->sdst))
         return false;
      if (!sdwa_source(ctx, ext.w1, 31, vsrc1_field, &f->vsrc1))
         return false;
   } else {
      f->vsrc1 = agc_isa_operand_vgpr((uint8_t)vsrc1_field);
   }
   unsigned extra;
   if (!literal(ctx, ext.words, &ext.src0, 1, &extra))
      return false;
   f->src0 = ext.src0;
   f->ext = ext.ext;
   finish(inst, AGC_ISA_VOPC, op, info->flags, ext.words + extra);
   return true;
}

static bool
is_vop3_interp(uint16_t op)
{
   return op == AGC_ISA_VOP3_INTERP_P1_F32 || op == AGC_ISA_VOP3_INTERP_P2_F32 || op == AGC_ISA_VOP3_INTERP_MOV_F32 ||
          op == AGC_ISA_VOP3_INTERP_P1LL_F16 || op == AGC_ISA_VOP3_INTERP_P1LV_F16 || op == AGC_ISA_VOP3_INTERP_P2_F16;
}

static bool
decode_vop3(const struct decode_ctx *ctx, uint32_t w0, struct agc_isa_inst *inst)
{
   uint32_t w1;
   uint16_t op;
   uint64_t flags;
   if (!read_word(ctx, 1, &w1) || !guest_opcode(ctx, AGC_ISA_VOP3, field(w0, 16, 0x3ff), &op))
      return false;
   if (!agc_isa_vop3_flags(op, &flags))
      return fail_unknown(ctx, AGC_ISA_VOP3, op);
   enum agc_isa_family base_family = AGC_ISA_VOP3;
   uint16_t base;
   agc_isa_vop3_base(op, &base_family, &base);
   bool vop3b = (flags & AGC_ISA_FLAG_VOP3B) || (base_family == AGC_ISA_VOP2 && (flags & AGC_ISA_FLAG_CARRY_OUT));
   struct agc_isa_vop3 *f = &inst->vop3;
   uint16_t vdst_code = field(w0, 0, 0xff);
   if (flags & AGC_ISA_FLAG_CMPX)
      f->vdst = raw_operand(ctx, vdst_code);
   else if (base_family != AGC_ISA_VOPC && !(flags & AGC_ISA_FLAG_SDST))
      f->vdst = agc_isa_operand_vgpr((uint8_t)vdst_code);
   else if (!scalar_destination(ctx, vdst_code, &f->vdst))
      return false;
   uint16_t codes[3] = {field(w1, 0, 0x1ff), field(w1, 9, 0x1ff), field(w1, 18, 0x1ff)};
   for (unsigned i = 0; i < 3; i++)
      f->src[i] = raw_operand(ctx, codes[i]);
   unsigned count = 0;
   if (is_vop3_interp(op)) {
      f->src[0] = agc_isa_operand_raw(codes[0]);
      if (op == AGC_ISA_VOP3_INTERP_MOV_F32)
         f->src[1] = agc_isa_operand_raw(codes[1]);
   } else {
      count = agc_isa_vop3_source_count(op, flags);
      for (unsigned i = 0; i < 3 && i < count; i++) {
         if (!source(ctx, codes[i], &f->src[i]))
            return false;
      }
   }
   if (vop3b) {
      if (!scalar_destination(ctx, field(w0, 8, 0x7f), &f->sdst))
         return false;
   } else {
      f->abs = (uint8_t)field(w0, 8, 7);
      f->opsel = (uint8_t)field(w0, 11, 0xf);
   }
   unsigned extra;
   if (!literal(ctx, 2, f->src, count < 3 ? count : 3, &extra))
      return false;
   f->neg = (uint8_t)field(w1, 29, 7);
   f->clamp = bit(w0, 15);
   f->omod = (uint8_t)field(w1, 27, 3);
   finish(inst, AGC_ISA_VOP3, op, flags, 2 + extra);
   return true;
}

static bool
decode_vop3p(const struct decode_ctx *ctx, uint32_t w0, struct agc_isa_inst *inst)
{
   uint32_t w1;
   uint16_t raw = field(w0, 16, 0x7f);
   uint16_t op;
   if (!read_word(ctx, 1, &w1) || !guest_opcode(ctx, AGC_ISA_VOP3P, raw, &op))
      return false;
   bool signed_dot = ctx->host && agc_isa_host_is_signed_dot(ctx->host, raw);
   if (signed_dot && field(w1, 29, 7) != AGC_ISA_SIGNED_DOT_SOURCES)
      return fail_unknown(ctx, AGC_ISA_VOP3P, raw);
   const struct agc_isa_op_info *info = agc_isa_op_lookup(AGC_ISA_VOP3P, op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_VOP3P, op);
   struct agc_isa_vop3p *f = &inst->vop3p;
   uint16_t codes[3] = {field(w1, 0, 0x1ff), field(w1, 9, 0x1ff), field(w1, 18, 0x1ff)};
   unsigned count;
   if (!agc_isa_flag_source_count(info->flags, &count))
      count = 3;
   for (unsigned i = 0; i < 3; i++)
      f->src[i] = raw_operand(ctx, codes[i]);
   for (unsigned i = 0; i < count; i++) {
      if (!source(ctx, codes[i], &f->src[i]))
         return false;
   }
   unsigned extra;
   if (!literal(ctx, 2, f->src, count, &extra))
      return false;
   f->vdst = agc_isa_operand_vgpr((uint8_t)w0);
   f->neg = signed_dot ? 0 : (uint8_t)field(w1, 29, 7);
   f->neg_hi = (uint8_t)field(w0, 8, 7);
   f->opsel = (uint8_t)field(w0, 11, 7);
   f->opsel_hi = (uint8_t)(field(w1, 27, 3) | (field(w0, 14, 1) << 2));
   f->clamp = bit(w0, 15);
   finish(inst, AGC_ISA_VOP3P, op, info->flags, 2 + extra);
   return true;
}

static bool
decode_vintrp(const struct decode_ctx *ctx, uint32_t w, struct agc_isa_inst *inst)
{
   uint16_t op = field(w, 16, 3);
   if (op > AGC_ISA_VINTRP_MOV)
      return fail_unknown(ctx, AGC_ISA_VINTRP, op);
   uint64_t flags = AGC_ISA_FLAG_READS_M0;
   if (op == AGC_ISA_VINTRP_P2)
      flags |= AGC_ISA_FLAG_READS_DST;
   inst->vintrp.vdst = (uint8_t)field(w, 18, 0xff);
   inst->vintrp.vsrc = (uint8_t)w;
   inst->vintrp.attr = (uint8_t)field(w, 10, 0x3f);
   inst->vintrp.chan = (uint8_t)field(w, 8, 3);
   finish(inst, AGC_ISA_VINTRP, op, flags, 1);
   return true;
}

static bool
decode_smem(const struct decode_ctx *ctx, uint32_t w0, struct agc_isa_inst *inst)
{
   uint32_t w1;
   uint16_t op;
   if (!read_word(ctx, 1, &w1) || !guest_opcode(ctx, AGC_ISA_SMEM, field(w0, 18, 0xff), &op))
      return false;
   const struct agc_isa_mem_op *info = agc_isa_smem_op(op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_SMEM, op);
   struct agc_isa_smem *s = &inst->smem;
   uint16_t sdata_code = field(w0, 6, 0x7f);
   if (info->dwords == 0)
      s->sdata = raw_operand(ctx, sdata_code);
   else if (!scalar_destination(ctx, sdata_code, &s->sdata))
      return false;
   uint16_t soffset_code = field(w1, 25, 0x7f);
   if (guest_code(ctx, soffset_code) == AGC_ISA_CODE_GFX10_NULL)
      s->soffset = agc_isa_operand_special(AGC_ISA_OPERAND_NULL);
   else if (!source(ctx, soffset_code, &s->soffset))
      return false;
   s->sbase = (uint8_t)(field(w0, 0, 0x3f) * 2);
   s->offset = sign_extend(w1 & 0x1fffff, 21);
   s->glc = bit(w0, ctx->gfx11 ? 14 : 16);
   s->dlc = bit(w0, ctx->gfx11 ? 13 : 14);
   s->dwords = info->dwords;
   s->buffer = agc_isa_is_buffer_smem(info);
   finish(inst, AGC_ISA_SMEM, op, info->flags, 2);
   return true;
}

static bool
decode_buffer(const struct decode_ctx *ctx, uint32_t w0, enum agc_isa_family family, struct agc_isa_inst *inst)
{
   static const uint8_t gfx10_bits[6] = {12, 13, 14, 15, 54, 55};
   static const uint8_t gfx11_bits[6] = {54, 55, 14, 13, 12, 53};
   uint32_t w1;
   if (!read_word(ctx, 1, &w1))
      return false;
   bool mubuf = family == AGC_ISA_MUBUF;
   uint16_t raw;
   uint8_t format = 0;
   if (mubuf) {
      raw = ctx->gfx11 ? field(w0, 18, 0xff) : (uint16_t)(field(w0, 18, 0x7f) | (field(w0, 25, 1) << 7));
   } else {
      raw = ctx->gfx11 ? field(w0, 15, 0xf) : (uint16_t)(field(w0, 16, 7) | (field(w1, 21, 1) << 3));
      format = (uint8_t)field(w0, 19, 0x7f);
   }
   uint16_t op;
   if (!guest_opcode(ctx, family, raw, &op))
      return false;
   const struct agc_isa_mem_op *info = mubuf ? agc_isa_mubuf_op(op) : agc_isa_mtbuf_op(op);
   if (!info)
      return fail_unknown(ctx, family, op);
   if (ctx->host && !mubuf && !agc_isa_host_guest_format(ctx->host, format, &format))
      return fail_unknown(ctx, family, op);
   const uint8_t *bits = ctx->gfx11 ? gfx11_bits : gfx10_bits;
   bool at[6];
   for (unsigned i = 0; i < 6; i++)
      at[i] = bits[i] < 32 ? bit(w0, bits[i]) : bit(w1, bits[i] - 32u);
   struct agc_isa_buffer *b = &inst->buffer;
   if (!source(ctx, field(w1, 24, 0xff), &b->soffset))
      return false;
   b->vdata = (uint8_t)field(w1, 8, 0xff);
   b->vaddr = (uint8_t)w1;
   b->srsrc = (uint8_t)(field(w1, 16, 0x1f) * 4);
   b->offset = field(w0, 0, 0xfff);
   b->offen = at[0];
   b->idxen = at[1];
   b->glc = at[2];
   b->dlc = at[3];
   b->slc = at[4];
   b->tfe = at[5];
   b->lds = mubuf && !ctx->gfx11 && bit(w0, 16);
   b->format = format;
   b->kind = info->kind;
   b->data_dwords = info->dwords;
   finish(inst, family, op, info->flags, 2);
   return true;
}

static bool
flat_segment(const struct decode_ctx *ctx, uint32_t w0, unsigned shift, uint8_t *segment)
{
   unsigned value = field(w0, shift, 3);
   if (value > AGC_ISA_SEGMENT_GLOBAL)
      return fail_encoding(ctx, w0);
   *segment = (uint8_t)value;
   return true;
}

static bool
decode_flat_gfx11(const struct decode_ctx *ctx, uint32_t w0, struct agc_isa_inst *inst)
{
   uint32_t w1;
   uint16_t op;
   if (!read_word(ctx, 1, &w1) || !guest_opcode(ctx, AGC_ISA_FLAT, field(w0, 18, 0x7f), &op))
      return false;
   const struct agc_isa_mem_op *info = agc_isa_flat_op(op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_FLAT, op);
   struct agc_isa_flat *f = &inst->flat;
   if (!flat_segment(ctx, w0, 16, &f->segment))
      return false;
   bool sve = bit(w1, 23);
   struct agc_isa_operand reg = agc_isa_operand_none();
   uint16_t saddr_code = field(w1, 16, 0x7f);
   if (saddr_code != AGC_ISA_CODE_GFX11_NULL && !source(ctx, saddr_code, &reg))
      return false;
   if (f->segment == AGC_ISA_SEGMENT_SCRATCH && sve) {
      f->saddr = agc_isa_operand_is_some(reg) ? reg : agc_isa_operand_special(AGC_ISA_OPERAND_NULL);
      f->uses_vaddr = true;
   } else {
      f->saddr = reg;
      f->uses_vaddr = f->segment != AGC_ISA_SEGMENT_SCRATCH && !agc_isa_flat_only_op(op);
   }
   uint32_t raw_offset = w0 & 0x1fff;
   f->vdst = (uint8_t)field(w1, 24, 0xff);
   f->vdata = (uint8_t)field(w1, 8, 0xff);
   f->vaddr = (uint8_t)w1;
   f->offset = f->segment == AGC_ISA_SEGMENT_FLAT ? (int32_t)(raw_offset & 0xfff) : sign_extend(raw_offset, 13);
   f->glc = bit(w0, 14);
   f->dlc = bit(w0, 13);
   f->slc = bit(w0, 15);
   f->kind = info->kind;
   f->data_dwords = info->dwords;
   finish(inst, AGC_ISA_FLAT, op, info->flags, 2);
   return true;
}

static bool
decode_flat(const struct decode_ctx *ctx, uint32_t w0, struct agc_isa_inst *inst)
{
   if (ctx->gfx11)
      return decode_flat_gfx11(ctx, w0, inst);
   uint32_t w1;
   uint16_t op;
   if (!read_word(ctx, 1, &w1) || !guest_opcode(ctx, AGC_ISA_FLAT, field(w0, 18, 0x7f), &op))
      return false;
   const struct agc_isa_mem_op *info = agc_isa_flat_op(op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_FLAT, op);
   struct agc_isa_flat *f = &inst->flat;
   if (!flat_segment(ctx, w0, 14, &f->segment))
      return false;
   bool addtid = agc_isa_flat_only_op(op);
   bool global_only = addtid || op == AGC_ISA_VMEM_ATOMIC_CSUB;
   bool no_scratch_form = f->segment == AGC_ISA_SEGMENT_SCRATCH && info->kind == AGC_ISA_MEM_ATOMIC;
   if ((global_only && f->segment != AGC_ISA_SEGMENT_GLOBAL) || no_scratch_form)
      return fail_unknown(ctx, AGC_ISA_FLAT, op);
   uint16_t saddr_code = field(w1, 16, 0x7f);
   if (saddr_code == FLAT_SADDR_OFF)
      f->saddr = agc_isa_operand_none();
   else if (saddr_code == FLAT_SADDR_NULL)
      f->saddr = agc_isa_operand_special(AGC_ISA_OPERAND_NULL);
   else if (!source(ctx, saddr_code, &f->saddr))
      return false;
   f->uses_vaddr = !addtid && (f->segment != AGC_ISA_SEGMENT_SCRATCH || f->saddr.kind == AGC_ISA_OPERAND_NULL);
   uint32_t raw_offset = w0 & 0xfff;
   f->vdst = (uint8_t)field(w1, 24, 0xff);
   f->vdata = (uint8_t)field(w1, 8, 0xff);
   f->vaddr = (uint8_t)w1;
   f->offset = f->segment == AGC_ISA_SEGMENT_FLAT ? (int32_t)raw_offset : sign_extend(raw_offset, 12);
   f->glc = bit(w0, 16);
   f->dlc = bit(w0, 12);
   f->slc = bit(w0, 17);
   f->lds = bit(w0, 13);
   f->kind = info->kind;
   f->data_dwords = info->dwords;
   finish(inst, AGC_ISA_FLAT, op, info->flags, 2);
   return true;
}

static bool
decode_ds(const struct decode_ctx *ctx, uint32_t w0, struct agc_isa_inst *inst)
{
   uint32_t w1;
   uint16_t op;
   if (!read_word(ctx, 1, &w1) || !guest_opcode(ctx, AGC_ISA_DS, field(w0, 18, 0xff), &op))
      return false;
   const struct agc_isa_ds_op *info = agc_isa_ds_op(op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_DS, op);
   bool gds = bit(w0, 17);
   bool writes_addr = (info->flags & AGC_ISA_DS_FLAG_STACK8) != 0;
   uint32_t capacity = w0 & 0xffff;
   if (writes_addr && (gds || bit(w0, 16) || capacity < 8 || capacity > 31 || field(w1, 16, 0xff) > 248 ||
                       (uint8_t)w1 == (uint8_t)field(w1, 24, 0xff)))
      return fail_operand(ctx, "Pro push8/pop1 requires LDS, capacity 8..31, a complete eight-register tuple and "
                               "distinct outputs");
   uint64_t flags = 0;
   if (gds || (info->flags & (AGC_ISA_DS_FLAG_GWS | AGC_ISA_DS_FLAG_GDS_ONLY)))
      flags |= AGC_ISA_FLAG_READS_M0;
   struct agc_isa_ds *d = &inst->ds;
   d->vdst = (uint8_t)field(w1, 24, 0xff);
   d->addr = (uint8_t)w1;
   d->data0 = (uint8_t)field(w1, 8, 0xff);
   d->data1 = (uint8_t)field(w1, 16, 0xff);
   d->offset0 = (uint8_t)w0;
   d->offset1 = (uint8_t)field(w0, 8, 0xff);
   d->gds = gds;
   d->dst_dwords = info->dst;
   d->data0_dwords = info->data0;
   d->data1_dwords = info->data1;
   d->uses_addr = !(info->flags & AGC_ISA_DS_FLAG_NO_ADDR);
   d->writes_addr = writes_addr;
   finish(inst, AGC_ISA_DS, op, flags, 2);
   return true;
}

static void
fill_image(struct agc_isa_image *m, const struct agc_isa_image_op *info, uint32_t w1, unsigned nsa_words)
{
   m->vdata = (uint8_t)field(w1, 8, 0xff);
   m->nsa = nsa_words > 0;
   m->nsa_words = (uint8_t)nsa_words;
   m->addr_dwords = agc_isa_image_address_dwords(info->flags, m->dim, m->a16);
   m->srsrc = (uint8_t)(field(w1, 16, 0x1f) * 4);
   m->uses_sampler = agc_isa_image_op_uses_sampler(info);
   m->kind = (uint8_t)agc_isa_image_op_kind(info);
   m->data_dwords = agc_isa_image_data_dwords(info->flags, m->dmask, m->d16);
}

static bool
decode_mimg_gfx11(const struct decode_ctx *ctx, uint32_t w0, struct agc_isa_inst *inst)
{
   uint32_t w1;
   uint16_t op;
   if (!read_word(ctx, 1, &w1) || !guest_opcode(ctx, AGC_ISA_MIMG, field(w0, 18, 0xff), &op))
      return false;
   const struct agc_isa_image_op *info = agc_isa_image_op(op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_MIMG, op);
   struct agc_isa_image *m = &inst->mimg;
   unsigned nsa_words = bit(w0, 0) ? 1 : 0;
   m->dim = (uint8_t)field(w0, 2, 7);
   m->a16 = bit(w0, 16);
   m->d16 = bit(w0, 17);
   m->dmask = (uint8_t)field(w0, 8, 0xf);
   m->addr[0] = (uint8_t)w1;
   if (nsa_words) {
      uint32_t w2;
      if (!read_word(ctx, 2, &w2))
         return false;
      for (unsigned i = 0; i < 4; i++)
         m->addr[1 + i] = (uint8_t)(w2 >> (i * 8));
   }
   if (nsa_words && (info->flags & AGC_ISA_IMG_BVH)) {
      const uint8_t widths[5] = {(info->flags & AGC_ISA_IMG_BVH64) ? 2 : 1, 1, 3, 3, m->a16 ? 0 : 3};
      uint8_t starts[5];
      memcpy(starts, m->addr, sizeof(starts));
      unsigned dword = 0;
      for (unsigned group = 0; group < 5; group++) {
         for (unsigned k = 0; k < widths[group]; k++)
            m->addr[dword++] = (uint8_t)(starts[group] + k);
      }
   }
   m->unorm = bit(w0, 7);
   m->glc = bit(w0, 14);
   m->slc = bit(w0, 12);
   m->dlc = bit(w0, 13);
   m->r128 = bit(w0, 15);
   m->tfe = bit(w1, 21);
   m->lwe = bit(w1, 22);
   m->ssamp = (uint8_t)(field(w1, 26, 0x3f) * 4);
   fill_image(m, info, w1, nsa_words);
   finish(inst, AGC_ISA_MIMG, op, (info->flags & AGC_ISA_IMG_CMPSWAP) ? AGC_ISA_FLAG_CMPSWAP : 0, 2 + nsa_words);
   return true;
}

static bool
decode_mimg(const struct decode_ctx *ctx, uint32_t w0, struct agc_isa_inst *inst)
{
   if (ctx->gfx11)
      return decode_mimg_gfx11(ctx, w0, inst);
   uint32_t w1;
   if (!read_word(ctx, 1, &w1))
      return false;
   uint16_t op = (uint16_t)(field(w0, 18, 0x7f) | ((w0 & 1u) << 7));
   const struct agc_isa_image_op *info = agc_isa_image_op(op);
   if (!info)
      return fail_unknown(ctx, AGC_ISA_MIMG, op);
   struct agc_isa_image *m = &inst->mimg;
   unsigned nsa_words = field(w0, 1, 3);
   m->dim = (uint8_t)field(w0, 3, 7);
   m->a16 = bit(w1, 30);
   m->d16 = bit(w1, 31);
   m->dmask = (uint8_t)field(w0, 8, 0xf);
   m->addr[0] = (uint8_t)w1;
   for (unsigned i = 0; i < nsa_words * 4; i++) {
      uint32_t word;
      if (!read_word(ctx, 2 + i / 4, &word))
         return false;
      m->addr[1 + i] = (uint8_t)(word >> ((i % 4) * 8));
   }
   m->bvh8 = (info->flags & AGC_ISA_IMG_BVH8) != 0;
   m->unorm = bit(w0, 12);
   m->glc = bit(w0, 13);
   m->slc = bit(w0, 25);
   m->dlc = bit(w0, 7);
   m->r128 = bit(w0, 15);
   m->tfe = bit(w0, 16);
   m->lwe = bit(w0, 17);
   m->ssamp = (uint8_t)(field(w1, 21, 0x1f) * 4);
   fill_image(m, info, w1, nsa_words);
   if (m->bvh8) {
      bool canonical = nsa_words == 2 && m->dim == 0 && !m->a16 && !m->d16 && m->unorm && m->r128 &&
                       m->dmask == 0xf && !m->tfe && !m->lwe && !bit(w0, 6) && !m->dlc && !m->glc &&
                       !bit(w0, 14) && !m->slc && !(w1 & 0xffe00000u) && m->addr[0] != 0xff &&
                       m->addr[0] + 1 == m->addr[1] && m->addr[2] != 0xff && m->addr[2] + 1 == m->addr[3] &&
                       m->addr[4] <= 253 && m->addr[5] <= 253 && m->vdata <= 246;
      if (!canonical)
         return fail_operand(ctx, "BVH8 requires canonical NSA2, complete register groups, DMASK=0xF, R128, UNORM "
                                  "and no cache, coordinate or status modifiers");
   } else if (nsa_words > 0 && m->addr_dwords > 1 + nsa_words * 4) {
      return fail_operand(ctx, "MIMG NSA payload is shorter than the address");
   }
   finish(inst, AGC_ISA_MIMG, op, (info->flags & AGC_ISA_IMG_CMPSWAP) ? AGC_ISA_FLAG_CMPSWAP : 0, 2 + nsa_words);
   return true;
}

static bool
decode_exp(const struct decode_ctx *ctx, uint32_t w0, struct agc_isa_inst *inst)
{
   uint32_t w1;
   if (!read_word(ctx, 1, &w1))
      return false;
   uint32_t reserved = ctx->gfx11 ? EXP_RESERVED_BITS & ~(1u << 13) : EXP_RESERVED_BITS;
   if (w0 & reserved)
      return fail_encoding(ctx, w0);
   struct agc_isa_export *e = &inst->exp;
   e->target = (uint8_t)field(w0, 4, 0x3f);
   e->en = (uint8_t)field(w0, 0, 0xf);
   e->compr = !ctx->gfx11 && bit(w0, 10);
   e->done = bit(w0, 11);
   e->vm = !ctx->gfx11 && bit(w0, 12);
   e->row_en = ctx->gfx11 && bit(w0, 13);
   for (unsigned i = 0; i < 4; i++)
      e->vsrc[i] = (uint8_t)(w1 >> (i * 8));
   finish(inst, AGC_ISA_EXP, 0, 0, 2);
   return true;
}

static bool
decode_ldsdir(uint32_t w, struct agc_isa_inst *inst)
{
   inst->ldsdir.vdst = (uint8_t)w;
   inst->ldsdir.attr = (uint8_t)field(w, 10, 0x3f);
   inst->ldsdir.chan = (uint8_t)field(w, 8, 3);
   inst->ldsdir.wait_vdst = (uint8_t)field(w, 16, 0xf);
   finish(inst, AGC_ISA_LDSDIR, field(w, 20, 3), AGC_ISA_FLAG_READS_M0, 1);
   return true;
}

static bool
decode_vinterp(const struct decode_ctx *ctx, uint32_t w0, struct agc_isa_inst *inst)
{
   uint32_t w1;
   struct agc_isa_vinterp *f = &inst->vinterp;
   if (!read_word(ctx, 1, &w1) || !source(ctx, field(w1, 0, 0x1ff), &f->src[0]) ||
       !source(ctx, field(w1, 9, 0x1ff), &f->src[1]) || !source(ctx, field(w1, 18, 0x1ff), &f->src[2]))
      return false;
   f->vdst = (uint8_t)w0;
   f->neg = (uint8_t)field(w1, 29, 7);
   f->opsel = (uint8_t)field(w0, 11, 0xf);
   f->clamp = bit(w0, 15);
   f->wait_exp = (uint8_t)field(w0, 8, 7);
   finish(inst, AGC_ISA_VINTERP, field(w0, 16, 0x7f), 0, 2);
   return true;
}

bool
agc_isa_family_of(uint32_t word, enum agc_isa_family *family)
{
   if (!(word & 0x80000000u)) {
      unsigned op = (word >> 25) & 0x3f;
      *family = op == 0x3e ? AGC_ISA_VOPC : op == 0x3f ? AGC_ISA_VOP1 : AGC_ISA_VOP2;
      return true;
   }
   if ((word & 0xc0000000u) == 0x80000000u) {
      unsigned op = (word >> 23) & 0x7f;
      *family = op == 0x7d   ? AGC_ISA_SOP1
                : op == 0x7e ? AGC_ISA_SOPC
                : op == 0x7f ? AGC_ISA_SOPP
                : op >= 0x60 ? AGC_ISA_SOPK
                             : AGC_ISA_SOP2;
      return true;
   }
   switch (word >> 26) {
   case 0x32:
      *family = AGC_ISA_VINTRP;
      return true;
   case 0x33:
      *family = AGC_ISA_VOP3P;
      return true;
   case 0x35:
      *family = AGC_ISA_VOP3;
      return true;
   case 0x36:
      *family = AGC_ISA_DS;
      return true;
   case 0x37:
      *family = AGC_ISA_FLAT;
      return true;
   case 0x38:
      *family = AGC_ISA_MUBUF;
      return true;
   case 0x3a:
      *family = AGC_ISA_MTBUF;
      return true;
   case 0x3c:
      *family = AGC_ISA_MIMG;
      return true;
   case 0x3d:
      *family = AGC_ISA_SMEM;
      return true;
   case 0x3e:
      *family = AGC_ISA_EXP;
      return true;
   default:
      return false;
   }
}

static bool
decode_instruction(const uint32_t *words, size_t word_count, size_t index, const struct agc_isa_host_tables *host,
                   struct agc_isa_inst *inst, struct agc_isa_error *error)
{
   if (index > UINT32_MAX / 4)
      return fail(error, AGC_ISA_ERROR_TRUNCATED, UINT32_MAX);
   struct decode_ctx ctx = {
      .words = words,
      .word_count = word_count,
      .index = index,
      .pc = (uint32_t)(index * 4),
      .host = host,
      .gfx11 = host && agc_isa_target_is_gfx11_plus(host->target),
      .error = error,
   };
   if (index >= word_count)
      return fail_truncated(&ctx);
   uint32_t word = words[index];
   enum agc_isa_family family;
   if (!agc_isa_family_of(word, &family))
      return fail_encoding(&ctx, word);
   if (ctx.gfx11) {
      if (word >> 24 == 0xcd)
         family = AGC_ISA_VINTERP;
      else if (word >> 24 == 0xce)
         family = AGC_ISA_LDSDIR;
      else if (family == AGC_ISA_VINTRP)
         return fail_encoding(&ctx, word);
   }
   memset(inst, 0, sizeof(*inst));
   inst->pc = ctx.pc;
   switch (family) {
   case AGC_ISA_SOP1:
      return decode_sop1(&ctx, word, inst);
   case AGC_ISA_SOP2:
      return decode_sop2(&ctx, word, inst);
   case AGC_ISA_SOPK:
      return decode_sopk(&ctx, word, inst);
   case AGC_ISA_SOPC:
      return decode_sopc(&ctx, word, inst);
   case AGC_ISA_SOPP:
      return decode_sopp(&ctx, word, inst);
   case AGC_ISA_VOP1:
      return decode_vop1(&ctx, word, inst);
   case AGC_ISA_VOP2:
      return decode_vop2(&ctx, word, inst);
   case AGC_ISA_VOPC:
      return decode_vopc(&ctx, word, inst);
   case AGC_ISA_VOP3:
      return decode_vop3(&ctx, word, inst);
   case AGC_ISA_VOP3P:
      return decode_vop3p(&ctx, word, inst);
   case AGC_ISA_VINTRP:
      return decode_vintrp(&ctx, word, inst);
   case AGC_ISA_SMEM:
      return decode_smem(&ctx, word, inst);
   case AGC_ISA_MUBUF:
   case AGC_ISA_MTBUF:
      return decode_buffer(&ctx, word, family, inst);
   case AGC_ISA_FLAT:
      return decode_flat(&ctx, word, inst);
   case AGC_ISA_DS:
      return decode_ds(&ctx, word, inst);
   case AGC_ISA_MIMG:
      return decode_mimg(&ctx, word, inst);
   case AGC_ISA_EXP:
      return decode_exp(&ctx, word, inst);
   case AGC_ISA_LDSDIR:
      return decode_ldsdir(word, inst);
   case AGC_ISA_VINTERP:
      return decode_vinterp(&ctx, word, inst);
   default:
      return fail_encoding(&ctx, word);
   }
}

bool
agc_isa_decode_at(const uint32_t *words, size_t word_count, size_t index, struct agc_isa_inst *inst,
                  struct agc_isa_error *error)
{
   return decode_instruction(words, word_count, index, NULL, inst, error);
}

struct byte_range {
   uint32_t start;
   uint32_t end;
};

struct skipped_word {
   uint32_t pc;
   struct agc_isa_error error;
};

struct scan {
   struct agc_isa_ir *ir;
   uint64_t *targets;
   uint64_t *starts;
   struct byte_range *runs;
   size_t run_count;
   size_t run_capacity;
   struct skipped_word *skipped;
   size_t skipped_count;
   size_t skipped_capacity;
   size_t furthest_target;
};

static inline bool
test_bit(const uint64_t *bits, size_t index)
{
   return (bits[index / 64] >> (index % 64)) & 1u;
}

static inline void
set_bit(uint64_t *bits, size_t index)
{
   bits[index / 64] |= UINT64_C(1) << (index % 64);
}

static inline void
clear_bit(uint64_t *bits, size_t index)
{
   bits[index / 64] &= ~(UINT64_C(1) << (index % 64));
}

static void
keep(struct scan *scan, const struct agc_isa_inst *inst)
{
   set_bit(scan->starts, inst->pc / 4);
   if (!inst->has_branch_target)
      return;
   size_t index = inst->branch_target / 4;
   if (scan->furthest_target < index)
      scan->furthest_target = index;
   set_bit(scan->targets, index);
}

static bool
grow(void **items, size_t *capacity, size_t count, size_t item_size)
{
   if (count < *capacity)
      return true;
   size_t grown = *capacity ? *capacity * 2 : 16;
   void *resized = realloc(*items, grown * item_size);
   if (!resized)
      return false;
   *items = resized;
   *capacity = grown;
   return true;
}

static bool
keep_run(struct scan *scan, size_t first)
{
   struct agc_isa_ir *ir = scan->ir;
   if (first == ir->instruction_count)
      return true;
   if (!grow((void **)&scan->runs, &scan->run_capacity, scan->run_count, sizeof(*scan->runs)))
      return false;
   struct byte_range *run = &scan->runs[scan->run_count++];
   run->start = ir->instructions[first].pc;
   run->end = agc_isa_inst_end_pc(&ir->instructions[ir->instruction_count - 1]);
   for (size_t i = first; i < ir->instruction_count; i++)
      keep(scan, &ir->instructions[i]);
   return true;
}

static bool
skip(struct scan *scan, uint32_t pc, const struct agc_isa_error *error)
{
   if (!grow((void **)&scan->skipped, &scan->skipped_capacity, scan->skipped_count, sizeof(*scan->skipped)))
      return false;
   scan->skipped[scan->skipped_count].pc = pc;
   scan->skipped[scan->skipped_count].error = *error;
   scan->skipped_count++;
   return true;
}

static void
drop_run(struct scan *scan, struct byte_range run)
{
   struct agc_isa_ir *ir = scan->ir;
   size_t kept = 0;
   for (size_t i = 0; i < ir->instruction_count; i++) {
      const struct agc_isa_inst *inst = &ir->instructions[i];
      if (inst->pc >= run.start && inst->pc < run.end) {
         clear_bit(scan->starts, inst->pc / 4);
         continue;
      }
      if (kept != i)
         ir->instructions[kept] = *inst;
      kept++;
   }
   ir->instruction_count = kept;
}

static bool
resolve_targets(struct scan *scan, struct agc_isa_error *error)
{
   struct agc_isa_ir *ir = scan->ir;
   for (size_t i = 0; i < ir->instruction_count;) {
      const struct agc_isa_inst *inst = &ir->instructions[i];
      if (!inst->has_branch_target || test_bit(scan->starts, inst->branch_target / 4)) {
         i++;
         continue;
      }
      size_t run = 0;
      while (run < scan->run_count && !(inst->pc >= scan->runs[run].start && inst->pc < scan->runs[run].end))
         run++;
      if (run == scan->run_count) {
         for (size_t s = 0; s < scan->skipped_count; s++) {
            if (scan->skipped[s].pc == inst->branch_target) {
               *error = scan->skipped[s].error;
               return false;
            }
         }
         return fail_branch(error, inst->pc, inst->branch_target);
      }
      struct byte_range dropped = scan->runs[run];
      scan->runs[run] = scan->runs[--scan->run_count];
      drop_run(scan, dropped);
      i = 0;
   }
   return true;
}

static bool
scan_program(struct scan *scan, const uint32_t *words, size_t word_count, bool through_jumps,
             struct agc_isa_error *error)
{
   struct agc_isa_ir *ir = scan->ir;
   bool pending = false;
   size_t run_first = 0;
   size_t index = 0;
   bool ended = false;
   size_t end = 0;
   while (index < word_count) {
      uint32_t pc = (uint32_t)(index * 4);
      if (pending && test_bit(scan->targets, index)) {
         if (!keep_run(scan, run_first))
            return fail(error, AGC_ISA_ERROR_OUT_OF_MEMORY, pc);
         pending = false;
      }
      if (!agc_isa_ir_reserve(ir, ir->instruction_count + 1))
         return fail(error, AGC_ISA_ERROR_OUT_OF_MEMORY, pc);
      struct agc_isa_inst *inst = &ir->instructions[ir->instruction_count];
      struct agc_isa_error problem;
      bool decoded = decode_instruction(words, word_count, index, NULL, inst, &problem);
      if (decoded && inst->has_branch_target &&
          (inst->branch_target % 4 != 0 || inst->branch_target / 4 >= word_count))
         decoded = fail_branch(&problem, pc, inst->branch_target);
      if (!decoded) {
         if (!pending) {
            *error = problem;
            return false;
         }
         ir->instruction_count = run_first;
         if (!skip(scan, pc, &problem))
            return fail(error, AGC_ISA_ERROR_OUT_OF_MEMORY, pc);
         index++;
         continue;
      }
      if (pending) {
         size_t swallowed = index + 1;
         while (swallowed < index + inst->word_count && !test_bit(scan->targets, swallowed))
            swallowed++;
         if (swallowed < index + inst->word_count) {
            ir->instruction_count = run_first;
            index = swallowed;
            continue;
         }
      }
      index += inst->word_count;
      bool transfers = !agc_isa_inst_falls_through(inst);
      ir->instruction_count++;
      if (pending) {
         if (!transfers)
            continue;
         if (!keep_run(scan, run_first))
            return fail(error, AGC_ISA_ERROR_OUT_OF_MEMORY, pc);
         run_first = ir->instruction_count;
      } else {
         keep(scan, inst);
         if (transfers) {
            pending = true;
            run_first = ir->instruction_count;
         }
      }
      if (!through_jumps && transfers && scan->furthest_target < index) {
         ended = true;
         end = index;
         break;
      }
   }
   if (pending)
      ir->instruction_count = run_first;
   if (!resolve_targets(scan, error))
      return false;
   if (through_jumps) {
      ended = ir->instruction_count > 0 &&
              !agc_isa_inst_falls_through(&ir->instructions[ir->instruction_count - 1]);
      if (ended) {
         const struct agc_isa_inst *last = &ir->instructions[ir->instruction_count - 1];
         end = last->pc / 4 + last->word_count;
      }
   }
   if (!ended)
      return fail(error, AGC_ISA_ERROR_MISSING_END_PROGRAM, 0);
   ir->code_words = end;
   return true;
}

bool
agc_isa_decode_scoped(const uint32_t *words, size_t word_count, const struct agc_isa_decode_scope *scope,
                      struct agc_isa_ir *ir, struct agc_isa_error *error)
{
   agc_isa_ir_clear(ir);
   if (word_count == 0)
      return fail(error, AGC_ISA_ERROR_EMPTY_PROGRAM, 0);
   size_t bitmap_words = word_count / 64 + 1;
   struct scan scan = {.ir = ir};
   scan.targets = calloc(bitmap_words * 2, sizeof(uint64_t));
   if (!scan.targets)
      return fail(error, AGC_ISA_ERROR_OUT_OF_MEMORY, 0);
   scan.starts = scan.targets + bitmap_words;
   for (size_t i = 0; i < scope->entry_count; i++) {
      uint32_t pc = scope->entries[i];
      if (pc % 4 != 0 || pc / 4 >= word_count)
         continue;
      set_bit(scan.targets, pc / 4);
      if (scan.furthest_target < pc / 4)
         scan.furthest_target = pc / 4;
   }
   bool ok = scan_program(&scan, words, word_count, scope->through_jumps, error);
   free(scan.targets);
   free(scan.runs);
   free(scan.skipped);
   if (!ok)
      agc_isa_ir_clear(ir);
   return ok;
}

bool
agc_isa_decode(const uint32_t *words, size_t word_count, struct agc_isa_ir *ir, struct agc_isa_error *error)
{
   static const struct agc_isa_decode_scope scope = {NULL, 0, false};
   return agc_isa_decode_scoped(words, word_count, &scope, ir, error);
}

static void
host_insert(struct agc_isa_host_tables *tables, enum agc_isa_family family, uint16_t host, uint16_t guest)
{
   if (host >= AGC_ISA_OPCODE_LIMIT)
      return;
   uint16_t *entry = &tables->opcodes[family][host];
   if (*entry == AGC_ISA_HOST_NO_OPCODE || guest == host)
      *entry = guest;
}

static void
host_add_mem_table(struct agc_isa_host_tables *tables, enum agc_isa_family family, enum agc_isa_mem_table which)
{
   size_t count;
   const struct agc_isa_mem_op *ops = agc_isa_mem_table(which, &count);
   for (size_t i = 0; i < count; i++) {
      struct agc_isa_xlat x = agc_isa_mem_op_translate(&ops[i], tables->target);
      if (x.kind == AGC_ISA_XLAT_OP)
         host_insert(tables, family, x.value, ops[i].op);
   }
}

void
agc_isa_host_tables_init(struct agc_isa_host_tables *tables, enum agc_isa_target target)
{
   static const enum agc_isa_family op_families[9] = {
      AGC_ISA_SOP1, AGC_ISA_SOP2, AGC_ISA_SOPK, AGC_ISA_SOPC,  AGC_ISA_SOPP,
      AGC_ISA_VOP1, AGC_ISA_VOP2, AGC_ISA_VOP3, AGC_ISA_VOP3P,
   };
   memset(tables, 0xff, sizeof(*tables));
   tables->target = target;
   tables->signed_dot_count = 0;
   for (unsigned f = 0; f < 9; f++) {
      enum agc_isa_family family = op_families[f];
      size_t count;
      const struct agc_isa_op_info *ops = agc_isa_op_table(family, &count);
      for (size_t i = 0; i < count; i++) {
         struct agc_isa_xlat x = agc_isa_op_translate(&ops[i], target);
         bool signed_dot = x.kind == AGC_ISA_XLAT_RULE && x.rule == AGC_ISA_RULE_DOT_SIGNED;
         bool keeps_opcode = x.kind == AGC_ISA_XLAT_RULE &&
                             (x.rule == AGC_ISA_RULE_HWREG || x.rule == AGC_ISA_RULE_SENDMSG);
         if (x.kind != AGC_ISA_XLAT_OP && !keeps_opcode && !signed_dot)
            continue;
         if (signed_dot && tables->signed_dot_count < sizeof(tables->signed_dots) / sizeof(tables->signed_dots[0]))
            tables->signed_dots[tables->signed_dot_count++] = x.value;
         host_insert(tables, family, x.value, ops[i].op);
         if ((family != AGC_ISA_VOP1 && family != AGC_ISA_VOP2) || (ops[i].flags & AGC_ISA_FLAG_NO_VOP3))
            continue;
         uint16_t promoted = family == AGC_ISA_VOP1 ? AGC_ISA_VOP3_VOP1_BASE : AGC_ISA_VOP3_VOP2_BASE;
         host_insert(tables, AGC_ISA_VOP3, (uint16_t)(promoted + x.value), (uint16_t)(promoted + ops[i].op));
      }
   }
   host_add_mem_table(tables, AGC_ISA_SMEM, AGC_ISA_MEM_TABLE_SMEM);
   host_add_mem_table(tables, AGC_ISA_MUBUF, AGC_ISA_MEM_TABLE_VMEM_COMMON);
   host_add_mem_table(tables, AGC_ISA_MUBUF, AGC_ISA_MEM_TABLE_MUBUF_ONLY);
   host_add_mem_table(tables, AGC_ISA_MTBUF, AGC_ISA_MEM_TABLE_MTBUF);
   host_add_mem_table(tables, AGC_ISA_FLAT, AGC_ISA_MEM_TABLE_VMEM_COMMON);
   host_add_mem_table(tables, AGC_ISA_FLAT, AGC_ISA_MEM_TABLE_FLAT_ONLY);
   size_t count;
   const struct agc_isa_ds_op *ds = agc_isa_ds_table(&count);
   for (size_t i = 0; i < count; i++) {
      struct agc_isa_xlat x = agc_isa_ds_op_translate(&ds[i], target);
      if (x.kind == AGC_ISA_XLAT_OP)
         host_insert(tables, AGC_ISA_DS, x.value, ds[i].op);
   }
   const struct agc_isa_image_op *images = agc_isa_image_table(&count);
   for (size_t i = 0; i < count; i++) {
      if (images[i].flags & AGC_ISA_IMG_ADJUST)
         continue;
      struct agc_isa_xlat x = agc_isa_image_op_translate(&images[i], target);
      if (x.kind == AGC_ISA_XLAT_OP)
         host_insert(tables, AGC_ISA_MIMG, x.value, images[i].op);
   }
   if (agc_isa_target_is_gfx11_plus(target))
      host_insert(tables, AGC_ISA_SOPP, AGC_ISA_GFX11_SOPP_WAITCNT, AGC_ISA_SOPP_WAITCNT);
   for (uint16_t op = 0; op <= 0xff; op++) {
      const struct agc_isa_vopc_info *info = agc_isa_vopc_decode(op);
      if (!info)
         continue;
      uint16_t host = agc_isa_target_is_gfx11_plus(target) ? info->gfx11_opcode : op;
      if (host <= 0xff && tables->vopc[host] == AGC_ISA_HOST_NO_OPCODE)
         tables->vopc[host] = op;
   }
   for (unsigned format = 0; format <= 0x7f; format++) {
      uint8_t host;
      if (agc_isa_remap_buffer_format((uint8_t)format, target, &host) &&
          tables->formats[host] == AGC_ISA_HOST_NO_FORMAT)
         tables->formats[host] = (uint8_t)format;
   }
}

bool
agc_isa_host_guest_opcode(const struct agc_isa_host_tables *tables, enum agc_isa_family family, uint16_t host,
                          uint16_t *guest)
{
   uint16_t found = AGC_ISA_HOST_NO_OPCODE;
   if (family == AGC_ISA_VOPC || (family == AGC_ISA_VOP3 && host <= AGC_ISA_VOP3_VOPC_LAST)) {
      if (host <= 0xff)
         found = tables->vopc[host];
   } else if ((unsigned)family < AGC_ISA_FAMILY_COUNT && host < AGC_ISA_OPCODE_LIMIT) {
      found = tables->opcodes[family][host];
   }
   if (found == AGC_ISA_HOST_NO_OPCODE)
      return false;
   *guest = found;
   return true;
}

bool
agc_isa_host_guest_format(const struct agc_isa_host_tables *tables, uint8_t host, uint8_t *guest)
{
   if (tables->formats[host] == AGC_ISA_HOST_NO_FORMAT)
      return false;
   *guest = tables->formats[host];
   return true;
}

bool
agc_isa_host_is_signed_dot(const struct agc_isa_host_tables *tables, uint16_t host)
{
   for (unsigned i = 0; i < tables->signed_dot_count; i++) {
      if (tables->signed_dots[i] == host)
         return true;
   }
   return false;
}

bool
agc_isa_host_decode(const struct agc_isa_host_tables *tables, const uint32_t *words, size_t word_count,
                    struct agc_isa_ir *ir, struct agc_isa_error *error)
{
   agc_isa_ir_clear(ir);
   if (tables->target != AGC_ISA_TARGET_GFX103 && !agc_isa_target_is_gfx11(tables->target)) {
      fail(error, AGC_ISA_ERROR_INVALID_OPTIONS, 0);
      error->reason = "host decoding covers gfx10.3, gfx11 and gfx11.5";
      return false;
   }
   size_t index = 0;
   while (index < word_count) {
      if (!agc_isa_ir_reserve(ir, ir->instruction_count + 1)) {
         agc_isa_ir_clear(ir);
         return fail(error, AGC_ISA_ERROR_OUT_OF_MEMORY, 0);
      }
      struct agc_isa_inst *inst = &ir->instructions[ir->instruction_count];
      if (!decode_instruction(words, word_count, index, tables, inst, error)) {
         agc_isa_ir_clear(ir);
         return false;
      }
      index += inst->word_count;
      ir->instruction_count++;
   }
   for (size_t i = 0; i < ir->instruction_count; i++) {
      const struct agc_isa_inst *inst = &ir->instructions[i];
      size_t found;
      if (inst->has_branch_target && !agc_isa_ir_instruction_at(ir, inst->branch_target, &found)) {
         fail_branch(error, inst->pc, inst->branch_target);
         agc_isa_ir_clear(ir);
         return false;
      }
   }
   ir->code_words = word_count;
   return true;
}
