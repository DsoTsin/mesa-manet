#include <string.h>

#include "agc_lower.h"

namespace agc {

static const uint32_t f32_inf = 0x7f800000u;
static const uint32_t f32_one = 0x3f800000u;
static const uint16_t hwreg_mode = 1;

bool
bvh_writes_mode(const agc_isa_inst *inst)
{
   switch (inst->fields) {
   case AGC_ISA_SOPP:
      return inst->opcode == AGC_ISA_SOPP_ROUND_MODE || inst->opcode == AGC_ISA_SOPP_DENORM_MODE;
   case AGC_ISA_SOPK:
      return (inst->opcode == AGC_ISA_SOPK_SETREG_B32 || inst->opcode == AGC_ISA_SOPK_SETREG_IMM32_B32) &&
             (inst->sopk.simm16 & AGC_ISA_HWREG_ID_MASK) == hwreg_mode;
   default:
      return false;
   }
}

static agc_isa_image
grouped(lower_ctx &ctx, const agc_isa_image &m, bool wide, unsigned spare, unsigned *scratch)
{
   Builder &bld = ctx.bld;
   unsigned guest[AGC_ISA_MAX_NSA_ADDRESSES];
   for (unsigned dword = 0; dword < m.addr_dwords; dword++)
      guest[dword] = agc_isa_image_address_register(&m, dword);
   unsigned node = wide ? 2 : 1;
   const unsigned groups[5][2] = {{0, node}, {node, 1}, {node + 1, 3}, {node + 4, 3}, {node + 7, 3}};
   unsigned group_count = m.a16 ? 3 : 5;
   auto consecutive = [&](unsigned first, unsigned len) {
      for (unsigned k = 1; k < len; k++) {
         if (guest[first + k] != guest[first] + k)
            return false;
      }
      return true;
   };
   unsigned copied = 0;
   for (unsigned g = 0; g < group_count; g++) {
      if (!consecutive(groups[g][0], groups[g][1]))
         copied += groups[g][1];
   }
   unsigned packed = m.a16 ? 3 : 0;
   unsigned next = scratch_vgprs(ctx, (uint8_t)(copied + packed + spare));
   *scratch = next + copied + packed;
   agc_isa_image image = m;
   memset(image.addr, 0, sizeof(image.addr));
   unsigned filled = 0;
   for (unsigned g = 0; g < group_count; g++) {
      unsigned first = groups[g][0];
      unsigned len = groups[g][1];
      unsigned base = guest[first];
      if (!consecutive(first, len)) {
         for (unsigned k = 0; k < len; k++)
            bld.vop1(aco_opcode::v_mov_b32, vdef(next + k), vop(guest[first + k]));
         base = next;
         next += len;
      }
      for (unsigned k = 0; k < len; k++)
         image.addr[filled + k] = (uint8_t)(base + k);
      filled += len;
   }
   if (m.a16) {
      Operand half[3] = {vop(guest[node + 4]), vop(guest[node + 5]), vop(guest[node + 6])};
      bld.vop3(aco_opcode::v_bfi_b32, vdef(next), lit(0xffff), half[0], half[1]);
      bld.vop3(aco_opcode::v_alignbit_b32, vdef(next + 1), half[2], half[0], imm(16));
      bld.vop3(aco_opcode::v_bfi_b32, vdef(next + 2), lit(0xffff), half[1], half[2]);
      for (unsigned k = 0; k < 3; k++)
         image.addr[filled + k] = (uint8_t)(next + k);
   }
   image.nsa = true;
   image.nsa_words = 1;
   return image;
}

void
lower_bvh(lower_ctx &ctx, const agc_isa_inst *inst)
{
   if (inst->fields != AGC_ISA_MIMG)
      fail(ctx, inst, "malformed MIMG");
   const agc_isa_image m = inst->mimg;
   if (m.dmask != 0xf || !m.r128 || m.d16 || m.tfe || m.lwe)
      fail(ctx, inst, "BVH intersection requires DMASK=0xF, R128=1 and no D16, TFE or LWE");
   if (ctx.ray_unit == AGC_SHADER_RAY_UNIT_REFERENCE)
      fail(ctx, inst, "ray intersection needs the host ray unit");
   const struct agc_isa_image_op *info = agc_isa_image_op(inst->opcode);
   if (!info)
      fail(ctx, inst, "the target has no ray unit");
   agc_isa_xlat xlat = agc_isa_image_op_translate(info, ctx.target);
   if (xlat.kind != AGC_ISA_XLAT_OP)
      fail(ctx, inst, "the target has no ray unit");
   agc_isa_inst host = agc_isa_inst_retarget(inst, xlat.value);
   if (!ctx.gfx11_plus) {
      push(ctx, host);
      return;
   }
   Builder &bld = ctx.bld;
   unsigned other;
   host.mimg = grouped(ctx, m, (info->flags & AGC_ISA_IMG_BVH64) != 0, 2, &other);
   unsigned mask = other + 1;
   unsigned numerator = m.vdata;
   unsigned denominator = m.vdata + 1u;
   bld.vop3(aco_opcode::v_bfe_u32, vdef(other), vop(agc_isa_image_address_register(&m, 0)), imm(2), imm(1));
   push(ctx, host);
   waitcnt(ctx, 0, AGC_ISA_GFX10_EXPCNT_MAX, AGC_ISA_GFX10_LGKMCNT_MAX);
   bld.vop2(aco_opcode::v_and_b32, vdef(mask), lit(0x7fffffffu), vop(numerator));
   bld.vop2(aco_opcode::v_xor_b32, vdef(mask), lit(f32_inf), vop(mask));
   bld.vop2(aco_opcode::v_or_b32, vdef(mask), vop(mask), vop(other));
   bld.vop2(aco_opcode::v_min_u32, vdef(mask), imm(1), vop(mask));
   bld.vop2(aco_opcode::v_add_u32, vdef(mask), imm(0xffffffffu), vop(mask));
   bld.vop3(aco_opcode::v_bfi_b32, vdef(numerator), vop(mask), lit(f32_inf), vop(numerator));
   bld.vop3(aco_opcode::v_bfi_b32, vdef(denominator), vop(mask), imm(f32_one), vop(denominator));
}

}
