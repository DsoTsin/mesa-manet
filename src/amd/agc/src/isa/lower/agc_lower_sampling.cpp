#include <string.h>

#include "agc_lower.h"

namespace agc {

static const uint16_t image_get_resinfo = AGC_ISA_MIMG_GET_RESINFO;
static const uint16_t image_get_lod = 0x60;
static const uint32_t sampler_unnormalized = 15u | (1u << 16);
static const uint32_t float_half = 0x3f000000u;
static const uint32_t float_one = 0x3f800000u;

struct opcode_pair {
   uint16_t from;
   uint16_t to;
};

static const opcode_pair coarse_to_fine[] = {
   {0x68, 0x22}, {0x69, 0x23}, {0x6a, 0x2a}, {0x6b, 0x2b}, {0x6c, 0x32}, {0x6d, 0x33}, {0x6e, 0x3a}, {0x6f, 0x3b},
   {0xe8, 0xa2}, {0xe9, 0xa3}, {0xea, 0xaa}, {0xeb, 0xab}, {0xec, 0xb2}, {0xed, 0xb3}, {0xee, 0xba}, {0xef, 0xbb},
};

static const opcode_pair gather_plain_ops[] = {
   {0x51, 0x41}, {0x54, 0x44}, {0x55, 0x45}, {0x56, 0x46}, {0x58, 0x48},
   {0x59, 0x49}, {0x5c, 0x4c}, {0x5d, 0x4d}, {0x5e, 0x4e}, {0xd8, 0xc8},
};

struct gather {
   lower_ctx &ctx;
   const agc_isa_inst *origin;
   agc_isa_image m;
};

template <size_t N>
static bool
replacement(const opcode_pair (&table)[N], uint16_t opcode, uint16_t *to)
{
   for (const opcode_pair &pair : table) {
      if (pair.from == opcode) {
         *to = pair.to;
         return true;
      }
   }
   return false;
}

static uint8_t
image_gradients(uint8_t dim)
{
   switch (dim) {
   case 0:
   case 4:
      return 1;
   case 2:
      return 3;
   default:
      return 2;
   }
}

static void
spread_addresses(agc_isa_image *m)
{
   if (m->nsa)
      return;
   for (unsigned k = 1; k < m->addr_dwords; k++)
      m->addr[k] = (uint8_t)(m->addr[0] + k);
   m->nsa = true;
}

bool
per_quad_derivatives(lower_ctx &ctx, const agc_isa_inst *inst, const struct agc_isa_image_op *info, agc_isa_inst *fine)
{
   uint16_t opcode;
   if (!replacement(coarse_to_fine, inst->opcode, &opcode))
      return false;
   if (inst->fields != AGC_ISA_MIMG)
      fail(ctx, inst, "malformed MIMG");
   agc_isa_image m = inst->mimg;
   unsigned first = ((info->flags & AGC_ISA_IMG_OFFSET) ? 1 : 0) + ((info->flags & AGC_ISA_IMG_COMPARE) ? 1 : 0);
   uint8_t grads = image_gradients(m.dim);
   uint8_t dwords = (info->flags & AGC_ISA_IMG_G16) || m.a16 ? (uint8_t)(2 * ((grads + 1) / 2)) : (uint8_t)(2 * grads);
   spread_addresses(&m);
   uint8_t copies = scratch_vgprs(ctx, dwords);
   for (unsigned k = 0; k < dwords; k++) {
      uint8_t copy = (uint8_t)(copies + k);
      quad_broadcast(ctx, inst, copy, agc_isa_operand_vgpr(m.addr[first + k]), 0);
      m.addr[first + k] = copy;
   }
   *fine = *inst;
   fine->opcode = opcode;
   fine->mimg = m;
   fine->word_count = 0;
   return true;
}

static void
query(gather &e, uint16_t opcode, unsigned vdata, uint8_t dmask, const uint8_t *addresses, unsigned count)
{
   const struct agc_isa_image_op *info = agc_isa_image_op(opcode);
   if (!info)
      fail(e.ctx, e.origin, "unknown MIMG opcode");
   agc_isa_image q = e.m;
   q.vdata = (uint8_t)vdata;
   q.dmask = dmask;
   memset(q.addr, 0, sizeof(q.addr));
   memcpy(q.addr, addresses, count);
   q.addr_dwords = (uint8_t)count;
   q.nsa = count > 1;
   q.nsa_words = 0;
   q.uses_sampler = agc_isa_image_op_uses_sampler(info);
   q.kind = (uint8_t)agc_isa_image_op_kind(info);
   q.data_dwords = (uint8_t)util_bitcount(dmask);
   q.d16 = false;
   q.tfe = false;
   q.lwe = false;
   q.glc = false;
   q.slc = false;
   q.dlc = false;
   agc_isa_inst inst = agc_isa_tmpl(AGC_ISA_MIMG, opcode);
   inst.mimg = q;
   lower_template(e.ctx, e.origin, inst);
   waitcnt(e.ctx, 0, AGC_ISA_GFX10_EXPCNT_MAX, AGC_ISA_GFX10_LGKMCNT_MAX);
}

static void
query_lod(gather &e, unsigned lod, unsigned coordinates)
{
   unsigned count = agc_isa_image_coordinates(e.m.dim);
   uint8_t addresses[AGC_ISA_MAX_NSA_ADDRESSES];
   memcpy(addresses, &e.m.addr[coordinates], count);
   query(e, image_get_lod, lod, 0x3, addresses, count);
}

static void
resinfo(gather &e, unsigned dims)
{
   e.ctx.bld.vop1(aco_opcode::v_mov_b32, vdef(dims), imm(0));
   uint8_t address = (uint8_t)dims;
   query(e, image_get_resinfo, dims, 0xb, &address, 1);
}

static unsigned
sampler_flag(gather &e, uint32_t field)
{
   unsigned s = transient_sgprs(e.ctx, e.origin, 2);
   save_scc(e.ctx, s + 1);
   e.ctx.bld.sop2(aco_opcode::s_bfe_u32, sdef(s), scc_def(), sop(e.m.ssamp), lit(field));
   restore_scc(e.ctx, s + 1);
   return s;
}

bool
gather_without_offset(lower_ctx &ctx, const agc_isa_inst *inst, const struct agc_isa_image_op *info,
                      agc_isa_inst *plain_gather)
{
   uint16_t plain;
   if (!replacement(gather_plain_ops, inst->opcode, &plain))
      return false;
   if (inst->fields != AGC_ISA_MIMG)
      fail(ctx, inst, "malformed MIMG");
   agc_isa_image m = inst->mimg;
   if (m.a16)
      fail(ctx, inst, "a gather with 16-bit addresses and an offset has no form on the target");
   spread_addresses(&m);
   uint32_t flags = info->flags;
   unsigned bias = (flags & AGC_ISA_IMG_BIAS) ? 1 : 0;
   unsigned compare = (flags & AGC_ISA_IMG_COMPARE) ? 1 : 0;
   unsigned coordinates = 1 + bias + compare;
   unsigned lod_operand = coordinates + agc_isa_image_coordinates(m.dim);
   const unsigned coordinate_registers[2] = {m.addr[coordinates], m.addr[coordinates + 1]};
   unsigned offset = m.addr[0];
   Builder &bld = ctx.bld;
   gather e = {ctx, inst, m};
   unsigned v = scratch_vgprs(ctx, 10);
   unsigned lod = v;
   unsigned dims = v + 2;
   unsigned level = v + 5;
   unsigned moved = v + 6;
   unsigned tmp = v + 8;
   if (flags & AGC_ISA_IMG_LOD) {
      bld.vop1(aco_opcode::v_mov_b32, vdef(lod), vop(m.addr[lod_operand]));
   } else {
      query_lod(e, lod, coordinates);
      if (flags & AGC_ISA_IMG_BIAS)
         bld.vop2(aco_opcode::v_add_f32, vdef(lod), vop(m.addr[1]), vop(lod + 1));
   }
   if (flags & AGC_ISA_IMG_CLAMP)
      bld.vop2(aco_opcode::v_max_f32, vdef(lod), vop(m.addr[lod_operand]), vop(lod));
   resinfo(e, dims);
   bld.vop2(aco_opcode::v_add_f32, vdef(lod), imm(float_half), vop(lod));
   bld.vop1(aco_opcode::v_floor_f32, vdef(lod), vop(lod));
   bld.vop1(aco_opcode::v_cvt_u32_f32, vdef(level), vop(lod));
   bld.vop2(aco_opcode::v_max_u32, vdef(dims + 2), imm(1), vop(dims + 2));
   bld.vop2(aco_opcode::v_add_u32, vdef(dims + 2), imm(0xffffffffu), vop(dims + 2));
   bld.vop2(aco_opcode::v_min_u32, vdef(level), vop(dims + 2), vop(level));
   unsigned unnormalized = sampler_flag(e, sampler_unnormalized);
   bld.vop1(aco_opcode::v_cvt_f32_u32, vdef(tmp), sop(unnormalized));
   bld.vop2(aco_opcode::v_sub_f32, vdef(tmp + 1), imm(float_one), vop(tmp));
   for (unsigned k = 0; k < 2; k++) {
      unsigned size = dims + k;
      unsigned target = moved + k;
      bld.vop2(aco_opcode::v_lshrrev_b32, vdef(size), vop(level), vop(size));
      bld.vop2(aco_opcode::v_max_u32, vdef(size), imm(1), vop(size));
      bld.vop1(aco_opcode::v_cvt_f32_u32, vdef(size), vop(size));
      bld.vop1(aco_opcode::v_rcp_f32, vdef(size), vop(size));
      bld.vop3(aco_opcode::v_fma_f32, vdef(size), vop(tmp + 1), vop(size), vop(tmp));
      bld.vop3(aco_opcode::v_bfe_i32, vdef(target), vop(offset), imm(8u * k), imm(6));
      bld.vop1(aco_opcode::v_cvt_f32_i32, vdef(target), vop(target));
      bld.vop3(aco_opcode::v_fma_f32, vdef(target), vop(target), vop(size), vop(coordinate_registers[k]));
   }
   agc_isa_image out = m;
   unsigned dwords = out.addr_dwords;
   memmove(&out.addr[0], &out.addr[1], dwords - 1);
   out.addr_dwords = (uint8_t)(out.addr_dwords - 1);
   out.addr[coordinates - 1] = (uint8_t)moved;
   out.addr[coordinates] = (uint8_t)(moved + 1);
   *plain_gather = *inst;
   plain_gather->opcode = plain;
   plain_gather->mimg = out;
   plain_gather->word_count = 0;
   return true;
}

}
