#include <string.h>

#include "agc_lower.h"

namespace agc {

static const uint16_t global_load_dword = AGC_ISA_VMEM_LOAD_DWORD;
static const uint16_t global_store_dword = AGC_ISA_VMEM_STORE_DWORD;
static const uint16_t global_atomic_swap = AGC_ISA_VMEM_ATOMIC_SWAP;
static const uint16_t global_atomic_cmpswap = AGC_ISA_VMEM_ATOMIC_CMPSWAP;
static const uint16_t global_atomic_add = AGC_ISA_VMEM_ATOMIC_ADD;
static const uint16_t global_atomic_sub = 0x33;
static const uint8_t ordered_swap = 0x10;
static const unsigned ordered_dwords_shift = 6;
static const char gws_not_emulated[] =
   "GWS is not emulated: ds_gws_* barriers and semaphores across workgroups are rejected";

enum access_kind {
   ACCESS_LOAD,
   ACCESS_STORE,
   ACCESS_ATOMIC,
   ACCESS_CMPSWAP,
   ACCESS_READ2,
   ACCESS_WRITE2,
   ACCESS_APPEND,
   ACCESS_ORDERED,
   ACCESS_NOP
};

struct access {
   enum access_kind kind;
   uint16_t op;
   bool rtn;
   uint32_t stride;
   bool subtract;
};

struct gds_emit {
   lower_ctx &ctx;
   Builder &bld;
   const agc_isa_inst *origin;
   uint64_t buffer;
};

static const uint16_t atomics[12] = {
   global_atomic_add, global_atomic_sub, 0, 0x3c, 0x3d, 0x35, 0x37, 0x36, 0x38, 0x39, 0x3a, 0x3b,
};

static const char *
atomic_access(uint16_t index, bool rtn, access *a)
{
   if (!atomics[index])
      return "GDS reverse subtracts are not emulated";
   a->kind = ACCESS_ATOMIC;
   a->op = atomics[index];
   a->rtn = rtn;
   return NULL;
}

static const char *
classify(uint16_t op, access *a)
{
   memset(a, 0, sizeof(*a));
   if (op <= 0x0b)
      return atomic_access(op, false, a);
   if (op >= 0x20 && op <= 0x2b)
      return atomic_access((uint16_t)(op - 0x20), true, a);
   a->kind = ACCESS_LOAD;
   switch (op) {
   case 0x2d:
      a->kind = ACCESS_ATOMIC;
      a->op = global_atomic_swap;
      a->rtn = true;
      return NULL;
   case 0x10:
   case 0x30:
      a->kind = ACCESS_CMPSWAP;
      a->rtn = op == 0x30;
      return NULL;
   case 0x0d:
   case 0x1e:
   case 0x1f:
   case 0x4d:
   case 0xa0:
   case 0xa1:
   case 0xde:
   case 0xdf: {
      static const struct {
         uint16_t ds;
         uint16_t global;
      } stores[] = {
         {0x0d, global_store_dword}, {0x1e, 0x18}, {0x1f, 0x1a}, {0x4d, 0x1d},
         {0xa0, 0x19},               {0xa1, 0x1b}, {0xde, 0x1f}, {0xdf, 0x1e},
      };
      a->kind = ACCESS_STORE;
      for (unsigned i = 0; i < sizeof(stores) / sizeof(stores[0]); i++) {
         if (stores[i].ds == op)
            a->op = stores[i].global;
      }
      return NULL;
   }
   case 0x36:
      a->op = global_load_dword;
      return NULL;
   case 0x39:
      a->op = 0x09;
      return NULL;
   case 0x3a:
      a->op = 0x08;
      return NULL;
   case 0x3b:
      a->op = 0x0b;
      return NULL;
   case 0x3c:
      a->op = 0x0a;
      return NULL;
   case 0x76:
      a->op = 0x0d;
      return NULL;
   case 0xa2:
   case 0xa3:
   case 0xa4:
   case 0xa5:
   case 0xa6:
   case 0xa7:
      a->op = (uint16_t)(0x20 + (op - 0xa2));
      return NULL;
   case 0xfe:
      a->op = 0x0f;
      return NULL;
   case 0xff:
      a->op = 0x0e;
      return NULL;
   case 0x0e:
   case 0x0f:
      a->kind = ACCESS_WRITE2;
      a->stride = op == 0x0e ? 4 : 256;
      return NULL;
   case 0x37:
   case 0x38:
      a->kind = ACCESS_READ2;
      a->stride = op == 0x37 ? 4 : 256;
      return NULL;
   case AGC_ISA_DS_CONSUME:
   case AGC_ISA_DS_APPEND:
      a->kind = ACCESS_APPEND;
      a->subtract = op == AGC_ISA_DS_CONSUME;
      return NULL;
   case AGC_ISA_DS_ORDERED_COUNT:
      a->kind = ACCESS_ORDERED;
      return NULL;
   case 0x14:
      a->kind = ACCESS_NOP;
      return NULL;
   default:
      break;
   }
   if (op >= 0x18 && op <= 0x1d)
      return gws_not_emulated;
   if ((op >= 0x11 && op <= 0x13) || op == 0x15 || (op >= 0x31 && op <= 0x33) || op == 0x55)
      return "float GDS atomics are not emulated";
   if ((op >= 0x40 && op <= 0x4c) || (op >= 0x50 && op <= 0x53) || (op >= 0x60 && op <= 0x73))
      return "64-bit GDS atomics are not emulated";
   return "this GDS operation is not emulated";
}

static void
address(gds_emit &e, const agc_isa_ds *d, uint32_t offset, unsigned v)
{
   if (d->uses_addr && offset == 0)
      e.bld.vop1(aco_opcode::v_mov_b32, vdef(v), vop(d->addr));
   else if (d->uses_addr)
      e.bld.vop2(aco_opcode::v_add_u32, vdef(v), imm(offset), vop(d->addr));
   else
      e.bld.vop1(aco_opcode::v_mov_b32, vdef(v), imm(offset));
   e.bld.vop2(aco_opcode::v_and_b32, vdef(v), imm(AGC_ISA_GDS_BYTES - 1), vop(v));
}

static unsigned
buffer_sgprs(gds_emit &e, uint8_t count)
{
   unsigned s = transient_sgprs(e.ctx, e.origin, count);
   e.bld.sop1(aco_opcode::s_mov_b32, sdef(s), imm((uint32_t)e.buffer));
   e.bld.sop1(aco_opcode::s_mov_b32, sdef(s + 1), imm((uint32_t)(e.buffer >> 32)));
   return s;
}

static void
global_at(gds_emit &e, unsigned buffer, uint16_t opcode, unsigned vdst, unsigned vaddr, unsigned vdata, bool rtn)
{
   const agc_isa_mem_op *info = agc_isa_flat_op(opcode);
   if (!info)
      fail(e.ctx, e.origin, "no global form");
   bool load = info->kind == AGC_ISA_MEM_LOAD;
   global_access(e.ctx, e.origin, opcode, vaddr, buffer, vdata, 0, vdst, load || rtn, load);
   if (load || rtn)
      waitcnt(e.ctx, 0, AGC_ISA_GFX10_EXPCNT_MAX, AGC_ISA_GFX10_LGKMCNT_MAX);
   else
      wait_for_stores(e.ctx);
}

static void
global(gds_emit &e, uint16_t opcode, unsigned vdst, unsigned vaddr, unsigned vdata, bool rtn)
{
   unsigned s = buffer_sgprs(e, 2);
   global_at(e, s, opcode, vdst, vaddr, vdata, rtn);
}

static void
first_lane(gds_emit &e, const agc_isa_ds *d, bool counter, uint32_t offset, bool subtract)
{
   lower_ctx &ctx = e.ctx;
   Builder &bld = e.bld;
   bool wave64 = ctx.wave64;
   unsigned s = buffer_sgprs(e, 10);
   unsigned saved = s + 2;
   unsigned mask = s + 4;
   unsigned count = s + 6;
   unsigned address_sgpr = s + 7;
   unsigned saved_scc = s + 8;
   unsigned result = s + 9;
   unsigned v = scratch_vgprs(ctx, 3);
   unsigned vaddr = v;
   unsigned vdata = v + 1;
   unsigned vresult = v + 2;
   save_scc(ctx, saved_scc);
   uint16_t opcode;
   unsigned data;
   if (counter) {
      bld.sop1(Builder::s_bcnt1_i32, sdef(count), scc_def(), mask_op(ctx, exec));
      bld.sop2(aco_opcode::s_lshr_b32, sdef(address_sgpr), scc_def(), Operand(m0, s1), imm(16));
      bld.sop2(aco_opcode::s_add_u32, sdef(address_sgpr), scc_def(), sop(address_sgpr), imm(offset));
      bld.sop2(aco_opcode::s_and_b32, sdef(address_sgpr), scc_def(), sop(address_sgpr), imm(AGC_ISA_GDS_BYTES - 1));
      bld.vop1(aco_opcode::v_mov_b32, vdef(vaddr), sop(address_sgpr));
      bld.vop1(aco_opcode::v_mov_b32, vdef(vdata), sop(count));
      opcode = subtract ? global_atomic_sub : global_atomic_add;
      data = vdata;
   } else {
      bld.vop1(aco_opcode::v_mov_b32, vdef(vaddr), imm(d->offset0));
      opcode = (d->offset1 & ordered_swap) ? global_atomic_swap : global_atomic_add;
      data = d->addr;
   }
   if (wave64) {
      bld.sop1(aco_opcode::s_mov_b64, sdef(saved, 2), Operand(exec, s2));
      bld.sop2(aco_opcode::s_sub_u32, sdef(mask), scc_def(), imm(0), Operand(exec_lo, s1));
      bld.sop2(aco_opcode::s_subb_u32, sdef(mask + 1), scc_def(), imm(0), Operand(exec_hi, s1), scc_op());
      bld.sop2(aco_opcode::s_and_b64, Definition(exec, s2), scc_def(), Operand(exec, s2), sop(mask, 2));
   } else {
      bld.sop1(aco_opcode::s_mov_b32, sdef(saved), Operand(exec_lo, s1));
      bld.sop2(aco_opcode::s_sub_u32, sdef(mask), scc_def(), imm(0), Operand(exec_lo, s1));
      bld.sop2(aco_opcode::s_and_b32, Definition(exec_lo, s1), scc_def(), Operand(exec_lo, s1), sop(mask));
   }
   global_at(e, s, opcode, vresult, vaddr, data, true);
   bld.sop1(Builder::s_mov, mask_def(ctx, exec), sop(saved, bld.lm.size()));
   bld.vop1(aco_opcode::v_readfirstlane_b32, sdef(result), vop(vresult));
   bld.vop1(aco_opcode::v_mov_b32, vdef(d->vdst), sop(result));
   restore_scc(ctx, saved_scc);
}

void
lower_gds(lower_ctx &ctx, const agc_isa_inst *inst, uint64_t buffer)
{
   if (inst->fields != AGC_ISA_DS)
      fail(ctx, inst, "malformed DS");
   const agc_isa_ds d = inst->ds;
   access a;
   const char *reason = classify(inst->opcode, &a);
   if (reason)
      fail(ctx, inst, reason);
   if (!d.gds && a.kind != ACCESS_NOP)
      fail(ctx, inst, "LDS ordered counts are not emulated");
   uint32_t offset = (uint32_t)d.offset0 | ((uint32_t)d.offset1 << 8);
   gds_emit e = {ctx, ctx.bld, inst, buffer};
   unsigned v;
   switch (a.kind) {
   case ACCESS_NOP:
      return;
   case ACCESS_LOAD:
      v = scratch_vgprs(ctx, 1);
      address(e, &d, offset, v);
      global(e, a.op, d.vdst, v, 0, false);
      return;
   case ACCESS_STORE:
      v = scratch_vgprs(ctx, 1);
      address(e, &d, offset, v);
      global(e, a.op, 0, v, d.data0, false);
      return;
   case ACCESS_ATOMIC:
      v = scratch_vgprs(ctx, 1);
      address(e, &d, offset, v);
      global(e, a.op, d.vdst, v, d.data0, a.rtn);
      return;
   case ACCESS_CMPSWAP:
      v = scratch_vgprs(ctx, 3);
      address(e, &d, offset, v);
      e.bld.vop1(aco_opcode::v_mov_b32, vdef(v + 1), vop(d.data1));
      e.bld.vop1(aco_opcode::v_mov_b32, vdef(v + 2), vop(d.data0));
      global(e, global_atomic_cmpswap, d.vdst, v, v + 1, a.rtn);
      return;
   case ACCESS_READ2: {
      v = scratch_vgprs(ctx, 1);
      const uint8_t parts[2] = {d.offset0, d.offset1};
      for (unsigned k = 0; k < 2; k++) {
         address(e, &d, parts[k] * a.stride, v);
         global(e, global_load_dword, d.vdst + k, v, 0, false);
      }
      return;
   }
   case ACCESS_WRITE2: {
      v = scratch_vgprs(ctx, 1);
      const uint8_t parts[2] = {d.offset0, d.offset1};
      const uint8_t datas[2] = {d.data0, d.data1};
      for (unsigned k = 0; k < 2; k++) {
         address(e, &d, parts[k] * a.stride, v);
         global(e, global_store_dword, 0, v, datas[k], false);
      }
      return;
   }
   case ACCESS_APPEND:
      first_lane(e, &d, true, offset, a.subtract);
      return;
   default:
      if (d.offset1 >> ordered_dwords_shift)
         fail(ctx, inst, "GDS ordered counts of more than one dword are not emulated");
      first_lane(e, &d, false, 0, false);
      return;
   }
}

}
