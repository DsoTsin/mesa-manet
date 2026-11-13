#include <string.h>

#include "agc_isa.h"
#include "test.h"

#define S_ENDPGM 0xbf810000u
#define WORDS(...) ((const uint32_t[]){__VA_ARGS__}), (sizeof((const uint32_t[]){__VA_ARGS__}) / sizeof(uint32_t))

static struct agc_isa_inst
one(const uint32_t *words, size_t count)
{
   struct agc_isa_inst inst;
   struct agc_isa_error error;
   memset(&inst, 0, sizeof(inst));
   bool ok = agc_isa_decode_at(words, count, 0, &inst, &error);
   CHECK(ok);
   if (!ok)
      memset(&inst, 0, sizeof(inst));
   return inst;
}

static struct agc_isa_error
bad(const uint32_t *words, size_t count)
{
   struct agc_isa_inst inst;
   struct agc_isa_error error;
   memset(&error, 0, sizeof(error));
   CHECK(!agc_isa_decode_at(words, count, 0, &inst, &error));
   return error;
}

static bool
is_op(struct agc_isa_operand op, enum agc_isa_operand_kind kind, uint32_t value)
{
   return op.kind == kind && op.value == value;
}

static bool
named(const struct agc_isa_inst *inst, const char *name)
{
   char scratch[AGC_ISA_MNEMONIC_SIZE];
   return !strcmp(agc_isa_inst_mnemonic(inst, scratch), name);
}

static bool
same_range(struct agc_isa_reg_range range, enum agc_isa_reg_file file, unsigned first, unsigned count)
{
   return range.file == file && range.first == first && range.count == count;
}

static void
test_family_prefixes(void)
{
   static const struct {
      uint32_t word;
      enum agc_isa_family family;
   } cases[] = {
      {0xbe800380, AGC_ISA_SOP1},  {0x8002ff00, AGC_ISA_SOP2},  {0xb0000000, AGC_ISA_SOPK},
      {0xbf068400, AGC_ISA_SOPC},  {S_ENDPGM, AGC_ISA_SOPP},    {0x7e0202f2, AGC_ISA_VOP1},
      {0x06000501, AGC_ISA_VOP2},  {0x7d840280, AGC_ISA_VOPC},  {0xd54b0000, AGC_ISA_VOP3},
      {0xcc0e0000, AGC_ISA_VOP3P}, {0xc8080500, AGC_ISA_VINTRP}, {0xf4080100, AGC_ISA_SMEM},
      {0xe0301000, AGC_ISA_MUBUF}, {0xea6b2000, AGC_ISA_MTBUF}, {0xdc308000, AGC_ISA_FLAT},
      {0xd8dc0201, AGC_ISA_DS},    {0xf0800f08, AGC_ISA_MIMG},  {0xf800180f, AGC_ISA_EXP},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
      enum agc_isa_family family;
      CHECK(agc_isa_family_of(cases[i].word, &family));
      CHECK_EQ(family, cases[i].family);
   }
   enum agc_isa_family family;
   CHECK(!agc_isa_family_of(0xffffffff, &family));
   CHECK(!agc_isa_family_of(0xc0000000, &family));
}

static void
test_scalar_formats(void)
{
   struct agc_isa_inst add = one(WORDS(0x8002ff00, 0x12345678));
   CHECK_EQ(add.family, AGC_ISA_SOP2);
   CHECK_EQ(add.fields, AGC_ISA_SOP2);
   CHECK_EQ(add.opcode, AGC_ISA_SOP2_ADD_U32);
   CHECK_EQ(add.word_count, 2);
   CHECK(is_op(add.sop2.sdst, AGC_ISA_OPERAND_SGPR, 2));
   CHECK(is_op(add.sop2.src0, AGC_ISA_OPERAND_SGPR, 0));
   CHECK(is_op(add.sop2.src1, AGC_ISA_OPERAND_LITERAL, 0x12345678));
   CHECK(named(&add, "s_add_u32"));
   CHECK(!add.has_branch_target);

   struct agc_isa_inst mov = one(WORDS(0xbefc03c1));
   CHECK_EQ(mov.opcode, AGC_ISA_SOP1_MOV_B32);
   CHECK(is_op(mov.sop1.sdst, AGC_ISA_OPERAND_M0, 0));
   CHECK(mov.sop1.src0.kind == AGC_ISA_OPERAND_INLINE_INT && mov.sop1.src0.inline_int == -1);

   struct agc_isa_inst saveexec = one(WORDS(0xbe803c6a));
   CHECK(agc_isa_inst_has_flag(&saveexec, AGC_ISA_FLAG_WRITES_EXEC));
   CHECK(is_op(saveexec.sop1.sdst, AGC_ISA_OPERAND_SGPR, 0));
   CHECK(is_op(saveexec.sop1.src0, AGC_ISA_OPERAND_VCC_LO, 0));

   struct agc_isa_inst literal = one(WORDS(0xbe8003ff, 0xdeadbeef));
   CHECK_EQ(literal.word_count, 2);
   CHECK(is_op(literal.sop1.src0, AGC_ISA_OPERAND_LITERAL, 0xdeadbeef));

   struct agc_isa_inst jump = one(WORDS(0xbe802006));
   CHECK_EQ(jump.opcode, AGC_ISA_SOP1_SETPC_B64);
   CHECK(agc_isa_inst_has_flag(&jump, AGC_ISA_FLAG_JUMP));
   CHECK(!agc_isa_inst_falls_through(&jump));

   struct agc_isa_inst call = one(WORDS(0xbe862104));
   CHECK_EQ(call.opcode, AGC_ISA_SOP1_SWAPPC_B64);
   CHECK(agc_isa_inst_has_flag(&call, AGC_ISA_FLAG_CALL));
   CHECK(agc_isa_inst_falls_through(&call));
   struct agc_isa_inst tail = one(WORDS(0xbefd2104));
   CHECK(is_op(tail.sop1.sdst, AGC_ISA_OPERAND_NULL, 0));
   CHECK(agc_isa_inst_has_flag(&tail, AGC_ISA_FLAG_JUMP | AGC_ISA_FLAG_NO_DST));
   CHECK(!agc_isa_inst_has_flag(&tail, AGC_ISA_FLAG_CALL | AGC_ISA_FLAG_D2 | AGC_ISA_FLAG_PC_RELATIVE));

   struct agc_isa_inst movk = one(WORDS(0xb0031234));
   CHECK_EQ(movk.family, AGC_ISA_SOPK);
   CHECK_EQ(movk.opcode, AGC_ISA_SOPK_MOVK_I32);
   CHECK(is_op(movk.sopk.sdst, AGC_ISA_OPERAND_SGPR, 3));
   CHECK_EQ(movk.sopk.simm16, 0x1234);
   CHECK(!movk.sopk.has_imm32);

   struct agc_isa_inst setreg = one(WORDS(0xba800001, 0x12345678));
   CHECK_EQ(setreg.opcode, AGC_ISA_SOPK_SETREG_IMM32_B32);
   CHECK_EQ(setreg.word_count, 2);
   CHECK(setreg.sopk.has_imm32);
   CHECK_EQ(setreg.sopk.imm32, 0x12345678);

   const uint32_t call_words[] = {0, 0xbb000002};
   struct agc_isa_inst callk;
   struct agc_isa_error error;
   CHECK(agc_isa_decode_at(call_words, 2, 1, &callk, &error));
   CHECK_EQ(callk.opcode, AGC_ISA_SOPK_CALL_B64);
   CHECK(callk.has_branch_target);
   CHECK_EQ(callk.branch_target, 16);
   CHECK_EQ(callk.pc, 4);

   struct agc_isa_inst cmp = one(WORDS(0xbf068400));
   CHECK_EQ(cmp.family, AGC_ISA_SOPC);
   CHECK_EQ(cmp.opcode, AGC_ISA_SOPC_CMP_EQ_U32);
   CHECK(is_op(cmp.sopc.src0, AGC_ISA_OPERAND_SGPR, 0));
   CHECK(cmp.sopc.src1.kind == AGC_ISA_OPERAND_INLINE_INT && cmp.sopc.src1.inline_int == 4);
   struct agc_isa_inst cmp_literal = one(WORDS(0xbf06ff01, 0x00010000));
   CHECK_EQ(cmp_literal.word_count, 2);
   CHECK(is_op(cmp_literal.sopc.src1, AGC_ISA_OPERAND_LITERAL, 0x10000));

   struct agc_isa_inst end = one(WORDS(S_ENDPGM));
   CHECK_EQ(end.family, AGC_ISA_SOPP);
   CHECK_EQ(end.opcode, AGC_ISA_SOPP_ENDPGM);
   CHECK(agc_isa_inst_has_flag(&end, AGC_ISA_FLAG_END));
   CHECK(agc_isa_inst_ends_block(&end));

   const uint32_t forward[] = {0, 0, 0xbf840002};
   struct agc_isa_inst branch;
   CHECK(agc_isa_decode_at(forward, 3, 2, &branch, &error));
   CHECK(branch.has_branch_target);
   CHECK_EQ(branch.branch_target, 20);
   const uint32_t backward[] = {0, 0, 0, 0xbf84fffd};
   CHECK(agc_isa_decode_at(backward, 4, 3, &branch, &error));
   CHECK_EQ(branch.branch_target, 4);
   CHECK(named(&branch, "s_cbranch_scc0"));
   CHECK(agc_isa_inst_is_branch(&branch));
   CHECK(agc_isa_inst_falls_through(&branch));
}

static void
test_vector_formats(void)
{
   struct agc_isa_inst mov = one(WORDS(0x7e0202f2));
   CHECK_EQ(mov.family, AGC_ISA_VOP1);
   CHECK_EQ(mov.opcode, AGC_ISA_VOP1_MOV_B32);
   CHECK(is_op(mov.vop1.vdst, AGC_ISA_OPERAND_VGPR, 1));
   CHECK(is_op(mov.vop1.src0, AGC_ISA_OPERAND_INLINE_FLOAT, 242));
   CHECK_EQ(mov.vop1.ext.kind, AGC_ISA_VOP_EXT_NONE);
   uint32_t bits;
   CHECK(agc_isa_inline_float_bits(242, &bits));
   CHECK_EQ(bits, 0x3f800000);

   struct agc_isa_inst nop = one(WORDS(0x7e000000));
   CHECK_EQ(nop.opcode, AGC_ISA_VOP1_NOP);
   CHECK(agc_isa_inst_has_flag(&nop, AGC_ISA_FLAG_NO_OPERANDS));

   struct agc_isa_inst first = one(WORDS(0x7e0a0503));
   CHECK_EQ(first.opcode, AGC_ISA_VOP1_READFIRSTLANE_B32);
   CHECK(is_op(first.vop1.vdst, AGC_ISA_OPERAND_SGPR, 5));
   CHECK(is_op(first.vop1.src0, AGC_ISA_OPERAND_VGPR, 3));

   struct agc_isa_inst sdwa = one(WORDS(0x060004f9, 0x02041501));
   CHECK_EQ(sdwa.family, AGC_ISA_VOP2);
   CHECK_EQ(sdwa.opcode, AGC_ISA_VOP2_ADD_F32);
   CHECK_EQ(sdwa.word_count, 2);
   CHECK(is_op(sdwa.vop2.vdst, AGC_ISA_OPERAND_VGPR, 0));
   CHECK(is_op(sdwa.vop2.src0, AGC_ISA_OPERAND_VGPR, 1));
   CHECK(is_op(sdwa.vop2.vsrc1, AGC_ISA_OPERAND_VGPR, 2));
   CHECK_EQ(sdwa.vop2.ext.kind, AGC_ISA_VOP_EXT_SDWA);
   CHECK_EQ(sdwa.vop2.ext.sdwa.dst_sel, AGC_ISA_SDWA_SEL_WORD_1);
   CHECK_EQ(sdwa.vop2.ext.sdwa.dst_unused, AGC_ISA_SDWA_UNUSED_PRESERVE);
   CHECK_EQ(sdwa.vop2.ext.sdwa.src0_sel, 4);
   CHECK_EQ(sdwa.vop2.ext.sdwa.src1_sel, 2);

   struct agc_isa_inst dpp16 = one(WORDS(0x7e0002fa, 0xff00b101));
   CHECK_EQ(dpp16.vop1.ext.kind, AGC_ISA_VOP_EXT_DPP16);
   CHECK(is_op(dpp16.vop1.src0, AGC_ISA_OPERAND_VGPR, 1));
   CHECK_EQ(dpp16.vop1.ext.dpp16.ctrl, 0xb1);
   CHECK_EQ(dpp16.vop1.ext.dpp16.row_mask, 0xf);
   CHECK_EQ(dpp16.vop1.ext.dpp16.bank_mask, 0xf);

   struct agc_isa_inst dpp8 = one(WORDS(0x7e0002ea, 0x00123401));
   CHECK_EQ(dpp8.vop1.ext.kind, AGC_ISA_VOP_EXT_DPP8);
   CHECK(dpp8.vop1.ext.dpp8.fetch_inactive);
   CHECK_EQ(dpp8.vop1.ext.dpp8.selects, 0x1234);
   CHECK(is_op(dpp8.vop1.src0, AGC_ISA_OPERAND_VGPR, 1));
   struct agc_isa_inst dpp8_plain = one(WORDS(0x7e0002e9, 0x00123401));
   CHECK(!dpp8_plain.vop1.ext.dpp8.fetch_inactive);

   struct agc_isa_inst madak = one(WORDS(0x42020501, 0x3f800000));
   CHECK_EQ(madak.opcode, AGC_ISA_VOP2_MADAK_F32);
   CHECK_EQ(madak.word_count, 2);
   CHECK(madak.vop2.has_constant);
   CHECK_EQ(madak.vop2.constant, 0x3f800000);
   CHECK(is_op(madak.vop2.src0, AGC_ISA_OPERAND_VGPR, 1));
   CHECK(is_op(madak.vop2.vsrc1, AGC_ISA_OPERAND_VGPR, 2));
   struct agc_isa_error two_literals = bad(WORDS(0x420204ff, 0x3f800000, 0x40000000));
   CHECK_EQ(two_literals.kind, AGC_ISA_ERROR_INVALID_OPERAND);

   struct agc_isa_inst cmp = one(WORDS(0x7d840280));
   CHECK(named(&cmp, "v_cmp_eq_u32"));
   CHECK_EQ(cmp.family, AGC_ISA_VOPC);
   CHECK(is_op(cmp.vopc.sdst, AGC_ISA_OPERAND_VCC_LO, 0));
   CHECK(cmp.vopc.src0.kind == AGC_ISA_OPERAND_INLINE_INT && cmp.vopc.src0.inline_int == 0);
   CHECK(is_op(cmp.vopc.vsrc1, AGC_ISA_OPERAND_VGPR, 1));
   struct agc_isa_inst cmpx = one(WORDS(0x7c280280));
   CHECK(named(&cmpx, "v_cmpx_gt_f32"));
   CHECK(agc_isa_inst_has_flag(&cmpx, AGC_ISA_FLAG_CMPX));
   CHECK(agc_isa_inst_has_flag(&cmpx, AGC_ISA_FLAG_WRITES_EXEC));
   CHECK(is_op(cmpx.vopc.sdst, AGC_ISA_OPERAND_EXEC_LO, 0));
   struct agc_isa_inst cmp_sdwa = one(WORDS(0x7d8404f9, 0x06868501));
   CHECK_EQ(cmp_sdwa.vopc.ext.kind, AGC_ISA_VOP_EXT_SDWA);
   CHECK(is_op(cmp_sdwa.vopc.sdst, AGC_ISA_OPERAND_SGPR, 5));
   CHECK(is_op(cmp_sdwa.vopc.src0, AGC_ISA_OPERAND_SGPR, 1));
   CHECK(is_op(cmp_sdwa.vopc.vsrc1, AGC_ISA_OPERAND_VGPR, 2));
   CHECK_EQ(cmp_sdwa.vopc.ext.sdwa.src0_sel, 6);
   CHECK_EQ(cmp_sdwa.vopc.ext.sdwa.src1_sel, 6);
   CHECK_EQ(cmp_sdwa.vopc.ext.sdwa.dst_sel, 0);

   struct agc_isa_inst fma = one(WORDS(0xd54b8200, 0x240e0501));
   CHECK_EQ(fma.family, AGC_ISA_VOP3);
   CHECK_EQ(fma.opcode, AGC_ISA_VOP3_FMA_F32);
   CHECK(is_op(fma.vop3.src[0], AGC_ISA_OPERAND_VGPR, 1));
   CHECK(is_op(fma.vop3.src[1], AGC_ISA_OPERAND_VGPR, 2));
   CHECK(is_op(fma.vop3.src[2], AGC_ISA_OPERAND_VGPR, 3));
   CHECK_EQ(fma.vop3.abs, 2);
   CHECK_EQ(fma.vop3.neg, 1);
   CHECK(fma.vop3.clamp);
   CHECK(!agc_isa_operand_is_some(fma.vop3.sdst));

   struct agc_isa_inst add_co = one(WORDS(0xd70f6a00, 0x00020501));
   CHECK_EQ(add_co.opcode, AGC_ISA_VOP3_ADD_CO_U32);
   CHECK(is_op(add_co.vop3.sdst, AGC_ISA_OPERAND_VCC_LO, 0));
   CHECK(is_op(add_co.vop3.src[0], AGC_ISA_OPERAND_VGPR, 1));
   CHECK(is_op(add_co.vop3.src[1], AGC_ISA_OPERAND_VGPR, 2));

   struct agc_isa_inst promoted = one(WORDS(0xd5030000, 0x000202ff, 0x40490fdb));
   CHECK_EQ(promoted.word_count, 3);
   CHECK(named(&promoted, "v_add_f32"));
   CHECK(is_op(promoted.vop3.src[0], AGC_ISA_OPERAND_LITERAL, 0x40490fdb));

   struct agc_isa_inst promoted_cmp = one(WORDS(0xd4c20005, 0x00020501));
   CHECK(named(&promoted_cmp, "v_cmp_eq_u32"));
   CHECK(is_op(promoted_cmp.vop3.vdst, AGC_ISA_OPERAND_SGPR, 5));

   struct agc_isa_inst dot = one(WORDS(0xcc160004, 0x040e0501));
   CHECK_EQ(dot.family, AGC_ISA_VOP3P);
   CHECK_EQ(dot.opcode, AGC_ISA_VOP3P_DOT4_I32_I8);
   CHECK(is_op(dot.vop3p.vdst, AGC_ISA_OPERAND_VGPR, 4));
   CHECK(is_op(dot.vop3p.src[2], AGC_ISA_OPERAND_VGPR, 3));
   struct agc_isa_inst packed = one(WORDS(0xcc0fc004, 0x180e0501));
   CHECK_EQ(packed.opcode, AGC_ISA_VOP3P_PK_ADD_F16);
   CHECK_EQ(packed.vop3p.opsel_hi, 7);
   CHECK(packed.vop3p.clamp);

   struct agc_isa_inst p1 = one(WORDS(0xc8080500));
   CHECK_EQ(p1.family, AGC_ISA_VINTRP);
   CHECK_EQ(p1.opcode, AGC_ISA_VINTRP_P1);
   CHECK_EQ(p1.vintrp.vdst, 2);
   CHECK_EQ(p1.vintrp.vsrc, 0);
   CHECK_EQ(p1.vintrp.attr, 1);
   CHECK_EQ(p1.vintrp.chan, 1);
   struct agc_isa_inst p2 = one(WORDS(0xc8090501));
   CHECK(agc_isa_inst_has_flag(&p2, AGC_ISA_FLAG_READS_DST));
   CHECK_EQ(bad(WORDS(0xc80b0501)).kind, AGC_ISA_ERROR_UNKNOWN_OPCODE);
}

static void
test_memory_formats(void)
{
   struct agc_isa_inst load = one(WORDS(0xf4080100, 0xfa000010));
   CHECK_EQ(load.family, AGC_ISA_SMEM);
   CHECK_EQ(load.opcode, AGC_ISA_SMEM_LOAD_DWORDX4);
   CHECK(is_op(load.smem.sdata, AGC_ISA_OPERAND_SGPR, 4));
   CHECK_EQ(load.smem.sbase, 0);
   CHECK_EQ(load.smem.offset, 0x10);
   CHECK_EQ(load.smem.dwords, 4);
   CHECK(is_op(load.smem.soffset, AGC_ISA_OPERAND_NULL, 0));
   CHECK(!load.smem.buffer);
   struct agc_isa_inst buffer_load = one(WORDS(0xf4200202, 0xfa000000));
   CHECK_EQ(buffer_load.smem.sbase, 4);
   CHECK(buffer_load.smem.buffer);
   struct agc_isa_inst negative = one(WORDS(0xf4000080, 0xfa1ffffc));
   CHECK_EQ(negative.smem.offset, -4);

   struct agc_isa_inst mubuf = one(WORDS(0xe0301000, 0x80020100));
   CHECK_EQ(mubuf.family, AGC_ISA_MUBUF);
   CHECK_EQ(mubuf.fields, AGC_ISA_MUBUF);
   CHECK_EQ(mubuf.opcode, AGC_ISA_VMEM_LOAD_DWORD);
   CHECK(named(&mubuf, "buffer_load_dword"));
   CHECK_EQ(mubuf.buffer.vdata, 1);
   CHECK_EQ(mubuf.buffer.vaddr, 0);
   CHECK_EQ(mubuf.buffer.srsrc, 8);
   CHECK(mubuf.buffer.soffset.kind == AGC_ISA_OPERAND_INLINE_INT && mubuf.buffer.soffset.inline_int == 0);
   CHECK(mubuf.buffer.offen && !mubuf.buffer.idxen);
   CHECK_EQ(mubuf.buffer.kind, AGC_ISA_MEM_LOAD);
   CHECK_EQ(mubuf.buffer.data_dwords, 1);

   struct agc_isa_inst mtbuf = one(WORDS(0xea6b2000, 0x80020400));
   CHECK(named(&mtbuf, "tbuffer_load_format_xyzw"));
   CHECK_EQ(mtbuf.fields, AGC_ISA_MTBUF);
   CHECK_EQ(mtbuf.buffer.format, 77);
   CHECK_EQ(mtbuf.buffer.data_dwords, 4);
   CHECK_EQ(mtbuf.buffer.vdata, 4);
   CHECK(mtbuf.buffer.idxen);

   struct agc_isa_inst global = one(WORDS(0xdc308000, 0x017d0002));
   CHECK_EQ(global.family, AGC_ISA_FLAT);
   CHECK(named(&global, "flat_load_dword"));
   CHECK_EQ(global.flat.segment, AGC_ISA_SEGMENT_GLOBAL);
   CHECK_EQ(global.flat.vdst, 1);
   CHECK_EQ(global.flat.vaddr, 2);
   CHECK(is_op(global.flat.saddr, AGC_ISA_OPERAND_NULL, 0));
   CHECK(global.flat.uses_vaddr);
   struct agc_isa_inst store = one(WORDS(0xdc708ff8, 0x00040302));
   CHECK(is_op(store.flat.saddr, AGC_ISA_OPERAND_SGPR, 4));
   CHECK_EQ(store.flat.offset, -8);
   CHECK_EQ(store.flat.vdata, 3);
   CHECK_EQ(store.flat.kind, AGC_ISA_MEM_STORE);
   struct agc_isa_inst flat = one(WORDS(0xdc300ff8, 0x017f0002));
   CHECK_EQ(flat.flat.segment, AGC_ISA_SEGMENT_FLAT);
   CHECK(!agc_isa_operand_is_some(flat.flat.saddr));
   CHECK_EQ(flat.flat.offset, 0xff8);
   struct agc_isa_inst scratch = one(WORDS(0xdc304004, 0x01050002));
   CHECK_EQ(scratch.flat.segment, AGC_ISA_SEGMENT_SCRATCH);
   CHECK(!scratch.flat.uses_vaddr);

   struct agc_isa_inst read2 = one(WORDS(0xd8dc0201, 0x00000002));
   CHECK_EQ(read2.family, AGC_ISA_DS);
   CHECK_EQ(read2.opcode, AGC_ISA_DS_READ2_B32);
   CHECK_EQ(read2.ds.vdst, 0);
   CHECK_EQ(read2.ds.addr, 2);
   CHECK_EQ(read2.ds.offset0, 1);
   CHECK_EQ(read2.ds.offset1, 2);
   CHECK_EQ(read2.ds.dst_dwords, 2);
   CHECK(read2.ds.uses_addr && !read2.ds.gds);
   struct agc_isa_inst gds = one(WORDS(0xd8360000, 0x00000100));
   CHECK(gds.ds.gds);
   CHECK(agc_isa_inst_has_flag(&gds, AGC_ISA_FLAG_READS_M0));

   struct agc_isa_inst exp = one(WORDS(0xf800180f, 0x03020100));
   CHECK_EQ(exp.family, AGC_ISA_EXP);
   CHECK(named(&exp, "exp"));
   CHECK_EQ(exp.exp.target, 0);
   CHECK_EQ(exp.exp.en, 0xf);
   CHECK(!exp.exp.compr && exp.exp.done && exp.exp.vm && !exp.exp.row_en);
   CHECK_EQ(exp.exp.vsrc[0], 0);
   CHECK_EQ(exp.exp.vsrc[3], 3);
   CHECK_EQ(bad(WORDS(0xf800380f, 0x03020100)).kind, AGC_ISA_ERROR_INVALID_ENCODING);
}

static void
check_addresses(const struct agc_isa_image *m, const uint8_t *expected, unsigned count)
{
   CHECK_EQ(m->addr_dwords, count);
   for (unsigned i = 0; i < count && i < m->addr_dwords; i++)
      CHECK_EQ(agc_isa_image_address_register(m, i), expected[i]);
}

static void
test_image_formats(void)
{
   struct agc_isa_inst sample = one(WORDS(0xf0800f08, 0x00820004));
   CHECK_EQ(sample.family, AGC_ISA_MIMG);
   CHECK_EQ(sample.opcode, AGC_ISA_MIMG_SAMPLE);
   CHECK_EQ(sample.mimg.dim, 1);
   CHECK_EQ(sample.mimg.dmask, 0xf);
   CHECK_EQ(sample.mimg.vdata, 0);
   CHECK_EQ(sample.mimg.srsrc, 8);
   CHECK_EQ(sample.mimg.ssamp, 16);
   CHECK_EQ(sample.mimg.addr[0], 4);
   CHECK_EQ(sample.mimg.addr_dwords, 2);
   CHECK_EQ(sample.mimg.data_dwords, 4);
   CHECK(sample.mimg.uses_sampler && !sample.mimg.nsa);

   struct agc_isa_inst nsa = one(WORDS(0xf0800f0a, 0x00820004, 0x00000007));
   CHECK_EQ(nsa.word_count, 3);
   CHECK(nsa.mimg.nsa);
   CHECK_EQ(agc_isa_image_address_register(&nsa.mimg, 0), 4);
   CHECK_EQ(agc_isa_image_address_register(&nsa.mimg, 1), 7);

   static const struct {
      uint32_t word0;
      uint16_t opcode;
      const char *name;
      uint8_t addresses;
   } adjusted[] = {
      {0xf1200109, 0xc8, "image_gather4_c_a", 3},
      {0xf1600109, 0xd8, "image_gather4_c_o_a", 4},
      {0xf0e00109, 0xb8, "image_sample_c_o_a", 4},
   };
   for (unsigned i = 0; i < 3; i++) {
      struct agc_isa_inst inst = one(WORDS(adjusted[i].word0, 0x00801e33));
      CHECK_EQ(inst.opcode, adjusted[i].opcode);
      CHECK(named(&inst, adjusted[i].name));
      CHECK_EQ(inst.mimg.addr_dwords, adjusted[i].addresses);
      CHECK_EQ(inst.mimg.addr[0], 51);
      CHECK_EQ(inst.mimg.vdata, 30);
      CHECK_EQ(inst.mimg.dim, 1);
      CHECK_EQ(inst.mimg.srsrc, 0);
      CHECK_EQ(inst.mimg.ssamp, 16);
      CHECK(inst.mimg.uses_sampler);
      struct agc_isa_inst plain = one(WORDS(adjusted[i].word0 & ~1u, 0x00801e33));
      CHECK_EQ(plain.opcode, adjusted[i].opcode - 0x80);
      CHECK_EQ(plain.mimg.addr_dwords, inst.mimg.addr_dwords);
      CHECK_EQ(plain.mimg.data_dwords, inst.mimg.data_dwords);
   }

   static const uint8_t bvh[] = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
   static const uint8_t bvh_nsa[] = {4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20};
   static const uint8_t bvh64_nsa[] = {4, 5, 9, 10, 11, 12, 13, 14, 15};
   struct agc_isa_inst ray = one(WORDS(0xf1989f01, 0x00010004));
   CHECK_EQ(ray.opcode, AGC_ISA_MIMG_BVH_INTERSECT_RAY);
   check_addresses(&ray.mimg, bvh, 11);
   CHECK(!ray.mimg.a16 && ray.mimg.r128 && ray.mimg.unorm);
   CHECK_EQ(ray.mimg.data_dwords, 4);
   CHECK_EQ(ray.mimg.srsrc, 4);
   struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
   CHECK_EQ(agc_isa_inst_defs(&ray, false, ranges), 1);
   CHECK(same_range(ranges[0], AGC_ISA_REG_VECTOR, 0, 4));
   struct agc_isa_inst ray_a16 = one(WORDS(0xf1989f01, 0x40010004));
   check_addresses(&ray_a16.mimg, bvh, 8);
   struct agc_isa_inst ray64 = one(WORDS(0xf19c9f01, 0x00010004));
   CHECK_EQ(ray64.opcode, AGC_ISA_MIMG_BVH64_INTERSECT_RAY);
   check_addresses(&ray64.mimg, bvh, 12);
   struct agc_isa_inst ray64_a16 = one(WORDS(0xf19c9f01, 0x40010004));
   check_addresses(&ray64_a16.mimg, bvh, 9);
   struct agc_isa_inst ray_nsa = one(WORDS(0xf1989f07, 0x00010004, 0x0c0b0a09, 0x100f0e0d, 0x00001411));
   CHECK_EQ(ray_nsa.word_count, 5);
   CHECK_EQ(ray_nsa.mimg.nsa_words, 3);
   check_addresses(&ray_nsa.mimg, bvh_nsa, 11);
   struct agc_isa_inst ray64_nsa = one(WORDS(0xf19c9f05, 0x40010004, 0x0b0a0905, 0x0f0e0d0c));
   CHECK_EQ(ray64_nsa.word_count, 4);
   check_addresses(&ray64_nsa.mimg, bvh64_nsa, 9);
   CHECK_EQ(one(WORDS(0xf1989101, 0x00010004)).mimg.data_dwords, 4);

   struct agc_isa_error unknown = bad(WORDS(0xf1909f05, 0x000b0010, 0x0d151411, 0x0000200a));
   CHECK_EQ(unknown.kind, AGC_ISA_ERROR_UNKNOWN_OPCODE);
   CHECK_EQ(unknown.family, AGC_ISA_MIMG);
   CHECK_EQ(unknown.opcode, 0xe4);

   static const char *const g16[] = {
      "image_sample_cd_g16",   "image_sample_cd_cl_g16",   "image_sample_c_cd_g16",   "image_sample_c_cd_cl_g16",
      "image_sample_cd_o_g16", "image_sample_cd_cl_o_g16", "image_sample_c_cd_o_g16", "image_sample_c_cd_cl_o_g16",
   };
   for (unsigned i = 0; i < 8; i++) {
      uint32_t op = 0xe8 + i;
      struct agc_isa_inst inst = one(WORDS(0xf0000f09 | ((op & 0x7f) << 18), 0x00820402));
      CHECK_EQ(inst.opcode, op);
      CHECK(named(&inst, g16[i]));
      CHECK(inst.mimg.uses_sampler);
   }
   CHECK_EQ(one(WORDS(0xf1a00f09, 0x00820402)).mimg.addr_dwords, 4);
   CHECK_EQ(bad(WORDS(0xf0880f0a, 0x00820004, 0x00000007)).kind, AGC_ISA_ERROR_INVALID_OPERAND);

   static const uint8_t bvh8_addresses[] = {0x10, 0x11, 0x14, 0x15, 0x18, 0x19, 0x1a, 0x1c, 0x1d, 0x1e, 0x30};
   struct agc_isa_inst bvh8 = one(WORDS(0xf1949f05, 0x00012010, 0x18151411, 0x0000301c));
   CHECK_EQ(bvh8.opcode, AGC_ISA_MIMG_BVH8_INTERSECT_RAY);
   CHECK_EQ(bvh8.word_count, 4);
   CHECK(bvh8.mimg.bvh8);
   CHECK_EQ(bvh8.mimg.data_dwords, 10);
   check_addresses(&bvh8.mimg, bvh8_addresses, 11);
   CHECK_EQ(agc_isa_inst_defs(&bvh8, false, ranges), 3);
   CHECK(same_range(ranges[0], AGC_ISA_REG_VECTOR, 0x18, 3));
   CHECK(same_range(ranges[1], AGC_ISA_REG_VECTOR, 0x1c, 3));
   CHECK(same_range(ranges[2], AGC_ISA_REG_VECTOR, 0x20, 10));
   CHECK_EQ(bad(WORDS(0xf1949f05, 0x00012010, 0x18151412, 0x0000301c)).kind, AGC_ISA_ERROR_INVALID_OPERAND);

   struct agc_isa_inst stack = one(WORDS(0xdb840010, 0x03080201));
   CHECK_EQ(stack.opcode, AGC_ISA_DS_BVH_STACK_PUSH8_POP1_RTN_B32);
   CHECK(stack.ds.writes_addr);
   CHECK_EQ(stack.ds.data1_dwords, 8);
   CHECK_EQ(bad(WORDS(0xdb840007, 0x03080201)).kind, AGC_ISA_ERROR_INVALID_OPERAND);
}

static bool
unknown(const uint32_t *words, size_t count)
{
   struct agc_isa_inst inst;
   struct agc_isa_error error;
   return !agc_isa_decode_at(words, count, 0, &inst, &error) && error.kind == AGC_ISA_ERROR_UNKNOWN_OPCODE;
}

static void
test_rejected_encodings(void)
{
   static const uint32_t global_atomics[] = {0x30, 0x31, 0x40, 0x50, 0x60};
   static const uint32_t global_only[] = {0x16, 0x17, 0x34};
   static const uint32_t vop3_only[] = {0x102, 0x10d, 0x120, 0x121, 0x12c, 0x12d, 0x137, 0x138, 0x13c, 0x182, 0x1e5, 0x1e8};
   for (unsigned i = 0; i < 5; i++) {
      uint32_t word = 0xdc000000 | (global_atomics[i] << 18);
      CHECK(unknown(WORDS(word | (1u << 14), 0x007f0102)));
      CHECK(!unknown(WORDS(word | (2u << 14), 0x007f0102)));
   }
   for (unsigned i = 0; i < 3; i++) {
      uint32_t word = 0xdc000000 | (global_only[i] << 18);
      CHECK(unknown(WORDS(word, 0x007f0102)));
      CHECK(unknown(WORDS(word | (1u << 14), 0x007f0102)));
      CHECK(!unknown(WORDS(word | (2u << 14), 0x007f0102)));
   }
   for (unsigned i = 0; i < sizeof(vop3_only) / sizeof(vop3_only[0]); i++)
      CHECK(unknown(WORDS(0xd4000000 | (vop3_only[i] << 16), 0x00020501)));
   CHECK(!unknown(WORDS(0xd5060000, 0x00020501)));
   CHECK_EQ(bad(WORDS(0xdc30c000, 0x017d0002)).kind, AGC_ISA_ERROR_INVALID_ENCODING);
   CHECK_EQ(bad(WORDS(0x800200d1)).kind, AGC_ISA_ERROR_INVALID_OPERAND);
   CHECK_EQ(bad(WORDS(0x7f900503)).kind, AGC_ISA_ERROR_INVALID_OPERAND);
}

static bool
decodes(const uint32_t *words, size_t count, const struct agc_isa_decode_scope *scope, struct agc_isa_ir *ir,
        struct agc_isa_error *error)
{
   return scope ? agc_isa_decode_scoped(words, count, scope, ir, error) : agc_isa_decode(words, count, ir, error);
}

static void
test_programs(void)
{
   struct agc_isa_ir ir;
   struct agc_isa_error error;
   agc_isa_ir_init(&ir);

   CHECK(decodes(WORDS(0xbf840001, S_ENDPGM, 0xbe800380, S_ENDPGM, 0xdeadbeef), NULL, &ir, &error));
   CHECK_EQ(ir.code_words, 4);
   CHECK_EQ(ir.instruction_count, 4);
   size_t index;
   CHECK(agc_isa_ir_instruction_at(&ir, 8, &index));
   CHECK_EQ(index, 2);
   CHECK(!agc_isa_ir_instruction_at(&ir, 6, &index));
   CHECK(!agc_isa_ir_instruction_at(&ir, 16, &index));

   CHECK(decodes(WORDS(0xbe800380, 0xbe802006, 0xffffffff), NULL, &ir, &error));
   CHECK_EQ(ir.code_words, 2);
   CHECK(agc_isa_inst_has_flag(&ir.instructions[1], AGC_ISA_FLAG_JUMP));

   CHECK(!agc_isa_decode(NULL, 0, &ir, &error));
   CHECK_EQ(error.kind, AGC_ISA_ERROR_EMPTY_PROGRAM);
   CHECK(!decodes(WORDS(0xbe800380), NULL, &ir, &error));
   CHECK_EQ(error.kind, AGC_ISA_ERROR_MISSING_END_PROGRAM);
   CHECK(!decodes(WORDS(0xbe807f00, S_ENDPGM), NULL, &ir, &error));
   CHECK_EQ(error.kind, AGC_ISA_ERROR_UNKNOWN_OPCODE);
   CHECK_EQ(error.pc, 0);
   CHECK_EQ(error.family, AGC_ISA_SOP1);
   CHECK_EQ(error.opcode, 0x7f);
   CHECK(!decodes(WORDS(0xd54b0000), NULL, &ir, &error));
   CHECK_EQ(error.kind, AGC_ISA_ERROR_TRUNCATED);
   CHECK(!decodes(WORDS(0xbf820010, S_ENDPGM), NULL, &ir, &error));
   CHECK_EQ(error.kind, AGC_ISA_ERROR_INVALID_BRANCH);
   CHECK_EQ(error.pc, 0);
   CHECK_EQ(error.branch_target, 68);
   CHECK(!decodes(WORDS(0xbf820001, S_ENDPGM, 0xffffffff), NULL, &ir, &error));
   CHECK_EQ(error.kind, AGC_ISA_ERROR_INVALID_ENCODING);
   CHECK_EQ(error.pc, 8);
   CHECK_EQ(error.word, 0xffffffff);
   char message[AGC_ISA_ERROR_TEXT_SIZE];
   agc_isa_error_format(&error, message, sizeof(message));
   CHECK(!strcmp(message, "invalid encoding 0xffffffff at 0x8"));

   static const uint32_t twelve[] = {12};
   struct agc_isa_decode_scope scope = {twelve, 1, false};
   CHECK(decodes(WORDS(0xbe802004, 0xdead0000, S_ENDPGM, 0xbe800380, S_ENDPGM), &scope, &ir, &error));
   CHECK_EQ(ir.code_words, 5);
   CHECK(agc_isa_ir_instruction_at(&ir, 12, &index));
   static const uint32_t eight[] = {8};
   scope.entries = eight;
   CHECK(decodes(WORDS(0xbe802004, 0x7e0202ff, S_ENDPGM, S_ENDPGM), &scope, &ir, &error));
   CHECK(agc_isa_ir_instruction_at(&ir, 8, &index));

   struct agc_isa_decode_scope open = {NULL, 0, true};
   CHECK(decodes(WORDS(0xbe802004, S_ENDPGM, 0x7e020200, S_ENDPGM, 0x7e020200, 0xffffffff, 0x7e0202ff, 0x00000111,
                       0xbf82fffd, 0x7e020200, 0xbf82fffc, 0x7e020200),
                 &open, &ir, &error));
   static const uint32_t kept[] = {0, 4, 8, 12, 24, 32};
   CHECK_EQ(ir.instruction_count, 6);
   for (unsigned i = 0; i < 6 && i < ir.instruction_count; i++)
      CHECK_EQ(ir.instructions[i].pc, kept[i]);
   CHECK_EQ(ir.code_words, 9);
   CHECK_EQ(ir.instructions[5].branch_target, 24);

   CHECK(decodes(WORDS(0xbf820002, 0x7e020200, 0xffffffff, S_ENDPGM), NULL, &ir, &error));
   CHECK_EQ(ir.instruction_count, 2);
   CHECK_EQ(ir.instructions[1].pc, 12);
   CHECK_EQ(ir.code_words, 4);

   CHECK(decodes(WORDS(0xbf850003, 0x7e020200, S_ENDPGM, 0xc0000000, 0x7e040201, S_ENDPGM, 0x12345678), NULL, &ir,
                 &error));
   CHECK_EQ(ir.instruction_count, 5);
   CHECK_EQ(ir.instructions[3].pc, 16);
   CHECK_EQ(ir.code_words, 6);

   agc_isa_ir_finish(&ir);
}

static void
test_registers(void)
{
   struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES];
   struct agc_isa_inst add = one(WORDS(0x80028100));
   CHECK_EQ(agc_isa_inst_defs(&add, false, ranges), 1);
   CHECK(same_range(ranges[0], AGC_ISA_REG_SCALAR, 2, 1));
   CHECK_EQ(agc_isa_inst_uses(&add, false, ranges), 1);
   CHECK(same_range(ranges[0], AGC_ISA_REG_SCALAR, 0, 1));

   struct agc_isa_inst saveexec = one(WORDS(0xbe80246a));
   CHECK_EQ(agc_isa_inst_defs(&saveexec, true, ranges), 2);
   CHECK(same_range(ranges[0], AGC_ISA_REG_SCALAR, 0, 2));
   CHECK(same_range(ranges[1], AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_EXEC_LO, 2));
   CHECK_EQ(agc_isa_inst_uses(&saveexec, true, ranges), 2);
   CHECK(same_range(ranges[0], AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_VCC_LO, 2));

   struct agc_isa_inst fma = one(WORDS(0xd54b0004, 0x040e0501));
   CHECK_EQ(agc_isa_inst_defs(&fma, false, ranges), 1);
   CHECK(same_range(ranges[0], AGC_ISA_REG_VECTOR, 4, 1));
   CHECK_EQ(agc_isa_inst_uses(&fma, true, ranges), 4);
   CHECK(same_range(ranges[2], AGC_ISA_REG_VECTOR, 3, 1));
   CHECK(same_range(ranges[3], AGC_ISA_REG_SCALAR, AGC_ISA_SCALAR_EXEC_LO, 2));

   struct agc_isa_inst sample = one(WORDS(0xf0800f08, 0x00820004));
   CHECK_EQ(agc_isa_inst_uses(&sample, false, ranges), 3);
   CHECK(same_range(ranges[0], AGC_ISA_REG_VECTOR, 4, 2));
   CHECK(same_range(ranges[1], AGC_ISA_REG_SCALAR, 8, 8));
   CHECK(same_range(ranges[2], AGC_ISA_REG_SCALAR, 16, 4));

   struct agc_isa_inst exp = one(WORDS(0xf800180f, 0x03020100));
   CHECK_EQ(agc_isa_inst_uses(&exp, false, ranges), 4);
   CHECK_EQ(agc_isa_inst_defs(&exp, false, ranges), 0);
}

static void
test_tables(void)
{
   struct agc_isa_operand op;
   CHECK(agc_isa_operand_from_code(105, &op) && is_op(op, AGC_ISA_OPERAND_SGPR, 105));
   CHECK(agc_isa_operand_from_code(123, &op) && is_op(op, AGC_ISA_OPERAND_TTMP, 15));
   CHECK(agc_isa_operand_from_code(208, &op) && op.inline_int == -16);
   CHECK(agc_isa_operand_from_code(511, &op) && is_op(op, AGC_ISA_OPERAND_VGPR, 255));
   CHECK(!agc_isa_operand_from_code(209, &op));
   CHECK(!agc_isa_operand_from_code(249, &op));
   CHECK(!agc_isa_operand_scalar_destination(128, &op));
   uint16_t code;
   CHECK(agc_isa_operand_code(agc_isa_operand_special(AGC_ISA_OPERAND_NULL), AGC_ISA_TARGET_GFX1013, &code));
   CHECK_EQ(code, 125);
   CHECK(agc_isa_operand_code(agc_isa_operand_special(AGC_ISA_OPERAND_NULL), AGC_ISA_TARGET_GFX115, &code));
   CHECK_EQ(code, 124);
   CHECK(!agc_isa_operand_code(agc_isa_operand_special(AGC_ISA_OPERAND_LDS_DIRECT), AGC_ISA_TARGET_GFX12, &code));
   CHECK(agc_isa_operand_equal(agc_isa_operand_constant(0x3f800000), agc_isa_operand_inline_float(242)));
   CHECK(agc_isa_operand_equal(agc_isa_operand_constant(65), agc_isa_operand_literal(65)));

   const struct agc_isa_op_info *mov = agc_isa_op_lookup(AGC_ISA_SOP1, AGC_ISA_SOP1_MOV_B32);
   CHECK(mov && !strcmp(mov->name, "s_mov_b32"));
   CHECK(agc_isa_xlat_equal(agc_isa_op_translate(mov, AGC_ISA_TARGET_GFX1013), agc_isa_xlat_op(0x03)));
   CHECK(agc_isa_xlat_equal(agc_isa_op_translate(mov, AGC_ISA_TARGET_GFX103), agc_isa_xlat_op(0x03)));
   CHECK(agc_isa_xlat_equal(agc_isa_op_translate(mov, AGC_ISA_TARGET_GFX115), agc_isa_xlat_op(0x00)));
   CHECK(!agc_isa_op_lookup(AGC_ISA_SOP1, 0x7f));
   CHECK(!agc_isa_op_lookup(AGC_ISA_MIMG, 0));
   const struct agc_isa_op_info *mad = agc_isa_op_lookup(AGC_ISA_VOP3, AGC_ISA_VOP3_MAD_F32);
   CHECK(mad && agc_isa_op_translate(mad, AGC_ISA_TARGET_GFX11).kind == AGC_ISA_XLAT_RULE);
   CHECK_EQ(agc_isa_op_translate(mad, AGC_ISA_TARGET_GFX11).rule, AGC_ISA_RULE_MAD);

   const struct agc_isa_vopc_info *cmp = agc_isa_vopc_decode(AGC_ISA_VOPC_CMP_EQ_U32);
   CHECK(cmp && cmp->type == AGC_ISA_CMP_U32 && cmp->cond == 2 && !cmp->exec);
   CHECK_EQ(cmp->gfx11_opcode, 0x4a);
   CHECK(!agc_isa_vopc_decode(0x40));
   bool constant;
   CHECK(agc_isa_vopc_is_constant_result(agc_isa_vopc_decode(0x00), &constant) && !constant);
   CHECK(agc_isa_vopc_is_constant_result(agc_isa_vopc_decode(0x0f), &constant) && constant);
   CHECK(!agc_isa_vopc_is_constant_result(cmp, &constant));

   const struct agc_isa_ds_op *cmpst = agc_isa_ds_op(AGC_ISA_DS_CMPST_B32);
   CHECK(cmpst && agc_isa_ds_op_swaps_data(cmpst, AGC_ISA_TARGET_GFX115));
   CHECK(!agc_isa_ds_op_swaps_data(cmpst, AGC_ISA_TARGET_GFX103));
   const struct agc_isa_ds_op *add_rtn = agc_isa_ds_op(AGC_ISA_DS_ADD_RTN_F32);
   CHECK(agc_isa_xlat_equal(agc_isa_ds_op_translate(add_rtn, AGC_ISA_TARGET_GFX12),
                            agc_isa_xlat_op(AGC_ISA_DS_ADD_RTN_F32_GFX11)));

   const struct agc_isa_image_op *adjusted = agc_isa_image_op(AGC_ISA_MIMG_SAMPLE_A);
   CHECK(adjusted && agc_isa_image_op_plain(adjusted) == agc_isa_image_op(AGC_ISA_MIMG_SAMPLE));
   CHECK(agc_isa_xlat_equal(agc_isa_image_op_translate(adjusted, AGC_ISA_TARGET_GFX115), agc_isa_xlat_op(0x1b)));
   CHECK(agc_isa_xlat_equal(agc_isa_image_op_translate(adjusted, AGC_ISA_TARGET_GFX1013), agc_isa_xlat_op(0xa0)));

   uint8_t format;
   CHECK(agc_isa_remap_buffer_format(77, AGC_ISA_TARGET_GFX115, &format) && format == 63);
   CHECK(!agc_isa_remap_buffer_format(30, AGC_ISA_TARGET_GFX115, &format));
   CHECK(agc_isa_remap_buffer_format(30, AGC_ISA_TARGET_GFX103, &format) && format == 30);

   char scratch[AGC_ISA_MNEMONIC_SIZE];
   CHECK(!strcmp(agc_isa_mnemonic(AGC_ISA_SOP1, 0x7f, scratch), "sop1_op_0x7f"));
   CHECK(!strcmp(agc_isa_mnemonic(AGC_ISA_VOP3, 0x182, scratch), "v_readfirstlane_b32"));
   CHECK(!strcmp(agc_isa_mnemonic(AGC_ISA_FLAT, AGC_ISA_FLAT_LOAD_DWORD_ADDTID, scratch), "flat_load_dword_addtid"));
   enum agc_isa_target target;
   CHECK(agc_isa_target_from_gfx_ip(11, 5, 1, &target) && target == AGC_ISA_TARGET_GFX115);
   CHECK(!agc_isa_target_from_gfx_ip(10, 1, 0, &target));
   CHECK_EQ(agc_isa_target_from_host(AGC_HOST_GFX115), AGC_ISA_TARGET_GFX115);
}

static void
test_host_decoding(void)
{
   static struct agc_isa_host_tables gfx115;
   static struct agc_isa_host_tables gfx103;
   static struct agc_isa_host_tables gfx12;
   agc_isa_host_tables_init(&gfx115, AGC_ISA_TARGET_GFX115);
   agc_isa_host_tables_init(&gfx103, AGC_ISA_TARGET_GFX103);
   agc_isa_host_tables_init(&gfx12, AGC_ISA_TARGET_GFX12);
   struct agc_isa_ir ir;
   struct agc_isa_error error;
   agc_isa_ir_init(&ir);
   uint16_t guest;
   CHECK(agc_isa_host_guest_opcode(&gfx115, AGC_ISA_SOP1, 0x00, &guest) && guest == AGC_ISA_SOP1_MOV_B32);
   CHECK(agc_isa_host_guest_opcode(&gfx115, AGC_ISA_SOPP, 0x30, &guest) && guest == AGC_ISA_SOPP_ENDPGM);
   CHECK(agc_isa_host_guest_opcode(&gfx115, AGC_ISA_VOPC, 0x4a, &guest) && guest == AGC_ISA_VOPC_CMP_EQ_U32);
   CHECK(agc_isa_host_guest_opcode(&gfx103, AGC_ISA_SOP1, 0x03, &guest) && guest == AGC_ISA_SOP1_MOV_B32);

   CHECK(agc_isa_host_decode(&gfx115, WORDS(0xbe80007c, 0xbf8903f1, 0xbfb00000), &ir, &error));
   CHECK_EQ(ir.instruction_count, 3);
   CHECK_EQ(ir.code_words, 3);
   CHECK_EQ(ir.instructions[0].opcode, AGC_ISA_SOP1_MOV_B32);
   CHECK(is_op(ir.instructions[0].sop1.src0, AGC_ISA_OPERAND_NULL, 0));
   CHECK_EQ(ir.instructions[1].opcode, AGC_ISA_SOPP_WAITCNT);
   CHECK_EQ(ir.instructions[1].sopp.simm16, 0x3f10);
   CHECK_EQ(ir.instructions[2].opcode, AGC_ISA_SOPP_ENDPGM);

   CHECK(agc_isa_host_decode(&gfx115, WORDS(0xce000204, 0xcd000305, 0x040e0501, 0xbe834d83), &ir, &error));
   CHECK_EQ(ir.instruction_count, 3);
   CHECK_EQ(ir.instructions[0].family, AGC_ISA_LDSDIR);
   CHECK_EQ(ir.instructions[0].opcode, AGC_ISA_LDSDIR_PARAM_LOAD);
   CHECK_EQ(ir.instructions[0].ldsdir.vdst, 4);
   CHECK_EQ(ir.instructions[0].ldsdir.chan, 2);
   CHECK_EQ(ir.instructions[1].family, AGC_ISA_VINTERP);
   CHECK_EQ(ir.instructions[1].opcode, AGC_ISA_VINTERP_P10_F32);
   CHECK_EQ(ir.instructions[1].vinterp.vdst, 5);
   CHECK_EQ(ir.instructions[1].vinterp.wait_exp, 3);
   CHECK(is_op(ir.instructions[1].vinterp.src[2], AGC_ISA_OPERAND_VGPR, 3));
   CHECK_EQ(ir.instructions[2].family, AGC_ISA_SMEM);
   CHECK_EQ(ir.instructions[2].opcode, AGC_ISA_SMEM_MEMREALTIME);
   CHECK(is_op(ir.instructions[2].smem.sdata, AGC_ISA_OPERAND_SGPR, 3));

   CHECK(agc_isa_host_decode(&gfx115, WORDS(0xf06c0f05, 0x40020004, 0x00000007), &ir, &error));
   CHECK_EQ(ir.instruction_count, 1);
   CHECK_EQ(ir.instructions[0].opcode, AGC_ISA_MIMG_SAMPLE);
   CHECK_EQ(ir.instructions[0].mimg.dim, 1);
   CHECK_EQ(ir.instructions[0].mimg.ssamp, 64);
   CHECK(ir.instructions[0].mimg.nsa);
   CHECK_EQ(ir.instructions[0].mimg.addr[1], 7);

   CHECK(agc_isa_host_decode(&gfx115, WORDS(0xdc520010, 0x01840002), &ir, &error));
   CHECK_EQ(ir.instructions[0].opcode, AGC_ISA_VMEM_LOAD_DWORD);
   CHECK_EQ(ir.instructions[0].flat.segment, AGC_ISA_SEGMENT_GLOBAL);
   CHECK(is_op(ir.instructions[0].flat.saddr, AGC_ISA_OPERAND_SGPR, 4));
   CHECK_EQ(ir.instructions[0].flat.offset, 0x10);

   CHECK(agc_isa_host_decode(&gfx103, WORDS(0xbe800380, 0xbf810000), &ir, &error));
   CHECK_EQ(ir.instruction_count, 2);
   CHECK(!agc_isa_host_decode(&gfx115, WORDS(0xc8080500), &ir, &error));
   CHECK_EQ(error.kind, AGC_ISA_ERROR_INVALID_ENCODING);
   CHECK(!agc_isa_host_decode(&gfx115, WORDS(0xbfa00004), &ir, &error));
   CHECK_EQ(error.kind, AGC_ISA_ERROR_INVALID_BRANCH);
   CHECK(!agc_isa_host_decode(&gfx12, WORDS(0xbfb00000), &ir, &error));
   CHECK_EQ(error.kind, AGC_ISA_ERROR_INVALID_OPTIONS);
   agc_isa_ir_finish(&ir);
}

int
main(void)
{
   test_family_prefixes();
   test_scalar_formats();
   test_vector_formats();
   test_memory_formats();
   test_image_formats();
   test_rejected_encodings();
   test_programs();
   test_registers();
   test_tables();
   test_host_decoding();
   return TEST_RESULT();
}
