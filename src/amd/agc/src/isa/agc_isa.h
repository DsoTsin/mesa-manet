#ifndef AGC_ISA_H
#define AGC_ISA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agc_shader_artifact.h"
#include "agc_types.h"

#include "agc_isa_opcodes.h"

enum agc_isa_target {
   AGC_ISA_TARGET_GFX1013,
   AGC_ISA_TARGET_GFX103,
   AGC_ISA_TARGET_GFX11,
   AGC_ISA_TARGET_GFX115,
   AGC_ISA_TARGET_GFX12,
   AGC_ISA_TARGET_COUNT
};

_Static_assert(AGC_ISA_TARGET_GFX103 == AGC_HOST_GFX103 + 1, "host order");
_Static_assert(AGC_ISA_TARGET_GFX11 == AGC_HOST_GFX11 + 1, "host order");
_Static_assert(AGC_ISA_TARGET_GFX115 == AGC_HOST_GFX115 + 1, "host order");
_Static_assert(AGC_ISA_TARGET_GFX12 == AGC_HOST_GFX12 + 1, "host order");

static inline enum agc_isa_target
agc_isa_target_from_host(enum agc_host host)
{
   return (enum agc_isa_target)(host + 1);
}

static inline bool
agc_isa_target_is_gfx11_plus(enum agc_isa_target target)
{
   return target == AGC_ISA_TARGET_GFX11 || target == AGC_ISA_TARGET_GFX115 || target == AGC_ISA_TARGET_GFX12;
}

static inline bool
agc_isa_target_is_gfx11(enum agc_isa_target target)
{
   return target == AGC_ISA_TARGET_GFX11 || target == AGC_ISA_TARGET_GFX115;
}

static inline bool
agc_isa_target_is_gfx12(enum agc_isa_target target)
{
   return target == AGC_ISA_TARGET_GFX12;
}

static inline unsigned
agc_isa_target_instruction_cache_line_bytes(enum agc_isa_target target)
{
   return agc_isa_target_is_gfx11_plus(target) ? 128 : 64;
}

const char *agc_isa_target_name(enum agc_isa_target target);
bool agc_isa_target_from_gfx_ip(uint32_t major, uint32_t minor, uint32_t stepping, enum agc_isa_target *target);

enum agc_isa_family {
   AGC_ISA_SOP1,
   AGC_ISA_SOP2,
   AGC_ISA_SOPK,
   AGC_ISA_SOPC,
   AGC_ISA_SOPP,
   AGC_ISA_VOP1,
   AGC_ISA_VOP2,
   AGC_ISA_VOP3,
   AGC_ISA_VOP3P,
   AGC_ISA_VOPC,
   AGC_ISA_VINTRP,
   AGC_ISA_SMEM,
   AGC_ISA_MUBUF,
   AGC_ISA_MTBUF,
   AGC_ISA_FLAT,
   AGC_ISA_DS,
   AGC_ISA_MIMG,
   AGC_ISA_EXP,
   AGC_ISA_LDSDIR,
   AGC_ISA_VINTERP,
   AGC_ISA_FAMILY_COUNT
};

const char *agc_isa_family_name(enum agc_isa_family family);

static inline bool
agc_isa_family_is_scalar_alu(enum agc_isa_family family)
{
   return family == AGC_ISA_SOP1 || family == AGC_ISA_SOP2 || family == AGC_ISA_SOPK ||
          family == AGC_ISA_SOPC || family == AGC_ISA_SOPP;
}

static inline bool
agc_isa_family_is_vector_alu(enum agc_isa_family family)
{
   return family == AGC_ISA_VOP1 || family == AGC_ISA_VOP2 || family == AGC_ISA_VOP3 ||
          family == AGC_ISA_VOP3P || family == AGC_ISA_VOPC || family == AGC_ISA_VINTRP ||
          family == AGC_ISA_VINTERP;
}

static inline bool
agc_isa_family_is_vector_memory(enum agc_isa_family family)
{
   return family == AGC_ISA_MUBUF || family == AGC_ISA_MTBUF || family == AGC_ISA_FLAT ||
          family == AGC_ISA_MIMG;
}

#define AGC_ISA_SGPR_LIMIT 106
#define AGC_ISA_VGPR_LIMIT 256
#define AGC_ISA_SCALAR_VCC_LO 106
#define AGC_ISA_SCALAR_VCC_HI 107
#define AGC_ISA_SCALAR_TTMP_BASE 108
#define AGC_ISA_SCALAR_M0 124
#define AGC_ISA_SCALAR_EXEC_LO 126
#define AGC_ISA_SCALAR_EXEC_HI 127
#define AGC_ISA_MAX_NSA_ADDRESSES 13

#define AGC_ISA_SDWA_SEL_WORD_1 5
#define AGC_ISA_SDWA_SEL_DWORD 6
#define AGC_ISA_SDWA_UNUSED_PAD 0
#define AGC_ISA_SDWA_UNUSED_SEXT 1
#define AGC_ISA_SDWA_UNUSED_PRESERVE 2

#define AGC_ISA_CODE_INLINE_INT_ZERO 128
#define AGC_ISA_CODE_INLINE_INT_MAX 192
#define AGC_ISA_CODE_INLINE_INT_NEG_MAX 208
#define AGC_ISA_CODE_DPP8 233
#define AGC_ISA_CODE_DPP8_FI 234
#define AGC_ISA_CODE_INLINE_FLOAT_FIRST 240
#define AGC_ISA_CODE_INLINE_FLOAT_LAST 248
#define AGC_ISA_CODE_SDWA 249
#define AGC_ISA_CODE_DPP16 250
#define AGC_ISA_CODE_LITERAL 255
#define AGC_ISA_CODE_VGPR_BASE 256
#define AGC_ISA_CODE_GFX10_M0 124
#define AGC_ISA_CODE_GFX10_NULL 125
#define AGC_ISA_CODE_GFX11_NULL 124
#define AGC_ISA_CODE_GFX11_M0 125

#define AGC_ISA_EXP_TARGET_NULL 9

#define AGC_ISA_VINTRP_P1 0
#define AGC_ISA_VINTRP_P2 1
#define AGC_ISA_VINTRP_MOV 2
#define AGC_ISA_LDSDIR_PARAM_LOAD 0
#define AGC_ISA_VINTERP_P10_F32 0
#define AGC_ISA_VINTERP_P2_F32 1
#define AGC_ISA_VINTERP_P10_F16_F32 2
#define AGC_ISA_VINTERP_P2_F16_F32 3
#define AGC_ISA_VOP3_VOPC_LAST 0xff
#define AGC_ISA_VOP3_VOP2_BASE 0x100
#define AGC_ISA_VOP3_VOP2_LAST 0x13f
#define AGC_ISA_VOP3_VOP1_BASE 0x180
#define AGC_ISA_VOP3_VOP1_LAST 0x1ff
#define AGC_ISA_OPCODE_LIMIT 0x400

#define AGC_ISA_DS_ADD_RTN_F32_GFX11 0x79
#define AGC_ISA_GFX11_SOP1_SENDMSG_RTN_B64 0x4d
#define AGC_ISA_GFX11_SOPP_WAITCNT 0x09
#define AGC_ISA_GFX12_MIMG_MSAA_LOAD 0x18
#define AGC_ISA_MSG_RTN_GET_REALTIME 0x83
#define AGC_ISA_SIGNED_DOT_SOURCES 3

#define AGC_ISA_FLAG_D2 (UINT64_C(1) << 0)
#define AGC_ISA_FLAG_S0_2 (UINT64_C(1) << 1)
#define AGC_ISA_FLAG_S1_2 (UINT64_C(1) << 2)
#define AGC_ISA_FLAG_S2_2 (UINT64_C(1) << 3)
#define AGC_ISA_FLAG_BRANCH (UINT64_C(1) << 4)
#define AGC_ISA_FLAG_COND_BRANCH (UINT64_C(1) << 5)
#define AGC_ISA_FLAG_END (UINT64_C(1) << 6)
#define AGC_ISA_FLAG_JUMP (UINT64_C(1) << 7)
#define AGC_ISA_FLAG_CALL (UINT64_C(1) << 8)
#define AGC_ISA_FLAG_NO_DST (UINT64_C(1) << 9)
#define AGC_ISA_FLAG_NO_SRC (UINT64_C(1) << 10)
#define AGC_ISA_FLAG_NO_OPERANDS (UINT64_C(1) << 11)
#define AGC_ISA_FLAG_WRITES_EXEC (UINT64_C(1) << 12)
#define AGC_ISA_FLAG_READS_EXEC (UINT64_C(1) << 13)
#define AGC_ISA_FLAG_READS_VCC (UINT64_C(1) << 14)
#define AGC_ISA_FLAG_WRITES_VCC (UINT64_C(1) << 15)
#define AGC_ISA_FLAG_READS_M0 (UINT64_C(1) << 16)
#define AGC_ISA_FLAG_READS_DST (UINT64_C(1) << 17)
#define AGC_ISA_FLAG_CARRY_IN (UINT64_C(1) << 18)
#define AGC_ISA_FLAG_CARRY_OUT (UINT64_C(1) << 19)
#define AGC_ISA_FLAG_SDST (UINT64_C(1) << 20)
#define AGC_ISA_FLAG_MASK_SRC2 (UINT64_C(1) << 21)
#define AGC_ISA_FLAG_SWAP (UINT64_C(1) << 22)
#define AGC_ISA_FLAG_TRANS (UINT64_C(1) << 23)
#define AGC_ISA_FLAG_T16 (UINT64_C(1) << 24)
#define AGC_ISA_FLAG_LITK (UINT64_C(1) << 25)
#define AGC_ISA_FLAG_CMPX (UINT64_C(1) << 26)
#define AGC_ISA_FLAG_CMPSWAP (UINT64_C(1) << 27)
#define AGC_ISA_FLAG_IMM_U (UINT64_C(1) << 28)
#define AGC_ISA_FLAG_VOP3B (UINT64_C(1) << 29)
#define AGC_ISA_FLAG_SMEM_ADDR (UINT64_C(1) << 30)
#define AGC_ISA_FLAG_SRC1 (UINT64_C(1) << 31)
#define AGC_ISA_FLAG_SRC2 (UINT64_C(1) << 32)
#define AGC_ISA_FLAG_SRC3 (UINT64_C(1) << 33)
#define AGC_ISA_FLAG_RELATIVE (UINT64_C(1) << 34)
#define AGC_ISA_FLAG_PC_RELATIVE (UINT64_C(1) << 35)
#define AGC_ISA_FLAG_LANE_SELECT (UINT64_C(1) << 36)
#define AGC_ISA_FLAG_NO_VOP3 (UINT64_C(1) << 38)

#define AGC_ISA_DS_FLAG_NO_ADDR (1u << 0)
#define AGC_ISA_DS_FLAG_GWS (1u << 1)
#define AGC_ISA_DS_FLAG_SRC2 (1u << 2)
#define AGC_ISA_DS_FLAG_GDS_ONLY (1u << 3)
#define AGC_ISA_DS_FLAG_CMPST (1u << 4)
#define AGC_ISA_DS_FLAG_NOT_GFX12 (1u << 5)
#define AGC_ISA_DS_FLAG_STACK8 (1u << 6)

#define AGC_ISA_IMG_SAMPLER (1u << 0)
#define AGC_ISA_IMG_GATHER (1u << 1)
#define AGC_ISA_IMG_OFFSET (1u << 2)
#define AGC_ISA_IMG_BIAS (1u << 3)
#define AGC_ISA_IMG_COMPARE (1u << 4)
#define AGC_ISA_IMG_LOD (1u << 5)
#define AGC_ISA_IMG_CLAMP (1u << 6)
#define AGC_ISA_IMG_DERIV (1u << 7)
#define AGC_ISA_IMG_LZ (1u << 8)
#define AGC_ISA_IMG_G16 (1u << 9)
#define AGC_ISA_IMG_MIP (1u << 10)
#define AGC_ISA_IMG_ATOMIC_OP (1u << 11)
#define AGC_ISA_IMG_STORE_OP (1u << 12)
#define AGC_ISA_IMG_RESINFO (1u << 13)
#define AGC_ISA_IMG_MSAA (1u << 14)
#define AGC_ISA_IMG_BVH (1u << 15)
#define AGC_ISA_IMG_BVH64 (1u << 16)
#define AGC_ISA_IMG_CMPSWAP (1u << 17)
#define AGC_ISA_IMG_PACKED_GATHER (1u << 18)
#define AGC_ISA_IMG_ADJUST (1u << 19)
#define AGC_ISA_IMG_BVH8 (1u << 20)
#define AGC_ISA_IMG_ADJUST_OPCODE_BIT 0x80
#define AGC_ISA_IMG_SAMPLER_ADJUST_BITS 0x3ffff000u

enum agc_isa_operand_kind {
   AGC_ISA_OPERAND_NONE,
   AGC_ISA_OPERAND_SGPR,
   AGC_ISA_OPERAND_VGPR,
   AGC_ISA_OPERAND_VCC_LO,
   AGC_ISA_OPERAND_VCC_HI,
   AGC_ISA_OPERAND_TTMP,
   AGC_ISA_OPERAND_M0,
   AGC_ISA_OPERAND_NULL,
   AGC_ISA_OPERAND_EXEC_LO,
   AGC_ISA_OPERAND_EXEC_HI,
   AGC_ISA_OPERAND_INLINE_INT,
   AGC_ISA_OPERAND_INLINE_FLOAT,
   AGC_ISA_OPERAND_LITERAL,
   AGC_ISA_OPERAND_SHARED_BASE,
   AGC_ISA_OPERAND_SHARED_LIMIT,
   AGC_ISA_OPERAND_PRIVATE_BASE,
   AGC_ISA_OPERAND_PRIVATE_LIMIT,
   AGC_ISA_OPERAND_POPS_EXITING_WAVE_ID,
   AGC_ISA_OPERAND_VCC_Z,
   AGC_ISA_OPERAND_EXEC_Z,
   AGC_ISA_OPERAND_SCC,
   AGC_ISA_OPERAND_LDS_DIRECT,
   AGC_ISA_OPERAND_RAW,
   AGC_ISA_OPERAND_KIND_COUNT
};

struct agc_isa_operand {
   uint8_t kind;
   union {
      uint32_t value;
      uint32_t reg;
      int32_t inline_int;
      uint32_t inline_float;
      uint32_t literal;
      uint32_t raw;
   };
};

static inline struct agc_isa_operand
agc_isa_operand_make(enum agc_isa_operand_kind kind, uint32_t value)
{
   struct agc_isa_operand op = {.kind = (uint8_t)kind, .value = value};
   return op;
}

static inline struct agc_isa_operand
agc_isa_operand_none(void)
{
   return agc_isa_operand_make(AGC_ISA_OPERAND_NONE, 0);
}

static inline struct agc_isa_operand
agc_isa_operand_sgpr(uint8_t reg)
{
   return agc_isa_operand_make(AGC_ISA_OPERAND_SGPR, reg);
}

static inline struct agc_isa_operand
agc_isa_operand_vgpr(uint8_t reg)
{
   return agc_isa_operand_make(AGC_ISA_OPERAND_VGPR, reg);
}

static inline struct agc_isa_operand
agc_isa_operand_ttmp(uint8_t reg)
{
   return agc_isa_operand_make(AGC_ISA_OPERAND_TTMP, reg);
}

static inline struct agc_isa_operand
agc_isa_operand_inline_int(int8_t value)
{
   return agc_isa_operand_make(AGC_ISA_OPERAND_INLINE_INT, (uint32_t)(int32_t)value);
}

static inline struct agc_isa_operand
agc_isa_operand_inline_float(uint8_t code)
{
   return agc_isa_operand_make(AGC_ISA_OPERAND_INLINE_FLOAT, code);
}

static inline struct agc_isa_operand
agc_isa_operand_literal(uint32_t value)
{
   return agc_isa_operand_make(AGC_ISA_OPERAND_LITERAL, value);
}

static inline struct agc_isa_operand
agc_isa_operand_raw(uint16_t code)
{
   return agc_isa_operand_make(AGC_ISA_OPERAND_RAW, code);
}

static inline struct agc_isa_operand
agc_isa_operand_special(enum agc_isa_operand_kind kind)
{
   return agc_isa_operand_make(kind, 0);
}

static inline bool
agc_isa_operand_equal(struct agc_isa_operand a, struct agc_isa_operand b)
{
   return a.kind == b.kind && a.value == b.value;
}

static inline bool
agc_isa_operand_is_some(struct agc_isa_operand op)
{
   return op.kind != AGC_ISA_OPERAND_NONE;
}

static inline bool
agc_isa_operand_is_vgpr(struct agc_isa_operand op)
{
   return op.kind == AGC_ISA_OPERAND_VGPR;
}

bool agc_isa_operand_from_code(uint16_t code, struct agc_isa_operand *op);
bool agc_isa_operand_scalar_destination(uint16_t code, struct agc_isa_operand *op);
struct agc_isa_operand agc_isa_operand_from_code_or_raw(uint16_t code);
bool agc_isa_operand_code(struct agc_isa_operand op, enum agc_isa_target target, uint16_t *code);
bool agc_isa_operand_scalar_index(struct agc_isa_operand op, uint16_t *index);
bool agc_isa_inline_float_bits(uint8_t code, uint32_t *bits);
bool agc_isa_operand_inline_32(uint32_t value, struct agc_isa_operand *op);
struct agc_isa_operand agc_isa_operand_constant(uint32_t value);

struct agc_isa_sdwa {
   uint8_t dst_sel;
   uint8_t dst_unused;
   bool clamp;
   uint8_t omod;
   uint8_t src0_sel;
   bool src0_sext;
   bool src0_neg;
   bool src0_abs;
   uint8_t src1_sel;
   bool src1_sext;
   bool src1_neg;
   bool src1_abs;
};

struct agc_isa_dpp16 {
   uint16_t ctrl;
   bool fetch_inactive;
   bool bound_ctrl;
   bool src0_neg;
   bool src0_abs;
   bool src1_neg;
   bool src1_abs;
   uint8_t bank_mask;
   uint8_t row_mask;
};

struct agc_isa_dpp8 {
   uint32_t selects;
   bool fetch_inactive;
};

enum agc_isa_vop_ext_kind {
   AGC_ISA_VOP_EXT_NONE,
   AGC_ISA_VOP_EXT_SDWA,
   AGC_ISA_VOP_EXT_DPP16,
   AGC_ISA_VOP_EXT_DPP8
};

struct agc_isa_vop_ext {
   uint8_t kind;
   union {
      struct agc_isa_sdwa sdwa;
      struct agc_isa_dpp16 dpp16;
      struct agc_isa_dpp8 dpp8;
   };
};

enum agc_isa_mem_kind {
   AGC_ISA_MEM_LOAD,
   AGC_ISA_MEM_STORE,
   AGC_ISA_MEM_ATOMIC,
   AGC_ISA_MEM_CONTROL
};

enum agc_isa_flat_segment {
   AGC_ISA_SEGMENT_FLAT,
   AGC_ISA_SEGMENT_SCRATCH,
   AGC_ISA_SEGMENT_GLOBAL
};

struct agc_isa_sop1 {
   struct agc_isa_operand sdst;
   struct agc_isa_operand src0;
};

struct agc_isa_sop2 {
   struct agc_isa_operand sdst;
   struct agc_isa_operand src0;
   struct agc_isa_operand src1;
};

struct agc_isa_sopk {
   struct agc_isa_operand sdst;
   uint32_t imm32;
   uint16_t simm16;
   bool has_imm32;
};

struct agc_isa_sopc {
   struct agc_isa_operand src0;
   struct agc_isa_operand src1;
};

struct agc_isa_sopp {
   uint16_t simm16;
};

struct agc_isa_vop1 {
   struct agc_isa_operand vdst;
   struct agc_isa_operand src0;
   struct agc_isa_vop_ext ext;
};

struct agc_isa_vop2 {
   struct agc_isa_operand vdst;
   struct agc_isa_operand src0;
   struct agc_isa_operand vsrc1;
   uint32_t constant;
   bool has_constant;
   struct agc_isa_vop_ext ext;
};

struct agc_isa_vopc {
   struct agc_isa_operand sdst;
   struct agc_isa_operand src0;
   struct agc_isa_operand vsrc1;
   struct agc_isa_vop_ext ext;
};

struct agc_isa_vop3 {
   struct agc_isa_operand vdst;
   struct agc_isa_operand sdst;
   struct agc_isa_operand src[3];
   uint8_t abs;
   uint8_t neg;
   uint8_t opsel;
   bool clamp;
   uint8_t omod;
};

struct agc_isa_vop3p {
   struct agc_isa_operand vdst;
   struct agc_isa_operand src[3];
   uint8_t neg;
   uint8_t neg_hi;
   uint8_t opsel;
   uint8_t opsel_hi;
   bool clamp;
};

struct agc_isa_interp {
   uint8_t vdst;
   uint8_t vsrc;
   uint8_t attr;
   uint8_t chan;
};

struct agc_isa_smem {
   struct agc_isa_operand sdata;
   struct agc_isa_operand soffset;
   int32_t offset;
   uint8_t sbase;
   bool glc;
   bool dlc;
   uint8_t dwords;
   bool buffer;
};

struct agc_isa_buffer {
   struct agc_isa_operand soffset;
   uint16_t offset;
   uint8_t vdata;
   uint8_t vaddr;
   uint8_t srsrc;
   bool offen;
   bool idxen;
   bool glc;
   bool dlc;
   bool slc;
   bool tfe;
   bool lds;
   uint8_t format;
   uint8_t kind;
   uint8_t data_dwords;
};

struct agc_isa_flat {
   struct agc_isa_operand saddr;
   int32_t offset;
   uint8_t segment;
   uint8_t vdst;
   uint8_t vdata;
   uint8_t vaddr;
   bool uses_vaddr;
   bool glc;
   bool dlc;
   bool slc;
   bool lds;
   uint8_t kind;
   uint8_t data_dwords;
};

struct agc_isa_ds {
   uint8_t vdst;
   uint8_t addr;
   uint8_t data0;
   uint8_t data1;
   uint8_t offset0;
   uint8_t offset1;
   bool gds;
   uint8_t dst_dwords;
   uint8_t data0_dwords;
   uint8_t data1_dwords;
   bool uses_addr;
   bool writes_addr;
};

struct agc_isa_image {
   uint8_t dim;
   uint8_t dmask;
   bool unorm;
   bool glc;
   bool slc;
   bool dlc;
   bool r128;
   bool tfe;
   bool lwe;
   bool a16;
   bool d16;
   uint8_t vdata;
   uint8_t addr[AGC_ISA_MAX_NSA_ADDRESSES];
   bool nsa;
   uint8_t nsa_words;
   uint8_t addr_dwords;
   bool bvh8;
   uint8_t srsrc;
   uint8_t ssamp;
   bool uses_sampler;
   uint8_t kind;
   uint8_t data_dwords;
};

struct agc_isa_export {
   uint8_t target;
   uint8_t en;
   bool compr;
   bool done;
   bool vm;
   bool row_en;
   uint8_t vsrc[4];
};

struct agc_isa_lds_dir {
   uint8_t vdst;
   uint8_t attr;
   uint8_t chan;
   uint8_t wait_vdst;
};

struct agc_isa_vinterp {
   struct agc_isa_operand src[3];
   uint8_t vdst;
   uint8_t neg;
   uint8_t opsel;
   bool clamp;
   uint8_t wait_exp;
};

#define AGC_ISA_MAX_INSTRUCTION_WORDS 5

struct agc_isa_inst {
   uint64_t flags;
   uint32_t pc;
   uint32_t branch_target;
   uint16_t opcode;
   uint8_t family;
   uint8_t fields;
   uint8_t word_count;
   bool has_branch_target;
   union {
      struct agc_isa_sop1 sop1;
      struct agc_isa_sop2 sop2;
      struct agc_isa_sopk sopk;
      struct agc_isa_sopc sopc;
      struct agc_isa_sopp sopp;
      struct agc_isa_vop1 vop1;
      struct agc_isa_vop2 vop2;
      struct agc_isa_vopc vopc;
      struct agc_isa_vop3 vop3;
      struct agc_isa_vop3p vop3p;
      struct agc_isa_interp vintrp;
      struct agc_isa_smem smem;
      struct agc_isa_buffer buffer;
      struct agc_isa_flat flat;
      struct agc_isa_ds ds;
      struct agc_isa_image mimg;
      struct agc_isa_export exp;
      struct agc_isa_lds_dir ldsdir;
      struct agc_isa_vinterp vinterp;
   };
};

static inline bool
agc_isa_inst_has_flag(const struct agc_isa_inst *inst, uint64_t mask)
{
   return (inst->flags & mask) != 0;
}

static inline bool
agc_isa_inst_is_branch(const struct agc_isa_inst *inst)
{
   return agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_BRANCH | AGC_ISA_FLAG_COND_BRANCH);
}

static inline bool
agc_isa_inst_ends_block(const struct agc_isa_inst *inst)
{
   return agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_BRANCH | AGC_ISA_FLAG_COND_BRANCH |
                                         AGC_ISA_FLAG_END | AGC_ISA_FLAG_JUMP);
}

static inline bool
agc_isa_inst_falls_through(const struct agc_isa_inst *inst)
{
   return !agc_isa_inst_has_flag(inst, AGC_ISA_FLAG_BRANCH | AGC_ISA_FLAG_END | AGC_ISA_FLAG_JUMP);
}

static inline uint32_t
agc_isa_inst_end_pc(const struct agc_isa_inst *inst)
{
   return inst->pc + 4 * (uint32_t)inst->word_count;
}

uint8_t agc_isa_image_address_register(const struct agc_isa_image *image, unsigned index);

enum agc_isa_reg_file {
   AGC_ISA_REG_SCALAR,
   AGC_ISA_REG_VECTOR
};

struct agc_isa_reg_range {
   uint8_t file;
   uint16_t first;
   uint16_t count;
};

#define AGC_ISA_MAX_REG_RANGES 24

static inline uint16_t
agc_isa_reg_range_last(struct agc_isa_reg_range range)
{
   return (uint16_t)(range.first + (range.count ? range.count : 1) - 1);
}

static inline bool
agc_isa_reg_range_contains(struct agc_isa_reg_range range, enum agc_isa_reg_file file, uint16_t index)
{
   return range.file == file && index >= range.first && index < range.first + range.count;
}

unsigned agc_isa_inst_defs(const struct agc_isa_inst *inst, bool wave64,
                           struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES]);
unsigned agc_isa_inst_uses(const struct agc_isa_inst *inst, bool wave64,
                           struct agc_isa_reg_range ranges[AGC_ISA_MAX_REG_RANGES]);

enum agc_isa_xlat_kind {
   AGC_ISA_XLAT_SAME,
   AGC_ISA_XLAT_OP,
   AGC_ISA_XLAT_RULE,
   AGC_ISA_XLAT_DROP,
   AGC_ISA_XLAT_NO
};

enum agc_isa_rule {
   AGC_ISA_RULE_NONE,
   AGC_ISA_RULE_BVH8,
   AGC_ISA_RULE_BVH_STACK8,
   AGC_ISA_RULE_MAC,
   AGC_ISA_RULE_MAC_LEGACY,
   AGC_ISA_RULE_MAD,
   AGC_ISA_RULE_MAD_LEGACY,
   AGC_ISA_RULE_MADMK,
   AGC_ISA_RULE_MADAK,
   AGC_ISA_RULE_DOT2C,
   AGC_ISA_RULE_DOT4C,
   AGC_ISA_RULE_DOT_SIGNED,
   AGC_ISA_RULE_CMPK,
   AGC_ISA_RULE_HWREG,
   AGC_ISA_RULE_SENDMSG,
   AGC_ISA_RULE_WAITCNT,
   AGC_ISA_RULE_WAITCNT_COUNTER,
   AGC_ISA_RULE_BARRIER,
   AGC_ISA_RULE_MEM_TIME,
   AGC_ISA_RULE_INTERP,
   AGC_ISA_RULE_WAVE_ID,
   AGC_ISA_RULE_FIND_FIRST_ZERO,
   AGC_ISA_RULE_DOT2,
   AGC_ISA_RULE_COUNT
};

struct agc_isa_xlat {
   uint8_t kind;
   uint8_t rule;
   uint16_t value;
};

static inline struct agc_isa_xlat
agc_isa_xlat_op(uint16_t op)
{
   struct agc_isa_xlat x = {AGC_ISA_XLAT_OP, AGC_ISA_RULE_NONE, op};
   return x;
}

static inline struct agc_isa_xlat
agc_isa_xlat_rule(enum agc_isa_rule rule, uint16_t value)
{
   struct agc_isa_xlat x = {AGC_ISA_XLAT_RULE, (uint8_t)rule, value};
   return x;
}

static inline struct agc_isa_xlat
agc_isa_xlat_of(enum agc_isa_xlat_kind kind)
{
   struct agc_isa_xlat x = {(uint8_t)kind, AGC_ISA_RULE_NONE, 0};
   return x;
}

static inline bool
agc_isa_xlat_equal(struct agc_isa_xlat a, struct agc_isa_xlat b)
{
   return a.kind == b.kind && a.rule == b.rule && a.value == b.value;
}

struct agc_isa_op_info {
   const char *name;
   uint64_t flags;
   uint16_t op;
   struct agc_isa_xlat gfx103;
   struct agc_isa_xlat gfx11;
   struct agc_isa_xlat gfx12;
};

const struct agc_isa_op_info *agc_isa_op_table(enum agc_isa_family family, size_t *count);
const struct agc_isa_op_info *agc_isa_op_lookup(enum agc_isa_family family, uint16_t op);
struct agc_isa_xlat agc_isa_op_translate(const struct agc_isa_op_info *info, enum agc_isa_target target);

bool agc_isa_vop3_base(uint16_t op, enum agc_isa_family *family, uint16_t *base);
bool agc_isa_vop3_flags(uint16_t op, uint64_t *flags);
bool agc_isa_flag_source_count(uint64_t flags, unsigned *count);
unsigned agc_isa_vop3_source_count(uint16_t op, uint64_t flags);
unsigned agc_isa_source_count(const struct agc_isa_inst *inst);

enum agc_isa_cmp_type {
   AGC_ISA_CMP_F16,
   AGC_ISA_CMP_F32,
   AGC_ISA_CMP_F64,
   AGC_ISA_CMP_I16,
   AGC_ISA_CMP_U16,
   AGC_ISA_CMP_I32,
   AGC_ISA_CMP_U32,
   AGC_ISA_CMP_I64,
   AGC_ISA_CMP_U64
};

#define AGC_ISA_CMP_COND_CLASS 0xff

struct agc_isa_vopc_info {
   const char *name;
   uint64_t flags;
   uint16_t op;
   uint16_t gfx11_opcode;
   uint8_t type;
   uint8_t cond;
   bool exec;
};

const struct agc_isa_vopc_info *agc_isa_vopc_decode(uint16_t op);
bool agc_isa_vopc_is_constant_result(const struct agc_isa_vopc_info *info, bool *result);

#define AGC_ISA_MNEMONIC_SIZE 40

const char *agc_isa_mnemonic(enum agc_isa_family family, uint16_t op, char scratch[AGC_ISA_MNEMONIC_SIZE]);

static inline const char *
agc_isa_inst_mnemonic(const struct agc_isa_inst *inst, char scratch[AGC_ISA_MNEMONIC_SIZE])
{
   return agc_isa_mnemonic((enum agc_isa_family)inst->family, inst->opcode, scratch);
}

struct agc_isa_mem_op {
   const char *name;
   uint64_t flags;
   uint16_t op;
   uint8_t kind;
   uint8_t dwords;
   struct agc_isa_xlat gfx103;
   struct agc_isa_xlat gfx11;
   struct agc_isa_xlat gfx12;
};

enum agc_isa_mem_table {
   AGC_ISA_MEM_TABLE_SMEM,
   AGC_ISA_MEM_TABLE_VMEM_COMMON,
   AGC_ISA_MEM_TABLE_MUBUF_ONLY,
   AGC_ISA_MEM_TABLE_MTBUF,
   AGC_ISA_MEM_TABLE_FLAT_ONLY,
   AGC_ISA_MEM_TABLE_COUNT
};

const struct agc_isa_mem_op *agc_isa_mem_table(enum agc_isa_mem_table table, size_t *count);
const struct agc_isa_mem_op *agc_isa_smem_op(uint16_t op);
const struct agc_isa_mem_op *agc_isa_mubuf_op(uint16_t op);
const struct agc_isa_mem_op *agc_isa_mtbuf_op(uint16_t op);
const struct agc_isa_mem_op *agc_isa_flat_op(uint16_t op);
bool agc_isa_flat_only_op(uint16_t op);
struct agc_isa_xlat agc_isa_mem_op_translate(const struct agc_isa_mem_op *op, enum agc_isa_target target);

static inline bool
agc_isa_is_buffer_smem(const struct agc_isa_mem_op *op)
{
   return (op->flags & AGC_ISA_FLAG_IMM_U) != 0;
}

struct agc_isa_ds_op {
   const char *name;
   uint16_t op;
   uint8_t dst;
   uint8_t data0;
   uint8_t data1;
   uint8_t flags;
};

const struct agc_isa_ds_op *agc_isa_ds_table(size_t *count);
const struct agc_isa_ds_op *agc_isa_ds_op(uint16_t op);
struct agc_isa_xlat agc_isa_ds_op_translate(const struct agc_isa_ds_op *op, enum agc_isa_target target);
bool agc_isa_ds_op_swaps_data(const struct agc_isa_ds_op *op, enum agc_isa_target target);

struct agc_isa_image_op {
   const char *name;
   uint32_t flags;
   uint16_t op;
   struct agc_isa_xlat gfx103;
   struct agc_isa_xlat gfx11;
   struct agc_isa_xlat gfx12;
};

const struct agc_isa_image_op *agc_isa_image_table(size_t *count);
const struct agc_isa_image_op *agc_isa_image_op(uint16_t op);
struct agc_isa_xlat agc_isa_image_op_translate(const struct agc_isa_image_op *op, enum agc_isa_target target);
const struct agc_isa_image_op *agc_isa_image_op_plain(const struct agc_isa_image_op *op);

static inline enum agc_isa_mem_kind
agc_isa_image_op_kind(const struct agc_isa_image_op *op)
{
   if (op->flags & AGC_ISA_IMG_ATOMIC_OP)
      return AGC_ISA_MEM_ATOMIC;
   if (op->flags & AGC_ISA_IMG_STORE_OP)
      return AGC_ISA_MEM_STORE;
   return AGC_ISA_MEM_LOAD;
}

static inline bool
agc_isa_image_op_uses_sampler(const struct agc_isa_image_op *op)
{
   return (op->flags & (AGC_ISA_IMG_SAMPLER | AGC_ISA_IMG_GATHER)) != 0;
}

static inline bool
agc_isa_gfx12_vsample(uint16_t opcode, bool uses_sampler)
{
   return uses_sampler || opcode == AGC_ISA_GFX12_MIMG_MSAA_LOAD;
}

uint8_t agc_isa_image_coordinates(uint8_t dim);
uint8_t agc_isa_image_address_dwords(uint32_t flags, uint8_t dim, bool a16);
uint8_t agc_isa_image_data_dwords(uint32_t flags, uint8_t dmask, bool d16);
bool agc_isa_remap_buffer_format(uint8_t format, enum agc_isa_target target, uint8_t *remapped);

enum agc_isa_error_kind {
   AGC_ISA_ERROR_NONE,
   AGC_ISA_ERROR_UNSUPPORTED,
   AGC_ISA_ERROR_INVALID_OPTIONS,
   AGC_ISA_ERROR_TARGET_MISMATCH,
   AGC_ISA_ERROR_EMPTY_PROGRAM,
   AGC_ISA_ERROR_TRUNCATED,
   AGC_ISA_ERROR_INVALID_ENCODING,
   AGC_ISA_ERROR_UNKNOWN_OPCODE,
   AGC_ISA_ERROR_INVALID_OPERAND,
   AGC_ISA_ERROR_INVALID_BRANCH,
   AGC_ISA_ERROR_MISSING_END_PROGRAM,
   AGC_ISA_ERROR_UNSUPPORTED_INSTRUCTION,
   AGC_ISA_ERROR_BRANCH_OUT_OF_RANGE,
   AGC_ISA_ERROR_OUT_OF_REGISTERS,
   AGC_ISA_ERROR_OUT_OF_MEMORY
};

struct agc_isa_error {
   enum agc_isa_error_kind kind;
   uint32_t pc;
   uint32_t word;
   uint32_t branch_target;
   uint16_t opcode;
   uint16_t vgprs;
   uint8_t family;
   uint8_t target;
   uint8_t compiler;
   uint8_t requested;
   const char *reason;
};

#define AGC_ISA_ERROR_TEXT_SIZE 192

int agc_isa_error_format(const struct agc_isa_error *error, char *text, size_t size);
bool agc_isa_error_is_unsupported(const struct agc_isa_error *error);
bool agc_isa_error_pc(const struct agc_isa_error *error, uint32_t *pc);

_Static_assert((int)AGC_ISA_TARGET_GFX1013 == (int)AGC_SHADER_ISA_GFX1013, "target order");
_Static_assert((int)AGC_ISA_TARGET_GFX103 == (int)AGC_SHADER_ISA_GFX103, "target order");
_Static_assert((int)AGC_ISA_TARGET_GFX11 == (int)AGC_SHADER_ISA_GFX11, "target order");
_Static_assert((int)AGC_ISA_TARGET_GFX115 == (int)AGC_SHADER_ISA_GFX115, "target order");
_Static_assert((int)AGC_ISA_TARGET_GFX12 == (int)AGC_SHADER_ISA_GFX12, "target order");

static inline enum agc_isa_target
agc_isa_target_from_shader_isa(enum agc_shader_isa isa)
{
   return (enum agc_isa_target)isa;
}

struct agc_isa_mask128 {
   uint64_t lo;
   uint64_t hi;
};

static inline struct agc_isa_mask128
agc_isa_mask128_bit(unsigned index)
{
   struct agc_isa_mask128 mask = {0, 0};
   if (index < 64)
      mask.lo = UINT64_C(1) << index;
   else if (index < 128)
      mask.hi = UINT64_C(1) << (index - 64);
   return mask;
}

static inline struct agc_isa_mask128
agc_isa_mask128_or(struct agc_isa_mask128 a, struct agc_isa_mask128 b)
{
   struct agc_isa_mask128 mask = {a.lo | b.lo, a.hi | b.hi};
   return mask;
}

static inline struct agc_isa_mask128
agc_isa_mask128_and(struct agc_isa_mask128 a, struct agc_isa_mask128 b)
{
   struct agc_isa_mask128 mask = {a.lo & b.lo, a.hi & b.hi};
   return mask;
}

static inline struct agc_isa_mask128
agc_isa_mask128_andnot(struct agc_isa_mask128 a, struct agc_isa_mask128 b)
{
   struct agc_isa_mask128 mask = {a.lo & ~b.lo, a.hi & ~b.hi};
   return mask;
}

static inline bool
agc_isa_mask128_test(struct agc_isa_mask128 mask, unsigned index)
{
   if (index < 64)
      return (mask.lo >> index) & 1;
   if (index < 128)
      return (mask.hi >> (index - 64)) & 1;
   return false;
}

static inline bool
agc_isa_mask128_is_zero(struct agc_isa_mask128 mask)
{
   return mask.lo == 0 && mask.hi == 0;
}

static inline bool
agc_isa_mask128_equal(struct agc_isa_mask128 a, struct agc_isa_mask128 b)
{
   return a.lo == b.lo && a.hi == b.hi;
}

static inline struct agc_isa_mask128
agc_isa_mask128_range(unsigned first, unsigned count)
{
   struct agc_isa_mask128 mask = {0, 0};
   for (unsigned index = first; index < first + count && index < 128; index++)
      mask = agc_isa_mask128_or(mask, agc_isa_mask128_bit(index));
   return mask;
}

static inline struct agc_isa_mask128
agc_isa_mask128_low(struct agc_isa_mask128 mask, unsigned bits)
{
   if (bits >= 128)
      return mask;
   if (bits >= 64) {
      mask.hi &= bits == 64 ? 0 : (UINT64_C(1) << (bits - 64)) - 1;
      return mask;
   }
   mask.hi = 0;
   mask.lo &= (UINT64_C(1) << bits) - 1;
   return mask;
}

static inline unsigned
agc_isa_mask128_length(struct agc_isa_mask128 mask)
{
   unsigned length = 0;
   uint64_t word = mask.hi ? mask.hi : mask.lo;
   for (; word; word >>= 1)
      length++;
   return mask.hi ? length + 64 : length;
}

#define AGC_ISA_MAX_BLOCK_SUCCESSORS 2

struct agc_isa_block {
   uint32_t id;
   uint32_t start_pc;
   uint32_t first;
   uint32_t end;
   uint32_t successors[AGC_ISA_MAX_BLOCK_SUCCESSORS];
   uint32_t successor_count;
   const uint32_t *predecessors;
   uint32_t predecessor_count;
   bool reachable;
};

enum agc_isa_resource_kind {
   AGC_ISA_RESOURCE_BUFFER_DESCRIPTOR,
   AGC_ISA_RESOURCE_IMAGE_DESCRIPTOR,
   AGC_ISA_RESOURCE_SAMPLER_DESCRIPTOR,
   AGC_ISA_RESOURCE_SCALAR_MEMORY_PAIR,
};

struct agc_isa_resource_binding {
   uint8_t kind;
   uint8_t base_sgpr;
   uint8_t dword_count;
   bool writable;
};

enum agc_isa_descriptor_access {
   AGC_ISA_ACCESS_SAMPLED,
   AGC_ISA_ACCESS_LOADED,
   AGC_ISA_ACCESS_WRITTEN,
   AGC_ISA_ACCESS_QUERIED,
};

enum agc_isa_scalar_offset_kind {
   AGC_ISA_SCALAR_OFFSET_BOUNDED,
   AGC_ISA_SCALAR_OFFSET_USER_SGPR,
   AGC_ISA_SCALAR_OFFSET_UNKNOWN,
};

struct agc_isa_scalar_offset {
   uint8_t kind;
   uint8_t sgpr;
   uint32_t bound;
};

struct agc_isa_buffer_access {
   bool scalar;
   bool indexed;
   bool vector_offset;
   uint32_t immediate_offset;
   struct agc_isa_scalar_offset scalar_offset;
};

enum agc_isa_address_form {
   AGC_ISA_ADDRESS_POINTER,
   AGC_ISA_ADDRESS_BUFFER,
};

#define AGC_ISA_MAX_TABLE_LINKS 16
#define AGC_ISA_MAX_TABLE_CHOICES 256

struct agc_isa_table_link {
   int64_t offset;
   uint8_t form;
};

struct agc_isa_descriptor_table {
   uint8_t root;
   uint8_t root_form;
   const struct agc_isa_table_link *links;
   uint32_t link_count;
   int64_t offset;
};

enum agc_isa_descriptor_source_kind {
   AGC_ISA_SOURCE_USER_SGPRS,
   AGC_ISA_SOURCE_TABLE,
   AGC_ISA_SOURCE_TABLE_CHOICES,
   AGC_ISA_SOURCE_DYNAMIC,
};

struct agc_isa_descriptor_source {
   uint8_t kind;
   uint8_t first;
   const struct agc_isa_descriptor_table *tables;
   uint32_t table_count;
};

struct agc_isa_descriptor_use {
   uint32_t pc;
   uint8_t kind;
   uint8_t base_sgpr;
   uint8_t dword_count;
   uint8_t access;
   struct agc_isa_descriptor_source source;
   bool has_buffer_access;
   struct agc_isa_buffer_access buffer_access;
};

struct agc_isa_buffer_fill {
   uint8_t buffer;
   uint8_t value;
};

struct agc_isa_export_info {
   uint32_t color_mask;
   uint8_t depth_mask;
   uint8_t position_mask;
   uint32_t param_mask;
   bool primitive;
   bool compressed;
   bool valid_mask;
};

static inline uint32_t
agc_isa_export_info_param_count(const struct agc_isa_export_info *exports)
{
   uint32_t count = 0;
   for (uint32_t mask = exports->param_mask; mask; mask >>= 1)
      count++;
   return count;
}

struct agc_isa_metadata {
   bool has_max_sgpr;
   uint8_t max_sgpr;
   bool has_max_vgpr;
   uint8_t max_vgpr;
   uint16_t sgpr_count;
   uint16_t vgpr_count;
   struct agc_isa_mask128 live_in_scalar;
   struct agc_isa_mask128 live_in_vector;
   uint8_t user_sgpr_count;
   bool uses_scratch;
   bool uses_pc_relative;
   bool uses_relative_indexing;
   bool uses_relative_sgprs;
   struct agc_isa_resource_binding *bindings;
   size_t binding_count;
   struct agc_isa_descriptor_use *descriptor_uses;
   size_t descriptor_use_count;
   struct agc_isa_descriptor_use *pointer_uses;
   size_t pointer_use_count;
   bool has_buffer_fill;
   struct agc_isa_buffer_fill buffer_fill;
   struct agc_isa_export_info exports;
   bool has_user_data_tail_sgpr;
   uint8_t user_data_tail_sgpr;
   uint8_t stage;
   bool uses_wave_id;
   void *storage;
};

void agc_isa_metadata_finish(struct agc_isa_metadata *metadata);

enum agc_isa_jump_kind {
   AGC_ISA_JUMP_STATIC,
   AGC_ISA_JUMP_RETURN,
   AGC_ISA_JUMP_EXTERNAL,
   AGC_ISA_JUMP_OPEN,
};

struct agc_isa_jump_target {
   uint8_t kind;
   uint32_t pc;
};

static inline struct agc_isa_jump_target
agc_isa_jump_target_make(enum agc_isa_jump_kind kind, uint32_t pc)
{
   struct agc_isa_jump_target target = {(uint8_t)kind, kind == AGC_ISA_JUMP_STATIC ? pc : 0};
   return target;
}

static inline bool
agc_isa_jump_target_equal(struct agc_isa_jump_target a, struct agc_isa_jump_target b)
{
   return a.kind == b.kind && a.pc == b.pc;
}

struct agc_isa_jump {
   uint32_t pc;
   struct agc_isa_jump_target target;
};

struct agc_isa_external_target {
   uint32_t pc;
   bool call;
   bool has_sgpr;
   uint8_t sgpr;
};

struct agc_isa_computed_flow {
   struct agc_isa_jump *jumps;
   size_t jump_count;
   uint32_t *entries;
   size_t entry_count;
   struct agc_isa_external_target *external;
   size_t external_count;
};

void agc_isa_computed_flow_finish(struct agc_isa_computed_flow *flow);
bool agc_isa_computed_flow_equal(const struct agc_isa_computed_flow *a, const struct agc_isa_computed_flow *b);
bool agc_isa_computed_flow_has_dynamic_jumps(const struct agc_isa_computed_flow *flow);
const struct agc_isa_jump_target *agc_isa_computed_flow_jump(const struct agc_isa_computed_flow *flow,
                                                             uint32_t pc);
bool agc_isa_computed_flow_has_entry(const struct agc_isa_computed_flow *flow, uint32_t pc);

struct agc_isa_linked_region {
   uint64_t address;
   uint32_t start;
   uint32_t bytes;
};

struct agc_isa_code_layout {
   bool has_base;
   uint64_t base;
   uint32_t own_bytes;
   struct agc_isa_linked_region *regions;
   size_t region_count;
   bool has_second_half;
   uint64_t second_half;
};

void agc_isa_code_layout_finish(struct agc_isa_code_layout *layout);
uint64_t agc_isa_code_layout_guest_address(const struct agc_isa_code_layout *layout, uint32_t pc);
bool agc_isa_code_layout_pc_of(const struct agc_isa_code_layout *layout, uint64_t address, uint32_t *pc);

struct agc_isa_ir {
   struct agc_isa_inst *instructions;
   size_t instruction_count;
   size_t instruction_capacity;
   size_t code_words;
   struct agc_isa_block *blocks;
   size_t block_count;
   struct agc_isa_metadata metadata;
   struct agc_isa_computed_flow flow;
   struct agc_isa_code_layout layout;
};

void agc_isa_ir_init(struct agc_isa_ir *ir);
void agc_isa_ir_finish(struct agc_isa_ir *ir);
void agc_isa_ir_clear(struct agc_isa_ir *ir);
bool agc_isa_ir_reserve(struct agc_isa_ir *ir, size_t capacity);
bool agc_isa_ir_instruction_at(const struct agc_isa_ir *ir, uint32_t pc, size_t *index);
void agc_isa_ir_free_analysis(struct agc_isa_ir *ir);

struct agc_isa_decode_scope {
   const uint32_t *entries;
   size_t entry_count;
   bool through_jumps;
};

struct agc_isa_host_tables {
   enum agc_isa_target target;
   uint16_t opcodes[AGC_ISA_FAMILY_COUNT][AGC_ISA_OPCODE_LIMIT];
   uint16_t vopc[256];
   uint16_t signed_dots[8];
   uint8_t signed_dot_count;
   uint8_t formats[256];
};

#define AGC_ISA_HOST_NO_OPCODE 0xffff
#define AGC_ISA_HOST_NO_FORMAT 0xff

void agc_isa_host_tables_init(struct agc_isa_host_tables *tables, enum agc_isa_target target);
bool agc_isa_host_guest_opcode(const struct agc_isa_host_tables *tables, enum agc_isa_family family,
                               uint16_t host, uint16_t *guest);
bool agc_isa_host_guest_format(const struct agc_isa_host_tables *tables, uint8_t host, uint8_t *guest);
bool agc_isa_host_is_signed_dot(const struct agc_isa_host_tables *tables, uint16_t host);

bool agc_isa_family_of(uint32_t word, enum agc_isa_family *family);
bool agc_isa_decode_at(const uint32_t *words, size_t word_count, size_t index, struct agc_isa_inst *inst,
                       struct agc_isa_error *error);
bool agc_isa_decode(const uint32_t *words, size_t word_count, struct agc_isa_ir *ir,
                    struct agc_isa_error *error);
bool agc_isa_decode_scoped(const uint32_t *words, size_t word_count,
                           const struct agc_isa_decode_scope *scope, struct agc_isa_ir *ir,
                           struct agc_isa_error *error);
bool agc_isa_host_decode(const struct agc_isa_host_tables *tables, const uint32_t *words, size_t word_count,
                         struct agc_isa_ir *ir, struct agc_isa_error *error);

bool agc_isa_cfg_build(struct agc_isa_ir *ir, struct agc_isa_error *error);

bool agc_isa_computed_decode_scope(const struct agc_isa_computed_flow *flow, uint32_t **entries,
                                   size_t *entry_count, bool *through_jumps);
bool agc_isa_computed_apply(struct agc_isa_ir *ir, const struct agc_isa_computed_flow *flow,
                            struct agc_isa_error *error);
bool agc_isa_computed_analyze(const struct agc_isa_ir *ir, bool has_base, uint64_t base, size_t code_words,
                              struct agc_isa_computed_flow *flow, struct agc_isa_error *error);

bool agc_isa_analyze_linked(enum agc_isa_target compiler, const uint32_t *words, size_t word_count,
                            const struct agc_shader_options *options, const struct agc_shader_interface *iface,
                            const struct agc_shader_linkage *linkage, struct agc_isa_ir *ir,
                            struct agc_isa_error *error);
bool agc_isa_analyze(enum agc_isa_target compiler, const uint32_t *words, size_t word_count,
                     const struct agc_shader_options *options, const struct agc_shader_interface *iface,
                     struct agc_isa_ir *ir, struct agc_isa_error *error);

#endif
