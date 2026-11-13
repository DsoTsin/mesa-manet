#ifndef AGC_LOWER_H
#define AGC_LOWER_H

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
#include <vector>

#include "aco_builder.h"
#include "aco_ir.h"

extern "C" {
#endif

#include "agc_shader_artifact.h"

#include "agc_isa.h"
#include "agc_isa_descriptors.h"
#include "agc_isa_regalloc.h"

#define AGC_ISA_S_CODE_END 0xbf9f0000u
#define AGC_ISA_PREFETCH_PADDING_LINES 3

#define AGC_ISA_LOCAL_LABEL_BASE 0xf0000000u
#define AGC_ISA_PROLOGUE_LABEL 0xffffffe0u
#define AGC_ISA_DISPATCH_LABEL 0xfffffff0u
#define AGC_ISA_MISS_LABEL 0xfffffff4u
#define AGC_ISA_EPILOGUE_LABEL 0xfffffff8u
#define AGC_ISA_EPILOGUE_END_LABEL 0xfffffffcu

#define AGC_ISA_DISPATCH_SGPRS 4
#define AGC_ISA_LINKED_DISPATCH_SGPRS 8
#define AGC_ISA_GFX10_VMCNT_MAX 63
#define AGC_ISA_GFX10_EXPCNT_MAX 7
#define AGC_ISA_GFX10_LGKMCNT_MAX 63
#define AGC_ISA_GFX10_VSCNT_MAX 63
#define AGC_ISA_SPI_SHADER_32_R 1u
#define AGC_ISA_GDS_BYTES 0x10000u
#define AGC_ISA_IMAGE_REPORTS_KNOWN_BUCKETS (1u << 13)
#define AGC_ISA_IMAGE_REPORTS_KNOWN_BUCKET_BYTES 8u
#define AGC_ISA_IMAGE_REPORTS_KNOWN_BYTES \
   (AGC_ISA_IMAGE_REPORTS_KNOWN_BUCKETS * AGC_ISA_IMAGE_REPORTS_KNOWN_BUCKET_BYTES)
#define AGC_ISA_IMAGE_REPORTS_REPORT_SLOTS (1u << 12)
#define AGC_ISA_IMAGE_REPORTS_REPORT_CHECK_OFFSET 32u
#define AGC_ISA_IMAGE_REPORTS_REPORTS_OFFSET AGC_ISA_IMAGE_REPORTS_KNOWN_BYTES
#define AGC_ISA_IMAGE_REPORTS_CHECK 0x5a3ca5c3u
#define AGC_ISA_IMAGE_REPORTS_SPREAD 0x9e3779b9u
#define AGC_ISA_IMAGE_REPORTS_STORE_CHECK 0xc35aa53cu
#define AGC_ISA_IMAGE_REPORTS_STORE_SALT 0x7f4a7c15u
#define AGC_ISA_IMAGE_REPORTS_STORE_SLOT ((AGC_ISA_IMAGE_REPORTS_REPORT_SLOTS >> 1) << 6)

static inline void
agc_isa_waitcnt_decode_gfx10(uint16_t simm16, uint8_t *vm, uint8_t *exp, uint8_t *lgkm)
{
   *vm = (uint8_t)((simm16 & 0xf) | ((simm16 >> 10) & 0x30));
   *exp = (uint8_t)((simm16 >> 4) & 0x7);
   *lgkm = (uint8_t)((simm16 >> 8) & 0x3f);
}

static inline uint16_t
agc_isa_waitcnt_encode_gfx10(uint8_t vm, uint8_t exp, uint8_t lgkm)
{
   const uint16_t v = vm & 0x3f;
   return (uint16_t)((v & 0xf) | ((v >> 4) << 14) | ((exp & 0x7) << 4) | ((lgkm & 0x3f) << 8));
}

static inline void
agc_isa_waitcnt_decode_gfx11(uint16_t simm16, uint8_t *vm, uint8_t *exp, uint8_t *lgkm)
{
   *vm = (uint8_t)((simm16 >> 10) & 0x3f);
   *exp = (uint8_t)(simm16 & 0x7);
   *lgkm = (uint8_t)((simm16 >> 4) & 0x3f);
}

static inline uint16_t
agc_isa_waitcnt_encode_gfx11(uint8_t vm, uint8_t exp, uint8_t lgkm)
{
   return (uint16_t)(((vm & 0x3f) << 10) | ((lgkm & 0x3f) << 4) | (exp & 0x7));
}

bool agc_isa_pixel_is_vop3_interp(uint16_t opcode);
bool agc_isa_pixel_col_format_for(bool has_col_format, uint32_t col_format, bool has_z_format, uint32_t z_format,
                                  bool null_exports, uint32_t *format);

struct agc_isa_pixel_interface {
   bool has_prim_mask;
   uint8_t prim_mask;
   bool has_col_format;
   uint32_t col_format;
   bool has_z_format;
   uint32_t z_format;
   bool dual_source;
};

struct agc_isa_lowerer {
   struct agc_shader_options options;
   enum agc_shader_precision precision;
   bool has_guest_code_address;
   uint64_t guest_code_address;
   bool has_descriptor_lut;
   uint64_t descriptor_lut;
   bool preconvert_user_descriptors;
   bool native_dcc;
   struct agc_isa_pixel_interface pixel;
   bool has_attribute_ring;
   struct agc_shader_attribute_ring attribute_ring;
   bool has_gds_buffer;
   uint64_t gds_buffer;
   bool has_mip_stats_table;
   uint64_t mip_stats_table;
   bool has_image_reports;
   uint64_t image_reports;
   bool hull;
   enum agc_shader_es_input es_input;
};

void agc_isa_lowerer_init(struct agc_isa_lowerer *lowerer, const struct agc_shader_options *options,
                          enum agc_shader_precision precision);
void agc_isa_lowerer_with_pixel_interface(struct agc_isa_lowerer *lowerer, bool has_prim_mask, uint8_t prim_mask,
                                          bool has_col_format, uint32_t col_format, bool has_z_format,
                                          uint32_t z_format, bool dual_source);
void agc_isa_lowerer_with_interface(struct agc_isa_lowerer *lowerer, const struct agc_isa_ir *ir,
                                    const struct agc_shader_interface *iface);
bool agc_isa_lowering_error_instruction(struct agc_isa_error *error, const struct agc_isa_inst *inst,
                                        enum agc_isa_target target, const char *reason);
unsigned agc_isa_lowering_source_operands(struct agc_isa_inst *inst, struct agc_isa_operand *operands[3]);

#define AGC_ISA_HWREG_ID_MASK 0x3f
#define AGC_ISA_HWREG_FLAT_SCR_LO 20
#define AGC_ISA_HWREG_FLAT_SCR_HI 21
#define AGC_ISA_HWREG_POPS_PACKER 25
#define AGC_ISA_MSG_ORDERED_PS_DONE 7
#define AGC_ISA_NO_SLOT 0xff
#define AGC_ISA_EXP_MRT0 0
#define AGC_ISA_EXP_MRT1 1
#define AGC_ISA_EXP_MRTZ 8
#define AGC_ISA_EXP_POS_FIRST 12
#define AGC_ISA_EXP_POS_LAST 16
#define AGC_ISA_EXP_PARAM_FIRST 32
#define AGC_ISA_BVH_SCRATCH_SGPRS 24
#define AGC_ISA_BVH8_SCRATCH_SGPRS 24
#define AGC_ISA_IMAGE_REPORT_SGPRS 6
#define AGC_ISA_HOISTED_CHANNELS 32
#define AGC_ISA_WAIT_QUEUE_LIMIT 63
#define AGC_ISA_SPILL_MAX_BLOCKS 16

struct agc_isa_pc_set {
   uint32_t *items;
   size_t count;
   size_t capacity;
};

enum agc_isa_wait_vm_type {
   AGC_ISA_WAIT_VM_LOAD,
   AGC_ISA_WAIT_VM_SAMPLE,
   AGC_ISA_WAIT_VM_BVH
};

enum agc_isa_wait_lgkm_type {
   AGC_ISA_WAIT_LGKM_DS,
   AGC_ISA_WAIT_LGKM_KM
};

struct agc_isa_mode_state {
   bool dynamic;
   uint8_t mode;
};

enum agc_isa_mode_restore_kind {
   AGC_ISA_MODE_RESTORE_UNCHANGED,
   AGC_ISA_MODE_RESTORE_DENORM,
   AGC_ISA_MODE_RESTORE_SAVED
};

struct agc_isa_mode_restore {
   uint8_t kind;
   uint16_t denorm;
   uint8_t sgpr;
};

struct agc_isa_kept {
   bool lane;
   uint8_t index;
};

struct agc_isa_descriptor_slots {
   uint8_t table;
   uint8_t vsharp;
   uint8_t tsharp;
   uint8_t ssharp;
   uint64_t address;
   bool native_dcc;
};

struct agc_isa_descriptor_cache {
   bool has_vsharp;
   uint8_t vsharp;
   bool has_tsharp;
   uint8_t tsharp;
   bool tsharp_r128;
   bool tsharp_stored;
   bool has_ssharp;
   uint8_t ssharp;
};

struct agc_isa_hoisted_channel {
   uint8_t attr;
   uint8_t chan;
   uint8_t vgpr;
};

struct agc_isa_pixel_plan {
   bool has_prim_mask;
   uint8_t prim_mask;
   struct agc_isa_hoisted_channel hoisted[AGC_ISA_HOISTED_CHANNELS];
   unsigned hoisted_count;
   struct agc_isa_pc_set from_prologue;
   uint8_t null_target;
   bool has_dual_source;
   uint8_t dual_source;
};

struct agc_isa_ngg_ring {
   struct agc_isa_kept descriptor;
   struct agc_isa_kept offset;
   uint8_t vertex;
   uint32_t words[4];
};

struct agc_isa_ngg_deferred {
   uint8_t target;
   uint8_t en;
   uint8_t data;
   struct agc_isa_kept exec;
};

struct agc_isa_ngg_plan {
   bool has_ring;
   struct agc_isa_ngg_ring ring;
   bool has_deferred;
   struct agc_isa_ngg_deferred deferred;
   uint8_t persistent_lanes;
};

struct agc_isa_spill_block {
   uint8_t base;
   uint8_t size;
   uint8_t first_lane;
};

struct agc_isa_spill_lanes {
   uint8_t vgpr;
   uint8_t persistent;
   struct agc_isa_spill_block saved[AGC_ISA_SPILL_MAX_BLOCKS];
   unsigned saved_count;
};

struct agc_isa_wave_offset_uses {
   struct agc_isa_pc_set private_pcs;
   bool computed;
};

struct agc_isa_image_report {
   uint64_t table;
   uint8_t source;
   uint8_t dwords;
   uint8_t s;
   uint8_t v;
   bool stored;
};

enum agc_isa_float_atomic_kind {
   AGC_ISA_FLOAT_ATOMIC_COMPARE_SWAP,
   AGC_ISA_FLOAT_ATOMIC_MIN_MAX
};

struct agc_isa_float_atomic {
   uint8_t kind;
   bool wide;
   bool max;
};

static inline struct agc_isa_inst
agc_isa_inst_derive(const struct agc_isa_inst *origin, enum agc_isa_family family, uint16_t opcode, uint64_t flags)
{
   struct agc_isa_inst inst;
   memset(&inst, 0, sizeof(inst));
   inst.pc = origin->pc;
   inst.family = (uint8_t)family;
   inst.fields = (uint8_t)family;
   inst.opcode = opcode;
   inst.flags = flags;
   return inst;
}

static inline struct agc_isa_inst
agc_isa_inst_retarget(const struct agc_isa_inst *inst, uint16_t opcode)
{
   struct agc_isa_inst out = *inst;
   out.opcode = opcode;
   out.word_count = 0;
   return out;
}

static inline struct agc_isa_inst
agc_isa_tmpl(enum agc_isa_family family, uint16_t opcode)
{
   struct agc_isa_inst inst;
   memset(&inst, 0, sizeof(inst));
   inst.family = (uint8_t)family;
   inst.fields = (uint8_t)family;
   inst.opcode = opcode;
   return inst;
}

static inline struct agc_isa_inst
agc_isa_tmpl_sop1(uint16_t opcode, struct agc_isa_operand sdst, struct agc_isa_operand src0)
{
   struct agc_isa_inst inst = agc_isa_tmpl(AGC_ISA_SOP1, opcode);
   inst.sop1.sdst = sdst;
   inst.sop1.src0 = src0;
   return inst;
}

static inline struct agc_isa_inst
agc_isa_tmpl_vop1(uint16_t opcode, struct agc_isa_operand vdst, struct agc_isa_operand src0)
{
   struct agc_isa_inst inst = agc_isa_tmpl(AGC_ISA_VOP1, opcode);
   inst.vop1.vdst = vdst;
   inst.vop1.src0 = src0;
   return inst;
}

static inline struct agc_isa_inst
agc_isa_tmpl_vop2(uint16_t opcode, struct agc_isa_operand vdst, struct agc_isa_operand src0,
                  struct agc_isa_operand vsrc1)
{
   struct agc_isa_inst inst = agc_isa_tmpl(AGC_ISA_VOP2, opcode);
   inst.vop2.vdst = vdst;
   inst.vop2.src0 = src0;
   inst.vop2.vsrc1 = vsrc1;
   return inst;
}

static inline struct agc_isa_vop3
agc_isa_vop3_new(struct agc_isa_operand vdst, struct agc_isa_operand src0, struct agc_isa_operand src1,
                 struct agc_isa_operand src2)
{
   struct agc_isa_vop3 v;
   memset(&v, 0, sizeof(v));
   v.vdst = vdst;
   v.src[0] = src0;
   v.src[1] = src1;
   v.src[2] = src2;
   return v;
}

static inline struct agc_isa_inst
agc_isa_tmpl_vop3v(uint16_t opcode, const struct agc_isa_vop3 *v)
{
   struct agc_isa_inst inst = agc_isa_tmpl(AGC_ISA_VOP3, opcode);
   inst.vop3 = *v;
   return inst;
}

static inline struct agc_isa_inst
agc_isa_lowering_vop3_inst(const struct agc_isa_inst *origin, uint16_t opcode, uint64_t flags,
                           const struct agc_isa_vop3 *v)
{
   struct agc_isa_inst inst = agc_isa_inst_derive(origin, AGC_ISA_VOP3, opcode, flags);
   inst.vop3 = *v;
   return inst;
}

void agc_isa_pc_set_finish(struct agc_isa_pc_set *set);
bool agc_isa_pc_set_insert(struct agc_isa_pc_set *set, uint32_t pc);
bool agc_isa_pc_set_contains(const struct agc_isa_pc_set *set, uint32_t pc);
bool agc_isa_lowering_wave_offset_find(const struct agc_isa_ir *ir, bool wave64, uint8_t sgpr,
                                       enum agc_isa_target target, struct agc_isa_wave_offset_uses *uses,
                                       struct agc_isa_error *error);
void agc_isa_descriptor_cache_invalidate(struct agc_isa_descriptor_cache *cache, const struct agc_isa_inst *inst,
                                         bool wave64);
bool agc_isa_lowering_pure_user_descriptors(const struct agc_isa_ir *ir, bool wave64,
                                            struct agc_isa_resource_binding **bindings, size_t *count);
bool agc_isa_lowering_interpolated(const struct agc_isa_inst *inst, uint8_t *attr, uint8_t *chan);
void agc_isa_lowering_report_counts(uint64_t counts[2]);
bool agc_isa_pixel_plan_init(struct agc_isa_pixel_plan *plan, const struct agc_isa_ir *ir,
                             struct agc_isa_regalloc *regs, bool wave64,
                             const struct agc_isa_pixel_interface *iface, struct agc_isa_error *error);
void agc_isa_pixel_plan_finish(struct agc_isa_pixel_plan *plan);
bool agc_isa_pixel_plan_hoisted(const struct agc_isa_pixel_plan *plan, uint32_t pc, uint8_t attr, uint8_t chan,
                                uint8_t *vgpr);
bool agc_isa_lowering_pops_check(const struct agc_isa_ir *ir, enum agc_isa_target target,
                                 enum agc_shader_stage stage, struct agc_isa_error *error);
bool agc_isa_lowering_pops_without_exiting_wave(const struct agc_isa_inst *inst, struct agc_isa_inst *rewritten);
bool agc_isa_ngg_plan_init(struct agc_isa_ngg_plan *plan, const struct agc_isa_ir *ir,
                           struct agc_isa_regalloc *regs, bool wave64, bool has_ring,
                           struct agc_shader_attribute_ring ring, bool spill, struct agc_isa_error *error);

extern const struct agc_shader_register_relocation agc_isa_ngg_program_registers[AGC_SHADER_NGG_REGISTER_COUNT];

uint32_t agc_isa_ngg_spi_vs_out_config(uint32_t param_count);
void agc_isa_ngg_ring_descriptor(struct agc_shader_attribute_ring ring, uint32_t param_count, uint32_t words[4]);

struct agc_isa_ring_literals {
   bool has_sgpr;
   uint8_t sgpr;
   uint32_t words[4];
};

struct agc_isa_encoded {
   uint32_t *code;
   size_t code_count;
   size_t program_words;
   struct agc_shader_binding *preconverted;
   size_t preconverted_count;
   bool has_attribute_ring;
   struct agc_isa_ring_literals attribute_ring;
};

void agc_isa_encoded_init(struct agc_isa_encoded *encoded);
void agc_isa_encoded_finish(struct agc_isa_encoded *encoded);
bool agc_isa_lower_encode(const struct agc_isa_lowerer *lowerer, const struct agc_isa_ir *ir,
                          struct agc_isa_regalloc *regs, struct agc_isa_encoded *encoded,
                          struct agc_isa_error *error);
bool agc_isa_lower_listing(const struct agc_isa_lowerer *lowerer, const struct agc_isa_ir *ir,
                           struct agc_isa_regalloc *regs, char **listing, struct agc_isa_error *error);

#ifdef __cplusplus
}

namespace agc {


using namespace aco;

struct lower_error {
   agc_isa_error error;
};

struct encode_error {
   agc_isa_error error;
};

struct segment {
   uint32_t label = 0;
   bool has_label = false;
   bool dynamic_entry = false;
   bool has_target = false;
   bool branch = false;
   bool falls = true;
   bool has_reject = false;
   uint32_t target = 0;
   uint32_t pc = 0;
   agc_isa_error reject;
   std::vector<aco_ptr<Instruction>> instructions;
};

struct lowered_block {
   size_t id;
   size_t first;
   size_t count;
   bool split;
};

struct wait_tracker {
   uint8_t vm[AGC_ISA_WAIT_QUEUE_LIMIT + 1];
   uint8_t vm_count;
   bool vm_prefix[3];
   uint8_t lgkm[AGC_ISA_WAIT_QUEUE_LIMIT + 1];
   uint8_t lgkm_count;
   bool lgkm_prefix[2];
};

struct lower_ctx {
   const agc_isa_lowerer *lowerer;
   const agc_isa_ir *ir;
   agc_isa_regalloc *regs;
   Program *program;

   enum agc_isa_target target;
   enum agc_shader_stage stage;
   enum agc_shader_precision precision;
   enum agc_shader_ray_unit ray_unit;
   bool wave64;
   bool gfx11_plus;
   bool gfx12;

   agc_isa_mode_state mode;
   agc_isa_code_layout layout;
   bool has_guest_base;
   uint64_t guest_base;
   bool has_dispatch;
   uint8_t dispatch;
   const agc_isa_pc_set *private_pcs;
   bool private_buffer;
   bool has_descriptors;
   agc_isa_descriptor_slots descriptors;
   const agc_isa_resource_binding *preconverted;
   size_t preconverted_count;
   const agc_isa_pixel_plan *pixel;
   bool has_ngg;
   agc_isa_ngg_plan ngg;
   bool has_wave_index;
   agc_isa_kept wave_index;
   bool has_gds;
   uint64_t gds;
   bool has_mip_stats;
   uint64_t mip_stats;
   bool has_image_reports;
   uint64_t image_reports;
   bool has_spill;
   uint8_t spill_vgpr;
   uint8_t spill_persistent;

   wait_tracker waits;
   agc_isa_descriptor_cache descriptor_cache;
   bool has_pending_dual_source;
   agc_isa_export pending_dual_source;
   bool epilogue;
   agc_isa_spill_lanes spill;

   Builder bld;
   std::vector<aco_ptr<Instruction>> pending;
   std::vector<segment> staged;
   bool split;
   bool has_next_label;
   uint32_t next_label;
   uint32_t pc;

   std::vector<segment> segments;
   std::vector<lowered_block> blocks;

   lower_ctx(Program *p) : program(p), bld(p, &pending) {}
};

struct inst_meta {
   bool has_label = false;
   uint32_t label = 0;
   bool ends = false;
   bool falls = true;
   bool has_target = false;
   uint32_t target = 0;
   bool branch = false;
   uint32_t pc = 0;
};

struct sdwa_core {
   enum { guest, fused, unfused } kind;
   uint16_t vop3;
   aco_opcode op;
   bool legacy;
};

static inline Operand
sop(unsigned reg, unsigned dwords = 1)
{
   return Operand(PhysReg{reg}, RegClass(RegType::sgpr, dwords));
}

static inline Definition
sdef(unsigned reg, unsigned dwords = 1)
{
   return Definition(PhysReg{reg}, RegClass(RegType::sgpr, dwords));
}

static inline Operand
vop(unsigned reg, unsigned dwords = 1)
{
   return Operand(PhysReg{256u + reg}, RegClass(RegType::vgpr, dwords));
}

static inline Definition
vdef(unsigned reg, unsigned dwords = 1)
{
   return Definition(PhysReg{256u + reg}, RegClass(RegType::vgpr, dwords));
}

static inline Definition
scc_def()
{
   return Definition(scc, s1);
}

static inline Operand
scc_op()
{
   return Operand(scc, s1);
}

static inline Operand
imm(uint32_t value)
{
   Operand op = Operand::c32(value);
   if (value == 0x3e22f983u)
      op.setFixed(PhysReg{248});
   return op;
}

static inline Operand
lit(uint32_t value)
{
   return Operand::literal32(value);
}

static inline void
modifiers(Instruction *instr, uint8_t abs, uint8_t neg, bool clamp, uint8_t omod)
{
   VALU_instruction &valu = instr->valu();
   valu.abs = abs;
   valu.neg = neg;
   valu.clamp = clamp;
   valu.omod = omod;
}

[[noreturn]] void fail(const lower_ctx &ctx, const agc_isa_inst *inst, const char *reason);
[[noreturn]] void fail_plain();
[[noreturn]] void fail_memory();
[[noreturn]] void fail_error(const agc_isa_error &error);

Operand src(const agc_isa_inst *inst, agc_isa_operand op, unsigned dwords = 1);
Definition dst(const agc_isa_inst *inst, agc_isa_operand op, unsigned dwords = 1);
Operand mask_op(const lower_ctx &ctx, PhysReg reg);
Definition mask_def(const lower_ctx &ctx, PhysReg reg);


void note_reject(lower_ctx &ctx, const agc_isa_error &error);
void flush(lower_ctx &ctx);
void push(lower_ctx &ctx, const agc_isa_inst &lowered);
void push_retarget(lower_ctx &ctx, const agc_isa_inst *inst, uint16_t opcode);
void label(lower_ctx &ctx, uint32_t label);
void label_next(lower_ctx &ctx, uint32_t label);
void branch(lower_ctx &ctx, aco_opcode op, uint32_t label);
void endpgm(lower_ctx &ctx);
void guest_endpgm(lower_ctx &ctx);
void setpc(lower_ctx &ctx, unsigned sgpr);
void smem_load(lower_ctx &ctx, unsigned dwords, Definition data, unsigned sbase, Operand soffset,
               uint32_t offset = 0);
void global_access(lower_ctx &ctx, const agc_isa_inst *origin, uint16_t opcode, unsigned vaddr, agc_isa_operand saddr,
                   unsigned vdata, int32_t offset, unsigned vdst = 0, bool glc = false, bool dlc = false);

static inline void
global_access(lower_ctx &ctx, const agc_isa_inst *origin, uint16_t opcode, unsigned vaddr, unsigned saddr, unsigned vdata,
              int32_t offset, unsigned vdst = 0, bool glc = false, bool dlc = false)
{
   global_access(ctx, origin, opcode, vaddr, agc_isa_operand_sgpr((uint8_t)saddr), vdata, offset, vdst, glc, dlc);
}

void wait_tracker_init(wait_tracker &tracker, bool program_entry);
void waitcnt(lower_ctx &ctx, uint8_t vm, uint8_t exp, uint8_t lgkm);
void gfx12_waits(lower_ctx &ctx, bool has_vm, uint8_t vm, bool has_exp, uint8_t exp, bool has_lgkm, uint8_t lgkm,
                 bool has_vs, uint8_t vs);

uint8_t scratch_vgprs(lower_ctx &ctx, uint8_t count);
uint8_t transient_sgprs(lower_ctx &ctx, const agc_isa_inst *origin, uint8_t count);
uint8_t spill_block(lower_ctx &ctx, const agc_isa_inst *origin, uint8_t count);
void read_lane(lower_ctx &ctx, const agc_isa_inst *origin, uint8_t sdst, uint8_t lane);
void write_lane(lower_ctx &ctx, const agc_isa_inst *origin, Operand src, uint8_t lane);
void restore_spills(lower_ctx &ctx);

void save_scc(lower_ctx &ctx, unsigned sgpr);
void restore_scc(lower_ctx &ctx, unsigned sgpr);
void save_masks(lower_ctx &ctx, unsigned sgpr);
void restore_masks(lower_ctx &ctx, unsigned sgpr);
void exec_and(lower_ctx &ctx, Operand mask);
void exec_andn2(lower_ctx &ctx, Operand mask);
void enter_f32_flush(lower_ctx &ctx, const agc_isa_inst *origin, agc_isa_mode_restore *restore);
void leave_f32_flush(lower_ctx &ctx, agc_isa_mode_restore restore);

void lower_inst(lower_ctx &ctx, const agc_isa_inst *inst);
void lower_template(lower_ctx &ctx, const agc_isa_inst *origin, agc_isa_inst tmpl);

void lower_vector(lower_ctx &ctx, const agc_isa_inst *inst);
void sdwa_expand(lower_ctx &ctx, const agc_isa_inst *inst, uint16_t vop3_op);
void sdwa_expand_with(lower_ctx &ctx, const agc_isa_inst *inst, const sdwa_core &core);

bool lower_pc_relative(lower_ctx &ctx, const agc_isa_inst *inst);
void set_second_half(lower_ctx &ctx, const agc_isa_inst *origin, agc_isa_mask128 live_in);
void lower_branch(lower_ctx &ctx, uint32_t to);
void dispatch_stub(lower_ctx &ctx, const agc_isa_inst *origin, uint32_t code_bytes);

uint8_t converted_vsharp(lower_ctx &ctx, const agc_isa_inst *inst, uint8_t source);
uint8_t converted_tsharp(lower_ctx &ctx, const agc_isa_inst *inst, uint8_t source, bool r128, bool has_report,
                         uint64_t report, bool stored);
uint8_t converted_ssharp(lower_ctx &ctx, const agc_isa_inst *inst, uint8_t source);
void load_descriptor_table(lower_ctx &ctx);

void enable_all_lanes(lower_ctx &ctx, uint8_t saved);
void restore_lanes(lower_ctx &ctx, uint8_t saved);
void load_hoisted_parameters(lower_ctx &ctx, const agc_isa_inst *origin);
void param_load(lower_ctx &ctx, uint8_t vdst, uint8_t attr, uint8_t chan);

void lower_interp(lower_ctx &ctx, const agc_isa_inst *inst);
void lower_interp_vop3(lower_ctx &ctx, const agc_isa_inst *inst);
void wait_for_parameters(lower_ctx &ctx);
void quad_broadcast(lower_ctx &ctx, const agc_isa_inst *origin, uint8_t vdst, agc_isa_operand src, uint16_t lane);

void pops_inputs(lower_ctx &ctx, agc_isa_mask128 live_in_scalar);
void pops_packer(lower_ctx &ctx, const agc_isa_inst *inst);

void ngg_prologue(lower_ctx &ctx, const agc_isa_inst *origin);
void init_full_exec(lower_ctx &ctx);
void ring_store(lower_ctx &ctx, const agc_isa_inst *inst, const agc_isa_export *e, const agc_isa_ngg_ring *ring);
void wait_for_stores(lower_ctx &ctx);
void defer_position(lower_ctx &ctx, const agc_isa_inst *inst, const agc_isa_export *e,
                    const agc_isa_ngg_deferred *deferred);
void ngg_epilogue(lower_ctx &ctx, const agc_isa_inst *origin, const agc_isa_ngg_deferred *deferred);

void lower_gds(lower_ctx &ctx, const agc_isa_inst *inst, uint64_t buffer);
void lds_access(lower_ctx &ctx, const agc_isa_inst *origin, uint16_t opcode, uint8_t addr, uint8_t vdst, uint8_t data0,
                uint8_t data1, uint8_t offset0, uint8_t offset1);
void lds_compare_and_swap(lower_ctx &ctx, const agc_isa_inst *inst);
void lds_src2(lower_ctx &ctx, const agc_isa_inst *inst);
bool float_atomic_named(uint16_t opcode, agc_isa_float_atomic *atomic);
void memory_atomic(lower_ctx &ctx, const agc_isa_inst *origin, const agc_isa_inst *tmpl, agc_isa_float_atomic atomic);

void image_atomic(lower_ctx &ctx, const agc_isa_inst *origin, const agc_isa_inst *image);
void image_fcmpswap(lower_ctx &ctx, const agc_isa_inst *origin, const agc_isa_inst *image);
bool per_quad_derivatives(lower_ctx &ctx, const agc_isa_inst *inst, const struct agc_isa_image_op *info,
                          agc_isa_inst *fine);
bool gather_without_offset(lower_ctx &ctx, const agc_isa_inst *inst, const struct agc_isa_image_op *info,
                           agc_isa_inst *gather);
bool image_report_new(lower_ctx &ctx, uint64_t table, uint8_t source, bool r128, uint8_t sgprs, bool stored,
                      agc_isa_image_report *report);
void image_report_begin(lower_ctx &ctx, const agc_isa_image_report *report);
void image_report_finish(lower_ctx &ctx, const agc_isa_image_report *report, const agc_isa_inst *origin);
void mip_stats_count(lower_ctx &ctx, const agc_isa_inst *inst, const agc_isa_image *image, uint32_t flags,
                     uint64_t table);

bool bvh_writes_mode(const agc_isa_inst *inst);
void lower_bvh(lower_ctx &ctx, const agc_isa_inst *inst);
void lower_bvh8(lower_ctx &ctx, const agc_isa_inst *inst);
void lower_bvh_stack(lower_ctx &ctx, const agc_isa_inst *inst);

}
#endif

#endif
