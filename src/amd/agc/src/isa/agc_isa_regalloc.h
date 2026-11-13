#ifndef AGC_ISA_REGALLOC_H
#define AGC_ISA_REGALLOC_H

#include "agc_isa.h"

#define AGC_ISA_RSRC1_VGPRS_MASK 0x3fu
#define AGC_ISA_RSRC1_SGPRS_MASK (0xfu << 6)
#define AGC_ISA_RSRC1_FLOAT_MODE_SHIFT 12
#define AGC_ISA_FLOAT_MODE_DEFAULT 0xc0u
#define AGC_ISA_RSRC1_DX10_CLAMP (1u << 21)
#define AGC_ISA_RSRC1_IEEE_MODE (1u << 23)
#define AGC_ISA_RSRC1_WGP_MODE (1u << 29)
#define AGC_ISA_RSRC1_MEM_ORDERED (1u << 30)
#define AGC_ISA_RSRC1_GRAPHICS_MEM_ORDERED (1u << 25)
#ifndef AGC_ISA_RSRC2_SCRATCH_EN
#define AGC_ISA_RSRC2_SCRATCH_EN 0x1u
#endif
#ifndef AGC_ISA_RSRC2_USER_SGPR_SHIFT
#define AGC_ISA_RSRC2_USER_SGPR_SHIFT 1
#endif
#ifndef AGC_ISA_RSRC2_USER_SGPR_MASK
#define AGC_ISA_RSRC2_USER_SGPR_MASK (0x1fu << AGC_ISA_RSRC2_USER_SGPR_SHIFT)
#endif
#define AGC_ISA_RSRC2_TRAP_PRESENT (1u << 6)
#ifndef AGC_ISA_RSRC2_TGID_X_EN
#define AGC_ISA_RSRC2_TGID_X_EN (1u << 7)
#endif
#define AGC_ISA_RSRC2_TG_SIZE_EN (1u << 10)
#define AGC_ISA_RSRC2_TIDIG_SHIFT 11
#define AGC_ISA_RSRC2_TIDIG_MASK (3u << AGC_ISA_RSRC2_TIDIG_SHIFT)
#define AGC_ISA_RSRC2_LDS_SHIFT 15
#define AGC_ISA_RSRC2_LDS_MASK (0x1ffu << AGC_ISA_RSRC2_LDS_SHIFT)
#define AGC_ISA_RSRC2_GS_LDS_SHIFT 19
#define AGC_ISA_RSRC2_GS_LDS_MASK (0xffu << AGC_ISA_RSRC2_GS_LDS_SHIFT)
#define AGC_ISA_RSRC2_HS_LDS_SHIFT 18
#define AGC_ISA_RSRC2_HS_LDS_MASK (0x1ffu << AGC_ISA_RSRC2_HS_LDS_SHIFT)
#define AGC_ISA_RSRC2_PS_EXTRA_LDS_SHIFT 8
#define AGC_ISA_RSRC2_PS_EXTRA_LDS_MASK (0xffu << AGC_ISA_RSRC2_PS_EXTRA_LDS_SHIFT)
#ifndef AGC_ISA_RSRC2_USER_SGPR_MSB
#define AGC_ISA_RSRC2_USER_SGPR_MSB (1u << 27)
#endif
#define AGC_ISA_RSRC2_PS_LOAD_COLLISION_WAVEID (1u << 25)
#define AGC_ISA_RSRC2_PS_LOAD_INTRAWAVE_COLLISION (1u << 26)
#define AGC_ISA_LDS_GRANULE_BYTES 512u
#ifndef AGC_ISA_MAX_COMPUTE_USER_SGPRS
#define AGC_ISA_MAX_COMPUTE_USER_SGPRS 16
#endif
#ifndef AGC_ISA_MAX_GRAPHICS_USER_SGPRS
#define AGC_ISA_MAX_GRAPHICS_USER_SGPRS 32
#endif
#define AGC_ISA_SYSTEM_SGPR_ENABLES 4
#ifndef AGC_ISA_NGG_SYSTEM_SGPRS
#define AGC_ISA_NGG_SYSTEM_SGPRS 8
#endif

struct agc_isa_shader_config {
   uint32_t pgm_rsrc1;
   uint32_t pgm_rsrc2;
   uint16_t num_sgprs;
   uint16_t num_vgprs;
   uint32_t lds_bytes;
   uint32_t scratch_bytes_per_lane;
   uint8_t user_sgpr_count;
};

struct agc_isa_system_sgprs {
   bool has_tg_size;
   uint8_t tg_size;
   bool has_host_tg_size;
   uint8_t host_tg_size;
   uint8_t workgroup_ids[3];
   bool has_host_scratch_wave_offset;
   uint8_t host_scratch_wave_offset;
   bool has_scratch_wave_offset;
   uint8_t scratch_wave_offset;
   bool has_collision_wave;
   uint8_t collision_wave;
   bool has_intrawave_collision;
   uint8_t intrawave_collision;
};

struct agc_isa_regalloc {
   enum agc_isa_target target;
   bool wave64;
   uint16_t scratch_base;
   uint16_t vgpr_high_water;
   uint16_t sgpr_high_water;
   bool has_scratch_sgpr_base;
   uint16_t scratch_sgpr_base;
   bool has_reserved_from;
   uint16_t reserved_from;
   bool has_reserved_end;
   uint16_t reserved_end;
   uint8_t float_mode;
   uint8_t thread_id_components;
   uint16_t vgpr_floor;
   struct agc_isa_system_sgprs system;
};


void agc_isa_regalloc_init(struct agc_isa_regalloc *regs, const struct agc_isa_metadata *meta,
                           const struct agc_shader_options *options, const struct agc_shader_interface *iface);
bool agc_isa_regalloc_tg_size_sgpr(const struct agc_isa_regalloc *regs, uint8_t *sgpr);
bool agc_isa_regalloc_scratch_wave_offset_sgpr(const struct agc_isa_regalloc *regs, uint8_t *sgpr);
bool agc_isa_regalloc_collision_wave_sgpr(const struct agc_isa_regalloc *regs, uint8_t *sgpr);
bool agc_isa_regalloc_intrawave_collision_sgpr(const struct agc_isa_regalloc *regs, uint8_t *sgpr);
uint8_t agc_isa_regalloc_thread_id_components(const struct agc_isa_regalloc *regs);
void agc_isa_regalloc_translation_sgprs(const struct agc_isa_regalloc *regs, uint16_t *start, uint16_t *end);
uint16_t agc_isa_regalloc_free_sgprs(const struct agc_isa_regalloc *regs);
bool agc_isa_regalloc_reserve_sgprs(struct agc_isa_regalloc *regs, uint8_t count, uint8_t *base);
bool agc_isa_regalloc_reserve_program_vgprs(struct agc_isa_regalloc *regs, uint8_t count, uint8_t *base,
                                            struct agc_isa_error *error);
uint8_t agc_isa_regalloc_initial_float_mode(const struct agc_isa_regalloc *regs);
bool agc_isa_regalloc_scratch_sgprs(struct agc_isa_regalloc *regs, uint8_t count, uint8_t *base);
bool agc_isa_regalloc_top_sgprs(struct agc_isa_regalloc *regs, uint8_t count, uint8_t *base);
enum agc_isa_target agc_isa_regalloc_target(const struct agc_isa_regalloc *regs);
uint16_t agc_isa_regalloc_scratch_base(const struct agc_isa_regalloc *regs);
bool agc_isa_regalloc_scratch_vgprs(struct agc_isa_regalloc *regs, uint8_t count, uint8_t *base,
                                    struct agc_isa_error *error);
void agc_isa_regalloc_reserve_vgprs(struct agc_isa_regalloc *regs, uint16_t count);
void agc_isa_regalloc_finish(const struct agc_isa_regalloc *regs, const struct agc_isa_metadata *meta,
                             const struct agc_shader_options *options, const struct agc_shader_interface *iface,
                             struct agc_isa_shader_config *config);

#endif
