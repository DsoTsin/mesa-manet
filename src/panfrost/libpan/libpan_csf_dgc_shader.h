/*
 * Copyright © 2026 PanVK contributors
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBPAN_CSF_DGC_SHADER_H
#define LIBPAN_CSF_DGC_SHADER_H

#include "libpan_csf_dgc.h"

/* FAU recipes are per 32-bit word, whereas fau_count is in 64-bit slots.
 * This also describes an odd number of compiler-promoted immediates without
 * overwriting the other half of a slot.
 */
#define PANLIB_DGC_MAX_FAU_WORDS 256
#define PANLIB_DGC_MAX_VARYINGS 128
#define PANLIB_DGC_EARLYZS_STATES 24
#define PANLIB_DGC_MAX_DYNAMIC_BUFFERS 24

enum panlib_dgc_fau_source {
   PANLIB_DGC_FAU_IMMEDIATE = 0,
   PANLIB_DGC_FAU_SYSVAL,
   PANLIB_DGC_FAU_PUSH_CONSTANT,
   PANLIB_DGC_FAU_SELF,
};

struct panlib_dgc_fau_word {
   uint32_t source;
   /* A byte offset into the stage's sysvals or sequence push constants, an
    * immediate value, or 0/4 for the low/high half of this stage's FAU address.
    */
   uint32_t value;
};

enum panlib_dgc_shader_flags {
   PANLIB_DGC_SHADER_WRITES_GLOBAL = 1u << 0,
   PANLIB_DGC_SHADER_CONTAINS_BARRIER = 1u << 1,
   PANLIB_DGC_VS_WRITES_POINT_SIZE = 1u << 2,
   PANLIB_DGC_VS_EXTENDED_FIFO = 1u << 3,
   PANLIB_DGC_VS_SECONDARY = 1u << 4,
   PANLIB_DGC_VS_IDVS = 1u << 5,
   PANLIB_DGC_FS_READS_FRAG_COORD = 1u << 6,
   PANLIB_DGC_FS_READS_POINT_COORD = 1u << 7,
   PANLIB_DGC_FS_READS_PRIMITIVE_ID = 1u << 8,
   PANLIB_DGC_FS_READS_FACE = 1u << 9,
   PANLIB_DGC_FS_CAN_DISCARD = 1u << 10,
   PANLIB_DGC_FS_WRITES_DEPTH = 1u << 11,
   PANLIB_DGC_FS_WRITES_STENCIL = 1u << 12,
   PANLIB_DGC_FS_WRITES_COVERAGE = 1u << 13,
   PANLIB_DGC_FS_SIDE_EFFECTS = 1u << 14,
   PANLIB_DGC_FS_SAMPLE_SHADING = 1u << 15,
   PANLIB_DGC_FS_EARLY_FRAGMENT_TESTS = 1u << 16,
   PANLIB_DGC_FS_CAN_EARLY_Z = 1u << 17,
   PANLIB_DGC_FS_CAN_FPK = 1u << 18,
   PANLIB_DGC_FS_UNTYPED_COLOR_OUTPUTS = 1u << 19,
   PANLIB_DGC_FS_TILE_IMAGE_Z_READ = 1u << 20,
   PANLIB_DGC_FS_TILE_IMAGE_S_READ = 1u << 21,
   PANLIB_DGC_FS_HSR_LD_TILE = 1u << 22,
   PANLIB_DGC_FS_HSR_WAIT_OR_TILE_BEFORE_ATEST = 1u << 23,
   PANLIB_DGC_FS_HSR_COVERAGE_READ = 1u << 24,
   PANLIB_DGC_FS_HSR_CENTROID_INTERPOLATION = 1u << 25,
   PANLIB_DGC_FS_HSR_VARYING_BEFORE_ATEST = 1u << 26,
};

/* Explicit encoding of pan_varying_slot: never share a C bitfield ABI with
 * OpenCL. Offset is a signed 12-bit value (-1 means not assigned yet).
 */
#define PANLIB_DGC_VARYING_LOCATION_SHIFT 0
#define PANLIB_DGC_VARYING_ALU_TYPE_SHIFT 7
#define PANLIB_DGC_VARYING_NCOMPS_SHIFT 15
#define PANLIB_DGC_VARYING_SECTION_SHIFT 18
#define PANLIB_DGC_VARYING_OFFSET_SHIFT 20
#define PANLIB_DGC_EARLYZS_UPDATE_SHIFT 0
#define PANLIB_DGC_EARLYZS_KILL_SHIFT 2
#define PANLIB_DGC_EARLYZS_READONLY_SHIFT 4

struct panlib_dgc_shader {
   uint64_t code;
   /* VS: point-position, triangle-position, varying (v10), or point/triangle
    * combined programs (v12+). FS/CS: shader program in spd[0].
    */
   uint64_t spd[3];
   uint64_t outputs_written;
   uint32_t tls_size;
   uint32_t wls_size;
   uint32_t local_size[3];
   uint32_t allow_merging_workgroups;
   uint32_t compute_size_workgroup;
   uint32_t workgroups_per_task;
   uint32_t work_reg_count;
   uint32_t fau_count;
   uint32_t used_set_mask;
   /* Always zero for IES entries. Fixed-pipeline DGC can use dynamic buffers. */
   uint32_t dyn_buf_count;
   uint32_t dyn_buf_map[PANLIB_DGC_MAX_DYNAMIC_BUFFERS];
   uint32_t flags;
   uint32_t input_attachment_read;
   uint32_t tile_image_color_read;
   uint32_t outputs_read;
   uint32_t fs_varying_attr_desc_count;
   uint32_t varying_count;
   uint32_t varying_generic_size;
   uint32_t varying_noperspective;
   /* Index = ((writes_zs_or_oq * 2 + alpha_to_coverage) * 2 +
    *          zs_always_passes) * 3 + zs_tilebuf_read_mode.
    */
   uint32_t earlyzs[PANLIB_DGC_EARLYZS_STATES];
   uint32_t varying_slots[PANLIB_DGC_MAX_VARYINGS];
   struct panlib_dgc_fau_word fau[PANLIB_DGC_MAX_FAU_WORDS];
};

struct panlib_dgc_execution_set_entry {
   /* PANLIB_DGC_{VS,FS,CS} bits. Zero denotes an uninitialized slot. */
   uint32_t stages;
   uint32_t reserved[3];
   struct panlib_dgc_shader shaders[PANLIB_DGC_STAGE_COUNT];
   /* Linked VS/FS varying ATTRIBUTE descriptors, ready for a driver set.
    * Shader-object entries keep this zero until stages are linked together.
    */
   uint32_t fs_varying_descs[PANLIB_DGC_MAX_VARYINGS][8];
} __attribute__((aligned(64)));

#endif /* LIBPAN_CSF_DGC_SHADER_H */
