#ifndef AGC_KERNELS_H
#define AGC_KERNELS_H

#include "agc_types.h"

struct radeon_info;
struct nir_shader;

#define AGC_KERNEL_MAX_USER_DATA 8u
#define AGC_KERNEL_CODE_ALIGNMENT 256u
#define AGC_KERNEL_CODE_ADDRESS_END (UINT64_C(1) << 48)
#define AGC_KERNEL_MAX_GROUPS_YZ 0xffffu
#define AGC_KERNEL_MAX_DISPATCH_DWORDS 34u
#define AGC_KERNEL_MAX_QUEUE_SETUP_DWORDS 128u

#define AGC_FILL32_MAX_DWORDS UINT32_C(0x40000000)
#define AGC_COPY32_MAX_DWORDS UINT32_C(0x40000000)

enum agc_fill32_user_data {
   AGC_FILL32_ADDRESS_LO,
   AGC_FILL32_ADDRESS_HI,
   AGC_FILL32_COUNT,
   AGC_FILL32_VALUE,
   AGC_FILL32_USER_DATA_DWORDS
};

enum agc_copy32_user_data {
   AGC_COPY32_DST_LO,
   AGC_COPY32_DST_HI,
   AGC_COPY32_SRC_LO,
   AGC_COPY32_SRC_HI,
   AGC_COPY32_COUNT,
   AGC_COPY32_USER_DATA_DWORDS
};

enum agc_element_copy_user_data {
   AGC_ELEMENT_COPY_PARAMS_LO,
   AGC_ELEMENT_COPY_PARAMS_HI,
   AGC_ELEMENT_COPY_USER_DATA_DWORDS
};

#define AGC_ELEMENT_COPY_PARAM_DWORDS 68u
#define AGC_ELEMENT_COPY_LAYOUT_DWORDS 32u
#define AGC_ELEMENT_COPY_GROUP_THREADS 64u

enum agc_display_overlay_user_data {
   AGC_DISPLAY_OVERLAY_DESCRIPTOR_LO,
   AGC_DISPLAY_OVERLAY_DESCRIPTOR_HI,
   AGC_DISPLAY_OVERLAY_DST_LO,
   AGC_DISPLAY_OVERLAY_DST_HI,
   AGC_DISPLAY_OVERLAY_WIDTH,
   AGC_DISPLAY_OVERLAY_HEIGHT,
   AGC_DISPLAY_OVERLAY_PITCH_BYTES,
   AGC_DISPLAY_OVERLAY_SIZE,
   AGC_DISPLAY_OVERLAY_USER_DATA_DWORDS
};

enum agc_display_convert_user_data {
   AGC_DISPLAY_CONVERT_DESCRIPTOR_LO,
   AGC_DISPLAY_CONVERT_DESCRIPTOR_HI,
   AGC_DISPLAY_CONVERT_DST_LO,
   AGC_DISPLAY_CONVERT_DST_HI,
   AGC_DISPLAY_CONVERT_WIDTH,
   AGC_DISPLAY_CONVERT_HEIGHT,
   AGC_DISPLAY_CONVERT_PITCH_BYTES,
   AGC_DISPLAY_CONVERT_USER_DATA_DWORDS
};

struct agc_kernel {
   uint32_t *code;
   uint32_t code_dwords;
   uint32_t exec_dwords;
   uint32_t pgm_rsrc1;
   uint32_t pgm_rsrc2;
   uint32_t pgm_rsrc3;
   uint32_t resource_limits;
   uint32_t block[3];
   uint32_t user_data_dwords;
   uint32_t wave_size;
   uint32_t num_sgprs;
   uint32_t num_vgprs;
   char *listing;
};

bool agc_kernel_compile(const struct radeon_info *info, struct nir_shader *nir,
                        struct agc_kernel *kernel, char *log, size_t log_bytes);
void agc_kernel_free(struct agc_kernel *kernel);

uint32_t agc_kernel_dispatch_dwords(const struct radeon_info *info, const struct agc_kernel *kernel);
uint32_t *agc_kernel_emit_dispatch(uint32_t *out, const struct radeon_info *info,
                                   const struct agc_kernel *kernel, uint64_t code_va,
                                   const uint32_t *user_data, uint32_t groups_x, uint32_t groups_y,
                                   uint32_t groups_z);

uint32_t agc_kernel_queue_setup_dwords(const struct radeon_info *info, bool compute_ring);
uint32_t *agc_kernel_emit_queue_setup(uint32_t *out, const struct radeon_info *info,
                                      bool compute_ring);

struct nir_shader *agc_kernel_fill32(void);
struct nir_shader *agc_kernel_copy32(void);
struct nir_shader *agc_kernel_display_convert(void);
struct nir_shader *agc_kernel_display_overlay(void);
struct nir_shader *agc_kernel_element_copy(void);

static inline uint32_t
agc_kernel_groups(uint32_t threads, uint32_t block)
{
   return threads / block + (threads % block != 0);
}

static inline void
agc_kernel_fill32_user_data(uint32_t user_data[AGC_FILL32_USER_DATA_DWORDS], uint64_t address,
                            uint32_t count, uint32_t value)
{
   user_data[AGC_FILL32_ADDRESS_LO] = (uint32_t)address;
   user_data[AGC_FILL32_ADDRESS_HI] = (uint32_t)(address >> 32);
   user_data[AGC_FILL32_COUNT] = count;
   user_data[AGC_FILL32_VALUE] = value;
}

static inline void
agc_kernel_copy32_user_data(uint32_t user_data[AGC_COPY32_USER_DATA_DWORDS], uint64_t dst,
                            uint64_t src, uint32_t count)
{
   user_data[AGC_COPY32_DST_LO] = (uint32_t)dst;
   user_data[AGC_COPY32_DST_HI] = (uint32_t)(dst >> 32);
   user_data[AGC_COPY32_SRC_LO] = (uint32_t)src;
   user_data[AGC_COPY32_SRC_HI] = (uint32_t)(src >> 32);
   user_data[AGC_COPY32_COUNT] = count;
}

static inline void
agc_kernel_display_convert_user_data(uint32_t user_data[AGC_DISPLAY_CONVERT_USER_DATA_DWORDS],
                                     uint64_t descriptor, uint64_t dst, uint32_t width,
                                     uint32_t height, uint32_t pitch_bytes)
{
   user_data[AGC_DISPLAY_CONVERT_DESCRIPTOR_LO] = (uint32_t)descriptor;
   user_data[AGC_DISPLAY_CONVERT_DESCRIPTOR_HI] = (uint32_t)(descriptor >> 32);
   user_data[AGC_DISPLAY_CONVERT_DST_LO] = (uint32_t)dst;
   user_data[AGC_DISPLAY_CONVERT_DST_HI] = (uint32_t)(dst >> 32);
   user_data[AGC_DISPLAY_CONVERT_WIDTH] = width;
   user_data[AGC_DISPLAY_CONVERT_HEIGHT] = height;
   user_data[AGC_DISPLAY_CONVERT_PITCH_BYTES] = pitch_bytes;
}

#endif
