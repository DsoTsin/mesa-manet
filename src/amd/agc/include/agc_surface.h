#ifndef AGC_SURFACE_H
#define AGC_SURFACE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define AGC_SURFACE_MAX_MIPS 16u
#define AGC_SURFACE_EQUATION_BITS 16u
#define AGC_SURFACE_COPY_PARAM_DWORDS 68u
#define AGC_SURFACE_COPY_LAYOUT_DWORDS 32u
#define AGC_SURFACE_COPY_GROUP_THREADS 64u
#define AGC_SURFACE_HOST_GB_ADDR_CONFIG 0x343u

#define AGC_SW_LINEAR 0u
#define AGC_SW_256B_S 1u
#define AGC_SW_256B_D 2u
#define AGC_SW_4KB_S 5u
#define AGC_SW_4KB_D 6u
#define AGC_SW_64KB_S 9u
#define AGC_SW_64KB_D 10u
#define AGC_SW_64KB_S_T 17u
#define AGC_SW_64KB_D_T 18u
#define AGC_SW_4KB_S_X 21u
#define AGC_SW_4KB_D_X 22u
#define AGC_SW_64KB_Z_X 24u
#define AGC_SW_64KB_S_X 25u
#define AGC_SW_64KB_D_X 26u
#define AGC_SW_64KB_R_X 27u

enum agc_surface_addressing {
   AGC_SURFACE_GUEST,
   AGC_SURFACE_HOST,
};

enum agc_surface_use {
   AGC_SURFACE_USE_THIN_2D,
   AGC_SURFACE_USE_1D,
   AGC_SURFACE_USE_MSAA,
   AGC_SURFACE_USE_VOLUME,
   AGC_SURFACE_USE_DEPTH,
};

enum agc_surface_tiling {
   AGC_SURFACE_TILING_LINEAR,
   AGC_SURFACE_TILING_THIN,
   AGC_SURFACE_TILING_THICK,
};

struct agc_surface_desc {
   uint32_t swizzle_mode;
   uint32_t width;
   uint32_t height;
   uint32_t depth;
   uint32_t layers;
   uint32_t mip_levels;
   uint32_t bytes_per_element;
   uint32_t pitch_elements;
   bool volume;
   uint32_t pipe_bank_xor;
};

struct agc_surface_mip {
   uint64_t offset;
   uint64_t size;
   uint32_t pitch;
   bool in_tail;
   uint32_t tail_origin[2];
};

struct agc_surface_layout {
   struct agc_surface_desc desc;
   enum agc_surface_addressing addressing;
   enum agc_surface_tiling tiling;
   uint64_t block_bytes;
   uint32_t block[3];
   uint32_t equation[AGC_SURFACE_EQUATION_BITS];
   uint32_t xor_bytes;
   uint32_t mip_count;
   struct agc_surface_mip mips[AGC_SURFACE_MAX_MIPS];
   uint64_t slice_stride;
   uint64_t size;
};

struct agc_image_view {
   uint64_t base;
   uint32_t swizzle_mode;
   uint32_t pipe_bank_xor;
   uint32_t type;
   uint32_t format;
   uint32_t width;
   uint32_t height;
   uint32_t depth;
   uint32_t last_array;
   uint32_t base_level;
   uint32_t last_level;
   uint32_t max_mip;
   uint32_t min_lod;
   uint32_t pitch;
   bool has_meta;
   uint64_t meta;
};

struct agc_texel_format {
   uint32_t bytes_per_element;
   uint32_t block_width;
   uint32_t block_height;
};

bool agc_texel_format_from_gfx10(uint32_t format, struct agc_texel_format *texel, char *error, size_t error_bytes);

bool agc_image_view_decode(const uint32_t words[8], struct agc_image_view *view, char *error, size_t error_bytes);
enum agc_surface_use agc_image_view_surface_use(const struct agc_image_view *view);
bool agc_image_view_surface_desc(const struct agc_image_view *view, struct agc_surface_desc *desc, char *error,
                                 size_t error_bytes);
uint32_t agc_image_view_first_reachable_level(const struct agc_image_view *view);
bool agc_image_view_samples_in_place(const struct agc_image_view *view);
uint32_t agc_image_view_host_swizzle_mode(const struct agc_image_view *view);

uint32_t agc_surface_host_swizzle_mode(uint32_t guest_mode, enum agc_surface_use use);
bool agc_surface_layout_init(const struct agc_surface_desc *desc, enum agc_surface_addressing addressing,
                             struct agc_surface_layout *layout, char *error, size_t error_bytes);
bool agc_surface_same_addressing(const struct agc_surface_layout *a, const struct agc_surface_layout *b);
bool agc_surface_element_offset(const struct agc_surface_layout *layout, uint32_t x, uint32_t y, uint32_t slice,
                                uint32_t mip, uint64_t *offset, char *error, size_t error_bytes);
void agc_surface_level_extent(const struct agc_surface_layout *layout, uint32_t level, uint32_t extent[3]);
bool agc_surface_copy_params(const struct agc_surface_layout *from, uint64_t from_base,
                             const struct agc_surface_layout *to, uint64_t to_base, uint32_t level,
                             uint32_t params[AGC_SURFACE_COPY_PARAM_DWORDS], uint32_t groups[3], char *error,
                             size_t error_bytes);

uint32_t agc_surface_type_host_swizzle_mode(uint32_t type, uint32_t swizzle_mode);
bool agc_surface_samples_in_place_for_format(uint32_t type, uint32_t swizzle_mode, uint32_t format);
void agc_surface_split_pipe_bank_xor(uint32_t swizzle_mode, uint64_t address, uint64_t *base,
                                     uint32_t *pipe_bank_xor);
uint32_t agc_surface_host_pipe_bank_xor(uint32_t block_log2, uint32_t pipe_bank_xor);
bool agc_surface_same_placement(const struct agc_surface_layout *a, const struct agc_surface_layout *b);
void agc_surface_mip_extent(const struct agc_surface_layout *layout, uint32_t mip, uint32_t extent[3]);
const uint32_t *agc_surface_guest_equation(uint32_t swizzle_mode, bool thick, uint32_t element_log2);
const uint32_t *agc_surface_host_equation(uint32_t swizzle_mode, uint32_t element_log2);
bool agc_surface_native_dcc(void);
void agc_surface_note_dcc_target(uint64_t guest_va, uint64_t meta_va, uint32_t fdcc_control);
bool agc_surface_dcc_target(uint64_t guest_va, uint64_t *meta_va, uint32_t *fdcc_control);

#endif
