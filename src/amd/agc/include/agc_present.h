#ifndef AGC_PRESENT_H
#define AGC_PRESENT_H

#include "amd_family.h"

#include "agc_types.h"
#include "agc_videoout.h"

struct radeon_info;
struct agc_backend;

#define AGC_PRESENT_PIXEL_FORMAT_TEN_BIT UINT64_C(0x0100000000000000)
#define AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8 UINT64_C(0x8000000000000000)
#define AGC_PRESENT_PIXEL_FORMAT_R8G8B8A8 UINT64_C(0x8000000022000000)
#define AGC_PRESENT_PIXEL_FORMAT_B10G10R10A2 (AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8 | AGC_PRESENT_PIXEL_FORMAT_TEN_BIT)
#define AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2 (AGC_PRESENT_PIXEL_FORMAT_R8G8B8A8 | AGC_PRESENT_PIXEL_FORMAT_TEN_BIT)
#define AGC_PRESENT_PIXEL_FORMAT_R16G16B16A16_FLOAT UINT64_C(0xc001000622000000)
#define AGC_PRESENT_PIXEL_FORMAT_B16G16R16A16_FLOAT UINT64_C(0xc001000600000000)
#define AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2_BT2100_PQ UINT64_C(0x8100070422000000)
#define AGC_PRESENT_PIXEL_FORMAT_B10G10R10A2_BT2100_PQ UINT64_C(0x8100070400000000)

#define AGC_PRESENT_TILING_64KB 0u
#define AGC_PRESENT_TILING_LINEAR 1u
#define AGC_PRESENT_TILING_GCN_2D 2u

#define AGC_PRESENT_ALIGNMENT_64KB 65536u
#define AGC_PRESENT_ALIGNMENT_4KB 4096u
#define AGC_PRESENT_MAX_EXTENT 16384u
#define AGC_PRESENT_ROW_ALIGNMENT 256u
#define AGC_PRESENT_SLOT_GRANULARITY UINT64_C(0x10000)
#define AGC_PRESENT_BLACK UINT32_C(0xff000000)
#define AGC_PRESENT_DESCRIPTOR_DWORDS 8u
#define AGC_PRESENT_SLOTS 3u

#define AGC_PRESENT_SW_LINEAR 0u
#define AGC_PRESENT_SW_64KB_R_X 27u
#define AGC_PRESENT_DCC_ALIGNMENT_LOG2 12u
#define AGC_PRESENT_BLOCK_64KB UINT64_C(0x10000)

#define AGC_PRESENT_ACQUIRE_DWORDS 8u
#define AGC_PRESENT_PARTIAL_FLUSH_DWORDS 2u

enum agc_present_encoding {
   AGC_PRESENT_UNORM8,
   AGC_PRESENT_UNORM10,
   AGC_PRESENT_PQ,
   AGC_PRESENT_HALF,
};

enum agc_present_layout {
   AGC_PRESENT_LAYOUT_LINEAR,
   AGC_PRESENT_LAYOUT_64KB_R_X,
   AGC_PRESENT_LAYOUT_GCN_2D,
};

struct agc_present_format {
   enum agc_present_encoding encoding;
   bool red_low;
   uint32_t texel_bytes;
};

struct agc_present_geometry {
   struct agc_present_format format;
   enum agc_present_layout layout;
   uint32_t pitch;
   uint32_t block_width;
   uint32_t block_height;
   uint64_t size;
};

struct agc_present_source {
   uint64_t va;
   uint64_t bytes;
   uint32_t width;
   uint32_t height;
   uint32_t pitch;
   bool custom_pitch;
   uint32_t texel_bytes;
   uint32_t swizzle_mode;
   uint32_t pipe_format;
   uint32_t swizzle[4];
   bool dcc;
   uint64_t meta_va;
   uint32_t dcc_max_compressed;
};

struct agc_present_canvas {
   uint32_t width;
   uint32_t height;
   uint32_t image_x;
   uint32_t image_y;
   uint32_t image_width;
   uint32_t image_height;
};

struct agc_present_rect {
   int32_t x;
   int32_t y;
   uint32_t width;
   uint32_t height;
};

struct agc_present_footprint {
   uint32_t width;
   uint32_t height;
   uint64_t row_pitch;
   uint64_t required_bytes;
   uint64_t slot_bytes;
};

const char *agc_present_format_name(uint64_t pixel_format);
bool agc_present_format_of(uint64_t pixel_format, struct agc_present_format *format, char *error,
                           size_t error_bytes);
bool agc_present_geometry_of(const struct agc_display_buffer *buffer, struct agc_present_geometry *geometry,
                             char *error, size_t error_bytes);
size_t agc_present_display_buffer_size(const struct agc_display_buffer *buffer);
uint64_t agc_present_source_address(uint64_t address, bool tiled);
bool agc_present_source_of(const struct agc_display_buffer *buffer, enum amd_gfx_level gfx_level,
                           struct agc_present_source *source, char *error, size_t error_bytes);
bool agc_present_descriptor(const struct radeon_info *info, const struct agc_present_source *source,
                            uint32_t words[AGC_PRESENT_DESCRIPTOR_DWORDS], char *error, size_t error_bytes);

bool agc_present_canvas_fit(uint32_t image_width, uint32_t image_height, uint32_t client_width,
                            uint32_t client_height, struct agc_present_canvas *canvas, char *error,
                            size_t error_bytes);
bool agc_present_canvas_equal(const struct agc_present_canvas *a, const struct agc_present_canvas *b);
bool agc_present_footprint_of(uint32_t width, uint32_t height, struct agc_present_footprint *footprint);
uint64_t agc_present_image_offset(const struct agc_present_canvas *canvas,
                                  const struct agc_present_footprint *footprint);
uint32_t agc_present_frame_slot(uint64_t frame);

uint32_t agc_present_gcr_before(void);
uint32_t agc_present_gcr_after(void);
uint32_t *agc_present_emit_acquire(uint32_t *out, uint32_t gcr_cntl);
uint32_t *agc_present_emit_partial_flush(uint32_t *out);
uint32_t agc_present_padding(uint32_t used, uint32_t multiple);
uint32_t *agc_present_emit_padding(uint32_t *out, uint32_t dwords);

struct agc_present_device {
   struct agc_backend *backend;
   const struct radeon_info *info;
   bool (*sync)(char *error, size_t error_bytes);
   bool (*resident)(uint64_t va, uint64_t bytes);
};

typedef bool (*agc_present_open_device)(struct agc_present_device *device, char *error, size_t error_bytes);

struct agc_present_timing {
   uint64_t frame;
   bool presented;
   int64_t total_ns;
   int64_t wait_ns;
   int64_t convert_ns;
   int64_t present_ns;
};

struct agc_present_frame {
   uint64_t number;
   bool cleared;
   uint32_t slot;
   const void *cpu;
   uint64_t va;
   uint64_t image_offset;
   uint32_t width;
   uint32_t height;
   uint64_t row_pitch;
   struct agc_present_rect rect;
};

const struct agc_presenter *agc_present_d3d12(agc_present_open_device open);
const struct agc_presenter *agc_present_d3d12_verified(agc_present_open_device open);
bool agc_present_last_timing(struct agc_present_timing *timing);
bool agc_present_last_frame(struct agc_present_frame *frame);
void agc_present_install(void);

#endif
