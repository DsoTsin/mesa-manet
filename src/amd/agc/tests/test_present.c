#include <string.h>

#include "ac_gpu_info.h"
#include "agc_memory.h"
#include "agc_present.h"
#include "agc_surface.h"
#include "sid.h"
#include "test.h"
#include "util/format/u_formats.h"

#define GUEST UINT64_C(0x240000000)

struct target {
   enum amd_gfx_level level;
   enum radeon_family family;
};

struct tsharp {
   uint64_t va;
   uint32_t format;
   uint32_t width;
   uint32_t height;
   uint32_t dst_sel;
   uint32_t sw;
   uint32_t depth;
   bool gfx11;
};

static void
describe(enum amd_gfx_level level, enum radeon_family family, struct radeon_info *info)
{
   memset(info, 0, sizeof(*info));
   info->gfx_level = level;
   info->family = family;
   info->has_graphics = true;
   info->has_image_opcodes = true;
   info->num_se = 2;
   info->max_se = 2;
   ac_fill_compiler_info(info, NULL, false);
}

static uint32_t
put(uint64_t value, unsigned lo, unsigned width)
{
   return (uint32_t)(value & ((UINT64_C(1) << width) - 1)) << lo;
}

static uint32_t
selects(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
   return x | y << 3 | z << 6 | w << 9;
}

static void
rust_tsharp(const struct tsharp *t, uint32_t w[8])
{
   memset(w, 0, 8 * sizeof(uint32_t));
   w[0] = (uint32_t)(t->va >> 8);
   w[1] = put(t->va >> 40, 0, 8) | put(t->format, 20, t->gfx11 ? 8 : 9) | put(t->width - 1, 30, 2);
   w[2] = put((t->width - 1) >> 2, 0, 12) | put(t->height - 1, 14, 14);
   if (!t->gfx11)
      w[2] |= put(1, 31, 1);
   w[3] = put(t->dst_sel, 0, 12) | put(t->sw, 20, 5) | put(9, 28, 4);
   w[4] = put(t->depth, 0, 13) | put(t->depth >> 13, 13, 1);
   w[5] = put(4, 20, 3);
}

static void
check_words(const uint32_t *got, const uint32_t *expected, const char *what)
{
   for (unsigned i = 0; i < 8; i++) {
      if (got[i] != expected[i]) {
         fprintf(stderr, "%s: word %u is 0x%08x, expected 0x%08x\n", what, i, got[i], expected[i]);
         CHECK(false);
      }
   }
}

static struct agc_display_buffer
display(uint64_t address, uint64_t format, uint32_t width, uint32_t height, uint32_t tiling, uint32_t pitch)
{
   struct agc_display_buffer buffer;
   memset(&buffer, 0, sizeof(buffer));
   buffer.address = address;
   buffer.pixel_format = format;
   buffer.width = width;
   buffer.height = height;
   buffer.tiling_mode = tiling;
   buffer.pitch_in_pixel = pitch;
   return buffer;
}

static void
test_formats(void)
{
   struct agc_present_format format;
   char error[256];
   CHECK(agc_present_format_of(AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, &format, error, sizeof(error)));
   CHECK(format.encoding == AGC_PRESENT_UNORM8 && !format.red_low && format.texel_bytes == 4);
   CHECK(agc_present_format_of(AGC_PRESENT_PIXEL_FORMAT_R8G8B8A8, &format, error, sizeof(error)));
   CHECK(format.encoding == AGC_PRESENT_UNORM8 && format.red_low && format.texel_bytes == 4);
   CHECK(agc_present_format_of(AGC_PRESENT_PIXEL_FORMAT_B10G10R10A2, &format, error, sizeof(error)));
   CHECK(format.encoding == AGC_PRESENT_UNORM10 && !format.red_low);
   CHECK(agc_present_format_of(AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2, &format, error, sizeof(error)));
   CHECK(format.encoding == AGC_PRESENT_UNORM10 && format.red_low);
   CHECK(agc_present_format_of(AGC_PRESENT_PIXEL_FORMAT_R16G16B16A16_FLOAT, &format, error, sizeof(error)));
   CHECK(format.encoding == AGC_PRESENT_HALF && format.red_low && format.texel_bytes == 8);
   CHECK(agc_present_format_of(AGC_PRESENT_PIXEL_FORMAT_B16G16R16A16_FLOAT, &format, error, sizeof(error)));
   CHECK(format.encoding == AGC_PRESENT_HALF && !format.red_low);
   CHECK(agc_present_format_of(AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2_BT2100_PQ, &format, error, sizeof(error)));
   CHECK(format.encoding == AGC_PRESENT_PQ && format.red_low && format.texel_bytes == 4);
   CHECK(agc_present_format_of(AGC_PRESENT_PIXEL_FORMAT_B10G10R10A2_BT2100_PQ, &format, error, sizeof(error)));
   CHECK(format.encoding == AGC_PRESENT_PQ && !format.red_low);
   CHECK(!agc_present_format_of(UINT64_C(0x8000000011000000), &format, error, sizeof(error)));
   CHECK(strstr(error, "0x8000000011000000") != NULL);
   CHECK(!strcmp(agc_present_format_name(AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8), "B8G8R8A8_SRGB"));
   CHECK(!strcmp(agc_present_format_name(AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2), "R10G10B10A2"));
   CHECK(!strcmp(agc_present_format_name(1), "unknown"));
}

static void
test_sizes(void)
{
   struct agc_display_buffer buffer =
      display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_64KB, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 15u * 9u * 65536u);
   struct agc_present_geometry geometry;
   char error[256];
   CHECK(agc_present_geometry_of(&buffer, &geometry, error, sizeof(error)));
   CHECK(geometry.layout == AGC_PRESENT_LAYOUT_64KB_R_X);
   CHECK_EQ(geometry.block_width, 128);
   CHECK_EQ(geometry.block_height, 128);

   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_R16G16B16A16_FLOAT, 1920, 1080, AGC_PRESENT_TILING_64KB, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 15u * 17u * 65536u);
   CHECK(agc_present_geometry_of(&buffer, &geometry, error, sizeof(error)));
   CHECK_EQ(geometry.block_width, 128);
   CHECK_EQ(geometry.block_height, 64);

   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_R8G8B8A8, 1920, 1080, AGC_PRESENT_TILING_LINEAR, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 1920u * 1080u * 4u);
   buffer.pitch_in_pixel = 2048;
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 2048u * 1080u * 4u);
   buffer.pitch_in_pixel = 1900;
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer.pitch_in_pixel = 16385;
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);

   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_GCN_2D, 1920);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 15u * 17u * 32768u);
   buffer.pitch_in_pixel = 1984;
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_R16G16B16A16_FLOAT, 1920, 1080, AGC_PRESENT_TILING_GCN_2D, 1920);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);

   buffer = display(GUEST + 0x1000, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_LINEAR, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer.alignment = AGC_PRESENT_ALIGNMENT_4KB;
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 1920u * 1080u * 4u);
   buffer.tiling_mode = AGC_PRESENT_TILING_64KB;
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer.alignment = 0x2000;
   buffer.tiling_mode = AGC_PRESENT_TILING_LINEAR;
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);

   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_64KB, 1920);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_LINEAR, 0);
   buffer.dcc_address = GUEST + 0x1000000;
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer.tiling_mode = AGC_PRESENT_TILING_64KB;
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 15u * 9u * 65536u);
   buffer.pixel_format = AGC_PRESENT_PIXEL_FORMAT_B16G16R16A16_FLOAT;
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_64KB, 0);
   buffer.dcc_clear_color = 1;
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);

   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 0, 1080, AGC_PRESENT_TILING_64KB, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 16385, 1080, AGC_PRESENT_TILING_LINEAR, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer = display(0, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_64KB, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, 3, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer = display(GUEST, 7, 1920, 1080, AGC_PRESENT_TILING_64KB, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   buffer = display(UINT64_MAX - 0xffff, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_64KB, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 0);
   CHECK_EQ(agc_present_display_buffer_size(NULL), 0);

   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1280, 720, AGC_PRESENT_TILING_64KB, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 10u * 6u * 65536u);
   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1, 1, AGC_PRESENT_TILING_64KB, 0);
   CHECK_EQ(agc_present_display_buffer_size(&buffer), 65536u);
}

static void
test_sources(void)
{
   struct agc_present_source source;
   char error[256];
   struct agc_display_buffer buffer =
      display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_64KB, 0);
   CHECK(agc_present_source_of(&buffer, GFX11_5, &source, error, sizeof(error)));
   CHECK_EQ(source.va, GUEST + AGC_MIRROR_OFFSET);
   CHECK_EQ(source.swizzle_mode, AGC_PRESENT_SW_64KB_R_X);
   CHECK_EQ(source.bytes, 15u * 9u * 65536u);
   CHECK_EQ(source.pipe_format, PIPE_FORMAT_R8G8B8A8_UNORM);
   CHECK(source.swizzle[0] == PIPE_SWIZZLE_X && source.swizzle[1] == PIPE_SWIZZLE_Y &&
         source.swizzle[2] == PIPE_SWIZZLE_Z && source.swizzle[3] == PIPE_SWIZZLE_W);
   CHECK(!source.custom_pitch);

   buffer.pixel_format = AGC_PRESENT_PIXEL_FORMAT_R8G8B8A8;
   buffer.tiling_mode = AGC_PRESENT_TILING_LINEAR;
   CHECK(agc_present_source_of(&buffer, GFX11, &source, error, sizeof(error)));
   CHECK_EQ(source.va, GUEST);
   CHECK_EQ(source.swizzle_mode, AGC_PRESENT_SW_LINEAR);
   CHECK_EQ(source.pitch, 1920);
   CHECK(!source.custom_pitch);
   CHECK(source.swizzle[0] == PIPE_SWIZZLE_Z && source.swizzle[2] == PIPE_SWIZZLE_X &&
         source.swizzle[3] == PIPE_SWIZZLE_W);

   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2, 1000, 600, AGC_PRESENT_TILING_LINEAR, 1088);
   CHECK(agc_present_source_of(&buffer, GFX11_5, &source, error, sizeof(error)));
   CHECK(source.custom_pitch);
   CHECK_EQ(source.pipe_format, PIPE_FORMAT_R10G10B10A2_UNORM);
   CHECK(source.swizzle[0] == PIPE_SWIZZLE_Z && source.swizzle[3] == PIPE_SWIZZLE_1);
   buffer.pitch_in_pixel = 1024;
   CHECK(agc_present_source_of(&buffer, GFX11_5, &source, error, sizeof(error)));
   CHECK(!source.custom_pitch);
   buffer.pitch_in_pixel = 1000;
   CHECK(!agc_present_source_of(&buffer, GFX11_5, &source, error, sizeof(error)));
   CHECK(strstr(error, "256 bytes") != NULL);

   buffer = display(AGC_DRIVER_AREA_BASE + 0x10000, AGC_PRESENT_PIXEL_FORMAT_B10G10R10A2, 1280, 720,
                    AGC_PRESENT_TILING_64KB, 0);
   CHECK(agc_present_source_of(&buffer, GFX11_5, &source, error, sizeof(error)));
   CHECK_EQ(source.va, AGC_DRIVER_AREA_BASE + 0x10000);
   CHECK(source.swizzle[0] == PIPE_SWIZZLE_X && source.swizzle[2] == PIPE_SWIZZLE_Z &&
         source.swizzle[3] == PIPE_SWIZZLE_1);

   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2_BT2100_PQ, 1920, 1080, AGC_PRESENT_TILING_64KB, 0);
   CHECK(!agc_present_source_of(&buffer, GFX11_5, &source, error, sizeof(error)));
   CHECK(strstr(error, "0x8100070422000000") != NULL && strstr(error, "R10G10B10A2_BT2100_PQ") != NULL);
   buffer.pixel_format = AGC_PRESENT_PIXEL_FORMAT_B16G16R16A16_FLOAT;
   CHECK(!agc_present_source_of(&buffer, GFX11_5, &source, error, sizeof(error)));
   CHECK(strstr(error, "B16G16R16A16_FLOAT") != NULL);
   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_GCN_2D, 1920);
   CHECK(!agc_present_source_of(&buffer, GFX11_5, &source, error, sizeof(error)));
   CHECK(strstr(error, "GCN") != NULL);
   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_64KB, 0);
   CHECK(!agc_present_source_of(&buffer, GFX12, &source, error, sizeof(error)));
   CHECK(!agc_present_source_of(&buffer, GFX10, &source, error, sizeof(error)));

   CHECK_EQ(agc_present_source_address(GUEST, true), GUEST + AGC_MIRROR_OFFSET);
   CHECK_EQ(agc_present_source_address(GUEST, false), GUEST);
   CHECK_EQ(agc_present_source_address(AGC_GUEST_WINDOW_END - 0x10000, true),
            AGC_GUEST_WINDOW_END - 0x10000 + AGC_MIRROR_OFFSET);
   CHECK_EQ(agc_present_source_address(AGC_GUEST_WINDOW_END, true), AGC_GUEST_WINDOW_END);
   CHECK_EQ(agc_present_source_address(AGC_DRIVER_AREA_BASE, true), AGC_DRIVER_AREA_BASE);
}

static void
descriptor_case(const struct target *target, const struct agc_display_buffer *buffer, const struct tsharp *rust,
                const uint32_t literal[8], const char *what)
{
   struct radeon_info info;
   struct agc_present_source source;
   uint32_t words[8];
   uint32_t expected[8];
   char error[256] = "";
   describe(target->level, target->family, &info);
   bool ok = agc_present_source_of(buffer, info.gfx_level, &source, error, sizeof(error)) &&
             agc_present_descriptor(&info, &source, words, error, sizeof(error));
   CHECK(ok);
   if (!ok) {
      fprintf(stderr, "%s: %s\n", what, error);
      return;
   }
   rust_tsharp(rust, expected);
   check_words(words, expected, what);
   check_words(expected, literal, what);
}

static void
test_descriptors(void)
{
   static const struct target strix = {GFX11_5, CHIP_STRIX_HALO};
   static const struct target navi31 = {GFX11, CHIP_NAVI31};
   static const struct target navi21 = {GFX10_3, CHIP_NAVI21};
   const uint32_t sel_bgra = selects(V_008F1C_SQ_SEL_X, V_008F1C_SQ_SEL_Y, V_008F1C_SQ_SEL_Z, V_008F1C_SQ_SEL_W);
   const uint32_t sel_rgba = selects(V_008F1C_SQ_SEL_Z, V_008F1C_SQ_SEL_Y, V_008F1C_SQ_SEL_X, V_008F1C_SQ_SEL_W);
   const uint32_t sel_r10 = selects(V_008F1C_SQ_SEL_Z, V_008F1C_SQ_SEL_Y, V_008F1C_SQ_SEL_X, V_008F1C_SQ_SEL_1);
   const uint32_t sel_b10 = selects(V_008F1C_SQ_SEL_X, V_008F1C_SQ_SEL_Y, V_008F1C_SQ_SEL_Z, V_008F1C_SQ_SEL_1);

   struct agc_display_buffer buffer =
      display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_64KB, 0);
   struct tsharp rust = {GUEST + AGC_MIRROR_OFFSET, 42, 1920, 1080, sel_bgra, 27, 0, true};
   static const uint32_t tiled_bgra[8] = {0x02400000, 0xc2a00001, 0x010dc1df, 0x91b00fac, 0, 0x00400000, 0, 0};
   descriptor_case(&strix, &buffer, &rust, tiled_bgra, "gfx11.5 tiled B8G8R8A8 1920x1080");

   buffer = display(UINT64_C(0x250000000), AGC_PRESENT_PIXEL_FORMAT_R8G8B8A8, 1920, 1080, AGC_PRESENT_TILING_LINEAR, 0);
   rust = (struct tsharp){UINT64_C(0x250000000), 42, 1920, 1080, sel_rgba, 0, 0, true};
   static const uint32_t linear_rgba[8] = {0x02500000, 0xc2a00000, 0x010dc1df, 0x90000f2e, 0, 0x00400000, 0, 0};
   descriptor_case(&navi31, &buffer, &rust, linear_rgba, "gfx11 linear R8G8B8A8 1920x1080");

   buffer = display(UINT64_C(0x260000000), AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1000, 600, AGC_PRESENT_TILING_LINEAR, 1088);
   rust = (struct tsharp){UINT64_C(0x260000000), 42, 1000, 600, sel_bgra, 0, 1087, true};
   static const uint32_t pitched[8] = {0x02600000, 0xc2a00000, 0x0095c0f9, 0x90000fac, 0x0000043f, 0x00400000, 0, 0};
   descriptor_case(&strix, &buffer, &rust, pitched, "gfx11.5 linear B8G8R8A8 1000x600 pitch 1088");

   buffer = display(UINT64_C(0x270000000), AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 8200, 16, AGC_PRESENT_TILING_LINEAR, 8320);
   rust = (struct tsharp){UINT64_C(0x270000000), 42, 8200, 16, sel_bgra, 0, 8319, true};
   static const uint32_t wide[8] = {0x02700000, 0xc2a00000, 0x0003c801, 0x90000fac, 0x0000207f, 0x00400000, 0, 0};
   descriptor_case(&navi31, &buffer, &rust, wide, "gfx11 linear B8G8R8A8 8200x16 pitch 8320");

   buffer = display(UINT64_C(0x300000000), AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2, 3840, 2160, AGC_PRESENT_TILING_64KB, 0);
   rust = (struct tsharp){UINT64_C(0x300000000) + AGC_MIRROR_OFFSET, 50, 3840, 2160, sel_r10, 27, 0, false};
   static const uint32_t tiled_r10[8] = {0x03000000, 0xc3200001, 0x821bc3bf, 0x91b0032e, 0, 0x00400000, 0, 0};
   descriptor_case(&navi21, &buffer, &rust, tiled_r10, "gfx10.3 tiled R10G10B10A2 3840x2160");

   buffer = display(AGC_DRIVER_AREA_BASE + 0x10000, AGC_PRESENT_PIXEL_FORMAT_B10G10R10A2, 1280, 720,
                    AGC_PRESENT_TILING_64KB, 0);
   rust = (struct tsharp){AGC_DRIVER_AREA_BASE + 0x10000, 36, 1280, 720, sel_b10, 27, 0, true};
   static const uint32_t driver_b10[8] = {0x00000100, 0xc2400002, 0x00b3c13f, 0x91b003ac, 0, 0x00400000, 0, 0};
   descriptor_case(&strix, &buffer, &rust, driver_b10, "gfx11.5 tiled B10G10R10A2 in driver memory");

   struct radeon_info info;
   struct agc_present_source source;
   uint32_t words[8];
   char error[256];
   describe(GFX11_5, CHIP_STRIX_HALO, &info);
   buffer = display(GUEST, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 1920, 1080, AGC_PRESENT_TILING_64KB, 0);
   CHECK(agc_present_source_of(&buffer, info.gfx_level, &source, error, sizeof(error)));
   info.has_image_opcodes = false;
   CHECK(!agc_present_descriptor(&info, &source, words, error, sizeof(error)));
   describe(GFX12, CHIP_GFX1201, &info);
   CHECK(!agc_present_descriptor(&info, &source, words, error, sizeof(error)));
   describe(GFX11_5, CHIP_STRIX_HALO, &info);
   source.custom_pitch = true;
   CHECK(!agc_present_descriptor(&info, &source, words, error, sizeof(error)));
   source.custom_pitch = false;
   source.va += 0x80;
   CHECK(!agc_present_descriptor(&info, &source, words, error, sizeof(error)));
   CHECK(!agc_present_descriptor(NULL, &source, words, error, sizeof(error)));
}

static void
expect_canvas(uint32_t iw, uint32_t ih, uint32_t cw, uint32_t ch, struct agc_present_canvas expected)
{
   struct agc_present_canvas canvas;
   char error[128];
   CHECK(agc_present_canvas_fit(iw, ih, cw, ch, &canvas, error, sizeof(error)));
   if (!agc_present_canvas_equal(&canvas, &expected)) {
      fprintf(stderr, "canvas of %ux%u in %ux%u: %ux%u image at %u,%u %ux%u\n", iw, ih, cw, ch, canvas.width,
              canvas.height, canvas.image_x, canvas.image_y, canvas.image_width, canvas.image_height);
      CHECK(false);
   }
}

static void
test_dcc_descriptors(void)
{
   struct radeon_info info;
   struct agc_present_source source;
   uint32_t words[8];
   char error[256] = "";
   describe(GFX11_5, CHIP_STRIX_HALO, &info);

   const uint64_t display_va = UINT64_C(0x240690000);
   const uint64_t meta_va = display_va + 0x2008000;
   struct agc_display_buffer buffer =
      display(display_va, AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8, 3840, 2160, AGC_PRESENT_TILING_64KB, 0);
   CHECK(agc_present_source_of(&buffer, info.gfx_level, &source, error, sizeof(error)));
   CHECK(!source.dcc);
   CHECK(agc_present_descriptor(&info, &source, words, error, sizeof(error)));
   CHECK_EQ(words[6], 0);
   CHECK_EQ(words[7], 0);

   buffer.dcc_address = meta_va;
   CHECK(agc_present_source_of(&buffer, info.gfx_level, &source, error, sizeof(error)));
   CHECK(source.dcc);
   CHECK_EQ(source.va, display_va + AGC_MIRROR_OFFSET);
   CHECK_EQ(source.meta_va, meta_va);
   CHECK_EQ(source.dcc_max_compressed, V_028C78_MAX_BLOCK_SIZE_64B);
   CHECK(agc_present_descriptor(&info, &source, words, error, sizeof(error)));
   CHECK_EQ(words[0], (uint32_t)(source.va >> 8));
   CHECK_EQ(words[6], S_00A018_MAX_UNCOMPRESSED_BLOCK_SIZE(V_028C78_MAX_BLOCK_SIZE_256B) |
                         S_00A018_MAX_COMPRESSED_BLOCK_SIZE(V_028C78_MAX_BLOCK_SIZE_64B) |
                         S_00A018_META_PIPE_ALIGNED(1) | S_00A018_COMPRESSION_EN(1) |
                         S_00A018_META_DATA_ADDRESS_LO(meta_va >> 8));
   CHECK_EQ(words[7], (uint32_t)(meta_va >> 16));
   CHECK_EQ(words[6] & S_00A018_ALPHA_IS_ON_MSB(1), 0);

   agc_surface_note_dcc_target(display_va, 0,
                               S_028C78_MAX_COMPRESSED_BLOCK_SIZE(V_028C78_MAX_BLOCK_SIZE_128B) |
                                  S_028C78_FDCC_ENABLE(1));
   CHECK(agc_present_source_of(&buffer, info.gfx_level, &source, error, sizeof(error)));
   CHECK_EQ(source.dcc_max_compressed, V_028C78_MAX_BLOCK_SIZE_128B);
   CHECK_EQ(source.meta_va, meta_va);
   const uint64_t rendered_va = meta_va + 0x20000;
   agc_surface_note_dcc_target(display_va, rendered_va,
                               S_028C78_MAX_COMPRESSED_BLOCK_SIZE(V_028C78_MAX_BLOCK_SIZE_128B) |
                                  S_028C78_FDCC_ENABLE(1));
   CHECK(agc_present_source_of(&buffer, info.gfx_level, &source, error, sizeof(error)));
   CHECK_EQ(source.meta_va, rendered_va);
   CHECK(agc_present_descriptor(&info, &source, words, error, sizeof(error)));
   CHECK_EQ(words[7], (uint32_t)(rendered_va >> 16));
   agc_surface_note_dcc_target(display_va, 0,
                               S_028C78_MAX_COMPRESSED_BLOCK_SIZE(V_028C78_MAX_BLOCK_SIZE_128B) |
                                  S_028C78_FDCC_ENABLE(1));
   CHECK(agc_present_source_of(&buffer, info.gfx_level, &source, error, sizeof(error)));
   CHECK(agc_present_descriptor(&info, &source, words, error, sizeof(error)));
   CHECK_EQ(G_00A018_MAX_COMPRESSED_BLOCK_SIZE(words[6]), V_028C78_MAX_BLOCK_SIZE_128B);

   buffer.dcc_address = meta_va + 0x80;
   CHECK(!agc_present_source_of(&buffer, info.gfx_level, &source, error, sizeof(error)));
   buffer.dcc_address = meta_va;
   CHECK(agc_present_source_of(&buffer, info.gfx_level, &source, error, sizeof(error)));
   source.swizzle_mode = AGC_PRESENT_SW_LINEAR;
   CHECK(!agc_present_descriptor(&info, &source, words, error, sizeof(error)));
}

static void
test_geometry(void)
{
   expect_canvas(1920, 1080, 1920, 1080, (struct agc_present_canvas){1920, 1080, 0, 0, 1920, 1080});
   expect_canvas(1920, 1080, 1280, 720, (struct agc_present_canvas){1920, 1080, 0, 0, 1920, 1080});
   expect_canvas(1920, 1080, 1280, 1024, (struct agc_present_canvas){1920, 1536, 0, 228, 1920, 1080});
   expect_canvas(1280, 720, 2560, 1080, (struct agc_present_canvas){1707, 720, 213, 0, 1280, 720});
   expect_canvas(1280, 700, 1280, 720, (struct agc_present_canvas){1280, 720, 0, 10, 1280, 700});
   expect_canvas(1281, 720, 1282, 720, (struct agc_present_canvas){1282, 720, 0, 0, 1281, 720});
   expect_canvas(16384, 16, 16, 16384, (struct agc_present_canvas){16384, 16384, 0, 8184, 16384, 16});
   struct agc_present_canvas canvas;
   char error[128];
   CHECK(!agc_present_canvas_fit(0, 1, 1, 1, &canvas, error, sizeof(error)));
   CHECK(!agc_present_canvas_fit(16385, 1, 1, 1, &canvas, error, sizeof(error)));
   CHECK(!agc_present_canvas_fit(1, 1, 0, 1, &canvas, error, sizeof(error)));

   struct agc_present_footprint footprint;
   CHECK(agc_present_footprint_of(1280, 720, &footprint));
   CHECK_EQ(footprint.row_pitch, 5120);
   CHECK_EQ(footprint.required_bytes, 5120u * 720u);
   CHECK_EQ(footprint.slot_bytes, 57u * 65536u);
   CHECK(agc_present_footprint_of(1000, 3, &footprint));
   CHECK_EQ(footprint.row_pitch, 4096);
   CHECK_EQ(footprint.required_bytes, 4096u * 2u + 4000u);
   CHECK_EQ(footprint.slot_bytes, 65536);
   CHECK(!agc_present_footprint_of(0, 1, &footprint));
   CHECK(!agc_present_footprint_of(1, 16385, &footprint));

   CHECK(agc_present_canvas_fit(1920, 1080, 1280, 1024, &canvas, error, sizeof(error)));
   CHECK(agc_present_footprint_of(canvas.width, canvas.height, &footprint));
   CHECK_EQ(agc_present_image_offset(&canvas, &footprint), 228u * 7680u);

   CHECK_EQ(agc_present_frame_slot(1), 1);
   CHECK_EQ(agc_present_frame_slot(2), 2);
   CHECK_EQ(agc_present_frame_slot(3), 0);
   CHECK_EQ(agc_present_frame_slot(4), 1);
}

static void
test_packets(void)
{
   uint32_t words[32];
   memset(words, 0x5a, sizeof(words));
   uint32_t *end = agc_present_emit_acquire(words, agc_present_gcr_before());
   CHECK_EQ(end - words, AGC_PRESENT_ACQUIRE_DWORDS);
   CHECK_EQ(words[0], PKT3(PKT3_ACQUIRE_MEM, 6, 0));
   CHECK_EQ(words[1], 0);
   CHECK_EQ(words[2], 0xffffffffu);
   CHECK_EQ(words[3], 0xff);
   CHECK_EQ(words[6], 0xa);
   CHECK_EQ(words[7], 0xc381);
   CHECK(G_587_GL2_INV(agc_present_gcr_before()) && G_587_GL2_WB(agc_present_gcr_before()));
   CHECK_EQ(agc_present_gcr_after(), 0xc330);

   end = agc_present_emit_partial_flush(words);
   CHECK_EQ(end - words, AGC_PRESENT_PARTIAL_FLUSH_DWORDS);
   CHECK_EQ(words[0], PKT3(PKT3_EVENT_WRITE, 0, 0));
   CHECK_EQ(words[1], 0x407);

   CHECK_EQ(agc_present_padding(16, 8), 0);
   CHECK_EQ(agc_present_padding(13, 8), 3);
   CHECK_EQ(agc_present_padding(15, 8), 9);
   CHECK_EQ(agc_present_padding(9, 8), 7);
   CHECK_EQ(agc_present_padding(5, 1), 0);
   CHECK_EQ(agc_present_padding(5, 0), 0);

   memset(words, 0x5a, sizeof(words));
   end = agc_present_emit_padding(words, 3);
   CHECK_EQ(end - words, 3);
   CHECK_EQ(words[0], PKT3(PKT3_NOP, 1, 0));
   CHECK_EQ(words[1], 0);
   CHECK_EQ(words[2], 0);
   CHECK_EQ(words[3], 0x5a5a5a5a);
   end = agc_present_emit_padding(words, 0);
   CHECK(end == words);
   end = agc_present_emit_padding(words, 2);
   CHECK_EQ(end - words, 2);
   CHECK_EQ(words[0], PKT3(PKT3_NOP, 0, 0));
   CHECK_EQ(words[1], 0);
}

int
main(void)
{
   test_formats();
   test_sizes();
   test_sources();
   test_descriptors();
   test_dcc_descriptors();
   test_geometry();
   test_packets();
   return TEST_RESULT();
}
