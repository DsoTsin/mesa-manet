#include <stdlib.h>
#include <string.h>

#include "agc_surface.h"
#include "test.h"

#define GUEST_ARENA UINT64_C(0xfc00000000)

static struct agc_surface_desc
desc_2d(uint32_t mode, uint32_t width, uint32_t height, uint32_t levels, uint32_t bpe)
{
   struct agc_surface_desc desc;
   memset(&desc, 0, sizeof(desc));
   desc.swizzle_mode = mode;
   desc.width = width;
   desc.height = height;
   desc.depth = 1;
   desc.layers = 1;
   desc.mip_levels = levels;
   desc.bytes_per_element = bpe;
   return desc;
}

static struct agc_surface_desc
desc_3d(uint32_t mode, uint32_t width, uint32_t height, uint32_t depth, uint32_t levels, uint32_t bpe)
{
   struct agc_surface_desc desc = desc_2d(mode, width, height, levels, bpe);
   desc.depth = depth;
   desc.volume = true;
   return desc;
}

static bool
layout_of(const struct agc_surface_desc *desc, enum agc_surface_addressing addressing,
          struct agc_surface_layout *layout)
{
   char error[512] = "";
   const bool ok = agc_surface_layout_init(desc, addressing, layout, error, sizeof(error));
   if (!ok)
      fprintf(stderr, "layout of mode %u: %s\n", desc->swizzle_mode, error);
   return ok;
}

static bool
layout_fails(const struct agc_surface_desc *desc, enum agc_surface_addressing addressing, const char *text)
{
   struct agc_surface_layout layout;
   char error[512] = "";
   if (agc_surface_layout_init(desc, addressing, &layout, error, sizeof(error)))
      return false;
   if (!strstr(error, text)) {
      fprintf(stderr, "layout error \"%s\" lacks \"%s\"\n", error, text);
      return false;
   }
   return layout.mip_count == 0 && layout.size == 0;
}

static uint64_t
offset_of(const struct agc_surface_layout *layout, uint32_t x, uint32_t y, uint32_t slice, uint32_t mip)
{
   uint64_t offset = 0;
   char error[512] = "";
   if (!agc_surface_element_offset(layout, x, y, slice, mip, &offset, error, sizeof(error))) {
      fprintf(stderr, "element offset: %s\n", error);
      return UINT64_MAX;
   }
   return offset;
}

static uint32_t
parity(uint32_t value)
{
   uint32_t bits = 0;
   while (value) {
      bits ^= value & 1;
      value >>= 1;
   }
   return bits;
}

static uint64_t
kernel_address(const uint32_t *layout, uint32_t x, uint32_t y, uint32_t slice)
{
   const uint32_t coordinates =
      ((layout[8] + x) & 0xfff) | (((y + layout[9]) & 0xfff) << 12) | ((slice & 0xff) << 24);
   uint32_t offset = 0;
   for (unsigned bit = 0; bit < 16; bit++)
      offset |= parity(layout[16 + bit] & coordinates) << bit;
   offset ^= layout[10];
   offset += ((x >> layout[5]) + (y >> layout[6]) * layout[4]) << layout[7];
   const uint64_t base = (uint64_t)layout[0] | (uint64_t)layout[1] << 32;
   const uint64_t stride = (uint64_t)layout[2] | (uint64_t)layout[3] << 32;
   return base + slice * stride + offset;
}

static void
check_copy_addresses(const struct agc_surface_layout *from, uint64_t from_base, const struct agc_surface_layout *to,
                     uint64_t to_base)
{
   for (uint32_t level = 0; level < from->mip_count; level++) {
      uint32_t params[AGC_SURFACE_COPY_PARAM_DWORDS];
      uint32_t groups[3];
      char error[512] = "";
      if (!agc_surface_copy_params(from, from_base, to, to_base, level, params, groups, error, sizeof(error))) {
         fprintf(stderr, "copy params of level %u: %s\n", level, error);
         CHECK(false);
         continue;
      }
      uint32_t extent[3];
      agc_surface_level_extent(from, level, extent);
      CHECK_EQ(params[64], extent[0]);
      CHECK_EQ(params[65], extent[1]);
      CHECK_EQ(1u << params[66], from->desc.bytes_per_element);
      CHECK_EQ(params[67], 0);
      CHECK_EQ(groups[0], (extent[0] + 63) / 64);
      CHECK_EQ(groups[1], extent[1]);
      CHECK_EQ(groups[2], extent[2]);
      unsigned mismatches = 0;
      for (uint32_t slice = 0; slice < groups[2]; slice++) {
         for (uint32_t y = 0; y < params[65]; y++) {
            for (uint32_t x = 0; x < params[64]; x++) {
               if (kernel_address(params, x, y, slice) != from_base + offset_of(from, x, y, slice, level))
                  mismatches++;
               if (kernel_address(&params[32], x, y, slice) != to_base + offset_of(to, x, y, slice, level))
                  mismatches++;
            }
         }
      }
      if (mismatches)
         fprintf(stderr, "level %u: %u kernel addresses differ from the layouts\n", level, mismatches);
      CHECK_EQ(mismatches, 0);
   }
}

static void
check_block_bijection(uint32_t mode, uint32_t bpe, enum agc_surface_addressing addressing, bool volume)
{
   struct agc_surface_desc desc = volume ? desc_3d(mode, 1, 1, 1, 1, bpe) : desc_2d(mode, 1, 1, 1, bpe);
   struct agc_surface_layout probe;
   struct agc_surface_layout layout;
   if (!layout_of(&desc, addressing, &probe)) {
      CHECK(false);
      return;
   }
   desc.width = probe.block[0];
   desc.height = probe.block[1];
   if (volume)
      desc.depth = probe.block[2];
   if (!layout_of(&desc, addressing, &layout)) {
      CHECK(false);
      return;
   }
   CHECK_EQ(layout.size, layout.block_bytes);
   const uint64_t elements = layout.block_bytes / bpe;
   CHECK_EQ(elements, (uint64_t)layout.block[0] * layout.block[1] * layout.block[2]);
   uint8_t *seen = calloc((size_t)elements, 1);
   unsigned bad = 0;
   for (uint32_t z = 0; z < layout.block[2]; z++) {
      for (uint32_t y = 0; y < layout.block[1]; y++) {
         for (uint32_t x = 0; x < layout.block[0]; x++) {
            const uint64_t offset = offset_of(&layout, x, y, z, 0);
            if (offset % bpe || offset >= layout.block_bytes || seen[offset / bpe]++)
               bad++;
         }
      }
   }
   free(seen);
   if (bad)
      fprintf(stderr, "mode %u, %u-byte elements, %s%s: %u elements collide\n", mode, bpe,
              addressing == AGC_SURFACE_HOST ? "host" : "guest", volume ? " volume" : "", bad);
   CHECK_EQ(bad, 0);
}

static void
test_texel_formats(void)
{
   static const struct {
      uint32_t format;
      uint32_t bytes;
      uint32_t block;
   } cases[] = {
      {1, 1, 1},     {6, 1, 1},     {7, 2, 1},    {13, 2, 1},    {14, 2, 1},    {19, 2, 1},    {20, 4, 1},
      {29, 4, 1},    {30, 4, 1},    {36, 4, 1},   {43, 4, 1},    {44, 4, 1},    {55, 4, 1},    {56, 4, 1},
      {61, 4, 1},    {62, 8, 1},    {64, 8, 1},   {65, 8, 1},    {71, 8, 1},    {72, 12, 1},   {74, 12, 1},
      {75, 16, 1},   {77, 16, 1},   {128, 1, 1},  {129, 2, 1},   {130, 4, 1},   {132, 4, 1},   {133, 2, 1},
      {134, 2, 1},   {135, 2, 1},   {136, 2, 1},  {137, 1, 1},   {140, 4, 1},   {141, 4, 1},   {142, 4, 1},
      {143, 4, 1},   {144, 4, 1},   {145, 8, 1},  {146, 8, 1},   {169, 8, 4},   {170, 8, 4},   {171, 16, 4},
      {172, 16, 4},  {173, 16, 4},  {174, 16, 4}, {175, 8, 4},   {176, 8, 4},   {177, 16, 4},  {178, 16, 4},
      {179, 16, 4},  {180, 16, 4},  {181, 16, 4}, {182, 16, 4},
   };
   for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
      struct agc_texel_format texel;
      char error[256] = "";
      CHECK(agc_texel_format_from_gfx10(cases[i].format, &texel, error, sizeof(error)));
      CHECK_EQ(texel.bytes_per_element, cases[i].bytes);
      CHECK_EQ(texel.block_width, cases[i].block);
      CHECK_EQ(texel.block_height, cases[i].block);
   }
   static const struct {
      uint32_t format;
      const char *kind;
   } unsupported[] = {
      {0, "(INVALID)"},   {78, "(reserved)"},  {131, "(6E4_FLOAT)"},  {138, "(1-bit)"},   {139, "(1-bit)"},
      {147, "(subsampled"}, {154, "(subsampled"}, {155, "(reserved)"}, {156, "(FMASK)"}, {168, "(FMASK)"},
      {183, "(reserved)"}, {265, "(multimedia)"}, {284, "(multimedia)"}, {285, "(7E3_FLOAT)"}, {286, "(YCBCR)"},
      {292, "(YCBCR)"},   {511, "(reserved)"},
   };
   for (size_t i = 0; i < sizeof(unsupported) / sizeof(unsupported[0]); i++) {
      struct agc_texel_format texel;
      char error[256] = "";
      CHECK(!agc_texel_format_from_gfx10(unsupported[i].format, &texel, error, sizeof(error)));
      CHECK(strstr(error, "is not supported") != NULL);
      CHECK(strstr(error, unsupported[i].kind) != NULL);
      CHECK_EQ(texel.bytes_per_element, 0);
   }
   unsigned supported = 0;
   for (uint32_t format = 0; format < 512; format++) {
      struct agc_texel_format texel;
      if (agc_texel_format_from_gfx10(format, &texel, NULL, 0))
         supported++;
   }
   CHECK_EQ(supported, 77 + 3 + 6 + 7 + 14);
}

static void
test_host_swizzle_modes(void)
{
   CHECK_EQ(agc_surface_type_host_swizzle_mode(9, 9), AGC_SW_64KB_D);
   CHECK_EQ(agc_surface_type_host_swizzle_mode(9, 25), AGC_SW_64KB_D_X);
   CHECK_EQ(agc_surface_type_host_swizzle_mode(9, 22), AGC_SW_4KB_D_X);
   CHECK_EQ(agc_surface_type_host_swizzle_mode(10, 25), AGC_SW_64KB_S_X);
   CHECK_EQ(agc_surface_type_host_swizzle_mode(10, 5), AGC_SW_4KB_S);
   CHECK_EQ(agc_surface_type_host_swizzle_mode(10, 9), AGC_SW_64KB_S);
   CHECK(!agc_surface_samples_in_place_for_format(10, 5, 56));
   CHECK(!agc_surface_samples_in_place_for_format(10, 9, 56));
   CHECK(agc_surface_samples_in_place_for_format(10, 0, 56));

   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_LINEAR, AGC_SURFACE_USE_THIN_2D), AGC_SW_LINEAR);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_256B_S, AGC_SURFACE_USE_THIN_2D), AGC_SW_256B_D);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_4KB_S, AGC_SURFACE_USE_THIN_2D), AGC_SW_4KB_D);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_S, AGC_SURFACE_USE_THIN_2D), AGC_SW_64KB_D);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_S_T, AGC_SURFACE_USE_THIN_2D), AGC_SW_64KB_D_T);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_4KB_S_X, AGC_SURFACE_USE_THIN_2D), AGC_SW_4KB_D_X);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_S_X, AGC_SURFACE_USE_THIN_2D), AGC_SW_64KB_D_X);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_D, AGC_SURFACE_USE_THIN_2D), AGC_SW_64KB_D);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_Z_X, AGC_SURFACE_USE_THIN_2D), AGC_SW_64KB_Z_X);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_R_X, AGC_SURFACE_USE_THIN_2D), AGC_SW_64KB_R_X);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_S_X | 0x20, AGC_SURFACE_USE_THIN_2D), AGC_SW_64KB_D_X);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_S_X, AGC_SURFACE_USE_1D), AGC_SW_64KB_R_X);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_Z_X, AGC_SURFACE_USE_MSAA), AGC_SW_64KB_Z_X);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_4KB_D, AGC_SURFACE_USE_MSAA), AGC_SW_64KB_R_X);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_LINEAR, AGC_SURFACE_USE_MSAA), AGC_SW_LINEAR);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_S_X, AGC_SURFACE_USE_VOLUME), AGC_SW_64KB_S_X);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_D, AGC_SURFACE_USE_VOLUME), AGC_SW_64KB_S);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_256B_D, AGC_SURFACE_USE_VOLUME), AGC_SW_64KB_S);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_LINEAR, AGC_SURFACE_USE_DEPTH), AGC_SW_64KB_Z_X);
   CHECK_EQ(agc_surface_host_swizzle_mode(AGC_SW_64KB_S, AGC_SURFACE_USE_DEPTH), AGC_SW_64KB_Z_X);
   CHECK_EQ(agc_surface_type_host_swizzle_mode(12, AGC_SW_64KB_D), AGC_SW_64KB_R_X);
   CHECK_EQ(agc_surface_type_host_swizzle_mode(15, AGC_SW_64KB_Z_X), AGC_SW_64KB_Z_X);
   CHECK_EQ(agc_surface_type_host_swizzle_mode(13, AGC_SW_4KB_S), AGC_SW_4KB_D);
   CHECK_EQ(agc_surface_type_host_swizzle_mode(11, AGC_SW_64KB_S_X), AGC_SW_64KB_D_X);
   CHECK_EQ(agc_surface_type_host_swizzle_mode(10, 28), 28);
   CHECK_EQ(agc_surface_type_host_swizzle_mode(10, 30), AGC_SW_64KB_S);
}

static void
test_direct_d_views(void)
{
   static const uint32_t types[] = {9, 11, 13};
   static const uint32_t formats[] = {56, 62, 65, 71, 72, 75, 77};
   for (size_t t = 0; t < 3; t++) {
      for (size_t f = 0; f < sizeof(formats) / sizeof(formats[0]); f++) {
         const uint32_t format = formats[f];
         const bool expected = (format >= 62 && format <= 71) || (format >= 75 && format <= 77);
         uint32_t words[8] = {0};
         words[0] = 0x10000000;
         words[1] = format << 20;
         words[3] = (types[t] << 28) | (10u << 20) | 0xfac;
         words[5] = (1u << 26) | (0x123u << 8);
         struct agc_image_view view;
         char error[256] = "";
         CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
         CHECK_EQ(agc_image_view_samples_in_place(&view), expected);
         CHECK_EQ(agc_surface_samples_in_place_for_format(types[t], AGC_SW_64KB_D, format), expected);
         CHECK_EQ(agc_image_view_host_swizzle_mode(&view), AGC_SW_64KB_D);
      }
   }
   CHECK(!agc_surface_samples_in_place_for_format(10, AGC_SW_64KB_D, 65));
   CHECK(!agc_surface_samples_in_place_for_format(9, AGC_SW_64KB_D_X, 65));
   CHECK(agc_surface_samples_in_place_for_format(14, AGC_SW_LINEAR, 0));

   for (uint32_t format = 62; format <= 77; format++) {
      struct agc_texel_format texel;
      if (format >= 72 && format <= 74)
         continue;
      CHECK(agc_texel_format_from_gfx10(format, &texel, NULL, 0));
      const struct agc_surface_desc desc = desc_2d(AGC_SW_64KB_D, 300, 200, 9, texel.bytes_per_element);
      struct agc_surface_layout guest;
      struct agc_surface_layout host;
      CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &guest));
      CHECK(layout_of(&desc, AGC_SURFACE_HOST, &host));
      CHECK(agc_surface_same_addressing(&guest, &host));
   }
   const struct agc_surface_desc rgba8 = desc_2d(AGC_SW_64KB_D, 300, 200, 9, 4);
   struct agc_surface_layout guest;
   struct agc_surface_layout host;
   CHECK(layout_of(&rgba8, AGC_SURFACE_GUEST, &guest));
   CHECK(layout_of(&rgba8, AGC_SURFACE_HOST, &host));
   CHECK(!agc_surface_same_addressing(&guest, &host));
   CHECK(agc_surface_same_placement(&guest, &host));
}

static void
test_pipe_bank_xor(void)
{
   for (unsigned i = 0; i < 2; i++) {
      const uint32_t mode = i ? AGC_SW_64KB_Z_X : AGC_SW_64KB_R_X;
      struct agc_surface_desc desc = desc_2d(mode, 1920, 1080, 1, 4);
      struct agc_surface_layout layout;
      desc.pipe_bank_xor = 0x5a;
      CHECK(layout_of(&desc, AGC_SURFACE_HOST, &layout));
      CHECK_EQ(layout.xor_bytes, 0x42u << 8);
      desc.pipe_bank_xor = 0x18;
      CHECK(layout_of(&desc, AGC_SURFACE_HOST, &layout));
      CHECK_EQ(layout.xor_bytes, 0);
      desc.pipe_bank_xor = 0x5a;
      CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &layout));
      CHECK_EQ(layout.xor_bytes, 0x5au << 8);
   }
   CHECK_EQ(agc_surface_host_pipe_bank_xor(16, 0xff), 0xe7);
   CHECK_EQ(agc_surface_host_pipe_bank_xor(16, 0xcf), 0xc7);
   CHECK_EQ(agc_surface_host_pipe_bank_xor(12, 0x0f), 0x07);
   CHECK_EQ(agc_surface_host_pipe_bank_xor(8, 0xff), 0x07);
   CHECK_EQ(agc_surface_host_pipe_bank_xor(18, 0x3ff), 0x1e7);
   CHECK_EQ(agc_surface_host_pipe_bank_xor(20, 0xffff), 0x1e7);

   static const struct {
      uint64_t base;
      uint32_t mode;
      uint32_t xor_bits;
   } cases[] = {
      {UINT64_C(0x1234560000), 25, 0xcf},
      {UINT64_C(0x1234560000), 24, 0x08},
      {UINT64_C(0x1234560000), 17, 0x45},
      {UINT64_C(0x1234567000), 21, 0x0f},
   };
   for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
      uint32_t words[8] = {0};
      words[0] = (uint32_t)(cases[i].base >> 8) | cases[i].xor_bits;
      words[1] = (uint32_t)(cases[i].base >> 40) | (56u << 20);
      words[3] = (cases[i].mode << 20) | (9u << 28);
      struct agc_image_view view;
      char error[256] = "";
      CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
      CHECK_EQ(view.base, cases[i].base);
      CHECK_EQ(view.pipe_bank_xor, cases[i].xor_bits);
      uint64_t base;
      uint32_t pipe_bank_xor;
      agc_surface_split_pipe_bank_xor(cases[i].mode, cases[i].base | (uint64_t)cases[i].xor_bits << 8, &base,
                                      &pipe_bank_xor);
      CHECK_EQ(base, cases[i].base);
      CHECK_EQ(pipe_bank_xor, cases[i].xor_bits);
   }
   uint64_t base;
   uint32_t pipe_bank_xor;
   agc_surface_split_pipe_bank_xor(AGC_SW_LINEAR, UINT64_C(0x1234567800), &base, &pipe_bank_xor);
   CHECK_EQ(base, UINT64_C(0x1234567800));
   CHECK_EQ(pipe_bank_xor, 0);
   agc_surface_split_pipe_bank_xor(AGC_SW_4KB_S, UINT64_C(0x1234567000), &base, &pipe_bank_xor);
   CHECK_EQ(base, UINT64_C(0x1234567000));
   CHECK_EQ(pipe_bank_xor, 0);
   agc_surface_split_pipe_bank_xor(20, UINT64_C(0x1234567f00), &base, &pipe_bank_xor);
   CHECK_EQ(base, UINT64_C(0x1234567f00));
   CHECK_EQ(pipe_bank_xor, 0);

   struct agc_surface_desc desc = desc_2d(AGC_SW_64KB_S_X, 256, 256, 9, 4);
   struct agc_surface_layout plain;
   struct agc_surface_layout xored;
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &plain));
   desc.pipe_bank_xor = 0x5;
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &xored));
   CHECK_EQ(offset_of(&xored, 1, 0, 0, 0), 131072 + (4 ^ 0x500));
   CHECK_EQ(offset_of(&xored, 1, 2, 0, 3), offset_of(&plain, 1, 2, 0, 3) ^ 0x500);
   CHECK(!agc_surface_same_addressing(&plain, &xored));
   CHECK(agc_surface_same_placement(&plain, &xored));

   desc = desc_2d(AGC_SW_64KB_D_X, 256, 256, 1, 4);
   desc.pipe_bank_xor = 0x10;
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "sets the column bits of an SW_64KB_D_X block"));
   CHECK(layout_of(&desc, AGC_SURFACE_HOST, &plain));
   CHECK_EQ(plain.xor_bytes, 0);
   desc = desc_2d(AGC_SW_64KB_S, 256, 256, 1, 4);
   desc.pipe_bank_xor = 1;
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "swizzle mode 9 (SW_64KB_S) has no pipe/bank XOR, got 0x1"));
   desc = desc_2d(AGC_SW_4KB_S_X, 256, 256, 1, 4);
   desc.pipe_bank_xor = 0x10;
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "pipe/bank XOR 0x10 does not fit a 4096-byte block"));
   desc.pipe_bank_xor = 0x0f;
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &plain));
   CHECK_EQ(plain.xor_bytes, 0xf00);
}

static void
test_block_extents(void)
{
   static const uint32_t thin[3][5][2] = {
      {{16, 16}, {16, 8}, {8, 8}, {8, 4}, {4, 4}},
      {{64, 64}, {64, 32}, {32, 32}, {32, 16}, {16, 16}},
      {{256, 256}, {256, 128}, {128, 128}, {128, 64}, {64, 64}},
   };
   static const uint32_t modes[3] = {AGC_SW_256B_D, AGC_SW_4KB_D_X, AGC_SW_64KB_R_X};
   static const uint64_t bytes[3] = {256, 4096, 65536};
   for (unsigned size = 0; size < 3; size++) {
      for (unsigned log2 = 0; log2 < 5; log2++) {
         const struct agc_surface_desc desc = desc_2d(modes[size], 1, 1, 1, 1u << log2);
         struct agc_surface_layout guest;
         struct agc_surface_layout host;
         CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &guest));
         CHECK(layout_of(&desc, AGC_SURFACE_HOST, &host));
         CHECK_EQ(guest.tiling, AGC_SURFACE_TILING_THIN);
         CHECK_EQ(guest.block_bytes, bytes[size]);
         CHECK_EQ(guest.block[0], thin[size][log2][0]);
         CHECK_EQ(guest.block[1], thin[size][log2][1]);
         CHECK_EQ(guest.block[2], 1);
         CHECK_EQ(host.block[0], thin[size][log2][0]);
         CHECK_EQ(host.block[1], thin[size][log2][1]);
         CHECK_EQ(guest.size, bytes[size]);
      }
   }
   static const uint32_t thick[2][5][3] = {
      {{16, 16, 16}, {8, 16, 16}, {8, 16, 8}, {8, 8, 8}, {4, 8, 8}},
      {{64, 32, 32}, {32, 32, 32}, {32, 32, 16}, {32, 16, 16}, {16, 16, 16}},
   };
   for (unsigned size = 0; size < 2; size++) {
      for (unsigned log2 = 0; log2 < 5; log2++) {
         const struct agc_surface_desc desc =
            desc_3d(size ? AGC_SW_64KB_S_X : AGC_SW_4KB_S, 1, 1, 1, 1, 1u << log2);
         struct agc_surface_layout layout;
         CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &layout));
         CHECK_EQ(layout.tiling, AGC_SURFACE_TILING_THICK);
         for (unsigned i = 0; i < 3; i++)
            CHECK_EQ(layout.block[i], thick[size][log2][i]);
      }
   }
   const struct agc_surface_desc linear = desc_2d(AGC_SW_LINEAR, 7, 3, 1, 4);
   struct agc_surface_layout layout;
   CHECK(layout_of(&linear, AGC_SURFACE_GUEST, &layout));
   CHECK_EQ(layout.tiling, AGC_SURFACE_TILING_LINEAR);
   CHECK_EQ(layout.block_bytes, 0);
   CHECK_EQ(layout.block[0], 1);
   CHECK_EQ(layout.block[1], 1);
   CHECK_EQ(layout.block[2], 1);
   const struct agc_surface_desc volume = desc_3d(AGC_SW_64KB_R_X, 64, 64, 4, 1, 4);
   CHECK(layout_of(&volume, AGC_SURFACE_GUEST, &layout));
   CHECK_EQ(layout.tiling, AGC_SURFACE_TILING_THIN);
   CHECK_EQ(layout.slice_stride, 65536);
   CHECK_EQ(layout.size, 4 * 65536);
}

static void
test_bijective_blocks(void)
{
   static const uint32_t guest_modes[] = {
      AGC_SW_256B_S,   AGC_SW_256B_D,  AGC_SW_4KB_S,    AGC_SW_4KB_D,    AGC_SW_64KB_S,
      AGC_SW_64KB_D,   AGC_SW_64KB_S_T, AGC_SW_64KB_D_T, AGC_SW_4KB_S_X,  AGC_SW_4KB_D_X,
      AGC_SW_64KB_Z_X, AGC_SW_64KB_S_X, AGC_SW_64KB_D_X, AGC_SW_64KB_R_X,
   };
   static const uint32_t host_modes[] = {
      AGC_SW_256B_D,   AGC_SW_4KB_D,    AGC_SW_64KB_D,   AGC_SW_64KB_D_T,
      AGC_SW_4KB_D_X,  AGC_SW_64KB_Z_X, AGC_SW_64KB_D_X, AGC_SW_64KB_R_X,
   };
   static const uint32_t thick_modes[] = {AGC_SW_4KB_S, AGC_SW_4KB_S_X, AGC_SW_64KB_S, AGC_SW_64KB_S_X,
                                          AGC_SW_64KB_D_X};
   for (uint32_t bpe = 1; bpe <= 16; bpe <<= 1) {
      for (size_t i = 0; i < sizeof(guest_modes) / sizeof(guest_modes[0]); i++)
         check_block_bijection(guest_modes[i], bpe, AGC_SURFACE_GUEST, false);
      for (size_t i = 0; i < sizeof(host_modes) / sizeof(host_modes[0]); i++) {
         if (host_modes[i] != AGC_SW_64KB_Z_X || bpe != 16)
            check_block_bijection(host_modes[i], bpe, AGC_SURFACE_HOST, false);
      }
      for (size_t i = 0; i < sizeof(thick_modes) / sizeof(thick_modes[0]); i++)
         check_block_bijection(thick_modes[i], bpe, AGC_SURFACE_GUEST, true);
   }
}

static void
test_equations(void)
{
   const uint32_t *equation = agc_surface_guest_equation(AGC_SW_64KB_Z_X, false, 0);
   CHECK(equation != NULL);
   CHECK_EQ(equation[8], 0x08008008);
   equation = agc_surface_guest_equation(AGC_SW_64KB_D_X, true, 4);
   CHECK(equation != NULL);
   CHECK_EQ(equation[9], 0x00010012);
   equation = agc_surface_host_equation(AGC_SW_64KB_R_X, 4);
   CHECK(equation != NULL);
   CHECK_EQ(equation[15], 0x01040020);
   equation = agc_surface_host_equation(AGC_SW_4KB_D_X, 2);
   CHECK(equation != NULL);
   CHECK_EQ(equation[8], 0x04008020);
   CHECK(agc_surface_host_equation(AGC_SW_64KB_Z_X, 4) == NULL);
   CHECK(agc_surface_host_equation(AGC_SW_64KB_S_X, 2) == NULL);
   CHECK(agc_surface_host_equation(AGC_SW_64KB_D, 5) == NULL);
   CHECK(agc_surface_guest_equation(AGC_SW_64KB_S, false, 2) == NULL);
   CHECK(agc_surface_guest_equation(AGC_SW_64KB_D, true, 2) == NULL);

   static const uint32_t standard_64kb_4[16] = {0,       0,       0x1,     0x2,    0x1000,  0x2000,
                                                0x4000,  0x4,     0x8000,  0x8,    0x10000, 0x10,
                                                0x20000, 0x20,    0x40000, 0x40};
   const struct agc_surface_desc desc = desc_2d(AGC_SW_64KB_S, 256, 256, 1, 4);
   struct agc_surface_layout layout;
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &layout));
   for (unsigned bit = 0; bit < 16; bit++)
      CHECK_EQ(layout.equation[bit], standard_64kb_4[bit]);
   static const uint32_t standard_256b_1[16] = {0x1, 0x2, 0x4, 0x8, 0x1000, 0x2000, 0x4000, 0x8000};
   const struct agc_surface_desc small = desc_2d(AGC_SW_256B_S, 16, 16, 1, 1);
   CHECK(layout_of(&small, AGC_SURFACE_GUEST, &layout));
   for (unsigned bit = 0; bit < 16; bit++)
      CHECK_EQ(layout.equation[bit], standard_256b_1[bit]);

   const struct agc_surface_desc table = desc_2d(AGC_SW_64KB_S_X, 256, 256, 1, 8);
   CHECK(layout_of(&table, AGC_SURFACE_GUEST, &layout));
   equation = agc_surface_guest_equation(AGC_SW_64KB_S_X, false, 3);
   for (unsigned bit = 0; bit < 16; bit++)
      CHECK_EQ(layout.equation[bit], equation[bit]);
   const struct agc_surface_desc linear = desc_2d(AGC_SW_LINEAR, 256, 256, 1, 8);
   CHECK(layout_of(&linear, AGC_SURFACE_GUEST, &layout));
   for (unsigned bit = 0; bit < 16; bit++)
      CHECK_EQ(layout.equation[bit], 0);
}

static void
check_mip(const struct agc_surface_layout *layout, uint32_t level, uint64_t offset, uint64_t size, uint32_t pitch,
          bool in_tail, uint32_t tail_x, uint32_t tail_y)
{
   const struct agc_surface_mip *mip = &layout->mips[level];
   CHECK_EQ(mip->offset, offset);
   CHECK_EQ(mip->size, size);
   CHECK_EQ(mip->pitch, pitch);
   CHECK_EQ(mip->in_tail, in_tail);
   CHECK_EQ(mip->tail_origin[0], tail_x);
   CHECK_EQ(mip->tail_origin[1], tail_y);
}

static void
test_mip_chains(void)
{
   const struct agc_surface_desc desc = desc_2d(AGC_SW_64KB_S_X, 256, 256, 9, 4);
   struct agc_surface_layout layout;
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &layout));
   CHECK_EQ(layout.mip_count, 9);
   CHECK_EQ(layout.block_bytes, 65536);
   check_mip(&layout, 0, 131072, 262144, 2, false, 0, 0);
   check_mip(&layout, 1, 65536, 65536, 1, false, 0, 0);
   check_mip(&layout, 2, 0, 65536, 1, true, 64, 0);
   check_mip(&layout, 3, 0, 65536, 1, true, 0, 64);
   check_mip(&layout, 4, 0, 65536, 1, true, 32, 0);
   check_mip(&layout, 5, 0, 65536, 1, true, 0, 32);
   check_mip(&layout, 6, 0, 65536, 1, true, 16, 0);
   check_mip(&layout, 7, 0, 65536, 1, true, 8, 16);
   check_mip(&layout, 8, 0, 65536, 1, true, 0, 24);
   CHECK_EQ(layout.slice_stride, 393216);
   CHECK_EQ(layout.size, 393216);
   struct agc_surface_desc array = desc;
   array.layers = 6;
   CHECK(layout_of(&array, AGC_SURFACE_GUEST, &layout));
   CHECK_EQ(layout.slice_stride, 393216);
   CHECK_EQ(layout.size, 6 * 393216);
   CHECK_EQ(offset_of(&layout, 0, 0, 5, 1), 5 * 393216 + 65536 + (0x800 | 0x200));
   CHECK_EQ(offset_of(&layout, 0, 0, 4, 1), 4 * 393216 + 65536 + 0x200);

   const struct agc_surface_desc small = desc_2d(AGC_SW_4KB_S_X, 64, 64, 7, 4);
   CHECK(layout_of(&small, AGC_SURFACE_GUEST, &layout));
   check_mip(&layout, 0, 8192, 16384, 2, false, 0, 0);
   check_mip(&layout, 1, 4096, 4096, 1, false, 0, 0);
   check_mip(&layout, 2, 0, 4096, 1, true, 16, 0);
   check_mip(&layout, 3, 0, 4096, 1, true, 8, 16);
   check_mip(&layout, 4, 0, 4096, 1, true, 0, 24);
   check_mip(&layout, 5, 0, 4096, 1, true, 0, 16);
   check_mip(&layout, 6, 0, 4096, 1, true, 8, 8);
   CHECK_EQ(layout.slice_stride, 24576);

   const struct agc_surface_desc color = desc_2d(AGC_SW_64KB_R_X, 256, 256, 9, 2);
   CHECK(layout_of(&color, AGC_SURFACE_GUEST, &layout));
   check_mip(&layout, 0, 65536, 131072, 1, false, 0, 0);
   check_mip(&layout, 1, 0, 65536, 1, true, 128, 0);
   CHECK_EQ(layout.slice_stride, 196608);
   const struct agc_surface_desc depth = desc_2d(AGC_SW_64KB_Z_X, 256, 256, 9, 2);
   CHECK(layout_of(&depth, AGC_SURFACE_GUEST, &layout));
   check_mip(&layout, 0, 131072, 131072, 1, false, 0, 0);
   check_mip(&layout, 1, 65536, 65536, 1, false, 0, 0);
   check_mip(&layout, 2, 0, 65536, 1, true, 128, 0);
   CHECK_EQ(layout.slice_stride, 262144);

   const struct agc_surface_desc single = desc_2d(AGC_SW_64KB_S_X, 32, 32, 1, 4);
   CHECK(layout_of(&single, AGC_SURFACE_GUEST, &layout));
   check_mip(&layout, 0, 0, 65536, 1, false, 0, 0);
   const struct agc_surface_desc tiny = desc_2d(AGC_SW_256B_D, 64, 64, 7, 4);
   CHECK(layout_of(&tiny, AGC_SURFACE_GUEST, &layout));
   CHECK(!layout.mips[6].in_tail);
   check_mip(&layout, 6, 0, 256, 1, false, 0, 0);
   check_mip(&layout, 2, 1024, 1024, 2, false, 0, 0);
   check_mip(&layout, 1, 2048, 4096, 4, false, 0, 0);
   check_mip(&layout, 0, 6144, 16384, 8, false, 0, 0);
}

static void
test_linear(void)
{
   const struct agc_surface_desc desc = desc_2d(AGC_SW_LINEAR, 100, 50, 3, 4);
   struct agc_surface_layout layout;
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &layout));
   check_mip(&layout, 2, 0, 3328, 64, false, 0, 0);
   check_mip(&layout, 1, 3328, 6400, 64, false, 0, 0);
   check_mip(&layout, 0, 9728, 25600, 128, false, 0, 0);
   CHECK_EQ(layout.slice_stride, 35328);
   CHECK_EQ(offset_of(&layout, 3, 2, 0, 0), 10764);
   CHECK_EQ(offset_of(&layout, 24, 12, 0, 2), (12 * 64 + 24) * 4);
   struct agc_surface_layout host;
   CHECK(layout_of(&desc, AGC_SURFACE_HOST, &host));
   CHECK(agc_surface_same_addressing(&layout, &host));

   const struct agc_surface_desc rgb32 = desc_2d(AGC_SW_LINEAR, 10, 3, 1, 12);
   CHECK(layout_of(&rgb32, AGC_SURFACE_GUEST, &layout));
   check_mip(&layout, 0, 0, 756, 21, false, 0, 0);
   CHECK_EQ(offset_of(&layout, 2, 1, 0, 0), 276);

   struct agc_surface_desc pitched = desc_2d(AGC_SW_LINEAR, 100, 4, 1, 4);
   pitched.pitch_elements = 128;
   CHECK(layout_of(&pitched, AGC_SURFACE_GUEST, &layout));
   check_mip(&layout, 0, 0, 2048, 128, false, 0, 0);
   CHECK_EQ(offset_of(&layout, 99, 3, 0, 0), (3 * 128 + 99) * 4);
   pitched.pitch_elements = 99;
   CHECK(layout_fails(&pitched, AGC_SURFACE_GUEST, "a row pitch of 99 elements"));
   pitched.pitch_elements = 128;
   pitched.mip_levels = 2;
   CHECK(layout_fails(&pitched, AGC_SURFACE_GUEST, "single-level linear surface"));
   pitched = desc_2d(AGC_SW_64KB_R_X, 100, 4, 1, 4);
   pitched.pitch_elements = 128;
   CHECK(layout_fails(&pitched, AGC_SURFACE_GUEST, "a row pitch of 128 elements"));

   const struct agc_surface_desc slices = desc_3d(AGC_SW_LINEAR, 16, 16, 5, 2, 4);
   CHECK(layout_of(&slices, AGC_SURFACE_GUEST, &layout));
   CHECK_EQ(layout.slice_stride, 64 * 16 * 4 + 64 * 8 * 4);
   CHECK_EQ(layout.size, 5 * layout.slice_stride);
   CHECK_EQ(offset_of(&layout, 1, 1, 1, 1), layout.slice_stride + (64 + 1) * 4);
   CHECK(layout_of(&slices, AGC_SURFACE_HOST, &host));
}

static void
test_offsets(void)
{
   const struct agc_surface_desc desc = desc_2d(AGC_SW_64KB_S_X, 256, 256, 9, 4);
   struct agc_surface_desc host_desc = desc;
   host_desc.swizzle_mode = agc_surface_host_swizzle_mode(desc.swizzle_mode, AGC_SURFACE_USE_THIN_2D);
   CHECK_EQ(host_desc.swizzle_mode, AGC_SW_64KB_D_X);
   struct agc_surface_layout guest;
   struct agc_surface_layout host;
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &guest));
   CHECK(layout_of(&host_desc, AGC_SURFACE_HOST, &host));
   CHECK(!agc_surface_same_addressing(&guest, &host));
   CHECK(agc_surface_same_placement(&guest, &host));
   CHECK_EQ(offset_of(&guest, 1, 0, 0, 0), 131076);
   CHECK_EQ(offset_of(&host, 1, 0, 0, 0), 131076);
   CHECK_EQ(offset_of(&guest, 0, 1, 0, 0), 131088);
   CHECK_EQ(offset_of(&host, 0, 1, 0, 0), 131080);
   CHECK_EQ(offset_of(&guest, 5, 3, 0, 0), 131252);
   CHECK_EQ(offset_of(&host, 5, 3, 0, 0), 131180);
   CHECK_EQ(offset_of(&guest, 100, 50, 0, 0), 178592);
   CHECK_EQ(offset_of(&host, 100, 50, 0, 0), 178016);
   CHECK_EQ(offset_of(&guest, 130, 129, 0, 0), 327704);
   CHECK_EQ(offset_of(&guest, 1, 2, 0, 3), 16932);
   CHECK_EQ(offset_of(&host, 1, 2, 0, 3), 16420);

   uint64_t offset = 7;
   char error[256] = "";
   CHECK(!agc_surface_element_offset(&guest, 0, 0, 0, 9, &offset, error, sizeof(error)));
   CHECK(strstr(error, "mip 9 is outside the 9-level surface") != NULL);
   CHECK_EQ(offset, 0);
   CHECK(!agc_surface_element_offset(&guest, 64, 0, 0, 2, &offset, error, sizeof(error)));
   CHECK(strstr(error, "element (64, 0) of slice 0 lies outside mip 2 (64x64, 1 slices)") != NULL);
   CHECK(!agc_surface_element_offset(&guest, 0, 0, 1, 0, &offset, error, sizeof(error)));

   const struct agc_surface_desc volume = desc_3d(AGC_SW_64KB_S_X, 64, 64, 64, 1, 4);
   struct agc_surface_layout thick;
   CHECK(layout_of(&volume, AGC_SURFACE_GUEST, &thick));
   CHECK_EQ(thick.tiling, AGC_SURFACE_TILING_THICK);
   CHECK_EQ(thick.slice_stride, 262144);
   CHECK_EQ(thick.size, 1048576);
   CHECK_EQ(offset_of(&thick, 0, 0, 16, 0), 262144 + 512);
   CHECK_EQ(offset_of(&thick, 1, 0, 1, 0), 20);
   const struct agc_surface_desc chain = desc_3d(AGC_SW_64KB_S_X, 64, 64, 64, 7, 4);
   CHECK(layout_of(&chain, AGC_SURFACE_GUEST, &thick));
   check_mip(&thick, 2, 0, 65536, 1, true, 16, 0);
   CHECK(!thick.mips[1].in_tail);
   CHECK(offset_of(&thick, 0, 0, 0, 6) < thick.size);
   CHECK(offset_of(&thick, 1, 1, 1, 5) < thick.size);
}

static void
test_layout_errors(void)
{
   struct agc_surface_desc desc = desc_2d(3, 64, 64, 1, 4);
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "swizzle mode 3 (SW_256B_R) is not a gfx10 swizzle mode"));
   desc.swizzle_mode = 40;
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "swizzle mode 40 (invalid)"));
   desc.swizzle_mode = AGC_SW_64KB_S;
   CHECK(layout_fails(&desc, AGC_SURFACE_HOST, "is not a host-layout mode"));
   desc.swizzle_mode = AGC_SW_64KB_S_T;
   CHECK(layout_fails(&desc, AGC_SURFACE_HOST, "swizzle mode 17 (SW_64KB_S_T) is not a host-layout mode"));
   desc = desc_3d(AGC_SW_64KB_R_X, 64, 64, 4, 1, 4);
   CHECK(layout_fails(&desc, AGC_SURFACE_HOST, "tiled host-layout volumes"));
   desc = desc_2d(AGC_SW_64KB_R_X, 0, 64, 1, 4);
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "surface extent 0x64 is outside 1..=16384"));
   desc.width = 16385;
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "surface extent 16385x64"));
   desc = desc_3d(AGC_SW_64KB_R_X, 64, 64, 4, 1, 4);
   desc.layers = 2;
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "surface with depth 4 and 2 array layers"));
   desc = desc_2d(AGC_SW_64KB_R_X, 64, 64, 1, 4);
   desc.depth = 2;
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "thin surfaces one depth slice"));
   desc = desc_2d(AGC_SW_64KB_R_X, 64, 64, 1, 4);
   desc.layers = 8193;
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "each at most 8192"));
   desc = desc_2d(AGC_SW_64KB_R_X, 64, 64, 17, 4);
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "surface mip count 17 is out of range"));
   desc = desc_2d(AGC_SW_64KB_R_X, 64, 64, 1, 0);
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "surface element size is zero"));
   desc.bytes_per_element = 12;
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "12-byte elements are not supported in swizzle mode 27 (SW_64KB_R_X)"));
   desc.bytes_per_element = 32;
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "32-byte elements"));
   desc = desc_3d(AGC_SW_64KB_D, 64, 64, 4, 1, 4);
   CHECK(layout_fails(&desc, AGC_SURFACE_GUEST, "3D surfaces in swizzle mode 10 (SW_64KB_D) are not implemented"));
   desc = desc_2d(AGC_SW_64KB_Z_X, 64, 64, 1, 16);
   CHECK(layout_fails(&desc, AGC_SURFACE_HOST, "no swizzle equation for mode 24 (SW_64KB_Z_X) with 16-byte elements"));
   struct agc_surface_layout layout;
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &layout));
   desc = desc_2d(AGC_SW_64KB_R_X, 64, 64, 1, 4);
   CHECK(layout_fails(&desc, (enum agc_surface_addressing)7, "neither guest nor host"));
   CHECK(!agc_surface_layout_init(NULL, AGC_SURFACE_GUEST, &layout, NULL, 0));
   CHECK(!agc_surface_layout_init(&desc, AGC_SURFACE_GUEST, NULL, NULL, 0));
   desc.mip_levels = 0;
   desc.layers = 0;
   desc.depth = 0;
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &layout));
   CHECK_EQ(layout.mip_count, 1);
   CHECK_EQ(layout.size, 65536);
   CHECK_EQ(layout.desc.mip_levels, 0);
   CHECK(layout_of(&layout.desc, AGC_SURFACE_GUEST, &layout));
   CHECK_EQ(layout.size, 65536);
}

static void
test_extents(void)
{
   const struct agc_surface_desc desc = desc_2d(AGC_SW_64KB_S_X, 100, 7, 4, 4);
   struct agc_surface_layout layout;
   uint32_t extent[3];
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &layout));
   agc_surface_level_extent(&layout, 3, extent);
   CHECK_EQ(extent[0], 13);
   CHECK_EQ(extent[1], 1);
   CHECK_EQ(extent[2], 1);
   agc_surface_mip_extent(&layout, 3, extent);
   CHECK_EQ(extent[0], 12);
   CHECK_EQ(extent[1], 1);
   CHECK_EQ(extent[2], 1);
   agc_surface_level_extent(&layout, 2, extent);
   CHECK_EQ(extent[0], 25);
   CHECK_EQ(extent[1], 2);
   agc_surface_mip_extent(&layout, 2, extent);
   CHECK_EQ(extent[0], 25);
   CHECK_EQ(extent[1], 1);
   agc_surface_level_extent(&layout, 40, extent);
   CHECK_EQ(extent[0], 1);
   CHECK_EQ(extent[1], 1);
   agc_surface_mip_extent(&layout, 40, extent);
   CHECK_EQ(extent[0], 1);
   CHECK_EQ(extent[1], 1);

   const struct agc_surface_desc volume = desc_3d(AGC_SW_4KB_S_X, 9, 9, 9, 4, 4);
   CHECK(layout_of(&volume, AGC_SURFACE_GUEST, &layout));
   agc_surface_level_extent(&layout, 2, extent);
   CHECK_EQ(extent[0], 3);
   CHECK_EQ(extent[2], 2);
   agc_surface_mip_extent(&layout, 2, extent);
   CHECK_EQ(extent[0], 2);
   CHECK_EQ(extent[2], 2);
   agc_surface_level_extent(&layout, 3, extent);
   CHECK_EQ(extent[2], 1);

   struct agc_surface_desc array = desc_2d(AGC_SW_64KB_D, 64, 64, 3, 4);
   array.layers = 6;
   CHECK(layout_of(&array, AGC_SURFACE_GUEST, &layout));
   agc_surface_level_extent(&layout, 2, extent);
   CHECK_EQ(extent[2], 6);
   agc_surface_mip_extent(&layout, 2, extent);
   CHECK_EQ(extent[2], 1);
}

static void
test_copy_params(void)
{
   const uint64_t source = UINT64_C(0x4000140000);
   const uint64_t mirror = UINT64_C(0x14000140000);
   const struct agc_surface_desc desc = desc_2d(AGC_SW_64KB_S_X, 256, 256, 9, 4);
   struct agc_surface_desc host_desc = desc;
   host_desc.swizzle_mode = AGC_SW_64KB_D_X;
   struct agc_surface_layout guest;
   struct agc_surface_layout host;
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &guest));
   CHECK(layout_of(&host_desc, AGC_SURFACE_HOST, &host));
   uint32_t params[AGC_SURFACE_COPY_PARAM_DWORDS];
   uint32_t groups[3];
   char error[512] = "";
   CHECK(agc_surface_copy_params(&guest, source, &host, mirror, 0, params, groups, error, sizeof(error)));
   const uint32_t level0[11] = {
      (uint32_t)(source + 131072), (uint32_t)((source + 131072) >> 32), 393216, 0, 2, 7, 7, 16, 0, 0, 0};
   for (unsigned i = 0; i < 11; i++)
      CHECK_EQ(params[i], level0[i]);
   for (unsigned i = 11; i < 16; i++) {
      CHECK_EQ(params[i], 0);
      CHECK_EQ(params[32 + i], 0);
   }
   for (unsigned bit = 0; bit < 16; bit++) {
      CHECK_EQ(params[16 + bit], guest.equation[bit]);
      CHECK_EQ(params[48 + bit], host.equation[bit]);
   }
   CHECK_EQ(params[32], (uint32_t)(mirror + 131072));
   CHECK_EQ(params[33], (uint32_t)((mirror + 131072) >> 32));
   CHECK_EQ(params[36], 2);
   CHECK_EQ(params[64], 256);
   CHECK_EQ(params[65], 256);
   CHECK_EQ(params[66], 2);
   CHECK_EQ(params[67], 0);
   CHECK_EQ(groups[0], 4);
   CHECK_EQ(groups[1], 256);
   CHECK_EQ(groups[2], 1);
   CHECK(agc_surface_copy_params(&guest, source, &host, mirror, 3, params, groups, error, sizeof(error)));
   CHECK_EQ(params[0], (uint32_t)source);
   CHECK_EQ(params[4], 1);
   CHECK_EQ(params[8], 0);
   CHECK_EQ(params[9], 64);
   CHECK_EQ(params[40], 0);
   CHECK_EQ(params[41], 64);
   CHECK_EQ(params[64], 32);
   CHECK_EQ(groups[0], 1);
   CHECK_EQ(groups[1], 32);
   check_copy_addresses(&guest, source, &host, mirror);
   check_copy_addresses(&host, mirror, &guest, source);

   struct agc_surface_desc xored = desc;
   xored.pipe_bank_xor = 0xc7;
   xored.layers = 3;
   struct agc_surface_desc xored_host = host_desc;
   xored_host.layers = 3;
   CHECK(layout_of(&xored, AGC_SURFACE_GUEST, &guest));
   CHECK(layout_of(&xored_host, AGC_SURFACE_HOST, &host));
   CHECK(agc_surface_copy_params(&guest, source, &host, mirror, 0, params, groups, error, sizeof(error)));
   CHECK_EQ(params[10], 0xc700);
   CHECK_EQ(params[42], 0);
   CHECK_EQ(groups[2], 3);
   check_copy_addresses(&guest, source, &host, mirror);

   const struct agc_surface_desc linear = desc_2d(AGC_SW_LINEAR, 100, 50, 3, 4);
   struct agc_surface_desc tiled = linear;
   tiled.swizzle_mode = AGC_SW_64KB_R_X;
   CHECK(layout_of(&linear, AGC_SURFACE_GUEST, &guest));
   CHECK(layout_of(&tiled, AGC_SURFACE_HOST, &host));
   CHECK(agc_surface_copy_params(&guest, source, &host, mirror, 0, params, groups, error, sizeof(error)));
   CHECK_EQ(params[4], 128);
   CHECK_EQ(params[5], 0);
   CHECK_EQ(params[6], 0);
   CHECK_EQ(params[7], 2);
   for (unsigned bit = 0; bit < 16; bit++)
      CHECK_EQ(params[16 + bit], 0);
   check_copy_addresses(&guest, source, &host, mirror);

   for (uint32_t bpe = 1; bpe <= 16; bpe <<= 1) {
      const struct agc_surface_desc color = desc_2d(AGC_SW_64KB_R_X, 64, 64, 1, bpe);
      struct agc_surface_desc native = color;
      native.swizzle_mode = agc_surface_host_swizzle_mode(color.swizzle_mode, AGC_SURFACE_USE_THIN_2D);
      CHECK_EQ(native.swizzle_mode, AGC_SW_64KB_R_X);
      CHECK(layout_of(&color, AGC_SURFACE_GUEST, &guest));
      CHECK(layout_of(&native, AGC_SURFACE_HOST, &host));
      CHECK(!agc_surface_same_addressing(&guest, &host));
      CHECK(agc_surface_copy_params(&guest, source, &host, mirror, 0, params, groups, error, sizeof(error)));
      CHECK_EQ(groups[0], 1);
      CHECK_EQ(groups[1], 64);
      CHECK_EQ(groups[2], 1);
      CHECK_EQ(params[66], (uint32_t)(bpe == 1 ? 0 : bpe == 2 ? 1 : bpe == 4 ? 2 : bpe == 8 ? 3 : 4));
      check_copy_addresses(&guest, source, &host, mirror);
   }

   const struct agc_surface_desc bc = desc_2d(AGC_SW_64KB_S, 25, 15, 5, 16);
   struct agc_surface_desc bc_host = bc;
   bc_host.swizzle_mode = AGC_SW_64KB_D;
   CHECK(layout_of(&bc, AGC_SURFACE_GUEST, &guest));
   CHECK(layout_of(&bc_host, AGC_SURFACE_HOST, &host));
   check_copy_addresses(&guest, source, &host, mirror);

   const struct agc_surface_desc volume = desc_3d(AGC_SW_64KB_S_X, 64, 64, 64, 1, 4);
   struct agc_surface_layout thick;
   CHECK(layout_of(&volume, AGC_SURFACE_GUEST, &thick));
   params[0] = 1;
   groups[0] = 1;
   CHECK(!agc_surface_copy_params(&thick, source, &thick, mirror, 0, params, groups, error, sizeof(error)));
   CHECK(strstr(error, "does not address thick (3D-block) layouts") != NULL);
   CHECK_EQ(params[0], 0);
   CHECK_EQ(groups[0], 0);
   const struct agc_surface_desc thin_volume = desc_3d(AGC_SW_64KB_R_X, 64, 64, 4, 1, 4);
   struct agc_surface_desc linear_volume = thin_volume;
   linear_volume.swizzle_mode = AGC_SW_LINEAR;
   CHECK(layout_of(&thin_volume, AGC_SURFACE_GUEST, &guest));
   CHECK(layout_of(&linear_volume, AGC_SURFACE_HOST, &host));
   check_copy_addresses(&guest, source, &host, mirror);

   const struct agc_surface_desc other = desc_2d(AGC_SW_64KB_D_X, 256, 128, 9, 4);
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &guest));
   CHECK(layout_of(&other, AGC_SURFACE_HOST, &host));
   CHECK(!agc_surface_copy_params(&guest, source, &host, mirror, 0, params, groups, error, sizeof(error)));
   CHECK(strstr(error, "two layouts of the same extent, levels and element size") != NULL);
   const struct agc_surface_desc rgb32 = desc_2d(AGC_SW_LINEAR, 10, 3, 1, 12);
   CHECK(layout_of(&rgb32, AGC_SURFACE_GUEST, &guest));
   CHECK(!agc_surface_copy_params(&guest, source, &guest, mirror, 0, params, groups, error, sizeof(error)));
   CHECK(strstr(error, "moves 1, 2, 4, 8 or 16-byte elements, not 12") != NULL);
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &guest));
   CHECK(layout_of(&host_desc, AGC_SURFACE_HOST, &host));
   CHECK(!agc_surface_copy_params(&guest, source, &host, mirror, 9, params, groups, error, sizeof(error)));
   CHECK(strstr(error, "mip 9 is outside the surface") != NULL);
}

static void
test_image_views(void)
{
   const uint64_t base = UINT64_C(0x4000140000);
   const uint32_t tsharp[8] = {
      (uint32_t)(base >> 8),
      ((uint32_t)(base >> 40) & 0xff) | (56u << 20) | (3u << 30),
      (255u >> 2) | (255u << 14),
      (9u << 28) | (AGC_SW_64KB_S << 20),
      0,
      0,
      0,
      0,
   };
   struct agc_image_view view;
   char error[512] = "";
   CHECK(agc_image_view_decode(tsharp, &view, error, sizeof(error)));
   CHECK_EQ(view.base, base);
   CHECK_EQ(view.type, 9);
   CHECK_EQ(view.format, 56);
   CHECK_EQ(view.swizzle_mode, AGC_SW_64KB_S);
   CHECK_EQ(view.width, 256);
   CHECK_EQ(view.height, 256);
   CHECK_EQ(view.depth, 1);
   CHECK(!view.has_meta);
   CHECK_EQ(agc_image_view_surface_use(&view), AGC_SURFACE_USE_THIN_2D);
   CHECK(!agc_image_view_samples_in_place(&view));
   CHECK_EQ(agc_image_view_host_swizzle_mode(&view), AGC_SW_64KB_D);
   struct agc_surface_desc desc;
   CHECK(agc_image_view_surface_desc(&view, &desc, error, sizeof(error)));
   CHECK_EQ(desc.width, 256);
   CHECK_EQ(desc.height, 256);
   CHECK_EQ(desc.mip_levels, 1);
   CHECK_EQ(desc.layers, 1);
   CHECK_EQ(desc.bytes_per_element, 4);
   CHECK(!desc.volume);
   struct agc_surface_desc host_desc = desc;
   host_desc.swizzle_mode = agc_image_view_host_swizzle_mode(&view);
   host_desc.pipe_bank_xor = 0;
   host_desc.pitch_elements = 0;
   struct agc_surface_layout guest;
   struct agc_surface_layout host;
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &guest));
   CHECK(layout_of(&host_desc, AGC_SURFACE_HOST, &host));
   CHECK(!agc_surface_same_addressing(&guest, &host));
   CHECK(agc_surface_same_placement(&guest, &host));
   CHECK_EQ(guest.size, 262144);
   CHECK_EQ(offset_of(&guest, 17, 3, 0, 0), 4 * 1 + 8 * 0 + 16 * 1 + 32 * 1 + 128 * 0 + 512 * 0 + 2048 * 1);
   CHECK_EQ(offset_of(&host, 17, 3, 0, 0), 4 * 1 + 8 * 1 + 16 * 0 + 32 * 1 + 2048 * 1);
   check_copy_addresses(&guest, view.base, &host, view.base + (UINT64_C(1) << 40));

   uint32_t words[8] = {0};
   words[0] = 0x10000;
   words[1] = 169u << 20 | (3u << 30) | (0x280u << 8);
   words[2] = (24u) | (59u << 14);
   words[3] = (13u << 28) | (AGC_SW_64KB_S_X << 20) | (9u << 16) | (1u << 12);
   words[4] = 3;
   words[5] = 7u << 4;
   words[6] = (1u << 21) | (0x34u << 24);
   words[7] = 0x12;
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(view.width, 100);
   CHECK_EQ(view.height, 60);
   CHECK_EQ(view.last_array, 3);
   CHECK_EQ(view.base_level, 1);
   CHECK_EQ(view.max_mip, 7);
   CHECK_EQ(view.last_level, 7);
   CHECK_EQ(view.min_lod, 0x280);
   CHECK(view.has_meta);
   CHECK_EQ(view.meta, UINT64_C(0x123400));
   CHECK_EQ(agc_image_view_first_reachable_level(&view), 2);
   CHECK(agc_image_view_surface_desc(&view, &desc, error, sizeof(error)));
   CHECK_EQ(desc.width, 25);
   CHECK_EQ(desc.height, 15);
   CHECK_EQ(desc.layers, 4);
   CHECK_EQ(desc.mip_levels, 8);
   CHECK_EQ(desc.bytes_per_element, 8);
   CHECK_EQ(agc_image_view_host_swizzle_mode(&view), AGC_SW_64KB_D_X);
   words[1] = (words[1] & ~(0xfffu << 8)) | (0x900u << 8);
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(agc_image_view_first_reachable_level(&view), 7);
   words[3] = (words[3] & ~(0xfu << 12)) | (5u << 12);
   words[1] &= ~(0xfffu << 8);
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(agc_image_view_first_reachable_level(&view), 5);
   words[5] = 3u << 4;
   words[3] = (words[3] & ~(0xfu << 12)) | (2u << 12);
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(view.last_level, 3);
   words[3] = (words[3] & ~(0xfu << 12)) | (6u << 12);
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(view.last_level, 6);

   memset(words, 0, sizeof(words));
   words[0] = 0x10000;
   words[1] = 56u << 20;
   words[3] = (11u << 28) | (AGC_SW_64KB_R_X << 20);
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(view.last_array, 5);
   CHECK(agc_image_view_surface_desc(&view, &desc, error, sizeof(error)));
   CHECK_EQ(desc.layers, 6);
   words[4] = 11 | (8u << 16);
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(view.last_array, 13);
   words[4] = 11;
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(view.last_array, 11);

   words[3] = (10u << 28) | (AGC_SW_64KB_S_X << 20);
   words[4] = 31;
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(view.depth, 32);
   CHECK_EQ(view.last_array, 0);
   CHECK_EQ(agc_image_view_surface_use(&view), AGC_SURFACE_USE_VOLUME);
   CHECK_EQ(agc_image_view_host_swizzle_mode(&view), AGC_SW_64KB_S_X);
   CHECK(agc_image_view_surface_desc(&view, &desc, error, sizeof(error)));
   CHECK(desc.volume);
   CHECK_EQ(desc.depth, 32);
   CHECK_EQ(desc.layers, 1);

   memset(words, 0, sizeof(words));
   words[0] = 0x10000;
   words[1] = (56u << 20) | (3u << 30);
   words[2] = 24;
   words[3] = 9u << 28;
   words[4] = 127;
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(view.width, 100);
   CHECK_EQ(view.pitch, 0);
   words[4] = 191;
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(view.pitch, 192);
   CHECK(agc_image_view_samples_in_place(&view));
   CHECK_EQ(agc_image_view_host_swizzle_mode(&view), AGC_SW_LINEAR);
   CHECK(agc_image_view_surface_desc(&view, &desc, error, sizeof(error)));
   CHECK_EQ(desc.pitch_elements, 192);
   CHECK(layout_of(&desc, AGC_SURFACE_GUEST, &guest));
   CHECK_EQ(guest.mips[0].pitch, 192);
   words[1] = (words[1] & ~(0x1ffu << 20)) | (131u << 20);
   CHECK(!agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK(strstr(error, "gfx10 texel format 131 (6E4_FLOAT) is not supported") != NULL);
   CHECK_EQ(view.base, 0);
   words[3] = 13u << 28;
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(view.last_array, 191);
   CHECK(!agc_image_view_surface_desc(&view, &desc, error, sizeof(error)));
   CHECK(strstr(error, "6E4_FLOAT") != NULL);

   static const enum agc_surface_use uses[8] = {
      AGC_SURFACE_USE_1D,      AGC_SURFACE_USE_THIN_2D, AGC_SURFACE_USE_VOLUME, AGC_SURFACE_USE_THIN_2D,
      AGC_SURFACE_USE_1D,      AGC_SURFACE_USE_THIN_2D, AGC_SURFACE_USE_MSAA,   AGC_SURFACE_USE_MSAA,
   };
   for (uint32_t type = 8; type < 16; type++) {
      memset(&view, 0, sizeof(view));
      view.type = type;
      CHECK_EQ(agc_image_view_surface_use(&view), uses[type - 8]);
   }

   memset(words, 0, sizeof(words));
   words[3] = 9u << 28;
   CHECK(!agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK(strstr(error, "image descriptor has a null base address") != NULL);
   words[0] = 0x5a;
   words[3] = (9u << 28) | (AGC_SW_64KB_S_X << 20);
   CHECK(!agc_image_view_decode(words, &view, error, sizeof(error)));
   words[3] = (9u << 28) | (AGC_SW_64KB_S << 20);
   CHECK(agc_image_view_decode(words, &view, error, sizeof(error)));
   CHECK_EQ(view.base, 0x5a00);
   CHECK_EQ(view.pipe_bank_xor, 0);
}

static uint64_t rng_state;

static uint64_t
rng_next(void)
{
   rng_state ^= rng_state << 13;
   rng_state ^= rng_state >> 7;
   rng_state ^= rng_state << 17;
   return rng_state;
}

static uint64_t
rng_below(uint64_t n)
{
   return rng_next() % n;
}

static uint32_t
put(uint64_t value, unsigned lo, unsigned width)
{
   return (uint32_t)(value & ((UINT64_C(1) << width) - 1)) << lo;
}

static uint32_t
next_multiple(uint32_t value, uint32_t unit)
{
   return (value + unit - 1) / unit * unit;
}

static void
check_reachable_layouts(const struct agc_image_view *view)
{
   struct agc_surface_desc desc;
   struct agc_surface_layout guest;
   struct agc_surface_layout host;
   if (!agc_image_view_surface_desc(view, &desc, NULL, 0))
      return;
   if (!agc_surface_layout_init(&desc, AGC_SURFACE_GUEST, &guest, NULL, 0))
      return;
   for (uint32_t level = 0; level < guest.mip_count; level++) {
      uint32_t extent[3];
      agc_surface_level_extent(&guest, level, extent);
      const uint64_t last = offset_of(&guest, extent[0] - 1, extent[1] - 1, extent[2] - 1, level);
      CHECK(last + desc.bytes_per_element <= guest.size);
   }
   struct agc_surface_desc host_desc = desc;
   host_desc.swizzle_mode = agc_image_view_host_swizzle_mode(view);
   host_desc.pipe_bank_xor = 0;
   host_desc.pitch_elements = 0;
   if (!agc_surface_layout_init(&host_desc, AGC_SURFACE_HOST, &host, NULL, 0))
      return;
   uint32_t extent[3];
   agc_surface_level_extent(&host, 0, extent);
   CHECK(offset_of(&host, extent[0] - 1, extent[1] - 1, extent[2] - 1, 0) + desc.bytes_per_element <= host.size);
   if (agc_image_view_samples_in_place(view) && !desc.pitch_elements)
      CHECK(agc_surface_same_addressing(&guest, &host));
}

static void
test_random_descriptors(void)
{
   static const uint32_t modes[] = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 17, 18, 21, 22, 24, 25, 26, 27};
   static const uint32_t array_pitches[] = {0, 0, 1, 1, 2, 15};
   rng_state = 1013;
   unsigned decoded = 0;
   for (unsigned i = 0; i < 20000; i++) {
      const uint32_t sw = modes[rng_below(sizeof(modes) / sizeof(modes[0]))];
      uint64_t block = 65536;
      uint32_t xor_mask = 0xff;
      if (sw <= 3) {
         block = 256;
         xor_mask = 0;
      } else if (sw <= 7) {
         block = 4096;
         xor_mask = 0;
      } else if (sw == 21 || sw == 22) {
         block = 4096;
         xor_mask = 0x0f;
      } else if (sw <= 11) {
         xor_mask = 0;
      }
      const uint64_t va = rng_below(GUEST_ARENA) & ~(block - 1);
      const uint32_t pipe_bank_xor = (uint32_t)rng_below(256) & xor_mask;
      const uint32_t format = (uint32_t)rng_below(512);
      const uint32_t width = (uint32_t)rng_below(16384) + 1;
      const uint32_t height = (uint32_t)rng_below(16384) + 1;
      const uint32_t dsel = (uint32_t)rng_below(1 << 12);
      const uint32_t base_level = (uint32_t)rng_below(16);
      const uint32_t last_level = (uint32_t)rng_below(16);
      const uint32_t bc = (uint32_t)rng_below(6);
      const uint32_t type = 8 + (uint32_t)rng_below(8);
      const uint32_t depth = (uint32_t)rng_below(1 << 13);
      const uint32_t base_array = (uint32_t)rng_below(1 << 13);
      const uint32_t array_pitch = array_pitches[rng_below(6)];
      const uint32_t min_lod_warn = (uint32_t)rng_below(1 << 12);
      const uint32_t perf_mod = (uint32_t)rng_below(8);
      const uint32_t corner = (uint32_t)rng_below(2);
      const uint32_t prt_default = (uint32_t)rng_below(2);
      const uint32_t min_lod = (uint32_t)rng_below(1 << 12);
      const uint32_t max_mip = (uint32_t)rng_below(16);
      const uint32_t lod_cnt_en = (uint32_t)rng_below(2);
      const uint32_t bank_id = (uint32_t)rng_below(256);
      const uint32_t comp = (uint32_t)rng_below(2);
      const uint32_t wce = (uint32_t)rng_below(2);
      const uint64_t meta = rng_below(GUEST_ARENA) & ~UINT64_C(0xff);
      const uint32_t it256 = (uint32_t)rng_below(2);
      uint32_t w[8];
      w[0] = (uint32_t)(va >> 8) | pipe_bank_xor;
      w[1] = put(va >> 40, 0, 8) | put(format, 20, 9) | put(width - 1, 30, 2) | put(min_lod, 8, 12);
      w[2] = put((width - 1) >> 2, 0, 12) | put(height - 1, 14, 14) | put(1, 31, 1);
      w[3] = put(dsel, 0, 12) | put(base_level, 12, 4) | put(last_level, 16, 4) | put(sw, 20, 5) | put(bc, 25, 3) |
             put(type, 28, 4);
      w[4] = put(depth, 0, 13) | put(base_array, 16, 13);
      w[5] = put(array_pitch, 0, 4) | put(min_lod_warn, 8, 12) | put(perf_mod, 20, 3) | put(corner, 23, 1) |
             put(prt_default, 26, 1) | put(max_mip, 4, 4) | put(lod_cnt_en, 25, 1);
      w[6] = put(bank_id, 0, 8) | put(comp, 21, 1) | put(wce, 20, 1) | put(meta >> 8, 24, 8) | put(it256, 10, 1);
      w[7] = comp ? (uint32_t)(meta >> 16) : 0;

      bool fails = va == 0;
      uint32_t pitch = 0;
      const bool pitched = (type == 8 || type == 9 || type == 14) && depth && sw == AGC_SW_LINEAR;
      if (pitched && !fails) {
         struct agc_texel_format texel;
         if (agc_texel_format_from_gfx10(format, &texel, NULL, 0)) {
            const uint32_t unit = 256 / texel.bytes_per_element;
            const uint32_t wanted = next_multiple(depth + 1 > width ? depth + 1 : width, unit);
            pitch = wanted == next_multiple(width, unit) ? 0 : wanted;
         } else {
            fails = true;
         }
      }
      struct agc_image_view view;
      char error[256] = "";
      const bool ok = agc_image_view_decode(w, &view, error, sizeof(error));
      CHECK_EQ(ok, !fails);
      if (!ok)
         continue;
      decoded++;
      uint32_t expected_last_array = 0;
      if (type == 11)
         expected_last_array = depth > base_array + 5 ? depth : base_array + 5;
      else if (type == 12 || type == 13 || type == 15)
         expected_last_array = depth;
      const uint32_t clamp = max_mip > base_level ? max_mip : base_level;
      CHECK_EQ(view.base, va);
      CHECK_EQ(view.pipe_bank_xor, pipe_bank_xor);
      CHECK_EQ(view.type, type);
      CHECK_EQ(view.swizzle_mode, sw);
      CHECK_EQ(view.format, format);
      CHECK_EQ(view.width, width);
      CHECK_EQ(view.height, height);
      CHECK_EQ(view.depth, type == 10 ? depth + 1 : 1);
      CHECK_EQ(view.last_array, expected_last_array);
      CHECK_EQ(view.base_level, base_level);
      CHECK_EQ(view.last_level, last_level < clamp ? last_level : clamp);
      CHECK_EQ(view.max_mip, max_mip);
      CHECK_EQ(view.min_lod, min_lod);
      CHECK_EQ(view.pitch, pitch);
      CHECK_EQ(view.has_meta, comp != 0);
      CHECK_EQ(view.meta, comp ? meta : 0);
      check_reachable_layouts(&view);
   }
   CHECK(decoded > 15000);
}

int
main(void)
{
   test_texel_formats();
   test_host_swizzle_modes();
   test_direct_d_views();
   test_pipe_bank_xor();
   test_block_extents();
   test_bijective_blocks();
   test_equations();
   test_mip_chains();
   test_linear();
   test_offsets();
   test_layout_errors();
   test_extents();
   test_copy_params();
   test_image_views();
   test_random_descriptors();
   return TEST_RESULT();
}
