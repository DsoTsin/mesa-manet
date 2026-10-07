/*
 * Copyright (C) 2022 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "pan_tiling.h"

#include <gtest/gtest.h>

/*
 * Reference tiling algorithm, written for clarity rather than performance. See
 * docs/drivers/panfrost.rst for details on the format.
 */

static unsigned
u_order(unsigned x, unsigned y)
{
   assert(x < 16 && y < 16);

   unsigned xy0 = ((x ^ y) & 1) ? 1 : 0;
   unsigned xy1 = ((x ^ y) & 2) ? 1 : 0;
   unsigned xy2 = ((x ^ y) & 4) ? 1 : 0;
   unsigned xy3 = ((x ^ y) & 8) ? 1 : 0;

   unsigned y0 = (y & 1) ? 1 : 0;
   unsigned y1 = (y & 2) ? 1 : 0;
   unsigned y2 = (y & 4) ? 1 : 0;
   unsigned y3 = (y & 8) ? 1 : 0;

   return (xy0 << 0) | (y0 << 1) | (xy1 << 2) | (y1 << 3) | (xy2 << 4) |
          (y2 << 5) | (xy3 << 6) | (y3 << 7);
}

/* x/y are in blocks */
static unsigned
tiled_offset(unsigned x, unsigned y, unsigned stride, unsigned tilesize,
             unsigned blocksize)
{
   unsigned tile_x = x / tilesize;
   unsigned tile_y = y / tilesize;

   unsigned x_in_tile = x % tilesize;
   unsigned y_in_tile = y % tilesize;

   unsigned index_in_tile = u_order(x_in_tile, y_in_tile);

   unsigned row_offset = tile_y * stride;
   unsigned col_offset = (tile_x * tilesize * tilesize) * blocksize;
   unsigned block_offset = index_in_tile * blocksize;

   return row_offset + col_offset + block_offset;
}

static unsigned
linear_offset(unsigned x, unsigned y, unsigned stride, unsigned blocksize)
{
   return (stride * y) + (x * blocksize);
}

static void
ref_access_tiled(void *dst, const void *src, unsigned region_x,
                 unsigned region_y, unsigned w, unsigned h, uint32_t dst_stride,
                 uint32_t src_stride, enum pipe_format format,
                 bool dst_is_tiled)
{
   const struct util_format_description *desc = util_format_description(format);
   ;

   unsigned tilesize = (desc->block.width > 1) ? 4 : 16;
   unsigned blocksize = (desc->block.bits / 8);

   unsigned w_block = w / desc->block.width;
   unsigned h_block = h / desc->block.height;

   unsigned region_x_block = region_x / desc->block.width;
   unsigned region_y_block = region_y / desc->block.height;

   for (unsigned linear_y_block = 0; linear_y_block < h_block;
        ++linear_y_block) {
      for (unsigned linear_x_block = 0; linear_x_block < w_block;
           ++linear_x_block) {

         unsigned tiled_x_block = region_x_block + linear_x_block;
         unsigned tiled_y_block = region_y_block + linear_y_block;

         unsigned dst_offset, src_offset;

         if (dst_is_tiled) {
            dst_offset = tiled_offset(tiled_x_block, tiled_y_block, dst_stride,
                                      tilesize, blocksize);
            src_offset = linear_offset(linear_x_block, linear_y_block,
                                       src_stride, blocksize);
         } else {
            dst_offset = linear_offset(linear_x_block, linear_y_block,
                                       dst_stride, blocksize);
            src_offset = tiled_offset(tiled_x_block, tiled_y_block, src_stride,
                                      tilesize, blocksize);
         }

         memcpy((uint8_t *)dst + dst_offset, (const uint8_t *)src + src_offset,
                desc->block.bits / 8);
      }
   }
}

/*
 * Helper to build test cases for tiled texture access. This test suite compares
 * the above reference tiling algorithm to the optimized algorithm used in
 * production.
 */
static void
test(unsigned width, unsigned height, unsigned rx, unsigned ry, unsigned rw,
     unsigned rh, unsigned linear_stride, enum pipe_format format, bool store)
{
   unsigned bpp = util_format_get_blocksize(format);
   unsigned tile_height = util_format_is_compressed(format) ? 4 : 16;

   unsigned tiled_width = ALIGN_POT(width, 16);
   unsigned tiled_height = ALIGN_POT(height, 16);
   unsigned tiled_stride = tiled_width * tile_height * bpp;

   unsigned dst_stride = store ? tiled_stride : linear_stride;
   unsigned src_stride = store ? linear_stride : tiled_stride;

   void *tiled = calloc(bpp, tiled_width * tiled_height);
   void *linear = calloc(bpp, rw * linear_stride);
   void *ref =
      calloc(bpp, store ? (tiled_width * tiled_height) : (rw * linear_stride));

   if (store) {
      for (unsigned i = 0; i < bpp * rw * linear_stride; ++i) {
         ((uint8_t *)linear)[i] = (i & 0xFF);
      }

      pan_store_tiled_image(tiled, linear, rx, ry, rw, rh, dst_stride,
                            src_stride, format, PAN_INTERLEAVE_NONE);
   } else {
      for (unsigned i = 0; i < bpp * tiled_width * tiled_height; ++i) {
         ((uint8_t *)tiled)[i] = (i & 0xFF);
      }

      pan_load_tiled_image(linear, tiled, rx, ry, rw, rh, dst_stride,
                           src_stride, format, PAN_INTERLEAVE_NONE);
   }

   ref_access_tiled(ref, store ? linear : tiled, rx, ry, rw, rh, dst_stride,
                    src_stride, format, store);

   if (store)
      EXPECT_EQ(memcmp(ref, tiled, bpp * tiled_width * tiled_height), 0);
   else
      EXPECT_EQ(memcmp(ref, linear, bpp * rw * linear_stride), 0);

   free(ref);
   free(tiled);
   free(linear);
}

static void
test_ldst(unsigned width, unsigned height, unsigned rx, unsigned ry,
          unsigned rw, unsigned rh, unsigned linear_stride,
          enum pipe_format format)
{
   test(width, height, rx, ry, rw, rh, linear_stride, format, true);
   test(width, height, rx, ry, rw, rh, linear_stride, format, false);
}

static unsigned
tiled_size(unsigned width, unsigned height, enum pipe_format format,
           unsigned *stride)
{
   const struct util_format_description *desc = util_format_description(format);
   unsigned tilesize = (desc->block.width > 1) ? 4 : 16;
   unsigned blocksize = desc->block.bits / 8;
   unsigned w_block =
      ALIGN_POT(DIV_ROUND_UP(width, desc->block.width), tilesize);
   unsigned h_block =
      ALIGN_POT(DIV_ROUND_UP(height, desc->block.height), tilesize);

   *stride = w_block * tilesize * blocksize;
   return (h_block / tilesize) * *stride;
}

static void
test_copy(unsigned src_width, unsigned src_height, unsigned dst_width,
          unsigned dst_height, unsigned src_x, unsigned src_y, unsigned dst_x,
          unsigned dst_y, unsigned w, unsigned h, enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   unsigned tilesize = (desc->block.width > 1) ? 4 : 16;
   unsigned blocksize = desc->block.bits / 8;
   unsigned src_stride, dst_stride;
   unsigned src_size = tiled_size(src_width, src_height, format, &src_stride);
   unsigned dst_size = tiled_size(dst_width, dst_height, format, &dst_stride);
   unsigned slack = 65536;

   uint8_t *src = (uint8_t *)malloc(src_size);
   uint8_t *dst = (uint8_t *)malloc(dst_size + slack);
   uint8_t *ref = (uint8_t *)malloc(dst_size + slack);

   for (unsigned i = 0; i < src_size; ++i)
      src[i] = (i * 7) & 0xFF;
   for (unsigned i = 0; i < dst_size + slack; ++i)
      dst[i] = ref[i] = (i * 13 + 5) & 0xFF;

   pan_copy_tiled_image(dst, src, dst_x, dst_y, src_x, src_y, w, h, dst_stride,
                        src_stride, format);

   unsigned src_x_block = src_x / desc->block.width;
   unsigned src_y_block = src_y / desc->block.height;
   unsigned dst_x_block = dst_x / desc->block.width;
   unsigned dst_y_block = dst_y / desc->block.height;

   for (unsigned y = 0; y < DIV_ROUND_UP(h, desc->block.height); ++y) {
      for (unsigned x = 0; x < DIV_ROUND_UP(w, desc->block.width); ++x) {
         memcpy(ref + tiled_offset(dst_x_block + x, dst_y_block + y,
                                   dst_stride, tilesize, blocksize),
                src + tiled_offset(src_x_block + x, src_y_block + y,
                                   src_stride, tilesize, blocksize),
                blocksize);
      }
   }

   EXPECT_EQ(memcmp(ref, dst, dst_size + slack), 0);

   free(src);
   free(dst);
   free(ref);
}

TEST(UInterleavedTiling, RegulatFormats)
{
   /* 8-bit */
   test_ldst(23, 17, 0, 0, 23, 17, 23, PIPE_FORMAT_R8_UINT);

   /* 16-bit */
   test_ldst(23, 17, 0, 0, 23, 17, 23 * 2, PIPE_FORMAT_R8G8_UINT);

   /* 24-bit */
   test_ldst(23, 17, 0, 0, 23, 17, 23 * 3, PIPE_FORMAT_R8G8B8_UINT);

   /* 32-bit */
   test_ldst(23, 17, 0, 0, 23, 17, 23 * 4, PIPE_FORMAT_R32_UINT);

   /* 48-bit */
   test_ldst(23, 17, 0, 0, 23, 17, 23 * 6, PIPE_FORMAT_R16G16B16_UINT);

   /* 64-bit */
   test_ldst(23, 17, 0, 0, 23, 17, 23 * 8, PIPE_FORMAT_R32G32_UINT);

   /* 96-bit */
   test_ldst(23, 17, 0, 0, 23, 17, 23 * 12, PIPE_FORMAT_R32G32B32_UINT);

   /* 128-bit */
   test_ldst(23, 17, 0, 0, 23, 17, 23 * 16, PIPE_FORMAT_R32G32B32A32_UINT);
}

TEST(UInterleavedTiling, UnpackedStrides)
{
   test_ldst(23, 17, 0, 0, 23, 17, 369 * 1, PIPE_FORMAT_R8_SINT);
   test_ldst(23, 17, 0, 0, 23, 17, 369 * 2, PIPE_FORMAT_R8G8_SINT);
   test_ldst(23, 17, 0, 0, 23, 17, 369 * 3, PIPE_FORMAT_R8G8B8_SINT);
   test_ldst(23, 17, 0, 0, 23, 17, 369 * 4, PIPE_FORMAT_R32_SINT);
   test_ldst(23, 17, 0, 0, 23, 17, 369 * 6, PIPE_FORMAT_R16G16B16_SINT);
   test_ldst(23, 17, 0, 0, 23, 17, 369 * 8, PIPE_FORMAT_R32G32_SINT);
   test_ldst(23, 17, 0, 0, 23, 17, 369 * 12, PIPE_FORMAT_R32G32B32_SINT);
   test_ldst(23, 17, 0, 0, 23, 17, 369 * 16, PIPE_FORMAT_R32G32B32A32_SINT);
}

TEST(UInterleavedTiling, PartialAccess)
{
   test_ldst(23, 17, 3, 1, 13, 7, 369 * 1, PIPE_FORMAT_R8_UNORM);
   test_ldst(23, 17, 3, 1, 13, 7, 369 * 2, PIPE_FORMAT_R8G8_UNORM);
   test_ldst(23, 17, 3, 1, 13, 7, 369 * 3, PIPE_FORMAT_R8G8B8_UNORM);
   test_ldst(23, 17, 3, 1, 13, 7, 369 * 4, PIPE_FORMAT_R32_UNORM);
   test_ldst(23, 17, 3, 1, 13, 7, 369 * 6, PIPE_FORMAT_R16G16B16_UNORM);
   test_ldst(23, 17, 3, 1, 13, 7, 369 * 8, PIPE_FORMAT_R32G32_UNORM);
   test_ldst(23, 17, 3, 1, 13, 7, 369 * 12, PIPE_FORMAT_R32G32B32_UNORM);
   test_ldst(23, 17, 3, 1, 13, 7, 369 * 16, PIPE_FORMAT_R32G32B32A32_UNORM);
}

TEST(UInterleavedTiling, ETC)
{
   /* Block alignment assumed */
   test_ldst(32, 32, 0, 0, 32, 32, 512, PIPE_FORMAT_ETC1_RGB8);
   test_ldst(32, 32, 0, 0, 32, 32, 512, PIPE_FORMAT_ETC2_RGB8A1);
   test_ldst(32, 32, 0, 0, 32, 32, 512, PIPE_FORMAT_ETC2_RG11_SNORM);
}

TEST(UInterleavedTiling, PartialETC)
{
   /* Block alignment assumed */
   test_ldst(32, 32, 4, 8, 16, 12, 512, PIPE_FORMAT_ETC1_RGB8);
   test_ldst(32, 32, 4, 8, 16, 12, 512, PIPE_FORMAT_ETC2_RGB8A1);
   test_ldst(32, 32, 4, 8, 16, 12, 512, PIPE_FORMAT_ETC2_RG11_SNORM);
}

TEST(UInterleavedTiling, DXT)
{
   /* Block alignment assumed */
   test_ldst(32, 32, 0, 0, 32, 32, 512, PIPE_FORMAT_DXT1_RGB);
   test_ldst(32, 32, 0, 0, 32, 32, 512, PIPE_FORMAT_DXT3_RGBA);
   test_ldst(32, 32, 0, 0, 32, 32, 512, PIPE_FORMAT_DXT5_RGBA);
}

TEST(UInterleavedTiling, PartialDXT)
{
   /* Block alignment assumed */
   test_ldst(32, 32, 4, 8, 16, 12, 512, PIPE_FORMAT_DXT1_RGB);
   test_ldst(32, 32, 4, 8, 16, 12, 512, PIPE_FORMAT_DXT3_RGBA);
   test_ldst(32, 32, 4, 8, 16, 12, 512, PIPE_FORMAT_DXT5_RGBA);
}

TEST(UInterleavedTiling, ASTC)
{
   /* Block alignment assumed */
   test_ldst(40, 40, 0, 0, 40, 40, 512, PIPE_FORMAT_ASTC_4x4);
   test_ldst(50, 40, 0, 0, 50, 40, 512, PIPE_FORMAT_ASTC_5x4);
   test_ldst(50, 50, 0, 0, 50, 50, 512, PIPE_FORMAT_ASTC_5x5);
}

TEST(UInterleavedTiling, PartialASTC)
{
   /* Block alignment assumed */
   test_ldst(40, 40, 4, 4, 16, 8, 512, PIPE_FORMAT_ASTC_4x4);
   test_ldst(50, 40, 5, 4, 10, 8, 512, PIPE_FORMAT_ASTC_5x4);
   test_ldst(50, 50, 5, 5, 10, 10, 512, PIPE_FORMAT_ASTC_5x5);
}

TEST(UInterleavedTiling, MultiTileAccess)
{
   test_ldst(67, 53, 3, 5, 61, 41, 61 * 1, PIPE_FORMAT_R8_UINT);
   test_ldst(67, 53, 3, 5, 61, 41, 61 * 2, PIPE_FORMAT_R8G8_UINT);
   test_ldst(67, 53, 3, 5, 61, 41, 61 * 4, PIPE_FORMAT_R32_UINT);
   test_ldst(67, 53, 3, 5, 61, 41, 61 * 8, PIPE_FORMAT_R32G32_UINT);
   test_ldst(67, 53, 3, 5, 61, 41, 61 * 16, PIPE_FORMAT_R32G32B32A32_UINT);
}

TEST(UInterleavedTiling, AlignedAccess)
{
   test_ldst(64, 48, 16, 16, 48, 32, 50 * 1, PIPE_FORMAT_R8_UINT);
   test_ldst(64, 48, 16, 16, 48, 32, 50 * 2, PIPE_FORMAT_R8G8_UINT);
   test_ldst(64, 48, 16, 16, 48, 32, 50 * 4, PIPE_FORMAT_R32_UINT);
   test_ldst(64, 48, 16, 16, 48, 32, 50 * 8, PIPE_FORMAT_R32G32_UINT);
   test_ldst(64, 48, 16, 16, 48, 32, 50 * 16, PIPE_FORMAT_R32G32B32A32_UINT);
}

TEST(UInterleavedTiling, Copy)
{
   test_copy(64, 64, 80, 48, 16, 16, 32, 16, 32, 32,
             PIPE_FORMAT_R8G8B8A8_UINT);
   test_copy(64, 64, 80, 48, 3, 5, 21, 7, 45, 37, PIPE_FORMAT_R8G8B8A8_UINT);
   test_copy(64, 64, 80, 48, 16, 16, 32, 0, 32, 32, PIPE_FORMAT_DXT1_RGB);
   test_copy(64, 64, 80, 48, 4, 8, 12, 4, 40, 36, PIPE_FORMAT_DXT5_RGBA);
   test_copy(200, 40, 200, 40, 80, 0, 0, 20, 80, 20, PIPE_FORMAT_ASTC_5x5);
   test_copy(160, 160, 160, 160, 80, 0, 0, 80, 80, 80, PIPE_FORMAT_ASTC_5x5);
   test_copy(450, 20, 450, 20, 25, 0, 5, 0, 400, 20, PIPE_FORMAT_ASTC_5x5);
   test_copy(451, 20, 451, 20, 45, 0, 45, 0, 406, 20, PIPE_FORMAT_ASTC_5x5);
}
