#include "agc_surface.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util/simple_mtx.h"

#include "agc_types.h"

#define SURFACE_MAX_EXTENT 16384u
#define SURFACE_MAX_SLICES 8192u

#define SURFACE_TYPE_1D 8u
#define SURFACE_TYPE_2D 9u
#define SURFACE_TYPE_3D 10u
#define SURFACE_TYPE_CUBE 11u
#define SURFACE_TYPE_1D_ARRAY 12u
#define SURFACE_TYPE_2D_ARRAY 13u
#define SURFACE_TYPE_2D_MSAA 14u
#define SURFACE_TYPE_2D_MSAA_ARRAY 15u

static const char *const surface_mode_names[32] = {
   "SW_LINEAR",   "SW_256B_S",   "SW_256B_D",   "SW_256B_R",   "SW_4KB_Z",    "SW_4KB_S",    "SW_4KB_D",
   "SW_4KB_R",    "SW_64KB_Z",   "SW_64KB_S",   "SW_64KB_D",   "SW_64KB_R",   "SW_VAR_Z",    "SW_VAR_S",
   "SW_VAR_D",    "SW_VAR_R",    "SW_64KB_Z_T", "SW_64KB_S_T", "SW_64KB_D_T", "SW_64KB_R_T", "SW_4KB_Z_X",
   "SW_4KB_S_X",  "SW_4KB_D_X",  "SW_4KB_R_X",  "SW_64KB_Z_X", "SW_64KB_S_X", "SW_64KB_D_X", "SW_64KB_R_X",
   "SW_VAR_Z_X",  "SW_VAR_S_X",  "SW_VAR_D_X",  "SW_VAR_R_X",
};

static const uint8_t surface_thick_4kb_log2[5][3] = {{4, 4, 4}, {3, 4, 4}, {3, 4, 3}, {3, 3, 3}, {2, 3, 3}};
static const uint8_t surface_thick_64kb_log2[5][3] = {{6, 5, 5}, {5, 5, 5}, {5, 5, 4}, {5, 4, 4}, {4, 4, 4}};
static const uint8_t surface_thick_micro_block[5][2] = {{8, 4}, {4, 4}, {4, 4}, {4, 2}, {2, 2}};

static const uint8_t surface_gfx11_3d_modes[] = {0, 5, 9, 17, 21, 24, 25, 26, 27, 28, 29, 31};

struct surface_format_range {
   uint16_t first;
   uint16_t last;
   uint8_t bytes;
   bool block;
};

static const struct surface_format_range surface_formats[] = {
   {1, 6, 1, false},     {7, 19, 2, false},    {20, 61, 4, false},   {62, 71, 8, false},  {72, 74, 12, false},
   {75, 77, 16, false},  {128, 128, 1, false}, {129, 129, 2, false}, {130, 130, 4, false}, {132, 132, 4, false},
   {133, 136, 2, false}, {137, 137, 1, false}, {140, 144, 4, false}, {145, 146, 8, false}, {169, 170, 8, true},
   {171, 174, 16, true}, {175, 176, 8, true},  {177, 182, 16, true},
};

struct surface_tail {
   uint32_t width;
   uint32_t height;
   uint32_t slots;
   uint32_t micro[2];
};

static bool AGC_PRINTF(3, 4)
surface_fail(char *error, size_t error_bytes, const char *format, ...)
{
   if (error && error_bytes) {
      va_list args;
      va_start(args, format);
      vsnprintf(error, error_bytes, format, args);
      va_end(args);
   }
   return false;
}

static const char *
surface_mode_name(uint32_t mode)
{
   return mode < 32 ? surface_mode_names[mode] : "invalid";
}

static uint32_t
surface_max(uint32_t a, uint32_t b)
{
   return a > b ? a : b;
}

static uint32_t
surface_min(uint32_t a, uint32_t b)
{
   return a < b ? a : b;
}

static uint32_t
surface_shr(uint32_t value, uint32_t shift)
{
   return shift >= 32 ? 0 : value >> shift;
}

static uint32_t
surface_shift_ceil(uint32_t value, uint32_t shift)
{
   if (shift >= 32)
      return value ? 1 : 0;
   return (uint32_t)(((uint64_t)value + (UINT64_C(1) << shift) - 1) >> shift);
}

static uint32_t
surface_div_ceil(uint32_t value, uint32_t divisor)
{
   return value / divisor + (value % divisor != 0);
}

static uint32_t
surface_next_multiple(uint32_t value, uint32_t unit)
{
   const uint32_t rest = value % unit;
   return rest ? value + (unit - rest) : value;
}

static bool
surface_is_pow2(uint32_t value)
{
   return value && !(value & (value - 1));
}

static uint32_t
surface_trailing_zeros(uint64_t value)
{
   uint32_t count = 0;
   if (!value)
      return 64;
   while (!(value & 1)) {
      value >>= 1;
      count++;
   }
   return count;
}

static uint32_t
surface_parity(uint32_t value)
{
   value ^= value >> 16;
   value ^= value >> 8;
   value ^= value >> 4;
   value ^= value >> 2;
   value ^= value >> 1;
   return value & 1;
}

static uint32_t
surface_swizzle(const uint32_t equation[AGC_SURFACE_EQUATION_BITS], uint32_t x, uint32_t y, uint32_t slice)
{
   const uint32_t coordinates = (x & 0xfff) | ((y & 0xfff) << 12) | ((slice & 0xff) << 24);
   uint32_t offset = 0;
   for (uint32_t bit = 0; bit < AGC_SURFACE_EQUATION_BITS; bit++)
      offset |= surface_parity(equation[bit] & coordinates) << bit;
   return offset;
}

static bool
surface_block_log2(uint32_t mode, uint32_t *log2)
{
   switch (mode) {
   case AGC_SW_LINEAR:
      *log2 = 0;
      return true;
   case AGC_SW_256B_S:
   case AGC_SW_256B_D:
      *log2 = 8;
      return true;
   case AGC_SW_4KB_S:
   case AGC_SW_4KB_D:
   case AGC_SW_4KB_S_X:
   case AGC_SW_4KB_D_X:
      *log2 = 12;
      return true;
   case AGC_SW_64KB_S:
   case AGC_SW_64KB_D:
   case AGC_SW_64KB_S_T:
   case AGC_SW_64KB_D_T:
   case AGC_SW_64KB_Z_X:
   case AGC_SW_64KB_S_X:
   case AGC_SW_64KB_D_X:
   case AGC_SW_64KB_R_X:
      *log2 = 16;
      return true;
   default:
      return false;
   }
}

static bool
surface_is_xor(uint32_t mode)
{
   switch (mode) {
   case AGC_SW_64KB_S_T:
   case AGC_SW_64KB_D_T:
   case AGC_SW_4KB_S_X:
   case AGC_SW_4KB_D_X:
   case AGC_SW_64KB_Z_X:
   case AGC_SW_64KB_S_X:
   case AGC_SW_64KB_D_X:
   case AGC_SW_64KB_R_X:
      return true;
   default:
      return false;
   }
}

static uint32_t
surface_standard_offset(uint32_t x, uint32_t y, uint32_t element_bytes)
{
   switch (element_bytes) {
   case 1:
      return ((y << 4) & 0x1f0) ^ ((y << 5) & 0x400) ^ (x & 0x00f) ^ ((x << 5) & 0x200) ^ ((x << 6) & 0x800);
   case 2:
      return ((y << 4) & 0x070) ^ ((y << 5) & 0x100) ^ ((y << 6) & 0x400) ^ ((x << 1) & 0x00e) ^
             ((x << 4) & 0x080) ^ ((x << 5) & 0x200) ^ ((x << 6) & 0x800);
   case 4:
      return ((y << 4) & 0x070) ^ ((y << 5) & 0x100) ^ ((y << 6) & 0x400) ^ ((x << 2) & 0x00c) ^
             ((x << 5) & 0x080) ^ ((x << 6) & 0x200) ^ ((x << 7) & 0x800);
   case 8:
      return ((y << 4) & 0x030) ^ ((y << 6) & 0x100) ^ ((y << 7) & 0x400) ^ ((x << 3) & 0x008) ^
             ((x << 5) & 0x0c0) ^ ((x << 6) & 0x200) ^ ((x << 7) & 0x800);
   default:
      return ((y << 4) & 0x030) ^ ((y << 6) & 0x100) ^ ((y << 7) & 0x400) ^ ((x << 6) & 0x0c0) ^
             ((x << 7) & 0x200) ^ ((x << 8) & 0x800);
   }
}

static uint32_t
surface_standard_64kb_extra(uint32_t x, uint32_t y, uint32_t element_bytes)
{
   switch (element_bytes) {
   case 1:
      return ((x << 7) & 0x2000) ^ ((x << 8) & 0x8000) ^ ((y << 6) & 0x1000) ^ ((y << 7) & 0x4000);
   case 2:
      return ((x << 7) & 0x2000) ^ ((x << 8) & 0x8000) ^ ((y << 7) & 0x1000) ^ ((y << 8) & 0x4000);
   case 4:
      return ((x << 8) & 0x2000) ^ ((x << 9) & 0x8000) ^ ((y << 7) & 0x1000) ^ ((y << 8) & 0x4000);
   case 8:
      return ((x << 8) & 0x2000) ^ ((x << 9) & 0x8000) ^ ((y << 8) & 0x1000) ^ ((y << 9) & 0x4000);
   default:
      return ((x << 9) & 0x2000) ^ ((x << 10) & 0x8000) ^ ((y << 8) & 0x1000) ^ ((y << 9) & 0x4000);
   }
}

static void
surface_standard_equation(uint32_t block_log2, uint32_t element_bytes, uint32_t equation[AGC_SURFACE_EQUATION_BITS])
{
   const uint32_t mask = (uint32_t)((UINT64_C(1) << block_log2) - 1);
   memset(equation, 0, AGC_SURFACE_EQUATION_BITS * sizeof(uint32_t));
   for (uint32_t input = 0; input < 24; input++) {
      const uint32_t x = input < 12 ? 1u << input : 0;
      const uint32_t y = input < 12 ? 0 : 1u << (input - 12);
      uint32_t column = surface_standard_offset(x, y, element_bytes);
      if (block_log2 == 16)
         column ^= surface_standard_64kb_extra(x, y, element_bytes);
      column &= mask;
      for (uint32_t bit = 0; bit < AGC_SURFACE_EQUATION_BITS; bit++) {
         if ((column >> bit) & 1)
            equation[bit] |= 1u << input;
      }
   }
}

static void
surface_tail_init(struct surface_tail *tail, uint32_t mode, bool thick, uint32_t block_log2, uint32_t element_log2,
                  const uint32_t block[3])
{
   tail->width = block[0];
   tail->height = block[1];
   if (!thick || block_log2 % 3 == 1)
      tail->width /= 2;
   else if (block_log2 % 3 == 0)
      tail->height /= 2;
   if (mode == AGC_SW_64KB_Z_X && element_log2 <= 1) {
      tail->width /= 2;
      if (element_log2 == 0)
         tail->height /= 2;
   }
   const uint32_t effective = thick ? block_log2 - (block_log2 - 8) / 3 : block_log2;
   tail->slots = effective <= 11 ? 1 + (1u << (effective - 9)) : effective - 4;
   if (thick) {
      tail->micro[0] = surface_thick_micro_block[element_log2][0];
      tail->micro[1] = surface_thick_micro_block[element_log2][1];
   } else {
      const uint32_t log2 = 8 - element_log2;
      tail->micro[0] = 1u << ((log2 + 1) / 2);
      tail->micro[1] = 1u << (log2 / 2);
   }
}

static bool
surface_tail_first_level(const struct surface_tail *tail, uint32_t width, uint32_t height, uint32_t levels,
                         uint32_t *first)
{
   for (uint32_t level = 0; level < levels; level++) {
      if (surface_shift_ceil(width, level) <= tail->width && surface_shift_ceil(height, level) <= tail->height &&
          levels - level <= tail->slots) {
         *first = level;
         return true;
      }
   }
   return false;
}

static void
surface_tail_origin(const struct surface_tail *tail, uint32_t index, uint32_t origin[2])
{
   const uint32_t slot = tail->slots - 1 - index;
   const uint32_t offset = slot > 6 ? 16u << slot : slot << 8;
   uint32_t x = 0;
   uint32_t y = 0;
   for (uint32_t bit = 0; bit < 6; bit++) {
      x |= ((offset >> (9 + 2 * bit)) & 1) << bit;
      y |= ((offset >> (8 + 2 * bit)) & 1) << bit;
   }
   origin[0] = x * tail->micro[0];
   origin[1] = y * tail->micro[1];
}

uint32_t
agc_surface_host_pipe_bank_xor(uint32_t block_log2, uint32_t pipe_bank_xor)
{
   const uint32_t interleave_log2 = 8;
   const uint32_t pipes_log2 = 3;
   const uint32_t column_bits = 2;
   const uint32_t max_bank_bits = 4;
   const uint32_t below = interleave_log2 + pipes_log2 + column_bits;
   const uint32_t bank_bits = surface_min(block_log2 > below ? block_log2 - below : 0, max_bank_bits);
   const uint32_t pipes = (1u << pipes_log2) - 1;
   const uint32_t banks = ((1u << bank_bits) - 1) << (pipes_log2 + column_bits);
   return pipe_bank_xor & (pipes | banks);
}

void
agc_surface_split_pipe_bank_xor(uint32_t swizzle_mode, uint64_t address, uint64_t *base, uint32_t *pipe_bank_xor)
{
   uint32_t log2;
   uint64_t mask = 0;
   if (surface_block_log2(swizzle_mode, &log2) && surface_is_xor(swizzle_mode))
      mask = (UINT64_C(1) << log2) - 1;
   if (base)
      *base = address & ~mask;
   if (pipe_bank_xor)
      *pipe_bank_xor = (uint32_t)((address & mask) >> 8);
}

uint32_t
agc_surface_type_host_swizzle_mode(uint32_t type, uint32_t swizzle_mode)
{
   if (swizzle_mode == AGC_SW_LINEAR)
      return AGC_SW_LINEAR;
   if (type == SURFACE_TYPE_3D) {
      for (size_t i = 0; i < sizeof(surface_gfx11_3d_modes); i++) {
         if (surface_gfx11_3d_modes[i] == swizzle_mode)
            return swizzle_mode;
      }
      return AGC_SW_64KB_S;
   }
   if (type == SURFACE_TYPE_1D || type == SURFACE_TYPE_1D_ARRAY || type == SURFACE_TYPE_2D_MSAA ||
       type == SURFACE_TYPE_2D_MSAA_ARRAY)
      return swizzle_mode == AGC_SW_64KB_Z_X ? AGC_SW_64KB_Z_X : AGC_SW_64KB_R_X;
   switch (swizzle_mode) {
   case AGC_SW_256B_S:
   case AGC_SW_4KB_S:
   case AGC_SW_64KB_S:
   case AGC_SW_64KB_S_T:
   case AGC_SW_4KB_S_X:
   case AGC_SW_64KB_S_X:
      return swizzle_mode + 1;
   default:
      return swizzle_mode;
   }
}

bool
agc_surface_samples_in_place_for_format(uint32_t type, uint32_t swizzle_mode, uint32_t format)
{
   if (swizzle_mode == AGC_SW_LINEAR)
      return true;
   const bool direct_type =
      (type == SURFACE_TYPE_2D || type == SURFACE_TYPE_CUBE || type == SURFACE_TYPE_2D_ARRAY) &&
      swizzle_mode == AGC_SW_64KB_D;
   const bool direct_format = (format >= 62 && format <= 71) || (format >= 75 && format <= 77);
   return direct_type && direct_format;
}

uint32_t
agc_surface_host_swizzle_mode(uint32_t guest_mode, enum agc_surface_use use)
{
   uint32_t type;
   switch (use) {
   case AGC_SURFACE_USE_DEPTH:
      return AGC_SW_64KB_Z_X;
   case AGC_SURFACE_USE_1D:
      type = SURFACE_TYPE_1D;
      break;
   case AGC_SURFACE_USE_MSAA:
      type = SURFACE_TYPE_2D_MSAA;
      break;
   case AGC_SURFACE_USE_VOLUME:
      type = SURFACE_TYPE_3D;
      break;
   case AGC_SURFACE_USE_THIN_2D:
   default:
      type = SURFACE_TYPE_2D;
      break;
   }
   return agc_surface_type_host_swizzle_mode(type, guest_mode & 0x1f);
}

static bool
surface_layout_build(const struct agc_surface_desc *desc, enum agc_surface_addressing addressing,
                     struct agc_surface_layout *layout, char *error, size_t error_bytes)
{
   const uint32_t mode = desc->swizzle_mode;
   uint32_t block_log2;
   if (addressing != AGC_SURFACE_GUEST && addressing != AGC_SURFACE_HOST)
      return surface_fail(error, error_bytes, "surface addressing %d is neither guest nor host", (int)addressing);
   if (!surface_block_log2(mode, &block_log2))
      return surface_fail(error, error_bytes,
                          "surface swizzle mode %u (%s) is not a gfx10 swizzle mode the PS5 GPU addresses", mode,
                          surface_mode_name(mode));
   if (addressing == AGC_SURFACE_HOST) {
      if (mode != AGC_SW_LINEAR && !agc_surface_host_equation(mode, 2))
         return surface_fail(error, error_bytes,
                             "swizzle mode %u (%s) is not a host-layout mode: host copies use linear, D, Z_X or "
                             "R_X modes",
                             mode, surface_mode_name(mode));
      if (desc->volume && mode != AGC_SW_LINEAR)
         return surface_fail(error, error_bytes,
                             "tiled host-layout volumes need the thick gfx11 equations, which are not generated");
   }
   const uint32_t depth = surface_max(desc->depth, 1);
   const uint32_t layers = surface_max(desc->layers, 1);
   const uint32_t levels = surface_max(desc->mip_levels, 1);
   if (!desc->width || !desc->height || desc->width > SURFACE_MAX_EXTENT || desc->height > SURFACE_MAX_EXTENT)
      return surface_fail(error, error_bytes, "surface extent %ux%u is outside 1..=%u", desc->width, desc->height,
                          SURFACE_MAX_EXTENT);
   if (depth > SURFACE_MAX_SLICES || layers > SURFACE_MAX_SLICES || (desc->volume && layers > 1) ||
       (!desc->volume && depth > 1))
      return surface_fail(error, error_bytes,
                          "surface with depth %u and %u array layers: volumes have one layer, thin surfaces one "
                          "depth slice, each at most %u",
                          depth, layers, SURFACE_MAX_SLICES);
   if (levels > AGC_SURFACE_MAX_MIPS)
      return surface_fail(error, error_bytes, "surface mip count %u is out of range", levels);
   const uint32_t bpe = desc->bytes_per_element;
   if (!bpe)
      return surface_fail(error, error_bytes, "surface element size is zero");
   const bool power_of_two = surface_is_pow2(bpe) && bpe <= 16;
   if (!power_of_two && (mode != AGC_SW_LINEAR || bpe != 12))
      return surface_fail(error, error_bytes, "%u-byte elements are not supported in swizzle mode %u (%s)", bpe,
                          mode, surface_mode_name(mode));
   const uint32_t element_log2 = power_of_two ? surface_trailing_zeros(bpe) : 0;

   enum agc_surface_tiling tiling;
   if (mode == AGC_SW_LINEAR)
      tiling = AGC_SURFACE_TILING_LINEAR;
   else if (!desc->volume || mode == AGC_SW_64KB_Z_X || mode == AGC_SW_64KB_R_X)
      tiling = AGC_SURFACE_TILING_THIN;
   else if (mode == AGC_SW_4KB_S || mode == AGC_SW_4KB_S_X || mode == AGC_SW_64KB_S || mode == AGC_SW_64KB_S_X ||
            mode == AGC_SW_64KB_D_X)
      tiling = AGC_SURFACE_TILING_THICK;
   else
      return surface_fail(error, error_bytes,
                          "3D surfaces in swizzle mode %u (%s) are not implemented: volumes are tiled thick in "
                          "SW_4KB_S, SW_4KB_S_X, SW_64KB_S, SW_64KB_S_X and SW_64KB_D_X, thin in SW_64KB_Z_X and "
                          "SW_64KB_R_X, or linear",
                          mode, surface_mode_name(mode));
   if (desc->pitch_elements &&
       (tiling != AGC_SURFACE_TILING_LINEAR || levels != 1 || desc->pitch_elements < desc->width))
      return surface_fail(error, error_bytes,
                          "a row pitch of %u elements needs a single-level linear surface no wider than the pitch",
                          desc->pitch_elements);
   const uint64_t block_bytes = tiling == AGC_SURFACE_TILING_LINEAR ? 0 : UINT64_C(1) << block_log2;
   const uint32_t pipe_bank_xor = desc->pipe_bank_xor;
   if (pipe_bank_xor) {
      if (!surface_is_xor(mode))
         return surface_fail(error, error_bytes, "swizzle mode %u (%s) has no pipe/bank XOR, got %#x", mode,
                             surface_mode_name(mode), pipe_bank_xor);
      if ((uint64_t)pipe_bank_xor << 8 >= block_bytes)
         return surface_fail(error, error_bytes, "pipe/bank XOR %#x does not fit a %" PRIu64 "-byte block",
                             pipe_bank_xor, block_bytes);
      if (addressing == AGC_SURFACE_GUEST && mode == AGC_SW_64KB_D_X && (pipe_bank_xor & 0x30))
         return surface_fail(error, error_bytes,
                             "pipe/bank XOR %#x sets the column bits of an SW_64KB_D_X block, which is not "
                             "implemented",
                             pipe_bank_xor);
   }

   if (tiling == AGC_SURFACE_TILING_THIN &&
       (mode == AGC_SW_256B_S || mode == AGC_SW_4KB_S || mode == AGC_SW_64KB_S)) {
      surface_standard_equation(block_log2, bpe, layout->equation);
   } else if (tiling != AGC_SURFACE_TILING_LINEAR) {
      const uint32_t *equation =
         addressing == AGC_SURFACE_HOST
            ? agc_surface_host_equation(mode, element_log2)
            : agc_surface_guest_equation(mode, tiling == AGC_SURFACE_TILING_THICK, element_log2);
      if (!equation)
         return surface_fail(error, error_bytes, "no swizzle equation for mode %u (%s) with %u-byte elements", mode,
                             surface_mode_name(mode), bpe);
      memcpy(layout->equation, equation, sizeof(layout->equation));
   }

   switch (tiling) {
   case AGC_SURFACE_TILING_THIN: {
      const uint32_t log2 = block_log2 - element_log2;
      layout->block[0] = 1u << ((log2 + 1) / 2);
      layout->block[1] = 1u << (log2 / 2);
      layout->block[2] = 1;
      break;
   }
   case AGC_SURFACE_TILING_THICK: {
      const uint8_t *log2 =
         block_log2 == 12 ? surface_thick_4kb_log2[element_log2] : surface_thick_64kb_log2[element_log2];
      for (unsigned i = 0; i < 3; i++)
         layout->block[i] = 1u << log2[i];
      break;
   }
   case AGC_SURFACE_TILING_LINEAR:
   default:
      layout->block[0] = 1;
      layout->block[1] = 1;
      layout->block[2] = 1;
      break;
   }

   uint64_t offset = 0;
   if (tiling == AGC_SURFACE_TILING_LINEAR) {
      for (uint32_t level = levels; level-- > 0;) {
         struct agc_surface_mip *mip = &layout->mips[level];
         mip->pitch = desc->pitch_elements ? desc->pitch_elements
                                           : surface_next_multiple(surface_shift_ceil(desc->width, level), 256 / bpe);
         mip->offset = offset;
         mip->size = (uint64_t)mip->pitch * surface_shift_ceil(desc->height, level) * bpe;
         offset += mip->size;
      }
   } else {
      uint32_t untailed = levels;
      if (block_log2 > 8 && levels > 1) {
         struct surface_tail tail;
         uint32_t first;
         surface_tail_init(&tail, mode, tiling == AGC_SURFACE_TILING_THICK, block_log2, element_log2,
                           layout->block);
         if (surface_tail_first_level(&tail, desc->width, desc->height, levels, &first)) {
            for (uint32_t level = first; level < levels; level++) {
               struct agc_surface_mip *mip = &layout->mips[level];
               mip->offset = 0;
               mip->size = block_bytes;
               mip->pitch = 1;
               mip->in_tail = true;
               surface_tail_origin(&tail, level - first, mip->tail_origin);
            }
            untailed = first;
            offset = block_bytes;
         }
      }
      for (uint32_t level = untailed; level-- > 0;) {
         struct agc_surface_mip *mip = &layout->mips[level];
         const uint32_t rows = surface_div_ceil(surface_shift_ceil(desc->height, level), layout->block[1]);
         mip->pitch = surface_div_ceil(surface_shift_ceil(desc->width, level), layout->block[0]);
         mip->offset = offset;
         mip->size = (uint64_t)mip->pitch * rows * block_bytes;
         offset += mip->size;
      }
   }

   uint32_t slices = layers;
   if (desc->volume)
      slices = tiling == AGC_SURFACE_TILING_THICK ? surface_div_ceil(depth, layout->block[2]) : depth;
   if (offset > UINT64_MAX / slices)
      return surface_fail(error, error_bytes, "surface size overflows");

   layout->desc = *desc;
   layout->addressing = addressing;
   layout->tiling = tiling;
   layout->block_bytes = block_bytes;
   layout->xor_bytes =
      (addressing == AGC_SURFACE_GUEST ? pipe_bank_xor : agc_surface_host_pipe_bank_xor(block_log2, pipe_bank_xor))
      << 8;
   layout->mip_count = levels;
   layout->slice_stride = offset;
   layout->size = offset * slices;
   return true;
}

bool
agc_surface_layout_init(const struct agc_surface_desc *desc, enum agc_surface_addressing addressing,
                        struct agc_surface_layout *layout, char *error, size_t error_bytes)
{
   if (!desc || !layout)
      return surface_fail(error, error_bytes, "a surface layout needs a description and a layout to fill");
   const struct agc_surface_desc copy = *desc;
   memset(layout, 0, sizeof(*layout));
   if (surface_layout_build(&copy, addressing, layout, error, error_bytes))
      return true;
   memset(layout, 0, sizeof(*layout));
   return false;
}

static bool
surface_same_mips(const struct agc_surface_layout *a, const struct agc_surface_layout *b)
{
   if (a->mip_count != b->mip_count)
      return false;
   for (uint32_t level = 0; level < a->mip_count && level < AGC_SURFACE_MAX_MIPS; level++) {
      const struct agc_surface_mip *x = &a->mips[level];
      const struct agc_surface_mip *y = &b->mips[level];
      if (x->offset != y->offset || x->size != y->size || x->pitch != y->pitch || x->in_tail != y->in_tail)
         return false;
      if (x->in_tail && (x->tail_origin[0] != y->tail_origin[0] || x->tail_origin[1] != y->tail_origin[1]))
         return false;
   }
   return true;
}

bool
agc_surface_same_addressing(const struct agc_surface_layout *a, const struct agc_surface_layout *b)
{
   if (!a || !b)
      return false;
   if (a->tiling != b->tiling || a->block_bytes != b->block_bytes || a->xor_bytes != b->xor_bytes ||
       a->slice_stride != b->slice_stride || a->size != b->size ||
       a->desc.bytes_per_element != b->desc.bytes_per_element)
      return false;
   for (unsigned i = 0; i < 3; i++) {
      if (a->block[i] != b->block[i])
         return false;
   }
   for (unsigned bit = 0; bit < AGC_SURFACE_EQUATION_BITS; bit++) {
      if (a->equation[bit] != b->equation[bit])
         return false;
   }
   return surface_same_mips(a, b);
}

bool
agc_surface_same_placement(const struct agc_surface_layout *a, const struct agc_surface_layout *b)
{
   if (!a || !b)
      return false;
   return a->block_bytes == b->block_bytes && surface_same_mips(a, b) && a->slice_stride == b->slice_stride &&
          a->size == b->size;
}

bool
agc_surface_element_offset(const struct agc_surface_layout *layout, uint32_t x, uint32_t y, uint32_t slice,
                           uint32_t mip, uint64_t *offset, char *error, size_t error_bytes)
{
   if (!layout || !offset)
      return surface_fail(error, error_bytes, "an element offset needs a layout and an output");
   *offset = 0;
   if (mip >= layout->mip_count || mip >= AGC_SURFACE_MAX_MIPS)
      return surface_fail(error, error_bytes, "mip %u is outside the %u-level surface", mip, layout->mip_count);
   const struct agc_surface_desc *desc = &layout->desc;
   const struct agc_surface_mip *level = &layout->mips[mip];
   const uint32_t width = surface_shift_ceil(desc->width, mip);
   const uint32_t height = surface_shift_ceil(desc->height, mip);
   const uint32_t slices = desc->volume ? surface_max(surface_shr(surface_max(desc->depth, 1), mip), 1)
                                        : surface_max(desc->layers, 1);
   if (x >= width || y >= height || slice >= slices)
      return surface_fail(error, error_bytes,
                          "element (%u, %u) of slice %u lies outside mip %u (%ux%u, %u slices)", x, y, slice, mip,
                          width, height, slices);
   if (layout->tiling == AGC_SURFACE_TILING_LINEAR) {
      const uint64_t row = (uint64_t)y * level->pitch + x;
      *offset = (uint64_t)slice * layout->slice_stride + level->offset + row * desc->bytes_per_element;
      return true;
   }
   const uint32_t stride_slice = layout->tiling == AGC_SURFACE_TILING_THICK ? slice / layout->block[2] : slice;
   uint64_t block = 0;
   uint32_t in_block;
   if (level->in_tail) {
      in_block = surface_swizzle(layout->equation, x + level->tail_origin[0], y + level->tail_origin[1], slice);
   } else {
      block = (uint64_t)(y / layout->block[1]) * level->pitch + x / layout->block[0];
      in_block = surface_swizzle(layout->equation, x, y, slice);
   }
   *offset = (uint64_t)stride_slice * layout->slice_stride + level->offset + block * layout->block_bytes +
             (in_block ^ layout->xor_bytes);
   return true;
}

void
agc_surface_mip_extent(const struct agc_surface_layout *layout, uint32_t mip, uint32_t extent[3])
{
   const struct agc_surface_desc *desc = &layout->desc;
   extent[0] = surface_max(surface_shr(desc->width, mip), 1);
   extent[1] = surface_max(surface_shr(desc->height, mip), 1);
   extent[2] = desc->volume ? surface_max(surface_shr(desc->depth, mip), 1) : 1;
}

void
agc_surface_level_extent(const struct agc_surface_layout *layout, uint32_t level, uint32_t extent[3])
{
   const struct agc_surface_desc *desc = &layout->desc;
   extent[0] = surface_shift_ceil(desc->width, level);
   extent[1] = surface_shift_ceil(desc->height, level);
   extent[2] = desc->volume ? surface_max(surface_shr(surface_max(desc->depth, 1), level), 1)
                            : surface_max(desc->layers, 1);
}

static bool
surface_same_extent(const struct agc_surface_desc *a, const struct agc_surface_desc *b)
{
   return a->width == b->width && a->height == b->height && surface_max(a->depth, 1) == surface_max(b->depth, 1) &&
          surface_max(a->layers, 1) == surface_max(b->layers, 1) &&
          surface_max(a->mip_levels, 1) == surface_max(b->mip_levels, 1) &&
          a->bytes_per_element == b->bytes_per_element && a->volume == b->volume;
}

static bool
surface_layout_params(const struct agc_surface_layout *layout, uint64_t base, uint32_t level,
                      uint32_t words[AGC_SURFACE_COPY_LAYOUT_DWORDS], char *error, size_t error_bytes)
{
   if (layout->tiling == AGC_SURFACE_TILING_THICK)
      return surface_fail(error, error_bytes, "the element-copy kernel does not address thick (3D-block) layouts");
   if (layout->slice_stride > UINT32_MAX)
      return surface_fail(error, error_bytes,
                          "the element-copy kernel offsets slices of at most 4 GiB, not %" PRIu64 " bytes",
                          layout->slice_stride);
   if (level >= layout->mip_count || level >= AGC_SURFACE_MAX_MIPS)
      return surface_fail(error, error_bytes, "mip %u is outside the surface", level);
   const struct agc_surface_mip *mip = &layout->mips[level];
   const uint64_t start = base + mip->offset;
   uint32_t block_width_log2 = 0;
   uint32_t block_height_log2 = 0;
   uint32_t block_bytes_log2 = surface_trailing_zeros(layout->desc.bytes_per_element);
   if (layout->tiling != AGC_SURFACE_TILING_LINEAR) {
      block_width_log2 = surface_trailing_zeros(layout->block[0]);
      block_height_log2 = surface_trailing_zeros(layout->block[1]);
      block_bytes_log2 = surface_trailing_zeros(layout->block_bytes);
   }
   memset(words, 0, AGC_SURFACE_COPY_LAYOUT_DWORDS * sizeof(uint32_t));
   words[0] = (uint32_t)start;
   words[1] = (uint32_t)(start >> 32);
   words[2] = (uint32_t)layout->slice_stride;
   words[3] = (uint32_t)(layout->slice_stride >> 32);
   words[4] = mip->pitch;
   words[5] = block_width_log2;
   words[6] = block_height_log2;
   words[7] = block_bytes_log2;
   words[8] = mip->in_tail ? mip->tail_origin[0] : 0;
   words[9] = mip->in_tail ? mip->tail_origin[1] : 0;
   words[10] = layout->xor_bytes;
   memcpy(&words[AGC_SURFACE_COPY_LAYOUT_DWORDS - AGC_SURFACE_EQUATION_BITS], layout->equation,
          sizeof(layout->equation));
   return true;
}

bool
agc_surface_copy_params(const struct agc_surface_layout *from, uint64_t from_base,
                        const struct agc_surface_layout *to, uint64_t to_base, uint32_t level,
                        uint32_t params[AGC_SURFACE_COPY_PARAM_DWORDS], uint32_t groups[3], char *error,
                        size_t error_bytes)
{
   if (!params || !groups)
      return surface_fail(error, error_bytes, "a surface copy needs parameter and group outputs");
   memset(params, 0, AGC_SURFACE_COPY_PARAM_DWORDS * sizeof(uint32_t));
   memset(groups, 0, 3 * sizeof(uint32_t));
   if (!from || !to)
      return surface_fail(error, error_bytes, "a surface copy needs two layouts");
   if (!surface_same_extent(&from->desc, &to->desc))
      return surface_fail(error, error_bytes,
                          "a surface copy needs two layouts of the same extent, levels and element size");
   const uint32_t bpe = from->desc.bytes_per_element;
   if (!surface_is_pow2(bpe) || bpe > 16)
      return surface_fail(error, error_bytes,
                          "the element-copy kernel moves 1, 2, 4, 8 or 16-byte elements, not %u", bpe);
   uint32_t extent[3];
   uint32_t words[AGC_SURFACE_COPY_PARAM_DWORDS];
   agc_surface_level_extent(from, level, extent);
   if (!surface_layout_params(from, from_base, level, words, error, error_bytes) ||
       !surface_layout_params(to, to_base, level, &words[AGC_SURFACE_COPY_LAYOUT_DWORDS], error, error_bytes))
      return false;
   words[2 * AGC_SURFACE_COPY_LAYOUT_DWORDS] = extent[0];
   words[2 * AGC_SURFACE_COPY_LAYOUT_DWORDS + 1] = extent[1];
   words[2 * AGC_SURFACE_COPY_LAYOUT_DWORDS + 2] = surface_trailing_zeros(bpe);
   words[2 * AGC_SURFACE_COPY_LAYOUT_DWORDS + 3] = 0;
   memcpy(params, words, sizeof(words));
   groups[0] = surface_div_ceil(extent[0], AGC_SURFACE_COPY_GROUP_THREADS);
   groups[1] = extent[1];
   groups[2] = extent[2];
   return true;
}

static const char *
surface_format_kind(uint32_t format)
{
   if (format == 0)
      return "INVALID";
   if (format == 131)
      return "6E4_FLOAT";
   if (format == 138 || format == 139)
      return "1-bit";
   if (format >= 147 && format <= 154)
      return "subsampled GB_GR/BG_RG";
   if (format >= 156 && format <= 168)
      return "FMASK";
   if (format >= 265 && format <= 284)
      return "multimedia";
   if (format == 285)
      return "7E3_FLOAT";
   if (format >= 286 && format <= 292)
      return "YCBCR";
   return "reserved";
}

bool
agc_texel_format_from_gfx10(uint32_t format, struct agc_texel_format *texel, char *error, size_t error_bytes)
{
   if (!texel)
      return surface_fail(error, error_bytes, "a texel format needs an output");
   memset(texel, 0, sizeof(*texel));
   for (size_t i = 0; i < sizeof(surface_formats) / sizeof(surface_formats[0]); i++) {
      const struct surface_format_range *range = &surface_formats[i];
      if (format < range->first || format > range->last)
         continue;
      texel->bytes_per_element = range->bytes;
      texel->block_width = range->block ? 4 : 1;
      texel->block_height = range->block ? 4 : 1;
      return true;
   }
   return surface_fail(error, error_bytes, "gfx10 texel format %u (%s) is not supported", format,
                       surface_format_kind(format));
}

static uint32_t
surface_field(uint32_t value, uint32_t lo, uint32_t width)
{
   return (value >> lo) & ((1u << width) - 1);
}

bool
agc_image_view_decode(const uint32_t words[8], struct agc_image_view *view, char *error, size_t error_bytes)
{
   if (!words || !view)
      return surface_fail(error, error_bytes, "an image view needs descriptor words and a view to fill");
   memset(view, 0, sizeof(*view));
   const uint32_t type = surface_field(words[3], 28, 4);
   const uint32_t swizzle_mode = surface_field(words[3], 20, 5);
   const uint64_t address = ((uint64_t)words[0] | ((uint64_t)surface_field(words[1], 0, 8) << 32)) << 8;
   uint64_t base;
   uint32_t pipe_bank_xor;
   agc_surface_split_pipe_bank_xor(swizzle_mode, address, &base, &pipe_bank_xor);
   if (!base)
      return surface_fail(error, error_bytes, "image descriptor has a null base address");
   const uint32_t format = surface_field(words[1], 20, 9);
   const uint32_t width = (surface_field(words[1], 30, 2) | (surface_field(words[2], 0, 12) << 2)) + 1;
   const uint32_t depth_field = surface_field(words[4], 0, 13);
   const uint32_t base_array = surface_field(words[4], 16, 13);
   const uint32_t max_mip = surface_field(words[5], 4, 4);
   const uint32_t base_level = surface_field(words[3], 12, 4);
   const uint32_t last_level = surface_min(surface_field(words[3], 16, 4), surface_max(max_mip, base_level));
   uint32_t depth = 1;
   uint32_t last_array = 0;
   uint32_t pitch = 0;
   switch (type) {
   case SURFACE_TYPE_3D:
      depth = depth_field + 1;
      break;
   case SURFACE_TYPE_CUBE:
      last_array = surface_max(depth_field, base_array + 5);
      break;
   case SURFACE_TYPE_1D_ARRAY:
   case SURFACE_TYPE_2D_ARRAY:
   case SURFACE_TYPE_2D_MSAA_ARRAY:
      last_array = depth_field;
      break;
   case SURFACE_TYPE_1D:
   case SURFACE_TYPE_2D:
   case SURFACE_TYPE_2D_MSAA:
      if (depth_field && swizzle_mode == AGC_SW_LINEAR) {
         struct agc_texel_format texel;
         if (!agc_texel_format_from_gfx10(format, &texel, error, error_bytes))
            return false;
         const uint32_t unit = 256 / texel.bytes_per_element;
         const uint32_t wanted = surface_next_multiple(surface_max(depth_field + 1, width), unit);
         pitch = wanted == surface_next_multiple(width, unit) ? 0 : wanted;
      }
      break;
   default:
      break;
   }
   view->base = base;
   view->swizzle_mode = swizzle_mode;
   view->pipe_bank_xor = pipe_bank_xor;
   view->type = type;
   view->format = format;
   view->width = width;
   view->height = surface_field(words[2], 14, 14) + 1;
   view->depth = depth;
   view->last_array = last_array;
   view->base_level = base_level;
   view->last_level = last_level;
   view->max_mip = max_mip;
   view->min_lod = surface_field(words[1], 8, 12);
   view->pitch = pitch;
   view->has_meta = surface_field(words[6], 21, 1) != 0;
   if (view->has_meta)
      view->meta = ((uint64_t)surface_field(words[6], 24, 8) | ((uint64_t)words[7] << 8)) << 8;
   return true;
}

enum agc_surface_use
agc_image_view_surface_use(const struct agc_image_view *view)
{
   switch (view->type) {
   case SURFACE_TYPE_1D:
   case SURFACE_TYPE_1D_ARRAY:
      return AGC_SURFACE_USE_1D;
   case SURFACE_TYPE_3D:
      return AGC_SURFACE_USE_VOLUME;
   case SURFACE_TYPE_2D_MSAA:
   case SURFACE_TYPE_2D_MSAA_ARRAY:
      return AGC_SURFACE_USE_MSAA;
   default:
      return AGC_SURFACE_USE_THIN_2D;
   }
}

bool
agc_image_view_surface_desc(const struct agc_image_view *view, struct agc_surface_desc *desc, char *error,
                            size_t error_bytes)
{
   if (!view || !desc)
      return surface_fail(error, error_bytes, "a surface description needs a view and a description to fill");
   memset(desc, 0, sizeof(*desc));
   struct agc_texel_format texel;
   if (!agc_texel_format_from_gfx10(view->format, &texel, error, error_bytes))
      return false;
   const bool arrayed = view->type == SURFACE_TYPE_CUBE || view->type == SURFACE_TYPE_1D_ARRAY ||
                        view->type == SURFACE_TYPE_2D_ARRAY || view->type == SURFACE_TYPE_2D_MSAA_ARRAY;
   desc->swizzle_mode = view->swizzle_mode;
   desc->width = surface_div_ceil(view->width, texel.block_width);
   desc->height = surface_div_ceil(view->height, texel.block_height);
   desc->depth = view->depth;
   desc->layers = arrayed ? view->last_array + 1 : 1;
   desc->mip_levels = surface_max(view->max_mip, view->last_level) + 1;
   desc->bytes_per_element = texel.bytes_per_element;
   desc->pitch_elements = view->pitch;
   desc->volume = view->type == SURFACE_TYPE_3D;
   desc->pipe_bank_xor = view->pipe_bank_xor;
   return true;
}

uint32_t
agc_image_view_first_reachable_level(const struct agc_image_view *view)
{
   return surface_min(surface_max(view->base_level, view->min_lod >> 8), view->last_level);
}

bool
agc_image_view_samples_in_place(const struct agc_image_view *view)
{
   return agc_surface_samples_in_place_for_format(view->type, view->swizzle_mode, view->format);
}

uint32_t
agc_image_view_host_swizzle_mode(const struct agc_image_view *view)
{
   if (agc_image_view_samples_in_place(view))
      return view->swizzle_mode;
   return agc_surface_host_swizzle_mode(view->swizzle_mode, agc_image_view_surface_use(view));
}

bool
agc_surface_native_dcc(void)
{
   static int native = -1;
   if (native < 0) {
      const char *value = getenv("AGC_NATIVE_DCC");
      native = !value || !*value || *value != '0';
   }
   return native != 0;
}

#define AGC_SURFACE_DCC_TARGETS 64u

static struct {
   simple_mtx_t lock;
   uint32_t count;
   uint32_t next;
   uint64_t va[AGC_SURFACE_DCC_TARGETS];
   uint64_t meta[AGC_SURFACE_DCC_TARGETS];
   uint32_t control[AGC_SURFACE_DCC_TARGETS];
} dcc_targets = {
   .lock = SIMPLE_MTX_INITIALIZER,
};

void
agc_surface_note_dcc_target(uint64_t guest_va, uint64_t meta_va, uint32_t fdcc_control)
{
   simple_mtx_lock(&dcc_targets.lock);
   uint32_t i = 0;
   while (i < dcc_targets.count && dcc_targets.va[i] != guest_va)
      i++;
   if (i == dcc_targets.count) {
      if (dcc_targets.count < AGC_SURFACE_DCC_TARGETS) {
         dcc_targets.count++;
      } else {
         i = dcc_targets.next;
         dcc_targets.next = (dcc_targets.next + 1) % AGC_SURFACE_DCC_TARGETS;
      }
      dcc_targets.va[i] = guest_va;
   }
   dcc_targets.meta[i] = meta_va;
   dcc_targets.control[i] = fdcc_control;
   simple_mtx_unlock(&dcc_targets.lock);
}

bool
agc_surface_dcc_target(uint64_t guest_va, uint64_t *meta_va, uint32_t *fdcc_control)
{
   bool found = false;
   simple_mtx_lock(&dcc_targets.lock);
   for (uint32_t i = 0; i < dcc_targets.count; i++) {
      if (dcc_targets.va[i] == guest_va) {
         *meta_va = dcc_targets.meta[i];
         *fdcc_control = dcc_targets.control[i];
         found = true;
         break;
      }
   }
   simple_mtx_unlock(&dcc_targets.lock);
   return found;
}
