#include "agc_isa_descriptors.h"

#include <string.h>

#define IMAGE_ENTRIES 512
#define MODE_ENTRIES 4096
#define BUFFER_ENTRIES 128
#define SHADOW_REDIRECT 1u
#define CLEAR_PITCH 2u
#define FORMAT_COUNT 128

static const struct {
   uint16_t gfx10;
   uint8_t gfx11;
} formats[FORMAT_COUNT] = {
   {0, 0},     {1, 1},     {2, 2},     {3, 3},     {4, 4},     {5, 5},     {6, 6},     {7, 7},
   {8, 8},     {9, 9},     {10, 10},   {11, 11},   {12, 12},   {13, 13},   {14, 14},   {15, 15},
   {16, 16},   {17, 17},   {18, 18},   {19, 19},   {20, 20},   {21, 21},   {22, 22},   {23, 23},
   {24, 24},   {25, 25},   {26, 26},   {27, 27},   {28, 28},   {29, 29},   {36, 30},   {43, 31},
   {44, 32},   {45, 33},   {48, 34},   {49, 35},   {50, 36},   {51, 37},   {52, 38},   {53, 39},
   {54, 40},   {55, 41},   {56, 42},   {57, 43},   {58, 44},   {59, 45},   {60, 46},   {61, 47},
   {62, 48},   {63, 49},   {64, 50},   {65, 51},   {66, 52},   {67, 53},   {68, 54},   {69, 55},
   {70, 56},   {71, 57},   {72, 58},   {73, 59},   {74, 60},   {75, 61},   {76, 62},   {77, 63},
   {128, 64},  {129, 65},  {130, 66},  {131, 227}, {132, 67},  {133, 68},  {134, 69},  {135, 70},
   {136, 71},  {137, 72},  {138, 73},  {139, 74},  {140, 75},  {141, 76},  {142, 77},  {143, 78},
   {144, 79},  {145, 80},  {146, 81},  {147, 82},  {148, 83},  {149, 84},  {150, 85},  {151, 86},
   {152, 87},  {153, 88},  {154, 89},  {169, 109}, {170, 110}, {171, 111}, {172, 112}, {173, 113},
   {174, 114}, {175, 115}, {176, 116}, {177, 117}, {178, 118}, {179, 119}, {180, 120}, {181, 121},
   {182, 122}, {265, 207}, {266, 208}, {267, 209}, {268, 210}, {269, 211}, {270, 212}, {271, 213},
   {272, 214}, {273, 215}, {274, 216}, {275, 217}, {276, 218}, {277, 219}, {278, 220}, {279, 90},
   {280, 91},  {281, 92},  {282, 93},  {283, 94},  {284, 95},  {285, 228}, {286, 205}, {292, 206},
};

static bool
gfx11_format(uint16_t gfx10, uint8_t *gfx11)
{
   for (unsigned i = 0; i < FORMAT_COUNT; i++) {
      if (formats[i].gfx10 == gfx10) {
         *gfx11 = formats[i].gfx11;
         return true;
      }
   }
   return false;
}

static bool
samples_in_place(uint8_t sw)
{
   return sw == 0;
}

static bool
direct_d_type(uint8_t type, uint8_t sw)
{
   return (type == 9 || type == 11 || type == 13) && sw == 10;
}

static bool
direct_d_format(uint16_t format)
{
   return (format >= 62 && format <= 71) || (format >= 75 && format <= 77);
}

static uint8_t
host_swizzle_mode(uint8_t type, uint8_t sw)
{
   static const uint8_t modes_3d[12] = {0, 5, 9, 17, 21, 24, 25, 26, 27, 28, 29, 31};
   if (sw == 0)
      return 0;
   if (type == 10) {
      for (unsigned i = 0; i < 12; i++) {
         if (modes_3d[i] == sw)
            return sw;
      }
      return 9;
   }
   if (type == 8 || type == 12 || type == 14 || type == 15)
      return sw == 24 ? 24 : 27;
   if (sw == 1 || sw == 5 || sw == 9 || sw == 17 || sw == 21 || sw == 25)
      return (uint8_t)(sw + 1);
   return sw;
}

static uint32_t
xor_bits(uint8_t sw)
{
   if (sw >= 20 && sw <= 23)
      return 0x0f;
   if ((sw >= 16 && sw <= 19) || (sw >= 24 && sw <= 31))
      return 0xff;
   return 0;
}

static uint32_t
swizzle_entry(uint32_t index)
{
   uint8_t type = (uint8_t)(index >> 8);
   uint8_t sw = (uint8_t)(index & 0x1f);
   uint32_t entry = (uint32_t)host_swizzle_mode(type, sw) << 20;
   bool in_place = samples_in_place(sw);
   if (!in_place)
      entry |= SHADOW_REDIRECT | (xor_bits(sw) << AGC_ISA_DESCRIPTOR_XOR_SHIFT);
   if (!in_place && (type == 8 || type == 9 || type == 14))
      entry |= CLEAR_PITCH;
   if (direct_d_type(type, sw))
      entry |= AGC_ISA_DESCRIPTOR_DIRECT_D;
   return entry;
}

static uint32_t
image_entry(uint32_t format)
{
   uint8_t gfx11;
   if (!gfx11_format((uint16_t)format, &gfx11))
      return 0;
   return ((uint32_t)gfx11 << 20) | (direct_d_format((uint16_t)format) ? AGC_ISA_DESCRIPTOR_DIRECT_D : 0);
}

static void
put_word(uint8_t *lut, size_t index, uint32_t word)
{
   lut[4 * index] = (uint8_t)word;
   lut[4 * index + 1] = (uint8_t)(word >> 8);
   lut[4 * index + 2] = (uint8_t)(word >> 16);
   lut[4 * index + 3] = (uint8_t)(word >> 24);
}

void
agc_isa_descriptor_lut(uint8_t lut[AGC_ISA_DESCRIPTOR_LUT_BYTES])
{
   memset(lut, 0, AGC_ISA_DESCRIPTOR_LUT_BYTES);
   for (unsigned i = 0; i < FORMAT_COUNT; i++) {
      uint16_t old = formats[i].gfx10;
      put_word(lut, old, image_entry(old));
      if (old < BUFFER_ENTRIES)
         put_word(lut, IMAGE_ENTRIES + MODE_ENTRIES + old, (uint32_t)formats[i].gfx11 << 12);
   }
   for (uint32_t index = 0; index < MODE_ENTRIES; index++)
      put_word(lut, IMAGE_ENTRIES + index, swizzle_entry(index));
}
