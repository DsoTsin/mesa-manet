#include "pan_graph_v15.h"
#include <assert.h>
#include <string.h>

static const struct pan_graph_v15_section sections[] = {
   {.type = 1, .common = {.registers = 7, .tsu_offset = 0, .instance = 0, .input = {4294967295u, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 4294967295u}, .output = 2u, .suppress_registers = false},
    .memory = {.domain = 0, .domain_bias = 0, .offset = 2, .level = 5, .override = 1, .flags = 0, .value = 1, .extra = 0}},
   {.type = 1, .common = {.registers = 7, .tsu_offset = 8, .instance = 1, .input = {4294967295u, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 4294967295u}, .output = 4u, .suppress_registers = false},
    .memory = {.domain = 0, .domain_bias = 0, .offset = 0, .level = 5, .override = 1, .flags = 0, .value = 1, .extra = 0}},
   {.type = 1, .common = {.registers = 1, .tsu_offset = 28, .instance = 2, .input = {4294967295u, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 4294967295u}, .output = 0u, .suppress_registers = false},
    .memory = {.domain = 2, .domain_bias = 0, .offset = 0, .level = 5, .override = 1, .flags = 0, .value = 1, .extra = 0}},
   {.type = 4, .common = {.registers = 7, .tsu_offset = 0, .instance = 0, .input = {2u, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 4294967295u}, .output = 1u, .suppress_registers = false},
    .transform = {.format = 8, .mode = 0, .immediate = 0, .plane = 1}},
   {.type = 2, .common = {.registers = 7, .tsu_offset = 7, .instance = 0, .input = {4u, 4294967295u, 1u, 4294967295u, 4294967295u, 4294967295u}, .output = 3u, .suppress_registers = false},
    .ce = {.opcode = 3, .width = 9, .height = 7, .parameters = {2u, 3u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, 1u}, .zero_point = 0, .wide = 0, .flags = {0u, 0u}}},
   {.type = 3, .common = {.registers = 4, .tsu_offset = 40, .instance = 0, .input = {3u, 4294967295u, 4294967295u, 0u, 4294967295u, 4294967295u}, .output = 5u, .suppress_registers = false},
    .rescale = {.mode = 1, .input_zero_point = 0, .output_zero_point = 0, .multiplier = 1, .shift = 0, .precision = 0, .coefficient_binding = 0u}},
   {.type = 6, .common = {.registers = 4, .tsu_offset = 40, .instance = 0, .input = {5u, 4294967295u, 4294967295u, 4294967295u, 4294967295u, 4294967295u}, .output = 4294967295u, .suppress_registers = false},
    .memory = {.domain = 3, .domain_bias = 0, .offset = 0, .level = 5, .override = 0, .flags = 0, .value = 0, .extra = 0}},
};

static const struct pan_graph_v15_nrt resources[] = {
   {.kind = 2, .width = 2, .size = 0, .flags = 0, .stride = 0, .tile = 0, .mode = 0, .dimensions = {0u, 0u, 0u, 0u}, .format = 0, .packed = false, .shared_buffer = false, .conv3d = false, .plane = 0, .wide = false, .binding = 1, .transfer = {1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u}},
   {.kind = 0, .width = 2, .size = 384, .flags = 5, .stride = 0, .tile = 1, .mode = 0, .dimensions = {0u, 0u, 0u, 0u}, .format = 1, .packed = true, .shared_buffer = false, .conv3d = false, .plane = 0, .wide = false, .binding = 7, .transfer = {1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u}},
   {.kind = 0, .width = 16, .size = 480, .flags = 0, .stride = 0, .tile = 1, .mode = 0, .dimensions = {3u, 1u, 1u, 3u}, .format = 0, .packed = false, .shared_buffer = false, .conv3d = false, .plane = 0, .wide = false, .binding = 7, .transfer = {1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u}},
   {.kind = 1, .width = 2, .size = 0, .flags = 0, .stride = 0, .tile = 0, .mode = 0, .dimensions = {2u, 1u, 2u, 2u}, .format = 0, .packed = false, .shared_buffer = false, .conv3d = false, .plane = 1, .wide = false, .binding = 4, .transfer = {1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u}},
   {.kind = 0, .width = 16, .size = 288, .flags = 1, .stride = 0, .tile = 1, .mode = 0, .dimensions = {9u, 1u, 1u, 9u}, .format = 0, .packed = false, .shared_buffer = false, .conv3d = false, .plane = 0, .wide = false, .binding = 7, .transfer = {1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u}},
   {.kind = 0, .width = 16, .size = 704, .flags = 6, .stride = 0, .tile = 2, .mode = 0, .dimensions = {21u, 1u, 1u, 7u}, .format = 2, .packed = false, .shared_buffer = false, .conv3d = false, .plane = 0, .wide = false, .binding = 4, .transfer = {1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u}},
};

static const struct pan_graph_v15_tsu_instr tsu[] = {
   {4u, 0u, 0u, 1u, 0u, 0u},
   {1u, 1u, 0u, 5u, 0u, 0u},
   {1u, 1u, 0u, 6u, 0u, 0u},
   {1u, 1u, 0u, 4u, 0u, 0u},
   {17u, 1u, 0u, 0u, 0u, 0u},
   {17u, 1u, 0u, 0u, 0u, 0u},
   {17u, 0u, 0u, 0u, 0u, 0u},
   {21u, 0u, 0u, 0u, 0u, 0u},
   {1u, 1u, 0u, 1u, 0u, 0u},
   {1u, 0u, 0u, 2u, 0u, 0u},
   {3u, 0u, 0u, 5u, 0u, 0u},
   {20u, 0u, 0u, 0u, 1u, 0u},
   {12u, 1u, 0u, 0u, 6u, 0u},
   {1u, 0u, 0u, 3u, 0u, 0u},
   {3u, 0u, 0u, 6u, 0u, 0u},
   {20u, 0u, 0u, 0u, 1u, 0u},
   {12u, 1u, 0u, 0u, 8u, 0u},
   {1u, 1u, 0u, 4u, 0u, 0u},
   {17u, 1u, 0u, 0u, 0u, 0u},
   {17u, 1u, 0u, 0u, 0u, 0u},
   {17u, 0u, 0u, 0u, 0u, 0u},
   {21u, 0u, 0u, 0u, 0u, 0u},
   {21u, 0u, 0u, 0u, 0u, 0u},
   {21u, 0u, 0u, 0u, 0u, 0u},
   {1u, 0u, 0u, 3u, 0u, 0u},
   {17u, 1u, 0u, 0u, 0u, 0u},
   {17u, 1u, 0u, 0u, 0u, 0u},
   {17u, 0u, 0u, 0u, 0u, 0u},
   {4u, 0u, 0u, 4u, 0u, 0u},
   {17u, 1u, 0u, 0u, 0u, 0u},
   {17u, 1u, 0u, 0u, 0u, 0u},
   {17u, 0u, 0u, 0u, 0u, 0u},
   {21u, 0u, 0u, 0u, 0u, 0u},
   {21u, 0u, 0u, 0u, 0u, 0u},
   {21u, 0u, 0u, 0u, 0u, 0u},
   {21u, 0u, 0u, 0u, 0u, 0u},
   {1u, 1u, 0u, 1u, 0u, 0u},
   {1u, 1u, 0u, 2u, 0u, 0u},
   {1u, 0u, 0u, 3u, 0u, 0u},
   {4u, 0u, 0u, 4u, 0u, 0u},
   {17u, 1u, 0u, 0u, 0u, 0u},
   {17u, 1u, 0u, 0u, 0u, 0u},
   {17u, 0u, 0u, 0u, 0u, 0u},
   {21u, 0u, 0u, 0u, 0u, 0u},
};

static const uint8_t expected[] = {
   0x4c, 0x14, 0x01, 0x30, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x05, 0x10, 0x00, 0x2f, 0x07, 0x30, 0x00, 0x41, 0x03, 0x30, 0x00, 0x00,
   0x02, 0x10, 0x00, 0x00, 0x07, 0x20, 0x04, 0xf2, 0x02, 0x00, 0x00, 0x00, 0x87, 0x40, 0x14, 0xf2,
   0x00, 0x00, 0x00, 0x00, 0xc1, 0x01, 0x26, 0xf2, 0x00, 0x00, 0x00, 0x00, 0x84, 0xb2, 0x05, 0xf0,
   0x00, 0x00, 0x00, 0x00, 0x77, 0x30, 0x04, 0x13, 0x08, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x40, 0x08, 0x00, 0x00, 0x07, 0x10, 0x02, 0x0c, 0x00, 0x04, 0x00, 0x00,
   0x84, 0x52, 0x03, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00,
   0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x11, 0x02, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x17, 0x14, 0x00, 0x0b,
   0x17, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf7, 0x00, 0x00, 0x0e, 0x10, 0x00, 0x03, 0x00,
   0x01, 0x00, 0x03, 0x00, 0x14, 0x01, 0x00, 0x00, 0x10, 0x00, 0x01, 0x00, 0x02, 0x00, 0x02, 0x02,
   0xf7, 0x04, 0x00, 0x08, 0x10, 0x00, 0x09, 0x00, 0x01, 0x00, 0x09, 0x00, 0xf4, 0x38, 0x00, 0x0a,
   0x18, 0x00, 0x15, 0x00, 0x01, 0x00, 0x07, 0x00, 0x07, 0x95, 0x99, 0x91, 0xe2, 0xe2, 0x62, 0x63,
   0x85, 0x09, 0x16, 0x6c, 0x01, 0xd8, 0x06, 0x0d, 0x1a, 0x6c, 0x01, 0xd8, 0x08, 0x91, 0xe2, 0xe2,
   0x62, 0x63, 0x63, 0x63, 0x0d, 0xe2, 0xe2, 0x62, 0x13, 0xe2, 0xe2, 0x62, 0x63, 0x63, 0x63, 0x63,
   0x85, 0x89, 0x0d, 0x13, 0xe2, 0xe2, 0x62, 0x63,
};

static void test_packet(void)
{
   uint8_t bytes[sizeof(expected) + 64];
   memset(bytes, 0xa5, sizeof(bytes));
   const uint32_t counts[8] = {0u, 3u, 1u, 1u, 1u, 1u, 0u, 6u};
   const struct pan_graph_v15_header header = {.transfer = {5u, 1u, 7u, 3u, 3u, 3u, 2u, 1u}, .output = {5u, 1u, 7u, 9u, 3u, 3u, 2u, 1u}};
   struct pan_graph_v15_layout layout;
   assert(pan_graph_v15_layout_init(&layout, counts, 48));
   assert(layout.size == sizeof(expected));
   assert(pan_graph_v15_pack_header(bytes + 32, 36, &layout, &header));
   static const unsigned type_index[8] = {0, 1, 3, 5, 4, 0, 2, 6};
   unsigned emitted[8] = {0};
   for (unsigned i = 0; i < sizeof(sections) / sizeof(sections[0]); i++) {
      unsigned slot = type_index[sections[i].type];
      unsigned stride = slot == 3 ? 20 : slot == 5 ? 32 : 8;
      size_t offset = layout.offset[slot] + stride * emitted[slot]++;
      assert(pan_graph_v15_pack_section(bytes + 32 + offset, stride, &sections[i]));
   }
   for (unsigned i = 0; i < sizeof(resources) / sizeof(resources[0]); i++)
      assert(pan_graph_v15_pack_nrt(bytes + 32 + layout.offset[7] + 12 * i,
                                    12, &resources[i]));
   size_t written;
   assert(pan_graph_v15_pack_tsu(bytes + 32 + layout.tsu_offset, 48, tsu,
                                 sizeof(tsu) / sizeof(tsu[0]), &written));
   assert(written == 48);
   assert(memcmp(bytes + 32, expected, sizeof(expected)) == 0);
   for (unsigned i = 0; i < 32; i++)
      assert(bytes[i] == 0xa5 && bytes[sizeof(bytes) - 1 - i] == 0xa5);
}

static void test_bounds(void)
{
   uint8_t bytes[64], original[64];
   memset(bytes, 0xa5, sizeof(bytes));
   memcpy(original, bytes, sizeof(bytes));
   size_t written = 99;
   struct pan_graph_v15_tsu_instr instruction = {.opcode = 5};
   assert(!pan_graph_v15_pack_tsu(bytes, 1, &instruction, 1, &written));
   assert(written == 0 && !memcmp(bytes, original, sizeof(bytes)));
   instruction.opcode = 2;
   assert(!pan_graph_v15_pack_tsu(bytes, sizeof(bytes), &instruction, 1, &written));
   assert(written == 0 && !memcmp(bytes, original, sizeof(bytes)));
   struct pan_graph_v15_nrt nrt = {.kind = 0, .tile = 0};
   assert(!pan_graph_v15_pack_nrt(bytes, sizeof(bytes), &nrt));
   assert(!memcmp(bytes, original, sizeof(bytes)));
   struct pan_graph_v15_section section = {.type = PAN_GRAPH_V15_MEMORY_INPUT,
      .memory = {.level = 1}};
   assert(!pan_graph_v15_pack_section(bytes, sizeof(bytes), &section));
   assert(!memcmp(bytes, original, sizeof(bytes)));
   section.type = PAN_GRAPH_V15_CE;
   assert(!pan_graph_v15_pack_section(bytes, 19, &section));
   assert(!memcmp(bytes, original, sizeof(bytes)));
   uint32_t counts[8] = {0};
   struct pan_graph_v15_layout layout;
   assert(!pan_graph_v15_layout_init(&layout, counts, 0));
   assert(!pan_graph_v15_layout_init(&layout, counts, 257));
}

static void test_binary(void)
{
   struct pan_graph_v15_section section = {
      .type = PAN_GRAPH_V15_VE,
      .common = {
         .registers = 4,
         .input = {2, 3, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX},
         .output = 5,
      },
      .ve_family = PAN_GRAPH_V15_VE_BINARY,
      .binary = {.op = PAN_GRAPH_V15_BINARY_ADD},
   };
   const uint32_t expected_words[8] = {
      0xf2025004, 0, 15, 1, 0, 1, 3, 1,
   };
   uint32_t words[8];
   assert(pan_graph_v15_pack_section(words, sizeof(words), &section));
   assert(!memcmp(words, expected_words, sizeof(words)));
   section.common.input[1] = section.common.input[0];
   assert(pan_graph_v15_pack_section(words, sizeof(words), &section));
   assert(words[6] == 2);
   uint32_t original[8];
   memcpy(original, words, sizeof(words));
   section.common.input[1] = UINT32_MAX;
   assert(!pan_graph_v15_pack_section(words, sizeof(words), &section));
   assert(!memcmp(words, original, sizeof(words)));
   section.common.input[1] = 3;
   section.common.input[3] = 1;
   assert(!pan_graph_v15_pack_section(words, sizeof(words), &section));
   assert(!memcmp(words, original, sizeof(words)));
   section.common.input[3] = UINT32_MAX;
   section.binary.op = -1;
   assert(!pan_graph_v15_pack_section(words, sizeof(words), &section));
   assert(!memcmp(words, original, sizeof(words)));
   section.binary.op = PAN_GRAPH_V15_BINARY_ADD;
   section.ve_family = -1;
   assert(!pan_graph_v15_pack_section(words, sizeof(words), &section));
   assert(!memcmp(words, original, sizeof(words)));
   section.ve_family = PAN_GRAPH_V15_VE_BINARY;
   assert(!pan_graph_v15_pack_section(words, sizeof(words) - 1, &section));
   assert(!memcmp(words, original, sizeof(words)));
}

int main(void)
{
   test_packet();
   test_bounds();
   test_binary();
   return 0;
}
