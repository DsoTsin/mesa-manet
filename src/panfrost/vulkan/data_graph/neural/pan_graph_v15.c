#include "pan_graph_v15.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static void
store32(void *dst, uint32_t value)
{
   uint8_t *bytes = dst;
   for (unsigned i = 0; i < 4; i++)
      bytes[i] = value >> (i * 8);
}

static void
store_words(void *dst, const uint32_t *words, unsigned count)
{
   for (unsigned i = 0; i < count; i++)
      store32((uint8_t *)dst + i * 4, words[i]);
}

bool
pan_graph_v15_layout_init(struct pan_graph_v15_layout *layout,
                         const uint32_t counts[8], size_t tsu_size)
{
   static const unsigned strides[8] = {8, 8, 8, 20, 8, 32, 20, 12};
   if (!tsu_size || tsu_size > 256)
      return false;

   struct pan_graph_v15_layout result = {.size = PAN_GRAPH_V15_HEADER_SIZE};
   for (unsigned i = 0; i < 8; i++) {
      if (counts[i] > (SIZE_MAX - result.size) / strides[i])
         return false;
      result.count[i] = counts[i];
      result.offset[i] = result.size;
      result.size += (size_t)counts[i] * strides[i];
   }
   if (result.size > SIZE_MAX - tsu_size)
      return false;
   result.tsu_offset = result.size;
   result.size += tsu_size;
   *layout = result;
   return true;
}

bool
pan_graph_v15_pack_header(void *dst, size_t size,
                         const struct pan_graph_v15_layout *layout,
                         const struct pan_graph_v15_header *info)
{
   if (size < PAN_GRAPH_V15_HEADER_SIZE ||
       layout->size <= layout->tsu_offset ||
       layout->size - layout->tsu_offset > 256)
      return false;

   const uint32_t *count = layout->count;
   uint32_t words[9] = {
      ((count[7] & 15) << 27) | ((count[3] & 1) << 10) | (count[0] & 1) |
         ((count[1] & 7) << 2) | ((count[4] & 7) << 12) |
         ((count[2] & 7) << 6) | ((count[5] & 7) << 16) |
         ((count[6] & 1) << 20),
   };
   for (unsigned i = 0; i < 4; i++)
      words[i + 1] = info->output[i + 4] == UINT32_MAX
                        ? 0 : info->output[i + 4] - 1;
   for (unsigned i = 0; i < 8; i++) {
      uint32_t extent = info->transfer[i] == UINT32_MAX
                           ? 1 : info->transfer[i] & 4095;
      words[5 + i / 2] |= extent << (12 * (i & 1));
   }
   words[5] |= ((layout->size - layout->tsu_offset - 1) & 255) << 24;
   words[6] |= 0x41000000;
   store_words(dst, words, 9);
   return true;
}

static unsigned
tsu_instruction(const struct pan_graph_v15_tsu_instr *info, uint8_t bytes[2])
{
   uint32_t opcode = info->opcode;
   uint32_t inc = info->increment << 7;
   uint32_t reg = (info->reg & 7) << 2;
   uint32_t flag = (info->flag & 1) << 2;
   uint32_t mode = (info->mode >> 3) & 3;
   bytes[1] = info->immediate;
   switch (opcode) {
   case 0: bytes[0] = reg | inc; return 1;
   case 1: bytes[0] = reg | inc | 1; return 1;
   case 3: bytes[0] = reg | inc | 2; return 1;
   case 4: bytes[0] = reg | 3; return 1;
   case 5: bytes[0] = reg | inc | mode | 0x20; return 2;
   case 6: bytes[0] = flag | inc | mode | 0x40; return 2;
   case 7: bytes[0] = flag | inc | mode | 0x48; return 2;
   case 8: case 9: case 10: case 11:
      bytes[0] = flag | inc | (0x50 + opcode - 8); return 2;
   case 12: bytes[0] = inc | mode | 0x58; return 2;
   case 13: bytes[0] = inc | mode | 0x5c; return 2;
   case 14: bytes[0] = inc | 0x5f; return 2;
   case 15: case 16: case 17:
      bytes[0] = inc | (0x60 + opcode - 15); return 1;
   case 18: case 19: case 20:
      bytes[0] = mode | inc | (0x64 + 4 * (opcode - 18)); return 2;
   case 21: bytes[0] = 0x63; return 1;
   default: return 0;
   }
}

bool
pan_graph_v15_pack_tsu(void *dst, size_t size,
                      const struct pan_graph_v15_tsu_instr *instructions,
                      size_t count, size_t *written)
{
   size_t length = 0;
   *written = 0;
   for (size_t i = 0; i < count; i++) {
      uint8_t bytes[2];
      unsigned n = tsu_instruction(&instructions[i], bytes);
      if (!n || n > size - length)
         return false;
      length += n;
   }
   uint8_t *bytes = dst;
   length = 0;
   for (size_t i = 0; i < count; i++) {
      uint8_t code[2];
      unsigned n = tsu_instruction(&instructions[i], code);
      memcpy(bytes + length, code, n);
      length += n;
   }
   *written = length;
   return true;
}

static enum pan_graph_tsu_status
encode_program(void *dst, size_t size, const struct pan_graph_tsu_program *program,
               size_t *written)
{
   *written = 0;
   uint8_t bytes[256];
   size_t length = 0;
   for (size_t i = 0; i < program->count; i++) {
      const struct pan_graph_tsu_instr *raw = &program->instructions[i];
      const struct pan_graph_v15_tsu_instr instr = {
         .opcode = raw->opcode, .increment = raw->increment,
         .flag = raw->flag, .reg = raw->reg,
         .immediate = raw->immediate, .mode = raw->shift,
      };
      uint8_t code[2];
      unsigned n = tsu_instruction(&instr, code);
      if (!n || n > sizeof(bytes) - length)
         return PAN_GRAPH_TSU_UNSUPPORTED;
      memcpy(bytes + length, code, n);
      length += n;
   }
   if (size < length)
      return PAN_GRAPH_TSU_INVALID;
   memcpy(dst, bytes, length);
   *written = length;
   return PAN_GRAPH_TSU_SUCCESS;
}

enum pan_graph_tsu_status
pan_graph_v15_compile_tsu(void *dst, size_t size,
                          const struct pan_graph_tsu_info *info, size_t *written)
{
   if (!written)
      return PAN_GRAPH_TSU_INVALID;
   *written = 0;
   if (!dst)
      return PAN_GRAPH_TSU_INVALID;
   struct pan_graph_tsu_program program = {0};
   uint32_t offset;
   enum pan_graph_tsu_status status = pan_graph_tsu_link(info, 1, &program, &offset);
   if (status == PAN_GRAPH_TSU_SUCCESS)
      status = encode_program(dst, size, &program, written);
   pan_graph_tsu_cleanup(&program);
   return status;
}

enum pan_graph_tsu_status
pan_graph_v15_compile(const struct pan_graph_v15_program_info *info,
                      struct pan_graph_v15_program *program)
{
   if (!info || !program || !info->tsu_count || !info->tsu ||
       (info->section_count && (!info->sections || !info->section_tsu)) ||
       (info->resource_count && !info->resources) || info->tsu_count > UINT32_MAX)
      return PAN_GRAPH_TSU_INVALID;
   if (info->tsu_count > SIZE_MAX / sizeof(uint32_t))
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   uint32_t *tsu_offsets = malloc(info->tsu_count * sizeof(*tsu_offsets));
   if (!tsu_offsets)
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   struct pan_graph_tsu_program linked = {0};
   enum pan_graph_tsu_status status =
      pan_graph_tsu_link(info->tsu, info->tsu_count, &linked, tsu_offsets);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      goto out;
   uint8_t tsu[256];
   size_t tsu_size = 0;
   status = encode_program(tsu, sizeof(tsu), &linked, &tsu_size);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      goto out;
   static const unsigned type_index[8] = {0, 1, 3, 5, 4, 0, 2, 6};
   static const unsigned strides[8] = {8, 8, 8, 20, 8, 32, 20, 12};
   uint32_t counts[8] = {0};
   for (size_t i = 0; i < info->section_count; i++) {
      unsigned type = info->sections[i].type;
      if (!type || type >= 8 || counts[type_index[type]] == UINT32_MAX ||
          info->section_tsu[i] >= info->tsu_count) {
         status = PAN_GRAPH_TSU_INVALID;
         goto out;
      }
      counts[type_index[type]]++;
   }
   if (info->resource_count > 15 || counts[0] > 1 || counts[1] > 7 ||
       counts[2] > 7 || counts[3] > 1 || counts[4] > 7 ||
       counts[5] > 7 || counts[6] > 1) {
      status = PAN_GRAPH_TSU_UNSUPPORTED;
      goto out;
   }
   counts[7] = info->resource_count;
   struct pan_graph_v15_layout layout;
   if (!pan_graph_v15_layout_init(&layout, counts, tsu_size)) {
      status = PAN_GRAPH_TSU_INVALID;
      goto out;
   }
   uint8_t *data = malloc(layout.size);
   if (!data) {
      status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
      goto out;
   }
   bool valid = pan_graph_v15_pack_header(data, layout.size, &layout, &info->header);
   uint32_t emitted[8] = {0};
   for (size_t i = 0; i < info->section_count && valid; i++) {
      struct pan_graph_v15_section section = info->sections[i];
      section.common.tsu_offset = tsu_offsets[info->section_tsu[i]];
      unsigned slot = type_index[section.type];
      size_t offset = layout.offset[slot] + strides[slot] * emitted[slot]++;
      valid = pan_graph_v15_pack_section(data + offset, strides[slot], &section);
   }
   for (size_t i = 0; i < info->resource_count && valid; i++)
      valid = pan_graph_v15_pack_nrt(data + layout.offset[7] + i * 12, 12,
                                     &info->resources[i]);
   if (!valid) {
      free(data);
      status = PAN_GRAPH_TSU_INVALID;
      goto out;
   }
   memcpy(data + layout.tsu_offset, tsu, tsu_size);
   *program = (struct pan_graph_v15_program){layout.size, data};
out:
   pan_graph_tsu_cleanup(&linked);
   free(tsu_offsets);
   return status;
}

void
pan_graph_v15_program_cleanup(struct pan_graph_v15_program *program)
{
   free(program->data);
   *program = (struct pan_graph_v15_program){0};
}

static uint32_t
common_header(const struct pan_graph_v15_common *info)
{
   return (info->suppress_registers ? 0 : info->registers & 15) |
          (info->tsu_offset << 4) | ((info->instance & 7) << 20) |
          ((info->output & 15) << 12);
}

static bool
pack_memory(uint32_t words[2], const struct pan_graph_v15_section *info)
{
   const struct pan_graph_v15_memory *memory = &info->memory;
   const struct pan_graph_v15_common *common = &info->common;
   if (memory->level < 2 || memory->level > 5)
      return false;

   static const unsigned low_default[4] = {0, 1, 2, 2};
   static const unsigned high_default[4] = {0, 0, 1, 0};
   unsigned low = memory->flags || memory->value
                     ? memory->value : low_default[memory->level - 2];
   unsigned high = memory->override ? 1 : high_default[memory->level - 2];
   unsigned domain = (memory->domain + memory->domain_bias) & 3;
   bool output = info->type == PAN_GRAPH_V15_MEMORY_OUTPUT;
   bool cache = output ? memory->extra & 1
                      : ((memory->override & 1) || memory->flags || memory->value)
                           ? !(memory->flags || memory->value)
                           : memory->level == 4 || memory->level == 5;
   words[0] = common_header(common);
   if (output)
      words[0] = (words[0] & ~(15u << 12)) |
                 ((common->input[0] & 15) << 16);
   words[0] |= domain << (output ? 12 : 16);
   words[0] |= (low & 3) << (output ? 14 : 18);
   words[0] |= high << 25 | (uint32_t)cache << 24;
   if (info->type == PAN_GRAPH_V15_MEMORY_CONSTANT) {
      int32_t count = common->registers;
      if (count > 4)
         count = 4;
      words[0] = (words[0] & 0x3ffffff) | ((uint32_t)count << 29);
   } else {
      words[0] |= (common->input[5] & 15) << 28;
   }
   words[1] = memory->offset;
   return true;
}

static uint32_t
ce_nibble(uint32_t parameter)
{
   int32_t value = parameter;
   return ((value > 16 ? 16u : parameter) - 1) & 15;
}

bool
pan_graph_v15_pack_section(void *dst, size_t size,
                          const struct pan_graph_v15_section *info)
{
   const struct pan_graph_v15_common *common = &info->common;
   if (info->type < 1 || info->type > 7)
      return false;
   static const unsigned lengths[8] = {0, 8, 20, 32, 8, 8, 8, 20};
   if (size < lengths[info->type])
      return false;

   uint32_t words[8] = {common_header(common)};
   if (info->type == PAN_GRAPH_V15_MEMORY_INPUT ||
       info->type == PAN_GRAPH_V15_MEMORY_CONSTANT ||
       info->type == PAN_GRAPH_V15_MEMORY_OUTPUT) {
      if (!pack_memory(words, info))
         return false;
   } else {
      words[0] |= (common->input[0] & 15) << 16;
   }
   switch (info->type) {
   case PAN_GRAPH_V15_CE: {
      const struct pan_graph_v15_ce *ce = &info->ce;
      const uint32_t *p = ce->parameters;
      words[0] |= (ce->opcode >= 1 && ce->opcode <= 4 ? ce->opcode : 0) << 24;
      words[0] |= (common->input[2] & 15) << 28;
      words[1] = ce->width - 1;
      words[2] = ce->height - 1;
      words[3] = ce->zero_point & 255;
      if (ce->has_bias_zero_point)
         words[3] |= (ce->bias_zero_point & 255) << 8;
      words[3] |= ce_nibble(p[3]) << 16 | ce_nibble(p[4]) << 20 |
                  ce_nibble(p[6]) << 24 | ce_nibble(p[7]) << 28;
      words[4] = (ce->input_nearest_2x ? 7 : 0) |
                 (p[10] & 31) << 6 | (p[9] & 31) << 11 |
                 (ce->wide & 1) << 16 | (ce->flags[1] & 1) << 17 |
                 (ce->flags[0] & 1) << 18;
      break;
   }
   case PAN_GRAPH_V15_VE: {
      if (info->ve_family == PAN_GRAPH_V15_VE_BINARY) {
         if (info->binary.op != PAN_GRAPH_V15_BINARY_ADD ||
             common->input[0] == UINT32_MAX ||
             common->input[1] == UINT32_MAX || common->output == UINT32_MAX)
            return false;
         for (unsigned i = 2; i < 6; i++) {
            if (common->input[i] != UINT32_MAX)
               return false;
         }
         words[0] |= 0x02000000 | (15u << 28);
         words[2] = 15;
         words[3] = words[5] = words[7] = 1;
         words[6] = common->input[1] & 15;
         break;
      }
      if (info->ve_family != PAN_GRAPH_V15_VE_RESCALE)
         return false;
      const struct pan_graph_v15_rescale *rescale = &info->rescale;
      bool external = rescale->coefficient_binding != UINT32_MAX;
      words[0] |= (common->input[4] & 15) << 28 |
                  (uint32_t)rescale->clamp << 27;
      words[1] = (rescale->minimum & 65535) |
                 ((uint32_t)rescale->maximum & 65535) << 16;
      words[2] = (rescale->mode >= 1 && rescale->mode <= 5 ? rescale->mode : 0) << 29;
      words[2] |= (rescale->precision & 31) << 24;
      words[2] |= rescale->output_zero_point == 0x8000
                     ? 0x20000 : (rescale->output_zero_point & 511) << 8;
      words[2] |= external ? rescale->coefficient_binding & 15 : 15;
      if (!external) {
         words[2] |= (rescale->shift & 63) << 18;
         words[3] = rescale->multiplier;
      }
      words[4] = rescale->input_zero_point == 0x8000
                     ? 0x20000 : (rescale->input_zero_point & 255) << 8;
      words[6] = common->input[1] & 15;
      break;
   }
   case PAN_GRAPH_V15_TRANSFORM: {
      static const unsigned formats[9] = {4, 5, 6, 7, 8, 9, 10, 12, 13};
      const struct pan_graph_v15_transform *transform = &info->transform;
      unsigned format = transform->format;
      words[0] |= (format >= 1 && format <= 9 ? formats[format - 1] : 1) << 24;
      if ((format >= 1 && format <= 5) || format == 9) {
         unsigned count = transform->count & 255;
         if (count >= 1 && count <= 15)
            words[0] |= count << 28;
      } else if (format == 0) {
         unsigned swizzle = transform->swizzle & 255;
         switch (swizzle) {
         case 0x6c: case 0x78: case 0x9c: case 0xb4: case 0xc6: case 0xd8: case 0xe1:
            words[1] = swizzle;
            break;
         default:
            words[1] = 0x27;
            break;
         }
      } else if (format == 7) {
         words[1] = transform->value;
      } else if (format == 8) {
         words[1] = transform->mode == 1 ? 512 : transform->mode == 2 ? 768 : 0;
         words[1] |= transform->immediate & 255;
         if (transform->plane >= 1 && transform->plane <= 3)
            words[1] |= transform->plane << 10;
      }
      break;
   }
   case PAN_GRAPH_V15_SAD: {
      const struct pan_graph_v15_sad *sad = &info->sad;
      const uint32_t *p = sad->parameters;
      words[0] |= (common->input[1] & 15) << 28 |
                  (uint32_t)sad->motion_vector << 24 | (uint32_t)!!p[12] << 25;
      words[1] = sad->width - 1;
      words[2] = sad->height - 1;
      words[3] = (p[1] & 7) | (p[0] & 7) << 8 | ((p[3] & 1) ^ 1) << 5 |
                 ((p[2] & 1) ^ 1) << 13 | (p[9] & 127) << 16 |
                 (p[8] & 127) << 23 | ((p[7] & 1) ^ 1) << 7 |
                 ((p[6] & 1) ^ 1) << 15;
      words[4] = ((p[11] - 1) & 15) | ((p[10] - 1) & 15) << 4;
      break;
   }
   default:
      break;
   }
   store_words(dst, words, lengths[info->type] / 4);
   return true;
}

bool
pan_graph_v15_pack_nrt(void *dst, size_t size,
                      const struct pan_graph_v15_nrt *info)
{
   if (size < 12 || info->kind > 3)
      return false;
   uint32_t words[3] = {
      (info->kind == 3 ? 0 : info->binding & 15) |
         ((info->width - 1) & 15) << 4 | info->kind << 8,
      16,
   };
   switch (info->kind) {
   case 0: {
      if (!info->tile || info->tile > 32)
         return false;
      int32_t quotient = (int32_t)info->size / (32 * (int32_t)info->tile);
      int32_t stride = info->stride / 32;
      words[0] |= (info->flags & 7) << 10 | ((info->tile - 1) & 7) << 13 |
                  ((uint32_t)stride & 255) << 16;
      words[0] = ((words[0] & 0xfffcff) | ((uint32_t)quotient << 24)) - (1u << 24);
      if (info->packed || info->shared_buffer) {
         words[1] = (info->format == 1 ? 6 : info->format == 2 ? 10 : 4) | 17;
         if (!info->packed) {
            const uint32_t *extent = info->transfer;
            uint64_t channels = ((uint64_t)extent[0] + 7) & ~UINT64_C(7);
            uint64_t alignment = info->wide ? 16 : 8;
            uint64_t blocks = ((uint64_t)extent[4] + alignment - 1) & ~(alignment - 1);
            uint64_t volume = (uint64_t)extent[info->conv3d ? 7 : 6] *
                              extent[info->conv3d ? 6 : 5];
            if (volume > INT32_MAX / (channels ? channels : 1))
               return false;
            volume *= channels;
            if (volume > INT32_MAX / (blocks ? blocks : 1))
               return false;
            volume *= blocks;
            uint32_t denominator = info->tile << 8;
            if (volume > INT32_MAX - denominator + 1)
               return false;
            words[1] |= (((volume + denominator - 1) / denominator) & 255) << 24;
         }
      } else {
         unsigned mode = info->mode;
         switch (info->format) {
         case 0: words[1] = mode == 0 ? 16 : 0; break;
         case 1: words[1] = mode == 0 ? 20 : mode == 1 ? 4 : 0; break;
         case 2: words[1] = 24; break;
         case 3: words[1] = 28; break;
         default: words[1] = 0; break;
         }
         words[1] |= (info->dimensions[0] & 2047) << 16;
         words[2] = (info->dimensions[1] & 2047) |
                    (info->dimensions[3] & 2047) << 16;
      }
      break;
   }
   case 1:
      words[1] |= info->format >= 1 && info->format <= 3 ? 2 * info->format : 0;
      if (info->plane >= 1 && info->plane <= 3)
         words[1] |= info->plane << 16;
      words[2] = (info->dimensions[0] & 63) |
                 (info->dimensions[2] & 63) << 16 |
                 (info->dimensions[3] & 63) << 24;
      break;
   case 2:
      words[1] |= info->format == 1 ? 4 : 0;
      break;
   case 3: {
      static const unsigned formats[4] = {4, 8, 12, 14};
      if (info->format >= 1 && info->format <= 4)
         words[1] |= formats[info->format - 1];
      break;
   }
   }
   store_words(dst, words, 3);
   return true;
}
