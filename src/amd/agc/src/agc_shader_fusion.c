#include "agc_shader_fusion.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include "c11/threads.h"

#include "agc_debug.h"
#include "agc_driver.h"
#include "agc_isa.h"
#include "agc_memory.h"
#include "agc_program.h"

#define RSRC1_GS 0x08au
#define RSRC2_GS 0x08bu
#define CHKSUM_HS 0x100u
#define RSRC1_HS 0x10au
#define RSRC2_HS 0x10bu
#define PGM_LO_LS 0x148u
#define PGM_LO_GS 0x088u
#define PGM_LO_ES 0x0c8u
#define S_CODE_END 0xbf9f0000u
#define S_NOP 0xbf800000u
#define S_GETPC_MASK 0xff80ff00u
#define S_GETPC 0xbe801f00u
#define S_SETPC_S6 0xbe802006u
#define S_SWAPPC_NULL_S6 0xbefd2106u
#define CODE_ALIGNMENT 256u
#define TRAILER_CODE_BYTES (8u + 3u * 4u)
#define RSRC1_VGPRS 0x3fu
#define RSRC2_OFFCHIP_LDS (UINT32_C(1) << 18)
#define RSRC2_USER_SGPRS ((UINT32_C(0x1f) << 1) | (UINT32_C(1) << 27))
#define HEADER_KEY_BYTES 0x5du
#define USER_DATA_KEY_BYTES 0x36u
#define MAX_LITERALS 4096u

static const char trailer_magic[8] = {'b', 'a', 'r', 'e', 'f', 'o', 'o', 't'};

struct literal {
   uint32_t offset;
   uint32_t value;
};

struct layout {
   uint32_t front_bytes;
   uint32_t constants_offset;
   uint32_t constants_bytes;
   uint32_t constants_source;
   struct literal *literals;
   uint32_t literal_count;
   uint32_t back_offset;
   uint32_t code_bytes;
   struct agc_register sh[256];
   struct agc_register cx[256];
   uint32_t sh_count;
   uint32_t cx_count;
   size_t sh_offset;
   size_t cx_offset;
   size_t specials_offset;
   size_t inputs_offset;
   size_t outputs_offset;
   size_t user_data_offset;
   size_t direct_offset;
   size_t sharp_offsets[4];
   size_t total_bytes;
};

struct fused {
   struct fused *next;
   uint8_t *key;
   size_t key_bytes;
   uint8_t *storage;
   struct agc_mem code;
   struct agc_shader *header;
};

static struct fused *fused_programs;
static mtx_t fused_lock;
static once_flag fused_once = ONCE_FLAG_INIT;

static void
fused_init(void)
{
   mtx_init(&fused_lock, mtx_plain);
}

static size_t
align_up(size_t value, size_t alignment)
{
   return (value + alignment - 1) & ~(alignment - 1);
}

static uint32_t
read_u32(const uint8_t *bytes)
{
   uint32_t value;
   memcpy(&value, bytes, sizeof(value));
   return value;
}

static bool
fail(char *error, size_t bytes, const char *text)
{
   snprintf(error, bytes, "%s", text);
   return false;
}

static bool
find(const struct agc_register *registers, uint32_t count, uint32_t offset, uint32_t occurrence, uint32_t *at)
{
   for (uint32_t i = 0; registers && i < count; i++) {
      if (registers[i].offset != offset)
         continue;
      if (occurrence--)
         continue;
      *at = i;
      return true;
   }
   return false;
}

static void
merge_max(struct agc_register *destination, const struct agc_register *source, uint32_t shift, uint32_t mask)
{
   uint32_t a = (destination->value >> shift) & mask;
   uint32_t b = (source->value >> shift) & mask;
   uint32_t field = a > b ? a : b;
   destination->value = (destination->value & ~(mask << shift)) | (field << shift);
}

static uint32_t
shared_vgprs(uint32_t front_rsrc1, uint32_t front_rsrc2, uint32_t back_rsrc1, uint32_t back_rsrc2)
{
   uint32_t front_vgprs = ((front_rsrc1 & RSRC1_VGPRS) + 1) * 4;
   uint32_t back_vgprs = ((back_rsrc1 & RSRC1_VGPRS) + 1) * 4;
   uint32_t front_total = front_vgprs + (front_rsrc2 >> 28) * 8;
   uint32_t back_total = back_vgprs + (back_rsrc2 >> 28) * 8;
   uint32_t total = front_total > back_total ? front_total : back_total;
   uint32_t largest = front_vgprs > back_vgprs ? front_vgprs : back_vgprs;
   uint32_t smallest = front_total < back_total ? front_total : back_total;
   if (largest >= total)
      return 0;
   return (total - smallest + 7) / 64;
}

static bool
validate_half(const struct agc_shader *shader, char *error, size_t bytes)
{
   if (shader->file_header != AGC_SHADER_MAGIC || shader->version != AGC_SHADER_VERSION)
      return fail(error, bytes, "invalid shader header or version");
   if (!shader->code || shader->shader_size == 0 || shader->shader_size % 4 != 0)
      return fail(error, bytes, "invalid shader half code");
   if ((shader->num_sh_registers && !shader->sh_registers) || (shader->num_cx_registers && !shader->cx_registers))
      return fail(error, bytes, "shader half registers missing");
   return true;
}

static bool
front_program(const struct agc_shader *front, uint32_t *program_bytes, uint32_t *code_bytes, uint32_t *trailer,
              char *error, size_t bytes)
{
   uint32_t size = front->shader_size;
   const uint8_t *code = (const uint8_t *)(uintptr_t)front->code;
   if (size < TRAILER_CODE_BYTES + 4)
      return fail(error, bytes, "front half has no program trailer");
   bool found = false;
   for (uint32_t offset = size - TRAILER_CODE_BYTES - 4 + 1; offset-- > 0;) {
      if (!memcmp(code + offset, trailer_magic, sizeof(trailer_magic))) {
         *trailer = offset;
         found = true;
         break;
      }
   }
   if (!found)
      return fail(error, bytes, "front half has no program trailer");
   *code_bytes = read_u32(code + *trailer + TRAILER_CODE_BYTES);
   if (*code_bytes == 0 || *code_bytes % 4 != 0 || *code_bytes > *trailer)
      return fail(error, bytes, "front half trailer has an invalid code size");
   const uint32_t *words = (const uint32_t *)code;
   uint32_t end = *code_bytes / 4;
   while (end > 0 && words[end - 1] == S_CODE_END)
      end--;
   if (end == 0 || (words[end - 1] != S_SETPC_S6 && words[end - 1] != S_SWAPPC_NULL_S6))
      return fail(error, bytes, "front half does not end with s_setpc_b64 s[6:7] or s_swappc_b64 null, s[6:7]");
   *program_bytes = (end - 1) * 4;
   return true;
}

static bool
constant_literals(const struct agc_shader *front, struct layout *layout, char *error, size_t bytes)
{
   uint32_t count = layout->front_bytes / 4;
   const uint32_t *code = (const uint32_t *)(uintptr_t)front->code;
   layout->literals = calloc(MAX_LITERALS, sizeof(*layout->literals));
   if (!layout->literals)
      return fail(error, bytes, "out of memory");
   uint32_t i = 0;
   while (i < count) {
      struct agc_isa_inst inst;
      struct agc_isa_error decode_error;
      if (!agc_isa_decode_at(code, count, i, &inst, &decode_error)) {
         char text[160];
         agc_isa_error_format(&decode_error, text, sizeof(text));
         snprintf(error, bytes,
                  "front half with an embedded constant buffer holds an instruction that cannot be decoded at word %u: %s",
                  i, text);
         return false;
      }
      uint32_t word = code[i];
      if ((word & S_GETPC_MASK) == S_GETPC) {
         uint32_t reg = (word >> 16) & 0x7f;
         uint32_t add = 0x80000000u | reg << 16 | reg << 8 | 0xff;
         uint32_t carry = 0x82000000u | (reg + 1) << 16 | (reg + 1) << 8 | 0x80;
         if (i + 3 >= count || code[i + 1] != add || code[i + 3] != carry)
            return fail(error, bytes,
                        "front half reads its embedded constant buffer through an s_getpc_b64 not followed by "
                        "s_add_u32 with a literal and s_addc_u32 0");
         uint64_t target = (uint64_t)(i + 1) * 4 + code[i + 2];
         if (target >= layout->front_bytes) {
            if (target < layout->constants_source ||
                target >= (uint64_t)layout->constants_source + layout->constants_bytes)
               return fail(error, bytes,
                           "front half's s_getpc_b64 offset points outside its program and embedded constant buffer");
            if (layout->literal_count == MAX_LITERALS)
               return fail(error, bytes, "front half has too many constant buffer references");
            layout->literals[layout->literal_count++] = (struct literal){
               (i + 2) * 4, code[i + 2] + (layout->constants_offset - layout->constants_source)};
         }
      }
      i += inst.word_count ? inst.word_count : 1;
   }
   return true;
}

static bool
half_register(const struct agc_shader *shader, uint32_t offset, uint32_t *value)
{
   uint32_t at;
   if (!find(shader->sh_registers, shader->num_sh_registers, offset, 0, &at))
      return false;
   *value = shader->sh_registers[at].value;
   return true;
}

static bool
geometry_layout(const struct agc_shader *front, const struct agc_shader *back, struct layout *layout, char *error,
                size_t bytes)
{
   memset(layout, 0, sizeof(*layout));
   if (!validate_half(back, error, bytes) || !validate_half(front, error, bytes))
      return false;
   if (back->embedded_constant_buffer_size_dqw)
      return fail(error, bytes, "fusing a back half with an embedded constant buffer is not implemented");
   uint32_t program_code_bytes, trailer;
   if (!front_program(front, &layout->front_bytes, &program_code_bytes, &trailer, error, bytes))
      return false;
   layout->back_offset = (uint32_t)align_up(layout->front_bytes, CODE_ALIGNMENT);
   layout->code_bytes = layout->back_offset + back->shader_size;
   layout->constants_bytes = front->embedded_constant_buffer_size_dqw * 16;
   if (layout->constants_bytes) {
      layout->constants_source = program_code_bytes;
      if ((uint64_t)layout->constants_source + layout->constants_bytes > trailer)
         return fail(error, bytes, "front half's embedded constant buffer runs into its program trailer");
      layout->constants_offset = (uint32_t)align_up(layout->code_bytes, 16);
      if (!constant_literals(front, layout, error, bytes))
         return false;
      layout->code_bytes = layout->constants_offset + layout->constants_bytes;
   }
   uint32_t front_rsrc1, front_rsrc2, back_rsrc1, back_rsrc2;
   if (!half_register(front, RSRC1_GS, &front_rsrc1) || !half_register(front, RSRC2_GS, &front_rsrc2))
      return fail(error, bytes, "front half has no GS resource registers");
   if (!half_register(back, RSRC1_GS, &back_rsrc1) || !half_register(back, RSRC2_GS, &back_rsrc2))
      return fail(error, bytes, "back half has no GS resource registers");
   uint32_t shared = shared_vgprs(front_rsrc1, front_rsrc2, back_rsrc1, back_rsrc2);
   uint32_t vgprs = (front_rsrc1 & RSRC1_VGPRS) > (back_rsrc1 & RSRC1_VGPRS) ? (front_rsrc1 & RSRC1_VGPRS)
                                                                             : (back_rsrc1 & RSRC1_VGPRS);
   uint32_t rsrc1_field = ((front_rsrc1 >> 29) & 3) > ((back_rsrc1 >> 29) & 3) ? ((front_rsrc1 >> 29) & 3)
                                                                                : ((back_rsrc1 >> 29) & 3);
   uint32_t rsrc2_field = ((front_rsrc2 >> 16) & 3) > ((back_rsrc2 >> 16) & 3) ? ((front_rsrc2 >> 16) & 3)
                                                                                : ((back_rsrc2 >> 16) & 3);
   uint32_t rsrc1 = (back_rsrc1 & ~(RSRC1_VGPRS | (UINT32_C(3) << 29))) | vgprs | (rsrc1_field << 29);
   uint32_t rsrc2 = (back_rsrc2 & ~(UINT32_C(0xf0000000) | (UINT32_C(3) << 16) | RSRC2_USER_SGPRS | RSRC2_OFFCHIP_LDS)) |
                    ((shared & 15) << 28) | (rsrc2_field << 16) | (front_rsrc2 & (RSRC2_USER_SGPRS | RSRC2_OFFCHIP_LDS));
   layout->sh_count = back->num_sh_registers;
   layout->cx_count = back->num_cx_registers;
   memcpy(layout->sh, back->sh_registers, layout->sh_count * sizeof(struct agc_register));
   if (layout->cx_count)
      memcpy(layout->cx, back->cx_registers, layout->cx_count * sizeof(struct agc_register));
   for (uint32_t i = 0; i < layout->sh_count; i++) {
      if (layout->sh[i].offset == RSRC1_GS)
         layout->sh[i].value = rsrc1;
      else if (layout->sh[i].offset == RSRC2_GS)
         layout->sh[i].value = rsrc2;
   }
   layout->sh_offset = align_up(sizeof(struct agc_shader), 8);
   layout->cx_offset = layout->sh_offset + layout->sh_count * sizeof(struct agc_register);
   layout->specials_offset = align_up(layout->cx_offset + layout->cx_count * sizeof(struct agc_register), 8);
   layout->inputs_offset = align_up(layout->specials_offset + back->special_sizes_bytes, 4);
   layout->outputs_offset = layout->inputs_offset + (size_t)front->num_input_semantics * 4;
   layout->user_data_offset = align_up(layout->outputs_offset + (size_t)back->num_output_semantics * 4, 8);
   const struct agc_shader_user_data *user_data = back->user_data;
   size_t end = layout->user_data_offset + (user_data ? sizeof(struct agc_shader_user_data) : 0);
   layout->direct_offset = align_up(end, 2);
   if (user_data)
      end = layout->direct_offset + (size_t)user_data->direct_resource_count * 2;
   for (unsigned i = 0; i < 4; i++) {
      layout->sharp_offsets[i] = align_up(end, 2);
      if (user_data)
         end = layout->sharp_offsets[i] + (size_t)user_data->sharp_resource_count[i] * 2;
   }
   layout->total_bytes = end;
   return true;
}

struct key_builder {
   uint8_t *bytes;
   size_t size;
   size_t capacity;
   bool failed;
};

static void
key_append(struct key_builder *key, const void *data, size_t bytes)
{
   if (!bytes || !data || key->failed)
      return;
   if (key->size + bytes > key->capacity) {
      size_t capacity = key->capacity ? key->capacity : 4096;
      while (capacity < key->size + bytes)
         capacity *= 2;
      uint8_t *grown = realloc(key->bytes, capacity);
      if (!grown) {
         key->failed = true;
         return;
      }
      key->bytes = grown;
      key->capacity = capacity;
   }
   memcpy(key->bytes + key->size, data, bytes);
   key->size += bytes;
}

static bool
fusion_key(const struct agc_shader *front, const struct agc_shader *back, const struct layout *layout,
           struct key_builder *key)
{
   memset(key, 0, sizeof(*key));
   key_append(key, front, HEADER_KEY_BYTES);
   key_append(key, back, HEADER_KEY_BYTES);
   key_append(key, (const void *)(uintptr_t)front->code, layout->front_bytes);
   key_append(key, (const void *)(uintptr_t)back->code, back->shader_size);
   if (layout->constants_bytes)
      key_append(key, (const uint8_t *)(uintptr_t)front->code + layout->constants_source, layout->constants_bytes);
   key_append(key, layout->sh, layout->sh_count * sizeof(struct agc_register));
   key_append(key, layout->cx, layout->cx_count * sizeof(struct agc_register));
   key_append(key, back->specials, back->special_sizes_bytes);
   key_append(key, front->input_semantics, (size_t)front->num_input_semantics * 4);
   key_append(key, back->output_semantics, (size_t)back->num_output_semantics * 4);
   if (back->user_data) {
      const struct agc_shader_user_data *user_data = back->user_data;
      key_append(key, user_data, USER_DATA_KEY_BYTES);
      key_append(key, user_data->direct_resource_offset, (size_t)user_data->direct_resource_count * 2);
      for (unsigned i = 0; i < 4; i++)
         key_append(key, user_data->sharp_resource_offset[i], (size_t)user_data->sharp_resource_count[i] * 2);
   }
   return !key->failed;
}

static void
copy_bytes(uint8_t *storage, size_t offset, const void *from, size_t bytes)
{
   if (bytes && from)
      memcpy(storage + offset, from, bytes);
}

static struct fused *
build_geometry(const struct agc_shader *front, const struct agc_shader *back, struct layout *layout, char *error,
               size_t bytes)
{
   struct fused *fused = calloc(1, sizeof(*fused));
   if (!fused) {
      fail(error, bytes, "out of memory");
      return NULL;
   }
   fused->storage = calloc(1, layout->total_bytes);
   if (!fused->storage || !agc_mem_suballoc(layout->code_bytes, CODE_ALIGNMENT, 0, &fused->code)) {
      free(fused->storage);
      free(fused);
      fail(error, bytes, "out of memory for the fused program");
      return NULL;
   }
   uint8_t *code = fused->code.cpu;
   const uint8_t *front_code = (const uint8_t *)(uintptr_t)front->code;
   const uint8_t *back_code = (const uint8_t *)(uintptr_t)back->code;
   memcpy(code, front_code, layout->front_bytes);
   for (uint32_t offset = layout->front_bytes; offset < layout->back_offset; offset += 4)
      memcpy(code + offset, &(uint32_t){S_NOP}, 4);
   memcpy(code + layout->back_offset, back_code, back->shader_size);
   if (layout->constants_bytes) {
      memcpy(code + layout->constants_offset, front_code + layout->constants_source, layout->constants_bytes);
      for (uint32_t i = 0; i < layout->literal_count; i++)
         memcpy(code + layout->literals[i].offset, &layout->literals[i].value, 4);
   }
   _mm_sfence();

   uint64_t base = fused->code.va;
   uint32_t pair;
   bool found = false;
   for (pair = 0; pair + 1 < layout->sh_count; pair++) {
      if (layout->sh[pair].offset == PGM_LO_GS && layout->sh[pair + 1].offset == PGM_LO_GS + 1) {
         found = true;
         break;
      }
   }
   if (!found) {
      fail(error, bytes, "back half has no program address register pair");
      return NULL;
   }
   uint64_t back_address = base + layout->back_offset;
   layout->sh[pair].value = (uint32_t)(back_address >> 8);
   layout->sh[pair + 1].value = (layout->sh[pair + 1].value & 0xffffff00u) | (uint32_t)((back_address >> 40) & 0xff);

   uint8_t *storage = fused->storage;
   struct agc_shader *header = (struct agc_shader *)storage;
   *header = *back;
   copy_bytes(storage, layout->sh_offset, layout->sh, layout->sh_count * sizeof(struct agc_register));
   copy_bytes(storage, layout->cx_offset, layout->cx, layout->cx_count * sizeof(struct agc_register));
   copy_bytes(storage, layout->specials_offset, back->specials, back->special_sizes_bytes);
   copy_bytes(storage, layout->inputs_offset, front->input_semantics, (size_t)front->num_input_semantics * 4);
   copy_bytes(storage, layout->outputs_offset, back->output_semantics, (size_t)back->num_output_semantics * 4);
   if (back->user_data) {
      const struct agc_shader_user_data *user_data = back->user_data;
      struct agc_shader_user_data *copied = (struct agc_shader_user_data *)(storage + layout->user_data_offset);
      *copied = *user_data;
      copy_bytes(storage, layout->direct_offset, user_data->direct_resource_offset,
                 (size_t)user_data->direct_resource_count * 2);
      copied->direct_resource_offset =
         user_data->direct_resource_count ? (uint16_t *)(storage + layout->direct_offset) : NULL;
      for (unsigned i = 0; i < 4; i++) {
         copy_bytes(storage, layout->sharp_offsets[i], user_data->sharp_resource_offset[i],
                    (size_t)user_data->sharp_resource_count[i] * 2);
         copied->sharp_resource_offset[i] =
            user_data->sharp_resource_count[i] ? (uint16_t *)(storage + layout->sharp_offsets[i]) : NULL;
      }
      header->user_data = copied;
   }
   header->header_size = (uint32_t)layout->total_bytes;
   header->code = base;
   header->sh_registers = (struct agc_register *)(storage + layout->sh_offset);
   header->cx_registers = layout->cx_count ? (struct agc_register *)(storage + layout->cx_offset) : NULL;
   header->specials =
      back->special_sizes_bytes ? (struct agc_shader_specials *)(storage + layout->specials_offset) : NULL;
   header->input_semantics = front->num_input_semantics ? (const uint32_t *)(storage + layout->inputs_offset) : NULL;
   header->output_semantics =
      back->num_output_semantics ? (const uint32_t *)(storage + layout->outputs_offset) : NULL;
   header->shader_size = layout->code_bytes;
   header->num_input_semantics = front->num_input_semantics;
   header->scratch_size_dw_per_thread = front->scratch_size_dw_per_thread > back->scratch_size_dw_per_thread
                                           ? front->scratch_size_dw_per_thread
                                           : back->scratch_size_dw_per_thread;
   header->kind = AGC_SHADER_GS;
   header->num_sh_registers = (uint8_t)layout->sh_count;
   header->num_cx_registers = (uint8_t)layout->cx_count;

   uint64_t host = 0;
   int32_t result = agc_program_translate_linked(header, base, (const uint32_t *)fused->code.cpu,
                                                 layout->code_bytes / 4, NULL, &host);
   if (result) {
      fail(error, bytes, "the fused geometry program could not be translated");
      return NULL;
   }
   if (agc_shader_rebase_program(header->sh_registers, header->num_sh_registers, PGM_LO_ES, host)) {
      fail(error, bytes, "fused program address patch failed");
      return NULL;
   }
   fused->header = header;
   return fused;
}

static int32_t
fuse_geometry(struct agc_shader *result, const struct agc_shader *front, const struct agc_shader *back,
              struct agc_register *scratch, char *error, size_t bytes)
{
   if (!scratch) {
      fail(error, bytes, "geometry halves need fused shader memory");
      return AGC_ERROR_INVALID_SHADER_PROGRAM;
   }
   char start_error[256];
   if (!agc_driver_start(start_error, sizeof(start_error))) {
      snprintf(error, bytes, "the GPU is unavailable: %s", start_error);
      return AGC_ERROR_INVALID_SHADER_PROGRAM;
   }
   struct layout *layout = calloc(1, sizeof(*layout));
   if (!layout) {
      fail(error, bytes, "out of memory");
      return AGC_ERROR_INVALID_SHADER_PROGRAM;
   }
   int32_t code = AGC_ERROR_INVALID_SHADER_PROGRAM;
   struct key_builder key;
   memset(&key, 0, sizeof(key));
   if (!geometry_layout(front, back, layout, error, bytes))
      goto out;
   if (!fusion_key(front, back, layout, &key)) {
      fail(error, bytes, "out of memory");
      goto out;
   }
   call_once(&fused_once, fused_init);
   mtx_lock(&fused_lock);
   struct fused *fused = NULL;
   for (struct fused *each = fused_programs; each; each = each->next) {
      if (each->key_bytes == key.size && !memcmp(each->key, key.bytes, key.size)) {
         fused = each;
         break;
      }
   }
   if (!fused) {
      fused = build_geometry(front, back, layout, error, bytes);
      if (fused) {
         fused->key = key.bytes;
         fused->key_bytes = key.size;
         key.bytes = NULL;
         fused->next = fused_programs;
         fused_programs = fused;
      }
   }
   mtx_unlock(&fused_lock);
   if (fused) {
      struct agc_shader copy = *fused->header;
      memcpy(scratch, copy.sh_registers, (size_t)copy.num_sh_registers * sizeof(struct agc_register));
      copy.sh_registers = scratch;
      *result = copy;
      code = 0;
   }
out:
   free(key.bytes);
   free(layout->literals);
   free(layout);
   return code;
}

static int32_t
fuse_hull(struct agc_shader *result, const struct agc_shader *front, const struct agc_shader *back,
          struct agc_register *scratch, char *error, size_t bytes)
{
   *result = *back;
   result->kind = AGC_SHADER_HS;
   if (scratch) {
      memcpy(scratch, back->sh_registers, (size_t)back->num_sh_registers * sizeof(struct agc_register));
      result->sh_registers = scratch;
   }
   struct agc_register *fused = result->sh_registers;
   uint32_t count = result->num_sh_registers;
   const struct agc_register *front_registers = front->sh_registers;
   uint32_t front_count = front->num_sh_registers;
   for (uint32_t occurrence = 0; occurrence < 2; occurrence++) {
      uint32_t to, from;
      if (!find(fused, count, CHKSUM_HS, occurrence, &to) ||
          !find(front_registers, front_count, CHKSUM_HS, occurrence, &from)) {
         snprintf(error, bytes, "shader half lacks register %u", CHKSUM_HS);
         return AGC_ERROR_INVALID_SHADER_PROGRAM;
      }
      fused[to].value = front_registers[from].value;
   }
   uint32_t front_rsrc1, front_rsrc2, rsrc1, rsrc2, lo, hi;
   if (!find(front_registers, front_count, RSRC1_HS, 0, &front_rsrc1) ||
       !find(front_registers, front_count, RSRC2_HS, 0, &front_rsrc2) || !find(fused, count, RSRC1_HS, 0, &rsrc1) ||
       !find(fused, count, RSRC2_HS, 0, &rsrc2) || !find(fused, count, PGM_LO_LS, 0, &lo) ||
       !find(fused, count, PGM_LO_LS + 1, 0, &hi)) {
      fail(error, bytes, "shader half lacks a hull register");
      return AGC_ERROR_INVALID_SHADER_PROGRAM;
   }
   uint32_t shared = shared_vgprs(front_registers[front_rsrc1].value, front_registers[front_rsrc2].value,
                                  fused[rsrc1].value, fused[rsrc2].value);
   fused[rsrc2].value = (fused[rsrc2].value & 0x0fffffffu) | ((shared & 0xf) << 28);
   merge_max(&fused[rsrc1], &front_registers[front_rsrc1], 0, 0x3f);
   merge_max(&fused[rsrc1], &front_registers[front_rsrc1], 28, 0x3);
   fused[rsrc2].value = (fused[rsrc2].value & 0xf7ffffc1u) | (front_registers[front_rsrc2].value & 0x0800003eu);
   result->user_data = NULL;

   struct agc_program_link link = {
      .second_half = back->code,
      .second_words = (const uint32_t *)(uintptr_t)back->code,
      .second_word_count = back->shader_size / 4,
   };
   uint64_t host = 0;
   int32_t code = agc_program_translate_linked(result, front->code, (const uint32_t *)(uintptr_t)front->code,
                                               front->shader_size / 4, &link, &host);
   if (code) {
      fail(error, bytes, "the fused hull program could not be translated");
      return code;
   }
   fused[lo].value = (uint32_t)(host >> 8);
   fused[hi].value = (fused[hi].value & 0xffffff00u) | (uint32_t)((host >> 40) & 0xff);
   return 0;
}

static bool
valid_halves(const struct agc_shader *front, const struct agc_shader *back)
{
   return (front->kind == AGC_SHADER_GS_FRONT && back->kind == AGC_SHADER_GS_BACK) ||
          (front->kind == AGC_SHADER_HS_FRONT && back->kind == AGC_SHADER_HS_BACK);
}

int32_t
agc_shader_fuse(struct agc_shader *result, const struct agc_shader *front, const struct agc_shader *back,
                struct agc_register *scratch)
{
   static const char function[] = "sceAgcFuseShaderHalves";
   if (!result || !front || !back)
      return AGC_ERROR_INVALID_ARGUMENT;
   if (!valid_halves(front, back) || !front->specials || !back->specials)
      return AGC_ERROR_INVALID_SHADER_HALVES;
   bool geometry = front->kind == AGC_SHADER_GS_FRONT;
   uint32_t wave_bit = geometry ? UINT32_C(1) << 22 : UINT32_C(1) << 21;
   if ((front->specials->vgt_shader_stages_en.value ^ back->specials->vgt_shader_stages_en.value) & wave_bit)
      return AGC_ERROR_INVALID_SHADER_HALVES;
   char error[512] = "";
   int32_t code = geometry ? fuse_geometry(result, front, back, scratch, error, sizeof(error))
                           : fuse_hull(result, front, back, scratch, error, sizeof(error));
   if (code) {
      agc_report(function, "%s", error);
      return AGC_ERROR_INVALID_SHADER_PROGRAM;
   }
   return 0;
}

int32_t
agc_shader_fused_size(struct agc_size_align *destination, const struct agc_shader *front,
                      const struct agc_shader *back)
{
   if (!destination || !front || !back)
      return AGC_ERROR_INVALID_ARGUMENT;
   if (!valid_halves(front, back))
      return AGC_ERROR_INVALID_SHADER_HALVES;
   if (front->kind == AGC_SHADER_GS_FRONT) {
      struct layout *layout = calloc(1, sizeof(*layout));
      char error[512] = "out of memory";
      bool ok = layout && geometry_layout(front, back, layout, error, sizeof(error));
      if (layout)
         free(layout->literals);
      free(layout);
      if (!ok) {
         agc_report("sceAgcGetFusedShaderSize", "%s", error);
         return AGC_ERROR_INVALID_SHADER_PROGRAM;
      }
   }
   destination->size = (uint64_t)back->num_sh_registers * sizeof(struct agc_register);
   destination->align = 4;
   return 0;
}
