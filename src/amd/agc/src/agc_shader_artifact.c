#include "agc_shader_artifact.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "util/crc32.h"
#include "util/detect_os.h"
#include "util/mesa-blake3.h"
#include "util/os_file.h"
#include "util/u_atomic.h"

#if DETECT_OS_WINDOWS
#include <windows.h>
#include <io.h>
#include <process.h>
#else
#include <unistd.h>
#endif

#define AGC_SHADER_REGION_HEADER_BYTES 16
#define AGC_SHADER_BINDING_BYTES 4
#define AGC_SHADER_EXTERNAL_TARGET_BYTES 8
#define AGC_SHADER_USE_BYTES 28
#define AGC_SHADER_TABLE_BYTES 16
#define AGC_SHADER_LINK_BYTES 12

#define AGC_SHADER_REQUEST_HAS_USER_SGPR_COUNT (1u << 0)
#define AGC_SHADER_REQUEST_HAS_GUEST_PGM_RSRC1 (1u << 1)
#define AGC_SHADER_REQUEST_HAS_GUEST_PGM_RSRC2 (1u << 2)
#define AGC_SHADER_REQUEST_HAS_GUEST_CODE_ADDRESS (1u << 3)
#define AGC_SHADER_REQUEST_HAS_PS_INPUT_ENA (1u << 4)
#define AGC_SHADER_REQUEST_HAS_PS_INPUT_ADDR (1u << 5)
#define AGC_SHADER_REQUEST_HAS_PS_COL_FORMAT (1u << 6)
#define AGC_SHADER_REQUEST_HAS_PS_Z_FORMAT (1u << 7)
#define AGC_SHADER_REQUEST_HAS_PARAM_EXPORTS (1u << 8)
#define AGC_SHADER_REQUEST_HAS_DESCRIPTOR_LUT (1u << 9)
#define AGC_SHADER_REQUEST_HAS_ATTRIBUTE_RING (1u << 10)
#define AGC_SHADER_REQUEST_HAS_GDS_BUFFER (1u << 11)
#define AGC_SHADER_REQUEST_HAS_MIP_STATS_TABLE (1u << 12)
#define AGC_SHADER_REQUEST_HAS_IMAGE_REPORTS (1u << 13)
#define AGC_SHADER_REQUEST_HAS_SECOND_HALF (1u << 14)
#define AGC_SHADER_REQUEST_PRESENT_MASK 0x7fffu

#define AGC_SHADER_ARTIFACT_HAS_USER_DATA_TAIL_SGPR (1u << 0)
#define AGC_SHADER_ARTIFACT_HAS_SPI_SHADER_COL_FORMAT (1u << 1)
#define AGC_SHADER_ARTIFACT_HAS_NGG (1u << 2)
#define AGC_SHADER_ARTIFACT_HAS_RING_RELOCATIONS (1u << 3)
#define AGC_SHADER_ARTIFACT_HAS_BUFFER_FILL (1u << 4)
#define AGC_SHADER_ARTIFACT_PRESENT_MASK 0x1fu

#define AGC_SHADER_ERROR_FIELD_PC (1u << 0)
#define AGC_SHADER_ERROR_FIELD_WORD (1u << 1)
#define AGC_SHADER_ERROR_FIELD_BRANCH_TARGET (1u << 2)
#define AGC_SHADER_ERROR_FIELD_OPCODE (1u << 3)
#define AGC_SHADER_ERROR_FIELD_VGPRS (1u << 4)
#define AGC_SHADER_ERROR_FIELD_TARGET (1u << 5)
#define AGC_SHADER_ERROR_FIELD_MISMATCH (1u << 6)
#define AGC_SHADER_ERROR_FIELD_REASON (1u << 7)

static const uint8_t agc_shader_error_fields[AGC_SHADER_ERROR_KIND_COUNT] = {
   [AGC_SHADER_ERROR_NONE] = 0,
   [AGC_SHADER_ERROR_UNSUPPORTED] = 0,
   [AGC_SHADER_ERROR_INVALID_OPTIONS] = AGC_SHADER_ERROR_FIELD_REASON,
   [AGC_SHADER_ERROR_TARGET_MISMATCH] = AGC_SHADER_ERROR_FIELD_MISMATCH,
   [AGC_SHADER_ERROR_EMPTY_PROGRAM] = 0,
   [AGC_SHADER_ERROR_TRUNCATED] = AGC_SHADER_ERROR_FIELD_PC,
   [AGC_SHADER_ERROR_INVALID_ENCODING] = AGC_SHADER_ERROR_FIELD_PC | AGC_SHADER_ERROR_FIELD_WORD,
   [AGC_SHADER_ERROR_UNKNOWN_OPCODE] = AGC_SHADER_ERROR_FIELD_PC | AGC_SHADER_ERROR_FIELD_OPCODE,
   [AGC_SHADER_ERROR_INVALID_OPERAND] = AGC_SHADER_ERROR_FIELD_PC | AGC_SHADER_ERROR_FIELD_REASON,
   [AGC_SHADER_ERROR_INVALID_BRANCH] = AGC_SHADER_ERROR_FIELD_PC | AGC_SHADER_ERROR_FIELD_BRANCH_TARGET,
   [AGC_SHADER_ERROR_MISSING_END_PROGRAM] = 0,
   [AGC_SHADER_ERROR_UNSUPPORTED_INSTRUCTION] = AGC_SHADER_ERROR_FIELD_PC | AGC_SHADER_ERROR_FIELD_OPCODE |
                                                AGC_SHADER_ERROR_FIELD_TARGET | AGC_SHADER_ERROR_FIELD_REASON,
   [AGC_SHADER_ERROR_BRANCH_OUT_OF_RANGE] = AGC_SHADER_ERROR_FIELD_PC,
   [AGC_SHADER_ERROR_OUT_OF_REGISTERS] = AGC_SHADER_ERROR_FIELD_VGPRS,
};

static const char *const agc_shader_file_status_names[AGC_SHADER_FILE_STATUS_COUNT] = {
   [AGC_SHADER_FILE_OK] = "ok",
   [AGC_SHADER_FILE_NOT_FOUND] = "not found",
   [AGC_SHADER_FILE_IO_ERROR] = "i/o error",
   [AGC_SHADER_FILE_OUT_OF_MEMORY] = "out of memory",
   [AGC_SHADER_FILE_BAD_MAGIC] = "bad magic",
   [AGC_SHADER_FILE_BAD_VERSION] = "bad version",
   [AGC_SHADER_FILE_TRUNCATED] = "truncated",
   [AGC_SHADER_FILE_BAD_CHECKSUM] = "bad checksum",
   [AGC_SHADER_FILE_MALFORMED] = "malformed",
   [AGC_SHADER_FILE_UNREPRESENTABLE] = "unrepresentable",
   [AGC_SHADER_FILE_KEY_MISMATCH] = "key mismatch",
};

static const char *const agc_shader_error_kind_names[AGC_SHADER_ERROR_KIND_COUNT] = {
   [AGC_SHADER_ERROR_NONE] = "None",
   [AGC_SHADER_ERROR_UNSUPPORTED] = "Unsupported",
   [AGC_SHADER_ERROR_INVALID_OPTIONS] = "InvalidOptions",
   [AGC_SHADER_ERROR_TARGET_MISMATCH] = "TargetMismatch",
   [AGC_SHADER_ERROR_EMPTY_PROGRAM] = "EmptyProgram",
   [AGC_SHADER_ERROR_TRUNCATED] = "Truncated",
   [AGC_SHADER_ERROR_INVALID_ENCODING] = "InvalidEncoding",
   [AGC_SHADER_ERROR_UNKNOWN_OPCODE] = "UnknownOpcode",
   [AGC_SHADER_ERROR_INVALID_OPERAND] = "InvalidOperand",
   [AGC_SHADER_ERROR_INVALID_BRANCH] = "InvalidBranch",
   [AGC_SHADER_ERROR_MISSING_END_PROGRAM] = "MissingEndProgram",
   [AGC_SHADER_ERROR_UNSUPPORTED_INSTRUCTION] = "UnsupportedInstruction",
   [AGC_SHADER_ERROR_BRANCH_OUT_OF_RANGE] = "BranchOutOfRange",
   [AGC_SHADER_ERROR_OUT_OF_REGISTERS] = "OutOfRegisters",
};

static int agc_shader_temp_counter;

const char *
agc_shader_file_status_name(enum agc_shader_file_status status)
{
   if ((unsigned)status >= AGC_SHADER_FILE_STATUS_COUNT)
      return "unknown";
   return agc_shader_file_status_names[status];
}

const char *
agc_shader_error_kind_name(enum agc_shader_error_kind kind)
{
   if ((unsigned)kind >= AGC_SHADER_ERROR_KIND_COUNT)
      return "Unknown";
   return agc_shader_error_kind_names[kind];
}

static uint32_t
agc_shader_crc32(const uint8_t *data, size_t size)
{
   return ~util_hash_crc32(data, size);
}

static void
agc_shader_store_u32(uint8_t *p, uint32_t value)
{
   p[0] = (uint8_t)value;
   p[1] = (uint8_t)(value >> 8);
   p[2] = (uint8_t)(value >> 16);
   p[3] = (uint8_t)(value >> 24);
}

static uint32_t
agc_shader_load_u32(const uint8_t *p)
{
   return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

struct agc_shader_writer {
   uint8_t *data;
   size_t size;
   size_t capacity;
   enum agc_shader_file_status status;
};

static void
agc_shader_writer_fail(struct agc_shader_writer *w, enum agc_shader_file_status status)
{
   if (w->status == AGC_SHADER_FILE_OK)
      w->status = status;
}

static uint8_t *
agc_shader_writer_grow(struct agc_shader_writer *w, uint64_t bytes)
{
   if (w->status != AGC_SHADER_FILE_OK)
      return NULL;
   if (bytes > UINT32_MAX - (uint64_t)w->size) {
      agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);
      return NULL;
   }
   uint64_t needed = (uint64_t)w->size + bytes;
   if (needed > w->capacity) {
      uint64_t capacity = w->capacity ? w->capacity : 256;
      while (capacity < needed)
         capacity *= 2;
      uint8_t *data = capacity <= SIZE_MAX ? realloc(w->data, (size_t)capacity) : NULL;
      if (!data) {
         agc_shader_writer_fail(w, AGC_SHADER_FILE_OUT_OF_MEMORY);
         return NULL;
      }
      w->data = data;
      w->capacity = (size_t)capacity;
   }
   uint8_t *out = w->data + w->size;
   w->size = (size_t)needed;
   return out;
}

static void
agc_shader_put_u8(struct agc_shader_writer *w, uint8_t value)
{
   uint8_t *p = agc_shader_writer_grow(w, 1);
   if (p)
      p[0] = value;
}

static void
agc_shader_put_u16(struct agc_shader_writer *w, uint16_t value)
{
   uint8_t *p = agc_shader_writer_grow(w, 2);
   if (!p)
      return;
   p[0] = (uint8_t)value;
   p[1] = (uint8_t)(value >> 8);
}

static void
agc_shader_put_u32(struct agc_shader_writer *w, uint32_t value)
{
   uint8_t *p = agc_shader_writer_grow(w, 4);
   if (p)
      agc_shader_store_u32(p, value);
}

static void
agc_shader_put_u64(struct agc_shader_writer *w, uint64_t value)
{
   uint8_t *p = agc_shader_writer_grow(w, 8);
   if (!p)
      return;
   agc_shader_store_u32(p, (uint32_t)value);
   agc_shader_store_u32(p + 4, (uint32_t)(value >> 32));
}

static void
agc_shader_put_zero(struct agc_shader_writer *w, size_t bytes)
{
   uint8_t *p = agc_shader_writer_grow(w, bytes);
   if (p)
      memset(p, 0, bytes);
}

static void
agc_shader_put_bool(struct agc_shader_writer *w, bool value)
{
   agc_shader_put_u8(w, value ? 1 : 0);
}

static void
agc_shader_put_enum(struct agc_shader_writer *w, unsigned value, unsigned count)
{
   if (value >= count)
      agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);
   agc_shader_put_u8(w, (uint8_t)value);
}

static void
agc_shader_put_words(struct agc_shader_writer *w, const uint32_t *words, uint32_t count)
{
   if (!count)
      return;
   if (!words) {
      agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);
      return;
   }
   uint8_t *p = agc_shader_writer_grow(w, (uint64_t)count * 4);
   if (!p)
      return;
   for (uint32_t i = 0; i < count; i++)
      agc_shader_store_u32(p + (size_t)i * 4, words[i]);
}

static void
agc_shader_put_header(struct agc_shader_writer *w, uint32_t magic, uint32_t version)
{
   agc_shader_put_u32(w, magic);
   agc_shader_put_u32(w, version);
   agc_shader_put_u32(w, 0);
   agc_shader_put_u32(w, 0);
}

static enum agc_shader_file_status
agc_shader_writer_finish(struct agc_shader_writer *w, uint8_t **data, size_t *size)
{
   if (w->status != AGC_SHADER_FILE_OK) {
      free(w->data);
      *data = NULL;
      *size = 0;
      return w->status;
   }
   agc_shader_store_u32(w->data + 8, (uint32_t)w->size);
   agc_shader_store_u32(w->data + 12, agc_shader_crc32(w->data + AGC_SHADER_FILE_HEADER_BYTES,
                                                       w->size - AGC_SHADER_FILE_HEADER_BYTES));
   *data = w->data;
   *size = w->size;
   return AGC_SHADER_FILE_OK;
}

struct agc_shader_reader {
   const uint8_t *data;
   size_t size;
   size_t at;
   enum agc_shader_file_status status;
};

static void
agc_shader_reject(struct agc_shader_reader *r)
{
   if (r->status == AGC_SHADER_FILE_OK)
      r->status = AGC_SHADER_FILE_MALFORMED;
}

static const uint8_t *
agc_shader_take(struct agc_shader_reader *r, uint64_t bytes)
{
   if (r->status != AGC_SHADER_FILE_OK)
      return NULL;
   if (bytes > r->size - r->at) {
      r->status = AGC_SHADER_FILE_TRUNCATED;
      return NULL;
   }
   const uint8_t *p = r->data + r->at;
   r->at += (size_t)bytes;
   return p;
}

static bool
agc_shader_fits(struct agc_shader_reader *r, uint32_t count, size_t element_bytes)
{
   if (r->status != AGC_SHADER_FILE_OK)
      return false;
   if (count > (r->size - r->at) / element_bytes) {
      r->status = AGC_SHADER_FILE_TRUNCATED;
      return false;
   }
   return true;
}

static uint8_t
agc_shader_get_u8(struct agc_shader_reader *r)
{
   const uint8_t *p = agc_shader_take(r, 1);
   return p ? p[0] : 0;
}

static uint16_t
agc_shader_get_u16(struct agc_shader_reader *r)
{
   const uint8_t *p = agc_shader_take(r, 2);
   return p ? (uint16_t)(p[0] | p[1] << 8) : 0;
}

static uint32_t
agc_shader_get_u32(struct agc_shader_reader *r)
{
   const uint8_t *p = agc_shader_take(r, 4);
   return p ? agc_shader_load_u32(p) : 0;
}

static uint64_t
agc_shader_get_u64(struct agc_shader_reader *r)
{
   const uint8_t *p = agc_shader_take(r, 8);
   return p ? (uint64_t)agc_shader_load_u32(p) | (uint64_t)agc_shader_load_u32(p + 4) << 32 : 0;
}

static bool
agc_shader_get_bool(struct agc_shader_reader *r)
{
   uint8_t value = agc_shader_get_u8(r);
   if (value > 1)
      agc_shader_reject(r);
   return value == 1;
}

static unsigned
agc_shader_get_enum(struct agc_shader_reader *r, unsigned count)
{
   uint8_t value = agc_shader_get_u8(r);
   if (value >= count) {
      agc_shader_reject(r);
      return 0;
   }
   return value;
}

static void
agc_shader_get_zero(struct agc_shader_reader *r, size_t bytes)
{
   const uint8_t *p = agc_shader_take(r, bytes);
   if (!p)
      return;
   for (size_t i = 0; i < bytes; i++) {
      if (p[i])
         agc_shader_reject(r);
   }
}

static uint8_t
agc_shader_get_optional_u8(struct agc_shader_reader *r, bool present)
{
   uint8_t value = agc_shader_get_u8(r);
   if (!present && value)
      agc_shader_reject(r);
   return value;
}

static uint32_t
agc_shader_get_optional_u32(struct agc_shader_reader *r, bool present)
{
   uint32_t value = agc_shader_get_u32(r);
   if (!present && value)
      agc_shader_reject(r);
   return value;
}

static uint64_t
agc_shader_get_optional_u64(struct agc_shader_reader *r, bool present)
{
   uint64_t value = agc_shader_get_u64(r);
   if (!present && value)
      agc_shader_reject(r);
   return value;
}

static void
agc_shader_get_words(struct agc_shader_reader *r, uint32_t *words, uint32_t count)
{
   if (!agc_shader_fits(r, count, 4))
      return;
   const uint8_t *p = agc_shader_take(r, (uint64_t)count * 4);
   if (!p || !words)
      return;
   for (uint32_t i = 0; i < count; i++)
      words[i] = agc_shader_load_u32(p + (size_t)i * 4);
}

static enum agc_shader_file_status
agc_shader_reader_open(struct agc_shader_reader *r, const uint8_t *data, size_t size, uint32_t magic,
                       uint32_t version)
{
   if (size >= 4 && agc_shader_load_u32(data) != magic)
      return AGC_SHADER_FILE_BAD_MAGIC;
   if (size < AGC_SHADER_FILE_HEADER_BYTES)
      return AGC_SHADER_FILE_TRUNCATED;
   if (agc_shader_load_u32(data + 4) != version)
      return AGC_SHADER_FILE_BAD_VERSION;
   uint32_t total = agc_shader_load_u32(data + 8);
   if (total < AGC_SHADER_FILE_HEADER_BYTES)
      return AGC_SHADER_FILE_MALFORMED;
   if (size < total)
      return AGC_SHADER_FILE_TRUNCATED;
   if (size > total)
      return AGC_SHADER_FILE_MALFORMED;
   if (agc_shader_load_u32(data + 12) !=
       agc_shader_crc32(data + AGC_SHADER_FILE_HEADER_BYTES, size - AGC_SHADER_FILE_HEADER_BYTES))
      return AGC_SHADER_FILE_BAD_CHECKSUM;
   r->data = data;
   r->size = size;
   r->at = AGC_SHADER_FILE_HEADER_BYTES;
   r->status = AGC_SHADER_FILE_OK;
   return AGC_SHADER_FILE_OK;
}

enum agc_shader_file_status
agc_shader_request_serialize(const struct agc_shader_request *request, uint8_t **data, size_t *size)
{
   const struct agc_shader_interface *iface = &request->iface;
   const struct agc_shader_linkage *linkage = &request->linkage;
   struct agc_shader_writer w = {0};
   uint32_t present = 0;

   if (iface->has_user_sgpr_count)
      present |= AGC_SHADER_REQUEST_HAS_USER_SGPR_COUNT;
   if (iface->has_guest_pgm_rsrc1)
      present |= AGC_SHADER_REQUEST_HAS_GUEST_PGM_RSRC1;
   if (iface->has_guest_pgm_rsrc2)
      present |= AGC_SHADER_REQUEST_HAS_GUEST_PGM_RSRC2;
   if (iface->has_guest_code_address)
      present |= AGC_SHADER_REQUEST_HAS_GUEST_CODE_ADDRESS;
   if (iface->has_ps_input_ena)
      present |= AGC_SHADER_REQUEST_HAS_PS_INPUT_ENA;
   if (iface->has_ps_input_addr)
      present |= AGC_SHADER_REQUEST_HAS_PS_INPUT_ADDR;
   if (iface->has_ps_col_format)
      present |= AGC_SHADER_REQUEST_HAS_PS_COL_FORMAT;
   if (iface->has_ps_z_format)
      present |= AGC_SHADER_REQUEST_HAS_PS_Z_FORMAT;
   if (iface->has_param_exports)
      present |= AGC_SHADER_REQUEST_HAS_PARAM_EXPORTS;
   if (iface->has_descriptor_lut)
      present |= AGC_SHADER_REQUEST_HAS_DESCRIPTOR_LUT;
   if (iface->has_attribute_ring)
      present |= AGC_SHADER_REQUEST_HAS_ATTRIBUTE_RING;
   if (iface->has_gds_buffer)
      present |= AGC_SHADER_REQUEST_HAS_GDS_BUFFER;
   if (iface->has_mip_stats_table)
      present |= AGC_SHADER_REQUEST_HAS_MIP_STATS_TABLE;
   if (iface->has_image_reports)
      present |= AGC_SHADER_REQUEST_HAS_IMAGE_REPORTS;
   if (linkage->has_second_half)
      present |= AGC_SHADER_REQUEST_HAS_SECOND_HALF;

   agc_shader_put_header(&w, AGC_SHADER_REQUEST_MAGIC, AGC_SHADER_REQUEST_VERSION);
   agc_shader_put_u32(&w, request->compiler_version);
   agc_shader_put_enum(&w, (unsigned)request->options.target, AGC_SHADER_ISA_COUNT);
   agc_shader_put_enum(&w, (unsigned)request->options.stage, AGC_SHADER_STAGE_COUNT);
   agc_shader_put_u8(&w, request->options.wave_size);
   agc_shader_put_enum(&w, (unsigned)request->options.ray_unit, AGC_SHADER_RAY_UNIT_COUNT);
   agc_shader_put_enum(&w, (unsigned)request->precision, AGC_SHADER_PRECISION_COUNT);
   agc_shader_put_enum(&w, (unsigned)iface->vertex_stage, AGC_SHADER_VERTEX_STAGE_COUNT);
   agc_shader_put_bool(&w, iface->dual_source_blend);
   agc_shader_put_bool(&w, iface->preconvert_user_descriptors);
   agc_shader_put_u32(&w, present);
   agc_shader_put_u8(&w, iface->has_user_sgpr_count ? iface->user_sgpr_count : 0);
   agc_shader_put_u8(&w, iface->has_param_exports ? iface->param_exports : 0);
   agc_shader_put_bool(&w, iface->native_dcc);
   agc_shader_put_enum(&w, (unsigned)iface->es_input, AGC_SHADER_ES_INPUT_COUNT);
   agc_shader_put_u32(&w, iface->has_guest_pgm_rsrc1 ? iface->guest_pgm_rsrc1 : 0);
   agc_shader_put_u32(&w, iface->has_guest_pgm_rsrc2 ? iface->guest_pgm_rsrc2 : 0);
   agc_shader_put_u32(&w, iface->lds_bytes);
   agc_shader_put_u32(&w, iface->scratch_bytes_per_lane);
   agc_shader_put_u32(&w, iface->has_ps_input_ena ? iface->ps_input_ena : 0);
   agc_shader_put_u32(&w, iface->has_ps_input_addr ? iface->ps_input_addr : 0);
   agc_shader_put_u32(&w, iface->has_ps_col_format ? iface->ps_col_format : 0);
   agc_shader_put_u32(&w, iface->has_ps_z_format ? iface->ps_z_format : 0);
   agc_shader_put_u32(&w, iface->has_attribute_ring ? iface->attribute_ring.bytes : 0);
   agc_shader_put_u64(&w, iface->has_guest_code_address ? iface->guest_code_address : 0);
   agc_shader_put_u64(&w, iface->has_descriptor_lut ? iface->descriptor_lut : 0);
   agc_shader_put_u64(&w, iface->has_attribute_ring ? iface->attribute_ring.address : 0);
   agc_shader_put_u64(&w, iface->has_gds_buffer ? iface->gds_buffer : 0);
   agc_shader_put_u64(&w, iface->has_mip_stats_table ? iface->mip_stats_table : 0);
   agc_shader_put_u64(&w, iface->has_image_reports ? iface->image_reports : 0);
   agc_shader_put_u64(&w, linkage->has_second_half ? linkage->second_half : 0);
   agc_shader_put_u32(&w, request->word_count);
   agc_shader_put_u32(&w, linkage->region_count);
   agc_shader_put_words(&w, request->words, request->word_count);
   if (linkage->region_count && !linkage->regions)
      agc_shader_writer_fail(&w, AGC_SHADER_FILE_UNREPRESENTABLE);
   for (uint32_t i = 0; i < linkage->region_count && w.status == AGC_SHADER_FILE_OK; i++) {
      const struct agc_shader_code_region *region = &linkage->regions[i];
      agc_shader_put_u64(&w, region->address);
      agc_shader_put_u32(&w, region->word_count);
      agc_shader_put_u32(&w, 0);
      agc_shader_put_words(&w, region->words, region->word_count);
   }
   return agc_shader_writer_finish(&w, data, size);
}

enum agc_shader_file_status
agc_shader_request_deserialize(const uint8_t *data, size_t size, struct agc_shader_request *request)
{
   struct agc_shader_reader r;
   struct agc_shader_request out;
   struct agc_shader_interface *iface = &out.iface;
   struct agc_shader_linkage *linkage = &out.linkage;

   memset(request, 0, sizeof(*request));
   memset(&out, 0, sizeof(out));
   enum agc_shader_file_status status =
      agc_shader_reader_open(&r, data, size, AGC_SHADER_REQUEST_MAGIC, AGC_SHADER_REQUEST_VERSION);
   if (status != AGC_SHADER_FILE_OK)
      return status;

   out.compiler_version = agc_shader_get_u32(&r);
   out.options.target = (enum agc_shader_isa)agc_shader_get_enum(&r, AGC_SHADER_ISA_COUNT);
   out.options.stage = (enum agc_shader_stage)agc_shader_get_enum(&r, AGC_SHADER_STAGE_COUNT);
   out.options.wave_size = agc_shader_get_u8(&r);
   out.options.ray_unit = (enum agc_shader_ray_unit)agc_shader_get_enum(&r, AGC_SHADER_RAY_UNIT_COUNT);
   out.precision = (enum agc_shader_precision)agc_shader_get_enum(&r, AGC_SHADER_PRECISION_COUNT);
   iface->vertex_stage = (enum agc_shader_vertex_stage)agc_shader_get_enum(&r, AGC_SHADER_VERTEX_STAGE_COUNT);
   iface->dual_source_blend = agc_shader_get_bool(&r);
   iface->preconvert_user_descriptors = agc_shader_get_bool(&r);
   uint32_t present = agc_shader_get_u32(&r);
   if (present & ~AGC_SHADER_REQUEST_PRESENT_MASK)
      agc_shader_reject(&r);
   iface->has_user_sgpr_count = (present & AGC_SHADER_REQUEST_HAS_USER_SGPR_COUNT) != 0;
   iface->has_guest_pgm_rsrc1 = (present & AGC_SHADER_REQUEST_HAS_GUEST_PGM_RSRC1) != 0;
   iface->has_guest_pgm_rsrc2 = (present & AGC_SHADER_REQUEST_HAS_GUEST_PGM_RSRC2) != 0;
   iface->has_guest_code_address = (present & AGC_SHADER_REQUEST_HAS_GUEST_CODE_ADDRESS) != 0;
   iface->has_ps_input_ena = (present & AGC_SHADER_REQUEST_HAS_PS_INPUT_ENA) != 0;
   iface->has_ps_input_addr = (present & AGC_SHADER_REQUEST_HAS_PS_INPUT_ADDR) != 0;
   iface->has_ps_col_format = (present & AGC_SHADER_REQUEST_HAS_PS_COL_FORMAT) != 0;
   iface->has_ps_z_format = (present & AGC_SHADER_REQUEST_HAS_PS_Z_FORMAT) != 0;
   iface->has_param_exports = (present & AGC_SHADER_REQUEST_HAS_PARAM_EXPORTS) != 0;
   iface->has_descriptor_lut = (present & AGC_SHADER_REQUEST_HAS_DESCRIPTOR_LUT) != 0;
   iface->has_attribute_ring = (present & AGC_SHADER_REQUEST_HAS_ATTRIBUTE_RING) != 0;
   iface->has_gds_buffer = (present & AGC_SHADER_REQUEST_HAS_GDS_BUFFER) != 0;
   iface->has_mip_stats_table = (present & AGC_SHADER_REQUEST_HAS_MIP_STATS_TABLE) != 0;
   iface->has_image_reports = (present & AGC_SHADER_REQUEST_HAS_IMAGE_REPORTS) != 0;
   linkage->has_second_half = (present & AGC_SHADER_REQUEST_HAS_SECOND_HALF) != 0;
   iface->user_sgpr_count = agc_shader_get_optional_u8(&r, iface->has_user_sgpr_count);
   iface->param_exports = agc_shader_get_optional_u8(&r, iface->has_param_exports);
   iface->native_dcc = agc_shader_get_bool(&r);
   iface->es_input = (enum agc_shader_es_input)agc_shader_get_enum(&r, AGC_SHADER_ES_INPUT_COUNT);
   iface->guest_pgm_rsrc1 = agc_shader_get_optional_u32(&r, iface->has_guest_pgm_rsrc1);
   iface->guest_pgm_rsrc2 = agc_shader_get_optional_u32(&r, iface->has_guest_pgm_rsrc2);
   iface->lds_bytes = agc_shader_get_u32(&r);
   iface->scratch_bytes_per_lane = agc_shader_get_u32(&r);
   iface->ps_input_ena = agc_shader_get_optional_u32(&r, iface->has_ps_input_ena);
   iface->ps_input_addr = agc_shader_get_optional_u32(&r, iface->has_ps_input_addr);
   iface->ps_col_format = agc_shader_get_optional_u32(&r, iface->has_ps_col_format);
   iface->ps_z_format = agc_shader_get_optional_u32(&r, iface->has_ps_z_format);
   iface->attribute_ring.bytes = agc_shader_get_optional_u32(&r, iface->has_attribute_ring);
   iface->guest_code_address = agc_shader_get_optional_u64(&r, iface->has_guest_code_address);
   iface->descriptor_lut = agc_shader_get_optional_u64(&r, iface->has_descriptor_lut);
   iface->attribute_ring.address = agc_shader_get_optional_u64(&r, iface->has_attribute_ring);
   iface->gds_buffer = agc_shader_get_optional_u64(&r, iface->has_gds_buffer);
   iface->mip_stats_table = agc_shader_get_optional_u64(&r, iface->has_mip_stats_table);
   iface->image_reports = agc_shader_get_optional_u64(&r, iface->has_image_reports);
   linkage->second_half = agc_shader_get_optional_u64(&r, linkage->has_second_half);
   out.word_count = agc_shader_get_u32(&r);
   linkage->region_count = agc_shader_get_u32(&r);

   size_t words_at = r.at;
   uint64_t region_words = 0;
   agc_shader_get_words(&r, NULL, out.word_count);
   if (agc_shader_fits(&r, linkage->region_count, AGC_SHADER_REGION_HEADER_BYTES)) {
      for (uint32_t i = 0; i < linkage->region_count && r.status == AGC_SHADER_FILE_OK; i++) {
         agc_shader_get_u64(&r);
         uint32_t count = agc_shader_get_u32(&r);
         agc_shader_get_zero(&r, 4);
         agc_shader_get_words(&r, NULL, count);
         region_words += count;
      }
   }
   if (r.status == AGC_SHADER_FILE_OK && r.at != r.size)
      agc_shader_reject(&r);
   if (r.status != AGC_SHADER_FILE_OK)
      return r.status;

   uint64_t region_bytes = (uint64_t)linkage->region_count * sizeof(struct agc_shader_code_region);
   uint64_t bytes = region_bytes + ((uint64_t)out.word_count + region_words) * sizeof(uint32_t);
   if (bytes > SIZE_MAX)
      return AGC_SHADER_FILE_OUT_OF_MEMORY;
   if (bytes) {
      out.storage = malloc((size_t)bytes);
      if (!out.storage)
         return AGC_SHADER_FILE_OUT_OF_MEMORY;
      struct agc_shader_code_region *regions = out.storage;
      uint32_t *words = (uint32_t *)((uint8_t *)out.storage + region_bytes);

      r.at = words_at;
      agc_shader_get_words(&r, words, out.word_count);
      out.words = out.word_count ? words : NULL;
      words += out.word_count;
      for (uint32_t i = 0; i < linkage->region_count; i++) {
         regions[i].address = agc_shader_get_u64(&r);
         regions[i].word_count = agc_shader_get_u32(&r);
         agc_shader_get_zero(&r, 4);
         agc_shader_get_words(&r, words, regions[i].word_count);
         regions[i].words = regions[i].word_count ? words : NULL;
         words += regions[i].word_count;
      }
      linkage->regions = linkage->region_count ? regions : NULL;
   }
   *request = out;
   return AGC_SHADER_FILE_OK;
}

void
agc_shader_request_free(struct agc_shader_request *request)
{
   free(request->storage);
   memset(request, 0, sizeof(*request));
}

static void
agc_shader_put_binding(struct agc_shader_writer *w, const struct agc_shader_binding *binding)
{
   agc_shader_put_enum(w, (unsigned)binding->kind, AGC_SHADER_RESOURCE_KIND_COUNT);
   agc_shader_put_u8(w, binding->base_sgpr);
   agc_shader_put_u8(w, binding->dword_count);
   agc_shader_put_bool(w, binding->writable);
}

static void
agc_shader_put_bindings(struct agc_shader_writer *w, const struct agc_shader_binding *bindings, uint32_t count)
{
   if (count && !bindings)
      agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);
   for (uint32_t i = 0; i < count && w->status == AGC_SHADER_FILE_OK; i++)
      agc_shader_put_binding(w, &bindings[i]);
}

static void
agc_shader_put_table(struct agc_shader_writer *w, const struct agc_shader_descriptor_table *table)
{
   agc_shader_put_u64(w, (uint64_t)table->offset);
   agc_shader_put_u32(w, table->link_count);
   agc_shader_put_u8(w, table->root);
   agc_shader_put_enum(w, (unsigned)table->root_form, AGC_SHADER_ADDRESS_FORM_COUNT);
   agc_shader_put_u16(w, 0);
   if (table->link_count && !table->links)
      agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);
   for (uint32_t i = 0; i < table->link_count && w->status == AGC_SHADER_FILE_OK; i++) {
      agc_shader_put_u64(w, (uint64_t)table->links[i].offset);
      agc_shader_put_enum(w, (unsigned)table->links[i].form, AGC_SHADER_ADDRESS_FORM_COUNT);
      agc_shader_put_zero(w, 3);
   }
}

static void
agc_shader_put_use(struct agc_shader_writer *w, const struct agc_shader_descriptor_use *use)
{
   const struct agc_shader_descriptor_source *source = &use->source;
   const struct agc_shader_buffer_access *access = &use->buffer_access;
   bool has_access = use->has_buffer_access;
   uint32_t table_count = 0;
   uint32_t offset_value = 0;

   if (source->kind == AGC_SHADER_SOURCE_TABLE) {
      if (source->table_count != 1)
         agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);
      table_count = 1;
   } else if (source->kind == AGC_SHADER_SOURCE_TABLE_CHOICES) {
      table_count = source->table_count;
   }
   if (table_count && !source->tables)
      agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);
   if (has_access && access->scalar_offset.kind == AGC_SHADER_SCALAR_OFFSET_BOUNDED)
      offset_value = access->scalar_offset.bound;
   else if (has_access && access->scalar_offset.kind == AGC_SHADER_SCALAR_OFFSET_USER_SGPR)
      offset_value = access->scalar_offset.sgpr;

   agc_shader_put_u32(w, use->pc);
   agc_shader_put_enum(w, (unsigned)use->kind, AGC_SHADER_RESOURCE_KIND_COUNT);
   agc_shader_put_u8(w, use->base_sgpr);
   agc_shader_put_u8(w, use->dword_count);
   agc_shader_put_enum(w, (unsigned)use->access, AGC_SHADER_ACCESS_COUNT);
   agc_shader_put_enum(w, (unsigned)source->kind, AGC_SHADER_SOURCE_KIND_COUNT);
   agc_shader_put_u8(w, source->kind == AGC_SHADER_SOURCE_USER_SGPRS ? source->first : 0);
   agc_shader_put_bool(w, has_access);
   agc_shader_put_bool(w, has_access && access->scalar);
   agc_shader_put_bool(w, has_access && access->indexed);
   agc_shader_put_bool(w, has_access && access->vector_offset);
   agc_shader_put_enum(w, has_access ? (unsigned)access->scalar_offset.kind : 0,
                       AGC_SHADER_SCALAR_OFFSET_KIND_COUNT);
   agc_shader_put_u8(w, 0);
   agc_shader_put_u32(w, has_access ? access->immediate_offset : 0);
   agc_shader_put_u32(w, offset_value);
   agc_shader_put_u32(w, table_count);
   for (uint32_t i = 0; i < table_count && w->status == AGC_SHADER_FILE_OK; i++)
      agc_shader_put_table(w, &source->tables[i]);
}

static void
agc_shader_put_uses(struct agc_shader_writer *w, const struct agc_shader_descriptor_use *uses, uint32_t count)
{
   if (count && !uses)
      agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);
   for (uint32_t i = 0; i < count && w->status == AGC_SHADER_FILE_OK; i++)
      agc_shader_put_use(w, &uses[i]);
}

static void
agc_shader_put_error(struct agc_shader_writer *w, const struct agc_shader_error *error)
{
   unsigned fields = agc_shader_error_fields[error->kind];
   bool has_opcode = (fields & AGC_SHADER_ERROR_FIELD_OPCODE) != 0;
   bool has_target = (fields & AGC_SHADER_ERROR_FIELD_TARGET) != 0;
   bool has_mismatch = (fields & AGC_SHADER_ERROR_FIELD_MISMATCH) != 0;
   const char *reason = (fields & AGC_SHADER_ERROR_FIELD_REASON) && error->reason ? error->reason : "";
   size_t reason_bytes = strlen(reason);

   agc_shader_put_enum(w, has_opcode ? (unsigned)error->family : 0, AGC_SHADER_FAMILY_COUNT);
   agc_shader_put_enum(w, has_target ? (unsigned)error->target : 0, AGC_SHADER_ISA_COUNT);
   agc_shader_put_enum(w, has_mismatch ? (unsigned)error->compiler : 0, AGC_SHADER_ISA_COUNT);
   agc_shader_put_enum(w, has_mismatch ? (unsigned)error->requested : 0, AGC_SHADER_ISA_COUNT);
   agc_shader_put_u32(w, fields & AGC_SHADER_ERROR_FIELD_PC ? error->pc : 0);
   agc_shader_put_u32(w, fields & AGC_SHADER_ERROR_FIELD_WORD ? error->word : 0);
   agc_shader_put_u32(w, fields & AGC_SHADER_ERROR_FIELD_BRANCH_TARGET ? error->branch_target : 0);
   agc_shader_put_u16(w, has_opcode ? error->opcode : 0);
   agc_shader_put_u16(w, fields & AGC_SHADER_ERROR_FIELD_VGPRS ? error->vgprs : 0);
   if (reason_bytes > UINT32_MAX) {
      agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);
      return;
   }
   agc_shader_put_u32(w, (uint32_t)reason_bytes);
   uint8_t *text = agc_shader_writer_grow(w, reason_bytes);
   if (text)
      memcpy(text, reason, reason_bytes);
   agc_shader_put_zero(w, (0u - (uint32_t)reason_bytes) & 3);
}

static void
agc_shader_put_translation(struct agc_shader_writer *w, const struct agc_shader_artifact *artifact)
{
   const struct agc_shader_ngg *ngg = &artifact->ngg;
   bool has_ngg = artifact->has_ngg;
   bool has_relocations = has_ngg && ngg->has_attribute_ring_relocations;
   bool has_fill = artifact->has_buffer_fill;
   uint8_t present = 0;

   if (artifact->has_user_data_tail_sgpr)
      present |= AGC_SHADER_ARTIFACT_HAS_USER_DATA_TAIL_SGPR;
   if (artifact->has_spi_shader_col_format)
      present |= AGC_SHADER_ARTIFACT_HAS_SPI_SHADER_COL_FORMAT;
   if (has_ngg)
      present |= AGC_SHADER_ARTIFACT_HAS_NGG;
   if (has_relocations)
      present |= AGC_SHADER_ARTIFACT_HAS_RING_RELOCATIONS;
   if (has_fill)
      present |= AGC_SHADER_ARTIFACT_HAS_BUFFER_FILL;
   if (artifact->program_words > artifact->code_count)
      agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);
   if (has_relocations && (ngg->attribute_ring_relocations[0] >= artifact->code_count ||
                           ngg->attribute_ring_relocations[1] >= artifact->code_count))
      agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);

   agc_shader_put_enum(w, (unsigned)artifact->options.target, AGC_SHADER_ISA_COUNT);
   agc_shader_put_enum(w, (unsigned)artifact->options.stage, AGC_SHADER_STAGE_COUNT);
   agc_shader_put_u8(w, artifact->options.wave_size);
   agc_shader_put_enum(w, (unsigned)artifact->options.ray_unit, AGC_SHADER_RAY_UNIT_COUNT);
   agc_shader_put_enum(w, (unsigned)artifact->precision, AGC_SHADER_PRECISION_COUNT);
   agc_shader_put_u8(w, artifact->user_sgpr_count);
   agc_shader_put_bool(w, artifact->uses_pc_relative);
   agc_shader_put_u8(w, present);
   agc_shader_put_u32(w, artifact->pgm_rsrc1);
   agc_shader_put_u32(w, artifact->pgm_rsrc2);
   agc_shader_put_u16(w, artifact->num_sgprs);
   agc_shader_put_u16(w, artifact->num_vgprs);
   agc_shader_put_u32(w, artifact->lds_bytes);
   agc_shader_put_u32(w, artifact->scratch_bytes_per_lane);
   agc_shader_put_u32(w, artifact->program_words);
   agc_shader_put_u32(w, artifact->guest_words);
   agc_shader_put_u64(w, artifact->guest_hash);
   agc_shader_put_u32(w, artifact->exports.color_mask);
   agc_shader_put_u32(w, artifact->exports.param_mask);
   agc_shader_put_u8(w, artifact->exports.depth_mask);
   agc_shader_put_u8(w, artifact->exports.position_mask);
   agc_shader_put_bool(w, artifact->exports.primitive);
   agc_shader_put_bool(w, artifact->exports.compressed);
   agc_shader_put_bool(w, artifact->exports.valid_mask);
   agc_shader_put_u8(w, artifact->has_user_data_tail_sgpr ? artifact->user_data_tail_sgpr : 0);
   agc_shader_put_u8(w, has_fill ? artifact->buffer_fill.buffer : 0);
   agc_shader_put_u8(w, has_fill ? artifact->buffer_fill.value : 0);
   agc_shader_put_u32(w, artifact->has_spi_shader_col_format ? artifact->spi_shader_col_format : 0);
   agc_shader_put_u32(w, has_ngg ? ngg->param_count : 0);
   agc_shader_put_u32(w, has_ngg ? ngg->spi_vs_out_config : 0);
   agc_shader_put_u32(w, has_relocations ? ngg->attribute_ring_relocations[0] : 0);
   agc_shader_put_u32(w, has_relocations ? ngg->attribute_ring_relocations[1] : 0);
   for (unsigned i = 0; i < AGC_SHADER_NGG_REGISTER_COUNT; i++) {
      agc_shader_put_u32(w, has_ngg ? ngg->registers[i].reg : 0);
      agc_shader_put_enum(w, has_ngg ? (unsigned)ngg->registers[i].source : 0, AGC_SHADER_REGISTER_SOURCE_COUNT);
      agc_shader_put_zero(w, 3);
   }
   agc_shader_put_u32(w, artifact->code_count);
   agc_shader_put_u32(w, artifact->binding_count);
   agc_shader_put_u32(w, artifact->preconverted_descriptor_count);
   agc_shader_put_u32(w, artifact->external_target_count);
   agc_shader_put_u32(w, artifact->descriptor_use_count);
   agc_shader_put_u32(w, artifact->pointer_use_count);
   agc_shader_put_words(w, artifact->code, artifact->code_count);
   agc_shader_put_bindings(w, artifact->bindings, artifact->binding_count);
   agc_shader_put_bindings(w, artifact->preconverted_descriptors, artifact->preconverted_descriptor_count);
   if (artifact->external_target_count && !artifact->external_targets)
      agc_shader_writer_fail(w, AGC_SHADER_FILE_UNREPRESENTABLE);
   for (uint32_t i = 0; i < artifact->external_target_count && w->status == AGC_SHADER_FILE_OK; i++) {
      const struct agc_shader_external_target *target = &artifact->external_targets[i];
      agc_shader_put_u32(w, target->pc);
      agc_shader_put_bool(w, target->call);
      agc_shader_put_bool(w, target->has_sgpr);
      agc_shader_put_u8(w, target->has_sgpr ? target->sgpr : 0);
      agc_shader_put_u8(w, 0);
   }
   agc_shader_put_uses(w, artifact->descriptor_uses, artifact->descriptor_use_count);
   agc_shader_put_uses(w, artifact->pointer_uses, artifact->pointer_use_count);
}

enum agc_shader_file_status
agc_shader_artifact_serialize(const struct agc_shader_artifact *artifact, uint8_t **data, size_t *size)
{
   struct agc_shader_writer w = {0};

   agc_shader_put_header(&w, AGC_SHADER_ARTIFACT_MAGIC, AGC_SHADER_ARTIFACT_VERSION);
   agc_shader_put_enum(&w, (unsigned)artifact->error.kind, AGC_SHADER_ERROR_KIND_COUNT);
   agc_shader_put_zero(&w, 3);
   if (w.status == AGC_SHADER_FILE_OK) {
      if (artifact->error.kind != AGC_SHADER_ERROR_NONE)
         agc_shader_put_error(&w, &artifact->error);
      else
         agc_shader_put_translation(&w, artifact);
   }
   return agc_shader_writer_finish(&w, data, size);
}

struct agc_shader_artifact_arrays {
   struct agc_shader_descriptor_use *descriptor_uses;
   struct agc_shader_descriptor_use *pointer_uses;
   struct agc_shader_descriptor_table *tables;
   struct agc_shader_table_link *links;
   struct agc_shader_binding *bindings;
   struct agc_shader_binding *preconverted_descriptors;
   struct agc_shader_external_target *external_targets;
   uint32_t *code;
   char *reason;
   uint64_t table_count;
   uint64_t link_count;
   uint64_t reason_size;
};

static void *
agc_shader_carve(void *base, uint64_t *offset, uint64_t count, size_t element_bytes, size_t alignment)
{
   uint64_t at = (*offset + alignment - 1) & ~((uint64_t)alignment - 1);
   *offset = at + count * element_bytes;
   return base && count ? (uint8_t *)base + at : NULL;
}

static uint64_t
agc_shader_artifact_layout(const struct agc_shader_artifact *counts, uint64_t table_count, uint64_t link_count,
                           uint64_t reason_size, void *base, struct agc_shader_artifact_arrays *arrays)
{
   uint64_t offset = 0;

   memset(arrays, 0, sizeof(*arrays));
   arrays->descriptor_uses = agc_shader_carve(base, &offset, counts->descriptor_use_count,
                                              sizeof(struct agc_shader_descriptor_use),
                                              _Alignof(struct agc_shader_descriptor_use));
   arrays->pointer_uses = agc_shader_carve(base, &offset, counts->pointer_use_count,
                                           sizeof(struct agc_shader_descriptor_use),
                                           _Alignof(struct agc_shader_descriptor_use));
   arrays->tables = agc_shader_carve(base, &offset, table_count, sizeof(struct agc_shader_descriptor_table),
                                     _Alignof(struct agc_shader_descriptor_table));
   arrays->links = agc_shader_carve(base, &offset, link_count, sizeof(struct agc_shader_table_link),
                                    _Alignof(struct agc_shader_table_link));
   arrays->bindings = agc_shader_carve(base, &offset, counts->binding_count, sizeof(struct agc_shader_binding),
                                       _Alignof(struct agc_shader_binding));
   arrays->preconverted_descriptors =
      agc_shader_carve(base, &offset, counts->preconverted_descriptor_count, sizeof(struct agc_shader_binding),
                       _Alignof(struct agc_shader_binding));
   arrays->external_targets =
      agc_shader_carve(base, &offset, counts->external_target_count, sizeof(struct agc_shader_external_target),
                       _Alignof(struct agc_shader_external_target));
   arrays->code = agc_shader_carve(base, &offset, counts->code_count, sizeof(uint32_t), _Alignof(uint32_t));
   arrays->reason = agc_shader_carve(base, &offset, reason_size, 1, 1);
   return offset;
}

static void
agc_shader_get_binding(struct agc_shader_reader *r, struct agc_shader_binding *binding)
{
   binding->kind = (enum agc_shader_resource_kind)agc_shader_get_enum(r, AGC_SHADER_RESOURCE_KIND_COUNT);
   binding->base_sgpr = agc_shader_get_u8(r);
   binding->dword_count = agc_shader_get_u8(r);
   binding->writable = agc_shader_get_bool(r);
}

static void
agc_shader_get_bindings(struct agc_shader_reader *r, struct agc_shader_binding *bindings, uint32_t count)
{
   if (!agc_shader_fits(r, count, AGC_SHADER_BINDING_BYTES))
      return;
   for (uint32_t i = 0; i < count; i++) {
      struct agc_shader_binding binding;
      agc_shader_get_binding(r, &binding);
      if (bindings)
         bindings[i] = binding;
   }
}

static void
agc_shader_get_table(struct agc_shader_reader *r, struct agc_shader_artifact_arrays *arrays,
                     struct agc_shader_descriptor_table *table)
{
   memset(table, 0, sizeof(*table));
   table->offset = (int64_t)agc_shader_get_u64(r);
   uint32_t link_count = agc_shader_get_u32(r);
   table->root = agc_shader_get_u8(r);
   table->root_form = (enum agc_shader_address_form)agc_shader_get_enum(r, AGC_SHADER_ADDRESS_FORM_COUNT);
   agc_shader_get_zero(r, 2);
   if (!agc_shader_fits(r, link_count, AGC_SHADER_LINK_BYTES))
      return;
   struct agc_shader_table_link *links = arrays->links ? arrays->links + arrays->link_count : NULL;
   arrays->link_count += link_count;
   for (uint32_t i = 0; i < link_count; i++) {
      struct agc_shader_table_link link;
      link.offset = (int64_t)agc_shader_get_u64(r);
      link.form = (enum agc_shader_address_form)agc_shader_get_enum(r, AGC_SHADER_ADDRESS_FORM_COUNT);
      agc_shader_get_zero(r, 3);
      if (links)
         links[i] = link;
   }
   table->links = link_count ? links : NULL;
   table->link_count = link_count;
}

static void
agc_shader_get_use(struct agc_shader_reader *r, struct agc_shader_artifact_arrays *arrays,
                   struct agc_shader_descriptor_use *use)
{
   struct agc_shader_descriptor_source *source = &use->source;
   struct agc_shader_buffer_access *access = &use->buffer_access;

   memset(use, 0, sizeof(*use));
   use->pc = agc_shader_get_u32(r);
   use->kind = (enum agc_shader_resource_kind)agc_shader_get_enum(r, AGC_SHADER_RESOURCE_KIND_COUNT);
   use->base_sgpr = agc_shader_get_u8(r);
   use->dword_count = agc_shader_get_u8(r);
   use->access = (enum agc_shader_descriptor_access)agc_shader_get_enum(r, AGC_SHADER_ACCESS_COUNT);
   source->kind = (enum agc_shader_source_kind)agc_shader_get_enum(r, AGC_SHADER_SOURCE_KIND_COUNT);
   source->first = agc_shader_get_optional_u8(r, source->kind == AGC_SHADER_SOURCE_USER_SGPRS);
   bool has_access = agc_shader_get_bool(r);
   use->has_buffer_access = has_access;
   access->scalar = agc_shader_get_bool(r);
   access->indexed = agc_shader_get_bool(r);
   access->vector_offset = agc_shader_get_bool(r);
   access->scalar_offset.kind =
      (enum agc_shader_scalar_offset_kind)agc_shader_get_enum(r, AGC_SHADER_SCALAR_OFFSET_KIND_COUNT);
   agc_shader_get_zero(r, 1);
   access->immediate_offset = agc_shader_get_optional_u32(r, has_access);
   uint32_t offset_value = agc_shader_get_optional_u32(r, has_access);
   uint32_t table_count = agc_shader_get_u32(r);

   if (!has_access && (access->scalar || access->indexed || access->vector_offset ||
                       access->scalar_offset.kind != AGC_SHADER_SCALAR_OFFSET_BOUNDED))
      agc_shader_reject(r);
   if (access->scalar_offset.kind == AGC_SHADER_SCALAR_OFFSET_BOUNDED) {
      access->scalar_offset.bound = offset_value;
   } else if (access->scalar_offset.kind == AGC_SHADER_SCALAR_OFFSET_USER_SGPR) {
      if (offset_value > UINT8_MAX)
         agc_shader_reject(r);
      access->scalar_offset.sgpr = (uint8_t)offset_value;
   } else if (offset_value) {
      agc_shader_reject(r);
   }
   if (source->kind == AGC_SHADER_SOURCE_TABLE) {
      if (table_count != 1)
         agc_shader_reject(r);
   } else if (source->kind != AGC_SHADER_SOURCE_TABLE_CHOICES && table_count) {
      agc_shader_reject(r);
   }
   if (!agc_shader_fits(r, table_count, AGC_SHADER_TABLE_BYTES))
      return;
   struct agc_shader_descriptor_table *tables = arrays->tables ? arrays->tables + arrays->table_count : NULL;
   arrays->table_count += table_count;
   for (uint32_t i = 0; i < table_count; i++) {
      struct agc_shader_descriptor_table table;
      agc_shader_get_table(r, arrays, &table);
      if (tables)
         tables[i] = table;
   }
   source->tables = table_count ? tables : NULL;
   source->table_count = table_count;
}

static void
agc_shader_get_uses(struct agc_shader_reader *r, struct agc_shader_artifact_arrays *arrays,
                    struct agc_shader_descriptor_use *uses, uint32_t count)
{
   if (!agc_shader_fits(r, count, AGC_SHADER_USE_BYTES))
      return;
   for (uint32_t i = 0; i < count && r->status == AGC_SHADER_FILE_OK; i++) {
      struct agc_shader_descriptor_use use;
      agc_shader_get_use(r, arrays, &use);
      if (uses)
         uses[i] = use;
   }
}

static void
agc_shader_get_error(struct agc_shader_reader *r, enum agc_shader_error_kind kind,
                     struct agc_shader_artifact_arrays *arrays, struct agc_shader_error *error)
{
   unsigned fields = agc_shader_error_fields[kind];
   bool has_opcode = (fields & AGC_SHADER_ERROR_FIELD_OPCODE) != 0;
   bool has_mismatch = (fields & AGC_SHADER_ERROR_FIELD_MISMATCH) != 0;
   bool has_reason = (fields & AGC_SHADER_ERROR_FIELD_REASON) != 0;

   error->kind = kind;
   error->family = (enum agc_shader_family)agc_shader_get_enum(r, AGC_SHADER_FAMILY_COUNT);
   error->target = (enum agc_shader_isa)agc_shader_get_enum(r, AGC_SHADER_ISA_COUNT);
   error->compiler = (enum agc_shader_isa)agc_shader_get_enum(r, AGC_SHADER_ISA_COUNT);
   error->requested = (enum agc_shader_isa)agc_shader_get_enum(r, AGC_SHADER_ISA_COUNT);
   error->pc = agc_shader_get_optional_u32(r, (fields & AGC_SHADER_ERROR_FIELD_PC) != 0);
   error->word = agc_shader_get_optional_u32(r, (fields & AGC_SHADER_ERROR_FIELD_WORD) != 0);
   error->branch_target = agc_shader_get_optional_u32(r, (fields & AGC_SHADER_ERROR_FIELD_BRANCH_TARGET) != 0);
   error->opcode = agc_shader_get_u16(r);
   error->vgprs = agc_shader_get_u16(r);
   uint32_t reason_bytes = agc_shader_get_optional_u32(r, has_reason);
   const uint8_t *text = agc_shader_take(r, reason_bytes);
   agc_shader_get_zero(r, (0u - reason_bytes) & 3);

   if ((!has_opcode && (error->family || error->opcode)) ||
       (!(fields & AGC_SHADER_ERROR_FIELD_TARGET) && error->target) ||
       (!has_mismatch && (error->compiler || error->requested)) ||
       (!(fields & AGC_SHADER_ERROR_FIELD_VGPRS) && error->vgprs))
      agc_shader_reject(r);
   if (!text || !has_reason)
      return;
   if (memchr(text, 0, reason_bytes)) {
      agc_shader_reject(r);
      return;
   }
   arrays->reason_size = (uint64_t)reason_bytes + 1;
   if (!arrays->reason)
      return;
   memcpy(arrays->reason, text, reason_bytes);
   arrays->reason[reason_bytes] = 0;
   error->reason = arrays->reason;
}

static void
agc_shader_get_translation(struct agc_shader_reader *r, struct agc_shader_artifact_arrays *arrays,
                           struct agc_shader_artifact *out)
{
   struct agc_shader_ngg *ngg = &out->ngg;

   out->options.target = (enum agc_shader_isa)agc_shader_get_enum(r, AGC_SHADER_ISA_COUNT);
   out->options.stage = (enum agc_shader_stage)agc_shader_get_enum(r, AGC_SHADER_STAGE_COUNT);
   out->options.wave_size = agc_shader_get_u8(r);
   out->options.ray_unit = (enum agc_shader_ray_unit)agc_shader_get_enum(r, AGC_SHADER_RAY_UNIT_COUNT);
   out->precision = (enum agc_shader_precision)agc_shader_get_enum(r, AGC_SHADER_PRECISION_COUNT);
   out->user_sgpr_count = agc_shader_get_u8(r);
   out->uses_pc_relative = agc_shader_get_bool(r);
   uint8_t present = agc_shader_get_u8(r);
   if (present & ~AGC_SHADER_ARTIFACT_PRESENT_MASK)
      agc_shader_reject(r);
   out->has_user_data_tail_sgpr = (present & AGC_SHADER_ARTIFACT_HAS_USER_DATA_TAIL_SGPR) != 0;
   out->has_spi_shader_col_format = (present & AGC_SHADER_ARTIFACT_HAS_SPI_SHADER_COL_FORMAT) != 0;
   out->has_ngg = (present & AGC_SHADER_ARTIFACT_HAS_NGG) != 0;
   ngg->has_attribute_ring_relocations = (present & AGC_SHADER_ARTIFACT_HAS_RING_RELOCATIONS) != 0;
   out->has_buffer_fill = (present & AGC_SHADER_ARTIFACT_HAS_BUFFER_FILL) != 0;
   if (ngg->has_attribute_ring_relocations && !out->has_ngg)
      agc_shader_reject(r);
   out->pgm_rsrc1 = agc_shader_get_u32(r);
   out->pgm_rsrc2 = agc_shader_get_u32(r);
   out->num_sgprs = agc_shader_get_u16(r);
   out->num_vgprs = agc_shader_get_u16(r);
   out->lds_bytes = agc_shader_get_u32(r);
   out->scratch_bytes_per_lane = agc_shader_get_u32(r);
   out->program_words = agc_shader_get_u32(r);
   out->guest_words = agc_shader_get_u32(r);
   out->guest_hash = agc_shader_get_u64(r);
   out->exports.color_mask = agc_shader_get_u32(r);
   out->exports.param_mask = agc_shader_get_u32(r);
   out->exports.depth_mask = agc_shader_get_u8(r);
   out->exports.position_mask = agc_shader_get_u8(r);
   out->exports.primitive = agc_shader_get_bool(r);
   out->exports.compressed = agc_shader_get_bool(r);
   out->exports.valid_mask = agc_shader_get_bool(r);
   out->user_data_tail_sgpr = agc_shader_get_optional_u8(r, out->has_user_data_tail_sgpr);
   out->buffer_fill.buffer = agc_shader_get_optional_u8(r, out->has_buffer_fill);
   out->buffer_fill.value = agc_shader_get_optional_u8(r, out->has_buffer_fill);
   out->spi_shader_col_format = agc_shader_get_optional_u32(r, out->has_spi_shader_col_format);
   ngg->param_count = agc_shader_get_optional_u32(r, out->has_ngg);
   ngg->spi_vs_out_config = agc_shader_get_optional_u32(r, out->has_ngg);
   ngg->attribute_ring_relocations[0] = agc_shader_get_optional_u32(r, ngg->has_attribute_ring_relocations);
   ngg->attribute_ring_relocations[1] = agc_shader_get_optional_u32(r, ngg->has_attribute_ring_relocations);
   for (unsigned i = 0; i < AGC_SHADER_NGG_REGISTER_COUNT; i++) {
      ngg->registers[i].reg = agc_shader_get_optional_u32(r, out->has_ngg);
      ngg->registers[i].source =
         (enum agc_shader_register_source)agc_shader_get_enum(r, AGC_SHADER_REGISTER_SOURCE_COUNT);
      if (!out->has_ngg && ngg->registers[i].source != AGC_SHADER_REGISTER_CODE_ADDRESS)
         agc_shader_reject(r);
      agc_shader_get_zero(r, 3);
   }
   out->code_count = agc_shader_get_u32(r);
   out->binding_count = agc_shader_get_u32(r);
   out->preconverted_descriptor_count = agc_shader_get_u32(r);
   out->external_target_count = agc_shader_get_u32(r);
   out->descriptor_use_count = agc_shader_get_u32(r);
   out->pointer_use_count = agc_shader_get_u32(r);
   if (out->program_words > out->code_count)
      agc_shader_reject(r);
   if (ngg->has_attribute_ring_relocations && (ngg->attribute_ring_relocations[0] >= out->code_count ||
                                               ngg->attribute_ring_relocations[1] >= out->code_count))
      agc_shader_reject(r);

   agc_shader_get_words(r, arrays->code, out->code_count);
   agc_shader_get_bindings(r, arrays->bindings, out->binding_count);
   agc_shader_get_bindings(r, arrays->preconverted_descriptors, out->preconverted_descriptor_count);
   if (agc_shader_fits(r, out->external_target_count, AGC_SHADER_EXTERNAL_TARGET_BYTES)) {
      for (uint32_t i = 0; i < out->external_target_count; i++) {
         struct agc_shader_external_target target;
         target.pc = agc_shader_get_u32(r);
         target.call = agc_shader_get_bool(r);
         target.has_sgpr = agc_shader_get_bool(r);
         target.sgpr = agc_shader_get_optional_u8(r, target.has_sgpr);
         agc_shader_get_zero(r, 1);
         if (arrays->external_targets)
            arrays->external_targets[i] = target;
      }
   }
   agc_shader_get_uses(r, arrays, arrays->descriptor_uses, out->descriptor_use_count);
   agc_shader_get_uses(r, arrays, arrays->pointer_uses, out->pointer_use_count);
   out->code = arrays->code;
   out->bindings = arrays->bindings;
   out->preconverted_descriptors = arrays->preconverted_descriptors;
   out->external_targets = arrays->external_targets;
   out->descriptor_uses = arrays->descriptor_uses;
   out->pointer_uses = arrays->pointer_uses;
}

static void
agc_shader_get_artifact(struct agc_shader_reader *r, struct agc_shader_artifact_arrays *arrays,
                        struct agc_shader_artifact *out)
{
   memset(out, 0, sizeof(*out));
   r->at = AGC_SHADER_FILE_HEADER_BYTES;
   arrays->table_count = 0;
   arrays->link_count = 0;
   arrays->reason_size = 0;
   enum agc_shader_error_kind kind =
      (enum agc_shader_error_kind)agc_shader_get_enum(r, AGC_SHADER_ERROR_KIND_COUNT);
   agc_shader_get_zero(r, 3);
   if (r->status != AGC_SHADER_FILE_OK)
      return;
   if (kind != AGC_SHADER_ERROR_NONE)
      agc_shader_get_error(r, kind, arrays, &out->error);
   else
      agc_shader_get_translation(r, arrays, out);
   if (r->status == AGC_SHADER_FILE_OK && r->at != r->size)
      agc_shader_reject(r);
}

enum agc_shader_file_status
agc_shader_artifact_deserialize(const uint8_t *data, size_t size, struct agc_shader_artifact *artifact)
{
   struct agc_shader_reader r;
   struct agc_shader_artifact out;
   struct agc_shader_artifact_arrays arrays;

   memset(artifact, 0, sizeof(*artifact));
   enum agc_shader_file_status status =
      agc_shader_reader_open(&r, data, size, AGC_SHADER_ARTIFACT_MAGIC, AGC_SHADER_ARTIFACT_VERSION);
   if (status != AGC_SHADER_FILE_OK)
      return status;

   memset(&arrays, 0, sizeof(arrays));
   agc_shader_get_artifact(&r, &arrays, &out);
   if (r.status != AGC_SHADER_FILE_OK)
      return r.status;

   uint64_t table_count = arrays.table_count;
   uint64_t link_count = arrays.link_count;
   uint64_t reason_size = arrays.reason_size;
   uint64_t bytes = agc_shader_artifact_layout(&out, table_count, link_count, reason_size, NULL, &arrays);
   void *storage = NULL;
   if (bytes > SIZE_MAX)
      return AGC_SHADER_FILE_OUT_OF_MEMORY;
   if (bytes) {
      storage = malloc((size_t)bytes);
      if (!storage)
         return AGC_SHADER_FILE_OUT_OF_MEMORY;
   }
   agc_shader_artifact_layout(&out, table_count, link_count, reason_size, storage, &arrays);
   agc_shader_get_artifact(&r, &arrays, &out);
   if (r.status != AGC_SHADER_FILE_OK) {
      free(storage);
      return r.status;
   }
   out.storage = storage;
   *artifact = out;
   return AGC_SHADER_FILE_OK;
}

void
agc_shader_artifact_free(struct agc_shader_artifact *artifact)
{
   free(artifact->storage);
   memset(artifact, 0, sizeof(*artifact));
}

void
agc_shader_key_of_bytes(const uint8_t *data, size_t size, struct agc_shader_key *key)
{
   _Static_assert(BLAKE3_OUT_LEN == AGC_SHADER_KEY_BYTES, "key size");
   _mesa_blake3_compute(data, size, key->bytes);
}

enum agc_shader_file_status
agc_shader_request_key(const struct agc_shader_request *request, struct agc_shader_key *key)
{
   uint8_t *data;
   size_t size;

   memset(key, 0, sizeof(*key));
   enum agc_shader_file_status status = agc_shader_request_serialize(request, &data, &size);
   if (status != AGC_SHADER_FILE_OK)
      return status;
   agc_shader_key_of_bytes(data, size, key);
   free(data);
   return AGC_SHADER_FILE_OK;
}

void
agc_shader_key_hex(const struct agc_shader_key *key, char hex[AGC_SHADER_KEY_HEX_SIZE])
{
   static const char digits[] = "0123456789abcdef";

   for (unsigned i = 0; i < AGC_SHADER_KEY_BYTES; i++) {
      hex[2 * i] = digits[key->bytes[i] >> 4];
      hex[2 * i + 1] = digits[key->bytes[i] & 0xf];
   }
   hex[2 * AGC_SHADER_KEY_BYTES] = 0;
}

static int
agc_shader_hex_digit(char c)
{
   if (c >= '0' && c <= '9')
      return c - '0';
   if (c >= 'a' && c <= 'f')
      return c - 'a' + 10;
   return -1;
}

bool
agc_shader_key_from_hex(const char *hex, struct agc_shader_key *key)
{
   memset(key, 0, sizeof(*key));
   for (unsigned i = 0; i < AGC_SHADER_KEY_BYTES; i++) {
      int high = agc_shader_hex_digit(hex[2 * i]);
      int low = high < 0 ? -1 : agc_shader_hex_digit(hex[2 * i + 1]);
      if (low < 0) {
         memset(key, 0, sizeof(*key));
         return false;
      }
      key->bytes[i] = (uint8_t)(high << 4 | low);
   }
   if (hex[2 * AGC_SHADER_KEY_BYTES]) {
      memset(key, 0, sizeof(*key));
      return false;
   }
   return true;
}

static enum agc_shader_file_status
agc_shader_read_bytes(const char *path, uint8_t **data, size_t *size)
{
   errno = 0;
   char *bytes = os_read_file(path, size);
   if (!bytes) {
      *data = NULL;
      *size = 0;
      return errno == ENOENT ? AGC_SHADER_FILE_NOT_FOUND : AGC_SHADER_FILE_IO_ERROR;
   }
   *data = (uint8_t *)bytes;
   return AGC_SHADER_FILE_OK;
}

static FILE *
agc_shader_create_exclusive(const char *path)
{
#if DETECT_OS_WINDOWS
   int fd = _open(path, _O_CREAT | _O_EXCL | _O_WRONLY | _O_BINARY, _S_IREAD | _S_IWRITE);
   if (fd < 0)
      return NULL;
   FILE *file = _fdopen(fd, "wb");
   if (!file)
      _close(fd);
   return file;
#else
   int fd = open(path, O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0644);
   if (fd < 0)
      return NULL;
   FILE *file = fdopen(fd, "wb");
   if (!file)
      close(fd);
   return file;
#endif
}

static bool
agc_shader_replace_file(const char *from, const char *to)
{
#if DETECT_OS_WINDOWS
   return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING) != 0;
#else
   return rename(from, to) == 0;
#endif
}

static unsigned long
agc_shader_process_id(void)
{
#if DETECT_OS_WINDOWS
   return (unsigned long)_getpid();
#else
   return (unsigned long)getpid();
#endif
}

static bool
agc_shader_file_holds(const char *path, const uint8_t *data, size_t size)
{
   uint8_t *stored;
   size_t stored_size;
   bool same = agc_shader_read_bytes(path, &stored, &stored_size) == AGC_SHADER_FILE_OK &&
               stored_size == size && memcmp(stored, data, size) == 0;

   free(stored);
   return same;
}

static enum agc_shader_file_status
agc_shader_write_bytes(const char *path, const uint8_t *data, size_t size)
{
   size_t length = strlen(path) + 48;
   char *temp = malloc(length);
   FILE *file = NULL;

   if (!temp)
      return AGC_SHADER_FILE_OUT_OF_MEMORY;
   for (unsigned attempt = 0; attempt < 16 && !file; attempt++) {
      snprintf(temp, length, "%s.%lx-%x.tmp", path, agc_shader_process_id(),
               (unsigned)p_atomic_inc_return(&agc_shader_temp_counter));
      errno = 0;
      file = agc_shader_create_exclusive(temp);
      if (!file && errno != EEXIST)
         break;
   }
   if (!file) {
      free(temp);
      return agc_shader_file_holds(path, data, size) ? AGC_SHADER_FILE_OK : AGC_SHADER_FILE_IO_ERROR;
   }
   bool ok = fwrite(data, 1, size, file) == size;
   ok = fclose(file) == 0 && ok;
   ok = ok && agc_shader_replace_file(temp, path);
   if (!ok) {
      remove(temp);
      ok = agc_shader_file_holds(path, data, size);
   }
   free(temp);
   return ok ? AGC_SHADER_FILE_OK : AGC_SHADER_FILE_IO_ERROR;
}

enum agc_shader_file_status
agc_shader_request_write_file(const struct agc_shader_request *request, const char *path)
{
   uint8_t *data;
   size_t size;
   enum agc_shader_file_status status = agc_shader_request_serialize(request, &data, &size);

   if (status != AGC_SHADER_FILE_OK)
      return status;
   status = agc_shader_write_bytes(path, data, size);
   free(data);
   return status;
}

enum agc_shader_file_status
agc_shader_request_read_file(const char *path, struct agc_shader_request *request)
{
   uint8_t *data;
   size_t size;

   memset(request, 0, sizeof(*request));
   enum agc_shader_file_status status = agc_shader_read_bytes(path, &data, &size);
   if (status != AGC_SHADER_FILE_OK)
      return status;
   status = agc_shader_request_deserialize(data, size, request);
   free(data);
   return status;
}

enum agc_shader_file_status
agc_shader_artifact_write_file(const struct agc_shader_artifact *artifact, const char *path)
{
   uint8_t *data;
   size_t size;
   enum agc_shader_file_status status = agc_shader_artifact_serialize(artifact, &data, &size);

   if (status != AGC_SHADER_FILE_OK)
      return status;
   status = agc_shader_write_bytes(path, data, size);
   free(data);
   return status;
}

enum agc_shader_file_status
agc_shader_artifact_read_file(const char *path, struct agc_shader_artifact *artifact)
{
   uint8_t *data;
   size_t size;

   memset(artifact, 0, sizeof(*artifact));
   enum agc_shader_file_status status = agc_shader_read_bytes(path, &data, &size);
   if (status != AGC_SHADER_FILE_OK)
      return status;
   status = agc_shader_artifact_deserialize(data, size, artifact);
   free(data);
   return status;
}

struct agc_shader_cache {
   char *directory;
   char *requests;
};

static bool
agc_shader_is_directory(const char *path)
{
   struct stat info;

   if (stat(path, &info) != 0)
      return false;
#if DETECT_OS_WINDOWS
   return (info.st_mode & _S_IFMT) == _S_IFDIR;
#else
   return S_ISDIR(info.st_mode);
#endif
}

static bool
agc_shader_make_directories(char *path)
{
   for (char *p = path + 1; *p; p++) {
      if (*p != '/' && *p != '\\')
         continue;
      char separator = *p;
      *p = 0;
      os_mkdir(path, 0755);
      *p = separator;
   }
   os_mkdir(path, 0755);
   return agc_shader_is_directory(path);
}

static char *
agc_shader_join(const char *directory, const char *name, const char *extension)
{
   size_t length = strlen(directory) + strlen(name) + strlen(extension) + 2;
   char *path = malloc(length);

   if (path)
      snprintf(path, length, "%s/%s%s", directory, name, extension);
   return path;
}

static char *
agc_shader_key_path(const char *directory, const struct agc_shader_key *key, const char *extension)
{
   char hex[AGC_SHADER_KEY_HEX_SIZE];

   agc_shader_key_hex(key, hex);
   return agc_shader_join(directory, hex, extension);
}

void
agc_shader_cache_close(struct agc_shader_cache *cache)
{
   if (!cache)
      return;
   free(cache->directory);
   free(cache->requests);
   free(cache);
}

enum agc_shader_file_status
agc_shader_cache_open(const char *directory, struct agc_shader_cache **cache)
{
   size_t length = strlen(directory);

   *cache = NULL;
   while (length > 1 && (directory[length - 1] == '/' || directory[length - 1] == '\\') &&
          directory[length - 2] != ':')
      length--;
   if (!length)
      return AGC_SHADER_FILE_IO_ERROR;
   struct agc_shader_cache *out = calloc(1, sizeof(*out));
   if (!out)
      return AGC_SHADER_FILE_OUT_OF_MEMORY;
   out->directory = malloc(length + 1);
   if (out->directory) {
      memcpy(out->directory, directory, length);
      out->directory[length] = 0;
      out->requests = agc_shader_join(out->directory, AGC_SHADER_REQUEST_DIRECTORY, "");
   }
   if (!out->directory || !out->requests) {
      agc_shader_cache_close(out);
      return AGC_SHADER_FILE_OUT_OF_MEMORY;
   }
   if (!agc_shader_make_directories(out->directory) || !agc_shader_make_directories(out->requests)) {
      agc_shader_cache_close(out);
      return AGC_SHADER_FILE_IO_ERROR;
   }
   *cache = out;
   return AGC_SHADER_FILE_OK;
}

enum agc_shader_file_status
agc_shader_cache_lookup(const struct agc_shader_cache *cache, const struct agc_shader_key *key,
                        struct agc_shader_artifact *artifact)
{
   char *path = agc_shader_key_path(cache->directory, key, AGC_SHADER_ARTIFACT_EXTENSION);

   memset(artifact, 0, sizeof(*artifact));
   if (!path)
      return AGC_SHADER_FILE_OUT_OF_MEMORY;
   enum agc_shader_file_status status = agc_shader_artifact_read_file(path, artifact);
   free(path);
   return status;
}

enum agc_shader_file_status
agc_shader_cache_store(const struct agc_shader_cache *cache, const struct agc_shader_key *key,
                       const struct agc_shader_artifact *artifact)
{
   char *path = agc_shader_key_path(cache->directory, key, AGC_SHADER_ARTIFACT_EXTENSION);

   if (!path)
      return AGC_SHADER_FILE_OUT_OF_MEMORY;
   enum agc_shader_file_status status = agc_shader_artifact_write_file(artifact, path);
   free(path);
   return status;
}

enum agc_shader_file_status
agc_shader_cache_miss(const struct agc_shader_cache *cache, const struct agc_shader_key *key,
                      const struct agc_shader_request *request)
{
   struct agc_shader_key actual;
   uint8_t *data;
   size_t size;
   enum agc_shader_file_status status = agc_shader_request_serialize(request, &data, &size);

   if (status != AGC_SHADER_FILE_OK)
      return status;
   agc_shader_key_of_bytes(data, size, &actual);
   char *path = agc_shader_key_path(cache->requests, key, AGC_SHADER_REQUEST_EXTENSION);
   if (memcmp(actual.bytes, key->bytes, AGC_SHADER_KEY_BYTES) != 0)
      status = AGC_SHADER_FILE_KEY_MISMATCH;
   else if (!path)
      status = AGC_SHADER_FILE_OUT_OF_MEMORY;
   else
      status = agc_shader_write_bytes(path, data, size);
   free(path);
   free(data);
   return status;
}
