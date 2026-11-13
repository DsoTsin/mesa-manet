#ifndef AGC_SHADER_ARTIFACT_H
#define AGC_SHADER_ARTIFACT_H

#include "agc_types.h"

#define AGC_SHADER_REQUEST_MAGIC 0x52534741u
#define AGC_SHADER_ARTIFACT_MAGIC 0x41534741u
#define AGC_SHADER_REQUEST_VERSION 1u
#define AGC_SHADER_ARTIFACT_VERSION 1u
#define AGC_SHADER_COMPILER_VERSION 1u

#define AGC_SHADER_FILE_HEADER_BYTES 16
#define AGC_SHADER_KEY_BYTES 32
#define AGC_SHADER_KEY_HEX_SIZE (2 * AGC_SHADER_KEY_BYTES + 1)
#define AGC_SHADER_NGG_REGISTER_COUNT 4

#define AGC_SHADER_ARTIFACT_EXTENSION ".art"
#define AGC_SHADER_REQUEST_EXTENSION ".req"
#define AGC_SHADER_REQUEST_DIRECTORY "requests"

enum agc_shader_file_status {
   AGC_SHADER_FILE_OK,
   AGC_SHADER_FILE_NOT_FOUND,
   AGC_SHADER_FILE_IO_ERROR,
   AGC_SHADER_FILE_OUT_OF_MEMORY,
   AGC_SHADER_FILE_BAD_MAGIC,
   AGC_SHADER_FILE_BAD_VERSION,
   AGC_SHADER_FILE_TRUNCATED,
   AGC_SHADER_FILE_BAD_CHECKSUM,
   AGC_SHADER_FILE_MALFORMED,
   AGC_SHADER_FILE_UNREPRESENTABLE,
   AGC_SHADER_FILE_KEY_MISMATCH,
   AGC_SHADER_FILE_STATUS_COUNT
};

enum agc_shader_isa {
   AGC_SHADER_ISA_GFX1013,
   AGC_SHADER_ISA_GFX103,
   AGC_SHADER_ISA_GFX11,
   AGC_SHADER_ISA_GFX115,
   AGC_SHADER_ISA_GFX12,
   AGC_SHADER_ISA_COUNT
};

enum agc_shader_stage {
   AGC_SHADER_STAGE_COMPUTE,
   AGC_SHADER_STAGE_VERTEX,
   AGC_SHADER_STAGE_FRAGMENT,
   AGC_SHADER_STAGE_COUNT
};

enum agc_shader_ray_unit {
   AGC_SHADER_RAY_UNIT_HOST,
   AGC_SHADER_RAY_UNIT_REFERENCE,
   AGC_SHADER_RAY_UNIT_COUNT
};

enum agc_shader_precision {
   AGC_SHADER_PRECISION_FUSED,
   AGC_SHADER_PRECISION_EXACT,
   AGC_SHADER_PRECISION_COUNT
};

enum agc_shader_vertex_stage {
   AGC_SHADER_VERTEX_STAGE_ES_GS,
   AGC_SHADER_VERTEX_STAGE_LS_HS,
   AGC_SHADER_VERTEX_STAGE_COUNT
};

enum agc_shader_es_input {
   AGC_SHADER_ES_VERTEX,
   AGC_SHADER_ES_DOMAIN,
   AGC_SHADER_ES_INPUT_COUNT
};

enum agc_shader_family {
   AGC_SHADER_FAMILY_SOP1,
   AGC_SHADER_FAMILY_SOP2,
   AGC_SHADER_FAMILY_SOPK,
   AGC_SHADER_FAMILY_SOPC,
   AGC_SHADER_FAMILY_SOPP,
   AGC_SHADER_FAMILY_VOP1,
   AGC_SHADER_FAMILY_VOP2,
   AGC_SHADER_FAMILY_VOP3,
   AGC_SHADER_FAMILY_VOP3P,
   AGC_SHADER_FAMILY_VOPC,
   AGC_SHADER_FAMILY_VINTRP,
   AGC_SHADER_FAMILY_SMEM,
   AGC_SHADER_FAMILY_MUBUF,
   AGC_SHADER_FAMILY_MTBUF,
   AGC_SHADER_FAMILY_FLAT,
   AGC_SHADER_FAMILY_DS,
   AGC_SHADER_FAMILY_MIMG,
   AGC_SHADER_FAMILY_EXP,
   AGC_SHADER_FAMILY_LDSDIR,
   AGC_SHADER_FAMILY_VINTERP,
   AGC_SHADER_FAMILY_COUNT
};

enum agc_shader_error_kind {
   AGC_SHADER_ERROR_NONE,
   AGC_SHADER_ERROR_UNSUPPORTED,
   AGC_SHADER_ERROR_INVALID_OPTIONS,
   AGC_SHADER_ERROR_TARGET_MISMATCH,
   AGC_SHADER_ERROR_EMPTY_PROGRAM,
   AGC_SHADER_ERROR_TRUNCATED,
   AGC_SHADER_ERROR_INVALID_ENCODING,
   AGC_SHADER_ERROR_UNKNOWN_OPCODE,
   AGC_SHADER_ERROR_INVALID_OPERAND,
   AGC_SHADER_ERROR_INVALID_BRANCH,
   AGC_SHADER_ERROR_MISSING_END_PROGRAM,
   AGC_SHADER_ERROR_UNSUPPORTED_INSTRUCTION,
   AGC_SHADER_ERROR_BRANCH_OUT_OF_RANGE,
   AGC_SHADER_ERROR_OUT_OF_REGISTERS,
   AGC_SHADER_ERROR_KIND_COUNT
};

enum agc_shader_resource_kind {
   AGC_SHADER_RESOURCE_BUFFER_DESCRIPTOR,
   AGC_SHADER_RESOURCE_IMAGE_DESCRIPTOR,
   AGC_SHADER_RESOURCE_SAMPLER_DESCRIPTOR,
   AGC_SHADER_RESOURCE_SCALAR_MEMORY_PAIR,
   AGC_SHADER_RESOURCE_KIND_COUNT
};

enum agc_shader_descriptor_access {
   AGC_SHADER_ACCESS_SAMPLED,
   AGC_SHADER_ACCESS_LOADED,
   AGC_SHADER_ACCESS_WRITTEN,
   AGC_SHADER_ACCESS_QUERIED,
   AGC_SHADER_ACCESS_COUNT
};

enum agc_shader_source_kind {
   AGC_SHADER_SOURCE_USER_SGPRS,
   AGC_SHADER_SOURCE_TABLE,
   AGC_SHADER_SOURCE_TABLE_CHOICES,
   AGC_SHADER_SOURCE_DYNAMIC,
   AGC_SHADER_SOURCE_KIND_COUNT
};

enum agc_shader_address_form {
   AGC_SHADER_ADDRESS_POINTER,
   AGC_SHADER_ADDRESS_BUFFER,
   AGC_SHADER_ADDRESS_FORM_COUNT
};

enum agc_shader_scalar_offset_kind {
   AGC_SHADER_SCALAR_OFFSET_BOUNDED,
   AGC_SHADER_SCALAR_OFFSET_USER_SGPR,
   AGC_SHADER_SCALAR_OFFSET_UNKNOWN,
   AGC_SHADER_SCALAR_OFFSET_KIND_COUNT
};

enum agc_shader_register_source {
   AGC_SHADER_REGISTER_CODE_ADDRESS,
   AGC_SHADER_REGISTER_GUEST_USER_DATA_ADDRESS,
   AGC_SHADER_REGISTER_SOURCE_COUNT
};

struct agc_shader_options {
   enum agc_shader_isa target;
   enum agc_shader_stage stage;
   uint8_t wave_size;
   enum agc_shader_ray_unit ray_unit;
};

struct agc_shader_attribute_ring {
   uint64_t address;
   uint32_t bytes;
};

struct agc_shader_interface {
   bool has_user_sgpr_count;
   uint8_t user_sgpr_count;
   bool has_guest_pgm_rsrc1;
   uint32_t guest_pgm_rsrc1;
   bool has_guest_pgm_rsrc2;
   uint32_t guest_pgm_rsrc2;
   uint32_t lds_bytes;
   uint32_t scratch_bytes_per_lane;
   bool has_guest_code_address;
   uint64_t guest_code_address;
   bool has_ps_input_ena;
   uint32_t ps_input_ena;
   bool has_ps_input_addr;
   uint32_t ps_input_addr;
   bool has_ps_col_format;
   uint32_t ps_col_format;
   bool has_ps_z_format;
   uint32_t ps_z_format;
   bool dual_source_blend;
   bool has_param_exports;
   uint8_t param_exports;
   bool has_descriptor_lut;
   uint64_t descriptor_lut;
   bool preconvert_user_descriptors;
   bool has_attribute_ring;
   struct agc_shader_attribute_ring attribute_ring;
   bool has_gds_buffer;
   uint64_t gds_buffer;
   bool has_mip_stats_table;
   uint64_t mip_stats_table;
   bool has_image_reports;
   uint64_t image_reports;
   enum agc_shader_vertex_stage vertex_stage;
   enum agc_shader_es_input es_input;
   bool native_dcc;
};

struct agc_shader_code_region {
   uint64_t address;
   const uint32_t *words;
   uint32_t word_count;
};

struct agc_shader_linkage {
   const struct agc_shader_code_region *regions;
   uint32_t region_count;
   bool has_second_half;
   uint64_t second_half;
};

struct agc_shader_request {
   uint32_t compiler_version;
   struct agc_shader_options options;
   enum agc_shader_precision precision;
   struct agc_shader_interface iface;
   const uint32_t *words;
   uint32_t word_count;
   struct agc_shader_linkage linkage;
   void *storage;
};

struct agc_shader_binding {
   enum agc_shader_resource_kind kind;
   uint8_t base_sgpr;
   uint8_t dword_count;
   bool writable;
};

struct agc_shader_exports {
   uint32_t color_mask;
   uint8_t depth_mask;
   uint8_t position_mask;
   uint32_t param_mask;
   bool primitive;
   bool compressed;
   bool valid_mask;
};

struct agc_shader_register_relocation {
   uint32_t reg;
   enum agc_shader_register_source source;
};

struct agc_shader_ngg {
   uint32_t param_count;
   uint32_t spi_vs_out_config;
   bool has_attribute_ring_relocations;
   uint32_t attribute_ring_relocations[2];
   struct agc_shader_register_relocation registers[AGC_SHADER_NGG_REGISTER_COUNT];
};

struct agc_shader_external_target {
   uint32_t pc;
   bool call;
   bool has_sgpr;
   uint8_t sgpr;
};

struct agc_shader_table_link {
   int64_t offset;
   enum agc_shader_address_form form;
};

struct agc_shader_descriptor_table {
   uint8_t root;
   enum agc_shader_address_form root_form;
   const struct agc_shader_table_link *links;
   uint32_t link_count;
   int64_t offset;
};

struct agc_shader_descriptor_source {
   enum agc_shader_source_kind kind;
   uint8_t first;
   const struct agc_shader_descriptor_table *tables;
   uint32_t table_count;
};

struct agc_shader_scalar_offset {
   enum agc_shader_scalar_offset_kind kind;
   uint32_t bound;
   uint8_t sgpr;
};

struct agc_shader_buffer_access {
   bool scalar;
   bool indexed;
   bool vector_offset;
   uint32_t immediate_offset;
   struct agc_shader_scalar_offset scalar_offset;
};

struct agc_shader_descriptor_use {
   uint32_t pc;
   enum agc_shader_resource_kind kind;
   uint8_t base_sgpr;
   uint8_t dword_count;
   enum agc_shader_descriptor_access access;
   struct agc_shader_descriptor_source source;
   bool has_buffer_access;
   struct agc_shader_buffer_access buffer_access;
};

struct agc_shader_buffer_fill {
   uint8_t buffer;
   uint8_t value;
};

struct agc_shader_error {
   enum agc_shader_error_kind kind;
   uint32_t pc;
   uint32_t word;
   uint32_t branch_target;
   uint16_t opcode;
   uint16_t vgprs;
   enum agc_shader_family family;
   enum agc_shader_isa target;
   enum agc_shader_isa compiler;
   enum agc_shader_isa requested;
   const char *reason;
};

struct agc_shader_artifact {
   struct agc_shader_error error;
   const uint32_t *code;
   uint32_t code_count;
   struct agc_shader_options options;
   enum agc_shader_precision precision;
   uint32_t pgm_rsrc1;
   uint32_t pgm_rsrc2;
   uint16_t num_sgprs;
   uint16_t num_vgprs;
   uint32_t lds_bytes;
   uint32_t scratch_bytes_per_lane;
   uint8_t user_sgpr_count;
   uint32_t program_words;
   uint32_t guest_words;
   uint64_t guest_hash;
   const struct agc_shader_binding *bindings;
   uint32_t binding_count;
   struct agc_shader_exports exports;
   bool has_user_data_tail_sgpr;
   uint8_t user_data_tail_sgpr;
   bool uses_pc_relative;
   const struct agc_shader_binding *preconverted_descriptors;
   uint32_t preconverted_descriptor_count;
   bool has_spi_shader_col_format;
   uint32_t spi_shader_col_format;
   bool has_ngg;
   struct agc_shader_ngg ngg;
   const struct agc_shader_external_target *external_targets;
   uint32_t external_target_count;
   const struct agc_shader_descriptor_use *descriptor_uses;
   uint32_t descriptor_use_count;
   const struct agc_shader_descriptor_use *pointer_uses;
   uint32_t pointer_use_count;
   bool has_buffer_fill;
   struct agc_shader_buffer_fill buffer_fill;
   void *storage;
};

struct agc_shader_key {
   uint8_t bytes[AGC_SHADER_KEY_BYTES];
};

struct agc_shader_cache;

const char *agc_shader_file_status_name(enum agc_shader_file_status status);
const char *agc_shader_error_kind_name(enum agc_shader_error_kind kind);

enum agc_shader_file_status agc_shader_request_serialize(const struct agc_shader_request *request,
                                                         uint8_t **data, size_t *size);
enum agc_shader_file_status agc_shader_request_deserialize(const uint8_t *data, size_t size,
                                                           struct agc_shader_request *request);
enum agc_shader_file_status agc_shader_request_write_file(const struct agc_shader_request *request,
                                                          const char *path);
enum agc_shader_file_status agc_shader_request_read_file(const char *path, struct agc_shader_request *request);
void agc_shader_request_free(struct agc_shader_request *request);

enum agc_shader_file_status agc_shader_artifact_serialize(const struct agc_shader_artifact *artifact,
                                                          uint8_t **data, size_t *size);
enum agc_shader_file_status agc_shader_artifact_deserialize(const uint8_t *data, size_t size,
                                                            struct agc_shader_artifact *artifact);
enum agc_shader_file_status agc_shader_artifact_write_file(const struct agc_shader_artifact *artifact,
                                                           const char *path);
enum agc_shader_file_status agc_shader_artifact_read_file(const char *path, struct agc_shader_artifact *artifact);
void agc_shader_artifact_free(struct agc_shader_artifact *artifact);

enum agc_shader_file_status agc_shader_request_key(const struct agc_shader_request *request,
                                                   struct agc_shader_key *key);
void agc_shader_key_of_bytes(const uint8_t *data, size_t size, struct agc_shader_key *key);
void agc_shader_key_hex(const struct agc_shader_key *key, char hex[AGC_SHADER_KEY_HEX_SIZE]);
bool agc_shader_key_from_hex(const char *hex, struct agc_shader_key *key);

enum agc_shader_file_status agc_shader_cache_open(const char *directory, struct agc_shader_cache **cache);
void agc_shader_cache_close(struct agc_shader_cache *cache);
enum agc_shader_file_status agc_shader_cache_lookup(const struct agc_shader_cache *cache,
                                                    const struct agc_shader_key *key,
                                                    struct agc_shader_artifact *artifact);
enum agc_shader_file_status agc_shader_cache_store(const struct agc_shader_cache *cache,
                                                   const struct agc_shader_key *key,
                                                   const struct agc_shader_artifact *artifact);
enum agc_shader_file_status agc_shader_cache_miss(const struct agc_shader_cache *cache,
                                                  const struct agc_shader_key *key,
                                                  const struct agc_shader_request *request);

#endif
