#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "agc_shader_artifact.h"
#include "util/detect_os.h"
#include "test.h"

#if DETECT_OS_WINDOWS
#include <direct.h>
#include <process.h>
#define test_rmdir _rmdir
#define test_getpid _getpid
#else
#include <unistd.h>
#define test_rmdir rmdir
#define test_getpid getpid
#endif

#define OK AGC_SHADER_FILE_OK
#define COUNT_OF(a) (sizeof(a) / sizeof((a)[0]))

enum file_kind {
   FILE_REQUEST,
   FILE_ARTIFACT
};

static uint32_t
rd32(const uint8_t *p)
{
   return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint64_t
rd64(const uint8_t *p)
{
   return (uint64_t)rd32(p) | (uint64_t)rd32(p + 4) << 32;
}

static void
wr32(uint8_t *p, uint32_t value)
{
   p[0] = (uint8_t)value;
   p[1] = (uint8_t)(value >> 8);
   p[2] = (uint8_t)(value >> 16);
   p[3] = (uint8_t)(value >> 24);
}

static uint32_t
reference_crc32(const uint8_t *data, size_t size)
{
   uint32_t crc = 0xffffffffu;
   for (size_t i = 0; i < size; i++) {
      crc ^= data[i];
      for (unsigned bit = 0; bit < 8; bit++)
         crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1)));
   }
   return ~crc;
}

static void
reseal(uint8_t *data, size_t size)
{
   wr32(data + 8, (uint32_t)size);
   wr32(data + 12, reference_crc32(data + 16, size - 16));
}

static const uint32_t shader_words[] = {0xf4080100, 0xfa000000, 0x34020082, 0xbf8cc07f, 0xbf810000};
static const uint32_t region_a_words[] = {0x7e020280, 0xbe80201e};
static const uint32_t region_b_words[] = {0xbf810000};
static const struct agc_shader_code_region shader_regions[] = {
   {UINT64_C(0x900000100), region_a_words, 2},
   {UINT64_C(0x900000200), NULL, 0},
   {UINT64_C(0xa00000300), region_b_words, 1},
};

static struct agc_shader_request
full_request(void)
{
   struct agc_shader_request r;
   memset(&r, 0, sizeof(r));
   r.compiler_version = AGC_SHADER_COMPILER_VERSION;
   r.options.target = AGC_SHADER_ISA_GFX115;
   r.options.stage = AGC_SHADER_STAGE_FRAGMENT;
   r.options.wave_size = 64;
   r.options.ray_unit = AGC_SHADER_RAY_UNIT_REFERENCE;
   r.precision = AGC_SHADER_PRECISION_EXACT;
   r.iface.has_user_sgpr_count = true;
   r.iface.user_sgpr_count = 0x21;
   r.iface.has_guest_pgm_rsrc1 = true;
   r.iface.guest_pgm_rsrc1 = 0x11111111;
   r.iface.has_guest_pgm_rsrc2 = true;
   r.iface.guest_pgm_rsrc2 = 0x22222222;
   r.iface.lds_bytes = 0x33333333;
   r.iface.scratch_bytes_per_lane = 0x44444444;
   r.iface.has_guest_code_address = true;
   r.iface.guest_code_address = UINT64_C(0x5555555566666666);
   r.iface.has_ps_input_ena = true;
   r.iface.ps_input_ena = 0x77777777;
   r.iface.has_ps_input_addr = true;
   r.iface.ps_input_addr = 0x88888888;
   r.iface.has_ps_col_format = true;
   r.iface.ps_col_format = 0x99999999;
   r.iface.has_ps_z_format = true;
   r.iface.ps_z_format = 0xaaaaaaaa;
   r.iface.dual_source_blend = true;
   r.iface.has_param_exports = true;
   r.iface.param_exports = 0xbb;
   r.iface.has_descriptor_lut = true;
   r.iface.descriptor_lut = UINT64_C(0xccccccccdddddddd);
   r.iface.preconvert_user_descriptors = true;
   r.iface.has_attribute_ring = true;
   r.iface.attribute_ring.address = UINT64_C(0xeeeeeeeeffffffff);
   r.iface.attribute_ring.bytes = 0x12121212;
   r.iface.has_gds_buffer = true;
   r.iface.gds_buffer = UINT64_C(0x1313131314141414);
   r.iface.has_mip_stats_table = true;
   r.iface.mip_stats_table = UINT64_C(0x1515151516161616);
   r.iface.has_image_reports = true;
   r.iface.image_reports = UINT64_C(0x1717171718181818);
   r.iface.vertex_stage = AGC_SHADER_VERTEX_STAGE_LS_HS;
   r.iface.native_dcc = true;
   r.words = shader_words;
   r.word_count = COUNT_OF(shader_words);
   r.linkage.regions = shader_regions;
   r.linkage.region_count = COUNT_OF(shader_regions);
   r.linkage.has_second_half = true;
   r.linkage.second_half = UINT64_C(0x1919191920202020);
   return r;
}

static struct agc_shader_request
minimal_request(void)
{
   struct agc_shader_request r;
   memset(&r, 0, sizeof(r));
   r.compiler_version = AGC_SHADER_COMPILER_VERSION;
   r.options.target = AGC_SHADER_ISA_GFX103;
   r.options.stage = AGC_SHADER_STAGE_COMPUTE;
   r.options.wave_size = 32;
   r.words = shader_words;
   r.word_count = COUNT_OF(shader_words);
   return r;
}

static bool
words_equal(const uint32_t *a, uint32_t a_count, const uint32_t *b, uint32_t b_count)
{
   return a_count == b_count && (!a_count || memcmp(a, b, (size_t)a_count * sizeof(uint32_t)) == 0);
}

static bool
request_equal(const struct agc_shader_request *a, const struct agc_shader_request *b)
{
   const struct agc_shader_interface *x = &a->iface;
   const struct agc_shader_interface *y = &b->iface;
   bool same = a->compiler_version == b->compiler_version && a->options.target == b->options.target &&
               a->options.stage == b->options.stage && a->options.wave_size == b->options.wave_size &&
               a->options.ray_unit == b->options.ray_unit && a->precision == b->precision;

   same = same && x->has_user_sgpr_count == y->has_user_sgpr_count &&
          (!x->has_user_sgpr_count || x->user_sgpr_count == y->user_sgpr_count);
   same = same && x->has_guest_pgm_rsrc1 == y->has_guest_pgm_rsrc1 &&
          (!x->has_guest_pgm_rsrc1 || x->guest_pgm_rsrc1 == y->guest_pgm_rsrc1);
   same = same && x->has_guest_pgm_rsrc2 == y->has_guest_pgm_rsrc2 &&
          (!x->has_guest_pgm_rsrc2 || x->guest_pgm_rsrc2 == y->guest_pgm_rsrc2);
   same = same && x->lds_bytes == y->lds_bytes && x->scratch_bytes_per_lane == y->scratch_bytes_per_lane;
   same = same && x->has_guest_code_address == y->has_guest_code_address &&
          (!x->has_guest_code_address || x->guest_code_address == y->guest_code_address);
   same = same && x->has_ps_input_ena == y->has_ps_input_ena &&
          (!x->has_ps_input_ena || x->ps_input_ena == y->ps_input_ena);
   same = same && x->has_ps_input_addr == y->has_ps_input_addr &&
          (!x->has_ps_input_addr || x->ps_input_addr == y->ps_input_addr);
   same = same && x->has_ps_col_format == y->has_ps_col_format &&
          (!x->has_ps_col_format || x->ps_col_format == y->ps_col_format);
   same = same && x->has_ps_z_format == y->has_ps_z_format &&
          (!x->has_ps_z_format || x->ps_z_format == y->ps_z_format);
   same = same && x->dual_source_blend == y->dual_source_blend;
   same = same && x->has_param_exports == y->has_param_exports &&
          (!x->has_param_exports || x->param_exports == y->param_exports);
   same = same && x->has_descriptor_lut == y->has_descriptor_lut &&
          (!x->has_descriptor_lut || x->descriptor_lut == y->descriptor_lut);
   same = same && x->preconvert_user_descriptors == y->preconvert_user_descriptors;
   same = same && x->has_attribute_ring == y->has_attribute_ring &&
          (!x->has_attribute_ring || (x->attribute_ring.address == y->attribute_ring.address &&
                                      x->attribute_ring.bytes == y->attribute_ring.bytes));
   same = same && x->has_gds_buffer == y->has_gds_buffer && (!x->has_gds_buffer || x->gds_buffer == y->gds_buffer);
   same = same && x->has_mip_stats_table == y->has_mip_stats_table &&
          (!x->has_mip_stats_table || x->mip_stats_table == y->mip_stats_table);
   same = same && x->has_image_reports == y->has_image_reports &&
          (!x->has_image_reports || x->image_reports == y->image_reports);
   same = same && x->vertex_stage == y->vertex_stage;
   same = same && x->es_input == y->es_input;
   same = same && x->native_dcc == y->native_dcc;
   same = same && words_equal(a->words, a->word_count, b->words, b->word_count);
   same = same && a->linkage.region_count == b->linkage.region_count;
   same = same && a->linkage.has_second_half == b->linkage.has_second_half &&
          (!a->linkage.has_second_half || a->linkage.second_half == b->linkage.second_half);
   for (uint32_t i = 0; same && i < a->linkage.region_count; i++) {
      const struct agc_shader_code_region *p = &a->linkage.regions[i];
      const struct agc_shader_code_region *q = &b->linkage.regions[i];
      same = p->address == q->address && words_equal(p->words, p->word_count, q->words, q->word_count);
   }
   return same;
}

static const uint32_t code_words[] = {0xbe8300ff, 0x80000001, 0xbe840080, 0xbf8b0000,
                                      0x81058103, 0xbfb00000, 0xbf9f0000, 0xbf9f0000};
static const struct agc_shader_binding code_bindings[] = {
   {AGC_SHADER_RESOURCE_SCALAR_MEMORY_PAIR, 0, 2, false},
   {AGC_SHADER_RESOURCE_BUFFER_DESCRIPTOR, 4, 4, true},
   {AGC_SHADER_RESOURCE_IMAGE_DESCRIPTOR, 8, 8, false},
};
static const struct agc_shader_binding code_preconverted[] = {
   {AGC_SHADER_RESOURCE_SAMPLER_DESCRIPTOR, 16, 4, false},
};
static const struct agc_shader_external_target code_externals[] = {
   {0x40, true, true, 6},
   {0x80, false, false, 0},
};
static const struct agc_shader_table_link links_a[] = {
   {0x10, AGC_SHADER_ADDRESS_POINTER},
   {-0x20, AGC_SHADER_ADDRESS_BUFFER},
};
static const struct agc_shader_table_link links_b[] = {
   {INT64_MIN, AGC_SHADER_ADDRESS_BUFFER},
};
static const struct agc_shader_descriptor_table table_single[] = {
   {.root = 2, .root_form = AGC_SHADER_ADDRESS_BUFFER, .links = links_a, .link_count = 2, .offset = 0x30},
};
static const struct agc_shader_descriptor_table table_choices[] = {
   {.root = 4, .root_form = AGC_SHADER_ADDRESS_POINTER, .links = NULL, .link_count = 0, .offset = -8},
   {.root = 5, .root_form = AGC_SHADER_ADDRESS_POINTER, .links = links_b, .link_count = 1, .offset = INT64_MAX},
   {.root = 6, .root_form = AGC_SHADER_ADDRESS_BUFFER, .links = links_a, .link_count = 2, .offset = 0},
};
static const struct agc_shader_descriptor_use code_uses[] = {
   {
      .pc = 0x10,
      .kind = AGC_SHADER_RESOURCE_IMAGE_DESCRIPTOR,
      .base_sgpr = 8,
      .dword_count = 8,
      .access = AGC_SHADER_ACCESS_SAMPLED,
      .source = {.kind = AGC_SHADER_SOURCE_USER_SGPRS, .first = 8},
   },
   {
      .pc = 0x14,
      .kind = AGC_SHADER_RESOURCE_BUFFER_DESCRIPTOR,
      .base_sgpr = 4,
      .dword_count = 4,
      .access = AGC_SHADER_ACCESS_WRITTEN,
      .source = {.kind = AGC_SHADER_SOURCE_TABLE, .tables = table_single, .table_count = 1},
      .has_buffer_access = true,
      .buffer_access = {.indexed = true,
                        .vector_offset = true,
                        .immediate_offset = 0x123,
                        .scalar_offset = {.kind = AGC_SHADER_SCALAR_OFFSET_BOUNDED, .bound = 0xfff}},
   },
   {
      .pc = 0x18,
      .kind = AGC_SHADER_RESOURCE_SAMPLER_DESCRIPTOR,
      .base_sgpr = 16,
      .dword_count = 4,
      .access = AGC_SHADER_ACCESS_SAMPLED,
      .source = {.kind = AGC_SHADER_SOURCE_TABLE_CHOICES, .tables = table_choices, .table_count = 3},
   },
   {
      .pc = 0x1c,
      .kind = AGC_SHADER_RESOURCE_BUFFER_DESCRIPTOR,
      .base_sgpr = 20,
      .dword_count = 4,
      .access = AGC_SHADER_ACCESS_LOADED,
      .source = {.kind = AGC_SHADER_SOURCE_TABLE_CHOICES},
      .has_buffer_access = true,
      .buffer_access = {.scalar = true,
                        .scalar_offset = {.kind = AGC_SHADER_SCALAR_OFFSET_USER_SGPR, .sgpr = 9}},
   },
   {
      .pc = 0x20,
      .kind = AGC_SHADER_RESOURCE_IMAGE_DESCRIPTOR,
      .base_sgpr = 24,
      .dword_count = 8,
      .access = AGC_SHADER_ACCESS_QUERIED,
      .source = {.kind = AGC_SHADER_SOURCE_DYNAMIC},
      .has_buffer_access = true,
      .buffer_access = {.scalar_offset = {.kind = AGC_SHADER_SCALAR_OFFSET_UNKNOWN}},
   },
};
static const struct agc_shader_descriptor_use code_pointer_uses[] = {
   {
      .pc = 0x24,
      .kind = AGC_SHADER_RESOURCE_SCALAR_MEMORY_PAIR,
      .base_sgpr = 0,
      .dword_count = 2,
      .access = AGC_SHADER_ACCESS_LOADED,
      .source = {.kind = AGC_SHADER_SOURCE_USER_SGPRS, .first = 0},
      .has_buffer_access = true,
      .buffer_access = {.scalar = true,
                        .immediate_offset = 0x40,
                        .scalar_offset = {.kind = AGC_SHADER_SCALAR_OFFSET_BOUNDED, .bound = 0}},
   },
};

static struct agc_shader_artifact
full_artifact(void)
{
   struct agc_shader_artifact a;
   memset(&a, 0, sizeof(a));
   a.code = code_words;
   a.code_count = COUNT_OF(code_words);
   a.options.target = AGC_SHADER_ISA_GFX12;
   a.options.stage = AGC_SHADER_STAGE_VERTEX;
   a.options.wave_size = 64;
   a.options.ray_unit = AGC_SHADER_RAY_UNIT_REFERENCE;
   a.precision = AGC_SHADER_PRECISION_EXACT;
   a.pgm_rsrc1 = 0x11111111;
   a.pgm_rsrc2 = 0x22222222;
   a.num_sgprs = 0x3333;
   a.num_vgprs = 0x4444;
   a.lds_bytes = 0x55555555;
   a.scratch_bytes_per_lane = 0x66666666;
   a.user_sgpr_count = 0x77;
   a.program_words = 6;
   a.guest_words = 0x88888888;
   a.guest_hash = UINT64_C(0x99999999aaaaaaaa);
   a.bindings = code_bindings;
   a.binding_count = COUNT_OF(code_bindings);
   a.exports.color_mask = 0xbbbbbbbb;
   a.exports.depth_mask = 0xcc;
   a.exports.position_mask = 0xdd;
   a.exports.param_mask = 0xeeeeeeee;
   a.exports.primitive = true;
   a.exports.compressed = true;
   a.exports.valid_mask = true;
   a.has_user_data_tail_sgpr = true;
   a.user_data_tail_sgpr = 0xf1;
   a.uses_pc_relative = true;
   a.preconverted_descriptors = code_preconverted;
   a.preconverted_descriptor_count = COUNT_OF(code_preconverted);
   a.has_spi_shader_col_format = true;
   a.spi_shader_col_format = 0x12121212;
   a.has_ngg = true;
   a.ngg.param_count = 0x13131313;
   a.ngg.spi_vs_out_config = 0x14141414;
   a.ngg.has_attribute_ring_relocations = true;
   a.ngg.attribute_ring_relocations[0] = 1;
   a.ngg.attribute_ring_relocations[1] = 7;
   a.ngg.registers[0].reg = 0xb320;
   a.ngg.registers[0].source = AGC_SHADER_REGISTER_CODE_ADDRESS;
   a.ngg.registers[1].reg = 0xb324;
   a.ngg.registers[1].source = AGC_SHADER_REGISTER_CODE_ADDRESS;
   a.ngg.registers[2].reg = 0xb220;
   a.ngg.registers[2].source = AGC_SHADER_REGISTER_GUEST_USER_DATA_ADDRESS;
   a.ngg.registers[3].reg = 0xb224;
   a.ngg.registers[3].source = AGC_SHADER_REGISTER_GUEST_USER_DATA_ADDRESS;
   a.external_targets = code_externals;
   a.external_target_count = COUNT_OF(code_externals);
   a.descriptor_uses = code_uses;
   a.descriptor_use_count = COUNT_OF(code_uses);
   a.pointer_uses = code_pointer_uses;
   a.pointer_use_count = COUNT_OF(code_pointer_uses);
   a.has_buffer_fill = true;
   a.buffer_fill.buffer = 0x15;
   a.buffer_fill.value = 0x16;
   return a;
}

static struct agc_shader_artifact
minimal_artifact(void)
{
   struct agc_shader_artifact a;
   memset(&a, 0, sizeof(a));
   a.options.target = AGC_SHADER_ISA_GFX103;
   a.options.stage = AGC_SHADER_STAGE_COMPUTE;
   a.options.wave_size = 32;
   return a;
}

static struct agc_shader_artifact
error_artifact(enum agc_shader_error_kind kind, const char *reason)
{
   struct agc_shader_artifact a = full_artifact();
   a.error.kind = kind;
   a.error.pc = 0x1a1a1a1a;
   a.error.word = 0x1b1b1b1b;
   a.error.branch_target = 0x1c1c1c1c;
   a.error.opcode = 0x1d1d;
   a.error.vgprs = 0x1e1e;
   a.error.family = AGC_SHADER_FAMILY_VINTERP;
   a.error.target = AGC_SHADER_ISA_GFX12;
   a.error.compiler = AGC_SHADER_ISA_GFX11;
   a.error.requested = AGC_SHADER_ISA_GFX115;
   a.error.reason = reason;
   return a;
}

static struct agc_shader_artifact
unsupported_artifact(void)
{
   struct agc_shader_artifact a;
   memset(&a, 0, sizeof(a));
   a.error.kind = AGC_SHADER_ERROR_UNSUPPORTED_INSTRUCTION;
   a.error.pc = 0x40;
   a.error.family = AGC_SHADER_FAMILY_MIMG;
   a.error.opcode = 0xe6;
   a.error.target = AGC_SHADER_ISA_GFX12;
   a.error.reason = "no gfx12 form";
   return a;
}

static bool
table_equal(const struct agc_shader_descriptor_table *a, const struct agc_shader_descriptor_table *b)
{
   bool same = a->root == b->root && a->root_form == b->root_form && a->link_count == b->link_count &&
               a->offset == b->offset;
   for (uint32_t i = 0; same && i < a->link_count; i++)
      same = a->links[i].offset == b->links[i].offset && a->links[i].form == b->links[i].form;
   return same;
}

static bool
use_equal(const struct agc_shader_descriptor_use *a, const struct agc_shader_descriptor_use *b)
{
   const struct agc_shader_buffer_access *x = &a->buffer_access;
   const struct agc_shader_buffer_access *y = &b->buffer_access;
   bool same = a->pc == b->pc && a->kind == b->kind && a->base_sgpr == b->base_sgpr &&
               a->dword_count == b->dword_count && a->access == b->access && a->source.kind == b->source.kind &&
               a->has_buffer_access == b->has_buffer_access;
   uint32_t tables = 0;

   if (!same)
      return false;
   if (a->source.kind == AGC_SHADER_SOURCE_USER_SGPRS)
      same = a->source.first == b->source.first;
   if (a->source.kind == AGC_SHADER_SOURCE_TABLE || a->source.kind == AGC_SHADER_SOURCE_TABLE_CHOICES) {
      same = same && a->source.table_count == b->source.table_count;
      tables = a->source.table_count;
   }
   for (uint32_t i = 0; same && i < tables; i++)
      same = table_equal(&a->source.tables[i], &b->source.tables[i]);
   if (same && a->has_buffer_access) {
      same = x->scalar == y->scalar && x->indexed == y->indexed && x->vector_offset == y->vector_offset &&
             x->immediate_offset == y->immediate_offset && x->scalar_offset.kind == y->scalar_offset.kind;
      if (same && x->scalar_offset.kind == AGC_SHADER_SCALAR_OFFSET_BOUNDED)
         same = x->scalar_offset.bound == y->scalar_offset.bound;
      if (same && x->scalar_offset.kind == AGC_SHADER_SCALAR_OFFSET_USER_SGPR)
         same = x->scalar_offset.sgpr == y->scalar_offset.sgpr;
   }
   return same;
}

static bool
bindings_equal(const struct agc_shader_binding *a, uint32_t a_count, const struct agc_shader_binding *b,
               uint32_t b_count)
{
   bool same = a_count == b_count;
   for (uint32_t i = 0; same && i < a_count; i++)
      same = a[i].kind == b[i].kind && a[i].base_sgpr == b[i].base_sgpr && a[i].dword_count == b[i].dword_count &&
             a[i].writable == b[i].writable;
   return same;
}

static bool
uses_equal(const struct agc_shader_descriptor_use *a, uint32_t a_count, const struct agc_shader_descriptor_use *b,
           uint32_t b_count)
{
   bool same = a_count == b_count;
   for (uint32_t i = 0; same && i < a_count; i++)
      same = use_equal(&a[i], &b[i]);
   return same;
}

static bool
error_equal(const struct agc_shader_error *a, const struct agc_shader_error *b)
{
   bool same = a->kind == b->kind && a->pc == b->pc && a->word == b->word &&
               a->branch_target == b->branch_target && a->opcode == b->opcode && a->vgprs == b->vgprs &&
               a->family == b->family && a->target == b->target && a->compiler == b->compiler &&
               a->requested == b->requested && !a->reason == !b->reason;
   return same && (!a->reason || strcmp(a->reason, b->reason) == 0);
}

static bool
artifact_equal(const struct agc_shader_artifact *a, const struct agc_shader_artifact *b)
{
   const struct agc_shader_ngg *x = &a->ngg;
   const struct agc_shader_ngg *y = &b->ngg;
   bool same;

   if (a->error.kind != AGC_SHADER_ERROR_NONE || b->error.kind != AGC_SHADER_ERROR_NONE)
      return error_equal(&a->error, &b->error);
   same = words_equal(a->code, a->code_count, b->code, b->code_count) && a->options.target == b->options.target &&
          a->options.stage == b->options.stage && a->options.wave_size == b->options.wave_size &&
          a->options.ray_unit == b->options.ray_unit && a->precision == b->precision &&
          a->pgm_rsrc1 == b->pgm_rsrc1 && a->pgm_rsrc2 == b->pgm_rsrc2 && a->num_sgprs == b->num_sgprs &&
          a->num_vgprs == b->num_vgprs && a->lds_bytes == b->lds_bytes &&
          a->scratch_bytes_per_lane == b->scratch_bytes_per_lane && a->user_sgpr_count == b->user_sgpr_count &&
          a->program_words == b->program_words && a->guest_words == b->guest_words &&
          a->guest_hash == b->guest_hash;
   same = same && bindings_equal(a->bindings, a->binding_count, b->bindings, b->binding_count);
   same = same && a->exports.color_mask == b->exports.color_mask && a->exports.depth_mask == b->exports.depth_mask &&
          a->exports.position_mask == b->exports.position_mask && a->exports.param_mask == b->exports.param_mask &&
          a->exports.primitive == b->exports.primitive && a->exports.compressed == b->exports.compressed &&
          a->exports.valid_mask == b->exports.valid_mask;
   same = same && a->has_user_data_tail_sgpr == b->has_user_data_tail_sgpr &&
          (!a->has_user_data_tail_sgpr || a->user_data_tail_sgpr == b->user_data_tail_sgpr);
   same = same && a->uses_pc_relative == b->uses_pc_relative;
   same = same && bindings_equal(a->preconverted_descriptors, a->preconverted_descriptor_count,
                                 b->preconverted_descriptors, b->preconverted_descriptor_count);
   same = same && a->has_spi_shader_col_format == b->has_spi_shader_col_format &&
          (!a->has_spi_shader_col_format || a->spi_shader_col_format == b->spi_shader_col_format);
   same = same && a->has_ngg == b->has_ngg;
   if (same && a->has_ngg) {
      same = x->param_count == y->param_count && x->spi_vs_out_config == y->spi_vs_out_config &&
             x->has_attribute_ring_relocations == y->has_attribute_ring_relocations;
      if (same && x->has_attribute_ring_relocations)
         same = x->attribute_ring_relocations[0] == y->attribute_ring_relocations[0] &&
                x->attribute_ring_relocations[1] == y->attribute_ring_relocations[1];
      for (unsigned i = 0; same && i < AGC_SHADER_NGG_REGISTER_COUNT; i++)
         same = x->registers[i].reg == y->registers[i].reg && x->registers[i].source == y->registers[i].source;
   }
   same = same && a->external_target_count == b->external_target_count;
   for (uint32_t i = 0; same && i < a->external_target_count; i++) {
      const struct agc_shader_external_target *p = &a->external_targets[i];
      const struct agc_shader_external_target *q = &b->external_targets[i];
      same = p->pc == q->pc && p->call == q->call && p->has_sgpr == q->has_sgpr &&
             (!p->has_sgpr || p->sgpr == q->sgpr);
   }
   same = same && uses_equal(a->descriptor_uses, a->descriptor_use_count, b->descriptor_uses, b->descriptor_use_count);
   same = same && uses_equal(a->pointer_uses, a->pointer_use_count, b->pointer_uses, b->pointer_use_count);
   same = same && a->has_buffer_fill == b->has_buffer_fill &&
          (!a->has_buffer_fill ||
           (a->buffer_fill.buffer == b->buffer_fill.buffer && a->buffer_fill.value == b->buffer_fill.value));
   return same;
}

static enum agc_shader_file_status
parse(enum file_kind kind, const uint8_t *data, size_t size, uint8_t **again, size_t *again_size)
{
   uint8_t *copy = malloc(size ? size : 1);
   enum agc_shader_file_status status;

   *again = NULL;
   *again_size = 0;
   if (size)
      memcpy(copy, data, size);
   if (kind == FILE_REQUEST) {
      struct agc_shader_request request;
      status = agc_shader_request_deserialize(copy, size, &request);
      if (status == OK)
         CHECK_EQ(agc_shader_request_serialize(&request, again, again_size), OK);
      else
         CHECK(request.storage == NULL && request.words == NULL && request.linkage.regions == NULL);
      agc_shader_request_free(&request);
   } else {
      struct agc_shader_artifact artifact;
      status = agc_shader_artifact_deserialize(copy, size, &artifact);
      if (status == OK)
         CHECK_EQ(agc_shader_artifact_serialize(&artifact, again, again_size), OK);
      else
         CHECK(artifact.storage == NULL && artifact.code == NULL && artifact.error.reason == NULL);
      agc_shader_artifact_free(&artifact);
   }
   free(copy);
   return status;
}

static enum agc_shader_file_status
parse_status(enum file_kind kind, const uint8_t *data, size_t size)
{
   uint8_t *again;
   size_t again_size;
   enum agc_shader_file_status status = parse(kind, data, size, &again, &again_size);
   free(again);
   return status;
}

static void
check_rejections(enum file_kind kind, const uint8_t *valid, size_t size, const size_t *counts, size_t count_fields)
{
   static const uint8_t masks[] = {0x01, 0x02, 0x04, 0x80, 0xff};
   static const uint32_t bad_counts[] = {0xffffffffu, 0x80000000u, 0x40000000u, 0x00010000u};
   uint8_t *work = malloc(size + 4);
   uint8_t *again;
   size_t again_size;
   unsigned accepted = 0;

   CHECK_EQ(parse(kind, valid, size, &again, &again_size), OK);
   CHECK(again_size == size && again && memcmp(again, valid, size) == 0);
   free(again);

   for (size_t length = 0; length < size; length++)
      CHECK_EQ(parse_status(kind, valid, length), AGC_SHADER_FILE_TRUNCATED);

   memcpy(work, valid, size);
   memset(work + size, 0, 4);
   CHECK_EQ(parse_status(kind, work, size + 1), AGC_SHADER_FILE_MALFORMED);
   CHECK_EQ(parse_status(kind, work, size + 4), AGC_SHADER_FILE_MALFORMED);
   reseal(work, size + 4);
   CHECK_EQ(parse_status(kind, work, size + 4), AGC_SHADER_FILE_MALFORMED);

   for (size_t length = AGC_SHADER_FILE_HEADER_BYTES; length < size; length++) {
      memcpy(work, valid, length);
      reseal(work, length);
      enum agc_shader_file_status status = parse_status(kind, work, length);
      CHECK(status == AGC_SHADER_FILE_TRUNCATED || status == AGC_SHADER_FILE_MALFORMED);
   }

   for (size_t at = 0; at < size; at++) {
      for (size_t m = 0; m < COUNT_OF(masks); m++) {
         memcpy(work, valid, size);
         work[at] ^= masks[m];
         enum agc_shader_file_status status = parse_status(kind, work, size);
         if (at < 4)
            CHECK_EQ(status, AGC_SHADER_FILE_BAD_MAGIC);
         else if (at < 8)
            CHECK_EQ(status, AGC_SHADER_FILE_BAD_VERSION);
         else if (at < 12)
            CHECK(status == AGC_SHADER_FILE_TRUNCATED || status == AGC_SHADER_FILE_MALFORMED);
         else
            CHECK_EQ(status, AGC_SHADER_FILE_BAD_CHECKSUM);
         if (at < AGC_SHADER_FILE_HEADER_BYTES)
            continue;

         reseal(work, size);
         status = parse(kind, work, size, &again, &again_size);
         if (status == OK) {
            accepted++;
            CHECK(again_size == size && again && memcmp(again, work, size) == 0);
         } else {
            CHECK(status == AGC_SHADER_FILE_TRUNCATED || status == AGC_SHADER_FILE_MALFORMED);
         }
         free(again);
      }
   }
   CHECK(accepted > 0 || size <= 64);

   for (size_t i = 0; i < count_fields; i++) {
      uint32_t count = rd32(valid + counts[i]);
      for (size_t k = 0; k < COUNT_OF(bad_counts) + 2; k++) {
         uint32_t bad = k < COUNT_OF(bad_counts) ? bad_counts[k] : k == COUNT_OF(bad_counts) ? count + 1 : count - 1;
         if (bad == count || (count == 0 && bad == count - 1))
            continue;
         memcpy(work, valid, size);
         wr32(work + counts[i], bad);
         reseal(work, size);
         enum agc_shader_file_status status = parse_status(kind, work, size);
         CHECK(status == AGC_SHADER_FILE_TRUNCATED || status == AGC_SHADER_FILE_MALFORMED);
      }
   }
   free(work);
}

static void
test_names(void)
{
   CHECK_EQ(reference_crc32((const uint8_t *)"123456789", 9), 0xcbf43926);
   CHECK(strcmp(agc_shader_file_status_name(AGC_SHADER_FILE_OK), "ok") == 0);
   CHECK(strcmp(agc_shader_file_status_name(AGC_SHADER_FILE_KEY_MISMATCH), "key mismatch") == 0);
   CHECK(strcmp(agc_shader_file_status_name((enum agc_shader_file_status)99), "unknown") == 0);
   for (unsigned i = 0; i < AGC_SHADER_FILE_STATUS_COUNT; i++)
      CHECK(agc_shader_file_status_name((enum agc_shader_file_status)i)[0] != 0);
   CHECK(strcmp(agc_shader_error_kind_name(AGC_SHADER_ERROR_NONE), "None") == 0);
   CHECK(strcmp(agc_shader_error_kind_name(AGC_SHADER_ERROR_OUT_OF_REGISTERS), "OutOfRegisters") == 0);
   CHECK(strcmp(agc_shader_error_kind_name((enum agc_shader_error_kind)99), "Unknown") == 0);
   for (unsigned i = 0; i < AGC_SHADER_ERROR_KIND_COUNT; i++)
      CHECK(agc_shader_error_kind_name((enum agc_shader_error_kind)i)[0] != 0);
   CHECK_EQ(AGC_SHADER_ISA_COUNT, 5);
   CHECK_EQ(AGC_SHADER_FAMILY_COUNT, 20);
   CHECK_EQ(AGC_SHADER_ERROR_KIND_COUNT, 14);
}

static void
test_request_layout(void)
{
   struct agc_shader_request request = full_request();
   uint8_t *data;
   size_t size;

   CHECK_EQ(agc_shader_request_serialize(&request, &data, &size), OK);
   CHECK_EQ(size, 136 + 4 * 5 + 3 * 16 + 4 * 3);
   CHECK(memcmp(data, "AGSR", 4) == 0);
   CHECK_EQ(rd32(data), AGC_SHADER_REQUEST_MAGIC);
   CHECK_EQ(rd32(data + 4), AGC_SHADER_REQUEST_VERSION);
   CHECK_EQ(rd32(data + 8), size);
   CHECK_EQ(rd32(data + 12), reference_crc32(data + 16, size - 16));
   CHECK_EQ(rd32(data + 16), AGC_SHADER_COMPILER_VERSION);
   CHECK_EQ(data[20], AGC_SHADER_ISA_GFX115);
   CHECK_EQ(data[21], AGC_SHADER_STAGE_FRAGMENT);
   CHECK_EQ(data[22], 64);
   CHECK_EQ(data[23], AGC_SHADER_RAY_UNIT_REFERENCE);
   CHECK_EQ(data[24], AGC_SHADER_PRECISION_EXACT);
   CHECK_EQ(data[25], AGC_SHADER_VERTEX_STAGE_LS_HS);
   CHECK_EQ(data[26], 1);
   CHECK_EQ(data[27], 1);
   CHECK_EQ(rd32(data + 28), 0x7fff);
   CHECK_EQ(data[32], 0x21);
   CHECK_EQ(data[33], 0xbb);
   CHECK_EQ(data[34], 1);
   CHECK_EQ(data[35], 0);
   CHECK_EQ(rd32(data + 36), 0x11111111);
   CHECK_EQ(rd32(data + 40), 0x22222222);
   CHECK_EQ(rd32(data + 44), 0x33333333);
   CHECK_EQ(rd32(data + 48), 0x44444444);
   CHECK_EQ(rd32(data + 52), 0x77777777);
   CHECK_EQ(rd32(data + 56), 0x88888888);
   CHECK_EQ(rd32(data + 60), 0x99999999);
   CHECK_EQ(rd32(data + 64), 0xaaaaaaaa);
   CHECK_EQ(rd32(data + 68), 0x12121212);
   CHECK_EQ(rd64(data + 72), UINT64_C(0x5555555566666666));
   CHECK_EQ(rd64(data + 80), UINT64_C(0xccccccccdddddddd));
   CHECK_EQ(rd64(data + 88), UINT64_C(0xeeeeeeeeffffffff));
   CHECK_EQ(rd64(data + 96), UINT64_C(0x1313131314141414));
   CHECK_EQ(rd64(data + 104), UINT64_C(0x1515151516161616));
   CHECK_EQ(rd64(data + 112), UINT64_C(0x1717171718181818));
   CHECK_EQ(rd64(data + 120), UINT64_C(0x1919191920202020));
   CHECK_EQ(rd32(data + 128), 5);
   CHECK_EQ(rd32(data + 132), 3);
   CHECK_EQ(rd32(data + 136), 0xf4080100);
   CHECK_EQ(rd32(data + 152), 0xbf810000);
   CHECK_EQ(rd64(data + 156), UINT64_C(0x900000100));
   CHECK_EQ(rd32(data + 164), 2);
   CHECK_EQ(rd32(data + 168), 0);
   CHECK_EQ(rd32(data + 172), 0x7e020280);
   CHECK_EQ(rd64(data + 180), UINT64_C(0x900000200));
   CHECK_EQ(rd32(data + 188), 0);
   CHECK_EQ(rd64(data + 196), UINT64_C(0xa00000300));
   CHECK_EQ(rd32(data + 204), 1);
   CHECK_EQ(rd32(data + 212), 0xbf810000);
   free(data);

   request = minimal_request();
   CHECK_EQ(agc_shader_request_serialize(&request, &data, &size), OK);
   CHECK_EQ(size, 136 + 4 * 5);
   CHECK_EQ(rd32(data + 28), 0);
   CHECK_EQ(data[24], AGC_SHADER_PRECISION_FUSED);
   for (size_t at = 32; at < 44; at++)
      CHECK_EQ(data[at], 0);
   for (size_t at = 52; at < 128; at++)
      CHECK_EQ(data[at], 0);
   free(data);
}

static void
check_request_round_trip(const struct agc_shader_request *request)
{
   struct agc_shader_request back;
   uint8_t *data;
   uint8_t *again;
   size_t size;
   size_t again_size;

   CHECK_EQ(agc_shader_request_serialize(request, &data, &size), OK);
   CHECK_EQ(agc_shader_request_deserialize(data, size, &back), OK);
   CHECK(request_equal(request, &back));
   CHECK_EQ(agc_shader_request_serialize(&back, &again, &again_size), OK);
   CHECK(size == again_size && memcmp(data, again, size) == 0);
   agc_shader_request_free(&back);
   CHECK(back.storage == NULL && back.words == NULL && back.word_count == 0);
   free(again);
   free(data);
}

static void
test_request_round_trip(void)
{
   struct agc_shader_request request = full_request();
   struct agc_shader_request back;
   uint8_t *data;
   size_t size;

   check_request_round_trip(&request);
   CHECK_EQ(agc_shader_request_serialize(&request, &data, &size), OK);
   CHECK_EQ(agc_shader_request_deserialize(data, size, &back), OK);
   CHECK(back.storage != NULL);
   CHECK(back.words != shader_words && back.linkage.regions != shader_regions);
   CHECK_EQ(back.iface.user_sgpr_count, 0x21);
   CHECK_EQ(back.iface.param_exports, 0xbb);
   CHECK_EQ(back.iface.guest_code_address, UINT64_C(0x5555555566666666));
   CHECK_EQ(back.iface.attribute_ring.address, UINT64_C(0xeeeeeeeeffffffff));
   CHECK_EQ(back.iface.attribute_ring.bytes, 0x12121212);
   CHECK_EQ(back.linkage.second_half, UINT64_C(0x1919191920202020));
   CHECK_EQ(back.linkage.region_count, 3);
   CHECK(back.linkage.regions[1].words == NULL && back.linkage.regions[1].word_count == 0);
   CHECK_EQ(back.linkage.regions[2].words[0], 0xbf810000);
   agc_shader_request_free(&back);
   free(data);

   request = minimal_request();
   check_request_round_trip(&request);
   CHECK_EQ(agc_shader_request_serialize(&request, &data, &size), OK);
   CHECK_EQ(agc_shader_request_deserialize(data, size, &back), OK);
   CHECK(!back.iface.has_user_sgpr_count && !back.iface.has_guest_pgm_rsrc1 && !back.iface.has_guest_pgm_rsrc2);
   CHECK(!back.iface.has_guest_code_address && !back.iface.has_ps_input_ena && !back.iface.has_ps_input_addr);
   CHECK(!back.iface.has_ps_col_format && !back.iface.has_ps_z_format && !back.iface.has_param_exports);
   CHECK(!back.iface.has_descriptor_lut && !back.iface.has_attribute_ring && !back.iface.has_gds_buffer);
   CHECK(!back.iface.has_mip_stats_table && !back.iface.has_image_reports && !back.linkage.has_second_half);
   CHECK(back.linkage.regions == NULL && back.linkage.region_count == 0);
   agc_shader_request_free(&back);
   free(data);

   request.words = NULL;
   request.word_count = 0;
   check_request_round_trip(&request);
   CHECK_EQ(agc_shader_request_serialize(&request, &data, &size), OK);
   CHECK_EQ(size, 136);
   CHECK_EQ(agc_shader_request_deserialize(data, size, &back), OK);
   CHECK(back.storage == NULL && back.words == NULL);
   agc_shader_request_free(&back);
   free(data);

   request = full_request();
   request.options.wave_size = 16;
   request.options.target = AGC_SHADER_ISA_GFX1013;
   check_request_round_trip(&request);
   request.options.target = AGC_SHADER_ISA_GFX12;
   request.options.stage = AGC_SHADER_STAGE_VERTEX;
   request.compiler_version = 0xfedcba98;
   check_request_round_trip(&request);
}

static void
test_request_canonical(void)
{
   struct agc_shader_request clean = minimal_request();
   struct agc_shader_request dirty = minimal_request();
   struct agc_shader_code_region region = {UINT64_C(0x1000), NULL, 2};
   uint8_t *a;
   uint8_t *b;
   size_t a_size;
   size_t b_size;

   dirty.iface.user_sgpr_count = 0x5a;
   dirty.iface.guest_pgm_rsrc1 = 0x5a5a5a5a;
   dirty.iface.guest_pgm_rsrc2 = 0x5a5a5a5a;
   dirty.iface.guest_code_address = UINT64_C(0x5a5a5a5a5a5a5a5a);
   dirty.iface.ps_input_ena = 0x5a5a5a5a;
   dirty.iface.ps_input_addr = 0x5a5a5a5a;
   dirty.iface.ps_col_format = 0x5a5a5a5a;
   dirty.iface.ps_z_format = 0x5a5a5a5a;
   dirty.iface.param_exports = 0x5a;
   dirty.iface.descriptor_lut = UINT64_C(0x5a5a5a5a5a5a5a5a);
   dirty.iface.attribute_ring.address = UINT64_C(0x5a5a5a5a5a5a5a5a);
   dirty.iface.attribute_ring.bytes = 0x5a5a5a5a;
   dirty.iface.gds_buffer = UINT64_C(0x5a5a5a5a5a5a5a5a);
   dirty.iface.mip_stats_table = UINT64_C(0x5a5a5a5a5a5a5a5a);
   dirty.iface.image_reports = UINT64_C(0x5a5a5a5a5a5a5a5a);
   dirty.linkage.second_half = UINT64_C(0x5a5a5a5a5a5a5a5a);
   dirty.linkage.regions = shader_regions;
   dirty.storage = &dirty;
   CHECK_EQ(agc_shader_request_serialize(&clean, &a, &a_size), OK);
   CHECK_EQ(agc_shader_request_serialize(&dirty, &b, &b_size), OK);
   CHECK(a_size == b_size && memcmp(a, b, a_size) == 0);
   free(a);
   free(b);

   dirty = minimal_request();
   dirty.options.target = AGC_SHADER_ISA_COUNT;
   CHECK_EQ(agc_shader_request_serialize(&dirty, &a, &a_size), AGC_SHADER_FILE_UNREPRESENTABLE);
   CHECK(a == NULL && a_size == 0);
   dirty = minimal_request();
   dirty.options.stage = AGC_SHADER_STAGE_COUNT;
   CHECK_EQ(agc_shader_request_serialize(&dirty, &a, &a_size), AGC_SHADER_FILE_UNREPRESENTABLE);
   dirty = minimal_request();
   dirty.options.ray_unit = AGC_SHADER_RAY_UNIT_COUNT;
   CHECK_EQ(agc_shader_request_serialize(&dirty, &a, &a_size), AGC_SHADER_FILE_UNREPRESENTABLE);
   dirty = minimal_request();
   dirty.precision = AGC_SHADER_PRECISION_COUNT;
   CHECK_EQ(agc_shader_request_serialize(&dirty, &a, &a_size), AGC_SHADER_FILE_UNREPRESENTABLE);
   dirty = minimal_request();
   dirty.iface.vertex_stage = AGC_SHADER_VERTEX_STAGE_COUNT;
   CHECK_EQ(agc_shader_request_serialize(&dirty, &a, &a_size), AGC_SHADER_FILE_UNREPRESENTABLE);
   dirty = minimal_request();
   dirty.iface.es_input = AGC_SHADER_ES_INPUT_COUNT;
   CHECK_EQ(agc_shader_request_serialize(&dirty, &a, &a_size), AGC_SHADER_FILE_UNREPRESENTABLE);
   dirty = minimal_request();
   dirty.options.target = (enum agc_shader_isa)-1;
   CHECK_EQ(agc_shader_request_serialize(&dirty, &a, &a_size), AGC_SHADER_FILE_UNREPRESENTABLE);
   dirty = minimal_request();
   dirty.words = NULL;
   CHECK_EQ(agc_shader_request_serialize(&dirty, &a, &a_size), AGC_SHADER_FILE_UNREPRESENTABLE);
   dirty = minimal_request();
   dirty.linkage.region_count = 1;
   CHECK_EQ(agc_shader_request_serialize(&dirty, &a, &a_size), AGC_SHADER_FILE_UNREPRESENTABLE);
   dirty.linkage.regions = &region;
   CHECK_EQ(agc_shader_request_serialize(&dirty, &a, &a_size), AGC_SHADER_FILE_UNREPRESENTABLE);
   CHECK(a == NULL && a_size == 0);
}

static void
test_request_rejects(void)
{
   static const size_t full_counts[] = {128, 132, 164, 188, 204};
   static const size_t minimal_counts[] = {128, 132};
   struct agc_shader_request request = full_request();
   uint8_t *data;
   size_t size;

   CHECK_EQ(agc_shader_request_serialize(&request, &data, &size), OK);
   check_rejections(FILE_REQUEST, data, size, full_counts, COUNT_OF(full_counts));
   CHECK_EQ(parse_status(FILE_ARTIFACT, data, size), AGC_SHADER_FILE_BAD_MAGIC);

   data[20] = AGC_SHADER_ISA_COUNT;
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_REQUEST, data, size), AGC_SHADER_FILE_MALFORMED);
   data[20] = AGC_SHADER_ISA_GFX115;
   data[26] = 2;
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_REQUEST, data, size), AGC_SHADER_FILE_MALFORMED);
   data[26] = 1;
   wr32(data + 28, 0x7fff | 0x8000);
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_REQUEST, data, size), AGC_SHADER_FILE_MALFORMED);
   wr32(data + 28, 0x7fff & ~1u);
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_REQUEST, data, size), AGC_SHADER_FILE_MALFORMED);
   wr32(data + 28, 0x7fff);
   data[34] = 2;
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_REQUEST, data, size), AGC_SHADER_FILE_MALFORMED);
   data[34] = 1;
   data[35] = AGC_SHADER_ES_DOMAIN;
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_REQUEST, data, size), OK);
   data[35] = AGC_SHADER_ES_INPUT_COUNT;
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_REQUEST, data, size), AGC_SHADER_FILE_MALFORMED);
   data[35] = 0;
   wr32(data + 168, 1);
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_REQUEST, data, size), AGC_SHADER_FILE_MALFORMED);
   wr32(data + 168, 0);
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_REQUEST, data, size), OK);
   wr32(data + 4, AGC_SHADER_REQUEST_VERSION + 1);
   CHECK_EQ(parse_status(FILE_REQUEST, data, size), AGC_SHADER_FILE_BAD_VERSION);
   free(data);

   request = minimal_request();
   CHECK_EQ(agc_shader_request_serialize(&request, &data, &size), OK);
   check_rejections(FILE_REQUEST, data, size, minimal_counts, COUNT_OF(minimal_counts));
   for (size_t at = 32; at < 128; at++) {
      if (at >= 44 && at < 52)
         continue;
      data[at] = at == 34 || at == 35 ? 2 : 1;
      reseal(data, size);
      CHECK_EQ(parse_status(FILE_REQUEST, data, size), AGC_SHADER_FILE_MALFORMED);
      data[at] = 0;
   }
   free(data);

   CHECK_EQ(agc_shader_request_deserialize(NULL, 0, &request), AGC_SHADER_FILE_TRUNCATED);
   CHECK(request.storage == NULL && request.word_count == 0);
}

static bool
mutate_request(struct agc_shader_request *r, uint32_t *words, struct agc_shader_code_region *regions,
               uint32_t *region_words, unsigned index)
{
   struct agc_shader_interface *f = &r->iface;

   switch (index) {
   case 0: r->compiler_version++; break;
   case 1: r->options.target = AGC_SHADER_ISA_GFX11; break;
   case 2: r->options.stage = AGC_SHADER_STAGE_VERTEX; break;
   case 3: r->options.wave_size = 32; break;
   case 4: r->options.ray_unit = AGC_SHADER_RAY_UNIT_HOST; break;
   case 5: r->precision = AGC_SHADER_PRECISION_FUSED; break;
   case 6: f->has_user_sgpr_count = false; break;
   case 7: f->user_sgpr_count ^= 1; break;
   case 8: f->has_guest_pgm_rsrc1 = false; break;
   case 9: f->guest_pgm_rsrc1 ^= 0x80000000; break;
   case 10: f->has_guest_pgm_rsrc2 = false; break;
   case 11: f->guest_pgm_rsrc2 ^= 1; break;
   case 12: f->lds_bytes ^= 0x100; break;
   case 13: f->scratch_bytes_per_lane ^= 0x100; break;
   case 14: f->has_guest_code_address = false; break;
   case 15: f->guest_code_address ^= UINT64_C(1) << 63; break;
   case 16: f->has_ps_input_ena = false; break;
   case 17: f->ps_input_ena ^= 2; break;
   case 18: f->has_ps_input_addr = false; break;
   case 19: f->ps_input_addr ^= 2; break;
   case 20: f->has_ps_col_format = false; break;
   case 21: f->ps_col_format ^= 2; break;
   case 22: f->has_ps_z_format = false; break;
   case 23: f->ps_z_format ^= 2; break;
   case 24: f->dual_source_blend = false; break;
   case 25: f->has_param_exports = false; break;
   case 26: f->param_exports ^= 0x80; break;
   case 27: f->has_descriptor_lut = false; break;
   case 28: f->descriptor_lut ^= UINT64_C(1) << 32; break;
   case 29: f->preconvert_user_descriptors = false; break;
   case 30: f->has_attribute_ring = false; break;
   case 31: f->attribute_ring.address ^= UINT64_C(1) << 16; break;
   case 32: f->attribute_ring.bytes ^= 0x10000; break;
   case 33: f->has_gds_buffer = false; break;
   case 34: f->gds_buffer ^= 1; break;
   case 35: f->has_mip_stats_table = false; break;
   case 36: f->mip_stats_table ^= 1; break;
   case 37: f->has_image_reports = false; break;
   case 38: f->image_reports ^= 1; break;
   case 39: f->vertex_stage = AGC_SHADER_VERTEX_STAGE_ES_GS; break;
   case 40: words[0] ^= 1; break;
   case 41: words[COUNT_OF(shader_words) - 1] ^= 0x80000000; break;
   case 42: r->word_count--; break;
   case 43: regions[0].address ^= 4; break;
   case 44: region_words[1] ^= 1; break;
   case 45: regions[0].word_count--; break;
   case 46: regions[2].address ^= UINT64_C(1) << 40; break;
   case 47: r->linkage.region_count--; break;
   case 48: r->linkage.has_second_half = false; break;
   case 49: r->linkage.second_half ^= 0x100; break;
   case 50: regions[1].words = region_words; regions[1].word_count = 1; break;
   case 51: f->native_dcc = false; break;
   case 52: f->es_input = AGC_SHADER_ES_DOMAIN; break;
   default: return false;
   }
   return true;
}

#define REQUEST_MUTATIONS 53

static void
test_request_key(void)
{
   static const char reference_hex[] = "6dea21a4955c16ae0c4b766ec01ee6811b6bc013f8c023194c9fff6af600bc51";
   struct agc_shader_key keys[REQUEST_MUTATIONS + 1];
   struct agc_shader_key again;
   struct agc_shader_key parsed;
   struct agc_shader_request base = full_request();
   char hex[AGC_SHADER_KEY_HEX_SIZE];
   uint8_t *base_data;
   size_t base_size;
   unsigned count = 0;

   CHECK_EQ(agc_shader_request_key(&base, &keys[0]), OK);
   CHECK_EQ(agc_shader_request_key(&base, &again), OK);
   CHECK(memcmp(&keys[0], &again, sizeof(again)) == 0);
   CHECK_EQ(agc_shader_request_serialize(&base, &base_data, &base_size), OK);
   agc_shader_key_of_bytes(base_data, base_size, &again);
   CHECK(memcmp(&keys[0], &again, sizeof(again)) == 0);
   agc_shader_key_hex(&keys[0], hex);
   CHECK_EQ(strlen(hex), 64);
   CHECK(strcmp(hex, reference_hex) == 0);
   if (strcmp(hex, reference_hex) != 0)
      fprintf(stderr, "reference request key is %s\n", hex);
   CHECK(agc_shader_key_from_hex(hex, &parsed));
   CHECK(memcmp(&keys[0], &parsed, sizeof(parsed)) == 0);

   for (unsigned index = 0;; index++) {
      uint32_t words[COUNT_OF(shader_words)];
      uint32_t region_words[COUNT_OF(region_a_words)];
      struct agc_shader_code_region regions[COUNT_OF(shader_regions)];
      struct agc_shader_request request = full_request();
      struct agc_shader_request back;
      uint8_t *data;
      size_t size;

      memcpy(words, shader_words, sizeof(words));
      memcpy(region_words, region_a_words, sizeof(region_words));
      memcpy(regions, shader_regions, sizeof(regions));
      regions[0].words = region_words;
      request.words = words;
      request.linkage.regions = regions;
      CHECK_EQ(agc_shader_request_key(&request, &again), OK);
      CHECK(memcmp(&keys[0], &again, sizeof(again)) == 0);
      if (!mutate_request(&request, words, regions, region_words, index))
         break;
      count++;
      CHECK(count <= REQUEST_MUTATIONS);
      if (count > REQUEST_MUTATIONS)
         break;
      CHECK_EQ(agc_shader_request_key(&request, &keys[count]), OK);
      for (unsigned other = 0; other < count; other++)
         CHECK(memcmp(&keys[other], &keys[count], sizeof(keys[0])) != 0);
      CHECK(!request_equal(&base, &request));
      CHECK_EQ(agc_shader_request_serialize(&request, &data, &size), OK);
      CHECK(size != base_size || memcmp(data, base_data, size) != 0);
      CHECK_EQ(agc_shader_request_deserialize(data, size, &back), OK);
      CHECK(request_equal(&request, &back));
      agc_shader_request_free(&back);
      free(data);
   }
   CHECK_EQ(count, REQUEST_MUTATIONS);
   free(base_data);

   base = minimal_request();
   CHECK_EQ(agc_shader_request_key(&base, &keys[0]), OK);
   base.iface.guest_pgm_rsrc1 = 0x1234;
   base.iface.descriptor_lut = 0x5678;
   base.linkage.second_half = 0x9abc;
   CHECK_EQ(agc_shader_request_key(&base, &again), OK);
   CHECK(memcmp(&keys[0], &again, sizeof(again)) == 0);
   base.iface.has_guest_pgm_rsrc1 = true;
   CHECK_EQ(agc_shader_request_key(&base, &again), OK);
   CHECK(memcmp(&keys[0], &again, sizeof(again)) != 0);
   base.options.target = AGC_SHADER_ISA_COUNT;
   CHECK_EQ(agc_shader_request_key(&base, &again), AGC_SHADER_FILE_UNREPRESENTABLE);
   for (unsigned i = 0; i < AGC_SHADER_KEY_BYTES; i++)
      CHECK_EQ(again.bytes[i], 0);

   for (unsigned i = 0; i < AGC_SHADER_KEY_BYTES; i++)
      parsed.bytes[i] = (uint8_t)(i * 17 + 3);
   agc_shader_key_hex(&parsed, hex);
   CHECK(strncmp(hex, "031425364758697a8b9cadbecfe0f102", 32) == 0);
   CHECK(agc_shader_key_from_hex(hex, &again) && memcmp(&parsed, &again, sizeof(again)) == 0);
   hex[10] = 'A';
   CHECK(!agc_shader_key_from_hex(hex, &again));
   hex[10] = 'g';
   CHECK(!agc_shader_key_from_hex(hex, &again));
   hex[10] = 'a';
   hex[63] = 0;
   CHECK(!agc_shader_key_from_hex(hex, &again));
   CHECK(!agc_shader_key_from_hex("", &again));
   CHECK(!agc_shader_key_from_hex("031425364758697a8b9cadbecfe0f102132435465768798a9bacbdcedff0011223", &again));
   agc_shader_key_of_bytes((const uint8_t *)"", 0, &again);
   agc_shader_key_hex(&again, hex);
   CHECK(strcmp(hex, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262") == 0);
}

static void
test_artifact_layout(void)
{
   static const char reference_hex[] = "60b2cc48840f87dff00204d605e64563085792146706e30908321ba03b40e061";
   struct agc_shader_artifact artifact = full_artifact();
   struct agc_shader_key digest;
   char hex[AGC_SHADER_KEY_HEX_SIZE];
   uint8_t *data;
   size_t size;
   size_t at;

   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   agc_shader_key_of_bytes(data, size, &digest);
   agc_shader_key_hex(&digest, hex);
   CHECK(strcmp(hex, reference_hex) == 0);
   if (strcmp(hex, reference_hex) != 0)
      fprintf(stderr, "reference artifact digest is %s\n", hex);
   CHECK(memcmp(data, "AGSA", 4) == 0);
   CHECK_EQ(rd32(data), AGC_SHADER_ARTIFACT_MAGIC);
   CHECK_EQ(rd32(data + 4), AGC_SHADER_ARTIFACT_VERSION);
   CHECK_EQ(rd32(data + 8), size);
   CHECK_EQ(rd32(data + 12), reference_crc32(data + 16, size - 16));
   CHECK_EQ(rd32(data + 16), 0);
   CHECK_EQ(data[20], AGC_SHADER_ISA_GFX12);
   CHECK_EQ(data[21], AGC_SHADER_STAGE_VERTEX);
   CHECK_EQ(data[22], 64);
   CHECK_EQ(data[23], AGC_SHADER_RAY_UNIT_REFERENCE);
   CHECK_EQ(data[24], AGC_SHADER_PRECISION_EXACT);
   CHECK_EQ(data[25], 0x77);
   CHECK_EQ(data[26], 1);
   CHECK_EQ(data[27], 0x1f);
   CHECK_EQ(rd32(data + 28), 0x11111111);
   CHECK_EQ(rd32(data + 32), 0x22222222);
   CHECK_EQ(rd32(data + 36), 0x44443333);
   CHECK_EQ(rd32(data + 40), 0x55555555);
   CHECK_EQ(rd32(data + 44), 0x66666666);
   CHECK_EQ(rd32(data + 48), 6);
   CHECK_EQ(rd32(data + 52), 0x88888888);
   CHECK_EQ(rd64(data + 56), UINT64_C(0x99999999aaaaaaaa));
   CHECK_EQ(rd32(data + 64), 0xbbbbbbbb);
   CHECK_EQ(rd32(data + 68), 0xeeeeeeee);
   CHECK_EQ(data[72], 0xcc);
   CHECK_EQ(data[73], 0xdd);
   CHECK_EQ(data[74], 1);
   CHECK_EQ(data[75], 1);
   CHECK_EQ(data[76], 1);
   CHECK_EQ(data[77], 0xf1);
   CHECK_EQ(data[78], 0x15);
   CHECK_EQ(data[79], 0x16);
   CHECK_EQ(rd32(data + 80), 0x12121212);
   CHECK_EQ(rd32(data + 84), 0x13131313);
   CHECK_EQ(rd32(data + 88), 0x14141414);
   CHECK_EQ(rd32(data + 92), 1);
   CHECK_EQ(rd32(data + 96), 7);
   CHECK_EQ(rd32(data + 100), 0xb320);
   CHECK_EQ(rd32(data + 104), AGC_SHADER_REGISTER_CODE_ADDRESS);
   CHECK_EQ(rd32(data + 108), 0xb324);
   CHECK_EQ(rd32(data + 112), AGC_SHADER_REGISTER_CODE_ADDRESS);
   CHECK_EQ(rd32(data + 116), 0xb220);
   CHECK_EQ(rd32(data + 120), AGC_SHADER_REGISTER_GUEST_USER_DATA_ADDRESS);
   CHECK_EQ(rd32(data + 124), 0xb224);
   CHECK_EQ(rd32(data + 128), AGC_SHADER_REGISTER_GUEST_USER_DATA_ADDRESS);
   CHECK_EQ(rd32(data + 132), 8);
   CHECK_EQ(rd32(data + 136), 3);
   CHECK_EQ(rd32(data + 140), 1);
   CHECK_EQ(rd32(data + 144), 2);
   CHECK_EQ(rd32(data + 148), 5);
   CHECK_EQ(rd32(data + 152), 1);
   CHECK_EQ(rd32(data + 156), 0xbe8300ff);
   CHECK_EQ(rd32(data + 184), 0xbf9f0000);
   at = 188;
   CHECK_EQ(rd32(data + at), 0x00020003);
   CHECK_EQ(rd32(data + at + 4), 0x01040400);
   CHECK_EQ(rd32(data + at + 8), 0x00080801);
   CHECK_EQ(rd32(data + at + 12), 0x00041002);
   at += 16;
   CHECK_EQ(rd32(data + at), 0x40);
   CHECK_EQ(rd32(data + at + 4), 0x00060101);
   CHECK_EQ(rd32(data + at + 8), 0x80);
   CHECK_EQ(rd32(data + at + 12), 0);
   at += 16;
   CHECK_EQ(rd32(data + at), 0x10);
   CHECK_EQ(rd32(data + at + 4), 0x00080801);
   CHECK_EQ(rd32(data + at + 8), 0x00000800);
   CHECK_EQ(rd32(data + at + 12), 0);
   CHECK_EQ(rd32(data + at + 16), 0);
   CHECK_EQ(rd32(data + at + 20), 0);
   CHECK_EQ(rd32(data + at + 24), 0);
   at += 28;
   CHECK_EQ(rd32(data + at), 0x14);
   CHECK_EQ(rd32(data + at + 4), 0x02040400);
   CHECK_EQ(rd32(data + at + 8), 0x00010001);
   CHECK_EQ(rd32(data + at + 12), 0x00000101);
   CHECK_EQ(rd32(data + at + 16), 0x123);
   CHECK_EQ(rd32(data + at + 20), 0xfff);
   CHECK_EQ(rd32(data + at + 24), 1);
   at += 28;
   CHECK_EQ(rd64(data + at), 0x30);
   CHECK_EQ(rd32(data + at + 8), 2);
   CHECK_EQ(rd32(data + at + 12), 0x00000102);
   CHECK_EQ(rd64(data + at + 16), 0x10);
   CHECK_EQ(rd32(data + at + 24), AGC_SHADER_ADDRESS_POINTER);
   CHECK_EQ(rd64(data + at + 28), (uint64_t)-0x20);
   CHECK_EQ(rd32(data + at + 36), AGC_SHADER_ADDRESS_BUFFER);
   at += 40;
   CHECK_EQ(rd32(data + at), 0x18);
   CHECK_EQ(rd32(data + at + 8), 0x00000002);
   CHECK_EQ(rd32(data + at + 24), 3);
   at += 28 + 16 + (16 + 12) + (16 + 24);
   CHECK_EQ(rd32(data + at), 0x1c);
   CHECK_EQ(rd32(data + at + 8), 0x01010002);
   CHECK_EQ(rd32(data + at + 12), 0x00010000);
   CHECK_EQ(rd32(data + at + 20), 9);
   CHECK_EQ(rd32(data + at + 24), 0);
   at += 28;
   CHECK_EQ(rd32(data + at), 0x20);
   CHECK_EQ(rd32(data + at + 4), 0x03081801);
   CHECK_EQ(rd32(data + at + 8), 0x00010003);
   CHECK_EQ(rd32(data + at + 12), 0x00020000);
   at += 28;
   CHECK_EQ(rd32(data + at), 0x24);
   CHECK_EQ(rd32(data + at + 16), 0x40);
   at += 28;
   CHECK_EQ(at, size);
   free(data);

   artifact = minimal_artifact();
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   CHECK_EQ(size, 156);
   CHECK_EQ(data[20], AGC_SHADER_ISA_GFX103);
   CHECK_EQ(data[22], 32);
   for (size_t i = 23; i < size; i++)
      CHECK_EQ(data[i], 0);
   free(data);

   artifact = error_artifact(AGC_SHADER_ERROR_UNSUPPORTED_INSTRUCTION, "no gfx12 form");
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   CHECK_EQ(size, 44 + 16);
   CHECK_EQ(rd32(data + 16), AGC_SHADER_ERROR_UNSUPPORTED_INSTRUCTION);
   CHECK_EQ(data[20], AGC_SHADER_FAMILY_VINTERP);
   CHECK_EQ(data[21], AGC_SHADER_ISA_GFX12);
   CHECK_EQ(data[22], 0);
   CHECK_EQ(data[23], 0);
   CHECK_EQ(rd32(data + 24), 0x1a1a1a1a);
   CHECK_EQ(rd32(data + 28), 0);
   CHECK_EQ(rd32(data + 32), 0);
   CHECK_EQ(rd32(data + 36), 0x00001d1d);
   CHECK_EQ(rd32(data + 40), 13);
   CHECK(memcmp(data + 44, "no gfx12 form\0\0\0", 16) == 0);
   free(data);

   artifact = error_artifact(AGC_SHADER_ERROR_TARGET_MISMATCH, "ignored");
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   CHECK_EQ(size, 44);
   CHECK_EQ(rd32(data + 16), AGC_SHADER_ERROR_TARGET_MISMATCH);
   CHECK_EQ(rd32(data + 20), (uint32_t)AGC_SHADER_ISA_GFX11 << 16 | (uint32_t)AGC_SHADER_ISA_GFX115 << 24);
   for (size_t i = 24; i < size; i++)
      CHECK_EQ(data[i], 0);
   free(data);

   artifact = error_artifact(AGC_SHADER_ERROR_OUT_OF_REGISTERS, NULL);
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   CHECK_EQ(size, 44);
   CHECK_EQ(rd32(data + 36), 0x1e1e0000);
   free(data);

   artifact = error_artifact(AGC_SHADER_ERROR_INVALID_ENCODING, NULL);
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   CHECK_EQ(rd32(data + 24), 0x1a1a1a1a);
   CHECK_EQ(rd32(data + 28), 0x1b1b1b1b);
   CHECK_EQ(rd32(data + 32), 0);
   free(data);

   artifact = error_artifact(AGC_SHADER_ERROR_INVALID_BRANCH, NULL);
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   CHECK_EQ(rd32(data + 24), 0x1a1a1a1a);
   CHECK_EQ(rd32(data + 28), 0);
   CHECK_EQ(rd32(data + 32), 0x1c1c1c1c);
   free(data);
}

static void
check_artifact_round_trip(const struct agc_shader_artifact *artifact)
{
   struct agc_shader_artifact back;
   uint8_t *data;
   uint8_t *again;
   size_t size;
   size_t again_size;

   CHECK_EQ(agc_shader_artifact_serialize(artifact, &data, &size), OK);
   CHECK_EQ(size % 4, 0);
   CHECK_EQ(agc_shader_artifact_deserialize(data, size, &back), OK);
   CHECK(artifact_equal(artifact, &back));
   CHECK_EQ(agc_shader_artifact_serialize(&back, &again, &again_size), OK);
   CHECK(size == again_size && memcmp(data, again, size) == 0);
   agc_shader_artifact_free(&back);
   CHECK(back.storage == NULL && back.code == NULL && back.code_count == 0);
   free(again);
   free(data);
}

static void
test_artifact_round_trip(void)
{
   struct agc_shader_artifact artifact = full_artifact();
   struct agc_shader_artifact back;
   uint8_t *data;
   size_t size;

   check_artifact_round_trip(&artifact);
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   CHECK_EQ(agc_shader_artifact_deserialize(data, size, &back), OK);
   CHECK(back.storage != NULL && back.code != code_words && back.descriptor_uses != code_uses);
   CHECK_EQ(back.error.kind, AGC_SHADER_ERROR_NONE);
   CHECK(back.error.reason == NULL);
   CHECK_EQ(back.code_count, 8);
   CHECK_EQ(back.code[7], 0xbf9f0000);
   CHECK_EQ(back.num_sgprs, 0x3333);
   CHECK_EQ(back.num_vgprs, 0x4444);
   CHECK_EQ(back.exports.depth_mask, 0xcc);
   CHECK_EQ(back.exports.position_mask, 0xdd);
   CHECK_EQ(back.exports.param_mask, 0xeeeeeeee);
   CHECK_EQ(back.user_data_tail_sgpr, 0xf1);
   CHECK_EQ(back.buffer_fill.buffer, 0x15);
   CHECK_EQ(back.buffer_fill.value, 0x16);
   CHECK_EQ(back.ngg.attribute_ring_relocations[1], 7);
   CHECK_EQ(back.ngg.registers[3].reg, 0xb224);
   CHECK_EQ(back.descriptor_use_count, 5);
   CHECK_EQ(back.descriptor_uses[1].source.tables[0].links[1].offset, (uint64_t)-0x20);
   CHECK_EQ(back.descriptor_uses[2].source.table_count, 3);
   CHECK(back.descriptor_uses[2].source.tables[0].links == NULL);
   CHECK(back.descriptor_uses[2].source.tables[1].offset == INT64_MAX);
   CHECK(back.descriptor_uses[2].source.tables[1].links[0].offset == INT64_MIN);
   CHECK_EQ(back.descriptor_uses[2].source.tables[2].links[0].offset, 0x10);
   CHECK(back.descriptor_uses[3].source.tables == NULL && back.descriptor_uses[3].source.table_count == 0);
   CHECK_EQ(back.descriptor_uses[3].buffer_access.scalar_offset.sgpr, 9);
   CHECK_EQ(back.descriptor_uses[3].buffer_access.scalar_offset.bound, 0);
   CHECK_EQ(back.descriptor_uses[4].buffer_access.scalar_offset.kind, AGC_SHADER_SCALAR_OFFSET_UNKNOWN);
   CHECK_EQ(back.pointer_uses[0].buffer_access.immediate_offset, 0x40);
   agc_shader_artifact_free(&back);
   free(data);

   artifact = minimal_artifact();
   check_artifact_round_trip(&artifact);
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   CHECK_EQ(agc_shader_artifact_deserialize(data, size, &back), OK);
   CHECK(back.storage == NULL && back.code == NULL && back.bindings == NULL && back.descriptor_uses == NULL);
   CHECK(!back.has_user_data_tail_sgpr && !back.has_spi_shader_col_format && !back.has_ngg && !back.has_buffer_fill);
   agc_shader_artifact_free(&back);
   free(data);

   artifact = full_artifact();
   artifact.has_ngg = true;
   artifact.ngg.has_attribute_ring_relocations = false;
   check_artifact_round_trip(&artifact);
   artifact.has_ngg = false;
   artifact.ngg.has_attribute_ring_relocations = true;
   check_artifact_round_trip(&artifact);
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   CHECK_EQ(data[27], 0x1f & ~0x0c);
   for (size_t at = 84; at < 132; at++)
      CHECK_EQ(data[at], 0);
   free(data);
   artifact = full_artifact();
   artifact.program_words = artifact.code_count;
   artifact.ngg.attribute_ring_relocations[0] = artifact.code_count - 1;
   check_artifact_round_trip(&artifact);
}

static void
test_artifact_errors(void)
{
   static const char *const reasons[] = {"", "a", "ab", "abc", "abcd", "abcde", "wave_size must be 32 or 64"};
   static const uint8_t expected[AGC_SHADER_ERROR_KIND_COUNT] = {
      [AGC_SHADER_ERROR_UNSUPPORTED] = 0,
      [AGC_SHADER_ERROR_INVALID_OPTIONS] = 0x80,
      [AGC_SHADER_ERROR_TARGET_MISMATCH] = 0x40,
      [AGC_SHADER_ERROR_EMPTY_PROGRAM] = 0,
      [AGC_SHADER_ERROR_TRUNCATED] = 0x01,
      [AGC_SHADER_ERROR_INVALID_ENCODING] = 0x03,
      [AGC_SHADER_ERROR_UNKNOWN_OPCODE] = 0x09,
      [AGC_SHADER_ERROR_INVALID_OPERAND] = 0x81,
      [AGC_SHADER_ERROR_INVALID_BRANCH] = 0x05,
      [AGC_SHADER_ERROR_MISSING_END_PROGRAM] = 0,
      [AGC_SHADER_ERROR_UNSUPPORTED_INSTRUCTION] = 0xa9,
      [AGC_SHADER_ERROR_BRANCH_OUT_OF_RANGE] = 0x01,
      [AGC_SHADER_ERROR_OUT_OF_REGISTERS] = 0x10,
   };

   for (unsigned kind = 1; kind < AGC_SHADER_ERROR_KIND_COUNT; kind++) {
      for (size_t k = 0; k < COUNT_OF(reasons); k++) {
         struct agc_shader_artifact artifact = error_artifact((enum agc_shader_error_kind)kind, reasons[k]);
         struct agc_shader_artifact back;
         unsigned fields = expected[kind];
         bool has_reason = (fields & 0x80) != 0;
         size_t length = has_reason ? strlen(reasons[k]) : 0;
         uint8_t *data;
         uint8_t *again;
         size_t size;
         size_t again_size;

         CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
         CHECK_EQ(size, 44 + (length + 3) / 4 * 4);
         CHECK_EQ(rd32(data + 16), kind);
         CHECK_EQ(rd32(data + 40), length);
         CHECK_EQ(agc_shader_artifact_deserialize(data, size, &back), OK);
         CHECK_EQ(back.error.kind, kind);
         CHECK_EQ(back.error.pc, fields & 0x01 ? 0x1a1a1a1a : 0);
         CHECK_EQ(back.error.word, fields & 0x02 ? 0x1b1b1b1b : 0);
         CHECK_EQ(back.error.branch_target, fields & 0x04 ? 0x1c1c1c1c : 0);
         CHECK_EQ(back.error.opcode, fields & 0x08 ? 0x1d1d : 0);
         CHECK_EQ(back.error.family, fields & 0x08 ? AGC_SHADER_FAMILY_VINTERP : 0);
         CHECK_EQ(back.error.vgprs, fields & 0x10 ? 0x1e1e : 0);
         CHECK_EQ(back.error.target, fields & 0x20 ? AGC_SHADER_ISA_GFX12 : 0);
         CHECK_EQ(back.error.compiler, fields & 0x40 ? AGC_SHADER_ISA_GFX11 : 0);
         CHECK_EQ(back.error.requested, fields & 0x40 ? AGC_SHADER_ISA_GFX115 : 0);
         if (has_reason)
            CHECK(back.error.reason && strcmp(back.error.reason, reasons[k]) == 0);
         else
            CHECK(back.error.reason == NULL);
         CHECK(back.code == NULL && back.code_count == 0 && back.bindings == NULL && !back.has_ngg);
         CHECK(back.descriptor_uses == NULL && back.pgm_rsrc1 == 0 && back.options.wave_size == 0);
         CHECK_EQ(agc_shader_artifact_serialize(&back, &again, &again_size), OK);
         CHECK(size == again_size && memcmp(data, again, size) == 0);
         free(again);
         agc_shader_artifact_free(&back);
         if (k == COUNT_OF(reasons) - 1 || k == 1)
            check_rejections(FILE_ARTIFACT, data, size, NULL, 0);
         free(data);
      }
   }

   struct agc_shader_artifact artifact = error_artifact(AGC_SHADER_ERROR_INVALID_OPERAND, NULL);
   struct agc_shader_artifact back;
   uint8_t *data;
   size_t size;
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   CHECK_EQ(size, 44);
   CHECK_EQ(agc_shader_artifact_deserialize(data, size, &back), OK);
   CHECK(back.error.reason && back.error.reason[0] == 0);
   agc_shader_artifact_free(&back);
   free(data);

   artifact = error_artifact(AGC_SHADER_ERROR_INVALID_OPTIONS, "abc");
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   CHECK_EQ(size, 48);
   data[45] = 0;
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_ARTIFACT, data, size), AGC_SHADER_FILE_MALFORMED);
   data[45] = 'b';
   data[47] = 1;
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_ARTIFACT, data, size), AGC_SHADER_FILE_MALFORMED);
   data[47] = 0;
   wr32(data + 40, 4);
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_ARTIFACT, data, size), AGC_SHADER_FILE_MALFORMED);
   wr32(data + 40, 5);
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_ARTIFACT, data, size), AGC_SHADER_FILE_TRUNCATED);
   wr32(data + 40, 0xffffffff);
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_ARTIFACT, data, size), AGC_SHADER_FILE_TRUNCATED);
   wr32(data + 40, 3);
   wr32(data + 16, AGC_SHADER_ERROR_KIND_COUNT);
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_ARTIFACT, data, size), AGC_SHADER_FILE_MALFORMED);
   wr32(data + 16, AGC_SHADER_ERROR_INVALID_OPTIONS | 0x100);
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_ARTIFACT, data, size), AGC_SHADER_FILE_MALFORMED);
   wr32(data + 16, AGC_SHADER_ERROR_EMPTY_PROGRAM);
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_ARTIFACT, data, size), AGC_SHADER_FILE_MALFORMED);
   wr32(data + 16, AGC_SHADER_ERROR_INVALID_OPTIONS);
   reseal(data, size);
   CHECK_EQ(parse_status(FILE_ARTIFACT, data, size), OK);
   free(data);

   artifact = error_artifact((enum agc_shader_error_kind)AGC_SHADER_ERROR_KIND_COUNT, NULL);
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), AGC_SHADER_FILE_UNREPRESENTABLE);
   artifact = error_artifact(AGC_SHADER_ERROR_UNKNOWN_OPCODE, NULL);
   artifact.error.family = AGC_SHADER_FAMILY_COUNT;
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), AGC_SHADER_FILE_UNREPRESENTABLE);
   artifact.error.kind = AGC_SHADER_ERROR_TRUNCATED;
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   free(data);
   artifact = error_artifact(AGC_SHADER_ERROR_TARGET_MISMATCH, NULL);
   artifact.error.requested = AGC_SHADER_ISA_COUNT;
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), AGC_SHADER_FILE_UNREPRESENTABLE);
   artifact = error_artifact(AGC_SHADER_ERROR_UNSUPPORTED_INSTRUCTION, NULL);
   artifact.error.target = AGC_SHADER_ISA_COUNT;
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), AGC_SHADER_FILE_UNREPRESENTABLE);
   CHECK(data == NULL && size == 0);
}

struct artifact_scratch {
   uint32_t code[COUNT_OF(code_words)];
   struct agc_shader_binding bindings[COUNT_OF(code_bindings)];
   struct agc_shader_binding preconverted[COUNT_OF(code_preconverted)];
   struct agc_shader_external_target externals[COUNT_OF(code_externals)];
   struct agc_shader_descriptor_use uses[COUNT_OF(code_uses)];
   struct agc_shader_descriptor_use pointer_uses[COUNT_OF(code_pointer_uses)];
   struct agc_shader_descriptor_table single[COUNT_OF(table_single)];
   struct agc_shader_descriptor_table choices[COUNT_OF(table_choices)];
   struct agc_shader_table_link links[COUNT_OF(links_a)];
};

static struct agc_shader_artifact
scratch_artifact(struct artifact_scratch *s)
{
   struct agc_shader_artifact a = full_artifact();

   memcpy(s->code, code_words, sizeof(s->code));
   memcpy(s->bindings, code_bindings, sizeof(s->bindings));
   memcpy(s->preconverted, code_preconverted, sizeof(s->preconverted));
   memcpy(s->externals, code_externals, sizeof(s->externals));
   memcpy(s->uses, code_uses, sizeof(s->uses));
   memcpy(s->pointer_uses, code_pointer_uses, sizeof(s->pointer_uses));
   memcpy(s->single, table_single, sizeof(s->single));
   memcpy(s->choices, table_choices, sizeof(s->choices));
   memcpy(s->links, links_a, sizeof(s->links));
   s->single[0].links = s->links;
   s->uses[1].source.tables = s->single;
   s->uses[2].source.tables = s->choices;
   a.code = s->code;
   a.bindings = s->bindings;
   a.preconverted_descriptors = s->preconverted;
   a.external_targets = s->externals;
   a.descriptor_uses = s->uses;
   a.pointer_uses = s->pointer_uses;
   return a;
}

static bool
mutate_artifact(struct agc_shader_artifact *a, struct artifact_scratch *s, unsigned index)
{
   switch (index) {
   case 0: s->code[0] ^= 1; break;
   case 1: s->code[7] ^= 0x80000000; break;
   case 2: a->code_count--; a->ngg.attribute_ring_relocations[1] = 6; break;
   case 3: a->options.target = AGC_SHADER_ISA_GFX11; break;
   case 4: a->options.stage = AGC_SHADER_STAGE_FRAGMENT; break;
   case 5: a->options.wave_size = 32; break;
   case 6: a->options.ray_unit = AGC_SHADER_RAY_UNIT_HOST; break;
   case 7: a->precision = AGC_SHADER_PRECISION_FUSED; break;
   case 8: a->pgm_rsrc1 ^= 1; break;
   case 9: a->pgm_rsrc2 ^= 1; break;
   case 10: a->num_sgprs ^= 1; break;
   case 11: a->num_vgprs ^= 1; break;
   case 12: a->lds_bytes ^= 1; break;
   case 13: a->scratch_bytes_per_lane ^= 1; break;
   case 14: a->user_sgpr_count ^= 1; break;
   case 15: a->program_words--; break;
   case 16: a->guest_words ^= 1; break;
   case 17: a->guest_hash ^= UINT64_C(1) << 63; break;
   case 18: s->bindings[0].kind = AGC_SHADER_RESOURCE_IMAGE_DESCRIPTOR; break;
   case 19: s->bindings[1].base_sgpr ^= 1; break;
   case 20: s->bindings[2].dword_count ^= 1; break;
   case 21: s->bindings[1].writable = false; break;
   case 22: a->binding_count--; break;
   case 23: a->exports.color_mask ^= 1; break;
   case 24: a->exports.depth_mask ^= 1; break;
   case 25: a->exports.position_mask ^= 1; break;
   case 26: a->exports.param_mask ^= 1; break;
   case 27: a->exports.primitive = false; break;
   case 28: a->exports.compressed = false; break;
   case 29: a->exports.valid_mask = false; break;
   case 30: a->has_user_data_tail_sgpr = false; break;
   case 31: a->user_data_tail_sgpr ^= 1; break;
   case 32: a->uses_pc_relative = false; break;
   case 33: s->preconverted[0].base_sgpr ^= 1; break;
   case 34: a->preconverted_descriptor_count = 0; break;
   case 35: a->has_spi_shader_col_format = false; break;
   case 36: a->spi_shader_col_format ^= 1; break;
   case 37: a->has_ngg = false; break;
   case 38: a->ngg.param_count ^= 1; break;
   case 39: a->ngg.spi_vs_out_config ^= 1; break;
   case 40: a->ngg.has_attribute_ring_relocations = false; break;
   case 41: a->ngg.attribute_ring_relocations[0] ^= 2; break;
   case 42: a->ngg.attribute_ring_relocations[1] ^= 2; break;
   case 43: a->ngg.registers[0].reg ^= 4; break;
   case 44: a->ngg.registers[1].source = AGC_SHADER_REGISTER_GUEST_USER_DATA_ADDRESS; break;
   case 45: a->ngg.registers[2].reg ^= 4; break;
   case 46: a->ngg.registers[3].source = AGC_SHADER_REGISTER_CODE_ADDRESS; break;
   case 47: s->externals[0].pc ^= 4; break;
   case 48: s->externals[0].call = false; break;
   case 49: s->externals[0].has_sgpr = false; break;
   case 50: s->externals[0].sgpr ^= 1; break;
   case 51: s->externals[1].has_sgpr = true; break;
   case 52: a->external_target_count--; break;
   case 53: s->uses[0].pc ^= 4; break;
   case 54: s->uses[0].kind = AGC_SHADER_RESOURCE_BUFFER_DESCRIPTOR; break;
   case 55: s->uses[0].base_sgpr ^= 1; break;
   case 56: s->uses[0].dword_count ^= 1; break;
   case 57: s->uses[0].access = AGC_SHADER_ACCESS_LOADED; break;
   case 58: s->uses[0].source.first ^= 1; break;
   case 59: s->uses[0].source.kind = AGC_SHADER_SOURCE_DYNAMIC; break;
   case 60: s->uses[0].has_buffer_access = true; break;
   case 61: s->uses[1].has_buffer_access = false; break;
   case 62: s->uses[1].buffer_access.scalar = true; break;
   case 63: s->uses[1].buffer_access.indexed = false; break;
   case 64: s->uses[1].buffer_access.vector_offset = false; break;
   case 65: s->uses[1].buffer_access.immediate_offset ^= 1; break;
   case 66: s->uses[1].buffer_access.scalar_offset.bound ^= 1; break;
   case 67: s->uses[1].buffer_access.scalar_offset.kind = AGC_SHADER_SCALAR_OFFSET_UNKNOWN; break;
   case 68: s->uses[3].buffer_access.scalar_offset.sgpr ^= 1; break;
   case 69: s->uses[1].source.kind = AGC_SHADER_SOURCE_TABLE_CHOICES; break;
   case 70: s->single[0].root ^= 1; break;
   case 71: s->single[0].root_form = AGC_SHADER_ADDRESS_POINTER; break;
   case 72: s->single[0].offset = -s->single[0].offset; break;
   case 73: s->single[0].link_count--; break;
   case 74: s->links[0].offset ^= INT64_C(1) << 40; break;
   case 75: s->links[1].form = AGC_SHADER_ADDRESS_POINTER; break;
   case 76: s->uses[2].source.table_count--; break;
   case 77: s->choices[1].offset--; break;
   case 78: s->uses[2].source.kind = AGC_SHADER_SOURCE_TABLE; s->uses[2].source.table_count = 1; break;
   case 79: a->descriptor_use_count--; break;
   case 80: s->pointer_uses[0].pc ^= 4; break;
   case 81: a->pointer_use_count = 0; break;
   case 82: a->has_buffer_fill = false; break;
   case 83: a->buffer_fill.buffer ^= 1; break;
   case 84: a->buffer_fill.value ^= 1; break;
   default: return false;
   }
   return true;
}

#define ARTIFACT_MUTATIONS 85

static void
test_artifact_fields(void)
{
   struct artifact_scratch base_scratch;
   struct agc_shader_artifact base = scratch_artifact(&base_scratch);
   uint8_t **seen = calloc(ARTIFACT_MUTATIONS + 1, sizeof(*seen));
   size_t *seen_size = calloc(ARTIFACT_MUTATIONS + 1, sizeof(*seen_size));
   unsigned count = 0;

   CHECK_EQ(agc_shader_artifact_serialize(&base, &seen[0], &seen_size[0]), OK);
   for (unsigned index = 0;; index++) {
      struct artifact_scratch scratch;
      struct agc_shader_artifact artifact = scratch_artifact(&scratch);
      struct agc_shader_artifact back;
      uint8_t *data;
      size_t size;

      CHECK(artifact_equal(&base, &artifact));
      if (!mutate_artifact(&artifact, &scratch, index))
         break;
      count++;
      CHECK(count <= ARTIFACT_MUTATIONS);
      if (count > ARTIFACT_MUTATIONS)
         break;
      CHECK(!artifact_equal(&base, &artifact));
      CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
      for (unsigned other = 0; other < count; other++)
         CHECK(size != seen_size[other] || memcmp(data, seen[other], size) != 0);
      seen[count] = data;
      seen_size[count] = size;
      CHECK_EQ(agc_shader_artifact_deserialize(data, size, &back), OK);
      CHECK(artifact_equal(&artifact, &back));
      agc_shader_artifact_free(&back);
   }
   CHECK_EQ(count, ARTIFACT_MUTATIONS);
   for (unsigned i = 0; i <= ARTIFACT_MUTATIONS; i++)
      free(seen[i]);
   free(seen);
   free(seen_size);
}

static void
test_artifact_canonical(void)
{
   struct artifact_scratch scratch;
   struct agc_shader_artifact clean = minimal_artifact();
   struct agc_shader_artifact dirty = minimal_artifact();
   struct agc_shader_artifact artifact;
   struct agc_shader_artifact back;
   uint8_t *a;
   uint8_t *b;
   size_t a_size;
   size_t b_size;

   dirty.user_data_tail_sgpr = 0x5a;
   dirty.spi_shader_col_format = 0x5a5a5a5a;
   dirty.ngg = full_artifact().ngg;
   dirty.buffer_fill.buffer = 0x5a;
   dirty.buffer_fill.value = 0x5a;
   dirty.error = error_artifact(AGC_SHADER_ERROR_NONE, "ignored").error;
   dirty.code = code_words;
   dirty.bindings = code_bindings;
   dirty.descriptor_uses = code_uses;
   dirty.storage = &dirty;
   CHECK_EQ(agc_shader_artifact_serialize(&clean, &a, &a_size), OK);
   CHECK_EQ(agc_shader_artifact_serialize(&dirty, &b, &b_size), OK);
   CHECK(a_size == b_size && memcmp(a, b, a_size) == 0);
   free(a);
   free(b);

   artifact = scratch_artifact(&scratch);
   scratch.uses[0].source.tables = table_choices;
   scratch.uses[0].source.table_count = 3;
   scratch.uses[4].source.first = 0x5a;
   scratch.uses[4].source.tables = table_single;
   scratch.uses[4].source.table_count = 1;
   scratch.uses[0].buffer_access = code_uses[1].buffer_access;
   scratch.uses[1].buffer_access.scalar_offset.sgpr = 0x5a;
   scratch.uses[3].buffer_access.scalar_offset.bound = 0x5a5a5a5a;
   scratch.uses[4].buffer_access.scalar_offset.bound = 0x5a5a5a5a;
   scratch.uses[4].buffer_access.scalar_offset.sgpr = 0x5a;
   scratch.externals[1].sgpr = 0x5a;
   clean = full_artifact();
   CHECK_EQ(agc_shader_artifact_serialize(&clean, &a, &a_size), OK);
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &b, &b_size), OK);
   CHECK(a_size == b_size && memcmp(a, b, a_size) == 0);
   CHECK_EQ(agc_shader_artifact_deserialize(b, b_size, &back), OK);
   CHECK(back.descriptor_uses[0].source.tables == NULL && back.descriptor_uses[0].source.table_count == 0);
   CHECK_EQ(back.descriptor_uses[0].buffer_access.immediate_offset, 0);
   CHECK_EQ(back.descriptor_uses[4].source.first, 0);
   CHECK_EQ(back.descriptor_uses[1].buffer_access.scalar_offset.sgpr, 0);
   CHECK_EQ(back.descriptor_uses[3].buffer_access.scalar_offset.bound, 0);
   CHECK_EQ(back.external_targets[1].sgpr, 0);
   agc_shader_artifact_free(&back);
   free(a);
   free(b);

#define UNREPRESENTABLE(change)                                                                              \
   do {                                                                                                      \
      artifact = scratch_artifact(&scratch);                                                                 \
      change;                                                                                                \
      CHECK_EQ(agc_shader_artifact_serialize(&artifact, &a, &a_size), AGC_SHADER_FILE_UNREPRESENTABLE);     \
      CHECK(a == NULL && a_size == 0);                                                                       \
   } while (0)

   UNREPRESENTABLE(artifact.program_words = artifact.code_count + 1);
   UNREPRESENTABLE(artifact.ngg.attribute_ring_relocations[0] = artifact.code_count);
   UNREPRESENTABLE(artifact.ngg.attribute_ring_relocations[1] = artifact.code_count);
   UNREPRESENTABLE(artifact.code = NULL);
   UNREPRESENTABLE(artifact.bindings = NULL);
   UNREPRESENTABLE(artifact.preconverted_descriptors = NULL);
   UNREPRESENTABLE(artifact.external_targets = NULL);
   UNREPRESENTABLE(artifact.descriptor_uses = NULL);
   UNREPRESENTABLE(artifact.pointer_uses = NULL);
   UNREPRESENTABLE(artifact.options.target = AGC_SHADER_ISA_COUNT);
   UNREPRESENTABLE(artifact.options.stage = AGC_SHADER_STAGE_COUNT);
   UNREPRESENTABLE(artifact.options.ray_unit = AGC_SHADER_RAY_UNIT_COUNT);
   UNREPRESENTABLE(artifact.precision = AGC_SHADER_PRECISION_COUNT);
   UNREPRESENTABLE(scratch.bindings[2].kind = AGC_SHADER_RESOURCE_KIND_COUNT);
   UNREPRESENTABLE(scratch.preconverted[0].kind = AGC_SHADER_RESOURCE_KIND_COUNT);
   UNREPRESENTABLE(artifact.ngg.registers[3].source = AGC_SHADER_REGISTER_SOURCE_COUNT);
   UNREPRESENTABLE(scratch.uses[4].kind = AGC_SHADER_RESOURCE_KIND_COUNT);
   UNREPRESENTABLE(scratch.uses[4].access = AGC_SHADER_ACCESS_COUNT);
   UNREPRESENTABLE(scratch.uses[4].source.kind = AGC_SHADER_SOURCE_KIND_COUNT);
   UNREPRESENTABLE(scratch.uses[4].buffer_access.scalar_offset.kind = AGC_SHADER_SCALAR_OFFSET_KIND_COUNT);
   UNREPRESENTABLE(scratch.uses[1].source.table_count = 0);
   UNREPRESENTABLE(scratch.uses[1].source.table_count = 2);
   UNREPRESENTABLE(scratch.uses[1].source.tables = NULL);
   UNREPRESENTABLE(scratch.uses[2].source.tables = NULL);
   UNREPRESENTABLE(scratch.single[0].links = NULL);
   UNREPRESENTABLE(scratch.single[0].root_form = AGC_SHADER_ADDRESS_FORM_COUNT);
   UNREPRESENTABLE(scratch.links[1].form = AGC_SHADER_ADDRESS_FORM_COUNT);
   UNREPRESENTABLE(scratch.pointer_uses[0].kind = AGC_SHADER_RESOURCE_KIND_COUNT);
#undef UNREPRESENTABLE

   artifact = scratch_artifact(&scratch);
   artifact.has_ngg = false;
   artifact.ngg.registers[0].source = AGC_SHADER_REGISTER_SOURCE_COUNT;
   artifact.ngg.attribute_ring_relocations[0] = 0xffffffff;
   scratch.uses[0].buffer_access.scalar_offset.kind = AGC_SHADER_SCALAR_OFFSET_KIND_COUNT;
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &a, &a_size), OK);
   free(a);
}

static void
test_artifact_rejects(void)
{
   static const size_t full_counts[] = {132, 136, 140, 144, 148, 152};
   struct agc_shader_artifact artifact = full_artifact();
   uint8_t *data;
   uint8_t *work;
   size_t size;
   size_t use;
   size_t table;

   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   check_rejections(FILE_ARTIFACT, data, size, full_counts, COUNT_OF(full_counts));
   CHECK_EQ(parse_status(FILE_REQUEST, data, size), AGC_SHADER_FILE_BAD_MAGIC);
   work = malloc(size);

#define MALFORMED(change)                                                                                    \
   do {                                                                                                      \
      memcpy(work, data, size);                                                                              \
      change;                                                                                                \
      reseal(work, size);                                                                                    \
      CHECK_EQ(parse_status(FILE_ARTIFACT, work, size), AGC_SHADER_FILE_MALFORMED);                         \
   } while (0)

   MALFORMED(work[17] = 1);
   MALFORMED(work[19] = 1);
   MALFORMED(work[20] = AGC_SHADER_ISA_COUNT);
   MALFORMED(work[21] = AGC_SHADER_STAGE_COUNT);
   MALFORMED(work[23] = AGC_SHADER_RAY_UNIT_COUNT);
   MALFORMED(work[24] = AGC_SHADER_PRECISION_COUNT);
   MALFORMED(work[26] = 2);
   MALFORMED(work[27] = 0x3f);
   MALFORMED(work[27] = 0x1f & ~0x04);
   MALFORMED(work[27] = 0x1f & ~0x01);
   MALFORMED(work[27] = 0x1f & ~0x02);
   MALFORMED(work[27] = 0x1f & ~0x08);
   MALFORMED(work[27] = 0x1f & ~0x10);
   MALFORMED(work[74] = 2);
   MALFORMED(work[75] = 2);
   MALFORMED(work[76] = 2);
   MALFORMED(wr32(work + 48, 9));
   MALFORMED(wr32(work + 92, 8));
   MALFORMED(wr32(work + 96, 8));
   MALFORMED(work[104] = AGC_SHADER_REGISTER_SOURCE_COUNT);
   MALFORMED(work[105] = 1);
   MALFORMED(work[131] = 1);
   MALFORMED(work[188] = AGC_SHADER_RESOURCE_KIND_COUNT);
   MALFORMED(work[191] = 2);
   MALFORMED(work[200] = AGC_SHADER_RESOURCE_KIND_COUNT);
   MALFORMED(work[208] = 2);
   MALFORMED(work[209] = 2);
   MALFORMED(work[211] = 1);
   MALFORMED(work[218] = 1);

   use = 220;
   MALFORMED(work[use + 4] = AGC_SHADER_RESOURCE_KIND_COUNT);
   MALFORMED(work[use + 7] = AGC_SHADER_ACCESS_COUNT);
   MALFORMED(work[use + 8] = AGC_SHADER_SOURCE_KIND_COUNT);
   MALFORMED(work[use + 8] = AGC_SHADER_SOURCE_DYNAMIC);
   MALFORMED(work[use + 8] = AGC_SHADER_SOURCE_TABLE);
   MALFORMED(work[use + 10] = 2);
   MALFORMED(work[use + 11] = 1);
   MALFORMED(work[use + 12] = 1);
   MALFORMED(work[use + 13] = 1);
   MALFORMED(work[use + 14] = AGC_SHADER_SCALAR_OFFSET_UNKNOWN);
   MALFORMED(work[use + 15] = 1);
   MALFORMED(wr32(work + use + 16, 1));
   MALFORMED(wr32(work + use + 20, 1));
   MALFORMED(wr32(work + use + 24, 1));

   use += 28;
   table = use + 28;
   MALFORMED(work[use + 9] = 1);
   MALFORMED(work[use + 14] = AGC_SHADER_SCALAR_OFFSET_KIND_COUNT);
   MALFORMED(work[use + 14] = AGC_SHADER_SCALAR_OFFSET_UNKNOWN);
   MALFORMED(work[use + 14] = AGC_SHADER_SCALAR_OFFSET_USER_SGPR);
   MALFORMED(wr32(work + use + 24, 0));
   MALFORMED(wr32(work + use + 24, 2));
   MALFORMED(work[table + 13] = AGC_SHADER_ADDRESS_FORM_COUNT);
   MALFORMED(work[table + 14] = 1);
   MALFORMED(work[table + 15] = 1);
   MALFORMED(work[table + 24] = AGC_SHADER_ADDRESS_FORM_COUNT);
   MALFORMED(work[table + 25] = 1);
   MALFORMED(work[table + 27] = 1);
#undef MALFORMED

   for (uint32_t links = 0; links < 8; links++) {
      if (links == 2)
         continue;
      memcpy(work, data, size);
      wr32(work + table + 8, links);
      reseal(work, size);
      enum agc_shader_file_status status = parse_status(FILE_ARTIFACT, work, size);
      CHECK(status == AGC_SHADER_FILE_TRUNCATED || status == AGC_SHADER_FILE_MALFORMED);
   }
   memcpy(work, data, size);
   wr32(work + table + 8, 0xffffffff);
   reseal(work, size);
   CHECK_EQ(parse_status(FILE_ARTIFACT, work, size), AGC_SHADER_FILE_TRUNCATED);
   memcpy(work, data, size);
   wr32(work + use + 24 + 28 + 40, 0x10000000);
   reseal(work, size);
   CHECK_EQ(parse_status(FILE_ARTIFACT, work, size), AGC_SHADER_FILE_TRUNCATED);
   memcpy(work, data, size);
   wr32(work + 4, AGC_SHADER_ARTIFACT_VERSION + 1);
   CHECK_EQ(parse_status(FILE_ARTIFACT, work, size), AGC_SHADER_FILE_BAD_VERSION);
   free(work);
   free(data);

   artifact = minimal_artifact();
   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   check_rejections(FILE_ARTIFACT, data, size, full_counts, COUNT_OF(full_counts));
   for (size_t at = 77; at < 132; at++) {
      data[at] = 1;
      reseal(data, size);
      CHECK_EQ(parse_status(FILE_ARTIFACT, data, size), AGC_SHADER_FILE_MALFORMED);
      data[at] = 0;
   }
   free(data);

   CHECK_EQ(agc_shader_artifact_deserialize(NULL, 0, &artifact), AGC_SHADER_FILE_TRUNCATED);
   CHECK(artifact.storage == NULL && artifact.code_count == 0);
}

static bool
file_exists(const char *path)
{
   FILE *file = fopen(path, "rb");
   if (file)
      fclose(file);
   return file != NULL;
}

static void
write_bytes(const char *path, const void *data, size_t size)
{
   FILE *file = fopen(path, "wb");
   CHECK(file != NULL);
   if (!file)
      return;
   CHECK_EQ(fwrite(data, 1, size, file), size);
   fclose(file);
}

static void
test_cache(void)
{
   const char *base = getenv("TEMP");
   struct agc_shader_request request = full_request();
   struct agc_shader_request request_back;
   struct agc_shader_artifact artifact = full_artifact();
   struct agc_shader_artifact failure = unsupported_artifact();
   struct agc_shader_artifact back;
   struct agc_shader_cache *cache = NULL;
   struct agc_shader_cache *second = NULL;
   struct agc_shader_key key;
   struct agc_shader_key other;
   char hex[AGC_SHADER_KEY_HEX_SIZE];
   char root[512];
   char outer[640];
   char directory[768];
   char requests[896];
   char art_path[1024];
   char req_path[1024];
   char plain_path[1024];
   uint8_t *data;
   size_t size;

   if (!base)
      base = getenv("TMPDIR");
   if (!base)
      base = "/tmp";
   snprintf(root, sizeof(root), "%s/agc-shader-artifact-test-%lx-%lx", base, (unsigned long)test_getpid(),
            (unsigned long)time(NULL));
   snprintf(outer, sizeof(outer), "%s/nested", root);
   snprintf(directory, sizeof(directory), "%s/cache", outer);
   snprintf(requests, sizeof(requests), "%s/%s", directory, AGC_SHADER_REQUEST_DIRECTORY);

   CHECK_EQ(agc_shader_request_key(&request, &key), OK);
   agc_shader_key_hex(&key, hex);
   snprintf(art_path, sizeof(art_path), "%s/%s%s", directory, hex, AGC_SHADER_ARTIFACT_EXTENSION);
   snprintf(req_path, sizeof(req_path), "%s/%s%s", requests, hex, AGC_SHADER_REQUEST_EXTENSION);
   snprintf(plain_path, sizeof(plain_path), "%s/plain", root);

   CHECK_EQ(agc_shader_cache_open(directory, &cache), OK);
   CHECK(cache != NULL);
   if (!cache)
      return;
   snprintf(outer, sizeof(outer), "%s/nested/cache//", root);
   CHECK_EQ(agc_shader_cache_open(outer, &second), OK);
   snprintf(outer, sizeof(outer), "%s/nested", root);

   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), AGC_SHADER_FILE_NOT_FOUND);
   CHECK(back.storage == NULL && back.code == NULL);
   CHECK(!file_exists(art_path) && !file_exists(req_path));

   CHECK_EQ(agc_shader_cache_miss(cache, &key, &request), OK);
   CHECK(file_exists(req_path));
   CHECK_EQ(agc_shader_request_read_file(req_path, &request_back), OK);
   CHECK(request_equal(&request, &request_back));
   CHECK_EQ(agc_shader_request_key(&request_back, &other), OK);
   CHECK(memcmp(&key, &other, sizeof(key)) == 0);
   agc_shader_request_free(&request_back);
   CHECK_EQ(agc_shader_cache_miss(cache, &key, &request), OK);
   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), AGC_SHADER_FILE_NOT_FOUND);

   other = key;
   other.bytes[0] ^= 1;
   CHECK_EQ(agc_shader_cache_miss(cache, &other, &request), AGC_SHADER_FILE_KEY_MISMATCH);
   request.options.target = AGC_SHADER_ISA_COUNT;
   CHECK_EQ(agc_shader_cache_miss(cache, &key, &request), AGC_SHADER_FILE_UNREPRESENTABLE);
   request = full_request();

   CHECK_EQ(agc_shader_cache_store(cache, &key, &failure), OK);
   CHECK(file_exists(art_path));
   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), OK);
   CHECK(artifact_equal(&failure, &back));
   CHECK_EQ(back.error.kind, AGC_SHADER_ERROR_UNSUPPORTED_INSTRUCTION);
   CHECK(back.error.reason && strcmp(back.error.reason, "no gfx12 form") == 0);
   agc_shader_artifact_free(&back);

   CHECK_EQ(agc_shader_cache_store(second, &key, &artifact), OK);
   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), OK);
   CHECK(artifact_equal(&artifact, &back));
   agc_shader_artifact_free(&back);
   CHECK_EQ(agc_shader_cache_lookup(cache, &other, &back), AGC_SHADER_FILE_NOT_FOUND);

   CHECK_EQ(agc_shader_artifact_serialize(&artifact, &data, &size), OK);
   write_bytes(art_path, data, size - 1);
   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), AGC_SHADER_FILE_TRUNCATED);
   CHECK(back.storage == NULL);
   data[size - 1] ^= 1;
   write_bytes(art_path, data, size);
   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), AGC_SHADER_FILE_BAD_CHECKSUM);
   write_bytes(art_path, "", 0);
   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), AGC_SHADER_FILE_TRUNCATED);
   memset(data, 0, size);
   write_bytes(art_path, data, size);
   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), AGC_SHADER_FILE_BAD_MAGIC);
   free(data);
   CHECK_EQ(agc_shader_request_serialize(&request, &data, &size), OK);
   write_bytes(art_path, data, size);
   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), AGC_SHADER_FILE_BAD_MAGIC);
   free(data);
   CHECK_EQ(agc_shader_cache_store(cache, &key, &artifact), OK);
   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), OK);
   CHECK(artifact_equal(&artifact, &back));
   agc_shader_artifact_free(&back);

   FILE *reader = fopen(art_path, "rb");
   CHECK(reader != NULL);
   CHECK_EQ(agc_shader_cache_store(cache, &key, &artifact), OK);
#if DETECT_OS_WINDOWS
   CHECK_EQ(agc_shader_cache_store(cache, &key, &failure), AGC_SHADER_FILE_IO_ERROR);
#else
   CHECK_EQ(agc_shader_cache_store(cache, &key, &failure), OK);
   CHECK_EQ(agc_shader_cache_store(cache, &key, &artifact), OK);
#endif
   if (reader)
      fclose(reader);
   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), OK);
   CHECK(artifact_equal(&artifact, &back));
   agc_shader_artifact_free(&back);

   artifact.program_words = artifact.code_count + 1;
   CHECK_EQ(agc_shader_cache_store(cache, &other, &artifact), AGC_SHADER_FILE_UNREPRESENTABLE);
   CHECK_EQ(agc_shader_cache_lookup(cache, &other, &back), AGC_SHADER_FILE_NOT_FOUND);
   artifact = full_artifact();

   CHECK_EQ(agc_shader_artifact_write_file(&artifact, plain_path), OK);
   CHECK_EQ(agc_shader_artifact_read_file(plain_path, &back), OK);
   CHECK(artifact_equal(&artifact, &back));
   agc_shader_artifact_free(&back);
   CHECK_EQ(agc_shader_request_read_file(plain_path, &request_back), AGC_SHADER_FILE_BAD_MAGIC);
   CHECK_EQ(agc_shader_request_write_file(&request, plain_path), OK);
   CHECK_EQ(agc_shader_request_read_file(plain_path, &request_back), OK);
   CHECK(request_equal(&request, &request_back));
   agc_shader_request_free(&request_back);
   CHECK_EQ(agc_shader_artifact_read_file(plain_path, &back), AGC_SHADER_FILE_BAD_MAGIC);
   agc_shader_cache_close(second);
   CHECK_EQ(agc_shader_cache_open(plain_path, &second), AGC_SHADER_FILE_IO_ERROR);
   CHECK(second == NULL);
   CHECK_EQ(remove(plain_path), 0);
   CHECK_EQ(agc_shader_artifact_read_file(plain_path, &back), AGC_SHADER_FILE_NOT_FOUND);
   CHECK_EQ(agc_shader_request_read_file(plain_path, &request_back), AGC_SHADER_FILE_NOT_FOUND);
   snprintf(plain_path, sizeof(plain_path), "%s/missing/plain", root);
   CHECK(agc_shader_artifact_write_file(&artifact, plain_path) == AGC_SHADER_FILE_IO_ERROR);
   CHECK(agc_shader_request_write_file(&request, plain_path) == AGC_SHADER_FILE_IO_ERROR);

   agc_shader_cache_close(cache);
   agc_shader_cache_close(NULL);
   CHECK_EQ(agc_shader_cache_open(directory, &cache), OK);
   CHECK_EQ(agc_shader_cache_lookup(cache, &key, &back), OK);
   CHECK(artifact_equal(&artifact, &back));
   agc_shader_artifact_free(&back);
   agc_shader_cache_close(cache);
   CHECK_EQ(agc_shader_cache_open("", &cache), AGC_SHADER_FILE_IO_ERROR);
   CHECK(cache == NULL);

   CHECK_EQ(remove(art_path), 0);
   CHECK_EQ(remove(req_path), 0);
   CHECK_EQ(test_rmdir(requests), 0);
   CHECK_EQ(test_rmdir(directory), 0);
   CHECK_EQ(test_rmdir(outer), 0);
   CHECK_EQ(test_rmdir(root), 0);
}

int
main(void)
{
   test_names();
   test_request_layout();
   test_request_round_trip();
   test_request_canonical();
   test_request_rejects();
   test_request_key();
   test_artifact_layout();
   test_artifact_round_trip();
   test_artifact_errors();
   test_artifact_fields();
   test_artifact_canonical();
   test_artifact_rejects();
   test_cache();
   return TEST_RESULT();
}
