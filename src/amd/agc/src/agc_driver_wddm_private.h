#ifndef AGC_DRIVER_WDDM_PRIVATE_H
#define AGC_DRIVER_WDDM_PRIVATE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "agc_driver_ops.h"

#define AGC_WDDM_ABI_VERSION 9u
#define AGC_WDDM_ABI_VERSION_D3D12 8u

#define AGC_WDDM_CONTEXT_HEADER_BYTES 0x40u
#define AGC_WDDM_CONTEXT_SECTION_BYTES 0x168u
#define AGC_WDDM_CONTEXT_FLAGS_COMPUTE 0x100000u
#define AGC_WDDM_CONTEXT_FLAGS_GFX 0x140000u
#define AGC_WDDM_CONTEXT_DW13 0x200000u

#define AGC_WDDM_QUEUE_FLAGS 0x21000u

#define AGC_WDDM_ALLOC_CREATE_FLAGS 0x80u
#define AGC_WDDM_ALLOC_SECTION_METADATA 0x1u

#define AGC_WDDM_BO_HEAP_BIT(heap) (1u << ((heap) - 1))
#define AGC_WDDM_BO_NOT_VIRTUAL 0x2000u
#define AGC_WDDM_BO_HOST_ALLOCATED 0x4000000u
#define AGC_WDDM_BO_CPU_VISIBLE 0x20000000u
#define AGC_WDDM_BO_FLAGS2 0x1000000u
#define AGC_WDDM_BO_PRIORITY 0x5u
#define AGC_WDDM_BO_PRIORITY_HOST 0x800u

#define AGC_WDDM_MTYPE_GL2_CACHED 0u
#define AGC_WDDM_MTYPE_GL2_BYPASS 4u
#define AGC_WDDM_MALL_POLICY_DEFAULT 0u
#define AGC_WDDM_MALL_POLICY_NEVER 1u

#define AGC_WDDM_SUBMIT_ENTRY_GFX_IB 0u
#define AGC_WDDM_SUBMIT_IP_GFX 1u

#define AGC_WDDM_UMD_BASIC_QUERY_BYTES 3008u
#define AGC_WDDM_UMD_GPU_QUERY_BYTES 17248u
#define AGC_WDDM_UMD_NODE_GFX 0x00
#define AGC_WDDM_UMD_NODE_COMPUTE 0x04
#define AGC_WDDM_UMD_NODE_SDMA 0x05

#pragma pack(push, 4)

struct agc_wddm_context_data {
   uint32_t header_size;
   uint32_t flags;
   uint32_t reserved0[11];
   uint32_t version;
   uint32_t dw13;
   uint32_t dw14;
   uint32_t section_size;
   uint32_t reserved1[89];
};

struct agc_wddm_queue_data {
   uint32_t header_size;
   uint32_t reserved0[3];
   uint32_t flags;
   uint32_t queue_id;
   uint32_t parent_queue_id;
   uint32_t reserved1[9];
};

struct agc_wddm_alloc_create_data {
   uint32_t adapter_id;
   uint32_t dw1;
   uint32_t flags;
   uint32_t checksum;
   uint32_t reserved[11];
   uint32_t records_size;
};

struct agc_wddm_alloc_header {
   uint32_t entry_size[4];
   uint32_t adapter_id;
   uint32_t checksum;
   uint32_t reserved[9];
   uint32_t entry_count;
};

struct agc_wddm_alloc_bo_info {
   uint32_t section_size;
   uint32_t dw1;
   uint32_t flags;
   uint32_t size;
   uint32_t alignment;
   uint32_t priority;
   uint8_t heaps[4];
   uint32_t unknown[16];
   uint32_t flags2;
   uint64_t physical_size;
   uint64_t va_address;
   uint64_t va_size;
   uint32_t reserved[2];
};

struct agc_wddm_alloc_entry {
   uint32_t entry_size;
   uint32_t enabled_sections;
   struct agc_wddm_alloc_bo_info bo_info;
   uint32_t num_planes;
   uint32_t unknown0[2];
   uint16_t samples;
   uint16_t mask;
   uint32_t mip_levels;
   uint32_t layers;
   uint32_t unknown1[4];
   uint32_t metadata_offset;
   uint32_t unknown2[2];
   uint32_t version;
   uint32_t unknown3;
};

struct agc_wddm_alloc_metadata {
   uint32_t section_size;
   uint32_t pad;
   uint64_t va_address;
   uint64_t va_size;
   uint32_t unknown0[45];
   uint32_t mtype;
   uint32_t flags;
   uint32_t mall_policy;
   uint32_t mall_range[2];
   uint32_t unknown1[13];
};

struct agc_wddm_alloc_records {
   struct agc_wddm_alloc_header header;
   struct agc_wddm_alloc_entry entry;
   struct agc_wddm_alloc_metadata metadata;
};

struct agc_wddm_submit_header {
   uint32_t version;
   uint32_t entry_count;
};

struct agc_wddm_submit_ib {
   uint32_t type;
   uint32_t section_size;
   uint32_t length;
   uint32_t flags;
   uint32_t dw4;
   uint32_t ip_type;
   uint32_t address_low;
   uint32_t address_high;
   uint32_t reserved[8];
};

struct agc_wddm_submit_data {
   struct agc_wddm_submit_header header;
   struct agc_wddm_submit_ib ibs[AGC_SUBMIT_MAX_IBS];
};

#pragma pack(pop)

#pragma pack(push, 1)

struct agc_wddm_umd_header {
   uint32_t flags[3];
   uint32_t vendor_id;
   uint32_t unknown0;
   uint32_t pci_id;
   uint32_t unknown1[9];
   uint16_t registry_path[100];
   uint32_t padding[490];
   uint32_t type;
   uint32_t something;
   uint32_t unknown2[42];
   uint32_t size;
};

struct agc_wddm_umd_basic {
   struct agc_wddm_umd_header header;
   uint32_t unknown0[11];
   uint32_t device_id;
   uint32_t family_id;
   uint32_t chip_external_rev;
   uint32_t subsystem_id;
   uint32_t unknown1[6];
   uint32_t vram_type;
   uint32_t vram_bit_width;
   uint32_t unknown2[43];
   char umd_registry_path[256];
   uint32_t unknown3[18];
   uint32_t num_rb_pipes;
   uint32_t num_enabled_rb_pipes;
   uint32_t enabled_rb_pipes_mask;
   uint32_t enabled_rb_pipes_mask_hi;
   uint32_t unknown4[8];
};

struct agc_wddm_umd_ib_alignment {
   uint32_t start;
   uint32_t size;
};

struct agc_wddm_umd_gpu {
   struct agc_wddm_umd_header header;
   uint32_t unknown0[550];
   uint32_t num_engines;
   char engines[64];
   uint32_t unknown1[5];
   uint32_t me_fw_version;
   uint32_t mec_fw_version;
   uint32_t pfp_fw_version;
   uint32_t unknown3[65];
   uint32_t device_id;
   uint32_t family_id;
   uint32_t chip_external_rev;
   uint32_t chip_rev;
   uint32_t gfx_engine_id;
   uint32_t unknown4[4];
   uint32_t clock_crystal_freq_hz;
   uint32_t unknown5;
   uint32_t unknown6;
   uint64_t vram_vis_size;
   uint64_t vram_inv_size;
   uint32_t unknown7[112];
   uint32_t gb_addr_cfg;
   uint32_t unknown8[21];
   uint32_t num_shader_visible_vgprs;
   uint32_t wave_front_size;
   uint32_t num_shader_engines;
   uint32_t num_active_rb_per_se;
   uint32_t num_shader_arrays_per_engine;
   uint32_t unknown9[2];
   uint32_t num_tcc_blocks;
   uint32_t max_gs_waves_per_vgt;
   uint32_t unknown10;
   uint32_t gs_vgt_table_depth;
   uint32_t gs_prim_buffer_depth;
   uint32_t unknown11[23];
   uint32_t num_cu_per_wgp;
   uint32_t unknown12[7];
   uint16_t wgp_bitmap[6][2];
   uint16_t wgp_ao_bitmap[6][2];
   uint32_t unknown13[10];
   uint32_t num_sqc_per_wgp;
   uint32_t sqc_inst_cache_size;
   uint32_t sqc_data_cache_size;
   uint32_t gl1c_cache_size;
   uint32_t unknown14;
   uint32_t gl2c_cache_size;
   uint32_t unknown15[4];
   uint32_t mall_size_mb;
   uint32_t unknown16[1824];
   uint64_t va_start;
   uint64_t va_end;
   uint32_t unknown17[256];
   uint32_t large_page;
   uint32_t unknown18[220];
   struct agc_wddm_umd_ib_alignment ib_alignments[8];
   uint32_t unknown19[525];
};

#pragma pack(pop)

_Static_assert(sizeof(struct agc_wddm_context_data) == 424, "context record");
_Static_assert(sizeof(struct agc_wddm_context_data) ==
                  AGC_WDDM_CONTEXT_HEADER_BYTES + AGC_WDDM_CONTEXT_SECTION_BYTES,
               "context record sections");
_Static_assert(offsetof(struct agc_wddm_context_data, header_size) == 0, "context header_size");
_Static_assert(offsetof(struct agc_wddm_context_data, flags) == 4, "context flags");
_Static_assert(offsetof(struct agc_wddm_context_data, version) == 52, "context version");
_Static_assert(offsetof(struct agc_wddm_context_data, dw13) == 56, "context dw13");
_Static_assert(offsetof(struct agc_wddm_context_data, section_size) == 64, "context section_size");

_Static_assert(sizeof(struct agc_wddm_queue_data) == 64, "queue record");
_Static_assert(offsetof(struct agc_wddm_queue_data, header_size) == 0, "queue header_size");
_Static_assert(offsetof(struct agc_wddm_queue_data, flags) == 16, "queue flags");
_Static_assert(offsetof(struct agc_wddm_queue_data, queue_id) == 20, "queue queue_id");
_Static_assert(offsetof(struct agc_wddm_queue_data, parent_queue_id) == 24,
               "queue parent_queue_id");

_Static_assert(sizeof(struct agc_wddm_alloc_create_data) == 64, "allocation private record");
_Static_assert(offsetof(struct agc_wddm_alloc_create_data, flags) == 8, "allocation flags");
_Static_assert(offsetof(struct agc_wddm_alloc_create_data, checksum) == 12, "allocation checksum");
_Static_assert(offsetof(struct agc_wddm_alloc_create_data, records_size) == 60,
               "allocation records_size");

_Static_assert(sizeof(struct agc_wddm_alloc_header) == 64, "allocation header");
_Static_assert(offsetof(struct agc_wddm_alloc_header, entry_size) == 0, "header entry_size");
_Static_assert(offsetof(struct agc_wddm_alloc_header, adapter_id) == 16, "header adapter_id");
_Static_assert(offsetof(struct agc_wddm_alloc_header, checksum) == 20, "header checksum");
_Static_assert(offsetof(struct agc_wddm_alloc_header, entry_count) == 60, "header entry_count");

_Static_assert(sizeof(struct agc_wddm_alloc_bo_info) == 128, "buffer object record");
_Static_assert(offsetof(struct agc_wddm_alloc_bo_info, section_size) == 0, "bo section_size");
_Static_assert(offsetof(struct agc_wddm_alloc_bo_info, flags) == 8, "bo flags");
_Static_assert(offsetof(struct agc_wddm_alloc_bo_info, size) == 12, "bo size");
_Static_assert(offsetof(struct agc_wddm_alloc_bo_info, alignment) == 16, "bo alignment");
_Static_assert(offsetof(struct agc_wddm_alloc_bo_info, priority) == 20, "bo priority");
_Static_assert(offsetof(struct agc_wddm_alloc_bo_info, heaps) == 24, "bo heaps");
_Static_assert(offsetof(struct agc_wddm_alloc_bo_info, flags2) == 92, "bo flags2");
_Static_assert(offsetof(struct agc_wddm_alloc_bo_info, physical_size) == 96, "bo physical_size");
_Static_assert(offsetof(struct agc_wddm_alloc_bo_info, va_address) == 104, "bo va_address");
_Static_assert(offsetof(struct agc_wddm_alloc_bo_info, va_size) == 112, "bo va_size");

_Static_assert(sizeof(struct agc_wddm_alloc_entry) == 196, "allocation entry");
_Static_assert(offsetof(struct agc_wddm_alloc_entry, entry_size) == 0, "entry entry_size");
_Static_assert(offsetof(struct agc_wddm_alloc_entry, enabled_sections) == 4, "entry sections");
_Static_assert(offsetof(struct agc_wddm_alloc_entry, bo_info) == 8, "entry bo_info");
_Static_assert(offsetof(struct agc_wddm_alloc_entry, metadata_offset) == 176,
               "entry metadata_offset");
_Static_assert(offsetof(struct agc_wddm_alloc_entry, version) == 188, "entry version");

_Static_assert(sizeof(struct agc_wddm_alloc_metadata) == 276, "allocation metadata");
_Static_assert(offsetof(struct agc_wddm_alloc_metadata, section_size) == 0,
               "metadata section_size");
_Static_assert(offsetof(struct agc_wddm_alloc_metadata, va_address) == 8, "metadata va_address");
_Static_assert(offsetof(struct agc_wddm_alloc_metadata, va_size) == 16, "metadata va_size");
_Static_assert(offsetof(struct agc_wddm_alloc_metadata, mtype) == 204, "metadata mtype");
_Static_assert(offsetof(struct agc_wddm_alloc_metadata, flags) == 208, "metadata flags");
_Static_assert(offsetof(struct agc_wddm_alloc_metadata, mall_policy) == 212,
               "metadata mall_policy");

_Static_assert(sizeof(struct agc_wddm_alloc_records) == 536, "allocation records");
_Static_assert(offsetof(struct agc_wddm_alloc_records, entry) == 64, "records entry");
_Static_assert(offsetof(struct agc_wddm_alloc_records, metadata) == 260, "records metadata");

_Static_assert(sizeof(struct agc_wddm_submit_header) == 8, "submit header");
_Static_assert(offsetof(struct agc_wddm_submit_header, version) == 0, "submit version");
_Static_assert(offsetof(struct agc_wddm_submit_header, entry_count) == 4, "submit entry_count");

_Static_assert(sizeof(struct agc_wddm_submit_ib) == 64, "submit IB record");
_Static_assert(offsetof(struct agc_wddm_submit_ib, type) == 0, "IB type");
_Static_assert(offsetof(struct agc_wddm_submit_ib, section_size) == 4, "IB section_size");
_Static_assert(offsetof(struct agc_wddm_submit_ib, length) == 8, "IB length");
_Static_assert(offsetof(struct agc_wddm_submit_ib, flags) == 12, "IB flags");
_Static_assert(offsetof(struct agc_wddm_submit_ib, ip_type) == 20, "IB ip_type");
_Static_assert(offsetof(struct agc_wddm_submit_ib, address_low) == 24, "IB address_low");
_Static_assert(offsetof(struct agc_wddm_submit_ib, address_high) == 28, "IB address_high");

_Static_assert(sizeof(struct agc_wddm_submit_data) == 8 + 64 * AGC_SUBMIT_MAX_IBS, "submit data");
_Static_assert(offsetof(struct agc_wddm_submit_data, ibs) == 8, "submit IB records");

_Static_assert(sizeof(struct agc_wddm_umd_header) == 2400, "UMD private header");
_Static_assert(offsetof(struct agc_wddm_umd_header, vendor_id) == 12, "UMD vendor_id");
_Static_assert(offsetof(struct agc_wddm_umd_header, pci_id) == 20, "UMD pci_id");
_Static_assert(offsetof(struct agc_wddm_umd_header, registry_path) == 60, "UMD registry_path");
_Static_assert(offsetof(struct agc_wddm_umd_header, type) == 2220, "UMD type");
_Static_assert(offsetof(struct agc_wddm_umd_header, size) == 2396, "UMD size");

_Static_assert(sizeof(struct agc_wddm_umd_basic) == 3040, "UMD basic properties");
_Static_assert(sizeof(struct agc_wddm_umd_basic) >= AGC_WDDM_UMD_BASIC_QUERY_BYTES,
               "UMD basic properties query");
_Static_assert(offsetof(struct agc_wddm_umd_basic, device_id) == 2444, "basic device_id");
_Static_assert(offsetof(struct agc_wddm_umd_basic, family_id) == 2448, "basic family_id");
_Static_assert(offsetof(struct agc_wddm_umd_basic, chip_external_rev) == 2452,
               "basic chip_external_rev");
_Static_assert(offsetof(struct agc_wddm_umd_basic, subsystem_id) == 2456, "basic subsystem_id");
_Static_assert(offsetof(struct agc_wddm_umd_basic, vram_type) == 2484, "basic vram_type");
_Static_assert(offsetof(struct agc_wddm_umd_basic, vram_bit_width) == 2488, "basic vram_bit_width");
_Static_assert(offsetof(struct agc_wddm_umd_basic, umd_registry_path) == 2664,
               "basic umd_registry_path");
_Static_assert(offsetof(struct agc_wddm_umd_basic, num_rb_pipes) == 2992, "basic num_rb_pipes");
_Static_assert(offsetof(struct agc_wddm_umd_basic, enabled_rb_pipes_mask) == 3000,
               "basic enabled_rb_pipes_mask");
_Static_assert(offsetof(struct agc_wddm_umd_basic, enabled_rb_pipes_mask_hi) == 3004,
               "basic enabled_rb_pipes_mask_hi");
_Static_assert(offsetof(struct agc_wddm_umd_basic, unknown4) == AGC_WDDM_UMD_BASIC_QUERY_BYTES,
               "basic properties end");

_Static_assert(sizeof(struct agc_wddm_umd_gpu) == AGC_WDDM_UMD_GPU_QUERY_BYTES,
               "UMD GPU properties");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, num_engines) == 4600, "gpu num_engines");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, engines) == 4604, "gpu engines");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, me_fw_version) == 4688, "gpu me_fw_version");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, mec_fw_version) == 4692, "gpu mec_fw_version");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, pfp_fw_version) == 4696, "gpu pfp_fw_version");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, device_id) == 4960, "gpu device_id");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, family_id) == 4964, "gpu family_id");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, chip_external_rev) == 4968,
               "gpu chip_external_rev");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, chip_rev) == 4972, "gpu chip_rev");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, gfx_engine_id) == 4976, "gpu gfx_engine_id");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, clock_crystal_freq_hz) == 4996,
               "gpu clock_crystal_freq_hz");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, vram_vis_size) == 5008, "gpu vram_vis_size");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, vram_inv_size) == 5016, "gpu vram_inv_size");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, gb_addr_cfg) == 5472, "gpu gb_addr_cfg");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, num_shader_visible_vgprs) == 5560,
               "gpu num_shader_visible_vgprs");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, wave_front_size) == 5564, "gpu wave_front_size");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, num_shader_engines) == 5568,
               "gpu num_shader_engines");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, num_active_rb_per_se) == 5572,
               "gpu num_active_rb_per_se");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, num_shader_arrays_per_engine) == 5576,
               "gpu num_shader_arrays_per_engine");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, num_tcc_blocks) == 5588, "gpu num_tcc_blocks");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, max_gs_waves_per_vgt) == 5592,
               "gpu max_gs_waves_per_vgt");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, gs_vgt_table_depth) == 5600,
               "gpu gs_vgt_table_depth");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, gs_prim_buffer_depth) == 5604,
               "gpu gs_prim_buffer_depth");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, num_cu_per_wgp) == 5700, "gpu num_cu_per_wgp");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, wgp_bitmap) == 5732, "gpu wgp_bitmap");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, wgp_ao_bitmap) == 5756, "gpu wgp_ao_bitmap");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, num_sqc_per_wgp) == 5820, "gpu num_sqc_per_wgp");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, sqc_inst_cache_size) == 5824,
               "gpu sqc_inst_cache_size");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, sqc_data_cache_size) == 5828,
               "gpu sqc_data_cache_size");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, gl1c_cache_size) == 5832, "gpu gl1c_cache_size");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, gl2c_cache_size) == 5840, "gpu gl2c_cache_size");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, mall_size_mb) == 5860, "gpu mall_size_mb");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, va_start) == 13160, "gpu va_start");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, va_end) == 13168, "gpu va_end");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, large_page) == 14200, "gpu large_page");
_Static_assert(offsetof(struct agc_wddm_umd_gpu, ib_alignments) == 15084, "gpu ib_alignments");

_Static_assert((AGC_WDDM_BO_CPU_VISIBLE | AGC_WDDM_BO_NOT_VIRTUAL | AGC_WDDM_BO_HEAP_BIT(4) |
                AGC_WDDM_BO_HEAP_BIT(3)) == 0x2000200cu,
               "device buffer object flags");
_Static_assert((AGC_WDDM_BO_CPU_VISIBLE | AGC_WDDM_BO_HOST_ALLOCATED | AGC_WDDM_BO_NOT_VIRTUAL |
                AGC_WDDM_BO_HEAP_BIT(4)) == 0x24002008u,
               "host buffer object flags");
_Static_assert((AGC_WDDM_BO_PRIORITY | AGC_WDDM_BO_PRIORITY_HOST) == 0x805u,
               "host buffer object priority");
_Static_assert(AGC_IB_PREAMBLE == 0xc, "preamble IB flags");

struct agc_wddm_alloc_params {
   uint32_t size;
   uint32_t flags;
   uint32_t alignment;
   uint32_t priority;
   uint8_t heaps[2];
   uint32_t mtype;
   uint32_t mall_policy;
   uint64_t va_address;
};

static inline uint32_t
agc_wddm_checksum(const void *data, size_t bytes)
{
   uint32_t sum = 0;
   for (size_t i = 0; i < bytes / sizeof(uint32_t); i++) {
      uint32_t word;
      memcpy(&word, (const uint8_t *)data + i * sizeof(uint32_t), sizeof(word));
      sum += (uint32_t)i ^ word;
   }
   return sum;
}

static inline void
agc_wddm_fill_context(struct agc_wddm_context_data *data, bool graphics)
{
   memset(data, 0, sizeof(*data));
   data->header_size = AGC_WDDM_CONTEXT_HEADER_BYTES;
   data->flags = graphics ? AGC_WDDM_CONTEXT_FLAGS_GFX : AGC_WDDM_CONTEXT_FLAGS_COMPUTE;
   data->version = graphics ? AGC_WDDM_ABI_VERSION : 0;
   data->dw13 = AGC_WDDM_CONTEXT_DW13;
   data->section_size = AGC_WDDM_CONTEXT_SECTION_BYTES;
}

static inline void
agc_wddm_fill_queue(struct agc_wddm_queue_data *data)
{
   memset(data, 0, sizeof(*data));
   data->header_size = sizeof(*data);
   data->flags = AGC_WDDM_QUEUE_FLAGS;
}

static inline void
agc_wddm_fill_alloc(struct agc_wddm_alloc_create_data *create,
                    struct agc_wddm_alloc_records *records,
                    const struct agc_wddm_alloc_params *params)
{
   memset(records, 0, sizeof(*records));
   records->header.entry_size[0] = sizeof(records->entry) + sizeof(records->metadata);
   records->header.entry_count = 1;
   records->header.checksum = agc_wddm_checksum(&records->header, sizeof(records->header));

   records->entry.entry_size = sizeof(records->entry) + sizeof(records->metadata);
   records->entry.enabled_sections = AGC_WDDM_ALLOC_SECTION_METADATA;
   records->entry.bo_info.section_size = sizeof(records->entry.bo_info);
   records->entry.bo_info.flags = params->flags;
   records->entry.bo_info.size = params->size;
   records->entry.bo_info.alignment = params->alignment;
   records->entry.bo_info.priority = params->priority;
   records->entry.bo_info.heaps[0] = params->heaps[0];
   records->entry.bo_info.heaps[1] = params->heaps[1];
   records->entry.bo_info.flags2 = AGC_WDDM_BO_FLAGS2;
   records->entry.bo_info.physical_size = params->size;
   records->entry.bo_info.va_address = params->va_address;
   records->entry.bo_info.va_size = params->size;
   records->entry.metadata_offset = sizeof(records->entry);
   records->entry.version = AGC_WDDM_ABI_VERSION;

   records->metadata.section_size = sizeof(records->metadata);
   records->metadata.va_address = params->va_address;
   records->metadata.va_size = params->size;
   records->metadata.mtype = params->mtype;
   records->metadata.mall_policy = params->mall_policy;

   memset(create, 0, sizeof(*create));
   create->flags = AGC_WDDM_ALLOC_CREATE_FLAGS;
   create->records_size = sizeof(*records);
   create->checksum = agc_wddm_checksum(create, sizeof(*create));
}

static inline uint32_t
agc_wddm_fill_submit(struct agc_wddm_submit_data *data, const struct agc_ib *ibs, uint32_t count)
{
   uint32_t bytes = (uint32_t)(sizeof(data->header) + count * sizeof(data->ibs[0]));
   memset(data, 0, bytes);
   data->header.version = AGC_WDDM_ABI_VERSION;
   data->header.entry_count = count;
   for (uint32_t i = 0; i < count; i++) {
      struct agc_wddm_submit_ib *ib = &data->ibs[i];
      ib->type = AGC_WDDM_SUBMIT_ENTRY_GFX_IB;
      ib->section_size = sizeof(*ib);
      ib->length = ibs[i].dwords * (uint32_t)sizeof(uint32_t);
      ib->flags = ibs[i].flags;
      ib->ip_type = AGC_WDDM_SUBMIT_IP_GFX;
      ib->address_low = (uint32_t)ibs[i].va;
      ib->address_high = (uint32_t)(ibs[i].va >> 32);
   }
   return bytes;
}

#endif
