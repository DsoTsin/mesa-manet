/*
 * Copyright © 2026 PanVK contributors
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBPAN_CSF_DGC_EXECUTE_H
#define LIBPAN_CSF_DGC_EXECUTE_H

#include "libpan_csf_dgc_shader.h"

#define PANLIB_DGC_RESOURCE_TABLE_SIZE 8

/* Queue submission creates one node for each occurrence of an Execute,
 * including repeated secondary command buffers. Compute and VT consume
 * separate FIFO links so their independent streams cannot race the cursor.
 */
struct panlib_dgc_submit_node {
   uint64_t next_compute;
   uint64_t next_vt;
   uint64_t prepare_fau;
   uint64_t prepare_tsd;
   uint64_t output;
   uint64_t tsd_table;
};

/* Private execution record. Its variable-size tail holds the selected
 * shader's FAUs and descriptors, sized from pipeline metadata at submission.
 * Decoded PC/VB state is consumed during preparation and is not retained.
 * A later sequence never overwrites these allocations.
 */
struct panlib_dgc_execution {
   uint64_t fau[PANLIB_DGC_STAGE_COUNT];
   uint64_t srt[PANLIB_DGC_STAGE_COUNT];
   uint64_t vs_spd[3];
   uint64_t fs_spd;
   uint64_t cs_spd;
   uint64_t tsd;
   uint64_t zsd;
   uint32_t compute_size_workgroup;
   uint32_t workgroups_per_task;
   uint32_t dcd[3];
   uint32_t tiler_flags;
   uint32_t varying_size;
   uint32_t primitive_size;
   uint32_t draw[5];
   uint32_t dispatch[3];
   struct panlib_dgc_index_buffer ib;
} __attribute__((aligned(64)));

struct panlib_dgc_execute {
   uint64_t preprocess;
   uint64_t output;
   uint64_t sequence_offset;
   uint32_t sequence_stride;
   uint32_t output_sequence_stride;
   uint32_t fau_offset[PANLIB_DGC_STAGE_COUNT];
   uint32_t driver_offset[PANLIB_DGC_STAGE_COUNT];
   uint32_t resource_offset[PANLIB_DGC_STAGE_COUNT];
   uint32_t depth_stencil_offset;
   uint32_t pipeline_stride;
   uint64_t pipelines;
   uint64_t tsd_table;
   uint64_t sysvals[PANLIB_DGC_STAGE_COUNT];
   uint64_t driver_set[PANLIB_DGC_STAGE_COUNT];
   uint64_t resource_table[PANLIB_DGC_STAGE_COUNT];
   uint32_t driver_set_size[PANLIB_DGC_STAGE_COUNT];
   uint32_t first_vertex_sysval;
   uint32_t base_instance_sysval;
   uint32_t num_workgroups_sysval;
   uint32_t local_group_size_sysval;
   uint32_t noperspective_sysval;
   uint32_t attribs_valid;
   uint32_t attribs_per_instance;
   uint32_t attribs_zero_divisor;
   uint32_t attrib_binding[PANLIB_DGC_MAX_VERTEX_BUFFERS];
   uint32_t attrib_offset[PANLIB_DGC_MAX_VERTEX_BUFFERS];
   uint32_t vertex_buffer_offset;
   uint32_t vertex_buffer_count;
   uint64_t depth_stencil;
   uint64_t layer_output_mask;
   uint64_t primitive_id_output_mask;
   uint32_t dcd[3];
   uint32_t tiler_flags;
   uint32_t primitive_size;
   uint32_t point_primitive;
   uint32_t fs_enabled;
   uint32_t earlyzs_index;
   uint32_t render_target_mask;
   uint32_t color_output_shift;
   uint32_t color_output_map[8];
   uint32_t input_attachment_mask[8];
   uint32_t depth_input_mask;
   uint32_t stencil_input_mask;
   uint32_t alpha_to_coverage;
   uint32_t per_sample;
   uint32_t blend_shader;
   uint32_t force_late_zs;
};

#endif /* LIBPAN_CSF_DGC_EXECUTE_H */
