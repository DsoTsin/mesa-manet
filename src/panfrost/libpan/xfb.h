#pragma once

#include "compiler/libcl/libcl.h"

struct panlib_xfb_params {
   uint64_t base[4];
   uint64_t size[4];
   uint64_t offsets;
   uint64_t query;
   uint64_t index_buffer;
   uint64_t index_range;
   uint32_t stride[4];
   uint32_t output_end[4];
   uint32_t topology;
   uint32_t index_size;
   uint32_t restart;
   uint32_t provoking_first;
   uint32_t vertex_count;
   uint32_t instance_count;
   uint32_t first_index;
   int32_t first_vertex;
   uint32_t first_instance;
   uint32_t draw_id;
   uint32_t vertices_per_primitive;
   uint32_t primitives_per_instance;
   uint32_t captured_vertices;
   uint32_t workgroups[3];
   uint32_t start[4];
};
