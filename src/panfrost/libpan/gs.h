#pragma once

#include "compiler/libcl/libcl.h"

struct panvk_gs_raster_params {
   uint64_t index_buffer;
   uint64_t position_buffer;
   uint64_t varying_buffer;
   uint64_t clip_position_buffer;
   uint32_t index_count;
   float viewport_scale[3];
   float viewport_offset[3];
};
