#ifndef PANVK_BVH_MICROMAP_H
#define PANVK_BVH_MICROMAP_H

#include "vk_bvh_defines.h"

#define PANVK_MICROMAP_HEADER_SIZE 64
#define PANVK_MICROMAP_MAX_LEVEL 10
#define PANVK_MICROMAP_BUCKETS 16
#define PANVK_MICROMAP_SERIAL_HEADER_SIZE 56

struct panvk_micromap_header {
   uint32_t size;
   uint32_t flags;
   uint32_t version;
   uint32_t triangle_count;
   uint32_t active_count;
   uint32_t usage_count;
   uint32_t summary_offset;
   uint32_t reserved[9];
};

struct panvk_micromap_triangle {
   uint32_t data_offset;
   uint16_t level;
   uint16_t format;
};

struct panvk_micromap_bucket {
   uint64_t offset;
   uint32_t stride;
   uint32_t counter;
};

struct panvk_omm_geometry {
   VOID_REF micromap;
   VOID_REF indices;
   VOID_REF buffer_base;
   uint32_t index_format;
   uint32_t index_stride;
   uint32_t base_triangle;
   uint32_t enabled;
};

struct panvk_micromap_build_args {
   VOID_REF src;
   VOID_REF triangles;
   VOID_REF dst;
   VOID_REF buckets;
   VOID_REF counters;
   uint32_t triangle_stride;
   uint32_t summary_offset;
};

struct panvk_micromap_copy_args {
   VOID_REF src;
   VOID_REF dst;
   uint32_t mode;
   uint32_t uuid[8];
};

#ifdef VULKAN
TYPE(panvk_micromap_header, 4);
TYPE(panvk_micromap_triangle, 4);
TYPE(panvk_micromap_bucket, 8);
TYPE(panvk_omm_geometry, 8);
#endif

#endif
