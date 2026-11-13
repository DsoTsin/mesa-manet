#ifndef NVK_BVH_LEAF_H
#define NVK_BVH_LEAF_H

#include <stdint.h>

#define NVK_BVH_LEAF_THREADS 128u
#define NVK_BVH_LEAF_ALWAYS_ACTIVE (1u << 0)
#define NVK_BVH_LEAF_PROPAGATE_CULL_FLAGS (1u << 1)
#define NVK_BVH_LEAF_HAS_QUADS (1u << 3)
#define NVK_BVH_LEAF_EARLY_PAIRS (1u << 4)
#define NVK_BVH_LEAF_INACTIVE_BOUNDS (1u << 6)
#define NVK_BVH_LEAF_STORE_VERTEX_IDS (1u << 7)
#define NVK_BVH_LEAF_OMM_ATTACHED (1u << 0)
#define NVK_BVH_LEAF_OMM_PRESERVE (1u << 1)

struct nvk_bvh_leaf_geometry {
   uint64_t data;
   uint64_t indices;
   uint64_t transform;
   uint32_t geometry_id;
   uint32_t geometry_type;
   uint32_t first_id;
   uint32_t stride;
   uint32_t vertex_format;
   uint32_t index_format;
   uint64_t indirect_range;
};

struct nvk_bvh_leaf_args {
   uint64_t ir;
   uint64_t header;
   uint64_t ids;
   struct nvk_bvh_leaf_geometry geom_data;
   uint32_t primitive_count;
   uint32_t ir_leaf_size;
   uint32_t build_flags;
   uint32_t pad;
   uint64_t micromap_array;
   uint64_t index_buffer;
   uint64_t output_metadata;
   uint32_t index_type;
   uint32_t index_stride;
   uint32_t base_triangle;
   uint32_t omm_flags;
};

#endif
