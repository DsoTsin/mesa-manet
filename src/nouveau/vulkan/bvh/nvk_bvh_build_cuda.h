#ifndef NVK_BVH_BUILD_CUDA_H
#define NVK_BVH_BUILD_CUDA_H

#include <stdint.h>

#define NVK_BVH_BUILD_CUDA_THREADS 256u
#define NVK_BVH_BUILD_CUDA_TILE_KEYS 512u
#define NVK_BVH_BUILD_CUDA_KEY32_SIZE 8u
#define NVK_BVH_BUILD_CUDA_KEY64_SIZE 12u
#define NVK_BVH_BUILD_CUDA_NODE_INFO_SIZE 16u
#define NVK_BVH_BUILD_CUDA_BOX_SIZE 40u
#define NVK_BVH_BUILD_CUDA_KEYS_64BIT (1u << 2)
#define NVK_BVH_BUILD_CUDA_PROPAGATE_FLAGS (1u << 1)
#define NVK_BVH_BUILD_CUDA_NATIVE_TILE_KEYS 3520u
#define NVK_BVH_BUILD_CUDA_NATIVE_TILE_THREADS 320u
#define NVK_BVH_BUILD_CUDA_NATIVE_SMALL_KEYS 256u
#define NVK_BVH_BUILD_CUDA_NATIVE_SMALL_THREADS 64u
#define NVK_BVH_BUILD_CUDA_NATIVE_MERGE_LIMIT 1802240u
#define NVK_BVH_BUILD_CUDA_NATIVE_RADIX_THREADS 256u
#define NVK_BVH_BUILD_CUDA_NATIVE_RADIX_ITEMS 8u
#define NVK_BVH_BUILD_CUDA_NATIVE_RADIX_TILE 2048u
#define NVK_BVH_BUILD_CUDA_NATIVE_RADIX_SCAN_THREADS 512u
#define NVK_BVH_BUILD_CUDA_NATIVE_RADIX_SCAN_ITEMS 23u
#define NVK_BVH_BUILD_CUDA_NATIVE_RADIX_SCAN_TILE 11776u
#define NVK_BVH_BUILD_CUDA_NATIVE_RADIX_PASSES 13u
#define NVK_BVH_BUILD_CUDA_NATIVE_AAC_THREADS 192u
#define NVK_BVH_BUILD_CUDA_NATIVE_AAC_MIN_LANES 4u
#define NVK_BVH_BUILD_CUDA_NATIVE_AAC_MAX4 4u
#define NVK_BVH_BUILD_CUDA_NATIVE_AAC_MAX8 8u
#define NVK_BVH_BUILD_CUDA_NATIVE_AAC_MAX16 16u

static inline uint32_t
nvk_bvh_build_cuda_radix_spine_count(uint32_t grid)
{
   return ((32u * grid + NVK_BVH_BUILD_CUDA_NATIVE_RADIX_SCAN_TILE - 1u) /
           NVK_BVH_BUILD_CUDA_NATIVE_RADIX_SCAN_TILE) *
          NVK_BVH_BUILD_CUDA_NATIVE_RADIX_SCAN_TILE;
}

static inline uint32_t
nvk_bvh_build_cuda_radix_pass_bit(uint32_t pass)
{
   return pass == 0 ? 0u : 4u + (pass - 1u) * 5u;
}

static inline uint32_t
nvk_bvh_build_cuda_radix_pass_width(uint32_t pass)
{
   return pass == 0 ? 4u : 5u;
}

struct nvk_bvh_build_cuda_args {
   uint64_t ir;
   uint64_t ir_header;
   uint64_t input_ids;
   uint64_t src_keys;
   uint64_t dst_keys;
   uint64_t node_info;
   uint32_t leaf_count;
   uint32_t ir_leaf_size;
   uint32_t internal_node_base;
   uint32_t build_flags;
   uint32_t merge_width;
   uint32_t reserved;
};

struct nvk_bvh_build_cuda_node_info {
   uint32_t path_count;
   uint32_t children[2];
   uint32_t parent;
};

struct nvk_bvh_build_cuda_box {
   float minimum[3];
   float maximum[3];
   uint32_t children[2];
   uint32_t bvh_offset;
   uint32_t flags;
};

struct nvk_bvh_build_cuda_planner_node {
   int32_t child_pair;
   union {
      struct {
         float half_area;
         int32_t parent;
         float primitive_cost;
      } before;
      struct {
         float subtree_cost;
         int32_t subtree_prims;
         int32_t owner_subtree_prims;
      } after;
   };
};

struct nvk_bvh_build_cuda_cluster {
   int32_t child_pair;
   int32_t subtree_prims;
   float minimum[3];
   float maximum[3];
};

#endif
