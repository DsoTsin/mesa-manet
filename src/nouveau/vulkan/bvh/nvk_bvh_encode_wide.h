#ifndef NVK_BVH_ENCODE_WIDE_H
#define NVK_BVH_ENCODE_WIDE_H

#include <stdint.h>
#include <stddef.h>

#define NVK_BVH_WIDE_WORKSPACE_SIZE 32u
#define NVK_BVH_WIDE_AUXILIARY_SIZE 32u
#define NVK_BVH_WIDE_MIN_CAPACITY 2u
#define NVK_BVH_WIDE_WORK_COUNTER_OFFSET 24u
#define NVK_BVH_WIDE_NODE_LIMIT_ERROR 1u
#define NVK_BVH_WIDE_CHILD_LIMIT_ERROR 2u
#define NVK_BVH_WIDE_ALLOCATION_ERROR 4u
#define NVK_BVH_WIDE_TRIANGLE_PACKING_REQUIRED 8u
#define NVK_BVH_ENCODE_TRIANGLE_PACKING (1u << 31)

struct nvk_bvh_wide_node {
   int32_t child_pair;
   float subtree_cost;
   int32_t subtree_primitives;
   int32_t owner_subtree_primitives;
};

struct nvk_bvh_wide_workspace {
   uint32_t allocated_nodes;
   uint32_t reserved0;
   uint32_t node_count;
   uint32_t max_leaf_primitives;
   uint32_t branching_factor;
   uint32_t error;
   uint64_t reserved1;
};

#ifdef __cplusplus
static_assert(sizeof(nvk_bvh_wide_node) == 16);
static_assert(sizeof(nvk_bvh_wide_workspace) == NVK_BVH_WIDE_WORKSPACE_SIZE);
static_assert(offsetof(nvk_bvh_wide_workspace, node_count) == 8);
static_assert(offsetof(nvk_bvh_wide_workspace, error) == 20);
#endif

#endif
