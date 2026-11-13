#ifndef NVK_BVH_PLANNER_H
#define NVK_BVH_PLANNER_H

#include <stdint.h>
#include <stddef.h>

#define NVK_BVH_PLANNER_INIT_SQUEEZE 1u
#define NVK_BVH_PLANNER_INVALID_CONFIGURATION 1u
#define NVK_BVH_PLANNER_SQUEEZE_BUDGET_ERROR 2u
#define NVK_BVH_PLANNER_INVALID_TOPOLOGY 4u
#define NVK_BVH_PLANNER_BOUNDS_THREADS 128u
#define NVK_BVH_PLANNER_DEFAULT_BRANCHING 12u
#define NVK_BVH_PLANNER_DEFAULT_LEAF_SIZE 16u
#define NVK_BVH_PLANNER_DEFAULT_NODE_COST 8.0f
#define NVK_BVH_PLANNER_DEFAULT_TRIANGLE_COST 0.7f

struct nvk_bvh_planner_args {
   uint64_t ir;
   uint64_t ir_header;
   uint64_t nodes;
   uint64_t pair_to_ir;
   uint64_t workspace;
   uint32_t capacity;
   uint32_t ir_leaf_size;
   uint32_t branching_factor;
   uint32_t max_leaf_size;
   float sah_node_cost;
   float sah_primitive_cost;
   uint32_t max_wide_nodes;
   uint32_t flags;
};

#ifdef __cplusplus
static_assert(sizeof(nvk_bvh_planner_args) == 72);
static_assert(offsetof(nvk_bvh_planner_args, capacity) == 40);
static_assert(offsetof(nvk_bvh_planner_args, sah_node_cost) == 56);
static_assert(offsetof(nvk_bvh_planner_args, max_wide_nodes) == 64);
#endif

#endif
