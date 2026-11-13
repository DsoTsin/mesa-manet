#ifndef NVK_BVH_H
#define NVK_BVH_H

#ifndef VULKAN
#include <stdint.h>
#include <stddef.h>
#endif

#define NVK_BVH_ENCODE_TRIANGLE_PACKING (1u << 31)
#define NVK_BVH_ENCODE_OPACITY_MICROMAPS (1u << 30)
#define NVK_BVH_HEADER_SIZE 512u
#define NVK_BVH_NODE_SIZE 128u
#define NVK_BVH_INSTANCE_SIZE 64u
#define NVK_BVH_INSTANCE_DATA_UNIT_SIZE 1152u
#define NVK_BVH_INSTANCE_DATA_UNIT_SLOTS 16u
#define NVK_BVH_BOUNDS_OFFSET 384u
#define NVK_BVH_COMPACTED_SIZE_OFFSET 408u
#define NVK_BVH_SERIALIZATION_SIZE_OFFSET 416u
#define NVK_BVH_PRIMITIVE_DATA_OFFSET 424u
#define NVK_BVH_GEOMETRY_TYPE_OFFSET 432u
#define NVK_BVH_ACTIVE_LEAF_COUNT_OFFSET 436u
#define NVK_BVH_PRIMITIVE_COUNT_OFFSET 440u
#define NVK_BVH_NODE_COUNT_OFFSET 444u
#define NVK_BVH_INSTANCE_COUNT_OFFSET 448u
#define NVK_BVH_ROOT_FLAGS_OFFSET 452u
#define NVK_BVH_MAGIC_OFFSET 456u
#define NVK_BVH_REFIT_DATA_OFFSET 464u
#define NVK_BVH_REFIT_LEAF_MAP_OFFSET 472u
#define NVK_BVH_REFIT_LEAF_OFFSET 480u
#define NVK_BVH_OMM_METADATA_OFFSET 488u
#define NVK_BVH_MAGIC 0x4e564b52u
#define NVK_BVH_INVALID_NODE 0xffffffffu
#define NVK_BVH_MAPPING_SIZE 16u
#define NVK_BVH_COMPAT_UUID_0 0x4e564b52u
#define NVK_BVH_COMPAT_UUID_1 0x30323101u
#define NVK_BVH_COMPAT_UUID_2 0x00000004u
#define NVK_BVH_COMPAT_UUID_3 0x00000000u

struct nvk_bvh_encode_args {
   uint64_t ir;
   uint64_t ir_header;
   uint64_t dst;
   uint64_t mapping;
   uint64_t leaf_map;
   uint64_t prefix_sums;
   uint64_t node_offset;
   uint64_t leaf_offset;
   uint64_t metadata_offset;
   uint64_t primitive_data_offset;
   uint64_t size;
   uint64_t refit_data_offset;
   uint64_t refit_leaf_map_offset;
   uint32_t leaf_capacity;
   uint32_t ir_leaf_size;
   uint32_t geometry_type;
   uint32_t pass;
   uint32_t build_flags;
   uint32_t geometry_count;
   uint64_t omm_metadata_offset;
};

struct nvk_bvh_refit_args {
   uint64_t ir;
   uint64_t dst;
   uint64_t counter;
   uint32_t ir_leaf_size;
   uint32_t pass;
};

struct nvk_bvh_copy_args {
   uint64_t src;
   uint64_t dst;
   uint64_t available;
   uint64_t size;
   uint32_t mode;
   uint32_t field;
};

#endif
