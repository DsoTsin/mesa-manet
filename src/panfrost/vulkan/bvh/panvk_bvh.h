/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_BVH_H
#define PANVK_BVH_H

#ifdef VULKAN
#define VK_UUID_SIZE 16
#else
#include <vulkan/vulkan.h>
#endif

#include "vk_bvh_defines.h"

#define PANVK_BVH_HEADER_SIZE  128
#define PANVK_BVH_NODE_SIZE    64
#define PANVK_BVH_MAX_CHILDREN 6
#define PANVK_BVH_VERSION      3

#define PANVK_BVH_FLAGS_AABBS     0x1
#define PANVK_BVH_FLAGS_INSTANCES 0x3
#define PANVK_BVH_FLAGS_TRIANGLES 0x5

#define PANVK_BVH_BOX_BLAS 0x80
#define PANVK_BVH_BOX_TLAS 0x81

#define PANVK_BVH_CHILD_INTERNAL 0x40
#define PANVK_BVH_CHILD_EMPTY    0xc0

#define PANVK_BVH_LEAF_OPAQUE   0x01
#define PANVK_BVH_LEAF_IN_ORDER 0x04

#define PANVK_BVH_INSTANCE_AABBS 0x8000000000000000ul

#define PANVK_BVH_SERIAL_HEADER_SIZE (2 * VK_UUID_SIZE + 3 * 8)

#define PANVK_BVH_SERIALIZATION_SIZE_OFFSET 0x48
#define PANVK_BVH_INSTANCE_COUNT_OFFSET     0x50

struct panvk_bvh_header {
   uint64_t size;
   uint32_t node_count;
   uint32_t version;
   uint32_t primitive_count;
   uint32_t geometry_count;
   vk_aabb bounds;
   uint32_t flags;
   uint32_t reserved0;
   uint64_t omm_data_offset;
   uint64_t omm_handles_offset;
   uint64_t serialization_size;
   uint64_t instance_count;
   uint32_t reserved1[10];
};

struct panvk_bvh_box_node {
   uint32_t origin_x_type;
   uint32_t origin_y_exp_x;
   uint32_t origin_z_exp_y;
   uint32_t internal_base;
   uint32_t leaf_base;
   uint32_t exp_z_leaf_count_child01;
   uint32_t child2345;
   uint32_t bounds[9];
};

struct panvk_bvh_triangle_leaf {
   float coords[4][3];
   uint32_t primitive_id_b;
   uint32_t primitive_id_a;
   uint32_t geometry_id_b;
   uint32_t geometry_id_a_flags;
};

struct panvk_bvh_aabb_leaf {
   uint32_t reserved[13];
   uint32_t primitive_id;
   uint32_t reserved1;
   uint32_t geometry_id_flags;
};

struct panvk_bvh_instance_leaf {
   mat3x4 wto_matrix;
   uint32_t mask_index;
   uint32_t flags_sbt_offset;
   uint64_t blas_ptr;
};

struct panvk_bvh_instance_extra {
   uint32_t custom_index;
   uint32_t instance_id;
};

#ifdef VULKAN
TYPE(panvk_bvh_header, 8);
TYPE(panvk_bvh_box_node, 4);
TYPE(panvk_bvh_triangle_leaf, 4);
TYPE(panvk_bvh_aabb_leaf, 4);
TYPE(panvk_bvh_instance_leaf, 8);
TYPE(panvk_bvh_instance_extra, 4);
#endif

struct panvk_encode_args {
   VOID_REF intermediate_bvh;
   VOID_REF output_bvh;
   REF(vk_ir_header) header;
   uint32_t leaf_node_count;
   uint32_t leaf_base;
   uint32_t geometry_type;
   uint32_t geometry_count;
};

#define PANVK_COPY_MODE_COPY        0
#define PANVK_COPY_MODE_SERIALIZE   1
#define PANVK_COPY_MODE_DESERIALIZE 2

struct panvk_copy_args {
   VOID_REF src_addr;
   VOID_REF dst_addr;
   uint32_t mode;
   uint32_t uuid[8];
};

#endif
