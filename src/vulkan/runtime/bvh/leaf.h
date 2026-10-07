/*
 * Copyright © 2022 Konstantin Seurer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "vk_bvh_helpers.h"

#define SpvCapabilitySignedZeroInfNanPreserve 4466
#define SpvExecutionModeSignedZeroInfNanPreserve 4461
spirv_execution_mode(extensions = ["SPV_KHR_float_controls"],
                     capabilities = [SpvCapabilitySignedZeroInfNanPreserve],
                     SpvExecutionModeSignedZeroInfNanPreserve, 32);

layout(local_size_x_id = SUBGROUP_SIZE_ID, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform CONSTS {
   leaf_args args;
};

triangle_vertices
load_triangle(vk_bvh_geometry_data geom_data, uint32_t global_id)
{
   triangle_indices indices = load_indices(geom_data.indices, geom_data.index_format, global_id);

   triangle_vertices vertices = load_vertices(geom_data.data, indices, geom_data.vertex_format, geom_data.stride);

   if (geom_data.transform != NULL) {
      mat4 transform = mat4(1.0);

      for (uint32_t col = 0; col < 4; col++)
      for (uint32_t row = 0; row < 3; row++)
      transform[col][row] = DEREF(INDEX(float, geom_data.transform, col + row * 4));

      for (uint32_t i = 0; i < 3; i++)
      vertices.vertex[i] = transform * vertices.vertex[i];
   }

   return vertices;
}

bool
triangle_has_nan(triangle_vertices vertices)
{
   return any(isnan(vertices.vertex[0])) || any(isnan(vertices.vertex[1])) || any(isnan(vertices.vertex[2]));
}

uint32_t
early_pair_edges(triangle_vertices a, triangle_vertices b)
{
   for (uint32_t e0 = 0; e0 < 3; e0++) {
      for (uint32_t e1 = 0; e1 < 3; e1++) {
         if (a.vertex[e0].xyz == b.vertex[(e1 + 1) % 3].xyz && a.vertex[(e0 + 1) % 3].xyz == b.vertex[e1].xyz)
            return e0 | (e1 << 2);
      }
   }
   return VK_BVH_INVALID_NODE;
}

bool
build_triangle(inout vk_aabb bounds, VOID_REF dst_ptr, vk_bvh_geometry_data geom_data, uint32_t global_id)
{
   bool is_valid = true;
   triangle_vertices vertices = load_triangle(geom_data, global_id);

   /* An inactive triangle is one for which the first (X) component of any vertex is NaN. If any
    * other vertex component is NaN, and the first is not, the behavior is undefined. We treat those
    * undefined cases as inactive to filter out NaNs. If the vertex format does not have a NaN
    * representation, then all triangles are considered active.
    */
   if (any(isnan(vertices.vertex[0])) || any(isnan(vertices.vertex[1])) || any(isnan(vertices.vertex[2]))) {
      if (!VK_TEST_BUILD_FLAG_ALWAYS_ACTIVE)
         return false;

      is_valid = false;
      vertices.vertex[0] = vec4(0);
      vertices.vertex[1] = vec4(0);
      vertices.vertex[2] = vec4(0);
   }

   REF(vk_ir_triangle_node) node = REF(vk_ir_triangle_node)(dst_ptr);

   bounds.min = vec3(INFINITY);
   bounds.max = vec3(-INFINITY);

   for (uint32_t coord = 0; coord < 3; coord++)
   for (uint32_t comp = 0; comp < 3; comp++) {
      DEREF(node).coords[coord][comp] = vertices.vertex[coord][comp];
      bounds.min[comp] = min(bounds.min[comp], vertices.vertex[coord][comp]);
      bounds.max[comp] = max(bounds.max[comp], vertices.vertex[coord][comp]);
   }

   DEREF(node).base.aabb = bounds;
   DEREF(node).triangle_id = global_id;
   DEREF(node).geometry_id_and_flags = geom_data.geometry_id;

   if (VK_TEST_BUILD_FLAG_HAS_QUADS) {
      REF(vk_ir_triangle_node_quad) quad = vk_ir_triangle_node_get_quad_ref(node);
      DEREF(quad).triangle_id = VK_QUAD_TRIANGLE_ID_UNUSED;
   }

   return is_valid;
}

bool
build_aabb(inout vk_aabb bounds, VOID_REF src_ptr, VOID_REF dst_ptr, uint32_t geometry_id, uint32_t global_id)
{
   bool is_valid = true;
   REF(vk_ir_aabb_node) node = REF(vk_ir_aabb_node)(dst_ptr);

   for (uint32_t vec = 0; vec < 2; vec++)
   for (uint32_t comp = 0; comp < 3; comp++) {
      float coord = DEREF(INDEX(float, src_ptr, comp + vec * 3));

      if (vec == 0)
      bounds.min[comp] = coord;
      else
      bounds.max[comp] = coord;
   }

   /* An inactive AABB is one for which the minimum X coordinate is NaN. If any other component is
    * NaN, and the first is not, the behavior is undefined. We treat those undefined cases as inactive
    * to filter out NaNs.
    */
   if (any(isnan(bounds.min)) || any(isnan(bounds.max))) {
      if (!VK_TEST_BUILD_FLAG_ALWAYS_ACTIVE)
         return false;

      is_valid = false;
      bounds.min = vec3(0);
      bounds.max = vec3(0);
   }

   DEREF(node).base.aabb = bounds;
   DEREF(node).primitive_id = global_id;
   DEREF(node).geometry_id_and_flags = geometry_id;

   return is_valid;
}

bool
build_instance(inout vk_aabb bounds, VOID_REF src_ptr, VOID_REF dst_ptr, uint32_t global_id)
{
   REF(vk_ir_instance_node) node = REF(vk_ir_instance_node)(dst_ptr);

   AccelerationStructureInstance instance = DEREF(REF(AccelerationStructureInstance)(src_ptr));

   /* An inactive instance is one whose acceleration structure handle is VK_NULL_HANDLE. */
   bool is_valid = instance.accelerationStructureReference != 0 && instance.custom_instance_and_mask >= (1u << 24u);
   if (!is_valid && !VK_TEST_BUILD_FLAG_ALWAYS_ACTIVE)
      return false;

   DEREF(node).base_ptr = instance.accelerationStructureReference;
   DEREF(node).custom_instance_and_mask = instance.custom_instance_and_mask;
   DEREF(node).sbt_offset_and_flags = instance.sbt_offset_and_flags;
   DEREF(node).instance_id = global_id;

   if (instance.accelerationStructureReference == 0) {
      bounds = vk_aabb(vec3(0.0), vec3(0.0));
      DEREF(node).otw_matrix = mat3x4(1.0);
      DEREF(node).base.aabb = bounds;
      DEREF(node).root_flags = 0;
      return false;
   }

   mat4 transform = mat4(instance.transform);
   DEREF(node).otw_matrix = mat3x4(transform);

   vk_aabb blas_aabb = DEREF(REF(vk_aabb)(instance.accelerationStructureReference + BVH_BOUNDS_OFFSET));

   bounds = calculate_instance_node_bounds(blas_aabb, mat3x4(transform));

#ifdef CALCULATE_FINE_INSTANCE_NODE_BOUNDS
   vec3 blas_aabb_extent = blas_aabb.max - blas_aabb.min;
   float blas_aabb_volume = blas_aabb_extent.x * blas_aabb_extent.y * blas_aabb_extent.z;
   blas_aabb_volume *= abs(determinant(mat3(transform)));

   vec3 bounds_extent = bounds.max - bounds.min;
   float bounds_volume = bounds_extent.x * bounds_extent.y * bounds_extent.z;

   /* Only try calculating finer-grained instance node bounds if the volume of the transformed
    * instance AABB is significantly higher than the volume of the BLAS without transformations
    * applied. Otherwise, the finer-grained bounds won't be much smaller and the additional overhead
    * wouldn't be worth it.
    */
   if (bounds_volume > 1.4f * blas_aabb_volume)
      bounds = CALCULATE_FINE_INSTANCE_NODE_BOUNDS(instance.accelerationStructureReference, mat3x4(transform));
#endif

   if (any(isnan(bounds.min)) || any(isnan(bounds.max))) {
      if (!VK_TEST_BUILD_FLAG_ALWAYS_ACTIVE)
         return false;
      is_valid = false;
      bounds = vk_aabb(vec3(0.0), vec3(0.0));
   }

   DEREF(node).base.aabb = bounds;

   if (!VK_TEST_BUILD_FLAG_PROPAGATE_CULL_FLAGS)
      return is_valid;

   uint32_t root_flags = 0;
   if ((instance.sbt_offset_and_flags & (VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR << 24)) != 0)
      root_flags = VK_BVH_BOX_FLAG_ONLY_OPAQUE;
   else if ((instance.sbt_offset_and_flags & (VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR << 24)) != 0)
      root_flags = VK_BVH_BOX_FLAG_NO_OPAQUE;
   else
      root_flags = DEREF(REF(uint32_t)(instance.accelerationStructureReference + ROOT_FLAGS_OFFSET));
   DEREF(node).root_flags = root_flags | ((~(instance.custom_instance_and_mask >> 24)) << VK_BVH_BOX_FLAGS_INV_CULL_MASK_SHIFT);

   return is_valid;
}

void
main(void)
{
   uint32_t global_id = gl_GlobalInvocationID.x;
   uint32_t primitive_id = args.geom_data.first_id + global_id;

   uint32_t src_offset = global_id * args.geom_data.stride;

   uint32_t dst_stride = vk_ir_node_size(args.geom_data.geometry_type);
   uint32_t node_type;
   if (args.geom_data.geometry_type == VK_GEOMETRY_TYPE_TRIANGLES_KHR)
      node_type = vk_ir_node_triangle;
   else if (args.geom_data.geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR)
      node_type = vk_ir_node_aabb;
   else
      node_type = vk_ir_node_instance;

   uint32_t dst_offset = primitive_id * dst_stride;
   VOID_REF dst_ptr = OFFSET(args.bvh, dst_offset);

   vk_aabb bounds;
   bool is_active;
   if (args.geom_data.geometry_type == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
      is_active = build_triangle(bounds, dst_ptr, args.geom_data, global_id);

      if (VK_TEST_BUILD_FLAG_EARLY_PAIRS && is_active && (global_id | 1u) < args.primitive_count) {
         triangle_vertices a = load_triangle(args.geom_data, global_id & ~1u);
         triangle_vertices b = load_triangle(args.geom_data, global_id | 1u);
         uint32_t edges = (triangle_has_nan(a) || triangle_has_nan(b)) ? VK_BVH_INVALID_NODE : early_pair_edges(a, b);

         if (edges != VK_BVH_INVALID_NODE) {
            if ((global_id & 1u) != 0u) {
               is_active = false;
            } else {
               vec3 v3 = b.vertex[((edges >> 2) + 2) % 3].xyz;
               REF(vk_ir_triangle_node) node = REF(vk_ir_triangle_node)(dst_ptr);
               REF(vk_ir_triangle_node_quad) quad = vk_ir_triangle_node_get_quad_ref(node);
               DEREF(quad).coords[0] = v3.x;
               DEREF(quad).coords[1] = v3.y;
               DEREF(quad).coords[2] = v3.z;
               DEREF(quad).triangle_id = (global_id + 1u) | (edges << 28);
               bounds.min = min(bounds.min, v3);
               bounds.max = max(bounds.max, v3);
               DEREF(node).base.aabb = bounds;
            }
         }
      }
   } else if (args.geom_data.geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR) {
      VOID_REF src_ptr = OFFSET(args.geom_data.data, src_offset);
      is_active = build_aabb(bounds, src_ptr, dst_ptr, args.geom_data.geometry_id, global_id);
   } else {
      VOID_REF src_ptr = OFFSET(args.geom_data.data, src_offset);
      /* arrayOfPointers */
      if (args.geom_data.stride == 8) {
         src_ptr = DEREF(REF(VOID_REF)(src_ptr));
      }

      is_active = build_instance(bounds, src_ptr, dst_ptr, global_id);
   }

   if (VK_TEST_BUILD_FLAG_ALWAYS_ACTIVE)
      is_active = true;

   uint32_t id = is_active ? pack_ir_node_id(dst_offset, node_type) : VK_BVH_INVALID_NODE;
   DEREF(INDEX(uint32_t, args.ids, primitive_id)) = id;

   uvec4 ballot = subgroupBallot(is_active);
   if (subgroupElect())
      atomicAdd(DEREF(args.header).active_leaf_count, subgroupBallotBitCount(ballot));

   if (is_active) {
      atomicMin(DEREF(args.header).min_bounds[0], to_emulated_float(bounds.min.x));
      atomicMin(DEREF(args.header).min_bounds[1], to_emulated_float(bounds.min.y));
      atomicMin(DEREF(args.header).min_bounds[2], to_emulated_float(bounds.min.z));
      atomicMax(DEREF(args.header).max_bounds[0], to_emulated_float(bounds.max.x));
      atomicMax(DEREF(args.header).max_bounds[1], to_emulated_float(bounds.max.y));
      atomicMax(DEREF(args.header).max_bounds[2], to_emulated_float(bounds.max.z));
   }
}
