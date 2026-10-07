/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_BVH_LEAF_H
#define PANVK_BVH_LEAF_H

mat3x4
panvk_invert_transform(mat3x4 m)
{
   mat3 inv = inverse(mat3(m[0].xyz, m[1].xyz, m[2].xyz));
   vec3 t = -(vec3(m[0].w, m[1].w, m[2].w) * inv);
   return mat3x4(vec4(inv[0], t.x), vec4(inv[1], t.y), vec4(inv[2], t.z));
}

void
panvk_write_instance_leaf(REF(panvk_bvh_instance_leaf) leaf, REF(panvk_bvh_instance_extra) extra,
                          uint64_t base_ptr, mat3x4 otw_matrix, uint32_t custom_instance_and_mask,
                          uint32_t sbt_offset_and_flags, uint32_t instance_id, uint32_t slot)
{
   uint64_t blas_ptr = 0;
   uint32_t mask = custom_instance_and_mask & 0xff000000;
   if (base_ptr != 0) {
      blas_ptr = base_ptr + PANVK_BVH_HEADER_SIZE;
      if (DEREF(REF(panvk_bvh_header)(base_ptr & ~0x3ful)).flags == PANVK_BVH_FLAGS_AABBS)
         blas_ptr |= PANVK_BVH_INSTANCE_AABBS;
   } else {
      mask = 0;
   }

   DEREF(leaf).wto_matrix = panvk_invert_transform(otw_matrix);
   DEREF(leaf).mask_index = mask | slot;
   DEREF(leaf).flags_sbt_offset = sbt_offset_and_flags;
   DEREF(leaf).blas_ptr = blas_ptr;

   DEREF(extra).custom_index = custom_instance_and_mask & 0xffffff;
   DEREF(extra).instance_id = instance_id;
}

#endif
