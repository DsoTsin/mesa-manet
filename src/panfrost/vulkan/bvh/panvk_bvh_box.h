/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_BVH_BOX_H
#define PANVK_BVH_BOX_H

u32vec3
panvk_origin_bits(vec3 v)
{
   u32vec3 b = floatBitsToUint(v);
   return mix(b & ~0xffu, (b + 0xffu) & ~0xffu, notEqual(b & 0x80000000u, u32vec3(0)));
}

uint32_t
panvk_quant_exponent(float extent)
{
   if (!(extent > 0.0))
      return 1;

   int e;
   frexp(extent / 255.0, e);
   e = clamp(e, -126, 127);
   if (extent * exp2(float(-e)) > 255.0)
      e++;
   return uint32_t(clamp(e + 127, 1, 254));
}

float
panvk_sub_round_down(float a, float b)
{
   float d = a - b;
   if (d > 0.0)
      d = uintBitsToFloat(floatBitsToUint(d) - 1);
   return d;
}

float
panvk_sub_round_up(float a, float b)
{
   float d = a - b;
   if (d > 0.0)
      d = uintBitsToFloat(floatBitsToUint(d) + 1);
   return d;
}

void
panvk_write_box(REF(panvk_bvh_box_node) dst, uint32_t type,
                vk_aabb child_bounds[PANVK_BVH_MAX_CHILDREN],
                uint32_t child_desc[PANVK_BVH_MAX_CHILDREN], uint32_t child_count,
                uint32_t internal_base, uint32_t leaf_base, uint32_t leaf_count)
{
   vec3 lo = vec3(INFINITY);
   vec3 hi = vec3(-INFINITY);
   for (uint32_t i = 0; i < child_count; i++) {
      lo = min(lo, child_bounds[i].min);
      hi = max(hi, child_bounds[i].max);
   }
   if (child_count == 0) {
      lo = vec3(0.0);
      hi = vec3(0.0);
   }

   u32vec3 obits = panvk_origin_bits(lo);
   vec3 origin = uintBitsToFloat(obits);
   u32vec3 exps = u32vec3(panvk_quant_exponent(hi.x - origin.x),
                          panvk_quant_exponent(hi.y - origin.y),
                          panvk_quant_exponent(hi.z - origin.z));
   vec3 inv_scale = vec3(exp2(127.0 - float(exps.x)), exp2(127.0 - float(exps.y)),
                         exp2(127.0 - float(exps.z)));

   uint32_t bounds[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
   for (uint32_t i = 0; i < child_count; i++) {
      for (uint32_t a = 0; a < 3; a++) {
         float qlo = floor(panvk_sub_round_down(child_bounds[i].min[a], origin[a]) * inv_scale[a]);
         float qhi = ceil(panvk_sub_round_up(child_bounds[i].max[a], origin[a]) * inv_scale[a]);
         uint32_t pair = uint32_t(clamp(qlo, 0.0, 255.0)) | (uint32_t(clamp(qhi, 0.0, 255.0)) << 8);
         uint32_t byte = a * 12 + i * 2;
         bounds[byte / 4] |= pair << ((byte % 4) * 8);
      }
   }

   uint32_t desc[PANVK_BVH_MAX_CHILDREN];
   for (uint32_t i = 0; i < PANVK_BVH_MAX_CHILDREN; i++)
      desc[i] = i < child_count ? child_desc[i] : PANVK_BVH_CHILD_EMPTY;

   DEREF(dst).origin_x_type = obits.x | type;
   DEREF(dst).origin_y_exp_x = obits.y | exps.x;
   DEREF(dst).origin_z_exp_y = obits.z | exps.y;
   DEREF(dst).internal_base = internal_base;
   DEREF(dst).leaf_base = leaf_base;
   DEREF(dst).exp_z_leaf_count_child01 = exps.z | (leaf_count << 8) | (desc[0] << 16) | (desc[1] << 24);
   DEREF(dst).child2345 = desc[2] | (desc[3] << 8) | (desc[4] << 16) | (desc[5] << 24);
   DEREF(dst).bounds = bounds;
}

uint32_t
panvk_box_desc(REF(panvk_bvh_box_node) node, uint32_t i)
{
   uint32_t word = i < 2 ? DEREF(node).exp_z_leaf_count_child01 >> 16 : DEREF(node).child2345;
   uint32_t shift = i < 2 ? i * 8 : (i - 2) * 8;
   return (word >> shift) & 0xff;
}

uint32_t
panvk_box_child(REF(panvk_bvh_box_node) node, uint32_t desc)
{
   if ((desc & 0xc0) == PANVK_BVH_CHILD_INTERNAL)
      return DEREF(node).internal_base + (desc & 0x3f);
   return DEREF(node).leaf_base + desc;
}

#endif
