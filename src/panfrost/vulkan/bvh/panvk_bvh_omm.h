#ifndef PANVK_BVH_OMM_H
#define PANVK_BVH_OMM_H

#include "panvk_micromap.h"

uint32_t
panvk_omm_index(panvk_omm_geometry geometry, uint32_t primitive)
{
   uint64_t address = geometry.indices + uint64_t(primitive) * geometry.index_stride;
   uint32_t index = primitive;
   uint32_t mask = ~0u;
   if (geometry.index_format == VK_INDEX_TYPE_UINT8_EXT) {
      index = uint32_t(DEREF(REF(uint8_t)(address)));
      mask = 255;
   } else if (geometry.index_format == VK_INDEX_TYPE_UINT16) {
      index = uint32_t(DEREF(REF(uint16_t)(address)));
      mask = 65535;
   } else if (geometry.index_format == VK_INDEX_TYPE_UINT32) {
      index = DEREF(REF(uint32_t)(address));
   }
   if (index >= mask - 3)
      index |= ~mask;
   return index;
}

bool
panvk_write_omm(REF(panvk_bvh_triangle_leaf) leaf, panvk_omm_geometry geometry,
                 uint32_t primitive, uint32_t opaque, uint32_t geometry_id)
{
   if (geometry.enabled == 0)
      return false;

   uint32_t index = panvk_omm_index(geometry, primitive);
   uint32_t level = 3;
   uint32_t bits = 2;
   uint64_t address = 0;
   uint64_t summary = 0;
   uint32_t values[4];

   if (index >= 0xfffffffcu) {
      uint32_t value = ~index;
      for (uint32_t i = 0; i < 4; i++)
         values[i] = value * 0x55555555u;
   } else {
      index += geometry.base_triangle;
      uint64_t entry = DEREF(INDEX(uint64_t, geometry.micromap + PANVK_MICROMAP_HEADER_SIZE, index));
      level = uint32_t(entry >> 48) & 255;
      bits = uint32_t(entry >> 56);
      address = geometry.micromap + (entry & 0xfffffffffffful);
      summary = DEREF(REF(panvk_micromap_header)(geometry.micromap)).summary_offset;
      if (level <= 3) {
         summary = address;
         address = 0;
      } else if (summary != 0) {
         summary += geometry.micromap + uint64_t(index) * 16;
      }
      for (uint32_t i = 0; i < 4; i++)
         values[i] = summary != 0 ? DEREF(INDEX(uint32_t, summary, i)) : ~0u;
   }

   uint64_t tail = uint64_t(leaf) + 36;
   for (uint32_t i = 0; i < 4; i++)
      DEREF(INDEX(uint32_t, tail, i)) = values[i];
   DEREF(leaf).primitive_id_a = uint32_t(address >> 6);
   DEREF(leaf).geometry_id_b = 0x40000000u | uint32_t(address >> 38);
   uint32_t flags = opaque | (4 * level) | (2 * (bits - 1));
   DEREF(leaf).geometry_id_a_flags = (flags << 24) | geometry_id;
   return true;
}

#endif
