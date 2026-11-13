#ifndef NVK_BVH_PACK_H
#define NVK_BVH_PACK_H

#include <stdint.h>

#ifdef __CUDACC__
#include "nvk_bvh_omm.h"

struct nvk_bvh_pack_vm {
   uint64_t address;
   uint32_t level;
   uint32_t type;
   uint32_t aperture;
};

static __device__ __forceinline__ nvk_bvh_pack_vm
nvk_pack_decode_vm(uint64_t metadata)
{
   nvk_bvh_pack_vm vm = {0, 15, 0, 0};
   if (metadata & NVK_OMM_ATTACHED) {
      if (metadata & NVK_OMM_DIRECT) {
         vm.level = 14;
         vm.type = (metadata >> 1) & 1u;
         vm.aperture = metadata & 1u;
      } else {
         vm.address = (metadata & NVK_OMM_ADDRESS_MASK) << 3;
         vm.level = (metadata >> NVK_OMM_META_LEVEL_SHIFT) & 15u;
         vm.type = (metadata >> NVK_OMM_META_TWO_STATE_SHIFT) & 1u;
      }
   }
   return vm;
}
#endif

struct nvk_bvh_pack_args {
   uint64_t ir;
   uint64_t dst;
   uint64_t plan;
   uint64_t span_sums;
   uint64_t workspace;
   uint32_t ir_leaf_size;
   uint32_t leaf_capacity;
   uint32_t span_length;
   uint32_t pass;
};

struct nvk_bvh_pack_plan {
   uint32_t header_or_vertices;
   uint32_t block;
   uint32_t local_and_vmap;
   uint32_t first_rank;
};

struct nvk_bvh_pack_span {
   uint32_t blocks;
   uint32_t first_block;
};

#endif
