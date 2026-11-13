#include "nvk_bvh.h"
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t
nvk_bw_copy_load(uint64_t address, uint32_t offset)
{
   return *reinterpret_cast<const uint32_t *>(address + offset);
}

__device__ __forceinline__ uint64_t
nvk_bw_copy_load64(uint64_t address, uint32_t offset)
{
   return *reinterpret_cast<const uint64_t *>(address + offset);
}

__device__ __forceinline__ void
nvk_bw_copy_store(uint64_t address, uint32_t offset, uint32_t value)
{
   *reinterpret_cast<uint32_t *>(address + offset) = value;
}

__device__ __forceinline__ void
nvk_bw_copy_store64(uint64_t address, uint32_t offset, uint64_t value)
{
   *reinterpret_cast<uint64_t *>(address + offset) = value;
}

extern "C" __global__ void
nvk_bvh_copy(nvk_bvh_copy_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t stride = gridDim.x * blockDim.x;
   if (args.mode == 3) {
      if (index == 0) {
         uint64_t value = args.field == 0xffffffffu ?
            uint64_t(nvk_bw_copy_load(args.src, NVK_BVH_INSTANCE_COUNT_OFFSET)) :
            nvk_bw_copy_load64(args.src, args.field);
         nvk_bw_copy_store64(args.dst, 0, value);
         nvk_bw_copy_store64(args.dst, 8, 0);
         __threadfence_system();
         *reinterpret_cast<volatile uint32_t *>(args.available) = 1;
      }
      return;
   }
   if (args.mode == 4) {
      if (index == 0)
         nvk_bw_copy_store64(args.dst, 208, args.dst);
      if (args.field == 0 || nvk_bw_copy_load(args.dst, NVK_BVH_GEOMETRY_TYPE_OFFSET) != 2)
         return;
      uint32_t count = nvk_bw_copy_load(args.dst, NVK_BVH_ACTIVE_LEAF_COUNT_OFFSET);
      uint64_t leaf_offset = nvk_bw_copy_load64(args.dst, 56);
      uint64_t metadata_offset = nvk_bw_copy_load64(args.dst, 48);
      for (uint32_t leaf = index; leaf < count; leaf += stride) {
         uint64_t node = args.dst + leaf_offset + uint64_t(leaf) * 64;
         uint32_t original_index = nvk_bw_copy_load(node, 0) - 1;
         uint64_t blas = nvk_bw_copy_load64(args.src, 56 + original_index * 8);
         uint64_t metadata = args.dst + metadata_offset + uint64_t(original_index >> 4) * 1152;
         nvk_bw_copy_store64(metadata, 128 + (original_index & 15u) * 8, blas);
         uint64_t flags = nvk_bw_copy_load64(node, 8) &
                          (~((uint64_t(1) << 49) - 1) | 0x18ull);
         nvk_bw_copy_store64(node, 8, flags | ((blas + nvk_bw_copy_load64(blas, 32)) &
                                            ((uint64_t(1) << 49) - 1)));
      }
      return;
   }
   uint64_t source = args.src;
   uint64_t destination = args.dst;
   uint64_t size = args.mode == 0 ? nvk_bw_copy_load64(source, 8) : args.size;
   if (args.mode == 1) {
      size = nvk_bw_copy_load64(source, 8);
      uint32_t instance_count = nvk_bw_copy_load(source, NVK_BVH_INSTANCE_COUNT_OFFSET);
      destination += 56 + uint64_t(instance_count) * 8;
      if (index == 0) {
         for (uint32_t uuid = 0; uuid < 2; uuid++) {
            nvk_bw_copy_store(args.dst, uuid * 16, NVK_BVH_COMPAT_UUID_0);
            nvk_bw_copy_store(args.dst, uuid * 16 + 4, NVK_BVH_COMPAT_UUID_1);
            nvk_bw_copy_store(args.dst, uuid * 16 + 8, NVK_BVH_COMPAT_UUID_2);
            nvk_bw_copy_store(args.dst, uuid * 16 + 12, NVK_BVH_COMPAT_UUID_3);
         }
         nvk_bw_copy_store64(args.dst, 32, size + 56 + uint64_t(instance_count) * 8);
         nvk_bw_copy_store64(args.dst, 40, size);
         nvk_bw_copy_store64(args.dst, 48, instance_count);
      }
      uint64_t metadata = source + nvk_bw_copy_load64(source, 48);
      for (uint32_t instance = index; instance < instance_count; instance += stride) {
         uint64_t group = metadata + uint64_t(instance >> 4) * 1152;
         nvk_bw_copy_store64(args.dst, 56 + instance * 8,
                             nvk_bw_copy_load64(group, 128 + (instance & 15u) * 8));
      }
   } else if (args.mode == 2) {
      uint64_t instance_count = nvk_bw_copy_load64(source, 48);
      size = nvk_bw_copy_load64(source, 40);
      source += 56 + instance_count * 8;
   }
   for (uint64_t word = index; word < size / 8; word += stride)
      *reinterpret_cast<uint64_t *>(destination + word * 8) =
         *reinterpret_cast<const uint64_t *>(source + word * 8);
   if (index == 0 && (size & 7u) != 0)
      nvk_bw_copy_store(destination, uint32_t(size & ~uint64_t(7)),
                         nvk_bw_copy_load(source, uint32_t(size & ~uint64_t(7))));
}
