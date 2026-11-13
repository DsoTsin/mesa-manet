#include "nvk_bvh_omm.h"
#include <stddef.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#define NVK_OMM_HD __host__ __device__ __forceinline__
#else
#define NVK_OMM_HD inline
#endif

static_assert(sizeof(nvk_bvh_omm_build_args) == 96, "build_args");
static_assert(sizeof(nvk_bvh_omm_copy_args) == 40, "copy_args");
static_assert(sizeof(nvk_bvh_omm_descriptor) == 8, "descriptor");
static_assert(sizeof(nvk_bvh_omm_group) == 16, "group");
static_assert(offsetof(nvk_bvh_omm_build_args, stride) == 48, "stride");
static_assert(offsetof(nvk_bvh_omm_build_args, uuid) == 64, "uuid");

NVK_OMM_HD uint64_t nvk_omm_load64(uint64_t address)
{
#ifdef __CUDA_ARCH__
   uint64_t value;
   asm volatile("ld.global.cg.b64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
   return value;
#else
   return *reinterpret_cast<const uint64_t *>(address);
#endif
}

NVK_OMM_HD uint32_t nvk_omm_load32(uint64_t address)
{
#ifdef __CUDA_ARCH__
   uint32_t value;
   asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
   return value;
#else
   return *reinterpret_cast<const uint32_t *>(address);
#endif
}

NVK_OMM_HD void nvk_omm_store64(uint64_t address, uint64_t value)
{
#ifdef __CUDA_ARCH__
   asm volatile("st.global.cg.b64 [%0], %1;" : : "l"(address), "l"(value) : "memory");
#else
   *reinterpret_cast<uint64_t *>(address) = value;
#endif
}

NVK_OMM_HD void nvk_omm_store32(uint64_t address, uint32_t value)
{
#ifdef __CUDA_ARCH__
   asm volatile("st.global.cg.b32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
#else
   *reinterpret_cast<uint32_t *>(address) = value;
#endif
}

NVK_OMM_HD nvk_bvh_omm_descriptor nvk_omm_descriptor(const nvk_bvh_omm_build_args &args, uint32_t index)
{
   return *reinterpret_cast<const nvk_bvh_omm_descriptor *>(args.triangles + uint64_t(index) * args.stride);
}

NVK_OMM_HD uint32_t nvk_omm_bucket(uint32_t size)
{
   return size > 127u ? 0u : size == 64u ? 1u : 2u;
}

NVK_OMM_HD uint32_t nvk_omm_group_location(const uint32_t sizes[32], uint32_t lane)
{
   uint32_t offset = 0, bucket = nvk_omm_bucket(sizes[lane]);
   for (uint32_t index = 0; index < 32; index++)
      if (nvk_omm_bucket(sizes[index]) < bucket ||
          (nvk_omm_bucket(sizes[index]) == bucket && index < lane))
         offset += sizes[index];
   return offset;
}

NVK_OMM_HD uint64_t nvk_omm_metadata(uint64_t address, uint32_t level, uint32_t format, uint32_t flags)
{
   return (address & NVK_OMM_ADDRESS_MASK) |
          (uint64_t(level & 15u) << NVK_OMM_META_LEVEL_SHIFT) |
          (uint64_t(flags & 15u) << NVK_OMM_META_FLAGS_SHIFT) |
          (format == 1u ? NVK_OMM_TWO_STATE : 0ull);
}

NVK_OMM_HD uint64_t nvk_omm_rebase_metadata(uint64_t value, uint64_t delta)
{
   return (value & ~NVK_OMM_ADDRESS_MASK) | ((value + delta) & NVK_OMM_ADDRESS_MASK);
}

NVK_OMM_HD uint64_t nvk_omm_header_word(const nvk_bvh_omm_build_args &args, uint32_t offset)
{
   if (offset == 0) return 0x0000000800000004ull;
   if (offset == 8) return args.size;
   if (offset == 16) return args.uuid[0] | (uint64_t(args.uuid[1]) << 32);
   if (offset == 24) return args.uuid[2] | (uint64_t(args.uuid[3]) << 32);
   if (offset == 32) return args.uuid[4] | (uint64_t(args.uuid[5]) << 32);
   if (offset == 40) return args.uuid[6] | (uint64_t(args.uuid[7]) << 32);
   if (offset == 48) return args.dst + nvk_omm_data_offset(args.count);
   if (offset == 56) return args.dst;
   if (offset == 64) return 0x01010101ull;
   if (offset == 80) return args.flags | (uint64_t(args.count) << 32);
   if (offset == 96) return 128ull;
   if (offset == 104) return args.data_size;
   return 0;
}

NVK_OMM_HD uint32_t nvk_omm_deinterleave(uint32_t value)
{
   value &= 0x55555555u;
   value = (value | (value >> 1)) & 0x33333333u;
   value = (value | (value >> 2)) & 0x0f0f0f0fu;
   value = (value | (value >> 4)) & 0x00ff00ffu;
   return (value | (value >> 8)) & 0xffffu;
}

NVK_OMM_HD void nvk_omm_micro_bounds(uint32_t index, uint32_t level, uint32_t minimum[3], uint32_t maximum[3])
{
   uint32_t d = nvk_omm_deinterleave(index) | (nvk_omm_deinterleave(index >> 1) << 16);
   uint32_t f = (d & 0xffffu) | ((d << 16) & ~d);
   for (uint32_t shift = 1; shift <= 8; shift *= 2) {
      uint32_t mask = 0xffffu >> shift;
      f ^= (f >> shift) & (mask | (mask << 16));
   }
   uint32_t t = (f ^ d) >> 16;
   uint32_t mask = (1u << level) - 1u;
   uint32_t u = ((f & ~t) | (d & ~t) | (~d & ~f & t)) & mask;
   uint32_t v = ((f >> 16) ^ d) & mask;
   uint32_t w = ((~f & ~t) | (d & ~t) | (~d & f & t)) & mask;
   uint32_t upright = (u ^ v ^ w) & 1u;
   minimum[0] = u;
   minimum[1] = v;
   minimum[2] = (1u << level) - u - v - (upright ? 1u : 2u);
   for (uint32_t axis = 0; axis < 3; axis++)
      maximum[axis] = minimum[axis] + 1u;
   if (level == 0)
      for (uint32_t axis = 0; axis < 3; axis++) { minimum[axis] = 0; maximum[axis] = 1; }
}

NVK_OMM_HD uint32_t nvk_omm_occupancy_mask(uint32_t minimum, uint32_t maximum, uint32_t level)
{
   uint32_t n = 1u << level;
   uint32_t lo = minimum * 32u / n;
   uint32_t hi = (maximum * 32u + n - 1u) / n - 1u;
   return uint32_t(((1ull << (hi + 1u)) - 1ull) & ~((1ull << lo) - 1ull));
}

NVK_OMM_HD uint32_t nvk_omm_hull_masks(const uint32_t masks[3])
{
   if (!(masks[0] | masks[1] | masks[2]))
      return NVK_OMM_EMPTY_HULL;
   uint32_t result = 0;
   for (uint32_t axis = 0; axis < 3; axis++) {
      uint32_t lo = 0, hi = 31;
      while (((masks[axis] >> lo) & 1u) == 0) lo++;
      while (((masks[axis] >> hi) & 1u) == 0) hi--;
      result |= lo << (10u * axis);
      result |= hi << (10u * axis + 5u);
   }
   if ((masks[0] & masks[1] & masks[2]) == 0xffffffffu)
      result |= NVK_OMM_HULL_FULL;
   return result;
}

NVK_OMM_HD uint64_t nvk_omm_copy_value(uint64_t value, uint64_t offset, uint64_t destination,
                                    uint64_t old_base, uint32_t count, uint32_t mode)
{
   uint64_t delta = mode == NVK_OMM_COPY_SERIALIZE ? 0ull - old_base : destination - old_base;
   if (offset == 48)
      return value + delta;
   if (offset == 56)
      return mode == NVK_OMM_COPY_SERIALIZE ? 0ull : destination;
   if (offset >= 128) {
      uint64_t relative = offset - 128;
      uint64_t slot = relative % 256;
      uint64_t index = (relative / 256) * 16 + slot / 8;
      if (slot < 128 && index < count)
         return nvk_omm_rebase_metadata(value, delta);
   }
   return value;
}

#ifdef __CUDACC__

__device__ __forceinline__ uint32_t nvk_omm_warp_inclusive(uint32_t value)
{
   unsigned lane = threadIdx.x & 31u;
   for (unsigned shift = 1; shift < 32; shift *= 2) {
      uint32_t previous = __shfl_up_sync(0xffffffffu, value, shift);
      if (lane >= shift) value += previous;
   }
   return value;
}

extern "C" __global__ __launch_bounds__(128) void nvk_bvh_omm_init_native(nvk_bvh_omm_build_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t stride = gridDim.x * blockDim.x;
   uint64_t data_offset = nvk_omm_data_offset(args.count);
   for (uint64_t offset = uint64_t(index) * 8; offset < data_offset; offset += uint64_t(stride) * 8) {
      uint64_t value = 0;
      if (offset < 128)
         value = nvk_omm_header_word(args, uint32_t(offset));
      else {
         uint64_t relative = offset - 128;
         uint32_t map = uint32_t((relative / 256) * 16 + (relative % 128) / 8);
         if ((relative % 256) >= 128 && map < args.count)
            value = NVK_OMM_DEFAULT_HULL;
      }
      nvk_omm_store64(args.dst + offset, value);
   }
   auto *groups = reinterpret_cast<nvk_bvh_omm_group *>(args.scratch);
   for (uint32_t group = index / 32; group < nvk_omm_group_count(args.count); group += stride / 32) {
      uint32_t map = group * 32 + (threadIdx.x & 31u);
      uint32_t bytes = 0;
      if (map < args.count) {
         auto descriptor = nvk_omm_descriptor(args, map);
         bytes = uint32_t(nvk_omm_map_size(descriptor.subdivision_level, descriptor.format));
      }
      uint32_t total = __reduce_add_sync(0xffffffffu, bytes);
      if ((threadIdx.x & 31u) == 0) {
         nvk_omm_store64(uint64_t(&groups[group].prefix_bytes), 0);
         nvk_omm_store64(uint64_t(&groups[group].size_bytes), (total + 127u) & ~127u);
      }
   }
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_omm_scan_tiles_native(nvk_bvh_omm_build_args args)
{
   using Scan = cub::BlockScan<unsigned long long, 256, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
   __shared__ typename Scan::TempStorage temporary;
   uint32_t count = nvk_omm_group_count(args.count);
   auto *groups = reinterpret_cast<nvk_bvh_omm_group *>(args.scratch);
   auto *sums = reinterpret_cast<unsigned long long *>(args.scratch + nvk_omm_scan_sums_offset(args.count));
   uint32_t blocks = (count + 255u) / 256u;
   for (uint32_t block = blockIdx.x; block < blocks; block += gridDim.x) {
      uint32_t group = block * 256u + threadIdx.x;
      unsigned long long input = group < count ? nvk_omm_load32(uint64_t(&groups[group].size_bytes)) : 0ull;
      unsigned long long prefix, aggregate;
      Scan(temporary).ExclusiveSum(input, prefix, aggregate);
      if (group < count) nvk_omm_store64(uint64_t(&groups[group].prefix_bytes), prefix);
      if (threadIdx.x == 0) nvk_omm_store64(uint64_t(sums + block), aggregate);
      __syncthreads();
   }
}

struct nvk_omm_prefix_callback {
   unsigned long long running;
   __device__ __forceinline__ unsigned long long operator()(unsigned long long aggregate)
   {
      unsigned long long previous = running;
      running += aggregate;
      return previous;
   }
};

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_omm_scan_sums_native(nvk_bvh_omm_build_args args)
{
   if (blockIdx.x != 0) return;
   using Scan = cub::BlockScan<unsigned long long, 256, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
   __shared__ typename Scan::TempStorage temporary;
   uint32_t blocks = (nvk_omm_group_count(args.count) + 255u) / 256u;
   auto *sums = reinterpret_cast<unsigned long long *>(args.scratch + nvk_omm_scan_sums_offset(args.count));
   nvk_omm_prefix_callback callback = {0};
   for (uint32_t base = 0; base < blocks; base += 256u) {
      uint32_t index = base + threadIdx.x;
      unsigned long long input = index < blocks ? nvk_omm_load64(uint64_t(sums + index)) : 0ull, prefix;
      Scan(temporary).ExclusiveSum(input, prefix, callback);
      if (index < blocks) nvk_omm_store64(uint64_t(sums + blocks + index), prefix);
      __syncthreads();
   }
}

extern "C" __global__ __launch_bounds__(128) void nvk_bvh_omm_scan_add_native(nvk_bvh_omm_build_args args)
{
   uint32_t count = nvk_omm_group_count(args.count);
   auto *groups = reinterpret_cast<nvk_bvh_omm_group *>(args.scratch);
   uint32_t blocks = (count + 255u) / 256u;
   auto *prefixes = reinterpret_cast<const uint64_t *>(args.scratch + nvk_omm_scan_sums_offset(args.count)) + blocks;
   for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += gridDim.x * blockDim.x)
      nvk_omm_store64(uint64_t(&groups[index].prefix_bytes),
                      nvk_omm_load64(uint64_t(&groups[index].prefix_bytes)) + nvk_omm_load64(uint64_t(prefixes + index / 256u)));
}

extern "C" __global__ __launch_bounds__(128) void nvk_bvh_omm_transcode_native(nvk_bvh_omm_build_args args)
{
   uint32_t lane = threadIdx.x & 31u;
   uint32_t group_count = nvk_omm_group_count(args.count);
   if (group_count == 0) return;
   auto *groups = reinterpret_cast<const nvk_bvh_omm_group *>(args.scratch);
   uint64_t total = nvk_omm_load64(uint64_t(&groups[group_count - 1].prefix_bytes)) + nvk_omm_load32(uint64_t(&groups[group_count - 1].size_bytes));
   uint64_t data_offset = nvk_omm_data_offset(args.count);
   for (uint64_t line = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
        line * 128 < total; line += uint64_t(gridDim.x) * blockDim.x / 32) {
      uint64_t offset = line * 128;
      uint32_t low = 0, high = group_count;
      while (low + 1 < high) {
         uint32_t middle = low + (high - low) / 2;
         if (nvk_omm_load64(uint64_t(&groups[middle].prefix_bytes)) <= offset) low = middle;
         else high = middle;
      }
      uint32_t map = low * 32 + lane;
      nvk_bvh_omm_descriptor descriptor = {};
      uint32_t size = 0;
      if (map < args.count) {
         descriptor = nvk_omm_descriptor(args, map);
         size = uint32_t(nvk_omm_map_size(descriptor.subdivision_level, descriptor.format));
      }
      uint32_t bucket = nvk_omm_bucket(size);
      uint32_t large = nvk_omm_warp_inclusive(bucket == 0 ? size : 0);
      uint32_t medium = nvk_omm_warp_inclusive(bucket == 1 ? size : 0);
      uint32_t small = nvk_omm_warp_inclusive(bucket == 2 ? size : 0);
      uint32_t large_total = __shfl_sync(0xffffffffu, large, 31);
      uint32_t medium_total = __shfl_sync(0xffffffffu, medium, 31);
      uint32_t position = bucket == 0 ? large - size : bucket == 1 ? large_total + medium - size : large_total + medium_total + small - size;
      uint64_t group_prefix = nvk_omm_load64(uint64_t(&groups[low].prefix_bytes));
      uint32_t local = uint32_t(offset - group_prefix);
      if (local == 0 && map < args.count) {
         uint64_t address = args.dst + data_offset + group_prefix + position;
         nvk_omm_store64(args.dst + nvk_omm_metadata_offset(map),
                          nvk_omm_metadata(address, descriptor.subdivision_level, descriptor.format, args.flags));
      }
      uint32_t word = 0;
      if (local < large_total) {
         uint32_t owner_mask = __ballot_sync(0xffffffffu, size > 127u && position <= local && local < position + size);
         uint32_t owner = uint32_t(__ffs(int(owner_mask)) - 1);
         uint32_t input = __shfl_sync(0xffffffffu, descriptor.data_offset, owner);
         uint32_t beginning = __shfl_sync(0xffffffffu, position, owner);
         const uint8_t *source = reinterpret_cast<const uint8_t *>(args.data + input + local - beginning + lane * 4u);
         for (uint32_t byte = 0; byte < 4; byte++) word |= uint32_t(source[byte]) << (byte * 8u);
      } else {
         for (uint32_t source_lane = 0; source_lane < 32; source_lane++) {
            uint32_t beginning = __shfl_sync(0xffffffffu, position, source_lane);
            uint32_t bytes = __shfl_sync(0xffffffffu, size, source_lane);
            uint32_t input = __shfl_sync(0xffffffffu, descriptor.data_offset, source_lane);
            for (uint32_t byte = 0; byte < 4; byte++) {
               uint32_t at = local + lane * 4u + byte;
               if (at >= beginning && at - beginning < bytes)
                  word |= uint32_t(*reinterpret_cast<const uint8_t *>(args.data + input + at - beginning)) << (byte * 8u);
            }
         }
      }
      nvk_omm_store32(args.dst + data_offset + offset + lane * 4u, word);
   }
}

struct nvk_omm_bit_or {
   __device__ __forceinline__ uint32_t operator()(uint32_t left, uint32_t right) const { return left | right; }
};

extern "C" __global__ __launch_bounds__(128) void nvk_bvh_omm_hull_native(nvk_bvh_omm_build_args args)
{
   using Reduce = cub::BlockReduce<uint32_t, 128>;
   __shared__ typename Reduce::TempStorage temporary;
   __shared__ uint32_t result[3];
   for (uint32_t map = blockIdx.x; map < args.count; map += gridDim.x) {
      auto descriptor = nvk_omm_descriptor(args, map);
      uint32_t level = descriptor.subdivision_level;
      if ((args.flags & 2u) != 0 || level == 0 || level >= 10) continue;
      uint32_t masks[3] = {};
      uint32_t width = descriptor.format == 1u ? 1u : 2u;
      const uint8_t *data = reinterpret_cast<const uint8_t *>(args.data + descriptor.data_offset);
      for (uint32_t index = threadIdx.x; index < (1u << (2u * level)); index += blockDim.x) {
         uint32_t state = (data[(index * width) / 8u] >> ((index * width) & 7u)) & ((1u << width) - 1u);
         if (state == 0) continue;
         uint32_t minimum[3], maximum[3];
         nvk_omm_micro_bounds(index, level, minimum, maximum);
         for (uint32_t axis = 0; axis < 3; axis++)
            masks[axis] |= nvk_omm_occupancy_mask(minimum[axis], maximum[axis], level);
      }
      for (uint32_t axis = 0; axis < 3; axis++) {
         uint32_t value = Reduce(temporary).Reduce(masks[axis], nvk_omm_bit_or());
         if (threadIdx.x == 0) result[axis] = value;
         __syncthreads();
      }
      if (threadIdx.x == 0)
         nvk_omm_store64(args.dst + nvk_omm_auxiliary_offset(map), nvk_omm_hull_masks(result));
      __syncthreads();
   }
}

__device__ __forceinline__ void nvk_omm_relocate(nvk_bvh_omm_copy_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t stride = gridDim.x * blockDim.x;
   uint64_t old_base = nvk_omm_load64(args.dst + 56);
   uint64_t delta = args.dst - old_base;
   uint32_t count = nvk_omm_load32(args.dst + 84);
   if (index == 0)
      nvk_omm_store64(args.dst + 48, nvk_omm_load64(args.dst + 48) + delta);
   for (uint32_t map = index; map < count; map += stride) {
      auto *metadata = reinterpret_cast<uint64_t *>(args.dst + nvk_omm_metadata_offset(map));
      nvk_omm_store64(uint64_t(metadata), nvk_omm_rebase_metadata(nvk_omm_load64(uint64_t(metadata)), delta));
   }
}

extern "C" __global__ __launch_bounds__(128) void nvk_bvh_omm_relocate_native(nvk_bvh_omm_copy_args args)
{
   nvk_omm_relocate(args);
}

extern "C" __global__ __launch_bounds__(128) void nvk_bvh_omm_relocate_finish_native(nvk_bvh_omm_copy_args args)
{
   if (blockIdx.x == 0 && threadIdx.x == 0)
      nvk_omm_store64(args.dst + 56, args.dst);
}

extern "C" __global__ __launch_bounds__(128) void nvk_bvh_omm_copy_native(nvk_bvh_omm_copy_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t stride = gridDim.x * blockDim.x;
   if (args.mode == NVK_OMM_COPY_RELOCATE) { nvk_omm_relocate(args); return; }
   if (args.mode == NVK_OMM_COPY_PROPERTY) {
      if (index == 0) {
         uint64_t value = nvk_omm_load64(args.src + 8);
         if (args.field == NVK_OMM_PROPERTY_SERIALIZED_SIZE) value += 56;
         nvk_omm_store64(args.dst, value);
         nvk_omm_store64(args.dst + 8, 0);
         __threadfence_system();
         if (args.available) *reinterpret_cast<volatile uint32_t *>(args.available) = 1;
      }
      return;
   }
   uint64_t source = args.src + (args.mode == NVK_OMM_COPY_DESERIALIZE ? 56ull : 0ull);
   uint64_t destination = args.dst + (args.mode == NVK_OMM_COPY_SERIALIZE ? 56ull : 0ull);
   uint64_t size = nvk_omm_load64(source + 8);
   uint64_t old_base = nvk_omm_load64(source + 56);
   uint32_t count = nvk_omm_load32(source + 84);
   if (args.mode == NVK_OMM_COPY_SERIALIZE && index == 0) {
      for (uint32_t word = 0; word < 4; word++)
         nvk_omm_store64(args.dst + word * 8u, nvk_omm_load64(source + 16 + word * 8u));
      nvk_omm_store64(args.dst + 32, size + 56);
      nvk_omm_store64(args.dst + 40, size);
      nvk_omm_store64(args.dst + 48, 0);
   }
   for (uint64_t word = index; word < size / 8; word += stride) {
      uint64_t offset = word * 8;
      uint64_t value = nvk_omm_load64(source + offset);
      value = nvk_omm_copy_value(value, offset, args.dst, old_base, count, args.mode);
      nvk_omm_store64(destination + offset, value);
   }
}

#endif
