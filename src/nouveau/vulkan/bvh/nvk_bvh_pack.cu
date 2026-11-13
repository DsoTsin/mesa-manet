#include "nvk_bvh_pack.h"
#include "nvk_bvh.h"
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>

static_assert(sizeof(nvk_bvh_pack_args) == 56, "pack parameter size");
static_assert(sizeof(nvk_bvh_pack_plan) == 16, "pack rank descriptor size");
static_assert(sizeof(nvk_bvh_pack_span) == 8, "pack span descriptor size");

__device__ __forceinline__ uint32_t
nvk_pack_load(uint64_t address, uint32_t offset)
{
   return *reinterpret_cast<const uint32_t *>(address + offset);
}

__device__ __forceinline__ uint64_t
nvk_pack_load64(uint64_t address, uint32_t offset)
{
   return uint64_t(nvk_pack_load(address, offset)) |
          (uint64_t(nvk_pack_load(address, offset + 4)) << 32);
}

__device__ __forceinline__ void
nvk_pack_store(uint64_t address, uint32_t offset, uint32_t value)
{
   *reinterpret_cast<uint32_t *>(address + offset) = value;
}

__device__ __forceinline__ uint32_t
nvk_pack_count(const nvk_bvh_pack_args &args)
{
   return nvk_pack_load(args.dst, NVK_BVH_ACTIVE_LEAF_COUNT_OFFSET);
}

__device__ __forceinline__ uint32_t
nvk_pack_span_length(const nvk_bvh_pack_args &args, uint32_t count)
{
   return args.span_length ? max(args.span_length, 32u) : max((count + 31u) / 32u, 32u);
}

__device__ __forceinline__ uint32_t
nvk_pack_span_count(uint32_t count, uint32_t length)
{
   return (count + length - 1) / length;
}

__device__ __forceinline__ uint64_t
nvk_pack_remap(const nvk_bvh_pack_args &args)
{
   return args.dst + nvk_pack_load64(args.dst, 64);
}

__device__ __forceinline__ uint32_t
nvk_pack_lower(uint32_t lane)
{
   return (1u << lane) - 1u;
}

extern "C" __global__ __launch_bounds__(128) void
nvk_bvh_pack_plan(nvk_bvh_pack_args args)
{
   const uint32_t lane = threadIdx.x & 31u;
   const uint32_t count = nvk_pack_count(args);
   const uint32_t length = nvk_pack_span_length(args, count);
   const uint32_t spans = nvk_pack_span_count(count, length);
   auto *plan = reinterpret_cast<struct nvk_bvh_pack_plan *>(args.plan);
   auto *sums = reinterpret_cast<nvk_bvh_pack_span *>(args.span_sums);
   const uint64_t remap = nvk_pack_remap(args);
   const uint64_t metadata = args.dst + nvk_pack_load64(args.dst, NVK_BVH_OMM_METADATA_OFFSET);
   const bool opacity = (nvk_pack_load(args.dst, 0) & 0x10000000u) != 0;
   const bool single = (nvk_pack_load(args.dst, 244) & 0x140u) != 0;
   const uint32_t warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
   const uint32_t warps = gridDim.x * (blockDim.x >> 5);
   for (uint32_t span = warp; span < spans; span += warps) {
      uint32_t first = span * length;
      const uint32_t end = min(first + length, count);
      uint32_t blocks = 0;
      while (first < end) {
         const bool active = lane < 10 && first + lane < end;
         uint32_t original = 0;
         uint32_t array = 0;
         uint64_t identity[3] = {0, 0, 0};
         uint64_t vm_address = 0;
         bool vm_normal = false;
         if (active) {
            original = nvk_pack_load(remap, (first + lane) * 4);
            const uint64_t input = args.ir + uint64_t(original) * args.ir_leaf_size;
            array = nvk_pack_load(input, 64) & 0x7fffffffu;
            if (opacity) {
               const nvk_bvh_pack_vm vm = nvk_pack_decode_vm(
                  nvk_pack_load64(metadata + uint64_t(original) * 8, 0));
               vm_address = vm.address;
               vm_normal = vm.level < 14;
            }
#pragma unroll
            for (uint32_t vertex = 0; vertex < 3; vertex++)
               identity[vertex] = nvk_pack_load64(input, 68 + vertex * 8);
         }
         const uint32_t triangle = lane / 3;
         const uint32_t component = lane % 3;
         const bool vertex_active = __shfl_sync(0xffffffffu, uint32_t(active), triangle) != 0;
         uint32_t match_lo = 0, match_hi = 0;
#pragma unroll
         for (uint32_t vertex = 0; vertex < 3; vertex++) {
            const uint32_t lo = __shfl_sync(0xffffffffu, uint32_t(identity[vertex]), triangle);
            const uint32_t hi = __shfl_sync(0xffffffffu, uint32_t(identity[vertex] >> 32), triangle);
            if (component == vertex) {
               match_lo = lo;
               match_hi = hi;
            }
         }
         const uint32_t match_array = __shfl_sync(0xffffffffu, array, triangle);
         const uint32_t vertex_mask = __ballot_sync(0xffffffffu, vertex_active);
         uint32_t first_match = lane;
         if (vertex_active) {
            const uint32_t match = __match_any_sync(vertex_mask, match_lo) &
                                   __match_any_sync(vertex_mask, match_hi) &
                                   __match_any_sync(vertex_mask, match_array);
            first_match = __ffs(match) - 1;
         }
         uint32_t unique_mask = __ballot_sync(0xffffffffu, vertex_active && first_match == lane);
         unique_mask |= 7u;
         uint32_t indices = 0;
         uint32_t writers = 0;
         uint32_t vmap = 0;
#pragma unroll
         for (uint32_t vertex = 0; vertex < 3; vertex++) {
            const uint32_t position = lane * 3 + vertex;
            const uint32_t matched = __shfl_sync(0xffffffffu, first_match, min(position, 31u));
            uint32_t slot;
            bool unique;
            if (position < 32) {
               slot = __popc(unique_mask & nvk_pack_lower(matched));
               unique = matched == position;
            } else {
               slot = __popc(unique_mask) + position - 32;
               unique = true;
            }
            if (lane == 0) {
               slot = vertex;
               unique = true;
            }
            indices |= (slot & 15u) << (vertex * 4);
            writers |= uint32_t(unique) << vertex;
            vmap |= (unique ? slot : 14u) << (vertex * 5);
         }
         uint32_t vertices = active ? __popc(writers) : 0;
         const uint32_t normal_mask = __ballot_sync(0xffffffffu, active && vm_normal);
         const uint32_t base_lane = normal_mask ? __ffs(normal_mask) - 1 : 0;
         const uint64_t vm_base = uint64_t(__shfl_sync(0xffffffffu, uint32_t(vm_address), base_lane)) |
            (uint64_t(__shfl_sync(0xffffffffu, uint32_t(vm_address >> 32), base_lane)) << 32);
         const int64_t vm_delta = vm_normal ? int64_t(vm_address) - int64_t(vm_base) : 0;
         const uint64_t delta_bits = vm_delta < 0 ? uint64_t(~vm_delta) : uint64_t(vm_delta);
         uint32_t pvmo = vm_delta == 0 ? 0 : 65u - __clzll(delta_bits);
         const uint32_t base_id = __shfl_sync(0xffffffffu, original, 0);
         uint32_t id_diff = active ? original ^ base_id : 0;
#pragma unroll
         for (uint32_t delta = 1; delta < 16; delta <<= 1) {
            const uint32_t previous_vertices = __shfl_up_sync(0xffffffffu, vertices, delta, 16);
            const uint32_t previous_ids = __shfl_up_sync(0xffffffffu, id_diff, delta, 16);
            const uint32_t previous_pvmo = __shfl_up_sync(0xffffffffu, pvmo, delta, 16);
            if ((lane & 15u) >= delta) {
               vertices += previous_vertices;
               id_diff |= previous_ids;
               pvmo = max(pvmo, previous_pvmo);
            }
         }
         const uint32_t pid = max(1, 32 - __clz(id_diff));
         const uint32_t bits = 162u + lane * (pid + 14u) + 96u * (vertices - 1u) +
            (opacity ? 59u + 6u * (lane + 1u) + lane * pvmo : 0u);
         const bool fits = active && vertices <= 9 && bits <= 1024 && (!single || lane == 0);
         const uint32_t fitting = __ballot_sync(0xffffffffu, fits);
         const uint32_t last = 31u - __clz(fitting);
         const uint32_t included = last + 1;
         const uint32_t block_pid = __shfl_sync(0xffffffffu, pid, last);
         const uint32_t block_pvmo = __shfl_sync(0xffffffffu, pvmo, last);
         const uint32_t header = (opacity ? 0x60007fffu : 0x20007fffu) |
            ((block_pid - 1u) << 15) | (last << 20);
         if (active && lane < included) {
            struct nvk_bvh_pack_plan value;
            value.header_or_vertices = lane == 0 ? header : indices | (writers << 12);
            value.block = blocks;
            value.local_and_vmap = lane | ((vmap | (lane == last ? 0x8000u : 0u)) << 8) |
               (block_pvmo << 24);
            value.first_rank = first;
            plan[first + lane] = value;
         }
         first += included;
         blocks++;
      }
      if (lane == 0) {
         sums[span].blocks = blocks;
         sums[span].first_block = 0;
      }
   }
}

extern "C" __global__ __launch_bounds__(256) void
nvk_bvh_pack_prefix(nvk_bvh_pack_args args)
{
   if (blockIdx.x != 0)
      return;
   using Scan = cub::BlockScan<uint32_t, 256>;
   __shared__ typename Scan::TempStorage temporary;
   __shared__ uint32_t carry;
   auto *sums = reinterpret_cast<nvk_bvh_pack_span *>(args.span_sums);
   const uint32_t count = nvk_pack_count(args);
   const uint32_t spans = nvk_pack_span_count(count, nvk_pack_span_length(args, count));
   if (threadIdx.x == 0)
      carry = 0;
   __syncthreads();
   for (uint32_t base = 0; base < spans; base += 256) {
      const uint32_t span = base + threadIdx.x;
      const uint32_t blocks = span < spans ? sums[span].blocks : 0;
      uint32_t first, total;
      Scan(temporary).ExclusiveSum(blocks, first, total);
      if (span < spans)
         sums[span].first_block = carry + first;
      __syncthreads();
      if (threadIdx.x == 0)
         carry += total;
      __syncthreads();
   }
   if (threadIdx.x == 0) {
      sums[spans].blocks = 0;
      sums[spans].first_block = carry;
   }
}

__device__ __forceinline__ void
nvk_pack_bits(uint32_t *words, uint32_t offset, uint32_t value, uint32_t width)
{
   const uint32_t bit = offset & 31u;
   if (width < 32)
      value &= (1u << width) - 1u;
   atomicOr(words + (offset >> 5), value << bit);
   if (bit != 0 && width > 32 - bit)
      atomicOr(words + (offset >> 5) + 1, value >> (32 - bit));
}

extern "C" __global__ __launch_bounds__(128) void
nvk_bvh_pack_emit(nvk_bvh_pack_args args)
{
   __shared__ uint32_t storage[4][32];
   const uint32_t lane = threadIdx.x & 31u;
   uint32_t *words = storage[threadIdx.x >> 5];
   const uint32_t count = nvk_pack_count(args);
   const uint32_t length = nvk_pack_span_length(args, count);
   const uint32_t spans = nvk_pack_span_count(count, length);
   const auto *plan = reinterpret_cast<const struct nvk_bvh_pack_plan *>(args.plan);
   const auto *sums = reinterpret_cast<const nvk_bvh_pack_span *>(args.span_sums);
   const uint64_t remap = nvk_pack_remap(args);
   const uint64_t blocks = args.dst + nvk_pack_load64(args.dst, 16);
   const uint64_t metadata = args.dst + nvk_pack_load64(args.dst, NVK_BVH_OMM_METADATA_OFFSET);
   auto *vmap = reinterpret_cast<uint16_t *>(args.dst + uint64_t(nvk_pack_load(args.dst, 240)) * 128);
   const uint32_t warp = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
   const uint32_t warps = gridDim.x * (blockDim.x >> 5);
   for (uint32_t span = warp; span < spans; span += warps) {
      uint32_t first = span * length;
      const uint32_t end = min(first + length, count);
      while (first < end) {
         const struct nvk_bvh_pack_plan base = plan[first];
         const uint32_t header = base.header_or_vertices;
         const uint32_t triangles = ((header >> 20) & 15u) + 1;
         const uint32_t pid = ((header >> 15) & 31u) + 1;
         const bool opacity = (header >> 29) == 3;
         const uint32_t pvmo = opacity ? base.local_and_vmap >> 24 : 0;
         uint64_t vm_address = 0;
         nvk_bvh_pack_vm vm = {0, 15, 0, 0};
         if (opacity && lane < triangles) {
            const uint32_t original = nvk_pack_load(remap, (first + lane) * 4);
            vm = nvk_pack_decode_vm(nvk_pack_load64(metadata + uint64_t(original) * 8, 0));
            vm_address = vm.address;
         }
         const uint32_t normal_mask = __ballot_sync(0xffffffffu, lane < triangles && vm.level < 14);
         const uint32_t base_lane = normal_mask ? __ffs(normal_mask) - 1 : 0;
         const uint64_t vm_base = uint64_t(__shfl_sync(0xffffffffu, uint32_t(vm_address), base_lane)) |
            (uint64_t(__shfl_sync(0xffffffffu, uint32_t(vm_address >> 32), base_lane)) << 32);
         const uint32_t anchor = opacity ? 901u : 960u;
         words[lane] = 0;
         __syncwarp();
         if (opacity && lane == 0) {
            nvk_pack_bits(words, 901, uint32_t(vm_base), 32);
            nvk_pack_bits(words, 933, uint32_t(vm_base >> 32), 20);
            nvk_pack_bits(words, 953, 1, 1);
            nvk_pack_bits(words, 954, pvmo, 6);
         }
         if (lane < triangles) {
            const uint32_t rank = first + lane;
            const struct nvk_bvh_pack_plan value = plan[rank];
            const uint32_t mapping = (value.local_and_vmap >> 8) & 0xffffu;
            const uint32_t original = nvk_pack_load(remap, rank * 4);
            const uint64_t input = args.ir + uint64_t(original) * args.ir_leaf_size;
#pragma unroll
            for (uint32_t vertex = 0; vertex < 3; vertex++) {
               const uint32_t slot = (mapping >> (vertex * 5)) & 31u;
               if (slot <= 8) {
#pragma unroll
                  for (uint32_t axis = 0; axis < 3; axis++)
                     words[slot * 3 + axis] = nvk_pack_load(input, 24 + (vertex * 3 + axis) * 4);
               }
            }
            if (lane == 0) {
               words[30] = original;
               words[31] = header;
            } else {
               nvk_pack_bits(words, anchor - 14u * lane, value.header_or_vertices, 12);
               nvk_pack_bits(words, anchor + 12u - 14u * triangles - lane * pid, original, pid);
            }
            if ((nvk_pack_load(input, 64) & 0x80000000u) == 0)
               nvk_pack_bits(words, anchor - 1u - 14u * lane, 1, 1);
            if (opacity) {
               uint32_t level = anchor - (14u * triangles - 12u) -
                  (triangles - 1u) * pid - 4u;
               if (lane != 0)
                  level -= 6u * lane + pvmo * (lane - 1u);
               nvk_pack_bits(words, level, vm.level, 4);
               nvk_pack_bits(words, level - 1u, vm.type, 1);
               nvk_pack_bits(words, level - 2u, vm.aperture, 1);
               if (lane != 0 && pvmo != 0) {
                  const uint64_t delta = vm.level < 14 ? vm.address - vm_base : 0;
                  nvk_pack_bits(words, level - 2u - pvmo, uint32_t(delta), min(pvmo, 32u));
                  if (pvmo > 32)
                     nvk_pack_bits(words, level - 2u - pvmo + 32u,
                                    uint32_t(delta >> 32), pvmo - 32u);
               }
            }
            vmap[rank] = mapping;
         }
         __syncwarp();
         const uint32_t output = sums[span].first_block + base.block;
         nvk_pack_store(blocks + uint64_t(output) * 128, lane * 4, words[lane]);
         __syncwarp();
         first += triangles;
      }
   }
}

__device__ __forceinline__ uint2
nvk_pack_cursor(const nvk_bvh_pack_args &args, uint32_t rank,
                uint32_t count, uint32_t length)
{
   const auto *sums = reinterpret_cast<const nvk_bvh_pack_span *>(args.span_sums);
   if (rank == count) {
      const uint32_t spans = nvk_pack_span_count(count, length);
      return make_uint2(sums[spans].first_block, 0);
   }
   const auto *plan = reinterpret_cast<const struct nvk_bvh_pack_plan *>(args.plan);
   const struct nvk_bvh_pack_plan value = plan[rank];
   return make_uint2(sums[rank / length].first_block + value.block,
                     value.local_and_vmap & 15u);
}

extern "C" __global__ __launch_bounds__(128) void
nvk_bvh_pack_patch(nvk_bvh_pack_args args)
{
   const uint32_t count = nvk_pack_count(args);
   const uint32_t length = nvk_pack_span_length(args, count);
   const uint32_t nodes = nvk_pack_load(args.dst, NVK_BVH_NODE_COUNT_OFFSET);
   const uint64_t node_base = args.dst + nvk_pack_load64(args.dst, 32);
   const uint64_t block_base = args.dst + nvk_pack_load64(args.dst, 16);
   const uint64_t remap = nvk_pack_remap(args);
   const uint64_t metadata = args.dst + nvk_pack_load64(args.dst, NVK_BVH_OMM_METADATA_OFFSET);
   const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
   const uint32_t threads = gridDim.x * blockDim.x;
   for (uint32_t index = thread; index < nodes; index += threads) {
      const uint64_t node = node_base + uint64_t(index) * 128;
      const uint64_t aux = args.workspace + uint64_t(index) * 32;
      uint32_t rank = nvk_pack_load(aux, 0);
      uint32_t last_leaf = 12;
      for (uint32_t slot = 0; slot < 12; slot++) {
         const uint32_t old = nvk_pack_load(node, 36 + slot * 8);
         if ((old & 0xffffu) != 0xffu && ((old >> 24) & 127u) != 0)
            last_leaf = slot;
      }
      bool first_leaf = true;
      for (uint32_t slot = 0; slot < 12; slot++) {
         const uint32_t old = nvk_pack_load(node, 36 + slot * 8);
         const bool leaf = (old & 0xffffu) != 0xffu && ((old >> 24) & 127u) != 0;
         const uint32_t packed = nvk_pack_load(aux, 4 + (slot & ~3u));
         const uint32_t primitives = leaf ? ((packed >> ((slot & 3u) * 8)) & 255u) : 0;
         if (!leaf || primitives == 0)
            continue;
         uint32_t alpha = (old >> 24) & 0x80u;
         for (uint32_t primitive = 0; primitive < primitives; primitive++) {
            const uint32_t original = nvk_pack_load(remap, (rank + primitive) * 4);
            if (nvk_pack_load64(metadata + uint64_t(original) * 8, 0) & NVK_OMM_ATTACHED)
               alpha = 0x80u;
         }
         const uint2 begin = nvk_pack_cursor(args, rank, count, length);
         uint2 end = nvk_pack_cursor(args, rank + primitives, count, length);
         if (slot == last_leaf && end.y == 0 && end.x - begin.x >= 2) {
            end.x--;
            end.y = 15;
         }
         if (first_leaf) {
            const uint64_t relative = (block_base + uint64_t(begin.x) * 128 - node) &
                                       ((1ull << 49) - 1);
            nvk_pack_store(node, 0, (nvk_pack_load(node, 0) & ~0xe0u) | 0x60u);
            nvk_pack_store(node, 16, uint32_t(relative));
            nvk_pack_store(node, 20, (nvk_pack_load(node, 20) & 0xf0fe0000u) |
                                    uint32_t(relative >> 32) | (begin.y << 24));
            first_leaf = false;
         }
         const uint32_t data = alpha | (end.x - begin.x) | (end.y << 3);
         nvk_pack_store(node, 36 + slot * 8, (old & 0x00ffffffu) | (data << 24));
         rank += primitives;
      }
   }
}
