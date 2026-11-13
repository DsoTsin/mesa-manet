#include "nvk_bvh.h"
#include "nvk_bvh_pack.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

static_assert(sizeof(nvk_bvh_refit_args) == 32, "refit parameter size");
static_assert(offsetof(nvk_bvh_refit_args, ir_leaf_size) == 24, "refit leaf stride offset");
static_assert(offsetof(nvk_bvh_refit_args, pass) == 28, "refit pass offset");

struct nvk_cuda_refit_child {
   float lo[3];
   float hi[3];
   uint32_t metadata;
   bool enabled;
   bool valid;
};

__device__ __forceinline__ uint32_t
nvk_cuda_refit_load(uint64_t address, uint32_t offset)
{
   uint32_t value;
   asm volatile("ld.global.cg.u32 %0, [%1];"
                : "=r"(value) : "l"(address + offset) : "memory");
   return value;
}

__device__ __forceinline__ uint64_t
nvk_cuda_refit_load64(uint64_t address, uint32_t offset)
{
   return uint64_t(nvk_cuda_refit_load(address, offset)) |
          (uint64_t(nvk_cuda_refit_load(address, offset + 4)) << 32);
}

__device__ __forceinline__ void
nvk_cuda_refit_store(uint64_t address, uint32_t offset, uint32_t value)
{
   asm volatile("st.global.u32 [%0], %1;"
                : : "l"(address + offset), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t
nvk_cuda_refit_atomic_add(uint64_t address, uint32_t value)
{
   uint32_t old;
   asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                : "=r"(old) : "l"(address), "r"(value) : "memory");
   return old;
}

__device__ __forceinline__ uint32_t
nvk_cuda_refit_work(uint64_t address)
{
   uint32_t work;
   asm volatile("{\n"
                ".reg .b32 lane, mask, leader, count, lower, prefix, base;\n"
                ".reg .pred elected;\n"
                "activemask.b32 mask;\n"
                "and.b32 mask, mask, 0x00010001;\n"
                "mov.u32 lane, %%laneid;\n"
                "bfind.u32 leader, mask;\n"
                "popc.b32 count, mask;\n"
                "mov.u32 lower, %%lanemask_lt;\n"
                "and.b32 lower, lower, mask;\n"
                "popc.b32 prefix, lower;\n"
                "setp.eq.u32 elected, lane, leader;\n"
                "mov.u32 base, 0;\n"
                "@elected atom.relaxed.gpu.global.add.u32 base, [%1], count;\n"
                "shfl.sync.idx.b32 base, base, leader, 0x1f, mask;\n"
                "add.u32 %0, base, prefix;\n"
                "}\n"
                : "=r"(work) : "l"(address) : "memory");
   return work;
}

__device__ __forceinline__ float
nvk_cuda_refit_flush(float value)
{
   const uint32_t bits = __float_as_uint(value);
   return (bits & 0x7f800000u) == 0 ? 0.0f : value;
}

__device__ __forceinline__ bool
nvk_cuda_refit_invalid(const nvk_cuda_refit_child &child)
{
   bool invalid = false;
#pragma unroll
   for (uint32_t axis = 0; axis < 3; axis++)
      invalid |= !isfinite(child.lo[axis]) || !isfinite(child.hi[axis]) ||
                 child.lo[axis] > child.hi[axis];
   return invalid;
}

__device__ __forceinline__ uint32_t
nvk_cuda_refit_quantize(float value, float origin, uint32_t exponent, bool upper)
{
   if (exponent < 8)
      return upper ? 255 : 0;
   const float multiplier = __uint_as_float((262u - exponent) << 23);
   float scaled;
   if (upper)
      scaled = ceilf(__fmul_ru(__fadd_ru(value, -origin), multiplier)) - 1.0f;
   else
      scaled = floorf(__fmul_rd(__fadd_rd(value, -origin), multiplier));
   return uint32_t(fminf(fmaxf(scaled, 0.0f), 255.0f));
}

__device__ __forceinline__ void
nvk_cuda_refit_write_child(uint64_t node, uint32_t slot,
                           const nvk_cuda_refit_child &child,
                           const float origin[3], const uint32_t exponent[3])
{
   uint32_t word0 = 0;
   uint32_t word1 = 0xff;
   if (child.enabled) {
      word1 = (child.metadata & 0xffffu) << 16;
      if (!child.valid) {
         word0 = 0x00ff00ffu;
         word1 |= 0x1ffu;
      } else {
         uint32_t lower[3], upper[3];
#pragma unroll
         for (uint32_t axis = 0; axis < 3; axis++) {
            lower[axis] = nvk_cuda_refit_quantize(child.lo[axis], origin[axis],
                                                  exponent[axis], false);
            upper[axis] = nvk_cuda_refit_quantize(child.hi[axis], origin[axis],
                                                  exponent[axis], true);
         }
         word0 = lower[0] | (upper[0] << 8) | (lower[1] << 16) | (upper[1] << 24);
         word1 |= lower[2] | (upper[2] << 8);
      }
   }
   nvk_cuda_refit_store(node, 32 + slot * 8, word0);
   nvk_cuda_refit_store(node, 36 + slot * 8, word1);
}

__device__ __forceinline__ uint64_t
nvk_cuda_refit_shuffle64(uint32_t mask, uint64_t value)
{
   return uint64_t(__shfl_sync(mask, uint32_t(value), 0, 16)) |
          (uint64_t(__shfl_sync(mask, uint32_t(value >> 32), 0, 16)) << 32);
}

__device__ __forceinline__ void
nvk_cuda_refit_vm(const nvk_bvh_refit_args &args, uint64_t destination,
                    uint64_t source, uint32_t original)
{
   const uint32_t header = nvk_cuda_refit_load(destination, 124);
   if ((header >> 29) != 3 || ((header >> 20) & 15u) != 0)
      return;
   const uint64_t metadata = args.dst +
      nvk_cuda_refit_load64(args.dst, NVK_BVH_OMM_METADATA_OFFSET) + uint64_t(original) * 8;
   const nvk_bvh_pack_vm vm = nvk_pack_decode_vm(nvk_cuda_refit_load64(metadata, 0));
   const uint32_t alpha = (nvk_cuda_refit_load(source, 64) & 0x80000000u) == 0;
   nvk_cuda_refit_store(destination, 108,
      (nvk_cuda_refit_load(destination, 108) & 0x1fffffffu) |
      (vm.aperture << 29) | (vm.type << 30) | ((vm.level & 1u) << 31));
   nvk_cuda_refit_store(destination, 112,
      (vm.level >> 1) | (alpha << 4) | (uint32_t(vm.address) << 5));
   nvk_cuda_refit_store(destination, 116, uint32_t(vm.address >> 27) | (1u << 25));
}

__device__ __forceinline__ void
nvk_cuda_refit_primitive(const nvk_bvh_refit_args &args, uint64_t source,
                         uint64_t destination, uint32_t original, uint32_t type,
                         uint32_t mapping, bool packed,
                         nvk_cuda_refit_child &bounds)
{
   const uint32_t primitive = nvk_cuda_refit_load(source, type == 0 ? 60 : 24);
   const uint32_t geometry = nvk_cuda_refit_load(source, type == 0 ? 64 : 28);
   const uint64_t metadata = args.dst +
      nvk_cuda_refit_load64(args.dst, NVK_BVH_PRIMITIVE_DATA_OFFSET) +
      uint64_t(original) * 8;
   nvk_cuda_refit_store(metadata, 0, primitive);
   nvk_cuda_refit_store(metadata, 4, geometry);
   bool valid = true;
   if (type == 1) {
      nvk_cuda_refit_store(destination, 0, primitive);
      nvk_cuda_refit_store(destination, 4, geometry);
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++) {
         bounds.lo[axis] = __uint_as_float(nvk_cuda_refit_load(source, axis * 4));
         bounds.hi[axis] = __uint_as_float(nvk_cuda_refit_load(source, 12 + axis * 4));
      }
      valid = bounds.lo[0] != bounds.hi[0] || bounds.lo[1] != bounds.hi[1] ||
              bounds.lo[2] != bounds.hi[2];
   } else {
      float vertices[3][3];
#pragma unroll
      for (uint32_t vertex = 0; vertex < 3; vertex++) {
         const uint32_t slot = packed ? (mapping >> (vertex * 5)) & 31u : vertex;
#pragma unroll
         for (uint32_t axis = 0; axis < 3; axis++) {
            const uint32_t bits = nvk_cuda_refit_load(source, 24 + (vertex * 3 + axis) * 4);
            const float value = __uint_as_float(bits);
            vertices[vertex][axis] = value;
            valid &= isfinite(value);
            bounds.lo[axis] = fminf(bounds.lo[axis], value);
            bounds.hi[axis] = fmaxf(bounds.hi[axis], value);
            if (!packed || slot <= 8)
               nvk_cuda_refit_store(destination, packed ? slot * 12 + axis * 4 :
                                     (vertex * 3 + axis) * 4, bits);
         }
      }
      if (!packed)
         nvk_cuda_refit_store(destination, 124,
                              (geometry & 0x80000000u) == 0 ? 1u << 26 : 0);
      bool same01 = true, same12 = true, same20 = true;
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++) {
         same01 &= vertices[0][axis] == vertices[1][axis];
         same12 &= vertices[1][axis] == vertices[2][axis];
         same20 &= vertices[2][axis] == vertices[0][axis];
      }
      valid &= !same01 && !same12 && !same20;
   }
   bounds.valid = valid && !nvk_cuda_refit_invalid(bounds);
#pragma unroll
   for (uint32_t axis = 0; axis < 3; axis++) {
      bounds.lo[axis] = bounds.valid ? nvk_cuda_refit_flush(bounds.lo[axis]) : FLT_MAX;
      bounds.hi[axis] = bounds.valid ? nvk_cuda_refit_flush(bounds.hi[axis]) : -FLT_MAX;
   }
}

__device__ __forceinline__ void
nvk_cuda_refit_leaves(const nvk_bvh_refit_args &args, uint64_t map,
                      uint64_t leaf_map, uint64_t leaf_base,
                      uint32_t first, uint32_t item_prefix, uint32_t item_count,
                      uint32_t type, uint32_t lane, uint32_t mask,
                      nvk_cuda_refit_child &bounds)
{
   const uint32_t items = __shfl_sync(mask, item_prefix + item_count, 15, 16);
   const uint32_t lower_mask = (1u << (threadIdx.x & 31u)) - 1u;
   const uint32_t item_end = item_prefix + item_count;
   uint32_t blocks = 0;
   uint32_t batch = 0;
   while (batch < items) {
      const bool process = item_count != 0 && item_prefix >= batch && item_end <= batch + 16;
      const uint32_t leaves = __ballot_sync(mask, process);
      const uint32_t last = (31u - __clz(leaves)) & 15u;
      const uint32_t end = __shfl_sync(mask, item_end, last, 16);
      const uint32_t item = batch + lane;
      uint32_t owner = 0;
#pragma unroll
      for (uint32_t delta = 8; delta != 0; delta >>= 1) {
         if (__shfl_sync(mask, item_prefix, owner + delta, 16) <= item)
            owner += delta;
      }
      const uint32_t owner_end = __shfl_sync(mask, item_end, owner, 16) - batch;
      const bool enabled = item < end;
      uint32_t mapping = 0;
      if (map != 0 && enabled) {
         const uint32_t rank = first + item;
         const uint32_t word = nvk_cuda_refit_load(map, (rank & ~1u) * 2);
         mapping = (word >> ((rank & 1u) * 16)) & 0xffffu;
      }
      const uint32_t ends = __ballot_sync(mask, enabled && (mapping & 0x8000u));
      const uint32_t block = blocks + __popc(ends & lower_mask);
      nvk_cuda_refit_child primitive_bounds;
      primitive_bounds.valid = false;
      primitive_bounds.metadata = 0;
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++) {
         primitive_bounds.lo[axis] = FLT_MAX;
         primitive_bounds.hi[axis] = -FLT_MAX;
      }
      if (enabled) {
         const uint32_t original = nvk_cuda_refit_load(leaf_map, (first + item) * 4);
         const uint64_t source = args.ir + uint64_t(original) * args.ir_leaf_size;
         const uint64_t destination = leaf_base +
            (map != 0 ? uint64_t(block) * 128 : uint64_t(item) * (type == 0 ? 128 : 8));
         if (type == 0) {
            const uint64_t metadata = args.dst +
               nvk_cuda_refit_load64(args.dst, NVK_BVH_OMM_METADATA_OFFSET) + uint64_t(original) * 8;
            if (nvk_cuda_refit_load64(metadata, 0) & NVK_OMM_ATTACHED)
               primitive_bounds.metadata = 0x8000u;
            if (map != 0)
               nvk_cuda_refit_vm(args, destination, source, original);
         }
         nvk_cuda_refit_primitive(args, source, destination, original, type,
                                   mapping, map != 0, primitive_bounds);
      }
      const uint32_t attached = __ballot_sync(mask, primitive_bounds.metadata != 0);
#pragma unroll
      for (uint32_t delta = 1; delta < 16; delta <<= 1) {
#pragma unroll
         for (uint32_t axis = 0; axis < 3; axis++) {
            const float lo = __shfl_down_sync(mask, primitive_bounds.lo[axis], delta, 16);
            const float hi = __shfl_down_sync(mask, primitive_bounds.hi[axis], delta, 16);
            if (lane + delta < owner_end) {
               primitive_bounds.lo[axis] = fminf(primitive_bounds.lo[axis], lo);
               primitive_bounds.hi[axis] = fmaxf(primitive_bounds.hi[axis], hi);
            }
         }
      }
      const uint32_t source_lane = process ? item_prefix - batch : 0;
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++) {
         const float lo = __shfl_sync(mask, primitive_bounds.lo[axis], source_lane, 16);
         const float hi = __shfl_sync(mask, primitive_bounds.hi[axis], source_lane, 16);
         if (process) {
            bounds.lo[axis] = lo;
            bounds.hi[axis] = hi;
         }
      }
      if (process) {
         bounds.valid = bounds.lo[0] <= bounds.hi[0];
         const uint32_t leaf_mask = ((1u << item_count) - 1u) <<
                                     ((threadIdx.x & 16u) + source_lane);
         if (attached & leaf_mask)
            bounds.metadata |= 0x8000u;
      }
      blocks += __popc(ends);
      batch = end;
      __syncwarp(mask);
   }
}

__device__ __forceinline__ void
nvk_cuda_refit_node(const nvk_bvh_refit_args &args, uint32_t index,
                    uint64_t auxiliary, uint64_t nodes, uint64_t leaf_map,
                    uint32_t type, uint32_t lane, uint32_t mask)
{
   const uint64_t node = nodes + uint64_t(index) * NVK_BVH_NODE_SIZE;
   const uint64_t aux = auxiliary + uint64_t(index) * 32;
   uint32_t first_item = 0;
   uint32_t internal_base = 0;
   uint64_t leaf_base = 0;
   if (lane == 0) {
      first_item = nvk_cuda_refit_load(aux, 0);
      internal_base = index + nvk_cuda_refit_load(node, 28);
      const uint64_t pointer = uint64_t(nvk_cuda_refit_load(node, 16)) |
         (uint64_t(nvk_cuda_refit_load(node, 20) &
                    (type == 1 ? 0x7fffu : 0x1ffffu)) << 32);
      const int64_t relative = type == 1 ? (int64_t(pointer << 17) >> 14) :
                                           (int64_t(pointer << 15) >> 15);
      leaf_base = uint64_t(int64_t(node) + relative);
   }
   first_item = __shfl_sync(mask, first_item, 0, 16);
   internal_base = __shfl_sync(mask, internal_base, 0, 16);
   leaf_base = nvk_cuda_refit_shuffle64(mask, leaf_base);

   nvk_cuda_refit_child bounds;
   const uint32_t old = lane < 12 ? nvk_cuda_refit_load(node, 36 + lane * 8) : 0xff;
   bounds.enabled = lane < 12 && (old & 0xffffu) != 0xffu;
   bounds.metadata = old >> 16;
   bounds.valid = false;
#pragma unroll
   for (uint32_t axis = 0; axis < 3; axis++) {
      bounds.lo[axis] = FLT_MAX;
      bounds.hi[axis] = -FLT_MAX;
   }
   const bool internal = bounds.enabled && ((old >> 24) & 0x7fu) == 0;
   const bool leaf = bounds.enabled && !internal;
   const uint32_t internal_mask = __ballot_sync(mask, internal);
   const uint32_t warp_lane = threadIdx.x & 31u;
   const uint32_t internal_prefix = __popc(internal_mask & ((1u << warp_lane) - 1u));
   uint32_t item_count = 0;
   if (leaf) {
      const uint32_t packed = nvk_cuda_refit_load(aux, 4 + (lane & ~3u));
      item_count = (packed >> ((lane & 3u) * 8)) & 255u;
   }
   uint32_t item_prefix = item_count;
#pragma unroll
   for (uint32_t delta = 1; delta < 16; delta <<= 1) {
      uint32_t previous = __shfl_up_sync(mask, item_prefix, delta, 16);
      if (lane >= delta)
         item_prefix += previous;
   }
   item_prefix -= item_count;
   const uint32_t semi_offset = type == 0 ? nvk_cuda_refit_load(args.dst, 240) : 0;
   uint64_t map = 0;
   if (semi_offset != 0) {
      leaf_map = args.dst + nvk_cuda_refit_load64(args.dst, 64);
      map = args.dst + uint64_t(semi_offset) * 128;
   }
   uint32_t rval = 0;
   if (internal) {
      const uint32_t child_index = internal_base + internal_prefix;
      const uint64_t source_node = nodes + uint64_t(child_index) * 128;
      const uint64_t source_aux = auxiliary + uint64_t(child_index) * 32;
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++) {
         bounds.lo[axis] = __uint_as_float(nvk_cuda_refit_load(source_node, 4 + axis * 4));
         bounds.hi[axis] = __uint_as_float(nvk_cuda_refit_load(source_aux, 16 + axis * 4));
      }
      rval = (nvk_cuda_refit_load(source_aux, 28) >> 8) & 255u;
      bounds.valid = !nvk_cuda_refit_invalid(bounds);
      if (!bounds.valid)
         rval = 0;
   }
   nvk_cuda_refit_leaves(args, map, leaf_map, leaf_base, first_item,
                          item_prefix, item_count, type, lane, mask, bounds);
   rval = 255u;
   bounds.metadata = (bounds.metadata & 0xff00u) | rval;
   float origin[3], maximum[3];
#pragma unroll
   for (uint32_t axis = 0; axis < 3; axis++) {
      origin[axis] = bounds.valid ? nvk_cuda_refit_flush(bounds.lo[axis]) : FLT_MAX;
      maximum[axis] = bounds.valid ? nvk_cuda_refit_flush(bounds.hi[axis]) : -FLT_MAX;
   }
   uint32_t complet_rval = rval;
#pragma unroll
   for (uint32_t delta = 1; delta < 16; delta <<= 1) {
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++) {
         origin[axis] = fminf(origin[axis], __shfl_xor_sync(mask, origin[axis], delta, 16));
         maximum[axis] = fmaxf(maximum[axis], __shfl_xor_sync(mask, maximum[axis], delta, 16));
      }
      complet_rval |= __shfl_xor_sync(mask, complet_rval, delta, 16);
   }
   uint32_t exponent[3] = {0, 0, 0};
   if (origin[0] <= maximum[0]) {
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++) {
         const float extent = __fadd_ru(maximum[axis], -origin[axis]);
         exponent[axis] = min((__float_as_uint(extent) + 0x7fffffu) >> 23, 255u);
      }
   }
   if (lane == 0) {
      nvk_cuda_refit_store(node, 0, (nvk_cuda_refit_load(node, 0) & 0xffu) |
         (exponent[0] << 8) | (exponent[1] << 16) | (exponent[2] << 24));
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++) {
         nvk_cuda_refit_store(node, 4 + axis * 4, __float_as_uint(origin[axis]));
         nvk_cuda_refit_store(aux, 16 + axis * 4, __float_as_uint(maximum[axis]));
      }
      nvk_cuda_refit_store(aux, 28, __popc(internal_mask) | (complet_rval << 8));
      if ((nvk_cuda_refit_load(node, 20) >> 28) == 15) {
#pragma unroll
         for (uint32_t axis = 0; axis < 3; axis++) {
            nvk_cuda_refit_store(args.dst, NVK_BVH_BOUNDS_OFFSET + axis * 4,
                                 __float_as_uint(origin[axis]));
            nvk_cuda_refit_store(args.dst, NVK_BVH_BOUNDS_OFFSET + 12 + axis * 4,
                                 __float_as_uint(maximum[axis]));
         }
      }
   }
   if (lane < 12)
      nvk_cuda_refit_write_child(node, lane, bounds, origin, exponent);
}

extern "C" __global__ __launch_bounds__(128) void
nvk_bvh_refit(nvk_bvh_refit_args args)
{
   const uint32_t count = nvk_cuda_refit_load(args.dst, NVK_BVH_NODE_COUNT_OFFSET);
   if (args.pass == 0) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
         nvk_cuda_refit_store(args.counter, 0, 0);
         nvk_cuda_refit_store(args.dst, 208, uint32_t(args.dst));
         nvk_cuda_refit_store(args.dst, 212, uint32_t(args.dst >> 32));
      }
      return;
   }
   const uint64_t auxiliary = args.dst + nvk_cuda_refit_load64(args.dst, NVK_BVH_REFIT_DATA_OFFSET);
   const uint64_t nodes = args.dst + nvk_cuda_refit_load64(args.dst, 32);
   const uint64_t leaf_map = args.dst + nvk_cuda_refit_load64(args.dst, NVK_BVH_REFIT_LEAF_MAP_OFFSET);
   const uint32_t type = nvk_cuda_refit_load(args.dst, NVK_BVH_GEOMETRY_TYPE_OFFSET);
   if (type > 1)
      return;
   const uint32_t lane = threadIdx.x & 15u;
   const uint32_t mask = 0xffffu << (threadIdx.x & 16u);
   const uint32_t group = blockIdx.x * (blockDim.x >> 4) + (threadIdx.x >> 4);
   const uint32_t groups = gridDim.x * (blockDim.x >> 4);
   int32_t index = int32_t(count) - 1 - int32_t(group);
   const int32_t fallback_base = int32_t(count) - 1 - int32_t(groups);
   if (index < 0)
      return;
   for (;;) {
      uint32_t claim = 1;
      if (lane == 0)
         claim = nvk_cuda_refit_atomic_add(auxiliary + uint64_t(index) * 32 + 28,
                                           0xffffffffu) & 255u;
      claim = __shfl_sync(mask, claim, 0, 16);
      __syncwarp(mask);
      if (claim == 0) {
         nvk_cuda_refit_node(args, uint32_t(index), auxiliary, nodes, leaf_map,
                             type, lane, mask);
         __syncwarp(mask);
         uint32_t parent_slot = 15;
         uint32_t parent_delta = 0;
         if (lane == 0) {
            const uint64_t node = nodes + uint64_t(index) * 128;
            parent_slot = nvk_cuda_refit_load(node, 20) >> 28;
            parent_delta = nvk_cuda_refit_load(node, 24);
            asm volatile("membar.gl;" : : : "memory");
         }
         parent_slot = __shfl_sync(mask, parent_slot, 0, 16);
         parent_delta = __shfl_sync(mask, parent_delta, 0, 16);
         if (parent_slot == 15)
            return;
         index = int32_t(uint32_t(index) + parent_delta);
      } else {
         uint32_t work = 0;
         if (lane == 0)
            work = nvk_cuda_refit_work(args.counter);
         work = __shfl_sync(mask, work, 0, 16);
         index = fallback_base - int32_t(work);
         if (index < 0)
            return;
      }
   }
}
