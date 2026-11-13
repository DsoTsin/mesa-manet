#include "nvk_bvh_build_cuda.h"
#include <math.h>
#include <stddef.h>

#ifdef __CUDACC__
#include <cuda/atomic>
#include <cub/agent/agent_radix_sort_downsweep.cuh>
#include <cub/agent/agent_radix_sort_upsweep.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <iterator>
#define NVK_BUILD_HD __host__ __device__ __forceinline__
#else
#define NVK_BUILD_HD inline
#endif

static_assert(sizeof(nvk_bvh_build_cuda_args) == 72, "args");
static_assert(sizeof(nvk_bvh_build_cuda_node_info) == 16, "node_info");
static_assert(sizeof(nvk_bvh_build_cuda_box) == 40, "box");
static_assert(sizeof(nvk_bvh_build_cuda_planner_node) == 16, "planner_node");
static_assert(sizeof(nvk_bvh_build_cuda_cluster) == 32, "cluster");
static_assert(offsetof(nvk_bvh_build_cuda_box, children) == 24, "children");
static_assert(offsetof(nvk_bvh_build_cuda_box, bvh_offset) == 32, "bvh_offset");

struct nvk_build_key {
   uint64_t key;
   uint32_t id;
};

struct nvk_native_radix_region {
   uint32_t begin;
   uint32_t end;
};

NVK_BUILD_HD nvk_native_radix_region nvk_native_radix_partition(uint32_t count,
                                                             uint32_t grid,
                                                             uint32_t block)
{
   if (count == 0 || grid == 0 || block >= grid)
      return {count, count};
   uint32_t grains = (count + 2047u) / 2048u;
   uint32_t quotient = grains / grid;
   uint32_t big_blocks = grains - quotient * grid;
   uint32_t begin = block < big_blocks ? block * (quotient + 1u) * 2048u :
                                        (big_blocks + block * quotient) * 2048u;
   uint32_t end = begin + (quotient + (block < big_blocks)) * 2048u;
   return {begin, end < count ? end : count};
}

NVK_BUILD_HD uint32_t nvk_native_load_u32_cg(const void *address)
{
#ifdef __CUDA_ARCH__
   uint32_t value;
   asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
   return value;
#else
   return *reinterpret_cast<const uint32_t *>(address);
#endif
}

NVK_BUILD_HD void nvk_native_store_u32_cg(void *address, uint32_t value)
{
#ifdef __CUDA_ARCH__
   asm volatile("st.global.cg.b32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
#else
   *reinterpret_cast<uint32_t *>(address) = value;
#endif
}

NVK_BUILD_HD uint32_t nvk_build_float_bits(float value)
{
   union { float f; uint32_t u; } bits;
   bits.f = value;
   return bits.u;
}

NVK_BUILD_HD float nvk_build_bits_float(uint32_t value)
{
   union { float f; uint32_t u; } bits;
   bits.u = value;
   return bits.f;
}

NVK_BUILD_HD nvk_bvh_build_cuda_planner_node nvk_native_load_planner_cg(const nvk_bvh_build_cuda_planner_node *address)
{
#ifdef __CUDA_ARCH__
   uint32_t words[4];
   asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3]) : "l"(address) : "memory");
   nvk_bvh_build_cuda_planner_node result;
   result.child_pair = int32_t(words[0]);
   result.before.half_area = nvk_build_bits_float(words[1]);
   result.before.parent = int32_t(words[2]);
   result.before.primitive_cost = nvk_build_bits_float(words[3]);
   return result;
#else
   return *address;
#endif
}

NVK_BUILD_HD void nvk_native_store_planner_cg(nvk_bvh_build_cuda_planner_node *address,
                                           nvk_bvh_build_cuda_planner_node value)
{
#ifdef __CUDA_ARCH__
   asm volatile("st.global.cg.v4.b32 [%0], {%1, %2, %3, %4};"
                : : "l"(address), "r"(uint32_t(value.child_pair)), "r"(nvk_build_float_bits(value.before.half_area)),
                    "r"(uint32_t(value.before.parent)), "r"(nvk_build_float_bits(value.before.primitive_cost)) : "memory");
#else
   *address = value;
#endif
}

NVK_BUILD_HD nvk_bvh_build_cuda_cluster nvk_native_load_cluster_cg(const nvk_bvh_build_cuda_cluster *address)
{
#ifdef __CUDA_ARCH__
   uint32_t words[8];
   asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(words[0]), "=r"(words[1]), "=r"(words[2]), "=r"(words[3]) : "l"(address) : "memory");
   asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4+16];"
                : "=r"(words[4]), "=r"(words[5]), "=r"(words[6]), "=r"(words[7]) : "l"(address) : "memory");
   nvk_bvh_build_cuda_cluster result;
   result.child_pair = int32_t(words[0]);
   result.subtree_prims = int32_t(words[1]);
   for (uint32_t axis = 0; axis < 3; axis++) {
      result.minimum[axis] = nvk_build_bits_float(words[2 + axis]);
      result.maximum[axis] = nvk_build_bits_float(words[5 + axis]);
   }
   return result;
#else
   return *address;
#endif
}

NVK_BUILD_HD void nvk_native_store_cluster_cg(nvk_bvh_build_cuda_cluster *address,
                                           nvk_bvh_build_cuda_cluster value)
{
#ifdef __CUDA_ARCH__
   asm volatile("st.global.cg.v4.b32 [%0], {%1, %2, %3, %4};"
                : : "l"(address), "r"(uint32_t(value.child_pair)), "r"(uint32_t(value.subtree_prims)),
                    "r"(nvk_build_float_bits(value.minimum[0])), "r"(nvk_build_float_bits(value.minimum[1])) : "memory");
   asm volatile("st.global.cg.v4.b32 [%0+16], {%1, %2, %3, %4};"
                : : "l"(address), "r"(nvk_build_float_bits(value.minimum[2])), "r"(nvk_build_float_bits(value.maximum[0])),
                    "r"(nvk_build_float_bits(value.maximum[1])), "r"(nvk_build_float_bits(value.maximum[2])) : "memory");
#else
   *address = value;
#endif
}

NVK_BUILD_HD float nvk_build_emulated_float(uint32_t value)
{
   return nvk_build_bits_float((value & 0x80000000u) ? 0x80000000u - value : value);
}

NVK_BUILD_HD uint32_t nvk_build_clz32(uint32_t value)
{
#ifdef __CUDA_ARCH__
   return __clz(value);
#else
   if (!value)
      return 32;
   uint32_t count = 0;
   for (uint32_t bit = 0x80000000u; !(value & bit); bit >>= 1)
      count++;
   return count;
#endif
}

NVK_BUILD_HD uint32_t nvk_build_clz64(uint64_t value)
{
   uint32_t high = uint32_t(value >> 32);
   return high ? nvk_build_clz32(high) : 32 + nvk_build_clz32(uint32_t(value));
}

NVK_BUILD_HD uint64_t nvk_build_expand64(uint64_t value)
{
   value = (value | value << 32) & 0x1f00000000ffffull;
   value = (value | value << 16) & 0x1f0000ff0000ffull;
   value = (value | value << 8) & 0x100f00f00f00f00full;
   value = (value | value << 4) & 0x10c30c30c30c30c3ull;
   return (value | value << 2) & 0x1249249249249249ull;
}

NVK_BUILD_HD uint32_t nvk_build_expand32(uint32_t value)
{
   value = (value | value << 16) & 0x030000ffu;
   value = (value | value << 8) & 0x0300f00fu;
   value = (value | value << 4) & 0x030c30c3u;
   return (value | value << 2) & 0x09249249u;
}

NVK_BUILD_HD nvk_build_key nvk_build_load_key(uint64_t address, uint32_t index, bool wide)
{
   const uint32_t *words = reinterpret_cast<const uint32_t *>(address) + uint64_t(index) * (wide ? 3 : 2);
   nvk_build_key result;
   result.id = words[0];
   result.key = words[1];
   if (wide)
      result.key |= uint64_t(words[2]) << 32;
   return result;
}

NVK_BUILD_HD void nvk_build_store_key(uint64_t address, uint32_t index, bool wide, nvk_build_key value)
{
   uint32_t *words = reinterpret_cast<uint32_t *>(address) + uint64_t(index) * (wide ? 3 : 2);
   words[0] = value.id;
   words[1] = uint32_t(value.key);
   if (wide)
      words[2] = uint32_t(value.key >> 32);
}

NVK_BUILD_HD bool nvk_build_less(nvk_build_key left, nvk_build_key right)
{
   return left.key < right.key || (left.key == right.key && left.id < right.id);
}

NVK_BUILD_HD void nvk_build_sort_pair(nvk_build_key *keys, uint32_t local, uint32_t length, uint32_t distance)
{
   uint32_t other = local ^ distance;
   if (other > local && ((local & length) ? nvk_build_less(keys[local], keys[other]) : nvk_build_less(keys[other], keys[local]))) {
      nvk_build_key temporary = keys[local];
      keys[local] = keys[other];
      keys[other] = temporary;
   }
}

NVK_BUILD_HD uint32_t nvk_build_merge_destination(const nvk_bvh_build_cuda_args &args, uint32_t index)
{
   bool wide = (args.build_flags & NVK_BVH_BUILD_CUDA_KEYS_64BIT) != 0;
   uint64_t pair_base = uint64_t(index) & ~((uint64_t(args.merge_width) * 2) - 1);
   uint64_t middle = pair_base + args.merge_width;
   uint64_t end = middle + args.merge_width;
   if (middle > args.leaf_count)
      middle = args.leaf_count;
   if (end > args.leaf_count)
      end = args.leaf_count;
   bool left = uint64_t(index) < middle;
   uint64_t other_begin = left ? middle : pair_base;
   uint64_t other_end = left ? end : middle;
   nvk_build_key key = nvk_build_load_key(args.src_keys, index, wide);
   uint64_t low = other_begin, high = other_end;
   while (low < high) {
      uint64_t at = low + (high - low) / 2;
      nvk_build_key other = nvk_build_load_key(args.src_keys, uint32_t(at), wide);
      bool before = left ? nvk_build_less(other, key) : !nvk_build_less(key, other);
      if (before)
         low = at + 1;
      else
         high = at;
   }
   return uint32_t(pair_base + (uint64_t(index) - (left ? pair_base : middle)) + low - other_begin);
}

NVK_BUILD_HD nvk_build_key nvk_build_make_key(const nvk_bvh_build_cuda_args &args, uint32_t id)
{
   nvk_build_key result = {~uint64_t(0), id};
   if (id == 0xffffffffu)
      return result;
   const float *bounds = reinterpret_cast<const float *>(args.ir + (id & ~3u));
   const uint32_t *header = reinterpret_cast<const uint32_t *>(args.ir_header);
   bool wide = (args.build_flags & NVK_BVH_BUILD_CUDA_KEYS_64BIT) != 0;
   uint32_t q[3];
   for (uint32_t axis = 0; axis < 3; axis++) {
      float minimum = nvk_build_emulated_float(header[axis]);
      float maximum = nvk_build_emulated_float(header[3 + axis]);
      float center = (bounds[axis] + bounds[3 + axis]) * 0.5f;
      float extent = maximum - minimum;
      float normalized = (center - minimum) / extent;
      if (!(normalized > 0.0f))
         normalized = 0.0f;
      if (normalized > 1.0f)
         normalized = 1.0f;
      q[axis] = uint32_t(normalized * (wide ? 2097151.0f : 255.0f));
   }
   if (wide) {
      result.key = (nvk_build_expand64(q[0]) << 2) |
                   (nvk_build_expand64(q[1]) << 1) | nvk_build_expand64(q[2]);
   } else {
      uint32_t key = ((nvk_build_expand32(q[0]) << 2) |
                      (nvk_build_expand32(q[1]) << 1) | nvk_build_expand32(q[2])) << 8;
      result.key = key < 0xfffffffeu ? key : 0xfffffffeu;
   }
   return result;
}

NVK_BUILD_HD int nvk_build_lcp(const nvk_bvh_build_cuda_args &args, int index, uint64_t key, int64_t other, uint32_t count)
{
   if (other < 0 || uint64_t(other) >= count)
      return -1;
   bool wide = (args.build_flags & NVK_BVH_BUILD_CUDA_KEYS_64BIT) != 0;
   uint64_t different = key ^ nvk_build_load_key(args.src_keys, uint32_t(other), wide).key;
   if (!different)
      return int(wide ? 64u + nvk_build_clz64(uint32_t(index) ^ uint32_t(other)) :
                        32u + nvk_build_clz32(uint32_t(index) ^ uint32_t(other)));
   return int(wide ? nvk_build_clz64(different) : nvk_build_clz32(uint32_t(different)));
}

NVK_BUILD_HD void nvk_build_topology_node(const nvk_bvh_build_cuda_args &args, uint32_t index)
{
   auto *nodes = reinterpret_cast<nvk_bvh_build_cuda_node_info *>(args.node_info);
   uint32_t count = reinterpret_cast<const uint32_t *>(args.ir_header)[6];
   bool wide = (args.build_flags & NVK_BVH_BUILD_CUDA_KEYS_64BIT) != 0;
   if (count <= 1) {
      if (index == 0) {
         nodes[0].parent = 0xffffffffu;
         nodes[0].path_count = 2;
         nodes[0].children[0] = count ? nvk_build_load_key(args.src_keys, 0, wide).id : 0xffffffffu;
         nodes[0].children[1] = 0xffffffffu;
      }
      return;
   }
   if (index >= count - 1)
      return;
   int i = int(index);
   uint64_t key = nvk_build_load_key(args.src_keys, index, wide).key;
   int left_lcp = nvk_build_lcp(args, i, key, int64_t(i) - 1, count);
   int right_lcp = nvk_build_lcp(args, i, key, int64_t(i) + 1, count);
   int direction = right_lcp > left_lcp ? 1 : -1;
   int minimum_lcp = left_lcp < right_lcp ? left_lcp : right_lcp;
   int64_t maximum_length = 128;
   while (nvk_build_lcp(args, i, key, int64_t(i) + direction * maximum_length, count) > minimum_lcp)
      maximum_length *= 2;
   int64_t length = 0;
   for (int64_t step = maximum_length / 2; step; step /= 2) {
      if (nvk_build_lcp(args, i, key, int64_t(i) + (length + step) * direction, count) > minimum_lcp)
         length += step;
   }
   int other = int(i + length * direction);
   int node_lcp = nvk_build_lcp(args, i, key, other, count);
   int64_t child_range = 0;
   for (int64_t divisor = 2; divisor < 2 * length; divisor *= 2) {
      int64_t step = (length + divisor - 1) / divisor;
      if (nvk_build_lcp(args, i, key, int64_t(i) + (child_range + step) * direction, count) > node_lcp)
         child_range += step;
   }
   int left = int(i + child_range * direction) + (direction < 0 ? -1 : 0);
   int right = left + 1;
   bool left_leaf = (i < other ? i : other) == left;
   bool right_leaf = (i > other ? i : other) == right;
   if (!left_leaf)
      nodes[left].parent = index;
   if (!right_leaf)
      nodes[right].parent = index | (1u << 29);
   nodes[index].path_count = uint32_t(left_leaf) + uint32_t(right_leaf);
   nodes[index].children[0] = nvk_build_load_key(args.src_keys, uint32_t(left), wide).id;
   nodes[index].children[1] = nvk_build_load_key(args.src_keys, uint32_t(right), wide).id;
   if (!index)
      nodes[0].parent = 0xffffffffu;
}

NVK_BUILD_HD uint32_t nvk_build_child_flags(uint64_t ir, uint32_t reference)
{
   const uint32_t *words = reinterpret_cast<const uint32_t *>(ir + (reference & ~3u));
   switch (reference & 3u) {
   case 0: return words[16] & 0x80000000u ? 1u : 2u;
   case 1: return words[9];
   case 2: return words[23];
   default: return words[7] & 0x80000000u ? 1u : 2u;
   }
}

NVK_BUILD_HD nvk_bvh_build_cuda_box nvk_build_box(const nvk_bvh_build_cuda_args &args, uint32_t left, uint32_t right)
{
   nvk_bvh_build_cuda_box result;
   result.children[0] = left;
   result.children[1] = right;
   result.bvh_offset = 0xffffffffu;
   result.flags = 0xffffffffu;
   for (uint32_t axis = 0; axis < 3; axis++) {
      result.minimum[axis] = nvk_build_bits_float(0x7f800000u);
      result.maximum[axis] = nvk_build_bits_float(0xff800000u);
   }
   for (uint32_t child = 0; child < 2; child++) {
      uint32_t reference = result.children[child];
      if (reference == 0xffffffffu)
         continue;
      const float *bounds = reinterpret_cast<const float *>(args.ir + (reference & ~3u));
      for (uint32_t axis = 0; axis < 3; axis++) {
         result.minimum[axis] = fminf(result.minimum[axis], bounds[axis]);
         result.maximum[axis] = fmaxf(result.maximum[axis], bounds[3 + axis]);
      }
      if (args.build_flags & NVK_BVH_BUILD_CUDA_PROPAGATE_FLAGS)
         result.flags &= nvk_build_child_flags(args.ir, reference);
   }
   return result;
}

NVK_BUILD_HD float nvk_native_flush(float value)
{
   uint32_t bits = nvk_build_float_bits(value);
   return (bits & 0x7fffffffu) < 0x00800000u ? nvk_build_bits_float(bits & 0x80000000u) : value;
}

NVK_BUILD_HD float nvk_native_add(float left, float right)
{
#ifdef __CUDA_ARCH__
   float result;
   asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(left), "f"(right));
   return result;
#else
   return nvk_native_flush(nvk_native_flush(left) + nvk_native_flush(right));
#endif
}

NVK_BUILD_HD float nvk_native_mul(float left, float right)
{
#ifdef __CUDA_ARCH__
   float result;
   asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(left), "f"(right));
   return result;
#else
   return nvk_native_flush(nvk_native_flush(left) * nvk_native_flush(right));
#endif
}

NVK_BUILD_HD float nvk_native_fma(float left, float right, float addend)
{
#ifdef __CUDA_ARCH__
   float result;
   asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(left), "f"(right), "f"(addend));
   return result;
#else
   return nvk_native_flush(fmaf(nvk_native_flush(left), nvk_native_flush(right), nvk_native_flush(addend)));
#endif
}

NVK_BUILD_HD float nvk_native_rcp(float value)
{
#ifdef __CUDA_ARCH__
   float result;
   asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
   return result;
#else
   return nvk_native_flush(1.0f / nvk_native_flush(value));
#endif
}

NVK_BUILD_HD float nvk_native_saturate(float value)
{
#ifdef __CUDA_ARCH__
   return __saturatef(value);
#else
   return !(value > 0.0f) ? 0.0f : value > 1.0f ? 1.0f : value;
#endif
}

NVK_BUILD_HD uint64_t nvk_native_morton_bits(uint32_t x, uint32_t y, uint32_t z, bool wide)
{
   uint32_t q[3] = {(x >> 3) & 0xfffffu, (y >> 3) & 0xfffffu, (z >> 3) & 0xfffffu};
   uint32_t high = (nvk_build_expand32(q[0] >> 10) << 2) |
                   (nvk_build_expand32(q[1] >> 10) << 1) | nvk_build_expand32(q[2] >> 10);
   if (!wide)
      return high;
   uint32_t low = (nvk_build_expand32(q[0] & 1023u) << 2) |
                  (nvk_build_expand32(q[1] & 1023u) << 1) | nvk_build_expand32(q[2] & 1023u);
   return (uint64_t(high) << 32) | low;
}

NVK_BUILD_HD nvk_build_key nvk_native_make_key(const nvk_bvh_build_cuda_args &args, uint32_t id)
{
   nvk_build_key result = {~uint64_t(0), id};
   if (id == 0xffffffffu)
      return result;
   const float *bounds = reinterpret_cast<const float *>(args.ir + (id & ~3u));
   const uint32_t *header = reinterpret_cast<const uint32_t *>(args.ir_header);
   float minimum[3], maximum[3], extent = 0.0f;
   for (uint32_t axis = 0; axis < 3; axis++) {
      minimum[axis] = nvk_build_emulated_float(header[axis]);
      maximum[axis] = nvk_build_emulated_float(header[3 + axis]);
      extent = fmaxf(extent, nvk_native_add(maximum[axis], -minimum[axis]));
   }
   float reciprocal = nvk_native_rcp(extent);
   uint32_t quantized[3];
   for (uint32_t axis = 0; axis < 3; axis++) {
      float model_center = nvk_native_mul(nvk_native_add(minimum[axis], maximum[axis]), 0.5f);
      float relative = nvk_native_fma(nvk_native_add(bounds[axis], bounds[3 + axis]), 0.5f, -model_center);
      float normalized = nvk_native_saturate(nvk_native_fma(reciprocal, relative, 0.5f));
      float encoded = fminf(nvk_native_add(normalized, 1.0f), nvk_build_bits_float(0x3fffffffu));
      quantized[axis] = nvk_build_float_bits(encoded);
   }
   result.key = nvk_native_morton_bits(quantized[0], quantized[1], quantized[2],
                                      (args.build_flags & NVK_BVH_BUILD_CUDA_KEYS_64BIT) != 0);
   return result;
}

NVK_BUILD_HD uint32_t nvk_native_float_to_uint(float value)
{
#ifdef __CUDA_ARCH__
   return __float2uint_rz(value);
#else
   if (!(value > 0.0f))
      return 0;
   if (value >= 4294967295.0f)
      return 0xffffffffu;
   return uint32_t(value);
#endif
}

NVK_BUILD_HD void nvk_native_aac_precompute(const nvk_bvh_build_cuda_cluster &cluster, uint32_t bits[3])
{
   for (uint32_t axis = 0; axis < 3; axis++)
      bits[axis] = nvk_native_float_to_uint(nvk_native_mul(nvk_native_add(cluster.minimum[axis], cluster.maximum[axis]), 2147483648.0f));
}

NVK_BUILD_HD uint32_t nvk_native_aac_phase(uint32_t value, int shift)
{
   uint32_t phase;
#ifdef __CUDA_ARCH__
   asm("shf.r.clamp.b32 %0, 0, %1, %2;" : "=r"(phase) : "r"(value), "r"(uint32_t(shift)));
#else
   uint32_t amount = uint32_t(shift) < 32u ? uint32_t(shift) : 32u;
   phase = amount == 0 ? 0 : amount == 32 ? value : value << (32 - amount);
#endif
   return phase & 0x80000000u ? 0u - phase : phase;
}

NVK_BUILD_HD float nvk_native_aac_half_area(const nvk_bvh_build_cuda_cluster &cluster)
{
   float extent[3];
   for (uint32_t axis = 0; axis < 3; axis++)
      extent[axis] = nvk_native_add(cluster.maximum[axis], -cluster.minimum[axis]);
   float area = nvk_native_mul(extent[0], extent[2]);
   area = nvk_native_fma(extent[1], extent[2], area);
   return nvk_native_fma(extent[0], extent[1], area);
}

NVK_BUILD_HD float nvk_native_aac_merge_cost(const nvk_bvh_build_cuda_cluster &left, const nvk_bvh_build_cuda_cluster &right,
                                           const uint32_t left_bits[3], const uint32_t right_bits[3], float coefficient)
{
   nvk_bvh_build_cuda_cluster combined;
   for (uint32_t axis = 0; axis < 3; axis++) {
      combined.minimum[axis] = fminf(left.minimum[axis], right.minimum[axis]);
      combined.maximum[axis] = fmaxf(left.maximum[axis], right.maximum[axis]);
   }
   float penalty = 0.0f;
   for (uint32_t axis = 0; axis < 3; axis++) {
      int shift = 31 - int(nvk_build_clz32(left_bits[axis] ^ right_bits[axis]));
      uint32_t phase_left = nvk_native_aac_phase(left_bits[axis], shift);
      uint32_t phase_right = nvk_native_aac_phase(right_bits[axis], shift);
      uint32_t distance = phase_left < phase_right ? phase_left : phase_right;
      uint32_t exponent = 0x10000000u + (uint32_t(shift) << 24);
      penalty = nvk_native_fma(float(distance), nvk_build_bits_float(exponent), penalty);
   }
   return nvk_native_fma(penalty, coefficient, nvk_native_aac_half_area(combined));
}

NVK_BUILD_HD uint32_t nvk_native_merge_destination(const nvk_bvh_build_cuda_args &args, uint32_t index)
{
   bool wide = (args.build_flags & NVK_BVH_BUILD_CUDA_KEYS_64BIT) != 0;
   uint32_t pair_width = args.merge_width * 2;
   uint32_t pair_base = index / pair_width * pair_width;
   uint32_t middle = pair_base + args.merge_width;
   uint32_t end = middle + args.merge_width;
   if (middle > args.leaf_count)
      middle = args.leaf_count;
   if (end > args.leaf_count)
      end = args.leaf_count;
   bool left = index < middle;
   uint32_t begin_other = left ? middle : pair_base;
   uint32_t end_other = left ? end : middle;
   uint32_t low = begin_other, high = end_other;
   uint64_t key = nvk_build_load_key(args.src_keys, index, wide).key;
   while (low < high) {
      uint32_t at = low + (high - low) / 2;
      uint64_t other = nvk_build_load_key(args.src_keys, at, wide).key;
      if (other < key || (left && other == key))
         low = at + 1;
      else
         high = at;
   }
   return pair_base + index - (left ? pair_base : middle) + low - begin_other;
}

#ifdef __CUDACC__
__device__ __forceinline__ nvk_bvh_build_cuda_cluster nvk_native_shuffle_cluster(unsigned mask, nvk_bvh_build_cuda_cluster value, int source)
{
   nvk_bvh_build_cuda_cluster result;
   result.child_pair = __shfl_sync(mask, value.child_pair, source);
   result.subtree_prims = __shfl_sync(mask, value.subtree_prims, source);
   for (uint32_t axis = 0; axis < 3; axis++) {
      result.minimum[axis] = __shfl_sync(mask, value.minimum[axis], source);
      result.maximum[axis] = __shfl_sync(mask, value.maximum[axis], source);
   }
   return result;
}

__device__ __forceinline__ unsigned nvk_native_lane_lt(unsigned lane)
{
   return lane ? (1u << lane) - 1u : 0u;
}

__device__ __forceinline__ unsigned nvk_native_lane_gt(unsigned lane)
{
   return lane == 31 ? 0u : ~((1u << (lane + 1u)) - 1u);
}
#endif

NVK_BUILD_HD nvk_bvh_build_cuda_cluster nvk_native_leaf_cluster(const nvk_bvh_build_cuda_args &args, uint32_t sorted_index)
{
   uint32_t reference = nvk_build_load_key(args.src_keys, sorted_index, true).id;
   const float *bounds = reinterpret_cast<const float *>(args.ir + (reference & ~3u));
   const uint32_t *header = reinterpret_cast<const uint32_t *>(args.ir_header);
   nvk_bvh_build_cuda_cluster result;
   result.child_pair = ~int32_t((reference & ~3u) / args.ir_leaf_size);
   result.subtree_prims = 1;
   float minimum[3], maximum[3], extent = 0.0f;
   for (uint32_t axis = 0; axis < 3; axis++) {
      minimum[axis] = nvk_build_emulated_float(header[axis]);
      maximum[axis] = nvk_build_emulated_float(header[3 + axis]);
      extent = fmaxf(extent, nvk_native_add(maximum[axis], -minimum[axis]));
   }
   float reciprocal = nvk_native_rcp(extent);
   for (uint32_t axis = 0; axis < 3; axis++) {
      float center = nvk_native_mul(nvk_native_add(minimum[axis], maximum[axis]), 0.5f);
      result.minimum[axis] = nvk_native_fma(nvk_native_add(bounds[axis], -center), reciprocal, 0.5f);
      result.maximum[axis] = nvk_native_fma(nvk_native_add(bounds[3 + axis], -center), reciprocal, 0.5f);
   }
   return result;
}

#ifdef __CUDACC__
template<int MaxClusters> __device__ __forceinline__ void
nvk_native_merge_cluster_group(const nvk_bvh_build_cuda_args &args, int first, int split, int last, int total)
{
   static_assert(MaxClusters == 4 || MaxClusters == 8 || MaxClusters == 16, "MaxClusters");
   constexpr unsigned GroupSize = MaxClusters * 2u;
   unsigned lane = threadIdx.x & 31u;
   unsigned group_base = lane & ~(GroupSize - 1u);
   unsigned relative = lane & (GroupSize - 1u);
   int left_count = split - first < MaxClusters ? split - first : MaxClusters;
   int right_count = last - split < MaxClusters ? last - split : MaxClusters;
   int size = left_count + right_count;
   if (relative >= unsigned(size))
      return;
   unsigned mask = size == 32 ? 0xffffffffu : ((1u << unsigned(size)) - 1u) << group_base;
   auto *indices = reinterpret_cast<int32_t *>(args.node_info);
   auto *clusters = reinterpret_cast<nvk_bvh_build_cuda_cluster *>(args.dst_keys);
   auto *nodes = reinterpret_cast<nvk_bvh_build_cuda_planner_node *>(args.dst_keys);
   int input = int(relative) + (int(relative) < left_count ? first : split - left_count);
   int cluster_index = int32_t(nvk_native_load_u32_cg(indices + input));
   nvk_bvh_build_cuda_cluster cluster;
   if (cluster_index < 0) {
      cluster_index = input;
      cluster = nvk_native_leaf_cluster(args, unsigned(input));
   } else {
      cluster = nvk_native_load_cluster_cg(clusters + cluster_index);
   }
   int limit = MaxClusters;
   if (first == 0 && last == total) {
      limit = 1;
      if (relative < 2) {
         cluster_index = relative == 0 ? -1 : 0;
         nvk_native_store_u32_cg(&nodes[relative + 2].child_pair, 0x80000000u);
      }
   }
   int merges_left = size - limit;
   do {
      unsigned active = __ballot_sync(mask, merges_left > 0);
      int rounds = __popc(active) >> 1;
      unsigned bits[3];
      nvk_native_aac_precompute(cluster, bits);
      uint32_t best_cost = 0xffffffffu;
      int pair = -1;
      unsigned next = lane, previous = lane;
      for (int round = 0; round < rounds; round++) {
         unsigned after = active & nvk_native_lane_gt(next);
         unsigned before = active & nvk_native_lane_lt(previous);
         next = unsigned(__ffs(int(after ? after : active)) - 1);
         previous = 31u - __clz(before ? before : active);
         auto other = nvk_native_shuffle_cluster(mask, cluster, int(next));
         unsigned other_bits[3];
         for (uint32_t axis = 0; axis < 3; axis++)
            other_bits[axis] = __shfl_sync(mask, bits[axis], int(next));
         float cost = fmaxf(nvk_native_aac_merge_cost(cluster, other, bits, other_bits, 1.0f), 0.0f);
         uint32_t next_cost = nvk_build_float_bits(cost);
         uint32_t previous_cost = __shfl_sync(mask, next_cost, int(previous));
         uint32_t minimum = best_cost < next_cost ? best_cost : next_cost;
         minimum = minimum < previous_cost ? minimum : previous_cost;
         if (minimum != best_cost)
            pair = -1;
         if (next_cost == minimum && int(next) > pair)
            pair = int(next);
         if (previous_cost == minimum && int(previous) > pair)
            pair = int(previous);
         best_cost = minimum;
      }
      int partner_pair = __shfl_sync(mask, pair, pair >= 0 ? pair : int(group_base));
      bool merge = pair >= 0 && partner_pair == int(lane);
      unsigned high_pairs = __ballot_sync(mask, merge && int(lane) > pair);
      if (__popc(high_pairs & nvk_native_lane_lt(lane)) >= merges_left)
         pair = -1;
      partner_pair = __shfl_sync(mask, pair, pair >= 0 ? pair : int(group_base));
      merge = pair >= 0 && partner_pair == int(lane);
      high_pairs &= __ballot_sync(mask, merge);
      merges_left -= __popc(high_pairs);
      uint32_t area = nvk_build_float_bits(fmaxf(nvk_native_aac_half_area(cluster), 0.0f));
      uint32_t partner_area = __shfl_sync(mask, area, pair >= 0 ? pair : int(group_base));
      bool first_child = area != partner_area ? area > partner_area : int(lane) < pair;
      int allocation_slot = __popc(active) - __popc(high_pairs & ~nvk_native_lane_lt(lane));
      int high_lane = int(lane) > pair ? int(lane) : pair;
      int shared_slot = __shfl_sync(mask, allocation_slot, high_lane);
      int allocation = __shfl_sync(mask, cluster_index, int(group_base) + shared_slot);
      int output = allocation * 2 + (first_child ? 0 : 1);
      auto partner = nvk_native_shuffle_cluster(mask, cluster, pair >= 0 ? pair : int(group_base));
      if (merge) {
         nvk_bvh_build_cuda_planner_node node;
         node.child_pair = cluster.child_pair;
         node.before.half_area = nvk_native_aac_half_area(cluster);
         node.before.parent = -1;
         node.before.primitive_cost = 0.0f;
         nvk_native_store_planner_cg(nodes + output, node);
         if (node.child_pair >= 0) {
            nvk_native_store_u32_cg(&nodes[node.child_pair].before.parent, uint32_t(output));
            nvk_native_store_u32_cg(&nodes[node.child_pair + 1].before.parent, uint32_t(output));
         }
         cluster.child_pair = output;
         cluster.subtree_prims += partner.subtree_prims;
         for (uint32_t axis = 0; axis < 3; axis++) {
            cluster.minimum[axis] = fminf(cluster.minimum[axis], partner.minimum[axis]);
            cluster.maximum[axis] = fmaxf(cluster.maximum[axis], partner.maximum[axis]);
         }
         if (!first_child)
            merges_left = -1;
      }
   } while (__any_sync(mask, merges_left > 0));
   unsigned survivors = __ballot_sync(mask, merges_left == 0);
   int slot = __popc(survivors & nvk_native_lane_lt(lane));
   int destination = __shfl_sync(mask, cluster_index, int(group_base) + slot);
   if (merges_left == 0 && destination >= 0) {
      nvk_native_store_u32_cg(indices + first + slot, uint32_t(destination));
      nvk_native_store_cluster_cg(clusters + destination, cluster);
   }
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_aac_init_native(nvk_bvh_build_cuda_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t capacity = args.leaf_count > 2 ? args.leaf_count : 2;
   if (!index)
      *reinterpret_cast<uint32_t *>(args.input_ids) = 0;
   if (index < capacity) {
      auto *workspace = reinterpret_cast<int32_t *>(args.node_info);
      workspace[index] = -1;
      workspace[capacity + index] = -1;
   }
}

template<int MaxClusters> __device__ __forceinline__ void
nvk_native_aac_execute(nvk_bvh_build_cuda_args args)
{
   constexpr unsigned GroupSize = MaxClusters * 2u;
   int total = int(reinterpret_cast<const uint32_t *>(args.ir_header)[6]);
   auto *nodes = reinterpret_cast<nvk_bvh_build_cuda_planner_node *>(args.dst_keys);
   if (total < 2) {
      if (!blockIdx.x && threadIdx.x < 2) {
         nvk_bvh_build_cuda_planner_node node = {};
         node.child_pair = int32_t(0x80000000u);
         node.before.parent = -1;
         if (int(threadIdx.x) < total) {
            uint32_t id = nvk_build_load_key(args.src_keys, threadIdx.x, true).id;
            node.child_pair = ~int32_t((id & ~3u) / args.ir_leaf_size);
         }
         nvk_native_store_planner_cg(nodes + threadIdx.x, node);
      }
      return;
   }
   unsigned lane = threadIdx.x & 31u;
   unsigned full = 0xffffffffu;
   uint32_t capacity = args.leaf_count > 2 ? args.leaf_count : 2;
   auto *ranges = reinterpret_cast<int32_t *>(args.node_info) + capacity;
   int first = -1, last = -1, split = -1;
   [[maybe_unused]] int timer = 0;
   do {
      unsigned traversing = __ballot_sync(full, last != -1);
      unsigned work_mask = __ballot_sync(full, __popc(traversing) < 4 && first == -1);
      if (work_mask & (1u << lane)) {
         unsigned leader = 31u - __clz(work_mask);
         uint32_t base = 0;
         if (lane == leader) {
            cuda::atomic_ref<uint32_t, cuda::thread_scope_device> counter(*reinterpret_cast<uint32_t *>(args.input_ids));
            base = counter.fetch_add(__popc(work_mask), cuda::memory_order_relaxed);
         }
         first = int(__shfl_sync(work_mask, base, int(leader)) + __popc(work_mask & nvk_native_lane_gt(lane)));
         last = first < total ? first + 1 : -1;
      }
      if (last != -1 && split == -1) {
         uint64_t low = first > 0 ? nvk_build_load_key(args.src_keys, unsigned(first), true).key ^ nvk_build_load_key(args.src_keys, unsigned(first - 1), true).key : ~uint64_t(1);
         uint64_t high = last < total ? nvk_build_load_key(args.src_keys, unsigned(last), true).key ^ nvk_build_load_key(args.src_keys, unsigned(last - 1), true).key : ~uint64_t(0);
         if (low == high) {
            low = uint32_t(first) ^ uint32_t(first - 1);
            high = uint32_t(last) ^ uint32_t(last - 1);
         }
         bool swap = low < high;
         int boundary = swap ? first : last;
         int value = swap ? last : first;
         cuda::atomic_ref<int32_t, cuda::thread_scope_device> arrival(ranges[boundary]);
         int other = arrival.exchange(value, cuda::memory_order_acq_rel);
         split = boundary;
         if (other == -1) {
            first = last = -1;
         } else {
            first = swap ? other : value;
            last = swap ? value : other;
         }
         int limit = first == 0 && last == total ? 1 : MaxClusters;
         if (last - first <= limit)
            split = -1;
      }
      unsigned pending = __ballot_sync(full, split != -1);
      if constexpr (GroupSize == 32) {
         while (pending) {
            int source = 31 - __clz(pending);
            pending ^= 1u << unsigned(source);
            int a = __shfl_sync(full, first, source);
            int b = __shfl_sync(full, split, source);
            int c = __shfl_sync(full, last, source);
            nvk_native_merge_cluster_group<MaxClusters>(args, a, b, c, total);
            cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
            __syncwarp(full);
         }
         split = -1;
      } else {
         constexpr int GroupsPerWarp = 32 / GroupSize;
         unsigned group = lane / GroupSize;
         bool perform = __popc(pending) >= GroupsPerWarp || (pending && timer > 0);
         timer++;
         if (perform) {
            timer = 0;
            unsigned selected = pending;
            for (unsigned at = 0; at < group; at++)
               selected &= selected - 1;
            int source = selected ? __ffs(int(selected)) - 1 : 0;
            int a = __shfl_sync(full, first, source);
            int b = __shfl_sync(full, split, source);
            int c = __shfl_sync(full, last, source);
            if (split != -1 && __popc(pending & nvk_native_lane_lt(lane)) < GroupsPerWarp)
               split = -1;
            if (selected)
               nvk_native_merge_cluster_group<MaxClusters>(args, a, b, c, total);
            cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
            __syncwarp(full);
         }
      }
   } while (__any_sync(full, first < total));
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_aac_native(nvk_bvh_build_cuda_args args)
{
   nvk_native_aac_execute<8>(args);
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_aac_max16_native(nvk_bvh_build_cuda_args args)
{
   nvk_native_aac_execute<16>(args);
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_aac_max4_native(nvk_bvh_build_cuda_args args)
{
   nvk_native_aac_execute<4>(args);
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_aac_ir_init_native(nvk_bvh_build_cuda_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t capacity = args.leaf_count > 2 ? args.leaf_count : 2;
   uint32_t count = reinterpret_cast<const uint32_t *>(args.ir_header)[6];
   if (!index) {
      auto *header = reinterpret_cast<uint32_t *>(args.ir_header);
      header[8] = 0;
      header[9] = 1;
      header[10] = 1;
   }
   if (index >= capacity)
      return;
   auto *workspace = reinterpret_cast<uint32_t *>(args.node_info);
   workspace[index] = 0xffffffffu;
   workspace[capacity + index] = 0;
   if (index != 0 && (index == 1 || index >= count))
      return;
   auto *nodes = reinterpret_cast<const nvk_bvh_build_cuda_planner_node *>(args.dst_keys);
   workspace[capacity + index] = uint32_t(int32_t(nvk_native_load_u32_cg(&nodes[index * 2].child_pair)) < 0) +
                                  uint32_t(int32_t(nvk_native_load_u32_cg(&nodes[index * 2 + 1].child_pair)) < 0);
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_aac_ir_native(nvk_bvh_build_cuda_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   uint32_t capacity = args.leaf_count > 2 ? args.leaf_count : 2;
   uint32_t count = reinterpret_cast<const uint32_t *>(args.ir_header)[6];
   if (index != 0 && (index == 1 || index >= count))
      return;
   auto *nodes = reinterpret_cast<const nvk_bvh_build_cuda_planner_node *>(args.dst_keys);
   auto *workspace = reinterpret_cast<uint32_t *>(args.node_info);
   auto *header = reinterpret_cast<uint32_t *>(args.ir_header);
   uint32_t type = count ? nvk_build_load_key(args.src_keys, 0, true).id & 3u : 0;
   for (;;) {
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(workspace[capacity + index]);
      if (ready.fetch_add(1, cuda::memory_order_acq_rel) != 2)
         return;
      uint32_t references[2];
      for (uint32_t child = 0; child < 2; child++) {
         int32_t reference = int32_t(nvk_native_load_u32_cg(&nodes[index * 2 + child].child_pair));
         references[child] = reference == int32_t(0x80000000u) ? 0xffffffffu :
                             reference < 0 ? uint32_t(~reference) * args.ir_leaf_size | type :
                             nvk_native_load_u32_cg(workspace + uint32_t(reference) / 2);
      }
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> allocator(header[8]);
      uint32_t output = allocator.fetch_add(1, cuda::memory_order_relaxed);
      uint32_t offset = args.internal_node_base + output * 40u;
      *reinterpret_cast<nvk_bvh_build_cuda_box *>(args.ir + offset) = nvk_build_box(args, references[0], references[1]);
      nvk_native_store_u32_cg(workspace + index, offset | 1u);
      int32_t parent = int32_t(nvk_native_load_u32_cg(&nodes[index * 2].before.parent));
      if (parent < 0)
         return;
      index = uint32_t(parent) / 2;
      cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
   }
}

template<typename T, bool IsKey> struct nvk_native_iterator {
   using value_type = T;
   using difference_type = ptrdiff_t;
   using pointer = T *;
   using reference = T;
   using iterator_category = std::random_access_iterator_tag;
   uint64_t address;
   uint32_t base;
   __device__ T operator[](difference_type index) const
   {
      nvk_build_key value = nvk_build_load_key(address, base + uint32_t(index), true);
      return IsKey ? T(value.key) : T(value.id);
   }
   __device__ nvk_native_iterator operator+(difference_type index) const
   {
      return {address, base + uint32_t(index)};
   }
};

template<uint32_t Threads, uint32_t Items> __device__ __forceinline__ void nvk_native_radix_tile(nvk_bvh_build_cuda_args args)
{
   using Key = unsigned long long;
   using Sort = cub::BlockRadixSort<Key, Threads, Items, uint32_t, 5, true, cub::BLOCK_SCAN_WARP_SCANS>;
   using KeyIterator = nvk_native_iterator<Key, true>;
   using ValueIterator = nvk_native_iterator<uint32_t, false>;
   using LoadKeys = cub::BlockLoad<Key, Threads, Items, cub::BLOCK_LOAD_TRANSPOSE>;
   using LoadValues = cub::BlockLoad<uint32_t, Threads, Items, cub::BLOCK_LOAD_TRANSPOSE>;
   __shared__ union {
      typename Sort::TempStorage sort;
      typename LoadKeys::TempStorage keys;
      typename LoadValues::TempStorage values;
   } temporary;
   Key keys[Items];
   uint32_t values[Items];
   uint32_t base = blockIdx.x * Threads * Items;
   uint32_t remaining = base < args.leaf_count ? args.leaf_count - base : 0;
   uint32_t valid = remaining < Threads * Items ? remaining : Threads * Items;
   LoadKeys(temporary.keys).Load(KeyIterator{args.src_keys, base}, keys, valid, ~Key(0));
   __syncthreads();
   LoadValues(temporary.values).Load(ValueIterator{args.src_keys, base}, values, valid, 0xffffffffu);
   __syncthreads();
   Sort(temporary.sort).SortBlockedToStriped(keys, values, 0, 64);
   for (uint32_t item = 0; item < Items; item++) {
      uint32_t index = base + item * Threads + threadIdx.x;
      if (index < args.leaf_count)
         nvk_build_store_key(args.dst_keys, index, true, nvk_build_key{keys[item], values[item]});
   }
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_morton_native(nvk_bvh_build_cuda_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (!index) {
      uint32_t *header = reinterpret_cast<uint32_t *>(args.ir_header);
      header[8] = 0;
      header[9] = 1;
      header[10] = 1;
   }
   if (index < args.leaf_count) {
      uint32_t id = reinterpret_cast<const uint32_t *>(args.input_ids)[index];
      nvk_build_store_key(args.dst_keys, index, (args.build_flags & NVK_BVH_BUILD_CUDA_KEYS_64BIT) != 0,
                          nvk_native_make_key(args, id));
   }
}

extern "C" __global__ __launch_bounds__(320) void nvk_bvh_build_radix_tiles_native(nvk_bvh_build_cuda_args args)
{
   nvk_native_radix_tile<320, 11>(args);
}

extern "C" __global__ __launch_bounds__(64) void nvk_bvh_build_radix_tiles_small_native(nvk_bvh_build_cuda_args args)
{
   nvk_native_radix_tile<64, 4>(args);
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_merge_native(nvk_bvh_build_cuda_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index >= args.leaf_count)
      return;
   bool wide = (args.build_flags & NVK_BVH_BUILD_CUDA_KEYS_64BIT) != 0;
   uint32_t destination = nvk_native_merge_destination(args, index);
   nvk_build_store_key(args.dst_keys, destination, wide, nvk_build_load_key(args.src_keys, index, wide));
}

template<int Bits> struct nvk_native_radix_policy {
   enum {
      BLOCK_THREADS = 256,
      ITEMS_PER_THREAD = 8,
      RADIX_BITS = Bits,
   };
   static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_DEFAULT;
};

template<int Bits> struct nvk_native_radix_down_policy {
   enum {
      BLOCK_THREADS = 256,
      ITEMS_PER_THREAD = 8,
      RADIX_BITS = Bits,
   };
   static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_DIRECT;
   static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_LDG;
   static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM = cub::RADIX_RANK_MEMOIZE;
   static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_RAKING_MEMOIZE;
};

template<int Bits> __device__ __forceinline__ void nvk_native_radix_upsweep(nvk_bvh_build_cuda_args args)
{
   using Agent = cub::AgentRadixSortUpsweep<nvk_native_radix_policy<Bits>, unsigned long long, int>;
   __shared__ typename Agent::TempStorage temporary;
   auto *spine = reinterpret_cast<int *>(args.node_info);
   auto *keys = reinterpret_cast<const unsigned long long *>(args.src_keys);
   auto region = nvk_native_radix_partition(args.leaf_count, gridDim.x, blockIdx.x);
   Agent agent(temporary, keys, int(args.merge_width), int(args.reserved));
   agent.ProcessRegion(int(region.begin), int(region.end));
   agent.template ExtractCounts<false>(spine, int(gridDim.x), int(blockIdx.x));
   uint32_t used = (1u << Bits) * gridDim.x;
   uint32_t padded = ((32u * gridDim.x + 11775u) / 11776u) * 11776u;
   for (uint32_t index = used + blockIdx.x * 256u + threadIdx.x;
        index < padded; index += gridDim.x * 256u)
      spine[index] = 0;
}

template<int Bits> __device__ __forceinline__ void nvk_native_radix_downsweep(nvk_bvh_build_cuda_args args)
{
   using Agent = cub::AgentRadixSortDownsweep<nvk_native_radix_down_policy<Bits>, false,
                                            unsigned long long, uint32_t, int>;
   __shared__ typename Agent::TempStorage temporary;
   auto *spine = reinterpret_cast<int *>(args.node_info);
   auto *input_keys = reinterpret_cast<const unsigned long long *>(args.src_keys);
   auto *output_keys = reinterpret_cast<unsigned long long *>(args.dst_keys);
   auto *input_values = reinterpret_cast<const uint32_t *>(args.src_keys + 8ull * args.ir_leaf_size);
   auto *output_values = reinterpret_cast<uint32_t *>(args.dst_keys + 8ull * args.ir_leaf_size);
   auto region = nvk_native_radix_partition(args.leaf_count, gridDim.x, blockIdx.x);
   Agent(temporary, int(args.leaf_count), spine, input_keys, output_keys, input_values,
         output_values, int(args.merge_width), int(args.reserved)).ProcessRegion(int(region.begin), int(region.end));
}

struct nvk_native_radix_prefix {
   int running;
   __device__ __forceinline__ int operator()(int aggregate)
   {
      int previous = running;
      running += aggregate;
      return previous;
   }
};

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_radix_pack_native(nvk_bvh_build_cuda_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < args.leaf_count) {
      auto value = nvk_build_load_key(args.src_keys, index, true);
      reinterpret_cast<unsigned long long *>(args.dst_keys)[index] = value.key;
      reinterpret_cast<uint32_t *>(args.dst_keys + 8ull * args.ir_leaf_size)[index] = value.id;
   }
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_radix_unpack_native(nvk_bvh_build_cuda_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index < args.leaf_count) {
      nvk_build_key value = {
         reinterpret_cast<const unsigned long long *>(args.src_keys)[index],
         reinterpret_cast<const uint32_t *>(args.src_keys + 8ull * args.ir_leaf_size)[index],
      };
      nvk_build_store_key(args.dst_keys, index, true, value);
   }
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_radix_upsweep_native(nvk_bvh_build_cuda_args args)
{
   nvk_native_radix_upsweep<5>(args);
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_radix_upsweep_alt_native(nvk_bvh_build_cuda_args args)
{
   nvk_native_radix_upsweep<4>(args);
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_radix_downsweep_native(nvk_bvh_build_cuda_args args)
{
   nvk_native_radix_downsweep<5>(args);
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_radix_downsweep_alt_native(nvk_bvh_build_cuda_args args)
{
   nvk_native_radix_downsweep<4>(args);
}

extern "C" __global__ __launch_bounds__(512) void nvk_bvh_build_radix_scan_native(nvk_bvh_build_cuda_args args)
{
   using Load = cub::BlockLoad<int, 512, 23, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
   using Scan = cub::BlockScan<int, 512, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
   using Store = cub::BlockStore<int, 512, 23, cub::BLOCK_STORE_WARP_TRANSPOSE>;
   __shared__ union {
      typename Load::TempStorage load;
      typename Scan::TempStorage scan;
      typename Store::TempStorage store;
   } temporary;
   if (blockIdx.x != 0)
      return;
   auto *spine = reinterpret_cast<int *>(args.node_info);
   uint32_t padded = ((32u * args.internal_node_base + 11775u) / 11776u) * 11776u;
   nvk_native_radix_prefix prefix = {0};
   for (uint32_t base = 0; base < padded; base += 11776u) {
      int values[23];
      Load(temporary.load).Load(spine + base, values);
      __syncthreads();
      Scan(temporary.scan).ExclusiveSum(values, values, prefix);
      __syncthreads();
      Store(temporary.store).Store(spine + base, values);
      __syncthreads();
   }
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_tiles(nvk_bvh_build_cuda_args args)
{
   __shared__ nvk_build_key keys[NVK_BVH_BUILD_CUDA_TILE_KEYS];
   uint32_t base = blockIdx.x * NVK_BVH_BUILD_CUDA_TILE_KEYS;
   bool wide = (args.build_flags & NVK_BVH_BUILD_CUDA_KEYS_64BIT) != 0;
   if (!blockIdx.x && !threadIdx.x) {
      uint32_t *header = reinterpret_cast<uint32_t *>(args.ir_header);
      header[8] = 0;
      header[9] = 1;
      header[10] = 1;
   }
   for (uint32_t local = threadIdx.x; local < NVK_BVH_BUILD_CUDA_TILE_KEYS; local += blockDim.x) {
      uint32_t index = base + local;
      uint32_t id = index < args.leaf_count ? reinterpret_cast<const uint32_t *>(args.input_ids)[index] : 0xffffffffu;
      keys[local] = nvk_build_make_key(args, id);
   }
   __syncthreads();
   for (uint32_t length = 2; length <= NVK_BVH_BUILD_CUDA_TILE_KEYS; length *= 2) {
      for (uint32_t distance = length / 2; distance; distance /= 2) {
         for (uint32_t local = threadIdx.x; local < NVK_BVH_BUILD_CUDA_TILE_KEYS; local += blockDim.x) {
            nvk_build_sort_pair(keys, local, length, distance);
         }
         __syncthreads();
      }
   }
   for (uint32_t local = threadIdx.x; local < NVK_BVH_BUILD_CUDA_TILE_KEYS; local += blockDim.x) {
      if (base + local < args.leaf_count)
         nvk_build_store_key(args.dst_keys, base + local, wide, keys[local]);
   }
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_merge(nvk_bvh_build_cuda_args args)
{
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index >= args.leaf_count)
      return;
   bool wide = (args.build_flags & NVK_BVH_BUILD_CUDA_KEYS_64BIT) != 0;
   nvk_build_key key = nvk_build_load_key(args.src_keys, index, wide);
   uint32_t destination = nvk_build_merge_destination(args, index);
   nvk_build_store_key(args.dst_keys, destination, wide, key);
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_topology(nvk_bvh_build_cuda_args args)
{
   nvk_build_topology_node(args, blockIdx.x * blockDim.x + threadIdx.x);
}

extern "C" __global__ __launch_bounds__(256) void nvk_bvh_build_ir(nvk_bvh_build_cuda_args args)
{
   uint32_t count = reinterpret_cast<const uint32_t *>(args.ir_header)[6];
   uint32_t internal_count = (count > 2 ? count : 2) - 1;
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (index >= internal_count)
      return;
   auto *nodes = reinterpret_cast<nvk_bvh_build_cuda_node_info *>(args.node_info);
   uint32_t *header = reinterpret_cast<uint32_t *>(args.ir_header);
   for (;;) {
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(nodes[index].path_count);
      if (ready.fetch_add(1, cuda::memory_order_acq_rel) != 2)
         return;
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> allocator(header[8]);
      uint32_t output = allocator.fetch_add(1, cuda::memory_order_relaxed);
      uint32_t offset = args.internal_node_base + output * 40u;
      uint32_t left = nodes[index].children[0];
      uint32_t right = nodes[index].children[1];
      *reinterpret_cast<nvk_bvh_build_cuda_box *>(args.ir + offset) = nvk_build_box(args, left, right);
      uint32_t parent = nodes[index].parent;
      if (parent == 0xffffffffu)
         return;
      uint32_t slot = (parent >> 29) & 1u;
      index = parent & ~(1u << 29);
      nodes[index].children[slot] = offset | 1u;
      cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
   }
}
#endif
