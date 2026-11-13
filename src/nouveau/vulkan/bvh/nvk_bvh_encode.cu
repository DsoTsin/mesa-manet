#include "nvk_bvh.h"
#include "nvk_bvh_encode_wide.h"
#include <cuda_runtime.h>
#ifdef __CUDACC__
#include <cuda/atomic>
#endif
#include <math.h>

static_assert(sizeof(nvk_bvh_encode_args) == 136);
static_assert(offsetof(nvk_bvh_encode_args, leaf_capacity) == 104);
static_assert(offsetof(nvk_bvh_encode_args, pass) == 116);

namespace {

struct nvk_encoder {
   const nvk_bvh_encode_args &args;

   __device__ __forceinline__ uint32_t load32(uint64_t address, uint32_t offset = 0)
   {
      return reinterpret_cast<const uint32_t *>(address + offset)[0];
   }

   __device__ __forceinline__ uint64_t load64(uint64_t address, uint32_t offset)
   {
      return uint64_t(load32(address, offset)) |
             (uint64_t(load32(address, offset + 4)) << 32);
   }

   __device__ __forceinline__ void store32(uint64_t address, uint32_t offset, uint32_t value)
   {
      reinterpret_cast<uint32_t *>(address + offset)[0] = value;
   }

   __device__ __forceinline__ void store64(uint64_t address, uint32_t offset, uint64_t value)
   {
      store32(address, offset, uint32_t(value));
      store32(address, offset + 4, uint32_t(value >> 32));
   }

   __device__ __forceinline__ float load_float(uint64_t address, uint32_t offset)
   {
      return __uint_as_float(load32(address, offset));
   }

   __device__ __forceinline__ uint64_t mapping_address(uint32_t index)
   {
      return args.mapping + uint64_t(index) * NVK_BVH_MAPPING_SIZE;
   }

   __device__ __forceinline__ uint64_t leaf_address(uint32_t rank)
   {
      uint32_t stride = args.geometry_type == 0 ? 128 : args.geometry_type == 1 ? 8 : 64;
      return args.dst + args.leaf_offset + uint64_t(rank) * stride;
   }

   __device__ __forceinline__ uint32_t node_child(uint32_t reference, uint32_t child)
   {
      return load32(args.ir + (reference & ~3u), 24 + child * 4);
   }

   __device__ __forceinline__ bool is_internal(uint32_t reference)
   {
      return reference != NVK_BVH_INVALID_NODE && (reference & 3u) == 1u;
   }

   __device__ __forceinline__ uint32_t internal_count()
   {
      return max(load32(args.ir_header, 32), 1u);
   }

   __device__ __forceinline__ uint32_t ir_reference(uint32_t index)
   {
      return (args.leaf_capacity * args.ir_leaf_size + index * 40u) | 1u;
   }

   __device__ __forceinline__ void prepare_header(bool prepare_mapping = true)
   {
      for (uint32_t offset = 0; offset < NVK_BVH_HEADER_SIZE; offset += 4)
         store32(args.dst, offset, 0);
      uint32_t active_count = load32(args.ir_header, 24);
      uint32_t node_count = internal_count();
      uint32_t root = ir_reference(node_count - 1u) & ~3u;
      bool instances = args.geometry_type == 2;
      store32(args.dst, 0, instances ? 0x80003u : 0x100003u);
      store32(args.dst, 4, 8);
      store64(args.dst, 8, args.size);
      store64(args.dst, 16, args.geometry_type == 0 ? args.leaf_offset : 0);
      store64(args.dst, 32, args.node_offset);
      store64(args.dst, 40, args.refit_data_offset);
      store64(args.dst, 48, instances ? args.metadata_offset : 0);
      store64(args.dst, 56, instances ? args.leaf_offset : 0);
      store64(args.dst, 64, args.refit_leaf_map_offset);
      store32(args.dst, 88, active_count);
      store32(args.dst, 96, args.leaf_capacity);
      store32(args.dst, 104, 12);
      store32(args.dst, 108, 12);
      store32(args.dst, 116, args.leaf_capacity);
      store32(args.dst, 120, args.geometry_count);
      store32(args.dst, 124, args.geometry_count);
      store32(args.dst, 168, node_count);
      store64(args.dst, 208, args.dst);
      store32(args.dst, 220, 0x01010101u);
      store32(args.dst, 224, NVK_BVH_COMPAT_UUID_0);
      store32(args.dst, 228, NVK_BVH_COMPAT_UUID_1);
      store32(args.dst, 232, NVK_BVH_COMPAT_UUID_2);
      store32(args.dst, 236, NVK_BVH_COMPAT_UUID_3);
      for (uint32_t axis = 0; axis < 3; axis++) {
         store32(args.dst, NVK_BVH_BOUNDS_OFFSET + axis * 4,
                 active_count == 0 ? 0 : load32(args.ir + root, axis * 4));
         store32(args.dst, NVK_BVH_BOUNDS_OFFSET + 12 + axis * 4,
                 active_count == 0 ? 0 : load32(args.ir + root, 12 + axis * 4));
      }
      uint32_t instance_count = instances ? args.leaf_capacity : 0;
      store64(args.dst, NVK_BVH_COMPACTED_SIZE_OFFSET, args.size);
      store64(args.dst, NVK_BVH_SERIALIZATION_SIZE_OFFSET,
              args.size + 56 + uint64_t(instance_count) * 8);
      store64(args.dst, NVK_BVH_PRIMITIVE_DATA_OFFSET, args.primitive_data_offset);
      store32(args.dst, NVK_BVH_GEOMETRY_TYPE_OFFSET, args.geometry_type);
      store32(args.dst, NVK_BVH_ACTIVE_LEAF_COUNT_OFFSET, active_count);
      store32(args.dst, NVK_BVH_PRIMITIVE_COUNT_OFFSET, args.leaf_capacity);
      store32(args.dst, NVK_BVH_NODE_COUNT_OFFSET, node_count);
      store32(args.dst, NVK_BVH_INSTANCE_COUNT_OFFSET, instance_count);
      store32(args.dst, NVK_BVH_ROOT_FLAGS_OFFSET, 0);
      store32(args.dst, NVK_BVH_MAGIC_OFFSET, NVK_BVH_MAGIC);
      store64(args.dst, NVK_BVH_REFIT_DATA_OFFSET, args.refit_data_offset);
      store64(args.dst, NVK_BVH_REFIT_LEAF_MAP_OFFSET, args.refit_leaf_map_offset);
      store64(args.dst, NVK_BVH_REFIT_LEAF_OFFSET, args.leaf_offset);
      store64(args.dst, NVK_BVH_OMM_METADATA_OFFSET, args.omm_metadata_offset);
      if (args.build_flags & NVK_BVH_ENCODE_OPACITY_MICROMAPS)
         store32(args.dst, 0, load32(args.dst, 0) | 0x10000000u);
      store32(args.dst, 244, args.build_flags & 0x3fffffffu);
      if (args.geometry_type == 0 && (args.build_flags & NVK_BVH_ENCODE_TRIANGLE_PACKING))
         store32(args.dst, 240, uint32_t((args.refit_leaf_map_offset + uint64_t(args.leaf_capacity) * 4 + 127) / 128));
      if (args.refit_data_offset != 0)
         store32(args.dst, 0, load32(args.dst, 0) | 0x20000u);
      store32(args.prefix_sums, 0, 1);
      store32(args.prefix_sums, 4, 0);
      if (prepare_mapping) {
         store32(mapping_address(node_count - 1u), 0, 0);
         store32(mapping_address(node_count - 1u), 12, NVK_BVH_INVALID_NODE);
      }
   }

   __device__ __forceinline__ void prepare_local_mapping(uint32_t index)
   {
      if ((args.refit_data_offset == 0 || load32(args.ir_header, 24) != args.leaf_capacity) &&
          index < args.leaf_capacity)
         store32(args.leaf_map, index * 4, NVK_BVH_INVALID_NODE);
      if (index == 0)
         prepare_header();
   }

   __device__ __forceinline__ void prepare_global_mapping(uint32_t index)
   {
      uint32_t count = internal_count();
      uint32_t internal = 0, leaves = 0;
      bool empty = load32(args.ir_header, 24) == 0;
      if (index < count && !empty) {
         for (uint32_t child = 0; child < 2; child++) {
            uint32_t reference = node_child(ir_reference(index), child);
            internal += uint32_t(is_internal(reference));
            leaves += uint32_t(reference != NVK_BVH_INVALID_NODE && !is_internal(reference));
         }
      }
      uint32_t lane = threadIdx.x & 31u;
      uint32_t internal_sum = internal, leaf_sum = leaves;
      for (uint32_t delta = 1; delta < 32; delta *= 2) {
         uint32_t i = __shfl_up_sync(0xffffffffu, internal_sum, delta);
         uint32_t l = __shfl_up_sync(0xffffffffu, leaf_sum, delta);
         if (lane >= delta) {
            internal_sum += i;
            leaf_sum += l;
         }
      }
      uint32_t internal_total = __shfl_sync(0xffffffffu, internal_sum, 31);
      uint32_t leaf_total = __shfl_sync(0xffffffffu, leaf_sum, 31);
      uint32_t internal_allocation = 0, leaf_allocation = 0;
      if (lane == 0) {
         uint32_t *allocator = reinterpret_cast<uint32_t *>(args.prefix_sums);
         if (internal_total)
            internal_allocation = atomicAdd(allocator, internal_total);
         if (leaf_total)
            leaf_allocation = atomicAdd(allocator + 1, leaf_total);
      }
      internal_allocation = __shfl_sync(0xffffffffu, internal_allocation, 0);
      leaf_allocation = __shfl_sync(0xffffffffu, leaf_allocation, 0);
      if (index >= count)
         return;
      uint32_t child_base = internal_allocation + internal_sum - internal;
      uint32_t leaf_base = leaf_allocation + leaf_sum - leaves;
      uint64_t mapping = mapping_address(index);
      store32(mapping, 4, child_base);
      store32(mapping, 8, leaf_base);
      for (uint32_t child = 0; child < 2; child++) {
         uint32_t reference = empty ? NVK_BVH_INVALID_NODE : node_child(ir_reference(index), child);
         if (is_internal(reference)) {
            uint32_t child_index = ((reference & ~3u) - args.leaf_capacity * args.ir_leaf_size) / 40u;
            store32(mapping_address(child_index), 0, child_base++);
            store32(mapping_address(child_index), 12, index * 16 + child);
         } else if (reference != NVK_BVH_INVALID_NODE) {
            uint32_t original = (reference & ~3u) / args.ir_leaf_size;
            store32(args.leaf_map, original * 4, leaf_base++);
         }
      }
   }

   __device__ __forceinline__ uint2 encode_child(uint32_t reference, const float origin[3], const uint32_t exponents[3])
   {
      if (reference == NVK_BVH_INVALID_NODE)
         return make_uint2(0, 0xff);
      uint64_t source = args.ir + (reference & ~3u);
      uint32_t data = is_internal(reference) ? 0u : 1u;
      if (args.geometry_type == 0 && !is_internal(reference) && (load32(source, 64) & 0x80000000u) == 0)
         data |= 0x80u;
      bool invalid = false;
      for (uint32_t axis = 0; axis < 3; axis++) {
         float lo = load_float(source, axis * 4);
         float hi = load_float(source, 12 + axis * 4);
         invalid = invalid || isnan(lo) || isnan(hi) || lo > hi;
      }
      if (invalid)
         return make_uint2(0x00ff00ffu, 0x000001ffu | (data << 24));
      uint32_t lower[3], upper[3];
      for (uint32_t axis = 0; axis < 3; axis++) {
         lower[axis] = 0;
         upper[axis] = 255;
         if (exponents[axis] >= 8) {
            float scale = __uint_as_float((262u - exponents[axis]) << 23);
            float lo = __fadd_rd(load_float(source, axis * 4), -origin[axis]);
            float hi = __fadd_ru(load_float(source, 12 + axis * 4), -origin[axis]);
            float qlo = floorf(__fmul_rd(lo, scale));
            float qhi = __fadd_rn(ceilf(__fmul_ru(hi, scale)), -1.0f);
            lower[axis] = uint32_t(fminf(fmaxf(qlo, 0.0f), 255.0f));
            upper[axis] = uint32_t(fminf(fmaxf(qhi, 0.0f), 255.0f));
         }
      }
      return make_uint2(lower[0] | (upper[0] << 8) | (lower[1] << 16) | (upper[1] << 24),
                        lower[2] | (upper[2] << 8) | 0xff0000u | (data << 24));
   }

   __device__ __forceinline__ void encode_node(uint32_t index)
   {
      uint64_t mapping = mapping_address(index);
      uint32_t reference = ir_reference(index);
      uint32_t native_index = load32(mapping, 0);
      uint32_t child_base = load32(mapping, 4);
      uint32_t leaf_base = load32(mapping, 8);
      uint32_t parent = load32(mapping, 12);
      uint64_t source = args.ir + (reference & ~3u);
      uint64_t destination = args.dst + args.node_offset + uint64_t(native_index) * NVK_BVH_NODE_SIZE;
      bool empty = load32(args.dst, NVK_BVH_ACTIVE_LEAF_COUNT_OFFSET) == 0;
      float origin[3];
      uint32_t exponents[3];
      for (uint32_t axis = 0; axis < 3; axis++) {
         origin[axis] = empty ? 0.0f : load_float(source, axis * 4);
         float maximum = empty ? 0.0f : load_float(source, 12 + axis * 4);
         float extent = fmaxf(0.0f, __fadd_ru(maximum, -origin[axis]));
         exponents[axis] = min((__float_as_uint(extent) + 0x7fffffu) >> 23, 255u);
      }
      uint32_t leaf_type = args.geometry_type == 0 ? 1u : args.geometry_type == 1 ? 0u : 2u;
      uint64_t pointer = leaf_address(leaf_base) - destination;
      pointer = args.geometry_type == 1 ? (pointer >> 3) & ((uint64_t(1) << 47) - 1) :
                                        pointer & ((uint64_t(1) << 49) - 1);
      uint32_t parent_slot = parent == NVK_BVH_INVALID_NODE ? 15u : parent & 15u;
      uint32_t parent_index = parent == NVK_BVH_INVALID_NODE ? native_index : load32(mapping_address(parent >> 4), 0);
      store32(destination, 0, 0x20u | (leaf_type << 6) | (exponents[0] << 8) |
              (exponents[1] << 16) | (exponents[2] << 24));
      for (uint32_t axis = 0; axis < 3; axis++)
         store32(destination, 4 + axis * 4, __float_as_uint(origin[axis]));
      store32(destination, 16, uint32_t(pointer));
      store32(destination, 20, uint32_t(pointer >> 32) | (parent_slot << 28));
      store32(destination, 24, parent_index - native_index);
      store32(destination, 28, child_base - native_index);
      uint32_t internal_children = 0, leaf_counts = 0;
      for (uint32_t child = 0; child < 2; child++) {
         uint32_t child_ref = empty ? NVK_BVH_INVALID_NODE : node_child(reference, child);
         internal_children += uint32_t(is_internal(child_ref));
         if (child_ref != NVK_BVH_INVALID_NODE && !is_internal(child_ref))
            leaf_counts |= 1u << (child * 8);
         uint2 encoded = encode_child(child_ref, origin, exponents);
         store32(destination, 32 + child * 8, encoded.x);
         store32(destination, 36 + child * 8, encoded.y);
      }
      for (uint32_t child = 2; child < 12; child++) {
         store32(destination, 32 + child * 8, 0);
         store32(destination, 36 + child * 8, 0xff);
      }
      if (args.refit_data_offset != 0) {
         uint64_t auxiliary = args.dst + args.refit_data_offset + uint64_t(native_index) * 32;
         store32(auxiliary, 0, leaf_base);
         store32(auxiliary, 4, leaf_counts);
         store32(auxiliary, 8, 0);
         store32(auxiliary, 12, 0);
         for (uint32_t axis = 0; axis < 3; axis++)
            store32(auxiliary, 16 + axis * 4, empty ? 0 : load32(source, 12 + axis * 4));
         store32(auxiliary, 28, internal_children | 0xff00u);
      }
      if (args.build_flags & NVK_BVH_ENCODE_TRIANGLE_PACKING) {
         uint64_t auxiliary = args.prefix_sums + 32 + uint64_t(native_index) * 32;
         store32(auxiliary, 0, leaf_base);
         store32(auxiliary, 4, leaf_counts);
         store32(auxiliary, 8, 0);
         store32(auxiliary, 12, 0);
         for (uint32_t axis = 0; axis < 3; axis++)
            store32(auxiliary, 16 + axis * 4, empty ? 0 : load32(source, 12 + axis * 4));
         store32(auxiliary, 28, internal_children | 0xff00u);
      }
   }

   __device__ __forceinline__ void encode_instance(uint32_t rank, uint64_t source)
   {
      uint64_t destination = leaf_address(rank);
      uint64_t blas = load64(source, 24);
      uint32_t custom = load32(source, 32);
      uint32_t sbt = load32(source, 36);
      uint32_t instance_id = load32(source, 88);
      uint32_t flags = sbt >> 24;
      uint32_t hardware_flags = 1u | ((flags & 1u) << 2) | (flags & 2u) | ((flags & 12u) << 2);
      float m[12], inverse[12];
      for (uint32_t component = 0; component < 12; component++)
         m[component] = load_float(source, 40 + component * 4);
      inverse[0] = m[5] * m[10] - m[6] * m[9];
      inverse[1] = m[2] * m[9] - m[1] * m[10];
      inverse[2] = m[1] * m[6] - m[2] * m[5];
      inverse[4] = m[6] * m[8] - m[4] * m[10];
      inverse[5] = m[0] * m[10] - m[2] * m[8];
      inverse[6] = m[2] * m[4] - m[0] * m[6];
      inverse[8] = m[4] * m[9] - m[5] * m[8];
      inverse[9] = m[1] * m[8] - m[0] * m[9];
      inverse[10] = m[0] * m[5] - m[1] * m[4];
      float rdet = 1.0f / (m[0] * inverse[0] + m[1] * inverse[4] + m[2] * inverse[8]);
      for (uint32_t row = 0; row < 3; row++) {
         for (uint32_t column = 0; column < 3; column++)
            inverse[row * 4 + column] *= rdet;
         inverse[row * 4 + 3] = -(inverse[row * 4] * m[3] +
                                  inverse[row * 4 + 1] * m[7] +
                                  inverse[row * 4 + 2] * m[11]);
      }
      uint64_t root = blas + load64(blas, 32);
      uint64_t packed = (root & ((uint64_t(1) << 49) - 1)) |
                        (uint64_t(custom >> 24) << 49) | (uint64_t(1) << 57) |
                        (uint64_t(hardware_flags) << 58);
      packed |= (uint64_t(flags & 0x30u) >> 1);
      store32(destination, 0, instance_id + 1);
      store32(destination, 4, (sbt & 0xffffffu) | ((flags & 0x3cu) << 26));
      store64(destination, 8, packed);
      for (uint32_t component = 0; component < 12; component++)
         store32(destination, 16 + component * 4, __float_as_uint(inverse[component]));
      uint32_t slot = instance_id & 15u;
      uint64_t metadata = args.dst + args.metadata_offset +
                          uint64_t(instance_id >> 4) * NVK_BVH_INSTANCE_DATA_UNIT_SIZE;
      store32(metadata, slot * 8, (custom >> 24) | (flags << 8));
      store32(metadata, slot * 8 + 4, sbt & 0xffffffu);
      store64(metadata, 128 + slot * 8, blas);
      for (uint32_t component = 0; component < 12; component++)
         store32(metadata, 256 + slot * 48 + component * 4, __float_as_uint(m[component]));
      store32(metadata, 1024 + slot * 8, rank);
      store32(metadata, 1028 + slot * 8, custom & 0xffffffu);
   }

   __device__ __forceinline__ void encode_leaf(uint32_t index)
   {
      uint32_t rank = load32(args.leaf_map, index * 4);
      if (rank == NVK_BVH_INVALID_NODE) {
         if (args.geometry_type == 2) {
            uint32_t slot = index & 15u;
            uint64_t metadata = args.dst + args.metadata_offset +
                                uint64_t(index >> 4) * NVK_BVH_INSTANCE_DATA_UNIT_SIZE;
            store64(metadata, slot * 8, 0);
            store64(metadata, 128 + slot * 8, 0);
            store64(metadata, 1024 + slot * 8, 0);
            for (uint32_t word = 0; word < 12; word++)
               store32(metadata, 256 + slot * 48 + word * 4, 0);
         }
         return;
      }
      uint64_t source = args.ir + uint64_t(index) * args.ir_leaf_size;
      if (args.refit_leaf_map_offset != 0)
         store32(args.dst + args.refit_leaf_map_offset, rank * 4, index);
      if (args.geometry_type == 2) {
         encode_instance(rank, source);
         return;
      }
      uint32_t primitive_id = load32(source, 24);
      uint32_t geometry_id = load32(source, 28);
      uint64_t primitive_data = args.dst + args.primitive_data_offset + uint64_t(index) * 8;
      store32(primitive_data, 0, primitive_id);
      store32(primitive_data, 4, geometry_id);
      uint64_t destination = leaf_address(rank);
      store32(destination, 0, primitive_id);
      store32(destination, 4, geometry_id);
   }

   __device__ __forceinline__ void encode_triangle_cooperative(uint32_t invocation)
   {
      uint32_t index = invocation / 16;
      uint32_t lane = invocation & 15u;
      if (index >= args.leaf_capacity)
         return;
      uint32_t rank = load32(args.leaf_map, index * 4);
      if (rank == NVK_BVH_INVALID_NODE)
         return;
      uint64_t source = args.ir + uint64_t(index) * args.ir_leaf_size;
      uint64_t destination = leaf_address(rank);
      if (lane == 0) {
         uint64_t primitive_data = args.dst + args.primitive_data_offset + uint64_t(index) * 8;
         store32(primitive_data, 0, load32(source, 60));
         store32(primitive_data, 4, load32(source, 64));
         if (args.refit_leaf_map_offset != 0)
            store32(args.dst + args.refit_leaf_map_offset, rank * 4, index);
      }
      if (args.build_flags & NVK_BVH_ENCODE_TRIANGLE_PACKING)
         return;
      for (uint32_t word = lane; word < 32; word += 16) {
         uint32_t value = 0;
         if (word < 9)
            value = load32(source, 24 + word * 4);
         else if (word == 30)
            value = index;
         else if (word == 31)
            value = (load32(source, 64) & 0x80000000u) == 0 ? 1u << 26 : 0;
         store32(destination, word * 4, value);
      }
   }

   __device__ __forceinline__ nvk_bvh_wide_workspace *wide_workspace()
   {
      return reinterpret_cast<nvk_bvh_wide_workspace *>(args.prefix_sums);
   }

   __device__ __forceinline__ nvk_bvh_wide_node wide_node(uint32_t index)
   {
      return reinterpret_cast<const nvk_bvh_wide_node *>(args.mapping)[index];
   }

   __device__ __forceinline__ uint32_t wide_pair_reference(uint32_t pair)
   {
      uint64_t references = args.mapping + uint64_t(max(args.leaf_capacity, NVK_BVH_WIDE_MIN_CAPACITY)) * 32;
      return load32(references, (pair / 2) * 4);
   }

   __device__ __forceinline__ uint64_t wide_auxiliary(uint32_t index)
   {
      return args.prefix_sums + NVK_BVH_WIDE_WORKSPACE_SIZE +
             uint64_t(index) * NVK_BVH_WIDE_AUXILIARY_SIZE;
   }

   __device__ __forceinline__ uint32_t wide_error()
   {
      return reinterpret_cast<volatile nvk_bvh_wide_workspace *>(args.prefix_sums)->error;
   }

   __device__ __forceinline__ uint32_t wide_load_mutable(uint64_t address)
   {
#ifdef __CUDA_ARCH__
      uint32_t result;
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(result) : "l"(address) : "memory");
      return result;
#else
      return reinterpret_cast<volatile uint32_t *>(address)[0];
#endif
   }

   __device__ __forceinline__ uint32_t wide_load_ready(uint64_t address)
   {
#ifdef __CUDA_ARCH__
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(*reinterpret_cast<uint32_t *>(address));
      return ready.load(cuda::memory_order_acquire);
#else
      return reinterpret_cast<volatile uint32_t *>(address)[0];
#endif
   }

   __device__ __forceinline__ void wide_publish(uint64_t address, uint32_t value)
   {
#ifdef __CUDA_ARCH__
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(*reinterpret_cast<uint32_t *>(address));
      ready.store(value, cuda::memory_order_release);
#else
      __threadfence();
      reinterpret_cast<volatile uint32_t *>(address)[0] = value;
#endif
   }

   __device__ __forceinline__ bool triangle_packing()
   {
      return args.geometry_type == 0 && (args.build_flags & NVK_BVH_ENCODE_TRIANGLE_PACKING) != 0;
   }

   __device__ __forceinline__ void prepare_wide_mapping(uint32_t index)
   {
      uint32_t count = max(wide_workspace()->node_count, 1u);
      if (index < args.leaf_capacity)
         store32(args.leaf_map, index * 4, NVK_BVH_INVALID_NODE);
      if (index < min(count, max(args.leaf_capacity, 2u))) {
         uint64_t auxiliary = wide_auxiliary(index);
         store32(auxiliary, 0, NVK_BVH_INVALID_NODE);
         store32(auxiliary, 4, 0);
         store32(auxiliary, 8, 0);
         store32(auxiliary, 12, 0);
      }
      if (index != 0)
         return;
      uint32_t planner_error = wide_workspace()->error;
      prepare_header(false);
      wide_workspace()->node_count = count;
      wide_workspace()->error = planner_error;
      store32(args.prefix_sums, NVK_BVH_WIDE_WORK_COUNTER_OFFSET, 0);
      if (count > internal_count())
         wide_workspace()->error |= NVK_BVH_WIDE_NODE_LIMIT_ERROR;
      if (args.geometry_type == 0 && wide_workspace()->max_leaf_primitives > 7 &&
          load32(args.ir_header, 24) > wide_workspace()->branching_factor && !triangle_packing())
         wide_workspace()->error |= NVK_BVH_WIDE_TRIANGLE_PACKING_REQUIRED;
      if (wide_workspace()->branching_factor == 0 || wide_workspace()->branching_factor > 12)
         wide_workspace()->error |= NVK_BVH_WIDE_CHILD_LIMIT_ERROR;
      store32(args.dst, 88, count);
      store32(args.dst, 168, count);
      store32(args.dst, NVK_BVH_NODE_COUNT_OFFSET, count);
      uint64_t root_source = args.ir + (wide_pair_reference(0) & ~3u);
      bool empty = load32(args.ir_header, 24) == 0;
      for (uint32_t axis = 0; axis < 3; axis++) {
         store32(args.dst, NVK_BVH_BOUNDS_OFFSET + axis * 4,
                 empty ? 0 : load32(root_source, axis * 4));
         store32(args.dst, NVK_BVH_BOUNDS_OFFSET + 12 + axis * 4,
                 empty ? 0 : load32(root_source, 12 + axis * 4));
      }
      store32(wide_auxiliary(0), 16, 0);
      store32(wide_auxiliary(0), 20, NVK_BVH_INVALID_NODE);
      store32(wide_auxiliary(0), 24, NVK_BVH_INVALID_NODE);
      store32(wide_auxiliary(0), 28, 0);
      store32(wide_auxiliary(0), 0, 0);
   }

   __device__ __forceinline__ uint32_t wide_primitive(int32_t child_pair, uint32_t primitive)
   {
      while (child_pair >= 0) {
         nvk_bvh_wide_node left = wide_node(uint32_t(child_pair));
         uint32_t left_count = uint32_t(left.subtree_primitives) & 0x7fffffffu;
         if (primitive >= left_count) {
            primitive -= left_count;
            child_pair++;
         }
         child_pair = wide_node(uint32_t(child_pair)).child_pair;
      }
      return ~uint32_t(child_pair);
   }

   __device__ __forceinline__ float half_area(uint64_t source)
   {
      float x, y, z, area;
#ifdef __CUDA_ARCH__
      float lo0 = load_float(source, 0), lo1 = load_float(source, 4), lo2 = load_float(source, 8);
      float hi0 = load_float(source, 12), hi1 = load_float(source, 16), hi2 = load_float(source, 20);
      asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(x) : "f"(hi0), "f"(lo0));
      asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(hi1), "f"(lo1));
      asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(z) : "f"(hi2), "f"(lo2));
      asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(area) : "f"(x), "f"(y));
      asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(area) : "f"(z), "f"(y), "f"(area));
      asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(area) : "f"(x), "f"(z), "f"(area));
#else
      x = __fadd_rn(load_float(source, 12), -load_float(source, 0));
      y = __fadd_rn(load_float(source, 16), -load_float(source, 4));
      z = __fadd_rn(load_float(source, 20), -load_float(source, 8));
      area = __fmaf_rn(x, z, __fmaf_rn(z, y, __fmul_rn(x, y)));
#endif
      return area;
   }

   __device__ __forceinline__ uint32_t nth_lane(uint32_t mask, uint32_t ordinal)
   {
      uint32_t lane = 0;
      for (uint32_t step = 16; step; step /= 2) {
         uint32_t count = __popc(mask & ((1u << step) - 1u));
         if (ordinal >= count) {
            ordinal -= count;
            mask >>= step;
            lane += step;
         }
      }
      return lane;
   }

   __device__ __forceinline__ void construct_wide(uint32_t index, uint32_t lane)
   {
      uint32_t count = max(wide_workspace()->node_count, 1u);
      if (index >= count)
         return;
      uint32_t initial_error = lane == 0 ? wide_error() : 0;
      initial_error = __shfl_sync(0xffffffffu, initial_error, 0);
      if (initial_error)
         return;
      uint64_t auxiliary = wide_auxiliary(index);
      uint32_t first_item = NVK_BVH_INVALID_NODE;
      uint32_t failed = 0;
      if (lane == 0) {
         first_item = wide_load_ready(auxiliary);
         while (first_item == NVK_BVH_INVALID_NODE) {
            failed = wide_error();
            if (failed)
               break;
            __nanosleep(200);
            first_item = wide_load_ready(auxiliary);
         }
      }
      failed = __shfl_sync(0xffffffffu, failed, 0);
      if (failed)
         return;
      first_item = __shfl_sync(0xffffffffu, first_item, 0);
      uint32_t root_pair = wide_load_mutable(auxiliary + 16);
      uint32_t parent = wide_load_mutable(auxiliary + 20);
      uint32_t parent_slot = wide_load_mutable(auxiliary + 24);
      uint32_t subtree_primitives = (uint32_t(wide_node(root_pair).subtree_primitives) +
                                     uint32_t(wide_node(root_pair + 1).subtree_primitives)) & 0x7fffffffu;
      nvk_bvh_wide_node child{};
      child.child_pair = int32_t(0x80000000u);
      uint32_t trail = lane;
      if (subtree_primitives > wide_workspace()->branching_factor) {
         trail = __brev(lane) >> 27;
         int32_t pair = int32_t(root_pair);
         do {
            child = wide_node(uint32_t(pair) + (trail & 1u));
            trail >>= 1;
            pair = child.child_pair;
         } while (pair >= 0 && child.owner_subtree_primitives >= int32_t(subtree_primitives));
      } else {
         int32_t pair = int32_t(root_pair);
         do {
            uint32_t left_count = uint32_t(wide_node(uint32_t(pair)).subtree_primitives) & 0x7fffffffu;
            if (trail >= left_count) {
               trail -= left_count;
               pair++;
            }
            child = wide_node(uint32_t(pair));
            pair = child.child_pair;
         } while (pair >= 0);
      }
      bool valid = trail == 0 && child.child_pair != int32_t(0x80000000u);
      uint32_t valid_mask = __ballot_sync(0xffffffffu, valid);
      uint32_t valid_count = __popc(valid_mask);
      if (args.geometry_type == 2 && index == 0) {
         bool order_last = false;
         if (valid && child.subtree_primitives < 0 && child.child_pair < 0) {
            uint32_t original = ~uint32_t(child.child_pair);
            float instance_area = half_area(args.ir + uint64_t(original) * args.ir_leaf_size);
            float world_area = half_area(args.ir + (wide_pair_reference(0) & ~3u));
            float threshold;
#ifdef __CUDA_ARCH__
            asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(threshold) : "f"(world_area), "f"(0.25f));
#else
            threshold = __fmul_rn(world_area, 0.25f);
#endif
            order_last = threshold < instance_area;
         }
         uint32_t last_mask = __ballot_sync(0xffffffffu, order_last);
         if (last_mask) {
            uint32_t first_mask = valid_mask & ~last_mask;
            uint32_t first_count = __popc(first_mask);
            uint32_t source_lane = 0;
            if (lane < valid_count)
               source_lane = nth_lane(lane < first_count ? first_mask : last_mask,
                                      lane < first_count ? lane : lane - first_count);
            child.child_pair = int32_t(__shfl_sync(0xffffffffu, uint32_t(child.child_pair), source_lane));
            child.subtree_primitives = int32_t(__shfl_sync(0xffffffffu, uint32_t(child.subtree_primitives), source_lane));
            valid = lane < valid_count;
            valid_mask = __ballot_sync(0xffffffffu, valid);
         }
      }
      bool leaf = valid && child.subtree_primitives < 0;
      uint32_t primitives = uint32_t(child.subtree_primitives) & 0x7fffffffu;
      uint32_t internal_mask = __ballot_sync(0xffffffffu, valid && !leaf);
      uint32_t internal_children = __popc(internal_mask);
      bool packing_required = leaf && args.geometry_type == 0 && primitives > 7 && !triangle_packing();
      bool child_limit = leaf && args.geometry_type == 2 && primitives != 1;
      bool any_packing_required = __any_sync(0xffffffffu, packing_required);
      bool any_child_limit = __any_sync(0xffffffffu, child_limit);
      if (valid_count > 12 || any_packing_required || any_child_limit) {
         if (lane == 0)
            atomicOr(&wide_workspace()->error, valid_count > 12 ? NVK_BVH_WIDE_CHILD_LIMIT_ERROR :
                     any_packing_required ? NVK_BVH_WIDE_TRIANGLE_PACKING_REQUIRED : NVK_BVH_WIDE_CHILD_LIMIT_ERROR);
         return;
      }
      uint32_t lower_mask = (1u << lane) - 1u;
      uint32_t child_slot = __popc(valid_mask & lower_mask);
      uint32_t child_index = __popc(internal_mask & lower_mask);
      uint32_t leaf_sum = leaf ? primitives : 0;
      uint32_t internal_sum = valid && !leaf ? primitives : 0;
      for (uint32_t delta = 1; delta < 32; delta *= 2) {
         uint32_t leaves = __shfl_up_sync(0xffffffffu, leaf_sum, delta);
         uint32_t internals = __shfl_up_sync(0xffffffffu, internal_sum, delta);
         if (lane >= delta) {
            leaf_sum += leaves;
            internal_sum += internals;
         }
      }
      uint32_t total_leaves = __shfl_sync(0xffffffffu, leaf_sum, 31);
      uint32_t item_child_begin = first_item + (leaf ? leaf_sum : internal_sum + total_leaves) - primitives;
      uint32_t child_base = 0;
      if (lane == 0 && internal_children)
         child_base = atomicAdd(&wide_workspace()->allocated_nodes, internal_children);
      child_base = __shfl_sync(0xffffffffu, child_base, 0);
      if (internal_children && child_base + internal_children > count) {
         if (lane == 0)
            atomicOr(&wide_workspace()->error, NVK_BVH_WIDE_ALLOCATION_ERROR);
         return;
      }
      child_index += child_base;
      if (valid && !leaf) {
         uint64_t child_auxiliary = wide_auxiliary(child_index);
         store32(child_auxiliary, 16, uint32_t(child.child_pair));
         store32(child_auxiliary, 20, index);
         store32(child_auxiliary, 24, child_slot);
         store32(child_auxiliary, 28, 0);
         wide_publish(child_auxiliary, item_child_begin);
      }
      uint32_t reference = wide_pair_reference(root_pair);
      uint64_t source = args.ir + (reference & ~3u);
      float origin[3];
      uint32_t exponents[3];
      for (uint32_t axis = 0; axis < 3; axis++) {
         origin[axis] = subtree_primitives == 0 ? 0.0f : load_float(source, axis * 4);
         float maximum = subtree_primitives == 0 ? 0.0f : load_float(source, 12 + axis * 4);
         float extent = fmaxf(0.0f, __fadd_ru(maximum, -origin[axis]));
         exponents[axis] = min((__float_as_uint(extent) + 0x7fffffu) >> 23, 255u);
      }
      uint64_t destination = args.dst + args.node_offset + uint64_t(index) * NVK_BVH_NODE_SIZE;
      if (lane == 0) {
         uint32_t leaf_type = args.geometry_type == 0 ? (triangle_packing() ? 0u : 1u) :
                              args.geometry_type == 1 ? 0u : 2u;
         uint64_t pointer = leaf_address(first_item) - destination;
         pointer = args.geometry_type == 1 ? (pointer >> 3) & ((uint64_t(1) << 47) - 1) :
                                           pointer & ((uint64_t(1) << 49) - 1);
         if (triangle_packing())
            pointer = first_item;
         store32(destination, 0, (triangle_packing() ? 0u : 0x20u) | (leaf_type << 6) | (exponents[0] << 8) |
                 (exponents[1] << 16) | (exponents[2] << 24));
         for (uint32_t axis = 0; axis < 3; axis++)
            store32(destination, 4 + axis * 4, __float_as_uint(origin[axis]));
         store32(destination, 16, uint32_t(pointer));
         store32(destination, 20, uint32_t(pointer >> 32) |
                 ((parent_slot == NVK_BVH_INVALID_NODE ? 15u : parent_slot) << 28));
         store32(destination, 24, parent == NVK_BVH_INVALID_NODE ? 0 : parent - index);
         store32(destination, 28, child_base - index);
         for (uint32_t axis = 0; axis < 3; axis++)
            store32(auxiliary, 16 + axis * 4, subtree_primitives == 0 ? 0 : load32(source, 12 + axis * 4));
         store32(auxiliary, 28, internal_children | 0xff00u);
         if (args.refit_data_offset) {
            uint64_t persistent = args.dst + args.refit_data_offset + uint64_t(index) * 32;
            store32(persistent, 0, first_item);
            store32(persistent, 4, 0);
            store32(persistent, 8, 0);
            store32(persistent, 12, 0);
            for (uint32_t axis = 0; axis < 3; axis++)
               store32(persistent, 16 + axis * 4, subtree_primitives == 0 ? 0 : load32(source, 12 + axis * 4));
            store32(persistent, 28, internal_children | 0xff00u);
         }
      }
      __syncwarp();
      if (valid) {
         uint32_t child_reference = child.child_pair >= 0 ? wide_pair_reference(uint32_t(child.child_pair)) :
                                    (~uint32_t(child.child_pair) * args.ir_leaf_size);
         uint2 encoded = encode_child(child_reference, origin, exponents);
         uint32_t data = leaf ? primitives | (args.geometry_type != 0 || triangle_packing() ? 0x40u : 0u) : 0;
         if (leaf && args.geometry_type == 0) {
            bool alpha = false;
            for (uint32_t item = 0; item < primitives; item++) {
               uint32_t original = wide_primitive(child.child_pair, item);
               alpha |= (load32(args.ir + uint64_t(original) * args.ir_leaf_size, 64) & 0x80000000u) == 0;
            }
            data |= alpha ? 0x80u : 0;
         }
         encoded.y = (encoded.y & 0x00ffffffu) | (data << 24);
         store32(destination, 32 + child_slot * 8, encoded.x);
         store32(destination, 36 + child_slot * 8, encoded.y);
         if (leaf) {
            reinterpret_cast<uint8_t *>(auxiliary)[4 + child_slot] = uint8_t(primitives);
            if (args.refit_data_offset) {
               uint64_t persistent = args.dst + args.refit_data_offset + uint64_t(index) * 32;
               reinterpret_cast<uint8_t *>(persistent)[4 + child_slot] = uint8_t(primitives);
            }
         }
      } else {
         uint32_t unused_slot = valid_count + __popc((~valid_mask) & lower_mask);
         if (unused_slot < 12) {
            store32(destination, 32 + unused_slot * 8, 0);
            store32(destination, 36 + unused_slot * 8, 0xff);
         }
      }
      uint32_t item_begin = leaf_sum - (leaf ? primitives : 0);
      for (uint32_t base = 0; base < total_leaves; base += 32) {
         uint32_t item = base + lane;
         uint32_t source_lane = 0;
         for (uint32_t step = 16; step; step /= 2) {
            uint32_t begin = __shfl_sync(0xffffffffu, item_begin, source_lane + step);
            if (begin <= item)
               source_lane += step;
         }
         int32_t pair = int32_t(__shfl_sync(0xffffffffu, uint32_t(child.child_pair), source_lane));
         uint32_t local_begin = __shfl_sync(0xffffffffu, item_begin, source_lane);
         if (item < total_leaves) {
            uint32_t original = wide_primitive(pair, item - local_begin);
            uint32_t rank = first_item + item;
            store32(args.leaf_map, original * 4, rank);
            if (args.refit_leaf_map_offset)
               store32(args.dst + args.refit_leaf_map_offset, rank * 4, original);
         }
      }
   }
};

}

extern "C" __global__ __launch_bounds__(256) void
nvk_bvh_encode(nvk_bvh_encode_args args)
{
   nvk_encoder encoder{args};
   uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
   if (args.pass == 0) {
      encoder.prepare_local_mapping(index);
      return;
   }
   if (args.pass == 1) {
      encoder.prepare_global_mapping(index);
      return;
   }
   if (args.pass == 3) {
      encoder.encode_triangle_cooperative(index);
      return;
   }
   if (index < encoder.load32(args.dst, NVK_BVH_NODE_COUNT_OFFSET))
      encoder.encode_node(index);
   if (args.geometry_type != 0 && index < args.leaf_capacity)
      encoder.encode_leaf(index);
}

extern "C" __global__ __launch_bounds__(256) void
nvk_bvh_encode_wide(nvk_bvh_encode_args args)
{
   nvk_encoder encoder{args};
   uint32_t local = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
   uint32_t threads = blockDim.x * blockDim.y * blockDim.z;
   uint32_t index = blockIdx.x * threads + local;
   if (args.pass == 0) {
      encoder.prepare_wide_mapping(index);
      return;
   }
   if (args.pass == 1) {
      uint32_t lane = local & 31u;
      for (;;) {
         uint32_t node_index = 0;
         if (lane == 0)
            node_index = atomicAdd(reinterpret_cast<uint32_t *>(args.prefix_sums + NVK_BVH_WIDE_WORK_COUNTER_OFFSET), 1u);
         node_index = __shfl_sync(0xffffffffu, node_index, 0);
         if (node_index >= max(encoder.wide_workspace()->node_count, 1u))
            return;
         encoder.construct_wide(node_index, lane);
      }
   }
   if (encoder.wide_error())
      return;
   if (args.pass == 3) {
      encoder.encode_triangle_cooperative(index);
      return;
   }
   if (args.geometry_type != 0 && index < args.leaf_capacity)
      encoder.encode_leaf(index);
}
