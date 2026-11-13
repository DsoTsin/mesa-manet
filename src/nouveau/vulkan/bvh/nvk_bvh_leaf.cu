#include "nvk_bvh_leaf.h"
#include "nvk_bvh.h"
#include "nvk_bvh_omm.h"
#include <cuda_runtime.h>
#include <vulkan/vulkan_core.h>
#include <float.h>
#include <math.h>
#include <stddef.h>

static_assert(sizeof(nvk_bvh_leaf_geometry) == 56, "leaf geometry size");
static_assert(offsetof(nvk_bvh_leaf_geometry, indirect_range) == 48, "leaf indirect offset");
static_assert(sizeof(nvk_bvh_leaf_args) == 136, "leaf parameter size");
static_assert(offsetof(nvk_bvh_leaf_args, micromap_array) == 96, "leaf micromap offset");
static_assert(offsetof(nvk_bvh_leaf_args, geom_data) == 24, "leaf geometry offset");
static_assert(offsetof(nvk_bvh_leaf_args, primitive_count) == 80, "leaf primitive count offset");
static_assert(offsetof(nvk_bvh_leaf_args, ir_leaf_size) == 84, "leaf stride offset");
static_assert(offsetof(nvk_bvh_leaf_args, build_flags) == 88, "leaf flags offset");

struct nvk_leaf_bounds {
   float minimum[3];
   float maximum[3];
};

struct nvk_leaf_triangle {
   float vertex[3][4];
   uint32_t index[3];
};

__device__ __forceinline__ uint32_t
nvk_leaf_load(uint64_t address, uint32_t offset)
{
   return *reinterpret_cast<const uint32_t *>(address + offset);
}

__device__ __forceinline__ uint64_t
nvk_leaf_load64(uint64_t address, uint32_t offset)
{
   return uint64_t(nvk_leaf_load(address, offset)) |
          (uint64_t(nvk_leaf_load(address, offset + 4)) << 32);
}

__device__ __forceinline__ void
nvk_leaf_store(uint64_t address, uint32_t offset, uint32_t value)
{
   *reinterpret_cast<uint32_t *>(address + offset) = value;
}

__device__ __forceinline__ float
nvk_leaf_add(float a, float b)
{
   float value;
   asm("add.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(a), "f"(b));
   return value;
}

__device__ __forceinline__ float
nvk_leaf_mul(float a, float b)
{
   float value;
   asm("mul.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(a), "f"(b));
   return value;
}

__device__ __forceinline__ float
nvk_leaf_min(float a, float b)
{
   float value;
   asm("min.f32 %0, %1, %2;" : "=f"(value) : "f"(a), "f"(b));
   return value;
}

__device__ __forceinline__ float
nvk_leaf_max(float a, float b)
{
   float value;
   asm("max.f32 %0, %1, %2;" : "=f"(value) : "f"(a), "f"(b));
   return value;
}

__device__ __forceinline__ float
nvk_leaf_half(uint16_t value)
{
   uint32_t sign = uint32_t(value & 0x8000u) << 16;
   uint32_t exponent = (value >> 10) & 31u;
   uint32_t fraction = value & 1023u;
   if (exponent == 0) {
      if (fraction == 0)
         return __uint_as_float(sign);
      const uint32_t shift = __clz(fraction) - 21;
      exponent = 113u - shift;
      fraction = (fraction << shift) & 1023u;
   } else {
      exponent = exponent == 31 ? 255 : exponent + 112;
   }
   return __uint_as_float(sign | (exponent << 23) | (fraction << 13));
}

__device__ __forceinline__ void
nvk_leaf_vertex(uint64_t address, uint32_t format, float vertex[4])
{
   vertex[0] = vertex[1] = vertex[2] = 0;
   vertex[3] = 1;
   uint32_t components = 3;
   if (format == VK_FORMAT_R32G32_SFLOAT || format == VK_FORMAT_R16G16_SFLOAT ||
       format == VK_FORMAT_R16G16_SNORM || format == VK_FORMAT_R16G16_UNORM ||
       format == VK_FORMAT_R8G8_SNORM || format == VK_FORMAT_R8G8_UNORM)
      components = 2;
   for (uint32_t axis = 0; axis < components; axis++) {
      switch (format) {
      case VK_FORMAT_R32G32_SFLOAT:
      case VK_FORMAT_R32G32B32_SFLOAT:
      case VK_FORMAT_R32G32B32A32_SFLOAT:
         vertex[axis] = __uint_as_float(nvk_leaf_load(address, axis * 4));
         break;
      case VK_FORMAT_R16G16_SFLOAT:
      case VK_FORMAT_R16G16B16_SFLOAT:
      case VK_FORMAT_R16G16B16A16_SFLOAT:
         vertex[axis] = nvk_leaf_half(reinterpret_cast<const uint16_t *>(address)[axis]);
         break;
      case VK_FORMAT_R16G16_SNORM:
      case VK_FORMAT_R16G16B16A16_SNORM:
         vertex[axis] = nvk_leaf_max(-1.0f, float(reinterpret_cast<const int16_t *>(address)[axis]) / 32767.0f);
         break;
      case VK_FORMAT_R16G16_UNORM:
      case VK_FORMAT_R16G16B16A16_UNORM:
         vertex[axis] = float(reinterpret_cast<const uint16_t *>(address)[axis]) / 65535.0f;
         break;
      case VK_FORMAT_R8G8_SNORM:
      case VK_FORMAT_R8G8B8A8_SNORM:
         vertex[axis] = nvk_leaf_max(-1.0f, float(reinterpret_cast<const int8_t *>(address)[axis]) / 127.0f);
         break;
      case VK_FORMAT_R8G8_UNORM:
      case VK_FORMAT_R8G8B8A8_UNORM:
         vertex[axis] = float(reinterpret_cast<const uint8_t *>(address)[axis]) / 255.0f;
         break;
      case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
         vertex[axis] = float((nvk_leaf_load(address, 0) >> (axis * 10)) & 1023u) / 1023.0f;
         break;
      }
   }
}

__device__ __forceinline__ uint32_t
nvk_leaf_index(const nvk_bvh_leaf_geometry &geometry, uint32_t index)
{
   switch (geometry.index_format) {
   case VK_INDEX_TYPE_UINT8_EXT:
      return reinterpret_cast<const uint8_t *>(geometry.indices)[index];
   case VK_INDEX_TYPE_UINT16:
      return reinterpret_cast<const uint16_t *>(geometry.indices)[index];
   case VK_INDEX_TYPE_UINT32:
      return reinterpret_cast<const uint32_t *>(geometry.indices)[index];
   default:
      return index;
   }
}

__device__ __forceinline__ nvk_leaf_triangle
nvk_leaf_load_triangle(const nvk_bvh_leaf_geometry &geometry, uint32_t primitive)
{
   nvk_leaf_triangle triangle;
#pragma unroll
   for (uint32_t vertex = 0; vertex < 3; vertex++) {
      triangle.index[vertex] = nvk_leaf_index(geometry, primitive * 3 + vertex);
      nvk_leaf_vertex(geometry.data + uint64_t(triangle.index[vertex]) * geometry.stride,
                       geometry.vertex_format, triangle.vertex[vertex]);
      if (geometry.transform != 0) {
         float result[4];
#pragma unroll
         for (uint32_t row = 0; row < 4; row++) {
            float value = 0;
#pragma unroll
            for (uint32_t column = 0; column < 4; column++) {
               const float matrix = row < 3 ? __uint_as_float(nvk_leaf_load(geometry.transform, (row * 4 + column) * 4))
                                            : (column == 3 ? 1.0f : 0.0f);
               const float product = nvk_leaf_mul(matrix, triangle.vertex[vertex][column]);
               value = column == 0 ? product : nvk_leaf_add(value, product);
            }
            result[row] = value;
         }
#pragma unroll
         for (uint32_t axis = 0; axis < 4; axis++)
            triangle.vertex[vertex][axis] = result[axis];
      }
   }
   return triangle;
}

__device__ __forceinline__ bool
nvk_leaf_triangle_nan(const nvk_leaf_triangle &triangle)
{
   bool invalid = false;
#pragma unroll
   for (uint32_t vertex = 0; vertex < 3; vertex++) {
#pragma unroll
      for (uint32_t axis = 0; axis < 4; axis++)
         invalid |= isnan(triangle.vertex[vertex][axis]);
   }
   return invalid;
}

__device__ __forceinline__ uint32_t
nvk_leaf_pair(const nvk_leaf_triangle &a, const nvk_leaf_triangle &b)
{
#pragma unroll
   for (uint32_t e0 = 0; e0 < 3; e0++) {
#pragma unroll
      for (uint32_t e1 = 0; e1 < 3; e1++) {
         bool matches = true;
#pragma unroll
         for (uint32_t axis = 0; axis < 3; axis++) {
            matches &= a.vertex[e0][axis] == b.vertex[(e1 + 1) % 3][axis];
            matches &= a.vertex[(e0 + 1) % 3][axis] == b.vertex[e1][axis];
         }
         if (matches)
            return e0 | (e1 << 2);
      }
   }
   return 0xffffffffu;
}

__device__ __forceinline__ void
nvk_leaf_write_bounds(uint64_t output, const nvk_leaf_bounds &bounds)
{
#pragma unroll
   for (uint32_t axis = 0; axis < 3; axis++) {
      nvk_leaf_store(output, axis * 4, __float_as_uint(bounds.minimum[axis]));
      nvk_leaf_store(output, 12 + axis * 4, __float_as_uint(bounds.maximum[axis]));
   }
}

__device__ __forceinline__ bool
nvk_leaf_triangle_node(const nvk_bvh_leaf_args &args, const nvk_bvh_leaf_geometry &geometry,
                        uint32_t primitive, uint64_t output, nvk_leaf_bounds &bounds)
{
   nvk_leaf_triangle triangle = nvk_leaf_load_triangle(geometry, primitive);
   const bool valid = !nvk_leaf_triangle_nan(triangle);
   if (!valid) {
#pragma unroll
      for (uint32_t vertex = 0; vertex < 3; vertex++) {
#pragma unroll
         for (uint32_t axis = 0; axis < 4; axis++)
            triangle.vertex[vertex][axis] = 0;
      }
   }
#pragma unroll
   for (uint32_t axis = 0; axis < 3; axis++) {
      bounds.minimum[axis] = __uint_as_float(0x7f800000u);
      bounds.maximum[axis] = -bounds.minimum[axis];
#pragma unroll
      for (uint32_t vertex = 0; vertex < 3; vertex++) {
         const float value = triangle.vertex[vertex][axis];
         nvk_leaf_store(output, 24 + (vertex * 3 + axis) * 4, __float_as_uint(value));
         bounds.minimum[axis] = nvk_leaf_min(bounds.minimum[axis], value);
         bounds.maximum[axis] = nvk_leaf_max(bounds.maximum[axis], value);
      }
   }
   nvk_leaf_write_bounds(output, bounds);
   nvk_leaf_store(output, 60, primitive);
   nvk_leaf_store(output, 64, geometry.geometry_id);
   const bool quads = (args.build_flags & NVK_BVH_LEAF_HAS_QUADS) != 0;
   if (quads)
      nvk_leaf_store(output, 68, 0xffffffffu);
   if (args.build_flags & NVK_BVH_LEAF_STORE_VERTEX_IDS) {
      const uint32_t offset = 68 + (quads ? 16 : 0);
#pragma unroll
      for (uint32_t vertex = 0; vertex < 3; vertex++) {
         const uint64_t identity = geometry.data + uint64_t(triangle.index[vertex]) * geometry.stride;
         nvk_leaf_store(output, offset + vertex * 8, uint32_t(identity));
         nvk_leaf_store(output, offset + vertex * 8 + 4, uint32_t(identity >> 32));
      }
   }
   return valid;
}

__device__ __forceinline__ bool
nvk_leaf_aabb_node(uint64_t input, uint64_t output, uint32_t primitive,
                    uint32_t geometry, nvk_leaf_bounds &bounds)
{
   bool valid = true;
#pragma unroll
   for (uint32_t axis = 0; axis < 3; axis++) {
      bounds.minimum[axis] = __uint_as_float(nvk_leaf_load(input, axis * 4));
      bounds.maximum[axis] = __uint_as_float(nvk_leaf_load(input, 12 + axis * 4));
      valid &= !isnan(bounds.minimum[axis]) && !isnan(bounds.maximum[axis]);
   }
   if (!valid) {
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++)
         bounds.minimum[axis] = bounds.maximum[axis] = 0;
   }
   nvk_leaf_write_bounds(output, bounds);
   nvk_leaf_store(output, 24, primitive);
   nvk_leaf_store(output, 28, geometry);
   return valid;
}

__device__ __forceinline__ bool
nvk_leaf_instance_node(const nvk_bvh_leaf_args &args, uint64_t input,
                        uint64_t output, uint32_t primitive, nvk_leaf_bounds &bounds)
{
   const uint32_t custom = nvk_leaf_load(input, 48);
   const uint32_t sbt = nvk_leaf_load(input, 52);
   const uint64_t blas = nvk_leaf_load64(input, 56);
   bool valid = blas != 0 && custom >= 0x01000000u;
   if (!valid && !(args.build_flags & NVK_BVH_LEAF_ALWAYS_ACTIVE))
      return false;
   nvk_leaf_store(output, 24, uint32_t(blas));
   nvk_leaf_store(output, 28, uint32_t(blas >> 32));
   nvk_leaf_store(output, 32, custom);
   nvk_leaf_store(output, 36, sbt);
   nvk_leaf_store(output, 88, primitive);
   if (blas == 0) {
#pragma unroll
      for (uint32_t row = 0; row < 3; row++) {
         bounds.minimum[row] = bounds.maximum[row] = 0;
#pragma unroll
         for (uint32_t column = 0; column < 4; column++)
            nvk_leaf_store(output, 40 + (row * 4 + column) * 4, __float_as_uint(row == column ? 1.0f : 0.0f));
      }
      nvk_leaf_store(output, 92, 0);
      nvk_leaf_write_bounds(output, bounds);
      return false;
   }
   float center[3], extent[3];
#pragma unroll
   for (uint32_t axis = 0; axis < 3; axis++) {
      const float lo = __uint_as_float(nvk_leaf_load(blas, NVK_BVH_BOUNDS_OFFSET + axis * 4));
      const float hi = __uint_as_float(nvk_leaf_load(blas, NVK_BVH_BOUNDS_OFFSET + 12 + axis * 4));
      center[axis] = nvk_leaf_mul(nvk_leaf_add(lo, hi), 0.5f);
      extent[axis] = nvk_leaf_mul(nvk_leaf_add(hi, -lo), 0.5f);
   }
#pragma unroll
   for (uint32_t row = 0; row < 3; row++) {
      float transformed_center = 0;
      float transformed_extent = 0;
#pragma unroll
      for (uint32_t column = 0; column < 4; column++) {
         const uint32_t bits = nvk_leaf_load(input, (row * 4 + column) * 4);
         const float matrix = __uint_as_float(bits);
         nvk_leaf_store(output, 40 + (row * 4 + column) * 4, bits);
         const float product = nvk_leaf_mul(column < 3 ? center[column] : 1.0f, matrix);
         transformed_center = column == 0 ? product : nvk_leaf_add(transformed_center, product);
         if (column < 3) {
            const float magnitude = __uint_as_float(bits & 0x7fffffffu);
            const float extent_product = nvk_leaf_mul(extent[column], magnitude);
            transformed_extent = column == 0 ? extent_product : nvk_leaf_add(transformed_extent, extent_product);
         }
      }
      bounds.minimum[row] = nvk_leaf_add(transformed_center, -transformed_extent);
      bounds.maximum[row] = nvk_leaf_add(transformed_center, transformed_extent);
   }
   bool nan = false;
#pragma unroll
   for (uint32_t axis = 0; axis < 3; axis++)
      nan |= isnan(bounds.minimum[axis]) || isnan(bounds.maximum[axis]);
   if (nan) {
      valid = false;
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++)
         bounds.minimum[axis] = bounds.maximum[axis] = 0;
   }
   nvk_leaf_write_bounds(output, bounds);
   if (args.build_flags & NVK_BVH_LEAF_PROPAGATE_CULL_FLAGS) {
      uint32_t flags;
      if (sbt & (VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR << 24))
         flags = 1;
      else if (sbt & (VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR << 24))
         flags = 2;
      else
         flags = nvk_leaf_load(blas, NVK_BVH_ROOT_FLAGS_OFFSET);
      flags |= (~(custom >> 24)) << 2;
      nvk_leaf_store(output, 92, flags);
   }
   return valid;
}

__device__ __forceinline__ int32_t
nvk_leaf_ordered(float value)
{
   const uint32_t bits = __float_as_uint(value);
   return int32_t((bits & 0x80000000u) ? 0x80000000u - bits : bits);
}

__device__ __forceinline__ void
nvk_leaf_opacity_micromap(const nvk_bvh_leaf_args &args, uint32_t local,
                          uint32_t primitive)
{
   if (args.output_metadata == 0 || (args.omm_flags & NVK_BVH_LEAF_OMM_PRESERVE))
      return;
   uint64_t metadata = 0;
   if (args.omm_flags & NVK_BVH_LEAF_OMM_ATTACHED) {
      uint32_t index = local;
      if (args.index_type == VK_INDEX_TYPE_UINT16) {
         index = *reinterpret_cast<const uint16_t *>(args.index_buffer +
                                                      uint64_t(local) * args.index_stride);
         if (index >= 0xfffcu)
            index |= 0xffff0000u;
      } else if (args.index_type == VK_INDEX_TYPE_UINT32) {
         index = nvk_leaf_load(args.index_buffer + uint64_t(local) * args.index_stride, 0);
      }
      if (index >= 0xfffffffcu)
         metadata = nvk_omm_special_metadata(int32_t(index));
      else
         metadata = nvk_leaf_load64(args.micromap_array +
                                     nvk_omm_metadata_offset(index + args.base_triangle), 0);
      metadata |= NVK_OMM_ATTACHED;
   }
   if (args.geom_data.geometry_id & 0x80000000u)
      metadata |= NVK_OMM_GEOMETRY_OPAQUE;
   const uint64_t output = args.output_metadata + uint64_t(primitive) * 8;
   nvk_leaf_store(output, 0, uint32_t(metadata));
   nvk_leaf_store(output, 4, uint32_t(metadata >> 32));
}

extern "C" __global__ __launch_bounds__(NVK_BVH_LEAF_THREADS) void
nvk_bvh_leaf(nvk_bvh_leaf_args args)
{
   const uint32_t global = blockIdx.x * blockDim.x + threadIdx.x;
   const uint32_t primitive = args.geom_data.first_id + global;
   nvk_bvh_leaf_geometry geometry = args.geom_data;
   uint32_t count = args.primitive_count;
   if (geometry.indirect_range != 0) {
      const uint32_t indirect_count = nvk_leaf_load(geometry.indirect_range, 0);
      const uint32_t offset = nvk_leaf_load(geometry.indirect_range, 4);
      if (geometry.geometry_type == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
         geometry.data += uint64_t(nvk_leaf_load(geometry.indirect_range, 8)) * geometry.stride;
         if (geometry.index_format == VK_INDEX_TYPE_NONE_KHR)
            geometry.data += offset;
         else
            geometry.indices += offset;
         if (geometry.transform != 0)
            geometry.transform += nvk_leaf_load(geometry.indirect_range, 12);
      } else {
         geometry.data += offset;
      }
      count = min(count, indirect_count);
   }
   nvk_leaf_bounds bounds;
   bool active = false;
   if (global < count) {
      const uint32_t offset = primitive * args.ir_leaf_size;
      const uint64_t output = args.ir + offset;
      uint32_t type;
      if (geometry.geometry_type == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
         type = 0;
         active = nvk_leaf_triangle_node(args, geometry, global, output, bounds);
         nvk_leaf_opacity_micromap(args, global, primitive);
         if ((args.build_flags & NVK_BVH_LEAF_EARLY_PAIRS) && active && (global | 1u) < count) {
            const nvk_leaf_triangle a = nvk_leaf_load_triangle(geometry, global & ~1u);
            const nvk_leaf_triangle b = nvk_leaf_load_triangle(geometry, global | 1u);
            const uint32_t edges = nvk_leaf_triangle_nan(a) || nvk_leaf_triangle_nan(b)
                                     ? 0xffffffffu : nvk_leaf_pair(a, b);
            if (edges != 0xffffffffu) {
               if (global & 1u) {
                  active = false;
               } else {
                  const uint32_t vertex = ((edges >> 2) + 2) % 3;
                  nvk_leaf_store(output, 68, (global + 1) | (edges << 28));
#pragma unroll
                  for (uint32_t axis = 0; axis < 3; axis++) {
                     const float value = vertex == 0 ? b.vertex[0][axis] :
                                         vertex == 1 ? b.vertex[1][axis] : b.vertex[2][axis];
                     nvk_leaf_store(output, 72 + axis * 4, __float_as_uint(value));
                     bounds.minimum[axis] = nvk_leaf_min(bounds.minimum[axis], value);
                     bounds.maximum[axis] = nvk_leaf_max(bounds.maximum[axis], value);
                  }
                  nvk_leaf_write_bounds(output, bounds);
               }
            }
         }
      } else {
         uint64_t input = geometry.data + uint64_t(global) * geometry.stride;
         if (geometry.geometry_type == VK_GEOMETRY_TYPE_AABBS_KHR) {
            type = 3;
            active = nvk_leaf_aabb_node(input, output, global, geometry.geometry_id, bounds);
         } else {
            type = 2;
            if (geometry.stride == 8)
               input = nvk_leaf_load64(input, 0);
            active = nvk_leaf_instance_node(args, input, output, global, bounds);
         }
      }
      if ((args.build_flags & NVK_BVH_LEAF_INACTIVE_BOUNDS) && !active) {
#pragma unroll
         for (uint32_t axis = 0; axis < 3; axis++) {
            bounds.minimum[axis] = FLT_MAX;
            bounds.maximum[axis] = -FLT_MAX;
         }
         nvk_leaf_write_bounds(output, bounds);
      }
      if (args.build_flags & NVK_BVH_LEAF_ALWAYS_ACTIVE)
         active = true;
      nvk_leaf_store(args.ids, primitive * 4, active ? (offset | type) : 0xffffffffu);
   } else if (global < args.primitive_count) {
      nvk_leaf_store(args.ids, primitive * 4, 0xffffffffu);
   }
   const uint32_t mask = __ballot_sync(0xffffffffu, active);
   int32_t minimum[3], maximum[3];
#pragma unroll
   for (uint32_t axis = 0; axis < 3; axis++) {
      minimum[axis] = active ? nvk_leaf_ordered(bounds.minimum[axis]) : 0x7fffffff;
      maximum[axis] = active ? nvk_leaf_ordered(bounds.maximum[axis]) : int32_t(0x80000000u);
   }
#pragma unroll
   for (uint32_t delta = 1; delta < 32; delta <<= 1) {
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++) {
         minimum[axis] = min(minimum[axis], __shfl_xor_sync(0xffffffffu, minimum[axis], delta));
         maximum[axis] = max(maximum[axis], __shfl_xor_sync(0xffffffffu, maximum[axis], delta));
      }
   }
   if ((threadIdx.x & 31u) == 0 && mask != 0) {
      atomicAdd(reinterpret_cast<uint32_t *>(args.header + 24), __popc(mask));
#pragma unroll
      for (uint32_t axis = 0; axis < 3; axis++) {
         atomicMin(reinterpret_cast<int32_t *>(args.header + axis * 4), minimum[axis]);
         atomicMax(reinterpret_cast<int32_t *>(args.header + 12 + axis * 4), maximum[axis]);
      }
   }
}
