#ifndef NVK_BVH_OMM_H
#define NVK_BVH_OMM_H

#include <stdint.h>

#define NVK_OMM_HEADER_SIZE 128u
#define NVK_OMM_ADDRESS_MASK ((1ull << 49) - 1ull)
#define NVK_OMM_META_LEVEL_SHIFT 49u
#define NVK_OMM_META_FLAGS_SHIFT 53u
#define NVK_OMM_META_TWO_STATE_SHIFT 57u
#define NVK_OMM_META_DIRECT_SHIFT 58u
#define NVK_OMM_META_GEOMETRY_OPAQUE_SHIFT 59u
#define NVK_OMM_META_ATTACHED_SHIFT 60u
#define NVK_OMM_TWO_STATE (1ull << NVK_OMM_META_TWO_STATE_SHIFT)
#define NVK_OMM_DIRECT (1ull << NVK_OMM_META_DIRECT_SHIFT)
#define NVK_OMM_GEOMETRY_OPAQUE (1ull << NVK_OMM_META_GEOMETRY_OPAQUE_SHIFT)
#define NVK_OMM_ATTACHED (1ull << NVK_OMM_META_ATTACHED_SHIFT)
#define NVK_OMM_DEFAULT_HULL 0x41f07c1fu
#define NVK_OMM_EMPTY_HULL 0xbfffffffu
#define NVK_OMM_HULL_FULL (1u << 30)
#define NVK_OMM_HULL_EMPTY (1u << 31)
#define NVK_OMM_SERIALIZED_HEADER_SIZE 56u
#define NVK_OMM_BUILD_THREADS 128u
#define NVK_OMM_SCAN_THREADS 256u
#define NVK_OMM_COPY_THREADS 128u
#define NVK_OMM_COPY_CLONE 0u
#define NVK_OMM_COPY_SERIALIZE 1u
#define NVK_OMM_COPY_DESERIALIZE 2u
#define NVK_OMM_COPY_RELOCATE 3u
#define NVK_OMM_COPY_PROPERTY 4u
#define NVK_OMM_PROPERTY_SIZE 0u
#define NVK_OMM_PROPERTY_SERIALIZED_SIZE 1u
#define NVK_OMM_PROPERTY_COMPACTED_SIZE 2u

#ifdef __CUDACC__
#define NVK_OMM_INLINE static __host__ __device__ __forceinline__
#else
#define NVK_OMM_INLINE static inline
#endif

NVK_OMM_INLINE uint64_t nvk_omm_metadata_offset(uint32_t index)
{
   return 128ull + 256ull * (index / 16u) + 8ull * (index % 16u);
}

NVK_OMM_INLINE uint64_t nvk_omm_auxiliary_offset(uint32_t index)
{
   return nvk_omm_metadata_offset(index) + 128ull;
}

NVK_OMM_INLINE uint32_t nvk_omm_group_count(uint32_t count)
{
   return (uint32_t)(((uint64_t)count + 31ull) / 32ull);
}

NVK_OMM_INLINE uint64_t nvk_omm_data_offset(uint32_t count)
{
   return 128ull + 2ull * ((8ull * count + 127ull) & ~127ull);
}

NVK_OMM_INLINE uint64_t nvk_omm_map_size(uint32_t level, uint32_t format)
{
   return ((1ull << (2u * level)) * (format == 1u ? 1ull : 2ull) + 7ull) / 8ull;
}

NVK_OMM_INLINE uint64_t nvk_omm_array_size(uint32_t count, uint64_t input_bytes)
{
   return (nvk_omm_data_offset(count) + input_bytes + 128ull * nvk_omm_group_count(count) + 127ull) & ~127ull;
}

NVK_OMM_INLINE uint64_t nvk_omm_scan_sums_offset(uint32_t count)
{
   return (16ull * nvk_omm_group_count(count) + 127ull) & ~127ull;
}

NVK_OMM_INLINE uint64_t nvk_omm_scratch_size(uint32_t count)
{
   uint64_t blocks = (nvk_omm_group_count(count) + 255ull) / 256ull;
   return nvk_omm_scan_sums_offset(count) + ((16ull * blocks + 127ull) & ~127ull) + 128ull;
}

NVK_OMM_INLINE uint64_t nvk_omm_special_metadata(int32_t index)
{
   return NVK_OMM_DIRECT | ((uint32_t)(~index) & 3u);
}

struct nvk_bvh_omm_build_args {
   uint64_t dst;
   uint64_t data;
   uint64_t triangles;
   uint64_t scratch;
   uint64_t size;
   uint64_t data_size;
   uint64_t stride;
   uint32_t count;
   uint32_t flags;
   uint32_t uuid[8];
};

struct nvk_bvh_omm_copy_args {
   uint64_t src;
   uint64_t dst;
   uint64_t available;
   uint64_t size;
   uint32_t mode;
   uint32_t field;
};

struct nvk_bvh_omm_descriptor {
   uint32_t data_offset;
   uint16_t subdivision_level;
   uint16_t format;
};

struct nvk_bvh_omm_group {
   uint64_t prefix_bytes;
   uint32_t size_bytes;
   uint32_t reserved;
};

#undef NVK_OMM_INLINE
#endif
