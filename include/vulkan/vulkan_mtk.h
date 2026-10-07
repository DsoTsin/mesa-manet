#ifndef VULKAN_MTK_H
#define VULKAN_MTK_H

#include "vulkan_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VK_MTK_dynamic_cache_memory 1
#define VK_MTK_DYNAMIC_CACHE_MEMORY_SPEC_VERSION 1
#define VK_MTK_DYNAMIC_CACHE_MEMORY_EXTENSION_NAME "VK_MTK_dynamic_cache_memory"

typedef struct VkDynamicCacheMemoryRegionMTK {
   VkStructureType sType;
   const void *pNext;
   uint32_t cacheKind;
   uint64_t reserved[2];
   VkFlags groupMask;
   uint32_t policy;
} VkDynamicCacheMemoryRegionMTK;

typedef struct VkDynamicCacheMemoryInfoMTK {
   VkStructureType sType;
   const void *pNext;
   uint32_t cacheCount;
   const VkDynamicCacheMemoryRegionMTK *pCaches;
} VkDynamicCacheMemoryInfoMTK;

typedef void (VKAPI_PTR *PFN_vkCmdSetDynamicCacheMemoryMTK)(
   VkCommandBuffer commandBuffer, const VkDynamicCacheMemoryInfoMTK *pInfo);
typedef void (VKAPI_PTR *PFN_vkCmdSetDynamicCacheAccessMTK)(
   VkCommandBuffer commandBuffer);

#ifndef VK_NO_PROTOTYPES
VKAPI_ATTR void VKAPI_CALL vkCmdSetDynamicCacheMemoryMTK(
   VkCommandBuffer commandBuffer, const VkDynamicCacheMemoryInfoMTK *pInfo);
VKAPI_ATTR void VKAPI_CALL vkCmdSetDynamicCacheAccessMTK(
   VkCommandBuffer commandBuffer);
#endif

#ifdef __cplusplus
}
#endif
#endif
