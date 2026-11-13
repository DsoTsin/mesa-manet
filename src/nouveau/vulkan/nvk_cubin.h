/*
 * Copyright © 2025 Autumn Ashton
 * SPDX-License-Identifier: MIT
 */

#ifndef NVK_CUBIN_H
#define NVK_CUBIN_H 1

#include "nvk_private.h"

struct nv_cubin_module;
struct nv_cubin_function;
struct nvk_cmd_buffer;
struct nvk_device;
struct nvk_prepared_dispatch;

struct nvk_cubin_module {
   struct vk_object_base base;

   struct nv_cubin_module *info;
   void *data;
};

struct nvk_cubin_function {
   struct vk_object_base base;

   struct nvk_cubin_module *module;
   const struct nv_cubin_function *info;

   uint32_t upload_size;
   uint64_t upload_addr;

   uint32_t constant2_size;
   uint64_t constant2_addr;

   uint32_t max_warps_per_sm;
};

struct nvk_cubin_launch_info {
   uint32_t grid_dim[3];
   uint32_t block_dim[3];
   uint32_t shared_memory_size;
   uint32_t param_count;
   const void *const *params;
   const void *param_data;
   size_t param_size;
};

VkResult nvk_cubin_module_create(struct nvk_device *dev, const void *data,
                                size_t size, const VkAllocationCallbacks *alloc,
                                struct nvk_cubin_module **out);
void nvk_cubin_module_destroy(struct nvk_device *dev,
                              struct nvk_cubin_module *module,
                              const VkAllocationCallbacks *alloc);
VkResult nvk_cubin_function_create(struct nvk_device *dev,
                                  struct nvk_cubin_module *module,
                                  const char *name,
                                  const VkAllocationCallbacks *alloc,
                                  struct nvk_cubin_function **out);
void nvk_cubin_function_destroy(struct nvk_device *dev,
                                struct nvk_cubin_function *function,
                                const VkAllocationCallbacks *alloc);
void nvk_cmd_cubin_launch(struct nvk_cmd_buffer *cmd,
                          const struct nvk_cubin_function *function,
                          const struct nvk_cubin_launch_info *info);
VkResult nvk_cmd_cubin_prepare(struct nvk_cmd_buffer *cmd,
                               const struct nvk_cubin_function *function,
                               const struct nvk_cubin_launch_info *info,
                               struct nvk_prepared_dispatch *out);

VK_DEFINE_NONDISP_HANDLE_CASTS(nvk_cubin_module, base, VkCuModuleNVX,
                               VK_OBJECT_TYPE_CU_MODULE_NVX)

VK_DEFINE_NONDISP_HANDLE_CASTS(nvk_cubin_function, base, VkCuFunctionNVX,
                               VK_OBJECT_TYPE_CU_FUNCTION_NVX)

#endif
