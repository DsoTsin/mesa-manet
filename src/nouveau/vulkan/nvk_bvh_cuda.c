#include "nvk_bvh_cuda.h"
#include "nvk_cmd_buffer.h"
#include "nv_push_cl90c0.h"

void
nvk_bvh_meta_begin(struct nvk_cmd_buffer *cmd, struct nvk_compute_state *saved)
{
   assert(cmd->state.cs.descriptors.flush_root == NULL);
   *saved = cmd->state.cs;
   nvk_cmd_invalidate_compute_state(cmd);
   struct nv_push *push = nvk_cmd_buffer_push(cmd, 1);
   P_IMMD_WORD(push, NV90C0, SET_RENDER_ENABLE_OVERRIDE, MODE_ALWAYS_RENDER);
}

void
nvk_bvh_meta_end(struct nvk_cmd_buffer *cmd, const struct nvk_compute_state *saved)
{
   struct nv_push *push = nvk_cmd_buffer_push(cmd, 1);
   P_IMMD_WORD(push, NV90C0, SET_RENDER_ENABLE_OVERRIDE, MODE_USE_RENDER_ENABLE);
   nvk_descriptor_state_fini(cmd, &cmd->state.cs.descriptors);
   cmd->state.cs = *saved;
}

#ifdef NVK_BVH_CUDA_ENABLED
#include "nvk_cmd_buffer.h"
#include "nvk_cubin.h"
#include "nvk_device.h"
#include "vk_meta.h"

static const uint8_t blackwell_cubin[] = {
#include "nvk_bvh_blackwell.cubin.h"
};

VkResult
nvk_bvh_cuda_launch(struct nvk_cmd_buffer *cmd,
                     const char *name, const void *args, size_t args_size,
                     uint32_t grid_x, uint32_t block_x)
{
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const struct {
      enum vk_meta_object_key_type type;
      uint32_t module;
   } module_key = { VK_META_OBJECT_KEY_DRIVER_OFFSET + 32, 0 };
   struct {
      enum vk_meta_object_key_type type;
      uint32_t module;
      char name[64];
   } function_key = { VK_META_OBJECT_KEY_DRIVER_OFFSET + 33, 0, {0} };
   assert(strlen(name) < sizeof(function_key.name));
   memcpy(function_key.name, name, strlen(name));
   VkCuFunctionNVX function_handle = (VkCuFunctionNVX)vk_meta_lookup_object(
      &dev->meta, VK_OBJECT_TYPE_CU_FUNCTION_NVX,
      &function_key, sizeof(function_key));
   if (function_handle == VK_NULL_HANDLE) {
      VkCuModuleNVX module_handle = (VkCuModuleNVX)vk_meta_lookup_object(
         &dev->meta, VK_OBJECT_TYPE_CU_MODULE_NVX,
         &module_key, sizeof(module_key));
      if (module_handle == VK_NULL_HANDLE) {
         const void *data = blackwell_cubin;
         const size_t size = sizeof(blackwell_cubin);
         struct nvk_cubin_module *module;
         VkResult result = nvk_cubin_module_create(dev, data, size, NULL, &module);
         if (result != VK_SUCCESS)
            return result;
         module_handle = (VkCuModuleNVX)vk_meta_cache_object(&dev->vk, &dev->meta,
            &module_key, sizeof(module_key), VK_OBJECT_TYPE_CU_MODULE_NVX,
            (uint64_t)nvk_cubin_module_to_handle(module));
      }
      struct nvk_cubin_function *function;
      VkResult result = nvk_cubin_function_create(dev,
         nvk_cubin_module_from_handle(module_handle), name, NULL, &function);
      if (result != VK_SUCCESS)
         return result;
      function_handle = (VkCuFunctionNVX)vk_meta_cache_object(&dev->vk, &dev->meta,
         &function_key, sizeof(function_key), VK_OBJECT_TYPE_CU_FUNCTION_NVX,
         (uint64_t)nvk_cubin_function_to_handle(function));
   }
   const struct nvk_cubin_launch_info launch = {
      .grid_dim = { grid_x, 1, 1 },
      .block_dim = { block_x, 1, 1 },
      .param_data = args,
      .param_size = args_size,
   };
   nvk_cmd_cubin_launch(cmd, nvk_cubin_function_from_handle(function_handle), &launch);
   return VK_SUCCESS;
}
#else
VkResult
nvk_bvh_cuda_launch(struct nvk_cmd_buffer *cmd,
                    const char *name, const void *args, size_t args_size,
                    uint32_t grid_x, uint32_t block_x)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}
#endif
