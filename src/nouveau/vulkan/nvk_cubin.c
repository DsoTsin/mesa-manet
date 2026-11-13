/*
 * Copyright © 2025 Autumn Ashton
 * SPDX-License-Identifier: MIT
 */
#include "nvk_cubin.h"

#include "nvk_cmd_buffer.h"
#include "nvk_cmd_dispatch.h"
#include "nvk_device.h"
#include "nvk_entrypoints.h"
#include "nvk_instance.h"

#include "nouveau/cubin/nv_cubin.h"
#include "nouveau/cubin/nv_cubin_defs.h"
#include "nouveau/cubin/nv_fatbin.h"
#include "clcdc0.h"

#include "util/u_process.h"
#include "util/simple_mtx.h"

#ifdef _WIN32
static simple_mtx_t nvk_cubin_elf_mutex = SIMPLE_MTX_INITIALIZER;
#endif

static void
nvk_cubin_dump(const char *extension, const void *data, size_t size)
{
   static int dump_idx = 0;

   char path[2048];
   const char *directory = "/tmp";
#ifdef _WIN32
   directory = getenv("TEMP");
   if (!directory)
      directory = ".";
#endif
   snprintf(path, sizeof(path), "%s/%s.%d.%s", directory,
            util_get_process_name(), (int)p_atomic_inc_return(&dump_idx), extension);
   FILE *dump_file = fopen(path, "wb");
   if (!dump_file) {
      mesa_loge("nvk_cubin_dump: Failed to dump to: %s\n", path);
      return;
   }

   fwrite(data, 1, size, dump_file);
   fclose(dump_file);
}

/* nvk_cubin_module */

static VkResult
nvk_cubin_module_init(struct nvk_device *dev, struct nvk_cubin_module *module,
                      const void *module_data, size_t module_size,
                      const VkAllocationCallbacks *pAllocator)
{
   const struct nvk_physical_device *pdev = nvk_device_physical(dev);
   const struct nvk_instance *instance = nvk_physical_device_instance(pdev);
   const void *data;
   size_t size;
   bool backwards_compat =
      instance->experimental_flags & NVK_EXPERIMENTAL_DLSS_BACK_COMPAT;

   struct nv_fatbin fatbin;
   if (nv_fatbin_init(&fatbin, module_data, module_size)) {
      /* Must be a fatbin. */
#ifndef NDEBUG
      if (debug_get_bool_option("NVK_CUBIN_DUMP", false))
         nvk_cubin_dump("fatbin", module_data, module_size);
#endif

      if (!nv_fatbin_get_bytecode(&fatbin, pdev->info.sm,
                                  backwards_compat, &data, &size))
         return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                         "Fatbin does not have compatible bytecode");
   } else {
      /* Must be a cubin. */
      data = module_data;
      size = module_size;
   }

#ifndef NDEBUG
   if (debug_get_bool_option("NVK_CUBIN_DUMP", false))
      nvk_cubin_dump("cubin", data, size);
#endif

   module->info =
      vk_zalloc2(&dev->vk.alloc, pAllocator, sizeof(struct nv_cubin_module), 8,
                 VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!module->info)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   module->data = vk_alloc2(&dev->vk.alloc, pAllocator, size, 8,
                            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!module->data)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   memcpy(module->data, data, size);

#ifdef _WIN32
   simple_mtx_lock(&nvk_cubin_elf_mutex);
#endif
   const bool parsed = nv_cubin_module_init(module->info, module->data, size);
#ifdef _WIN32
   simple_mtx_unlock(&nvk_cubin_elf_mutex);
#endif
   if (!parsed)
      return VK_ERROR_INITIALIZATION_FAILED;

   if (!nv_is_sm_compatible(pdev->info.sm, module->info->sm, backwards_compat))
         return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                          "Incompatible cubin module: pdev sm_%u vs cubin sm_%u",
                          pdev->info.sm, module->info->sm);

   return VK_SUCCESS;
}

void
nvk_cubin_module_destroy(struct nvk_device *dev,
                         struct nvk_cubin_module *module,
                         const VkAllocationCallbacks *pAllocator)
{
   if (!module)
      return;
   if (module->info) {
#ifdef _WIN32
      simple_mtx_lock(&nvk_cubin_elf_mutex);
#endif
      nv_cubin_module_fini(module->info);
#ifdef _WIN32
      simple_mtx_unlock(&nvk_cubin_elf_mutex);
#endif
      vk_free2(&dev->vk.alloc, pAllocator, module->info);
   }
   vk_free2(&dev->vk.alloc, pAllocator, module->data);
   vk_object_free(&dev->vk, pAllocator, module);
}

VkResult
nvk_cubin_module_create(struct nvk_device *dev, const void *data, size_t size,
                        const VkAllocationCallbacks *pAllocator,
                        struct nvk_cubin_module **out)
{
   struct nvk_cubin_module *module;
   VkResult result;
   *out = NULL;

   module = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*module),
                             VK_OBJECT_TYPE_CU_MODULE_NVX);
   if (!module)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = nvk_cubin_module_init(dev, module, data, size, pAllocator);
   if (result != VK_SUCCESS) {
      nvk_cubin_module_destroy(dev, module, pAllocator);
      return result;
   }

   *out = module;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_CreateCuModuleNVX(VkDevice _device,
                      const VkCuModuleCreateInfoNVX *pCreateInfo,
                      const VkAllocationCallbacks *pAllocator,
                      VkCuModuleNVX *pModule)
{
   VK_FROM_HANDLE(nvk_device, dev, _device);
   struct nvk_cubin_module *module;
   VkResult result = nvk_cubin_module_create(dev, pCreateInfo->pData,
      pCreateInfo->dataSize, pAllocator, &module);
   if (result == VK_SUCCESS)
      *pModule = nvk_cubin_module_to_handle(module);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
nvk_DestroyCuModuleNVX(VkDevice _device, VkCuModuleNVX _module,
                       const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(nvk_device, dev, _device);
   VK_FROM_HANDLE(nvk_cubin_module, module, _module);

   if (!module)
      return;

   nvk_cubin_module_destroy(dev, module, pAllocator);
}

/* nvk_cubin_function */

static VkResult
nvk_cubin_function_init(struct nvk_device *dev,
                        struct nvk_cubin_function *function,
                        struct nvk_cubin_module *module, const char *name)
{
   const struct nvk_physical_device *pdev = nvk_device_physical(dev);
   const struct nak_compiler *nak = pdev->nak;

   function->module = module;
   function->info =
      nv_cubin_module_find_function(module->info, name);
   if (!function->info)
      return VK_ERROR_INITIALIZATION_FAILED;
   if (function->info->code_size > UINT32_MAX)
      return VK_ERROR_INITIALIZATION_FAILED;
   if (function->info->constant2_size > NVK_MAX_CBUF_SIZE)
      return VK_ERROR_INITIALIZATION_FAILED;
   if (function->info->stack_size_unknown)
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
   if (pdev->info.cls_compute >= BLACKWELL_COMPUTE_A &&
       (function->info->crs_size || function->info->static_lmem_size ||
        function->info->slm_size > 0x7ffc0))
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);

   const int alignment = 0x80; /* maxwell+ */

   VkResult result = nvk_heap_upload(
      dev, &dev->shader_heap, function->info->code_ptr,
      function->info->code_size, alignment, &function->upload_addr);
   if (result != VK_SUCCESS)
      return result;

   function->upload_size = function->info->code_size;
   if (function->info->constant2_size) {
      result = nvk_heap_upload(dev, &dev->shader_heap,
         function->info->constant2_ptr, function->info->constant2_size,
         nvk_min_cbuf_alignment(&pdev->info), &function->constant2_addr);
      if (result != VK_SUCCESS)
         return result;
      function->constant2_size = function->info->constant2_size;
   }
   function->max_warps_per_sm = nak_max_warps_per_sm(function->info->gpr_count, nak);

   return VK_SUCCESS;
}

void
nvk_cubin_function_destroy(struct nvk_device *dev,
                           struct nvk_cubin_function *function,
                           const VkAllocationCallbacks *pAllocator)
{
   if (!function)
      return;
   if (function->upload_size) {
      nvk_heap_free(dev, &dev->shader_heap, function->upload_addr,
                    function->upload_size);
   }
   if (function->constant2_size)
      nvk_heap_free(dev, &dev->shader_heap, function->constant2_addr,
                    function->constant2_size);

   vk_object_free(&dev->vk, pAllocator, function);
}

VkResult
nvk_cubin_function_create(struct nvk_device *dev,
                          struct nvk_cubin_module *module, const char *name,
                          const VkAllocationCallbacks *pAllocator,
                          struct nvk_cubin_function **out)
{
   struct nvk_cubin_function *function;
   VkResult result;
   *out = NULL;

   function = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*function),
                               VK_OBJECT_TYPE_CU_FUNCTION_NVX);
   if (!function)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = nvk_cubin_function_init(dev, function, module, name);
   if (result != VK_SUCCESS) {
      nvk_cubin_function_destroy(dev, function, pAllocator);
      return result;
   }

   *out = function;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_CreateCuFunctionNVX(VkDevice _device,
                        const VkCuFunctionCreateInfoNVX *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator,
                        VkCuFunctionNVX *pFunction)
{
   VK_FROM_HANDLE(nvk_device, dev, _device);
   VK_FROM_HANDLE(nvk_cubin_module, module, pCreateInfo->module);
   struct nvk_cubin_function *function;
   VkResult result = nvk_cubin_function_create(dev, module,
      pCreateInfo->pName, pAllocator, &function);
   if (result == VK_SUCCESS)
      *pFunction = nvk_cubin_function_to_handle(function);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
nvk_DestroyCuFunctionNVX(VkDevice _device, VkCuFunctionNVX _function,
                         const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(nvk_device, dev, _device);
   VK_FROM_HANDLE(nvk_cubin_function, function, _function);

   if (!function)
      return;

   nvk_cubin_function_destroy(dev, function, pAllocator);
}

/* kernel launch */

struct nvk_cubin_root_descriptor {
   uint8_t data[4096];
};

/* extra launch parameters taken from vkd3d-proton */
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((const void *)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE    ((const void *)0x02)
#define CU_LAUNCH_PARAM_END            ((const void *)0x00)

VkResult
nvk_cmd_cubin_prepare(struct nvk_cmd_buffer *cmd,
                      const struct nvk_cubin_function *function,
                      const struct nvk_cubin_launch_info *info,
                      struct nvk_prepared_dispatch *out)
{
   const struct nv_cubin_function *function_info = function->info;

   /*
    * Use a root descriptor on the stack if its < 4kb,
    * otherwise, allocate one big enough for us to hold
    * the kernel parameters.
    */
   struct nvk_cubin_root_descriptor stack_root;
   struct nvk_cubin_root_descriptor *root = NULL;
   const size_t root_size =
      align(MAX2(sizeof(struct nvk_cubin_root_descriptor),
                 function_info->params_offset + function_info->params_size),
            0x100);
   if (root_size <= 4096) {
      root = &stack_root;
   } else {
      root = malloc(root_size);
      if (!root)
         return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   memset(root, 0, root_size);
   const bool new_abi = function->module->info->abi >= NV_ELF_OSABI_41;
   memcpy(root->data + (new_abi ? 0x360 : 0), info->block_dim,
          sizeof(info->block_dim));
   memcpy(root->data + (new_abi ? 0x370 : 12), info->grid_dim,
          sizeof(info->grid_dim));
   uint32_t local_size_high = 0;
   const struct nvk_physical_device *pdev =
      nvk_device_physical(nvk_cmd_buffer_device(cmd));
   if (new_abi && pdev->info.cls_compute >= BLACKWELL_COMPUTE_A &&
       function_info->slm_size) {
      const uint32_t stack_pointer = 0xffffc0;
      memcpy(root->data + 0x37c, &stack_pointer, sizeof(stack_pointer));
      local_size_high = align(function_info->slm_size + 0x40, 16);
   }

   if (info->param_data) {
      if (info->param_count)
         goto invalid_params;

      const struct nv_cubin_function_param_info *param_info =
         nv_cubin_function_get_param_info(function_info, 0);
      if (!param_info || param_info->offset > function_info->params_size ||
          info->param_size > function_info->params_size - param_info->offset)
         goto invalid_params;

      const size_t param_offset =
         function_info->params_offset + param_info->offset;

      void *param_root_dst = ((uint8_t *)root) + param_offset;
      memcpy(param_root_dst, info->param_data, info->param_size);
   } else {
      if (info->param_size || info->param_count > function_info->param_info_count ||
          (info->param_count && !info->params))
         goto invalid_params;
      for (uint32_t i = 0; i < info->param_count; i++) {
         const void *param_data = info->params[i];

         const struct nv_cubin_function_param_info *param_info =
            nv_cubin_function_get_param_info(function_info, i);
         if (!param_info || !param_data ||
             param_info->offset > function_info->params_size ||
             param_info->size > function_info->params_size - param_info->offset)
            goto invalid_params;

         const size_t param_offset =
            function_info->params_offset + param_info->offset;

         void *param_root_dst = ((uint8_t *)root) + param_offset;
         memcpy(param_root_dst, param_data, param_info->size);
      }
   }

   struct nvk_shader shader;
   memset(&shader, 0, sizeof(shader));
   shader.upload_size = function->upload_size;
   shader.upload_addr = function->upload_addr;
   shader.hdr_addr = function->upload_addr + function_info->func_offset;
   shader.data_addr = function->constant2_addr;
   shader.data_size = function->constant2_size;
   shader.info.stage = MESA_SHADER_COMPUTE;
   shader.info.sm = function->module->info->sm;
   shader.info.num_gprs = function_info->gpr_count;
   shader.info.num_control_barriers = function_info->num_control_barriers;
   shader.info.max_warps_per_sm = function->max_warps_per_sm;
   shader.info.slm_size = local_size_high ? 0 : function_info->slm_size;
   shader.info.crs_size = function_info->crs_size;
   for (unsigned i = 0; i < 3; i++)
      shader.info.cs.local_size[i] = info->block_dim[i];
   shader.info.cs.smem_size =
      function_info->static_smem_size + info->shared_memory_size;
   shader.cbuf_map.cbuf_count = 1;
   shader.cbuf_map.cbufs[0].type = NVK_CBUF_TYPE_ROOT_DESC;
   if (function->constant2_size) {
      shader.cbuf_map.cbuf_count = 3;
      shader.cbuf_map.cbufs[2].type = NVK_CBUF_TYPE_SHADER_DATA;
   }

   VkResult result = nvk_cmd_prepare_dispatch_with_root(cmd, &shader, NULL,
      root, root_size, info->grid_dim, local_size_high, out);

   if (root != &stack_root)
      free(root);
   return result;

invalid_params:
   if (root != &stack_root)
      free(root);
   return VK_ERROR_INITIALIZATION_FAILED;
}

void
nvk_cmd_cubin_launch(struct nvk_cmd_buffer *cmd,
                     const struct nvk_cubin_function *function,
                     const struct nvk_cubin_launch_info *info)
{
   struct nvk_prepared_dispatch prepared;
   VkResult result = nvk_cmd_cubin_prepare(cmd, function, info, &prepared);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   nvk_cmd_launch_prepared_dispatch(cmd, &prepared, true);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdCuLaunchKernelNVX(VkCommandBuffer commandBuffer,
                         const VkCuLaunchInfoNVX *pLaunchInfo)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(nvk_cubin_function, function, pLaunchInfo->function);
   struct nvk_cubin_launch_info info = {
      .grid_dim = {pLaunchInfo->gridDimX, pLaunchInfo->gridDimY, pLaunchInfo->gridDimZ},
      .block_dim = {pLaunchInfo->blockDimX, pLaunchInfo->blockDimY, pLaunchInfo->blockDimZ},
      .shared_memory_size = pLaunchInfo->sharedMemBytes,
      .param_count = pLaunchInfo->paramCount,
      .params = pLaunchInfo->pParams,
   };
   const uint32_t *param_size = NULL;
   if (pLaunchInfo->extraCount) {
      const void *const *extra = pLaunchInfo->pExtras;
      while (*extra != CU_LAUNCH_PARAM_END) {
         if (*extra == CU_LAUNCH_PARAM_BUFFER_POINTER) {
            extra++;
            info.param_data = *extra;
         } else if (*extra == CU_LAUNCH_PARAM_BUFFER_SIZE) {
            extra++;
            param_size = *extra;
         }
         extra++;
      }
   }
   if (info.param_data && param_size)
      info.param_size = *param_size;
   else
      info.param_data = NULL;
   nvk_cmd_cubin_launch(cmd, function, &info);
}
