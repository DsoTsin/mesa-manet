/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_priv_bo.h"
#include "panvk_shader_instrumentation.h"

#include "vk_log.h"
#include "vk_util.h"

#define PANVK_SHADER_INSTRUMENTATION_CHUNK_SIZE 0x8000

static const struct {
   const char *name;
   const char *description;
} panvk_shader_instrumentation_metrics[PAN_INSTRUMENTATION_METRICS] = {
   {"fma", "Number of FMA instructions"},
   {"cvt", "Number of CVT instructions."},
   {"sfu", "Number of SFU instructions."},
   {"memaccesses", "Number of memory accesses."},
   {"texturing", "Number of Texturing instructions."},
   {"varying", "Number of Varying instructions."},
};

VKAPI_ATTR VkResult VKAPI_CALL
panvk_EnumeratePhysicalDeviceShaderInstrumentationMetricsARM(
   VkPhysicalDevice physicalDevice, uint32_t *pDescriptionCount,
   VkShaderInstrumentationMetricDescriptionARM *pDescriptions)
{
   VK_OUTARRAY_MAKE_TYPED(VkShaderInstrumentationMetricDescriptionARM, out,
                          pDescriptions, pDescriptionCount);

   for (unsigned i = 0; i < PAN_INSTRUMENTATION_METRICS; i++) {
      vk_outarray_append_typed(VkShaderInstrumentationMetricDescriptionARM,
                               &out, desc) {
         snprintf(desc->name, sizeof(desc->name), "%s",
                  panvk_shader_instrumentation_metrics[i].name);
         snprintf(desc->description, sizeof(desc->description), "%s",
                  panvk_shader_instrumentation_metrics[i].description);
      }
   }

   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_CreateShaderInstrumentationARM(
   VkDevice _device, const VkShaderInstrumentationCreateInfoARM *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkShaderInstrumentationARM *pInstrumentation)
{
   VK_FROM_HANDLE(panvk_device, dev, _device);
   struct panvk_shader_instrumentation *instr =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*instr),
                       VK_OBJECT_TYPE_SHADER_INSTRUMENTATION_ARM);
   if (!instr)
      return panvk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   util_dynarray_init(&instr->chunks, NULL);
   util_dynarray_init(&instr->records, NULL);

   *pInstrumentation = panvk_shader_instrumentation_to_handle(instr);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_DestroyShaderInstrumentationARM(VkDevice _device,
                                      VkShaderInstrumentationARM instrumentation,
                                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(panvk_device, dev, _device);
   VK_FROM_HANDLE(panvk_shader_instrumentation, instr, instrumentation);

   if (!instr)
      return;

   util_dynarray_foreach(&instr->chunks, struct panvk_priv_bo *, bo)
      panvk_priv_bo_unref(*bo);

   util_dynarray_fini(&instr->chunks);
   util_dynarray_fini(&instr->records);
   vk_object_free(&dev->vk, pAllocator, instr);
}

uint64_t
panvk_shader_instrumentation_record(
   struct panvk_device *dev, struct panvk_shader_instrumentation *instr,
   const VkShaderStageFlags stages[PANVK_SHADER_INSTRUMENTATION_SLOTS])
{
   const uint32_t result_index = instr->next_result_index++;

   unsigned slots = 0;
   for (unsigned s = 0; s < PANVK_SHADER_INSTRUMENTATION_SLOTS; s++) {
      if (stages[s])
         slots = s + 1;
   }

   if (!slots)
      return PAN_SHADER_OOB_ADDRESS;

   const uint32_t size = slots * PAN_INSTRUMENTATION_SLOT_SIZE;
   if (!util_dynarray_num_elements(&instr->chunks, struct panvk_priv_bo *) ||
       instr->chunk_used + size > PANVK_SHADER_INSTRUMENTATION_CHUNK_SIZE) {
      struct panvk_priv_bo *bo;
      if (panvk_priv_bo_create(dev, PANVK_SHADER_INSTRUMENTATION_CHUNK_SIZE, 0,
                               VK_SYSTEM_ALLOCATION_SCOPE_OBJECT,
                               &bo) != VK_SUCCESS)
         return PAN_SHADER_OOB_ADDRESS;

      memset(bo->addr.host, 0, PANVK_SHADER_INSTRUMENTATION_CHUNK_SIZE);
      panvk_priv_bo_flush(bo, 0, PANVK_SHADER_INSTRUMENTATION_CHUNK_SIZE);
      util_dynarray_append(&instr->chunks, bo);
      instr->chunk_used = 0;
   }

   struct panvk_priv_bo *bo =
      util_dynarray_top(&instr->chunks, struct panvk_priv_bo *);
   struct panvk_shader_instrumentation_record rec = {
      .result_index = result_index,
      .counters = (uint64_t *)((uint8_t *)bo->addr.host + instr->chunk_used),
   };
   memcpy(rec.stages, stages, sizeof(rec.stages));
   util_dynarray_append(&instr->records, rec);

   const uint64_t addr = bo->addr.dev + instr->chunk_used;
   instr->chunk_used += size;
   return addr;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_GetShaderInstrumentationValuesARM(
   VkDevice _device, VkShaderInstrumentationARM instrumentation,
   uint32_t *pMetricBlockCount, void *pMetricValues,
   VkShaderInstrumentationValuesFlagsARM flags)
{
   VK_FROM_HANDLE(panvk_shader_instrumentation, instr, instrumentation);

   uint32_t count = 0;
   util_dynarray_foreach(&instr->records,
                         struct panvk_shader_instrumentation_record, rec) {
      for (unsigned s = 0; s < PANVK_SHADER_INSTRUMENTATION_SLOTS; s++)
         count += rec->stages[s] != 0;
   }

   if (!pMetricValues) {
      *pMetricBlockCount = count;
      return VK_SUCCESS;
   }

   util_dynarray_foreach(&instr->chunks, struct panvk_priv_bo *, bo)
      panvk_priv_bo_invalidate(*bo, 0, PANVK_SHADER_INSTRUMENTATION_CHUNK_SIZE);

   const size_t stride = sizeof(VkShaderInstrumentationMetricDataHeaderARM) +
                         PAN_INSTRUMENTATION_SLOT_SIZE;
   uint8_t *out = pMetricValues;
   uint32_t written = 0;

   util_dynarray_foreach(&instr->records,
                         struct panvk_shader_instrumentation_record, rec) {
      for (unsigned s = 0; s < PANVK_SHADER_INSTRUMENTATION_SLOTS; s++) {
         if (!rec->stages[s])
            continue;

         if (written == *pMetricBlockCount)
            return VK_INCOMPLETE;

         const VkShaderInstrumentationMetricDataHeaderARM header = {
            .resultIndex = rec->result_index,
            .stages = rec->stages[s],
         };
         memcpy(out, &header, sizeof(header));
         memcpy(out + sizeof(header),
                rec->counters + s * PAN_INSTRUMENTATION_METRICS,
                PAN_INSTRUMENTATION_SLOT_SIZE);
         out += stride;
         written++;
      }
   }

   *pMetricBlockCount = written;
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_ClearShaderInstrumentationMetricsARM(
   VkDevice _device, VkShaderInstrumentationARM instrumentation)
{
   VK_FROM_HANDLE(panvk_shader_instrumentation, instr, instrumentation);

   util_dynarray_foreach(&instr->chunks, struct panvk_priv_bo *, bo) {
      memset((*bo)->addr.host, 0, PANVK_SHADER_INSTRUMENTATION_CHUNK_SIZE);
      panvk_priv_bo_flush(*bo, 0, PANVK_SHADER_INSTRUMENTATION_CHUNK_SIZE);
   }
}
