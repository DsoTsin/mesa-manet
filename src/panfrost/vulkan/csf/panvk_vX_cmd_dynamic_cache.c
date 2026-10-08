#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdSetDynamicCacheMemoryMTK)(
   VkCommandBuffer commandBuffer, const VkDynamicCacheMemoryInfoMTK *pInfo)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   if (PAN_ARCH != 15 || !dev->dynamic_cache)
      return;
   panvk_cache_policy_add(&cmdbuf->cache_pending, pInfo);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdSetDynamicCacheAccessMTK)(VkCommandBuffer commandBuffer)
{
}

void
panvk_per_arch(cmd_cache_reset)(struct panvk_cmd_buffer *cmdbuf)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   list_for_each_entry_safe(struct panvk_cache_pass, pass,
                            &cmdbuf->cache_passes, link) {
      list_del(&pass->link);
      vk_free(&dev->vk.alloc, pass);
   }
   memset(&cmdbuf->cache_pending, 0, sizeof(cmdbuf->cache_pending));
   cmdbuf->cache_pass_count = 0;
}

void
panvk_per_arch(cmd_cache_begin)(struct panvk_cmd_buffer *cmdbuf, bool compute)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   if (PAN_ARCH != 15 || !dev->dynamic_cache)
      return;
   struct cs_builder *b = panvk_get_cs_builder(
      cmdbuf, compute ? PANVK_SUBQUEUE_COMPUTE : PANVK_SUBQUEUE_FRAGMENT);
   uint16_t id = 0;
   const bool enabled = cmdbuf->cache_pending.count != 0;
   if (enabled) {
      if (cmdbuf->cache_pass_count == UINT16_MAX) {
         vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }
      struct panvk_cache_pass *pass =
         vk_alloc(&dev->vk.alloc, sizeof(*pass), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
      if (!pass) {
         vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }
      id = ++cmdbuf->cache_pass_count;
      *pass = (struct panvk_cache_pass){
         .policy = cmdbuf->cache_pending, .id = id, .compute = compute,
      };
      list_addtail(&pass->link, &cmdbuf->cache_passes);
      memset(&cmdbuf->cache_pending, 0, sizeof(cmdbuf->cache_pending));
   }
   uint64_t word = panvk_cache_control_word(compute, enabled);
   memcpy(cs_alloc_ins(b), &word, sizeof(word));
   word = panvk_cache_pass_word(compute, id);
   memcpy(cs_alloc_ins(b), &word, sizeof(word));
}

void
panvk_per_arch(cmd_cache_end)(struct panvk_cmd_buffer *cmdbuf, bool compute)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   if (PAN_ARCH != 15 || !dev->dynamic_cache)
      return;
   if (compute)
      panvk_per_arch(cmd_pilot_close)(cmdbuf, PANVK_SUBQUEUE_COMPUTE);
   struct cs_builder *b = panvk_get_cs_builder(
      cmdbuf, compute ? PANVK_SUBQUEUE_COMPUTE : PANVK_SUBQUEUE_FRAGMENT);
   uint64_t word = compute ? UINT64_C(0x103e3e0600000214)
                           : panvk_cache_control_word(false, false);
   memcpy(cs_alloc_ins(b), &word, sizeof(word));
   word = panvk_cache_pass_word(compute, 0);
   memcpy(cs_alloc_ins(b), &word, sizeof(word));
}
