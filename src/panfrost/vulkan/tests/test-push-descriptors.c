#include "panvk_cmd_meta.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_sampler.h"
#include "panvk_tensor.h"
#include "vk_alloc.h"
#include "vk_ycbcr_conversion.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct panvk_device device;
static struct panvk_sampler sampler[4];
static struct vk_ycbcr_conversion conversion[3];
static VkSampler samplers[4][MAX_PUSH_DESCS];
static unsigned cases, tensor_writes;

static struct panvk_descriptor_set_layout *
create_layout(const uint8_t *types, unsigned count, unsigned array_count)
{
   VkDescriptorSetLayoutBinding bindings[MAX_PUSH_DESCS] = {0};
   for (unsigned i = 0; i < count; i++) {
      bindings[i] = (VkDescriptorSetLayoutBinding) {
         .binding = i,
         .descriptorCount = array_count,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
         .descriptorType = types[i] == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER :
                           types[i] == 1 ? VK_DESCRIPTOR_TYPE_TENSOR_ARM :
                                           VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImmutableSamplers = types[i] >= 2 ? samplers[types[i] - 2] : NULL,
      };
   }
   VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = count,
      .pBindings = bindings,
   };
   VkDescriptorSetLayoutSupport support = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT,
   };
   VkDevice handle = panvk_device_to_handle(&device);
   panvk_per_arch(GetDescriptorSetLayoutSupport)(handle, &info, &support);
   assert(support.supported);
   VkDescriptorSetLayout layout;
   assert(panvk_per_arch(CreateDescriptorSetLayout)(handle, &info, NULL, &layout) == VK_SUCCESS);
   return panvk_descriptor_set_layout_from_handle(layout);
}

static void
check_storage(struct panvk_descriptor_set_layout *layout, bool write_tensors)
{
   struct {
      uint64_t before[4];
      struct panvk_push_set push;
      uint64_t after[4];
   } *storage = calloc(1, sizeof(*storage));
   struct {
      uint64_t before[4];
      struct panvk_cmd_meta_compute_save_ctx compute;
      uint64_t after[4];
   } *compute = calloc(1, sizeof(*compute));
   struct {
      uint64_t before[4];
      struct panvk_cmd_meta_graphics_save_ctx gfx;
      uint64_t after[4];
   } *gfx = calloc(1, sizeof(*gfx));
   assert(storage && compute && gfx);
   memset(storage->before, 0xa5, sizeof(storage->before));
   memset(storage->after, 0x5a, sizeof(storage->after));
   memset(compute->before, 0xa5, sizeof(compute->before));
   memset(compute->after, 0x5a, sizeof(compute->after));
   memset(gfx->before, 0xa5, sizeof(gfx->before));
   memset(gfx->after, 0x5a, sizeof(gfx->after));
   struct panvk_descriptor_set *set = &storage->push.set;
   set->layout = layout;
   set->desc_count = layout->desc_count;
   set->descs.host = storage->push.descs;
   size_t bytes = set->desc_count * PANVK_DESCRIPTOR_SIZE;
   if (bytes > sizeof(storage->push.descs))
      fprintf(stderr, "desc_count=%u bytes=%zu capacity=%zu\n", set->desc_count, bytes, sizeof(storage->push.descs));
   assert(bytes <= sizeof(storage->push.descs));
   assert(bytes <= sizeof(compute->compute.push_set0.desc_storage));
   assert(bytes <= sizeof(gfx->gfx.push_set0.desc_storage));
   compute->compute.push_set0.descs_dev_addr = UINT64_C(0x123456789abcdef0);
   gfx->gfx.push_set0.descs_dev_addr = UINT64_C(0x123456789abcdef0);
   compute->compute.push_set0.desc_count = gfx->gfx.push_set0.desc_count = 0xfedcba98;
   compute->compute.push_set0.dirty = gfx->gfx.push_set0.dirty = true;
   for (unsigned i = 0; i < bytes; i++)
      ((uint8_t *)set->descs.host)[i] = i * 37 + 11;

   if (write_tensors) {
      struct panvk_tensor_view view = {0};
      vk_object_base_init(&device.vk, &view.base, VK_OBJECT_TYPE_TENSOR_VIEW_ARM);
      for (unsigned i = 0; i < ARRAY_SIZE(view.desc); i++)
         view.desc[i] = 0xcafe0000 + i;
      VkTensorViewARM handle = panvk_tensor_view_to_handle(&view);
      VkWriteDescriptorSetTensorARM tensor = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM,
         .tensorViewCount = 1,
         .pTensorViews = &handle,
      };
      for (unsigned b = 0; b < layout->binding_count; b++) {
         if (layout->bindings[b].type != VK_DESCRIPTOR_TYPE_TENSOR_ARM)
            continue;
         for (unsigned element = 0;
              element < layout->bindings[b].desc_count; element++) {
            VkWriteDescriptorSet write = {
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .pNext = &tensor,
               .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
               .descriptorCount = 1,
               .dstBinding = b,
               .dstArrayElement = element,
            };
            assert(panvk_per_arch(descriptor_set_write)(set, &write, true) ==
                   VK_SUCCESS);
            unsigned index = layout->bindings[b].desc_idx +
                             element * panvk_get_desc_stride(&layout->bindings[b]);
            const uint8_t *p = (uint8_t *)set->descs.host +
                               index * PANVK_DESCRIPTOR_SIZE;
            assert(memcmp(p, view.desc, sizeof(view.desc)) == 0);
            tensor_writes++;
         }
      }
      vk_object_base_finish(&view.base);
   }

   memcpy(compute->compute.push_set0.desc_storage, set->descs.host, bytes);
   memcpy(gfx->gfx.push_set0.desc_storage, set->descs.host, bytes);
   memset(set->descs.host, 0, bytes);
   memcpy(set->descs.host, compute->compute.push_set0.desc_storage, bytes);
   assert(memcmp(set->descs.host, gfx->gfx.push_set0.desc_storage, bytes) == 0);
   memset(set->descs.host, 0, bytes);
   memcpy(set->descs.host, gfx->gfx.push_set0.desc_storage, bytes);
   assert(memcmp(set->descs.host, compute->compute.push_set0.desc_storage, bytes) == 0);
   assert(compute->compute.push_set0.descs_dev_addr == UINT64_C(0x123456789abcdef0));
   assert(gfx->gfx.push_set0.descs_dev_addr == UINT64_C(0x123456789abcdef0));
   assert(compute->compute.push_set0.desc_count == 0xfedcba98);
   assert(gfx->gfx.push_set0.desc_count == 0xfedcba98);
   assert(compute->compute.push_set0.dirty && gfx->gfx.push_set0.dirty);
   for (unsigned i = 0; i < 4; i++) {
      assert(storage->before[i] == UINT64_C(0xa5a5a5a5a5a5a5a5));
      assert(storage->after[i] == UINT64_C(0x5a5a5a5a5a5a5a5a));
      assert(compute->before[i] == UINT64_C(0xa5a5a5a5a5a5a5a5));
      assert(compute->after[i] == UINT64_C(0x5a5a5a5a5a5a5a5a));
      assert(gfx->before[i] == UINT64_C(0xa5a5a5a5a5a5a5a5));
      assert(gfx->after[i] == UINT64_C(0x5a5a5a5a5a5a5a5a));
   }
   free(gfx);
   free(compute);
   free(storage);
   vk_descriptor_set_layout_unref(&device.vk, &layout->vk);
   cases++;
}

int main(void)
{
   device.vk.alloc = *vk_default_allocator();
   vk_object_base_init(NULL, &device.vk.base, VK_OBJECT_TYPE_DEVICE);
   const VkFormat formats[] = {VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
                               VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                               VK_FORMAT_G8B8G8R8_422_UNORM};
   for (unsigned k = 0; k < 4; k++) {
      vk_object_base_init(&device.vk, &sampler[k].vk.base, VK_OBJECT_TYPE_SAMPLER);
      if (k < 3) {
         conversion[k].state.format = formats[k];
         sampler[k].vk.ycbcr_conversion = &conversion[k];
      }
      sampler[k].desc_count = k < 3 ? PANVK_MAX_DESCS_PER_SAMPLER : 1;
      for (unsigned i = 0; i < MAX_PUSH_DESCS; i++)
         samplers[k][i] = panvk_sampler_to_handle(&sampler[k]);
   }
   uint8_t tensor_types[MAX_PUSH_DESCS];
   memset(tensor_types, 1, sizeof(tensor_types));
   for (unsigned n = 1; n <= MAX_PUSH_DESCS; n++) {
      struct panvk_descriptor_set_layout *layout = create_layout(tensor_types, n, 1);
      assert(layout->desc_count == n * 2);
      check_storage(layout, true);
      layout = create_layout(tensor_types, 1, n);
      assert(layout->desc_count == n * 2);
      check_storage(layout, true);
   }
   uint8_t path[MAX_PUSH_DESCS + 1][PANVK_MAX_PUSH_DESC_SLOTS + 1][MAX_PUSH_DESCS] = {0};
   bool reachable[MAX_PUSH_DESCS + 1][PANVK_MAX_PUSH_DESC_SLOTS + 1] = {0};
   reachable[0][0] = true;
   for (unsigned n = 0; n < MAX_PUSH_DESCS; n++) {
      for (unsigned offset = 0; offset <= PANVK_MAX_PUSH_DESC_SLOTS; offset++) {
         if (!reachable[n][offset])
            continue;
         for (uint8_t type = 0; type < 6; type++) {
            unsigned next = type == 1 ? ALIGN_POT(offset, 2) + 2 :
                            offset + (type >= 2 ? PANVK_MAX_PLANES + PANVK_MAX_DESCS_PER_SAMPLER - (type - 2) : 1);
            assert(next <= PANVK_MAX_PUSH_DESC_SLOTS);
            if (reachable[n + 1][next])
               continue;
            reachable[n + 1][next] = true;
            memcpy(path[n + 1][next], path[n][offset], n);
            path[n + 1][next][n] = type;
            struct panvk_descriptor_set_layout *layout = create_layout(path[n + 1][next], n + 1, 1);
            assert(layout->desc_count == next);
            check_storage(layout, true);
         }
      }
   }
   assert(reachable[MAX_PUSH_DESCS][PANVK_MAX_PUSH_DESC_SLOTS]);
   for (unsigned k = 0; k < 4; k++)
      vk_object_base_finish(&sampler[k].vk.base);
   vk_object_base_finish(&device.vk.base);
   printf("push_descriptor_cases=%u tensor_writes=%u max_slots=%u bytes=%u\n", cases, tensor_writes, PANVK_MAX_PUSH_DESC_SLOTS, PANVK_MAX_PUSH_DESC_SLOTS * PANVK_DESCRIPTOR_SIZE);
   return 0;
}
