/*
 * Copyright 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "genxml/gen_macros.h"

#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_meta.h"
#include "panvk_physical_device.h"
#include "panvk_tensor.h"

#include "nir_builder.h"
#include "util/u_math.h"
#include "vk_log.h"
#include "vk_meta.h"

static enum mali_tensor_element_format
tensor_element_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R8_UINT:
      return MALI_TENSOR_ELEMENT_FORMAT_U8;
   case VK_FORMAT_R8_SINT:
   case VK_FORMAT_R8_BOOL_ARM:
      return MALI_TENSOR_ELEMENT_FORMAT_S8;
   case VK_FORMAT_R16_UINT:
      return MALI_TENSOR_ELEMENT_FORMAT_U16;
   case VK_FORMAT_R16_SINT:
      return MALI_TENSOR_ELEMENT_FORMAT_S16;
   case VK_FORMAT_R16_SFLOAT:
      return MALI_TENSOR_ELEMENT_FORMAT_F16;
   case VK_FORMAT_R32_UINT:
      return MALI_TENSOR_ELEMENT_FORMAT_U32;
   case VK_FORMAT_R32_SINT:
      return MALI_TENSOR_ELEMENT_FORMAT_S32;
   case VK_FORMAT_R32_SFLOAT:
      return MALI_TENSOR_ELEMENT_FORMAT_F32;
   case VK_FORMAT_R64_UINT:
      return MALI_TENSOR_ELEMENT_FORMAT_U64;
   case VK_FORMAT_R64_SINT:
      return MALI_TENSOR_ELEMENT_FORMAT_S64;
   default:
      UNREACHABLE("Unsupported tensor format");
   }
}

static unsigned
wrap_shift(uint32_t wrap)
{
   unsigned n = 0;
   while (n < 16 && !((1u << n) < wrap && wrap <= (2u << n)))
      n++;
   return n;
}

static bool
wrap_is_npot(uint32_t wrap, uint32_t dim)
{
   return wrap != dim && !util_is_power_of_two_nonzero(wrap);
}

static int
wrap_divisor_dim(const uint32_t dims[4], const uint32_t wraps[4])
{
   int dim = -1;
   for (unsigned i = 0; i < 3; i++) {
      if (wrap_is_npot(wraps[i], dims[i]))
         dim = i;
   }

   if (wrap_is_npot(wraps[3], dims[3]))
      return 3;

   if (dim >= 0)
      return dim;

   for (dim = 0; dim < 3; dim++) {
      if (wraps[dim] > 1 || wraps[dim] == dims[dim])
         return dim;
   }

   return wraps[3] < 2 && wraps[3] != dims[3] ? -1 : 3;
}

struct tensor_wraps {
   unsigned fields[PANVK_MAX_TENSOR_DIMS];
   unsigned divisor_dim;
   uint32_t divisor;
   uint32_t reciprocal;
};

static struct tensor_wraps
get_tensor_wraps(const uint32_t dims[4], const uint32_t wraps[4])
{
   const int div_dim = wrap_divisor_dim(dims, wraps);
   struct tensor_wraps w = {
      .divisor_dim = MAX2(div_dim, 0),
   };

   for (unsigned i = 0; i < PANVK_MAX_TENSOR_DIMS; i++) {
      if ((int)i != div_dim) {
         w.fields[i] = wraps[i] == dims[i] ? 31 : util_logbase2(wraps[i]) & 31;
      } else if (wraps[i] != dims[i]) {
         w.fields[i] = wrap_shift(wraps[i]);
         w.divisor = wraps[i];
         w.reciprocal = ((1ull << (32 + w.fields[i])) + w.divisor) / w.divisor;
      }
   }

   return w;
}

static void
pack_tensor_desc(const struct panvk_tensor *tensor, VkFormat format,
                 struct mali_tensor_packed *packed)
{
   const unsigned n = tensor->dim_count;
   const unsigned pad = PANVK_MAX_TENSOR_DIMS - n;
   uint32_t dims[PANVK_MAX_TENSOR_DIMS] = {1, 1, 1, 1};
   uint32_t strides[PANVK_MAX_TENSOR_DIMS] = {1, 1, 1, 1};
   uint32_t wraps[PANVK_MAX_TENSOR_DIMS] = {1, 1, 1, 1};

   for (unsigned i = 0; i < n; i++) {
      dims[pad + i] = tensor->dims[i];
      strides[pad + i] = tensor->strides[i];
      if (tensor->rolling)
         wraps[pad + i] = tensor->wraps[i];
   }

   for (unsigned i = 0; i < pad; i++)
      strides[i] = strides[pad] * dims[pad];

   int wide = -1;
   for (unsigned i = 0; i < PANVK_MAX_TENSOR_DIMS; i++) {
      if (dims[i] > UINT16_MAX)
         wide = i;
   }

   const struct tensor_wraps w =
      tensor->rolling ? get_tensor_wraps(dims, wraps) : (struct tensor_wraps){0};

   pan_pack(packed, TENSOR, cfg) {
      cfg.rolling = tensor->rolling;
      cfg.element_format = tensor_element_format(format);
      cfg.tiling = (enum mali_tensor_tiling)tensor->tiling;
      cfg.wide_dimension = MAX2(wide, 0);
      cfg.wide_dimension_high = wide >= 0 ? dims[wide] >> 16 : 0;
      cfg.dimension_0 = dims[0] & UINT16_MAX;
      cfg.dimension_1 = dims[1] & UINT16_MAX;
      cfg.dimension_2 = dims[2] & UINT16_MAX;
      cfg.dimension_3 = dims[3] & UINT16_MAX;
      cfg.stride_0 = strides[0];
      cfg.stride_1 = strides[1];
      cfg.stride_2 = strides[2];
      cfg.address = tensor->dev_addr;
      cfg.wrap_0 = w.fields[0];
      cfg.wrap_1 = w.fields[1];
      cfg.wrap_2 = w.fields[2];
      cfg.wrap_3 = w.fields[3];
      cfg.wrap_divisor_dimension = w.divisor_dim;
      cfg.wrap_divisor = w.divisor;
      cfg.wrap_divisor_reciprocal = w.reciprocal;
   }
}

static void
init_tensor_view(struct panvk_tensor_view *view, struct panvk_tensor *tensor,
                 VkFormat format)
{
   static_assert(sizeof(view->desc) == sizeof(struct mali_tensor_packed),
                 "tensor descriptor size mismatch");

   view->tensor = tensor;
   view->format = format;
   pack_tensor_desc(tensor, format, (struct mali_tensor_packed *)view->desc);
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CreateTensorViewARM)(VkDevice _device,
                                    const VkTensorViewCreateInfoARM *pCreateInfo,
                                    const VkAllocationCallbacks *pAllocator,
                                    VkTensorViewARM *pView)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_tensor, tensor, pCreateInfo->tensor);

   struct panvk_tensor_view *view =
      vk_object_zalloc(&device->vk, pAllocator, sizeof(*view),
                       VK_OBJECT_TYPE_TENSOR_VIEW_ARM);
   if (!view)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   init_tensor_view(view, tensor, pCreateInfo->format);

   *pView = panvk_tensor_view_to_handle(view);
   return VK_SUCCESS;
}

#define TENSOR_COPY_WG_SIZE 64

struct tensor_copy_key {
   enum panvk_meta_object_key_type type;
   uint32_t rank;
   uint32_t elem_bits;
   uint32_t elems;
};

struct tensor_copy_info {
   uint32_t dims[PANVK_MAX_TENSOR_DIMS];
   uint32_t total;
   uint32_t row;
};

#define load_copy_info(b, field)                                               \
   nir_load_push_constant(b, 1, 32,                                            \
                          nir_imm_int(b, offsetof(struct tensor_copy_info,     \
                                                  field)))

static nir_def *
load_copy_tensor_desc(nir_builder *b, unsigned binding)
{
   nir_def *index = nir_vulkan_resource_index(
      b, 3, 32, nir_imm_int(b, 0), .binding = binding,
      .desc_type = nir_descriptor_type_tensor_arm);

   return nir_load_vulkan_descriptor(b, 3, 32, index,
                                     .desc_type = nir_descriptor_type_tensor_arm);
}

static nir_shader *
build_tensor_copy_shader(const struct tensor_copy_key *key)
{
   nir_builder builder = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL, "panvk-meta-copy-tensor");
   nir_builder *b = &builder;

   b->shader->info.workgroup_size[0] = TENSOR_COPY_WG_SIZE;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;

   nir_def *id = nir_load_global_invocation_id(b, 32);
   nir_def *idx = nir_iadd(b, nir_channel(b, id, 0),
                           nir_imul(b, nir_channel(b, id, 1),
                                    load_copy_info(b, row)));

   nir_push_if(b, nir_ult(b, idx, load_copy_info(b, total)));

   nir_def *coords[PANVK_MAX_TENSOR_DIMS];
   for (unsigned d = key->rank - 1; d > 0; d--) {
      nir_def *dim = load_copy_info(b, dims[d]);
      coords[d] = nir_umod(b, idx, dim);
      idx = nir_udiv(b, idx, dim);
   }
   coords[0] = idx;
   coords[key->rank - 1] =
      nir_imul_imm(b, coords[key->rank - 1], key->elems);

   nir_def *c = nir_vec(b, coords, key->rank);
   nir_def *data = nir_tensor_read_arm(
      b, key->elem_bits, load_copy_tensor_desc(b, 0), c,
      nir_imm_zero(b, key->elems, key->elem_bits));
   nir_tensor_write_arm(b, load_copy_tensor_desc(b, 1), c, data);

   nir_pop_if(b, NULL);

   return b->shader;
}

static VkResult
get_tensor_copy_pipeline(struct panvk_device *dev,
                         const struct tensor_copy_key *key,
                         VkPipelineLayout *layout, VkPipeline *pipeline)
{
   const VkDescriptorSetLayoutBinding bindings[] = {
      {
         .binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };
   const VkDescriptorSetLayoutCreateInfo set_layout = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
      .bindingCount = ARRAY_SIZE(bindings),
      .pBindings = bindings,
   };
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct tensor_copy_info),
   };

   VkResult result =
      vk_meta_get_pipeline_layout(&dev->vk, &dev->meta, &set_layout,
                                  &push_range, &key->type, sizeof(key->type),
                                  layout);
   if (result != VK_SUCCESS)
      return result;

   *pipeline = vk_meta_lookup_pipeline(&dev->meta, key, sizeof(*key));
   if (*pipeline != VK_NULL_HANDLE)
      return VK_SUCCESS;

   const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
      .nir = build_tensor_copy_shader(key),
   };
   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = &nir_info,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .pName = "main",
      },
      .layout = *layout,
   };

   result = vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &info, key,
                                            sizeof(*key), pipeline);
   ralloc_free(nir_info.nir);
   return result;
}

static VkFormat
tensor_copy_format(unsigned elem_size)
{
   switch (elem_size) {
   case 1:
      return VK_FORMAT_R8_UINT;
   case 2:
      return VK_FORMAT_R16_UINT;
   case 4:
      return VK_FORMAT_R32_UINT;
   default:
      return VK_FORMAT_R64_UINT;
   }
}

void
panvk_per_arch(meta_copy_tensor)(struct panvk_cmd_buffer *cmdbuf,
                                 const VkCopyTensorInfoARM *info)
{
   VK_FROM_HANDLE(panvk_tensor, src, info->srcTensor);
   VK_FROM_HANDLE(panvk_tensor, dst, info->dstTensor);
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct panvk_physical_device *pdev =
      to_panvk_physical_device(dev->vk.physical);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkCommandBuffer cmd = panvk_cmd_buffer_to_handle(cmdbuf);

   const unsigned elem_size = panvk_tensor_format_size(src->format);
   const unsigned rank = src->dim_count;
   const uint32_t inner = src->dims[rank - 1];
   const struct tensor_copy_key key = {
      .type = PANVK_META_OBJECT_KEY_COPY_TENSOR,
      .rank = rank,
      .elem_bits = elem_size * 8,
      .elems = MIN2(16 / elem_size, 1u << (ffs(inner) - 1)),
   };

   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result = get_tensor_copy_pipeline(dev, &key, &layout, &pipeline);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmdbuf->vk, result);
      return;
   }

   struct tensor_copy_info push = {.total = 1};
   for (unsigned d = 0; d < rank; d++) {
      push.dims[d] = d == rank - 1 ? inner / key.elems : src->dims[d];
      push.total *= push.dims[d];
   }

   const uint32_t groups = DIV_ROUND_UP(push.total, TENSOR_COPY_WG_SIZE);
   const uint32_t groups_x =
      MIN2(groups, pdev->vk.properties.maxComputeWorkGroupCount[0]);
   push.row = groups_x * TENSOR_COPY_WG_SIZE;

   struct panvk_tensor_view views[2];
   struct panvk_tensor *tensors[] = {src, dst};
   VkTensorViewARM handles[2];
   VkWriteDescriptorSetTensorARM tensor_writes[2];
   VkWriteDescriptorSet writes[2];
   for (unsigned i = 0; i < 2; i++) {
      vk_object_base_init(&dev->vk, &views[i].base,
                          VK_OBJECT_TYPE_TENSOR_VIEW_ARM);
      init_tensor_view(&views[i], tensors[i], tensor_copy_format(elem_size));
      handles[i] = panvk_tensor_view_to_handle(&views[i]);
      tensor_writes[i] = (VkWriteDescriptorSetTensorARM){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM,
         .tensorViewCount = 1,
         .pTensorViews = &handles[i],
      };
      writes[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .pNext = &tensor_writes[i],
         .dstBinding = i,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
      };
   }

   disp->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   disp->CmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                                 0, ARRAY_SIZE(writes), writes);
   disp->CmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                          sizeof(push), &push);
   disp->CmdDispatch(cmd, groups_x, DIV_ROUND_UP(groups, groups_x), 1);

   for (unsigned i = 0; i < 2; i++)
      vk_object_base_finish(&views[i].base);
}
