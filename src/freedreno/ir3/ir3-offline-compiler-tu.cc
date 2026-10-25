#include "ir3-offline-compiler.h"
#include "tu_descriptor_set.h"
#include "tu_device.h"
#include "tu_shader.h"
#include "vk_alloc.h"

struct ir3_offline_context {
   ir3_offline_options options;
   tu_instance instance;
   tu_physical_device physical;
   tu_device device;
   struct fd_dev_info info;
   tu_shader_key key;
   tu_shader *shader;
};

struct ir3_offline_context *
ir3_offline_context_create(const struct ir3_offline_options *options)
{
   auto ctx = static_cast<ir3_offline_context *>(calloc(1, sizeof(ir3_offline_context)));
   if (!ctx)
      return nullptr;
   ctx->options = *options;
   ctx->info = fd_dev_info(&options->dev_id);
   if (ctx->info.chip < 6 || ctx->info.chip > 8) {
      free(ctx);
      return nullptr;
   }
   glsl_type_singleton_init_or_ref();
   ctx->instance.vk.app_info.api_version = VK_API_VERSION_1_4;
   vk_object_base_instance_init(&ctx->instance.vk, &ctx->instance.vk.base,
                               VK_OBJECT_TYPE_INSTANCE);
   ctx->instance.vk.base.client_visible = true;
   list_inithead(&ctx->instance.vk.debug_report.callbacks);
   list_inithead(&ctx->instance.vk.debug_utils.callbacks);
   ctx->physical.instance = &ctx->instance;
   ctx->physical.vk.instance = &ctx->instance.vk;
   ctx->physical.vk.properties.apiVersion = VK_API_VERSION_1_4;
   ctx->physical.vk.properties.subgroupSize = ctx->info.threadsize_base;
   auto &properties = ctx->physical.vk.properties;
   properties.subgroupSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT;
   properties.subgroupSupportedOperations =
      VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
      VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
      VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT | VK_SUBGROUP_FEATURE_ROTATE_BIT_KHR |
      VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT_KHR |
      VK_SUBGROUP_FEATURE_CLUSTERED_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
   if (ctx->info.props.has_getfiberid) {
      properties.subgroupSupportedStages |= VK_SHADER_STAGE_ALL_GRAPHICS;
      properties.subgroupSupportedOperations |= VK_SUBGROUP_FEATURE_QUAD_BIT;
   }
   properties.subgroupQuadOperationsInAllStages = ctx->info.props.has_getfiberid;
   properties.minSubgroupSize = ctx->info.threadsize_base;
   properties.maxSubgroupSize = ctx->info.threadsize_base;
   properties.requiredSubgroupSizeStages = VK_SHADER_STAGE_ALL;
   memset(&ctx->physical.vk.supported_features, 1,
          sizeof(ctx->physical.vk.supported_features));
   ctx->physical.info = &ctx->info;
   ctx->physical.dev_id = options->dev_id;
   ctx->physical.usable_sets = ctx->info.props.max_sets;
   ctx->physical.reserved_set_idx = -1;
   ctx->device.physical_device = &ctx->physical;
   ctx->device.vk.physical = &ctx->physical.vk;
   ctx->device.vk.alloc = *vk_default_allocator();
   vk_object_base_init(&ctx->device.vk, &ctx->device.vk.base, VK_OBJECT_TYPE_DEVICE);
   ctx->device.vk.base.client_visible = true;
   mtx_init(&ctx->device.softfloat_mutex, mtx_plain);
   ir3_compiler_options compiler_options = {};
   compiler_options.push_ubo_with_preamble = true;
   compiler_options.disable_cache = true;
   compiler_options.bindless_fb_read_descriptor = -1;
   compiler_options.bindless_fb_read_slot = -1;
   compiler_options.storage_16bit = ctx->info.props.storage_16bit;
   compiler_options.storage_8bit = ctx->info.props.storage_8bit;
   compiler_options.shared_push_consts = true;
   compiler_options.lower_draw_id = ctx->info.chip >= 8;
   ctx->device.compiler = ir3_compiler_create(nullptr, &ctx->physical.dev_id,
                                             &ctx->info, &compiler_options);
   if (!ctx->device.compiler) {
      ir3_offline_context_destroy(ctx);
      return nullptr;
   }
   VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup = {};
   subgroup.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
   subgroup.requiredSubgroupSize = options->subgroup_size;
   tu_shader_key_subgroup_size(&ctx->key, !options->subgroup_size, false,
                              options->subgroup_size ? &subgroup : nullptr,
                              &ctx->device);
   ctx->key.robust_storage_access2 = options->robust;
   ctx->key.robust_uniform_access2 = options->robust;
   ctx->key.force_sample_interp = options->samples > 1;
   return ctx;
}

nir_shader *
ir3_offline_to_nir(struct ir3_offline_context *ctx, const uint32_t *words,
                  size_t bytes, const char *entry, const VkSpecializationInfo *spec)
{
   VkShaderModuleCreateInfo module = {};
   module.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
   module.codeSize = bytes;
   module.pCode = words;
   VkPipelineShaderStageCreateInfo stage = {};
   stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
   stage.pNext = &module;
   stage.stage = mesa_to_vk_shader_stage(ctx->options.stage);
   stage.pName = entry;
   stage.pSpecializationInfo = spec;
   return tu_spirv_to_nir(&ctx->device, nullptr, 0, &stage, &ctx->key,
                         ctx->options.stage);
}

struct ir3_shader_variant *
ir3_offline_compile(struct ir3_offline_context *ctx, nir_shader *nir,
                    const struct ir3_offline_binding *bindings, unsigned count)
{
   tu_pipeline_layout layout = {};
   layout.push_constant_size = ctx->options.push_constant_size;
   bool ok = true;
   for (unsigned i = 0; i < count; i++)
      layout.num_sets = MAX2(layout.num_sets, bindings[i].set + 1);
   if (layout.num_sets > ctx->physical.usable_sets) {
      ralloc_free(nir);
      return nullptr;
   }
   for (unsigned set = 0; set < layout.num_sets; set++) {
      VkDescriptorSetLayoutBinding vk_bindings[256];
      unsigned n = 0;
      for (unsigned i = 0; i < count; i++)
         if (bindings[i].set == set)
            vk_bindings[n++] = bindings[i].vk;
      VkDescriptorSetLayoutCreateInfo create = {};
      create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      create.bindingCount = n;
      create.pBindings = vk_bindings;
      VkDescriptorSetLayout handle;
      if (tu_CreateDescriptorSetLayout(tu_device_to_handle(&ctx->device), &create,
                                       nullptr, &handle) != VK_SUCCESS) {
         ok = false;
         break;
      }
      layout.set[set].layout = tu_descriptor_set_layout_from_handle(handle);
   }
   if (ok) {
      tu_shader_info info = {};
      ir3_shader_key key = {};
      tu_lower_nir(&ctx->device, nir, &ctx->key, &key, &info);
      NIR_PASS(_, nir, nir_recompute_io_bases, nir_var_shader_in | nir_var_shader_out);
      ok = tu_shader_create(&ctx->device, &ctx->shader, nir, &ctx->key, &info,
                            &key, nullptr, 0, &layout, true) == VK_SUCCESS;
   } else {
      ralloc_free(nir);
   }
   for (unsigned set = 0; set < layout.num_sets; set++)
      if (layout.set[set].layout)
         vk_descriptor_set_layout_unref(&ctx->device.vk, &layout.set[set].layout->vk);
   return ok ? const_cast<ir3_shader_variant *>(ctx->shader->variant) : nullptr;
}

void
ir3_offline_context_destroy(struct ir3_offline_context *ctx)
{
   if (!ctx)
      return;
   if (ctx->shader) {
      ralloc_free(const_cast<ir3_shader_variant *>(ctx->shader->variant));
      ralloc_free(const_cast<ir3_shader_variant *>(ctx->shader->safe_const_variant));
      vk_free(&ctx->device.vk.alloc, ctx->shader);
   }
   tu_destroy_softfloat(&ctx->device);
   ir3_compiler_destroy(ctx->device.compiler);
   mtx_destroy(&ctx->device.softfloat_mutex);
   glsl_type_singleton_decref();
   free(ctx);
}
