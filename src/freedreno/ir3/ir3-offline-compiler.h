#ifndef IR3_OFFLINE_COMPILER_H
#define IR3_OFFLINE_COMPILER_H

#include <vulkan/vulkan.h>
#include "ir3_shader.h"

BEGINC;

struct ir3_offline_binding {
   unsigned set;
   VkDescriptorSetLayoutBinding vk;
   bool overridden;
};

struct ir3_offline_options {
   struct fd_dev_id dev_id;
   mesa_shader_stage stage;
   unsigned subgroup_size;
   unsigned samples;
   unsigned push_constant_size;
   bool robust;
};

struct ir3_offline_context;

struct ir3_offline_context *
ir3_offline_context_create(const struct ir3_offline_options *options);

nir_shader *ir3_offline_to_nir(struct ir3_offline_context *ctx,
                             const uint32_t *words, size_t bytes,
                             const char *entry, const VkSpecializationInfo *spec);

struct ir3_shader_variant *
ir3_offline_compile(struct ir3_offline_context *ctx, nir_shader *nir,
                    const struct ir3_offline_binding *bindings, unsigned count);

void ir3_offline_context_destroy(struct ir3_offline_context *ctx);

ENDC;

#endif
