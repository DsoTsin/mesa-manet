/*
 * Copyright © 2026 PanVK contributors
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_DGC_H
#define PANVK_DGC_H

#include "compiler/shader_enums.h"
#include "libpan_csf_dgc_shader.h"
#include "panvk_macros.h"
#include "util/simple_mtx.h"
#include "vk_device_generated_commands.h"

#define PANVK_DGC_SHADER_STAGES                                               \
   (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |                \
    VK_SHADER_STAGE_COMPUTE_BIT)

struct panvk_shader;
struct panvk_device;
struct panvk_priv_bo;
struct panvk_dgc_shader_snapshot;

struct panvk_indirect_command_layout {
   struct vk_indirect_command_layout vk;

   /* The common layout retains the IES token offset and type, but not its
    * stage mask. Shader execution-set indices are packed in this order.
    */
   VkShaderStageFlags ies_stages;

   /* Keep API token order and stage-specific PC/SI destinations. */
   struct panlib_dgc_token tokens[PANLIB_DGC_MAX_TOKENS];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_indirect_command_layout, vk.base,
                             VkIndirectCommandsLayoutEXT,
                             VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_EXT)

struct panvk_indirect_execution_set_entry {
   /* Zero denotes an entry that has never been initialized. */
   VkShaderStageFlags stages;
   struct panvk_shader *shaders[MESA_SHADER_STAGES];
};

struct panvk_indirect_execution_set {
   struct vk_object_base base;

   VkIndirectExecutionSetInfoTypeEXT type;
   VkPipelineBindPoint bind_point;
   VkShaderStageFlags stages;
   uint32_t capacity;

   /* Slot addresses never change. Each slot occupies complete cache lines,
    * permitting updates while other slots are in use by the GPU.
    */
   struct panvk_priv_bo *gpu_bo;
   void *gpu_map;
   uint64_t gpu_addr;
   uint32_t gpu_stride;

   /* Submission scans every slot for TLS/WLS sizing, including slots an
    * application may legally update while unrelated slots are executing.
    * Serialize these internal host reads with updates, never with GPU work.
    */
   simple_mtx_t metadata_lock;

   /* Pipeline entries retain the compiled shaders through their pipeline
    * cache objects. Shader-object entries point into owned snapshots instead.
    * No application pipeline or shader handle is retained here.
    */
   struct panvk_indirect_execution_set_entry *entries;

   /* Initial shader state survives overwrites of the initial slots. */
   struct panvk_indirect_execution_set_entry initial;

   /* Only allocated for shader-object execution sets. Updates require no
    * allocation, because vkUpdateIndirectExecutionSetShaderEXT returns void.
    */
   struct panvk_dgc_shader_snapshot *shader_snapshots;
   struct panvk_dgc_shader_snapshot *initial_shader_snapshots;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_indirect_execution_set, base,
                             VkIndirectExecutionSetEXT,
                             VK_OBJECT_TYPE_INDIRECT_EXECUTION_SET_EXT)

#ifdef PAN_ARCH
void panvk_per_arch(dgc_fill_pipeline)(
   struct panvk_device *device, struct panlib_dgc_execution_set_entry *out,
   const struct panvk_shader *vs, const struct panvk_shader *fs,
   const struct panvk_shader *cs);
#endif

#endif /* PANVK_DGC_H */
