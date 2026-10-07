/*
 * Copyright © 2026 PanVK contributors
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "genxml/gen_macros.h"

#include "panvk_device.h"
#include "panvk_dgc.h"
#include "panvk_entrypoints.h"
#include "panvk_shader.h"

#include "pan_encoder.h"
#include "pan_format.h"

#include "vk_alloc.h"
#include "vk_pipeline.h"
#include "vk_pipeline_cache.h"
#include "vk_util.h"

static uint32_t
dgc_stages(VkShaderStageFlags stages)
{
   return ((stages & VK_SHADER_STAGE_VERTEX_BIT) ? 1u << PANLIB_DGC_VS : 0) |
          ((stages & VK_SHADER_STAGE_FRAGMENT_BIT) ? 1u << PANLIB_DGC_FS : 0) |
          ((stages & VK_SHADER_STAGE_COMPUTE_BIT) ? 1u << PANLIB_DGC_CS : 0);
}

static void
write_fau_recipe(struct panvk_device *device,
                 const struct panvk_shader_variant *shader,
                 struct panlib_dgc_shader *gpu)
{
   STATIC_ASSERT(PAN_MAX_PUSH <= PANLIB_DGC_MAX_FAU_WORDS);
   assert(shader->fau.total_count * 2 <= PANLIB_DGC_MAX_FAU_WORDS);
   gpu->fau_count = shader->fau.total_count;

   uint32_t dst = 0, w;
   BITSET_FOREACH_SET(w, shader->fau.used_sysvals, MAX_SYSVAL_FAUS) {
      const uint32_t offset = w * FAU_WORD_SIZE;
      for (uint32_t half = 0; half < 2; half++) {
         struct panlib_dgc_fau_word *word = &gpu->fau[dst++];
         if (offset == offsetof(struct panvk_common_sysvals,
                                common.push_uniforms)) {
            word->source = PANLIB_DGC_FAU_SELF;
            word->value = half * 4;
         } else if (offset == offsetof(struct panvk_common_sysvals,
                                       common.printf_buffer_address)) {
            /* prepare_push_uniforms() synthesizes the common sysvals;
             * they are not stored in the command buffer's sysval snapshot.
             */
            const uint64_t address = device->printf.bo
               ? device->printf.bo->addr.dev : 0;
            word->source = PANLIB_DGC_FAU_IMMEDIATE;
            word->value = address >> (half * 32);
         } else if (offset == offsetof(struct panvk_common_sysvals,
                                       common.instr_counters)) {
            word->source = PANLIB_DGC_FAU_IMMEDIATE;
            word->value = PAN_SHADER_OOB_ADDRESS >> (half * 32);
         } else {
            word->source = PANLIB_DGC_FAU_SYSVAL;
            word->value = offset + half * 4;
         }
      }
   }

   BITSET_FOREACH_SET(w, shader->fau.used_push_consts, MAX_PUSH_CONST_FAUS) {
      for (uint32_t half = 0; half < 2; half++) {
         gpu->fau[dst++] = (struct panlib_dgc_fau_word) {
            .source = PANLIB_DGC_FAU_PUSH_CONSTANT,
            .value = w * FAU_WORD_SIZE + half * 4,
         };
      }
   }
   assert(dst == shader->info.fau.reserved);

   pan_fau_foreach_imm(&shader->info.fau, i) {
      gpu->fau[i] = (struct panlib_dgc_fau_word) {
         .source = PANLIB_DGC_FAU_IMMEDIATE,
         .value = shader->info.fau.words[i].constant,
      };
   }

   /* The PanVK compiler only promotes immediates, not UBO loads. */
   pan_fau_foreach_reloc(&shader->info.fau, i)
      UNREACHABLE("Unexpected UBO relocation in a PanVK FAU layout");
}

static void
write_gpu_shader(struct panvk_device *device, const struct panvk_shader *shader,
                 struct panlib_dgc_shader *gpu)
{
   const struct panvk_shader_variant *variant =
      shader->vk.stage == MESA_SHADER_VERTEX
         ? panvk_shader_hw_variant(shader) : panvk_shader_only_variant(shader);
   const struct pan_shader_info *info = &variant->info;

   gpu->code = panvk_shader_variant_get_dev_addr(variant);
   gpu->tls_size = info->tls_size;
   gpu->wls_size = info->wls_size;
   gpu->work_reg_count = info->work_reg_count;
   gpu->outputs_written = info->outputs_written;
   gpu->used_set_mask = shader->desc_info.used_set_mask;
   STATIC_ASSERT(MAX_DYNAMIC_BUFFERS <= PANLIB_DGC_MAX_DYNAMIC_BUFFERS);
   gpu->dyn_buf_count = shader->desc_info.dyn_bufs.count;
   memcpy(gpu->dyn_buf_map, shader->desc_info.dyn_bufs.map,
          gpu->dyn_buf_count * sizeof(gpu->dyn_buf_map[0]));
   gpu->flags = (info->writes_global ? PANLIB_DGC_SHADER_WRITES_GLOBAL : 0) |
                (info->contains_barrier
                    ? PANLIB_DGC_SHADER_CONTAINS_BARRIER : 0);
   write_fau_recipe(device, variant, gpu);

   if (shader->vk.stage == MESA_SHADER_VERTEX) {
#if PAN_ARCH >= 12
      gpu->spd[0] = panvk_priv_mem_dev_addr(variant->spds.all_points);
      gpu->spd[1] = panvk_priv_mem_dev_addr(variant->spds.all_triangles);
#else
      gpu->spd[0] = panvk_priv_mem_dev_addr(variant->spds.pos_points);
      gpu->spd[1] = panvk_priv_mem_dev_addr(variant->spds.pos_triangles);
      gpu->spd[2] = panvk_priv_mem_dev_addr(variant->spds.var);
#endif
      gpu->flags |=
         (info->vs.writes_point_size ? PANLIB_DGC_VS_WRITES_POINT_SIZE : 0) |
         (info->vs.needs_extended_fifo ? PANLIB_DGC_VS_EXTENDED_FIFO : 0) |
         (info->vs.secondary_enable ? PANLIB_DGC_VS_SECONDARY : 0) |
         (info->vs.idvs ? PANLIB_DGC_VS_IDVS : 0);
   } else {
      gpu->spd[0] = panvk_priv_mem_dev_addr(variant->spd);
   }

   if (shader->vk.stage == MESA_SHADER_COMPUTE) {
      gpu->local_size[0] = variant->cs.local_size.x;
      gpu->local_size[1] = variant->cs.local_size.y;
      gpu->local_size[2] = variant->cs.local_size.z;
      gpu->allow_merging_workgroups = info->cs.allow_merging_workgroups;
      struct mali_compute_size_workgroup_packed wg;
      pan_pack(&wg, COMPUTE_SIZE_WORKGROUP, cfg) {
         cfg.workgroup_size_x = variant->cs.local_size.x;
         cfg.workgroup_size_y = variant->cs.local_size.y;
         cfg.workgroup_size_z = variant->cs.local_size.z;
         cfg.allow_merging_workgroups = info->cs.allow_merging_workgroups;
      }
      gpu->compute_size_workgroup = wg.opaque[0];
      gpu->workgroups_per_task = pan_calc_workgroups_per_task(
         &variant->cs.local_size, &device->kmod.dev->props,
         info->work_reg_count);
      return;
   }

   STATIC_ASSERT(PAN_MAX_VARYINGS <= PANLIB_DGC_MAX_VARYINGS);
   const struct pan_varying_layout *varyings = &info->varyings.formats;
   gpu->varying_count = varyings->count;
   gpu->varying_generic_size = varyings->generic_size_B;
   gpu->varying_noperspective = info->varyings.noperspective;
   for (uint32_t i = 0; i < varyings->count; i++) {
      const struct pan_varying_slot *slot = &varyings->slots[i];
      gpu->varying_slots[i] =
         ((uint32_t)slot->location << PANLIB_DGC_VARYING_LOCATION_SHIFT) |
         ((uint32_t)slot->alu_type << PANLIB_DGC_VARYING_ALU_TYPE_SHIFT) |
         ((uint32_t)slot->ncomps << PANLIB_DGC_VARYING_NCOMPS_SHIFT) |
         ((uint32_t)slot->section << PANLIB_DGC_VARYING_SECTION_SHIFT) |
         (((uint32_t)slot->offset & 0xfffu) << PANLIB_DGC_VARYING_OFFSET_SHIFT);
   }

   if (shader->vk.stage != MESA_SHADER_FRAGMENT)
      return;

   gpu->fs_varying_attr_desc_count =
      shader->desc_info.fs_varying_attr_desc_count;
   gpu->input_attachment_read = variant->fs.input_attachment_read;
   gpu->tile_image_color_read = variant->fs.tile_image_color_read;
   gpu->outputs_read = info->fs.outputs_read;
#define FS_FLAG(field, flag) if (info->fs.field) gpu->flags |= flag
   FS_FLAG(reads_frag_coord, PANLIB_DGC_FS_READS_FRAG_COORD);
   FS_FLAG(reads_point_coord, PANLIB_DGC_FS_READS_POINT_COORD);
   FS_FLAG(reads_primitive_id, PANLIB_DGC_FS_READS_PRIMITIVE_ID);
   FS_FLAG(reads_face, PANLIB_DGC_FS_READS_FACE);
   FS_FLAG(can_discard, PANLIB_DGC_FS_CAN_DISCARD);
   FS_FLAG(writes_depth, PANLIB_DGC_FS_WRITES_DEPTH);
   FS_FLAG(writes_stencil, PANLIB_DGC_FS_WRITES_STENCIL);
   FS_FLAG(writes_coverage, PANLIB_DGC_FS_WRITES_COVERAGE);
   FS_FLAG(sidefx, PANLIB_DGC_FS_SIDE_EFFECTS);
   FS_FLAG(sample_shading, PANLIB_DGC_FS_SAMPLE_SHADING);
   FS_FLAG(early_fragment_tests, PANLIB_DGC_FS_EARLY_FRAGMENT_TESTS);
   FS_FLAG(can_early_z, PANLIB_DGC_FS_CAN_EARLY_Z);
   FS_FLAG(can_fpk, PANLIB_DGC_FS_CAN_FPK);
   FS_FLAG(untyped_color_outputs, PANLIB_DGC_FS_UNTYPED_COLOR_OUTPUTS);
   FS_FLAG(hsr.ld_tile, PANLIB_DGC_FS_HSR_LD_TILE);
   FS_FLAG(hsr.wait_or_tile_access_before_atest_zsemit,
           PANLIB_DGC_FS_HSR_WAIT_OR_TILE_BEFORE_ATEST);
   FS_FLAG(hsr.rasterizer_coverage_read, PANLIB_DGC_FS_HSR_COVERAGE_READ);
   FS_FLAG(hsr.centroid_interpolation, PANLIB_DGC_FS_HSR_CENTROID_INTERPOLATION);
   FS_FLAG(hsr.varying_before_atest_zsemit,
           PANLIB_DGC_FS_HSR_VARYING_BEFORE_ATEST);
#undef FS_FLAG
   if (variant->fs.tile_image_z_read)
      gpu->flags |= PANLIB_DGC_FS_TILE_IMAGE_Z_READ;
   if (variant->fs.tile_image_s_read)
      gpu->flags |= PANLIB_DGC_FS_TILE_IMAGE_S_READ;

   uint32_t index = 0;
   for (uint32_t writes = 0; writes < 2; writes++) {
      for (uint32_t alpha = 0; alpha < 2; alpha++) {
         for (uint32_t passes = 0; passes < 2; passes++) {
            for (uint32_t read = 0; read < 3; read++) {
               struct pan_earlyzs_state zs = pan_earlyzs_get(
                  variant->fs.earlyzs_lut, writes, alpha, passes, read);
               gpu->earlyzs[index++] =
                  (zs.update << PANLIB_DGC_EARLYZS_UPDATE_SHIFT) |
                  (zs.kill << PANLIB_DGC_EARLYZS_KILL_SHIFT) |
                  (zs.shader_readonly_zs << PANLIB_DGC_EARLYZS_READONLY_SHIFT);
            }
         }
      }
   }
}

static void
write_fs_varying_descs(struct panlib_dgc_execution_set_entry *out,
                       const struct panvk_shader *vs_shader,
                       const struct panvk_shader *fs_shader)
{
   STATIC_ASSERT(pan_size(ATTRIBUTE) == sizeof(out->fs_varying_descs[0]));
   if (!vs_shader || !fs_shader ||
       !fs_shader->desc_info.fs_varying_attr_desc_count)
      return;

   const struct panvk_shader_variant *vs = panvk_shader_hw_variant(vs_shader);
   const struct panvk_shader_variant *fs = panvk_shader_only_variant(fs_shader);
   const struct pan_varying_layout *vs_layout = &vs->info.varyings.formats;
   const struct pan_varying_layout *fs_format = &fs->info.varyings.formats;
   pan_varying_layout_require_layout(vs_layout);
   pan_varying_layout_require_format(fs_format);

   for (uint32_t i = 0; i < fs_format->count; i++) {
      const struct pan_varying_slot *fs_slot =
         pan_varying_layout_slot_at(fs_format, i);
      if (!fs_slot || fs_slot->section != PAN_VARYING_SECTION_GENERIC)
         continue;

      unsigned offset = 0;
      enum pipe_format format = PIPE_FORMAT_NONE;
      const struct pan_varying_slot *vs_slot =
         pan_varying_layout_find_slot(vs_layout, fs_slot->location);
      if (vs_slot) {
         nir_alu_type base_type = nir_alu_type_get_base_type(fs_slot->alu_type);
         nir_alu_type bit_size = nir_alu_type_get_type_size(vs_slot->alu_type);
         offset = vs_slot->offset;
         format = pan_varying_format(base_type | bit_size, vs_slot->ncomps);
      }

      pan_cast_and_pack(out->fs_varying_descs[i], ATTRIBUTE, cfg) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_VERTEX_PACKET;
         cfg.offset_enable = false;
         cfg.format = GENX(pan_format_from_pipe_format)(format)->hw;
         cfg.table = 61;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_VERTEX;
         cfg.offset = 1024 + offset;
         cfg.buffer_index = PAN_ARCH >= 12 ? 1 : 0;
         cfg.attribute_stride = vs_layout->generic_size_B;
         cfg.packet_stride = vs_layout->generic_size_B + 16;
      }
   }
}

void
panvk_per_arch(dgc_fill_pipeline)(
   struct panvk_device *device, struct panlib_dgc_execution_set_entry *out,
   const struct panvk_shader *vs, const struct panvk_shader *fs,
   const struct panvk_shader *cs)
{
   memset(out, 0, sizeof(*out));
   const struct panvk_shader *shaders[PANLIB_DGC_STAGE_COUNT] = {vs, fs, cs};
   for (uint32_t i = 0; i < ARRAY_SIZE(shaders); i++) {
      if (shaders[i]) {
         write_gpu_shader(device, shaders[i], &out->shaders[i]);
         out->stages |= 1u << i;
      }
   }
   write_fs_varying_descs(out, vs, fs);
}

static void
write_gpu_entry(struct panvk_device *device,
                struct panvk_indirect_execution_set *set, uint32_t index,
                bool flush)
{
   assert(index < set->capacity);
   const size_t offset = (size_t)index * set->gpu_stride;
   struct panlib_dgc_execution_set_entry *gpu =
      (void *)((uint8_t *)set->gpu_map + offset);
   const struct panvk_indirect_execution_set_entry *entry = &set->entries[index];

   u_foreach_bit(stage, entry->stages)
      assert(entry->shaders[stage]->desc_info.dyn_bufs.count == 0);
   panvk_per_arch(dgc_fill_pipeline)(device, gpu,
                                    entry->shaders[MESA_SHADER_VERTEX],
                                    entry->shaders[MESA_SHADER_FRAGMENT],
                                    entry->shaders[MESA_SHADER_COMPUTE]);
   if (flush)
      panvk_priv_bo_flush(set->gpu_bo, offset, set->gpu_stride);
}

/* Shader objects do not have pipeline-cache references. Keep a private copy
 * of their immutable execution metadata and reference every GPU allocation it
 * addresses. Reserve both vertex variants so updates never need to allocate.
 */
struct panvk_dgc_shader_snapshot {
   uint8_t data[sizeof(struct panvk_shader) +
                PANVK_VS_VARIANTS * sizeof(struct panvk_shader_variant)]
      __attribute__((aligned(__alignof__(struct panvk_shader))));
};

static struct panvk_shader *
snapshot_shader(struct panvk_dgc_shader_snapshot *snapshot)
{
   return (struct panvk_shader *)snapshot->data;
}

static void
reference_mem(struct panvk_priv_mem *mem)
{
   struct panvk_priv_bo *bo = panvk_priv_mem_bo(*mem);
   if (!bo)
      return;

   panvk_priv_bo_ref(bo);
   /* This reference belongs to the snapshot, even for pool-owned memory. */
   mem->bo &= ~PANVK_PRIV_MEM_OWNED_BY_POOL;
}

static void
snapshot_init(struct panvk_dgc_shader_snapshot *snapshot,
              const struct panvk_shader *source)
{
   struct panvk_shader *shader = snapshot_shader(snapshot);
   const unsigned variants = panvk_shader_num_variants(source->vk.stage);
   assert(variants <= PANVK_VS_VARIANTS);

   memcpy(shader, source,
          sizeof(*shader) + variants * sizeof(shader->variants[0]));

   /* This is execution metadata, not another Vulkan shader object. In
    * particular, do not keep its application name or cache-object pointers.
    */
   shader->vk = (struct vk_shader) {
      .stage = source->vk.stage,
      .stack_size = source->vk.stack_size,
      .scratch_size = source->vk.scratch_size,
      .ray_queries = source->vk.ray_queries,
   };

   panvk_shader_foreach_variant(shader, variant) {
      variant->bin_ptr = NULL;
      variant->own_bin = false;
      variant->nir_str = NULL;
      variant->asm_str = NULL;

      reference_mem(&variant->code_mem);
      if (variant->info.stage != MESA_SHADER_VERTEX) {
         reference_mem(&variant->spd);
      } else {
#if PAN_ARCH >= 12
         reference_mem(&variant->spds.all_points);
         reference_mem(&variant->spds.all_triangles);
#else
         reference_mem(&variant->spds.pos_points);
         reference_mem(&variant->spds.pos_triangles);
         reference_mem(&variant->spds.var);
#endif
      }
   }
}

static void
snapshot_finish(struct panvk_shader *shader)
{
   panvk_shader_foreach_variant(shader, variant) {
      panvk_pool_free_mem(&variant->code_mem);
      if (variant->info.stage != MESA_SHADER_VERTEX) {
         panvk_pool_free_mem(&variant->spd);
      } else {
#if PAN_ARCH >= 12
         panvk_pool_free_mem(&variant->spds.all_points);
         panvk_pool_free_mem(&variant->spds.all_triangles);
#else
         panvk_pool_free_mem(&variant->spds.pos_points);
         panvk_pool_free_mem(&variant->spds.pos_triangles);
         panvk_pool_free_mem(&variant->spds.var);
#endif
      }
   }
}

static void
entry_finish(struct panvk_device *device,
             struct panvk_indirect_execution_set *set,
             struct panvk_indirect_execution_set_entry *entry)
{
   u_foreach_bit(s, entry->stages) {
      const mesa_shader_stage stage = vk_to_mesa_shader_stage(1u << s);
      struct panvk_shader *shader = entry->shaders[stage];
      assert(shader != NULL);
      if (set->type == VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT) {
         vk_pipeline_cache_object_unref(&device->vk,
                                        &shader->vk.pipeline.cache_obj);
      } else {
         snapshot_finish(shader);
      }
   }

   memset(entry, 0, sizeof(*entry));
}

static VkResult
entry_init_pipeline(struct panvk_device *device,
                    struct panvk_indirect_execution_set_entry *entry,
                    struct vk_pipeline *pipeline)
{
   assert(!(pipeline->stages & ~PANVK_DGC_SHADER_STAGES));
   assert(entry->stages == 0);

   /* Validate the complete stage list before taking any references. The
    * runtime also permits pipelines without get_shader (and libraries with
    * uncompiled stages), neither of which can populate this execution set.
    */
   struct vk_shader *shaders[MESA_SHADER_STAGES] = {0};
   u_foreach_bit(s, pipeline->stages) {
      const mesa_shader_stage stage = vk_to_mesa_shader_stage(1u << s);
      struct vk_shader *shader = vk_pipeline_get_shader(pipeline, stage);
      if (!shader || !shader->pipeline.cache_obj.ops)
         return panvk_error(device, VK_ERROR_UNKNOWN);

      assert(shader->stage == stage);
      shaders[stage] = shader;
   }

   for (unsigned stage = 0; stage < ARRAY_SIZE(shaders); stage++) {
      struct vk_shader *shader = shaders[stage];
      if (!shader)
         continue;

      /* Match vk_pipeline's own reference: keeping the cache object alive
       * retains shader metadata and every code/SPD allocation after the
       * originating VkPipeline and VkPipelineCache have been destroyed.
       */
      vk_pipeline_cache_object_ref(&shader->pipeline.cache_obj);
      entry->shaders[stage] = container_of(shader, struct panvk_shader, vk);
   }
   entry->stages = pipeline->stages;
   return VK_SUCCESS;
}

static void
entry_init_shader(struct panvk_indirect_execution_set_entry *entry,
                  struct panvk_dgc_shader_snapshot *snapshot,
                  struct vk_shader *shader)
{
   assert(mesa_to_vk_shader_stage(shader->stage) & PANVK_DGC_SHADER_STAGES);
   assert(!(entry->stages & mesa_to_vk_shader_stage(shader->stage)));

   snapshot_init(snapshot, container_of(shader, struct panvk_shader, vk));
   entry->shaders[shader->stage] = snapshot_shader(snapshot);
   entry->stages |= mesa_to_vk_shader_stage(shader->stage);
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CreateIndirectCommandsLayoutEXT)(
   VkDevice _device, const VkIndirectCommandsLayoutCreateInfoEXT *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkIndirectCommandsLayoutEXT *pIndirectCommandsLayout)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   *pIndirectCommandsLayout = VK_NULL_HANDLE;

   if (pCreateInfo->shaderStages & ~PANVK_DGC_SHADER_STAGES)
      return panvk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
   assert(pCreateInfo->tokenCount <= PANLIB_DGC_MAX_TOKENS);

   struct panvk_indirect_command_layout *layout =
      vk_indirect_command_layout_create(&device->vk, pCreateInfo, pAllocator,
                                        sizeof(*layout));
   if (!layout)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   for (uint32_t i = 0; i < pCreateInfo->tokenCount; i++) {
      const VkIndirectCommandsLayoutTokenEXT *token = &pCreateInfo->pTokens[i];
      struct panlib_dgc_token *out = &layout->tokens[i];
      out->offset = token->offset;
      out->stages = dgc_stages(pCreateInfo->shaderStages);

      switch (token->type) {
      case VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT:
         out->type = PANLIB_DGC_TOKEN_EXECUTION_SET;
         layout->ies_stages = token->data.pExecutionSet->shaderStages;
         out->stages = dgc_stages(layout->ies_stages);
         out->size = layout->vk.is_shaders
            ? util_bitcount(out->stages) * sizeof(uint32_t) : sizeof(uint32_t);
         break;
      case VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT:
         out->type = PANLIB_DGC_TOKEN_PUSH_CONSTANT;
         out->target = token->data.pPushConstant->updateRange.offset;
         out->size = token->data.pPushConstant->updateRange.size;
         out->stages =
            dgc_stages(token->data.pPushConstant->updateRange.stageFlags);
         break;
      case VK_INDIRECT_COMMANDS_TOKEN_TYPE_SEQUENCE_INDEX_EXT:
         out->type = PANLIB_DGC_TOKEN_SEQUENCE_INDEX;
         out->target = token->data.pPushConstant->updateRange.offset;
         out->size = sizeof(uint32_t);
         out->stages =
            dgc_stages(token->data.pPushConstant->updateRange.stageFlags);
         break;
      case VK_INDIRECT_COMMANDS_TOKEN_TYPE_INDEX_BUFFER_EXT:
         out->type = PANLIB_DGC_TOKEN_INDEX_BUFFER;
         out->size = sizeof(VkBindIndexBufferIndirectCommandEXT);
         break;
      case VK_INDIRECT_COMMANDS_TOKEN_TYPE_VERTEX_BUFFER_EXT:
         out->type = PANLIB_DGC_TOKEN_VERTEX_BUFFER;
         out->target = token->data.pVertexBuffer->vertexBindingUnit;
         out->size = sizeof(VkBindVertexBufferIndirectCommandEXT);
         break;
      case VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_EXT:
         out->type = PANLIB_DGC_TOKEN_DRAW;
         out->size = sizeof(VkDrawIndirectCommand);
         break;
      case VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_EXT:
         out->type = PANLIB_DGC_TOKEN_DRAW_INDEXED;
         out->size = sizeof(VkDrawIndexedIndirectCommand);
         break;
      case VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_COUNT_EXT:
         out->type = PANLIB_DGC_TOKEN_DRAW_COUNT;
         out->size = sizeof(VkDrawIndirectCountIndirectCommandEXT);
         break;
      case VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_COUNT_EXT:
         out->type = PANLIB_DGC_TOKEN_DRAW_INDEXED_COUNT;
         out->size = sizeof(VkDrawIndirectCountIndirectCommandEXT);
         break;
      case VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT:
         out->type = PANLIB_DGC_TOKEN_DISPATCH;
         out->size = sizeof(VkDispatchIndirectCommand);
         break;
      default:
         vk_indirect_command_layout_destroy(&device->vk, pAllocator,
                                             &layout->vk);
         return panvk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);
      }
   }

   *pIndirectCommandsLayout = panvk_indirect_command_layout_to_handle(layout);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(DestroyIndirectCommandsLayoutEXT)(
   VkDevice _device, VkIndirectCommandsLayoutEXT indirectCommandsLayout,
   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_indirect_command_layout, layout,
                  indirectCommandsLayout);
   if (layout)
      vk_indirect_command_layout_destroy(&device->vk, pAllocator, &layout->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CreateIndirectExecutionSetEXT)(
   VkDevice _device, const VkIndirectExecutionSetCreateInfoEXT *pCreateInfo,
   const VkAllocationCallbacks *pAllocator,
   VkIndirectExecutionSetEXT *pIndirectExecutionSet)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   *pIndirectExecutionSet = VK_NULL_HANDLE;

   uint32_t capacity;
   uint32_t initial_shader_count = 0;
   VkShaderStageFlags stages = 0;
   VkPipelineBindPoint bind_point;

   switch (pCreateInfo->type) {
   case VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT: {
      const VkIndirectExecutionSetPipelineInfoEXT *info =
         pCreateInfo->info.pPipelineInfo;
      VK_FROM_HANDLE(vk_pipeline, pipeline, info->initialPipeline);
      capacity = info->maxPipelineCount;
      stages = pipeline->stages;
      bind_point = pipeline->bind_point;
      break;
   }
   case VK_INDIRECT_EXECUTION_SET_INFO_TYPE_SHADER_OBJECTS_EXT: {
      const VkIndirectExecutionSetShaderInfoEXT *info =
         pCreateInfo->info.pShaderInfo;
      capacity = info->maxShaderCount;
      initial_shader_count = info->shaderCount;
      for (uint32_t i = 0; i < info->shaderCount; i++) {
         VK_FROM_HANDLE(vk_shader, shader, info->pInitialShaders[i]);
         stages |= mesa_to_vk_shader_stage(shader->stage);
      }
      bind_point = stages == VK_SHADER_STAGE_COMPUTE_BIT
                      ? VK_PIPELINE_BIND_POINT_COMPUTE
                      : VK_PIPELINE_BIND_POINT_GRAPHICS;
      break;
   }
   default:
      UNREACHABLE("Invalid indirect execution set type");
   }

   if (stages & ~PANVK_DGC_SHADER_STAGES)
      return panvk_error(device, VK_ERROR_FEATURE_NOT_PRESENT);

   assert(capacity > 0 && capacity >= initial_shader_count);
   const bool shader_objects = pCreateInfo->type ==
      VK_INDIRECT_EXECUTION_SET_INFO_TYPE_SHADER_OBJECTS_EXT;
   const size_t entry_size =
      sizeof(struct panvk_indirect_execution_set_entry) +
      (shader_objects ? sizeof(struct panvk_dgc_shader_snapshot) : 0);
   const size_t fixed_size = sizeof(struct panvk_indirect_execution_set) +
      (size_t)initial_shader_count * sizeof(struct panvk_dgc_shader_snapshot);
   if (capacity > (SIZE_MAX - fixed_size) / entry_size)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   VK_MULTIALLOC(ma);
   VK_MULTIALLOC_DECL(&ma, struct panvk_indirect_execution_set, set, 1);
   VK_MULTIALLOC_DECL(&ma, struct panvk_indirect_execution_set_entry, entries,
                     (size_t)capacity);
   VK_MULTIALLOC_DECL(&ma, struct panvk_dgc_shader_snapshot, snapshots,
                     shader_objects ? (size_t)capacity : 0);
   VK_MULTIALLOC_DECL(&ma, struct panvk_dgc_shader_snapshot, initial_snapshots,
                     (size_t)initial_shader_count);

   set = vk_multialloc_zalloc2(&ma, &device->vk.alloc, pAllocator,
                               VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!set)
      return panvk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);

   vk_object_base_init(&device->vk, &set->base,
                       VK_OBJECT_TYPE_INDIRECT_EXECUTION_SET_EXT);
   simple_mtx_init(&set->metadata_lock, mtx_plain);
   set->type = pCreateInfo->type;
   set->bind_point = bind_point;
   set->stages = stages;
   set->capacity = capacity;
   set->entries = entries;
   set->shader_snapshots = shader_objects ? snapshots : NULL;
   set->initial_shader_snapshots = shader_objects ? initial_snapshots : NULL;

   const uint32_t cache_line = MAX2(
      64, device->vk.physical->properties.nonCoherentAtomSize);
   set->gpu_stride = ALIGN_POT(
      sizeof(struct panlib_dgc_execution_set_entry), cache_line);
   const uint64_t gpu_size = (uint64_t)capacity * set->gpu_stride;
   const uint32_t bo_flags =
      panvk_device_adjust_bo_flags(device, PAN_KMOD_BO_FLAG_WB_MMAP);
   VkResult result = panvk_priv_bo_create(device, gpu_size, bo_flags,
                                          VK_SYSTEM_ALLOCATION_SCOPE_OBJECT,
                                          &set->gpu_bo);
   if (result != VK_SUCCESS) {
      simple_mtx_destroy(&set->metadata_lock);
      vk_object_base_finish(&set->base);
      vk_free2(&device->vk.alloc, pAllocator, set);
      return result;
   }
   set->gpu_addr = set->gpu_bo->addr.dev;
   set->gpu_map = set->gpu_bo->addr.host;
   memset(set->gpu_map, 0, gpu_size);

   if (!shader_objects) {
      VK_FROM_HANDLE(vk_pipeline, pipeline,
                     pCreateInfo->info.pPipelineInfo->initialPipeline);
      result = entry_init_pipeline(device, &set->entries[0], pipeline);
      if (result != VK_SUCCESS)
         goto fail_entries;
      result = entry_init_pipeline(device, &set->initial, pipeline);
      if (result != VK_SUCCESS)
         goto fail_entries;
      write_gpu_entry(device, set, 0, false);
   } else {
      const VkIndirectExecutionSetShaderInfoEXT *info =
         pCreateInfo->info.pShaderInfo;
      for (uint32_t i = 0; i < info->shaderCount; i++) {
         VK_FROM_HANDLE(vk_shader, shader, info->pInitialShaders[i]);
         entry_init_shader(&set->entries[i], &snapshots[i], shader);
         entry_init_shader(&set->initial, &initial_snapshots[i], shader);
         write_gpu_entry(device, set, i, false);
      }
   }
   panvk_priv_bo_flush(set->gpu_bo, 0, gpu_size);

   *pIndirectExecutionSet = panvk_indirect_execution_set_to_handle(set);
   return VK_SUCCESS;

fail_entries:
   entry_finish(device, set, &set->entries[0]);
   entry_finish(device, set, &set->initial);
   panvk_priv_bo_unref(set->gpu_bo);
   simple_mtx_destroy(&set->metadata_lock);
   vk_object_base_finish(&set->base);
   vk_free2(&device->vk.alloc, pAllocator, set);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(DestroyIndirectExecutionSetEXT)(
   VkDevice _device, VkIndirectExecutionSetEXT indirectExecutionSet,
   const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_indirect_execution_set, set, indirectExecutionSet);
   if (!set)
      return;

   for (uint32_t i = 0; i < set->capacity; i++)
      entry_finish(device, set, &set->entries[i]);
   entry_finish(device, set, &set->initial);
   panvk_priv_bo_unref(set->gpu_bo);

   simple_mtx_destroy(&set->metadata_lock);
   vk_object_base_finish(&set->base);
   vk_free2(&device->vk.alloc, pAllocator, set);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(UpdateIndirectExecutionSetPipelineEXT)(
   VkDevice _device, VkIndirectExecutionSetEXT indirectExecutionSet,
   uint32_t executionSetWriteCount,
   const VkWriteIndirectExecutionSetPipelineEXT *pExecutionSetWrites)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_indirect_execution_set, set, indirectExecutionSet);
   assert(set->type == VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT);

   for (uint32_t i = 0; i < executionSetWriteCount; i++) {
      const VkWriteIndirectExecutionSetPipelineEXT *write =
         &pExecutionSetWrites[i];
      VK_FROM_HANDLE(vk_pipeline, pipeline, write->pipeline);
      assert(write->index < set->capacity);
      assert(pipeline->bind_point == set->bind_point);
      assert(pipeline->stages == set->stages);

      struct panvk_indirect_execution_set_entry entry = {0};
      if (entry_init_pipeline(device, &entry, pipeline) != VK_SUCCESS)
         return;
      simple_mtx_lock(&set->metadata_lock);
      entry_finish(device, set, &set->entries[write->index]);
      set->entries[write->index] = entry;
      write_gpu_entry(device, set, write->index, true);
      simple_mtx_unlock(&set->metadata_lock);
   }
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(UpdateIndirectExecutionSetShaderEXT)(
   VkDevice _device, VkIndirectExecutionSetEXT indirectExecutionSet,
   uint32_t executionSetWriteCount,
   const VkWriteIndirectExecutionSetShaderEXT *pExecutionSetWrites)
{
   VK_FROM_HANDLE(panvk_device, device, _device);
   VK_FROM_HANDLE(panvk_indirect_execution_set, set, indirectExecutionSet);
   assert(set->type == VK_INDIRECT_EXECUTION_SET_INFO_TYPE_SHADER_OBJECTS_EXT);

   for (uint32_t i = 0; i < executionSetWriteCount; i++) {
      const VkWriteIndirectExecutionSetShaderEXT *write =
         &pExecutionSetWrites[i];
      VK_FROM_HANDLE(vk_shader, shader, write->shader);
      assert(write->index < set->capacity);
      assert(mesa_to_vk_shader_stage(shader->stage) & set->stages);

      simple_mtx_lock(&set->metadata_lock);
      entry_finish(device, set, &set->entries[write->index]);
      entry_init_shader(&set->entries[write->index],
                        &set->shader_snapshots[write->index], shader);
      write_gpu_entry(device, set, write->index, true);
      simple_mtx_unlock(&set->metadata_lock);
   }
}
