/*
 * Copyright © 2026 PanVK contributors
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "libpan_csf_dgc.h"
#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_dgc.h"
#include "panvk_cmd_precomp.h"
#include "panvk_dgc.h"
#include "panvk_dgc_submit.h"
#include "panvk_entrypoints.h"
#include "panvk_precomp_cache.h"
#include "panvk_shader_instrumentation.h"

static_assert(sizeof(struct panlib_dgc_execution) % 64 == 0,
              "DGC variable execution payload starts on a cache line");
static_assert(PANLIB_DGC_INDEX_UINT16 == VK_INDEX_TYPE_UINT16 &&
                 PANLIB_DGC_INDEX_UINT32 == VK_INDEX_TYPE_UINT32 &&
                 PANLIB_DGC_INDEX_UINT8 == VK_INDEX_TYPE_UINT8,
              "DGC index token ABI matches VkIndexType");

static uint32_t
dgc_sequence_stride(void)
{
   return sizeof(struct panlib_dgc_sequence);
}

static uint64_t
dgc_sequence_offset(uint32_t max_sequences)
{
   return sizeof(struct panlib_dgc_header) +
          (uint64_t)max_sequences * PANLIB_DGC_SEQUENCE_CS_SIZE;
}

static uint64_t
dgc_preprocess_size(uint32_t max_sequences)
{
   return dgc_sequence_offset(max_sequences) +
          (uint64_t)max_sequences * dgc_sequence_stride();
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(GetGeneratedCommandsMemoryRequirementsEXT)(
   VkDevice device,
   const VkGeneratedCommandsMemoryRequirementsInfoEXT *info,
   VkMemoryRequirements2 *requirements)
{
   VK_FROM_HANDLE(panvk_device, dev, device);
   const struct panvk_physical_device *phys =
      to_panvk_physical_device(dev->vk.physical);

   requirements->memoryRequirements = (VkMemoryRequirements) {
      .size = dgc_preprocess_size(info->maxSequenceCount),
      .alignment = 64,
      .memoryTypeBits = BITFIELD_MASK(phys->memory.type_count),
   };
}

static void
preprocess(struct panvk_cmd_buffer *cmdbuf,
            const VkGeneratedCommandsInfoEXT *info,
            const struct panvk_cmd_buffer *state_cmdbuf)
{
   VK_FROM_HANDLE(panvk_indirect_command_layout, layout,
                  info->indirectCommandsLayout);
   const struct panvk_cmd_graphics_state *gfx = &state_cmdbuf->state.gfx;
   struct pan_ptr allocation = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, sizeof(struct panlib_dgc_preprocess), 64);
   if (!allocation.gpu)
      return;

   struct panlib_dgc_preprocess *params = allocation.cpu;
   *params = (struct panlib_dgc_preprocess) {
      .stream = info->indirectAddress,
      .count = info->sequenceCountAddress,
      .output = info->preprocessAddress,
      .sequence_offset = dgc_sequence_offset(info->maxSequenceCount),
      .sequence_stride = dgc_sequence_stride(),
      .max_sequences = info->maxSequenceCount,
      .stream_stride = layout->vk.stride,
      .token_count = layout->vk.token_count,
      .index_mode_is_dx = layout->vk.index_mode_is_dx,
      .execution_set_is_shaders = layout->vk.is_shaders,
      .scratch_register = PANVK_CS_REG_SCRATCH_START,
      .context_register = PANVK_CS_REG_SUBQUEUE_CTX_START,
      .program_table_offset =
         offsetof(struct panvk_cs_subqueue_context, dgc.program_table),
      .load_store_scoreboard = SB_ID(LS),
      .max_draw_count = info->maxDrawCount,
   };
   assert(layout->vk.token_count <= ARRAY_SIZE(params->tokens));
   memcpy(params->tokens, layout->tokens,
          layout->vk.token_count * sizeof(params->tokens[0]));

   for (unsigned s = 0; s < PANLIB_DGC_STAGE_COUNT; s++) {
      memcpy(params->initial.push_constants[s],
             state_cmdbuf->state.push_constants.dgc_stage_data[s],
             sizeof(params->initial.push_constants[s]));
   }

   if (info->shaderStages & VK_SHADER_STAGE_VERTEX_BIT) {
      const struct vk_dynamic_graphics_state *dyn =
         &state_cmdbuf->vk.dynamic_graphics_state;
      for (unsigned v = 0; v < PANLIB_DGC_MAX_VERTEX_BUFFERS; v++) {
         params->initial.vb[v] = (struct panlib_dgc_vertex_buffer) {
            .address = gfx->vb.bufs[v].address,
            .size = gfx->vb.bufs[v].size,
            .stride = dyn->vi_binding_strides[v],
         };
      }
      params->initial.ib = (struct panlib_dgc_index_buffer) {
         .address = gfx->ib.dev_addr,
         .size = MIN2(gfx->ib.size, UINT32_MAX),
         .type = gfx->ib.index_size == 4 ? VK_INDEX_TYPE_UINT32 :
                 gfx->ib.index_size == 1 ? VK_INDEX_TYPE_UINT8 :
                                          VK_INDEX_TYPE_UINT16,
      };
   }

   struct panvk_precomp_ctx ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   panlib_dgc_preprocess(&ctx,
                         panlib_1d(MAX2(DIV_ROUND_UP(info->maxSequenceCount, 64),
                                          1)),
                         PANLIB_BARRIER_NONE, allocation.gpu);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdPreprocessGeneratedCommandsEXT)(
   VkCommandBuffer commandBuffer, const VkGeneratedCommandsInfoEXT *info,
   VkCommandBuffer stateCommandBuffer)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_cmd_buffer, state_cmdbuf, stateCommandBuffer);
   preprocess(cmdbuf, info, state_cmdbuf);
}

static void
emit_idvs_once(struct panvk_cmd_buffer *cmdbuf, enum mali_index_type index_type)
{
   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, PANVK_SUBQUEUE_VERTEX_TILER);
   const struct cs_tracing_ctx *trace =
      &cmdbuf->state.cs[PANVK_SUBQUEUE_VERTEX_TILER].tracing;
   struct mali_primitive_flags_packed flags;
   pan_pack_nodefaults(&flags, PRIMITIVE_FLAGS, cfg)
      cfg.index_type = index_type;

   /* A DRAW token is vkCmdDraw, so DrawIndex is zero in every sequence.
    * SequenceIndex is independently written by the preprocessing shader.
    */
   cs_move32_to(b, cs_scratch_reg32(b, 9), 0);
#if PAN_ARCH >= 12
   cs_trace_run_idvs2(b, trace, cs_scratch_reg_tuple(b, 10, 4), flags.opaque[0],
                      true, cs_scratch_reg32(b, 9), MALI_IDVS_SHADING_MODE_EARLY);
#else
   cs_trace_run_idvs(b, trace, cs_scratch_reg_tuple(b, 10, 4), flags.opaque[0],
                     true, cs_shader_res_sel(0, 0, 1, 0),
                     cs_shader_res_sel(2, 2, 2, 0), cs_scratch_reg32(b, 9));
#endif
}

static void
emit_idvs(struct panvk_cmd_buffer *cmdbuf, enum mali_index_type index_type)
{
   const unsigned count = DIV_ROUND_UP(cmdbuf->state.gfx.render.layer_count,
                                       MAX_LAYERS_PER_TILER_DESC);
   if (count <= 1) {
      emit_idvs_once(cmdbuf, index_type);
      return;
   }

   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, PANVK_SUBQUEUE_VERTEX_TILER);
   struct cs_index counter = cs_scratch_reg32(b, 17);
   struct cs_index tiler = cs_sr_reg64(b, IDVS, TILER_CTX);
   cs_move32_to(b, counter, count);
   cs_while(b, MALI_CS_CONDITION_GREATER, counter) {
      emit_idvs_once(cmdbuf, index_type);
      cs_add_imm32(b, counter, counter, -1);
      cs_update_vt_ctx(b)
         cs_add_imm64(b, tiler, tiler, pan_size(TILER_CONTEXT));
   }
   cs_update_vt_ctx(b)
      cs_add_imm64(b, tiler, tiler, -(count * pan_size(TILER_CONTEXT)));
}

static void
emit_draw_program(struct panvk_cmd_buffer *cmdbuf, bool indexed)
{
   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, PANVK_SUBQUEUE_VERTEX_TILER);
   struct cs_index seq = cs_scratch_reg64(b, 0);
   struct cs_index exec = cs_scratch_reg64(b, 2);
   cs_add_imm64(b, exec, seq, 0);

   cs_update_vt_ctx(b) {
      cs_load64_to(b, cs_sr_reg64(b, IDVS, VERTEX_FAU), exec,
                   offsetof(struct panlib_dgc_execution, fau[PANLIB_DGC_VS]));
      cs_load64_to(b, cs_sr_reg64(b, IDVS, FRAGMENT_FAU), exec,
                   offsetof(struct panlib_dgc_execution, fau[PANLIB_DGC_FS]));
      cs_load64_to(b, cs_sr_reg64(b, IDVS, VERTEX_SRT), exec,
                   offsetof(struct panlib_dgc_execution, srt[PANLIB_DGC_VS]));
      cs_load64_to(b, cs_sr_reg64(b, IDVS, FRAGMENT_SRT), exec,
                   offsetof(struct panlib_dgc_execution, srt[PANLIB_DGC_FS]));
      const unsigned vs_spd = cmdbuf->state.gfx.idvs.prim == MESA_PRIM_POINTS ? 0 : 1;
#if PAN_ARCH >= 12
      cs_load64_to(b, cs_sr_reg64(b, IDVS, VERTEX_SPD), exec,
                   offsetof(struct panlib_dgc_execution, vs_spd) + vs_spd * 8);
      cs_load64_to(b, cs_sr_reg64(b, IDVS, VERTEX_TSD), exec,
                   offsetof(struct panlib_dgc_execution, tsd));
      cs_load64_to(b, cs_sr_reg64(b, IDVS, FRAGMENT_TSD), exec,
                   offsetof(struct panlib_dgc_execution, tsd));
#else
      cs_load64_to(b, cs_sr_reg64(b, IDVS, VERTEX_POS_SPD), exec,
                   offsetof(struct panlib_dgc_execution, vs_spd) + vs_spd * 8);
      cs_load64_to(b, cs_sr_reg64(b, IDVS, VERTEX_VARY_SPD), exec,
                   offsetof(struct panlib_dgc_execution, vs_spd[2]));
      cs_load64_to(b, cs_sr_reg64(b, IDVS, TSD_0), exec,
                   offsetof(struct panlib_dgc_execution, tsd));
#endif
      cs_load64_to(b, cs_sr_reg64(b, IDVS, FRAGMENT_SPD), exec,
                   offsetof(struct panlib_dgc_execution, fs_spd));
      cs_load64_to(b, cs_sr_reg64(b, IDVS, ZSD), exec,
                   offsetof(struct panlib_dgc_execution, zsd));
      cs_load32_to(b, cs_sr_reg32(b, IDVS, DCD0), exec,
                   offsetof(struct panlib_dgc_execution, dcd[0]));
      cs_load32_to(b, cs_sr_reg32(b, IDVS, DCD1), exec,
                   offsetof(struct panlib_dgc_execution, dcd[1]));
      cs_load32_to(b, cs_sr_reg32(b, IDVS, DCD2), exec,
                   offsetof(struct panlib_dgc_execution, dcd[2]));
      cs_load32_to(b, cs_sr_reg32(b, IDVS, TILER_FLAGS), exec,
                   offsetof(struct panlib_dgc_execution, tiler_flags));
      cs_load32_to(b, cs_sr_reg32(b, IDVS, VARY_SIZE), exec,
                   offsetof(struct panlib_dgc_execution, varying_size));
#if PAN_ARCH >= 13
      cs_load32_to(b, cs_sr_reg32(b, IDVS, LINE_WIDTH), exec,
                   offsetof(struct panlib_dgc_execution, primitive_size));
#else
      cs_load32_to(b, cs_sr_reg32(b, IDVS, PRIMITIVE_SIZE), exec,
                   offsetof(struct panlib_dgc_execution, primitive_size));
#endif
      cs_move32_to(b, cs_sr_reg32(b, IDVS, GLOBAL_ATTRIBUTE_OFFSET), 0);
      cs_load_to(b, cs_sr_reg_tuple(b, IDVS, INDEX_COUNT, 5), seq, 0x1f,
                 offsetof(struct panlib_dgc_execution, draw));
      if (indexed) {
         cs_load64_to(b, cs_sr_reg64(b, IDVS, INDEX_BUFFER), seq,
                      offsetof(struct panlib_dgc_execution, ib.address));
         cs_load32_to(b, cs_sr_reg32(b, IDVS, INDEX_BUFFER_SIZE), seq,
                      offsetof(struct panlib_dgc_execution, ib.size));
      }
      cs_move32_to(b, cs_sr_reg32(b, IDVS, INSTANCE_OFFSET), 0);
   }

   if (indexed) {
      cs_load32_to(b, cs_scratch_reg32(b, 8), seq,
                   offsetof(struct panlib_dgc_execution, ib.type));
   }
   cs_wait_slot(b, SB_ID(LS));
   struct cs_index nonempty = cs_scratch_reg32(b, 6);
   cs_umin32(b, nonempty, cs_sr_reg32(b, IDVS, INDEX_COUNT),
             cs_sr_reg32(b, IDVS, INSTANCE_COUNT));
   cs_if(b, MALI_CS_CONDITION_NEQUAL, nonempty) {
      if (!indexed) {
         emit_idvs(cmdbuf, MALI_INDEX_TYPE_NONE);
      } else {
         cs_match(b, cs_scratch_reg32(b, 8), cs_scratch_reg32(b, 7)) {
            cs_case(b, VK_INDEX_TYPE_UINT16)
               emit_idvs(cmdbuf, MALI_INDEX_TYPE_UINT16);
            cs_case(b, VK_INDEX_TYPE_UINT32)
               emit_idvs(cmdbuf, MALI_INDEX_TYPE_UINT32);
            cs_case(b, VK_INDEX_TYPE_UINT8)
               emit_idvs(cmdbuf, MALI_INDEX_TYPE_UINT8);
         }
      }
   }
}

static void
emit_dispatch_program(struct panvk_cmd_buffer *cmdbuf)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, PANVK_SUBQUEUE_COMPUTE);
   const struct cs_tracing_ctx *trace =
      &cmdbuf->state.cs[PANVK_SUBQUEUE_COMPUTE].tracing;
   struct cs_index seq = cs_scratch_reg64(b, 0);
   struct cs_index exec = cs_scratch_reg64(b, 2);

   /* The per-execute WLS allocation is reusable once the preceding dispatch
    * has completed. In particular, do not let iterator scoreboard rotation
    * turn sequential DGC dispatches into concurrent users of this WLS.
    */
   cs_wait_slots(b, dev->csf.sb.all_iters_mask);
   cs_add_imm64(b, exec, seq, 0);
   cs_update_compute_ctx(b) {
      cs_load64_to(b, cs_reg64(b, PANVK_COMPUTE_FAU), exec,
                   offsetof(struct panlib_dgc_execution, fau[PANLIB_DGC_CS]));
      cs_load64_to(b, cs_reg64(b, PANVK_COMPUTE_SRT), exec,
                   offsetof(struct panlib_dgc_execution, srt[PANLIB_DGC_CS]));
      cs_load64_to(b, cs_reg64(b, PANVK_COMPUTE_SPD), exec,
                   offsetof(struct panlib_dgc_execution, cs_spd));
      cs_load64_to(b, cs_reg64(b, PANVK_COMPUTE_TSD), exec,
                   offsetof(struct panlib_dgc_execution, tsd));
      cs_load32_to(b, cs_sr_reg32(b, COMPUTE, WG_SIZE), exec,
                   offsetof(struct panlib_dgc_execution, compute_size_workgroup));
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, GLOBAL_ATTRIBUTE_OFFSET), 0);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_X), 0);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Y), 0);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Z), 0);
      cs_load_to(b, cs_sr_reg_tuple(b, COMPUTE, JOB_SIZE_X, 3), seq, 7,
                 offsetof(struct panlib_dgc_execution, dispatch));
   }
   cs_wait_slot(b, SB_ID(LS));
   struct cs_index nonempty = cs_scratch_reg32(b, 4);
   cs_umin32(b, nonempty, cs_sr_reg32(b, COMPUTE, JOB_SIZE_X),
             cs_sr_reg32(b, COMPUTE, JOB_SIZE_Y));
   cs_umin32(b, nonempty, nonempty, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Z));
   cs_if(b, MALI_CS_CONDITION_NEQUAL, nonempty) {
      cs_next_iter_sb(cmdbuf, PANVK_SUBQUEUE_COMPUTE,
                      cs_scratch_reg_tuple(b, 0, 2));
      cs_trace_run_compute(b, trace, cs_scratch_reg_tuple(b, 0, 4),
                           1, MALI_TASK_AXIS_X, PANVK_COMPUTE_RES_SEL);
   }
}

static struct panlib_dgc_program
build_program(struct panvk_cmd_buffer *cmdbuf, bool compute, bool indexed)
{
   struct cs_builder *b = panvk_get_cs_builder(
      cmdbuf, compute ? PANVK_SUBQUEUE_COMPUTE : PANVK_SUBQUEUE_VERTEX_TILER);
   struct cs_builder saved = *b;
   cs_builder_init(b, &saved.conf, (struct cs_buffer){0});
   if (compute)
      emit_dispatch_program(cmdbuf);
   else
      emit_draw_program(cmdbuf, indexed);
   cs_end(b);
   struct panlib_dgc_program program = {0};
   if (cs_is_valid(b)) {
      program.address = cs_root_chunk_gpu_addr(b);
      program.size = cs_root_chunk_size(b);
   } else {
      vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   saved.req_resource_mask |= b->req_resource_mask;
   cs_builder_fini(b);
   *b = saved;
   return program;
}

static void
consume_submit_node(struct panvk_cmd_buffer *cmdbuf, enum panvk_subqueue_id subqueue)
{
   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, subqueue);
   struct cs_index node = cs_scratch_reg64(b, 0);
   struct cs_index next = cs_scratch_reg64(b, 2);
   cs_load64_to(b, node, cs_subqueue_ctx_reg(b),
                offsetof(struct panvk_cs_subqueue_context, dgc.next));
   cs_store64(b, node, cs_subqueue_ctx_reg(b),
              offsetof(struct panvk_cs_subqueue_context, dgc.current));
   cs_load64_to(b, next, node, subqueue == PANVK_SUBQUEUE_COMPUTE
      ? offsetof(struct panlib_dgc_submit_node, next_compute)
      : offsetof(struct panlib_dgc_submit_node, next_vt));
   cs_store64(b, next, cs_subqueue_ctx_reg(b),
              offsetof(struct panvk_cs_subqueue_context, dgc.next));
   cs_flush_stores(b);
}

static VkResult
prepare_execution_set_resources(struct panvk_cmd_buffer *cmdbuf,
                                 struct panlib_dgc_execute *params, bool compute)
{
   const struct panvk_descriptor_state *descs = compute
      ? &cmdbuf->state.compute.desc_state : &cmdbuf->state.gfx.desc_state;
   const unsigned first = compute ? PANLIB_DGC_CS : PANLIB_DGC_VS;
   const unsigned end = compute ? PANLIB_DGC_CS + 1 : PANLIB_DGC_FS + 1;
   STATIC_ASSERT(MAX_SETS + 1 <= PANLIB_DGC_RESOURCE_TABLE_SIZE);
   for (unsigned s = first; s < end; s++) {
      struct pan_ptr table = panvk_cmd_alloc_desc_array(
         cmdbuf, PANLIB_DGC_RESOURCE_TABLE_SIZE, RESOURCE);
      if (!table.gpu)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      struct mali_resource_packed *resources = table.cpu;
      memset(resources, 0, PANLIB_DGC_RESOURCE_TABLE_SIZE * pan_size(RESOURCE));
      for (unsigned i = 0; i < MAX_SETS; i++) {
         const struct panvk_descriptor_set *set = descs->sets[i];
         if (set) {
            pan_pack(&resources[i + 1], RESOURCE, cfg) {
               cfg.address = set->descs.dev;
               cfg.size = set->desc_count * PANVK_DESCRIPTOR_SIZE;
               cfg.contains_descriptors = true;
            }
         }
      }
      params->resource_table[s] = table.gpu | PANLIB_DGC_RESOURCE_TABLE_SIZE;
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdExecuteGeneratedCommandsEXT)(
   VkCommandBuffer commandBuffer, VkBool32 isPreprocessed,
   const VkGeneratedCommandsInfoEXT *info)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_indirect_command_layout, layout, info->indirectCommandsLayout);
   VK_FROM_HANDLE(panvk_indirect_execution_set, ies, info->indirectExecutionSet);
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const bool compute = info->shaderStages & VK_SHADER_STAGE_COMPUTE_BIT;
   const bool indexed = layout->vk.dgc_info & BITFIELD_BIT(MESA_VK_DGC_DRAW_INDEXED);

   if (cmdbuf->state.shader_instr)
      cmdbuf->state.shader_instr->next_result_index++;

   /* Shader-object binding and multi-draw-count tokens are optional and are
    * not included in the advertised binding stages/token properties.
    */
   if ((ies && ies->type != VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT) ||
       layout->vk.draw_count) {
      vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   if (!isPreprocessed) {
      preprocess(cmdbuf, info, cmdbuf);
      panvk_per_arch(cmd_signal_barrier)(cmdbuf, PANVK_CSF_BARRIER_WAIT);
   }

   struct pan_ptr allocation = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, sizeof(struct panlib_dgc_execute), 64);
   if (!allocation.gpu)
      return;
   struct panlib_dgc_execute *params = allocation.cpu;
   *params = (struct panlib_dgc_execute) {
      .preprocess = info->preprocessAddress,
      .sequence_offset = dgc_sequence_offset(info->maxSequenceCount),
      .sequence_stride = dgc_sequence_stride(),
   };
   if (ies) {
      struct panvk_descriptor_state *descs = compute
         ? &cmdbuf->state.compute.desc_state : &cmdbuf->state.gfx.desc_state;
      VkResult result = panvk_per_arch(cmd_prepare_push_descs)(
         cmdbuf, descs, BITFIELD_MASK(MAX_SETS));
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmdbuf->vk, result);
         return;
      }
   }
   VkResult result = compute
      ? panvk_per_arch(cmd_prepare_dgc_dispatch)(cmdbuf, params)
      : panvk_per_arch(cmd_prepare_dgc_draw)(cmdbuf, layout,
                                            info->maxSequenceCount, params);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmdbuf->vk, result);
      return;
   }

   struct pan_ptr pipeline = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, sizeof(struct panlib_dgc_execution_set_entry), 64);
   if (!pipeline.gpu)
      return;
   panvk_per_arch(dgc_fill_pipeline)(dev, pipeline.cpu,
      compute ? NULL : cmdbuf->state.gfx.vs.shader,
      compute ? NULL : get_fs(cmdbuf),
      compute ? cmdbuf->state.compute.shader : NULL);
   params->pipelines = pipeline.gpu;
   if (ies) {
      params->pipelines = ies->gpu_addr;
      params->pipeline_stride = ies->gpu_stride;
      result = prepare_execution_set_resources(cmdbuf, params, compute);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmdbuf->vk, result);
         return;
      }
   }

   struct panlib_dgc_program program = build_program(cmdbuf, compute, indexed);
   const uint32_t program_count = ies ? ies->capacity : 1;
   struct pan_ptr table = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, program_count * sizeof(program), 16);
   if (!table.gpu || !program.address)
      return;
   for (uint32_t i = 0; i < program_count; i++)
      ((struct panlib_dgc_program *)table.cpu)[i] = program;

   const struct panvk_dgc_job job = {
      .params = *params,
      .execution_set = ies,
      .fixed_pipeline = pipeline.cpu,
      .prepare_shader = panvk_per_arch(precomp_cache_get)(
         dev->precomp_cache, PANLIB_DGC_PREPARE_EXECUTION),
      .max_sequences = info->maxSequenceCount,
      .graphics = !compute,
   };
   if (!job.prepare_shader) {
      vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   result = panvk_per_arch(dgc_record_execute)(cmdbuf, &job);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmdbuf->vk, result);
      return;
   }

   struct panvk_precomp_ctx ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   ctx.dgc = true;
   consume_submit_node(cmdbuf, PANVK_SUBQUEUE_COMPUTE);
   panlib_dgc_prepare_execution(&ctx,
      panlib_1d(MAX2(DIV_ROUND_UP(info->maxSequenceCount, 64), 1)),
      PANLIB_BARRIER_CSF_WAIT, allocation.gpu);

   const enum panvk_subqueue_id subqueue =
      compute ? PANVK_SUBQUEUE_COMPUTE : PANVK_SUBQUEUE_VERTEX_TILER;
   {
      struct panvk_cs_deps deps = {0};
      deps.src[PANVK_SUBQUEUE_COMPUTE].wait_sb_mask = SB_MASK(LS);
      /* The prepare shader writes descriptors and FAUs as well as CSF.
       * Shader descriptor/instruction caches are not LS-coherent.
       */
      deps.src[PANVK_SUBQUEUE_COMPUTE].cache_flush.others =
         MALI_CS_OTHER_FLUSH_MODE_INVALIDATE;
      if (!compute)
         deps.dst[subqueue].wait_subqueue_mask =
            BITFIELD_BIT(PANVK_SUBQUEUE_COMPUTE);
      panvk_per_arch(emit_barrier)(cmdbuf, deps);
   }
   if (!compute) {
      consume_submit_node(cmdbuf, PANVK_SUBQUEUE_VERTEX_TILER);
   }
   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, subqueue);
   cs_move64_to(b, cs_scratch_reg64(b, 0), table.gpu);
   cs_store64(b, cs_scratch_reg64(b, 0), cs_subqueue_ctx_reg(b),
              offsetof(struct panvk_cs_subqueue_context, dgc.program_table));
   cs_flush_stores(b);

   panvk_cond_render(cmdbuf, b) {
      cs_load64_to(b, cs_scratch_reg64(b, 0), cs_subqueue_ctx_reg(b),
                   offsetof(struct panvk_cs_subqueue_context, dgc.current));
      cs_load64_to(b, cs_scratch_reg64(b, 0), cs_scratch_reg64(b, 0),
                   offsetof(struct panlib_dgc_submit_node, output));
      cs_load32_to(b, cs_scratch_reg32(b, 4), cs_scratch_reg64(b, 0),
                   offsetof(struct panlib_dgc_header, cs_size));
      cs_add_imm64(b, cs_scratch_reg64(b, 2), cs_scratch_reg64(b, 0),
                   sizeof(struct panlib_dgc_header));
      cs_wait_slot(b, SB_ID(LS));
      cs_if(b, MALI_CS_CONDITION_NEQUAL, cs_scratch_reg32(b, 4))
         cs_call(b, cs_scratch_reg64(b, 2), cs_scratch_reg32(b, 4));
   }

   /* GPU-selected buffer and push state is not represented by CPU dirty
    * tracking. Force subsequent ordinary draws/dispatches to restore it.
    */
   if (compute) {
      compute_state_set_dirty(cmdbuf, CS);
      compute_state_set_dirty(cmdbuf, DESC_STATE);
      compute_state_set_dirty(cmdbuf, PUSH_UNIFORMS);
   } else {
      cmdbuf->state.gfx.fs.dgc_execution_set = false;
      cmdbuf->state.gfx.tsd = 0;
      gfx_state_set_dirty(cmdbuf, VS);
      gfx_state_set_dirty(cmdbuf, FS);
      gfx_state_set_dirty(cmdbuf, IDVS);
      gfx_state_set_dirty(cmdbuf, VB);
      gfx_state_set_dirty(cmdbuf, DESC_STATE);
      gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);
      gfx_state_set_dirty(cmdbuf, FS_PUSH_UNIFORMS);
   }
}
