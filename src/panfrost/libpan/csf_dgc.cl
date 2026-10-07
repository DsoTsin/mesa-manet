/*
 * Copyright © 2026 PanVK contributors
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "libpan_csf_dgc.h"
#include "libpan_csf_dgc_execute.h"
#include "genxml/gen_macros.h"
#include "lib/pan_encoder.h"

/* Application token offsets need only four-byte alignment. In particular a
 * VkBindVertexBufferIndirectCommandEXT at offset 4 has an unaligned address.
 * Do not turn these stream accesses into aligned 64-bit loads.
 */
static uint64_t
load_address(global const uint32_t *words)
{
   return (uint64_t)words[0] | ((uint64_t)words[1] << 32);
}

static void
emit_sequence_call(global uint64_t *cs, uint64_t sequence, uint32_t pipeline,
                   constant struct panlib_dgc_preprocess *info)
{
   const uint32_t reg = info->scratch_register;

   /* The callee receives the decoded sequence address in scratch[0:1]. The
    * context register belongs to the queue and survives nested CALLs. No
    * mutable execution state is baked into this preprocessed CSF block.
    */
   pan_pack((global struct mali_cs_move32_packed *)&cs[0], CS_MOVE32, cfg) {
      cfg.destination = reg;
      cfg.immediate = sequence;
   }
   pan_pack((global struct mali_cs_move32_packed *)&cs[1], CS_MOVE32, cfg) {
      cfg.destination = reg + 1;
      cfg.immediate = sequence >> 32;
   }
   pan_pack((global struct mali_cs_load_multiple_packed *)&cs[2],
            CS_LOAD_MULTIPLE, cfg) {
      cfg.base_register = reg + 2;
      cfg.address = info->context_register;
      cfg.mask = 3;
      cfg.offset = info->program_table_offset;
   }
   pan_pack((global struct mali_cs_wait_packed *)&cs[3], CS_WAIT, cfg) {
      cfg.wait_mask = 1u << info->load_store_scoreboard;
   }
   pan_pack((global struct mali_cs_add_imm64_packed *)&cs[4], CS_ADD_IMM64, cfg) {
      cfg.source = reg + 2;
      cfg.destination = reg + 2;
      cfg.immediate = pipeline * sizeof(struct panlib_dgc_program);
   }
   pan_pack((global struct mali_cs_load_multiple_packed *)&cs[5],
            CS_LOAD_MULTIPLE, cfg) {
      cfg.base_register = reg + 4;
      cfg.address = reg + 2;
      cfg.mask = 7;
   }
   pan_pack((global struct mali_cs_wait_packed *)&cs[6], CS_WAIT, cfg) {
      cfg.wait_mask = 1u << info->load_store_scoreboard;
   }
   pan_pack((global struct mali_cs_call_packed *)&cs[7], CS_CALL, cfg) {
      cfg.address = reg + 4;
      cfg.length = reg + 6;
   }
}

KERNEL(64)
panlib_dgc_preprocess(constant struct panlib_dgc_preprocess *info)
{
   const uint32_t idx = cl_global_id.x;
   const uint32_t count = info->count
      ? min(*(global const uint32_t *)info->count, info->max_sequences)
      : info->max_sequences;
   global struct panlib_dgc_header *header =
      (global struct panlib_dgc_header *)info->output;

   if (idx == 0) {
      header->sequence_count = count;
      header->cs_size = count * PANLIB_DGC_SEQUENCE_CS_SIZE;
   }
   if (idx >= count)
      return;

   global const uint8_t *stream = (global const uint8_t *)info->stream +
      (uint64_t)idx * info->stream_stride;
   global struct panlib_dgc_sequence *seq =
      (global struct panlib_dgc_sequence *)(info->output +
         info->sequence_offset + (uint64_t)idx * info->sequence_stride);
   *seq = info->initial;
   seq->index = idx;

   for (uint32_t t = 0; t < info->token_count; t++) {
      constant struct panlib_dgc_token *token = &info->tokens[t];
      global const uint32_t *words =
         (global const uint32_t *)(stream + token->offset);

      switch (token->type) {
      case PANLIB_DGC_TOKEN_EXECUTION_SET:
         if (info->execution_set_is_shaders) {
            uint32_t word = 0;
            for (uint32_t s = 0; s < PANLIB_DGC_STAGE_COUNT; s++) {
               if (token->stages & (1u << s))
                  seq->execution_set[s] = words[word++];
            }
         } else {
            seq->execution_set[0] = words[0];
         }
         break;
      case PANLIB_DGC_TOKEN_PUSH_CONSTANT:
      case PANLIB_DGC_TOKEN_SEQUENCE_INDEX:
         for (uint32_t s = 0; s < PANLIB_DGC_STAGE_COUNT; s++) {
            if (!(token->stages & (1u << s)))
               continue;
            for (uint32_t w = 0; w < token->size / 4; w++) {
               seq->push_constants[s][token->target / 4 + w] =
                  token->type == PANLIB_DGC_TOKEN_SEQUENCE_INDEX ? idx : words[w];
            }
         }
         break;
      case PANLIB_DGC_TOKEN_VERTEX_BUFFER:
         seq->vb[token->target].address = load_address(words);
         seq->vb[token->target].size = words[2];
         seq->vb[token->target].stride = words[3];
         break;
      case PANLIB_DGC_TOKEN_INDEX_BUFFER:
         seq->ib.address = load_address(words);
         seq->ib.size = words[2];
         /* DXGI_FORMAT_R8_UINT = 62, R16_UINT = 57, R32_UINT = 42. */
         seq->ib.type = !info->index_mode_is_dx ? words[3] :
            words[3] == 62 ? PANLIB_DGC_INDEX_UINT8 :
            words[3] == 57 ? PANLIB_DGC_INDEX_UINT16 : PANLIB_DGC_INDEX_UINT32;
         break;
      case PANLIB_DGC_TOKEN_DRAW:
         seq->draw[0] = words[0];
         seq->draw[1] = words[1];
         seq->draw[2] = 0;
         seq->draw[3] = words[2];
         seq->draw[4] = words[3];
         break;
      case PANLIB_DGC_TOKEN_DRAW_INDEXED:
         for (uint32_t w = 0; w < 5; w++)
            seq->draw[w] = words[w];
         break;
      case PANLIB_DGC_TOKEN_DRAW_COUNT:
      case PANLIB_DGC_TOKEN_DRAW_INDEXED_COUNT:
         seq->draw_count_address = load_address(words);
         seq->draw_count_stride = words[2];
         seq->draw_count = min(words[3], info->max_draw_count);
         break;
      case PANLIB_DGC_TOKEN_DISPATCH:
         for (uint32_t w = 0; w < 3; w++)
            seq->dispatch[w] = words[w];
         break;
      }
   }

   global uint64_t *cs = (global uint64_t *)(header + 1) +
      idx * (PANLIB_DGC_SEQUENCE_CS_SIZE / sizeof(uint64_t));
   emit_sequence_call(cs, (uint64_t)seq, seq->execution_set[0], info);
}

static uint32_t
sequence_sysval(constant struct panlib_dgc_execute *info,
                global const struct panlib_dgc_sequence *seq,
                global const struct panlib_dgc_execution_set_entry *pipeline,
                uint32_t stage, uint32_t offset)
{
   if (stage == PANLIB_DGC_VS) {
      if (offset == info->first_vertex_sysval)
         return seq->draw[3];
      if (offset == info->base_instance_sysval)
         return seq->draw[4];
      if (offset == info->noperspective_sysval)
         return pipeline->shaders[PANLIB_DGC_FS].varying_noperspective;
   } else if (stage == PANLIB_DGC_CS &&
              offset >= info->num_workgroups_sysval &&
              offset < info->num_workgroups_sysval + 12) {
      return seq->dispatch[(offset - info->num_workgroups_sysval) / 4];
   }
   if (stage == PANLIB_DGC_CS && offset >= info->local_group_size_sysval &&
       offset < info->local_group_size_sysval + 12)
      return pipeline->shaders[stage].local_size[
         (offset - info->local_group_size_sysval) / 4];

   return ((global const uint32_t *)info->sysvals[stage])[offset / 4];
}

static void
prepare_graphics_registers(constant struct panlib_dgc_execute *info,
                           global const struct panlib_dgc_execution_set_entry *pipeline,
                           global struct panlib_dgc_execution *exec)
{
   global const struct panlib_dgc_shader *vs = &pipeline->shaders[PANLIB_DGC_VS];
   global const struct panlib_dgc_shader *fs = &pipeline->shaders[PANLIB_DGC_FS];
   const bool fs_enabled = info->fs_enabled &&
                          (pipeline->stages & (1u << PANLIB_DGC_FS));
   const bool reads_z = (fs->input_attachment_read & info->depth_input_mask) ||
                       (fs->flags & PANLIB_DGC_FS_TILE_IMAGE_Z_READ);
   const bool reads_s = (fs->input_attachment_read & info->stencil_input_mask) ||
                       (fs->flags & PANLIB_DGC_FS_TILE_IMAGE_S_READ);
   const uint32_t zs_read_mode = reads_z || reads_s || info->force_late_zs ? 1 : 0;
   const uint32_t earlyzs = fs->earlyzs[info->earlyzs_index + zs_read_mode];
   uint32_t written = 0;
   uint32_t read = fs->tile_image_color_read & info->render_target_mask;
   for (uint32_t i = 0; i < 8; i++) {
      if ((fs->outputs_written >> info->color_output_shift) & info->color_output_map[i])
         written |= 1u << i;
      if (fs->input_attachment_read & info->input_attachment_mask[i])
         read |= 1u << i;
   }

   pan_unpack((constant struct mali_dcd_flags_0_packed *)&info->dcd[0],
               DCD_FLAGS_0, dcd0);
   pan_pack((global struct mali_dcd_flags_0_packed *)&exec->dcd[0], DCD_FLAGS_0, cfg) {
      cfg = dcd0;
      if (fs_enabled) {
         /* Disabling FPK is conservative across arbitrary execution-set
          * shaders and avoids deriving optimization state from the initial
          * shader's render-target reads or blend lowering.
          */
         cfg.allow_forward_pixel_to_kill = false;
         cfg.allow_forward_pixel_to_be_killed = !(fs->flags & PANLIB_DGC_SHADER_WRITES_GLOBAL);
         cfg.pixel_kill_operation = (earlyzs >> PANLIB_DGC_EARLYZS_KILL_SHIFT) & 3;
         cfg.zs_update_operation = (earlyzs >> PANLIB_DGC_EARLYZS_UPDATE_SHIFT) & 3;
         cfg.evaluate_per_sample = info->per_sample &&
            ((fs->flags & PANLIB_DGC_FS_SAMPLE_SHADING) || info->blend_shader);
         cfg.shader_modifies_coverage = info->alpha_to_coverage ||
            (fs->flags & (PANLIB_DGC_FS_CAN_DISCARD | PANLIB_DGC_FS_WRITES_COVERAGE));
      }
   }
   pan_unpack((constant struct mali_dcd_flags_1_packed *)&info->dcd[1],
               DCD_FLAGS_1, dcd1);
   pan_pack((global struct mali_dcd_flags_1_packed *)&exec->dcd[1], DCD_FLAGS_1, cfg) {
      cfg = dcd1;
      cfg.render_target_mask = fs_enabled ? written : 0;
   }
   pan_unpack((constant struct mali_dcd_flags_2_packed *)&info->dcd[2],
               DCD_FLAGS_2, dcd2);
   pan_pack((global struct mali_dcd_flags_2_packed *)&exec->dcd[2], DCD_FLAGS_2, cfg) {
      cfg = dcd2;
      cfg.read_mask = fs_enabled ? read : 0;
      cfg.write_mask = fs_enabled ? written : 0;
#if PAN_ARCH >= 11
      cfg.no_shader_depth_read = !fs_enabled || !reads_z;
      cfg.no_shader_stencil_read = !fs_enabled || !reads_s;
#endif
   }

   pan_unpack((constant struct mali_primitive_flags_packed *)&info->tiler_flags,
               PRIMITIVE_FLAGS, primitive);
   pan_pack((global struct mali_primitive_flags_packed *)&exec->tiler_flags,
            PRIMITIVE_FLAGS, cfg) {
      cfg = primitive;
      const bool writes_psiz = info->point_primitive &&
                               (vs->flags & PANLIB_DGC_VS_WRITES_POINT_SIZE);
      cfg.position_fifo_format = writes_psiz || (vs->flags & PANLIB_DGC_VS_EXTENDED_FIFO)
         ? MALI_FIFO_FORMAT_EXTENDED : MALI_FIFO_FORMAT_BASIC;
#if PAN_ARCH < 13
      cfg.point_size_array_format = writes_psiz ? MALI_POINT_SIZE_ARRAY_FORMAT_FP16
                                               : MALI_POINT_SIZE_ARRAY_FORMAT_NONE;
#endif
      cfg.layer_index_enable = (vs->outputs_written & info->layer_output_mask) != 0;
      cfg.primitive_index_enable = fs_enabled && (fs->flags & PANLIB_DGC_FS_READS_PRIMITIVE_ID);
      cfg.primitive_index_override = cfg.primitive_index_enable &&
         (vs->outputs_written & info->primitive_id_output_mask);
      cfg.secondary_shader = fs_enabled && (vs->flags & PANLIB_DGC_VS_SECONDARY);
   }

   pan_unpack((global const struct mali_depth_stencil_packed *)info->depth_stencil,
               DEPTH_STENCIL, zs);
   global struct mali_depth_stencil_packed *depth_stencil =
      (global struct mali_depth_stencil_packed *)((uint64_t)exec +
         info->depth_stencil_offset);
   pan_pack(depth_stencil,
            DEPTH_STENCIL, cfg) {
      cfg = zs;
#if PAN_ARCH == 10
      cfg.shader_read_only_z_s = fs_enabled &&
         ((earlyzs >> PANLIB_DGC_EARLYZS_READONLY_SHIFT) & 1);
#endif
   }
   exec->zsd = (uint64_t)depth_stencil;
   exec->varying_size = vs->varying_generic_size;
   exec->primitive_size = info->primitive_size;
}

KERNEL(64)
panlib_dgc_prepare_execution(constant struct panlib_dgc_execute *info)
{
   const uint32_t idx = cl_global_id.x;
   global const struct panlib_dgc_header *header =
      (global const struct panlib_dgc_header *)info->preprocess;
   if (idx == 0)
      *(global struct panlib_dgc_header *)info->output = *header;
   if (idx >= header->sequence_count)
      return;

   global const struct panlib_dgc_sequence *seq =
      (global const struct panlib_dgc_sequence *)(info->preprocess +
         info->sequence_offset + (uint64_t)idx * info->sequence_stride);
   global struct panlib_dgc_execution *exec =
      (global struct panlib_dgc_execution *)(info->output + info->sequence_offset +
         (uint64_t)idx * info->output_sequence_stride);
   for (uint32_t i = 0; i < 5; i++)
      exec->draw[i] = seq->draw[i];
   for (uint32_t i = 0; i < 3; i++)
      exec->dispatch[i] = seq->dispatch[i];
   exec->ib = seq->ib;

   /* Application preprocess memory is consumed in DRAW_INDIRECT. Shaders
    * may outlive that stage (notably fragments of an open render pass), so
    * they must never retain pointers into that application-owned buffer.
    */
   global const uint64_t *input_cs = (global const uint64_t *)(header + 1) + idx * 8;
   global uint64_t *output_cs =
      (global uint64_t *)(info->output + sizeof(*header)) + idx * 8;
   for (uint32_t i = 0; i < 8; i++)
      output_cs[i] = input_cs[i];
   pan_unpack((global const struct mali_cs_move32_packed *)&input_cs[0],
               CS_MOVE32, move_lo);
   pan_pack((global struct mali_cs_move32_packed *)&output_cs[0], CS_MOVE32, cfg) {
      cfg = move_lo;
      cfg.immediate = (uint64_t)exec;
   }
   pan_unpack((global const struct mali_cs_move32_packed *)&input_cs[1],
               CS_MOVE32, move_hi);
   pan_pack((global struct mali_cs_move32_packed *)&output_cs[1], CS_MOVE32, cfg) {
      cfg = move_hi;
      cfg.immediate = (uint64_t)exec >> 32;
   }

   global const struct panlib_dgc_execution_set_entry *pipeline =
      (global const struct panlib_dgc_execution_set_entry *)(info->pipelines +
         (uint64_t)seq->execution_set[0] * info->pipeline_stride);

   for (uint32_t s = 0; s < PANLIB_DGC_STAGE_COUNT; s++) {
      global const struct panlib_dgc_shader *shader = &pipeline->shaders[s];
      if (!(pipeline->stages & (1u << s)) ||
          (s == PANLIB_DGC_FS && !info->fs_enabled)) {
         exec->fau[s] = 0;
         exec->srt[s] = 0;
         continue;
      }

      const uint64_t fau = (uint64_t)exec + info->fau_offset[s];
      for (uint32_t w = 0; w < shader->fau_count * 2; w++) {
         global const struct panlib_dgc_fau_word *recipe = &shader->fau[w];
         uint32_t value = recipe->value;
         switch (recipe->source) {
         case PANLIB_DGC_FAU_SYSVAL:
            value = sequence_sysval(info, seq, pipeline, s, value);
            break;
         case PANLIB_DGC_FAU_PUSH_CONSTANT:
            value = seq->push_constants[s][value / 4];
            break;
         case PANLIB_DGC_FAU_SELF:
            value = fau >> (value * 8);
            break;
         }
         ((global uint32_t *)fau)[w] = value;
      }
      exec->fau[s] = shader->fau_count ? fau | ((uint64_t)shader->fau_count << 56)
                                          : 0;

      global uint32_t *driver =
         (global uint32_t *)((uint64_t)exec + info->driver_offset[s]);
      uint32_t driver_size = info->driver_set_size[s];
      global const uint32_t *template_driver =
         (global const uint32_t *)info->driver_set[s];
      for (uint32_t w = 0; w < driver_size / 4; w++)
         driver[w] = template_driver[w];

      if (info->pipeline_stride && s == PANLIB_DGC_FS) {
         for (uint32_t w = 0; w < shader->fs_varying_attr_desc_count * 8; w++)
            driver[w] = ((global const uint32_t *)pipeline->fs_varying_descs)[w];
         pan_pack((global struct mali_sampler_packed *)(driver +
                      shader->fs_varying_attr_desc_count * 8), SAMPLER, cfg) {
            cfg.clamp_integer_array_indices = false;
         }
         driver_size = (shader->fs_varying_attr_desc_count + 1) * 32;
      }

      const uint32_t resource_count = info->resource_table[s] & 63;
      global const uint32_t *template_resources =
         (global const uint32_t *)(info->resource_table[s] & ~63ul);
      global uint32_t *resources =
         (global uint32_t *)((uint64_t)exec + info->resource_offset[s]);
      for (uint32_t w = 0; w < resource_count * 4; w++)
         resources[w] = template_resources[w];

      pan_pack((global struct mali_resource_packed *)resources,
               RESOURCE, cfg) {
         cfg.address = (uint64_t)driver;
         cfg.size = driver_size;
         cfg.contains_descriptors = true;
      }
      exec->srt[s] = (uint64_t)resources | resource_count;
   }

   if (pipeline->stages & (1u << PANLIB_DGC_VS)) {
      global uint32_t *driver =
         (global uint32_t *)((uint64_t)exec + info->driver_offset[PANLIB_DGC_VS]);
      for (uint32_t a = 0; a < PANLIB_DGC_MAX_VERTEX_BUFFERS; a++) {
         if (!(info->attribs_valid & (1u << a)))
            continue;
         const uint32_t stride = seq->vb[info->attrib_binding[a]].stride;
         global struct mali_attribute_packed *attribute =
            (global struct mali_attribute_packed *)(driver + a * 8);
         pan_unpack(attribute, ATTRIBUTE, old);
         pan_pack(attribute, ATTRIBUTE, cfg) {
            cfg = old;
            cfg.offset = info->attrib_offset[a];
            if (info->attribs_per_instance & (1u << a))
               cfg.offset += seq->draw[4] * stride;
            /* These vertex-packet fields alias the 1D stride word. The
             * unpacker fills every alias, and packing ORs them together.
             * Retaining the old aliases would turn stride 8 -> 16 into 24.
             */
            cfg.attribute_stride = 0;
            cfg.packet_stride = 0;
            cfg.stride = (info->attribs_zero_divisor & (1u << a)) ? 0 : stride;
         }
      }

      for (uint32_t v = 0; v < info->vertex_buffer_count; v++) {
         global const struct panlib_dgc_vertex_buffer *vb = &seq->vb[v];
         global struct mali_buffer_packed *buffer =
            (global struct mali_buffer_packed *)(driver +
               (info->vertex_buffer_offset + v) * 8);
         if (vb->address) {
            pan_pack(buffer, BUFFER, cfg) {
               cfg.address = vb->address;
               cfg.size = vb->size;
            }
         } else {
            /* A null vertex-buffer token can still contain a nonzero size
             * and stride. Match ordinary null VB bindings: a BUFFER at VA 0
             * would perform real memory accesses instead of returning zero.
             */
            pan_pack((global struct mali_null_descriptor_packed *)buffer,
                     NULL_DESCRIPTOR, cfg) {
            }
         }
      }
   }

   for (uint32_t i = 0; i < 3; i++)
      exec->vs_spd[i] = pipeline->shaders[PANLIB_DGC_VS].spd[i];
   exec->fs_spd = info->fs_enabled ? pipeline->shaders[PANLIB_DGC_FS].spd[0] : 0;
   exec->cs_spd = pipeline->shaders[PANLIB_DGC_CS].spd[0];
   /* The compiler omits executables for shaders with no observable work.
    * Ordinary draws/dispatches skip these on the CPU. With an execution set
    * the selected shader is only known here, so suppress the GPU work too.
    */
   if (!exec->cs_spd)
      exec->dispatch[0] = 0;
   if (!exec->vs_spd[info->point_primitive ? 0 : 1])
      exec->draw[0] = 0;
   exec->compute_size_workgroup =
      pipeline->shaders[PANLIB_DGC_CS].compute_size_workgroup;
   exec->workgroups_per_task = pipeline->shaders[PANLIB_DGC_CS].workgroups_per_task;
   if (info->tsd_table)
      exec->tsd = ((global const uint64_t *)info->tsd_table)[
         info->pipeline_stride ? seq->execution_set[0] : 0];
   if (pipeline->stages & (1u << PANLIB_DGC_VS))
      prepare_graphics_registers(info, pipeline, exec);
}
