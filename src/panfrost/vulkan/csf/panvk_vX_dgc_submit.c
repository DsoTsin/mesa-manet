/*
 * Copyright © 2026 PanVK contributors
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "bifrost/bifrost_compile.h"
#include "drm-uapi/panthor_drm.h"
#include "genxml/cs_builder.h"
#include "pan_desc.h"
#include "pan_encoder.h"
#include "panvk_cmd_buffer.h"
#include "panvk_dgc.h"
#include "panvk_dgc_submit.h"
#include "panvk_instr.h"
#include "panvk_physical_device.h"
#include "panvk_priv_bo.h"
#include "panvk_queue.h"
#include "panvk_shader.h"
#include "util/bitscan.h"
#include "vk_alloc.h"
#include "vk_command_pool.h"
#include "xf86drm.h"

VkResult
panvk_per_arch(dgc_record_execute)(struct panvk_cmd_buffer *cmdbuf,
                                  const struct panvk_dgc_job *job)
{
   struct panvk_dgc_record *record = vk_alloc(
      &cmdbuf->vk.pool->alloc, sizeof(*record), 8,
      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!record)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   record->kind = PANVK_DGC_RECORD_EXECUTE;
   record->job = *job;
   list_addtail(&record->link, &cmdbuf->dgc_records);
   return VK_SUCCESS;
}

VkResult
panvk_per_arch(dgc_record_secondary)(struct panvk_cmd_buffer *cmdbuf,
                                    struct panvk_cmd_buffer *secondary)
{
   struct panvk_dgc_record *record = vk_alloc(
      &cmdbuf->vk.pool->alloc, sizeof(*record), 8,
      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!record)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   record->kind = PANVK_DGC_RECORD_SECONDARY;
   record->secondary = secondary;
   list_addtail(&record->link, &cmdbuf->dgc_records);
   return VK_SUCCESS;
}

void
panvk_per_arch(dgc_records_reset)(struct panvk_cmd_buffer *cmdbuf)
{
   list_for_each_entry_safe(struct panvk_dgc_record, record,
                            &cmdbuf->dgc_records, link) {
      list_del(&record->link);
      vk_free(&cmdbuf->vk.pool->alloc, record);
   }
}

void
panvk_per_arch(dgc_submit_destroy)(struct panvk_device *dev,
                                  struct panvk_dgc_submit *submit)
{
   if (!submit)
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(submit->completion_syncobjs); i++) {
      if (submit->completion_syncobjs[i])
         drmSyncobjDestroy(dev->drm_fd, submit->completion_syncobjs[i]);
   }
   util_dynarray_foreach(&submit->bos, struct panvk_priv_bo *, bo)
      panvk_priv_bo_unref(*bo);
   util_dynarray_fini(&submit->bos);
   free(submit);
}

static VkResult
alloc_bo(struct panvk_device *dev, struct panvk_dgc_submit *submit,
         uint64_t size, uint32_t flags, struct panvk_priv_bo **out)
{
   struct panvk_priv_bo **slot = util_dynarray_grow(
      &submit->bos, struct panvk_priv_bo *, 1);
   if (!slot)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   *slot = NULL;
   VkResult result = panvk_priv_bo_create(
      dev, size, flags, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE, slot);
   if (result == VK_SUCCESS)
      *out = *slot;
   return result;
}

static uint64_t
stack_allocation_size(unsigned tls_size, const struct pan_kmod_dev_props *props)
{
   if (!tls_size)
      return 0;
   return (16ull << pan_get_stack_shift(tls_size)) *
          pan_query_thread_tls_alloc(props) * pan_query_core_id_range(props);
}

static uint64_t
wls_allocation_size(unsigned wls_size, unsigned instances,
                    const struct pan_kmod_dev_props *props)
{
   return wls_size ? (uint64_t)pan_wls_adjust_size(wls_size) * instances *
                        pan_query_core_id_range(props) : 0;
}

static const struct panlib_dgc_execution_set_entry *
job_pipeline(const struct panvk_dgc_job *job, unsigned index)
{
   if (!job->execution_set)
      return job->fixed_pipeline;
   return (const void *)((const uint8_t *)job->execution_set->gpu_map +
                        (size_t)index * job->execution_set->gpu_stride);
}

static unsigned
pipeline_wls_instances(const struct panlib_dgc_execution_set_entry *pipeline,
                       const struct pan_kmod_dev_props *props)
{
   const struct panlib_dgc_shader *cs = &pipeline->shaders[PANLIB_DGC_CS];
   if (!(pipeline->stages & BITFIELD_BIT(PANLIB_DGC_CS)) || !cs->wls_size)
      return 0;

   struct pan_compute_dim local = {
      cs->local_size[0], cs->local_size[1], cs->local_size[2],
   };
   return pan_calc_wls_instances(&local, props, NULL, cs->work_reg_count);
}

static VkResult
prepare_job(struct panvk_device *dev, struct panvk_dgc_submit *submit,
            const struct panvk_dgc_job *job)
{
   const struct panvk_physical_device *phys =
      to_panvk_physical_device(dev->vk.physical);
   const struct pan_kmod_dev_props *props = &phys->kmod.dev->props;
   const struct panvk_shader_variant *prepare = job->prepare_shader;
   const unsigned pipeline_count = job->execution_set ?
      job->execution_set->capacity : 1;
   unsigned tls_size = 0;
   uint64_t wls_size = 0;
   uint32_t stage_fau_size[PANLIB_DGC_STAGE_COUNT] = {0};
   uint32_t stage_driver_size[PANLIB_DGC_STAGE_COUNT];
   memcpy(stage_driver_size, job->params.driver_set_size,
          sizeof(stage_driver_size));

   /* Only execution-set metadata is inspected. Sequence count, shader indices
    * and all application tokens remain entirely on the GPU. */
   for (unsigned i = 0; i < pipeline_count; i++) {
      const struct panlib_dgc_execution_set_entry *pipeline =
         job_pipeline(job, i);
      u_foreach_bit(s, pipeline->stages) {
         tls_size = MAX2(tls_size, pipeline->shaders[s].tls_size);
         stage_fau_size[s] = MAX2(stage_fau_size[s],
                                 pipeline->shaders[s].fau_count * 8);
         if (job->execution_set && s == PANLIB_DGC_FS) {
            stage_driver_size[s] = MAX2(
               stage_driver_size[s],
               (pipeline->shaders[s].fs_varying_attr_desc_count + 1) * 32);
         }
      }
      unsigned instances = pipeline_wls_instances(pipeline, props);
      if (instances)
         wls_size = MAX2(wls_size, wls_allocation_size(
            pipeline->shaders[PANLIB_DGC_CS].wls_size, instances, props));
   }

   /* Size the private sequence tail from the latest execution-set contents.
    * Updates after command recording may increase FAUs or FS varying tables.
    * Keep the recorded driver-template lengths unchanged: the GPU copies only
    * those bytes before replacing any execution-set-specific descriptors.
    */
   uint32_t stage_fau_offset[PANLIB_DGC_STAGE_COUNT];
   uint32_t stage_driver_offset[PANLIB_DGC_STAGE_COUNT];
   uint32_t stage_resource_offset[PANLIB_DGC_STAGE_COUNT];
   uint32_t output_stride = ALIGN_POT(sizeof(struct panlib_dgc_execution), 64);
   for (unsigned s = 0; s < PANLIB_DGC_STAGE_COUNT; s++) {
      stage_fau_offset[s] = ALIGN_POT(output_stride, 16);
      output_stride = stage_fau_offset[s] + stage_fau_size[s];
      stage_driver_offset[s] = ALIGN_POT(output_stride, 32);
      output_stride = stage_driver_offset[s] + stage_driver_size[s];
      stage_resource_offset[s] = ALIGN_POT(output_stride, 64);
      output_stride = stage_resource_offset[s] +
         (job->params.resource_table[s] & 63) * 16;
   }
   uint32_t depth_stencil_offset = 0;
   if (job->graphics) {
      depth_stencil_offset = ALIGN_POT(output_stride, 32);
      output_stride = depth_stencil_offset + 32;
   }
   output_stride = ALIGN_POT(output_stride, 64);

   /* Prepare and application shaders have separate backing allocations. The
    * application arena can remain in use by fragments while later operations
    * prepare on the compute subqueue. */
   unsigned prepare_wls_instances = prepare->info.wls_size ?
      pan_calc_wls_instances(&prepare->cs.local_size, props, NULL,
                             prepare->info.work_reg_count) : 0;
   uint64_t app_tls_offset = 0;
   uint64_t app_wls_offset = ALIGN_POT(
      stack_allocation_size(tls_size, props), 4096);
   uint64_t prepare_tls_offset = ALIGN_POT(app_wls_offset + wls_size, 4096);
   uint64_t prepare_wls_offset = ALIGN_POT(prepare_tls_offset +
      stack_allocation_size(prepare->info.tls_size, props), 4096);
   uint64_t scratch_size = prepare_wls_offset + wls_allocation_size(
      prepare->info.wls_size, prepare_wls_instances, props);
   struct panvk_priv_bo *scratch = NULL;
   VkResult result;
   if (scratch_size) {
      result = alloc_bo(dev, submit, scratch_size,
                        PAN_KMOD_BO_FLAG_NO_MMAP, &scratch);
      if (result != VK_SUCCESS)
         return result;
   }
   uint64_t scratch_gpu = scratch ? scratch->addr.dev : 0;

   const uint64_t params_offset = ALIGN_POT(sizeof(struct panlib_dgc_submit_node), 64);
   const uint64_t fau_offset = ALIGN_POT(params_offset + sizeof(job->params), 16);
   const uint64_t fau_size = BIFROST_PRECOMPILED_KERNEL_SYSVALS_SIZE + sizeof(uint64_t);
   const uint64_t tsd_offset = ALIGN_POT(fau_offset + fau_size, 64);
   /* LOCAL_STORAGE occupies 32 bytes but each descriptor must start on a
    * 64-byte boundary. In particular, IDVS cannot use a densely packed odd
    * execution-set slot as its fragment thread-storage descriptor. */
   const uint64_t tsd_stride = ALIGN_POT(pan_size(LOCAL_STORAGE),
                                        pan_alignment(LOCAL_STORAGE));
   const uint64_t tsd_table_offset = ALIGN_POT(
      tsd_offset + ((uint64_t)pipeline_count + 1) * tsd_stride, 64);
   const uint64_t output_offset = ALIGN_POT(
      tsd_table_offset + (uint64_t)pipeline_count * sizeof(uint64_t), 64);
   const uint64_t output_size = job->params.sequence_offset +
      (uint64_t)job->max_sequences * output_stride;
   if (output_size > SIZE_MAX - output_offset)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct panvk_priv_bo *bo;
   result = alloc_bo(dev, submit, output_offset + output_size, 0, &bo);
   if (result != VK_SUCCESS)
      return result;
   memset(bo->addr.host, 0, output_offset);
   struct panlib_dgc_submit_node *node = bo->addr.host;
   struct panlib_dgc_execute *params = bo->addr.host + params_offset;
   uint64_t *tsd_table = bo->addr.host + tsd_table_offset;
   *params = job->params;
   memcpy(params->fau_offset, stage_fau_offset, sizeof(stage_fau_offset));
   memcpy(params->driver_offset, stage_driver_offset,
          sizeof(stage_driver_offset));
   memcpy(params->resource_offset, stage_resource_offset,
          sizeof(stage_resource_offset));
   params->depth_stencil_offset = depth_stencil_offset;
   params->output_sequence_stride = output_stride;
   params->output = bo->addr.dev + output_offset;
   params->tsd_table = bo->addr.dev + tsd_table_offset;
   node->output = params->output;
   node->tsd_table = params->tsd_table;
   node->prepare_tsd = bo->addr.dev + tsd_offset +
      (uint64_t)pipeline_count * tsd_stride;
   node->prepare_fau = (bo->addr.dev + fau_offset) |
      ((uint64_t)DIV_ROUND_UP(fau_size, 8) << 56);

   uint64_t params_gpu = bo->addr.dev + params_offset;
   struct bifrost_precompiled_kernel_sysvals sysvals = {
      .num_workgroups = {MAX2(DIV_ROUND_UP(job->max_sequences, 64), 1), 1, 1},
      .printf_buffer_address = dev->printf.bo->addr.dev,
   };
   bifrost_precompiled_kernel_prepare_push_uniforms(
      bo->addr.host + fau_offset, &params_gpu, sizeof(params_gpu), &sysvals);

   for (unsigned i = 0; i < pipeline_count; i++) {
      const struct panlib_dgc_execution_set_entry *pipeline = job_pipeline(job, i);
      unsigned instances = pipeline_wls_instances(pipeline, props);
      struct pan_tls_info tls = {
         .tls = {.size = tls_size, .ptr = scratch_gpu + app_tls_offset},
         .wls = {
            .size = instances ? pipeline->shaders[PANLIB_DGC_CS].wls_size : 0,
            .instances = instances,
            .ptr = scratch_gpu + app_wls_offset,
         },
      };
      uint64_t offset = tsd_offset + (uint64_t)i * tsd_stride;
      GENX(pan_emit_tls)(&tls, bo->addr.host + offset);
      tsd_table[i] = bo->addr.dev + offset;
   }
   struct pan_tls_info prepare_tls = {
      .tls = {.size = prepare->info.tls_size,
              .ptr = scratch_gpu + prepare_tls_offset},
      .wls = {.size = prepare->info.wls_size,
              .instances = prepare_wls_instances,
              .ptr = scratch_gpu + prepare_wls_offset},
   };
   GENX(pan_emit_tls)(&prepare_tls, bo->addr.host + tsd_offset +
                     (uint64_t)pipeline_count * tsd_stride);

   unsigned queue_mask = BITFIELD_BIT(PANVK_SUBQUEUE_COMPUTE);
   if (job->graphics)
      queue_mask |= BITFIELD_BIT(PANVK_SUBQUEUE_VERTEX_TILER);
   u_foreach_bit(q, queue_mask) {
      if (submit->tails[q])
         *submit->tails[q] = bo->addr.dev;
      else
         submit->heads[q] = bo->addr.dev;
      submit->tails[q] = q == PANVK_SUBQUEUE_COMPUTE ?
         &node->next_compute : &node->next_vt;
   }
   return VK_SUCCESS;
}

static VkResult
prepare_cmdbuf(struct panvk_device *dev, struct panvk_dgc_submit *submit,
               struct panvk_cmd_buffer *cmdbuf)
{
   list_for_each_entry(struct panvk_dgc_record, record,
                       &cmdbuf->dgc_records, link) {
      struct panvk_indirect_execution_set *set =
         record->kind == PANVK_DGC_RECORD_EXECUTE
            ? record->job.execution_set : NULL;
      if (set)
         simple_mtx_lock(&set->metadata_lock);
      VkResult result = record->kind == PANVK_DGC_RECORD_EXECUTE ?
         prepare_job(dev, submit, &record->job) :
         prepare_cmdbuf(dev, submit, record->secondary);
      if (set)
         simple_mtx_unlock(&set->metadata_lock);
      if (result != VK_SUCCESS)
         return result;
   }
   return VK_SUCCESS;
}

VkResult
panvk_per_arch(dgc_submit_prepare)(struct panvk_gpu_queue *queue,
                                  const struct vk_queue_submit *vk_submit,
                                  struct panvk_dgc_submit **out)
{
   *out = NULL;
   bool has_records = false;
   for (unsigned i = 0; i < vk_submit->command_buffer_count; i++) {
      struct panvk_cmd_buffer *cmdbuf = container_of(
         vk_submit->command_buffers[i], struct panvk_cmd_buffer, vk);
      has_records |= !list_is_empty(&cmdbuf->dgc_records);
   }
   if (!has_records)
      return VK_SUCCESS;

   struct panvk_device *dev = to_panvk_device(queue->vk.base.device);
   struct panvk_dgc_submit *submit = calloc(1, sizeof(*submit));
   if (!submit)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   util_dynarray_init(&submit->bos, NULL);
   VkResult result;
   for (unsigned i = 0; i < vk_submit->command_buffer_count; i++) {
      struct panvk_cmd_buffer *cmdbuf = container_of(
         vk_submit->command_buffers[i], struct panvk_cmd_buffer, vk);
      result = prepare_cmdbuf(dev, submit, cmdbuf);
      if (result != VK_SUCCESS)
         goto fail;
   }
   if (!submit->heads[PANVK_SUBQUEUE_COMPUTE]) {
      panvk_per_arch(dgc_submit_destroy)(dev, submit);
      return VK_SUCCESS;
   }

   struct panvk_priv_bo *prefix_bo;
   result = alloc_bo(dev, submit, 4096, 0, &prefix_bo);
   if (result != VK_SUCCESS)
      goto fail;
   const struct drm_panthor_csif_info *csif = panvk_get_csif_props(dev);
   for (unsigned q = 0; q < PANVK_SUBQUEUE_COUNT; q++) {
      if (!submit->heads[q])
         continue;
      struct cs_builder b;
      struct cs_builder_conf conf = {
         .nr_registers = csif->cs_reg_count,
         .nr_kernel_registers = MAX2(csif->unpreserved_cs_reg_count, 4),
         .ls_sb_slot = SB_ID(LS),
      };
      struct cs_buffer buffer = {
         .cpu = prefix_bo->addr.host + q * 1024,
         .gpu = prefix_bo->addr.dev + q * 1024,
         .capacity = 1024 / sizeof(uint64_t),
      };
      cs_builder_init(&b, &conf, buffer);
      cs_move64_to(&b, cs_subqueue_ctx_reg(&b),
                   panvk_priv_mem_dev_addr(queue->subqueues[q].context));
      cs_move64_to(&b, cs_scratch_reg64(&b, 0), submit->heads[q]);
      cs_store64(&b, cs_scratch_reg64(&b, 0), cs_subqueue_ctx_reg(&b),
                  offsetof(struct panvk_cs_subqueue_context, dgc.next));
      cs_flush_stores(&b);
      cs_end(&b);
      assert(cs_is_valid(&b));
      submit->prefix[q].gpu = cs_root_chunk_gpu_addr(&b);
      submit->prefix[q].size = cs_root_chunk_size(&b);
      cs_builder_fini(&b);
   }
   util_dynarray_foreach(&submit->bos, struct panvk_priv_bo *, bo) {
      if ((*bo)->addr.host)
         panvk_priv_bo_flush(*bo, 0, pan_kmod_bo_size((*bo)->bo));
   }
   *out = submit;
   return VK_SUCCESS;

fail:
   panvk_per_arch(dgc_submit_destroy)(dev, submit);
   return result;
}
