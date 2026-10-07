/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>

#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_shader.h"

#include "pan_nir.h"
#include "nir_builder.h"
#include "vk_meta.h"

static void
pilot_batch_reset(struct panvk_pilot_batch *batch)
{
   util_dynarray_clear(&batch->entries);
   batch->fau = NULL;
   batch->grid = NULL;
   batch->open = false;
}

void
panvk_per_arch(cmd_pilots_init)(struct panvk_cmd_buffer *cmdbuf)
{
   util_dynarray_init(&cmdbuf->pilots.vt.entries, NULL);
   util_dynarray_init(&cmdbuf->pilots.cs.entries, NULL);
   util_dynarray_init(&cmdbuf->pilots.fs, NULL);
   pilot_batch_reset(&cmdbuf->pilots.vt);
   pilot_batch_reset(&cmdbuf->pilots.cs);
   cmdbuf->pilots.fs_after_tiling = false;
}

void
panvk_per_arch(cmd_pilots_reset)(struct panvk_cmd_buffer *cmdbuf)
{
   pilot_batch_reset(&cmdbuf->pilots.vt);
   pilot_batch_reset(&cmdbuf->pilots.cs);
   util_dynarray_clear(&cmdbuf->pilots.fs);
   cmdbuf->pilots.fs_after_tiling = false;
}

void
panvk_per_arch(cmd_pilots_fini)(struct panvk_cmd_buffer *cmdbuf)
{
   util_dynarray_fini(&cmdbuf->pilots.vt.entries);
   util_dynarray_fini(&cmdbuf->pilots.cs.entries);
   util_dynarray_fini(&cmdbuf->pilots.fs);
}

void
panvk_per_arch(cmd_pilot_close_all)(struct panvk_cmd_buffer *cmdbuf)
{
   panvk_per_arch(cmd_pilot_close)(cmdbuf, PANVK_SUBQUEUE_VERTEX_TILER);
   panvk_per_arch(cmd_pilot_close)(cmdbuf, PANVK_SUBQUEUE_COMPUTE);
}

#if PAN_ARCH == 10 || PAN_ARCH >= 15

#define PILOT_WG_SIZE      16
#define PILOT_FAU_SIZE     16
#define PILOT_FAU_STRIDE   64
#define PILOT_FTZ_MODES    4
#define PILOT_RECORD_SIZE  32
#define PILOT_PAIR_SIZE    16
#define PILOT_GUARD_SIZE   32
#define PILOT_SAVE_SIZE    160

#if PAN_ARCH >= 15
#define PILOT_MAX_REGS 128
#endif

struct panvk_pilot_dispatcher_key {
   char name[24];
   uint32_t ftz;
};

static nir_def *
pilot_guard_passes(nir_builder *b, nir_def *guard)
{
   nir_def *passes;

   nir_push_if(b, nir_ine_imm(b, guard, 0));
   {
      nir_def *head =
         nir_load_global_constant(b, 4, 32, guard, .align_mul = 16);
      nir_def *tail = nir_load_global_constant(
         b, 2, 32, nir_iadd_imm(b, guard, 16), .align_mul = 16);
      nir_def *params = nir_pack_64_2x32(b, nir_channels(b, head, 0x3));
      nir_def *words = nir_channel(b, head, 2);
      nir_def *index = nir_channel(b, head, 3);
      nir_def *count_addr = nir_pack_64_2x32(b, tail);
      nir_def *has_count = nir_ine_imm(b, count_addr, 0);
      nir_def *args =
         nir_load_global_constant(b, 3, 32, params, .align_mul = 4);
      nir_def *count = nir_load_global_constant(
         b, 1, 32, nir_bcsel(b, has_count, count_addr, guard), .align_mul = 4);

      passes = nir_iand(b, nir_ine_imm(b, nir_channel(b, args, 0), 0),
                        nir_ine_imm(b, nir_channel(b, args, 1), 0));
      passes = nir_iand(b, passes,
                        nir_ior(b, nir_ult_imm(b, words, 3),
                                nir_ine_imm(b, nir_channel(b, args, 2), 0)));
      passes = nir_iand(b, passes,
                        nir_ior(b, nir_inot(b, has_count),
                                nir_ult(b, index, count)));
   }
   nir_push_else(b, NULL);
   nir_def *unguarded = nir_imm_true(b);
   nir_pop_if(b, NULL);

   return nir_if_phi(b, passes, unguarded);
}

static nir_shader *
pilot_dispatcher_nir(void)
{
   const struct nir_shader_compiler_options *options =
      pan_get_nir_shader_compiler_options(PAN_ARCH, MESA_SHADER_COMPUTE,
                                          false);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options,
                                                  "panvk_pilot_dispatcher");
   b.shader->info.workgroup_size[0] = PILOT_WG_SIZE;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *records = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0));
   nir_def *count = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8));
   nir_def *wg = nir_channel(&b, nir_load_workgroup_id(&b), 0);

   nir_push_if(&b, nir_ult(&b, wg, count));
   {
      nir_def *rec_addr =
         nir_iadd(&b, records, nir_u2u64(&b, nir_ishl_imm(&b, wg, 5)));
      nir_def *rec =
         nir_load_global_constant(&b, 4, 32, rec_addr, .align_mul = 16);
      nir_def *guard = nir_pack_64_2x32(
         &b, nir_load_global_constant(&b, 2, 32, nir_iadd_imm(&b, rec_addr, 16),
                                      .align_mul = 16));

      nir_push_if(&b, pilot_guard_passes(&b, guard));
      {
         nir_def *lane =
            nir_channel(&b, nir_load_local_invocation_id(&b), 0);
         nir_def *last = nir_iadd_imm(&b, nir_channel(&b, rec, 1), -1);
         nir_def *pair_addr = nir_iadd(
            &b, nir_pack_64_2x32(&b, nir_channels(&b, rec, 0xc)),
            nir_u2u64(&b, nir_ishl_imm(&b, nir_umin(&b, lane, last), 4)));
         nir_def *pair =
            nir_load_global_constant(&b, 4, 32, pair_addr, .align_mul = 16);
         nir_def *fau = nir_pack_64_2x32_split(
            &b, nir_channel(&b, pair, 0),
            nir_iand_imm(&b, nir_channel(&b, pair, 1), BITFIELD_MASK(24)));
         nir_def *srt = nir_pack_64_2x32_split(
            &b, nir_iand_imm(&b, nir_channel(&b, pair, 2), ~BITFIELD_MASK(6)),
            nir_channel(&b, pair, 3));
         nir_jump_pilot_pan(&b, nir_channel(&b, rec, 0), fau, srt);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);

   return b.shader;
}

static VkResult
get_pilot_dispatcher(struct panvk_device *dev, uint32_t ftz,
                     struct panvk_internal_shader **shader_out)
{
   struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);
   const struct panvk_pilot_dispatcher_key key = {
      .name = "panvk-pilot-dispatcher",
      .ftz = ftz,
   };
   struct panvk_internal_shader *shader;
   VkShaderEXT shader_handle = (VkShaderEXT)vk_meta_lookup_object(
      &dev->meta, VK_OBJECT_TYPE_SHADER_EXT, &key, sizeof(key));
   if (shader_handle != VK_NULL_HANDLE)
      goto out;

   nir_shader *nir = pilot_dispatcher_nir();
   PAN_NIR_SET_BLAKE3_INTERNAL(nir, &key);

   struct pan_compile_inputs inputs = {
      .gpu_id = phys_dev->kmod.dev->props.gpu_id,
      .gpu_variant = phys_dev->kmod.dev->props.gpu_variant,
      .fau.reserved = PILOT_FAU_SIZE / 4,
   };

   pan_preprocess_nir(nir, inputs.gpu_id);

   VkResult result =
      panvk_per_arch(create_internal_shader)(dev, nir, &inputs, &shader);
   ralloc_free(nir);
   if (result != VK_SUCCESS)
      return result;

   shader->spd = panvk_pool_alloc_desc(&dev->mempools.rw, SHADER_PROGRAM);
   if (!panvk_priv_mem_check_alloc(shader->spd)) {
      vk_shader_destroy(&dev->vk, &shader->vk, NULL);
      return panvk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   panvk_priv_mem_write_desc(shader->spd, 0, SHADER_PROGRAM, cfg) {
      cfg.stage = MALI_SHADER_STAGE_COMPUTE;
#if PAN_ARCH >= 15
      cfg.register_count = PILOT_MAX_REGS;
      cfg.preload.r0_r15 = shader->info.preload;
#else
      cfg.register_allocation = MALI_SHADER_REGISTER_ALLOCATION_64_PER_THREAD;
      cfg.preload.r48_r63 = shader->info.preload >> 48;
#endif
      cfg.binary = panvk_priv_mem_dev_addr(shader->code_mem);
      cfg.flush_to_zero_mode = ftz;
   }

   shader_handle = (VkShaderEXT)vk_meta_cache_object(
      &dev->vk, &dev->meta, &key, sizeof(key), VK_OBJECT_TYPE_SHADER_EXT,
      (uint64_t)panvk_internal_shader_to_handle(shader));

out:
   *shader_out = panvk_internal_shader_from_handle(shader_handle);
   return VK_SUCCESS;
}

static uint32_t
pilot_ftz(const struct panvk_shader_variant *pilot)
{
   if (!pilot->info.ftz_fp32)
      return MALI_FLUSH_TO_ZERO_MODE_PRESERVE_SUBNORMALS;

   return pilot->info.ftz_fp16 ? MALI_FLUSH_TO_ZERO_MODE_ALWAYS
                               : MALI_FLUSH_TO_ZERO_MODE_DX11;
}

static void
pilot_set_job(struct panvk_cmd_buffer *cmdbuf, struct cs_builder *b,
              const struct panvk_internal_shader *dispatcher, uint64_t fau)
{
   struct mali_compute_size_workgroup_packed wg;
   pan_pack(&wg, COMPUTE_SIZE_WORKGROUP, cfg) {
      cfg.workgroup_size_x = PILOT_WG_SIZE;
      cfg.workgroup_size_y = 1;
      cfg.workgroup_size_z = 1;
   }

   cs_move64_to(b, cs_reg64(b, MALI_COMPUTE_SR_SRT_3), 0);
   cs_move64_to(b, cs_reg64(b, MALI_COMPUTE_SR_FAU_3),
                fau | ((uint64_t)(PILOT_FAU_SIZE / 8) << 56));
#if PAN_ARCH >= 15
   struct mali_shader_program_pointer_packed spp;
   pan_pack(&spp, SHADER_PROGRAM_POINTER, cfg) {
      cfg.register_count = PILOT_MAX_REGS;
      cfg.pointer = panvk_priv_mem_dev_addr(dispatcher->spd);
   }
   cs_move64_to(b, cs_reg64(b, MALI_COMPUTE_SR_SPD_3),
                ((uint64_t)spp.opaque[1] << 32) | spp.opaque[0]);
#else
   cs_move64_to(b, cs_reg64(b, MALI_COMPUTE_SR_SPD_3),
                panvk_priv_mem_dev_addr(dispatcher->spd));
#endif
   cs_move64_to(b, cs_reg64(b, MALI_COMPUTE_SR_TSD_3),
                cmdbuf->state.tls.desc.gpu);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, GLOBAL_ATTRIBUTE_OFFSET), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, WG_SIZE), wg.opaque[0]);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_X), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Y), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Z), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Y), 1);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Z), 1);
}

static void
pilot_patch_grid(struct cs_builder *b, struct cs_maybe *grid,
                 uint32_t wg_count)
{
   struct mali_cs_move32_packed move;

   assert(grid->num_instrs == 1);

   pan_pack(&move, CS_MOVE32, I) {
      I.destination = MALI_COMPUTE_SR_JOB_SIZE_X;
      I.immediate = wg_count;
   }
   memcpy(&grid->instrs[0], &move, sizeof(move));
   cs_patch_maybe(b, grid);
}

static void
pilot_use_tls(struct panvk_cmd_buffer *cmdbuf,
              const struct panvk_shader_variant *pilot)
{
   cmdbuf->state.tls.info.tls.size =
      MAX2(cmdbuf->state.tls.info.tls.size, pilot->info.tls_size);
}

static uint16_t
pilot_res3_mask(unsigned base)
{
   static const unsigned regs[] = {
      MALI_COMPUTE_SR_SRT_3,
      MALI_COMPUTE_SR_FAU_3,
      MALI_COMPUTE_SR_SPD_3,
      MALI_COMPUTE_SR_TSD_3,
   };
   uint16_t mask = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(regs); i++) {
      if (regs[i] >= base && regs[i] < base + 16)
         mask |= BITFIELD_RANGE(regs[i] - base, 2);
   }
   return mask;
}

static void
pilot_job_regs_store(struct cs_builder *b, struct cs_index addr,
                     uint32_t offset)
{
   cs_store(b, cs_reg_tuple(b, 0, 16), addr, pilot_res3_mask(0), offset);
   cs_store(b, cs_reg_tuple(b, 16, 16), addr, pilot_res3_mask(16),
            offset + 64);
   cs_store(b, cs_sr_reg_tuple(b, COMPUTE, GLOBAL_ATTRIBUTE_OFFSET, 8), addr,
            BITFIELD_MASK(8), offset + 128);
}

static void
pilot_job_regs_load(struct cs_builder *b, struct cs_index addr,
                    uint32_t offset)
{
   cs_load_to(b, cs_reg_tuple(b, 0, 16), addr, pilot_res3_mask(0), offset);
   cs_load_to(b, cs_reg_tuple(b, 16, 16), addr, pilot_res3_mask(16),
              offset + 64);
   cs_load_to(b, cs_sr_reg_tuple(b, COMPUTE, GLOBAL_ATTRIBUTE_OFFSET, 8),
              addr, BITFIELD_MASK(8), offset + 128);
}

static struct cs_index
pilot_addr_reg(struct cs_builder *b)
{
#if PAN_ARCH >= 12
   return cs_scratch_reg64(b, 40);
#else
   return cs_reg64(b, 64);
#endif
}

static uint32_t
pilot_code_hi(const struct panvk_internal_shader *dispatcher)
{
   return panvk_priv_mem_dev_addr(dispatcher->code_mem) >> 32;
}

static struct panvk_pilot_entry
pilot_entry(const struct panvk_shader_variant *pilot, uint64_t fau,
            uint64_t srt)
{
   return (struct panvk_pilot_entry){
      .fau = fau,
      .srt = srt,
      .pc = panvk_shader_variant_get_dev_addr(pilot),
      .ftz = pilot_ftz(pilot),
   };
}

static uint64_t
pilot_upload_guard(struct panvk_cmd_buffer *cmdbuf,
                   const struct panvk_pilot_guard *guard)
{
   struct pan_ptr mem =
      panvk_cmd_alloc_dev_mem(cmdbuf, desc, PILOT_GUARD_SIZE, PILOT_GUARD_SIZE);
   if (!mem.gpu)
      return 0;

   uint32_t *words = mem.cpu;
   words[0] = guard->params;
   words[1] = guard->params >> 32;
   words[2] = guard->words;
   words[3] = guard->index;
   words[4] = guard->count;
   words[5] = guard->count >> 32;
   words[6] = 0;
   words[7] = 0;
   return mem.gpu;
}

static int
pilot_entry_cmp(const void *a, const void *b)
{
   const struct panvk_pilot_entry *x = a;
   const struct panvk_pilot_entry *y = b;

   if (x->ftz != y->ftz)
      return x->ftz < y->ftz ? -1 : 1;
   if (x->pc != y->pc)
      return x->pc < y->pc ? -1 : 1;
   if (x->guard != y->guard)
      return x->guard < y->guard ? -1 : 1;
   return 0;
}

static uint32_t
pilot_run_length(const struct panvk_pilot_entry *entries, uint32_t count,
                 uint32_t first)
{
   uint32_t run = 1;

   if (entries[first].guard)
      return run;

   while (first + run < count && run < PANVK_PILOT_PAIRS_MAX &&
          entries[first + run].pc == entries[first].pc &&
          !entries[first + run].guard)
      run++;

   return run;
}

static uint32_t
pilot_write_records(struct panvk_cmd_buffer *cmdbuf,
                    const struct panvk_pilot_entry *entries, uint32_t count,
                    uint32_t code_hi, uint32_t *fau)
{
   uint32_t records = 0;

   for (uint32_t i = 0; i < count; i += pilot_run_length(entries, count, i))
      records++;

   const uint32_t pairs_offset = records * PILOT_RECORD_SIZE;
   struct pan_ptr mem = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, pairs_offset + count * PILOT_PAIR_SIZE, 64);
   if (!mem.gpu)
      return 0;

   uint32_t *rec = mem.cpu;
   uint64_t *pairs = (uint64_t *)((uint8_t *)mem.cpu + pairs_offset);
   uint32_t i = 0;

   for (uint32_t r = 0; r < records; r++) {
      const uint32_t run = pilot_run_length(entries, count, i);
      const uint64_t addr = mem.gpu + pairs_offset + i * PILOT_PAIR_SIZE;
      uint32_t *words = rec + r * (PILOT_RECORD_SIZE / 4);

      assert(entries[i].pc >> 32 == code_hi);
      words[0] = entries[i].pc;
      words[1] = run;
      words[2] = addr;
      words[3] = addr >> 32;
      words[4] = entries[i].guard;
      words[5] = entries[i].guard >> 32;
      words[6] = 0;
      words[7] = 0;

      for (uint32_t p = 0; p < run; p++, i++) {
         pairs[i * 2] = entries[i].fau;
         pairs[i * 2 + 1] = entries[i].srt;
      }
   }

   fau[0] = mem.gpu;
   fau[1] = mem.gpu >> 32;
   fau[2] = records;
   fau[3] = 0;
   return records;
}

#if PAN_ARCH >= 11
static void
pilot_select_own_slot(struct cs_builder *b, struct cs_index iter_sb,
                      struct cs_index slot)
{
   cs_load32_to(b, iter_sb, cs_subqueue_ctx_reg(b),
                offsetof(struct panvk_cs_subqueue_context, iter_sb));
   cs_next_sb_entry(b, slot, MALI_CS_SCOREBOARD_TYPE_ENDPOINT,
                    MALI_CS_NEXT_SB_ENTRY_FORMAT_INDEX);
}

static void
pilot_wait_own_slot(struct cs_builder *b, struct cs_index iter_sb,
                    struct cs_index slot, struct cs_index mask)
{
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_SEL_ENDPOINT, iter_sb);
   cs_move32_to(b, mask, 0);
   cs_bit_set32(b, mask, mask, slot);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_WAIT, mask);
   cs_wait_indirect(b);
   cs_move32_to(b, mask, 0);
   cs_bit_set32(b, mask, mask, iter_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_WAIT, mask);
}
#endif

static struct panvk_pilot_batch *
pilot_batch(struct panvk_cmd_buffer *cmdbuf, enum panvk_subqueue_id subqueue)
{
   assert(subqueue == PANVK_SUBQUEUE_VERTEX_TILER ||
          subqueue == PANVK_SUBQUEUE_COMPUTE);

   return subqueue == PANVK_SUBQUEUE_COMPUTE ? &cmdbuf->pilots.cs
                                             : &cmdbuf->pilots.vt;
}

static bool
pilot_batch_open(struct panvk_cmd_buffer *cmdbuf,
                 enum panvk_subqueue_id subqueue,
                 struct panvk_pilot_batch *batch, uint32_t ftz)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   struct panvk_internal_shader *dispatcher;

   if (get_pilot_dispatcher(dev, ftz, &dispatcher) != VK_SUCCESS)
      return false;

   struct pan_ptr fau =
      panvk_cmd_alloc_dev_mem(cmdbuf, desc, PILOT_FAU_SIZE, PILOT_FAU_STRIDE);
   if (!fau.gpu)
      return false;

   memset(fau.cpu, 0, PILOT_FAU_SIZE);
   batch->fau = fau.cpu;
   batch->code_hi = pilot_code_hi(dispatcher);
   batch->ftz = ftz;
   batch->open = true;

   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, subqueue);
   struct cs_index iter_sb = cs_scratch_reg32(b, 0);

#if PAN_ARCH >= 11
   struct cs_index slot = cs_scratch_reg32(b, 1);
   struct cs_index mask = cs_scratch_reg32(b, 2);

   cs_update_cmdbuf_regs(b) {
      pilot_set_job(cmdbuf, b, dispatcher, fau.gpu);
      cs_flush_loads(b);
      cs_maybe(b, &batch->grid)
         cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_X), 1);
      pilot_patch_grid(b, batch->grid, 1);
      cs_flush_stores(b);
      pilot_select_own_slot(b, iter_sb, slot);
      cs_run_compute(b, 1, MALI_TASK_AXIS_X, cs_shader_res_sel(3, 3, 3, 3));
      pilot_wait_own_slot(b, iter_sb, slot, mask);
   }
#else
   struct cs_index cmp = cs_scratch_reg32(b, 1);
   struct cs_index saved = cs_scratch_reg64(b, 2);
   struct cs_index job_size_yz = cs_reg64(b, MALI_COMPUTE_SR_JOB_SIZE_Y);

   cs_update_cmdbuf_regs(b) {
      cs_add_imm64(b, saved, job_size_yz, 0);
      pilot_set_job(cmdbuf, b, dispatcher, fau.gpu);
      cs_flush_loads(b);
      cs_maybe(b, &batch->grid)
         cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_X), 1);
      pilot_patch_grid(b, batch->grid, 1);
      cs_flush_stores(b);
      cs_load32_to(b, iter_sb, cs_subqueue_ctx_reg(b),
                   offsetof(struct panvk_cs_subqueue_context, iter_sb));
      cs_match_iter_sb(b, x, iter_sb, cmp) {
         const unsigned pilot_slot = SB_ITER((x + 1) % dev->csf.sb.iter_count);

         cs_wait_slot(b, pilot_slot);
         cs_select_endpoint_sb(b, pilot_slot);
         cs_run_compute(b, 1, MALI_TASK_AXIS_X, cs_shader_res_sel(3, 3, 3, 3));
         cs_select_endpoint_sb(b, SB_ITER(x));
         cs_wait_slot(b, pilot_slot);
      }
      cs_add_imm64(b, job_size_yz, saved, 0);
   }
#endif

   b->req_resource_mask |= CS_COMPUTE_RES;
   return true;
}

void
panvk_per_arch(cmd_pilot_close)(struct panvk_cmd_buffer *cmdbuf,
                                enum panvk_subqueue_id subqueue)
{
   struct panvk_pilot_batch *batch = pilot_batch(cmdbuf, subqueue);

   if (!batch->open)
      return;

   struct panvk_pilot_entry *entries = util_dynarray_begin(&batch->entries);
   const uint32_t count =
      util_dynarray_num_elements(&batch->entries, struct panvk_pilot_entry);

   if (count) {
      panvk_csstat_inc(to_panvk_device(cmdbuf->vk.base.device), pilot_batches);
      qsort(entries, count, sizeof(*entries), pilot_entry_cmp);

      const uint32_t records = pilot_write_records(cmdbuf, entries, count,
                                                   batch->code_hi, batch->fau);
      if (records)
         pilot_patch_grid(panvk_get_cs_builder(cmdbuf, subqueue), batch->grid,
                          records);
   }

   pilot_batch_reset(batch);
}

bool
panvk_per_arch(cmd_pilot_queue)(struct panvk_cmd_buffer *cmdbuf,
                                enum panvk_subqueue_id subqueue,
                                const struct panvk_shader_variant *shader,
                                uint64_t fau, uint64_t srt,
                                const struct panvk_pilot_guard *guard)
{
   if (cmdbuf->state.cond_render.enabled || cmdbuf->state.cond_render.inherited)
      return false;

   const struct panvk_shader_variant *pilot = shader->preamble;
   struct panvk_pilot_batch *batch = pilot_batch(cmdbuf, subqueue);
   struct panvk_pilot_entry entry = pilot_entry(pilot, fau, srt);

   if (guard) {
      entry.guard = pilot_upload_guard(cmdbuf, guard);
      if (!entry.guard)
         return false;
   }

   if (batch->open &&
       (batch->ftz != entry.ftz ||
        util_dynarray_num_elements(&batch->entries, struct panvk_pilot_entry) >=
           PANVK_PILOT_BATCH_MAX))
      panvk_per_arch(cmd_pilot_close)(cmdbuf, subqueue);

   if (!batch->open && !pilot_batch_open(cmdbuf, subqueue, batch, entry.ftz))
      return false;

   assert(entry.pc >> 32 == batch->code_hi);

   struct panvk_pilot_entry *slot =
      util_dynarray_grow(&batch->entries, struct panvk_pilot_entry, 1);
   if (!slot)
      return false;

   *slot = entry;
   pilot_use_tls(cmdbuf, pilot);
   return true;
}

void
panvk_per_arch(cmd_pilot_run)(struct panvk_cmd_buffer *cmdbuf,
                              enum panvk_subqueue_id subqueue,
                              const struct panvk_shader_variant *shader,
                              struct cs_index fau, struct cs_index srt)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct panvk_shader_variant *pilot = shader->preamble;
   struct panvk_internal_shader *dispatcher;

   if (get_pilot_dispatcher(dev, pilot_ftz(pilot), &dispatcher) != VK_SUCCESS)
      return;

   const uint32_t pair_offset = PILOT_FAU_SIZE + PILOT_RECORD_SIZE;
   const uint32_t save_offset = pair_offset + PILOT_PAIR_SIZE;
   struct pan_ptr mem =
      panvk_cmd_alloc_dev_mem(cmdbuf, desc, save_offset + PILOT_SAVE_SIZE, 64);
   if (!mem.gpu)
      return;

   assert(panvk_shader_variant_get_dev_addr(pilot) >> 32 ==
          pilot_code_hi(dispatcher));

   uint32_t *fau_words = mem.cpu;
   uint32_t *rec = (uint32_t *)((uint8_t *)mem.cpu + PILOT_FAU_SIZE);
   fau_words[0] = mem.gpu + PILOT_FAU_SIZE;
   fau_words[1] = (mem.gpu + PILOT_FAU_SIZE) >> 32;
   fau_words[2] = 1;
   fau_words[3] = 0;
   memset(rec, 0, PILOT_RECORD_SIZE);
   rec[0] = panvk_shader_variant_get_dev_addr(pilot);
   rec[1] = 1;
   rec[2] = mem.gpu + pair_offset;
   rec[3] = (mem.gpu + pair_offset) >> 32;
   pilot_use_tls(cmdbuf, pilot);

   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, subqueue);
   struct cs_index addr = pilot_addr_reg(b);

   cs_update_cmdbuf_regs(b) {
      cs_move64_to(b, addr, mem.gpu);
      cs_store64(b, fau, addr, pair_offset);
      cs_store64(b, srt, addr, pair_offset + 8);
      pilot_job_regs_store(b, addr, save_offset);
      cs_flush_stores(b);
      pilot_set_job(cmdbuf, b, dispatcher, mem.gpu);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_X), 1);
      cs_run_compute(b, 1, MALI_TASK_AXIS_X, cs_shader_res_sel(3, 3, 3, 3));
      cs_wait_slots(b, dev->csf.sb.all_iters_mask);
      pilot_job_regs_load(b, addr, save_offset);
      cs_flush_loads(b);
   }

   b->req_resource_mask |= CS_COMPUTE_RES;
}

VkResult
panvk_per_arch(cmd_pilot_queue_fs)(struct panvk_cmd_buffer *cmdbuf,
                                   const struct panvk_shader_variant *shader,
                                   uint64_t fau, uint64_t srt)
{
   const struct panvk_shader_variant *pilot = shader->preamble;
   struct panvk_pilot_entry *entry =
      util_dynarray_grow(&cmdbuf->pilots.fs, struct panvk_pilot_entry, 1);
   if (!entry)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   *entry = pilot_entry(pilot, fau, srt);
   pilot_use_tls(cmdbuf, pilot);
   return VK_SUCCESS;
}

void
panvk_per_arch(cmd_flush_fs_pilots)(struct panvk_cmd_buffer *cmdbuf)
{
   struct util_dynarray *queued = &cmdbuf->pilots.fs;
   const uint32_t count =
      util_dynarray_num_elements(queued, struct panvk_pilot_entry);

   cmdbuf->pilots.fs_after_tiling = false;
   if (!count)
      return;

   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   struct panvk_pilot_entry *entries = util_dynarray_begin(queued);
   const uint32_t save_offset = PILOT_FTZ_MODES * PILOT_FAU_STRIDE;
   struct pan_ptr mem = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, save_offset + PILOT_SAVE_SIZE, PILOT_FAU_STRIDE);
   if (!mem.gpu) {
      util_dynarray_clear(queued);
      return;
   }

   qsort(entries, count, sizeof(*entries), pilot_entry_cmp);

   struct cs_builder *b =
      panvk_get_cs_builder(cmdbuf, PANVK_SUBQUEUE_FRAGMENT);
   struct cs_index addr = pilot_addr_reg(b);
#if PAN_ARCH >= 11
   struct cs_index iter_sb = cs_scratch_reg32(b, 0);
   struct cs_index slot = cs_scratch_reg32(b, 1);
   struct cs_index mask = cs_scratch_reg32(b, 2);
#endif
   uint32_t group = 0;

   cs_update_cmdbuf_regs(b) {
      cs_move64_to(b, addr, mem.gpu);
      pilot_job_regs_store(b, addr, save_offset);
      cs_flush_stores(b);
#if PAN_ARCH >= 11
      pilot_select_own_slot(b, iter_sb, slot);
#endif

      for (uint32_t first = 0; first < count && group < PILOT_FTZ_MODES;
           group++) {
         const uint32_t ftz = entries[first].ftz;
         uint32_t end = first + 1;
         struct panvk_internal_shader *dispatcher;

         while (end < count && entries[end].ftz == ftz)
            end++;

         if (get_pilot_dispatcher(dev, ftz, &dispatcher) != VK_SUCCESS)
            break;

         const uint32_t fau_offset = group * PILOT_FAU_STRIDE;
         const uint32_t records = pilot_write_records(
            cmdbuf, entries + first, end - first, pilot_code_hi(dispatcher),
            (uint32_t *)((uint8_t *)mem.cpu + fau_offset));

         if (records) {
            pilot_set_job(cmdbuf, b, dispatcher, mem.gpu + fau_offset);
            cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_X), records);
            cs_run_compute(b, 1, MALI_TASK_AXIS_X,
                           cs_shader_res_sel(3, 3, 3, 3));
         }

         first = end;
      }

#if PAN_ARCH >= 11
      pilot_wait_own_slot(b, iter_sb, slot, mask);
#else
      cs_wait_slots(b, dev->csf.sb.all_iters_mask);
#endif
      pilot_job_regs_load(b, addr, save_offset);
      cs_flush_loads(b);
   }

   b->req_resource_mask |= CS_COMPUTE_RES;
   util_dynarray_clear(queued);
}

#else

void
panvk_per_arch(cmd_pilot_close)(struct panvk_cmd_buffer *cmdbuf,
                                enum panvk_subqueue_id subqueue)
{
}

bool
panvk_per_arch(cmd_pilot_queue)(struct panvk_cmd_buffer *cmdbuf,
                                enum panvk_subqueue_id subqueue,
                                const struct panvk_shader_variant *shader,
                                uint64_t fau, uint64_t srt,
                                const struct panvk_pilot_guard *guard)
{
   assert(!shader->preamble);
   return false;
}

void
panvk_per_arch(cmd_pilot_run)(struct panvk_cmd_buffer *cmdbuf,
                              enum panvk_subqueue_id subqueue,
                              const struct panvk_shader_variant *shader,
                              struct cs_index fau, struct cs_index srt)
{
   assert(!shader->preamble);
}

VkResult
panvk_per_arch(cmd_pilot_queue_fs)(struct panvk_cmd_buffer *cmdbuf,
                                   const struct panvk_shader_variant *shader,
                                   uint64_t fau, uint64_t srt)
{
   assert(!shader->preamble);
   return VK_SUCCESS;
}

void
panvk_per_arch(cmd_flush_fs_pilots)(struct panvk_cmd_buffer *cmdbuf)
{
   cmdbuf->pilots.fs_after_tiling = false;
   util_dynarray_clear(&cmdbuf->pilots.fs);
}

#endif
