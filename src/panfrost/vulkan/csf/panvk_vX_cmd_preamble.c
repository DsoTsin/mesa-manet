#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"

void
panvk_per_arch(cmd_emit_preamble)(struct panvk_cmd_buffer *cmdbuf,
                                  enum panvk_subqueue_id subqueue,
                                  const struct panvk_shader_variant *shader,
                                  struct cs_index fau, struct cs_index srt)
{
   if (!shader->preamble)
      return;

   assert(PAN_ARCH == 10 || PAN_ARCH == 11);
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, subqueue);
   const struct panvk_shader_variant *pilot = shader->preamble;
   struct pan_ptr saved = panvk_cmd_alloc_dev_mem(cmdbuf, desc, 512, 64);
   if (!saved.gpu)
      return;

   cmdbuf->state.tls.info.tls.size =
      MAX2(cmdbuf->state.tls.info.tls.size, pilot->info.tls_size);
   struct cs_index save_addr = cs_reg64(b, 64);
   cs_wait_slots(b, dev->csf.sb.all_iters_mask);
   cs_update_cmdbuf_regs(b)
   {
      cs_move64_to(b, save_addr, saved.gpu);
      for (unsigned i = 0; i < 64; i += 16)
         cs_store(b, cs_reg_tuple(b, i, 16), save_addr, 0xffff, i * 4);
      cs_store(b, cs_scratch_reg_tuple(b, 0, 16), save_addr, 0xffff, 256);
      cs_store(b, cs_scratch_reg_tuple(b, 16, 2), save_addr, 3, 320);
      cs_store32(b, cs_extract32(b, fau, 0), save_addr, 336);
      cs_add_imm32(b, cs_scratch_reg32(b, 0), cs_extract32(b, fau, 1),
                   -(shader->fau.total_count << 24));
      cs_store32(b, cs_scratch_reg32(b, 0), save_addr, 340);
      cs_flush_stores(b);

      cs_add_imm64(b, cs_reg64(b, PANVK_PRECOMP_SRT), srt, 0);
      cs_move64_to(b, cs_reg64(b, PANVK_PRECOMP_FAU),
                   (saved.gpu + 336) | (1ull << 56));
      cs_move64_to(b, cs_reg64(b, PANVK_PRECOMP_SPD),
                   panvk_priv_mem_dev_addr(pilot->spd));
      cs_move64_to(b, cs_reg64(b, PANVK_PRECOMP_TSD),
                   cmdbuf->state.tls.desc.gpu);
      struct mali_compute_size_workgroup_packed wg;
      pan_pack(&wg, COMPUTE_SIZE_WORKGROUP, cfg) {
         cfg.workgroup_size_x = 1;
         cfg.workgroup_size_y = 1;
         cfg.workgroup_size_z = 1;
      }
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, WG_SIZE), wg.opaque[0]);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, GLOBAL_ATTRIBUTE_OFFSET), 0);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_X), 0);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Y), 0);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Z), 0);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_X), 1);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Y), 1);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Z), 1);
      cs_run_compute(b, 1, MALI_TASK_AXIS_X, PANVK_PRECOMP_RES_SEL);
      cs_wait_slots(b, dev->csf.sb.all_iters_mask);
      cs_move32_to(b, cs_scratch_reg32(b, 0), 0);
      cs_flush_caches(b, MALI_CS_FLUSH_MODE_NONE, MALI_CS_FLUSH_MODE_CLEAN,
                      MALI_CS_OTHER_FLUSH_MODE_INVALIDATE,
                      cs_scratch_reg32(b, 0), cs_defer(0, SB_ID(IMM_FLUSH)));
      cs_wait_slot(b, SB_ID(IMM_FLUSH));

      for (unsigned i = 0; i < 64; i += 16)
         cs_load_to(b, cs_reg_tuple(b, i, 16), save_addr, 0xffff, i * 4);
      cs_load_to(b, cs_scratch_reg_tuple(b, 0, 16), save_addr, 0xffff, 256);
      cs_load_to(b, cs_scratch_reg_tuple(b, 16, 2), save_addr, 3, 320);
      cs_flush_loads(b);
   }
}
