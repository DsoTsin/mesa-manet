#include "agc_queue_state.h"

#include <stdio.h>
#include <string.h>
#include <xmmintrin.h>

#include "util/u_math.h"

#include "ac_cmdbuf.h"
#include "ac_gpu_info.h"
#include "ac_pm4.h"
#include "ac_shadowed_regs.h"
#include "sid.h"

#include "agc_cb_builders.h"
#include "agc_driver_ops.h"
#include "agc_registers.h"
#include "agc_runtime.h"

#define FIXED_BYTES UINT64_C(0x80000)
#define PREAMBLE_OFFSET UINT64_C(0)
#define RESET_OFFSET UINT64_C(0x1000)
#define CONTEXT_OFFSET UINT64_C(0x2000)
#define CONTEXT_SLOT_BYTES UINT64_C(0x400)
#define WORK_OFFSET UINT64_C(0x4000)
#define CP_DMA_CHUNK_BYTES (32767u & ~31u)
#define IB_DWORD_MULTIPLE 8u
#define INIT_TIMEOUT_NS INT64_C(5000000000)

#define SX_BLEND_OPT_OFF (S_028760_COLOR_COMB_FCN(V_028760_OPT_COMB_NONE) | S_028760_ALPHA_COMB_FCN(V_028760_OPT_COMB_NONE))

struct host_default {
   uint32_t reg;
   uint32_t value;
};

static const struct host_default host_only_defaults[] = {
   {R_03096C_GE_CNTL, 0x1f810080},
   {R_030980_GE_PC_ALLOC, 0x7fe},
   {R_0287FC_GE_MAX_OUTPUT_PER_SUBGROUP, 0x80},
   {R_028B4C_GE_NGG_SUBGRP_CNTL, 1},
   {R_028B90_VGT_GS_INSTANCE_CNT, 4},
   {R_030998_VGT_GS_OUT_PRIM_TYPE, 2},
   {R_028708_SPI_SHADER_IDX_FORMAT, 1},
   {R_02870C_SPI_SHADER_POS_FORMAT, 4},
   {R_00B21C_SPI_SHADER_PGM_RSRC3_GS, 0x003fffff},
   {R_00B204_SPI_SHADER_PGM_RSRC4_GS, 1},
   {R_028C44_PA_SC_BINNER_CNTL_0, 0x10040123},
   {R_028BE8_PA_CL_GB_VERT_CLIP_ADJ, 0x3f800000},
   {R_028BEC_PA_CL_GB_VERT_DISC_ADJ, 0x3f800000},
   {R_028BF0_PA_CL_GB_HORZ_CLIP_ADJ, 0x3f800000},
   {R_028BF4_PA_CL_GB_HORZ_DISC_ADJ, 0x3f800000},
   {R_028040_DB_Z_INFO, 0xc},
   {R_028044_DB_STENCIL_INFO, UINT32_C(1) << 29},
   {R_02882C_PA_SU_PRIM_FILTER_CNTL, 0xc0000000},
   {R_028A08_PA_SU_LINE_CNTL, 8},
   {R_028A0C_PA_SC_LINE_STIPPLE, 0x40000000},
   {R_028760_SX_MRT0_BLEND_OPT + 0 * 4, SX_BLEND_OPT_OFF},
   {R_028760_SX_MRT0_BLEND_OPT + 1 * 4, SX_BLEND_OPT_OFF},
   {R_028760_SX_MRT0_BLEND_OPT + 2 * 4, SX_BLEND_OPT_OFF},
   {R_028760_SX_MRT0_BLEND_OPT + 3 * 4, SX_BLEND_OPT_OFF},
   {R_028760_SX_MRT0_BLEND_OPT + 4 * 4, SX_BLEND_OPT_OFF},
   {R_028760_SX_MRT0_BLEND_OPT + 5 * 4, SX_BLEND_OPT_OFF},
   {R_028760_SX_MRT0_BLEND_OPT + 6 * 4, SX_BLEND_OPT_OFF},
   {R_028760_SX_MRT0_BLEND_OPT + 7 * 4, SX_BLEND_OPT_OFF},
};

static void
fail(char *error, size_t bytes, const char *text)
{
   snprintf(error, bytes, "%s", text);
}

static uint32_t *
emit_set(uint32_t *p, uint32_t reg, uint32_t value)
{
   if (reg >= CIK_UCONFIG_REG_OFFSET) {
      *p++ = PKT3(PKT3_SET_UCONFIG_REG, 1, 0);
      *p++ = (reg - CIK_UCONFIG_REG_OFFSET) >> 2;
   } else if (reg >= SI_CONTEXT_REG_OFFSET) {
      *p++ = PKT3(PKT3_SET_CONTEXT_REG, 1, 0);
      *p++ = (reg - SI_CONTEXT_REG_OFFSET) >> 2;
   } else {
      *p++ = PKT3(PKT3_SET_SH_REG, 1, 0);
      *p++ = (reg - SI_SH_REG_OFFSET) >> 2;
   }
   *p++ = value;
   return p;
}

static uint32_t *
emit_pairs(uint32_t *p, unsigned bank, const struct agc_register *registers, uint32_t count)
{
   static const uint32_t base[3] = {SI_CONTEXT_REG_OFFSET, SI_SH_REG_OFFSET, CIK_UCONFIG_REG_OFFSET};
   for (uint32_t i = 0; i < count; i++)
      p = emit_set(p, base[bank] + registers[i].offset * 4, registers[i].value);
   return p;
}

static uint32_t *
emit_pm4(uint32_t *p, const struct ac_pm4_state *pm4)
{
   memcpy(p, pm4->pm4, pm4->ndw * sizeof(uint32_t));
   return p + pm4->ndw;
}

static uint32_t *
pad(uint32_t *begin, uint32_t *p)
{
   uint32_t dwords = (uint32_t)(p - begin);
   uint32_t padded = align(dwords, IB_DWORD_MULTIPLE);
   agc_fill_nops(p, padded - dwords);
   return begin + padded;
}

static bool
run(struct agc_backend *be, struct agc_hwq *hwq, const struct agc_queue_state *state, const uint32_t *begin,
    const uint32_t *end)
{
   struct agc_ib ib = {
      .va = state->fixed.va + (uint64_t)((const char *)begin - (const char *)state->fixed.cpu),
      .dwords = (uint32_t)(end - begin),
      .flags = AGC_IB_MAIN,
   };
   _mm_sfence();
   uint64_t fence = be->ops->submit(hwq, &ib, 1);
   return fence && be->ops->fence_wait(hwq, fence, INIT_TIMEOUT_NS);
}

static uint32_t *
emit_load_range(const struct radeon_info *info, uint32_t *p, enum ac_reg_range_type type, uint64_t image_va)
{
   struct ac_pm4_state *pm4 = ac_pm4_create_sized(info, false, 256, false);
   if (!pm4)
      return NULL;
   ac_build_load_reg(info, pm4, type, image_va);
   p = emit_pm4(p, pm4);
   ac_pm4_free_state(pm4);
   return p;
}

static uint32_t *
emit_loads(const struct radeon_info *info, uint32_t *p, uint64_t image_va)
{
   for (unsigned type = 0; p && type < SI_NUM_REG_RANGES; type++)
      p = emit_load_range(info, p, type, image_va);
   return p;
}

static uint32_t *
emit_copy(uint32_t *p, uint64_t dst, uint64_t src, uint32_t bytes)
{
   *p++ = PKT3(PKT3_PFP_SYNC_ME, 0, 0);
   *p++ = 0;
   for (uint32_t done = 0; done < bytes;) {
      uint32_t chunk = bytes - done > CP_DMA_CHUNK_BYTES ? CP_DMA_CHUNK_BYTES : bytes - done;
      bool last = done + chunk == bytes;
      *p++ = PKT3(PKT3_DMA_DATA, 5, 0);
      *p++ = S_501_ENGINE_SEL(V_501_PREFETCH_PARSER) | S_501_SRC_SEL(V_501_SRC_ADDR_USING_L2) |
             S_501_DST_SEL(V_501_DST_ADDR_USING_L2) | S_501_CP_SYNC(last);
      *p++ = (uint32_t)(src + done);
      *p++ = (uint32_t)((src + done) >> 32);
      *p++ = (uint32_t)(dst + done);
      *p++ = (uint32_t)((dst + done) >> 32);
      *p++ = S_506_BYTE_COUNT(chunk);
      done += chunk;
   }
   return p;
}

static uint32_t *
emit_context_op(const struct radeon_info *info, const struct agc_queue_state *state, uint32_t *p, uint32_t op)
{
   uint64_t shadow = state->shadow.va + SI_SHADOWED_CONTEXT_REG_OFFSET;
   uint64_t defaults = state->defaults.va + SI_SHADOWED_CONTEXT_REG_OFFSET;
   if (op == 1 || op == 3)
      p = emit_copy(p, state->saved.va, shadow, SI_CONTEXT_REG_SPACE_SIZE);
   if (op == 0 || op == 3)
      p = emit_copy(p, shadow, defaults, SI_CONTEXT_REG_SPACE_SIZE);
   if (op == 2)
      p = emit_copy(p, shadow, state->saved.va, SI_CONTEXT_REG_SPACE_SIZE);
   if (op != 1)
      p = emit_load_range(info, p, SI_REG_RANGE_CONTEXT, state->shadow.va);
   return p;
}

void
agc_queue_state_fini(struct agc_queue_state *state)
{
   struct agc_mem *all[] = {&state->shadow, &state->defaults, &state->fixed, &state->attribute_ring, &state->saved};
   for (unsigned i = 0; i < ARRAY_SIZE(all); i++) {
      if (all[i]->cpu)
         agc_mem_free(all[i]);
   }
   memset(state, 0, sizeof(*state));
}

bool
agc_queue_state_init(struct agc_backend *be, const struct radeon_info *info, struct agc_hwq *hwq,
                     const struct agc_host_defaults *defaults, struct agc_queue_state *state, char *error,
                     size_t error_bytes)
{
   memset(state, 0, sizeof(*state));
   state->image_bytes = align64(SI_SHADOWED_REG_BUFFER_SIZE, 0x10000);
   state->attribute_ring_bytes = align64((uint64_t)info->attribute_ring_size_per_se * MAX2(info->max_se, 1), 0x10000);
   if (!agc_mem_alloc(state->image_bytes, 0, &state->shadow) ||
       !agc_mem_alloc(state->image_bytes, 0, &state->defaults) || !agc_mem_alloc(FIXED_BYTES, 0, &state->fixed) ||
       !agc_mem_alloc_at(AGC_RUNTIME_ATTRIBUTE_RING_VA, state->attribute_ring_bytes, 0, &state->attribute_ring) ||
       !agc_mem_alloc(align64(SI_CONTEXT_REG_SPACE_SIZE, 0x10000), 0, &state->saved)) {
      fail(error, error_bytes, "out of driver memory for the queue state");
      agc_queue_state_fini(state);
      return false;
   }
   memset(state->shadow.cpu, 0, state->image_bytes);
   memset(state->defaults.cpu, 0, state->image_bytes);

   uint32_t *fixed = state->fixed.cpu;
   uint32_t *limit = fixed + FIXED_BYTES / 4 - 64;
   struct ac_pm4_state *preamble = ac_create_shadowing_ib_preamble(info, state->shadow.va, false);
   if (!preamble) {
      fail(error, error_bytes, "no register shadowing preamble for this chip");
      agc_queue_state_fini(state);
      return false;
   }
   uint32_t *begin = fixed + PREAMBLE_OFFSET / 4;
   uint32_t *p = pad(begin, emit_pm4(begin, preamble));
   state->preamble_va = state->fixed.va + PREAMBLE_OFFSET;
   state->preamble_dwords = (uint32_t)(p - begin);

   begin = fixed + WORK_OFFSET / 4;
   struct ac_preamble_state config = {.gfx11.compute_dispatch_interleave = 64};
   struct ac_pm4_state *graphics = ac_pm4_create_sized(info, false, 1024, false);
   struct ac_pm4_state *compute = ac_pm4_create_sized(info, false, 256, true);
   bool ok = graphics && compute;
   if (ok) {
      ac_init_graphics_preamble_state(&config, graphics);
      ac_pm4_finalize(graphics);
      ac_init_compute_preamble_state(&config, compute);
      ac_pm4_finalize(compute);
      p = emit_pm4(begin, preamble);
      p = emit_pm4(p, compute);
      p = emit_pm4(p, graphics);
      if (info->gfx_level >= GFX11 && info->gfx_level < GFX12) {
         p = emit_set(p, R_031110_SPI_GS_THROTTLE_CNTL1, 0x12355123);
         p = emit_set(p, R_031114_SPI_GS_THROTTLE_CNTL2, 0x1544d);
         p = emit_set(p, R_031118_SPI_ATTRIBUTE_RING_BASE, (uint32_t)(state->attribute_ring.va >> 16));
         p = emit_set(p, R_03111C_SPI_ATTRIBUTE_RING_SIZE,
                      S_03111C_MEM_SIZE((info->attribute_ring_size_per_se >> 16) - 1) | S_03111C_L1_POLICY(1));
         for (unsigned i = 0; i < ARRAY_SIZE(host_only_defaults); i++)
            p = emit_set(p, host_only_defaults[i].reg, host_only_defaults[i].value);
      }
      for (unsigned bank = 0; defaults && bank < 3; bank++) {
         if (p + (size_t)defaults->count[bank] * 3 > limit) {
            ok = false;
            break;
         }
         p = emit_pairs(p, bank, defaults->registers[bank], defaults->count[bank]);
      }
      if (ok)
         p = emit_copy(p, state->defaults.va, state->shadow.va, (uint32_t)SI_SHADOWED_REG_BUFFER_SIZE);
      ok = ok && run(be, hwq, state, begin, pad(begin, p));
   }
   if (graphics)
      ac_pm4_free_state(graphics);
   if (compute)
      ac_pm4_free_state(compute);
   ac_pm4_free_state(preamble);
   if (!ok) {
      fail(error, error_bytes, "loading the default state into the register shadow did not complete");
      agc_queue_state_fini(state);
      return false;
   }
   begin = fixed + RESET_OFFSET / 4;
   p = emit_copy(begin, state->shadow.va, state->defaults.va, (uint32_t)SI_SHADOWED_REG_BUFFER_SIZE);
   p = emit_loads(info, p, state->shadow.va);
   if (!p || (uint64_t)(pad(begin, p) - begin) * 4 > CONTEXT_OFFSET - RESET_OFFSET) {
      fail(error, error_bytes, "the reset sequence does not fit its slot");
      agc_queue_state_fini(state);
      return false;
   }
   p = pad(begin, p);
   state->reset_va = state->fixed.va + RESET_OFFSET;
   state->reset_dwords = (uint32_t)(p - begin);

   for (uint32_t op = 0; op < AGC_CONTEXT_OPS; op++) {
      begin = fixed + (CONTEXT_OFFSET + op * CONTEXT_SLOT_BYTES) / 4;
      p = emit_context_op(info, state, begin, op);
      if (!p || (uint64_t)(pad(begin, p) - begin) * 4 > CONTEXT_SLOT_BYTES) {
         fail(error, error_bytes, "a context state sequence does not fit its slot");
         agc_queue_state_fini(state);
         return false;
      }
      p = pad(begin, p);
      state->context_va[op] = state->fixed.va + CONTEXT_OFFSET + op * CONTEXT_SLOT_BYTES;
      state->context_dwords[op] = (uint32_t)(p - begin);
   }
   _mm_sfence();
   return true;
}

bool
agc_queue_state_read(const struct agc_queue_state *state, unsigned bank, uint32_t offset, uint32_t *value)
{
   static const uint64_t base[3] = {SI_SHADOWED_CONTEXT_REG_OFFSET, SI_SHADOWED_SH_REG_OFFSET,
                                    SI_SHADOWED_UCONFIG_REG_OFFSET};
   static const uint32_t size[3] = {SI_CONTEXT_REG_SPACE_SIZE, SI_SH_REG_SPACE_SIZE, SI_UCONFIG_REG_SPACE_SIZE};
   if (bank >= 3 || !state->shadow.cpu || (uint64_t)offset * 4 >= size[bank])
      return false;
   const volatile uint32_t *image = state->shadow.cpu;
   *value = image[(base[bank] + (uint64_t)offset * 4) / 4];
   return true;
}
