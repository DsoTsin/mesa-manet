#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include "util/macros.h"
#include "util/os_time.h"
#include "util/u_math.h"

#include "ac_gpu_info.h"
#include "ac_pm4.h"
#include "ac_shadowed_regs.h"
#include "sid.h"

#include "agc_driver_ops.h"

#define M1_OK 0
#define M1_FAILED 1
#define M1_UNCONFIRMED 2

#define M1_BUFFER_BYTES UINT64_C(0x10000)
#define M1_GUARD 0xa5a5a5a5u
#define M1_IB_START_BYTES 256u
#define M1_IB_DWORD_MULTIPLE 8u

struct m1 {
   struct agc_backend *be;
   const struct agc_driver_ops *ops;
   struct agc_hwq *queue;
   enum agc_queue_kind kind;
   struct agc_bo *code;
   struct agc_bo *aux;
   struct agc_bo *data;
   uint32_t *code_cpu;
   uint32_t *aux_cpu;
   uint32_t *data_cpu;
   uint64_t code_va;
   uint64_t aux_va;
   uint64_t data_va;
   uint32_t code_used;
   int64_t timeout_ns;
};

static void say(const char *name, const char *format, ...) AGC_PRINTF(2, 3);

static void
say(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   printf("%s: ", name);
   vprintf(format, args);
   putchar('\n');
   va_end(args);
   fflush(stdout);
}

static enum agc_device_state
print_state(struct m1 *m)
{
   static const char *const names[] = {
      [AGC_DEVICE_ACTIVE] = "active",         [AGC_DEVICE_HUNG] = "hung",
      [AGC_DEVICE_PAGE_FAULT] = "page-fault", [AGC_DEVICE_REMOVED] = "removed",
      [AGC_DEVICE_UNKNOWN] = "unknown",
   };
   struct agc_wddm_device_state detail;
   uint64_t fault_va = 0;
   enum agc_device_state state = m->ops->device_state(m->be, &fault_va);
   agc_wddm_device_state_detail(m->be, &detail);
   say("device state", "%s, fault va 0x%" PRIx64 ", execution %d, page fault status %d, fault stage %u",
       names[state], fault_va, detail.execution, detail.page_fault_status, detail.fault_stage);
   return state;
}

static uint64_t
data_va(const struct m1 *m, uint32_t slot)
{
   return m->data_va + (uint64_t)slot * 4;
}

static uint32_t *
emit_nops(uint32_t *p, uint32_t dwords)
{
   if (dwords == 1) {
      *p++ = PKT3_NOP_PAD;
   } else if (dwords > 1) {
      *p++ = PKT3(PKT3_NOP, dwords - 2, 0);
      for (uint32_t i = 1; i < dwords; i++)
         *p++ = 0;
   }
   return p;
}

static uint32_t *
emit_set(uint32_t *p, uint32_t opcode, uint32_t base, uint32_t reg, const uint32_t *values, uint32_t count)
{
   *p++ = PKT3(opcode, count, 0);
   *p++ = (reg - base) >> 2;
   memcpy(p, values, count * sizeof(uint32_t));
   return p + count;
}

static uint32_t *
emit_copy_reg(uint32_t *p, uint32_t reg, uint64_t dst)
{
   *p++ = PKT3(PKT3_COPY_DATA, 4, 0);
   *p++ = COPY_DATA_SRC_SEL(COPY_DATA_REG) | COPY_DATA_DST_SEL(COPY_DATA_DST_MEM) | COPY_DATA_WR_CONFIRM;
   *p++ = reg >> 2;
   *p++ = 0;
   *p++ = (uint32_t)dst;
   *p++ = (uint32_t)(dst >> 32);
   return p;
}

static uint32_t *
emit_copy_mem(uint32_t *p, uint64_t src, uint64_t dst)
{
   *p++ = PKT3(PKT3_COPY_DATA, 4, 0);
   *p++ = COPY_DATA_SRC_SEL(COPY_DATA_SRC_MEM) | COPY_DATA_DST_SEL(COPY_DATA_DST_MEM) | COPY_DATA_WR_CONFIRM;
   *p++ = (uint32_t)src;
   *p++ = (uint32_t)(src >> 32);
   *p++ = (uint32_t)dst;
   *p++ = (uint32_t)(dst >> 32);
   return p;
}

static uint32_t *
emit_write(uint32_t *p, uint64_t dst, uint32_t value)
{
   *p++ = PKT3(PKT3_WRITE_DATA, 3, 0);
   *p++ = S_371_DST_SEL(V_371_MEMORY) | S_371_WR_CONFIRM(V_371_WAIT_FOR_WRITE_CONFIRMATION) |
          S_371_ENGINE_SEL(V_371_MICRO_ENGINE);
   *p++ = (uint32_t)dst;
   *p++ = (uint32_t)(dst >> 32);
   *p++ = value;
   return p;
}

static uint32_t *
emit_indirect(uint32_t *p, uint64_t target, uint32_t dwords, bool chain)
{
   *p++ = PKT3(PKT3_INDIRECT_BUFFER, 2, 0);
   *p++ = (uint32_t)target;
   *p++ = (uint32_t)(target >> 32);
   *p++ = S_3F3_IB_SIZE(dwords) | S_3F3_CHAIN(chain) | S_3F3_VALID(1);
   return p;
}

static uint32_t *
ib_begin(struct m1 *m)
{
   m->code_used = align(m->code_used * 4, M1_IB_START_BYTES) / 4;
   return m->code_cpu + m->code_used;
}

static int
ib_submit(struct m1 *m, const uint32_t *begin, uint32_t dwords, const char *what)
{
   struct agc_ib ib = {
      .va = m->code_va + (uint64_t)(begin - m->code_cpu) * 4,
      .dwords = dwords,
      .flags = AGC_IB_MAIN,
   };
   _mm_sfence();
   uint64_t fence = m->ops->submit(m->queue, &ib, 1);
   if (!fence) {
      struct agc_wddm_description description;
      agc_wddm_describe(m->be, &description);
      say("FAIL submit", "%s: the submission was refused", what);
      if (!description.lost)
         return M1_FAILED;
      print_state(m);
      return M1_UNCONFIRMED;
   }
   int64_t start = os_time_get_nano();
   if (!m->ops->fence_wait(m->queue, fence, m->timeout_ns)) {
      say("FAIL wait", "%s: fence %" PRIu64 " not reached within %" PRId64 " ms, completed %" PRIu64, what,
          fence, m->timeout_ns / 1000000, m->ops->fence_completed(m->queue));
      print_state(m);
      return M1_UNCONFIRMED;
   }
   say("ran", "%s: IB 0x%" PRIx64 " of %u dwords, fence %" PRIu64 " in %.3f ms", what, ib.va, dwords, fence,
       (double)(os_time_get_nano() - start) / 1e6);
   return M1_OK;
}

static int
ib_run(struct m1 *m, uint32_t *begin, uint32_t *end, const char *what)
{
   uint32_t dwords = (uint32_t)(end - begin);
   uint32_t padded = align(dwords, M1_IB_DWORD_MULTIPLE);
   emit_nops(end, padded - dwords);
   m->code_used = (uint32_t)(begin - m->code_cpu) + padded;
   return ib_submit(m, begin, padded, what);
}

static void
data_reset(struct m1 *m)
{
   for (uint32_t i = 0; i < M1_BUFFER_BYTES / 4; i++)
      m->data_cpu[i] = M1_GUARD;
}

static uint32_t
data_get(const struct m1 *m, uint32_t slot)
{
   const volatile uint32_t *words = m->data_cpu;
   return words[slot];
}

static bool
expect(const struct m1 *m, const char *what, uint32_t slot, uint32_t value)
{
   uint32_t got = data_get(m, slot);
   say(got == value ? "PASS" : "FAIL", "%s: slot %u is 0x%08x, expected 0x%08x", what, slot, got, value);
   return got == value;
}

static struct agc_bo *
buffer(struct m1 *m, const char *name, uint32_t **cpu, uint64_t *va)
{
   struct agc_bo *bo = m->ops->bo_create(m->be, M1_BUFFER_BYTES, AGC_BO_GL2_BYPASS, 0);
   if (!bo) {
      say("FAIL buffer", "%s: allocation failed", name);
      return NULL;
   }
   *cpu = m->ops->bo_cpu(bo);
   *va = m->ops->bo_va(bo);
   say("buffer", "%s: GPU va 0x%" PRIx64 ", CPU %p", name, *va, (void *)*cpu);
   return bo;
}

static int
m1_open(struct m1 *m)
{
   m->be = agc_wddm_open();
   if (!m->be) {
      say("FAIL open", "no AMD adapter with a verified profile could be opened");
      return M1_FAILED;
   }
   m->ops = m->be->ops;
   m->queue = m->ops->queue_create(m->be, m->kind);
   if (!m->queue) {
      say("FAIL queue", "no %s hardware queue", m->kind == AGC_QUEUE_GFX ? "graphics" : "compute");
      return M1_FAILED;
   }
   say("queue", "%s", m->kind == AGC_QUEUE_GFX ? "graphics" : "compute");
   m->code = buffer(m, "code", &m->code_cpu, &m->code_va);
   m->aux = buffer(m, "aux", &m->aux_cpu, &m->aux_va);
   m->data = buffer(m, "data", &m->data_cpu, &m->data_va);
   if (!m->code || !m->aux || !m->data)
      return M1_FAILED;
   data_reset(m);
   return M1_OK;
}

struct reg {
   const char *name;
   uint32_t opcode;
   uint32_t base;
   uint32_t address;
   uint32_t first;
   uint32_t second;
};

static const struct reg cx_regs[] = {
   {"CB_BLEND_RED", PKT3_SET_CONTEXT_REG, SI_CONTEXT_REG_OFFSET, R_028414_CB_BLEND_RED, 0x3dcccccd, 0x3f000001},
   {"CB_BLEND_GREEN", PKT3_SET_CONTEXT_REG, SI_CONTEXT_REG_OFFSET, R_028414_CB_BLEND_RED + 4, 0x3e4ccccd, 0x3f000002},
   {"CB_BLEND_BLUE", PKT3_SET_CONTEXT_REG, SI_CONTEXT_REG_OFFSET, R_028414_CB_BLEND_RED + 8, 0x3e99999a, 0x3f000003},
   {"CB_BLEND_ALPHA", PKT3_SET_CONTEXT_REG, SI_CONTEXT_REG_OFFSET, R_028414_CB_BLEND_RED + 12, 0x3ecccccd, 0x3f000004},
};

static const struct reg sh_regs[] = {
   {"COMPUTE_USER_DATA_0", PKT3_SET_SH_REG, SI_SH_REG_OFFSET, R_00B900_COMPUTE_USER_DATA_0, 0x11110001, 0x22220001},
   {"COMPUTE_USER_DATA_1", PKT3_SET_SH_REG, SI_SH_REG_OFFSET, R_00B900_COMPUTE_USER_DATA_0 + 4, 0x11110002, 0x22220002},
   {"COMPUTE_USER_DATA_2", PKT3_SET_SH_REG, SI_SH_REG_OFFSET, R_00B900_COMPUTE_USER_DATA_0 + 8, 0x11110003, 0x22220003},
   {"COMPUTE_USER_DATA_3", PKT3_SET_SH_REG, SI_SH_REG_OFFSET, R_00B900_COMPUTE_USER_DATA_0 + 12, 0x11110004, 0x22220004},
};

static const struct reg uc_regs[] = {
   {"VGT_INSTANCE_BASE_ID", PKT3_SET_UCONFIG_REG, CIK_UCONFIG_REG_OFFSET, R_030968_VGT_INSTANCE_BASE_ID, 0x00c0ffee, 0x00facade},
   {"VGT_TF_MEMORY_BASE", PKT3_SET_UCONFIG_REG, CIK_UCONFIG_REG_OFFSET, R_030940_VGT_TF_MEMORY_BASE, 0x00123400, 0x00432100},
};

struct bank {
   const char *name;
   const struct reg *regs;
   uint32_t count;
   uint32_t load_opcode;
   bool graphics_only;
};

static const struct bank banks[] = {
   {"cx", cx_regs, ARRAY_SIZE(cx_regs), PKT3_LOAD_CONTEXT_REG_INDEX, true},
   {"sh", sh_regs, ARRAY_SIZE(sh_regs), PKT3_LOAD_SH_REG_INDEX, false},
   {"uc", uc_regs, ARRAY_SIZE(uc_regs), PKT3_LOAD_UCONFIG_REG_INDEX, true},
};

static bool
bank_usable(const struct m1 *m, const struct bank *bank)
{
   return m->kind == AGC_QUEUE_GFX || !bank->graphics_only;
}

static uint32_t *
emit_readback(struct m1 *m, uint32_t *p, uint32_t first_slot)
{
   uint32_t slot = first_slot;
   for (uint32_t b = 0; b < ARRAY_SIZE(banks); b++) {
      for (uint32_t r = 0; r < banks[b].count; r++, slot++) {
         if (bank_usable(m, &banks[b]))
            p = emit_copy_reg(p, banks[b].regs[r].address, data_va(m, slot));
      }
   }
   return p;
}

static void
report_readback(const struct m1 *m, const char *what, uint32_t first_slot, bool second)
{
   uint32_t slot = first_slot;
   for (uint32_t b = 0; b < ARRAY_SIZE(banks); b++) {
      for (uint32_t r = 0; r < banks[b].count; r++, slot++) {
         const struct reg *reg = &banks[b].regs[r];
         char text[96];
         if (!bank_usable(m, &banks[b]))
            continue;
         snprintf(text, sizeof(text), "%s %s", what, reg->name);
         expect(m, text, slot, second ? reg->second : reg->first);
      }
   }
}

static int
cmd_persist(struct m1 *m)
{
   uint32_t *begin = ib_begin(m);
   uint32_t *p = begin;
   for (uint32_t b = 0; b < ARRAY_SIZE(banks); b++) {
      for (uint32_t r = 0; r < banks[b].count && bank_usable(m, &banks[b]); r++) {
         const struct reg *reg = &banks[b].regs[r];
         p = emit_set(p, reg->opcode, reg->base, reg->address, &reg->first, 1);
      }
   }
   p = emit_readback(m, p, 0);
   int result = ib_run(m, begin, p, "set and read back in one IB");
   if (result != M1_OK)
      return result;
   report_readback(m, "same IB:", 0, false);

   static const unsigned delays_ms[] = {0, 300, 1500};
   for (uint32_t i = 0; i < ARRAY_SIZE(delays_ms); i++) {
      char what[96];
      os_time_sleep((int64_t)delays_ms[i] * 1000);
      begin = ib_begin(m);
      p = emit_readback(m, begin, 16 * (i + 1));
      snprintf(what, sizeof(what), "read back in a later submission after %u ms", delays_ms[i]);
      result = ib_run(m, begin, p, what);
      if (result != M1_OK)
         return result;
      snprintf(what, sizeof(what), "later submission (+%u ms):", delays_ms[i]);
      report_readback(m, what, 16 * (i + 1), false);
   }
   return M1_OK;
}

static int
cmd_rmw(struct m1 *m)
{
   const uint32_t before = 0x12345678;
   const uint32_t mask = 0x0000ffff;
   const uint32_t data = 0xabcd4321;
   uint32_t *begin = ib_begin(m);
   uint32_t *p = emit_set(begin, PKT3_SET_CONTEXT_REG, SI_CONTEXT_REG_OFFSET, R_028414_CB_BLEND_RED, &before, 1);
   *p++ = PKT3(PKT3_CONTEXT_REG_RMW, 2, 0);
   *p++ = (R_028414_CB_BLEND_RED - SI_CONTEXT_REG_OFFSET) >> 2;
   *p++ = mask;
   *p++ = data;
   p = emit_copy_reg(p, R_028414_CB_BLEND_RED, data_va(m, 0));
   int result = ib_run(m, begin, p, "CONTEXT_REG_RMW");
   if (result != M1_OK)
      return result;
   uint32_t got = data_get(m, 0);
   say("value", "before 0x%08x, mask 0x%08x, data 0x%08x, after 0x%08x", before, mask, data, got);
   if (got == ((before & ~mask) | (data & mask)))
      say("PASS", "CONTEXT_REG_RMW: reg = (reg & ~mask) | (data & mask)");
   else if (got == ((before & mask) | data))
      say("INFO", "CONTEXT_REG_RMW: reg = (reg & mask) | data");
   else
      say("FAIL", "CONTEXT_REG_RMW: neither known meaning matches");
   return M1_OK;
}

static int
cmd_load(struct m1 *m, const char *bank_name)
{
   const struct bank *bank = NULL;
   for (uint32_t b = 0; b < ARRAY_SIZE(banks); b++) {
      if (!strcmp(banks[b].name, bank_name))
         bank = &banks[b];
   }
   if (!bank || !bank_usable(m, bank)) {
      say("FAIL load", "bank %s is unknown or not usable on this queue", bank_name);
      return M1_FAILED;
   }

   const uint32_t pairs_slot = 0x400;
   uint32_t pairs = 0;
   for (uint32_t r = bank->count; r-- > 1;) {
      m->data_cpu[pairs_slot + pairs * 2] = (bank->regs[r].address - bank->regs[r].base) >> 2;
      m->data_cpu[pairs_slot + pairs * 2 + 1] = bank->regs[r].second;
      pairs++;
   }

   uint32_t *begin = ib_begin(m);
   uint32_t *p = begin;
   for (uint32_t r = 0; r < bank->count; r++)
      p = emit_set(p, bank->regs[r].opcode, bank->regs[r].base, bank->regs[r].address, &bank->regs[r].first, 1);
   uint64_t address = data_va(m, pairs_slot);
   *p++ = PKT3(bank->load_opcode, 3, 0);
   *p++ = (uint32_t)address & ~UINT32_C(3);
   *p++ = (uint32_t)(address >> 32);
   *p++ = S_9F3_DATA_FORMAT(V_9F3_OFFSET_AND_DATA);
   *p++ = pairs;
   for (uint32_t r = 0; r < bank->count; r++)
      p = emit_copy_reg(p, bank->regs[r].address, data_va(m, r));
   char what[96];
   snprintf(what, sizeof(what), "indexed load of %u offset-and-data pairs, bank %s", pairs, bank->name);
   int result = ib_run(m, begin, p, what);
   if (result != M1_OK)
      return result;

   uint32_t loaded = 0;
   for (uint32_t r = 0; r < bank->count; r++) {
      uint32_t got = data_get(m, r);
      const char *state = got == bank->regs[r].second ? "loaded" : got == bank->regs[r].first ? "unchanged" : "other";
      say("value", "%s is 0x%08x (%s; set 0x%08x, pair 0x%08x)", bank->regs[r].name, got, state,
          bank->regs[r].first, bank->regs[r].second);
      if (r > 0 && got == bank->regs[r].second)
         loaded++;
   }
   bool untouched = data_get(m, 0) == bank->regs[0].first;
   if (loaded == pairs && untouched)
      say("PASS", "bank %s: the count is in pairs and every pair was loaded", bank->name);
   else
      say("FAIL", "bank %s: %u of %u pairs loaded, first register %s", bank->name, loaded, pairs,
          untouched ? "untouched" : "changed");
   return M1_OK;
}

static int
chain_case(struct m1 *m, const char *what, uint32_t aux_byte_offset, uint32_t extra_nops, uint32_t slot,
           uint32_t value)
{
   uint32_t *target = m->aux_cpu + aux_byte_offset / 4;
   uint32_t *t = emit_write(target, data_va(m, slot), value);
   t = emit_copy_mem(t, data_va(m, slot), data_va(m, slot + 1));
   t = emit_nops(t, extra_nops);
   uint32_t target_dwords = (uint32_t)(t - target);

   uint32_t *begin = ib_begin(m);
   uint32_t *p = emit_nops(begin, 4);
   p = emit_indirect(p, m->aux_va + aux_byte_offset, target_dwords, true);
   char text[128];
   snprintf(text, sizeof(text), "%s: chain to +0x%x, %u dwords", what, aux_byte_offset, target_dwords);
   m->code_used = (uint32_t)(p - m->code_cpu);
   int result = ib_submit(m, begin, (uint32_t)(p - begin), text);
   if (result != M1_OK)
      return result;
   expect(m, what, slot, value);
   expect(m, what, slot + 1, value);
   return M1_OK;
}

static int
cmd_chain(struct m1 *m, const char *which)
{
   int result;
   if (!strcmp(which, "aligned"))
      return chain_case(m, "aligned target, 16 dwords", 0x000, 5, 0, 0x1001);
   if (!strcmp(which, "odd-size"))
      return chain_case(m, "aligned target, 13 dwords", 0x100, 2, 2, 0x1002);
   if (!strcmp(which, "odd-start"))
      return chain_case(m, "target at +4 bytes, 16 dwords", 0x204, 5, 4, 0x1003);
   if (!strcmp(which, "odd-both"))
      return chain_case(m, "target at +0x24 bytes, 11 dwords", 0x324, 0, 6, 0x1004);

   if (!strcmp(which, "back")) {
      uint32_t *begin = ib_begin(m);
      const uint32_t total = 24;
      const uint32_t rest_at = 9;
      uint32_t *stub = m->aux_cpu + 0x400 / 4;
      uint64_t main_va = m->code_va + (uint64_t)(begin - m->code_cpu) * 4;

      uint32_t *p = emit_write(begin, data_va(m, 8), 0x2001);
      uint32_t *s = emit_copy_mem(stub, data_va(m, 8), data_va(m, 9));
      s = emit_nops(s, 6);
      s = emit_indirect(s, main_va + rest_at * 4, total - rest_at, true);
      p = emit_indirect(p, m->aux_va + 0x400, (uint32_t)(s - stub), true);
      if (p != begin + rest_at) {
         say("FAIL chain", "the trampoline layout is wrong");
         return M1_FAILED;
      }
      p = emit_copy_mem(p, data_va(m, 9), data_va(m, 10));
      p = emit_nops(p, total - (uint32_t)(p - begin));
      m->code_used = (uint32_t)(p - m->code_cpu);
      result = ib_submit(m, begin, total, "chain out of the middle of an IB and back to dword 9");
      if (result != M1_OK)
         return result;
      expect(m, "before the chain", 8, 0x2001);
      expect(m, "in the stub", 9, 0x2001);
      expect(m, "after chaining back", 10, 0x2001);
      return M1_OK;
   }

   if (!strcmp(which, "call")) {
      uint32_t *sub = m->aux_cpu + 0x800 / 4;
      uint32_t *s = emit_copy_mem(sub, data_va(m, 12), data_va(m, 13));
      s = emit_nops(s, 2);
      uint32_t *begin = ib_begin(m);
      uint32_t *p = emit_write(begin, data_va(m, 12), 0x3001);
      p = emit_indirect(p, m->aux_va + 0x800, (uint32_t)(s - sub), false);
      p = emit_copy_mem(p, data_va(m, 13), data_va(m, 14));
      result = ib_run(m, begin, p, "call a second-level IB and return");
      if (result != M1_OK)
         return result;
      expect(m, "before the call", 12, 0x3001);
      expect(m, "in the called IB", 13, 0x3001);
      expect(m, "after the return", 14, 0x3001);
      return M1_OK;
   }

   say("FAIL chain", "unknown case %s", which);
   return M1_FAILED;
}

static struct agc_bo *
big_buffer(struct m1 *m, const char *name, uint64_t bytes, uint32_t **cpu, uint64_t *va)
{
   struct agc_bo *bo = m->ops->bo_create(m->be, bytes, AGC_BO_GL2_BYPASS, 0);
   if (!bo) {
      say("FAIL buffer", "%s: allocation of 0x%" PRIx64 " bytes failed", name, bytes);
      return NULL;
   }
   *cpu = m->ops->bo_cpu(bo);
   *va = m->ops->bo_va(bo);
   memset(*cpu, 0, bytes);
   say("buffer", "%s: GPU va 0x%" PRIx64 ", 0x%" PRIx64 " bytes", name, *va, bytes);
   return bo;
}

static int
submit_ibs(struct m1 *m, const struct agc_ib *ibs, uint32_t count, const char *what, double *milliseconds)
{
   _mm_sfence();
   int64_t start = os_time_get_nano();
   uint64_t fence = m->ops->submit(m->queue, ibs, count);
   if (!fence) {
      struct agc_wddm_description description;
      agc_wddm_describe(m->be, &description);
      say("FAIL submit", "%s: the submission was refused", what);
      if (!description.lost)
         return M1_FAILED;
      print_state(m);
      return M1_UNCONFIRMED;
   }
   if (!m->ops->fence_wait(m->queue, fence, m->timeout_ns)) {
      say("FAIL wait", "%s: fence %" PRIu64 " not reached within %" PRId64 " ms", what, fence,
          m->timeout_ns / 1000000);
      print_state(m);
      return M1_UNCONFIRMED;
   }
   double elapsed = (double)(os_time_get_nano() - start) / 1e6;
   if (milliseconds)
      *milliseconds = elapsed;
   else
      say("ran", "%s: %u IBs, fence %" PRIu64 " in %.3f ms", what, count, fence, elapsed);
   return M1_OK;
}

struct shadow {
   struct agc_bo *bo;
   uint32_t *cpu;
   uint64_t va;
   uint32_t *preamble;
   uint64_t preamble_va;
   uint32_t preamble_dwords;
};

static int
shadow_setup(struct m1 *m, struct shadow *shadow)
{
   struct radeon_info info;
   memset(&info, 0, sizeof(info));
   if (!m->ops->query_info(m->be, &info)) {
      say("FAIL shadow", "the GPU did not describe itself");
      return M1_FAILED;
   }
   uint64_t bytes = align64(SI_SHADOWED_REG_BUFFER_SIZE, 0x10000);
   shadow->bo = big_buffer(m, "shadow", bytes, &shadow->cpu, &shadow->va);
   if (!shadow->bo)
      return M1_FAILED;
   struct ac_pm4_state *pm4 = ac_create_shadowing_ib_preamble(&info, shadow->va, false);
   if (!pm4) {
      say("FAIL shadow", "no shadowing preamble for this chip");
      return M1_FAILED;
   }
   shadow->preamble_dwords = align(pm4->ndw, M1_IB_DWORD_MULTIPLE);
   shadow->preamble = m->aux_cpu + 0x4000 / 4;
   shadow->preamble_va = m->aux_va + 0x4000;
   memcpy(shadow->preamble, pm4->pm4, pm4->ndw * sizeof(uint32_t));
   emit_nops(shadow->preamble + pm4->ndw, shadow->preamble_dwords - pm4->ndw);
   say("shadow", "preamble of %u dwords (%u padded), shadow memory 0x%" PRIx64 " bytes", pm4->ndw,
       shadow->preamble_dwords, bytes);
   ac_pm4_free_state(pm4);
   return M1_OK;
}

static int
shadow_run(struct m1 *m, const struct shadow *shadow, bool separate, uint32_t *begin, uint32_t *end,
           const char *what)
{
   uint32_t dwords = (uint32_t)(end - begin);
   uint32_t padded = align(dwords, M1_IB_DWORD_MULTIPLE);
   emit_nops(end, padded - dwords);
   m->code_used = (uint32_t)(begin - m->code_cpu) + padded;
   struct agc_ib ibs[2];
   uint32_t count = 0;
   if (separate) {
      ibs[count++] = (struct agc_ib){
         .va = shadow->preamble_va,
         .dwords = shadow->preamble_dwords,
         .flags = AGC_IB_PREAMBLE,
      };
   }
   ibs[count++] = (struct agc_ib){
      .va = m->code_va + (uint64_t)(begin - m->code_cpu) * 4,
      .dwords = padded,
      .flags = AGC_IB_MAIN,
   };
   return submit_ibs(m, ibs, count, what, NULL);
}

static uint32_t *
shadow_begin(struct m1 *m, const struct shadow *shadow, bool separate)
{
   uint32_t *begin = ib_begin(m);
   if (separate)
      return begin;
   memcpy(begin, shadow->preamble, shadow->preamble_dwords * sizeof(uint32_t));
   return begin + shadow->preamble_dwords;
}

static int
cmd_shadow(struct m1 *m, const char *mode)
{
   bool separate = !strcmp(mode, "preamble");
   struct shadow shadow = {0};
   if (!separate && strcmp(mode, "inline")) {
      say("FAIL shadow", "unknown mode %s", mode);
      return M1_FAILED;
   }
   int result = shadow_setup(m, &shadow);
   if (result != M1_OK)
      return result;

   uint32_t *begin = ib_begin(m);
   uint32_t *p = shadow_begin(m, &shadow, separate);
   for (uint32_t b = 0; b < ARRAY_SIZE(banks); b++) {
      for (uint32_t r = 0; r < banks[b].count; r++) {
         const struct reg *reg = &banks[b].regs[r];
         p = emit_set(p, reg->opcode, reg->base, reg->address, &reg->first, 1);
      }
   }
   p = emit_readback(m, p, 0);
   result = shadow_run(m, &shadow, separate, begin, p, "shadowing on: set and read back in one submission");
   if (result != M1_OK)
      return result;
   report_readback(m, "same submission:", 0, false);

   const struct {
      const char *name;
      uint32_t offset;
      uint32_t expected;
   } mirrored[] = {
      {"CB_BLEND_RED", SI_SHADOWED_CONTEXT_REG_OFFSET + (R_028414_CB_BLEND_RED - SI_CONTEXT_REG_OFFSET), cx_regs[0].first},
      {"COMPUTE_USER_DATA_0", SI_SHADOWED_SH_REG_OFFSET + (R_00B900_COMPUTE_USER_DATA_0 - SI_SH_REG_OFFSET), sh_regs[0].first},
      {"VGT_TF_MEMORY_BASE", SI_SHADOWED_UCONFIG_REG_OFFSET + (R_030940_VGT_TF_MEMORY_BASE - CIK_UCONFIG_REG_OFFSET), uc_regs[1].first},
   };
   for (uint32_t i = 0; i < ARRAY_SIZE(mirrored); i++) {
      const volatile uint32_t *words = shadow.cpu;
      uint32_t got = words[mirrored[i].offset / 4];
      say(got == mirrored[i].expected ? "PASS" : "FAIL", "shadow memory holds %s: 0x%08x, expected 0x%08x",
          mirrored[i].name, got, mirrored[i].expected);
   }

   static const unsigned delays_ms[] = {0, 300, 1500};
   for (uint32_t i = 0; i < ARRAY_SIZE(delays_ms); i++) {
      char what[96];
      os_time_sleep((int64_t)delays_ms[i] * 1000);
      begin = ib_begin(m);
      p = shadow_begin(m, &shadow, separate);
      p = emit_readback(m, p, 16 * (i + 1));
      snprintf(what, sizeof(what), "shadowing on: read back after %u ms", delays_ms[i]);
      result = shadow_run(m, &shadow, separate, begin, p, what);
      if (result != M1_OK)
         return result;
      snprintf(what, sizeof(what), "later submission (+%u ms):", delays_ms[i]);
      report_readback(m, what, 16 * (i + 1), false);
   }

   double total = 0;
   const unsigned rounds = 200;
   for (unsigned i = 0; i < rounds; i++) {
      double elapsed = 0;
      begin = ib_begin(m);
      if ((uint32_t)(begin - m->code_cpu) + 2 * shadow.preamble_dwords + 64 > M1_BUFFER_BYTES / 4) {
         m->code_used = 0;
         begin = ib_begin(m);
      }
      p = shadow_begin(m, &shadow, separate);
      p = emit_nops(p, 8);
      uint32_t dwords = (uint32_t)(p - begin);
      m->code_used = (uint32_t)(begin - m->code_cpu) + dwords;
      struct agc_ib ibs[2];
      uint32_t count = 0;
      if (separate)
         ibs[count++] = (struct agc_ib){.va = shadow.preamble_va, .dwords = shadow.preamble_dwords, .flags = AGC_IB_PREAMBLE};
      ibs[count++] = (struct agc_ib){.va = m->code_va + (uint64_t)(begin - m->code_cpu) * 4, .dwords = dwords, .flags = AGC_IB_MAIN};
      result = submit_ibs(m, ibs, count, "empty submission with the shadowing preamble", &elapsed);
      if (result != M1_OK)
         return result;
      total += elapsed;
   }
   say("timing", "%u empty submissions with the shadowing preamble (%s): %.1f us each", rounds, mode,
       total / rounds * 1000.0);
   total = 0;
   for (unsigned i = 0; i < rounds; i++) {
      double elapsed = 0;
      begin = ib_begin(m);
      if ((uint32_t)(begin - m->code_cpu) + 64 > M1_BUFFER_BYTES / 4) {
         m->code_used = 0;
         begin = ib_begin(m);
      }
      p = emit_nops(begin, 8);
      m->code_used = (uint32_t)(p - m->code_cpu);
      struct agc_ib ib = {.va = m->code_va + (uint64_t)(begin - m->code_cpu) * 4, .dwords = 8, .flags = AGC_IB_MAIN};
      result = submit_ibs(m, &ib, 1, "empty submission", &elapsed);
      if (result != M1_OK)
         return result;
      total += elapsed;
   }
   say("timing", "%u empty submissions without it: %.1f us each", rounds, total / rounds * 1000.0);
   return M1_OK;
}

static int
cmd_bench(struct m1 *m, const char *mode)
{
   bool shadowed = !strcmp(mode, "shadow");
   struct shadow shadow = {0};
   if (!shadowed && strcmp(mode, "plain")) {
      say("FAIL bench", "unknown mode %s", mode);
      return M1_FAILED;
   }
   if (shadowed) {
      int result = shadow_setup(m, &shadow);
      if (result != M1_OK)
         return result;
   }
   const uint32_t writes = 240000;
   const uint64_t bytes = UINT64_C(4) << 20;
   uint32_t *cpu;
   uint64_t va;
   struct agc_bo *bo = big_buffer(m, "bench", bytes, &cpu, &va);
   if (!bo)
      return M1_FAILED;

   for (unsigned kind = 0; kind < 3; kind++) {
      static const char *const names[] = {"context", "shader", "uconfig"};
      const struct reg *reg = kind == 0 ? &cx_regs[0] : kind == 1 ? &sh_regs[0] : &uc_regs[1];
      uint32_t *p = cpu;
      if (shadowed) {
         memcpy(p, shadow.preamble, shadow.preamble_dwords * sizeof(uint32_t));
         p += shadow.preamble_dwords;
      }
      for (uint32_t i = 0; i < writes; i++) {
         uint32_t value = 0x00100000 + (i & 0xffff);
         p = emit_set(p, reg->opcode, reg->base, reg->address + 4 * (i & 3) * (kind != 2), &value, 1);
      }
      uint32_t dwords = align((uint32_t)(p - cpu), M1_IB_DWORD_MULTIPLE);
      emit_nops(p, dwords - (uint32_t)(p - cpu));
      double best = 1e9;
      for (unsigned round = 0; round < 5; round++) {
         double elapsed = 0;
         struct agc_ib ib = {.va = va, .dwords = dwords, .flags = AGC_IB_MAIN};
         int result = submit_ibs(m, &ib, 1, "register write benchmark", &elapsed);
         if (result != M1_OK)
            return result;
         if (elapsed < best)
            best = elapsed;
      }
      say("timing", "%u %s register writes, shadowing %s: best of 5 %.3f ms, %.1f ns per write", writes,
          names[kind], shadowed ? "on" : "off", best, best * 1e6 / writes);
   }
   return M1_OK;
}

static int
usage(void)
{
   fprintf(stderr,
           "usage: agc_m1_probe <persist|rmw|load BANK|chain CASE|shadow MODE|bench KIND> [--queue gfx|compute] "
           "[--timeout-ms N]\n"
           "  BANK: cx, sh, uc    CASE: aligned, odd-size, odd-start, odd-both, back, call\n"
           "  MODE: inline, preamble    KIND: plain, shadow\n");
   return M1_FAILED;
}

int
main(int argc, char **argv)
{
   struct m1 m = {.kind = AGC_QUEUE_GFX, .timeout_ns = INT64_C(5000) * 1000000};
   const char *command = argc > 1 ? argv[1] : NULL;
   const char *argument = NULL;
   if (!command)
      return usage();
   for (int i = 2; i < argc; i++) {
      if (!strcmp(argv[i], "--queue") && i + 1 < argc) {
         i++;
         if (!strcmp(argv[i], "compute"))
            m.kind = AGC_QUEUE_COMPUTE;
         else if (strcmp(argv[i], "gfx"))
            return usage();
      } else if (!strcmp(argv[i], "--timeout-ms") && i + 1 < argc) {
         m.timeout_ns = (int64_t)strtoul(argv[++i], NULL, 10) * 1000000;
      } else if (!argument) {
         argument = argv[i];
      } else {
         return usage();
      }
   }

   int result = m1_open(&m);
   if (result == M1_OK) {
      if (!strcmp(command, "persist"))
         result = cmd_persist(&m);
      else if (!strcmp(command, "rmw"))
         result = cmd_rmw(&m);
      else if (!strcmp(command, "load") && argument)
         result = cmd_load(&m, argument);
      else if (!strcmp(command, "chain") && argument)
         result = cmd_chain(&m, argument);
      else if (!strcmp(command, "shadow") && argument)
         result = cmd_shadow(&m, argument);
      else if (!strcmp(command, "bench") && argument)
         result = cmd_bench(&m, argument);
      else
         result = usage();
   }
   if (m.be && result != M1_UNCONFIRMED) {
      if (print_state(&m) != AGC_DEVICE_ACTIVE)
         result = M1_UNCONFIRMED;
   }
   say("result", "%s", result == M1_OK ? "completed" : result == M1_FAILED ? "failed" : "unconfirmed");
   return result;
}
