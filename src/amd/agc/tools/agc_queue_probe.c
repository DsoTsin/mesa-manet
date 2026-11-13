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
#include "sid.h"

#include "agc_anyps5.h"
#include "agc_cb_builders.h"
#include "agc_driver_ops.h"
#include "agc_memory.h"
#include "agc_queue_state.h"
#include "agc_registers.h"
#include "agc_runtime.h"

#define PROBE_OK 0
#define PROBE_FAILED 1
#define PROBE_UNCONFIRMED 2

#define IB_BYTES UINT64_C(0x10000)
#define IB_SLOT_DWORDS 1024u
#define TIMEOUT_NS INT64_C(5000000000)

struct probe {
   struct agc_backend *be;
   const struct agc_driver_ops *ops;
   struct radeon_info info;
   struct agc_hwq *hwq;
   struct agc_queue_state state;
   struct agc_mem ib;
   struct agc_mem data;
   uint32_t next_slot;
   int failures;
};

static agc_guest_import_release release_callback;

static void *
host_map_import_alias(uintptr_t address, size_t bytes, uint32_t *refused)
{
   (void)address;
   (void)bytes;
   *refused = AGC_GUEST_IMPORT_NOT_GPU_VISIBLE;
   return NULL;
}

static void
host_unmap_import_alias(void *alias)
{
   (void)alias;
}

static size_t
host_gpu_visible(uintptr_t address, size_t bytes, struct agc_guest_range *pieces, size_t capacity)
{
   (void)address;
   (void)bytes;
   (void)pieces;
   (void)capacity;
   return 0;
}

static void
host_forget_import(uintptr_t address, size_t bytes)
{
   (void)address;
   (void)bytes;
}

static uint64_t
host_generation(void)
{
   return 1;
}

static void
host_set_import_release(agc_guest_import_release release, void *user)
{
   (void)user;
   release_callback = release;
}

static const struct agc_anyps5 host_table = {
   .map_import_alias = host_map_import_alias,
   .unmap_import_alias = host_unmap_import_alias,
   .gpu_visible = host_gpu_visible,
   .forget_import = host_forget_import,
   .generation = host_generation,
   .gpu_generation = host_generation,
   .set_import_release = host_set_import_release,
};

static void
quiesce(void *user)
{
   (void)user;
}

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

static uint64_t
slot_va(const struct probe *p, uint32_t slot)
{
   return p->data.va + (uint64_t)slot * 4;
}

static uint32_t
slot_value(const struct probe *p, uint32_t slot)
{
   const volatile uint32_t *words = p->data.cpu;
   return words[slot];
}

static uint32_t *
emit_set(uint32_t *out, uint32_t reg, uint32_t value)
{
   if (reg >= CIK_UCONFIG_REG_OFFSET) {
      *out++ = PKT3(PKT3_SET_UCONFIG_REG, 1, 0);
      *out++ = (reg - CIK_UCONFIG_REG_OFFSET) >> 2;
   } else if (reg >= SI_CONTEXT_REG_OFFSET) {
      *out++ = PKT3(PKT3_SET_CONTEXT_REG, 1, 0);
      *out++ = (reg - SI_CONTEXT_REG_OFFSET) >> 2;
   } else {
      *out++ = PKT3(PKT3_SET_SH_REG, 1, 0);
      *out++ = (reg - SI_SH_REG_OFFSET) >> 2;
   }
   *out++ = value;
   return out;
}

static uint32_t *
emit_read(struct probe *p, uint32_t *out, uint32_t reg, uint32_t slot)
{
   uint64_t dst = slot_va(p, slot);
   *out++ = PKT3(PKT3_COPY_DATA, 4, 0);
   *out++ = COPY_DATA_SRC_SEL(COPY_DATA_REG) | COPY_DATA_DST_SEL(COPY_DATA_DST_MEM) | COPY_DATA_WR_CONFIRM;
   *out++ = reg >> 2;
   *out++ = 0;
   *out++ = (uint32_t)dst;
   *out++ = (uint32_t)(dst >> 32);
   return out;
}

static uint32_t *
emit_load_pairs(struct probe *p, uint32_t *out, unsigned opcode, uint32_t base, uint32_t reg, uint32_t value,
                uint32_t slot)
{
   volatile uint32_t *words = p->data.cpu;
   words[slot] = (reg - base) >> 2;
   words[slot + 1] = value;
   uint64_t va = slot_va(p, slot);
   *out++ = PKT3(opcode, 3, 0);
   *out++ = (uint32_t)va & ~UINT32_C(3);
   *out++ = (uint32_t)(va >> 32);
   *out++ = S_9F3_DATA_FORMAT(V_9F3_OFFSET_AND_DATA);
   *out++ = 1;
   return out;
}

static uint32_t *
emit_call(uint32_t *out, uint64_t va, uint32_t dwords)
{
   *out++ = PKT3(PKT3_INDIRECT_BUFFER, 2, 0);
   *out++ = (uint32_t)va;
   *out++ = (uint32_t)(va >> 32);
   *out++ = S_3F3_IB_SIZE(dwords) | S_3F3_VALID(1);
   return out;
}

static enum agc_device_state
print_state(struct probe *p)
{
   struct agc_wddm_device_state detail;
   uint64_t fault_va = 0;
   enum agc_device_state state = p->ops->device_state(p->be, &fault_va);
   agc_wddm_device_state_detail(p->be, &detail);
   say("device state", "%d, fault va 0x%" PRIx64 ", execution %d, page fault status %d", (int)state, fault_va,
       detail.execution, detail.page_fault_status);
   return state;
}

static int
submit(struct probe *p, uint32_t *begin, uint32_t *end, const char *what, double *milliseconds)
{
   uint32_t dwords = (uint32_t)(end - begin);
   uint32_t padded = align(dwords, 8);
   agc_fill_nops(end, padded - dwords);
   struct agc_ib ibs[2] = {
      {.va = p->state.preamble_va, .dwords = p->state.preamble_dwords, .flags = AGC_IB_MAIN},
      {.va = p->ib.va + (uint64_t)((char *)begin - (char *)p->ib.cpu), .dwords = padded, .flags = AGC_IB_MAIN},
   };
   _mm_sfence();
   int64_t start = os_time_get_nano();
   uint64_t fence = p->ops->submit(p->hwq, ibs, 2);
   if (!fence) {
      say("FAIL submit", "%s: refused", what);
      print_state(p);
      return PROBE_UNCONFIRMED;
   }
   if (!p->ops->fence_wait(p->hwq, fence, TIMEOUT_NS)) {
      say("FAIL wait", "%s: fence %" PRIu64 " not reached", what, fence);
      print_state(p);
      return PROBE_UNCONFIRMED;
   }
   double elapsed = (double)(os_time_get_nano() - start) / 1e6;
   if (milliseconds)
      *milliseconds = elapsed;
   say("ran", "%s: %u dwords after a %u-dword preamble, %.3f ms", what, padded, p->state.preamble_dwords, elapsed);
   return PROBE_OK;
}

static uint32_t *
next_ib(struct probe *p)
{
   uint32_t *out = (uint32_t *)p->ib.cpu + (size_t)p->next_slot * IB_SLOT_DWORDS;
   p->next_slot++;
   return out;
}

static void
expect(struct probe *p, const char *what, uint32_t slot, uint32_t value)
{
   uint32_t got = slot_value(p, slot);
   if (got != value)
      p->failures++;
   say(got == value ? "PASS" : "FAIL", "%s: 0x%08x, expected 0x%08x", what, got, value);
}

static int
run_tests(struct probe *p)
{
   const uint32_t red = R_028414_CB_BLEND_RED;
   const uint32_t user0 = R_00B900_COMPUTE_USER_DATA_0;
   const uint32_t ge_cntl = R_03096C_GE_CNTL;
   const uint32_t ring = R_031118_SPI_ATTRIBUTE_RING_BASE;
   int result;
   uint32_t *b, *o;

   b = next_ib(p);
   o = emit_set(b, red, 0x11);
   o = emit_set(o, user0, 0x22);
   o = emit_read(p, o, red, 0);
   o = emit_read(p, o, user0, 1);
   o = emit_read(p, o, ring, 2);
   o = emit_read(p, o, ge_cntl, 3);
   if ((result = submit(p, b, o, "set and read", NULL)))
      return result;
   expect(p, "same submission: CB_BLEND_RED", 0, 0x11);
   expect(p, "same submission: COMPUTE_USER_DATA_0", 1, 0x22);
   expect(p, "default: SPI_ATTRIBUTE_RING_BASE", 2, (uint32_t)(AGC_RUNTIME_ATTRIBUTE_RING_VA >> 16));
   expect(p, "default: GE_CNTL", 3, 0x1f810080);

   b = next_ib(p);
   o = emit_read(p, b, red, 4);
   o = emit_read(p, o, user0, 5);
   if ((result = submit(p, b, o, "read in a later submission", NULL)))
      return result;
   expect(p, "later submission: CB_BLEND_RED", 4, 0x11);
   expect(p, "later submission: COMPUTE_USER_DATA_0", 5, 0x22);

   b = next_ib(p);
   o = emit_call(b, p->state.context_va[1], p->state.context_dwords[1]);
   o = emit_set(o, red, 0x33);
   o = emit_read(p, o, red, 6);
   o = emit_call(o, p->state.context_va[2], p->state.context_dwords[2]);
   o = emit_read(p, o, red, 7);
   if ((result = submit(p, b, o, "push, change, pop", NULL)))
      return result;
   expect(p, "after push and a change: CB_BLEND_RED", 6, 0x33);
   expect(p, "after pop: CB_BLEND_RED", 7, 0x11);

   b = next_ib(p);
   o = emit_call(b, p->state.context_va[0], p->state.context_dwords[0]);
   o = emit_read(p, o, red, 8);
   o = emit_read(p, o, user0, 9);
   if ((result = submit(p, b, o, "clear context", NULL)))
      return result;
   expect(p, "after clear: CB_BLEND_RED", 8, 0);
   expect(p, "after clear: COMPUTE_USER_DATA_0 (not context)", 9, 0x22);

   b = next_ib(p);
   o = emit_set(b, red, 0x55);
   o = emit_call(o, p->state.context_va[3], p->state.context_dwords[3]);
   o = emit_read(p, o, red, 10);
   o = emit_call(o, p->state.context_va[2], p->state.context_dwords[2]);
   o = emit_read(p, o, red, 11);
   if ((result = submit(p, b, o, "push and clear, pop", NULL)))
      return result;
   expect(p, "after push and clear: CB_BLEND_RED", 10, 0);
   expect(p, "after pop: CB_BLEND_RED", 11, 0x55);

   double reset_ms = 0;
   b = next_ib(p);
   o = emit_set(b, red, 0x66);
   o = emit_set(o, user0, 0x77);
   o = emit_call(o, p->state.reset_va, p->state.reset_dwords);
   o = emit_read(p, o, red, 12);
   o = emit_read(p, o, user0, 13);
   o = emit_read(p, o, ge_cntl, 14);
   o = emit_read(p, o, ring, 15);
   if ((result = submit(p, b, o, "reset", &reset_ms)))
      return result;
   expect(p, "after reset: CB_BLEND_RED", 12, 0);
   expect(p, "after reset: COMPUTE_USER_DATA_0", 13, 0);
   expect(p, "after reset: GE_CNTL", 14, 0x1f810080);
   expect(p, "after reset: SPI_ATTRIBUTE_RING_BASE", 15, (uint32_t)(AGC_RUNTIME_ATTRIBUTE_RING_VA >> 16));

   b = next_ib(p);
   o = emit_read(p, b, red, 16);
   o = emit_read(p, o, user0, 17);
   if ((result = submit(p, b, o, "read after reset in a later submission", NULL)))
      return result;
   expect(p, "later submission: CB_BLEND_RED", 16, 0);
   expect(p, "later submission: COMPUTE_USER_DATA_0", 17, 0);

   const uint32_t user1 = R_00B904_COMPUTE_USER_DATA_1;
   const uint32_t prim = R_03096C_GE_CNTL;
   b = next_ib(p);
   o = emit_load_pairs(p, b, PKT3_LOAD_CONTEXT_REG_INDEX, SI_CONTEXT_REG_OFFSET, red, 0x88, 64);
   o = emit_load_pairs(p, o, PKT3_LOAD_SH_REG_INDEX, SI_SH_REG_OFFSET, user1, 0x99, 66);
   o = emit_load_pairs(p, o, PKT3_LOAD_UCONFIG_REG_INDEX, CIK_UCONFIG_REG_OFFSET, prim, 0x1f8100c0, 68);
   o = emit_read(p, o, red, 20);
   o = emit_read(p, o, user1, 21);
   o = emit_read(p, o, prim, 22);
   _mm_sfence();
   if ((result = submit(p, b, o, "indexed loads", NULL)))
      return result;
   expect(p, "indexed load, same submission: CB_BLEND_RED", 20, 0x88);
   expect(p, "indexed load, same submission: COMPUTE_USER_DATA_1", 21, 0x99);
   expect(p, "indexed load, same submission: GE_CNTL", 22, 0x1f8100c0);

   b = next_ib(p);
   o = emit_read(p, b, red, 23);
   o = emit_read(p, o, user1, 24);
   o = emit_read(p, o, prim, 25);
   if ((result = submit(p, b, o, "read indexed loads in a later submission", NULL)))
      return result;
   expect(p, "indexed load, later submission: CB_BLEND_RED", 23, 0x88);
   expect(p, "indexed load, later submission: COMPUTE_USER_DATA_1", 24, 0x99);
   expect(p, "indexed load, later submission: GE_CNTL", 25, 0x1f8100c0);

   double total = 0;
   const unsigned rounds = 50;
   for (unsigned i = 0; i < rounds; i++) {
      double elapsed = 0;
      p->next_slot = 32;
      b = next_ib(p);
      o = emit_call(b, p->state.reset_va, p->state.reset_dwords);
      if ((result = submit(p, b, o, "timed reset", &elapsed)))
         return result;
      total += elapsed;
   }
   say("timing", "submission with one reset: %.1f us on average over %u", total / rounds * 1000.0, rounds);
   return p->failures ? PROBE_FAILED : PROBE_OK;
}

static bool
select_host(const struct radeon_info *info)
{
   switch (info->gfx_level) {
   case GFX10_3:
      return agc_host_select(AGC_HOST_GFX103);
   case GFX11:
      return agc_host_select(AGC_HOST_GFX11);
   case GFX11_5:
      return agc_host_select(AGC_HOST_GFX115);
   case GFX12:
      return agc_host_select(AGC_HOST_GFX12);
   default:
      return false;
   }
}

int
main(void)
{
   struct probe p;
   char error[256];
   memset(&p, 0, sizeof(p));
   p.be = agc_wddm_open();
   if (!p.be) {
      say("FAIL open", "no AMD adapter with a verified profile could be opened");
      return PROBE_FAILED;
   }
   p.ops = p.be->ops;
   if (!p.ops->query_info(p.be, &p.info) || !select_host(&p.info)) {
      say("FAIL info", "the GPU did not describe itself or is not supported");
      return PROBE_FAILED;
   }
   p.hwq = p.ops->queue_create(p.be, AGC_QUEUE_GFX);
   if (!p.hwq) {
      say("FAIL queue", "no graphics queue");
      return PROBE_FAILED;
   }
   struct agc_memory_config config = {.backend = p.be, .anyps5 = &host_table, .quiesce = quiesce};
   if (!agc_memory_init(&config, error, sizeof(error)) || !agc_runtime_init(error, sizeof(error))) {
      say("FAIL memory", "%s", error);
      return PROBE_FAILED;
   }
   say("PASS memory", "windows reserved and runtime tables placed");
   if (!agc_mem_alloc(IB_BYTES, 0, &p.ib) || !agc_mem_alloc(0x10000, AGC_BO_GL2_BYPASS, &p.data)) {
      say("FAIL memory", "driver memory for the probe");
      return PROBE_FAILED;
   }
   memset(p.data.cpu, 0xa5, 0x10000);
   int64_t start = os_time_get_nano();
   if (!agc_queue_state_init(p.be, &p.info, p.hwq, NULL, &p.state, error, sizeof(error))) {
      say("FAIL queue state", "%s", error);
      print_state(&p);
      return PROBE_UNCONFIRMED;
   }
   say("PASS queue state", "initialised in %.3f ms: preamble %u dwords, reset %u dwords, context ops %u/%u/%u/%u dwords",
       (double)(os_time_get_nano() - start) / 1e6, p.state.preamble_dwords, p.state.reset_dwords,
       p.state.context_dwords[0], p.state.context_dwords[1], p.state.context_dwords[2], p.state.context_dwords[3]);
   int result = run_tests(&p);
   if (print_state(&p) != AGC_DEVICE_ACTIVE)
      result = PROBE_UNCONFIRMED;
   say("result", "%s, %d failed checks", result == PROBE_OK ? "PASS" : result == PROBE_FAILED ? "FAIL" : "UNCONFIRMED",
       p.failures);
   return result;
}
