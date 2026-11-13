#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#include "util/format/u_format.h"
#include "util/macros.h"
#include "util/os_misc.h"
#include "util/os_time.h"
#include "util/u_math.h"

#include "ac_barrier.h"
#include "ac_cmdbuf.h"
#include "ac_debug.h"
#include "ac_descriptors.h"
#include "ac_drm_fourcc.h"
#include "ac_gpu_info.h"
#include "ac_surface.h"
#include "amd_family.h"
#include "sid.h"

#include "agc_driver_ops.h"
#include "agc_kernels.h"

#define PROBE_OK 0
#define PROBE_FAILED 1
#define PROBE_UNCONFIRMED 2

#define PROBE_BUFFER_BYTES UINT32_C(0x10000)
#define PROBE_BUFFER_DWORDS (PROBE_BUFFER_BYTES / 4u)
#define PROBE_GUARD 0xa5a5a5a5u
#define PROBE_IB_DWORDS 512u
#define PROBE_BARRIER_DWORDS 16u
#define PROBE_TIMEOUT_MS 5000u
#define PROBE_TIMEOUT_MAX_MS 600000u

#define PROBE_FILL_FIRST 24u
#define PROBE_FILL_COUNT 1000u
#define PROBE_FILL_VALUE 0x5eed1234u

#define PROBE_COPY_SRC_FIRST 8u
#define PROBE_COPY_DST_FIRST 40u
#define PROBE_COPY_COUNT 3001u
#define PROBE_COPY_PATTERN 0xc0de0000u

#define PROBE_IMAGE_WIDTH 100u
#define PROBE_IMAGE_HEIGHT 37u
#define PROBE_IMAGE_PITCH_BYTES 448u
#define PROBE_IMAGE_FIRST_BYTE (2u * PROBE_IMAGE_PITCH_BYTES + 16u)
#define PROBE_IMAGE_BACKGROUND 0x33
#define PROBE_DESCRIPTOR_OFFSET 0x40u
#define PROBE_PIXEL_REPORTS 4u

#define PROBE_TEST_FILL32 (1u << 0)
#define PROBE_TEST_COPY32 (1u << 1)
#define PROBE_TEST_DISPLAY (1u << 2)
#define PROBE_TEST_ALL (PROBE_TEST_FILL32 | PROBE_TEST_COPY32 | PROBE_TEST_DISPLAY)

static_assert(AGC_KERNEL_MAX_QUEUE_SETUP_DWORDS + AGC_KERNEL_MAX_DISPATCH_DWORDS +
                 2 * PROBE_BARRIER_DWORDS <
              PROBE_IB_DWORDS,
              "one probe IB holds the queue setup, both barriers and a dispatch");
static_assert(PROBE_FILL_FIRST + PROBE_FILL_COUNT < PROBE_BUFFER_DWORDS,
              "the fill fits the target buffer");
static_assert(PROBE_COPY_SRC_FIRST + PROBE_COPY_COUNT < PROBE_BUFFER_DWORDS,
              "the copy source fits the source buffer");
static_assert(PROBE_COPY_DST_FIRST + PROBE_COPY_COUNT < PROBE_BUFFER_DWORDS,
              "the copy fits the target buffer");
static_assert(PROBE_IMAGE_WIDTH * 4u <= PROBE_IMAGE_PITCH_BYTES, "a row fits the pitch");
static_assert(PROBE_IMAGE_FIRST_BYTE % 4u == 0 && PROBE_IMAGE_PITCH_BYTES % 4u == 0,
              "pixels land on dwords");
static_assert(PROBE_IMAGE_FIRST_BYTE + PROBE_IMAGE_HEIGHT * PROBE_IMAGE_PITCH_BYTES <
                 PROBE_BUFFER_BYTES,
              "the converted image fits the target buffer");

enum probe_kernel {
   PROBE_FILL32,
   PROBE_COPY32,
   PROBE_DISPLAY_CONVERT,
   PROBE_KERNEL_COUNT
};

struct probe_kernel_source {
   const char *name;
   struct nir_shader *(*build)(void);
   uint32_t user_data_dwords;
};

static const struct probe_kernel_source probe_kernels[PROBE_KERNEL_COUNT] = {
   [PROBE_FILL32] = {"fill32", agc_kernel_fill32, AGC_FILL32_USER_DATA_DWORDS},
   [PROBE_COPY32] = {"copy32", agc_kernel_copy32, AGC_COPY32_USER_DATA_DWORDS},
   [PROBE_DISPLAY_CONVERT] = {"display_convert", agc_kernel_display_convert,
                              AGC_DISPLAY_CONVERT_USER_DATA_DWORDS},
};

struct probe_buffer {
   struct agc_bo *bo;
   uint32_t *cpu;
   uint64_t va;
};

struct probe {
   struct agc_backend *be;
   const struct agc_driver_ops *ops;
   struct agc_hwq *queue;
   enum agc_queue_kind kind;
   int64_t timeout_ns;
   uint32_t tests;
   bool dump;
   bool compile_only;
   bool override_initiator;
   uint32_t initiator;
   struct radeon_info info;
   struct agc_kernel kernels[PROBE_KERNEL_COUNT];
   uint64_t kernel_va[PROBE_KERNEL_COUNT];
   struct probe_buffer code;
   struct probe_buffer ib;
   struct probe_buffer source;
   struct probe_buffer target;
   struct probe_buffer texture;
   struct probe_buffer descriptor;
   uint32_t ib_used_bytes;
   uint32_t *expected;
   uint8_t *payload;
};

static void probe_step(const char *name, const char *format, ...) AGC_PRINTF(2, 3);
static void probe_pass(const char *name, const char *format, ...) AGC_PRINTF(2, 3);
static int probe_fail(const char *name, const char *format, ...) AGC_PRINTF(2, 3);

static void
probe_line(const char *prefix, const char *name, const char *format, va_list args)
{
   printf("%s%s: ", prefix, name);
   vprintf(format, args);
   putchar('\n');
   fflush(stdout);
}

static void
probe_step(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   probe_line("", name, format, args);
   va_end(args);
}

static void
probe_pass(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   probe_line("PASS ", name, format, args);
   va_end(args);
}

static int
probe_fail(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   probe_line("FAIL ", name, format, args);
   va_end(args);
   return PROBE_FAILED;
}

static const char *
probe_queue_name(const struct probe *p)
{
   return p->kind == AGC_QUEUE_COMPUTE ? "compute" : "graphics";
}

static enum amd_ip_type
probe_ip(const struct probe *p)
{
   return p->kind == AGC_QUEUE_COMPUTE ? AMD_IP_COMPUTE : AMD_IP_GFX;
}

static enum agc_device_state
probe_print_state(struct probe *p)
{
   static const char *const names[] = {
      [AGC_DEVICE_ACTIVE] = "active",         [AGC_DEVICE_HUNG] = "hung",
      [AGC_DEVICE_PAGE_FAULT] = "page-fault", [AGC_DEVICE_REMOVED] = "removed",
      [AGC_DEVICE_UNKNOWN] = "unknown",
   };
   struct agc_wddm_device_state detail;
   struct agc_wddm_description description;
   char text[320];
   uint64_t fault_va = 0;
   enum agc_device_state state = p->ops->device_state(p->be, &fault_va);
   agc_wddm_device_state_detail(p->be, &detail);
   agc_wddm_device_state_format(&detail, text, sizeof(text));
   agc_wddm_describe(p->be, &description);
   probe_step("device state", "%s, fault va 0x%" PRIx64 ", transport %s; %s",
              (unsigned)state < ARRAY_SIZE(names) ? names[state] : "unknown", fault_va,
              description.lost ? "refuses further work" : "usable", text);
   return state;
}

static bool
probe_device_active(struct probe *p)
{
   uint64_t fault_va = 0;
   if (p->ops->device_state(p->be, &fault_va) == AGC_DEVICE_ACTIVE)
      return true;
   probe_print_state(p);
   return false;
}

static int
probe_open(struct probe *p)
{
   p->be = agc_wddm_open();
   if (!p->be)
      return probe_fail("open", "no AMD adapter with a verified profile could be opened");
   p->ops = p->be->ops;

   if (!p->ops->query_info(p->be, &p->info))
      return probe_fail("chip", "the chip information could not be filled");

   struct agc_wddm_description description;
   agc_wddm_describe(p->be, &description);
   const struct amd_ip_info *ip = &p->info.ip[probe_ip(p)];
   probe_step("chip", "\"%s\", family %s, gfx level %d, %u of %u shader engines, %u compute units, "
                      "source: %s",
              p->info.marketing_name, ac_get_family_name(p->info.family), (int)p->info.gfx_level,
              p->info.num_se, p->info.max_se, p->info.num_cu,
              description.chip_source == AGC_WDDM_CHIP_KMD_QUERY
                 ? "KMD query checked against the profile table"
                 : "profile table only");
   probe_step("queue kind", "%s, IB start alignment %u bytes, IB size multiple %u dwords",
              probe_queue_name(p), ip->ib_alignment, ip->ib_pad_dw_mask + 1);
   return PROBE_OK;
}

static int
probe_compile(struct probe *p)
{
   os_set_option("ACO_DEBUG", "validateir,validatera,validate-livevars,validateopt", false);

   for (unsigned i = 0; i < PROBE_KERNEL_COUNT; i++) {
      const struct probe_kernel_source *source = &probe_kernels[i];
      struct agc_kernel *kernel = &p->kernels[i];
      char log[256];

      if (!agc_kernel_compile(&p->info, source->build(), kernel, log, sizeof(log)))
         return probe_fail(source->name, "compilation failed: %s", log);
      if (kernel->user_data_dwords != source->user_data_dwords ||
          (kernel->code_dwords * 4u) % AGC_KERNEL_CODE_ALIGNMENT != 0 ||
          kernel->exec_dwords > kernel->code_dwords) {
         return probe_fail(source->name, "unexpected kernel: %u user data dwords, %u code dwords, "
                                         "%u instruction dwords",
                           kernel->user_data_dwords, kernel->code_dwords, kernel->exec_dwords);
      }
      probe_step(source->name, "%u instruction dwords in %u code dwords, block %ux%ux%u, wave%u, "
                               "%u user data dwords, rsrc1 0x%08x rsrc2 0x%08x rsrc3 0x%08x "
                               "limits 0x%08x",
                 kernel->exec_dwords, kernel->code_dwords, kernel->block[0], kernel->block[1],
                 kernel->block[2], kernel->wave_size, kernel->user_data_dwords, kernel->pgm_rsrc1,
                 kernel->pgm_rsrc2, kernel->pgm_rsrc3, kernel->resource_limits);
      if (p->dump) {
         if (kernel->listing)
            printf("%s\n", kernel->listing);
         printf("code %s:", source->name);
         for (uint32_t word = 0; word < kernel->exec_dwords; word++)
            printf(" %08x", kernel->code[word]);
         putchar('\n');
         fflush(stdout);
      }
   }
   probe_pass("compile", "%u kernels compiled for %s", (unsigned)PROBE_KERNEL_COUNT,
              ac_get_family_name(p->info.family));
   return PROBE_OK;
}

static bool
probe_buffer_create(struct probe *p, const char *name, uint32_t flags, struct probe_buffer *buffer)
{
   buffer->bo = p->ops->bo_create(p->be, PROBE_BUFFER_BYTES, flags, 0);
   if (!buffer->bo) {
      probe_fail(name, "allocation of 0x%x bytes failed", PROBE_BUFFER_BYTES);
      return false;
   }
   buffer->cpu = p->ops->bo_cpu(buffer->bo);
   buffer->va = p->ops->bo_va(buffer->bo);
   if (!buffer->cpu || !buffer->va || p->ops->bo_size(buffer->bo) < PROBE_BUFFER_BYTES) {
      probe_fail(name, "the buffer has no CPU mapping, no GPU address or is too small");
      return false;
   }
   probe_step(name, "GPU va 0x%" PRIx64 ", CPU %p, 0x%" PRIx64 " bytes%s%s", buffer->va,
              (void *)buffer->cpu, p->ops->bo_size(buffer->bo),
              flags & AGC_BO_EXECUTE ? ", executable" : "",
              flags & AGC_BO_GL2_BYPASS ? ", bypasses GL2" : "");
   return true;
}

static void
probe_buffer_destroy(struct probe *p, struct probe_buffer *buffer)
{
   if (buffer->bo)
      p->ops->bo_destroy(p->be, buffer->bo);
   memset(buffer, 0, sizeof(*buffer));
}

static int
probe_prepare(struct probe *p)
{
   p->queue = p->ops->queue_create(p->be, p->kind);
   if (!p->queue)
      return probe_fail("queue", "no %s hardware queue", probe_queue_name(p));

   struct agc_wddm_queue_description queue;
   agc_wddm_queue_describe(p->queue, &queue);
   probe_step("queue", "%s on node %u, id %u", probe_queue_name(p), queue.node, queue.queue_id);

   if (!probe_buffer_create(p, "code buffer", AGC_BO_EXECUTE, &p->code) ||
       !probe_buffer_create(p, "ib buffer", AGC_BO_GL2_BYPASS, &p->ib) ||
       !probe_buffer_create(p, "source buffer", 0, &p->source) ||
       !probe_buffer_create(p, "target buffer", AGC_BO_GL2_BYPASS, &p->target) ||
       !probe_buffer_create(p, "texture buffer", 0, &p->texture) ||
       !probe_buffer_create(p, "descriptor buffer", 0, &p->descriptor))
      return PROBE_FAILED;

   p->expected = calloc(PROBE_BUFFER_DWORDS, sizeof(uint32_t));
   p->payload = calloc(PROBE_BUFFER_DWORDS, sizeof(uint8_t));
   if (!p->expected || !p->payload)
      return probe_fail("memory", "out of memory");

   memset(p->code.cpu, 0, PROBE_BUFFER_BYTES);
   uint32_t offset = 0;
   for (unsigned i = 0; i < PROBE_KERNEL_COUNT; i++) {
      const struct agc_kernel *kernel = &p->kernels[i];
      const uint32_t bytes = kernel->code_dwords * 4u;
      offset = align(offset, AGC_KERNEL_CODE_ALIGNMENT);
      if (bytes > PROBE_BUFFER_BYTES - offset)
         return probe_fail("upload", "the kernels do not fit the code buffer");
      memcpy((uint8_t *)p->code.cpu + offset, kernel->code, bytes);
      p->kernel_va[i] = p->code.va + offset;
      probe_step("upload", "%s at 0x%" PRIx64 ", %u bytes", probe_kernels[i].name,
                 p->kernel_va[i], bytes);
      offset += bytes;
   }
   return PROBE_OK;
}

static void
probe_nops(uint32_t *words, uint32_t dwords)
{
   if (dwords == 1) {
      words[0] = PKT3_NOP_PAD;
   } else if (dwords > 1) {
      words[0] = PKT3(PKT3_NOP, dwords - 2, 0);
      memset(words + 1, 0, (dwords - 1) * sizeof(uint32_t));
   }
}

static int
probe_submit(struct probe *p, const uint32_t *words, uint32_t dwords, const char *what)
{
   const uint32_t alignment = p->info.ip[probe_ip(p)].ib_alignment;
   if (!util_is_power_of_two_nonzero(alignment) || alignment > PROBE_BUFFER_BYTES)
      return probe_fail(what, "IB start alignment %u is unusable", alignment);
   const uint32_t start = align(p->ib_used_bytes, alignment);
   if (start > PROBE_BUFFER_BYTES || dwords * 4u > PROBE_BUFFER_BYTES - start)
      return probe_fail(what, "the IB buffer is full");
   memcpy((uint8_t *)p->ib.cpu + start, words, dwords * sizeof(uint32_t));
   p->ib_used_bytes = start + dwords * 4u;

   const struct agc_ib ib = {
      .va = p->ib.va + start,
      .dwords = dwords,
      .flags = AGC_IB_MAIN,
   };
   probe_step(what, "submitting IB 0x%" PRIx64 " of %u dwords to the %s queue", ib.va, dwords,
              probe_queue_name(p));
   _mm_sfence();
   int64_t begin = os_time_get_nano();
   uint64_t fence = p->ops->submit(p->queue, &ib, 1);
   if (!fence) {
      struct agc_wddm_description description;
      agc_wddm_describe(p->be, &description);
      probe_fail(what, "the submission was refused");
      if (!description.lost)
         return PROBE_FAILED;
      probe_print_state(p);
      return PROBE_UNCONFIRMED;
   }
   if (!p->ops->fence_wait(p->queue, fence, p->timeout_ns)) {
      probe_fail(what, "fence %" PRIu64 " not reached within %" PRId64 " ms, completed %" PRIu64,
                 fence, p->timeout_ns / 1000000, p->ops->fence_completed(p->queue));
      probe_print_state(p);
      return PROBE_UNCONFIRMED;
   }
   probe_step(what, "fence %" PRIu64 " reached in %.3f ms", fence,
              (double)(os_time_get_nano() - begin) / 1e6);
   if (!probe_device_active(p)) {
      probe_fail(what, "the device left the active state");
      return PROBE_UNCONFIRMED;
   }
   return PROBE_OK;
}

static int
probe_run(struct probe *p, enum probe_kernel which, const uint32_t *user_data, uint32_t groups_x,
          uint32_t groups_y, uint32_t groups_z, const char *what)
{
   const struct agc_kernel *kernel = &p->kernels[which];
   const enum amd_ip_type ip = probe_ip(p);
   const bool compute_ring = p->kind == AGC_QUEUE_COMPUTE;
   const uint32_t multiple = p->info.ip[ip].ib_pad_dw_mask + 1;
   const uint32_t setup_dwords = agc_kernel_queue_setup_dwords(&p->info, compute_ring);
   const uint32_t dispatch_dwords = agc_kernel_dispatch_dwords(&p->info, kernel);
   uint32_t words[PROBE_IB_DWORDS];
   struct ac_cmdbuf cs = {
      .max_dw = PROBE_IB_DWORDS,
      .reserved_dw = PROBE_IB_DWORDS,
      .buf = words,
   };
   enum ac_rgp_flush_bits flush_bits = 0;
   bool context_roll = false;

   if (!setup_dwords || setup_dwords > AGC_KERNEL_MAX_QUEUE_SETUP_DWORDS || !dispatch_dwords ||
       dispatch_dwords > AGC_KERNEL_MAX_DISPATCH_DWORDS ||
       !util_is_power_of_two_nonzero(multiple) ||
       multiple > PROBE_IB_DWORDS - AGC_KERNEL_MAX_QUEUE_SETUP_DWORDS -
                     AGC_KERNEL_MAX_DISPATCH_DWORDS - 2 * PROBE_BARRIER_DWORDS) {
      return probe_fail(what, "unusable stream sizes: %u setup dwords, %u dispatch dwords, IB "
                              "multiple %u",
                        setup_dwords, dispatch_dwords, multiple);
   }

   uint32_t *end = agc_kernel_emit_queue_setup(words, &p->info, compute_ring);
   if (!end || (uint32_t)(end - words) != setup_dwords)
      return probe_fail(what, "the queue setup stream is not the %u dwords announced", setup_dwords);
   cs.cdw = setup_dwords;

   const struct ac_barrier_state invalidate = {
      .flags = AC_BARRIER_ALL_COMPUTE,
   };
   ac_emit_barrier(&cs, p->info.gfx_level, ip, &invalidate, &context_roll, &flush_bits);
   const uint32_t invalidate_dwords = cs.cdw - setup_dwords;
   if (!invalidate_dwords || invalidate_dwords > PROBE_BARRIER_DWORDS)
      return probe_fail(what, "the cache invalidation took %u dwords", invalidate_dwords);

   uint32_t *dispatch = words + cs.cdw;
   end = agc_kernel_emit_dispatch(dispatch, &p->info, kernel, p->kernel_va[which], user_data,
                                  groups_x, groups_y, groups_z);
   if (!end || (uint32_t)(end - dispatch) != dispatch_dwords)
      return probe_fail(what, "the dispatch stream is not the %u dwords announced", dispatch_dwords);
   if (p->override_initiator)
      end[-1] = p->initiator;
   const uint32_t initiator = end[-1];
   cs.cdw += dispatch_dwords;

   const struct ac_barrier_state flush = {
      .flags = AC_BARRIER_SYNC_CS | AC_BARRIER_INV_L2,
   };
   const uint32_t before_flush = cs.cdw;
   ac_emit_barrier(&cs, p->info.gfx_level, ip, &flush, &context_roll, &flush_bits);
   const uint32_t flush_dwords = cs.cdw - before_flush;
   if (!flush_dwords || flush_dwords > PROBE_BARRIER_DWORDS)
      return probe_fail(what, "the cache flush took %u dwords", flush_dwords);

   const uint32_t dwords = align(cs.cdw, multiple);
   probe_nops(words + cs.cdw, dwords - cs.cdw);

   probe_step(what, "kernel at 0x%" PRIx64 ", %ux%ux%u groups, initiator 0x%08x%s; queue setup %u "
                    "dwords, invalidation %u, dispatch %u, flush %u, padding %u",
              p->kernel_va[which], groups_x, groups_y, groups_z, initiator,
              p->override_initiator ? " (overridden)" : "", setup_dwords, invalidate_dwords,
              dispatch_dwords, flush_dwords, dwords - cs.cdw);
   if (p->dump) {
      struct ac_ib_parser parser = {
         .f = stdout,
         .ib = words,
         .num_dw = dwords,
         .gfx_level = p->info.gfx_level,
         .family = p->info.family,
         .ip_type = ip,
      };
      ac_parse_ib(&parser, what);
      fflush(stdout);
   }
   return probe_submit(p, words, dwords, what);
}

static void
probe_expect_reset(struct probe *p)
{
   for (uint32_t i = 0; i < PROBE_BUFFER_DWORDS; i++) {
      p->expected[i] = PROBE_GUARD;
      p->target.cpu[i] = PROBE_GUARD;
   }
   memset(p->payload, 0, PROBE_BUFFER_DWORDS);
}

static void
probe_expect(struct probe *p, uint32_t dword, uint32_t value)
{
   p->expected[dword] = value;
   p->payload[dword] = 1;
}

static int
probe_verify(struct probe *p, const char *what)
{
   const volatile uint32_t *got = p->target.cpu;
   uint32_t payload = 0, wrong = 0, untouched = 0, first_wrong = 0, wrong_value = 0;
   uint32_t guards = 0, changed = 0, first_changed = 0, changed_value = 0;

   for (uint32_t i = 0; i < PROBE_BUFFER_DWORDS; i++) {
      const uint32_t value = got[i];
      if (p->payload[i]) {
         payload++;
         if (value != p->expected[i]) {
            if (!wrong) {
               first_wrong = i;
               wrong_value = value;
            }
            wrong++;
            untouched += value == PROBE_GUARD;
         }
      } else {
         guards++;
         if (value != PROBE_GUARD) {
            if (!changed) {
               first_changed = i;
               changed_value = value;
            }
            changed++;
         }
      }
   }

   if (!wrong && !changed) {
      probe_pass(what, "%u dwords match, %u guard dwords intact", payload, guards);
      return PROBE_OK;
   }
   if (wrong) {
      probe_fail(what, "%u of %u dwords are wrong, %u of them still hold the guard; the first is "
                       "dword %u (+0x%x): 0x%08x, expected 0x%08x",
                 wrong, payload, untouched, first_wrong, first_wrong * 4u, wrong_value,
                 p->expected[first_wrong]);
   }
   if (changed) {
      probe_fail(what, "%u of %u guard dwords changed; the first is dword %u (+0x%x): 0x%08x",
                 changed, guards, first_changed, first_changed * 4u, changed_value);
   }
   return PROBE_FAILED;
}

static int
probe_test_fill32(struct probe *p)
{
   const struct agc_kernel *kernel = &p->kernels[PROBE_FILL32];
   uint32_t user_data[AGC_FILL32_USER_DATA_DWORDS];

   probe_expect_reset(p);
   for (uint32_t i = 0; i < PROBE_FILL_COUNT; i++)
      probe_expect(p, PROBE_FILL_FIRST + i, PROBE_FILL_VALUE);
   agc_kernel_fill32_user_data(user_data, p->target.va + PROBE_FILL_FIRST * 4u, PROBE_FILL_COUNT,
                               PROBE_FILL_VALUE);

   int result = probe_run(p, PROBE_FILL32, user_data,
                          agc_kernel_groups(PROBE_FILL_COUNT, kernel->block[0]), 1, 1, "fill32");
   return result == PROBE_OK ? probe_verify(p, "fill32") : result;
}

static int
probe_test_copy32(struct probe *p)
{
   const struct agc_kernel *kernel = &p->kernels[PROBE_COPY32];
   uint32_t user_data[AGC_COPY32_USER_DATA_DWORDS];

   for (uint32_t i = 0; i < PROBE_BUFFER_DWORDS; i++)
      p->source.cpu[i] = PROBE_COPY_PATTERN | i;
   probe_expect_reset(p);
   for (uint32_t i = 0; i < PROBE_COPY_COUNT; i++)
      probe_expect(p, PROBE_COPY_DST_FIRST + i, PROBE_COPY_PATTERN | (PROBE_COPY_SRC_FIRST + i));
   agc_kernel_copy32_user_data(user_data, p->target.va + PROBE_COPY_DST_FIRST * 4u,
                               p->source.va + PROBE_COPY_SRC_FIRST * 4u, PROBE_COPY_COUNT);

   int result = probe_run(p, PROBE_COPY32, user_data,
                          agc_kernel_groups(PROBE_COPY_COUNT, kernel->block[0]), 1, 1, "copy32");
   return result == PROBE_OK ? probe_verify(p, "copy32") : result;
}

static void
probe_image_texel(uint32_t x, uint32_t y, uint8_t rgba[4])
{
   rgba[0] = (uint8_t)x;
   rgba[1] = (uint8_t)y;
   rgba[2] = (uint8_t)(0x80u | ((x + y) & 0x7fu));
   rgba[3] = (uint8_t)(0xffu - x);
}

static void
probe_image_converted(uint32_t x, uint32_t y, const enum pipe_swizzle swizzle[4], uint8_t out[4])
{
   uint8_t rgba[4];
   probe_image_texel(x, y, rgba);
   for (unsigned i = 0; i < 4; i++)
      out[i] = rgba[swizzle[i] - PIPE_SWIZZLE_X];
}

static int
probe_image_layout(struct probe *p, struct radeon_surf *surf)
{
   const char *what = "display_convert";
   if (!p->info.has_image_opcodes)
      return probe_fail(what, "the chip description has no image opcodes");

   struct ac_addrlib *addrlib = ac_addrlib_create(&p->info, NULL);
   if (!addrlib)
      return probe_fail(what, "Mesa's address library rejected the chip description");

   const struct ac_surf_config config = {
      .info = {
         .width = PROBE_IMAGE_WIDTH,
         .height = PROBE_IMAGE_HEIGHT,
         .depth = 1,
         .samples = 1,
         .storage_samples = 1,
         .levels = 1,
         .num_channels = 4,
         .array_size = 1,
      },
      .blk_w = 1,
      .blk_h = 1,
      .bpe = 4,
      .surf_flags = RADEON_SURF_DISABLE_DCC | RADEON_SURF_NO_FMASK | RADEON_SURF_NO_HTILE |
                    RADEON_SURF_NO_RENDER_TARGET,
      .modifier = DRM_FORMAT_MOD_INVALID,
   };
   int status = ac_compute_surface(addrlib, &p->info, &config, RADEON_SURF_MODE_LINEAR_ALIGNED, surf);
   ac_addrlib_destroy(addrlib);
   if (status)
      return probe_fail(what, "ac_compute_surface failed with %d", status);

   const uint64_t pitch = surf->u.gfx9.surf_pitch;
   const uint64_t last_byte = surf->u.gfx9.surf_offset + (pitch * PROBE_IMAGE_HEIGHT) * 4u;
   probe_step("image", "%s %ux%u R8G8B8A8, pitch %" PRIu64 " texels, %" PRIu64 " bytes, "
                       "alignment 2^%u bytes, swizzle mode %u%s",
              surf->is_linear ? "linear" : "tiled", PROBE_IMAGE_WIDTH, PROBE_IMAGE_HEIGHT, pitch,
              surf->total_size, (unsigned)surf->alignment_log2,
              (unsigned)surf->u.gfx9.swizzle_mode,
              surf->u.gfx9.uses_custom_pitch ? ", custom pitch" : "");
   if (!surf->is_linear || pitch < PROBE_IMAGE_WIDTH || surf->surf_size > PROBE_BUFFER_BYTES ||
       surf->total_size > PROBE_BUFFER_BYTES || last_byte > surf->surf_size ||
       surf->alignment_log2 > 16 || p->texture.va % (UINT64_C(1) << surf->alignment_log2) != 0 ||
       surf->meta_offset || surf->tile_swizzle)
      return probe_fail(what, "the surface layout does not fit the texture buffer");

   memset(p->texture.cpu, PROBE_IMAGE_BACKGROUND, PROBE_BUFFER_BYTES);
   uint8_t *texels = (uint8_t *)p->texture.cpu + surf->u.gfx9.surf_offset;
   for (uint32_t y = 0; y < PROBE_IMAGE_HEIGHT; y++) {
      for (uint32_t x = 0; x < PROBE_IMAGE_WIDTH; x++)
         probe_image_texel(x, y, texels + (y * pitch + x) * 4u);
   }
   return PROBE_OK;
}

static int
probe_display_convert(struct probe *p, const char *what, const struct radeon_surf *surf,
                      const enum pipe_swizzle swizzle[4])
{
   const struct agc_kernel *kernel = &p->kernels[PROBE_DISPLAY_CONVERT];
   const uint64_t descriptor_va = p->descriptor.va + PROBE_DESCRIPTOR_OFFSET;
   uint32_t user_data[AGC_DISPLAY_CONVERT_USER_DATA_DWORDS];
   uint32_t descriptor[8];

   const struct ac_texture_state texture = {
      .surf = surf,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .img_format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width = PROBE_IMAGE_WIDTH,
      .height = PROBE_IMAGE_HEIGHT,
      .depth = 0,
      .type = V_008F1C_SQ_RSRC_IMG_2D,
      .swizzle = {swizzle[0], swizzle[1], swizzle[2], swizzle[3]},
      .num_samples = 1,
      .num_storage_samples = 1,
      .first_level = 0,
      .last_level = 0,
      .num_levels = 1,
      .first_layer = 0,
      .last_layer = 0,
   };
   const struct ac_mutable_tex_state address = {
      .surf = surf,
      .va = p->texture.va,
   };
   ac_build_texture_descriptor(&p->info, &texture, descriptor);
   ac_set_mutable_tex_desc_fields(&p->info, &address, descriptor);
   memcpy((uint8_t *)p->descriptor.cpu + PROBE_DESCRIPTOR_OFFSET, descriptor, sizeof(descriptor));
   probe_step(what, "descriptor at 0x%" PRIx64 ": %08x %08x %08x %08x %08x %08x %08x %08x",
              descriptor_va, descriptor[0], descriptor[1], descriptor[2], descriptor[3],
              descriptor[4], descriptor[5], descriptor[6], descriptor[7]);

   probe_expect_reset(p);
   for (uint32_t y = 0; y < PROBE_IMAGE_HEIGHT; y++) {
      for (uint32_t x = 0; x < PROBE_IMAGE_WIDTH; x++) {
         uint8_t bytes[4];
         probe_image_converted(x, y, swizzle, bytes);
         probe_expect(p, (PROBE_IMAGE_FIRST_BYTE + y * PROBE_IMAGE_PITCH_BYTES) / 4u + x,
                      bytes[0] | (uint32_t)bytes[1] << 8 | (uint32_t)bytes[2] << 16 |
                         (uint32_t)bytes[3] << 24);
      }
   }
   agc_kernel_display_convert_user_data(user_data, descriptor_va,
                                        p->target.va + PROBE_IMAGE_FIRST_BYTE, PROBE_IMAGE_WIDTH,
                                        PROBE_IMAGE_HEIGHT, PROBE_IMAGE_PITCH_BYTES);

   int result = probe_run(p, PROBE_DISPLAY_CONVERT, user_data,
                          agc_kernel_groups(PROBE_IMAGE_WIDTH, kernel->block[0]),
                          agc_kernel_groups(PROBE_IMAGE_HEIGHT, kernel->block[1]), 1, what);
   if (result != PROBE_OK)
      return result;
   result = probe_verify(p, what);
   if (result == PROBE_OK)
      return result;

   const volatile uint32_t *got = p->target.cpu;
   uint32_t reported = 0;
   for (uint32_t y = 0; y < PROBE_IMAGE_HEIGHT && reported < PROBE_PIXEL_REPORTS; y++) {
      for (uint32_t x = 0; x < PROBE_IMAGE_WIDTH && reported < PROBE_PIXEL_REPORTS; x++) {
         const uint32_t dword = (PROBE_IMAGE_FIRST_BYTE + y * PROBE_IMAGE_PITCH_BYTES) / 4u + x;
         const uint32_t value = got[dword];
         if (value == p->expected[dword])
            continue;
         probe_step(what, "pixel (%u, %u) is %02x %02x %02x %02x, expected %02x %02x %02x %02x", x,
                    y, value & 0xff, (value >> 8) & 0xff, (value >> 16) & 0xff, value >> 24,
                    p->expected[dword] & 0xff, (p->expected[dword] >> 8) & 0xff,
                    (p->expected[dword] >> 16) & 0xff, p->expected[dword] >> 24);
         reported++;
      }
   }
   return result;
}

static int
probe_test_display(struct probe *p)
{
   static const enum pipe_swizzle rgba[4] = {
      PIPE_SWIZZLE_X,
      PIPE_SWIZZLE_Y,
      PIPE_SWIZZLE_Z,
      PIPE_SWIZZLE_W,
   };
   static const enum pipe_swizzle bgra[4] = {
      PIPE_SWIZZLE_Z,
      PIPE_SWIZZLE_Y,
      PIPE_SWIZZLE_X,
      PIPE_SWIZZLE_W,
   };
   struct radeon_surf surf;

   int result = probe_image_layout(p, &surf);
   if (result != PROBE_OK)
      return result;
   result = probe_display_convert(p, "display_convert rgba", &surf, rgba);
   if (result != PROBE_OK)
      return result;
   return probe_display_convert(p, "display_convert bgra", &surf, bgra);
}

static int
probe_tests(struct probe *p)
{
   int result = PROBE_OK;

   if (p->tests & PROBE_TEST_FILL32) {
      int test = probe_test_fill32(p);
      if (test == PROBE_UNCONFIRMED)
         return test;
      result = MAX2(result, test);
   }
   if (p->tests & PROBE_TEST_COPY32) {
      int test = probe_test_copy32(p);
      if (test == PROBE_UNCONFIRMED)
         return test;
      result = MAX2(result, test);
   }
   if (p->tests & PROBE_TEST_DISPLAY) {
      if (result != PROBE_OK) {
         probe_step("display_convert", "skipped: fill32 or copy32 failed");
         return result;
      }
      result = probe_test_display(p);
   }
   return result;
}

static int
probe_close(struct probe *p, int result)
{
   if (result == PROBE_UNCONFIRMED) {
      probe_step("teardown", "skipped: GPU completion is unconfirmed, the process exit releases "
                             "everything");
      return result;
   }
   if (!p->be)
      return result;
   if (probe_print_state(p) != AGC_DEVICE_ACTIVE) {
      probe_step("teardown", "skipped: the device is not active");
      return PROBE_UNCONFIRMED;
   }

   probe_buffer_destroy(p, &p->descriptor);
   probe_buffer_destroy(p, &p->texture);
   probe_buffer_destroy(p, &p->target);
   probe_buffer_destroy(p, &p->source);
   probe_buffer_destroy(p, &p->ib);
   probe_buffer_destroy(p, &p->code);
   if (p->queue)
      p->ops->queue_destroy(p->queue);
   p->queue = NULL;
   for (unsigned i = 0; i < PROBE_KERNEL_COUNT; i++)
      agc_kernel_free(&p->kernels[i]);
   free(p->expected);
   free(p->payload);

   struct agc_wddm_description description;
   agc_wddm_describe(p->be, &description);
   p->ops->destroy(p->be);
   p->be = NULL;
   if (description.lost)
      return probe_fail("close", "the device was lost");
   if (description.live_allocations != 0)
      return probe_fail("close", "%u allocations were still live", description.live_allocations);
   probe_step("close", "buffers, queue, device and adapter released");
   return result;
}

static int
probe_usage(void)
{
   fprintf(stderr,
           "usage: agc_kernel_probe [--queue compute|gfx] [--only fill32|copy32|display] "
           "[--timeout-ms N] [--initiator HEX] [--compile-only] [--dump]\n"
           "  Compiles the driver kernels for the GPU the transport opens and runs them on it.\n"
           "  --compile-only  open the GPU and compile, submit nothing\n"
           "  --only NAME     run one test instead of fill32, copy32, then display_convert\n"
           "  --initiator HEX replace the dispatch initiator of every dispatch\n"
           "  --dump          print the kernel listings and every command stream\n"
           "  exit code 0: all passed, 1: a check failed, 2: GPU completion unconfirmed\n");
   return PROBE_FAILED;
}

int
main(int argc, char **argv)
{
   static struct probe p = {
      .kind = AGC_QUEUE_COMPUTE,
      .timeout_ns = INT64_C(1000000) * PROBE_TIMEOUT_MS,
      .tests = PROBE_TEST_ALL,
   };

   for (int i = 1; i < argc; i++) {
      const char *value = i + 1 < argc ? argv[i + 1] : NULL;
      if (!strcmp(argv[i], "--queue") && value) {
         if (!strcmp(value, "compute"))
            p.kind = AGC_QUEUE_COMPUTE;
         else if (!strcmp(value, "gfx"))
            p.kind = AGC_QUEUE_GFX;
         else
            return probe_usage();
         i++;
      } else if (!strcmp(argv[i], "--only") && value) {
         if (!strcmp(value, "fill32"))
            p.tests = PROBE_TEST_FILL32;
         else if (!strcmp(value, "copy32"))
            p.tests = PROBE_TEST_COPY32;
         else if (!strcmp(value, "display"))
            p.tests = PROBE_TEST_DISPLAY;
         else
            return probe_usage();
         i++;
      } else if (!strcmp(argv[i], "--timeout-ms") && value) {
         char *end = NULL;
         unsigned long long milliseconds = strtoull(value, &end, 10);
         if (!end || *end || !milliseconds || milliseconds > PROBE_TIMEOUT_MAX_MS)
            return probe_usage();
         p.timeout_ns = INT64_C(1000000) * (int64_t)milliseconds;
         i++;
      } else if (!strcmp(argv[i], "--initiator") && value) {
         char *end = NULL;
         unsigned long long initiator = strtoull(value, &end, 16);
         if (!end || *end || end == value || initiator > UINT32_MAX)
            return probe_usage();
         p.override_initiator = true;
         p.initiator = (uint32_t)initiator;
         i++;
      } else if (!strcmp(argv[i], "--compile-only")) {
         p.compile_only = true;
      } else if (!strcmp(argv[i], "--dump")) {
         p.dump = true;
      } else {
         return probe_usage();
      }
   }

   int result = probe_open(&p);
   if (result == PROBE_OK)
      result = probe_compile(&p);
   if (result == PROBE_OK && !p.compile_only) {
      result = probe_prepare(&p);
      if (result == PROBE_OK)
         result = probe_tests(&p);
   }
   result = probe_close(&p, result);

   probe_step("result", "%s",
              result == PROBE_OK ? "PASS" : result == PROBE_FAILED ? "FAIL" : "UNCONFIRMED");
   return result;
}
