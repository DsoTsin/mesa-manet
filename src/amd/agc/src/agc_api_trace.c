#include "agc_api_trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agc_cb.h"
#include "agc_debug.h"
#include "c11/threads.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/u_atomic.h"

#define TRACE_VERSION 1u
#define TRACE_ARGS_MAX 16u
#define TRACE_BUFFER_BYTES (UINT32_C(4) << 20)
#define TRACE_FLUSH_NS (UINT64_C(200) * 1000 * 1000)
#define TRACE_LIMIT_MB_DEFAULT UINT64_C(2048)
#define TRACE_COMMAND_BUFFER 1u

volatile uint8_t agc_trace_on = 1;

static struct {
   mtx_t lock;
   FILE *file;
   uint8_t *buffer;
   size_t used;
   uint64_t written;
   uint64_t limit;
   uint64_t start;
   uint64_t flushed_at;
} trace;

static once_flag trace_once = ONCE_FLAG_INIT;
static uint32_t trace_threads;
static _Thread_local uint32_t trace_thread;

static void
trace_flush_locked(void)
{
   if (!trace.file)
      return;
   if (trace.used && fwrite(trace.buffer, 1, trace.used, trace.file) != trace.used)
      agc_trace_on = 0;
   fflush(trace.file);
   trace.written += trace.used;
   trace.used = 0;
   trace.flushed_at = os_time_get_nano();
   if (trace.written >= trace.limit) {
      agc_trace_on = 0;
      agc_stat("agc_api_trace", "the API trace reached its limit of %llu bytes and stops",
               (unsigned long long)trace.limit);
   }
}

static void
trace_exit(void)
{
   mtx_lock(&trace.lock);
   trace_flush_locked();
   fclose(trace.file);
   trace.file = NULL;
   agc_trace_on = 0;
   mtx_unlock(&trace.lock);
}

static void
trace_put(const void *data, size_t bytes)
{
   if (trace.file && fwrite(data, 1, bytes, trace.file) != bytes)
      agc_trace_on = 0;
}

static void
trace_init(void)
{
   const char *path = getenv("AGC_TRACE_API");
   if (!path || !*path) {
      agc_trace_on = 0;
      return;
   }
   mtx_init(&trace.lock, mtx_plain);
   trace.buffer = malloc(TRACE_BUFFER_BYTES);
   trace.file = fopen(path, "wb");
   if (!trace.buffer || !trace.file) {
      agc_report("agc_api_trace", "cannot write the API trace to %s", path);
      agc_trace_on = 0;
      return;
   }
   const char *limit = getenv("AGC_TRACE_API_LIMIT_MB");
   trace.limit = (limit && *limit ? strtoull(limit, NULL, 10) : TRACE_LIMIT_MB_DEFAULT) << 20;
   trace.start = os_time_get_nano();
   trace.flushed_at = trace.start;
   const uint32_t head[2] = {TRACE_VERSION, agc_trace_function_count};
   trace_put("AGCTRACE", 8);
   trace_put(head, sizeof(head));
   for (uint32_t i = 0; i < agc_trace_function_count; i++) {
      const struct agc_trace_function *function = &agc_trace_functions[i];
      const uint8_t entry[3] = {(uint8_t)strlen(function->name), function->params, function->command_buffer};
      trace_put(entry, 1);
      trace_put(function->name, entry[0]);
      trace_put(entry + 1, 2);
   }
   fflush(trace.file);
   atexit(trace_exit);
   agc_stat("agc_api_trace", "tracing %u AGC exports to %s", agc_trace_function_count, path);
}

void AGC_SYSV
agc_trace_log(uint32_t id, const uint64_t *saved, const uint64_t *stack)
{
   static const uint8_t order[6] = {7, 6, 5, 4, 3, 2};
   call_once(&trace_once, trace_init);
   if (!agc_trace_on || id >= agc_trace_function_count)
      return;
   if (!trace_thread)
      trace_thread = p_atomic_inc_return(&trace_threads);
   const struct agc_trace_function *function = &agc_trace_functions[id];
   uint32_t count = MIN2(function->params, TRACE_ARGS_MAX);
   uint64_t record[2 + TRACE_ARGS_MAX + 4];
   uint32_t words = 2;
   for (uint32_t i = 0; i < count; i++)
      record[words++] = i < 6 ? saved[order[i]] : stack[i - 6];
   uint64_t flags = 0;
   if (function->command_buffer && count && record[2]) {
      const struct agc_cb *cb = (const struct agc_cb *)(uintptr_t)record[2];
      record[words++] = (uintptr_t)cb->bottom;
      record[words++] = (uintptr_t)cb->top;
      record[words++] = (uintptr_t)cb->up;
      record[words++] = (uintptr_t)cb->down;
      flags |= TRACE_COMMAND_BUFFER;
   }
   record[0] = os_time_get_nano() - trace.start;
   record[1] = trace_thread | (uint64_t)id << 32 | (uint64_t)count << 48 | flags << 56;
   size_t bytes = words * sizeof(uint64_t);
   mtx_lock(&trace.lock);
   if (agc_trace_on && trace.file) {
      if (trace.used + bytes > TRACE_BUFFER_BYTES)
         trace_flush_locked();
      memcpy(trace.buffer + trace.used, record, bytes);
      trace.used += bytes;
      if (os_time_get_nano() - trace.flushed_at > TRACE_FLUSH_NS)
         trace_flush_locked();
   }
   mtx_unlock(&trace.lock);
}

__attribute__((naked)) void
agc_trace_enter(void)
{
   __asm__("cmpb $0, agc_trace_on(%rip)\n\t"
           "jne 1f\n\t"
           "leaq agc_trace_real(%rip), %r10\n\t"
           "jmpq *(%r10,%r11,8)\n"
           "1:\n\t"
           "pushq %rdi\n\t"
           "pushq %rsi\n\t"
           "pushq %rdx\n\t"
           "pushq %rcx\n\t"
           "pushq %r8\n\t"
           "pushq %r9\n\t"
           "pushq %r11\n\t"
           "pushq %rax\n\t"
           "subq $136, %rsp\n\t"
           "movdqu %xmm0, 0(%rsp)\n\t"
           "movdqu %xmm1, 16(%rsp)\n\t"
           "movdqu %xmm2, 32(%rsp)\n\t"
           "movdqu %xmm3, 48(%rsp)\n\t"
           "movdqu %xmm4, 64(%rsp)\n\t"
           "movdqu %xmm5, 80(%rsp)\n\t"
           "movdqu %xmm6, 96(%rsp)\n\t"
           "movdqu %xmm7, 112(%rsp)\n\t"
           "movl %r11d, %edi\n\t"
           "leaq 136(%rsp), %rsi\n\t"
           "leaq 208(%rsp), %rdx\n\t"
           "callq agc_trace_log\n\t"
           "movdqu 0(%rsp), %xmm0\n\t"
           "movdqu 16(%rsp), %xmm1\n\t"
           "movdqu 32(%rsp), %xmm2\n\t"
           "movdqu 48(%rsp), %xmm3\n\t"
           "movdqu 64(%rsp), %xmm4\n\t"
           "movdqu 80(%rsp), %xmm5\n\t"
           "movdqu 96(%rsp), %xmm6\n\t"
           "movdqu 112(%rsp), %xmm7\n\t"
           "addq $136, %rsp\n\t"
           "popq %rax\n\t"
           "popq %r11\n\t"
           "popq %r9\n\t"
           "popq %r8\n\t"
           "popq %rcx\n\t"
           "popq %rdx\n\t"
           "popq %rsi\n\t"
           "popq %rdi\n\t"
           "leaq agc_trace_real(%rip), %r10\n\t"
           "jmpq *(%r10,%r11,8)");
}
