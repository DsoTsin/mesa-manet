#ifndef AGC_API_TRACE_H
#define AGC_API_TRACE_H

#include <stdbool.h>
#include <stdint.h>

#include "agc_types.h"

struct agc_trace_function {
   const char *name;
   uint8_t params;
   bool command_buffer;
};

extern const void *const agc_trace_real[];
extern const struct agc_trace_function agc_trace_functions[];
extern const uint32_t agc_trace_function_count;
extern volatile uint8_t agc_trace_on;

void agc_trace_enter(void);
void AGC_SYSV agc_trace_log(uint32_t id, const uint64_t *saved, const uint64_t *stack);

#define AGC_TRACE_THUNK(name, id)                                                                \
   __attribute__((naked)) void agc_trace_thunk_##name(void)                                      \
   {                                                                                             \
      __asm__("movl $" #id ", %r11d\n\t"                                                         \
              "jmp agc_trace_enter");                                                            \
   }

#endif
