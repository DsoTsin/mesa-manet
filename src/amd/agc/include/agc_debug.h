#ifndef AGC_DEBUG_H
#define AGC_DEBUG_H

#include "agc_types.h"

typedef void (*agc_fatal_handler)(const char *message);

void agc_set_fatal_handler(agc_fatal_handler handler);
AGC_NORETURN void agc_fatal(const char *function, const char *format, ...) AGC_PRINTF(2, 3);
void agc_report(const char *function, const char *format, ...) AGC_PRINTF(2, 3);
void agc_stat(const char *function, const char *format, ...) AGC_PRINTF(2, 3);
uint32_t agc_report_count(void);

#endif
