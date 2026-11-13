#include "agc_debug.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "util/u_atomic.h"

#define AGC_REPORT_LIMIT 64

static agc_fatal_handler fatal_handler;
static uint32_t report_count;

void
agc_set_fatal_handler(agc_fatal_handler handler)
{
   fatal_handler = handler;
}

void
agc_fatal(const char *function, const char *format, ...)
{
   char message[512];
   int used = snprintf(message, sizeof(message), "%s: ", function);
   if (used < 0 || (size_t)used >= sizeof(message))
      used = 0;
   va_list args;
   va_start(args, format);
   vsnprintf(message + used, sizeof(message) - (size_t)used, format, args);
   va_end(args);
   if (fatal_handler)
      fatal_handler(message);
   fprintf(stderr, "[agc] fatal: %s\n", message);
   abort();
}

void
agc_report(const char *function, const char *format, ...)
{
   if ((uint32_t)p_atomic_inc_return(&report_count) > AGC_REPORT_LIMIT)
      return;
   va_list args;
   va_start(args, format);
   fprintf(stderr, "[agc] %s: ", function);
   vfprintf(stderr, format, args);
   fputc('\n', stderr);
   va_end(args);
}

void
agc_stat(const char *function, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   fprintf(stderr, "[agc] %s: ", function);
   vfprintf(stderr, format, args);
   fputc('\n', stderr);
   va_end(args);
}

uint32_t
agc_report_count(void)
{
   return (uint32_t)p_atomic_read(&report_count);
}
