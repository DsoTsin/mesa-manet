#ifndef AGC_TEST_H
#define AGC_TEST_H

#include <stdbool.h>
#include <stdio.h>

static int agc_test_failures;

static inline void
agc_test_check(bool ok, const char *file, int line, const char *text)
{
   if (ok)
      return;
   fprintf(stderr, "%s:%d: %s\n", file, line, text);
   agc_test_failures++;
}

static inline void
agc_test_check_eq(unsigned long long a, unsigned long long b, const char *file, int line, const char *text)
{
   if (a == b)
      return;
   fprintf(stderr, "%s:%d: %s: %#llx != %#llx\n", file, line, text, a, b);
   agc_test_failures++;
}

#define CHECK(cond) agc_test_check((cond), __FILE__, __LINE__, #cond)
#define CHECK_EQ(a, b) agc_test_check_eq((unsigned long long)(a), (unsigned long long)(b), __FILE__, __LINE__, #a " == " #b)
#define TEST_RESULT() (agc_test_failures ? 1 : 0)

#endif
