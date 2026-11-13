#ifndef AGC_TYPES_H
#define AGC_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(__clang__) || defined(__GNUC__)
#define AGC_SYSV __attribute__((sysv_abi))
#else
#error "the guest calling convention needs a compiler with sysv_abi"
#endif
#else
#define AGC_SYSV
#endif

#if defined(__clang__) || defined(__GNUC__)
#define AGC_LIKELY(x) __builtin_expect(!!(x), 1)
#define AGC_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define AGC_NORETURN __attribute__((noreturn))
#define AGC_PRINTF(text_index, first_index) __attribute__((format(printf, text_index, first_index)))
#else
#define AGC_LIKELY(x) (x)
#define AGC_UNLIKELY(x) (x)
#define AGC_NORETURN
#define AGC_PRINTF(text_index, first_index)
#endif

struct agc_register {
   uint32_t offset;
   uint32_t value;
};

enum agc_host {
   AGC_HOST_GFX103,
   AGC_HOST_GFX11,
   AGC_HOST_GFX115,
   AGC_HOST_GFX12,
   AGC_HOST_COUNT
};

#endif
