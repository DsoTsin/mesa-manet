#include "agc_anyps5.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "c11/threads.h"
#include "util/u_atomic.h"

#include "agc_debug.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <dlfcn.h>
#endif

#define NID_NAME_MAX 256

static uint32_t
rotate_left(uint32_t value, unsigned bits)
{
   return value << bits | value >> (32 - bits);
}

static void
sha1_block(uint32_t state[5], const uint8_t *block)
{
   uint32_t w[80];
   for (unsigned i = 0; i < 16; i++) {
      w[i] = (uint32_t)block[4 * i] << 24 | (uint32_t)block[4 * i + 1] << 16 | (uint32_t)block[4 * i + 2] << 8 |
             block[4 * i + 3];
   }
   for (unsigned i = 16; i < 80; i++)
      w[i] = rotate_left(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
   uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4];
   for (unsigned i = 0; i < 80; i++) {
      uint32_t f, k;
      if (i < 20) {
         f = (b & c) | (~b & d);
         k = 0x5a827999;
      } else if (i < 40) {
         f = b ^ c ^ d;
         k = 0x6ed9eba1;
      } else if (i < 60) {
         f = (b & c) | (b & d) | (c & d);
         k = 0x8f1bbcdc;
      } else {
         f = b ^ c ^ d;
         k = 0xca62c1d6;
      }
      uint32_t next = rotate_left(a, 5) + f + e + k + w[i];
      e = d;
      d = c;
      c = rotate_left(b, 30);
      b = a;
      a = next;
   }
   state[0] += a;
   state[1] += b;
   state[2] += c;
   state[3] += d;
   state[4] += e;
}

void
agc_nid(const char *name, char out[AGC_NID_BYTES])
{
   static const uint8_t key[16] = {
      0x51, 0x8d, 0x64, 0xa6, 0x35, 0xde, 0xd8, 0xc1, 0xe6, 0xb0, 0x39, 0xb1, 0xc3, 0xe5, 0x52, 0x30,
   };
   static const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-";
   uint8_t message[NID_NAME_MAX + sizeof(key) + 128];
   size_t length = strnlen(name, NID_NAME_MAX);
   memset(message, 0, sizeof(message));
   memcpy(message, name, length);
   memcpy(message + length, key, sizeof(key));
   length += sizeof(key);
   message[length] = 0x80;
   size_t padded = (length + 1 + 8 + 63) & ~(size_t)63;
   uint64_t bits = (uint64_t)length * 8;
   for (unsigned i = 0; i < 8; i++)
      message[padded - 1 - i] = (uint8_t)(bits >> (8 * i));

   uint32_t state[5] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};
   for (size_t at = 0; at < padded; at += 64)
      sha1_block(state, message + at);

   uint64_t value = 0;
   for (unsigned i = 0; i < 8; i++) {
      uint8_t byte = (uint8_t)(state[i / 4] >> (24 - 8 * (i % 4)));
      value |= (uint64_t)byte << (8 * i);
   }
   for (unsigned i = 0; i < 10; i++)
      out[i] = alphabet[(value >> (58 - 6 * i)) & 0x3f];
   out[10] = alphabet[(value << 2) & 0x3f];
   out[11] = 0;
}

static void
export_name(const char *name, char *out, size_t bytes)
{
   static const char postfix[] = "_nid_postfix";
   size_t length = strlen(name);
   size_t suffix = sizeof(postfix) - 1;
   if (length > suffix && strcmp(name + length - suffix, postfix) == 0 && length - suffix < NID_NAME_MAX) {
      char stripped[NID_NAME_MAX];
      memcpy(stripped, name, length - suffix);
      stripped[length - suffix] = 0;
      char nid[AGC_NID_BYTES];
      agc_nid(stripped, nid);
      snprintf(out, bytes, "%s", nid);
      return;
   }
   snprintf(out, bytes, "%s", name);
}

void *
agc_anyps5_find_export(const char *name)
{
   char export[256];
   export_name(name, export, sizeof(export));
#ifdef _WIN32
   HMODULE fixed[1024];
   HMODULE *modules = fixed;
   DWORD capacity = sizeof(fixed);
   DWORD needed = 0;
   void *found = NULL;
   for (;;) {
      if (!K32EnumProcessModules(GetCurrentProcess(), modules, capacity, &needed))
         break;
      if (needed <= capacity) {
         DWORD count = needed / sizeof(HMODULE);
         for (DWORD i = 0; i < count && !found; i++)
            found = (void *)GetProcAddress(modules[i], export);
         break;
      }
      if (modules != fixed)
         free(modules);
      capacity = needed + 64 * sizeof(HMODULE);
      modules = malloc(capacity);
      if (!modules)
         return NULL;
   }
   if (modules != fixed)
      free(modules);
   return found;
#else
   return dlsym(RTLD_DEFAULT, export);
#endif
}

struct import {
   const char *name;
   size_t offset;
   bool optional;
};

#define IMPORT(member, text) {text, offsetof(struct agc_anyps5, member), false}
#define IMPORT_OPTIONAL(member, text) {text, offsetof(struct agc_anyps5, member), true}

static const struct import imports[] = {
   IMPORT(map_import_alias, "GuestArenaMapImportAlias_nid_no_patch"),
   IMPORT(unmap_import_alias, "GuestArenaUnmapImportAlias_nid_no_patch"),
   IMPORT(import_alias_pages, "GuestArenaImportAliasPages_nid_no_patch"),
   IMPORT(map_backing_alias, "GuestArenaMapAliasForImport_nid_no_patch"),
   IMPORT(unmap_backing_alias, "GuestArenaUnmapAlias_nid_postfix"),
   IMPORT(gpu_visible, "GuestAllocationsGpuVisible_nid_no_patch"),
   IMPORT(forget_import, "GuestAllocationsForgetImport_nid_no_patch"),
   IMPORT(generation, "GuestAllocationsGeneration_nid_postfix"),
   IMPORT_OPTIONAL(gpu_generation, "GuestAllocationsGpuGeneration_nid_no_patch"),
   IMPORT(arena_range, "GuestArenaRange_nid_postfix"),
   IMPORT(write_watched, "GuestArenaWriteWatched_nid_postfix"),
   IMPORT(collect_writes, "GuestArenaCollectWrites_nid_postfix"),
   IMPORT(set_import_release, "GuestAllocationsSetImportRelease_nid_no_patch"),
   IMPORT(sparse_ranges, "GuestAllocationsSparseRanges_nid_no_patch"),
   IMPORT(add_data_event, "EqueueAddDataEvent_nid_postfix"),
   IMPORT(trigger_event, "EqueueTriggerEvent_nid_postfix"),
   IMPORT(delete_event, "EqueueDeleteEvent_nid_postfix"),
   IMPORT(note_gpu_submit, "KernelNoteGpuSubmit_nid_no_patch"),
};

bool
agc_anyps5_resolve(struct agc_anyps5 *host, char *missing, size_t missing_bytes)
{
   size_t used = 0;
   bool complete = true;
   memset(host, 0, sizeof(*host));
   if (missing && missing_bytes)
      missing[0] = 0;
   for (size_t i = 0; i < sizeof(imports) / sizeof(imports[0]); i++) {
      void *function = agc_anyps5_find_export(imports[i].name);
      memcpy((char *)host + imports[i].offset, &function, sizeof(function));
      if (function || imports[i].optional)
         continue;
      complete = false;
      if (missing && used < missing_bytes) {
         int wrote = snprintf(missing + used, missing_bytes - used, "%s%s", used ? ", " : "", imports[i].name);
         if (wrote > 0)
            used += (size_t)wrote;
      }
   }
   if (!host->gpu_generation)
      host->gpu_generation = host->generation;
   return complete;
}

static struct agc_anyps5 resolved_host;
static const struct agc_anyps5 *current_host;
static once_flag resolve_once = ONCE_FLAG_INIT;

static void
resolve_host(void)
{
   char missing[1024];
   if (p_atomic_read(&current_host))
      return;
   if (!agc_anyps5_resolve(&resolved_host, missing, sizeof(missing))) {
      agc_report("agc_anyps5", "AnyPS5 exports none of %s", missing);
      return;
   }
   p_atomic_set(&current_host, &resolved_host);
}

const struct agc_anyps5 *
agc_anyps5_get(void)
{
   const struct agc_anyps5 *host = p_atomic_read(&current_host);
   if (AGC_LIKELY(host))
      return host;
   call_once(&resolve_once, resolve_host);
   return p_atomic_read(&current_host);
}

void
agc_anyps5_set(const struct agc_anyps5 *host)
{
   p_atomic_set(&current_host, host);
}
