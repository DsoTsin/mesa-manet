#include "agc_runtime.h"

#include <stdio.h>
#include <string.h>
#include <xmmintrin.h>

#include "agc_isa_descriptors.h"
#include "agc_memory.h"

#define MIP_STATS_SLOTS 256u
#define MIP_STATS_NO_MIP UINT32_C(0xf)
#define SLOT_BYTES UINT64_C(0x10000)

static struct agc_mem tables[4];

void
agc_runtime_mip_stats_empty(uint32_t words[AGC_RUNTIME_MIP_STATS_BYTES / 4])
{
   for (uint32_t slot = 0; slot < MIP_STATS_SLOTS; slot++) {
      words[2 * slot] = slot << 24;
      words[2 * slot + 1] = MIP_STATS_NO_MIP << 24;
   }
}

static bool
place(unsigned index, uint64_t va, uint64_t bytes, char *error, size_t error_bytes, const char *what)
{
   uint64_t size = (bytes + SLOT_BYTES - 1) & ~(SLOT_BYTES - 1);
   if (!agc_mem_alloc_at(va, size, 0, &tables[index])) {
      snprintf(error, error_bytes, "the %s cannot be placed at 0x%llx", what, (unsigned long long)va);
      return false;
   }
   memset(tables[index].cpu, 0, size);
   return true;
}

_Static_assert(AGC_IMAGE_REPORT_TABLE_BYTES == AGC_RUNTIME_IMAGE_REPORTS_BYTES, "image report table size");

uint32_t
agc_image_report_fold(const uint32_t tsharp[8])
{
   return tsharp[1] ^ tsharp[2] ^ tsharp[3] ^ tsharp[4] ^ tsharp[5];
}

uint32_t
agc_image_report_hash(const uint32_t tsharp[8], bool stored)
{
   const uint32_t hash =
      ((tsharp[0] * AGC_IMAGE_REPORT_SPREAD) ^ agc_image_report_fold(tsharp)) * AGC_IMAGE_REPORT_SPREAD;
   return stored ? hash ^ AGC_IMAGE_REPORT_STORE_SALT : hash;
}

uint32_t
agc_image_report_known_offset(uint32_t hash)
{
   return (hash >> 19) * AGC_IMAGE_KNOWN_BUCKET_BYTES;
}

static volatile uint32_t *
image_word(uint32_t offset)
{
   return (volatile uint32_t *)((uint8_t *)tables[3].cpu + offset);
}

uint32_t
agc_runtime_image_reports_take(uint32_t (*tsharps)[8], bool *stored, uint32_t capacity)
{
   if (!tables[3].cpu)
      return 0;
   uint32_t count = 0;
   for (uint32_t slot = 0; slot < AGC_IMAGE_REPORT_SLOTS; slot++) {
      const uint32_t at = AGC_IMAGE_REPORTS_OFFSET + slot * AGC_IMAGE_REPORT_BYTES;
      const uint32_t check = *image_word(at + AGC_IMAGE_REPORT_CHECK_OFFSET);
      if (!check)
         continue;
      uint32_t tsharp[8];
      for (unsigned k = 0; k < 8; k++)
         tsharp[k] = *image_word(at + k * 4);
      *image_word(at + AGC_IMAGE_REPORT_CHECK_OFFSET) = 0;
      const uint32_t fold = agc_image_report_fold(tsharp);
      const bool store = (fold ^ AGC_IMAGE_REPORT_STORE_CHECK) == check;
      if ((!store && (fold ^ AGC_IMAGE_REPORT_CHECK) != check) || count == capacity)
         continue;
      stored[count] = store;
      memcpy(tsharps[count++], tsharp, sizeof(tsharp));
   }
   _mm_sfence();
   return count;
}

void
agc_runtime_image_mark(const uint32_t tsharp[8], bool stored)
{
   static bool evicted;
   if (!tables[3].cpu)
      return;
   const uint32_t hash = agc_image_report_hash(tsharp, stored);
   const uint32_t bucket = agc_image_report_known_offset(hash);
   if (*image_word(bucket) == hash || *image_word(bucket + 4) == hash)
      return;
   uint32_t way;
   if (!*image_word(bucket)) {
      way = 0;
   } else if (!*image_word(bucket + 4)) {
      way = 4;
   } else {
      evicted = !evicted;
      way = evicted ? 4 : 0;
   }
   *image_word(bucket + way) = hash;
   _mm_sfence();
}

void
agc_runtime_image_unmark(const uint32_t tsharp[8], bool stored)
{
   if (!tables[3].cpu)
      return;
   const uint32_t hash = agc_image_report_hash(tsharp, stored);
   const uint32_t bucket = agc_image_report_known_offset(hash);
   for (uint32_t way = 0; way < 8; way += 4) {
      if (hash && *image_word(bucket + way) == hash)
         *image_word(bucket + way) = 0;
   }
   _mm_sfence();
}

bool
agc_runtime_init(char *error, size_t error_bytes)
{
   if (!place(0, AGC_RUNTIME_DESCRIPTOR_LUT_VA, AGC_RUNTIME_DESCRIPTOR_LUT_BYTES, error, error_bytes,
              "descriptor conversion table") ||
       !place(1, AGC_RUNTIME_MIP_STATS_VA, 2 * AGC_RUNTIME_MIP_STATS_BYTES, error, error_bytes,
              "mip statistics table") ||
       !place(2, AGC_RUNTIME_GDS_VA, AGC_RUNTIME_GDS_BUFFER_BYTES, error, error_bytes, "GDS buffer") ||
       !place(3, AGC_RUNTIME_IMAGE_REPORTS_VA, AGC_RUNTIME_IMAGE_REPORTS_BYTES, error, error_bytes,
              "image report table"))
      return false;
   agc_isa_descriptor_lut(tables[0].cpu);
   agc_runtime_mip_stats_empty(tables[1].cpu);
   agc_runtime_mip_stats_empty((uint32_t *)((uint8_t *)tables[1].cpu + AGC_RUNTIME_MIP_STATS_BYTES));
   _mm_sfence();
   return true;
}
