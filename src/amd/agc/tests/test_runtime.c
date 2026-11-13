#include <stdlib.h>
#include <string.h>

#include "util/mesa-blake3.h"

#include "agc_isa_descriptors.h"
#include "agc_runtime.h"
#include "test.h"

static void
hex(const uint8_t *data, size_t bytes, char out[2 * BLAKE3_OUT_LEN + 1])
{
   blake3_hash hash;
   _mesa_blake3_compute(data, bytes, hash);
   for (unsigned i = 0; i < BLAKE3_OUT_LEN; i++)
      snprintf(out + 2 * i, 3, "%02x", hash[i]);
}

int
main(void)
{
   char digest[2 * BLAKE3_OUT_LEN + 1];
   uint8_t *lut = calloc(1, AGC_ISA_DESCRIPTOR_LUT_BYTES);
   CHECK(lut != NULL);
   CHECK_EQ(AGC_ISA_DESCRIPTOR_LUT_BYTES, AGC_RUNTIME_DESCRIPTOR_LUT_BYTES);
   agc_isa_descriptor_lut(lut);
   hex(lut, AGC_ISA_DESCRIPTOR_LUT_BYTES, digest);
   CHECK(!strcmp(digest, "b4b03aa39d60741bdb413225310842ca045fd1dc324bae725c546bbf56d54f0f"));
   free(lut);

   uint32_t words[AGC_RUNTIME_MIP_STATS_BYTES / 4];
   agc_runtime_mip_stats_empty(words);
   hex((const uint8_t *)words, sizeof(words), digest);
   CHECK(!strcmp(digest, "32dbba34b64d7661f7265b5533905e8072c883b2a99095d1e5435a04b6acc52f"));

   CHECK(AGC_RUNTIME_GDS_VA + AGC_RUNTIME_GDS_BUFFER_BYTES <= AGC_RUNTIME_IMAGE_REPORTS_VA);
   CHECK(AGC_RUNTIME_IMAGE_REPORTS_VA + AGC_RUNTIME_IMAGE_REPORTS_BYTES <= AGC_RUNTIME_ATTRIBUTE_RING_VA);
   CHECK(AGC_RUNTIME_TABLES_END <= AGC_RUNTIME_BASE + AGC_RUNTIME_BYTES);
   CHECK(AGC_RUNTIME_ATTRIBUTE_RING_VA % 0x10000 == 0);
   return TEST_RESULT();
}
