#include <string.h>

#include "agc_anyps5.h"
#include "test.h"

static uint64_t
generation(void)
{
   return 7;
}

int
main(void)
{
   char nid[AGC_NID_BYTES];
   agc_nid("sceAgcInit_0090", nid);
   CHECK(strcmp(nid, "23LRUSvYu1M") == 0);
   agc_nid("", nid);
   CHECK_EQ(strlen(nid), 11);

   CHECK(agc_anyps5_find_export("GuestAllocationsGeneration_nid_postfix") == NULL);
   CHECK(agc_anyps5_find_export("AgcTestNoSuchExport_nid_no_patch") == NULL);

   struct agc_anyps5 host;
   char missing[2048];
   CHECK(!agc_anyps5_resolve(&host, missing, sizeof(missing)));
   CHECK(strstr(missing, "GuestArenaMapImportAlias_nid_no_patch") != NULL);
   CHECK(strstr(missing, "KernelNoteGpuSubmit_nid_no_patch") != NULL);
   CHECK(strstr(missing, "GuestAllocationsGpuGeneration") == NULL);
   CHECK(host.map_import_alias == NULL && host.gpu_generation == NULL);

   char small[24];
   CHECK(!agc_anyps5_resolve(&host, small, sizeof(small)));
   CHECK(strlen(small) < sizeof(small));

   memset(&host, 0, sizeof(host));
   host.generation = generation;
   agc_anyps5_set(&host);
   CHECK(agc_anyps5_get() == &host);
   CHECK_EQ(agc_anyps5_get()->generation(), 7);
   agc_anyps5_set(NULL);
   CHECK(agc_anyps5_get() == NULL);
   return TEST_RESULT();
}
