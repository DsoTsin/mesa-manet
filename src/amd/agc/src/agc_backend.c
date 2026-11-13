#include "agc_driver.h"

#include "agc_driver_ops.h"

struct agc_backend *
agc_backend_open(void)
{
#ifdef _WIN32
   return agc_wddm_open();
#else
   return NULL;
#endif
}
