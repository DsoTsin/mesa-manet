#include "agc_present.h"

#include <stdio.h>

#include "agc_driver.h"
#include "agc_memory.h"
#include "agc_videoout.h"

static bool
agc_present_driver_open(struct agc_present_device *device, char *error, size_t error_bytes)
{
   if (!agc_driver_open_device(error, error_bytes))
      return false;
   device->backend = agc_driver_backend();
   device->info = agc_driver_info();
   device->sync = agc_memory_sync;
   device->resident = agc_memory_imported;
   if (device->backend && device->info)
      return true;
   snprintf(error, error_bytes, "the driver started without a device");
   return false;
}

void
agc_present_install(void)
{
   agc_videoout_set_presenter(agc_present_d3d12(agc_present_driver_open));
}
