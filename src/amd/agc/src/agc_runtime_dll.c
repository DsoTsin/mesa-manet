#include "agc_driver.h"
#include "agc_present.h"

#ifdef _WIN32
#include <windows.h>

BOOL WINAPI
DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
   (void)reserved;
   if (reason == DLL_PROCESS_ATTACH) {
      DisableThreadLibraryCalls(instance);
      agc_present_install();
   }
   return TRUE;
}
#endif
