#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <intrin.h>

extern "C" {

extern DWORD _tls_index;
extern const IMAGE_TLS_DIRECTORY64 _tls_used;
__declspec(thread) static volatile unsigned char reserved[0x200] = {};

static HMODULE compiler_image;
static PIMAGE_TLS_CALLBACK *compiler_callbacks;
static bool enabled;

static void NTAPI dispatch(void *, DWORD reason, void *reserved)
{
   if (enabled && compiler_callbacks)
      for (auto callback = compiler_callbacks; *callback; ++callback)
         (*callback)(compiler_image, reason, reserved);
}

__declspec(dllexport) void *aoc_tls_current()
{
   (void)reserved[0];
   auto blocks = reinterpret_cast<void**>(__readgsqword(0x58));
   return blocks[_tls_index];
}

__declspec(dllexport) DWORD aoc_tls_get_index()
{
   return _tls_index;
}

__declspec(dllexport) BOOL aoc_tls_bind(HMODULE image, PIMAGE_TLS_CALLBACK *callbacks,
                                      const void *initial, SIZE_T size)
{
   if (size != 0x188 || !image || !callbacks ||
       size > _tls_used.EndAddressOfRawData - _tls_used.StartAddressOfRawData) return FALSE;
   auto source = static_cast<const volatile unsigned char*>(initial);
   auto current = static_cast<volatile unsigned char*>(aoc_tls_current());
   auto next = reinterpret_cast<volatile unsigned char*>(_tls_used.StartAddressOfRawData);
   DWORD protection;
   if (!VirtualProtect(const_cast<unsigned char*>(next), size, PAGE_READWRITE, &protection)) return FALSE;
   for (SIZE_T i = 0; i < size; ++i) current[i] = next[i] = source[i];
   DWORD previous;
   if (!VirtualProtect(const_cast<unsigned char*>(next), size, protection, &previous)) return FALSE;
   compiler_image = image;
   compiler_callbacks = callbacks;
   return TRUE;
}

__declspec(dllexport) void aoc_tls_notify(DWORD reason)
{
   if (reason == DLL_THREAD_ATTACH) enabled = true;
   dispatch(compiler_image, reason, nullptr);
}

__declspec(dllexport) void aoc_tls_unbind()
{
   enabled = false;
   compiler_callbacks = nullptr;
   compiler_image = nullptr;
}

#pragma section(".CRT$XLY", read)
__declspec(allocate(".CRT$XLY")) extern const PIMAGE_TLS_CALLBACK aoc_tls_callback = dispatch;

}
