#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <windows.h>

#include "util/macros.h"
#include "util/os_time.h"

#include "sid.h"

#include "agc_anyps5.h"
#include "agc_cb.h"
#include "agc_driver.h"
#include "agc_driver_ops.h"
#include "agc_exports.h"
#include "agc_memory.h"
#include "agc_record.h"

#define ARENA_FIRST UINT64_C(0x300000000)
#define ARENA_STEP UINT64_C(0x40000000)
#define ARENA_BYTES (UINT64_C(16) << 20)
#define SECTION_BYTES (UINT64_C(4) << 20)
#define PAGE UINT64_C(0x4000)
#define WINDOW (UINT64_C(8) << 20)
#define CB_OFFSET UINT64_C(0)
#define CB_BYTES UINT64_C(0x40000)
#define ARRAY_OFFSET UINT64_C(0x80000)
#define LABEL_OFFSET UINT64_C(0x90000)
#define MAX_ALIASES 64

typedef PVOID(WINAPI *virtual_alloc2_fn)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void *, ULONG);
typedef PVOID(WINAPI *map_view3_fn)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void *, ULONG);
typedef BOOL(WINAPI *unmap_view2_fn)(HANDLE, PVOID, ULONG);

struct alias {
   void *view;
   void *alias;
   bool live;
};

static struct {
   virtual_alloc2_fn allocate;
   map_view3_fn map;
   unmap_view2_fn unmap;
   HANDLE section;
   uint64_t base;
   struct alias aliases[MAX_ALIASES];
   agc_guest_import_release release;
   void *release_user;
   unsigned submits;
   unsigned events_added;
} guest;

static int failures;

static void say(const char *name, const char *format, ...) AGC_PRINTF(2, 3);

static void
say(const char *name, const char *format, ...)
{
   va_list args;
   va_start(args, format);
   printf("%s: ", name);
   vprintf(format, args);
   putchar('\n');
   va_end(args);
   fflush(stdout);
}

static void
expect(const char *what, uint32_t got, uint32_t value)
{
   if (got != value)
      failures++;
   say(got == value ? "PASS" : "FAIL", "%s: 0x%08x, expected 0x%08x", what, got, value);
}

static void *
host_map_import_alias(uintptr_t address, size_t bytes, uint32_t *refused)
{
   *refused = AGC_GUEST_IMPORT_MADE;
   if (address < guest.base || address + bytes > guest.base + SECTION_BYTES) {
      *refused = AGC_GUEST_IMPORT_NOT_GPU_VISIBLE;
      return NULL;
   }
   if (address % PAGE || bytes % PAGE) {
      *refused = AGC_GUEST_IMPORT_NOT_WHOLE_PAGES;
      return NULL;
   }
   if (address / WINDOW != (address + bytes - 1) / WINDOW) {
      *refused = AGC_GUEST_IMPORT_CROSSES_WINDOW;
      return NULL;
   }
   for (unsigned i = 0; i < MAX_ALIASES; i++) {
      if (guest.aliases[i].live)
         continue;
      uint64_t offset = address - guest.base;
      uint64_t lead = offset % 0x10000;
      void *view = guest.map(guest.section, GetCurrentProcess(), NULL, offset - lead, (SIZE_T)(lead + bytes), 0,
                             PAGE_READWRITE, NULL, 0);
      if (!view) {
         *refused = AGC_GUEST_IMPORT_NOT_SECTION_MEMORY;
         return NULL;
      }
      guest.aliases[i] = (struct alias){view, (uint8_t *)view + lead, true};
      return guest.aliases[i].alias;
   }
   *refused = AGC_GUEST_IMPORT_NOT_SECTION_MEMORY;
   return NULL;
}

static void
host_unmap_import_alias(void *alias)
{
   for (unsigned i = 0; i < MAX_ALIASES; i++) {
      if (guest.aliases[i].live && guest.aliases[i].alias == alias) {
         guest.unmap(GetCurrentProcess(), guest.aliases[i].view, 0);
         guest.aliases[i].live = false;
      }
   }
}

static size_t
host_gpu_visible(uintptr_t address, size_t bytes, struct agc_guest_range *pieces, size_t capacity)
{
   uint64_t start = MAX2((uint64_t)address, guest.base);
   uint64_t end = MIN2((uint64_t)address + bytes, guest.base + SECTION_BYTES);
   if (start >= end)
      return 0;
   if (capacity)
      pieces[0] = (struct agc_guest_range){(uintptr_t)start, (size_t)(end - start)};
   return 1;
}

static void
host_forget_import(uintptr_t address, size_t bytes)
{
   (void)address;
   (void)bytes;
}

static uint64_t
host_generation(void)
{
   return 1;
}

static void
host_set_import_release(agc_guest_import_release release, void *user)
{
   guest.release = release;
   guest.release_user = user;
}

static int32_t AGC_SYSV
host_add_data_event(int64_t queue, uintptr_t ident, int16_t filter, uint16_t flags, intptr_t data, void *udata)
{
   (void)queue;
   (void)ident;
   (void)filter;
   (void)flags;
   (void)data;
   (void)udata;
   guest.events_added++;
   return 0;
}

static int32_t AGC_SYSV
host_trigger_event(int64_t queue, uintptr_t ident, int16_t filter, void *data)
{
   (void)queue;
   (void)ident;
   (void)filter;
   (void)data;
   return 0;
}

static int32_t AGC_SYSV
host_delete_event(int64_t queue, uintptr_t ident, int16_t filter)
{
   (void)queue;
   (void)ident;
   (void)filter;
   return 0;
}

static void
host_note_gpu_submit(void)
{
   guest.submits++;
}

static const struct agc_anyps5 host_table = {
   .map_import_alias = host_map_import_alias,
   .unmap_import_alias = host_unmap_import_alias,
   .gpu_visible = host_gpu_visible,
   .forget_import = host_forget_import,
   .generation = host_generation,
   .gpu_generation = host_generation,
   .set_import_release = host_set_import_release,
   .add_data_event = host_add_data_event,
   .trigger_event = host_trigger_event,
   .delete_event = host_delete_event,
   .note_gpu_submit = host_note_gpu_submit,
};

static bool
guest_open(void)
{
   HMODULE kernel = GetModuleHandleW(L"KernelBase.dll");
   guest.allocate = (virtual_alloc2_fn)(void *)GetProcAddress(kernel, "VirtualAlloc2");
   guest.map = (map_view3_fn)(void *)GetProcAddress(kernel, "MapViewOfFile3");
   guest.unmap = (unmap_view2_fn)(void *)GetProcAddress(kernel, "UnmapViewOfFile2");
   if (!guest.allocate || !guest.map || !guest.unmap)
      return false;
   for (unsigned i = 0; i < 64 && !guest.base; i++) {
      uint64_t address = ARENA_FIRST + i * ARENA_STEP;
      if (guest.allocate(GetCurrentProcess(), (void *)(uintptr_t)address, (SIZE_T)ARENA_BYTES,
                         MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, NULL, 0))
         guest.base = address;
   }
   if (!guest.base)
      return false;
   guest.section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, (DWORD)SECTION_BYTES, NULL);
   if (!guest.section)
      return false;
   if (!VirtualFree((void *)(uintptr_t)guest.base, (SIZE_T)SECTION_BYTES, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER))
      return false;
   void *view = guest.map(guest.section, GetCurrentProcess(), (void *)(uintptr_t)guest.base, 0, (SIZE_T)SECTION_BYTES,
                          MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, NULL, 0);
   return view == (void *)(uintptr_t)guest.base;
}

static uint32_t *
guest_ptr(uint64_t offset)
{
   return (uint32_t *)(uintptr_t)(guest.base + offset);
}

static uint32_t *
emit_copy_reg(struct agc_cb *cb, uint32_t reg, uint64_t dst)
{
   uint32_t *p = agc_record_alloc(cb, 6);
   p[0] = PKT3(PKT3_COPY_DATA, 4, 0);
   p[1] = COPY_DATA_SRC_SEL(COPY_DATA_REG) | COPY_DATA_DST_SEL(COPY_DATA_DST_MEM) | COPY_DATA_WR_CONFIRM;
   p[2] = reg >> 2;
   p[3] = 0;
   p[4] = (uint32_t)dst;
   p[5] = (uint32_t)(dst >> 32);
   return p;
}

static bool
submit(struct agc_cb *cb, const char *what)
{
   struct agc_packet {
      const uint32_t *address;
      uint32_t dwords;
      uint8_t flags;
      uint8_t reserved[3];
   } packet = {cb->bottom, (uint32_t)(cb->up - cb->bottom), 0, {0}};
   int64_t start = os_time_get_nano();
   int32_t result = sceAgcDriverSubmitDcb((const struct agc_packet *)&packet);
   if (result) {
      say("FAIL submit", "%s: 0x%08x", what, (uint32_t)result);
      failures++;
      return false;
   }
   agc_driver_wait_idle();
   say("ran", "%s: %u dwords, %.3f ms including the wait", what, packet.dwords,
       (double)(os_time_get_nano() - start) / 1e6);
   return true;
}

int
main(void)
{
   char error[256];
   if (!guest_open()) {
      say("FAIL guest", "no guest arena (error %lu)", GetLastError());
      return 1;
   }
   say("guest", "4 MiB of section memory at 0x%" PRIx64, guest.base);
   agc_anyps5_set(&host_table);
   int64_t start = os_time_get_nano();
   if (!agc_driver_start(error, sizeof(error))) {
      say("FAIL start", "%s", error);
      return 1;
   }
   say("PASS start", "driver started in %.1f ms", (double)(os_time_get_nano() - start) / 1e6);

   const uint32_t red = 0x105, green = 0x106, blue = 0x107;
   const uint64_t labels = guest.base + LABEL_OFFSET;
   memset(guest_ptr(LABEL_OFFSET), 0xa5, 0x1000);
   struct agc_register *array = (struct agc_register *)guest_ptr(ARRAY_OFFSET);
   array[0] = (struct agc_register){green, 0x3e800000};
   array[1] = (struct agc_register){blue, 0x3e000000};

   struct agc_cb cb;
   agc_cb_init(&cb, guest_ptr(CB_OFFSET), CB_BYTES, NULL, NULL);
   sceAgcDcbSetCxRegisterDirect(&cb, (struct agc_register){red, 0x3f000000});
   uint32_t *indirect = sceAgcDcbSetCxRegistersIndirect(&cb, array, 1);
   emit_copy_reg(&cb, R_028414_CB_BLEND_RED, labels + 0);
   emit_copy_reg(&cb, R_028418_CB_BLEND_GREEN, labels + 4);
   emit_copy_reg(&cb, R_02841C_CB_BLEND_BLUE, labels + 8);
   uint32_t marker = 0x600d600d;
   sceAgcDcbWriteData(&cb, 5, 0, labels + 12, &marker, 1, 1, 1);
   if (sceAgcSetCxRegIndirectPatchSetNumRegisters(indirect, 2)) {
      say("FAIL patch", "set the register count");
      failures++;
   }
   if (!submit(&cb, "direct and indirect registers, copies, write data"))
      return 1;
   expect("CB_BLEND_RED (direct)", guest_ptr(LABEL_OFFSET)[0], 0x3f000000);
   expect("CB_BLEND_GREEN (indirect)", guest_ptr(LABEL_OFFSET)[1], 0x3e800000);
   expect("CB_BLEND_BLUE (indirect, count patched)", guest_ptr(LABEL_OFFSET)[2], 0x3e000000);
   expect("write data marker", guest_ptr(LABEL_OFFSET)[3], marker);

   agc_cb_reset(&cb);
   emit_copy_reg(&cb, R_028414_CB_BLEND_RED, labels + 16);
   emit_copy_reg(&cb, R_028418_CB_BLEND_GREEN, labels + 20);
   marker = 0x0dd0600d;
   sceAgcDcbWriteData(&cb, 5, 0, labels + 24, &marker, 1, 1, 1);
   if (!submit(&cb, "registers read back in a later submission"))
      return 1;
   expect("later submission: CB_BLEND_RED", guest_ptr(LABEL_OFFSET)[4], 0x3f000000);
   expect("later submission: CB_BLEND_GREEN", guest_ptr(LABEL_OFFSET)[5], 0x3e800000);
   expect("later submission: marker", guest_ptr(LABEL_OFFSET)[6], marker);

   double total = 0;
   const unsigned rounds = 100;
   for (unsigned i = 0; i < rounds; i++) {
      agc_cb_reset(&cb);
      sceAgcDcbSetCxRegistersIndirect(&cb, array, 2);
      marker = 0x1000 + i;
      sceAgcDcbWriteData(&cb, 5, 0, labels + 28, &marker, 1, 1, 1);
      int64_t t = os_time_get_nano();
      struct {
         const uint32_t *address;
         uint32_t dwords;
         uint8_t flags;
         uint8_t reserved[3];
      } packet = {cb.bottom, (uint32_t)(cb.up - cb.bottom), 0, {0}};
      if (sceAgcDriverSubmitDcb((const struct agc_packet *)&packet)) {
         failures++;
         break;
      }
      agc_driver_wait_idle();
      total += (double)(os_time_get_nano() - t) / 1e3;
   }
   expect("last of 100 submissions: marker", guest_ptr(LABEL_OFFSET)[7], 0x1000 + rounds - 1);
   say("timing", "submit + wait of a small buffer with one indirect array: %.1f us on average", total / rounds);

   struct agc_driver_counters counters;
   agc_driver_get_counters(&counters);
   say("counters", "%" PRIu64 " submits, %" PRIu64 " kernel submits, %" PRIu64 " register arrays, %" PRIu64
                   " converted registers, %u notes to libkernel",
       counters.submits, counters.kernel_submits, counters.register_arrays, counters.converted_registers,
       guest.submits);
   agc_driver_shutdown();
   say("result", "%s, %d failed checks", failures ? "FAIL" : "PASS", failures);
   return failures ? 1 : 0;
}
