#ifndef AGC_ANYPS5_H
#define AGC_ANYPS5_H

#include "agc_types.h"

#define AGC_NID_BYTES 12

#define AGC_EVFILT_GRAPHICS_CORE ((int16_t)-14)
#define AGC_EV_ADD UINT16_C(0x0001)
#define AGC_EV_CLEAR UINT16_C(0x0020)

enum agc_guest_import {
   AGC_GUEST_IMPORT_MADE = 0,
   AGC_GUEST_IMPORT_NOT_GPU_VISIBLE = 1,
   AGC_GUEST_IMPORT_NOT_SECTION_MEMORY = 2,
   AGC_GUEST_IMPORT_NOT_WHOLE_PAGES = 3,
   AGC_GUEST_IMPORT_CROSSES_WINDOW = 4,
   AGC_GUEST_IMPORT_HOST_WRITE_OPEN = 5,
};

struct agc_guest_range {
   uintptr_t address;
   size_t bytes;
};

struct agc_guest_sparse_range {
   uintptr_t address;
   size_t bytes;
   uint32_t resident;
   uint32_t reserved;
};

typedef bool (*agc_guest_import_release)(void *user, uintptr_t address, size_t bytes);

struct agc_anyps5 {
   void *(*map_import_alias)(uintptr_t address, size_t bytes, uint32_t *refused);
   void (*unmap_import_alias)(void *alias);
   size_t (*import_alias_pages)(const void *alias, uint64_t *pages, size_t capacity);
   void *(*map_backing_alias)(uintptr_t address, size_t bytes, bool *bounds_refused);
   void (*unmap_backing_alias)(void *alias);
   size_t (*gpu_visible)(uintptr_t address, size_t bytes, struct agc_guest_range *pieces, size_t capacity);
   void (*forget_import)(uintptr_t address, size_t bytes);
   uint64_t (*generation)(void);
   uint64_t (*gpu_generation)(void);
   void (*arena_range)(uintptr_t *base, size_t *bytes);
   bool (*write_watched)(void);
   bool (*collect_writes)(uintptr_t address, size_t bytes, void **pages, size_t *count, bool clear);
   void (*set_import_release)(agc_guest_import_release release, void *user);
   size_t (*sparse_ranges)(uint64_t *generation, struct agc_guest_sparse_range *ranges, size_t capacity);
   int32_t (AGC_SYSV *add_data_event)(int64_t queue, uintptr_t ident, int16_t filter, uint16_t flags,
                                      intptr_t data, void *udata);
   int32_t (AGC_SYSV *trigger_event)(int64_t queue, uintptr_t ident, int16_t filter, void *data);
   int32_t (AGC_SYSV *delete_event)(int64_t queue, uintptr_t ident, int16_t filter);
   void (*note_gpu_submit)(void);
};

void agc_nid(const char *name, char out[AGC_NID_BYTES]);
void *agc_anyps5_find_export(const char *name);
bool agc_anyps5_resolve(struct agc_anyps5 *host, char *missing, size_t missing_bytes);
const struct agc_anyps5 *agc_anyps5_get(void);
void agc_anyps5_set(const struct agc_anyps5 *host);

#endif
