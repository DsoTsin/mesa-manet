#include "agc_cb.h"

_Static_assert(sizeof(struct agc_cb) == 56, "TwoSidedAllocator size");
_Static_assert(offsetof(struct agc_cb, bottom) == 0, "m_bottom");
_Static_assert(offsetof(struct agc_cb, top) == 8, "m_top");
_Static_assert(offsetof(struct agc_cb, up) == 16, "m_upCursor");
_Static_assert(offsetof(struct agc_cb, down) == 24, "m_downCursor");
_Static_assert(offsetof(struct agc_cb, callback) == 32, "m_callback");
_Static_assert(offsetof(struct agc_cb, user_data) == 40, "m_userData");
_Static_assert(offsetof(struct agc_cb, reserved_dwords) == 48, "m_reservedSpaceInDwords");

void
agc_cb_init(struct agc_cb *cb, void *buffer, uint64_t bytes, agc_cb_out_of_memory callback, void *user_data)
{
   cb->bottom = buffer;
   cb->top = (uint32_t *)buffer + bytes / sizeof(uint32_t);
   cb->up = cb->bottom;
   cb->down = cb->top;
   cb->callback = callback;
   cb->user_data = user_data;
   cb->reserved_dwords = 0;
}

bool
agc_cb_grow(struct agc_cb *cb, uint32_t dwords)
{
   if (!cb->callback || !cb->callback(cb, dwords + cb->reserved_dwords, cb->user_data))
      return false;
   return dwords <= agc_cb_remaining_dwords(cb);
}

static bool
agc_cb_reserve_aligned(struct agc_cb *cb, uint64_t bytes, size_t alignment, uint64_t *dwords)
{
   uint64_t size = (bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
   uint64_t slack = (alignment + sizeof(uint32_t) - 1) / sizeof(uint32_t) - 1;
   if (!bytes || size + slack > UINT32_MAX)
      return false;
   *dwords = size;
   return agc_cb_reserve(cb, (uint32_t)(size + slack));
}

void *
agc_cb_alloc_down(struct agc_cb *cb, uint64_t bytes, size_t alignment)
{
   uint64_t dwords;
   if (!agc_cb_reserve_aligned(cb, bytes, alignment, &dwords))
      return NULL;
   uintptr_t at = (uintptr_t)(cb->down - dwords) & ~((uintptr_t)alignment - 1);
   cb->down = (uint32_t *)at;
   return cb->down;
}

void *
agc_cb_alloc_up_aligned(struct agc_cb *cb, uint64_t bytes, size_t alignment)
{
   uint64_t dwords;
   if (!agc_cb_reserve_aligned(cb, bytes, alignment, &dwords))
      return NULL;
   uintptr_t mask = (uintptr_t)alignment - 1;
   uint32_t *out = (uint32_t *)(((uintptr_t)cb->up + mask) & ~mask);
   cb->up = out + dwords;
   return out;
}
