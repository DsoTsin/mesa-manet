#ifndef AGC_CB_H
#define AGC_CB_H

#include "agc_types.h"

struct agc_cb;

typedef bool (AGC_SYSV *agc_cb_out_of_memory)(struct agc_cb *cb, uint32_t dwords, void *user_data);

struct agc_cb {
   uint32_t *bottom;
   uint32_t *top;
   uint32_t *up;
   uint32_t *down;
   agc_cb_out_of_memory callback;
   void *user_data;
   uint32_t reserved_dwords;
};

void agc_cb_init(struct agc_cb *cb, void *buffer, uint64_t bytes, agc_cb_out_of_memory callback, void *user_data);
bool agc_cb_grow(struct agc_cb *cb, uint32_t dwords);
void *agc_cb_alloc_down(struct agc_cb *cb, uint64_t bytes, size_t alignment);
void *agc_cb_alloc_up_aligned(struct agc_cb *cb, uint64_t bytes, size_t alignment);

static inline void
agc_cb_reset(struct agc_cb *cb)
{
   cb->up = cb->bottom;
   cb->down = cb->top;
}

static inline uint32_t
agc_cb_remaining_dwords(const struct agc_cb *cb)
{
   uint32_t space = (uint32_t)(cb->down - cb->up);
   return space > cb->reserved_dwords ? space - cb->reserved_dwords : 0;
}

static inline uint64_t
agc_cb_used_up_bytes(const struct agc_cb *cb)
{
   return (uint64_t)(cb->up - cb->bottom) * sizeof(uint32_t);
}

static inline uint64_t
agc_cb_used_down_bytes(const struct agc_cb *cb)
{
   return (uint64_t)(cb->top - cb->down) * sizeof(uint32_t);
}

static inline bool
agc_cb_reserve(struct agc_cb *cb, uint32_t dwords)
{
   if (AGC_LIKELY(dwords <= agc_cb_remaining_dwords(cb)))
      return true;
   return agc_cb_grow(cb, dwords);
}

static inline uint32_t *
agc_cb_alloc_up(struct agc_cb *cb, uint32_t dwords)
{
   if (AGC_UNLIKELY(!agc_cb_reserve(cb, dwords)))
      return NULL;
   uint32_t *out = cb->up;
   cb->up = out + dwords;
   return out;
}

#endif
