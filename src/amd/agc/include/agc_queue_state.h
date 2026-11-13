#ifndef AGC_QUEUE_STATE_H
#define AGC_QUEUE_STATE_H

#include "agc_memory.h"

struct radeon_info;
struct agc_backend;
struct agc_hwq;

#define AGC_CONTEXT_OPS 4u

struct agc_host_defaults {
   const struct agc_register *registers[3];
   uint32_t count[3];
};

struct agc_queue_state {
   struct agc_mem shadow;
   struct agc_mem defaults;
   struct agc_mem fixed;
   struct agc_mem attribute_ring;
   struct agc_mem saved;
   uint64_t image_bytes;
   uint64_t preamble_va;
   uint32_t preamble_dwords;
   uint64_t reset_va;
   uint32_t reset_dwords;
   uint64_t attribute_ring_bytes;
   uint64_t context_va[AGC_CONTEXT_OPS];
   uint32_t context_dwords[AGC_CONTEXT_OPS];
};

bool agc_queue_state_init(struct agc_backend *be, const struct radeon_info *info, struct agc_hwq *hwq,
                          const struct agc_host_defaults *defaults, struct agc_queue_state *state,
                          char *error, size_t error_bytes);
void agc_queue_state_fini(struct agc_queue_state *state);
bool agc_queue_state_read(const struct agc_queue_state *state, unsigned bank, uint32_t offset, uint32_t *value);

#endif
