#ifndef AGC_DRIVER_H
#define AGC_DRIVER_H

#include "agc_types.h"

struct radeon_info;
struct agc_backend;
struct agc_host_defaults;

#define AGC_QUEUE_GRAPHICS UINT32_C(0)
#define AGC_QUEUE_COMPUTE_FIRST UINT32_C(0x20)
#define AGC_QUEUE_COMPUTE_END UINT32_C(0x59)
#define AGC_QUEUE_COMPUTE_MULTI_END UINT32_C(0x58)

#define AGC_DRIVER_ERROR_INVALID ((int32_t)0x8a6c0001u)

struct agc_driver_counters {
   uint64_t submits;
   uint64_t kernel_submits;
   uint64_t submitted_dwords;
   uint64_t register_arrays;
   uint64_t converted_registers;
   uint64_t flips;
   uint64_t interrupts;
   uint64_t rendering_waits;
   uint64_t label_waits;
   uint64_t label_wait_ns;
   uint64_t label_releases;
   uint64_t arena_bytes;
};

struct agc_backend *agc_backend_open(void);
bool agc_driver_open_device(char *error, size_t error_bytes);
bool agc_driver_start(char *error, size_t error_bytes);
bool agc_driver_started(void);
struct agc_backend *agc_driver_backend(void);
const struct radeon_info *agc_driver_info(void);
int32_t agc_driver_submit(uint32_t queue, const uint32_t *address, uint32_t dwords, const char *function);
void agc_driver_set_tess_factor_ring(uint64_t base, uint32_t bytes);
void agc_driver_set_hs_offchip(uint32_t param);
void agc_driver_wait_idle(void);
void agc_driver_shutdown(void);
void agc_driver_fail(const char *format, ...) AGC_PRINTF(1, 2);
void agc_driver_get_counters(struct agc_driver_counters *counters);
void agc_driver_set_defaults(const struct agc_host_defaults *defaults);
bool agc_driver_reset_sequence(uint64_t *va, uint32_t *dwords);
bool agc_driver_context_sequence(uint32_t operation, uint64_t *va, uint32_t *dwords);
bool agc_driver_read_register(unsigned bank, uint32_t offset, uint32_t *value);
void agc_runtime_anchor(void);

#endif
