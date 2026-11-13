#ifndef AGC_IMAGES_H
#define AGC_IMAGES_H

#include <stdbool.h>
#include <stdint.h>

struct agc_backend;
struct agc_hwq;
struct radeon_info;

struct agc_images_stats {
   uint64_t reports;
   uint64_t converted;
   uint64_t direct;
   uint64_t rendered;
   uint64_t refused;
   uint64_t waiting;
   uint64_t released;
   uint64_t levels;
   uint64_t bytes;
   uint64_t batches;
   uint64_t targets;
   uint64_t cpu_converted;
   uint64_t refreshed;
   uint64_t turned;
   uint64_t kept;
   uint64_t stores;
   uint64_t unwatched;
   uint64_t checks;
   uint64_t blind;
   uint64_t wrote;
};

void agc_images_frame(struct agc_backend *backend, const struct radeon_info *info);
void agc_images_note_target(uint64_t guest_base);
void agc_images_released(void *user, uint64_t guest_va, uint64_t bytes);
void agc_images_stats(struct agc_images_stats *stats);
void agc_images_fini(void);

#endif
