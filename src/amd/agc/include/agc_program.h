#ifndef AGC_PROGRAM_H
#define AGC_PROGRAM_H

#include "agc_shader.h"

struct agc_program_link {
   uint64_t second_half;
   const uint32_t *second_words;
   uint32_t second_word_count;
};

struct agc_program_counters {
   uint64_t translated;
   uint64_t cache_hits;
   uint64_t oracle_runs;
   uint64_t aco_translations;
   uint64_t aco_fallbacks;
   uint64_t placed_bytes;
   uint64_t external_targets;
};

int32_t agc_program_translate(struct agc_shader *shader, uint64_t guest_code, uint64_t *host_code);
int32_t agc_program_translate_linked(struct agc_shader *shader, uint64_t guest_code, const uint32_t *words,
                                     uint32_t word_count, const struct agc_program_link *link,
                                     uint64_t *host_code);
void agc_program_install(void);
void agc_program_get_counters(struct agc_program_counters *counters);

#endif
