#ifndef AGC_SHADER_FUSION_H
#define AGC_SHADER_FUSION_H

#include "agc_shader.h"

struct agc_size_align {
   uint64_t size;
   size_t align;
};

int32_t agc_shader_fuse(struct agc_shader *result, const struct agc_shader *front, const struct agc_shader *back,
                        struct agc_register *scratch);
int32_t agc_shader_fused_size(struct agc_size_align *destination, const struct agc_shader *front,
                              const struct agc_shader *back);

#endif
