#ifndef AGC_ISA_TRANSLATE_H
#define AGC_ISA_TRANSLATE_H

#include "agc_shader_artifact.h"

#include "agc_isa.h"

#ifdef __cplusplus
extern "C" {
#endif

bool agc_isa_translate(const struct agc_shader_request *request, struct agc_shader_artifact *artifact,
                       struct agc_isa_error *error);
bool agc_isa_translate_listing(const struct agc_shader_request *request, char **listing, struct agc_isa_error *error);

#ifdef __cplusplus
}
#endif

#endif
