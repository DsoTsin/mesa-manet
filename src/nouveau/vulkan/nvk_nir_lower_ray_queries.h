#ifndef NVK_NIR_LOWER_RAY_QUERIES_H
#define NVK_NIR_LOWER_RAY_QUERIES_H

#include "nir/nir.h"

bool nvk_nir_lower_ray_queries(nir_shader *nir);
bool nvk_nir_lower_ray_queries_with_omm(nir_shader *nir,
                                       bool opacity_micromap);

#endif
