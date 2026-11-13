#ifndef NVK_NIR_SER_H
#define NVK_NIR_SER_H

#include "nir.h"

#define NVK_HIT_OBJECT_SIZE 512u
#define NVK_HIT_OBJECT_RECORD 256u
#define NVK_HIT_OBJECT_ATTRIBUTE 464u
#define NVK_HIT_OBJECT_TYPE 496u
#define NVK_HIT_OBJECT_SBT_INDEX 500u
#define NVK_HIT_OBJECT_TIME 504u
#define NVK_HIT_OBJECT_EMPTY 0u
#define NVK_HIT_OBJECT_MISS 1u
#define NVK_HIT_OBJECT_HIT 2u

bool nvk_nir_lower_hit_objects(nir_shader *nir);
bool nvk_nir_lower_hit_objects_with_reorder(nir_shader *nir, bool reorder);

#endif
