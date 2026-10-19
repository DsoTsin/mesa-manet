/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_RAY_TRACING_H
#define PANVK_RAY_TRACING_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined"
#endif

#include "nir_builder.h"
#include "vulkan/vulkan_core.h"

#include "panvk_macros.h"

#define PANVK_MAX_RAY_QUERY_SLOTS  8
#define PANVK_RAY_QUERY_STATE_SIZE 128
#define PANVK_RT_EXTERNAL_ROOT_OFFSET 232
#define PANVK_RT_EXTERNAL_SCALE_OFFSET 224

static inline nir_def *
panvk_nir_rt_state_addr(nir_builder *b, nir_def *root, nir_def *slot)
{
   nir_def *base = nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, root, PANVK_RT_EXTERNAL_ROOT_OFFSET), .align_mul = 8);
   nir_def *scale = nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, root, PANVK_RT_EXTERNAL_SCALE_OFFSET), .align_mul = 8);
   return nir_iadd(b, base, nir_imul(b, nir_ishl_imm(b, nir_u2u64(b, slot), 7), scale));
}

struct panvk_device;

void panvk_per_arch(device_init_accel_struct)(struct panvk_device *dev);

VkResult panvk_per_arch(device_reserve_ray_query_slots)(
   struct panvk_device *dev, uint32_t slots);

void panvk_per_arch(device_finish_ray_query)(struct panvk_device *dev);

uint32_t panvk_per_arch(nir_lower_ray_queries)(nir_shader *nir,
                                               uint32_t slot_base);

uint32_t panvk_per_arch(nir_lower_rt_ray_queries)(nir_shader *nir,
                                                 uint32_t frame_offset);

#endif
