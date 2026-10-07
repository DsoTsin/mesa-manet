/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_RAY_TRACING_H
#define PANVK_RAY_TRACING_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined"
#endif

#include "nir.h"
#include "vulkan/vulkan_core.h"

#include "panvk_macros.h"

#define PANVK_MAX_RAY_QUERY_SLOTS  8
#define PANVK_RAY_QUERY_STATE_SIZE 128

struct panvk_device;

void panvk_per_arch(device_init_accel_struct)(struct panvk_device *dev);

VkResult panvk_per_arch(device_reserve_ray_query_slots)(
   struct panvk_device *dev, uint32_t slots);

void panvk_per_arch(device_finish_ray_query)(struct panvk_device *dev);

uint32_t panvk_per_arch(nir_lower_ray_queries)(nir_shader *nir);

#endif
