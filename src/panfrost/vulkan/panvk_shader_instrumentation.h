/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_SHADER_INSTRUMENTATION_H
#define PANVK_SHADER_INSTRUMENTATION_H

#include <stdint.h>

#include "util/u_dynarray.h"
#include "vk_object.h"

#include "pan_compiler.h"

struct panvk_device;
struct panvk_priv_bo;

#define PANVK_SHADER_INSTRUMENTATION_SLOTS 2

struct panvk_shader_instrumentation_record {
   uint32_t result_index;
   VkShaderStageFlags stages[PANVK_SHADER_INSTRUMENTATION_SLOTS];
   uint64_t *counters;
};

struct panvk_shader_instrumentation {
   struct vk_object_base base;
   struct util_dynarray chunks;
   struct util_dynarray records;
   uint32_t chunk_used;
   uint32_t next_result_index;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_shader_instrumentation, base,
                               VkShaderInstrumentationARM,
                               VK_OBJECT_TYPE_SHADER_INSTRUMENTATION_ARM)

uint64_t panvk_shader_instrumentation_record(
   struct panvk_device *dev, struct panvk_shader_instrumentation *instr,
   const VkShaderStageFlags stages[PANVK_SHADER_INSTRUMENTATION_SLOTS]);

#ifdef PAN_ARCH
struct panvk_cmd_buffer;

void panvk_per_arch(cmd_instrument_draw)(struct panvk_cmd_buffer *cmdbuf);
void panvk_per_arch(cmd_instrument_dispatch)(struct panvk_cmd_buffer *cmdbuf);
#endif

#endif
