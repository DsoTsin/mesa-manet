/*
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_BLEND_STATE_H
#define PANVK_BLEND_STATE_H

#include <stdbool.h>
#include <stdint.h>

#include "pan_blend.h"

#include "panvk_macros.h"

#include "vk_graphics_state.h"

struct panvk_blend_static_key {
   bool valid;
   bool alpha_to_one;
   bool logic_op_enable;
   uint8_t logic_op;
   uint8_t rt_count;
   uint8_t color_write_enables;
   uint8_t color_map[MESA_VK_MAX_COLOR_ATTACHMENTS];
   struct vk_color_blend_attachment_state attachments[MESA_VK_MAX_COLOR_ATTACHMENTS];
   VkFormat formats[MESA_VK_MAX_COLOR_ATTACHMENTS];
};

#ifdef PAN_ARCH

void panvk_per_arch(blend_static_key_init)(
   struct panvk_blend_static_key *key,
   const struct vk_graphics_pipeline_state *state);

uint8_t panvk_per_arch(blend_fixed_function_locations)(
   const struct panvk_blend_static_key *key);

void panvk_per_arch(blend_fill_rt)(
   struct pan_blend_rt_state *rt,
   const struct vk_color_blend_attachment_state *att,
   enum pipe_format format, uint8_t nr_samples, const float *constants);

bool panvk_per_arch(blend_skips_rt)(bool logicop_enable,
                                    enum pipe_logicop logicop_func,
                                    uint8_t color_write_enables,
                                    uint8_t rt_idx, VkFormat format,
                                    uint8_t write_mask);

bool panvk_per_arch(blend_needs_shader)(const struct pan_blend_state *state,
                                        unsigned rt_idx,
                                        unsigned *ff_blend_constant);

#endif

#endif
