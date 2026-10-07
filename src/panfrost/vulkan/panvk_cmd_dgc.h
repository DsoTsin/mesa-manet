/*
 * Copyright © 2026 PanVK contributors
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_CMD_DGC_H
#define PANVK_CMD_DGC_H

#include "libpan_csf_dgc_execute.h"
#include "panvk_macros.h"
#include "vulkan/vulkan_core.h"

struct panvk_cmd_buffer;
struct panvk_indirect_command_layout;

VkResult panvk_per_arch(cmd_prepare_dgc_draw)(
   struct panvk_cmd_buffer *cmdbuf,
   const struct panvk_indirect_command_layout *layout,
   uint32_t max_sequences, struct panlib_dgc_execute *params);

VkResult panvk_per_arch(cmd_prepare_dgc_dispatch)(
   struct panvk_cmd_buffer *cmdbuf, struct panlib_dgc_execute *params);

#endif /* PANVK_CMD_DGC_H */
