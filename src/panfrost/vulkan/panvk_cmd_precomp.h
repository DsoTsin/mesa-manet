/*
 * Copyright © 2024 Collabora Ltd.
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_CMD_PRECOMP_H
#define PANVK_CMD_PRECOMP_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined"
#endif

#include "genxml/gen_macros.h"
#include "util/simple_mtx.h"
#include "libpan_dgc.h"
#include "libpan_shaders.h"
#include "panvk_macros.h"

struct panvk_cmd_buffer;

struct panvk_precomp_ctx {
   struct panvk_cmd_buffer *cmdbuf;
   /* DGC execution prepares per-submit FAUs/TSDs in the queue arena. */
   bool dgc;
};

static inline struct panvk_precomp_ctx
panvk_per_arch(precomp_cs)(struct panvk_cmd_buffer *cmdbuf)
{
   return (struct panvk_precomp_ctx){.cmdbuf = cmdbuf};
}

enum libpan_shaders_program;
void panvk_per_arch(dispatch_precomp)(struct panvk_precomp_ctx *ctx,
                                      struct panlib_precomp_grid grid,
                                      enum panlib_barrier barrier,
                                      enum libpan_shaders_program idx,
                                      void *data, size_t data_size);

#define MESA_DISPATCH_PRECOMP panvk_per_arch(dispatch_precomp)

#endif
