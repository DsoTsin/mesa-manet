/*
 * Copyright © 2026 PanVK contributors
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef LIBPAN_CSF_DGC_H
#define LIBPAN_CSF_DGC_H

#include "compiler/libcl/libcl.h"

/* Shared CPU/preprocess-shader ABI. None of the application stream is read by
 * the CPU. Keeping decoded state separate from executable CSF also lets an
 * explicitly preprocessed stream inherit the execution command buffer's
 * render-pass and descriptor state.
 */
#define PANLIB_DGC_MAX_TOKENS 16
#define PANLIB_DGC_MAX_VERTEX_BUFFERS 16
#define PANLIB_DGC_PUSH_WORDS 64
#define PANLIB_DGC_STAGE_COUNT 3
#define PANLIB_DGC_VS 0
#define PANLIB_DGC_FS 1
#define PANLIB_DGC_CS 2
#define PANLIB_DGC_SEQUENCE_CS_SIZE 64
/* VkIndexType values, shared with the OpenCL preprocessing kernel. */
enum panlib_dgc_index_type {
   PANLIB_DGC_INDEX_UINT16 = 0,
   PANLIB_DGC_INDEX_UINT32 = 1,
   PANLIB_DGC_INDEX_UINT8 = 1000265000,
};

enum panlib_dgc_token_type {
   PANLIB_DGC_TOKEN_EXECUTION_SET,
   PANLIB_DGC_TOKEN_PUSH_CONSTANT,
   PANLIB_DGC_TOKEN_SEQUENCE_INDEX,
   PANLIB_DGC_TOKEN_INDEX_BUFFER,
   PANLIB_DGC_TOKEN_VERTEX_BUFFER,
   PANLIB_DGC_TOKEN_DRAW,
   PANLIB_DGC_TOKEN_DRAW_INDEXED,
   PANLIB_DGC_TOKEN_DRAW_COUNT,
   PANLIB_DGC_TOKEN_DRAW_INDEXED_COUNT,
   PANLIB_DGC_TOKEN_DISPATCH,
};

struct panlib_dgc_token {
   uint32_t type;
   uint32_t offset;
   /* PC destination byte offset, or vertex buffer binding. */
   uint32_t target;
   uint32_t size;
   /* PANLIB_DGC_{VS,FS,CS} bits, not VkShaderStageFlags. */
   uint32_t stages;
};

struct panlib_dgc_vertex_buffer {
   uint64_t address;
   uint32_t size;
   uint32_t stride;
};

struct panlib_dgc_index_buffer {
   uint64_t address;
   uint32_t size;
   /* Vulkan VkIndexType, including UINT8. */
   uint32_t type;
};

struct panlib_dgc_sequence {
   uint32_t push_constants[PANLIB_DGC_STAGE_COUNT][PANLIB_DGC_PUSH_WORDS];
   struct panlib_dgc_vertex_buffer vb[PANLIB_DGC_MAX_VERTEX_BUFFERS];
   struct panlib_dgc_index_buffer ib;
   /* IDVS INDEX_COUNT, INSTANCE_COUNT, INDEX_OFFSET, VERTEX_OFFSET and
    * INSTANCE_OFFSET. Non-indexed draws have INDEX_OFFSET = 0.
    */
   uint32_t draw[5];
   uint32_t dispatch[3];
   uint32_t index;
   uint32_t execution_set[PANLIB_DGC_STAGE_COUNT];
   uint64_t draw_count_address;
   uint32_t draw_count_stride;
   uint32_t draw_count;
} __attribute__((aligned(64)));

/* The executable stream begins immediately after this cache-line header. */
struct panlib_dgc_header {
   uint32_t sequence_count;
   uint32_t cs_size;
   uint32_t reserved[14];
};

/* CALL target selected at execution, not at preprocessing. Each slot is
 * naturally aligned so the address and size can be loaded with one CSF load.
 */
struct panlib_dgc_program {
   uint64_t address;
   uint32_t size;
   uint32_t reserved;
};

struct panlib_dgc_preprocess {
   uint64_t stream;
   uint64_t count;
   uint64_t output;
   uint64_t sequence_offset;
   uint32_t sequence_stride;
   uint32_t max_sequences;
   uint32_t stream_stride;
   uint32_t token_count;
   uint32_t index_mode_is_dx;
   uint32_t execution_set_is_shaders;
   uint32_t scratch_register;
   uint32_t context_register;
   uint32_t program_table_offset;
   uint32_t load_store_scoreboard;
   uint32_t max_draw_count;
   uint32_t reserved;
   struct panlib_dgc_sequence initial;
   struct panlib_dgc_token tokens[PANLIB_DGC_MAX_TOKENS];
};

#endif /* LIBPAN_CSF_DGC_H */
