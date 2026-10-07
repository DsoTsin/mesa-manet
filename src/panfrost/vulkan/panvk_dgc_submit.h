/*
 * Copyright © 2026 PanVK contributors
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_DGC_SUBMIT_H
#define PANVK_DGC_SUBMIT_H

#include "genxml/gen_macros.h"
#include "libpan_csf_dgc_execute.h"
#include "panvk_macros.h"
#include "util/list.h"
#include "util/u_dynarray.h"

struct panvk_cmd_buffer;
struct panvk_device;
struct panvk_gpu_queue;
struct panvk_indirect_execution_set;
struct panvk_shader_variant;
struct vk_queue_submit;

/* The pointers below refer to immutable command-buffer templates or device
 * cache entries. Execution-set metadata is read afresh at queue submission. */
struct panvk_dgc_job {
   struct panlib_dgc_execute params;
   struct panvk_indirect_execution_set *execution_set;
   const struct panlib_dgc_execution_set_entry *fixed_pipeline;
   const struct panvk_shader_variant *prepare_shader;
   uint32_t max_sequences;
   bool graphics;
};

struct panvk_dgc_record {
   struct list_head link;
   enum { PANVK_DGC_RECORD_EXECUTE, PANVK_DGC_RECORD_SECONDARY } kind;
   union {
      struct panvk_dgc_job job;
      struct panvk_cmd_buffer *secondary;
   };
};

VkResult panvk_per_arch(dgc_record_execute)(struct panvk_cmd_buffer *cmdbuf,
                                          const struct panvk_dgc_job *job);
VkResult panvk_per_arch(dgc_record_secondary)(struct panvk_cmd_buffer *cmdbuf,
                                            struct panvk_cmd_buffer *secondary);
void panvk_per_arch(dgc_records_reset)(struct panvk_cmd_buffer *cmdbuf);

/* A batch owns all GPU-written state for every occurrence of Execute in one
 * submission. Its two lists are consumed independently by compute and VT.
 * The queue owns retirement, including completion of fragment work. */
struct panvk_dgc_submit {
   struct list_head link;
   struct util_dynarray bos;
   uint64_t heads[3];
   uint64_t *tails[3];
   struct { uint64_t gpu; uint32_t size; } prefix[3];
   uint64_t target_seqnos[3];
   uint32_t completion_syncobjs[3];
   bool submitted;
   bool completion_known;
};

VkResult panvk_per_arch(dgc_submit_prepare)(struct panvk_gpu_queue *queue,
                                          const struct vk_queue_submit *submit,
                                          struct panvk_dgc_submit **out);
void panvk_per_arch(dgc_submit_destroy)(struct panvk_device *dev,
                                      struct panvk_dgc_submit *submit);

#endif
