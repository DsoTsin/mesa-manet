/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#ifndef NVK_QUEUE_H
#define NVK_QUEUE_H 1

#include "nvk_mem_stream.h"
#include "nv_push.h"

#include "vk_queue.h"
#include "nvkmd/nvkmd.h"

struct nouveau_ws_bo;
struct nouveau_ws_context;
struct novueau_ws_push;
struct nvk_device;
struct nvk_physical_device;
struct nvkmd_mem;
struct nvkmd_ctx;

struct nvk_queue_state {
   uint32_t shader_upload_seq;

   struct {
      uint32_t alloc_count;
   } images;

   struct {
      uint32_t alloc_count;
   } samplers;

   struct {
      struct nvkmd_mem *mem;
      uint32_t bytes_per_warp;
      uint32_t bytes_per_tpc;
   } slm;
};

struct nvk_queue {
   struct vk_queue vk;

   enum nvkmd_engines engines;

   struct nvkmd_ctx *bind_ctx;
   struct nvkmd_ctx *exec_ctx;

   /* Memory stream to use for anything we need to push that isn't part of a
    * command buffer.
    */
   struct nvk_mem_stream push_stream;

   struct nvk_queue_state state;

   /* CB0 for all draw commands on this queue */
   struct nvkmd_mem *draw_cb0;
};

static inline struct nvk_device *
nvk_queue_device(struct nvk_queue *queue)
{
   return (struct nvk_device *)queue->vk.base.device;
}

enum nvkmd_engines
nvk_queue_engines_from_queue_flags(const struct nvk_physical_device *pdev,
                                  VkQueueFlags queue_flags);

static inline uint8_t
nvk_queue_subchannels_from_engines(enum nvkmd_engines engines)
{
   /* Note: These line up with nouveau_ws_context_create */
   uint8_t subc_mask = 0;

   if (engines & NVKMD_ENGINE_COPY)
      subc_mask |= BITFIELD_BIT(SUBC_NV90B5);

   if (engines & NVKMD_ENGINE_2D)
      subc_mask |= BITFIELD_BIT(SUBC_NV902D);

   if (engines & NVKMD_ENGINE_3D)
      subc_mask |= BITFIELD_BIT(SUBC_NV9097);

   if (engines & NVKMD_ENGINE_M2MF)
      subc_mask |= BITFIELD_BIT(SUBC_NV9039);

   if (engines & NVKMD_ENGINE_COMPUTE)
      subc_mask |= BITFIELD_BIT(SUBC_NV90C0);

   if (engines & NVKMD_ENGINE_VDEC)
      subc_mask |= BITFIELD_BIT(SUBC_NVC5B0);

   return subc_mask;
}

VkResult nvk_queue_create(struct nvk_device *dev,
                          const VkDeviceQueueCreateInfo *pCreateInfo,
                          uint32_t index_in_family);

void nvk_queue_destroy(struct nvk_device *dev, struct nvk_queue *queue);

VkResult nvk_push_draw_state_init(struct nvk_queue *queue,
                                  struct nv_push *p);

VkResult nvk_push_dispatch_state_init(struct nvk_queue *queue,
                                      struct nv_push *p);

#endif
