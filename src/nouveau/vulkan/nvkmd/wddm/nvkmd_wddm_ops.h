#ifndef NVKMD_WDDM_OPS_H
#define NVKMD_WDDM_OPS_H

#include "../nvkmd.h"
#include "nvkmd_wddm.h"

VkResult nvkmd_wddm_create_pdev(const struct nvkmd_wddm_adapter_info *adapter,
                               struct vk_object_base *log_obj,
                               enum nvk_debug debug_flags,
                               struct nvkmd_pdev **pdev_out);

VkResult nvkmd_wddm_wrap_memory(struct nvkmd_dev *dev,
                                struct nvkmd_wddm_memory *owned,
                                uint64_t size, struct nvkmd_mem **out);

const struct nvkmd_wddm_device *
nvkmd_wddm_native_device(struct nvkmd_dev *dev);

VkResult nvkmd_wddm_ctx_get_reflected_sked_address(struct nvkmd_ctx *ctx,
                                                uint64_t *out);

#endif
