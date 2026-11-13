#ifndef NVK_BVH_CUDA_H
#define NVK_BVH_CUDA_H

#include "nvk_private.h"

struct nvk_cmd_buffer;
struct nvk_compute_state;

void nvk_bvh_meta_begin(struct nvk_cmd_buffer *cmd, struct nvk_compute_state *saved);
void nvk_bvh_meta_end(struct nvk_cmd_buffer *cmd, const struct nvk_compute_state *saved);

VkResult nvk_bvh_cuda_launch(struct nvk_cmd_buffer *cmd,
                              const char *name, const void *args, size_t args_size,
                              uint32_t grid_x, uint32_t block_x);

#endif
