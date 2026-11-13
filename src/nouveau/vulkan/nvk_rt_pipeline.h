#ifndef NVK_RT_PIPELINE_H
#define NVK_RT_PIPELINE_H

#include "nvk_nir_rt_pipeline.h"
#include "vk_pipeline.h"

struct nvk_shader;
struct vk_pipeline_layout;
struct nvk_cmd_buffer;

#define NVK_RT_STACK_LANE_BLOCK_SIZE 16

struct nvk_rt_group {
   uint32_t stages[3];
   uint8_t handle[NVK_RT_GROUP_HANDLE_SIZE];
};

struct nvk_rt_pipeline {
   struct vk_pipeline vk;
   void *mem_ctx;
   struct vk_pipeline_layout *layout;
   struct nvk_shader *shader;
   struct nvk_nir_rt_pipeline_stage *stages;
   struct nvk_rt_group *groups;
   uint32_t stage_count;
   uint32_t group_count;
   uint32_t frame_size;
   uint32_t stack_size;
   uint32_t storage_size;
   uint32_t *per_stage_stack_size;
   bool dynamic_stack_size;
   bool uses_ser_queue;
};

struct nvk_rt_dispatch {
   VkTraceRaysIndirectCommand2KHR command;
   uint64_t stack_stride;
   uint64_t push_addr;
   uint32_t worker_count;
   uint32_t indirect_mode;
   uint64_t indirect_addr;
   uint64_t stack_addr;
   uint64_t ser_queue_addr;
};

void nvk_cmd_bind_rt_pipeline(struct nvk_cmd_buffer *cmd,
                              struct nvk_rt_pipeline *pipeline);

#endif
