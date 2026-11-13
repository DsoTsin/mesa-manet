#ifndef NVK_CMD_DISPATCH_H
#define NVK_CMD_DISPATCH_H

#include "nvk_private.h"

struct nvk_cmd_buffer;
struct nvk_descriptor_state;
struct nvk_shader;
struct nvk_ser_queue;

struct nvk_prepared_dispatch {
   uint64_t qmd_addr;
   void *qmd_map;
};

struct nvk_ser_dispatch {
   struct nvk_prepared_dispatch init;
   struct nvk_prepared_dispatch shade;
   struct nvk_ser_queue *queue_map;
   uint64_t queue_addr;
   uint64_t completion_addr;
   uint32_t owner_capacity;
};

VkResult nvk_cmd_prepare_dispatch_with_root(
   struct nvk_cmd_buffer *cmd, struct nvk_shader *shader,
   const struct nvk_descriptor_state *desc, const void *root, size_t root_size,
   const uint32_t group_count[3], uint32_t local_size_high,
   struct nvk_prepared_dispatch *out);

VkResult nvk_cmd_prepare_cs_dispatch(struct nvk_cmd_buffer *cmd,
                                    const uint32_t group_count[3],
                                    struct nvk_prepared_dispatch *out);

void nvk_cmd_launch_prepared_dispatch(struct nvk_cmd_buffer *cmd,
                                      const struct nvk_prepared_dispatch *dispatch,
                                      bool unconditional);

VkResult nvk_cmd_prepare_ser_queue(struct nvk_cmd_buffer *cmd,
                                  uint32_t owner_capacity,
                                  struct nvk_ser_dispatch *out);
VkResult nvk_cmd_prepare_ser_shade(struct nvk_cmd_buffer *cmd,
                                  struct nvk_ser_dispatch *dispatch);
void nvk_cmd_launch_ser_dispatch(struct nvk_cmd_buffer *cmd,
                                 const struct nvk_ser_dispatch *dispatch);

#endif
