/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#include "nvk_entrypoints.h"
#include "nvk_acceleration_structure.h"
#include "nvk_cmd_buffer.h"
#include "nvk_device.h"
#include "nvk_bvh_cuda.h"
#include "nvk_micromap.h"
#include "nvk_physical_device.h"
#include "bvh/nvk_bvh_build_cuda.h"
#include "bvh/nvk_bvh_pack.h"
#include "bvh/nvk_bvh_leaf.h"
#include "bvh/nvk_bvh_planner.h"
#include "nvk_query_pool.h"
#include "nvk_shader.h"
#include "nvkmd/nvkmd.h"
#include "bvh/nvk_bvh.h"
#include "vk_acceleration_structure.h"
#include "vk_command_buffer.h"
#include "vk_meta.h"
#include "bvh/vk_bvh_defines.h"
#include "nv_push_cl90c0.h"

struct nvk_bvh_layout {
   VkDeviceSize metadata;
   VkDeviceSize nodes;
   VkDeviceSize leaves;
   VkDeviceSize primitive_data;
   VkDeviceSize refit_data;
   VkDeviceSize refit_leaf_map;
   VkDeviceSize omm_metadata;
   VkDeviceSize size;
};

static struct vk_acceleration_structure_build_args
nvk_bvh_build_args(void)
{
   return (struct vk_acceleration_structure_build_args) {
      .subgroup_size = 32,
      .has_update = true,
      .bvh_bounds_offset = NVK_BVH_BOUNDS_OFFSET,
      .root_flags_offset = NVK_BVH_ROOT_FLAGS_OFFSET,
      .morton_sort_workgroup_size = 256,
      .morton_sort_kvs_per_thread = 2,
   };
}

static struct nvk_bvh_layout
nvk_bvh_layout(VkGeometryTypeKHR type, uint32_t leaf_count, bool updateable)
{
   const VkDeviceSize boxes = MAX2(leaf_count, 2) - 1;
   VkDeviceSize offset = NVK_BVH_HEADER_SIZE;
   struct nvk_bvh_layout layout = {0};
   if (type == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
      layout.metadata = offset;
      offset += DIV_ROUND_UP((VkDeviceSize)leaf_count, 16) *
                NVK_BVH_INSTANCE_DATA_UNIT_SIZE;
   }
   layout.nodes = ALIGN_POT(offset, 128);
   layout.leaves = layout.nodes + boxes * NVK_BVH_NODE_SIZE;
   uint32_t leaf_size = type == VK_GEOMETRY_TYPE_TRIANGLES_KHR ? 128 :
                        type == VK_GEOMETRY_TYPE_AABBS_KHR ? 8 : 64;
   offset = layout.leaves + (VkDeviceSize)leaf_count * leaf_size;
   layout.primitive_data = ALIGN_POT(offset, 8);
   offset = layout.primitive_data + (VkDeviceSize)leaf_count * 8;
   if (updateable) {
      layout.refit_data = ALIGN_POT(offset, 32);
      offset = layout.refit_data + boxes * 32;
   }
   if (type != VK_GEOMETRY_TYPE_INSTANCES_KHR) {
      layout.refit_leaf_map = ALIGN_POT(offset, 4);
      offset = layout.refit_leaf_map + (VkDeviceSize)leaf_count * 4;
      if (type == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
         offset = ALIGN_POT(offset, 128) + (VkDeviceSize)leaf_count * 2;
         layout.omm_metadata = ALIGN_POT(offset, 8);
         offset = layout.omm_metadata + (VkDeviceSize)leaf_count * 8;
      }
   }
   layout.size = ALIGN_POT(offset, 128);
   return layout;
}

static bool
nvk_bvh_has_opacity_micromaps(const VkAccelerationStructureBuildGeometryInfoKHR *info)
{
   if (info->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_UPDATE_BIT_EXT)
      return true;
   for (uint32_t i = 0; i < info->geometryCount; i++) {
      const VkAccelerationStructureGeometryKHR *geometry = info->pGeometries ?
         &info->pGeometries[i] : info->ppGeometries[i];
      if (geometry->geometryType != VK_GEOMETRY_TYPE_TRIANGLES_KHR)
         continue;
      const VkAccelerationStructureTrianglesOpacityMicromapEXT *omm =
         vk_find_struct_const(geometry->geometry.triangles.pNext,
                              ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT);
      if (omm)
         return true;
   }
   return false;
}

static void
nvk_bvh_get_build_config(VkDevice device,
                         struct vk_acceleration_structure_build_state *state)
{
   if (state->config.updateable)
      state->config.build_flags |= VK_BUILD_FLAG_INACTIVE_BOUNDS;
   state->config.u64_keys = true;
   if (vk_get_as_geometry_type(state->build_info) == VK_GEOMETRY_TYPE_TRIANGLES_KHR)
      state->config.build_flags |= VK_BUILD_FLAG_STORE_VERTEX_IDS;
   if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
      state->config.internal_type = VK_INTERNAL_BUILD_TYPE_LBVH;
}

static VkDeviceSize
nvk_bvh_get_as_size(VkDevice device,
                   const struct vk_acceleration_structure_build_state *state)
{
   return nvk_bvh_layout(vk_get_as_geometry_type(state->build_info),
                         state->leaf_node_count, state->config.updateable).size;
}

static VkDeviceSize
nvk_bvh_get_encode_scratch_size(
   VkDevice device, const struct vk_acceleration_structure_build_state *state)
{
   return (VkDeviceSize)MAX2(state->leaf_node_count, 2) * 96 + 64;
}

static uint32_t
nvk_bvh_planner_budget(uint32_t leaf_count, uint32_t max_leaf_size)
{
   if (max_leaf_size != NVK_BVH_PLANNER_DEFAULT_LEAF_SIZE)
      return MAX2(leaf_count, 1);
   const uint64_t count = (uint64_t)MAX2(leaf_count, 1) + 1;
   const uint64_t lower = 2 * (count > 17 ? count - 17 : 0) / 162 + 1;
   const float scaled_count = (float)count * 0x1.450748p-4f;
   return CLAMP((uint64_t)(scaled_count + 8.0f), lower, count - 1);
}

static uint32_t
nvk_bvh_persistent_grid(struct nvk_cmd_buffer *cmd, VkDeviceSize threads,
                        uint32_t block_size)
{
   const struct nv_device_info *info = &nvk_device_physical(nvk_cmd_buffer_device(cmd))->info;
   const uint32_t warps = info->tpc_count * info->mp_per_tpc * info->max_warps_per_mp;
   return MIN2(MAX2(1, DIV_ROUND_UP(threads, block_size)),
               MAX2(1, DIV_ROUND_UP(warps * 32, block_size)));
}

static void
nvk_bvh_dispatch_unaligned(VkCommandBuffer command_buffer, uint32_t x,
                           uint32_t y, uint32_t z)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   const struct nak_shader_info *info = &cmd->state.cs.shader->info;
   nvk_CmdDispatchBase(command_buffer, 0, 0, 0,
                       DIV_ROUND_UP(x, info->cs.local_size[0]),
                       DIV_ROUND_UP(y, info->cs.local_size[1]),
                       DIV_ROUND_UP(z, info->cs.local_size[2]));
}

static void
nvk_bvh_write_buffer_cp(VkCommandBuffer command_buffer, VkDeviceAddress address,
                        void *data, uint32_t size)
{
   const VkDeviceAddressRangeKHR range = { .address = address, .size = size };
   nvk_CmdUpdateMemoryKHR(command_buffer, &range, 0, size, data);
}

static void
nvk_bvh_flush_buffer_write_cp(VkCommandBuffer command_buffer)
{
   vk_bvh_build_barrier_transfer_to_compute(command_buffer);
}

static void
nvk_bvh_fill_buffer_addr(VkCommandBuffer command_buffer, VkDeviceAddress address,
                         VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const VkDeviceAddressRangeKHR range = { .address = address, .size = size };
   vk_meta_fill_memory(&cmd->vk, &dev->meta, &range, 0, data);
}

static void
nvk_bvh_encode(VkCommandBuffer command_buffer, struct vk_device *device,
                struct vk_meta_device *meta,
                const struct vk_acceleration_structure_build_args *build_args,
                struct vk_acceleration_structure_build_state *states,
                uint32_t build_count, bool flushed_compute)
{
   bool any_build = false;
   for (uint32_t index = 0; index < build_count; index++)
      any_build |= states[index].config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE;
   if (!any_build)
      return;
   for (uint32_t pass = 0; pass < 4; pass++) {
      if (pass < 3)
         vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
      for (uint32_t index = 0; index < build_count; index++) {
         const struct vk_acceleration_structure_build_state *state = &states[index];
         if (state->config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
            continue;
         VK_FROM_HANDLE(vk_acceleration_structure, dst,
                        state->build_info->dstAccelerationStructure);
         const VkGeometryTypeKHR type = vk_get_as_geometry_type(state->build_info);
         if (pass == 3 && type != VK_GEOMETRY_TYPE_TRIANGLES_KHR)
            continue;
         const struct nvk_bvh_layout layout = nvk_bvh_layout(
            type, state->leaf_node_count, state->config.updateable);
         const VkDeviceAddress scratch = state->build_info->scratchData.deviceAddress;
         const VkDeviceSize capacity = MAX2(state->leaf_node_count, 2);
         const VkDeviceAddress nodes = ALIGN_POT(
            scratch + state->scratch.sort_buffer_offset[0] + capacity * 24, 16);
         struct nvk_bvh_encode_args args = {
            .ir = scratch + state->scratch.ir_offset,
            .ir_header = scratch + state->scratch.header_offset,
            .dst = vk_acceleration_structure_get_va(dst),
            .mapping = nodes,
            .leaf_map = nodes + capacity * 36,
            .prefix_sums = nodes + capacity * 40,
            .node_offset = layout.nodes,
            .leaf_offset = layout.leaves,
            .metadata_offset = layout.metadata,
            .primitive_data_offset = layout.primitive_data,
            .size = layout.size,
            .refit_data_offset = layout.refit_data,
            .refit_leaf_map_offset = layout.refit_leaf_map,
            .leaf_capacity = state->leaf_node_count,
            .ir_leaf_size = vk_ir_node_size(type, state->config.build_flags),
            .geometry_type = type,
            .pass = pass,
            .build_flags = state->build_info->flags |
                           (type == VK_GEOMETRY_TYPE_TRIANGLES_KHR ? NVK_BVH_ENCODE_TRIANGLE_PACKING : 0) |
                           (nvk_bvh_has_opacity_micromaps(state->build_info) ?
                              NVK_BVH_ENCODE_OPACITY_MICROMAPS : 0),
            .geometry_count = state->build_info->geometryCount,
            .omm_metadata_offset = layout.omm_metadata,
         };
         VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
         const uint32_t groups = pass == 1 ? nvk_bvh_persistent_grid(cmd, capacity * 32, 256) :
            pass == 3 ? MAX2(1, DIV_ROUND_UP((VkDeviceSize)state->leaf_node_count * 16, 256)) :
                        DIV_ROUND_UP(capacity, 256);
         VkResult result = nvk_bvh_cuda_launch(cmd,
            "nvk_bvh_encode_wide", &args, sizeof(args), groups, 256);
         if (result != VK_SUCCESS) {
            vk_command_buffer_set_error(&cmd->vk, result);
            return;
         }
      }
   }
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   for (uint32_t index = 0; index < build_count; index++) {
      const struct vk_acceleration_structure_build_state *state = &states[index];
      if (state->config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE ||
          vk_get_as_geometry_type(state->build_info) != VK_GEOMETRY_TYPE_TRIANGLES_KHR)
         continue;
      VK_FROM_HANDLE(vk_acceleration_structure, dst, state->build_info->dstAccelerationStructure);
      const VkDeviceAddress scratch = state->build_info->scratchData.deviceAddress;
      const uint32_t capacity = MAX2(state->leaf_node_count, 2);
      const uint32_t boxes = MAX2(state->leaf_node_count, 2) - 1;
      const VkDeviceAddress nodes = ALIGN_POT(
         scratch + state->scratch.sort_buffer_offset[0] + (VkDeviceSize)capacity * 24, 16);
      const uint32_t span = MIN2(MAX2(DIV_ROUND_UP(state->leaf_node_count, 32), 32), 97);
      struct nvk_bvh_pack_args args = {
         .ir = scratch + state->scratch.ir_offset,
         .dst = vk_acceleration_structure_get_va(dst),
         .plan = scratch + state->scratch.sort_buffer_offset[0],
         .span_sums = scratch + state->scratch.sort_buffer_offset[0] + (VkDeviceSize)capacity * 16,
         .workspace = nodes + (VkDeviceSize)capacity * 40 + 32,
         .ir_leaf_size = vk_ir_node_size(VK_GEOMETRY_TYPE_TRIANGLES_KHR, state->config.build_flags),
         .leaf_capacity = state->leaf_node_count,
         .span_length = span,
      };
      const char *stages[] = { "nvk_bvh_pack_plan", "nvk_bvh_pack_prefix",
                               "nvk_bvh_pack_emit", "nvk_bvh_pack_patch" };
      for (uint32_t pass = 0; pass < ARRAY_SIZE(stages); pass++) {
         args.pass = pass;
         vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
         const uint32_t block = pass == 1 ? 256 : 128;
         const uint32_t grid = pass == 1 ? 1 : pass == 3 ? MAX2(1, DIV_ROUND_UP(boxes, 128)) :
            MAX2(1, DIV_ROUND_UP(DIV_ROUND_UP(state->leaf_node_count, span), 4));
         VkResult result = nvk_bvh_cuda_launch(cmd, stages[pass], &args, sizeof(args), grid, block);
         if (result != VK_SUCCESS) {
            vk_command_buffer_set_error(&cmd->vk, result);
            return;
         }
      }
   }

}

static VkDeviceSize
nvk_bvh_get_update_scratch_size(
   VkDevice device, const struct vk_acceleration_structure_build_state *state)
{
   return state->scratch.size;
}

static void
nvk_bvh_init_update_scratch(
   VkCommandBuffer command_buffer,
   const struct vk_acceleration_structure_build_state *states,
   uint32_t build_count)
{
   for (uint32_t index = 0; index < build_count; index++) {
      const struct vk_acceleration_structure_build_state *state = &states[index];
      if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;
      struct vk_ir_header header = {
         .min_bounds = {0x7fffffff, 0x7fffffff, 0x7fffffff},
         .max_bounds = {0x80000000, 0x80000000, 0x80000000},
         .dispatch_size_y = 1,
         .dispatch_size_z = 1,
      };
      nvk_bvh_write_buffer_cp(command_buffer,
         state->build_info->scratchData.deviceAddress + state->scratch.header_offset,
         &header, sizeof(header));
   }
}

static void
nvk_bvh_update(VkCommandBuffer command_buffer, struct vk_device *device,
               struct vk_meta_device *meta,
               const struct vk_acceleration_structure_build_args *build_args,
               struct vk_acceleration_structure_build_state *states,
               uint32_t build_count)
{
   bool any_update = false;
   for (uint32_t index = 0; index < build_count; index++)
      any_update |= states[index].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE;
   if (!any_update)
      return;

   bool need_copy = false;
   for (uint32_t index = 0; index < build_count; index++) {
      const struct vk_acceleration_structure_build_state *state = &states[index];
      need_copy |= state->config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE &&
                   state->build_info->srcAccelerationStructure !=
                   state->build_info->dstAccelerationStructure;
   }
   if (need_copy) {
      vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
      for (uint32_t index = 0; index < build_count; index++) {
         const struct vk_acceleration_structure_build_state *state = &states[index];
         if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE ||
             state->build_info->srcAccelerationStructure ==
             state->build_info->dstAccelerationStructure)
            continue;
         VK_FROM_HANDLE(vk_acceleration_structure, src,
                        state->build_info->srcAccelerationStructure);
         VK_FROM_HANDLE(vk_acceleration_structure, dst,
                        state->build_info->dstAccelerationStructure);
         struct nvk_bvh_copy_args copy = {
            .src = vk_acceleration_structure_get_va(src),
            .dst = vk_acceleration_structure_get_va(dst),
         };
         VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
         VkResult result = nvk_bvh_cuda_launch(cmd, "nvk_bvh_copy", &copy, sizeof(copy), 64, 128);
         if (result != VK_SUCCESS) {
            vk_command_buffer_set_error(&cmd->vk, result);
            return;
         }
      }
   }
   vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
   VkResult result = vk_build_update_leaf_nodes(command_buffer, device, meta,
                                                build_args, states, build_count);
   if (result != VK_SUCCESS) {
      VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   for (uint32_t pass = 0; pass < 2; pass++) {
      vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
      for (uint32_t index = 0; index < build_count; index++) {
         const struct vk_acceleration_structure_build_state *state = &states[index];
         if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
            continue;
         VK_FROM_HANDLE(vk_acceleration_structure, dst,
                        state->build_info->dstAccelerationStructure);
         const VkDeviceAddress scratch = state->build_info->scratchData.deviceAddress;
         const struct nvk_bvh_refit_args args = {
            .ir = scratch + state->scratch.ir_offset,
            .dst = vk_acceleration_structure_get_va(dst),
            .counter = scratch + state->scratch.header_offset,
            .ir_leaf_size = vk_ir_node_size(vk_get_as_geometry_type(state->build_info),
                                             state->config.build_flags),
            .pass = pass,
         };
         VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
         VkResult result = nvk_bvh_cuda_launch(cmd,
            "nvk_bvh_refit", &args, sizeof(args),
            pass == 0 ? 1 : MAX2(1, MIN2(DIV_ROUND_UP(state->leaf_node_count, 128), 256)), 128);
         if (result != VK_SUCCESS) {
            vk_command_buffer_set_error(&cmd->vk, result);
            return;
         }
      }
   }
}

static void
nvk_bvh_build_encode(VkCommandBuffer command_buffer, struct vk_device *device,
                     struct vk_meta_device *meta,
                     const struct vk_acceleration_structure_build_args *build_args,
                     struct vk_acceleration_structure_build_state *states,
                     uint32_t build_count, bool flushed_compute)
{
   nvk_bvh_encode(command_buffer, device, meta, build_args, states,
                   build_count, flushed_compute);
   nvk_bvh_update(command_buffer, device, meta, build_args, states, build_count);
}

static VkResult
nvk_bvh_build_leaves_cuda(VkCommandBuffer command_buffer, struct vk_device *device,
                          struct vk_acceleration_structure_build_state *states,
                          uint32_t build_count, bool update)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   _Static_assert(sizeof(struct nvk_bvh_leaf_geometry) == sizeof(struct vk_bvh_geometry_data),
                  "geometry parameter size");
   for (uint32_t index = 0; index < build_count; index++) {
      struct vk_acceleration_structure_build_state *state = &states[index];
      if ((state->config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE) != update)
         continue;
      const VkAccelerationStructureBuildGeometryInfoKHR *info = state->build_info;
      const VkDeviceAddress scratch = info->scratchData.deviceAddress;
      uint64_t leaf_capacity = 0;
      for (uint32_t i = 0; i < info->geometryCount; i++)
         leaf_capacity += state->build_range_infos[i].primitiveCount;
      leaf_capacity = MIN2(leaf_capacity * state->config.leaf_count_multiplier,
                            state->config.leaf_count_limit);
      VK_FROM_HANDLE(vk_acceleration_structure, dst, info->dstAccelerationStructure);
      const struct nvk_bvh_layout layout = nvk_bvh_layout(
         vk_get_as_geometry_type(info), leaf_capacity, state->config.updateable);
      struct nvk_bvh_leaf_args args = {
         .ir = scratch + state->scratch.ir_offset,
         .header = scratch + state->scratch.header_offset,
         .ids = scratch + state->scratch.sort_buffer_offset[0],
         .ir_leaf_size = vk_ir_node_size(vk_get_as_geometry_type(info),
                                          state->config.build_flags),
         .build_flags = state->config.build_flags,
         .output_metadata = layout.omm_metadata ?
            vk_acceleration_structure_get_va(dst) + layout.omm_metadata : 0,
      };
      for (uint32_t geometry_index = 0; geometry_index < info->geometryCount; geometry_index++) {
         const VkAccelerationStructureGeometryKHR *geometry = info->pGeometries ?
            &info->pGeometries[geometry_index] : info->ppGeometries[geometry_index];
         const VkAccelerationStructureBuildRangeInfoKHR *range = &state->build_range_infos[geometry_index];
         if (!range->primitiveCount)
            continue;
         const struct vk_bvh_geometry_data data = vk_fill_geometry_data(
            info->type, state->leaf_node_count, geometry_index, geometry, range);
         memcpy(&args.geom_data, &data, sizeof(data));
         args.micromap_array = 0;
         args.index_buffer = 0;
         args.index_type = VK_INDEX_TYPE_NONE_KHR;
         args.index_stride = 0;
         args.base_triangle = 0;
         args.omm_flags = 0;
         if (geometry->geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
            const VkAccelerationStructureTrianglesOpacityMicromapEXT *omm =
               vk_find_struct_const(geometry->geometry.triangles.pNext,
                                    ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT);
            if (omm) {
               VK_FROM_HANDLE(nvk_micromap, micromap, omm->micromap);
               args.micromap_array = micromap ? nvk_micromap_get_va(micromap) : 0;
               args.index_buffer = omm->indexBuffer.deviceAddress;
               args.index_type = omm->indexType;
               args.index_stride = omm->indexStride;
               args.base_triangle = omm->baseTriangle;
               args.omm_flags = NVK_BVH_LEAF_OMM_ATTACHED;
            } else if (update) {
               args.omm_flags = NVK_BVH_LEAF_OMM_PRESERVE;
            }
         }
         if (state->indirect_address)
            args.geom_data.indirect_range = state->indirect_address +
                                           (uint64_t)geometry_index * state->indirect_stride;
         args.primitive_count = range->primitiveCount;
         const uint32_t leaf_count = MIN2((uint64_t)range->primitiveCount *
                                            state->config.leaf_count_multiplier,
                                         state->config.leaf_count_limit - state->leaf_node_count);
         VkResult result = nvk_bvh_cuda_launch(cmd, "nvk_bvh_leaf", &args, sizeof(args),
                                                MAX2(1, DIV_ROUND_UP(leaf_count, 128)), 128);
         if (result != VK_SUCCESS)
            return result;
         state->leaf_node_count += leaf_count;
      }
   }
   return VK_SUCCESS;
}

static VkResult
nvk_bvh_build_internal_cuda(VkCommandBuffer command_buffer, struct vk_device *device,
                             struct vk_acceleration_structure_build_state *states,
                             uint32_t build_count)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   for (uint32_t index = 0; index < build_count; index++) {
      const struct vk_acceleration_structure_build_state *state = &states[index];
      if (state->config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;
      const VkDeviceAddress scratch = state->build_info->scratchData.deviceAddress;
      struct nvk_bvh_build_cuda_args args = {
         .ir = scratch + state->scratch.ir_offset,
         .ir_header = scratch + state->scratch.header_offset,
         .input_ids = scratch + state->scratch.sort_buffer_offset[0],
         .src_keys = scratch + state->scratch.sort_buffer_offset[0],
         .dst_keys = scratch + state->scratch.sort_buffer_offset[1],
         .node_info = scratch + state->scratch.lbvh_node_offset,
         .leaf_count = state->leaf_node_count,
         .ir_leaf_size = vk_ir_node_size(vk_get_as_geometry_type(state->build_info),
                                          state->config.build_flags),
         .internal_node_base = state->scratch.internal_node_offset - state->scratch.ir_offset,
         .build_flags = state->config.build_flags,
      };
      VkResult result = nvk_bvh_cuda_launch(cmd,
         "nvk_bvh_build_morton_native", &args, sizeof(args),
         MAX2(1, DIV_ROUND_UP(args.leaf_count, 256)), 256);
      if (result != VK_SUCCESS)
         return result;
      vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
      const bool small = state->build_info->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR &&
                         args.leaf_count <= 16384;
      const uint32_t tile_keys = small ? 256 : 3520;
      SWAP(args.src_keys, args.dst_keys);
      result = nvk_bvh_cuda_launch(cmd,
         small ? "nvk_bvh_build_radix_tiles_small_native" : "nvk_bvh_build_radix_tiles_native",
         &args, sizeof(args), MAX2(1, DIV_ROUND_UP(args.leaf_count, tile_keys)),
         small ? 64 : 320);
      if (result != VK_SUCCESS)
         return result;
      for (uint32_t width = tile_keys; width < args.leaf_count; width *= 2) {
         vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
         args.merge_width = width;
         SWAP(args.src_keys, args.dst_keys);
         result = nvk_bvh_cuda_launch(cmd,
            "nvk_bvh_build_merge_native", &args, sizeof(args),
            MAX2(1, DIV_ROUND_UP(args.leaf_count, 256)), 256);
         if (result != VK_SUCCESS)
            return result;
      }
      args.src_keys = args.dst_keys;
      const VkDeviceSize capacity = MAX2(args.leaf_count, 2);
      args.dst_keys = ALIGN_POT(scratch + state->scratch.sort_buffer_offset[0] + capacity * 24, 16);
      args.node_info = args.dst_keys + capacity * 32;
      args.input_ids = args.dst_keys + capacity * 40;
      const char *stages[] = {
         "nvk_bvh_build_aac_init_native", "nvk_bvh_build_aac_native",
         "nvk_bvh_build_aac_ir_init_native", "nvk_bvh_build_aac_ir_native",
      };
      for (uint32_t stage = 0; stage < ARRAY_SIZE(stages); stage++) {
         vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
         result = nvk_bvh_cuda_launch(cmd, stages[stage], &args, sizeof(args),
            MAX2(1, DIV_ROUND_UP(args.leaf_count, 256)), 256);
         if (result != VK_SUCCESS)
            return result;
      }
      const VkGeometryTypeKHR type = vk_get_as_geometry_type(state->build_info);
      const VkBuildAccelerationStructureFlagsKHR omm_update_flags =
         VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_UPDATE_BIT_EXT |
         VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_DATA_UPDATE_BIT_EXT;
      struct nvk_bvh_planner_args planner = {
         .ir = args.ir,
         .ir_header = args.ir_header,
         .nodes = args.dst_keys,
         .pair_to_ir = args.node_info,
         .workspace = args.dst_keys + capacity * 40,
         .capacity = capacity,
         .ir_leaf_size = args.ir_leaf_size,
         .branching_factor = NVK_BVH_PLANNER_DEFAULT_BRANCHING,
         .max_leaf_size = type != VK_GEOMETRY_TYPE_TRIANGLES_KHR ||
                          (state->build_info->flags & omm_update_flags) ? 1 :
                          nvk_bvh_has_opacity_micromaps(state->build_info) ? 7 :
                          NVK_BVH_PLANNER_DEFAULT_LEAF_SIZE,
         .sah_node_cost = NVK_BVH_PLANNER_DEFAULT_NODE_COST,
         .sah_primitive_cost = NVK_BVH_PLANNER_DEFAULT_TRIANGLE_COST,
      };
      planner.max_wide_nodes = nvk_bvh_planner_budget(args.leaf_count,
                                                     planner.max_leaf_size);
      const char *planner_stages[] = {
         "nvk_bvh_planner_init", "nvk_bvh_planner_exec",
         "nvk_bvh_planner_init", "nvk_bvh_planner_squeeze",
         "nvk_bvh_planner_bounds_init", "nvk_bvh_planner_bounds_parents",
         "nvk_bvh_planner_bounds",
      };
      const uint32_t planner_blocks[] = { 128, 512, 128, 64, 128, 128, 128 };
      for (uint32_t stage = 0; stage < ARRAY_SIZE(planner_stages); stage++) {
         planner.flags = stage == 2 ? NVK_BVH_PLANNER_INIT_SQUEEZE : 0;
         uint32_t grid = DIV_ROUND_UP(capacity, planner_blocks[stage]);
         if (stage == 1 || stage == 3)
            grid = nvk_bvh_persistent_grid(cmd, capacity * 2, planner_blocks[stage]);
         vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
         result = nvk_bvh_cuda_launch(cmd, planner_stages[stage], &planner,
                                      sizeof(planner), grid, planner_blocks[stage]);
         if (result != VK_SUCCESS)
            return result;
      }
   }
   return VK_SUCCESS;
}

static const struct vk_acceleration_structure_build_ops nvk_bvh_build_ops = {
   .build_leaves = nvk_bvh_build_leaves_cuda,
   .build_internal = nvk_bvh_build_internal_cuda,
   .get_build_config = nvk_bvh_get_build_config,
   .get_as_size = nvk_bvh_get_as_size,
   .get_encode_scratch_size = nvk_bvh_get_encode_scratch_size,
   .get_update_scratch_size = nvk_bvh_get_update_scratch_size,
   .init_update_scratch = nvk_bvh_init_update_scratch,
   .encode = nvk_bvh_build_encode,
};

void
nvk_device_init_accel_struct(struct nvk_device *dev)
{
   dev->vk.as_build_ops = &nvk_bvh_build_ops;
   dev->vk.cmd_dispatch_unaligned = nvk_bvh_dispatch_unaligned;
   dev->vk.cmd_fill_buffer_addr = nvk_bvh_fill_buffer_addr;
   dev->vk.write_buffer_cp = nvk_bvh_write_buffer_cp;
   dev->vk.flush_buffer_write_cp = nvk_bvh_flush_buffer_write_cp;
}

VKAPI_ATTR void VKAPI_CALL
nvk_GetAccelerationStructureBuildSizesKHR(
   VkDevice device, VkAccelerationStructureBuildTypeKHR build_type,
   const VkAccelerationStructureBuildGeometryInfoKHR *info,
   const uint32_t *max_primitive_counts,
   VkAccelerationStructureBuildSizesInfoKHR *sizes)
{
   const struct vk_acceleration_structure_build_args args = nvk_bvh_build_args();
   vk_get_as_build_sizes(device, build_type, info, max_primitive_counts,
                         sizes, &args);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer command_buffer, uint32_t count,
   const VkAccelerationStructureBuildGeometryInfoKHR *infos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ranges)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   const struct vk_acceleration_structure_build_args args = nvk_bvh_build_args();
   struct nvk_compute_state saved;
   nvk_bvh_meta_begin(cmd, &saved);
   vk_cmd_build_acceleration_structures(command_buffer, &dev->vk, &dev->meta,
                                        count, infos, ranges, &args);
   nvk_bvh_meta_end(cmd, &saved);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdBuildAccelerationStructuresIndirectKHR(
   VkCommandBuffer command_buffer, uint32_t count,
   const VkAccelerationStructureBuildGeometryInfoKHR *infos,
   const VkDeviceAddress *addresses, const uint32_t *strides,
   const uint32_t *const *max_primitive_counts)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   VkAccelerationStructureBuildRangeInfoKHR **ranges = calloc(count, sizeof(*ranges));
   if (!ranges) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   bool allocation_failed = false;
   for (uint32_t index = 0; index < count; index++) {
      const uint32_t geometry_count = infos[index].geometryCount;
      if (geometry_count == 0)
         continue;
      ranges[index] = calloc(geometry_count, sizeof(*ranges[index]));
      if (!ranges[index]) {
         allocation_failed = true;
         break;
      }
      for (uint32_t geometry = 0; geometry < geometry_count; geometry++)
         ranges[index][geometry].primitiveCount = max_primitive_counts[index][geometry];
   }
   if (allocation_failed) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
   } else {
      struct vk_acceleration_structure_build_args args = nvk_bvh_build_args();
      args.indirect_addresses = addresses;
      args.indirect_strides = strides;
      struct nvk_compute_state saved;
      nvk_bvh_meta_begin(cmd, &saved);
      vk_cmd_build_acceleration_structures(command_buffer, &dev->vk, &dev->meta,
         count, infos, (const VkAccelerationStructureBuildRangeInfoKHR *const *)ranges,
         &args);
      nvk_bvh_meta_end(cmd, &saved);
   }
   for (uint32_t index = 0; index < count; index++)
      free(ranges[index]);
   free(ranges);
}

static void
nvk_bvh_copy(VkCommandBuffer command_buffer, struct nvk_bvh_copy_args args)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, command_buffer);
   struct nvk_device *dev = nvk_cmd_buffer_device(cmd);
   struct nvk_compute_state saved;
   nvk_bvh_meta_begin(cmd, &saved);
   vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
   VkResult result = nvk_bvh_cuda_launch(cmd, "nvk_bvh_copy", &args, sizeof(args), 64, 128);
   if (result == VK_SUCCESS && (args.mode == 0 || args.mode == 2)) {
      args.field = args.mode == 2;
      args.mode = 4;
      vk_bvh_build_barrier_compute_to_compute(command_buffer, false);
      result = nvk_bvh_cuda_launch(cmd, "nvk_bvh_copy", &args, sizeof(args), 64, 128);
   }
   if (result != VK_SUCCESS)
      vk_command_buffer_set_error(&cmd->vk, result);
   nvk_bvh_meta_end(cmd, &saved);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer command_buffer, const VkCopyAccelerationStructureInfoKHR *info)
{
   VK_FROM_HANDLE(vk_acceleration_structure, src, info->src);
   VK_FROM_HANDLE(vk_acceleration_structure, dst, info->dst);
   nvk_bvh_copy(command_buffer, (struct nvk_bvh_copy_args) {
      .src = vk_acceleration_structure_get_va(src),
      .dst = vk_acceleration_structure_get_va(dst),
      .size = src->size,
   });
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdCopyAccelerationStructureToMemoryKHR(
   VkCommandBuffer command_buffer,
   const VkCopyAccelerationStructureToMemoryInfoKHR *info)
{
   VK_FROM_HANDLE(vk_acceleration_structure, src, info->src);
   nvk_bvh_copy(command_buffer, (struct nvk_bvh_copy_args) {
      .src = vk_acceleration_structure_get_va(src),
      .dst = info->dst.deviceAddress,
      .mode = 1,
   });
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdCopyMemoryToAccelerationStructureKHR(
   VkCommandBuffer command_buffer,
   const VkCopyMemoryToAccelerationStructureInfoKHR *info)
{
   VK_FROM_HANDLE(vk_acceleration_structure, dst, info->dst);
   nvk_bvh_copy(command_buffer, (struct nvk_bvh_copy_args) {
      .src = info->src.deviceAddress,
      .dst = vk_acceleration_structure_get_va(dst),
      .mode = 2,
   });
}

VKAPI_ATTR void VKAPI_CALL
nvk_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice device, const VkAccelerationStructureVersionInfoKHR *version,
   VkAccelerationStructureCompatibilityKHR *compatibility)
{
   const uint32_t uuid[4] = {
      NVK_BVH_COMPAT_UUID_0, NVK_BVH_COMPAT_UUID_1,
      NVK_BVH_COMPAT_UUID_2, NVK_BVH_COMPAT_UUID_3,
   };
   *compatibility = memcmp(version->pVersionData, uuid, 16) == 0 &&
                    memcmp(version->pVersionData + 16, uuid, 16) == 0 ?
      VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR :
      VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdWriteAccelerationStructuresPropertiesKHR(
   VkCommandBuffer command_buffer, uint32_t count,
   const VkAccelerationStructureKHR *structures, VkQueryType type,
   VkQueryPool query_pool, uint32_t first_query)
{
   VK_FROM_HANDLE(nvk_query_pool, pool, query_pool);
   uint32_t field;
   switch (type) {
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
      field = NVK_BVH_COMPACTED_SIZE_OFFSET;
      break;
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR:
      field = NVK_BVH_SERIALIZATION_SIZE_OFFSET;
      break;
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
      field = 8;
      break;
   case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR:
      field = 0xffffffffu;
      break;
   default:
      UNREACHABLE("Unsupported acceleration structure property");
   }
   for (uint32_t index = 0; index < count; index++) {
      VK_FROM_HANDLE(vk_acceleration_structure, src, structures[index]);
      const uint32_t query = first_query + index;
      const uint64_t base = pool->mem->va->addr;
      const uint64_t available_stride = pool->layout == NVK_QUERY_POOL_LAYOUT_SEPARATE ?
                                         sizeof(uint32_t) : pool->query_stride;
      nvk_bvh_copy(command_buffer, (struct nvk_bvh_copy_args) {
         .src = vk_acceleration_structure_get_va(src),
         .dst = base + pool->reports_start + (uint64_t)query * pool->query_stride,
         .available = base + (uint64_t)query * available_stride,
         .mode = 3,
         .field = field,
      });
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
nvk_GetAccelerationStructureOpaqueCaptureDescriptorDataEXT(
    VkDevice device,
    const VkAccelerationStructureCaptureDescriptorDataInfoEXT *pInfo,
    void *pData)
{
   return VK_SUCCESS;
}
