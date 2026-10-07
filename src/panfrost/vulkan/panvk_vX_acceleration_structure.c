/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "vk_acceleration_structure.h"

#include "bvh/panvk_bvh.h"

#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_dispatch.h"
#include "panvk_cmd_meta.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_instance.h"
#include "panvk_meta.h"
#include "panvk_physical_device.h"
#include "panvk_priv_bo.h"
#include "panvk_ray_tracing.h"

#include "pan_props.h"
#include "util/u_debug.h"

static const uint32_t leaf_spv[] = {
#include "bvh/leaf.spv.h"
};

static const uint32_t encode_spv[] = {
#include "bvh/encode.spv.h"
};

static const uint32_t copy_spv[] = {
#include "bvh/copy.spv.h"
};

static const uint32_t update_spv[] = {
#include "bvh/update.spv.h"
};

static_assert(sizeof(struct panvk_bvh_header) == PANVK_BVH_HEADER_SIZE,
              "unexpected header size");
static_assert(offsetof(struct panvk_bvh_header, serialization_size) ==
                 PANVK_BVH_SERIALIZATION_SIZE_OFFSET,
              "unexpected header layout");
static_assert(offsetof(struct panvk_bvh_header, instance_count) ==
                 PANVK_BVH_INSTANCE_COUNT_OFFSET,
              "unexpected header layout");
static_assert(sizeof(struct panvk_bvh_box_node) == PANVK_BVH_NODE_SIZE,
              "unexpected box node size");
static_assert(sizeof(struct panvk_bvh_triangle_leaf) == PANVK_BVH_NODE_SIZE,
              "unexpected triangle leaf size");
static_assert(sizeof(struct panvk_bvh_aabb_leaf) == PANVK_BVH_NODE_SIZE,
              "unexpected AABB leaf size");
static_assert(sizeof(struct panvk_bvh_instance_leaf) == PANVK_BVH_NODE_SIZE,
              "unexpected instance leaf size");

#define PANVK_BVH_ENCODE_WG_SIZE 64
#define PANVK_BVH_UPDATE_WG_SIZE 64
#define PANVK_BVH_COPY_WG_COUNT  64

static uint32_t
bvh_leaf_base(VkGeometryTypeKHR type, uint32_t leaf_count)
{
   uint32_t box_count = MAX2(leaf_count, 2) - 1;

   return type == VK_GEOMETRY_TYPE_AABBS_KHR ? box_count + leaf_count
                                             : box_count;
}

static VkDeviceSize
bvh_size(VkGeometryTypeKHR type, uint32_t leaf_count)
{
   VkDeviceSize size =
      PANVK_BVH_HEADER_SIZE +
      (VkDeviceSize)(bvh_leaf_base(type, leaf_count) + leaf_count) *
         PANVK_BVH_NODE_SIZE;

   if (type == VK_GEOMETRY_TYPE_INSTANCES_KHR)
      size += leaf_count * sizeof(struct panvk_bvh_instance_extra);

   return size;
}

static struct vk_acceleration_structure_build_args
bvh_build_args(void)
{
   return (struct vk_acceleration_structure_build_args){
      .subgroup_size = pan_subgroup_size(PAN_ARCH),
      .bvh_bounds_offset = offsetof(struct panvk_bvh_header, bounds),
      .morton_sort_workgroup_size = 512,
      .morton_sort_kvs_per_thread = 2,
      .has_update = true,
      .has_tlas_update = true,
   };
}

struct panvk_update_layout {
   VkDeviceSize ready_offset;
   VkDeviceSize bounds_offset;
   VkDeviceSize size;
};

static struct panvk_update_layout
update_layout(const struct vk_acceleration_structure_build_state *state)
{
   const VkGeometryTypeKHR type = vk_get_as_geometry_type(state->build_info);
   const uint32_t boxes = bvh_leaf_base(type, state->leaf_node_count);
   const VkDeviceSize nodes = (VkDeviceSize)boxes + state->leaf_node_count;
   const VkDeviceSize bounds_offset =
      ALIGN_POT((VkDeviceSize)boxes * sizeof(uint32_t), 8);

   return (struct panvk_update_layout){
      .ready_offset = 0,
      .bounds_offset = bounds_offset,
      .size = bounds_offset + nodes * sizeof(vk_aabb),
   };
}

static VkResult
bind_bvh_pipeline(VkCommandBuffer commandBuffer,
                  const struct vk_acceleration_structure_build_args *args,
                  enum panvk_meta_object_key_type key, const uint32_t *spv,
                  uint32_t spv_size, uint32_t push_size, uint32_t flags,
                  VkPipelineLayout *layout)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   VkPipeline pipeline;

   VkResult result = vk_get_bvh_build_pipeline_spv(
      &dev->vk, &dev->meta, (enum vk_meta_object_key_type)key, spv, spv_size,
      push_size, args, flags, &pipeline, false);
   if (result == VK_SUCCESS)
      result = vk_get_bvh_build_pipeline_layout(&dev->vk, &dev->meta,
                                                push_size, layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmdbuf->vk, result);
      return result;
   }

   dev->vk.dispatch_table.CmdBindPipeline(
      commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   return VK_SUCCESS;
}

static VkDeviceSize
get_as_size(VkDevice device,
            const struct vk_acceleration_structure_build_state *state)
{
   const VkGeometryTypeKHR type = vk_get_as_geometry_type(state->build_info);
   VkDeviceSize size = bvh_size(type, state->leaf_node_count);

   if (state->config.updateable)
      size += ((VkDeviceSize)bvh_leaf_base(type, state->leaf_node_count) +
               state->leaf_node_count) *
              sizeof(uint32_t);

   return size;
}

static VkDeviceSize
get_update_scratch_size(VkDevice device,
                        const struct vk_acceleration_structure_build_state *state)
{
   return update_layout(state).size;
}

static void
init_update_scratch(VkCommandBuffer commandBuffer,
                    const struct vk_acceleration_structure_build_state *states,
                    uint32_t build_count)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);

   for (uint32_t i = 0; i < build_count; i++) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;

      const struct panvk_update_layout layout = update_layout(state);
      dev->vk.cmd_fill_buffer_addr(
         commandBuffer,
         state->build_info->scratchData.deviceAddress + layout.ready_offset,
         layout.bounds_offset - layout.ready_offset, 0);
   }
}

static void
dispatch_copy(VkCommandBuffer commandBuffer,
              const struct vk_acceleration_structure_build_args *args,
              uint64_t src, uint64_t dst, uint32_t mode)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);
   const struct panvk_instance *instance =
      to_panvk_instance(phys_dev->vk.instance);
   struct panvk_copy_args push = {
      .src_addr = src,
      .dst_addr = dst,
      .mode = mode,
   };
   VkPipelineLayout layout;

   memcpy(push.uuid, instance->driver_build_sha, VK_UUID_SIZE);
   memcpy(push.uuid + VK_UUID_SIZE / 4, phys_dev->cache_uuid, VK_UUID_SIZE);

   if (bind_bvh_pipeline(commandBuffer, args, PANVK_META_OBJECT_KEY_BVH_COPY,
                         copy_spv, sizeof(copy_spv), sizeof(push), 0,
                         &layout) != VK_SUCCESS)
      return;

   dev->vk.dispatch_table.CmdPushConstants(commandBuffer, layout,
                                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                           sizeof(push), &push);
   dev->vk.dispatch_table.CmdDispatch(commandBuffer, PANVK_BVH_COPY_WG_COUNT,
                                      1, 1);
}

static void
update_as(VkCommandBuffer commandBuffer, struct vk_device *device,
          const struct vk_acceleration_structure_build_args *args,
          struct vk_acceleration_structure_build_state *states,
          uint32_t build_count)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VkPipelineLayout layout;
   bool copied = false;

   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);

   for (uint32_t i = 0; i < build_count; i++) {
      const VkAccelerationStructureBuildGeometryInfoKHR *info =
         states[i].build_info;
      if (states[i].config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE ||
          info->srcAccelerationStructure == info->dstAccelerationStructure)
         continue;

      VK_FROM_HANDLE(vk_acceleration_structure, src,
                     info->srcAccelerationStructure);
      VK_FROM_HANDLE(vk_acceleration_structure, dst,
                     info->dstAccelerationStructure);
      dispatch_copy(commandBuffer, args, vk_acceleration_structure_get_va(src),
                    vk_acceleration_structure_get_va(dst),
                    PANVK_COPY_MODE_COPY);
      copied = true;
   }

   if (copied)
      vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);

   if (bind_bvh_pipeline(commandBuffer, args, PANVK_META_OBJECT_KEY_BVH_UPDATE,
                         update_spv, sizeof(update_spv),
                         sizeof(struct panvk_update_args), 0,
                         &layout) != VK_SUCCESS)
      return;

   for (uint32_t i = 0; i < build_count; i++) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      const VkAccelerationStructureBuildGeometryInfoKHR *info =
         state->build_info;
      if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE ||
          !state->leaf_node_count)
         continue;

      struct pan_ptr geometries = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc,
         info->geometryCount * sizeof(struct vk_bvh_geometry_data),
         sizeof(uint64_t));
      if (!geometries.gpu)
         return;

      struct vk_bvh_geometry_data *geometry_data = geometries.cpu;
      for (uint32_t j = 0; j < info->geometryCount; j++) {
         const VkAccelerationStructureGeometryKHR *geometry =
            info->pGeometries ? &info->pGeometries[j] : info->ppGeometries[j];
         geometry_data[j] = vk_fill_geometry_data(
            info->type, 0, j, geometry, &state->build_range_infos[j]);
      }

      VK_FROM_HANDLE(vk_acceleration_structure, dst,
                     info->dstAccelerationStructure);
      const struct panvk_update_layout scratch = update_layout(state);
      const uint64_t scratch_addr = info->scratchData.deviceAddress;
      const struct panvk_update_args push = {
         .bvh = vk_acceleration_structure_get_va(dst),
         .geometries = geometries.gpu,
         .ready = scratch_addr + scratch.ready_offset,
         .bounds = scratch_addr + scratch.bounds_offset,
         .geometry_type = vk_get_as_geometry_type(info),
      };

      device->dispatch_table.CmdPushConstants(commandBuffer, layout,
                                              VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                              sizeof(push), &push);
      device->dispatch_table.CmdDispatch(
         commandBuffer,
         DIV_ROUND_UP(state->leaf_node_count, PANVK_BVH_UPDATE_WG_SIZE), 1, 1);
   }
}

static uint64_t
encode_scratch_addr(const struct vk_acceleration_structure_build_state *state)
{
   return state->build_info->scratchData.deviceAddress +
          state->scratch.header_offset + sizeof(struct vk_ir_header);
}

static VkDeviceSize
get_encode_scratch_size(VkDevice device,
                        const struct vk_acceleration_structure_build_state *state)
{
   return (VkDeviceSize)MAX2(state->internal_node_count, 1) *
          sizeof(struct panvk_encode_node);
}

static void
get_build_config(VkDevice device,
                 struct vk_acceleration_structure_build_state *state)
{
   if (vk_get_as_geometry_type(state->build_info) ==
          VK_GEOMETRY_TYPE_INSTANCES_KHR &&
       !state->config.updateable && PAN_ARCH == 15 &&
       debug_get_bool_option("PANVK_TLAS_SPLIT", false)) {
      state->config.leaf_count_multiplier = PANVK_BVH_MAX_CHILDREN;
      state->config.leaf_count_limit = 0xffffff;
      state->config.build_flags |= VK_BUILD_FLAG_SPLIT_INSTANCES;
   }
   if (vk_get_as_geometry_type(state->build_info) ==
          VK_GEOMETRY_TYPE_TRIANGLES_KHR &&
       !state->config.updateable) {
      state->config.early_pair_compression = true;
      state->config.late_pair_compression = true;
   }
}

static void
encode_builds(VkCommandBuffer commandBuffer, struct vk_device *device,
              const struct vk_acceleration_structure_build_args *args,
              struct vk_acceleration_structure_build_state *states,
              uint32_t build_count)
{
   VkPipelineLayout layout;
   uint32_t bound_flags = ~0u;

   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);

   for (uint32_t i = 0; i < build_count; i++) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;

      device->cmd_fill_buffer_addr(commandBuffer, encode_scratch_addr(state),
                                   get_encode_scratch_size(NULL, state), 0);
   }

   for (uint32_t pass = PANVK_ENCODE_PASS_COLLAPSE;
        pass <= PANVK_ENCODE_PASS_WRITE; pass++) {
      vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);

      for (uint32_t i = 0; i < build_count; i++) {
         const struct vk_acceleration_structure_build_state *state = &states[i];
         if (state->config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
            continue;

         const uint32_t flags =
            state->config.build_flags & VK_BUILD_FLAG_HAS_QUADS;
         if (flags != bound_flags) {
            if (bind_bvh_pipeline(commandBuffer, args,
                                  PANVK_META_OBJECT_KEY_BVH_ENCODE, encode_spv,
                                  sizeof(encode_spv),
                                  sizeof(struct panvk_encode_args), flags,
                                  &layout) != VK_SUCCESS)
               return;
            bound_flags = flags;
         }

         VK_FROM_HANDLE(vk_acceleration_structure, dst,
                        state->build_info->dstAccelerationStructure);
         VkGeometryTypeKHR type = vk_get_as_geometry_type(state->build_info);
         uint64_t scratch = state->build_info->scratchData.deviceAddress;
         const struct panvk_encode_args push = {
            .intermediate_bvh = scratch + state->scratch.ir_offset,
            .output_bvh = vk_acceleration_structure_get_va(dst),
            .header = scratch + state->scratch.header_offset,
            .nodes = encode_scratch_addr(state),
            .leaf_node_count = state->leaf_node_count,
            .leaf_base = bvh_leaf_base(type, state->leaf_node_count),
            .geometry_type = type,
            .geometry_count = state->build_info->geometryCount,
            .updateable = state->config.updateable,
            .pass = pass,
         };

         device->dispatch_table.CmdPushConstants(commandBuffer, layout,
                                                 VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                                 sizeof(push), &push);
         device->dispatch_table.CmdDispatch(
            commandBuffer,
            DIV_ROUND_UP(MAX2(state->internal_node_count, 1),
                         PANVK_BVH_ENCODE_WG_SIZE),
            1, 1);
      }
   }
}

static void
encode_as(VkCommandBuffer commandBuffer, struct vk_device *device,
          struct vk_meta_device *meta,
          const struct vk_acceleration_structure_build_args *args,
          struct vk_acceleration_structure_build_state *states,
          uint32_t build_count, bool flushed_compute_after_init_update_scratch)
{
   bool any_build = false;
   bool any_update = false;

   for (uint32_t i = 0; i < build_count; i++) {
      if (states[i].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
         any_update = true;
      else
         any_build = true;
   }

   if (any_build)
      encode_builds(commandBuffer, device, args, states, build_count);
   if (any_update)
      update_as(commandBuffer, device, args, states, build_count);
}

static const struct vk_acceleration_structure_build_ops as_build_ops = {
   .leaf_spirv_override = leaf_spv,
   .leaf_spirv_override_size = sizeof(leaf_spv),
   .get_build_config = get_build_config,
   .get_encode_scratch_size = get_encode_scratch_size,
   .get_as_size = get_as_size,
   .get_update_scratch_size = get_update_scratch_size,
   .init_update_scratch = init_update_scratch,
   .encode = encode_as,
};

static void
write_buffer_cp(VkCommandBuffer commandBuffer, VkDeviceAddress addr,
                void *data, uint32_t size)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, PANVK_SUBQUEUE_COMPUTE);
   struct cs_index dst = cs_scratch_reg64(b, 0);
   const uint32_t *words = data;
   const uint32_t word_count = size / sizeof(uint32_t);

   assert(size % sizeof(uint32_t) == 0);

   cs_move64_to(b, dst, addr);
   for (uint32_t i = 0; i < word_count; i += CS_MAX_REG_TUPLE_SIZE) {
      const uint32_t count = MIN2(word_count - i, CS_MAX_REG_TUPLE_SIZE);

      for (uint32_t j = 0; j < count; j++)
         cs_move32_to(b, cs_scratch_reg32(b, 2 + j), words[i + j]);

      cs_store(b, cs_scratch_reg_tuple(b, 2, count), dst,
               BITFIELD_MASK(count), i * sizeof(uint32_t));
   }
}

static void
flush_buffer_write_cp(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   cs_flush_stores(panvk_get_cs_builder(cmdbuf, PANVK_SUBQUEUE_COMPUTE));
}

VkResult
panvk_per_arch(device_reserve_ray_query_slots)(struct panvk_device *dev,
                                               uint32_t slots)
{
   const struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);
   const struct pan_kmod_dev_props *props = &phys_dev->kmod.dev->props;
   VkResult result = VK_SUCCESS;

   if (slots > PANVK_MAX_RAY_QUERY_SLOTS)
      return panvk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "%u ray queries exceed the limit of %u", slots,
                          PANVK_MAX_RAY_QUERY_SLOTS);

   simple_mtx_lock(&dev->ray_query.lock);
   if (slots > dev->ray_query.slots) {
      uint64_t size = (uint64_t)pan_query_core_count(props) *
                      props->max_threads_per_core * slots *
                      PANVK_RAY_QUERY_STATE_SIZE;
      struct panvk_priv_bo *bo;

      if (PANVK_DEBUG(STARTUP))
         mesa_logi("ray query state: %u cores x %u threads x %u slots",
                   pan_query_core_count(props), props->max_threads_per_core,
                   slots);

      result = panvk_priv_bo_create(dev, size, PAN_KMOD_BO_FLAG_NO_MMAP,
                                    VK_SYSTEM_ALLOCATION_SCOPE_DEVICE, &bo);
      if (result == VK_SUCCESS) {
         if (dev->ray_query.bo)
            util_dynarray_append(&dev->ray_query.retired, dev->ray_query.bo);
         p_atomic_set(&dev->ray_query.bo, bo);
         dev->ray_query.slots = slots;
      }
   }
   simple_mtx_unlock(&dev->ray_query.lock);

   return result;
}

void
panvk_per_arch(device_finish_ray_query)(struct panvk_device *dev)
{
   util_dynarray_foreach(&dev->ray_query.retired, struct panvk_priv_bo *, bo)
      panvk_priv_bo_unref(*bo);
   panvk_priv_bo_unref(dev->ray_query.bo);
}

void
panvk_per_arch(device_init_accel_struct)(struct panvk_device *dev)
{
   dev->vk.as_build_ops = &as_build_ops;
   dev->vk.write_buffer_cp = write_buffer_cp;
   dev->vk.flush_buffer_write_cp = flush_buffer_write_cp;
   dev->vk.cmd_dispatch_unaligned = panvk_per_arch(cmd_dispatch_unaligned);
   dev->vk.cmd_fill_buffer_addr = panvk_per_arch(cmd_fill_buffer_addr);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(GetAccelerationStructureBuildSizesKHR)(
   VkDevice device, VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   const struct vk_acceleration_structure_build_args args = bvh_build_args();

   vk_get_as_build_sizes(device, buildType, pBuildInfo, pMaxPrimitiveCounts,
                         pSizeInfo, &args);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBuildAccelerationStructuresKHR)(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct vk_acceleration_structure_build_args args = bvh_build_args();
   struct panvk_cmd_meta_compute_save_ctx save = {0};

   panvk_per_arch(cmd_meta_compute_start)(cmdbuf, &save);
   vk_cmd_build_acceleration_structures(commandBuffer, &dev->vk, &dev->meta,
                                        infoCount, pInfos, ppBuildRangeInfos,
                                        &args);
   panvk_per_arch(cmd_meta_compute_end)(cmdbuf, &save);
}

static void
cmd_copy_as(VkCommandBuffer commandBuffer, uint64_t src, uint64_t dst,
            uint32_t mode)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   const struct vk_acceleration_structure_build_args args = bvh_build_args();
   struct panvk_cmd_meta_compute_save_ctx save = {0};

   panvk_per_arch(cmd_meta_compute_start)(cmdbuf, &save);
   dispatch_copy(commandBuffer, &args, src, dst, mode);
   panvk_per_arch(cmd_meta_compute_end)(cmdbuf, &save);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdCopyAccelerationStructureKHR)(
   VkCommandBuffer commandBuffer, const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(vk_acceleration_structure, src, pInfo->src);
   VK_FROM_HANDLE(vk_acceleration_structure, dst, pInfo->dst);

   cmd_copy_as(commandBuffer, vk_acceleration_structure_get_va(src),
               vk_acceleration_structure_get_va(dst), PANVK_COPY_MODE_COPY);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdCopyAccelerationStructureToMemoryKHR)(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   VK_FROM_HANDLE(vk_acceleration_structure, src, pInfo->src);

   cmd_copy_as(commandBuffer, vk_acceleration_structure_get_va(src),
               pInfo->dst.deviceAddress, PANVK_COPY_MODE_SERIALIZE);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdCopyMemoryToAccelerationStructureKHR)(
   VkCommandBuffer commandBuffer,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(vk_acceleration_structure, dst, pInfo->dst);

   cmd_copy_as(commandBuffer, pInfo->src.deviceAddress,
               vk_acceleration_structure_get_va(dst),
               PANVK_COPY_MODE_DESERIALIZE);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(GetDeviceAccelerationStructureCompatibilityKHR)(
   VkDevice _device, const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   VK_FROM_HANDLE(panvk_device, dev, _device);
   const struct panvk_physical_device *phys_dev =
      to_panvk_physical_device(dev->vk.physical);
   const struct panvk_instance *instance =
      to_panvk_instance(phys_dev->vk.instance);
   bool compat =
      !memcmp(pVersionInfo->pVersionData, instance->driver_build_sha,
              VK_UUID_SIZE) &&
      !memcmp(pVersionInfo->pVersionData + VK_UUID_SIZE, phys_dev->cache_uuid,
              VK_UUID_SIZE);

   *pCompatibility = compat
                        ? VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR
                        : VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}
