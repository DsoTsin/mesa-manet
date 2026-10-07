/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "panvk_cmd_buffer.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_shader.h"
#include "panvk_shader_instrumentation.h"

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginShaderInstrumentationARM)(
   VkCommandBuffer commandBuffer, VkShaderInstrumentationARM instrumentation)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   VK_FROM_HANDLE(panvk_shader_instrumentation, instr, instrumentation);

   cmdbuf->state.shader_instr = instr;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndShaderInstrumentationARM)(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   cmdbuf->state.shader_instr = NULL;
}

static VkShaderStageFlags
instrumented_stage(const struct panvk_shader *shader, VkShaderStageFlags stage)
{
   return shader && shader->variants[0].info.instrumented ? stage : 0;
}

static uint64_t
record(struct panvk_cmd_buffer *cmdbuf,
       const VkShaderStageFlags stages[PANVK_SHADER_INSTRUMENTATION_SLOTS])
{
   if (!cmdbuf->state.shader_instr)
      return PAN_SHADER_OOB_ADDRESS;

   return panvk_shader_instrumentation_record(
      to_panvk_device(cmdbuf->vk.base.device), cmdbuf->state.shader_instr,
      stages);
}

void
panvk_per_arch(cmd_instrument_draw)(struct panvk_cmd_buffer *cmdbuf)
{
   const VkShaderStageFlags stages[PANVK_SHADER_INSTRUMENTATION_SLOTS] = {
      instrumented_stage(cmdbuf->state.gfx.vs.shader,
                         VK_SHADER_STAGE_VERTEX_BIT),
      instrumented_stage(cmdbuf->state.gfx.fs.shader,
                         VK_SHADER_STAGE_FRAGMENT_BIT),
   };
   const uint64_t counters = record(cmdbuf, stages);

   if (cmdbuf->state.gfx.sysvals.common.instr_counters == counters)
      return;

   cmdbuf->state.gfx.sysvals.common.instr_counters = counters;
   if (stages[0])
      gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);
   if (stages[1])
      gfx_state_set_dirty(cmdbuf, FS_PUSH_UNIFORMS);
}

void
panvk_per_arch(cmd_instrument_dispatch)(struct panvk_cmd_buffer *cmdbuf)
{
   const VkShaderStageFlags stages[PANVK_SHADER_INSTRUMENTATION_SLOTS] = {
      instrumented_stage(cmdbuf->state.compute.shader,
                         VK_SHADER_STAGE_COMPUTE_BIT),
   };
   const uint64_t counters = record(cmdbuf, stages);

   if (cmdbuf->state.compute.sysvals.common.instr_counters == counters)
      return;

   cmdbuf->state.compute.sysvals.common.instr_counters = counters;
   if (stages[0])
      compute_state_set_dirty(cmdbuf, PUSH_UNIFORMS);
}
