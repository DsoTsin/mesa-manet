/*
 * Copyright © 2024 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "util/format/u_format.h"

#include "vk_blend.h"
#include "vk_format.h"
#include "vk_graphics_state.h"

#include "pan_blend.h"
#include "pan_format.h"

#include "panvk_blend_state.h"

void
panvk_per_arch(blend_fill_rt)(struct pan_blend_rt_state *rt,
                              const struct vk_color_blend_attachment_state *att,
                              enum pipe_format format, uint8_t nr_samples,
                              const float *constants)
{
   rt->format = format;
   rt->nr_samples = nr_samples;
   rt->equation.blend_enable = att->blend_enable;
   rt->equation.is_float = util_format_is_float(format);
   rt->equation.color_mask = att->write_mask;

   rt->equation.rgb_func = vk_blend_op_to_pipe(att->color_blend_op);
   rt->equation.rgb_src_factor =
      vk_blend_factor_to_pipe(att->src_color_blend_factor);
   rt->equation.rgb_dst_factor =
      vk_blend_factor_to_pipe(att->dst_color_blend_factor);
   rt->equation.alpha_func = vk_blend_op_to_pipe(att->alpha_blend_op);
   rt->equation.alpha_src_factor =
      vk_blend_factor_to_pipe(att->src_alpha_blend_factor);
   rt->equation.alpha_dst_factor =
      vk_blend_factor_to_pipe(att->dst_alpha_blend_factor);

   pan_blend_optimize_equation(&rt->equation, format, constants);
}

bool
panvk_per_arch(blend_skips_rt)(bool logicop_enable,
                               enum pipe_logicop logicop_func,
                               uint8_t color_write_enables, uint8_t rt_idx,
                               VkFormat format, uint8_t write_mask)
{
   if (!(color_write_enables & BITFIELD_BIT(rt_idx)))
      return true;

   if (format == VK_FORMAT_UNDEFINED || !write_mask)
      return true;

   enum pipe_format pfmt = vk_format_to_pipe_format(format);
   return logicop_enable && logicop_func == PIPE_LOGICOP_NOOP &&
          !(util_format_is_float(pfmt) || util_format_is_srgb(pfmt));
}

bool
panvk_per_arch(blend_needs_shader)(const struct pan_blend_state *state,
                                   unsigned rt_idx,
                                   unsigned *ff_blend_constant)
{
   const struct pan_blend_rt_state *rt = &state->rts[rt_idx];

   /* LogicOp requires a blend shader */
   if (state->logicop_enable)
      return true;

   /* alpha-to-one always requires a blend shader */
   if (state->alpha_to_one)
      return true;

   /* If the output is opaque, we don't need a blend shader, no matter the
    * format.
    */
   if (pan_blend_is_opaque(rt->equation))
      return false;

   /* Not all formats can be blended by fixed-function hardware */
   if (!GENX(pan_format_supports_hw_blend)(rt->format))
      return true;

   bool supports_2src = pan_blend_supports_2src(PAN_ARCH);
   if (!pan_blend_can_fixed_function(PAN_ARCH, rt->equation, supports_2src))
      return true;

   unsigned constant_mask = pan_blend_constant_mask(rt->equation);

   /* v6 doesn't support blend constants in FF blend equations. */
   if (constant_mask && PAN_ARCH == 6)
      return true;

   if (!pan_blend_is_homogenous_constant(constant_mask, state->constants))
      return true;

   /* v7+ only uses the constant from RT 0. If we're not RT0, all previous
    * RTs using FF with a blend constant need to have the same constant,
    * otherwise we need a blend shader.
    */
   unsigned blend_const = ~0;
   if (constant_mask) {
      const float blend_const_f =
         pan_blend_get_constant(constant_mask, state->constants);
      blend_const = pan_pack_blend_constant(rt->format, blend_const_f);

      if (*ff_blend_constant != ~0 && blend_const != *ff_blend_constant)
         return true;
   }

   /* Update the fixed function blend constant, if we use it. */
   if (blend_const != ~0)
      *ff_blend_constant = blend_const;

   return false;
}

void
panvk_per_arch(blend_static_key_init)(
   struct panvk_blend_static_key *key,
   const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));

   if (!state || !state->cb || !state->rp || !state->cal || !state->ms ||
       !vk_render_pass_state_has_attachment_info(state->rp))
      return;

   static const enum mesa_vk_dynamic_graphics_state deps[] = {
      MESA_VK_DYNAMIC_CB_LOGIC_OP_ENABLE,
      MESA_VK_DYNAMIC_CB_LOGIC_OP,
      MESA_VK_DYNAMIC_CB_ATTACHMENT_COUNT,
      MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES,
      MESA_VK_DYNAMIC_CB_BLEND_ENABLES,
      MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS,
      MESA_VK_DYNAMIC_CB_WRITE_MASKS,
      MESA_VK_DYNAMIC_CB_BLEND_ADVANCED,
      MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE,
      MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(deps); i++) {
      if (BITSET_TEST(state->dynamic, deps[i]))
         return;
   }

   const struct vk_color_blend_state *cb = state->cb;
   key->valid = true;
   key->alpha_to_one = state->ms->alpha_to_one_enable;
   key->logic_op_enable = cb->logic_op_enable;
   key->logic_op = cb->logic_op_enable ? cb->logic_op : 0;
   key->rt_count = MIN2(state->rp->color_attachment_count, MESA_VK_MAX_COLOR_ATTACHMENTS);
   key->color_write_enables = cb->color_write_enables;

   for (uint8_t i = 0; i < key->rt_count; i++) {
      const struct vk_color_blend_attachment_state *att = &cb->attachments[i];
      key->color_map[i] = state->cal->color_map[i];
      key->formats[i] = state->rp->color_attachment_formats[i];
      key->attachments[i].blend_enable = att->blend_enable;
      key->attachments[i].src_color_blend_factor = att->src_color_blend_factor;
      key->attachments[i].dst_color_blend_factor = att->dst_color_blend_factor;
      key->attachments[i].src_alpha_blend_factor = att->src_alpha_blend_factor;
      key->attachments[i].dst_alpha_blend_factor = att->dst_alpha_blend_factor;
      key->attachments[i].write_mask = att->write_mask;
      key->attachments[i].color_blend_op = att->color_blend_op;
      key->attachments[i].alpha_blend_op = att->alpha_blend_op;
   }
}

uint8_t
panvk_per_arch(blend_fixed_function_locations)(
   const struct panvk_blend_static_key *key)
{
   if (!key->valid)
      return 0;

   struct pan_blend_state bs = {
      .alpha_to_one = key->alpha_to_one,
      .logicop_enable = key->logic_op_enable,
      .logicop_func = vk_logic_op_to_pipe(key->logic_op),
      .rt_count = key->rt_count,
   };

   uint8_t mask = 0;
   for (uint8_t i = 0; i < key->rt_count; i++) {
      uint8_t loc = key->color_map[i];
      if (loc >= MESA_VK_MAX_COLOR_ATTACHMENTS)
         continue;

      if (panvk_per_arch(blend_skips_rt)(
             bs.logicop_enable, bs.logicop_func, key->color_write_enables, i,
             key->formats[i], key->attachments[i].write_mask)) {
         mask |= BITFIELD_BIT(loc);
         continue;
      }

      struct pan_blend_rt_state *rt = &bs.rts[i];
      panvk_per_arch(blend_fill_rt)(
         rt, &key->attachments[i], vk_format_to_pipe_format(key->formats[i]),
         1, NULL);
      if (pan_blend_constant_mask(rt->equation))
         continue;

      unsigned ff_blend_constant = ~0;
      if (!panvk_per_arch(blend_needs_shader)(&bs, i, &ff_blend_constant))
         mask |= BITFIELD_BIT(loc);
   }

   return mask;
}
