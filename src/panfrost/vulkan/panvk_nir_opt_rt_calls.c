#include "panvk_nir_rt_pipeline.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "util/u_dynarray.h"
#include "util/hash_table.h"
#include "util/set.h"

nir_def *
panvk_nir_load_rt_dispatch(nir_builder *b, nir_def *dispatch, unsigned offset,
                           unsigned comps, unsigned bits,
                           unsigned params_sysval_offset)
{
   unsigned size = comps * (bits / 8);
   unsigned relative = UINT32_MAX;
   if (offset >= PANVK_RT_DISPATCH_PARAMS_OFFSET &&
       offset - PANVK_RT_DISPATCH_PARAMS_OFFSET <= PANVK_RT_DISPATCH_PARAMS_SIZE &&
       size <= PANVK_RT_DISPATCH_PARAMS_SIZE - (offset - PANVK_RT_DISPATCH_PARAMS_OFFSET))
      relative = offset - PANVK_RT_DISPATCH_PARAMS_OFFSET;
   else if (offset >= 88 && offset <= 100 && size <= 100 - offset)
      relative = PANVK_RT_DISPATCH_PARAMS_SIZE + offset - 88;
   if (params_sysval_offset && (bits == 32 || bits == 64) &&
        relative != UINT32_MAX &&
       offset % (bits / 8) == 0) {
      return nir_load_push_constant(b, comps, bits, nir_imm_int(b, 0),
          .base = params_sysval_offset + relative,
         .range = size, .align_mul = bits / 8);
   }
   unsigned align = bits == 64 ? 8 : 4;
   return nir_load_global_constant(b, comps, bits,
      nir_iadd_imm(b, dispatch, offset), .align_mul = align,
      .align_offset = offset % align);
}

static const struct {
   uint8_t base, src, comp, bits;
   uint16_t mask;
} rt_args[] = {
   {0, 1, 0, 64, 0x0003},
   {5, 1, 1, 64, 0x000c},
   {2, 1, 2, 64, 0x0030},
   {6, 1, 3, 64, 0x00c0},
   {7, 2, 0, 64, 0x0300},
   {8, 2, 2, 64, 0x0c00},
   {9, 2, 4, 32, 0x1000},
   {10, 2, 5, 32, 0x2000},
   {11, 2, 6, 32, 0x4000},
   {12, 2, 7, 32, 0x8000},
};

#define RT_ARG_COUNT ARRAY_SIZE(rt_args)

static bool
rt_call_barriers(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_rt_call_pan &&
       intr->intrinsic != nir_intrinsic_rt_call_result_pan)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_barrier(b, .execution_scope = SCOPE_NONE, .memory_scope = SCOPE_SUBGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_global);
   b->cursor = nir_after_instr(&intr->instr);
   nir_barrier(b, .execution_scope = SCOPE_NONE, .memory_scope = SCOPE_SUBGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_global);
   return true;
}

bool
panvk_nir_lower_rt_call_barriers(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, rt_call_barriers,
                                     nir_metadata_control_flow, NULL);
}

static bool
rt_lower_tls_arg(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_rt_arg_pan) {
      for (unsigned a = 0; a < 4; a++) {
         if (nir_intrinsic_base(intr) != rt_args[a].base)
            continue;
         b->cursor = nir_before_instr(&intr->instr);
         nir_def *offset = nir_load_rt_arg_pan(b, 32, .base = 13 + a);
         nir_def_replace(&intr->def, nir_iadd(b,
            nir_load_rt_tls_base_pan(b), nir_u2u64(b, offset)));
         return true;
      }
      return false;
   }
   if (intr->intrinsic != nir_intrinsic_rt_call_pan &&
       intr->intrinsic != nir_intrinsic_rt_call_result_pan)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *base = nir_load_rt_tls_base_pan(b);
   nir_def *args[4];
   for (unsigned a = 0; a < 4; a++)
      args[a] = nir_u2u64(b, nir_u2u32(b,
         nir_isub(b, nir_channel(b, intr->src[1].ssa, a), base)));
   nir_src_rewrite(&intr->src[1], nir_vec(b, args, 4));
   unsigned mask = nir_intrinsic_rt_arg_mask(intr);
   nir_intrinsic_set_rt_arg_mask(intr, (mask & ~0xaa) | ((mask & 0xaa) >> 1));
   nir_intrinsic_set_rt_call_flags(intr, nir_intrinsic_rt_call_flags(intr) | 1);
   return true;
}

bool
panvk_nir_lower_rt_tls(nir_shader *nir)
{
   bool progress = nir_shader_intrinsics_pass(nir, rt_lower_tls_arg,
                                             nir_metadata_control_flow, NULL);
   if (progress) {
      nir_opt_copy_prop(nir);
      nir_opt_constant_folding(nir);
      nir_opt_dce(nir);
      nir_validate_shader(nir, "RT TLS-relative arguments");
   }
   return progress;
}

struct rt_launch_state {
   uint8_t masks[MESA_ALL_SHADER_STAGES];
   bool entry;
};

static nir_def *
rt_launch_component(nir_builder *b, const struct rt_launch_state *state,
                    unsigned component)
{
   if (state->entry)
      return nir_channel(b, nir_load_global_invocation_id(b, 32), component);
   return nir_load_rt_arg_pan(b, 32, .base = 17 + component);
}

static bool
rt_lower_launch_id(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct rt_launch_state *state = data;
   if (intr->intrinsic == nir_intrinsic_load_ray_launch_id) {
      b->cursor = nir_before_instr(&intr->instr);
      unsigned mask = nir_def_components_read(&intr->def);
      nir_def *components[3];
      for (unsigned c = 0; c < 3; c++)
         components[c] = mask & BITFIELD_BIT(c) ? rt_launch_component(b, state, c)
                                               : nir_imm_int(b, 0);
      nir_def_replace(&intr->def, nir_vec(b, components, 3));
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_rt_call_pan &&
       intr->intrinsic != nir_intrinsic_rt_call_result_pan)
      return false;
   unsigned launch_mask = state->masks[nir_intrinsic_base(intr)];
   if (!launch_mask)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *args[4];
   unsigned mask = nir_intrinsic_rt_arg_mask(intr);
   for (unsigned c = 0; c < 4; c++) {
      args[c] = nir_channel(b, intr->src[1].ssa, c);
      if (launch_mask & BITFIELD_BIT(c)) {
         args[c] = nir_pack_64_2x32_split(b, nir_u2u32(b, args[c]),
                                         rt_launch_component(b, state, c));
         mask |= BITFIELD_BIT(2 * c + 1);
      }
   }
   nir_src_rewrite(&intr->src[1], nir_vec(b, args, 4));
   nir_intrinsic_set_rt_arg_mask(intr, mask);
   return true;
}

bool
panvk_nir_lower_rt_launch_id(const struct panvk_nir_rt_pipeline_info *info,
                           struct panvk_nir_rt_pipeline_result *result)
{
   if (!info->native || !info->tls)
      return false;
   struct rt_launch_state state = {0};
   bool present[MESA_ALL_SHADER_STAGES] = {0};
   for (unsigned i = 0; i < info->stage_count; i++)
      present[info->stages[i].stage] = true;
   for (unsigned s = MESA_SHADER_RAYGEN; s <= MESA_SHADER_CALLABLE; s++) {
      if (!present[s])
         state.masks[s] = 7;
   }
   bool changed;
   do {
      changed = false;
      for (unsigned i = 0; i < info->stage_count; i++) {
         unsigned s = info->stages[i].stage, mask = state.masks[s];
         nir_foreach_function_impl(impl, result->functions[i]) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic == nir_intrinsic_load_ray_launch_id)
                     mask |= nir_def_components_read(&intr->def);
                  else if (intr->intrinsic == nir_intrinsic_rt_call_pan ||
                           intr->intrinsic == nir_intrinsic_rt_call_result_pan)
                     mask |= state.masks[nir_intrinsic_base(intr)];
               }
            }
         }
         changed |= mask != state.masks[s];
         state.masks[s] = mask;
      }
   } while (changed);
   bool progress = false;
   for (unsigned i = 0; i <= info->stage_count; i++) {
      nir_shader *nir = i == info->stage_count ? result->nir : result->functions[i];
      state.entry = i == info->stage_count;
      progress |= nir_shader_intrinsics_pass(nir, rt_lower_launch_id,
                                             nir_metadata_control_flow, &state);
      nir_opt_copy_prop(nir);
      nir_opt_constant_folding(nir);
      nir_opt_dce(nir);
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      nir_validate_shader(nir, "RT launch arguments");
   }
   return progress;
}

static void
rt_deref_shape(nir_deref_instr *deref, nir_deref_instr *parent)
{
   deref->modes = parent->modes;
   deref->def.bit_size = parent->def.bit_size;
   deref->def.num_components = parent->def.num_components;
   nir_foreach_use(use, &deref->def) {
      nir_instr *instr = nir_src_use_instr(use);
      if (instr->type == nir_instr_type_deref)
         rt_deref_shape(nir_instr_as_deref(instr), deref);
   }
}

static bool
rt_resolve_private_deref(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_deref)
      return false;
   nir_deref_instr *cast = nir_instr_as_deref(instr);
   if (cast->deref_type != nir_deref_type_cast || cast->modes != nir_var_mem_global ||
       cast->cast.ptr_stride || cast->cast.align_mul ||
       nir_deref_instr_has_complex_use(cast, 0))
      return false;
   nir_scalar address = nir_scalar_resolved(cast->parent.ssa, 0);
   if (nir_def_instr(address.def)->type == nir_instr_type_alu &&
       nir_scalar_alu_op(address) == nir_op_u2u64)
      address = nir_scalar_chase_alu_src(address, 0);
   if (address.comp || nir_def_instr(address.def)->type != nir_instr_type_deref)
      return false;
   nir_deref_instr *parent = nir_instr_as_deref(nir_def_instr(address.def));
   if (!(parent->modes & (nir_var_shader_temp | nir_var_function_temp)) ||
       parent->type != cast->type)
      return false;
   rt_deref_shape(cast, parent);
   nir_def_replace(&cast->def, &parent->def);
   return true;
}

bool
panvk_nir_opt_rt_private_io(nir_shader *nir)
{
   bool changed = false, progress;
   do {
      progress = nir_opt_copy_prop(nir);
      progress |= nir_opt_constant_folding(nir);
      progress |= nir_opt_algebraic(nir);
      progress |= nir_opt_dead_cf(nir);
      progress |= nir_opt_remove_phis(nir);
      progress |= nir_shader_instructions_pass(nir, rt_resolve_private_deref,
                                               nir_metadata_control_flow, NULL);
      progress |= nir_opt_deref(nir);
      progress |= nir_lower_vars_to_ssa(nir);
      progress |= nir_opt_dce(nir);
      changed |= progress;
   } while (progress);
   nir_validate_shader(nir, "RT private IO before explicit lowering");
   return changed;
}

static bool
rt_lto_candidate(nir_shader *nir)
{
   if (nir->constant_data_size)
      return false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_call)
               return false;
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_rt_arg_pan)
               continue;
            bool argument = false;
            for (unsigned a = 0; a < RT_ARG_COUNT; a++) {
               argument |= nir_intrinsic_base(intr) == rt_args[a].base &&
                           intr->def.bit_size == rt_args[a].bits;
            }
            if (!argument) {
               nir_foreach_use_including_if(use, &intr->def) {
                  if (nir_src_is_if(use) ||
                      nir_src_use_instr(use)->type != nir_instr_type_intrinsic ||
                      (nir_instr_as_intrinsic(nir_src_use_instr(use))->intrinsic !=
                         nir_intrinsic_rt_return_pan &&
                       nir_instr_as_intrinsic(nir_src_use_instr(use))->intrinsic !=
                         nir_intrinsic_rt_return_value_pan))
                     return false;
               }
            }
         }
      }
   }
   return true;
}

static bool
rt_lto_param(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_rt_frame_size_pan) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_load_param(b, RT_ARG_COUNT));
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_rt_return_pan ||
       intr->intrinsic == nir_intrinsic_rt_return_value_pan) {
      if (intr->intrinsic == nir_intrinsic_rt_return_value_pan) {
         b->cursor = nir_before_instr(&intr->instr);
         nir_bindgen_return(b, intr->src[2].ssa);
      }
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_load_rt_arg_pan)
      return false;
   for (unsigned a = 0; a < RT_ARG_COUNT; a++) {
      if (nir_intrinsic_base(intr) != rt_args[a].base)
         continue;
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_load_param(b, a));
      return true;
   }
   return false;
}

struct rt_lto_state {
   nir_shader *callee;
   nir_function_impl *impl;
   struct set *seen;
   struct hash_table *covered;
   unsigned stage, target, target_count, frame_offset;
   bool needs_frame, frame_reserved;
};

static nir_intrinsic_instr *
rt_lto_return(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_rt_return_pan ||
                intr->intrinsic == nir_intrinsic_rt_return_value_pan)
               return intr;
         }
      }
   }
   return NULL;
}

static bool
rt_lto_inline(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct rt_lto_state *state = data;
   if ((intr->intrinsic != nir_intrinsic_rt_call_pan &&
        intr->intrinsic != nir_intrinsic_rt_call_result_pan) ||
       nir_intrinsic_base(intr) != state->stage ||
       _mesa_set_search(state->seen, intr))
      return false;
   bool conditional = false;
   struct hash_entry *coverage = _mesa_hash_table_search(state->covered, intr);
   unsigned covered = coverage ? (uintptr_t)coverage->data : 0;
   if (state->target) {
      if (nir_src_is_const(intr->src[4])) {
         if (nir_src_as_uint(intr->src[4]) != state->target)
            return false;
      } else {
         conditional = covered + 1 < state->target_count;
      }
   }
   nir_intrinsic_instr *ret = rt_lto_return(b->shader);
   if (state->needs_frame &&
       (!ret || (!state->frame_reserved &&
                 nir_intrinsic_stack_size(ret) > UINT32_MAX - state->callee->scratch_size)))
      return false;
   _mesa_set_add(state->seen, intr);
   nir_def *params[RT_ARG_COUNT + 1];
   b->cursor = nir_before_instr(&intr->instr);
   for (unsigned a = 0; a < RT_ARG_COUNT; a++) {
      nir_def *source = intr->src[rt_args[a].src].ssa;
      if (rt_args[a].src == 2 && rt_args[a].bits == 64) {
         params[a] = nir_pack_64_2x32_split(b,
            nir_channel(b, source, rt_args[a].comp),
            nir_channel(b, source, rt_args[a].comp + 1));
      } else {
         params[a] = nir_channel(b, source, rt_args[a].comp);
      }
   }
   if (state->needs_frame) {
      if (!state->frame_reserved) {
         state->frame_offset = nir_intrinsic_stack_size(ret);
         nir_intrinsic_set_stack_size(ret, state->frame_offset + state->callee->scratch_size);
         state->frame_reserved = true;
      }
      params[0] = nir_iadd_imm(b, nir_load_rt_arg_pan(b, 64, .base = 0), state->frame_offset);
   }
   params[RT_ARG_COUNT] = state->needs_frame
      ? nir_iadd_imm(b, nir_load_rt_frame_size_pan(b), -(int64_t)state->frame_offset)
      : nir_imm_int(b, 0);
   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   if (coverage)
      _mesa_hash_table_remove(state->covered, coverage);
   if (conditional)
      nir_push_if(b, nir_ieq_imm(b, intr->src[4].ssa, state->target));
   nir_def *value = nir_inline_function_impl(b, state->impl, params, remap);
   if (conditional) {
      nir_push_else(b, NULL);
      nir_intrinsic_instr *fallback = nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intr->instr));
      nir_builder_instr_insert(b, &fallback->instr);
      _mesa_set_add(state->seen, fallback);
      _mesa_hash_table_insert(state->covered, fallback, (void *)(uintptr_t)(covered + 1));
      nir_pop_if(b, NULL);
      if (intr->intrinsic == nir_intrinsic_rt_call_result_pan)
         value = nir_if_phi(b, value, &fallback->def);
   }
   if (intr->intrinsic == nir_intrinsic_rt_call_result_pan) {
      assert(value && value->num_components == intr->def.num_components);
      nir_def_rewrite_uses(&intr->def, value);
   }
   _mesa_hash_table_destroy(remap, NULL);
   nir_instr_remove(&intr->instr);
   return true;
}

struct rt_lto_budget {
   uint32_t calls;
   bool partial;
};

static void
rt_lto_measure(nir_shader *nir, const unsigned *counts, const unsigned *candidates,
               struct rt_lto_budget *budget)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_rt_call_pan &&
                intr->intrinsic != nir_intrinsic_rt_call_result_pan)
               continue;
            unsigned stage = nir_intrinsic_base(intr);
            budget->calls += candidates[stage];
            budget->partial = candidates[stage] < counts[stage];
         }
      }
   }
}

static bool
rt_clear_target_identity(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_rt_call_pan &&
       intr->intrinsic != nir_intrinsic_rt_call_result_pan)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[4], nir_imm_int(b, 0));
   return true;
}

bool
panvk_nir_opt_rt_lto(const struct panvk_nir_rt_pipeline_info *info,
                    struct panvk_nir_rt_pipeline_result *result)
{
   if (info->disable_lto || !info->native ||
       (info->flags & (VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR |
                      VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT)))
      return false;
   unsigned counts[MESA_ALL_SHADER_STAGES] = {0};
   unsigned candidates[MESA_ALL_SHADER_STAGES] = {0};
   void *mem_ctx = ralloc_context(NULL);
   bool *eligible = rzalloc_array(mem_ctx, bool, info->stage_count);
   for (unsigned i = 0; i < info->stage_count; i++) {
      unsigned stage = info->stages[i].stage;
      counts[stage]++;
      nir_opt_dce(result->functions[i]);
      eligible[i] = stage != MESA_SHADER_RAYGEN &&
                    rt_lto_candidate(result->functions[i]);
   }
   for (unsigned i = 0; i < info->stage_count; i++) {
      unsigned stage = info->stages[i].stage;
      eligible[i] &= counts[stage] == 1 ||
                     (info->direct_handles &&
                      info->stage_count <= PANVK_RT_NATIVE_SHADER_INDEX_MASK);
      candidates[stage] += eligible[i];
   }
   bool *enabled = rzalloc_array(mem_ctx, bool, info->stage_count + 1);
   bool any_enabled = false;
   bool raygen = counts[MESA_SHADER_RAYGEN] != 0;
   for (unsigned i = 0; i <= info->stage_count; i++) {
      if (i == info->stage_count && raygen)
         continue;
      if (i < info->stage_count && info->stages[i].stage != MESA_SHADER_RAYGEN)
         continue;
      struct rt_lto_budget budget = {0};
      rt_lto_measure(i == info->stage_count ? result->nir : result->functions[i],
                      counts, candidates, &budget);
      for (unsigned c = 0; c < info->stage_count; c++) {
         if (eligible[c])
            rt_lto_measure(result->functions[c], counts, candidates, &budget);
      }
      enabled[i] = budget.calls <= 25 || budget.partial;
      any_enabled |= enabled[i];
   }
   for (unsigned i = 0; i < info->stage_count; i++) {
      if (info->stages[i].stage != MESA_SHADER_RAYGEN)
         enabled[i] = any_enabled;
   }
   struct hash_table *covered = _mesa_pointer_hash_table_create(mem_ctx);
   bool progress = false;
   for (unsigned c = 0; c < info->stage_count; c++) {
      unsigned s = info->stages[c].stage;
      if (!eligible[c] || !any_enabled)
         continue;
      nir_shader *callee = result->functions[c];
      nir_opt_dce(callee);
      nir_shader *copy = nir_shader_clone(mem_ctx, callee);
      nir_intrinsic_instr *ret = rt_lto_return(copy);
      copy->scratch_size = ret ? nir_intrinsic_stack_size(ret) : 0;
      nir_function *func = nir_shader_get_entrypoint(copy)->function;
      bool needs_frame = copy->scratch_size != 0;
      nir_foreach_block(block, func->impl) {
         nir_foreach_instr(instr, block) {
            needs_frame |= instr->type == nir_instr_type_intrinsic &&
               nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_rt_frame_size_pan;
         }
      }
      func->num_params = RT_ARG_COUNT + 1;
      func->params = ralloc_array(func, nir_parameter, RT_ARG_COUNT + 1);
      for (unsigned a = 0; a < RT_ARG_COUNT; a++)
         func->params[a] = (nir_parameter){1, rt_args[a].bits};
      func->params[RT_ARG_COUNT] = (nir_parameter){1, 32};
      nir_shader_intrinsics_pass(copy, rt_lto_param, nir_metadata_control_flow, NULL);
      nir_opt_dce(copy);
      for (unsigned i = 0; i <= info->stage_count; i++) {
         if (i == c || !enabled[i])
            continue;
         if (i < info->stage_count &&
             info->stages[i].stage != MESA_SHADER_RAYGEN &&
             info->stages[i].stage != MESA_SHADER_INTERSECTION &&
             info->stages[i].stage != MESA_SHADER_ANY_HIT)
            continue;
         nir_shader *caller = i == info->stage_count ? result->nir : result->functions[i];
         if (caller->info.float_controls_execution_mode !=
             callee->info.float_controls_execution_mode)
            continue;
         struct rt_lto_state state = {
            .callee = copy, .impl = func->impl, .stage = s, .needs_frame = needs_frame,
            .target = counts[s] == 1 ? 0 : c + 1,
            .target_count = counts[s], .covered = covered,
            .seen = _mesa_pointer_set_create(NULL),
         };
         if (nir_shader_intrinsics_pass(caller, rt_lto_inline, nir_metadata_none, &state)) {
            nir_index_ssa_defs(nir_shader_get_entrypoint(caller));
            nir_opt_copy_prop(caller);
            nir_opt_dce(caller);
            nir_shader_gather_info(caller, nir_shader_get_entrypoint(caller));
            nir_validate_shader(caller, "RT runtime LTO");
            progress = true;
         }
         _mesa_set_destroy(state.seen, NULL);
      }
      ralloc_free(copy);
   }
   for (unsigned i = 0; i <= info->stage_count; i++) {
      nir_shader *nir = i == info->stage_count ? result->nir : result->functions[i];
      nir_shader_intrinsics_pass(nir, rt_clear_target_identity,
                                  nir_metadata_control_flow, NULL);
      nir_opt_copy_prop(nir);
      nir_opt_dce(nir);
   }
   ralloc_free(mem_ctx);
   return progress;
}

struct rt_arg_expr {
   bool valid;
   nir_op op;
   uint8_t variable, constant_bits;
   uint64_t constant;
};

struct rt_arg_summary {
   unsigned calls;
   struct {
      bool constant;
      uint64_t value;
      uint16_t equal;
      struct rt_arg_expr expr;
   } args[RT_ARG_COUNT];
};

static nir_scalar
rt_call_arg(nir_intrinsic_instr *call, unsigned arg)
{
   nir_scalar value = nir_scalar_resolved(call->src[rt_args[arg].src].ssa,
                                      rt_args[arg].comp);
   if (rt_args[arg].bits == 64 && rt_args[arg].src == 2) {
      if (nir_def_instr(value.def)->type != nir_instr_type_alu ||
          nir_scalar_alu_op(value) != nir_op_unpack_64_2x32 || value.comp != 0)
         return (nir_scalar){0};
      nir_scalar high = nir_scalar_resolved(call->src[2].ssa, rt_args[arg].comp + 1);
      if (value.def != high.def || high.comp != 1)
         return (nir_scalar){0};
      value = nir_scalar_chase_alu_src(value, 0);
   }
   return value;
}

static struct rt_arg_expr
rt_arg_expression(nir_scalar value)
{
   if (!value.def || nir_def_instr(value.def)->type != nir_instr_type_alu ||
       value.def->num_components != 1 || !list_is_singular(&value.def->uses))
      return (struct rt_arg_expr){0};
   nir_op op = nir_scalar_alu_op(value);
   switch (op) {
   case nir_op_iadd: case nir_op_isub:
   case nir_op_iand: case nir_op_ior: case nir_op_ixor:
   case nir_op_ishl: case nir_op_ushr: case nir_op_ishr:
      break;
   default:
      return (struct rt_arg_expr){0};
   }
   nir_scalar a = nir_scalar_chase_alu_src(value, 0);
   nir_scalar b = nir_scalar_chase_alu_src(value, 1);
   if (nir_scalar_is_const(a) == nir_scalar_is_const(b))
      return (struct rt_arg_expr){0};
   unsigned variable = nir_scalar_is_const(a) ? 1 : 0;
   nir_scalar dependency = variable ? b : a;
   nir_scalar constant = variable ? a : b;
   if (dependency.def->bit_size != value.def->bit_size)
      return (struct rt_arg_expr){0};
   return (struct rt_arg_expr) {
      .valid = true, .op = op, .variable = variable,
      .constant_bits = constant.def->bit_size,
      .constant = nir_scalar_as_uint(constant),
   };
}

static bool
rt_expr_equal(struct rt_arg_expr a, struct rt_arg_expr b)
{
   return a.valid && b.valid && a.op == b.op && a.variable == b.variable &&
          a.constant_bits == b.constant_bits && a.constant == b.constant;
}

static void
rt_summarize_call(struct rt_arg_summary *summary, nir_intrinsic_instr *call,
                   bool rematerialize)
{
   nir_scalar values[RT_ARG_COUNT];
   for (unsigned i = 0; i < RT_ARG_COUNT; i++)
      values[i] = rt_call_arg(call, i);
   for (unsigned i = 0; i < RT_ARG_COUNT; i++) {
      nir_scalar value = values[i];
      bool constant = value.def && nir_scalar_is_const(value);
      uint64_t bits = constant ? nir_scalar_as_uint(value) : 0;
      uint16_t equal = 0;
      for (unsigned j = 0; j < i; j++) {
         if (value.def && rt_args[i].bits == rt_args[j].bits &&
             value.def == values[j].def && value.comp == values[j].comp)
            equal |= BITFIELD_BIT(j);
      }
      struct rt_arg_expr expr = rematerialize ? rt_arg_expression(value) :
                                                (struct rt_arg_expr){0};
      if (!summary->calls) {
         summary->args[i].constant = constant;
         summary->args[i].value = bits;
         summary->args[i].equal = equal;
         summary->args[i].expr = expr;
      } else {
         summary->args[i].constant &= constant && summary->args[i].value == bits;
         summary->args[i].equal &= equal;
         summary->args[i].expr.valid &= rt_expr_equal(summary->args[i].expr, expr);
      }
   }
   summary->calls++;
}

static bool
rt_rewrite_arg(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_rt_arg_pan)
      return false;
   const struct rt_arg_summary *summary = data;
   if (!summary->calls)
      return false;
   unsigned i;
   for (i = 0; i < RT_ARG_COUNT; i++) {
      if (rt_args[i].base == nir_intrinsic_base(intr))
         break;
   }
   if (i == RT_ARG_COUNT)
      return false;
   unsigned original = i;
   if (summary->args[i].equal)
      i = ffs(summary->args[i].equal) - 1;
   struct rt_arg_expr expr = summary->args[i].expr;
   if (i == original && !summary->args[i].constant && !expr.valid)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value;
   if (summary->args[i].constant) {
      value = nir_imm_intN_t(b, summary->args[i].value, rt_args[i].bits);
   } else {
      value = nir_load_rt_arg_pan(b, rt_args[i].bits, .base = rt_args[i].base);
      if (expr.valid) {
         nir_def *constant = nir_imm_intN_t(b, expr.constant, expr.constant_bits);
         value = nir_build_alu(b, expr.op, expr.variable ? constant : value,
                              expr.variable ? value : constant, NULL, NULL);
      }
   }
   nir_def_replace(&intr->def, value);
   return true;
}

static void
rt_rewrite_call(nir_intrinsic_instr *call, const struct rt_arg_summary *summary,
                unsigned mask)
{
   nir_builder b = nir_builder_at(nir_before_instr(&call->instr));
   nir_def *args[4], *hit[8];
   for (unsigned i = 0; i < 4; i++)
      args[i] = nir_channel(&b, call->src[1].ssa, i);
   for (unsigned i = 0; i < 8; i++)
      hit[i] = nir_channel(&b, call->src[2].ssa, i);
   for (unsigned i = 0; i < RT_ARG_COUNT; i++) {
      nir_def *value = NULL;
      if (!(mask & rt_args[i].mask)) {
         value = nir_imm_intN_t(&b, 0, rt_args[i].bits);
      } else if (summary && summary->args[i].expr.valid) {
         nir_scalar scalar = rt_call_arg(call, i);
         scalar = nir_scalar_chase_alu_src(scalar, summary->args[i].expr.variable);
         value = nir_channel(&b, scalar.def, scalar.comp);
      }
      if (!value)
         continue;
      if (rt_args[i].src == 1) {
         args[rt_args[i].comp] = value;
      } else if (rt_args[i].bits == 64) {
         value = nir_unpack_64_2x32(&b, value);
         hit[rt_args[i].comp] = nir_channel(&b, value, 0);
         hit[rt_args[i].comp + 1] = nir_channel(&b, value, 1);
      } else {
         hit[rt_args[i].comp] = value;
      }
   }
   if (!(mask & 0x8000))
      hit[7] = nir_imm_int(&b, 0);
   nir_src_rewrite(&call->src[1], nir_vec(&b, args, 4));
   nir_src_rewrite(&call->src[2], nir_vec(&b, hit, 8));
   nir_intrinsic_set_rt_arg_mask(call, mask);
   nir_progress(true, b.impl, nir_metadata_control_flow);
}

static bool
rt_opt_arg_identity(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if ((alu->op != nir_op_ixor && alu->op != nir_op_isub) ||
       alu->def.num_components != 1)
      return false;
   nir_scalar s = nir_get_scalar(&alu->def, 0);
   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   nir_scalar c = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   if (!nir_scalar_equal(a, c)) {
      if (!nir_def_is_intrinsic(a.def) || !nir_def_is_intrinsic(c.def))
         return false;
      nir_intrinsic_instr *ai = nir_def_as_intrinsic(a.def);
      nir_intrinsic_instr *ci = nir_def_as_intrinsic(c.def);
      if (ai->intrinsic != nir_intrinsic_load_rt_arg_pan ||
          ci->intrinsic != nir_intrinsic_load_rt_arg_pan ||
          nir_intrinsic_base(ai) != nir_intrinsic_base(ci) || a.comp != c.comp)
         return false;
   }
   b->cursor = nir_before_instr(&alu->instr);
   nir_def_replace(&alu->def, nir_imm_intN_t(b, 0, alu->def.bit_size));
   return true;
}

static void
rt_opt_cleanup(nir_shader *nir, bool canonicalize)
{
   bool progress;
   do {
      progress = nir_opt_copy_prop(nir);
      progress |= nir_opt_constant_folding(nir);
      if (canonicalize)
         progress |= nir_shader_alu_pass(nir, rt_opt_arg_identity,
                                         nir_metadata_control_flow, NULL);
      progress |= nir_opt_dce(nir);
   } while (progress);
}

bool
panvk_nir_opt_rt_calls(const struct panvk_nir_rt_pipeline_info *info,
                      struct panvk_nir_rt_pipeline_result *result)
{
   if (info->flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)
      return false;

   struct rt_arg_summary summaries[MESA_ALL_SHADER_STAGES] = {0};
   bool targets[MESA_ALL_SHADER_STAGES] = {0};
   bool nonleaf[MESA_ALL_SHADER_STAGES] = {0};
   for (unsigned i = 0; i < info->stage_count; i++)
      targets[info->stages[i].stage] = true;
   struct util_dynarray calls;
   util_dynarray_init(&calls, NULL);
   for (unsigned i = 0; i <= info->stage_count; i++) {
      nir_shader *nir = i == info->stage_count ? result->nir : result->functions[i];
      rt_opt_cleanup(nir, false);
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_rt_call_pan &&
                   intr->intrinsic != nir_intrinsic_rt_call_result_pan)
                  continue;
               unsigned stage = nir_intrinsic_base(intr);
               assert(stage < MESA_ALL_SHADER_STAGES);
               if (i < info->stage_count)
                  nonleaf[info->stages[i].stage] = true;
               rt_summarize_call(&summaries[stage], intr, true);
               util_dynarray_append(&calls, intr);
            }
         }
      }
   }
   for (unsigned s = 0; s < MESA_ALL_SHADER_STAGES; s++) {
      if (nonleaf[s]) {
         for (unsigned a = 0; a < RT_ARG_COUNT; a++)
            summaries[s].args[a].equal = 0;
      }
   }
   bool progress = false;
   bool canonicalize[MESA_ALL_SHADER_STAGES] = {0};
   for (unsigned i = 0; i < info->stage_count; i++) {
      bool rewritten = nir_shader_intrinsics_pass(result->functions[i], rt_rewrite_arg,
         nir_metadata_control_flow, &summaries[info->stages[i].stage]);
      canonicalize[info->stages[i].stage] |= rewritten;
      progress |= rewritten;
   }
   util_dynarray_foreach(&calls, nir_intrinsic_instr *, call) {
      unsigned stage = nir_intrinsic_base(*call);
      if (targets[stage])
         rt_rewrite_call(*call, &summaries[stage], nir_intrinsic_rt_arg_mask(*call));
   }

   bool changed;
   do {
      for (unsigned i = 0; i < info->stage_count; i++) {
         unsigned stage = info->stages[i].stage;
         if (canonicalize[stage] && !nonleaf[stage])
            rt_opt_cleanup(result->functions[i], true);
      }
      memset(canonicalize, 0, sizeof(canonicalize));
      memset(summaries, 0, sizeof(summaries));
      util_dynarray_foreach(&calls, nir_intrinsic_instr *, call) {
         unsigned stage = nir_intrinsic_base(*call);
         rt_summarize_call(&summaries[stage], *call, false);
      }
      for (unsigned s = 0; s < MESA_ALL_SHADER_STAGES; s++) {
         for (unsigned a = 0; a < RT_ARG_COUNT; a++) {
            if (nonleaf[s])
               summaries[s].args[a].equal = 0;
            summaries[s].args[a].expr.valid = false;
         }
      }
      changed = false;
      unsigned masks[MESA_ALL_SHADER_STAGES] = {0};
      bool present[MESA_ALL_SHADER_STAGES] = {0};
      for (unsigned i = 0; i < info->stage_count; i++) {
         nir_shader *nir = result->functions[i];
         unsigned stage = info->stages[i].stage;
         present[stage] = true;
         bool rewritten = nir_shader_intrinsics_pass(nir, rt_rewrite_arg,
            nir_metadata_control_flow, &summaries[stage]);
         canonicalize[stage] |= rewritten;
         changed |= rewritten;
         rt_opt_cleanup(nir, false);
         nir_foreach_function_impl(impl, nir) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic != nir_intrinsic_load_rt_arg_pan)
                     continue;
                  for (unsigned a = 0; a < RT_ARG_COUNT; a++) {
                     if (rt_args[a].base == nir_intrinsic_base(intr))
                        masks[stage] |= rt_args[a].mask;
                  }
               }
            }
         }
      }
      util_dynarray_foreach(&calls, nir_intrinsic_instr *, call) {
         unsigned stage = nir_intrinsic_base(*call);
         if (!present[stage])
            continue;
         unsigned old = nir_intrinsic_rt_arg_mask(*call);
         assert((masks[stage] & old) == masks[stage]);
         if (old != masks[stage]) {
            rt_rewrite_call(*call, NULL, masks[stage]);
            changed = true;
         }
      }
      progress |= changed;
   } while (changed);
   rt_opt_cleanup(result->nir, false);
   util_dynarray_fini(&calls);
   return progress;
}
