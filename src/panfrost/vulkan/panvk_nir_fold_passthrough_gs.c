/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "panvk_nir.h"

#include "nir.h"
#include "nir_builder.h"

#define GS_FOLD_MAX_OUTPUTS 32

struct gs_fold_src {
   nir_variable *var;
   uint8_t comp;
   int8_t vertex;
};

struct gs_fold_out {
   nir_variable *var;
   struct gs_fold_src src[4];
   uint8_t mask;
};

struct gs_fold {
   unsigned out_count;
   struct gs_fold_out outs[GS_FOLD_MAX_OUTPUTS];
};

static unsigned
gs_prim_vertices(enum mesa_prim in, enum mesa_prim out)
{
   if (in == MESA_PRIM_POINTS && out == MESA_PRIM_POINTS)
      return 1;
   if (in == MESA_PRIM_LINES && out == MESA_PRIM_LINE_STRIP)
      return 2;
   if (in == MESA_PRIM_TRIANGLES && out == MESA_PRIM_TRIANGLE_STRIP)
      return 3;
   return 0;
}

static bool
simple_var(const nir_variable *var, const struct glsl_type *type)
{
   return glsl_type_is_vector_or_scalar(type) &&
          glsl_get_vector_elements(type) + var->data.location_frac <= 4 &&
          glsl_get_bit_size(type) == 32 && !var->data.compact &&
          !var->data.per_view && !var->data.per_primitive;
}

static struct gs_fold_out *
fold_out(struct gs_fold *fold, nir_variable *var)
{
   for (unsigned i = 0; i < fold->out_count; i++) {
      if (fold->outs[i].var == var)
         return &fold->outs[i];
   }

   return NULL;
}

static bool
fold_input(nir_scalar s, struct gs_fold_src *src)
{
   s = nir_scalar_chase_movs(s);

   nir_intrinsic_instr *load = nir_scalar_as_intrinsic(s);
   if (!load || load->intrinsic != nir_intrinsic_load_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(load->src[0]);
   if (!deref || deref->deref_type != nir_deref_type_array ||
       !nir_src_is_const(deref->arr.index))
      return false;

   nir_deref_instr *parent = nir_deref_instr_parent(deref);
   if (parent->deref_type != nir_deref_type_var ||
       parent->var->data.mode != nir_var_shader_in ||
       !simple_var(parent->var, glsl_get_array_element(parent->var->type)))
      return false;

   src->var = parent->var;
   src->comp = s.comp;
   src->vertex = nir_src_as_uint(deref->arr.index);
   return true;
}

static bool
same_src(const struct gs_fold_src *a, const struct gs_fold_src *b)
{
   return a->var == b->var && a->comp == b->comp;
}

static bool
analyze_gs(const nir_shader *gs, struct gs_fold *fold)
{
   const unsigned vertices =
      gs_prim_vertices(gs->info.gs.input_primitive, gs->info.gs.output_primitive);

   if (vertices == 0 || gs->info.gs.invocations > 1 ||
       gs->info.gs.vertices_out < vertices ||
       (gs->info.gs.active_stream_mask & ~1u) || gs->xfb_info)
      return false;

   nir_foreach_shader_out_variable(var, gs) {
      if (!simple_var(var, var->type) || var->data.stream != 0 ||
          fold->out_count == GS_FOLD_MAX_OUTPUTS)
         return false;

      fold->outs[fold->out_count++] = (struct gs_fold_out){.var = var};
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(gs);
   if (!impl || exec_list_length(&impl->body) != 1)
      return false;

   nir_block *block = nir_start_block(impl);
   unsigned emitted = 0;
   bool ended = false;

   nir_foreach_instr(instr, block) {
      switch (instr->type) {
      case nir_instr_type_alu:
      case nir_instr_type_deref:
      case nir_instr_type_load_const:
      case nir_instr_type_undef:
         continue;
      case nir_instr_type_intrinsic:
         break;
      default:
         return false;
      }

      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_deref: {
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         if (!nir_deref_mode_is(deref, nir_var_shader_in))
            return false;
         break;
      }

      case nir_intrinsic_store_deref: {
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         if (ended || deref->deref_type != nir_deref_type_var)
            return false;

         struct gs_fold_out *out = fold_out(fold, deref->var);
         if (!out)
            return false;

         u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
            nir_scalar s =
               nir_scalar_chase_movs(nir_get_scalar(intr->src[1].ssa, c));
            if (nir_scalar_is_undef(s))
               continue;

            struct gs_fold_src src;
            if (!fold_input(s, &src))
               return false;

            const gl_varying_slot slot = out->var->data.location;
            if (slot != VARYING_SLOT_LAYER && slot != VARYING_SLOT_VIEWPORT &&
                src.vertex != (int)emitted)
               return false;

            if (out->mask & BITFIELD_BIT(c)) {
               if (!same_src(&out->src[c], &src))
                  return false;
            } else {
               out->src[c] = src;
               out->mask |= BITFIELD_BIT(c);
            }
         }
         break;
      }

      case nir_intrinsic_emit_vertex:
      case nir_intrinsic_emit_vertex_with_counter:
         if (ended || nir_intrinsic_stream_id(intr) != 0)
            return false;
         emitted++;
         break;

      case nir_intrinsic_end_primitive:
      case nir_intrinsic_end_primitive_with_counter:
         if (nir_intrinsic_stream_id(intr) != 0 || emitted != vertices)
            return false;
         ended = true;
         break;

      default:
         return false;
      }
   }

   if (emitted != vertices)
      return false;

   for (unsigned i = 0; i < fold->out_count; i++) {
      if (fold->outs[i].var->data.location == VARYING_SLOT_POS &&
          fold->outs[i].mask)
         return true;
   }

   return false;
}

struct producer_outputs {
   nir_scalar vals[VARYING_SLOT_MAX][4];
   BITSET_DECLARE(complex, VARYING_SLOT_MAX);
};

static bool
collect_producer_outputs(nir_function_impl *impl,
                         struct producer_outputs *outs)
{
   nir_block *last = nir_impl_last_block(impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_deref &&
             intr->intrinsic != nir_intrinsic_copy_deref &&
             intr->intrinsic != nir_intrinsic_load_deref)
            continue;

         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         if (!nir_deref_mode_is(deref, nir_var_shader_out))
            continue;

         if (intr->intrinsic != nir_intrinsic_store_deref || block != last)
            return false;

         nir_variable *var = nir_deref_instr_get_variable(deref);
         if (!var || var->data.location >= VARYING_SLOT_MAX)
            return false;

         if (deref->deref_type != nir_deref_type_var ||
             !simple_var(var, var->type)) {
            BITSET_SET(outs->complex, var->data.location);
            continue;
         }

         u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
            outs->vals[var->data.location][var->data.location_frac + c] =
               nir_get_scalar(intr->src[1].ssa, c);
         }
      }
   }

   return true;
}

bool
panvk_nir_fold_passthrough_gs(nir_shader *producer, const nir_shader *gs)
{
   if (producer->info.stage != MESA_SHADER_VERTEX &&
       producer->info.stage != MESA_SHADER_TESS_EVAL)
      return false;

   struct gs_fold *fold = calloc(1, sizeof(*fold));
   struct producer_outputs *outs = calloc(1, sizeof(*outs));
   bool progress = false;

   if (!fold || !outs || !analyze_gs(gs, fold))
      goto out;

   nir_function_impl *impl = nir_shader_get_entrypoint(producer);
   if (!collect_producer_outputs(impl, outs))
      goto out;

   for (unsigned i = 0; i < fold->out_count; i++) {
      u_foreach_bit(c, fold->outs[i].mask) {
         const nir_variable *in = fold->outs[i].src[c].var;
         if (BITSET_TEST(outs->complex, in->data.location))
            goto out;
      }
   }

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_deref &&
             nir_deref_mode_is(nir_src_as_deref(intr->src[0]),
                               nir_var_shader_out))
            nir_instr_remove(instr);
      }
   }

   nir_foreach_shader_out_variable_safe(var, producer)
      exec_node_remove(&var->node);

   nir_builder b =
      nir_builder_at(nir_after_block_before_jump(nir_impl_last_block(impl)));

   for (unsigned i = 0; i < fold->out_count; i++) {
      const struct gs_fold_out *out = &fold->outs[i];
      if (!out->mask || out->var->data.location == VARYING_SLOT_VIEWPORT)
         continue;

      nir_variable *var = nir_variable_clone(out->var, producer);
      var->data.mode = nir_var_shader_out;
      var->data.stream = 0;
      nir_shader_add_variable(producer, var);

      const unsigned num_comps = glsl_get_vector_elements(var->type);
      nir_scalar comps[4];
      nir_def *undef = nir_undef(&b, 1, 32);

      for (unsigned c = 0; c < num_comps; c++) {
         comps[c] = nir_get_scalar(undef, 0);
         if (!(out->mask & BITFIELD_BIT(c)))
            continue;

         const struct gs_fold_src *src = &out->src[c];
         nir_scalar val = outs->vals[src->var->data.location]
                                    [src->var->data.location_frac + src->comp];
         if (val.def && val.def->bit_size == 32)
            comps[c] = val;
      }

      nir_store_var(&b, var, nir_vec_scalars(&b, comps, num_comps),
                    nir_component_mask(num_comps));
   }

   producer->info.next_stage = gs->info.next_stage;
   nir_progress(true, impl, nir_metadata_control_flow);
   nir_opt_dce(producer);
   nir_shader_gather_info(producer, impl);
   progress = true;

out:
   free(outs);
   free(fold);
   return progress;
}
