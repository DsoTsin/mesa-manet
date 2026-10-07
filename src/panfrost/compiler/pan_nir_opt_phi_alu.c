/*
 * SPDX-License-Identifier: MIT
 */

#include "nir_builder.h"
#include "pan_nir.h"

static bool
src_available_in(nir_src *src, nir_block *block)
{
   return nir_block_dominates(nir_def_block(src->ssa), block);
}

static bool
alu_src_is_simple(const nir_alu_instr *alu, unsigned s)
{
   unsigned comps = nir_ssa_alu_instr_src_components(alu, s);
   if (alu->src[s].src.ssa->num_components != comps)
      return false;

   for (unsigned c = 0; c < comps; c++) {
      if (alu->src[s].swizzle[c] != c)
         return false;
   }

   return true;
}

static bool
const_is_float(nir_const_value v, unsigned bit_size, double f, bool any_sign)
{
   double x = nir_const_value_as_float(v, bit_size);
   if (any_sign)
      return x == f;
   return x == f && signbit(x) == signbit(f);
}

static bool
const_is_uint(nir_const_value v, unsigned bit_size, uint64_t u)
{
   return nir_const_value_as_uint(v, bit_size) ==
          (u & BITFIELD64_MASK(bit_size));
}

enum fold_kind {
   FOLD_NONE,
   FOLD_SRC,
   FOLD_CONST,
   FOLD_EVAL,
};

struct fold {
   enum fold_kind kind;
   unsigned src;
   uint64_t value;
};

static struct fold
fold_src(unsigned src)
{
   return (struct fold){.kind = FOLD_SRC, .src = src};
}

static struct fold
fold_const(uint64_t value)
{
   return (struct fold){.kind = FOLD_CONST, .value = value};
}

static struct fold
fold_alu(const nir_alu_instr *alu, unsigned phi_src, nir_const_value c)
{
   const struct fold none = {.kind = FOLD_NONE};
   unsigned num_srcs = nir_op_infos[alu->op].num_inputs;
   bool all_const = true;
   for (unsigned s = 0; s < num_srcs; s++) {
      if (s != phi_src && !nir_src_is_const(alu->src[s].src))
         all_const = false;
   }
   if (all_const) {
      switch (alu->op) {
      case nir_op_fneg:
      case nir_op_fabs:
      case nir_op_fsat:
      case nir_op_fsat_signed:
      case nir_op_fclamp_pos:
      case nir_op_ineg:
      case nir_op_inot:
      case nir_op_mov:
         return none;
      default:
         if (nir_op_is_vec(alu->op) || nir_op_infos[alu->op].num_inputs < 2)
            return none;
         return (struct fold){.kind = FOLD_EVAL};
      }
   }

   unsigned bits = alu->def.bit_size;
   bool no_sz = !nir_alu_instr_is_signed_zero_preserve(alu);
   bool no_inf_nan = !nir_alu_instr_is_inf_preserve(alu) &&
                     !nir_alu_instr_is_nan_preserve(alu);
   unsigned other = 1 - phi_src;

   switch (alu->op) {
   case nir_op_fadd:
      if (const_is_float(c, bits, -0.0, false) ||
          (no_sz && const_is_float(c, bits, 0.0, true)))
         return fold_src(other);
      return none;

   case nir_op_fmul:
      if (const_is_float(c, bits, 1.0, false))
         return fold_src(other);
      if (no_sz && no_inf_nan && const_is_float(c, bits, 0.0, true))
         return fold_const(0);
      return none;

   case nir_op_ffma:
      if (phi_src < 2 && no_sz && no_inf_nan &&
          const_is_float(c, bits, 0.0, true))
         return fold_src(2);
      return none;

   case nir_op_iadd:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_umax:
      if (const_is_uint(c, bits, 0))
         return fold_src(other);
      if ((alu->op == nir_op_ior || alu->op == nir_op_umax) &&
          const_is_uint(c, bits, ~0ull))
         return fold_const(BITFIELD64_MASK(bits));
      return none;

   case nir_op_iand:
   case nir_op_umin:
      if (const_is_uint(c, bits, ~0ull))
         return fold_src(other);
      if (const_is_uint(c, bits, 0))
         return fold_const(0);
      return none;

   case nir_op_imul:
      if (const_is_uint(c, bits, 1))
         return fold_src(other);
      if (const_is_uint(c, bits, 0))
         return fold_const(0);
      return none;

   case nir_op_isub:
      if (phi_src == 1 && const_is_uint(c, bits, 0))
         return fold_src(0);
      return none;

   case nir_op_ishl:
   case nir_op_ishr:
   case nir_op_ushr:
      if (phi_src == 1 &&
          const_is_uint(c, nir_src_bit_size(alu->src[1].src), 0))
         return fold_src(0);
      if (phi_src == 0 && const_is_uint(c, bits, 0))
         return fold_const(0);
      return none;

   default:
      return none;
   }
}

static nir_alu_instr *
clone_into_pred(nir_builder *b, nir_alu_instr *alu, unsigned phi_src,
                nir_phi_src *src)
{
   nir_alu_instr *clone =
      nir_instr_as_alu(nir_instr_clone(b->shader, &alu->instr));
   nir_instr_insert(nir_after_block_before_jump(src->pred), &clone->instr);
   nir_src_rewrite(&clone->src[phi_src].src, src->src.ssa);
   return clone;
}

static bool
try_fold_phi_alu(nir_builder *b, nir_phi_instr *phi)
{
   nir_block *merge = phi->instr.block;

   if (phi->def.num_components != 1 || !list_is_singular(&phi->def.uses))
      return false;

   nir_src *use = list_first_entry(&phi->def.uses, nir_src, use_link);
   if (nir_src_is_if(use))
      return false;

   nir_instr *use_instr = nir_src_use_instr(use);
   if (use_instr->type != nir_instr_type_alu || use_instr->block != merge)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(use_instr);
   if (alu->def.num_components != 1 || nir_alu_instr_no_transform(alu))
      return false;

   unsigned num_srcs = nir_op_infos[alu->op].num_inputs;
   int phi_src = -1;
   for (unsigned s = 0; s < num_srcs; s++) {
      if (alu->src[s].src.ssa == &phi->def) {
         if (phi_src >= 0)
            return false;
         phi_src = s;
      }
   }
   assert(phi_src >= 0);

   unsigned const_preds = 0, other_preds = 0;
   nir_foreach_phi_src(src, phi) {
      for (unsigned s = 0; s < num_srcs; s++) {
         if (s != (unsigned)phi_src &&
             !src_available_in(&alu->src[s].src, src->pred))
            return false;
      }

      if (nir_src_is_const(src->src)) {
         struct fold f =
            fold_alu(alu, phi_src, *nir_src_as_const_value(src->src));
         if (f.kind == FOLD_NONE ||
             (f.kind == FOLD_SRC && !alu_src_is_simple(alu, f.src)))
            return false;
         const_preds++;
      } else {
         other_preds++;
      }
   }

   if (const_preds == 0 || other_preds > 1)
      return false;

   nir_phi_instr *new_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&new_phi->instr, &new_phi->def, 1, alu->def.bit_size);

   nir_foreach_phi_src(src, phi) {
      nir_def *value;
      if (nir_src_is_const(src->src)) {
         struct fold f =
            fold_alu(alu, phi_src, *nir_src_as_const_value(src->src));
         if (f.kind == FOLD_SRC) {
            value = alu->src[f.src].src.ssa;
         } else if (f.kind == FOLD_CONST) {
            b->cursor = nir_after_block_before_jump(src->pred);
            value = nir_imm_intN_t(b, f.value, alu->def.bit_size);
         } else {
            value = &clone_into_pred(b, alu, phi_src, src)->def;
         }
      } else {
         value = &clone_into_pred(b, alu, phi_src, src)->def;
      }
      nir_phi_instr_add_src(new_phi, src->pred, value);
   }

   nir_instr_insert(nir_before_block(merge), &new_phi->instr);
   nir_def_replace(&alu->def, &new_phi->def);
   nir_instr_remove(&phi->instr);
   return true;
}

bool
pan_nir_opt_phi_alu(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_dominance);
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      nir_foreach_block(block, impl) {
         nir_cf_node *prev = nir_cf_node_prev(&block->cf_node);
         if (!prev || prev->type != nir_cf_node_if)
            continue;

         nir_foreach_phi_safe(phi, block)
            impl_progress |= try_fold_phi_alu(&b, phi);
      }

      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }

   return progress;
}
