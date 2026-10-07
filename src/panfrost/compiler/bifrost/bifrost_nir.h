/*
 * Copyright (C) 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <stdbool.h>
#include "nir.h"
#include "nir_builder.h"
#include "nir_search.h"

bool bifrost_nir_lower_algebraic_late(nir_shader *shader, unsigned gpu_arch,
                                      bool is_kraid);
bool bifrost_nir_opt_boolean_bitwise(nir_shader *shader);
bool bifrost_nir_opt_fp16(nir_shader *shader, bool is_kraid);

bool bifrost_nir_opt_uniform_reassoc(nir_shader *shader);

static inline bool
bi_def_is_uniform_expr(const nir_def *def, unsigned depth)
{
   const nir_instr *instr = nir_def_instr(def);

   switch (instr->type) {
   case nir_instr_type_load_const:
      return true;

   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_push_constant)
         return nir_src_is_const(intr->src[0]);
      if (intr->intrinsic == nir_intrinsic_load_ubo)
         return nir_src_is_const(intr->src[0]) && nir_src_is_const(intr->src[1]);
      return false;
   }

   case nir_instr_type_alu: {
      if (depth == 0)
         return false;

      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         if (!bi_def_is_uniform_expr(alu->src[i].src.ssa, depth - 1))
            return false;
      }
      return true;
   }

   default:
      return false;
   }
}

static inline bool
is_uniform_expr(UNUSED const nir_search_state *state,
                const nir_alu_instr *instr, unsigned src,
                UNUSED unsigned num_components,
                UNUSED const uint8_t *swizzle)
{
   return bi_def_is_uniform_expr(instr->src[src].src.ssa, 4);
}

static inline bool
is_not_uniform_expr(const nir_search_state *state,
                    const nir_alu_instr *instr, unsigned src,
                    unsigned num_components, const uint8_t *swizzle)
{
   return !is_uniform_expr(state, instr, src, num_components, swizzle);
}
