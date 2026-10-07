/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "pan_nir.h"
#include "nir_builder.h"

static nir_def *
bf2f(nir_builder *b, nir_def *x)
{
   return nir_pack_32_2x16_split(b, nir_imm_zero(b, x->num_components, 16), x);
}

static nir_def *
f2bf(nir_builder *b, nir_def *x)
{
   return nir_unpack_32_2x16_split_y(b, x);
}

static nir_def *
bfdot(nir_builder *b, nir_def *x, nir_def *y)
{
   b->fp_math_ctrl = nir_fp_exact;
   nir_def *acc = nir_fmul(b, bf2f(b, nir_channel(b, x, 0)),
                           bf2f(b, nir_channel(b, y, 0)));
   for (unsigned i = 1; i < x->num_components; i++) {
      acc = nir_ffma(b, bf2f(b, nir_channel(b, x, i)),
                     bf2f(b, nir_channel(b, y, i)), acc);
   }
   return f2bf(b, acc);
}

static bool
lower_bf16_instr(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   b->cursor = nir_before_instr(&alu->instr);

   nir_def *res;
   switch (alu->op) {
   case nir_op_bf2f:
      res = bf2f(b, nir_ssa_for_alu_src(b, alu, 0));
      break;
   case nir_op_f2bf:
      res = f2bf(b, nir_ssa_for_alu_src(b, alu, 0));
      break;
   case nir_op_bfdot2:
   case nir_op_bfdot3:
   case nir_op_bfdot4:
   case nir_op_bfdot8:
   case nir_op_bfdot16:
      res = bfdot(b, nir_ssa_for_alu_src(b, alu, 0),
                  nir_ssa_for_alu_src(b, alu, 1));
      break;
   default:
      return false;
   }

   nir_def_replace(&alu->def, res);
   return true;
}

bool
pan_nir_lower_bf16(nir_shader *shader)
{
   return nir_shader_alu_pass(shader, lower_bf16_instr,
                              nir_metadata_control_flow, NULL);
}
