/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef IR3_LDG_STG_H_
#define IR3_LDG_STG_H_

#include "ir3_context.h"

/* Offset handling for ldg/stg and their register-offset ldg.a/stg.a forms,
 * shared between the pre-a6xx and a6xx+ backends.
 */

/* On a6xx, on top of the offset_shift that applies to the whole offset, there's
 * a second shift that only applies to the GPR part of the offset (so not to the
 * immediate part). We extract that here by simply pattern matching for ishl on
 * the offset src. Returns the shift if a match is found and it fits in the
 * 2-bit field, in which case *offset_src is set to the src of ishl and
 * *offset_src_comp to the component of *offset_src.
 */
static unsigned
parse_src_shift(struct ir3_context *ctx, nir_src **offset_src,
                unsigned *offset_src_comp)
{
   *offset_src_comp = 0;

   if (ctx->compiler->gen >= 7) {
      return 0;
   }

   nir_scalar offset = nir_scalar_resolved((*offset_src)->ssa, 0);

   if (!nir_scalar_is_alu(offset) || nir_scalar_alu_op(offset) != nir_op_ishl) {
      return 0;
   }

   nir_scalar shift_src = nir_scalar_chase_alu_src(offset, 1);

   if (!nir_scalar_is_const(shift_src)) {
      return 0;
   }

   unsigned shift = nir_scalar_as_uint(shift_src);

   if (shift >= (1 << 2)) {
      return 0;
   }

   nir_alu_instr *offset_alu = nir_def_as_alu(offset.def);
   *offset_src = &offset_alu->src[0].src;
   *offset_src_comp = offset_alu->src[0].swizzle[offset.comp];
   return shift;
}

static bool
base_fits_ldg_stg_a(struct ir3_compiler *compiler, unsigned base)
{
   if (compiler->gen >= 7) {
      return base < (1 << 8);
   }

   return base < (1 << 2);
}

/* Represents an offset for ldg/stg(.a):
 * - src == NULL: ldg/stg base_address + imm
 * - src != NULL:
 *   - a6xx: ldg/stg.a base_addr + (src << src_shift) + imm
 *   - a7xx: ldg/stg.a base_addr + src + imm
 */
struct ldg_stg_offset {
   struct ir3_instruction *src;
   struct ir3_instruction *src_shift;
   struct ir3_instruction *imm;
};

static struct ldg_stg_offset
ldg_stg_offset(struct ir3_context *ctx, nir_intrinsic_instr *intr)
{
   assert(intr->intrinsic == nir_intrinsic_load_global_offset ||
          intr->intrinsic == nir_intrinsic_store_global_offset ||
          intr->intrinsic == nir_intrinsic_load_global ||
          intr->intrinsic == nir_intrinsic_store_global);

   if (intr->intrinsic == nir_intrinsic_load_global ||
       intr->intrinsic == nir_intrinsic_store_global) {
      return (struct ldg_stg_offset){
         .imm = create_immed(&ctx->build, 0),
      };
   }

   if (ctx->compiler->gen >= 7) {
      assert(nir_intrinsic_offset_shift(intr) == 0);
   } else {
      ASSERTED unsigned bit_size =
          intr->intrinsic == nir_intrinsic_load_global_offset
            ? intr->def.bit_size
            : intr->src[0].ssa->bit_size;
      assert(nir_intrinsic_offset_shift(intr) == ffs(bit_size / 8) - 1);
   }

   struct ldg_stg_offset offset = {};
   nir_src *offset_src = nir_get_io_offset_src(intr);
   uint32_t base = nir_intrinsic_base(intr);
   unsigned offset_shift = nir_intrinsic_offset_shift(intr);
   struct ir3_builder *b = &ctx->build;

   if (nir_src_is_const(*offset_src)) {
      uint32_t full_imm_offset = base + nir_src_as_uint(*offset_src);
      uint32_t full_imm_offset_bytes = full_imm_offset << offset_shift;

      /* ldg/stg offset immediate is 13 bits signed but we only get unsigned
       * offsets out if NIR so we can only use 12 of those bits. Note that
       * ldg/stg use byte offsets even on a6xx.
       */
      if (full_imm_offset_bytes < (1 << 12)) {
         offset.imm = create_immed(b, full_imm_offset_bytes);
      } else {
         /* The immediate offset does not fit. Generate ldg/stg.a with the
          * immediate in a GPR.
          */
         offset.src = create_immed(b, full_imm_offset);
         offset.src_shift = create_immed(b, 0);
         offset.imm = create_immed(b, 0);
      }
   } else {
      if (base_fits_ldg_stg_a(ctx->compiler, base)) {
         unsigned offset_src_comp;
         unsigned shift = parse_src_shift(ctx, &offset_src, &offset_src_comp);
         offset.src = ir3_get_src(ctx, offset_src)[offset_src_comp];
         offset.src_shift = create_immed(b, shift);
         offset.imm = create_immed(b, base);
      } else {
         /* This should be rare, but various passes might update
          * base/offset_shift in a way that makes the combination illegal.
          * Detect that here and replace base by an add.
          */
         offset.src = ir3_ADD_U(b, ir3_get_src(ctx, offset_src)[0], 0,
                                create_immed(b, base), 0);
         offset.src_shift = create_immed(b, 0);
         offset.imm = create_immed(b, 0);
      }
   }

   return offset;
}

#endif /* IR3_LDG_STG_H_ */
