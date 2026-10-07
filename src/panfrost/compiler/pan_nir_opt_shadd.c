#include "compiler/nir/nir_builder.h"
#include "pan_nir.h"
#include "util/hash_table.h"

static bool
opt_shadd(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if (alu->op != nir_op_iadd || alu->def.bit_size != 64 ||
       alu->def.num_components != 1)
      return false;

   nir_scalar add = nir_get_scalar(&alu->def, 0);
   for (unsigned i = 0; i < 2; i++) {
      nir_scalar shl = nir_scalar_chase_alu_src(add, i);
      bool narrow_shift = false;
      if (nir_scalar_is_alu(shl)) {
         if (nir_scalar_alu_op(shl) == nir_op_u2u64) {
            shl = nir_scalar_chase_alu_src(shl, 0);
            narrow_shift = true;
         } else if (nir_scalar_alu_op(shl) == nir_op_pack_64_2x32_split) {
            nir_scalar hi = nir_scalar_chase_alu_src(shl, 1);
            if (!nir_scalar_is_const(hi) || nir_scalar_as_uint(hi) != 0)
               continue;
            shl = nir_scalar_chase_alu_src(shl, 0);
            narrow_shift = true;
         }
      }
      if (!nir_scalar_is_alu(shl))
         continue;

      nir_op shift_op = nir_scalar_alu_op(shl);
      if (shift_op != nir_op_ishl && shift_op != nir_op_lshift_or_pan)
         continue;
      if (shift_op == nir_op_lshift_or_pan) {
         nir_scalar other = nir_scalar_chase_alu_src(shl, 2);
         if (!nir_scalar_is_const(other) || nir_scalar_as_uint(other) != 0)
            continue;
      }
      if (narrow_shift && shl.def->bit_size != 32)
         continue;

      nir_scalar shift = nir_scalar_chase_alu_src(shl, 1);
      if (!nir_scalar_is_const(shift) || nir_scalar_as_uint(shift) < 1 ||
          nir_scalar_as_uint(shift) > 7)
         continue;

      nir_scalar index = nir_scalar_chase_alu_src(shl, 0);
      nir_op op = narrow_shift ? nir_op_shadd_u32_pan : nir_op_shadd_pan;
      if (narrow_shift) {
         if (shift_op == nir_op_ishl) {
            if (!nir_is_scalar_nuw(b->shader, data, shl))
               continue;
         } else if (nir_unsigned_upper_bound(b->shader, data, index) >
                    (UINT32_MAX >> nir_scalar_as_uint(shift))) {
            continue;
         }
      } else if (nir_scalar_is_alu(index) &&
          (nir_scalar_alu_op(index) == nir_op_u2u64 ||
           nir_scalar_alu_op(index) == nir_op_i2i64)) {
         nir_scalar narrow = nir_scalar_chase_alu_src(index, 0);
         if (narrow.def->bit_size == 16 || narrow.def->bit_size == 32) {
            op = nir_scalar_alu_op(index) == nir_op_i2i64 ?
                 nir_op_shadd_s32_pan : nir_op_shadd_u32_pan;
            index = narrow;
         }
      }

      b->cursor = nir_before_instr(&alu->instr);
      nir_scalar base = nir_scalar_chase_alu_src(add, 1 - i);
      nir_def *value = nir_channel(b, index.def, index.comp);
      if (op == nir_op_shadd_s32_pan)
         value = nir_i2i32(b, value);
      else if (op == nir_op_shadd_u32_pan)
         value = nir_u2u32(b, value);

      nir_def *res = nir_build_alu(b, op,
                                 nir_channel(b, base.def, base.comp), value,
                                 nir_imm_int(b, nir_scalar_as_uint(shift)), NULL);
      nir_def_replace(&alu->def, res);
      _mesa_hash_table_clear(data, NULL);
      return true;
   }

   return false;
}

bool
pan_nir_opt_shadd(nir_shader *nir, unsigned arch)
{
   if (arch != 15 || !pan_use_kraid(arch, nir->info.stage, nir->info.internal))
      return false;

   struct hash_table *range_ht = _mesa_pointer_hash_table_create(NULL);
   bool progress = nir_shader_alu_pass(nir, opt_shadd,
                                       nir_metadata_control_flow, range_ht);
   _mesa_hash_table_destroy(range_ht, NULL);
   return progress;
}
