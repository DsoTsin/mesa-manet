/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "pan_nir.h"
#include "nir_builder.h"

static bool
block_has_work(const nir_block *block)
{
   nir_foreach_instr(instr, block) {
      if (instr->type == nir_instr_type_alu ||
          instr->type == nir_instr_type_tex ||
          instr->type == nir_instr_type_intrinsic)
         return true;
   }

   return false;
}

static void
instrument_block(nir_builder *b, nir_block *block, unsigned counters_fau,
                 unsigned table_offset)
{
   b->cursor = nir_after_phis(block);

   nir_def *active = nir_ballot(b, 1, 32, nir_imm_int(b, ~0));
   nir_def *eligible = active;
   if (b->shader->info.stage == MESA_SHADER_FRAGMENT) {
      nir_def *covered = nir_ine_pan(b, nir_load_sample_mask_in(b),
                                     nir_imm_int(b, 0));
      eligible = nir_ballot(b, 1, 32, covered);
   }

   nir_def *leader = nir_ieq_pan(b, nir_load_subgroup_invocation(b),
                                 nir_ufind_msb(b, eligible));

   nir_push_if(b, leader);
   {
      nir_def *threads = nir_bit_count(b, active);
      nir_def *counters = nir_pack_64_2x32(
         b, nir_load_push_constant(b, 2, 32, nir_imm_int(b, counters_fau)));
      counters = nir_iadd_imm(b, counters,
                              pan_instrumentation_slot(b->shader->info.stage) *
                                 PAN_INSTRUMENTATION_SLOT_SIZE);

      nir_def *table =
         nir_iadd_imm(b, nir_load_constant_base_ptr(b, 1, 64), table_offset);
      nir_def *counts[2] = {
         nir_load_global_constant(b, 4, 32, table, .align_mul = 16),
         nir_load_global_constant(b, 2, 32, nir_iadd_imm(b, table, 16),
                                  .align_mul = 16),
      };

      for (unsigned m = 0; m < PAN_INSTRUMENTATION_METRICS; m++) {
         nir_def *count = nir_channel(b, counts[m / 4], m % 4);
         nir_def *value = nir_pack_64_2x32_split(b, nir_imul(b, threads, count),
                                                 nir_imm_int(b, 0));
         nir_global_atomic(b, 64, nir_iadd_imm(b, counters, m * 8), value,
                           .atomic_op = nir_atomic_op_iadd);
      }
   }
   nir_pop_if(b, NULL);

   nir_instrument_block_pan(b, .base = table_offset);
}

bool
pan_nir_instrument(nir_shader *nir, unsigned counters_fau)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);

   unsigned count = 0;
   nir_foreach_block(block, impl)
      count += block_has_work(block);

   if (!count)
      return nir_no_progress(impl);

   nir_block **blocks = malloc(count * sizeof(*blocks));
   if (!blocks)
      return nir_no_progress(impl);

   unsigned i = 0;
   nir_foreach_block(block, impl) {
      if (block_has_work(block))
         blocks[i++] = block;
   }

   const unsigned table =
      ALIGN_POT(nir->constant_data_size, PAN_INSTRUMENTATION_TABLE_STRIDE);
   const unsigned size = table + count * PAN_INSTRUMENTATION_TABLE_STRIDE;
   nir->constant_data = rerzalloc_size(nir, nir->constant_data,
                                       nir->constant_data_size, size);
   nir->constant_data_size = size;

   nir_builder b = nir_builder_create(impl);
   for (i = 0; i < count; i++)
      instrument_block(&b, blocks[i], counters_fau,
                       table + i * PAN_INSTRUMENTATION_TABLE_STRIDE);

   free(blocks);

   nir_progress(true, impl, nir_metadata_none);
   nir_index_blocks(impl);
   return true;
}
