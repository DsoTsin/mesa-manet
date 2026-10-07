/*
 * Copyright 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "pan_nir.h"

#define PAN_TENSOR_DIMS 4

static nir_def *
pad_coords(nir_builder *b, nir_def *coords)
{
   assert(coords->num_components <= PAN_TENSOR_DIMS);

   unsigned pad = PAN_TENSOR_DIMS - coords->num_components;
   nir_def *comps[PAN_TENSOR_DIMS];
   for (unsigned i = 0; i < PAN_TENSOR_DIMS; i++)
      comps[i] = i < pad ? nir_imm_int(b, 0) : nir_channel(b, coords, i - pad);

   return nir_vec(b, comps, PAN_TENSOR_DIMS);
}

static nir_def *
offset_coords(nir_builder *b, nir_def *coords, unsigned elems)
{
   if (!elems)
      return coords;

   return nir_vector_insert_imm(
      b, coords,
      nir_iadd_imm(b, nir_channel(b, coords, PAN_TENSOR_DIMS - 1), elems),
      PAN_TENSOR_DIMS - 1);
}

static unsigned
chunk_bytes(unsigned remaining)
{
   return MIN2(1u << util_logbase2(remaining), 16);
}

static nir_def *
chunk_bits(nir_builder *b, nir_def *data, unsigned start_bit, unsigned bytes)
{
   const unsigned bit_size = MIN2(bytes, 4) * 8;
   return nir_extract_bits(b, &data, 1, start_bit, bytes * 8 / bit_size,
                           bit_size);
}

static bool
lower_tensor_read(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);

   nir_def *handle = intr->src[0].ssa;
   nir_def *coords = pad_coords(b, intr->src[1].ssa);
   nir_def *oob = intr->src[2].ssa;
   const unsigned bit_size = intr->def.bit_size;
   const unsigned elem_bytes = bit_size / 8;
   const unsigned total = elem_bytes * intr->def.num_components;

   nir_def *parts[NIR_MAX_VEC_COMPONENTS];
   unsigned num_parts = 0;
   for (unsigned offset = 0; offset < total;) {
      const unsigned bytes = chunk_bytes(total - offset);
      nir_def *chunk = nir_load_tensor_pan(
         b, 32, handle, offset_coords(b, coords, offset / elem_bytes),
         nir_u2u32(b, chunk_bits(b, oob, offset * 8, bytes)),
         .access = nir_intrinsic_access(intr), .range = bytes);

      parts[num_parts++] = bytes < 4 ? nir_u2uN(b, chunk, bytes * 8) : chunk;
      offset += bytes;
   }

   nir_def_replace(&intr->def,
                   nir_extract_bits(b, parts, num_parts, 0,
                                    intr->def.num_components, bit_size));
   return true;
}

static bool
lower_tensor_write(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);

   nir_def *handle = intr->src[0].ssa;
   nir_def *coords = pad_coords(b, intr->src[1].ssa);
   nir_def *value = intr->src[2].ssa;
   const unsigned elem_bytes = value->bit_size / 8;
   const unsigned total = elem_bytes * value->num_components;

   for (unsigned offset = 0; offset < total;) {
      const unsigned bytes = chunk_bytes(total - offset);
      nir_store_tensor_pan(b, chunk_bits(b, value, offset * 8, bytes), handle,
                           offset_coords(b, coords, offset / elem_bytes),
                           .access = nir_intrinsic_access(intr));
      offset += bytes;
   }

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_tensor_size(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);

   nir_def *handle = intr->src[0].ssa;
   const unsigned dim = PAN_TENSOR_DIMS - nir_intrinsic_tensor_rank(intr) +
                        nir_intrinsic_base(intr);

   nir_def *hdr = pan_nir_load_va_desc(b, 2, 16, handle, 0);
   nir_def *dims = pan_nir_load_va_desc(b, PAN_TENSOR_DIMS, 16, handle, 4);

   nir_def *ext_dim = nir_ushr_imm(b, nir_u2u32(b, nir_channel(b, hdr, 0)), 14);
   nir_def *ext_hi = nir_ishl_imm(b, nir_u2u32(b, nir_channel(b, hdr, 1)), 16);
   nir_def *size = nir_u2u32(b, nir_channel(b, dims, dim));
   size = nir_bcsel(b, nir_ieq_imm(b, ext_dim, dim), nir_ior(b, size, ext_hi),
                    size);

   nir_def_replace(&intr->def, size);
   return true;
}

static bool
lower_tensor_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_tensor_read_arm:
      return lower_tensor_read(b, intr);
   case nir_intrinsic_tensor_write_arm:
      return lower_tensor_write(b, intr);
   case nir_intrinsic_tensor_size_arm:
      return lower_tensor_size(b, intr);
   default:
      return false;
   }
}

bool
pan_nir_lower_tensor(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_tensor_intr,
                                     nir_metadata_control_flow, NULL);
}
