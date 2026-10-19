/*
 * Copyright 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "nir_builder.h"
#include "vtn_private.h"

void
vtn_handle_tensor_type(struct vtn_builder *b, struct vtn_value *val,
                       const uint32_t *w, unsigned count)
{
   struct vtn_type *element = vtn_get_type(b, w[2]);
   vtn_fail_if(element->base_type != vtn_base_type_scalar,
               "OpTypeTensorARM Element Type must be a scalar type");

   val->type->base_type = vtn_base_type_tensor;
   val->type->tensor_element_type = element;
   val->type->tensor_rank = count > 3 ? vtn_constant_uint(b, w[3]) : 0;
   val->type->type = nir_address_format_to_glsl_type(
      vtn_mode_to_address_format(b, vtn_variable_mode_tensor));
}

static unsigned
tensor_element_bit_size(const struct vtn_type *tensor)
{
   const struct glsl_type *type = tensor->tensor_element_type->type;
   return glsl_type_is_boolean(type) ? 8 : glsl_get_bit_size(type);
}

static nir_def *
tensor_coords(struct vtn_builder *b, uint32_t id, unsigned rank)
{
   struct vtn_ssa_value *coords = vtn_ssa_value(b, id);
   vtn_fail_if(!glsl_type_is_array(coords->type) ||
               glsl_get_length(coords->type) != rank ||
               !nir_num_components_valid(rank),
               "Tensor Coordinates must be an array of Rank integers");

   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < rank; i++)
      comps[i] = nir_u2u32(&b->nb, coords->elems[i]->def);

   return nir_vec(&b->nb, comps, rank);
}

static nir_def *
tensor_bool_to_byte(nir_builder *b, nir_def *def)
{
   return nir_bcsel(b, def, nir_imm_intN_t(b, 0xff, 8), nir_imm_intN_t(b, 0, 8));
}

static unsigned
tensor_chunk_size(unsigned remaining)
{
   return remaining >= 16 ? 16 : remaining >= 8 ? 8
                            : remaining >= 5 ? 5 : remaining;
}

static nir_def *
tensor_offset_coords(struct vtn_builder *b, nir_def *coords, unsigned offset)
{
   if (!offset)
      return coords;
   unsigned last = coords->num_components - 1;
   nir_def *inner = nir_iadd_imm(&b->nb, nir_channel(&b->nb, coords, last),
                                offset);
   return nir_vector_insert_imm(&b->nb, coords, inner, last);
}

static nir_def *
tensor_value_to_nir(struct vtn_builder *b, struct vtn_ssa_value *val,
                    unsigned offset, unsigned count)
{
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < count; i++) {
      nir_def *def = glsl_type_is_array(val->type) ? val->elems[offset + i]->def
                                                    : val->def;
      comps[i] = def->bit_size == 1 ? tensor_bool_to_byte(&b->nb, def) : def;
   }
   return nir_vec(&b->nb, comps, count);
}

static enum gl_access_qualifier
tensor_access(struct vtn_builder *b, uint32_t tensor_id, uint32_t operands)
{
   enum gl_access_qualifier access = 0;

   if (vtn_value_is_non_uniform(b, vtn_untyped_value(b, tensor_id)))
      access |= ACCESS_NON_UNIFORM;
   if (operands & SpvTensorOperandsNontemporalARMMask)
      access |= ACCESS_NON_TEMPORAL;

   return access;
}

void
vtn_handle_tensor_instruction(struct vtn_builder *b, SpvOp opcode,
                              const uint32_t *w, unsigned count)
{
   switch (opcode) {
   case SpvOpTensorReadARM: {
      struct vtn_type *res_type = vtn_get_type(b, w[1]);
      struct vtn_type *tensor = vtn_get_value_type(b, w[3]);
      vtn_fail_if(tensor->base_type != vtn_base_type_tensor,
                  "OpTensorReadARM Tensor must be an OpTypeTensorARM");

      const bool is_array = glsl_type_is_array(res_type->type);
      const unsigned num_components =
         is_array ? glsl_get_length(res_type->type) : 1;
      const unsigned bit_size = tensor_element_bit_size(tensor);

      const uint32_t operands = count > 5 ? w[5] : 0;
      unsigned arg = 6;

      nir_def *oob = nir_imm_zero(&b->nb, 1, bit_size);
      if (operands & SpvTensorOperandsOutOfBoundsValueARMMask) {
         nir_def *value = vtn_get_nir_ssa(b, w[arg++]);
         if (value->bit_size == 1)
            value = tensor_bool_to_byte(&b->nb, value);
         oob = value;
      }

      SpvScope scope = SpvScopeInvocation;
      const bool make_visible =
         operands & SpvTensorOperandsMakeElementVisibleARMMask;
      if (make_visible) {
         vtn_fail_if(!(operands & SpvTensorOperandsNonPrivateElementARMMask),
                     "MakeElementVisibleARM requires NonPrivateElementARM");
         scope = vtn_constant_uint(b, w[arg++]);
      }

      nir_def *handle = vtn_get_nir_ssa(b, w[3]);
      nir_def *coords = tensor_coords(b, w[4], tensor->tensor_rank);
      struct vtn_ssa_value *ssa = vtn_create_ssa_value(b, res_type->type);
      for (unsigned offset = 0; offset < num_components;) {
         unsigned chunk = tensor_chunk_size(num_components - offset);
         nir_def *res = nir_tensor_read_arm(
            &b->nb, bit_size, handle, tensor_offset_coords(b, coords, offset),
            nir_replicate(&b->nb, oob, chunk),
            .access = tensor_access(b, w[3], operands));
         if (glsl_type_is_boolean(tensor->tensor_element_type->type))
            res = nir_ine_imm(&b->nb, res, 0);
         if (is_array) {
            for (unsigned i = 0; i < chunk; i++)
               ssa->elems[offset + i]->def = nir_channel(&b->nb, res, i);
         } else {
            ssa->def = res;
         }
         offset += chunk;
      }
      if (make_visible)
         vtn_emit_memory_barrier(b, scope,
                                 SpvMemorySemanticsMakeVisibleMask |
                                 SpvMemorySemanticsImageMemoryMask);
      vtn_push_ssa_value(b, w[2], ssa);
      break;
   }

   case SpvOpTensorWriteARM: {
      struct vtn_type *tensor = vtn_get_value_type(b, w[1]);
      vtn_fail_if(tensor->base_type != vtn_base_type_tensor,
                  "OpTensorWriteARM Tensor must be an OpTypeTensorARM");

      const uint32_t operands = count > 4 ? w[4] : 0;
      if (operands & SpvTensorOperandsMakeElementAvailableARMMask) {
         vtn_fail_if(!(operands & SpvTensorOperandsNonPrivateElementARMMask),
                     "MakeElementAvailableARM requires NonPrivateElementARM");
         vtn_emit_memory_barrier(b, vtn_constant_uint(b, w[5]),
                                 SpvMemorySemanticsMakeAvailableMask |
                                 SpvMemorySemanticsImageMemoryMask);
      }

      struct vtn_ssa_value *object = vtn_ssa_value(b, w[3]);
      unsigned elements = glsl_type_is_array(object->type)
                             ? glsl_get_length(object->type) : 1;
      nir_def *handle = vtn_get_nir_ssa(b, w[1]);
      nir_def *coords = tensor_coords(b, w[2], tensor->tensor_rank);
      for (unsigned offset = 0; offset < elements;) {
         unsigned chunk = tensor_chunk_size(elements - offset);
         nir_tensor_write_arm(&b->nb, handle,
                              tensor_offset_coords(b, coords, offset),
                              tensor_value_to_nir(b, object, offset, chunk),
                              .access = tensor_access(b, w[1], operands));
         offset += chunk;
      }
      break;
   }

   case SpvOpTensorQuerySizeARM: {
      struct vtn_type *res_type = vtn_get_type(b, w[1]);
      struct vtn_type *tensor = vtn_get_value_type(b, w[3]);
      vtn_fail_if(tensor->base_type != vtn_base_type_tensor,
                  "OpTensorQuerySizeARM Tensor must be an OpTypeTensorARM");

      const uint32_t dim = vtn_constant_uint(b, w[4]);
      vtn_fail_if(dim >= tensor->tensor_rank,
                  "OpTensorQuerySizeARM Dimension must be less than Rank");

      nir_def *size =
         nir_tensor_size_arm(&b->nb, vtn_get_nir_ssa(b, w[3]), .base = dim,
                             .tensor_rank = tensor->tensor_rank);
      vtn_push_nir_ssa(b, w[2],
                       nir_u2uN(&b->nb, size,
                                glsl_get_bit_size(res_type->type)));
      break;
   }

   default:
      vtn_fail_with_opcode("Invalid tensor opcode", opcode);
   }
}
