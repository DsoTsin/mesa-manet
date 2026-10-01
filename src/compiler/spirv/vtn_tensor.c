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
               rank > NIR_MAX_VEC_COMPONENTS,
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

static nir_def *
tensor_value_to_nir(struct vtn_builder *b, struct vtn_ssa_value *val)
{
   unsigned count = glsl_type_is_array(val->type) ?
                    glsl_get_length(val->type) : 1;
   vtn_fail_if(count > NIR_MAX_VEC_COMPONENTS,
               "Too many tensor elements accessed");

   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < count; i++) {
      nir_def *def = glsl_type_is_array(val->type) ? val->elems[i]->def
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
      vtn_fail_if(num_components > NIR_MAX_VEC_COMPONENTS,
                  "Too many tensor elements read");

      const uint32_t operands = count > 5 ? w[5] : 0;
      unsigned arg = 6;

      nir_def *oob = nir_imm_zero(&b->nb, num_components, bit_size);
      if (operands & SpvTensorOperandsOutOfBoundsValueARMMask) {
         nir_def *value = vtn_get_nir_ssa(b, w[arg++]);
         if (value->bit_size == 1)
            value = tensor_bool_to_byte(&b->nb, value);
         oob = nir_replicate(&b->nb, value, num_components);
      }

      SpvScope scope = SpvScopeInvocation;
      const bool make_visible =
         operands & SpvTensorOperandsMakeElementVisibleARMMask;
      if (make_visible) {
         vtn_fail_if(!(operands & SpvTensorOperandsNonPrivateElementARMMask),
                     "MakeElementVisibleARM requires NonPrivateElementARM");
         scope = vtn_constant_uint(b, w[arg++]);
      }

      nir_def *res = nir_tensor_read_arm(
         &b->nb, bit_size, vtn_get_nir_ssa(b, w[3]),
         tensor_coords(b, w[4], tensor->tensor_rank), oob,
         .access = tensor_access(b, w[3], operands));

      if (make_visible)
         vtn_emit_memory_barrier(b, scope,
                                 SpvMemorySemanticsMakeVisibleMask |
                                 SpvMemorySemanticsImageMemoryMask);

      if (glsl_type_is_boolean(tensor->tensor_element_type->type))
         res = nir_ine_imm(&b->nb, res, 0);

      struct vtn_ssa_value *ssa = vtn_create_ssa_value(b, res_type->type);
      if (is_array) {
         for (unsigned i = 0; i < num_components; i++)
            ssa->elems[i]->def = nir_channel(&b->nb, res, i);
      } else {
         ssa->def = res;
      }
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

      nir_def *value = tensor_value_to_nir(b, vtn_ssa_value(b, w[3]));

      nir_tensor_write_arm(&b->nb, vtn_get_nir_ssa(b, w[1]),
                           tensor_coords(b, w[2], tensor->tensor_rank), value,
                           .access = tensor_access(b, w[1], operands));
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
