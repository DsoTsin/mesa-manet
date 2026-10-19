#undef NDEBUG
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_constant_expressions.h"
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv.h"
#include "compiler/spirv/spirv_info.h"
#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct module {
   uint32_t words[1024];
   unsigned count;
};

static void
emit(struct module *m, SpvOp op, unsigned count, ...)
{
   assert(m->count + count + 1 <= ARRAY_SIZE(m->words));
   m->words[m->count++] = ((count + 1) << 16) | op;
   va_list args;
   va_start(args, count);
   for (unsigned i = 0; i < count; i++)
      m->words[m->count++] = va_arg(args, uint32_t);
   va_end(args);
}

static struct module
make_module(unsigned count, unsigned bits, bool floating, bool sign,
            bool scalar, unsigned read_start, unsigned write_start,
            bool visibility, unsigned rank)
{
   struct module m = {.words = {SpvMagicNumber, 0x00010500, 0, 64, 0},
                      .count = 5};
   emit(&m, SpvOpCapability, 1, SpvCapabilityShader);
   emit(&m, SpvOpCapability, 1, SpvCapabilityTensorsARM);
   if (bits == 8)
      emit(&m, SpvOpCapability, 1, SpvCapabilityInt8);
   if (bits == 16)
      emit(&m, SpvOpCapability, 1,
           floating ? SpvCapabilityFloat16 : SpvCapabilityInt16);
   if (bits == 64)
      emit(&m, SpvOpCapability, 1,
           floating ? SpvCapabilityFloat64 : SpvCapabilityInt64);
   if (visibility) {
      emit(&m, SpvOpCapability, 1, SpvCapabilityVulkanMemoryModel);
      emit(&m, SpvOpCapability, 1, SpvCapabilityVulkanMemoryModelDeviceScope);
   }
   emit(&m, SpvOpExtension, 4, 0x5f565053, 0x5f4d5241, 0x736e6574, 0x0073726f);
   emit(&m, SpvOpMemoryModel, 2, SpvAddressingModelLogical,
        visibility ? SpvMemoryModelVulkan : SpvMemoryModelGLSL450);
   emit(&m, SpvOpEntryPoint, 6, SpvExecutionModelGLCompute, 1, 0x6e69616d, 0, 9,
        10);
   emit(&m, SpvOpExecutionMode, 5, 1, SpvExecutionModeLocalSize, 1, 1, 1);
   emit(&m, SpvOpDecorate, 3, 9, SpvDecorationDescriptorSet, 0);
   emit(&m, SpvOpDecorate, 3, 9, SpvDecorationBinding, 0);
   emit(&m, SpvOpDecorate, 3, 10, SpvDecorationDescriptorSet, 0);
   emit(&m, SpvOpDecorate, 3, 10, SpvDecorationBinding, 1);
   emit(&m, SpvOpTypeVoid, 1, 2);
   emit(&m, SpvOpTypeFunction, 2, 3, 2);
   emit(&m, SpvOpTypeInt, 3, 4, 32, 0);
   emit(&m, SpvOpConstant, 3, 4, 5, rank);
   if (bits == 1)
      emit(&m, SpvOpTypeBool, 1, 6);
   else if (floating)
      emit(&m, SpvOpTypeFloat, 2, 6, bits);
   else
      emit(&m, SpvOpTypeInt, 3, 6, bits, sign);
   emit(&m, SpvOpTypeTensorARM, 3, 7, 6, 5);
   emit(&m, SpvOpTypePointer, 3, 8, SpvStorageClassUniformConstant, 7);
   emit(&m, SpvOpVariable, 3, 8, 9, SpvStorageClassUniformConstant);
   emit(&m, SpvOpVariable, 3, 8, 10, SpvStorageClassUniformConstant);
   emit(&m, SpvOpTypeArray, 3, 11, 4, 5);
   emit(&m, SpvOpConstant, 3, 4, 12, 1);
   emit(&m, SpvOpConstant, 3, 4, 13, read_start);
   emit(&m, SpvOpConstant, 3, 4, 14, write_start);
   unsigned pos = m.count;
   m.words[m.count++] = ((rank + 3) << 16) | SpvOpConstantComposite;
   m.words[m.count++] = 11;
   m.words[m.count++] = 15;
   for (unsigned i = 0; i < rank; i++)
      m.words[m.count++] = i == rank - 1 ? 13 : 12;
   assert(m.count == pos + rank + 3);
   m.words[m.count++] = ((rank + 3) << 16) | SpvOpConstantComposite;
   m.words[m.count++] = 11;
   m.words[m.count++] = 16;
   for (unsigned i = 0; i < rank; i++)
      m.words[m.count++] = i == rank - 1 ? 14 : 12;
   emit(&m, SpvOpConstant, 3, 4, 17, count);
   emit(&m, SpvOpTypeArray, 3, 18, 6, 17);
   if (bits == 1)
      emit(&m, SpvOpConstantTrue, 2, 6, 19);
   else if (bits == 64)
      emit(&m, SpvOpConstant, 4, 6, 19, 0xa55a37c1, 0x45b27e91);
   else
      emit(&m, SpvOpConstant, 3, 6, 19,
           bits == 8    ? 0xc1
           : bits == 16 ? 0x37c1
                        : 0xa55a37c1);
   emit(&m, SpvOpFunction, 4, 2, 1, SpvFunctionControlMaskNone, 3);
   emit(&m, SpvOpLabel, 1, 20);
   emit(&m, SpvOpLoad, 3, 7, 21, 9);
   emit(&m, SpvOpLoad, 3, 7, 22, 10);
   unsigned read_flags = SpvTensorOperandsOutOfBoundsValueARMMask |
                         SpvTensorOperandsNontemporalARMMask;
   unsigned write_flags = SpvTensorOperandsNontemporalARMMask;
   if (visibility) {
      read_flags |= SpvTensorOperandsMakeElementVisibleARMMask |
                    SpvTensorOperandsNonPrivateElementARMMask;
      write_flags |= SpvTensorOperandsMakeElementAvailableARMMask |
                     SpvTensorOperandsNonPrivateElementARMMask;
      emit(&m, SpvOpTensorReadARM, 7, scalar ? 6 : 18, 23, 21, 15, read_flags,
           19, 12);
      emit(&m, SpvOpTensorWriteARM, 5, 22, 16, 23, write_flags, 12);
   } else {
      emit(&m, SpvOpTensorReadARM, 6, scalar ? 6 : 18, 23, 21, 15, read_flags,
           19);
      emit(&m, SpvOpTensorWriteARM, 4, 22, 16, 23, write_flags);
   }
   emit(&m, SpvOpReturn, 0);
   emit(&m, SpvOpFunctionEnd, 0);
   return m;
}

static void
execute(nir_shader *nir, uint8_t *input, uint8_t *output, unsigned width,
        unsigned bytes, unsigned *read_elements, unsigned *write_elements,
        unsigned *barriers)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   nir_const_value(*values)[NIR_MAX_VEC_COMPONENTS] =
       calloc(impl->ssa_alloc, sizeof(*values));
   nir_foreach_block(block, impl)
   {
      nir_foreach_instr(instr, block)
      {
         if (instr->type == nir_instr_type_load_const) {
            nir_load_const_instr *load = nir_instr_as_load_const(instr);
            memcpy(values[load->def.index], load->value,
                   load->def.num_components * sizeof(nir_const_value));
         } else if (instr->type == nir_instr_type_alu) {
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            nir_const_value args[NIR_ALU_MAX_INPUTS][NIR_MAX_VEC_COMPONENTS] = {
                0};
            nir_const_value *srcs[NIR_ALU_MAX_INPUTS];
            unsigned bits =
                nir_alu_type_get_type_size(nir_op_infos[alu->op].output_type)
                    ? 0
                    : alu->def.bit_size;
            for (unsigned s = 0; s < nir_op_infos[alu->op].num_inputs; s++) {
               srcs[s] = args[s];
               if (!bits && !nir_alu_type_get_type_size(
                                nir_op_infos[alu->op].input_types[s]))
                  bits = alu->src[s].src.ssa->bit_size;
               for (unsigned c = 0;
                    c < nir_ssa_alu_instr_src_components(alu, s); c++)
                  args[s][c] = values[alu->src[s].src.ssa->index]
                                     [alu->src[s].swizzle[c]];
            }
            nir_eval_const_opcode(alu->op, values[alu->def.index], NULL,
                                  alu->def.num_components, bits ? bits : 32,
                                  srcs,
                                  nir->info.float_controls_execution_mode);
         } else if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_vulkan_resource_index:
               values[intr->def.index][0].u32 = nir_intrinsic_binding(intr);
               break;
            case nir_intrinsic_load_vulkan_descriptor:
               memcpy(values[intr->def.index], values[intr->src[0].ssa->index],
                      intr->def.num_components * sizeof(nir_const_value));
               break;
            case nir_intrinsic_tensor_read_arm:
            case nir_intrinsic_tensor_write_arm: {
               bool read = intr->intrinsic == nir_intrinsic_tensor_read_arm;
               assert(values[intr->src[0].ssa->index][0].u32 == (read ? 0 : 1));
               unsigned n = read ? intr->def.num_components
                                 : intr->src[2].ssa->num_components;
               assert(nir_num_components_valid(n));
               assert(nir_intrinsic_access(intr) & ACCESS_NON_TEMPORAL);
               assert(intr->src[1].ssa->num_components == 2);
               assert((read ? intr->def.bit_size
                            : intr->src[2].ssa->bit_size) == bytes * 8);
               unsigned row = values[intr->src[1].ssa->index][0].u32;
               unsigned column = values[intr->src[1].ssa->index][1].u32;
               for (unsigned c = 0; c < n; c++) {
                  bool inside = row < 3 && (uint64_t)column + c < width;
                  unsigned index = (row * width + column + c) * bytes;
                  if (read) {
                     if (inside)
                        memcpy(&values[intr->def.index][c], input + index,
                               bytes);
                     else
                        values[intr->def.index][c] =
                            values[intr->src[2].ssa->index][c];
                     (*read_elements)++;
                  } else {
                     assert(inside);
                     memcpy(output + index, &values[intr->src[2].ssa->index][c],
                            bytes);
                     (*write_elements)++;
                  }
               }
               break;
            }
            case nir_intrinsic_barrier:
               (*barriers)++;
               break;
            default:
               fprintf(stderr, "unhandled intrinsic %s\n",
                       nir_intrinsic_infos[intr->intrinsic].name);
               abort();
            }
         } else if (instr->type == nir_instr_type_jump) {
            assert(nir_instr_as_jump(instr)->type == nir_jump_return);
         } else {
            fprintf(stderr, "unhandled instruction %u\n", instr->type);
            abort();
         }
      }
   }
   free(values);
}

static void
check(unsigned count, unsigned bits, bool floating, bool sign, bool scalar,
      bool oob, bool visibility)
{
   unsigned bytes = MAX2(bits, 8) / 8, width = count + 8;
   unsigned start = oob ? width - count / 2 : 3;
   struct module m = make_module(count, bits, floating, sign, scalar, start, 3,
                                 visibility, 2);
   struct spirv_capabilities caps;
   memset(&caps, 1, sizeof(caps));
   struct spirv_to_nir_options options = {
       .environment = NIR_SPIRV_VULKAN,
       .capabilities = &caps,
       .tensor_addr_format = nir_address_format_vec2_index_32bit_offset,
       .skip_os_break_in_debug_build = true,
   };
   const nir_shader_compiler_options nir_options = {0};
   nir_shader *nir = spirv_to_nir(m.words, m.count, NULL, MESA_SHADER_COMPUTE,
                                  "main", &options, &nir_options);
   assert(nir);
   nir_validate_shader(nir, "tensor array access");
   unsigned size = 3 * width * bytes;
   uint8_t *input = malloc(size), *output = malloc(size),
           *expected = malloc(size);
   for (unsigned i = 0; i < size; i++)
      input[i] = bits == 1 ? i % 3 ? i * 37 + 17 : 0 : i * 37 + 17;
   memset(output, 0xa5, size);
   memset(expected, 0xa5, size);
   uint64_t fallback = bits == 1    ? 0xff
                       : bits == 8  ? 0xc1
                       : bits == 16 ? 0x37c1
                       : bits == 32 ? 0xa55a37c1
                                    : UINT64_C(0x45b27e91a55a37c1);
   for (unsigned i = 0; i < count; i++) {
      uint64_t value = fallback;
      if ((uint64_t)start + i < width)
         memcpy(&value, input + (width + start + i) * bytes, bytes);
      if (bits == 1)
         value = value ? 0xff : 0;
      memcpy(expected + (width + 3 + i) * bytes, &value, bytes);
   }
   unsigned reads = 0, writes = 0, barriers = 0;
   execute(nir, input, output, width, bytes, &reads, &writes, &barriers);
   assert(reads == count && writes == count);
   assert(barriers == (visibility ? 2 : 0));
   assert(!memcmp(output, expected, size));
   free(input);
   free(output);
   free(expected);
   ralloc_free(nir);
}

static void
check_invalid(bool write)
{
   struct module m = make_module(12, 8, false, true, false, 3, 3, true, 2);
   for (unsigned i = 5; i < m.count; i += m.words[i] >> 16)
      if ((m.words[i] & 0xffff) ==
          (unsigned)(write ? SpvOpTensorWriteARM : SpvOpTensorReadARM))
         m.words[i + (write ? 4 : 5)] &=
             ~SpvTensorOperandsNonPrivateElementARMMask;
   struct spirv_capabilities caps;
   memset(&caps, 1, sizeof(caps));
   struct spirv_to_nir_options options = {
       .environment = NIR_SPIRV_VULKAN,
       .capabilities = &caps,
       .tensor_addr_format = nir_address_format_vec2_index_32bit_offset,
       .skip_os_break_in_debug_build = true,
   };
   const nir_shader_compiler_options nir_options = {0};
   nir_shader *nir = spirv_to_nir(m.words, m.count, NULL, MESA_SHADER_COMPUTE,
                                  "main", &options, &nir_options);
   assert(!nir);
}

int
main(int argc, char **argv)
{
   glsl_type_singleton_init_or_ref();
   unsigned cases = 0;
   unsigned counts[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  11, 12,
                        13, 15, 16, 17, 23, 31, 32, 33, 63, 64, 65};
   for (unsigned kind = 0; kind < 12; kind++) {
      unsigned bits = kind == 11 ? 1
                      : kind < 4 ? 8u << kind
                      : kind < 8 ? 8u << (kind - 4)
                                 : 16u << (kind - 8);
      bool floating = kind >= 8 && kind < 11;
      bool sign = kind < 4;
      for (unsigned i = 0; i < ARRAY_SIZE(counts); i++) {
         for (unsigned flags = 0; flags < 4; flags++) {
            check(counts[i], bits, floating, sign, false, flags & 1, flags & 2);
            cases++;
         }
      }
      check(1, bits, floating, sign, true, false, false);
      cases++;
   }
   check_invalid(false);
   check_invalid(true);
   if (argc == 2) {
      unsigned probes[] = {1,  5,  6,  7,  8,  11, 12, 13, 15,
                           16, 17, 31, 32, 33, 63, 64, 65};
      for (unsigned i = 0; i < ARRAY_SIZE(probes); i++) {
         struct module m =
             make_module(probes[i], 8, false, true, false, 3, 3, false, 2);
         char name[4096];
         snprintf(name, sizeof(name), "%s/copy-i8-%u.spv", argv[1], probes[i]);
         FILE *file = fopen(name, "wb");
         assert(file);
         assert(fwrite(m.words, 4, m.count, file) == m.count);
         fclose(file);
      }
   }
   glsl_type_singleton_decref();
   printf(
       "tensor access translation and CPU semantics: %u positive, 2 negative "
       "cases passed\n",
       cases);
   return 0;
}
