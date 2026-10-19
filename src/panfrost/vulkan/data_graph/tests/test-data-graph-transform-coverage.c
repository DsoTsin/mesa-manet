#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/spirv/spirv.h"
#include "util/ralloc.h"
#include "nir_builder.h"
#include "panvk_data_graph.h"
#include "vk_format.h"

#ifdef PANVK_GRAPH_TEST_ALLOC_FAILURE
static size_t allocation_calls, fail_at = SIZE_MAX;
static bool allocation_tracking, allocation_armed;
void *__real_malloc(size_t size);
void *__real_calloc(size_t count, size_t size);
void *__wrap_malloc(size_t size);
void *__wrap_calloc(size_t count, size_t size);

void *
__wrap_malloc(size_t size)
{
   return allocation_armed && allocation_calls++ == fail_at
             ? NULL
             : __real_malloc(size);
}

void *
__wrap_calloc(size_t count, size_t size)
{
   return allocation_armed && allocation_calls++ == fail_at
             ? NULL
             : __real_calloc(count, size);
}
#endif

struct tensor_shape {
   unsigned rank;
   uint32_t dims[4];
};

struct test_op {
   enum panvk_graph_op op;
   unsigned input;
   struct tensor_shape shape;
   unsigned permutation[4];
   unsigned attribute;
};

struct graph_case {
   VkFormat format;
   struct tensor_shape input;
   struct test_op ops[3];
   unsigned op_count;
   unsigned outputs[2];
   unsigned output_count;
   unsigned passes;
   bool constant;
   bool wide;
   bool strided;
   bool unsigned_attrs;
};

struct module {
   uint32_t words[4096];
   unsigned count;
   uint32_t next;
   uint32_t attrs[3];
};

static uint32_t
new_id(struct module *m)
{
   return m->next++;
}

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

static void
emit_string(struct module *m, SpvOp op, uint32_t id, const char *s)
{
   unsigned words = DIV_ROUND_UP(strlen(s) + 1, 4);
   m->words[m->count++] = ((words + 1 + (id != 0)) << 16) | op;
   if (id)
      m->words[m->count++] = id;
   memset(m->words + m->count, 0, words * 4);
   memcpy(m->words + m->count, s, strlen(s));
   m->count += words;
}

static uint32_t
constant(struct module *m, uint32_t type, uint32_t value, bool wide)
{
   uint32_t id = new_id(m);
   if (wide)
      emit(m, SpvOpConstant, 4, type, id, value, 0u);
   else
      emit(m, SpvOpConstant, 3, type, id, value);
   return id;
}

static uint32_t
array(struct module *m, uint32_t u32, uint32_t type, unsigned count,
      const uint32_t *values, bool wide)
{
   uint32_t length = constant(m, u32, count, false);
   uint32_t array_type = new_id(m), id = new_id(m), ids[4];
   emit(m, SpvOpTypeArray, 3, array_type, type, length);
   for (unsigned i = 0; i < count; i++)
      ids[i] = constant(m, type, values[i], wide);
   m->words[m->count++] = ((3 + count) << 16) | SpvOpConstantComposite;
   m->words[m->count++] = array_type;
   m->words[m->count++] = id;
   for (unsigned i = 0; i < count; i++)
      m->words[m->count++] = ids[i];
   return id;
}

static uint32_t
tensor_type(struct module *m, uint32_t u32, uint32_t type,
            const struct tensor_shape *shape)
{
   uint32_t dims = array(m, u32, u32, shape->rank, shape->dims, false);
   uint32_t rank = constant(m, u32, shape->rank, false);
   uint32_t id = new_id(m);
   emit(m, SpvOpTypeTensorARM, 4, id, type, rank, dims);
   return id;
}

static void
attribute_tensor(struct module *m, uint32_t u32, uint32_t type, unsigned count,
                 const uint32_t *values, uint32_t *id)
{
   const struct tensor_shape shape = {1, {count}};
   uint32_t tensor = tensor_type(m, u32, type, &shape), ids[4];
   for (unsigned i = 0; i < count; i++)
      ids[i] = constant(m, type, values[i], false);
   *id = new_id(m);
   m->words[m->count++] = ((3 + count) << 16) | SpvOpConstantComposite;
   m->words[m->count++] = tensor;
   m->words[m->count++] = *id;
   for (unsigned i = 0; i < count; i++)
      m->words[m->count++] = ids[i];
}

static void
build_module(const struct graph_case *c, struct module *m)
{
   *m = (struct module){.count = 5, .next = 1};
   bool has_mul = false, has_clamp = false;
   for (unsigned i = 0; i < c->op_count; i++) {
      has_mul |= c->ops[i].op == PANVK_GRAPH_OP_MUL;
      has_clamp |= c->ops[i].op == PANVK_GRAPH_OP_CLAMP;
   }
   m->words[0] = SpvMagicNumber;
   m->words[1] = 0x10600;
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityShader);
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityVulkanMemoryModel);
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityGraphARM);
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityTensorsARM);
   if (c->format == VK_FORMAT_R8_SINT || has_mul)
      emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityInt8);
   if (c->wide)
      emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityInt64);
   emit_string(m, SpvOpExtension, 0, "SPV_ARM_graph");
   emit_string(m, SpvOpExtension, 0, "SPV_ARM_tensors");
   uint32_t tosa = new_id(m);
   emit_string(m, SpvOpExtInstImport, tosa, "TOSA.001000.1");
   emit(m, SpvOpMemoryModel, 2, (uint32_t)SpvAddressingModelLogical,
        (uint32_t)SpvMemoryModelVulkan);
   uint32_t variables[3], input_count = !c->constant;
   unsigned variable_count = input_count + c->output_count;
   for (unsigned i = 0; i < variable_count; i++) {
      variables[i] = new_id(m);
      emit(m, SpvOpDecorate, 3, variables[i],
           (uint32_t)SpvDecorationDescriptorSet, 0u);
      emit(m, SpvOpDecorate, 3, variables[i], (uint32_t)SpvDecorationBinding,
           i);
   }
   uint32_t u32 = new_id(m), type = u32;
   emit(m, SpvOpTypeInt, 3, u32, 32u, 0u);
   if (c->format != VK_FORMAT_R32_UINT) {
      type = new_id(m);
      if (c->format == VK_FORMAT_R32_SFLOAT)
         emit(m, SpvOpTypeFloat, 2, type, 32u);
      else
         emit(m, SpvOpTypeInt, 3, type,
              c->format == VK_FORMAT_R8_SINT ? 8u : 32u, 1u);
   }
   uint32_t attr_type = c->unsigned_attrs ? u32 : type;
   if (c->wide || (!c->unsigned_attrs && c->format != VK_FORMAT_R32_SINT)) {
      attr_type = new_id(m);
      emit(m, SpvOpTypeInt, 3, attr_type, c->wide ? 64u : 32u,
           c->unsigned_attrs ? 0u : 1u);
   }
   uint32_t types[4], values[4];
   types[0] = tensor_type(m, u32, type, &c->input);
   values[0] = new_id(m);
   for (unsigned i = 0; i < c->op_count; i++) {
      const struct test_op *op = &c->ops[i];
      types[i + 1] = tensor_type(m, u32, type, &op->shape);
      values[i + 1] = new_id(m);
      if (op->op == PANVK_GRAPH_OP_RESHAPE ||
          op->op == PANVK_GRAPH_OP_TRANSPOSE) {
         const uint32_t *attribute = op->op == PANVK_GRAPH_OP_TRANSPOSE
                                        ? op->permutation
                                        : op->shape.dims;
         if (c->unsigned_attrs && !c->wide)
            attribute_tensor(m, u32, attr_type, op->shape.rank, attribute,
                             &m->attrs[i]);
         else
            m->attrs[i] =
               array(m, u32, attr_type, op->shape.rank, attribute, c->wide);
      }
   }
   for (unsigned i = 0; i < variable_count; i++) {
      uint32_t t = types[i < input_count ? 0 : c->outputs[i - input_count]];
      uint32_t ptr = new_id(m);
      emit(m, SpvOpTypePointer, 3, ptr,
           (uint32_t)SpvStorageClassUniformConstant, t);
      emit(m, SpvOpVariable, 3, ptr, variables[i],
           (uint32_t)SpvStorageClassUniformConstant);
   }
   if (c->constant)
      emit(m, SpvOpGraphConstantARM, 3, types[0], values[0], 7u);
   uint32_t indices[3] = {constant(m, u32, 0, false),
                          constant(m, u32, 1, false),
                          constant(m, u32, 2, false)};
   uint32_t shift = 0, bounds[2] = {0};
   if (has_mul) {
      uint32_t i8 = new_id(m);
      emit(m, SpvOpTypeInt, 3, i8, 8u, 1u);
      const struct tensor_shape shape = {1, {1}};
      uint32_t t = tensor_type(m, u32, i8, &shape);
      shift = new_id(m);
      emit(m, SpvOpConstantNull, 2, t, shift);
   }
   if (has_clamp) {
      uint32_t low = c->format == VK_FORMAT_R32_SFLOAT ? 0xc2800000 : 0x80000000;
      uint32_t high = c->format == VK_FORMAT_R32_SFLOAT ? 0x42800000 : 0x7fffffff;
      bounds[0] = constant(m, type, low, false);
      bounds[1] = constant(m, type, high, false);
   }
   uint32_t graph_type = new_id(m), graph = new_id(m);
   m->words[m->count++] = ((3 + variable_count) << 16) | SpvOpTypeGraphARM;
   m->words[m->count++] = graph_type;
   m->words[m->count++] = input_count;
   if (input_count)
      m->words[m->count++] = types[0];
   for (unsigned i = 0; i < c->output_count; i++)
      m->words[m->count++] = types[c->outputs[i]];
   m->words[m->count++] =
      ((4 + variable_count) << 16) | SpvOpGraphEntryPointARM;
   m->words[m->count++] = graph;
   m->words[m->count++] = 0x6e69616d;
   m->words[m->count++] = 0;
   for (unsigned i = 0; i < variable_count; i++)
      m->words[m->count++] = variables[i];
   emit(m, SpvOpGraphARM, 2, graph_type, graph);
   if (input_count)
      emit(m, SpvOpGraphInputARM, 3, types[0], values[0], indices[0]);
   for (unsigned i = 0; i < c->op_count; i++) {
      const struct test_op *op = &c->ops[i];
      uint32_t a = values[op->input];
      if (op->op == PANVK_GRAPH_OP_CLAMP) {
         emit(m, SpvOpExtInst, 8, types[i + 1], values[i + 1], tosa, op->op,
              bounds[0], bounds[1], indices[op->attribute + 1], a);
      } else if (op->op == PANVK_GRAPH_OP_MUL) {
         emit(m, SpvOpExtInst, 7, types[i + 1], values[i + 1], tosa, op->op, a,
              a, shift);
      } else {
         emit(m, SpvOpExtInst, 6, types[i + 1], values[i + 1], tosa, op->op,
              op->op == PANVK_GRAPH_OP_TRANSPOSE ? m->attrs[i] : a,
              op->op == PANVK_GRAPH_OP_RESHAPE ? m->attrs[i] : a);
      }
   }
   for (unsigned i = 0; i < c->output_count; i++)
      emit(m, SpvOpGraphSetOutputARM, 2, values[c->outputs[i]], indices[i]);
   emit(m, SpvOpGraphEndARM, 0);
   m->words[3] = m->next;
}

static unsigned
elements(const struct tensor_shape *s)
{
   unsigned count = 1;
   for (unsigned d = 0; d < s->rank; d++)
      count *= s->dims[d];
   return count;
}

static void
unflatten(unsigned index, const struct tensor_shape *s, uint32_t *coords)
{
   for (int d = s->rank - 1; d >= 0; d--) {
      coords[d] = index % s->dims[d];
      index /= s->dims[d];
   }
   assert(!index);
}

static unsigned
flatten(const uint32_t *coords, const struct tensor_shape *s)
{
   unsigned index = 0;
   for (unsigned d = 0; d < s->rank; d++) {
      assert(coords[d] < s->dims[d]);
      index = index * s->dims[d] + coords[d];
   }
   return index;
}

static uint8_t *
reference(const struct graph_case *c, uint8_t **values)
{
   unsigned bytes = vk_format_get_blocksize(c->format);
   bool arithmetic = false;
   for (unsigned i = 0; i < c->op_count; i++)
      arithmetic |= c->ops[i].op != PANVK_GRAPH_OP_RESHAPE &&
                    c->ops[i].op != PANVK_GRAPH_OP_TRANSPOSE;
   values[0] = malloc(elements(&c->input) * bytes);
   assert(values[0]);
   for (unsigned i = 0; i < elements(&c->input); i++) {
      uint32_t value = 0x9e3779b9u * i;
      if (c->format == VK_FORMAT_R32_SFLOAT) {
         if (arithmetic) {
            float f = ((int)i - 100) * 0.25f;
            memcpy(&value, &f, 4);
         } else {
            static const uint32_t special[] = {
               0,          0x80000000, 0x7fc01234, 0xffc05678,
               0x7f800000, 0xff800000, 1,          0x807fffff,
            };
            value = i < ARRAY_SIZE(special) ? special[i] : value;
         }
      }
      memcpy(values[0] + i * bytes, &value, bytes);
   }
   for (unsigned i = 0; i < c->op_count; i++) {
      const struct test_op *op = &c->ops[i];
      const struct tensor_shape *source =
         op->input ? &c->ops[op->input - 1].shape : &c->input;
      unsigned count = elements(source);
      values[i + 1] = malloc(count * bytes);
      assert(values[i + 1] && count == elements(&op->shape));
      if (op->op == PANVK_GRAPH_OP_TRANSPOSE) {
         for (unsigned input = 0; input < count; input++) {
            uint32_t coords[4], output[4];
            unflatten(input, source, coords);
            for (unsigned d = 0; d < source->rank; d++)
               output[d] = coords[op->permutation[d]];
            unsigned dest = flatten(output, &op->shape);
            memcpy(values[i + 1] + dest * bytes,
                   values[op->input] + input * bytes, bytes);
         }
      } else if (op->op == PANVK_GRAPH_OP_RESHAPE) {
         memcpy(values[i + 1], values[op->input], count * bytes);
      } else {
         assert((op->op == PANVK_GRAPH_OP_ADD || op->op == PANVK_GRAPH_OP_SUB ||
                 op->op == PANVK_GRAPH_OP_MUL ||
                 op->op == PANVK_GRAPH_OP_CLAMP) &&
                bytes == 4);
         for (unsigned j = 0; j < count; j++) {
            uint32_t value;
            memcpy(&value, values[op->input] + j * 4, 4);
            if (c->format == VK_FORMAT_R32_SFLOAT) {
               float f;
               memcpy(&f, &value, 4);
               if (op->op == PANVK_GRAPH_OP_ADD)
                  f += f;
               else if (op->op == PANVK_GRAPH_OP_SUB)
                  f -= f;
               else if (op->op == PANVK_GRAPH_OP_MUL)
                  f *= f;
               memcpy(&value, &f, 4);
            } else {
               assert(op->op != PANVK_GRAPH_OP_MUL);
               if (op->op == PANVK_GRAPH_OP_ADD)
                  value += value;
               else if (op->op == PANVK_GRAPH_OP_SUB)
                  value = 0;
            }
            memcpy(values[i + 1] + j * 4, &value, 4);
         }
      }
   }
   return values[0];
}

static VkResult
parse_case(const struct graph_case *c, const struct module *m, const void *data,
           struct panvk_data_graph **g)
{
   int64_t dims[3][4];
   int64_t constant_dims[4];
   int64_t strides[4];
   unsigned bytes = vk_format_get_blocksize(c->format);
   uint64_t stride = bytes * 3, extent = bytes;
   for (int d = c->input.rank - 1; d >= 0; d--) {
      strides[d] = stride;
      extent += (c->input.dims[d] - 1) * stride;
      stride = stride * c->input.dims[d] + bytes * 5;
      constant_dims[d] = c->input.dims[d];
   }
   uint8_t *strided = NULL;
   if (data && c->strided) {
      strided = calloc(extent, 1);
      assert(strided);
      for (unsigned i = 0; i < elements(&c->input); i++) {
         uint32_t coords[4];
         unflatten(i, &c->input, coords);
         uint64_t offset = 0;
         for (unsigned d = 0; d < c->input.rank; d++)
            offset += coords[d] * strides[d];
         memcpy(strided + offset, (const uint8_t *)data + i * bytes, bytes);
      }
   }
   VkTensorDescriptionARM descs[3],
      constant_desc = {
         .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
         .format = c->format,
         .tiling = VK_TENSOR_TILING_LINEAR_ARM,
         .dimensionCount = c->input.rank,
         .pDimensions = constant_dims,
         .pStrides = c->strided ? strides : NULL,
         .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
      };
   VkDataGraphPipelineConstantARM value = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM,
      .pNext = &constant_desc,
      .id = 7,
      .pConstantData = strided ? strided : data,
   };
   VkDataGraphPipelineShaderModuleCreateInfoARM module = {
      .sType =
         VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
      .pName = "main",
      .constantCount = c->constant,
      .pConstants = &value,
   };
   unsigned inputs = !c->constant, count = inputs + c->output_count;
   VkDataGraphPipelineResourceInfoARM resources[3];
   for (unsigned i = 0; i < count; i++) {
      unsigned index = i < inputs ? 0 : c->outputs[i - inputs];
      const struct tensor_shape *s =
         index ? &c->ops[index - 1].shape : &c->input;
      for (unsigned d = 0; d < s->rank; d++)
         dims[i][d] = s->dims[d];
      descs[i] = constant_desc;
      descs[i].pStrides = NULL;
      descs[i].dimensionCount = s->rank;
      descs[i].pDimensions = dims[i];
      resources[i] = (VkDataGraphPipelineResourceInfoARM){
         .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
         .pNext = &descs[i],
         .binding = i,
      };
   }
   VkDataGraphPipelineCreateInfoARM info = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
      .resourceInfoCount = count,
      .pResourceInfos = resources,
   };
#ifdef PANVK_GRAPH_TEST_ALLOC_FAILURE
   allocation_armed = allocation_tracking;
   allocation_calls = 0;
#endif
   VkResult result =
      panvk_data_graph_parse(NULL, m->words, m->count * 4, &info, &module, g);
#ifdef PANVK_GRAPH_TEST_ALLOC_FAILURE
   allocation_armed = false;
#endif
   free(strided);
   return result;
}

static unsigned
descriptor_index(nir_def *def)
{
   nir_intrinsic_instr *load = nir_instr_as_intrinsic(nir_def_instr(def));
   assert(load->intrinsic == nir_intrinsic_load_vulkan_descriptor);
   nir_intrinsic_instr *index =
      nir_instr_as_intrinsic(nir_def_instr(load->src[0].ssa));
   assert(index->intrinsic == nir_intrinsic_vulkan_resource_index);
   assert(nir_src_is_const(index->src[0]));
   return nir_src_as_uint(index->src[0]);
}

static unsigned
tensor_index(const struct panvk_graph_value *v, nir_src coords)
{
   assert(nir_src_is_const(coords) && coords.ssa->num_components == v->rank);
   unsigned index = 0;
   for (unsigned d = 0; d < v->rank; d++) {
      unsigned coord = nir_src_comp_as_uint(coords, d);
      assert(coord < v->dims[d]);
      index = index * v->dims[d] + coord;
   }
   return index;
}

static void
evaluate_invocation(const struct panvk_data_graph *g, unsigned pass,
                    nir_shader *template, unsigned invocation,
                    uint8_t **buffers)
{
   nir_shader *nir = nir_shader_clone(NULL, template);
   nir_builder b = nir_builder_create(nir_shader_get_entrypoint(nir));
   nir_foreach_block(block, b.impl)
   {
      nir_foreach_instr_safe(instr, block)
      {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_global_invocation_id) {
            b.cursor = nir_before_instr(instr);
            nir_def *gid = nir_imm_ivec3(&b, invocation % (65535 * 64),
                                         invocation / (65535 * 64), 0);
            nir_def_rewrite_uses(&intr->def, gid);
            nir_instr_remove(instr);
         }
      }
   }
   while (nir_opt_constant_folding(nir)) {
   }
   nir_opt_dead_cf(nir);
   unsigned writes = 0;
   nir_foreach_block(block, b.impl)
   {
      nir_foreach_instr_safe(instr, block)
      {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_tensor_read_arm)
            continue;
         unsigned id = descriptor_index(intr->src[0].ssa);
         assert(id < g->count && g->values[id].live && buffers[id]);
         unsigned bytes = intr->def.bit_size / 8;
         unsigned index = tensor_index(&g->values[id], intr->src[1]);
         uint32_t value = 0;
         memcpy(&value, buffers[id] + index * bytes, bytes);
         b.cursor = nir_before_instr(instr);
         nir_def_rewrite_uses(&intr->def,
                              nir_imm_intN_t(&b, value, intr->def.bit_size));
         nir_instr_remove(instr);
      }
   }
   while (nir_opt_constant_folding(nir)) {
   }
   nir_foreach_block(block, b.impl)
   {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_tensor_write_arm)
            continue;
         unsigned id = descriptor_index(intr->src[0].ssa);
         assert(id == g->passes[pass] && buffers[id]);
         unsigned index = tensor_index(&g->values[id], intr->src[1]);
         assert(nir_src_is_const(intr->src[2]));
         uint32_t value = nir_src_as_uint(intr->src[2]);
         memcpy(buffers[id] + index * (intr->src[2].ssa->bit_size / 8), &value,
                intr->src[2].ssa->bit_size / 8);
         writes++;
      }
   }
   assert(writes == (invocation < g->values[g->passes[pass]].elements));
   ralloc_free(nir);
}

static unsigned graphs, invocations, rejects;
static const char *dump_directory;

static void
run_case(const struct graph_case *c)
{
   struct module module;
   build_module(c, &module);
   uint8_t *reference_values[4];
   uint8_t *input = reference(c, reference_values);
   struct panvk_data_graph *g;
   assert(parse_case(c, &module, input, &g) == VK_SUCCESS);
   if (g->pass_count != c->passes)
      fprintf(
         stderr,
         "graph=%u format=%u rank=%u constant=%u wide=%u ops=%u,%u,%u outputs=%u passes=%u expected=%u\n",
         graphs, c->format, c->input.rank, c->constant, c->wide, c->ops[0].op,
         c->ops[1].op, c->ops[2].op, c->output_count, g->pass_count, c->passes);
   assert(g->pass_count == c->passes);
   if (dump_directory) {
      char name[1024];
      snprintf(name, sizeof(name), "%s/%06u.spv", dump_directory, graphs);
      FILE *file = fopen(name, "wb");
      assert(file);
      assert(fwrite(module.words, 4, module.count, file) == module.count);
      assert(!fclose(file));
   }
   uint8_t **buffers = calloc(g->count, sizeof(*buffers));
   uint8_t *transient = malloc(MAX2(g->transient_size, 1));
   uint8_t *constants = malloc(MAX2(g->constant_size, 1));
   assert(buffers && transient && constants);
   memset(transient, 0xa5, MAX2(g->transient_size, 1));
   memset(constants, 0x5a, MAX2(g->constant_size, 1));
   for (unsigned i = 0; i < g->count; i++) {
      const struct panvk_graph_value *v = &g->values[i];
      if (!v->live || v->fused_into != UINT32_MAX)
         continue;
      switch (v->kind) {
      case PANVK_GRAPH_INPUT:
         buffers[i] = input;
         break;
      case PANVK_GRAPH_CONSTANT:
         if (v->elements > 1) {
            assert(v->offset + v->size <= g->constant_size);
            buffers[i] = constants + v->offset;
            memcpy(buffers[i], v->data, v->size);
         }
         break;
      case PANVK_GRAPH_OPERATION:
         assert(v->offset + v->size <= g->transient_size);
         buffers[i] = transient + v->offset;
         break;
      case PANVK_GRAPH_OUTPUT:
         buffers[i] = malloc(v->size);
         assert(buffers[i]);
         memset(buffers[i], 0x3c, v->size);
         break;
      }
   }
   for (unsigned pass = 0; pass < g->pass_count; pass++) {
      const struct panvk_graph_value *v = &g->values[g->passes[pass]];
      nir_shader *nir = panvk_data_graph_shader(g, pass);
      assert(nir);
      assert(!nir->info.internal);
      for (unsigned index = 0; index < v->elements; index++) {
         evaluate_invocation(g, pass, nir, index, buffers);
         invocations++;
      }
      evaluate_invocation(g, pass, nir, v->elements, buffers);
      evaluate_invocation(g, pass, nir, ALIGN_POT(v->elements, 64) + 63,
                          buffers);
      invocations += 2;
      ralloc_free(nir);
   }
   unsigned outputs = 0;
   for (unsigned i = 0; i < g->count; i++) {
      const struct panvk_graph_value *v = &g->values[i];
      if (v->kind == PANVK_GRAPH_OUTPUT) {
         assert(outputs < c->output_count);
         assert(!memcmp(buffers[i], reference_values[c->outputs[outputs]],
                        v->size));
         outputs++;
         free(buffers[i]);
      }
   }
   assert(outputs == c->output_count);
   for (unsigned i = 0; i <= c->op_count; i++)
      free(reference_values[i]);
   free(buffers);
   free(transient);
   free(constants);
   ralloc_free(g);
   graphs++;
}

static struct test_op
transpose(unsigned input, const struct tensor_shape *shape,
          const unsigned *perm)
{
   struct test_op op = {.op = PANVK_GRAPH_OP_TRANSPOSE,
                        .input = input,
                        .shape.rank = shape->rank};
   for (unsigned d = 0; d < shape->rank; d++) {
      op.permutation[d] = perm[d];
      op.shape.dims[d] = shape->dims[perm[d]];
   }
   return op;
}

static struct test_op
reshape(unsigned input, const struct tensor_shape *shape, unsigned rank,
        bool distribute)
{
   struct test_op op = {.op = PANVK_GRAPH_OP_RESHAPE,
                        .input = input,
                        .shape.rank = rank};
   unsigned remaining = elements(shape);
   for (unsigned d = 0; d < rank; d++) {
      unsigned extent = 1;
      if (distribute) {
         for (unsigned factor = 2; factor <= remaining; factor++) {
            if (remaining % factor == 0) {
               extent = factor;
               break;
            }
         }
      }
      op.shape.dims[d] = extent;
      remaining /= extent;
   }
   op.shape.dims[0] *= remaining;
   return op;
}

static bool
next_permutation(unsigned *p, unsigned rank)
{
   int i = rank - 2;
   while (i >= 0 && p[i] > p[i + 1])
      i--;
   if (i < 0)
      return false;
   unsigned j = rank - 1;
   while (p[j] < p[i])
      j--;
   unsigned tmp = p[i];
   p[i] = p[j];
   p[j] = tmp;
   for (unsigned left = i + 1, right = rank - 1; left < right;
        left++, right--) {
      tmp = p[left];
      p[left] = p[right];
      p[right] = tmp;
   }
   return true;
}

static void
permutation_cases(void)
{
   const VkFormat formats[] = {VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT,
                               VK_FORMAT_R32_UINT, VK_FORMAT_R32_SFLOAT};
   const uint32_t dimensions[3][4] = {{1, 1, 1, 1},
                                      {2, 3, 5, 7},
                                      {65, 1, 1, 1}};
   for (unsigned rank = 1; rank <= 4; rank++) {
      unsigned perm[] = {0, 1, 2, 3};
      do {
         for (unsigned f = 0; f < ARRAY_SIZE(formats); f++) {
            for (unsigned variant = 0; variant < ARRAY_SIZE(dimensions);
                 variant++) {
               for (unsigned flags = 0; flags < 8; flags++) {
                  if ((flags & 4) && variant != 1)
                     continue;
                  struct graph_case c = {
                     .format = formats[f],
                     .input.rank = rank,
                     .op_count = 1,
                     .outputs = {1},
                     .output_count = 1,
                     .passes = 1,
                     .constant = flags & 1,
                     .wide = flags & 2,
                     .strided = (flags & 1) && variant == 2,
                     .unsigned_attrs = flags & 4,
                  };
                  memcpy(c.input.dims, dimensions[variant],
                         sizeof(c.input.dims));
                  c.ops[0] = transpose(0, &c.input, perm);
                  run_case(&c);
               }
            }
            for (unsigned flags = 0; flags < 4; flags++) {
               struct graph_case c = {
                  .format = formats[f],
                  .input.rank = rank,
                  .op_count = 2,
                  .outputs = {2},
                  .output_count = 1,
                  .constant = flags & 1,
                  .wide = flags & 2,
               };
               memcpy(c.input.dims, dimensions[1], sizeof(c.input.dims));
               c.ops[0] = transpose(0, &c.input, perm);
               c.ops[1] = reshape(1, &c.ops[0].shape, rank % 4 + 1, true);
               c.passes = c.constant ? 1 : 2;
               run_case(&c);
               unsigned inverse[4];
               for (unsigned d = 0; d < rank; d++)
                  inverse[perm[d]] = d;
               c.ops[1] = transpose(1, &c.ops[0].shape, inverse);
               run_case(&c);
               c.output_count = 2;
               c.outputs[0] = 1;
               c.outputs[1] = 2;
               c.passes = c.constant ? 2 : 3;
               run_case(&c);
               if (c.format != VK_FORMAT_R8_SINT) {
                  c.output_count = 1;
                  c.outputs[0] = 2;
                  const enum panvk_graph_op pointwise[] = {
                     PANVK_GRAPH_OP_ADD, PANVK_GRAPH_OP_SUB,
                     PANVK_GRAPH_OP_CLAMP, PANVK_GRAPH_OP_MUL};
                  for (unsigned p = 0; p < ARRAY_SIZE(pointwise); p++) {
                     if (pointwise[p] == PANVK_GRAPH_OP_MUL &&
                         c.format != VK_FORMAT_R32_SFLOAT)
                        continue;
                     c.ops[1] = (struct test_op){.op = pointwise[p],
                                                 .input = 1,
                                                 .shape = c.ops[0].shape};
                     c.passes = 1;
                     run_case(&c);
                     if (pointwise[p] == PANVK_GRAPH_OP_CLAMP) {
                        c.ops[1].attribute = 1;
                        run_case(&c);
                     }
                     c.op_count = 3;
                     c.ops[2] = reshape(1, &c.ops[0].shape, 1, false);
                     run_case(&c);
                     c.op_count = 2;
                  }
                  c.ops[0] = (struct test_op){.op = PANVK_GRAPH_OP_ADD,
                                              .input = 0,
                                              .shape = c.input};
                  c.ops[1] = transpose(1, &c.input, perm);
                  c.passes = 2;
                  run_case(&c);
               }
            }
         }
      } while (next_permutation(perm, rank));
   }
}

static void
reshape_cases(void)
{
   const VkFormat formats[] = {VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT,
                               VK_FORMAT_R32_UINT, VK_FORMAT_R32_SFLOAT};
   for (unsigned input_rank = 1; input_rank <= 4; input_rank++) {
      for (unsigned output_rank = 1; output_rank <= 4; output_rank++) {
         for (unsigned f = 0; f < ARRAY_SIZE(formats); f++) {
            for (unsigned flags = 0; flags < 16; flags++) {
               struct graph_case c = {
                  .format = formats[f],
                  .input = {input_rank, {2, 3, 5, 7}},
                  .op_count = 1,
                  .outputs = {1},
                  .output_count = 1,
                  .passes = 1,
                  .constant = flags & 1,
                  .wide = flags & 2,
                  .strided = (flags & 1) && (flags & 4),
                  .unsigned_attrs = flags & 8,
               };
               c.ops[0] = reshape(0, &c.input, output_rank, flags & 4);
               run_case(&c);
            }
         }
      }
   }
}

static uint32_t *
definition(struct module *m, uint32_t id)
{
   for (unsigned i = 5; i < m->count; i += m->words[i] >> 16) {
      unsigned op = m->words[i] & 0xffff;
      unsigned result =
         op == SpvOpConstant || op == SpvOpConstantComposite ? 2 : 1;
      if ((op == SpvOpConstant || op == SpvOpConstantComposite ||
           op == SpvOpTypeArray || op == SpvOpTypeTensorARM ||
           op == SpvOpTypeInt) &&
          m->words[i + result] == id)
         return m->words + i;
   }
   abort();
}

static void
reject_case(const struct graph_case *c, const struct module *m,
            const void *data)
{
   struct panvk_data_graph *g = (void *)1;
   assert(parse_case(c, m, data, &g) == VK_ERROR_FEATURE_NOT_PRESENT && !g);
   rejects++;
}

static void
negative_cases(void)
{
   for (unsigned rank = 1; rank <= 4; rank++) {
      for (unsigned reshape_case = 0; reshape_case < 2; reshape_case++) {
         for (unsigned flags = 0; flags < 8; flags++) {
            struct graph_case c = {
               .format = VK_FORMAT_R32_SFLOAT,
               .input = {rank, {2, 3, 5, 7}},
               .op_count = 1,
               .outputs = {1},
               .output_count = 1,
               .passes = 1,
               .constant = flags & 1,
               .wide = flags & 2,
               .unsigned_attrs = flags & 4,
            };
            unsigned perm[] = {0, 1, 2, 3};
            c.ops[0] = reshape_case ? reshape(0, &c.input, rank, true)
                                    : transpose(0, &c.input, perm);
            struct module original;
            build_module(&c, &original);
            uint8_t *values[4];
            reference(&c, values);
            for (unsigned mutation = 0; mutation < 14; mutation++) {
               struct module m = original;
               uint32_t *attr = definition(&m, m.attrs[0]);
               uint32_t *value = definition(&m, attr[3]);
               if (mutation == 0) {
                  value[3] = UINT32_MAX;
               } else if (mutation == 1) {
                  value[3] = reshape_case ? 0 : rank;
               } else if (mutation == 2) {
                  if (reshape_case || rank < 2)
                     continue;
                  attr[4] = attr[3];
               } else if (mutation == 3) {
                  if (!c.wide)
                     continue;
                  value[4] = 1;
               } else if (mutation == 4) {
                  if (!reshape_case)
                     continue;
                  value[3]++;
               } else if (mutation == 5) {
                  uint32_t *array_type = definition(&m, attr[1]);
                  uint32_t *length = definition(&m, array_type[3]);
                  length[3]++;
               } else if (mutation == 6) {
                  uint32_t *type = definition(&m, value[1]);
                  type[2] = 16;
               } else if (mutation == 7) {
                  for (unsigned i = 5; i < m.count; i += m.words[i] >> 16) {
                     if ((m.words[i] & 0xffff) == SpvOpExtInst)
                        m.words[i + (reshape_case ? 6 : 5)] = 0;
                  }
               } else if (mutation == 8) {
                  if (!c.wide)
                     continue;
                  value[3] = value[4] = UINT32_MAX;
               } else if (mutation == 9 || mutation == 10) {
                  for (unsigned i = 5; i < m.count; i += m.words[i] >> 16) {
                     if ((m.words[i] & 0xffff) == SpvOpTypeTensorARM) {
                        uint32_t *rank_value = definition(&m, m.words[i + 3]);
                        rank_value[3] = mutation == 9 ? 0 : 5;
                        break;
                     }
                  }
               } else if (mutation == 11) {
                  for (unsigned i = 5; i < m.count; i += m.words[i] >> 16) {
                     if ((m.words[i] & 0xffff) == SpvOpTypeFloat) {
                        m.words[i + 2] = 16;
                        break;
                     }
                  }
               } else if (mutation == 12) {
                  value[1] = 0;
               } else {
                  for (unsigned i = 5; i < m.count; i += m.words[i] >> 16) {
                     if ((m.words[i] & 0xffff) == SpvOpExtInst)
                        m.words[i + (reshape_case ? 6 : 5)] = attr[1];
                  }
               }
               reject_case(&c, &m, values[0]);
            }
            if (!flags) {
               for (unsigned i = 5; i < original.count;
                    i += original.words[i] >> 16) {
                  struct module m = original;
                  m.words[i] &= 0xffff;
                  reject_case(&c, &m, values[0]);
                  m.words[i] = UINT32_MAX;
                  reject_case(&c, &m, values[0]);
               }
               struct module m = original;
               m.count--;
               reject_case(&c, &m, values[0]);
            }
            for (unsigned i = 0; i <= c.op_count; i++)
               free(values[i]);
         }
      }
   }
}

#ifdef PANVK_GRAPH_TEST_ALLOC_FAILURE
static void
oom_cases(void)
{
   for (unsigned variant = 0; variant < 8; variant++) {
      struct graph_case c = {
         .format = VK_FORMAT_R32_SFLOAT,
         .input = {4, {2, 3, 5, 7}},
         .op_count = 2,
         .outputs = {1, 2},
         .output_count = 2,
         .constant = variant & 1,
         .strided = variant & 1,
         .wide = variant & 2,
         .unsigned_attrs = variant & 4,
      };
      unsigned perm[] = {3, 1, 0, 2}, inverse[] = {2, 1, 3, 0};
      c.ops[0] = transpose(0, &c.input, perm);
      c.ops[1] = transpose(1, &c.ops[0].shape, inverse);
      struct module module;
      build_module(&c, &module);
      uint8_t *values[4];
      reference(&c, values);
      struct panvk_data_graph *g;
      allocation_tracking = true;
      fail_at = SIZE_MAX;
      assert(parse_case(&c, &module, values[0], &g) == VK_SUCCESS);
      size_t total = allocation_calls;
      assert(total);
      ralloc_free(g);
      for (size_t i = 0; i < total; i++) {
         fail_at = i;
         g = (void *)1;
         VkResult result = parse_case(&c, &module, values[0], &g);
         if (result != VK_ERROR_OUT_OF_HOST_MEMORY || g)
            fprintf(stderr,
                    "OOM variant=%u allocation=%zu/%zu result=%d graph=%p\n",
                    variant, i, total, result, (void *)g);
         assert(result == VK_ERROR_OUT_OF_HOST_MEMORY && !g);
         rejects++;
         fail_at = SIZE_MAX;
         assert(parse_case(&c, &module, values[0], &g) == VK_SUCCESS);
         ralloc_free(g);
      }
      allocation_tracking = false;
      for (unsigned i = 0; i <= c.op_count; i++)
         free(values[i]);
   }
}
#endif

int
main(int argc, char **argv)
{
   assert(argc == 1 || argc == 2);
   if (argc == 2)
      dump_directory = argv[1];
#ifdef PANVK_GRAPH_TEST_ALLOC_FAILURE
   oom_cases();
#else
   permutation_cases();
   reshape_cases();
   negative_cases();
#endif
   printf("transform coverage graphs=%u invocations=%u rejected=%u\n", graphs,
          invocations, rejects);
   return 0;
}
