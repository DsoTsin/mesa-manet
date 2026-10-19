#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/spirv/spirv.h"
#include "util/ralloc.h"
#include "nir_builder.h"
#include "panvk_data_graph.h"
#include "panvk_data_graph_neural.h"
#include "vk_format.h"
#include "data-graph-nir-interpreter.h"

struct test_shape {
   unsigned rank;
   uint32_t dims[4];
};

enum source_kind {
   SOURCE_INPUT,
   SOURCE_EXTERNAL,
   SOURCE_COMPOSITE,
   SOURCE_NULL,
};

struct test_source {
   enum source_kind kind;
   struct test_shape shape;
   unsigned width;
   unsigned sign;
   VkFormat storage;
   unsigned pattern;
   bool floating;
   bool strided;
   bool missing;
   unsigned aggregate;
   unsigned malformed;
};

struct test_node {
   enum panvk_graph_op op;
   struct test_shape shape;
   unsigned width;
   unsigned sign;
   unsigned inputs[4];
   unsigned axis;
   unsigned permutation[4];
   bool floating;
};

struct test_case {
   struct test_source sources[4];
   struct test_node nodes[3];
   unsigned source_count;
   unsigned node_count;
   unsigned output;
   VkFormat output_storage;
   bool invalid;
   bool neural;
};

struct module {
   uint32_t words[8192];
   unsigned count;
   uint32_t next;
   uint32_t u32;
   uint32_t ints[2][3];
   uint32_t f32;
};

static unsigned cases, rejected, shaders, neural, constants;
static uint64_t invocations, checked;
static const char *dump_directory;

static unsigned
elements(const struct test_shape *s)
{
   unsigned count = 1;
   for (unsigned d = 0; d < s->rank; d++)
      count *= s->dims[d];
   return count;
}

static const struct test_shape *
value_shape(const struct test_case *c, unsigned value)
{
   return value < c->source_count ? &c->sources[value].shape
                                  : &c->nodes[value - c->source_count].shape;
}

static unsigned
value_width(const struct test_case *c, unsigned value)
{
   return value < c->source_count ? c->sources[value].width
                                  : c->nodes[value - c->source_count].width;
}

static uint32_t
pattern(unsigned width, unsigned mode, unsigned index)
{
   static const uint32_t i8[] = {0x80, 0xff, 0, 1, 0x7f, 0xef, 33, 0xc0, 64};
   static const uint32_t i32[] = {0x80000000, 0x7fffffff, 0xffffffff, 0, 1,
                                  0xffffffef, 0x80000001, 0xff000001, 0xffff0000};
   if (mode == 1)
      return width == 8 ? 0xfd : 0xfffffff9;
   if (mode == 2)
      return 2;
   if (mode == 3)
      return 0;
   return width == 8 ? i8[index % ARRAY_SIZE(i8)]
                     : i32[index % ARRAY_SIZE(i32)];
}

static uint32_t
source_bits(const struct test_source *s, unsigned index)
{
   unsigned inner = elements(&s->shape) / s->shape.dims[0];
   if (s->kind == SOURCE_NULL || (s->aggregate == 1 && !(index / inner & 1)))
      return 0;
   if (s->aggregate == 2)
      index %= inner;
   if (s->floating) {
      static const uint32_t values[] = {0x80000000, 0x7fc01234, 0xff800000,
                                        0x7f800000, 0x3f800000, 0xbf800000,
                                        1, 0x807fffff, 0x7f7fffff};
      return values[index % ARRAY_SIZE(values)];
   }
   return pattern(s->width, s->pattern, index);
}

static int32_t
signed_bits(uint32_t value, unsigned width)
{
   if (width == 8)
      return value & 0x80 ? (int32_t)(value & 0xff) - 256 : value & 0xff;
   int32_t result;
   memcpy(&result, &value, sizeof(result));
   return result;
}

static void
coordinates(const struct test_shape *s, unsigned index, uint32_t *coords)
{
   for (int d = s->rank - 1; d >= 0; d--) {
      coords[d] = index % s->dims[d];
      index /= s->dims[d];
   }
   assert(!index);
}

static unsigned
offset(const struct test_shape *s, const uint32_t *coords)
{
   unsigned index = 0;
   for (unsigned d = 0; d < s->rank; d++) {
      assert(coords[d] < s->dims[d]);
      index = index * s->dims[d] + coords[d];
   }
   return index;
}

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
emit_string(struct module *m, SpvOp op, uint32_t id, const char *string)
{
   unsigned count = DIV_ROUND_UP(strlen(string) + 1, 4);
   m->words[m->count++] = ((count + 1 + (id != 0)) << 16) | op;
   if (id)
      m->words[m->count++] = id;
   memset(m->words + m->count, 0, count * 4);
   memcpy(m->words + m->count, string, strlen(string));
   m->count += count;
}

static uint32_t
scalar_type(struct module *m, unsigned width, unsigned sign, bool floating)
{
   uint32_t *id = floating ? &m->f32 : &m->ints[width == 32][sign];
   if (!*id) {
      *id = new_id(m);
      if (floating)
         emit(m, SpvOpTypeFloat, 2, *id, width);
      else
         emit(m, SpvOpTypeInt, 3, *id, width, sign);
   }
   return *id;
}

static uint32_t
constant(struct module *m, uint32_t type, uint32_t value)
{
   uint32_t id = new_id(m);
   if (type == m->ints[0][1] && (value & 0x80))
      value |= 0xffffff00;
   emit(m, SpvOpConstant, 3, type, id, value);
   return id;
}

static uint32_t
tensor_type(struct module *m, uint32_t element,
            const struct test_shape *shape)
{
   uint32_t length = constant(m, m->u32, shape->rank);
   uint32_t array = new_id(m);
   emit(m, SpvOpTypeArray, 3, array, m->u32, length);
   uint32_t ids[4];
   for (unsigned i = 0; i < shape->rank; i++)
      ids[i] = constant(m, m->u32, shape->dims[i]);
   uint32_t dims = new_id(m);
   m->words[m->count++] = ((3 + shape->rank) << 16) | SpvOpConstantComposite;
   m->words[m->count++] = array;
   m->words[m->count++] = dims;
   for (unsigned i = 0; i < shape->rank; i++)
      m->words[m->count++] = ids[i];
   uint32_t type = new_id(m);
   emit(m, SpvOpTypeTensorARM, 4, type, element, length, dims);
   return type;
}

static uint32_t
attribute(struct module *m, unsigned count, const uint32_t *values)
{
   const struct test_shape shape = {1, {count}};
   uint32_t type = tensor_type(m, m->u32, &shape), ids[4];
   for (unsigned i = 0; i < count; i++)
      ids[i] = constant(m, m->u32, values[i]);
   uint32_t id = new_id(m);
   m->words[m->count++] = ((3 + count) << 16) | SpvOpConstantComposite;
   m->words[m->count++] = type;
   m->words[m->count++] = id;
   for (unsigned i = 0; i < count; i++)
      m->words[m->count++] = ids[i];
   return id;
}

static void
composite(struct module *m, const struct test_source *source, uint32_t scalar,
          uint32_t type, uint32_t id, const struct test_shape *shape,
          unsigned depth, unsigned *position)
{
   uint32_t ids[128];
   unsigned count = shape->dims[0];
   assert(count <= ARRAY_SIZE(ids));
   struct test_shape child = {.rank = shape->rank - 1};
   for (unsigned d = 0; d < child.rank; d++)
      child.dims[d] = shape->dims[d + 1];
   if (!depth && source->malformed == 1 && child.rank)
      child.dims[0]++;
   uint32_t child_type = child.rank ? tensor_type(m, scalar, &child) : scalar;
   unsigned child_count = child.rank ? elements(&child) : 1;
   for (unsigned i = 0; i < count; i++) {
      if (!depth && source->malformed == 4 && !i) {
         ids[i] = id;
         *position += child_count;
      } else if (!depth && source->aggregate == 2 && i) {
         ids[i] = ids[0];
         *position += child_count;
      } else if (!depth && source->aggregate == 1 && !(i & 1)) {
         ids[i] = new_id(m);
         emit(m, SpvOpConstantNull, 2, child_type, ids[i]);
         *position += child_count;
      } else if (child.rank) {
         ids[i] = new_id(m);
         composite(m, source, scalar, child_type, ids[i], &child,
                   depth + 1, position);
      } else {
         uint32_t leaf_type = source->malformed == 2 && !i
                                 ? scalar_type(m, source->width == 8 ? 32 : 8,
                                               source->sign, false)
                                 : scalar;
         ids[i] = constant(m, leaf_type, source_bits(source, (*position)++));
      }
   }
   if (!depth && source->malformed == 3)
      count--;
   m->words[m->count++] = ((3 + count) << 16) | SpvOpConstantComposite;
   m->words[m->count++] = type;
   m->words[m->count++] = id;
   for (unsigned i = 0; i < count; i++)
      m->words[m->count++] = ids[i];
}

static void
build_module(const struct test_case *c, struct module *m)
{
   *m = (struct module){.count = 5, .next = 1};
   m->words[0] = SpvMagicNumber;
   m->words[1] = 0x10600;
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityShader);
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityVulkanMemoryModel);
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityGraphARM);
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityTensorsARM);
   bool int8 = false;
   for (unsigned i = 0; i < c->source_count; i++)
      int8 |= c->sources[i].width == 8;
   if (int8)
      emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityInt8);
   emit_string(m, SpvOpExtension, 0, "SPV_ARM_graph");
   emit_string(m, SpvOpExtension, 0, "SPV_ARM_tensors");
   uint32_t tosa = new_id(m);
   emit_string(m, SpvOpExtInstImport, tosa, "TOSA.001000.1");
   emit(m, SpvOpMemoryModel, 2, (uint32_t)SpvAddressingModelLogical,
        (uint32_t)SpvMemoryModelVulkan);
   uint32_t variables[5], inputs[4], input_count = 0;
   for (unsigned i = 0; i < c->source_count; i++) {
      if (c->sources[i].kind == SOURCE_INPUT)
         inputs[input_count++] = i;
   }
   for (unsigned i = 0; i <= input_count; i++) {
      variables[i] = new_id(m);
      emit(m, SpvOpDecorate, 3, variables[i],
           (uint32_t)SpvDecorationDescriptorSet, 0u);
      emit(m, SpvOpDecorate, 3, variables[i], (uint32_t)SpvDecorationBinding,
           i);
   }
   m->u32 = scalar_type(m, 32, 0, false);
   uint32_t types[7], values[7], attrs[3] = {0};
   for (unsigned i = 0; i < c->source_count; i++) {
      const struct test_source *s = c->sources + i;
      uint32_t scalar = scalar_type(m, s->width, s->sign, s->floating);
      types[i] = tensor_type(m, scalar, &s->shape);
      values[i] = new_id(m);
      if (s->kind == SOURCE_EXTERNAL) {
         emit(m, SpvOpGraphConstantARM, 3, types[i], values[i], i + 7u);
      } else if (s->kind == SOURCE_NULL) {
         emit(m, SpvOpConstantNull, 2, types[i], values[i]);
      } else if (s->kind == SOURCE_COMPOSITE) {
         unsigned position = 0;
         composite(m, s, scalar, types[i], values[i], &s->shape, 0, &position);
      }
   }
   for (unsigned i = 0; i < c->node_count; i++) {
      const struct test_node *n = c->nodes + i;
      unsigned value = c->source_count + i;
      types[value] = tensor_type(m, scalar_type(m, n->width, n->sign, n->floating),
                                 &n->shape);
      values[value] = new_id(m);
      if (n->op == PANVK_GRAPH_OP_RESHAPE)
         attrs[i] = attribute(m, n->shape.rank, n->shape.dims);
      else if (n->op == PANVK_GRAPH_OP_TRANSPOSE)
         attrs[i] = attribute(m, n->shape.rank, n->permutation);
      else if (n->op >= PANVK_GRAPH_OP_REDUCE_MAX &&
               n->op <= PANVK_GRAPH_OP_REDUCE_SUM)
         attrs[i] = constant(m, m->u32, n->axis);
   }
   for (unsigned i = 0; i <= input_count; i++) {
      uint32_t type = types[i < input_count ? inputs[i] : c->output];
      uint32_t pointer = new_id(m);
      emit(m, SpvOpTypePointer, 3, pointer,
           (uint32_t)SpvStorageClassUniformConstant, type);
      emit(m, SpvOpVariable, 3, pointer, variables[i],
           (uint32_t)SpvStorageClassUniformConstant);
   }
   uint32_t indices[4];
   for (unsigned i = 0; i < ARRAY_SIZE(indices); i++)
      indices[i] = constant(m, m->u32, i);
   uint32_t nan = constant(m, m->u32, 1);
   uint32_t graph_type = new_id(m), graph = new_id(m);
   m->words[m->count++] = ((4 + input_count) << 16) | SpvOpTypeGraphARM;
   m->words[m->count++] = graph_type;
   m->words[m->count++] = input_count;
   for (unsigned i = 0; i < input_count; i++)
      m->words[m->count++] = types[inputs[i]];
   m->words[m->count++] = types[c->output];
   m->words[m->count++] = ((5 + input_count) << 16) | SpvOpGraphEntryPointARM;
   m->words[m->count++] = graph;
   m->words[m->count++] = 0x6e69616d;
   m->words[m->count++] = 0;
   for (unsigned i = 0; i <= input_count; i++)
      m->words[m->count++] = variables[i];
   emit(m, SpvOpGraphARM, 2, graph_type, graph);
   for (unsigned i = 0; i < input_count; i++)
      emit(m, SpvOpGraphInputARM, 3, types[inputs[i]], values[inputs[i]],
           indices[i]);
   for (unsigned i = 0; i < c->node_count; i++) {
      const struct test_node *n = c->nodes + i;
      unsigned value = c->source_count + i;
      uint32_t a = values[n->inputs[0]];
      if (n->op == PANVK_GRAPH_OP_MATMUL)
         emit(m, SpvOpExtInst, 8, types[value], values[value], tosa, n->op,
              a, values[n->inputs[1]], values[n->inputs[2]],
              values[n->inputs[3]]);
      else if (n->op == PANVK_GRAPH_OP_RESHAPE)
         emit(m, SpvOpExtInst, 6, types[value], values[value], tosa, n->op,
              a, attrs[i]);
      else if (n->op == PANVK_GRAPH_OP_TRANSPOSE)
         emit(m, SpvOpExtInst, 6, types[value], values[value], tosa, n->op,
              attrs[i], a);
      else if (n->op == PANVK_GRAPH_OP_REDUCE_MAX ||
               n->op == PANVK_GRAPH_OP_REDUCE_MIN)
         emit(m, SpvOpExtInst, 7, types[value], values[value], tosa, n->op,
              attrs[i], nan, a);
      else if (n->op == PANVK_GRAPH_OP_REDUCE_PRODUCT ||
               n->op == PANVK_GRAPH_OP_REDUCE_SUM)
         emit(m, SpvOpExtInst, 6, types[value], values[value], tosa, n->op,
              attrs[i], a);
      else
         emit(m, SpvOpExtInst, 6, types[value], values[value], tosa, n->op,
              a, values[n->inputs[1]]);
   }
   emit(m, SpvOpGraphSetOutputARM, 2, values[c->output], indices[0]);
   emit(m, SpvOpGraphEndARM, 0);
   m->words[3] = m->next;
}

static uint8_t *
source_data(const struct test_source *s)
{
   unsigned bytes = s->width / 8;
   uint8_t *data = malloc(elements(&s->shape) * bytes);
   assert(data);
   for (unsigned i = 0; i < elements(&s->shape); i++) {
      uint32_t value = source_bits(s, i);
      memcpy(data + i * bytes, &value, bytes);
   }
   return data;
}

static VkResult
parse_case(const struct test_case *c, const struct module *m,
           uint8_t *const *data, struct panvk_data_graph **graph)
{
   int64_t dims[5][4], constant_dims[4][4], strides[4][4];
   VkTensorDescriptionARM descs[5], constant_descs[4];
   VkDataGraphPipelineResourceInfoARM resources[5];
   VkDataGraphPipelineConstantARM external[4];
   uint8_t *strided[4] = {0};
   unsigned count = 0, constant_count = 0;
   for (unsigned i = 0; i < c->source_count; i++) {
      const struct test_source *s = c->sources + i;
      if (s->kind == SOURCE_INPUT) {
         for (unsigned d = 0; d < s->shape.rank; d++)
            dims[count][d] = s->shape.dims[d];
         descs[count] = (VkTensorDescriptionARM){
            .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
            .format = s->storage,
            .dimensionCount = s->shape.rank,
            .pDimensions = dims[count],
            .tiling = VK_TENSOR_TILING_LINEAR_ARM,
            .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
         };
         resources[count] = (VkDataGraphPipelineResourceInfoARM){
            .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
            .pNext = descs + count,
            .binding = count,
         };
         count++;
      } else if (s->kind == SOURCE_EXTERNAL && !s->missing) {
         unsigned bytes = s->width / 8;
         uint64_t stride = bytes * 3, size = bytes;
         for (int d = s->shape.rank - 1; d >= 0; d--) {
            constant_dims[i][d] = s->shape.dims[d];
            strides[i][d] = stride;
            size += (s->shape.dims[d] - 1) * stride;
            stride = stride * s->shape.dims[d] + bytes * 5;
         }
         if (s->strided) {
            strided[i] = calloc(size, 1);
            assert(strided[i]);
            for (unsigned j = 0; j < elements(&s->shape); j++) {
               uint32_t coords[4];
               coordinates(&s->shape, j, coords);
               uint64_t pos = 0;
               for (unsigned d = 0; d < s->shape.rank; d++)
                  pos += coords[d] * strides[i][d];
               memcpy(strided[i] + pos, data[i] + j * bytes, bytes);
            }
         }
         constant_descs[i] = (VkTensorDescriptionARM){
            .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
            .format = s->storage,
            .dimensionCount = s->shape.rank,
            .pDimensions = constant_dims[i],
            .pStrides = s->strided ? strides[i] : NULL,
            .tiling = VK_TENSOR_TILING_LINEAR_ARM,
            .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
         };
         external[constant_count++] = (VkDataGraphPipelineConstantARM){
            .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM,
            .pNext = constant_descs + i,
            .id = i + 7,
            .pConstantData = strided[i] ? strided[i] : data[i],
         };
      }
   }
   const struct test_shape *out = value_shape(c, c->output);
   for (unsigned d = 0; d < out->rank; d++)
      dims[count][d] = out->dims[d];
   descs[count] = (VkTensorDescriptionARM){
      .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
      .format = c->output_storage,
      .dimensionCount = out->rank,
      .pDimensions = dims[count],
      .tiling = VK_TENSOR_TILING_LINEAR_ARM,
      .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
   };
   resources[count] = (VkDataGraphPipelineResourceInfoARM){
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
      .pNext = descs + count,
      .binding = count,
   };
   VkDataGraphPipelineCreateInfoARM info = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
      .resourceInfoCount = count + 1,
      .pResourceInfos = resources,
   };
   VkDataGraphPipelineShaderModuleCreateInfoARM module = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
      .pName = "main",
      .constantCount = constant_count,
      .pConstants = external,
   };
   VkResult result = panvk_data_graph_parse(NULL, m->words, m->count * 4,
                                           &info, &module, graph);
   for (unsigned i = 0; i < c->source_count; i++)
      free(strided[i]);
   return result;
}

static uint32_t *
reference(const struct test_case *c, uint8_t *const *data, uint32_t **values)
{
   for (unsigned i = 0; i < c->source_count; i++) {
      const struct test_source *s = c->sources + i;
      values[i] = calloc(elements(&s->shape), sizeof(uint32_t));
      assert(values[i]);
      for (unsigned j = 0; j < elements(&s->shape); j++)
         memcpy(values[i] + j, data[i] + j * (s->width / 8), s->width / 8);
   }
   for (unsigned i = 0; i < c->node_count; i++) {
      const struct test_node *n = c->nodes + i;
      unsigned v = c->source_count + i;
      const struct test_shape *input = value_shape(c, n->inputs[0]);
      const uint32_t *a = values[n->inputs[0]];
      values[v] = calloc(elements(&n->shape), sizeof(uint32_t));
      assert(values[v]);
      if (n->op == PANVK_GRAPH_OP_TRANSPOSE) {
         for (unsigned j = 0; j < elements(input); j++) {
            uint32_t coords[4], output[4];
            coordinates(input, j, coords);
            for (unsigned d = 0; d < input->rank; d++)
               output[d] = coords[n->permutation[d]];
            values[v][offset(&n->shape, output)] = a[j];
         }
      } else if (n->op == PANVK_GRAPH_OP_RESHAPE) {
         assert(elements(input) == elements(&n->shape));
         memcpy(values[v], a, elements(input) * sizeof(uint32_t));
      } else if (n->op == PANVK_GRAPH_OP_ADD || n->op == PANVK_GRAPH_OP_SUB) {
         const struct test_shape *other = value_shape(c, n->inputs[1]);
         const uint32_t *b = values[n->inputs[1]];
         assert(elements(other) == 1 || elements(other) == elements(input));
         for (unsigned j = 0; j < elements(input); j++) {
            uint32_t right = b[elements(other) == 1 ? 0 : j];
            values[v][j] = n->op == PANVK_GRAPH_OP_ADD ? a[j] + right
                                                      : a[j] - right;
         }
      } else if (n->op == PANVK_GRAPH_OP_MATMUL) {
         const struct test_shape *bshape = value_shape(c, n->inputs[1]);
         const uint32_t *b = values[n->inputs[1]];
         int32_t za = signed_bits(values[n->inputs[2]][0], 8);
         int32_t zb = signed_bits(values[n->inputs[3]][0], 8);
         for (unsigned batch = 0; batch < input->dims[0]; batch++) {
            for (unsigned row = 0; row < input->dims[1]; row++) {
               for (unsigned col = 0; col < bshape->dims[2]; col++) {
                  int64_t sum = 0;
                  for (unsigned k = 0; k < input->dims[2]; k++) {
                     unsigned ai = (batch * input->dims[1] + row) * input->dims[2] + k;
                     unsigned bi = (batch * bshape->dims[1] + k) * bshape->dims[2] + col;
                     sum += (int64_t)(signed_bits(a[ai], 8) - za) *
                            (signed_bits(b[bi], 8) - zb);
                  }
                  unsigned dest = (batch * input->dims[1] + row) * bshape->dims[2] + col;
                  values[v][dest] = (uint32_t)sum;
               }
            }
         }
      } else {
         assert(n->op >= PANVK_GRAPH_OP_REDUCE_MAX &&
                n->op <= PANVK_GRAPH_OP_REDUCE_SUM);
         for (unsigned j = 0; j < elements(&n->shape); j++) {
            uint32_t coords[4];
            coordinates(&n->shape, j, coords);
            uint32_t accum = n->op == PANVK_GRAPH_OP_REDUCE_PRODUCT ? 1 : 0;
            for (unsigned k = 0; k < input->dims[n->axis]; k++) {
               coords[n->axis] = k;
               uint32_t x = a[offset(input, coords)];
               if (n->op == PANVK_GRAPH_OP_REDUCE_SUM)
                  accum += x;
               else if (n->op == PANVK_GRAPH_OP_REDUCE_PRODUCT)
                  accum *= x;
               else if (!k || (n->op == PANVK_GRAPH_OP_REDUCE_MAX
                                  ? signed_bits(x, 32) > signed_bits(accum, 32)
                                  : signed_bits(x, 32) < signed_bits(accum, 32)))
                  accum = x;
            }
            values[v][j] = accum;
         }
      }
   }
   return values[c->output];
}

static void
check_neural(const struct test_case *c, const struct panvk_data_graph *graph)
{
   struct panvk_data_graph_matmul_add extracted = {0};
   assert(panvk_data_graph_matmul_add_extract(graph, &extracted) == VK_SUCCESS);
   assert(extracted.add_count == c->node_count - 1);
   const struct test_shape *a = &c->sources[0].shape;
   const struct test_shape *b = &c->sources[1].shape;
   assert(extracted.info.batch == a->dims[0]);
   assert(extracted.info.rows == a->dims[1]);
   assert(extracted.info.inner == a->dims[2]);
   assert(extracted.info.columns == b->dims[2]);
   assert(extracted.info.a_zero_point ==
          signed_bits(c->sources[2].kind == SOURCE_NULL ? 0 :
                         pattern(8, c->sources[2].pattern, 0), 8));
   assert(extracted.info.b_zero_point ==
          signed_bits(c->sources[3].kind == SOURCE_NULL ? 0 :
                         pattern(8, c->sources[3].pattern, 0), 8));
   for (unsigned i = 0; i < 2; i++) {
      assert(extracted.input_indices[i] == i);
      assert(graph->values[extracted.inputs[i]].format == VK_FORMAT_R8_SINT);
      assert(graph->values[extracted.inputs[i]].storage_format == c->sources[i].storage);
   }
   panvk_data_graph_matmul_add_finish(&extracted);
   neural++;
}

static void
check_case(const struct test_case *c)
{
   struct module m;
   build_module(c, &m);
   uint8_t *data[4] = {0};
   for (unsigned i = 0; i < c->source_count; i++)
      data[i] = source_data(c->sources + i);
   struct panvk_data_graph *g = NULL;
   VkResult result = parse_case(c, &m, data, &g);
   unsigned case_id = cases++;
   if (c->invalid) {
      if (result != VK_ERROR_FEATURE_NOT_PRESENT || g)
         fprintf(stderr, "reject case=%u result=%d graph=%p\n", case_id, result, (void *)g);
      assert(result == VK_ERROR_FEATURE_NOT_PRESENT && !g);
      rejected++;
      goto cleanup;
   }
   if (result != VK_SUCCESS) {
      fprintf(stderr, "valid case=%u width=%u sign=%u storage=%u output=%u source=%u result=%d\n",
              case_id, c->sources[0].width, c->sources[0].sign,
              c->sources[0].storage, c->output_storage, c->sources[0].kind, result);
      abort();
   }
   assert(g->count <= c->source_count + c->node_count + 1);
   if (dump_directory) {
      char path[1024];
      assert(snprintf(path, sizeof(path), "%s/integer-%05u.spv", dump_directory,
                      case_id) < sizeof(path));
      FILE *f = fopen(path, "wb");
      assert(f && fwrite(m.words, 4, m.count, f) == m.count);
      assert(!fclose(f));
      assert(snprintf(path, sizeof(path), "%s/integer-%05u.case", dump_directory,
                      case_id) < sizeof(path));
      f = fopen(path, "wb");
      assert(f && fwrite(c, sizeof(*c), 1, f) == 1);
      assert(!fclose(f));
   }
   uint32_t *values[7] = {0};
   const uint32_t *expected = reference(c, data, values);
   uint8_t **buffers = calloc(g->count, sizeof(*buffers));
   uint8_t *transient = calloc(MAX2(g->transient_size, 1), 1);
   unsigned bytes = value_width(c, c->output) / 8;
   unsigned count = elements(value_shape(c, c->output));
   uint8_t *output = malloc(count * bytes + 32);
   assert(buffers && transient && output);
   memset(output, 0xa5, count * bytes + 32);
   unsigned inputs[4], input_count = 0;
   for (unsigned i = 0; i < c->source_count; i++) {
      if (c->sources[i].kind == SOURCE_INPUT)
         inputs[input_count++] = i;
   }
   for (unsigned i = 0; i < g->count; i++) {
      const struct panvk_graph_value *v = g->values + i;
      assert(v->format == VK_FORMAT_R8_SINT || v->format == VK_FORMAT_R32_SINT ||
             v->format == VK_FORMAT_R32_SFLOAT);
      switch (v->kind) {
      case PANVK_GRAPH_INPUT:
         assert(v->resource_index < input_count);
         assert(v->storage_format == c->sources[inputs[v->resource_index]].storage);
         buffers[i] = data[inputs[v->resource_index]];
         break;
      case PANVK_GRAPH_CONSTANT:
         buffers[i] = v->data;
         constants++;
         break;
      case PANVK_GRAPH_OPERATION:
         if (v->live && v->fused_into == UINT32_MAX) {
            assert(v->offset + v->size <= g->transient_size);
            buffers[i] = transient + v->offset;
         }
         break;
      case PANVK_GRAPH_OUTPUT:
         assert(v->storage_format == c->output_storage);
         buffers[i] = output;
         break;
      }
   }
   if (c->neural)
      check_neural(c, g);
   for (unsigned pass = 0; pass < g->pass_count; pass++) {
      nir_shader *nir = panvk_data_graph_shader(g, pass);
      assert(nir && !nir->info.internal);
      nir_validate_shader(nir, "integer storage semantics");
      struct graph_interpreter e;
      graph_interpreter_init(&e, g, nir, buffers);
      unsigned pass_count = g->values[g->passes[pass]].elements;
      unsigned launched = ALIGN_POT(pass_count, 64) + 64;
      for (unsigned i = 0; i < launched; i++)
         assert(graph_interpret(&e, i) == (i < pass_count));
      assert(!graph_interpret(&e, 65535 * 64));
      invocations += launched + 1;
      shaders++;
      free(e.values);
      ralloc_free(nir);
   }
   for (unsigned i = 0; i < count; i++) {
      uint32_t actual = 0;
      memcpy(&actual, output + i * bytes, bytes);
      if (actual != expected[i]) {
         fprintf(stderr, "numeric case=%u element=%u actual=%08x expected=%08x\n",
                 case_id, i, actual, expected[i]);
         abort();
      }
   }
   for (unsigned i = 0; i < 32; i++)
      assert(output[count * bytes + i] == 0xa5);
   checked += count;
   for (unsigned i = 0; i < c->source_count + c->node_count; i++)
      free(values[i]);
   free(output);
   free(transient);
   free(buffers);
   ralloc_free(g);
cleanup:
   for (unsigned i = 0; i < c->source_count; i++)
      free(data[i]);
}

static VkFormat
storage(unsigned width, bool unsign)
{
   return width == 8 ? (unsign ? VK_FORMAT_R8_UINT : VK_FORMAT_R8_SINT)
                     : (unsign ? VK_FORMAT_R32_UINT : VK_FORMAT_R32_SINT);
}

static struct test_case
transform_case(unsigned width, unsigned sign, enum source_kind kind,
               unsigned input_storage, unsigned output_storage,
               const struct test_shape *shape, bool chain)
{
   struct test_case c = {
      .sources = {{.kind = kind, .shape = *shape, .width = width, .sign = sign,
                   .storage = storage(width, input_storage)}},
      .source_count = 1,
      .node_count = chain ? 2 : 1,
      .output = chain ? 2 : 1,
      .output_storage = storage(width, output_storage),
   };
   c.nodes[0] = (struct test_node){
      .op = chain ? PANVK_GRAPH_OP_TRANSPOSE : PANVK_GRAPH_OP_RESHAPE,
      .shape = *shape, .width = width, .sign = sign,
   };
   if (chain) {
      for (unsigned d = 0; d < shape->rank; d++) {
         c.nodes[0].permutation[d] = shape->rank - d - 1;
         c.nodes[0].shape.dims[d] = shape->dims[shape->rank - d - 1];
      }
      c.nodes[1] = (struct test_node){
         .op = PANVK_GRAPH_OP_RESHAPE, .shape = *shape,
         .width = width, .sign = sign, .inputs = {1},
      };
   }
   return c;
}

static void
check_transforms(void)
{
   const struct test_shape shapes[] = {
      {1, {1}}, {1, {63}}, {1, {64}}, {1, {65}}, {2, {3, 23}},
      {3, {2, 3, 11}}, {4, {1, 2, 3, 11}}, {4, {1, 1, 1, 1}},
   };
   for (unsigned width = 8; width <= 32; width += 24) {
      for (unsigned sign = 0; sign < 2; sign++) {
         for (unsigned kind = SOURCE_INPUT; kind <= SOURCE_NULL; kind++) {
            for (unsigned in = 0; in < 2; in++) {
               for (unsigned out = 0; out < 2; out++) {
                  for (unsigned s = 0; s < ARRAY_SIZE(shapes); s++) {
                     for (unsigned chain = 0; chain < 2; chain++) {
                        struct test_case c = transform_case(width, sign, kind, in,
                                                            out, shapes + s, chain);
                        c.sources[0].strided = kind == SOURCE_EXTERNAL && (s & 1);
                        check_case(&c);
                     }
                  }
                  struct test_case direct = transform_case(width, sign, kind,
                                                           in, out, shapes + 3,
                                                           false);
                  direct.node_count = 0;
                  direct.output = 0;
                  check_case(&direct);
               }
            }
         }
      }
   }
}

static void
check_reductions(void)
{
   const struct test_shape shapes[] = {
      {1, {65}}, {2, {3, 23}}, {3, {2, 3, 11}}, {4, {1, 2, 3, 11}},
   };
   for (unsigned sign = 0; sign < 2; sign++) {
      for (unsigned in = 0; in < 2; in++) {
         for (unsigned out = 0; out < 2; out++) {
            for (unsigned s = 0; s < ARRAY_SIZE(shapes); s++) {
               for (unsigned axis = 0; axis < shapes[s].rank; axis++) {
                  for (unsigned op = PANVK_GRAPH_OP_REDUCE_MAX;
                       op <= PANVK_GRAPH_OP_REDUCE_SUM; op++) {
                     for (unsigned chain = 0; chain < 2; chain++) {
                        struct test_case c = {
                           .sources = {{.kind = SOURCE_INPUT, .shape = shapes[s],
                                        .width = 32, .sign = sign,
                                        .storage = storage(32, in)},
                                       {.kind = SOURCE_COMPOSITE, .shape = {1, {1}},
                                        .width = 32, .sign = 1 - sign, .pattern = 1}},
                           .source_count = chain ? 2 : 1,
                           .node_count = chain ? 2 : 1,
                           .output = chain ? 3 : 1,
                           .output_storage = storage(32, out),
                        };
                         c.sources[1].shape.rank = shapes[s].rank;
                         for (unsigned d = 0; d < shapes[s].rank; d++)
                            c.sources[1].shape.dims[d] = 1;
                         if (chain)
                           c.nodes[0] = (struct test_node){
                              .op = PANVK_GRAPH_OP_ADD, .shape = shapes[s],
                              .width = 32, .sign = sign, .inputs = {0, 1},
                           };
                        c.nodes[chain] = (struct test_node){
                           .op = op, .shape = shapes[s], .width = 32,
                           .sign = 1 - sign, .inputs = {chain ? 2 : 0}, .axis = axis,
                        };
                        c.nodes[chain].shape.dims[axis] = 1;
                        check_case(&c);
                     }
                  }
               }
            }
         }
      }
   }
}

static struct test_case
matmul_case(unsigned sign, unsigned storage_mask, unsigned batch, unsigned rows,
            unsigned inner, unsigned columns, unsigned zero_kind, bool add)
{
   struct test_case c = {
      .sources = {
         {.kind = SOURCE_INPUT, .shape = {3, {batch, rows, inner}}, .width = 8,
          .sign = sign, .storage = storage(8, storage_mask & 1)},
         {.kind = SOURCE_INPUT, .shape = {3, {batch, inner, columns}}, .width = 8,
          .sign = 1 - sign, .storage = storage(8, storage_mask & 2)},
         {.kind = zero_kind, .shape = {1, {1}}, .width = 8,
          .sign = sign, .storage = VK_FORMAT_R8_UINT, .pattern = 1},
         {.kind = zero_kind, .shape = {1, {1}}, .width = 8,
          .sign = 1 - sign, .storage = VK_FORMAT_R8_SINT, .pattern = 2},
      },
      .nodes = {{.op = PANVK_GRAPH_OP_MATMUL,
                 .shape = {3, {batch, rows, columns}}, .width = 32,
                 .sign = sign, .inputs = {0, 1, 2, 3}},
                {.op = PANVK_GRAPH_OP_ADD, .shape = {3, {batch, rows, columns}},
                 .width = 32, .sign = 1 - sign, .inputs = {4, 4}}},
      .source_count = 4, .node_count = add ? 2 : 1,
      .output = add ? 5 : 4, .output_storage = storage(32, storage_mask & 4),
      .neural = true,
   };
   return c;
}

static void
check_matmuls(void)
{
   const unsigned shapes[][4] = {
      {1, 1, 1, 1}, {2, 3, 5, 4}, {3, 1, 17, 7}, {1, 2, 4095, 1},
      {1, 2, 4096, 1},
   };
   for (unsigned sign = 0; sign < 2; sign++) {
      for (unsigned mask = 0; mask < 8; mask++) {
         for (unsigned k = SOURCE_EXTERNAL; k <= SOURCE_NULL; k++) {
            for (unsigned s = 0; s < ARRAY_SIZE(shapes); s++) {
               for (unsigned add = 0; add < 2; add++) {
                  struct test_case c = matmul_case(sign, mask, shapes[s][0],
                                                  shapes[s][1], shapes[s][2],
                                                  shapes[s][3], k, add);
                  check_case(&c);
               }
            }
         }
      }
   }
}

static void
check_nested_constants(void)
{
   const struct test_shape shapes[] = {
      {1, {65}}, {2, {3, 23}}, {3, {2, 3, 11}}, {4, {1, 2, 3, 11}},
   };
   for (unsigned width = 8; width <= 32; width += 24) {
      for (unsigned sign = 0; sign < 2; sign++) {
         for (unsigned out = 0; out < 2; out++) {
            for (unsigned s = 0; s < ARRAY_SIZE(shapes); s++) {
               for (unsigned mode = 1; mode <= 2; mode++) {
                  for (unsigned chain = 0; chain < 2; chain++) {
                     struct test_case c = transform_case(width, sign,
                                                         SOURCE_COMPOSITE, 0,
                                                         out, shapes + s, chain);
                     c.sources[0].aggregate = mode;
                     check_case(&c);
                  }
               }
            }
         }
         for (unsigned s = 1; s < ARRAY_SIZE(shapes); s++) {
            for (unsigned bad = 1; bad <= 4; bad++) {
               struct test_case c = transform_case(width, sign, SOURCE_COMPOSITE,
                                                   0, 0, shapes + s, true);
               c.sources[0].malformed = bad;
               c.invalid = true;
               check_case(&c);
            }
         }
      }
   }
   for (unsigned s = 0; s < ARRAY_SIZE(shapes); s++) {
      for (unsigned mode = 0; mode <= 2; mode++) {
         for (unsigned chain = 0; chain < 2; chain++) {
            struct test_case c = transform_case(32, 0, SOURCE_COMPOSITE, 0, 0,
                                                shapes + s, chain);
            c.sources[0].floating = true;
            c.sources[0].aggregate = mode;
            c.output_storage = VK_FORMAT_R32_SFLOAT;
            for (unsigned i = 0; i < c.node_count; i++)
               c.nodes[i].floating = true;
            check_case(&c);
         }
      }
   }
}

static void
check_rejections(void)
{
   const struct test_shape shape = {2, {3, 23}};
   const VkFormat invalid8[] = {VK_FORMAT_R32_SINT, VK_FORMAT_R32_UINT,
                                VK_FORMAT_R32_SFLOAT, VK_FORMAT_R8_UNORM};
   const VkFormat invalid32[] = {VK_FORMAT_R8_SINT, VK_FORMAT_R8_UINT,
                                 VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16_SINT};
   for (unsigned width = 8; width <= 32; width += 24) {
      for (unsigned sign = 0; sign < 2; sign++) {
         for (unsigned kind = SOURCE_INPUT; kind <= SOURCE_EXTERNAL; kind++) {
            for (unsigned position = 0; position < 2; position++) {
               for (unsigned f = 0; f < ARRAY_SIZE(invalid8); f++) {
                  struct test_case c = transform_case(width, sign, kind, 1, 1,
                                                      &shape, true);
                  VkFormat bad = width == 8 ? invalid8[f] : invalid32[f];
                  if (position)
                     c.output_storage = bad;
                  else
                     c.sources[0].storage = bad;
                  c.invalid = true;
                  check_case(&c);
               }
            }
            struct test_case c = transform_case(width, sign, kind, 0, 0,
                                                &shape, false);
            c.sources[0].sign = 2;
            c.invalid = true;
            check_case(&c);
            if (kind == SOURCE_EXTERNAL) {
               c.sources[0].sign = sign;
               c.sources[0].missing = true;
               check_case(&c);
            }
         }
      }
   }
   for (unsigned f = 0; f < 2; f++) {
      struct test_case c = transform_case(32, 0, SOURCE_INPUT, f, 0, &shape, false);
      c.sources[0].floating = true;
      c.invalid = true;
      check_case(&c);
   }
   for (unsigned width = 8; width <= 32; width += 24) {
      for (unsigned sign = 0; sign < 2; sign++) {
         for (unsigned kind = SOURCE_INPUT; kind <= SOURCE_NULL; kind++) {
            struct test_case c = transform_case(width, sign, kind, 1, 1,
                                                &shape, false);
            c.nodes[0].width = width == 8 ? 32 : 8;
            c.output_storage = storage(c.nodes[0].width, true);
            c.invalid = true;
            check_case(&c);
         }
      }
   }
}

static void
probe(bool nested)
{
   const struct test_shape shape = nested ? (struct test_shape){3, {2, 3, 11}}
                                          : (struct test_shape){1, {65}};
   for (unsigned width = 8; width <= 32; width += 24) {
      for (unsigned sign = 0; sign < 2; sign++) {
         for (unsigned kind = SOURCE_INPUT; kind <= SOURCE_NULL; kind++) {
            for (unsigned store = 0; store < 2; store++) {
               struct test_case c = transform_case(width, sign, kind, store,
                                                   store, &shape, false);
               struct module m;
               build_module(&c, &m);
               uint8_t *data[4] = {source_data(c.sources)};
               struct panvk_data_graph *g = NULL;
               VkResult result = parse_case(&c, &m, data, &g);
               printf("{\"width\":%u,\"sign\":%u,\"source\":%u,\"storage\":%u,\"result\":%d}\n",
                      width, sign, kind, c.sources[0].storage, result);
               if (g)
                  ralloc_free(g);
               free(data[0]);
            }
         }
      }
   }
}

int
main(int argc, char **argv)
{
   if (argc == 2 && (!strcmp(argv[1], "--probe") ||
                     !strcmp(argv[1], "--probe-nested"))) {
      probe(!strcmp(argv[1], "--probe-nested"));
      return 0;
   }
   assert(argc <= 2);
   dump_directory = argc == 2 ? argv[1] : NULL;
   check_transforms();
   check_reductions();
   check_matmuls();
   check_nested_constants();
   check_rejections();
   printf("integer-types cases=%u valid=%u rejected=%u shaders=%u invocations=%" PRIu64
          " outputs=%" PRIu64 " neural=%u constants=%u\n",
          cases, cases - rejected, rejected, shaders, invocations, checked,
          neural, constants);
   return 0;
}
