#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/spirv/spirv.h"
#include "util/ralloc.h"
#include "panvk_data_graph.h"
#include "vk_format.h"
#include "data-graph-nir-interpreter.h"

struct tensor {
   unsigned rank;
   uint32_t dims[4];
   VkFormat format;
   const void *data;
   bool input;
};

struct module {
   uint32_t words[8192], count, next, u32, i8, fp32, boolean, tosa;
   uint32_t types[8], values[8], variables[9], indices[8], input_count;
   uint32_t output_type, output_variable, zero;
};

static unsigned cases, shaders, checks, neural_plans;
static unsigned required_views;

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

static uint32_t
constant(struct module *m, uint32_t value)
{
   uint32_t id = m->next++;
   emit(m, SpvOpConstant, 3, m->u32, id, value);
   return id;
}

static uint32_t
boolean(struct module *m, bool value)
{
   uint32_t id = m->next++;
   emit(m, value ? SpvOpConstantTrue : SpvOpConstantFalse, 2, m->boolean, id);
   return id;
}

static uint32_t
vector(struct module *m, unsigned count, const int32_t *values, bool replicate)
{
   uint32_t length = constant(m, count), array = m->next++, ids[4];
   emit(m, SpvOpTypeArray, 3, array, m->u32, length);
   for (unsigned i = 0; i < count; i++)
      ids[i] = constant(m, values[i]);
   uint32_t id = m->next++;
   if (replicate) {
      emit(m, SpvOpConstantCompositeReplicateEXT, 3, array, id, ids[0]);
   } else {
      m->words[m->count++] = ((count + 3) << 16) | SpvOpConstantComposite;
      m->words[m->count++] = array;
      m->words[m->count++] = id;
      for (unsigned i = 0; i < count; i++)
         m->words[m->count++] = ids[i];
   }
   return id;
}

static uint32_t
type(struct module *m, const struct tensor *t)
{
   uint32_t rank = constant(m, t->rank);
   uint32_t shape = vector(m, t->rank, (const int32_t *)t->dims, false);
   uint32_t id = m->next++;
   emit(m, SpvOpTypeTensorARM, 4, id,
        t->format == VK_FORMAT_R8_SINT ? m->i8 :
        t->format == VK_FORMAT_R32_SFLOAT ? m->fp32 : m->u32, rank, shape);
   return id;
}

static uint32_t
variable(struct module *m, uint32_t type, unsigned binding)
{
   uint32_t pointer = m->next++, id = m->next++;
   emit(m, SpvOpTypePointer, 3, pointer, (uint32_t)SpvStorageClassUniformConstant, type);
   emit(m, SpvOpVariable, 3, pointer, id, (uint32_t)SpvStorageClassUniformConstant);
   emit(m, SpvOpDecorate, 3, id, (uint32_t)SpvDecorationDescriptorSet, 0u);
   emit(m, SpvOpDecorate, 3, id, (uint32_t)SpvDecorationBinding, binding);
   return id;
}

static void
initialize(struct module *m, const struct tensor *sources, unsigned count,
           const struct tensor *output)
{
   *m = (struct module){.count = 5, .next = 1};
   m->words[0] = SpvMagicNumber;
   m->words[1] = 0x10600;
   m->tosa = m->next++;
   emit(m, SpvOpExtInstImport, 5, m->tosa, 0x41534f54u, 0x3130302eu, 0x2e303030u, 0x31u);
   m->u32 = m->next++;
   m->i8 = m->next++;
   m->boolean = m->next++;
   m->fp32 = m->next++;
   emit(m, SpvOpTypeInt, 3, m->u32, 32u, 0u);
   emit(m, SpvOpTypeInt, 3, m->i8, 8u, 0u);
   emit(m, SpvOpTypeBool, 1, m->boolean);
   emit(m, SpvOpTypeFloat, 2, m->fp32, 32u);
   m->zero = constant(m, 0);
   for (unsigned i = 0; i < count; i++) {
      m->types[i] = type(m, &sources[i]);
      m->values[i] = m->next++;
      if (sources[i].input) {
         m->indices[i] = constant(m, m->input_count);
         m->variables[m->input_count] = variable(m, m->types[i], m->input_count);
         m->input_count++;
      } else {
         emit(m, SpvOpGraphConstantARM, 3, m->types[i], m->values[i], i + 1);
      }
   }
   m->output_type = type(m, output);
   m->output_variable = variable(m, m->output_type, m->input_count);
}

static void
begin(struct module *m, const struct tensor *sources, unsigned count)
{
   uint32_t graph_type = m->next++, graph = m->next++;
   m->words[m->count++] = ((m->input_count + 4) << 16) | SpvOpTypeGraphARM;
   m->words[m->count++] = graph_type;
   m->words[m->count++] = m->input_count;
   for (unsigned i = 0; i < count; i++)
      if (sources[i].input)
         m->words[m->count++] = m->types[i];
   m->words[m->count++] = m->output_type;
   m->words[m->count++] = ((m->input_count + 5) << 16) | SpvOpGraphEntryPointARM;
   m->words[m->count++] = graph;
   m->words[m->count++] = 0x6e69616d;
   m->words[m->count++] = 0;
   for (unsigned i = 0; i < m->input_count; i++)
      m->words[m->count++] = m->variables[i];
   m->words[m->count++] = m->output_variable;
   emit(m, SpvOpGraphARM, 2, graph_type, graph);
   for (unsigned i = 0; i < count; i++)
      if (sources[i].input)
         emit(m, SpvOpGraphInputARM, 3, m->types[i], m->values[i], m->indices[i]);
}

static void
evaluate(struct module *m, uint32_t result, const struct tensor *sources,
         unsigned count, const struct tensor *output, const void *expected,
         unsigned expected_neural)
{
   emit(m, SpvOpGraphSetOutputARM, 2, result, m->zero);
   emit(m, SpvOpGraphEndARM, 0);
   m->words[3] = m->next;
   int64_t dims[9][4];
   VkTensorDescriptionARM descs[9];
   VkDataGraphPipelineResourceInfoARM resources[9];
   VkDataGraphPipelineConstantARM constants[8];
   unsigned resource_count = 0, constant_count = 0;
   for (unsigned i = 0; i <= count; i++) {
      const struct tensor *t = i < count ? &sources[i] : output;
      for (unsigned d = 0; d < t->rank; d++)
         dims[i][d] = t->dims[d];
      descs[i] = (VkTensorDescriptionARM){
         .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
         .format = t->format, .tiling = VK_TENSOR_TILING_LINEAR_ARM,
         .dimensionCount = t->rank, .pDimensions = dims[i],
         .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
      };
      if (i == count || t->input) {
         resources[resource_count] = (VkDataGraphPipelineResourceInfoARM){
            .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
            .pNext = &descs[i], .binding = resource_count,
         };
         resource_count++;
      } else {
         constants[constant_count++] = (VkDataGraphPipelineConstantARM){
            .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM,
            .pNext = &descs[i], .id = i + 1, .pConstantData = t->data,
         };
      }
   }
   VkDataGraphPipelineCreateInfoARM info = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
      .resourceInfoCount = resource_count, .pResourceInfos = resources,
   };
   VkDataGraphPipelineShaderModuleCreateInfoARM module = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
      .pName = "main", .constantCount = constant_count, .pConstants = constants,
   };
   struct panvk_data_graph *g = NULL;
   VkResult status = panvk_data_graph_parse(NULL, m->words, 4 * m->count, &info, &module, &g);
   if (status != VK_SUCCESS) {
      fprintf(stderr, "quantized case %u parse=%d\n", cases, status);
      abort();
   }
   uint8_t **buffers = calloc(g->count, sizeof(*buffers));
   uint8_t *transient = calloc(MAX2(g->transient_size, 1), 1);
   uint8_t *actual = NULL;
   uint32_t output_size = 0, plans = 0;
   for (unsigned i = 0; i < g->count; i++) {
      const struct panvk_graph_value *v = &g->values[i];
      plans += v->matmul != NULL || v->conv != NULL;
      if (v->kind == PANVK_GRAPH_CONSTANT) {
         buffers[i] = v->data;
      } else if (v->kind == PANVK_GRAPH_OPERATION) {
         if (v->live && v->fused_into == UINT32_MAX)
            buffers[i] = transient + v->offset;
      } else if (v->kind == PANVK_GRAPH_OUTPUT) {
         output_size = v->size;
         actual = malloc(output_size + 32);
         assert(actual);
         memset(actual, 0xa5, output_size + 32);
         buffers[i] = actual;
      } else {
         unsigned input = 0;
         for (unsigned s = 0; s < count; s++)
            if (sources[s].input && input++ == v->resource_index) {
               buffers[i] = (uint8_t *)sources[s].data;
               break;
            }
      }
   }
   assert(plans == expected_neural);
   unsigned views = 0;
   for (unsigned i = 0; i < g->count; i++)
      views += g->values[i].storage != NULL;
   assert(views == required_views);
   neural_plans += plans;
   for (unsigned pass = 0; pass < g->pass_count; pass++) {
      nir_shader *nir = panvk_data_graph_shader(g, pass);
      assert(nir);
      struct graph_interpreter interpreter;
      graph_interpreter_init(&interpreter, g, nir, buffers);
      unsigned elements = g->values[g->passes[pass]].elements;
      for (unsigned i = 0; i < ALIGN_POT(elements, 64) + 64; i++)
         assert(graph_interpret(&interpreter, i) == (i < elements));
      free(interpreter.values);
      ralloc_free(nir);
      shaders++;
   }
   if (memcmp(actual, expected, output_size)) {
      unsigned i = 0;
      while (actual[i] == ((const uint8_t *)expected)[i]) i++;
      fprintf(stderr, "quantized case %u byte %u actual=%u expected=%u\n",
              cases, i, actual[i], ((const uint8_t *)expected)[i]);
      abort();
   }
   for (unsigned i = 0; i < 32; i++) assert(actual[output_size + i] == 0xa5);
   checks += output_size / vk_format_get_blocksize(output->format);
   cases++;
   free(actual);
   free(transient);
   free(buffers);
   ralloc_free(g);
}

static void
check_conv(unsigned batch, unsigned stride, unsigned dilation, int8_t azp, int8_t wzp,
           bool dynamic_weights, bool quantized, bool resize)
{
   int8_t input[2 * 7 * 7 * 3], weights[5 * 3 * 3 * 3];
   int32_t bias[5] = {INT32_C(10001), -300, 0, 71, -901};
   for (unsigned i = 0; i < ARRAY_SIZE(input); i++) input[i] = (i * 67 + 91) & 255;
   for (unsigned i = 0; i < ARRAY_SIZE(weights); i++) weights[i] = (i * 29 + 193) & 255;
   unsigned factor = resize ? 2 : 1;
   unsigned size = (7 * factor + 2 - (2 * dilation + 1)) / stride + 1;
   struct tensor sources[] = {
      {4, {batch, 7, 7, 3}, VK_FORMAT_R8_SINT, input, true},
      {4, {5, 3, 3, 3}, VK_FORMAT_R8_SINT, weights, dynamic_weights},
      {1, {5}, VK_FORMAT_R32_SINT, bias},
      {1, {1}, VK_FORMAT_R8_SINT, &azp},
      {1, {1}, VK_FORMAT_R8_SINT, &wzp},
      {1, {1}, VK_FORMAT_R32_SINT, (int32_t[]){1 << 24}},
      {1, {1}, VK_FORMAT_R8_SINT, (int8_t[]){32}},
      {1, {1}, VK_FORMAT_R32_SINT, (int32_t[]){0}},
   };
   struct tensor output = {4, {batch, size, size, 5}, VK_FORMAT_R32_SINT};
   int32_t expected[2 * 14 * 14 * 5];
   for (unsigned n = 0; n < batch; n++)
      for (unsigned y = 0; y < size; y++)
         for (unsigned x = 0; x < size; x++)
            for (unsigned oc = 0; oc < 5; oc++) {
               int32_t sum = bias[oc];
               for (unsigned ky = 0; ky < 3; ky++)
                  for (unsigned kx = 0; kx < 3; kx++)
                     for (unsigned ic = 0; ic < 3; ic++) {
                        int iy = y * stride + ky * dilation - 1;
                        int ix = x * stride + kx * dilation - 1;
                        if (iy >= 0 && iy < (int)(7 * factor) && ix >= 0 && ix < (int)(7 * factor))
                           sum += (input[((n * 7 + iy / factor) * 7 + ix / factor) * 3 + ic] - azp) *
                                  (weights[((oc * 3 + ky) * 3 + kx) * 3 + ic] - wzp);
                     }
               expected[((n * size + y) * size + x) * 5 + oc] = sum;
            }
   struct tensor accumulator = output;
   int8_t scaled[2 * 14 * 14 * 5];
   if (quantized) {
      output.format = VK_FORMAT_R8_SINT;
      for (unsigned i = 0; i < batch * size * size * 5; i++)
         scaled[i] = CLAMP((((int64_t)expected[i] * (1 << 24) + (INT64_C(1) << 31)) >> 32) + azp, -128, 127);
   }
   struct module m;
   initialize(&m, sources, ARRAY_SIZE(sources), &output);
   uint32_t accumulator_type = type(&m, &accumulator);
   uint32_t pad = vector(&m, 4, (int32_t[]){1, 1, 1, 1}, false);
   uint32_t st = vector(&m, 2, (int32_t[]){stride, stride}, true);
   uint32_t dil = vector(&m, 2, (int32_t[]){dilation, dilation}, true);
   uint32_t acc = constant(&m, 1), bound = boolean(&m, false), result = m.next++;
   uint32_t scale32 = boolean(&m, true), mode = constant(&m, 1);
   uint32_t flag = boolean(&m, false);
   uint32_t resized_type = 0, resize_scale = 0, resize_offset = 0, resize_border = 0;
   if (resize) {
      struct tensor resized = sources[0];
      resized.dims[1] *= 2;
      resized.dims[2] *= 2;
      resized_type = type(&m, &resized);
      resize_scale = vector(&m, 4, (int32_t[]){4, 2, 4, 2}, false);
      resize_offset = vector(&m, 2, (int32_t[]){-1, -1}, true);
      resize_border = vector(&m, 2, (int32_t[]){1, 1}, true);
   }
   begin(&m, sources, ARRAY_SIZE(sources));
   uint32_t input_value = m.values[0];
   if (resize) {
      input_value = m.next++;
      emit(&m, SpvOpExtInst, 9, resized_type, input_value, m.tosa, (uint32_t)PANVK_GRAPH_OP_RESIZE,
           mode, m.values[0], resize_scale, resize_offset, resize_border);
   }
   emit(&m, SpvOpExtInst, 14, accumulator_type, result, m.tosa, (uint32_t)PANVK_GRAPH_OP_CONV2D,
        pad, st, dil, acc, bound, input_value, m.values[1], m.values[2], m.values[3], m.values[4]);
   if (quantized) {
      uint32_t next = m.next++;
      emit(&m, SpvOpExtInst, 14, m.output_type, next, m.tosa, (uint32_t)PANVK_GRAPH_OP_RESCALE,
           scale32, mode, flag, flag, flag, result, m.values[5], m.values[6], m.values[7], m.values[3]);
      result = next;
   }
   evaluate(&m, result, sources, ARRAY_SIZE(sources), &output,
            quantized ? (const void *)scaled : expected, !dynamic_weights);
}

static void
check_rescale(unsigned rounding, bool per_channel, bool output_unsigned, unsigned shift_base)
{
   int32_t input[128], multipliers[4] = {INT32_MAX, 1073741824, 123456789, 0}, izp = 0;
   int8_t shifts[4], ozp = output_unsigned ? 125 : -17;
   uint8_t expected[128];
   for (unsigned i = 0; i < 4; i++) shifts[i] = MIN2(shift_base + i, 62);
   for (unsigned i = 0; i < 128; i++) {
      static const int32_t samples[] = {INT32_MIN, INT32_MAX, -1, 0, 1, -1073741825, 1073741825, 7919};
      unsigned c = per_channel ? i % 4 : 0;
      input[i] = samples[i / 4 % ARRAY_SIZE(samples)];
      if (shifts[c] < 32)
         input[i] %= 1 << (shifts[c] - 1);
      int64_t product = (int64_t)input[i] * multipliers[c];
      int64_t adjustment = INT64_C(1) << (shifts[c] - 1);
      if (rounding == 3 && shifts[c] > 31)
         adjustment += input[i] < 0 ? -(INT64_C(1) << 30) : INT64_C(1) << 30;
      int64_t scaled = ((product + adjustment) >> shifts[c]) +
                       (output_unsigned ? (uint8_t)ozp : ozp);
      expected[i] = CLAMP(scaled, output_unsigned ? 0 : -128, output_unsigned ? 255 : 127);
   }
   unsigned nc = per_channel ? 4 : 1;
   struct tensor sources[] = {
      {2, {32, 4}, VK_FORMAT_R32_SINT, input, true},
      {1, {nc}, VK_FORMAT_R32_SINT, multipliers},
      {1, {nc}, VK_FORMAT_R8_SINT, shifts},
      {1, {1}, VK_FORMAT_R32_SINT, &izp},
      {1, {1}, VK_FORMAT_R8_SINT, &ozp},
   };
   struct tensor output = {2, {32, 4}, VK_FORMAT_R8_SINT};
   struct module m;
   initialize(&m, sources, ARRAY_SIZE(sources), &output);
   uint32_t scale32 = boolean(&m, true), mode = constant(&m, rounding);
   uint32_t channel = boolean(&m, per_channel), iu = boolean(&m, false), ou = boolean(&m, output_unsigned);
   uint32_t result = m.next++;
   begin(&m, sources, ARRAY_SIZE(sources));
   emit(&m, SpvOpExtInst, 14, m.output_type, result, m.tosa, (uint32_t)PANVK_GRAPH_OP_RESCALE,
        scale32, mode, channel, iu, ou, m.values[0], m.values[1], m.values[2], m.values[3], m.values[4]);
   evaluate(&m, result, sources, ARRAY_SIZE(sources), &output, expected, 0);
}

static void
check_table(void)
{
   int8_t input[256], table[256], expected[256];
   for (unsigned i = 0; i < 256; i++) { input[i] = i; table[i] = (i * 17 + 31) & 255; }
   for (unsigned i = 0; i < 256; i++) expected[i] = table[(int)input[i] + 128];
   struct tensor sources[] = {
      {2, {16, 16}, VK_FORMAT_R8_SINT, input, true},
      {1, {256}, VK_FORMAT_R8_SINT, table},
   };
   struct tensor output = sources[0];
   struct module m;
   initialize(&m, sources, ARRAY_SIZE(sources), &output);
   uint32_t result = m.next++;
   begin(&m, sources, ARRAY_SIZE(sources));
   emit(&m, SpvOpExtInst, 6, m.output_type, result, m.tosa, (uint32_t)PANVK_GRAPH_OP_TABLE, m.values[0], m.values[1]);
   evaluate(&m, result, sources, ARRAY_SIZE(sources), &output, expected, 0);
}

static void
check_resize(unsigned mode, bool fp, unsigned scale, int offset)
{
   int8_t input[2 * 5 * 5 * 3];
   float floats[ARRAY_SIZE(input)], expected_fp[2 * 18 * 18 * 3];
   int32_t expected_i32[ARRAY_SIZE(expected_fp)];
   int8_t expected_i8[ARRAY_SIZE(expected_fp)];
   for (unsigned i = 0; i < ARRAY_SIZE(input); i++) {
      input[i] = (i * 67 + 19) & 255;
      floats[i] = input[i] * 0.125f;
   }
   int den = 2, border = (den - (4 * scale - offset) % den) % den;
   unsigned extent = (4 * scale - offset + border) / den + 1;
   assert(extent <= 18);
   struct tensor source = {4, {2, 5, 5, 3}, fp ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R8_SINT,
                           fp ? (const void *)floats : input, true};
   struct tensor output = {4, {2, extent, extent, 3}, fp ? VK_FORMAT_R32_SFLOAT :
                           mode == 1 ? VK_FORMAT_R8_SINT : VK_FORMAT_R32_SINT};
   for (unsigned n = 0; n < 2; n++)
      for (unsigned y = 0; y < extent; y++)
         for (unsigned x = 0; x < extent; x++)
            for (unsigned c = 0; c < 3; c++) {
               int py = y * den + offset, px = x * den + offset;
               int fy = (int)floor((double)py / scale), fx = (int)floor((double)px / scale);
               int ry = py - fy * scale, rx = px - fx * scale;
               unsigned dst = ((n * extent + y) * extent + x) * 3 + c;
               if (mode == 1) {
                  int sy = CLAMP(fy + (2 * ry >= scale), 0, 4);
                  int sx = CLAMP(fx + (2 * rx >= scale), 0, 4);
                  unsigned src = ((n * 5 + sy) * 5 + sx) * 3 + c;
                  expected_i8[dst] = input[src];
                  expected_fp[dst] = floats[src];
               } else {
                  int32_t sum = 0;
                  for (unsigned ky = 0; ky < 2; ky++)
                     for (unsigned kx = 0; kx < 2; kx++) {
                        unsigned src = ((n * 5 + CLAMP(fy + (int)ky, 0, 4)) * 5 +
                                         CLAMP(fx + (int)kx, 0, 4)) * 3 + c;
                        sum += input[src] * (ky ? ry : scale - ry) * (kx ? rx : scale - rx);
                     }
                  expected_i32[dst] = sum;
                  expected_fp[dst] = sum * (0.125f / (scale * scale));
               }
            }
   struct module m;
   initialize(&m, &source, 1, &output);
   uint32_t sc = vector(&m, 4, (int32_t[]){scale, den, scale, den}, false);
   uint32_t off = vector(&m, 2, (int32_t[]){offset, offset}, true);
   uint32_t brd = vector(&m, 2, (int32_t[]){border, border}, true);
   uint32_t method = constant(&m, mode), result = m.next++;
   begin(&m, &source, 1);
   emit(&m, SpvOpExtInst, 9, m.output_type, result, m.tosa, (uint32_t)PANVK_GRAPH_OP_RESIZE,
        method, m.values[0], sc, off, brd);
   evaluate(&m, result, &source, 1, &output,
            fp ? (void *)expected_fp : mode == 1 ? (void *)expected_i8 : expected_i32, 0);
}

static void
check_concat(unsigned rank, unsigned axis, unsigned count, bool narrow)
{
   uint8_t inputs[6][1024], expected[6144];
   struct tensor sources[6], output = {rank, {2, 2, 2, 2},
                                      narrow ? VK_FORMAT_R8_SINT : VK_FORMAT_R32_SINT};
   for (unsigned d = rank; d < 4; d++) output.dims[d] = 0;
   output.dims[axis] = 0;
   unsigned bytes = narrow ? 1 : 4, inner = 1, outer = 1;
   for (unsigned d = 0; d < rank; d++) {
      if (d < axis) outer *= 2;
      if (d > axis) inner *= 2;
   }
   for (unsigned i = 0; i < count; i++) {
      sources[i] = output;
      sources[i].dims[axis] = i + 1;
      sources[i].data = inputs[i];
      sources[i].input = true;
      output.dims[axis] += i + 1;
      for (unsigned j = 0; j < sizeof(inputs[i]); j++) inputs[i][j] = j * 13 + i * 71;
   }
   for (unsigned o = 0; o < outer; o++) {
      unsigned pos = o * output.dims[axis] * inner * bytes;
      for (unsigned i = 0; i < count; i++) {
         unsigned size = (i + 1) * inner * bytes;
         memcpy(expected + pos, inputs[i] + o * size, size);
         pos += size;
      }
   }
   struct module m;
   initialize(&m, sources, count, &output);
   uint32_t ax = constant(&m, axis), result = m.next++;
   begin(&m, sources, count);
   m.words[m.count++] = ((6 + count) << 16) | SpvOpExtInst;
   m.words[m.count++] = m.output_type;
   m.words[m.count++] = result;
   m.words[m.count++] = m.tosa;
   m.words[m.count++] = PANVK_GRAPH_OP_CONCAT;
   m.words[m.count++] = ax;
   for (unsigned i = 0; i < count; i++) m.words[m.count++] = m.values[i];
   evaluate(&m, result, sources, count, &output, expected, 0);
}

static void
check_concat_views(unsigned batch, unsigned height, unsigned width,
                   unsigned count, unsigned mode)
{
   int8_t input[6][2 * 5 * 13 * 48], table[256], expected[2 * 5 * 13 * 160];
   struct tensor sources[7], joined = {4, {batch, height, width, 0}, VK_FORMAT_R8_SINT};
   unsigned channels[6], elements = batch * height * width;
   for (unsigned i = 0; i < 256; i++)
      table[i] = (int8_t)((i * 73 + 19) ^ 0xa5);
   for (unsigned i = 0; i < count; i++) {
      channels[i] = mode == 1 || mode == 3 ? 16 : (unsigned[]){16, 32, 16, 48, 32, 16}[i];
      if (mode == 2 && !i)
         channels[i]--;
      sources[i] = joined;
      sources[i].dims[3] = channels[i];
      sources[i].data = input[i];
      sources[i].input = true;
      joined.dims[3] += channels[i];
      for (unsigned j = 0; j < elements * channels[i]; j++)
         input[i][j] = (int8_t)(j * 67 + i * 37 + 13);
   }
   sources[count] = (struct tensor){1, {256}, VK_FORMAT_R8_SINT, table};
   for (unsigned p = 0; p < elements; p++) {
      unsigned offset = p * joined.dims[3];
      for (unsigned i = 0; i < count; i++) {
         for (unsigned c = 0; c < channels[i]; c++) {
            int8_t value = mode == 1 && i == 1 ? table[input[0][p * 16 + c] + 128]
                           : mode == 3 ? input[0][p * 16 + c]
                                       : input[i][p * channels[i] + c];
            expected[offset++] = table[value + 128];
         }
      }
   }
   struct tensor output = {2, {batch, height * width * joined.dims[3]}, VK_FORMAT_R8_SINT};
   struct module m;
   initialize(&m, sources, count + 1, &output);
   uint32_t joined_type = type(&m, &joined), values[6];
   uint32_t axis = constant(&m, 3), shape = vector(&m, 2,
      (int32_t[]){output.dims[0], output.dims[1]}, false);
   begin(&m, sources, count + 1);
   for (unsigned i = 0; i < count; i++) {
      values[i] = m.next++;
      emit(&m, SpvOpExtInst, 6, m.types[i], values[i], m.tosa,
           (uint32_t)PANVK_GRAPH_OP_TABLE, mode == 1 && i == 1 ? values[0] : m.values[i],
           m.values[count]);
   }
   uint32_t concat = m.next++, result = m.next++;
   m.words[m.count++] = ((6 + count) << 16) | SpvOpExtInst;
   m.words[m.count++] = joined_type;
   m.words[m.count++] = concat;
   m.words[m.count++] = m.tosa;
   m.words[m.count++] = PANVK_GRAPH_OP_CONCAT;
   m.words[m.count++] = axis;
   for (unsigned i = 0; i < count; i++)
      m.words[m.count++] = mode == 3 ? values[0] : values[i];
   emit(&m, SpvOpExtInst, 6, m.output_type, result, m.tosa,
        (uint32_t)PANVK_GRAPH_OP_RESHAPE, concat, shape);
   required_views = mode == 2 || mode == 3 ? 0 : count;
   evaluate(&m, result, sources, count + 1, &output, expected, 0);
   required_views = 0;
}

int
main(void)
{
   for (unsigned batch = 1; batch <= 2; batch++)
      for (unsigned stride = 1; stride <= 2; stride++)
         for (unsigned dilation = 1; dilation <= 2; dilation++) {
            for (unsigned dynamic = 0; dynamic < 2; dynamic++)
               for (unsigned quantized = 0; quantized < 2; quantized++) {
                  check_conv(batch, stride, dilation, -128, 0, dynamic, quantized, false);
                  check_conv(batch, stride, dilation, 17, -13, dynamic, quantized, false);
               }
         }
   setenv("PANVK_GRAPH_NATIVE_CONV", "1", 1);
   setenv("PANVK_GRAPH_CONV_RESCALE", "1", 1);
   for (unsigned batch = 1; batch <= 2; batch++)
      for (unsigned dynamic = 0; dynamic < 2; dynamic++)
         for (unsigned quantized = 0; quantized < 2; quantized++) {
            check_conv(batch, 1, 1, -128, 0, dynamic, quantized, true);
            check_conv(batch, 1, 1, 17, -13, dynamic, quantized, true);
         }
   unsetenv("PANVK_GRAPH_NATIVE_CONV");
   unsetenv("PANVK_GRAPH_CONV_RESCALE");
   for (unsigned rounding = 1; rounding <= 3; rounding++)
      for (unsigned pc = 0; pc < 2; pc++)
         for (unsigned ou = 0; ou < 2; ou++)
            for (unsigned shift = 2; shift <= 62; shift += 3)
               check_rescale(rounding, pc, ou, shift);
   check_table();
   for (unsigned mode = 1; mode <= 2; mode++)
      for (unsigned fp = 0; fp < 2; fp++)
         for (unsigned scale = 2; scale <= (mode == 1 ? 6 : 4); scale += 2)
            for (int offset = -(int)scale; offset < (int)scale; offset++)
               check_resize(mode, fp, scale, offset);
   for (unsigned rank = 1; rank <= 4; rank++)
      for (unsigned axis = 0; axis < rank; axis++)
         for (unsigned count = 1; count <= 6; count++)
            for (unsigned narrow = 0; narrow < 2; narrow++)
               check_concat(rank, axis, count, narrow);
   setenv("PANVK_GRAPH_CONV_RESCALE", "1", 1);
   for (unsigned batch = 1; batch <= 2; batch++)
      for (unsigned h = 1; h <= 5; h += 2)
         for (unsigned w = 1; w <= 13; w += 6)
            for (unsigned count = 2; count <= 6; count++) {
               check_concat_views(batch, h, w, count, 0);
               check_concat_views(batch, h, w, count, 2);
               if (count == 2) {
                  check_concat_views(batch, h, w, count, 1);
                  check_concat_views(batch, h, w, count, 3);
               }
            }
   unsetenv("PANVK_GRAPH_CONV_RESCALE");
   printf("quantized cases=%u shaders=%u outputs=%u neural_plans=%u\n", cases, shaders, checks, neural_plans);
   return 0;
}
