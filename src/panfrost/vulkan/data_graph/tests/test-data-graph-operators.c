#include <float.h>
#include <math.h>
#include <stdio.h>
#include "compiler/spirv/spirv.h"
#include "util/ralloc.h"
#include "nir_builder.h"
#include "panvk_data_graph.h"
#include "vk_format.h"

#ifdef PANVK_GRAPH_TEST_ALLOC_FAILURE
static size_t allocation_calls, fail_at = SIZE_MAX;
static bool allocation_armed;
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

struct operator_shape {
   unsigned rank;
   int64_t dims[4];
};

struct operator_stage {
   enum panvk_graph_op op;
   struct operator_shape output;
   unsigned axis, nan_mode;
   unsigned kernel[2], stride[2], dilation[2], padding[4];
   unsigned bias_count;
   unsigned input, other;
};

struct operator_case {
   const char *name;
   const uint32_t *code;
   size_t size;
   unsigned input_count;
   struct operator_shape inputs[2];
   unsigned stage_count;
   struct operator_stage stages[6];
   unsigned data_mode;
   bool valid;
   int softmax_axis;
   unsigned pass_count, pass_masks[6];
   bool fp32_accumulate;
};

#include "data-graph-nir-interpreter.h"
#include "data-graph-operator-fixtures.h"
#include "data-graph-fusion-fixtures.h"
#include "data-graph-minmax-fixtures.h"

static unsigned
elements(const struct operator_shape *s)
{
   unsigned count = 1;
   for (unsigned d = 0; d < s->rank; d++)
      count *= s->dims[d];
   return count;
}

static void
coordinates(const struct operator_shape *s, unsigned index, unsigned *coords)
{
   for (int d = s->rank - 1; d >= 0; d--) {
      coords[d] = index % s->dims[d];
      index /= s->dims[d];
   }
   assert(!index);
}

static unsigned
offset(const struct operator_shape *s, const unsigned *coords)
{
   unsigned index = 0;
   for (unsigned d = 0; d < s->rank; d++) {
      assert(coords[d] < s->dims[d]);
      index = index * s->dims[d] + coords[d];
   }
   return index;
}

static float
input_value(unsigned i, unsigned resource, unsigned mode)
{
   if (mode == 1)
      return NAN;
   if (mode == 2) {
      const float special[] = {NAN,  INFINITY, -INFINITY, -0.0f,
                               0.0f, 0.25f,    -1.0f};
      return special[i % ARRAY_SIZE(special)];
   }
   if (mode == 3 && i % 3 == 0)
      return NAN;
   if (mode == 8 && i % 5 == 0)
      return INFINITY;
   if (mode == 9)
      return -INFINITY;
   if (mode == 10 && i % 3 == 0)
      return -INFINITY;
   if (mode == 11)
      return i % 3 ? -FLT_MAX : FLT_MAX;
   if (mode == 12)
      return i % 3 ? (i % 2 ? -FLT_TRUE_MIN : FLT_TRUE_MIN) : -0.0f;
   if (mode >= 5 && mode <= 7)
      return mode == 6 ? 10000.0f
                       : (mode == 5 ? 10000.0f : -10000.0f) +
                            ((int)(i * 37 % 31) - 15) * 0.125f;
   if (mode == 4) {
      const float special[] = {
         NAN,          INFINITY,      -INFINITY,   0.0f,         -0.0f,
         FLT_TRUE_MIN, -FLT_TRUE_MIN, FLT_MIN,     -FLT_MIN,     FLT_MAX,
         -FLT_MAX,     1000.0f,       -1000.0f,    90.0f,        -90.0f,
         0.49999997f,  0.5f,          0.50000006f, -0.49999997f, -0.5f,
         -0.50000006f, 0.000001f,     -0.000001f,
      };
      if (i < ARRAY_SIZE(special))
         return special[i];
      if (i % 3 == 0) {
         uint32_t bits =
            ((i * 2654435761u) & 0x807fffffu) | ((i % 254 + 1) << 23);
         float value;
         memcpy(&value, &bits, sizeof(value));
         return value;
      }
      return ((int)(i % 4097) - 2048) / (i % 2 ? 64.0f : 4096.0f);
   }
   return ((int)((i * 37 + resource * 11) % 31) - 15) / 16.0f;
}

static double
extremum(double a, double b, bool maximum, unsigned nan_mode)
{
   if (nan_mode == 1 && (isnan(a) || isnan(b)))
      return NAN;
   return maximum ? fmax(a, b) : fmin(a, b);
}

static float *
reference_stage(const struct operator_case *c, unsigned stage,
                const struct operator_shape *input_shape, const float *input,
                const float *weights, const struct operator_shape *other_shape,
                const float *other, VkFormat format)
{
   const struct operator_stage *s = &c->stages[stage];
   float *output = calloc(elements(&s->output), sizeof(float));
   assert(output);
   for (unsigned i = 0; i < elements(&s->output); i++) {
      unsigned coords[4] = {0};
      coordinates(&s->output, i, coords);
      if (format != VK_FORMAT_R32_SFLOAT) {
         assert(s->op >= PANVK_GRAPH_OP_REDUCE_MAX &&
                s->op <= PANVK_GRAPH_OP_REDUCE_SUM);
         uint32_t result = s->op == PANVK_GRAPH_OP_REDUCE_PRODUCT;
         for (unsigned k = 0; k < input_shape->dims[s->axis]; k++) {
            coords[s->axis] = k;
            uint32_t value;
            memcpy(&value, input + offset(input_shape, coords), sizeof(value));
            if (s->op == PANVK_GRAPH_OP_REDUCE_MAX ||
                s->op == PANVK_GRAPH_OP_REDUCE_MIN) {
               bool maximum = s->op == PANVK_GRAPH_OP_REDUCE_MAX;
               bool greater = (int32_t)value > (int32_t)result;
               if (!k || greater == maximum)
                  result = value;
            } else if (s->op == PANVK_GRAPH_OP_REDUCE_SUM) {
               result += value;
            } else {
               result *= value;
            }
         }
         memcpy(output + i, &result, sizeof(result));
         continue;
      }
      double result = 0;
      if (s->op == PANVK_GRAPH_OP_AVG_POOL2D ||
          s->op == PANVK_GRAPH_OP_CONV2D ||
          s->op == PANVK_GRAPH_OP_MAX_POOL2D) {
         bool conv = s->op == PANVK_GRAPH_OP_CONV2D;
         result = s->op == PANVK_GRAPH_OP_MAX_POOL2D ? NAN : 0;
         unsigned samples = 0;
         for (unsigned ky = 0; ky < s->kernel[0]; ky++) {
            for (unsigned kx = 0; kx < s->kernel[1]; kx++) {
               int y = coords[1] * s->stride[0] + ky * s->dilation[0] -
                       s->padding[0];
               int x = coords[2] * s->stride[1] + kx * s->dilation[1] -
                       s->padding[2];
               if (y < 0 || x < 0 || y >= input_shape->dims[1] ||
                   x >= input_shape->dims[2])
                  continue;
               for (unsigned ic = 0; ic < (conv ? input_shape->dims[3] : 1);
                    ic++) {
                  unsigned ac[] = {coords[0], y, x, conv ? ic : coords[3]};
                  double value = input[offset(input_shape, ac)];
                  if (conv) {
                     unsigned wc[] = {coords[3], ky, kx, ic};
                     result += value * weights[offset(&c->inputs[1], wc)];
                  } else if (s->op == PANVK_GRAPH_OP_AVG_POOL2D) {
                     result += value;
                  } else {
                     result = samples
                                 ? extremum(result, value, true, s->nan_mode)
                                 : value;
                  }
                  samples++;
               }
            }
         }
         if (conv)
            result += ((int)(s->bias_count == 1 ? 0 : coords[3]) - 2) * 0.125;
         else if (s->op == PANVK_GRAPH_OP_AVG_POOL2D) {
            assert(samples);
            result /= samples;
         }
      } else if (s->op >= PANVK_GRAPH_OP_REDUCE_MAX &&
                 s->op <= PANVK_GRAPH_OP_REDUCE_SUM) {
         result = s->op == PANVK_GRAPH_OP_REDUCE_PRODUCT ? 1 : 0;
         for (unsigned k = 0; k < input_shape->dims[s->axis]; k++) {
            coords[s->axis] = k;
            double value = input[offset(input_shape, coords)];
            if (s->op == PANVK_GRAPH_OP_REDUCE_MAX ||
                s->op == PANVK_GRAPH_OP_REDUCE_MIN)
               result =
                  k ? extremum(result, value,
                               s->op == PANVK_GRAPH_OP_REDUCE_MAX, s->nan_mode)
                    : value;
            else if (s->op == PANVK_GRAPH_OP_REDUCE_PRODUCT)
               result *= value;
            else {
               result += value;
               if (c->fp32_accumulate)
                  result = (float)result;
            }
         }
      } else if (s->op == PANVK_GRAPH_OP_SIGMOID) {
         double x = input[i];
         result = x < 0 ? exp(x) / (1 + exp(x)) : 1 / (1 + exp(-x));
      } else if (s->op == PANVK_GRAPH_OP_TANH) {
         result = tanh(input[i]);
      } else if (s->op == PANVK_GRAPH_OP_EXP) {
         result = exp(input[i]);
      } else if (s->op == PANVK_GRAPH_OP_RECIPROCAL) {
         result = 1.0 / input[i];
      } else if (s->op == PANVK_GRAPH_OP_ADD || s->op == PANVK_GRAPH_OP_SUB ||
                 s->op == PANVK_GRAPH_OP_MUL) {
         unsigned ac[4], bc[4];
         for (unsigned d = 0; d < s->output.rank; d++) {
            ac[d] = input_shape->dims[d] == 1 ? 0 : coords[d];
            bc[d] = other_shape->dims[d] == 1 ? 0 : coords[d];
         }
         double a = input[offset(input_shape, ac)];
         double b = other[offset(other_shape, bc)];
          result = s->op == PANVK_GRAPH_OP_ADD    ? a + b
                   : s->op == PANVK_GRAPH_OP_SUB ? a - b
                                                : a * b;
      } else {
         assert(s->op == PANVK_GRAPH_OP_CLAMP);
         result = isnan(input[i]) ? NAN : fmax(0, input[i]);
      }
      output[i] = result;
   }
   return output;
}

static bool
matches(float actual, double expected, enum panvk_graph_op op,
        float activation_input)
{
   if (isnan(expected))
      return isnan(actual);
   if (isinf(expected))
      return actual == expected;
   if (op == PANVK_GRAPH_OP_TANH && expected == 0 && activation_input == 0)
      return actual == 0 && !!signbit(actual) == !!signbit(activation_input);
   if (op == PANVK_GRAPH_OP_RECIPROCAL && expected == 0)
      return actual == 0 && !!signbit(actual) == !!signbit(expected);
   double bound = 8 * FLT_EPSILON * fmax(fabs(expected), FLT_MIN);
   if (op == PANVK_GRAPH_OP_SIGMOID || op == PANVK_GRAPH_OP_TANH) {
      if (isinf(activation_input))
         return actual == expected;
      double factor = op == PANVK_GRAPH_OP_SIGMOID
                         ? 2 * (1 + fabs(activation_input))
                         : 4 * (3 + 2 * fabs(activation_input));
      bound = FLT_EPSILON *
              fmax(fmax(fabs(expected), FLT_MIN),
                   op == PANVK_GRAPH_OP_TANH ? 0.5 / factor : 0) *
              factor;
   } else if (op == PANVK_GRAPH_OP_EXP) {
      bound = FLT_EPSILON * fmax(fabs(expected), FLT_MIN) *
              (3 + 2 * fabs(activation_input));
   }
   return fabs((double)actual - expected) <= bound ||
          (actual == 0 && fabs(expected) < FLT_MIN);
}

static VkResult
parse_case(const struct operator_case *c, VkFormat format,
           struct panvk_data_graph **graph)
{
   VkTensorDescriptionARM descs[3];
   VkDataGraphPipelineResourceInfoARM resources[3];
   for (unsigned i = 0; i <= c->input_count; i++) {
      const struct operator_shape *s =
         i < c->input_count ? &c->inputs[i]
                            : &c->stages[c->stage_count - 1].output;
      descs[i] = (VkTensorDescriptionARM){
         .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
         .format = format,
         .dimensionCount = s->rank,
         .pDimensions = s->dims,
         .tiling = VK_TENSOR_TILING_LINEAR_ARM,
         .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
      };
      resources[i] = (VkDataGraphPipelineResourceInfoARM){
         .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
         .pNext = &descs[i],
         .binding = i,
      };
   }
   VkDataGraphPipelineCreateInfoARM info = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
      .resourceInfoCount = c->input_count + 1,
      .pResourceInfos = resources,
   };
   VkDataGraphPipelineShaderModuleCreateInfoARM module = {
      .sType =
         VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
      .pName = "main",
   };
   return panvk_data_graph_parse(NULL, c->code, c->size, &info, &module, graph);
}

static unsigned
check_case(const struct operator_case *c, unsigned *shaders, VkFormat format)
{
   struct panvk_data_graph *g = NULL;
   VkResult result = parse_case(c, format, &g);
   if (!c->valid) {
      if (result != VK_ERROR_FEATURE_NOT_PRESENT || g)
         fprintf(stderr, "negative case=%s result=%d graph=%p\n", c->name,
                 result, (void *)g);
      assert(result == VK_ERROR_FEATURE_NOT_PRESENT && !g);
      return 0;
   }
   if (result != VK_SUCCESS) {
      fprintf(stderr, "parse failed case=%s result=%d\n", c->name, result);
      abort();
   }
   if (c->pass_count) {
      assert(g->pass_count == c->pass_count);
      unsigned masks[6] = {0}, stage = 0;
      for (unsigned i = 0; i < g->count; i++) {
         const struct panvk_graph_value *v = &g->values[i];
         if (v->kind != PANVK_GRAPH_OPERATION)
            continue;
         assert(stage < c->stage_count);
         unsigned root = v->fused_into == UINT32_MAX ? i : v->fused_into;
         for (unsigned p = 0; p < g->pass_count; p++) {
            if (g->passes[p] == root)
               masks[p] |= 1u << stage;
         }
         stage++;
      }
      assert(stage == c->stage_count);
      for (unsigned p = 0; p < c->pass_count; p++) {
         if (masks[p] != c->pass_masks[p])
            fprintf(stderr, "fusion case=%s pass=%u mask=%x expected=%x\n",
                    c->name, p, masks[p], c->pass_masks[p]);
         assert(masks[p] == c->pass_masks[p]);
      }
   }
   float *inputs[2] = {0};
   for (unsigned r = 0; r < c->input_count; r++) {
      inputs[r] = calloc(elements(&c->inputs[r]), sizeof(float));
      assert(inputs[r]);
      for (unsigned i = 0; i < elements(&c->inputs[r]); i++) {
         if (format == VK_FORMAT_R32_SFLOAT) {
            inputs[r][i] = input_value(i, r, c->data_mode);
         } else {
            const uint32_t values[] = {0,          1,          0xffffffff,
                                       0x80000000, 0x7fffffff, 0xffff0000,
                                       17,         0x80000001, 0xff000001};
            memcpy(inputs[r] + i, &values[i % ARRAY_SIZE(values)],
                   sizeof(uint32_t));
         }
      }
   }
   float *references[7] = {inputs[0]};
   const struct operator_shape *shapes[7] = {&c->inputs[0]};
   for (unsigned stage = 0; stage < c->stage_count; stage++) {
      const struct operator_stage *s = &c->stages[stage];
      assert(s->input <= stage && s->other <= stage);
      references[stage + 1] = reference_stage(
         c, stage, shapes[s->input], references[s->input], inputs[1],
         shapes[s->other], references[s->other], format);
      shapes[stage + 1] = &s->output;
   }
   const struct operator_stage *last = &c->stages[c->stage_count - 1];
   const struct operator_shape *shape = &last->output;
   const float *reference = references[c->stage_count];
   const float *activation_input = references[last->input];
   if (c->softmax_axis >= 0)
      assert(g->pass_count == 4);
   else if (!strcmp(c->name, "conv_relu_sigmoid"))
      assert(g->pass_count == 1);
   else if (!strcmp(c->name, "pool_reduce_tanh"))
      assert(g->pass_count == 2);
   else if (!strcmp(c->name, "conv_pool_reduce_sigmoid"))
      assert(g->pass_count == 3);
   uint8_t **buffers = calloc(g->count, sizeof(*buffers));
   uint8_t *transient = calloc(MAX2(g->transient_size, 1), 1);
   float *output = calloc(elements(shape), sizeof(float));
   assert(buffers && transient && output);
   for (unsigned i = 0; i < g->count; i++) {
      const struct panvk_graph_value *v = &g->values[i];
      switch (v->kind) {
      case PANVK_GRAPH_INPUT:
         buffers[i] = (uint8_t *)inputs[v->resource_index];
         break;
      case PANVK_GRAPH_CONSTANT:
         buffers[i] = v->data;
         break;
      case PANVK_GRAPH_OPERATION:
         if (v->live && v->fused_into == UINT32_MAX) {
            assert(v->offset + v->size <= g->transient_size);
            buffers[i] = transient + v->offset;
         }
         break;
      case PANVK_GRAPH_OUTPUT:
         buffers[i] = (uint8_t *)output;
         break;
      }
   }
   for (unsigned pass = 0; pass < g->pass_count; pass++) {
      nir_shader *nir = panvk_data_graph_shader(g, pass);
      assert(nir && !nir->info.internal);
      nir_validate_shader(nir, "operator numeric execution");
      struct graph_interpreter e;
      graph_interpreter_init(&e, g, nir, buffers);
      unsigned count = g->values[g->passes[pass]].elements;
      for (unsigned index = 0; index < count; index++)
         assert(graph_interpret(&e, index) == 1);
      assert(!graph_interpret(&e, count));
      assert(!graph_interpret(&e, ALIGN_POT(count, 64) + 63));
      free(e.values);
      ralloc_free(nir);
      (*shaders)++;
   }
   unsigned checked = elements(shape);
   for (unsigned i = 0; i < checked; i++) {
      if (format != VK_FORMAT_R32_SFLOAT) {
         assert(!memcmp(output + i, reference + i, sizeof(uint32_t)));
         continue;
      }
      double expected = reference[i];
      float x = 0;
      if (last->op == PANVK_GRAPH_OP_SIGMOID ||
          last->op == PANVK_GRAPH_OP_TANH || last->op == PANVK_GRAPH_OP_EXP ||
          last->op == PANVK_GRAPH_OP_RECIPROCAL) {
         x = activation_input[i];
         expected = last->op == PANVK_GRAPH_OP_TANH         ? tanh(x)
                    : last->op == PANVK_GRAPH_OP_EXP        ? exp(x)
                    : last->op == PANVK_GRAPH_OP_RECIPROCAL ? 1.0 / x
                    : x < 0 ? exp(x) / (1 + exp(x))
                            : 1 / (1 + exp(-x));
         if (expected > FLT_MAX)
            expected = INFINITY;
         else if (expected < -FLT_MAX)
            expected = -INFINITY;
      }
      if (!matches(output[i], expected, last->op, x)) {
         fprintf(stderr,
                 "case=%s mode=%u index=%u actual=%a expected=%a input=%a\n",
                 c->name, c->data_mode, i, output[i], expected, x);
         abort();
      }
   }
   if (c->softmax_axis >= 0) {
      unsigned axis = c->softmax_axis;
      for (unsigned i = 0; i < checked; i++) {
         unsigned coords[4] = {0};
         coordinates(shape, i, coords);
         if (coords[axis])
            continue;
         double maximum = -INFINITY, total = 0, sum = 0;
         bool has_nan = false;
         for (unsigned k = 0; k < shape->dims[axis]; k++) {
            coords[axis] = k;
            float value = inputs[0][offset(shape, coords)];
            has_nan |= isnan(value);
            maximum = fmax(maximum, value);
         }
         if (has_nan || isinf(maximum)) {
            for (unsigned k = 0; k < shape->dims[axis]; k++) {
               coords[axis] = k;
               assert(isnan(output[offset(shape, coords)]));
            }
            continue;
         }
         for (unsigned k = 0; k < shape->dims[axis]; k++) {
            coords[axis] = k;
            unsigned index = offset(shape, coords);
            total += exp(inputs[0][index] - maximum);
            assert(isfinite(output[index]) && output[index] >= 0);
            sum += output[index];
         }
         assert(fabs(sum - 1) <= 16 * FLT_EPSILON);
         for (unsigned k = 0; k < shape->dims[axis]; k++) {
            coords[axis] = k;
            unsigned index = offset(shape, coords);
            double expected = exp(inputs[0][index] - maximum) / total;
            assert(fabs(output[index] - expected) <=
                   16 * FLT_EPSILON * expected);
         }
      }
   }
   for (unsigned stage = 1; stage <= c->stage_count; stage++)
      free(references[stage]);
   free(output);
   free(transient);
   free(buffers);
   for (unsigned r = 0; r < c->input_count; r++)
      free(inputs[r]);
   ralloc_free(g);
   return checked;
}

#ifdef PANVK_GRAPH_TEST_ALLOC_FAILURE
int
main(void)
{
   unsigned graphs = 0, failures = 0;
   for (unsigned c = 0;
        c < ARRAY_SIZE(operator_cases) + ARRAY_SIZE(fusion_cases); c++) {
      const struct operator_case *test =
         c < ARRAY_SIZE(operator_cases)
            ? &operator_cases[c]
            : &fusion_cases[c - ARRAY_SIZE(operator_cases)];
      if (!test->valid)
         continue;
      struct panvk_data_graph *g;
      allocation_armed = true;
      allocation_calls = 0;
      fail_at = SIZE_MAX;
      VkResult result = parse_case(test, VK_FORMAT_R32_SFLOAT, &g);
      allocation_armed = false;
      assert(result == VK_SUCCESS && g && allocation_calls);
      size_t count = allocation_calls;
      ralloc_free(g);
      for (size_t i = 0; i < count; i++) {
         allocation_calls = 0;
         fail_at = i;
         g = (void *)1;
         allocation_armed = true;
         result = parse_case(test, VK_FORMAT_R32_SFLOAT, &g);
         allocation_armed = false;
         if (result != VK_ERROR_OUT_OF_HOST_MEMORY || g)
            fprintf(stderr, "OOM case=%s allocation=%zu/%zu result=%d\n",
                    test->name, i, count, result);
         assert(result == VK_ERROR_OUT_OF_HOST_MEMORY && !g);
         fail_at = SIZE_MAX;
         assert(parse_case(test, VK_FORMAT_R32_SFLOAT, &g) == VK_SUCCESS);
         ralloc_free(g);
         failures++;
      }
      graphs++;
   }
   printf("operator OOM graphs=%u failures=%u\n", graphs, failures);
   return 0;
}
#else
static unsigned
check_clamp_nan(void)
{
   const uint32_t patterns[] = {
      0,          0x80000000, 1,          0x80000001, 0x007fffff, 0x807fffff,
      0x00800000, 0x80800000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
      0x40400000, 0xc0400000, 0x40800000, 0xc0800000, 0x7f7fffff, 0xff7fffff,
      0x7f800000, 0xff800000, 0x7fc01234, 0xffc05678, 0x7f800001,
   };
   const uint32_t *codes[] = {graph_clamp_mode_0, graph_clamp_mode_1,
                               graph_clamp_mode_2};
   const size_t sizes[] = {sizeof(graph_clamp_mode_0),
                           sizeof(graph_clamp_mode_1),
                           sizeof(graph_clamp_mode_2)};
   unsigned checked = 0;
   for (unsigned mode = 0; mode < ARRAY_SIZE(codes); mode++) {
      struct operator_case c = {
         .code = codes[mode],
         .size = sizes[mode],
         .input_count = 1,
         .inputs = {{1, {ARRAY_SIZE(patterns)}}},
         .stage_count = 1,
         .stages = {{.op = PANVK_GRAPH_OP_CLAMP,
                      .output = {1, {ARRAY_SIZE(patterns)}}}},
      };
      struct panvk_data_graph *g = NULL;
      assert(parse_case(&c, VK_FORMAT_R32_SFLOAT, &g) == VK_SUCCESS);
      assert(g && g->pass_count == 1);
      uint8_t **buffers = calloc(g->count, sizeof(*buffers));
      assert(buffers);
      float *output = NULL;
      for (unsigned i = 0; i < g->count; i++) {
         const struct panvk_graph_value *v = &g->values[i];
         buffers[i] = calloc(v->size, 1);
         assert(buffers[i]);
         if (v->kind == PANVK_GRAPH_INPUT)
            memcpy(buffers[i], patterns, sizeof(patterns));
         else if (v->kind == PANVK_GRAPH_OUTPUT)
            output = (float *)buffers[i];
      }
      assert(output);
      nir_shader *nir = panvk_data_graph_shader(g, 0);
      assert(nir);
      nir_validate_shader(nir, "clamp NaN-mode execution");
      struct graph_interpreter e;
      graph_interpreter_init(&e, g, nir, buffers);
      for (unsigned i = 0; i < ARRAY_SIZE(patterns); i++)
         assert(graph_interpret(&e, i) == 1);
      assert(!graph_interpret(&e, ARRAY_SIZE(patterns)));
      for (unsigned i = 0; i < ARRAY_SIZE(patterns); i++) {
         float input;
         memcpy(&input, patterns + i, sizeof(input));
         float expected = isnan(input) ? (mode == 2 ? -2.0f : NAN)
                          : input < -2 ? -2.0f
                          : input > 3  ? 3.0f
                                       : input;
         bool valid = isnan(expected)
                         ? isnan(output[i])
                         : !memcmp(output + i, &expected, sizeof(expected));
         if (!valid)
            fprintf(stderr,
                    "clamp mode=%u input=%08x actual=%a expected=%a\n",
                    mode, patterns[i], output[i], expected);
         assert(valid);
         checked++;
      }
      free(e.values);
      ralloc_free(nir);
      for (unsigned i = 0; i < g->count; i++)
         free(buffers[i]);
      free(buffers);
      ralloc_free(g);
   }
   return checked;
}

int
main(void)
{
   unsigned outputs = 0, shaders = 0, positive = 0, negative = 0, integer = 0;
   unsigned softmax_special = 0, fusion = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(operator_cases); i++) {
      const struct operator_case *c = &operator_cases[i];
      outputs += check_case(c, &shaders, VK_FORMAT_R32_SFLOAT);
      positive += operator_cases[i].valid;
      negative += !operator_cases[i].valid;
      if (c->softmax_axis >= 0 && !c->data_mode) {
         const unsigned modes[] = {1, 2, 3, 8, 9, 10, 11, 12};
         for (unsigned m = 0; m < ARRAY_SIZE(modes); m++) {
            struct operator_case variant = *c;
            variant.data_mode = modes[m];
            outputs += check_case(&variant, &shaders, VK_FORMAT_R32_SFLOAT);
            softmax_special++;
         }
      }
      if (!c->valid || c->data_mode || c->stage_count != 1 ||
          c->stages[0].nan_mode != 1 ||
          c->stages[0].op < PANVK_GRAPH_OP_REDUCE_MAX ||
          c->stages[0].op > PANVK_GRAPH_OP_REDUCE_SUM)
         continue;
      for (unsigned signedness = 0; signedness < 2; signedness++) {
         struct operator_case variant = *c;
         uint32_t *code = calloc(c->size / 4 + 1, sizeof(uint32_t));
         assert(code);
         memcpy(code, c->code, 5 * sizeof(uint32_t));
         uint32_t uint_type = 0, float_type = 0;
         for (unsigned w = 5; w < c->size / 4; w += c->code[w] >> 16) {
            unsigned op = c->code[w] & 0xffff;
            if (op == SpvOpTypeInt && c->code[w + 2] == 32 &&
                c->code[w + 3] == 0)
               uint_type = c->code[w + 1];
            if (op == SpvOpTypeFloat)
               float_type = c->code[w + 1];
         }
         assert(uint_type && float_type);
         unsigned out = 5;
         for (unsigned w = 5; w < c->size / 4;) {
            unsigned count = c->code[w] >> 16, op = c->code[w] & 0xffff;
            if (op == SpvOpTypeFloat) {
               assert(count == 3 && c->code[w + 2] == 32);
               if (signedness) {
                  code[out++] = (4 << 16) | SpvOpTypeInt;
                  code[out++] = c->code[w + 1];
                  code[out++] = 32;
                  code[out++] = signedness;
               }
            } else {
               memcpy(code + out, c->code + w, count * sizeof(uint32_t));
               unsigned type = op == SpvOpTypeTensorARM ? 2
                               : op == SpvOpConstant ||
                                    op == SpvOpConstantNull ||
                                    op == SpvOpConstantComposite
                                  ? 1
                                  : 0;
               if (!signedness && type && code[out + type] == float_type)
                  code[out + type] = uint_type;
               out += count;
            }
            w += count;
         }
         variant.code = code;
         variant.size = out * sizeof(uint32_t);
         const VkFormat formats[] = {VK_FORMAT_R32_SINT, VK_FORMAT_R32_UINT};
         for (unsigned f = 0; f < ARRAY_SIZE(formats); f++) {
            outputs += check_case(&variant, &shaders, formats[f]);
            integer++;
         }
         free(code);
      }
   }
   for (unsigned i = 0; i < ARRAY_SIZE(fusion_cases); i++) {
      const unsigned modes[] = {0, 1, 2, 3, 8, 9, 10, 11, 12};
      for (unsigned m = 0; m < ARRAY_SIZE(modes); m++) {
         struct operator_case variant = fusion_cases[i];
         variant.data_mode = modes[m];
         outputs += check_case(&variant, &shaders, VK_FORMAT_R32_SFLOAT);
         fusion++;
      }
   }
   unsigned clamp_nan = check_clamp_nan();
   printf(
      "operators positive=%u integer=%u softmax_special=%u fusion=%u clamp_nan=%u rejected=%u shaders=%u outputs=%u\n",
      positive, integer, softmax_special, fusion, clamp_nan, negative, shaders,
      outputs);
   return 0;
}
#endif
