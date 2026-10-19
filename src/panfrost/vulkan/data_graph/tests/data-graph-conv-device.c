#define main tensor_test_main
#include "../../tests/tensor-device.c"
#undef main
#include <assert.h>
#include <stdarg.h>
#include <stdbool.h>
#include "compiler/spirv/spirv.h"

FN(CreateDataGraphPipelinesARM);
FN(CreateDataGraphPipelineSessionARM);
FN(DestroyDataGraphPipelineSessionARM);
FN(GetDataGraphPipelineSessionBindPointRequirementsARM);
FN(GetDataGraphPipelineSessionMemoryRequirementsARM);
FN(BindDataGraphPipelineSessionMemoryARM);
FN(CmdDispatchDataGraphARM);
FN(CmdPipelineBarrier2);

struct conv_case {
   unsigned id, batch, stride, dilation, pad, ci, co, height, width, oh, ow;
   unsigned padding[4];
   int8_t izp, wzp;
   int8_t *input, *weights;
   int32_t *bias, *expected;
   bool rescale, per_channel, double_round, lookup, resize, concat, chain;
   int8_t table[256];
   int8_t ozp, *shift;
   int32_t *multiplier;
   unsigned input_size, weight_size, elements;
};

struct graph_module {
   uint32_t words[2048], count, next, u32, i32, i8, boolean;
};

static void
emit(struct graph_module *m, SpvOp op, unsigned count, ...)
{
   assert(m->count + count + 1 <= sizeof(m->words) / sizeof(m->words[0]));
   m->words[m->count++] = ((count + 1) << 16) | op;
   va_list args;
   va_start(args, count);
   for (unsigned i = 0; i < count; i++)
      m->words[m->count++] = va_arg(args, uint32_t);
   va_end(args);
}

static void
emit_string(struct graph_module *m, SpvOp op, uint32_t id, const char *text)
{
   unsigned count = (strlen(text) + 4) / 4;
   m->words[m->count++] = ((count + 1 + (id != 0)) << 16) | op;
   if (id)
      m->words[m->count++] = id;
   memset(&m->words[m->count], 0, count * 4);
   memcpy(&m->words[m->count], text, strlen(text));
   m->count += count;
}

static uint32_t
constant(struct graph_module *m, uint32_t value)
{
   uint32_t id = m->next++;
   emit(m, SpvOpConstant, 3, m->u32, id, value);
   return id;
}

static uint32_t
vector(struct graph_module *m, unsigned count, const int64_t *values)
{
   uint32_t length = constant(m, count), array = m->next++, ids[4];
   emit(m, SpvOpTypeArray, 3, array, m->u32, length);
   for (unsigned i = 0; i < count; i++)
      ids[i] = constant(m, values[i]);
   uint32_t id = m->next++;
   m->words[m->count++] = ((count + 3) << 16) | SpvOpConstantComposite;
   m->words[m->count++] = array;
   m->words[m->count++] = id;
   for (unsigned i = 0; i < count; i++)
      m->words[m->count++] = ids[i];
   return id;
}

static uint32_t
tensor_type(struct graph_module *m, uint32_t element, unsigned rank,
            const int64_t *dims)
{
   uint32_t r = constant(m, rank), shape = vector(m, rank, dims), id = m->next++;
   emit(m, SpvOpTypeTensorARM, 4, id, element, r, shape);
   return id;
}

static void
build_module(const struct conv_case *c, struct graph_module *m)
{
   *m = (struct graph_module){.count = 5, .next = 1};
   m->words[0] = SpvMagicNumber;
   m->words[1] = 0x10600;
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityShader);
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityInt8);
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityVulkanMemoryModel);
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityGraphARM);
   emit(m, SpvOpCapability, 1, (uint32_t)SpvCapabilityTensorsARM);
   emit_string(m, SpvOpExtension, 0, "SPV_ARM_graph");
   emit_string(m, SpvOpExtension, 0, "SPV_ARM_tensors");
   uint32_t tosa = m->next++;
   emit_string(m, SpvOpExtInstImport, tosa, "TOSA.001000.1");
   emit(m, SpvOpMemoryModel, 2, (uint32_t)SpvAddressingModelLogical,
        (uint32_t)SpvMemoryModelVulkan);
   uint32_t variables[2] = {m->next++, m->next++};
   for (unsigned i = 0; i < 2; i++) {
      emit(m, SpvOpDecorate, 3, variables[i], (uint32_t)SpvDecorationDescriptorSet, 0u);
      emit(m, SpvOpDecorate, 3, variables[i], (uint32_t)SpvDecorationBinding, i);
   }
   m->u32 = m->next++;
   m->i32 = m->next++;
   m->i8 = m->next++;
   m->boolean = m->next++;
   emit(m, SpvOpTypeInt, 3, m->u32, 32u, 0u);
   emit(m, SpvOpTypeInt, 3, m->i32, 32u, 1u);
   emit(m, SpvOpTypeInt, 3, m->i8, 8u, 1u);
   emit(m, SpvOpTypeBool, 1, m->boolean);
   uint32_t input = tensor_type(m, m->i8, 4, (int64_t[]){c->batch, c->height, c->width, c->ci});
   uint32_t resized = c->resize ? tensor_type(m, m->i8, 4,
      (int64_t[]){c->batch, c->height * 2, c->width * 2, c->ci}) : input;
   uint32_t output = tensor_type(m, m->i32, 4, (int64_t[]){c->batch, c->oh, c->ow, c->co});
   uint32_t final_output = c->rescale ?
      tensor_type(m, m->i8, 4, (int64_t[]){c->batch, c->oh, c->ow, c->co}) : output;
   uint32_t graph_output = c->concat ? tensor_type(m, m->i8, 4,
      (int64_t[]){c->batch, c->oh, c->ow, 2 * c->co}) : final_output;
   uint32_t weight = tensor_type(m, m->i8, 4, (int64_t[]){c->co, 3, 3, c->ci});
   uint32_t bias = tensor_type(m, m->i32, 1, (int64_t[]){c->co});
   uint32_t zp = tensor_type(m, m->i8, 1, (int64_t[]){1});
   uint32_t types[2] = {input, graph_output};
   for (unsigned i = 0; i < 2; i++) {
      uint32_t pointer = m->next++;
      emit(m, SpvOpTypePointer, 3, pointer, (uint32_t)SpvStorageClassUniformConstant, types[i]);
      emit(m, SpvOpVariable, 3, pointer, variables[i], (uint32_t)SpvStorageClassUniformConstant);
   }
   uint32_t weights = m->next++, biases = m->next++, izp = m->next++, wzp = m->next++;
   emit(m, SpvOpGraphConstantARM, 3, weight, weights, 1u);
   emit(m, SpvOpGraphConstantARM, 3, bias, biases, 2u);
   emit(m, SpvOpGraphConstantARM, 3, zp, izp, 3u);
   emit(m, SpvOpGraphConstantARM, 3, zp, wzp, 4u);
   uint32_t multiplier = 0, shift = 0, input_zero = 0, output_zero = 0;
   uint32_t scale32 = 0, rounding = 0, per_channel = 0;
   uint32_t table = 0;
   if (c->rescale) {
      unsigned count = c->per_channel ? c->co : 1;
      uint32_t mt = tensor_type(m, m->i32, 1, (int64_t[]){count});
      uint32_t st = tensor_type(m, m->i8, 1, (int64_t[]){count});
      uint32_t zt = tensor_type(m, m->i32, 1, (int64_t[]){1});
      multiplier = m->next++; shift = m->next++;
      input_zero = m->next++; output_zero = m->next++;
      emit(m, SpvOpGraphConstantARM, 3, mt, multiplier, 5u);
      emit(m, SpvOpGraphConstantARM, 3, st, shift, 6u);
      emit(m, SpvOpGraphConstantARM, 3, zt, input_zero, 7u);
      emit(m, SpvOpGraphConstantARM, 3, zp, output_zero, 8u);
      scale32 = m->next++;
      emit(m, SpvOpConstantTrue, 2, m->boolean, scale32);
      rounding = constant(m, c->double_round ? 3 : 1);
      per_channel = m->next++;
      emit(m, c->per_channel ? SpvOpConstantTrue : SpvOpConstantFalse,
           2, m->boolean, per_channel);
   }
   if (c->lookup) {
      uint32_t tt = tensor_type(m, m->i8, 1, (int64_t[]){256});
      table = m->next++;
      emit(m, SpvOpGraphConstantARM, 3, tt, table, 9u);
   }
   uint32_t pad = vector(m, 4, (int64_t[]){c->padding[0], c->padding[1],
                                         c->padding[2], c->padding[3]});
   uint32_t stride = vector(m, 2, (int64_t[]){c->stride, c->stride});
   uint32_t dilation = vector(m, 2, (int64_t[]){c->dilation, c->dilation});
   uint32_t acc = constant(m, 1), zero = constant(m, 0), bound = m->next++;
   uint32_t resize_scale = 0, resize_offset = 0, resize_border = 0;
   if (c->resize) {
      resize_scale = vector(m, 4, (int64_t[]){4, 2, 4, 2});
      uint32_t length = constant(m, 2), array = m->next++, negative = m->next++;
      emit(m, SpvOpTypeArray, 3, array, m->i32, length);
      emit(m, SpvOpConstant, 3, m->i32, negative, UINT32_MAX);
      resize_offset = m->next++;
      emit(m, SpvOpConstantComposite, 4, array, resize_offset, negative, negative);
      resize_border = vector(m, 2, (int64_t[]){1, 1});
   }
   emit(m, SpvOpConstantFalse, 2, m->boolean, bound);
   uint32_t concat_axis = 0, concat_shape = 0;
   if (c->concat) {
      concat_axis = constant(m, 3);
      concat_shape = vector(m, 4, (int64_t[]){c->batch, c->oh, c->ow, 2 * c->co});
   }
   uint32_t gt = m->next++, graph = m->next++, value = m->next++, result = m->next++;
   emit(m, SpvOpTypeGraphARM, 4, gt, 1u, input, graph_output);
   emit(m, SpvOpGraphEntryPointARM, 5, graph, 0x6e69616du, 0u, variables[0], variables[1]);
   emit(m, SpvOpGraphARM, 2, gt, graph);
   emit(m, SpvOpGraphInputARM, 3, input, value, zero);
   if (c->resize) {
      uint32_t scaled = m->next++;
      emit(m, SpvOpExtInst, 9, resized, scaled, tosa, 63u, acc, value,
           resize_scale, resize_offset, resize_border);
      value = scaled;
   }
   uint32_t branches[2];
   for (unsigned branch = 0; branch < (c->concat ? 2u : 1u); branch++) {
      if (branch)
         result = m->next++;
      emit(m, SpvOpExtInst, 14, output, result, tosa, 2u, pad, stride, dilation,
           acc, bound, branch && c->chain ? branches[0] : value, weights, biases,
           branch && !c->chain ? wzp : izp, wzp);
      if (c->rescale) {
         uint32_t scaled = m->next++;
         emit(m, SpvOpExtInst, 14, final_output, scaled, tosa, 65u,
              scale32, rounding, per_channel, bound, bound,
              result, multiplier, shift, input_zero, output_zero);
         result = scaled;
      }
      if (c->lookup) {
         uint32_t mapped = m->next++;
         emit(m, SpvOpExtInst, 6, final_output, mapped, tosa, 30u, result, table);
         result = mapped;
      }
      branches[branch] = result;
   }
   if (c->concat) {
      uint32_t joined = m->next++;
      emit(m, SpvOpExtInst, 7, graph_output, joined, tosa, 54u,
           concat_axis, branches[0], branches[1]);
      result = m->next++;
      emit(m, SpvOpExtInst, 6, graph_output, result, tosa, 56u, joined, concat_shape);
   }
   emit(m, SpvOpGraphSetOutputARM, 2, result, zero);
   emit(m, SpvOpGraphEndARM, 0);
   m->words[3] = m->next;
}

static struct conv_case
make_case(unsigned id)
{
   struct conv_case c = {.id = id, .height = 7, .width = 9};
   c.rescale = getenv("CONV_RESCALE") != NULL;
   c.lookup = c.rescale && getenv("CONV_TABLE") != NULL;
   c.resize = getenv("CONV_RESIZE") != NULL;
   c.concat = c.rescale && getenv("CONV_CONCAT") != NULL;
   c.per_channel = id % 2;
   c.double_round = id % 3 == 0;
   c.ozp = (int8_t[]){-128, -54, -2, 17, 127}[id % 5];
   unsigned code = id;
   c.ci = (code % 2) ? 12 : 3; code /= 2;
   c.co = (unsigned[]){4, 5, 16}[code % 3]; code /= 3;
   c.wzp = (code % 2) ? -13 : 0; code /= 2;
   c.izp = (code % 2) ? 17 : -128; code /= 2;
   c.pad = code % 2; code /= 2;
   c.dilation = code % 2 + 1; code /= 2;
   c.stride = code % 2 + 1; code /= 2;
   c.batch = code % 2 + 1;
   for (unsigned i = 0; i < 4; i++)
      c.padding[i] = c.pad;
   if (id >= 384) {
      const unsigned shapes[14][3] = {
         {12, 32, 2}, {32, 32, 1}, {32, 32, 2}, {32, 32, 1},
         {32, 64, 2}, {64, 64, 1}, {64, 32, 1}, {32, 16, 1},
         {48, 32, 1}, {32, 16, 1}, {32, 16, 1}, {48, 16, 1},
         {16, 16, 1}, {16, 4, 1},
      };
      const unsigned *s = shapes[(id - 384) % 14];
      c.ci = s[0];
      c.co = s[1];
      c.stride = s[2];
      c.batch = c.dilation = c.pad = 1;
      c.wzp = 0;
      c.izp = -128;
      c.height = 17 * c.stride;
      c.width = 37 * c.stride;
      c.padding[0] = c.padding[2] = 1;
      c.padding[1] = c.padding[3] = c.stride == 1;
      if (id >= 398) {
         c.ci = (unsigned[]){17, 31, 33, 65}[(id - 398) % 4];
         c.co = 32;
         c.batch = (id - 398) / 4 + 1;
         c.stride = 1;
         c.height = 17;
         c.width = 37;
         c.izp = id % 2 ? 17 : -128;
         for (unsigned i = 0; i < 4; i++)
            c.padding[i] = 1;
      }
   }
   unsigned factor = c.resize ? 2 : 1;
   if (c.resize) {
      c.padding[1] += (c.stride - (c.height * factor + c.padding[0] + c.padding[1] -
                                 (2 * c.dilation + 1)) % c.stride) % c.stride;
      c.padding[3] += (c.stride - (c.width * factor + c.padding[2] + c.padding[3] -
                                 (2 * c.dilation + 1)) % c.stride) % c.stride;
   }
   c.oh = (c.height * factor + c.padding[0] + c.padding[1] - (2 * c.dilation + 1)) / c.stride + 1;
   c.ow = (c.width * factor + c.padding[2] + c.padding[3] - (2 * c.dilation + 1)) / c.stride + 1;
   c.input_size = c.batch * c.height * c.width * c.ci;
   c.weight_size = c.co * 3 * 3 * c.ci;
   c.chain = c.concat && !c.resize && c.ci == c.co && c.stride == 1 &&
             c.dilation == 1 && c.oh == c.height && c.ow == c.width;
   unsigned output_channels = c.co * (c.concat ? 2 : 1);
   c.elements = c.batch * c.oh * c.ow * output_channels;
   c.input = malloc(c.input_size);
   c.weights = malloc(c.weight_size);
   c.bias = malloc(c.co * 4);
   c.expected = malloc(c.elements * 4);
   c.multiplier = malloc(c.co * 4);
   c.shift = malloc(c.co);
   assert(c.input && c.weights && c.bias && c.expected && c.multiplier && c.shift);
   for (unsigned i = 0; i < c.input_size; i++)
      c.input[i] = (uint8_t)(i * 67 + 91 + id * 23);
   for (unsigned i = 0; i < c.weight_size; i++)
      c.weights[i] = (uint8_t)(i * 29 + 193 + id * 19);
   for (unsigned i = 0; i < c.co; i++) {
      c.bias[i] = (int32_t)((i * 7919 + id * 89) % 20001) - 10000;
      c.multiplier[i] = 1073741824 + ((i * 7919 + id * 89) & 0x3fffffff);
      c.shift[i] = 30 + (i + id) % 20;
   }
   for (unsigned i = 0; i < 256; i++)
      c.table[i] = (int8_t)((i * 73 + id * 19) ^ 0xa5);
   for (unsigned branch = 0; branch < (c.concat ? 2u : 1u); branch++)
   for (unsigned n = 0; n < c.batch; n++)
      for (unsigned y = 0; y < c.oh; y++)
         for (unsigned x = 0; x < c.ow; x++)
            for (unsigned oc = 0; oc < c.co; oc++) {
               int64_t sum = c.bias[oc];
               for (unsigned ky = 0; ky < 3; ky++)
                  for (unsigned kx = 0; kx < 3; kx++)
                     for (unsigned ic = 0; ic < c.ci; ic++) {
                        int iy = y * c.stride + ky * c.dilation - c.padding[0];
                        int ix = x * c.stride + kx * c.dilation - c.padding[2];
                        if (iy >= 0 && iy < (int)(c.height * factor) && ix >= 0 && ix < (int)(c.width * factor)) {
                           int value = branch && c.chain ?
                              c.expected[((n * c.oh + iy) * c.ow + ix) * output_channels + ic] :
                              c.input[((n * c.height + iy / factor) * c.width + ix / factor) * c.ci + ic];
                           int zero = branch && !c.chain ? c.wzp : c.izp;
                           sum += (int64_t)(value - zero) *
                                  (c.weights[((oc * 3 + ky) * 3 + kx) * c.ci + ic] - c.wzp);
                        }
                     }
               assert(sum >= INT32_MIN && sum <= INT32_MAX);
               if (c.rescale) {
                  unsigned channel = c.per_channel ? oc : 0;
                  int64_t round = INT64_C(1) << (c.shift[channel] - 1);
                  if (c.double_round && c.shift[channel] > 31)
                     round += sum >= 0 ? INT64_C(1) << 30 : -(INT64_C(1) << 30);
                  sum = ((sum * c.multiplier[channel] + round) >> c.shift[channel]) + c.ozp;
                  sum = sum < -128 ? -128 : sum > 127 ? 127 : sum;
               }
               if (c.lookup)
                  sum = c.table[sum + 128];
               c.expected[((n * c.oh + y) * c.ow + x) * output_channels + branch * c.co + oc] = sum;
            }
   return c;
}

static void
free_case(struct conv_case *c)
{
   free(c->input);
   free(c->weights);
   free(c->bias);
   free(c->expected);
   free(c->multiplier);
   free(c->shift);
}

static void
write_file(const char *dir, unsigned id, const char *suffix, const void *data, size_t size)
{
   char path[1024];
   snprintf(path, sizeof(path), "%s/conv-%03u.%s", dir, id, suffix);
   FILE *f = fopen(path, "wb");
   if (!f || fwrite(data, 1, size, f) != size || fclose(f)) {
      fprintf(stderr, "write failed: %s\n", path);
      exit(1);
   }
}

static void
dump_case(const char *dir, const struct conv_case *c, const struct graph_module *m)
{
   write_file(dir, c->id, "spv", m->words, m->count * 4);
   write_file(dir, c->id, "input.bin", c->input, c->input_size);
   write_file(dir, c->id, "weights.bin", c->weights, c->weight_size);
   write_file(dir, c->id, "bias.bin", c->bias, c->co * 4);
   write_file(dir, c->id, "expected.bin", c->expected, c->elements * 4);
   if (c->rescale) {
      write_file(dir, c->id, "multiplier.bin", c->multiplier, (c->per_channel ? c->co : 1) * 4);
      write_file(dir, c->id, "shift.bin", c->shift, c->per_channel ? c->co : 1);
   }
   if (c->lookup)
      write_file(dir, c->id, "table.bin", c->table, sizeof(c->table));
   char json[1024];
   int n = snprintf(json, sizeof(json),
      "{\"id\":%u,\"input_shape\":[%u,%u,%u,%u],\"weight_shape\":[%u,3,3,%u],"
      "\"output_shape\":[%u,%u,%u,%u],\"pad\":[%u,%u,%u,%u],\"stride\":[%u,%u],"
      "\"dilation\":[%u,%u],\"input_zero_point\":%d,\"weight_zero_point\":%d,"
      "\"input_format\":\"R8_SINT\",\"output_format\":\"%s\","
      "\"rescale\":%u,\"per_channel\":%u,\"double_round\":%u,\"output_zero_point\":%d,"
      "\"lookup\":%u,\"resize_2x\":%u,\"concat\":%u,\"chain\":%u,"
      "\"constant_ids\":{\"weights\":1,\"bias\":2,\"input_zero_point\":3,\"weight_zero_point\":4}}\n",
      c->id, c->batch, c->height, c->width, c->ci, c->co, c->ci,
      c->batch, c->oh, c->ow, c->co * (c->concat ? 2 : 1),
      c->padding[0], c->padding[1], c->padding[2], c->padding[3],
      c->stride, c->stride, c->dilation, c->dilation, c->izp, c->wzp,
      c->rescale ? "R8_SINT" : "R32_SINT", c->rescale, c->per_channel, c->double_round, c->ozp,
      c->lookup, c->resize, c->concat, c->chain);
   assert(n > 0 && (size_t)n < sizeof(json));
   write_file(dir, c->id, "json", json, n);
}

static unsigned
run_case(const struct conv_case *c, const struct graph_module *m, unsigned repeats)
{
   printf("CASE id=%u batch=%u stride=%u dilation=%u pad=%u ci=%u co=%u izp=%d wzp=%d output=%ux%u\n",
          c->id, c->batch, c->stride, c->dilation, c->pad, c->ci, c->co,
          c->izp, c->wzp, c->oh, c->ow);
   int64_t dims[11][4] = {{c->batch, c->height, c->width, c->ci},
                         {c->batch, c->oh, c->ow, c->co * (c->concat ? 2 : 1)},
                         {c->co, 3, 3, c->ci}, {c->co}, {1}, {1},
                         {c->per_channel ? c->co : 1}, {c->per_channel ? c->co : 1}, {1}, {1}, {256}};
   VkFormat formats[11] = {VK_FORMAT_R8_SINT, c->rescale ? VK_FORMAT_R8_SINT : VK_FORMAT_R32_SINT,
                          VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8_SINT, VK_FORMAT_R8_SINT,
                          VK_FORMAT_R32_SINT, VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8_SINT,
                          VK_FORMAT_R8_SINT};
   struct tensor tensors[2];
   VkTensorDescriptionARM desc[11];
   for (unsigned i = 0; i < 11; i++)
      desc[i] = (VkTensorDescriptionARM){
         .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
         .tiling = VK_TENSOR_TILING_LINEAR_ARM,
         .format = formats[i],
         .dimensionCount = i < 3 ? 4 : 1,
         .pDimensions = dims[i],
         .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM};
   for (unsigned i = 0; i < 2; i++)
      tensors[i] = make_tensor(formats[i], 4, dims[i], NULL);
   memcpy(tensors[0].map, c->input, c->input_size);
   VkDataGraphPipelineResourceInfoARM resources[2];
   VkDescriptorSetLayoutBinding bindings[2];
   for (unsigned i = 0; i < 2; i++) {
      resources[i] = (VkDataGraphPipelineResourceInfoARM){
         .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
         .pNext = &desc[i], .binding = i};
      bindings[i] = (VkDescriptorSetLayoutBinding){
         .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL};
   }
   VkDescriptorSetLayoutCreateInfo li = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = bindings};
   VkDescriptorSetLayout sl;
   CHECK(CreateDescriptorSetLayout(device, &li, NULL, &sl));
   VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &sl};
   VkPipelineLayout layout;
   CHECK(CreatePipelineLayout(device, &pli, NULL, &layout));
   VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_TENSOR_ARM, 2};
   VkDescriptorPoolCreateInfo pi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
   VkDescriptorPool pool;
   CHECK(CreateDescriptorPool(device, &pi, NULL, &pool));
   VkDescriptorSetAllocateInfo sai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &sl};
   VkDescriptorSet set;
   CHECK(AllocateDescriptorSets(device, &sai, &set));
   for (unsigned i = 0; i < 2; i++) {
      VkWriteDescriptorSetTensorARM tw = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_TENSOR_ARM,
         .tensorViewCount = 1, .pTensorViews = &tensors[i].view};
      VkWriteDescriptorSet w = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .pNext = &tw, .dstSet = set, .dstBinding = i,
         .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM};
      UpdateDescriptorSets(device, 1, &w, 0, NULL);
   }
   VkShaderModuleCreateInfo mi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = m->count * 4, .pCode = m->words};
   VkShaderModule module;
   CHECK(CreateShaderModule(device, &mi, NULL, &module));
   const int32_t input_zero = 0;
   const void *data[9] = {c->weights, c->bias, &c->izp, &c->wzp,
                         c->multiplier, c->shift, &input_zero, &c->ozp, c->table};
   VkDataGraphPipelineConstantARM constants[9];
   unsigned constant_count = c->rescale ? 8 + c->lookup : 4;
   for (unsigned i = 0; i < constant_count; i++)
      constants[i] = (VkDataGraphPipelineConstantARM){
         .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM,
         .pNext = &desc[i + 2], .id = i + 1, .pConstantData = data[i]};
   VkDataGraphPipelineShaderModuleCreateInfoARM shader = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
      .module = module, .pName = "main", .constantCount = constant_count, .pConstants = constants};
   VkDataGraphPipelineCreateInfoARM ci = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
      .pNext = &shader, .layout = layout,
      .resourceInfoCount = 2, .pResourceInfos = resources};
   VkPipeline pipeline;
   CHECK(CreateDataGraphPipelinesARM(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &ci, NULL, &pipeline));
   VkDataGraphPipelineSessionCreateInfoARM sci = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM,
      .dataGraphPipeline = pipeline};
   VkDataGraphPipelineSessionARM session;
   CHECK(CreateDataGraphPipelineSessionARM(device, &sci, NULL, &session));
   VkDataGraphPipelineSessionBindPointRequirementsInfoARM sri = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENTS_INFO_ARM,
      .session = session};
   uint32_t n = 0;
   CHECK(GetDataGraphPipelineSessionBindPointRequirementsARM(device, &sri, &n, NULL));
   VkDataGraphPipelineSessionBindPointRequirementARM *reqs = calloc(n, sizeof(*reqs));
   for (unsigned i = 0; i < n; i++)
      reqs[i].sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM;
   CHECK(GetDataGraphPipelineSessionBindPointRequirementsARM(device, &sri, &n, reqs));
   VkDeviceMemory *allocations = NULL;
   unsigned allocations_count = 0;
   for (unsigned i = 0; i < n; i++)
      for (unsigned j = 0; j < reqs[i].numObjects; j++) {
         VkDataGraphPipelineSessionMemoryRequirementsInfoARM ri = {
            .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_MEMORY_REQUIREMENTS_INFO_ARM,
            .session = session, .bindPoint = reqs[i].bindPoint, .objectIndex = j};
         VkMemoryRequirements2 mr = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
         GetDataGraphPipelineSessionMemoryRequirementsARM(device, &ri, &mr);
         uint32_t mt = 0;
         while (!(mr.memoryRequirements.memoryTypeBits & (1u << mt)))
            mt++;
         VkMemoryAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mr.memoryRequirements.size, .memoryTypeIndex = mt};
         VkDeviceMemory mem;
         CHECK(AllocateMemory(device, &ai, NULL, &mem));
         allocations = realloc(allocations, sizeof(*allocations) * (allocations_count + 1));
         assert(allocations);
         allocations[allocations_count++] = mem;
         VkBindDataGraphPipelineSessionMemoryInfoARM bi = {
            .sType = VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM,
            .session = session, .bindPoint = reqs[i].bindPoint, .objectIndex = j, .memory = mem};
         CHECK(BindDataGraphPipelineSessionMemoryARM(device, 1, &bi));
      }
   free(reqs);
   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family};
   VkCommandPool cp;
   CHECK(CreateCommandPool(device, &cpi, NULL, &cp));
   VkCommandBufferAllocateInfo cai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(AllocateCommandBuffers(device, &cai, &cmd));
   VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(BeginCommandBuffer(cmd, &begin));
   VkMemoryBarrier2 upload = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
      .dstAccessMask = VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM | VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM};
   VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &upload};
   CmdPipelineBarrier2(cmd, &dep);
   CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, pipeline);
   CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, layout, 0, 1, &set, 0, NULL);
   VkDataGraphPipelineDispatchInfoARM dispatch = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_DISPATCH_INFO_ARM};
   CmdDispatchDataGraphARM(cmd, session, &dispatch);
   VkMemoryBarrier2 download = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
      .srcAccessMask = VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM,
      .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
   dep.pMemoryBarriers = &download;
   CmdPipelineBarrier2(cmd, &dep);
   CHECK(EndCommandBuffer(cmd));
   VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
   unsigned failures = 0;
   for (unsigned repeat = 0; repeat < repeats; repeat++) {
      memset(tensors[1].map, 0xcd, tensors[1].size);
      memset(tensors[1].map, repeat & 1 ? 0x97 : 0xcd,
             c->elements * (c->rescale ? 1 : 4));
      CHECK(QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
      CHECK(QueueWaitIdle(queue));
      for (unsigned i = 0; i < c->elements; i++) {
         int32_t actual = c->rescale ? ((int8_t *)tensors[1].map)[i] :
                                     ((int32_t *)tensors[1].map)[i];
         if (actual != c->expected[i]) {
            if (failures < 8 || (repeat && i < 2))
               printf("MISMATCH id=%u repeat=%u index=%u actual=%d expected=%d\n",
                      c->id, repeat, i, actual, c->expected[i]);
            failures++;
         }
      }
      for (unsigned t = 0; t < 2; t++)
         for (uint64_t i = tensors[t].size - 64; i < tensors[t].size; i++)
            if (((uint8_t *)tensors[t].map)[i] != 0xcd) {
               if (failures < 8)
                  printf("GUARD id=%u repeat=%u tensor=%u offset=%llu\n",
                         c->id, repeat, t, (unsigned long long)i);
               failures++;
            }
      for (unsigned i = 0; i < c->input_size; i++)
         if (((int8_t *)tensors[0].map)[i] != c->input[i])
            failures++;
   }
   printf("CONV id=%u checks=%u repeats=%u guard=128 failures=%u\n",
          c->id, c->elements * repeats, repeats, failures);
   DestroyCommandPool(device, cp, NULL);
   DestroyDataGraphPipelineSessionARM(device, session, NULL);
   for (unsigned i = 0; i < allocations_count; i++)
      FreeMemory(device, allocations[i], NULL);
   free(allocations);
   DestroyPipeline(device, pipeline, NULL);
   DestroyShaderModule(device, module, NULL);
   DestroyDescriptorPool(device, pool, NULL);
   DestroyPipelineLayout(device, layout, NULL);
   DestroyDescriptorSetLayout(device, sl, NULL);
   for (unsigned i = 0; i < 2; i++)
      destroy_tensor(&tensors[i]);
   return failures;
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s driver [first [count [repeats]]]\n       %s --dump directory [first [count]]\n", argv[0], argv[0]);
      return 2;
   }
   setvbuf(stdout, NULL, _IONBF, 0);
   int dump = !strcmp(argv[1], "--dump");
   int arg = dump ? 3 : 2;
   if (dump && argc < 3)
      return 2;
   unsigned first = argc > arg ? strtoul(argv[arg], NULL, 10) : 0;
   unsigned count = argc > arg + 1 ? strtoul(argv[arg + 1], NULL, 10) : 406 - first;
   unsigned repeats = argc > arg + 2 ? strtoul(argv[arg + 2], NULL, 10) : 1;
   if (first >= 406 || !count || count > 406 - first || !repeats || repeats > 1000)
      return 2;
   if (!dump) {
      init(argv[1], 0);
      tensor_extra_usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM;
      LOAD(CreateDataGraphPipelinesARM);
      LOAD(CreateDataGraphPipelineSessionARM);
      LOAD(DestroyDataGraphPipelineSessionARM);
      LOAD(GetDataGraphPipelineSessionBindPointRequirementsARM);
      LOAD(GetDataGraphPipelineSessionMemoryRequirementsARM);
      LOAD(BindDataGraphPipelineSessionMemoryARM);
      LOAD(CmdDispatchDataGraphARM);
      LOAD(CmdPipelineBarrier2);
   }
   unsigned failures = 0;
   uint64_t checks = 0;
   for (unsigned id = first; id < first + count; id++) {
      struct conv_case c = make_case(id);
      struct graph_module m;
      build_module(&c, &m);
      if (dump)
         dump_case(argv[2], &c, &m);
      else
         failures += run_case(&c, &m, repeats);
      checks += c.elements * repeats;
      free_case(&c);
   }
   if (!dump)
      DestroyDevice(device, NULL);
   printf("SUMMARY cases=%u first=%u checks=%llu failures=%u dumped=%d\n",
          count, first, (unsigned long long)checks, failures, dump);
   return failures ? 1 : 0;
}
