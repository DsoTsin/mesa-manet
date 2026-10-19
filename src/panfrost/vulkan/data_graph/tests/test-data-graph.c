#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/spirv/spirv.h"
#include "util/ralloc.h"
#include "nir.h"
#include "panvk_data_graph.h"
#include "panvk_data_graph_neural.h"
#include "vk_format.h"

#include "data-graph-fixtures.h"
#include "data-graph-broadcast-fixtures.h"
#include "data-graph-nir-interpreter.h"

static const struct {
   const uint32_t *code;
   size_t size;
   unsigned rank;
   int64_t dims[3][4];
   unsigned passes;
   uint64_t transient;
   VkFormat formats[3];
   unsigned ranks[3];
   bool reject;
} cases[] = {
   {graph_add, sizeof(graph_add), 2, {{3, 19}, {1, 19}, {3, 19}}, 1, 0},
   {graph_chain, sizeof(graph_chain), 2, {{3, 19}, {1, 19}, {3, 19}}, 1, 0},
   {graph_matmul,
    sizeof(graph_matmul),
    3,
    {{2, 3, 5}, {2, 5, 4}, {2, 3, 4}},
    1,
    0},
   {graph_fanout, sizeof(graph_fanout), 2, {{3, 19}, {1, 19}, {3, 19}}, 1, 0},
   {graph_mul, sizeof(graph_mul), 2, {{3, 19}, {1, 19}, {3, 19}}, 1, 0},
   {graph_matmul_i8,
    sizeof(graph_matmul_i8),
    3,
    {{2, 3, 5}, {2, 5, 4}, {2, 3, 4}},
    1,
    0,
    {VK_FORMAT_R8_SINT, VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT}},
   {graph_matmul_i8_add,
    sizeof(graph_matmul_i8_add),
    3,
    {{2, 3, 5}, {2, 5, 4}, {2, 3, 4}},
    2,
    128,
    {VK_FORMAT_R8_SINT, VK_FORMAT_R8_SINT, VK_FORMAT_R32_SINT}},
   {.code = graph_broadcast_0,
    .size = sizeof(graph_broadcast_0),
    .rank = 3,
    .dims = {{2, 3, 5}, {5}, {2, 3, 5}},
    .passes = 1,
    .ranks = {3, 1, 3},
    .reject = true},
   {.code = graph_broadcast_1,
    .size = sizeof(graph_broadcast_1),
    .rank = 3,
    .dims = {{2, 3, 5}, {1, 5}, {2, 3, 5}},
    .passes = 1,
    .ranks = {3, 2, 3},
    .reject = true},
   {.code = graph_broadcast_2,
    .size = sizeof(graph_broadcast_2),
    .rank = 4,
    .dims = {{2, 7, 3, 5}, {1, 3, 1}, {2, 7, 3, 5}},
    .passes = 1,
    .ranks = {4, 3, 4},
    .reject = true},
   {.code = graph_broadcast_3,
    .size = sizeof(graph_broadcast_3),
    .rank = 4,
    .dims = {{2, 3, 5, 7}, {1}, {2, 3, 5, 7}},
    .passes = 1,
    .ranks = {4, 1, 4},
    .reject = true},
   {.code = graph_broadcast_valid_0,
    .size = sizeof(graph_broadcast_valid_0),
    .rank = 3,
    .dims = {{2, 3, 5}, {1, 1, 5}, {2, 3, 5}},
    .passes = 1},
   {.code = graph_broadcast_valid_1,
    .size = sizeof(graph_broadcast_valid_1),
    .rank = 3,
    .dims = {{2, 3, 5}, {1, 1, 5}, {2, 3, 5}},
    .passes = 1},
   {.code = graph_broadcast_valid_2,
    .size = sizeof(graph_broadcast_valid_2),
    .rank = 4,
    .dims = {{2, 7, 3, 5}, {1, 1, 3, 1}, {2, 7, 3, 5}},
    .passes = 1},
   {.code = graph_broadcast_valid_3,
    .size = sizeof(graph_broadcast_valid_3),
    .rank = 4,
    .dims = {{2, 3, 5, 7}, {1, 1, 1, 1}, {2, 3, 5, 7}},
    .passes = 1},
};

static void
check_plan(const struct panvk_data_graph *g)
{
   for (unsigned i = 0; i < g->pass_count; i++) {
      const struct panvk_graph_value *v = &g->values[g->passes[i]];
      for (unsigned j = 0; j < v->input_count; j++) {
         assert(v->inputs[j] < g->passes[i]);
         assert(g->values[v->inputs[j]].last_use >= i);
      }
      if (v->kind != PANVK_GRAPH_OPERATION)
         continue;
      assert(v->offset % 64 == 0 && v->offset + v->size <= g->transient_size);
      for (unsigned j = 0; j < i; j++) {
         const struct panvk_graph_value *a = &g->values[g->passes[j]];
         if (a->kind == PANVK_GRAPH_OPERATION && a->last_use >= i)
            assert(v->offset >= a->offset + a->size ||
                   a->offset >= v->offset + v->size);
      }
   }
}

static unsigned
check_broadcast(const struct panvk_data_graph *g,
                const VkTensorDescriptionARM *desc)
{
   unsigned count = 1, small_count = 1;
   for (unsigned d = 0; d < desc[0].dimensionCount; d++)
      count *= desc[0].pDimensions[d];
   for (unsigned d = 0; d < desc[1].dimensionCount; d++)
      small_count *= desc[1].pDimensions[d];
   float *a = calloc(count, sizeof(float));
   float *b = calloc(small_count, sizeof(float));
   float *output = calloc(count, sizeof(float));
   uint8_t **buffers = calloc(g->count, sizeof(*buffers));
   assert(a && b && output && buffers && g->pass_count == 1 &&
          !g->transient_size);
   for (unsigned i = 0; i < g->count; i++) {
      const struct panvk_graph_value *v = &g->values[i];
      if (v->kind == PANVK_GRAPH_INPUT)
         buffers[i] = (uint8_t *)(v->resource_index ? b : a);
      else if (v->kind == PANVK_GRAPH_CONSTANT)
         buffers[i] = v->data;
      else if (v->kind == PANVK_GRAPH_OUTPUT)
         buffers[i] = (uint8_t *)output;
   }
   nir_shader *nir = panvk_data_graph_shader(g, 0);
   assert(nir);
   unsigned checked = 0;
   for (unsigned mode = 0; mode < 3; mode++) {
      const uint32_t special[] = {0,          0x80000000, 0x7f800000, 0xff800000,
                                  0x7fc12345, 0xffc54321, 1,          0x80000001};
      for (unsigned i = 0; i < count; i++) {
         if (mode)
            memcpy(&a[i], &special[i % ARRAY_SIZE(special)], sizeof(float));
         else
            a[i] = ((int)(i * 13 % 31) - 15) / 16.0f;
      }
      for (unsigned i = 0; i < small_count; i++) {
         if (mode)
            memcpy(&b[i], &special[(i + mode) % ARRAY_SIZE(special)],
                   sizeof(float));
         else
            b[i] = (i + 1) / 8.0f;
      }
      struct graph_interpreter e;
      graph_interpreter_init(&e, g, nir, buffers);
      for (unsigned i = 0; i < count; i++) {
         assert(graph_interpret(&e, i) == 1);
         unsigned source = 0, source_stride = 1, target_stride = 1;
         for (unsigned d = desc[0].dimensionCount; d-- > 0;) {
            unsigned coordinate = i / target_stride % desc[0].pDimensions[d];
            int sd = d + desc[1].dimensionCount - desc[0].dimensionCount;
            if (sd >= 0) {
               if (desc[1].pDimensions[sd] != 1)
                  source += coordinate * source_stride;
               source_stride *= desc[1].pDimensions[sd];
            }
            target_stride *= desc[0].pDimensions[d];
         }
         assert(source < small_count);
         float reciprocal = 1.0f / b[source];
         float expected = a[i] * reciprocal;
         assert(isnan(expected) ? isnan(output[i])
                                : !memcmp(&expected, &output[i], sizeof(float)));
         checked++;
      }
      assert(!graph_interpret(&e, count));
      assert(!graph_interpret(&e, ALIGN_POT(count, 64) + 63));
      free(e.values);
   }
   ralloc_free(nir);
   free(buffers);
   free(output);
   free(b);
   free(a);
   return checked;
}

static unsigned
check_fanout(const struct panvk_data_graph *g)
{
   float a[57], b[19], output[57];
   uint8_t **buffers = calloc(g->count, sizeof(*buffers));
   assert(buffers && g->pass_count == 1 && !g->transient_size);
   for (unsigned i = 0; i < g->count; i++) {
      const struct panvk_graph_value *v = &g->values[i];
      if (v->kind == PANVK_GRAPH_INPUT)
         buffers[i] = (uint8_t *)(v->resource_index ? b : a);
      else if (v->kind == PANVK_GRAPH_CONSTANT)
         buffers[i] = v->data;
      else if (v->kind == PANVK_GRAPH_OUTPUT)
         buffers[i] = (uint8_t *)output;
   }
   nir_shader *nir = panvk_data_graph_shader(g, 0);
   assert(nir);
   unsigned checked = 0;
   for (unsigned mode = 0; mode < 3; mode++) {
      const uint32_t special[] = {0,          0x80000000, 0x7f800000, 0xff800000,
                                  0x7fc12345, 0xffc54321, 1,          0x80000001};
      for (unsigned i = 0; i < ARRAY_SIZE(a); i++) {
         if (mode)
            memcpy(&a[i], &special[i % ARRAY_SIZE(special)], sizeof(float));
         else
            a[i] = ((int)(i * 13 % 31) - 15) / 16.0f;
      }
      for (unsigned i = 0; i < ARRAY_SIZE(b); i++) {
         if (mode)
            memcpy(&b[i], &special[(i + mode) % ARRAY_SIZE(special)],
                   sizeof(float));
         else
            b[i] = ((int)(i * 7 % 13) - 6) / 8.0f;
      }
      struct graph_interpreter e;
      graph_interpreter_init(&e, g, nir, buffers);
      for (unsigned i = 0; i < ARRAY_SIZE(output); i++) {
         assert(graph_interpret(&e, i) == 1);
         float t = a[i] + b[i % ARRAY_SIZE(b)];
         float l = t + a[i];
         float r = t - b[i % ARRAY_SIZE(b)];
         float expected = l + r;
         assert(isnan(expected) ? isnan(output[i])
                                : !memcmp(&expected, &output[i], sizeof(float)));
         checked++;
      }
      assert(!graph_interpret(&e, ARRAY_SIZE(output)));
      free(e.values);
   }
   ralloc_free(nir);
   free(buffers);
   return checked;
}

int
main(void)
{
   unsigned validated = 0, rejected = 0, fanout = 0, broadcast = 0;
   for (unsigned c = 0; c < ARRAY_SIZE(cases); c++) {
      VkTensorDescriptionARM desc[3];
      VkDataGraphPipelineResourceInfoARM resources[3];
      for (unsigned i = 0; i < 3; i++) {
         desc[i] = (VkTensorDescriptionARM){
            .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
            .format =
               cases[c].formats[i] ? cases[c].formats[i] : VK_FORMAT_R32_SFLOAT,
            .tiling = VK_TENSOR_TILING_LINEAR_ARM,
            .dimensionCount =
               cases[c].ranks[i] ? cases[c].ranks[i] : cases[c].rank,
            .pDimensions = cases[c].dims[i],
            .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
         };
         resources[i] = (VkDataGraphPipelineResourceInfoARM){
            .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
            .pNext = &desc[i],
            .binding = i,
         };
      }
      const VkDataGraphPipelineCreateInfoARM info = {
         .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
         .resourceInfoCount = 3,
         .pResourceInfos = resources,
      };
      const VkDataGraphPipelineShaderModuleCreateInfoARM module = {
         .sType =
            VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
         .pName = "main",
      };
      struct panvk_data_graph *g;
      VkResult result = panvk_data_graph_parse(
         NULL, cases[c].code, cases[c].size, &info, &module, &g);
      if (cases[c].reject) {
         assert(result == VK_ERROR_FEATURE_NOT_PRESENT && !g);
         rejected++;
         const enum panvk_graph_op variants[] = {PANVK_GRAPH_OP_ADD,
                                                 PANVK_GRAPH_OP_SUB};
         for (unsigned v = 0; v < ARRAY_SIZE(variants); v++) {
            uint32_t *code = malloc(cases[c].size);
            assert(code);
            memcpy(code, cases[c].code, cases[c].size);
            bool changed = false;
            for (unsigned i = 5; i < cases[c].size / 4;
                 i += code[i] >> 16) {
               if ((code[i] & 0xffff) == SpvOpExtInst &&
                   code[i + 4] == PANVK_GRAPH_OP_MUL) {
                  assert(code[i] >> 16 == 8);
                  code[i] -= 1u << 16;
                  code[i + 4] = variants[v];
                  memmove(&code[i + 7], &code[i + 8],
                          cases[c].size - (i + 8) * sizeof(uint32_t));
                  changed = true;
                  break;
               }
            }
            assert(changed);
            assert(panvk_data_graph_parse(NULL, code,
                                          cases[c].size - sizeof(uint32_t),
                                          &info, &module, &g) ==
                      VK_ERROR_FEATURE_NOT_PRESENT &&
                   !g);
            free(code);
            rejected++;
         }
         continue;
      }
      assert(result == VK_SUCCESS);
      if (g->pass_count != cases[c].passes ||
          g->transient_size != cases[c].transient)
         fprintf(stderr, "case=%u passes=%u transient=%llu expected=%u/%llu\n",
                 c, g->pass_count, (unsigned long long)g->transient_size,
                 cases[c].passes, (unsigned long long)cases[c].transient);
      assert(g->pass_count == cases[c].passes &&
             g->transient_size == cases[c].transient);
      check_plan(g);
      if (c == 3)
         fanout += check_fanout(g);
      if (c >= 7)
         broadcast += check_broadcast(g, desc);
      struct panvk_data_graph_matmul_add extracted = {0};
      VkResult extraction = panvk_data_graph_matmul_add_extract(g, &extracted);
      if (cases[c].formats[0] == VK_FORMAT_R8_SINT) {
         assert(extraction == VK_SUCCESS);
         assert(extracted.info.batch == 2 && extracted.info.rows == 3);
         assert(extracted.info.inner == 5 && extracted.info.columns == 4);
         assert(extracted.info.a_zero_point == 3);
         assert(extracted.info.b_zero_point == -2);
         assert(extracted.add_count == (c == 6));
         assert(g->values[extracted.values[0]].matmul);
         if (extracted.add_count) {
            assert(extracted.adds[0].input[0] == 0);
            assert(extracted.adds[0].input[1] == 0);
         }
         assert(g->values[extracted.output].kind == PANVK_GRAPH_OUTPUT);
      } else {
         assert(extraction == VK_ERROR_FEATURE_NOT_PRESENT);
      }
      panvk_data_graph_matmul_add_finish(&extracted);
      if (cases[c].formats[0] == VK_FORMAT_R8_SINT) {
         const struct panvk_graph_value *v = &g->values[g->passes[0]];
         if (v->kind == PANVK_GRAPH_OUTPUT) {
            v = &g->values[v->inputs[0]];
            assert(v->fused_into == g->passes[0]);
         } else {
            assert(g->pass_count == 2 && v->fused_into == UINT32_MAX);
            assert(v->last_use == 1 && v->offset == 0);
         }
         assert(v->op == PANVK_GRAPH_OP_MATMUL &&
                v->format == VK_FORMAT_R32_SINT);
         assert(g->values[v->inputs[0]].elements == 30);
         assert(g->values[v->inputs[0]].size == 30);
         assert(g->values[v->inputs[1]].elements == 40);
         assert(g->values[v->inputs[1]].size == 40);
         assert(*(int8_t *)g->values[v->inputs[2]].data == 3);
         assert(*(int8_t *)g->values[v->inputs[3]].data == -2);
         assert(v->matmul && v->matmul->info.batch == 2);
         assert(v->matmul->info.rows == 3 && v->matmul->info.inner == 5);
         assert(v->matmul->info.columns == 4);
         assert(v->matmul->info.a_zero_point == 3);
         assert(v->matmul->info.b_zero_point == -2);
      }
      for (unsigned i = 0; i < g->pass_count; i++) {
         nir_shader *nir = panvk_data_graph_shader(g, i);
         assert(nir);
         assert(!nir->info.internal);
         nir_validate_shader(nir, "graph unit test");
         nir_foreach_function_impl(impl, nir) {
            nir_foreach_block(block, impl)
            {
               nir_foreach_instr(instr, block) {
                  if (instr->type == nir_instr_type_alu) {
                     nir_alu_instr *alu = nir_instr_as_alu(instr);
                     unsigned required = nir_op_valid_fp_math_ctrl(
                        alu->op, nir_fp_preserve_sz_inf_nan);
                     assert((alu->fp_math_ctrl & required) == required);
                  }
               }
            }
         }
         ralloc_free(nir);
         validated++;
      }
      ralloc_free(g);
      const VkTensorTilingARM unsupported_tilings[] = {
         VK_TENSOR_TILING_BRICK_16_WIDE_ARM,
         VK_TENSOR_TILING_BRICK_8_WIDE_ARM,
         VK_TENSOR_TILING_BRICK_4_WIDE_ARM,
         VK_TENSOR_TILING_BLOCK_U_INTERLEAVED_ARM,
         VK_TENSOR_TILING_BLOCK_U_INTERLEAVED_64K_ARM,
      };
      for (unsigned i = 0; i < ARRAY_SIZE(unsupported_tilings); i++) {
         desc[0].tiling = unsupported_tilings[i];
         assert(panvk_data_graph_parse(NULL, cases[c].code, cases[c].size,
                                       &info, &module, &g) != VK_SUCCESS &&
                !g);
         rejected++;
      }
      desc[0].tiling = VK_TENSOR_TILING_LINEAR_ARM;
      VkFormat output_format = desc[2].format;
      desc[2].format = output_format == VK_FORMAT_R32_SFLOAT
                          ? VK_FORMAT_R32_SINT
                          : VK_FORMAT_R32_SFLOAT;
      assert(panvk_data_graph_parse(NULL, cases[c].code, cases[c].size, &info,
                                    &module, &g) != VK_SUCCESS &&
             !g);
      rejected++;
      desc[2].format = output_format;
      for (size_t bytes = 0; bytes < cases[c].size; bytes++) {
         assert(panvk_data_graph_parse(NULL, cases[c].code, bytes, &info,
                                       &module, &g) != VK_SUCCESS &&
                !g);
         rejected++;
      }
      uint32_t *code = malloc(cases[c].size);
      assert(code);
      memcpy(code, cases[c].code, cases[c].size);
      for (size_t i = 5; i < cases[c].size / 4;) {
         uint32_t old = code[i];
         unsigned words = old >> 16;
         code[i] &= 0xffff;
         assert(panvk_data_graph_parse(NULL, code, cases[c].size, &info,
                                       &module, &g) != VK_SUCCESS &&
                !g);
         code[i] = UINT32_MAX;
         assert(panvk_data_graph_parse(NULL, code, cases[c].size, &info,
                                       &module, &g) != VK_SUCCESS &&
                !g);
         rejected += 2;
         code[i] = old;
         i += words;
      }
      free(code);
   }
   printf("graph shaders=%u rejected=%u fanout=%u broadcast=%u\n", validated,
          rejected, fanout, broadcast);
   return 0;
}
