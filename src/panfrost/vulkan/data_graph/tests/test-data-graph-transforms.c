#include "panvk_data_graph.h"
#include "compiler/spirv/spirv.h"
#include "nir_builder.h"
#include "util/ralloc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data-graph-transform-fixtures.h"

static const struct transform_case {
   const uint32_t *code;
   size_t size;
   VkFormat format;
   unsigned rank;
   int64_t dims[4];
   unsigned permutation[4];
   unsigned output_rank;
   int64_t output_dims[4];
   unsigned passes;
   bool reshape;
   bool constant;
   bool shared;
   bool add;
} cases[] = {
   {transform_transpose1, sizeof(transform_transpose1), VK_FORMAT_R32_SFLOAT,
    1, {7}, {0}, 1, {7}, 1},
   {transform_transpose2, sizeof(transform_transpose2), VK_FORMAT_R32_UINT,
    2, {3, 5}, {1, 0}, 2, {5, 3}, 1},
   {transform_transpose3, sizeof(transform_transpose3), VK_FORMAT_R8_SINT,
    3, {2, 3, 4}, {2, 0, 1}, 3, {4, 2, 3}, 1},
   {transform_transpose4, sizeof(transform_transpose4), VK_FORMAT_R32_SINT,
    4, {2, 3, 2, 5}, {3, 1, 0, 2}, 4, {5, 3, 2, 2}, 1},
   {transform_reshape, sizeof(transform_reshape), VK_FORMAT_R32_SFLOAT,
    3, {2, 3, 4}, {0}, 2, {6, 4}, 1, true},
   {transform_chain, sizeof(transform_chain), VK_FORMAT_R32_SFLOAT,
    3, {2, 3, 4}, {2, 0, 1}, 2, {6, 4}, 2, true},
   {transform_constant_chain, sizeof(transform_constant_chain), VK_FORMAT_R32_SFLOAT,
    3, {2, 3, 4}, {2, 0, 1}, 2, {6, 4}, 1, true, true},
   {transform_shared_constant, sizeof(transform_shared_constant), VK_FORMAT_R8_SINT,
    3, {2, 3, 4}, {2, 0, 1}, 3, {4, 2, 3}, 2, false, true, true},
   {transform_add_transpose, sizeof(transform_add_transpose), VK_FORMAT_R32_SFLOAT,
    2, {3, 3}, {1, 0}, 2, {3, 3}, 2, false, false, false, true},
};

static VkResult
parse(const struct transform_case *c, const uint32_t *code,
      struct panvk_data_graph **graph)
{
   uint32_t bits[24];
   int8_t bytes[24];
   for (unsigned i = 0; i < ARRAY_SIZE(bits); i++) {
      bits[i] = i % 2 ? 0x80000000u : 0x7fc00000u + i;
      bytes[i] = i - 12;
   }
   VkTensorDescriptionARM input = {
      .sType = VK_STRUCTURE_TYPE_TENSOR_DESCRIPTION_ARM,
      .format = c->format,
      .dimensionCount = c->rank,
      .pDimensions = c->dims,
      .tiling = VK_TENSOR_TILING_LINEAR_ARM,
      .usage = VK_TENSOR_USAGE_DATA_GRAPH_BIT_ARM,
   };
   VkTensorDescriptionARM output = input;
   output.dimensionCount = c->output_rank;
   output.pDimensions = c->output_dims;
   VkDataGraphPipelineConstantARM constant = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CONSTANT_ARM,
      .pNext = &input,
      .id = 7,
      .pConstantData = c->format == VK_FORMAT_R8_SINT ? (void *)bytes : (void *)bits,
   };
   VkDataGraphPipelineShaderModuleCreateInfoARM module = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM,
      .pName = "main",
      .constantCount = c->constant,
      .pConstants = &constant,
   };
   VkDataGraphPipelineResourceInfoARM resources[3];
   unsigned count = 0;
   const VkTensorDescriptionARM *descs[] = {&input, &output, &input};
   for (unsigned i = c->constant; i < 2 + c->shared; i++) {
      resources[count] = (VkDataGraphPipelineResourceInfoARM){
         .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_RESOURCE_INFO_ARM,
         .pNext = descs[i],
         .binding = count,
      };
      count++;
   }
   const VkDataGraphPipelineCreateInfoARM info = {
      .sType = VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_CREATE_INFO_ARM,
      .resourceInfoCount = count,
      .pResourceInfos = resources,
   };
   return panvk_data_graph_parse(NULL, code, c->size, &info, &module, graph);
}

static void
coordinates(uint32_t index, unsigned rank, const int64_t *dims, uint32_t *coords)
{
   for (int d = rank - 1; d >= 0; d--) {
      coords[d] = index % dims[d];
      index /= dims[d];
   }
   assert(!index);
}

static unsigned
check_coordinates(const struct transform_case *c, const struct panvk_data_graph *g)
{
   unsigned checked = 0;
   for (unsigned pass = 0; pass < g->pass_count; pass++) {
      const struct panvk_graph_value *v = &g->values[g->passes[pass]];
      for (uint32_t index = 0; index < v->elements; index++) {
         nir_shader *nir = panvk_data_graph_shader(g, pass);
         assert(nir);
         nir_builder b = nir_builder_create(nir_shader_get_entrypoint(nir));
         nir_foreach_block(block, b.impl) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_global_invocation_id) {
                  b.cursor = nir_before_instr(instr);
                  nir_def_rewrite_uses(&intr->def, nir_imm_ivec3(&b, index, 0, 0));
                  nir_instr_remove(instr);
               }
            }
         }
         while (nir_opt_constant_folding(nir)) {}
         uint32_t out[4], expected[4];
         int64_t dims[4];
         for (unsigned d = 0; d < v->rank; d++)
            dims[d] = v->dims[d];
         coordinates(index, v->rank, dims, out);
         unsigned read_rank = v->rank;
         bool transpose = !c->constant &&
                          (!c->reshape || c->passes == 2) &&
                          (!c->add || pass == 1) &&
                          (!c->reshape || pass == 0);
         if (transpose) {
            for (unsigned d = 0; d < c->rank; d++)
               expected[c->permutation[d]] = out[d];
         } else if (!c->constant && c->reshape) {
            read_rank = c->rank;
            for (unsigned d = 0; d < c->rank; d++)
               dims[d] = c->dims[c->passes == 2 ? c->permutation[d] : d];
            coordinates(index, read_rank, dims, expected);
         } else {
            memcpy(expected, out, sizeof(out));
         }
         unsigned reads = 0, writes = 0;
         nir_foreach_block(block, b.impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               bool read = intr->intrinsic == nir_intrinsic_tensor_read_arm;
               bool write = intr->intrinsic == nir_intrinsic_tensor_write_arm;
               if (!read && !write)
                  continue;
               assert(nir_src_is_const(intr->src[1]));
               unsigned rank = read ? read_rank : v->rank;
               assert(intr->src[1].ssa->num_components == rank);
               for (unsigned d = 0; d < rank; d++)
                  assert(nir_src_comp_as_uint(intr->src[1], d) ==
                         (read ? expected[d] : out[d]));
               reads += read;
               writes += write;
               checked++;
            }
         }
         assert(reads && writes == 1);
         nir_validate_shader(nir, "transform coordinates");
         ralloc_free(nir);
      }
   }
   return checked;
}

static void
check_constant(const struct transform_case *c, const struct panvk_data_graph *g)
{
   static const uint8_t order[] = {
      0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21,
      2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23,
   };
   unsigned constants = 0;
   for (unsigned pass = 0; pass < g->pass_count; pass++) {
      const struct panvk_graph_value *out = &g->values[g->passes[pass]];
      assert(out->kind == PANVK_GRAPH_OUTPUT);
      const struct panvk_graph_value *v = &g->values[out->inputs[0]];
      assert(v->kind == PANVK_GRAPH_CONSTANT && v->live && !v->input_count);
      for (unsigned i = 0; i < 24; i++) {
         unsigned source = pass ? i : order[i];
         if (c->format == VK_FORMAT_R8_SINT)
            assert(((int8_t *)v->data)[i] == (int)source - 12);
         else
            assert(((uint32_t *)v->data)[i] ==
                   (source % 2 ? 0x80000000u : 0x7fc00000u + source));
      }
   }
   for (unsigned i = 0; i < g->count; i++)
      constants += g->values[i].live && g->values[i].kind == PANVK_GRAPH_CONSTANT;
   assert(constants == 1 + c->shared && !g->transient_size);
   assert(g->constant_size == 128);
}

static uint32_t *
find_id(uint32_t *code, size_t words, uint32_t id)
{
   for (size_t i = 5; i < words; i += code[i] >> 16) {
      unsigned op = code[i] & 0xffff;
      if ((op == SpvOpConstant || op == SpvOpConstantComposite) && code[i + 2] == id)
         return code + i;
   }
   abort();
}

static unsigned
check_rejection(const struct transform_case *c)
{
   uint32_t *code = malloc(c->size);
   assert(code);
   unsigned rejected = 0;
   for (size_t i = 5; i < c->size / 4; i += c->code[i] >> 16) {
      if ((c->code[i] & 0xffff) != SpvOpExtInst ||
          (c->code[i + 4] != 56 && c->code[i + 4] != 60))
         continue;
      for (unsigned test = 0; test < 5; test++) {
         memcpy(code, c->code, c->size);
         bool transpose = code[i + 4] == 60;
         uint32_t *attr = find_id(code, c->size / 4, code[i + (transpose ? 5 : 6)]);
         uint32_t *value = find_id(code, c->size / 4, attr[3]);
         if (test == 0) {
            value[3] = UINT32_MAX;
         } else if (test == 1) {
            value[3] = transpose ? c->rank : 0;
         } else if (test == 2) {
            if (!transpose || (attr[0] >> 16) < 5)
               continue;
            attr[4] = attr[3];
         } else if (test == 3) {
            if ((value[0] >> 16) != 5)
               continue;
            value[4] = 1;
         } else {
            if (transpose)
               continue;
            value[3]++;
         }
         struct panvk_data_graph *g = (void *)1;
         assert(parse(c, code, &g) == VK_ERROR_FEATURE_NOT_PRESENT && !g);
         rejected++;
      }
   }
   free(code);
   return rejected;
}

int
main(void)
{
   unsigned checked = 0, rejected = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(cases); i++) {
      const struct transform_case *c = &cases[i];
      struct panvk_data_graph *g;
      assert(parse(c, c->code, &g) == VK_SUCCESS);
      assert(g->pass_count == c->passes);
      if (c->constant)
         check_constant(c, g);
      checked += check_coordinates(c, g);
      rejected += check_rejection(c);
      ralloc_free(g);
   }
   printf("transform graphs=%zu coordinates=%u rejected=%u\n",
          ARRAY_SIZE(cases), checked, rejected);
   return 0;
}
