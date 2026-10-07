#include "panvk_nir_rt_traversal.h"
#include "bvh/panvk_bvh.h"
#include "util/ralloc.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

struct fixture {
   _Alignas(8) uint8_t as[512];
   _Alignas(8) uint8_t blas[512];
};

static bool
load_fixture(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_global ||
       !nir_src_is_const(intr->src[0]))
      return false;
   struct fixture *fixture = data;
   uint64_t address = nir_src_as_uint(intr->src[0]);
   const uint8_t *bytes;
   if (address >= 0x1000 && address < 0x1000 + sizeof(fixture->as))
      bytes = fixture->as + address - 0x1000;
   else {
      assert(address >= 0x2000 && address < 0x2000 + sizeof(fixture->blas));
      bytes = fixture->blas + address - 0x2000;
   }
   nir_const_value values[4] = {0};
   for (unsigned c = 0; c < intr->def.num_components; c++)
      memcpy(&values[c], bytes + c * intr->def.bit_size / 8, intr->def.bit_size / 8);
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = nir_build_imm(b, intr->def.num_components, intr->def.bit_size, values);
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

static void
check_vertex(unsigned primitive, unsigned rotation, unsigned column,
             unsigned expected)
{
   struct fixture fixture = {0};
   struct panvk_bvh_header *as = (void *)fixture.as;
   as->node_count = 2;
   struct panvk_bvh_instance_leaf *instance = (void *)(fixture.as + 256);
   instance->blas_ptr = 0x2000;
   struct panvk_bvh_triangle_leaf *leaf = (void *)(fixture.blas + 192);
   for (unsigned v = 0; v < 4; v++)
      for (unsigned c = 0; c < 3; c++)
         leaf->coords[v][c] = 10 * v + c + 0.25f;
   leaf->geometry_id_a_flags = rotation << (primitive ? 28 : 26);
   const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, NULL);
   nir_def *value = panvk_nir_rt_triangle_vertex(&b, nir_imm_int64(&b, 0x1000),
      nir_imm_int(&b, 0x80000000), nir_imm_int(&b, 12 + primitive), column);
   nir_store_global(&b, value, nir_imm_int64(&b, 0x3000), .align_mul = 4);
   bool progress;
   do {
      progress = nir_opt_constant_folding(b.shader);
      progress |= nir_shader_intrinsics_pass(b.shader, load_fixture, nir_metadata_none, &fixture);
      progress |= nir_opt_dce(b.shader);
   } while (progress);
   nir_validate_shader(b.shader, "position fetch");
   nir_intrinsic_instr *store = nir_instr_as_intrinsic(nir_block_last_instr(nir_impl_last_block(b.impl)));
   assert(store->intrinsic == nir_intrinsic_store_global);
   assert(nir_src_is_const(store->src[0]));
   const nir_const_value *result = nir_src_as_const_value(store->src[0]);
   for (unsigned c = 0; c < 3; c++)
      assert(result[c].f32 == leaf->coords[expected][c]);
   ralloc_free(b.shader);
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   const unsigned orders[2][3][3] = {
      {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}},
      {{3, 2, 1}, {2, 1, 3}, {1, 3, 2}},
   };
   const unsigned codes[] = {1, 3, 2};
   for (unsigned triangle = 0; triangle < 2; triangle++)
      for (unsigned rotation = 0; rotation < 3; rotation++)
         for (unsigned vertex = 0; vertex < 3; vertex++)
            check_vertex(triangle, codes[rotation], vertex, orders[triangle][rotation][vertex]);
   glsl_type_singleton_decref();
   puts("18 position fetch cases passed");
   return 0;
}
