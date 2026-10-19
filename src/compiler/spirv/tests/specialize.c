#undef NDEBUG
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv.h"
#include "compiler/glsl_types.h"
#include "util/ralloc.h"
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

struct module {
   uint32_t words[1024];
   unsigned count;
};

static void
emit(struct module *m, SpvOp op, unsigned count, ...)
{
   assert(m->count + count + 1 <= 1024);
   m->words[m->count++] = ((count + 1) << 16) | op;
   va_list args;
   va_start(args, count);
   for (unsigned i = 0; i < count; i++)
      m->words[m->count++] = va_arg(args, uint32_t);
   va_end(args);
}

static const uint32_t *
find(const uint32_t *words, size_t count, uint32_t id)
{
   for (size_t i = 5; i < count; i += words[i] >> 16) {
      unsigned op = words[i] & 0xffff;
      assert(op != SpvOpSpecConstant && op != SpvOpSpecConstantOp &&
             op != SpvOpSpecConstantTrue && op != SpvOpSpecConstantFalse &&
             op != SpvOpSpecConstantComposite);
      if ((op == SpvOpConstant || op == SpvOpConstantTrue ||
           op == SpvOpConstantFalse || op == SpvOpConstantComposite) &&
          words[i + 2] == id)
         return words + i;
   }
   assert(!"constant missing");
   return NULL;
}

static void
check(unsigned bits, bool override)
{
   struct module m = {.words = {SpvMagicNumber, 0x10600, 0, 32, 0}, .count = 5};
   emit(&m, SpvOpDecorate, 3, 10u, (uint32_t)SpvDecorationSpecId, 7u);
   emit(&m, SpvOpTypeInt, 3, 1u, bits, 0u);
   if (bits == 64)
      emit(&m, SpvOpSpecConstant, 4, 1u, 10u, 3u, 0u);
   else
      emit(&m, SpvOpSpecConstant, 3, 1u, 10u, 3u);
   emit(&m, SpvOpSpecConstantOp, 5, 1u, 11u, (uint32_t)SpvOpIAdd, 10u, 10u);
   emit(&m, SpvOpSpecConstantOp, 5, 1u, 12u, (uint32_t)SpvOpIMul, 11u, 10u);
   uint64_t input = bits == 64 ? UINT64_C(0x100000001) : 7;
   struct nir_spirv_specialization *spec = vtn_alloc_specialization(1);
   assert(vtn_add_specialization_entry(spec, 0, 7, bits / 8, &input, false));
   size_t count;
   uint32_t *words = spirv_specialize(NULL, m.words, m.count,
                                     override ? spec : NULL, &count);
   assert(words);
   uint64_t value = override ? input : 3;
   uint64_t mask = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
   const uint32_t *c = find(words, count, 12);
   uint64_t actual = c[3];
   if (bits == 64)
      actual |= (uint64_t)c[4] << 32;
   assert(actual == ((value * 2 * value) & mask));
   ralloc_free(words);
   vtn_free_specialization(spec);
}

static void
check_composite(bool override)
{
   struct module m = {.words = {SpvMagicNumber, 0x10600, 0, 40, 0}, .count = 5};
   emit(&m, SpvOpDecorate, 3, 10u, (uint32_t)SpvDecorationSpecId, 1u);
   emit(&m, SpvOpDecorate, 3, 11u, (uint32_t)SpvDecorationSpecId, 2u);
   emit(&m, SpvOpTypeBool, 1, 1u);
   emit(&m, SpvOpTypeInt, 3, 2u, 32u, 0u);
   emit(&m, SpvOpTypeVector, 3, 3u, 2u, 2u);
   emit(&m, SpvOpConstant, 3, 2u, 9u, 2u);
   emit(&m, SpvOpTypeArray, 3, 4u, 2u, 9u);
   emit(&m, SpvOpSpecConstantFalse, 2, 1u, 10u);
   emit(&m, SpvOpSpecConstant, 3, 2u, 11u, 3u);
   emit(&m, SpvOpSpecConstantOp, 6, 2u, 12u, (uint32_t)SpvOpSelect, 10u, 11u, 9u);
   emit(&m, SpvOpSpecConstantComposite, 4, 3u, 13u, 11u, 12u);
   emit(&m, SpvOpSpecConstantComposite, 4, 4u, 14u, 12u, 11u);
   emit(&m, SpvOpSpecConstantOp, 5, 2u, 15u,
        (uint32_t)SpvOpCompositeExtract, 14u, 1u);
   emit(&m, SpvOpSpecConstantOp, 6, 3u, 16u,
        (uint32_t)SpvOpCompositeInsert, 9u, 13u, 1u);
   emit(&m, SpvOpSpecConstantOp, 5, 2u, 17u,
        (uint32_t)SpvOpCompositeExtract, 16u, 1u);
   uint32_t yes = 123, input = 37;
   struct nir_spirv_specialization *spec = vtn_alloc_specialization(2);
   assert(vtn_add_specialization_entry(spec, 0, 1, 4, &yes, false));
   assert(vtn_add_specialization_entry(spec, 1, 2, 4, &input, false));
   size_t count;
   uint32_t *words = spirv_specialize(NULL, m.words, m.count,
                                     override ? spec : NULL, &count);
   assert(words && count > m.count);
   assert((find(words, count, 10)[0] & 0xffff) ==
          (override ? SpvOpConstantTrue : SpvOpConstantFalse));
   assert(find(words, count, 12)[3] == (override ? input : 2));
   assert(find(words, count, 15)[3] == (override ? input : 3));
   assert(find(words, count, 17)[3] == 2);
   ralloc_free(words);
   vtn_free_specialization(spec);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   for (unsigned bits = 8; bits <= 64; bits *= 2) {
      check(bits, false);
      check(bits, true);
   }
   check_composite(false);
   check_composite(true);
   glsl_type_singleton_decref();
   puts("specialization: 10 cases passed");
   return 0;
}
