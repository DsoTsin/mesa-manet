/*
 * Copyright (C) 2021-2026 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "pan_nir.h"

#define MAX_UBO_WORDS (65536 / 16)

struct pushable_ubo {
   uint32_t key;
   uint32_t handle;
   BITSET_DECLARE(pushed, MAX_UBO_WORDS);
   BITSET_DECLARE(read, MAX_UBO_WORDS);
   BITSET_DECLARE(wide, MAX_UBO_WORDS);
   uint8_t range[MAX_UBO_WORDS];
   uint32_t weight[MAX_UBO_WORDS];
   int16_t vword[MAX_UBO_WORDS];
};

struct opt_push_ubo_ctx {
   const struct pan_compile_inputs *inputs;
   struct pan_fau_virtual *virt;
   uint32_t ubo_mask;
   struct hash_table_u64 *ubos;
};

struct ubo_range {
   int32_t key;
   int8_t nr_words;
   uint16_t word;
};

static struct pushable_ubo *
get_ubo(struct opt_push_ubo_ctx *ctx, uint32_t key, bool create)
{
   struct pushable_ubo *ubo = _mesa_hash_table_u64_search(ctx->ubos, key);
   if (!ubo && create) {
      ubo = rzalloc(ctx->ubos, struct pushable_ubo);
      ubo->key = key;
      memset(ubo->vword, 0xff, sizeof(ubo->vword));
      _mesa_hash_table_u64_insert(ctx->ubos, key, ubo);
   }
   return ubo;
}

static struct ubo_range
get_pushable_ubo_range(nir_intrinsic_instr *load,
                       const struct opt_push_ubo_ctx *ctx)
{
   struct ubo_range range = {
      .key = -1,
      .nr_words = -1,
   };

   if (load->intrinsic != nir_intrinsic_load_ubo ||
       !nir_src_is_const(load->src[0]) ||
       !nir_src_is_const(load->src[1]))
      return range;

   const uint32_t handle = nir_src_as_uint(load->src[0]);
   const unsigned index = pan_res_handle_get_index(handle);
   if (ctx->inputs->fau.push_ubo_handles) {
      if (index >= (1u << PAN_UBO_RELOC_INDEX_BITS) || (handle >> 24) >= 64)
         return range;
      range.key = pan_ubo_reloc_key(pan_res_handle_get_table(handle), index);
   } else {
      if (index >= 32)
         return range;
      range.key = index;
      if (!(ctx->inputs->fau.pushable_ubos & BITFIELD_BIT(index)))
         return range;
   }

   const uint32_t offset = nir_src_as_uint(load->src[1]);
   assert(load->def.bit_size >= 8);
   const unsigned bytes = load->def.num_components * (load->def.bit_size / 8);

   if ((offset % 4) != 0 || offset >= MAX_UBO_WORDS * 4 ||
       bytes > MAX_UBO_WORDS * 4 - offset)
      return range;

   range.word = offset / 4;
   range.nr_words = DIV_ROUND_UP(bytes, 4);

   return range;
}

static uint32_t
ubo_read_words(nir_intrinsic_instr *load, unsigned nr_words)
{
   const unsigned bits = load->def.bit_size;
   uint32_t mask = 0;
   u_foreach_bit(c, nir_def_components_read(&load->def)) {
      unsigned last = MIN2(((c + 1) * bits - 1) / 32, nr_words - 1);
      for (unsigned w = c * bits / 32; w <= last; w++)
         mask |= BITFIELD_BIT(w);
   }
   return mask;
}

static bool
analyze_ubo_intr(nir_builder *b, nir_intrinsic_instr *load, void *data)
{
   struct opt_push_ubo_ctx *ctx = data;
   const struct ubo_range range = get_pushable_ubo_range(load, ctx);
   if (range.key < 0 || range.nr_words < 0)
      return false;

   const uint32_t read = ubo_read_words(load, range.nr_words);
   if (!read)
      return false;

   struct pushable_ubo *ubo = get_ubo(ctx, range.key, true);
   ubo->handle = nir_src_as_uint(load->src[0]);
   ubo->range[range.word] = MAX2(ubo->range[range.word], range.nr_words);
   u_foreach_bit(w, read) {
      BITSET_SET(ubo->read, range.word + w);
      if (load->def.bit_size == 64)
         BITSET_SET(ubo->wide, range.word + w);
   }

   unsigned depth = 0;
   for (nir_cf_node *node = load->instr.block->cf_node.parent; node;
        node = node->parent)
      depth += node->type == nir_cf_node_loop;

   uint64_t weight = ubo->weight[range.word];
   weight += pan_loop_weight(depth);
   ubo->weight[range.word] = MIN2(weight, UINT32_MAX);

   return false;
}

struct ubo_candidate {
   struct pushable_ubo *ubo;
   unsigned word, range, read;
   uint32_t weight;
};

static int
compare_ubo_candidates(const void *a_, const void *b_)
{
   const struct ubo_candidate *a = a_, *b = b_;
   uint64_t wa = (uint64_t)a->weight * b->read;
   uint64_t wb = (uint64_t)b->weight * a->read;
   if (wa != wb)
      return wa > wb ? -1 : 1;
   if (a->ubo->key != b->ubo->key)
      return a->ubo->key > b->ubo->key ? -1 : 1;
   return (a->word > b->word) - (a->word < b->word);
}

static unsigned
chunk_align(unsigned size)
{
   return size >= 4 ? 4 : size >= 2 ? 2 : 1;
}

static unsigned
max_chunk_words(const struct opt_push_ubo_ctx *ctx, const struct pushable_ubo *ubo,
                unsigned word)
{
   if (pan_arch(ctx->inputs->gpu_id) >= 14)
      return 4 - (word % 4);
   return BITSET_TEST(ubo->wide, word) ? 2 : 1;
}

static void
pick_ubo_push_words(struct opt_push_ubo_ctx *ctx, unsigned budget)
{
   struct util_dynarray candidates;
   util_dynarray_init(&candidates, ctx->ubos);
   hash_table_u64_foreach(ctx->ubos, entry) {
      struct pushable_ubo *ubo = entry.data;
      for (unsigned word = 0; word < MAX_UBO_WORDS; word++) {
         if (ubo->range[word]) {
            unsigned read = 0;
            for (unsigned w = 0; w < ubo->range[word]; w++)
               read += BITSET_TEST(ubo->read, word + w);
            struct ubo_candidate candidate = {
               .ubo = ubo,
               .word = word,
               .range = ubo->range[word],
               .read = read,
               .weight = ubo->weight[word],
            };
            util_dynarray_append_typed(&candidates, struct ubo_candidate,
                                       candidate);
         }
      }
   }

   unsigned count =
      util_dynarray_num_elements(&candidates, struct ubo_candidate);
   if (count > 1)
      qsort(util_dynarray_begin(&candidates), count,
            sizeof(struct ubo_candidate), compare_ubo_candidates);

   unsigned used = 0;
   util_dynarray_foreach(&candidates, struct ubo_candidate, c) {
      unsigned needed = 0;
      for (unsigned w = c->word; w < c->word + c->range; w++)
         needed += BITSET_TEST(c->ubo->read, w) && !BITSET_TEST(c->ubo->pushed, w);
      if (used + needed > budget)
         continue;
      for (unsigned w = c->word; w < c->word + c->range; w++) {
         if (BITSET_TEST(c->ubo->read, w))
            BITSET_SET(c->ubo->pushed, w);
      }
      used += needed;
   }
   util_dynarray_fini(&candidates);

   hash_table_u64_foreach(ctx->ubos, entry) {
      struct pushable_ubo *ubo = entry.data;
      unsigned word = 0;
      while (word < MAX_UBO_WORDS) {
         if (!BITSET_TEST(ubo->pushed, word)) {
            word++;
            continue;
         }
         unsigned limit = max_chunk_words(ctx, ubo, word);
         unsigned end = word + 1;
         while (end < MAX_UBO_WORDS && end - word < limit &&
                BITSET_TEST(ubo->pushed, end))
            end++;
         unsigned size = end - word;
         unsigned index = pan_fau_virtual_add(
            ctx->virt, size, chunk_align(size), PAN_FAU_VALUE_UBO,
            (struct pan_ubo_relocation){
               .ubo = ubo->key,
               .offset = word * 4,
            });
         ctx->virt->values[index].handle = ubo->handle;
         ctx->virt->values[index].loadable = ctx->inputs->fau.push_ubo_handles;
         for (unsigned w = 0; w < size; w++)
            ubo->vword[word + w] = ctx->virt->values[index].word + w;
         word = end;
      }
   }
}

static bool
ubo_range_is_pushed(struct opt_push_ubo_ctx *ctx, struct ubo_range range,
                    uint32_t read)
{
   if (range.nr_words < 0 || !read)
      return false;

   const struct pushable_ubo *ubo = get_ubo(ctx, range.key, false);
   if (!ubo)
      return false;
   u_foreach_bit(w, read) {
      if (!BITSET_TEST(ubo->pushed, range.word + w))
         return false;
   }
   return true;
}

static bool
lower_ubo_intr(nir_builder *b, nir_intrinsic_instr *load, void *data)
{
   struct opt_push_ubo_ctx *ctx = data;
   if (load->intrinsic != nir_intrinsic_load_ubo)
      return false;
   const struct ubo_range range = get_pushable_ubo_range(load, ctx);
   if (range.key < 0) {
      if (!ctx->inputs->fau.push_ubo_handles)
         ctx->ubo_mask = ~0;
      return false;
   }

   const uint32_t read =
      range.nr_words > 0 ? ubo_read_words(load, range.nr_words) : 0;
   if (!ubo_range_is_pushed(ctx, range, read)) {
      if (!ctx->inputs->fau.push_ubo_handles)
         ctx->ubo_mask |= BITFIELD_BIT(range.key);
      return false;
   }

   const struct pushable_ubo *ubo = get_ubo(ctx, range.key, false);
   b->cursor = nir_before_instr(&load->instr);

   nir_def *words[NIR_MAX_VEC_COMPONENTS * 2];
   for (unsigned w = 0; w < range.nr_words; w++) {
      if (!(read & BITFIELD_BIT(w))) {
         words[w] = nir_undef(b, 1, 32);
         continue;
      }
      assert(ubo->vword[range.word + w] >= 0);
      words[w] = nir_load_preamble(b, 1, 32,
                                   .base = ubo->vword[range.word + w]);
   }

   nir_def *val = nir_extract_bits(b, words, range.nr_words, 0,
                                   load->def.num_components,
                                   load->def.bit_size);
   nir_def_replace(&load->def, val);

   return true;
}

bool
pan_nir_opt_push_ubo(nir_shader *nir, const struct pan_compile_inputs *inputs,
                     struct pan_fau_virtual *virt, unsigned budget,
                     uint32_t *ubo_mask_out)
{
   struct opt_push_ubo_ctx ctx = {
      .inputs = inputs,
      .virt = virt,
   };

   ctx.ubos = _mesa_hash_table_u64_create(nir);

   nir_shader_intrinsics_pass(nir, analyze_ubo_intr, nir_metadata_all, &ctx);
   pick_ubo_push_words(&ctx, budget);

   bool progress = nir_shader_intrinsics_pass(nir, lower_ubo_intr,
                                              nir_metadata_control_flow, &ctx);

   *ubo_mask_out = ctx.ubo_mask;
   ralloc_free(ctx.ubos);

   return progress;
}
