/*
 * Copyright (C) 2021-2026 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "pan_nir.h"

/* This optimization pass, intended to run once right before code emission,
 * analyzes direct word-aligned UBO reads and promotes a subset to moves to
 * push constant loads. It is the sole populator of the UBO push data
 * structure returned back to the command stream.
 */

/* Represents use data for a single UBO */

#define MAX_UBO_WORDS (65536 / 16)

struct pushable_ubo {
   uint32_t key;
   BITSET_DECLARE(pushed, MAX_UBO_WORDS);

   uint8_t range[MAX_UBO_WORDS];
   uint32_t weight[MAX_UBO_WORDS];
};

static_assert(PAN_MAX_PUSH <= 256, "We assume an FAU index fits in a uint8_t");

typedef struct {
   BITSET_DECLARE(row, PAN_MAX_PUSH);
} adjacency_row;

struct opt_push_ubo_ctx {
   struct pan_fau_layout *fau;
   const struct pan_compile_inputs *inputs;

   /* Start index of what we can reorder */
   unsigned reord_start;

   /* Mask of UBOs that are still UBOs at the end of this pass */
   uint32_t ubo_mask;

   struct hash_table_u64 *ubos;

   /* Interference graph for push re-ordering */
   adjacency_row adjacency[PAN_MAX_PUSH];
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

   /* We can't handle unaligned push constant access today */
   if ((offset % 4) != 0 || offset >= MAX_UBO_WORDS * 4 ||
       bytes > MAX_UBO_WORDS * 4 - offset)
      return range;

   range.word = offset / 4;
   range.nr_words = DIV_ROUND_UP(bytes, 4);

   return range;
}

static bool
analyze_ubo_intr(nir_builder *b, nir_intrinsic_instr *load, void *data)
{
   struct opt_push_ubo_ctx *ctx = data;
   const struct ubo_range range = get_pushable_ubo_range(load, ctx);
   if (range.key < 0 || range.nr_words < 0)
      return false;

   /* Blend constants are handled by bi_pick_blend_constants, don't
    * push them a second time.
    */
   if (b->shader->info.stage == MESA_SHADER_FRAGMENT &&
       !ctx->inputs->fau.push_ubo_handles) {
      /* PAN_UBO_SYSVALS from the gallium driver */
      unsigned sysval_ubo = 1;
      if (range.key == sysval_ubo &&
          range.word + range.nr_words <= 4)
         return false;
   }

   struct pushable_ubo *ubo = get_ubo(ctx, range.key, true);
   ubo->range[range.word] = MAX2(ubo->range[range.word], range.nr_words);

   unsigned depth = 0;
   for (nir_cf_node *node = load->instr.block->cf_node.parent; node;
        node = node->parent)
      depth += node->type == nir_cf_node_loop;

   uint64_t weight = ubo->weight[range.word];
   weight += pan_loop_weight(depth);
   ubo->weight[range.word] = MIN2(weight, UINT32_MAX);

   return false;
}

static void
add_ubo_push(struct opt_push_ubo_ctx *ctx, struct pushable_ubo *ubo, unsigned word)
{
   assert(!BITSET_TEST(ubo->pushed, word));

   BITSET_SET(ubo->pushed, word);

   pan_fau_emit_reloc(ctx->fau, (struct pan_ubo_relocation) {
      .ubo = ubo->key,
      .offset = word * 4,
   });
}

static bool
ubo_range_is_pushed(struct opt_push_ubo_ctx *ctx, struct ubo_range range)
{
   if (range.nr_words < 0)
      return false;

   const struct pushable_ubo *ubo = get_ubo(ctx, range.key, false);
   if (!ubo)
      return false;
   for (unsigned w = 0; w < range.nr_words; w++) {
      if (!BITSET_TEST(ubo->pushed, range.word + w))
         return false;
   }
   return true;
}

/* We always map blend constants from the first slot in the sysval UBO to the
 * first four FAU words, so that they can be accessed from a consistent
 * location from the blend shader.
 */
static void
add_blend_constants(struct opt_push_ubo_ctx *ctx)
{
   /* PAN_UBO_SYSVALS from the gallium driver */
   unsigned sysval_ubo = 1;
   assert(ctx->inputs->fau.pushable_ubos & BITFIELD_BIT(sysval_ubo));

   /* Blend constants are the first, non-reorderable ("fixed") relocations */
   assert(ctx->fau->count == 0 && ctx->fau->reserved == 0);

   struct pushable_ubo *ubo = get_ubo(ctx, sysval_ubo, true);
   for (unsigned channel = 0; channel < 4; channel++)
      add_ubo_push(ctx, ubo, channel);
}

struct ubo_candidate {
   struct pushable_ubo *ubo;
   unsigned word, range;
   uint32_t weight;
};

static int
compare_ubo_candidates(const void *a_, const void *b_)
{
   const struct ubo_candidate *a = a_, *b = b_;
   uint64_t wa = (uint64_t)a->weight * b->range;
   uint64_t wb = (uint64_t)b->weight * a->range;
   if (wa != wb)
      return wa > wb ? -1 : 1;
   if (a->ubo->key != b->ubo->key)
      return a->ubo->key > b->ubo->key ? -1 : 1;
   return (a->word > b->word) - (a->word < b->word);
}

static void
pick_ubo_push_words(struct opt_push_ubo_ctx *ctx)
{
   struct util_dynarray candidates;
   util_dynarray_init(&candidates, ctx->ubos);
   hash_table_u64_foreach(ctx->ubos, entry) {
      struct pushable_ubo *ubo = entry.data;
      for (unsigned word = 0; word < MAX_UBO_WORDS; word++) {
         if (ubo->range[word]) {
            struct ubo_candidate candidate = {
               .ubo = ubo,
               .word = word,
               .range = ubo->range[word],
               .weight = ctx->inputs->fau.push_ubo_handles ? ubo->weight[word] : 0,
            };
            util_dynarray_append_typed(&candidates, struct ubo_candidate, candidate);
         }
      }
   }

   unsigned count = util_dynarray_num_elements(&candidates, struct ubo_candidate);
   if (count > 1)
      qsort(util_dynarray_begin(&candidates), count,
            sizeof(struct ubo_candidate), compare_ubo_candidates);

   util_dynarray_foreach(&candidates, struct ubo_candidate, c) {
      unsigned needed = 0;
      for (unsigned w = 0; w < c->range; w++)
         needed += !BITSET_TEST(c->ubo->pushed, c->word + w);
      if (needed > pan_fau_available(ctx->fau)) {
         if (ctx->inputs->fau.push_ubo_handles)
            continue;
         break;
      }
      for (unsigned w = 0; w < c->range; w++) {
         if (!BITSET_TEST(c->ubo->pushed, c->word + w))
            add_ubo_push(ctx, c->ubo, c->word + w);
      }
   }
   util_dynarray_fini(&candidates);
}

/*
 * Create an undirected graph where nodes are 32-bit uniform indices and edges
 * represent that two nodes are used in the same instruction.
 *
 * The graph is constructed as an adjacency matrix stored in ctx->adjacency.
 */
static bool
analyze_alu_intr(nir_builder *b, nir_alu_instr *alu, void *data)
{
   struct opt_push_ubo_ctx *ctx = data;
   if (nir_op_is_vec_or_mov(alu->op))
      return false;

   uint8_t nodes[NIR_MAX_VEC_COMPONENTS * 2];
   uint8_t node_count = 0;

   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
      /* We only care about the first swizzle component since this pass should
       * be run after we've already reduced ALU widths down to where we only
       * really access one word per ALU op unless it's 64-bit.
       *
       * In theory, it might be useful to chase [un]pack but this pass is
       * only ever used for OpenGL where most uniforms are 32-bit.
       */
      nir_scalar s =
         nir_scalar_resolved(alu->src[i].src.ssa, alu->src[i].swizzle[0]);

      nir_instr *s_instr = nir_def_instr(s.def);
      if (s_instr->type != nir_instr_type_intrinsic)
         continue;

      const struct ubo_range range =
         get_pushable_ubo_range(nir_instr_as_intrinsic(s_instr), ctx);
      if (!ubo_range_is_pushed(ctx, range))
         continue;

      unsigned word = range.word + (s.comp * s.def->bit_size) / 32;
      for (unsigned w = 0; w < DIV_ROUND_UP(s.def->bit_size, 32); w++) {
         unsigned fau_word = pan_lookup_pushed_ubo(ctx->fau, range.key,
                                                  (word + w) * 4);
         assert(fau_word < PAN_MAX_PUSH);
         assert(!BITSET_TEST(ctx->fau->is_const, fau_word));
         if (fau_word >= ctx->reord_start)
            nodes[node_count++] = fau_word;
      }
   }

   /* Create clique connecting nodes[] */
   for (unsigned i = 0; i < node_count; ++i) {
      for (unsigned j = 0; j < node_count; ++j) {
         if (i == j)
            continue;

         unsigned x = nodes[i], y = nodes[j];

         /* Add undirected edge between the nodes */
         BITSET_SET(ctx->adjacency[x].row, y);
         BITSET_SET(ctx->adjacency[y].row, x);
      }
   }

   return false;
}

/* Find the connected component containing `node` with depth-first search */
static void
find_component(const adjacency_row *adjacency, BITSET_WORD *visited,
               uint8_t *component, uint8_t *size, uint8_t node)
{
   uint32_t neighbour;

   BITSET_SET(visited, node);
   component[(*size)++] = node;

   BITSET_FOREACH_SET(neighbour, adjacency[node].row, PAN_MAX_PUSH) {
      if (!BITSET_TEST(visited, neighbour)) {
         find_component(adjacency, visited, component, size, neighbour);
      }
   }
}

/*
 * Optimization pass to reorder uniforms. The goal is to reduce the number of
 * moves we emit when lowering FAU. The pass groups uniforms used by the same
 * ALU instruction.
 *
 * The pass works by creating a graph of pushed uniforms, where edges denote
 * the "both 32-bit uniforms required by the same instruction" relationship.
 * This is done by analyze_alu_intr() above.  We then perform depth-first
 * search on this graph to find the connected components, where each connected
 * component is a cluster of uniforms that are used together. We then select
 * pairs of uniforms from each connected component.  The remaining unpaired
 * uniforms (from components of odd sizes) are paired together arbitrarily.
 */
static void
reorder_ubo_push_words(struct opt_push_ubo_ctx *ctx)
{
   BITSET_DECLARE(visited, PAN_MAX_PUSH) = {0};

   uint8_t ordering[PAN_MAX_PUSH] = {0};
   uint8_t unpaired[PAN_MAX_PUSH] = {0};
   uint8_t pushed = ctx->reord_start, unpaired_count = 0;

   for (unsigned i = ctx->reord_start; i < ctx->fau->count; i++) {
      /* We're the only thing to push anything so far */
      assert(!BITSET_TEST(ctx->fau->is_const, i));
      if (BITSET_TEST(visited, i))
         continue;

      uint8_t component[PAN_MAX_PUSH] = {0};
      uint8_t size = 0;
      find_component(ctx->adjacency, visited, component, &size, i);

      /* If there is an odd number of uses, at least one use must be
       * unpaired. Arbitrarily take the last one.
       */
      if (size % 2)
         unpaired[unpaired_count++] = component[--size];

      /* The rest of uses are paired */
      assert((size % 2) == 0);

      /* Push the paired uses */
      assert(pushed + (unsigned)size <= PAN_MAX_PUSH);
      typed_memcpy(ordering + pushed, component, size);
      pushed += size;
   }

   /* Push unpaired nodes at the end */
   typed_memcpy(ordering + pushed, unpaired, unpaired_count);
   pushed += unpaired_count;

   assert(pushed == ctx->fau->count);

   union pan_fau_entry fau_words[PAN_MAX_PUSH];
   typed_memcpy(fau_words, ctx->fau->words, ctx->fau->count);

   for (unsigned i = ctx->reord_start; i < pushed; i++) {
      assert(ordering[i] < ctx->fau->count);
      ctx->fau->words[i] = fau_words[ordering[i]];
   }
}

static bool
lower_ubo_intr(nir_builder *b, nir_intrinsic_instr *load, void *data)
{
   struct opt_push_ubo_ctx *ctx = data;
   if (load->intrinsic != nir_intrinsic_load_ubo)
      return false;
   const struct ubo_range range = get_pushable_ubo_range(load, ctx);
   if (range.key < 0) {
      /* We don't even know the UBO index */
      if (!ctx->inputs->fau.push_ubo_handles)
         ctx->ubo_mask = ~0;
      return false;
   }

   if (!ubo_range_is_pushed(ctx, range)) {
      /* We couldn't push this one */
      if (!ctx->inputs->fau.push_ubo_handles)
         ctx->ubo_mask |= BITFIELD_BIT(range.key);
      return false;
   }

   b->cursor = nir_before_instr(&load->instr);

   /* After re-ordering, we can't use word_idx anymore and we have to just
    * search for the result.  We could theoretically plumb the ordering
    * array through here but that would get fragile.
    *
    * We also can't assume that load_ubo are contiguous so we need to break
    * it into per-word loads.  Fortunately, Kraid should be able to clean up
    * this mess.
    */
   nir_def *words[NIR_MAX_VEC_COMPONENTS * 2];
   for (unsigned w = 0; w < range.nr_words; w++) {
      uint32_t fau_word =
         pan_lookup_pushed_ubo(ctx->fau, range.key, (range.word + w) * 4);
      words[w] = nir_load_push_constant(b, 1, 32, nir_imm_int(b, fau_word * 4),
                                        .align_mul = 4, .align_offset = 0);
   }

   nir_def *val = nir_extract_bits(b, words, range.nr_words, 0,
                                   load->def.num_components,
                                   load->def.bit_size);
   nir_def_replace(&load->def, val);

   return true;
}

bool
pan_nir_opt_push_ubo(nir_shader *nir,
                     const struct pan_compile_inputs *inputs,
                     struct pan_fau_layout *fau,
                     uint32_t *ubo_mask_out)
{
   struct opt_push_ubo_ctx ctx = {
      .fau = fau,
      .inputs = inputs,
   };

   assert(fau->reserved <= fau->count && fau->count <= fau->max);
   ctx.ubos = _mesa_hash_table_u64_create(nir);

   /* Analyze load_ubo intrinsics */
   nir_shader_intrinsics_pass(nir, analyze_ubo_intr, nir_metadata_all, &ctx);

   /* We first pick the blend constants, those cannot be reordered */
   if (nir->info.stage == MESA_SHADER_FRAGMENT && !inputs->fau.push_ubo_handles)
      add_blend_constants(&ctx);
   ctx.reord_start = fau->count;

   pick_ubo_push_words(&ctx);
   ctx.reord_start = MIN2(ALIGN_POT(ctx.reord_start, 2), fau->count);

   /* Analyze ALU instructions to build the interference graph */
   nir_shader_alu_pass(nir, analyze_alu_intr, nir_metadata_all, &ctx);

   reorder_ubo_push_words(&ctx);

   bool progress = nir_shader_intrinsics_pass(nir, lower_ubo_intr,
                                              nir_metadata_control_flow, &ctx);

   *ubo_mask_out = ctx.ubo_mask;
   ralloc_free(ctx.ubos);

   return progress;
}
