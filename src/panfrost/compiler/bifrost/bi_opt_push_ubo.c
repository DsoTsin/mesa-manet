/*
 * Copyright (C) 2021 Collabora, Ltd.
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "bi_builder.h"
#include "compiler.h"
#include "util/u_dynarray.h"

/* This optimization pass, intended to run once after code emission but before
 * copy propagation, analyzes direct word-aligned UBO reads and promotes a
 * subset to moves from FAU. It is the sole populator of the UBO push data
 * structure returned back to the command stream.
 *
 * UBOs are identified by a key: the UBO number for the Gallium driver, or the
 * resource table and index of a Valhall resource handle when the driver sets
 * inputs->fau.push_ubo_handles (see pan_ubo_reloc_key()). */

#define BI_UBO_KEY_INVALID UINT32_MAX

static bool
bi_is_ubo(bi_instr *ins)
{
   return (bi_get_opcode_props(ins)->message == BIFROST_MESSAGE_LOAD) &&
          (ins->seg == BI_SEG_UBO);
}

static uint32_t
bi_ubo_key(bi_context *ctx, uint32_t handle)
{
   if (!ctx->inputs->fau.push_ubo_handles)
      return pan_res_handle_get_index(handle);

   unsigned index = pan_res_handle_get_index(handle);
   if (index >= (1u << PAN_UBO_RELOC_INDEX_BITS))
      return BI_UBO_KEY_INVALID;

   return pan_ubo_reloc_key(pan_res_handle_get_table(handle), index);
}

static bool
bi_is_pushable_ubo(bi_context *ctx, bi_instr *ins)
{
   if (!(bi_is_ubo(ins) && (ins->src[0].type == BI_INDEX_CONSTANT) &&
         (ins->src[1].type == BI_INDEX_CONSTANT)))
      return false;

   uint32_t ubo = bi_ubo_key(ctx, ins->src[1].value);
   unsigned offset = ins->src[0].value;

   if ((offset & 0x3) != 0 || ubo == BI_UBO_KEY_INVALID)
      return false;

   return ctx->inputs->fau.push_ubo_handles ||
          (ctx->inputs->fau.pushable_ubos & BITFIELD_BIT(ubo));
}

/* Represents use data for a single UBO */

#define MAX_UBO_WORDS (65536 / 16)

struct bi_ubo_block {
   BITSET_DECLARE(pushed, MAX_UBO_WORDS);
   uint8_t range[MAX_UBO_WORDS];
   /* Reads of each word, weighted by loop depth */
   uint32_t weight[MAX_UBO_WORDS];
};

/* Resource handles don't give a small dense UBO numbering, so blocks are
 * allocated per distinct key, up to this many UBOs per shader. */
#define BI_MAX_PUSH_HANDLE_UBOS 32

struct bi_ubo_analysis {
   /* Per block analysis */
   unsigned nr_blocks;
   struct bi_ubo_block *blocks;
   /* Key of each block (identity for the Gallium numbering) */
   uint32_t *keys;
};

/* Block of a key, or -1 (the handle numbering only creates blocks up to
 * BI_MAX_PUSH_HANDLE_UBOS keys). */
static int
bi_ubo_block_index(bi_context *ctx, struct bi_ubo_analysis *res, uint32_t key,
                   bool create)
{
   if (!ctx->inputs->fau.push_ubo_handles)
      return key < res->nr_blocks ? (int)key : -1;

   for (unsigned i = 0; i < res->nr_blocks; ++i) {
      if (res->keys[i] == key)
         return i;
   }

   if (!create || res->nr_blocks == BI_MAX_PUSH_HANDLE_UBOS)
      return -1;

   res->keys[res->nr_blocks] = key;
   return res->nr_blocks++;
}

static struct bi_ubo_analysis
bi_analyze_ranges(bi_context *ctx)
{
   const bool handles = ctx->inputs->fau.push_ubo_handles;
   const unsigned max_blocks =
      handles ? BI_MAX_PUSH_HANDLE_UBOS : ctx->nir->info.num_ubos + 1;
   struct bi_ubo_analysis res = {
      .nr_blocks = handles ? 0 : max_blocks,
   };

   res.blocks = calloc(max_blocks, sizeof(struct bi_ubo_block));
   res.keys = calloc(max_blocks, sizeof(uint32_t));
   for (unsigned i = 0; !handles && i < max_blocks; ++i)
      res.keys[i] = i;

   uint8_t *depth = bi_loop_depths(ctx);

   bi_foreach_block(ctx, blk)
   bi_foreach_instr_in_block(blk, ins) {
      if (!bi_is_pushable_ubo(ctx, ins))
         continue;

      uint32_t ubo = bi_ubo_key(ctx, ins->src[1].value);
      unsigned word = ins->src[0].value / 4;
      unsigned channels = bi_get_opcode_props(ins)->sr_count;

      assert(channels > 0 && channels <= 4);

      /* Blend constants are handled by bi_pick_blend_constants, don't push
       * them a second time. */
      if (ctx->stage == MESA_SHADER_FRAGMENT && !handles) {
         /* PAN_UBO_SYSVALS from the gallium driver */
         unsigned sysval_ubo = 1;
         if(ubo == sysval_ubo && word == 0)
            continue;
      }

      if (word >= MAX_UBO_WORDS)
         continue;

      int block = bi_ubo_block_index(ctx, &res, ubo, true);
      if (block < 0)
         continue;

      /* Must use max if the same base is read with different channel
       * counts, which is possible with nir_opt_shrink_vectors */
      uint8_t *range = res.blocks[block].range;
      range[word] = MAX2(range[word], channels);
      res.blocks[block].weight[word] += bi_loop_weight(depth[blk->index]);
   }

   free(depth);
   return res;
}

/* We always map blend constants from the first slot in the sysval UBO to the
 * first four FAU words, so that they can be accessed from a consistent
 * location from the blend shader. */
static void
bi_pick_blend_constants(bi_context *ctx, struct bi_ubo_analysis *analysis)
{
   /* PAN_UBO_SYSVALS from the gallium driver */
   unsigned sysval_ubo = 1;
   unsigned offset = 0;
   struct pan_fau_layout *fau = ctx->info.fau;

   assert(ctx->inputs->fau.pushable_ubos & BITFIELD_BIT(sysval_ubo));
   /* Blend constants are the first, non-reorderable ("fixed") relocations */
   assert(fau->count == 0 && fau->reserved == 0);

   for (unsigned channel = 0; channel < 4; channel++) {
      pan_fau_emit_reloc(fau, (struct pan_ubo_relocation) {
         .ubo = sysval_ubo,
         .offset = offset + channel * 4,
      });
   }

   BITSET_SET(analysis->blocks[sysval_ubo].pushed, offset);
}

/* Select UBO words to push. A sophisticated implementation would consider the
 * number of uses and perhaps the control flow to estimate benefit. This is not
 * sophisticated. Select from the last UBO first to prioritize sysvals. */

struct bi_ubo_candidate {
   unsigned block, word, range;
   uint32_t weight;
};

/* Most weight per pushed word first, then the analysis order */
static int
bi_cmp_ubo_candidate(const void *a_, const void *b_)
{
   const struct bi_ubo_candidate *a = a_, *b = b_;
   uint64_t wa = (uint64_t)a->weight * b->range;
   uint64_t wb = (uint64_t)b->weight * a->range;

   if (wa != wb)
      return wa > wb ? -1 : 1;
   if (a->block != b->block)
      return a->block > b->block ? -1 : 1;
   return a->word < b->word ? -1 : 1;
}

static void
bi_pick_ubo(struct pan_fau_layout *fau, struct bi_ubo_analysis *analysis)
{
   struct util_dynarray candidates;
   util_dynarray_init(&candidates, NULL);

   for (unsigned ubo = 0; ubo < analysis->nr_blocks; ++ubo) {
      struct bi_ubo_block *block = &analysis->blocks[ubo];

      for (unsigned r = 0; r < MAX_UBO_WORDS; ++r) {
         /* Don't push something we don't access */
         if (block->range[r] == 0)
            continue;

         util_dynarray_append_typed(&candidates, struct bi_ubo_candidate,
                                    ((struct bi_ubo_candidate){
                                       ubo, r, block->range[r],
                                       block->weight[r]}));
      }
   }

   qsort(util_dynarray_begin(&candidates),
         util_dynarray_num_elements(&candidates, struct bi_ubo_candidate),
         sizeof(struct bi_ubo_candidate), bi_cmp_ubo_candidate);

   util_dynarray_foreach(&candidates, struct bi_ubo_candidate, c) {
      /* Don't push more than possible, but keep filling with smaller ranges */
      if (pan_fau_available(fau) < c->range)
         continue;

      for (unsigned offs = 0; offs < c->range; ++offs)
         pan_fau_emit_reloc(fau, (struct pan_ubo_relocation) {
            .ubo = analysis->keys[c->block],
            .offset = (c->word + offs) * 4,
         });

      /* Mark it as pushed so we can rewrite */
      BITSET_SET(analysis->blocks[c->block].pushed, c->word);
   }

   util_dynarray_fini(&candidates);
}

void
bi_opt_push_ubo(bi_context *ctx)
{
   struct bi_ubo_analysis analysis = bi_analyze_ranges(ctx);
   struct pan_fau_layout *fau = ctx->info.fau;
   const bool handles = ctx->inputs->fau.push_ubo_handles;

   /* We first pick the blend constants, those cannot be reordered */
   if (ctx->stage == MESA_SHADER_FRAGMENT && !handles)
      bi_pick_blend_constants(ctx, &analysis);

   ctx->ubo_reloc.start = fau->count;
   bi_pick_ubo(fau, &analysis);
   ctx->ubo_reloc.end = fau->count;
   ctx->ubo_mask = 0;

   bi_foreach_instr_global_safe(ctx, ins) {
      if (!bi_is_ubo(ins))
         continue;

      if (!bi_is_pushable_ubo(ctx, ins)) {
         /* The load can't be pushed, so this UBO needs to be
          * uploaded conventionally */
         if (!handles) {
            if (ins->src[1].type == BI_INDEX_CONSTANT)
               ctx->ubo_mask |=
                  BITSET_BIT(pan_res_handle_get_index(ins->src[1].value));
            else
               ctx->ubo_mask = ~0;
         }

         continue;
      }

      uint32_t ubo = bi_ubo_key(ctx, ins->src[1].value);
      unsigned offset = ins->src[0].value;

      /* Check if we decided to push this */
      int block = bi_ubo_block_index(ctx, &analysis, ubo, false);
      if (block < 0 || offset / 4 >= MAX_UBO_WORDS ||
          !BITSET_TEST(analysis.blocks[block].pushed, offset / 4)) {
         if (!handles)
            ctx->ubo_mask |= BITSET_BIT(ubo);
         continue;
      }

      /* Replace the UBO load with moves from FAU */
      bi_builder b = bi_init_builder(ctx, bi_after_instr(ins));

      unsigned nr = bi_get_opcode_props(ins)->sr_count;
      bi_instr *vec = bi_collect_i32_to(&b, ins->dest[0], nr);

      bi_foreach_src(vec, w) {
         /* FAU is grouped in pairs (2 x 4-byte) */
         unsigned base =
            pan_lookup_pushed_ubo(ctx->info.fau, ubo, (offset + 4 * w));

         unsigned fau_idx = (base >> 1);
         unsigned fau_hi = (base & 1);

         vec->src[w] = bi_fau(BIR_FAU_UNIFORM | fau_idx, fau_hi);
      }

      bi_remove_instruction(ins);
   }

   free(analysis.blocks);
   free(analysis.keys);
}

typedef struct {
   BITSET_DECLARE(row, PAN_MAX_PUSH);
} adjacency_row;

/* Find the connected component containing `node` with depth-first search */
static void
bi_find_component(adjacency_row *adjacency, BITSET_WORD *visited,
                  unsigned *component, unsigned *size, unsigned node)
{
   unsigned neighbour;

   BITSET_SET(visited, node);
   component[(*size)++] = node;

   BITSET_FOREACH_SET(neighbour, adjacency[node].row, PAN_MAX_PUSH) {
      if (!BITSET_TEST(visited, neighbour)) {
         bi_find_component(adjacency, visited, component, size, neighbour);
      }
   }
}

static bool
bi_is_uniform(bi_index idx)
{
   return (idx.type == BI_INDEX_FAU) && (idx.value & BIR_FAU_UNIFORM);
}

/* Get the index of a uniform in 32-bit words from the start of FAU-RAM */
static unsigned
bi_uniform_word(bi_index idx)
{
   assert(bi_is_uniform(idx));
   assert(idx.offset <= 1);

   return ((idx.value & ~BIR_FAU_UNIFORM) << 1) | idx.offset;
}

/*
 * Create an undirected graph where nodes are 32-bit uniform indices and edges
 * represent that two nodes are used in the same instruction.
 *
 * The graph is constructed as an adjacency matrix stored in adjacency.
 */
static void
bi_create_fau_interference_graph(bi_context *ctx, adjacency_row *adjacency)
{
   bi_foreach_instr_global(ctx, I) {
      unsigned nodes[BI_MAX_SRCS] = {};
      unsigned node_count = 0;

      /* Set nodes[] to 32-bit uniforms accessed */
      bi_foreach_src(I, s) {
         if (bi_is_uniform(I->src[s])) {
            unsigned word = bi_uniform_word(I->src[s]);

            if (word >= ctx->ubo_reloc.start && word < ctx->ubo_reloc.end)
               nodes[node_count++] = word;
         }
      }

      /* Create clique connecting nodes[] */
      for (unsigned i = 0; i < node_count; ++i) {
         for (unsigned j = 0; j < node_count; ++j) {
            if (i == j)
               continue;

            unsigned x = nodes[i], y = nodes[j];
            assert(MAX2(x, y) < ctx->ubo_reloc.end);

            /* Add undirected edge between the nodes */
            BITSET_SET(adjacency[x].row, y);
            BITSET_SET(adjacency[y].row, x);
         }
      }
   }
}

/*
 * Optimization pass to reorder uniforms. The goal is to reduce the number of
 * moves we emit when lowering FAU. The pass groups uniforms used by the same
 * instruction.
 *
 * The pass works by creating a graph of pushed uniforms, where edges denote the
 * "both 32-bit uniforms required by the same instruction" relationship. We
 * perform depth-first search on this graph to find the connected components,
 * where each connected component is a cluster of uniforms that are used
 * together. We then select pairs of uniforms from each connected component.
 * The remaining unpaired uniforms (from components of odd sizes) are paired
 * together arbitrarily.
 *
 * After a new ordering is selected, pushed uniforms in the program and the
 * pan_fau_layout data structure must be remapped to use the new ordering.
 */
void
bi_opt_reorder_push(bi_context *ctx)
{
   adjacency_row adjacency[PAN_MAX_PUSH] = {0};
   BITSET_DECLARE(visited, PAN_MAX_PUSH) = {0};

   unsigned ordering[PAN_MAX_PUSH] = {0};
   unsigned unpaired[PAN_MAX_PUSH] = {0};
   unsigned pushed = 0, unpaired_count = 0;

   struct pan_fau_layout *fau = ctx->info.fau;
   unsigned push_offset = ctx->ubo_reloc.start;

   bi_create_fau_interference_graph(ctx, adjacency);

   for (unsigned i = push_offset; i < ctx->ubo_reloc.end; ++i) {
      /* All our UBO relocations are contiguous */
      assert(!BITSET_TEST(fau->is_const, i));
      if (BITSET_TEST(visited, i))
         continue;

      unsigned component[PAN_MAX_PUSH] = {0};
      unsigned size = 0;
      bi_find_component(adjacency, visited, component, &size, i);

      /* If there is an odd number of uses, at least one use must be
       * unpaired. Arbitrarily take the last one.
       */
      if (size % 2)
         unpaired[unpaired_count++] = component[--size];

      /* The rest of uses are paired */
      assert((size % 2) == 0);

      /* Push the paired uses */
      memcpy(ordering + pushed, component, sizeof(unsigned) * size);
      pushed += size;
   }

   /* Push unpaired nodes at the end */
   memcpy(ordering + pushed, unpaired, sizeof(unsigned) * unpaired_count);
   pushed += unpaired_count;

   /* Ordering is a permutation. Invert it for O(1) lookup. */
   unsigned old_to_new[PAN_MAX_PUSH] = {0};

   for (unsigned i = 0; i < push_offset; ++i) {
      old_to_new[i] = i;
   }

   for (unsigned i = 0; i < pushed; ++i) {
      assert(ordering[i] >= push_offset);
      old_to_new[ordering[i]] = push_offset + i;
   }

   /* Use new ordering throughout the program */
   bi_foreach_instr_global(ctx, I) {
      bi_foreach_src(I, s) {
         if (bi_is_uniform(I->src[s])) {
            unsigned node = bi_uniform_word(I->src[s]);
            unsigned new_node = old_to_new[node];
            I->src[s].value = BIR_FAU_UNIFORM | (new_node >> 1);
            I->src[s].offset = new_node & 1;
         }
      }
   }

   /* Use new ordering for push */
   struct pan_fau_layout old = *fau;
   for (unsigned i = 0; i < pushed; ++i)
      fau->words[push_offset + i] = old.words[ordering[i]];

   assert(ctx->ubo_reloc.end == ctx->ubo_reloc.start + pushed);
}
