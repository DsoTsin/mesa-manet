/*
 * Copyright © 2023 Bas Nieuwenhuizen
 * Copyright © 2024 Collabora, Ltd.
 * Copyright © 2026 Google LLC.
 * SPDX-License-Identifier: MIT
 */

#include "panvk_nir.h"
#include "nir_builder.h"
#include "util/hash_table.h"

#define CMAT_DIM 4
#define CMAT_PAIR_COLS 8
#define CMAT_QUAD_K 16

enum cmat_layout {
   CMAT_LAYOUT_WORD,
   CMAT_LAYOUT_PAIR,
   CMAT_LAYOUT_QUAD_A,
   CMAT_LAYOUT_QUAD_B,
};

struct cmat_block {
   unsigned rows, cols, elems;
};

static bool
cmat_paired(enum glsl_base_type type)
{
   return type == GLSL_TYPE_FLOAT16 || type == GLSL_TYPE_FLOAT_E4M3FN ||
          type == GLSL_TYPE_FLOAT_E5M2;
}

static enum cmat_layout
cmat_layout(struct glsl_cmat_description desc)
{
   if (cmat_paired(desc.element_type))
      return CMAT_LAYOUT_PAIR;

   if (desc.element_type == GLSL_TYPE_INT8 ||
       desc.element_type == GLSL_TYPE_UINT8) {
      if (desc.use == GLSL_CMAT_USE_A && desc.cols % CMAT_QUAD_K == 0)
         return CMAT_LAYOUT_QUAD_A;
      if (desc.use == GLSL_CMAT_USE_B && desc.rows % CMAT_QUAD_K == 0)
         return CMAT_LAYOUT_QUAD_B;
   }

   return CMAT_LAYOUT_WORD;
}

static bool
cmat_quad(struct glsl_cmat_description desc)
{
   enum cmat_layout layout = cmat_layout(desc);
   return layout == CMAT_LAYOUT_QUAD_A || layout == CMAT_LAYOUT_QUAD_B;
}

static struct glsl_cmat_description
cmat_word_desc(struct glsl_cmat_description desc)
{
   desc.use = GLSL_CMAT_USE_ACCUMULATOR;
   return desc;
}

static struct cmat_block
cmat_block(struct glsl_cmat_description desc)
{
   switch (cmat_layout(desc)) {
   case CMAT_LAYOUT_PAIR:
      return (struct cmat_block){CMAT_DIM, MIN2(desc.cols, CMAT_PAIR_COLS), 2};
   case CMAT_LAYOUT_QUAD_A:
      return (struct cmat_block){CMAT_DIM, CMAT_QUAD_K, 4};
   case CMAT_LAYOUT_QUAD_B:
      return (struct cmat_block){CMAT_QUAD_K, CMAT_DIM, 4};
   default:
      return (struct cmat_block){CMAT_DIM, CMAT_DIM, 1};
   }
}

static unsigned
get_cmat_length(struct glsl_cmat_description desc)
{
   const struct cmat_block blk = cmat_block(desc);
   return (desc.rows / blk.rows) * (desc.cols / blk.cols) * blk.elems;
}

static void
cmat_coord(nir_builder *b, struct glsl_cmat_description desc, nir_def *lane,
           unsigned comp, nir_def **row, nir_def **col)
{
   const struct cmat_block blk = cmat_block(desc);
   const unsigned idx = comp / blk.elems;
   const unsigned sub = comp % blk.elems;
   const unsigned bpr = desc.cols / blk.cols;
   const unsigned row0 = blk.rows * (idx / bpr);
   const unsigned col0 = blk.cols * (idx % bpr);
   nir_def *q = nir_iand_imm(b, lane, CMAT_DIM - 1);
   nir_def *g = nir_ushr_imm(b, lane, 2);

   switch (cmat_layout(desc)) {
   case CMAT_LAYOUT_PAIR:
      q = nir_iadd_imm(b, nir_ishl_imm(b, q, 1), sub);
      if (blk.cols < CMAT_PAIR_COLS)
         q = nir_iand_imm(b, q, blk.cols - 1);
      *row = nir_iadd_imm(b, g, row0);
      *col = nir_iadd_imm(b, q, col0);
      break;
   case CMAT_LAYOUT_QUAD_A:
      *row = nir_iadd_imm(b, g, row0);
      *col = nir_iadd_imm(b, nir_ishl_imm(b, q, 2), col0 + sub);
      break;
   case CMAT_LAYOUT_QUAD_B:
      *row = nir_iadd_imm(b, nir_ishl_imm(b, g, 2), row0 + sub);
      *col = nir_iadd_imm(b, q, col0);
      break;
   default:
      *row = nir_iadd_imm(b, g, row0);
      *col = nir_iadd_imm(b, q, col0);
      break;
   }
}

static unsigned
cmat_access_width(struct glsl_cmat_description desc, bool row_major)
{
   switch (cmat_layout(desc)) {
   case CMAT_LAYOUT_PAIR:
      return row_major ? 2 : 1;
   case CMAT_LAYOUT_QUAD_A:
      return row_major ? 4 : 1;
   case CMAT_LAYOUT_QUAD_B:
      return row_major ? 1 : 4;
   default:
      return 1;
   }
}

static const struct glsl_type *
remap_matrix_type(struct hash_table *mapping, const struct glsl_type *orig)
{
   struct hash_entry *entry = _mesa_hash_table_search(mapping, orig);
   if (entry)
      return entry->data;

   const struct glsl_type *new_type = orig;
   const struct glsl_type *leaf = glsl_without_array(orig);

   if (glsl_type_is_cmat(leaf)) {
      struct glsl_cmat_description desc = *glsl_get_cmat_description(leaf);
      new_type = glsl_type_wrap_in_arrays(
         glsl_vector_type(desc.element_type, get_cmat_length(desc)), orig);
   }

   _mesa_hash_table_insert(mapping, orig, (void *)new_type);
   return new_type;
}

static bool
remap_type_in_place(struct hash_table *mapping, const struct glsl_type **type)
{
   const struct glsl_type *new_type = remap_matrix_type(mapping, *type);
   if (new_type == *type)
      return false;

   *type = new_type;
   return true;
}

static struct glsl_cmat_description
cmat_src_desc(nir_src src)
{
   return *glsl_get_cmat_description(nir_src_as_deref(src)->type);
}

static nir_def *
load_cmat_src(nir_builder *b, nir_src src)
{
   nir_deref_instr *deref = nir_src_as_deref(src);
   struct glsl_cmat_description desc = *glsl_get_cmat_description(deref->type);

   return nir_build_load_deref(b, get_cmat_length(desc),
                               glsl_base_type_bit_size(desc.element_type),
                               &deref->def, 0);
}

static void
store_cmat_src(nir_builder *b, nir_src dst, nir_def *val)
{
   nir_store_deref(b, nir_src_as_deref(dst), val, ~0);
}

static bool
lower_cmat_load_store(nir_builder *b, nir_intrinsic_instr *intr)
{
   const bool is_load = intr->intrinsic == nir_intrinsic_cmat_load;
   const struct glsl_cmat_description desc = cmat_src_desc(intr->src[!is_load]);
   const bool row_major =
      nir_intrinsic_matrix_layout(intr) == GLSL_MATRIX_LAYOUT_ROW_MAJOR;
   const unsigned length = get_cmat_length(desc);
   const unsigned width = cmat_access_width(desc, row_major);
   const bool half_block = cmat_layout(desc) == CMAT_LAYOUT_PAIR &&
                           cmat_block(desc).cols < CMAT_PAIR_COLS;
   const unsigned elem_size = glsl_base_type_bit_size(desc.element_type) / 8;
   nir_deref_instr *deref = nir_src_as_deref(intr->src[is_load]);

   const unsigned ptr_vec = glsl_get_vector_elements(deref->type);
   deref = nir_build_deref_cast(b, &deref->def, deref->modes,
                                glsl_scalar_type(desc.element_type), elem_size);
   const unsigned idx_bits = deref->def.bit_size;
   nir_def *stride =
      nir_imul_imm(b, nir_u2uN(b, intr->src[2].ssa, idx_bits), ptr_vec);
   nir_def *lane = nir_load_subgroup_invocation(b);

   nir_def *elems[NIR_MAX_VEC_COMPONENTS];
   if (!is_load) {
      nir_def *src = load_cmat_src(b, intr->src[1]);
      for (unsigned i = 0; i < length; i++)
         elems[i] = nir_channel(b, src, i);

      if (half_block)
         nir_push_if(b, nir_ult_imm(b, nir_iand_imm(b, lane, CMAT_DIM - 1), 2));
   }

   for (unsigned i = 0; i < length; i += width) {
      nir_def *row, *col;
      cmat_coord(b, desc, lane, i, &row, &col);

      nir_def *outer = row_major ? row : col;
      nir_def *inner = row_major ? col : row;
      nir_def *idx = nir_iadd(b, nir_imul(b, nir_u2uN(b, outer, idx_bits), stride),
                              nir_u2uN(b, inner, idx_bits));
      nir_deref_instr *e = nir_build_deref_ptr_as_array(b, deref, idx);

      if (width > 1) {
         e = nir_build_deref_cast_with_alignment(
            b, &e->def, e->modes, glsl_vector_type(desc.element_type, width),
            width * elem_size, width * elem_size, 0);
      }

      if (is_load) {
         nir_def *v = nir_load_deref(b, e);
         for (unsigned c = 0; c < width; c++)
            elems[i + c] = nir_channel(b, v, c);
      } else {
         nir_store_deref(b, e, nir_vec(b, &elems[i], width),
                         BITFIELD_MASK(width));
      }
   }

   if (is_load)
      store_cmat_src(b, intr->src[0], nir_vec(b, elems, length));
   else if (half_block)
      nir_pop_if(b, NULL);

   nir_instr_remove(&intr->instr);
   return true;
}

static nir_def *
convert_elems(nir_builder *b, nir_def *v, enum glsl_base_type src,
              enum glsl_base_type dst, bool sat, nir_rounding_mode rnd)
{
   if (src == dst)
      return v;

   if (src == GLSL_TYPE_FLOAT_E4M3FN || src == GLSL_TYPE_FLOAT_E5M2) {
      bool e4m3 = src == GLSL_TYPE_FLOAT_E4M3FN;
      if (dst == GLSL_TYPE_FLOAT16 || cmat_paired(dst)) {
         v = e4m3 ? nir_e4m3fn2f16_pan(b, v) : nir_e5m22f16_pan(b, v);
         return convert_elems(b, v, GLSL_TYPE_FLOAT16, dst, sat, rnd);
      }
      v = e4m3 ? nir_e4m3fn2f(b, v) : nir_e5m22f(b, v);
      return convert_elems(b, v, GLSL_TYPE_FLOAT, dst, sat, rnd);
   }

   if (src == GLSL_TYPE_BFLOAT16)
      return convert_elems(b, nir_bf2f(b, v), GLSL_TYPE_FLOAT, dst, sat, rnd);

   if (dst == GLSL_TYPE_FLOAT_E4M3FN || dst == GLSL_TYPE_FLOAT_E5M2) {
      bool e4m3 = dst == GLSL_TYPE_FLOAT_E4M3FN;
      bool directed = rnd != nir_rounding_mode_undef &&
                      rnd != nir_rounding_mode_rtne;
      if (src == GLSL_TYPE_FLOAT16 && !directed) {
         if (e4m3)
            return sat ? nir_f162e4m3fn_sat_pan(b, v) : nir_f162e4m3fn_pan(b, v);
         return sat ? nir_f162e5m2_sat_pan(b, v) : nir_f162e5m2_pan(b, v);
      }
      v = convert_elems(b, v, src, GLSL_TYPE_FLOAT, false,
                        nir_rounding_mode_undef);
      v = nir_build_alu1(b, nir_float8_conversion_op(e4m3, sat, rnd), v);
      nir_def_as_alu(v)->fp_math_ctrl |= nir_fp_preserve_signed_zero;
      return v;
   }

   if (dst == GLSL_TYPE_BFLOAT16)
      return nir_f2bf(b, convert_elems(b, v, src, GLSL_TYPE_FLOAT, false,
                                       nir_rounding_mode_undef));

   nir_op op = nir_type_conversion_op(nir_get_nir_type_for_glsl_base_type(src),
                                      nir_get_nir_type_for_glsl_base_type(dst),
                                      nir_rounding_mode_undef);
   return nir_build_alu1(b, op, v);
}

static nir_def *
cmat_word(nir_builder *b, nir_def *v, struct glsl_cmat_description desc,
          unsigned word)
{
   if (cmat_block(desc).elems == 1)
      return nir_channel(b, v, word);

   nir_def *pair = nir_channels(b, v, 0x3 << (2 * word));
   if (pair->bit_size == 8)
      return nir_pack_32_4x8_split(b, nir_channel(b, pair, 0),
                                   nir_channel(b, pair, 1),
                                   nir_imm_intN_t(b, 0, 8),
                                   nir_imm_intN_t(b, 0, 8));
   return nir_pack_32_2x16(b, pair);
}

static nir_def *
cmat_relayout(nir_builder *b, nir_def *v, struct glsl_cmat_description src,
              struct glsl_cmat_description dst)
{
   const unsigned src_per = cmat_block(src).elems;
   const unsigned src_bw = cmat_block(src).cols;
   const unsigned src_bpr = src.cols / src_bw;
   const unsigned words = get_cmat_length(src) / src_per;
   const unsigned bits = v->bit_size;
   nir_def *lane = nir_load_subgroup_invocation(b);

   nir_def *bcast[NIR_MAX_VEC_COMPONENTS][CMAT_DIM];
   for (unsigned w = 0; w < words; w++) {
      nir_def *word = cmat_word(b, v, src, w);
      for (unsigned k = 0; k < CMAT_DIM; k++)
         bcast[w][k] = nir_quad_broadcast(b, word, nir_imm_int(b, k));
   }

   nir_def *elems[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < get_cmat_length(dst); i++) {
      nir_def *row, *col;
      cmat_coord(b, dst, lane, i, &row, &col);

      nir_def *blk = nir_iadd(b, nir_imul_imm(b, nir_ushr_imm(b, row, 2), src_bpr),
                              nir_udiv_imm(b, col, src_bw));
      nir_def *k = nir_umod_imm(b, col, src_bw);
      if (src_per == 2)
         k = nir_ushr_imm(b, k, 1);

      nir_def *val = NULL;
      for (unsigned w = 0; w < words; w++) {
         nir_def *x = bcast[w][CMAT_DIM - 1];
         for (int j = CMAT_DIM - 2; j >= 0; j--)
            x = nir_bcsel(b, nir_ieq_imm(b, k, j), bcast[w][j], x);
         val = val ? nir_bcsel(b, nir_ieq_imm(b, blk, w), x, val) : x;
      }

      if (src_per == 2) {
         nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, col, 1), 0);
         nir_def *sh = nir_bcsel(b, odd, nir_imm_int(b, bits), nir_imm_int(b, 0));
         val = nir_ushr(b, val, sh);
      }
      elems[i] = nir_u2uN(b, val, bits);
   }

   return nir_vec(b, elems, get_cmat_length(dst));
}

static void
cmat_transpose_bytes(nir_builder *b, nir_def **words, unsigned count,
                     unsigned spacing)
{
   nir_def *g = nir_iand_imm(
      b, nir_udiv_imm(b, nir_load_subgroup_invocation(b), spacing),
      CMAT_DIM - 1);
   nir_def *shift[CMAT_DIM], *recv[CMAT_DIM];

   for (unsigned d = 0; d < CMAT_DIM; d++) {
      shift[d] = nir_imul_imm(b, nir_ixor(b, g, nir_imm_int(b, d)), 8);
      nir_def *send = nir_imm_int(b, 0);
      for (unsigned n = 0; n < count; n++) {
         nir_def *byte =
            nir_iand_imm(b, nir_ushr(b, words[n], shift[d]), 0xff);
         send = nir_ior(b, send, nir_ishl_imm(b, byte, 8 * n));
      }
      recv[d] = d ? nir_shuffle_xor(b, send, nir_imm_int(b, d * spacing))
                  : send;
   }

   for (unsigned n = 0; n < count; n++) {
      nir_def *out = nir_imm_int(b, 0);
      for (unsigned d = 0; d < CMAT_DIM; d++) {
         out = nir_ior(b, out,
                       nir_ishl(b, nir_extract_u8_imm(b, recv[d], n), shift[d]));
      }
      words[n] = out;
   }
}

static nir_def *
cmat_quad_transpose(nir_builder *b, nir_def *v,
                    struct glsl_cmat_description desc, bool to_quad)
{
   const unsigned length = get_cmat_length(desc);
   const bool is_b = cmat_layout(desc) == CMAT_LAYOUT_QUAD_B;
   const unsigned groups = is_b ? desc.rows / CMAT_QUAD_K : length / 4;
   const unsigned count = is_b ? desc.cols / CMAT_DIM : 1;
   nir_def *elems[NIR_MAX_VEC_COMPONENTS];

   for (unsigned grp = 0; grp < groups; grp++) {
      for (unsigned base = 0; base < count; base += CMAT_DIM) {
         const unsigned n = MIN2(count - base, CMAT_DIM);
         unsigned from[CMAT_DIM][CMAT_DIM], to[CMAT_DIM][CMAT_DIM];
         nir_def *words[CMAT_DIM];

         for (unsigned w = 0; w < n; w++) {
            nir_def *bytes[CMAT_DIM];
            for (unsigned j = 0; j < CMAT_DIM; j++) {
               const unsigned quad = is_b ? 4 * (grp * count + base + w) + j
                                          : 4 * grp + j;
               const unsigned word = is_b ? (4 * grp + j) * count + base + w
                                          : 4 * grp + j;
               from[w][j] = to_quad ? word : quad;
               to[w][j] = to_quad ? quad : word;
               bytes[j] = nir_channel(b, v, from[w][j]);
            }
            words[w] = nir_pack_32_4x8(b, nir_vec(b, bytes, CMAT_DIM));
         }

         cmat_transpose_bytes(b, words, n, is_b ? CMAT_DIM : 1);

         for (unsigned w = 0; w < n; w++) {
            nir_def *bytes = nir_unpack_32_4x8(b, words[w]);
            for (unsigned j = 0; j < CMAT_DIM; j++)
               elems[to[w][j]] = nir_channel(b, bytes, j);
         }
      }
   }

   return nir_vec(b, elems, length);
}

static nir_def *
cmat_convert(nir_builder *b, nir_def *v, struct glsl_cmat_description src,
             struct glsl_cmat_description dst, bool convert, bool sat,
             unsigned signed_mask, nir_rounding_mode rnd)
{
   enum glsl_base_type src_type = glsl_apply_signedness_to_base_type(
      src.element_type, signed_mask & NIR_CMAT_A_SIGNED);
   enum glsl_base_type dst_type = glsl_apply_signedness_to_base_type(
      dst.element_type, signed_mask & NIR_CMAT_RESULT_SIGNED);
   const bool relayout = cmat_layout(src) != cmat_layout(dst);

   if (relayout && cmat_quad(src)) {
      v = cmat_quad_transpose(b, v, src, false);
      src = cmat_word_desc(src);
   }

   struct glsl_cmat_description mid = cmat_quad(dst) ? cmat_word_desc(dst)
                                                       : dst;
   if (relayout && cmat_layout(src) != cmat_layout(mid))
      v = cmat_relayout(b, v, src, mid);

   if (convert)
      v = convert_elems(b, v, src_type, dst_type, sat, rnd);

   if (relayout && cmat_quad(dst))
      v = cmat_quad_transpose(b, v, dst, true);

   return v;
}

static nir_def *
mmul(nir_builder *b, nir_def *a, nir_def *bm, nir_def *c, nir_alu_type src,
     nir_alu_type dst, unsigned submat)
{
   return nir_cmat_muladd_pan(b, a, bm, c, .src_type = src, .dest_type = dst,
                              .flags = submat);
}

static nir_def *
pack_pair(nir_builder *b, nir_def *v, unsigned pair)
{
   return nir_pack_32_2x16(b, nir_channels(b, v, 0x3 << (2 * pair)));
}

static nir_def *
lower_muladd_int8(nir_builder *b, nir_def *a, nir_def *bm, nir_def *c,
                  struct glsl_cmat_description a_desc,
                  struct glsl_cmat_description b_desc, nir_alu_type type)
{
   const unsigned kq = a_desc.cols / CMAT_DIM;
   const unsigned nq = b_desc.cols / CMAT_DIM;
   nir_def *zero = nir_imm_intN_t(b, 0, 8);
   nir_def *d[NIR_MAX_VEC_COMPONENTS];

   for (unsigned n = 0; n < nq; n++) {
      d[n] = nir_channel(b, c, n);
      for (unsigned k = 0; k < kq; k += 4) {
         nir_def *ab[4], *bb[4];
         for (unsigned p = 0; p < 4; p++) {
            ab[p] = k + p < kq ? nir_channel(b, a, k + p) : zero;
            bb[p] = k + p < kq ? nir_channel(b, bm, (k + p) * nq + n) : zero;
         }
         d[n] = mmul(b, nir_pack_32_4x8(b, nir_vec(b, ab, 4)),
                     nir_pack_32_4x8(b, nir_vec(b, bb, 4)), d[n], type,
                     nir_type_int32, 0);
      }
   }

   return nir_vec(b, d, nq);
}

static nir_def *
lower_muladd_int8_quad(nir_builder *b, nir_def *a, nir_def *bm, nir_def *c,
                       struct glsl_cmat_description a_desc,
                       struct glsl_cmat_description b_desc, nir_alu_type type)
{
   const unsigned kq = a_desc.cols / CMAT_QUAD_K;
   const unsigned nq = b_desc.cols / CMAT_DIM;
   nir_def *d[NIR_MAX_VEC_COMPONENTS];

   for (unsigned n = 0; n < nq; n++) {
      d[n] = nir_channel(b, c, n);
      for (unsigned k = 0; k < kq; k++) {
         d[n] = mmul(b, nir_pack_32_4x8(b, nir_channels(b, a, 0xf << (4 * k))),
                     nir_pack_32_4x8(b, nir_channels(b, bm, 0xf << (4 * (k * nq + n)))),
                     d[n], type, nir_type_int32, 0);
      }
   }

   return nir_vec(b, d, nq);
}

static nir_def *
lower_muladd_f32(nir_builder *b, nir_def *a, nir_def *bm, nir_def *c,
                 struct glsl_cmat_description a_desc,
                 struct glsl_cmat_description b_desc)
{
   const unsigned kq = a_desc.cols / CMAT_DIM;
   const unsigned nq = b_desc.cols / CMAT_DIM;
   nir_def *d[NIR_MAX_VEC_COMPONENTS];

   for (unsigned n = 0; n < nq; n++) {
      d[n] = nir_channel(b, c, n);
      for (unsigned k = 0; k < kq; k++) {
         d[n] = mmul(b, nir_channel(b, a, k), nir_channel(b, bm, k * nq + n),
                     d[n], nir_type_float32, nir_type_float32, 0);
      }
   }

   return nir_vec(b, d, nq);
}

static nir_def *
lower_muladd_f16(nir_builder *b, nir_def *a, nir_def *bm, nir_def *c,
                 struct glsl_cmat_description a_desc,
                 struct glsl_cmat_description b_desc, bool f16_acc)
{
   const unsigned kq = a_desc.cols / CMAT_DIM;
   const unsigned a_halves = cmat_block(a_desc).cols / CMAT_DIM;
   const unsigned b_halves = cmat_block(b_desc).cols / CMAT_DIM;
   const unsigned b_bpr = b_desc.cols / cmat_block(b_desc).cols;
   nir_def *d[NIR_MAX_VEC_COMPONENTS];

   if (f16_acc) {
      for (unsigned bc = 0; bc < b_bpr; bc++) {
         nir_def *acc = pack_pair(b, c, bc);
         for (unsigned k = 0; k < kq; k++) {
            acc = mmul(b, pack_pair(b, a, k / a_halves),
                       pack_pair(b, bm, k * b_bpr + bc), acc, nir_type_float16,
                       nir_type_float16, k % a_halves);
         }
         nir_def *halves = nir_unpack_32_2x16(b, acc);
         d[2 * bc] = nir_channel(b, halves, 0);
         d[2 * bc + 1] = nir_channel(b, halves, 1);
      }
      return nir_vec(b, d, 2 * b_bpr);
   }

   const unsigned nq = b_desc.cols / CMAT_DIM;
   for (unsigned n = 0; n < nq; n++) {
      const unsigned bc = n / b_halves;
      d[n] = nir_channel(b, c, n);
      for (unsigned k = 0; k < kq; k++) {
         d[n] = mmul(b, pack_pair(b, a, k / a_halves),
                     pack_pair(b, bm, k * b_bpr + bc), d[n], nir_type_float16,
                     nir_type_float32, (k % a_halves) | ((n % b_halves) << 1));
      }
   }

   return nir_vec(b, d, nq);
}

static bool
lower_cmat_muladd(nir_builder *b, nir_intrinsic_instr *intr)
{
   const struct glsl_cmat_description a_desc = cmat_src_desc(intr->src[1]);
   const struct glsl_cmat_description b_desc = cmat_src_desc(intr->src[2]);
   const struct glsl_cmat_description c_desc = cmat_src_desc(intr->src[3]);
   const enum glsl_base_type ab = a_desc.element_type;

   nir_def *a = load_cmat_src(b, intr->src[1]);
   nir_def *bm = load_cmat_src(b, intr->src[2]);
   nir_def *c = load_cmat_src(b, intr->src[3]);
   nir_def *d;

   if (glsl_base_type_bit_size(ab) == 8 && !cmat_paired(ab)) {
      const bool a_signed =
         nir_intrinsic_cmat_signed_mask(intr) & NIR_CMAT_A_SIGNED;
      const nir_alu_type type = a_signed ? nir_type_int8 : nir_type_uint8;
      d = cmat_quad(a_desc)
             ? lower_muladd_int8_quad(b, a, bm, c, a_desc, b_desc, type)
             : lower_muladd_int8(b, a, bm, c, a_desc, b_desc, type);
   } else if (cmat_paired(ab)) {
      a = convert_elems(b, a, ab, GLSL_TYPE_FLOAT16, false,
                        nir_rounding_mode_undef);
      bm = convert_elems(b, bm, ab, GLSL_TYPE_FLOAT16, false,
                         nir_rounding_mode_undef);
      d = lower_muladd_f16(b, a, bm, c, a_desc, b_desc,
                           cmat_paired(c_desc.element_type));
   } else {
      a = convert_elems(b, a, ab, GLSL_TYPE_FLOAT, false,
                        nir_rounding_mode_undef);
      bm = convert_elems(b, bm, ab, GLSL_TYPE_FLOAT, false,
                         nir_rounding_mode_undef);
      d = lower_muladd_f32(b, a, bm, c, a_desc, b_desc);
   }

   store_cmat_src(b, intr->src[0], d);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
lower_cmat_instr(nir_builder *b, nir_instr *instr, struct hash_table *mapping)
{
   if (instr->type == nir_instr_type_deref) {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      return remap_type_in_place(mapping, &deref->type);
   }

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   b->cursor = nir_before_instr(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_cmat_construct: {
      nir_def *r = nir_replicate(b, intr->src[1].ssa,
                                 get_cmat_length(cmat_src_desc(intr->src[0])));
      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_length:
      nir_def_replace(&intr->def,
                      nir_imm_int(b, get_cmat_length(nir_intrinsic_cmat_desc(intr))));
      return true;

   case nir_intrinsic_cmat_extract: {
      nir_def *mat = load_cmat_src(b, intr->src[0]);
      nir_def_replace(&intr->def, nir_vector_extract(b, mat, intr->src[1].ssa));
      return true;
   }

   case nir_intrinsic_cmat_insert: {
      nir_def *mat = load_cmat_src(b, intr->src[2]);
      nir_def *r = nir_vector_insert(b, mat, intr->src[1].ssa, intr->src[3].ssa);
      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_copy:
      nir_build_copy_deref(b, intr->src[0].ssa, intr->src[1].ssa);
      nir_instr_remove(instr);
      return true;

   case nir_intrinsic_cmat_load:
   case nir_intrinsic_cmat_store:
      return lower_cmat_load_store(b, intr);

   case nir_intrinsic_cmat_muladd:
      return lower_cmat_muladd(b, intr);

   case nir_intrinsic_cmat_unary_op:
   case nir_intrinsic_cmat_binary_op:
   case nir_intrinsic_cmat_scalar_op: {
      nir_def *a = load_cmat_src(b, intr->src[1]);
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);

      nir_def *r;
      if (intr->intrinsic == nir_intrinsic_cmat_unary_op) {
         r = nir_build_alu1(b, nir_intrinsic_alu_op(intr), a);
      } else {
         nir_def *y = intr->intrinsic == nir_intrinsic_cmat_binary_op
                         ? load_cmat_src(b, intr->src[2])
                         : intr->src[2].ssa;
         r = nir_build_alu2(b, nir_intrinsic_alu_op(intr), a, y);
      }

      b->fp_math_ctrl = nir_fp_fast_math;
      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_bitcast: {
      nir_def *r = cmat_convert(b, load_cmat_src(b, intr->src[1]),
                                cmat_src_desc(intr->src[1]),
                                cmat_src_desc(intr->src[0]), false, false, 0,
                                nir_rounding_mode_undef);
      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_cmat_convert: {
      b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
      nir_def *r = cmat_convert(b, load_cmat_src(b, intr->src[1]),
                                cmat_src_desc(intr->src[1]),
                                cmat_src_desc(intr->src[0]), true,
                                nir_intrinsic_saturate(intr),
                                nir_intrinsic_cmat_signed_mask(intr),
                                nir_intrinsic_rounding_mode(intr));
      b->fp_math_ctrl = nir_fp_fast_math;
      store_cmat_src(b, intr->src[0], r);
      nir_instr_remove(instr);
      return true;
   }

   default:
      return false;
   }
}

static bool
lower_cmat_impl(nir_function_impl *impl, struct hash_table *mapping)
{
   bool progress = false;

   nir_foreach_function_temp_variable(var, impl) {
      if (remap_type_in_place(mapping, &var->type))
         progress = true;
   }

   nir_builder b = nir_builder_create(impl);
   nir_foreach_block_reverse_safe(block, impl) {
      nir_foreach_instr_reverse_safe(instr, block) {
         if (lower_cmat_instr(&b, instr, mapping))
            progress = true;
      }
   }

   return nir_progress(progress, impl, nir_metadata_none);
}

static enum glsl_base_type
cmat_acc_compute_type(enum glsl_base_type type)
{
   switch (type) {
   case GLSL_TYPE_BFLOAT16:
      return GLSL_TYPE_FLOAT;
   case GLSL_TYPE_FLOAT_E4M3FN:
   case GLSL_TYPE_FLOAT_E5M2:
      return GLSL_TYPE_FLOAT16;
   default:
      return type;
   }
}

static bool
widen_cmat_acc(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_cmat_muladd)
      return false;

   struct glsl_cmat_description desc = cmat_src_desc(intr->src[3]);
   enum glsl_base_type wide = cmat_acc_compute_type(desc.element_type);
   if (wide == desc.element_type)
      return false;

   desc.element_type = wide;
   const struct glsl_type *type = glsl_cmat_type(&desc);
   b->cursor = nir_before_instr(&intr->instr);

   nir_deref_instr *c = nir_build_deref_var(
      b, nir_local_variable_create(b->impl, type, "cmat_acc"));
   nir_deref_instr *d = nir_build_deref_var(
      b, nir_local_variable_create(b->impl, type, "cmat_acc"));

   nir_cmat_convert(b, &c->def, intr->src[3].ssa);
   nir_cmat_muladd(b, &d->def, intr->src[1].ssa, intr->src[2].ssa, &c->def,
                   .saturate = nir_intrinsic_saturate(intr),
                   .cmat_signed_mask = nir_intrinsic_cmat_signed_mask(intr));
   nir_cmat_convert(b, intr->src[0].ssa, &d->def);
   nir_instr_remove(&intr->instr);
   return true;
}

struct cmat_dims {
   unsigned k_mask;
   unsigned n_mask;
};

static void
gather_cmat_type(const struct glsl_type *type, struct cmat_dims *dims)
{
   type = glsl_without_array(type);
   if (!glsl_type_is_cmat(type))
      return;

   const struct glsl_cmat_description *desc = glsl_get_cmat_description(type);
   if (desc->use == GLSL_CMAT_USE_A)
      dims->k_mask |= desc->cols;
   else
      dims->n_mask |= desc->cols;

   if (desc->use == GLSL_CMAT_USE_B)
      dims->k_mask |= desc->rows;
}

static unsigned
cmat_granularity(unsigned mask, unsigned max)
{
   return mask ? MIN2(1u << (ffs(mask) - 1), max) : CMAT_DIM;
}

static struct cmat_dims
gather_cmat_dims(nir_shader *nir)
{
   struct cmat_dims dims = { 0 };

   nir_foreach_variable_with_modes(var, nir, nir_var_shader_temp)
      gather_cmat_type(var->type, &dims);

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_function_temp_variable(var, impl)
         gather_cmat_type(var->type, &dims);
   }

   return dims;
}

bool
panvk_nir_lower_cooperative_matrix(nir_shader *nir, unsigned subgroup_size)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE ||
       !nir->info.cs.has_cooperative_matrix)
      return false;

   assert(subgroup_size == CMAT_DIM * CMAT_DIM);

   bool progress = nir_shader_intrinsics_pass(nir, widen_cmat_acc,
                                              nir_metadata_control_flow, NULL);

   struct cmat_dims dims = gather_cmat_dims(nir);
   const unsigned n_gran =
      cmat_granularity(dims.n_mask, NIR_MAX_VEC_COMPONENTS * CMAT_DIM);

   struct nir_lower_coopmat_args args = {
      .m_gran = CMAT_DIM,
      .n_gran = n_gran,
      .k_gran = cmat_granularity(
         dims.k_mask, NIR_MAX_VEC_COMPONENTS * CMAT_DIM * CMAT_DIM / n_gran),
   };
   if (nir_lower_cooperative_matrix_flexible_dimensions(nir, &args)) {
      progress = true;
      NIR_PASS(_, nir, nir_opt_deref);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_remove_dead_variables,
               nir_var_function_temp | nir_var_shader_temp, NULL);
   }

   struct hash_table *mapping = _mesa_pointer_hash_table_create(NULL);

   nir_foreach_variable_with_modes(var, nir, nir_var_shader_temp) {
      if (remap_type_in_place(mapping, &var->type))
         progress = true;
   }

   nir_foreach_function_impl(impl, nir) {
      if (lower_cmat_impl(impl, mapping))
         progress = true;
   }

   _mesa_hash_table_destroy(mapping, NULL);
   return progress;
}
