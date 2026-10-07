/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "compiler/spirv/spirv.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "util/bitset.h"
#include "util/hash_table.h"

#include "bvh/panvk_bvh.h"
#include "panvk_ray_tracing.h"
#include "panvk_shader.h"

#define RQ_COMMITTED   0x24
#define RQ_CANDIDATE   0x64
#define RQ_REC_T       0x00
#define RQ_REC_BARY    0x04
#define RQ_REC_PRIM    0x0c
#define RQ_REC_KIND    0x10
#define RQ_REC_DWORDS  7

#define RQ_STATUS_AABB 2
#define RQ_STATUS_HIT  5

#define RQ_KIND_COMMIT    0x80000000u
#define RQ_KIND_GENERATED 0xe0000000u

#define RQ_FLAGS_MASK      0x7f7u
#define RQ_FLAGS_FAST_MASK 0x577u
#define RQ_FLAGS_FAST                                                          \
   (SpvRayFlagsCullNoOpaqueKHRMask | SpvRayFlagsSkipAABBsKHRMask)

enum rq_field {
   rq_field_accel_struct,
   rq_field_status,
   rq_field_started,
   rq_field_terminated,
   rq_field_origin,
   rq_field_direction,
   rq_field_tmin,
   rq_field_flags,
   rq_field_committed_t,
   rq_field_committed_type,
   rq_field_commit_t,
   rq_field_cand_lo,
   rq_field_cand_hi,
   rq_field_comm_lo,
   rq_field_comm_hi,
   rq_field_count,
};

struct rq_var {
   nir_variable *var;
   uint32_t slot;
   bool has_proceed;
   bool fast;
   bool candidates;
   bool resume;
   bool records;
   unsigned eager;
};

struct lower_rq_ctx {
   struct hash_table *vars;
   nir_def *lane_index;
   nir_def *lane_count;
   uint32_t slot_count;
};

static const struct glsl_type *
rq_type(void)
{
   struct glsl_struct_field fields[rq_field_count];

#define FIELD(field_name, field_type)                                          \
   fields[rq_field_##field_name] = (struct glsl_struct_field){                 \
      .type = field_type,                                                      \
      .name = #field_name,                                                     \
   }

   FIELD(accel_struct, glsl_uint64_t_type());
   FIELD(status, glsl_uint_type());
   FIELD(started, glsl_bool_type());
   FIELD(terminated, glsl_bool_type());
   FIELD(origin, glsl_vec_type(3));
   FIELD(direction, glsl_vec_type(3));
   FIELD(tmin, glsl_float_type());
   FIELD(flags, glsl_uint_type());
   FIELD(committed_t, glsl_uint_type());
   FIELD(committed_type, glsl_uint_type());
   FIELD(commit_t, glsl_uint_type());
   FIELD(cand_lo, glsl_vec_type(4));
   FIELD(cand_hi, glsl_vec_type(3));
   FIELD(comm_lo, glsl_vec_type(4));
   FIELD(comm_hi, glsl_vec_type(3));

#undef FIELD

   return glsl_struct_type(fields, rq_field_count, "panvk_ray_query", false);
}

static struct rq_var *
rq_var_get(struct lower_rq_ctx *ctx, nir_intrinsic_instr *intr)
{
   struct hash_entry *entry =
      _mesa_hash_table_search(ctx->vars, nir_intrinsic_get_var(intr, 0));
   assert(entry);
   return entry->data;
}

static nir_deref_instr *
rq_deref(nir_builder *b, struct lower_rq_ctx *ctx, nir_def *def,
         nir_def **slot)
{
   nir_deref_path path;
   nir_deref_path_init(&path, nir_def_as_deref(def), NULL);
   assert(path.path[0]->deref_type == nir_deref_type_var);

   struct hash_entry *entry =
      _mesa_hash_table_search(ctx->vars, path.path[0]->var);
   assert(entry);
   const struct rq_var *rq = entry->data;

   nir_deref_instr *deref = nir_build_deref_var(b, rq->var);
   nir_def *index = nir_imm_int(b, 0);

   for (nir_deref_instr **p = &path.path[1]; *p; p++) {
      assert((*p)->deref_type == nir_deref_type_array);
      unsigned length = glsl_get_length(nir_deref_instr_parent(*p)->type);
      index = nir_iadd(b, nir_imul_imm(b, index, length),
                       nir_u2u32(b, (*p)->arr.index.ssa));
      deref = nir_build_deref_array(b, deref, (*p)->arr.index.ssa);
   }

   nir_deref_path_finish(&path);

   if (slot)
      *slot = nir_iadd_imm(b, index, rq->slot);

   return deref;
}

#define rq_load(b, rq, field)                                                  \
   nir_load_deref(b, nir_build_deref_struct(b, rq, rq_field_##field))
#define rq_store(b, rq, field, value)                                          \
   nir_store_deref(b, nir_build_deref_struct(b, rq, rq_field_##field), value,  \
                   BITFIELD_MASK((value)->num_components))

static nir_def *
rq_state_addr(nir_builder *b, struct lower_rq_ctx *ctx, nir_def *slot)
{
   nir_def *index =
      nir_iadd(b, nir_imul(b, slot, ctx->lane_count), ctx->lane_index);

   return nir_iadd(b, load_sysval(b, common, 64, ray_query_state),
                   nir_u2u64(b, nir_ishl_imm(b, index, 7)));
}

static nir_def *
load_state(nir_builder *b, nir_def *state, unsigned offset, unsigned comps)
{
   return nir_load_global(b, comps, 32, nir_iadd_imm(b, state, offset),
                          .align_mul = 4);
}

static void
store_state(nir_builder *b, nir_def *state, unsigned offset, nir_def *value)
{
   nir_store_global(b, value, nir_iadd_imm(b, state, offset), .align_mul = 4);
}

static nir_def *
load_as(nir_builder *b, nir_def *addr, unsigned comps, unsigned bit_size)
{
   return nir_load_global(b, comps, bit_size, addr, .align_mul = 4,
                          .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static nir_def *
root_node(nir_builder *b, nir_def *as)
{
   return nir_bcsel(b, nir_ine_imm(b, as, 0),
                    nir_iadd_imm(b, as, PANVK_BVH_HEADER_SIZE),
                    nir_imm_int64(b, 0));
}

static nir_def *
status_code(nir_builder *b, nir_def *status)
{
   return nir_ushr_imm(b, status, 29);
}

static nir_def *
status_is_candidate(nir_builder *b, nir_def *status)
{
   return nir_ult_imm(b, nir_iadd_imm(b, status_code(b, status), -1), 2);
}

static nir_def *
status_is_hit(nir_builder *b, nir_def *status)
{
   return nir_ieq_imm(b, nir_iand_imm(b, status, 0xe0000000u),
                      RQ_STATUS_HIT << 29);
}

static void
store_trace_result(nir_builder *b, nir_deref_instr *rq, nir_def *out,
                   const struct rq_var *info, nir_def *state)
{
   rq_store(b, rq, status, nir_channel(b, out, 0));
   rq_store(b, rq, committed_t, nir_channel(b, out, 1));

   if (info->eager) {
      rq_store(b, rq, cand_lo, load_state(b, state, RQ_CANDIDATE, 4));
      rq_store(b, rq, cand_hi, load_state(b, state, RQ_CANDIDATE + 16, 3));
   }
   if (info->eager > 1) {
      rq_store(b, rq, comm_lo, load_state(b, state, RQ_COMMITTED, 4));
      rq_store(b, rq, comm_hi, load_state(b, state, RQ_COMMITTED + 16, 3));
   }
}

static nir_def *
record_load(nir_builder *b, nir_deref_instr *rq, const struct rq_var *info,
            nir_def *state, bool committed, unsigned offset, unsigned comps)
{
   if (info->eager > (committed ? 1 : 0)) {
      nir_def *lo = committed ? rq_load(b, rq, comm_lo) : rq_load(b, rq, cand_lo);
      nir_def *hi = committed ? rq_load(b, rq, comm_hi) : rq_load(b, rq, cand_hi);
      nir_def *c[RQ_REC_DWORDS];
      for (unsigned i = 0; i < RQ_REC_DWORDS; i++)
         c[i] = nir_channel(b, i < 4 ? lo : hi, i % 4);
      return nir_vec(b, &c[offset / 4], comps);
   }

   return load_state(b, state, (committed ? RQ_COMMITTED : RQ_CANDIDATE) + offset,
                     comps);
}

static void
lower_rq_initialize(nir_builder *b, struct lower_rq_ctx *ctx,
                    const struct rq_var *info, nir_intrinsic_instr *intr)
{
   nir_def *slot;
   nir_deref_instr *rq = rq_deref(b, ctx, intr->src[0].ssa, &slot);
   nir_def *as = intr->src[1].ssa;
   nir_def *flags = intr->src[2].ssa;
   nir_def *cull_mask = intr->src[3].ssa;
   nir_def *origin = intr->src[4].ssa;
   nir_def *tmin = intr->src[5].ssa;
   nir_def *direction = intr->src[6].ssa;
   nir_def *tmax = intr->src[7].ssa;

   rq_store(b, rq, accel_struct, as);
   rq_store(b, rq, origin, origin);
   rq_store(b, rq, direction, direction);
   rq_store(b, rq, tmin, tmin);
   rq_store(b, rq, flags, flags);
   rq_store(b, rq, committed_type, nir_imm_int(b, 0));
   rq_store(b, rq, commit_t, tmax);
   rq_store(b, rq, started, nir_imm_false(b));
   rq_store(b, rq, terminated, nir_imm_false(b));

   nir_def *hw_flags =
      info->fast ? nir_ior_imm(b, nir_iand_imm(b, flags, RQ_FLAGS_FAST_MASK),
                               RQ_FLAGS_FAST)
                 : nir_iand_imm(b, flags, RQ_FLAGS_MASK);
   hw_flags = nir_ior(b, hw_flags,
                      nir_ishl_imm(b, nir_iand_imm(b, cull_mask, 0xff), 24));

   nir_def *state = info->records || info->resume
                       ? rq_state_addr(b, ctx, slot)
                       : nir_imm_int64(b, 0);

   nir_def *out =
      nir_rt_trace_begin_pan(b, state, root_node(b, as), origin, direction,
                             tmin, tmax, hw_flags, nir_imm_int(b, 0));
   store_trace_result(b, rq, out, info, state);
}

static nir_def *
lower_rq_proceed(nir_builder *b, struct lower_rq_ctx *ctx,
                 const struct rq_var *info, nir_intrinsic_instr *intr)
{
   if (!info->candidates)
      return nir_imm_false(b);

   nir_def *slot;
   nir_deref_instr *rq = rq_deref(b, ctx, intr->src[0].ssa, &slot);

   if (info->resume) {
      nir_push_if(b, rq_load(b, rq, started));
      {
         nir_def *terminated = rq_load(b, rq, terminated);
         nir_def *status = rq_load(b, rq, status);

         nir_push_if(b, nir_iand(b, nir_inot(b, terminated),
                                 status_is_candidate(b, status)));
         {
            nir_def *state = rq_state_addr(b, ctx, slot);
            nir_def *candidate =
               record_load(b, rq, info, state, false, RQ_REC_PRIM, 2);
            if (info->eager > 1) {
               store_state(b, state, RQ_COMMITTED, rq_load(b, rq, comm_lo));
               store_state(b, state, RQ_COMMITTED + 16, rq_load(b, rq, comm_hi));
            }
            nir_def *out = nir_rt_trace_resume_pan(
               b, state, root_node(b, rq_load(b, rq, accel_struct)),
               candidate);
            store_trace_result(b, rq, out, info, state);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);

      rq_store(b, rq, started, nir_imm_true(b));
   }

   return nir_iand(b, nir_inot(b, rq_load(b, rq, terminated)),
                   status_is_candidate(b, rq_load(b, rq, status)));
}

static nir_def *
commit_candidate(nir_builder *b, nir_deref_instr *rq, const struct rq_var *info,
                 nir_def *state, nir_def *t, uint32_t kind_bits)
{
   nir_def *lo = record_load(b, rq, info, state, false, 0, 4);
   nir_def *hi = record_load(b, rq, info, state, false, 16, RQ_REC_DWORDS - 4);
   nir_def *comps[RQ_REC_DWORDS];

   for (unsigned i = 0; i < RQ_REC_DWORDS; i++)
      comps[i] = nir_channel(b, i < 4 ? lo : hi, i % 4);

   if (t)
      comps[RQ_REC_T / 4] = t;

   comps[RQ_REC_KIND / 4] =
      kind_bits == RQ_KIND_GENERATED
         ? nir_ior_imm(b, nir_iand_imm(b, comps[RQ_REC_KIND / 4], 0x1fffffff),
                       kind_bits)
         : nir_ior_imm(b, comps[RQ_REC_KIND / 4], kind_bits);

   if (info->eager > 1) {
      rq_store(b, rq, comm_lo, nir_vec(b, comps, 4));
      rq_store(b, rq, comm_hi, nir_vec(b, &comps[4], 3));
   } else {
      store_state(b, state, RQ_COMMITTED, nir_vec(b, comps, 4));
      store_state(b, state, RQ_COMMITTED + 16, nir_vec(b, &comps[4], 3));
   }

   return comps[RQ_REC_T / 4];
}

static void
lower_rq_commit(nir_builder *b, struct lower_rq_ctx *ctx,
                const struct rq_var *info, nir_intrinsic_instr *intr,
                bool generated)
{
   nir_def *slot;
   nir_deref_instr *rq = rq_deref(b, ctx, intr->src[0].ssa, &slot);
   nir_def *t = generated ? intr->src[1].ssa : NULL;

   t = commit_candidate(b, rq, info, rq_state_addr(b, ctx, slot), t,
                        generated ? RQ_KIND_GENERATED : RQ_KIND_COMMIT);

   rq_store(b, rq, committed_t, t);
   rq_store(b, rq, commit_t, t);
   rq_store(b, rq, committed_type, nir_imm_int(b, generated ? 2 : 1));
}

static void
lower_rq_terminate(nir_builder *b, struct lower_rq_ctx *ctx,
                   nir_intrinsic_instr *intr)
{
   nir_deref_instr *rq = rq_deref(b, ctx, intr->src[0].ssa, NULL);

   rq_store(b, rq, terminated, nir_imm_true(b));
}

static nir_def *
committed_type(nir_builder *b, nir_deref_instr *rq, const struct rq_var *info)
{
   nir_def *status = rq_load(b, rq, status);

   if (!info->candidates)
      return nir_b2i32(b, status_is_hit(b, status));

   nir_def *done =
      nir_bcsel(b, status_is_hit(b, status), nir_imm_int(b, 1),
                nir_bcsel(b, nir_ieq_imm(b, status_code(b, status), 0),
                          nir_imm_int(b, 0), nir_imm_int(b, 2)));
   nir_def *pending =
      nir_bcsel(b, nir_ine(b, rq_load(b, rq, committed_t),
                           rq_load(b, rq, commit_t)),
                nir_imm_int(b, 1), rq_load(b, rq, committed_type));

   return nir_bcsel(b, status_is_candidate(b, status), pending, done);
}

static nir_def *
instance_leaf(nir_builder *b, nir_def *as, nir_def *kind)
{
   nir_def *node_count =
      load_as(b, nir_iadd_imm(b, as, offsetof(struct panvk_bvh_header, node_count)), 1, 32);
   nir_def *index = nir_iadd(b, node_count, nir_iand_imm(b, kind, 0xffffff));

   return nir_iadd(b, nir_iadd_imm(b, as, PANVK_BVH_HEADER_SIZE),
                   nir_u2u64(b, nir_imul_imm(b, index, PANVK_BVH_NODE_SIZE)));
}

static nir_def *
instance_extra(nir_builder *b, nir_def *as, nir_def *kind, unsigned offset)
{
   nir_def *node_count =
      load_as(b, nir_iadd_imm(b, as, offsetof(struct panvk_bvh_header, node_count)), 1, 32);
   nir_def *prim_count =
      load_as(b, nir_iadd_imm(b, as, offsetof(struct panvk_bvh_header, primitive_count)), 1, 32);
   nir_def *extra =
      nir_iadd(b, nir_iadd_imm(b, as, PANVK_BVH_HEADER_SIZE),
               nir_u2u64(b, nir_imul_imm(b, nir_iadd(b, node_count, prim_count),
                                        PANVK_BVH_NODE_SIZE)));
   nir_def *entry =
      nir_imul_imm(b, nir_iand_imm(b, kind, 0xffffff),
                   sizeof(struct panvk_bvh_instance_extra));

   return load_as(b, nir_iadd_imm(b, nir_iadd(b, extra, nir_u2u64(b, entry)), offset), 1, 32);
}

static nir_def *
primitive_leaf_word(nir_builder *b, nir_def *as, nir_def *kind, nir_def *prim,
                    unsigned word)
{
   nir_def *inst = instance_leaf(b, as, kind);
   nir_def *blas = load_as(b, nir_iadd_imm(b, inst, offsetof(struct panvk_bvh_instance_leaf, blas_ptr)), 1, 64);
   nir_def *nodes = nir_iand_imm(b, blas, ~(PANVK_BVH_INSTANCE_AABBS | 0x3full));
   nir_def *leaf =
      nir_iadd(b, nodes, nir_u2u64(b, nir_imul_imm(b, nir_ushr_imm(b, prim, 2),
                                                  PANVK_BVH_NODE_SIZE)));
   nir_def *offset =
      nir_isub(b, nir_imm_int(b, word * 4),
               nir_imul_imm(b, nir_iand_imm(b, prim, 3), 4));

   return load_as(b, nir_iadd(b, leaf, nir_u2u64(b, offset)), 1, 32);
}

static nir_def *
world_to_object_row(nir_builder *b, nir_def *inst, unsigned row)
{
   return load_as(b, nir_iadd_imm(b, inst, row * 16), 4, 32);
}

static nir_def *
transform_point(nir_builder *b, nir_def *rows[3], nir_def *v, bool point)
{
   uint32_t fp_math_ctrl = b->fp_math_ctrl;
   nir_def *res[3];

   b->fp_math_ctrl = nir_fp_no_fast_math;
   for (unsigned r = 0; r < 3; r++) {
      nir_def *m[4];
      for (unsigned c = 0; c < 4; c++)
         m[c] = nir_channel(b, rows[r], c);

      nir_def *sum = point ? m[3] : nir_imm_float(b, -0.0f);
      sum = nir_ffma(b, m[1], nir_channel(b, v, 1), sum);
      sum = nir_ffma(b, m[0], nir_channel(b, v, 0), sum);
      sum = nir_ffma(b, m[2], nir_channel(b, v, 2), sum);
      res[r] = point ? sum : nir_ffma(b, m[3], nir_imm_float(b, 0.0f), sum);
   }
   b->fp_math_ctrl = fp_math_ctrl;

   return nir_vec(b, res, 3);
}

static nir_def *
object_to_world_column(nir_builder *b, nir_def *rows[3], unsigned column)
{
   nir_def *m[3][3];
   for (unsigned r = 0; r < 3; r++)
      for (unsigned c = 0; c < 3; c++)
         m[r][c] = nir_channel(b, rows[r], c);

   nir_def *cof[3][3];
   for (unsigned r = 0; r < 3; r++) {
      for (unsigned c = 0; c < 3; c++) {
         unsigned r0 = (r + 1) % 3, r1 = (r + 2) % 3;
         unsigned c0 = (c + 1) % 3, c1 = (c + 2) % 3;
         cof[r][c] = nir_fsub(b, nir_fmul(b, m[r0][c0], m[r1][c1]),
                              nir_fmul(b, m[r0][c1], m[r1][c0]));
      }
   }

   nir_def *det = nir_fadd(b, nir_fadd(b, nir_fmul(b, m[0][0], cof[0][0]),
                                       nir_fmul(b, m[0][1], cof[0][1])),
                           nir_fmul(b, m[0][2], cof[0][2]));
   nir_def *inv_det = nir_frcp(b, det);

   nir_def *inv[3][3];
   for (unsigned r = 0; r < 3; r++)
      for (unsigned c = 0; c < 3; c++)
         inv[r][c] = nir_fmul(b, cof[c][r], inv_det);

   nir_def *res[3];
   if (column < 3) {
      for (unsigned r = 0; r < 3; r++)
         res[r] = inv[r][column];
   } else {
      for (unsigned r = 0; r < 3; r++) {
         nir_def *t = nir_fadd(b, nir_fadd(b, nir_fmul(b, inv[r][0], nir_channel(b, rows[0], 3)),
                                           nir_fmul(b, inv[r][1], nir_channel(b, rows[1], 3))),
                               nir_fmul(b, inv[r][2], nir_channel(b, rows[2], 3)));
         res[r] = nir_fneg(b, t);
      }
   }

   return nir_vec(b, res, 3);
}


static nir_def *
lower_rq_load(nir_builder *b, struct lower_rq_ctx *ctx,
              const struct rq_var *info, nir_intrinsic_instr *intr)
{
   nir_def *slot;
   nir_deref_instr *rq = rq_deref(b, ctx, intr->src[0].ssa, &slot);
   bool committed = nir_intrinsic_committed(intr);
   nir_ray_query_value value = nir_intrinsic_ray_query_value(intr);

   switch (value) {
   case nir_ray_query_value_tmin:
      return rq_load(b, rq, tmin);
   case nir_ray_query_value_flags:
      return rq_load(b, rq, flags);
   case nir_ray_query_value_world_ray_origin:
      return rq_load(b, rq, origin);
   case nir_ray_query_value_world_ray_direction:
      return rq_load(b, rq, direction);
   default:
      break;
   }

   if (value == nir_ray_query_value_intersection_type) {
      if (committed)
         return committed_type(b, rq, info);

      return nir_b2i32(b, nir_ieq_imm(b, status_code(b, rq_load(b, rq, status)),
                                      RQ_STATUS_AABB));
   }

   if (value == nir_ray_query_value_intersection_t && committed)
      return rq_load(b, rq, committed_t);

   nir_def *state = rq_state_addr(b, ctx, slot);
   nir_def *kind = record_load(b, rq, info, state, committed, RQ_REC_KIND, 1);

   switch (value) {
   case nir_ray_query_value_intersection_t:
      return record_load(b, rq, info, state, committed, RQ_REC_T, 1);
   case nir_ray_query_value_intersection_barycentrics:
      return record_load(b, rq, info, state, committed, RQ_REC_BARY, 2);
   case nir_ray_query_value_intersection_front_face:
      return nir_i2b(b, nir_ubitfield_extract_imm(b, kind, 24, 1));
   default:
      break;
   }

   nir_def *as = rq_load(b, rq, accel_struct);

   switch (value) {
   case nir_ray_query_value_intersection_instance_custom_index:
      return nir_iand_imm(b, instance_extra(b, as, kind, offsetof(struct panvk_bvh_instance_extra, custom_index)), 0xffffff);
   case nir_ray_query_value_intersection_instance_id:
      return instance_extra(b, as, kind, offsetof(struct panvk_bvh_instance_extra, instance_id));
   case nir_ray_query_value_intersection_instance_sbt_index: {
      nir_def *inst = instance_leaf(b, as, kind);
      return nir_iand_imm(b, load_as(b, nir_iadd_imm(b, inst, offsetof(struct panvk_bvh_instance_leaf, flags_sbt_offset)), 1, 32), 0xffffff);
   }
   case nir_ray_query_value_intersection_primitive_index:
      return primitive_leaf_word(b, as, kind, record_load(b, rq, info, state, committed, RQ_REC_PRIM, 1), 13);
   case nir_ray_query_value_intersection_geometry_index:
      return nir_iand_imm(b, primitive_leaf_word(b, as, kind, record_load(b, rq, info, state, committed, RQ_REC_PRIM, 1), 15), 0xffffff);
   case nir_ray_query_value_intersection_candidate_aabb_opaque: {
      nir_def *inst = instance_leaf(b, as, kind);
      nir_def *inst_flags = nir_ushr_imm(b, load_as(b, nir_iadd_imm(b, inst, offsetof(struct panvk_bvh_instance_leaf, flags_sbt_offset)), 1, 32), 24);
      nir_def *leaf_flags = nir_ushr_imm(b, primitive_leaf_word(b, as, kind, record_load(b, rq, info, state, committed, RQ_REC_PRIM, 1), 15), 24);
      nir_def *ray_flags = rq_load(b, rq, flags);
      nir_def *opaque = nir_i2b(b, nir_iand_imm(b, leaf_flags, PANVK_BVH_LEAF_OPAQUE));
      opaque = nir_bcsel(b, nir_test_mask(b, inst_flags, VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR), nir_imm_true(b), opaque);
      opaque = nir_bcsel(b, nir_test_mask(b, inst_flags, VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR), nir_imm_false(b), opaque);
      opaque = nir_bcsel(b, nir_test_mask(b, ray_flags, SpvRayFlagsNoOpaqueKHRMask), nir_imm_false(b), opaque);
      return nir_bcsel(b, nir_test_mask(b, ray_flags, SpvRayFlagsOpaqueKHRMask), nir_imm_true(b), opaque);
   }
   default:
      break;
   }

   nir_def *inst = instance_leaf(b, as, kind);
   nir_def *rows[3];
   for (unsigned r = 0; r < 3; r++)
      rows[r] = world_to_object_row(b, inst, r);

   switch (value) {
   case nir_ray_query_value_intersection_object_ray_origin:
      return transform_point(b, rows, rq_load(b, rq, origin), true);
   case nir_ray_query_value_intersection_object_ray_direction:
      return transform_point(b, rows, rq_load(b, rq, direction), false);
   case nir_ray_query_value_intersection_world_to_object: {
      unsigned column = nir_intrinsic_column(intr);
      return nir_vec3(b, nir_channel(b, rows[0], column),
                      nir_channel(b, rows[1], column),
                      nir_channel(b, rows[2], column));
   }
   case nir_ray_query_value_intersection_object_to_world:
      return object_to_world_column(b, rows, nir_intrinsic_column(intr));
   default:
      UNREACHABLE("Unsupported ray query value");
   }
}

static void
create_rq_var(nir_shader *nir, nir_function_impl *impl, nir_variable *var,
              struct lower_rq_ctx *ctx)
{
   struct rq_var *rq = rzalloc(ctx->vars, struct rq_var);
   const struct glsl_type *type = glsl_type_wrap_in_arrays(rq_type(), var->type);

   rq->var = impl ? nir_local_variable_create(impl, type, "ray_query")
                  : nir_variable_create(nir, nir_var_shader_temp, type, "ray_query");
   rq->fast = true;
   rq->resume = glsl_type_is_array(var->type);

   _mesa_hash_table_insert(ctx->vars, var, rq);
}

static bool
def_is_proceed(nir_def *def, nir_def *proceed, bool *negated)
{
   *negated = false;
   while (def != proceed) {
      if (!nir_def_is_alu(def))
         return false;

      nir_alu_instr *alu = nir_def_as_alu(def);
      if (alu->op == nir_op_inot)
         *negated = !*negated;
      else if (alu->op != nir_op_mov)
         return false;

      def = alu->src[0].src.ssa;
   }

   return true;
}

static bool
cf_list_is_jump(struct exec_list *list, nir_jump_type type)
{
   if (!exec_list_is_singular(list))
      return false;

   nir_cf_node *node = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   nir_block *block = nir_cf_node_as_block(node);

   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_jump ||
          nir_instr_as_jump(instr)->type != type)
         return false;
   }

   return type == nir_jump_continue || nir_block_ends_in_jump(block);
}

static bool
proceed_in_empty_loop(nir_intrinsic_instr *proceed)
{
   nir_cf_node *parent = proceed->instr.block->cf_node.parent;
   if (!parent || parent->type != nir_cf_node_loop)
      return false;

   nir_loop *loop = nir_cf_node_as_loop(parent);
   if (nir_loop_has_continue_construct(loop))
      return false;

   bool exits = false;
   foreach_list_typed(nir_cf_node, node, node, &loop->body) {
      if (node->type == nir_cf_node_block) {
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            bool negated;
            if (instr == &proceed->instr ||
                instr->type == nir_instr_type_deref ||
                instr->type == nir_instr_type_load_const)
               continue;
            if (instr->type == nir_instr_type_alu &&
                def_is_proceed(&nir_instr_as_alu(instr)->def, &proceed->def,
                               &negated))
               continue;
            if (instr->type == nir_instr_type_jump &&
                nir_instr_as_jump(instr)->type == nir_jump_continue)
               continue;
            return false;
         }
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         bool negated;
         if (!def_is_proceed(nif->condition.ssa, &proceed->def, &negated))
            return false;

         struct exec_list *exit = negated ? &nif->then_list : &nif->else_list;
         struct exec_list *stay = negated ? &nif->else_list : &nif->then_list;
         if (!cf_list_is_jump(exit, nir_jump_break) ||
             !cf_list_is_jump(stay, nir_jump_continue))
            return false;

         exits = true;
      } else {
         return false;
      }
   }

   return exits;
}

static int
scan_rq_block(nir_block *block, nir_instr *after, nir_variable *var)
{
   bool active = !after;

   nir_foreach_instr(instr, block) {
      if (!active) {
         active = instr == after;
         continue;
      }

      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if ((intr->intrinsic != nir_intrinsic_rq_initialize &&
           intr->intrinsic != nir_intrinsic_rq_proceed) ||
          nir_intrinsic_get_var(intr, 0) != var)
         continue;

      return intr->intrinsic == nir_intrinsic_rq_proceed ? 1 : -1;
   }

   return 0;
}

static bool
proceed_reaches_proceed(nir_function_impl *impl, nir_intrinsic_instr *from)
{
   nir_variable *var = nir_intrinsic_get_var(from, 0);
   int found = scan_rq_block(from->instr.block, &from->instr, var);
   if (found)
      return found > 0;

   BITSET_WORD *visited = calloc(BITSET_WORDS(impl->num_blocks), sizeof(BITSET_WORD));
   nir_block **stack = malloc(impl->num_blocks * sizeof(*stack));
   unsigned top = 0;
   bool reached = false;
   nir_block *block = from->instr.block;

   while (block) {
      for (unsigned s = 0; s < 2; s++) {
         nir_block *succ = block->successors[s];
         if (succ && !BITSET_TEST(visited, succ->index)) {
            BITSET_SET(visited, succ->index);
            stack[top++] = succ;
         }
      }

      block = NULL;
      while (top && !block && !reached) {
         nir_block *next = stack[--top];
         found = scan_rq_block(next, NULL, var);
         if (found > 0)
            reached = true;
         else if (!found)
            block = next;
      }
   }

   free(visited);
   free(stack);
   return reached;
}

static bool
flags_exclude_candidates(nir_src flags)
{
   if (!nir_src_is_const(flags))
      return false;

   uint32_t value = nir_src_as_uint(flags);
   return (value & (SpvRayFlagsOpaqueKHRMask | SpvRayFlagsCullNoOpaqueKHRMask |
                    SpvRayFlagsSkipTrianglesKHRMask)) &&
          (value & SpvRayFlagsSkipAABBsKHRMask);
}

static void
analyze_rq(nir_function_impl *impl, struct lower_rq_ctx *ctx)
{
   nir_metadata_require(impl, nir_metadata_block_index);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (!nir_intrinsic_is_ray_query(intr->intrinsic))
            continue;

         struct rq_var *rq = rq_var_get(ctx, intr);
         switch (intr->intrinsic) {
         case nir_intrinsic_rq_initialize:
            if (!flags_exclude_candidates(intr->src[2]))
               rq->candidates = true;
            break;
         case nir_intrinsic_rq_proceed:
            rq->has_proceed = true;
            if (!proceed_in_empty_loop(intr))
               rq->fast = false;
            if (proceed_reaches_proceed(impl, intr))
               rq->resume = true;
            break;
         case nir_intrinsic_rq_confirm_intersection:
         case nir_intrinsic_rq_generate_intersection:
            rq->fast = false;
            rq->records = true;
            break;
         case nir_intrinsic_rq_terminate:
            rq->fast = false;
            break;
         case nir_intrinsic_rq_load: {
            nir_ray_query_value value = nir_intrinsic_ray_query_value(intr);
            bool committed = nir_intrinsic_committed(intr);
            if (value == nir_ray_query_value_tmin ||
                value == nir_ray_query_value_flags ||
                value == nir_ray_query_value_world_ray_origin ||
                value == nir_ray_query_value_world_ray_direction)
               break;
            if (!committed)
               rq->fast = false;
            if (value != nir_ray_query_value_intersection_type &&
                (value != nir_ray_query_value_intersection_t || !committed))
               rq->records = true;
            break;
         }
         default:
            UNREACHABLE("Unsupported ray query intrinsic");
         }
      }
   }
}

uint32_t
panvk_per_arch(nir_lower_ray_queries)(nir_shader *nir)
{
   struct lower_rq_ctx ctx = {
      .vars = _mesa_pointer_hash_table_create(NULL),
   };

   nir_foreach_variable_in_shader(var, nir) {
      if (var->data.ray_query)
         create_rq_var(nir, NULL, var, &ctx);
   }

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_function_temp_variable(var, impl) {
         if (var->data.ray_query)
            create_rq_var(nir, impl, var, &ctx);
      }
   }

   if (!ctx.vars->entries) {
      ralloc_free(ctx.vars);
      return 0;
   }

   nir_foreach_function_impl(impl, nir)
      analyze_rq(impl, &ctx);

   hash_table_foreach(ctx.vars, entry) {
      struct rq_var *rq = entry->data;
      const nir_variable *var = entry->key;

      rq->fast &= rq->has_proceed;
      if (rq->fast)
         rq->candidates = false;
      if (!rq->candidates)
         rq->resume = false;
      if (rq->resume) {
         const char *variant = getenv("PANVK_RQ_VARIANT");
         rq->eager = variant ? atoi(variant) : 0;
      }

      if (rq->records || rq->resume) {
         rq->slot = ctx.slot_count;
         ctx.slot_count += MAX2(glsl_get_aoa_size(var->type), 1);
      }
   }

   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_at(nir_before_impl(impl));

      nir_def *lanes_per_core =
         nir_ishl_imm(&b, nir_iadd_imm(&b, nir_load_warp_max_id_arm(&b), 1), 4);
      ctx.lane_count = nir_imul(&b, nir_load_core_count_arm(&b), lanes_per_core);
      ctx.lane_index =
         nir_iadd(&b, nir_imul(&b, nir_load_core_id(&b), lanes_per_core),
                  nir_iadd(&b, nir_ishl_imm(&b, nir_load_warp_id_arm(&b), 4),
                           nir_load_subgroup_invocation(&b)));

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (!nir_intrinsic_is_ray_query(intr->intrinsic))
               continue;

            const struct rq_var *info = rq_var_get(&ctx, intr);
            b.cursor = nir_before_instr(instr);

            nir_def *def = NULL;
            switch (intr->intrinsic) {
            case nir_intrinsic_rq_initialize:
               lower_rq_initialize(&b, &ctx, info, intr);
               break;
            case nir_intrinsic_rq_proceed:
               def = lower_rq_proceed(&b, &ctx, info, intr);
               break;
            case nir_intrinsic_rq_confirm_intersection:
               lower_rq_commit(&b, &ctx, info, intr, false);
               break;
            case nir_intrinsic_rq_generate_intersection:
               lower_rq_commit(&b, &ctx, info, intr, true);
               break;
            case nir_intrinsic_rq_terminate:
               lower_rq_terminate(&b, &ctx, intr);
               break;
            case nir_intrinsic_rq_load:
               def = lower_rq_load(&b, &ctx, info, intr);
               break;
            default:
               UNREACHABLE("Unsupported ray query intrinsic");
            }

            if (def)
               nir_def_rewrite_uses(&intr->def, def);

            nir_instr_remove(instr);
         }
      }

      nir_progress(true, impl, nir_metadata_none);
   }

   ralloc_free(ctx.vars);
   return ctx.slot_count;
}
