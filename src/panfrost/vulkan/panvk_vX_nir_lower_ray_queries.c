/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#include "compiler/spirv/spirv.h"
#include "nir_builder.h"
#include "nir_deref.h"
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

#define RQ_STATUS_TRIANGLE  1
#define RQ_STATUS_AABB      2
#define RQ_STATUS_HIT       5
#define RQ_STATUS_GENERATED 7

#define RQ_COMMITTED_FROM_RECORD 0xffffffffu

#define RQ_KIND_COMMIT    0x80000000u
#define RQ_KIND_GENERATED 0xe0000000u

enum rq_field {
   rq_field_accel_struct,
   rq_field_status,
   rq_field_pending,
   rq_field_terminated,
   rq_field_origin,
   rq_field_direction,
   rq_field_tmin,
   rq_field_flags,
   rq_field_committed_t,
   rq_field_committed_type,
   rq_field_count,
};

struct rq_var {
   nir_variable *var;
   uint32_t slot;
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
   FIELD(pending, glsl_bool_type());
   FIELD(terminated, glsl_bool_type());
   FIELD(origin, glsl_vec_type(3));
   FIELD(direction, glsl_vec_type(3));
   FIELD(tmin, glsl_float_type());
   FIELD(flags, glsl_uint_type());
   FIELD(committed_t, glsl_float_type());
   FIELD(committed_type, glsl_uint_type());

#undef FIELD

   return glsl_struct_type(fields, rq_field_count, "panvk_ray_query", false);
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

static void
rt_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_NONE,
               .memory_scope = SCOPE_SUBGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
}

static void
store_trace_result(nir_builder *b, nir_deref_instr *rq, nir_def *out)
{
   nir_def *status = nir_channel(b, out, 0);
   nir_def *code = nir_ushr_imm(b, status, 29);
   nir_def *type =
      nir_bcsel(b, nir_ieq_imm(b, code, RQ_STATUS_HIT), nir_imm_int(b, 1),
                nir_bcsel(b, nir_ieq_imm(b, code, 0), nir_imm_int(b, 0),
                          nir_bcsel(b, nir_ult_imm(b, code, RQ_STATUS_AABB + 1),
                                    nir_imm_int(b, RQ_COMMITTED_FROM_RECORD),
                                    nir_imm_int(b, 2))));

   rq_store(b, rq, status, status);
   rq_store(b, rq, committed_t, nir_channel(b, out, 1));
   rq_store(b, rq, committed_type, type);
}

static void
lower_rq_initialize(nir_builder *b, struct lower_rq_ctx *ctx,
                    nir_intrinsic_instr *intr)
{
   nir_def *slot;
   nir_deref_instr *rq = rq_deref(b, ctx, intr->src[0].ssa, &slot);
   nir_def *state = rq_state_addr(b, ctx, slot);
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

   store_state(b, state, RQ_COMMITTED + RQ_REC_KIND, nir_imm_int(b, 0));

   rt_barrier(b);
   nir_def *out = nir_rt_trace_begin_pan(
      b, state, root_node(b, as), origin, direction, tmin, tmax,
      nir_ior(b, flags, nir_ishl_imm(b, nir_iand_imm(b, cull_mask, 0xff), 24)),
      nir_imm_int(b, 0));
   rt_barrier(b);
   store_trace_result(b, rq, out);
   rq_store(b, rq, pending, nir_imm_true(b));
   rq_store(b, rq, terminated, nir_imm_false(b));
}

static nir_def *
status_is_candidate(nir_builder *b, nir_def *status)
{
   nir_def *code = nir_ushr_imm(b, status, 29);

   return nir_ior(b, nir_ieq_imm(b, code, RQ_STATUS_TRIANGLE),
                  nir_ieq_imm(b, code, RQ_STATUS_AABB));
}

static nir_def *
lower_rq_proceed(nir_builder *b, struct lower_rq_ctx *ctx,
                 nir_intrinsic_instr *intr)
{
   nir_def *slot;
   nir_deref_instr *rq = rq_deref(b, ctx, intr->src[0].ssa, &slot);
   nir_def *pending = rq_load(b, rq, pending);
   nir_def *terminated = rq_load(b, rq, terminated);
   nir_def *status = rq_load(b, rq, status);

   nir_push_if(b, nir_iand(b, nir_inot(b, nir_ior(b, pending, terminated)),
                           status_is_candidate(b, status)));
   {
      nir_def *state = rq_state_addr(b, ctx, slot);
      nir_def *candidate = load_state(b, state, RQ_CANDIDATE + RQ_REC_PRIM, 2);

      rt_barrier(b);
      nir_def *out = nir_rt_trace_resume_pan(
         b, state, root_node(b, rq_load(b, rq, accel_struct)), candidate);
      rt_barrier(b);

      store_trace_result(b, rq, out);
   }
   nir_pop_if(b, NULL);

   rq_store(b, rq, pending, nir_imm_false(b));

   return nir_iand(b, nir_inot(b, rq_load(b, rq, terminated)),
                   status_is_candidate(b, rq_load(b, rq, status)));
}

static nir_def *
commit_candidate(nir_builder *b, nir_def *state, nir_def *t, uint32_t kind_bits)
{
   nir_def *lo = load_state(b, state, RQ_CANDIDATE, 4);
   nir_def *hi = load_state(b, state, RQ_CANDIDATE + 16, RQ_REC_DWORDS - 4);
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

   store_state(b, state, RQ_COMMITTED, nir_vec(b, comps, 4));
   store_state(b, state, RQ_COMMITTED + 16, nir_vec(b, &comps[4], 3));

   return comps[RQ_REC_T / 4];
}

static void
lower_rq_confirm_intersection(nir_builder *b, struct lower_rq_ctx *ctx,
                              nir_intrinsic_instr *intr)
{
   nir_def *slot;
   nir_deref_instr *rq = rq_deref(b, ctx, intr->src[0].ssa, &slot);

   rq_store(b, rq, committed_t,
            commit_candidate(b, rq_state_addr(b, ctx, slot), NULL, RQ_KIND_COMMIT));
   rq_store(b, rq, committed_type, nir_imm_int(b, 1));
}

static void
lower_rq_generate_intersection(nir_builder *b, struct lower_rq_ctx *ctx,
                               nir_intrinsic_instr *intr)
{
   nir_def *slot;
   nir_deref_instr *rq = rq_deref(b, ctx, intr->src[0].ssa, &slot);

   rq_store(b, rq, committed_t,
            commit_candidate(b, rq_state_addr(b, ctx, slot), intr->src[1].ssa,
                             RQ_KIND_GENERATED));
   rq_store(b, rq, committed_type, nir_imm_int(b, 2));
}

static void
lower_rq_terminate(nir_builder *b, struct lower_rq_ctx *ctx,
                   nir_intrinsic_instr *intr)
{
   nir_deref_instr *rq = rq_deref(b, ctx, intr->src[0].ssa, NULL);

   rq_store(b, rq, terminated, nir_imm_true(b));
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
              nir_intrinsic_instr *intr)
{
   nir_def *slot;
   nir_deref_instr *rq = rq_deref(b, ctx, intr->src[0].ssa, &slot);
   bool committed = nir_intrinsic_committed(intr);
   unsigned rec = committed ? RQ_COMMITTED : RQ_CANDIDATE;
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

   if (value == nir_ray_query_value_intersection_type && !committed) {
      nir_def *code = nir_ushr_imm(b, rq_load(b, rq, status), 29);
      return nir_b2i32(b, nir_ieq_imm(b, code, RQ_STATUS_AABB));
   }

   if (value == nir_ray_query_value_intersection_t && committed)
      return rq_load(b, rq, committed_t);

   nir_def *state = rq_state_addr(b, ctx, slot);
   nir_def *kind = load_state(b, state, rec + RQ_REC_KIND, 1);

   switch (value) {
   case nir_ray_query_value_intersection_type: {
      nir_def *code = nir_ushr_imm(b, kind, 29);
      nir_def *record_type =
         nir_bcsel(b, nir_ieq_imm(b, code, RQ_STATUS_GENERATED),
                   nir_imm_int(b, 2),
                   nir_b2i32(b, nir_uge_imm(b, code, RQ_STATUS_HIT)));
      nir_def *type = rq_load(b, rq, committed_type);
      return nir_bcsel(b, nir_ieq_imm(b, type, RQ_COMMITTED_FROM_RECORD),
                       record_type, type);
   }
   case nir_ray_query_value_intersection_t:
      return load_state(b, state, rec + RQ_REC_T, 1);
   case nir_ray_query_value_intersection_barycentrics:
      return load_state(b, state, rec + RQ_REC_BARY, 2);
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
      return primitive_leaf_word(b, as, kind, load_state(b, state, rec + RQ_REC_PRIM, 1), 13);
   case nir_ray_query_value_intersection_geometry_index:
      return nir_iand_imm(b, primitive_leaf_word(b, as, kind, load_state(b, state, rec + RQ_REC_PRIM, 1), 15), 0xffffff);
   case nir_ray_query_value_intersection_candidate_aabb_opaque: {
      nir_def *inst = instance_leaf(b, as, kind);
      nir_def *inst_flags = nir_ushr_imm(b, load_as(b, nir_iadd_imm(b, inst, offsetof(struct panvk_bvh_instance_leaf, flags_sbt_offset)), 1, 32), 24);
      nir_def *leaf_flags = nir_ushr_imm(b, primitive_leaf_word(b, as, kind, load_state(b, state, rec + RQ_REC_PRIM, 1), 15), 24);
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
   struct rq_var *rq = ralloc(ctx->vars, struct rq_var);
   const struct glsl_type *type = glsl_type_wrap_in_arrays(rq_type(), var->type);

   rq->var = impl ? nir_local_variable_create(impl, type, "ray_query")
                  : nir_variable_create(nir, nir_var_shader_temp, type, "ray_query");
   rq->slot = ctx->slot_count;
   ctx->slot_count += MAX2(glsl_get_aoa_size(var->type), 1);

   _mesa_hash_table_insert(ctx->vars, var, rq);
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

   if (!ctx.slot_count) {
      ralloc_free(ctx.vars);
      return 0;
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

            b.cursor = nir_before_instr(instr);

            nir_def *def = NULL;
            switch (intr->intrinsic) {
            case nir_intrinsic_rq_initialize:
               lower_rq_initialize(&b, &ctx, intr);
               break;
            case nir_intrinsic_rq_proceed:
               def = lower_rq_proceed(&b, &ctx, intr);
               break;
            case nir_intrinsic_rq_confirm_intersection:
               lower_rq_confirm_intersection(&b, &ctx, intr);
               break;
            case nir_intrinsic_rq_generate_intersection:
               lower_rq_generate_intersection(&b, &ctx, intr);
               break;
            case nir_intrinsic_rq_terminate:
               lower_rq_terminate(&b, &ctx, intr);
               break;
            case nir_intrinsic_rq_load:
               def = lower_rq_load(&b, &ctx, intr);
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
