#include "panvk_nir_rt_pipeline.h"
#include "panvk_nir_rt_traversal.h"
#include "panvk_ray_tracing.h"
#include "panvk_shader.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "compiler/spirv/spirv.h"
#include "bvh/panvk_bvh.h"
#include "util/hash_table.h"
#include "util/u_math.h"

static_assert(sizeof(VkTraceRaysIndirectCommand2KHR) == PANVK_RT_DISPATCH_PARAMS_OFFSET,
              "ray tracing dispatch parameters offset");
static_assert(sizeof(struct panvk_rt_dispatch_params) == PANVK_RT_DISPATCH_PARAMS_SIZE,
              "ray tracing dispatch parameters range");

#define RT_HEADER 384
#define RT_NATIVE_HEADER 256
#define RT_STATE 256
#define RT_COMMITTED (RT_STATE + 0x24)
#define RT_CANDIDATE (RT_STATE + 0x64)
#define RT_REPORT_TOKEN UINT32_MAX

enum rt_offset {
   RT_PARENT = 0, RT_RETURN = 8, RT_RAY = 16,
   RT_RECORD = 24, RT_SHADER_RECORD = 32, RT_PAYLOAD = 40,
   RT_HIT_KIND = 52, RT_ACTION = 56, RT_AS = 64, RT_FLAGS = 72,
   RT_CULL = 76, RT_SBT_OFFSET = 80, RT_SBT_STRIDE = 84,
   RT_MISS = 88, RT_TMIN = 92, RT_ORIGIN = 96, RT_TMAX = 108,
   RT_DIRECTION = 112, RT_STATUS = 124, RT_COMMITTED_T = 128,
   RT_GENERATED_KIND = 132, RT_DEPTH = 136, RT_REPORT_T = 144,
   RT_REPORT_KIND = 148, RT_REPORT_ACCEPTED = 152, RT_REPORT_RESUME = 156,
   RT_ATTR = 160, RT_COMMITTED_ATTR = 192, RT_PC = 224, RT_FRAME = 232,
   RT_LAUNCH = 240,
};

enum rt_phase {
   RT_BEGIN, RT_RESUME, RT_CANDIDATE_ROUTE, RT_TRIANGLE_COMMIT,
   RT_REPORT_START, RT_REPORT_COMMIT, RT_FINISH, RT_TRACE_RETURN, RT_PHASE_COUNT,
};

struct rt_stage {
   nir_shader *nir;
   nir_shader **resumes;
   uint32_t resume_count;
   uint32_t resume_base;
   mesa_shader_stage stage;
   uint32_t id;
   uint32_t ray_query_slots;
   uint32_t ray_offset;
   uint32_t frame_size;
};

struct rt_context {
   const struct panvk_nir_rt_pipeline_info *info;
   struct rt_stage *stages;
   uint32_t phase_base;
   uint32_t frame_size;
   uint32_t external_frame_size;
};

struct rt_part {
   struct rt_context *ctx;
   struct rt_stage *stage;
   nir_variable *yielded;
   nir_variable *action;
   nir_variable *hit_kind;
   nir_variable *hit_t;
   bool private_io;
};

static nir_def *
rt_load(nir_builder *b, nir_def *base, unsigned offset, unsigned comps,
        unsigned bits)
{
   return nir_load_global(b, comps, bits, nir_iadd_imm(b, base, offset),
                          .align_mul = bits == 64 ? 8 : 4);
}

static void
rt_store(nir_builder *b, nir_def *base, unsigned offset, nir_def *value)
{
   nir_store_global(b, value, nir_iadd_imm(b, base, offset),
                    .align_mul = value->bit_size == 64 ? 8 : 4);
}

static nir_def *
rt_load_constant(nir_builder *b, nir_def *base, unsigned offset,
                 unsigned comps, unsigned bits)
{
   return nir_load_global_constant(b, comps, bits,
      nir_iadd_imm(b, base, offset), .align_mul = bits == 64 ? 8 : 4);
}

static nir_def *
rt_load_dispatch(nir_builder *b, const struct rt_context *ctx, nir_def *dispatch,
                  unsigned offset, unsigned comps, unsigned bits)
{
   return panvk_nir_load_rt_dispatch(b, dispatch, offset, comps, bits,
      ctx->info->native ? ctx->info->dispatch_sysval_offset + 8 : 0);
}

static nir_def *
rt_stage_load(nir_builder *b, const struct rt_part *part, nir_def *base,
              unsigned offset, unsigned comps, unsigned bits)
{
   if (!part->ctx->info->native)
      return rt_load(b, base, offset, comps, bits);

   unsigned arg;
   switch (offset) {
   case RT_PAYLOAD: arg = 5; break;
   case RT_RAY: arg = 6; break;
   case RT_RECORD: arg = 7; break;
   case RT_SHADER_RECORD: arg = 8; break;
   case RT_TMAX: arg = 9; break;
   case RT_HIT_KIND: arg = 10; break;
   case RT_DEPTH: arg = 11; break;
   default: return rt_load_constant(b, base, offset, comps, bits);
   }
   assert(comps == 1);
   return nir_load_rt_arg_pan(b, bits, .base = arg);
}

static nir_def *
rt_param(nir_builder *b, unsigned index)
{
   return nir_load_uniform(b, 1, 64, nir_imm_int(b, index), .base = 0, .range = 8);
}

static nir_def *
rt_ray(nir_builder *b, nir_def *frame)
{
   return rt_load(b, frame, RT_RAY, 1, 64);
}

static nir_def *
rt_state(nir_builder *b, struct rt_context *ctx, nir_def *frame)
{
   if (ctx->info->native) {
      return panvk_nir_rt_state_addr(b, nir_load_rt_arg_pan(b, 64, .base = 2),
         nir_iand_imm(b, nir_load_rt_arg_pan(b, 32, .base = 12), 0xffff));
   }
   return nir_iadd_imm(b, frame, RT_STATE);
}

static void
rt_select(nir_builder *b, nir_def *root, nir_def *frame, nir_def *pc)
{
   rt_store(b, root, RT_FRAME, frame);
   rt_store(b, root, RT_PC, pc);
}

static void
rt_select_phase(nir_builder *b, struct rt_context *ctx, nir_def *root,
                nir_def *frame, enum rt_phase phase)
{
   rt_select(b, root, frame, nir_imm_int(b, ctx->phase_base + phase));
}

static void
rt_return(nir_builder *b, nir_def *root, nir_def *frame, bool any_hit)
{
   nir_def *parent = rt_load(b, frame, RT_PARENT, 1, 64);
   if (any_hit)
      rt_store(b, parent, RT_ACTION, rt_load(b, frame, RT_ACTION, 1, 32));
   rt_select(b, root, parent, rt_load(b, frame, RT_RETURN, 1, 32));
}

static nir_def *
rt_token_pc(nir_builder *b, struct rt_context *ctx, nir_def *dispatch, nir_def *token,
            mesa_shader_stage stage)
{
   nir_def *pc = nir_imm_int(b, 0);
   for (unsigned i = 0; i < ctx->info->stage_count; i++) {
      if (ctx->stages[i].stage == stage)
         pc = nir_bcsel(b, nir_ieq(b, token, rt_load(b, rt_load_constant(b, dispatch, 128, 1, 64), i * 4, 1, 32)),
                        nir_imm_int(b, i + 1), pc);
   }
   return pc;
}

static nir_def *
rt_child(nir_builder *b, struct rt_context *ctx, nir_def *parent)
{
   return nir_iadd_imm(b, parent, ctx->frame_size);
}

static nir_def *
rt_child_valid(nir_builder *b, struct rt_context *ctx, nir_def *dispatch,
               nir_def *root, nir_def *child)
{
   return nir_uge(b, rt_load_constant(b, dispatch, 112, 1, 64),
                  nir_iadd_imm(b, nir_isub(b, child, root), ctx->frame_size));
}

static void
rt_init_child(nir_builder *b, nir_def *child, nir_def *parent,
              nir_def *ray, nir_def *record, nir_def *sbt,
              nir_def *payload, nir_def *return_pc)
{
   rt_store(b, child, RT_PARENT, parent);
   rt_store(b, child, RT_RETURN, return_pc);
   rt_store(b, child, RT_RAY, ray);
   rt_store(b, child, RT_RECORD, record);
   rt_store(b, child, RT_SHADER_RECORD, nir_iadd_imm(b, sbt, 32));
   rt_store(b, child, RT_PAYLOAD, payload);
   rt_store(b, child, RT_DEPTH, rt_load(b, parent, RT_DEPTH, 1, 32));
   rt_store(b, child, RT_ACTION, nir_imm_int(b, PANVK_NIR_RT_ACCEPT));
}

static nir_def *
rt_sbt(nir_builder *b, nir_def *dispatch, unsigned base, unsigned stride,
       nir_def *index)
{
   nir_def *address = rt_load_constant(b, dispatch, base, 1, 64);
   return nir_bcsel(b, nir_ine_imm(b, address, 0),
      nir_iadd(b, address,
         nir_imul(b, nir_u2u64(b, index), rt_load_constant(b, dispatch, stride, 1, 64))),
      nir_imm_int64(b, 0));
}

static nir_def *
rt_sbt_token(nir_builder *b, nir_def *sbt, unsigned slot)
{
   nir_push_if(b, nir_ine_imm(b, sbt, 0));
   nir_def *token = rt_load(b, sbt, slot, 1, 32);
   nir_push_else(b, NULL);
   nir_def *zero = nir_imm_int(b, 0);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, token, zero);
}

static void
rt_invoke(nir_builder *b, struct rt_context *ctx, nir_def *root,
          nir_def *dispatch, nir_def *parent, nir_def *ray,
          nir_def *record, nir_def *sbt, unsigned slot,
          mesa_shader_stage stage, uint32_t return_pc,
          nir_def *t, nir_def *hit_kind)
{
   nir_def *pc = rt_token_pc(b, ctx, dispatch, rt_sbt_token(b, sbt, slot), stage);
   nir_def *child = rt_child(b, ctx, parent);
   nir_push_if(b, nir_iand(b, nir_ine_imm(b, pc, 0),
                            rt_child_valid(b, ctx, dispatch, root, child)));
   rt_init_child(b, child, parent, ray, record, sbt,
                 rt_load(b, ray, RT_PAYLOAD, 1, 64),
                 nir_imm_int(b, return_pc));
   rt_store(b, child, RT_TMAX, t);
   rt_store(b, child, RT_HIT_KIND, hit_kind);
   rt_select(b, root, child, pc);
   nir_push_else(b, NULL);
   rt_store(b, parent, RT_ACTION, nir_imm_int(b, PANVK_NIR_RT_ACCEPT));
   rt_select(b, root, parent, nir_imm_int(b, return_pc));
   nir_pop_if(b, NULL);
}

static nir_def *
rt_instance(nir_builder *b, nir_def *as, nir_def *kind)
{
   nir_def *count = rt_load(b, as, offsetof(struct panvk_bvh_header, node_count), 1, 32);
   nir_def *index = nir_iadd(b, count, nir_iand_imm(b, kind, 0xffffff));
   return nir_iadd(b, nir_iadd_imm(b, as, 128),
                   nir_u2u64(b, nir_imul_imm(b, index, 64)));
}

static nir_def *
rt_extra(nir_builder *b, nir_def *as, nir_def *kind, unsigned offset)
{
   nir_def *nodes = rt_load(b, as, offsetof(struct panvk_bvh_header, node_count), 1, 32);
   nir_def *prims = rt_load(b, as, offsetof(struct panvk_bvh_header, primitive_count), 1, 32);
   nir_def *base = nir_iadd(b, nir_iadd_imm(b, as, 128),
                             nir_u2u64(b, nir_imul_imm(b, nir_iadd(b, nodes, prims), 64)));
   return rt_load(b, nir_iadd(b, base,
                  nir_u2u64(b, nir_imul_imm(b, nir_iand_imm(b, kind, 0xffffff), 8))),
                  offset, 1, 32);
}

static nir_def *
rt_leaf_word(nir_builder *b, nir_def *as, nir_def *record, unsigned word)
{
   nir_def *kind = rt_load(b, record, 16, 1, 32);
   nir_def *prim = rt_load(b, record, 12, 1, 32);
   return panvk_nir_rt_leaf_word(b, as, kind, prim, word);
}

static nir_def *
rt_opaque(nir_builder *b, nir_def *ray, nir_def *record)
{
   nir_def *as = rt_load(b, ray, RT_AS, 1, 64);
   nir_def *inst = rt_instance(b, as, rt_load(b, record, 16, 1, 32));
   nir_def *inst_flags = nir_ushr_imm(b, rt_load(b, inst, 52, 1, 32), 24);
   nir_def *leaf_flags = nir_ushr_imm(b, rt_leaf_word(b, as, record, 15), 24);
   nir_def *flags = rt_load(b, ray, RT_FLAGS, 1, 32);
   nir_def *opaque = nir_test_mask(b, leaf_flags, PANVK_BVH_LEAF_OPAQUE);
   opaque = nir_bcsel(b, nir_test_mask(b, inst_flags, VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR), nir_imm_true(b), opaque);
   opaque = nir_bcsel(b, nir_test_mask(b, inst_flags, VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR), nir_imm_false(b), opaque);
   opaque = nir_bcsel(b, nir_test_mask(b, flags, SpvRayFlagsNoOpaqueKHRMask), nir_imm_false(b), opaque);
   return nir_bcsel(b, nir_test_mask(b, flags, SpvRayFlagsOpaqueKHRMask), nir_imm_true(b), opaque);
}

static nir_def *
rt_transform(nir_builder *b, nir_def **rows, nir_def *v, bool point)
{
   uint32_t saved = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_fp_no_fast_math;
   nir_def *out[3];
   for (unsigned r = 0; r < 3; r++) {
      nir_def *sum = point ? nir_channel(b, rows[r], 3) : nir_imm_float(b, -0.0);
      sum = nir_ffma(b, nir_channel(b, rows[r], 1), nir_channel(b, v, 1), sum);
      sum = nir_ffma(b, nir_channel(b, rows[r], 0), nir_channel(b, v, 0), sum);
      sum = nir_ffma(b, nir_channel(b, rows[r], 2), nir_channel(b, v, 2), sum);
      out[r] = point ? sum : nir_ffma(b, nir_channel(b, rows[r], 3), nir_imm_float(b, 0), sum);
   }
   b->fp_math_ctrl = saved;
   return nir_vec(b, out, 3);
}

static nir_def *
rt_inverse_column(nir_builder *b, nir_def **rows, unsigned column)
{
   nir_def *m[3][3], *cof[3][3], *inv[3][3];
   for (unsigned r = 0; r < 3; r++)
      for (unsigned c = 0; c < 3; c++)
         m[r][c] = nir_channel(b, rows[r], c);
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
   nir_def *invdet = nir_frcp(b, det);
   for (unsigned r = 0; r < 3; r++)
      for (unsigned c = 0; c < 3; c++)
         inv[r][c] = nir_fmul(b, cof[c][r], invdet);
   nir_def *out[3];
   for (unsigned r = 0; r < 3; r++) {
      if (column < 3)
         out[r] = inv[r][column];
      else
         out[r] = nir_fneg(b, nir_fadd(b,
            nir_fadd(b, nir_fmul(b, inv[r][0], nir_channel(b, rows[0], 3)),
                        nir_fmul(b, inv[r][1], nir_channel(b, rows[1], 3))),
            nir_fmul(b, inv[r][2], nir_channel(b, rows[2], 3))));
   }
   return nir_vec(b, out, 3);
}

static bool
rt_lower_builtin(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct rt_part *part = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *frame = rt_param(b, 0);
   nir_def *dispatch = rt_param(b, 1);
   nir_def *value = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ray_launch_id:
      if (part->ctx->info->native && part->ctx->info->tls)
         return false;
      value = part->ctx->info->native
         ? rt_load_constant(b, rt_param(b, 2), RT_LAUNCH, 3, 32)
         : nir_load_global_invocation_id(b, 32);
      break;
   case nir_intrinsic_load_ray_launch_size:
      value = rt_load_dispatch(b, part->ctx, dispatch, 88, 3, 32); break;
   case nir_intrinsic_load_shader_record_ptr:
      value = rt_stage_load(b, part, frame, RT_SHADER_RECORD, 1, 64); break;
   case nir_intrinsic_load_push_constant: {
      if (nir_intrinsic_base(intr) >= SYSVALS_PUSH_CONST_BASE)
         return false;
      nir_def *base = rt_load_dispatch(b, part->ctx, dispatch, 120, 1, 64);
      unsigned offset = (part->stage->stage - MESA_SHADER_RAYGEN) * 256;
      value = nir_load_global_constant(b, intr->def.num_components, intr->def.bit_size,
         nir_iadd(b, nir_iadd_imm(b, base, offset + nir_intrinsic_base(intr)),
                     nir_u2u64(b, intr->src[0].ssa)), .align_mul = 4);
      break;
   }
   default: break;
   }
   if (!value) {
      switch (intr->intrinsic) {
      case nir_intrinsic_load_ray_world_origin:
      case nir_intrinsic_load_ray_world_direction:
      case nir_intrinsic_load_ray_t_min:
      case nir_intrinsic_load_ray_t_max:
         if (intr->intrinsic == nir_intrinsic_load_ray_t_max &&
             part->ctx->info->native && part->stage->stage == MESA_SHADER_INTERSECTION)
            return false;
         FALLTHROUGH;
      case nir_intrinsic_load_ray_hit_kind:
      case nir_intrinsic_load_ray_flags:
      case nir_intrinsic_load_cull_mask:
      case nir_intrinsic_load_ray_geometry_index:
      case nir_intrinsic_load_ray_instance_custom_index:
      case nir_intrinsic_load_instance_id:
      case nir_intrinsic_load_primitive_id:
      case nir_intrinsic_load_ray_triangle_vertex_positions:
      case nir_intrinsic_load_ray_object_origin:
      case nir_intrinsic_load_ray_object_direction:
      case nir_intrinsic_load_ray_world_to_object:
      case nir_intrinsic_load_ray_object_to_world:
         break;
      default: return false;
      }
      nir_def *ray = rt_stage_load(b, part, frame, RT_RAY, 1, 64);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_ray_world_origin: value = rt_load(b, ray, RT_ORIGIN, 3, 32); break;
      case nir_intrinsic_load_ray_world_direction: value = rt_load(b, ray, RT_DIRECTION, 3, 32); break;
      case nir_intrinsic_load_ray_t_min: value = rt_load(b, ray, RT_TMIN, 1, 32); break;
      case nir_intrinsic_load_ray_t_max:
         value = part->stage->stage == MESA_SHADER_INTERSECTION
            ? rt_load(b, ray, RT_COMMITTED_T, 1, 32)
            : rt_stage_load(b, part, frame, RT_TMAX, 1, 32); break;
      case nir_intrinsic_load_ray_hit_kind: value = rt_stage_load(b, part, frame, RT_HIT_KIND, 1, 32); break;
      case nir_intrinsic_load_ray_flags: value = rt_load(b, ray, RT_FLAGS, 1, 32); break;
      case nir_intrinsic_load_cull_mask: value = rt_load(b, ray, RT_CULL, 1, 32); break;
      default: break;
      }
      if (!value) {
         nir_def *record = rt_stage_load(b, part, frame, RT_RECORD, 1, 64);
         nir_def *as = rt_load(b, ray, RT_AS, 1, 64);
         nir_def *kind = rt_load(b, record, 16, 1, 32);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_ray_geometry_index:
            value = nir_iand_imm(b, rt_leaf_word(b, as, record, 15), 0xffffff); break;
         case nir_intrinsic_load_ray_instance_custom_index:
            value = nir_iand_imm(b, rt_extra(b, as, kind, 0), 0xffffff); break;
         case nir_intrinsic_load_instance_id: value = rt_extra(b, as, kind, 4); break;
         case nir_intrinsic_load_primitive_id: value = rt_leaf_word(b, as, record, 13); break;
         case nir_intrinsic_load_ray_triangle_vertex_positions:
            value = panvk_nir_rt_triangle_vertex(b, as, kind,
               rt_load(b, record, 12, 1, 32), nir_intrinsic_column(intr));
            break;
         default: {
            nir_def *inst = rt_instance(b, as, kind);
            nir_def *rows[3];
            for (unsigned r = 0; r < 3; r++)
               rows[r] = rt_load(b, inst, r * 16, 4, 32);
            if (intr->intrinsic == nir_intrinsic_load_ray_object_origin)
               value = rt_transform(b, rows, rt_load(b, ray, RT_ORIGIN, 3, 32), true);
            else if (intr->intrinsic == nir_intrinsic_load_ray_object_direction)
               value = rt_transform(b, rows, rt_load(b, ray, RT_DIRECTION, 3, 32), false);
            else if (intr->intrinsic == nir_intrinsic_load_ray_object_to_world)
               value = rt_inverse_column(b, rows, nir_intrinsic_column(intr));
            else {
               unsigned c = nir_intrinsic_column(intr);
               value = nir_vec3(b, nir_channel(b, rows[0], c),
                                  nir_channel(b, rows[1], c),
                                  nir_channel(b, rows[2], c));
            }
            break;
         }
         }
      }
   }
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
rt_lower_deref(nir_builder *b, nir_instr *instr, void *data)
{
   struct rt_part *part = data;
   if (instr->type != nir_instr_type_deref)
      return false;
   nir_deref_instr *deref = nir_instr_as_deref(instr);
   if (deref->deref_type == nir_deref_type_cast &&
       deref->modes == nir_var_mem_constant &&
       nir_def_instr(deref->parent.ssa)->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(nir_def_instr(deref->parent.ssa))->intrinsic ==
          nir_intrinsic_load_shader_record_ptr) {
      deref->modes = nir_var_mem_global;
      return true;
   }
   if (deref->deref_type != nir_deref_type_var)
      return false;
   nir_variable *var = deref->var;
   nir_variable_mode modes = nir_var_shader_temp | nir_var_function_temp |
      nir_var_shader_call_data | nir_var_ray_hit_attrib;
   if (!(var->data.mode & modes))
      return false;
   if (part->ctx->info->native && !part->private_io &&
       (var->data.mode & (nir_var_shader_temp | nir_var_function_temp)))
      return false;
   b->cursor = nir_before_instr(instr);
   nir_def *frame = part->private_io ? nir_load_rt_arg_pan(b, 64, .base = 0) : rt_param(b, 0);
   nir_def *base;
   if (var->data.mode == nir_var_shader_call_data)
      base = rt_stage_load(b, part, frame, RT_PAYLOAD, 1, 64);
   else if (var->data.mode == nir_var_ray_hit_attrib)
      base = nir_iadd_imm(b, rt_stage_load(b, part, frame, RT_RAY, 1, 64),
         part->stage->stage == MESA_SHADER_CLOSEST_HIT ? RT_COMMITTED_ATTR : RT_ATTR);
   else
      base = nir_iadd_imm(b, frame, part->ctx->info->native ? part->stage->frame_size : RT_HEADER);
   nir_deref_instr *cast = nir_build_deref_cast(b,
      nir_iadd_imm(b, base, var->data.driver_location),
      nir_var_mem_global, var->type, 0);
   nir_def_rewrite_uses(&deref->def, &cast->def);
   nir_instr_remove(instr);
   return true;
}

static bool
rt_prepare_report(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_report_ray_intersection)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *frame = rt_param(b, 0);
   rt_store(b, frame, RT_REPORT_T, intr->src[0].ssa);
   rt_store(b, frame, RT_REPORT_KIND, intr->src[1].ssa);
   nir_execute_callable(b, nir_imm_int(b, RT_REPORT_TOKEN), nir_imm_int64(b, 0));
   nir_def *accepted = nir_i2b(b, rt_load(b, frame, RT_REPORT_ACCEPTED, 1, 32));
   nir_def_rewrite_uses(&intr->def, accepted);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
rt_remat(nir_instr *instr, void *data)
{
   return instr->type == nir_instr_type_intrinsic &&
      nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_uniform;
}

static bool
rt_lower_part(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct rt_part *part = data;
   struct rt_context *ctx = part->ctx;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *frame = rt_param(b, 0), *dispatch = rt_param(b, 1), *root = rt_param(b, 2);
   if (intr->intrinsic == nir_intrinsic_rt_trace_ray ||
       intr->intrinsic == nir_intrinsic_rt_execute_callable ||
       intr->intrinsic == nir_intrinsic_ignore_ray_intersection ||
       intr->intrinsic == nir_intrinsic_terminate_ray ||
       intr->intrinsic == nir_intrinsic_accept_ray_intersection)
      nir_store_var(b, part->yielded, nir_imm_true(b), 1);
   nir_def *value = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_scratch_base_ptr:
      value = nir_iadd_imm(b, frame, RT_HEADER);
      break;
   case nir_intrinsic_load_stack:
      value = nir_load_global(b, intr->def.num_components, intr->def.bit_size,
         nir_iadd_imm(b, frame, RT_HEADER + nir_intrinsic_base(intr)),
         .align_mul = nir_intrinsic_align_mul(intr),
         .align_offset = nir_intrinsic_align_offset(intr));
      break;
   case nir_intrinsic_store_stack:
      nir_store_global(b, intr->src[0].ssa,
         nir_iadd_imm(b, frame, RT_HEADER + nir_intrinsic_base(intr)),
         .align_mul = nir_intrinsic_align_mul(intr),
         .align_offset = nir_intrinsic_align_offset(intr),
         .write_mask = nir_intrinsic_write_mask(intr));
      break;
   case nir_intrinsic_rt_resume: break;
   case nir_intrinsic_ignore_ray_intersection:
   case nir_intrinsic_terminate_ray:
   case nir_intrinsic_accept_ray_intersection:
      rt_store(b, frame, RT_ACTION, nir_imm_int(b,
         intr->intrinsic == nir_intrinsic_ignore_ray_intersection ? PANVK_NIR_RT_IGNORE :
         intr->intrinsic == nir_intrinsic_terminate_ray ? PANVK_NIR_RT_TERMINATE : PANVK_NIR_RT_ACCEPT));
      rt_return(b, root, frame, true);
      nir_instr *last = nir_block_last_instr(intr->instr.block);
      if (last->type == nir_instr_type_jump) {
         assert(nir_instr_as_jump(last)->type == nir_jump_halt);
         nir_instr_as_jump(last)->type = nir_jump_return;
         nir_instr_remove(&intr->instr);
      } else {
         b->cursor = nir_instr_remove(&intr->instr);
         nir_jump(b, nir_jump_return);
      }
      return true;
   case nir_intrinsic_rt_trace_ray: {
      nir_def *child = rt_child(b, ctx, frame);
      nir_def *depth = nir_iadd_imm(b, rt_load(b, frame, RT_DEPTH, 1, 32), 1);
      nir_def *valid = nir_iand(b, rt_child_valid(b, ctx, dispatch, root, child),
                                  nir_ule_imm(b, depth, ctx->info->max_recursion_depth));
      nir_push_if(b, valid);
      rt_init_child(b, child, frame, child, nir_imm_int64(b, 0), nir_imm_int64(b, 0),
                    nir_u2u64(b, intr->src[10].ssa),
                    nir_imm_int(b, part->stage->resume_base + nir_intrinsic_call_idx(intr)));
      rt_store(b, child, RT_DEPTH, depth);
      rt_store(b, child, RT_AS, intr->src[0].ssa);
      unsigned pipeline_flags = 0;
      if (ctx->info->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR)
         pipeline_flags |= SpvRayFlagsSkipTrianglesKHRMask;
      if (ctx->info->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR)
         pipeline_flags |= SpvRayFlagsSkipAABBsKHRMask;
      rt_store(b, child, RT_FLAGS, nir_ior_imm(b, intr->src[1].ssa, pipeline_flags));
      rt_store(b, child, RT_CULL, nir_iand_imm(b, intr->src[2].ssa, 255));
      rt_store(b, child, RT_SBT_OFFSET, nir_iand_imm(b, intr->src[3].ssa, 15));
      rt_store(b, child, RT_SBT_STRIDE, nir_iand_imm(b, intr->src[4].ssa, 15));
      rt_store(b, child, RT_MISS, nir_iand_imm(b, intr->src[5].ssa, 65535));
      rt_store(b, child, RT_ORIGIN, intr->src[6].ssa);
      rt_store(b, child, RT_TMIN, intr->src[7].ssa);
      rt_store(b, child, RT_DIRECTION, intr->src[8].ssa);
      rt_store(b, child, RT_TMAX, intr->src[9].ssa);
      rt_select_phase(b, ctx, root, child, RT_BEGIN);
      nir_push_else(b, NULL);
      rt_select(b, root, frame,
                nir_imm_int(b, part->stage->resume_base + nir_intrinsic_call_idx(intr)));
      nir_pop_if(b, NULL);
      break;
   }
   case nir_intrinsic_rt_execute_callable: {
      uint32_t resume = part->stage->resume_base + nir_intrinsic_call_idx(intr);
      if (part->stage->stage == MESA_SHADER_INTERSECTION &&
          nir_src_is_const(intr->src[0]) && nir_src_as_uint(intr->src[0]) == RT_REPORT_TOKEN) {
         rt_store(b, frame, RT_REPORT_RESUME, nir_imm_int(b, resume));
         rt_select_phase(b, ctx, root, frame, RT_REPORT_START);
      } else {
         nir_def *sbt = rt_sbt(b, dispatch, 64, 80, intr->src[0].ssa);
         nir_def *pc = rt_token_pc(b, ctx, dispatch, rt_sbt_token(b, sbt, 0), MESA_SHADER_CALLABLE);
         nir_def *child = rt_child(b, ctx, frame);
         nir_push_if(b, nir_iand(b, nir_ine_imm(b, pc, 0),
                                  rt_child_valid(b, ctx, dispatch, root, child)));
         rt_init_child(b, child, frame, rt_ray(b, frame),
                       rt_load(b, frame, RT_RECORD, 1, 64), sbt,
                       nir_u2u64(b, intr->src[1].ssa), nir_imm_int(b, resume));
         rt_store(b, child, RT_TMAX, rt_load(b, frame, RT_TMAX, 1, 32));
         rt_store(b, child, RT_HIT_KIND, rt_load(b, frame, RT_HIT_KIND, 1, 32));
         rt_select(b, root, child, pc);
         nir_push_else(b, NULL);
         rt_select(b, root, frame, nir_imm_int(b, resume));
         nir_pop_if(b, NULL);
      }
      break;
   }
   default: return false;
   }
   if (value)
      nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
rt_lower_param(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_uniform)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *param = nir_load_param(b, nir_src_as_uint(intr->src[0]));
   nir_def_rewrite_uses(&intr->def, param);
   nir_instr_remove(&intr->instr);
   return true;
}

static nir_function_impl *
rt_entrypoint(nir_shader *nir)
{
   nir_foreach_function(func, nir) {
      if (func->is_entrypoint)
         return func->impl;
   }
   UNREACHABLE("Missing RT entrypoint");
}

static void
rt_prepare_part(struct rt_context *ctx, struct rt_stage *stage, nir_shader *nir)
{
   nir_builder start = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
   struct rt_part part = {
      .ctx = ctx, .stage = stage,
      .yielded = nir_local_variable_create(start.impl, glsl_bool_type(), "rt_yielded"),
   };
   nir_store_var(&start, part.yielded, nir_imm_false(&start), 1);
   nir_shader_intrinsics_pass(nir, rt_lower_part, nir_metadata_none, &part);
   nir_builder b = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   if (!nir_block_ends_in_jump(nir_impl_last_block(b.impl))) {
      nir_push_if(&b, nir_inot(&b, nir_load_var(&b, part.yielded)));
      rt_return(&b, rt_param(&b, 2), rt_param(&b, 0), stage->stage == MESA_SHADER_ANY_HIT);
      nir_pop_if(&b, NULL);
   }
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_jump && nir_instr_as_jump(instr)->type == nir_jump_halt)
            nir_instr_as_jump(instr)->type = nir_jump_return;
      }
   }
   nir_lower_returns(nir);
   nir_opt_dce(nir);
   nir_opt_dead_cf(nir);
   nir_function *func = b.impl->function;
   func->num_params = 3;
   func->params = ralloc_array(func, nir_parameter, 3);
   for (unsigned p = 0; p < 3; p++)
      func->params[p] = (nir_parameter){.num_components = 1, .bit_size = 64};
   nir_shader_intrinsics_pass(nir, rt_lower_param, nir_metadata_none, NULL);
   nir_validate_shader(nir, "panvk rt continuation part");
}

static void
rt_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_NONE, .memory_scope = SCOPE_SUBGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_global);
}

static nir_def *
rt_root_node(nir_builder *b, nir_def *as)
{
   return nir_bcsel(b, nir_ine_imm(b, as, 0), nir_iadd_imm(b, as, 128), nir_imm_int64(b, 0));
}

static void
rt_trace_result(nir_builder *b, struct rt_context *ctx, nir_def *root,
                nir_def *frame, nir_def *out)
{
   rt_store(b, frame, RT_STATUS, nir_channel(b, out, 0));
   rt_store(b, frame, RT_COMMITTED_T, nir_channel(b, out, 1));
   rt_select_phase(b, ctx, root, frame, RT_CANDIDATE_ROUTE);
}

static nir_def *
rt_tri_kind(nir_builder *b, nir_def *record)
{
   return nir_bcsel(b, nir_test_mask(b, rt_load(b, record, 16, 1, 32), 1u << 24),
                    nir_imm_int(b, 0xfe), nir_imm_int(b, 0xff));
}

static void
rt_commit(nir_builder *b, struct rt_context *ctx, nir_def *ray,
          nir_def *t, nir_def *kind, bool generated, nir_def *native_state)
{
   nir_def *state = native_state ? native_state : rt_state(b, ctx, ray);
   nir_def *record = nir_iadd_imm(b, state, 0x64);
   nir_def *lo = rt_load(b, record, 0, 4, 32), *hi = rt_load(b, record, 16, 3, 32);
   nir_def *parts[7];
   for (unsigned i = 0; i < 7; i++)
      parts[i] = nir_channel(b, i < 4 ? lo : hi, i % 4);
   parts[0] = t;
   parts[4] = generated ? nir_ior_imm(b, nir_iand_imm(b, parts[4], 0x1fffffff), 0xe0000000)
                        : nir_ior_imm(b, parts[4], 0x80000000);
   rt_store(b, state, 0x24, nir_vec(b, parts, 4));
   rt_store(b, state, 0x34, nir_vec(b, &parts[4], 3));
   if (!ctx->info->native)
      rt_store(b, ray, RT_COMMITTED_T, t);
   if (generated && !ctx->info->native)
      rt_store(b, ray, RT_GENERATED_KIND, kind);
   if (!ctx->info->native || generated) {
      for (unsigned i = 0; i < 32; i += 16)
         rt_store(b, ray, RT_COMMITTED_ATTR + i, rt_load(b, ray, RT_ATTR + i, 4, 32));
   }
}

static nir_def *
rt_terminated(nir_builder *b, nir_def *ray, nir_def *action)
{
   return nir_ior(b, nir_ieq_imm(b, action, PANVK_NIR_RT_TERMINATE),
                    nir_test_mask(b, rt_load(b, ray, RT_FLAGS, 1, 32), SpvRayFlagsTerminateOnFirstHitKHRMask));
}

static void
rt_report_commit(nir_builder *b, struct rt_context *ctx, nir_def *root,
                 nir_def *frame)
{
   nir_def *ray = rt_ray(b, frame);
   nir_def *action = rt_load(b, frame, RT_ACTION, 1, 32);
   nir_def *accept = nir_ine_imm(b, action, PANVK_NIR_RT_IGNORE);
   rt_store(b, frame, RT_REPORT_ACCEPTED, nir_b2i32(b, accept));
   nir_push_if(b, accept);
   rt_commit(b, ctx, ray, rt_load(b, frame, RT_REPORT_T, 1, 32),
             rt_load(b, frame, RT_REPORT_KIND, 1, 32), true, NULL);
   nir_pop_if(b, NULL);
   nir_push_if(b, nir_iand(b, accept, rt_terminated(b, ray, action)));
   rt_select_phase(b, ctx, root, ray, RT_FINISH);
   nir_push_else(b, NULL);
   rt_select(b, root, frame, rt_load(b, frame, RT_REPORT_RESUME, 1, 32));
   nir_pop_if(b, NULL);
}

static void
rt_emit_phase(nir_builder *b, struct rt_context *ctx, nir_def *root,
              nir_def *dispatch, nir_def *frame, enum rt_phase phase)
{
   nir_def *candidate = nir_iadd_imm(b, frame, RT_CANDIDATE);
   nir_def *committed = nir_iadd_imm(b, frame, RT_COMMITTED);
   switch (phase) {
   case RT_BEGIN: {
      rt_store(b, frame, RT_COMMITTED + 16, nir_imm_int(b, 0));
      rt_store(b, frame, RT_GENERATED_KIND, nir_imm_int(b, 0));
      rt_barrier(b);
      nir_def *out = nir_rt_trace_begin_pan(b, nir_iadd_imm(b, frame, RT_STATE),
         rt_root_node(b, rt_load(b, frame, RT_AS, 1, 64)),
         rt_load(b, frame, RT_ORIGIN, 3, 32), rt_load(b, frame, RT_DIRECTION, 3, 32),
         rt_load(b, frame, RT_TMIN, 1, 32), rt_load(b, frame, RT_TMAX, 1, 32),
         nir_ior(b, rt_load(b, frame, RT_FLAGS, 1, 32),
                    nir_ishl_imm(b, rt_load(b, frame, RT_CULL, 1, 32), 24)),
         panvk_nir_rt_pack_sbt(b, rt_load(b, frame, RT_SBT_OFFSET, 1, 32),
                              rt_load(b, frame, RT_SBT_STRIDE, 1, 32),
                              rt_load(b, frame, RT_MISS, 1, 32)));
      rt_barrier(b);
      rt_trace_result(b, ctx, root, frame, out);
      break;
   }
   case RT_RESUME: {
      rt_barrier(b);
      nir_def *out = nir_rt_trace_resume_pan(b, nir_iadd_imm(b, frame, RT_STATE),
         rt_root_node(b, rt_load(b, frame, RT_AS, 1, 64)), rt_load(b, candidate, 12, 2, 32));
      rt_barrier(b);
      rt_trace_result(b, ctx, root, frame, out);
      break;
   }
   case RT_CANDIDATE_ROUTE: {
      nir_def *status = rt_load(b, frame, RT_STATUS, 1, 32);
      nir_def *code = nir_ushr_imm(b, status, 29);
      nir_def *sbt = rt_sbt(b, dispatch, 40, 56, nir_iand_imm(b, status, 0x0fffffff));
      nir_push_if(b, nir_ieq_imm(b, code, 1));
      rt_store(b, frame, RT_ATTR, rt_load(b, candidate, 4, 2, 32));
      rt_invoke(b, ctx, root, dispatch, frame, frame, candidate, sbt, 8,
                MESA_SHADER_ANY_HIT, ctx->phase_base + RT_TRIANGLE_COMMIT,
                rt_load(b, candidate, 0, 1, 32), rt_tri_kind(b, candidate));
      nir_push_else(b, NULL);
      nir_push_if(b, nir_ieq_imm(b, code, 2));
      rt_invoke(b, ctx, root, dispatch, frame, frame, candidate, sbt, 0,
                MESA_SHADER_INTERSECTION, ctx->phase_base + RT_RESUME,
                rt_load(b, frame, RT_COMMITTED_T, 1, 32), nir_imm_int(b, 0));
      nir_push_else(b, NULL);
      rt_select_phase(b, ctx, root, frame, RT_FINISH);
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
      break;
   }
   case RT_TRIANGLE_COMMIT: {
      nir_def *action = rt_load(b, frame, RT_ACTION, 1, 32);
      nir_def *accepted = nir_ine_imm(b, action, PANVK_NIR_RT_IGNORE);
      nir_push_if(b, accepted);
      rt_commit(b, ctx, frame, rt_load(b, candidate, 0, 1, 32), rt_tri_kind(b, candidate), false, NULL);
      nir_pop_if(b, NULL);
      nir_def *next = nir_bcsel(b, nir_iand(b, accepted, rt_terminated(b, frame, action)),
         nir_imm_int(b, ctx->phase_base + RT_FINISH), nir_imm_int(b, ctx->phase_base + RT_RESUME));
      rt_select(b, root, frame, next);
      break;
   }
   case RT_REPORT_START: {
      nir_def *ray = rt_ray(b, frame);
      nir_def *record = rt_load(b, frame, RT_RECORD, 1, 64);
      nir_def *t = rt_load(b, frame, RT_REPORT_T, 1, 32);
      uint32_t saved = b->fp_math_ctrl;
      b->fp_math_ctrl = nir_fp_no_fast_math;
      nir_def *valid = nir_iand(b, nir_fge(b, t, rt_load(b, ray, RT_TMIN, 1, 32)),
                                   nir_fge(b, rt_load(b, ray, RT_COMMITTED_T, 1, 32), t));
      b->fp_math_ctrl = saved;
      nir_push_if(b, valid);
      nir_def *sbt = rt_sbt(b, dispatch, 40, 56,
         nir_iand_imm(b, rt_load(b, ray, RT_STATUS, 1, 32), 0x0fffffff));
      nir_push_if(b, rt_opaque(b, ray, record));
      rt_store(b, frame, RT_ACTION, nir_imm_int(b, PANVK_NIR_RT_ACCEPT));
      rt_select_phase(b, ctx, root, frame, RT_REPORT_COMMIT);
      nir_push_else(b, NULL);
      rt_invoke(b, ctx, root, dispatch, frame, ray, record, sbt, 8,
                MESA_SHADER_ANY_HIT, ctx->phase_base + RT_REPORT_COMMIT,
                t, rt_load(b, frame, RT_REPORT_KIND, 1, 32));
      nir_pop_if(b, NULL);
      nir_push_else(b, NULL);
      rt_store(b, frame, RT_REPORT_ACCEPTED, nir_imm_int(b, 0));
      rt_select(b, root, frame, rt_load(b, frame, RT_REPORT_RESUME, 1, 32));
      nir_pop_if(b, NULL);
      break;
   }
   case RT_REPORT_COMMIT: rt_report_commit(b, ctx, root, frame); break;
   case RT_FINISH: {
      nir_def *type = nir_ushr_imm(b, rt_load(b, committed, 16, 1, 32), 29);
      nir_def *hit = nir_ior(b, nir_ieq_imm(b, type, 5), nir_ieq_imm(b, type, 7));
      nir_push_if(b, hit);
      nir_push_if(b, nir_inot(b, nir_test_mask(b, rt_load(b, frame, RT_FLAGS, 1, 32),
                                               SpvRayFlagsSkipClosestHitShaderKHRMask)));
      nir_push_if(b, nir_ieq_imm(b, type, 5));
      rt_store(b, frame, RT_COMMITTED_ATTR, rt_load(b, committed, 4, 2, 32));
      nir_pop_if(b, NULL);
      nir_def *sbt = rt_sbt(b, dispatch, 40, 56,
         nir_iand_imm(b, rt_load(b, frame, RT_STATUS, 1, 32), 0x0fffffff));
      rt_invoke(b, ctx, root, dispatch, frame, frame, committed, sbt, 16,
                MESA_SHADER_CLOSEST_HIT, ctx->phase_base + RT_TRACE_RETURN,
                rt_load(b, frame, RT_COMMITTED_T, 1, 32),
                nir_bcsel(b, nir_ieq_imm(b, type, 5), rt_tri_kind(b, committed),
                             rt_load(b, frame, RT_GENERATED_KIND, 1, 32)));
      nir_push_else(b, NULL);
      rt_select_phase(b, ctx, root, frame, RT_TRACE_RETURN);
      nir_pop_if(b, NULL);
      nir_push_else(b, NULL);
      sbt = rt_sbt(b, dispatch, 16, 32, rt_load(b, frame, RT_MISS, 1, 32));
      rt_invoke(b, ctx, root, dispatch, frame, frame, nir_imm_int64(b, 0), sbt, 0,
                MESA_SHADER_MISS, ctx->phase_base + RT_TRACE_RETURN,
                rt_load(b, frame, RT_TMAX, 1, 32), nir_imm_int(b, 0));
      nir_pop_if(b, NULL);
      break;
   }
   case RT_TRACE_RETURN: rt_return(b, root, frame, false); break;
   default: UNREACHABLE("Invalid RT phase");
   }
}

static nir_def *
rt_native_target(nir_builder *b, struct rt_context *ctx, nir_def *dispatch,
                  nir_def *sbt, unsigned slot, mesa_shader_stage stage)
{
   nir_def *token = rt_load(b, sbt, slot, 2, 32);
   if (ctx->info->direct_handles) {
      unsigned targets = 0;
      for (unsigned i = 0; i < ctx->info->stage_count; i++)
         targets += ctx->info->stages[i].stage == stage;
      nir_def *index = nir_imm_int(b, 0);
      if (!ctx->info->disable_lto && targets > 1 &&
          ctx->info->stage_count <= PANVK_RT_NATIVE_SHADER_INDEX_MASK &&
          !(ctx->info->flags & (VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR |
                               VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT))) {
         nir_def *indices = rt_load_constant(b, sbt, 24, 1, 64);
         index = nir_iand_imm(b, nir_u2u32(b, nir_ushr_imm(b, indices,
            (slot / 8) * PANVK_RT_NATIVE_SHADER_INDEX_BITS)), PANVK_RT_NATIVE_SHADER_INDEX_MASK);
      }
      return nir_vec3(b, nir_channel(b, token, 0), nir_channel(b, token, 1), index);
   }

   token = nir_channel(b, token, 0);

   nir_variable *lo = nir_local_variable_create(b->impl, glsl_uint_type(), "rt_lo");
   nir_variable *hi = nir_local_variable_create(b->impl, glsl_uint_type(), "rt_hi");
   nir_def *table = rt_load_dispatch(b, ctx, dispatch, 128, 1, 64);
   nir_store_var(b, lo, nir_imm_int(b, 0), 1);
   nir_store_var(b, hi, nir_imm_int(b, ctx->info->stage_count), 1);
   nir_push_loop(b);
   nir_def *l = nir_load_var(b, lo), *h = nir_load_var(b, hi);
   nir_push_if(b, nir_uge(b, l, h));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_def *mid = nir_iadd(b, l, nir_ushr_imm(b, nir_isub(b, h, l), 1));
   nir_def *entry = nir_iadd(b, table, nir_ishl_imm(b, nir_u2u64(b, mid), 4));
   nir_def *less = nir_ult(b, rt_load(b, entry, 0, 1, 32), token);
   nir_store_var(b, lo, nir_bcsel(b, less, nir_iadd_imm(b, mid, 1), l), 1);
   nir_store_var(b, hi, nir_bcsel(b, less, h, mid), 1);
   nir_pop_loop(b, NULL);
   nir_def *index = nir_load_var(b, lo);
   nir_push_if(b, nir_ult_imm(b, index, ctx->info->stage_count));
   entry = nir_iadd(b, table, nir_ishl_imm(b, nir_u2u64(b, index), 4));
   nir_def *valid = nir_iand(b, nir_ieq(b, token, rt_load(b, entry, 0, 1, 32)),
                               nir_ieq_imm(b, rt_load(b, entry, 4, 1, 32), stage));
   nir_def *target = nir_bcsel(b, valid, rt_load(b, entry, 8, 2, 32), nir_imm_ivec2(b, 0, 0));
   nir_push_else(b, NULL);
   nir_def *zero = nir_imm_ivec2(b, 0, 0);
   nir_pop_if(b, NULL);
   target = nir_if_phi(b, target, zero);
   return nir_vec3(b, nir_channel(b, target, 0), nir_channel(b, target, 1), nir_imm_int(b, 0));
}

static nir_def *
rt_native_child(nir_builder *b, nir_def *frame)
{
   return nir_iadd(b, frame, nir_u2u64(b, nir_load_rt_frame_size_pan(b)));
}

static nir_def *
rt_native_valid(nir_builder *b, struct rt_context *ctx, nir_def *dispatch,
                 nir_def *root, nir_def *frame,
                 nir_def *target)
{
   return nir_uge(b, rt_load_dispatch(b, ctx, dispatch, 112, 1, 64),
      nir_iadd(b, nir_isub(b, frame, root), nir_u2u64(b, nir_channel(b, target, 1))));
}

static nir_def *
rt_native_queries(nir_builder *b, const struct rt_part *part, bool trace)
{
   nir_def *layout = nir_load_rt_arg_pan(b, 32, .base = 12);
   nir_def *base = nir_iand_imm(b, layout, 0xffff);
   nir_def *incoming = trace ? nir_ishl_imm(b, base, 16)
                            : nir_iand_imm(b, layout, 0xffff0000);
   return nir_ior(b, incoming, nir_iadd_imm(b, base, 1 + part->stage->ray_query_slots));
}

static nir_def *
rt_native_call(nir_builder *b, struct rt_context *ctx, nir_def *dispatch,
               nir_def *root, nir_def *frame, nir_def *target,
               nir_def *ray, nir_def *record, nir_def *sbt, nir_def *payload,
               nir_def *depth, nir_def *t, nir_def *kind,
               mesa_shader_stage stage, nir_def *queries)
{
   nir_def *record_words = nir_unpack_64_2x32(b, record);
   nir_def *sbt_words = nir_unpack_64_2x32(b, nir_iadd_imm(b, sbt, 32));
   nir_def *hit[] = {
      nir_channel(b, record_words, 0), nir_channel(b, record_words, 1),
      nir_channel(b, sbt_words, 0), nir_channel(b, sbt_words, 1),
      t, kind, depth, queries,
   };
   nir_def *args = nir_vec4(b, frame, payload, root, ray);
   nir_def *params = nir_vec(b, hit, ARRAY_SIZE(hit));
   nir_def *stack = frame;
   nir_def *identity = nir_channel(b, target, 2);
   target = nir_channel(b, target, 0);
   nir_def *result;
   if (stage == MESA_SHADER_ANY_HIT || stage == MESA_SHADER_INTERSECTION) {
      result = nir_rt_call_result_pan(b, stage == MESA_SHADER_ANY_HIT ? 1 : 3,
         target, args, params, stack, identity, .base = stage, .rt_arg_mask = 0xffff);
   } else {
      nir_rt_call_pan(b, target, args, params, stack, identity, .base = stage, .rt_arg_mask = 0xffff);
      result = nir_imm_int(b, PANVK_NIR_RT_ACCEPT);
   }
   return result;
}

static nir_def *
rt_native_invoke(nir_builder *b, struct rt_context *ctx, nir_def *root,
                 nir_def *dispatch, nir_def *child, nir_def *ray,
                 nir_def *record, nir_def *sbt, unsigned slot,
                 mesa_shader_stage stage, nir_def *t, nir_def *kind,
                 nir_def *payload, nir_def *depth, nir_def *queries)
{
   nir_def *target = rt_native_target(b, ctx, dispatch, sbt, slot, stage);
   nir_push_if(b, nir_iand(b, nir_ine_imm(b, nir_channel(b, target, 0), 0),
                            rt_native_valid(b, ctx, dispatch, root, child, target)));
   nir_def *action = rt_native_call(b, ctx, dispatch, root, child, target, ray, record, sbt,
                  payload, depth, t, kind, stage, queries);
   nir_push_else(b, NULL);
   nir_def *accept = stage == MESA_SHADER_INTERSECTION
      ? nir_vec3(b, nir_imm_int(b, PANVK_NIR_RT_IGNORE), kind, t)
      : nir_imm_int(b, PANVK_NIR_RT_ACCEPT);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, action, accept);
}

static void
rt_native_traverse(nir_builder *b, struct rt_context *ctx, nir_def *root,
                   nir_def *dispatch, nir_def *ray, nir_def *child, nir_def *queries,
                   const struct panvk_nir_rt_trace *trace,
                   nir_def *payload, nir_def *depth)
{
   nir_def *state = rt_state(b, ctx, ray);
   nir_def *as = rt_root_node(b, trace->accel_struct);
   nir_def *candidate = nir_iadd_imm(b, state, 0x64);
   nir_def *committed = nir_iadd_imm(b, state, 0x24);
   nir_variable *status_var = nir_local_variable_create(b->impl, glsl_uint_type(), "rt_status");
   nir_variable *t_var = nir_local_variable_create(b->impl, glsl_float_type(), "rt_t");
   nir_variable *kind_var = nir_local_variable_create(b->impl, glsl_uint_type(), "rt_kind");
   rt_store(b, committed, 16, nir_imm_int(b, 0));
   nir_store_var(b, kind_var, nir_imm_int(b, 0), 1);
   rt_barrier(b);
   nir_def *out = nir_rt_trace_begin_pan(b, state, as,
      trace->origin, trace->direction, trace->tmin, trace->tmax,
      nir_ior(b, trace->flags, nir_ishl_imm(b, trace->cull_mask, 24)),
      panvk_nir_rt_pack_sbt(b, trace->sbt_offset, trace->sbt_stride, trace->miss_index));
   rt_barrier(b);
   nir_store_var(b, status_var, nir_channel(b, out, 0), 1);
   nir_store_var(b, t_var, nir_channel(b, out, 1), 1);
   nir_push_loop(b);
   nir_def *status = nir_load_var(b, status_var);
   nir_def *code = nir_ushr_imm(b, status, 29);
   nir_def *sbt = rt_sbt(b, dispatch, 40, 56, nir_iand_imm(b, status, 0x0fffffff));
   nir_push_if(b, nir_ieq_imm(b, code, 1));
   rt_store(b, ray, RT_ATTR, rt_load(b, candidate, 4, 2, 32));
   nir_def *action = rt_native_invoke(b, ctx, root, dispatch, child, ray, candidate, sbt, 8,
                    MESA_SHADER_ANY_HIT, rt_load(b, candidate, 0, 1, 32),
                    rt_tri_kind(b, candidate), payload, depth, queries);
   nir_def *accept = nir_ine_imm(b, action, PANVK_NIR_RT_IGNORE);
   nir_push_if(b, accept);
   nir_def *candidate_t = rt_load(b, candidate, 0, 1, 32);
   rt_commit(b, ctx, ray, candidate_t, rt_tri_kind(b, candidate), false, state);
   nir_store_var(b, t_var, candidate_t, 1);
   nir_pop_if(b, NULL);
   nir_push_if(b, nir_iand(b, accept,
      nir_ior(b, nir_ieq_imm(b, action, PANVK_NIR_RT_TERMINATE),
                 nir_test_mask(b, trace->flags, SpvRayFlagsTerminateOnFirstHitKHRMask))));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, code, 2));
   rt_store(b, ray, RT_STATUS, status);
   nir_def *reported = rt_native_invoke(b, ctx, root, dispatch, child, ray, candidate, sbt, 0,
                    MESA_SHADER_INTERSECTION, nir_load_var(b, t_var),
                    nir_load_var(b, kind_var), payload, depth, queries);
   nir_store_var(b, t_var, nir_channel(b, reported, 2), 1);
   nir_store_var(b, kind_var, nir_channel(b, reported, 1), 1);
   nir_push_if(b, nir_ieq_imm(b, nir_channel(b, reported, 0), PANVK_NIR_RT_TERMINATE));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_push_else(b, NULL);
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   rt_barrier(b);
   out = nir_rt_trace_resume_pan(b, state, as, rt_load(b, candidate, 12, 2, 32));
   rt_barrier(b);
   nir_store_var(b, status_var, nir_channel(b, out, 0), 1);
   nir_store_var(b, t_var, nir_channel(b, out, 1), 1);
   nir_pop_loop(b, NULL);
   nir_def *type = nir_ushr_imm(b, rt_load(b, committed, 16, 1, 32), 29);
   nir_push_if(b, nir_ior(b, nir_ieq_imm(b, type, 5), nir_ieq_imm(b, type, 7)));
   nir_push_if(b, nir_inot(b, nir_test_mask(b, trace->flags,
                                           SpvRayFlagsSkipClosestHitShaderKHRMask)));
   nir_push_if(b, nir_ieq_imm(b, type, 5));
   rt_store(b, ray, RT_COMMITTED_ATTR, rt_load(b, committed, 4, 2, 32));
   nir_pop_if(b, NULL);
   sbt = rt_sbt(b, dispatch, 40, 56, nir_iand_imm(b, nir_load_var(b, status_var), 0x0fffffff));
   rt_native_invoke(b, ctx, root, dispatch, child, ray, committed, sbt, 16,
      MESA_SHADER_CLOSEST_HIT, nir_load_var(b, t_var),
      nir_bcsel(b, nir_ieq_imm(b, type, 5), rt_tri_kind(b, committed),
                nir_load_var(b, kind_var)), payload, depth, queries);
   nir_pop_if(b, NULL);
   nir_push_else(b, NULL);
   sbt = rt_sbt(b, dispatch, 16, 32, trace->miss_index);
   rt_native_invoke(b, ctx, root, dispatch, child, ray, nir_imm_int64(b, 0), sbt, 0,
                    MESA_SHADER_MISS, trace->tmax, nir_imm_int(b, 0), payload, depth, queries);
   nir_pop_if(b, NULL);
}

static bool
rt_native_lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct rt_part *part = data;
   struct rt_context *ctx = part->ctx;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *frame = rt_param(b, 0), *dispatch = rt_param(b, 1), *root = rt_param(b, 2);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ray_t_max:
      assert(part->hit_t);
      nir_def_replace(&intr->def, nir_load_var(b, part->hit_t));
      return true;
   case nir_intrinsic_trace_ray: {
      nir_def *child = nir_iadd_imm(b, frame, part->stage->ray_offset);
      nir_def *depth = nir_iadd_imm(b, rt_stage_load(b, part, frame, RT_DEPTH, 1, 32), 1);
      nir_push_if(b, nir_ule_imm(b, depth, ctx->info->max_recursion_depth));
      uint32_t flags = 0;
      if (ctx->info->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR)
         flags |= SpvRayFlagsSkipTrianglesKHRMask;
      if (ctx->info->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR)
         flags |= SpvRayFlagsSkipAABBsKHRMask;
      const struct panvk_nir_rt_trace trace = {
         .accel_struct = intr->src[0].ssa,
         .flags = nir_ior_imm(b, intr->src[1].ssa, flags),
         .cull_mask = nir_iand_imm(b, intr->src[2].ssa, 255),
         .sbt_offset = nir_iand_imm(b, intr->src[3].ssa, 15),
         .sbt_stride = nir_iand_imm(b, intr->src[4].ssa, 15),
         .miss_index = nir_iand_imm(b, intr->src[5].ssa, 65535),
         .origin = intr->src[6].ssa,
         .tmin = intr->src[7].ssa,
         .direction = intr->src[8].ssa,
         .tmax = intr->src[9].ssa,
      };
      rt_store(b, child, RT_AS, trace.accel_struct);
      rt_store(b, child, RT_FLAGS, trace.flags);
      rt_store(b, child, RT_CULL, trace.cull_mask);
      rt_store(b, child, RT_ORIGIN, trace.origin);
      rt_store(b, child, RT_TMIN, trace.tmin);
      rt_store(b, child, RT_DIRECTION, trace.direction);
      rt_store(b, child, RT_TMAX, trace.tmax);
      rt_native_traverse(b, ctx, root, dispatch, child, rt_native_child(b, frame),
                         rt_native_queries(b, part, true), &trace,
                         nir_u2u64(b, intr->src[10].ssa), depth);
      nir_pop_if(b, NULL);
      break;
   }
   case nir_intrinsic_execute_callable: {
      nir_def *sbt = rt_sbt(b, dispatch, 64, 80, intr->src[0].ssa);
      nir_def *target = rt_native_target(b, ctx, dispatch, sbt, 0, MESA_SHADER_CALLABLE);
      nir_def *child = rt_native_child(b, frame);
      nir_push_if(b, nir_iand(b, nir_ine_imm(b, nir_channel(b, target, 0), 0),
                            rt_native_valid(b, ctx, dispatch, root, child, target)));
      rt_native_call(b, ctx, dispatch, root, child, target,
                     rt_stage_load(b, part, frame, RT_RAY, 1, 64),
                     rt_stage_load(b, part, frame, RT_RECORD, 1, 64),
                     sbt, nir_u2u64(b, intr->src[1].ssa),
                     rt_stage_load(b, part, frame, RT_DEPTH, 1, 32),
                     rt_stage_load(b, part, frame, RT_TMAX, 1, 32),
                     rt_stage_load(b, part, frame, RT_HIT_KIND, 1, 32),
                     MESA_SHADER_CALLABLE, rt_native_queries(b, part, false));
      nir_pop_if(b, NULL);
      break;
   }
   case nir_intrinsic_report_ray_intersection: {
      nir_def *ray = rt_stage_load(b, part, frame, RT_RAY, 1, 64), *t = intr->src[0].ssa, *kind = intr->src[1].ssa;
      nir_def *record = rt_stage_load(b, part, frame, RT_RECORD, 1, 64);
      uint32_t saved = b->fp_math_ctrl;
      b->fp_math_ctrl = nir_fp_no_fast_math;
      nir_def *valid = nir_iand(b, nir_fge(b, t, rt_load(b, ray, RT_TMIN, 1, 32)),
                                  nir_fge(b, nir_load_var(b, part->hit_t), t));
      b->fp_math_ctrl = saved;
      nir_push_if(b, valid);
      nir_push_if(b, nir_inot(b, rt_opaque(b, ray, record)));
      nir_def *sbt = rt_sbt(b, dispatch, 40, 56,
         nir_iand_imm(b, rt_load(b, ray, RT_STATUS, 1, 32), 0x0fffffff));
      nir_def *any_hit_action = rt_native_invoke(b, ctx, root, dispatch, rt_native_child(b, frame), ray, record, sbt, 8,
                       MESA_SHADER_ANY_HIT, t, kind,
                       rt_stage_load(b, part, frame, RT_PAYLOAD, 1, 64),
                       rt_stage_load(b, part, frame, RT_DEPTH, 1, 32),
                       rt_native_queries(b, part, false));
      nir_push_else(b, NULL);
      nir_def *opaque_action = nir_imm_int(b, PANVK_NIR_RT_ACCEPT);
      nir_pop_if(b, NULL);
      nir_def *action = nir_if_phi(b, any_hit_action, opaque_action);
      nir_def *accepted = nir_ine_imm(b, action, PANVK_NIR_RT_IGNORE);
      nir_push_if(b, accepted);
      nir_def *state = panvk_nir_rt_state_addr(b, root,
         nir_ushr_imm(b, nir_load_rt_arg_pan(b, 32, .base = 12), 16));
      rt_commit(b, ctx, ray, t, kind, true, state);
      nir_store_var(b, part->hit_t, t, 1);
      nir_store_var(b, part->hit_kind, kind, 1);
      nir_store_var(b, part->action, nir_imm_int(b, PANVK_NIR_RT_ACCEPT), 1);
      nir_push_if(b, rt_terminated(b, ray, action));
      nir_store_var(b, part->action, nir_imm_int(b, PANVK_NIR_RT_TERMINATE), 1);
      nir_jump(b, nir_jump_return);
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
      nir_push_else(b, NULL);
      nir_def *rejected = nir_imm_false(b);
      nir_pop_if(b, NULL);
      nir_def_rewrite_uses(&intr->def, nir_if_phi(b, accepted, rejected));
      break;
   }
   case nir_intrinsic_ignore_ray_intersection:
   case nir_intrinsic_terminate_ray:
   case nir_intrinsic_accept_ray_intersection: {
      nir_store_var(b, part->action, nir_imm_int(b,
         intr->intrinsic == nir_intrinsic_ignore_ray_intersection ? PANVK_NIR_RT_IGNORE :
         intr->intrinsic == nir_intrinsic_terminate_ray ? PANVK_NIR_RT_TERMINATE : PANVK_NIR_RT_ACCEPT), 1);
      nir_instr *last = nir_block_last_instr(intr->instr.block);
      if (last->type == nir_instr_type_jump) {
         assert(nir_instr_as_jump(last)->type == nir_jump_halt);
         nir_instr_as_jump(last)->type = nir_jump_return;
         nir_instr_remove(&intr->instr);
      } else {
         b->cursor = nir_instr_remove(&intr->instr);
         nir_jump(b, nir_jump_return);
      }
      return true;
   }
   default: return false;
   }
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
rt_native_param(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_uniform)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   const struct rt_context *ctx = data;
   unsigned index = nir_src_as_uint(intr->src[0]);
   nir_def *value = index == 1
      ? nir_load_push_constant(b, 1, 64, nir_imm_int(b, 0),
           .base = ctx->info->dispatch_sysval_offset, .range = 8)
      : nir_load_rt_arg_pan(b, 64, .base = index);
   nir_def_replace(&intr->def, value);
   return true;
}

static void
rt_native_finish(nir_shader *nir)
{
   nir->info.workgroup_size[0] = 16;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;
   nir_lower_vars_to_ssa(nir);
   nir_opt_dce(nir);
   nir_index_ssa_defs(nir_shader_get_entrypoint(nir));
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   nir_validate_shader(nir, "panvk native RT function");
}

static nir_def *
rt_flush_denorm(nir_builder *b, nir_def *value)
{
   uint64_t exponent = value->bit_size == 16 ? 0x7c00 :
                       value->bit_size == 32 ? 0x7f800000 :
                                              UINT64_C(0x7ff0000000000000);
   uint64_t sign = UINT64_C(1) << (value->bit_size - 1);
   return nir_bcsel(b, nir_ieq_imm(b, nir_iand_imm(b, value, exponent), 0),
                   nir_iand_imm(b, value, sign), value);
}

static bool
rt_lower_denorm(nir_builder *b, nir_alu_instr *alu, void *data)
{
   unsigned mode = b->shader->info.float_controls_execution_mode;
   const nir_op_info *op = &nir_op_infos[alu->op];
   if (alu->op == nir_op_unpack_half_2x16 && nir_is_denorm_flush_to_zero(mode, 16)) {
      b->cursor = nir_before_instr(&alu->instr);
      nir_def *value = nir_f2f32(b, rt_flush_denorm(b,
         nir_unpack_32_2x16(b, nir_ssa_for_alu_src(b, alu, 0))));
      nir_def_rewrite_uses(&alu->def, value);
      nir_instr_remove(&alu->instr);
      return true;
   }
   if (nir_alu_type_get_base_type(op->output_type) == nir_type_float &&
       nir_is_denorm_flush_to_zero(mode, alu->def.bit_size)) {
      b->cursor = nir_after_instr(&alu->instr);
      nir_def *value = rt_flush_denorm(b, &alu->def);
      nir_def_rewrite_uses_after(&alu->def, value);
      return true;
   }
   return false;
}

static bool
rt_lower_collective_denorm(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_reduce &&
       intr->intrinsic != nir_intrinsic_inclusive_scan &&
       intr->intrinsic != nir_intrinsic_exclusive_scan)
      return false;
   const nir_op_info *op = &nir_op_infos[nir_intrinsic_reduction_op(intr)];
   if (nir_alu_type_get_base_type(op->output_type) != nir_type_float ||
       !nir_is_denorm_flush_to_zero(b->shader->info.float_controls_execution_mode,
                                   intr->def.bit_size))
      return false;
   b->cursor = nir_after_instr(&intr->instr);
   nir_def_rewrite_uses_after(&intr->def, rt_flush_denorm(b, &intr->def));
   return true;
}

static void
rt_native_private_io(struct rt_context *ctx, struct rt_stage *stage)
{
   nir_shader *nir = stage->nir;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_rt_return_pan ||
             intr->intrinsic == nir_intrinsic_rt_return_value_pan)
            stage->frame_size = MAX2(stage->frame_size, nir_intrinsic_stack_size(intr));
      }
   }
   panvk_nir_opt_rt_private_io(nir);
   nir_variable_mode modes = nir_var_shader_temp | nir_var_function_temp;
   nir_remove_dead_variables(nir, modes, NULL);
   nir->scratch_size = 0;
   nir_lower_vars_to_explicit_types(nir, modes, glsl_get_natural_size_align_bytes);
   struct rt_part part = {.ctx = ctx, .stage = stage, .private_io = true};
   nir_shader_instructions_pass(nir, rt_lower_deref, nir_metadata_none, &part);
   nir_fixup_deref_modes(nir);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_deref) {
               nir_deref_instr *deref = nir_instr_as_deref(instr);
               if (deref->modes == nir_var_mem_global) {
                  deref->def.bit_size = 64;
                  deref->def.num_components = 1;
               }
            }
         }
      }
   }
   nir_lower_explicit_io(nir, nir_var_mem_global, nir_address_format_64bit_global);
   stage->frame_size += ALIGN_POT(nir->scratch_size, 16);
   nir->scratch_size = 0;
   nir_remove_dead_variables(nir, modes | nir_var_shader_call_data | nir_var_ray_hit_attrib, NULL);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_rt_return_pan ||
                intr->intrinsic == nir_intrinsic_rt_return_value_pan)
               nir_intrinsic_set_stack_size(intr, stage->frame_size);
         }
      }
   }
   rt_native_finish(nir);
   nir_opt_copy_prop(nir);
   nir_opt_shrink_vectors(nir, true);
   nir_opt_dce(nir);
   nir_validate_shader(nir, "RT explicit private IO ranges");
}

static void
rt_native_create(void *mem_ctx, struct rt_context *ctx,
                 struct panvk_nir_rt_pipeline_result *result)
{
   result->functions = ralloc_array(mem_ctx, nir_shader *, ctx->info->stage_count);
   for (unsigned i = 0; i < ctx->info->stage_count; i++) {
      struct rt_stage *stage = &ctx->stages[i];
      struct rt_part part = {.ctx = ctx, .stage = stage};
      nir_shader *nir = stage->nir;
      nir_builder b = nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
      if (stage->stage == MESA_SHADER_ANY_HIT || stage->stage == MESA_SHADER_INTERSECTION) {
         part.action = nir_local_variable_create(b.impl, glsl_uint_type(), "rt_action");
         nir_store_var(&b, part.action, nir_imm_int(&b,
            stage->stage == MESA_SHADER_ANY_HIT ? PANVK_NIR_RT_ACCEPT : PANVK_NIR_RT_IGNORE), 1);
      }
      if (stage->stage == MESA_SHADER_INTERSECTION) {
         part.hit_kind = nir_local_variable_create(b.impl, glsl_uint_type(), "rt_kind");
         part.hit_t = nir_local_variable_create(b.impl, glsl_float_type(), "rt_t");
         nir_store_var(&b, part.hit_kind, nir_load_rt_arg_pan(&b, 32, .base = 10), 1);
         nir_store_var(&b, part.hit_t, nir_load_rt_arg_pan(&b, 32, .base = 9), 1);
      }
      nir_shader_intrinsics_pass(nir, rt_native_lower, nir_metadata_none, &part);
      nir_lower_returns(nir);
      nir_opt_dead_cf(nir);
      b.cursor = nir_after_impl(nir_shader_get_entrypoint(nir));
      nir_def *target = nir_load_rt_arg_pan(&b, 32, .base = 4);
      nir_def *stack = nir_load_rt_arg_pan(&b, 64, .base = 3);
      unsigned mode = (stage->stage == MESA_SHADER_RAYGEN) | (ctx->info->tls ? 2 : 0);
      if (part.action) {
         nir_def *value = nir_load_var(&b, part.action);
         if (part.hit_t)
            value = nir_vec3(&b, value, nir_load_var(&b, part.hit_kind), nir_load_var(&b, part.hit_t));
         nir_rt_return_value_pan(&b, target, stack, value, .base = mode,
                                 .stack_size = stage->frame_size);
      } else {
         nir_rt_return_pan(&b, target, stack, .base = mode,
                           .stack_size = stage->frame_size);
      }
      nir_shader_intrinsics_pass(nir, rt_native_param, nir_metadata_none, ctx);
      nir_shader_alu_pass(nir, rt_lower_denorm, nir_metadata_control_flow, NULL);
      nir_shader_intrinsics_pass(nir, rt_lower_collective_denorm,
                                 nir_metadata_control_flow, NULL);
      nir->info.float_controls_execution_mode &=
         ~(FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP16 |
           FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32 |
           FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP64);
      rt_native_finish(nir);
      ralloc_steal(mem_ctx, nir);
      result->functions[i] = nir;
      result->per_stage_stack_size[i] = stage->frame_size;
   }
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, ctx->info->options,
                                                  "panvk native RT entry");
   nir_def *dispatch = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0),
      .base = ctx->info->dispatch_sysval_offset, .range = 8);
   nir_def *launch = nir_load_global_invocation_id(&b, 32);
   nir_def *size = rt_load_dispatch(&b, ctx, dispatch, 88, 3, 32);
   nir_def *valid = nir_imm_true(&b);
   for (unsigned c = 0; c < 3; c++)
      valid = nir_iand(&b, valid, nir_ult(&b, nir_channel(&b, launch, c), nir_channel(&b, size, c)));
   nir_push_if(&b, valid);
   nir_def *warps = nir_iadd_imm(&b, nir_load_warp_max_id_arm(&b), 1);
   nir_def *index = nir_iadd(&b, nir_imul_imm(&b,
      nir_iadd(&b, nir_imul(&b, nir_load_core_id(&b), warps), nir_load_warp_id_arm(&b)), 16),
      nir_load_subgroup_invocation(&b));
   nir_def *stack_base = rt_load_dispatch(&b, ctx, dispatch, 104, 1, 64);
   nir_def *scale = rt_load_dispatch(&b, ctx, dispatch, 152, 1, 64);
   nir_def *root, *local_size;
   if (ctx->info->tls) {
      nir_def *tls_offset = rt_load_dispatch(&b, ctx, dispatch, 144, 1, 32);
      root = nir_iadd(&b, nir_load_scratch_base_ptr(&b, 1, 64),
                        nir_u2u64(&b, tls_offset));
      local_size = nir_imm_int64(&b, 0);
   } else {
      nir_def *stride = rt_load_dispatch(&b, ctx, dispatch, 112, 1, 64);
      root = nir_iadd(&b, stack_base, nir_imul(&b, nir_u2u64(&b, index), stride));
      local_size = nir_imul(&b, scale, stride);
   }
   nir_def *external_root = nir_iadd(&b, nir_iadd(&b, stack_base, local_size),
      nir_ishl_imm(&b, nir_u2u64(&b, index), 7));
   nir_def *sbt = rt_load_constant(&b, dispatch, 0, 1, 64);
   if (!ctx->info->tls)
      rt_store(&b, root, RT_LAUNCH, launch);
   rt_store(&b, root, PANVK_RT_EXTERNAL_ROOT_OFFSET, external_root);
   rt_store(&b, root, PANVK_RT_EXTERNAL_SCALE_OFFSET,
      rt_load_dispatch(&b, ctx, dispatch, 152, 1, 64));
   nir_def *target = rt_native_target(&b, ctx, dispatch, sbt, 0, MESA_SHADER_RAYGEN);
   nir_def *frame = nir_iadd_imm(&b, root, RT_NATIVE_HEADER);
   nir_push_if(&b, nir_iand(&b, nir_ine_imm(&b, nir_channel(&b, target, 0), 0),
                           rt_native_valid(&b, ctx, dispatch, root, frame, target)));
   rt_native_call(&b, ctx, dispatch, root, frame, target, root,
                  nir_imm_int64(&b, 0), sbt, nir_imm_int64(&b, 0),
                  nir_imm_int(&b, 0), nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                  MESA_SHADER_RAYGEN, nir_imm_int(&b, 0));
   nir_pop_if(&b, NULL);
   nir_pop_if(&b, NULL);
   rt_native_finish(b.shader);
   ralloc_steal(mem_ctx, b.shader);
   result->nir = b.shader;
   panvk_nir_opt_rt_lto(ctx->info, result);
   panvk_nir_opt_rt_calls(ctx->info, result);
   for (unsigned i = 0; i < ctx->info->stage_count; i++) {
      rt_native_private_io(ctx, &ctx->stages[i]);
      result->per_stage_stack_size[i] = ctx->stages[i].frame_size;
   }
   panvk_nir_opt_rt_calls(ctx->info, result);
   if (ctx->info->tls) {
      for (unsigned i = 0; i < ctx->info->stage_count; i++)
         panvk_nir_lower_rt_tls(result->functions[i]);
      panvk_nir_lower_rt_tls(result->nir);
      panvk_nir_lower_rt_launch_id(ctx->info, result);
   }
   for (unsigned i = 0; i < ctx->info->stage_count; i++)
      panvk_nir_lower_rt_call_barriers(result->functions[i]);
   panvk_nir_lower_rt_call_barriers(result->nir);
}

VkResult
panvk_nir_rt_pipeline_create(void *mem_ctx,
                            const struct panvk_nir_rt_pipeline_info *info,
                            struct panvk_nir_rt_pipeline_result *result)
{
   memset(result, 0, sizeof(*result));
   if (info->max_hit_attribute_size > 32)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   void *tmp = ralloc_context(NULL);
   struct rt_context ctx = {.info = info};
   ctx.stages = rzalloc_array(tmp, struct rt_stage, info->stage_count);
   result->per_stage_stack_size = rzalloc_array(mem_ctx, uint32_t, info->stage_count);
   uint32_t next_pc = info->stage_count + 1;
   uint32_t max_scratch = 0;
   uint32_t ray_query_slots = 0;
   nir_lower_shader_calls_options call_options = {
      .address_format = nir_address_format_64bit_global,
      .stack_alignment = 16,
      .should_remat_callback = rt_remat,
   };
   for (unsigned i = 0; i < info->stage_count; i++) {
      struct rt_stage *stage = &ctx.stages[i];
      stage->id = info->stages[i].id;
      stage->stage = info->stages[i].stage;
      stage->nir = nir_shader_clone(tmp, info->stages[i].nir);
      nir_shader *nir = stage->nir;
      nir_lower_returns(nir);
      nir_inline_functions(nir);
      nir_cleanup_functions(nir);
      if (info->native) {
         nir_lower_global_vars_to_local(nir);
         nir_lower_vars_to_ssa(nir);
         bool progress;
         do {
            progress = nir_opt_copy_prop(nir);
            progress |= nir_opt_deref(nir);
            progress |= nir_opt_copy_prop_vars(nir);
            progress |= nir_opt_dce(nir);
         } while (progress);
         nir_opt_combine_stores(nir, nir_var_shader_temp | nir_var_function_temp |
                                    nir_var_shader_call_data | nir_var_ray_hit_attrib);
      }
      if (info->native)
         stage->ray_query_slots = panvk_per_arch(nir_lower_rt_ray_queries)(nir, RT_HEADER);
      else
         ray_query_slots = panvk_per_arch(nir_lower_ray_queries)(nir, ray_query_slots);
      nir_lower_system_values(nir);
      nir_variable_mode modes = nir_var_shader_temp | nir_var_function_temp |
         nir_var_shader_call_data | nir_var_ray_hit_attrib;
      nir_lower_vars_to_explicit_types(nir, modes, glsl_get_natural_size_align_bytes);
      struct rt_part part = {.ctx = &ctx, .stage = stage};
      nir_shader_instructions_pass(nir, rt_lower_deref, nir_metadata_none, &part);
      nir_fixup_deref_modes(nir);
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_deref) {
                  nir_deref_instr *deref = nir_instr_as_deref(instr);
                  if (deref->modes == nir_var_mem_global) {
                     deref->def.bit_size = 64;
                     deref->def.num_components = 1;
                  }
               }
            }
         }
      }
      if (!info->native)
         nir_lower_explicit_io(nir, nir_var_mem_global, nir_address_format_64bit_global);
      nir_lower_explicit_io(nir, nir_var_mem_push_const,
                           nir_address_format_32bit_offset);
      if (info->native) {
         stage->ray_offset = 0;
         bool trace = false;
         nir_foreach_function_impl(impl, nir) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  trace |= instr->type == nir_instr_type_intrinsic &&
                     nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_trace_ray;
               }
            }
         }
         stage->frame_size = trace ? RT_PC : 0;
         nir->scratch_size = 0;
      }
      nir_remove_dead_variables(nir, modes, NULL);
      nir_shader_intrinsics_pass(nir, rt_lower_builtin, nir_metadata_none, &part);
      if (!info->native)
         nir_shader_intrinsics_pass(nir, rt_prepare_report, nir_metadata_none, NULL);
      nir->info.stage = MESA_SHADER_COMPUTE;
      nir_opt_dce(nir);
      nir_validate_shader(nir, "panvk rt source explicit IO");
      if (!info->native)
         nir_lower_shader_calls(nir, &call_options, &stage->resumes, &stage->resume_count, tmp);
      stage->resume_base = next_pc;
      next_pc += stage->resume_count;
      max_scratch = MAX2(max_scratch, nir->scratch_size);
      if (stage->ray_query_slots >= UINT32_MAX / PANVK_RAY_QUERY_STATE_SIZE) {
         ralloc_free(tmp);
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      }
      ctx.external_frame_size = MAX2(ctx.external_frame_size,
         (stage->ray_query_slots + 1) * PANVK_RAY_QUERY_STATE_SIZE);
      for (unsigned p = 0; p < stage->resume_count; p++)
         max_scratch = MAX2(max_scratch, stage->resumes[p]->scratch_size);
   }
   ctx.phase_base = next_pc;
   ctx.frame_size = info->native ? RT_NATIVE_HEADER : ALIGN_POT(RT_HEADER + max_scratch, 128);
   result->frame_size = ctx.frame_size;
   result->ray_query_slots = ray_query_slots;
   result->external_frame_size = info->native ? ctx.external_frame_size : 0;
   result->stack_size = info->native ? 0 : (2 * info->max_recursion_depth + 8) * ctx.frame_size;
   if (info->native) {
      uint64_t slots[MESA_ALL_SHADER_STAGES] = {0};
      for (unsigned i = 0; i < info->stage_count; i++)
         slots[ctx.stages[i].stage] = MAX2(slots[ctx.stages[i].stage],
                                          (uint64_t)ctx.stages[i].ray_query_slots + 1);
      uint32_t d = info->max_recursion_depth;
      uint64_t total = slots[MESA_SHADER_RAYGEN] + MIN2(d, 1) *
         MAX3(slots[MESA_SHADER_CLOSEST_HIT], slots[MESA_SHADER_MISS],
              slots[MESA_SHADER_INTERSECTION] + slots[MESA_SHADER_ANY_HIT]) +
         (d ? d - 1 : 0) * MAX2(slots[MESA_SHADER_CLOSEST_HIT], slots[MESA_SHADER_MISS]) +
         2 * slots[MESA_SHADER_CALLABLE];
      if (total >= 65536) {
         ralloc_free(tmp);
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      }
      result->external_frame_size = MAX2(total, 1) * PANVK_RAY_QUERY_STATE_SIZE;
      rt_native_create(mem_ctx, &ctx, result);
      ralloc_free(tmp);
      return VK_SUCCESS;
   }
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, info->options, "panvk rt pipeline");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *dispatch = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0),
      .base = info->dispatch_sysval_offset, .range = 8);
   nir_def *launch = nir_load_global_invocation_id(&b, 32);
   nir_def *size = rt_load_constant(&b, dispatch, 88, 3, 32);
   nir_def *in_bounds = nir_imm_true(&b);
   for (unsigned c = 0; c < 3; c++)
      in_bounds = nir_iand(&b, in_bounds, nir_ult(&b, nir_channel(&b, launch, c), nir_channel(&b, size, c)));
   nir_push_if(&b, in_bounds);
   nir_def *lane = nir_load_subgroup_invocation(&b);
   nir_def *warp = nir_load_warp_id_arm(&b);
   nir_def *core = nir_load_core_id(&b);
   nir_def *warps = nir_iadd_imm(&b, nir_load_warp_max_id_arm(&b), 1);
   nir_def *index = nir_iadd(&b, nir_imul_imm(&b, nir_iadd(&b, nir_imul(&b, core, warps), warp), 16), lane);
   nir_def *root = nir_iadd(&b, rt_load_constant(&b, dispatch, 104, 1, 64),
      nir_imul(&b, nir_u2u64(&b, index), rt_load_constant(&b, dispatch, 112, 1, 64)));
   nir_def *sbt = rt_load_constant(&b, dispatch, 0, 1, 64);
   rt_store(&b, root, RT_PARENT, nir_imm_int64(&b, 0));
   rt_store(&b, root, RT_RETURN, nir_imm_int(&b, 0));
   rt_store(&b, root, RT_RAY, root);
   rt_store(&b, root, RT_DEPTH, nir_imm_int(&b, 0));
   rt_store(&b, root, RT_RECORD, nir_imm_int64(&b, 0));
   rt_store(&b, root, RT_PAYLOAD, nir_imm_int64(&b, 0));
   rt_store(&b, root, RT_SHADER_RECORD, nir_iadd_imm(&b, sbt, 32));
   rt_select(&b, root, root, rt_token_pc(&b, &ctx, dispatch, rt_sbt_token(&b, sbt, 0), MESA_SHADER_RAYGEN));
   nir_push_loop(&b);
   nir_def *pc = rt_load(&b, root, RT_PC, 1, 32);
   nir_push_if(&b, nir_ieq_imm(&b, pc, 0));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);
   nir_def *frame = rt_load(&b, root, RT_FRAME, 1, 64);
   nir_def *params[3] = {frame, dispatch, root};
   for (unsigned i = 0; i < info->stage_count; i++) {
      struct rt_stage *stage = &ctx.stages[i];
      result->per_stage_stack_size[i] = 2 * ctx.frame_size;
      for (unsigned p = 0; p <= stage->resume_count; p++) {
         nir_shader *part = p ? stage->resumes[p - 1] : stage->nir;
         rt_prepare_part(&ctx, stage, part);
         nir_push_if(&b, nir_ieq_imm(&b, pc, p ? stage->resume_base + p - 1 : i + 1));
         struct hash_table *remap = _mesa_pointer_hash_table_create(tmp);
         nir_inline_function_impl(&b, rt_entrypoint(part), params, remap);
         _mesa_hash_table_destroy(remap, NULL);
         nir_pop_if(&b, NULL);
      }
   }
   for (unsigned phase = 0; phase < RT_PHASE_COUNT; phase++) {
      nir_push_if(&b, nir_ieq_imm(&b, pc, ctx.phase_base + phase));
      rt_emit_phase(&b, &ctx, root, dispatch, frame, phase);
      nir_pop_if(&b, NULL);
   }
   nir_pop_loop(&b, NULL);
   nir_pop_if(&b, NULL);
   nir_index_ssa_defs(nir_shader_get_entrypoint(b.shader));
   nir_progress(true, b.impl, nir_metadata_none);
   nir_lower_vars_to_ssa(b.shader);
   nir_opt_dce(b.shader);
   nir_index_ssa_defs(nir_shader_get_entrypoint(b.shader));
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_validate_shader(b.shader, "panvk linked RT pipeline");
   ralloc_steal(mem_ctx, b.shader);
   result->nir = b.shader;
   ralloc_free(tmp);
   return VK_SUCCESS;
}
