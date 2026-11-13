#include "nvk_nir_rt_pipeline.h"
#include "nvk_nir_lower_ray_queries.h"
#include "nvk_nir_ser.h"
#include "nvk_ser.h"
#include "nvk_private.h"
#include "nvk_rt_pipeline.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "compiler/spirv/spirv.h"
#include "util/hash_table.h"
#include "util/u_math.h"

#define RT_HEADER 768
#define RT_COMMITTED 512
#define RT_CANDIDATE 256
#define RT_REPORT_TOKEN UINT32_MAX

enum rt_action { NVK_NIR_RT_ACCEPT, NVK_NIR_RT_IGNORE, NVK_NIR_RT_TERMINATE };

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
   RT_LAUNCH = 240, RT_SER_SUSPENDED = 252,
};

#define RT_HIT_OBJECT 144

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
};

struct rt_context {
   const struct nvk_nir_rt_pipeline_info *info;
   struct rt_stage *stages;
   uint32_t phase_base;
   uint32_t frame_size;
   nir_variable *queries;
   bool needs_triangle_vertices;
   bool uses_hit_objects;
   nir_def *owner;
};

struct rt_stack_io_state {
   uint32_t storage_size;
   nir_def *owner;
};

struct rt_part {
   struct rt_context *ctx;
   struct rt_stage *stage;
   nir_variable *yielded;
};

static nir_def *
rt_load(nir_builder *b, nir_def *base, unsigned offset, unsigned comps,
        unsigned bits)
{
   return nir_load_scratch(b, comps, bits,
      nir_iadd_imm(b, nir_u2u32(b, base), offset),
      .align_mul = bits == 64 ? 8 : 4);
}

static void
rt_store(nir_builder *b, nir_def *base, unsigned offset, nir_def *value)
{
   nir_store_scratch(b, value,
      nir_iadd_imm(b, nir_u2u32(b, base), offset),
      .align_mul = value->bit_size == 64 ? 8 : 4);
}

static nir_def *
rt_stack_address(nir_builder *b, nir_def *offset,
                 const struct rt_stack_io_state *state)
{
   nir_def *dispatch = nir_load_push_constant(b, 1, 64, nir_imm_int(b, 0),
      .base = NVK_RT_DISPATCH_PUSH_OFFSET, .range = 8);
   nir_def *base = nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, dispatch, offsetof(struct nvk_rt_dispatch, stack_addr)),
      .align_mul = 8);
   nir_def *group = state->owner ?
      nir_udiv_imm(b, state->owner, NVK_RT_WORKGROUP_SIZE) :
      nir_channel(b, nir_load_workgroup_id(b), 0);
   nir_def *lane = state->owner ?
      nir_iand_imm(b, state->owner, NVK_RT_WORKGROUP_SIZE - 1) :
      nir_channel(b, nir_load_local_invocation_id(b), 0);
   nir_def *group_offset = nir_imul_imm(b, nir_u2u64(b, group),
      (uint64_t)state->storage_size * NVK_RT_WORKGROUP_SIZE);
   nir_def *block = nir_imul_imm(b, nir_u2u64(b,
      nir_iand_imm(b, offset, ~(NVK_RT_STACK_LANE_BLOCK_SIZE - 1))),
      NVK_RT_WORKGROUP_SIZE);
   nir_def *lane_offset = nir_iadd(b,
      nir_imul_imm(b, lane, NVK_RT_STACK_LANE_BLOCK_SIZE),
      nir_iand_imm(b, offset, NVK_RT_STACK_LANE_BLOCK_SIZE - 1));
   return nir_iadd(b, nir_iadd(b, base, group_offset),
      nir_iadd(b, block, nir_u2u64(b, lane_offset)));
}

static bool
rt_lower_stack_io(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool load = intr->intrinsic == nir_intrinsic_load_scratch;
   if (!load && intr->intrinsic != nir_intrinsic_store_scratch)
      return false;
   const struct rt_stack_io_state *state = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *offset = intr->src[load ? 0 : 1].ssa;
   unsigned bits = load ? intr->def.bit_size : intr->src[0].ssa->bit_size;
   unsigned components = load ? intr->def.num_components :
                               intr->src[0].ssa->num_components;
   unsigned parts = bits == 64 ? 2 : 1;
   unsigned part_bits = bits / parts;
   unsigned alignment = MIN2(nir_intrinsic_align_mul(intr), part_bits / 8);
   nir_def *values[NIR_MAX_VEC_COMPONENTS];
   for (unsigned component = 0; component < components; component++) {
      if (!load && !(nir_intrinsic_write_mask(intr) & BITFIELD_BIT(component)))
         continue;
      nir_def *parts_value[2];
      nir_def *value = load ? NULL : nir_channel(b, intr->src[0].ssa, component);
      for (unsigned part = 0; part < parts; part++) {
         unsigned byte_offset = component * (bits / 8) + part * (part_bits / 8);
         unsigned align_offset =
            (nir_intrinsic_align_offset(intr) + byte_offset) % alignment;
         nir_def *address = rt_stack_address(b,
            nir_iadd_imm(b, offset, byte_offset), state);
         if (load) {
            parts_value[part] = nir_load_global(b, 1, part_bits, address,
               .align_mul = alignment, .align_offset = align_offset);
         } else {
            nir_def *scalar = parts == 2 ?
               nir_channel(b, nir_unpack_64_2x32(b, value), part) : value;
            nir_store_global(b, scalar, address,
               .align_mul = alignment, .align_offset = align_offset);
         }
      }
      if (load)
         values[component] = parts == 2 ?
            nir_pack_64_2x32_split(b, parts_value[0], parts_value[1]) : parts_value[0];
   }
   if (load)
      nir_def_rewrite_uses(&intr->def, nir_vec(b, values, components));
   nir_instr_remove(&intr->instr);
   return true;
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
   if (offset >= sizeof(VkTraceRaysIndirectCommand2KHR))
      return rt_load_constant(b, dispatch, offset, comps, bits);
   nir_def *mode = rt_load_constant(b, dispatch, offsetof(struct nvk_rt_dispatch, indirect_mode), 1, 32);
   nir_def *indirect = rt_load_constant(b, dispatch, offsetof(struct nvk_rt_dispatch, indirect_addr), 1, 64);
   nir_def *base = nir_bcsel(b, nir_ieq_imm(b, mode, 2), indirect, dispatch);
   if (offset >= offsetof(VkTraceRaysIndirectCommand2KHR, width)) {
      base = nir_bcsel(b, nir_ieq_imm(b, mode, 1),
         nir_iadd_imm(b, indirect, -(int)offsetof(VkTraceRaysIndirectCommand2KHR, width)), base);
   }
   return rt_load_constant(b, base, offset, comps, bits);
}

static nir_def *
rt_stage_load(nir_builder *b, const struct rt_part *part, nir_def *base,
              unsigned offset, unsigned comps, unsigned bits)
{
   return rt_load(b, base, offset, comps, bits);
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
rt_token_pc(nir_builder *b, struct rt_context *ctx, nir_def *dispatch,
            nir_def *token, mesa_shader_stage stage)
{
   nir_def *pc = nir_imm_int(b, 0);
   for (unsigned i = 0; i < ctx->info->stage_count; i++) {
      if (ctx->stages[i].stage == stage)
         pc = nir_bcsel(b, nir_ieq_imm(b, token, ctx->stages[i].id),
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
   return nir_uge(b, rt_load_constant(b, dispatch,
      offsetof(struct nvk_rt_dispatch, stack_stride), 1, 64),
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
   rt_store(b, child, RT_ACTION, nir_imm_int(b, NVK_NIR_RT_ACCEPT));
}

static nir_def *
rt_sbt(nir_builder *b, nir_def *dispatch, unsigned base, unsigned stride,
       nir_def *index)
{
   nir_def *address = rt_load_dispatch(b, NULL, dispatch, base, 1, 64);
   return nir_bcsel(b, nir_ine_imm(b, address, 0),
      nir_iadd(b, address,
         nir_imul(b, nir_u2u64(b, index), rt_load_dispatch(b, NULL, dispatch, stride, 1, 64))),
      nir_imm_int64(b, 0));
}

static nir_def *
rt_sbt_token(nir_builder *b, nir_def *sbt, unsigned slot)
{
   nir_push_if(b, nir_ine_imm(b, sbt, 0));
   nir_def *token = rt_load_constant(b, sbt, slot, 1, 32);
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
   rt_store(b, parent, RT_ACTION, nir_imm_int(b, NVK_NIR_RT_ACCEPT));
   rt_select(b, root, parent, nir_imm_int(b, return_pc));
   nir_pop_if(b, NULL);
}

static bool
rt_lower_builtin(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct rt_part *part = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *frame = rt_param(b, 0), *dispatch = rt_param(b, 1);
   nir_def *value = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_hit_object_load: {
      if (nir_intrinsic_base(intr) != nir_hit_object_value_shader_record)
         return false;
      nir_def *object = intr->src[0].ssa;
      nir_def *index = rt_load(b, object, NVK_HIT_OBJECT_SBT_INDEX, 1, 32);
      nir_def *type = rt_load(b, object, NVK_HIT_OBJECT_TYPE, 1, 32);
      nir_def *sbt = nir_bcsel(b, nir_ieq_imm(b, type, NVK_HIT_OBJECT_HIT),
         rt_sbt(b, dispatch, 40, 56, index), rt_sbt(b, dispatch, 16, 32, index));
      nir_def *address = nir_iadd_imm(b, sbt, NVK_RT_GROUP_HANDLE_SIZE);
      value = nir_unpack_64_2x32(b, address);
      break;
   }
   case nir_intrinsic_load_ray_launch_id:
      value = rt_load(b, rt_param(b, 2), RT_LAUNCH, 3, 32); break;
   case nir_intrinsic_load_ray_launch_size:
      value = rt_load_dispatch(b, part->ctx, dispatch, 88, 3, 32); break;
   case nir_intrinsic_load_shader_record_ptr:
      value = rt_load(b, frame, RT_SHADER_RECORD, 1, 64); break;
   case nir_intrinsic_load_push_constant: {
      if (nir_intrinsic_base(intr) >= NVK_RT_DISPATCH_PUSH_OFFSET)
         return false;
      nir_def *base = rt_load_dispatch(b, part->ctx, dispatch,
         offsetof(struct nvk_rt_dispatch, push_addr), 1, 64);
      unsigned stage_offset =
         (part->stage->stage - MESA_SHADER_RAYGEN) * NVK_MAX_PUSH_SIZE;
      value = nir_load_global_constant(b, intr->def.num_components, intr->def.bit_size,
         nir_iadd(b, nir_iadd_imm(b, base, stage_offset + nir_intrinsic_base(intr)),
                      nir_u2u64(b, intr->src[0].ssa)), .align_mul = 4);
      break;
   }
   default: break;
   }
   if (!value) {
      nir_def *ray = rt_load(b, frame, RT_RAY, 1, 64);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_ray_world_origin: value = rt_load(b, ray, RT_ORIGIN, 3, 32); break;
      case nir_intrinsic_load_ray_world_direction: value = rt_load(b, ray, RT_DIRECTION, 3, 32); break;
      case nir_intrinsic_load_ray_t_min: value = rt_load(b, ray, RT_TMIN, 1, 32); break;
      case nir_intrinsic_load_ray_t_max: value = rt_load(b, frame, RT_TMAX, 1, 32); break;
      case nir_intrinsic_load_ray_hit_kind: value = rt_load(b, frame, RT_HIT_KIND, 1, 32); break;
      case nir_intrinsic_load_ray_flags: value = rt_load(b, ray, RT_FLAGS, 1, 32); break;
      case nir_intrinsic_load_cull_mask: value = rt_load(b, ray, RT_CULL, 1, 32); break;
      default: break;
      }
      if (!value) {
         nir_def *record = rt_load(b, frame, RT_RECORD, 1, 64);
         unsigned offset = 0;
         switch (intr->intrinsic) {
         case nir_intrinsic_load_primitive_id: offset = 12; break;
         case nir_intrinsic_load_instance_id: offset = 16; break;
         case nir_intrinsic_load_ray_geometry_index: offset = 20; break;
         case nir_intrinsic_load_ray_instance_custom_index: offset = 24; break;
         case nir_intrinsic_load_ray_object_origin: offset = 40; break;
         case nir_intrinsic_load_ray_object_direction: offset = 52; break;
         case nir_intrinsic_load_ray_object_to_world: offset = 64 + 12 * nir_intrinsic_column(intr); break;
         case nir_intrinsic_load_ray_world_to_object: offset = 112 + 12 * nir_intrinsic_column(intr); break;
         case nir_intrinsic_load_ray_triangle_vertex_positions: offset = 160 + 12 * nir_intrinsic_column(intr); break;
         default: return false;
         }
         value = rt_load(b, record, offset, intr->def.num_components, intr->def.bit_size);
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
   b->cursor = nir_before_instr(instr);
   nir_def *frame = rt_param(b, 0);
   nir_def *base;
   if (var->data.mode == nir_var_shader_call_data)
      base = rt_stage_load(b, part, frame, RT_PAYLOAD, 1, 64);
   else if (var->data.mode == nir_var_ray_hit_attrib)
      base = nir_iadd_imm(b, rt_stage_load(b, part, frame, RT_RAY, 1, 64),
         part->stage->stage == MESA_SHADER_CLOSEST_HIT ? RT_COMMITTED_ATTR : RT_ATTR);
   else
      base = nir_iadd_imm(b, frame, RT_HEADER);
   nir_deref_instr *cast = nir_build_deref_cast(b,
      nir_u2u32(b, nir_iadd_imm(b, base, var->data.driver_location)),
      nir_var_function_temp, var->type, 0);
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
   if (intr->intrinsic == nir_intrinsic_rt_hit_object_trace_ray ||
       intr->intrinsic == nir_intrinsic_rt_reorder_thread ||
       intr->intrinsic == nir_intrinsic_rt_hit_object_execute_shader ||
       intr->intrinsic == nir_intrinsic_rt_trace_ray ||
       intr->intrinsic == nir_intrinsic_rt_execute_callable ||
       intr->intrinsic == nir_intrinsic_ignore_ray_intersection ||
       intr->intrinsic == nir_intrinsic_terminate_ray ||
       intr->intrinsic == nir_intrinsic_accept_ray_intersection)
      nir_store_var(b, part->yielded, nir_imm_true(b), 1);
   nir_def *value = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_rt_reorder_thread: {
      assert(ctx->info->ser);
      nir_def *queue = rt_load_constant(b, dispatch,
         offsetof(struct nvk_rt_dispatch, ser_queue_addr), 1, 64);
      nir_def *owner = nir_u2u32(b, rt_param(b, 3));
      nir_def *packets = rt_load_constant(b, queue,
         offsetof(struct nvk_ser_queue, packets), 1, 64);
      nir_def *packet = nir_iadd(b, packets,
         nir_imul_imm(b, nir_u2u64(b, owner), sizeof(struct nvk_ser_packet)));
      nir_def *object = nir_u2u64(b, intr->src[0].ssa);
      nir_push_if(b, nir_ine_imm(b, object, 0));
      nir_def *type = rt_load(b, object, NVK_HIT_OBJECT_TYPE, 1, 32);
      nir_def *index = rt_load(b, object, NVK_HIT_OBJECT_SBT_INDEX, 1, 32);
      nir_def *key = nir_ior(b, nir_ishl_imm(b, type, 30),
         nir_iand_imm(b, index, 0x3fffffff));
      nir_push_else(b, NULL);
      nir_def *no_object = nir_imm_int(b, UINT32_MAX);
      nir_pop_if(b, NULL);
      key = nir_if_phi(b, key, no_object);
      uint32_t pc = part->stage->resume_base + nir_intrinsic_call_idx(intr);
      rt_select(b, root, frame, nir_imm_int(b, pc));
      rt_store(b, root, RT_SER_SUSPENDED, nir_imm_int(b, 1));
      nir_store_global(b, nir_vec4(b, owner, nir_u2u32(b, frame),
         nir_imm_int(b, pc), key), packet, .align_mul = 16);
      nir_store_global(b, rt_param(b, 4), nir_iadd_imm(b, packet, 16),
         .align_mul = 8);
      nir_store_global(b, nir_vec2(b, intr->src[1].ssa, intr->src[2].ssa),
         nir_iadd_imm(b, packet, 24), .align_mul = 8);
      break;
   }
   case nir_intrinsic_load_scratch:
   case nir_intrinsic_store_scratch: {
      unsigned source = intr->intrinsic == nir_intrinsic_load_scratch ? 0 : 1;
      if (!nir_src_is_const(intr->src[source]))
         return false;
      nir_def *offset = nir_iadd(b,
         nir_iadd_imm(b, nir_u2u32(b, frame), RT_HEADER), intr->src[source].ssa);
      nir_src_rewrite(&intr->src[source], offset);
      return true;
   }
   case nir_intrinsic_load_scratch_base_ptr:
      value = nir_iadd_imm(b, nir_u2uN(b, frame, intr->def.bit_size), RT_HEADER);
      break;
   case nir_intrinsic_load_stack:
      value = nir_load_scratch(b, intr->def.num_components, intr->def.bit_size,
         nir_iadd_imm(b, nir_u2u32(b, frame), RT_HEADER + nir_intrinsic_base(intr)),
         .align_mul = nir_intrinsic_align_mul(intr),
         .align_offset = nir_intrinsic_align_offset(intr));
      break;
   case nir_intrinsic_store_stack:
      nir_store_scratch(b, intr->src[0].ssa,
         nir_iadd_imm(b, nir_u2u32(b, frame), RT_HEADER + nir_intrinsic_base(intr)),
         .align_mul = nir_intrinsic_align_mul(intr),
         .align_offset = nir_intrinsic_align_offset(intr),
         .write_mask = nir_intrinsic_write_mask(intr));
      break;
   case nir_intrinsic_rt_resume: break;
   case nir_intrinsic_ignore_ray_intersection:
   case nir_intrinsic_terminate_ray:
   case nir_intrinsic_accept_ray_intersection:
      rt_store(b, frame, RT_ACTION, nir_imm_int(b,
         intr->intrinsic == nir_intrinsic_ignore_ray_intersection ? NVK_NIR_RT_IGNORE :
         intr->intrinsic == nir_intrinsic_terminate_ray ? NVK_NIR_RT_TERMINATE : NVK_NIR_RT_ACCEPT));
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
   case nir_intrinsic_rt_hit_object_trace_ray:
   case nir_intrinsic_rt_trace_ray: {
      bool object_trace = intr->intrinsic == nir_intrinsic_rt_hit_object_trace_ray;
      nir_def *src[11];
      for (unsigned i = 0; i < 11; i++)
         src[i] = intr->src[i].ssa;
      if (object_trace) {
         src[0] = intr->src[1].ssa;
         src[1] = nir_channel(b, intr->src[2].ssa, 0);
         src[2] = nir_channel(b, intr->src[2].ssa, 1);
      }
      nir_def *child = rt_child(b, ctx, frame);
      nir_def *depth = nir_iadd_imm(b, rt_load(b, frame, RT_DEPTH, 1, 32), 1);
      nir_def *valid = nir_iand(b, rt_child_valid(b, ctx, dispatch, root, child),
                                  nir_ule_imm(b, depth, ctx->info->max_recursion_depth));
      nir_push_if(b, valid);
      rt_init_child(b, child, frame, child, nir_imm_int64(b, 0), nir_imm_int64(b, 0),
                    nir_u2u64(b, src[10]),
                    nir_imm_int(b, part->stage->resume_base + nir_intrinsic_call_idx(intr)));
      rt_store(b, child, RT_DEPTH, depth);
      if (ctx->uses_hit_objects)
         rt_store(b, child, RT_HIT_OBJECT, object_trace ?
                  nir_u2u64(b, intr->src[0].ssa) : nir_imm_int64(b, 0));
      rt_store(b, child, RT_AS, src[0]);
      unsigned pipeline_flags = 0;
      if (ctx->info->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_TRIANGLES_BIT_KHR)
         pipeline_flags |= SpvRayFlagsSkipTrianglesKHRMask;
      if (ctx->info->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_SKIP_AABBS_BIT_KHR)
         pipeline_flags |= SpvRayFlagsSkipAABBsKHRMask;
      rt_store(b, child, RT_FLAGS, nir_ior_imm(b, src[1], pipeline_flags));
      rt_store(b, child, RT_CULL, nir_iand_imm(b, src[2], 255));
      rt_store(b, child, RT_SBT_OFFSET, nir_iand_imm(b, src[3], 15));
      rt_store(b, child, RT_SBT_STRIDE, nir_iand_imm(b, src[4], 15));
      rt_store(b, child, RT_MISS, nir_iand_imm(b, src[5], 65535));
      rt_store(b, child, RT_ORIGIN, src[6]);
      rt_store(b, child, RT_TMIN, src[7]);
      rt_store(b, child, RT_DIRECTION, src[8]);
      rt_store(b, child, RT_TMAX, src[9]);
      rt_select_phase(b, ctx, root, child, RT_BEGIN);
      nir_push_else(b, NULL);
      rt_select(b, root, frame,
                nir_imm_int(b, part->stage->resume_base + nir_intrinsic_call_idx(intr)));
      nir_pop_if(b, NULL);
      break;
   }
   case nir_intrinsic_rt_hit_object_execute_shader: {
      nir_def *object = intr->src[0].ssa;
      nir_def *type = rt_load(b, object, NVK_HIT_OBJECT_TYPE, 1, 32);
      nir_def *index = rt_load(b, object, NVK_HIT_OBJECT_SBT_INDEX, 1, 32);
      nir_def *ray = rt_child(b, ctx, frame);
      nir_def *depth = nir_iadd_imm(b, rt_load(b, frame, RT_DEPTH, 1, 32), 1);
      nir_def *valid = nir_iand(b,
         nir_iand(b, nir_ine_imm(b, type, NVK_HIT_OBJECT_EMPTY),
                     nir_ine_imm(b, index, UINT32_MAX)),
         nir_iand(b, rt_child_valid(b, ctx, dispatch, root, ray),
                     nir_ule_imm(b, depth, ctx->info->max_recursion_depth)));
      nir_push_if(b, valid);
      uint32_t resume = part->stage->resume_base + nir_intrinsic_call_idx(intr);
      rt_init_child(b, ray, frame, ray, nir_iadd_imm(b, ray, RT_COMMITTED),
                    nir_imm_int64(b, 0), nir_u2u64(b, intr->src[1].ssa), nir_imm_int(b, resume));
      rt_store(b, ray, RT_DEPTH, depth);
      rt_store(b, ray, RT_HIT_OBJECT, nir_imm_int64(b, 0));
      static const struct { unsigned offset, components; } fields[] = {
         {RT_AS, 2}, {RT_FLAGS, 1}, {RT_CULL, 1}, {RT_SBT_OFFSET, 1},
         {RT_SBT_STRIDE, 1}, {RT_MISS, 1}, {RT_TMIN, 1}, {RT_ORIGIN, 3},
         {RT_TMAX, 1}, {RT_DIRECTION, 3}, {RT_HIT_KIND, 1},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(fields); i++)
         rt_store(b, ray, fields[i].offset,
            rt_load(b, object, fields[i].offset, fields[i].components, 32));
      for (unsigned byte = 0; byte < 208; byte += 16)
         rt_store(b, ray, RT_COMMITTED + byte,
            rt_load(b, object, NVK_HIT_OBJECT_RECORD + byte, MIN2(4, (208 - byte) / 4), 32));
      for (unsigned byte = 0; byte < 32; byte += 16)
         rt_store(b, ray, RT_COMMITTED_ATTR + byte,
            rt_load(b, object, NVK_HIT_OBJECT_ATTRIBUTE + byte, 4, 32));
      nir_push_if(b, nir_ieq_imm(b, type, NVK_HIT_OBJECT_HIT));
      nir_def *sbt = rt_sbt(b, dispatch, 40, 56, index);
      rt_invoke(b, ctx, root, dispatch, ray, ray, nir_iadd_imm(b, ray, RT_COMMITTED),
                sbt, 16, MESA_SHADER_CLOSEST_HIT, ctx->phase_base + RT_TRACE_RETURN,
                rt_load(b, ray, RT_TMAX, 1, 32), rt_load(b, ray, RT_HIT_KIND, 1, 32));
      nir_push_else(b, NULL);
      sbt = rt_sbt(b, dispatch, 16, 32, index);
      rt_invoke(b, ctx, root, dispatch, ray, ray, nir_imm_int64(b, 0),
                sbt, 0, MESA_SHADER_MISS, ctx->phase_base + RT_TRACE_RETURN,
                rt_load(b, ray, RT_TMAX, 1, 32), nir_imm_int(b, 0));
      nir_pop_if(b, NULL);
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
   func->num_params = ctx->info->ser ? 5 : 3;
   func->params = ralloc_array(func, nir_parameter, func->num_params);
   for (unsigned p = 0; p < func->num_params; p++)
      func->params[p] = (nir_parameter){.num_components = 1, .bit_size = 64};
   nir_shader_intrinsics_pass(nir, rt_lower_param, nir_metadata_none, NULL);
   nir_validate_shader(nir, "nvk rt continuation part");
}


static nir_def *
rt_query(nir_builder *b, struct rt_context *ctx, nir_def *ray)
{
   nir_def *index = nir_udiv_imm(b, nir_u2u32(b, ray), ctx->frame_size);
   return &nir_build_deref_array(b, nir_build_deref_var(b, ctx->queries), index)->def;
}

static nir_def *
rt_query_load(nir_builder *b, nir_def *query, nir_ray_query_value value,
               bool committed, unsigned components, unsigned bits,
               unsigned column)
{
   return nir_rq_load(b, components, bits, query,
      .ray_query_value = value, .committed = committed, .column = column);
}

static void
rt_snapshot(nir_builder *b, struct rt_context *ctx, nir_def *query,
             nir_def *record, bool committed)
{
   static const struct {
      nir_ray_query_value value;
      unsigned offset;
      unsigned components;
   } values[] = {
      {nir_ray_query_value_intersection_t, 0, 1},
      {nir_ray_query_value_intersection_primitive_index, 12, 1},
      {nir_ray_query_value_intersection_instance_id, 16, 1},
      {nir_ray_query_value_intersection_geometry_index, 20, 1},
      {nir_ray_query_value_intersection_instance_custom_index, 24, 1},
      {nir_ray_query_value_intersection_instance_sbt_index, 28, 1},
      {nir_ray_query_value_intersection_object_ray_origin, 40, 3},
      {nir_ray_query_value_intersection_object_ray_direction, 52, 3},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(values); i++)
      rt_store(b, record, values[i].offset,
         rt_query_load(b, query, values[i].value, committed,
                       values[i].components, 32, 0));
   for (unsigned c = 0; c < 4; c++) {
      rt_store(b, record, 64 + c * 12,
         rt_query_load(b, query, nir_ray_query_value_intersection_object_to_world,
                       committed, 3, 32, c));
      rt_store(b, record, 112 + c * 12,
         rt_query_load(b, query, nir_ray_query_value_intersection_world_to_object,
                       committed, 3, 32, c));
   }
   nir_def *type = rt_query_load(b, query, nir_ray_query_value_intersection_type,
                                 committed, 1, 32, 0);
   nir_push_if(b, nir_ieq_imm(b, type, committed ? 1 : 0));
   rt_store(b, record, 4,
      rt_query_load(b, query, nir_ray_query_value_intersection_barycentrics,
                    committed, 2, 32, 0));
   rt_store(b, record, 32, nir_b2i32(b,
      rt_query_load(b, query, nir_ray_query_value_intersection_front_face,
                    committed, 1, 1, 0)));
   if (ctx->needs_triangle_vertices) {
      for (unsigned v = 0; v < 3; v++)
         rt_store(b, record, 160 + v * 12,
            rt_query_load(b, query,
               nir_ray_query_value_intersection_triangle_vertex_positions,
               committed, 3, 32, v));
   }
   nir_push_else(b, NULL);
   if (!committed)
      rt_store(b, record, 36, nir_b2i32(b,
         rt_query_load(b, query,
            nir_ray_query_value_intersection_candidate_aabb_opaque,
            false, 1, 1, 0)));
   nir_pop_if(b, NULL);
}

static nir_def *
rt_hit_sbt_index(nir_builder *b, nir_def *ray, nir_def *record)
{
   return nir_iadd(b, rt_load(b, record, 28, 1, 32),
      nir_iadd(b, rt_load(b, ray, RT_SBT_OFFSET, 1, 32),
         nir_imul(b, rt_load(b, record, 20, 1, 32),
                     rt_load(b, ray, RT_SBT_STRIDE, 1, 32))));
}

static nir_def *
rt_tri_kind(nir_builder *b, nir_def *record)
{
   return nir_bcsel(b, nir_ine_imm(b, rt_load(b, record, 32, 1, 32), 0),
                     nir_imm_int(b, 0xfe), nir_imm_int(b, 0xff));
}

static void
rt_commit(nir_builder *b, struct rt_context *ctx, nir_def *ray,
           nir_def *t, nir_def *kind, bool generated)
{
   nir_def *query = rt_query(b, ctx, ray);
   if (generated) {
      nir_rq_generate_intersection(b, query, t);
      rt_store(b, ray, RT_GENERATED_KIND, kind);
   } else {
      nir_rq_confirm_intersection(b, query);
   }
   rt_store(b, ray, RT_COMMITTED_T, t);
   for (unsigned i = 0; i < 32; i += 16)
      rt_store(b, ray, RT_COMMITTED_ATTR + i,
         rt_load(b, ray, RT_ATTR + i, 4, 32));
}

static nir_def *
rt_terminated(nir_builder *b, nir_def *ray, nir_def *action)
{
   return nir_ior(b, nir_ieq_imm(b, action, NVK_NIR_RT_TERMINATE),
      nir_test_mask(b, rt_load(b, ray, RT_FLAGS, 1, 32),
                      SpvRayFlagsTerminateOnFirstHitKHRMask));
}

static void
rt_report_commit(nir_builder *b, struct rt_context *ctx, nir_def *root,
                  nir_def *frame)
{
   nir_def *ray = rt_ray(b, frame);
   nir_def *action = rt_load(b, frame, RT_ACTION, 1, 32);
   nir_def *accept = nir_ine_imm(b, action, NVK_NIR_RT_IGNORE);
   rt_store(b, frame, RT_REPORT_ACCEPTED, nir_b2i32(b, accept));
   nir_push_if(b, accept);
   rt_commit(b, ctx, ray, rt_load(b, frame, RT_REPORT_T, 1, 32),
      rt_load(b, frame, RT_REPORT_KIND, 1, 32), true);
   nir_pop_if(b, NULL);
   nir_push_if(b, nir_iand(b, accept, rt_terminated(b, ray, action)));
   nir_rq_terminate(b, rt_query(b, ctx, ray));
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
   case RT_BEGIN:
      rt_store(b, frame, RT_GENERATED_KIND, nir_imm_int(b, 0));
      rt_store(b, frame, RT_COMMITTED_T, rt_load(b, frame, RT_TMAX, 1, 32));
      nir_rq_initialize(b, rt_query(b, ctx, frame),
         rt_load(b, frame, RT_AS, 1, 64),
         rt_load(b, frame, RT_FLAGS, 1, 32),
         rt_load(b, frame, RT_CULL, 1, 32),
         rt_load(b, frame, RT_ORIGIN, 3, 32),
         rt_load(b, frame, RT_TMIN, 1, 32),
         rt_load(b, frame, RT_DIRECTION, 3, 32),
         rt_load(b, frame, RT_TMAX, 1, 32));
      rt_select_phase(b, ctx, root, frame, RT_RESUME);
      break;
   case RT_RESUME: {
      nir_def *query = rt_query(b, ctx, frame);
      nir_def *more = nir_rq_proceed(b, 1, query);
      nir_push_if(b, more);
      rt_snapshot(b, ctx, query, candidate, false);
      nir_def *type = rt_query_load(b, query,
         nir_ray_query_value_intersection_type, false, 1, 32, 0);
      nir_def *status = nir_ior(b, rt_hit_sbt_index(b, frame, candidate),
         nir_bcsel(b, nir_ieq_imm(b, type, 0),
                      nir_imm_int(b, 1u << 29), nir_imm_int(b, 2u << 29)));
      rt_store(b, frame, RT_STATUS, status);
      rt_select_phase(b, ctx, root, frame, RT_CANDIDATE_ROUTE);
      nir_push_else(b, NULL);
      rt_select_phase(b, ctx, root, frame, RT_FINISH);
      nir_pop_if(b, NULL);
      break;
   }
   case RT_CANDIDATE_ROUTE: {
      nir_def *status = rt_load(b, frame, RT_STATUS, 1, 32);
      nir_def *sbt = rt_sbt(b, dispatch, 40, 56,
         nir_iand_imm(b, status, 0x0fffffff));
      nir_push_if(b, nir_ieq_imm(b, nir_ushr_imm(b, status, 29), 1));
      rt_store(b, frame, RT_ATTR, rt_load(b, candidate, 4, 2, 32));
      rt_invoke(b, ctx, root, dispatch, frame, frame, candidate, sbt, 8,
         MESA_SHADER_ANY_HIT, ctx->phase_base + RT_TRIANGLE_COMMIT,
         rt_load(b, candidate, 0, 1, 32), rt_tri_kind(b, candidate));
      nir_push_else(b, NULL);
      rt_invoke(b, ctx, root, dispatch, frame, frame, candidate, sbt, 0,
         MESA_SHADER_INTERSECTION, ctx->phase_base + RT_RESUME,
         rt_load(b, frame, RT_COMMITTED_T, 1, 32), nir_imm_int(b, 0));
      nir_pop_if(b, NULL);
      break;
   }
   case RT_TRIANGLE_COMMIT: {
      nir_def *action = rt_load(b, frame, RT_ACTION, 1, 32);
      nir_def *accepted = nir_ine_imm(b, action, NVK_NIR_RT_IGNORE);
      nir_push_if(b, accepted);
      rt_commit(b, ctx, frame, rt_load(b, candidate, 0, 1, 32),
         rt_tri_kind(b, candidate), false);
      nir_pop_if(b, NULL);
      nir_def *terminate = nir_iand(b, accepted, rt_terminated(b, frame, action));
      nir_push_if(b, terminate);
      nir_rq_terminate(b, rt_query(b, ctx, frame));
      nir_pop_if(b, NULL);
      rt_select(b, root, frame, nir_bcsel(b, terminate,
         nir_imm_int(b, ctx->phase_base + RT_FINISH),
         nir_imm_int(b, ctx->phase_base + RT_RESUME)));
      break;
   }
   case RT_REPORT_START: {
      nir_def *ray = rt_ray(b, frame);
      nir_def *record = rt_load(b, frame, RT_RECORD, 1, 64);
      nir_def *t = rt_load(b, frame, RT_REPORT_T, 1, 32);
      uint32_t saved = b->fp_math_ctrl;
      b->fp_math_ctrl = nir_fp_no_fast_math;
      nir_def *valid = nir_iand(b,
         nir_fge(b, t, rt_load(b, ray, RT_TMIN, 1, 32)),
         nir_fge(b, rt_load(b, ray, RT_COMMITTED_T, 1, 32), t));
      b->fp_math_ctrl = saved;
      nir_push_if(b, valid);
      nir_def *sbt = rt_sbt(b, dispatch, 40, 56,
         nir_iand_imm(b, rt_load(b, ray, RT_STATUS, 1, 32), 0x0fffffff));
      nir_push_if(b, nir_ine_imm(b, rt_load(b, record, 36, 1, 32), 0));
      rt_store(b, frame, RT_ACTION, nir_imm_int(b, NVK_NIR_RT_ACCEPT));
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
   case RT_REPORT_COMMIT:
      rt_report_commit(b, ctx, root, frame);
      break;
   case RT_FINISH: {
      if (ctx->uses_hit_objects) {
         nir_def *object = rt_load(b, frame, RT_HIT_OBJECT, 1, 64);
         nir_push_if(b, nir_ine_imm(b, object, 0));
         nir_def *query = rt_query(b, ctx, frame);
         nir_def *type = rt_query_load(b, query, nir_ray_query_value_intersection_type, true, 1, 32, 0);
         for (unsigned byte = 64; byte < 128; byte += 16)
            rt_store(b, object, byte, rt_load(b, frame, byte, 4, 32));
         rt_store(b, object, NVK_HIT_OBJECT_TIME, nir_imm_int(b, 0));
         nir_push_if(b, nir_ine_imm(b, type, 0));
         rt_snapshot(b, ctx, query, committed, true);
         rt_store(b, object, RT_TMAX, rt_load(b, committed, 0, 1, 32));
         rt_store(b, object, RT_HIT_KIND, nir_bcsel(b, nir_ieq_imm(b, type, 1),
            rt_tri_kind(b, committed), rt_load(b, frame, RT_GENERATED_KIND, 1, 32)));
         for (unsigned byte = 0; byte < 208; byte += 16)
            rt_store(b, object, NVK_HIT_OBJECT_RECORD + byte,
               rt_load(b, committed, byte, MIN2(4, (208 - byte) / 4), 32));
         nir_push_if(b, nir_ieq_imm(b, type, 1));
         rt_store(b, frame, RT_COMMITTED_ATTR, rt_load(b, committed, 4, 2, 32));
         nir_pop_if(b, NULL);
         for (unsigned byte = 0; byte < 32; byte += 16)
            rt_store(b, object, NVK_HIT_OBJECT_ATTRIBUTE + byte,
               rt_load(b, frame, RT_COMMITTED_ATTR + byte, 4, 32));
         rt_store(b, object, NVK_HIT_OBJECT_TYPE, nir_imm_int(b, NVK_HIT_OBJECT_HIT));
         nir_def *index = rt_hit_sbt_index(b, frame, committed);
         rt_store(b, object, NVK_HIT_OBJECT_SBT_INDEX, nir_bcsel(b,
            nir_test_mask(b, rt_load(b, frame, RT_FLAGS, 1, 32), SpvRayFlagsSkipClosestHitShaderKHRMask),
            nir_imm_int(b, UINT32_MAX), index));
         nir_push_else(b, NULL);
         rt_store(b, object, NVK_HIT_OBJECT_TYPE, nir_imm_int(b, NVK_HIT_OBJECT_MISS));
         rt_store(b, object, NVK_HIT_OBJECT_SBT_INDEX, rt_load(b, frame, RT_MISS, 1, 32));
         nir_pop_if(b, NULL);
         rt_select_phase(b, ctx, root, frame, RT_TRACE_RETURN);
         nir_push_else(b, NULL);
      }
      nir_def *query = rt_query(b, ctx, frame);
      nir_def *type = rt_query_load(b, query,
         nir_ray_query_value_intersection_type, true, 1, 32, 0);
      nir_push_if(b, nir_ine_imm(b, type, 0));
      rt_snapshot(b, ctx, query, committed, true);
      rt_store(b, frame, RT_COMMITTED_T, rt_load(b, committed, 0, 1, 32));
      nir_push_if(b, nir_inot(b, nir_test_mask(b,
         rt_load(b, frame, RT_FLAGS, 1, 32), SpvRayFlagsSkipClosestHitShaderKHRMask)));
      nir_push_if(b, nir_ieq_imm(b, type, 1));
      rt_store(b, frame, RT_COMMITTED_ATTR, rt_load(b, committed, 4, 2, 32));
      nir_pop_if(b, NULL);
      nir_def *sbt = rt_sbt(b, dispatch, 40, 56,
         rt_hit_sbt_index(b, frame, committed));
      rt_invoke(b, ctx, root, dispatch, frame, frame, committed, sbt, 16,
         MESA_SHADER_CLOSEST_HIT, ctx->phase_base + RT_TRACE_RETURN,
         rt_load(b, committed, 0, 1, 32),
         nir_bcsel(b, nir_ieq_imm(b, type, 1), rt_tri_kind(b, committed),
            rt_load(b, frame, RT_GENERATED_KIND, 1, 32)));
      nir_push_else(b, NULL);
      rt_select_phase(b, ctx, root, frame, RT_TRACE_RETURN);
      nir_pop_if(b, NULL);
      nir_push_else(b, NULL);
      sbt = rt_sbt(b, dispatch, 16, 32,
         rt_load(b, frame, RT_MISS, 1, 32));
      rt_invoke(b, ctx, root, dispatch, frame, frame, nir_imm_int64(b, 0), sbt, 0,
         MESA_SHADER_MISS, ctx->phase_base + RT_TRACE_RETURN,
         rt_load(b, frame, RT_TMAX, 1, 32), nir_imm_int(b, 0));
      nir_pop_if(b, NULL);
      if (ctx->uses_hit_objects)
         nir_pop_if(b, NULL);
      break;
   }
   case RT_TRACE_RETURN:
      rt_return(b, root, frame, false);
      break;
   default:
      UNREACHABLE("Invalid RT phase");
   }
}

VkResult
nvk_nir_rt_pipeline_create(void *mem_ctx,
                            const struct nvk_nir_rt_pipeline_info *info,
                            struct nvk_nir_rt_pipeline_result *result)
{
   memset(result, 0, sizeof(*result));
   bool has_reorder = false;
   for (unsigned stage = 0; stage < info->stage_count; stage++) {
      nir_foreach_function_impl(impl, info->stages[stage].nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               has_reorder |= instr->type == nir_instr_type_intrinsic &&
                  nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_reorder_thread;
            }
         }
      }
   }
   struct nvk_nir_rt_pipeline_info effective_info = *info;
   if (!has_reorder)
      effective_info.ser = NULL;
   info = &effective_info;
   result->uses_ser_queue = info->ser != NULL;
   if (info->ser && (!info->ser->emit_suspend || !info->ser->emit_complete))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   if (info->max_hit_attribute_size > 32)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   void *tmp = ralloc_context(NULL);
   if (!tmp)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct rt_context ctx = {.info = info};
   ctx.stages = rzalloc_array(tmp, struct rt_stage, info->stage_count);
   result->per_stage_stack_size = rzalloc_array(mem_ctx, uint32_t, info->stage_count);
   if ((info->stage_count && !ctx.stages) ||
       (info->stage_count && !result->per_stage_stack_size)) {
      ralloc_free(tmp);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   uint32_t next_pc = info->stage_count + 1;
   uint32_t max_scratch = 0;
   nir_lower_shader_calls_options call_options = {
      .address_format = nir_address_format_32bit_offset,
      .stack_alignment = 16,
      .should_remat_callback = rt_remat,
   };
   for (unsigned i = 0; i < info->stage_count; i++) {
      struct rt_stage *stage = &ctx.stages[i];
      stage->id = info->stages[i].id;
      stage->stage = info->stages[i].stage;
      stage->nir = nir_shader_clone(tmp, info->stages[i].nir);
      nir_shader *nir = stage->nir;
      nir_lower_variable_initializers(nir,
         nir_var_shader_temp | nir_var_function_temp);
      nir_lower_returns(nir);
      nir_inline_functions(nir);
      nir_cleanup_functions(nir);
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               ctx.needs_triangle_vertices |=
                  intr->intrinsic == nir_intrinsic_hit_object_load &&
                  nir_intrinsic_base(intr) == nir_hit_object_value_triangle_positions;
            }
         }
      }
      nvk_nir_lower_hit_objects_with_reorder(nir, info->ser != NULL);
      nvk_nir_lower_ray_queries_with_omm(nir,
                                        info->ray_query_opacity_micromap);
      nir_lower_system_values(nir);
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               ctx.uses_hit_objects |= instr->type == nir_instr_type_intrinsic &&
                  (nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_hit_object_trace_ray ||
                   nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_hit_object_execute_shader);
               ctx.needs_triangle_vertices |=
                  instr->type == nir_instr_type_intrinsic &&
                  nir_instr_as_intrinsic(instr)->intrinsic ==
                     nir_intrinsic_load_ray_triangle_vertex_positions;
            }
         }
      }
      nir_lower_global_vars_to_local(nir);
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
                  if (deref->modes == nir_var_function_temp) {
                     deref->def.bit_size = 32;
                     deref->def.num_components = 1;
                  }
               }
            }
         }
      }
      nir_lower_explicit_io(nir, nir_var_function_temp,
                           nir_address_format_32bit_offset);
      nir_lower_explicit_io(nir, nir_var_mem_global,
                           nir_address_format_64bit_global);
      nir_lower_explicit_io(nir, nir_var_mem_push_const,
                           nir_address_format_32bit_offset);
      nir_remove_dead_variables(nir, modes, NULL);
      nir_shader_intrinsics_pass(nir, rt_lower_builtin, nir_metadata_none, &part);
      nir_shader_intrinsics_pass(nir, rt_prepare_report, nir_metadata_none, NULL);
      nir->info.stage = MESA_SHADER_COMPUTE;
      nir_opt_dce(nir);
      nir_lower_shader_calls(nir, &call_options, &stage->resumes,
                             &stage->resume_count, tmp);
      stage->resume_base = next_pc;
      next_pc += stage->resume_count;
      max_scratch = MAX2(max_scratch, nir->scratch_size);
      for (unsigned p = 0; p < stage->resume_count; p++)
         max_scratch = MAX2(max_scratch, stage->resumes[p]->scratch_size);
   }
   ctx.phase_base = next_pc;
   uint64_t frame_size = ALIGN_POT((uint64_t)RT_HEADER + max_scratch, 128);
   if (frame_size > UINT32_MAX / 2) {
      ralloc_free(tmp);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   ctx.frame_size = frame_size;
   result->frame_size = frame_size;
   for (unsigned i = 0; i < info->stage_count; i++)
      result->per_stage_stack_size[i] = 2 * ctx.frame_size;
   if (info->flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR) {
      ralloc_free(tmp);
      return VK_SUCCESS;
   }
   uint64_t frame_count = 2ull * info->max_recursion_depth + 8;
   if (frame_size * frame_count > UINT32_MAX) {
      ralloc_free(tmp);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   result->stack_size = frame_size * frame_count;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      info->options, "nvk rt pipeline");
   b.shader->info.workgroup_size[0] = NVK_RT_WORKGROUP_SIZE;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->scratch_size = result->stack_size;
   ctx.queries = nir_local_variable_create(b.impl,
      glsl_array_type(glsl_uint64_t_type(), frame_count, 0), "rt_queries");
   ctx.queries->data.ray_query = true;
   nir_def *dispatch = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0),
      .base = NVK_RT_DISPATCH_PUSH_OFFSET, .range = 8);
   nir_def *size = rt_load_dispatch(&b, &ctx, dispatch, 88, 3, 32);
   nir_def *linear = nir_u2u64(&b, nir_channel(&b,
      nir_load_global_invocation_id(&b, 32), 0));
   nir_def *ser_queue = NULL, *resume_pc = NULL, *resume_frame = NULL;
   nir_variable *resumed = NULL;
   if (info->ser) {
      ser_queue = rt_load_constant(&b, dispatch,
         offsetof(struct nvk_rt_dispatch, ser_queue_addr), 1, 64);
      nir_def *queue_index = nir_load_sysval_nv(&b, 32, .base = 0x61);
      nir_def *records = rt_load_constant(&b, ser_queue,
         offsetof(struct nvk_ser_queue, resume_owner_ids), 1, 64);
      nir_def *mask = rt_load_constant(&b, ser_queue,
         offsetof(struct nvk_ser_queue, resume_record_mask), 1, 32);
      nir_def *index = nir_iand(&b, nir_iadd(&b,
         nir_imul_imm(&b, queue_index, NVK_RT_WORKGROUP_SIZE),
         nir_load_local_invocation_index(&b)), mask);
      ctx.owner = nir_load_global(&b, 1, 32,
         nir_iadd(&b, records, nir_ishl_imm(&b, nir_u2u64(&b, index), 2)),
         .align_mul = 4);
      nir_def *capacity = rt_load_constant(&b, ser_queue,
         offsetof(struct nvk_ser_queue, owner_capacity), 1, 32);
      nir_push_if(&b, nir_uge(&b, ctx.owner, capacity));
      nir_jump(&b, nir_jump_return);
      nir_pop_if(&b, NULL);
      nir_def *packets = rt_load_constant(&b, ser_queue,
         offsetof(struct nvk_ser_queue, packets), 1, 64);
      nir_def *packet = nir_iadd(&b, packets,
         nir_imul_imm(&b, nir_u2u64(&b, ctx.owner), sizeof(struct nvk_ser_packet)));
      nir_def *packet_state = nir_load_global(&b, 4, 32, packet,
         .align_mul = 16);
      resume_frame = nir_channel(&b, packet_state, 1);
      resume_pc = nir_channel(&b, packet_state, 2);
      linear = nir_load_global(&b, 1, 64, nir_iadd_imm(&b, packet,
         offsetof(struct nvk_ser_packet, ray_index)), .align_mul = 8);
      nir_store_global(&b, nir_imm_int(&b, UINT32_MAX), packet, .align_mul = 4);
      resumed = nir_local_variable_create(b.impl, glsl_bool_type(), "rt_resumed");
      nir_store_var(&b, resumed, nir_ine_imm(&b, resume_pc, 0), 1);
   }
   nir_def *count = nir_imul(&b, nir_u2u64(&b, nir_channel(&b, size, 0)),
      nir_imul(&b, nir_u2u64(&b, nir_channel(&b, size, 1)),
                    nir_u2u64(&b, nir_channel(&b, size, 2))));
   nir_def *workers = nir_u2u64(&b, rt_load_dispatch(&b, &ctx, dispatch,
      offsetof(struct nvk_rt_dispatch, worker_count), 1, 32));
   nir_variable *ray_index = nir_local_variable_create(b.impl,
      glsl_uint64_t_type(), "rt_ray_index");
   nir_store_var(&b, ray_index, linear, 1);
   nir_def *root = nir_imm_int64(&b, 0);
   if (info->ser)
      rt_store(&b, root, RT_SER_SUSPENDED, nir_imm_int(&b, 0));
   nir_push_loop(&b);
   linear = nir_load_var(&b, ray_index);
   nir_push_if(&b, nir_uge(&b, linear, count));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);
   nir_def *width = nir_u2u64(&b, nir_channel(&b, size, 0));
   nir_def *height = nir_u2u64(&b, nir_channel(&b, size, 1));
   nir_def *xy = nir_udiv(&b, linear, width);
   nir_def *launch = nir_vec3(&b, nir_u2u32(&b, nir_umod(&b, linear, width)),
      nir_u2u32(&b, nir_umod(&b, xy, height)),
      nir_u2u32(&b, nir_udiv(&b, xy, height)));
   if (info->ser) {
      nir_push_if(&b, nir_load_var(&b, resumed));
      rt_select(&b, root, nir_u2u64(&b, resume_frame), resume_pc);
      nir_push_else(&b, NULL);
   }
   rt_store(&b, root, RT_LAUNCH, launch);
   nir_def *sbt = rt_load_dispatch(&b, &ctx, dispatch, 0, 1, 64);
   rt_store(&b, root, RT_PARENT, nir_imm_int64(&b, 0));
   rt_store(&b, root, RT_RETURN, nir_imm_int(&b, 0));
   rt_store(&b, root, RT_RAY, root);
   rt_store(&b, root, RT_DEPTH, nir_imm_int(&b, 0));
   rt_store(&b, root, RT_RECORD, nir_imm_int64(&b, 0));
   rt_store(&b, root, RT_PAYLOAD, nir_imm_int64(&b, 0));
   rt_store(&b, root, RT_SHADER_RECORD, nir_iadd_imm(&b, sbt, 32));
   rt_select(&b, root, root, rt_token_pc(&b, &ctx, dispatch,
      rt_sbt_token(&b, sbt, 0), MESA_SHADER_RAYGEN));
   if (info->ser)
      nir_pop_if(&b, NULL);
   nir_push_loop(&b);
   nir_def *pc = rt_load(&b, root, RT_PC, 1, 32);
   nir_def *finished = nir_ieq_imm(&b, pc, 0);
   if (info->ser)
      finished = nir_ior(&b, finished, nir_ine_imm(&b,
         rt_load(&b, root, RT_SER_SUSPENDED, 1, 32), 0));
   nir_push_if(&b, finished);
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);
   nir_def *frame = rt_load(&b, root, RT_FRAME, 1, 64);
   nir_def *params[5] = {frame, dispatch, root, NULL, NULL};
   if (info->ser) {
      params[3] = nir_u2u64(&b, ctx.owner);
      params[4] = linear;
   }
   for (unsigned i = 0; i < info->stage_count; i++) {
      struct rt_stage *stage = &ctx.stages[i];
      for (unsigned p = 0; p <= stage->resume_count; p++) {
         nir_shader *part = p ? stage->resumes[p - 1] : stage->nir;
         rt_prepare_part(&ctx, stage, part);
         nir_push_if(&b, nir_ieq_imm(&b, pc,
            p ? stage->resume_base + p - 1 : i + 1));
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
   if (info->ser) {
      nir_push_if(&b, nir_ine_imm(&b,
         rt_load(&b, root, RT_SER_SUSPENDED, 1, 32), 0));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);
      nir_store_var(&b, resumed, nir_imm_false(&b), 1);
   }
   nir_store_var(&b, ray_index, nir_iadd(&b, linear, workers), 1);
   nir_pop_loop(&b, NULL);
   if (info->ser) {
      nir_push_if(&b, nir_ieq_imm(&b,
         rt_load(&b, root, RT_SER_SUSPENDED, 1, 32), 0));
      info->ser->emit_complete(&b, ser_queue, ctx.owner, info->ser->data);
      nir_push_else(&b, NULL);
      info->ser->emit_suspend(&b, ser_queue, ctx.owner, info->ser->data);
      nir_pop_if(&b, NULL);
   }
   nir_lower_returns(b.shader);
   nvk_nir_lower_ray_queries_with_omm(b.shader,
      info->flags & VK_PIPELINE_CREATE_2_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT);
   nir_index_ssa_defs(nir_shader_get_entrypoint(b.shader));
   nir_progress(true, b.impl, nir_metadata_none);
   nir_lower_vars_to_ssa(b.shader);
   nir_opt_dce(b.shader);
   nir_variable_mode local_modes = nir_var_function_temp | nir_var_shader_temp;
   nir_lower_vars_to_explicit_types(b.shader, local_modes,
                                    glsl_get_natural_size_align_bytes);
   nir_lower_explicit_io(b.shader, local_modes,
                         nir_address_format_32bit_offset);
   nir_remove_dead_variables(b.shader, local_modes, NULL);
   uint64_t storage_size = ALIGN_POT((uint64_t)b.shader->scratch_size,
                                     NVK_RT_STACK_LANE_BLOCK_SIZE);
   if (storage_size > UINT32_MAX) {
      ralloc_free(b.shader);
      ralloc_free(tmp);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   if (info->ser || info->force_global_stack || storage_size >= (1u << 24)) {
      result->storage_size = storage_size;
      const struct rt_stack_io_state stack_io = {
         .storage_size = result->storage_size, .owner = ctx.owner,
      };
      nir_shader_intrinsics_pass(b.shader, rt_lower_stack_io, nir_metadata_none,
                                  (void *)&stack_io);
      b.shader->scratch_size = 0;
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   nir_validate_shader(b.shader, "nvk linked RT pipeline");
   ralloc_steal(mem_ctx, b.shader);
   result->nir = b.shader;
   ralloc_free(tmp);
   return VK_SUCCESS;
}
