#include "panvk_nir_rt_traversal.h"
#include "compiler/spirv/spirv.h"

#define RT_COMMITTED 0x24
#define RT_CANDIDATE 0x64
#define RT_REC_PRIM 0x0c
#define RT_REC_KIND 0x10
#define RT_SBT_MASK 0x0fffffff

struct panvk_nir_rt_report_context {
   const struct panvk_nir_rt_trace *trace;
   const struct panvk_nir_rt_callbacks *callbacks;
   void *data;
   nir_variable *terminated;
   nir_variable *committed_t;
   nir_variable *generated_hit_kind;
};

static nir_def *
rt_load(nir_builder *b, nir_def *addr, unsigned offset, unsigned components)
{
   return nir_load_global(b, components, 32, nir_iadd_imm(b, addr, offset),
                          .align_mul = 4);
}

static void
rt_store(nir_builder *b, nir_def *addr, unsigned offset, nir_def *value)
{
   nir_store_global(b, value, nir_iadd_imm(b, addr, offset), .align_mul = 4);
}

static nir_def *
rt_var_load(nir_builder *b, nir_variable *var)
{
   return nir_load_var(b, var);
}

static void
rt_var_store(nir_builder *b, nir_variable *var, nir_def *value)
{
   nir_store_var(b, var, value, 1);
}

static void
rt_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_NONE,
               .memory_scope = SCOPE_SUBGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
}

nir_def *
panvk_nir_rt_pack_sbt(nir_builder *b, nir_def *sbt_offset,
                      nir_def *sbt_stride, nir_def *miss_index)
{
   return nir_ior(b, nir_iand_imm(b, sbt_offset, 15),
                   nir_ior(b, nir_ishl_imm(b, nir_iand_imm(b, sbt_stride, 15), 4),
                              nir_ishl_imm(b, nir_iand_imm(b, miss_index, 65535), 8)));
}

nir_def *
panvk_nir_rt_sbt_address(nir_builder *b,
                         const struct panvk_nir_rt_sbt *sbt, nir_def *index)
{
   return nir_bcsel(b, nir_ine_imm(b, sbt->base, 0),
                   nir_iadd(b, sbt->base,
                            nir_imul(b, nir_u2u64(b, nir_iand_imm(b, index, RT_SBT_MASK)),
                                        nir_u2u64(b, sbt->stride))),
                   nir_imm_int64(b, 0));
}

static nir_def *
rt_root(nir_builder *b, nir_def *as)
{
   return nir_bcsel(b, nir_ine_imm(b, as, 0), nir_iadd_imm(b, as, 128),
                    nir_imm_int64(b, 0));
}

static nir_def *
rt_triangle_hit_kind(nir_builder *b, nir_def *kind)
{
   return nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, kind, 1u << 24), 0),
                    nir_imm_int(b, 0xfe), nir_imm_int(b, 0xff));
}

static struct panvk_nir_rt_event
rt_event(nir_builder *b, const struct panvk_nir_rt_trace *trace,
          nir_def *record, const struct panvk_nir_rt_sbt *sbt,
          nir_def *sbt_index, nir_def *t, nir_def *hit_kind)
{
   return (struct panvk_nir_rt_event){
      .trace = trace,
      .record_addr = record,
      .sbt_addr = panvk_nir_rt_sbt_address(b, sbt, sbt_index),
      .sbt_index = nir_iand_imm(b, sbt_index, RT_SBT_MASK),
      .t = t,
      .hit_kind = hit_kind,
   };
}

static void
rt_commit(nir_builder *b, struct panvk_nir_rt_report_context *report,
           const struct panvk_nir_rt_event *event, nir_def *t,
           nir_def *hit_kind, bool generated)
{
   nir_def *lo = rt_load(b, event->record_addr, 0, 4);
   nir_def *hi = rt_load(b, event->record_addr, 16, 3);
   nir_def *parts[7];
   for (unsigned i = 0; i < 7; i++)
      parts[i] = nir_channel(b, i < 4 ? lo : hi, i % 4);
   parts[0] = t;
   parts[4] = generated
      ? nir_ior_imm(b, nir_iand_imm(b, parts[4], 0x1fffffff), 0xe0000000)
      : nir_ior_imm(b, parts[4], 0x80000000);
   rt_store(b, report->trace->state_addr, RT_COMMITTED, nir_vec(b, parts, 4));
   rt_store(b, report->trace->state_addr, RT_COMMITTED + 16,
             nir_vec(b, &parts[4], 3));
   rt_var_store(b, report->committed_t, t);
   if (generated)
      rt_var_store(b, report->generated_hit_kind, hit_kind);
}

static nir_def *
rt_any_hit(nir_builder *b, struct panvk_nir_rt_report_context *report,
            const struct panvk_nir_rt_event *event)
{
   if (!report->callbacks || !report->callbacks->any_hit)
      return nir_imm_int(b, PANVK_NIR_RT_ACCEPT);
   return report->callbacks->any_hit(b, event, report->data);
}

static void
rt_check_termination(nir_builder *b,
                      struct panvk_nir_rt_report_context *report,
                      nir_def *action)
{
   nir_def *terminate = nir_ior(
      b, nir_ieq_imm(b, action, PANVK_NIR_RT_TERMINATE),
      nir_ine_imm(b, nir_iand_imm(b, report->trace->flags,
                                 SpvRayFlagsTerminateOnFirstHitKHRMask), 0));
   rt_var_store(b, report->terminated,
                 nir_ior(b, rt_var_load(b, report->terminated), terminate));
}

nir_def *
panvk_nir_rt_report_tmax(
   nir_builder *b, struct panvk_nir_rt_report_context *report)
{
   return rt_var_load(b, report->committed_t);
}

nir_def *
panvk_nir_rt_report_terminated(
   nir_builder *b, struct panvk_nir_rt_report_context *report)
{
   return rt_var_load(b, report->terminated);
}

nir_def *
panvk_nir_rt_report_intersection(
   nir_builder *b, struct panvk_nir_rt_report_context *report,
   const struct panvk_nir_rt_event *event, nir_def *t,
   nir_def *hit_kind, nir_def *opaque)
{
   nir_variable *accepted = nir_local_variable_create(
      b->impl, glsl_bool_type(), "rt_report_accepted");
   nir_variable *action = nir_local_variable_create(
      b->impl, glsl_uint_type(), "rt_report_action");
   rt_var_store(b, accepted, nir_imm_false(b));

   uint32_t fp_math_ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_fp_no_fast_math;
   nir_def *valid = nir_iand(
      b, nir_fge(b, t, report->trace->tmin),
      nir_fge(b, rt_var_load(b, report->committed_t), t));
   b->fp_math_ctrl = fp_math_ctrl;
   valid = nir_iand(b, valid, nir_inot(b, rt_var_load(b, report->terminated)));
   nir_push_if(b, valid);
   {
      struct panvk_nir_rt_event reported = *event;
      reported.t = t;
      reported.hit_kind = hit_kind;
      rt_var_store(b, action, nir_imm_int(b, PANVK_NIR_RT_ACCEPT));
      if (report->callbacks && report->callbacks->any_hit) {
         nir_push_if(b, nir_inot(b, opaque));
         rt_var_store(b, action, rt_any_hit(b, report, &reported));
         nir_pop_if(b, NULL);
      }
      nir_push_if(b, nir_ine_imm(b, rt_var_load(b, action), PANVK_NIR_RT_IGNORE));
      {
         rt_commit(b, report, &reported, t, hit_kind, true);
         rt_var_store(b, accepted, nir_imm_true(b));
         rt_check_termination(b, report, rt_var_load(b, action));
         nir_push_if(b, panvk_nir_rt_report_terminated(b, report));
         report->callbacks->intersection_terminate(b, report->data);
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
   return rt_var_load(b, accepted);
}

struct panvk_nir_rt_result
panvk_nir_rt_emit_traversal(
   nir_builder *b, const struct panvk_nir_rt_trace *trace,
   const struct panvk_nir_rt_callbacks *callbacks, void *data)
{
   assert(!callbacks || !callbacks->intersection ||
          callbacks->intersection_terminate);
   struct panvk_nir_rt_report_context report = {
      .trace = trace,
      .callbacks = callbacks,
      .data = data,
      .terminated = nir_local_variable_create(b->impl, glsl_bool_type(),
                                               "rt_terminated"),
      .committed_t = nir_local_variable_create(b->impl, glsl_float_type(),
                                                "rt_committed_t"),
      .generated_hit_kind = nir_local_variable_create(b->impl, glsl_uint_type(),
                                                       "rt_generated_hit_kind"),
   };
   nir_variable *status = nir_local_variable_create(
      b->impl, glsl_uint_type(), "rt_status");
   nir_def *root = rt_root(b, trace->accel_struct);
   nir_def *candidate = nir_iadd_imm(b, trace->state_addr, RT_CANDIDATE);
   nir_def *committed = nir_iadd_imm(b, trace->state_addr, RT_COMMITTED);

   rt_var_store(b, report.terminated, nir_imm_false(b));
   rt_var_store(b, report.generated_hit_kind, nir_imm_int(b, 0));
   rt_store(b, committed, RT_REC_KIND, nir_imm_int(b, 0));
   rt_barrier(b);
   nir_def *out = nir_rt_trace_begin_pan(
      b, trace->state_addr, root, trace->origin, trace->direction,
      trace->tmin, trace->tmax,
      nir_ior(b, trace->flags,
                 nir_ishl_imm(b, nir_iand_imm(b, trace->cull_mask, 255), 24)),
      panvk_nir_rt_pack_sbt(b, trace->sbt_offset, trace->sbt_stride,
                            trace->miss_index));
   rt_barrier(b);
   rt_var_store(b, status, nir_channel(b, out, 0));
   rt_var_store(b, report.committed_t, nir_channel(b, out, 1));

   nir_push_loop(b);
   {
      nir_def *code = nir_ushr_imm(b, rt_var_load(b, status), 29);
      nir_def *is_candidate = nir_ior(b, nir_ieq_imm(b, code, 1),
                                         nir_ieq_imm(b, code, 2));
      nir_push_if(b, nir_ior(b, nir_inot(b, is_candidate),
                               rt_var_load(b, report.terminated)));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_def *kind = rt_load(b, candidate, RT_REC_KIND, 1);
      struct panvk_nir_rt_event event = rt_event(
         b, trace, candidate, &trace->hit, rt_var_load(b, status),
         rt_load(b, candidate, 0, 1), rt_triangle_hit_kind(b, kind));
      nir_push_if(b, nir_ieq_imm(b, code, 1));
      {
         nir_def *action = rt_any_hit(b, &report, &event);
         nir_push_if(b, nir_ine_imm(b, action, PANVK_NIR_RT_IGNORE));
         rt_commit(b, &report, &event, event.t, event.hit_kind, false);
         rt_check_termination(b, &report, action);
         nir_pop_if(b, NULL);
      }
      nir_push_else(b, NULL);
      if (callbacks && callbacks->intersection)
         callbacks->intersection(b, &event, &report, data);
      nir_pop_if(b, NULL);

      nir_push_if(b, rt_var_load(b, report.terminated));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      rt_barrier(b);
      out = nir_rt_trace_resume_pan(b, trace->state_addr, root,
                                     rt_load(b, candidate, RT_REC_PRIM, 2));
      rt_barrier(b);
      rt_var_store(b, status, nir_channel(b, out, 0));
      rt_var_store(b, report.committed_t, nir_channel(b, out, 1));
   }
   nir_pop_loop(b, NULL);

   nir_def *committed_kind = rt_load(b, committed, RT_REC_KIND, 1);
   nir_def *type = nir_ushr_imm(b, committed_kind, 29);
   nir_def *hit = nir_ior(b, nir_ieq_imm(b, type, 5), nir_ieq_imm(b, type, 7));
   nir_def *hit_kind = nir_bcsel(
      b, nir_ieq_imm(b, type, 5), rt_triangle_hit_kind(b, committed_kind),
      rt_var_load(b, report.generated_hit_kind));

   nir_push_if(b, hit);
   if (callbacks && callbacks->closest_hit) {
      nir_push_if(b, nir_ieq_imm(
         b, nir_iand_imm(b, trace->flags, SpvRayFlagsSkipClosestHitShaderKHRMask), 0));
      struct panvk_nir_rt_event event = rt_event(
         b, trace, committed, &trace->hit,
         rt_var_load(b, status),
         rt_var_load(b, report.committed_t), hit_kind);
      callbacks->closest_hit(b, &event, data);
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   if (callbacks && callbacks->miss) {
      struct panvk_nir_rt_event event = rt_event(
         b, trace, nir_imm_int64(b, 0), &trace->miss,
         nir_iand_imm(b, trace->miss_index, 65535), trace->tmax, nir_imm_int(b, 0));
      callbacks->miss(b, &event, data);
   }
   nir_pop_if(b, NULL);

   return (struct panvk_nir_rt_result){
      .hit = hit,
      .t = rt_var_load(b, report.committed_t),
      .hit_kind = hit_kind,
   };
}
