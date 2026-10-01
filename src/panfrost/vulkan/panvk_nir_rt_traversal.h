#ifndef PANVK_NIR_RT_TRAVERSAL_H
#define PANVK_NIR_RT_TRAVERSAL_H

#include "nir_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

enum panvk_nir_rt_action {
   PANVK_NIR_RT_ACCEPT = 0,
   PANVK_NIR_RT_IGNORE = 1,
   PANVK_NIR_RT_TERMINATE = 2,
};

struct panvk_nir_rt_sbt {
   nir_def *base;
   nir_def *stride;
};

struct panvk_nir_rt_trace {
   nir_def *state_addr;
   nir_def *accel_struct;
   nir_def *flags;
   nir_def *cull_mask;
   nir_def *sbt_offset;
   nir_def *sbt_stride;
   nir_def *miss_index;
   nir_def *origin;
   nir_def *tmin;
   nir_def *direction;
   nir_def *tmax;
   struct panvk_nir_rt_sbt hit;
   struct panvk_nir_rt_sbt miss;
};

struct panvk_nir_rt_event {
   const struct panvk_nir_rt_trace *trace;
   nir_def *record_addr;
   nir_def *sbt_addr;
   nir_def *sbt_index;
   nir_def *t;
   nir_def *hit_kind;
};

struct panvk_nir_rt_report_context;

struct panvk_nir_rt_callbacks {
   nir_def *(*any_hit)(nir_builder *b,
                      const struct panvk_nir_rt_event *event, void *data);
   void (*intersection)(nir_builder *b,
                        const struct panvk_nir_rt_event *event,
                        struct panvk_nir_rt_report_context *report, void *data);
   void (*intersection_terminate)(nir_builder *b, void *data);
   void (*closest_hit)(nir_builder *b,
                       const struct panvk_nir_rt_event *event, void *data);
   void (*miss)(nir_builder *b,
                const struct panvk_nir_rt_event *event, void *data);
};

struct panvk_nir_rt_result {
   nir_def *hit;
   nir_def *t;
   nir_def *hit_kind;
};

nir_def *panvk_nir_rt_pack_sbt(nir_builder *b, nir_def *sbt_offset,
                               nir_def *sbt_stride, nir_def *miss_index);

nir_def *panvk_nir_rt_sbt_address(nir_builder *b,
                                  const struct panvk_nir_rt_sbt *sbt,
                                  nir_def *index);

nir_def *panvk_nir_rt_report_tmax(
   nir_builder *b, struct panvk_nir_rt_report_context *report);

nir_def *panvk_nir_rt_report_terminated(
   nir_builder *b, struct panvk_nir_rt_report_context *report);

nir_def *panvk_nir_rt_report_intersection(
   nir_builder *b, struct panvk_nir_rt_report_context *report,
   const struct panvk_nir_rt_event *event, nir_def *t,
   nir_def *hit_kind, nir_def *opaque);

struct panvk_nir_rt_result panvk_nir_rt_emit_traversal(
   nir_builder *b, const struct panvk_nir_rt_trace *trace,
   const struct panvk_nir_rt_callbacks *callbacks, void *data);

#ifdef __cplusplus
}
#endif

#endif
