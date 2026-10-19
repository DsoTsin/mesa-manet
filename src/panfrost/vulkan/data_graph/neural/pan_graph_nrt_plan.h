#ifndef PAN_GRAPH_NRT_PLAN_H
#define PAN_GRAPH_NRT_PLAN_H

#include <stddef.h>

#include "pan_graph_nrt.h"

#define PAN_GRAPH_NRT_PLAN_MAX_RESOURCES 15

enum pan_graph_nrt_buffer_group {
   PAN_GRAPH_NRT_BUFFER_FIXED,
   PAN_GRAPH_NRT_BUFFER_INPUT,
   PAN_GRAPH_NRT_BUFFER_WEIGHT,
   PAN_GRAPH_NRT_BUFFER_OUTPUT,
};

struct pan_graph_nrt_plan_request {
   struct pan_graph_nrt_request allocation;
   enum pan_graph_nrt_buffer_group buffers;
};

struct pan_graph_nrt_plan {
   struct pan_graph_nrt_layout layouts[PAN_GRAPH_NRT_PLAN_MAX_RESOURCES];
   uint32_t buffers[3];
};

enum pan_graph_nrt_status pan_graph_nrt_plan_trial(
   const struct pan_graph_nrt_target *target,
   const struct pan_graph_nrt_plan_request *requests, size_t count,
   const uint32_t buffers[3], struct pan_graph_nrt_plan *plan);

enum pan_graph_nrt_status pan_graph_nrt_plan_search(
   const struct pan_graph_nrt_target *target,
   const struct pan_graph_nrt_plan_request *requests, size_t count,
   uint32_t buffer_limit, struct pan_graph_nrt_plan *plan);

#endif
