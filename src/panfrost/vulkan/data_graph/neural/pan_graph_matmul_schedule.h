#ifndef PAN_GRAPH_MATMUL_SCHEDULE_H
#define PAN_GRAPH_MATMUL_SCHEDULE_H

#include "pan_graph_matmul.h"

struct pan_graph_matmul_schedule_policy {
   uint32_t granule[PAN_GRAPH_MATMUL_RANK];
   int32_t axes[4];
   uint8_t allowed_axes;
};

struct pan_graph_matmul_parallel_score {
   uint32_t tile[2];
   uint32_t groups[2];
   uint32_t total_groups;
   float coverage[2];
   float score;
};

struct pan_graph_matmul_parallel_options {
   uint32_t core_budget;
   uint32_t column_divisor;
   uint8_t output_axis_mask;
   bool multiple_axes;
   bool preferred_pair_only;
};

#define PAN_GRAPH_MATMUL_PARALLEL_CAPS_MAX 23

struct pan_graph_matmul_parallel_caps {
   uint32_t extents[PAN_GRAPH_MATMUL_PARALLEL_CAPS_MAX][PAN_GRAPH_MATMUL_RANK];
   uint32_t count;
};

bool pan_graph_matmul_schedule_policy_init(
   const struct pan_graph_matmul_info *info,
   const uint32_t alignment[PAN_GRAPH_MATMUL_RANK], uint8_t output_axis_mask,
   bool multiple_axes, struct pan_graph_matmul_schedule_policy *policy);

bool pan_graph_matmul_parallel_score(
   const uint32_t dimensions[PAN_GRAPH_MATMUL_RANK],
   const uint32_t alignment[PAN_GRAPH_MATMUL_RANK], const int32_t axes[2],
   uint32_t core_budget, uint32_t column_divisor,
   struct pan_graph_matmul_parallel_score *score);

bool pan_graph_matmul_assign_loops(
   const struct pan_graph_matmul_info *info,
   const uint32_t transfer[PAN_GRAPH_MATMUL_RANK],
   const uint32_t granule[PAN_GRAPH_MATMUL_RANK],
   const struct pan_graph_matmul_parallel_options *options, uint32_t loops[4]);

bool pan_graph_assign_loops(
   const uint32_t dimensions[PAN_GRAPH_MATMUL_RANK],
   const uint32_t transfer[PAN_GRAPH_MATMUL_RANK],
   const uint32_t granule[PAN_GRAPH_MATMUL_RANK], uint8_t allowed_axes,
   const struct pan_graph_matmul_parallel_options *options, uint32_t loops[4]);

bool pan_graph_matmul_parallel_caps(
   const struct pan_graph_matmul_info *info,
   const uint32_t granule[PAN_GRAPH_MATMUL_RANK],
   const struct pan_graph_matmul_parallel_options *options,
   struct pan_graph_matmul_parallel_caps *caps);

#endif
