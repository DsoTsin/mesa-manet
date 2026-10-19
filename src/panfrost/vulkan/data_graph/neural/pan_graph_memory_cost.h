#ifndef PAN_GRAPH_MEMORY_COST_H
#define PAN_GRAPH_MEMORY_COST_H

#include "pan_graph_cost.h"
#include "pan_graph_v15.h"

struct pan_graph_memory_cost_resource {
   float bandwidth[2];
   uint32_t latency[2];
   uint32_t parallel;
   uint32_t line_size;
   uint64_t size;
};

struct pan_graph_memory_cost_target {
   struct pan_graph_memory_cost_resource memory[6];
   uint32_t descriptor_bytes[4];
   uint32_t section_bytes[PAN_GRAPH_V15_SECTION_TYPES];
   uint32_t alignment;
   uint32_t issue_width;
   uint32_t parallel_budget;
};

enum pan_graph_memory_cost_view {
   PAN_GRAPH_MEMORY_COST_TENSOR_VIEW = 1,
   PAN_GRAPH_MEMORY_COST_WEIGHT_VIEW = 2,
};

enum pan_graph_memory_cost_layout {
   PAN_GRAPH_MEMORY_COST_COMPACT_LINEAR,
   PAN_GRAPH_MEMORY_COST_BLOCKED_260 = 260,
   PAN_GRAPH_MEMORY_COST_BLOCKED_261 = 261,
   PAN_GRAPH_MEMORY_COST_BLOCKED_262 = 262,
};

struct pan_graph_memory_cost_section {
   enum pan_graph_v15_section_type type;
   enum pan_graph_memory_cost_layout layout;
   uint32_t full[3];
   uint32_t transfer[3];
   uint32_t block[4];
   uint8_t bits;
   uint8_t replication_rank;
   bool retain_read_peak;
};

struct pan_graph_memory_cost_program {
   const struct pan_graph_memory_cost_section *sections;
   uint32_t section_count;
   const enum pan_graph_memory_cost_view *input_views;
   uint32_t input_view_count;
   uint32_t output_view_count;
   uint32_t resource_count;
   struct pan_graph_cost_workload workload;
};

bool pan_graph_memory_cost_estimate(
   const struct pan_graph_memory_cost_target *target,
   const struct pan_graph_memory_cost_program *program,
   struct pan_graph_cost_report reports[3]);

#endif
