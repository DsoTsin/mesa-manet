#ifndef PAN_GRAPH_MEMORY_POLICY_H
#define PAN_GRAPH_MEMORY_POLICY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pan_graph_tsu.h"

#define PAN_GRAPH_MEMORY_POLICY_MAX_RANK 7

struct pan_graph_memory_policy {
   uint32_t level;
   uint8_t override;
   uint8_t flags;
   uint8_t value;
};

struct pan_graph_memory_policy_view {
   uint64_t uid;
   struct pan_graph_memory_policy policy;
};

struct pan_graph_memory_policy_input {
   const struct pan_graph_memory_policy_view *inputs;
   size_t input_count;
   const struct pan_graph_memory_policy_view *outputs;
   size_t output_count;
};

bool pan_graph_memory_policy_lookup(
   const struct pan_graph_memory_policy_input *input, uint64_t uid,
   struct pan_graph_memory_policy *policy);

struct pan_graph_memory_policy_source_view {
   uint64_t uid;
   uint32_t key;
   uint32_t kind;
   uint32_t rank;
   int32_t low[PAN_GRAPH_MEMORY_POLICY_MAX_RANK];
   int32_t high[PAN_GRAPH_MEMORY_POLICY_MAX_RANK];
   struct pan_graph_memory_policy policy;
};

struct pan_graph_memory_policy_snapshot {
   uint32_t key;
   uint32_t level;
};

struct pan_graph_memory_policy_finalize_input {
   const struct pan_graph_memory_policy_source_view *inputs;
   size_t input_count;
   const struct pan_graph_memory_policy_source_view *outputs;
   size_t output_count;
   const struct pan_graph_memory_policy_snapshot *input_snapshot;
   size_t input_snapshot_count;
   const struct pan_graph_memory_policy_snapshot *output_snapshot;
   size_t output_snapshot_count;
   uint8_t load_speculative;
   uint8_t store_speculative;
};

struct pan_graph_memory_policy_finalized {
   struct pan_graph_memory_policy *policies;
   size_t input_count;
   size_t output_count;
};

enum pan_graph_tsu_status pan_graph_memory_policy_finalize(
   const struct pan_graph_memory_policy_finalize_input *input,
   struct pan_graph_memory_policy_finalized *output);

void pan_graph_memory_policy_finalized_finish(
   struct pan_graph_memory_policy_finalized *output);

#endif
