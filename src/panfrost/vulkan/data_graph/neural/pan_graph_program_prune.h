#ifndef PAN_GRAPH_PROGRAM_PRUNE_H
#define PAN_GRAPH_PROGRAM_PRUNE_H

#include "pan_graph_cost.h"
#include "pan_graph_tsu.h"

struct pan_graph_program_cost {
   const uint64_t *groups;
   struct pan_graph_cost_report cost;
};

struct pan_graph_program_prune_entry {
   bool retained;
   struct pan_graph_cost_report cost;
};

struct pan_graph_program_prune_result {
   size_t count;
   struct pan_graph_program_prune_entry *entries;
};

struct pan_graph_program_choice {
   size_t count;
   uint64_t *selected;
   uint64_t cycles;
};

enum pan_graph_program_choice_policy {
   PAN_GRAPH_PROGRAM_CHOICE_CYCLES,
   PAN_GRAPH_PROGRAM_CHOICE_COUNT,
};

enum pan_graph_tsu_status pan_graph_program_prune_costs(
   uint32_t group_count, const struct pan_graph_program_cost *candidates,
   size_t count, struct pan_graph_program_prune_result *result);

void pan_graph_program_prune_finish(struct pan_graph_program_prune_result *result);

enum pan_graph_tsu_status pan_graph_program_choose_costs(
   uint32_t group_count, const uint64_t *request,
   const struct pan_graph_program_cost *candidates, size_t count,
   struct pan_graph_program_choice *result);

enum pan_graph_tsu_status pan_graph_program_choose(
   uint32_t group_count, const uint64_t *request,
   const struct pan_graph_program_cost *candidates, size_t count,
   enum pan_graph_program_choice_policy policy,
   struct pan_graph_program_choice *result);

void pan_graph_program_choice_finish(struct pan_graph_program_choice *result);

enum pan_graph_tsu_status pan_graph_program_select_costs(
   uint32_t group_count, const struct pan_graph_program_cost *candidates,
   size_t count, struct pan_graph_program_choice *result);

#endif
