#ifndef PAN_GRAPH_PROGRAM_DEPS_H
#define PAN_GRAPH_PROGRAM_DEPS_H

#include "pan_graph_program_memory.h"

struct pan_graph_program_deps {
   uint32_t *offsets;
   uint32_t *indices;
   uint32_t program_count;
   uint32_t index_count;
};

enum pan_graph_tsu_status pan_graph_program_deps_matmul_add_v15(
   const struct pan_graph_matmul_info *info, uint32_t add_count,
   const struct pan_graph_program_memory *memory,
   struct pan_graph_program_deps *result);

void pan_graph_program_deps_finish(struct pan_graph_program_deps *deps);

uint16_t pan_graph_program_deps_wait_mask(
   const struct pan_graph_program_deps *deps, uint32_t program,
   unsigned first_slot, unsigned slot_count, uint8_t *waited);

#endif
