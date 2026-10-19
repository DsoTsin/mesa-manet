#ifndef PAN_GRAPH_TSU_H
#define PAN_GRAPH_TSU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define PAN_GRAPH_TSU_MAX_RANK 16

enum pan_graph_tsu_status {
   PAN_GRAPH_TSU_SUCCESS,
   PAN_GRAPH_TSU_INVALID,
   PAN_GRAPH_TSU_UNSUPPORTED,
   PAN_GRAPH_TSU_OUT_OF_MEMORY,
};

enum pan_graph_tsu_transform_kind {
   PAN_GRAPH_TSU_AFFINE,
   PAN_GRAPH_TSU_DIVIDE,
   PAN_GRAPH_TSU_SPLIT,
   PAN_GRAPH_TSU_NEGATE,
   PAN_GRAPH_TSU_CLAMP,
   PAN_GRAPH_TSU_SUBTRACT,
};

struct pan_graph_tsu_transform {
   enum pan_graph_tsu_transform_kind kind;
   uint32_t rows;
   uint32_t columns;
   uint32_t index;
   uint32_t other;
   int64_t value;
   const int64_t *low;
   const int64_t *high;
   const int64_t *divisors;
};

struct pan_graph_tsu_info {
   uint32_t rank;
   const int64_t *low;
   const int64_t *high;
   size_t count;
   const struct pan_graph_tsu_transform *transforms;
};

struct pan_graph_tsu_instr {
   uint32_t opcode;
   uint32_t increment;
   uint32_t flag;
   uint32_t reg;
   uint64_t immediate;
   uint32_t shift;
   uint32_t reserved;
   uint32_t size;
   uint32_t reserved2;
};

struct pan_graph_tsu_program {
   size_t count;
   struct pan_graph_tsu_instr *instructions;
};

enum pan_graph_tsu_status
pan_graph_tsu_compile(const struct pan_graph_tsu_info *info,
                      struct pan_graph_tsu_program *program);

enum pan_graph_tsu_status
pan_graph_tsu_link(const struct pan_graph_tsu_info *infos, size_t count,
                   struct pan_graph_tsu_program *program, uint32_t *offsets);

void pan_graph_tsu_cleanup(struct pan_graph_tsu_program *program);

#endif
