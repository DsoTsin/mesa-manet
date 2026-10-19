#ifndef PAN_GRAPH_COST_H
#define PAN_GRAPH_COST_H

#include <stdbool.h>
#include <stdint.h>

#define PAN_GRAPH_COST_MAX_RANK 12

struct pan_graph_cost_report {
   uint64_t resource_bytes[2][6];
   uint64_t peak_resource[2];
   uint64_t memory_cycles[6];
   uint64_t engine_cycles[8];
   uint32_t critical_kind;
   uint64_t critical_cycles;
   uint64_t cycles;
};

struct pan_graph_cost_workload {
   uint64_t extent[PAN_GRAPH_COST_MAX_RANK];
   uint64_t transfer[PAN_GRAPH_COST_MAX_RANK];
   uint32_t step[2];
   uint8_t axis[2];
   uint8_t rank;
   uint32_t repeats;
   uint32_t latency;
   float minimum;
};

enum pan_graph_cost_ve_i32_op {
   PAN_GRAPH_COST_VE_I32_IDENTITY,
   PAN_GRAPH_COST_VE_I32_ADD,
};

bool pan_graph_cost_v15_ve_i32(enum pan_graph_cost_ve_i32_op op,
                              const uint32_t extent[4],
                              const uint32_t transfer[4], uint64_t *cycles);

bool pan_graph_cost_combine(const struct pan_graph_cost_report a[3],
                            const struct pan_graph_cost_report b[3],
                            struct pan_graph_cost_report *result);

bool pan_graph_cost_accumulate_serial(const struct pan_graph_cost_report *left,
                                      const struct pan_graph_cost_report *right,
                                      struct pan_graph_cost_report *result);

bool pan_graph_cost_apply_workload(const struct pan_graph_cost_workload *workload,
                                   uint64_t cycles, uint64_t *result);

#endif
