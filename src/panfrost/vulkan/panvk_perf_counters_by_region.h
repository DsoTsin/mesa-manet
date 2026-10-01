/*
 * Copyright © 2026 Pix Philosophy (HK) Limited
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_PERF_COUNTERS_BY_REGION_H
#define PANVK_PERF_COUNTERS_BY_REGION_H

#include <stdint.h>

#define PANVK_PERF_COUNTERS_BY_REGION_COUNT 128
#define PANVK_PERF_COUNTERS_BY_REGION_MAX   4
#define PANVK_PERF_COUNTERS_REGION_SIZE     64
#define PANVK_PERF_COUNTERS_REGION_STRIDE   16
#define PANVK_PERF_COUNTERS_ROW_ALIGN       64

struct panvk_perf_counter_by_region {
   uint32_t id;
   const char *name;
};

extern const struct panvk_perf_counter_by_region
   panvk_perf_counters_by_region[PANVK_PERF_COUNTERS_BY_REGION_COUNT];

uint32_t panvk_perf_counters_by_region_select(const uint32_t *ids,
                                              uint32_t count);

#endif
