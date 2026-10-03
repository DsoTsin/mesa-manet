#ifndef KRAIDOC_REPORT_H
#define KRAIDOC_REPORT_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "compiler/shader_enums.h"

struct panvk_shader;

struct kraidoc_report_context {
   const char *filename, *entrypoint, *input_hash;
   uint64_t gpu_id;
   uint32_t gpu_variant, shader_cores;
   mesa_shader_stage stage;
   bool preamble_enabled;
   const void *context;
   void (*write_context)(FILE *, const void *, const char *);
};

bool kraidoc_write_report(FILE *, bool, bool,
                          const struct kraidoc_report_context *,
                          const struct panvk_shader *);
bool kraidoc_write_list(FILE *, bool);
bool kraidoc_write_info(FILE *, bool, uint64_t, uint32_t);

#endif
