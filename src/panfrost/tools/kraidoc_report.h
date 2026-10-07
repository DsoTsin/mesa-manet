#ifndef KRAIDOC_REPORT_H
#define KRAIDOC_REPORT_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "compiler/shader_enums.h"

struct pan_shader_info;

struct kraidoc_program {
   const struct pan_shader_info *info;
   const void *bin;
   uint32_t size;
};

struct kraidoc_variant {
   unsigned index;
   struct kraidoc_program main;
   struct kraidoc_program pilot;
   const char *nir_str;
};

struct kraidoc_report_context {
   const char *filename, *entrypoint, *input_hash;
   uint64_t gpu_id;
   uint32_t gpu_variant, shader_cores;
   mesa_shader_stage stage;
   bool preamble_enabled;
   const void *context;
   void (*write_context)(FILE *, const void *, const char *);
};

static inline bool
kraidoc_arch_supported(unsigned arch)
{
   return (arch >= 9 && arch <= 11) || arch == 15;
}

bool kraidoc_write_report(FILE *, bool, bool,
                          const struct kraidoc_report_context *,
                          const struct kraidoc_variant *, unsigned);
bool kraidoc_write_list(FILE *, bool);
bool kraidoc_write_info(FILE *, bool, uint64_t, uint32_t);

#endif
