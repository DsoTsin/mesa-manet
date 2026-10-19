#ifndef PAN_GRAPH_V15_H
#define PAN_GRAPH_V15_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pan_graph_tsu.h"

#define PAN_GRAPH_V15_HEADER_SIZE 36
#define PAN_GRAPH_V15_SECTION_TYPES 8

enum pan_graph_v15_section_type {
   PAN_GRAPH_V15_MEMORY_INPUT = 1,
   PAN_GRAPH_V15_CE = 2,
   PAN_GRAPH_V15_VE = 3,
   PAN_GRAPH_V15_TRANSFORM = 4,
   PAN_GRAPH_V15_MEMORY_CONSTANT = 5,
   PAN_GRAPH_V15_MEMORY_OUTPUT = 6,
   PAN_GRAPH_V15_SAD = 7,
};

struct pan_graph_v15_tsu_instr {
   uint32_t opcode;
   uint32_t increment;
   uint32_t flag;
   uint32_t reg;
   uint32_t immediate;
   uint32_t mode;
};

struct pan_graph_v15_layout {
   uint32_t count[PAN_GRAPH_V15_SECTION_TYPES];
   size_t offset[PAN_GRAPH_V15_SECTION_TYPES];
   size_t tsu_offset;
   size_t size;
};

struct pan_graph_v15_header {
   uint32_t transfer[8];
   uint32_t output[8];
};

struct pan_graph_v15_common {
   uint32_t registers;
   uint32_t tsu_offset;
   uint32_t instance;
   uint32_t input[6];
   uint32_t output;
   bool suppress_registers;
};

struct pan_graph_v15_memory {
   uint32_t domain;
   uint32_t domain_bias;
   uint32_t offset;
   uint8_t level;
   uint8_t override;
   uint8_t flags;
   uint8_t value;
   uint8_t extra;
};

struct pan_graph_v15_ce {
   uint32_t opcode;
   uint32_t width;
   uint32_t height;
   uint32_t parameters[13];
   uint32_t zero_point;
   uint32_t bias_zero_point;
   uint8_t wide;
   uint8_t flags[2];
   bool has_bias_zero_point;
   bool input_nearest_2x;
};

struct pan_graph_v15_rescale {
   int32_t minimum;
   int32_t maximum;
   uint32_t mode;
   uint32_t input_zero_point;
   uint32_t output_zero_point;
   uint32_t multiplier;
   uint32_t shift;
   uint32_t precision;
   uint32_t coefficient_binding;
   bool clamp;
};

enum pan_graph_v15_ve_family {
   PAN_GRAPH_V15_VE_RESCALE,
   PAN_GRAPH_V15_VE_BINARY,
};

enum pan_graph_v15_binary_op {
   PAN_GRAPH_V15_BINARY_ADD,
};

struct pan_graph_v15_binary {
   enum pan_graph_v15_binary_op op;
};

struct pan_graph_v15_transform {
   uint32_t format;
   uint32_t count;
   uint32_t swizzle;
   uint32_t mode;
   uint32_t immediate;
   uint32_t value;
   uint32_t plane;
};

struct pan_graph_v15_sad {
   uint32_t width;
   uint32_t height;
   uint32_t parameters[13];
   bool motion_vector;
};

struct pan_graph_v15_section {
   enum pan_graph_v15_section_type type;
   struct pan_graph_v15_common common;
   enum pan_graph_v15_ve_family ve_family;
   union {
      struct pan_graph_v15_memory memory;
      struct pan_graph_v15_ce ce;
      struct pan_graph_v15_rescale rescale;
      struct pan_graph_v15_binary binary;
      struct pan_graph_v15_transform transform;
      struct pan_graph_v15_sad sad;
   };
};

struct pan_graph_v15_nrt {
   uint32_t kind;
   uint32_t width;
   uint32_t size;
   uint32_t flags;
   int32_t stride;
   uint32_t tile;
   uint32_t mode;
   uint32_t dimensions[4];
   uint32_t format;
   uint32_t binding;
   uint32_t plane;
   uint32_t transfer[8];
   bool wide;
   bool packed;
   bool shared_buffer;
   bool conv3d;
};

struct pan_graph_v15_program_info {
   struct pan_graph_v15_header header;
   size_t section_count;
   const struct pan_graph_v15_section *sections;
   const uint32_t *section_tsu;
   size_t resource_count;
   const struct pan_graph_v15_nrt *resources;
   size_t tsu_count;
   const struct pan_graph_tsu_info *tsu;
};

struct pan_graph_v15_program {
   size_t size;
   void *data;
};

enum pan_graph_tsu_status
pan_graph_v15_compile_tsu(void *dst, size_t size,
                          const struct pan_graph_tsu_info *info, size_t *written);
enum pan_graph_tsu_status
pan_graph_v15_compile(const struct pan_graph_v15_program_info *info,
                      struct pan_graph_v15_program *program);
void pan_graph_v15_program_cleanup(struct pan_graph_v15_program *program);

bool pan_graph_v15_layout_init(struct pan_graph_v15_layout *layout,
                              const uint32_t counts[8], size_t tsu_size);
bool pan_graph_v15_pack_header(void *dst, size_t size,
                              const struct pan_graph_v15_layout *layout,
                              const struct pan_graph_v15_header *info);
bool pan_graph_v15_pack_tsu(void *dst, size_t size,
                           const struct pan_graph_v15_tsu_instr *instructions,
                           size_t count, size_t *written);
bool pan_graph_v15_pack_section(void *dst, size_t size,
                               const struct pan_graph_v15_section *info);
bool pan_graph_v15_pack_nrt(void *dst, size_t size,
                           const struct pan_graph_v15_nrt *info);

#endif
