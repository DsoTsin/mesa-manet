#ifndef AGC_SHADER_H
#define AGC_SHADER_H

#include "agc_types.h"

#define AGC_SHADER_MAGIC 0x34333231u
#define AGC_SHADER_VERSION 0x18u

#define AGC_ERROR_INVALID_ALIGNMENT ((int32_t)0x8a6c0002u)
#define AGC_ERROR_INVALID_SHADER_PROGRAM ((int32_t)0x8a6c0005u)
#define AGC_ERROR_INVALID_SHADER_HALVES ((int32_t)0x8a6c0008u)
#define AGC_ERROR_INVALID_ARGUMENT ((int32_t)0x8a6c000au)
#define AGC_ERROR_OUT_OF_RANGE ((int32_t)0x8a6c000bu)
#define AGC_ERROR_NOT_A_GEOMETRY_SHADER ((int32_t)0x8a6c003cu)

enum agc_shader_kind {
   AGC_SHADER_CS = 0,
   AGC_SHADER_PS = 1,
   AGC_SHADER_GS = 2,
   AGC_SHADER_HS = 3,
   AGC_SHADER_GS_FRONT = 4,
   AGC_SHADER_HS_FRONT = 5,
   AGC_SHADER_GS_BACK = 6,
   AGC_SHADER_HS_BACK = 7,
   AGC_SHADER_FS = 8,
   AGC_SHADER_KIND_COUNT
};

struct agc_shader_user_data {
   uint16_t *direct_resource_offset;
   uint16_t *sharp_resource_offset[4];
   uint16_t eud_size_dw;
   uint16_t srt_size_dw;
   uint16_t direct_resource_count;
   uint16_t sharp_resource_count[4];
};

struct agc_shader_specials {
   struct agc_register ge_cntl;
   struct agc_register vgt_shader_stages_en;
   uint32_t dispatch_modifier;
   uint16_t user_data_range[2];
   uint32_t draw_modifier[2];
   struct agc_register vgt_gs_out_prim_type;
   struct agc_register ge_user_vgpr_en;
};

struct agc_shader {
   uint32_t file_header;
   uint32_t version;
   struct agc_shader_user_data *user_data;
   uint64_t code;
   struct agc_register *cx_registers;
   struct agc_register *sh_registers;
   struct agc_shader_specials *specials;
   const uint32_t *input_semantics;
   const uint32_t *output_semantics;
   uint32_t header_size;
   uint32_t shader_size;
   uint32_t embedded_constant_buffer_size_dqw;
   uint32_t target;
   uint32_t num_input_semantics;
   uint16_t scratch_size_dw_per_thread;
   uint16_t num_output_semantics;
   uint16_t special_sizes_bytes;
   uint8_t kind;
   uint8_t num_cx_registers;
   uint8_t num_sh_registers;
   uint8_t reserved[3];
};

typedef int32_t (*agc_shader_program_fn)(struct agc_shader *shader, uint64_t guest_code, uint64_t *host_code);

void agc_shader_set_program_hook(agc_shader_program_fn hook);

bool agc_shader_program_register(uint8_t kind, uint32_t *lo);
int32_t agc_shader_rebase_program(struct agc_register *registers, uint32_t count, uint32_t lo, uint64_t base);
int32_t agc_shader_create(struct agc_shader **destination, struct agc_shader *header, uint64_t code);
int32_t agc_shader_gs_prim_payload(uint32_t *payload, const struct agc_shader *gs);
int32_t agc_shader_create_prim_state(struct agc_register *cx, struct agc_register *uc,
                                     const struct agc_shader *hs, const struct agc_shader *gs,
                                     uint32_t primitive);
int32_t agc_shader_update_prim_state(struct agc_register *cx, struct agc_register *uc, uint32_t primitive);
int32_t agc_shader_interpolant_mapping(struct agc_register *registers, const struct agc_shader *gs,
                                       const struct agc_shader *ps, bool fill, const char *function);
int32_t agc_shader_gs_oversubscription(struct agc_register *registers, const struct agc_shader *gs,
                                       int32_t budget, float factor);

#endif
