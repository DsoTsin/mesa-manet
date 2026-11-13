#include "agc_shader.h"

#include <math.h>
#include <string.h>

#include "agc_debug.h"

#define REG_SPI_PS_INPUT_CNTL_0 0x191u
#define REG_GE_MAX_OUTPUT_PER_SUBGROUP 0x1ffu
#define REG_SPI_VS_OUT_CONFIG 0x1b1u
#define REG_SPI_SHADER_COL_FORMAT_ALIAS 0x1c2u
#define REG_PA_CL_VS_OUT_CNTL 0x207u
#define REG_VGT_PRIMITIVE_TYPE 0x242u
#define REG_GE_CNTL 0x25bu
#define REG_GE_PC_ALLOC 0x260u
#define REG_VGT_GS_ONCHIP_CNTL 0x291u
#define REG_VGT_GS_OUT_PRIM_TYPE 0x29bu
#define REG_GE_NGG_SUBGRP_CNTL 0x2d3u
#define REG_VGT_SHADER_STAGES_EN 0x2d5u
#define REG_SPI_SHADER_PGM_RSRC4_GS 0x081u

_Static_assert(sizeof(struct agc_shader) == 0x60, "Shader");
_Static_assert(sizeof(struct agc_shader_user_data) == 0x38, "UserDataLayout");
_Static_assert(sizeof(struct agc_shader_specials) == 0x30, "SpecialRegisters");
_Static_assert(offsetof(struct agc_shader, code) == 0x10, "m_code");
_Static_assert(offsetof(struct agc_shader, specials) == 0x28, "m_specials");
_Static_assert(offsetof(struct agc_shader, header_size) == 0x40, "m_headerSize");
_Static_assert(offsetof(struct agc_shader, kind) == 0x5a, "m_type");

static agc_shader_program_fn program_hook;

void
agc_shader_set_program_hook(agc_shader_program_fn hook)
{
   program_hook = hook;
}

bool
agc_shader_program_register(uint8_t kind, uint32_t *lo)
{
   switch (kind) {
   case AGC_SHADER_CS:
      *lo = 0x20c;
      return true;
   case AGC_SHADER_PS:
      *lo = 0x008;
      return true;
   case AGC_SHADER_GS:
      *lo = 0x0c8;
      return true;
   case AGC_SHADER_HS:
      *lo = 0x148;
      return true;
   case AGC_SHADER_GS_BACK:
      *lo = 0x088;
      return true;
   case AGC_SHADER_HS_BACK:
      *lo = 0x108;
      return true;
   case AGC_SHADER_GS_FRONT:
   case AGC_SHADER_HS_FRONT:
   case AGC_SHADER_FS:
      *lo = 0;
      return true;
   default:
      return false;
   }
}

int32_t
agc_shader_rebase_program(struct agc_register *registers, uint32_t count, uint32_t lo, uint64_t base)
{
   const uint64_t limit = UINT64_C(0xffffffffffff);
   for (uint32_t at = 0; at < count; at++) {
      if (registers[at].offset != lo)
         continue;
      if (at + 1 >= count || registers[at + 1].offset != lo + 1)
         return AGC_ERROR_INVALID_SHADER_PROGRAM;
      uint64_t relative = (uint64_t)registers[at].value << 8 | (uint64_t)(registers[at + 1].value & 0xff) << 40;
      if (base > limit || relative > limit - base)
         return AGC_ERROR_INVALID_SHADER_PROGRAM;
      uint64_t address = base + relative;
      registers[at].value = (uint32_t)(address >> 8);
      registers[at + 1].value = (registers[at + 1].value & 0xffffff00u) | (uint32_t)((address >> 40) & 0xff);
      return 0;
   }
   return AGC_ERROR_INVALID_SHADER_PROGRAM;
}

static void
resolve(void *field)
{
   uintptr_t offset;
   memcpy(&offset, field, sizeof(offset));
   if (!offset)
      return;
   offset += (uintptr_t)field;
   memcpy(field, &offset, sizeof(offset));
}

int32_t
agc_shader_create(struct agc_shader **destination, struct agc_shader *header, uint64_t code)
{
   static const char function[] = "sceAgcCreateShader";
   if (!destination || !header || !code)
      return AGC_ERROR_INVALID_ARGUMENT;
   if (code & UINT64_C(0xffff0000000000ff))
      return AGC_ERROR_INVALID_ALIGNMENT;
   if (header->file_header != AGC_SHADER_MAGIC || header->version != AGC_SHADER_VERSION) {
      agc_report(function, "invalid shader header or version");
      return AGC_ERROR_INVALID_SHADER_PROGRAM;
   }
   if (header->shader_size == 0 || header->shader_size % 4 != 0) {
      agc_report(function, "invalid shader code size");
      return AGC_ERROR_INVALID_SHADER_PROGRAM;
   }
   uint32_t lo;
   if (!agc_shader_program_register(header->kind, &lo)) {
      agc_report(function, "invalid shader binary type");
      return AGC_ERROR_INVALID_SHADER_PROGRAM;
   }

   resolve(&header->cx_registers);
   resolve(&header->sh_registers);
   resolve(&header->user_data);
   resolve(&header->specials);
   resolve(&header->input_semantics);
   resolve(&header->output_semantics);
   if (header->user_data) {
      resolve(&header->user_data->direct_resource_offset);
      for (unsigned i = 0; i < 4; i++)
         resolve(&header->user_data->sharp_resource_offset[i]);
   }
   header->code = code;

   uint64_t program = code;
   if (program_hook) {
      int32_t result = program_hook(header, code, &program);
      if (result)
         return result;
   }
   if (lo) {
      if (!header->sh_registers || header->num_sh_registers == 0) {
         agc_report(function, "the shader has no program address registers");
         return AGC_ERROR_INVALID_SHADER_PROGRAM;
      }
      int32_t result = agc_shader_rebase_program(header->sh_registers, header->num_sh_registers, lo, program);
      if (result) {
         agc_report(function, "program address registers are missing or out of range");
         return result;
      }
   }
   *destination = header;
   return 0;
}

static bool
context_register(const struct agc_shader *shader, uint32_t offset, uint32_t *value)
{
   if (!shader->cx_registers)
      return false;
   for (uint32_t i = 0; i < shader->num_cx_registers; i++) {
      if (shader->cx_registers[i].offset == offset) {
         *value = shader->cx_registers[i].value;
         return true;
      }
   }
   return false;
}

int32_t
agc_shader_gs_prim_payload(uint32_t *payload, const struct agc_shader *gs)
{
   if (!gs || !payload)
      return AGC_ERROR_INVALID_ARGUMENT;
   if (gs->kind != AGC_SHADER_GS)
      return AGC_ERROR_NOT_A_GEOMETRY_SHADER;
   bool two_comp = false;
   for (uint32_t i = 0; gs->cx_registers && i < gs->num_cx_registers; i++) {
      if (gs->cx_registers[i].offset == REG_SPI_SHADER_COL_FORMAT_ALIAS && (gs->cx_registers[i].value & 0xf) == 2)
         two_comp = true;
   }
   *payload = two_comp ? 8 : 0;
   return 0;
}

static bool
gs_out(uint32_t primitive, uint32_t *out)
{
   switch (primitive) {
   case 1:
      *out = 0;
      return true;
   case 2:
   case 3:
   case 10:
   case 11:
   case 18:
      *out = 1;
      return true;
   case 7:
      *out = 3;
      return true;
   case 17:
      *out = 4;
      return true;
   case 0:
   case 4:
   case 5:
   case 6:
   case 9:
   case 12:
   case 13:
   case 19:
   case 20:
   case 21:
      *out = 2;
      return true;
   default:
      return false;
   }
}

int32_t
agc_shader_create_prim_state(struct agc_register *cx, struct agc_register *uc, const struct agc_shader *hs,
                             const struct agc_shader *gs, uint32_t primitive)
{
   static const char function[] = "sceAgcCreatePrimState";
   if (!cx && !uc)
      return 0;
   if (!gs)
      return AGC_ERROR_INVALID_ARGUMENT;
   if (gs->kind != AGC_SHADER_GS || !gs->specials || (hs && (hs->kind != AGC_SHADER_HS || !hs->specials)))
      return AGC_ERROR_INVALID_SHADER_HALVES;
   uint32_t out_type;
   if (!gs_out(primitive, &out_type)) {
      agc_report(function, "invalid primitive type");
      return AGC_ERROR_OUT_OF_RANGE;
   }
   const struct agc_shader_specials *gs_specials = gs->specials;
   const struct agc_shader_specials *hs_specials = hs ? hs->specials : NULL;
   if (gs_specials->vgt_shader_stages_en.offset != REG_VGT_SHADER_STAGES_EN ||
       (hs_specials && hs_specials->vgt_shader_stages_en.offset != REG_VGT_SHADER_STAGES_EN) ||
       gs_specials->ge_cntl.offset != REG_GE_CNTL) {
      agc_report(function, "special register at an unexpected offset");
      return AGC_ERROR_INVALID_SHADER_PROGRAM;
   }

   struct agc_register stages = gs_specials->vgt_shader_stages_en;
   if (hs_specials)
      stages.value |= hs_specials->vgt_shader_stages_en.value;

   struct agc_register out;
   if (stages.value & 0x20)
      out = gs_specials->vgt_gs_out_prim_type;
   else if (hs_specials)
      out = hs_specials->vgt_gs_out_prim_type;
   else
      out = (struct agc_register){REG_VGT_GS_OUT_PRIM_TYPE, out_type};
   if (out.offset != REG_VGT_GS_OUT_PRIM_TYPE) {
      agc_report(function, "VGT_GS_OUT_PRIM_TYPE special register at an unexpected offset");
      return AGC_ERROR_INVALID_SHADER_PROGRAM;
   }

   if (cx) {
      cx[0] = stages;
      cx[1] = out;
   }
   if (uc) {
      uc[0] = gs_specials->ge_cntl;
      uc[1] = (hs_specials ? hs_specials : gs_specials)->ge_user_vgpr_en;
      uc[2] = (struct agc_register){REG_VGT_PRIMITIVE_TYPE, primitive};
   }
   return 0;
}

int32_t
agc_shader_update_prim_state(struct agc_register *cx, struct agc_register *uc, uint32_t primitive)
{
   if (cx && (cx[0].value & (0x20 | 0x04)) == 0) {
      uint32_t out_type;
      if (!gs_out(primitive, &out_type)) {
         agc_report("sceAgcUpdatePrimState", "invalid primitive type");
         return AGC_ERROR_OUT_OF_RANGE;
      }
      cx[1].value = (cx[1].value & ~UINT32_C(7)) | out_type;
   }
   if (uc)
      uc[2].value = (uc[2].value & ~UINT32_C(0x1f)) | primitive;
   return 0;
}

static uint32_t
interpolant_f16(uint32_t p, bool mapped, uint32_t g)
{
   uint32_t v = (p << 4) & 0x03000000u;
   if (!mapped) {
      v |= 0x00180020u;
   } else {
      uint32_t c = p & g;
      v &= 0xfff7ffdfu;
      v |= (c >> 15) & 0x20;
      v ^= 0x00080020u;
      v &= ~UINT32_C(0x00100000);
      v |= (~c >> 1) & 0x00100000u;
   }
   return (v & ~UINT32_C(0x00600000)) | ((p >> 30) & 3) << 21;
}

static uint32_t
interpolant(uint32_t p, bool mapped, uint32_t g)
{
   if (((p >> 20) & 3) == 2) {
      uint32_t v = ((p << 4) & 0x03000000u) | 0x00080000u;
      if (!mapped || (p & g & 0x00200000u) == 0)
         v |= 0x20;
      v = (v & ~UINT32_C(0x00600000)) | ((p >> 30) & 3) << 21;
      v |= ((p >> 28) & 3) << 8;
      if (!mapped)
         return v;
      v |= (g >> 8) & 0x1f;
      if (p & 0x01400000u)
         v |= 0x400;
      return v;
   }

   uint32_t base;
   if (p & 0x00300000u)
      base = interpolant_f16(p, mapped, g);
   else if ((p & 0x01000000u) || !mapped)
      base = 0x20;
   else
      base = 0;

   uint32_t v = base & ~UINT32_C(0x1f);
   if (mapped) {
      uint32_t flat = (p & 0x00400000u) || (p & 0x01000000u) ? 0x400 : 0;
      v |= (g >> 8) & 0x1f;
      v = (v & ~UINT32_C(0x400)) | flat;
   } else {
      v &= ~UINT32_C(0x400);
   }
   return (v & ~UINT32_C(0x300)) | ((p >> 28) & 3) << 8;
}

int32_t
agc_shader_interpolant_mapping(struct agc_register *registers, const struct agc_shader *gs,
                               const struct agc_shader *ps, bool fill, const char *function)
{
   if (!registers)
      return AGC_ERROR_INVALID_ARGUMENT;
   if (!ps && !fill)
      return AGC_ERROR_INVALID_ARGUMENT;
   uint32_t inputs = ps ? ps->num_input_semantics : 0;
   if (inputs > 32) {
      agc_report(function, "more than 32 pixel inputs");
      return AGC_ERROR_OUT_OF_RANGE;
   }

   struct agc_register values[32];
   for (uint32_t i = 0; i < 32; i++)
      values[i] = (struct agc_register){REG_SPI_PS_INPUT_CNTL_0 + i, i};

   if (ps && inputs > 0) {
      if (!gs)
         return AGC_ERROR_INVALID_ARGUMENT;
      if (!ps->input_semantics || (gs->num_output_semantics != 0 && !gs->output_semantics)) {
         agc_report(function, "missing semantics");
         return AGC_ERROR_INVALID_SHADER_PROGRAM;
      }
      uint32_t outputs = gs->output_semantics ? gs->num_output_semantics : 0;
      for (uint32_t i = 0; i < inputs; i++) {
         uint32_t p = ps->input_semantics[i];
         bool mapped = false;
         uint32_t g = 0;
         for (uint32_t o = 0; o < outputs; o++) {
            if ((gs->output_semantics[o] & 0xff) == (p & 0xff)) {
               g = gs->output_semantics[o];
               mapped = true;
               break;
            }
         }
         values[i].value = interpolant(p, mapped, g);
      }
   }

   memcpy(registers, values, (fill ? 32 : inputs) * sizeof(values[0]));
   return 0;
}

struct oversubscription_limits {
   uint32_t vertices;
   uint32_t exports;
};

static bool
oversubscription_limits(const struct agc_shader *gs, uint32_t stages, uint32_t vertices, uint32_t exports_cap,
                        struct oversubscription_limits *out)
{
   uint32_t onchip, subgroup, vs_out, cl_out, max_out;
   if (!context_register(gs, REG_VGT_GS_ONCHIP_CNTL, &onchip) ||
       !context_register(gs, REG_GE_NGG_SUBGRP_CNTL, &subgroup) ||
       !context_register(gs, REG_SPI_VS_OUT_CONFIG, &vs_out) ||
       !context_register(gs, REG_PA_CL_VS_OUT_CNTL, &cl_out) ||
       !context_register(gs, REG_GE_MAX_OUTPUT_PER_SUBGROUP, &max_out))
      return false;
   uint32_t out_waves = ((max_out & 0x3ff) + 31) >> 5;
   if (out_waves == 0)
      return false;
   uint32_t work = ((((onchip >> 11) & 0x7ff) * (subgroup & 0x1ff)) + 31) >> 5;
   uint32_t waves = work > out_waves ? work : out_waves;
   if ((stages & 0x00400000u) == 0)
      waves >>= 1;
   if (waves == 0)
      waves = 1;
   uint32_t exports = 1 + ((cl_out >> 21) & 1) + ((cl_out >> 22) & 1) + ((cl_out >> 23) & 1);
   uint32_t export_limit = (exports_cap / exports) * 4;
   uint32_t vertex_limit = (vs_out & 0x80) ? 2048 : vertices / (((vs_out >> 2) & 0xf) + 1);
   out->vertices = waves * (vertex_limit / out_waves);
   out->exports = waves * (export_limit / out_waves);
   return true;
}

static uint32_t
min_u32(uint32_t a, uint32_t b)
{
   return a < b ? a : b;
}

int32_t
agc_shader_gs_oversubscription(struct agc_register *registers, const struct agc_shader *gs, int32_t budget,
                               float factor)
{
   static const char function[] = "sceAgcGetGsOversubscription";
   if (!gs || !registers)
      return AGC_ERROR_INVALID_ARGUMENT;
   if (budget < -1 || isnan(factor) || factor < 0.0f)
      return AGC_ERROR_OUT_OF_RANGE;
   if (gs->kind != AGC_SHADER_GS)
      return AGC_ERROR_INVALID_SHADER_HALVES;

   uint32_t alloc = 0;
   uint32_t rsrc4 = 0;
   if (budget == -1) {
      alloc = 0x7ff;
      rsrc4 = 0x007f0000u;
   } else if (budget != 0) {
      if (!gs->specials) {
         agc_report(function, "the shader has no special registers");
         return AGC_ERROR_INVALID_SHADER_PROGRAM;
      }
      uint32_t stages = gs->specials->vgt_shader_stages_en.value;
      struct oversubscription_limits base, extended;
      if (!oversubscription_limits(gs, stages, 1024, 128, &base) ||
          !oversubscription_limits(gs, stages, 2048, 382, &extended)) {
         agc_report(function, "the shader lacks a context register the limits need");
         return AGC_ERROR_INVALID_SHADER_PROGRAM;
      }
      uint32_t base_min = min_u32(base.vertices, base.exports);
      uint32_t shift = (stages & 0x00400000u) ? 5 : 6;
      uint32_t extended_limit = min_u32(min_u32(min_u32(extended.vertices, extended.exports), (uint32_t)budget >> shift), 1024);
      uint32_t head = extended_limit > base_min ? extended_limit - base_min : 0;
      float target_f = fmaf(factor, (float)head, (float)base_min);
      if (!(target_f >= 0.0f && target_f < 4294967296.0f))
         return AGC_ERROR_OUT_OF_RANGE;
      uint32_t target = (uint32_t)target_f;
      if (target > base_min) {
         if (target < base.exports) {
            uint64_t range = extended.vertices > base.vertices ? extended.vertices - base.vertices : 0;
            if (range == 0)
               range = 1;
            uint64_t numerator = target > base.vertices ? target - base.vertices : 0;
            uint64_t value = (numerator << 10) / range;
            if (value < 1)
               value = 1;
            if (value > 1024)
               value = 1024;
            alloc = (uint32_t)((value << 1) - 1) & 0x7ff;
            rsrc4 = 0x007f0000u;
         } else {
            uint64_t range = extended.exports > base.exports ? extended.exports - base.exports : 0;
            if (range == 0)
               range = 1;
            uint64_t value = (uint64_t)(target - base.exports) * 127 / range;
            if (value > 127)
               value = 127;
            alloc = 0x7ff;
            rsrc4 = (uint32_t)value << 16;
         }
      }
   }
   registers[0] = (struct agc_register){REG_GE_PC_ALLOC, alloc};
   registers[1] = (struct agc_register){REG_SPI_SHADER_PGM_RSRC4_GS, rsrc4};
   return 0;
}
