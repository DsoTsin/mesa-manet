#include <string.h>

#include "util/u_atomic.h"

#include "agc_debug.h"
#include "agc_driver.h"
#include "agc_exports.h"
#include "agc_present.h"
#include "agc_program.h"
#include "agc_register_defaults.h"
#include "agc_shader.h"
#include "agc_shader_fusion.h"

#define REG_VGT_GS_ONCHIP_CNTL 0x291u
#define GE_CNTL_GROUP_FIELDS UINT32_C(0x3ffff)
#define GE_CNTL_GROUP_MASK UINT32_C(0x1ff)
#define DEFAULTS_VERSION_LIMIT 14u
#define SEMAPHORE_ALIGNMENT UINT64_C(0x4000)
#define SEMAPHORE_LIMIT UINT64_C(0x40000000)
#define SEMAPHORE_STRIDE UINT64_C(32)
#define ERROR_SEMAPHORES_SET ((int32_t)0x8a6c0048u)
#define ERROR_SEMAPHORES_UNSET ((int32_t)0x8a6c0049u)

static uint32_t instrumentation;
static uint64_t semaphore_bytes;
static uint64_t semaphore_memory;

static void
install(void)
{
   agc_program_install();
}

int32_t AGC_SYSV
sceAgcCreateShader(struct agc_shader **destination, struct agc_shader *header, uint64_t code)
{
   install();
   return agc_shader_create(destination, header, code);
}

int32_t AGC_SYSV
sceAgcGetGsPrimPayload(uint32_t *payload, const struct agc_shader *gs)
{
   return agc_shader_gs_prim_payload(payload, gs);
}

static void
fold_onchip(struct agc_register *uc, const struct agc_shader *gs)
{
   uint32_t onchip = 0;
   for (uint32_t i = 0; gs->cx_registers && i < gs->num_cx_registers; i++) {
      if (gs->cx_registers[i].offset == REG_VGT_GS_ONCHIP_CNTL)
         onchip = gs->cx_registers[i].value;
   }
   if (!onchip)
      return;
   uint32_t prims = (onchip >> 11) & 0x7ff;
   uint32_t verts = onchip & 0x7ff;
   uc[0].value = (uc[0].value & ~GE_CNTL_GROUP_FIELDS) | (prims & GE_CNTL_GROUP_MASK) |
                 (verts & GE_CNTL_GROUP_MASK) << 9;
}

int32_t AGC_SYSV
sceAgcCreatePrimState(struct agc_register *cx, struct agc_register *uc, const struct agc_shader *hs,
                      const struct agc_shader *gs, uint32_t primitive)
{
   int32_t result = agc_shader_create_prim_state(cx, uc, hs, gs, primitive);
   if (!result && uc && gs)
      fold_onchip(uc, gs);
   return result;
}

int32_t AGC_SYSV
sceAgcUpdatePrimState(struct agc_register *cx, struct agc_register *uc, uint32_t primitive)
{
   return agc_shader_update_prim_state(cx, uc, primitive);
}

int32_t AGC_SYSV
sceAgcCreateInterpolantMapping_10000(struct agc_register *registers, const struct agc_shader *gs,
                                     const struct agc_shader *ps)
{
   return agc_shader_interpolant_mapping(registers, gs, ps, true, "sceAgcCreateInterpolantMapping");
}

int32_t AGC_SYSV
sceAgcUpdateInterpolantMapping_10000(struct agc_register *registers, const struct agc_shader *gs,
                                     const struct agc_shader *ps)
{
   return agc_shader_interpolant_mapping(registers, gs, ps, false, "sceAgcUpdateInterpolantMapping");
}

int32_t AGC_SYSV
sceAgcGetGsOversubscription(struct agc_register *registers, const struct agc_shader *gs, int32_t budget,
                            float factor)
{
   return agc_shader_gs_oversubscription(registers, gs, budget, factor);
}

const void *AGC_SYSV
sceAgcGetRegisterDefaults2(uint32_t version)
{
   return agc_register_defaults_get(version, false);
}

const void *AGC_SYSV
sceAgcGetRegisterDefaults2Internal(uint32_t version)
{
   return agc_register_defaults_get(version, true);
}

int32_t AGC_SYSV
sceAgcGetDefaultCxStateFlat(uint32_t *destination, uint32_t first, uint32_t end, uint32_t state)
{
   (void)state;
   return agc_default_cx_state_flat(agc_register_defaults_init_version(), destination, first, end);
}

int32_t AGC_SYSV
sceAgcInit_0090(uint32_t *state, uint32_t version)
{
   static const char function[] = "sceAgcInit_0090";
   if (!state || (uintptr_t)state % 4 != 0) {
      agc_report(function, "null or misaligned state");
      return AGC_ERROR_INVALID_ARGUMENT;
   }
   agc_register_defaults_init(version);
   if (version < DEFAULTS_VERSION_LIMIT) {
      agc_register_defaults_get(version, false);
      agc_register_defaults_get(version, true);
   }
   install();
   agc_present_install();
   char error[256];
   if (!agc_driver_start(error, sizeof(error)))
      agc_report(function, "%s", error);
   return 0;
}

int32_t AGC_SYSV
sceAgcGetIsTrinityMode(bool *trinity)
{
   if (!trinity) {
      agc_report("sceAgcGetIsTrinityMode", "null output");
      return AGC_ERROR_INVALID_ARGUMENT;
   }
   *trinity = false;
   return 0;
}

int32_t AGC_SYSV
sceAgcSetShaderInstrumentation(uint32_t flags)
{
   p_atomic_set(&instrumentation, flags);
   return 0;
}

uint32_t AGC_SYSV
sceAgcGetShaderInstrumentation(void)
{
   return p_atomic_read(&instrumentation);
}

int32_t AGC_SYSV
sceAgcDebugRaiseException(uint32_t id, uint32_t detail)
{
   (void)detail;
   agc_report("sceAgcDebugRaiseException", "exception 0x%08x", id);
   return 0;
}

int32_t AGC_SYSV
sceAgcSuspendPoint(void)
{
   return 0;
}

static int32_t
register_semaphores(uint8_t *memory, uint64_t bytes, bool limited, const char *function)
{
   if (p_atomic_read(&semaphore_bytes))
      return ERROR_SEMAPHORES_SET;
   if (bytes % SEMAPHORE_ALIGNMENT || (uintptr_t)memory % SEMAPHORE_ALIGNMENT)
      return AGC_ERROR_INVALID_ALIGNMENT;
   if (limited && bytes > SEMAPHORE_LIMIT)
      return AGC_ERROR_OUT_OF_RANGE;
   if (!memory || !bytes || (uintptr_t)memory + bytes < (uintptr_t)memory) {
      agc_report(function, "null or wrapping semaphore memory");
      return AGC_ERROR_INVALID_ARGUMENT;
   }
   if (p_atomic_cmpxchg(&semaphore_bytes, 0, bytes) != 0)
      return ERROR_SEMAPHORES_SET;
   memset(memory, 0, bytes);
   p_atomic_set(&semaphore_memory, (uint64_t)(uintptr_t)memory);
   return 0;
}

int32_t AGC_SYSV
sceAgcSetAmmSemaphoreMemory(uint8_t *memory, uint64_t bytes)
{
   return register_semaphores(memory, bytes, false, "sceAgcSetAmmSemaphoreMemory");
}

int32_t AGC_SYSV
sceAgcSetSemaphoreMemory(uint8_t *memory, uint64_t bytes)
{
   return register_semaphores(memory, bytes, true, "sceAgcSetSemaphoreMemory");
}

int32_t AGC_SYSV
sceAgcGetSemaphoreLabel(uint32_t index, uint8_t **label)
{
   uint64_t bytes = p_atomic_read(&semaphore_bytes);
   uint64_t memory = p_atomic_read(&semaphore_memory);
   if (!bytes || !memory)
      return ERROR_SEMAPHORES_UNSET;
   if ((uint64_t)index >= bytes / SEMAPHORE_STRIDE)
      return AGC_ERROR_OUT_OF_RANGE;
   if (!label) {
      agc_report("sceAgcGetSemaphoreLabel", "null label output");
      return AGC_ERROR_INVALID_ARGUMENT;
   }
   *label = (uint8_t *)(uintptr_t)(memory + (uint64_t)index * SEMAPHORE_STRIDE);
   return 0;
}

int32_t AGC_SYSV
sceAgcFuseShaderHalves_0200(struct agc_shader *result, const struct agc_shader *front, const struct agc_shader *back,
                            struct agc_register *scratch)
{
   install();
   return agc_shader_fuse(result, front, back, scratch);
}

int32_t AGC_SYSV
sceAgcGetFusedShaderSize_0080(struct agc_size_align *destination, const struct agc_shader *front,
                              const struct agc_shader *back)
{
   return agc_shader_fused_size(destination, front, back);
}
