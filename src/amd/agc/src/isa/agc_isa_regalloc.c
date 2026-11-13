#include "agc_isa_regalloc.h"

#include <string.h>

#define SCRATCH_SGPR_ALIGN 4
#define NGG_SCRATCH_WAVE_OFFSET_SGPR 5
#define RSRC1_HS_LS_VGPR_COMP_CNT_SHIFT 28
#define RSRC2_GS_ES_VGPR_COMP_CNT_SHIFT 16
#define NGG_ES_FIRST_VGPR 5

static uint16_t
max_u16(uint16_t a, uint16_t b)
{
   return a > b ? a : b;
}

static uint32_t
div_ceil_u32(uint32_t value, uint32_t divisor)
{
   return value / divisor + (value % divisor ? 1 : 0);
}

static uint16_t
next_multiple_u16(uint16_t value, uint16_t multiple)
{
   uint16_t rest = (uint16_t)(value % multiple);
   return rest ? (uint16_t)(value + multiple - rest) : value;
}

static unsigned
count_ones(uint32_t value)
{
   unsigned count = 0;
   for (; value; value &= value - 1)
      count++;
   return count;
}

static unsigned
highest_live_sgpr(const struct agc_isa_metadata *meta)
{
   return agc_isa_mask128_length(agc_isa_mask128_low(meta->live_in_scalar, AGC_ISA_SGPR_LIMIT));
}

static uint16_t
agc_isa_vgpr_granule(bool wave64)
{
   return wave64 ? 4 : 8;
}

static uint32_t
agc_isa_encode_vgpr_blocks(uint16_t count, bool wave64)
{
   uint32_t granule = agc_isa_vgpr_granule(wave64);
   uint32_t wanted = count > 1 ? count : 1;
   return div_ceil_u32(wanted, granule) - 1;
}

static uint16_t
agc_isa_decode_vgpr_blocks(uint32_t rsrc1, bool wave64)
{
   return (uint16_t)(((rsrc1 & AGC_ISA_RSRC1_VGPRS_MASK) + 1) * agc_isa_vgpr_granule(wave64));
}

static uint32_t
agc_isa_default_pgm_rsrc1(enum agc_shader_stage stage)
{
   uint32_t common = (AGC_ISA_FLOAT_MODE_DEFAULT << AGC_ISA_RSRC1_FLOAT_MODE_SHIFT) | AGC_ISA_RSRC1_DX10_CLAMP;
   if (stage == AGC_SHADER_STAGE_COMPUTE)
      return common | AGC_ISA_RSRC1_MEM_ORDERED | AGC_ISA_RSRC1_WGP_MODE;
   return common | AGC_ISA_RSRC1_GRAPHICS_MEM_ORDERED;
}

static uint16_t
agc_isa_ps_input_vgprs(uint32_t ena, uint32_t addr)
{
   static const uint16_t sizes[16] = {2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
   uint32_t layout = addr | ena;
   uint16_t next = 0;
   uint16_t end = 0;
   for (unsigned bit = 0; bit < 16; bit++) {
      if (!(layout & (1u << bit)))
         continue;
      if (ena & (1u << bit))
         end = (uint16_t)(next + sizes[bit]);
      next = (uint16_t)(next + sizes[bit]);
   }
   return end;
}

static uint8_t
agc_isa_compute_thread_id_components(const struct agc_isa_metadata *meta)
{
   if (meta->live_in_vector.lo & 4)
      return 2;
   if (meta->live_in_vector.lo & 2)
      return 1;
   return 0;
}

static uint8_t
implied_tgids(const struct agc_isa_metadata *meta)
{
   unsigned highest = highest_live_sgpr(meta) & 0xff;
   unsigned above = highest > meta->user_sgpr_count ? highest - meta->user_sgpr_count : 0;
   unsigned limit = AGC_ISA_SYSTEM_SGPR_ENABLES - 1;
   return (uint8_t)(above < limit ? above : limit);
}

static void
system_sgprs_init(struct agc_isa_system_sgprs *system, const struct agc_isa_metadata *meta,
                  const struct agc_shader_options *options, const struct agc_shader_interface *iface)
{
   memset(system, 0, sizeof(*system));
   memset(system->workgroup_ids, UINT8_MAX, sizeof(system->workgroup_ids));
   bool has_rsrc2 = iface->has_guest_pgm_rsrc2;
   uint32_t rsrc2 = iface->guest_pgm_rsrc2;
   bool scratch_en = has_rsrc2 && (rsrc2 & AGC_ISA_RSRC2_SCRATCH_EN) != 0;
   if (options->stage == AGC_SHADER_STAGE_VERTEX) {
      system->has_scratch_wave_offset = scratch_en;
      system->scratch_wave_offset = scratch_en ? NGG_SCRATCH_WAVE_OFFSET_SGPR : 0;
      return;
   }
   if (options->stage == AGC_SHADER_STAGE_FRAGMENT) {
      uint8_t next = (uint8_t)(meta->user_sgpr_count + 1);
      if (has_rsrc2 && (rsrc2 & AGC_ISA_RSRC2_PS_LOAD_COLLISION_WAVEID)) {
         system->has_collision_wave = true;
         system->collision_wave = next++;
      }
      if (has_rsrc2 && (rsrc2 & AGC_ISA_RSRC2_PS_LOAD_INTRAWAVE_COLLISION)) {
         system->has_intrawave_collision = true;
         system->intrawave_collision = next++;
      }
      if (scratch_en) {
         system->has_scratch_wave_offset = true;
         system->scratch_wave_offset = next++;
      }
      return;
   }
   uint8_t user = meta->user_sgpr_count;
   uint8_t tgids = has_rsrc2 ? (uint8_t)count_ones((rsrc2 >> 7) & 7) : implied_tgids(meta);
   bool guest_tg_size = has_rsrc2 && (rsrc2 & AGC_ISA_RSRC2_TG_SIZE_EN) != 0;
   enum agc_isa_target target = agc_isa_target_from_shader_isa(options->target);
   bool gfx12 = agc_isa_target_is_gfx12(target);
   bool wave_index_tg_size = meta->uses_wave_id && !gfx12 && target != AGC_ISA_TARGET_GFX1013;
   if (guest_tg_size || wave_index_tg_size) {
      system->has_tg_size = true;
      system->tg_size = (uint8_t)(user + tgids);
   }
   system->has_host_tg_size = gfx12 ? guest_tg_size : system->has_tg_size;
   system->host_tg_size = gfx12 ? user : system->tg_size;
   if (gfx12) {
      uint8_t next = user;
      for (unsigned i = 0; i < 3; i++) {
         bool enabled = has_rsrc2 ? (rsrc2 & (AGC_ISA_RSRC2_TGID_X_EN << i)) != 0 : i < tgids;
         if (enabled)
            system->workgroup_ids[i] = next++;
      }
   }
   if (scratch_en) {
      system->has_scratch_wave_offset = true;
      system->scratch_wave_offset = (uint8_t)(user + tgids + (guest_tg_size ? 1 : 0));
      if (target == AGC_ISA_TARGET_GFX103 && wave_index_tg_size && !guest_tg_size) {
         system->has_host_scratch_wave_offset = true;
         system->host_scratch_wave_offset = system->scratch_wave_offset + 1;
      }
   }
}

static uint16_t
input_vgpr_floor(const struct agc_shader_options *options, const struct agc_shader_interface *iface,
                 uint8_t thread_ids)
{
   enum agc_isa_target target = agc_isa_target_from_shader_isa(options->target);
   switch (options->stage) {
   case AGC_SHADER_STAGE_COMPUTE:
      if (agc_isa_target_is_gfx11_plus(target))
         return 1;
      return (uint16_t)(thread_ids + 1);
   case AGC_SHADER_STAGE_FRAGMENT:
      if (iface->has_ps_input_ena && iface->has_ps_input_addr)
         return agc_isa_ps_input_vgprs(iface->ps_input_ena, iface->ps_input_addr);
      return 0;
   default:
      break;
   }
   if (iface->vertex_stage == AGC_SHADER_VERTEX_STAGE_LS_HS) {
      if (!iface->has_guest_pgm_rsrc1)
         return 6;
      return (uint16_t)(3 + ((iface->guest_pgm_rsrc1 >> RSRC1_HS_LS_VGPR_COMP_CNT_SHIFT) & 3));
   }
   if (!iface->has_guest_pgm_rsrc2)
      return 0;
   return (uint16_t)(NGG_ES_FIRST_VGPR + ((iface->guest_pgm_rsrc2 >> RSRC2_GS_ES_VGPR_COMP_CNT_SHIFT) & 3) + 1);
}

void
agc_isa_regalloc_init(struct agc_isa_regalloc *regs, const struct agc_isa_metadata *meta,
                      const struct agc_shader_options *options, const struct agc_shader_interface *iface)
{
   memset(regs, 0, sizeof(*regs));
   bool wave64 = options->wave_size == 64;
   uint16_t scratch_base = meta->vgpr_count;
   if (meta->uses_relative_indexing) {
      uint16_t guest_allocation = iface->has_guest_pgm_rsrc1
                                     ? agc_isa_decode_vgpr_blocks(iface->guest_pgm_rsrc1, wave64)
                                     : AGC_ISA_VGPR_LIMIT;
      scratch_base = max_u16(scratch_base, guest_allocation);
   }
   uint8_t float_mode = AGC_ISA_FLOAT_MODE_DEFAULT;
   if (iface->has_guest_pgm_rsrc1)
      float_mode = (uint8_t)((iface->guest_pgm_rsrc1 >> AGC_ISA_RSRC1_FLOAT_MODE_SHIFT) & 0xff);
   uint8_t thread_id_components = 0;
   if (options->stage == AGC_SHADER_STAGE_COMPUTE) {
      if (iface->has_guest_pgm_rsrc2)
         thread_id_components =
            (uint8_t)((iface->guest_pgm_rsrc2 & AGC_ISA_RSRC2_TIDIG_MASK) >> AGC_ISA_RSRC2_TIDIG_SHIFT);
      else
         thread_id_components = agc_isa_compute_thread_id_components(meta);
   }
   system_sgprs_init(&regs->system, meta, options, iface);
   uint16_t loaded;
   switch (options->stage) {
   case AGC_SHADER_STAGE_COMPUTE:
      loaded = regs->system.has_tg_size ? (uint16_t)(regs->system.tg_size + 1) : 0;
      if (regs->system.has_host_scratch_wave_offset)
         loaded = max_u16(loaded, regs->system.host_scratch_wave_offset + 1);
      break;
   case AGC_SHADER_STAGE_VERTEX:
      loaded = AGC_ISA_NGG_SYSTEM_SGPRS;
      break;
   default:
      loaded = (uint16_t)(meta->user_sgpr_count + 1);
      break;
   }
   bool has_guest_sgprs = !meta->uses_relative_sgprs;
   uint16_t guest_sgprs = max_u16(max_u16(meta->sgpr_count, meta->user_sgpr_count), loaded);
   regs->target = agc_isa_target_from_shader_isa(options->target);
   regs->wave64 = wave64;
   regs->scratch_base = scratch_base;
   regs->vgpr_high_water = meta->vgpr_count;
   regs->sgpr_high_water = meta->sgpr_count;
   regs->has_scratch_sgpr_base = has_guest_sgprs;
   regs->scratch_sgpr_base = has_guest_sgprs ? next_multiple_u16(guest_sgprs, SCRATCH_SGPR_ALIGN) : 0;
   regs->has_reserved_from = regs->has_scratch_sgpr_base;
   regs->reserved_from = regs->scratch_sgpr_base;
   regs->has_reserved_end = has_guest_sgprs;
   regs->reserved_end = has_guest_sgprs ? guest_sgprs : 0;
   regs->float_mode = float_mode;
   regs->thread_id_components = thread_id_components;
   regs->vgpr_floor = input_vgpr_floor(options, iface, thread_id_components);
}

bool
agc_isa_regalloc_tg_size_sgpr(const struct agc_isa_regalloc *regs, uint8_t *sgpr)
{
   if (regs->system.has_tg_size)
      *sgpr = regs->system.tg_size;
   return regs->system.has_tg_size;
}

bool
agc_isa_regalloc_scratch_wave_offset_sgpr(const struct agc_isa_regalloc *regs, uint8_t *sgpr)
{
   if (regs->system.has_scratch_wave_offset)
      *sgpr = regs->system.scratch_wave_offset;
   return regs->system.has_scratch_wave_offset;
}

bool
agc_isa_regalloc_collision_wave_sgpr(const struct agc_isa_regalloc *regs, uint8_t *sgpr)
{
   if (regs->system.has_collision_wave)
      *sgpr = regs->system.collision_wave;
   return regs->system.has_collision_wave;
}

bool
agc_isa_regalloc_intrawave_collision_sgpr(const struct agc_isa_regalloc *regs, uint8_t *sgpr)
{
   if (regs->system.has_intrawave_collision)
      *sgpr = regs->system.intrawave_collision;
   return regs->system.has_intrawave_collision;
}

uint8_t
agc_isa_regalloc_thread_id_components(const struct agc_isa_regalloc *regs)
{
   return regs->thread_id_components;
}

void
agc_isa_regalloc_translation_sgprs(const struct agc_isa_regalloc *regs, uint16_t *start, uint16_t *end)
{
   if (regs->has_reserved_from) {
      *start = regs->reserved_from;
      *end = AGC_ISA_SGPR_LIMIT;
   } else {
      *start = 0;
      *end = 0;
   }
}

uint16_t
agc_isa_regalloc_free_sgprs(const struct agc_isa_regalloc *regs)
{
   if (!regs->has_scratch_sgpr_base || regs->scratch_sgpr_base >= AGC_ISA_SGPR_LIMIT)
      return 0;
   return (uint16_t)(AGC_ISA_SGPR_LIMIT - regs->scratch_sgpr_base);
}

bool
agc_isa_regalloc_reserve_sgprs(struct agc_isa_regalloc *regs, uint8_t count, uint8_t *base)
{
   if (!regs->has_scratch_sgpr_base)
      return false;
   uint16_t first = regs->scratch_sgpr_base;
   uint16_t end = (uint16_t)(first + count);
   if (end > AGC_ISA_SGPR_LIMIT)
      return false;
   regs->scratch_sgpr_base = next_multiple_u16(end, SCRATCH_SGPR_ALIGN);
   if (count != 0) {
      regs->has_reserved_end = true;
      regs->reserved_end = end;
   }
   regs->sgpr_high_water = max_u16(regs->sgpr_high_water, end);
   *base = (uint8_t)first;
   return true;
}

static bool
out_of_registers(struct agc_isa_error *error, uint16_t vgprs)
{
   memset(error, 0, sizeof(*error));
   error->kind = AGC_ISA_ERROR_OUT_OF_REGISTERS;
   error->vgprs = vgprs;
   return false;
}

bool
agc_isa_regalloc_reserve_program_vgprs(struct agc_isa_regalloc *regs, uint8_t count, uint8_t *base,
                                       struct agc_isa_error *error)
{
   uint16_t first = regs->scratch_base;
   uint16_t end = (uint16_t)(first + count);
   if (end > AGC_ISA_VGPR_LIMIT)
      return out_of_registers(error, end);
   regs->scratch_base = end;
   regs->vgpr_high_water = max_u16(regs->vgpr_high_water, end);
   *base = (uint8_t)first;
   return true;
}

uint8_t
agc_isa_regalloc_initial_float_mode(const struct agc_isa_regalloc *regs)
{
   return regs->float_mode;
}

bool
agc_isa_regalloc_scratch_sgprs(struct agc_isa_regalloc *regs, uint8_t count, uint8_t *base)
{
   if (!regs->has_scratch_sgpr_base)
      return false;
   uint16_t first = regs->scratch_sgpr_base;
   uint16_t end = (uint16_t)(first + count);
   if (end > AGC_ISA_SGPR_LIMIT)
      return false;
   regs->sgpr_high_water = max_u16(regs->sgpr_high_water, end);
   *base = (uint8_t)first;
   return true;
}

bool
agc_isa_regalloc_top_sgprs(struct agc_isa_regalloc *regs, uint8_t count, uint8_t *base)
{
   if (count == 0 || count > AGC_ISA_SGPR_LIMIT)
      return false;
   uint16_t first = (uint16_t)((AGC_ISA_SGPR_LIMIT - count) & ~1u);
   if (!regs->has_reserved_end || first < regs->reserved_end)
      return false;
   regs->sgpr_high_water = max_u16(regs->sgpr_high_water, (uint16_t)(first + count));
   *base = (uint8_t)first;
   return true;
}

enum agc_isa_target
agc_isa_regalloc_target(const struct agc_isa_regalloc *regs)
{
   return regs->target;
}

uint16_t
agc_isa_regalloc_scratch_base(const struct agc_isa_regalloc *regs)
{
   return regs->scratch_base;
}

bool
agc_isa_regalloc_scratch_vgprs(struct agc_isa_regalloc *regs, uint8_t count, uint8_t *base,
                               struct agc_isa_error *error)
{
   uint16_t end = (uint16_t)(regs->scratch_base + count);
   if (end > AGC_ISA_VGPR_LIMIT)
      return out_of_registers(error, end);
   regs->vgpr_high_water = max_u16(regs->vgpr_high_water, end);
   *base = (uint8_t)regs->scratch_base;
   return true;
}

void
agc_isa_regalloc_reserve_vgprs(struct agc_isa_regalloc *regs, uint16_t count)
{
   regs->vgpr_high_water = max_u16(regs->vgpr_high_water, count);
}

static uint32_t
lds_blocks(const struct agc_isa_regalloc *regs, uint32_t blocks)
{
   if (agc_isa_target_is_gfx11_plus(regs->target))
      return blocks + (blocks & 1);
   return blocks;
}

static uint32_t
min_u32(uint32_t a, uint32_t b)
{
   return a < b ? a : b;
}

static uint32_t
config_rsrc1(const struct agc_isa_regalloc *regs, uint16_t num_vgprs, const struct agc_shader_options *options,
             const struct agc_shader_interface *iface)
{
   uint32_t value = iface->has_guest_pgm_rsrc1 ? iface->guest_pgm_rsrc1 : agc_isa_default_pgm_rsrc1(options->stage);
   value &= ~(AGC_ISA_RSRC1_VGPRS_MASK | AGC_ISA_RSRC1_SGPRS_MASK);
   if (agc_isa_target_is_gfx12(regs->target)) {
      value &= ~(AGC_ISA_RSRC1_DX10_CLAMP | AGC_ISA_RSRC1_IEEE_MODE);
      if (options->stage == AGC_SHADER_STAGE_VERTEX && iface->vertex_stage == AGC_SHADER_VERTEX_STAGE_LS_HS) {
         uint32_t field = 3u << RSRC1_HS_LS_VGPR_COMP_CNT_SHIFT;
         bool instance = (value & field) == field;
         value &= ~((1u << 24) | field);
         value |= (uint32_t)instance << RSRC1_HS_LS_VGPR_COMP_CNT_SHIFT;
      } else if (options->stage == AGC_SHADER_STAGE_VERTEX) {
         uint32_t count = (value >> 29) & 3;
         value &= ~(AGC_ISA_RSRC1_GRAPHICS_MEM_ORDERED | (3u << 29));
         value |= min_u32(count, 2) << 29;
      } else if (options->stage == AGC_SHADER_STAGE_FRAGMENT) {
         value &= ~AGC_ISA_RSRC1_GRAPHICS_MEM_ORDERED;
      }
   }
   return value | agc_isa_encode_vgpr_blocks(num_vgprs, regs->wave64);
}

static uint32_t
config_rsrc2(const struct agc_isa_regalloc *regs, const struct agc_isa_metadata *meta, uint32_t lds_bytes,
             const struct agc_shader_options *options, const struct agc_shader_interface *iface)
{
   uint32_t user = meta->user_sgpr_count;
   bool gfx11_plus = agc_isa_target_is_gfx11_plus(regs->target);
   uint32_t value = iface->has_guest_pgm_rsrc2 ? iface->guest_pgm_rsrc2 : 0;
   value &= ~(AGC_ISA_RSRC2_SCRATCH_EN | AGC_ISA_RSRC2_USER_SGPR_MASK | AGC_ISA_RSRC2_USER_SGPR_MSB);
   if (meta->uses_scratch)
      value |= AGC_ISA_RSRC2_SCRATCH_EN;
   value |= (user & 0x1f) << AGC_ISA_RSRC2_USER_SGPR_SHIFT;
   if (options->stage == AGC_SHADER_STAGE_COMPUTE) {
      if (meta->uses_wave_id && regs->target != AGC_ISA_TARGET_GFX1013 && !agc_isa_target_is_gfx12(regs->target))
         value |= AGC_ISA_RSRC2_TG_SIZE_EN;
      if (!iface->has_guest_pgm_rsrc2) {
         unsigned highest = highest_live_sgpr(meta);
         unsigned system = highest > user ? highest - user : 0;
         for (unsigned i = 0; i < system && i < AGC_ISA_SYSTEM_SGPR_ENABLES; i++)
            value |= AGC_ISA_RSRC2_TGID_X_EN << i;
         value |= (uint32_t)agc_isa_compute_thread_id_components(meta) << AGC_ISA_RSRC2_TIDIG_SHIFT;
      }
      if (agc_isa_target_is_gfx12(regs->target))
         value &= ~(7u << 7);
      if (lds_bytes > 0) {
         uint32_t blocks = lds_blocks(regs, div_ceil_u32(lds_bytes, AGC_ISA_LDS_GRANULE_BYTES));
         value &= ~AGC_ISA_RSRC2_LDS_MASK;
         value |= min_u32(blocks, 0x1ff) << AGC_ISA_RSRC2_LDS_SHIFT;
      }
   } else {
      if (user >= 32)
         value |= AGC_ISA_RSRC2_USER_SGPR_MSB;
      unsigned shift = AGC_ISA_RSRC2_PS_EXTRA_LDS_SHIFT;
      uint32_t mask = AGC_ISA_RSRC2_PS_EXTRA_LDS_MASK;
      if (options->stage == AGC_SHADER_STAGE_VERTEX) {
         if (iface->vertex_stage == AGC_SHADER_VERTEX_STAGE_LS_HS) {
            shift = AGC_ISA_RSRC2_HS_LDS_SHIFT;
            mask = AGC_ISA_RSRC2_HS_LDS_MASK;
         } else {
            shift = AGC_ISA_RSRC2_GS_LDS_SHIFT;
            mask = AGC_ISA_RSRC2_GS_LDS_MASK;
         }
      }
      uint32_t guest = (value & mask) >> shift;
      uint32_t host;
      if (options->stage == AGC_SHADER_STAGE_FRAGMENT && gfx11_plus)
         host = div_ceil_u32(guest, 2);
      else
         host = lds_blocks(regs, guest);
      value = (value & ~mask) | (min_u32(host, 0xff) << shift);
      if (options->stage == AGC_SHADER_STAGE_FRAGMENT && gfx11_plus)
         value &= ~(AGC_ISA_RSRC2_PS_LOAD_COLLISION_WAVEID | AGC_ISA_RSRC2_PS_LOAD_INTRAWAVE_COLLISION);
      if (agc_isa_target_is_gfx12(regs->target) && options->stage == AGC_SHADER_STAGE_VERTEX &&
          iface->vertex_stage == AGC_SHADER_VERTEX_STAGE_ES_GS && iface->es_input == AGC_SHADER_ES_VERTEX) {
         uint32_t field = 3u << RSRC2_GS_ES_VGPR_COMP_CNT_SHIFT;
         bool instance = (value & field) == field;
         value = (value & ~field) | (uint32_t)instance << RSRC2_GS_ES_VGPR_COMP_CNT_SHIFT;
      }
   }
   if (agc_isa_target_is_gfx12(regs->target))
      value &= ~AGC_ISA_RSRC2_TRAP_PRESENT;
   return value;
}

void
agc_isa_regalloc_finish(const struct agc_isa_regalloc *regs, const struct agc_isa_metadata *meta,
                        const struct agc_shader_options *options, const struct agc_shader_interface *iface,
                        struct agc_isa_shader_config *config)
{
   uint16_t num_vgprs = max_u16(max_u16(regs->vgpr_high_water, regs->vgpr_floor), 1);
   uint32_t lds_bytes = 0;
   if (iface->lds_bytes > 0)
      lds_bytes = iface->lds_bytes;
   else if (options->stage == AGC_SHADER_STAGE_COMPUTE && iface->has_guest_pgm_rsrc2)
      lds_bytes = ((iface->guest_pgm_rsrc2 & AGC_ISA_RSRC2_LDS_MASK) >> AGC_ISA_RSRC2_LDS_SHIFT) *
                  AGC_ISA_LDS_GRANULE_BYTES;
   config->pgm_rsrc1 = config_rsrc1(regs, num_vgprs, options, iface);
   config->pgm_rsrc2 = config_rsrc2(regs, meta, lds_bytes, options, iface);
   config->num_sgprs = max_u16(regs->sgpr_high_water, meta->user_sgpr_count);
   config->num_vgprs = num_vgprs;
   config->lds_bytes = lds_bytes;
   config->scratch_bytes_per_lane = iface->scratch_bytes_per_lane;
   config->user_sgpr_count = meta->user_sgpr_count;
}
