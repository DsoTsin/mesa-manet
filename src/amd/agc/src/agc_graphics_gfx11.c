#include "agc_graphics.h"

#include <stdlib.h>

#include "util/u_atomic.h"

#include "agc_debug.h"
#include "agc_surface.h"
#include "sid.h"

#define SW_64KB_S   9
#define SW_64KB_Z_X 24
#define CB_SLOT_REGS 15u

void agc_reg_hook_cb_blend_control(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_cb_color_attrib(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_cb_color_attrib3(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_cb_color_base(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_cb_color_base_ext(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_cb_color_control(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_cb_color_dcc_base(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_cb_color_dcc_control(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_cb_color_info(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_db_base(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_db_base_hi(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_db_render_control(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_db_shader_control(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_db_stencil_info(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_db_z_info(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_db_htile_surface(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_fsr_disabled(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_ge_cntl(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_ge_multi_prim_ib_reset_en(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_gs_instance_cnt(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_scissor_tl(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_pa_cl_ngg_cntl(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_pa_su_hardware_screen_offset(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_spi_ps_input_cntl(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_texture_gradient_factors(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_tmpring_size(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_unsupported(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_vgt_hs_offchip_param(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_vgt_primitive_type(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_vgt_gs_out_prim_type(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_vgt_shader_stages_en(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_vgt_strmout_config(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);
void agc_reg_hook_vgt_tf_ring_size(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);

static uint32_t
field(uint32_t value, unsigned lo, unsigned width)
{
   return (value >> lo) & ((UINT32_C(1) << width) - 1);
}

static uint32_t
replace(uint32_t value, unsigned lo, unsigned width, uint32_t field_value)
{
   uint32_t mask = ((UINT32_C(1) << width) - 1) << lo;
   return (value & ~mask) | ((field_value << lo) & mask);
}

static uint32_t swizzle_xor_mask(uint32_t mode);

static uint32_t
shader_engines(void)
{
   return agc_host_caps.shader_engines ? agc_host_caps.shader_engines : 1;
}

void
agc_reg_hook_db_render_control(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   const uint32_t depth_copy = UINT32_C(1) << 2;
   const uint32_t stencil_copy = UINT32_C(1) << 3;
   const uint32_t decompress_enable = UINT32_C(1) << 12;
   uint32_t guest = args->value;
   if (guest & (depth_copy | stencil_copy))
      agc_fatal(args->function, "DB_RENDER_CONTROL depth/stencil copy passes need a shader copy");
   uint32_t host = (guest & 0x0fff) | (guest & decompress_enable) | (field(guest, 13, 1) << 14);
   static int uncompressed = -1;
   if (uncompressed < 0) {
      const char *value = getenv("AGC_DEPTH_COMPRESS");
      uncompressed = value && *value == '0';
   }
   if (uncompressed)
      host |= S_028000_DEPTH_COMPRESS_DISABLE(1) | S_028000_STENCIL_COMPRESS_DISABLE(1);
   agc_reg_write(out, args->host_offset, host);
}

void
agc_reg_hook_fsr_disabled(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   (void)out;
   if (args->value != 0) {
      agc_fatal(args->function,
                "flexible scaled rasterization (PS5 FSR distortion viewports) has no gfx11 counterpart");
   }
}

void
agc_reg_hook_db_z_info(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t host = args->value | S_028040_ITERATE_256(1);
   if (field(host, 0, 2) == 0)
      host = replace(host, 2, 2, 3);
   else
      host = replace(host, 4, 5, SW_64KB_Z_X);
   agc_reg_write(out, args->host_offset, host);
}

void
agc_reg_hook_db_stencil_info(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t host = args->value | S_028044_ITERATE_256(1);
   if (field(host, 0, 1) != 0)
      host = replace(host, 4, 5, SW_64KB_Z_X);
   agc_reg_write(out, args->host_offset, host);
}

void
agc_reg_hook_db_htile_surface(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   (void)args->value;
   agc_reg_write(out, args->host_offset, S_028ABC_FULL_CACHE(1) | S_028ABC_PIPE_ALIGNED(1));
}

void
agc_reg_hook_pa_su_hardware_screen_offset(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t x = field(args->value, 0, 9);
   uint32_t y = field(args->value, 16, 9);
   agc_reg_write(out, args->host_offset, (x & ~UINT32_C(1)) | ((y & ~UINT32_C(1)) << 16));
}

void
agc_reg_hook_spi_ps_input_cntl(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t guest = args->value;
   if (field(guest, 13, 4) != 0)
      agc_fatal(args->function, "SPI_PS_INPUT_CNTL CYL_WRAP has no gfx11 field; emulate in the pixel shader");
   agc_reg_write(out, args->host_offset,
                 guest & ~(UINT32_C(0xf) << 13) & ~(UINT32_C(1) << 12) & ~(UINT32_C(1) << 11));
}

void
agc_reg_hook_tmpring_size(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t engines = shader_engines();
   uint32_t waves = field(args->value, 0, 12);
   uint32_t wave_256 = field(args->value, 12, 13) * 4;
   if (wave_256 >= UINT32_C(1) << 15)
      agc_fatal(args->function, "TMPRING_SIZE wave size does not fit the gfx11 field");
   agc_reg_write(out, args->host_offset, (waves + engines - 1) / engines | (wave_256 << 12));
}

static uint32_t
blend_factor(const struct agc_reg_hook_args *args, uint32_t guest)
{
   if (guest <= V_028780_BLEND_SRC_ALPHA_SATURATE)
      return guest;
   if (guest == V_028780_BLEND_BOTH_SRC_ALPHA)
      return V_028780_BLEND_SRC_ALPHA;
   if (guest == V_028780_BLEND_BOTH_INV_SRC_ALPHA)
      return V_028780_BLEND_ONE_MINUS_SRC_ALPHA;
   if (guest <= V_028780_BLEND_ONE_MINUS_CONSTANT_ALPHA_GFX6)
      return guest - 2;
   agc_fatal(args->function, "blend factor is out of range");
}

void
agc_reg_hook_cb_blend_control(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   static const uint8_t pairs[2][2] = {{0, 8}, {16, 24}};
   uint32_t guest = args->value;
   uint32_t host = guest;
   for (unsigned i = 0; i < 2; i++) {
      uint32_t source = field(guest, pairs[i][0], 5);
      uint32_t destination = field(guest, pairs[i][1], 5);
      if (source == V_028780_BLEND_BOTH_SRC_ALPHA)
         destination = V_028780_BLEND_ONE_MINUS_SRC_ALPHA;
      else if (source == V_028780_BLEND_BOTH_INV_SRC_ALPHA)
         destination = V_028780_BLEND_SRC_ALPHA;
      host = replace(host, pairs[i][0], 5, blend_factor(args, source));
      host = replace(host, pairs[i][1], 5, blend_factor(args, destination));
   }
   agc_reg_write(out, args->host_offset, host);
}

void
agc_reg_hook_cb_color_control(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t guest = args->value;
   uint32_t mode = field(guest, 4, 3);
   switch (mode) {
   case V_028808_CB_DISABLE:
   case V_028808_CB_NORMAL:
      break;
   case V_028808_CB_ELIMINATE_FAST_CLEAR:
   case V_028808_CB_DECOMPRESS:
   case V_028808_CB_FMASK_DECOMPRESS:
   case V_028808_CB_DCC_DECOMPRESS_GFX8: {
      static uint32_t reported[8];
      if (!p_atomic_xchg(&reported[mode], 1))
         agc_stat(args->function, "CB_COLOR_CONTROL mode %u (a fast clear eliminate or decompress pass) draws nothing on "
                                  "gfx11: DCC stays compressed and is read through its metadata",
                  mode);
      mode = V_028808_CB_DISABLE;
      break;
   }
   case V_028808_CB_RESOLVE:
      agc_fatal(args->function, "CB_COLOR_CONTROL RESOLVE has no gfx11 mode; resolve in a shader");
   default:
      agc_fatal(args->function, "CB_COLOR_CONTROL reserved mode");
   }
   agc_reg_write(out, args->host_offset, replace(guest & 0x00ff0009, 4, 3, mode));
}

void
agc_reg_hook_db_shader_control(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   const uint32_t coverage_to_mask_enable = UINT32_C(1) << 7;
   const uint32_t primitive_ordered_pixel_shader = UINT32_C(1) << 16;
   const uint32_t exec_if_overlapped = UINT32_C(1) << 17;
   const uint32_t override_intrinsic_rate_enable = UINT32_C(1) << 25;
   uint32_t guest = args->value;
   uint32_t host = guest & ~exec_if_overlapped & ~(UINT32_C(7) << 20) & ~coverage_to_mask_enable;
   if (guest & primitive_ordered_pixel_shader)
      host |= override_intrinsic_rate_enable | (field(guest, 20, 3) << 26);
   agc_reg_write(out, args->host_offset, host);
}

void
agc_reg_hook_vgt_shader_stages_en(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   const uint32_t hs_en = UINT32_C(1) << 2;
   const uint32_t primgen_en = UINT32_C(1) << 13;
   const uint32_t vs_w32_en = UINT32_C(1) << 23;
   const uint32_t keep = 0x07fff1ff;
   uint32_t guest = args->value;
   if (guest == 0)
      return;
   if (!(guest & primgen_en)) {
      agc_fatal(args->function,
                "VGT_SHADER_STAGES_EN without PRIMGEN_EN: legacy vertex routing has no gfx11 path");
   }
   uint32_t host = guest & keep & ~vs_w32_en;
   host = replace(host, 3, 2, guest & hs_en ? V_028B54_ES_STAGE_DS : V_028B54_ES_STAGE_REAL);
   agc_reg_write(out, args->host_offset, host);
}

void
agc_reg_hook_vgt_tf_ring_size(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t size = field(args->value, 0, 17) / shader_engines();
   agc_reg_write(out, args->host_offset, size ? size : 1);
}

void
agc_reg_hook_vgt_hs_offchip_param(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t blocks = (field(args->value, 0, 9) + 1) / shader_engines();
   agc_reg_write(out, args->host_offset, ((blocks ? blocks : 1) - 1) | (field(args->value, 9, 2) << 10));
}

static bool
output_primitive(uint32_t primitive, uint32_t *output)
{
   switch (primitive) {
   case V_030908_DI_PT_POINTLIST:
      *output = V_030998_POINTLIST;
      return true;
   case V_030908_DI_PT_LINELIST:
   case V_030908_DI_PT_LINESTRIP:
   case V_030908_DI_PT_LINELIST_ADJ:
   case V_030908_DI_PT_LINESTRIP_ADJ:
   case V_030908_DI_PT_LINELOOP:
      *output = V_030998_LINESTRIP;
      return true;
   case V_030908_DI_PT_2D_RECTANGLE:
      *output = V_030998_RECT_2D;
      return true;
   case V_030908_DI_PT_RECTLIST:
      *output = V_030998_RECTLIST;
      return true;
   case V_030908_DI_PT_NONE:
   case V_030908_DI_PT_TRILIST:
   case V_030908_DI_PT_TRIFAN:
   case V_030908_DI_PT_TRISTRIP:
   case V_030908_DI_PT_PATCH:
   case V_030908_DI_PT_TRILIST_ADJ:
   case V_030908_DI_PT_TRISTRIP_ADJ:
   case V_030908_DI_PT_QUADLIST:
   case V_030908_DI_PT_QUADSTRIP:
   case V_030908_DI_PT_POLYGON:
      *output = V_030998_TRISTRIP;
      return true;
   default:
      return false;
   }
}

#define STAGES_HS_EN (UINT32_C(1) << 2)
#define STAGES_GS_EN (UINT32_C(1) << 5)

void
agc_reg_hook_vgt_primitive_type(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t primitive = field(args->value, 0, 6);
   uint32_t host = args->value;
   uint32_t stages = 0;
   bool geometry = agc_reg_context(args, (R_028B54_VGT_SHADER_STAGES_EN - SI_CONTEXT_REG_OFFSET) / 4, &stages) &&
                   (stages & (STAGES_GS_EN | STAGES_HS_EN));
   uint32_t output = 0;
   if (!geometry && !output_primitive(primitive, &output)) {
      agc_fatal(args->function,
                "VGT_PRIMITIVE_TYPE %u has no output primitive for VGT_GS_OUT_PRIM_TYPE; "
                "sceAgcCreatePrimState rejects it too",
                primitive);
   }
   if (primitive == V_030908_DI_PT_2D_RECTANGLE)
      host = (host & ~UINT32_C(0x3f)) | V_030908_DI_PT_RECTLIST;
   agc_reg_write(out, args->host_offset, host);
   if (!geometry)
      agc_reg_write(out, (R_030998_VGT_GS_OUT_PRIM_TYPE - CIK_UCONFIG_REG_OFFSET) / 4, output);
}

void
agc_reg_hook_vgt_gs_out_prim_type(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t stages = 0;
   agc_reg_state(args, (R_028B54_VGT_SHADER_STAGES_EN - SI_CONTEXT_REG_OFFSET) / 4, &stages);
   if (stages & (STAGES_GS_EN | STAGES_HS_EN))
      agc_reg_write_uc(out, (R_030998_VGT_GS_OUT_PRIM_TYPE - CIK_UCONFIG_REG_OFFSET) / 4, field(args->value, 0, 6));
}

void
agc_reg_hook_ge_multi_prim_ib_reset_en(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   agc_reg_write(out, args->host_offset, (args->value & 3) | 4);
}

static uint32_t
cx_offset(uint32_t reg)
{
   return (reg - SI_CONTEXT_REG_OFFSET) / 4;
}

static uint32_t
host_fdcc_control(const struct agc_reg_hook_args *args, uint32_t guest, bool dcc, uint32_t samples_log2)
{
   const uint32_t kept = 0x3fc | S_028C78_DISABLE_CONSTANT_ENCODE_REG(1) | (UINT32_C(1) << 19);
   uint32_t host = (guest & kept) | (field(guest, 20, 1) << 10);
   if (!dcc)
      return host;
   host &= ~(UINT32_C(1) << 19);
   host |= S_028C78_DISABLE_CONSTANT_ENCODE_REG(1) | S_028C78_FDCC_ENABLE(1);
   if (args->host == AGC_HOST_GFX115)
      host |= S_028C78_ENABLE_MAX_COMP_FRAG_OVERRIDE(1) | S_028C78_MAX_COMP_FRAGS(samples_log2 >= 2);
   return host;
}

static void
cb_write_fdcc_control(const struct agc_reg_hook_args *args, uint32_t slot, uint32_t info, struct agc_reg_writes *out)
{
   const uint32_t control_offset = cx_offset(R_028C78_CB_COLOR0_FDCC_CONTROL) + slot * CB_SLOT_REGS;
   const bool dcc = field(info, 28, 1) && agc_surface_native_dcc();
   uint32_t control = 0;
   uint32_t attrib = 0;
   if (!agc_reg_state(args, cx_offset(R_028C78_CB_COLOR0_DCC_CONTROL) + slot * CB_SLOT_REGS, &control)) {
      if (dcc) {
         agc_fatal(args->function,
                   "CB_COLOR%u_INFO (cx:0x%03x) enables DCC, which gfx11 enables in CB_COLOR%u_FDCC_CONTROL: "
                   "neither this register array nor the context state before it sets CB_COLOR%u_DCC_CONTROL",
                   slot, args->offset, slot, slot);
      }
      agc_reg_write(out, control_offset, 0);
      return;
   }
   if (dcc && !agc_reg_state(args, cx_offset(R_028C74_CB_COLOR0_ATTRIB) + slot * CB_SLOT_REGS, &attrib)) {
      agc_fatal(args->function,
                "CB_COLOR%u_INFO (cx:0x%03x) enables DCC: neither this register array nor the context state before "
                "it sets CB_COLOR%u_ATTRIB, whose sample count decides MAX_COMP_FRAGS",
                slot, args->offset, slot);
   }
   const uint32_t host = host_fdcc_control(args, control, dcc, field(attrib, 12, 3));
   agc_reg_write(out, control_offset, host);
   uint32_t base = 0;
   uint32_t base_ext = 0;
   uint32_t attrib3 = 0;
   if (dcc && agc_reg_sibling(args, cx_offset(R_028C60_CB_COLOR0_BASE) + slot * CB_SLOT_REGS, &base) &&
       agc_reg_sibling(args, cx_offset(R_028E40_CB_COLOR0_BASE_EXT) + slot, &base_ext) &&
       agc_reg_sibling(args, cx_offset(R_028EE0_CB_COLOR0_ATTRIB3) + slot, &attrib3)) {
      const uint32_t xor_mask = swizzle_xor_mask(field(attrib3, 14, 5));
      const uint64_t low = base & ~xor_mask;
      uint32_t meta = 0;
      uint32_t meta_ext = 0;
      uint64_t meta_va = 0;
      if (agc_reg_sibling(args, cx_offset(R_028C94_CB_COLOR0_DCC_BASE) + slot * CB_SLOT_REGS, &meta) &&
          agc_reg_sibling(args, cx_offset(R_028EA0_CB_COLOR0_DCC_BASE_EXT) + slot, &meta_ext))
         meta_va = ((meta & ~(xor_mask & 0xf)) | (uint64_t)(meta_ext & 0xff) << 32) << 8;
      agc_surface_note_dcc_target((low | (uint64_t)(base_ext & 0xff) << 32) << 8, meta_va, host);
   }
}

void
agc_reg_hook_cb_color_info(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t guest = args->value;
   uint32_t format = field(guest, 2, 5);
   if (format == 31)
      agc_fatal(args->function, "CB_COLOR_INFO format 2_10_10_10_6E4 has no gfx11 encoding");
   uint32_t host = format | (field(guest, 7, 1) << 7) | (field(guest, 8, 3) << 8) | (field(guest, 11, 2) << 11) |
                   (field(guest, 15, 4) << 15) | (field(guest, 20, 6) << 20);
   agc_reg_write(out, args->host_offset, host);
   cb_write_fdcc_control(args, (args->offset - cx_offset(R_028C70_CB_COLOR0_INFO)) / CB_SLOT_REGS, guest, out);
}

void
agc_reg_hook_cb_color_dcc_control(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t slot = (args->offset - cx_offset(R_028C78_CB_COLOR0_DCC_CONTROL)) / CB_SLOT_REGS;
   uint32_t info;
   if (agc_reg_sibling(args, cx_offset(R_028C70_CB_COLOR0_INFO) + slot * CB_SLOT_REGS, &info))
      return;
   if (agc_reg_state(args, cx_offset(R_028C70_CB_COLOR0_INFO) + slot * CB_SLOT_REGS, &info)) {
      cb_write_fdcc_control(args, slot, info, out);
      return;
   }
   if (agc_surface_native_dcc()) {
      agc_fatal(args->function,
                "CB_COLOR%u_DCC_CONTROL (cx:0x%03x) needs CB_COLOR%u_INFO, which neither this register array nor "
                "the context state before it sets: the guest enables DCC there and gfx11 in FDCC_CONTROL",
                slot, args->offset, slot);
   }
   agc_reg_write(out, args->host_offset, host_fdcc_control(args, args->value, false, 0));
}

void
agc_reg_hook_cb_color_attrib(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t guest = args->value;
   if (field(guest, 12, 3) > field(guest, 15, 2)) {
      agc_fatal(args->function,
                "CB_COLOR_ATTRIB has more samples than fragments; gfx11 has no FMASK for the extra coverage");
   }
   agc_reg_write(out, args->host_offset,
                 field(guest, 15, 2) | (field(guest, 17, 1) << 2) | (field(guest, 19, 1) << 4));
   uint32_t slot = (args->offset - cx_offset(R_028C74_CB_COLOR0_ATTRIB)) / CB_SLOT_REGS;
   uint32_t info_offset = cx_offset(R_028C70_CB_COLOR0_INFO) + slot * CB_SLOT_REGS;
   uint32_t info;
   if (agc_surface_native_dcc() && !agc_reg_sibling(args, info_offset, &info) &&
       agc_reg_state(args, info_offset, &info) && field(info, 28, 1))
      cb_write_fdcc_control(args, slot, info, out);
}

static uint32_t
host_swizzle_mode(uint32_t mode, bool volume)
{
   if (mode == 0)
      return 0;
   if (volume) {
      switch (mode) {
      case 5:
      case 9:
      case 17:
      case 21:
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 31:
         return mode;
      default:
         return SW_64KB_S;
      }
   }
   switch (mode) {
   case 1:
   case 5:
   case 9:
   case 17:
   case 21:
   case 25:
      return mode + 1;
   default:
      return mode;
   }
}

void
agc_reg_hook_cb_color_attrib3(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   const uint32_t keep = 0x1fff | (UINT32_C(1) << 13) | (UINT32_C(3) << 24) | (UINT32_C(1) << 30);
   uint32_t guest = args->value;
   uint32_t mode = host_swizzle_mode(field(guest, 14, 5), field(guest, 24, 2) == 2);
   agc_reg_write(out, args->host_offset, replace(guest & keep, 14, 5, mode));
}

static uint32_t
swizzle_xor_mask(uint32_t mode)
{
   switch (mode) {
   case 21:
   case 22:
      return 0x0f;
   case 17:
   case 18:
   case 24:
   case 25:
   case 26:
   case 27:
      return 0xff;
   default:
      return 0;
   }
}

static uint32_t
mirrored_base_hi(uint32_t mode, uint32_t low, uint32_t high)
{
   if (mode == 0)
      return high;
   if (high == 0 && (low & ~swizzle_xor_mask(mode)) == 0)
      return 0;
   return (high + 1) & 0xff;
}

void
agc_reg_hook_cb_color_base(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t slot = (args->offset - (R_028C60_CB_COLOR0_BASE - SI_CONTEXT_REG_OFFSET) / 4) / 15;
   uint32_t attrib3;
   if ((args->value & 0xff) == 0) {
      agc_reg_write(out, args->host_offset, args->value);
      return;
   }
   if (!agc_reg_sibling(args, (R_028EE0_CB_COLOR0_ATTRIB3 - SI_CONTEXT_REG_OFFSET) / 4 + slot, &attrib3)) {
      agc_fatal(args->function,
                "CB_COLOR%u_BASE (cx:0x%03x) needs CB_COLOR%u_ATTRIB3 in the same register array: "
                "its swizzle mode decides the host base, and the companion is missing",
                slot, args->offset, slot);
   }
   agc_reg_write(out, args->host_offset, args->value & ~swizzle_xor_mask(field(attrib3, 14, 5)));
}

void
agc_reg_hook_cb_color_base_ext(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t slot = args->offset - (R_028E40_CB_COLOR0_BASE_EXT - SI_CONTEXT_REG_OFFSET) / 4;
   uint32_t high = args->value & 0xff;
   uint32_t attrib3 = 0;
   uint32_t low = 0;
   bool known = agc_reg_sibling(args, (R_028EE0_CB_COLOR0_ATTRIB3 - SI_CONTEXT_REG_OFFSET) / 4 + slot, &attrib3);
   uint32_t mode = field(attrib3, 14, 5);
   if (known && mode == 0) {
      agc_reg_write(out, args->host_offset, high);
      return;
   }
   if (high == 0) {
      if (!agc_reg_sibling(args, (R_028C60_CB_COLOR0_BASE - SI_CONTEXT_REG_OFFSET) / 4 + slot * 15, &low)) {
         agc_fatal(args->function,
                   "CB_COLOR%u_BASE_EXT (cx:0x%03x) needs CB_COLOR%u_BASE in the same register array: "
                   "a null base stays null, and the companion is missing",
                   slot, args->offset, slot);
      }
      if (low == 0) {
         agc_reg_write(out, args->host_offset, 0);
         return;
      }
   }
   if (!known) {
      agc_fatal(args->function,
                "CB_COLOR%u_BASE_EXT (cx:0x%03x) needs CB_COLOR%u_ATTRIB3 in the same register array: "
                "its swizzle mode decides the host base, and the companion is missing",
                slot, args->offset, slot);
   }
   agc_reg_write(out, args->host_offset, mirrored_base_hi(mode, low, high));
}

void
agc_reg_hook_cb_color_dcc_base(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t slot = (args->offset - cx_offset(R_028C94_CB_COLOR0_DCC_BASE)) / CB_SLOT_REGS;
   uint32_t attrib3;
   if ((args->value & 0xf) == 0) {
      agc_reg_write(out, args->host_offset, args->value);
      return;
   }
   if (!agc_reg_sibling(args, cx_offset(R_028EE0_CB_COLOR0_ATTRIB3) + slot, &attrib3)) {
      agc_fatal(args->function,
                "CB_COLOR%u_DCC_BASE (cx:0x%03x) needs CB_COLOR%u_ATTRIB3 in the same register array: its swizzle "
                "mode decides which address bits hold the surface's XOR",
                slot, args->offset, slot);
   }
   agc_reg_write(out, args->host_offset, args->value & ~(swizzle_xor_mask(field(attrib3, 14, 5)) & 0xf));
}

static const char *const db_plane_names[4] = {"DB_Z_READ_BASE", "DB_STENCIL_READ_BASE", "DB_Z_WRITE_BASE",
                                              "DB_STENCIL_WRITE_BASE"};

static bool
db_plane_mode(const struct agc_reg_hook_args *args, uint32_t plane, uint32_t *mode)
{
   uint32_t info_offset = (plane & 1 ? R_028044_DB_STENCIL_INFO : R_028040_DB_Z_INFO) - SI_CONTEXT_REG_OFFSET;
   uint32_t info = 0;
   bool known = agc_reg_sibling(args, info_offset / 4, &info);
   *mode = field(info, 4, 5);
   return known;
}

static void
db_plane_mode_missing(const struct agc_reg_hook_args *args, uint32_t plane, const char *suffix)
{
   agc_fatal(args->function,
             "%s%s (cx:0x%03x) needs %s in the same register array: "
             "its swizzle mode decides the host base, and the companion is missing",
             db_plane_names[plane], suffix, args->offset, plane & 1 ? "DB_STENCIL_INFO" : "DB_Z_INFO");
}

void
agc_reg_hook_db_base(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t plane = args->offset - (R_028048_DB_Z_READ_BASE - SI_CONTEXT_REG_OFFSET) / 4;
   uint32_t mode;
   if ((args->value & 0xff) == 0) {
      agc_reg_write(out, args->host_offset, args->value);
      return;
   }
   if (!db_plane_mode(args, plane, &mode))
      db_plane_mode_missing(args, plane, "");
   agc_reg_write(out, args->host_offset, args->value & ~swizzle_xor_mask(mode));
}

void
agc_reg_hook_db_base_hi(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t plane = args->offset - (R_028068_DB_Z_READ_BASE_HI - SI_CONTEXT_REG_OFFSET) / 4;
   uint32_t high = args->value & 0xff;
   uint32_t mode;
   uint32_t low = 0;
   bool known = db_plane_mode(args, plane, &mode);
   if (known && mode == 0) {
      agc_reg_write(out, args->host_offset, high);
      return;
   }
   if (high == 0) {
      if (!agc_reg_sibling(args, (R_028048_DB_Z_READ_BASE - SI_CONTEXT_REG_OFFSET) / 4 + plane, &low)) {
         agc_fatal(args->function,
                   "%s_HI (cx:0x%03x) needs %s in the same register array: "
                   "a null base stays null, and the companion is missing",
                   db_plane_names[plane], args->offset, db_plane_names[plane]);
      }
      if (low == 0) {
         agc_reg_write(out, args->host_offset, 0);
         return;
      }
   }
   if (!known)
      db_plane_mode_missing(args, plane, "_HI");
   agc_reg_write(out, args->host_offset, mirrored_base_hi(mode, low, high));
}

#define GE_CNTL_FALLBACK_SUBGROUP 128u

void
agc_reg_hook_ge_cntl(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   const uint32_t break_wave_at_eoi = UINT32_C(1) << 18;
   const uint32_t packet_to_one_pa = UINT32_C(1) << 19;
   const uint32_t break_primgrp_at_eoi = UINT32_C(1) << 20;
   bool gfx12 = args->host == AGC_HOST_GFX12;
   uint32_t prim_grp_size = gfx12 ? 256 : 252;
   uint32_t guest = args->value;
   uint32_t prims = field(guest, 0, 9);
   uint32_t verts = field(guest, 9, 9);
   if (!prims && !verts) {
      prims = GE_CNTL_FALLBACK_SUBGROUP;
      verts = GE_CNTL_FALLBACK_SUBGROUP;
   }
   uint32_t host = prims | (verts << 9) | (guest & packet_to_one_pa) | (prim_grp_size << 21);
   if (guest & break_wave_at_eoi)
      host |= break_primgrp_at_eoi;
   if (gfx12)
      host |= S_03096C_DIS_PG_SIZE_ADJUST_FOR_STRIP(1);
   agc_reg_write(out, args->host_offset, host);
}

void
agc_reg_hook_scissor_tl(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   agc_reg_write(out, args->host_offset, field(args->value, 0, 15) | (field(args->value, 16, 15) << 16));
}

void
agc_reg_hook_gs_instance_cnt(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   uint32_t count = field(args->value, 2, 7);
   if (count > 63)
      agc_fatal(args->function, "VGT_GS_INSTANCE_CNT CNT %u does not fit the gfx12 field", count);
   agc_reg_write(out, args->host_offset, args->value);
}

void
agc_reg_hook_pa_cl_ngg_cntl(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   agc_reg_write(out, args->host_offset, (args->value & 3) | (UINT32_C(30) << 2));
}

void
agc_reg_hook_vgt_strmout_config(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   (void)out;
   if (args->value & 0x7f)
      agc_fatal(args->function, "streamout (VGT_STRMOUT_CONFIG) is not supported");
}

void
agc_reg_hook_texture_gradient_factors(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   (void)out;
   if (args->value != 0x40000040)
      agc_fatal(args->function, "texture gradient factors other than identity have no register on this host");
}

void
agc_reg_hook_unsupported(const struct agc_reg_hook_args *args, struct agc_reg_writes *out)
{
   (void)out;
   if (args->value != 0) {
      agc_fatal(args->function, "register %s:0x%03x = 0x%08x asks for a function this host does not have",
                agc_reg_bank_name(args->bank), args->offset, args->value);
   }
}
