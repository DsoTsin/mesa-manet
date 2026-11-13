#include "agc_present.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "ac_descriptors.h"
#include "ac_gpu_info.h"
#include "ac_surface.h"
#include "sid.h"
#include "util/format/u_formats.h"
#include "util/u_math.h"

#include "agc_memory.h"
#include "agc_surface.h"

static bool AGC_PRINTF(3, 4)
agc_present_fail(char *error, size_t error_bytes, const char *format, ...)
{
   if (error && error_bytes) {
      va_list args;
      va_start(args, format);
      vsnprintf(error, error_bytes, format, args);
      va_end(args);
   }
   return false;
}

static uint64_t
agc_present_div_ceil(uint64_t value, uint64_t divisor)
{
   return (value + divisor - 1) / divisor;
}

static uint64_t
agc_present_rounded_ratio(uint64_t a, uint64_t b, uint64_t c)
{
   return (a * b + c / 2) / c;
}

const char *
agc_present_format_name(uint64_t pixel_format)
{
   switch (pixel_format) {
   case AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8:
      return "B8G8R8A8_SRGB";
   case AGC_PRESENT_PIXEL_FORMAT_R8G8B8A8:
      return "R8G8B8A8_SRGB";
   case AGC_PRESENT_PIXEL_FORMAT_B10G10R10A2:
      return "B10G10R10A2";
   case AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2:
      return "R10G10B10A2";
   case AGC_PRESENT_PIXEL_FORMAT_R16G16B16A16_FLOAT:
      return "R16G16B16A16_FLOAT";
   case AGC_PRESENT_PIXEL_FORMAT_B16G16R16A16_FLOAT:
      return "B16G16R16A16_FLOAT";
   case AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2_BT2100_PQ:
      return "R10G10B10A2_BT2100_PQ";
   case AGC_PRESENT_PIXEL_FORMAT_B10G10R10A2_BT2100_PQ:
      return "B10G10R10A2_BT2100_PQ";
   default:
      return "unknown";
   }
}

bool
agc_present_format_of(uint64_t pixel_format, struct agc_present_format *format, char *error, size_t error_bytes)
{
   const uint64_t base = pixel_format & ~AGC_PRESENT_PIXEL_FORMAT_TEN_BIT;
   memset(format, 0, sizeof(*format));
   format->texel_bytes = 4;
   switch (pixel_format) {
   case AGC_PRESENT_PIXEL_FORMAT_R16G16B16A16_FLOAT:
   case AGC_PRESENT_PIXEL_FORMAT_B16G16R16A16_FLOAT:
      format->encoding = AGC_PRESENT_HALF;
      format->red_low = pixel_format == AGC_PRESENT_PIXEL_FORMAT_R16G16B16A16_FLOAT;
      format->texel_bytes = 8;
      return true;
   case AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2_BT2100_PQ:
   case AGC_PRESENT_PIXEL_FORMAT_B10G10R10A2_BT2100_PQ:
      format->encoding = AGC_PRESENT_PQ;
      format->red_low = pixel_format == AGC_PRESENT_PIXEL_FORMAT_R10G10B10A2_BT2100_PQ;
      return true;
   default:
      break;
   }
   if (base != AGC_PRESENT_PIXEL_FORMAT_R8G8B8A8 && base != AGC_PRESENT_PIXEL_FORMAT_B8G8R8A8)
      return agc_present_fail(error, error_bytes, "unsupported display pixel format 0x%016" PRIx64, pixel_format);
   format->encoding = pixel_format & AGC_PRESENT_PIXEL_FORMAT_TEN_BIT ? AGC_PRESENT_UNORM10 : AGC_PRESENT_UNORM8;
   format->red_low = base == AGC_PRESENT_PIXEL_FORMAT_R8G8B8A8;
   return true;
}

static bool
agc_present_gcn_geometry(uint32_t pitch, uint32_t height, uint32_t texel_bytes,
                         struct agc_present_geometry *geometry, char *error, size_t error_bytes)
{
   static const uint32_t macro[5][3] = {{4, 16, 4}, {2, 16, 2}, {1, 16, 2}, {1, 16, 2}, {1, 8, 1}};
   if (!pitch || !height || pitch > AGC_PRESENT_MAX_EXTENT || height > AGC_PRESENT_MAX_EXTENT)
      return agc_present_fail(error, error_bytes, "PS4 display surface %ux%u is outside 1..%u", pitch, height,
                              AGC_PRESENT_MAX_EXTENT);
   if (!util_is_power_of_two_nonzero(texel_bytes) || texel_bytes > 16)
      return agc_present_fail(error, error_bytes, "%u-byte elements have no PS4 2D display tiling", texel_bytes);
   const uint32_t log2 = util_logbase2(texel_bytes);
   const uint32_t bank_height = macro[log2][0];
   const uint32_t banks = macro[log2][1];
   const uint32_t aspect = macro[log2][2];
   geometry->block_width = 64 * aspect;
   geometry->block_height = 8 * bank_height * banks / aspect;
   const uint64_t block_bytes = (uint64_t)geometry->block_width * geometry->block_height * texel_bytes;
   geometry->size = agc_present_div_ceil(pitch, geometry->block_width) *
                    agc_present_div_ceil(height, geometry->block_height) * block_bytes;
   return true;
}

static void
agc_present_r_x_geometry(uint32_t width, uint32_t height, uint32_t texel_bytes,
                         struct agc_present_geometry *geometry)
{
   const uint32_t log2 = 16 - util_logbase2(texel_bytes);
   geometry->block_width = 1u << ((log2 + 1) / 2);
   geometry->block_height = 1u << (log2 / 2);
   geometry->size = agc_present_div_ceil(width, geometry->block_width) *
                    agc_present_div_ceil(height, geometry->block_height) * AGC_PRESENT_BLOCK_64KB;
}

bool
agc_present_geometry_of(const struct agc_display_buffer *buffer, struct agc_present_geometry *geometry,
                        char *error, size_t error_bytes)
{
   memset(geometry, 0, sizeof(*geometry));
   if (!buffer)
      return agc_present_fail(error, error_bytes, "null display buffer");
   if (!buffer->width || !buffer->height || buffer->width > AGC_PRESENT_MAX_EXTENT ||
       buffer->height > AGC_PRESENT_MAX_EXTENT)
      return agc_present_fail(error, error_bytes, "display buffer %ux%u is outside 1..%u", buffer->width,
                              buffer->height, AGC_PRESENT_MAX_EXTENT);
   if (!agc_present_format_of(buffer->pixel_format, &geometry->format, error, error_bytes))
      return false;
   const struct agc_present_format format = geometry->format;
   if (format.encoding == AGC_PRESENT_HALF && buffer->dcc_address)
      return agc_present_fail(error, error_bytes,
                              "DCC metadata on a 16-bit float display buffer is not implemented");
   const uint32_t alignment = buffer->alignment ? buffer->alignment : AGC_PRESENT_ALIGNMENT_64KB;
   if (alignment != AGC_PRESENT_ALIGNMENT_64KB && alignment != AGC_PRESENT_ALIGNMENT_4KB)
      return agc_present_fail(error, error_bytes, "display buffer alignment 0x%x is not supported", alignment);
   if (alignment == AGC_PRESENT_ALIGNMENT_4KB && buffer->tiling_mode != AGC_PRESENT_TILING_LINEAR &&
       buffer->tiling_mode != AGC_PRESENT_TILING_GCN_2D)
      return agc_present_fail(error, error_bytes,
                              "4 KiB display buffer alignment is only supported for linear and PS4 buffers");
   if (!buffer->address || buffer->address % alignment)
      return agc_present_fail(error, error_bytes, "display buffer 0x%" PRIx64 " is not %u KiB aligned",
                              buffer->address, alignment / 1024);

   switch (buffer->tiling_mode) {
   case AGC_PRESENT_TILING_GCN_2D:
      if (buffer->dcc_address || format.encoding == AGC_PRESENT_HALF)
         return agc_present_fail(error, error_bytes, "PS4 display buffers carry no DCC and no half-float texels");
      if (buffer->pitch_in_pixel < buffer->width || buffer->pitch_in_pixel % 128)
         return agc_present_fail(error, error_bytes, "PS4 tiled display pitch %u is not padded to 128 pixels",
                                 buffer->pitch_in_pixel);
      if (!agc_present_gcn_geometry(buffer->pitch_in_pixel, buffer->height, format.texel_bytes, geometry, error,
                                    error_bytes))
         return false;
      geometry->layout = AGC_PRESENT_LAYOUT_GCN_2D;
      geometry->pitch = buffer->pitch_in_pixel;
      break;
   case AGC_PRESENT_TILING_64KB:
   case AGC_PRESENT_TILING_LINEAR: {
      const bool tiled = buffer->tiling_mode == AGC_PRESENT_TILING_64KB;
      if (tiled && buffer->pitch_in_pixel)
         return agc_present_fail(error, error_bytes, "a pitch on a tiled display buffer is not supported");
      if (buffer->dcc_address && !tiled)
         return agc_present_fail(error, error_bytes, "a linear display buffer cannot carry DCC metadata");
      if (!buffer->dcc_address && buffer->dcc_clear_color)
         return agc_present_fail(error, error_bytes, "a DCC clear color needs DCC metadata");
      const uint32_t pitch = buffer->pitch_in_pixel ? buffer->pitch_in_pixel : buffer->width;
      if (pitch < buffer->width || pitch > AGC_PRESENT_MAX_EXTENT)
         return agc_present_fail(error, error_bytes, "invalid linear display pitch %u", pitch);
      if (tiled) {
         agc_present_r_x_geometry(buffer->width, buffer->height, format.texel_bytes, geometry);
         geometry->layout = AGC_PRESENT_LAYOUT_64KB_R_X;
      } else {
         geometry->layout = AGC_PRESENT_LAYOUT_LINEAR;
         geometry->pitch = pitch;
         geometry->block_width = 1;
         geometry->block_height = 1;
         geometry->size = (uint64_t)pitch * buffer->height * format.texel_bytes;
      }
      break;
   }
   default:
      return agc_present_fail(error, error_bytes, "unsupported display tiling mode %u", buffer->tiling_mode);
   }
   if (geometry->size > SIZE_MAX || buffer->address > UINT64_MAX - geometry->size)
      return agc_present_fail(error, error_bytes, "display buffer range overflows");
   return true;
}

size_t
agc_present_display_buffer_size(const struct agc_display_buffer *buffer)
{
   struct agc_present_geometry geometry;
   if (!agc_present_geometry_of(buffer, &geometry, NULL, 0))
      return 0;
   return (size_t)geometry.size;
}

uint64_t
agc_present_source_address(uint64_t address, bool tiled)
{
   return tiled && agc_va_in_guest_window(address) ? agc_mirror_va(address) : address;
}

bool
agc_present_source_of(const struct agc_display_buffer *buffer, enum amd_gfx_level gfx_level,
                      struct agc_present_source *source, char *error, size_t error_bytes)
{
   memset(source, 0, sizeof(*source));
   struct agc_present_geometry geometry;
   if (!agc_present_geometry_of(buffer, &geometry, error, error_bytes))
      return false;
   if (gfx_level < GFX10_3 || gfx_level > GFX11_5)
      return agc_present_fail(error, error_bytes,
                              "display conversion is implemented for gfx10.3, gfx11 and gfx11.5 hosts, not gfx "
                              "level %d",
                              (int)gfx_level);
   const struct agc_present_format format = geometry.format;
   if (format.encoding == AGC_PRESENT_PQ || format.encoding == AGC_PRESENT_HALF)
      return agc_present_fail(error, error_bytes,
                              "display pixel format 0x%016" PRIx64 " (%s) needs tone mapping to SDR, which the "
                              "display conversion does not do",
                              buffer->pixel_format, agc_present_format_name(buffer->pixel_format));
   if (geometry.layout == AGC_PRESENT_LAYOUT_GCN_2D)
      return agc_present_fail(error, error_bytes,
                              "PS4 (GCN 2D) display tiling of the %ux%u buffer at 0x%" PRIx64 " has no host swizzle "
                              "mode; it needs a conversion of its own",
                              buffer->width, buffer->height, buffer->address);
   if (geometry.layout == AGC_PRESENT_LAYOUT_LINEAR) {
      if ((uint64_t)geometry.pitch * format.texel_bytes % AGC_PRESENT_ROW_ALIGNMENT)
         return agc_present_fail(error, error_bytes,
                                 "the linear display pitch of %u pixels is not a multiple of %u bytes, which an "
                                 "image descriptor needs",
                                 geometry.pitch, AGC_PRESENT_ROW_ALIGNMENT);
      source->swizzle_mode = AGC_PRESENT_SW_LINEAR;
      source->pitch = geometry.pitch;
      source->custom_pitch =
         geometry.pitch != align(buffer->width, AGC_PRESENT_ROW_ALIGNMENT / format.texel_bytes);
   } else {
      source->swizzle_mode = AGC_PRESENT_SW_64KB_R_X;
   }
   source->va = agc_present_source_address(buffer->address, geometry.layout != AGC_PRESENT_LAYOUT_LINEAR);
   source->bytes = geometry.size;
   source->width = buffer->width;
   source->height = buffer->height;
   source->texel_bytes = format.texel_bytes;
   const bool ten = format.encoding == AGC_PRESENT_UNORM10;
   source->pipe_format = ten ? PIPE_FORMAT_R10G10B10A2_UNORM : PIPE_FORMAT_R8G8B8A8_UNORM;
   source->swizzle[0] = format.red_low ? PIPE_SWIZZLE_Z : PIPE_SWIZZLE_X;
   source->swizzle[1] = PIPE_SWIZZLE_Y;
   source->swizzle[2] = format.red_low ? PIPE_SWIZZLE_X : PIPE_SWIZZLE_Z;
   source->swizzle[3] = ten ? PIPE_SWIZZLE_1 : PIPE_SWIZZLE_W;
   if (buffer->dcc_address && (gfx_level == GFX10_3 || agc_surface_native_dcc())) {
      if (buffer->dcc_address % AGC_PRESENT_ROW_ALIGNMENT)
         return agc_present_fail(error, error_bytes, "DCC metadata of the display buffer at 0x%" PRIx64
                                 " is at 0x%" PRIx64 ", not 256-byte aligned",
                                 buffer->address, buffer->dcc_address);
      uint32_t control = 0;
      uint64_t rendered = 0;
      const bool target = agc_surface_dcc_target(buffer->address, &rendered, &control);
      source->dcc = true;
      source->meta_va = target && rendered ? rendered : buffer->dcc_address;
      source->dcc_max_compressed = target ? G_028C78_MAX_COMPRESSED_BLOCK_SIZE(control) : V_028C78_MAX_BLOCK_SIZE_64B;
   }
   return true;
}

bool
agc_present_descriptor(const struct radeon_info *info, const struct agc_present_source *source,
                       uint32_t words[AGC_PRESENT_DESCRIPTOR_DWORDS], char *error, size_t error_bytes)
{
   memset(words, 0, AGC_PRESENT_DESCRIPTOR_DWORDS * sizeof(uint32_t));
   if (!info || !source)
      return agc_present_fail(error, error_bytes, "no chip description or no source");
   if (info->gfx_level < GFX10_3 || info->gfx_level > GFX11_5)
      return agc_present_fail(error, error_bytes, "image descriptors are built for gfx10.3 to gfx11.5, not gfx level %d",
                              (int)info->gfx_level);
   if (!info->has_image_opcodes)
      return agc_present_fail(error, error_bytes, "the chip description has no image opcodes");
   if (!source->va || source->va % 256 || !source->width || !source->height ||
       source->width > AGC_PRESENT_MAX_EXTENT || source->height > AGC_PRESENT_MAX_EXTENT ||
       (source->texel_bytes != 4 && source->texel_bytes != 8))
      return agc_present_fail(error, error_bytes, "source 0x%" PRIx64 " %ux%u with %u-byte texels is not addressable",
                              source->va, source->width, source->height, source->texel_bytes);
   if (source->custom_pitch &&
       (source->swizzle_mode != AGC_PRESENT_SW_LINEAR || source->pitch < source->width ||
        (uint64_t)source->pitch * source->texel_bytes % AGC_PRESENT_ROW_ALIGNMENT))
      return agc_present_fail(error, error_bytes, "a custom pitch of %u texels needs a linear source with 256-byte rows",
                              source->pitch);
   if (source->dcc && (source->swizzle_mode != AGC_PRESENT_SW_64KB_R_X || !source->meta_va ||
                       source->meta_va % AGC_PRESENT_ROW_ALIGNMENT || source->dcc_max_compressed > 2))
      return agc_present_fail(error, error_bytes,
                              "DCC metadata at 0x%" PRIx64 " needs a 64KB_R_X source and a 256-byte aligned address",
                              source->meta_va);

   struct radeon_surf surf;
   memset(&surf, 0, sizeof(surf));
   surf.blk_w = 1;
   surf.blk_h = 1;
   surf.bpe = source->texel_bytes;
   surf.is_linear = source->swizzle_mode == AGC_PRESENT_SW_LINEAR;
   surf.u.gfx9.swizzle_mode = source->swizzle_mode;
   surf.u.gfx9.surf_pitch = source->pitch;
   surf.u.gfx9.uses_custom_pitch = source->custom_pitch;
   if (source->dcc) {
      surf.meta_offset = source->meta_va - source->va;
      surf.meta_alignment_log2 = AGC_PRESENT_DCC_ALIGNMENT_LOG2;
      surf.u.gfx9.color.dcc.pipe_aligned = 1;
      surf.u.gfx9.color.dcc.rb_aligned = 1;
      surf.u.gfx9.color.dcc.max_compressed_block_size = source->dcc_max_compressed;
   }

   const struct ac_texture_state texture = {
      .surf = &surf,
      .format = (enum pipe_format)source->pipe_format,
      .img_format = (enum pipe_format)source->pipe_format,
      .width = source->width,
      .height = source->height,
      .depth = 0,
      .type = V_008F1C_SQ_RSRC_IMG_2D,
      .swizzle = {(enum pipe_swizzle)source->swizzle[0], (enum pipe_swizzle)source->swizzle[1],
                  (enum pipe_swizzle)source->swizzle[2], (enum pipe_swizzle)source->swizzle[3]},
      .num_samples = 1,
      .num_storage_samples = 1,
      .first_level = 0,
      .last_level = 0,
      .num_levels = 1,
      .first_layer = 0,
      .last_layer = 0,
      .dcc_enabled = source->dcc,
   };
   const struct ac_mutable_tex_state address = {
      .surf = &surf,
      .va = source->va,
      .dcc_enabled = source->dcc,
   };
   ac_build_texture_descriptor(info, &texture, words);
   ac_set_mutable_tex_desc_fields(info, &address, words);
   return true;
}

bool
agc_present_canvas_fit(uint32_t image_width, uint32_t image_height, uint32_t client_width, uint32_t client_height,
                       struct agc_present_canvas *canvas, char *error, size_t error_bytes)
{
   memset(canvas, 0, sizeof(*canvas));
   if (!image_width || !image_height || image_width > AGC_PRESENT_MAX_EXTENT ||
       image_height > AGC_PRESENT_MAX_EXTENT)
      return agc_present_fail(error, error_bytes, "frame %ux%u is outside 1..%u", image_width, image_height,
                              AGC_PRESENT_MAX_EXTENT);
   if (!client_width || !client_height)
      return agc_present_fail(error, error_bytes, "client area %ux%u is empty", client_width, client_height);
   uint64_t width;
   uint64_t height;
   if ((uint64_t)image_width * client_height >= (uint64_t)client_width * image_height) {
      width = image_width;
      height = agc_present_rounded_ratio(image_width, client_height, client_width);
   } else {
      width = agc_present_rounded_ratio(image_height, client_width, client_height);
      height = image_height;
   }
   width = CLAMP(width, image_width, AGC_PRESENT_MAX_EXTENT);
   height = CLAMP(height, image_height, AGC_PRESENT_MAX_EXTENT);
   canvas->width = (uint32_t)width;
   canvas->height = (uint32_t)height;
   canvas->image_x = (canvas->width - image_width) / 2;
   canvas->image_y = (canvas->height - image_height) / 2;
   canvas->image_width = image_width;
   canvas->image_height = image_height;
   return true;
}

bool
agc_present_canvas_equal(const struct agc_present_canvas *a, const struct agc_present_canvas *b)
{
   return a->width == b->width && a->height == b->height && a->image_x == b->image_x &&
          a->image_y == b->image_y && a->image_width == b->image_width && a->image_height == b->image_height;
}

bool
agc_present_footprint_of(uint32_t width, uint32_t height, struct agc_present_footprint *footprint)
{
   memset(footprint, 0, sizeof(*footprint));
   if (!width || !height || width > AGC_PRESENT_MAX_EXTENT || height > AGC_PRESENT_MAX_EXTENT)
      return false;
   footprint->width = width;
   footprint->height = height;
   footprint->row_pitch = align64((uint64_t)width * 4, AGC_PRESENT_ROW_ALIGNMENT);
   footprint->required_bytes = footprint->row_pitch * (height - 1) + (uint64_t)width * 4;
   footprint->slot_bytes = align64(footprint->required_bytes, AGC_PRESENT_SLOT_GRANULARITY);
   return true;
}

uint64_t
agc_present_image_offset(const struct agc_present_canvas *canvas, const struct agc_present_footprint *footprint)
{
   return (uint64_t)canvas->image_y * footprint->row_pitch + (uint64_t)canvas->image_x * 4;
}

uint32_t
agc_present_frame_slot(uint64_t frame)
{
   return (uint32_t)(frame % AGC_PRESENT_SLOTS);
}

uint32_t
agc_present_gcr_before(void)
{
   return S_587_GLI_INV(V_587_GLI_ALL) | S_587_GLK_INV(1) | S_587_GLV_INV(1) | S_587_GL1_INV(1) |
          S_587_GL2_INV(1) | S_587_GL2_WB(1);
}

uint32_t
agc_present_gcr_after(void)
{
   return S_587_GLM_WB(1) | S_587_GLM_INV(1) | S_587_GLV_INV(1) | S_587_GL1_INV(1) | S_587_GL2_INV(1) |
          S_587_GL2_WB(1);
}

uint32_t *
agc_present_emit_acquire(uint32_t *out, uint32_t gcr_cntl)
{
   *out++ = PKT3(PKT3_ACQUIRE_MEM, 6, 0);
   *out++ = 0;
   *out++ = 0xffffffff;
   *out++ = 0xff;
   *out++ = 0;
   *out++ = 0;
   *out++ = 0x0000000a;
   *out++ = gcr_cntl;
   return out;
}

uint32_t *
agc_present_emit_partial_flush(uint32_t *out)
{
   *out++ = PKT3(PKT3_EVENT_WRITE, 0, 0);
   *out++ = EVENT_TYPE(V_028A90_CS_PARTIAL_FLUSH) | EVENT_INDEX(4);
   return out;
}

uint32_t
agc_present_padding(uint32_t used, uint32_t multiple)
{
   if (!multiple)
      multiple = 1;
   uint32_t padding = (multiple - used % multiple) % multiple;
   if (padding == 1)
      padding += multiple;
   return padding;
}

uint32_t *
agc_present_emit_padding(uint32_t *out, uint32_t dwords)
{
   if (dwords == 1) {
      *out++ = PKT3_NOP_PAD;
   } else if (dwords > 1) {
      *out++ = PKT3(PKT3_NOP, dwords - 2, 0);
      memset(out, 0, (dwords - 1) * sizeof(uint32_t));
      out += dwords - 1;
   }
   return out;
}
