/* SPDX-License-Identifier: MIT */

#include "pan_afbc.h"
#include "pan_desc.h"
#include "pan_fb.h"
#include "pan_mod.h"

bool pan_image_view_can_crc_v14(const struct pan_image_view *, unsigned);

static unsigned failures;
static struct pan_image_plane plane;
static struct pan_image image;
static struct pan_image_view view;

#define CHECK(expr)                                                             \
   do {                                                                        \
      if (!(expr)) {                                                           \
         fprintf(stderr, "%s:%u: %s failed\n", __FILE__, __LINE__, #expr);       \
         failures++;                                                          \
         return;                                                              \
      }                                                                       \
   } while (0)
#define CHECK_FALSE(expr) CHECK(!(expr))
#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))

static void
fixture_init(void)
{
   memset(&plane, 0, sizeof(plane));
   memset(&image, 0, sizeof(image));
   memset(&view, 0, sizeof(view));

   image.props.modifier = DRM_FORMAT_MOD_ARM_AFBC(
      AFBC_FORMAT_MOD_BLOCK_SIZE_16x16 | AFBC_FORMAT_MOD_SPARSE |
      AFBC_FORMAT_MOD_TILED);
   image.props.format = PIPE_FORMAT_R8G8_UNORM;
   image.props.extent_px = (struct pan_image_extent){2560, 1440, 1};
   image.props.nr_samples = 1;
   image.props.dim = MALI_TEXTURE_DIMENSION_2D;
   image.props.nr_slices = 1;
   image.props.array_size = 1;
   image.props.crc = true;
   image.props.afbc_crc = true;
   image.mod_handler = pan_mod_get_handler(15, image.props.modifier);
   image.planes[0] = &plane;
   view.format = image.props.format;
   view.dim = image.props.dim;
   view.nr_samples = 1;
   view.planes[0] = (struct pan_image_plane_ref){&image, 0};
}

static void
CRCV15_BackendAndGenerationGate(void)
{
   CHECK(pan_image_view_can_crc_v15(&view, 2048));
   CHECK_FALSE(pan_image_view_can_crc_v14(&view, 2048));
   image.props.nr_samples = 2;
   CHECK_FALSE(pan_image_view_can_crc_v15(&view, 2048));
   image.props.nr_samples = 1;
   image.props.afbc_crc = false;
   CHECK_FALSE(pan_image_view_can_crc_v15(&view, 2048));
   image.props.modifier = DRM_FORMAT_MOD_LINEAR;
   CHECK(pan_image_view_can_crc_v15(&view, 2048));
}

static void
CRCV15_MissingCRCAndMip(void)
{
   CHECK_FALSE(pan_image_view_can_crc_v15(NULL, 2048));
   image.props.crc = false;
   CHECK_FALSE(pan_image_view_can_crc_v15(&view, 2048));
   image.props.crc = true;
   view.first_level = 1;
   CHECK_FALSE(pan_image_view_can_crc_v15(&view, 2048));
}

static void
CRCV15_SuperblockMustFitTile(void)
{
   const struct {
      uint64_t block;
      unsigned tile;
      bool allowed;
   } cases[] = {
      {AFBC_FORMAT_MOD_BLOCK_SIZE_16x16, 256, true},
      {AFBC_FORMAT_MOD_BLOCK_SIZE_16x16, 128, false},
      {AFBC_FORMAT_MOD_BLOCK_SIZE_32x8, 256, false},
      {AFBC_FORMAT_MOD_BLOCK_SIZE_32x8, 512, true},
      {AFBC_FORMAT_MOD_BLOCK_SIZE_64x4, 1024, false},
      {AFBC_FORMAT_MOD_BLOCK_SIZE_64x4, 2048, true},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(cases); i++) {
      image.props.modifier = DRM_FORMAT_MOD_ARM_AFBC(
         cases[i].block | AFBC_FORMAT_MOD_SPARSE | AFBC_FORMAT_MOD_TILED);
      CHECK_EQ(pan_image_view_can_crc_v15(&view, cases[i].tile), cases[i].allowed);
   }
}

static void
CRCV15_CapturedGBufferCRCLayout(void)
{
   CHECK(pan_image_layout_init(15, &image, 0, NULL));
   const struct pan_image_slice_layout *slice = &plane.layout.slices[0];
   CHECK_EQ(slice->crc.header_offset_B, 0x7bc000u);
   CHECK_EQ(slice->crc.offset_B, 0x7bc040u);
   CHECK_EQ(slice->crc.stride_B, 1280u);
   CHECK_EQ(slice->crc.size_B, 117760u);
   CHECK_LE(slice->crc.offset_B + slice->crc.size_B,
             plane.layout.data_size_B);
}

static void
CRCV15_MRTSelectionAndDescriptor(void)
{
   CHECK(pan_image_layout_init(15, &image, 0, NULL));
   plane.base = 0x5ff48b2000ull;
   const struct pan_image_slice_layout *slice = &plane.layout.slices[0];
   struct pan_fb_layout fb = {};
   fb.width_px = 2560;
   fb.height_px = 1440;
   fb.sample_count = 1;
   fb.rt_count = 6;
   fb.tile_size_px = 2048;
   fb.tile_rt_alloc_B = 6 * 2048 * 4;
   for (unsigned rt = 0; rt < fb.rt_count; rt++)
      fb.rt_formats[rt] = PIPE_FORMAT_R8G8_UNORM;

   struct pan_fb_store store = {};
   store.rts[2] = pan_fb_store_iview(&view);
   store.rts[2].crc_header_addr = plane.base + slice->crc.header_offset_B;
   struct pan_fb_desc_info info = {};
   info.fb = &fb;
   info.store = &store;
   struct pan_fb_crc_rt_info crc = {};
   CHECK(pan_fb_get_crc_rt_info_v15(&info, &crc));
   CHECK_EQ(crc.rt, 2);
   CHECK_EQ(crc.header_addr, 0x5ff506e000ull);
   CHECK(pan_fb_needs_zs_crc_ext_v15(&info));

   struct mali_zs_crc_extension_packed dbd;
   struct mali_rgb_render_target_packed rtd[PAN_MAX_RTS];
   struct pan_fb_descs descs = {};
   descs.zs_crc = &dbd;
   descs.rts = rtd;
   pan_emit_fb_desc_v15(&info, &descs);
   pan_unpack(&dbd, ZS_CRC_EXTENSION, cfg);
   CHECK_EQ(cfg.crc.render_target, 2u);
   CHECK_EQ(cfg.crc.row_stride, 1280u);
   CHECK_EQ(cfg.crc.base, 0x5ff506e040ull);
   CHECK_EQ(cfg.crc.clear_color & 0xffffu, 0u);
   CHECK_NE(cfg.crc.clear_color, 0u);

   store.rts[2].store = false;
   CHECK_FALSE(pan_fb_get_crc_rt_info_v15(&info, &crc));
   CHECK_FALSE(pan_fb_needs_zs_crc_ext_v15(&info));
   store.rts[2].store = true;
   store.rts[2].crc_header_addr = 0;
   CHECK_FALSE(pan_fb_get_crc_rt_info_v15(&info, &crc));
   store.rts[2].crc_header_addr = plane.base + slice->crc.header_offset_B;
   fb.tile_size_px = 128;
   CHECK_FALSE(pan_fb_get_crc_rt_info_v15(&info, &crc));
   fb.tile_size_px = 2048;
   info.store = NULL;
   CHECK_FALSE(pan_fb_get_crc_rt_info_v15(&info, &crc));
}

static void
PolygonListV15_CapturedWordsRoundTrip(void)
{
   const uint64_t words[] = {
      0x008c0060007c9300ull,
      0x00800060000f4280ull,
      0x00040060000ddb00ull,
      0x000c0060000d8e00ull,
      0x00ac0060000c1880ull,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(words); i++) {
      uint64_t word = words[i];
      struct mali_tiler_context_packed input = {};
      memcpy(&input, &word, sizeof(word));
      pan_unpack(&input, TILER_CONTEXT, cfg);
      CHECK_EQ(cfg.polygon_list, word & 0xffffffffffffull);
      CHECK_EQ(cfg.polygon_list_flags, word >> 48);
      struct mali_tiler_context_packed output;
      MALI_TILER_CONTEXT_pack(&output, &cfg);
      CHECK_EQ(memcmp(&input, &output, sizeof(word)), 0);
   }
}

static void
PolygonListV15_InitialStateIsZero(void)
{
   struct mali_tiler_context_packed td;
   pan_pack(&td, TILER_CONTEXT, cfg) {
      cfg.effective_tile_size = 2048;
      cfg.fb_width = 2560;
      cfg.fb_height = 1440;
   }
   uint64_t word;
   memcpy(&word, &td, sizeof(word));
   CHECK_EQ(word, 0u);
}

int
main(void)
{
   fixture_init();
   CRCV15_BackendAndGenerationGate();
   fixture_init();
   CRCV15_MissingCRCAndMip();
   fixture_init();
   CRCV15_SuperblockMustFitTile();
   fixture_init();
   CRCV15_CapturedGBufferCRCLayout();
   fixture_init();
   CRCV15_MRTSelectionAndDescriptor();
   fixture_init();
   PolygonListV15_CapturedWordsRoundTrip();
   fixture_init();
   PolygonListV15_InitialStateIsZero();
   printf("CRC/Polygon List: 7 cases, %u failures\n", failures);
   return failures != 0;
}
