#include <setjmp.h>
#include <string.h>

#include "agc_debug.h"
#include "agc_graphics.h"
#include "agc_register_defaults.h"
#include "agc_shader.h"
#include "agc_surface.h"
#include "sid.h"
#include "test.h"

#define CX AGC_REG_BANK_CX
#define SH AGC_REG_BANK_SH
#define UC AGC_REG_BANK_UC

#define CX_REG(reg) (((reg) - SI_CONTEXT_REG_OFFSET) / 4)
#define SH_REG(reg) (((reg) - SI_SH_REG_OFFSET) / 4)
#define UC_REG(reg) (((reg) - CIK_UCONFIG_REG_OFFSET) / 4)

#define COUNT_OF(a) (sizeof(a) / sizeof((a)[0]))

#define CONVERTED_MAX 256

struct converted {
   uint32_t count;
   struct agc_register reg[CONVERTED_MAX];
};

static jmp_buf fatal_jump;
static char fatal_text[512];

static void
on_fatal(const char *message)
{
   strncpy(fatal_text, message, sizeof(fatal_text) - 1);
   longjmp(fatal_jump, 1);
}

#define EXPECT_FATAL(statement, needle)                 \
   do {                                                 \
      fatal_text[0] = 0;                                \
      if (setjmp(fatal_jump) == 0) {                    \
         statement;                                     \
         CHECK(!"expected a fatal error");              \
      } else {                                          \
         CHECK(strstr(fatal_text, needle) != NULL);     \
      }                                                 \
   } while (0)

static struct converted
convert_array(enum agc_reg_bank bank, const struct agc_register *guest, uint32_t count)
{
   struct converted out;
   out.count = 0;
   if (agc_reg_convert_pairs_bound(count) <= CONVERTED_MAX)
      out.count = agc_reg_convert_pairs(bank, guest, count, out.reg, "test");
   else
      CHECK(!"array too long for the test buffer");
   return out;
}

static struct converted
convert(enum agc_reg_bank bank, uint32_t offset, uint32_t value)
{
   struct agc_register guest = {offset, value};
   return convert_array(bank, &guest, 1);
}

static bool
find_write(const struct converted *out, uint32_t offset, uint32_t *value)
{
   for (uint32_t i = 0; i < out->count; i++) {
      if (out->reg[i].offset == offset) {
         *value = out->reg[i].value;
         return true;
      }
   }
   return false;
}

static void
expect_one(enum agc_reg_bank bank, uint32_t offset, uint32_t value, uint32_t host_offset, uint32_t host_value,
           const char *file, int line)
{
   struct converted out = convert(bank, offset, value);
   agc_test_check_eq(out.count, 1, file, line, "one host write");
   if (out.count != 1)
      return;
   agc_test_check_eq(out.reg[0].offset, host_offset, file, line, "host offset");
   agc_test_check_eq(out.reg[0].value, host_value, file, line, "host value");
}

static void
expect_none(enum agc_reg_bank bank, uint32_t offset, uint32_t value, const char *file, int line)
{
   agc_test_check_eq(convert(bank, offset, value).count, 0, file, line, "no host write");
}

static void
expect_blend(unsigned mrt, uint32_t value, uint32_t host_value, uint32_t opt_value, const char *file, int line)
{
   (void)opt_value;
   uint32_t blend = CX_REG(R_028780_CB_BLEND0_CONTROL) + mrt;
   struct converted out = convert(CX, blend, value);
   agc_test_check_eq(out.count, 1, file, line, "one host write");
   if (out.count != 1)
      return;
   agc_test_check_eq(out.reg[0].offset, blend, file, line, "blend control offset");
   agc_test_check_eq(out.reg[0].value, host_value, file, line, "blend control value");
}

static void
expect_write(const struct converted *out, uint32_t offset, uint32_t host_value, const char *file, int line)
{
   uint32_t found = 0xdeadbeef;
   agc_test_check(find_write(out, offset, &found), file, line, "host write present");
   agc_test_check_eq(found, host_value, file, line, "host write value");
}

static void
expect_info(uint32_t offset, uint32_t value, uint32_t host_value, const char *file, int line)
{
   struct converted out = convert(AGC_REG_BANK_CX, offset, value);
   agc_test_check_eq(out.count, 2, file, line, "CB_COLOR_INFO and FDCC_CONTROL");
   expect_write(&out, offset, host_value, file, line);
   expect_write(&out, offset + 2, 0, file, line);
}

#define EXPECT_INFO(o, v, hv)       expect_info(o, v, hv, __FILE__, __LINE__)
#define EXPECT_ONE(b, o, v, ho, hv) expect_one(b, o, v, ho, hv, __FILE__, __LINE__)
#define EXPECT_SAME(b, o, v, hv)    expect_one(b, o, v, o, hv, __FILE__, __LINE__)
#define EXPECT_NONE(b, o, v)        expect_none(b, o, v, __FILE__, __LINE__)
#define EXPECT_BLEND(m, v, hv, ov)  expect_blend(m, v, hv, ov, __FILE__, __LINE__)
#define EXPECT_WRITE(c, o, hv)      expect_write(&(c), o, hv, __FILE__, __LINE__)

static void
expect_primitive(uint32_t value, uint32_t host_value, uint32_t output, const char *file, int line)
{
   struct converted out = convert(UC, UC_REG(R_030908_VGT_PRIMITIVE_TYPE), value);
   agc_test_check_eq(out.count, 2, file, line, "two host writes");
   if (out.count != 2)
      return;
   agc_test_check_eq(out.reg[0].offset, UC_REG(R_030908_VGT_PRIMITIVE_TYPE), file, line, "primitive offset");
   agc_test_check_eq(out.reg[0].value, host_value, file, line, "primitive value");
   agc_test_check_eq(out.reg[1].offset, UC_REG(R_030998_VGT_GS_OUT_PRIM_TYPE), file, line, "output offset");
   agc_test_check_eq(out.reg[1].value, output, file, line, "output value");
}

#define EXPECT_PRIMITIVE(v, hv, o) expect_primitive(v, hv, o, __FILE__, __LINE__)

static enum agc_reg_class
class_of(enum agc_reg_bank bank, uint32_t offset)
{
   const struct agc_reg_entry *entry = agc_reg_lookup(agc_host_regs, bank, offset);
   return entry ? (enum agc_reg_class)entry->cls : AGC_REG_UNDEFINED;
}

static void
test_pending(enum agc_host host)
{
   const struct agc_reg_tables *tables = agc_reg_tables_get(host);
   uint32_t pending = 0;
   for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++) {
      for (uint32_t offset = 0; offset < tables->bank[bank].count; offset++) {
         if (tables->bank[bank].entries[offset].cls != AGC_REG_PENDING)
            continue;
         pending++;
      }
   }
   CHECK_EQ(pending, 0);
   CHECK_EQ(agc_reg_class_count(host, CX, AGC_REG_PENDING), 0);
   CHECK_EQ(agc_reg_class_count(host, SH, AGC_REG_PENDING), 0);
   CHECK_EQ(agc_reg_class_count(host, UC, AGC_REG_PENDING), 0);
}

static void
test_classes(void)
{
   CHECK_EQ(class_of(CX, CX_REG(R_028008_DB_DEPTH_VIEW)), AGC_REG_PASS);
   CHECK_EQ(class_of(CX, CX_REG(R_0286C4_SPI_VS_OUT_CONFIG)), AGC_REG_PASS);
   CHECK_EQ(class_of(CX, CX_REG(R_02803C_DB_RESERVED_REG_2)), AGC_REG_PASS);
   CHECK_EQ(class_of(CX, CX_REG(R_0287FC_GE_MAX_OUTPUT_PER_SUBGROUP)), AGC_REG_PASS);
   CHECK_EQ(class_of(CX, CX_REG(R_028B4C_GE_NGG_SUBGRP_CNTL)), AGC_REG_PASS);
   CHECK_EQ(class_of(CX, CX_REG(R_02870C_SPI_SHADER_POS_FORMAT)), AGC_REG_PASS);
   CHECK_EQ(class_of(SH, SH_REG(R_00B028_SPI_SHADER_PGM_RSRC1_PS)), AGC_REG_PASS);
   CHECK_EQ(class_of(SH, SH_REG(R_00B22C_SPI_SHADER_PGM_RSRC2_GS)), AGC_REG_PASS);

   CHECK_EQ(class_of(CX, CX_REG(R_028080_TA_BC_BASE_ADDR)), AGC_REG_ADDR);
   CHECK_EQ(class_of(SH, SH_REG(R_00B020_SPI_SHADER_PGM_LO_PS)), AGC_REG_ADDR);
   CHECK_EQ(class_of(SH, SH_REG(R_00B320_SPI_SHADER_PGM_LO_ES)), AGC_REG_ADDR);
   CHECK_EQ(class_of(SH, SH_REG(R_00B324_SPI_SHADER_PGM_HI_ES)), AGC_REG_ADDR);
   CHECK_EQ(class_of(SH, SH_REG(R_00B520_SPI_SHADER_PGM_LO_LS)), AGC_REG_ADDR);
   CHECK_EQ(class_of(SH, SH_REG(R_00B830_COMPUTE_PGM_LO)), AGC_REG_ADDR);
   CHECK_EQ(class_of(UC, UC_REG(R_030E00_TA_CS_BC_BASE_ADDR)), AGC_REG_ADDR);
   CHECK_EQ(class_of(UC, UC_REG(R_030940_VGT_TF_MEMORY_BASE)), AGC_REG_ADDR);

   CHECK_EQ(class_of(CX, CX_REG(R_028C60_CB_COLOR0_BASE)), AGC_REG_MERGE);
   CHECK_EQ(class_of(CX, CX_REG(R_028C60_CB_COLOR0_BASE) + 7 * 15), AGC_REG_MERGE);
   CHECK_EQ(class_of(CX, CX_REG(R_028E40_CB_COLOR0_BASE_EXT) + 7), AGC_REG_MERGE);
   CHECK_EQ(class_of(CX, CX_REG(R_028048_DB_Z_READ_BASE)), AGC_REG_MERGE);
   CHECK_EQ(class_of(CX, CX_REG(R_028074_DB_STENCIL_WRITE_BASE_HI)), AGC_REG_MERGE);
   CHECK_EQ(class_of(CX, CX_REG(R_028780_CB_BLEND0_CONTROL) + 7), AGC_REG_GUEST);
   CHECK_EQ(class_of(UC, UC_REG(R_03096C_GE_CNTL)), AGC_REG_GUEST);
   CHECK_EQ(class_of(UC, UC_REG(R_030908_VGT_PRIMITIVE_TYPE)), AGC_REG_SPLIT);
   CHECK_EQ(class_of(CX, CX_REG(R_028A6C_VGT_GS_OUT_PRIM_TYPE)), AGC_REG_GUEST);

   CHECK_EQ(class_of(CX, CX_REG(R_028014_DB_HTILE_DATA_BASE)), AGC_REG_ADDR);
   CHECK_EQ(class_of(CX, CX_REG(R_028078_DB_HTILE_DATA_BASE_HI)), AGC_REG_ADDR);
   CHECK_EQ(class_of(CX, CX_REG(R_028ABC_DB_HTILE_SURFACE)), AGC_REG_GUEST);
   CHECK_EQ(class_of(CX, CX_REG(R_028A44_VGT_GS_ONCHIP_CNTL)), AGC_REG_DROP);
   CHECK_EQ(class_of(CX, CX_REG(R_028C44_PA_SC_BINNER_CNTL_0)), AGC_REG_DROP);
   CHECK_EQ(class_of(CX, CX_REG(R_028B50_VGT_TESS_DISTRIBUTION)), AGC_REG_DROP);
   CHECK_EQ(class_of(SH, SH_REG(R_00B01C_SPI_SHADER_PGM_RSRC3_PS)), AGC_REG_PASS);
   CHECK_EQ(class_of(SH, SH_REG(R_00B204_SPI_SHADER_PGM_RSRC4_GS)), AGC_REG_PASS);
   CHECK_EQ(class_of(SH, SH_REG(R_00B8A0_COMPUTE_PGM_RSRC3)), AGC_REG_PASS);
   CHECK_EQ(class_of(CX, CX_REG(R_028714_SPI_SHADER_COL_FORMAT)), AGC_REG_PASS);
   CHECK_EQ(class_of(SH, SH_REG(R_00B220_SPI_SHADER_PGM_LO_GS)), AGC_REG_DROP);
   CHECK_EQ(class_of(SH, SH_REG(R_00B424_SPI_SHADER_PGM_HI_HS)), AGC_REG_DROP);
   CHECK_EQ(class_of(SH, SH_REG(R_00B9F4_COMPUTE_DISPATCH_TUNNEL)), AGC_REG_DROP);
   CHECK_EQ(class_of(UC, UC_REG(R_030980_GE_PC_ALLOC)), AGC_REG_DROP);
   CHECK_EQ(class_of(UC, UC_REG(R_030800_GRBM_GFX_INDEX)), AGC_REG_DROP);
   for (uint32_t offset = 0x1000; offset < 0x2000; offset++) {
      enum agc_reg_class cls = class_of(UC, offset);
      CHECK(cls == AGC_REG_DROP || cls == AGC_REG_UNDEFINED);
   }
   for (unsigned slot = 0; slot < 8; slot++) {
      uint32_t block = slot * 15;
      CHECK_EQ(class_of(CX, CX_REG(R_028C7C_CB_COLOR0_CMASK) + block), AGC_REG_DROP);
      CHECK_EQ(class_of(CX, CX_REG(R_028C84_CB_COLOR0_FMASK) + block), AGC_REG_DROP);
      CHECK_EQ(class_of(CX, CX_REG(R_028C8C_CB_COLOR0_CLEAR_WORD0) + block), AGC_REG_DROP);
      CHECK_EQ(class_of(CX, CX_REG(R_028C90_CB_COLOR0_CLEAR_WORD1) + block), AGC_REG_DROP);
      CHECK_EQ(class_of(CX, CX_REG(R_028C94_CB_COLOR0_DCC_BASE) + block), AGC_REG_MERGE);
      CHECK_EQ(class_of(CX, CX_REG(R_028E60_CB_COLOR0_CMASK_BASE_EXT) + slot), AGC_REG_DROP);
      CHECK_EQ(class_of(CX, CX_REG(R_028E80_CB_COLOR0_FMASK_BASE_EXT) + slot), AGC_REG_DROP);
      CHECK_EQ(class_of(CX, CX_REG(R_028EA0_CB_COLOR0_DCC_BASE_EXT) + slot), AGC_REG_ADDR);
      CHECK_EQ(class_of(CX, CX_REG(R_028760_SX_MRT0_BLEND_OPT) + slot), AGC_REG_DROP);
      CHECK_EQ(class_of(CX, CX_REG(R_028C78_CB_COLOR0_DCC_CONTROL) + block), AGC_REG_GUEST);
   }

   CHECK_EQ(class_of(CX, CX_REG(R_028004_DB_COUNT_CONTROL)), AGC_REG_FIELD);
   CHECK_EQ(class_of(CX, CX_REG(R_0286D8_SPI_PS_IN_CONTROL)), AGC_REG_FIELD);
   CHECK_EQ(class_of(SH, SH_REG(R_00B87C_COMPUTE_MISC_RESERVED)), AGC_REG_FIELD);
   CHECK_EQ(class_of(UC, UC_REG(R_030D20_SQC_CACHES)), AGC_REG_FIELD);

   CHECK_EQ(class_of(SH, SH_REG(R_00B208_SPI_SHADER_USER_DATA_ADDR_LO_GS)), AGC_REG_MOVE);
   CHECK_EQ(class_of(SH, SH_REG(R_00B40C_SPI_SHADER_USER_DATA_ADDR_HI_HS)), AGC_REG_MOVE);
}

static void
test_ps5_only_registers(void)
{
   static const struct {
      enum agc_reg_bank bank;
      uint32_t offset;
   } dropped[] = {
      {CX, 0x0db}, {SH, 0x280}, {UC, 0x0a2}, {UC, 0x035}, {UC, 0x03c}, {UC, 0x248}, {CX, 0x20d}, {CX, 0x0eb},
      {CX, 0x0ec}, {CX, 0x0f1}, {CX, 0x0f2}, {UC, 0x2bb}, {UC, 0x2bf}, {UC, 0x2c4}, {UC, 0x2d3}, {UC, 0x2e4},
      {UC, 0x2eb}, {UC, 0x384},
   };
   for (unsigned i = 0; i < COUNT_OF(dropped); i++) {
      CHECK_EQ(class_of(dropped[i].bank, dropped[i].offset), AGC_REG_DROP);
      EXPECT_NONE(dropped[i].bank, dropped[i].offset, 0xffffffff);
   }
   static const struct {
      enum agc_reg_bank bank;
      uint32_t offset;
   } gated[] = {
      {CX, 0x006}, {CX, 0x0e8}, {CX, 0x0e9}, {CX, 0x0ea}, {CX, 0x188}, {CX, 0x18e}, {UC, 0x26c}, {UC, 0x26e},
      {UC, 0x278},
   };
   for (unsigned i = 0; i < COUNT_OF(gated); i++) {
      CHECK_EQ(class_of(gated[i].bank, gated[i].offset), AGC_REG_GUEST);
      EXPECT_NONE(gated[i].bank, gated[i].offset, 0);
      EXPECT_FATAL(convert(gated[i].bank, gated[i].offset, 1), "asks for a function this host does not have");
   }
   EXPECT_NONE(CX, 0x0f0, 0);
   EXPECT_FATAL(convert(CX, 0x0f0, 1), "cx:0x0f0");
   EXPECT_NONE(UC, 0x383, 0x40000040);
   EXPECT_FATAL(convert(UC, 0x383, 0x40000080), "texture gradient factors other than identity");
}

static void
test_gfx12(void)
{
   uint32_t fixed = S_03096C_PRIM_GRP_SIZE_GFX11(256) | S_03096C_DIS_PG_SIZE_ADJUST_FOR_STRIP(1);
   uint32_t reg = UC_REG(R_03096C_GE_CNTL);
   EXPECT_SAME(UC, reg, 0, S_03096C_PRIMS_PER_SUBGRP(128) | S_03096C_VERTS_PER_SUBGRP(128) | fixed);
   EXPECT_SAME(UC, reg,
               S_03096C_PRIM_GRP_SIZE_GFX10(120) | S_03096C_VERT_GRP_SIZE(100) | S_03096C_BREAK_WAVE_AT_EOI(1) |
                  S_03096C_PACKET_TO_ONE_PA(1),
               120 | (100 << 9) | (3u << 19) | (256u << 21) | (1u << 31));

   EXPECT_BLEND(0, 0x20010001, 0x20010001, 0x06000600);
   EXPECT_ONE(CX, CX_REG(R_028808_CB_COLOR_CONTROL), 0x00cc0010, CX_REG(R_028858_CB_COLOR_CONTROL), 0x00cc0010);
   EXPECT_FATAL(convert(CX, CX_REG(R_028808_CB_COLOR_CONTROL), S_028808_MODE(V_028808_CB_RESOLVE)), "RESOLVE");
   EXPECT_PRIMITIVE(V_030908_DI_PT_2D_RECTANGLE, V_030908_DI_PT_RECTLIST, V_030998_RECT_2D);
   EXPECT_PRIMITIVE(V_030908_DI_PT_TRISTRIP, V_030908_DI_PT_TRISTRIP, V_030998_TRISTRIP);

   static const uint32_t scissors[] = {CX_REG(R_028204_PA_SC_WINDOW_SCISSOR_TL),
                                       CX_REG(R_028240_PA_SC_GENERIC_SCISSOR_TL),
                                       CX_REG(R_028250_PA_SC_VPORT_SCISSOR_0_TL),
                                       CX_REG(R_028250_PA_SC_VPORT_SCISSOR_0_TL) + 30};
   for (unsigned i = 0; i < COUNT_OF(scissors); i++) {
      EXPECT_SAME(CX, scissors[i], 0x80000000, 0);
      EXPECT_SAME(CX, scissors[i], 0x80000000 | 0x12340567, 0x12340567);
      EXPECT_SAME(CX, scissors[i], 0x7fff7fff, 0x7fff7fff);
   }
   EXPECT_NONE(CX, CX_REG(R_028200_PA_SC_WINDOW_OFFSET), 0);
   EXPECT_FATAL(convert(CX, CX_REG(R_028200_PA_SC_WINDOW_OFFSET), 0x00100010), "cx:0x080");

   reg = CX_REG(R_028B90_VGT_GS_INSTANCE_CNT);
   EXPECT_ONE(CX, reg, 1 | (63 << 2), CX_REG(0x028B3C), 1 | (63 << 2));
   EXPECT_FATAL(convert(CX, reg, 1 | (64 << 2)), "does not fit the gfx12 field");

   agc_host_caps.shader_engines = 2;
   EXPECT_SAME(CX, CX_REG(R_0286E8_SPI_TMPRING_SIZE), 1280 | (8 << 12), 640 | (32 << 12));
   EXPECT_SAME(UC, UC_REG(R_030938_VGT_TF_RING_SIZE), 0x8000, 0x4000);
   EXPECT_SAME(UC, UC_REG(R_03093C_VGT_HS_OFFCHIP_PARAM), 511, 255);
   agc_host_caps.shader_engines = 1;

   EXPECT_NONE(CX, CX_REG(R_028C78_CB_COLOR0_DCC_CONTROL), 0x48);
   EXPECT_NONE(CX, CX_REG(R_028C94_CB_COLOR0_DCC_BASE), 0x1234);
   EXPECT_NONE(CX, CX_REG(R_028ABC_DB_HTILE_SURFACE), 0x3);
   EXPECT_NONE(CX, 0x00a, 0x55);
   EXPECT_NONE(CX, 0x00b, 0x3f800000);
   EXPECT_NONE(SH, SH_REG(R_00B9F4_COMPUTE_DISPATCH_TUNNEL), 0x4ff);
   EXPECT_SAME(SH, SH_REG(R_00B028_SPI_SHADER_PGM_RSRC1_PS), 0xffffffff, 0xffffffff);
   EXPECT_SAME(SH, SH_REG(R_00B848_COMPUTE_PGM_RSRC1), 0xffffffff, 0xffffffff);
   EXPECT_NONE(CX, CX_REG(R_028A6C_VGT_GS_OUT_PRIM_TYPE), V_028A6C_TRISTRIP);
   EXPECT_FATAL(convert(UC, 0x41d, 1), "uc:0x41d");

   uint32_t pending = 0;
   for (uint32_t version = 0; version <= AGC_DEFAULTS_VERSION_MAX; version++) {
      uint32_t group_count = 0;
      const struct agc_defaults_group *groups = agc_register_defaults_groups(version, false, &group_count);
      for (uint32_t g = 0; g < group_count; g++) {
         for (uint32_t r = 0; r < groups[g].count; r++) {
            enum agc_reg_class cls = class_of((enum agc_reg_bank)groups[g].bank, groups[g].registers[r].offset);
            CHECK(cls != AGC_REG_UNDEFINED);
            if (cls == AGC_REG_PENDING) {
               pending++;
               CHECK_EQ(groups[g].bank, CX);
            }
         }
      }
   }
   CHECK(pending > 0);
}

static void
test_other_hosts(void)
{
   const struct agc_reg_tables *gfx103 = agc_reg_tables_get(AGC_HOST_GFX103);
   const struct agc_reg_tables *gfx12 = agc_reg_tables_get(AGC_HOST_GFX12);
   CHECK_EQ(agc_reg_lookup(gfx103, CX, CX_REG(R_028000_DB_RENDER_CONTROL))->cls, AGC_REG_PASS);
   CHECK_EQ(agc_reg_lookup(gfx103, CX, CX_REG(R_028C60_CB_COLOR0_BASE))->cls, AGC_REG_PASS);
   CHECK_EQ(agc_reg_lookup(gfx103, CX, CX_REG(R_028014_DB_HTILE_DATA_BASE))->cls, AGC_REG_PASS);
   CHECK_EQ(agc_reg_lookup(gfx103, CX, CX_REG(R_028A6C_VGT_GS_OUT_PRIM_TYPE))->cls, AGC_REG_PASS);
   CHECK_EQ(agc_reg_lookup(gfx103, CX, CX_REG(R_028780_CB_BLEND0_CONTROL))->cls, AGC_REG_PASS);
   CHECK_EQ(agc_reg_lookup(gfx103, SH, SH_REG(R_00B220_SPI_SHADER_PGM_LO_GS))->cls, AGC_REG_PASS);
   CHECK_EQ(agc_reg_lookup(gfx103, SH, SH_REG(R_00B020_SPI_SHADER_PGM_LO_PS))->cls, AGC_REG_ADDR);
   CHECK_EQ(agc_reg_lookup(gfx103, UC, UC_REG(R_03096C_GE_CNTL))->cls, AGC_REG_PASS);
   CHECK_EQ(agc_reg_lookup(gfx12, SH, SH_REG(R_00B020_SPI_SHADER_PGM_LO_PS))->cls, AGC_REG_ADDR);
   CHECK_EQ(agc_reg_lookup(gfx12, SH, SH_REG(R_00B320_SPI_SHADER_PGM_LO_ES))->cls, AGC_REG_ADDR);
   CHECK_EQ(agc_reg_lookup(gfx12, SH, SH_REG(R_00B320_SPI_SHADER_PGM_LO_ES))->host_offset,
            SH_REG(R_00B224_SPI_SHADER_PGM_LO_ES));
   CHECK_EQ(agc_reg_lookup(gfx12, UC, 0x094)->cls, AGC_REG_GUEST);
   CHECK_EQ(agc_reg_lookup(gfx12, CX, CX_REG(R_028A44_VGT_GS_ONCHIP_CNTL))->cls, AGC_REG_DROP);
   CHECK_EQ(agc_reg_lookup(gfx103, CX, CX_REG(R_028A44_VGT_GS_ONCHIP_CNTL))->cls, AGC_REG_PASS);
   for (unsigned bank = 0; bank < AGC_REG_BANK_COUNT; bank++) {
      CHECK_EQ(agc_reg_class_count(AGC_HOST_GFX103, bank, AGC_REG_SPLIT), 0);
      CHECK_EQ(agc_reg_class_count(AGC_HOST_GFX103, bank, AGC_REG_MERGE), 0);
      CHECK_EQ(agc_reg_class_count(AGC_HOST_GFX12, bank, AGC_REG_MERGE), 0);
   }

   CHECK(agc_host_select(AGC_HOST_GFX103));
   test_ps5_only_registers();
   EXPECT_NONE(UC, 0x094, 0);
   EXPECT_FATAL(convert(UC, 0x095, 2), "uc:0x095");
   EXPECT_SAME(CX, CX_REG(R_02881C_PA_CL_VS_OUT_CNTL), 0xffffffff, 0x8bffffff);
   EXPECT_SAME(CX, CX_REG(R_02881C_PA_CL_VS_OUT_CNTL), 0x06ffffff, 0x02ffffff);
   EXPECT_SAME(CX, CX_REG(R_028A48_PA_SC_MODE_CNTL_0), 0x13, 0x03);
   EXPECT_SAME(CX, CX_REG(R_028A98_VGT_DRAW_PAYLOAD_CNTL), 0x7f, 0x3a);
   EXPECT_SAME(CX, CX_REG(R_028830_PA_SU_SMALL_PRIM_FILTER_CNTL), 0x7f, 0x1f);
   EXPECT_SAME(CX, CX_REG(R_028830_PA_SU_SMALL_PRIM_FILTER_CNTL), 0x21, 0x01);
   EXPECT_SAME(CX, CX_REG(R_028000_DB_RENDER_CONTROL), 0x0000200c, 0x0000200c);
   EXPECT_SAME(CX, CX_REG(R_028A6C_VGT_GS_OUT_PRIM_TYPE), 3, 3);

   uint32_t pending = 0;
   for (uint32_t version = 0; version <= AGC_DEFAULTS_VERSION_MAX; version++) {
      for (unsigned internal = 0; internal < 2; internal++) {
         uint32_t group_count = 0;
         const struct agc_defaults_group *groups = agc_register_defaults_groups(version, internal != 0, &group_count);
         for (uint32_t g = 0; g < group_count; g++) {
            for (uint32_t r = 0; r < groups[g].count; r++) {
               uint32_t offset = groups[g].registers[r].offset;
               enum agc_reg_class cls = class_of((enum agc_reg_bank)groups[g].bank, offset);
               CHECK(cls != AGC_REG_UNDEFINED);
               if (cls != AGC_REG_PENDING)
                  continue;
               pending++;
               CHECK_EQ(groups[g].bank, CX);
               CHECK(offset >= CX_REG(R_028C70_CB_COLOR0_INFO));
               CHECK_EQ((offset - CX_REG(R_028C70_CB_COLOR0_INFO)) % 15, 0);
               CHECK((offset - CX_REG(R_028C70_CB_COLOR0_INFO)) / 15 < 8);
            }
         }
      }
   }
   CHECK(pending > 0);

   CHECK(agc_host_select(AGC_HOST_GFX12));
   test_ps5_only_registers();
   test_gfx12();
}

static void
test_depth(void)
{
   uint32_t reg = CX_REG(R_028000_DB_RENDER_CONTROL);
   uint32_t uncompressed = S_028000_STENCIL_COMPRESS_DISABLE(1) | S_028000_DEPTH_COMPRESS_DISABLE(1);
   EXPECT_SAME(CX, reg, 0, 0);
   EXPECT_SAME(CX, reg, uncompressed, uncompressed);
   EXPECT_SAME(CX, reg, S_028000_DEPTH_CLEAR_ENABLE(1) | S_028000_STENCIL_CLEAR_ENABLE(1),
               S_028000_DEPTH_CLEAR_ENABLE(1) | S_028000_STENCIL_CLEAR_ENABLE(1));
   EXPECT_SAME(CX, reg,
               S_028000_RESUMMARIZE_ENABLE(1) | S_028000_DECOMPRESS_ENABLE(1) | S_028000_COPY_CENTROID(1) |
                  S_028000_COPY_SAMPLE(5),
               S_028000_RESUMMARIZE_ENABLE(1) | S_028000_DECOMPRESS_ENABLE(1) | S_028000_COPY_CENTROID(1) |
                  S_028000_COPY_SAMPLE(5));
   EXPECT_SAME(CX, reg, S_028000_PS_INVOKE_DISABLE_GFX103(1), S_028000_PS_INVOKE_DISABLE_GFX11(1));
   EXPECT_SAME(CX, reg, 0xfffffff3, 0x00005ff3);
   EXPECT_FATAL(convert(CX, reg, S_028000_DEPTH_COPY(1)), "depth/stencil copy passes need a shader copy");
   EXPECT_FATAL(convert(CX, reg, S_028000_STENCIL_COPY(1) | S_028000_COPY_CENTROID(1)), "shader copy");

   reg = CX_REG(R_028040_DB_Z_INFO);
   uint32_t iterate = S_028040_ITERATE_256(1);
   EXPECT_SAME(CX, reg, 0x80000180, 0x8000018c | iterate);
   EXPECT_SAME(CX, reg, 0x80000000, 0x8000000c | iterate);
   uint32_t depth = S_028040_FORMAT(V_028040_Z_32_FLOAT) | S_028040_SW_MODE(24) | S_028040_MAXMIP(3) |
                    S_028040_DECOMPRESS_ON_N_ZPLANES(5) | S_028040_ALLOW_EXPCLEAR(1) | S_028040_ZRANGE_PRECISION(1);
   EXPECT_SAME(CX, reg, depth | S_028040_TILE_SURFACE_ENABLE(1), depth | S_028040_TILE_SURFACE_ENABLE(1) | iterate);
   EXPECT_SAME(CX, reg, 0xa0000183, 0xa0000183 | iterate);
   EXPECT_SAME(CX, reg, S_028040_FORMAT(V_028040_Z_16) | S_028040_NUM_SAMPLES(2) | S_028040_SW_MODE(8),
               S_028040_FORMAT(V_028040_Z_16) | S_028040_NUM_SAMPLES(2) | S_028040_SW_MODE(24) | iterate);
   EXPECT_SAME(CX, reg, S_028040_TILE_SURFACE_ENABLE(1) | S_028040_SW_MODE(9),
               S_028040_TILE_SURFACE_ENABLE(1) | S_028040_NUM_SAMPLES(3) | S_028040_SW_MODE(9) | iterate);

   reg = CX_REG(R_028044_DB_STENCIL_INFO);
   uint32_t stencil_iterate = S_028044_ITERATE_256(1);
   EXPECT_SAME(CX, reg, 0x20000180, 0x20000180 | stencil_iterate);
   EXPECT_SAME(CX, reg, 0, stencil_iterate);
   EXPECT_SAME(CX, reg, S_028044_SW_MODE(8), S_028044_SW_MODE(8) | stencil_iterate);
   EXPECT_SAME(CX, reg, S_028044_FORMAT(V_028044_STENCIL_8) | S_028044_SW_MODE(8) | S_028044_ALLOW_EXPCLEAR(1),
               S_028044_FORMAT(V_028044_STENCIL_8) | S_028044_SW_MODE(24) | S_028044_ALLOW_EXPCLEAR(1) |
                  stencil_iterate);

   EXPECT_SAME(CX, CX_REG(R_028ABC_DB_HTILE_SURFACE), S_028ABC_PIPE_ALIGNED(1),
               S_028ABC_FULL_CACHE(1) | S_028ABC_PIPE_ALIGNED(1));
   EXPECT_SAME(CX, CX_REG(R_028014_DB_HTILE_DATA_BASE), 0x02cf0100, 0x02cf0100);
   EXPECT_SAME(CX, CX_REG(R_028078_DB_HTILE_DATA_BASE_HI), 0x2, 0x2);

   reg = CX_REG(R_02880C_DB_SHADER_CONTROL);
   EXPECT_SAME(CX, reg, 0, 0);
   EXPECT_SAME(CX, reg, 0x30470, 0x10470 | S_02880C_OVERRIDE_INTRINSIC_RATE_ENABLE(1));
   EXPECT_SAME(CX, reg, S_02880C_PRIMITIVE_ORDERED_PIXEL_SHADER(1) | S_02880C_POPS_OVERLAP_NUM_SAMPLES(2),
               S_02880C_PRIMITIVE_ORDERED_PIXEL_SHADER(1) | S_02880C_OVERRIDE_INTRINSIC_RATE_ENABLE(1) |
                  S_02880C_OVERRIDE_INTRINSIC_RATE(2));
   EXPECT_SAME(CX, reg,
               0x50 | S_02880C_EXEC_IF_OVERLAPPED(1) | S_02880C_POPS_OVERLAP_NUM_SAMPLES(3) |
                  S_02880C_COVERAGE_TO_MASK_ENABLE(1),
               0x50);
   EXPECT_SAME(CX, reg, S_02880C_Z_EXPORT_ENABLE(1) | S_02880C_KILL_ENABLE(1) | S_02880C_MASK_EXPORT_ENABLE(1) |
                           S_02880C_PRE_SHADER_DEPTH_COVERAGE_ENABLE(1),
               S_02880C_Z_EXPORT_ENABLE(1) | S_02880C_KILL_ENABLE(1) | S_02880C_MASK_EXPORT_ENABLE(1) |
                  S_02880C_PRE_SHADER_DEPTH_COVERAGE_ENABLE(1));
}

static void
test_rasterizer(void)
{
   uint32_t reg = CX_REG(R_028234_PA_SU_HARDWARE_SCREEN_OFFSET);
   EXPECT_SAME(CX, reg, 0, 0);
   EXPECT_SAME(CX, reg, S_028234_HW_SCREEN_OFFSET_X(4) | S_028234_HW_SCREEN_OFFSET_Y(8),
               S_028234_HW_SCREEN_OFFSET_X(4) | S_028234_HW_SCREEN_OFFSET_Y(8));
   EXPECT_SAME(CX, reg, S_028234_HW_SCREEN_OFFSET_X(508) | S_028234_HW_SCREEN_OFFSET_Y(508),
               S_028234_HW_SCREEN_OFFSET_X(508) | S_028234_HW_SCREEN_OFFSET_Y(508));
   EXPECT_SAME(CX, reg, S_028234_HW_SCREEN_OFFSET_X(5) | S_028234_HW_SCREEN_OFFSET_Y(7),
               S_028234_HW_SCREEN_OFFSET_X(4) | S_028234_HW_SCREEN_OFFSET_Y(6));
   EXPECT_SAME(CX, reg, 0xffffffff, 0x01fe01fe);

   reg = CX_REG(R_028838_PA_CL_NGG_CNTL);
   EXPECT_SAME(CX, reg, 0, S_028838_VERTEX_REUSE_DEPTH(30));
   EXPECT_SAME(CX, reg, S_028838_VERTEX_REUSE_OFF(1) | S_028838_INDEX_BUF_EDGE_FLAG_ENA(1),
               S_028838_VERTEX_REUSE_OFF(1) | S_028838_INDEX_BUF_EDGE_FLAG_ENA(1) | S_028838_VERTEX_REUSE_DEPTH(30));
   EXPECT_SAME(CX, reg, 0xffffffff, 0x7b);

   EXPECT_NONE(CX, 0x0f0, 0);
   EXPECT_FATAL(convert(CX, 0x0f0, 1), "flexible scaled rasterization");
   EXPECT_FATAL(convert(CX, 0x0f0, 0x80000000), "has no gfx11 counterpart");
   for (uint32_t offset = 0x0f1; offset <= 0x0f2; offset++)
      EXPECT_NONE(CX, offset, 7);
   for (uint32_t offset = 0x2bb; offset <= 0x2bf; offset++)
      EXPECT_NONE(UC, offset, 0xffffffff);
   for (uint32_t offset = 0x2c4; offset <= 0x2d3; offset++)
      EXPECT_NONE(UC, offset, 0xffffffff);
   for (uint32_t offset = 0x2e4; offset <= 0x2eb; offset++)
      EXPECT_NONE(UC, offset, 0xffffffff);
   EXPECT_NONE(CX, 0x0eb, 0xff00ff00);
   EXPECT_NONE(CX, 0x0ec, 0x00010000);

   EXPECT_NONE(UC, 0x383, 0x40000040);
   EXPECT_FATAL(convert(UC, 0x383, 0), "texture gradient factors");
   EXPECT_FATAL(convert(UC, 0x383, 0x40000041), "texture gradient factors");
   EXPECT_NONE(UC, 0x384, 0x00010100);
}

static void
test_pixel_inputs(void)
{
   static const unsigned indices[] = {0, 1, 19, 20, 31};
   uint32_t attribute = S_028644_OFFSET(5) | S_028644_DEFAULT_VAL(2) | S_028644_FLAT_SHADE(1) |
                        S_028644_PT_SPRITE_TEX(1) | S_028644_DUP(1) | S_028644_FP16_INTERP_MODE(1) |
                        S_028644_USE_DEFAULT_ATTR1(1) | S_028644_DEFAULT_VAL_ATTR1(3) |
                        S_028644_PT_SPRITE_TEX_ATTR1(1) | S_028644_ATTR0_VALID(1) | S_028644_ATTR1_VALID(1);
   for (unsigned i = 0; i < COUNT_OF(indices); i++) {
      uint32_t reg = CX_REG(R_028644_SPI_PS_INPUT_CNTL_0) + indices[i];
      EXPECT_SAME(CX, reg, 0, 0);
      EXPECT_SAME(CX, reg, attribute, attribute);
      EXPECT_SAME(CX, reg, attribute | S_028644_ROTATE_PC_PTR(1) | S_028644_PRIM_ATTR(1), attribute);
      EXPECT_SAME(CX, reg, 0x20, 0x20);
      EXPECT_FATAL(convert(CX, reg, attribute | S_028644_CYL_WRAP(1)), "CYL_WRAP has no gfx11 field");
      EXPECT_FATAL(convert(CX, reg, S_028644_CYL_WRAP(8)), "emulate in the pixel shader");
   }

   uint32_t reg = CX_REG(R_0286D8_SPI_PS_IN_CONTROL);
   EXPECT_SAME(CX, reg, S_0286D8_NUM_INTERP(17) | S_0286D8_PS_W32_EN(1),
               S_0286D8_NUM_INTERP(17) | S_0286D8_PS_W32_EN(1));
   EXPECT_SAME(CX, reg,
               S_0286D8_NUM_INTERP(3) | S_0286D8_PARAM_GEN(1) | S_0286D8_OFFCHIP_PARAM_EN(1) |
                  S_0286D8_LATE_PC_DEALLOC(1) | S_0286D8_BC_OPTIMIZE_DISABLE(1),
               S_0286D8_NUM_INTERP(3));
   EXPECT_SAME(CX, reg, 0xffffffff, 0x0000803f);
}

static void
test_scratch(void)
{
   uint32_t graphics = CX_REG(R_0286E8_SPI_TMPRING_SIZE);
   uint32_t compute = SH_REG(R_00B860_COMPUTE_TMPRING_SIZE);
   agc_host_caps.shader_engines = 2;
   EXPECT_SAME(CX, graphics, 0, 0);
   EXPECT_SAME(CX, graphics, 1280 | (8 << 12), S_0286E8_WAVES(640) | S_0286E8_WAVESIZE(32));
   EXPECT_SAME(CX, graphics, 5 | (1 << 12), S_0286E8_WAVES(3) | S_0286E8_WAVESIZE(4));
   EXPECT_SAME(CX, graphics, 0xfff | (0x1fff << 12), S_0286E8_WAVES(2048) | S_0286E8_WAVESIZE(0x7ffc));
   EXPECT_SAME(CX, graphics, 0xffffffff, S_0286E8_WAVES(2048) | S_0286E8_WAVESIZE(0x7ffc));
   EXPECT_SAME(SH, compute, 1280 | (8 << 12), S_0286E8_WAVES(640) | S_0286E8_WAVESIZE(32));
   agc_host_caps.shader_engines = 1;
   EXPECT_SAME(CX, graphics, 1280 | (8 << 12), S_0286E8_WAVES(1280) | S_0286E8_WAVESIZE(32));
   EXPECT_SAME(SH, compute, 5 | (1 << 12), S_0286E8_WAVES(5) | S_0286E8_WAVESIZE(4));
   agc_host_caps.shader_engines = 0;
   EXPECT_SAME(CX, graphics, 3 | (2 << 12), S_0286E8_WAVES(3) | S_0286E8_WAVESIZE(8));
   agc_host_caps.shader_engines = 1;
}

static void
test_blend(void)
{
   uint32_t disabled = S_028760_COLOR_COMB_FCN(V_028760_OPT_COMB_BLEND_DISABLED) |
                       S_028760_ALPHA_COMB_FCN(V_028760_OPT_COMB_BLEND_DISABLED);
   CHECK_EQ(disabled, 0x06000600);
   for (unsigned mrt = 0; mrt < 8; mrt++)
      EXPECT_BLEND(mrt, 0x20010001, 0x20010001, disabled);

   uint32_t alpha = S_028780_ENABLE(1) | S_028780_COLOR_SRCBLEND(V_028780_BLEND_SRC_ALPHA) |
                    S_028780_COLOR_DESTBLEND(V_028780_BLEND_ONE_MINUS_SRC_ALPHA) |
                    S_028780_COLOR_COMB_FCN(V_028780_COMB_DST_PLUS_SRC);
   uint32_t alpha_opt = S_028760_COLOR_SRC_OPT(V_028760_BLEND_OPT_PRESERVE_A1_IGNORE_A0) |
                        S_028760_COLOR_DST_OPT(V_028760_BLEND_OPT_PRESERVE_A0_IGNORE_A1) |
                        S_028760_COLOR_COMB_FCN(V_028760_OPT_COMB_ADD) |
                        S_028760_ALPHA_SRC_OPT(V_028760_BLEND_OPT_PRESERVE_A1_IGNORE_A0) |
                        S_028760_ALPHA_DST_OPT(V_028760_BLEND_OPT_PRESERVE_A0_IGNORE_A1) |
                        S_028760_ALPHA_COMB_FCN(V_028760_OPT_COMB_ADD);
   CHECK_EQ(alpha_opt, 0x01540154);
   EXPECT_BLEND(0, alpha, alpha, alpha_opt);
   EXPECT_BLEND(7, alpha, alpha, alpha_opt);

   uint32_t constant = S_028780_ENABLE(1) | S_028780_SEPARATE_ALPHA_BLEND(1) |
                       S_028780_COLOR_COMB_FCN(V_028780_COMB_SRC_MINUS_DST) |
                       S_028780_ALPHA_COMB_FCN(V_028780_COMB_DST_MINUS_SRC);
   uint32_t none = V_028760_BLEND_OPT_PRESERVE_NONE_IGNORE_NONE;
   uint32_t constant_opt = S_028760_COLOR_SRC_OPT(none) | S_028760_COLOR_DST_OPT(none) |
                           S_028760_COLOR_COMB_FCN(V_028760_OPT_COMB_SUBTRACT) | S_028760_ALPHA_SRC_OPT(none) |
                           S_028760_ALPHA_DST_OPT(none) | S_028760_ALPHA_COMB_FCN(V_028760_OPT_COMB_REVSUBTRACT);
   EXPECT_BLEND(1,
                constant | S_028780_COLOR_SRCBLEND(V_028780_BLEND_CONSTANT_COLOR_GFX6) |
                   S_028780_COLOR_DESTBLEND(V_028780_BLEND_ONE_MINUS_CONSTANT_COLOR_GFX6) |
                   S_028780_ALPHA_SRCBLEND(V_028780_BLEND_CONSTANT_ALPHA_GFX6) |
                   S_028780_ALPHA_DESTBLEND(V_028780_BLEND_ONE_MINUS_CONSTANT_ALPHA_GFX6),
                constant | S_028780_COLOR_SRCBLEND(V_028780_BLEND_CONSTANT_COLOR_GFX11) |
                   S_028780_COLOR_DESTBLEND(V_028780_BLEND_ONE_MINUS_CONSTANT_COLOR_GFX11) |
                   S_028780_ALPHA_SRCBLEND(V_028780_BLEND_CONSTANT_ALPHA_GFX11) |
                   S_028780_ALPHA_DESTBLEND(V_028780_BLEND_ONE_MINUS_CONSTANT_ALPHA_GFX11),
                constant_opt);
   EXPECT_BLEND(2,
                constant | S_028780_COLOR_SRCBLEND(V_028780_BLEND_SRC1_COLOR_GFX6) |
                   S_028780_COLOR_DESTBLEND(V_028780_BLEND_INV_SRC1_COLOR_GFX6) |
                   S_028780_ALPHA_SRCBLEND(V_028780_BLEND_SRC1_ALPHA_GFX6) |
                   S_028780_ALPHA_DESTBLEND(V_028780_BLEND_INV_SRC1_ALPHA_GFX6),
                constant | S_028780_COLOR_SRCBLEND(V_028780_BLEND_SRC1_COLOR_GFX11) |
                   S_028780_COLOR_DESTBLEND(V_028780_BLEND_INV_SRC1_COLOR_GFX11) |
                   S_028780_ALPHA_SRCBLEND(V_028780_BLEND_SRC1_ALPHA_GFX11) |
                   S_028780_ALPHA_DESTBLEND(V_028780_BLEND_INV_SRC1_ALPHA_GFX11),
                constant_opt);

   EXPECT_BLEND(3, S_028780_ENABLE(1) | S_028780_COLOR_SRCBLEND(V_028780_BLEND_BOTH_SRC_ALPHA), alpha, alpha_opt);
   EXPECT_BLEND(3,
                S_028780_ENABLE(1) | S_028780_COLOR_SRCBLEND(V_028780_BLEND_BOTH_INV_SRC_ALPHA) |
                   S_028780_COLOR_DESTBLEND(V_028780_BLEND_ONE),
                S_028780_ENABLE(1) | S_028780_COLOR_SRCBLEND(V_028780_BLEND_ONE_MINUS_SRC_ALPHA) |
                   S_028780_COLOR_DESTBLEND(V_028780_BLEND_SRC_ALPHA),
                S_028760_COLOR_SRC_OPT(V_028760_BLEND_OPT_PRESERVE_A0_IGNORE_A1) |
                   S_028760_COLOR_DST_OPT(V_028760_BLEND_OPT_PRESERVE_A1_IGNORE_A0) |
                   S_028760_COLOR_COMB_FCN(V_028760_OPT_COMB_ADD) |
                   S_028760_ALPHA_SRC_OPT(V_028760_BLEND_OPT_PRESERVE_A0_IGNORE_A1) |
                   S_028760_ALPHA_DST_OPT(V_028760_BLEND_OPT_PRESERVE_A1_IGNORE_A0) |
                   S_028760_ALPHA_COMB_FCN(V_028760_OPT_COMB_ADD));
   EXPECT_BLEND(4,
                S_028780_ENABLE(1) | S_028780_COLOR_SRCBLEND(V_028780_BLEND_ONE) |
                   S_028780_COLOR_DESTBLEND(V_028780_BLEND_BOTH_SRC_ALPHA),
                S_028780_ENABLE(1) | S_028780_COLOR_SRCBLEND(V_028780_BLEND_ONE) |
                   S_028780_COLOR_DESTBLEND(V_028780_BLEND_SRC_ALPHA),
                S_028760_COLOR_SRC_OPT(V_028760_BLEND_OPT_PRESERVE_ALL_IGNORE_NONE) |
                   S_028760_COLOR_DST_OPT(V_028760_BLEND_OPT_PRESERVE_A1_IGNORE_A0) |
                   S_028760_COLOR_COMB_FCN(V_028760_OPT_COMB_ADD) |
                   S_028760_ALPHA_SRC_OPT(V_028760_BLEND_OPT_PRESERVE_ALL_IGNORE_NONE) |
                   S_028760_ALPHA_DST_OPT(V_028760_BLEND_OPT_PRESERVE_A1_IGNORE_A0) |
                   S_028760_ALPHA_COMB_FCN(V_028760_OPT_COMB_ADD));

   uint32_t minimum = S_028780_ENABLE(1) | S_028780_COLOR_COMB_FCN(V_028780_COMB_MIN_DST_SRC);
   EXPECT_BLEND(5, minimum, minimum, 0x03110311);
   uint32_t maximum = S_028780_ENABLE(1) | S_028780_COLOR_COMB_FCN(V_028780_COMB_MAX_DST_SRC) |
                      S_028780_COLOR_SRCBLEND(V_028780_BLEND_DST_COLOR);
   EXPECT_BLEND(5, maximum, maximum, 0x04110411);
   uint32_t saturate = S_028780_ENABLE(1) | S_028780_COLOR_SRCBLEND(V_028780_BLEND_SRC_ALPHA_SATURATE);
   EXPECT_BLEND(6, saturate, saturate, 0x01710166);
   uint32_t modulate = S_028780_ENABLE(1) | S_028780_COLOR_SRCBLEND(V_028780_BLEND_DST_COLOR) |
                       S_028780_COLOR_DESTBLEND(V_028780_BLEND_ZERO);
   EXPECT_BLEND(6, modulate, modulate, 0x01770177);

   uint32_t reg = CX_REG(R_028780_CB_BLEND0_CONTROL);
   EXPECT_FATAL(convert(CX, reg, S_028780_COLOR_SRCBLEND(21)), "blend factor is out of range");
   EXPECT_FATAL(convert(CX, reg + 7, S_028780_ALPHA_DESTBLEND(31)), "blend factor is out of range");
   for (unsigned mrt = 0; mrt < 8; mrt++)
      EXPECT_NONE(CX, CX_REG(R_028760_SX_MRT0_BLEND_OPT) + mrt, 0x01540154);
   EXPECT_NONE(CX, CX_REG(R_028754_SX_PS_DOWNCONVERT), 0x55555555);
   EXPECT_NONE(CX, CX_REG(R_028758_SX_BLEND_OPT_EPSILON), 0x11111111);
   EXPECT_NONE(CX, CX_REG(R_02875C_SX_BLEND_OPT_CONTROL), 0x11111111);

   reg = CX_REG(R_028808_CB_COLOR_CONTROL);
   EXPECT_SAME(CX, reg, 0x00cc0010, 0x00cc0010);
   EXPECT_SAME(CX, reg, S_028808_ROP3(V_028808_ROP3_COPY), S_028808_ROP3(V_028808_ROP3_COPY));
   static const uint32_t passes[] = {V_028808_CB_ELIMINATE_FAST_CLEAR, V_028808_CB_DECOMPRESS,
                                     V_028808_CB_FMASK_DECOMPRESS, V_028808_CB_DCC_DECOMPRESS_GFX8};
   for (unsigned i = 0; i < COUNT_OF(passes); i++) {
      EXPECT_SAME(CX, reg, S_028808_MODE(passes[i]) | S_028808_ROP3(V_028808_ROP3_COPY),
                  S_028808_MODE(V_028808_CB_DISABLE) | S_028808_ROP3(V_028808_ROP3_COPY));
   }
   EXPECT_SAME(CX, reg,
               S_028808_DISABLE_DUAL_QUAD(1) | S_028808_ENABLE_1FRAG_PS_INVOKE(1) | S_028808_DEGAMMA_ENABLE(1) |
                  S_028808_MODE(V_028808_CB_NORMAL) | S_028808_ROP3(V_028808_ROP3_NO_OP) | 0xff000000,
               S_028808_DISABLE_DUAL_QUAD(1) | S_028808_DEGAMMA_ENABLE(1) | S_028808_MODE(V_028808_CB_NORMAL) |
                  S_028808_ROP3(V_028808_ROP3_NO_OP));
   EXPECT_FATAL(convert(CX, reg, S_028808_MODE(V_028808_CB_RESOLVE)), "RESOLVE has no gfx11 mode");
   EXPECT_FATAL(convert(CX, reg, S_028808_MODE(V_028808_CB_RESERVED_GFX10)), "reserved mode");
}

static struct converted
convert_bank_state(enum agc_reg_bank bank, const struct agc_register *guest, uint32_t count,
                   struct agc_reg_shadow *shadow, struct converted *uc)
{
   struct converted out;
   uc->count = 0;
   out.count = agc_reg_convert_pairs_state(bank, guest, count, out.reg, shadow, uc->reg, AGC_REGISTERS_UC_MAX,
                                           &uc->count, "test");
   return out;
}

static struct converted
convert_state(const struct agc_register *guest, uint32_t count, struct agc_reg_shadow *shadow)
{
   struct converted uc;
   return convert_bank_state(CX, guest, count, shadow, &uc);
}

static void
test_colour_target_state(enum agc_host host)
{
   const uint32_t slot = 5;
   const uint32_t info = CX_REG(R_028C70_CB_COLOR0_INFO) + slot * 15;
   const uint32_t attrib = CX_REG(R_028C74_CB_COLOR0_ATTRIB) + slot * 15;
   const uint32_t control = CX_REG(R_028C78_CB_COLOR0_DCC_CONTROL) + slot * 15;
   const uint32_t fdcc = CX_REG(R_028C78_CB_COLOR0_FDCC_CONTROL) + slot * 15;
   struct agc_reg_shadow shadow;
   agc_reg_shadow_reset(&shadow);

   const struct agc_register target[] = {
      {attrib, S_028C74_NUM_SAMPLES(2) | S_028C74_NUM_FRAGMENTS_GFX6(2)},
      {control, 0x48},
   };
   convert_state(target, COUNT_OF(target), &shadow);
   const struct agc_register enable[] = {{info, S_028C70_FORMAT_GFX6(V_028C70_COLOR_8_8_8_8) | S_028C70_DCC_ENABLE(1)}};
   struct converted out = convert_state(enable, COUNT_OF(enable), &shadow);
   uint32_t value = 0;
   CHECK(find_write(&out, fdcc, &value));
   CHECK(G_028C78_FDCC_ENABLE(value) == 1);
   if (host == AGC_HOST_GFX115)
      CHECK(G_028C78_MAX_COMP_FRAGS(value) == 1);

   const struct agc_register single[] = {{attrib, 0}};
   out = convert_state(single, COUNT_OF(single), &shadow);
   CHECK(find_write(&out, fdcc, &value));
   CHECK(G_028C78_FDCC_ENABLE(value) == 1);
   CHECK(G_028C78_MAX_COMP_FRAGS(value) == 0);

   const struct agc_register disable[] = {{info, S_028C70_FORMAT_GFX6(V_028C70_COLOR_8_8_8_8)}};
   convert_state(disable, COUNT_OF(disable), &shadow);
   out = convert_state(single, COUNT_OF(single), &shadow);
   CHECK(!find_write(&out, fdcc, &value));
}

static void
test_colour_targets(void)
{
   for (unsigned slot = 0; slot < 8; slot++) {
      uint32_t info = CX_REG(R_028C70_CB_COLOR0_INFO) + slot * 15;
      uint32_t common = S_028C70_LINEAR_GENERAL(1) | S_028C70_NUMBER_TYPE(V_028C70_NUMBER_SRGB) |
                        S_028C70_COMP_SWAP(V_028C70_SWAP_ALT) | S_028C70_BLEND_CLAMP(1) | S_028C70_BLEND_BYPASS(1) |
                        S_028C70_SIMPLE_FLOAT(1) | S_028C70_ROUND_MODE(1) | S_028C70_BLEND_OPT_DONT_RD_DST(3) |
                        S_028C70_BLEND_OPT_DISCARD_PIXEL(5);
      uint32_t compression = S_028C70_ENDIAN(2) | S_028C70_FAST_CLEAR(1) | S_028C70_COMPRESSION(1) |
                             S_028C70_CMASK_IS_LINEAR(1) | S_028C70_FMASK_COMPRESSION_DISABLE(1) |
                             S_028C70_FMASK_COMPRESS_1FRAG_ONLY(1) | S_028C70_CMASK_ADDR_TYPE(2) |
                             S_028C70_ALT_TILE_MODE(1);
      EXPECT_INFO(info, 0, 0);
      EXPECT_INFO(info, S_028C70_FORMAT_GFX6(V_028C70_COLOR_8_8_8_8) | common | compression,
                  S_028C70_FORMAT_GFX11(V_028C70_COLOR_8_8_8_8) | common);
      EXPECT_INFO(info,
                  S_028C70_FORMAT_GFX6(V_028C70_COLOR_16_16_16_16) | S_028C70_NUMBER_TYPE(V_028C70_NUMBER_FLOAT),
                  S_028C70_FORMAT_GFX11(V_028C70_COLOR_16_16_16_16) | S_028C70_NUMBER_TYPE(V_028C70_NUMBER_FLOAT));
      EXPECT_INFO(info, S_028C70_FORMAT_GFX6(30), S_028C70_FORMAT_GFX11(30));
      EXPECT_FATAL(convert(CX, info, S_028C70_FORMAT_GFX6(V_028C70_COLOR_8_8_8_8) | S_028C70_DCC_ENABLE(1)),
                   "nor the context state before it sets CB_COLOR");
      EXPECT_FATAL(convert(CX, info, S_028C70_FORMAT_GFX6(V_028C70_COLOR_2_10_10_10_6E4)),
                   "format 2_10_10_10_6E4 has no gfx11 encoding");

      uint32_t attrib = CX_REG(R_028C74_CB_COLOR0_ATTRIB) + slot * 15;
      EXPECT_SAME(CX, attrib, 0, 0);
      EXPECT_SAME(CX, attrib,
                  S_028C74_TILE_MODE_INDEX(31) | S_028C74_FMASK_TILE_MODE_INDEX(31) | S_028C74_FMASK_BANK_HEIGHT(3) |
                     S_028C74_NUM_SAMPLES(2) | S_028C74_NUM_FRAGMENTS_GFX6(2) | S_028C74_FORCE_DST_ALPHA_1_GFX6(1) |
                     S_028C74_DISABLE_FMASK_NOFETCH_OPT(1) | S_028C74_LIMIT_COLOR_FETCH_TO_256B_MAX_GFX10(1),
                  S_028C74_NUM_FRAGMENTS_GFX11(2) | S_028C74_FORCE_DST_ALPHA_1_GFX11(1) |
                     S_028C74_LIMIT_COLOR_FETCH_TO_256B_MAX_GFX11(1));
      EXPECT_SAME(CX, attrib, S_028C74_NUM_SAMPLES(3) | S_028C74_NUM_FRAGMENTS_GFX6(3),
                  S_028C74_NUM_FRAGMENTS_GFX11(3));
      EXPECT_SAME(CX, attrib, S_028C74_FORCE_DST_ALPHA_1_GFX6(1), S_028C74_FORCE_DST_ALPHA_1_GFX11(1));
      EXPECT_FATAL(convert(CX, attrib, S_028C74_NUM_SAMPLES(2) | S_028C74_NUM_FRAGMENTS_GFX6(1)),
                   "more samples than fragments");
      EXPECT_FATAL(convert(CX, attrib, S_028C74_NUM_SAMPLES(4) | S_028C74_NUM_FRAGMENTS_GFX6(3)), "FMASK");

      uint32_t control = CX_REG(R_028C78_CB_COLOR0_DCC_CONTROL) + slot * 15;
      EXPECT_FATAL(convert(CX, control, 0x48), "needs CB_COLOR");
      const uint32_t every_field =
         S_028C78_OVERWRITE_COMBINER_DISABLE(1) | S_028C78_KEY_CLEAR_ENABLE(1) |
         S_028C78_MAX_UNCOMPRESSED_BLOCK_SIZE(2) | S_028C78_MIN_COMPRESSED_BLOCK_SIZE(1) |
         S_028C78_MAX_COMPRESSED_BLOCK_SIZE(3) | S_028C78_COLOR_TRANSFORM(2) | S_028C78_INDEPENDENT_64B_BLOCKS(1) |
         S_028C78_LOSSY_RGB_PRECISION(15) | S_028C78_LOSSY_ALPHA_PRECISION(15) |
         S_028C78_DISABLE_CONSTANT_ENCODE_REG(1) | S_028C78_ENABLE_CONSTANT_ENCODE_REG_WRITE(1) |
         S_028C78_INDEPENDENT_128B_BLOCKS_GFX10(1);
      const uint32_t mapped = S_028C78_MAX_UNCOMPRESSED_BLOCK_SIZE(2) | S_028C78_MIN_COMPRESSED_BLOCK_SIZE(1) |
                              S_028C78_MAX_COMPRESSED_BLOCK_SIZE(3) | S_028C78_COLOR_TRANSFORM(2) |
                              S_028C78_INDEPENDENT_64B_BLOCKS(1) | S_028C78_DISABLE_CONSTANT_ENCODE_REG(1) |
                              S_028C78_INDEPENDENT_128B_BLOCKS_GFX11(1);
      struct agc_register plain[2] = {
         {control, every_field},
         {info, S_028C70_FORMAT_GFX6(V_028C70_COLOR_8_8_8_8)},
      };
      struct converted out = convert_array(CX, plain, 2);
      CHECK_EQ(out.count, 2);
      EXPECT_WRITE(out, control, mapped | S_028C78_ENABLE_CONSTANT_ENCODE_REG_WRITE(1));
      plain[0].value = 0xffffffff;
      out = convert_array(CX, plain, 2);
      CHECK_EQ(out.count, 2);
      EXPECT_WRITE(out, control, 0x000c07fc);

      uint32_t base = CX_REG(R_028C60_CB_COLOR0_BASE) + slot * 15;
      uint32_t base_ext = CX_REG(R_028E40_CB_COLOR0_BASE_EXT) + slot;
      uint32_t colour_attrib3 = CX_REG(R_028EE0_CB_COLOR0_ATTRIB3) + slot;
      const bool fragment_override = agc_host_current() == AGC_HOST_GFX115;
      const uint32_t enabled = S_028C78_DISABLE_CONSTANT_ENCODE_REG(1) | S_028C78_FDCC_ENABLE(1) |
                               S_028C78_ENABLE_MAX_COMP_FRAG_OVERRIDE(fragment_override);
      static const struct {
         uint32_t samples;
         uint32_t fragments;
         uint32_t comp_frags;
      } sampling[] = {{0, 0, 0}, {1, 1, 0}, {2, 2, 1}, {3, 3, 1}};
      for (unsigned s = 0; s < COUNT_OF(sampling); s++) {
         struct agc_register target[6] = {
            {info, S_028C70_FORMAT_GFX6(V_028C70_COLOR_8_8_8_8) | S_028C70_DCC_ENABLE(1)},
            {attrib, S_028C74_NUM_SAMPLES(sampling[s].samples) | S_028C74_NUM_FRAGMENTS_GFX6(sampling[s].fragments)},
            {control, every_field},
            {base, 0x00654300 + slot},
            {base_ext, 0x02},
            {colour_attrib3, S_028EE0_COLOR_SW_MODE(27) | S_028EE0_RESOURCE_TYPE(1)},
         };
         out = convert_array(CX, target, COUNT_OF(target));
         CHECK_EQ(out.count, 6);
         const uint32_t expected =
            mapped | enabled | S_028C78_MAX_COMP_FRAGS(fragment_override ? sampling[s].comp_frags : 0);
         EXPECT_WRITE(out, control, expected);
         uint32_t recorded = 0;
         uint64_t recorded_meta = 1;
         CHECK(agc_surface_dcc_target(UINT64_C(0x20065430000), &recorded_meta, &recorded));
         CHECK_EQ(recorded, expected);
         CHECK_EQ(recorded_meta, 0);
      }
      struct agc_register no_attrib[2] = {
         {info, S_028C70_FORMAT_GFX6(V_028C70_COLOR_8_8_8_8) | S_028C70_DCC_ENABLE(1)},
         {control, 0x00080208},
      };
      EXPECT_FATAL(convert_array(CX, no_attrib, 2), "MAX_COMP_FRAGS");

      uint32_t dcc_base = CX_REG(R_028C94_CB_COLOR0_DCC_BASE) + slot * 15;
      uint32_t dcc_base_ext = CX_REG(R_028EA0_CB_COLOR0_DCC_BASE_EXT) + slot;
      EXPECT_SAME(CX, dcc_base, 0x02426980, 0x02426980);
      EXPECT_SAME(CX, dcc_base_ext, 0x70, 0x70);
      EXPECT_FATAL(convert(CX, dcc_base, 0x02426985), "ATTRIB3");
      static const struct {
         uint8_t mode;
         uint32_t value;
         uint32_t host;
      } metadata[] = {
         {27, 0x0242698f, 0x02426980}, {24, 0x0242698f, 0x02426980}, {21, 0x0242698f, 0x02426980},
         {22, 0x02426985, 0x02426980}, {9, 0x0242698f, 0x0242698f},  {0, 0x0242698f, 0x0242698f},
         {27, 0x024269f0, 0x024269f0},
      };
      for (unsigned i = 0; i < COUNT_OF(metadata); i++) {
         struct agc_register guest[2] = {
            {dcc_base, metadata[i].value},
            {colour_attrib3, S_028EE0_COLOR_SW_MODE(metadata[i].mode) | S_028EE0_RESOURCE_TYPE(1)},
         };
         out = convert_array(CX, guest, 2);
         CHECK_EQ(out.count, 2);
         EXPECT_WRITE(out, dcc_base, metadata[i].host);
      }

      uint32_t attrib3 = CX_REG(R_028EE0_CB_COLOR0_ATTRIB3) + slot;
      EXPECT_SAME(CX, attrib3, 0x08c6c000, S_028EE0_COLOR_SW_MODE(27));
      uint32_t kept = S_028EE0_MIP0_DEPTH(0x1abc) | S_028EE0_META_LINEAR(1) | S_028EE0_DCC_PIPE_ALIGNED(1);
      uint32_t gone = S_028EE0_FMASK_SW_MODE(24) | S_028EE0_CMASK_PIPE_ALIGNED(1) | S_028EE0_RESOURCE_LEVEL(1) |
                      S_028EE0_VRS_RATE_HINT_ENABLE(1);
      static const uint8_t thin[][2] = {{0, 0},   {1, 2},   {2, 2},   {5, 6},   {6, 6},   {9, 10},  {10, 10},
                                        {17, 18}, {18, 18}, {21, 22}, {22, 22}, {24, 24}, {25, 26}, {26, 26},
                                        {27, 27}, {4, 4},   {8, 8},   {31, 31}};
      for (unsigned i = 0; i < COUNT_OF(thin); i++) {
         for (unsigned type = 0; type < 2; type++) {
            EXPECT_SAME(CX, attrib3, kept | gone | S_028EE0_RESOURCE_TYPE(type) | S_028EE0_COLOR_SW_MODE(thin[i][0]),
                        kept | S_028EE0_RESOURCE_TYPE(type) | S_028EE0_COLOR_SW_MODE(thin[i][1]));
         }
      }
      static const uint8_t volume[][2] = {{0, 0},   {1, 9},   {2, 9},   {5, 5},   {6, 9},   {9, 9},   {10, 9},
                                          {13, 9},  {17, 17}, {18, 9},  {21, 21}, {22, 9},  {24, 24}, {25, 25},
                                          {26, 26}, {27, 27}, {28, 28}, {29, 29}, {30, 9},  {31, 31}};
      for (unsigned i = 0; i < COUNT_OF(volume); i++) {
         EXPECT_SAME(CX, attrib3, kept | gone | S_028EE0_RESOURCE_TYPE(2) | S_028EE0_COLOR_SW_MODE(volume[i][0]),
                     kept | S_028EE0_RESOURCE_TYPE(2) | S_028EE0_COLOR_SW_MODE(volume[i][1]));
      }
   }
}

static void
test_colour_bases(void)
{
   static const struct {
      uint8_t mode;
      uint32_t base;
      uint32_t ext;
      uint32_t host_base;
      uint32_t host_ext;
   } vectors[] = {
      {0, 0x00123456, 0x00, 0x00123456, 0x00}, {0, 0x00000000, 0x00, 0x00000000, 0x00},
      {0, 0x001234ff, 0x02, 0x001234ff, 0x02}, {27, 0x00123456, 0x00, 0x00123400, 0x01},
      {27, 0x00123456, 0x02, 0x00123400, 0x03}, {27, 0x00000000, 0x00, 0x00000000, 0x00},
      {27, 0x000000ff, 0x00, 0x00000000, 0x00}, {27, 0x00000000, 0x04, 0x00000000, 0x05},
      {27, 0x00000100, 0x00, 0x00000100, 0x01}, {24, 0xffffffff, 0xff, 0xffffff00, 0x00},
      {25, 0x00123456, 0x00, 0x00123400, 0x01}, {26, 0x00123456, 0x00, 0x00123400, 0x01},
      {17, 0x00123456, 0x00, 0x00123400, 0x01}, {18, 0x00123456, 0x00, 0x00123400, 0x01},
      {21, 0x00123456, 0x00, 0x00123450, 0x01}, {22, 0x0012345f, 0x00, 0x00123450, 0x01},
      {21, 0x0000000f, 0x00, 0x00000000, 0x00}, {1, 0x00123456, 0x00, 0x00123456, 0x01},
      {5, 0x00123456, 0x00, 0x00123456, 0x01},  {9, 0x00123456, 0x00, 0x00123456, 0x01},
      {10, 0x00123456, 0x01, 0x00123456, 0x02}, {9, 0x00000000, 0x00, 0x00000000, 0x00},
      {16, 0x00123456, 0x00, 0x00123456, 0x01}, {28, 0x00123456, 0x00, 0x00123456, 0x01},
   };
   for (unsigned slot = 0; slot < 8; slot++) {
      uint32_t base = CX_REG(R_028C60_CB_COLOR0_BASE) + slot * 15;
      uint32_t ext = CX_REG(R_028E40_CB_COLOR0_BASE_EXT) + slot;
      uint32_t attrib3 = CX_REG(R_028EE0_CB_COLOR0_ATTRIB3) + slot;
      for (unsigned i = 0; i < COUNT_OF(vectors); i++) {
         struct agc_register guest[3] = {
            {base, vectors[i].base},
            {ext, vectors[i].ext},
            {attrib3, S_028EE0_COLOR_SW_MODE(vectors[i].mode) | S_028EE0_RESOURCE_TYPE(1)},
         };
         struct converted out = convert_array(CX, guest, 3);
         CHECK_EQ(out.count, 3);
         EXPECT_WRITE(out, base, vectors[i].host_base);
         EXPECT_WRITE(out, ext, vectors[i].host_ext);
      }

      struct agc_register reordered[4] = {
         {attrib3, S_028EE0_COLOR_SW_MODE(0)},
         {ext, 0},
         {base, 0x00abcdef},
         {attrib3, S_028EE0_COLOR_SW_MODE(27)},
      };
      struct converted out = convert_array(CX, reordered, 4);
      CHECK_EQ(out.count, 4);
      EXPECT_WRITE(out, base, 0x00abcd00);
      EXPECT_WRITE(out, ext, 1);

      struct agc_register other = {CX_REG(R_028EE0_CB_COLOR0_ATTRIB3) + ((slot + 1) & 7), S_028EE0_COLOR_SW_MODE(27)};
      struct agc_register alone[2] = {{base, 0x00123456}, other};
      EXPECT_FATAL(convert_array(CX, alone, 2), "the companion is missing");
      CHECK(strstr(fatal_text, "_ATTRIB3") != NULL);
      alone[0] = (struct agc_register){ext, 0};
      EXPECT_FATAL(convert_array(CX, alone, 2), "the companion is missing");
      EXPECT_SAME(CX, base, 0, 0);
      EXPECT_SAME(CX, base, 0x00123400, 0x00123400);
      struct agc_register null_pair[2] = {{base, 0}, {ext, 0}};
      out = convert_array(CX, null_pair, 2);
      CHECK_EQ(out.count, 2);
      EXPECT_WRITE(out, base, 0);
      EXPECT_WRITE(out, ext, 0);
      struct agc_register linear[2] = {{ext, 5}, {attrib3, S_028EE0_COLOR_SW_MODE(0)}};
      out = convert_array(CX, linear, 2);
      EXPECT_WRITE(out, ext, 5);
      struct agc_register unknown_mode[2] = {{ext, 0}, {base, 0x00123400}};
      EXPECT_FATAL(convert_array(CX, unknown_mode, 2), "_ATTRIB3 in the same register array");
      unknown_mode[0].value = 7;
      EXPECT_FATAL(convert_array(CX, unknown_mode, 2), "_ATTRIB3 in the same register array");
      EXPECT_FATAL(convert(CX, ext, 0), "the companion is missing");

      struct agc_register without_base[2] = {{ext, 0}, {attrib3, S_028EE0_COLOR_SW_MODE(27)}};
      EXPECT_FATAL(convert_array(CX, without_base, 2), "a null base stays null");
      without_base[0].value = 3;
      out = convert_array(CX, without_base, 2);
      EXPECT_WRITE(out, ext, 4);
      without_base[1].value = S_028EE0_COLOR_SW_MODE(0);
      without_base[0].value = 0;
      out = convert_array(CX, without_base, 2);
      EXPECT_WRITE(out, ext, 0);

      uint32_t host_offset;
      uint32_t host_value;
      EXPECT_FATAL(agc_reg_convert(CX, base, 0x1234, &host_offset, &host_value, "test"), "the companion is missing");
   }
}

static void
test_depth_bases(void)
{
   static const uint32_t low[4] = {CX_REG(R_028048_DB_Z_READ_BASE), CX_REG(R_02804C_DB_STENCIL_READ_BASE),
                                   CX_REG(R_028050_DB_Z_WRITE_BASE), CX_REG(R_028054_DB_STENCIL_WRITE_BASE)};
   static const uint32_t high[4] = {CX_REG(R_028068_DB_Z_READ_BASE_HI), CX_REG(R_02806C_DB_STENCIL_READ_BASE_HI),
                                    CX_REG(R_028070_DB_Z_WRITE_BASE_HI), CX_REG(R_028074_DB_STENCIL_WRITE_BASE_HI)};
   uint32_t z_info = CX_REG(R_028040_DB_Z_INFO);
   uint32_t stencil_info = CX_REG(R_028044_DB_STENCIL_INFO);

   struct agc_register bound[10] = {
      {z_info, S_028040_FORMAT(V_028040_Z_32_FLOAT) | S_028040_SW_MODE(24)},
      {stencil_info, S_028044_FORMAT(V_028044_STENCIL_8) | S_028044_SW_MODE(24)},
      {low[0], 0x00aabb11}, {low[1], 0x00ccdd22}, {low[2], 0x00aabb33}, {low[3], 0x00ccdd44},
      {high[0], 0x00},      {high[1], 0x01},      {high[2], 0x00},      {high[3], 0x01},
   };
   struct converted out = convert_array(CX, bound, 10);
   CHECK_EQ(out.count, 10);
   EXPECT_WRITE(out, z_info, S_028040_FORMAT(V_028040_Z_32_FLOAT) | S_028040_SW_MODE(24) | S_028040_ITERATE_256(1));
   EXPECT_WRITE(out, low[0], 0x00aabb00);
   EXPECT_WRITE(out, low[1], 0x00ccdd00);
   EXPECT_WRITE(out, low[2], 0x00aabb00);
   EXPECT_WRITE(out, low[3], 0x00ccdd00);
   EXPECT_WRITE(out, high[0], 0x01);
   EXPECT_WRITE(out, high[1], 0x02);
   EXPECT_WRITE(out, high[2], 0x01);
   EXPECT_WRITE(out, high[3], 0x02);

   struct agc_register mixed[10] = {
      {z_info, S_028040_FORMAT(V_028040_Z_16) | S_028040_SW_MODE(0)},
      {stencil_info, S_028044_FORMAT(V_028044_STENCIL_8) | S_028044_SW_MODE(8)},
      {low[0], 0x00aabb11}, {low[1], 0x00ccdd22}, {low[2], 0x00aabb33}, {low[3], 0x00ccdd44},
      {high[0], 0x07},      {high[1], 0x00},      {high[2], 0x07},      {high[3], 0x00},
   };
   out = convert_array(CX, mixed, 10);
   CHECK_EQ(out.count, 10);
   EXPECT_WRITE(out, low[0], 0x00aabb11);
   EXPECT_WRITE(out, low[2], 0x00aabb33);
   EXPECT_WRITE(out, high[0], 0x07);
   EXPECT_WRITE(out, high[2], 0x07);
   EXPECT_WRITE(out, low[1], 0x00ccdd22);
   EXPECT_WRITE(out, low[3], 0x00ccdd44);
   EXPECT_WRITE(out, high[1], 0x01);
   EXPECT_WRITE(out, high[3], 0x01);

   struct agc_register reset[10] = {
      {z_info, 0x80000180}, {stencil_info, 0x20000180}, {low[0], 0},  {low[1], 0},  {low[2], 0},
      {low[3], 0},          {high[0], 0},               {high[1], 0}, {high[2], 0}, {high[3], 0},
   };
   out = convert_array(CX, reset, 10);
   CHECK_EQ(out.count, 10);
   for (unsigned plane = 0; plane < 4; plane++) {
      EXPECT_WRITE(out, low[plane], 0);
      EXPECT_WRITE(out, high[plane], 0);
   }

   for (unsigned plane = 0; plane < 4; plane++) {
      bool stencil = plane & 1;
      struct agc_register wrong_info = {stencil ? z_info : stencil_info, S_028040_SW_MODE(24)};
      struct agc_register right_info = {stencil ? stencil_info : z_info, S_028040_SW_MODE(24)};
      struct agc_register pair[2] = {{low[plane], 0x1234}, wrong_info};
      EXPECT_FATAL(convert_array(CX, pair, 2), "the companion is missing");
      CHECK(strstr(fatal_text, stencil ? "DB_STENCIL_INFO" : "DB_Z_INFO") != NULL);
      pair[0] = (struct agc_register){high[plane], 0};
      EXPECT_FATAL(convert_array(CX, pair, 2), "the companion is missing");
      EXPECT_SAME(CX, low[plane], 0, 0);
      EXPECT_SAME(CX, low[plane], 0x00abcd00, 0x00abcd00);
      struct agc_register null_plane[2] = {{high[plane], 0}, {low[plane], 0}};
      out = convert_array(CX, null_plane, 2);
      CHECK_EQ(out.count, 2);
      EXPECT_WRITE(out, high[plane], 0);
      struct agc_register no_info[2] = {{high[plane], 3}, {low[plane], 0x100}};
      EXPECT_FATAL(convert_array(CX, no_info, 2), stencil ? "DB_STENCIL_INFO" : "DB_Z_INFO");
      EXPECT_FATAL(convert(CX, high[plane], 0), "the companion is missing");
      pair[1] = right_info;
      EXPECT_FATAL(convert_array(CX, pair, 2), "a null base stays null");
      pair[0].value = 0x10;
      out = convert_array(CX, pair, 2);
      EXPECT_WRITE(out, high[plane], 0x11);
      pair[0] = (struct agc_register){low[plane], 0x000000ab};
      out = convert_array(CX, pair, 2);
      EXPECT_WRITE(out, low[plane], 0);
   }
}

static void
test_geometry(void)
{
   uint32_t reg = CX_REG(R_028B54_VGT_SHADER_STAGES_EN);
   EXPECT_NONE(CX, reg, 0);
   EXPECT_SAME(CX, reg, S_028B54_PRIMGEN_EN(1), S_028B54_PRIMGEN_EN(1) | S_028B54_ES_EN(V_028B54_ES_STAGE_REAL));
   EXPECT_SAME(CX, reg,
               S_028B54_GS_EN(1) | S_028B54_PRIMGEN_EN(1) | S_028B54_GS_W32_EN(1) | S_028B54_VS_W32_EN(1) |
                  S_028B54_PRIMGEN_PASSTHRU_EN(1),
               S_028B54_ES_EN(V_028B54_ES_STAGE_REAL) | S_028B54_GS_EN(1) | S_028B54_PRIMGEN_EN(1) |
                  S_028B54_GS_W32_EN(1) | S_028B54_PRIMGEN_PASSTHRU_EN(1));
   EXPECT_SAME(CX, reg,
               S_028B54_LS_EN(V_028B54_LS_STAGE_ON) | S_028B54_HS_EN(1) | S_028B54_ES_EN(V_028B54_ES_STAGE_REAL) |
                  S_028B54_GS_EN(1) | S_028B54_DISPATCH_DRAW_EN(1) | S_028B54_DIS_DEALLOC_ACCUM_0(1) |
                  S_028B54_DIS_DEALLOC_ACCUM_1(1) | S_028B54_PRIMGEN_EN(1) | S_028B54_MAX_PRIMGRP_IN_WAVE(2) |
                  S_028B54_HS_W32_EN(1) | S_028B54_GS_W32_EN(1) | S_028B54_VS_W32_EN(1) | S_028B54_NGG_WAVE_ID_EN(1),
               S_028B54_LS_EN(V_028B54_LS_STAGE_ON) | S_028B54_HS_EN(1) | S_028B54_ES_EN(V_028B54_ES_STAGE_DS) |
                  S_028B54_GS_EN(1) | S_028B54_PRIMGEN_EN(1) | S_028B54_MAX_PRIMGRP_IN_WAVE(2) |
                  S_028B54_HS_W32_EN(1) | S_028B54_GS_W32_EN(1) | S_028B54_NGG_WAVE_ID_EN(1));
   EXPECT_FATAL(convert(CX, reg, S_028B54_GS_EN(1)), "without PRIMGEN_EN");
   EXPECT_FATAL(convert(CX, reg, S_028B54_VS_EN(V_028B54_VS_STAGE_COPY_SHADER)), "legacy vertex routing");

   reg = UC_REG(R_03096C_GE_CNTL);
   uint32_t fixed = S_03096C_PRIM_GRP_SIZE_GFX11(252);
   EXPECT_SAME(UC, reg, 0, S_03096C_PRIMS_PER_SUBGRP(128) | S_03096C_VERTS_PER_SUBGRP(128) | fixed);
   EXPECT_SAME(UC, reg, S_03096C_PRIM_GRP_SIZE_GFX10(64) | S_03096C_VERT_GRP_SIZE(96),
               S_03096C_PRIMS_PER_SUBGRP(64) | S_03096C_VERTS_PER_SUBGRP(96) | fixed);
   EXPECT_SAME(UC, reg,
               S_03096C_PRIM_GRP_SIZE_GFX10(120) | S_03096C_VERT_GRP_SIZE(100) | S_03096C_BREAK_WAVE_AT_EOI(1) |
                  S_03096C_PACKET_TO_ONE_PA(1),
               S_03096C_PRIMS_PER_SUBGRP(120) | S_03096C_VERTS_PER_SUBGRP(100) | S_03096C_PACKET_TO_ONE_PA(1) |
                  S_03096C_BREAK_PRIMGRP_AT_EOI(1) | fixed);
   EXPECT_SAME(UC, reg, S_03096C_BREAK_WAVE_AT_EOI(1),
               S_03096C_PRIMS_PER_SUBGRP(128) | S_03096C_VERTS_PER_SUBGRP(128) | S_03096C_BREAK_PRIMGRP_AT_EOI(1) |
                  fixed);
   EXPECT_SAME(UC, reg, S_03096C_PRIM_GRP_SIZE_GFX10(128) | S_03096C_VERT_GRP_SIZE(128), 0x1f810080);
   EXPECT_SAME(UC, reg, S_03096C_VERT_GRP_SIZE(256), S_03096C_VERTS_PER_SUBGRP(256) | fixed);
   EXPECT_SAME(UC, reg, 0x80000000 | S_03096C_PRIM_GRP_SIZE_GFX10(64), S_03096C_PRIMS_PER_SUBGRP(64) | fixed);
   EXPECT_SAME(UC, reg, 0xffffffff, 0x1f9bffff);
   EXPECT_NONE(CX, CX_REG(R_028A44_VGT_GS_ONCHIP_CNTL), 100 | (120 << 11) | (120 << 22));

   EXPECT_PRIMITIVE(V_030908_DI_PT_NONE, V_030908_DI_PT_NONE, V_030998_TRISTRIP);
   EXPECT_PRIMITIVE(V_030908_DI_PT_POINTLIST, V_030908_DI_PT_POINTLIST, V_030998_POINTLIST);
   EXPECT_PRIMITIVE(V_030908_DI_PT_LINESTRIP, V_030908_DI_PT_LINESTRIP, V_030998_LINESTRIP);
   EXPECT_PRIMITIVE(V_030908_DI_PT_LINELOOP, V_030908_DI_PT_LINELOOP, V_030998_LINESTRIP);
   EXPECT_PRIMITIVE(V_030908_DI_PT_TRILIST, V_030908_DI_PT_TRILIST, V_030998_TRISTRIP);
   EXPECT_PRIMITIVE(V_030908_DI_PT_2D_RECTANGLE, V_030908_DI_PT_RECTLIST, V_030998_RECT_2D);
   EXPECT_PRIMITIVE(V_030908_DI_PT_RECTLIST, V_030908_DI_PT_RECTLIST, V_030998_RECTLIST);
   EXPECT_PRIMITIVE(V_030908_DI_PT_PATCH, V_030908_DI_PT_PATCH, V_030998_TRISTRIP);
   EXPECT_PRIMITIVE(0x100 | V_030908_DI_PT_2D_RECTANGLE, 0x100 | V_030908_DI_PT_RECTLIST, V_030998_RECT_2D);
   uint32_t reg_primitive = UC_REG(R_030908_VGT_PRIMITIVE_TYPE);
   EXPECT_FATAL(convert(UC, reg_primitive, V_030908_DI_PT_TRI_WITH_WFLAGS), "has no output primitive");
   EXPECT_FATAL(convert(UC, reg_primitive, 8), "sceAgcCreatePrimState rejects it too");
   for (uint32_t primitive = 0; primitive < 64; primitive++) {
      struct agc_register prim_cx[2] = {{CX_REG(R_028B54_VGT_SHADER_STAGES_EN), S_028B54_PRIMGEN_EN(1)},
                                        {CX_REG(R_028A6C_VGT_GS_OUT_PRIM_TYPE), 7}};
      struct agc_register prim_uc[3] = {{UC_REG(R_03096C_GE_CNTL), 0},
                                        {UC_REG(R_030988_GE_USER_VGPR_EN), 0},
                                        {UC_REG(R_030908_VGT_PRIMITIVE_TYPE), 0}};
      if (agc_shader_update_prim_state(prim_cx, prim_uc, primitive) != 0) {
         EXPECT_FATAL(convert(UC, reg_primitive, primitive), "has no output primitive");
         continue;
      }
      struct converted host = convert(UC, reg_primitive, prim_uc[2].value);
      EXPECT_WRITE(host, UC_REG(R_030998_VGT_GS_OUT_PRIM_TYPE), prim_cx[1].value);
      CHECK_EQ(convert_array(CX, prim_cx, 2).count, 1);
   }

   reg = CX_REG(R_028A6C_VGT_GS_OUT_PRIM_TYPE);
   EXPECT_NONE(CX, reg, V_028A6C_TRISTRIP);
   EXPECT_NONE(CX, reg, V_028A6C_POINTLIST);
   struct agc_register stages_ngg[2] = {{CX_REG(R_028B54_VGT_SHADER_STAGES_EN), S_028B54_PRIMGEN_EN(1)},
                                        {reg, V_028A6C_LINESTRIP}};
   struct converted out_prim = convert_array(CX, stages_ngg, 2);
   CHECK_EQ(out_prim.count, 1);
   CHECK_EQ(out_prim.reg[0].offset, CX_REG(R_028B54_VGT_SHADER_STAGES_EN));
   struct agc_register stages_gs[2] = {
      {CX_REG(R_028B54_VGT_SHADER_STAGES_EN), S_028B54_PRIMGEN_EN(1) | S_028B54_GS_EN(1)},
      {reg, V_028A6C_TRISTRIP}};
   EXPECT_FATAL(convert_array(CX, stages_gs, 2), "needs a uconfig write");
   struct agc_reg_shadow context;
   agc_reg_shadow_reset(&context);
   struct converted uc;
   convert_bank_state(CX, stages_gs, 2, &context, &uc);
   uint32_t uc_value = 0;
   CHECK(find_write(&uc, UC_REG(R_030998_VGT_GS_OUT_PRIM_TYPE), &uc_value));
   CHECK_EQ(uc_value, V_028A6C_TRISTRIP);
   struct agc_register primitive_list[1] = {{reg_primitive, V_030908_DI_PT_POINTLIST}};
   struct converted primitive_out = convert_bank_state(UC, primitive_list, 1, &context, &uc);
   CHECK(!find_write(&primitive_out, UC_REG(R_030998_VGT_GS_OUT_PRIM_TYPE), &uc_value));
   CHECK(find_write(&primitive_out, reg_primitive, &uc_value));
   struct agc_register patch_list[1] = {{reg_primitive, 8}};
   convert_bank_state(UC, patch_list, 1, &context, &uc);
   stages_gs[0].value = S_028B54_PRIMGEN_EN(1) | S_028B54_HS_EN(1) | S_028B54_LS_EN(V_028B54_LS_STAGE_ON);
   EXPECT_FATAL(convert_array(CX, stages_gs, 2), "cx:0x29b");

   reg = UC_REG(R_03092C_GE_MULTI_PRIM_IB_RESET_EN);
   EXPECT_SAME(UC, reg, 0, S_03092C_DISABLE_FOR_AUTO_INDEX(1));
   EXPECT_SAME(UC, reg, S_03092C_RESET_EN(1) | S_03092C_MATCH_ALL_BITS(1),
               S_03092C_RESET_EN(1) | S_03092C_MATCH_ALL_BITS(1) | S_03092C_DISABLE_FOR_AUTO_INDEX(1));
   EXPECT_SAME(UC, reg, 0xffffffff, 7);

   uint32_t ring = UC_REG(R_030938_VGT_TF_RING_SIZE);
   uint32_t offchip = UC_REG(R_03093C_VGT_HS_OFFCHIP_PARAM);
   agc_host_caps.shader_engines = 2;
   EXPECT_SAME(UC, ring, 0x8000, 0x4000);
   EXPECT_SAME(UC, ring, 0, 1);
   EXPECT_SAME(UC, ring, 1, 1);
   EXPECT_SAME(UC, ring, 0xffffffff, 0xffff);
   EXPECT_SAME(UC, offchip, 511, 255);
   EXPECT_SAME(UC, offchip, 255 | (1 << 9), 127 | (1 << 10));
   EXPECT_SAME(UC, offchip, 0, 0);
   EXPECT_SAME(UC, offchip,
               S_03093C_OFFCHIP_BUFFERING_GFX7(63) | S_03093C_OFFCHIP_GRANULARITY_GFX7(V_03093C_X_2K_DWORDS),
               S_03093C_OFFCHIP_BUFFERING_GFX103(31) | S_03093C_OFFCHIP_GRANULARITY_GFX103(V_03093C_X_2K_DWORDS));
   agc_host_caps.shader_engines = 1;
   EXPECT_SAME(UC, ring, 0x8000, 0x8000);
   EXPECT_SAME(UC, offchip, 511 | (3 << 9), 511 | (3 << 10));

   reg = CX_REG(R_028B94_VGT_STRMOUT_CONFIG);
   EXPECT_NONE(CX, reg, 0);
   EXPECT_NONE(CX, reg, 0x80);
   EXPECT_FATAL(convert(CX, reg, 1), "streamout (VGT_STRMOUT_CONFIG) is not supported");
   EXPECT_FATAL(convert(CX, reg, 0x40), "streamout");
}

static void
test_fields(enum agc_host host)
{
   uint32_t reg = CX_REG(R_028004_DB_COUNT_CONTROL);
   EXPECT_SAME(CX, reg, 0x11000100, 0x11000100);
   EXPECT_SAME(CX, reg, 0xffffffff, 0xfffffffe);
   EXPECT_SAME(CX, reg,
               S_028004_ZPASS_INCREMENT_DISABLE(1) | S_028004_PERFECT_ZPASS_COUNTS(1) | S_028004_SAMPLE_RATE(3),
               S_028004_PERFECT_ZPASS_COUNTS(1) | S_028004_SAMPLE_RATE(3));

   reg = CX_REG(R_02800C_DB_RENDER_OVERRIDE);
   EXPECT_SAME(CX, reg, 0xffffffff, 0xffff7fff);
   EXPECT_SAME(CX, reg,
               S_02800C_FORCE_QC_SMASK_CONFLICT(1) | S_02800C_FORCE_Z_VALID(1) | S_02800C_FORCE_HIZ_ENABLE(2),
               S_02800C_FORCE_Z_VALID(1) | S_02800C_FORCE_HIZ_ENABLE(2));

   reg = CX_REG(R_02881C_PA_CL_VS_OUT_CNTL);
   EXPECT_SAME(CX, reg, 0xffffffff, 0x89ffffff);
   EXPECT_SAME(CX, reg, 0x06ffffff, 0x00ffffff);

   reg = CX_REG(R_028824_PA_SU_LINE_STIPPLE_CNTL);
   EXPECT_SAME(CX, reg, S_028824_LINE_STIPPLE_RESET(2) | S_028824_EXPAND_FULL_LENGTH(1) | S_028824_DIAMOND_ADJUST(1),
               S_028824_LINE_STIPPLE_RESET(2) | S_028824_EXPAND_FULL_LENGTH(1));

   reg = CX_REG(R_028A48_PA_SC_MODE_CNTL_0);
   EXPECT_SAME(CX, reg, 2, 2);
   EXPECT_SAME(CX, reg, S_028A48_MSAA_ENABLE(1) | S_028A48_VPORT_SCISSOR_ENABLE(1) | S_028A48_SCALE_LINE_WIDTH_PAD(1),
               S_028A48_MSAA_ENABLE(1) | S_028A48_VPORT_SCISSOR_ENABLE(1));

   reg = CX_REG(R_028A98_VGT_DRAW_PAYLOAD_CNTL);
   EXPECT_SAME(CX, reg,
               S_028A98_OBJPRIM_ID_EN(1) | S_028A98_EN_REG_RT_INDEX(1) | S_028A98_OBJECT_ID_INST_EN_GFX10(1) |
                  S_028A98_EN_PRIM_PAYLOAD(1) | S_028A98_EN_DRAW_VP(1),
               S_028A98_EN_REG_RT_INDEX(1) | S_028A98_EN_PRIM_PAYLOAD(1) | S_028A98_EN_DRAW_VP(1));
   EXPECT_SAME(CX, reg, 0x7f, 0x3a);

   reg = CX_REG(R_028B6C_VGT_TF_PARAM);
   EXPECT_SAME(CX, reg, 0xffffffff, 0xffe7fcff);
   EXPECT_SAME(CX, reg,
               S_028B6C_TYPE(2) | S_028B6C_PARTITIONING(3) | S_028B6C_TOPOLOGY(3) | S_028B6C_RESERVED_REDUC_AXIS(1) |
                  S_028B6C_DEPRECATED(1) | S_028B6C_DISABLE_DONUTS(1) | S_028B6C_DISTRIBUTION_MODE(3) |
                  S_028B6C_DETECT_ONE(1) | S_028B6C_DETECT_ZERO(1),
               S_028B6C_TYPE(2) | S_028B6C_PARTITIONING(3) | S_028B6C_TOPOLOGY(3) | S_028B6C_DISABLE_DONUTS(1) |
                  S_028B6C_DISTRIBUTION_MODE(3));

   reg = CX_REG(R_028C40_PA_SC_SHADER_CONTROL);
   EXPECT_SAME(CX, reg, 0x6f, 0x63);
   EXPECT_SAME(CX, reg, 0xff, 0x73);

   reg = CX_REG(R_028830_PA_SU_SMALL_PRIM_FILTER_CNTL);
   EXPECT_SAME(CX, reg, 0x21, 0x01);
   EXPECT_SAME(CX, reg, 0x01, 0x01);
   EXPECT_SAME(CX, reg, 0x7f, host == AGC_HOST_GFX115 ? 0x5f : 0x1f);

   reg = CX_REG(R_028840_PA_STEREO_CNTL);
   EXPECT_SAME(CX, reg, 0x04000000, 0);
   EXPECT_SAME(CX, reg, 0xffffffff, 0x007f0ffe);
   EXPECT_SAME(CX, reg, S_028840_STEREO_MODE(2) | S_028840_RT_SLICE_OFFSET_GFX10(1) | S_028840_VP_ID_MODE_GFX10(1),
               S_028840_STEREO_MODE(2) | S_028840_RT_SLICE_OFFSET_GFX10(1) | S_028840_VP_ID_MODE_GFX10(1));

   reg = UC_REG(R_030D20_SQC_CACHES);
   EXPECT_SAME(UC, reg,
               S_030D20_TARGET_INST(1) | S_030D20_TARGET_DATA(1) | S_030D20_INVALIDATE(1) | S_030D20_WRITEBACK(1) |
                  S_030D20_VOL(1) | S_030D20_COMPLETE(1) | S_030D20_L2_WB_POLICY(3),
               S_030D20_TARGET_INST(1) | S_030D20_TARGET_DATA(1) | S_030D20_INVALIDATE(1) | S_030D20_COMPLETE(1));

   reg = SH_REG(R_00B87C_COMPUTE_MISC_RESERVED);
   EXPECT_SAME(SH, reg, 0xffffffff, 0xfffffffb);

   reg = UC_REG(R_03106C_GDS_GWS_RESOURCE);
   EXPECT_SAME(UC, reg, 0x3fffffff, 0xe7ffffff);
   EXPECT_SAME(UC, reg, 0xc0000000, 0);
   EXPECT_SAME(UC, reg, 0x08000000, 0x20000000);

   reg = CX_REG(R_0286C4_SPI_VS_OUT_CONFIG);
   EXPECT_SAME(CX, reg,
               S_0286C4_VS_EXPORT_COUNT(7) | S_0286C4_VS_HALF_PACK(1) | S_0286C4_NO_PC_EXPORT(1) |
                  S_0286C4_PRIM_EXPORT_COUNT(3),
               S_0286C4_VS_EXPORT_COUNT(7) | S_0286C4_VS_HALF_PACK(1) | S_0286C4_NO_PC_EXPORT(1) |
                  S_0286C4_PRIM_EXPORT_COUNT(3));
   EXPECT_SAME(CX, CX_REG(R_02803C_DB_RESERVED_REG_2), 0x10000000, 0x10000000);
}

static void
test_shader_registers(void)
{
   EXPECT_ONE(SH, SH_REG(R_00B208_SPI_SHADER_USER_DATA_ADDR_LO_GS), 0x12345678,
              SH_REG(R_00B220_SPI_SHADER_PGM_LO_GS), 0x12345678);
   EXPECT_ONE(SH, SH_REG(R_00B20C_SPI_SHADER_USER_DATA_ADDR_HI_GS), 0x9a,
              SH_REG(R_00B224_SPI_SHADER_PGM_HI_GS), 0x9a);
   EXPECT_ONE(SH, SH_REG(R_00B408_SPI_SHADER_USER_DATA_ADDR_LO_HS), 0x12345678,
              SH_REG(R_00B420_SPI_SHADER_PGM_LO_HS), 0x12345678);
   EXPECT_ONE(SH, SH_REG(R_00B40C_SPI_SHADER_USER_DATA_ADDR_HI_HS), 0x9a,
              SH_REG(R_00B424_SPI_SHADER_PGM_HI_HS), 0x9a);
   EXPECT_NONE(SH, SH_REG(R_00B220_SPI_SHADER_PGM_LO_GS), 0x00445566);
   EXPECT_NONE(SH, SH_REG(R_00B224_SPI_SHADER_PGM_HI_GS), 0x01);
   EXPECT_NONE(SH, SH_REG(R_00B420_SPI_SHADER_PGM_LO_HS), 0x00445566);
   EXPECT_NONE(SH, SH_REG(R_00B424_SPI_SHADER_PGM_HI_HS), 0x01);

   static const uint32_t passed[] = {
      SH_REG(R_00B020_SPI_SHADER_PGM_LO_PS),   SH_REG(R_00B024_SPI_SHADER_PGM_HI_PS),
      SH_REG(R_00B028_SPI_SHADER_PGM_RSRC1_PS), SH_REG(R_00B02C_SPI_SHADER_PGM_RSRC2_PS),
      SH_REG(R_00B320_SPI_SHADER_PGM_LO_ES),   SH_REG(R_00B324_SPI_SHADER_PGM_HI_ES),
      SH_REG(R_00B228_SPI_SHADER_PGM_RSRC1_GS), SH_REG(R_00B22C_SPI_SHADER_PGM_RSRC2_GS),
      SH_REG(R_00B520_SPI_SHADER_PGM_LO_LS),   SH_REG(R_00B524_SPI_SHADER_PGM_HI_LS),
      SH_REG(R_00B428_SPI_SHADER_PGM_RSRC1_HS), SH_REG(R_00B42C_SPI_SHADER_PGM_RSRC2_HS),
      SH_REG(R_00B830_COMPUTE_PGM_LO),         SH_REG(R_00B834_COMPUTE_PGM_HI),
      SH_REG(R_00B848_COMPUTE_PGM_RSRC1),      SH_REG(R_00B84C_COMPUTE_PGM_RSRC2),
      SH_REG(R_00B8A0_COMPUTE_PGM_RSRC3),      SH_REG(R_00B004_SPI_SHADER_PGM_RSRC4_PS),
      SH_REG(R_00B01C_SPI_SHADER_PGM_RSRC3_PS), SH_REG(R_00B204_SPI_SHADER_PGM_RSRC4_GS),
      SH_REG(R_00B21C_SPI_SHADER_PGM_RSRC3_GS), SH_REG(R_00B404_SPI_SHADER_PGM_RSRC4_HS),
      SH_REG(R_00B41C_SPI_SHADER_PGM_RSRC3_HS),
   };
   for (unsigned i = 0; i < COUNT_OF(passed); i++) {
      EXPECT_SAME(SH, passed[i], 0xa5a5a5a5, 0xa5a5a5a5);
      EXPECT_SAME(SH, passed[i], 0xffffffff, 0xffffffff);
   }
   EXPECT_SAME(CX, CX_REG(R_0286C4_SPI_VS_OUT_CONFIG), 0xffffffff, 0xffffffff);
   EXPECT_SAME(CX, CX_REG(R_028714_SPI_SHADER_COL_FORMAT), 0xffffffff, 0xffffffff);
   EXPECT_SAME(CX, CX_REG(R_028714_SPI_SHADER_COL_FORMAT), 0x00004444, 0x00004444);
   EXPECT_SAME(CX, CX_REG(R_0287FC_GE_MAX_OUTPUT_PER_SUBGROUP), 0, 0);
   EXPECT_SAME(CX, CX_REG(R_028B4C_GE_NGG_SUBGRP_CNTL), 0, 0);
   EXPECT_SAME(CX, CX_REG(R_02870C_SPI_SHADER_POS_FORMAT), 0, 0);

   static const uint32_t dropped[] = {
      SH_REG(R_00B9F4_COMPUTE_DISPATCH_TUNNEL), 0x216, 0x217, 0x219, 0x21a, 0x280,
   };
   for (unsigned i = 0; i < COUNT_OF(dropped); i++)
      EXPECT_NONE(SH, dropped[i], 0xffffffff);
   EXPECT_NONE(UC, UC_REG(R_030980_GE_PC_ALLOC), 0x7ff);
}

static void
test_unsupported(void)
{
   static const struct {
      enum agc_reg_bank bank;
      uint32_t offset;
      const char *name;
   } gates[] = {
      {CX, 0x006, "cx:0x006"}, {CX, 0x0e8, "cx:0x0e8"}, {CX, 0x0e9, "cx:0x0e9"}, {CX, 0x0ea, "cx:0x0ea"},
      {CX, 0x188, "cx:0x188"}, {CX, 0x18e, "cx:0x18e"}, {CX, 0x319, "cx:0x319"}, {CX, 0x31a, "cx:0x31a"},
      {CX, 0x382, "cx:0x382"}, {CX, 0x290, "cx:0x290"}, {CX, 0x2a5, "cx:0x2a5"}, {CX, 0x2e6, "cx:0x2e6"},
      {CX, 0x100, "cx:0x100"}, {SH, 0x048, "sh:0x048"}, {SH, 0x04c, "sh:0x04c"}, {SH, 0x06b, "sh:0x06b"},
      {SH, 0x0cc, "sh:0x0cc"}, {SH, 0x15b, "sh:0x15b"}, {SH, 0x031, "sh:0x031"}, {UC, 0x26c, "uc:0x26c"},
      {UC, 0x26e, "uc:0x26e"}, {UC, 0x278, "uc:0x278"}, {UC, 0x094, "uc:0x094"}, {UC, 0x095, "uc:0x095"},
      {UC, 0x07c, "uc:0x07c"}, {UC, 0x0cc, "uc:0x0cc"}, {UC, 0x240, "uc:0x240"}, {UC, 0x440, "uc:0x440"},
   };
   for (unsigned i = 0; i < COUNT_OF(gates); i++) {
      CHECK_EQ(class_of(gates[i].bank, gates[i].offset), AGC_REG_GUEST);
      EXPECT_NONE(gates[i].bank, gates[i].offset, 0);
      EXPECT_FATAL(convert(gates[i].bank, gates[i].offset, 0x20), "asks for a function this host does not have");
      CHECK(strstr(fatal_text, gates[i].name) != NULL);
      CHECK(strstr(fatal_text, "0x00000020") != NULL);
   }

   static const struct {
      enum agc_reg_bank bank;
      uint32_t offset;
   } ignored[] = {
      {CX, 0x0db}, {UC, 0x0a2}, {UC, 0x035}, {UC, 0x03c}, {UC, 0x248}, {CX, 0x20d},  {CX, 0x00e},
      {CX, 0x01f}, {CX, 0x104}, {CX, 0x109}, {CX, 0x0d7}, {CX, 0x311}, {CX, 0x312},  {CX, 0x314},
      {CX, 0x2d4}, {CX, 0x316}, {CX, 0x317}, {UC, 0x200},
      {UC, 0x1000}, {UC, 0x11c0}, {UC, 0x1800}, {UC, 0x19c0}, {UC, 0x1c80}, {UC, 0x1e02},
   };
   for (unsigned i = 0; i < COUNT_OF(ignored); i++)
      EXPECT_NONE(ignored[i].bank, ignored[i].offset, 0xffffffff);

   uint32_t host_offset;
   uint32_t host_value;
   CHECK(!agc_reg_convert(CX, CX_REG(R_028A6C_VGT_GS_OUT_PRIM_TYPE), 2, &host_offset, &host_value, "test"));
   EXPECT_FATAL(agc_reg_convert(UC, UC_REG(R_030908_VGT_PRIMITIVE_TYPE), 4, &host_offset, &host_value, "test"),
                "needs a trampoline");
}

static bool
metadata_register(enum agc_reg_bank bank, uint32_t offset)
{
   if (bank != CX)
      return false;
   if (offset >= CX_REG(R_028E60_CB_COLOR0_CMASK_BASE_EXT) && offset < CX_REG(R_028EA0_CB_COLOR0_DCC_BASE_EXT))
      return true;
   if (offset < CX_REG(R_028C60_CB_COLOR0_BASE) || offset >= CX_REG(R_028C60_CB_COLOR0_BASE) + 8 * 15)
      return false;
   uint32_t member = (offset - CX_REG(R_028C60_CB_COLOR0_BASE)) % 15;
   return member >= CX_REG(R_028C7C_CB_COLOR0_CMASK) - CX_REG(R_028C60_CB_COLOR0_BASE) &&
          member <= CX_REG(R_028C90_CB_COLOR0_CLEAR_WORD1) - CX_REG(R_028C60_CB_COLOR0_BASE);
}

static void
check_host_write(enum agc_reg_bank bank, const struct agc_register *write)
{
   CHECK(!metadata_register(bank, write->offset));
   if (bank != CX)
      return;
   if (write->offset == CX_REG(R_028ABC_DB_HTILE_SURFACE))
      CHECK_EQ(write->value, S_028ABC_FULL_CACHE(1) | S_028ABC_PIPE_ALIGNED(1));
   if (write->offset == CX_REG(R_028808_CB_COLOR_CONTROL))
      CHECK(G_028808_MODE(write->value) <= V_028808_CB_NORMAL);
   if (write->offset >= CX_REG(R_028C60_CB_COLOR0_BASE) && write->offset < CX_REG(R_028C60_CB_COLOR0_BASE) + 8 * 15) {
      uint32_t member = (write->offset - CX_REG(R_028C60_CB_COLOR0_BASE)) % 15;
      if (member == CX_REG(R_028C70_CB_COLOR0_INFO) - CX_REG(R_028C60_CB_COLOR0_BASE))
         CHECK_EQ(write->value & ~UINT32_C(0x03f79f9f), 0);
      if (member == CX_REG(R_028C78_CB_COLOR0_FDCC_CONTROL) - CX_REG(R_028C60_CB_COLOR0_BASE))
         CHECK(!G_028C78_FDCC_ENABLE(write->value));
   }
}

static void
test_default_tables(void)
{
   static struct agc_register guest[AGC_DEFAULTS_BLOCK_REGISTERS];
   static struct agc_register host[AGC_DEFAULTS_BLOCK_REGISTERS * AGC_REG_WRITES_MAX];
   uint32_t registers = 0;
   for (uint32_t version = 0; version <= AGC_DEFAULTS_VERSION_MAX; version++) {
      for (unsigned internal = 0; internal < 2; internal++) {
         uint32_t group_count = 0;
         const struct agc_defaults_group *groups = agc_register_defaults_groups(version, internal != 0, &group_count);
         CHECK(group_count > 0);
         for (uint32_t g = 0; g < group_count; g++) {
            const struct agc_defaults_group *group = &groups[g];
            enum agc_reg_bank bank = (enum agc_reg_bank)group->bank;
            uint32_t count = 0;
            CHECK(group->count <= AGC_DEFAULTS_BLOCK_REGISTERS);
            for (uint32_t r = 0; r < group->count && r < AGC_DEFAULTS_BLOCK_REGISTERS; r++) {
               enum agc_reg_class cls = class_of(bank, group->registers[r].offset);
               registers++;
               CHECK(cls != AGC_REG_UNDEFINED);
               CHECK(cls != AGC_REG_PENDING);
               guest[count++] = group->registers[r];
            }
            fatal_text[0] = 0;
            if (setjmp(fatal_jump) == 0) {
               uint32_t written = agc_reg_convert_pairs(bank, guest, count, host, "test");
               CHECK(written <= count * AGC_REG_WRITES_MAX);
               for (uint32_t w = 0; w < written && w < COUNT_OF(host); w++)
                  check_host_write(bank, &host[w]);
            } else {
               fprintf(stderr, "defaults version %u, group type %#010x: %s\n", version, group->type, fatal_text);
               CHECK(!"a default table does not convert");
            }
         }
      }
   }
   CHECK(registers > 0);
}

static void
test_host(enum agc_host host)
{
   CHECK(agc_host_select(host));
   agc_host_caps.shader_engines = 1;
   test_pending(host);
   test_classes();
   test_depth();
   test_rasterizer();
   test_pixel_inputs();
   test_scratch();
   test_blend();
   test_colour_targets();
   test_colour_target_state(host);
   test_colour_bases();
   test_depth_bases();
   test_geometry();
   test_fields(host);
   test_shader_registers();
   test_unsupported();
   test_default_tables();
}

int
main(void)
{
   agc_set_fatal_handler(on_fatal);
   test_host(AGC_HOST_GFX11);
   test_host(AGC_HOST_GFX115);
   test_other_hosts();
   return TEST_RESULT();
}
