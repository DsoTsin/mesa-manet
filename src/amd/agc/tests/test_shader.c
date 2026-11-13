#include <stddef.h>
#include <string.h>

#include "agc_shader.h"
#include "test.h"

struct blob {
   struct agc_shader header;
   struct agc_register cx[6];
   struct agc_register sh[4];
   struct agc_shader_specials specials;
   struct agc_shader_user_data user_data;
   uint16_t direct[4];
   uint16_t sharp[4][2];
   uint32_t inputs[4];
   uint32_t outputs[4];
};

static void
relative(void *field, const void *target)
{
   uintptr_t value = (uintptr_t)target - (uintptr_t)field;
   memcpy(field, &value, sizeof(value));
}

static void
make_blob(struct blob *blob, uint8_t kind, uint32_t program_lo)
{
   memset(blob, 0, sizeof(*blob));
   struct agc_shader *header = &blob->header;
   header->file_header = AGC_SHADER_MAGIC;
   header->version = AGC_SHADER_VERSION;
   header->header_size = sizeof(*blob);
   header->shader_size = 0x40;
   header->kind = kind;
   header->num_cx_registers = 6;
   header->num_sh_registers = 4;
   header->num_input_semantics = 2;
   header->num_output_semantics = 2;
   relative(&header->cx_registers, blob->cx);
   relative(&header->sh_registers, blob->sh);
   relative(&header->specials, &blob->specials);
   relative(&header->user_data, &blob->user_data);
   relative(&header->input_semantics, blob->inputs);
   relative(&header->output_semantics, blob->outputs);
   relative(&blob->user_data.direct_resource_offset, blob->direct);
   for (unsigned i = 0; i < 4; i++)
      relative(&blob->user_data.sharp_resource_offset[i], blob->sharp[i]);
   blob->sh[0] = (struct agc_register){0x00a, 0x55};
   blob->sh[1] = (struct agc_register){program_lo, 0x10};
   blob->sh[2] = (struct agc_register){program_lo + 1, 0xabcdef00};
   blob->sh[3] = (struct agc_register){0x00b, 0x66};
   blob->specials.ge_cntl = (struct agc_register){0x25b, 0x1234};
   blob->specials.vgt_shader_stages_en = (struct agc_register){0x2d5, 0x00400000};
   blob->specials.vgt_gs_out_prim_type = (struct agc_register){0x29b, 2};
   blob->specials.ge_user_vgpr_en = (struct agc_register){0x262, 3};
}

static uint64_t hook_host;
static int32_t hook_result;
static unsigned hook_calls;

static int32_t
on_program(struct agc_shader *shader, uint64_t guest_code, uint64_t *host_code)
{
   hook_calls++;
   CHECK_EQ(shader->code, guest_code);
   *host_code = hook_host;
   return hook_result;
}

static void
test_program_register(void)
{
   static const uint32_t expected[AGC_SHADER_KIND_COUNT] = {
      [AGC_SHADER_CS] = 0x20c,   [AGC_SHADER_PS] = 0x008,      [AGC_SHADER_GS] = 0x0c8,
      [AGC_SHADER_HS] = 0x148,   [AGC_SHADER_GS_BACK] = 0x088, [AGC_SHADER_HS_BACK] = 0x108,
   };
   for (unsigned kind = 0; kind < AGC_SHADER_KIND_COUNT; kind++) {
      uint32_t lo = ~0u;
      CHECK(agc_shader_program_register((uint8_t)kind, &lo));
      CHECK_EQ(lo, expected[kind]);
   }
   uint32_t lo;
   CHECK(!agc_shader_program_register(AGC_SHADER_KIND_COUNT, &lo));
   CHECK(!agc_shader_program_register(0xff, &lo));
}

static void
test_rebase(void)
{
   struct agc_register registers[3] = {{0x00a, 1}, {0x008, 0x10}, {0x009, 0xabcdef02}};
   CHECK_EQ(agc_shader_rebase_program(registers, 3, 0x008, UINT64_C(0x200001000)), 0);
   uint64_t address = UINT64_C(0x200001000) + (UINT64_C(0x10) << 8) + (UINT64_C(2) << 40);
   CHECK_EQ(registers[1].value, (uint32_t)(address >> 8));
   CHECK_EQ(registers[2].value, 0xabcdef00u | (uint32_t)((address >> 40) & 0xff));
   CHECK_EQ(registers[0].value, 1);

   struct agc_register alone[1] = {{0x008, 0x10}};
   CHECK_EQ(agc_shader_rebase_program(alone, 1, 0x008, 0x1000), AGC_ERROR_INVALID_SHADER_PROGRAM);
   struct agc_register split[2] = {{0x008, 0x10}, {0x00a, 0}};
   CHECK_EQ(agc_shader_rebase_program(split, 2, 0x008, 0x1000), AGC_ERROR_INVALID_SHADER_PROGRAM);
   struct agc_register none[2] = {{0x00a, 0x10}, {0x00b, 0}};
   CHECK_EQ(agc_shader_rebase_program(none, 2, 0x008, 0x1000), AGC_ERROR_INVALID_SHADER_PROGRAM);
   struct agc_register high[2] = {{0x008, 0xffffffff}, {0x009, 0xff}};
   CHECK_EQ(agc_shader_rebase_program(high, 2, 0x008, 0x1000), AGC_ERROR_INVALID_SHADER_PROGRAM);
   CHECK_EQ(high[0].value, 0xffffffffu);
}

static void
test_create(void)
{
   struct blob blob;
   struct agc_shader *shader = NULL;
   const uint64_t code = UINT64_C(0x212345600);

   agc_shader_set_program_hook(NULL);
   make_blob(&blob, AGC_SHADER_PS, 0x008);
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), 0);
   CHECK(shader == &blob.header);
   CHECK(shader->cx_registers == blob.cx);
   CHECK(shader->sh_registers == blob.sh);
   CHECK(shader->specials == &blob.specials);
   CHECK(shader->user_data == &blob.user_data);
   CHECK(shader->input_semantics == blob.inputs);
   CHECK(shader->output_semantics == blob.outputs);
   CHECK(shader->user_data->direct_resource_offset == blob.direct);
   for (unsigned i = 0; i < 4; i++)
      CHECK(shader->user_data->sharp_resource_offset[i] == blob.sharp[i]);
   CHECK_EQ(shader->code, code);
   uint64_t address = code + (UINT64_C(0x10) << 8);
   CHECK_EQ(blob.sh[1].value, (uint32_t)(address >> 8));
   CHECK_EQ(blob.sh[2].value, 0xabcdef00u | (uint32_t)((address >> 40) & 0xff));
   CHECK_EQ(blob.sh[0].value, 0x55);
   CHECK_EQ(blob.sh[3].value, 0x66);

   make_blob(&blob, AGC_SHADER_GS_FRONT, 0x0c8);
   uint32_t before = blob.sh[1].value;
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), 0);
   CHECK_EQ(blob.sh[1].value, before);

   make_blob(&blob, AGC_SHADER_PS, 0x008);
   blob.header.user_data = NULL;
   blob.header.specials = NULL;
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), 0);
   CHECK(shader->user_data == NULL && shader->specials == NULL);

   make_blob(&blob, AGC_SHADER_PS, 0x008);
   CHECK_EQ(agc_shader_create(NULL, &blob.header, code), AGC_ERROR_INVALID_ARGUMENT);
   CHECK_EQ(agc_shader_create(&shader, NULL, code), AGC_ERROR_INVALID_ARGUMENT);
   CHECK_EQ(agc_shader_create(&shader, &blob.header, 0), AGC_ERROR_INVALID_ARGUMENT);
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code + 4), AGC_ERROR_INVALID_ALIGNMENT);
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code | UINT64_C(1) << 48), AGC_ERROR_INVALID_ALIGNMENT);
   blob.header.file_header ^= 1;
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), AGC_ERROR_INVALID_SHADER_PROGRAM);
   blob.header.file_header ^= 1;
   blob.header.version++;
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), AGC_ERROR_INVALID_SHADER_PROGRAM);
   blob.header.version--;
   blob.header.shader_size = 0;
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), AGC_ERROR_INVALID_SHADER_PROGRAM);
   blob.header.shader_size = 0x42;
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), AGC_ERROR_INVALID_SHADER_PROGRAM);
   blob.header.shader_size = 0x40;
   blob.header.kind = AGC_SHADER_KIND_COUNT;
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), AGC_ERROR_INVALID_SHADER_PROGRAM);
   CHECK(blob.header.cx_registers != blob.cx);

   make_blob(&blob, AGC_SHADER_PS, 0x00c);
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), AGC_ERROR_INVALID_SHADER_PROGRAM);
   make_blob(&blob, AGC_SHADER_PS, 0x008);
   blob.header.num_sh_registers = 0;
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), AGC_ERROR_INVALID_SHADER_PROGRAM);
}

static void
test_hook(void)
{
   struct blob blob;
   struct agc_shader *shader = NULL;
   const uint64_t code = UINT64_C(0x212345600);

   agc_shader_set_program_hook(on_program);
   hook_calls = 0;
   hook_result = 0;
   hook_host = UINT64_C(0x20000400000);
   make_blob(&blob, AGC_SHADER_CS, 0x20c);
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), 0);
   CHECK_EQ(hook_calls, 1);
   CHECK_EQ(shader->code, code);
   uint64_t address = hook_host + (UINT64_C(0x10) << 8);
   CHECK_EQ(blob.sh[1].value, (uint32_t)(address >> 8));
   CHECK_EQ(blob.sh[2].value, 0xabcdef00u | (uint32_t)((address >> 40) & 0xff));

   hook_result = AGC_ERROR_INVALID_SHADER_PROGRAM;
   make_blob(&blob, AGC_SHADER_CS, 0x20c);
   shader = NULL;
   CHECK_EQ(agc_shader_create(&shader, &blob.header, code), AGC_ERROR_INVALID_SHADER_PROGRAM);
   CHECK(shader == NULL);
   CHECK_EQ(blob.sh[1].value, 0x10);
   agc_shader_set_program_hook(NULL);
}

static void
test_prim_state(void)
{
   struct blob gs_blob, hs_blob;
   struct agc_shader *gs = NULL;
   struct agc_shader *hs = NULL;
   const uint64_t code = UINT64_C(0x212345600);
   make_blob(&gs_blob, AGC_SHADER_GS, 0x0c8);
   make_blob(&hs_blob, AGC_SHADER_HS, 0x148);
   CHECK_EQ(agc_shader_create(&gs, &gs_blob.header, code), 0);
   CHECK_EQ(agc_shader_create(&hs, &hs_blob.header, code), 0);

   uint32_t payload = ~0u;
   CHECK_EQ(agc_shader_gs_prim_payload(&payload, gs), 0);
   CHECK_EQ(payload, 0);
   gs_blob.cx[3] = (struct agc_register){0x1c2, 0x12};
   CHECK_EQ(agc_shader_gs_prim_payload(&payload, gs), 0);
   CHECK_EQ(payload, 8);
   CHECK_EQ(agc_shader_gs_prim_payload(&payload, hs), AGC_ERROR_NOT_A_GEOMETRY_SHADER);
   CHECK_EQ(agc_shader_gs_prim_payload(NULL, gs), AGC_ERROR_INVALID_ARGUMENT);
   CHECK_EQ(agc_shader_gs_prim_payload(&payload, NULL), AGC_ERROR_INVALID_ARGUMENT);

   struct agc_register cx[2];
   struct agc_register uc[3];
   memset(cx, 0xff, sizeof(cx));
   memset(uc, 0xff, sizeof(uc));
   CHECK_EQ(agc_shader_create_prim_state(cx, uc, NULL, gs, 4), 0);
   CHECK(cx[0].offset == 0x2d5 && cx[0].value == 0x00400000);
   CHECK(cx[1].offset == 0x29b && cx[1].value == 2);
   CHECK(uc[0].offset == 0x25b && uc[0].value == 0x1234);
   CHECK(uc[1].offset == 0x262 && uc[1].value == 3);
   CHECK(uc[2].offset == 0x242 && uc[2].value == 4);

   CHECK_EQ(agc_shader_create_prim_state(cx, NULL, NULL, gs, 1), 0);
   CHECK_EQ(cx[1].value, 0);
   CHECK_EQ(agc_shader_create_prim_state(cx, NULL, NULL, gs, 2), 0);
   CHECK_EQ(cx[1].value, 1);
   CHECK_EQ(agc_shader_create_prim_state(cx, NULL, NULL, gs, 7), 0);
   CHECK_EQ(cx[1].value, 3);
   CHECK_EQ(agc_shader_create_prim_state(cx, NULL, NULL, gs, 17), 0);
   CHECK_EQ(cx[1].value, 4);
   CHECK_EQ(agc_shader_create_prim_state(cx, NULL, NULL, gs, 8), AGC_ERROR_OUT_OF_RANGE);

   hs_blob.specials.vgt_shader_stages_en.value = 0x04;
   hs_blob.specials.vgt_gs_out_prim_type.value = 5;
   hs_blob.specials.ge_user_vgpr_en.value = 9;
   CHECK_EQ(agc_shader_create_prim_state(cx, uc, hs, gs, 9), 0);
   CHECK_EQ(cx[0].value, 0x00400004);
   CHECK_EQ(cx[1].value, 5);
   CHECK_EQ(uc[1].value, 9);
   CHECK_EQ(uc[2].value, 9);

   gs_blob.specials.vgt_shader_stages_en.value |= 0x20;
   gs_blob.specials.vgt_gs_out_prim_type.value = 6;
   CHECK_EQ(agc_shader_create_prim_state(cx, uc, hs, gs, 9), 0);
   CHECK_EQ(cx[1].value, 6);
   gs_blob.specials.vgt_shader_stages_en.value &= ~UINT32_C(0x20);

   CHECK_EQ(agc_shader_create_prim_state(NULL, NULL, NULL, NULL, 99), 0);
   CHECK_EQ(agc_shader_create_prim_state(cx, uc, NULL, NULL, 4), AGC_ERROR_INVALID_ARGUMENT);
   CHECK_EQ(agc_shader_create_prim_state(cx, uc, NULL, hs, 4), AGC_ERROR_INVALID_SHADER_HALVES);
   CHECK_EQ(agc_shader_create_prim_state(cx, uc, gs, gs, 4), AGC_ERROR_INVALID_SHADER_HALVES);
   gs_blob.specials.ge_cntl.offset = 0x25c;
   CHECK_EQ(agc_shader_create_prim_state(cx, uc, NULL, gs, 4), AGC_ERROR_INVALID_SHADER_PROGRAM);
   gs_blob.specials.ge_cntl.offset = 0x25b;

   CHECK_EQ(agc_shader_create_prim_state(cx, uc, NULL, gs, 4), 0);
   CHECK_EQ(agc_shader_update_prim_state(cx, uc, 2), 0);
   CHECK_EQ(cx[1].value, 1);
   CHECK_EQ(uc[2].value, 2);
   CHECK_EQ(agc_shader_update_prim_state(cx, uc, 8), AGC_ERROR_OUT_OF_RANGE);
   cx[0].value |= 0x20;
   cx[1].value = 6;
   CHECK_EQ(agc_shader_update_prim_state(cx, uc, 4), 0);
   CHECK_EQ(cx[1].value, 6);
   CHECK_EQ(uc[2].value, 4);
   CHECK_EQ(agc_shader_update_prim_state(NULL, NULL, 4), 0);
}

static void
test_interpolants(void)
{
   struct blob gs_blob, ps_blob;
   struct agc_shader *gs = NULL;
   struct agc_shader *ps = NULL;
   const uint64_t code = UINT64_C(0x212345600);
   make_blob(&gs_blob, AGC_SHADER_GS, 0x0c8);
   make_blob(&ps_blob, AGC_SHADER_PS, 0x008);
   CHECK_EQ(agc_shader_create(&gs, &gs_blob.header, code), 0);
   CHECK_EQ(agc_shader_create(&ps, &ps_blob.header, code), 0);

   gs_blob.outputs[0] = 3u << 8 | 5;
   gs_blob.outputs[1] = 7u << 8 | 9;
   ps_blob.inputs[0] = 9;
   ps_blob.inputs[1] = 4;

   struct agc_register registers[32];
   memset(registers, 0xff, sizeof(registers));
   CHECK_EQ(agc_shader_interpolant_mapping(registers, gs, ps, false, "test"), 0);
   CHECK(registers[0].offset == 0x191 && registers[0].value == 7);
   CHECK(registers[1].offset == 0x192 && registers[1].value == 0x20);
   CHECK_EQ(registers[2].offset, 0xffffffffu);

   CHECK_EQ(agc_shader_interpolant_mapping(registers, gs, ps, true, "test"), 0);
   CHECK_EQ(registers[0].value, 7);
   for (unsigned i = 2; i < 32; i++)
      CHECK(registers[i].offset == 0x191 + i && registers[i].value == i);

   ps_blob.inputs[0] = 5 | 0x00400000;
   CHECK_EQ(agc_shader_interpolant_mapping(registers, gs, ps, false, "test"), 0);
   CHECK_EQ(registers[0].value, 0x403);
   ps_blob.inputs[0] = 5 | 2u << 28;
   CHECK_EQ(agc_shader_interpolant_mapping(registers, gs, ps, false, "test"), 0);
   CHECK_EQ(registers[0].value, 0x203);

   CHECK_EQ(agc_shader_interpolant_mapping(registers, NULL, NULL, true, "test"), 0);
   for (unsigned i = 0; i < 32; i++)
      CHECK(registers[i].offset == 0x191 + i && registers[i].value == i);
   CHECK_EQ(agc_shader_interpolant_mapping(registers, gs, NULL, false, "test"), AGC_ERROR_INVALID_ARGUMENT);
   CHECK_EQ(agc_shader_interpolant_mapping(NULL, gs, ps, true, "test"), AGC_ERROR_INVALID_ARGUMENT);
   CHECK_EQ(agc_shader_interpolant_mapping(registers, NULL, ps, false, "test"), AGC_ERROR_INVALID_ARGUMENT);
   ps_blob.header.num_input_semantics = 33;
   CHECK_EQ(agc_shader_interpolant_mapping(registers, gs, ps, false, "test"), AGC_ERROR_OUT_OF_RANGE);
}

static void
test_oversubscription(void)
{
   struct blob gs_blob;
   struct agc_shader *gs = NULL;
   make_blob(&gs_blob, AGC_SHADER_GS, 0x0c8);
   CHECK_EQ(agc_shader_create(&gs, &gs_blob.header, UINT64_C(0x212345600)), 0);

   struct agc_register registers[2];
   CHECK_EQ(agc_shader_gs_oversubscription(registers, gs, -1, 0.0f), 0);
   CHECK(registers[0].offset == 0x260 && registers[0].value == 0x7ff);
   CHECK(registers[1].offset == 0x081 && registers[1].value == 0x007f0000);
   CHECK_EQ(agc_shader_gs_oversubscription(registers, gs, 0, 1.0f), 0);
   CHECK(registers[0].value == 0 && registers[1].value == 0);
   CHECK_EQ(agc_shader_gs_oversubscription(registers, gs, -2, 1.0f), AGC_ERROR_OUT_OF_RANGE);
   CHECK_EQ(agc_shader_gs_oversubscription(registers, gs, 1, -1.0f), AGC_ERROR_OUT_OF_RANGE);
   CHECK_EQ(agc_shader_gs_oversubscription(NULL, gs, 1, 1.0f), AGC_ERROR_INVALID_ARGUMENT);
   CHECK_EQ(agc_shader_gs_oversubscription(registers, gs, 4096, 1.0f), AGC_ERROR_INVALID_SHADER_PROGRAM);

   gs_blob.cx[0] = (struct agc_register){0x291, 64u << 11};
   gs_blob.cx[1] = (struct agc_register){0x2d3, 3};
   gs_blob.cx[2] = (struct agc_register){0x1b1, 1u << 2};
   gs_blob.cx[3] = (struct agc_register){0x207, 0};
   gs_blob.cx[4] = (struct agc_register){0x1ff, 128};
   CHECK_EQ(agc_shader_gs_oversubscription(registers, gs, 1 << 20, 0.0f), 0);
   CHECK(registers[0].value == 0 && registers[1].value == 0);
   CHECK_EQ(agc_shader_gs_oversubscription(registers, gs, 1 << 20, 1.0f), 0);
   CHECK(registers[0].offset == 0x260 && registers[1].offset == 0x081);
   CHECK(registers[0].value != 0 || registers[1].value != 0);
}

int
main(void)
{
   test_program_register();
   test_rebase();
   test_create();
   test_hook();
   test_prim_state();
   test_interpolants();
   test_oversubscription();
   return TEST_RESULT();
}
