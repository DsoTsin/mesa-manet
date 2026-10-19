#include "panvk_data_graph_csf.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void
emit(const struct panvk_neural_program *program, uint64_t words[32])
{
   struct cs_builder b;
   const struct cs_builder_conf conf = {
      .nr_registers = 96,
      .nr_kernel_registers = 4,
   };
   const struct cs_buffer buffer = {.cpu = words, .capacity = 32};
   cs_builder_init(&b, &conf, buffer);
   panvk_cs_neural_program(&b, program);
   cs_end(&b);
   assert(b.root_chunk.size == 18);
   assert(cs_is_valid(&b));
   cs_builder_fini(&b);
}

int
main(int argc, char **argv)
{
   if (argc == 3) {
      FILE *file = fopen(argv[1], "rb");
      assert(file);
      uint32_t count;
      assert(fread(&count, sizeof(count), 1, file) == 1);
      for (unsigned i = 0; i < count; i++) {
         uint32_t endpoint, mask, r[40];
         uint64_t context[2], words[32] = {0};
         assert(fread(&endpoint, sizeof(endpoint), 1, file) == 1);
         assert(fread(&mask, sizeof(mask), 1, file) == 1);
         assert(fread(context, sizeof(context), 1, file) == 1);
         assert(fread(r, sizeof(r), 1, file) == 1);
         struct panvk_neural_program program = {
            .pipeline_id = r[14], .dispatch_id = r[15],
            .address = context[0] ? context[0] : r[24] | (uint64_t)r[25] << 32,
            .size = context[0] ? context[1] : r[26],
         };
         memcpy(program.offset, r, sizeof(program.offset));
         memcpy(program.extent, r + 4, sizeof(program.extent));
         memcpy(program.loops, r + 8, sizeof(program.loops));
         struct cs_builder b;
         const struct cs_builder_conf conf = {
            .nr_registers = 96, .nr_kernel_registers = 4,
         };
         const struct cs_buffer buffer = {.cpu = words, .capacity = 32};
         cs_builder_init(&b, &conf, buffer);
         panvk_cs_neural_program_deps(&b, &program, endpoint, mask);
         cs_end(&b);
         assert(cs_is_valid(&b));
         assert(b.root_chunk.size == (mask ? 21 : 20));
         assert(b.req_resource_mask == CS_NEURAL_RES);
         assert(fwrite(words, sizeof(*words), b.root_chunk.size, stdout) ==
                b.root_chunk.size);
         cs_builder_fini(&b);
      }
      assert(fgetc(file) == EOF);
      fclose(file);
      return 0;
   }
   if (argc == 2) {
      FILE *file = fopen(argv[1], "rb");
      assert(file);
      uint32_t count;
      assert(fread(&count, sizeof(count), 1, file) == 1);
      for (unsigned i = 0; i < count; i++) {
         uint64_t context[2], words[32] = {0};
         uint32_t r[40];
         assert(fread(context, sizeof(context), 1, file) == 1);
         assert(fread(r, sizeof(r), 1, file) == 1);
         struct panvk_neural_program program = {
            .pipeline_id = r[14],
            .dispatch_id = r[15],
            .address = context[0] ? context[0] : r[24] | (uint64_t)r[25] << 32,
            .size = context[0] ? context[1] : r[26],
         };
         memcpy(program.offset, r, sizeof(program.offset));
         memcpy(program.extent, r + 4, sizeof(program.extent));
         memcpy(program.loops, r + 8, sizeof(program.loops));
         emit(&program, words);
         assert(fwrite(words, sizeof(*words), 18, stdout) == 18);
      }
      assert(fgetc(file) == EOF);
      fclose(file);
      return 0;
   }
   const struct panvk_neural_program program = {
      .extent = {16, 1, 1, 16},
      .address = UINT64_C(0x123456780000),
      .size = 224,
   };
   uint64_t words[32] = {0};
   emit(&program, words);
   const uint64_t expected[] = {
      UINT64_C(0x0208000000000000), UINT64_C(0x0209000000000000),
      UINT64_C(0x020a000000000000), UINT64_C(0x020b000000000000),
      UINT64_C(0x020c000000000010), UINT64_C(0x020d000000000001),
      UINT64_C(0x020e000000000001), UINT64_C(0x020f000000000010),
      UINT64_C(0x0203000000000000), UINT64_C(0x0202000000000000),
      UINT64_C(0x1000000600051010), UINT64_C(0x1001010600000818),
      UINT64_C(0x1001010600001008), UINT64_C(0x0204000000000000),
      UINT64_C(0x0205000000000000), UINT64_C(0x02120000000000e0),
      UINT64_C(0x0210000056780000), UINT64_C(0x021100001c001234),
   };
   assert(!memcmp(words, expected, sizeof(expected)));
   struct cs_builder b;
   const struct cs_builder_conf conf = {.nr_registers = 96, .nr_kernel_registers = 4};
   const struct cs_buffer buffer = {.cpu = words, .capacity = ARRAY_SIZE(words)};
   cs_builder_init(&b, &conf, buffer);
   const uint64_t address[4] = {0x10000, 0x20000, 0x30000, 0x40000};
   const uint32_t size[4] = {64, 128, 192, 256};
   panvk_cs_neural_resources(&b, address, size);
   cs_run_neural(&b, 0, 16, 36);
   cs_end(&b);
   assert(cs_is_valid(&b));
   assert(b.req_resource_mask == CS_NEURAL_RES);
   assert(b.root_chunk.size == 9);
   assert(words[8] == UINT64_C(0x0d00000024100000));
   for (unsigned i = 0; i < 4; i++) {
      assert(words[i] == (UINT64_C(0x0200000000000000) |
             (uint64_t)(22 + i * 4) << 48 | size[i]));
      assert(words[4 + i] == (UINT64_C(0x0100000000000000) |
             (uint64_t)(20 + i * 4) << 48 | address[i]));
   }
   cs_builder_fini(&b);
   memset(words, 0, sizeof(words));
   cs_builder_init(&b, &conf, buffer);
   panvk_cs_neural_invalidate(&b, 0xfffc, cs_reg32(&b, 84), 0);
   cs_end(&b);
   const uint64_t invalidate[] = {
      UINT64_C(0x03000000fffc0000), UINT64_C(0x0254000000000000),
      UINT64_C(0x2400540000008000), UINT64_C(0x0300000000010000),
   };
   assert(cs_is_valid(&b));
   assert(b.root_chunk.size == ARRAY_SIZE(invalidate));
   assert(!memcmp(words, invalidate, sizeof(invalidate)));
   cs_builder_fini(&b);
   return 0;
}
