#include "pan_graph_matmul_v15.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static uint32_t
word(const struct pan_graph_v15_program *program, unsigned offset)
{
   assert(offset + 4 <= program->size);
   const uint8_t *p = (const uint8_t *)program->data + offset;
   return p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
          (uint32_t)p[3] << 24;
}

static void
test_epilogue(void)
{
   const struct pan_graph_matmul_info info = {2, 3, 5, 4, 0, 0};
   const struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   struct pan_graph_matmul_plan plan;
   assert(pan_graph_matmul_i8_init(&plan, &info));
   const uint32_t transfer[7] = {4, 1, 1, 3, 5, 1, 1};
   const uint32_t loops[4] = {2, 0, 1, 0};
   struct pan_graph_matmul_add add = {.input = {0, 0}};
   struct pan_graph_matmul_epilogue epilogue = {
      .adds = &add, .count = 1, .output_view_count = 2,
   };
   struct pan_graph_matmul_kernel kernel;
   assert(pan_graph_matmul_v15_compile_program_candidate(
             &plan, &target, transfer, loops, &epilogue, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(kernel.program.size == 276);
   assert(word(&kernel.program, 0) == 0x30021488);
   assert(word(&kernel.program, 52) == 0xf005b204);
   assert(word(&kernel.program, 60) == 0xf015b204);
   assert(word(&kernel.program, 128) == 0xf2145204);
   assert(word(&kernel.program, 152) == 4);
   assert(word(&kernel.program, 208) == 0x01001414);
   assert(word(&kernel.program, 220) == 0x01001cf4);
   assert(!memcmp(kernel.loops, loops, sizeof(loops)));
   assert(kernel.extent[0] == 4 && kernel.extent[2] == 2 && kernel.extent[3] == 3);
   pan_graph_v15_program_cleanup(&kernel.program);

   struct pan_graph_matmul_kernel unchanged;
   memset(&unchanged, 0xa5, sizeof(unchanged));
   for (unsigned i = 0; i < 6; i++) {
      struct pan_graph_matmul_epilogue bad = epilogue;
      enum pan_graph_tsu_status expected = PAN_GRAPH_TSU_INVALID;
      switch (i) {
      case 0: bad.adds = NULL; break;
      case 1: bad.count = 7; expected = PAN_GRAPH_TSU_UNSUPPORTED; break;
      case 2: bad.output_view_count = 0; break;
      case 3: bad.output_view_count = 8; expected = PAN_GRAPH_TSU_UNSUPPORTED; break;
      case 4: add.input[0] = 1; break;
      case 5: add.input[1] = UINT32_MAX; break;
      }
      kernel = unchanged;
      assert(pan_graph_matmul_v15_compile_program_candidate(
                &plan, &target, transfer, loops, &bad, &kernel) == expected);
      assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
      add = (struct pan_graph_matmul_add){.input = {0, 0}};
   }
   assert(pan_graph_matmul_v15_compile_program_candidate(
             &plan, &target, transfer, loops, NULL, &kernel) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
}

int
main(void)
{
   test_epilogue();
   const struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   struct pan_graph_matmul_info dimensions = {2, 3, 5, 4, 3, -2};
   struct pan_graph_matmul_plan plan;
   assert(pan_graph_matmul_i8_init(&plan, &dimensions));
   struct pan_graph_matmul_kernel kernel = {0};
   assert(pan_graph_matmul_v15_compile(&plan, &target, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(kernel.program.size == 224);
   assert(kernel.extent[0] == 4 && kernel.extent[1] == 1 &&
          kernel.extent[2] == 2 && kernel.extent[3] == 3);
   assert(kernel.loops[0] == 2 && kernel.loops[1] == 0 &&
          kernel.loops[2] == 1 && kernel.loops[3] == 0 &&
          kernel.loops[4] == 0 && kernel.loops[5] == 0);
   assert(word(&kernel.program, 0) == 0x28011448);
   assert(word(&kernel.program, 4) == 4);
   assert(word(&kernel.program, 36) == 0xf2041005);
   assert(word(&kernel.program, 40) == 2);
   assert(word(&kernel.program, 44) == 0xf21430c5);
   assert(word(&kernel.program, 48) == 0);
   assert(word(&kernel.program, 60) == 0x03032087);
   assert(word(&kernel.program, 64) == 2);
   assert(word(&kernel.program, 68) == 1);
   assert(word(&kernel.program, 72) == 0xfe03);
   assert(word(&kernel.program, 84) == 0xbfe);
   const uint32_t registers[5] = {5, 5, 4, 5, 4};
   for (unsigned i = 0; i < 5; i++)
      assert((word(&kernel.program, 120 + 12 * i) & 15) == registers[i]);
   pan_graph_v15_program_cleanup(&kernel.program);
   assert(!kernel.program.data && !kernel.program.size);

   const uint32_t sizes[] = {1, 3, 17, 32};
   const uint32_t batches[] = {1, 2, 3, 4, 17};
   unsigned successes = 0, unsupported = 0;
   for (unsigned bi = 0; bi < 5; bi++) {
      const uint32_t b = batches[bi];
      for (unsigned m = 0; m < 4; m++) {
         for (unsigned k = 0; k < 4; k++) {
            for (unsigned n = 0; n < 4; n++) {
               dimensions = (struct pan_graph_matmul_info){
                  b, sizes[m], sizes[k], sizes[n], -128, 127,
               };
               assert(pan_graph_matmul_i8_init(&plan, &dimensions));
               enum pan_graph_tsu_status status =
                  pan_graph_matmul_v15_compile(&plan, &target, &kernel);
               if (status == PAN_GRAPH_TSU_UNSUPPORTED) {
                  unsupported++;
                  continue;
               }
               assert(status == PAN_GRAPH_TSU_SUCCESS);
               assert(kernel.extent[0] == sizes[n] && kernel.extent[1] == 1 &&
                      kernel.extent[2] == b && kernel.extent[3] == sizes[m]);
               assert(kernel.loops[0] == (b > 1 ? 2 : 0));
               assert(kernel.loops[2] == (b > 1));
               assert(word(&kernel.program, 64) == sizes[m] - 1);
               assert(word(&kernel.program, 68) == b - 1);
               assert(word(&kernel.program, 72) == 0x7f80);
               assert((word(&kernel.program, 84) & 255) == 127);
               pan_graph_v15_program_cleanup(&kernel.program);
               successes++;
            }
         }
      }
   }

   dimensions = (struct pan_graph_matmul_info){1, 32, 32, 32, 0, 0};
   assert(pan_graph_matmul_i8_init(&plan, &dimensions));
   struct pan_graph_matmul_kernel full = {0};
   assert(pan_graph_matmul_v15_compile(&plan, &target, &full) ==
          PAN_GRAPH_TSU_SUCCESS);
   uint32_t transfer[7] = {32, 1, 1, 8, 32, 1, 1};
   uint32_t loops[4] = {3, 0, 8, 0};
   assert(pan_graph_matmul_v15_compile_candidate(&plan, &target, transfer,
                                                 loops, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(kernel.program.size == 224 && kernel.extent[3] == 32);
   assert(kernel.loops[0] == 3 && kernel.loops[2] == 8);
   assert(!kernel.loops[4] && !kernel.loops[5]);
   assert(word(&kernel.program, 24) == 0x41008001);
   assert(word(&kernel.program, 120) == 0x1f001415);
   assert(word(&kernel.program, 132) == 0x0f0020f5);
   assert(word(&kernel.program, 152) == 0x02010002);
   assert(word(&kernel.program, 156) == 0x070008f5);
   assert(!memcmp((const uint8_t *)full.program.data + 180,
                  (const uint8_t *)kernel.program.data + 180, 44));
   pan_graph_v15_program_cleanup(&kernel.program);
   pan_graph_v15_program_cleanup(&full.program);
   const uint32_t tile_rows[] = {16, 8, 4, 2};
   const uint32_t tile_steps[] = {16, 8, 8, 6};
   for (unsigned i = 0; i < 4; i++) {
      transfer[3] = tile_rows[i];
      loops[2] = tile_steps[i];
      assert(pan_graph_matmul_v15_compile_candidate(&plan, &target, transfer,
                                                    loops, &kernel) ==
             PAN_GRAPH_TSU_SUCCESS);
      assert(kernel.loops[2] == tile_steps[i]);
      pan_graph_v15_program_cleanup(&kernel.program);
   }
   dimensions = (struct pan_graph_matmul_info){2, 3, 5, 4, -128, 127};
   assert(pan_graph_matmul_i8_init(&plan, &dimensions));
   transfer[0] = 2;
   transfer[3] = 3;
   transfer[4] = 5;
   loops[0] = 0;
   loops[1] = 2;
   loops[2] = 2;
   loops[3] = 1;
   assert(pan_graph_matmul_v15_compile_candidate(&plan, &target, transfer,
                                                 loops, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(kernel.extent[0] == 4 && kernel.extent[2] == 2);
   assert(!memcmp(kernel.loops, loops, sizeof(loops)));
   assert(!kernel.loops[4] && kernel.loops[5] == 2);
   pan_graph_v15_program_cleanup(&kernel.program);
   transfer[4] = 4;
   assert(pan_graph_matmul_v15_compile_candidate(&plan, &target, transfer,
                                                 loops, &kernel) ==
          PAN_GRAPH_TSU_SUCCESS);
   assert(word(&kernel.program, 4) == 4);
   assert((word(&kernel.program, 28) & 4095) == 4);
   assert(!memcmp(kernel.loops, loops, sizeof(loops)));
   pan_graph_v15_program_cleanup(&kernel.program);
   transfer[4] = 5;

   struct pan_graph_matmul_info reduction = {1, 1, 8193, 1, 0, 0};
   struct pan_graph_matmul_plan reduction_plan;
   assert(pan_graph_matmul_i8_init(&reduction_plan, &reduction));
   uint32_t reduction_transfer[7] = {1, 1, 1, 1, 1024, 1, 1};
   const uint32_t reduction_loops[4] = {0};
   assert(pan_graph_matmul_v15_compile_candidate(
             &reduction_plan, &target, reduction_transfer, reduction_loops,
             &kernel) == PAN_GRAPH_TSU_SUCCESS);
   assert(kernel.program.size == 224);
   assert(word(&kernel.program, 4) == 8192);
   assert((word(&kernel.program, 28) & 4095) == 1024);
   assert(kernel.extent[0] == 1 && kernel.extent[2] == 1);
   assert(!memcmp(kernel.loops, reduction_loops, sizeof(reduction_loops)));
   pan_graph_v15_program_cleanup(&kernel.program);

   const struct pan_graph_matmul_info large[] = {
      {8193, 1, 1, 1, 0, 0}, {1, 8193, 1, 1, 0, 0},
      {1, 1, 1, 8193, 0, 0},
   };
   const uint32_t large_transfers[][7] = {
      {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 128, 1, 1, 1},
      {683, 1, 1, 1, 1, 1, 1},
   };
   const uint32_t large_loops[][4] = {
      {2, 0, 1366, 0}, {3, 0, 1408, 0}, {0, 0, 1366, 0},
   };
   for (unsigned i = 0; i < 3; i++) {
      struct pan_graph_matmul_plan large_plan;
      assert(pan_graph_matmul_i8_init(&large_plan, &large[i]));
      assert(pan_graph_matmul_v15_compile_candidate(
                &large_plan, &target, large_transfers[i], large_loops[i],
                &kernel) == PAN_GRAPH_TSU_SUCCESS);
      assert(kernel.program.size == 224);
      assert(word(&kernel.program, 64) == large[i].rows - 1);
      assert(word(&kernel.program, 68) == large[i].batch - 1);
      assert(kernel.extent[0] == large[i].columns &&
             kernel.extent[2] == large[i].batch &&
             kernel.extent[3] == large[i].rows);
      assert(!memcmp(kernel.loops, large_loops[i], sizeof(large_loops[i])));
      pan_graph_v15_program_cleanup(&kernel.program);
   }

   struct pan_graph_matmul_kernel unchanged;
   memset(&unchanged, 0xa5, sizeof(unchanged));
   kernel = unchanged;
   for (unsigned i = 0; i < 7; i++) {
      uint32_t saved = transfer[i];
      transfer[i] = 0;
      assert(pan_graph_matmul_v15_compile_candidate(&plan, &target, transfer,
                                                    loops, &kernel) ==
             PAN_GRAPH_TSU_INVALID);
      assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
      transfer[i] = UINT32_MAX;
      assert(pan_graph_matmul_v15_compile_candidate(&plan, &target, transfer,
                                                    loops, &kernel) ==
             PAN_GRAPH_TSU_INVALID);
      assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
      transfer[i] = saved;
   }
   transfer[2] = 2;
   assert(pan_graph_matmul_v15_compile_candidate(&plan, &target, transfer,
                                                 loops, &kernel) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
   transfer[2] = 1;
   reduction_transfer[4] = 4096;
   assert(pan_graph_matmul_v15_compile_candidate(
             &reduction_plan, &target, reduction_transfer, reduction_loops,
             &kernel) == PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
   loops[1] = 0;
   assert(pan_graph_matmul_v15_compile_candidate(&plan, &target, transfer,
                                                 loops, &kernel) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
   loops[1] = 7;
   assert(pan_graph_matmul_v15_compile_candidate(&plan, &target, transfer,
                                                 loops, &kernel) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
   loops[1] = 2;
   assert(pan_graph_matmul_v15_compile_candidate(&plan, &target, NULL,
                                                 loops, &kernel) ==
          PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_matmul_v15_compile_candidate(&plan, &target, transfer,
                                                 NULL, &kernel) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
   dimensions = (struct pan_graph_matmul_info){1, 4095, 4095, 4095, 0, 0};
   assert(pan_graph_matmul_i8_init(&plan, &dimensions));
   assert(pan_graph_matmul_v15_compile(&plan, &target, &kernel) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
   dimensions = (struct pan_graph_matmul_info){1, 1, 1, 4096, 0, 0};
   assert(pan_graph_matmul_i8_init(&plan, &dimensions));
   assert(pan_graph_matmul_v15_compile(&plan, &target, &kernel) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
   struct pan_graph_nrt_target other = target;
   other.alignment = 64;
   dimensions = (struct pan_graph_matmul_info){1, 1, 1, 1, 0, 0};
   assert(pan_graph_matmul_i8_init(&plan, &dimensions));
   assert(pan_graph_matmul_v15_compile(&plan, &other, &kernel) ==
          PAN_GRAPH_TSU_UNSUPPORTED);
   assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
   plan.info.a_zero_point = 128;
   assert(pan_graph_matmul_v15_compile(&plan, &target, &kernel) ==
          PAN_GRAPH_TSU_INVALID);
   assert(!memcmp(&kernel, &unchanged, sizeof(kernel)));
   assert(pan_graph_matmul_v15_compile(NULL, &target, &kernel) ==
          PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_matmul_v15_compile(&plan, NULL, &kernel) ==
          PAN_GRAPH_TSU_INVALID);
   assert(pan_graph_matmul_v15_compile(&plan, &target, NULL) ==
          PAN_GRAPH_TSU_INVALID);
   printf("MatMul v15 semantic frontend: %u grid successes, %u unsupported; "
          "exact field vectors, quantization, single-batch loops and "
          "explicit tiling, two-axis loops and transactional rejection passed\n",
          successes, unsupported);
   return 0;
}
