#include "pan_graph_conv_v15.h"

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

static uint64_t checks;

static unsigned
partition_coverage(const struct pan_graph_conv_plan *plan,
                   const uint32_t transfer[7], uint32_t budget)
{
   uint32_t loops[4];
   assert(pan_graph_conv_v15_assign_loops(plan, transfer, budget, loops));
   const uint32_t full[4] = {plan->info.output_channels, plan->info.batch,
                            plan->output_height, plan->output_width};
   uint8_t visits[16 * 2 * 7 * 9] = {0};
   unsigned count[2] = {1, 1};
   for (unsigned i = 0; i < 2; i++) {
      if (!loops[2 + i])
         continue;
      unsigned axis = loops[i];
      assert(axis < 4 && loops[2 + i] % transfer[axis] == 0);
      count[i] = (full[axis] + loops[2 + i] - 1) / loops[2 + i];
   }
   assert(!loops[3] || (loops[2] && loops[0] < loops[1]));
   assert(count[0] * count[1] <= budget);
   bool divisible = false;
   for (unsigned axis = 0; axis < 4; axis++)
      divisible |= full[axis] > transfer[axis];
   assert(budget < 2 || !divisible || count[0] * count[1] > 1);
   for (unsigned a = 0; a < count[0]; a++)
   for (unsigned b = 0; b < count[1]; b++) {
      const unsigned group[2] = {a, b};
      unsigned low[4] = {0}, high[4];
      memcpy(high, full, sizeof(high));
      for (unsigned i = 0; i < 2; i++) {
         if (!loops[2 + i])
            continue;
         unsigned axis = loops[i];
         low[axis] = group[i] * loops[2 + i];
         if (low[axis] + loops[2 + i] < high[axis])
            high[axis] = low[axis] + loops[2 + i];
      }
      for (unsigned n = low[1]; n < high[1]; n++)
      for (unsigned y = low[2]; y < high[2]; y++)
      for (unsigned x = low[3]; x < high[3]; x++)
      for (unsigned c = low[0]; c < high[0]; c++) {
         unsigned at = ((n * full[2] + y) * full[3] + x) * full[0] + c;
         assert(at < sizeof(visits));
         visits[at]++;
      }
   }
   unsigned total = full[0] * full[1] * full[2] * full[3];
   for (unsigned i = 0; i < total; i++)
      assert(visits[i] == 1);
   return total;
}

static void
partitions(void)
{
   const unsigned channels[] = {4, 5, 16};
   const unsigned budgets[] = {1, 2, 3, 6, 12};
   unsigned cases = 0;
   uint64_t outputs = 0;
   for (unsigned c = 0; c < 3; c++)
   for (unsigned n = 1; n <= 2; n++)
   for (unsigned h = 1; h <= 7; h += 6)
   for (unsigned w = 1; w <= 9; w += 8) {
      const struct pan_graph_conv_info info = {
         .batch = n, .input_height = h, .input_width = w,
         .input_channels = 3, .output_channels = channels[c],
         .kernel = {1, 1}, .stride = {1, 1}, .dilation = {1, 1},
      };
      struct pan_graph_conv_plan plan;
      assert(pan_graph_conv_i8_init(&plan, &info));
      for (unsigned tc = 1; tc <= channels[c] && tc <= 8; tc *= 2)
      for (unsigned th = 1; th <= h && th <= 4; th *= 2)
      for (unsigned tw = 1; tw <= w && tw <= 8; tw *= 2)
      for (unsigned b = 0; b < 5; b++) {
         const uint32_t transfer[7] = {tc, 1, th, tw, 3, 1, 1};
         outputs += partition_coverage(&plan, transfer, budgets[b]);
         cases++;
      }
   }
   printf("conv partition cases=%u outputs=%llu\n", cases,
          (unsigned long long)outputs);
}

static void
map(const struct pan_graph_conv_tsu *view, const int64_t coordinate[7],
    int64_t low[4], int64_t high[4])
{
   for (unsigned row = 0; row < 4; row++) {
      low[row] = view->matrix[row * 8 + 7];
      for (unsigned col = 0; col < 7; col++)
         low[row] += view->matrix[row * 8 + col] * coordinate[col];
      high[row] = low[row];
   }
   for (unsigned i = 1; i < view->info.count; i++) {
      const struct pan_graph_tsu_transform *t = view->transforms + i;
      assert(t->kind == PAN_GRAPH_TSU_CLAMP);
      unsigned index = t->index;
      low[index] = low[index] < 0 ? 0 : low[index] > t->value ? t->value : low[index];
      high[index] = high[index] > t->value ? t->value : high[index];
   }
}

static void
geometry(void)
{
   const uint32_t padding[][4] = {
      {0, 0, 0, 0}, {1, 1, 1, 1}, {0, 1, 2, 0}, {2, 0, 0, 1},
   };
   for (unsigned kh = 1; kh <= 3; kh++)
   for (unsigned kw = 1; kw <= 3; kw++)
   for (unsigned sy = 1; sy <= 2; sy++)
   for (unsigned sx = 1; sx <= 2; sx++)
   for (unsigned dy = 1; dy <= 2; dy++)
   for (unsigned dx = 1; dx <= 2; dx++)
   for (unsigned p = 0; p < sizeof(padding) / sizeof(padding[0]); p++) {
      const uint32_t *pad = padding[p];
      const struct pan_graph_conv_info info = {
         .batch = 2, .input_height = 11, .input_width = 13,
         .input_channels = 3, .output_channels = 5,
         .kernel = {kh, kw}, .stride = {sy, sx}, .dilation = {dy, dx},
         .padding = {pad[0], pad[1], pad[2], pad[3]}, .input_zero_point = -128,
      };
      struct pan_graph_conv_plan plan;
      assert(pan_graph_conv_i8_init(&plan, &info));
      assert(plan.output_height == (11 + pad[0] + pad[1] - (kh - 1) * dy - 1) / sy + 1);
      assert(plan.output_width == (13 + pad[2] + pad[3] - (kw - 1) * dx - 1) / sx + 1);
      for (unsigned type = 0; type <= PAN_GRAPH_CONV_OUTPUT; type++) {
         struct pan_graph_conv_tsu view;
         assert(pan_graph_conv_i8_view(&plan, type, &view));
         struct pan_graph_tsu_program program = {0};
         assert(pan_graph_tsu_compile(&view.info, &program) == PAN_GRAPH_TSU_SUCCESS);
         pan_graph_tsu_cleanup(&program);
         for (unsigned y = 0; y < plan.output_height; y++)
         for (unsigned x = 0; x < plan.output_width; x++)
         for (unsigned ky = 0; ky < kh; ky++)
         for (unsigned kx = 0; kx < kw; kx++) {
            const int64_t point[7] = {4, 1, y, x, 2, ky, kx};
            int64_t low[4], high[4];
            map(&view, point, low, high);
            int64_t expected[4] = {1, y, x, 4};
            if (type == PAN_GRAPH_CONV_WEIGHT) {
               expected[0] = 4;
               expected[1] = ky;
               expected[2] = kx;
               expected[3] = 2;
            } else if (type == PAN_GRAPH_CONV_INPUT) {
               expected[1] = (int64_t)y * sy + ky * dy - pad[0];
               expected[2] = (int64_t)x * sx + kx * dx - pad[2];
               expected[3] = 2;
            }
            for (unsigned i = 0; i < 4; i++) {
               int64_t l = expected[i], h = expected[i];
               if (type == PAN_GRAPH_CONV_INPUT && (i == 1 || i == 2)) {
                  int64_t limit = i == 1 ? 10 : 12;
                  l = l < 0 ? 0 : l > limit ? limit : l;
                  h = h > limit ? limit : h;
               }
               assert(low[i] == l && high[i] == h);
               checks++;
            }
         }
      }
   }
}

static void
invalid(void)
{
   const struct pan_graph_conv_info valid = {
      .batch = 1, .input_height = 3, .input_width = 3,
      .input_channels = 1, .output_channels = 1,
      .kernel = {3, 3}, .stride = {1, 1}, .dilation = {1, 1},
   };
   struct pan_graph_conv_plan plan;
   assert(!pan_graph_conv_i8_init(NULL, &valid));
   assert(!pan_graph_conv_i8_init(&plan, NULL));
   struct pan_graph_conv_info info = valid;
   info.kernel[0] = 0;
   assert(!pan_graph_conv_i8_init(&plan, &info));
   info = valid;
   info.input_height = 2;
   assert(!pan_graph_conv_i8_init(&plan, &info));
   info = valid;
   info.input_zero_point = 128;
   assert(!pan_graph_conv_i8_init(&plan, &info));
   info = valid;
   info.padding[0] = INT32_MAX;
   assert(!pan_graph_conv_i8_init(&plan, &info));
   assert(pan_graph_conv_i8_init(&plan, &valid));
   uint32_t loops[4] = {1, 2, 3, 4};
   const uint32_t unchanged[4] = {1, 2, 3, 4};
   uint32_t transfer[7] = {1, 1, 1, 1, 1, 3, 3};
   assert(!pan_graph_conv_v15_assign_loops(NULL, transfer, 6, loops));
   assert(!pan_graph_conv_v15_assign_loops(&plan, NULL, 6, loops));
   assert(!pan_graph_conv_v15_assign_loops(&plan, transfer, 0, loops));
   assert(!pan_graph_conv_v15_assign_loops(&plan, transfer, UINT32_MAX, loops));
   assert(!pan_graph_conv_v15_assign_loops(&plan, transfer, 6, NULL));
   transfer[0] = 0;
   assert(!pan_graph_conv_v15_assign_loops(&plan, transfer, 6, loops));
   transfer[0] = 1;
   assert(!memcmp(loops, unchanged, sizeof(loops)));
   struct pan_graph_conv_tsu view;
   assert(!pan_graph_conv_i8_view(&plan, -1, &view));
   plan.high[0]++;
   assert(!pan_graph_conv_i8_view(&plan, PAN_GRAPH_CONV_INPUT, &view));
}

static void
window_packets(void)
{
   const struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   const struct pan_graph_matmul_memory memory = {
      .inputs = {{.level = 5, .override = 1, .value = 1},
                 {.offset = 2, .level = 5, .override = 1, .value = 1}},
      .output = {.domain = 3, .level = 5},
   };
   for (unsigned dy = 1; dy <= 2; dy++)
   for (unsigned dx = 1; dx <= 2; dx++) {
      const struct pan_graph_conv_info info = {
         .batch = 1, .input_height = 7, .input_width = 9,
         .input_channels = 3, .output_channels = 4,
         .kernel = {3, 3}, .stride = {1, 1}, .dilation = {dy, dx},
         .input_zero_point = -128,
      };
      struct pan_graph_conv_plan plan;
      assert(pan_graph_conv_i8_init(&plan, &info));
      const uint32_t transfer[7] = {4, 1, 2, 4, 3, 3, 3};
      const uint32_t loops[4] = {0};
      struct pan_graph_matmul_kernel kernel = {0};
      assert(pan_graph_conv_v15_compile_candidate(&plan, &target, transfer, loops,
                                                 &memory, &kernel) ==
             PAN_GRAPH_TSU_SUCCESS);
      const uint8_t *bytes = kernel.program.data;
      const uint32_t *ce = (const uint32_t *)(bytes + 60);
      assert(ce[1] == 8 && ce[2] == 6);
      assert(ce[3] == (128u | (dx - 1) << 24 | (dy - 1) << 28));
      assert(ce[4] == ((uint32_t)(dx > 1 || dy > 1) << 17 |
                       (uint32_t)(dy > 1) << 18));
      const uint32_t *activation = (const uint32_t *)(bytes + 156);
      unsigned width_blocks = dx == 2 ? 3 : 2;
      unsigned height = dy == 2 ? 7 : 4;
      assert(activation[1] == (16u | width_blocks * height << 16));
      assert(activation[2] == (width_blocks | width_blocks * height << 16));
      assert((activation[0] & 15) == (dy == 2 ? 5 : dx == 2 ? 6 : 7));
      pan_graph_v15_program_cleanup(&kernel.program);
   }
}

static void
packets(void)
{
   const uint32_t shapes[][5] = {
      {696, 320, 12, 32, 2}, {348, 160, 32, 32, 1},
      {348, 160, 32, 32, 2}, {174, 80, 32, 32, 1},
      {174, 80, 32, 64, 2}, {87, 40, 64, 64, 1},
      {87, 40, 64, 32, 1}, {174, 80, 32, 16, 1},
      {174, 80, 48, 32, 1}, {174, 80, 32, 16, 1},
      {348, 160, 32, 16, 1}, {348, 160, 48, 16, 1},
      {348, 160, 16, 16, 1}, {696, 320, 16, 4, 1},
   };
   const struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   const struct pan_graph_matmul_memory memory = {
      .inputs = {{.level = 5, .override = 1, .value = 1},
                 {.offset = 2, .level = 5, .override = 1, .value = 1}},
      .output = {.domain = 3, .level = 5},
   };
   for (unsigned i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
      const uint32_t *s = shapes[i];
      struct pan_graph_conv_info info = {
         .batch = 1, .input_height = s[0], .input_width = s[1],
         .input_channels = s[2], .output_channels = s[3],
         .kernel = {3, 3}, .stride = {s[4], s[4]}, .dilation = {1, 1},
         .padding = {1, s[4] == 1, 1, s[4] == 1}, .input_zero_point = -128,
      };
      struct pan_graph_conv_plan plan;
      assert(pan_graph_conv_i8_init(&plan, &info));
      unsigned compiled = 0;
      for (unsigned co = 4; co <= 32 && co <= s[3]; co *= 2)
      for (unsigned h = 1; h <= 4; h *= 2)
      for (unsigned w = 4; w <= 16; w *= 2) {
         uint32_t transfer[7] = {co, 1, h, w, s[2], 3, 3};
         uint32_t loops[4];
         assert(pan_graph_conv_v15_assign_loops(&plan, transfer, 6, loops));
         struct pan_graph_matmul_kernel kernel = {0};
         enum pan_graph_tsu_status status = pan_graph_conv_v15_compile_candidate(
            &plan, &target, transfer, loops, &memory, &kernel);
         assert(status == PAN_GRAPH_TSU_SUCCESS || status == PAN_GRAPH_TSU_UNSUPPORTED);
         if (status == PAN_GRAPH_TSU_SUCCESS) {
            assert(kernel.program.data && kernel.program.size);
            assert(loops[2] && !memcmp(kernel.loops, loops, sizeof(loops)));
            assert(kernel.extent[0] == s[3] && kernel.extent[1] == 1);
            assert(kernel.extent[2] == plan.output_height &&
                   kernel.extent[3] == plan.output_width);
            pan_graph_v15_program_cleanup(&kernel.program);
            const struct pan_graph_conv_rescale rescale = {
               .parameters = {.mode = 1, .output_zero_point = -128,
                              .clamp = true, .minimum = -128, .maximum = 127},
               .coefficients = {.domain = 2, .level = 5, .override = 1, .value = 1},
            };
            assert(pan_graph_conv_v15_compile_rescale_candidate(
                      &plan, &target, transfer, loops, &memory, &rescale,
                      &kernel) == PAN_GRAPH_TSU_SUCCESS);
            const uint8_t *bytes = kernel.program.data;
            const uint32_t *header = (const uint32_t *)bytes;
            assert((header[0] >> 27 & 15) == 6);
            assert((header[0] >> 2 & 7) == 3);
            const uint32_t *coeff = (const uint32_t *)(bytes + 52);
            assert((coeff[0] & 15) == 1);
            assert((coeff[0] >> 12 & 15) == 0);
            assert((coeff[0] >> 16 & 3) == 2);
            static const uint8_t coefficient_tsu[] = {
               0x0d, 0xe2, 0xe2, 0x62, 0x13, 0xe2, 0xe2, 0x62, 0x63,
            };
            size_t coefficient_offset = 200 + ((coeff[0] >> 4) & 255);
            assert(coefficient_offset + sizeof(coefficient_tsu) <= kernel.program.size);
            assert(!memcmp(bytes + coefficient_offset, coefficient_tsu,
                           sizeof(coefficient_tsu)));
            const uint32_t *ve = (const uint32_t *)(bytes + 96);
            assert(ve[0] & (1u << 27));
            assert(ve[1] == 0x007fff80);
            assert(ve[2] == 0x20018000);
            const uint32_t *output_nrt = (const uint32_t *)(bytes + 188);
            assert((output_nrt[1] & 31) == 16);
            const uint32_t *coefficient_nrt = (const uint32_t *)(bytes + 128);
            assert((coefficient_nrt[0] >> 8 & 3) == 2);
            assert(coefficient_nrt[1] == 16 && coefficient_nrt[2] == 0);
            pan_graph_v15_program_cleanup(&kernel.program);
            struct pan_graph_conv_rescale lookup = rescale;
            lookup.has_table = true;
            lookup.table = (struct pan_graph_v15_memory){
               .domain = 2, .offset = 2, .level = 5, .override = 1, .value = 1,
            };
            assert(pan_graph_conv_v15_compile_rescale_candidate(
                      &plan, &target, transfer, loops, &memory, &lookup,
                      &kernel) == PAN_GRAPH_TSU_SUCCESS);
            bytes = kernel.program.data;
            header = (const uint32_t *)bytes;
            assert((header[0] >> 27 & 15) == 7);
            assert((header[0] >> 2 & 7) == 4);
            const uint32_t *table_mi = (const uint32_t *)(bytes + 60);
            assert((table_mi[0] & 15) == 0);
            assert((table_mi[0] >> 12 & 15) == 1);
            static const uint8_t table_tsu[] = {
               0xe2, 0xe2, 0xe2, 0x62, 0xe4, 0xff, 0xe2, 0xe2, 0x62, 0x63,
            };
            size_t table_offset = 220 + ((table_mi[0] >> 4) & 255);
            assert(table_offset + sizeof(table_tsu) <= kernel.program.size);
            assert(!memcmp(bytes + table_offset, table_tsu, sizeof(table_tsu)));
            ve = (const uint32_t *)(bytes + 104);
            assert((ve[0] >> 28) == 1);
            const uint32_t *table_nrt = (const uint32_t *)(bytes + 148);
            assert(table_nrt[0] == 0x300 && table_nrt[1] == 16 && !table_nrt[2]);
            pan_graph_v15_program_cleanup(&kernel.program);
            compiled++;
         }
      }
      assert(compiled);
      printf("conv layer=%u candidates=%u\n", i, compiled);
   }
}

static void
nearest_packets(void)
{
   const unsigned shapes[3][4] = {{174, 80, 32, 16}, {348, 160, 32, 16}, {696, 320, 16, 4}};
   const char *golden[3] = {
      "8509166c015900dcad0d1a6c01d84f91e2e262070668015f026401d8560a68015f026401d82763",
      "8509166c015901dc5b0d1a6c015900dc9f91e2e262070668015f0264015900dcad0a68015f026401d84f63",
      "8509166c015902dcb70d1a6c015901dc3f91e2e262070668015f0264015901dc5b0a68015f0264015900dc9f63",
   };
   const struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   const struct pan_graph_matmul_memory memory = {
      .inputs = {{.level = 5, .override = 1, .value = 1},
                 {.offset = 2, .level = 5, .override = 1, .value = 1}},
      .output = {.domain = 3, .level = 5},
   };
   for (unsigned i = 0; i < 3; i++) {
      struct pan_graph_conv_info info = {
         .batch = 1, .input_height = shapes[i][0], .input_width = shapes[i][1],
         .input_channels = shapes[i][2], .output_channels = shapes[i][3],
         .kernel = {3, 3}, .stride = {1, 1}, .dilation = {1, 1},
         .padding = {1, 1, 1, 1}, .input_zero_point = -128,
         .input_nearest_2x = true,
      };
      struct pan_graph_conv_plan plan;
      assert(pan_graph_conv_i8_init(&plan, &info));
      uint32_t transfer[7] = {4, 1, 8, 16, shapes[i][2], 3, 3}, loops[4];
      assert(pan_graph_conv_v15_assign_loops(&plan, transfer, 6, loops));
      struct pan_graph_matmul_kernel kernel = {0};
      assert(pan_graph_conv_v15_compile_candidate(
                &plan, &target, transfer, loops, &memory, &kernel) == PAN_GRAPH_TSU_SUCCESS);
      const uint8_t *bytes = kernel.program.data;
      const uint32_t *ce = (const uint32_t *)(bytes + 60);
      assert((ce[4] & 63) == 7);
      for (unsigned j = 0; j < strlen(golden[i]) / 2; j++) {
         unsigned expected;
         assert(sscanf(golden[i] + 2 * j, "%2x", &expected) == 1);
         assert(bytes[180 + j] == expected);
      }
      pan_graph_v15_program_cleanup(&kernel.program);
      info.stride[0] = 2;
      assert(!pan_graph_conv_i8_init(&plan, &info));
   }
}

static void
reduction_packets(void)
{
   const struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   const struct pan_graph_matmul_memory memory = {
      .inputs = {{.level = 5, .override = 1, .value = 1},
                 {.offset = 2, .level = 5, .override = 1, .value = 1}},
      .output = {.domain = 3, .level = 5},
   };
   const struct pan_graph_conv_rescale rescale = {
      .parameters = {.mode = 1, .output_zero_point = -128,
                     .clamp = true, .minimum = -128, .maximum = 127},
      .coefficients = {.domain = 2, .level = 5, .override = 1, .value = 1},
   };
   const unsigned channels[] = {17, 31, 33, 48, 64, 65};
   unsigned count = 0;
   for (unsigned c = 0; c < 6; c++) {
      struct pan_graph_conv_info info = {
         .batch = 1, .input_height = 17, .input_width = 37,
         .input_channels = channels[c], .output_channels = 32,
         .kernel = {3, 3}, .stride = {1, 1}, .dilation = {1, 1},
         .padding = {1, 1, 1, 1}, .input_zero_point = 17,
      };
      struct pan_graph_conv_plan plan;
      assert(pan_graph_conv_i8_init(&plan, &info));
      for (unsigned ci = 8; ci < channels[c] && ci <= 32; ci *= 2) {
         uint32_t transfer[7] = {8, 1, 4, 16, ci, 3, 3}, loops[4];
         assert(pan_graph_conv_v15_assign_loops(&plan, transfer, 6, loops));
         struct pan_graph_matmul_kernel kernel = {0};
         assert(pan_graph_conv_v15_compile_rescale_candidate(
                   &plan, &target, transfer, loops, &memory, &rescale,
                   &kernel) == PAN_GRAPH_TSU_SUCCESS);
         const uint8_t *bytes = kernel.program.data;
         const uint32_t *words = (const uint32_t *)bytes;
         assert(words[1] == channels[c] - 1 && (words[7] & 4095) == ci);
         assert((*(const uint32_t *)(bytes + 68) & 15) == 7);
         assert((*(const uint32_t *)(bytes + 96) & 15) == 4);
         assert((*(const uint32_t *)(bytes + 164) & 0x30f) == 0x104);
         pan_graph_v15_program_cleanup(&kernel.program);
         transfer[4] = 7;
         assert(pan_graph_conv_v15_compile_rescale_candidate(
                   &plan, &target, transfer, loops, &memory, &rescale,
                   &kernel) == PAN_GRAPH_TSU_UNSUPPORTED);
         count++;
      }
   }
   printf("conv partial reduction packets=%u\n", count);
}

static void
ce_cost(void)
{
   const unsigned fixtures[4][12] = {
      {33, 8, 1, 1, 1, 2, 16, 4, 8, 16, 0, 43214},
      {48, 32, 4, 1, 1, 1, 1, 64, 8, 24, 0, 80464},
      {64, 16, 3, 2, 2, 2, 2, 8, 16, 16, 1, 768},
      {48, 32, 4, 5, 1, 1, 1, 4, 32, 24, 1, 1920},
   };
   for (unsigned i = 0; i < 4; i++) {
      const unsigned *s = fixtures[i];
      const struct pan_graph_conv_info info = {
         .batch = 1, .input_height = 97, .input_width = 107,
         .input_channels = s[0], .output_channels = s[1],
         .kernel = {s[2], s[3]}, .stride = {s[4], s[5]}, .dilation = {1, 1},
      };
      struct pan_graph_conv_plan plan;
      assert(pan_graph_conv_i8_init(&plan, &info));
      uint32_t transfer[7] = {s[8], 1, s[6], s[7], s[9], s[2], s[3]};
      uint32_t loops[4] = {0, 0, s[8], 0};
      if (s[10]) {
         loops[0] = 2;
         loops[1] = 3;
         loops[2] = s[6] * 2;
         loops[3] = s[7];
      }
      uint64_t cycles = UINT64_MAX;
      assert(pan_graph_conv_v15_ce_cycles(&plan, transfer, loops, &cycles));
      assert(cycles == s[11]);
      transfer[4] = 0;
      assert(!pan_graph_conv_v15_ce_cycles(&plan, transfer, loops, &cycles));
   }
   printf("conv vendor CE cost fixtures=4\n");
}

int
main(void)
{
   invalid();
   geometry();
   partitions();
   window_packets();
   packets();
   nearest_packets();
   reduction_packets();
   ce_cost();
   printf("graph convolution geometry checks=%llu\n", (unsigned long long)checks);
   return 0;
}
