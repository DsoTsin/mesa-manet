#include "pan_graph_conv_v15.h"
#include "pan_graph_matmul_schedule.h"
#include "pan_graph_nrt_plan.h"

#include <limits.h>
#include <math.h>
#include <string.h>

bool
pan_graph_conv_v15_assign_loops(
   const struct pan_graph_conv_plan *plan, const uint32_t transfer[7],
   uint32_t core_budget, uint32_t loops[4])
{
   struct pan_graph_conv_plan canonical;
   if (!plan || !core_budget ||
       !pan_graph_conv_i8_init(&canonical, &plan->info) ||
       memcmp(canonical.high, plan->high, sizeof(plan->high)))
      return false;
   uint32_t dimensions[7];
   for (unsigned i = 0; i < 7; i++)
      dimensions[i] = canonical.high[i] + 1;
   const uint32_t granule[7] = {1, 1, 1, 1, 1, 1, 1};
   const struct pan_graph_matmul_parallel_options options = {
      .core_budget = core_budget, .output_axis_mask = 15, .multiple_axes = true,
   };
   return pan_graph_assign_loops(dimensions, transfer, granule, 15,
                                 &options, loops);
}

static uint32_t
schedule_type(const uint32_t dimensions[3])
{
   static const uint32_t types[4] = {0, 2, 1, 3};
   static const uint32_t blocks[4][3] = {
      {32, 2, 2}, {8, 4, 4}, {32, 4, 1}, {8, 16, 1},
   };
   uint64_t best_tiles = UINT64_MAX;
   bool best_small = true;
   float best_waste = 1.0f;
   uint32_t best = 0;
   uint64_t volume = (uint64_t)dimensions[0] * dimensions[1] * dimensions[2];
   for (unsigned i = 0; i < 4; i++) {
      uint32_t type = types[i];
      uint64_t tiles = 1;
      for (unsigned j = 0; j < 3; j++)
         tiles *= ((uint64_t)dimensions[j] + blocks[type][j] - 1) /
                  blocks[type][j];
      bool small = type == 1 || type == 3;
      float waste = 1.0f - (float)volume / (float)(tiles * 128);
      if (tiles < best_tiles ||
          (tiles == best_tiles &&
           (small < best_small ||
            (small == best_small && waste < best_waste)))) {
         best_tiles = tiles;
         best_small = small;
         best_waste = waste;
         best = type;
      }
   }
   return best;
}

bool
pan_graph_conv_v15_ce_cycles(const struct pan_graph_conv_plan *plan,
                            const uint32_t transfer[7],
                            const uint32_t loops[4], uint64_t *cycles)
{
   struct pan_graph_conv_plan canonical;
   if (!plan || !transfer || !loops || !cycles ||
       !pan_graph_conv_i8_init(&canonical, &plan->info) ||
       memcmp(canonical.high, plan->high, sizeof(plan->high)) ||
       plan->info.input_nearest_2x ||
       (loops[2] && loops[0] >= 4) ||
       (loops[3] && (!loops[2] || loops[1] >= 4 || loops[0] == loops[1])))
      return false;
   const struct pan_graph_conv_info *info = &plan->info;
   for (unsigned i = 0; i < 2; i++) {
      if (info->kernel[i] > 5 || info->dilation[i] != 1 || info->stride[i] > 2)
         return false;
   }
   float replication = 1.0f;
   for (unsigned i = 0; i < 7; i++) {
      uint32_t extent = canonical.high[i] + 1;
      if (!transfer[i] || transfer[i] > extent || transfer[i] > 4095)
         return false;
      for (unsigned j = 0; j < 2; j++) {
         if (loops[j + 2] && loops[j] == i && extent > loops[j + 2])
            extent = loops[j + 2];
      }
      volatile float product = replication * ((float)extent / transfer[i]);
      replication = product;
   }
   if (transfer[5] != info->kernel[0] || transfer[6] != info->kernel[1])
      return false;
   static const uint32_t blocks[4][3] = {
      {32, 2, 2}, {8, 4, 4}, {32, 4, 1}, {8, 16, 1},
   };
   uint32_t tile[3] = {transfer[0], transfer[3], transfer[2]};
   const uint32_t *block = blocks[schedule_type(tile)];
   uint64_t area = (uint64_t)info->kernel[0] * info->kernel[1];
   uint64_t factor = (area + 7) / area, pack = factor * 8;
   uint64_t compute = factor * area > 8 ? factor * area : 8;
   compute *= ((uint64_t)transfer[3] + block[1] - 1) / block[1];
   compute *= ((uint64_t)transfer[2] + block[2] - 1) / block[2];
   uint64_t weight = pack * area * block[0] / 32;
   uint64_t load = (info->kernel[1] +
                    (uint64_t)info->stride[1] * (transfer[3] - 1) + 3) / 4;
   load *= (pack + 7) / 8;
   load *= info->kernel[0] + (uint64_t)info->stride[0] * (transfer[2] - 1);
   uint64_t base = compute > weight ? compute : weight;
   base *= ((uint64_t)transfer[0] + block[0] - 1) / block[0];
   base = base > load ? base : load;
   base *= ((uint64_t)transfer[4] + pack - 1) / pack;
   volatile float value = replication * (float)base;
   float rounded = ceilf(value);
   if (!isfinite(rounded) || rounded < 0.0f || rounded >= 0x1p63f)
      return false;
   *cycles = rounded;
   return true;
}

static bool
window_modes(const struct pan_graph_conv_info *info, bool modes[2])
{
   uint64_t height = (uint64_t)info->kernel[0] * info->dilation[0];
   uint64_t width = (uint64_t)info->kernel[1] * info->dilation[1];
   uint32_t height_limit = info->dilation[0] > 1 ? info->dilation[0] : 5;
   bool height_window = height > height_limit && height < 17;
   uint32_t width_limit = height_window ? 0 :
      info->dilation[1] > 1 ? info->dilation[1] : 5;
   modes[1] = width > width_limit && width < 17;
   modes[0] = height_window && modes[1];
   for (unsigned axis = 0; axis < 2; axis++)
      if (info->dilation[axis] > 1 && info->kernel[axis] > 1 && !modes[axis])
         return false;
   return true;
}

static bool
resources(const struct pan_graph_conv_plan *plan,
          const struct pan_graph_nrt_target *target, const uint32_t transfer[7],
          uint32_t schedule, const bool windows[2], unsigned output_bits,
          struct pan_graph_nrt_layout layouts[5])
{
   const struct pan_graph_conv_info *info = &plan->info;
   uint64_t width = (uint64_t)(transfer[3] - 1) * info->stride[1] +
                    (uint64_t)(transfer[6] - 1) * info->dilation[1] + 1;
   uint64_t height = (uint64_t)(transfer[2] - 1) * info->stride[0] +
                     (uint64_t)(transfer[5] - 1) * info->dilation[0] + 1;
   if (windows[1])
      width += info->dilation[1] - 1;
   if (windows[0])
      height += info->dilation[0] - 1;
   uint32_t input_width = info->input_width, input_height = info->input_height;
   if (info->input_nearest_2x) {
      width = width / 2 + 3;
      height = height / 2 + 3;
      input_width /= 2;
      input_height /= 2;
   }
   width = width < input_width ? width : input_width;
   height = height < input_height ? height : input_height;
   struct pan_graph_nrt_plan_request requests[4] = {
      {
         .allocation = {
            .dimensions = {transfer[4], transfer[6], transfer[5], transfer[0]},
            .element_bits = 8, .buffers = 2, .front = true,
         },
         .buffers = PAN_GRAPH_NRT_BUFFER_INPUT,
      },
      {
         .allocation = {
            .dimensions = {transfer[4], width, height, 1},
            .element_bits = 8, .buffers = 2, .front = true,
         },
         .buffers = PAN_GRAPH_NRT_BUFFER_INPUT,
      },
      {
         .allocation = {
            .dimensions = {transfer[0], transfer[3], transfer[2], 1},
            .element_bits = output_bits, .buffers = 2,
         },
         .buffers = PAN_GRAPH_NRT_BUFFER_OUTPUT,
      },
      {
         .allocation = {
            .type = PAN_GRAPH_NRT_WEIGHT,
            .dimensions = {transfer[4], transfer[6], transfer[5], transfer[0]},
            .element_bits = 8, .weight_format = 1, .initial_format = 1,
            .buffers = 2, .narrow_weight = true,
         },
      },
   };
   struct pan_graph_nrt_plan allocation;
   if (pan_graph_nrt_plan_search(target, requests, 4, 16, &allocation) !=
       PAN_GRAPH_NRT_SUCCESS)
      return false;
   const unsigned indices[4] = {1, 3, 4, 0};
   for (unsigned i = 0; i < 4; i++)
      layouts[indices[i]] = allocation.layouts[i];
   const struct pan_graph_nrt_accumulator_request accumulator = {
      .dimensions = {transfer[0], transfer[3], transfer[2], 1},
      .schedule_type = schedule, .producer_mode = 3, .input_bits = {8, 8},
   };
   return pan_graph_nrt_accumulator_layout(&accumulator, layouts + 2);
}

static unsigned
registers(const struct pan_graph_conv_tsu *view)
{
   unsigned count = 0;
   for (unsigned row = 0; row < 4; row++)
      for (unsigned col = 0; col < 7; col++)
         if (view->matrix[row * 8 + col] && col >= count)
            count = col + 1;
   return count;
}

enum pan_graph_tsu_status
pan_graph_conv_v15_compile_rescale_candidate(
   const struct pan_graph_conv_plan *plan,
   const struct pan_graph_nrt_target *target, const uint32_t transfer[7],
   const uint32_t loops[4], const struct pan_graph_matmul_memory *memory,
   const struct pan_graph_conv_rescale *rescale,
   struct pan_graph_matmul_kernel *kernel)
{
   if (!plan || !target || !transfer || !loops || !memory || !kernel)
      return PAN_GRAPH_TSU_INVALID;
   if (rescale && (rescale->parameters.mode > 1 ||
                   rescale->parameters.precision ||
                   rescale->parameters.input_zero_point))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct pan_graph_conv_plan canonical;
   if (!pan_graph_conv_i8_init(&canonical, &plan->info) ||
       memcmp(canonical.high, plan->high, sizeof(plan->high)))
      return PAN_GRAPH_TSU_INVALID;
   plan = &canonical;
   const struct pan_graph_conv_info *info = &plan->info;
   bool table = rescale && rescale->has_table;
   if (target->shared_size != 65536 || target->alignment != 32 ||
       target->banks != 8 || target->bank_parts != 1 ||
       info->weight_zero_point || transfer[1] != 1 ||
       (transfer[4] != info->input_channels &&
        ((transfer[4] % 8) || info->input_nearest_2x)) ||
       transfer[5] != info->kernel[0] || transfer[6] != info->kernel[1])
      return PAN_GRAPH_TSU_UNSUPPORTED;
   for (unsigned i = 0; i < 7; i++) {
      if (!transfer[i] || transfer[i] > (uint64_t)plan->high[i] + 1)
         return PAN_GRAPH_TSU_INVALID;
      if (transfer[i] > 4095)
         return PAN_GRAPH_TSU_UNSUPPORTED;
   }
   for (unsigned i = 0; i < 2; i++)
      if (info->stride[i] > 16 || info->dilation[i] > 16 ||
          info->padding[i * 2] > 31)
         return PAN_GRAPH_TSU_UNSUPPORTED;
   if ((loops[3] && (!loops[2] || loops[0] == loops[1])) ||
       (loops[2] && (loops[0] >= 4 || loops[2] > INT32_MAX)) ||
       (loops[3] && (loops[1] >= 4 || loops[3] > INT32_MAX)))
      return PAN_GRAPH_TSU_INVALID;
   const uint32_t tile[3] = {transfer[0], transfer[3], transfer[2]};
   uint32_t schedule = schedule_type(tile);
   bool windows[2];
   if (!window_modes(info, windows))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct pan_graph_nrt_layout layouts[7];
   if (!resources(plan, target, transfer, schedule, windows,
                  rescale ? 8 : 32, layouts))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   if (rescale)
      pan_graph_nrt_kind2_layout(true, layouts + 5);
   if (table)
      pan_graph_nrt_kind3_layout(8, 0, layouts + 6);
   struct pan_graph_conv_tsu input, weight, output;
   if (!pan_graph_conv_i8_view(plan, PAN_GRAPH_CONV_INPUT, &input) ||
       !pan_graph_conv_i8_view(plan, PAN_GRAPH_CONV_WEIGHT, &weight) ||
       !pan_graph_conv_i8_view(plan, PAN_GRAPH_CONV_OUTPUT, &output))
      return PAN_GRAPH_TSU_INVALID;
   int64_t input_high[5 * 8];
   memcpy(input_high, input.matrix, sizeof(input_high));
   for (unsigned axis = 0; axis < 2; axis++) {
      if (!windows[axis])
         continue;
      unsigned row = axis + 1;
      input.matrix[row * 8 + 5 + axis] = 0;
      input_high[row * 8 + 5 + axis] = 0;
      input_high[row * 8 + 7] +=
         (int64_t)info->kernel[axis] * info->dilation[axis] - 1;
   }
   input.transforms[0].high = input_high;
   int64_t coefficient_matrix[5 * 8] = {0};
   coefficient_matrix[3 * 8] = 1;
   coefficient_matrix[4 * 8 + 7] = 1;
   const struct pan_graph_tsu_transform coefficient_transform = {
      .kind = PAN_GRAPH_TSU_AFFINE, .rows = 5, .columns = 8,
      .low = coefficient_matrix, .high = coefficient_matrix,
   };
   int64_t table_low[5 * 8] = {0}, table_high[5 * 8] = {0};
   table_low[4 * 8 + 7] = table_high[4 * 8 + 7] = 1;
   table_high[3 * 8 + 7] = 255;
   const struct pan_graph_tsu_transform table_transform = {
      .kind = PAN_GRAPH_TSU_AFFINE, .rows = 5, .columns = 8,
      .low = table_low, .high = table_high,
   };
   const struct pan_graph_tsu_info tsus[6] = {
      input.info, weight.info,
      {.rank = 7, .low = input.low, .high = input.high}, output.info,
      {.rank = 7, .low = input.low, .high = input.high,
       .count = 1, .transforms = &coefficient_transform},
      {.rank = 7, .low = input.low, .high = input.high,
       .count = 1, .transforms = &table_transform},
   };
   const unsigned section_tsus[8] = {1, 0, 1, 2, 3, 3, 4, 5};
   const enum pan_graph_v15_section_type types[8] = {
      PAN_GRAPH_V15_MEMORY_INPUT, PAN_GRAPH_V15_MEMORY_INPUT,
      PAN_GRAPH_V15_TRANSFORM, PAN_GRAPH_V15_CE,
      PAN_GRAPH_V15_VE, PAN_GRAPH_V15_MEMORY_OUTPUT,
      PAN_GRAPH_V15_MEMORY_INPUT,
      PAN_GRAPH_V15_MEMORY_INPUT,
   };
   const uint32_t outputs[8] = {1, 3, 0, 2, 4, UINT32_MAX, 5, 6};
   const uint32_t counts[8] = {
      registers(&weight), registers(&input), registers(&weight),
      7, registers(&output), registers(&output), 1, 0,
   };
   struct pan_graph_v15_section sections[8] = {0};
   for (unsigned i = 0; i < 6u + !!rescale + table; i++) {
      sections[i].type = types[i];
      sections[i].common.registers = counts[i];
      sections[i].common.output = outputs[i];
      for (unsigned j = 0; j < 6; j++)
         sections[i].common.input[j] = UINT32_MAX;
   }
   sections[0].memory = memory->inputs[1];
   sections[1].common.instance = 1;
   sections[1].memory = memory->inputs[0];
   sections[2].common.input[0] = 1;
   sections[2].transform = (struct pan_graph_v15_transform){
      .format = 8, .mode = 0, .immediate = info->weight_zero_point,
      .plane = schedule,
   };
   sections[3].common.input[0] = 3;
   sections[3].common.input[2] = 0;
   sections[3].ce = (struct pan_graph_v15_ce){
      .opcode = 3, .width = info->input_width, .height = info->input_height,
      .zero_point = info->input_zero_point,
      .input_nearest_2x = info->input_nearest_2x,
      .bias_zero_point = info->weight_zero_point, .has_bias_zero_point = true,
      .flags = {windows[0], windows[1]},
      .parameters = {
         info->kernel[1], info->kernel[0], 1,
         info->stride[1], info->stride[0], 1,
         info->dilation[1], info->dilation[0], 1,
         info->padding[0], info->padding[2], 0, info->padding[1],
      },
   };
   sections[4].common.input[0] = 2;
   sections[4].rescale = (struct pan_graph_v15_rescale){
      .multiplier = 1, .coefficient_binding = UINT32_MAX,
   };
   if (rescale) {
      sections[4].rescale = rescale->parameters;
      sections[4].rescale.coefficient_binding = 5;
      sections[4].common.input[3] = 5;
      sections[6].common.instance = 2;
      sections[6].memory = rescale->coefficients;
   }
   if (table) {
      sections[4].common.input[4] = 6;
      sections[7].common.instance = 3;
      sections[7].memory = rescale->table;
   }
   sections[5].common.input[0] = 4;
   sections[5].memory = memory->output;
   const unsigned bindings[7] = {counts[2], counts[0], counts[4], counts[1], counts[4], 1, 0};
   struct pan_graph_v15_nrt nrt[7] = {0};
   for (unsigned i = 0; i < 5u + !!rescale + table; i++) {
      const struct pan_graph_nrt_layout *layout = layouts + i;
      nrt[i] = (struct pan_graph_v15_nrt){
         .kind = layout->kind, .width = layout->buffers, .size = layout->size,
         .flags = layout->base, .tile = layout->banks, .mode = i == 0,
         .format = layout->format, .binding = bindings[i],
         .plane = i == 2 ? schedule : 0, .packed = i == 0,
      };
      for (unsigned j = 0; j < 4; j++) {
         if (layout->dimensions[j] > (i == 2 ? 63 : 2047))
            return PAN_GRAPH_TSU_UNSUPPORTED;
         nrt[i].dimensions[j] = layout->dimensions[j];
      }
      for (unsigned j = 0; j < 8; j++)
         nrt[i].transfer[j] = j < 7 ? transfer[j] : 1;
   }
   if (rescale) {
      struct pan_graph_v15_nrt coefficient = nrt[5];
      struct pan_graph_v15_nrt lookup = table ? nrt[6] : (struct pan_graph_v15_nrt){0};
      unsigned first = 1 + table;
      memmove(nrt + first, nrt, 5 * sizeof(*nrt));
      nrt[0] = coefficient;
      if (table)
         nrt[1] = lookup;
      for (unsigned i = 0; i < 7u + table; i++) {
         struct pan_graph_v15_common *common = &sections[i].common;
         if (common->output != UINT32_MAX)
            common->output = common->output >= 5 ? common->output - 5 : common->output + first;
         for (unsigned j = 0; j < 6; j++) {
            if (common->input[j] != UINT32_MAX)
               common->input[j] = common->input[j] >= 5 ? common->input[j] - 5 : common->input[j] + first;
         }
      }
      sections[4].rescale.coefficient_binding = 0;
   }
   struct pan_graph_v15_program_info program = {
      .section_count = 6 + !!rescale + table, .sections = sections, .section_tsu = section_tsus,
      .resource_count = 5 + !!rescale + table, .resources = nrt,
      .tsu_count = 4 + !!rescale + table, .tsu = tsus,
   };
   for (unsigned i = 0; i < 8; i++) {
      program.header.transfer[i] = i < 7 ? transfer[i] : 1;
      program.header.output[i] = i < 7 ? plan->high[i] + 1 : 1;
   }
   struct pan_graph_matmul_kernel result = {
      .extent = {info->output_channels, info->batch,
                 plan->output_height, plan->output_width},
   };
   memcpy(result.loops, loops, 4 * sizeof(uint32_t));
   result.loops[5] = loops[2] && !loops[0] ? 2 : 0;
   enum pan_graph_tsu_status status = pan_graph_v15_compile(&program, &result.program);
   if (status == PAN_GRAPH_TSU_SUCCESS)
      *kernel = result;
   return status;
}

enum pan_graph_tsu_status
pan_graph_conv_v15_compile_candidate(
   const struct pan_graph_conv_plan *plan,
   const struct pan_graph_nrt_target *target, const uint32_t transfer[7],
   const uint32_t loops[4], const struct pan_graph_matmul_memory *memory,
   struct pan_graph_matmul_kernel *kernel)
{
   return pan_graph_conv_v15_compile_rescale_candidate(
      plan, target, transfer, loops, memory, NULL, kernel);
}
