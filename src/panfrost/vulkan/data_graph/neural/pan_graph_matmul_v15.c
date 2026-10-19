#include "pan_graph_matmul_v15.h"
#include "pan_graph_nrt_plan.h"

#include <string.h>

static uint32_t
schedule_type(const struct pan_graph_matmul_info *info)
{
   static const uint32_t types[4] = {0, 2, 1, 3};
   static const uint32_t blocks[4][3] = {
      {32, 2, 2},
      {8, 4, 4},
      {32, 4, 1},
      {8, 16, 1},
   };
   const uint32_t dimensions[3] = {
      info->columns,
      info->rows,
      1,
   };
   uint64_t best_tiles = UINT64_MAX;
   bool best_small = true;
   float best_waste = 1.0f;
   uint32_t best = 0;
   uint64_t volume = (uint64_t)dimensions[0] * dimensions[1] * dimensions[2];
   for (unsigned i = 0; i < 4; i++) {
      const uint32_t type = types[i];
      uint64_t tiles = 1;
      for (unsigned j = 0; j < 3; j++)
         tiles *=
            ((uint64_t)dimensions[j] + blocks[type][j] - 1) / blocks[type][j];
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

static bool
plan_resources(const struct pan_graph_matmul_info *info,
               const struct pan_graph_nrt_target *target, unsigned adds,
               struct pan_graph_nrt_layout layouts[11])
{
   struct pan_graph_nrt_plan_request requests[10] = {
      {
         .allocation =
            {
               .dimensions = {info->columns, info->inner, 1, 1},
               .element_bits = 8,
               .buffers = 2,
               .front = true,
            },
         .buffers = PAN_GRAPH_NRT_BUFFER_INPUT,
      },
      {
         .allocation =
            {
               .dimensions = {info->inner, info->rows, 1, 1},
               .element_bits = 8,
               .buffers = 2,
               .front = true,
            },
         .buffers = PAN_GRAPH_NRT_BUFFER_INPUT,
      },
      {
         .allocation =
            {
               .dimensions = {info->columns, info->rows, 1, 1},
               .element_bits = 32,
               .buffers = 2,
            },
         .buffers = PAN_GRAPH_NRT_BUFFER_OUTPUT,
      },
      {
         .allocation =
            {
               .type = PAN_GRAPH_NRT_WEIGHT,
               .dimensions = {info->inner, 1, 1, info->columns},
               .element_bits = 8,
               .weight_format = 1,
               .initial_format = 1,
               .buffers = 2,
               .narrow_weight = true,
            },
      },
   };
   for (unsigned i = 0; i < adds; i++) {
      requests[4 + i].allocation = (struct pan_graph_nrt_request){
         .dimensions = {info->columns, info->rows, 1, 1},
         .element_bits = 32,
         .buffers = 2,
      };
   }
   struct pan_graph_nrt_plan plan;
   if (pan_graph_nrt_plan_search(target, requests, 4 + adds, 16, &plan) !=
       PAN_GRAPH_NRT_SUCCESS)
      return false;
   const unsigned resource_ids[4] = {1, 3, 4 + adds, 0};
   for (unsigned i = 0; i < 4 + adds; i++)
      layouts[i < 4 ? resource_ids[i] : i] = plan.layouts[i];

   const struct pan_graph_nrt_accumulator_request accumulator = {
      .dimensions = {info->columns, info->rows, 1, 1},
      .schedule_type = schedule_type(info),
      .producer_mode = 3,
      .input_bits = {8, 8},
   };
   return pan_graph_nrt_accumulator_layout(&accumulator, &layouts[2]);
}

static uint32_t
affine_registers(const struct pan_graph_matmul_affine *affine)
{
   uint32_t count = 0;
   for (unsigned i = 0; i + 1 < affine->rows; i++) {
      for (unsigned j = 0; j + 1 < affine->columns; j++) {
         if ((affine->low[i * affine->columns + j] ||
              affine->high[i * affine->columns + j]) &&
             j >= count)
            count = j + 1;
      }
   }
   return count;
}

enum pan_graph_tsu_status
pan_graph_matmul_v15_compile_bound_program_candidate(
   const struct pan_graph_matmul_plan *plan,
   const struct pan_graph_nrt_target *target, const uint32_t transfer[7],
   const uint32_t loops[4], const struct pan_graph_matmul_epilogue *epilogue,
   const struct pan_graph_matmul_memory *memory,
   struct pan_graph_matmul_kernel *kernel)
{
   if (!plan || !target || !transfer || !loops || !epilogue || !memory ||
       !kernel || (epilogue->count && !epilogue->adds) ||
       !epilogue->output_view_count)
      return PAN_GRAPH_TSU_INVALID;
   if (epilogue->count > 6 || epilogue->output_view_count > 7)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   for (unsigned i = 0; i < epilogue->count; i++) {
      if (epilogue->adds[i].input[0] > i || epilogue->adds[i].input[1] > i)
         return PAN_GRAPH_TSU_INVALID;
   }
   struct pan_graph_matmul_plan canonical;
   if (!pan_graph_matmul_i8_init(&canonical, &plan->info))
      return PAN_GRAPH_TSU_INVALID;
   if (target->shared_size != 65536 || target->alignment != 32 ||
       target->banks != 8 || target->bank_parts != 1)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   plan = &canonical;
   const struct pan_graph_matmul_info *info = &plan->info;
   for (unsigned i = 0; i < 7; i++) {
      if (!transfer[i] || transfer[i] > (uint64_t)plan->conv_high[i] + 1)
         return PAN_GRAPH_TSU_INVALID;
      if (transfer[i] > 4095)
         return PAN_GRAPH_TSU_UNSUPPORTED;
   }
   if ((loops[3] && (!loops[2] || loops[0] == loops[1])) ||
       (loops[2] && (loops[0] >= 4 || loops[2] > INT32_MAX)) ||
       (loops[3] && (loops[1] >= 4 || loops[3] > INT32_MAX)))
      return PAN_GRAPH_TSU_INVALID;
   if (transfer[1] != 1 || transfer[2] != 1 || transfer[5] != 1 ||
       transfer[6] != 1)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct pan_graph_matmul_info tile = *info;
   tile.batch = 1;
   tile.inner = transfer[4];
   tile.rows = transfer[3];
   tile.columns = transfer[0];
   struct pan_graph_nrt_layout layouts[11];
   if (!plan_resources(&tile, target, epilogue->count, layouts))
      return PAN_GRAPH_TSU_UNSUPPORTED;

   struct pan_graph_matmul_affine affines[6];
   struct pan_graph_tsu_transform transforms[6];
   struct pan_graph_tsu_info tsus[6];
   const int64_t low[7] = {0};
   const enum pan_graph_matmul_view views[6] = {
      PAN_GRAPH_MATMUL_READ_B,      PAN_GRAPH_MATMUL_CONV_A,
      PAN_GRAPH_MATMUL_CONV_WEIGHT, PAN_GRAPH_MATMUL_VIEW_COUNT,
      PAN_GRAPH_MATMUL_CONV_OUTPUT, PAN_GRAPH_MATMUL_CONV_OUTPUT,
   };
   const enum pan_graph_v15_section_type types[6] = {
      PAN_GRAPH_V15_MEMORY_INPUT, PAN_GRAPH_V15_MEMORY_INPUT,
      PAN_GRAPH_V15_TRANSFORM,    PAN_GRAPH_V15_CE,
      PAN_GRAPH_V15_VE,           PAN_GRAPH_V15_MEMORY_OUTPUT,
   };
   struct pan_graph_v15_section sections[18] = {0};
   uint32_t section_tsus[18] = {0, 1, 2, 3, 4, 5};
   const uint32_t outputs[6] = {1, 3, 0, 2, 4, UINT32_MAX};
   for (unsigned i = 0; i < 6; i++) {
      tsus[i] = (struct pan_graph_tsu_info){
         .rank = 7,
         .low = low,
         .high = plan->conv_high,
      };
      sections[i].type = types[i];
      struct pan_graph_v15_common *common = &sections[i].common;
      common->output = outputs[i];
      for (unsigned j = 0; j < 6; j++)
         common->input[j] = UINT32_MAX;
      if (views[i] == PAN_GRAPH_MATMUL_VIEW_COUNT) {
         common->registers = 7;
         continue;
      }
      if (!pan_graph_matmul_section_affine(plan, views[i], &affines[i]))
         return PAN_GRAPH_TSU_INVALID;
      transforms[i] = (struct pan_graph_tsu_transform){
         .kind = PAN_GRAPH_TSU_AFFINE,
         .rows = affines[i].rows,
         .columns = affines[i].columns,
         .low = affines[i].low,
         .high = affines[i].high,
      };
      tsus[i].count = 1;
      tsus[i].transforms = &transforms[i];
      common->registers = affine_registers(&affines[i]);
   }
   sections[0].memory = memory->inputs[1];
   sections[1].common.instance = 1;
   sections[1].memory = memory->inputs[0];
   const uint32_t schedule = schedule_type(&tile);
   sections[2].common.input[0] = 1;
   sections[2].transform = (struct pan_graph_v15_transform){
      .format = 8,
      .mode = 2,
      .immediate = info->b_zero_point,
      .plane = schedule,
   };
   sections[3].common.input[0] = 3;
   sections[3].common.input[2] = 0;
   sections[3].ce = (struct pan_graph_v15_ce){
      .opcode = 3,
      .width = info->rows,
      .height = info->batch,
      .zero_point = info->a_zero_point,
      .bias_zero_point = info->b_zero_point,
      .has_bias_zero_point = true,
   };
   for (unsigned i = 0; i < 9; i++)
      sections[3].ce.parameters[i] = 1;
   sections[4].common.input[0] = 2;
   sections[4].rescale = (struct pan_graph_v15_rescale){
      .multiplier = 1,
      .coefficient_binding = UINT32_MAX,
   };
   for (unsigned i = 0; i < epilogue->count; i++) {
      struct pan_graph_v15_section *section = &sections[5 + i];
      *section = (struct pan_graph_v15_section){
         .type = PAN_GRAPH_V15_VE,
         .common = {.registers = sections[4].common.registers,
                    .instance = i + 1,
                    .output = 5 + i},
         .ve_family = PAN_GRAPH_V15_VE_BINARY,
         .binary = {.op = PAN_GRAPH_V15_BINARY_ADD},
      };
      for (unsigned j = 0; j < 6; j++)
         section->common.input[j] =
            j < 2 ? 4 + epilogue->adds[i].input[j] : UINT32_MAX;
      section_tsus[5 + i] = 4;
   }
   for (unsigned i = 0; i < epilogue->output_view_count; i++) {
      unsigned index = 5 + epilogue->count + i;
      struct pan_graph_v15_section *section = &sections[index];
      *section = (struct pan_graph_v15_section){
         .type = PAN_GRAPH_V15_MEMORY_OUTPUT,
         .common = {.registers = affine_registers(&affines[5]),
                    .instance = i,
                    .output = UINT32_MAX},
         .memory = memory->output,
      };
      for (unsigned j = 0; j < 6; j++)
         section->common.input[j] = j ? UINT32_MAX : 4 + epilogue->count;
      section_tsus[index] = 5;
   }

   uint32_t registers[11] = {
      sections[2].common.registers,
      sections[0].common.registers,
      sections[4].common.registers,
      sections[1].common.registers,
   };
   for (unsigned i = 4; i < 5 + epilogue->count; i++)
      registers[i] = sections[4].common.registers;
   struct pan_graph_v15_nrt resources[11] = {0};
   for (unsigned i = 0; i < 5 + epilogue->count; i++) {
      const struct pan_graph_nrt_layout *layout = &layouts[i];
      resources[i] = (struct pan_graph_v15_nrt){
         .kind = layout->kind,
         .width = layout->buffers,
         .size = layout->size,
         .flags = layout->base,
         .tile = layout->banks,
         .mode = i == 0,
         .format = layout->format,
         .binding = registers[i],
         .plane = i == 2 ? schedule : 0,
         .packed = i == 0,
      };
      for (unsigned j = 0; j < 4; j++) {
         uint32_t limit = i == 2 ? 63 : 2047;
         if (layout->dimensions[j] > limit)
            return PAN_GRAPH_TSU_UNSUPPORTED;
         resources[i].dimensions[j] = layout->dimensions[j];
      }
   }
   struct pan_graph_v15_program_info program = {
      .section_count = 5 + epilogue->count + epilogue->output_view_count,
      .sections = sections,
      .section_tsu = section_tsus,
      .resource_count = 5 + epilogue->count,
      .resources = resources,
      .tsu_count = 6,
      .tsu = tsus,
   };
   for (unsigned i = 0; i < 8; i++) {
      program.header.transfer[i] = i < 7 ? transfer[i] : 1;
      program.header.output[i] = i < 7 ? plan->conv_high[i] + 1 : 1;
   }
   struct pan_graph_matmul_kernel result = {
      .extent = {info->columns, 1, info->batch, info->rows},
   };
   memcpy(result.loops, loops, 4 * sizeof(uint32_t));
   result.loops[5] = loops[2] && !loops[0] ? 2 : 0;
   enum pan_graph_tsu_status status =
      pan_graph_v15_compile(&program, &result.program);
   if (status == PAN_GRAPH_TSU_SUCCESS)
      *kernel = result;
   return status;
}

enum pan_graph_tsu_status
pan_graph_matmul_v15_compile_program_candidate(
   const struct pan_graph_matmul_plan *plan,
   const struct pan_graph_nrt_target *target, const uint32_t transfer[7],
   const uint32_t loops[4], const struct pan_graph_matmul_epilogue *epilogue,
   struct pan_graph_matmul_kernel *kernel)
{
   const struct pan_graph_matmul_memory memory = {
      .inputs =
         {
            {.level = 5, .override = 1, .value = 1},
            {.offset = 2, .level = 5, .override = 1, .value = 1},
         },
      .output = {.domain = 3, .level = 5},
   };
   return pan_graph_matmul_v15_compile_bound_program_candidate(
      plan, target, transfer, loops, epilogue, &memory, kernel);
}

enum pan_graph_tsu_status
pan_graph_matmul_v15_compile_candidate(
   const struct pan_graph_matmul_plan *plan,
   const struct pan_graph_nrt_target *target, const uint32_t transfer[7],
   const uint32_t loops[4], struct pan_graph_matmul_kernel *kernel)
{
   const struct pan_graph_matmul_epilogue epilogue = {.output_view_count = 1};
   return pan_graph_matmul_v15_compile_program_candidate(
      plan, target, transfer, loops, &epilogue, kernel);
}

enum pan_graph_tsu_status
pan_graph_matmul_v15_compile(const struct pan_graph_matmul_plan *plan,
                             const struct pan_graph_nrt_target *target,
                             struct pan_graph_matmul_kernel *kernel)
{
   if (!plan || !target || !kernel)
      return PAN_GRAPH_TSU_INVALID;
   struct pan_graph_matmul_plan canonical;
   if (!pan_graph_matmul_i8_init(&canonical, &plan->info))
      return PAN_GRAPH_TSU_INVALID;
   uint32_t transfer[7], loops[4] = {0};
   for (unsigned i = 0; i < 7; i++)
      transfer[i] = canonical.transfer_high[i] + 1;
   if (canonical.info.batch > 1) {
      loops[0] = 2;
      loops[2] = 1;
   }
   return pan_graph_matmul_v15_compile_candidate(&canonical, target, transfer,
                                                 loops, kernel);
}
