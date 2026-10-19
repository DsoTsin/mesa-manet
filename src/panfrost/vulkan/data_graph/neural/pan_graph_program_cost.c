#include "pan_graph_program_cost.h"
#include "pan_graph_matmul_cost.h"
#include "pan_graph_memory_layout.h"
#include "pan_graph_program_boundary.h"
#include "pan_graph_vector_cost.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

bool
pan_graph_program_cost_target_v15_init(
   uint32_t l2_slices, uint32_t l2_line_size, uint32_t l2_size_per_slice,
   uint32_t core_budget, struct pan_graph_memory_cost_target *target)
{
   struct pan_graph_matmul_cost_target memory;
   if (!target || !core_budget || core_budget > INT32_MAX ||
       !pan_graph_matmul_cost_target_v15_init(
          l2_slices, l2_line_size, l2_size_per_slice, &memory))
      return false;
   struct pan_graph_memory_cost_target next = {
      .descriptor_bytes = {36, 64, 32, 12},
      .section_bytes = {0, 8, 20, 32, 8, 8, 8, 20},
      .alignment = memory.alignment,
      .issue_width = 32,
      .parallel_budget = core_budget,
   };
   for (unsigned i = 0; i < 6; i++) {
      for (unsigned j = 0; j < 2; j++) {
         next.memory[i].bandwidth[j] = memory.memory[i].bandwidth[j];
         next.memory[i].latency[j] = memory.memory[i].latency[j];
      }
      next.memory[i].parallel = memory.memory[i].parallel;
      next.memory[i].line_size = memory.memory[i].line_size;
      next.memory[i].size = memory.memory[i].size;
   }
   *target = next;
   return true;
}

static bool
contains(const uint64_t *selected, uint32_t group)
{
   return (selected[group / 64] >> (group % 64)) & 1;
}

enum pan_graph_tsu_status
pan_graph_program_cost_plan_matmul_add_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const uint64_t *selected, struct pan_graph_program_cost_plan *result)
{
   struct pan_graph_matmul_plan matmul;
   if (!info || !selected || !result || (add_count && !adds) ||
       add_count > INT32_MAX - 4 || !pan_graph_matmul_i8_init(&matmul, info))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   uint32_t group_count = add_count + 2, tensor_count = add_count + 4;
   if (SIZE_MAX / group_count <
          sizeof(struct pan_graph_program_boundary_group) ||
       SIZE_MAX / tensor_count < sizeof(uint32_t) * 2)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   for (uint32_t i = 0; i < add_count; i++) {
      if (adds[i].input[0] > i || adds[i].input[1] > i)
         return PAN_GRAPH_TSU_UNSUPPORTED;
   }
   struct pan_graph_program_boundary_group *groups =
      calloc(group_count, sizeof(*groups));
   uint32_t *kinds = calloc(tensor_count, sizeof(*kinds));
   uint32_t *scratch = calloc(tensor_count, 2 * sizeof(*scratch));
   struct pan_graph_program_boundary boundary = {0};
   struct pan_graph_memory_cost_section *sections = NULL;
   enum pan_graph_memory_cost_view *views = NULL;
   enum pan_graph_cost_ve_i32_op *vector_ops = NULL;
   uint32_t *read_tensors = NULL, *vector_groups = NULL;
   enum pan_graph_tsu_status status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!groups || !kinds || !scratch)
      goto cleanup;
   for (uint32_t i = 0; i < tensor_count; i++)
      kinds[i] = i > add_count + 1 ? 3 : i == add_count ? 4 : 1;
   groups[0] = (struct pan_graph_program_boundary_group){
      .kind = PAN_GRAPH_PROGRAM_BOUNDARY_WEIGHT,
      .inputs = {{add_count + 3, 1, 0}},
      .output = {add_count + 1, 2, 0},
   };
   groups[1] = (struct pan_graph_program_boundary_group){
      .kind = PAN_GRAPH_PROGRAM_BOUNDARY_CE,
      .inputs = {{add_count + 2, 1, 0}, {add_count + 1, 3, 0}},
      .output = {0, 2, 0},
   };
   for (uint32_t i = 0; i < add_count; i++) {
      groups[i + 2] = (struct pan_graph_program_boundary_group){
         .kind = PAN_GRAPH_PROGRAM_BOUNDARY_ADD,
         .inputs = {{adds[i].input[0], 1, 0}, {adds[i].input[1], 257, 0}},
         .output = {i + 1, 2, 0},
      };
   }
   const struct pan_graph_program_boundary_input boundary_input = {
      .tensor_kinds = kinds,
      .groups = groups,
      .selected = selected,
      .tensor_count = tensor_count,
      .group_count = group_count,
   };
   status =
      pan_graph_program_boundary_matmul_add_v15(&boundary_input, &boundary);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      goto cleanup;
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   bool ce = contains(selected, 1);
   uint32_t *read_ids = scratch, *read_flags = scratch + tensor_count;
   uint32_t reads = 0, vectors = ce;
   for (uint32_t i = 0; i < add_count; i++) {
      if (!contains(selected, i + 2))
         continue;
      vectors++;
      for (unsigned j = 0; j < 2; j++) {
         uint32_t value = adds[i].input[j];
         if (contains(selected, value + 1))
            continue;
         if (ce)
            goto cleanup;
         if (!read_flags[value]) {
            read_ids[reads++] = value;
            read_flags[value] = 1;
         }
      }
   }
   if (ce) {
      if (boundary.input_count != 2 || vectors > 8)
         goto cleanup;
      reads = 2;
      read_ids[0] = add_count + 3;
      read_ids[1] = add_count + 2;
   }
   uint64_t fixed_sections = (ce ? 4 : reads) + (uint64_t)vectors;
   if (!vectors || !reads || !boundary.output_count ||
       boundary.input_count > INT32_MAX || boundary.output_count > INT32_MAX ||
       fixed_sections > INT32_MAX ||
       boundary.output_count > INT32_MAX - fixed_sections)
      goto cleanup;
   uint32_t section_count = (ce ? 4 : reads) + vectors + boundary.output_count;
   if (SIZE_MAX / section_count < sizeof(*sections) ||
       SIZE_MAX / boundary.input_count < sizeof(*views) ||
       SIZE_MAX / vectors < sizeof(*vector_ops))
      goto cleanup;
   sections = calloc(section_count, sizeof(*sections));
   views = calloc(boundary.input_count, sizeof(*views));
   vector_ops = calloc(vectors, sizeof(*vector_ops));
   read_tensors = calloc(reads, sizeof(*read_tensors));
   vector_groups = calloc(vectors, sizeof(*vector_groups));
   status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!sections || !views || !vector_ops || !read_tensors || !vector_groups)
      goto cleanup;
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   uint32_t height = info->columns <= 1 ? 16 : info->columns <= 2 ? 8 : 4;
   enum pan_graph_memory_layout layout =
      height == 16  ? PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_260
      : height == 8 ? PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_261
                    : PAN_GRAPH_MEMORY_LAYOUT_BLOCKED_262;
   uint32_t block[4];
   if (!pan_graph_memory_layout_v15_block(layout, 32, block))
      goto cleanup;
   for (size_t i = 0; i < boundary.input_count; i++)
      views[i] = ce && (boundary.inputs[i].key & 31) == 3
                    ? PAN_GRAPH_MEMORY_COST_WEIGHT_VIEW
                    : PAN_GRAPH_MEMORY_COST_TENSOR_VIEW;
   for (uint32_t i = 0; i < vectors; i++)
      vector_ops[i] =
         ce && !i ? PAN_GRAPH_COST_VE_I32_IDENTITY : PAN_GRAPH_COST_VE_I32_ADD;
   memcpy(read_tensors, read_ids, reads * sizeof(*read_tensors));
   uint32_t at = 0;
   if (ce)
      vector_groups[at++] = 1;
   for (uint32_t i = 0; i < add_count; i++) {
      if (contains(selected, i + 2))
         vector_groups[at++] = i + 2;
   }
   uint32_t write_begin = section_count - boundary.output_count;
   for (uint32_t i = 0; i < section_count; i++) {
      struct pan_graph_memory_cost_section *section = sections + i;
      section->type = i < reads         ? PAN_GRAPH_V15_MEMORY_INPUT
                      : ce && i == 2    ? PAN_GRAPH_V15_TRANSFORM
                      : ce && i == 3    ? PAN_GRAPH_V15_CE
                      : i < write_begin ? PAN_GRAPH_V15_VE
                                        : PAN_GRAPH_V15_MEMORY_OUTPUT;
      if (section->type != PAN_GRAPH_V15_MEMORY_INPUT &&
          section->type != PAN_GRAPH_V15_MEMORY_OUTPUT)
         continue;
      bool read = i < reads;
      section->bits = ce && read ? 8 : 32;
      section->replication_rank = ce ? read ? 5 : 4 : 3;
      section->full[0] = ce && read && i == 1 ? info->inner : info->columns;
      section->full[1] = ce && read ? i ? info->rows : info->inner : info->rows;
      section->full[2] = info->batch;
      bool blocked = !ce && read;
      if (!read) {
         uint32_t output = boundary.outputs[i - write_begin].tensor;
         if (output > add_count)
            goto cleanup;
         blocked = output != add_count;
      }
      if (blocked) {
         section->layout = (enum pan_graph_memory_cost_layout)layout;
         memcpy(section->block, block, sizeof(block));
      }
      if (read) {
         section->retain_read_peak = !ce || !i;
         for (uint32_t node = 0; !ce && node < add_count; node++) {
            if (!contains(selected, node + 2) ||
                (adds[node].input[0] != read_ids[i] &&
                 adds[node].input[1] != read_ids[i]))
               continue;
            if (contains(selected, adds[node].input[0] + 1) ||
                contains(selected, adds[node].input[1] + 1))
               section->retain_read_peak = false;
         }
      }
   }
   struct pan_graph_memory_cost_program program = {
      .sections = sections,
      .section_count = section_count,
      .input_views = views,
      .input_view_count = boundary.input_count,
      .output_view_count = boundary.output_count,
      .resource_count = vectors + (ce ? 4 : reads),
      .workload = {.rank = ce ? 7 : 3, .repeats = 1, .minimum = .5f},
   };
   for (uint32_t i = 0; i < program.workload.rank; i++) {
      program.workload.extent[i] =
         ce       ? i == 2 ? info->batch : matmul.transfer_high[i] + 1
         : i == 0 ? info->batch
         : i == 1 ? info->rows
                  : info->columns;
   }
   *result = (struct pan_graph_program_cost_plan){
      .info = *info,
      .boundary = boundary,
      .program = program,
      .sections = sections,
      .input_views = views,
      .vector_ops = vector_ops,
      .read_tensors = read_tensors,
      .vector_groups = vector_groups,
      .read_count = reads,
      .vector_count = vectors,
   };
   boundary = (struct pan_graph_program_boundary){0};
   sections = NULL;
   views = NULL;
   vector_ops = NULL;
   read_tensors = vector_groups = NULL;
   status = PAN_GRAPH_TSU_SUCCESS;
cleanup:
   free(groups);
   free(kinds);
   free(scratch);
   free(sections);
   free(views);
   free(vector_ops);
   free(read_tensors);
   free(vector_groups);
   pan_graph_program_boundary_finish(&boundary);
   return status;
}

enum pan_graph_tsu_status
pan_graph_program_cost_plan_estimate(
   struct pan_graph_program_cost_plan *plan, const uint32_t transfer[7],
   const uint32_t loops[4], const struct pan_graph_memory_cost_target *target,
   struct pan_graph_program_cost_result *result)
{
   if (!plan || !transfer || !loops || !target || !result || !plan->sections ||
       !plan->input_views || !plan->vector_ops || !plan->read_tensors ||
       !plan->vector_groups || !plan->read_count || !plan->vector_count)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   bool ce = plan->program.workload.rank == 7;
   if (!ce && ((loops[2] && (loops[0] < 1 || loops[0] > 3)) ||
               (loops[3] && (loops[1] < 1 || loops[1] > 3))))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   for (uint32_t i = 0; i < plan->program.section_count; i++) {
      struct pan_graph_memory_cost_section *section = plan->sections + i;
      bool read = section->type == PAN_GRAPH_V15_MEMORY_INPUT;
      if (!read && section->type != PAN_GRAPH_V15_MEMORY_OUTPUT)
         continue;
      section->transfer[0] =
         ce ? read && i == 1 ? transfer[4] : transfer[0] : transfer[2];
      section->transfer[1] =
         ce ? read && !i ? transfer[4] : transfer[3] : transfer[1];
      section->transfer[2] = ce ? 1 : transfer[0];
   }
   struct pan_graph_memory_cost_program program = plan->program;
   program.workload.axis[0] = ce ? loops[0] : loops[2] ? loops[0] - 1 : 0;
   program.workload.axis[1] = ce ? loops[1] : loops[3] ? loops[1] - 1 : 0;
   program.workload.step[0] = loops[2];
   program.workload.step[1] = loops[3];
   program.workload.latency = target->memory[5].latency[0];
   for (uint32_t i = 0; i < program.workload.rank; i++)
      program.workload.transfer[i] = transfer[i];
   struct pan_graph_program_cost_result next = {
      .input_view_count = plan->boundary.input_count,
      .output_view_count = plan->boundary.output_count,
      .rank = program.workload.rank,
      .read_count = plan->read_count,
      .vector_count = plan->vector_count,
      .resource_count = program.resource_count,
   };
   if (ce) {
      struct pan_graph_matmul_cost_result cost;
      if (!pan_graph_matmul_cost_estimate_program(&plan->info, target, &program,
                                                  plan->vector_ops,
                                                  plan->vector_count, &cost))
         return PAN_GRAPH_TSU_UNSUPPORTED;
      memcpy(next.memory, cost.memory, sizeof(next.memory));
      memcpy(next.engine, cost.engine, sizeof(next.engine));
      next.total = cost.total;
   } else {
      struct pan_graph_vector_cost_result cost;
      if (!pan_graph_vector_cost_estimate_program_add_i32(target, &program,
                                                          &cost))
         return PAN_GRAPH_TSU_UNSUPPORTED;
      memcpy(next.memory, cost.memory, sizeof(next.memory));
      memcpy(next.engine, cost.engine, sizeof(next.engine));
      next.total = cost.total;
   }
   *result = next;
   return PAN_GRAPH_TSU_SUCCESS;
}

void
pan_graph_program_cost_plan_finish(struct pan_graph_program_cost_plan *plan)
{
   if (!plan)
      return;
   free(plan->sections);
   free(plan->input_views);
   free(plan->vector_ops);
   free(plan->read_tensors);
   free(plan->vector_groups);
   pan_graph_program_boundary_finish(&plan->boundary);
   *plan = (struct pan_graph_program_cost_plan){0};
}

enum pan_graph_tsu_status
pan_graph_program_cost_matmul_add_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const uint64_t *selected, const uint32_t transfer[7],
   const uint32_t loops[4], const struct pan_graph_memory_cost_target *target,
   struct pan_graph_program_cost_result *result)
{
   if (!transfer || !loops || !target || !result)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct pan_graph_program_cost_plan plan = {0};
   enum pan_graph_tsu_status status =
      pan_graph_program_cost_plan_matmul_add_v15(info, adds, add_count,
                                                 selected, &plan);
   if (status == PAN_GRAPH_TSU_SUCCESS)
      status = pan_graph_program_cost_plan_estimate(&plan, transfer, loops,
                                                    target, result);
   pan_graph_program_cost_plan_finish(&plan);
   return status;
}
