#include "pan_graph_program_schedule.h"
#include "pan_graph_matmul_select.h"
#include "pan_graph_program_expand.h"
#include "pan_graph_program_order.h"
#include "pan_graph_program_prune.h"
#include "pan_graph_vector_schedule.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static uint32_t
gcd(uint32_t a, uint32_t b)
{
   while (b) {
      uint32_t r = a % b;
      a = b;
      b = r;
   }
   return a;
}

static bool
output_granules(const struct pan_graph_program_cost_plan *plan,
                uint32_t granule[7])
{
   for (unsigned i = 0; i < 7; i++)
      granule[i] = 1;
   for (uint32_t i = 0; i < plan->program.section_count; i++) {
      const struct pan_graph_memory_cost_section *s = plan->sections + i;
      if (s->type != PAN_GRAPH_V15_MEMORY_OUTPUT ||
          s->layout == PAN_GRAPH_MEMORY_COST_COMPACT_LINEAR)
         continue;
      uint32_t mapped[7] = {1, 1, 1, 1, 1, 1, 1};
      if (plan->program.workload.rank == 7) {
         mapped[0] = s->block[0];
         mapped[2] = s->block[2];
         mapped[3] = s->block[1];
      } else {
         mapped[0] = s->block[2];
         mapped[1] = s->block[1];
         mapped[2] = s->block[0];
      }
      for (unsigned j = 0; j < 7; j++) {
         if (!mapped[j])
            return false;
         uint64_t multiple =
            (uint64_t)granule[j] / gcd(granule[j], mapped[j]) * mapped[j];
         if (multiple > INT32_MAX)
            return false;
         granule[j] = multiple;
      }
   }
   return true;
}

static enum pan_graph_tsu_status
order_programs(struct pan_graph_program_schedule *schedule)
{
   size_t references = 0, count = schedule->count;
   for (size_t i = 0; i < count; i++) {
      size_t n = schedule->programs[i].plan.boundary.input_count;
      if (references > SIZE_MAX - n)
         return PAN_GRAPH_TSU_OUT_OF_MEMORY;
      references += n;
   }
   size_t words = ((size_t)schedule->group_count + 63) / 64;
   if (count > UINT32_MAX ||
       count > SIZE_MAX / sizeof(struct pan_graph_program_order_node) ||
       references > SIZE_MAX / sizeof(uint64_t) ||
       SIZE_MAX / schedule->group_count < sizeof(uint32_t))
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   struct pan_graph_program_order_node *nodes = calloc(count, sizeof(*nodes));
   uint64_t *dependencies =
      references ? malloc(references * sizeof(*dependencies)) : NULL;
   uint32_t *owners = malloc((size_t)schedule->group_count * sizeof(*owners));
   struct pan_graph_program_order order = {0};
   struct pan_graph_scheduled_program *programs = NULL;
   uint64_t *groups = NULL;
   enum pan_graph_tsu_status status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!nodes || (references && !dependencies) || !owners)
      goto cleanup;
   for (uint32_t g = 0; g < schedule->group_count; g++)
      owners[g] = UINT32_MAX;
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   for (size_t i = 0; i < count; i++) {
      const struct pan_graph_scheduled_program *p = schedule->programs + i;
      for (uint32_t g = 0; g < schedule->group_count; g++) {
         if (!((p->groups[g / 64] >> (g % 64)) & 1))
            continue;
         if (owners[g] != UINT32_MAX)
            goto cleanup;
         owners[g] = i;
      }
   }
   for (uint32_t g = 0; g < schedule->group_count; g++) {
      if (owners[g] == UINT32_MAX)
         goto cleanup;
   }
   size_t at = 0;
   for (size_t i = 0; i < count; i++) {
      const struct pan_graph_program_boundary *boundary =
         &schedule->programs[i].plan.boundary;
      nodes[i].uid = i + 1;
      nodes[i].dependencies = dependencies ? dependencies + at : NULL;
      for (size_t j = 0; j < boundary->input_count; j++) {
         uint32_t tensor = boundary->inputs[j].tensor;
         if (tensor >= schedule->group_count)
            continue;
         uint32_t group = tensor + 1 == schedule->group_count ? 0 : tensor + 1;
         if (owners[group] != i) {
            dependencies[at++] = (uint64_t)owners[group] + 1;
            nodes[i].dependency_count++;
         }
      }
   }
   status = pan_graph_program_order_build_selected(nodes, count, &order);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      goto cleanup;
   status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   programs = malloc(count * sizeof(*programs));
   groups = malloc(count * words * sizeof(*groups));
   if (!programs || !groups)
      goto cleanup;
   for (size_t i = 0; i < count; i++) {
      programs[i] = schedule->programs[order.indices[i]];
      memcpy(groups + i * words, programs[i].groups, words * sizeof(*groups));
      programs[i].groups = groups + i * words;
   }
   free(schedule->programs);
   free(schedule->groups);
   schedule->programs = programs;
   schedule->groups = groups;
   programs = NULL;
   groups = NULL;
   status = PAN_GRAPH_TSU_SUCCESS;
cleanup:
   free(nodes);
   free(dependencies);
   free(owners);
   free(programs);
   free(groups);
   pan_graph_program_order_finish(&order);
   return status;
}

enum pan_graph_tsu_status
pan_graph_program_schedule_v15(
   struct pan_graph_program_cost_plan *plan,
   const struct pan_graph_nrt_target *target,
   const struct pan_graph_memory_cost_target *cost_target, uint32_t core_budget,
   struct pan_graph_program_selection *selection)
{
   if (!plan || !target || !cost_target || !selection || !plan->sections ||
       !plan->boundary.output_count || !plan->vector_count || !core_budget ||
       core_budget > INT32_MAX || cost_target->parallel_budget != core_budget)
      return PAN_GRAPH_TSU_INVALID;
   bool ce = plan->program.workload.rank == 7;
   if (!ce && plan->program.workload.rank != 3)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct pan_graph_program_selection next = {0};
   if (!output_granules(plan, next.granule))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct pan_graph_matmul_candidate_list mm = {0}, seed = {0};
   struct pan_graph_vector_candidate_list ve = {0};
   struct pan_graph_vector_schedule_info vector = {
      .full = {plan->info.batch, plan->info.rows, plan->info.columns},
      .read_count = plan->read_count,
      .vector_count = plan->vector_count,
      .write_count = plan->boundary.output_count,
   };
   memcpy(vector.granule, next.granule, sizeof(vector.granule));
   const struct pan_graph_matmul_parallel_options options = {
      .core_budget = core_budget,
      .output_axis_mask = 15,
      .multiple_axes = true,
   };
   enum pan_graph_tsu_status status;
   if (ce) {
      uint32_t full[7];
      for (unsigned i = 0; i < 7; i++)
         full[i] = plan->program.workload.extent[i];
      status = pan_graph_matmul_program_base_candidates(
         &plan->info, full, target, -1, 0, false, plan->vector_count, &seed);
      if (status == PAN_GRAPH_TSU_SUCCESS)
         status = pan_graph_matmul_program_parallel_candidates(
            &plan->info, full, target, -1, false, &options, next.granule,
            plan->vector_count, &seed, &mm);
      next.candidate_count = mm.count;
   } else {
      for (size_t i = 0; i < plan->boundary.output_count; i++) {
         size_t j = 0;
         for (; j < i; j++) {
            if (plan->boundary.outputs[j].tensor ==
                plan->boundary.outputs[i].tensor)
               break;
         }
         vector.export_count += j == i;
      }
      status = pan_graph_vector_parallel_candidates(&vector, target,
                                                    core_budget, &ve);
      next.candidate_count = ve.count;
   }
   if (status != PAN_GRAPH_TSU_SUCCESS)
      goto cleanup;
   uint64_t best = INT64_MAX;
   for (size_t i = 0; i < next.candidate_count; i++) {
      uint32_t transfer[7] = {0}, loops[4];
      if (ce) {
         memcpy(transfer, mm.candidates[i].transfer, sizeof(transfer));
         if (!pan_graph_matmul_assign_loops(&plan->info, transfer, next.granule,
                                            &options, loops))
            continue;
      } else {
         memcpy(transfer, ve.candidates[i].transfer,
                sizeof(ve.candidates[i].transfer));
         if (!pan_graph_vector_assign_loops(&vector, transfer, core_budget,
                                            loops))
            continue;
      }
      struct pan_graph_program_cost_result cost;
      if (pan_graph_program_cost_plan_estimate(plan, transfer, loops,
                                               cost_target, &cost) !=
             PAN_GRAPH_TSU_SUCCESS ||
          cost.total.cycles >= best)
         continue;
      best = cost.total.cycles;
      memcpy(next.transfer, transfer, sizeof(transfer));
      memcpy(next.loops, loops, sizeof(loops));
      next.schedule_type = ce ? mm.candidates[i].schedule_type : 0;
      next.cost = cost;
   }
   if (best == INT64_MAX) {
      status = PAN_GRAPH_TSU_UNSUPPORTED;
      goto cleanup;
   }
   *selection = next;
cleanup:
   pan_graph_matmul_candidate_list_finish(&mm);
   pan_graph_matmul_candidate_list_finish(&seed);
   pan_graph_vector_candidates_finish(&ve);
   return status;
}

enum pan_graph_tsu_status
pan_graph_program_schedule_matmul_add_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const struct pan_graph_nrt_target *target,
   const struct pan_graph_memory_cost_target *cost_target, uint32_t core_budget,
   uint32_t io_limit, struct pan_graph_program_schedule *schedule)
{
   struct pan_graph_matmul_plan matmul;
   if (!target || !cost_target || !schedule || !core_budget ||
       core_budget > INT32_MAX || cost_target->parallel_budget != core_budget ||
       !pan_graph_matmul_i8_init(&matmul, info))
      return PAN_GRAPH_TSU_INVALID;
   struct pan_graph_program_subsets pool = {0};
   enum pan_graph_tsu_status status =
      pan_graph_program_unique_candidates_matmul_add_v15(adds, add_count,
                                                         io_limit, &pool);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      return status;
   struct pan_graph_program_cost_plan *plans = NULL;
   struct pan_graph_program_selection *selections = NULL;
   struct pan_graph_program_cost *costs = NULL;
   struct pan_graph_program_choice chosen = {0};
   struct pan_graph_program_schedule next = {.group_count = pool.group_count};
   size_t count = 0;
   bool transferred = false;
   status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (pool.count > SIZE_MAX / sizeof(*plans) ||
       pool.count > SIZE_MAX / sizeof(*selections) ||
       pool.count > SIZE_MAX / sizeof(*costs))
      goto cleanup;
   plans = calloc(pool.count, sizeof(*plans));
   selections = calloc(pool.count, sizeof(*selections));
   costs = calloc(pool.count, sizeof(*costs));
   if (!plans || !selections || !costs)
      goto cleanup;
   size_t words = ((size_t)pool.group_count + 63) / 64;
   for (size_t i = 0; i < pool.count; i++) {
      const uint64_t *groups = pool.groups + i * words;
      status = pan_graph_program_cost_plan_matmul_add_v15(
         info, adds, add_count, groups, plans + count);
      if (status == PAN_GRAPH_TSU_OUT_OF_MEMORY)
         goto cleanup;
      if (status != PAN_GRAPH_TSU_SUCCESS)
         continue;
      status = pan_graph_program_schedule_v15(
         plans + count, target, cost_target, core_budget, selections + count);
      if (status != PAN_GRAPH_TSU_SUCCESS) {
         pan_graph_program_cost_plan_finish(plans + count);
         if (status == PAN_GRAPH_TSU_OUT_OF_MEMORY)
            goto cleanup;
         continue;
      }
      costs[count] = (struct pan_graph_program_cost){
         .groups = groups,
         .cost = selections[count].cost.total,
      };
      count++;
   }
   status =
      pan_graph_program_select_costs(pool.group_count, costs, count, &chosen);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      goto cleanup;
   next.candidate_count = count;
   next.cycles = chosen.cycles;
   for (size_t i = 0; i < count; i++)
      next.count += (chosen.selected[i / 64] >> (i % 64)) & 1;
   if (!next.count) {
      status = PAN_GRAPH_TSU_UNSUPPORTED;
      goto cleanup;
   }
   status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (next.count > SIZE_MAX / sizeof(*next.programs) ||
       next.count > SIZE_MAX / words / sizeof(*next.groups))
      goto cleanup;
   next.programs = calloc(next.count, sizeof(*next.programs));
   next.groups = calloc(next.count * words, sizeof(*next.groups));
   if (!next.programs || !next.groups)
      goto cleanup;
   size_t at = 0;
   for (size_t i = 0; i < count; i++) {
      if (!((chosen.selected[i / 64] >> (i % 64)) & 1))
         continue;
      next.programs[at].plan = plans[i];
      next.programs[at].selection = selections[i];
      memcpy(next.groups + at * words, costs[i].groups,
             words * sizeof(*next.groups));
      next.programs[at].groups = next.groups + at * words;
      at++;
   }
   status = order_programs(&next);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      goto cleanup;
   transferred = true;
   *schedule = next;
   next.programs = NULL;
   next.groups = NULL;
   status = PAN_GRAPH_TSU_SUCCESS;
cleanup:
   for (size_t i = 0; i < count; i++) {
      if (!transferred || !((chosen.selected[i / 64] >> (i % 64)) & 1))
         pan_graph_program_cost_plan_finish(plans + i);
   }
   free(next.programs);
   free(next.groups);
   free(plans);
   free(selections);
   free(costs);
   pan_graph_program_choice_finish(&chosen);
   pan_graph_program_subsets_finish(&pool);
   return status;
}

void
pan_graph_program_schedule_finish(struct pan_graph_program_schedule *schedule)
{
   if (!schedule)
      return;
   for (size_t i = 0; i < schedule->count; i++)
      pan_graph_program_cost_plan_finish(&schedule->programs[i].plan);
   free(schedule->programs);
   free(schedule->groups);
   *schedule = (struct pan_graph_program_schedule){0};
}
