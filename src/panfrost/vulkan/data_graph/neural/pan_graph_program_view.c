#include "pan_graph_program_view.h"
#include "pan_graph_program_boundary.h"

#include <limits.h>
#include <stdlib.h>

enum pan_graph_tsu_status
pan_graph_program_view_bounds_init(
   uint32_t rank, const uint32_t *shape,
   const struct pan_graph_program_view_bounds *explicit_bounds,
   struct pan_graph_program_view_bounds *result)
{
   if (!result || (!explicit_bounds && !shape))
      return PAN_GRAPH_TSU_INVALID;
   if (explicit_bounds)
      rank = explicit_bounds->rank;
   if (!rank || rank > PAN_GRAPH_PROGRAM_VIEW_MAX_RANK)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct pan_graph_program_view_bounds next = {.rank = rank};
   for (uint32_t i = 0; i < rank; i++) {
      if (explicit_bounds) {
         if (explicit_bounds->low[i] > explicit_bounds->high[i])
            return PAN_GRAPH_TSU_INVALID;
         next.low[i] = explicit_bounds->low[i];
         next.high[i] = explicit_bounds->high[i];
      } else {
         if (!shape[i] || shape[i] > INT32_MAX)
            return PAN_GRAPH_TSU_UNSUPPORTED;
         next.high[i] = shape[i];
      }
   }
   *result = next;
   return PAN_GRAPH_TSU_SUCCESS;
}

void
pan_graph_program_views_finish(struct pan_graph_program_views *result)
{
   if (!result)
      return;
   free(result->inputs);
   free(result->outputs);
   *result = (struct pan_graph_program_views){0};
}

static struct pan_graph_program_view_origin
decode_origin(uint32_t token)
{
   if (token < 2)
      return (struct pan_graph_program_view_origin){0, 0, token == 1};
   if (token < 5)
      return (struct pan_graph_program_view_origin){1, token == 3, token == 4};
   uint32_t index = (token - 5) % 3;
   return (struct pan_graph_program_view_origin){
      2 + (token - 5) / 3, index == 1, index == 2};
}

enum pan_graph_tsu_status
pan_graph_program_view_matmul_add_v15(
   const struct pan_graph_matmul_info *info,
   const struct pan_graph_matmul_add *adds, uint32_t add_count,
   const uint64_t *selected, struct pan_graph_program_views *result)
{
   struct pan_graph_matmul_plan matmul;
   if (!info || !selected || !result || (add_count && !adds) ||
       add_count > INT32_MAX - 4 || add_count > (UINT32_MAX - 5) / 3 ||
       !pan_graph_matmul_i8_init(&matmul, info))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   uint32_t group_count = add_count + 2, tensor_count = add_count + 4;
   if (SIZE_MAX / group_count < sizeof(struct pan_graph_program_boundary_group) ||
       SIZE_MAX / tensor_count < sizeof(uint32_t))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   for (uint32_t i = 0; i < add_count; i++) {
      if (adds[i].input[0] > i || adds[i].input[1] > i)
         return PAN_GRAPH_TSU_UNSUPPORTED;
   }
   struct pan_graph_program_boundary_group *groups =
      calloc(group_count, sizeof(*groups));
   uint32_t *kinds = calloc(tensor_count, sizeof(*kinds));
   struct pan_graph_program_boundary boundary = {0};
   struct pan_graph_program_views next = {0};
   enum pan_graph_tsu_status status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!groups || !kinds)
      goto cleanup;
   for (uint32_t i = 0; i < tensor_count; i++)
      kinds[i] = i > add_count + 1 ? 3 : i == add_count ? 4 : 1;
   groups[0] = (struct pan_graph_program_boundary_group){
      .kind = PAN_GRAPH_PROGRAM_BOUNDARY_WEIGHT,
      .inputs = {{add_count + 3, 1, 0}},
      .output = {add_count + 1, 2, 1},
   };
   groups[1] = (struct pan_graph_program_boundary_group){
      .kind = PAN_GRAPH_PROGRAM_BOUNDARY_CE,
      .inputs = {{add_count + 2, 1, 2}, {add_count + 1, 3, 3}},
      .output = {0, 2, 4},
   };
   for (uint32_t i = 0; i < add_count; i++) {
      uint32_t origin = 5 + 3 * i;
      groups[i + 2] = (struct pan_graph_program_boundary_group){
         .kind = PAN_GRAPH_PROGRAM_BOUNDARY_ADD,
         .inputs = {{adds[i].input[0], 1, origin},
                    {adds[i].input[1], 257, origin + 1}},
         .output = {i + 1, 2, origin + 2},
      };
   }
   const struct pan_graph_program_boundary_input input = {
      kinds, groups, selected, tensor_count, group_count};
   status = pan_graph_program_boundary_matmul_add_v15(&input, &boundary);
   if (status != PAN_GRAPH_TSU_SUCCESS)
      goto cleanup;
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   if (SIZE_MAX / sizeof(*next.inputs) < boundary.input_count ||
       SIZE_MAX / sizeof(*next.outputs) < boundary.output_count)
      goto cleanup;
   next.input_count = boundary.input_count;
   next.output_count = boundary.output_count;
   next.root = boundary.root;
   next.inputs = next.input_count ? calloc(next.input_count, sizeof(*next.inputs))
                                  : NULL;
   next.outputs = next.output_count
                     ? calloc(next.output_count, sizeof(*next.outputs))
                     : NULL;
   status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if ((next.input_count && !next.inputs) ||
       (next.output_count && !next.outputs))
      goto cleanup;
   for (size_t side = 0; side < 2; side++) {
      const struct pan_graph_program_boundary_view *views =
         side ? boundary.outputs : boundary.inputs;
      size_t count = side ? boundary.output_count : boundary.input_count;
      struct pan_graph_program_view *destination =
         side ? next.outputs : next.inputs;
      for (size_t i = 0; i < count; i++) {
         uint32_t tensor = views[i].tensor;
         uint32_t shape[4] = {info->columns, info->rows, info->batch, 0};
         uint32_t rank = 3;
         if (tensor == add_count + 1) {
            rank = 4;
            shape[0] = info->inner;
            shape[1] = shape[2] = 1;
            shape[3] = info->columns;
         } else if (tensor == add_count + 2) {
            shape[0] = info->inner;
         } else if (tensor == add_count + 3) {
            shape[1] = info->inner;
         }
         destination[i] = (struct pan_graph_program_view){
            .tensor = tensor,
            .key = views[i].key,
            .kind = kinds[tensor],
            .origin = decode_origin(views[i].origin),
         };
         status = pan_graph_program_view_bounds_init(
            rank, shape, NULL, &destination[i].bounds);
         if (status != PAN_GRAPH_TSU_SUCCESS)
            goto cleanup;
      }
   }
   *result = next;
   next = (struct pan_graph_program_views){0};
   status = PAN_GRAPH_TSU_SUCCESS;
cleanup:
   free(groups);
   free(kinds);
   pan_graph_program_boundary_finish(&boundary);
   pan_graph_program_views_finish(&next);
   return status;
}
