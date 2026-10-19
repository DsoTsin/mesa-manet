#include "pan_graph_program_boundary.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct boundary_tensor {
   uint32_t producer;
   uint32_t begin;
   uint32_t end;
};

struct boundary_frame {
   uint32_t node;
   uint32_t view;
   uint32_t edge;
   bool output;
   bool started;
   bool boundary;
};

static bool
contains(const uint64_t *mask, uint32_t index)
{
   return (mask[index / 64] >> (index % 64)) & 1;
}

static void
include(uint64_t *mask, uint32_t index)
{
   mask[index / 64] |= UINT64_C(1) << (index % 64);
}

static unsigned
input_count(uint32_t node)
{
   return node ? 2 : 1;
}

void
pan_graph_program_boundary_finish(struct pan_graph_program_boundary *result)
{
   if (!result)
      return;
   free(result->inputs);
   free(result->outputs);
   *result = (struct pan_graph_program_boundary){0};
}

static enum pan_graph_tsu_status
append_view(struct pan_graph_program_boundary_view **views, size_t *count,
            size_t *capacity, struct pan_graph_program_boundary_view view,
            uint32_t counters[32], bool weight)
{
   uint32_t role = view.key & 31;
   if (!weight && counters[role] > (UINT32_MAX >> 8))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   if (*count == *capacity) {
      if (*capacity > SIZE_MAX / 2)
         return PAN_GRAPH_TSU_UNSUPPORTED;
      size_t next_capacity = *capacity ? 2 * *capacity : 8;
      if (next_capacity > SIZE_MAX / sizeof(**views))
         return PAN_GRAPH_TSU_UNSUPPORTED;
      struct pan_graph_program_boundary_view *next =
         calloc(next_capacity, sizeof(*next));
      if (!next)
         return PAN_GRAPH_TSU_OUT_OF_MEMORY;
      if (*count)
         memcpy(next, *views, *count * sizeof(*next));
      free(*views);
      *views = next;
      *capacity = next_capacity;
   }
   view.key = weight ? 3 : role | (counters[role]++ << 8);
   (*views)[(*count)++] = view;
   return PAN_GRAPH_TSU_SUCCESS;
}

enum pan_graph_tsu_status
pan_graph_program_boundary_matmul_add_v15(
   const struct pan_graph_program_boundary_input *input,
   struct pan_graph_program_boundary *result)
{
   if (!input || !result || !input->tensor_kinds || !input->groups ||
       !input->selected || input->group_count < 2 ||
       input->group_count > INT32_MAX || !input->tensor_count ||
       input->tensor_count > INT32_MAX)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   size_t words = ((size_t)input->group_count + 63) / 64;
   if ((input->group_count % 64 &&
        input->selected[words - 1] >> (input->group_count % 64)) ||
       contains(input->selected, 0) != contains(input->selected, 1))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   uint32_t root = UINT32_MAX;
   for (uint32_t i = 0; i < input->tensor_count; i++) {
      if (input->tensor_kinds[i] < 1 || input->tensor_kinds[i] > 4)
         return PAN_GRAPH_TSU_UNSUPPORTED;
   }
   for (uint32_t i = 0; i < input->group_count; i++) {
      const struct pan_graph_program_boundary_group *group = input->groups + i;
      enum pan_graph_program_boundary_kind expected =
         i == 0 ? PAN_GRAPH_PROGRAM_BOUNDARY_WEIGHT :
         i == 1 ? PAN_GRAPH_PROGRAM_BOUNDARY_CE : PAN_GRAPH_PROGRAM_BOUNDARY_ADD;
      if (group->kind != expected || group->output.tensor >= input->tensor_count ||
          group->output.key != 2)
         return PAN_GRAPH_TSU_UNSUPPORTED;
      uint32_t output_kind = input->tensor_kinds[group->output.tensor];
      if ((output_kind != 1 && output_kind != 4) || (!i && output_kind != 1))
         return PAN_GRAPH_TSU_UNSUPPORTED;
      for (unsigned j = 0; j < input_count(i); j++) {
         uint32_t key = j ? (i == 1 ? 3 : 257) : 1;
         if (group->inputs[j].tensor >= input->tensor_count ||
             group->inputs[j].key != key)
            return PAN_GRAPH_TSU_UNSUPPORTED;
         if (i < 2 && !j && input->tensor_kinds[group->inputs[j].tensor] != 3)
            return PAN_GRAPH_TSU_UNSUPPORTED;
      }
      if (contains(input->selected, i))
         root = i;
   }
   if (root == UINT32_MAX)
      return PAN_GRAPH_TSU_UNSUPPORTED;
   if (contains(input->selected, 1))
      root = 1;

   size_t edge_count = 2 * (size_t)input->group_count - 1;
   if (SIZE_MAX / input->tensor_count < sizeof(struct boundary_tensor) ||
       SIZE_MAX / edge_count < sizeof(uint32_t) ||
       SIZE_MAX / input->group_count < sizeof(struct boundary_frame) ||
       words > SIZE_MAX / (2 * sizeof(uint64_t)))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct boundary_tensor *tensors = calloc(input->tensor_count, sizeof(*tensors));
   uint32_t *consumers = calloc(edge_count, sizeof(*consumers));
   struct boundary_frame *frames = malloc(input->group_count * sizeof(*frames));
   uint64_t *masks = calloc(2 * words, sizeof(*masks));
   struct pan_graph_program_boundary next = {.root = root};
   enum pan_graph_tsu_status status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!tensors || !consumers || !frames || !masks)
      goto cleanup;
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   for (uint32_t i = 0; i < input->tensor_count; i++)
      tensors[i].producer = UINT32_MAX;
   for (uint32_t i = 0; i < input->group_count; i++) {
      const struct pan_graph_program_boundary_group *group = input->groups + i;
      struct boundary_tensor *output = tensors + group->output.tensor;
      if (output->producer != UINT32_MAX)
         goto cleanup;
      output->producer = i;
      for (unsigned j = 0; j < input_count(i); j++)
         tensors[group->inputs[j].tensor].end++;
   }
   uint32_t edge = 0;
   for (uint32_t i = 0; i < input->tensor_count; i++) {
      uint32_t count = tensors[i].end;
      tensors[i].begin = tensors[i].end = edge;
      edge += count;
   }
   for (uint32_t i = 0; i < input->group_count; i++) {
      const struct pan_graph_program_boundary_group *group = input->groups + i;
      for (unsigned j = 0; j < input_count(i); j++) {
         struct boundary_tensor *tensor = tensors + group->inputs[j].tensor;
         if ((i < 2 && !j && tensor->producer != UINT32_MAX) ||
             ((i >= 2 || j) &&
              (tensor->producer == UINT32_MAX || tensor->producer >= i)))
            goto cleanup;
         if ((i == 1 && j && tensor->producer != 0) ||
             (i >= 2 && tensor->producer == 0))
            goto cleanup;
         consumers[tensor->end++] = i;
      }
   }

   uint64_t *visited = masks, *reached = masks + words;
   uint32_t counters[32] = {0};
   size_t input_capacity = 0, output_capacity = 0, depth = 0;
   frames[0] = (struct boundary_frame){.node = root};
   include(visited, root);
   include(reached, root);
   for (;;) {
      struct boundary_frame *frame = frames + depth;
      const struct pan_graph_program_boundary_group *group =
         input->groups + frame->node;
      if (!frame->output && frame->view == input_count(frame->node)) {
         frame->output = true;
         frame->started = false;
      }
      struct pan_graph_program_boundary_view view =
         frame->output ? group->output : group->inputs[frame->view];
      const struct boundary_tensor *tensor = tensors + view.tensor;
      if (!frame->started) {
         frame->edge = frame->output ? tensor->begin : 0;
         frame->boundary = frame->output ?
            input->tensor_kinds[view.tensor] == 4 || tensor->begin == tensor->end :
            tensor->producer == UINT32_MAX;
         frame->started = true;
      }
      uint32_t neighbor = UINT32_MAX;
      if (frame->output) {
         if (frame->edge < tensor->end)
            neighbor = consumers[frame->edge++];
      } else if (!frame->edge++) {
         neighbor = tensor->producer;
      }
      if (neighbor != UINT32_MAX) {
         if (!contains(input->selected, neighbor))
            frame->boundary = true;
         else if (!contains(visited, neighbor)) {
            depth++;
            frames[depth] = (struct boundary_frame){.node = neighbor};
            include(visited, neighbor);
            include(reached, neighbor);
         }
         continue;
      }
      if (frame->boundary) {
         status = frame->output ?
            append_view(&next.outputs, &next.output_count, &output_capacity,
                        view, counters, false) :
            append_view(&next.inputs, &next.input_count, &input_capacity,
                        view, counters, frame->node == 0);
         if (status != PAN_GRAPH_TSU_SUCCESS)
            goto cleanup;
      }
      if (!frame->output) {
         frame->view++;
         frame->started = false;
         continue;
      }
      visited[frame->node / 64] &= ~(UINT64_C(1) << (frame->node % 64));
      if (!depth)
         break;
      depth--;
   }
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   for (size_t i = 0; i < words; i++) {
      if (reached[i] != input->selected[i])
         goto cleanup;
   }
   status = PAN_GRAPH_TSU_SUCCESS;
   *result = next;
   next = (struct pan_graph_program_boundary){0};

cleanup:
   free(tensors);
   free(consumers);
   free(frames);
   free(masks);
   pan_graph_program_boundary_finish(&next);
   return status;
}
