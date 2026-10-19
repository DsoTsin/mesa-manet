#include "panvk_data_graph_neural.h"
#include "panvk_data_graph.h"

#include <stdlib.h>
#include <string.h>

static bool
same_shape(const struct panvk_graph_value *a, const struct panvk_graph_value *b)
{
   return a->format == b->format && a->rank == 3 && b->rank == 3 &&
          !memcmp(a->dims, b->dims, 3 * sizeof(a->dims[0]));
}

void
panvk_data_graph_matmul_add_finish(struct panvk_data_graph_matmul_add *graph)
{
   if (!graph)
      return;
   free(graph->adds);
   free(graph->values);
   memset(graph, 0, sizeof(*graph));
}

VkResult
panvk_data_graph_matmul_add_extract(const struct panvk_data_graph *graph,
                                    struct panvk_data_graph_matmul_add *result)
{
   if (!graph || !graph->count || !graph->values || !result)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   uint32_t matmul = UINT32_MAX, output = UINT32_MAX, add_count = 0;
   for (uint32_t i = 0; i < graph->count; i++) {
      const struct panvk_graph_value *v = graph->values + i;
      if (!v->live)
         continue;
      switch (v->kind) {
      case PANVK_GRAPH_OPERATION:
         if (v->op == PANVK_GRAPH_OP_MATMUL && matmul == UINT32_MAX)
            matmul = i;
         else if (v->op == PANVK_GRAPH_OP_ADD)
            add_count++;
         else
            return VK_ERROR_FEATURE_NOT_PRESENT;
         break;
      case PANVK_GRAPH_OUTPUT:
         if (output != UINT32_MAX)
            return VK_ERROR_FEATURE_NOT_PRESENT;
         output = i;
         break;
      case PANVK_GRAPH_INPUT:
      case PANVK_GRAPH_CONSTANT:
         break;
      default:
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
   }
   if (matmul == UINT32_MAX || output == UINT32_MAX)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   const struct panvk_graph_value *mm = graph->values + matmul;
   const struct panvk_graph_value *out = graph->values + output;
   if (!mm->matmul || mm->format != VK_FORMAT_R32_SINT || mm->rank != 3 ||
       mm->input_count != 4 || out->input_count != 1 ||
       out->inputs[0] >= output || !same_shape(mm, out))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   const struct pan_graph_matmul_info *info = &mm->matmul->info;
   if (!info->batch || !info->rows || !info->inner || !info->columns ||
       mm->dims[0] != info->batch || mm->dims[1] != info->rows ||
       mm->dims[2] != info->columns)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   for (uint32_t i = 0; i < 4; i++) {
      if (mm->inputs[i] >= matmul)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      const struct panvk_graph_value *v = graph->values + mm->inputs[i];
      if (v->format != VK_FORMAT_R8_SINT)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      if (i < 2) {
         uint32_t rows = i ? info->inner : info->rows;
         uint32_t columns = i ? info->columns : info->inner;
         if (v->kind != PANVK_GRAPH_INPUT || v->rank != 3 || v->input_count ||
             v->dims[0] != info->batch || v->dims[1] != rows ||
             v->dims[2] != columns)
            return VK_ERROR_FEATURE_NOT_PRESENT;
      } else {
         int32_t zero = i == 2 ? info->a_zero_point : info->b_zero_point;
         if (v->kind != PANVK_GRAPH_CONSTANT || v->rank != 1 ||
             v->dims[0] != 1 || v->elements != 1 || v->size != 1 || !v->data ||
             *(const int8_t *)v->data != zero)
            return VK_ERROR_FEATURE_NOT_PRESENT;
      }
   }
   struct value_map {
      uint32_t normalized;
      bool used;
   };
   size_t value_count = graph->count, adds_size = add_count;
   if (SIZE_MAX / sizeof(struct value_map) < value_count ||
       SIZE_MAX / sizeof(struct pan_graph_matmul_add) < adds_size ||
       SIZE_MAX / sizeof(uint32_t) <= adds_size)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct value_map *map = malloc((size_t)graph->count * sizeof(*map));
   struct panvk_data_graph_matmul_add extracted = {
      .info = *info,
      .add_count = add_count,
      .inputs = {mm->inputs[0], mm->inputs[1]},
      .input_indices = {graph->values[mm->inputs[0]].resource_index,
                        graph->values[mm->inputs[1]].resource_index},
      .output = output,
      .values = malloc(((size_t)add_count + 1) * sizeof(uint32_t)),
      .adds =
         add_count ? malloc((size_t)add_count * sizeof(*extracted.adds)) : NULL,
   };
   VkResult status = VK_ERROR_OUT_OF_HOST_MEMORY;
   if (!map || !extracted.values || (add_count && !extracted.adds))
      goto cleanup;
   for (uint32_t i = 0; i < graph->count; i++)
      map[i] = (struct value_map){.normalized = UINT32_MAX};
   uint32_t next = 1;
   map[matmul].normalized = 0;
   extracted.values[0] = matmul;
   status = VK_ERROR_FEATURE_NOT_PRESENT;
   for (uint32_t i = 0; i < graph->count; i++) {
      const struct panvk_graph_value *v = graph->values + i;
      if (!v->live)
         continue;
      if (v->kind == PANVK_GRAPH_OPERATION && v->op == PANVK_GRAPH_OP_ADD) {
         if (i < matmul || v->input_count != 2 || !same_shape(mm, v))
            goto cleanup;
         struct pan_graph_matmul_add *add = extracted.adds + next - 1;
         for (uint32_t j = 0; j < 2; j++) {
            uint32_t input = v->inputs[j];
            if (input >= i || map[input].normalized == UINT32_MAX)
               goto cleanup;
            add->input[j] = map[input].normalized;
            map[input].used = true;
         }
         map[i].normalized = next;
         extracted.values[next++] = i;
      } else if (v->kind == PANVK_GRAPH_INPUT) {
         if (i != mm->inputs[0] && i != mm->inputs[1])
            goto cleanup;
      } else if (v->kind == PANVK_GRAPH_CONSTANT) {
         if (i != mm->inputs[2] && i != mm->inputs[3])
            goto cleanup;
      }
   }
   if (map[out->inputs[0]].normalized != add_count)
      goto cleanup;
   map[out->inputs[0]].used = true;
   for (uint32_t i = 0; i <= add_count; i++) {
      if (!map[extracted.values[i]].used)
         goto cleanup;
   }
   *result = extracted;
   extracted = (struct panvk_data_graph_matmul_add){0};
   status = VK_SUCCESS;
cleanup:
   free(map);
   panvk_data_graph_matmul_add_finish(&extracted);
   return status;
}
