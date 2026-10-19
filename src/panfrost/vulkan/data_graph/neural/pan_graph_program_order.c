#include "pan_graph_program_order.h"

#include <limits.h>
#include <stdlib.h>

struct order_node {
   uint32_t previous;
   uint32_t next;
   uint32_t remaining;
   bool done;
};

static uint32_t
find_node(const struct pan_graph_program_order_node *nodes, size_t count,
          uint64_t uid)
{
   for (size_t i = 0; i < count; i++) {
      if (nodes[i].uid == uid)
         return i;
   }
   return UINT32_MAX;
}

static enum pan_graph_tsu_status
build(const struct pan_graph_program_order_node *nodes, size_t count,
      bool selected, struct pan_graph_program_order *result)
{
   if (!result || (count && !nodes) || count >= UINT32_MAX ||
       count > SIZE_MAX / sizeof(struct order_node) ||
       count > SIZE_MAX / sizeof(uint32_t))
      return PAN_GRAPH_TSU_INVALID;

   for (size_t i = 0; i < count; i++) {
      if ((nodes[i].dependency_count && !nodes[i].dependencies) ||
          nodes[i].dependency_count > INT32_MAX)
         return PAN_GRAPH_TSU_INVALID;
      for (size_t j = 0; j < i; j++) {
         if (nodes[i].uid == nodes[j].uid)
            return PAN_GRAPH_TSU_INVALID;
      }
   }

   if (!count) {
      *result = (struct pan_graph_program_order){0};
      return PAN_GRAPH_TSU_SUCCESS;
   }

   struct order_node *work = malloc(count * sizeof(*work));
   uint32_t *indices = malloc(count * sizeof(*indices));
   if (!work || !indices) {
      free(work);
      free(indices);
      return PAN_GRAPH_TSU_OUT_OF_MEMORY;
   }

   size_t queued = 0;
   for (size_t i = 0; i < count; i++) {
      work[i] = (struct order_node){0};
      for (size_t j = 0; j < nodes[i].dependency_count; j++) {
         if (find_node(nodes, count, nodes[i].dependencies[j]) != UINT32_MAX)
            work[i].remaining++;
      }
      if (!work[i].remaining)
         indices[queued++] = i;
   }
   for (size_t next = 0; next < queued; next++) {
      uint64_t uid = nodes[indices[next]].uid;
      for (size_t i = 0; i < count; i++) {
         for (size_t j = 0; j < nodes[i].dependency_count; j++) {
            if (nodes[i].dependencies[j] == uid &&
                !--work[i].remaining)
               indices[queued++] = i;
         }
      }
   }
   if (queued != count) {
      free(work);
      free(indices);
      return PAN_GRAPH_TSU_INVALID;
   }

   for (size_t i = 0; i < count; i++) {
      indices[i] = i;
      if (selected) {
         size_t j = i;
         while (j && nodes[indices[j - 1]].uid < nodes[i].uid) {
            indices[j] = indices[j - 1];
            j--;
         }
         indices[j] = i;
      }
   }
   for (size_t i = 0; i < count; i++) {
      work[indices[i]].previous = i ? indices[i - 1] : UINT32_MAX;
      work[indices[i]].next = i + 1 < count ? indices[i + 1] : UINT32_MAX;
      work[indices[i]].done = false;
   }

   uint32_t head = indices[0];
   for (uint32_t current = head; current != UINT32_MAX;) {
      uint32_t following = work[current].next;
      uint32_t dependency = UINT32_MAX;
      for (size_t i = 0; i < nodes[current].dependency_count; i++) {
         uint32_t node = find_node(nodes, count, nodes[current].dependencies[i]);
         if (node != UINT32_MAX && !work[node].done) {
            dependency = node;
            break;
         }
      }
      if (dependency == UINT32_MAX) {
         work[current].done = true;
      } else {
         uint32_t previous = work[current].previous;
         if (previous == UINT32_MAX)
            head = following;
         else
            work[previous].next = following;
         if (following != UINT32_MAX)
            work[following].previous = previous;

         uint32_t after = work[dependency].next;
         work[current].previous = dependency;
         work[current].next = after;
         work[dependency].next = current;
         if (after != UINT32_MAX)
            work[after].previous = current;
      }
      current = following;
   }

   size_t length = 0;
   for (uint32_t current = head; current != UINT32_MAX;
        current = work[current].next)
      indices[length++] = current;
   free(work);
   *result = (struct pan_graph_program_order){indices, count};
   return PAN_GRAPH_TSU_SUCCESS;
}

enum pan_graph_tsu_status
pan_graph_program_order_build_dependencies(
   const struct pan_graph_program_order_node *nodes, size_t count,
   struct pan_graph_program_order *result)
{
   return build(nodes, count, false, result);
}

enum pan_graph_tsu_status
pan_graph_program_order_build_selected(
   const struct pan_graph_program_order_node *nodes, size_t count,
   struct pan_graph_program_order *result)
{
   return build(nodes, count, true, result);
}

void
pan_graph_program_order_finish(struct pan_graph_program_order *result)
{
   if (!result)
      return;
   free(result->indices);
   *result = (struct pan_graph_program_order){0};
}
