#include "pan_graph_memory_registry.h"
#include "pan_graph_memory_layout.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

enum callback {
   UNKNOWN,
   ALLOW,
   DENY,
   RESHAPE,
   INTERNAL,
};

struct state {
   size_t pending;
   bool allowed;
};

struct tensor_info {
   size_t consumer_count;
   size_t group_count;
   uint32_t minimum_width;
};

struct relation {
   size_t parent;
   size_t child;
};

static enum callback
callback(uint32_t opcode)
{
   static const uint32_t allowed[] = {
      32769, 32770, 32771, 33793, 37889, 40961, 40962, 40963,
      41986, 41988, 41990, 41991, 41992, 41993,
      43009, 43010, 43011, 43012, 43014, 43015,
      53249, 53250, 53251, 53252, 53253, 53254, 53255, 53256,
      53257, 53258, 53259, 53260, 53261, 53262, 53263, 53265,
      53266, 53267, 53268, 53269, 53270,
      57345, 57346, 57348, 57352, 57356, 57361,
      98306, 98307, 98308, 98309, 98310, 98311, 98312, 100353,
   };
   if (opcode == PAN_GRAPH_MEMORY_REGISTRY_FUNCTION ||
       (opcode >= 0x10000 && opcode <= 0x10003))
      return INTERNAL;
   if (opcode == PAN_GRAPH_MEMORY_REGISTRY_RESHAPE)
      return RESHAPE;
   if (opcode == 98305 || opcode == 98313)
      return DENY;
   for (unsigned i = 0; i < sizeof(allowed) / sizeof(allowed[0]); i++) {
      if (allowed[i] == opcode)
         return ALLOW;
   }
   return UNKNOWN;
}

static bool
shape_valid(const struct pan_graph_memory_registry_shape *shape)
{
   if (!shape->rank || shape->rank > PAN_GRAPH_MEMORY_REGISTRY_MAX_RANK)
      return false;
   for (unsigned i = 0; i < shape->rank; i++) {
      if (!shape->dims[i] || shape->dims[i] > INT32_MAX)
         return false;
   }
   return true;
}

static bool
shape_equal(const struct pan_graph_memory_registry_shape *a,
            const struct pan_graph_memory_registry_shape *b)
{
   uint32_t rank = a->rank < 4 ? 4 : a->rank;
   if (rank != (b->rank < 4 ? 4 : b->rank))
      return false;
   for (unsigned i = 0; i < rank; i++) {
      uint32_t x = i < a->rank ? a->dims[i] : 1;
      uint32_t y = i < b->rank ? b->dims[i] : 1;
      if (x != y)
         return false;
   }
   return true;
}

static const struct pan_graph_memory_registry_view *
view(const struct pan_graph_memory_registry_view *views, uint32_t count,
     uint32_t key)
{
   for (uint32_t i = 0; i < count; i++) {
      if (views[i].key == key)
         return &views[i];
   }
   return NULL;
}

static bool
views_valid(const struct pan_graph_memory_registry_view *views, uint32_t count,
            uint32_t tensor_count)
{
   if (count && !views)
      return false;
   for (uint32_t i = 0; i < count; i++) {
      if (views[i].tensor >= tensor_count || !shape_valid(&views[i].shape))
         return false;
      for (uint32_t j = 0; j < i; j++) {
         if (views[i].key == views[j].key)
            return false;
      }
   }
   return true;
}

static bool
graph_valid(const struct pan_graph_memory_registry_graph *graph,
            size_t *relation_count)
{
   if (!graph || !graph->tensor_count || !graph->tensors ||
       (graph->operation_count && !graph->operations))
      return false;
   for (uint32_t i = 0; i < graph->tensor_count; i++) {
      const struct pan_graph_memory_registry_tensor *tensor = &graph->tensors[i];
      if (!shape_valid(&tensor->shape) ||
          (tensor->bits != 8 && tensor->bits != 32))
         return false;
   }
   *relation_count = 0;
   for (uint32_t i = 0; i < graph->operation_count; i++) {
      const struct pan_graph_memory_registry_operation *op = &graph->operations[i];
      if (callback(op->opcode) == UNKNOWN ||
          !views_valid(op->inputs, op->input_count, graph->tensor_count) ||
          !views_valid(op->outputs, op->output_count, graph->tensor_count))
         return false;
      bool recursive = op->opcode == PAN_GRAPH_MEMORY_REGISTRY_RESHAPE ||
                       op->opcode == PAN_GRAPH_MEMORY_REGISTRY_OP_18002;
      if (recursive || op->opcode == PAN_GRAPH_MEMORY_REGISTRY_GATHER ||
          op->opcode == PAN_GRAPH_MEMORY_REGISTRY_SCATTER) {
         if (!view(op->inputs, op->input_count, 1) ||
             !view(op->outputs, op->output_count, 2))
            return false;
      }
      if (recursive) {
         if ((size_t)op->output_count > SIZE_MAX / 2)
            return false;
         size_t count = (size_t)op->output_count * 2;
         if (SIZE_MAX - count < op->input_count)
            return false;
         count += op->input_count;
         if (SIZE_MAX - *relation_count < count)
            return false;
         *relation_count += count;
      }
      if (op->opcode == PAN_GRAPH_MEMORY_REGISTRY_FUNCTION) {
         for (unsigned direction = 0; direction < 2; direction++) {
            uint32_t size = direction ? op->input_count : op->output_count;
            const struct pan_graph_memory_registry_view *views =
               direction ? op->inputs : op->outputs;
            if (size > (UINT32_MAX >> 8))
               return false;
            for (uint32_t j = 0; j < size; j++) {
               if (!view(views, size, (direction ? 1 : 2) + (j << 8)))
                  return false;
            }
         }
      }
   }
   return true;
}

static void
relate(struct relation *relations, size_t *count, struct state *states,
       size_t parent, size_t child)
{
   relations[(*count)++] = (struct relation){parent, child};
   states[parent].pending++;
}

enum pan_graph_tsu_status
pan_graph_memory_registry_analyze(
   const struct pan_graph_memory_registry_graph *graph, uint32_t tensor,
   uint32_t paired, struct pan_graph_memory_registry_result *result)
{
   size_t capacity;
   if (!result || !graph_valid(graph, &capacity) || tensor >= graph->tensor_count ||
       (paired != PAN_GRAPH_MEMORY_REGISTRY_NO_PAIR &&
        (paired >= graph->tensor_count || !graph->tensors[paired].block[0])))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   size_t count = (size_t)graph->tensor_count * 2;
   if (count / 2 != graph->tensor_count ||
       count > SIZE_MAX / sizeof(struct state) ||
       count > SIZE_MAX / sizeof(size_t) ||
       (size_t)graph->tensor_count > SIZE_MAX / sizeof(struct tensor_info) ||
       capacity > SIZE_MAX / sizeof(struct relation))
      return PAN_GRAPH_TSU_UNSUPPORTED;
   struct state *states = calloc(count, sizeof(*states));
   struct tensor_info *info = calloc(graph->tensor_count, sizeof(*info));
   struct relation *relations = capacity ? calloc(capacity, sizeof(*relations)) : NULL;
   size_t *queue = calloc(count, sizeof(*queue));
   enum pan_graph_tsu_status status = PAN_GRAPH_TSU_OUT_OF_MEMORY;
   if (!states || !info || (capacity && !relations) || !queue)
      goto done;
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   for (size_t i = 0; i < count; i++)
      states[i].allowed = true;
   for (uint32_t i = 0; i < graph->tensor_count; i++)
      info[i].minimum_width = graph->tensors[i].shape.dims[0];
   for (uint32_t i = 0; i < graph->operation_count; i++) {
      const struct pan_graph_memory_registry_operation *op = &graph->operations[i];
      for (unsigned direction = 0; direction < 2; direction++) {
         const struct pan_graph_memory_registry_view *views =
            direction ? op->inputs : op->outputs;
         uint32_t size = direction ? op->input_count : op->output_count;
         for (uint32_t j = 0; j < size; j++) {
            struct tensor_info *data = &info[views[j].tensor];
            if (op->opcode & 0x8000) {
               if (views[j].shape.dims[0] < data->minimum_width)
                  data->minimum_width = views[j].shape.dims[0];
               if (direction) {
                  if (data->consumer_count == SIZE_MAX)
                     goto done;
                  data->consumer_count++;
               }
            }
            if (!direction && op->opcode == PAN_GRAPH_MEMORY_REGISTRY_CE_GROUP) {
               if (data->group_count == SIZE_MAX)
                  goto done;
               data->group_count++;
            }
         }
      }
   }
   size_t relations_count = 0;
   for (uint32_t i = 0; i < graph->operation_count; i++) {
      const struct pan_graph_memory_registry_operation *op = &graph->operations[i];
      enum callback kind = callback(op->opcode);
      bool recursive = op->opcode == PAN_GRAPH_MEMORY_REGISTRY_RESHAPE ||
                       op->opcode == PAN_GRAPH_MEMORY_REGISTRY_OP_18002;
      const struct pan_graph_memory_registry_view *input =
         view(op->inputs, op->input_count, 1);
      const struct pan_graph_memory_registry_view *output =
         view(op->outputs, op->output_count, 2);
      bool allowed = kind != DENY &&
                     (kind != RESHAPE || shape_equal(&input->shape, &output->shape));
      for (unsigned direction = 0; direction < 2; direction++) {
         const struct pan_graph_memory_registry_view *views =
            direction ? op->inputs : op->outputs;
         uint32_t size = direction ? op->input_count : op->output_count;
         for (uint32_t j = 0; j < size; j++) {
            size_t state = (size_t)views[j].tensor * 2 + direction;
            if (op->opcode == PAN_GRAPH_MEMORY_REGISTRY_FUNCTION) {
               if (views[j].attributes)
                  states[state].allowed = false;
            } else if (kind != INTERNAL) {
               if (!allowed || (!direction && op->opcode == PAN_GRAPH_MEMORY_REGISTRY_SCATTER) ||
                   (direction && (op->opcode == PAN_GRAPH_MEMORY_REGISTRY_SCATTER ||
                                  op->opcode == PAN_GRAPH_MEMORY_REGISTRY_GATHER) &&
                    input->tensor == views[j].tensor))
                  states[state].allowed = false;
               if (recursive) {
                  size_t dependency = direction ? (size_t)output->tensor * 2 + 1 :
                                                 (size_t)input->tensor * 2;
                  relate(relations, &relations_count, states, state, dependency);
                  if (!direction && info[input->tensor].consumer_count > 1)
                     relate(relations, &relations_count, states, state, dependency + 1);
               }
            }
         }
      }
   }
   size_t head = 0, tail = 0;
   for (size_t i = 0; i < count; i++) {
      if (!states[i].pending)
         queue[tail++] = i;
   }
   while (head < tail) {
      size_t child = queue[head++];
      for (size_t i = 0; i < relations_count; i++) {
         if (relations[i].child == child) {
            size_t parent = relations[i].parent;
            states[parent].allowed &= states[child].allowed;
            if (!--states[parent].pending)
               queue[tail++] = parent;
         }
      }
   }
   status = PAN_GRAPH_TSU_UNSUPPORTED;
   if (head != count)
      goto done;
   const struct pan_graph_memory_registry_tensor *value = &graph->tensors[tensor];
   struct pan_graph_memory_registry_result plan = {
      .layout = value->layout,
      .producer_allowed = states[(size_t)tensor * 2].allowed,
      .consumer_allowed = states[(size_t)tensor * 2 + 1].allowed,
      .producer_flag = value->producer_flag,
   };
   memcpy(plan.block, value->block, sizeof(plan.block));
   if (value->layout == 3 && value->shape.rank <= 4 &&
       plan.producer_allowed && plan.consumer_allowed) {
      uint32_t width = info[tensor].minimum_width;
      if (paired != PAN_GRAPH_MEMORY_REGISTRY_NO_PAIR &&
          graph->tensors[paired].block[0] < width)
         width = graph->tensors[paired].block[0];
      plan.layout = width <= 512 / 16 / value->bits ? 260 :
                    width <= 512 / 8 / value->bits ? 261 : 262;
      pan_graph_memory_layout_v15_block(plan.layout, value->bits, plan.block);
      plan.producer_flag = info[tensor].group_count == 1 &&
                           (paired == PAN_GRAPH_MEMORY_REGISTRY_NO_PAIR ||
                            info[paired].group_count == 1);
      plan.changed = true;
   }
   *result = plan;
   status = PAN_GRAPH_TSU_SUCCESS;
done:
   free(queue);
   free(relations);
   free(info);
   free(states);
   return status;
}
