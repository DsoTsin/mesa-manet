#include "panvk_data_graph.h"

#include "compiler/spirv/spirv.h"
#include "neural/pan_graph_alloc.h"
#include "util/ralloc.h"
#include "util/u_math.h"
#include "vk_format.h"
#include "vk_util.h"

struct graph_id {
   const uint32_t *words;
   uint32_t value;
   uint32_t set;
   uint32_t binding;
};

struct graph_parser {
   struct graph_id *ids;
   uint32_t bound;
   struct panvk_data_graph *graph;
   const VkDataGraphPipelineCreateInfoARM *info;
   const VkDataGraphPipelineShaderModuleCreateInfoARM *module;
   bool failed;
   bool oom;
};

static const uint32_t *
instruction(struct graph_parser *p, uint32_t id, SpvOp op, unsigned min_words)
{
   if (id && id < p->bound && p->ids[id].words) {
      const uint32_t *w = p->ids[id].words;
      if ((w[0] & 0xffff) == op && (w[0] >> 16) >= min_words)
         return w;
   }
   p->failed = true;
   return NULL;
}

static uint32_t
scalar(struct graph_parser *p, uint32_t id)
{
   const uint32_t *w = instruction(p, id, SpvOpConstant, 4);
   if (!w || (w[0] >> 16) != 4) {
      p->failed = true;
      return 0;
   }
   const uint32_t *type = instruction(p, w[1], SpvOpTypeInt, 4);
   if (!type || (type[0] >> 16) != 4 || type[2] != 32 || type[3] > 1) {
      p->failed = true;
      return 0;
   }
   return w[3];
}

static VkFormat
scalar_format(const uint32_t *s)
{
   if ((s[0] & 0xffff) == SpvOpTypeFloat && (s[0] >> 16) == 3 && s[2] == 32)
      return VK_FORMAT_R32_SFLOAT;
   if ((s[0] & 0xffff) == SpvOpTypeInt && (s[0] >> 16) == 4) {
      if (s[2] == 8 && s[3] <= 1)
         return VK_FORMAT_R8_SINT;
      if (s[2] == 32 && s[3] <= 1)
         return VK_FORMAT_R32_SINT;
   }
   return VK_FORMAT_UNDEFINED;
}

static bool
constant_vector(struct graph_parser *p, uint32_t id, unsigned count,
                uint32_t *values, bool signed_values)
{
   if (!id || id >= p->bound || !p->ids[id].words)
      return false;
   const uint32_t *w = p->ids[id].words;
   unsigned op = w[0] & 0xffff;
   if ((op != SpvOpConstantComposite || (w[0] >> 16) != count + 3) &&
       (op != SpvOpConstantCompositeReplicateEXT || (w[0] >> 16) != 4) &&
       (op != SpvOpConstantNull || (w[0] >> 16) != 3))
      return false;
   if (!w[1] || w[1] >= p->bound || !p->ids[w[1]].words)
      return false;
   const uint32_t *container = p->ids[w[1]].words;
   if ((container[0] & 0xffff) == SpvOpTypeTensorARM) {
      if ((container[0] >> 16) != 5 || scalar(p, container[3]) != 1)
         return false;
      const uint32_t *shape =
         instruction(p, container[4], SpvOpConstantComposite, 4);
      if (!shape || (shape[0] >> 16) != 4 || scalar(p, shape[3]) != count)
         return false;
   } else if ((container[0] & 0xffff) != SpvOpTypeArray ||
              (container[0] >> 16) != 4 || scalar(p, container[3]) != count) {
      return false;
   }
   const uint32_t *type = instruction(p, container[2], SpvOpTypeInt, 4);
   if (!type || (type[0] >> 16) != 4 || (type[2] != 32 && type[2] != 64) ||
       type[3] > 1)
      return false;
   if (op == SpvOpConstantNull) {
      memset(values, 0, count * sizeof(*values));
      return !p->failed;
   }
   for (unsigned i = 0; i < count; i++) {
      const uint32_t *value = instruction(
         p, w[op == SpvOpConstantCompositeReplicateEXT ? 3 : 3 + i],
         SpvOpConstant, 4);
      if (!value || value[1] != container[2] ||
          (value[0] >> 16) != 3 + type[2] / 32 ||
          (type[2] == 64 && value[4] !=
             (signed_values && (value[3] & 0x80000000) ? UINT32_MAX : 0)) ||
          (!signed_values && type[2] == 32 && type[3] &&
           (value[3] & 0x80000000)))
         return false;
      values[i] = value[3];
   }
   return !p->failed;
}

static bool
constant_shape(struct graph_parser *p, uint32_t id, unsigned count,
               uint32_t *values)
{
   return constant_vector(p, id, count, values, false);
}

static bool
nan_mode(struct graph_parser *p, uint32_t id, uint32_t *mode)
{
   uint32_t value = scalar(p, id);
   if (p->failed || value > 2)
      return false;
   *mode = value == 2 ? 1 : 0;
   return true;
}

static bool
constant_boolean(struct graph_parser *p, uint32_t id, uint32_t *value)
{
   if (!id || id >= p->bound || !p->ids[id].words)
      return false;
   const uint32_t *w = p->ids[id].words;
   unsigned op = w[0] & 0xffff;
   if ((w[0] >> 16) != 3 ||
       (op != SpvOpConstantTrue && op != SpvOpConstantFalse) ||
       !instruction(p, w[1], SpvOpTypeBool, 2))
      return false;
   *value = op == SpvOpConstantTrue;
   return true;
}

static bool
shape(struct graph_parser *p, struct panvk_graph_value *v, uint32_t type)
{
   if (type && type < p->bound && p->ids[type].words) {
      const uint32_t *s = p->ids[type].words;
      v->format = scalar_format(s);
      if (v->format) {
         v->rank = 1;
         v->dims[0] = 1;
         return true;
      }
   }
   const uint32_t *w = instruction(p, type, SpvOpTypeTensorARM, 4);
   if (!w)
      return false;
   v->rank = scalar(p, w[3]);
   if (p->failed || !v->rank || v->rank > ARRAY_SIZE(v->dims))
      return false;
   if (!w[2] || w[2] >= p->bound || !p->ids[w[2]].words)
      return false;
   const uint32_t *s = p->ids[w[2]].words;
   v->format = scalar_format(s);
   if (v->format == VK_FORMAT_UNDEFINED)
      return false;
   if ((w[0] >> 16) == 5) {
      const uint32_t *dims = instruction(p, w[4], SpvOpConstantComposite, 3);
      if (!dims || (dims[0] >> 16) != v->rank + 3)
         return false;
      for (unsigned d = 0; d < v->rank; d++)
         v->dims[d] = scalar(p, dims[3 + d]);
   } else if ((w[0] >> 16) != 4) {
      return false;
   }
   return !p->failed;
}

static bool
finish_shape(struct panvk_graph_value *v)
{
   unsigned element_size = vk_format_get_blocksize(v->format);
   uint64_t size = element_size;
   for (unsigned d = 0; d < v->rank; d++) {
      if (!v->dims[d] || size > UINT32_MAX / v->dims[d])
         return false;
      size *= v->dims[d];
   }
   v->size = size;
   v->elements = size / element_size;
   return true;
}

static bool
description(struct panvk_graph_value *v, const VkTensorDescriptionARM *desc)
{
   if (!desc || desc->dimensionCount != v->rank ||
       (desc->tiling != VK_TENSOR_TILING_LINEAR_ARM &&
        desc->tiling != VK_TENSOR_TILING_OPTIMAL_ARM))
      return false;
   bool compatible = desc->format == v->format;
   if (v->format == VK_FORMAT_R8_SINT)
      compatible |= desc->format == VK_FORMAT_R8_UINT;
   else if (v->format == VK_FORMAT_R32_SINT)
      compatible |= desc->format == VK_FORMAT_R32_UINT;
   if (!compatible)
      return false;
   for (unsigned d = 0; d < v->rank; d++) {
      if (desc->pDimensions[d] <= 0 || desc->pDimensions[d] > UINT32_MAX ||
          (v->dims[d] && v->dims[d] != desc->pDimensions[d]))
         return false;
      v->dims[d] = desc->pDimensions[d];
   }
   v->storage_format = desc->format;
   return finish_shape(v);
}

static bool
resource(struct graph_parser *p, struct panvk_graph_value *v, uint32_t variable,
         uint32_t element)
{
   if (!variable || variable >= p->bound)
      return false;
   struct graph_id *id = &p->ids[variable];
   if (!instruction(p, variable, SpvOpVariable, 4))
      return false;
   for (unsigned i = 0; i < p->info->resourceInfoCount; i++) {
      const VkDataGraphPipelineResourceInfoARM *r = &p->info->pResourceInfos[i];
      if (r->descriptorSet != id->set || r->binding != id->binding ||
          r->arrayElement != element)
         continue;
      if (!description(v,
                       vk_find_struct_const(r->pNext, TENSOR_DESCRIPTION_ARM)))
         return false;
      v->set = id->set;
      v->binding = id->binding;
      v->array_element = element;
      return true;
   }
   return false;
}

static bool
copy_constant(struct graph_parser *p, const uint32_t *w,
              const struct panvk_graph_value *v, char *dst)
{
   unsigned op = w[0] & 0xffff, n = w[0] >> 16;
   unsigned element_size = vk_format_get_blocksize(v->format);
   if (op == SpvOpConstantNull) {
      if (n != 3)
         return false;
      memset(dst, 0, v->size);
      return true;
   }
   if (op == SpvOpConstant) {
      if (n != 4 || v->elements != 1 ||
          scalar_format(p->ids[w[1]].words) != v->format)
         return false;
      memcpy(dst, &w[3], element_size);
      return true;
   }
   if (((op != SpvOpConstantComposite || n != (uint64_t)v->dims[0] + 3) &&
        (op != SpvOpConstantCompositeReplicateEXT || n != 4)) ||
       !instruction(p, w[1], SpvOpTypeTensorARM, 4))
      return false;
   uint64_t offset = 0;
   for (unsigned i = 0; i < v->dims[0]; i++) {
      uint32_t id = w[op == SpvOpConstantCompositeReplicateEXT ? 3 : 3 + i];
      if (!id || id >= p->bound || !p->ids[id].words)
         return false;
      const uint32_t *child = p->ids[id].words;
      struct panvk_graph_value component = {0};
      if ((child[0] >> 16) < 3 || !shape(p, &component, child[1]) ||
          !finish_shape(&component) || component.format != v->format)
         return false;
      bool tensor =
         (p->ids[child[1]].words[0] & 0xffff) == SpvOpTypeTensorARM;
      if (tensor != (v->rank > 1) ||
          component.rank != (v->rank > 1 ? v->rank - 1 : 1))
         return false;
      for (unsigned d = 0; d + 1 < v->rank; d++)
         if (component.dims[d] != v->dims[d + 1])
            return false;
      if (component.size > v->size - offset ||
          !copy_constant(p, child, &component, dst + offset))
         return false;
      offset += component.size;
   }
   return offset == v->size && !p->failed;
}

static uint32_t
get_value(struct graph_parser *p, uint32_t id)
{
   if (!id || id >= p->bound || !p->ids[id].words)
      return UINT32_MAX;
   if (p->ids[id].value != UINT32_MAX)
      return p->ids[id].value;
   const uint32_t *w = p->ids[id].words;
   unsigned op = w[0] & 0xffff, n = w[0] >> 16;
   if (n < 3 || (op != SpvOpConstant && op != SpvOpConstantNull &&
                 op != SpvOpConstantComposite &&
                 op != SpvOpConstantCompositeReplicateEXT &&
                 op != SpvOpGraphConstantARM))
      return UINT32_MAX;
   struct panvk_graph_value v = {.kind = PANVK_GRAPH_CONSTANT};
   if (!shape(p, &v, w[1]))
      return UINT32_MAX;
   const VkTensorDescriptionARM *desc = NULL;
   const void *src = NULL;
   if (op == SpvOpGraphConstantARM) {
      if (n != 4)
         return UINT32_MAX;
      for (unsigned i = 0; i < p->module->constantCount; i++) {
         const VkDataGraphPipelineConstantARM *c = &p->module->pConstants[i];
         if (c->id != w[3])
            continue;
         desc = vk_find_struct_const(c->pNext, TENSOR_DESCRIPTION_ARM);
         src = c->pConstantData;
         break;
      }
      if (!src || !description(&v, desc) ||
          desc->tiling != VK_TENSOR_TILING_LINEAR_ARM)
         return UINT32_MAX;
   } else if (!finish_shape(&v)) {
      return UINT32_MAX;
   }
   v.data = rzalloc_size(p->graph, v.size);
   if (!v.data) {
      p->failed = true;
      p->oom = true;
      return UINT32_MAX;
   }
   char *dst = v.data;
   unsigned element_size = vk_format_get_blocksize(v.format);
   if (src) {
      for (uint32_t i = 0; i < v.elements; i++) {
         uint32_t remaining = i;
         uint64_t offset = 0, packed_stride = element_size;
         for (int d = v.rank - 1; d >= 0; d--) {
            uint64_t stride =
               desc->pStrides ? desc->pStrides[d] : packed_stride;
            if (!stride || stride > UINT32_MAX)
               return UINT32_MAX;
            offset += (remaining % v.dims[d]) * stride;
            remaining /= v.dims[d];
            packed_stride *= v.dims[d];
         }
         if (offset > SIZE_MAX - element_size)
            return UINT32_MAX;
         memcpy(dst + i * element_size, (const char *)src + offset,
                element_size);
      }
   } else if (!copy_constant(p, w, &v, dst))
      return UINT32_MAX;
   if (p->failed)
      return UINT32_MAX;
   uint32_t index = p->graph->count++;
   p->graph->values[index] = v;
   p->ids[id].value = index;
   return index;
}

static bool
broadcast(struct panvk_graph_value *v, const struct panvk_graph_value *a)
{
   if (a->rank > v->rank || a->format != v->format)
      return false;
   for (unsigned d = 0; d < v->rank; d++) {
      unsigned extent =
         d + a->rank < v->rank ? 1 : a->dims[d + a->rank - v->rank];
      if (v->dims[d] && extent != 1 && extent != v->dims[d])
         return false;
      v->dims[d] = MAX2(v->dims[d], extent);
   }
   return true;
}

static bool
spatial_shape(struct panvk_graph_value *v, const struct panvk_graph_value *a,
              uint32_t channels)
{
   if (v->rank != 4 || a->rank != 4)
      return false;
   uint32_t dims[] = {a->dims[0], 0, 0, channels};
   for (unsigned d = 0; d < 2; d++) {
      uint64_t total =
         (uint64_t)a->dims[d + 1] + v->padding[d * 2] + v->padding[d * 2 + 1];
      if (!v->kernel[d] || !v->stride[d] || !v->dilation[d] ||
          total > INT32_MAX)
         return false;
      uint64_t window = (uint64_t)(v->kernel[d] - 1) * v->dilation[d] + 1;
      if (total < window || (total - window) % v->stride[d])
         return false;
      dims[d + 1] = (total - window) / v->stride[d] + 1;
   }
   for (unsigned d = 0; d < 4; d++) {
      if (v->dims[d] && v->dims[d] != dims[d])
         return false;
      v->dims[d] = dims[d];
   }
   return true;
}

static bool
zero_point(const struct panvk_graph_value *v, VkFormat format)
{
   return v->kind == PANVK_GRAPH_CONSTANT && v->format == format &&
          v->rank == 1 && v->elements == 1 &&
          (format != VK_FORMAT_R32_SFLOAT ||
           !(*(const uint32_t *)v->data & 0x7fffffff));
}

static bool
lower_quantized_conv(struct graph_parser *p, struct panvk_graph_value *v)
{
   struct panvk_data_graph *g = p->graph;
   const struct panvk_graph_value *a = &g->values[v->inputs[0]];
   const struct panvk_graph_value *w = &g->values[v->inputs[1]];
   if (w->kind != PANVK_GRAPH_CONSTANT)
      return true;
   const char *native_conv = getenv("PANVK_GRAPH_NATIVE_CONV");
   if (native_conv && !strcmp(native_conv, "1")) {
      struct pan_graph_conv_info info = {
         .batch = a->dims[0], .input_height = a->dims[1],
         .input_width = a->dims[2], .input_channels = a->dims[3],
         .output_channels = w->dims[0],
         .input_zero_point = *(int8_t *)g->values[v->inputs[3]].data,
         .weight_zero_point = *(int8_t *)g->values[v->inputs[4]].data,
      };
      memcpy(info.kernel, v->kernel, sizeof(info.kernel));
      memcpy(info.stride, v->stride, sizeof(info.stride));
      memcpy(info.dilation, v->dilation, sizeof(info.dilation));
      memcpy(info.padding, v->padding, sizeof(info.padding));
      struct panvk_graph_value zero = {
         .kind = PANVK_GRAPH_CONSTANT, .format = VK_FORMAT_R32_SINT,
         .rank = 1, .dims = {1},
      };
      struct panvk_graph_value conv = *v;
      conv.conv = rzalloc(g, struct pan_graph_conv_plan);
      zero.data = rzalloc(g, uint32_t);
      if (!conv.conv || !zero.data) {
         p->oom = true;
         return false;
      }
      if (!pan_graph_conv_i8_init(conv.conv, &info) || !finish_shape(&zero))
         return false;
      conv.inputs[2] = g->count;
      g->values[g->count++] = zero;
      uint32_t conv_index = g->count;
      g->passes[g->pass_count++] = g->count;
      g->values[g->count++] = conv;
      v->op = PANVK_GRAPH_OP_ADD;
      v->input_count = 2;
      v->inputs[1] = v->inputs[2];
      v->inputs[0] = conv_index;
      return true;
   }
   uint64_t inner = (uint64_t)w->dims[1] * w->dims[2] * w->dims[3];
   uint64_t rows = (uint64_t)v->dims[1] * v->dims[2];
   if (inner > UINT32_MAX || rows > UINT32_MAX)
      return false;
   struct panvk_graph_value weights = {
      .kind = PANVK_GRAPH_CONSTANT, .format = VK_FORMAT_R8_SINT,
      .rank = 3, .dims = {a->dims[0], inner, w->dims[0]},
   };
   struct panvk_graph_value columns = *v;
   columns.op = PANVK_GRAPH_OP_IM2COL;
   columns.format = VK_FORMAT_R8_SINT;
   columns.rank = 3;
   columns.dims[0] = a->dims[0];
   columns.dims[1] = rows;
   columns.dims[2] = inner;
   columns.dims[3] = 0;
   columns.attribute = v->dims[2];
   columns.input_count = 2;
   columns.inputs[1] = v->inputs[3];
   struct panvk_graph_value mm = {
      .kind = PANVK_GRAPH_OPERATION, .op = PANVK_GRAPH_OP_MATMUL,
      .format = VK_FORMAT_R32_SINT, .rank = 3,
      .dims = {a->dims[0], rows, w->dims[0]}, .input_count = 4,
      .inputs = {g->count + 1, g->count, v->inputs[3], v->inputs[4]},
   };
   if (!finish_shape(&weights) || !finish_shape(&columns) || !finish_shape(&mm))
      return false;
   weights.data = ralloc_size(g, weights.size);
   mm.matmul = rzalloc(g, struct pan_graph_matmul_plan);
   if (!weights.data || !mm.matmul) {
      p->oom = true;
      return false;
   }
   const struct pan_graph_matmul_info info = {
      .batch = mm.dims[0], .rows = rows, .inner = inner, .columns = w->dims[0],
      .a_zero_point = *(int8_t *)g->values[v->inputs[3]].data,
      .b_zero_point = *(int8_t *)g->values[v->inputs[4]].data,
   };
   if (!pan_graph_matmul_i8_init(mm.matmul, &info))
      return false;
   for (uint32_t batch = 0; batch < weights.dims[0]; batch++)
      for (uint32_t k = 0; k < inner; k++)
         for (uint32_t c = 0; c < w->dims[0]; c++)
            ((uint8_t *)weights.data)[(batch * inner + k) * w->dims[0] + c] =
               ((const uint8_t *)w->data)[c * inner + k];
   g->values[g->count++] = weights;
   g->passes[g->pass_count++] = g->count;
   g->values[g->count++] = columns;
   uint32_t mm_index = g->count;
   g->passes[g->pass_count++] = g->count;
   g->values[g->count++] = mm;
   struct panvk_graph_value reshape = *v;
   reshape.op = PANVK_GRAPH_OP_RESHAPE;
   reshape.input_count = 1;
   reshape.inputs[0] = mm_index;
   uint32_t reshape_index = g->count;
   g->passes[g->pass_count++] = g->count;
   g->values[g->count++] = reshape;
   v->op = PANVK_GRAPH_OP_ADD;
   v->input_count = 2;
   v->inputs[1] = v->inputs[2];
   v->inputs[0] = reshape_index;
   return true;
}

static bool
operation(struct graph_parser *p, const uint32_t *w)
{
   unsigned n = w[0] >> 16;
   const uint32_t *set =
      n >= 5 ? instruction(p, w[3], SpvOpExtInstImport, 2) : NULL;
   if (!set || !memchr(&set[2], 0, ((set[0] >> 16) - 2) * 4) ||
       strcmp((const char *)&set[2], "TOSA.001000.1"))
      return false;
   struct panvk_graph_value v = {.kind = PANVK_GRAPH_OPERATION,
                                 .op = (enum panvk_graph_op)w[4]};
   if (!shape(p, &v, w[1]))
      return false;
   if (v.format == VK_FORMAT_R8_SINT && v.op != PANVK_GRAPH_OP_RESHAPE &&
       v.op != PANVK_GRAPH_OP_TRANSPOSE && v.op != PANVK_GRAPH_OP_RESCALE &&
       v.op != PANVK_GRAPH_OP_RESIZE && v.op != PANVK_GRAPH_OP_CONCAT &&
       v.op != PANVK_GRAPH_OP_TABLE)
      return false;
   unsigned args[6];
   switch (v.op) {
   case PANVK_GRAPH_OP_AVG_POOL2D:
   case PANVK_GRAPH_OP_MAX_POOL2D:
      if (n != (v.op == PANVK_GRAPH_OP_AVG_POOL2D ? 12 : 10) ||
          v.format != VK_FORMAT_R32_SFLOAT ||
          !constant_shape(p, w[5], 2, v.kernel) ||
          !constant_shape(p, w[6], 2, v.stride) ||
          !constant_shape(p, w[7], 4, v.padding))
         return false;
      if (v.op == PANVK_GRAPH_OP_AVG_POOL2D ? scalar(p, w[8]) != 3
                                            : !nan_mode(p, w[8], &v.attribute))
         return false;
      v.dilation[0] = v.dilation[1] = 1;
      v.input_count = v.op == PANVK_GRAPH_OP_AVG_POOL2D ? 3 : 1;
      for (unsigned i = 0; i < v.input_count; i++)
         args[i] = 9 + i;
      break;
   case PANVK_GRAPH_OP_CONV2D:
      if (n != 15 || (v.format != VK_FORMAT_R32_SFLOAT &&
                      v.format != VK_FORMAT_R32_SINT) ||
          !constant_shape(p, w[5], 4, v.padding) ||
          !constant_shape(p, w[6], 2, v.stride) ||
          !constant_shape(p, w[7], 2, v.dilation) ||
          scalar(p, w[8]) != (v.format == VK_FORMAT_R32_SFLOAT ? 3 : 1) ||
          !constant_boolean(p, w[9], &v.attribute))
         return false;
      v.input_count = 5;
      for (unsigned i = 0; i < 5; i++)
         args[i] = 10 + i;
      break;
   case PANVK_GRAPH_OP_RESCALE: {
      uint32_t scale32, per_channel, input_unsigned, output_unsigned;
      if (n != 15 || (v.format != VK_FORMAT_R8_SINT &&
                      v.format != VK_FORMAT_R32_SINT) ||
          !constant_boolean(p, w[5], &scale32) || !scale32 ||
          !constant_boolean(p, w[7], &per_channel) ||
          !constant_boolean(p, w[8], &input_unsigned) ||
          !constant_boolean(p, w[9], &output_unsigned))
         return false;
      uint32_t rounding = scalar(p, w[6]);
      if (rounding < 1 || rounding > 3)
         return false;
      v.attribute = (rounding == 3) | (per_channel << 1) |
                    (input_unsigned << 2) | (output_unsigned << 3);
      v.input_count = 5;
      for (unsigned i = 0; i < 5; i++)
         args[i] = 10 + i;
      break;
   }
   case PANVK_GRAPH_OP_TABLE:
      if (n != 7 || v.format != VK_FORMAT_R8_SINT)
         return false;
      v.input_count = 2;
      args[0] = 5;
      args[1] = 6;
      break;
   case PANVK_GRAPH_OP_CONCAT:
      if (n < 7 || n - 6 > ARRAY_SIZE(v.inputs))
         return false;
      v.axis = scalar(p, w[5]);
      if (v.axis >= v.rank)
         return false;
      v.input_count = n - 6;
      for (unsigned i = 0; i < v.input_count; i++)
         args[i] = 6 + i;
      break;
   case PANVK_GRAPH_OP_RESIZE:
      if (n != 10 || v.rank != 4 ||
          !constant_shape(p, w[7], 4, (uint32_t *)v.resize_scale) ||
          !constant_vector(p, w[8], 2, (uint32_t *)v.resize_offset, true) ||
          !constant_vector(p, w[9], 2, (uint32_t *)v.resize_border, true))
         return false;
      v.attribute = scalar(p, w[5]);
      if (v.attribute != 1 && v.attribute != 2)
         return false;
      v.input_count = 1;
      args[0] = 6;
      break;
   case PANVK_GRAPH_OP_SIGMOID:
   case PANVK_GRAPH_OP_TANH:
   case PANVK_GRAPH_OP_EXP:
   case PANVK_GRAPH_OP_RECIPROCAL:
      if (n != 6 || v.format != VK_FORMAT_R32_SFLOAT)
         return false;
      v.input_count = 1;
      args[0] = 5;
      break;
   case PANVK_GRAPH_OP_REDUCE_MAX:
   case PANVK_GRAPH_OP_REDUCE_MIN:
      if (n != 8 || !nan_mode(p, w[6], &v.attribute))
         return false;
      v.axis = scalar(p, w[5]);
      v.input_count = 1;
      args[0] = 7;
      break;
   case PANVK_GRAPH_OP_REDUCE_PRODUCT:
   case PANVK_GRAPH_OP_REDUCE_SUM:
      if (n != 7)
         return false;
      v.axis = scalar(p, w[5]);
      v.input_count = 1;
      args[0] = 6;
      break;
   case PANVK_GRAPH_OP_ADD:
   case PANVK_GRAPH_OP_SUB:
      if (n != 7)
         return false;
      v.input_count = 2;
      args[0] = 5;
      args[1] = 6;
      break;
   case PANVK_GRAPH_OP_MUL:
      if (n != 8 || v.format != VK_FORMAT_R32_SFLOAT)
         return false;
      v.input_count = 2;
      args[0] = 5;
      args[1] = 6;
      break;
   case PANVK_GRAPH_OP_CLAMP:
      if (n != 9)
         return false;
      if (!nan_mode(p, w[7], &v.attribute))
         return false;
      v.input_count = 3;
      args[0] = 8;
      args[1] = 5;
      args[2] = 6;
      break;
   case PANVK_GRAPH_OP_MATMUL:
      if (n != 9 ||
          (v.format != VK_FORMAT_R32_SFLOAT && v.format != VK_FORMAT_R32_SINT))
         return false;
      v.input_count = 4;
      for (unsigned i = 0; i < 4; i++)
         args[i] = 5 + i;
      break;
   case PANVK_GRAPH_OP_RESHAPE:
   case PANVK_GRAPH_OP_TRANSPOSE:
      if (n != 7)
         return false;
      v.input_count = 1;
      args[0] = v.op == PANVK_GRAPH_OP_TRANSPOSE ? 6 : 5;
      break;
   default:
      return false;
   }
   for (unsigned i = 0; i < v.input_count; i++) {
      v.inputs[i] = get_value(p, w[args[i]]);
      if (v.inputs[i] == UINT32_MAX)
         return false;
   }
   if (v.op == PANVK_GRAPH_OP_CONV2D) {
      const struct panvk_graph_value *a = &p->graph->values[v.inputs[0]];
      const struct panvk_graph_value *weights = &p->graph->values[v.inputs[1]];
      const struct panvk_graph_value *bias = &p->graph->values[v.inputs[2]];
      VkFormat input_format = v.format == VK_FORMAT_R32_SINT
                                 ? VK_FORMAT_R8_SINT : v.format;
      if (a->format != input_format || weights->format != input_format ||
          bias->format != v.format || weights->rank != 4 || a->rank != 4 ||
          bias->rank != 1 || a->dims[3] != weights->dims[3] ||
          (bias->dims[0] != 1 && bias->dims[0] != weights->dims[0]) ||
          !zero_point(&p->graph->values[v.inputs[3]], input_format) ||
          !zero_point(&p->graph->values[v.inputs[4]], input_format))
         return false;
      v.kernel[0] = weights->dims[1];
      v.kernel[1] = weights->dims[2];
      if (!spatial_shape(&v, a, weights->dims[0]))
         return false;
   } else if (v.op == PANVK_GRAPH_OP_RESCALE || v.op == PANVK_GRAPH_OP_TABLE) {
      const struct panvk_graph_value *a = &p->graph->values[v.inputs[0]];
      if (a->rank != v.rank ||
          (a->format != VK_FORMAT_R8_SINT && a->format != VK_FORMAT_R32_SINT))
         return false;
      for (unsigned d = 0; d < v.rank; d++) {
         if (v.dims[d] && v.dims[d] != a->dims[d])
            return false;
         v.dims[d] = a->dims[d];
      }
      if (v.op == PANVK_GRAPH_OP_TABLE) {
         const struct panvk_graph_value *table = &p->graph->values[v.inputs[1]];
         if (a->format != VK_FORMAT_R8_SINT || table->format != a->format ||
             table->rank != 1 || table->elements != 256)
            return false;
      } else {
         const struct panvk_graph_value *mul = &p->graph->values[v.inputs[1]];
         const struct panvk_graph_value *shift = &p->graph->values[v.inputs[2]];
         const struct panvk_graph_value *izp = &p->graph->values[v.inputs[3]];
         const struct panvk_graph_value *ozp = &p->graph->values[v.inputs[4]];
         unsigned count = v.attribute & 2 ? v.dims[v.rank - 1] : 1;
         if (mul->kind != PANVK_GRAPH_CONSTANT || shift->kind != PANVK_GRAPH_CONSTANT ||
             mul->format != VK_FORMAT_R32_SINT || shift->format != VK_FORMAT_R8_SINT ||
             mul->rank != 1 || shift->rank != 1 || mul->elements != count ||
             shift->elements != count || !zero_point(izp, a->format) ||
             !zero_point(ozp, v.format) ||
             (a->format == VK_FORMAT_R32_SINT &&
              (*(int32_t *)izp->data || (v.attribute & 4))) ||
             (v.format == VK_FORMAT_R32_SINT &&
              (*(int32_t *)ozp->data || (v.attribute & 8))))
            return false;
         for (unsigned i = 0; i < count; i++)
            if (((int32_t *)mul->data)[i] < 0 || ((int8_t *)shift->data)[i] < 2 ||
                ((int8_t *)shift->data)[i] > 62)
               return false;
      }
   } else if (v.op == PANVK_GRAPH_OP_CONCAT) {
      uint64_t extent = 0;
      for (unsigned i = 0; i < v.input_count; i++) {
         const struct panvk_graph_value *a = &p->graph->values[v.inputs[i]];
         if (a->rank != v.rank || a->format != v.format)
            return false;
         extent += a->dims[v.axis];
         for (unsigned d = 0; d < v.rank; d++) {
            if (d == v.axis)
               continue;
            if (v.dims[d] && v.dims[d] != a->dims[d])
               return false;
            v.dims[d] = a->dims[d];
         }
      }
      if (extent > UINT32_MAX || (v.dims[v.axis] && v.dims[v.axis] != extent))
         return false;
      v.dims[v.axis] = extent;
   } else if (v.op == PANVK_GRAPH_OP_RESIZE) {
      const struct panvk_graph_value *a = &p->graph->values[v.inputs[0]];
      VkFormat expected = v.attribute == 2 && a->format == VK_FORMAT_R8_SINT
                             ? VK_FORMAT_R32_SINT : a->format;
      if (a->rank != 4 || v.format != expected ||
          (a->format != VK_FORMAT_R8_SINT && a->format != VK_FORMAT_R32_SFLOAT))
         return false;
      uint32_t dims[4] = {a->dims[0], 0, 0, a->dims[3]};
      for (unsigned d = 0; d < 2; d++) {
         int64_t num = v.resize_scale[2 * d], den = v.resize_scale[2 * d + 1];
         int64_t offset = v.resize_offset[d], border = v.resize_border[d];
         int64_t extent = (a->dims[d + 1] - 1) * num - offset + border;
         if (num <= 0 || den <= 0 || num > 2048 || den >= 16 * num ||
             offset < -num || offset >= 16 * num || border < -16 * num ||
             border >= num || extent < 0 || extent % den ||
             extent / den + 1 >= 16384 || a->dims[d + 1] >= 16384)
            return false;
         dims[d + 1] = extent / den + 1;
      }
      for (unsigned d = 0; d < 4; d++) {
         if (v.dims[d] && v.dims[d] != dims[d])
            return false;
         v.dims[d] = dims[d];
      }
   } else if (v.op == PANVK_GRAPH_OP_AVG_POOL2D ||
              v.op == PANVK_GRAPH_OP_MAX_POOL2D) {
      const struct panvk_graph_value *a = &p->graph->values[v.inputs[0]];
      if (a->format != v.format || a->rank != 4 ||
          v.padding[0] >= v.kernel[0] || v.padding[1] >= v.kernel[0] ||
          v.padding[2] >= v.kernel[1] || v.padding[3] >= v.kernel[1] ||
          (uint64_t)v.kernel[0] * v.kernel[1] > UINT32_MAX ||
          !spatial_shape(&v, a, a->dims[3]))
         return false;
      if (v.op == PANVK_GRAPH_OP_AVG_POOL2D &&
          (!zero_point(&p->graph->values[v.inputs[1]], v.format) ||
           !zero_point(&p->graph->values[v.inputs[2]], v.format)))
         return false;
   } else if (v.op == PANVK_GRAPH_OP_SIGMOID || v.op == PANVK_GRAPH_OP_TANH ||
              v.op == PANVK_GRAPH_OP_EXP || v.op == PANVK_GRAPH_OP_RECIPROCAL ||
              (v.op >= PANVK_GRAPH_OP_REDUCE_MAX &&
               v.op <= PANVK_GRAPH_OP_REDUCE_SUM)) {
      const struct panvk_graph_value *a = &p->graph->values[v.inputs[0]];
      bool reduction = v.op >= PANVK_GRAPH_OP_REDUCE_MAX;
      if (a->format != v.format || a->rank != v.rank ||
          (reduction && v.axis >= a->rank))
         return false;
      for (unsigned d = 0; d < v.rank; d++) {
         uint32_t extent = reduction && d == v.axis ? 1 : a->dims[d];
         if (v.dims[d] && v.dims[d] != extent)
            return false;
         v.dims[d] = extent;
      }
   } else if (v.op == PANVK_GRAPH_OP_MATMUL) {
      const struct panvk_graph_value *a = &p->graph->values[v.inputs[0]];
      const struct panvk_graph_value *b = &p->graph->values[v.inputs[1]];
      if (a->rank != 3 || b->rank != 3 || v.rank != 3 ||
          a->format != b->format ||
          a->format !=
             (v.format == VK_FORMAT_R32_SINT ? VK_FORMAT_R8_SINT : v.format) ||
          a->dims[0] != b->dims[0] || a->dims[2] != b->dims[1])
         return false;
      for (unsigned i = 2; i < 4; i++) {
         const struct panvk_graph_value *zp = &p->graph->values[v.inputs[i]];
         if (zp->kind != PANVK_GRAPH_CONSTANT || zp->format != a->format ||
             zp->rank != 1 || zp->elements != 1)
            return false;
         if (v.format == VK_FORMAT_R32_SFLOAT &&
             (*(uint32_t *)zp->data & 0x7fffffff) != 0)
            return false;
      }
      uint32_t dims[] = {a->dims[0], a->dims[1], b->dims[2]};
      for (unsigned d = 0; d < 3; d++) {
         if (v.dims[d] && v.dims[d] != dims[d])
            return false;
         v.dims[d] = dims[d];
      }
   } else if (v.op == PANVK_GRAPH_OP_RESHAPE ||
              v.op == PANVK_GRAPH_OP_TRANSPOSE) {
      const struct panvk_graph_value *a = &p->graph->values[v.inputs[0]];
      uint32_t shape_values[4];
      if (a->format != v.format ||
          !constant_shape(p, w[v.op == PANVK_GRAPH_OP_TRANSPOSE ? 5 : 6],
                          v.rank, shape_values))
         return false;
      unsigned axes = 0;
      for (unsigned d = 0; d < v.rank; d++) {
         uint32_t extent = shape_values[d];
         if (v.op == PANVK_GRAPH_OP_TRANSPOSE) {
            if (a->rank != v.rank || extent >= a->rank ||
                (axes & (1u << extent)))
               return false;
            axes |= 1u << extent;
            v.permutation[d] = extent;
            extent = a->dims[extent];
         }
         if (!extent || (v.dims[d] && v.dims[d] != extent))
            return false;
         v.dims[d] = extent;
      }
      if (!finish_shape(&v) || v.elements != a->elements)
         return false;
   } else {
      unsigned count = v.op == PANVK_GRAPH_OP_MUL ? 2 : v.input_count;
      for (unsigned i = 0; i < count; i++) {
         const struct panvk_graph_value *a = &p->graph->values[v.inputs[i]];
         if (((v.op != PANVK_GRAPH_OP_CLAMP || i == 0) && a->rank != v.rank) ||
             !broadcast(&v, a))
            return false;
      }
   }
   if (!finish_shape(&v) || p->failed)
      return false;
   if (v.op == PANVK_GRAPH_OP_CONV2D && v.format == VK_FORMAT_R32_SINT &&
       !lower_quantized_conv(p, &v))
      return false;
   if (v.op == PANVK_GRAPH_OP_MATMUL && v.format == VK_FORMAT_R32_SINT) {
      const struct panvk_graph_value *a = &p->graph->values[v.inputs[0]];
      const struct pan_graph_matmul_info info = {
         .batch = v.dims[0],
         .rows = v.dims[1],
         .inner = a->dims[2],
         .columns = v.dims[2],
         .a_zero_point = *(int8_t *)p->graph->values[v.inputs[2]].data,
         .b_zero_point = *(int8_t *)p->graph->values[v.inputs[3]].data,
      };
      v.matmul = rzalloc(p->graph, struct pan_graph_matmul_plan);
      if (!v.matmul) {
         p->oom = true;
         return false;
      }
      if (!pan_graph_matmul_i8_init(v.matmul, &info))
         return false;
   }
   uint32_t index = p->graph->count++;
   p->graph->values[index] = v;
   p->ids[w[2]].value = index;
   p->graph->passes[p->graph->pass_count++] = index;
   return true;
}

static bool
fold_constant_transforms(struct panvk_data_graph *g)
{
   for (unsigned i = 0; i < g->count; i++) {
      struct panvk_graph_value *v = &g->values[i];
      if (!v->live || v->kind != PANVK_GRAPH_OPERATION ||
          (v->op != PANVK_GRAPH_OP_RESHAPE &&
           v->op != PANVK_GRAPH_OP_TRANSPOSE))
         continue;
      const struct panvk_graph_value *a = &g->values[v->inputs[0]];
      if (a->kind != PANVK_GRAPH_CONSTANT)
         continue;
      v->data = ralloc_size(g, v->size);
      if (!v->data)
         return false;
      if (v->op == PANVK_GRAPH_OP_RESHAPE) {
         memcpy(v->data, a->data, v->size);
      } else {
         unsigned bytes = vk_format_get_blocksize(v->format);
         for (uint32_t index = 0; index < v->elements; index++) {
            uint32_t coords[4], remaining = index, source = 0;
            for (int d = v->rank - 1; d >= 0; d--) {
               coords[v->permutation[d]] = remaining % v->dims[d];
               remaining /= v->dims[d];
            }
            for (unsigned d = 0; d < a->rank; d++)
               source = source * a->dims[d] + coords[d];
            memcpy((char *)v->data + index * bytes,
                   (const char *)a->data + source * bytes, bytes);
         }
      }
      v->kind = PANVK_GRAPH_CONSTANT;
      v->input_count = 0;
   }
   return true;
}

static void
prune_values(struct panvk_data_graph *g)
{
   for (unsigned i = 0; i < g->count; i++)
      g->values[i].live = g->values[i].kind == PANVK_GRAPH_OUTPUT;
   for (unsigned i = g->count; i-- > 0;) {
      const struct panvk_graph_value *v = &g->values[i];
      if (v->live) {
         for (unsigned j = 0; j < v->input_count; j++)
            g->values[v->inputs[j]].live = true;
      }
   }
   unsigned count = 0;
   for (unsigned i = 0; i < g->pass_count; i++) {
      uint32_t id = g->passes[i];
      if (g->values[id].live && g->values[id].kind != PANVK_GRAPH_CONSTANT)
         g->passes[count++] = id;
   }
   g->pass_count = count;
}

static bool
is_pointwise(const struct panvk_graph_value *v)
{
   if (v->kind != PANVK_GRAPH_OPERATION)
      return false;
   switch (v->op) {
   case PANVK_GRAPH_OP_CLAMP:
   case PANVK_GRAPH_OP_SIGMOID:
   case PANVK_GRAPH_OP_TANH:
   case PANVK_GRAPH_OP_ADD:
   case PANVK_GRAPH_OP_EXP:
   case PANVK_GRAPH_OP_RECIPROCAL:
   case PANVK_GRAPH_OP_MUL:
   case PANVK_GRAPH_OP_SUB:
   case PANVK_GRAPH_OP_RESCALE:
   case PANVK_GRAPH_OP_TABLE:
      return true;
   default:
      return false;
   }
}

static bool
same_shape(const struct panvk_graph_value *a,
           const struct panvk_graph_value *b)
{
   return a->rank == b->rank && !memcmp(a->dims, b->dims, sizeof(a->dims));
}

static bool
fuse_pointwise(struct panvk_data_graph *g)
{
   uint32_t *users = rzalloc_array(g, uint32_t, g->count);
   uint32_t *consumer = rzalloc_array(g, uint32_t, g->count);
   if (!users || !consumer) {
      ralloc_free(users);
      ralloc_free(consumer);
      return false;
   }
   for (unsigned i = 0; i < g->count; i++)
      g->values[i].fused_into = UINT32_MAX;

   for (unsigned i = 0; i < g->count; i++) {
      const struct panvk_graph_value *v = &g->values[i];
      if (v->live) {
         for (unsigned j = 0; j < v->input_count; j++)
            users[v->inputs[j]]++;
      }
   }
   const char *rescale = getenv("PANVK_GRAPH_CONV_RESCALE");
   for (unsigned i = 0; rescale && !strcmp(rescale, "1") && i < g->count; i++) {
      struct panvk_graph_value *v = &g->values[i];
      if (!v->live || !v->conv)
         continue;
      struct panvk_graph_value *resize = &g->values[v->inputs[0]];
      if (resize->kind != PANVK_GRAPH_OPERATION ||
          resize->op != PANVK_GRAPH_OP_RESIZE || resize->attribute != 1 ||
          resize->format != VK_FORMAT_R8_SINT || users[v->inputs[0]] != 1)
         continue;
      const struct panvk_graph_value *input = &g->values[resize->inputs[0]];
      bool nearest = input->rank == 4 && resize->rank == 4;
      for (unsigned d = 0; d < 2; d++)
         nearest &= resize->resize_scale[2 * d] == 4 &&
                    resize->resize_scale[2 * d + 1] == 2 &&
                    resize->resize_offset[d] == -1 &&
                    (uint64_t)input->dims[d + 1] * 2 == resize->dims[d + 1];
      struct pan_graph_conv_info info = v->conv->info;
      info.input_nearest_2x = true;
      struct pan_graph_conv_plan plan;
      if (!nearest || !pan_graph_conv_i8_init(&plan, &info))
         continue;
      *v->conv = plan;
      resize->fused_into = i;
   }
   for (unsigned i = 0; rescale && !strcmp(rescale, "1") && i < g->count; i++) {
      struct panvk_graph_value *v = &g->values[i];
      if (!v->live || v->kind != PANVK_GRAPH_OPERATION ||
          v->op != PANVK_GRAPH_OP_RESCALE || v->format != VK_FORMAT_R8_SINT ||
          (v->attribute & ~3u))
         continue;
      struct panvk_graph_value *add = &g->values[v->inputs[0]];
      if (add->kind != PANVK_GRAPH_OPERATION || add->op != PANVK_GRAPH_OP_ADD ||
          users[v->inputs[0]] != 1 || !same_shape(v, add))
         continue;
      struct panvk_graph_value *conv = &g->values[add->inputs[0]];
      const struct panvk_graph_value *bias = &g->values[add->inputs[1]];
      if (!conv->conv || users[add->inputs[0]] != 1 ||
          !same_shape(v, conv) || bias->kind != PANVK_GRAPH_CONSTANT ||
          bias->format != VK_FORMAT_R32_SINT || bias->rank != 1 ||
          (bias->elements != 1 && bias->elements != v->dims[3]) ||
          conv->conv->info.weight_zero_point ||
          conv->conv->info.output_channels > UINT16_MAX ||
          conv->conv->info.dilation[0] != 1 ||
          conv->conv->info.dilation[1] != 1)
         continue;
      v->conv_rescale = conv;
      conv->fused_into = i;
      add->fused_into = i;
   }

   for (unsigned i = 0; rescale && !strcmp(rescale, "1") && i < g->count; i++) {
      struct panvk_graph_value *v = &g->values[i];
      if (!v->live || v->kind != PANVK_GRAPH_OPERATION ||
          v->op != PANVK_GRAPH_OP_TABLE || v->format != VK_FORMAT_R8_SINT)
         continue;
      struct panvk_graph_value *scale = &g->values[v->inputs[0]];
      const struct panvk_graph_value *table = &g->values[v->inputs[1]];
      if (!scale->conv_rescale || scale->op != PANVK_GRAPH_OP_RESCALE ||
          users[v->inputs[0]] != 1 || !same_shape(v, scale) ||
          table->kind != PANVK_GRAPH_CONSTANT || table->format != VK_FORMAT_R8_SINT ||
          table->rank != 1 || table->elements != 256)
         continue;
      v->conv_rescale = scale->conv_rescale;
      scale->conv_rescale->fused_into = i;
      g->values[scale->inputs[0]].fused_into = i;
      scale->fused_into = i;
   }

   for (unsigned phase = 0; phase < 2; phase++) {
      memset(users, 0, g->count * sizeof(*users));
      memset(consumer, 0xff, g->count * sizeof(*consumer));
      for (unsigned i = 0; i < g->count; i++) {
         const struct panvk_graph_value *v = &g->values[i];
         if (!v->live)
            continue;
         uint32_t dst = phase && v->fused_into != UINT32_MAX ? v->fused_into : i;
         for (unsigned j = 0; j < v->input_count; j++) {
            uint32_t src = v->inputs[j];
            if (phase && g->values[src].fused_into != UINT32_MAX)
               src = g->values[src].fused_into;
            if (src != dst && consumer[src] != dst) {
               if (!users[src])
                  consumer[src] = dst;
               users[src] = MIN2(users[src] + 1, 2);
            }
         }
      }
      for (unsigned i = 0; i < g->count; i++) {
         struct panvk_graph_value *v = &g->values[i];
         if (!v->live || v->kind != PANVK_GRAPH_OPERATION || users[i] != 1 ||
             v->fused_into != UINT32_MAX)
            continue;
         const struct panvk_graph_value *c = &g->values[consumer[i]];
         bool same = same_shape(v, c);
         if (phase) {
            if (!same || !((v->op != PANVK_GRAPH_OP_MATMUL && !v->conv &&
                            !v->conv_rescale && is_pointwise(c)) ||
                           c->kind == PANVK_GRAPH_OUTPUT))
               continue;
         } else {
            if (v->conv || v->conv_rescale || !is_pointwise(v) || !is_pointwise(c))
               continue;
            if (!same) {
               bool simple = v->op == PANVK_GRAPH_OP_ADD ||
                             v->op == PANVK_GRAPH_OP_SUB ||
                             v->op == PANVK_GRAPH_OP_MUL ||
                             v->op == PANVK_GRAPH_OP_RECIPROCAL;
               for (unsigned j = 0; j < v->input_count; j++)
                  simple &= g->values[v->inputs[j]].fused_into != i;
               if (!simple || v->rank != c->rank)
                  continue;
               bool broadcast = true;
               for (unsigned d = 0; d < v->rank; d++)
                  broadcast &= v->dims[d] == 1 ||
                               v->dims[d] == c->dims[d + c->rank - v->rank];
               if (!broadcast)
                  continue;
            }
         }
         v->fused_into = consumer[i];
      }
      for (unsigned i = g->count; i-- > 0;) {
         struct panvk_graph_value *v = &g->values[i];
         if (v->fused_into != UINT32_MAX) {
            uint32_t target = g->values[v->fused_into].fused_into;
            if (target != UINT32_MAX)
               v->fused_into = target;
         }
      }
   }
   unsigned count = 0;
   for (unsigned i = 0; i < g->pass_count; i++) {
      uint32_t pass = g->passes[i];
      if (g->values[pass].fused_into == UINT32_MAX)
         g->passes[count++] = pass;
   }
   g->pass_count = count;
   ralloc_free(users);
   ralloc_free(consumer);
   return true;
}

static VkResult
allocate_transient(struct panvk_data_graph *g)
{
   const char *fusion = getenv("PANVK_GRAPH_CONV_RESCALE");
   for (unsigned i = 0; fusion && !strcmp(fusion, "1") && i < g->pass_count; i++) {
      struct panvk_graph_value *v = &g->values[g->passes[i]];
      if (v->kind != PANVK_GRAPH_OPERATION || v->op != PANVK_GRAPH_OP_CONCAT ||
          v->rank != 4 || v->axis != 3 || v->format != VK_FORMAT_R8_SINT ||
          v->fused_into != UINT32_MAX || v->input_count < 2)
         continue;
      bool compatible = true;
      for (unsigned j = 0; j < v->input_count; j++) {
         const struct panvk_graph_value *a = &g->values[v->inputs[j]];
         compatible &= a->kind == PANVK_GRAPH_OPERATION && a->rank == 4 &&
                       a->op != PANVK_GRAPH_OP_CONCAT && !a->storage &&
                       a->fused_into == UINT32_MAX && !(a->dims[3] % 16);
         for (unsigned k = 0; k < j; k++)
            compatible &= v->inputs[j] != v->inputs[k];
      }
      if (!compatible)
         continue;
      uint64_t size = (uint64_t)v->dims[0] * v->dims[1] *
                      DIV_ROUND_UP(v->dims[2], 4) * v->dims[3] * 4;
      if (size > UINT32_MAX)
         continue;
      v->alias_concat = true;
      v->size = size;
      uint32_t offset = 0;
      for (unsigned j = 0; j < v->input_count; j++) {
         struct panvk_graph_value *a = &g->values[v->inputs[j]];
         a->storage = v;
         a->view_offset = offset * 4;
         offset += a->dims[3];
      }
   }
   for (unsigned i = 0; i < g->count; i++)
      g->values[i].last_use = 0;
   for (unsigned i = 0; i < g->pass_count; i++) {
      const struct panvk_graph_value *v = &g->values[g->passes[i]];
      g->values[g->passes[i]].last_use = MAX2(v->last_use, i);
      for (unsigned j = 0; j < v->input_count; j++)
         g->values[v->inputs[j]].last_use = i;
   }
   for (unsigned i = g->count; i-- > 0;) {
      const struct panvk_graph_value *v = &g->values[i];
      if (v->fused_into != UINT32_MAX) {
         for (unsigned j = 0; j < v->input_count; j++) {
            struct panvk_graph_value *a = &g->values[v->inputs[j]];
            a->last_use = MAX2(a->last_use, v->last_use);
         }
      }
   }
   for (unsigned i = 0; i < g->count; i++) {
      struct panvk_graph_value *v = &g->values[i];
      if (v->live && v->kind == PANVK_GRAPH_CONSTANT && v->elements > 1) {
         v->offset = g->constant_size;
         g->constant_size += ALIGN_POT(v->size, 64);
      }
   }
   struct pan_graph_alloc_request *requests =
      rzalloc_array(g, struct pan_graph_alloc_request, g->count);
   struct pan_graph_allocation *allocations =
      ralloc_array(g, struct pan_graph_allocation, g->pass_count);
   uint32_t *ids = ralloc_array(g, uint32_t, g->pass_count);
   if (!requests || !allocations || !ids) {
      ralloc_free(requests);
      ralloc_free(allocations);
      ralloc_free(ids);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   for (unsigned i = 0; i < g->pass_count; i++) {
      uint32_t id = g->passes[i];
      const struct panvk_graph_value *v = &g->values[id];
      if (v->kind == PANVK_GRAPH_OPERATION) {
         const struct panvk_graph_value *owner = v->storage ? v->storage : v;
         struct pan_graph_alloc_request *request = requests + (owner - g->values);
         if (!request->size)
            *request = (struct pan_graph_alloc_request){i, v->last_use, owner->size};
         request->first = MIN2(request->first, i);
         request->last = MAX2(request->last, v->last_use);
      }
   }
   unsigned count = 0;
   for (unsigned i = 0; i < g->count; i++) {
      if (requests[i].size) {
         ids[count] = i;
         requests[count++] = requests[i];
      }
   }
   struct pan_graph_alloc_stats stats;
   enum pan_graph_alloc_status status =
      pan_graph_allocate(requests, count, 64, 500, allocations, &stats);
   if (status == PAN_GRAPH_ALLOC_SUCCESS) {
      g->transient_size = stats.peak;
      for (unsigned i = 0; i < count; i++)
         g->values[ids[i]].offset = allocations[i].offset;
      for (unsigned i = 0; i < g->count; i++) {
         struct panvk_graph_value *v = &g->values[i];
         if (v->storage)
            v->offset = v->storage->offset + v->view_offset;
      }
      count = 0;
      for (unsigned i = 0; i < g->pass_count; i++) {
         if (!g->values[g->passes[i]].alias_concat)
            g->passes[count++] = g->passes[i];
      }
      g->pass_count = count;
   }
   ralloc_free(requests);
   ralloc_free(allocations);
   ralloc_free(ids);
   return status == PAN_GRAPH_ALLOC_SUCCESS     ? VK_SUCCESS
          : status == PAN_GRAPH_ALLOC_NO_MEMORY ? VK_ERROR_OUT_OF_HOST_MEMORY
                                                : VK_ERROR_UNKNOWN;
}

VkResult
panvk_data_graph_parse(
   void *mem_ctx, const uint32_t *code, size_t size,
   const VkDataGraphPipelineCreateInfoARM *info,
   const VkDataGraphPipelineShaderModuleCreateInfoARM *module,
   struct panvk_data_graph **result)
{
   *result = NULL;
   if (size < 20 || size % 4 || code[0] != SpvMagicNumber || !code[3] ||
       code[3] > size || module->pSpecializationInfo)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct panvk_data_graph *g = rzalloc(mem_ctx, struct panvk_data_graph);
   if (!g)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct graph_parser p = {.bound = code[3],
                            .graph = g,
                            .info = info,
                            .module = module};
   p.ids = rzalloc_array(g, struct graph_id, p.bound);
   g->values = rzalloc_array(g, struct panvk_graph_value, size / 4);
   g->passes = rzalloc_array(g, uint32_t, size / 4);
   if (!p.ids || !g->values || !g->passes) {
      ralloc_free(g);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   for (unsigned i = 0; i < p.bound; i++) {
      p.ids[i].value = UINT32_MAX;
      p.ids[i].set = UINT32_MAX;
      p.ids[i].binding = UINT32_MAX;
   }
   const uint32_t *entry = NULL;
   for (size_t i = 5; i < size / 4;) {
      const uint32_t *w = code + i;
      unsigned n = w[0] >> 16, op = w[0] & 0xffff, index = 0;
      if (!n || n > size / 4 - i)
         goto unsupported;
      switch (op) {
      case SpvOpTypeInt:
      case SpvOpTypeFloat:
      case SpvOpTypeBool:
      case SpvOpTypeArray:
      case SpvOpTypePointer:
      case SpvOpTypeTensorARM:
      case SpvOpTypeGraphARM:
      case SpvOpExtInstImport:
         index = 1;
         break;
      case SpvOpConstant:
      case SpvOpConstantTrue:
      case SpvOpConstantFalse:
      case SpvOpConstantNull:
      case SpvOpConstantComposite:
      case SpvOpConstantCompositeReplicateEXT:
      case SpvOpGraphConstantARM:
      case SpvOpGraphARM:
      case SpvOpGraphInputARM:
      case SpvOpExtInst:
      case SpvOpVariable:
         index = 2;
         break;
      case SpvOpDecorate:
         if (n == 4 && w[1] && w[1] < p.bound) {
            if (w[2] == SpvDecorationDescriptorSet)
               p.ids[w[1]].set = w[3];
            if (w[2] == SpvDecorationBinding)
               p.ids[w[1]].binding = w[3];
         }
         break;
      case SpvOpGraphEntryPointARM:
         if (n < 4 || !memchr(w + 2, 0, (n - 2) * 4))
            goto unsupported;
         if (!strcmp((const char *)(w + 2), module->pName)) {
            if (entry)
               goto unsupported;
            entry = w;
         }
         break;
      default:
         break;
      }
      if (index) {
         if (index >= n || !w[index] || w[index] >= p.bound ||
             p.ids[w[index]].words)
            goto unsupported;
         p.ids[w[index]].words = w;
      }
      i += n;
   }
   if (!entry)
      goto unsupported;
   const uint32_t *graph = instruction(&p, entry[1], SpvOpGraphARM, 3);
   const uint32_t *type =
      graph ? instruction(&p, graph[1], SpvOpTypeGraphARM, 3) : NULL;
   unsigned start = 2 + DIV_ROUND_UP(strlen((const char *)(entry + 2)) + 1, 4);
   if (!type || type[2] >= (type[0] >> 16) - 3 ||
       (entry[0] >> 16) - start != (type[0] >> 16) - 3)
      goto unsupported;
   bool inside = false, complete = false;
   uint32_t output_count = 0;
   for (size_t i = 5; i < size / 4;) {
      const uint32_t *w = code + i;
      unsigned n = w[0] >> 16, op = w[0] & 0xffff;
      if (op == SpvOpGraphARM)
         inside = w == graph;
      else if (op == SpvOpGraphEndARM) {
         if (inside)
            complete = true;
         inside = false;
      } else if (inside && op == SpvOpGraphInputARM) {
         struct panvk_graph_value v = {.kind = PANVK_GRAPH_INPUT};
         if (n < 4 || n > 5 || !shape(&p, &v, w[1]))
            goto unsupported;
         uint32_t idx = scalar(&p, w[3]);
         uint32_t element = n == 5 ? scalar(&p, w[4]) : 0;
         if (p.failed || idx >= type[2] ||
             !resource(&p, &v, entry[start + idx], element))
            goto unsupported;
         v.resource_index = idx;
         p.ids[w[2]].value = g->count;
         g->values[g->count++] = v;
      } else if (inside && op == SpvOpExtInst) {
         if (!operation(&p, w))
            goto unsupported;
      } else if (inside && op == SpvOpGraphSetOutputARM) {
         output_count++;
      } else if (inside && op != SpvOpNop && op != SpvOpLine &&
                 op != SpvOpNoLine) {
         goto unsupported;
      }
      i += n;
   }
   if (!complete)
      goto unsupported;
   inside = false;
   for (size_t i = 5; i < size / 4;) {
      const uint32_t *w = code + i;
      unsigned n = w[0] >> 16, op = w[0] & 0xffff;
      if (op == SpvOpGraphARM)
         inside = w == graph;
      else if (op == SpvOpGraphEndARM)
         inside = false;
      else if (inside && op == SpvOpGraphSetOutputARM) {
         if (n < 3 || n > 4)
            goto unsupported;
         uint32_t src = get_value(&p, w[1]), idx = scalar(&p, w[2]);
         uint32_t element = n == 4 ? scalar(&p, w[3]) : 0;
         if (src == UINT32_MAX || p.failed ||
             idx >= (type[0] >> 16) - 3 - type[2])
            goto unsupported;
         struct panvk_graph_value v = g->values[src];
         v.kind = PANVK_GRAPH_OUTPUT;
         v.input_count = 1;
         v.inputs[0] = src;
         v.data = NULL;
         if (!resource(&p, &v, entry[start + type[2] + idx], element))
            goto unsupported;
         v.resource_index = idx;
         g->passes[g->pass_count++] = g->count;
         g->values[g->count++] = v;
      }
      i += n;
   }
   if (p.failed || !output_count)
      goto unsupported;
   prune_values(g);
   if (!fold_constant_transforms(g)) {
      ralloc_free(g);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   prune_values(g);
   if (!fuse_pointwise(g)) {
      ralloc_free(g);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   VkResult allocation_result = allocate_transient(g);
   if (allocation_result != VK_SUCCESS) {
      ralloc_free(g);
      return allocation_result;
   }
   ralloc_free(p.ids);
   *result = g;
   return VK_SUCCESS;
unsupported:
   ralloc_free(g);
   return p.oom ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_ERROR_FEATURE_NOT_PRESENT;
}
