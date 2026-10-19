#include "panvk_data_graph.h"

#include "nir_builder.h"
#include "vk_format.h"

static nir_def *resize(nir_builder *b, const struct panvk_data_graph *g,
                       const struct panvk_graph_value *v, nir_def **values,
                       nir_def **coords);

static nir_def *
tensor_desc(nir_builder *b, uint32_t index)
{
   nir_def *idx = nir_vulkan_resource_index(
      b, 3, 32, nir_imm_int(b, index), .desc_set = 0, .binding = 0,
      .desc_type = nir_descriptor_type_tensor_arm);
   return nir_load_vulkan_descriptor(
      b, 3, 32, idx, .desc_type = nir_descriptor_type_tensor_arm);
}

static nir_def *
load_tensor(nir_builder *b, const struct panvk_data_graph *g, uint32_t index,
            nir_def **values, nir_def **coords)
{
   const struct panvk_graph_value *v = &g->values[index];
   if (v->fused_into != UINT32_MAX) {
      if (v->op == PANVK_GRAPH_OP_RESIZE)
         return resize(b, g, v, values, coords);
      assert(values[index]);
      return values[index];
   }
   unsigned bits = vk_format_get_blocksizebits(v->format);
   if (v->kind == PANVK_GRAPH_CONSTANT && v->elements == 1) {
      uint32_t value = 0;
      memcpy(&value, v->data, v->size);
      return nir_imm_intN_t(b, value, bits);
   }
   return nir_tensor_read_arm(b, bits, tensor_desc(b, index),
                              nir_vec(b, coords, v->rank),
                              nir_imm_intN_t(b, 0, bits));
}

static nir_def *
load_input(nir_builder *b, const struct panvk_data_graph *g,
           const struct panvk_graph_value *v, unsigned input, nir_def **values,
           nir_def **coords)
{
   uint32_t index = v->inputs[input];
   const struct panvk_graph_value *a = &g->values[index];
   nir_def *c[4];
   for (unsigned d = 0; d < a->rank; d++)
      c[d] =
         a->dims[d] == 1 ? nir_imm_int(b, 0) : coords[d + v->rank - a->rank];
   return load_tensor(b, g, index, values, c);
}

static nir_def *
matmul(nir_builder *b, const struct panvk_data_graph *g,
       const struct panvk_graph_value *v, nir_def **values, nir_def **coords)
{
   const struct panvk_graph_value *a = &g->values[v->inputs[0]];
   bool fp = v->format == VK_FORMAT_R32_SFLOAT;
   nir_variable *acc = nir_local_variable_create(
      b->impl, fp ? glsl_float_type() : glsl_int_type(), "acc");
   nir_variable *k = nir_local_variable_create(b->impl, glsl_uint_type(), "k");
   nir_store_var(b, acc, nir_imm_int(b, 0), 1);
   nir_store_var(b, k, nir_imm_int(b, 0), 1);
   nir_push_loop(b);
   nir_def *ki = nir_load_var(b, k);
   nir_push_if(b, nir_uge_imm(b, ki, a->dims[2]));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_def *ac[] = {coords[0], coords[1], ki};
   nir_def *bc[] = {coords[0], ki, coords[2]};
   nir_def *av = load_tensor(b, g, v->inputs[0], values, ac);
   nir_def *bv = load_tensor(b, g, v->inputs[1], values, bc);
   nir_def *sum = nir_load_var(b, acc);
   if (fp) {
      sum = nir_fadd(b, sum, nir_fmul(b, av, bv));
   } else {
      const int8_t *azp = g->values[v->inputs[2]].data;
      const int8_t *bzp = g->values[v->inputs[3]].data;
      av = nir_iadd_imm(b, nir_i2i32(b, av), -(int32_t)*azp);
      bv = nir_iadd_imm(b, nir_i2i32(b, bv), -(int32_t)*bzp);
      sum = nir_iadd(b, sum, nir_imul(b, av, bv));
   }
   nir_store_var(b, acc, sum, 1);
   nir_store_var(b, k, nir_iadd_imm(b, ki, 1), 1);
   nir_pop_loop(b, NULL);
   return nir_load_var(b, acc);
}

static nir_def *
begin_loop(nir_builder *b, unsigned first, unsigned count,
           nir_variable **counter)
{
   *counter = nir_local_variable_create(b->impl, glsl_uint_type(), "index");
   nir_store_var(b, *counter, nir_imm_int(b, first), 1);
   nir_push_loop(b);
   nir_def *index = nir_load_var(b, *counter);
   nir_push_if(b, nir_uge_imm(b, index, count));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   return index;
}

static void
end_loop(nir_builder *b, nir_variable *counter, nir_def *index)
{
   nir_store_var(b, counter, nir_iadd_imm(b, index, 1), 1);
   nir_pop_loop(b, NULL);
}

static nir_def *
float_minmax(nir_builder *b, nir_def *a, nir_def *c, bool maximum,
             bool propagate_nan)
{
   if (propagate_nan)
      return maximum ? nir_fmax_nan_pan(b, a, c) : nir_fmin_nan_pan(b, a, c);
   return maximum ? nir_fmax(b, a, c) : nir_fmin(b, a, c);
}

static nir_def *
spatial(nir_builder *b, const struct panvk_data_graph *g,
        const struct panvk_graph_value *v, nir_def **values, nir_def **coords)
{
   const struct panvk_graph_value *a = &g->values[v->inputs[0]];
   bool convolution = v->op == PANVK_GRAPH_OP_CONV2D;
   bool integer = v->format == VK_FORMAT_R32_SINT;
   bool average = v->op == PANVK_GRAPH_OP_AVG_POOL2D;
   unsigned channels = convolution ? a->dims[3] : 1;
   unsigned count = v->kernel[0] * v->kernel[1] * channels;
   nir_variable *acc =
      nir_local_variable_create(b->impl, integer ? glsl_int_type() : glsl_float_type(), "acc");
   nir_def *initial = nir_imm_float(b, 0.0f);
   if (!convolution && !average)
      initial = v->attribute ? nir_imm_int(b, 0x7fc00000)
                             : nir_imm_float(b, -INFINITY);
   nir_store_var(b, acc, initial, 1);
   nir_variable *valid_count = NULL;
   if (average) {
      valid_count =
         nir_local_variable_create(b->impl, glsl_uint_type(), "count");
      nir_store_var(b, valid_count, nir_imm_int(b, 0), 1);
   }
   nir_variable *counter;
   nir_def *index = begin_loop(b, 0, count, &counter);
   nir_def *kernel = nir_udiv_imm(b, index, channels);
   nir_def *ky = nir_udiv_imm(b, kernel, v->kernel[1]);
   nir_def *kx = nir_umod_imm(b, kernel, v->kernel[1]);
   nir_def *y =
      nir_iadd_imm(b,
                   nir_iadd(b, nir_imul_imm(b, coords[1], v->stride[0]),
                            nir_imul_imm(b, ky, v->dilation[0])),
                   -(int64_t)v->padding[0]);
   nir_def *x =
      nir_iadd_imm(b,
                   nir_iadd(b, nir_imul_imm(b, coords[2], v->stride[1]),
                            nir_imul_imm(b, kx, v->dilation[1])),
                   -(int64_t)v->padding[2]);
   nir_push_if(b, nir_iand(b, nir_ult_imm(b, y, a->dims[1]),
                           nir_ult_imm(b, x, a->dims[2])));
   nir_def *c[] = {coords[0], y, x,
                   convolution ? nir_umod_imm(b, index, channels) : coords[3]};
   nir_def *value = load_tensor(b, g, v->inputs[0], values, c);
   nir_def *sum = nir_load_var(b, acc);
   if (convolution) {
      nir_def *wc[] = {coords[3], ky, kx, c[3]};
      nir_def *weight = load_tensor(b, g, v->inputs[1], values, wc);
      if (integer) {
         int32_t azp = *(int8_t *)g->values[v->inputs[3]].data;
         int32_t wzp = *(int8_t *)g->values[v->inputs[4]].data;
         value = nir_iadd_imm(b, nir_i2i32(b, value), -azp);
         weight = nir_iadd_imm(b, nir_i2i32(b, weight), -wzp);
         sum = nir_iadd(b, sum, nir_imul(b, value, weight));
      } else {
         sum = nir_fadd(b, sum, nir_fmul(b, value, weight));
      }
   } else if (average) {
      sum = nir_fadd(b, sum, value);
      nir_store_var(b, valid_count,
                    nir_iadd_imm(b, nir_load_var(b, valid_count), 1), 1);
   } else {
      sum = float_minmax(b, sum, value, true, v->attribute == 0);
   }
   nir_store_var(b, acc, sum, 1);
   nir_pop_if(b, NULL);
   end_loop(b, counter, index);
   nir_def *result = nir_load_var(b, acc);
   if (convolution) {
      const struct panvk_graph_value *bias = &g->values[v->inputs[2]];
      nir_def *bc[] = {bias->dims[0] == 1 ? nir_imm_int(b, 0) : coords[3]};
      nir_def *bias_value = load_tensor(b, g, v->inputs[2], values, bc);
      result = integer ? nir_iadd(b, result, bias_value)
                       : nir_fadd(b, result, bias_value);
   } else if (average) {
      result = nir_fdiv(b, result, nir_u2f32(b, nir_load_var(b, valid_count)));
   }
   return result;
}

static nir_def *
reduce(nir_builder *b, const struct panvk_data_graph *g,
       const struct panvk_graph_value *v, nir_def **values, nir_def **coords)
{
   const struct panvk_graph_value *a = &g->values[v->inputs[0]];
   bool fp = v->format == VK_FORMAT_R32_SFLOAT;
   bool extrema =
      v->op == PANVK_GRAPH_OP_REDUCE_MAX || v->op == PANVK_GRAPH_OP_REDUCE_MIN;
   nir_variable *acc = nir_local_variable_create(
      b->impl, fp ? glsl_float_type() : glsl_int_type(), "acc");
   nir_def *c[4];
   memcpy(c, coords, a->rank * sizeof(*c));
   c[v->axis] = nir_imm_int(b, 0);
   nir_def *initial =
      extrema ? load_tensor(b, g, v->inputs[0], values, c)
      : fp    ? nir_imm_float(b,
                           v->op == PANVK_GRAPH_OP_REDUCE_PRODUCT ? 1.0f : 0.0f)
              : nir_imm_int(b, v->op == PANVK_GRAPH_OP_REDUCE_PRODUCT ? 1 : 0);
   nir_store_var(b, acc, initial, 1);
   nir_variable *counter;
   nir_def *index = begin_loop(b, extrema, a->dims[v->axis], &counter);
   c[v->axis] = index;
   nir_def *value = load_tensor(b, g, v->inputs[0], values, c);
   nir_def *sum = nir_load_var(b, acc);
   if (extrema) {
      bool maximum = v->op == PANVK_GRAPH_OP_REDUCE_MAX;
      if (fp)
         sum = float_minmax(b, sum, value, maximum, v->attribute == 0);
      else
         sum = maximum ? nir_imax(b, sum, value) : nir_imin(b, sum, value);
   } else if (v->op == PANVK_GRAPH_OP_REDUCE_PRODUCT) {
      sum = fp ? nir_fmul(b, sum, value) : nir_imul(b, sum, value);
   } else {
      sum = fp ? nir_fadd(b, sum, value) : nir_iadd(b, sum, value);
   }
   nir_store_var(b, acc, sum, 1);
   end_loop(b, counter, index);
   return nir_load_var(b, acc);
}

static nir_def *
activation(nir_builder *b, nir_def *input, bool tanh)
{
   nir_def *x = nir_fabs(b, input);
   nir_def *z = nir_fexp2(
      b, nir_fmul_imm(b, x, tanh ? -2.8853900817779268 : -1.4426950408889634));
   nir_def *denom = nir_fadd_imm(b, z, 1.0f);
   if (!tanh) {
      return nir_fdiv(
         b,
         nir_bcsel(b, nir_flt_imm(b, input, 0.0f), z, nir_imm_float(b, 1.0f)),
         denom);
   }
   nir_def *large = nir_fdiv(b, nir_fsub_imm(b, 1.0f, z), denom);
   const double coefficients[] = {
      -1.0 / 3.0,
      2.0 / 15.0,
      -17.0 / 315.0,
      62.0 / 2835.0,
      -1382.0 / 155925.0,
      21844.0 / 6081075.0,
      -929569.0 / 638512875.0,
   };
   nir_def *square = nir_fmul(b, x, x);
   nir_def *p = nir_imm_float(b, coefficients[ARRAY_SIZE(coefficients) - 1]);
   for (int i = ARRAY_SIZE(coefficients) - 2; i >= 0; i--)
      p = nir_fadd_imm(b, nir_fmul(b, square, p), coefficients[i]);
   nir_def *small = nir_fadd(b, x, nir_fmul(b, nir_fmul(b, x, square), p));
   nir_def *magnitude = nir_bcsel(b, nir_flt_imm(b, x, 0.5f), small, large);
   return nir_ior(b, nir_iand_imm(b, input, 0x80000000),
                  nir_iand_imm(b, magnitude, 0x7fffffff));
}

static nir_def *
im2col(nir_builder *b, const struct panvk_data_graph *g,
       const struct panvk_graph_value *v, nir_def **values, nir_def **coords)
{
   const struct panvk_graph_value *a = &g->values[v->inputs[0]];
   nir_def *kernel = nir_udiv_imm(b, coords[2], a->dims[3]);
   nir_def *oy = nir_udiv_imm(b, coords[1], v->attribute);
   nir_def *ox = nir_umod_imm(b, coords[1], v->attribute);
   nir_def *y = nir_iadd_imm(
      b, nir_iadd(b, nir_imul_imm(b, oy, v->stride[0]),
                  nir_imul_imm(b, nir_udiv_imm(b, kernel, v->kernel[1]), v->dilation[0])),
      -(int64_t)v->padding[0]);
   nir_def *x = nir_iadd_imm(
      b, nir_iadd(b, nir_imul_imm(b, ox, v->stride[1]),
                  nir_imul_imm(b, nir_umod_imm(b, kernel, v->kernel[1]), v->dilation[1])),
      -(int64_t)v->padding[2]);
   nir_variable *result = nir_local_variable_create(b->impl, glsl_int8_t_type(), "im2col");
   nir_store_var(b, result, nir_imm_intN_t(b, *(uint8_t *)g->values[v->inputs[1]].data, 8), 1);
   nir_push_if(b, nir_iand(b, nir_ult_imm(b, y, a->dims[1]),
                           nir_ult_imm(b, x, a->dims[2])));
   nir_def *c[] = {coords[0], y, x, nir_umod_imm(b, coords[2], a->dims[3])};
   nir_store_var(b, result, load_tensor(b, g, v->inputs[0], values, c), 1);
   nir_pop_if(b, NULL);
   return nir_load_var(b, result);
}

static nir_def *
rescale(nir_builder *b, const struct panvk_data_graph *g,
        const struct panvk_graph_value *v, nir_def **values, nir_def **coords)
{
   nir_def *channel[] = {v->attribute & 2 ? coords[v->rank - 1] : nir_imm_int(b, 0)};
   nir_def *zero[] = {nir_imm_int(b, 0)};
   nir_def *x = load_input(b, g, v, 0, values, coords);
   nir_def *izp = load_tensor(b, g, v->inputs[3], values, zero);
   nir_def *ozp = load_tensor(b, g, v->inputs[4], values, zero);
   x = v->attribute & 4 ? nir_u2u32(b, x) : nir_i2i32(b, x);
   izp = v->attribute & 4 ? nir_u2u32(b, izp) : nir_i2i32(b, izp);
   ozp = v->attribute & 8 ? nir_u2u32(b, ozp) : nir_i2i32(b, ozp);
   x = nir_isub(b, x, izp);
   nir_def *mul = load_tensor(b, g, v->inputs[1], values, channel);
   nir_def *shift = nir_u2u32(b, load_tensor(b, g, v->inputs[2], values, channel));
   nir_def *round = nir_ishl(b, nir_imm_int64(b, 1), nir_iadd_imm(b, shift, -1));
   if (v->attribute & 1) {
      nir_def *extra = nir_bcsel(b, nir_ige_imm(b, x, 0),
                                  nir_imm_int64(b, 1ll << 30),
                                  nir_imm_int64(b, -(1ll << 30)));
      round = nir_iadd(b, round,
                       nir_bcsel(b, nir_ugt_imm(b, shift, 31), extra,
                                    nir_imm_int64(b, 0)));
   }
   nir_def *product = nir_imul(b, nir_i2i64(b, x), nir_i2i64(b, mul));
   nir_def *scaled = nir_ishr(b, nir_iadd(b, product, round), shift);
   scaled = nir_iadd(b, scaled, nir_i2i64(b, ozp));
   unsigned bits = vk_format_get_blocksizebits(v->format);
   int64_t lo = v->attribute & 8 ? 0 : -(1ll << (bits - 1));
   int64_t hi = v->attribute & 8 ? (1ll << bits) - 1 : (1ll << (bits - 1)) - 1;
   scaled = nir_imax(b, nir_imm_int64(b, lo),
                     nir_imin(b, scaled, nir_imm_int64(b, hi)));
   return nir_i2iN(b, scaled, bits);
}

static nir_def *
resize(nir_builder *b, const struct panvk_data_graph *g,
       const struct panvk_graph_value *v, nir_def **values, nir_def **coords)
{
   const struct panvk_graph_value *a = &g->values[v->inputs[0]];
   nir_def *low[2], *high[2], *fraction[2];
   bool fp = a->format == VK_FORMAT_R32_SFLOAT;
   for (unsigned d = 0; d < 2; d++) {
      int32_t num = v->resize_scale[2 * d], den = v->resize_scale[2 * d + 1];
      int64_t bias = (int64_t)v->resize_offset[d] + num / 2;
      if (v->attribute == 1 && num > 0 && den > 0 && num % den == 0 &&
          bias >= 0 && bias < den &&
          (uint64_t)a->dims[d + 1] * (num / den) == v->dims[d + 1]) {
         low[d] = nir_udiv_imm(b, coords[d + 1], num / den);
         continue;
      }
      nir_def *pos = nir_iadd_imm(b, nir_imul_imm(b, coords[d + 1], den), v->resize_offset[d]);
      nir_def *q = nir_idiv(b, pos, nir_imm_int(b, num));
      nir_def *rem = nir_isub(b, pos, nir_imul_imm(b, q, num));
      q = nir_iadd(b, q, nir_bcsel(b, nir_ilt_imm(b, rem, 0), nir_imm_int(b, -1), nir_imm_int(b, 0)));
      fraction[d] = nir_isub(b, pos, nir_imul_imm(b, q, num));
      low[d] = nir_imax(b, q, nir_imm_int(b, 0));
      high[d] = nir_imin(b, nir_iadd_imm(b, q, 1), nir_imm_int(b, a->dims[d + 1] - 1));
      if (v->attribute == 1)
         low[d] = nir_bcsel(b, nir_ige_imm(b, nir_imul_imm(b, fraction[d], 2), num), high[d], low[d]);
      else if (fp)
         fraction[d] = nir_fmul_imm(b, nir_i2f32(b, fraction[d]), 1.0 / num);
   }
   nir_def *c[] = {coords[0], low[0], low[1], coords[3]};
   if (v->attribute == 1)
      return load_tensor(b, g, v->inputs[0], values, c);
   nir_def *sum = nir_imm_int(b, 0);
   for (unsigned y = 0; y < 2; y++) {
      for (unsigned x = 0; x < 2; x++) {
         c[1] = y ? high[0] : low[0];
         c[2] = x ? high[1] : low[1];
         nir_def *value = load_tensor(b, g, v->inputs[0], values, c);
         nir_def *wy = fraction[0], *wx = fraction[1];
         if (fp) {
            if (!y) wy = nir_fsub_imm(b, 1.0f, wy);
            if (!x) wx = nir_fsub_imm(b, 1.0f, wx);
            sum = nir_fadd(b, sum, nir_fmul(b, nir_fmul(b, value, wy), wx));
         } else {
            if (!y) wy = nir_isub(b, nir_imm_int(b, v->resize_scale[0]), wy);
            if (!x) wx = nir_isub(b, nir_imm_int(b, v->resize_scale[2]), wx);
            sum = nir_iadd(b, sum, nir_imul(b, nir_imul(b, nir_i2i32(b, value), wy), wx));
         }
      }
   }
   return sum;
}

static nir_def *
emit_value(nir_builder *b, const struct panvk_data_graph *g, uint32_t index,
           nir_def **values, nir_def **coords)
{
   const struct panvk_graph_value *v = &g->values[index];
   nir_def *r;
   if (v->kind == PANVK_GRAPH_OUTPUT) {
      r = load_input(b, g, v, 0, values, coords);
   } else if (v->op == PANVK_GRAPH_OP_MATMUL) {
      r = matmul(b, g, v, values, coords);
   } else if (v->op == PANVK_GRAPH_OP_IM2COL) {
      r = im2col(b, g, v, values, coords);
   } else if (v->op == PANVK_GRAPH_OP_RESCALE) {
      r = rescale(b, g, v, values, coords);
   } else if (v->op == PANVK_GRAPH_OP_RESIZE) {
      r = resize(b, g, v, values, coords);
   } else if (v->op == PANVK_GRAPH_OP_TABLE) {
      nir_def *input = load_input(b, g, v, 0, values, coords);
      nir_def *c[] = {nir_iadd_imm(b, nir_i2i32(b, input), 128)};
      r = load_tensor(b, g, v->inputs[1], values, c);
   } else if (v->op == PANVK_GRAPH_OP_CONCAT) {
      nir_variable *result = nir_local_variable_create(
         b->impl, v->format == VK_FORMAT_R8_SINT ? glsl_int8_t_type() : glsl_uint_type(), "concat");
      nir_def *c[4];
      memcpy(c, coords, sizeof(c));
      uint32_t offset = 0;
      for (unsigned i = 0; i < v->input_count; i++) {
         uint32_t extent = g->values[v->inputs[i]].dims[v->axis];
         c[v->axis] = nir_iadd_imm(b, coords[v->axis], -(int64_t)offset);
         nir_push_if(b, nir_ult_imm(b, c[v->axis], extent));
         nir_store_var(b, result, load_tensor(b, g, v->inputs[i], values, c), 1);
         nir_pop_if(b, NULL);
         offset += extent;
      }
      r = nir_load_var(b, result);
   } else if (v->op == PANVK_GRAPH_OP_AVG_POOL2D ||
              v->op == PANVK_GRAPH_OP_CONV2D ||
              v->op == PANVK_GRAPH_OP_MAX_POOL2D) {
      r = spatial(b, g, v, values, coords);
   } else if (v->op >= PANVK_GRAPH_OP_REDUCE_MAX &&
              v->op <= PANVK_GRAPH_OP_REDUCE_SUM) {
      r = reduce(b, g, v, values, coords);
   } else if (v->op == PANVK_GRAPH_OP_SIGMOID || v->op == PANVK_GRAPH_OP_TANH) {
      r = activation(b, load_input(b, g, v, 0, values, coords),
                     v->op == PANVK_GRAPH_OP_TANH);
   } else if (v->op == PANVK_GRAPH_OP_EXP) {
      r = nir_fexp2(b, nir_fmul_imm(b, load_input(b, g, v, 0, values, coords),
                                    1.4426950408889634));
   } else if (v->op == PANVK_GRAPH_OP_RECIPROCAL) {
      r = nir_frcp(b, load_input(b, g, v, 0, values, coords));
   } else if (v->op == PANVK_GRAPH_OP_TRANSPOSE) {
      nir_def *c[4];
      for (unsigned d = 0; d < v->rank; d++)
         c[v->permutation[d]] = coords[d];
      r = load_tensor(b, g, v->inputs[0], values, c);
   } else if (v->op == PANVK_GRAPH_OP_RESHAPE) {
      const struct panvk_graph_value *a = &g->values[v->inputs[0]];
      nir_def *index = nir_imm_int(b, 0), *c[4];
      for (unsigned d = 0; d < v->rank; d++)
         index = nir_iadd(b, nir_imul_imm(b, index, v->dims[d]), coords[d]);
      for (int d = a->rank - 1; d >= 0; d--) {
         c[d] = nir_umod_imm(b, index, a->dims[d]);
         index = nir_udiv_imm(b, index, a->dims[d]);
      }
      r = load_tensor(b, g, v->inputs[0], values, c);
   } else {
      nir_def *a = load_input(b, g, v, 0, values, coords);
      nir_def *c = load_input(b, g, v, 1, values, coords);
      bool fp = v->format == VK_FORMAT_R32_SFLOAT;
      switch (v->op) {
      case PANVK_GRAPH_OP_ADD:
         r = fp ? nir_fadd(b, a, c) : nir_iadd(b, a, c);
         break;
      case PANVK_GRAPH_OP_SUB:
         r = fp ? nir_fsub(b, a, c) : nir_isub(b, a, c);
         break;
      case PANVK_GRAPH_OP_MUL:
         r = nir_fmul(b, a, c);
         break;
      case PANVK_GRAPH_OP_CLAMP: {
         nir_def *hi = load_input(b, g, v, 2, values, coords);
         if (fp) {
            bool propagate_nan = v->attribute == 0;
            r = float_minmax(b, a, hi, false, propagate_nan);
            r = float_minmax(b, r, c, true, propagate_nan);
            if (!propagate_nan)
               r = nir_bcsel(b, nir_fneu(b, a, a), c, r);
         } else {
            r = nir_imax(b, nir_imin(b, a, hi), c);
         }
         break;
      }
      default:
         UNREACHABLE("unsupported data graph operation");
      }
   }
   return r;
}

nir_shader *
panvk_data_graph_shader(const struct panvk_data_graph *g, uint32_t pass)
{
   uint32_t index = g->passes[pass];
   const struct panvk_graph_value *v = &g->values[index];
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL,
                                                  "panvk-data-graph-%u", pass);
   b.shader->info.internal = false;
   b.fp_math_ctrl = nir_fp_preserve_sz_inf_nan;
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def **values = rzalloc_array(b.shader, nir_def *, g->count);
   if (!values) {
      ralloc_free(b.shader);
      return NULL;
   }
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *idx =
      nir_iadd(&b, nir_channel(&b, gid, 0),
               nir_imul_imm(&b, nir_channel(&b, gid, 1), 65535 * 64));
   nir_push_if(&b, nir_ult_imm(&b, idx, v->elements));
   nir_def *coords[4], *remaining = idx;
   for (int d = v->rank - 1; d >= 0; d--) {
      coords[d] = nir_umod_imm(&b, remaining, v->dims[d]);
      remaining = nir_udiv_imm(&b, remaining, v->dims[d]);
   }
   for (unsigned i = 0; i < index; i++) {
      const struct panvk_graph_value *source = &g->values[i];
      if (source->fused_into == index && source->op != PANVK_GRAPH_OP_RESIZE) {
         nir_def *c[4];
         for (unsigned d = 0; d < source->rank; d++)
            c[d] = source->dims[d] == 1
                      ? nir_imm_int(&b, 0)
                      : coords[d + v->rank - source->rank];
         values[i] = emit_value(&b, g, i, values, c);
      }
   }
   nir_def *r = emit_value(&b, g, index, values, coords);
   nir_tensor_write_arm(&b, tensor_desc(&b, index),
                        nir_vec(&b, coords, v->rank), r);
   nir_pop_if(&b, NULL);
   nir_validate_shader(b.shader, "data graph lowering");
   return b.shader;
}
