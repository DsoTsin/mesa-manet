#include "pan_nir.h"

struct preamble_ctx {
   const struct pan_compile_inputs *inputs;
   uint8_t *allowed;
   unsigned result_base;
   bool fau_pressure;
   bool buffer_addresses;
};

static int
pilot_handle_table(nir_def *handle)
{
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(handle, 0));
   uint64_t table = UINT64_MAX;

   if (nir_scalar_is_const(s)) {
      table = nir_scalar_as_uint(s) >> 24;
   } else if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_ior) {
      for (unsigned i = 0; i < 2; i++) {
         nir_scalar src = nir_scalar_chase_alu_src(s, i);
         if (nir_scalar_is_const(src) &&
             !(nir_scalar_as_uint(src) & BITFIELD_MASK(24)))
            table = nir_scalar_as_uint(src) >> 24;
      }
   }

   return table < 64 ? table : -1;
}

static bool
pilot_desc_allowed(nir_intrinsic_instr *intr)
{
   return pilot_handle_table(intr->src[0].ssa) >= 0;
}

static bool
pilot_push_allowed(nir_intrinsic_instr *intr, const struct preamble_ctx *ctx)
{
   if (!nir_src_is_const(intr->src[0]))
      return false;

   unsigned offset = nir_intrinsic_base(intr) + nir_src_as_uint(intr->src[0]);
   unsigned bytes =
      DIV_ROUND_UP(intr->def.num_components * intr->def.bit_size, 8);
   unsigned first = offset / 4;
   unsigned last = (offset + bytes - 1) / 4;

   if (first >= PAN_MAX_PUSH)
      return true;

   return !BITSET_TEST_RANGE(ctx->inputs->fau.pilot_volatile, first,
                             MIN2(last, PAN_MAX_PUSH - 1));
}

static bool allowed_def(nir_def *def, struct preamble_ctx *ctx);

static bool
allowed_src(nir_src *src, void *data)
{
   return allowed_def(src->ssa, data);
}

static bool
allowed_def(nir_def *def, struct preamble_ctx *ctx)
{
   if (ctx->allowed[def->index])
      return ctx->allowed[def->index] == 2;

   ctx->allowed[def->index] = 1;
   nir_instr *instr = nir_def_instr(def);
   bool allowed = false;
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_alu:
   case nir_instr_type_phi:
      allowed = true;
      break;
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_push_constant:
         allowed = pilot_push_allowed(intr, ctx);
         break;
      case nir_intrinsic_load_ubo:
         allowed = pilot_desc_allowed(intr);
         break;
      case nir_intrinsic_load_constant:
      case nir_intrinsic_load_global_constant:
         allowed = true;
         break;
      case nir_intrinsic_load_ssbo:
         allowed = (nir_intrinsic_access(intr) & ACCESS_CAN_REORDER) &&
                   !(nir_intrinsic_access(intr) & ACCESS_VOLATILE) &&
                   pilot_desc_allowed(intr);
         break;
      case nir_intrinsic_load_ssbo_address:
         allowed = ctx->buffer_addresses && pilot_desc_allowed(intr);
         break;
      default:
         break;
      }
      break;
   }
   default:
      break;
   }

   allowed &= nir_foreach_src(instr, allowed_src, ctx);
   ctx->allowed[def->index] = allowed ? 2 : 3;
   return allowed;
}

static bool
avoid_instr(const nir_instr *instr, const void *data)
{
   struct preamble_ctx *ctx = (void *)data;
   nir_def *def = nir_instr_def((nir_instr *)instr);
   return !def || def->bit_size < 8 ||
          def->bit_size * def->num_components > 128 || !allowed_def(def, ctx);
}

static void
def_size(nir_def *def, unsigned *size, unsigned *align,
         nir_preamble_class *class_)
{
   *size = DIV_ROUND_UP(def->num_components * def->bit_size, 32);
   *align = def->bit_size == 64 ? 2 : 1;
   *class_ = nir_preamble_class_general;
}

static unsigned
block_loop_depth(nir_block *block)
{
   unsigned depth = 0;
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent)
      depth += node->type == nir_cf_node_loop;
   return depth;
}

static unsigned
use_loop_depth(nir_def *def)
{
   unsigned depth = 0;
   nir_foreach_use_including_if(use, def) {
      nir_block *block;
      if (nir_src_is_if(use))
         block = nir_cf_node_as_block(
            nir_cf_node_prev(&nir_src_use_if(use)->cf_node));
      else
         block = nir_src_use_instr(use)->block;
      depth = MAX2(depth, block_loop_depth(block));
   }
   return depth;
}

static unsigned
alu_lanes(const nir_def *def)
{
   if (def->bit_size == 8)
      return DIV_ROUND_UP(def->num_components, 4);
   if (def->bit_size == 16)
      return DIV_ROUND_UP(def->num_components, 2);
   return def->num_components;
}

static bool
pushable_ubo_load(const nir_intrinsic_instr *intr,
                  const struct preamble_ctx *ctx)
{
   return intr->intrinsic == nir_intrinsic_load_ubo &&
          ctx->inputs->fau.pushable_ubos && nir_src_is_const(intr->src[0]) &&
          nir_src_is_const(intr->src[1]);
}

static float
base_instr_cost(nir_instr *instr, const struct preamble_ctx *ctx)
{
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
      case nir_op_mov:
      case nir_op_vec2:
      case nir_op_vec3:
      case nir_op_vec4:
      case nir_op_fneg:
      case nir_op_fabs:
         return 0;
      case nir_op_frcp:
      case nir_op_frsq:
      case nir_op_fsqrt:
      case nir_op_fexp2:
      case nir_op_flog2:
      case nir_op_fsin:
      case nir_op_fcos:
         return 5 * alu_lanes(&alu->def);
      case nir_op_fmul:
      case nir_op_imul:
      case nir_op_ffma:
         return 2 * alu_lanes(&alu->def);
      default:
         return alu_lanes(&alu->def);
      }
   }

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_ubo:
         if (pushable_ubo_load(intr, ctx))
            return ctx->fau_pressure ? 4 : 0;
         return 5;
      case nir_intrinsic_load_ssbo:
      case nir_intrinsic_load_ssbo_address:
      case nir_intrinsic_load_global_constant:
      case nir_intrinsic_load_constant:
         return 5;
      default:
         return 0;
      }
   }

   return instr->type == nir_instr_type_phi ? 1 : 0;
}

static float
instr_cost(nir_instr *instr, const void *data)
{
   const struct preamble_ctx *ctx = data;
   float cost = base_instr_cost(instr, ctx);
   nir_def *def = nir_instr_def(instr);
   if (cost == 0 || !def)
      return cost;
   return cost * (use_loop_depth(def) + 1);
}

static float
rewrite_cost(nir_def *def, const void *data)
{
   const struct preamble_ctx *ctx = data;
   unsigned words = DIV_ROUND_UP(def->num_components * def->bit_size, 32);
   return ctx->fau_pressure ? 4.0f * words : words / 16.0f;
}

static int
compare_u64(const void *a, const void *b)
{
   uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
   return (x > y) - (x < y);
}

static unsigned
count_pushable_ubo_words(nir_function_impl *impl,
                         const struct preamble_ctx *ctx)
{
   struct util_dynarray words;
   util_dynarray_init(&words, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (!pushable_ubo_load(intr, ctx))
            continue;
         uint64_t handle = nir_src_as_uint(intr->src[0]);
         unsigned offset = nir_src_as_uint(intr->src[1]);
         unsigned bytes = intr->def.num_components * (intr->def.bit_size / 8);
         for (unsigned b = offset / 4; b < DIV_ROUND_UP(offset + bytes, 4); b++)
            util_dynarray_append_typed(&words, uint64_t, (handle << 32) | b);
      }
   }
   unsigned count = util_dynarray_num_elements(&words, uint64_t);
   uint64_t *data = util_dynarray_begin(&words);
   if (count > 1)
      qsort(data, count, sizeof(uint64_t), compare_u64);
   unsigned unique = 0;
   for (unsigned i = 0; i < count; i++)
      unique += i == 0 || data[i] != data[i - 1];
   util_dynarray_fini(&words);
   return unique;
}

static void
mark_speculatable_uniform_loads(nir_function_impl *impl,
                                const struct preamble_ctx *ctx)
{
   bool robust_ubo = ctx->inputs->robust_modes & nir_var_mem_ubo;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         bool desc_load =
            intr->intrinsic == nir_intrinsic_load_ssbo &&
            nir_src_is_const(intr->src[0]) && nir_src_is_const(intr->src[1]) &&
            pan_res_handle_get_table(nir_src_as_uint(intr->src[0])) == 62;
         bool buffer_address =
            ctx->buffer_addresses &&
            intr->intrinsic == nir_intrinsic_load_ssbo_address &&
            nir_src_is_const(intr->src[0]) && pilot_desc_allowed(intr);
         bool uniform_load =
            (intr->intrinsic == nir_intrinsic_load_push_constant &&
             nir_src_is_const(intr->src[0])) ||
            (!robust_ubo && pushable_ubo_load(intr, ctx)) || desc_load ||
            buffer_address;
         if (uniform_load && nir_intrinsic_has_access(intr))
            nir_intrinsic_set_access(intr, nir_intrinsic_access(intr) |
                                              ACCESS_CAN_REORDER |
                                              ACCESS_CAN_SPECULATE);
      }
   }
}

struct pilot_values {
   struct pan_fau_virtual *virt;
   int16_t *value_of_base;
};

static bool
rebase_main_load(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct pilot_values *pv = data;
   if (intr->intrinsic != nir_intrinsic_load_preamble)
      return false;

   int index = pv->value_of_base[nir_intrinsic_base(intr)];
   assert(index >= 0);
   nir_intrinsic_set_base(intr, pv->virt->values[index].word);
   return true;
}

static unsigned
pilot_value_align(const nir_def *def, unsigned size)
{
   if (size >= 4)
      return 4;
   if (size >= 2 || def->bit_size == 64)
      return 2;
   return 1;
}

static void
register_pilot_values(nir_shader *pilot, struct pilot_values *pv)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(pilot);
   struct hash_table *seen = _mesa_pointer_hash_table_create(NULL);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_preamble)
            continue;

         nir_def *value = intr->src[0].ssa;
         unsigned base = nir_intrinsic_base(intr);
         unsigned size = DIV_ROUND_UP(value->num_components * value->bit_size, 32);
         unsigned index = pan_fau_virtual_add(pv->virt, size,
                                              pilot_value_align(value, size),
                                              PAN_FAU_VALUE_PILOT,
                                              (struct pan_ubo_relocation){0});
         pv->value_of_base[base] = index;

         struct hash_entry *entry = _mesa_hash_table_search(seen, value);
         if (!entry) {
            _mesa_hash_table_insert(seen, value, intr);
            nir_intrinsic_set_base(intr, index);
            continue;
         }

         nir_intrinsic_instr *kept = entry->data;
         pv->virt->values[index].alias = nir_intrinsic_base(kept);
         if (kept->instr.block != block) {
            nir_instr *def_instr = nir_def_instr(value);
            nir_instr_move(def_instr->type == nir_instr_type_phi
                              ? nir_after_phis(def_instr->block)
                              : nir_after_instr(def_instr),
                           &kept->instr);
         }
         nir_instr_remove(instr);
      }
   }

   _mesa_hash_table_destroy(seen, NULL);
   nir_progress(true, impl, nir_metadata_control_flow);
}

#define PILOT_SINK_ADDRESS_HI 0x80002000

struct pilot_buffer {
   nir_def *lo;
   nir_def *hi;
   nir_def *size;
};

static struct pilot_buffer
pilot_load_buffer(nir_builder *b, nir_def *handle, unsigned table,
                  bool need_size, bool check_rack)
{
   nir_def *srt = nir_load_pilot_arg_pan(b, .base = 1);
   nir_def *index = nir_iand_imm(b, handle, BITFIELD_MASK(24));

   if (table == 62) {
      nir_def *entry =
         nir_iadd(b, srt, nir_u2u64(b, nir_ishl_imm(b, index, 4)));
      nir_def *res = nir_load_global_constant(b, need_size ? 3 : 2, 32, entry,
                                              .align_mul = 16);
      return (struct pilot_buffer){
         .lo = nir_channel(b, res, 0),
         .hi = nir_iand_imm(b, nir_channel(b, res, 1), BITFIELD_MASK(24)),
         .size = need_size ? nir_channel(b, res, 2) : NULL,
      };
   }

   nir_def *rack = nir_load_global_constant(
      b, check_rack ? 3 : 2, 32, nir_iadd_imm(b, srt, table * 16),
      .align_mul = 16);
   nir_def *rack_hi =
      nir_iand_imm(b, nir_channel(b, rack, 1), BITFIELD_MASK(24));
   if (check_rack) {
      nir_def *end = nir_iadd_imm(b, nir_ishl_imm(b, index, 5), 32);
      rack_hi = nir_bcsel(b, nir_uge(b, nir_channel(b, rack, 2), end), rack_hi,
                          nir_imm_int(b, PILOT_SINK_ADDRESS_HI));
   }
   nir_def *rack_addr =
      nir_pack_64_2x32_split(b, nir_channel(b, rack, 0), rack_hi);
   nir_def *desc =
      nir_iadd(b, rack_addr, nir_u2u64(b, nir_ishl_imm(b, index, 5)));
   if (!need_size) {
      nir_def *addr = nir_load_global_constant(b, 2, 32, nir_iadd_imm(b, desc, 8),
                                               .align_mul = 32,
                                               .align_offset = 8);
      return (struct pilot_buffer){
         .lo = nir_channel(b, addr, 0),
         .hi = nir_channel(b, addr, 1),
      };
   }
   nir_def *words = nir_load_global_constant(b, 4, 32, desc, .align_mul = 32);
   return (struct pilot_buffer){
      .lo = nir_channel(b, words, 2),
      .hi = nir_channel(b, words, 3),
      .size = nir_channel(b, words, 1),
   };
}

static nir_def *
pilot_buffer_address(nir_builder *b, struct pilot_buffer buf,
                     nir_def *in_bounds)
{
   STATIC_ASSERT(((uint64_t)PILOT_SINK_ADDRESS_HI << 32) &
                 PAN_SHADER_OOB_ADDRESS);
   nir_def *hi =
      nir_bcsel(b, in_bounds, buf.hi, nir_imm_int(b, PILOT_SINK_ADDRESS_HI));
   return nir_pack_64_2x32_split(b, buf.lo, hi);
}

static nir_def *
pilot_load_bounded(nir_builder *b, struct pilot_buffer buf, nir_def *offset,
                   unsigned num_components, unsigned bit_size)
{
   unsigned bytes = num_components * bit_size / 8;
   nir_scalar off = nir_scalar_chase_movs(nir_get_scalar(offset, 0));
   nir_def *in_bounds;

   if (nir_scalar_is_const(off)) {
      uint64_t end = nir_scalar_as_uint(off) + bytes;
      in_bounds = end > UINT32_MAX
                     ? nir_imm_false(b)
                     : nir_uge(b, buf.size, nir_imm_int(b, end));
   } else {
      in_bounds =
         nir_iand(b, nir_uge(b, buf.size, nir_imm_int(b, bytes)),
                  nir_uge(b, nir_isub(b, buf.size, nir_imm_int(b, bytes)),
                          offset));
   }

   nir_def *addr = nir_iadd(b, pilot_buffer_address(b, buf, in_bounds),
                            nir_u2u64(b, offset));
   return nir_load_global_constant(b, num_components, bit_size, addr,
                                   .align_mul = bit_size / 8);
}

static bool
lower_preamble(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct preamble_ctx *ctx = data;
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_push_constant: {
      unsigned offset = nir_src_as_uint(intr->src[0]);
      assert(nir_intrinsic_base(intr) == 0);
      assert(offset + DIV_ROUND_UP(
                         intr->def.bit_size * intr->def.num_components, 8) <=
             ctx->inputs->fau.reserved * 4);
      nir_def *value = nir_load_global_constant(
         b, intr->def.num_components, intr->def.bit_size,
         nir_iadd_imm(b, nir_load_pilot_arg_pan(b, .base = 0), offset),
         .align_mul = 8,
         .align_offset = offset % 8);
      nir_def_replace(&intr->def, value);
      return true;
   }
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo: {
      int table = pilot_handle_table(intr->src[0].ssa);
      assert(table >= 0);
      struct pilot_buffer buf =
         pilot_load_buffer(b, intr->src[0].ssa, table, true, false);
      nir_def *value =
         pilot_load_bounded(b, buf, intr->src[1].ssa, intr->def.num_components,
                            intr->def.bit_size);
      nir_def_replace(&intr->def, value);
      return true;
   }
   case nir_intrinsic_load_ssbo_address: {
      int table = pilot_handle_table(intr->src[0].ssa);
      assert(table >= 0);
      bool null_check = ctx->inputs->robust_descriptors;
      struct pilot_buffer buf =
         pilot_load_buffer(b, intr->src[0].ssa, table, null_check, true);
      nir_def *addr =
         null_check ? pilot_buffer_address(b, buf, nir_ine_imm(b, buf.size, 0))
                    : nir_pack_64_2x32_split(b, buf.lo, buf.hi);
      nir_scalar off = nir_scalar_chase_movs(nir_get_scalar(intr->src[1].ssa, 0));
      if (!nir_scalar_is_const(off) || nir_scalar_as_uint(off) != 0)
         addr = nir_iadd(b, addr, nir_u2u64(b, intr->src[1].ssa));
      nir_def_replace(&intr->def, addr);
      return true;
   }
   default:
      return false;
   }
}

static nir_shader *
pilot_create(const nir_shader *nir, nir_function **func)
{
   nir_shader *pilot =
      nir_shader_create(NULL, MESA_SHADER_COMPUTE, nir->options);
   pilot->info.internal = true;
   pilot->info.float_controls_execution_mode =
      nir->info.float_controls_execution_mode;
   pilot->info.workgroup_size[0] = 1;
   pilot->info.workgroup_size[1] = 1;
   pilot->info.workgroup_size[2] = 1;
   pilot->info.num_ubos = nir->info.num_ubos;
   *func = nir_function_create(pilot, "main");
   (*func)->is_entrypoint = true;
   return pilot;
}

static bool
pilot_vectorize_cb(unsigned align_mul, unsigned align_offset,
                   unsigned bit_size, unsigned num_components,
                   int64_t hole_size, nir_intrinsic_instr *low,
                   nir_intrinsic_instr *high, void *data)
{
   return hole_size <= 0 && num_components <= 4 &&
          num_components * bit_size <= 128;
}

static void
pilot_optimize(nir_shader *pilot)
{
   const nir_load_store_vectorize_options opts = {
      .modes = nir_var_mem_global,
      .callback = pilot_vectorize_cb,
   };

   NIR_PASS(_, pilot, nir_opt_copy_prop);
   NIR_PASS(_, pilot, nir_opt_constant_folding);
   NIR_PASS(_, pilot, nir_opt_cse);
   NIR_PASS(_, pilot, nir_opt_load_store_vectorize, &opts);
   NIR_PASS(_, pilot, nir_opt_copy_prop);
   NIR_PASS(_, pilot, nir_opt_cse);
   NIR_PASS(_, pilot, nir_opt_dce);
   nir_shader_gather_info(pilot, nir_shader_get_entrypoint(pilot));
   nir_validate_shader(pilot, "preamble pilot");
}

struct buffer_base {
   uint32_t handle;
   uint32_t weight;
   nir_def *def;
};

struct buffer_bases {
   struct buffer_base *bases;
   unsigned count;
};

static bool
buffer_bases_supported(const struct pan_compile_inputs *inputs)
{
   return pan_arch(inputs->gpu_id) == 15 &&
          !(inputs->robust_modes & nir_var_mem_ssbo);
}

static bool
buffer_access_handle(const nir_intrinsic_instr *intr, uint32_t *handle)
{
   if ((intr->intrinsic != nir_intrinsic_load_ssbo &&
        intr->intrinsic != nir_intrinsic_load_ssbo_address) ||
       !nir_src_is_const(intr->src[0]))
      return false;

   *handle = nir_src_as_uint(intr->src[0]);
   return (*handle >> 24) < 64;
}

static int
compare_buffer_base(const void *a, const void *b)
{
   const struct buffer_base *x = a, *y = b;
   if (x->weight != y->weight)
      return x->weight > y->weight ? -1 : 1;
   return (x->handle > y->handle) - (x->handle < y->handle);
}

static void
collect_buffer_bases(nir_function_impl *impl, struct util_dynarray *bases)
{
   nir_foreach_block(block, impl) {
      uint32_t weight = 1u << MIN2(4 * block_loop_depth(block), 24);
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         uint32_t handle;
         if (!buffer_access_handle(nir_instr_as_intrinsic(instr), &handle))
            continue;
         struct buffer_base *base = NULL;
         util_dynarray_foreach(bases, struct buffer_base, it) {
            if (it->handle == handle)
               base = it;
         }
         if (!base) {
            util_dynarray_append_typed(bases, struct buffer_base,
                                       (struct buffer_base){.handle = handle});
            base = util_dynarray_last_ptr(bases, struct buffer_base);
         }
         base->weight = MIN2((uint64_t)base->weight + weight, UINT32_MAX);
      }
   }

   unsigned count = util_dynarray_num_elements(bases, struct buffer_base);
   if (count > 1)
      qsort(util_dynarray_begin(bases), count, sizeof(struct buffer_base),
            compare_buffer_base);
}

static nir_def *
buffer_address(nir_builder *b, nir_def *base, nir_def *offset, unsigned shift,
               unsigned align_offset, bool fold_imm)
{
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(offset, 0));
   if (nir_scalar_is_const(s)) {
      uint32_t bytes = (uint32_t)(nir_scalar_as_uint(s) << shift);
      return nir_iadd_imm(b, base, bytes);
   }

   nir_def *rest = shift ? nir_ishl_imm(b, offset, shift) : offset;
   unsigned fold = 0;
   if (!shift && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd) {
      for (unsigned i = 0; i < 2; i++) {
         nir_scalar c = nir_scalar_chase_alu_src(s, i);
         if (!nir_scalar_is_const(c))
            continue;
         nir_scalar x = nir_scalar_chase_alu_src(s, 1 - i);
         uint32_t imm = (uint32_t)nir_scalar_as_uint(c);
         nir_def *xd = nir_channel(b, x.def, x.comp);
         if (fold_imm && imm <= INT16_MAX) {
            rest = xd;
            fold = imm;
         } else if (align_offset > 0 && align_offset <= INT16_MAX) {
            uint32_t delta = imm - align_offset;
            rest = delta ? nir_iadd_imm(b, xd, delta) : xd;
            fold = align_offset;
         }
         break;
      }
   }

   nir_def *zext = nir_pack_64_2x32_split(b, rest, nir_imm_int(b, 0));
   nir_def *addr = nir_iadd(b, base, zext);
   return fold ? nir_iadd_imm(b, addr, fold) : addr;
}

static unsigned
address_user_align_offset(nir_def *def)
{
   if (!list_is_singular(&def->uses))
      return 0;

   nir_src *use = list_first_entry(&def->uses, nir_src, use_link);
   if (nir_src_is_if(use) ||
       nir_src_use_instr(use)->type != nir_instr_type_intrinsic)
      return 0;

   nir_intrinsic_instr *user = nir_instr_as_intrinsic(nir_src_use_instr(use));
   switch (user->intrinsic) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
      return use == &user->src[0] ? nir_intrinsic_align_offset(user) : 0;
   case nir_intrinsic_store_global:
      return use == &user->src[1] ? nir_intrinsic_align_offset(user) : 0;
   default:
      return 0;
   }
}

static bool
address_user_folds_offset(nir_def *def)
{
   if (!list_is_singular(&def->uses))
      return false;

   nir_src *use = list_first_entry(&def->uses, nir_src, use_link);
   if (nir_src_is_if(use) ||
       nir_src_use_instr(use)->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *user = nir_instr_as_intrinsic(nir_src_use_instr(use));
   switch (user->intrinsic) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
      return use == &user->src[0];
   case nir_intrinsic_store_global:
      return use == &user->src[1];
   default:
      return false;
   }
}

static bool
split_buffer_load(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   uint32_t handle;
   if (intr->intrinsic != nir_intrinsic_load_ssbo ||
       !buffer_access_handle(intr, &handle) ||
       ((nir_intrinsic_access(intr) & ACCESS_CAN_REORDER) &&
        !(nir_intrinsic_access(intr) & ACCESS_VOLATILE)))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *offset = intr->src[1].ssa;
   if (nir_intrinsic_offset_shift(intr))
      offset = nir_ishl_imm(b, offset, nir_intrinsic_offset_shift(intr));
   nir_def *addr = nir_load_ssbo_address(b, 1, 64, intr->src[0].ssa, offset);
   nir_def *value = nir_load_global(
      b, intr->def.num_components, intr->def.bit_size, addr,
      .access = nir_intrinsic_access(intr),
      .align_mul = nir_intrinsic_align_mul(intr),
      .align_offset = nir_intrinsic_align_offset(intr));
   nir_def_replace(&intr->def, value);
   return true;
}

static nir_def *
buffer_base_def(const struct buffer_bases *bb, uint32_t handle)
{
   for (unsigned i = 0; i < bb->count; i++) {
      if (bb->bases[i].handle == handle)
         return bb->bases[i].def;
   }
   return NULL;
}

static bool
restore_split_load(nir_builder *b, nir_intrinsic_instr *load,
                   const struct buffer_bases *bb)
{
   nir_instr *src = nir_def_instr(load->src[0].ssa);
   if (src->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *addr = nir_instr_as_intrinsic(src);
   uint32_t handle;
   if (addr->intrinsic != nir_intrinsic_load_ssbo_address ||
       !buffer_access_handle(addr, &handle) || buffer_base_def(bb, handle) ||
       !list_is_singular(&addr->def.uses))
      return false;

   b->cursor = nir_before_instr(&load->instr);
   nir_def *value = nir_load_ssbo(
      b, load->def.num_components, load->def.bit_size, addr->src[0].ssa,
      addr->src[1].ssa, .access = nir_intrinsic_access(load),
      .align_mul = nir_intrinsic_align_mul(load),
      .align_offset = nir_intrinsic_align_offset(load));
   nir_def_replace(&load->def, value);
   nir_instr_remove(&addr->instr);
   return true;
}

static bool
rewrite_buffer_access(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct buffer_bases *bb = data;
   if (intr->intrinsic == nir_intrinsic_load_global)
      return restore_split_load(b, intr, bb);

   uint32_t handle;
   if (!buffer_access_handle(intr, &handle))
      return false;

   nir_def *base = buffer_base_def(bb, handle);
   if (!base)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   if (intr->intrinsic == nir_intrinsic_load_ssbo_address) {
      nir_def_replace(&intr->def,
                      buffer_address(b, base, intr->src[1].ssa, 0,
                                     address_user_align_offset(&intr->def),
                                     address_user_folds_offset(&intr->def)));
      return true;
   }

   nir_def *addr = buffer_address(b, base, intr->src[1].ssa,
                                  nir_intrinsic_offset_shift(intr),
                                  nir_intrinsic_align_offset(intr), true);
   nir_def *value = nir_load_global(
      b, intr->def.num_components, intr->def.bit_size, addr,
      .access = nir_intrinsic_access(intr),
      .align_mul = nir_intrinsic_align_mul(intr),
      .align_offset = nir_intrinsic_align_offset(intr));
   nir_def_replace(&intr->def, value);
   return true;
}

#define PILOT_FAU_PAGE_WORDS 128

static unsigned
buffer_base_room(nir_function_impl *main, unsigned storage,
                 unsigned result_base, unsigned max_push)
{
   unsigned used = 0;
   bool wide = false;
   if (main->preamble) {
      nir_foreach_block(block, main->preamble->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_store_preamble)
               continue;
            nir_def *value = nir_instr_as_intrinsic(instr)->src[0].ssa;
            unsigned size =
               DIV_ROUND_UP(value->num_components * value->bit_size, 32);
            used += size >= 2 ? ALIGN_POT(size, 2) : size;
            wide |= size > 2;
         }
      }
   }

   used = ALIGN_POT(used, 2);
   for (unsigned page = PILOT_FAU_PAGE_WORDS; wide && page < max_push;
        page += PILOT_FAU_PAGE_WORDS) {
      if (result_base < page)
         used += 4;
   }
   return storage > used ? storage - used : 0;
}

static void
pilot_emit_buffer_bases(nir_shader *pilot, nir_function_impl *main,
                        struct buffer_bases *bb, unsigned first)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(pilot);
   nir_builder pb = nir_builder_at(nir_after_impl(impl));
   nir_builder mb = nir_builder_at(nir_before_impl(main));

   for (unsigned i = 0; i < bb->count; i++) {
      unsigned base = first + 2 * i;
      nir_def *addr =
         nir_load_ssbo_address(&pb, 1, 64, nir_imm_int(&pb, bb->bases[i].handle),
                               nir_imm_int(&pb, 0));
      nir_store_preamble(&pb, addr, .base = base);
      bb->bases[i].def = nir_load_preamble(&mb, 1, 64, .base = base);
   }

   nir_progress(true, impl, nir_metadata_none);
}

nir_shader *
pan_nir_opt_preamble(nir_shader *nir, const struct pan_compile_inputs *inputs,
                     nir_shader **preamble, struct pan_fau_virtual *virt)
{
   *preamble = NULL;
   unsigned max_push = pan_max_push(inputs->gpu_id, inputs->gpu_variant);
   if (inputs->fau.reserved >= max_push)
      return NULL;

   nir_shader *clone = nir_shader_clone(NULL, nir);
   nir_function_impl *main = nir_shader_get_entrypoint(clone);
   bool buffer_addresses = buffer_bases_supported(inputs);
   if (buffer_addresses)
      NIR_PASS(_, clone, nir_shader_intrinsics_pass, split_buffer_load,
               nir_metadata_control_flow, NULL);
   nir_index_ssa_defs(main);
   struct preamble_ctx ctx = {
      .inputs = inputs,
      .result_base = ALIGN_POT(inputs->fau.reserved, 2),
      .allowed = calloc(main->ssa_alloc, 1),
      .buffer_addresses = buffer_addresses,
   };
   if (!ctx.allowed) {
      ralloc_free(clone);
      return NULL;
   }
   unsigned ubo_words = count_pushable_ubo_words(main, &ctx);
   unsigned available = max_push - ctx.result_base;
   ctx.fau_pressure = (ctx.result_base + ubo_words) * 4 > max_push * 3;
   mark_speculatable_uniform_loads(main, &ctx);
   unsigned storage = ctx.fau_pressure ? available :
                      available - MIN2(ubo_words, available);
   if (!storage) {
      free(ctx.allowed);
      ralloc_free(clone);
      return NULL;
   }
   const nir_opt_preamble_options opts = {
      .def_size = def_size,
      .preamble_storage_size = {storage},
      .instr_cost_cb = instr_cost,
      .rewrite_cost_cb = rewrite_cost,
      .avoid_instr_cb = avoid_instr,
      .cb_data = &ctx,
   };
   unsigned size[nir_preamble_num_classes] = {0};
   bool progress = nir_opt_preamble(clone, &opts, size);
   free(ctx.allowed);

   struct util_dynarray bases;
   util_dynarray_init(&bases, NULL);
   unsigned first = ALIGN_POT(size[0], 2);
   struct buffer_bases bb = {0};
   unsigned room = buffer_addresses ? buffer_base_room(main, storage,
                                                       ctx.result_base,
                                                       max_push)
                                    : 0;
   if (room >= 2) {
      collect_buffer_bases(main, &bases);
      bb.bases = util_dynarray_begin(&bases);
      bb.count = MIN2(util_dynarray_num_elements(&bases, struct buffer_base),
                      room / 2);
   }

   if (!progress && !bb.count) {
      util_dynarray_fini(&bases);
      ralloc_free(clone);
      return NULL;
   }

   nir_function *func;
   nir_shader *pilot = pilot_create(nir, &func);
   if (nir->constant_data_size) {
      pilot->constant_data = ralloc_size(pilot, nir->constant_data_size);
      memcpy(pilot->constant_data, nir->constant_data, nir->constant_data_size);
      pilot->constant_data_size = nir->constant_data_size;
   }
   if (progress) {
      func->impl = nir_function_impl_clone(pilot, main->preamble->impl);
      func->impl->function = func;
      nir_foreach_block(block, func->impl)
      {
         nir_if *nif = nir_block_get_following_if(block);
         if (nif)
            nif->control = nir_selection_control_dont_flatten;
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_load_push_constant:
            case nir_intrinsic_load_ubo:
            case nir_intrinsic_load_constant:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_ssbo:
            case nir_intrinsic_load_ssbo_address:
            case nir_intrinsic_store_preamble:
               break;
            default:
               util_dynarray_fini(&bases);
               ralloc_free(pilot);
               ralloc_free(clone);
               return NULL;
            }
         }
      }
      exec_node_remove(&main->preamble->node);
      main->preamble = NULL;
   } else {
      nir_function_impl_create(func);
   }

   unsigned slots = size[0];
   if (bb.count) {
      pilot_emit_buffer_bases(pilot, main, &bb, first);
      slots = first + 2 * bb.count;
   }
   if (buffer_addresses)
      NIR_PASS(_, clone, nir_shader_intrinsics_pass, rewrite_buffer_access,
               nir_metadata_control_flow, &bb);
   if (bb.count)
      NIR_PASS(_, clone, nir_opt_cse);
   util_dynarray_fini(&bases);

   NIR_PASS(_, pilot, nir_shader_intrinsics_pass, lower_preamble,
            nir_metadata_control_flow, &ctx);
   NIR_PASS(_, pilot, nir_opt_copy_prop);
   NIR_PASS(_, pilot, nir_opt_constant_folding);
   NIR_PASS(_, pilot, nir_opt_cse);

   int16_t *value_of_base = malloc(MAX2(slots, 1) * sizeof(int16_t));
   memset(value_of_base, 0xff, MAX2(slots, 1) * sizeof(int16_t));
   struct pilot_values pv = {
      .virt = virt,
      .value_of_base = value_of_base,
   };
   register_pilot_values(pilot, &pv);
   NIR_PASS(_, clone, nir_shader_intrinsics_pass, rebase_main_load,
            nir_metadata_control_flow, &pv);
   free(value_of_base);

   NIR_PASS(_, clone, nir_opt_dce);
   nir_validate_shader(clone, "preamble main");
   nir_validate_shader(pilot, "preamble pilot");
   *preamble = pilot;
   return clone;
}

static bool
place_pilot_store(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct pan_fau_virtual *virt = data;
   if (intr->intrinsic != nir_intrinsic_store_preamble)
      return false;

   const struct pan_fau_value *value = &virt->values[nir_intrinsic_base(intr)];
   assert(value->alias < 0 && virt->map[value->word] >= 0);

   b->cursor = nir_before_instr(&intr->instr);
   nir_store_global(b, intr->src[0].ssa,
                    nir_iadd_imm(b, nir_load_pilot_arg_pan(b, .base = 0),
                                 virt->map[value->word] * 4),
                    .align_mul = 4);
   nir_instr_remove(&intr->instr);
   return true;
}

void
pan_nir_pilot_place_results(nir_shader *pilot,
                            const struct pan_fau_virtual *virt)
{
   assert(virt->placed);
   NIR_PASS(_, pilot, nir_shader_intrinsics_pass, place_pilot_store,
            nir_metadata_control_flow, (void *)virt);
   pilot_optimize(pilot);
}

struct pilot_copy_run {
   unsigned key;
   unsigned word;
   unsigned offset;
   unsigned count;
};

static void
pilot_emit_fau_copies(nir_builder *b, const struct pan_fau_layout *fau)
{
   struct pilot_copy_run runs[PAN_MAX_PUSH];
   unsigned run_count = 0;

   pan_fau_foreach_reloc(fau, i) {
      struct pan_ubo_relocation reloc = fau->words[i].relocation;
      struct pilot_copy_run *last = run_count ? &runs[run_count - 1] : NULL;

      if (last && last->key == reloc.ubo && last->count < 4 &&
          last->word + last->count == i &&
          last->offset + 4 * last->count == reloc.offset &&
          (last->offset & ~15) == (reloc.offset & ~15)) {
         last->count++;
      } else {
         runs[run_count++] = (struct pilot_copy_run){
            .key = reloc.ubo,
            .word = i,
            .offset = reloc.offset,
            .count = 1,
         };
      }
   }

   nir_def *fau_base = nir_load_pilot_arg_pan(b, .base = 0);

   for (unsigned r = 0; r < run_count; r++) {
      const unsigned table = pan_ubo_reloc_table(runs[r].key);
      const unsigned index = pan_ubo_reloc_index(runs[r].key);
      struct pilot_buffer buf =
         pilot_load_buffer(b, nir_imm_int(b, pan_res_handle(table, index)),
                           table, true, false);
      nir_def *chunk_end = nir_imm_int(b, (runs[r].offset & ~15) + 16);
      nir_def *addr =
         pilot_buffer_address(b, buf, nir_uge(b, buf.size, chunk_end));
      nir_def *value = nir_load_global_constant(
         b, runs[r].count, 32, nir_iadd_imm(b, addr, runs[r].offset),
         .align_mul = 4);
      nir_store_global(b, value, nir_iadd_imm(b, fau_base, runs[r].word * 4),
                       .align_mul = 4);
   }
}

bool
pan_nir_pilot_add_fau_copies(nir_shader **pilot, const nir_shader *nir,
                             const struct pan_fau_layout *fau)
{
   bool has_relocs = false;
   pan_fau_foreach_reloc(fau, i) {
      has_relocs = true;
      break;
   }

   if (!has_relocs)
      return false;

   if (!*pilot) {
      nir_function *func;
      *pilot = pilot_create(nir, &func);
      nir_function_impl_create(func);
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(*pilot);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   pilot_emit_fau_copies(&b, fau);
   nir_progress(true, impl, nir_metadata_none);
   pilot_optimize(*pilot);
   return true;
}
