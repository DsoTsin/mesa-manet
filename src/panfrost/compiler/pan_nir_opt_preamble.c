#include "pan_nir.h"

struct preamble_ctx {
   const struct pan_compile_inputs *inputs;
   uint8_t *allowed;
   unsigned result_base;
   bool fau_pressure;
};

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
         allowed = nir_src_is_const(intr->src[0]);
         break;
      case nir_intrinsic_load_ubo:
      case nir_intrinsic_load_constant:
      case nir_intrinsic_load_global_constant:
         allowed = true;
         break;
      case nir_intrinsic_load_ssbo:
         allowed = (nir_intrinsic_access(intr) & ACCESS_CAN_REORDER) &&
                   !(nir_intrinsic_access(intr) & ACCESS_VOLATILE);
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
         bool uniform_load =
            (intr->intrinsic == nir_intrinsic_load_push_constant &&
             nir_src_is_const(intr->src[0])) ||
            (!robust_ubo && pushable_ubo_load(intr, ctx));
         if (uniform_load && nir_intrinsic_has_access(intr))
            nir_intrinsic_set_access(intr, nir_intrinsic_access(intr) |
                                              ACCESS_CAN_REORDER |
                                              ACCESS_CAN_SPECULATE);
      }
   }
}

static bool
lower_main(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct preamble_ctx *ctx = data;
   if (intr->intrinsic != nir_intrinsic_load_preamble)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = nir_load_push_constant(
      b, intr->def.num_components, intr->def.bit_size,
      nir_imm_int(b, (ctx->result_base + nir_intrinsic_base(intr)) * 4));
   nir_def_replace(&intr->def, value);
   return true;
}

static bool
lower_preamble(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct preamble_ctx *ctx = data;
   if (intr->intrinsic != nir_intrinsic_load_push_constant &&
       intr->intrinsic != nir_intrinsic_store_preamble)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *base = nir_load_push_constant(b, 1, 64, nir_imm_int(b, 0));
   if (intr->intrinsic == nir_intrinsic_store_preamble) {
      unsigned offset = (ctx->result_base + nir_intrinsic_base(intr)) * 4;
      nir_store_global(b, intr->src[0].ssa, nir_iadd_imm(b, base, offset),
                       .align_mul = 4);
      nir_instr_remove(&intr->instr);
   } else {
      unsigned offset = nir_src_as_uint(intr->src[0]);
      assert(nir_intrinsic_base(intr) == 0);
      assert(offset + DIV_ROUND_UP(
                         intr->def.bit_size * intr->def.num_components, 8) <=
             ctx->inputs->fau.reserved * 4);
      nir_def *value = nir_load_global_constant(
         b, intr->def.num_components, intr->def.bit_size,
         nir_iadd_imm(b, base, offset), .align_mul = 8,
         .align_offset = offset % 8);
      nir_def_replace(&intr->def, value);
   }
   return true;
}

nir_shader *
pan_nir_opt_preamble(nir_shader *nir, const struct pan_compile_inputs *inputs,
                     nir_shader **preamble, unsigned *fau_words)
{
   *preamble = NULL;
   *fau_words = inputs->fau.reserved;
   if (inputs->fau.reserved >= PAN_MAX_PUSH)
      return NULL;

   nir_shader *clone = nir_shader_clone(NULL, nir);
   nir_function_impl *main = nir_shader_get_entrypoint(clone);
   nir_index_ssa_defs(main);
   struct preamble_ctx ctx = {
      .inputs = inputs,
      .result_base = ALIGN_POT(inputs->fau.reserved, 2),
      .allowed = calloc(main->ssa_alloc, 1),
   };
   if (!ctx.allowed) {
      ralloc_free(clone);
      return NULL;
   }
   mark_speculatable_uniform_loads(main, &ctx);
   unsigned ubo_words = count_pushable_ubo_words(main, &ctx);
   unsigned available = PAN_MAX_PUSH - ctx.result_base;
   ctx.fau_pressure = (ctx.result_base + ubo_words) * 4 > PAN_MAX_PUSH * 3;
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
   if (!progress) {
      ralloc_free(clone);
      return NULL;
   }

   nir_shader *pilot =
      nir_shader_create(NULL, MESA_SHADER_COMPUTE, nir->options);
   pilot->info.internal = true;
   pilot->info.float_controls_execution_mode =
      nir->info.float_controls_execution_mode;
   pilot->info.workgroup_size[0] = 1;
   pilot->info.workgroup_size[1] = 1;
   pilot->info.workgroup_size[2] = 1;
   pilot->info.num_ubos = nir->info.num_ubos;
   if (nir->constant_data_size) {
      pilot->constant_data = ralloc_size(pilot, nir->constant_data_size);
      memcpy(pilot->constant_data, nir->constant_data, nir->constant_data_size);
      pilot->constant_data_size = nir->constant_data_size;
   }
   nir_function *func = nir_function_create(pilot, "main");
   func->is_entrypoint = true;
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
         case nir_intrinsic_store_preamble:
            break;
         default:
            ralloc_free(pilot);
            ralloc_free(clone);
            return NULL;
         }
      }
   }
   exec_node_remove(&main->preamble->node);
   main->preamble = NULL;

   NIR_PASS(_, clone, nir_shader_intrinsics_pass, lower_main,
            nir_metadata_control_flow, &ctx);
   NIR_PASS(_, pilot, nir_shader_intrinsics_pass, lower_preamble,
            nir_metadata_control_flow, &ctx);
   NIR_PASS(_, clone, nir_opt_dce);
   NIR_PASS(_, pilot, nir_opt_copy_prop);
   NIR_PASS(_, pilot, nir_opt_constant_folding);
   NIR_PASS(_, pilot, nir_opt_cse);
   NIR_PASS(_, pilot, nir_opt_dce);
   nir_shader_gather_info(pilot, func->impl);
   nir_validate_shader(clone, "preamble main");
   nir_validate_shader(pilot, "preamble pilot");
   *preamble = pilot;
   *fau_words = ALIGN_POT(ctx.result_base + size[0], 2);
   return clone;
}
