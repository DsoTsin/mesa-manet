#ifndef PANVK_DATA_GRAPH_NIR_INTERPRETER_H
#define PANVK_DATA_GRAPH_NIR_INTERPRETER_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "nir_constant_expressions.h"
#include "nir_deref.h"

struct graph_interpreter {
   const struct panvk_data_graph *graph;
   nir_shader *nir;
   nir_const_value (*values)[NIR_MAX_VEC_COMPONENTS];
   nir_variable *variables[32];
   nir_const_value locals[32][NIR_MAX_VEC_COMPONENTS];
   unsigned variable_count;
   uint8_t **buffers;
   uint32_t invocation;
   unsigned writes;
   unsigned instructions;
};

static nir_const_value *
graph_source(struct graph_interpreter *e, nir_src src)
{
   return e->values[src.ssa->index];
}

static unsigned
graph_local(struct graph_interpreter *e, nir_src src)
{
   nir_variable *v = nir_deref_instr_get_variable(nir_src_as_deref(src));
   for (unsigned i = 0; i < e->variable_count; i++) {
      if (e->variables[i] == v)
         return i;
   }
   abort();
}

static void
graph_intrinsic(struct graph_interpreter *e, nir_intrinsic_instr *intr)
{
   nir_const_value *dest = nir_intrinsic_infos[intr->intrinsic].has_dest
                              ? e->values[intr->def.index]
                              : NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global_invocation_id:
      dest[0].u32 = e->invocation % (65535 * 64);
      dest[1].u32 = e->invocation / (65535 * 64);
      dest[2].u32 = 0;
      break;
   case nir_intrinsic_vulkan_resource_index:
      dest[0] = graph_source(e, intr->src[0])[0];
      dest[1].u32 = dest[2].u32 = 0;
      break;
   case nir_intrinsic_load_vulkan_descriptor:
      memcpy(dest, graph_source(e, intr->src[0]), 3 * sizeof(*dest));
      break;
   case nir_intrinsic_load_deref:
      memcpy(dest, e->locals[graph_local(e, intr->src[0])],
             intr->def.num_components * sizeof(*dest));
      break;
   case nir_intrinsic_store_deref: {
      nir_const_value *local = e->locals[graph_local(e, intr->src[0])];
      nir_const_value *value = graph_source(e, intr->src[1]);
      for (unsigned c = 0; c < intr->src[1].ssa->num_components; c++) {
         if (nir_intrinsic_write_mask(intr) & (1u << c))
            local[c] = value[c];
      }
      break;
   }
   case nir_intrinsic_tensor_read_arm:
   case nir_intrinsic_tensor_write_arm: {
      unsigned id = graph_source(e, intr->src[0])[0].u32;
      assert(id < e->graph->count && e->buffers[id]);
      const struct panvk_graph_value *v = &e->graph->values[id];
      nir_const_value *coords = graph_source(e, intr->src[1]);
      unsigned index = 0;
      assert(intr->src[1].ssa->num_components == v->rank);
      for (unsigned d = 0; d < v->rank; d++) {
         assert(coords[d].u32 < v->dims[d]);
         index = index * (v->storage ? v->storage->dims[d] : v->dims[d]) + coords[d].u32;
      }
      const struct panvk_graph_value *owner = v->storage ? v->storage : v;
      if (owner->alias_concat) {
         assert(v->rank == 4 && v->format == VK_FORMAT_R8_SINT);
         index = (((coords[0].u32 * owner->dims[1] + coords[1].u32) *
                   DIV_ROUND_UP(owner->dims[2], 4) + coords[2].u32 / 4) *
                  owner->dims[3] * 4) + (coords[3].u32 / 16) * 64 +
                 (coords[2].u32 % 4) * 16 + coords[3].u32 % 16;
      }
      unsigned bytes = vk_format_get_blocksize(v->format);
      if (intr->intrinsic == nir_intrinsic_tensor_read_arm) {
         assert(intr->def.bit_size == bytes * 8);
         memcpy(&dest[0], e->buffers[id] + index * bytes, bytes);
      } else {
         assert(intr->src[2].ssa->bit_size == bytes * 8);
         memcpy(e->buffers[id] + index * bytes, graph_source(e, intr->src[2]),
                bytes);
         e->writes++;
      }
      break;
   }
   default:
      abort();
   }
}

static unsigned graph_execute_list(struct graph_interpreter *e,
                                   struct exec_list *list);

static unsigned
graph_execute_block(struct graph_interpreter *e, nir_block *block)
{
   nir_foreach_instr(instr, block) {
      assert(++e->instructions < 10000000);
      switch (instr->type) {
      case nir_instr_type_load_const: {
         nir_load_const_instr *load = nir_instr_as_load_const(instr);
         memcpy(e->values[load->def.index], load->value,
                load->def.num_components * sizeof(nir_const_value));
         break;
      }
      case nir_instr_type_alu: {
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         nir_const_value inputs[NIR_ALU_MAX_INPUTS][NIR_MAX_VEC_COMPONENTS] = {
            0};
         nir_const_value *srcs[NIR_ALU_MAX_INPUTS];
         unsigned bits =
            nir_alu_type_get_type_size(nir_op_infos[alu->op].output_type)
               ? 0
               : alu->def.bit_size;
         for (unsigned s = 0; s < nir_op_infos[alu->op].num_inputs; s++) {
            srcs[s] = inputs[s];
            if (!bits && !nir_alu_type_get_type_size(
                            nir_op_infos[alu->op].input_types[s]))
               bits = alu->src[s].src.ssa->bit_size;
            for (unsigned c = 0; c < nir_ssa_alu_instr_src_components(alu, s);
                 c++)
               inputs[s][c] =
                  graph_source(e, alu->src[s].src)[alu->src[s].swizzle[c]];
         }
         nir_eval_const_opcode(alu->op, e->values[alu->def.index], NULL,
                               alu->def.num_components, bits ? bits : 32, srcs,
                               e->nir->info.float_controls_execution_mode);
         break;
      }
      case nir_instr_type_intrinsic:
         graph_intrinsic(e, nir_instr_as_intrinsic(instr));
         break;
      case nir_instr_type_deref:
      case nir_instr_type_undef:
         break;
      case nir_instr_type_jump: {
         nir_jump_instr *jump = nir_instr_as_jump(instr);
         assert(jump->type == nir_jump_break);
         return 1;
      }
      default:
         abort();
      }
   }
   return 0;
}

static unsigned
graph_execute_list(struct graph_interpreter *e, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_block) {
         if (graph_execute_block(e, nir_cf_node_as_block(node)))
            return 1;
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         bool condition = graph_source(e, nif->condition)[0].b;
         if (graph_execute_list(e,
                                condition ? &nif->then_list : &nif->else_list))
            return 1;
      } else {
         assert(node->type == nir_cf_node_loop);
         nir_loop *loop = nir_cf_node_as_loop(node);
         assert(exec_list_is_empty(&loop->continue_list));
         unsigned iterations = 0;
         while (!graph_execute_list(e, &loop->body))
            assert(++iterations < 1000000);
      }
   }
   return 0;
}

static unsigned
graph_interpret(struct graph_interpreter *e, uint32_t invocation)
{
   e->invocation = invocation;
   e->writes = e->instructions = 0;
   nir_function_impl *impl = nir_shader_get_entrypoint(e->nir);
   memset(e->values, 0, impl->ssa_alloc * sizeof(*e->values));
   memset(e->locals, 0, sizeof(e->locals));
   assert(!graph_execute_list(e, &impl->body));
   return e->writes;
}

static void
graph_interpreter_init(struct graph_interpreter *e,
                       const struct panvk_data_graph *g, nir_shader *nir,
                       uint8_t **buffers)
{
   *e = (struct graph_interpreter){.graph = g, .nir = nir, .buffers = buffers};
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   e->values = calloc(impl->ssa_alloc, sizeof(*e->values));
   assert(e->values);
   nir_foreach_function_temp_variable(var, impl)
   {
      assert(e->variable_count < ARRAY_SIZE(e->variables));
      e->variables[e->variable_count++] = var;
   }
}

#endif
