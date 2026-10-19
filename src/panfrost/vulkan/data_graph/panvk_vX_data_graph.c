#include "genxml/gen_macros.h"
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv.h"
#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_meta.h"
#include "panvk_data_graph.h"
#include "panvk_data_graph_neural.h"
#include "panvk_device.h"
#include "panvk_device_memory.h"
#include "panvk_entrypoints.h"
#include "panvk_physical_device.h"
#include "panvk_priv_bo.h"
#include "panvk_tensor.h"
#include "panvk_tensor_desc.h"
#include "vk_format.h"
#include "vk_log.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"
#include "vk_shader_module.h"
#include "vk_util.h"

#if PAN_ARCH >= 15
#ifdef HAVE_PAN_KMOD_KBASE
#include "kmod/kbase_kmod.h"
#endif
#include "neural/pan_graph_matmul_select.h"
#include "neural/pan_graph_conv_v15.h"
#include "neural/pan_graph_matmul_v15.h"
#include "neural/pan_graph_program_deps.h"
#include "neural/pan_graph_program_memory.h"
#include "neural/pan_graph_program_v15.h"
#include "panvk_data_graph_csf.h"

struct panvk_data_graph_neural {
   struct panvk_priv_bo *code;
   uint64_t coefficients;
   uint64_t table;
   uint32_t coefficient_channels;
   uint32_t inputs[2];
   uint16_t wait_mask;
   struct panvk_neural_program dispatch;
};

struct panvk_data_graph_whole {
   struct panvk_data_graph_neural *programs;
   struct pan_graph_program_memory memory;
   uint32_t inputs[2];
   uint32_t output;
   uint32_t count;
   bool dependency_sync;
};
#endif

struct panvk_data_graph_pipeline {
   struct vk_pipeline vk;
   VkAllocationCallbacks alloc;
   int refs;
   struct panvk_data_graph *graph;
   VkDescriptorSetLayout set_layout;
   VkPipelineLayout layout;
   struct panvk_priv_bo *constants;
   VkPipeline *passes;
   uint64_t transient_size;
   uint32_t dispatch_count;
   bool whole_neural;
#if PAN_ARCH >= 15
   struct panvk_data_graph_neural *neural;
   struct panvk_data_graph_whole whole;
   uint32_t neural_count;
#endif
};

struct panvk_data_graph_session {
   struct vk_object_base base;
   struct panvk_data_graph_pipeline *pipeline;
   uint64_t transient;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_data_graph_pipeline, vk.base, VkPipeline,
                               VK_OBJECT_TYPE_PIPELINE)
VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_data_graph_session, base,
                               VkDataGraphPipelineSessionARM,
                               VK_OBJECT_TYPE_DATA_GRAPH_PIPELINE_SESSION_ARM)

#if PAN_ARCH >= 15
static void
release_whole(const VkAllocationCallbacks *alloc,
              struct panvk_data_graph_whole *whole)
{
   for (uint32_t i = 0; whole->programs && i < whole->count; i++) {
      if (whole->programs[i].code)
         panvk_priv_bo_unref(whole->programs[i].code);
   }
   vk_free(alloc, whole->programs);
   pan_graph_program_memory_finish(&whole->memory);
   *whole = (struct panvk_data_graph_whole){0};
}

#endif
static void
release_pipeline(struct vk_device *dev, struct panvk_data_graph_pipeline *p)
{
   if (!p_atomic_dec_zero(&p->refs))
      return;
   VkDevice device = vk_device_to_handle(dev);
   if (p->passes) {
      for (unsigned i = 0; i < p->graph->pass_count; i++)
         dev->dispatch_table.DestroyPipeline(device, p->passes[i], &p->alloc);
   }
   dev->dispatch_table.DestroyPipelineLayout(device, p->layout, &p->alloc);
   dev->dispatch_table.DestroyDescriptorSetLayout(device, p->set_layout,
                                                  &p->alloc);
   if (p->constants)
      panvk_priv_bo_unref(p->constants);
#if PAN_ARCH >= 15
   release_whole(&p->alloc, &p->whole);
   if (p->neural) {
      for (unsigned i = 0; i < p->graph->pass_count; i++) {
         if (p->neural[i].code)
            panvk_priv_bo_unref(p->neural[i].code);
      }
   }
   vk_free(&p->alloc, p->neural);
#endif
   vk_free(&p->alloc, p->passes);
   ralloc_free(p->graph);
   vk_pipeline_free(dev, &p->alloc, &p->vk);
}

static void
destroy_pipeline(struct vk_device *dev, struct vk_pipeline *pipeline,
                 const VkAllocationCallbacks *alloc)
{
   release_pipeline(
      dev, container_of(pipeline, struct panvk_data_graph_pipeline, vk));
}

static void
bind_pipeline(struct vk_command_buffer *cmd, struct vk_pipeline *pipeline)
{
   struct panvk_cmd_buffer *cmdbuf =
      container_of(cmd, struct panvk_cmd_buffer, vk);
   cmdbuf->state.data_graph.pipeline = pipeline;
}

static const struct vk_pipeline_ops graph_ops = {
   .destroy = destroy_pipeline,
   .cmd_bind = bind_pipeline,
};

#if PAN_ARCH >= 15
static VkResult
create_whole_neural(struct panvk_device *dev,
                    struct panvk_data_graph_pipeline *pipeline)
{
   struct pan_graph_memory_cost_target cost_target;
   uint32_t core_budget = 0;
#ifdef HAVE_PAN_KMOD_KBASE
   struct kbase_kmod_l2_cache_props cache;
   if (to_panvk_physical_device(dev->vk.physical)->kbase_node_path[0] &&
       kbase_kmod_csf_has_neural(dev->kmod.dev) &&
       kbase_kmod_get_l2_cache_props(dev->kmod.dev, &cache)) {
      core_budget = kbase_kmod_csf_neural_core_count(dev->kmod.dev);
      if (!pan_graph_program_cost_target_v15_init(cache.slices, cache.line_size,
                                                  cache.size_per_slice,
                                                  core_budget, &cost_target))
         core_budget = 0;
   }
#endif
   if (!core_budget)
      return VK_SUCCESS;
   struct panvk_data_graph_matmul_add graph = {0};
   VkResult result =
      panvk_data_graph_matmul_add_extract(pipeline->graph, &graph);
   if (result != VK_SUCCESS)
      return result == VK_ERROR_OUT_OF_HOST_MEMORY ? result : VK_SUCCESS;
   struct pan_graph_program_schedule schedule = {0};
   struct pan_graph_program_deps deps = {0};
   uint8_t *waited = NULL;
   struct panvk_data_graph_whole next = {0};
   const struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   result = VK_SUCCESS;
   if (graph.input_indices[0] > 1 || graph.input_indices[1] > 1 ||
       graph.input_indices[0] == graph.input_indices[1] ||
       pipeline->graph->values[graph.output].resource_index != 0)
      goto cleanup;
   enum pan_graph_tsu_status status = pan_graph_program_schedule_matmul_add_v15(
      &graph.info, graph.adds, graph.add_count, &target, &cost_target,
      core_budget, 8, &schedule);
   if (status != PAN_GRAPH_TSU_SUCCESS) {
      if (status == PAN_GRAPH_TSU_OUT_OF_MEMORY)
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto cleanup;
   }
   if (!schedule.count || schedule.count > UINT32_MAX ||
       SIZE_MAX / schedule.count < sizeof(*next.programs))
      goto cleanup;
   struct pan_graph_program_memory_binding bindings[3];
   for (unsigned i = 0; i < 3; i++) {
      const struct panvk_graph_value *v =
         pipeline->graph->values + (i < 2 ? graph.inputs[i] : graph.output);
      bindings[i] = (struct pan_graph_program_memory_binding){
         .binding = {v->binding, v->set, v->array_element},
         .ordinal = i < 2 ? graph.input_indices[i] : v->resource_index,
      };
   }
   status = pan_graph_program_memory_matmul_add_v15(
      &graph.info, graph.adds, graph.add_count, &schedule, &cost_target,
      bindings, &next.memory);
   if (status != PAN_GRAPH_TSU_SUCCESS) {
      if (status == PAN_GRAPH_TSU_OUT_OF_MEMORY)
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto cleanup;
   }
   next.count = schedule.count;
   if (next.count > 1 && dev->csf.sb.iter_count) {
      status = pan_graph_program_deps_matmul_add_v15(
         &graph.info, graph.add_count, &next.memory, &deps);
      if (status == PAN_GRAPH_TSU_OUT_OF_MEMORY) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto cleanup;
      }
      next.dependency_sync = status == PAN_GRAPH_TSU_SUCCESS;
      if (next.dependency_sync) {
         waited = calloc(next.count, sizeof(*waited));
         if (!waited) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto cleanup;
         }
      }
   }
   next.programs =
      vk_zalloc(&pipeline->alloc, next.count * sizeof(*next.programs), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!next.programs) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto cleanup;
   }
   for (uint32_t i = 0; i < next.count; i++) {
      if (next.dependency_sync)
         next.programs[i].wait_mask = pan_graph_program_deps_wait_mask(
            &deps, i, PANVK_SB_ITER_START, dev->csf.sb.iter_count, waited);
      struct pan_graph_program_kernel kernel = {0};
      const struct pan_graph_scheduled_program *program = schedule.programs + i;
      status = pan_graph_program_v15_compile_candidate(
         &program->plan, &program->selection, graph.adds, graph.add_count,
         next.memory.program_bindings + i, &target, &kernel);
      if (status != PAN_GRAPH_TSU_SUCCESS) {
         if (status == PAN_GRAPH_TSU_OUT_OF_MEMORY)
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
         pan_graph_v15_program_cleanup(&kernel.program);
         goto cleanup;
      }
      struct panvk_data_graph_neural *neural = next.programs + i;
      result =
         panvk_priv_bo_create(dev, kernel.program.size, 0,
                              VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, &neural->code);
      if (result == VK_SUCCESS) {
         memcpy(neural->code->addr.host, kernel.program.data,
                kernel.program.size);
         panvk_priv_bo_flush(neural->code, 0, kernel.program.size);
         memcpy(neural->dispatch.extent, kernel.extent, sizeof(kernel.extent));
         memcpy(neural->dispatch.loops, kernel.loops, sizeof(kernel.loops));
         neural->dispatch.address = neural->code->addr.dev;
         neural->dispatch.size = kernel.program.size;
         neural->dispatch.dispatch_id = i;
      }
      pan_graph_v15_program_cleanup(&kernel.program);
      if (result != VK_SUCCESS)
         goto cleanup;
   }
   for (unsigned i = 0; i < 2; i++)
      next.inputs[graph.input_indices[i]] = graph.inputs[i];
   next.output = graph.output;
   pipeline->whole = next;
   next = (struct panvk_data_graph_whole){0};
   pipeline->whole_neural = true;
   pipeline->transient_size = pipeline->whole.memory.lifetimes.stats.peak;
   pipeline->dispatch_count = pipeline->whole.count;
   pipeline->neural_count = pipeline->whole.count;
cleanup:
   free(waited);
   release_whole(&pipeline->alloc, &next);
   pan_graph_program_deps_finish(&deps);
   pan_graph_program_schedule_finish(&schedule);
   panvk_data_graph_matmul_add_finish(&graph);
   return result;
}

static VkResult
create_neural_pass(struct panvk_device *dev,
                   struct panvk_data_graph_pipeline *pipeline, unsigned pass)
{
   const struct panvk_data_graph *graph = pipeline->graph;
   const struct panvk_graph_value *value = &graph->values[graph->passes[pass]];
   const struct panvk_graph_value *operation = value;
   if (value->kind == PANVK_GRAPH_OUTPUT)
      operation = &graph->values[value->inputs[0]];
   const struct panvk_graph_value *rescale = NULL;
   const struct panvk_graph_value *bias = NULL;
   const struct panvk_graph_value *table = NULL;
   if (operation->conv_rescale && operation->op == PANVK_GRAPH_OP_TABLE) {
      table = &graph->values[operation->inputs[1]];
      operation = &graph->values[operation->inputs[0]];
   }
   if (operation->conv_rescale) {
      rescale = operation;
      const struct panvk_graph_value *add = &graph->values[rescale->inputs[0]];
      bias = &graph->values[add->inputs[1]];
      operation = rescale->conv_rescale;
   }
   bool supported = false;
#ifdef HAVE_PAN_KMOD_KBASE
   supported = to_panvk_physical_device(dev->vk.physical)->kbase_node_path[0] &&
               kbase_kmod_csf_has_neural(dev->kmod.dev);
#endif
   if ((!operation->matmul && !operation->conv) || operation->kind != PANVK_GRAPH_OPERATION ||
       !supported)
      return VK_SUCCESS;

   const struct pan_graph_nrt_target target = pan_graph_nrt_target_v15();
   struct pan_graph_matmul_kernel kernel = {0};
   struct pan_graph_matmul_cost_target cost_target = {0};
   uint32_t core_budget = 0;
#ifdef HAVE_PAN_KMOD_KBASE
   struct kbase_kmod_l2_cache_props cache;
   if (kbase_kmod_get_l2_cache_props(dev->kmod.dev, &cache) &&
       pan_graph_matmul_cost_target_v15_init(
          cache.slices, cache.line_size, cache.size_per_slice, &cost_target))
      core_budget = kbase_kmod_csf_neural_core_count(dev->kmod.dev);
#endif
   if (!core_budget)
      return VK_SUCCESS;
   enum pan_graph_tsu_status status = PAN_GRAPH_TSU_UNSUPPORTED;
   if (operation->conv) {
      const struct pan_graph_conv_plan *plan = operation->conv;
      for (unsigned axis = 0; axis < 2; axis++) {
         if (plan->info.kernel[axis] > 1 && plan->info.dilation[axis] > 1)
            return VK_SUCCESS;
      }
      const struct pan_graph_matmul_memory memory = {
         .inputs = {{.level = 5, .override = 1, .value = 1},
                    {.offset = 2, .level = 5, .override = 1, .value = 1}},
         .output = {.domain = 3, .level = 5},
      };
      struct pan_graph_conv_rescale epilogue = {0};
      if (rescale) {
         epilogue.parameters.clamp = true;
         epilogue.parameters.minimum = -128;
         epilogue.parameters.maximum = 127;
         epilogue.parameters.mode = !(rescale->attribute & 1);
         epilogue.parameters.output_zero_point =
            *(int8_t *)graph->values[rescale->inputs[4]].data;
         epilogue.coefficients = (struct pan_graph_v15_memory){
            .domain = 2, .level = 5, .override = 1, .value = 1,
         };
         epilogue.has_table = table != NULL;
         epilogue.table = (struct pan_graph_v15_memory){
            .domain = 2, .offset = 2, .level = 5, .override = 1, .value = 1,
         };
      }
      uint64_t best_volume = 0;
      uint64_t best_cycles = UINT64_MAX;
      const uint32_t probe[7] = {1,
                                 1,
                                 1,
                                 1,
                                 plan->info.input_channels,
                                 plan->info.kernel[0],
                                 plan->info.kernel[1]};
      const uint32_t no_loops[4] = {0};
      uint64_t probe_cycles;
      bool costed =
         pan_graph_conv_v15_ce_cycles(plan, probe, no_loops, &probe_cycles);
      for (unsigned ci = plan->info.input_channels; ci;) {
         for (unsigned co = 4; co <= 32 && co <= plan->info.output_channels;
              co *= 2) {
            for (unsigned h = 1;
                 h <= (costed ? 16 : 8) && h <= plan->output_height; h *= 2) {
               for (unsigned w = 4;
                    w <= (costed ? 64 : 32) && w <= plan->output_width;
                    w *= 2) {
                  uint32_t transfer[7] = {co,
                                          1,
                                          h,
                                          w,
                                          ci,
                                          plan->info.kernel[0],
                                          plan->info.kernel[1]};
                  uint32_t loops[4];
                  if (!pan_graph_conv_v15_assign_loops(plan, transfer,
                                                       core_budget, loops))
                     continue;
                  struct pan_graph_matmul_kernel candidate = {0};
                  enum pan_graph_tsu_status result =
                     pan_graph_conv_v15_compile_rescale_candidate(
                        plan, &target, transfer, loops, &memory,
                        rescale ? &epilogue : NULL, &candidate);
                  if (result == PAN_GRAPH_TSU_OUT_OF_MEMORY) {
                     pan_graph_v15_program_cleanup(&kernel.program);
                     return VK_ERROR_OUT_OF_HOST_MEMORY;
                  }
                  if (result != PAN_GRAPH_TSU_SUCCESS)
                     continue;
                  uint64_t volume = (uint64_t)co * h * w;
                  uint64_t cycles = 0;
                  if (costed && !pan_graph_conv_v15_ce_cycles(plan, transfer,
                                                              loops, &cycles)) {
                     pan_graph_v15_program_cleanup(&candidate.program);
                     continue;
                  }
                  if (cycles < best_cycles ||
                      (cycles == best_cycles && volume > best_volume)) {
                     pan_graph_v15_program_cleanup(&kernel.program);
                     kernel = candidate;
                     best_volume = volume;
                     best_cycles = cycles;
                     status = PAN_GRAPH_TSU_SUCCESS;
                  } else {
                     pan_graph_v15_program_cleanup(&candidate.program);
                  }
               }
            }
         }
         unsigned next = 8;
         while (next < ci && next <= (ci - 1) / 2)
            next *= 2;
         ci = costed && next < ci ? next : 0;
      }
   } else {
      struct pan_graph_matmul_selection selected;
      status =
         pan_graph_matmul_select_v15(&operation->matmul->info, &target,
                                     &cost_target, core_budget, &selected);
      if (status == PAN_GRAPH_TSU_SUCCESS)
         status = pan_graph_matmul_v15_compile_candidate(
            operation->matmul, &target, selected.candidate.transfer,
            selected.loops, &kernel);
   }
   if (status == PAN_GRAPH_TSU_OUT_OF_MEMORY)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   if (status != PAN_GRAPH_TSU_SUCCESS)
      return VK_SUCCESS;
   if (!pipeline->neural) {
      pipeline->neural = vk_zalloc(
         &pipeline->alloc, graph->pass_count * sizeof(*pipeline->neural), 8,
         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!pipeline->neural) {
         pan_graph_v15_program_cleanup(&kernel.program);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
   }
   struct panvk_data_graph_neural *neural = &pipeline->neural[pass];
   size_t coefficient_offset = ALIGN_POT(kernel.program.size, 64);
   size_t coefficient_size = rescale ? operation->conv->info.output_channels * 10 : 0;
   size_t table_offset = ALIGN_POT(coefficient_offset + coefficient_size, 64);
   size_t code_size = table ? table_offset + 256 : coefficient_offset + coefficient_size;
   VkResult result =
      panvk_priv_bo_create(dev, code_size, 0,
                           VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, &neural->code);
   if (result == VK_SUCCESS) {
      memcpy(neural->code->addr.host, kernel.program.data, kernel.program.size);
      if (rescale) {
         const int32_t *multipliers = graph->values[rescale->inputs[1]].data;
         const int8_t *shifts = graph->values[rescale->inputs[2]].data;
         const int32_t *biases = bias->data;
         uint8_t *coefficients = (uint8_t *)neural->code->addr.host + coefficient_offset;
         neural->coefficient_channels = operation->conv->info.output_channels;
         neural->coefficients = neural->code->addr.dev + coefficient_offset;
         for (unsigned i = 0; i < neural->coefficient_channels; i++) {
            unsigned channel = rescale->attribute & 2 ? i : 0;
            int16_t shift = shifts[channel];
            memcpy(coefficients + 10 * i, &shift, 2);
            memcpy(coefficients + 10 * i + 2, biases + (bias->elements > 1 ? i : 0), 4);
            memcpy(coefficients + 10 * i + 6, multipliers + channel, 4);
         }
      }
      if (table) {
         memcpy((uint8_t *)neural->code->addr.host + table_offset, table->data, 256);
         neural->table = neural->code->addr.dev + table_offset;
      }
      panvk_priv_bo_flush(neural->code, 0, code_size);
      memcpy(neural->inputs, operation->inputs, sizeof(neural->inputs));
      if (operation->conv && operation->conv->info.input_nearest_2x)
         neural->inputs[0] = graph->values[operation->inputs[0]].inputs[0];
      memcpy(neural->dispatch.extent, kernel.extent, sizeof(kernel.extent));
      memcpy(neural->dispatch.loops, kernel.loops, sizeof(kernel.loops));
      neural->dispatch.address = neural->code->addr.dev;
      neural->dispatch.size = kernel.program.size;
      neural->dispatch.dispatch_id = pass;
      pipeline->neural_count++;
   }
   pan_graph_v15_program_cleanup(&kernel.program);
   return result;
}
#endif

static VkResult
create_pipeline(struct panvk_device *dev, VkPipelineCache cache,
                const VkDataGraphPipelineCreateInfoARM *info,
                const VkAllocationCallbacks *alloc, VkPipeline *handle)
{
   const VkDataGraphPipelineShaderModuleCreateInfoARM *module =
      vk_find_struct_const(info->pNext,
                           DATA_GRAPH_PIPELINE_SHADER_MODULE_CREATE_INFO_ARM);
   if (!module || (info->flags &
                   VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT))
      return VK_PIPELINE_COMPILE_REQUIRED;
   const uint32_t *code;
   size_t size;
   if (module->module) {
      VK_FROM_HANDLE(vk_shader_module, shader, module->module);
      code = (const void *)shader->data;
      size = shader->size;
   } else {
      const VkShaderModuleCreateInfo *shader =
         vk_find_struct_const(module->pNext, SHADER_MODULE_CREATE_INFO);
      if (!shader)
         shader = vk_find_struct_const(info->pNext, SHADER_MODULE_CREATE_INFO);
      if (!shader)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      code = shader->pCode;
      size = shader->codeSize;
   }
   if (size < 20 || size % sizeof(uint32_t))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   bool specialize = module->pSpecializationInfo != NULL;
   for (size_t i = 5; i < size / 4;) {
      unsigned n = code[i] >> 16, op = code[i] & 0xffff;
      if (!n || n > size / 4 - i)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      specialize |= (op >= SpvOpSpecConstantTrue && op <= SpvOpSpecConstantOp) ||
                    op == SpvOpSpecConstantCompositeReplicateEXT;
      i += n;
   }
   uint32_t *specialized = NULL;
   VkDataGraphPipelineShaderModuleCreateInfoARM resolved_module = *module;
   if (specialize) {
      if (!dev->vk.enabled_features.dataGraphSpecializationConstants)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      struct nir_spirv_specialization *spec =
         vk_spec_info_to_nir_spirv(module->pSpecializationInfo);
      if (module->pSpecializationInfo &&
          module->pSpecializationInfo->mapEntryCount && !spec)
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      size_t word_count;
      specialized = spirv_specialize(NULL, code, size / 4, spec, &word_count);
      vtn_free_specialization(spec);
      if (!specialized)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      code = specialized;
      size = word_count * sizeof(uint32_t);
      resolved_module.pSpecializationInfo = NULL;
   }
   struct panvk_data_graph *graph;
   VkResult result = panvk_data_graph_parse(NULL, code, size, info,
                                            &resolved_module, &graph);
   ralloc_free(specialized);
   if (result != VK_SUCCESS)
      return panvk_errorf(dev, result,
                          "Unsupported data graph operation or tensor shape");
   VK_FROM_HANDLE(vk_pipeline_layout, user_layout, info->layout);
   for (unsigned i = 0; i < graph->count; i++) {
      struct panvk_graph_value *v = &graph->values[i];
      if (v->kind != PANVK_GRAPH_INPUT && v->kind != PANVK_GRAPH_OUTPUT)
         continue;
      if (v->set >= user_layout->set_count || v->set >= MAX_SETS ||
          !user_layout->set_layouts[v->set]) {
         ralloc_free(graph);
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
      const struct panvk_descriptor_set_layout *sl =
         to_panvk_descriptor_set_layout(user_layout->set_layouts[v->set]);
      if (v->binding >= sl->binding_count ||
          sl->bindings[v->binding].type != VK_DESCRIPTOR_TYPE_TENSOR_ARM ||
          v->array_element >= sl->bindings[v->binding].desc_count) {
         ralloc_free(graph);
         return VK_ERROR_FEATURE_NOT_PRESENT;
      }
      v->descriptor_offset = panvk_get_desc_index(&sl->bindings[v->binding],
                                                  v->array_element, NO_SUBDESC);
   }
   struct panvk_data_graph_pipeline *p = vk_pipeline_zalloc(
      &dev->vk, &graph_ops, VK_PIPELINE_BIND_POINT_DATA_GRAPH_ARM, info->flags,
      alloc, sizeof(*p));
   if (!p) {
      ralloc_free(graph);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   p->alloc = alloc ? *alloc : dev->vk.alloc;
   p->refs = 1;
   p->graph = graph;
   p->transient_size = graph->transient_size;
   p->dispatch_count = graph->pass_count;
   p->passes = vk_zalloc(&p->alloc, graph->pass_count * sizeof(*p->passes), 8,
                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!p->passes) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail;
   }
   VkDevice device = panvk_device_to_handle(dev);
   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_TENSOR_ARM,
      .descriptorCount = graph->count,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo li = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   result = dev->vk.dispatch_table.CreateDescriptorSetLayout(
      device, &li, &p->alloc, &p->set_layout);
   if (result != VK_SUCCESS)
      goto fail;
   const VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &p->set_layout,
   };
   result = dev->vk.dispatch_table.CreatePipelineLayout(device, &pli, &p->alloc,
                                                        &p->layout);
   if (result != VK_SUCCESS)
      goto fail;
   if (graph->constant_size) {
      result =
         panvk_priv_bo_create(dev, graph->constant_size, 0,
                              VK_SYSTEM_ALLOCATION_SCOPE_OBJECT, &p->constants);
      if (result != VK_SUCCESS)
         goto fail;
      for (unsigned i = 0; i < graph->count; i++) {
         const struct panvk_graph_value *v = &graph->values[i];
         if (v->live && v->kind == PANVK_GRAPH_CONSTANT && v->elements > 1)
            memcpy((char *)p->constants->addr.host + v->offset, v->data,
                   v->size);
      }
      panvk_priv_bo_flush(p->constants, 0, graph->constant_size);
   }
#if PAN_ARCH >= 15
   result = create_whole_neural(dev, p);
   if (result != VK_SUCCESS)
      goto fail;
#endif
   for (unsigned i = 0; !p->whole_neural && i < graph->pass_count; i++) {
#if PAN_ARCH >= 15
      result = create_neural_pass(dev, p, i);
      if (result != VK_SUCCESS)
         goto fail;
      if (p->neural && p->neural[i].code)
         continue;
#endif
      nir_shader *nir = panvk_data_graph_shader(graph, i);
      if (!nir) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail;
      }
      const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
         .nir = nir,
      };
      const VkComputePipelineCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .layout = p->layout,
         .stage =
            {
               .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               .pNext = &nir_info,
               .stage = VK_SHADER_STAGE_COMPUTE_BIT,
               .pName = "main",
            },
      };
      result = dev->vk.dispatch_table.CreateComputePipelines(
         device, cache, 1, &ci, &p->alloc, &p->passes[i]);
      ralloc_free(nir);
      if (result != VK_SUCCESS)
         goto fail;
   }
#if PAN_ARCH >= 15
   if (p->neural_count) {
      static int next_pipeline_id;
      uint32_t id = (p_atomic_inc_return(&next_pipeline_id) - 1) & 255;
      if (p->whole_neural) {
         for (uint32_t i = 0; i < p->whole.count; i++)
            p->whole.programs[i].dispatch.pipeline_id = id;
      } else {
         for (unsigned i = 0; i < graph->pass_count; i++)
            p->neural[i].dispatch.pipeline_id = id;
      }
   }
#endif
   *handle = panvk_data_graph_pipeline_to_handle(p);
   if (PANVK_DEBUG(STARTUP)) {
      uint32_t neural = 0;
#if PAN_ARCH >= 15
      neural = p->neural_count;
#endif
      mesa_logi("data graph: EE=%u NE=%u transient=%" PRIu64 " constants=%" PRIu64,
                p->dispatch_count - neural, neural, p->transient_size,
                graph->constant_size);
   }
   return VK_SUCCESS;
fail:
   release_pipeline(&dev->vk, p);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CreateDataGraphPipelinesARM)(
   VkDevice device, VkDeferredOperationKHR deferred, VkPipelineCache cache,
   uint32_t count, const VkDataGraphPipelineCreateInfoARM *infos,
   const VkAllocationCallbacks *alloc, VkPipeline *pipelines)
{
   VK_FROM_HANDLE(panvk_device, dev, device);
   VkResult result = VK_SUCCESS;
   memset(pipelines, 0, count * sizeof(*pipelines));
   if (deferred)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   for (unsigned i = 0; i < count; i++) {
      VkResult r = create_pipeline(dev, cache, &infos[i], alloc, &pipelines[i]);
      if (r == VK_SUCCESS)
         continue;
      if (result >= 0 || r < 0)
         result = r;
      if (infos[i].flags & VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT)
         break;
   }
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(CreateDataGraphPipelineSessionARM)(
   VkDevice device, const VkDataGraphPipelineSessionCreateInfoARM *info,
   const VkAllocationCallbacks *alloc, VkDataGraphPipelineSessionARM *handle)
{
   VK_FROM_HANDLE(panvk_device, dev, device);
   VK_FROM_HANDLE(panvk_data_graph_pipeline, pipeline, info->dataGraphPipeline);
   *handle = VK_NULL_HANDLE;
   if (info->flags)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct panvk_data_graph_session *session =
      vk_object_zalloc(&dev->vk, alloc, sizeof(*session),
                       VK_OBJECT_TYPE_DATA_GRAPH_PIPELINE_SESSION_ARM);
   if (!session)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   p_atomic_inc(&pipeline->refs);
   session->pipeline = pipeline;
   *handle = panvk_data_graph_session_to_handle(session);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(DestroyDataGraphPipelineSessionARM)(
   VkDevice device, VkDataGraphPipelineSessionARM handle,
   const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(panvk_device, dev, device);
   VK_FROM_HANDLE(panvk_data_graph_session, session, handle);
   if (!session)
      return;
   release_pipeline(&dev->vk, session->pipeline);
   vk_object_free(&dev->vk, alloc, session);
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(GetDataGraphPipelineSessionBindPointRequirementsARM)(
   VkDevice device,
   const VkDataGraphPipelineSessionBindPointRequirementsInfoARM *info,
   uint32_t *count, VkDataGraphPipelineSessionBindPointRequirementARM *reqs)
{
   VK_FROM_HANDLE(panvk_data_graph_session, session, info->session);
   VK_OUTARRAY_MAKE_TYPED(VkDataGraphPipelineSessionBindPointRequirementARM,
                          out, reqs, count);
   if (session->pipeline->transient_size) {
      vk_outarray_append_typed(
         VkDataGraphPipelineSessionBindPointRequirementARM, &out, r)
      {
         r->bindPoint = VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM;
         r->bindPointType =
            VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM;
         r->numObjects = 1;
      }
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(GetDataGraphPipelineSessionMemoryRequirementsARM)(
   VkDevice device,
   const VkDataGraphPipelineSessionMemoryRequirementsInfoARM *info,
   VkMemoryRequirements2 *req)
{
   VK_FROM_HANDLE(panvk_device, dev, device);
   VK_FROM_HANDLE(panvk_data_graph_session, session, info->session);
   const struct panvk_physical_device *pdev =
      to_panvk_physical_device(dev->vk.physical);
   req->memoryRequirements = (VkMemoryRequirements){
      .size = session->pipeline->transient_size,
      .alignment = 64,
      .memoryTypeBits = BITFIELD_MASK(pdev->memory.type_count),
   };
   VkMemoryDedicatedRequirements *ded =
      vk_find_struct(req->pNext, MEMORY_DEDICATED_REQUIREMENTS);
   if (ded) {
      ded->requiresDedicatedAllocation = false;
      ded->prefersDedicatedAllocation = false;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(BindDataGraphPipelineSessionMemoryARM)(
   VkDevice device, uint32_t count,
   const VkBindDataGraphPipelineSessionMemoryInfoARM *infos)
{
   for (unsigned i = 0; i < count; i++) {
      const VkBindDataGraphPipelineSessionMemoryInfoARM *info = &infos[i];
      VK_FROM_HANDLE(panvk_data_graph_session, session, info->session);
      VK_FROM_HANDLE(panvk_device_memory, mem, info->memory);
      if (info->bindPoint !=
             VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM ||
          info->objectIndex || session->transient || !mem ||
          info->memoryOffset % 64 || info->memoryOffset > mem->vk.size ||
          session->pipeline->transient_size > mem->vk.size - info->memoryOffset)
         return VK_ERROR_UNKNOWN;
   }
   for (unsigned i = 0; i < count; i++) {
      VK_FROM_HANDLE(panvk_data_graph_session, session, infos[i].session);
      VK_FROM_HANDLE(panvk_device_memory, mem, infos[i].memory);
      session->transient = mem->addr.dev + infos[i].memoryOffset;
   }
   return VK_SUCCESS;
}

static void
pack_tensor(struct mali_tensor_packed *desc, const struct panvk_graph_value *v,
            uint64_t address)
{
   unsigned pad = 4 - v->rank;
   uint32_t dims[4] = {1, 1, 1, 1}, strides[4];
   uint32_t stride = vk_format_get_blocksize(v->format);
   unsigned wide = 0;
   for (int d = 3; d >= 0; d--) {
      if (d >= (int)pad)
         dims[d] = v->dims[d - pad];
      if (dims[d] > UINT16_MAX)
         wide = d;
      strides[d] = stride;
      stride *= v->storage ? v->storage->dims[d] : dims[d];
   }
   bool brick = v->storage || v->alias_concat;
   if (brick) {
      const struct panvk_graph_value *owner = v->storage ? v->storage : v;
      strides[2] = owner->dims[3] * 4;
      strides[1] = DIV_ROUND_UP(owner->dims[2], 4) * strides[2];
      strides[0] = owner->dims[1] * strides[1];
   }
   pan_pack(desc, TENSOR, cfg) {
      cfg.element_format =
         v->format == VK_FORMAT_R8_SINT      ? MALI_TENSOR_ELEMENT_FORMAT_S8
         : v->format == VK_FORMAT_R32_SFLOAT ? MALI_TENSOR_ELEMENT_FORMAT_F32
         : v->format == VK_FORMAT_R32_SINT   ? MALI_TENSOR_ELEMENT_FORMAT_S32
                                             : MALI_TENSOR_ELEMENT_FORMAT_U32;
      cfg.tiling = brick ? MALI_TENSOR_TILING_BRICK_4 : MALI_TENSOR_TILING_LINEAR;
      cfg.wide_dimension = wide;
      cfg.wide_dimension_high = dims[wide] >> 16;
      cfg.dimension_0 = dims[0] & UINT16_MAX;
      cfg.dimension_1 = dims[1] & UINT16_MAX;
      cfg.dimension_2 = dims[2] & UINT16_MAX;
      cfg.dimension_3 = dims[3] & UINT16_MAX;
      cfg.stride_0 = strides[0];
      cfg.stride_1 = strides[1];
      cfg.stride_2 = strides[2];
      cfg.address = address;
   }
}

static void
graph_dispatch_barrier(struct panvk_cmd_buffer *cmdbuf)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
      .srcAccessMask =
         VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_DATA_GRAPH_BIT_ARM,
      .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                       VK_ACCESS_2_SHADER_WRITE_BIT |
                       VK_ACCESS_2_DATA_GRAPH_READ_BIT_ARM |
                       VK_ACCESS_2_DATA_GRAPH_WRITE_BIT_ARM,
   };
   const VkDependencyInfo dependency = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
   };
   dev->vk.dispatch_table.CmdPipelineBarrier2(
      panvk_cmd_buffer_to_handle(cmdbuf), &dependency);
}

#if PAN_ARCH >= 15
static void
dispatch_neural_program(struct panvk_cmd_buffer *cmdbuf,
                        const struct panvk_neural_program *program,
                        const uint64_t address[4], const uint32_t size[4])
{
   panvk_per_arch(cmd_pilot_close_all)(cmdbuf);
   struct cs_builder *b = panvk_get_cs_builder(cmdbuf, PANVK_SUBQUEUE_COMPUTE);
   cs_next_iter_sb(cmdbuf, PANVK_SUBQUEUE_COMPUTE,
                   cs_scratch_reg_tuple(b, 0, 2));
   cs_update_compute_ctx(b) {
      panvk_cs_neural_resources(b, address, size);
      panvk_cs_neural_program(b, program);
   }
   cs_run_neural(b, 0, 16, 36);
   compute_state_set_dirty(cmdbuf, CS);
   compute_state_set_dirty(cmdbuf, DESC_STATE);
   compute_state_set_dirty(cmdbuf, PUSH_UNIFORMS);
}

static bool
dispatch_neural(struct panvk_cmd_buffer *cmdbuf,
                const struct panvk_data_graph_neural *neural,
                const struct mali_tensor_packed *descriptors, unsigned output)
{
   struct pan_ptr table =
      panvk_cmd_alloc_dev_mem(cmdbuf, desc, (3 + !!neural->coefficients + !!neural->table) * PANVK_TENSOR_DESCRIPTOR_SIZE,
                              PANVK_TENSOR_DESCRIPTOR_SIZE);
   if (!table.gpu) {
      vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   struct mali_tensor_packed *bindings = table.cpu;
   bindings[0] = descriptors[neural->inputs[0]];
   bindings[1] = descriptors[neural->inputs[1]];
   unsigned output_index = 2;
   if (neural->coefficients) {
      pan_pack(&bindings[2], TENSOR, cfg) {
         cfg.element_format = MALI_TENSOR_ELEMENT_FORMAT_RESCALE_COEFFICIENTS;
         cfg.tiling = MALI_TENSOR_TILING_LINEAR;
         cfg.dimension_0 = cfg.dimension_1 = cfg.dimension_2 = 1;
         cfg.dimension_3 = neural->coefficient_channels;
         cfg.stride_0 = cfg.stride_1 = cfg.stride_2 = neural->coefficient_channels * 10;
         cfg.address = neural->coefficients;
      }
      output_index++;
   }
   if (neural->table) {
      pan_pack(&bindings[output_index], TENSOR, cfg) {
         cfg.element_format = MALI_TENSOR_ELEMENT_FORMAT_S8;
         cfg.tiling = MALI_TENSOR_TILING_LINEAR;
         cfg.dimension_0 = cfg.dimension_1 = cfg.dimension_2 = 1;
         cfg.dimension_3 = 256;
         cfg.stride_0 = cfg.stride_1 = cfg.stride_2 = 256;
         cfg.address = neural->table;
      }
      output_index++;
   }
   bindings[output_index] = descriptors[output];
   const uint64_t address[4] = {
      table.gpu, 0, neural->coefficients ? table.gpu + 128 : 0,
      table.gpu + output_index * PANVK_TENSOR_DESCRIPTOR_SIZE,
   };
   const uint32_t size[4] = {128, 0, 0, 64};
   uint32_t resource_size[4];
   memcpy(resource_size, size, sizeof(size));
   resource_size[2] = neural->coefficients ? (1 + !!neural->table) * 64 : 0;
   dispatch_neural_program(cmdbuf, &neural->dispatch, address, resource_size);
   return true;
}

static void
dispatch_whole_neural(struct panvk_cmd_buffer *cmdbuf,
                      const struct panvk_data_graph_pipeline *pipeline,
                      uint64_t transient)
{
   const struct panvk_data_graph_whole *whole = &pipeline->whole;
   const struct pan_graph_memory_resources *resources =
      &whole->memory.resources;
   uint32_t size[4];
   uint64_t address[4] = {0}, total = 0;
   for (unsigned d = 0; d < 4; d++) {
      if (resources->domains[d].count >
          UINT32_MAX / PANVK_TENSOR_DESCRIPTOR_SIZE)
         goto allocation_failed;
      size[d] = resources->domains[d].count * PANVK_TENSOR_DESCRIPTOR_SIZE;
      total += size[d];
   }
   if (!total || total > UINT32_MAX)
      goto allocation_failed;
   struct pan_ptr table = panvk_cmd_alloc_dev_mem(cmdbuf, desc, total,
                                                  PANVK_TENSOR_DESCRIPTOR_SIZE);
   if (!table.gpu)
      goto allocation_failed;
   uint32_t offset = 0;
   for (unsigned d = 0; d < 4; d++) {
      if (size[d])
         address[d] = table.gpu + offset;
      for (uint32_t i = 0; i < resources->domains[d].count; i++) {
         const struct pan_graph_memory_resource_entry *entry =
            resources->domains[d].entries + i;
         struct mali_tensor_packed *descriptor =
            (void *)((char *)table.cpu + offset +
                     i * PANVK_TENSOR_DESCRIPTOR_SIZE);
         memcpy(descriptor, entry->descriptor.words,
                PANVK_TENSOR_DESCRIPTOR_SIZE);
         if (d == PAN_GRAPH_MEMORY_RESOURCE_TRANSIENT) {
            uint64_t relative = (uint64_t)entry->descriptor.words[6] |
                                (uint64_t)entry->descriptor.words[7] << 32;
            if (transient > UINT64_MAX - relative)
               goto invalid;
            uint64_t absolute = transient + relative;
            uint32_t words[2] = {absolute, absolute >> 32};
            memcpy((char *)descriptor + PANVK_TENSOR_DESC_ADDRESS_OFFSET, words,
                   sizeof(words));
         } else if (d == PAN_GRAPH_MEMORY_RESOURCE_INPUT ||
                    d == PAN_GRAPH_MEMORY_RESOURCE_OUTPUT) {
            if ((d == PAN_GRAPH_MEMORY_RESOURCE_INPUT && i >= 2) ||
                (d == PAN_GRAPH_MEMORY_RESOURCE_OUTPUT && i))
               goto invalid;
            uint32_t id = d == PAN_GRAPH_MEMORY_RESOURCE_INPUT
                             ? whole->inputs[i]
                             : whole->output;
            const struct panvk_graph_value *v = pipeline->graph->values + id;
            if (v->set >= MAX_SETS)
               goto invalid;
            const struct panvk_descriptor_set *set =
               cmdbuf->state.data_graph.desc_state.sets[v->set];
            if (!set || !set->descs.host ||
                v->descriptor_offset > set->desc_count ||
                set->desc_count - v->descriptor_offset < 2)
               goto invalid;
            panvk_tensor_desc_patch(
               descriptor, (char *)set->descs.host +
                              v->descriptor_offset * PANVK_DESCRIPTOR_SIZE);
         } else {
            goto invalid;
         }
      }
      offset += size[d];
   }
   struct panvk_cmd_meta_compute_save_ctx save = {0};
   panvk_per_arch(cmd_meta_compute_start)(cmdbuf, &save);
   if (whole->dependency_sync) {
      struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
      panvk_per_arch(cmd_pilot_close_all)(cmdbuf);
      struct cs_builder *b = panvk_get_cs_builder(cmdbuf, PANVK_SUBQUEUE_COMPUTE);
      cs_wait_slots(b, dev->csf.sb.all_iters_mask | SB_MASK(LS) |
                          SB_MASK(DEFERRED_FLUSH));
      cs_update_compute_ctx(b) {
         panvk_cs_neural_resources(b, address, size);
         for (uint32_t i = 0; i < whole->count; i++) {
            panvk_cs_neural_program_deps(b, &whole->programs[i].dispatch,
                                         SB_ITER(i % dev->csf.sb.iter_count),
                                         whole->programs[i].wait_mask);
         }
      }
      cs_wait_slots(b, dev->csf.sb.all_iters_mask);
      cs_next_iter_sb(cmdbuf, PANVK_SUBQUEUE_COMPUTE,
                      cs_scratch_reg_tuple(b, 0, 2));
   } else {
      for (uint32_t i = 0; i < whole->count; i++) {
         if (i)
            graph_dispatch_barrier(cmdbuf);
         dispatch_neural_program(cmdbuf, &whole->programs[i].dispatch, address,
                                 size);
      }
   }
   panvk_per_arch(cmd_meta_compute_end)(cmdbuf, &save);
   compute_state_set_dirty(cmdbuf, DESC_STATE);
   compute_state_set_dirty(cmdbuf, PUSH_UNIFORMS);
   return;
allocation_failed:
   vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return;
invalid:
   vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_UNKNOWN);
}
#endif

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDispatchDataGraphARM)(
   VkCommandBuffer cmd, VkDataGraphPipelineSessionARM handle,
   const VkDataGraphPipelineDispatchInfoARM *info)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, cmd);
   VK_FROM_HANDLE(panvk_data_graph_session, session, handle);
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   struct panvk_data_graph_pipeline *p = session->pipeline;
   const struct panvk_data_graph *g = p->graph;
   if (cmdbuf->state.data_graph.pipeline != &p->vk ||
       (p->transient_size && !session->transient)) {
      vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_UNKNOWN);
      return;
   }
#if PAN_ARCH >= 15
   if (p->whole_neural) {
      dispatch_whole_neural(cmdbuf, p, session->transient);
      return;
   }
#endif
   struct pan_ptr table = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, g->count * PANVK_TENSOR_DESCRIPTOR_SIZE,
      PANVK_TENSOR_DESCRIPTOR_SIZE);
   if (!table.gpu) {
      vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   struct mali_tensor_packed *descs = table.cpu;
   for (unsigned i = 0; i < g->count; i++) {
      const struct panvk_graph_value *v = &g->values[i];
      if (!v->live || (v->kind == PANVK_GRAPH_CONSTANT && v->elements == 1) ||
          v->fused_into != UINT32_MAX) {
         memset(&descs[i], 0, sizeof(descs[i]));
         continue;
      }
      if (v->kind == PANVK_GRAPH_INPUT || v->kind == PANVK_GRAPH_OUTPUT) {
         if (v->set >= MAX_SETS)
            goto invalid;
         const struct panvk_descriptor_set *set =
            cmdbuf->state.data_graph.desc_state.sets[v->set];
         if (!set || !set->descs.host ||
             v->descriptor_offset > set->desc_count ||
             set->desc_count - v->descriptor_offset < 2)
            goto invalid;
         pack_tensor(&descs[i], v, 0);
         panvk_tensor_desc_patch(
            &descs[i], (char *)set->descs.host +
                          v->descriptor_offset * PANVK_DESCRIPTOR_SIZE);
      } else {
         uint64_t base = v->kind == PANVK_GRAPH_CONSTANT
                            ? p->constants->addr.dev
                            : session->transient;
         pack_tensor(&descs[i], v, base + v->offset);
      }
   }
   struct panvk_descriptor_set private_set = {
      .layout = panvk_descriptor_set_layout_from_handle(p->set_layout),
      .descs = {.dev = table.gpu, .host = table.cpu},
      .desc_count = g->count * 2,
   };
   struct panvk_cmd_meta_compute_save_ctx save = {0};
   panvk_per_arch(cmd_meta_compute_start)(cmdbuf, &save);
   struct panvk_descriptor_state saved_desc = cmdbuf->state.compute.desc_state;
   cmdbuf->state.compute.desc_state =
      (struct panvk_descriptor_state){.sets[0] = &private_set};
   compute_state_set_dirty(cmdbuf, DESC_STATE);
   for (unsigned i = 0; i < p->dispatch_count; i++) {
      if (i)
         graph_dispatch_barrier(cmdbuf);
#if PAN_ARCH >= 15
      if (p->neural && p->neural[i].code) {
         if (!dispatch_neural(cmdbuf, &p->neural[i], descs, g->passes[i]))
            break;
         continue;
      }
#endif
      dev->vk.dispatch_table.CmdBindPipeline(
         cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->passes[i]);
      uint32_t groups = DIV_ROUND_UP(g->values[g->passes[i]].elements, 64);
      dev->vk.dispatch_table.CmdDispatch(cmd, MIN2(groups, 65535),
                                         DIV_ROUND_UP(groups, 65535), 1);
   }
   cmdbuf->state.compute.desc_state = saved_desc;
   panvk_per_arch(cmd_meta_compute_end)(cmdbuf, &save);
   compute_state_set_dirty(cmdbuf, DESC_STATE);
   return;
invalid:
   vk_command_buffer_set_error(&cmdbuf->vk, VK_ERROR_UNKNOWN);
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(GetDataGraphPipelineAvailablePropertiesARM)(
   VkDevice device, const VkDataGraphPipelineInfoARM *info, uint32_t *count,
   VkDataGraphPipelinePropertyARM *props)
{
   VK_OUTARRAY_MAKE_TYPED(VkDataGraphPipelinePropertyARM, out, props, count);
   vk_outarray_append_typed(VkDataGraphPipelinePropertyARM, &out, p)
   {
      *p = VK_DATA_GRAPH_PIPELINE_PROPERTY_CREATION_LOG_ARM;
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
panvk_per_arch(GetDataGraphPipelinePropertiesARM)(
   VkDevice device, const VkDataGraphPipelineInfoARM *info, uint32_t count,
   VkDataGraphPipelinePropertyQueryResultARM *props)
{
   VK_FROM_HANDLE(panvk_data_graph_pipeline, p, info->dataGraphPipeline);
   char log[256];
   uint32_t neural_count = 0;
#if PAN_ARCH >= 15
   neural_count = p->neural_count;
#endif
   int length = snprintf(
      log, sizeof(log),
      "EE NIR/Kraid: %u dispatches; NE: %u dispatches; transient=%" PRIu64
      " bytes; constants=%" PRIu64 " bytes",
      p->dispatch_count - neural_count, neural_count, p->transient_size,
      p->graph->constant_size);
   size_t size = length + 1;
   VkResult result = VK_SUCCESS;
   for (unsigned i = 0; i < count; i++) {
      if (props[i].property != VK_DATA_GRAPH_PIPELINE_PROPERTY_CREATION_LOG_ARM)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      props[i].isText = VK_TRUE;
      if (props[i].pData) {
         size_t n = MIN2(props[i].dataSize, size);
         memcpy(props[i].pData, log, n);
         if (n < size)
            result = VK_INCOMPLETE;
         props[i].dataSize = n;
      } else {
         props[i].dataSize = size;
      }
   }
   return result;
}
