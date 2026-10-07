#include "pan_nir.h"

static bool
link_constant_varyings(nir_shader *producer, nir_shader *consumer)
{
   const nir_shader_compiler_options *original = consumer->options;
   nir_shader_compiler_options options = *original;
   options.max_varying_expression_cost = 0;
   consumer->options = &options;
   bool progress = nir_link_opt_varyings(producer, consumer);
   consumer->options = original;
   return progress;
}

static bool
cleanup_linked_shader(nir_shader *nir)
{
   bool progress = false;
   bool iteration;
   do {
      iteration = false;
      NIR_PASS(iteration, nir, nir_lower_global_vars_to_local);
      NIR_PASS(iteration, nir, nir_lower_vars_to_ssa);
      NIR_PASS(iteration, nir, nir_opt_copy_prop);
      NIR_PASS(iteration, nir, nir_opt_constant_folding);
      NIR_PASS(iteration, nir, nir_opt_dce);
      NIR_PASS(iteration, nir, nir_opt_dead_cf);
      NIR_PASS(iteration, nir, nir_opt_cse);
      NIR_PASS(iteration, nir, nir_opt_deref);
      NIR_PASS(iteration, nir, nir_remove_dead_variables,
               nir_var_shader_in | nir_var_shader_out | nir_var_function_temp,
               NULL);
      progress |= iteration;
   } while (iteration);
   return progress;
}

bool
pan_nir_link_varyings(nir_shader *producer, nir_shader *consumer)
{
   if (producer->info.stage != MESA_SHADER_VERTEX ||
       consumer->info.stage != MESA_SHADER_FRAGMENT || producer->xfb_info ||
       producer->info.has_transform_feedback_varyings)
      return false;

   nir_foreach_shader_out_variable(var, producer) {
      if (var->data.always_active_io)
         return false;
   }

   bool progress = false;
   NIR_PASS(progress, producer, nir_lower_var_copies);
   NIR_PASS(progress, consumer, nir_lower_var_copies);
   NIR_PASS(progress, producer, nir_lower_io_vars_to_scalar, nir_var_shader_out);
   NIR_PASS(progress, consumer, nir_lower_io_vars_to_scalar, nir_var_shader_in);

   bool iteration;
   do {
      progress |= cleanup_linked_shader(producer);
      progress |= cleanup_linked_shader(consumer);
      iteration = link_constant_varyings(producer, consumer);
      progress |= cleanup_linked_shader(consumer);
      iteration |= nir_remove_unused_varyings(producer, consumer);
      progress |= iteration;
   } while (iteration);

   progress |= cleanup_linked_shader(producer);
   progress |= cleanup_linked_shader(consumer);
   nir_shader_gather_info(producer, nir_shader_get_entrypoint(producer));
   nir_shader_gather_info(consumer, nir_shader_get_entrypoint(consumer));
   return progress;
}
