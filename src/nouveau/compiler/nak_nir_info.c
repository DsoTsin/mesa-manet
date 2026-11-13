#include "nak_private.h"

void
nak_nir_get_shader_info(const nir_shader *nir,
                         struct nak_nir_shader_info *info)
{
   *info = (struct nak_nir_shader_info) {
      .stage = nir->info.stage,
      .workgroup_size = {
         nir->info.workgroup_size[0],
         nir->info.workgroup_size[1],
         nir->info.workgroup_size[2],
      },
      .clip_distance_array_size = nir->info.clip_distance_array_size,
      .cull_distance_array_size = nir->info.cull_distance_array_size,
      .shared_size = nir->info.shared_size,
      .task_payload_size = nir->info.task_payload_size,
      .float_controls_execution_mode = nir->info.float_controls_execution_mode,
   };

   switch (nir->info.stage) {
   case MESA_SHADER_GEOMETRY:
      info->gs = (struct nak_nir_gs_info) {
         .output_primitive = nir->info.gs.output_primitive,
         .input_primitive = nir->info.gs.input_primitive,
         .vertices_out = nir->info.gs.vertices_out,
         .invocations = nir->info.gs.invocations,
         .active_stream_mask = nir->info.gs.active_stream_mask,
      };
      break;
   case MESA_SHADER_FRAGMENT:
      info->fs = (struct nak_nir_fs_info) {
         .post_depth_coverage = nir->info.fs.post_depth_coverage,
         .early_fragment_tests = nir->info.fs.early_fragment_tests,
         .uses_sample_shading = nir->info.fs.uses_sample_shading,
      };
      break;
   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_TESS_EVAL:
      info->tess = (struct nak_nir_tess_info) {
         .primitive_mode = nir->info.tess._primitive_mode,
         .tcs_vertices_out = nir->info.tess.tcs_vertices_out,
         .spacing = nir->info.tess.spacing,
         .ccw = nir->info.tess.ccw,
         .point_mode = nir->info.tess.point_mode,
      };
      break;
   case MESA_SHADER_MESH:
      info->mesh = (struct nak_nir_mesh_info) {
         .primitive_type = nir->info.mesh.primitive_type,
         .max_vertices_out = nir->info.mesh.max_vertices_out,
         .max_primitives_out = nir->info.mesh.max_primitives_out,
      };
      break;
   default:
      break;
   }
}
