#include "nak_private.h"
#include <stddef.h>

size_t
nak_test_nir_info_layout(unsigned index)
{
   const size_t values[] = {
      sizeof(struct nak_nir_shader_info),
      _Alignof(struct nak_nir_shader_info),
      offsetof(struct nak_nir_shader_info, stage),
      offsetof(struct nak_nir_shader_info, workgroup_size),
      offsetof(struct nak_nir_shader_info, clip_distance_array_size),
      offsetof(struct nak_nir_shader_info, cull_distance_array_size),
      offsetof(struct nak_nir_shader_info, shared_size),
      offsetof(struct nak_nir_shader_info, task_payload_size),
      offsetof(struct nak_nir_shader_info, float_controls_execution_mode),
      offsetof(struct nak_nir_shader_info, gs),
      offsetof(struct nak_nir_shader_info, fs),
      offsetof(struct nak_nir_shader_info, tess),
      offsetof(struct nak_nir_shader_info, mesh),
      sizeof(struct nak_nir_gs_info),
      sizeof(struct nak_nir_fs_info),
      sizeof(struct nak_nir_tess_info),
      sizeof(struct nak_nir_mesh_info),
      offsetof(struct nak_nir_gs_info, active_stream_mask),
      offsetof(struct nak_nir_tess_info, spacing),
      offsetof(struct nak_nir_tess_info, ccw),
      offsetof(struct nak_nir_tess_info, point_mode),
      offsetof(struct nak_nir_mesh_info, max_vertices_out),
   };
   assert(index < ARRAY_SIZE(values));
   return values[index];
}

void
nak_test_nir_info(unsigned stage, unsigned variant,
                  struct nak_nir_shader_info *info)
{
   nir_shader nir = {0};
   nir.info.stage = stage;
   nir.info.workgroup_size[0] = 3;
   nir.info.workgroup_size[1] = 5;
   nir.info.workgroup_size[2] = 7;
   nir.info.clip_distance_array_size = 2;
   nir.info.cull_distance_array_size = 3;
   nir.info.shared_size = 1536;
   nir.info.task_payload_size = 320;
   nir.info.float_controls_execution_mode = 0x1234;

   switch (stage) {
   case MESA_SHADER_FRAGMENT:
      nir.info.fs.post_depth_coverage = variant & 1;
      nir.info.fs.early_fragment_tests = (variant >> 1) & 1;
      nir.info.fs.uses_sample_shading = (variant >> 2) & 1;
      break;
   case MESA_SHADER_GEOMETRY:
      nir.info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
      nir.info.gs.input_primitive = MESA_PRIM_LINES_ADJACENCY;
      nir.info.gs.vertices_out = 257 + variant;
      nir.info.gs.invocations = 1 + variant % 32;
      nir.info.gs.active_stream_mask = variant % 16;
      break;
   case MESA_SHADER_TESS_CTRL:
   case MESA_SHADER_TESS_EVAL: {
      const enum tess_primitive_mode modes[] = {
         TESS_PRIMITIVE_TRIANGLES,
         TESS_PRIMITIVE_QUADS,
         TESS_PRIMITIVE_ISOLINES,
      };
      nir.info.tess._primitive_mode = modes[variant % 3];
      nir.info.tess.tcs_vertices_out = 1 + variant % 32;
      nir.info.tess.spacing = (variant / 3) % 4;
      nir.info.tess.ccw = (variant / 12) & 1;
      nir.info.tess.point_mode = (variant / 24) & 1;
      break;
   }
   case MESA_SHADER_MESH: {
      const enum mesa_prim modes[] = {
         MESA_PRIM_POINTS, MESA_PRIM_LINES, MESA_PRIM_TRIANGLES,
      };
      nir.info.mesh.primitive_type = modes[variant % 3];
      nir.info.mesh.max_vertices_out = 401 + variant;
      nir.info.mesh.max_primitives_out = 271 + variant;
      break;
   }
   default:
      break;
   }
   nak_nir_get_shader_info(&nir, info);
}
