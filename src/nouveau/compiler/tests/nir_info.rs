use compiler::bindings::*;
use nak_bindings::*;
use std::mem::{align_of, offset_of, size_of};

extern "C" {
    fn nak_test_nir_info_layout(index: u32) -> usize;
    fn nak_test_nir_info(
        stage: u32,
        variant: u32,
        info: *mut nak_nir_shader_info,
    );
}

fn main() {
    let layout = [
        size_of::<nak_nir_shader_info>(),
        align_of::<nak_nir_shader_info>(),
        offset_of!(nak_nir_shader_info, stage),
        offset_of!(nak_nir_shader_info, workgroup_size),
        offset_of!(nak_nir_shader_info, clip_distance_array_size),
        offset_of!(nak_nir_shader_info, cull_distance_array_size),
        offset_of!(nak_nir_shader_info, shared_size),
        offset_of!(nak_nir_shader_info, task_payload_size),
        offset_of!(nak_nir_shader_info, float_controls_execution_mode),
        offset_of!(nak_nir_shader_info, gs),
        offset_of!(nak_nir_shader_info, fs),
        offset_of!(nak_nir_shader_info, tess),
        offset_of!(nak_nir_shader_info, mesh),
        size_of::<nak_nir_gs_info>(),
        size_of::<nak_nir_fs_info>(),
        size_of::<nak_nir_tess_info>(),
        size_of::<nak_nir_mesh_info>(),
        offset_of!(nak_nir_gs_info, active_stream_mask),
        offset_of!(nak_nir_tess_info, spacing),
        offset_of!(nak_nir_tess_info, ccw),
        offset_of!(nak_nir_tess_info, point_mode),
        offset_of!(nak_nir_mesh_info, max_vertices_out),
    ];
    for (index, value) in layout.iter().enumerate() {
        assert_eq!(
            *value,
            unsafe { nak_test_nir_info_layout(index as u32) },
            "layout index {index}"
        );
    }
    let stages = [
        MESA_SHADER_VERTEX,
        MESA_SHADER_TESS_CTRL,
        MESA_SHADER_TESS_EVAL,
        MESA_SHADER_GEOMETRY,
        MESA_SHADER_FRAGMENT,
        MESA_SHADER_COMPUTE,
        MESA_SHADER_TASK,
        MESA_SHADER_MESH,
    ];
    let mut cases = 0;
    for stage in stages {
        for variant in 0..48 {
            let mut info = nak_nir_shader_info::default();
            unsafe { nak_test_nir_info(stage as u32, variant, &mut info) };
            assert_eq!(info.stage, stage as u32);
            assert_eq!(info.workgroup_size, [3, 5, 7]);
            assert_eq!(info.clip_distance_array_size, 2);
            assert_eq!(info.cull_distance_array_size, 3);
            assert_eq!(info.shared_size, 1536);
            assert_eq!(info.task_payload_size, 320);
            assert_eq!(info.float_controls_execution_mode, 0x1234);
            match stage {
                MESA_SHADER_FRAGMENT => {
                    assert_eq!(info.fs.post_depth_coverage as u32, variant & 1);
                    assert_eq!(
                        info.fs.early_fragment_tests as u32,
                        (variant >> 1) & 1
                    );
                    assert_eq!(
                        info.fs.uses_sample_shading as u32,
                        (variant >> 2) & 1
                    );
                }
                MESA_SHADER_GEOMETRY => {
                    assert_eq!(
                        info.gs.output_primitive,
                        MESA_PRIM_TRIANGLE_STRIP as u32
                    );
                    assert_eq!(
                        info.gs.input_primitive,
                        MESA_PRIM_LINES_ADJACENCY as u32
                    );
                    assert_eq!(info.gs.vertices_out as u32, 257 + variant);
                    assert_eq!(info.gs.invocations as u32, 1 + variant % 32);
                    assert_eq!(info.gs.active_stream_mask as u32, variant % 16);
                }
                MESA_SHADER_TESS_CTRL | MESA_SHADER_TESS_EVAL => {
                    let modes = [
                        TESS_PRIMITIVE_TRIANGLES,
                        TESS_PRIMITIVE_QUADS,
                        TESS_PRIMITIVE_ISOLINES,
                    ];
                    assert_eq!(
                        info.tess.primitive_mode,
                        modes[variant as usize % 3] as u32
                    );
                    assert_eq!(
                        info.tess.tcs_vertices_out as u32,
                        1 + variant % 32
                    );
                    assert_eq!(info.tess.spacing as u32, (variant / 3) % 4);
                    assert_eq!(info.tess.ccw as u32, (variant / 12) & 1);
                    assert_eq!(info.tess.point_mode as u32, (variant / 24) & 1);
                }
                MESA_SHADER_MESH => {
                    let modes = [
                        MESA_PRIM_POINTS,
                        MESA_PRIM_LINES,
                        MESA_PRIM_TRIANGLES,
                    ];
                    assert_eq!(
                        info.mesh.primitive_type,
                        modes[variant as usize % 3] as u32
                    );
                    assert_eq!(
                        info.mesh.max_vertices_out as u32,
                        401 + variant
                    );
                    assert_eq!(
                        info.mesh.max_primitives_out as u32,
                        271 + variant
                    );
                }
                _ => {}
            }
            cases += 1;
        }
    }
    println!("{{\"cases\":{cases},\"layout_checks\":{},\"mismatches\":0,\"snapshot_bytes\":{},\"snapshot_alignment\":{}}}", layout.len(), size_of::<nak_nir_shader_info>(), align_of::<nak_nir_shader_info>());
}
