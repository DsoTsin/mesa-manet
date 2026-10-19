# Copyright © 2018 Rob Clark
# Copyright © 2019 Collabora
# SPDX-License-Identifier: MIT

if panfrost_kmds.contains('kbase')
  if not with_panfrost_rust
    error('The kbase panfrost KMD backend requires -Dpanfrost-rust=true')
  endif
  subdir('kbase-rs')
endif
subdir('kmod')

pixel_format_versions = ['5', '6', '7', '9', '10', '11', '12', '13', '14', '15']
libpanfrost_pixel_format = []

deps_for_libpanfrost = [dep_libdrm, idep_pan_packers, idep_mesautil, libpanfrost_model_dep]

foreach ver : pixel_format_versions
  libpanfrost_pixel_format += static_library(
    'pan-format-v' + ver,
    ['pan_format.c', pan_packers],
    include_directories : [inc_src, inc_include, inc_panfrost],
    c_args : ['-DPAN_ARCH=' + ver],
    gnu_symbol_visibility : 'hidden',
    dependencies : [deps_for_libpanfrost, dep_valgrind],
)
endforeach

libpanfrost_per_arch = []

foreach ver : ['4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15']
  libpanfrost_per_arch += static_library(
    'pan-arch-v' + ver,
    [
      'pan_blend.c',
      'pan_buffer.c',
      'pan_desc.c',
      'pan_fb.c',
      'pan_fb_nir.c',
      'pan_mod.c',
      'pan_texture.c',
    ],
    include_directories : [inc_include, inc_src, inc_panfrost],
    c_args : ['-DPAN_ARCH=' + ver],
    gnu_symbol_visibility : 'hidden',
    dependencies : [deps_for_libpanfrost, idep_nir],
  )
endforeach

libpanfrost_lib_files = files(
  'pan_encoder.h',

  'pan_blend.c',
  'pan_clear.c',
  'pan_earlyzs.c',
  '../vulkan/data_graph/neural/pan_graph_alloc.c',
  '../vulkan/data_graph/neural/pan_graph_cost.c',
  '../vulkan/data_graph/neural/pan_graph_matmul.c',
    '../vulkan/data_graph/neural/pan_graph_conv.c',
    '../vulkan/data_graph/neural/pan_graph_conv_v15.c',
  '../vulkan/data_graph/neural/pan_graph_matmul_candidate.c',
  '../vulkan/data_graph/neural/pan_graph_matmul_cost.c',
  '../vulkan/data_graph/neural/pan_graph_matmul_schedule.c',
  '../vulkan/data_graph/neural/pan_graph_matmul_search.c',
  '../vulkan/data_graph/neural/pan_graph_matmul_select.c',
  '../vulkan/data_graph/neural/pan_graph_matmul_v15.c',
  '../vulkan/data_graph/neural/pan_graph_memory_cost.c',
  '../vulkan/data_graph/neural/pan_graph_memory_bytes.c',
  '../vulkan/data_graph/neural/pan_graph_memory_descriptor.c',
  '../vulkan/data_graph/neural/pan_graph_memory_layout.c',
  '../vulkan/data_graph/neural/pan_graph_memory_policy.c',
  '../vulkan/data_graph/neural/pan_graph_memory_lifetime.c',
  '../vulkan/data_graph/neural/pan_graph_memory_registry.c',
  '../vulkan/data_graph/neural/pan_graph_memory_resource.c',
  '../vulkan/data_graph/neural/pan_graph_program_prune.c',
  '../vulkan/data_graph/neural/pan_graph_program_expand.c',
  '../vulkan/data_graph/neural/pan_graph_program_boundary.c',
  '../vulkan/data_graph/neural/pan_graph_program_cost.c',
  '../vulkan/data_graph/neural/pan_graph_program_order.c',
  '../vulkan/data_graph/neural/pan_graph_program_schedule.c',
  '../vulkan/data_graph/neural/pan_graph_program_v15.c',
  '../vulkan/data_graph/neural/pan_graph_program_view.c',
  '../vulkan/data_graph/neural/pan_graph_program_memory.c',
  '../vulkan/data_graph/neural/pan_graph_program_deps.c',
  '../vulkan/data_graph/neural/pan_graph_vector_cost.c',
  '../vulkan/data_graph/neural/pan_graph_vector_schedule.c',
  '../vulkan/data_graph/neural/pan_graph_vector_v15.c',
  '../vulkan/data_graph/neural/pan_graph_nrt.c',
  '../vulkan/data_graph/neural/pan_graph_nrt_plan.c',
  '../vulkan/data_graph/neural/pan_graph_tsu.c',
  '../vulkan/data_graph/neural/pan_graph_v15.c',
  'pan_samples.c',
  'pan_tiler.c',
  'pan_layout.c',
  'pan_scratch.c',
  'pan_trace.c',
  'pan_props.c',
  'pan_util.c',
  'pan_afbc.c',
)

libpanfrost_lib = static_library(
  'panfrost_lib',
  [libpanfrost_lib_files, pan_packers],
  include_directories : [inc_include, inc_src, inc_panfrost],
  c_args : [no_override_init_args, soong_compat_c_args],
  gnu_symbol_visibility : 'hidden',
  dependencies: [dep_libdrm, idep_nir, idep_mesautil, libpanfrost_model_dep],
  build_by_default : false,
  link_with: [libpanfrost_pixel_format, libpanfrost_per_arch, libpankmod_lib],
)

libpanfrost_dep = declare_dependency(
  link_with: [libpanfrost_lib, libpanfrost_decode, libpanfrost_pixel_format, libpanfrost_per_arch],
  include_directories: [inc_include, inc_src, inc_panfrost],
  dependencies: [deps_for_libpanfrost, libpankmod_dep, idep_nir],
)

if with_tests
  test(
    'panfrost_graph_conv',
    executable(
      'panfrost_graph_conv_test',
      files('../vulkan/data_graph/neural/tests/test-graph-conv.c'),
      link_with : libpanfrost_lib,
      include_directories : [inc_src, inc_include, inc_panfrost,
                             include_directories('../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_program_deps',
    executable(
      'panfrost_graph_program_deps_test',
      files('../vulkan/data_graph/neural/tests/test-graph-program-deps.c', '../vulkan/data_graph/neural/pan_graph_program_deps.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_program_view',
    executable(
      'panfrost_graph_program_view_test',
      files('../vulkan/data_graph/neural/tests/test-graph-program-view.c', '../vulkan/data_graph/neural/pan_graph_program_view.c',
            '../vulkan/data_graph/neural/pan_graph_program_boundary.c', '../vulkan/data_graph/neural/pan_graph_matmul.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_program_memory',
    executable(
      'panfrost_graph_program_memory_test',
      files('../vulkan/data_graph/neural/tests/test-graph-program-memory.c', '../vulkan/data_graph/neural/pan_graph_program_memory.c',
            '../vulkan/data_graph/neural/pan_graph_program_view.c', '../vulkan/data_graph/neural/pan_graph_program_boundary.c',
            '../vulkan/data_graph/neural/pan_graph_program_schedule.c', '../vulkan/data_graph/neural/pan_graph_program_order.c',
            '../vulkan/data_graph/neural/pan_graph_program_cost.c', '../vulkan/data_graph/neural/pan_graph_program_expand.c',
            '../vulkan/data_graph/neural/pan_graph_program_prune.c', '../vulkan/data_graph/neural/pan_graph_vector_schedule.c',
            '../vulkan/data_graph/neural/pan_graph_vector_cost.c', '../vulkan/data_graph/neural/pan_graph_matmul_select.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_search.c', '../vulkan/data_graph/neural/pan_graph_matmul_candidate.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_schedule.c', '../vulkan/data_graph/neural/pan_graph_matmul_cost.c',
            '../vulkan/data_graph/neural/pan_graph_matmul.c', '../vulkan/data_graph/neural/pan_graph_cost.c',
            '../vulkan/data_graph/neural/pan_graph_memory_cost.c', '../vulkan/data_graph/neural/pan_graph_memory_bytes.c',
            '../vulkan/data_graph/neural/pan_graph_memory_layout.c', '../vulkan/data_graph/neural/pan_graph_memory_lifetime.c',
            '../vulkan/data_graph/neural/pan_graph_memory_resource.c', '../vulkan/data_graph/neural/pan_graph_memory_descriptor.c',
            '../vulkan/data_graph/neural/pan_graph_memory_policy.c', '../vulkan/data_graph/neural/pan_graph_alloc.c',
            '../vulkan/data_graph/neural/pan_graph_nrt.c', '../vulkan/data_graph/neural/pan_graph_nrt_plan.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_program_order',
    executable(
      'panfrost_graph_program_order_test',
      files('../vulkan/data_graph/neural/tests/test-graph-program-order.c', '../vulkan/data_graph/neural/pan_graph_program_order.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_memory_policy',
    executable(
      'panfrost_graph_memory_policy_test',
      files('../vulkan/data_graph/neural/tests/test-graph-memory-policy.c', '../vulkan/data_graph/neural/pan_graph_memory_policy.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_program_v15',
    executable(
      'panfrost_graph_program_v15_test',
      files('../vulkan/data_graph/neural/tests/test-graph-program-v15.c', '../vulkan/data_graph/neural/pan_graph_program_v15.c',
            '../vulkan/data_graph/neural/pan_graph_program_cost.c', '../vulkan/data_graph/neural/pan_graph_program_boundary.c',
            '../vulkan/data_graph/neural/pan_graph_vector_cost.c', '../vulkan/data_graph/neural/pan_graph_vector_v15.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_cost.c', '../vulkan/data_graph/neural/pan_graph_matmul_v15.c',
            '../vulkan/data_graph/neural/pan_graph_matmul.c', '../vulkan/data_graph/neural/pan_graph_memory_cost.c',
            '../vulkan/data_graph/neural/pan_graph_memory_bytes.c', '../vulkan/data_graph/neural/pan_graph_memory_layout.c',
            '../vulkan/data_graph/neural/pan_graph_cost.c', '../vulkan/data_graph/neural/pan_graph_nrt.c', '../vulkan/data_graph/neural/pan_graph_nrt_plan.c',
            '../vulkan/data_graph/neural/pan_graph_tsu.c', '../vulkan/data_graph/neural/pan_graph_v15.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_program_schedule',
    executable(
      'panfrost_graph_program_schedule_test',
      files('../vulkan/data_graph/neural/tests/test-graph-program-schedule.c', '../vulkan/data_graph/neural/pan_graph_program_schedule.c',
            '../vulkan/data_graph/neural/pan_graph_program_order.c',
            '../vulkan/data_graph/neural/pan_graph_program_cost.c', '../vulkan/data_graph/neural/pan_graph_program_boundary.c',
            '../vulkan/data_graph/neural/pan_graph_program_expand.c', '../vulkan/data_graph/neural/pan_graph_program_prune.c',
            '../vulkan/data_graph/neural/pan_graph_vector_schedule.c', '../vulkan/data_graph/neural/pan_graph_vector_cost.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_select.c', '../vulkan/data_graph/neural/pan_graph_matmul_search.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_candidate.c', '../vulkan/data_graph/neural/pan_graph_matmul_schedule.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_cost.c', '../vulkan/data_graph/neural/pan_graph_matmul.c', '../vulkan/data_graph/neural/pan_graph_cost.c',
            '../vulkan/data_graph/neural/pan_graph_memory_cost.c', '../vulkan/data_graph/neural/pan_graph_memory_bytes.c',
            '../vulkan/data_graph/neural/pan_graph_memory_layout.c', '../vulkan/data_graph/neural/pan_graph_nrt.c',
            '../vulkan/data_graph/neural/pan_graph_nrt_plan.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_vector_schedule',
    executable(
      'panfrost_graph_vector_schedule_test',
      files('../vulkan/data_graph/neural/tests/test-graph-vector-schedule.c', '../vulkan/data_graph/neural/pan_graph_vector_schedule.c',
            '../vulkan/data_graph/neural/pan_graph_nrt.c', '../vulkan/data_graph/neural/pan_graph_nrt_plan.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_schedule.c', '../vulkan/data_graph/neural/pan_graph_matmul.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_vector_v15',
    executable(
      'panfrost_graph_vector_v15_test',
      files('../vulkan/data_graph/neural/tests/test-graph-vector-v15.c', '../vulkan/data_graph/neural/pan_graph_vector_v15.c',
            '../vulkan/data_graph/neural/pan_graph_v15.c', '../vulkan/data_graph/neural/pan_graph_tsu.c', '../vulkan/data_graph/neural/pan_graph_nrt.c',
            '../vulkan/data_graph/neural/pan_graph_nrt_plan.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_program_cost',
    executable(
      'panfrost_graph_program_cost_test',
      files('../vulkan/data_graph/neural/tests/test-graph-program-cost.c', '../vulkan/data_graph/neural/pan_graph_program_cost.c',
            '../vulkan/data_graph/neural/pan_graph_program_boundary.c', '../vulkan/data_graph/neural/pan_graph_vector_cost.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_cost.c', '../vulkan/data_graph/neural/pan_graph_matmul.c',
            '../vulkan/data_graph/neural/pan_graph_memory_cost.c', '../vulkan/data_graph/neural/pan_graph_memory_bytes.c',
            '../vulkan/data_graph/neural/pan_graph_memory_layout.c', '../vulkan/data_graph/neural/pan_graph_cost.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_program_boundary',
    executable(
      'panfrost_graph_program_boundary_test',
      files('../vulkan/data_graph/neural/tests/test-graph-program-boundary.c', '../vulkan/data_graph/neural/pan_graph_program_boundary.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_memory_lifetime',
    executable(
      'panfrost_graph_memory_lifetime_test',
      files('../vulkan/data_graph/neural/tests/test-graph-memory-lifetime.c', '../vulkan/data_graph/neural/pan_graph_memory_lifetime.c',
            '../vulkan/data_graph/neural/pan_graph_alloc.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_program_expand',
    executable(
      'panfrost_graph_program_expand_test',
      files('../vulkan/data_graph/neural/tests/test-graph-program-expand.c', '../vulkan/data_graph/neural/pan_graph_program_expand.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_vector_cost',
    executable(
      'panfrost_graph_vector_cost_test',
      files('../vulkan/data_graph/neural/tests/test-graph-vector-cost.c', '../vulkan/data_graph/neural/pan_graph_vector_cost.c',
            '../vulkan/data_graph/neural/pan_graph_memory_cost.c', '../vulkan/data_graph/neural/pan_graph_memory_bytes.c',
            '../vulkan/data_graph/neural/pan_graph_memory_layout.c', '../vulkan/data_graph/neural/pan_graph_cost.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_program_prune',
    executable(
      'panfrost_graph_program_prune_test',
      files('../vulkan/data_graph/neural/tests/test-graph-program-prune.c', '../vulkan/data_graph/neural/pan_graph_program_prune.c',
            '../vulkan/data_graph/neural/pan_graph_cost.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_matmul_select',
    executable(
      'panfrost_graph_matmul_select_test',
      files('../vulkan/data_graph/neural/tests/test-graph-matmul-select.c', '../vulkan/data_graph/neural/pan_graph_matmul_select.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_search.c', '../vulkan/data_graph/neural/pan_graph_matmul_candidate.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_schedule.c', '../vulkan/data_graph/neural/pan_graph_matmul_cost.c',
            '../vulkan/data_graph/neural/pan_graph_matmul.c', '../vulkan/data_graph/neural/pan_graph_nrt.c', '../vulkan/data_graph/neural/pan_graph_cost.c',
            '../vulkan/data_graph/neural/pan_graph_memory_cost.c', '../vulkan/data_graph/neural/pan_graph_memory_bytes.c',
            '../vulkan/data_graph/neural/pan_graph_memory_layout.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_matmul_search',
    executable(
      'panfrost_graph_matmul_search_test',
      files('../vulkan/data_graph/neural/tests/test-graph-matmul-search.c', '../vulkan/data_graph/neural/pan_graph_matmul_search.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_candidate.c', '../vulkan/data_graph/neural/pan_graph_matmul.c',
            '../vulkan/data_graph/neural/pan_graph_nrt.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_matmul_candidate',
    executable(
      'panfrost_graph_matmul_candidate_test',
      files('../vulkan/data_graph/neural/tests/test-graph-matmul-candidate.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_candidate.c', '../vulkan/data_graph/neural/pan_graph_matmul.c',
            '../vulkan/data_graph/neural/pan_graph_nrt.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_matmul_schedule',
    executable(
      'panfrost_graph_matmul_schedule_test',
      files('../vulkan/data_graph/neural/tests/test-graph-matmul-schedule.c',
            '../vulkan/data_graph/neural/pan_graph_matmul_schedule.c', '../vulkan/data_graph/neural/pan_graph_matmul.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_matmul_cost',
    executable(
      'panfrost_graph_matmul_cost_test',
      files('../vulkan/data_graph/neural/tests/test-graph-matmul-cost.c', '../vulkan/data_graph/neural/pan_graph_matmul_cost.c',
            '../vulkan/data_graph/neural/pan_graph_matmul.c', '../vulkan/data_graph/neural/pan_graph_cost.c',
            '../vulkan/data_graph/neural/pan_graph_memory_cost.c', '../vulkan/data_graph/neural/pan_graph_memory_bytes.c',
            '../vulkan/data_graph/neural/pan_graph_memory_layout.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_memory_cost',
    executable(
      'panfrost_graph_memory_cost_test',
      files('../vulkan/data_graph/neural/tests/test-graph-memory-cost.c', '../vulkan/data_graph/neural/pan_graph_memory_cost.c',
            '../vulkan/data_graph/neural/pan_graph_memory_bytes.c', '../vulkan/data_graph/neural/pan_graph_memory_layout.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_memory_descriptor',
    executable(
      'panfrost_graph_memory_descriptor_test',
      files('../vulkan/data_graph/neural/tests/test-graph-memory-descriptor.c',
            '../vulkan/data_graph/neural/pan_graph_memory_descriptor.c', '../vulkan/data_graph/neural/pan_graph_memory_layout.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_memory_resource',
    executable(
      'panfrost_graph_memory_resource_test',
      files('../vulkan/data_graph/neural/tests/test-graph-memory-resource.c', '../vulkan/data_graph/neural/pan_graph_memory_resource.c',
            '../vulkan/data_graph/neural/pan_graph_memory_descriptor.c', '../vulkan/data_graph/neural/pan_graph_memory_layout.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_memory_registry',
    executable(
      'panfrost_graph_memory_registry_test',
      files('../vulkan/data_graph/neural/tests/test-graph-memory-registry.c', '../vulkan/data_graph/neural/pan_graph_memory_registry.c',
            '../vulkan/data_graph/neural/pan_graph_memory_layout.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_memory_bytes',
    executable(
      'panfrost_graph_memory_bytes_test',
      files('../vulkan/data_graph/neural/tests/test-graph-memory-bytes.c', '../vulkan/data_graph/neural/pan_graph_memory_bytes.c',
            '../vulkan/data_graph/neural/pan_graph_memory_layout.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_memory_layout',
    executable(
      'panfrost_graph_memory_layout_test',
      files('../vulkan/data_graph/neural/tests/test-graph-memory-layout.c', '../vulkan/data_graph/neural/pan_graph_memory_layout.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_cost',
    executable(
      'panfrost_graph_cost_test',
      files('../vulkan/data_graph/neural/tests/test-graph-cost.c', '../vulkan/data_graph/neural/pan_graph_cost.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      dependencies : [dep_m],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_matmul_v15',
    executable(
      'panfrost_graph_matmul_v15_test',
      files('../vulkan/data_graph/neural/tests/test-graph-matmul-v15.c', '../vulkan/data_graph/neural/pan_graph_matmul_v15.c',
            '../vulkan/data_graph/neural/pan_graph_matmul.c', '../vulkan/data_graph/neural/pan_graph_nrt.c', '../vulkan/data_graph/neural/pan_graph_nrt_plan.c',
            '../vulkan/data_graph/neural/pan_graph_tsu.c',
            '../vulkan/data_graph/neural/pan_graph_v15.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_nrt_plan',
    executable(
      'panfrost_graph_nrt_plan_test',
      files('../vulkan/data_graph/neural/tests/test-graph-nrt-plan.c', '../vulkan/data_graph/neural/pan_graph_nrt_plan.c',
            '../vulkan/data_graph/neural/pan_graph_nrt.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_nrt',
    executable(
      'panfrost_graph_nrt_test',
      files('../vulkan/data_graph/neural/tests/test-graph-nrt.c', '../vulkan/data_graph/neural/pan_graph_nrt.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_matmul',
    executable(
      'panfrost_graph_matmul_test',
      files('../vulkan/data_graph/neural/tests/test-graph-matmul.c', '../vulkan/data_graph/neural/pan_graph_matmul.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  if host_machine.system() != 'windows' and host_machine.system() != 'darwin'
    test(
      'panfrost_graph_tsu_oom',
      executable(
        'panfrost_graph_tsu_oom_test',
        files('../vulkan/data_graph/neural/tests/test-graph-tsu-oom.c', '../vulkan/data_graph/neural/pan_graph_tsu.c', '../vulkan/data_graph/neural/pan_graph_v15.c'),
        c_args : [c_msvc_compat_args],
        link_args : ['-Wl,--wrap=malloc', '-Wl,--wrap=realloc', '-Wl,--wrap=free'],
        include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
      ),
      suite : ['panfrost'],
    )
  endif
  test(
    'panfrost_graph_tsu',
    executable(
      'panfrost_graph_tsu_test',
      files('../vulkan/data_graph/neural/tests/test-graph-tsu.c', '../vulkan/data_graph/neural/pan_graph_tsu.c', '../vulkan/data_graph/neural/pan_graph_v15.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_alloc',
    executable(
      'panfrost_graph_alloc_test',
      files('../vulkan/data_graph/neural/tests/test-graph-alloc.c', '../vulkan/data_graph/neural/pan_graph_alloc.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )
  test(
    'panfrost_graph_v15',
    executable(
      'panfrost_graph_v15_test',
      files('../vulkan/data_graph/neural/tests/test-graph-v15.c', '../vulkan/data_graph/neural/pan_graph_v15.c', '../vulkan/data_graph/neural/pan_graph_tsu.c'),
      c_args : [c_msvc_compat_args],
      include_directories : [include_directories('.', '../vulkan/data_graph/neural')],
    ),
    suite : ['panfrost'],
  )

  test(
    'panfrost_crc_v15',
    executable(
      'panfrost_crc_v15_test',
      files('tests/test-crc-v15.c'),
      c_args : [c_msvc_compat_args, '-DPAN_ARCH=15'],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src, inc_panfrost],
      dependencies: [libpanfrost_dep],
    ),
    suite : ['panfrost'],
  )

  test(
    'panfrost_attribute',
    executable(
      'panfrost_attribute_test',
      files('tests/test-attribute.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies: [libpanfrost_dep],
    ),
    suite : ['panfrost'],
  )

  test(
    'panfrost_blend',
    executable(
      'panfrost_blend_test',
      files('tests/test-blend.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies: [libpanfrost_dep],
    ),
    suite : ['panfrost'],
  )

  test(
    'panfrost_clear',
    executable(
      'panfrost_clear_test',
      files('tests/test-clear.c'),
      c_args : [c_msvc_compat_args, no_override_init_args],
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src],
      dependencies: [libpanfrost_dep],
    ),
    suite : ['panfrost'],
  )

  test(
    'panfrost_tests',
    executable(
      'panfrost_tests',
      files(
        'tests/test-earlyzs.cpp',
        'tests/test-layout.cpp',
      ),
      cpp_args : cpp_msvc_compat_args,
      gnu_symbol_visibility : 'hidden',
      include_directories : [inc_include, inc_src, inc_panfrost],
      dependencies: [idep_gtest, libpanfrost_dep],
    ),
    suite : ['panfrost'],
    protocol : 'gtest',
  )
endif
