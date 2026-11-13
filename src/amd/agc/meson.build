inc_agc = include_directories('include')

agc_regclass_tables_h = custom_target(
  'agc_regclass_tables.h',
  input : files(
    'gen/agc_regclass_gen.py',
    'gen/agc_reg_overrides.json',
    'gen/agc_guest_defaults.json',
    '../registers/gfx10.json',
    '../registers/gfx103.json',
    '../registers/gfx11.json',
    '../registers/gfx115.json',
    '../registers/gfx12.json',
  ),
  output : 'agc_regclass_tables.h',
  command : [
    prog_python, '@INPUT0@',
    '--overrides', '@INPUT1@',
    '--defaults', '@INPUT2@',
    '--guest', '@INPUT3@',
    '--host', 'gfx103', '@INPUT4@',
    '--host', 'gfx11', '@INPUT5@',
    '--host', 'gfx115', '@INPUT6@',
    '--host', 'gfx12', '@INPUT7@',
    '--out', '@OUTPUT@',
  ],
)

agc_export_inputs = files('gen/agc_exports.py', 'gen/agc_exports.json', 'gen/agc_exports_core.json')

agc_exports_h = custom_target(
  'agc_exports.h',
  input : agc_export_inputs,
  output : 'agc_exports.h',
  command : [prog_python, '@INPUT0@', '@INPUT1@', '@INPUT2@', '--header', '@OUTPUT@'],
)

agc_export_stubs_c = custom_target(
  'agc_export_stubs.c',
  input : agc_export_inputs,
  output : 'agc_export_stubs.c',
  command : [prog_python, '@INPUT0@', '@INPUT1@', '@INPUT2@', '--stubs', '@OUTPUT@'],
)

agc_trace_thunks_c = custom_target(
  'agc_trace_thunks.c',
  input : agc_export_inputs,
  output : 'agc_trace_thunks.c',
  command : [prog_python, '@INPUT0@', '@INPUT1@', '@INPUT2@', '--thunks', '@OUTPUT@'],
)

agc_register_defaults_tables_h = custom_target(
  'agc_register_defaults_tables.h',
  input : files('gen/agc_defaults_gen.py', 'gen/agc_guest_defaults.json'),
  output : 'agc_register_defaults_tables.h',
  command : [prog_python, '@INPUT0@', '@INPUT1@', '--out', '@OUTPUT@'],
)

agc_inc = [inc_agc, inc_include, inc_src, inc_amd, inc_amd_common]
agc_deps = [idep_amd_generated_headers, idep_mesautil, idep_nir_headers]

libagc = static_library(
  'agc',
  files(
    'src/agc_acb.c',
    'src/agc_cb.c',
    'src/agc_cb_builders.c',
    'src/agc_context_state.c',
    'src/agc_dcb.c',
    'src/agc_debug.c',
    'src/agc_graphics.c',
    'src/agc_graphics_gfx11.c',
    'src/agc_anyps5.c',
    'src/agc_record.c',
    'src/agc_register_defaults.c',
    'src/agc_shader.c',
    'src/agc_texture.c',
  ),
  agc_regclass_tables_h,
  agc_register_defaults_tables_h,
  agc_exports_h,
  agc_export_stubs_c,
  include_directories : agc_inc,
  dependencies : agc_deps,
  link_with : libamd_common,
  gnu_symbol_visibility : 'hidden',
)

libagc_base = static_library(
  'agc_base',
  files(
    'src/agc_cb.c',
    'src/agc_debug.c',
    'src/agc_anyps5.c',
    'src/agc_record.c',
    'src/agc_surface.c',
    'src/agc_surface_tables.c',
  ),
  include_directories : agc_inc,
  dependencies : agc_deps,
  gnu_symbol_visibility : 'hidden',
)

libagc_kernels = static_library(
  'agc_kernels',
  files(
    'src/agc_kernels.c',
  ),
  include_directories : agc_inc,
  dependencies : [agc_deps, idep_nir, idep_aco],
  link_with : libamd_common,
  gnu_symbol_visibility : 'hidden',
)

libagc_memory = static_library(
  'agc_memory',
  files(
    'src/agc_memory.c',
  ),
  include_directories : agc_inc,
  dependencies : agc_deps,
  gnu_symbol_visibility : 'hidden',
)

libagc_videoout = static_library(
  'agc_videoout',
  files(
    'src/agc_videoout.c',
  ),
  include_directories : agc_inc,
  dependencies : agc_deps,
  gnu_symbol_visibility : 'hidden',
)

libagc_shader_artifact = static_library(
  'agc_shader_artifact',
  files(
    'src/agc_shader_artifact.c',
  ),
  include_directories : agc_inc,
  dependencies : agc_deps,
  gnu_symbol_visibility : 'hidden',
)

executable(
  'agc_shader_artifact_tool',
  files('tools/agc_shader_artifact_tool.c'),
  include_directories : agc_inc,
  dependencies : agc_deps,
  link_with : libagc_shader_artifact,
)

libagc_present = static_library(
  'agc_present',
  files(
    'src/agc_present.c',
  ) + (host_machine.system() == 'windows' ? files('src/agc_present_d3d12.c', 'src/agc_present_install.c') : []),
  include_directories : agc_inc,
  dependencies : [agc_deps, host_machine.system() == 'windows' ? [cc.find_library('gdi32'), cc.find_library('user32')] : []],
  link_with : libamd_common,
  gnu_symbol_visibility : 'hidden',
)

subdir('src/isa')

executable(
  'agc_isa_lower_tool',
  files('tools/agc_isa_lower_tool.c'),
  include_directories : agc_inc,
  dependencies : [agc_deps, idep_agc_isa, idep_agc_isa_aco],
  link_with : [libagc_shader_artifact, libagc_base, libamd_common],
)

libagc_driver = static_library(
  'agc_driver',
  files(
    'src/agc_backend.c',
    'src/agc_driver.c',
    'src/agc_driver_exports.c',
    'src/agc_images.c',
    'src/agc_indirect_exports.c',
    'src/agc_program.c',
    'src/agc_queue_state.c',
    'src/agc_runtime.c',
    'src/agc_shader_exports.c',
    'src/agc_shader_fusion.c',
  ),
  agc_exports_h,
  include_directories : agc_inc,
  dependencies : [agc_deps, idep_agc_isa],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

if host_machine.system() == 'windows'
  libagc_wddm = static_library(
    'agc_wddm',
    files('src/agc_driver_wddm.c'),
    include_directories : agc_inc,
    dependencies : [agc_deps, cc.find_library('gdi32')],
    link_with : libamd_common,
    gnu_symbol_visibility : 'hidden',
  )

  executable(
    'agc_memory_probe',
    files('tools/agc_memory_probe.c'),
    include_directories : agc_inc,
    dependencies : agc_deps,
    link_with : [libagc_wddm, libagc_memory, libagc_base],
  )

  executable(
    'agc_wddm_probe',
    files('tools/agc_wddm_probe.c'),
    include_directories : agc_inc,
    dependencies : agc_deps,
    link_with : [libagc_wddm, libagc, libagc_base],
  )

  executable(
    'agc_m1_probe',
    files('tools/agc_m1_probe.c'),
    include_directories : agc_inc,
    dependencies : agc_deps,
    link_with : [libagc_wddm, libagc_base],
  )

  agc_prx_inputs = files('gen/agc_prx.py', 'gen/agc_exports.json', 'gen/agc_exports_core.json')

  agcdrv_def = custom_target(
    'agcdrv.def',
    input : agc_prx_inputs,
    output : 'agcdrv.def',
    command : [prog_python, '@INPUT0@', '--manifest', '@INPUT1@', '--manifest', '@INPUT2@', '--def', '@OUTPUT@'],
  )

  agcdrv = shared_library(
    'agcdrv',
    files('src/agc_runtime_dll.c', 'src/agc_api_trace.c'),
    agc_trace_thunks_c,
    include_directories : agc_inc,
    dependencies : [agc_deps, idep_agc_isa, idep_agc_isa_aco, idep_nir, idep_aco],
    link_with : [libagc_driver, libagc, libagc_videoout, libagc_memory, libagc_kernels, libagc_present,
                 libagc_shader_artifact, libagc_wddm, libagc_base, libamd_common],
    vs_module_defs : agcdrv_def,
    name_prefix : '',
    build_by_default : false,
  )

  agc_prx_shells = []
  foreach module : ['libSceAgc', 'libSceAgcDriver']
    agc_prx_shells += custom_target(
      module + '.prx',
      input : agc_prx_inputs,
      output : module + '.prx',
      command : [prog_python, '@INPUT0@', '--manifest', '@INPUT1@', '--manifest', '@INPUT2@',
                 '--module', module, '--runtime', 'agcdrv', '--out', '@OUTPUT@'],
      build_by_default : false,
    )
  endforeach
  foreach module : ['libSceAgcResourceRegistration', 'libSceAgcVsh', 'libSceAgcVshDebug']
    agc_prx_shells += custom_target(
      module + '.prx',
      input : agc_prx_inputs,
      output : module + '.prx',
      command : [prog_python, '@INPUT0@', '--manifest', '@INPUT1@', '--manifest', '@INPUT2@',
                 '--module', module, '--empty', '--runtime', 'agcdrv', '--out', '@OUTPUT@'],
      build_by_default : false,
    )
  endforeach

  alias_target('agcdrv_package', agcdrv, agc_prx_shells)

  executable(
    'agc_driver_probe',
    files('tools/agc_driver_probe.c'),
    agc_exports_h,
    include_directories : agc_inc,
    dependencies : [agc_deps, idep_agc_isa, idep_nir, idep_aco],
    link_with : [libagc_driver, libagc, libagc_videoout, libagc_memory, libagc_kernels,
                 libagc_shader_artifact, libagc_wddm, libagc_base, libamd_common],
  )

  executable(
    'agc_queue_probe',
    files('tools/agc_queue_probe.c'),
    include_directories : agc_inc,
    dependencies : [agc_deps, idep_agc_isa],
    link_with : [libagc_driver, libagc, libagc_memory, libagc_wddm, libagc_base, libamd_common],
  )

  executable(
    'agc_kernel_probe',
    files('tools/agc_kernel_probe.c'),
    include_directories : agc_inc,
    dependencies : agc_deps,
    link_with : [libagc_kernels, libagc_wddm, libagc_base],
  )

  executable(
    'agc_present_probe',
    files('tools/agc_present_probe.c'),
    include_directories : agc_inc,
    dependencies : agc_deps,
    link_with : [libagc_present, libagc_kernels, libagc_memory, libagc_wddm, libagc_base],
  )
endif

executable(
  'agc_hang_parse',
  files('tools/agc_hang_parse.c'),
  include_directories : agc_inc,
  dependencies : agc_deps,
  link_with : libamd_common,
)

if with_tests
  foreach name : ['cb', 'registers', 'record', 'driver', 'graphics', 'register_defaults', 'shader', 'default_state']
    test(
      'agc_' + name,
      executable(
        'agc_test_' + name,
        files('tests/test_' + name + '.c'),
        agc_exports_h,
        include_directories : agc_inc,
        dependencies : agc_deps,
        link_with : [libagc, libagc_base],
      ),
      suite : ['amd', 'agc'],
    )
  endforeach

  agc_record_test_libs = [libagc, libagc_driver, libagc_videoout, libagc_memory, libagc_kernels,
                          libagc_shader_artifact, libagc_base, libamd_common]
  if host_machine.system() == 'windows'
    agc_record_test_libs += libagc_wddm
  endif
  foreach name : ['cb_builders', 'dcb', 'acb']
    test(
      'agc_' + name,
      executable(
        'agc_test_' + name,
        files('tests/test_' + name + '.c'),
        agc_exports_h,
        include_directories : agc_inc,
        dependencies : [agc_deps, idep_agc_isa, idep_nir, idep_aco],
        link_with : agc_record_test_libs,
      ),
      suite : ['amd', 'agc'],
    )
  endforeach

  agc_module_tests = {
    'anyps5' : [],
    'kernels' : [libagc_kernels],
    'memory' : [libagc_memory],
    'videoout' : [libagc_videoout],
    'shader_artifact' : [libagc_shader_artifact],
    'present' : [libagc_present],
    'surface' : [],
  }
  test(
    'agc_runtime',
    executable(
      'agc_test_runtime',
      files('tests/test_runtime.c', 'src/agc_runtime.c'),
      include_directories : agc_inc,
      dependencies : [agc_deps, idep_agc_isa],
      link_with : [libagc_memory, libagc_base],
    ),
    suite : ['amd', 'agc'],
  )

  foreach name, libs : agc_module_tests
    test(
      'agc_' + name,
      executable(
        'agc_test_' + name,
        files('tests/test_' + name + '.c'),
        include_directories : agc_inc,
        dependencies : agc_deps,
        link_with : [libs, libagc_base],
      ),
      suite : ['amd', 'agc'],
    )
  endforeach
endif
