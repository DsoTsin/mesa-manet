inc_agc_isa = include_directories('.')
inc_agc_isa_lower = include_directories('lower')

agc_isa_generated = custom_target(
  'agc_isa_tables',
  input : files(
    '../../gen/agc_isa_gen.py',
    '../../gen/agc_isa_ps5.json',
  ),
  output : ['agc_isa_opcodes.h', 'agc_isa_tables.h'],
  command : [
    prog_python, '@INPUT0@',
    '--ps5', '@INPUT1@',
    '--out-opcodes', '@OUTPUT0@',
    '--out-tables', '@OUTPUT1@',
  ],
)

libagc_isa = static_library(
  'agc_isa',
  files(
    'agc_isa_analyze.c',
    'agc_isa_cfg.c',
    'agc_isa_computed.c',
    'agc_isa_decode.c',
    'agc_isa_descriptors.c',
    'agc_isa_ir.c',
    'agc_isa_opcodes.c',
    'agc_isa_regalloc.c',
    'agc_isa_sources.c',
  ),
  agc_isa_generated,
  include_directories : [inc_agc, inc_agc_isa],
  gnu_symbol_visibility : 'hidden',
)

idep_agc_isa = declare_dependency(
  sources : agc_isa_generated[0],
  include_directories : [inc_agc, inc_agc_isa],
  link_with : libagc_isa,
)

libagc_isa_aco = static_library(
  'agc_isa_aco',
  files(
    'agc_isa_translate.c',
    'lower/agc_lower.cpp',
    'lower/agc_lower_bvh.cpp',
    'lower/agc_lower_bvh8.cpp',
    'lower/agc_lower_bvh_stack.cpp',
    'lower/agc_lower_computed.cpp',
    'lower/agc_lower_descriptors.cpp',
    'lower/agc_lower_emit.cpp',
    'lower/agc_lower_gds.cpp',
    'lower/agc_lower_image_atomic.cpp',
    'lower/agc_lower_image_reports.cpp',
    'lower/agc_lower_inst.cpp',
    'lower/agc_lower_interp.cpp',
    'lower/agc_lower_lds.cpp',
    'lower/agc_lower_memory_atomic.cpp',
    'lower/agc_lower_mip_stats.cpp',
    'lower/agc_lower_ngg.cpp',
    'lower/agc_lower_pixel.cpp',
    'lower/agc_lower_sampling.cpp',
    'lower/agc_lower_sdwa.cpp',
    'lower/agc_lower_spill.cpp',
    'lower/agc_lower_vector.cpp',
  ),
  include_directories : [inc_agc, inc_agc_isa, inc_agc_isa_lower, inc_include, inc_src, inc_amd, inc_amd_common],
  dependencies : [idep_agc_isa, idep_aco, idep_amd_generated_headers, idep_nir_headers, idep_mesautil],
  link_with : libamd_common,
  cpp_args : [cpp_msvc_compat_args],
  override_options : ['cpp_std=c++20'],
  gnu_symbol_visibility : 'hidden',
)

idep_agc_isa_aco = declare_dependency(
  include_directories : [inc_agc_isa, inc_agc_isa_lower],
  link_with : [libagc_isa_aco, libamd_common],
  dependencies : [idep_agc_isa, idep_aco],
)

executable(
  'agc_isa_dump',
  files('../../tools/agc_isa_dump.c'),
  dependencies : idep_agc_isa,
)

if with_tests
  test(
    'agc_isa_decode',
    executable(
      'agc_test_isa_decode',
      files('../../tests/test_isa_decode.c'),
      dependencies : idep_agc_isa,
    ),
    suite : ['amd', 'agc'],
  )
  test(
    'agc_isa_aco',
    executable(
      'agc_test_isa_aco',
      files('../../tests/test_isa_aco.c'),
      dependencies : idep_agc_isa_aco,
      link_with : libagc_shader_artifact,
    ),
    suite : ['amd', 'agc'],
  )
  test(
    'agc_isa_lower_stages',
    executable(
      'agc_test_isa_lower_stages',
      files('../../tests/test_isa_lower_stages.c'),
      dependencies : idep_agc_isa_aco,
    ),
    suite : ['amd', 'agc'],
  )
  test(
    'agc_isa_lower_vector',
    executable(
      'agc_test_isa_lower_vector',
      files('../../tests/test_isa_lower_vector.c'),
      dependencies : idep_agc_isa_aco,
    ),
    suite : ['amd', 'agc'],
  )
endif
