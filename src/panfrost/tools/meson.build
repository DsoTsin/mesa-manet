# Copyright © 2021 Collabora
# SPDX-License-Identifier: MIT

coredumpdec = executable(
  'panfrostdump',
  files('panfrostdump.c'),
  c_args : [c_msvc_compat_args, no_override_init_args, compile_args_panfrost],
  gnu_symbol_visibility : 'hidden',
  include_directories : [inc_include, inc_src],
  dependencies: [libpanfrost_dep],
  link_with: [libpanfrost_compiler],
  build_by_default : true,
  install: true
)

panfrost_texfeatures = executable(
  'panfrost_texfeatures',
  files('panfrost_texfeatures.c'),
  c_args : [c_msvc_compat_args, no_override_init_args, compile_args_panfrost],
  gnu_symbol_visibility : 'hidden',
  include_directories : [inc_include, inc_src],
  dependencies: [libpanfrost_dep],
  build_by_default : true,
  install: true
)

malioc_dependencies = []
malioc_link_args = []
if host_machine.system() == 'windows'
  malioc_dependencies += cc.find_library('bcrypt')
  if cc.get_argument_syntax() != 'msvc'
    malioc_link_args += '-municode'
  endif
endif

malioc_disasm = executable(
  'malioc_disasm',
  files('malioc_disasm.c'),
  c_args : [c_msvc_compat_args, no_override_init_args],
  gnu_symbol_visibility : 'hidden',
  include_directories : include_directories('../compiler/bifrost'),
  dependencies : malioc_dependencies,
  link_args : malioc_link_args,
  link_with : [libpanfrost_valhall_disasm],
  build_by_default : true,
  install : false,
)

if with_tests
  test(
    'malioc_disasm',
    prog_python,
    args : [files('test_malioc_disasm.py'), malioc_disasm],
    suite : ['panfrost'],
  )
endif
