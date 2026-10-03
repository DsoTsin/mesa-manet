# Copyright © 2021 Collabora
# SPDX-License-Identifier: MIT

if host_machine.system() != 'windows' or with_gallium_panfrost or with_panfrost_vk
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
endif

malioc_dependencies = []
malioc_link_args = []
if host_machine.system() == 'windows'
  malioc_dependencies += cc.find_library('bcrypt')
  if cc.get_argument_syntax() != 'msvc'
    malioc_link_args += '-municode'
  endif
endif

if with_kraidoc
  kraidoc_common = static_library(
    'kraidoc_common',
    files('../vulkan/panvk_nir_lower_cooperative_matrix.c',
          '../vulkan/panvk_nir_lower_tile_image.c',
          '../lib/pan_earlyzs.c'),
    include_directories : [inc_include, inc_src, inc_panfrost,
                           include_directories('../vulkan')],
    dependencies : [idep_nir_headers, idep_pan_packers, idep_mesautil,
                    idep_vulkan_util_headers, idep_vulkan_runtime_headers],
    c_args : [c_msvc_compat_args, no_override_init_args, soong_compat_c_args,
              '-DPANVK_OFFLINE_ONLY'],
  )
  kraidoc_arch_libs = []
  foreach arch : [9, 10, 11]
    kraidoc_arch_libs += static_library(
      'kraidoc_v@0@'.format(arch),
      files('../vulkan/panvk_vX_shader.c',
            '../vulkan/panvk_vX_descriptor_set_layout.c',
            '../vulkan/panvk_vX_nir_lower_descriptors.c',
            '../vulkan/panvk_vX_nir_lower_input_attachment_loads.c'),
      include_directories : [inc_include, inc_src, inc_panfrost,
                             include_directories('../vulkan')],
      dependencies : [idep_nir_headers, idep_pan_packers, idep_mesautil,
                      idep_vulkan_util_headers, idep_vulkan_runtime_headers],
      c_args : [c_msvc_compat_args, no_override_init_args, soong_compat_c_args,
                '-DPAN_ARCH=@0@'.format(arch), '-DPANVK_OFFLINE_ONLY'],
    )
  endforeach
  kraidoc = executable(
    'kraidoc',
    [files('kraidoc.c'), sha1_h],
    include_directories : [inc_include, inc_src, inc_panfrost,
                           include_directories('../vulkan')],
    dependencies : [dep_m, dep_thread, idep_nir, idep_pan_packers, idep_vulkan_util,
                    idep_vulkan_runtime, idep_mesautil,
                    dep_spirv_tools],
    link_with : [kraidoc_arch_libs, kraidoc_common, libpanfrost_compiler],
    c_args : [c_msvc_compat_args, no_override_init_args, soong_compat_c_args,
              '-DPAN_ARCH=10', '-DPANVK_OFFLINE_ONLY'],
    link_args : [ld_args_gc_sections],
    gnu_symbol_visibility : 'hidden',
    install : true,
  )
  if with_tests and prog_glslang.found()
    test(
      'kraidoc',
      prog_python,
      args : [files('test_kraidoc.py'), kraidoc, prog_glslang],
      suite : ['panfrost'],
      timeout : 120,
    )
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
