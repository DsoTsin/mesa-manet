# Copyright © 2026 Pix Philosophy (HK) Limited
# SPDX-License-Identifier: MIT

panvk_bvh_shaders = [
  ['encode.comp', 'encode'],
  ['copy.comp', 'copy'],
]

panvk_bvh_include_dir = dir_source_root + '/src/panfrost/vulkan/bvh'

panvk_bvh_includes = files('panvk_bvh.h')

panvk_bvh_spv = []
foreach s : panvk_bvh_shaders
  command = [
    prog_glslang, '-V', '-I' + vk_bvh_include_dir, '-I' + panvk_bvh_include_dir,
    '--target-env', 'spirv1.5', '-x', '-o', '@OUTPUT@', '@INPUT@'
  ]
  command += glslang_quiet
  command += vk_glsl_shader_preamble

  panvk_bvh_spv += custom_target(
    s[1] + '.spv.h',
    input : s[0],
    output : s[1] + '.spv.h',
    command : command,
    depend_files : [vk_bvh_includes, panvk_bvh_includes],
  )
endforeach
