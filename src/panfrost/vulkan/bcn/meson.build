# Copyright © 2026 Pix Philosophy (HK) Limited
# SPDX-License-Identifier: MIT

panvk_bcn_spv = custom_target(
  'decode.spv.h',
  input : 'decode.comp',
  output : 'decode.spv.h',
  command : [
    prog_glslang, '-V', '--target-env', 'spirv1.5', '-x', '-o', '@OUTPUT@', '@INPUT@'
  ] + glslang_quiet,
)

panvk_bcn_3d_spv = custom_target(
  'decode_3d.spv.h',
  input : 'decode.comp',
  output : 'decode_3d.spv.h',
  command : [
    prog_glslang, '-V', '--target-env', 'spirv1.5', '-DBC_DECODE_3D', '-x', '-o', '@OUTPUT@', '@INPUT@'
  ] + glslang_quiet,
)

panvk_bcn_astc_spv = custom_target(
  'encode_astc.spv.h',
  input : 'decode.comp',
  output : 'encode_astc.spv.h',
  command : [
    prog_glslang, '-V', '--target-env', 'spirv1.5', '-DBC_ENCODE_ASTC', '-x', '-o', '@OUTPUT@', '@INPUT@'
  ] + glslang_quiet,
)
