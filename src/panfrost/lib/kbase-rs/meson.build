# Copyright © 2026 Pix Philosophy (HK) Limited
# SPDX-License-Identifier: MIT

dep_rust_libc = dependency('rust-libc',
  version : '>= 0.2.168',
  fallback : ['libc-0.2-rs', 'dep_libc'],
  required : true,
)

libpankmod_kbase = static_library(
  'pankmod_kbase',
  files('lib.rs'),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'c',
  rust_args : [panfrost_rust_args],
  dependencies : [dep_rust_libc],
)
