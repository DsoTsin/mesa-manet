nvkmd_wddm_rust = custom_target(
  'nvkmd_wddm_rust',
  output : 'nvkmd_wddm.lib',
  command : [
    prog_python, files('build_rust.py'),
    '--manifest', meson.current_source_dir() / 'Cargo.toml',
    '--target-dir', meson.current_build_dir() / 'cargo',
    '--output', '@OUTPUT@',
  ],
  depend_files : files('Cargo.toml', 'Cargo.lock'),
  build_always_stale : true,
)

nvkmd_wddm_deps = declare_dependency(
  link_with : nvkmd_wddm_rust,
  dependencies : [
    cc.find_library('kernel32'),
    cc.find_library('ntdll'),
    cc.find_library('gdi32'),
    cc.find_library('user32'),
    cc.find_library('advapi32'),
    cc.find_library('bcrypt'),
    cc.find_library('userenv'),
    cc.find_library('ws2_32'),
  ],
)
