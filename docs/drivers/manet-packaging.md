# Rock 5B Manet runtime package

One `mesa-manet-rock5b` arm64 deb installs `libvulkan_manet.so`, Zink,
`libEGL_manet.so.0`, `libGLX_manet.so.0`, GBM, and the Kraid compiler runtime.
Libraries and manifests live in `/opt/mesa-manet`. The deb configures the display
manager, systemd user environment, X11 sessions and login shells to select Manet
after reboot. `/usr/bin/manet-run` selects the package immediately for Vulkan,
OpenGL and OpenGL ES applications. Kraid is enabled by
default through `PAN_USE_KRAID=all`; an explicitly set value takes precedence.
Both drivers use Release builds with LTO (`b_lto=true`, `b_lto_threads=6` by
default). The script explicitly enables Mesa's `allow-broken-lto` build option,
which is required by this Mesa branch to accept LTO. LTO jobs follow `--jobs`.
System-wide Vulkan and EGL manifests are installed under
`/usr/share/vulkan/icd.d` and `/usr/share/glvnd/egl_vendor.d`.
`/etc/ld.so.conf.d/00-mesa-manet.conf` registers the runtime libraries with
`ldconfig`, including the private Gallium library and packaged GBM. This also
makes the runtime discoverable when a program does not inherit session driver
environment variables.

## Build

From the Windows checkout, using an arm64 Debian/Ubuntu build host:

```powershell
python bin/package-manet.py --host zq@rock-5b --bootstrap-glvnd
```

The script transfers the current tracked source and unignored new files to a
new remote directory, builds there, and downloads the deb, SHA256SUMS and build
manifest into `build/packages`. It does not install the package. Use `--output`
to select another artifact directory, `--remote-root` for the remote workspace,
and `--jobs` to change the default six build jobs.
When using the copy of the script shipped beside or inside the deb, pass
`--source /path/to/mesa-manet` to locate the source checkout explicitly.

On an arm64 Debian/Ubuntu machine with the checkout:

```sh
python3 bin/package-manet.py --bootstrap-glvnd
```

Build prerequisites include Python 3.11+, Meson, Ninja, a C/C++ toolchain,
Rust/bindgen, LLVM/Clang development files, pkg-config, dpkg-dev, libdrm,
libelf, zlib, zstd, expat, X11/XCB, Wayland and GLVND development packages.
`--bootstrap-glvnd` downloads and extracts `libglvnd-core-dev` into the build
directory when GLVND headers are missing. Other dependencies must already be
installed. No root access is needed to build or package.

The deb depends on runtime packages discovered from the built ELF files, plus
the Vulkan and GLVND loaders. Build on the oldest Debian/Ubuntu release you
intend to support. A package built on Ubuntu 26.04 is not automatically
compatible with older Ubuntu or Debian releases.

## Install and use

```sh
sudo apt install ./mesa-manet-rock5b_*_arm64.deb
sudo reboot
```

After logging back into the graphical desktop:

```sh
vulkaninfo --summary
glxinfo -B
```

The Vulkan device should be the Mali GPU; OpenGL's renderer should identify
Zink on that GPU. A renderer containing `llvmpipe` or `softpipe` means software
rendering is still selected. Installing the libraries alone does not prove
that a particular desktop compositor can use them.

Before reboot, select the new runtime explicitly:

```sh
manet-run vulkaninfo --summary
manet-run glxinfo -B
manet-run your-application
```

For applications launched from the same shell:

```sh
. /opt/mesa-manet/env.sh
```

The kernel must expose the supported kbase GPU device (`/dev/mali0`) or a
supported Panthor DRM device, accessible to the application user. GLX requires
an available X server; EGL also includes Wayland and surfaceless support.
The package does not replace the kernel driver. Its desktop configuration is
installed under `/usr/lib/environment.d`, `/etc/profile.d`,
`/etc/X11/Xsession.d` and `/usr/lib/systemd/system/display-manager.service.d`.
The two shell session files are dpkg-managed conffiles and check that the
runtime still exists before loading it.
Installation reloads systemd configuration without restarting the active
desktop. A udev rule grants the active local seat access to `/dev/mali0`.

Restore the previous desktop defaults with
`sudo apt purge mesa-manet-rock5b`, then reboot. Use purge so that dpkg removes
the desktop configuration files too. The Python packaging script,
this guide and the build manifest are also installed under
`/usr/share/doc/mesa-manet-rock5b/`.

AI-assisted build documentation provenance: biblioklept.

## GNOME System Details detection

The first package relied entirely on environment variables for driver discovery.
GNOME Control Center 50.3 starts its renderer helper with only `DISPLAY`,
`WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR` and optional Switcheroo variables. Its
[helper launcher](https://github.com/GNOME/gnome-control-center/blob/50.3/panels/system/about/cc-system-details-window.c)
therefore discards the package's Vulkan, EGL and shared-library overrides.

On Rock 5B, the desktop process had loaded Manet and `glxinfo -B` reported
`Accelerated: yes` with Zink/Mali-G610. The same GNOME helper with the reduced
environment returned llvmpipe, displayed by System Details as Software
Rendering. System-wide ICD/EGL registration and the dynamic-linker configuration
address this packaging defect.

A second reproduced failure occurred with both Manet and Lavapipe available:
Zink's existing fallback required exactly one Vulkan physical device after DRM
matching failed. A CPU Vulkan device prevented selection of the hardware GPU.
The fallback now accepts a unique non-CPU device; exact DRM/LUID matching keeps
precedence, explicit software rendering keeps the CPU path, and ambiguous
hardware devices remain unselected.

Validation must run the original GNOME helper with the reduced environment,
including the Switcheroo `DRI_PRIME` values, as well as normal GL/Vulkan queries.
The Settings process must be restarted after upgrading; closing and reopening
only the System Details dialog retained the earlier result on this machine.

The corrected package `26.3.0~devel+git0553ede44c10.20260927094937` was installed
and validated on Rock 5B. The original GNOME helper returned Zink/Mali-G610 for
all three reduced environments: no `DRI_PRIME`, `platform-display-subsystem`,
and `platform-fdab0000_npu`. A Vulkan query without driver overrides enumerated
both Manet/Mali-G610 and Lavapipe. After restarting Settings, both graphics rows
visibly showed `zink Vulkan 1.4(Mali-G610 MC4)`.
Local evidence is in `build/desktop-diagnosis/`: before/after screenshots,
helper output, accessibility text and the rebuild log. The corrected deb is in
`build/packages-fixed/`. Its seven installed ELF hashes matched the build
manifest; the display manager remained active throughout the upgrade.
