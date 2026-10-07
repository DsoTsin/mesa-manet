#!/usr/bin/env python3

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile


PACKAGE = "mesa-manet-rock5b"
PREFIX = "/opt/mesa-manet"
LIBDIR = "lib/aarch64-linux-gnu"


def run(argv, **kwargs):
    print("+ " + shlex.join(map(str, argv)), flush=True)
    return subprocess.run(list(map(str, argv)), check=True, **kwargs)


def capture(argv, **kwargs):
    return run(argv, stdout=subprocess.PIPE, text=True, **kwargs).stdout.strip()


def write(path, value, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")
    if executable:
        path.chmod(0o755)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_info(source):
    snapshot = source / "manet-source.json"
    if snapshot.exists():
        return json.loads(snapshot.read_text(encoding="utf-8"))
    commit = capture(["git", "-C", source, "rev-parse", "HEAD"])
    status = capture(["git", "-C", source, "status", "--porcelain"])
    epoch = int(capture(["git", "-C", source, "show", "-s", "--format=%ct", "HEAD"]))
    return {"commit": commit, "dirty": bool(status), "status": status, "epoch": epoch}


def remote_build(args, source, output):
    info = source_info(source)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job = args.remote_root.rstrip("/") + "/" + stamp
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", job) or ".." in job.split("/"):
        raise ValueError("--remote-root must be an absolute path without spaces or '..'")
    ssh = ["ssh", "-4", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host]
    run(ssh + ["mkdir -p " + shlex.quote(job + "/source")])
    files = run(["git", "-C", source, "ls-files", "-z", "--cached", "--others",
                 "--exclude-standard"], stdout=subprocess.PIPE).stdout.split(b"\0")
    with tempfile.TemporaryDirectory(prefix="manet-source-") as temporary:
        archive = Path(temporary) / "source.tar.gz"
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            for raw in sorted(set(files)):
                if not raw:
                    continue
                name = raw.decode("utf-8")
                path = source / name
                if path.is_file() or path.is_symlink():
                    tar.add(path, arcname=name, recursive=False)
            metadata = Path(temporary) / "manet-source.json"
            write(metadata, json.dumps(info, indent=2) + "\n")
            tar.add(metadata, arcname=metadata.name)
        run(["scp", "-q", archive, args.host + ":" + job + "/source.tar.gz"])
    run(ssh + [shlex.join(["tar", "-xzf", job + "/source.tar.gz", "-C", job + "/source"])])
    command = ["python3", job + "/source/bin/package-manet.py", "--source", job + "/source",
               "--work-dir", job + "/work", "--output", job + "/output", "--jobs", str(args.jobs)]
    if args.version:
        command += ["--version", args.version]
    if args.bootstrap_glvnd:
        command += ["--bootstrap-glvnd"]
    run(ssh + [shlex.join(command)])
    run(["scp", "-q", args.host + ":" + job + "/output/*", output])
    write(output / "remote-build.json", json.dumps({"host": args.host, "directory": job,
          "source": info}, indent=2) + "\n")


def build_environment(work, bootstrap):
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + env["PATH"]
    if "LIBCLANG_PATH" not in env:
        candidates = sorted(Path("/usr/lib").glob("llvm-*/lib/libclang.so"),
                            key=lambda path: int(path.parents[1].name.split("-")[1]))
        if candidates:
            env["LIBCLANG_PATH"] = str(candidates[-1].parent)
    found = subprocess.run(["pkg-config", "--exists", "libglvnd"], env=env).returncode == 0
    if not found and bootstrap:
        deps = work / "glvnd-build-deps"
        deps.mkdir(parents=True, exist_ok=True)
        run(["apt-get", "download", "libglvnd-core-dev"], cwd=deps)
        for package in deps.glob("libglvnd-core-dev_*.deb"):
            run(["dpkg-deb", "--extract", package, deps / "root"])
        pc_dirs = set()
        for pc in (deps / "root").rglob("*.pc"):
            content = pc.read_text().replace("prefix=/usr", "prefix=" + str(deps / "root/usr"))
            write(pc, content)
            pc_dirs.add(str(pc.parent))
        env["PKG_CONFIG_PATH"] = os.pathsep.join(sorted(pc_dirs) + [env.get("PKG_CONFIG_PATH", "")])
    run(["pkg-config", "--atleast-version=1.3.2", "libglvnd"], env=env)
    return env


def runtime_files(stage):
    return sorted(path for path in stage.rglob("*") if path.is_file() and not path.is_symlink())


def package_runtime(args, source, work, output, build, env, info, version):
    with tempfile.TemporaryDirectory(prefix="package-", dir=work) as temporary:
        temporary = Path(temporary)
        stage = temporary / "stage"
        run(["meson", "install", "-C", build, "--no-rebuild", "--tags", "runtime",
             "--destdir", stage], env=env)
        install = stage / PREFIX.lstrip("/")
        libraries = install / LIBDIR
        manifests = list((install / "share/vulkan/icd.d").glob("manet_icd*.json"))
        if len(manifests) != 1:
            raise RuntimeError("Expected exactly one Manet Vulkan ICD manifest")
        icd = manifests[0]
        data = json.loads(icd.read_text())
        data["ICD"]["library_path"] = PREFIX + "/" + LIBDIR + "/libvulkan_manet.so"
        write(icd, json.dumps(data, indent=2) + "\n")
        egl_manifest = install / "share/glvnd/egl_vendor.d/50_manet.json"
        egl_data = json.loads(egl_manifest.read_text())
        egl_data["ICD"]["library_path"] = PREFIX + "/" + LIBDIR + "/libEGL_manet.so.0"
        write(egl_manifest, json.dumps(egl_data, indent=2) + "\n")
        system_icd = stage / "usr/share/vulkan/icd.d" / icd.name
        system_icd.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(icd, system_icd)
        system_egl = stage / "usr/share/glvnd/egl_vendor.d/10_manet.json"
        system_egl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(egl_manifest, system_egl)
        write(stage / "etc/ld.so.conf.d/00-mesa-manet.conf", PREFIX + "/" + LIBDIR + "\n")
        required = [libraries / "libvulkan_manet.so", libraries / "libEGL_manet.so.0",
                    libraries / "libGLX_manet.so.0", libraries / "dri/zink_dri.so",
                    libraries / "libgbm.so.1", install / "share/glvnd/egl_vendor.d/50_manet.json"]
        for path in required:
            if not path.is_file():
                raise RuntimeError("Missing runtime file: " + str(path))
        if list(stage.rglob("libvulkan_panfrost*")):
            raise RuntimeError("Obsolete Vulkan library in package")
        env_text = (
            f'export VK_DRIVER_FILES={PREFIX}/share/vulkan/icd.d/{icd.name}\n'
            'export VK_ICD_FILENAMES="$VK_DRIVER_FILES"\n'
            f'export LD_LIBRARY_PATH="{PREFIX}/{LIBDIR}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"\n'
            f'export LIBGL_DRIVERS_PATH={PREFIX}/{LIBDIR}/dri\n'
            'export __GLX_VENDOR_LIBRARY_NAME=manet\n'
            f'export __EGL_VENDOR_LIBRARY_FILENAMES={PREFIX}/share/glvnd/egl_vendor.d/50_manet.json\n'
            'export MESA_LOADER_DRIVER_OVERRIDE=zink\n'
            'export GALLIUM_DRIVER=zink\n'
            'export PAN_USE_KRAID="${PAN_USE_KRAID-all}"\n'
        )
        write(install / "env.sh", env_text)
        desktop_environment = {
            "VK_DRIVER_FILES": PREFIX + "/share/vulkan/icd.d/" + icd.name,
            "VK_ICD_FILENAMES": PREFIX + "/share/vulkan/icd.d/" + icd.name,
            "LD_LIBRARY_PATH": PREFIX + "/" + LIBDIR,
            "LIBGL_DRIVERS_PATH": PREFIX + "/" + LIBDIR + "/dri",
            "__GLX_VENDOR_LIBRARY_NAME": "manet",
            "__EGL_VENDOR_LIBRARY_FILENAMES": PREFIX + "/share/glvnd/egl_vendor.d/50_manet.json",
            "MESA_LOADER_DRIVER_OVERRIDE": "zink",
            "GALLIUM_DRIVER": "zink",
            "PAN_USE_KRAID": "all",
        }
        environment_file = "".join(f"{key}={value}\n" for key, value in desktop_environment.items())
        write(install / "environment", environment_file)
        write(stage / "usr/lib/environment.d/90-mesa-manet.conf", environment_file)
        session_env = f"if [ -r {PREFIX}/env.sh ]; then\n    . {PREFIX}/env.sh\nfi\n"
        write(stage / "etc/profile.d/mesa-manet.sh", session_env)
        write(stage / "etc/X11/Xsession.d/20mesa-manet", session_env)
        write(stage / "usr/lib/systemd/system/display-manager.service.d/50-mesa-manet.conf",
              "[Service]\nEnvironmentFile=-" + PREFIX + "/environment\n")
        write(stage / "usr/lib/udev/rules.d/70-mesa-manet.rules",
              'SUBSYSTEM=="misc", KERNEL=="mali[0-9]*", TAG+="uaccess"\n')
        write(stage / "usr/bin/manet-run", '#!/bin/sh\nset -eu\n'
              'if [ "$#" -eq 0 ]; then\n'
              '    printf "%s\\n" "Usage: manet-run PROGRAM [ARGUMENT ...]" >&2\n'
              '    exit 2\nfi\n'
              f'. {PREFIX}/env.sh\nexec "$@"\n', executable=True)
        documentation = stage / "usr/share/doc" / PACKAGE
        write(documentation / "copyright", (source / "docs/license.rst").read_text(encoding="utf-8"))
        shutil.copytree(source / "licenses", documentation / "licenses")
        shutil.copy2(source / "bin/package-manet.py", documentation / "package-manet.py")
        shutil.copy2(source / "docs/drivers/manet-packaging.md", documentation / "README.md")
        elf_files = []
        for path in runtime_files(stage):
            with path.open("rb") as stream:
                if stream.read(4) == b"\x7fELF":
                    elf_files.append(path)
        needed = {}
        private_shlibs = []
        for path in elf_files:
            header = capture(["readelf", "-h", path])
            if "AArch64" not in header:
                raise RuntimeError("Non-arm64 ELF: " + str(path))
            dynamic = capture(["readelf", "-d", path])
            needed[str(path.relative_to(stage))] = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
            soname = re.search(r"\(SONAME\).*?\[(.*?)\]", dynamic)
            if soname:
                match = (re.fullmatch(r"(.+)\.so\.(.+)", soname[1]) or
                         re.fullmatch(r"(.+)-(\d.*)\.so", soname[1]))
                if match:
                    private_shlibs.append(f"{match[1]} {match[2]} {PACKAGE} (= {version})")
        write(temporary / "debian/control", f"Source: {PACKAGE}\nMaintainer: Mesa Manet <mesa-manet@localhost>\n\n"
              f"Package: {PACKAGE}\nArchitecture: arm64\nDescription: Manet Vulkan and Zink OpenGL drivers\n")
        write(temporary / "debian/shlibs.local", "\n".join(private_shlibs) + "\n")
        deps = capture(["dpkg-shlibdeps", "-O", "-x" + PACKAGE, "-l" + str(libraries),
                        "-l" + str(libraries / "dri"), *["-e" + str(path) for path in elf_files]],
                       cwd=temporary, env=env)
        deps = next(line.split("=", 1)[1] for line in deps.splitlines() if line.startswith("shlibs:Depends="))
        dependencies = sorted(set(filter(None, deps.split(", "))) |
                              {"libvulkan1", "libegl1", "libglx0", "libgl1", "libgles2"})
        manifest = {"package": PACKAGE, "version": version, "architecture": "arm64", "source": info,
                    "os_release": Path("/etc/os-release").read_text(), "prefix": PREFIX,
                    "dependencies": dependencies, "needed": needed,
                    "meson_options": json.loads((build / "meson-info/intro-buildoptions.json").read_text()),
                    "sha256": {str(path.relative_to(stage)): sha256(path) for path in elf_files}}
        write(documentation / "build-manifest.json", json.dumps(manifest, indent=2) + "\n")
        installed_kb = sum(path.stat().st_size for path in runtime_files(stage)) // 1024 + 1
        write(stage / "DEBIAN/control", f"Package: {PACKAGE}\nVersion: {version}\nArchitecture: arm64\n"
              "Maintainer: Mesa Manet <mesa-manet@localhost>\nSection: graphics\nPriority: optional\n"
              f"Installed-Size: {installed_kb}\nDepends: {', '.join(dependencies)}\n"
              "Description: Manet Vulkan and Zink OpenGL drivers for Rock 5B\n"
              " Vulkan, EGL and GLX runtime with Zink and the Kraid shader compiler.\n"
              " Enabled for desktop sessions after reboot; manet-run selects it immediately.\n"
              " Build provenance: biblioklept.\n")
        write(stage / "DEBIAN/conffiles", "\n".join("/" + path.relative_to(stage).as_posix()
              for path in sorted((stage / "etc").rglob("*")) if path.is_file()) + "\n")
        for hook in ("postinst", "postrm"):
            write(stage / "DEBIAN" / hook, '#!/bin/sh\nset -e\n'
                  'ldconfig\n'
                  'if [ -d /run/systemd/system ]; then\n'
                  '    systemctl daemon-reload\nfi\n'
                  'if command -v udevadm >/dev/null 2>&1; then\n'
                  '    udevadm control --reload-rules || true\nfi\n', executable=True)
        checksums = []
        for path in runtime_files(stage):
            if "DEBIAN" not in path.relative_to(stage).parts:
                with path.open("rb") as stream:
                    checksums.append(hashlib.file_digest(stream, "md5").hexdigest() + "  " + path.relative_to(stage).as_posix())
        write(stage / "DEBIAN/md5sums", "\n".join(checksums) + "\n")
        package = output / f"{PACKAGE}_{version}_arm64.deb"
        run(["dpkg-deb", "--root-owner-group", "--build", "-Zxz", "-z6", stage, package], env=env)
        run(["dpkg-deb", "--info", package])
        manifest["deb"] = {"file": package.name, "sha256": sha256(package)}
        write(output / "build-manifest.json", json.dumps(manifest, indent=2) + "\n")
        write(output / "SHA256SUMS", manifest["deb"]["sha256"] + "  " + package.name + "\n")
        shutil.copy2(source / "bin/package-manet.py", output / "package-manet.py")
        shutil.copy2(source / "docs/drivers/manet-packaging.md", output / "README.md")
        print("PACKAGE: " + str(package), flush=True)


def native_build(args, source, output):
    if sys.platform != "linux" or platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("Native packaging requires arm64 Debian/Ubuntu; use --host zq@rock-5b")
    work = Path(args.work_dir or source / "build/manet-package").resolve()
    work.mkdir(parents=True, exist_ok=True)
    info = source_info(source)
    upstream = (source / "VERSION").read_text().strip().replace("-devel", "~devel")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
    version = args.version or upstream + "+git" + info["commit"][:12] + "." + stamp
    if not re.fullmatch(r"[0-9][A-Za-z0-9.+:~\-]*", version):
        raise ValueError("Invalid Debian package version")
    env = build_environment(work, args.bootstrap_glvnd)
    env.setdefault("SOURCE_DATE_EPOCH", str(info["epoch"]))
    build = work / "build"
    options = ["--prefix=" + PREFIX, "--libdir=" + LIBDIR, "--sysconfdir=" + PREFIX + "/etc",
               "--buildtype=release", "-Dstrip=true", "-Db_lto=true",
               "-Db_lto_threads=" + str(args.jobs), "-Dallow-broken-lto=true", "-Dgallium-drivers=zink",
               "-Dvulkan-drivers=panfrost", "-Dpanfrost-kmds=kbase,panthor", "-Dpanfrost-rust=true",
               "-Dplatforms=x11,wayland", "-Dglx=dri", "-Degl=enabled", "-Dgbm=enabled",
               "-Dopengl=true", "-Dgles1=disabled", "-Dgles2=enabled", "-Dglvnd=enabled",
               "-Dglvnd-vendor-name=manet", "-Dllvm=enabled", "-Dshared-llvm=enabled",
               "-Ddraw-use-llvm=false", "-Dmesa-clc=enabled", "-Dprecomp-compiler=enabled",
               "-Dgallium-va=disabled", "-Dvulkan-layers=",
               "-Dtools=", "-Dvalgrind=disabled", "-Dlibunwind=disabled", "-Dbuild-tests=false"]
    command = ["meson", "setup"]
    if (build / "meson-private/coredata.dat").exists():
        command += ["--reconfigure"]
    run(command + [build, source] + options, env=env)
    run(["ninja", "-C", build, "-j", str(args.jobs)], env=env)
    package_runtime(args, source, work, output, build, env, info, version)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--work-dir")
    parser.add_argument("--host")
    parser.add_argument("--remote-root", default="/home/zq/fmcmp/manet-packaging")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--version")
    parser.add_argument("--bootstrap-glvnd", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    output = (args.output or source / "build/packages").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.host:
        remote_build(args, source, output)
    else:
        native_build(args, source, output)


if __name__ == "__main__":
    main()
