import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


class Blob(ctypes.Structure):
    _fields_ = [("vtable", ctypes.POINTER(ctypes.c_void_p))]


def blob_call(blob, index, result):
    return ctypes.WINFUNCTYPE(result, ctypes.POINTER(Blob))(
        blob.contents.vtable[index]
    )(blob)


def compile_dxbc(source, profile, output):
    dll = ctypes.WinDLL("d3dcompiler_47.dll")
    compile_shader = dll.D3DCompile
    compile_shader.restype = ctypes.c_long
    compile_shader.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint,
        ctypes.c_uint, ctypes.POINTER(ctypes.POINTER(Blob)),
        ctypes.POINTER(ctypes.POINTER(Blob)),
    ]
    code, errors = ctypes.POINTER(Blob)(), ctypes.POINTER(Blob)()
    data = source.read_bytes()
    status = compile_shader(data, len(data), b"fixture.hlsl", None, None,
                            b"main", profile.encode(), 0, 0,
                            ctypes.byref(code), ctypes.byref(errors))
    diagnostic = ""
    if errors:
        diagnostic = ctypes.string_at(blob_call(errors, 3, ctypes.c_void_p),
                                      blob_call(errors, 4, ctypes.c_size_t)).decode(errors="replace")
        blob_call(errors, 2, ctypes.c_ulong)
    if status < 0:
        raise RuntimeError(diagnostic)
    try:
        output.write_bytes(ctypes.string_at(blob_call(code, 3, ctypes.c_void_p),
                                           blob_call(code, 4, ctypes.c_size_t)))
    finally:
        blob_call(code, 2, ctypes.c_ulong)


def sdk_tool(name):
    found = shutil.which(name)
    if found:
        return found
    roots = [Path(os.environ.get("VULKAN_SDK", "C:/VulkanSDK/missing"))]
    roots += sorted(Path("C:/VulkanSDK").glob("*"), reverse=True)
    for root in roots:
        candidate = root / "Bin" / (name + ".exe")
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("Cannot locate " + name)


def run_checked(args, cwd):
    result = subprocess.run([str(arg) for arg in args], cwd=cwd,
                            capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout.decode(errors="replace") +
                           result.stderr.decode(errors="replace"))
    return result


def text_words(text):
    return [int(hi + lo, 16) for hi, lo, body in re.findall(
        r"^[ \t]*\d+\[([0-9a-fA-F]{8})_([0-9a-fA-F]{8})\]([^\r\n]*)",
        text, re.MULTILINE
    ) if body.strip() and not body.rstrip().endswith(":")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("both", "loadlibrary", "process"), default="both")
    args = parser.parse_args()
    backends = ("process", "loadlibrary") if args.backend == "both" else (args.backend,)
    tool = args.tool.resolve()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    fixtures = root / "shader inputs"
    fixtures.mkdir(exist_ok=True)
    d3d_inputs = root / "d3d_inputs"
    d3d_inputs.mkdir(exist_ok=True)

    sources = {
        "buffer.comp": """#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0,std430) buffer Data { float values[]; };
void main() { uint i=gl_GlobalInvocationID.x; values[i]=values[i]*2.0+1.0; }
""",
        "branch.comp": """#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0,std430) buffer Data { float values[]; };
void main() {
    uint i=gl_GlobalInvocationID.x;
    float x=values[i];
    for (uint j=0;j<(i&7u);++j) x=x>0.0 ? x*0.75+1.0 : -x;
    values[i]=x;
}
""",
        "compute.hlsl": """RWStructuredBuffer<float> values : register(u0);
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID) { values[id.x]=values[id.x]*2.0+1.0; }
""",
        "vertex.hlsl": """float4 main(float4 position:POSITION):SV_Position {
    return position*position.yzwx+position.wxyz;
}
""",
        "pixel.hlsl": """float4 main(float4 position:SV_Position):SV_Target {
    return position*position.yzwx+position.wxyz;
}
""",
    }
    for name, source in sources.items():
        (fixtures / name).write_text(source, encoding="utf-8")
    glslang, dxc = sdk_tool("glslangValidator"), sdk_tool("dxc")
    for name in ("buffer.comp", "branch.comp"):
        run_checked([glslang, "-V", fixtures / name, "-o", fixtures / (name + ".spv")], root)
    for name, profile in (("vertex", "vs_5_0"), ("pixel", "ps_5_0"), ("compute", "cs_5_0")):
        compile_dxbc(fixtures / (name + ".hlsl"), profile,
                     d3d_inputs / (name + "." + profile.split("_")[0] + ".dxbc"))
    shutil.copyfile(fixtures / "compute.hlsl", d3d_inputs / "compute.hlsl")
    run_checked([dxc, "-T", "cs_6_0", "-E", "main", d3d_inputs / "compute.hlsl",
                 "-Fo", d3d_inputs / "compute.cs.dxil"], root)
    root_signature = {
        "version": 1,
        "desc_1_0": {
            "num_parameters": 1,
            "parameters": [{"parameter_type": 4,
                            "descriptor": {"shader_register": 0, "register_space": 0},
                            "shader_visibility": 0}],
            "num_static_samplers": 0, "static_samplers": [], "flags": "0x0",
        },
    }
    dxil_pipeline = d3d_inputs / "dxil_pipeline.json"
    dxil_pipeline.write_text(json.dumps({
        "type": "compute", "shaders": {"compute": (d3d_inputs / "compute.cs.dxil").as_posix()},
        "root_signature": root_signature,
    }), encoding="utf-8")
    cases = []
    for arch in ("a650", "a740", "a830"):
        cases.append(("spv-buffer-" + arch, ["-arch=" + arch, fixtures / "buffer.comp.spv"]))
    for arch in ("a650", "a830"):
        cases.append(("spv-branch-" + arch, ["-arch=" + arch, fixtures / "branch.comp.spv"]))
    for arch in ("a690", "a741"):
        cases.append(("dxbc-vertex-" + arch, ["-arch=" + arch, d3d_inputs / "vertex.vs.dxbc"]))
        cases.append(("dxil-compute-" + arch, ["-arch=" + arch, "-link_info", dxil_pipeline]))
    cases.append(("dxbc-pixel-a741", ["-arch=a741", d3d_inputs / "pixel.ps.dxbc"]))
    cases.append(("dxbc-graphics-a741", ["-arch=a741", d3d_inputs / "vertex.vs.dxbc",
                                        d3d_inputs / "pixel.ps.dxbc"]))
    cases.append(("spv-dump-all", ["-arch=a830", "-dump=all", fixtures / "buffer.comp.spv"]))
    shutil.copyfile(d3d_inputs / "vertex.vs.dxbc", fixtures / "vertex.vs.dxbc")
    cases.append(("dxbc-spaces-a741", ["-arch=a741", fixtures / "vertex.vs.dxbc"]))
    shutil.copyfile(d3d_inputs / "compute.cs.dxil", fixtures / "compute.cs.dxil")
    dxil_spaces = d3d_inputs / "dxil_spaces.json"
    dxil_spaces.write_text(json.dumps({
        "type": "compute", "shaders": {"compute": (fixtures / "compute.cs.dxil").as_posix()},
        "root_signature": root_signature,
    }), encoding="utf-8")
    cases.append(("dxil-spaces-a741", ["-arch=a741", "-link_info", dxil_spaces]))
    results = []
    for backend in backends:
        for name, compiler_args in cases:
            target = root / backend / name
            target.mkdir(parents=True, exist_ok=True)
            text, vendor, raw = target / "mesa.isa", target / "aoc.log", target / "raw"
            result = run_checked([tool, "--backend", backend, "--output", text, "--aoc-log", vendor,
                                  "--raw-dir", raw, "--", *compiler_args], root)
            raw_files = sorted(raw.glob("*.bin"), key=lambda p: int(p.name.split("-")[0]))
            words = []
            for file in raw_files:
                data = file.read_bytes()
                if not data or len(data) % 8:
                    raise RuntimeError("Invalid raw ISA length: " + str(file))
                words.extend(struct.unpack("<" + "Q" * (len(data) // 8), data))
            mesa_text, vendor_text = text.read_text(), vendor.read_text()
            if words != text_words(mesa_text) or words != text_words(vendor_text):
                raise RuntimeError("Raw, Mesa and vendor instruction words differ: " + name)
            if ".quad" in mesa_text or "unmatched instruction" in mesa_text:
                raise RuntimeError("Unmatched instructions: " + name)
            if name.startswith("dxbc-vertex") and not any("binning" in p.name for p in raw_files):
                raise RuntimeError("Missing DXBC binning variant")
            if len(words) <= 1:
                raise RuntimeError("Fixture compiled to trivial ISA: " + name)
            record = {"backend": backend, "case": name, "variants": len(raw_files), "instructions": len(words),
                      "bytes": len(words) * 8, "stderr": result.stderr.decode().strip()}
            if backend == "loadlibrary" and "process" in backends:
                reference = root / "process" / name
                if (reference / "mesa.isa").read_bytes() != text.read_bytes():
                    raise RuntimeError("Process and LoadLibrary ISA text differs: " + name)
                for file in raw_files:
                    if (reference / "raw" / file.name).read_bytes() != file.read_bytes():
                        raise RuntimeError("Process and LoadLibrary raw ISA differs: " + name)
            results.append(record)
            print(json.dumps(record), flush=True)

    for backend in backends:
        for name in ("spv-buffer-a830", "dxbc-graphics-a741", "dxil-compute-a741"):
            compiler_args = next(command for case, command in cases if case == name)
            first = root / backend / name / "mesa.isa"
            repeated = root / backend / (name + "-repeated.isa")
            result = run_checked([tool, "--backend", backend, "--repeat", "20",
                                  "--output", repeated, "--", *compiler_args], root)
            if first.read_bytes() != repeated.read_bytes():
                raise RuntimeError("Repeated compilation differs: " + backend + " " + name)
            print(json.dumps({"backend": backend, "repeated": name,
                              "stderr": result.stderr.decode().strip()}), flush=True)

    if "loadlibrary" in backends:
        result = run_checked([tool, "--", "-arch=a830", fixtures / "buffer.comp.spv"], root)
        reference = root / "loadlibrary" / "spv-buffer-a830" / "mesa.isa"
        if result.stdout.decode().replace("\r\n", "\n") != reference.read_text():
            raise RuntimeError("Default LoadLibrary backend did not restore ISA stdout")
        print(json.dumps({"backend": "default", "stdout": "restored"}), flush=True)

    bad = fixtures / "invalid.comp.spv"
    bad.write_bytes(b"invalid SPIR-V")
    failures = [
        ("missing-input", ["--", fixtures / "missing.comp.spv"]),
        ("invalid-spirv", ["--", bad]),
        ("unsupported-binary", ["--aoc", tool, "--", fixtures / "buffer.comp.spv"]),
        ("unbound-dxbc-uav", ["--", "-arch=a741", d3d_inputs / "compute.cs.dxbc"]),
    ]
    for backend in backends:
        for name, command in failures:
            output = root / backend / (name + ".isa")
            output.unlink(missing_ok=True)
            result = subprocess.run([str(tool), "--backend", backend, "--output", str(output), *map(str, command)],
                                    cwd=root, capture_output=True, timeout=120)
            if result.returncode != 1 or output.exists():
                raise RuntimeError("Failure was not rejected cleanly: " + name)
            print(json.dumps({"backend": backend, "failure": name, "exit": result.returncode}), flush=True)
    if list(root.glob("aoc-isa-*.json")) or list(root.glob("aoc-isa-*.dxbc")):
        raise RuntimeError("Temporary compiler inputs were not removed")
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("PASS: raw/vendor/Mesa ISA equality, backend parity, repeated teardown and failure paths")


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("This integration test requires Windows")
    main()
