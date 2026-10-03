import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


tool, glslang = sys.argv[1:3]
env = dict(os.environ, PAN_USE_KRAID="all")


def run(*args, success=True, extra_env=None):
    result = subprocess.run([tool, *map(str, args)], env=extra_env or env,
                            capture_output=True, text=True)
    assert (result.returncode == 0) == success, result.stderr + result.stdout
    return result


def fixture(root, name, source):
    path = root / name
    path.write_text(source)
    spv = path.with_suffix(path.suffix + ".spv")
    subprocess.run([glslang, "-V", "--target-env", "vulkan1.2", str(path),
                    "-o", str(spv)], check=True, capture_output=True)
    return spv


def compile_shader(root, spv, name, *args):
    prefix = root / name
    result = run(spv, "-o", prefix, "--format", "json", *args)
    report = json.loads(result.stdout)
    assert report == json.loads(Path(str(prefix) + ".report.json").read_text())
    assert report["compiler"] == "Kraid"
    assert report["original_pipeline_state"] is False
    assert len(report["input_blake3"]) == 64
    assert report["programs"]
    for program in report["programs"]:
        stem = f"{prefix}.{program['variant']}.{program['kind']}"
        binary = Path(stem + ".bin").read_bytes()
        assert len(binary) == program["binary_bytes"]
        assert len(program["binary_blake3"]) == 64
        assert program["instructions"] > 0
        assert Path(stem + ".isa.txt").read_text().strip()
    return report


with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    compute = fixture(root, "simple.comp", """#version 450
layout(local_size_x=32) in;
layout(constant_id=3) const uint value=7;
layout(set=0,binding=1) buffer Data { uint data[]; } output_data;
void main() { output_data.data[gl_GlobalInvocationID.x] = value; }
""")
    vertex = fixture(root, "simple.vert", """#version 450
layout(location=0) in vec4 position;
layout(location=0) out vec4 color;
void main() { gl_Position=position; color=position*2.0; }
""")
    fragment = fixture(root, "simple.frag", """#version 450
layout(location=0) in vec4 color;
layout(location=0) out vec4 result;
void main() { result=color; }
""")
    pilot = fixture(root, "pilot.comp", """#version 450
layout(local_size_x=32) in;
layout(set=0,binding=0) uniform Config { vec4 a; vec4 b; } config;
layout(set=0,binding=1) buffer Data { float data[]; } output_data;
void main() {
  float y=sin(config.a.x)*cos(config.a.y)+sqrt(abs(config.b.z));
  output_data.data[gl_GlobalInvocationID.x]=y+float(gl_GlobalInvocationID.x);
}
""")
    texture = fixture(root, "texture.frag", """#version 450
layout(set=0,binding=7) uniform sampler2D image;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 result;
void main() { result=texture(image,uv); }
""")
    separate_texture = fixture(root, "separate.frag", """#version 450
layout(set=0,binding=7) uniform texture2D image;
layout(set=1,binding=9) uniform sampler image_sampler;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 result;
void main() { result=texture(sampler2D(image,image_sampler),uv); }
""")
    targets = ["Mali-G57", "Mali-G68", "Mali-G610", "Mali-G710",
               "Mali-G310v1", "Mali-G310v5", "Mali-G615", "Mali-G715"]
    for target in targets:
        for spv in [compute, vertex, fragment]:
            compile_shader(root, spv, target + spv.suffixes[0], "-c", target)
    original = compile_shader(root, compute, "original", "--nir")
    assert original["bindings"][0]["type"] == 7
    assert Path(str(root / "original") + ".0.main.nir.txt").is_file()
    assert any(p["kind"] == "varying" for p in
               compile_shader(root, vertex, "idvs")["programs"])
    spec_prefix = 'spec with spaces' if os.name == 'nt' else 'spec "quoted"'
    specialized = compile_shader(root, compute, spec_prefix,
                                 "--spec", "3:4:99")
    assert specialized["specialization"] == [{"id": 3, "bytes": 4,
                                               "bits": "0x63"}]
    assert original["programs"][0]["binary_blake3"] != \
        specialized["programs"][0]["binary_blake3"]
    disabled = compile_shader(root, compute, "disabled", "--no-preamble")
    assert disabled["preamble_enabled"] is False
    assert all(p["kind"] != "pilot" for p in disabled["programs"])
    assert any(p["kind"] == "pilot" for p in
               compile_shader(root, pilot, "pilot")["programs"])
    assert compile_shader(root, texture, "texture")["bindings"] == [
        {"set": 0, "binding": 7, "type": 1, "count": 1,
         "source": "NIR reflection"}]
    separated = compile_shader(root, separate_texture, "separate")["bindings"]
    assert {(b["set"], b["binding"], b["type"]) for b in separated} == {
        (0, 7, 2), (1, 9, 0)}
    overridden = compile_shader(root, compute, "layout", "--binding",
                                "0:1:storage-buffer:8")
    assert overridden["bindings"][0]["source"] == "argument"
    assert overridden["bindings"][0]["count"] == 8
    for args in [("--stage", "vertex"), ("-n", "missing"),
                 ("--gpu-id", "0xc8000000"), ("-c", "Mali-G52"),
                 ("--shader-cores", "0"), ("--spec", "3:4:0x100000000"),
                 ("--spec", "3:3:7"),
                 ("--binding", "0:1:uniform-buffer:1"),
                 ("--binding", "0:1:storage-buffer:0")]:
        run(compute, *args, success=False)
    run(compute, "-o", root / "absent" / "output", success=False)
    run(compute, extra_env=dict(env, PAN_USE_KRAID="0"), success=False)
    run(fragment, extra_env=dict(env, PAN_USE_KRAID="cs,internal"), success=False)
    for data in [b"", b"\x03\x02\x23\x07", compute.read_bytes()[:-1],
                 struct.pack("<6I", 0x07230203, 0x10000, 0, 5, 0, 0)]:
        bad = root / "bad.spv"
        bad.write_bytes(data)
        run(bad, success=False)
    words = list(struct.unpack("<" + "I" * (compute.stat().st_size // 4),
                               compute.read_bytes()))
    i = 5
    while words[i] & 0xffff != 15:
        i += words[i] >> 16
    words[i:i] = words[i:i + (words[i] >> 16)]
    ambiguous = root / "ambiguous.spv"
    ambiguous.write_bytes(struct.pack("<" + "I" * len(words), *words))
    run(ambiguous, success=False)
    assert "Mali-G610" in run("--list").stdout
    assert "kraidoc" in run("--version").stdout
    print("kraidoc CPU tests passed")
