import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile


tool, glslang = sys.argv[1:3]
tool = str(Path(tool).resolve())
if Path(glslang).is_file():
    glslang = str(Path(glslang).resolve())
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
    result = run(spv, "-o", prefix, "--format", "json",
                 "--report-format", "kraid", *args)
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


def shader_isa(root, spv, name, *args):
    compile_shader(root, spv, name, "--no-preamble", "--nir", *args)
    return Path(str(root / name) + ".0.main.isa.txt").read_text()


def add_float_controls(root, spv, name, rtz, width=16):
    data = spv.read_bytes()
    words = list(struct.unpack("<" + "I" * (len(data) // 4), data))
    i = 5
    while words[i] & 0xffff == 17:
        i += words[i] >> 16
    rounding_capability = 4468 if rtz else 4467
    words[i:i] = [2 << 16 | 17, 4464, 2 << 16 | 17, 4466,
                  2 << 16 | 17, rounding_capability]
    while words[i] & 0xffff != 15:
        i += words[i] >> 16
    entry = words[i + 2]
    while words[i] & 0xffff != 16:
        i += words[i] >> 16
    rounding_mode = 4463 if rtz else 4462
    words[i:i] = [4 << 16 | 16, entry, 4459, width,
                  4 << 16 | 16, entry, 4461, width,
                  4 << 16 | 16, entry, rounding_mode, width]
    output = root / name
    output.write_bytes(struct.pack("<" + "I" * len(words), *words))
    return output


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
    frag_coord = fixture(root, "fragcoord.frag", """#version 450
layout(location=0) out vec4 result;
void main() { result=vec4(gl_FragCoord.z,gl_FragCoord.w,gl_PointCoord); }
""")
    targets = ["Mali-G57", "Mali-G68", "Mali-G610", "Mali-G710",
               "Mali-G310v1", "Mali-G310v5", "Mali-G615", "Mali-G715"]
    for target in targets:
        for spv in [compute, vertex, fragment]:
            compile_shader(root, spv, target + spv.suffixes[0], "-c", target)
    fp16_source = """#version 450
layout(local_size_x=32) in;
layout(set=0,binding=0) buffer Data { vec4 data[]; } values;
void main() {
  uint i=gl_GlobalInvocationID.x;
  PRECISION vec2 a=values.data[i].xy;
  PRECISION vec2 b=values.data[i].zw;
  PRECISION vec2 s=a+b;
  PRECISION vec2 f=fma(a,b,vec2(0.5));
  values.data[i]=vec4(s,f);
}
"""
    mediump = fixture(root, "mediump.comp",
                      fp16_source.replace("PRECISION", "mediump"))
    highp = fixture(root, "highp.comp",
                    fp16_source.replace("PRECISION", "highp"))
    fp16 = fixture(root, "fp16.comp", fp16_source.replace(
        "#version 450", "#version 450\n#extension "
        "GL_EXT_shader_explicit_arithmetic_types_float16 : require").replace(
        "PRECISION vec2 a=values.data[i].xy",
        "f16vec2 a=f16vec2(values.data[i].xy)").replace(
        "PRECISION vec2 b=values.data[i].zw",
        "f16vec2 b=f16vec2(values.data[i].zw)").replace(
        "PRECISION vec2", "f16vec2").replace("vec2(0.5)", "f16vec2(0.5)").replace(
        "vec4(s,f)", "vec4(vec2(s),vec2(f))"))
    precise = fixture(root, "precise.comp", fp16_source.replace(
        "PRECISION", "mediump").replace(
        "mediump vec2 f=fma(a,b,vec2(0.5))",
        "precise mediump vec2 f=a*b+vec2(0.5)"))
    scalar = fixture(root, "scalar-fp16.comp", fp16_source.replace(
        "PRECISION", "mediump").replace("vec2", "float").replace(
        ".xy", ".x").replace(".zw", ".y").replace(
        "vec4(s,f)", "vec4(s,f,0.0,0.0)"))
    derivatives = fixture(root, "mediump-derivatives.frag", """#version 450
layout(location=0) in mediump vec2 uv;
layout(location=0) out vec4 result;
void main() { result=vec4(dFdx(uv),dFdy(uv)); }
""")
    fp16_math = fixture(root, "mediump-math.comp", """#version 450
layout(local_size_x=32) in;
layout(set=0,binding=0) buffer Data { vec4 data[]; } values;
void main() {
  uint i=gl_GlobalInvocationID.x;
  mediump vec2 a=abs(values.data[i].xy);
  mediump vec2 b=values.data[i].zw;
  mediump vec2 c=clamp(a,b,vec2(16.0))+fract(a)+floor(b);
  mediump vec2 d=sqrt(a)+inversesqrt(a)+vec2(1.0)/a;
  mediump vec2 e=exp2(a)+log2(a)+pow(a,b)+sin(a)+cos(b);
  values.data[i]=vec4(c,d+e);
}
""")
    fp16_rte = add_float_controls(root, fp16, "fp16-rte.spv", False)
    fp16_rtz = add_float_controls(root, fp16, "fp16-rtz.spv", True)
    mixed = fixture(root, "fp16-mixed.comp", """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x=32) in;
layout(set=0,binding=0) buffer Data { vec4 data[]; } values;
void main() {
  uint i=gl_GlobalInvocationID.x;
  float s=values.data[i].x+values.data[i].y;
  float16_t h=float16_t(s);
  float16_t c=clamp(h+float16_t(values.data[i].z),float16_t(0),float16_t(1));
  values.data[i]=vec4(s,float(h),float(c),float(clamp(c,float16_t(0),float16_t(1))));
}
""")
    different_rounds = add_float_controls(root, mixed, "fp16-mixed-rte.spv", False)
    different_rounds = add_float_controls(root, different_rounds,
                                          "fp16-mixed-rounds.spv", True, 32)
    for target in targets:
        for name, spv in [("mediump", mediump), ("explicit", fp16),
                          ("scalar", scalar), ("rte", fp16_rte),
                          ("rtz", fp16_rtz)]:
            isa = shader_isa(root, spv, name + target, "-c", target)
            if name != "scalar":
                assert "FADD.v2f16" in isa, isa
            assert re.search(r"^FMA\.v2f16[^\n]+(?<!neg)$", isa, re.M), isa
            assert "F16_TO_F32" in isa, isa
            if name == "rtz":
                assert re.search(r"^FADD\.v2f16\.round_zero", isa, re.M), isa
                assert re.search(r"^FMA\.v2f16\.round_zero", isa, re.M), isa
            elif name == "rte":
                assert "round_zero" not in isa, isa
        isa = shader_isa(root, highp, "highp" + target, "-c", target)
        assert not re.search(r"\.(?:v2)?f16\b", isa), isa
        isa = shader_isa(root, precise, "precise" + target, "-c", target)
        multiplies = re.findall(r"^FMA\.v2f16[^\n]+", isa, re.M)
        assert multiplies and all("k0.h00.neg" in op for op in multiplies), isa
        isa = shader_isa(root, derivatives, "derivatives" + target, "-c", target)
        assert "CLPER" in isa and "FADD.f32" in isa, isa
        assert not re.search(r"\.(?:v2)?f16\b", isa), isa
        isa = shader_isa(root, fp16_math, "math" + target, "-c", target)
        assert "FRCP.f16" in isa and "FRSQ.f16" in isa, isa
        assert "FMIN.v2f16" in isa and "FMAX.v2f16" in isa, isa
        nir = Path(str(root / ("math" + target)) + ".0.main.nir.txt").read_text()
        for op in ["fexp2", "flog2", "fpow", "fsin", "fcos"]:
            assert re.search(r"32\s+%\d+ = " + op + r"\b", nir), nir
        isa = shader_isa(root, mixed, "mixed" + target, "-c", target)
        assert re.search(r"^FADD\.f32\S* r\d+,", isa, re.M), isa
        isa = shader_isa(root, different_rounds, "rounds" + target, "-c", target)
        assert re.search(r"^FADD\.f32\.round_zero\S* r\d+,", isa, re.M), isa
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
    special = compile_shader(root, frag_coord, "fragcoord")
    special_isa = Path(str(root / "fragcoord") + ".0.main.isa.txt").read_text()
    assert special["programs"] and "LD_VAR_SPECIAL" in special_isa
    overridden = compile_shader(root, compute, "layout", "--binding",
                                "0:1:storage-buffer:8")
    assert overridden["bindings"][0]["source"] == "argument"
    assert overridden["bindings"][0]["count"] == 8
    for spv, selector, stage in [(compute, "-C", "Compute"),
                                 (vertex, "-v", "Vertex"),
                                 (fragment, "-f", "Fragment")]:
        report_path = root / (stage + ".json")
        isa_prefix = root / (stage + "-stats")
        result = run("--vulkan", "--spirv", selector, "--name", "main",
                     "--format=json", "-cMali-G610", "-d", "-o", report_path,
                     "--isa-prefix", isa_prefix, spv)
        assert result.stdout == ""
        report = json.loads(report_path.read_text())
        assert report["schema"] == {"name": "performance", "version": 2}
        assert report["producer"]["name"] == "kraidoc"
        shader = report["shaders"][0]
        assert shader["shader"] == {"api": "Vulkan", "type": stage}
        assert shader["hardware"]["core"] == "Mali-G610"
        assert shader["kraid"]["original_pipeline_state"] is False
        legacy = json.loads(Path(str(isa_prefix) + ".report.json").read_text())
        assert len(shader["variants"]) == len(legacy["programs"])
        for variant, program in zip(shader["variants"], legacy["programs"]):
            props = {p["name"]: p["value"] for p in variant["properties"]}
            assert props["instructions"] == program["instructions"]
            assert 0 < props["work_registers_used"] <= program["work_reg_count"]
            assert props["thread_occupancy"] in [50, 100]
            assert props["stack_size"] == props["stack_alloca_bytes"] + props["stack_spill_bytes"]
            perf = variant["performance"]
            assert len(perf["pipelines"]) == len(perf["total_cycles"]["cycle_count"])
            assert all(v >= 0 for v in perf["total_cycles"]["cycle_count"])
            assert set(perf["total_cycles"]["bound_pipelines"]) <= set(perf["pipelines"])
            assert perf["shortest_path_cycles"] is None
            assert perf["longest_path_cycles"] is None
    typed = compile_shader(root, compute, "typed", "-S3=99")
    assert typed["programs"] == specialized["programs"]
    typed_types = fixture(root, "typed-types.comp", """#version 450
layout(local_size_x=1) in;
layout(constant_id=4) const int signed_value=-7;
layout(constant_id=5) const float float_value=1.0;
layout(constant_id=6) const bool bool_value=true;
layout(set=0,binding=0) buffer Data { float data[]; } output_data;
void main() {
  output_data.data[gl_GlobalInvocationID.x]=
    bool_value ? float(signed_value)+float_value : float_value;
}
""")
    typed_values = compile_shader(root, typed_types, "typed-values",
                                  "--sconst", "4=-23", "-S5=1.5", "-S6=false")
    raw_values = compile_shader(root, typed_types, "raw-values",
                                "--spec", "4:4:0xffffffe9",
                                "--spec", "5:4:0x3fc00000", "--spec", "6:4:0")
    assert typed_values["programs"] == raw_values["programs"]
    assert typed_values["specialization"] == raw_values["specialization"]
    for value in ["4=2147483648", "4=-2147483649", "5=invalid", "6=2"]:
        run(typed_types, "-S", value, success=False)
    scratch = fixture(root, "scratch.comp", """#version 450
layout(local_size_x=1) in;
layout(set=0,binding=0) buffer Data { float data[]; } output_data;
void main() {
  float values[128];
  uint id=gl_GlobalInvocationID.x;
  for (int i=0;i<128;i++) values[i]=sin(float(id)+float(i));
  output_data.data[id]=values[id%128];
}
""")
    scratch_report = json.loads(run("-C", "--format", "json", scratch).stdout)
    scratch_props = {p["name"]: p["value"] for p in
                     scratch_report["shaders"][0]["variants"][0]["properties"]}
    assert scratch_props["stack_alloca_bytes"] > 0
    assert scratch_props["stack_spill_bytes"] == 0
    assert scratch_props["has_stack_spilling"] is False
    assert "Work registers:" in run("-C", compute).stdout
    assert "Total instruction cycles:" in run("--compute", "--detailed", compute).stdout
    stdout_report = json.loads(run("-C", "--format", "json", compute).stdout)
    assert stdout_report["shaders"][0]["shader"]["type"] == "Compute"
    stdin_report = subprocess.run([tool, "-C", "--format", "json", "-"],
                                  input=compute.read_bytes(), env=env,
                                  capture_output=True, check=True)
    assert json.loads(stdin_report.stdout)["shaders"][0]["filename"] == "-"
    listed = json.loads(run("-l", "--format", "json").stdout)
    assert "Mali-G610" in [c["core"] for c in listed["cores"]]
    info = json.loads(run("-i", "-c", "Mali-G610", "--format", "json").stdout)
    assert info["schema"]["name"] == "info"
    assert info["hardware"]["architecture"] == "Valhall"
    assert info["apis"]["vulkan"]["max_version"] == 1.6
    for args in [("-S", "3=-1"), ("-S", "3=4294967296"),
                 ("-S", "99=1"), ("-S3=1", "-S3=2"),
                 ("--spec", "3:4:1", "-S3=2"), ("--opengles",),
                 ("--geometry",), ("-C", "-v")]:
        run(compute, *args, success=False)
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
