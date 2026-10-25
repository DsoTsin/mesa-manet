import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('compiler')
    parser.add_argument('glslang')
    args = parser.parse_args()
    sources = {
        'vert': 'void main() { gl_Position = vec4(float(gl_VertexIndex), 0, 0, 1); }',
        'frag': 'layout(location=0) out vec4 colour; void main() { colour = vec4(gl_FragCoord.xy, 0, 1); }',
        'comp': '''layout(local_size_x=64) in;
layout(constant_id=7) const uint factor=3;
layout(constant_id=8) const float addend=1.0;
layout(set=0,binding=0) buffer Data { uint values[]; } data_out;
void main() { uint i=gl_GlobalInvocationID.x; data_out.values[i]=i*factor+uint(addend); }''',
        'array.frag': '''#extension GL_EXT_nonuniform_qualifier : require
layout(set=0,binding=0) uniform sampler2D images[];
layout(location=0) flat in uint index;
layout(location=0) out vec4 colour;
void main() { colour=texture(images[nonuniformEXT(index)], vec2(0.5)); }''',
        'query.comp': '''#extension GL_EXT_ray_query : require
layout(local_size_x=1) in;
layout(set=0,binding=0) uniform accelerationStructureEXT scene;
layout(set=0,binding=1) buffer Data { uint values[]; } data_out;
void main() { rayQueryEXT q;
rayQueryInitializeEXT(q,scene,gl_RayFlagsOpaqueEXT,255,vec3(0),0.01,vec3(0,0,1),10.0);
while(rayQueryProceedEXT(q)) {}
data_out.values[gl_GlobalInvocationID.x]=rayQueryGetIntersectionTypeEXT(q,true); }''',
        'rgen': '''#extension GL_EXT_ray_tracing : require
void main() {}''',
    }
    checks = 0
    with tempfile.TemporaryDirectory(prefix='ir3-offline-') as temporary:
        root = Path(temporary)
        modules = {}
        for name, source in sources.items():
            path = root / ('shader.' + name)
            path.write_text('#version 460\n' + source + '\n')
            module = path.with_name(path.name + '.spv')
            subprocess.run([args.glslang, '-V', '--target-env', 'vulkan1.2', '-o', str(module), str(path)],
                           check=True, capture_output=True)
            modules[name] = module

        def run(*values, success=True):
            nonlocal checks
            p = subprocess.run([args.compiler, *map(str, values)], capture_output=True, text=True)
            assert (p.returncode == 0) == success, (values, p.returncode, p.stdout, p.stderr)
            assert p.returncode >= 0, (values, p.stderr)
            checks += 1
            return p

        assert 'Adreno' in run('--list').stdout
        for gpu in ('a650', 'a730', 'a830'):
            for name in ('vert', 'frag', 'comp'):
                prefix = root / (gpu + '-' + name)
                run('-c', gpu, '-o', prefix, modules[name])
                report = json.loads(prefix.with_suffix('.json').read_text())
                assert report['variants'][0]['instructions'] > 0
                assert prefix.with_suffix('.0.main.bin').stat().st_size > 0
                assert prefix.with_suffix('.0.main.isa.txt').stat().st_size > 0
                assert prefix.with_suffix('.initial.nir').stat().st_size > 0
                assert prefix.with_suffix('.final.nir').stat().st_size > 0
        default = json.loads(run(modules['comp']).stdout)
        typed = json.loads(run('--sconst', '7=5', '--sconst', '8=2.5', modules['comp']).stdout)
        raw = json.loads(run('--spec', '7:4:5', '--spec', '8:4:0x40200000', modules['comp']).stdout)
        assert typed['variants'][0]['binary_blake3'] == raw['variants'][0]['binary_blake3']
        assert default['variants'][0]['binary_blake3'] != typed['variants'][0]['binary_blake3']
        assert default['variants'][0]['binary_blake3'] == json.loads(run(modules['comp']).stdout)['variants'][0]['binary_blake3']
        run('--sconst', '999=1', modules['comp'], success=False)
        run('--sconst', '7=-1', modules['comp'], success=False)
        run('--spec', '7:4:1', '--sconst', '7=2', modules['comp'], success=False)
        run('--gpu', 'a999', modules['comp'], success=False)
        run('--gpu', 'a330', modules['comp'], success=False)
        run('--subgroup-size', '3', modules['comp'], success=False)
        run('--entry', 'missing', modules['comp'], success=False)
        run('--stage', 'fragment', modules['comp'], success=False)
        run('--binding', '0:0:uniform-buffer:1', modules['comp'], success=False)
        run(modules['array.frag'], success=False)
        array = json.loads(run('--binding', '0:0:combined-image-sampler:256', modules['array.frag']).stdout)
        assert array['bindings'][0]['count'] == 256
        run('--robust', '--push-constant-size', '0', modules['comp'])
        run(modules['query.comp'])
        assert 'RT pipeline' in run(modules['rgen'], success=False).stderr
        invalid = root / 'invalid.spv'
        invalid.write_bytes(b'invalid')
        run(invalid, success=False)
        words = bytearray(modules['comp'].read_bytes())
        words[20:24] = b'\0\0\0\0'
        invalid.write_bytes(words)
        run(invalid, success=False)
    print(f'{checks} compiler checks passed')


if __name__ == '__main__':
    main()
