import argparse
import itertools
import json
import math
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--glslang', default='glslangValidator')
parser.add_argument('--spirv-val', default='spirv-val')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
cases = []
for rank, size, width, padded in itertools.product(range(1, 5), [1, 2, 4, 8], [1, 3, 4, 8, 16], [0, 1]):
    if size * width > 16 or rank == 1 and padded:
        continue
    ty = {1: 'uint8_t', 2: 'uint16_t', 4: 'uint', 8: 'uint64_t'}[size]
    dims = [2, 3, 5, 19][:rank]
    dims[-1] = 19
    for mode in [0, 1, 2] if width == 1 else [0, 1]:
        lines = ['uint index=gl_GlobalInvocationID.x;']
        lines += [f'if(tensorSizeARM(a,{d})!={v}u || tensorSizeARM(b,{d})!={v}u)return;' for d, v in enumerate(dims)]
        if mode == 1:
            lines += ['if(index!=0)return;']
            coords = ['0u'] * rank
            coords[-1] = '200u'
        else:
            blocks = (19 + width - 1) // width
            lines += [f'if(index>={math.prod(dims[:-1])*blocks}u)return;', f'uint x=(index%{blocks}u)*{width}u;', f'index/={blocks}u;']
            coords = ['0u'] * rank
            coords[-1] = 'x'
            for d in reversed(range(rank - 1)):
                lines += [f'uint d{d}=index%{dims[d]}u;', f'index/={dims[d]}u;']
                coords[d] = f'd{d}'
        lines += [f'uint c[{rank}]=uint[](' + ','.join(coords) + ');']
        lines += [f'{ty} v' + (f'[{width}]' if width > 1 else '') + ';']
        lines += ['tensorReadARM(a,c,v' + (f', 2, {ty}(0x3b)' if mode == 1 else '') + ');']
        if mode == 1:
            lines += [f'uint dst[{rank}]=uint[](' + ','.join(['0u'] * rank) + ');', 'tensorWriteARM(b,dst,v);']
        else:
            lines += ['tensorWriteARM(b,c,v);']
        name = f'r{rank}-s{size}-w{width}-p{padded}-m{mode}'
        src = args.out / (name + '.comp')
        spv = args.out / (name + '.spv')
        src.write_text(f'''#version 460
#extension GL_ARM_tensors : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
layout(local_size_x=64) in;
layout(set=0,binding=0) uniform tensorARM<{ty},{rank}> a;
layout(set=0,binding=1) uniform tensorARM<{ty},{rank}> b;
void main(){{{''.join(lines)}}}
''')
        subprocess.run([args.glslang, '-V', '--target-env', 'vulkan1.3', str(src), '-o', str(spv)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([args.spirv_val, '--target-env', 'vulkan1.3', str(spv)], check=True)
        cases.append({'name': name, 'spirv': spv.name, 'args': [rank, size, width, padded, mode]})
(args.out / 'cases.json').write_text(json.dumps(cases, indent=2))
