import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--spirv-as', default='spirv-as')
parser.add_argument('--spirv-val', default='spirv-val')
parser.add_argument('--extended', action='store_true')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
source = Path(__file__).with_name('data-graph-matmul.spvasm').read_text()
shapes = [(1, n, n, n) for n in [1, 17, 32]]
shapes += [(b, 3, 5, 4) for b in [2, 3, 4, 17]]
configurations = [(*shape, 0, 0) for shape in shapes]
configurations += [(2, 3, 5, 4, a, b) for a, b in
                   [(3, -2), (-7, 11), (127, -128), (-128, 127)]]
if args.extended:
    configurations += [(1, 1, k, 1, 0, 0) for k in [1023, 1025, 2049, 8193]]
    configurations += [(2, 3, 4081, 4, 3, -2),
                       (3, 17, 4096, 32, -128, 127),
                       (2, 3, 65, 4, 3, -2)]
    for n in [4095, 4096, 8193]:
        configurations += [(n, 1, 1, 1, 0, 0),
                           (1, n, 1, 1, 0, 0),
                           (1, 1, 1, n, 0, 0)]
cases = []
for batch, rows, inner, columns, azp, bzp in configurations:
    for mixed in [False, True]:
        text = source
        for key, old, value in [('batch', 2, batch), ('rows', 3, rows),
                                ('inner', 5, inner), ('columns', 4, columns)]:
            text = text.replace(f'%{key} = OpConstant %u {old}',
                                f'%{key} = OpConstant %u {value}')
        text = text.replace('%azp = OpConstant %f 3',
                            f'%azp = OpConstant %f {azp}')
        text = text.replace('%bzp = OpConstant %f -2',
                            f'%bzp = OpConstant %f {bzp}')
        if mixed:
            text = text.replace('OpGraphSetOutputARM %sum %c0',
                                '%result = OpExtInst %t2 %tosa ADD %sum %sum\n'
                                'OpGraphSetOutputARM %result %c0')
        mode = 'matmul-add' if mixed else 'matmul'
        name = f'{mode}-{batch}-{rows}-{inner}-{columns}-zp{azp}-{bzp}'
        asm = args.out / (name + '.spvasm')
        spv = args.out / (name + '.spv')
        asm.write_text(text)
        subprocess.run([args.spirv_as, '--target-env', 'vulkan1.4',
                        str(asm), '-o', str(spv)], check=True)
        subprocess.run([args.spirv_val, '--target-env', 'vulkan1.4',
                        str(spv)], check=True)
        for padded in [0, 1]:
            environment = dict(DG_TEST_DTYPE=3, DG_INTEGER_DATA=1,
                               DG_TEST_BATCH=batch, DG_TEST_M=rows,
                               DG_TEST_K=inner, DG_TEST_N=columns,
                               DG_A_ZP=azp, DG_B_ZP=bzp,
                               DG_TEST_PADDED=padded)
            cases.append(dict(name=name + f'-pad{padded}', spirv=spv.name,
                              mode=mode, environment=environment,
                              checks=batch * rows * columns,
                              sha256=hashlib.sha256(spv.read_bytes()).hexdigest()))
(args.out / 'cases.json').write_text(json.dumps(cases, indent=2))
