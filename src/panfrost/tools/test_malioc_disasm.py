import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


EXECUTABLE = Path(sys.argv.pop(1)).resolve()


def chunk(tag, payload):
    return tag + struct.pack('<I', len(payload)) + payload


def ebin(code=b'\0' * 8, children=None):
    header = struct.pack('<8I', 0, 0, 0xffffffff, 32, 0, 0, 0, 128)
    return chunk(b'EBIN', header + (chunk(b'OBJC', code) if children is None else children))


def mbs2(stage, binaries, constants=()):
    common = b''.join(chunk(tag, b'\0' * 4) for tag in [b'VELA', *([b'SSYM'] * 6), b'UBUF'])
    common += struct.pack('<HHI', 0, 0, len(constants)) + b''.join(chunk(b'FCST', v) for v in constants)
    common += struct.pack('<I', len(binaries)) + b''.join(binaries) + struct.pack('<I', 0)
    tags = {'vertex': b'CVER', 'fragment': b'CFRA', 'compute': b'CCOM'}
    return chunk(b'MBS2', struct.pack('<I', 54) + chunk(b'VEHW', b'\0' * 12)
                 + chunk(tags[stage], chunk(b'CMMN', common)))


def spirv(entries):
    data = struct.pack('<5I', 0x07230203, 0x10000, 0, 16, 0)
    for model, name in entries:
        text = name.encode() + b'\0'
        text += b'\0' * (-len(text) % 4)
        count = 3 + len(text) // 4
        data += struct.pack('<3I', count << 16 | 15, model, 1) + text
    return data


class NativeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / 'input.bin'
        self.prefix = self.root / 'nested' / 'output'
        self.library = self.root / 'unrecognized.dll'
        self.library.write_bytes(b'not the analyzed library')

    def invoke(self, data, *options):
        self.input.write_bytes(data)
        return subprocess.run([str(EXECUTABLE), str(self.input), *map(str, options)],
                              capture_output=True, timeout=20)

    def reject(self, data, *options):
        result = self.invoke(data, *options)
        self.assertEqual(result.returncode, 1, result.stderr.decode(errors='replace'))
        self.assertIn(b'malioc ISA dump failed:', result.stderr)
        return result.stderr

    def check_spirv(self, data, name='main', stage=None, valid=False):
        options = ['-o', self.prefix, '-n', name, '--library', self.library]
        if stage:
            options += ['--stage', stage]
        error = self.reject(data, *options)
        reached_compiler = (b'unsupported compiler DLL' in error or
                            b'compiler ABI requires Windows x64' in error)
        self.assertEqual(reached_compiler, valid, error.decode(errors='replace'))

    def extract(self, data, stage='compute', extra=()):
        result = self.invoke(data, '--mbs2', '--stage', stage, '-o', self.prefix, *extra)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        report = json.loads(Path(f'{self.prefix}.json').read_text(encoding='utf-8'))
        self.assertEqual(Path(f'{self.prefix}.mbs2.bin').read_bytes(), data)
        for binary in report['binaries']:
            self.assertEqual(data[binary['ebin_offset']:binary['ebin_offset'] + 4], b'EBIN')
            self.assertEqual(data[binary['objc_offset']:binary['objc_offset'] + 4], b'OBJC')
            self.assertEqual(len(Path(binary['binary']).read_bytes()), binary['size'])
        return report

    def test_stages_and_selected_entrypoint(self):
        data = spirv([(0, 'vs'), (4, 'fs'), (5, 'cs')])
        for name, stage in [('vs', 'vertex'), ('fs', 'fragment'), ('cs', 'compute')]:
            with self.subTest(stage=stage):
                self.check_spirv(data, name, valid=True)
                self.check_spirv(data, name, stage, valid=True)

    def test_missing_and_mismatched_stage(self):
        data = spirv([(0, 'foo')])
        self.check_spirv(data)
        self.check_spirv(data, 'foo', 'compute')

    def test_ambiguous_entrypoint(self):
        data = spirv([(0, 'main'), (4, 'main')])
        self.check_spirv(data)
        self.check_spirv(data, stage='fragment', valid=True)
        self.check_spirv(spirv([(4, 'main'), (4, 'main')]), stage='fragment')

    def test_unsupported_model(self):
        self.check_spirv(spirv([(3, 'main')]))

    def test_malformed_instructions(self):
        data = spirv([(5, 'main')])
        cases = [data[:16], data[:-1], b'BAD!' + data[4:], data + b'\0' * 4,
                 data[:20] + struct.pack('<I', 100 << 16 | 15) + data[24:],
                 data[:32] + b'abcdabcd', data[:12] + b'\0' * 4 + data[16:],
                 data[:16] + b'\1\0\0\0' + data[20:]]
        for case in cases:
            with self.subTest(data=case):
                self.check_spirv(case)

    def test_all_stages_and_binaries(self):
        for stage in ('vertex', 'fragment', 'compute'):
            with self.subTest(stage=stage):
                data = mbs2(stage, [ebin(b'\1' * 8), ebin(b'\2' * 16)])
                report = self.extract(data, stage, ['--extract-only'])
                self.assertEqual([Path(b['binary']).read_bytes() for b in report['binaries']],
                                 [b'\1' * 8, b'\2' * 16])
                self.assertTrue(all('isa' not in b for b in report['binaries']))

    def test_constants_are_not_scanned(self):
        report = self.extract(mbs2('compute', [ebin()], constants=[b'OBJC\0\0\0\0']))
        self.assertEqual(len(report['binaries']), 1)

    def test_code_is_not_scanned_for_chunks(self):
        payload = b'OBJC\0\0\0\0EBIN\0\0\0\0'
        report = self.extract(mbs2('fragment', [ebin(payload)]), 'fragment', ['--extract-only'])
        self.assertEqual(Path(report['binaries'][0]['binary']).read_bytes(), payload)

    def test_missing_objc_does_not_read_next_ebin(self):
        data = mbs2('vertex', [ebin(children=chunk(b'UBUF', b'')), ebin()])
        self.reject(data, '--mbs2', '--stage', 'vertex', '-o', self.prefix)

    def test_duplicate_objc_and_oversized_child(self):
        cases = [chunk(b'OBJC', b'\0' * 8) * 2, b'OBJC' + struct.pack('<I', 64) + b'\0' * 8,
                 b'OBJC' + struct.pack('<I', 0xffffffff)]
        for children in cases:
            with self.subTest(children=children):
                self.reject(mbs2('compute', [ebin(children=children)]),
                            '--mbs2', '--stage', 'compute', '-o', self.prefix)

    def test_truncated_container(self):
        data = mbs2('fragment', [ebin()])
        for length in range(len(data)):
            with self.subTest(length=length):
                self.reject(data[:length], '--mbs2', '--stage', 'fragment', '-o', self.prefix)

    def test_stage_version_alignment_and_empty_binary(self):
        data = mbs2('vertex', [ebin()])
        cases = [(data, 'compute'), (data[:8] + struct.pack('<I', 53) + data[12:], 'vertex'),
                 (data + b'\0', 'vertex'), (mbs2('vertex', [ebin(b'1234')]), 'vertex'),
                 (mbs2('vertex', [ebin(b'')]), 'vertex'), (mbs2('vertex', []), 'vertex')]
        for case, stage in cases:
            with self.subTest(stage=stage, data=case):
                self.reject(case, '--mbs2', '--stage', stage, '-o', self.prefix)

    def test_metadata_before_objc(self):
        children = chunk(b'SPDf', b'OBJC') + chunk(b'OBJC', b'\0' * 8)
        report = self.extract(mbs2('compute', [ebin(children=children)]))
        binary = report['binaries'][0]
        self.assertTrue(Path(binary['isa']).read_bytes())
        raw = self.invoke(Path(binary['binary']).read_bytes(), '--raw')
        self.assertEqual(raw.returncode, 0)
        self.assertEqual(raw.stdout, Path(binary['isa']).read_bytes())

    def test_invalid_counts_and_reserved_field(self):
        data = mbs2('compute', [ebin()])
        common_start = data.index(b'CMMN') + 8
        fields = common_start + 8 * 12
        for offset, value in [(fields, b'\1\0'), (fields + 4, b'\xff' * 4),
                              (fields + 8, b'\xff' * 4)]:
            case = data[:offset] + value + data[offset + len(value):]
            self.reject(case, '--mbs2', '--stage', 'compute', '-o', self.prefix)

    def test_unicode_output_paths_and_json(self):
        self.prefix = self.root / '\u6d4b\u8bd5' / "name with 'quote"
        report = self.extract(mbs2('compute', [ebin()]))
        self.assertEqual(Path(report['binaries'][0]['binary']), Path(f'{self.prefix}.ebin0.objc'))

    def test_cli_errors(self):
        for options in [[], ['-o'], ['--stage', 'geometry', '-o', self.prefix],
                        ['--mbs2', '-o', self.prefix], ['--raw', '-o', self.prefix],
                        ['--stage', 'vertex', '--stage', 'bad', '-o', self.prefix],
                        ['--unknown']]:
            self.reject(b'', *options)
        self.reject(b'1234', '--raw')


if __name__ == '__main__':
    unittest.main()
