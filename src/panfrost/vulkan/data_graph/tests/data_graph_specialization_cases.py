import argparse
import re
import struct
from pathlib import Path


def generate():
    text = Path(__file__).with_name('data-graph-fixtures.h').read_text()
    body = re.search(r'graph_chain\[\] = \{(.*?)\};', text, re.S).group(1)
    words = [int(value, 16) for value in re.findall(r'0x[0-9a-f]+', body)]
    output = words[:5]
    column_id, boolean_id, float_id, boolean_type = range(words[3], words[3] + 4)
    output[3] = words[3] + 4
    decorated = False
    index = 5
    while index < len(words):
        count, opcode = words[index] >> 16, words[index] & 0xffff
        inst = words[index:index + count]
        if not decorated and opcode == 21:
            for target, spec_id in [(column_id, 43), (float_id, 42), (boolean_id, 44)]:
                output.extend([(4 << 16) | 71, target, 1, spec_id])
            decorated = True
        if opcode == 43 and inst[2] == 13:
            output.extend([(4 << 16) | 50, 5, column_id, 18])
            inst = [(6 << 16) | 52, 5, 13, 128, column_id, 8]
        elif opcode == 43 and inst[2] == 15:
            output.extend([(2 << 16) | 20, boolean_type])
            output.extend([(3 << 16) | 48, boolean_type, boolean_id])
            output.extend([(4 << 16) | 50, 6, float_id, 0x3f800000])
            inst = [(7 << 16) | 52, 6, 15, 169, boolean_id, float_id, 14]
        elif opcode == 44:
            inst[0] = (count << 16) | 51
        output.extend(inst)
        index += count
    return struct.pack('<' + 'I' * len(output), *output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_bytes(generate())
