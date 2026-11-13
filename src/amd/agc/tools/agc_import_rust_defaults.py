import argparse
import json
import pathlib
import re

KINDS = {"PUBLIC": "primary", "INTERNAL": "internal"}
SPACES = ("cx", "sh", "uc", "uc")
VERSIONS = 14
REGS = re.compile(r"static (\w+): \[\(u32, u32\); (\d+)\] = \[(.*?)\];", re.S)
PAIR = re.compile(r"\((0x[0-9a-fA-F]+), (0x[0-9a-fA-F]+)\)")
PTRS = re.compile(r"static (\w+): \[u16; (\d+)\] = \[(.*?)\];", re.S)
WORDS = re.compile(r"static (\w+): \[u32; (\d+)\] = \[(.*?)\];", re.S)
HEX = re.compile(r"0x[0-9a-fA-F]+")
TABLE = re.compile(r"static (\w+): Table = Table \{ banks: \[(.*?)\], types: &(\w+) \};")
BANK = re.compile(r"Some\(Bank \{ registers: &(\w+), table: \d+, pointers: &(\w+) \}\)|None")
NAME = re.compile(r"(PUBLIC|INTERNAL)_REG_DEFAULTS_V(\d+)")
MAP = re.compile(r"pub static (PUBLIC|INTERNAL): \[&Table; (\d+)\] = \[(.*?)\];", re.S)


def arrays(pattern, text, parse):
    found = {}
    for name, length, body in pattern.findall(text):
        values = parse(body)
        if len(values) != int(length):
            raise SystemExit("%s declares %s entries and holds %d" % (name, length, len(values)))
        found[name] = values
    return found


def table_groups(name, banks, types):
    kind_name, version = NAME.fullmatch(name).groups()
    if len(types) % 3:
        raise SystemExit("%s: the type table is not a list of triples" % name)
    used = [[0] * len(bank[1]) if bank else [] for bank in banks]
    for number, bank in enumerate(banks):
        if bank is None:
            continue
        registers, pointers = bank
        if not pointers or pointers[0] != 0 or any(b <= a for a, b in zip(pointers, pointers[1:])):
            raise SystemExit("%s bank %d: pointers do not rise from 0" % (name, number))
        if pointers[-1] >= len(registers):
            raise SystemExit("%s bank %d: a pointer is past the registers" % (name, number))
    groups = []
    for at in range(0, len(types), 3):
        type_hash, word, reserved = types[at:at + 3]
        table = word & 3
        index = (word >> 2) & 0xff
        per = (word >> 10) & 0xff
        elements = word >> 18
        if reserved != 0:
            raise SystemExit("%s type %#010x: reserved word %#x" % (name, type_hash, reserved))
        if banks[table] is None or index >= len(banks[table][1]):
            raise SystemExit("%s type %#010x: no pointer %d in table %d" % (name, type_hash, index, table))
        registers, pointers = banks[table]
        end = pointers[index + 1] if index + 1 < len(pointers) else len(registers)
        span = registers[pointers[index]:end]
        if per * elements != len(span):
            raise SystemExit("%s type %#010x: %d x %d registers, the pointers hold %d" % (
                name, type_hash, per, elements, len(span)))
        used[table][index] += 1
        groups.append({
            "kind": KINDS[kind_name],
            "space": SPACES[table],
            "table": table,
            "index": index,
            "type": type_hash,
            "word": word,
            "registers": [[offset, value] for offset, value in span],
        })
    for number, counts in enumerate(used):
        if any(count != 1 for count in counts):
            raise SystemExit("%s bank %d: a pointer is not named by exactly one type" % (name, number))
    return int(version), KINDS[kind_name], groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rust")
    ap.add_argument("out")
    args = ap.parse_args()
    text = pathlib.Path(args.rust).read_text(encoding="utf-8")
    registers = arrays(REGS, text, lambda body: [(int(o, 16), int(v, 16)) for o, v in PAIR.findall(body)])
    pointers = arrays(PTRS, text, lambda body: [int(x, 16) for x in HEX.findall(body)])
    words = arrays(WORDS, text, lambda body: [int(x, 16) for x in HEX.findall(body)])
    tables = {}
    for name, body, types in TABLE.findall(text):
        banks = []
        for match in BANK.finditer(body):
            banks.append((registers[match.group(1)], pointers[match.group(2)]) if match.group(1) else None)
        if len(banks) != 4:
            raise SystemExit("%s has %d banks" % (name, len(banks)))
        tables[name] = table_groups(name, banks, words[types])
    mapping = {}
    used = set()
    for kind_name, length, body in MAP.findall(text):
        names = [entry.strip().lstrip("&") for entry in body.split(",") if entry.strip()]
        if int(length) != VERSIONS or len(names) != VERSIONS:
            raise SystemExit("%s does not map %d versions" % (kind_name, VERSIONS))
        mapping[KINDS[kind_name]] = [tables[name][0] for name in names]
        used.update(names)
    if set(mapping) != set(KINDS.values()) or used != set(tables):
        raise SystemExit("the version maps and the tables do not match")
    versions = {}
    for kind in KINDS.values():
        for name in sorted(tables, key=lambda name: tables[name][0]):
            version, table_kind, groups = tables[name]
            if table_kind == kind:
                versions.setdefault(str(version), []).extend(groups)
    out = {
        "source": "AnyPS5 driver/runtime/src/agc/defaults_data.rs",
        "map": mapping,
        "versions": dict(sorted(versions.items(), key=lambda item: int(item[0]))),
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
