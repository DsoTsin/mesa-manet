import argparse
import json
import sys

KINDS = ("primary", "internal")
TABLE_SPACES = ("cx", "sh", "uc", "uc")
SPACES = ("cx", "sh", "uc")
VERSIONS = 14
INDEX_LIMIT = 0x100
OFFSET_LIMIT = 0x10000
GROUP_LIMIT = 256
PAIRS_PER_LINE = 4


def word(value, what):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xffffffff:
        raise SystemExit("%s: %r is not a 32-bit value" % (what, value))
    return value


def parse_group(where, group):
    table = group.get("table")
    if table not in range(len(TABLE_SPACES)) or group.get("space") != TABLE_SPACES[table]:
        raise SystemExit("%s: table %r does not hold space %r" % (where, table, group.get("space")))
    index = word(group["index"], where + " index")
    type_word = word(group["word"], where + " word")
    here = "%s table %d index %d" % (where, table, index)
    if index >= INDEX_LIMIT or type_word & 3 != table or (type_word >> 2) & 0xff != index:
        raise SystemExit("%s: word %#010x names another slot" % (here, type_word))
    registers = []
    for pair in group["registers"]:
        if len(pair) != 2:
            raise SystemExit("%s: %r is not an offset and a value" % (here, pair))
        offset = word(pair[0], here + " register offset")
        if offset >= OFFSET_LIMIT:
            raise SystemExit("%s: register offset %#x is out of range" % (here, offset))
        registers.append((offset, word(pair[1], here + " register value")))
    if not 1 <= len(registers) <= GROUP_LIMIT:
        raise SystemExit("%s: %d registers in one group" % (here, len(registers)))
    if ((type_word >> 10) & 0xff) * (type_word >> 18) != len(registers):
        raise SystemExit("%s: word %#010x does not count %d registers" % (here, type_word, len(registers)))
    return {
        "space": group["space"],
        "table": table,
        "index": index,
        "type": word(group["type"], here + " type"),
        "word": type_word,
        "registers": registers,
    }


def parse_set(where, groups):
    if not groups:
        raise SystemExit("%s: no groups" % where)
    types = set()
    for group in groups:
        if group["type"] in types:
            raise SystemExit("%s: type %#010x is repeated" % (where, group["type"]))
        types.add(group["type"])
    banks = []
    for table in range(len(TABLE_SPACES)):
        mine = sorted((group for group in groups if group["table"] == table), key=lambda group: group["index"])
        if [group["index"] for group in mine] != list(range(len(mine))):
            raise SystemExit("%s table %d: the indices are not 0 to %d" % (where, table, len(mine) - 1))
        banks.append(mine)
    first = 0
    for bank in banks:
        for group in bank:
            group["first"] = first
            first += len(group["registers"])
    return {
        "groups": groups,
        "banks": banks,
        "registers": [pair for bank in banks for group in bank for pair in group["registers"]],
    }


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sets = {}
    for key, groups in data["versions"].items():
        if not key.isdigit() or int(key) >= VERSIONS:
            raise SystemExit("bad version %r" % key)
        for group in groups:
            if group["kind"] not in KINDS:
                raise SystemExit("version %s: unknown kind %r" % (key, group["kind"]))
        for kind in KINDS:
            where = "version %s %s" % (key, kind)
            mine = [parse_group(where, group) for group in groups if group["kind"] == kind]
            if mine:
                sets[(int(key), kind)] = parse_set(where, mine)
    mapping = {}
    for kind in KINDS:
        versions = data["map"][kind]
        if len(versions) != VERSIONS:
            raise SystemExit("the %s map does not name %d versions" % (kind, VERSIONS))
        for version in versions:
            if (version, kind) not in sets:
                raise SystemExit("the %s map names version %r, which has no %s groups" % (kind, version, kind))
        mapping[kind] = versions
    unused = set(sets) - {(version, kind) for kind in KINDS for version in mapping[kind]}
    if unused:
        raise SystemExit("tables no version uses: %s" % sorted(unused))
    return sets, mapping


def emit_set(w, name, table):
    registers = table["registers"]
    w("static const struct agc_register %s_registers[%d] = {\n" % (name, len(registers)))
    for start in range(0, len(registers), PAIRS_PER_LINE):
        row = registers[start:start + PAIRS_PER_LINE]
        w("   " + " ".join("{0x%04x, 0x%08x}," % pair for pair in row) + "\n")
    w("};\n\n")
    w("static const struct agc_defaults_group %s_groups[%d] = {\n" % (name, len(table["groups"])))
    for group in table["groups"]:
        w("   {0x%08xu, 0x%08xu, %d, %d, AGC_REG_BANK_%s, %d, &%s_registers[%d]},\n" % (
            group["type"], group["word"], group["index"], len(group["registers"]), group["space"].upper(),
            group["table"], name, group["first"]))
    w("};\n\n")


def emit(out, sets, mapping):
    w = out.write
    ordered = sorted(sets, key=lambda key: (key[0], KINDS.index(key[1])))
    number = {key: at for at, key in enumerate(ordered)}
    w("#ifndef AGC_REGISTER_DEFAULTS_TABLES_H\n#define AGC_REGISTER_DEFAULTS_TABLES_H\n\n")
    w("#define AGC_DEFAULTS_MAP_VERSIONS %du\n" % VERSIONS)
    w("#define AGC_DEFAULTS_SETS %du\n" % len(ordered))
    for space in SPACES:
        end = 1 + max(offset for table in sets.values() for group in table["groups"]
                      if group["space"] == space for offset, _ in group["registers"])
        w("#define AGC_DEFAULTS_%s_END 0x%xu\n" % (space.upper(), end))
    largest = max(len(group["registers"]) for table in sets.values() for group in table["groups"])
    w("#define AGC_DEFAULTS_GROUP_REGISTERS_MAX %du\n\n" % largest)
    for key in ordered:
        emit_set(w, "agc_defaults_v%d_%s" % key, sets[key])
    w("static const struct agc_defaults_set agc_defaults_sets[AGC_DEFAULTS_SETS] = {\n")
    for key in ordered:
        name = "agc_defaults_v%d_%s" % key
        table = sets[key]
        counts = ", ".join(str(sum(len(group["registers"]) for group in bank)) for bank in table["banks"])
        pointers = ", ".join(str(len(bank)) for bank in table["banks"])
        w("   {%s_groups, %s_registers, %d, %d, {%s}, {%s}},\n" % (
            name, name, len(table["groups"]), key[0], counts, pointers))
    w("};\n\n")
    w("static const uint8_t agc_defaults_version_sets[%d][AGC_DEFAULTS_MAP_VERSIONS] = {\n" % len(KINDS))
    for kind in KINDS:
        w("   {" + ", ".join(str(number[(version, kind)]) for version in mapping[kind]) + "},\n")
    w("};\n\n#endif\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("defaults")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sets, mapping = load(args.defaults)
    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        emit(out, sets, mapping)


if __name__ == "__main__":
    sys.exit(main())
