import argparse
import json
import pathlib
import re

SPACES = {"kAgcRegisterDefaultSpaceCx": "cx", "kAgcRegisterDefaultSpaceSh": "sh", "kAgcRegisterDefaultSpaceUc": "uc"}
ARRAY = re.compile(r"(?:/\*\s*([A-Za-z0-9_ ,/.\-]+?)\s*\*/\s*)?static const AgcRegisterDefaultValue (s_regdef_v\d+_\w+?_regs_\d+)\[\] = \{(.*?)\};", re.S)
PAIR = re.compile(r"\{\s*(0x[0-9a-fA-F]+|\d+)\s*,\s*(0x[0-9a-fA-F]+|\d+)\s*\}")
GROUP = re.compile(r"\{\s*(kAgcRegisterDefaultSpace\w+)\s*,\s*(\d+)\s*,\s*(0x[0-9a-fA-F]+)u?\s*,\s*(\d+)\s*,\s*(s_regdef_v\d+_\w+?_regs_\d+)\s*\}")
TABLE = re.compile(r"static const AgcRegisterDefaultsGroup s_regdef_v\d+_(primary|internal)_defaults\[\] = \{(.*?)\};", re.S)


def parse_version(path):
    text = path.read_text(encoding="utf-8")
    arrays = {}
    for name, ident, body in ARRAY.findall(text):
        arrays[ident] = (name.strip(), [(int(o, 0), int(v, 0)) for o, v in PAIR.findall(body)])
    groups = []
    for kind, body in TABLE.findall(text):
        for space, index, type_hash, count, ident in GROUP.findall(body):
            name, regs = arrays[ident]
            if len(regs) != int(count):
                raise SystemExit("%s: %s has %d registers, table says %s" % (path.name, ident, len(regs), count))
            groups.append({
                "kind": kind,
                "space": SPACES[space],
                "index": int(index),
                "type": int(type_hash, 16),
                "name": name,
                "registers": [[o, v] for o, v in regs],
            })
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("openagc")
    ap.add_argument("out")
    args = ap.parse_args()
    src = pathlib.Path(args.openagc) / "src"
    versions = {}
    for path in sorted(src.glob("register_defaults_v*.c")):
        m = re.fullmatch(r"register_defaults_v(\d+)\.c", path.name)
        if not m:
            continue
        versions[int(m.group(1))] = parse_version(path)
    out = {"source": "OpenAGC register_defaults_v*.c", "versions": {str(k): versions[k] for k in sorted(versions)}}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
