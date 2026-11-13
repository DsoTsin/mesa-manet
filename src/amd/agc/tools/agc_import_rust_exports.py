import argparse
import json
import pathlib
import re

EXPORT = re.compile(r"agc_export!\s*\{\s*fn\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([^{]+?))?\s*\{", re.S)

TYPES = {
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "usize": "size_t",
    "isize": "ptrdiff_t",
    "f32": "float",
    "f64": "double",
    "bool": "bool",
    "c_void": "void",
    "c_char": "char",
    "CommandBuffer": "struct agc_cb",
    "ShaderRegister": "struct agc_register",
}
KEYWORDS = {"register": "reg", "default": "def", "const": "constant", "union": "merged", "signed": "is_signed", "auto": "automatic"}


def c_type(rust):
    rust = " ".join(rust.split())
    pointer = re.fullmatch(r"\*(mut|const) (.+)", rust)
    if pointer:
        inner = c_type(pointer.group(2))
        if pointer.group(1) == "const":
            return "const " + inner + " *" if not inner.endswith("*") else inner + " const *"
        return inner + " *" if not inner.endswith("*") else inner + "*"
    name = rust.split("::")[-1]
    if name in TYPES:
        return TYPES[name]
    if "fn" in rust.split("(")[0].split():
        return "void *"
    return "struct agc_" + re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def split_args(text):
    out = []
    depth = 0
    start = 0
    for index, ch in enumerate(text):
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:index])
            start = index + 1
    out.append(text[start:])
    return [part.strip() for part in out if part.strip()]


def param_name(name, ctype):
    name = name.strip().lstrip("_")
    if ctype == "struct agc_cb *":
        return "cb"
    return KEYWORDS.get(name, name)


def dump(functions):
    lines = ["{", ' "functions": [']
    for index, function in enumerate(functions):
        comma = "," if index + 1 < len(functions) else ""
        lines.append("  " + json.dumps(function, separators=(", ", ": ")) + comma)
    lines += [" ]", "}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("driver")
    ap.add_argument("manifest")
    args = ap.parse_args()
    root = pathlib.Path(args.driver) / "runtime" / "src" / "agc"
    manifest = pathlib.Path(args.manifest)
    known = {}
    if manifest.exists():
        for function in json.loads(manifest.read_text(encoding="utf-8"))["functions"]:
            known[function["name"]] = function
    added = 0
    for path in sorted(root.rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        module = "libSceAgcDriver" if path.relative_to(root).parts[0] == "driver" else "libSceAgc"
        for name, arglist, ret in EXPORT.findall(text):
            if name in known:
                continue
            params = []
            for arg in split_args(arglist):
                pname, ptype = arg.split(":", 1)
                ctype = c_type(ptype.strip())
                params.append([ctype, param_name(pname, ctype)])
            known[name] = {
                "name": name,
                "module": module,
                "return": c_type(ret.strip()) if ret else "void",
                "params": params,
                "state": "todo",
            }
            added += 1
    manifest.write_text(dump([known[name] for name in sorted(known)]), encoding="utf-8", newline="\n")
    print("functions %d, added %d" % (len(known), added))


if __name__ == "__main__":
    main()
