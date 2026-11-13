import argparse
import json
import re
import sys

STATES = ("done", "todo", "constant", "unimplemented", "host")
MODULES = ("libSceAgc", "libSceAgcDriver")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
VALUE = re.compile(r"-?(0x[0-9a-fA-F]+|[0-9]+)")
BUILTIN = {
    "void", "char", "bool", "float", "double", "size_t", "ptrdiff_t",
    "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
}


def check(function):
    name = function["name"]
    if function["state"] not in STATES:
        raise SystemExit("%s: unknown state %s" % (name, function["state"]))
    if function["module"] not in MODULES:
        raise SystemExit("%s: unknown module %s" % (name, function["module"]))
    for ctype, pname in function["params"]:
        if not IDENT.fullmatch(pname):
            raise SystemExit("%s: bad parameter name %r" % (name, pname))
    if function["state"] in ("constant", "unimplemented"):
        if not VALUE.fullmatch(str(function.get("value", ""))):
            raise SystemExit("%s: state %s needs a numeric value" % (name, function["state"]))
        if function["return"] == "void" or function["return"].endswith("*"):
            raise SystemExit("%s: state %s needs an integer return type" % (name, function["state"]))


def load_all(paths):
    functions = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)["functions"]
        names = set()
        for entry in entries:
            name = entry["name"]
            if not IDENT.fullmatch(name) or name in names:
                raise SystemExit("%s: bad or repeated export name %r" % (path, name))
            names.add(name)
            if "module" in entry:
                if name in functions:
                    raise SystemExit("%s: %s is defined twice" % (path, name))
                functions[name] = dict(entry)
            else:
                if name not in functions:
                    raise SystemExit("%s: %s overrides a function no earlier manifest defines" % (path, name))
                for key, value in entry.items():
                    if key != "name":
                        functions[name][key] = value
    result = sorted(functions.values(), key=lambda function: function["name"])
    for function in result:
        check(function)
    return result


def load(path):
    return load_all([path])


def structs(functions):
    names = set()
    for function in functions:
        for ctype in [function["return"]] + [ctype for ctype, _ in function["params"]]:
            for match in re.finditer(r"struct (\w+)", ctype):
                names.add(match.group(1))
            for word in IDENT.findall(re.sub(r"struct \w+", "", ctype)):
                if word not in BUILTIN and word != "const":
                    raise SystemExit("%s: unknown type word %s" % (function["name"], word))
    return sorted(names)


def declarator(ctype, name):
    return ctype + name if ctype.endswith("*") else ctype + " " + name


def signature(function):
    params = ", ".join(declarator(ctype, pname) for ctype, pname in function["params"]) or "void"
    return "%s(%s)" % (declarator(function["return"], "AGC_SYSV " + function["name"]), params)


def guest(functions):
    return [function for function in functions if function["state"] != "host"]


def emit_header(out, functions):
    functions = guest(functions)
    out.write("#ifndef AGC_EXPORTS_H\n#define AGC_EXPORTS_H\n\n#include \"agc_types.h\"\n\n")
    for name in structs(functions):
        out.write("struct %s;\n" % name)
    out.write("\n")
    for function in functions:
        out.write(signature(function) + ";\n")
    out.write("\n#endif\n")


def emit_stubs(out, functions):
    out.write("#include \"agc_debug.h\"\n#include \"agc_exports.h\"\n\n")
    for function in guest(functions):
        state = function["state"]
        if state == "done":
            continue
        out.write(signature(function) + "\n{\n")
        for _, pname in function["params"]:
            out.write("   (void)%s;\n" % pname)
        if state == "todo":
            out.write("   agc_fatal(\"%s\", \"not implemented\");\n}\n\n" % function["name"])
            continue
        if state == "unimplemented":
            out.write("   agc_report(\"%s\", \"not implemented\");\n" % function["name"])
        out.write("   return (%s)%s;\n}\n\n" % (function["return"], function["value"]))


def takes_command_buffer(function):
    params = function["params"]
    return bool(params) and params[0][0].replace(" ", "") == "structagc_cb*"


def emit_thunks(out, functions):
    functions = guest(functions)
    out.write("#include \"agc_api_trace.h\"\n\n")
    for function in functions:
        out.write("void %s(void);\n" % function["name"])
    out.write("\nconst void *const agc_trace_real[] = {\n")
    for function in functions:
        out.write("   (const void *)%s,\n" % function["name"])
    out.write("};\n\nconst struct agc_trace_function agc_trace_functions[] = {\n")
    for function in functions:
        out.write("   {\"%s\", %d, %s},\n" % (function["name"], len(function["params"]),
                                            "true" if takes_command_buffer(function) else "false"))
    out.write("};\n\nconst uint32_t agc_trace_function_count = %d;\n\n" % len(functions))
    for index, function in enumerate(functions):
        out.write("AGC_TRACE_THUNK(%s, %d)\n" % (function["name"], index))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--header")
    ap.add_argument("--stubs")
    ap.add_argument("--thunks")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    functions = load_all(args.manifests)
    if args.header:
        with open(args.header, "w", encoding="utf-8", newline="\n") as out:
            emit_header(out, functions)
    if args.stubs:
        with open(args.stubs, "w", encoding="utf-8", newline="\n") as out:
            emit_stubs(out, functions)
    if args.thunks:
        with open(args.thunks, "w", encoding="utf-8", newline="\n") as out:
            emit_thunks(out, functions)
    if args.summary:
        for module in MODULES:
            mine = [f for f in functions if f["module"] == module]
            counts = ", ".join("%d %s" % (sum(f["state"] == state for f in mine), state) for state in STATES)
            print("%s: %d exports: %s" % (module, len(mine), counts))


if __name__ == "__main__":
    sys.exit(main())
