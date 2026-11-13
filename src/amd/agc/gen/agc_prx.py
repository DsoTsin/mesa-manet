import argparse
import base64
import hashlib
import json
import os
import re
import struct
import sys

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
MODULE = re.compile(r"[A-Za-z0-9_\-]+")
NID_SUFFIX = bytes.fromhex("518d64a635ded8c1e6b039b1c3e55230")
RUNTIME = "agcdrv"
ANCHOR = "agc_runtime_anchor"

SECTION_RVA = 0x1000
SECTION_ALIGNMENT = 0x1000
FILE_ALIGNMENT = 0x200
HEADERS_SIZE = 0x200
PE_OFFSET = 0x40
IMAGE_BASE = 0x180000000
DIRECTORIES = 16
DIRECTORY_EXPORT = 0
DIRECTORY_IMPORT = 1
DIRECTORY_IAT = 12

MACHINE_AMD64 = 0x8664
MAGIC_PE32_PLUS = 0x20b
FILE_EXECUTABLE_IMAGE = 0x0002
FILE_LARGE_ADDRESS_AWARE = 0x0020
FILE_DLL = 0x2000
SUBSYSTEM_WINDOWS_GUI = 2
DLL_HIGH_ENTROPY_VA = 0x0020
DLL_DYNAMIC_BASE = 0x0040
DLL_NX_COMPAT = 0x0100
SECTION_INITIALIZED_DATA = 0x00000040
SECTION_READ = 0x40000000
SECTION_WRITE = 0x80000000

COFF_HEADER = (
    ("H", "machine"), ("H", "sections"), ("I", "timestamp"), ("I", "symbol_table"), ("I", "symbols"),
    ("H", "optional_header_size"), ("H", "characteristics"),
)
OPTIONAL_HEADER = (
    ("H", "magic"), ("B", "linker_major"), ("B", "linker_minor"), ("I", "code_size"),
    ("I", "initialized_data_size"), ("I", "uninitialized_data_size"), ("I", "entry_point"), ("I", "code_base"),
    ("Q", "image_base"), ("I", "section_alignment"), ("I", "file_alignment"),
    ("H", "os_major"), ("H", "os_minor"), ("H", "image_major"), ("H", "image_minor"),
    ("H", "subsystem_major"), ("H", "subsystem_minor"), ("I", "win32_version"),
    ("I", "image_size"), ("I", "headers_size"), ("I", "checksum"), ("H", "subsystem"), ("H", "dll_characteristics"),
    ("Q", "stack_reserve"), ("Q", "stack_commit"), ("Q", "heap_reserve"), ("Q", "heap_commit"),
    ("I", "loader_flags"), ("I", "directories"),
)
SECTION_HEADER = (
    ("8s", "name"), ("I", "virtual_size"), ("I", "rva"), ("I", "raw_size"), ("I", "raw_offset"),
    ("I", "relocations_offset"), ("I", "line_numbers_offset"), ("H", "relocations"), ("H", "line_numbers"),
    ("I", "characteristics"),
)
EXPORT_DIRECTORY = (
    ("I", "characteristics"), ("I", "timestamp"), ("H", "major"), ("H", "minor"), ("I", "name"),
    ("I", "ordinal_base"), ("I", "functions"), ("I", "names"), ("I", "function_table"), ("I", "name_table"),
    ("I", "ordinal_table"),
)
IMPORT_DESCRIPTOR = (
    ("I", "lookup_table"), ("I", "timestamp"), ("I", "forwarder_chain"), ("I", "name"), ("I", "address_table"),
)


def nid(name):
    digest = hashlib.sha1(name.encode("utf-8") + NID_SUFFIX).digest()
    return base64.b64encode(digest[7::-1], altchars=b"+-").decode("ascii").rstrip("=")


def align(value, alignment):
    return -(-value // alignment) * alignment


def size_of(layout):
    return struct.calcsize("<" + "".join(code for code, _ in layout))


def pack(layout, **fields):
    values = [fields.pop(name, b"" if code.endswith("s") else 0) for code, name in layout]
    if fields:
        raise KeyError(", ".join(sorted(fields)))
    return struct.pack("<" + "".join(code for code, _ in layout), *values)


def check_text(text, what):
    if not text or not text.isascii() or "\0" in text:
        raise SystemExit("export shell: invalid %s %r" % (what, text))


class Section:
    def __init__(self):
        self.data = bytearray()

    def rva(self, at=None):
        return SECTION_RVA + (len(self.data) if at is None else at)

    def reserve(self, size, alignment):
        self.data += bytes(-len(self.data) % alignment)
        at = len(self.data)
        self.data += bytes(size)
        return at

    def string(self, text):
        rva = self.rva()
        self.data += text.encode("ascii") + b"\0"
        return rva

    def put(self, at, code, *values):
        struct.pack_into("<" + code, self.data, at, *values)


def build_shell(dll_name, forwards, dependency=None):
    check_text(dll_name, "DLL name")
    forwards = list(forwards)
    for name, target in forwards:
        check_text(name, "export name")
        check_text(target, "forwarder")
        module, _, symbol = target.partition(".")
        if not module or not symbol:
            raise SystemExit("export shell: forwarder %r is not module.symbol" % target)
    forwards.sort(key=lambda forward: forward[0].encode("ascii"))
    for (before, _), (name, _) in zip(forwards, forwards[1:]):
        if before == name:
            raise SystemExit("export shell: %r is exported twice" % name)
    count = len(forwards)
    if count > 0xffff:
        raise SystemExit("export shell: %d exports do not fit 16-bit ordinals" % count)

    section = Section()
    directory = section.reserve(size_of(EXPORT_DIRECTORY), 4)
    functions = section.reserve(4 * count, 4)
    names = section.reserve(4 * count, 4)
    ordinals = section.reserve(2 * count, 2)
    dll_name_rva = section.string(dll_name)
    for index, (name, target) in enumerate(forwards):
        name_rva = section.string(name)
        target_rva = section.string(target)
        section.put(names + 4 * index, "I", name_rva)
        section.put(functions + 4 * index, "I", target_rva)
        section.put(ordinals + 2 * index, "H", index)
    exports_size = len(section.data) - directory
    section.data[directory:directory + size_of(EXPORT_DIRECTORY)] = pack(
        EXPORT_DIRECTORY, name=dll_name_rva, ordinal_base=1, functions=count, names=count,
        function_table=section.rva(functions), name_table=section.rva(names), ordinal_table=section.rva(ordinals))
    directories = [(0, 0)] * DIRECTORIES
    directories[DIRECTORY_EXPORT] = (section.rva(directory), exports_size)

    if dependency is not None:
        dll, symbol = dependency
        check_text(dll, "dependency")
        check_text(symbol, "dependency symbol")
        descriptors = section.reserve(2 * size_of(IMPORT_DESCRIPTOR), 8)
        lookup = section.reserve(16, 8)
        addresses = section.reserve(16, 8)
        hint = section.reserve(2, 2)
        section.string(symbol)
        dll_rva = section.string(dll)
        section.data[descriptors:descriptors + size_of(IMPORT_DESCRIPTOR)] = pack(
            IMPORT_DESCRIPTOR, lookup_table=section.rva(lookup), name=dll_rva, address_table=section.rva(addresses))
        section.put(lookup, "Q", section.rva(hint))
        section.put(addresses, "Q", section.rva(hint))
        directories[DIRECTORY_IMPORT] = (section.rva(descriptors), 2 * size_of(IMPORT_DESCRIPTOR))
        directories[DIRECTORY_IAT] = (section.rva(addresses), 16)

    virtual_size = len(section.data)
    raw_size = align(virtual_size, FILE_ALIGNMENT)
    optional = pack(
        OPTIONAL_HEADER, magic=MAGIC_PE32_PLUS, linker_major=14, initialized_data_size=raw_size,
        image_base=IMAGE_BASE, section_alignment=SECTION_ALIGNMENT, file_alignment=FILE_ALIGNMENT,
        os_major=6, subsystem_major=6, image_size=SECTION_RVA + align(virtual_size, SECTION_ALIGNMENT),
        headers_size=HEADERS_SIZE, subsystem=SUBSYSTEM_WINDOWS_GUI,
        dll_characteristics=DLL_HIGH_ENTROPY_VA | DLL_DYNAMIC_BASE | DLL_NX_COMPAT,
        stack_reserve=0x100000, stack_commit=0x1000, heap_reserve=0x100000, heap_commit=0x1000,
        directories=DIRECTORIES)
    optional += b"".join(struct.pack("<II", rva, size) for rva, size in directories)
    writable = SECTION_WRITE if dependency is not None else 0
    headers = b"MZ".ljust(0x3c, b"\0") + struct.pack("<I", PE_OFFSET) + b"PE\0\0"
    headers += pack(
        COFF_HEADER, machine=MACHINE_AMD64, sections=1, optional_header_size=len(optional),
        characteristics=FILE_EXECUTABLE_IMAGE | FILE_LARGE_ADDRESS_AWARE | FILE_DLL)
    headers += optional
    headers += pack(
        SECTION_HEADER, name=b".rdata", virtual_size=virtual_size, rva=SECTION_RVA, raw_size=raw_size,
        raw_offset=HEADERS_SIZE, characteristics=SECTION_INITIALIZED_DATA | SECTION_READ | writable)
    return headers.ljust(HEADERS_SIZE, b"\0") + bytes(section.data).ljust(raw_size, b"\0")


def load(paths):
    import agc_exports
    return agc_exports.load_all(paths)


def module_shell(functions, module, runtime, anchor, dll_name, empty=False):
    names = [function["name"] for function in functions if function["module"] == module]
    if bool(names) == empty:
        if empty:
            raise SystemExit("module %s is not empty: the manifest gives it %d functions" % (module, len(names)))
        known = ", ".join(sorted({function["module"] for function in functions}))
        raise SystemExit("no function of module %s in the manifest (it has %s)" % (module, known))
    forwards = [(nid(name), "%s.%s" % (runtime, name)) for name in names]
    return build_shell(dll_name, forwards, (runtime + ".dll", anchor))


def emit_def(out, functions, anchor):
    out.write("EXPORTS\n")
    targets = {anchor: anchor}
    for function in functions:
        name = function["name"]
        targets[name] = name if function["state"] == "host" else "agc_trace_thunk_" + name
    for name in sorted(targets):
        out.write("    %s\n" % (name if targets[name] == name else "%s=%s" % (name, targets[name])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, action="append")
    ap.add_argument("--module")
    ap.add_argument("--runtime", default=RUNTIME)
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--out")
    ap.add_argument("--def", dest="definitions")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--empty", action="store_true")
    args = ap.parse_args()
    if bool(args.module) != bool(args.out):
        ap.error("--module and --out go together")
    if not (args.out or args.definitions or args.list):
        ap.error("nothing to write: give --out, --def or --list")
    if not MODULE.fullmatch(args.runtime):
        ap.error("bad runtime name %r" % args.runtime)
    if not IDENT.fullmatch(args.anchor):
        ap.error("bad anchor name %r" % args.anchor)
    functions = load(args.manifest)
    if args.out:
        image = module_shell(functions, args.module, args.runtime, args.anchor, os.path.basename(args.out), args.empty)
        with open(args.out, "wb") as out:
            out.write(image)
    if args.definitions:
        with open(args.definitions, "w", encoding="utf-8", newline="\n") as out:
            emit_def(out, functions, args.anchor)
    if args.list:
        for function in functions:
            print("%s\t%s\t%s" % (nid(function["name"]), function["name"], function["module"]))


if __name__ == "__main__":
    sys.exit(main())
