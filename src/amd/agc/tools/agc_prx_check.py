import argparse
import collections
import struct
import sys

MACHINE_AMD64 = 0x8664
MAGIC_PE32_PLUS = 0x20b
FILE_RELOCS_STRIPPED = 0x0001
FILE_EXECUTABLE_IMAGE = 0x0002
FILE_DLL = 0x2000
SECTION_CODE = 0x00000020
SECTION_EXECUTE = 0x20000000
DIRECTORY_EXPORT = 0
DIRECTORY_IMPORT = 1
IMPORT_BY_ORDINAL = 1 << 63

Shell = collections.namedtuple("Shell", "name exports imports")
Section = collections.namedtuple("Section", "name rva size offset")


class Malformed(Exception):
    pass


def require(condition, message, *values):
    if not condition:
        raise Malformed(message % values)


def power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def align(value, alignment):
    return -(-value // alignment) * alignment


class Image:
    def __init__(self, data):
        self.data = data
        require(self.read(0, 2) == b"MZ", "no MZ signature")
        pe, = self.unpack("<I", 0x3c)
        require(self.read(pe, 4) == b"PE\0\0", "no PE signature")
        machine, count, _, _, _, optional_size, flags = self.unpack("<HHIIIHH", pe + 4)
        require(machine == MACHINE_AMD64, "machine 0x%04x is not AMD64", machine)
        require(flags & FILE_EXECUTABLE_IMAGE, "not an executable image")
        require(flags & FILE_DLL, "not a DLL")
        require(not flags & FILE_RELOCS_STRIPPED, "relocations are stripped, the image cannot leave its base")

        optional = pe + 24
        require(optional_size >= 112, "optional header of %d bytes", optional_size)
        magic, = self.unpack("<H", optional)
        require(magic == MAGIC_PE32_PLUS, "optional header magic 0x%x is not PE32+", magic)
        entry, = self.unpack("<I", optional + 16)
        require(entry == 0, "entry point 0x%x in a shell without code", entry)
        section_alignment, file_alignment = self.unpack("<II", optional + 32)
        require(power_of_two(file_alignment) and 0x200 <= file_alignment <= 0x10000,
                "file alignment 0x%x", file_alignment)
        require(power_of_two(section_alignment) and section_alignment >= file_alignment,
                "section alignment 0x%x", section_alignment)
        image_size, headers_size = self.unpack("<II", optional + 56)
        directories, = self.unpack("<I", optional + 108)
        require(optional_size >= 112 + 8 * directories, "%d data directories leave the optional header", directories)
        self.directories = [self.unpack("<II", optional + 112 + 8 * index) for index in range(directories)]

        table = optional + optional_size
        require(count > 0, "no sections")
        require(table + 40 * count <= headers_size <= len(data) and headers_size % file_alignment == 0,
                "headers size 0x%x", headers_size)
        self.sections = []
        rva = align(headers_size, section_alignment)
        for index in range(count):
            name, virtual_size, section_rva, raw_size, raw_offset, _, _, _, _, characteristics = self.unpack(
                "<8sIIIIIIHHI", table + 40 * index)
            name = name.rstrip(b"\0").decode("ascii", "replace")
            require(section_rva == rva, "section %s at RVA 0x%x, expected 0x%x", name, section_rva, rva)
            require(not characteristics & (SECTION_CODE | SECTION_EXECUTE), "section %s holds code", name)
            require(raw_offset % file_alignment == 0 and raw_size % file_alignment == 0,
                    "section %s is not file-aligned", name)
            require(not raw_size or headers_size <= raw_offset and raw_offset + raw_size <= len(data),
                    "section %s lies outside the file", name)
            self.sections.append(Section(name, section_rva, min(virtual_size or raw_size, raw_size), raw_offset))
            rva = align(section_rva + (virtual_size or raw_size), section_alignment)
        require(image_size == rva, "image size 0x%x, the sections end at 0x%x", image_size, rva)

    def read(self, at, size):
        require(0 <= at and at + size <= len(self.data), "truncated at 0x%x", at)
        return self.data[at:at + size]

    def unpack(self, layout, at):
        return struct.unpack(layout, self.read(at, struct.calcsize(layout)))

    def directory(self, index):
        return self.directories[index] if index < len(self.directories) else (0, 0)

    def section(self, rva, size):
        for section in self.sections:
            if section.rva <= rva and rva + size <= section.rva + section.size:
                return section
        raise Malformed("RVA 0x%x (%d bytes) is not in a section's file data" % (rva, size))

    def at(self, layout, rva):
        size = struct.calcsize(layout)
        section = self.section(rva, size)
        return self.unpack(layout, section.offset + rva - section.rva)

    def text(self, rva, what):
        section = self.section(rva, 1)
        start = section.offset + rva - section.rva
        end = self.data.find(b"\0", start, section.offset + section.size)
        require(end > start, "%s at RVA 0x%x is empty or unterminated", what, rva)
        text = self.data[start:end]
        require(all(0x20 < byte < 0x7f for byte in text), "%s %r is not printable ASCII", what, text)
        return text.decode("ascii")


def read_exports(image):
    rva, size = image.directory(DIRECTORY_EXPORT)
    require(rva and size >= 40, "no export directory")
    _, _, _, _, name, _, functions, names, function_table, name_table, ordinal_table = image.at("<IIHHIIIIIII", rva)
    dll_name = image.text(name, "DLL name")
    require(functions == names, "%d functions but %d names", functions, names)
    exports = []
    ordinals = set()
    for index in range(names):
        export = image.text(image.at("<I", name_table + 4 * index)[0], "export name")
        require(not exports or exports[-1][0] < export,
                "export %s is out of order, the loader's binary search would miss it", export)
        ordinal, = image.at("<H", ordinal_table + 2 * index)
        require(ordinal < functions and ordinal not in ordinals, "export %s has ordinal index %d", export, ordinal)
        ordinals.add(ordinal)
        function, = image.at("<I", function_table + 4 * ordinal)
        require(rva <= function < rva + size, "export %s is not a forwarder", export)
        target = image.text(function, "forwarder")
        require(function + len(target) < rva + size, "forwarder %s leaves the export directory", target)
        module, _, symbol = target.partition(".")
        require(module and symbol and "." not in symbol, "forwarder %s is not module.symbol", target)
        exports.append((export, target))
    return dll_name, exports


def read_imports(image):
    rva, size = image.directory(DIRECTORY_IMPORT)
    imports = []
    while rva:
        lookup, _, _, name, addresses = image.at("<IIIII", rva + 20 * len(imports))
        require(20 * (len(imports) + 1) <= size, "import descriptors leave the import directory")
        if not (lookup or name or addresses):
            break
        require(name and addresses, "import descriptor %d is incomplete", len(imports))
        dll = image.text(name, "imported DLL name")
        symbols = []
        while True:
            thunk, = image.at("<Q", (lookup or addresses) + 8 * len(symbols))
            image.at("<Q", addresses + 8 * len(symbols))
            if not thunk:
                break
            require(not thunk & IMPORT_BY_ORDINAL and thunk < 1 << 32, "%s: import thunk 0x%x is not a name", dll, thunk)
            symbols.append(image.text(thunk + 2, "imported symbol"))
        require(symbols, "nothing is imported from %s", dll)
        imports.append((dll, symbols))
    return imports


def parse(data):
    image = Image(data)
    name, exports = read_exports(image)
    imports = read_imports(image)
    loaded = {dll.lower() for dll, _ in imports}
    for module in sorted({target.partition(".")[0] for _, target in exports}):
        require(module.lower() + ".dll" in loaded,
                "forwards to %s without importing %s.dll, so the runtime is not loaded together with the shell",
                module, module)
    return Shell(name, exports, imports)


def cross_check(path, shell):
    try:
        import pefile
    except ImportError:
        raise SystemExit("--pefile needs the pefile module")

    def text(value):
        return (value or b"").decode("ascii", "replace")

    try:
        pe = pefile.PE(path)
    except pefile.PEFormatError as error:
        raise Malformed("pefile: %s" % error)
    try:
        directory = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
        exports = [(text(symbol.name), text(symbol.forwarder)) for symbol in (directory.symbols if directory else [])]
        imports = [(text(entry.dll), [text(symbol.name) for symbol in entry.imports])
                   for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])]
    finally:
        pe.close()
    require(sorted(exports) == sorted(shell.exports), "pefile reads %d exports that differ from ours", len(exports))
    require(imports == shell.imports, "pefile reads imports %r", imports)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shell")
    ap.add_argument("--pefile", action="store_true")
    args = ap.parse_args()
    try:
        with open(args.shell, "rb") as f:
            shell = parse(f.read())
        if args.pefile:
            cross_check(args.shell, shell)
    except (OSError, Malformed) as error:
        print("%s: %s" % (args.shell, error), file=sys.stderr)
        return 1
    for name, target in shell.exports:
        print("%s\t%s" % (name, target))
    modules = sorted({target.partition(".")[0] for _, target in shell.exports})
    anchors = ", ".join("%s!%s" % (dll, symbol) for dll, symbols in shell.imports for symbol in symbols)
    print("%s: %d exports forward to %s; imports %s" % (
        shell.name, len(shell.exports), ", ".join(modules) or "nothing", anchors or "nothing"), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
