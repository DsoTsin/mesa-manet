import argparse
import json
import re
import sys

OP_TABLES = (
    ("sop1", "SOP1", "s_"),
    ("sop2", "SOP2", "s_"),
    ("sopk", "SOPK", "s_"),
    ("sopc", "SOPC", "s_"),
    ("sopp", "SOPP", "s_"),
    ("vop1", "VOP1", "v_"),
    ("vop2", "VOP2", "v_"),
    ("vop3", "VOP3", "v_"),
    ("vop3p", "VOP3P", "v_"),
)
MEM_TABLES = (
    ("smem", "SMEM", "s_"),
    ("vmem_common", "VMEM", ""),
    ("mubuf_only", "MUBUF", ""),
    ("mtbuf", "MTBUF", "tbuffer_"),
    ("flat_only", "FLAT", ""),
)
TARGETS = ("gfx103", "gfx11", "gfx12")
MEM_KINDS = ("load", "store", "atomic", "control")
CMP_TYPES = ("f16", "f32", "f64", "i16", "u16", "i32", "u32", "i64", "u64")
RULES = (
    "bvh8", "bvh_stack8", "mac", "mac_legacy", "mad", "mad_legacy", "madmk", "madak", "dot2c", "dot4c",
    "dot_signed", "cmpk", "hwreg", "sendmsg", "waitcnt", "waitcnt_counter", "barrier", "mem_time", "interp",
    "wave_id", "find_first_zero", "dot2",
)
RULES_WITH_VALUE = ("dot_signed", "cmpk", "hwreg", "sendmsg", "dot2")

HOSTS = (
    ("gfx103", "gfx103", "rdna2"),
    ("gfx11", "gfx11", "rdna3"),
    ("gfx115", "gfx11", "rdna3_5"),
    ("gfx12", "gfx12", "rdna4"),
)
VOP3_VOP2_BASE = 0x100
VOP3_VOP1_BASE = 0x180
FLAT_ATOMIC_CSUB = 0x34
DS_ADD_RTN_F32 = 0x55
DS_ADD_RTN_F32_GFX11 = 0x79
GFX12_MSAA_LOAD = 0x18
GFX12_MTBUF_BASE = 0x80
IMAGE_ADJUST_BIT = 0x80
VINTRP_NAMES = ("v_interp_p1_f32", "v_interp_p2_f32", "v_interp_mov_f32")

def fail(message):
    raise SystemExit("agc_isa_gen: " + message)


def hex_of(text, what):
    if not text.startswith("0x"):
        fail("%s: %r is not a hex number" % (what, text))
    return int(text, 16)


class Xlat:
    def __init__(self, text, what):
        words = text.split()
        self.kind = words[0] if words else ""
        self.rule = None
        self.value = 0
        if self.kind in ("same", "drop", "no") and len(words) == 1:
            return
        if self.kind == "op" and len(words) == 2:
            self.value = hex_of(words[1], what)
            return
        if self.kind == "rule" and len(words) >= 2 and words[1] in RULES:
            self.rule = words[1]
            if (len(words) == 3) != (self.rule in RULES_WITH_VALUE) or len(words) > 3:
                fail("%s: rule %s takes %s value" % (what, self.rule, "a" if self.rule in RULES_WITH_VALUE else "no"))
            if len(words) == 3:
                self.value = hex_of(words[2], what)
            return
        fail("%s: bad translation %r" % (what, text))

    def c(self):
        rule = "AGC_ISA_RULE_%s" % self.rule.upper() if self.rule else "AGC_ISA_RULE_NONE"
        return "{AGC_ISA_XLAT_%s, %s, 0x%02x}" % (self.kind.upper(), rule, self.value)


class Entry:
    def __init__(self, table, raw, with_xlat):
        self.op = hex_of(raw["op"], table)
        self.name = raw["name"]
        what = "%s %s" % (table, self.name)
        self.flags = [flag for flag in raw["flags"].split("|") if flag]
        self.raw = raw
        if not re.fullmatch("[a-z][a-z0-9_]*", self.name):
            fail("%s: bad name" % what)
        for flag in self.flags:
            if not re.fullmatch("[A-Z][A-Z0-9_]*", flag):
                fail("%s: bad flag %r" % (what, flag))
        if with_xlat:
            self.xlat = {target: Xlat(raw[target], what) for target in TARGETS}

    def has(self, flag):
        return flag in self.flags

    def c_flags(self, prefix):
        return " | ".join(prefix + flag for flag in self.flags) or "0"

    def c_xlats(self):
        return ", ".join(self.xlat[target].c() for target in TARGETS)


class Guest:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        expected = [key for key, _, _ in OP_TABLES] + ["vopc"] + [key for key, _, _ in MEM_TABLES] + ["ds", "mimg"]
        if sorted(data) != sorted(expected):
            fail("%s: tables must be exactly %s" % (path, ", ".join(expected)))
        self.ops = {key: self.table(data, key) for key, _, _ in OP_TABLES}
        self.mem = {key: self.table(data, key) for key, _, _ in MEM_TABLES}
        self.ds = self.table(data, "ds", with_xlat=False)
        self.mimg = self.table(data, "mimg")
        self.vopc = self.table(data, "vopc", with_xlat=False)
        for entry in self.vopc:
            if entry.raw["type"] not in CMP_TYPES:
                fail("vopc %s: unknown type %s" % (entry.name, entry.raw["type"]))
            if entry.raw["cond"] != "class" and not 0 <= int(entry.raw["cond"]) < 16:
                fail("vopc %s: bad condition %s" % (entry.name, entry.raw["cond"]))
        for key in self.mem:
            for entry in self.mem[key]:
                if entry.raw["kind"] not in MEM_KINDS:
                    fail("%s %s: unknown kind %s" % (key, entry.name, entry.raw["kind"]))
        self.mubuf = self.chain("mubuf", self.mem["vmem_common"], self.mem["mubuf_only"])
        self.flat = self.chain("flat", self.mem["vmem_common"], self.mem["flat_only"])

    @staticmethod
    def table(data, key, with_xlat=True):
        entries = [Entry(key, raw, with_xlat) for raw in data[key]]
        seen = {}
        for entry in entries:
            if entry.op in seen:
                fail("%s: %s and %s share opcode %#x" % (key, seen[entry.op], entry.name, entry.op))
            seen[entry.op] = entry.name
        if not entries or len(entries) >= 255:
            fail("%s: %d entries do not fit the index type" % (key, len(entries)))
        return entries

    @staticmethod
    def chain(what, first, second):
        ops = {entry.op for entry in first}
        for entry in second:
            if entry.op in ops:
                fail("%s: opcode %#x is listed twice" % (what, entry.op))
        return first + second


def enum_name(prefix, strip, name):
    if not name.startswith(strip):
        fail("%s does not start with %s" % (name, strip))
    return "AGC_ISA_%s_%s" % (prefix, name[len(strip):].upper())


def emit_opcodes(out, guest):
    w = out.write
    w("#ifndef AGC_ISA_OPCODES_H\n#define AGC_ISA_OPCODES_H\n\n")
    groups = [(key, prefix, strip, guest.ops[key]) for key, prefix, strip in OP_TABLES]
    groups.append(("vopc", "VOPC", "v_", guest.vopc))
    groups += [(key, prefix, strip, guest.mem[key]) for key, prefix, strip in MEM_TABLES]
    groups.append(("ds", "DS", "ds_", guest.ds))
    groups.append(("mimg", "MIMG", "image_", guest.mimg))
    seen = set()
    for key, prefix, strip, entries in groups:
        w("enum agc_isa_%s_opcode {\n" % key)
        for entry in entries:
            name = enum_name(prefix, strip, entry.name)
            if name in seen:
                fail("opcode name %s is produced twice" % name)
            seen.add(name)
            w("   %s = 0x%02x,\n" % (name, entry.op))
        w("};\n\n")
    w("#endif\n")


def emit_index(out, name, entries):
    w = out.write
    size = max(entry.op for entry in entries) + 1
    w("static const uint8_t %s[0x%x] = {\n" % (name, size))
    for position, entry in enumerate(entries):
        w("   [0x%02x] = %d,\n" % (entry.op, position + 1))
    w("};\n\n")


def emit_tables(out, guest):
    w = out.write
    w("#ifndef AGC_ISA_TABLES_H\n#define AGC_ISA_TABLES_H\n\n")
    for key, _, _ in OP_TABLES:
        w("static const struct agc_isa_op_info agc_isa_%s_ops[] = {\n" % key)
        for entry in guest.ops[key]:
            w("   {\"%s\", %s, 0x%02x, %s},\n" % (entry.name, entry.c_flags("AGC_ISA_FLAG_"), entry.op, entry.c_xlats()))
        w("};\n\n")
        emit_index(out, "agc_isa_%s_index" % key, guest.ops[key])
    w("static const struct agc_isa_vopc_info agc_isa_vopc_ops[] = {\n")
    for entry in guest.vopc:
        cond = "AGC_ISA_CMP_COND_CLASS" if entry.raw["cond"] == "class" else entry.raw["cond"]
        w("   {\"%s\", %s, 0x%02x, 0x%02x, AGC_ISA_CMP_%s, %s, %s},\n" % (
            entry.name, entry.c_flags("AGC_ISA_FLAG_"), entry.op, hex_of(entry.raw["gfx11"], entry.name),
            entry.raw["type"].upper(), cond, "true" if entry.raw["exec"] else "false"))
    w("};\n\n")
    emit_index(out, "agc_isa_vopc_index", guest.vopc)
    for key, _, _ in MEM_TABLES:
        w("static const struct agc_isa_mem_op agc_isa_%s_ops[] = {\n" % key)
        for entry in guest.mem[key]:
            w("   {\"%s\", %s, 0x%02x, AGC_ISA_MEM_%s, %d, %s},\n" % (
                entry.name, entry.c_flags("AGC_ISA_FLAG_"), entry.op, entry.raw["kind"].upper(),
                entry.raw["dwords"], entry.c_xlats()))
        w("};\n\n")
    emit_index(out, "agc_isa_smem_index", guest.mem["smem"])
    emit_index(out, "agc_isa_mtbuf_index", guest.mem["mtbuf"])
    for key, prefix, entries in (("mubuf", "buffer_", guest.mubuf), ("flat", "flat_", guest.flat)):
        emit_index(out, "agc_isa_%s_index" % key, entries)
        size = max(entry.op for entry in entries) + 1
        w("static const char *const agc_isa_%s_names[0x%x] = {\n" % (key, size))
        for entry in entries:
            w("   [0x%02x] = \"%s%s\",\n" % (entry.op, prefix, entry.name))
        w("};\n\n")
    w("static const struct agc_isa_ds_op agc_isa_ds_ops[] = {\n")
    for entry in guest.ds:
        w("   {\"%s\", 0x%02x, %d, %d, %d, %s},\n" % (
            entry.name, entry.op, entry.raw["dst"], entry.raw["data0"], entry.raw["data1"],
            entry.c_flags("AGC_ISA_DS_FLAG_")))
    w("};\n\n")
    emit_index(out, "agc_isa_ds_index", guest.ds)
    w("static const struct agc_isa_image_op agc_isa_image_ops[] = {\n")
    for entry in guest.mimg:
        w("   {\"%s\", %s, 0x%02x, %s},\n" % (entry.name, entry.c_flags("AGC_ISA_IMG_"), entry.op, entry.c_xlats()))
    w("};\n\n")
    emit_index(out, "agc_isa_image_index", guest.mimg)
    w("#endif\n")


class Rules:
    def __init__(self, guest):
        self.entries = []
        encodings = ("Sop1", "Sop2", "Sopk", "Sopc", "Sopp", "Vop1", "Vop2", "Vop3", "Vop3p")
        for (key, _, _), encoding in zip(OP_TABLES, encodings):
            base = {"Vop1": VOP3_VOP1_BASE, "Vop2": VOP3_VOP2_BASE}.get(encoding)
            for entry in guest.ops[key]:
                self.add(encoding, entry.op, entry.name, lambda host, column, entry=entry, encoding=encoding:
                         self.emit(self.translate(entry, column), encoding))
                if base is not None and not entry.has("NO_VOP3"):
                    self.add("Vop3", base + entry.op, entry.name, lambda host, column, entry=entry, base=base:
                             self.promoted(self.translate(entry, column), base))
        for entry in guest.vopc:
            for encoding in ("Vopc", "Vop3"):
                self.add(encoding, entry.op, entry.name, lambda host, column, entry=entry, encoding=encoding:
                         self.compare(entry, host, encoding))
        for op, name in enumerate(VINTRP_NAMES):
            self.add("Vintrp", op, name, lambda host, column, op=op:
                     ("op", "Vintrp", op) if host == "gfx103" else ("rule",))
        self.add("Exp", 0, "exp", lambda host, column: ("op", "Exp", 0))
        for entry in guest.mem["smem"]:
            self.add("Smem", entry.op, entry.name, lambda host, column, entry=entry:
                     self.emit(self.translate(entry, column), "Smem"))
        for entry in guest.mubuf:
            self.add("Mubuf", entry.op, "buffer_" + entry.name, lambda host, column, entry=entry:
                     self.emit(self.translate(entry, column), "Vbuffer" if host == "gfx12" else "Mubuf"))
        for entry in guest.mem["mtbuf"]:
            self.add("Mtbuf", entry.op, entry.name, lambda host, column, entry=entry:
                     self.typed_buffer(self.translate(entry, column), host))
        flat_only = {entry.op for entry in guest.mem["flat_only"]}
        for entry in guest.flat:
            global_only = entry.op in flat_only or entry.op == FLAT_ATOMIC_CSUB
            for encoding in ("Flat", "Global", "Scratch"):
                if (global_only and encoding != "Global") or (entry.raw["kind"] == "atomic" and encoding == "Scratch"):
                    continue
                self.add(encoding, entry.op, "%s_%s" % (encoding.lower(), entry.name),
                         lambda host, column, entry=entry, encoding=encoding:
                         self.emit(self.translate(entry, column), encoding))
        for entry in guest.ds:
            self.add("Ds", entry.op, entry.name, lambda host, column, entry=entry: self.local_data(entry, host))
        images = {entry.op: entry for entry in guest.mimg}
        for entry in guest.mimg:
            self.add("Mimg", entry.op, entry.name, lambda host, column, entry=entry:
                     self.image(entry, images, host, column))

    def add(self, encoding, opcode, name, emit):
        self.entries.append((encoding, opcode, name, {host: emit(host, column) for host, column, _ in HOSTS}))

    @staticmethod
    def translate(entry, column):
        x = entry.xlat[column]
        if x.kind == "same":
            return ("op", entry.op)
        return ("op", x.value) if x.kind == "op" else (x.kind,)

    @staticmethod
    def emit(x, encoding):
        return ("op", encoding, x[1]) if x[0] == "op" else x

    @staticmethod
    def promoted(x, base):
        return ("op", "Vop3", base + x[1]) if x[0] == "op" else x

    @staticmethod
    def compare(entry, host, encoding):
        is_float = entry.raw["type"].startswith("f")
        if host == "gfx12" and entry.raw["cond"] in ("0", "15" if is_float else "7"):
            return ("rule",)
        if host == "gfx103":
            return ("op", encoding, entry.op)
        return ("op", encoding, hex_of(entry.raw["gfx11"], entry.name))

    @staticmethod
    def typed_buffer(x, host):
        if x[0] == "op" and host == "gfx12":
            return ("op", "Vbuffer", GFX12_MTBUF_BASE | x[1])
        return Rules.emit(x, "Mtbuf")

    @staticmethod
    def local_data(entry, host):
        if entry.has("STACK8"):
            return ("rule",)
        removed = ("SRC2", "GWS", "GDS_ONLY", "NOT_GFX12") if host == "gfx12" else ("SRC2",)
        if any(entry.has(flag) for flag in removed):
            return ("no",)
        if host != "gfx103" and entry.op == DS_ADD_RTN_F32:
            return ("op", "Ds", DS_ADD_RTN_F32_GFX11)
        return ("op", "Ds", entry.op)

    @staticmethod
    def image(entry, images, host, column):
        if entry.has("BVH8"):
            return ("rule",)
        plain = entry
        if entry.has("ADJUST"):
            plain = images.get(entry.op & ~IMAGE_ADJUST_BIT)
            if plain is None:
                return ("no",)
        x = Rules.translate(plain, column)
        if x[0] != "op":
            return x
        if host != "gfx12":
            return ("op", "Mimg", x[1], plain.name)
        sampled = entry.has("SAMPLER") or entry.has("GATHER") or x[1] == GFX12_MSAA_LOAD
        return ("op", "Vsample" if sampled else "Vimage", x[1], plain.name)


def report(rules):
    for host, _, key in HOSTS:
        counts = {"same": 0, "moved": 0, "rule": 0, "drop": 0, "no": 0}
        listed = {"rule": [], "drop": [], "no": []}
        for encoding, opcode, name, emits in rules.entries:
            emit = emits[host]
            if emit[0] == "op":
                counts["same" if emit[1:3] == (encoding, opcode) else "moved"] += 1
            else:
                counts[emit[0]] += 1
                listed[emit[0]].append("%s %s" % (encoding, name))
        print("%s (%s): %d guest forms: %s" % (host, key, len(rules.entries),
                                               ", ".join("%s %d" % item for item in counts.items())))
        for kind, names in listed.items():
            for name in names:
                print("   %-4s %s" % (kind, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps5", required=True)
    ap.add_argument("--out-opcodes")
    ap.add_argument("--out-tables")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    guest = Guest(args.ps5)
    if args.report:
        report(Rules(guest))

    outputs = ((args.out_opcodes, emit_opcodes, guest), (args.out_tables, emit_tables, guest))
    for path, emit, source in outputs:
        if path:
            with open(path, "w", encoding="utf-8", newline="\n") as out:
                emit(out, source)


if __name__ == "__main__":
    sys.exit(main())
