import argparse
import collections
import json
import re
import sys

BANKS = (("cx", 0x28000, 0x400), ("sh", 0xB000, 0x400), ("uc", 0x30000, 0x2000))
HOSTS = ("gfx103", "gfx11", "gfx115", "gfx12")
CLASSES = ("UNDEFINED", "PASS", "MOVE", "FIELD", "ADDR", "DROP", "SPLIT", "MERGE", "GUEST", "PENDING")
HOOKED = ("SPLIT", "MERGE", "GUEST")
VALUE_PRESERVING = ("PASS", "MOVE", "ADDR")
RULE_KEYS = ("match", "bank", "first", "last", "hosts", "class", "hook", "to", "drop", "same", "strict")
NAME_KEYS = ("bank", "at", "name", "count")
ALL_BITS = 0xFFFFFFFF


def bank_of(at):
    for index, (_, base, length) in enumerate(BANKS):
        if base <= at < base + length * 4:
            return index
    return None


def bank_index(name):
    for index, (bank_name, _, _) in enumerate(BANKS):
        if bank_name == name:
            return index
    raise SystemExit("unknown register bank %r" % (name,))


class Chip:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.types = data["register_types"]
        self.enums = data["enums"]
        self.by_name = {}
        self.by_at = {}
        for mapping in data["register_mappings"]:
            at = mapping["map"]["at"]
            self.by_name[mapping["name"]] = (at, mapping.get("type_ref"))
            self.by_at[at] = mapping["name"]

    def fields(self, type_ref):
        if type_ref is None or type_ref not in self.types:
            return None
        out = {}
        for field in self.types[type_ref]["fields"]:
            out[field["name"]] = (field["bits"][0], field["bits"][1], field.get("enum_ref"))
        return out

    def enum(self, enum_ref):
        if enum_ref is None or enum_ref not in self.enums:
            return None
        return {entry["name"]: entry["value"] for entry in self.enums[enum_ref]["entries"]}


def mask(lo, hi):
    return ((1 << (hi - lo + 1)) - 1) << lo


def layout(guest, gtype, host, htype, drop=(), same=()):
    gf = guest.fields(gtype)
    hf = host.fields(htype)
    if gf is None or hf is None:
        return None
    reasons = []
    keep = 0
    moves = []
    dropped = 0
    for name, (glo, ghi, genum) in sorted(gf.items(), key=lambda item: item[1][0]):
        if name in drop:
            dropped |= mask(glo, ghi)
            continue
        if name not in hf:
            reasons.append("field %s is gone" % name)
            continue
        hlo, hhi, henum = hf[name]
        if hhi - hlo < ghi - glo:
            reasons.append("field %s is narrower" % name)
            continue
        ge = guest.enum(genum)
        he = host.enum(henum)
        if name not in same and ge is not None and he is not None:
            for entry, value in ge.items():
                if he.get(entry) != value:
                    reasons.append("enum %s of field %s differs at %s" % (genum, name, entry))
                    break
        if hlo == glo:
            keep |= mask(glo, ghi)
        else:
            moves.append((glo, hlo, ghi - glo + 1))
    added = 0
    for name, (hlo, hhi, _) in hf.items():
        if name not in gf:
            added |= mask(hlo, hhi)
    return reasons, keep, tuple(moves), dropped, added


def compare(guest, gtype, host, htype):
    found = layout(guest, gtype, host, htype)
    if found is None:
        return "same", None, []
    reasons, keep, moves, _, _ = found
    if reasons:
        return "differs", None, reasons
    if moves:
        return "moved", (keep, moves), []
    gf = guest.fields(gtype)
    hf = host.fields(htype)
    if set(gf) == set(hf) and all(gf[n][:2] == hf[n][:2] for n in gf):
        return "same", None, []
    return "host_adds", None, []


def declared_program(label, guest, gtype, host, htype, rule):
    found = layout(guest, gtype, host, htype, rule.drop, rule.same)
    if found is None:
        raise SystemExit("%s: FIELD needs the fields of both registers" % label)
    gf = guest.fields(gtype)
    for name in rule.drop + rule.same:
        if name not in gf:
            raise SystemExit("%s: the guest register has no field %s" % (label, name))
    reasons, keep, moves, dropped, added = found
    if reasons:
        raise SystemExit("%s: %s" % (label, "; ".join(reasons)))
    if not moves and not rule.strict:
        keep |= ALL_BITS & ~dropped & ~added
    taken = keep
    for _, dst, width in moves:
        bits = mask(dst, dst + width - 1)
        if taken & bits:
            raise SystemExit("%s: two guest fields land on the same host bits" % label)
        taken |= bits
    return keep, moves


class Rule:
    def __init__(self, raw):
        self.label = json.dumps(raw, sort_keys=True)
        for key in raw:
            if key not in RULE_KEYS:
                raise SystemExit("override %s: unknown key %s" % (self.label, key))
        self.pattern = re.compile(raw.get("match", ".*"))
        self.bank = bank_index(raw["bank"]) if "bank" in raw else None
        self.first = int(raw.get("first", "0"), 0)
        self.last = int(raw.get("last", "0xffff"), 0)
        if ("first" in raw or "last" in raw) and self.bank is None:
            raise SystemExit("override %s: an offset range needs a bank" % self.label)
        if "match" not in raw and self.bank is None:
            raise SystemExit("override %s: needs a name pattern or a bank" % self.label)
        self.cls = raw.get("class")
        if self.cls not in CLASSES or self.cls in ("UNDEFINED",):
            raise SystemExit("override %s: bad class %s" % (self.label, self.cls))
        self.hook = raw.get("hook")
        if (self.cls in HOOKED) != (self.hook is not None):
            raise SystemExit("override %s: a hook is given exactly for SPLIT, MERGE and GUEST" % self.label)
        if self.hook is not None and not re.fullmatch(r"[a-z][a-z0-9_]*", self.hook):
            raise SystemExit("override %s: bad hook name %r" % (self.label, self.hook))
        self.hosts = tuple(raw.get("hosts", HOSTS))
        for host in self.hosts:
            if host not in HOSTS:
                raise SystemExit("override %s: unknown host %s" % (self.label, host))
        self.to = raw.get("to")
        self.drop = tuple(raw.get("drop", ()))
        self.same = tuple(raw.get("same", ()))
        self.strict = bool(raw.get("strict", False))
        self.declared = "drop" in raw or "same" in raw or "strict" in raw
        if self.declared and self.cls != "FIELD":
            raise SystemExit("override %s: drop, same and strict describe a FIELD conversion" % self.label)
        self.used = False

    def matches(self, name, host, bank, offset):
        if host not in self.hosts:
            return None
        if self.bank is not None and (bank != self.bank or not self.first <= offset <= self.last):
            return None
        return self.pattern.fullmatch(name)


class Rules:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.names = {}
        for raw in data.get("names", ()):
            for key in raw:
                if key not in NAME_KEYS:
                    raise SystemExit("name %s: unknown key %s" % (json.dumps(raw, sort_keys=True), key))
            bank = bank_index(raw["bank"])
            first = int(raw["at"], 0)
            count = raw.get("count", 1)
            for index in range(count):
                key = (bank, first + index)
                if key in self.names:
                    raise SystemExit("register %s:%#x is named twice" % (BANKS[bank][0], first + index))
                self.names[key] = raw["name"] % index if "count" in raw else raw["name"]
        self.rules = [Rule(raw) for raw in data["rules"]]

    def find(self, name, host, bank, offset):
        for rule in self.rules:
            found = rule.matches(name, host, bank, offset)
            if found is not None:
                rule.used = True
                return rule, found
        return None

    def unused(self):
        return [rule.label for rule in self.rules if not rule.used]


def guest_universe(guest, defaults_path, names=None):
    regs = {}
    used = set()
    names = dict(names or {})
    taken = set(guest.by_name)
    for name, (at, type_ref) in guest.by_name.items():
        bank = bank_of(at)
        if bank is None:
            continue
        regs[(bank, (at - BANKS[bank][1]) // 4)] = (name, type_ref)
    if defaults_path:
        with open(defaults_path, encoding="utf-8") as f:
            defaults = json.load(f)
        spaces = {name: index for index, (name, _, _) in enumerate(BANKS)}
        for groups in defaults["versions"].values():
            for group in groups:
                bank = spaces[group["space"]]
                for offset, _ in group["registers"]:
                    if offset >= BANKS[bank][2]:
                        raise SystemExit("default register %s:%#x is outside the table" % (group["space"], offset))
                    if (bank, offset) not in regs:
                        name = names.pop((bank, offset), "PS5_%s_%04X" % (group["space"].upper(), offset))
                        if name in taken:
                            raise SystemExit("register name %s is used twice" % name)
                        taken.add(name)
                        regs[(bank, offset)] = (name, None)
                    used.add((bank, offset))
    if names:
        raise SystemExit("names for registers that need none: %s" % ", ".join(
            "%s:%#x" % (BANKS[bank][0], offset) for bank, offset in sorted(names)))
    return regs, used


def classify(guest, regs, host_name, host, rules, progs, hooks):
    table = {}
    notes = {}
    decided = {}
    for (bank, offset), (name, gtype) in sorted(regs.items()):
        forced = rules.find(name, host_name, bank, offset)
        entry = None
        htype = None
        if name in host.by_name:
            hat, htype = host.by_name[name]
            hbank = bank_of(hat)
            if hbank != bank:
                notes[(bank, offset)] = "moved to another bank"
                htype = None
            else:
                hoff = (hat - BANKS[bank][1]) // 4
                kind, prog, reasons = compare(guest, gtype, host, htype)
                if kind in ("same", "host_adds"):
                    entry = ["PASS" if hoff == offset else "MOVE", hoff, 0]
                elif kind == "moved":
                    entry = ["FIELD", hoff, progs.setdefault(prog, len(progs))]
                else:
                    notes[(bank, offset)] = "; ".join(reasons)
                    entry = ["PENDING", hoff, 0]
        else:
            notes[(bank, offset)] = "not defined on the host"
        if forced is not None:
            rule, found = forced
            label = "override for %s on %s" % (name, host_name)
            cls = rule.cls
            hoff = entry[1] if entry else offset
            target = found.expand(rule.to) if rule.to is not None else None
            if target is not None:
                if target not in host.by_name or bank_of(host.by_name[target][0]) != bank:
                    raise SystemExit("%s: target %s is not in the same bank" % (label, target))
                hoff = (host.by_name[target][0] - BANKS[bank][1]) // 4
                htype = host.by_name[target][1]
            if cls == "ADDR" and (entry is None or entry[0] not in VALUE_PRESERVING):
                raise SystemExit("%s: ADDR needs a register the host has with the same layout" % label)
            if cls in ("MOVE", "PASS", "FIELD") and entry is None and target is None:
                raise SystemExit("%s: no host register" % label)
            extra = 0
            if cls == "FIELD" and rule.declared:
                prog = declared_program(label, guest, gtype, host, htype, rule)
                if prog == (ALL_BITS, ()):
                    cls = "PASS"
                else:
                    extra = progs.setdefault(prog, len(progs))
            elif cls == "FIELD":
                if entry is None or entry[0] != "FIELD" or target is not None:
                    raise SystemExit("%s: FIELD cannot be forced" % label)
                extra = entry[2]
            elif cls in HOOKED:
                extra = hooks.setdefault(rule.hook, len(hooks))
            if cls in ("MOVE", "PASS"):
                cls = "PASS" if hoff == offset else "MOVE"
            entry = [cls, hoff, extra]
            notes.pop((bank, offset), None)
            decided[(bank, offset)] = rule
        if entry is None:
            entry = ["PENDING", offset, 0]
        table[(bank, offset)] = entry
    return table, notes, decided


def emit(out, tables, progs, hooks, regs):
    w = out.write
    w("#ifndef AGC_REGCLASS_TABLES_H\n#define AGC_REGCLASS_TABLES_H\n\n")
    moves = []
    w("static const struct agc_field_prog agc_field_progs[] = {\n")
    for (keep, prog_moves), index in sorted(progs.items(), key=lambda item: item[1]):
        w("   {0x%08xu, %d, %d},\n" % (keep, len(moves), len(prog_moves)))
        moves.extend(prog_moves)
    if not progs:
        w("   {0, 0, 0},\n")
    w("};\n\nstatic const struct agc_field_move agc_field_moves[] = {\n")
    for src, dst, width in moves:
        w("   {%d, %d, %d},\n" % (src, dst, width))
    if not moves:
        w("   {0, 0, 0},\n")
    w("};\n\n")
    ordered = [name for name, _ in sorted(hooks.items(), key=lambda item: item[1])]
    w("enum agc_reg_hook {\n")
    for index, name in enumerate(ordered):
        w("   AGC_REG_HOOK_%s = %d,\n" % (name.upper(), index))
    w("   AGC_REG_HOOK_COUNT = %d\n};\n\n" % len(ordered))
    for name in ordered:
        w("void agc_reg_hook_%s(const struct agc_reg_hook_args *args, struct agc_reg_writes *out);\n" % name)
    w("\nstatic const agc_reg_hook_fn agc_reg_hooks[AGC_REG_HOOK_COUNT + 1] = {\n")
    for name in ordered:
        w("   agc_reg_hook_%s,\n" % name)
    w("   NULL,\n};\n\n")
    for host in HOSTS:
        table = tables[host]
        for bank, (bank_name, _, length) in enumerate(BANKS):
            w("static const struct agc_reg_entry agc_regs_%s_%s[%d] = {\n" % (host, bank_name, length))
            for offset in range(length):
                entry = table.get((bank, offset))
                if entry is None:
                    continue
                w("   [0x%03x] = {%d, AGC_REG_%s, %d},\n" % (offset, entry[1], entry[0], entry[2]))
            w("};\n\n")
            words = [0] * (length // 64)
            for offset in range(length):
                entry = table.get((bank, offset))
                if entry is not None and entry[0] == "PASS":
                    words[offset // 64] |= 1 << (offset % 64)
            w("static const uint64_t agc_regs_%s_%s_pass[%d] = {\n" % (host, bank_name, len(words)))
            for start in range(0, len(words), 4):
                w("   " + " ".join("0x%016xull," % word for word in words[start:start + 4]) + "\n")
            w("};\n\n")
    w("static const struct agc_reg_tables agc_reg_tables[AGC_HOST_COUNT] = {\n")
    for host in HOSTS:
        w("   [AGC_HOST_%s] = {{\n" % host.upper())
        for bank_name, _, length in BANKS:
            w("      {agc_regs_%s_%s, agc_regs_%s_%s_pass, %d},\n" % (host, bank_name, host, bank_name, length))
        w("   }},\n")
    w("};\n\n")
    w("static const uint16_t agc_reg_class_counts[AGC_HOST_COUNT][AGC_REG_BANK_COUNT][AGC_REG_CLASS_COUNT] = {\n")
    for host in HOSTS:
        w("   [AGC_HOST_%s] = {\n" % host.upper())
        for bank in range(len(BANKS)):
            counts = collections.Counter(entry[0] for (b, _), entry in tables[host].items() if b == bank)
            w("      {" + ", ".join(str(counts.get(cls, 0)) for cls in CLASSES) + "},\n")
        w("   },\n")
    w("};\n\n#endif\n")


def report(tables, notes, regs, used):
    for host in HOSTS:
        for label, keys in (("all", set(tables[host])), ("in default tables", used)):
            counts = collections.Counter()
            for key in keys:
                counts[(BANKS[key[0]][0], tables[host][key][0])] += 1
            total = collections.Counter()
            for (bank, cls), count in counts.items():
                total[cls] += count
            print("%s, %s: %s" % (host, label, ", ".join("%s %d" % (cls, total[cls]) for cls in CLASSES if total[cls])))
            for bank_name, _, _ in BANKS:
                listed = ", ".join("%s %d" % (cls, counts[(bank_name, cls)])
                                   for cls in CLASSES if counts[(bank_name, cls)])
                print("   %s: %s" % (bank_name, listed))
    for host in HOSTS:
        print("\n%s pending:" % host)
        for (bank, offset), entry in sorted(tables[host].items()):
            if entry[0] == "PENDING":
                mark = "*" if (bank, offset) in used else " "
                note = notes[host].get((bank, offset), "")
                print(" %s %s 0x%03x %-40s %s" % (mark, BANKS[bank][0], offset, regs[(bank, offset)][0], note))


def decisions(tables, decided, progs, regs, used, hosts):
    programs = {index: prog for prog, index in progs.items()}
    for host in HOSTS:
        print("\n%s decided by an override:" % host)
        for (bank, offset), rule in sorted(decided[host].items()):
            cls, hoff, extra = tables[host][(bank, offset)]
            mark = "*" if (bank, offset) in used else " "
            detail = ""
            if cls in HOOKED:
                detail = rule.hook
            elif cls == "FIELD":
                keep, moves = programs[extra]
                detail = "keep 0x%08x" % keep + "".join(" %d>%d:%d" % move for move in moves)
            if cls in ("MOVE", "FIELD", "ADDR") + HOOKED and hoff != offset:
                target = hosts[host].by_at.get(BANKS[bank][1] + hoff * 4, "0x%03x" % hoff)
                detail = (detail + " " if detail else "") + "to " + target
            print(" %s %s 0x%03x %-40s %-7s %s" % (mark, BANKS[bank][0], offset, regs[(bank, offset)][0], cls, detail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guest", required=True)
    ap.add_argument("--defaults")
    ap.add_argument("--overrides", required=True)
    ap.add_argument("--host", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    ap.add_argument("--out")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--decisions", action="store_true")
    args = ap.parse_args()

    guest = Chip(args.guest)
    paths = {name: path for name, path in args.host}
    if tuple(sorted(paths)) != tuple(sorted(HOSTS)):
        raise SystemExit("hosts must be exactly %s" % ", ".join(HOSTS))
    rules = Rules(args.overrides)
    regs, used = guest_universe(guest, args.defaults, rules.names)
    progs = {}
    hooks = {}
    tables = {}
    notes = {}
    decided = {}
    chips = {}
    for host in HOSTS:
        chips[host] = Chip(paths[host])
        tables[host], notes[host], decided[host] = classify(guest, regs, host, chips[host], rules, progs, hooks)
    unused = rules.unused()
    if unused:
        raise SystemExit("overrides that match nothing: %s" % ", ".join(unused))
    if args.report:
        report(tables, notes, regs, used)
    if args.decisions:
        decisions(tables, decided, progs, regs, used, chips)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as out:
            emit(out, tables, progs, hooks, regs)


if __name__ == "__main__":
    sys.exit(main())
