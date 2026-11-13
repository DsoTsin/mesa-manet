import argparse
import collections
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gen"))

import agc_exports

MASKS = {"uint8_t": 0xff, "int8_t": 0xff, "uint16_t": 0xffff, "int16_t": 0xffff, "uint32_t": 0xffffffff,
         "int32_t": 0xffffffff, "bool": 0xff}
SIGNED = {"int8_t": 8, "int16_t": 16, "int32_t": 32, "int64_t": 64}


def read_trace(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"AGCTRACE":
        raise SystemExit("%s is not an AGC API trace" % path)
    version, count = struct.unpack_from("<II", data, 8)
    if version != 1:
        raise SystemExit("unknown trace version %d" % version)
    at = 16
    functions = []
    for _ in range(count):
        length = data[at]
        name = data[at + 1:at + 1 + length].decode()
        params, command_buffer = data[at + 1 + length], data[at + 2 + length]
        functions.append((name, params, bool(command_buffer)))
        at += 3 + length
    records = []
    while at + 16 <= len(data):
        time, packed = struct.unpack_from("<QQ", data, at)
        at += 16
        thread = packed & 0xffffffff
        index = (packed >> 32) & 0xffff
        argc = (packed >> 48) & 0xff
        flags = packed >> 56
        words = argc + (4 if flags & 1 else 0)
        if at + 8 * words > len(data):
            break
        values = struct.unpack_from("<%dQ" % words, data, at)
        at += 8 * words
        records.append((time, thread, index, values[:argc], values[argc:] if flags & 1 else None))
    return functions, records


def load_signatures():
    manifests = [os.path.join(HERE, "..", "gen", name) for name in ("agc_exports.json", "agc_exports_core.json")]
    return {function["name"]: function for function in agc_exports.load_all(manifests)}


def format_value(ctype, value):
    ctype = ctype.replace("const ", "").strip()
    if ctype.endswith("*"):
        return "0x%x" % value
    mask = MASKS.get(ctype)
    if mask is not None:
        value &= mask
    bits = SIGNED.get(ctype)
    if bits and value >> (bits - 1):
        value -= 1 << bits
    if ctype == "bool":
        return "true" if value else "false"
    return str(value) if -4096 < value < 4096 else "0x%x" % value


def format_call(name, signature, args, cb):
    params = signature["params"] if signature else [("uint64_t", "arg%d" % i) for i in range(len(args))]
    parts = []
    for i, value in enumerate(args):
        ctype, pname = params[i] if i < len(params) else ("uint64_t", "arg%d" % i)
        if i == 0 and cb:
            bottom, top, up, down = cb
            parts.append("%s=0x%x[bottom 0x%x up +%d down 0x%x top 0x%x]" %
                         (pname, value, bottom, (up - bottom) // 4 if up >= bottom else -1, down, top))
        else:
            parts.append("%s=%s" % (pname, format_value(ctype, value)))
    return "%s(%s)" % (name, ", ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--text", action="store_true")
    ap.add_argument("--function", default="")
    ap.add_argument("--thread", type=int)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    functions, records = read_trace(args.trace)
    signatures = load_signatures()
    pattern = re.compile(args.function) if args.function else None
    if not args.text:
        calls = collections.Counter(functions[index][0] for _, _, index, _, _ in records)
        threads = collections.Counter(thread for _, thread, _, _, _ in records)
        span = (records[-1][0] - records[0][0]) / 1e9 if records else 0.0
        print("%d calls of %d functions over %.3f s on %d threads" % (len(records), len(calls), span, len(threads)))
        for name, n in calls.most_common():
            print("%10d  %s" % (n, name))
        print("threads: " + ", ".join("t%d %d" % (t, n) for t, n in sorted(threads.items())))
        return
    shown = 0
    for time, thread, index, values, cb in records:
        seconds = time / 1e9
        if seconds < args.start or (args.end is not None and seconds > args.end):
            continue
        name = functions[index][0]
        if pattern and not pattern.search(name):
            continue
        if args.thread is not None and thread != args.thread:
            continue
        print("%12.6f t%-3d %s" % (seconds, thread, format_call(name, signatures.get(name), values, cb)))
        shown += 1
        if args.limit and shown >= args.limit:
            break


if __name__ == "__main__":
    sys.exit(main())
