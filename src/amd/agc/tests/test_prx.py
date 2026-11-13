import argparse
import collections
import ctypes
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
AGC = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(AGC / "gen"), str(AGC / "tools")]

import agc_prx
import agc_prx_check

GENERATOR = AGC / "gen" / "agc_prx.py"
CHECKER = AGC / "tools" / "agc_prx_check.py"
VSENV = AGC.parents[2] / "build" / "agc-vs2022.cmd"
CLANG_CL = pathlib.Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\Llvm\x64\bin\clang-cl.exe")
ENTRY_STUB_FLAGS = 0x1100
MODULES = ("libSceAgc", "libSceAgcDriver")
RUNTIME = "agcprxtestrt"
ANCHOR = "agc_runtime_anchor"
RUST_DEPENDENCY = ("agcdrv.dll", "agc_runtime_anchor")
RUST_EMPTY_SHELL = ("libSceAgcVsh.prx", "7b4b134186663d59d4a0ca1c1106f8cafe45d93e854673075e22b1b15abe8b35")
RUST_PROBE_SHELL = ("agc-shell-probe.prx", "22f33340f8cfdba840d62ab8a65d4de81ed1a13d7e8e36a37d5f4cf69dbf340b")
RUST_PROBE = "mesa_agc_shell_probe"

NIDS_FROM_AGC_NIDS_H = (
    ("sceAgcAcbAtomicGds_0900", "gQkqkLttcpw"),
    ("sceAgcAcbCondExecGetSize", "ozKzBP4aki4"),
    ("sceAgcAcbDmaData", "-RnpfpxIhec"),
    ("sceAgcAcbJumpGetSize", "b-oySn+G2tE"),
    ("sceAgcAcbPushMarker", "cpCILPya5Zk"),
    ("sceAgcAcbWriteData", "eZ4+17OQz4Q"),
    ("sceAgcCbNop", "LtTouSCZjHM"),
    ("sceAgcCondExecPatchSetEnd", "ORWsxIbk4TE"),
    ("sceAgcDcbAcquireMemGetSize", "-vnlTPPXPrw"),
    ("sceAgcDcbDispatchIndirect", "CtB+A9-VxO0"),
    ("sceAgcDcbDispatchIndirectGetSize", "w8HVkEeXPv8"),
    ("sceAgcDcbDrawIndexOffset", "B+aG9DUnTKA"),
    ("sceAgcDcbMemSemaphore", "G0jrLdvEqDw"),
    ("sceAgcDcbPushMarker", "+kSrjIVxKFE"),
    ("sceAgcDcbSetBoolPredicationEnableGetSize", "yheJGN-ay+A"),
    ("sceAgcDcbSetCxRegistersIndirect", "ZvwO9euwYzc"),
    ("sceAgcDcbSetNumInstancesGetSize", "6DFuRKT4C9w"),
    ("sceAgcDcbSetShRegistersIndirect", "-HOOCn0JY48"),
    ("sceAgcDcbSetZPassPredicationEnableGetSize", "XN+Iuu7XsM8"),
    ("sceAgcDcbStallCommandBufferParserGetSize", "+u6dKSLWM2o"),
    ("sceAgcDriverAgrSubmitMultiDcbs", "+T8Xo6LtFJI"),
    ("sceAgcDriverAllocateToolMemoryForGpuReset", "T8xEHO2Y+XU"),
    ("sceAgcDriverGetGpuRefClks", "g68eYcZS7PY"),
    ("sceAgcDriverGetResourceShaderGuid", "mXn+K9E-wOA"),
    ("sceAgcDriverGetSetWorkloadCompletePacketSize", "WNyjOWq8-Vk"),
    ("sceAgcDriverGetTFRing", "e-YMQ+2tj9M"),
    ("sceAgcDriverIsSubmitValidationEnabled", "qspAL8bgcBY"),
    ("sceAgcDriverIsTraceInProgress", "+TN0oRTBxJQ"),
    ("sceAgcDriverRegisterResource", "W5z4eZrjEas"),
    ("sceAgcDriverSetWorkloadComplete", "i6bfTi13ApA"),
    ("sceAgcDriverSubmitToRazorACQ", "jJyVJyhi5h8"),
    ("sceAgcDriverUnregisterWorkloadStream", "n5ElQVYsU1A"),
    ("sceAgcDriverUserDataWritePacket", "-vc-xL+G8u0"),
    ("sceAgcDriverUserDataWritePushMarker", "+b34-CLWc0s"),
    ("sceAgcFuseShaderHalves_0200", "fd5Bp5tGTgo"),
    ("sceAgcGetDataPacketPayloadAddress_0090", "V++UgBtQhn0"),
    ("sceAgcGetFusedShaderSize_0080", "dolOmWH+huQ"),
    ("sceAgcInit_0090", "23LRUSvYu1M"),
    ("sceAgcJumpPatchSetTarget", "2BS4EtAaF28"),
    ("sceAgcSetUcRegIndirectPatchSetNumRegisters", "fRG-JOH5+sI"),
)
NIDS_FROM_ANYPS5_NID_PATCHER = (
    ("AgcDriverSubmitFlip", "qZZjNmeeL6Y"),
    ("AgcDriverWaitIdle", "3+CfJvLex4A"),
    ("ComputeContainSize", "7mdvqZc+Uzo"),
    ("agc_runtime_anchor", "lG9cDFd1w3w"),
    ("exit", "uMei1W9uyNo"),
    ("hostFunctionLongName", "dfcE5C-gCeo"),
    ("puts", "YQ0navp+YIc"),
    ("sceAgcBranchPatchSetElseTarget_0300", "rP5xLdOf26k"),
    ("sceAgcBranchPatchSetThenTarget_0300", "7Wa3aeJgeVU"),
    ("sceAgcCreateInterpolantMapping_10000", "dbOlWdppb4o"),
    ("sceAgcDcbAtomicGds_0900", "zARR5aCmkoY"),
    ("sceAgcDcbContextStateOp_0100", "qj7QZpgr9Uw"),
    ("sceAgcDcbSetIndexSize", "GIIW2J37e70"),
    ("sceAgcDcbSetIndexSize_0600", "-KRzWekV120"),
    ("sceAgcDriverSetHsOffchipParam", "MM4IZSEYytQ"),
    ("sceAgcInit_0090", "23LRUSvYu1M"),
    ("sceAgcJumpPatchSetTarget_0300", "Ikfdt-rIqCE"),
    ("sceAgcUpdateInterpolantMapping_10000", "vieBRwlh1Lw"),
    ("sceKernelFoo", "f6tExkOnmQ0"),
    ("sceNetRecv", "9wO9XrMsNhc"),
)
NIDS_FROM_ANYPS5_TOOL_TESTS = (
    ("ACProcessMain", "mFq1M6vw-JM"),
    ("sceAmprUnknown00", "6OkOa1SOKuY"),
)

Function = collections.namedtuple("Function", "name module nid params body arguments result")
FUNCTIONS = (
    Function("sceAgcInit_0090", "libSceAgc", "23LRUSvYu1M", ("a", "b"), "a + b", (40, 2), 42),
    Function("sceAgcDcbPushMarker", "libSceAgc", "+kSrjIVxKFE", ("a",), "a * 3", (5,), 15),
    Function("sceAgcAcbDmaData", "libSceAgc", "-RnpfpxIhec", (), "1486", (), 1486),
    Function("sceAgcDcbDispatchIndirect", "libSceAgc", "CtB+A9-VxO0", ("a", "b", "c"), "a - b * c", (60, 2, 4), 52),
    Function("sceAgcDriverUserDataWritePacket", "libSceAgcDriver", "-vc-xL+G8u0", ("a", "b"), "a ^ b", (85, 15), 90),
    Function("sceAgcDriverUserDataWritePushMarker", "libSceAgcDriver", "+b34-CLWc0s", ("a",), "-a", (-7,), 7),
    Function("sceAgcDriverGetTFRing", "libSceAgcDriver", "e-YMQ+2tj9M", (), "2026", (), 2026),
)


def check(condition, message, *values):
    if not condition:
        raise SystemExit("FAIL: " + message % values)


def run(command):
    return subprocess.run(command, capture_output=True, text=True, errors="replace")


def python(script, *arguments):
    return run([sys.executable, "-B", str(script), *map(str, arguments)])


def refused(function, *arguments):
    try:
        function(*arguments)
    except SystemExit:
        return True
    return False


def malformed(data):
    try:
        agc_prx_check.parse(data)
    except agc_prx_check.Malformed as error:
        return str(error)
    return ""


def patched(image, at, layout, *values):
    data = bytearray(image)
    struct.pack_into(layout, data, at, *values)
    return bytes(data)


def forwards(module):
    return sorted((function.nid, "%s.%s" % (RUNTIME, function.name))
                  for function in FUNCTIONS if function.module == module)


def manifest(functions):
    return json.dumps({"functions": [
        {"name": function.name, "module": function.module, "return": "int32_t",
         "params": [["int32_t", param] for param in function.params], "state": "done"}
        for function in functions]})


def test_nids():
    tables = (NIDS_FROM_AGC_NIDS_H, NIDS_FROM_ANYPS5_NID_PATCHER, NIDS_FROM_ANYPS5_TOOL_TESTS,
              [(function.name, function.nid) for function in FUNCTIONS])
    for table in tables:
        for name, nid in table:
            check(agc_prx.nid(name) == nid, "nid(%s) is %s, expected %s", name, agc_prx.nid(name), nid)
    print("ok: %d known NIDs" % len({pair for table in tables for pair in table}))


def test_builder():
    for (name, digest), exports in ((RUST_EMPTY_SHELL, []), (RUST_PROBE_SHELL, [RUST_PROBE])):
        image = agc_prx.build_shell(
            name, [(agc_prx.nid(export), "agcdrv." + export) for export in exports], RUST_DEPENDENCY)
        check(hashlib.sha256(image).hexdigest() == digest, "%s differs from the shell the Rust writer makes", name)
    image = agc_prx.build_shell("two.prx", [("b", "m.second"), ("a", "m.first")], ("m.dll", ANCHOR))
    shell = agc_prx_check.parse(image)
    check(shell == ("two.prx", [("a", "m.first"), ("b", "m.second")], [("m.dll", [ANCHOR])]), "read back %r", shell)
    for bad in ([("a", "m.x"), ("a", "m.y")], [("a", "nodot")], [("a", ".x")], [("a", "m.")], [("", "m.x")],
                [("a\u00e9", "m.x")], [("a", "m.x\0")]):
        check(refused(agc_prx.build_shell, "bad.prx", bad, RUST_DEPENDENCY), "build_shell accepted %r", bad)
    print("ok: shell builder, byte-identical with two shells of the Rust writer")


def test_generator(work):
    path = work / "manifest.json"
    path.write_text(manifest(FUNCTIONS), encoding="utf-8")
    listing = python(GENERATOR, "--manifest", path, "--list")
    expected = "".join("%s\t%s\t%s\n" % (function.nid, function.name, function.module) for function in sorted(FUNCTIONS))
    check(listing.returncode == 0 and listing.stdout == expected, "--list: %s%s", listing.stdout, listing.stderr)

    definitions = work / (RUNTIME + ".def")
    result = python(GENERATOR, "--manifest", path, "--def", definitions)
    names = sorted([ANCHOR] + [function.name for function in FUNCTIONS])
    expected = "EXPORTS\n" + "".join("    %s\n" % name for name in names)
    check(result.returncode == 0 and definitions.read_bytes() == expected.encode("ascii"), "--def: %s", result.stderr)

    shells = {}
    for module in MODULES:
        out = work / (module + ".prx")
        result = python(GENERATOR, "--manifest", path, "--module", module, "--runtime", RUNTIME, "--out", out)
        check(result.returncode == 0, "--out %s: %s", out.name, result.stderr)
        shells[module] = out.read_bytes()
        again = agc_prx.module_shell(agc_prx.load(path), module, RUNTIME, ANCHOR, out.name)
        check(shells[module] == again, "%s is not reproducible", out.name)
        shell = agc_prx_check.parse(shells[module])
        check(shell == (out.name, forwards(module), [(RUNTIME + ".dll", [ANCHOR])]), "%s reads back as %r", out.name, shell)
        checked = python(CHECKER, out)
        expected = "".join("%s\t%s\n" % forward for forward in forwards(module))
        check(checked.returncode == 0 and checked.stdout == expected, "checker on %s: %s%s", out.name, checked.stdout,
              checked.stderr)

    anchor = FUNCTIONS[0].name
    out = work / "anchored.prx"
    definitions = work / "anchored.def"
    result = python(GENERATOR, "--manifest", path, "--module", MODULES[0], "--runtime", RUNTIME, "--anchor", anchor,
                    "--out", out, "--def", definitions)
    check(result.returncode == 0, "--anchor %s: %s", anchor, result.stderr)
    check(agc_prx_check.parse(out.read_bytes()).imports == [(RUNTIME + ".dll", [anchor])], "--anchor is not imported")
    expected = "EXPORTS\n" + "".join("    %s\n" % name for name in names if name != ANCHOR)
    check(definitions.read_bytes() == expected.encode("ascii"), "--anchor %s changed the exports of --def", anchor)

    out = work / "libSceAgcVsh.prx"
    result = python(GENERATOR, "--manifest", path, "--module", "libSceAgcVsh", "--runtime", RUNTIME, "--out", out)
    check(result.returncode != 0 and not out.exists(), "a module without functions got a shell")
    repeated = work / "repeated.json"
    repeated.write_text(manifest(FUNCTIONS + FUNCTIONS[:1]), encoding="utf-8")
    check(python(GENERATOR, "--manifest", repeated, "--list").returncode != 0, "a repeated export name was accepted")
    print("ok: generator command lines for %s, --def and --list" % ", ".join(MODULES))
    return shells


def test_checker(work, shells):
    image = shells[MODULES[0]]
    coff = agc_prx.PE_OFFSET + 4
    optional = coff + agc_prx.size_of(agc_prx.COFF_HEADER)
    directory = optional + agc_prx.size_of(agc_prx.OPTIONAL_HEADER)
    section = directory + 8 * agc_prx.DIRECTORIES

    def offset(rva):
        return rva - agc_prx.SECTION_RVA + agc_prx.HEADERS_SIZE

    exports, = struct.unpack_from("<I", image, directory)
    function_table, name_table = struct.unpack_from("<II", image, offset(exports) + 28)
    first, second = struct.unpack_from("<II", image, offset(name_table))
    target, = struct.unpack_from("<I", image, offset(function_table))
    dot = image.index(b".", offset(target))
    broken = (
        ("no MZ signature", b"not a shell"),
        ("truncated", image[:0x100]),
        ("lies outside the file", image[:-1]),
        ("is not AMD64", patched(image, coff, "<H", 0x14c)),
        ("not a DLL", patched(image, coff + 18, "<H", 0x0022)),
        ("relocations are stripped", patched(image, coff + 18, "<H", 0x2023)),
        ("entry point", patched(image, optional + 16, "<I", agc_prx.SECTION_RVA)),
        ("holds code", patched(image, section + 36, "<I", 0x60000020)),
        ("out of order", patched(image, offset(name_table), "<II", second, first)),
        ("is not a forwarder", patched(image, directory + 4, "<I", 40)),
        ("is not module.symbol", patched(image, dot, "<B", ord("_"))),
        ("without importing", agc_prx.build_shell("bare.prx", forwards(MODULES[0]), None)),
    )
    check(malformed(image) == "", "the checker refuses a good shell: %s", malformed(image))
    for reason, data in broken:
        check(reason in malformed(data), "expected \"%s\", the checker said \"%s\"", reason, malformed(data))
    bare = work / "bare.prx"
    bare.write_bytes(broken[-1][1])
    result = python(CHECKER, bare)
    check(result.returncode == 1 and result.stdout == "" and "without importing" in result.stderr,
          "checker on a shell without the anchor: %d %s", result.returncode, result.stderr)
    check(python(CHECKER, work / "absent.prx").returncode == 1, "the checker accepted a missing file")
    print("ok: checker refuses %d kinds of broken shell" % len(broken))

    if importlib.util.find_spec("pefile") is None:
        print("skipped: pefile is not installed, no cross-check of the reader")
        return
    for module in MODULES:
        result = python(CHECKER, work / (module + ".prx"), "--pefile")
        check(result.returncode == 0, "pefile disagrees on %s: %s", module, result.stderr)
    print("ok: pefile reads the same exports and imports")


def build_runtime(work, cc, vsenv):
    source = work / (RUNTIME + ".c")
    lines = ["void %s(void) {}\n" % ANCHOR]
    for function in FUNCTIONS:
        params = ", ".join("int " + param for param in function.params) or "void"
        lines.append("int %s(%s) { return %s; }\n" % (function.name, params, function.body))
    source.write_text("".join(lines), encoding="ascii")
    dll = work / (RUNTIME + ".dll")
    command = [cc, "/nologo", "/O1", "/GS-", "-fuse-ld=lld", str(source), "/Fo" + str(work / (RUNTIME + ".obj")),
               "/Fe" + str(dll), "/LD", "/link", "/NOENTRY", "/NODEFAULTLIB", "/DEF:" + str(work / (RUNTIME + ".def"))]
    if vsenv:
        command = 'cmd /s /c ""%s" %s"' % (vsenv, subprocess.list2cmdline(command))
    try:
        result = run(command)
    except OSError as error:
        raise SystemExit("FAIL: cannot run %s: %s" % (cc, error))
    check(result.returncode == 0 and dll.is_file(), "the runtime DLL did not build: %s\n%s%s", command, result.stdout,
          result.stderr)
    return dll


def test_load(work, cc, vsenv):
    if sys.platform != "win32":
        print("skipped: loading a shell needs the Windows loader")
        return
    if cc is None:
        cc = str(CLANG_CL) if CLANG_CL.is_file() else shutil.which("clang-cl")
        check(cc, "no clang-cl to build the runtime DLL with, pass --cc")
        if vsenv is None and VSENV.is_file():
            vsenv = str(VSENV)
    dll = build_runtime(work, str(pathlib.Path(cc)), vsenv)
    check("holds code" in malformed(dll.read_bytes()), "the checker takes the runtime DLL for a shell")
    result = python(pathlib.Path(__file__).resolve(), "--load", work)
    sys.stdout.write(result.stdout)
    check(result.returncode == 0, "loading the shells failed: %s", result.stderr)


def load(directory):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LoadLibraryExW.restype = ctypes.c_void_p
    kernel.LoadLibraryExW.argtypes = (ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32)
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    kernel.GetModuleHandleW.argtypes = (ctypes.c_wchar_p,)
    kernel.GetModuleFileNameW.restype = ctypes.c_uint32
    kernel.GetModuleFileNameW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32)
    kernel.GetProcAddress.restype = ctypes.c_void_p
    kernel.GetProcAddress.argtypes = (ctypes.c_void_p, ctypes.c_char_p)

    runtime_dll = RUNTIME + ".dll"
    check(not kernel.GetModuleHandleW(runtime_dll), "%s is loaded before any shell", runtime_dll)
    handles = {}
    for module in MODULES:
        path = os.path.join(directory, module + ".prx")
        handles[module] = kernel.LoadLibraryExW(path, None, ENTRY_STUB_FLAGS)
        check(handles[module], "LoadLibraryExW(%s, 0x%x) failed with error %d", path, ENTRY_STUB_FLAGS,
              ctypes.get_last_error())
        runtime = kernel.GetModuleHandleW(runtime_dll)
        check(runtime, "loading %s.prx did not load %s", module, runtime_dll)
    located = ctypes.create_unicode_buffer(0x8000)
    kernel.GetModuleFileNameW(runtime, located, len(located))
    check(os.path.samefile(located.value, os.path.join(directory, runtime_dll)), "the runtime came from %s", located.value)

    for function in FUNCTIONS:
        name = function.nid.encode("ascii")
        address = kernel.GetProcAddress(handles[function.module], name)
        direct = kernel.GetProcAddress(runtime, function.name.encode("ascii"))
        check(address and address == direct, "%s (%s) resolves to %s, the runtime has it at %s", function.nid,
              function.name, address, direct)
        call = ctypes.CFUNCTYPE(ctypes.c_int, *[ctypes.c_int] * len(function.arguments))(address)
        result = call(*function.arguments)
        check(result == function.result, "%s%r returned %d, expected %d", function.name, function.arguments, result,
              function.result)
        for module in MODULES:
            other = kernel.GetProcAddress(handles[module], name)
            check(module == function.module or not other, "%s.prx also exports %s", module, function.nid)
    print("ok: Windows loaded %s with %s and called %d forwarded exports" % (
        ", ".join(module + ".prx" for module in MODULES), runtime_dll, len(FUNCTIONS)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cc")
    ap.add_argument("--vsenv")
    ap.add_argument("--load")
    args = ap.parse_args()
    if args.load:
        return load(args.load)
    test_nids()
    test_builder()
    with tempfile.TemporaryDirectory(prefix="agc-prx-") as work:
        work = pathlib.Path(work)
        shells = test_generator(work)
        test_checker(work, shells)
        test_load(work, args.cc, args.vsenv)
    print("agc_prx: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
