# AOC 7.0.15 SPIR-V / DXBC / DXIL ISA extraction

Analysis artifact: biblioklept. Local Windows analysis; no vendor binaries are distributed.

## Identity and implementation boundary

- Binary: `C:/Program Files/Qualcomm/Adreno Offline Compiler/aoc.exe`
- SHA-256: `100eebaddfb0f125ee7a7e1e621d56f32ecba7a15f80c2e45461ca5a941f5648`
- AOC 7.0.15; compiler E17.52.07.00; PE AMD64; preferred image base `0x140000000`.
- IDA 9.3 SP2 evidence: `D:/IrisBuild/.build/aoc-isa/targeted.i64`, `0x*-asm.txt`, `0x*-c.txt`, `object.shader.bin`, and runtime traces in that directory.
- Scope: run the installed AOC compiler normally, expose its existing HW dump option as input state, copy the compiled instruction span before AOC releases it, and print with Mesa `ir3_isa_disasm`. This is a compiler adapter, not a replacement compiler or an independent reconstruction of Qualcomm's decoder.
- Use Windows hardware execution breakpoints. Do not change instructions, the installed file, its signature, or the vendor allocator. A full file hash gates all private RVAs.

## Confirmed functions

Names below are descriptive unless they are `main`; private symbols are stripped. Runtime address is module base plus RVA.

| Function | VA | RVA | Observed ABI / role |
|---|---:|---:|---|
| main | `0x1401577b0` | `0x1577b0` | Constructs the 0x950-byte AOC application, runs it, destroys it |
| Application constructor | `0x14015b3d0` | `0x15b3d0` | Called by main before argument parsing |
| Application destructor | `0x14015ceb0` | `0x15ceb0` | Called by main after run returns |
| AOC run | `0x14016fe60` | `0x16fe60` | `(AocApplication*, argc, argv)`; parses arguments before API dispatch |
| Vulkan / GLES branch | `0x140171800` | `0x171800` | `(AocApplication*)`; API 1 is Vulkan; builds shaders and links them |
| Direct3D branch | `0x140170b80` | `0x170b80` | `(AocApplication*)`; DXBC / DXIL compiler frontend, accepts root signature JSON |
| Compile shaders | `0x140322040` | `0x322040` | `(context, uint32 count, ShaderInput** inputs, LinkOptions*, CompileResult*)` |
| Link program | `0x140322a60` | `0x322a60` | `(context, flags, stage-results, output-handle)` |
| AOC shader dump | `0x1401666b0` | `0x1666b0` | `(application, shader-object-data, object-size, stage, stats, binning, verbose)`; gets object section 10 |
| Shared HW dump | `0x1402babf0` | `0x2babf0` | `(uint32 stage, MetadataContext*, InstructionSpan*, callback-context, callback, extra, uint32 flags, bool plain, bool json)` |
| Dump formatter | `0x1402bafa0` | `0x2bafa0` | Converts instruction count to bytes with `count * 8` |
| Vendor ISA text wrapper | `0x1402b5b40` | `0x2b5b40` | `(ostream, code, byte-count, extra, metadata, architecture)` |
| Vendor instruction decoder | `0x14309b620` | `0x309b620` | `(ostream*, architecture*, uint64* code, uint64 instruction-count, DisasmOptions*, source-map*)` |
| Metadata object accessor | `0x1406392c0` | `0x6392c0` | `context->reader->object->data` |
| Metadata section lookup | `0x1406398d0` | `0x6398d0` | `(MetadataContext*, section-id, data-out, descriptor-out)` |
| Section lookup implementation | `0x140639910` | `0x639910` | Reads 20-byte section descriptors, bounds checks against the object span |
| Release stage results | `0x140322ba0` | `0x322ba0` | Releases result members through AOC callbacks, including disassembled shader collection at +0x38 |
| Release context | `0x140322000` | `0x322000` | Closes compiler context and invokes its matching release callback |

Windows x64 shared HW dump entry: RCX = stage, RDX = metadata context, R8 = instruction span, R9 = callback context. Flags are the seventh argument at `[RSP+0x38]`. ISA text is requested by flags bit `0x8`; statistics use `0x20000`. Capture only the ISA call, avoiding duplicate statistics calls.

## Confirmed layouts

Recovered layout names are descriptive; unobserved fields remain opaque in the adapter header.

| Structure | Size / offset | Meaning and evidence |
|---|---|---|
| AocApplication | size 0x950 | Stack allocation in main |
| AocApplication | +0x300, uint32 | API: 0 GLES, 1 Vulkan, 2 Direct3D, 3 OpenCL; run dispatch and runtime SPIR-V |
| AocApplication | +0x594, byte | Dump instruction statistics |
| AocApplication | +0x595, byte | Dump source |
| AocApplication | +0x596, byte | HW ISA dump; confirmed by setting this input byte before the API branch and observing flags 8 / native ISA |
| AocApplication | +0x597, byte | Extra dump detail; its complete semantics remain unresolved |
| AocApplication | +0x5c8, byte | JSON output mode |
| AocApplication | +0x948, byte | Performance projection statistics |
| InstructionSpan | size 16; +0 data, +8 uint64 count | Count is **instructions**, not bytes; caller takes section descriptor +12, formatter multiplies by 8 |
| ObjectSpan | size 16; +0 data, +8 uint64 bytes | Borrowed shader object view |
| MetadataContext | +0x688 pointer | Reader; validated by accessor and runtime |
| MetadataReader | +0x10 pointer | ObjectSpan; other fields remain opaque |
| Shader object header | +0x04 uint32 | Signature `0xe74f4751` in tested binaries |
| Shader object header | +0x14 uint32 | Section table byte offset |
| Shader object header | +0x18 uint32 | Section count |
| SectionDescriptor | size 20 | uint32 type, byte offset, byte size, element count, element size at offsets 0,4,8,12,16 |
| Section 10 | element size 8 | Actual Adreno machine instructions; byte size = instruction count * 8 |
| Section 1 | +0x20 uint32 | Shader flags; bit 0x40 marks binning, used by shared dump formatter |
| DISASSEMBLED_SHADER | size 24 | Data pointer +0, byte size +8 (exact field width requires further registration analysis), encoded stage/type +16 |
| DISASSEMBLED_SHADERS_COLLECTIONS | size 24 | metadataHandle +0, shader count +8, shaders pointer +16; cleanup uses shaders stride 24 |
| CompileResult | size 64 | Initializer clears 64 bytes; disassembledShaders pointer +56; this result is not read by the adapter |
| ShaderInput | size 144 | Common fields; source-specific descriptor pointer +136. SPIR-V descriptor stores stage +0, data +8, bytes +16, specialization count +24, values +32 |

Capture validates the object signature and section bounds, the equality between the dump span and section 10, and element size 8. It copies bytes while the debug event holds the child paused. All input/output pointers remain AOC-owned and are released by its original control flow.

## Runtime evidence before implementation

- SPIR-V compute, A830: 10 machine words / 80 bytes; native dump includes `ldib`, `add.f`, `stib`, and `end`.
- DXIL compute, A741, root UAV descriptor: 15 words / 120 bytes; native dump includes `ldg.a`, `mad.f32`, `stg.a`, `end`.
- DXBC vertex, A741: rendering and binning objects both reach the ISA path. Input POSITION can be reused as position output, so the minimal fixture compiles to `end`; validation will use a nontrivial independent shader as well.
- DXBC compute with an unbound UAV is rejected by AOC (`non-bindless UAV found when bindless UAVs enabled`); root/resource requirements belong to the compiler and are preserved.
- AOC's text-mode process return can be zero after compilation failure. The adapter must require a successful compilation message plus at least one validated ISA capture.

## Native Windows validation

Built with clang-cl and Visual Studio 2022 x64 libraries. The executable is `D:/projects/mesa-manet/build/aoc-isa-windows/src/freedreno/ir3/ir3-aoc-isa.exe`. It uses the installed compiler; it does not need Frida, IDA, DXC, or glslang at runtime. DXC and glslang are fixture generators for the integration test.

The integration test generated fresh SPIR-V with glslang, DXBC through Windows `D3DCompile` (`vs_5_0`, `ps_5_0`, `cs_5_0`), and DXIL with DXC (`cs_6_0`). Fourteen compilation cases produced 18 shader variants, 170 instruction words / 1360 bytes. Every extracted word matched both AOC's original formatter output and the word prefixes in the Mesa output. No instructions in this corpus were unmatched.

| Cases | Tested targets | Captured variants / words |
|---|---|---|
| SPIR-V buffer compute | A650, A740, A830 | 1 / 11, 1 / 9, 1 / 10 |
| SPIR-V dynamic loop and branch | A650, A830 | 1 / 19, 1 / 19 |
| DXBC arithmetic vertex | A690, A741 | 2 / 8 on each target, rendering and binning |
| DXIL UAV compute with root signature | A690, A741 | 1 / 10, 1 / 15 |
| DXBC arithmetic pixel | A741 | 1 / 10 |
| DXBC vertex + pixel pipeline | A741 | 3 / 18, rendering VS, binning VS, FS |
| SPIR-V with `-dump=all` | A830 | 1 / 10; statistics did not create duplicate captures |
| DXBC input path containing spaces | A741 | 2 / 8 |
| DXIL JSON shader path containing spaces | A741 | 1 / 15 |

Repeated SPIR-V compilation produced identical ISA text. Missing input, invalid SPIR-V, a different executable hash, and DXBC compute without its required UAV binding all returned status 1 without creating an ISA output. The compiler's native text-mode zero exit after a failed compilation was correctly rejected. Temporary input copies were removed after the run.

AOC's Direct3D path can crash with a directory-qualified `-link_info` filename or a whitespace-containing direct shader argument. These were reproduced outside the adapter. The adapter copies the JSON to a unique leaf filename in the caller's working directory, preserving its contents and the caller's relative shader path semantics. A DXBC/DXIL command-line input containing spaces is likewise copied under a unique name retaining its stage and format extensions. The copies are removed after success or failure. Paths embedded in JSON remain unchanged. Run simultaneous AOC jobs in different working directories because AOC itself also writes temporary files there.

The reused Mesa decoder needed its existing `strndup` and `vasprintf` compatibility headers and the Windows `NUL` device for branch-label prepasses. Freedreno's XML include checker now normalizes Windows path separators before passing files to its allowlist. These are the only changes outside `src/freedreno/ir3`.

Mesa's ISA tables determine mnemonic and field coverage. For example, some vendor `ldib` encodings print `dontcare bits` annotations, and special-register spelling differs. Raw words remain intact. Unknown instructions are printed as `.quad` and make the tool return status 2; compilation, capture, hash, and output failures return 1. Successful captures with no unmatched instruction return 0. This corpus does not establish complete ISA coverage or semantic identity between Qualcomm's and Mesa's printed syntax.

Evidence: `D:/IrisBuild/.build/aoc-isa/validation.log`, `validation/results.json`, and each case's `mesa.isa`, `aoc.log`, and `raw/*.bin`.

## Build and use

From an x64 VS developer shell with clang-cl, Python, Meson, and Ninja available:

```bat
set CC=clang-cl
set CXX=clang-cl
meson setup build/aoc-isa-windows --buildtype=debugoptimized -Dtools=freedreno -Dgallium-drivers=[] -Dvulkan-drivers=[] -Dplatforms=[] -Dllvm=disabled -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dgles1=disabled -Dgles2=disabled -Dopengl=false -Dshared-glapi=disabled -Dzstd=disabled
ninja -C build/aoc-isa-windows src/freedreno/ir3/ir3-aoc-isa.exe
```

```bat
build\aoc-isa-windows\src\freedreno\ir3\ir3-aoc-isa.exe --output shader.isa --raw-dir shader-raw --aoc-log compiler.log -- -arch=a830 shader.comp.spv
build\aoc-isa-windows\src\freedreno\ir3\ir3-aoc-isa.exe --output shader.isa -- -arch=a741 shader.vs.dxbc
build\aoc-isa-windows\src\freedreno\ir3\ir3-aoc-isa.exe --output shader.isa -- -arch=a741 -link_info pipeline.json
python src\freedreno\ir3\tests\aoc-isa-windows.py --tool build\aoc-isa-windows\src\freedreno\ir3\ir3-aoc-isa.exe --work-dir D:\IrisBuild\.build\aoc-isa\validation
```

Pass AOC options after `--`. Default architectures are A650 for Vulkan and A690 for Direct3D; both use generation 600 in Mesa. `--output` omitted prints to stdout. `--raw-dir` and `--aoc-log` are optional. Stage-qualified extensions or explicit AOC API/stage switches are required, as in AOC's own CLI. Direct3D root signatures are passed through the AOC JSON format shown by `aoc.exe -h`.

## LoadLibrary experiment

`LoadLibraryW(aoc.exe)` successfully mapped the installed EXE, and `GetProcAddress` found `LoadQCLC`. The PE has 22 exports, predominantly `cl_compiler_*`; the SPIR-V and Direct3D entry points listed above have no named exports. The EXE has a TLS directory and its process startup entry is RVA `0x31ac74c`.

The mapped `GetLastError`, `GetStdHandle`, and `malloc` IAT entries still contained import-name RVAs (`0x4469452`, `0x4469486`, `0x446b600`), rather than callable function addresses. This agrees with [Microsoft's LoadLibraryW documentation](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryw): EXE loads omit static import resolution.

In a separate probe process, all 409 static imports were resolved against their native DLLs. Calling the recovered `main(2, {"aoc.exe", "-h"})` still raised an access violation reading address 8. This establishes that import binding alone is insufficient. It does not identify the exact missing initialization or establish that an in-process loader is impossible. CRT initialization tables, TLS registration/callbacks, compiler global state, loader exception handling, thread lifecycle, and teardown would need their own recovered and tested bootstrap before replacing the working process adapter. No successful LoadLibrary-based shader compilation is claimed.

Probe evidence: `D:/IrisBuild/.build/aoc-isa/loadlibrary_import_probe.py` and `loadlibrary_import_probe.log`. The probe modified only IAT data in its own mapped module; the installed EXE was unchanged.
