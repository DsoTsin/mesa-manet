# malioc ISA 导出工具

`malioc_disasm.c` 实现完整流程：调用本机 Mali Offline Compiler DLL 编译 Vulkan SPIR-V，解析返回的 MBS2，提取每个 EBIN 的 OBJC，再直接调用本仓库的 `disassemble_valhall` 输出 ISA。一次运行 `malioc_disasm.exe` 即可完成，不需要 Python 包装程序或外部反汇编进程。

## 支持范围

- 编译：Windows x64、Arm Performance Studio 2026.5 的 `Mali-Gxx_r56p1-00rel0.dll`。
- 输入：Vulkan SPIR-V 二进制；支持 fragment、compute、vertex，通过所选 `OpEntryPoint` 自动识别阶段，不依赖扩展名。
- ISA：Valhall；默认目标为 `Mali-G610 r0p0`。`--core` 和 `--revision` 选择 DLL 核心表中的目标，程序检查报告中的架构。
- Vertex 默认请求 IDVS，保留 Position、Varying 和编译器实际生成的 pilot；`--no-idvs` 请求合并的 vertex shader。
- `--mbs2` 和 `--raw` 处理已有容器或机器码，不加载 Arm DLL。

默认 DLL 路径：

```text
C:/Program Files/Arm/Arm Performance Studio 2026.5/mali_offline_compiler/graphics/Mali-Gxx_r56p1-00rel0.dll
```

DLL ABI 依据 IDA 分析确认。程序加载前检查以下 SHA256，其他版本需要先分析和适配：

```text
23f5873642b6794f2754d926b652bf43b2f4e22aa71f3c4980580e54cbfbb0fd
```

当前不接入 GLSL 源码、OpenCL kernel、Bifrost/Midgard ISA 和下表中的其他阶段。DLL 接受某个阶段名称，并不代表当前 GPU 或导出工具支持该阶段。工具通过动态加载调用已安装的 DLL，构建时不需要 Arm SDK、导入库或复制 DLL。

## 构建

以下相对路径均以 `D:/projects/mesa-manet` 为工作目录。

### Windows 单独构建

无需构建完整 Mesa。需要 VS2022 x64 开发环境、PATH 中的 `clang-cl`、Python 3 和 `mako`。Python 仅用于运行现有 Valhall 代码生成器和测试，编译后的工具运行不需要 Python。Windows SDK 的 `bcrypt.lib` 用于 SHA256 检查。

在 CMD 中检查工具；缺少 `mako` 时安装到同一个 Python 环境：

```bat
where python
where clang-cl
python -c "import mako; print('mako: OK')"
python -m pip install mako
```

在 **CMD** 中执行以下构建命令；`call` 和 `^` 是 CMD 语法。VS2022 路径按本机 Enterprise 安装位置填写，其他版本需替换相应目录。

```bat
cd /d D:\projects\mesa-manet
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if not exist build\malioc-isa mkdir build\malioc-isa

python src/panfrost/compiler/bifrost/valhall/disasm.py --xml src/panfrost/compiler/bifrost/valhall/ISA.xml > build/malioc-isa/valhall_disasm.c

clang-cl /nologo /W4 /O2 /MD /D_CRT_SECURE_NO_WARNINGS ^
  src/panfrost/tools/malioc_disasm.c build/malioc-isa/valhall_disasm.c ^
  /I src/panfrost/compiler/bifrost ^
  /I src/panfrost/compiler/bifrost/valhall ^
  /Fobuild/malioc-isa/ /Fe:build/malioc-isa/malioc_disasm.exe ^
  /link bcrypt.lib
```

两项 `/I` 分别用于定位 `valhall/disassemble.h` 和生成实现引用的 `disassemble.h`。`/MD` 使用动态 CRT，DLL 的分配与释放回调使用 `ucrtbase.dll` 导出的 `malloc/free`；DLL 输出由对应导出的释放函数管理。

| 产物 | 内容 |
|---|---|
| `build/malioc-isa/valhall_disasm.c` | 当前 `ISA.xml` 生成的 Valhall 反汇编实现 |
| `build/malioc-isa/malioc_disasm.obj` | 命令行、DLL 调用、MBS2 解析和输出实现 |
| `build/malioc-isa/valhall_disasm.obj` | Valhall 反汇编实现 |
| `build/malioc-isa/malioc_disasm.exe` | 可直接编译 SPIR-V 并导出 ISA 的完整工具 |

### 已有 Mesa Meson 构建目录

对于已经配置 `-Dtools=panfrost` 的构建目录：

```sh
meson compile -C build malioc_disasm
```

程序位于 `build/src/panfrost/tools/malioc_disasm`，Windows 扩展名为 `.exe`。目标复用 `libpanfrost_valhall_disasm`，Windows 额外链接系统 BCrypt，设置 `install: false`。下面的运行示例使用单独构建的 EXE，采用 Meson 时替换 EXE 路径即可。

## 一次调用导出 ISA

在仓库根目录的 PowerShell 中运行：

```powershell
& build/malioc-isa/malioc_disasm.exe shader.frag.spv -o build/isa/fragment
& build/malioc-isa/malioc_disasm.exe shader.comp.spv -o build/isa/compute
& build/malioc-isa/malioc_disasm.exe shader.vert.spv -o build/isa/vertex
```

默认入口为 `main`。通过 `-n` 选择入口，通过 `--stage` 明确阶段并校验该入口的执行模型：

```powershell
& build/malioc-isa/malioc_disasm.exe shader.blob -n foo --stage vertex -o build/isa/foo
```

合并 vertex 的 Position/Varying：

```powershell
& build/malioc-isa/malioc_disasm.exe shader.vert.spv --no-idvs -o build/isa/vertex-combined
```

仅导出 MBS2、OBJC 和统计 JSON：

```powershell
& build/malioc-isa/malioc_disasm.exe shader.comp.spv --extract-only -o build/isa/compute
```

选择其他 DLL 路径或 GPU：

```powershell
& build/malioc-isa/malioc_disasm.exe shader.comp.spv --library "D:/ArmCompiler/Mali-Gxx_r56p1-00rel0.dll" -c Mali-G610 --revision r0p0 -o build/isa/compute
```

`--library` 只改变加载路径，仍要求文件 SHA256 与已分析版本一致。

## 参数

| 参数 | 用途与默认值 |
|---|---|
| `input` | Vulkan SPIR-V 文件，必填 |
| `-o`, `--output-prefix` | 输出前缀，必填；自动创建父目录 |
| `-n`, `--entrypoint` | 入口名称，默认 `main` |
| `--stage` | `vertex`、`fragment` 或 `compute`；默认自动识别 |
| `--library` | 编译器 DLL，默认使用上述安装路径 |
| `-c`, `--core` | GPU 名称，默认 `Mali-G610` |
| `--revision` | GPU revision，默认 `r0p0` |
| `--no-idvs` | 请求合并的 vertex shader |
| `--extract-only` | 跳过 ISA 文本，保留机器码及 JSON |
| `--mbs2` | 将输入作为已有 MBS2；必须指定 `--stage` |
| `--raw` | 将输入作为 Valhall OBJC，向标准输出反汇编；不使用 `-o` |
| `-h`, `--help` | 显示帮助 |

## 输出文件

以 `-o build/isa/compute` 为例：

```text
build/isa/compute.mbs2.bin
build/isa/compute.ebin0.objc
build/isa/compute.ebin0.isa.txt
build/isa/compute.ebin1.objc
build/isa/compute.ebin1.isa.txt
build/isa/compute.json
```

- `.mbs2.bin`：编译器返回的完整容器。
- `.ebin<N>.objc`：第 N 个 EBIN 的原始机器码。
- `.ebin<N>.isa.txt`：对应的 Mesa Valhall 反汇编文本。
- `.json`：阶段、入口、GPU/revision、IDVS、输入及 DLL 的 SHA256、各变体统计、警告，以及每个 EBIN 的头部、大小、偏移和输出路径。

`variants` 保留 DLL 的 `variant` 和 `is_pilot` 属性。不能用 EBIN 头部的 `flags_extended` 字段推断 pilot；合并 vertex 的 pilot 就可能将该字段置为 0。变体数量和顺序由实际编译结果决定，程序导出全部 EBIN。

输出路径按传入的前缀记录；相对路径以运行时工作目录为基准。重复使用前缀会覆盖本次生成的同名文件；旧的额外变体文件可能仍然存在，应以本次 JSON 的 `binaries` 列表为准。成功返回 0；参数、输入、DLL、GPU、编译或写入错误返回非零。SPIR-V 的入口、阶段和指令边界在加载 DLL 前检查。

## 已有机器码

直接反汇编 OBJC，不加载编译器：

```powershell
& build/malioc-isa/malioc_disasm.exe --raw build/isa/compute.ebin0.objc
```

提取并反汇编已有 MBS2：

```powershell
& build/malioc-isa/malioc_disasm.exe --mbs2 dump.mbs2.bin --stage compute -o build/isa/dump
```

该模式解析版本 54 的容器，检查 chunk 边界及每个 EBIN 内的 OBJC，不通过扫描标签寻找代码。其 JSON 包含阶段和二进制信息；已有容器无法提供新的 DLL 编译统计，`variants` 和 `warnings` 为空。

## r56p1 阶段枚举

以下是该 DLL 的 `malioc_compile` 将 `shader_type` 转换为后端阶段值的实际映射。IDA 地址基于 ImageBase `0x180000000`，对应 `0x1800A456B..0x1800A46DD`。

| 数值 | DLL 选项字符串 | 阶段 | 当前工具 |
|---:|---|---|---|
| 0 | `vertex` | 顶点 | 支持 |
| 1 | `tessellation_control` | 曲面细分控制 | 未接入 |
| 2 | `tessellation_evaluation` | 曲面细分求值 | 未接入 |
| 3 | `geometry` | 几何 | 未接入 |
| 4 | `fragment` | 片元 | 支持 |
| 5 | `compute` | 计算 | 支持 |
| 7 | `ray_generation` | 光线生成 | 未接入 |
| 8 | `ray_intersection` | 光线求交 | 未接入 |
| 9 | `ray_anyhit` | 任意命中 | 未接入 |
| 10 | `ray_closest_hit` | 最近命中 | 未接入 |
| 11 | `ray_miss` | 未命中 | 未接入 |
| 12 | `callable` | 可调用着色器 | 未接入 |

数值 6 未出现在这个入口的解析分支中，含义尚未确认。这里是 DLL 枚举，其他阶段不能直接用这些值作为 SPIR-V ExecutionModel，也不能把这些字符串直接作为 malioc.exe 命令行选项。

## 构建后测试

以下命令在仓库根目录的 PowerShell 中执行。解析测试直接调用新编译的 EXE，不需要 Arm DLL：

```powershell
python -B src/panfrost/tools/test_malioc_disasm.py build/malioc-isa/malioc_disasm.exe
```

应显示 `Ran 16 tests` 和 `OK`。测试覆盖阶段识别、入口选择、指令边界、各阶段 MBS2、多 EBIN、FCST、标签误扫描、元数据、缺失/重复 OBJC、每个截断位置、畸形长度/数量、Unicode 路径、JSON、原始机器码和命令行错误。

使用 Arm 安装包附带的真实样例测试完整编译流程：

```powershell
$maliocSamples = 'C:/Program Files/Arm/Arm Performance Studio 2026.5/mali_offline_compiler/samples/vulkan'
& build/malioc-isa/malioc_disasm.exe "$maliocSamples/shader.frag.spv" -o build/isa/build-test/fragment
& build/malioc-isa/malioc_disasm.exe "$maliocSamples/shader.comp.spv" -o build/isa/build-test/compute
& build/malioc-isa/malioc_disasm.exe "$maliocSamples/shader.vert.spv" -o build/isa/build-test/vertex
& build/malioc-isa/malioc_disasm.exe "$maliocSamples/shader-entrypoint-foo.vert.spv" -n foo -o build/isa/build-test/vertex-foo
& build/malioc-isa/malioc_disasm.exe "$maliocSamples/shader.vert.spv" --no-idvs -o build/isa/build-test/vertex-combined
```

启用了 `-Dbuild-tests=true` 的 Meson 构建可以执行：

```sh
meson test -C build malioc_disasm --print-errorlogs
```

2026-09-27 已实际完成 Windows x64 单独构建（VS2022、clang-cl、`/W4 /O2 /MD`），零编译警告；完整 EXE 为 97,792 字节。16 项原生测试通过。真实样例导出结果：

| 样例 | EBIN 数量 | OBJC 字节数 |
|---|---:|---|
| fragment | 1 | 24 |
| compute | 2 | 48、96 |
| vertex IDVS | 3 | 24、584、800 |
| vertex 合并模式 | 2 | 616、848 |
| vertex `foo` 入口 | 3 | 24、584、800 |

六个完整命令行场景的 MBS2 和 DLL 统计与之前验证的实现一致；三个阶段的工作寄存器、uniform 寄存器和栈大小与官方 malioc 统计一致。另验证了 14 个已有 MBS2、同进程各阶段连续三次编译与释放、编译失败释放后再次成功编译，以及入口、阶段、GPU 和 DLL hash 错误。

本机验证记录位于 `build/isa/native-validation/results.json`。公开接口的 64 字节 output 和后端的 72 字节 compilation result 均用结构体声明，并通过 `sizeof`/`offsetof` 编译期检查核对 DLL ABI。以上验证为工具的独立构建及运行测试，未进行全量 Mesa 构建。

文档生成标记：`biblioklept`（仓库 `AGENTS.md` 要求）。
