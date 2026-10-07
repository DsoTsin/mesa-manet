param(
    [string]$BuildDirectory = "build/kraidoc-windows",
    [string]$VulkanSdk = $env:VULKAN_SDK
)

$ErrorActionPreference = "Stop"
$repository = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
if (-not $VulkanSdk -or -not (Test-Path -LiteralPath (Join-Path $VulkanSdk "Lib/SPIRV-Tools.lib"))) {
    throw "Set VULKAN_SDK to a Windows Vulkan SDK containing SPIRV-Tools.lib."
}
$sdkPath = (Resolve-Path -LiteralPath $VulkanSdk).Path
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
$visualStudio = & $vswhere -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $visualStudio) {
    throw "Visual Studio C++ x64 build tools were not found."
}
$vcvars = Join-Path $visualStudio "VC/Auxiliary/Build/vcvars64.bat"
$env:PATH = (Split-Path $vswhere) + ";" + $env:PATH
$environmentLines = & $env:COMSPEC /d /c "call `"$vcvars`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw "vcvars64.bat failed." }
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
$env:PATH = (Join-Path $sdkPath "Bin") + ";" + $env:PATH
$clang = (Get-Command clang-cl.exe -ErrorAction Stop).Source
$rust = (Get-Command rustc.exe -ErrorAction Stop).Source
$buildPath = [IO.Path]::GetFullPath((Join-Path $repository $BuildDirectory))
New-Item -ItemType Directory -Force -Path $buildPath | Out-Null
$nativeFile = Join-Path $buildPath "kraidoc-native.ini"
$clangPath = $clang.Replace('\', '/')
$rustPath = $rust.Replace('\', '/')
$sdkNativePath = $sdkPath.Replace('\', '/')
$nativeText = @"
[binaries]
c = '$clangPath'
cpp = '$clangPath'
rust = '$rustPath'
[properties]
vulkan-sdk = '$sdkNativePath'
"@
[IO.File]::WriteAllText($nativeFile, $nativeText, [Text.UTF8Encoding]::new($false))
$setupArguments = @('setup', $buildPath, $repository, '--native-file', $nativeFile,
    '-Dgallium-drivers=', '-Dvulkan-drivers=', '-Dtools=panfrost',
    '-Dpanfrost-rust=true', '-Dpanfrost-kmds=', '-Dplatforms=windows',
    '-Dglx=disabled', '-Degl=disabled', '-Dgbm=disabled', '-Dopengl=false',
    '-Dgles1=disabled', '-Dgles2=disabled', '-Dllvm=disabled',
    '-Dprecomp-compiler=auto', '-Dbuild-tests=false', '-Dzlib=disabled',
    '-Dzstd=disabled', '-Dxmlconfig=disabled', '-Db_vscrt=md',
    '-Dbuildtype=debugoptimized')
if (Test-Path -LiteralPath (Join-Path $buildPath 'meson-private/coredata.dat')) {
    $setupArguments += '--reconfigure'
}
$ErrorActionPreference = 'Continue'
& meson @setupArguments 2>&1 | ForEach-Object { Write-Output "$_" }
if ($LASTEXITCODE -ne 0) { throw "Meson configuration failed." }
& meson compile -C $buildPath kraidoc 2>&1 | ForEach-Object { Write-Output "$_" }
if ($LASTEXITCODE -ne 0) { throw "kraidoc compilation failed." }
$tool = Join-Path $buildPath "src/panfrost/tools/kraidoc.exe"
& python (Join-Path $PSScriptRoot "test_kraidoc.py") $tool (Join-Path $sdkPath "Bin/glslangValidator.exe")
if ($LASTEXITCODE -ne 0) { throw "kraidoc CPU tests failed." }
Write-Output $tool
