param(
    [ValidateSet("Debug", "Release")]
    [string]$Config = "Debug",
    [string]$BuildDir = "build"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")

if ([System.IO.Path]::IsPathRooted($BuildDir)) {
    $ResolvedBuildDir = $BuildDir
} else {
    $ResolvedBuildDir = Join-Path $RepoRoot $BuildDir
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    Write-Host "==> $Name"
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name (exit code: $LASTEXITCODE)"
    }
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "cmake not found in PATH. Install CMake and reopen PowerShell."
}

if (-not (Get-Command ctest -ErrorAction SilentlyContinue)) {
    throw "ctest not found in PATH. Install CMake tools and reopen PowerShell."
}

Invoke-Step -Name "Configure ($Config)" -Action {
    cmake -S $RepoRoot -B $ResolvedBuildDir "-DCMAKE_BUILD_TYPE=${Config}"
}

Invoke-Step -Name "Build ($Config)" -Action {
    cmake --build $ResolvedBuildDir --config $Config
}

Invoke-Step -Name "Run Tests ($Config)" -Action {
    ctest --test-dir $ResolvedBuildDir -C $Config --output-on-failure
}

Write-Host ""
Write-Host "Test plan completed successfully."
