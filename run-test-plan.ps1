param(
    [ValidateSet("Debug", "Release")]
    [string]$Config = "Debug",
    [string]$BuildDir = "build"
)

$ErrorActionPreference = "Stop"

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
    cmake -S . -B $BuildDir -DCMAKE_BUILD_TYPE=$Config
}

Invoke-Step -Name "Build ($Config)" -Action {
    cmake --build $BuildDir --config $Config
}

Invoke-Step -Name "Run Tests ($Config)" -Action {
    ctest --test-dir $BuildDir -C $Config --output-on-failure
}

Write-Host ""
Write-Host "Test plan completed successfully."
