param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [string]$BuildDir = "build",
    [string]$Config = "Debug",
    [string]$BaselineCsv = "tests/netlib_baseline.csv",
    [string]$OutputCsv = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RunnerPath {
    param(
        [string]$BuildDir,
        [string]$Config
    )

    $candidate1 = Join-Path $BuildDir "lp_solver_netlib_runner.exe"
    $candidate2 = Join-Path (Join-Path $BuildDir $Config) "lp_solver_netlib_runner.exe"

    if (Test-Path $candidate1) { return $candidate1 }
    if (Test-Path $candidate2) { return $candidate2 }
    throw "Cannot find lp_solver_netlib_runner.exe under '$BuildDir'. Build target 'lp_solver_netlib_runner' first."
}

function ConvertFrom-KeyValueOutput {
    param([string[]]$Lines)
    $map = @{}
    foreach ($line in $Lines) {
        if ($line -match "^[^=]+=") {
            $parts = $line.Split("=", 2)
            $map[$parts[0].Trim()] = $parts[1].Trim()
        }
    }
    return $map
}

function Import-NetlibReadmeBaseline {
    param([string]$ReadmePath)

    $map = @{}
    if (-not (Test-Path $ReadmePath)) {
        return $map
    }

    $inTable = $false
    foreach ($line in Get-Content $ReadmePath) {
        if ($line -match "PROBLEM SUMMARY TABLE") {
            $inTable = $true
            continue
        }
        if (-not $inTable) {
            continue
        }
        if ($line -match "^\s*Name\s+Rows") {
            continue
        }
        if ($line -match "^\s*-{5,}\s*$") {
            continue
        }
        if ($line -match "^\s*$") {
            continue
        }
        if ($line -match "^\*\*") {
            continue
        }

        $parts = @($line -split "\s+" | Where-Object { $_ -ne "" })
        if ($parts.Length -lt 2) {
            continue
        }

        $name = $parts[0].ToUpperInvariant()
        $objective = $parts[-1]
        if ($objective -match "^[\-+]?[\d.]+([Ee][\-\+]?\d+)?$") {
            $map[$name] = [pscustomobject]@{
                problem   = $name
                status    = "optimal"
                objective = $objective
                tolerance = "1e-6"
            }
        }
    }

    return $map
}

$runner = Resolve-RunnerPath -BuildDir $BuildDir -Config $Config
$dataDirResolved = Resolve-Path $DataDir

$baseline = @{}
if (Test-Path $BaselineCsv) {
    $baselineRows = Import-Csv $BaselineCsv
    foreach ($row in $baselineRows) {
        $name = $row.problem.Trim().ToUpperInvariant()
        $baseline[$name] = $row
    }
}

$readmePath = Join-Path $dataDirResolved "README.netlib"
$readmeBaseline = Import-NetlibReadmeBaseline -ReadmePath $readmePath
foreach ($entry in $readmeBaseline.GetEnumerator()) {
    if (-not $baseline.ContainsKey($entry.Key)) {
        $baseline[$entry.Key] = $entry.Value
    }
}

if ([string]::IsNullOrWhiteSpace($OutputCsv)) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputCsv = "netlib-results-$timestamp.csv"
}

$results = @()
$mpsFiles = @(Get-ChildItem -Path $dataDirResolved -Filter "*.mps" -File | Sort-Object Name)

if ($mpsFiles.Length -eq 0) {
    throw "No .mps files found under '$dataDirResolved'. Decompress .mps.gz files before running."
}

foreach ($file in $mpsFiles) {
    $problem = [System.IO.Path]::GetFileNameWithoutExtension($file.Name).ToUpperInvariant()
    $runnerArgs = @($file.FullName)

    if ($baseline.ContainsKey($problem)) {
        $b = $baseline[$problem]
        $runnerArgs += @("--ref", "$($b.objective)", "--tol", "$($b.tolerance)")
    }

    Write-Host "Running $problem ..."
    $output = & $runner @runnerArgs 2>&1
    $exitCode = $LASTEXITCODE

    $kv = ConvertFrom-KeyValueOutput -Lines $output
    $results += [pscustomobject]@{
        problem         = $problem
        file            = $file.FullName
        exit_code       = $exitCode
        classification  = ($kv["classification"] | ForEach-Object { $_ }) -join ""
        status          = ($kv["status"] | ForEach-Object { $_ }) -join ""
        rows            = ($kv["rows"] | ForEach-Object { $_ }) -join ""
        cols            = ($kv["cols"] | ForEach-Object { $_ }) -join ""
        iterations      = ($kv["iterations"] | ForEach-Object { $_ }) -join ""
        time_ms         = ($kv["time_ms"] | ForEach-Object { $_ }) -join ""
        objective       = ($kv["objective"] | ForEach-Object { $_ }) -join ""
        reference       = ($kv["reference"] | ForEach-Object { $_ }) -join ""
        objective_diff  = ($kv["objective_diff"] | ForEach-Object { $_ }) -join ""
        objective_tol   = ($kv["objective_tol"] | ForEach-Object { $_ }) -join ""
        objective_match = ($kv["objective_match"] | ForEach-Object { $_ }) -join ""
        error           = ($kv["error"] | ForEach-Object { $_ }) -join ""
    }
}

$results | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8
Write-Host "Done. Wrote results to $OutputCsv"
