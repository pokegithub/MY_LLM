param(
    [switch]$DryRun,
    [switch]$Execute,
    [string]$TaskClass = "candidate_readiness_smoke",
    [string]$CandidateModelPath = "./run_artifacts/local_models/qwen2.5-coder-0.5b-instruct",
    [string]$GpuName = ""
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptRoot "..\..")
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$argsList = @(
    (Join-Path $ScriptRoot "cloud_readiness_runner.py"),
    "--task-class", $TaskClass,
    "--candidate-model-path", $CandidateModelPath
)

if ($GpuName -ne "") {
    $argsList += @("--gpu-name", $GpuName)
}

if ($Execute) {
    $argsList += @("--run-smoke", "--execute")
} else {
    $argsList += @("--dry-run")
}

& $Python @argsList
exit $LASTEXITCODE
