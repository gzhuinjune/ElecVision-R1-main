$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Push-Location $Root
try {
    python -B -m unittest discover -s tests
    python -B -m elecvision_r1.eval_metrics --predictions examples/verification_predictions.json --ground-truth examples/verification_annotations.json --summary-only
    python -B scripts/export_verification_results.py --out-dir results
}
finally {
    Pop-Location
}
