# Compile le rapport Markdown en PDF (nécessite Pandoc + LaTeX)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "Extraction des métriques..."
python scripts/extract_report_metrics.py --fill reports/rapport_aps_scania.md

$Input = "reports/rapport_aps_scania_filled.md"
if (-not (Test-Path $Input)) {
    $Input = "reports/rapport_aps_scania.md"
}

Write-Host "Compilation PDF depuis $Input ..."
pandoc $Input `
    -o reports/rapport_aps_scania.pdf `
    --from markdown `
    --toc `
    --number-sections `
    -V geometry:margin=2.5cm `
    -V lang=fr `
    --resource-path=".;reports;reports/figures"

Write-Host "PDF : reports/rapport_aps_scania.pdf"
