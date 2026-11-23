$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$py='.\.venv\Scripts\python.exe'
& $py --version
if ($LASTEXITCODE) {throw 'Python environment failed'}
& $py -m unittest -v
if ($LASTEXITCODE) {throw 'Unit tests failed'}
& $py study.py
if ($LASTEXITCODE) {throw 'Study failed'}
& $py verify_reports.py
if ($LASTEXITCODE) {throw 'Report verification failed'}
