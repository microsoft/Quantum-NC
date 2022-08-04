# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

$ErrorActionPreference = 'Stop'

& "$PSScriptRoot/set-env.ps1"
$allOk = $True

function Pack-One() {
    Param($project)

    dotnet pack (Join-Path $PSScriptRoot $project) `
        --no-build `
        -c $Env:BUILD_CONFIGURATION `
        -v $Env:BUILD_VERBOSITY `
        -o $Env:NUGET_OUTDIR `
        /property:PackageVersion=$Env:NUGET_VERSION 

    $script:allOk = ($LastExitCode -eq 0) -and $script:allOk
}

function Pack-Wheel() {
    param(
        [string] $Path
    );

    Push-Location (Join-Path $PSScriptRoot $Path)
        python setup.py bdist_wheel sdist --formats=gztar

        if  ($LastExitCode -ne 0) {
            Write-Host "##vso[task.logissue type=error;]Failed to build $Path."
            $script:all_ok = $False
        } else {
            Copy-Item "dist/*.whl" $Env:PYTHON_OUTDIR
            Copy-Item "dist/*.tar.gz" $Env:PYTHON_OUTDIR
        }
    Pop-Location

}

Write-Host "##[info]Packing Research libraries..."
@(
    '../src/chemistry/chemistry.csproj',
    '../src/characterization/characterization.csproj',
    '../src/simulation/qsp/QuantumSignalProcessing.fsproj'
    '../src/research/research.csproj'
) | ForEach-Object { Pack-One $_ }


if ($Env:ENABLE_PYTHON -eq "false") {
    Write-Host "##vso[task.logissue type=warning;]Skipping Creating Python packages. Env:ENABLE_PYTHON was set to 'false'."
} else {
    Write-Host "##[info]Packing Python wheel..."
    python --version
    Pack-Wheel '../src/topogap-protocol'
}

if (-not $allOk) {
    throw "At least one test failed execution. Check the logs."
}
