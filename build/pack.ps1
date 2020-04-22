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

Write-Host "##[info]Packing Research libraries..."
@(
    '../src/chemistry/chemistry.csproj',
    '../src/characterization/characterization.csproj',
    '../src/simulation/qsp/QuantumSignalProcessing.fsproj'
    '../src/research/research.csproj'
) | ForEach-Object { Pack-One $_ }


if (-not $allOk) {
    throw "At least one test failed execution. Check the logs."
}
