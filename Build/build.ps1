# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

$ErrorActionPreference = 'Stop'

& "$PSScriptRoot/set-env.ps1"
$allOk = $True

function Build-One {
    param(
        [string]$action,
        [string]$project
    );

    Write-Host "##[info]Building $project"
    dotnet $action (Join-Path $PSScriptRoot $project) `
        -c $Env:BUILD_CONFIGURATION `
        -v $Env:BUILD_VERBOSITY `
        /property:DefineConstants=$Env:ASSEMBLY_CONSTANTS `
        /property:Version=$Env:ASSEMBLY_VERSION `
        /property:QsharpDocsOutputPath=$Env:DOCS_OUTDIR

    $script:allOk = ($LastExitCode -eq 0) -and $script:allOk
}

Build-One 'publish' '../Quantum-NC.sln'
Build-One 'build' '../Samples.sln'

if (-not $allOk) {
    throw "At least one test failed execution. Check the logs."
}
