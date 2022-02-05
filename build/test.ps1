# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

$ErrorActionPreference = 'Stop'

& "$PSScriptRoot/set-env.ps1"
$all_ok = $True

function Test-One {
    Param($project)

    $TestsLogs = Join-Path $Env:LOGS_OUTDIR log-tests-quantum-nc.txt

    dotnet test (Join-Path $PSScriptRoot $project) --diag:"$TestsLogs" `
        -c $Env:BUILD_CONFIGURATION `
        -v $Env:BUILD_VERBOSITY `
        --logger trx `
        /property:DefineConstants=$Env:ASSEMBLY_CONSTANTS `
        /property:Version=$Env:ASSEMBLY_VERSION

    $script:all_ok = ($LastExitCode -eq 0) -and $script:all_ok
}

Write-Host "##[info]Testing Quantum-NC.sln"
Test-One '../Quantum-NC.sln'


if (-not $all_ok) {
    throw "At least one test failed execution. Check the logs."
}
