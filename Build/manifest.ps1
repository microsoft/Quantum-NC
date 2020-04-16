#!/usr/bin/env pwsh
#Requires -PSEdition Core

& "$PSScriptRoot/set-env.ps1"

@{
    Packages = @(
        "Microsoft.Quantum.Research.Chemistry",
        "Microsoft.Quantum.Research.Characterization",
        "Microsoft.Quantum.Research"
    );
    Assemblies = @(
        ".\src\characterization\bin\$Env:BUILD_CONFIGURATION\netstandard2.1\Microsoft.Quantum.Research.Characterization.dll",
        ".\src\chemistry\bin\$Env:BUILD_CONFIGURATION\netstandard2.1\Microsoft.Quantum.Research.Chemistry.dll"
    ) | ForEach-Object { Get-Item (Join-Path $PSScriptRoot ".." $_) };
} | Write-Output;
