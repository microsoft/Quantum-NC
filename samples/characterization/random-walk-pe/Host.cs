// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.

#nullable enable

using System;
using System.Threading.Tasks;
using Microsoft.Quantum.Simulation.Simulators;

namespace Microsoft.Quantum.Research.Samples
{
    class Driver
    {
        static async Task Main(string[] args)
        {
            using var qsim = new QuantumSimulator();
            var truePhase = 0.314;
            var est = await PhaseEstimationSample.Run(qsim, truePhase);
            Console.WriteLine($"True phase was {truePhase}, estimated {est}.");
        }
    }
}
