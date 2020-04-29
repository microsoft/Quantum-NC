// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.

#nullable enable

using System;
using System.Threading.Tasks;

using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;
using Microsoft.Quantum.Research.QuantumSignalProcessing;

namespace Microsoft.Quantum.Research.Samples
{
    public class Host
    {
        static async Task Main(string[] args)
        {
            var tau = 5.31;
            var qspResult = QSP.JacobiAngerExpansion(1.0e-20, tau);
            var dirs = new QArray<double>( QSP.ConvertToAnglesForParity01(qspResult) );
            using var qsim = new QuantumSimulator();
            await SampleHamiltonianEvolutionByQSP.Run(qsim, tau, dirs);
            Console.WriteLine("JacobiAnger QSP Done.");
        }
    }
}
