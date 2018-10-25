// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the 
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries 
// and Samples. See LICENSE in the project root for license information.

using System;
using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;

namespace Microsoft.Quantum.Research.RandomWalkPhaseEstimation
{
    class Driver
    {
        static void Main(string[] args)
        {
            
            // We begin by defining a quantum simulator to be our target
            // machine.
            var sim = new QuantumSimulator(throwOnReleasingQubitsNotInZeroState: true);

            // Next, we pick an arbitrary value for the eigenphase to be
            // estimated. Note that we have assumed in the Q# operations that
            // the prior for the phase φ is supported only on the interval
            // [0, 1], so you might get inconsistent answers if you violate
            // that constraint. Try it out!
            const Double eigenphase = 0.344;

            System.Console.WriteLine("Bayesian Phase Estimation w/ Random Walk:");
            var est = PhaseEstimationSample.Run(sim, eigenphase).Result;
            System.Console.WriteLine($"Expected {eigenphase}, estimated {est}.");
            System.Console.ReadLine();

        }
    }
}
