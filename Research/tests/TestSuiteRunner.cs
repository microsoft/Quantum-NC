// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the 
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries 
// and Samples. See LICENSE in the project root for license information.

using Microsoft.Quantum.Simulation.XUnit;
using Microsoft.Quantum.Simulation.Simulators;
using Xunit.Abstractions;
using System.Diagnostics;
using System.Text;
using System;
using System.Security.Cryptography;

namespace Microsoft.Quantum.Research.Tests
{
    public class TestSuiteRunner
    {
        private readonly ITestOutputHelper output;

        public TestSuiteRunner(ITestOutputHelper output)
        {
            this.output = output;
        }

        /// <summary>
        /// This driver will run all Q# tests (operations named "...Test") 
        /// that belong to the namespace Microsoft.Quantum.Research.Tests.
        /// </summary>
        [OperationDriver]
        public void TestTarget(TestOperation op)
        {
            // It is convenient to use a seed for test that can fail with small probability
            uint? seed = RetrieveGeneratedSeed(op);

            using (var sim = new QuantumSimulator(randomNumberGeneratorSeed: seed))
            {
                // OnLog defines action(s) performed when Q# test calls function Message
                sim.OnLog += (msg) => { output.WriteLine(msg); };
                sim.OnLog += (msg) => { Debug.WriteLine(msg); };
                op.TestOperationRunner(sim);
            }
        }

        /// <summary>
        /// Returns a seed to use for the test run based on the class
        /// </summary>
        private uint? RetrieveGeneratedSeed(TestOperation op)
        {
            byte[] bytes = Encoding.Unicode.GetBytes(op.fullClassName);
            byte[] hash = hashMethod.ComputeHash(bytes);
            uint seed = BitConverter.ToUInt32(hash, 0);

            string msg = $"Using generated seed: (\"{ op.fullClassName}\",{ seed })";
            output.WriteLine(msg);
            Debug.WriteLine(msg);

            return seed;
        }

        private static readonly SHA256Managed hashMethod = new SHA256Managed();
    }
}
