// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the 
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries 
// and Samples. See LICENSE in the project root for license information.

namespace Microsoft.Research.Quantum.Tests {
    open Microsoft.Quantum.Primitive;
	open Microsoft.Quantum.Canon;
    open Microsoft.Research.Quantum.RandomWalkPhaseEstimation;

    operation AllocateQubitTest () : ()
    {
        body
        {
            using (qs = Qubit[1]) {
                Assert([PauliZ], [qs[0]], Zero, "Newly allocated qubit must be in |0> state");
            }
			Message("Test passed");
        }
    }

    operation BayesianPERandomWalkTest () : () {
        body {
            let expected = 0.571;
            let actual = PhaseEstimationSample(expected);
            // Give a very generous tolerance to reduce false positive rate.
            AssertAlmostEqualTol(expected, actual, 0.05);
        }
    }

}