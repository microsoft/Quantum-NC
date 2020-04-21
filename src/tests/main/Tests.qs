// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.
namespace Microsoft.Quantum.Research.Tests {
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Oracles;
    open Microsoft.Quantum.Research.Characterization;    
    
    internal operation ExpOracle (eigenphase : Double, time : Double, register : Qubit[])
    : Unit is Adj + Ctl {
        Rz((2.0 * eigenphase) * time, register[0]);
    }

    @Test("QuantumSimulator")
    operation BayesianPERandomWalkTest () : Unit {        
        let expected = 0.571;
        let oracle = ContinuousOracle(ExpOracle(expected, _, _));
        
        mutable actual = 0.0;
        using (eigenstate = Qubit()) {
            X(eigenstate);
            set actual = RandomWalkPhaseEstimation(0.0, 1.0, 61, 100000, 0, oracle, [eigenstate]);
            Reset(eigenstate);
        }
        
        // Give a very generous tolerance to reduce false positive rate.
        EqualityWithinToleranceFact(expected, actual, 0.05);
    }
    
}
