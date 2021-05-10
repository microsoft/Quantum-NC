// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.
namespace Microsoft.Quantum.Research.Tests {
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Diagnostics;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Oracles;
    open Microsoft.Quantum.Research.Characterization;    
    
    internal operation EvolveForTime(eigenphase : Double, time : Double, register : Qubit[])
    : Unit is Adj + Ctl {
        Rz((2.0 * eigenphase) * time, register[0]);
    }

    @Test("QuantumSimulator")
    operation CheckBayesianPERandomWalk() : Unit {        
        let expected = 0.571;
        let oracle = ContinuousOracle(EvolveForTime(expected, _, _));
        let nAttemptsAllowed = 3;
        mutable nAttemptsSoFar = 0;

        repeat {
            mutable actual = 0.0;
            use eigenstate = Qubit() {
                X(eigenstate);
                set actual = RandomWalkPhaseEstimation(0.0, 1.0, 61, 100000, 0, oracle, [eigenstate]);
                Reset(eigenstate);
            }
            if (AbsD(expected - actual) >= 0.05) {
                set nAttemptsSoFar += 1;
                Message($"CheckBayesianPERandomWalk failed on attempt {nAttemptsSoFar}, expected {expected} but got {actual}.");
            } else {
                return ();
            }
        }
        until nAttemptsSoFar >= nAttemptsAllowed;

        fail $"CheckBayesianPERandomWalk failed {nAttemptsSoFar} times in a row.";
    }
    
}
