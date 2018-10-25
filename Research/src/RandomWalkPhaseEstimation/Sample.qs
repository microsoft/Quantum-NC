// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.
namespace Microsoft.Quantum.Research.RandomWalkPhaseEstimation {
    
    open Microsoft.Quantum.Primitive;
    open Microsoft.Quantum.Canon;
    
    
    operation ExpOracle (eigenphase : Double, time : Double, register : Qubit[]) : Unit {
        
        body (...) {
            Rz((2.0 * eigenphase) * time, register[0]);
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    operation PhaseEstimationSample (eigenphase : Double) : Double {
        
        let oracle = ContinuousOracle(ExpOracle(eigenphase, _, _));
        mutable est = 0.0;
        
        using (eigenstate = Qubit[1]) {
            X(eigenstate[0]);
            set est = RandomWalkPhaseEstimation(0.0, 1.0, 61, 100000, 0, oracle, eigenstate);
            Reset(eigenstate[0]);
        }
        
        return est;
    }
    
}


