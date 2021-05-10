// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.


namespace Microsoft.Quantum.Research.Samples {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Oracles;
    open Microsoft.Quantum.Research.Characterization;


    operation ExpOracle (eigenphase : Double, time : Double, register : Qubit[])
    : Unit is Adj + Ctl {
        Rz((2.0 * eigenphase) * time, register[0]);
    }


    operation RunPhaseEstimationSample(eigenphase : Double) : Double {
        let oracle = ContinuousOracle(ExpOracle(eigenphase, _, _));

        use eigenstate = Qubit();
        X(eigenstate);
        let est = RandomWalkPhaseEstimation(0.0, 1.0, 61, 100000, 0, oracle, [eigenstate]);
        Reset(eigenstate);
        return est;
    }

    @EntryPoint()
    operation RunMain() : Unit {
        let truePhase = 0.314;
        let est = RunPhaseEstimationSample(truePhase);
        Message($"True phase was {truePhase}, estimated was {est}.");
    }

}

