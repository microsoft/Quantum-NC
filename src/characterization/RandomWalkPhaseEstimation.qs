// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.
namespace Microsoft.Quantum.Research.Characterization {
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Math;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Oracles;
    open Microsoft.Quantum.Characterization;

    internal operation _PrepAndMeasurePhaseEst(
        wInv : Double,
        t : Double,
        op : ((Double, Double, Qubit) => Unit))
    : Result {
        use target = Qubit();
        op(t, wInv, target);
        return MResetZ(target);
    }

    // NB: we take std.dev instead of variance here to avoid having to take a square root.

    /// # Summary
    /// Performs iterative phase estimation using a random walk to approximate
    /// Bayesian inference on the classical measurement results from a given
    /// oracle and eigenstate.
    ///
    /// # Input
    /// ## oracle
    /// An operation representing a unitary $U$ such that $U(t)\ket{\phi} = e^{i t \phi}\ket{\phi}$
    /// for eigenstates $\ket{\phi}$ with unknown phase $\phi \in \mathbb{R}^+$.
    /// ## targetState
    /// A register that $U$ acts on.
    /// ## initialMean
    /// Mean of the initial normal prior distribution over $\phi$.
    /// ## initialStdDev
    /// Standard deviation of the initial normal prior distribution over $\phi$.
    /// ## nMeasurements
    /// Number of measurements to be accepted into the final posterior estimate.
    /// ## maxMeasurements
    /// Total number of measurements than can be taken before the operation is considered to have failed.
    /// ## unwind
    /// Number of results to forget when consistency checks fail.
    ///
    /// # Output
    /// The final estimate $\hat{\phi} \mathrel{:=} \expect[\phi]$ , where
    /// the expectation is over the posterior given all accepted data.
    ///
    /// # Remarks
    /// ### Iterative Phase Estimation and Eigenstates
    /// In general, the input register `eigenstate` need not be an
    /// eigenstate $\ket{\phi}$ of $U$, but can be a superposition over
    /// eigenstates. Suppose that the input state is given by
    /// \begin{align}
    ///     \ket{\psi} & = \sum\_{j} \alpha\_j \ket{\phi\_j},
    /// \end{align}
    /// where $\{\alpha\_j\}$ are complex coefficients such that
    /// $\sum\_j |\alpha\_j|^2 = 1$ and where $U\ket{\phi\_j} = \phi\_j\ket{\phi\_j}$.
    ///
    /// Then, performing iterative phase estimation will eventually converge
    /// to a single eigenstate, as described in the
    /// [development guide](xref:microsoft.quantum.libraries.characterization#iterative-phase-estimation-without-eigenstates).
    ///
    /// ### Experiment Design
    /// The measurement times $t$ and inversion angles $\theta$
    /// passed to `oracle` are chosen according to
    /// the *particle guess heuristic*,
    /// \begin{align}
    ///     \theta \sim \Pr(\phi),\quad t \approx \frac{1}{\variance{\phi}}.
    /// \end{align}
    /// This heuristic is optimal for reducing the expected posterior variance
    /// in iterative phase estimation under the assumption of a normal prior.
    ///
    /// ### Optimality
    /// This operation approximates the optimal estimator for the phase
    /// $\phi$, as evaluated using the
    /// quadratic loss $L(\phi, \hat{\phi}) \mathrel{:=} (\phi - \hat{\phi})^2$.
    ///
    /// See [Bayesian Phase Estimation](xref:microsoft.quantum.libraries.characterization#bayesian-phase-estimation)
    /// for more details on the statistics of iterative phase estimation.
    ///
    /// # References
    /// - Ferrie *et al.* 2011 [doi:10/tfx](https://doi.org/10.1007/s11128-012-0407-6),
    ///   [arXiv:1110.3067](https://arxiv.org/abs/1110.3067).
    /// - Wiebe *et al.* 2013 [doi:10/tf3](https://doi.org/10.1103/PhysRevLett.112.190501),
    ///   [arXiv:1309.0876](https://arxiv.org/abs/1309.0876)
    /// - Wiebe and Granade 2018 *(in preparation)*.
    operation RandomWalkPhaseEstimation(
        initialMean : Double,
        initialStdDev : Double,
        nMeasurements : Int,
        maxMeasurements : Int,
        unwind : Int,
        oracle : ContinuousOracle,
        targetState : Qubit[]
    )
    : Double {
        let PREFACTOR = 0.79506009762065011;
        let INV_SQRT_E = 0.60653065971263342;
        let inner = ContinuousPhaseEstimationIteration(oracle, _, _, targetState, _);
        let sampleOp = _PrepAndMeasurePhaseEst(_, _, inner);
        mutable dataRecord = new Result[0];
        mutable mu = initialMean;
        mutable sigma = initialStdDev;
        mutable datum = Zero;
        mutable nTotalMeasurements = 0;
        mutable nAcceptedMeasurements = 0;

        repeat {

            if nTotalMeasurements >= maxMeasurements {
                return mu;
            }

            let wInv = mu - (PI() * sigma) / 2.0;
            let t = 1.0 / sigma;
            set datum = sampleOp(wInv, t);
            set nTotalMeasurements += 1;

            if datum == Zero {
                set mu -= sigma * INV_SQRT_E;
            } else {
                set mu += sigma * INV_SQRT_E;
            }

            set sigma *= PREFACTOR;
            set dataRecord += [datum];

            // Perform consistency check.
            if (nTotalMeasurements >= maxMeasurements) {
                return mu;
            }

            set nAcceptedMeasurements = nAcceptedMeasurements + 1;
        }
        until nAcceptedMeasurements >= nMeasurements;

        return mu;
    }

}


