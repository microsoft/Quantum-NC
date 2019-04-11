// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.

// This applies some methods described in
// Improving Quantum Algorithms for Quantum Chemistry
// M. B. Hastings, D. Wecker, B. Bauer, M. Troyer
// https://arxiv.org/abs/1403.1539
namespace Microsoft.Quantum.Research.Chemistry {
    open Microsoft.Quantum.Simulation;
    open Microsoft.Quantum.Primitive;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Extensions.Math;
    open Microsoft.Quantum.Chemistry;
    open Microsoft.Quantum.Chemistry.JordanWigner;
    open Microsoft.Quantum.Arrays;
    open Microsoft.Quantum.Math;
    
    
    // This evolution set runs off data optimized for a jordan-wigner encoding.
    // This collects terms Z, ZZ, PQandPQQR, hpqrs separately.
    // This only apples the needed hpqrs XXXX XXYY terms.
    // Operations here are expressed in terms of Z rotations and Clifford gates.
    // The CNOT-trick to double phase of controlled time-evolution in phase estimation is enabled.
    // This computes parity in one auxillary qubits. Assuming data is provided in lexicographic order,
    // this evolution set will enable Jordan-Wigner CNOT cancellation.
    
    /// # Summary
    /// Computes difference in parity between a previous PQRS... terms
    /// and the next PQRS... term. This difference is computed on a auxiliary
    /// qubit.
    ///
    /// # Input
    /// ## prevFermionicTerm
    /// List of indices to previous PQRS... terms.
    /// ## nextFermionicTerm
    /// List of indices to next PQRS... terms.
    /// ## aux
    /// Auxiliary qubit onto which parity computation results are stored.
    /// ## qubits
    /// Qubit acted on by all PQRS... terms.
    ///
    /// # Remarks
    /// This assumes that indices of P < Q < R < S < ... for both prevPQ and nextPQ.
    operation ApplyDeltaParity (prevFermionicTerm : Int[], nextFermionicTerm : Int[], aux : Qubit, qubits : Qubit[]) : Unit {
        
        body (...) {
            // Circuit with cancellation of neighbouring CNOTS
            let (minInt, bitStingApplyCNOT) = _DeltaParityCNOTbitstring_(prevFermionicTerm, nextFermionicTerm);
            
            for (idx in 0 .. Length(bitStingApplyCNOT) - 1) {
                
                if (bitStingApplyCNOT[idx] == true) {
                    CNOT(qubits[idx + minInt], aux);
                }
            }
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Classical processing step of `ApplyDeltaParity`.
    /// This computes a list of control qubits for evaluating parity
    /// difference between any two PQRS... terms of even length.
    ///
    /// # Remarks
    /// This assumes that the length of terms is even.
    /// Computes list of controls for parity difference between any two terms.
    /// This assumes that the input lists is sorted in ascending order.
    function _DeltaParityCNOTbitstring_ (prevFermionicTerm : Int[], nextFermionicTerm : Int[]) : (Int, Bool[]) {
        
        let minInt = Min(prevFermionicTerm + nextFermionicTerm);
        let maxInt = Max(prevFermionicTerm + nextFermionicTerm);
        let nInts = (maxInt - minInt) + 1;
        
        // Default Bool initialized to false.
        mutable prevBitString = new Bool[nInts];
        mutable nextBitString = new Bool[nInts];
        
        for (idxGroup in 0 .. Length(prevFermionicTerm) / 2 - 1) {
            
            for (idxQubit in (prevFermionicTerm[idxGroup * 2] + 1) - minInt .. (prevFermionicTerm[idxGroup * 2 + 1] - 1) - minInt) {
                set prevBitString[idxQubit] = true;
            }
        }
        
        for (idxGroup in 0 .. Length(nextFermionicTerm) / 2 - 1) {
            
            for (idxQubit in (nextFermionicTerm[idxGroup * 2] + 1) - minInt .. (nextFermionicTerm[idxGroup * 2 + 1] - 1) - minInt) {
                set nextBitString[idxQubit] = true;
            }
        }
        
        for (idx in 0 .. nInts - 1) {
            set nextBitString[idx] = XOR(prevBitString[idx], nextBitString[idx]);
        }
        
        return (minInt, nextBitString);
    }
    
    
    /// # Summary
    /// Used to change the basis of a Z operator to a Y operator by
    /// conjugation.
    ///
    /// # Input
    /// ## qubit
    /// Qubit whose basis is to be changed.
    operation _Ybasis_ (qubit : Qubit) : Unit {
        
        body (...) {
            Adjoint S(qubit);
            H(qubit);
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Used to change the basis of a Z operator to a Y operator.
    /// conjugation.
    ///
    /// # Input
    /// ## qubit
    /// Qubit whose basis is to be changed.
    operation _Xbasis_ (qubit : Qubit) : Unit {
        
        body (...) {
            H(qubit);
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    operation _ApplyBasisChange_ (ops : (Qubit => Unit : Adjoint, Controlled)[], qubits : Qubit[], targetQubit : Qubit) : Unit {
        
        body (...) {
            let allQubits = qubits + [targetQubit];
            
            for (idxOp in 0 .. Length(ops) - 1) {
                let op = ops[idxOp];
                ops[idxOp](allQubits[idxOp]);
            }
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Applies a Rz rotation, with a C-NOT trick to double phase
    /// in phase estimation.
    ///
    /// # Input
    /// ## angle
    /// Angle of Rz rotation.
    /// ## parityQubit
    /// Qubit that determines the sign of time-evolution.
    /// ## qubits
    /// Qubit acted on by Rz.
    operation _JWOptimizedZ_ (angle : Double, parityQubit : Qubit, qubit : Qubit) : Unit {
        
        body (...) {
            WithCA(CNOT(parityQubit, _), Rz(-2.0 * angle, _), qubit);
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Applies time-evolution by a Z term described by a `GeneratorIndex`.
    ///
    /// # Input
    /// ## term
    /// `GeneratorIndex` representing a Z term.
    /// ## stepSize
    /// Duration of time-evolution.
    /// ## parityQubit
    /// Qubit that determines the sign of time-evolution.
    /// ## qubits
    /// Qubits of Hamiltonian.
    operation _JWOptimizedZTerm_ (term : GeneratorIndex, stepSize : Double, parityQubit : Qubit, qubits : Qubit[]) : Unit {
        
        body (...) {
            let ((idxTermType, coeff), idxFermions) = term!;
            let angle = (1.0 * coeff[0]) * stepSize;
            let qubit = qubits[idxFermions[0]];
            _JWOptimizedZ_(angle, parityQubit, qubit);
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Applies time-evolution by a ZZ term described by a `GeneratorIndex`.
    ///
    /// # Input
    /// ## term
    /// `GeneratorIndex` representing a ZZ term.
    /// ## stepSize
    /// Duration of time-evolution.
    /// ## parityQubit
    /// Qubit that determines the sign of time-evolution.
    /// ## qubits
    /// Qubits of Hamiltonian.
    operation _JWOptimizedZZTerm_ (term : GeneratorIndex, stepSize : Double, parityQubit : Qubit, qubits : Qubit[]) : Unit {
        
        body (...) {
            let ((idxTermType, coeff), idxFermions) = term!;
            let angle = (1.0 * coeff[0]) * stepSize;
            let q1 = qubits[idxFermions[0]];
            let q2 = qubits[idxFermions[1]];
            WithCA(CNOT(q1, _), _JWOptimizedZ_(angle, parityQubit, _), q2);
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Applies time-evolution by a PQ term described by a `GeneratorIndex`.
    ///
    /// # Input
    /// ## term
    /// `GeneratorIndex` representing a PQ term.
    /// ## stepSize
    /// Duration of time-evolution.
    /// ## parityQubit
    /// Qubit that determines the sign of time-evolution.
    /// ## qubits
    /// Qubits of Hamiltonian.
    operation _JWOptimizedHpqTerm_ (term : GeneratorIndex, stepSize : Double, parityQubit : Qubit, qubits : Qubit[]) : Unit {
        
        body (...) {
            let ((idxTermType, coeff), idxFermions) = term!;
            WithCA(ApplyDeltaParity(new Int[0], idxFermions, parityQubit, _), _JWOptimizedHpqTerm__(term, stepSize, parityQubit, _), qubits);
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Implementation step of `JWOptimizedHpqTerm_`.
    operation _JWOptimizedHpqTerm__ (term : GeneratorIndex, stepSize : Double, parityQubit : Qubit, qubits : Qubit[]) : Unit {
        
        body (...) {
            let ((idxTermType, coeff), idxFermions) = term!;
            let angle = (1.0 * coeff[0]) * stepSize;
            let qubitP = qubits[idxFermions[0]];
            let qubitQ = qubits[idxFermions[1]];
            let qubitsPQ = Subarray(idxFermions[0 .. 1], qubits);
            let x = _Xbasis_(_);
            let y = _Ybasis_(_);
            let ops = [[x, x], [y, y]];
            let op0 = _JWOptimizedZ_(angle, parityQubit, _);
            let op1 = WithCA(CNOTChainTarget([qubitP], _), op0, _);
            
            for (idxOp in 0 .. Length(ops) - 1) {
                let op2 = WithCA(_ApplyBasisChange_(ops[idxOp], [qubitP], _), op1, _);
                op2(qubitQ);
            }
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Applies time-evolution by a PQ or PQQR term described by a `GeneratorIndex`.
    ///
    /// # Input
    /// ## term
    /// `GeneratorIndex` representing a PQ or PQQr term.
    /// ## stepSize
    /// Duration of time-evolution.
    /// ## parityQubit
    /// Qubit that determines the sign of time-evolution.
    /// ## qubits
    /// Qubits of Hamiltonian.
    operation _JWOptimizedPQandPQQRTerm_ (term : GeneratorIndex, stepSize : Double, parityQubit : Qubit, qubits : Qubit[]) : Unit {
        
        body (...) {
            let ((idxTermType, coeff), idxFermions) = term!;
            let angle = (1.0 * coeff[0]) * stepSize;
            let qubitQidx = idxFermions[1];
            
            // For all cases, do the same thing:
            // p < r < q (-1/2)(1+Z_q)(Z_{r-1,p+1})(X_p X_r + Y_p Y_r) (same as Hermitian conjugate of r < p < q)
            // q < p < r (-1/2)(1+Z_q)(Z_{r-1,p+1})(X_p X_r + Y_p Y_r)
            // p < q < r (-1/2)(1+Z_q)(Z_{r-1,p+1})(X_p X_r + Y_p Y_r)
            
            // This amounts to applying a PQ term, followed by same PQ term after a CNOT from q to the parity bit.
            let termPR0 = GeneratorIndex((idxTermType, [1.0]), idxFermions);
            
            if (Length(idxFermions) == 2) {
                _JWOptimizedHpqTerm_(termPR0, angle, parityQubit, qubits);
            }
            else {
                let termPR1 = GeneratorIndex((idxTermType, [1.0]), [idxFermions[0], idxFermions[3]]);
                WithCA(CNOT(qubits[qubitQidx], _), _JWOptimizedHpqTerm_(termPR1, angle, _, qubits), parityQubit);
            }
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Applies time-evolution by a PQRS term described by a `GeneratorIndex`.
    ///
    /// # Input
    /// ## term
    /// `GeneratorIndex` representing a PQRS term.
    /// ## stepSize
    /// Duration of time-evolution.
    /// ## parityQubit
    /// Qubit that determines the sign of time-evolution.
    /// ## qubits
    /// Qubits of Hamiltonian.
    operation _JWOptimized0123Term_ (term : GeneratorIndex, stepSize : Double, parityQubit : Qubit, qubits : Qubit[]) : Unit {
        
        body (...) {
            let ((idxTermType, coeff), idxFermions) = term!;
            WithCA(ApplyDeltaParity(new Int[0], idxFermions, parityQubit, _), _JWOptimized0123Term__(term, stepSize, parityQubit, _), qubits);
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Implementation step of `JWOptimized0123Term_`;
    operation _JWOptimized0123Term__ (term : GeneratorIndex, stepSize : Double, parityQubit : Qubit, qubits : Qubit[]) : Unit {
        
        body (...) {
            let x = _Xbasis_(_);
            let y = _Ybasis_(_);
            let angle = stepSize;
            
            // v0 v1 v2 v3 v0 v1 v2 v3
            let ops = [[x, x, x, x], [x, x, y, y], [x, y, x, y], [y, x, x, y], [y, y, y, y], [y, y, x, x], [y, x, y, x], [x, y, y, x]];
            let ((idxTermType, v0123), idxFermions) = term!;
            let qubitsPQRS = Subarray(idxFermions, qubits);
            let qubitsPQR = qubitsPQRS[0 .. Length(qubitsPQRS) - 2];
            let qubitS = qubitsPQRS[3];
            
            for (idxOp in 0 .. 7) {
                
                if (IsNotZero(v0123[idxOp % 4])) {
                    let op0 = _JWOptimizedZ_(angle * v0123[idxOp % 4], parityQubit, _);
                    let op1 = WithCA(CNOTChainTarget(qubitsPQR, _), op0, _);
                    let op2 = WithCA(_ApplyBasisChange_(ops[idxOp], qubitsPQR, _), op1, _);
                    op2(qubitS);
                }
            }
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Converts a Hamiltonian described by `JWOptimizedHTerms`
    /// to a `GeneratorSystem` expressed in terms of the
    /// `GeneratorIndex` convention defined in this file.
    ///
    /// # Input
    /// ## data
    /// Description of Hamiltonian in `JWOptimizedHTerms` format.
    ///
    /// # Output
    /// Representation of Hamiltonian as `GeneratorSystem`.
    function JWOptimizedGeneratorSystem (data : JWOptimizedHTerms) : GeneratorSystem {
        
        let (ZData, ZZData, PQandPQQRData, h0123Data) = data!;
        let ZGenSys = HTermsToGenSys(ZData, [0]);
        let ZZGenSys = HTermsToGenSys(ZZData, [1]);
        let PQandPQQRGenSys = HTermsToGenSys(PQandPQQRData, [2]);
        let h0123GenSys = HTermsToGenSys(h0123Data, [3]);
        return SumGeneratorSystems([ZGenSys, ZZGenSys, PQandPQQRGenSys, h0123GenSys]);
    }
    
    
    /// # Summary
    /// Simple state preparation of trial state by occupying
    /// spin-orbitals
    ///
    /// # Input
    /// ## qubitIndices
    /// Indices of qubits to be occupied by electrons.
    /// ## qubits
    /// Qubits of Hamiltonian.
    operation JWOptimizedStatePreparation (qubitIndices : Int[], qubits : Qubit[]) : Unit {
        
        body (...) {
            ApplyToEachCA(X, Subarray(qubitIndices, qubits));
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Represents a dynamical generator as a set of simulatable gates and an
    /// expansion in the JWOptimized basis.
    ///
    /// See [Dynamical Generator Modeling](../libraries/data-structures#dynamical-generator-modeling)
    /// for more details.
    ///
    /// # Input
    /// ## generatorIndex
    /// A generator index to be represented as unitary evolution in the JWOptimized
    /// basis.
    /// ## stepSize
    /// A multiplier on the duration of time-evolution by the term referenced
    /// in `generatorIndex`.
    /// ## parityQubit
    /// Qubit that determines the sign of time-evolution.
    /// ## qubits
    /// Register acted upon by time-evolution operator.
    operation JWOptimizedFermionEvolutionImpl (generatorIndex : GeneratorIndex, stepSize : Double, parityQubit : Qubit, qubits : Qubit[]) : Unit {
        
        body (...) {
            let ((idxTermType, idxDoubles), idxFermions) = generatorIndex!;
            let termType = idxTermType[0];
            
            if (termType == 0) {
                _JWOptimizedZTerm_(generatorIndex, stepSize, parityQubit, qubits);
            }
            elif (termType == 1) {
                _JWOptimizedZZTerm_(generatorIndex, stepSize, parityQubit, qubits);
            }
            elif (termType == 2) {
                _JWOptimizedPQandPQQRTerm_(generatorIndex, stepSize, parityQubit, qubits);
            }
            elif (termType == 3) {
                _JWOptimized0123Term_(generatorIndex, stepSize, parityQubit, qubits);
            }
        }
        
        adjoint invert;
        controlled distribute;
        controlled adjoint distribute;
    }
    
    
    /// # Summary
    /// Represents a dynamical generator as a set of simulatable gates and an
    /// expansion in the JWOptimized basis.
    ///
    /// # Input
    /// ## generatorIndex
    /// A generator index to be represented as unitary evolution in the JWOptimized
    /// basis.
    /// ## parityQubit
    /// Qubit that determines the sign of time-evolution.
    ///
    /// # Output
    /// An `EvolutionUnitary` representing time-evolution by the term
    /// referenced in `generatorIndex.
    function JWOptimizedFermionEvolutionFunction (generatorIndex : GeneratorIndex, parityQubit : Qubit) : EvolutionUnitary {
        
        return EvolutionUnitary(JWOptimizedFermionEvolutionImpl(generatorIndex, _, parityQubit, _));
    }
    
    
    /// # Summary
    /// Represents a dynamical generator as a set of simulatable gates and an
    /// expansion in the Pauli basis.
    ///
    /// # Input
    /// ## parityQubit
    /// Qubit that determines the sign of time-evolution.
    ///
    /// # Output
    /// An `EvolutionSet` that maps a `GeneratorIndex` for the JWOptimized basis to
    /// an `EvolutionUnitary.
    function JordanWignerOptimizedFermionEvolutionSet (parityQubit : Qubit) : EvolutionSet {
        
        return EvolutionSet(JWOptimizedFermionEvolutionFunction(_, parityQubit));
    }
    
    
    /// # Summary
    /// Returns optimized Trotter step operation and the parameters necessary to run it.
    ///
    /// # Input
    /// ## qSharpData
    /// Hamiltonian described by `JordanWignerEncodingData` format.
    /// ## trotterStepSize
    /// Step size of Trotter integrator.
    /// ## trotterOrder
    /// Order of Trotter integrator.
    ///
    /// # Output
    /// A tuple where: `Int` is the number of qubits allocated,
    /// `Double` is `1.0/trotterStepSize`, and the operation
    /// is the Trotter step.
    function OptimizedTrotterStepOracle (qSharpData : JordanWignerEncodingData, trotterStepSize : Double, trotterOrder : Int) : (Int, (Double, (Qubit[] => Unit : Adjoint, Controlled))) {
        
        let (nSpinOrbitals, data, statePrepData, energyShift) = qSharpData!;
        let oracle = _ApplyOptimizedTrotterStep_(qSharpData, trotterStepSize, _);
        let nTargetRegisterQubits = nSpinOrbitals + 1;
        let rescaleFactor = 0.5 / trotterStepSize;
        return (nTargetRegisterQubits, (rescaleFactor, oracle));
    }
    
    
    /// This operation applies an optimized Trotter step for a Hamiltonian described by
    /// the `JordanWignerEncodingData` type.
    operation _ApplyOptimizedTrotterStep_ (qSharpData : JordanWignerEncodingData, trotterStepSize : Double, allQubits : Qubit[]) : Unit {
        
        body (...) {
            let (nSpinOrbitals, data, statePrepData, energyShift) = qSharpData!;
            let parityQubit = allQubits[Length(allQubits) - 1];
            let systemQubits = allQubits[0 .. Length(allQubits) - 2];
            let generatorSystem = JWOptimizedGeneratorSystem(data);
            let evolutionGenerator = EvolutionGenerator(JordanWignerOptimizedFermionEvolutionSet(parityQubit), generatorSystem);
            let trotterOrder = 1;
            let simulationAlgorithm = TrotterSimulationAlgorithm(trotterStepSize, trotterOrder);
            simulationAlgorithm!(trotterStepSize, evolutionGenerator, systemQubits);
        }
        
        adjoint invert;
        
        controlled (ctrlRegister, ...) {
            let parityQubit = allQubits[Length(allQubits) - 1];
            X(parityQubit);
            Controlled X(ctrlRegister, parityQubit);
            _ApplyOptimizedTrotterStep_(qSharpData, trotterStepSize, allQubits);
            Controlled X(ctrlRegister, parityQubit);
            X(parityQubit);
        }
        
        controlled adjoint invert;
    }
    
}


