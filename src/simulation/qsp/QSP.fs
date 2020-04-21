// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.

module Microsoft.Quantum.Research.QuantumSignalProcessing.QSP

open System
open MPNumber
open QSPPrimitives
open MPPolynomials

/// <summary>
/// Finds roots by evaluating the polynomial through the Lindsey-Fox method.
/// For small degree polynomial we use the companion matrix method.
/// </summary>
/// <remark>
/// This is specialized to palindromic polynomials.
/// </remark>
let GetCrudePolynomialRoots (poly: RealPolynomial) =
    if poly.Degree < 100 then
        ObtainRootsByCompanionMatrix poly
    else
        let innerBound = poly.LowerBoundOnRootMagnitude
        let outerBound = 1.0
        let nRadii = max 100 poly.Degree
        let lfroots = ObtainSomeRootsByLindseyFoxMethod innerBound outerBound nRadii poly
        let inv = lfroots |> Array.map (fun z -> z.Inverse)
        Array.concat [| lfroots; inv |]

/// <summary>
/// Computes interspersing unitaries for quantum signal processing.
/// The result is returned in a 2-tuple of a 2-by-2 unitary and a list of normalized 3-vectors
/// that determine the primitive matrices.
/// </summary>
/// <param name="epsilon">Accuracy parameter. e.g. 1.0e-4 </param>
/// <param name="parity">(0,0),(0,1),(1,0), or (1,1).
/// The parity variable selects particular parity parts of the input function.</param>
/// <param name="mpInputValues"> Function values on Fourier points on the unit circle.
/// The length of the list must be a power of 2.
/// The sampling points should be sufficiently dense that the function can be determined by these function values.
/// </param>
/// <remark>
/// The input mpInputValues can be anything as long as the magnitude is below 1 - epsilon.
/// The precision of mpInputValues must be sufficiently high
/// so that the magnitude condition is satisfied within the numerical precision.
/// See [Haah, https://arxiv.org/abs/1806.10236].
/// </remark>
let QuantumSignalProcessingForDefiniteParity (epsilon: float) (parity: int*int) (mpInputValues: MPComplex[]) =

    let (evenReal, oddReal, evenImag, oddImag) =
        // Read off all parity components of the input function, and returns respective rational pure Laurent polynomials
        ConvertFunctionTableToRationalLaurentPolynomial epsilon mpInputValues

    let polyA = if fst parity % 2 = 0 then evenReal else oddReal
    let polyB = if snd parity % 2 = 0 then evenImag else oddImag

    #if DEBUG
    printfn "polyA = %A" polyA
    printfn "polyB = %A" polyB
    #endif

    // The following values are to be used in testing the accuracy of the decomposition,
    // which may be bad due to numerical instability.
    let projectedInput = TakeParityParts parity mpInputValues

    // Now that input real-on-circle polynomial functions (signal functions)
    // are determined in the form of rational Laurent polynomials,
    // we prepare for the decomposition into primitive matrices.
    // The following computes an integral polynomial whose roots are to be found.
    let ingredients = PrepareToFindComplementaryPolynomials GetCrudePolynomialRoots (polyA, polyB)

    // The decomposition of the signal polynomial is now computed.
    // In the worst case, this requires very high precision,
    // but the precision is dynamically determined under an exponential scheduling of the number of bits of precision.

    let rec compute (prec: uint32) =
        let qsp = ComputeFullSignalPolynomial ingredients prec
                        |> FullSignalPolynomialToPrimitiveDirections
        let maxdif = measureDistanceInFourierCoefficients projectedInput qsp
        //let maxdif = measureDistanceInFunctionValues prec projectedInput qsp

        if maxdif < 1.0 * epsilon / float qsp.PointsOnBlochSphere.Length then
            qsp
        else // Error is too large, so try with doubled precision.

            #if DEBUG
            printfn "unitarity violation of e0 = %A" qsp.FrontUnitaryCorrection.DeviationFromUnitary.Float
            printfn "eps = %A , seqlength = %A, maxdif = %A" epsilon qsp.PointsOnBlochSphere.Length maxdif
            //let outputCoef = ReconstructSignalPolynomial (e0,dirs) |> Array.map MPMatrix2x2.HalfSumOfAllEntries
            //let degree = max polyA.Degree polyB.Degree
            //let coeA = polyA.ConvertToComplexCoefficientList prec degree
            //let coeB = polyB.ConvertToComplexCoefficientList prec degree
            /// outputCoef is supposed to be very close to coaA + Sqrt{-1}*coeB.
            #endif

            compute (2u * prec)

    let initprec = 30.0 + Math.Log( float mpInputValues.Length / epsilon, 2.0)
    let result = compute (uint32 initprec)

    // This point is reached only if the reconstructed signal polynomial from the decomposition
    // is sufficiently close to the chosen parity sector of the input function.
    // The accuracy tolerance is 2*epsilon in the sup-norm of the functions.
    result

/// <summary>
/// Finds angles that encode the result of QuantumSignalProcessingForDefiniteParity
/// in the special case where the real part of the signal function is even and imaginary part is odd.
/// </summary>
/// <remark>
/// The convention of angles here should align with the usage of the quantum signal processing in Q#.
/// </remark>
/// <param name="qspResult"></param>
let ConvertToAnglesForParity01 (qspResult: QSPResult) =
    [   // For parity (0,1), the 3-vectors in the result is always in the equator of the Bloch sphere,
        // This is true not only for the 3-vectors determining the primitive matrices,
        // but also for the unitary correction $E_0 = \exp(i \phi \sigma^z \phi /2)$.
        // In Q# we will need rotations about z-axis. So, We convert each of the 3-vectors into an angle
        // measured from x-axis.
        [| qspResult.FrontUnitaryCorrection |> DiagonalUnitaryToAngle |]; // The unitary correction $E_0$.
        qspResult.PointsOnBlochSphere |> Array.map BlochVectorOnEquatorToAngleFromXaxis // all the 3-vectors.
    ]
    |> Array.concat
    |> Array.rev // This reversion is because we use left action for unitaries, while in Q# we write right-most gates in formulas on top.


/// <summary>
/// Wraps the entire classical computation of quantum signal processing for the signal function
/// $e^{i\theta} = z \mapsto \exp(\frac{\tau}{2}(z - z^{-1}) = e^{i \tau \sin \theta}$.
/// </summary>
/// <output>
/// Angles in the convension of ConvertToAnglesForParity01.
/// </output>
/// <param name="epsilon">Accuracy parameter in the open interval (0.0, 1.0).</param>
/// <param name="tau"> A nonzero real parameter. For time evolution by a Hamiltonian using qubitization,
/// this is equal to the evolution time multiplied by 1-norm of coefficients in the Pauli basis.</param>
let JacobiAngerExpansion (epsilon: float) (tau: float) =
    assert (epsilon > 0.0)
    assert (epsilon < 0.1)
    assert (tau <> 0.0)

    let log2N = Math.Log( Math.E * (Math.Abs tau) + 2.0 * Math.Log( 1.0 / epsilon ), 2.0) + 1.0 |> int

    let prec = 64.0 + Math.Log( float (1L <<< log2N) / epsilon, 2.0) |> uint32
    let mpfTau = MPF.fromFloat(tau, prec)
    let scaleFactor =
        // A rigorous analysis says that we might need to set the scaling factor as 1 - 10 eps,
        // but experiment with a wide range of epsilon (1.0e-2 to 1.0e-30) reveals that 1 - eps is sufficient.
        MPF.One - MPF.fromFloat(epsilon, prec)
    assert( scaleFactor.Sign > 0 )

    let twopiOver2N = (MPPi prec) |> MPF.Multiply2Power (1 - log2N)

    let functionValues = [| for k in 0..((1 <<< log2N) - 1) ->
                                // The high precision is needed to handle cases where epsilon is very small.
                                let theta = mpfTau * MPSin (k * twopiOver2N)
                                let v = { Real = MPCos theta; Imag = MPSin theta } * scaleFactor
                                assert( v.InsideUnitCircle )
                                v
                         |]


    // The function value for the Jacobi-Anger expansion has even real part and odd imaginary part, hence (0,1)
    // The result is a tuple whose first component is the unitary correction $E_0$ and real 3-vectors of length 1
    // that specify a point in the Bloch sphere, which in turn determines the projector in the primitive matrices.
    QuantumSignalProcessingForDefiniteParity epsilon (0,1) functionValues

/// <summary>
/// Wraps the entire classical computation of quantum signal processing for the signal function
/// $e^{i\theta} = z \mapsto \frac{2i}{\kappa (z - z^{-1}) = \frac{1}{\kappa \sin \theta}$.
/// </summary>
/// <output>
/// A 2-tuple of a 2-by-2 unitary and an array of normalized real 3-vectors that determine primitive matrices.
/// </output>
/// <param name="epsilon">Accuracy parameter in the open interval (0.0, 1.0).</param>
/// <param name="kappa">A real paramter >= 1.0; the accuracy is only guaranteed for $|\sin \theta| \ge \kappa^{-1}$.</param>
let OneOverSine (epsilon: float) (kappa: float) =
    assert (epsilon > 0.0)
    assert (epsilon < 0.1)
    assert (kappa >= 1.0)

    let b = kappa * kappa * Math.Log(2.0 / epsilon) |> Math.Ceiling |> uint32
    let bprime = kappa * Math.Log( 8.0 / epsilon ) |> Math.Ceiling |> int
    let log2N = Math.Log( 1.0 + 4.0 * float bprime, 2.0) |> Math.Ceiling |> int
    let N = 1 <<< log2N
    let prec = 32 + int (Math.Log( 1.0 / epsilon, 2.0)) + log2N |> uint32
    let mpfKappa = MPF.fromFloat(kappa, 64u)
    let scaleFactor = MPF.fromFloat( 0.5 / Math.Log( 8.0 / epsilon ), 64u)
    let twopiOver2N = (MPPi prec) |> MPF.Multiply2Power (1 - log2N)
    let myfun k =
        if k = 0 || k = N / 2 then
            MPComplex.ZeroP prec
        else
            let theta = k * twopiOver2N
            let ksinmpk = mpfKappa * MPSin theta
            let envelop = MPF.One - (MPFPower (2u * b) (MPCos theta))
            MPComplex.CreateReal (ksinmpk.Inverse * envelop * scaleFactor)
    let functionValues = [| for k in 0..(N - 1) -> myfun k |]

    QuantumSignalProcessingForDefiniteParity epsilon (1,0) functionValues

let LinearRamp (epsilon: float) (slope: float) =
    assert (epsilon > 0.0)
    assert (epsilon < 0.1)
    assert (slope > 0.0)

    let log2N = 15 // TODO: This has to be computed as a function of epsilon and slope.
    let prec = 48 + log2N + int(Math.Log(1.0/epsilon, 2.0)) |> uint32
    let gibbs = MPF.fromFloat(0.8, prec) // TODO: Replace it with a possibly better number.
    let twopiOver2N = (MPPi prec) |> MPF.Multiply2Power (1 - log2N)
    let myfun k =
        let theta = k * twopiOver2N
        let slopeSinTheta = slope * MPSin theta
        if slopeSinTheta.Sign >= 0 then
            if (slopeSinTheta - gibbs).Sign < 0 then
                MPComplex.CreateReal slopeSinTheta
            else
                MPComplex.CreateReal gibbs
        else
            if (slopeSinTheta + gibbs).Sign > 0 then
                MPComplex.CreateReal slopeSinTheta
            else
                MPComplex.CreateReal (-gibbs)
    let N = 1 <<< log2N
    let funval = [| for k in 0..(N - 1) -> myfun k |]
    printfn "%A" funval
    QuantumSignalProcessingForDefiniteParity epsilon (1,0) funval