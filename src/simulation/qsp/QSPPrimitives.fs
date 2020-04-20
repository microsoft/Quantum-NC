// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.

module Microsoft.Quantum.Research.QuantumSignalProcessing.QSPPrimitives

open System
open MPNumber
open MPFFT
open MPPolynomials

/// <summary>
/// Record type that contains the specification for the product decomposition of a signal polynomial.
/// </summary>
/// <remark>
/// The convention here follows that of [Haah, https://arxiv.org/abs/1806.10236].
/// FrontUnitaryCorrection contains the matrix $E_0$,
/// and PointsOnBlochSphere, an array of length $2n$,
/// determines the projectors $P_1,\ldots,P_{2n}$ of the primitive matrices.
/// </remark>
type QSPResult = { FrontUnitaryCorrection: MPMatrix2x2; PointsOnBlochSphere: MPVector[] }

let ProjectorToBlochVector (proj: MPMatrix2x2) =
    // TODO: Consider unwrapping the arithmetic for the Pauli matrices.
    let x = (MPMatrix2x2.SigmaX * proj |> MPMatrix2x2.Tr).Real
    let y = (MPMatrix2x2.SigmaY * proj |> MPMatrix2x2.Tr).Real
    let z = (MPMatrix2x2.SigmaZ * proj |> MPMatrix2x2.Tr).Real
    // The imaginary part is supposed to be tiny.
    MPVector.Normalize {X = x; Y = y; Z = z}

let BlochVectorToProjector (v: MPVector) =
    // The input vector (x,y,z) must have a unit norm.
    let ans = {
            A = { Real = MPF.One + v.Z; Imag = MPF.ZeroP v.Z.Precision }
            B = { Real = v.X; Imag = -v.Y }
            C = { Real = v.X; Imag = v.Y }
            D = { Real = MPF.One - v.Z; Imag = MPF.ZeroP v.Z.Precision }
        }
    ans.Half

let ReconstructSignalPolynomial (qsp: QSPResult) =
    let degPoly = qsp.PointsOnBlochSphere.Length
    let matrixRegister = Array.create (degPoly + 1) qsp.FrontUnitaryCorrection
    if degPoly > 0 then
        matrixRegister.[1..] <- (qsp.PointsOnBlochSphere |> Array.map BlochVectorToProjector)

    for k = 1 to degPoly do
        let nowP = matrixRegister.[k]
        for idx = k downto 0 do
        // matrixRegister.[0..(k-1)] contains the result for E_0, P_1, ..., P_{k-1}
            let coefdif =
                if idx = 0 then
                    - matrixRegister.[0]
                elif idx < k then
                    matrixRegister.[idx - 1] - matrixRegister.[idx]
                else
                    matrixRegister.[idx - 1]

            if idx = k then
                matrixRegister.[k  ] <- coefdif * nowP
            else
                matrixRegister.[idx] <- matrixRegister.[idx] + coefdif * nowP

    matrixRegister

let EvaluateSignalPolynomialUsingBlochVectors (qsp: QSPResult) (z: MPComplex) =
    let t = MPComplex.Sqrt z
    let tInverse = t.Conjugate
    let tp = t - tInverse
    let init = qsp.FrontUnitaryCorrection
    let update now bv =
        let projP = BlochVectorToProjector bv
        let u = { A = tp * projP.A + tInverse
                  B = tp * projP.B
                  C = tp * projP.C
                  D = tp * projP.D + tInverse
                }
        now * u
    Array.fold update init qsp.PointsOnBlochSphere

let ReconstructFunctionValues (prec: uint32) (log2N: int) (qsp:QSPResult) =
    let tf = TwiddleFactor(log2N, prec)
    let N = 1 <<< log2N
    [|
        for idx in 0..(N - 1) ->
            EvaluateSignalPolynomialUsingBlochVectors qsp (tf.Compute idx)
            |> MPMatrix2x2.HalfSumOfAllEntries
    |]

type IngredientsForQSP =
    {
        PolyA                      : RationalPureLaurentPolynomial
        PolyB                      : RationalPureLaurentPolynomial
        OneSubAB                   : IntegralLaurentPolynomial
        SquareHasEvenExponentsOnly : bool
        Degree                     : int
        RootRef                    : RootRefinery
    }

let PrepareToFindComplementaryPolynomials
    (initialRootFinder: RealPolynomial -> MPComplex[])
    (polyA: RationalPureLaurentPolynomial, polyB: RationalPureLaurentPolynomial)
    =
    let denA = polyA.Denominator
    let denB = polyB.Denominator
    let commonDen = LeastCommonMultiple [| denA; denB |]
    let a2 = (IntegralLaurentPolynomial.Square polyA.Numerator) * ((commonDen / denA) * (commonDen / denA))
    let b2 = (IntegralLaurentPolynomial.Square polyB.Numerator) * ((commonDen / denB) * (commonDen / denB))
    let scaledOne = IntegralLaurentPolynomial( [| commonDen * commonDen |], 0 )
    let oneMinusAsqBsq =
        match (polyA.Parity % 2 = 0, polyB.Parity % 2 = 0) with
        | (true , true ) -> scaledOne - a2 - b2
        | (true , false) -> scaledOne - a2 + b2
        | (false, true ) -> scaledOne + a2 - b2
        | (false, false) -> scaledOne + a2 + b2 // taking care of the missing factor \sqrt{-1}.

    let squareHasEvenExponentsOnly, oneSubAB =
        // When f(z) = 1 - polyA^2 - polyB^2,
        // oneSubAB is equal to
        // f if not squareHasEvenExponentsOnly, or
        // g defined by g(z^2) = f(z) if squareHasEvenExponentsOnly
        oneMinusAsqBsq.ToPolyOfSquaredVariable()

    assert(oneSubAB.IsReciprocal)

    let rootref = RootRefinery(oneSubAB.ToRealPolynomial)

    let initialRootApproximation = oneSubAB.ToRealPolynomial |> initialRootFinder
    // Feed the found approximate roots to the root refinery.
    initialRootApproximation |> rootref.SetApproximateRoots

    #if DEBUG
    printfn "Initially Found %A roots out of %A" initialRootApproximation.Length (2 * oneSubAB.Degree)
    #endif

    // Refine roots to 64 bits of precision.
    rootref.ComputeRoots 64u

    { // IngredientsForQSP
        PolyA = polyA
        PolyB = polyB
        SquareHasEvenExponentsOnly = squareHasEvenExponentsOnly
        OneSubAB = oneSubAB
        Degree = max polyA.Numerator.Degree polyB.Numerator.Degree
        RootRef = rootref
    }

let EvaluateIntermediatePolynomialE (ingredients: IngredientsForQSP) (z: MPComplex) =
    let zInverse = z.Inverse
    let init =
        if ingredients.SquareHasEvenExponentsOnly then
            MPComplex.Create(1.0, 0.0, z.Precision)
        else
            let nprime = ingredients.RootRef.RootsInsideUnitDisk.Length |> uint32
            MPComplex.Power (nprime / 2u) zInverse
    let update sofar root =
            if ingredients.SquareHasEvenExponentsOnly then
                sofar * (z - root * zInverse)
            else
                sofar * (z - root)
    Array.fold update init ingredients.RootRef.RootsInsideUnitDisk

let ExpandIntermediatePolynomialE (ingredients: IngredientsForQSP) =
    let log2N = Math.Log (float (2 * ingredients.Degree) + 3.0) / (Math.Log 2.0) |> Math.Ceiling |> int
    let tf = TwiddleFactor(log2N, ingredients.RootRef.CurrentPrecision + uint32 log2N)

    let valueTable =  // TODO: Optimize and parallelize
        [|
            for idx in 0..((1 <<< log2N) - 1) ->
                EvaluateIntermediatePolynomialE ingredients (tf.Compute idx)
        |]
    let fou = MPFFT.FFTWithGivenTwiddleFactor valueTable tf |> MPFFT.DivideByLength
    let pos = fou.[.. ingredients.Degree]
    let neg = fou.[(fou.Length - ingredients.Degree)..]
    let coef = Array.concat [ neg; pos ]
    coef

let ComputeFullSignalPolynomial (ingredients: IngredientsForQSP) (precision: uint32) =
    //This is the core routine of all.

    ingredients.RootRef.ComputeRoots precision

    assert(2 * ingredients.RootRef.RootsInsideUnitDisk.Length = ingredients.RootRef.NRoots)

    let polyA = ingredients.PolyA
    let polyB = ingredients.PolyB

    let polyEValueAtOne = MPComplex.Create(1.0, 0.0, precision) |> EvaluateIntermediatePolynomialE ingredients

    let halfAlphaSqrt =
        let e1sq = polyEValueAtOne * polyEValueAtOne
        assert(e1sq.Real.Sign > 0)
        let rationalAt1Sq (poly:RationalPureLaurentPolynomial) =
            let mysum = poly.Numerator.CoefficientSum
            let n = MPComplex.CreateReal (MPF.fromBigInteger mysum            |> curry MPF.ChangePrecision precision)
            let d = MPComplex.CreateReal (MPF.fromBigInteger poly.Denominator |> curry MPF.ChangePrecision precision)
            (n / d) * (n / d)
        let a1sq = rationalAt1Sq ingredients.PolyA
        let b1sq = rationalAt1Sq ingredients.PolyB
        let one = MPComplex.One |> MPComplex.ChangePrecision precision
        let num =
            match (polyA.Parity % 2 = 0, polyB.Parity % 2 = 0) with // taking care of the missing factor \sqrt{-1}.
            | (true , true ) -> one - a1sq - b1sq
            | (true , false) -> one - a1sq + b1sq
            | (false, true ) -> one + a1sq - b1sq
            | (false, false) -> one + a1sq + b1sq
        assert(num.Real.Sign > 0)
        let alphaSqrt = (num / e1sq) |> MPComplex.Sqrt

        #if DEBUG
        printfn "alphaSqrt/2 = %O" alphaSqrt.Half
        #endif

        alphaSqrt.Half

    // Formation of a SU(2)-valued Laurent polynomial.
    // Only the coefficients will be returned; the exponents are inferred from the length of the returned array.
    let Ecoeff = ExpandIntermediatePolynomialE ingredients |> Array.map (fun z -> z * halfAlphaSqrt)
    let degree = ingredients.Degree
    // polyA or polyB may have smaller degree then ingredients.Degree.
    // So we need to tell the following converter the "degree = length of the output."
    let coefA = polyA.ConvertToComplexCoefficientList precision degree
    let coefB = polyB.ConvertToComplexCoefficientList precision degree


    let sigpoly =
        // A completed signal polynomial, which is valued in SU(2) on the unit circle.
        // It is a list of coefficients that are 2x2 matrices starting from exponent -deg to +deg.
        let timesI (z: MPComplex) = z * MPComplex.Create(0.0, 1.0)
        Array.init (2 * degree + 1) (fun idx ->
            let coefC  = Ecoeff.[idx] + Ecoeff.[Ecoeff.Length - idx - 1]
            let icoefD = Ecoeff.[idx] - Ecoeff.[Ecoeff.Length - idx - 1]
            // aI + biX + diY + ciZ; c is reciprocal, while d is anti-reciprocal.
            // This is a convenient choice as in Sec. 3.1 of arXiv:1806.10236
            let mat11 = coefA.[idx] + (timesI coefC)
            let mat22 = coefA.[idx] - (timesI coefC)
            let mat12 = coefB.[idx] - icoefD |> timesI
            let mat21 = coefB.[idx] + icoefD |> timesI
            { A = mat11; B = mat12; C = mat21; D = mat22 }
        )

    // Now the sigpoly is constructed.
    sigpoly

let FullSignalPolynomialToPrimitiveDirections (sigpoly: MPMatrix2x2[]) =
    // Decomposing into primitive matrices, and return Bloch vectors.

    let nowCoefList = Array.init sigpoly.Length (fun idx -> sigpoly.[idx]) // copy
    let blochVs = Array.create (sigpoly.Length - 1) MPVector.Zero

    for d = sigpoly.Length downto 2 do

        let projQ = nowCoefList.[d - 1] |> MPMatrix2x2.ComputeRightProjector

        #if DEBUG
        let pn = projQ |> MPMatrix2x2.FrobeniusNorm |> fun f -> f.Float
        if pn < 0.9999 || pn > 1.0001 then
            printfn "projector norm = %A" pn
        #endif
        blochVs.[d - 2] <- projQ |> ProjectorToBlochVector

        for idx = 0 to d - 2 do
           nowCoefList.[idx] <-
                nowCoefList.[idx] + (nowCoefList.[idx + 1] - nowCoefList.[idx]) * projQ

    { FrontUnitaryCorrection = nowCoefList.[0]; PointsOnBlochSphere = blochVs }

let TakeEvenPart (ftnVals: MPComplex[]) =
    Array.init ftnVals.Length
        (fun idx ->
            if idx = 0 then
                ftnVals.[0]
            else
                let s = ftnVals.[idx] + ftnVals.[ftnVals.Length - idx]
                s.Half
        )

let TakeOddPart (ftnVals: MPComplex[]) =
    Array.init ftnVals.Length
        (fun idx ->
            if idx = 0 then
                MPComplex.ZeroP ftnVals.[0].Precision
            else
                let s = ftnVals.[idx] - ftnVals.[ftnVals.Length - idx]
                s.Half
        )

let TakeParityParts (parity: int*int) (mpInputValues: MPComplex[]) =
    // Projector onto definite parity parts.

    let evenPart = TakeEvenPart mpInputValues
    let oddPart  = TakeOddPart  mpInputValues

    let realevenpart = evenPart |> Array.map (fun z -> z.Real)
    let realoddpart  = oddPart  |> Array.map (fun z -> z.Real)
    let imagevenpart = evenPart |> Array.map (fun z -> z.Imag)
    let imagoddpart  = oddPart  |> Array.map (fun z -> z.Imag)

    let realDefinite = if fst parity % 2 = 0 then realevenpart  else realoddpart
    let imagDefinite = if snd parity % 2 = 0 then imagevenpart  else imagoddpart

    Array.init mpInputValues.Length (fun idx -> { Real = realDefinite.[idx]; Imag = imagDefinite.[idx] })

let ConvertFunctionTableToRationalLaurentPolynomial (eps: float) (ftn: MPComplex[])  =
    // The input ftn is a list of function values at Fourier points on the complex unit circle
    // starting with the value at z = 1 going counterclockwise.
    // The length of ftn must be a power of 2.

    let truncate (keep: int) (parity: int) (extraMinus: bool) (reallist:MPF[]) =
        let evenQ = if parity % 2 = 0 then true else false
        for idx in 0..(reallist.Length - 1) do
            reallist.[idx] <- reallist.[idx] |> MPF.RoundDown (-keep)
        let (den, numList) = reallist |> MPFArrayToMPZArray
        let degree =
            let mutable pe = 0
            for idx = (if evenQ then 0 else 1) to ftn.Length / 2 - 1 do
                if not numList.[idx].IsZero then
                    pe <- idx
            pe
        let pos = Array.init (degree + 1) (fun idx -> if extraMinus then - numList.[idx] else numList.[idx] )
        let neg = Array.init degree (fun idx -> if evenQ then pos.[degree - idx] else - pos.[degree - idx])
        let coeff = Array.concat [ neg; pos ]
        let ilp = IntegralLaurentPolynomial(coeff, -degree)
        { Denominator = den; Numerator = ilp; Parity = if evenQ then 0 else 1 }

    let fou = MPFFT.FFT ftn // Fourier transform

    let evenpart  = TakeEvenPart fou
    let oddpart  = TakeOddPart fou

    let nBitsToKeep = 2.0 + Math.Log (float ftn.Length / eps, 2.0) |> int

    let evenReal = evenpart |> Array.map (fun z -> z.Real) |> truncate nBitsToKeep 0 false
    let oddImag  = oddpart  |> Array.map (fun z -> z.Real) |> truncate nBitsToKeep 1 true
    let evenImag = evenpart |> Array.map (fun z -> z.Imag) |> truncate nBitsToKeep 0 false
    let oddReal  = oddpart  |> Array.map (fun z -> z.Imag) |> truncate nBitsToKeep 1 false
    // Everything is a real-on-circle Laurent polynomial on its own.

    (evenReal, oddReal, evenImag, oddImag)

let MaxDifference (listA: MPComplex[]) (listB: MPComplex[]) =
    Array.map2 (fun (x: MPComplex) (y:MPComplex) -> (x - y).AbsSq.Float) listA listB
    |> Array.max
    |> Math.Sqrt

let measureDistanceInFunctionValues (testPrecision: uint32) (inVals: MPComplex[]) (qsp: QSPResult) =
    //This is ultimately desired, but it is too slow.
    let log2N = inVals.Length |> float |> (fun x -> Math.Log(x,2.0)) |> int
    let reconstructed = ReconstructFunctionValues testPrecision log2N qsp
    MaxDifference reconstructed inVals

let measureDistanceInFourierCoefficients (inVals: MPComplex[]) (qsp: QSPResult) =
    let outputCoef =
        ReconstructSignalPolynomial qsp
        |> Array.map MPMatrix2x2.HalfSumOfAllEntries

    let fou = inVals |> FFT
    let degree = qsp.PointsOnBlochSphere.Length / 2
    let pos = fou.[..degree]
    let neg = fou.[(fou.Length - degree)..]
    let inputCoef = Array.concat [ neg; pos ]

    MaxDifference inputCoef outputCoef

let BlochVectorOnEquatorToAngleFromXaxis (vec: MPVector) =
    let x = vec.X.Float
    let y = vec.Y.Float
    Math.Atan2(y, x)

let DiagonalUnitaryToAngle (mat: MPMatrix2x2) =
    let x = mat.D.Real.Float
    let y = mat.D.Imag.Float
    Math.Atan2(y, x) * 2.0
