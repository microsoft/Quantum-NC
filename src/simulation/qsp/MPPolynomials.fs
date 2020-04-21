// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.

module Microsoft.Quantum.Research.QuantumSignalProcessing.MPPolynomials

open System
open System.Numerics
open MPNumber

/// <summary>
/// Class to represent univariate polynomials with real coefficients of high precision.
/// f(x) = \sum_k coefficients.[k] x^k
/// </summary>
type RealPolynomial( coefficients: MPF[] ) = // Exponent == index.

    let degree = coefficients.Length - 1

    let floatCoeff = coefficients |> Array.map (fun f -> f.Float)

    let HornerEvaluate (z:MPComplex) =
        let mutable aux = coefficients.[degree] |> curry MPF.ChangePrecision z.Precision |> MPComplex.CreateReal
        for idx = degree - 1 downto 0 do
            let anc = aux * z
            aux <- { anc with Real = anc.Real + coefficients.[idx] }
        aux

    let HornerEvaluateFloat (z: Complex) =
        if Complex.Abs z < 1.0 then
            let mutable aux  = Complex(floatCoeff.[degree], 0.0)
            for idx = degree - 1 downto 0 do
                assert( Math.Abs aux.Real < 1.0e300)
                assert( Math.Abs aux.Imaginary < 1.0e300)
                aux <- aux * z + Complex(floatCoeff.[idx], 0.0)
                assert(not(Double.IsNaN aux.Real))
            aux
        else
            let mutable aux  = Complex(floatCoeff.[0], 0.0)
            let zinv = Complex(1.0, 0.0) / z
            for idx = 1 to degree do
                assert( Math.Abs aux.Real < 1.0e300)
                assert( Math.Abs aux.Imaginary < 1.0e300)
                aux <- aux * zinv + Complex(floatCoeff.[idx], 0.0)
                assert(not(Double.IsNaN aux.Real))
                assert(not(Double.IsNaN aux.Imaginary))
            aux * (Complex.Pow (z, float degree))

    let EvaluateDerivativeFloat (z: Complex) =
        // This will give a false answer for Degree = - 1, i.e., coeff.Length = 0
        let mutable aux  = Complex(floatCoeff.[degree] * float degree, 0.0)
        for idx = degree - 1 downto 1 do
            aux <- aux * z + Complex(floatCoeff.[idx] * float idx, 0.0)
        aux

    let FastEvaluate (z: MPComplex) (unitModulus: bool) =
        // Adopted from Knuth, The Art of Computer Programming Vol.2. pg. 487.
        // Ref. BIT 5 (1965) 142
        // Ref. Goertzel, AMM 65 (1958) 34
        if degree = 0 then
            MPComplex.CreateReal coefficients.[0]
        else
            let zAbsSq = z.AbsSq
            let prec = z.Precision
            let mutable aaa     = coefficients.[degree    ] |> curry MPF.ChangePrecision prec
            let mutable bbb     = coefficients.[degree - 1] |> curry MPF.ChangePrecision prec
            let mutable oldaaa  = MPF.Zero
            let rrr = z.Real.TimesTwo // rrr = 2 Re(z)
            // The following loop implements the sequences
            // a_0 = f_n, b_0 = f_{n-1}
            // a_j = 2 Re(z) * a_{j-1} + b_{j-1}
            // b_j = f_{deg - j - 1} - |z|^2 a_{j-1}
            // f(z) will be a_{deg-1} z + b_{deg-1}
            for idx = 1 to degree - 1 do
                oldaaa <- aaa
                aaa <- rrr * oldaaa + bbb
                if not unitModulus then
                    bbb <- coefficients.[degree - idx - 1] - zAbsSq * oldaaa
                else
                    bbb <- coefficients.[degree - idx - 1] - oldaaa
            { Real = bbb + z.Real * aaa; Imag = z.Imag * aaa }

    /// <summary>
    /// Returns a tuple of the function value and its deriviative.
    /// </summary>
    /// <remark>
    /// Adopted from Knuth, The Art of Computer Programming Vol.2. pg. 489.
    /// Ref. Shaw and Traub, JACM 21, 161-167 (1974)
    /// </remark>
    let ShawTraubEvaluate =
        // preparing temporary registers to avoid repeated allocation. not initialized.
        let shawtraubT = float degree / 2.0 |> Math.Sqrt |> Math.Ceiling |> int

        // now input depedent part. Overall, there will be 2n + O(sqrt n) real multiplications.
        fun (z:MPComplex) ->
            let zPowers         = Array.init (max 2 (shawtraubT + 1)) (fun _ -> MPComplex.One)
            let zInversePowers  = Array.init (max 2 (shawtraubT + 1)) (fun _ -> MPComplex.One)

            // Powers of z, positive and negative exponents.
            zPowers.[1] <- z
            zInversePowers.[1] <- z.Inverse
            for idx = 2 to shawtraubT do
                zPowers.[idx] <- zPowers.[idx - 1] * z
                zInversePowers.[idx] <- zInversePowers.[idx - 1] * zInversePowers.[1]
            let zToThe2t = zPowers.[shawtraubT] * zPowers.[shawtraubT]

            let ff j = // The range (in the codomain) is {-t,-t+1,...,t-1,t}
                if j = degree then
                    shawtraubT
                else
                    shawtraubT - 1 - ( (degree - 1 - j) % (2*shawtraubT) )

            let getVV idx =
                if (ff idx) >= 0 then
                    zPowers.[ff idx] * coefficients.[idx]
                else
                    zInversePowers.[-(ff idx)] * coefficients.[idx]

            let aaa = [| z; z |]
            let bbb = [| z; z |] // initial garbage values

            aaa.[degree % 2] <- getVV degree
            bbb.[degree % 2] <- aaa.[degree % 2]

            for idx = degree - 1 downto 0 do
                aaa.[idx % 2] <- getVV idx
                if ((degree - 1 - idx) > 0) && ((degree - 1 - idx) % (2*shawtraubT)) = 0 then
                    aaa.[idx % 2] <- aaa.[idx % 2] + aaa.[(idx + 1) % 2] * zToThe2t
                    bbb.[idx % 2] <- aaa.[idx % 2] + bbb.[(idx + 1) % 2] * zToThe2t
                else
                    aaa.[idx % 2] <- aaa.[idx % 2] + aaa.[(idx + 1) % 2]
                    bbb.[idx % 2] <- aaa.[idx % 2] + bbb.[(idx + 1) % 2]

            let fz k =
                let res = if k = 0 then aaa.[0] else bbb.[1]
                if ff k >= 0 then
                    res * zInversePowers.[ff k]
                else
                    res * zPowers.[-(ff k)]

            (fz 0, fz 1)

    /// <summary>
    /// Instantiate an object of real polynomial using a given list of integer coefficients.
    /// </summary>
    new ( coeff: int[] ) =
        RealPolynomial( Array.map (fun (k:int) -> MPF.fromInt(k)) coeff )

    member __.Degree = degree
    member __.Coefficient deg = coefficients.[deg]
    member __.CoefficientsInFloat = coefficients |> Array.map (fun f -> f.Float)
    member __.MaxPrecisionOfCoefficients = coefficients |> Array.map (fun r -> r.Precision) |> Array.max

    member rp.Evaluate( z: MPComplex ) =
        FastEvaluate z false

    member rp.Evaluate (z: Complex) =
        HornerEvaluateFloat z

    /// <summary>
    /// Evaluate the polynomial with the assumption that the argument z has unit modulus.
    /// </summary>
    /// <param name="z"></param>
    member rp.EvaluateOnCircle( z: MPComplex ) =
        FastEvaluate z true

    /// <summary>
    /// Returns f(z) / f'(z).
    /// </summary>
    member rp.LogDerivativeInverse (z: MPComplex) =
        let (fz, fpz) = ShawTraubEvaluate z // 2n + O(sqrt n) real multiplications
        fz / fpz

    /// <summary>
    /// Returns f(z) / f'(z).
    /// </summary>
    member rp.LogDerivativeInverse (z: Complex) =
        let fz = HornerEvaluateFloat z
        let fpz= EvaluateDerivativeFloat z
        assert( fpz <> Complex(0.0,0.0) )
        assert(not(Double.IsNaN fz.Real))
        assert(not(Double.IsNaN fpz.Real))
        fz / fpz

    member rp.STEvaluate          (z: MPComplex) = ShawTraubEvaluate z |> fst
    member rp.STEvaluateDerivative(z: MPComplex) = ShawTraubEvaluate z |> snd

    member rp.EvaluateDerivativeSlow( z: MPComplex ) =
        // This will give a false answer for Degree = - 1, i.e., coeff.Length = 0
        // Do not use it as it is not the best way to compute the derivative.
        let mutable aux  =
            { (MPComplex.ZeroP z.Precision) with Real = coefficients.[rp.Degree] * rp.Degree }
        for idx = rp.Degree - 1 downto 1 do
            let anc = aux * z
            aux <- { anc with Real = anc.Real + coefficients.[idx] * idx }
        aux

    override rp.ToString() =
        let mutable str = coefficients.[0].ToString()
        for idx = 1 to rp.Degree do
            str <- str + " + " + coefficients.[idx].ToString() + "z" + idx.ToString()
        str

    /// <remark>
    /// This is derived from the observation that if a root is too small in magnitude,
    /// then the equation -a_0 = \sum_{k=1}^n a_k z^k would be absurd.
    /// </remark>
    member rp.LowerBoundOnRootMagnitude =
        let coef = rp.CoefficientsInFloat
        [| for idx in 1..(coef.Length - 1) ->
            0.5 * Math.Pow( coef.[0] / coef.[idx] |> Math.Abs, 1.0 / float idx)
        |]
        |> Array.min


/// <summary>
/// Returns some candidates for roots on the annulus of the complex plane specified by the inner and outer radii.
/// nRadii is the number of mesh points along the radial direction.
/// </summary>
/// <remark>
/// Lindsey-Fox method is a root finding algorithm tailored to polynomials whose roots are near the unit circle.
/// In our case the roots are not necessarily near the unit circle, but nonetheless the method is effective.
/// The key idea is that polynomial function values on equally space n points on any circle centered at the origin
/// can be computed in time O(n log n), rather than O(n^2), by Fourier transform of the coefficients.
/// This speedup makes it possible to investigate function values on a fine mesh of points.
/// We make the grid spacing inversely proportional to the degree of the polynomial.
/// For polynomials of degree of several thousand or higher, this method returns roots of additive error 10^{-3} or less,
/// which is often sufficient to guarantee convergence by Aberth iteration that is used in RootRefinery.
/// There is a polynomial evaluation precision issue by which a lot of roots can be misidentified.
/// To correct the issue, we examine the magnitude of the Newton iteration computed in high precision,
/// and reject a hypothetical root if the change is too large.
///
/// Inspired by J. P. Lindsey and J. W. Fox,
/// "A method of factoring long Z-transform polynomials,"
/// Computational Methods in Geoscience 33, 78 (1992)
/// </remark>
let ObtainSomeRootsByLindseyFoxMethod (innerRadius: float) (outerRadius: float) (nRadii: int)  (poly:RealPolynomial) =

    let radiusIncrement = (outerRadius - innerRadius) / (1.0 + float nRadii)
    let radius (k:int) = innerRadius + radiusIncrement * (float k) // radius 0 == innerCirc, radius (nRadii+1) == 1

    let log2N = Math.Ceiling(Math.Log( 2.0 * float (max 100 poly.Degree), 2.0 )) |> int

    let LindseyFoxEvaluateFloat =
        let ComplexFFT (v:Complex[]) = MathNet.Numerics.IntegralTransforms.Fourier.Forward v
        fun (r: float) ->
            let scale (coef: float[]) = Array.init (1<<<log2N) (fun idx ->
                    if idx <= poly.Degree then // poly.CoefficientArray.Length then
                        Complex(coef.[idx] * (Math.Pow (r, float idx)), 0.0)
                    else
                        Complex(0.0, 0.0)
                    )
            let rescaledPoly = // only with 50 most significant bits of the coefficients.
                poly.CoefficientsInFloat |> scale
            ComplexFFT rescaledPoly
            let ftnValues = rescaledPoly // Just renaming to note what is inside.
            ftnValues |> Array.map (fun z -> z.Magnitude)

    let LindseyFoxEvaluateMPF =
        // The result "ftnTable" is the same as the following line.
        // The complexity is only O(N log N), whereas the following line is O(N^2).
        // Array.init (1 <<< log2N) (fun idx -> poly.Evaluate (tf.Compute -idx))
        // This is more accurate than float version, but it is ~20x slower.
        let prec = poly.MaxPrecisionOfCoefficients + (uint32 log2N)
        let tf = MPFFT.TwiddleFactor( log2N, prec )
        fun (r: float) ->
            let rho = MPF.fromFloat(r)
            let rescaledPoly =
                Array.init (1<<<log2N) (fun idx ->
                    if idx <= poly.Degree then // < poly.CoefficientArray.Length then
                        let rhoPower = MPFPower (uint32 idx) rho
                        { (MPComplex.ZeroP prec) with Real = rhoPower * (poly.Coefficient idx) }
                    else
                        MPComplex.ZeroP prec
                    )
            let ftnTable = MPFFT.FFTWithGivenTwiddleFactor rescaledPoly tf
            printf "."
            ftnTable
            |> Array.map (fun z -> z.AbsSq.Float / (float (1<<<log2N)) |> Math.Sqrt)

    let EvaluateByFFT (r: float) = LindseyFoxEvaluateFloat r

    let magnitude = Array.create 3 [| |]
    magnitude.[0] <- EvaluateByFFT (radius 0)
    magnitude.[1] <- EvaluateByFFT (radius 1)
    let mutable approxRoots = []
    for k = 1 to nRadii do
        magnitude.[(k + 1) % 3] <- EvaluateByFFT (radius (k + 1))
        let inner = magnitude.[(k - 1) % 3]
        let center= magnitude.[k % 3]
        let outer = magnitude.[(k + 1) % 3]
        let n = center.Length
        for t = 0 to n - 1 do
            let cc = center.[t]
            assert(not (Double.IsNaN cc))
            // By maximum modulus principle,
            // a point is close to a root if its value is the least among nine points' near it.
            if  (cc < center.[(t + 1) % n]    )  &&
                (cc < center.[(t + n - 1) % n])  &&
                (cc < inner.[(t + 1) % n]     )  &&
                (cc < inner.[t]               )  &&
                (cc < inner.[(t + n - 1) % n] )  &&
                (cc < outer.[(t + 1) % n]     )  &&
                (cc < outer.[t]               )  &&
                (cc < outer.[(t + n - 1) % n] )
            then
                // cc is found to be a root.
                // Since it could be misidentified due to numerical error or grid spacing,
                // we test if it is legitimate by looking at the magnitude of the Newton method's correction.
                let theta = 2.0 * Math.PI * (float t) / (float n)
                let r = radius k
                let initPoint = MPComplex.Create(r * Math.Cos theta, r * Math.Sin theta)
                let newtonChange = poly.LogDerivativeInverse initPoint
                let changeSize = newtonChange.AbsSq.Float |> Math.Sqrt
                if changeSize * (float poly.Degree) < 2.0 then
                    approxRoots <- (initPoint - newtonChange) :: approxRoots
    approxRoots
    |> List.toArray


/// <summary>
/// Returns the eigenvalues of the companion matrix of a polynomial.
/// These are the roots of the polynomial.
/// </summary>
/// <remark>
/// This works well with excellent accuracy, especially when the coefficients fit in to float (64bit) variables,
/// but consumes a large (quadratic) amount of memory, and running time scales cubically.
/// For example, it consumes about 1GB of RAM for degree 2000.
/// </remark>
let ObtainRootsByCompanionMatrix (poly: RealPolynomial) =
    poly.CoefficientsInFloat
    |> MathNet.Numerics.FindRoots.Polynomial
    |> Array.filter (fun r -> r <> Complex(0.0,0.0) )
    |> Array.map MPComplex.Create


/// <summary>
/// A polynomial (without negative exponents) of real coefficients are loaded to this class.
/// An instance of the class stores roots of the input real polynomial,
/// and can refine the roots to higher precision.
/// The variables that stores the roots will occupy larger memory as the increased precision requires.
/// For effective use, initial roots should be set.
/// The default is a equally spaced points on the unit disk with arbitrary rotation,
/// which generally works okay but can be very slow.
/// Much better choices are the Lindsey-Fox method or Companion matrix method.
/// The root refinement is performed by Aberth iteration by default.
/// There are three iteration algorithms implemented: Aberth, Weiestrass, and Newton methods.
/// Aberth has the highest cost of evaluation per iteration,
/// but generally faster convergence compensates the evaluation cost.
/// Moreover, Aberth method appears to have much larger region of attraction,
/// which makes the algorithm attractive.
/// One can choose other algorithms by modifying one line in GetRoots member.
/// </summary>
type RootRefinery (poly: RealPolynomial) =

    let currentPrecision = Array.create poly.Degree 0u

    let approximateRoots = // initial values
        Array.init poly.Degree (fun idx ->
            let t = 0.0123456 + 2.0 * Math.PI * (float idx / float poly.Degree)
            MPComplex.Create(Math.Cos t, Math.Sin t)
            )

    let mutable nAllowedIterations = 8
    member rr.NAllowedIterations
        with get() = nAllowedIterations
        and set(v) = nAllowedIterations <- v

    member rr.NRoots = poly.Degree

    /// <summary>
    /// Sets the internal registry of roots as the input's first n elements where n is the degree or the length of the input.
    /// If n is less than the degree, the remaining registry is filled with equally spaced points on the unit circle.
    /// </summary>
    member rr.SetApproximateRoots (roots: MPComplex[]) =
        for idx = 0 to (min roots.Length poly.Degree) - 1 do
            approximateRoots.[idx] <- roots.[idx]
        let resid = poly.Degree - roots.Length
        for idx = 0 to resid - 1 do
            let t = 0.0123456 + 2.0 * Math.PI * (float idx / float resid)
            approximateRoots.[roots.Length + idx] <- MPComplex.Create(Math.Cos t, Math.Sin t)
        for idx = 0 to poly.Degree - 1 do
            currentPrecision.[idx] <- 0u

    static member AberthChangeFloat (rp:RealPolynomial) (approxRoots:Complex[]) (index: int) =
        let ratio = approxRoots.[index] |> rp.LogDerivativeInverse
        assert( not (Double.IsInfinity ratio.Real)  )
        assert( not (Double.IsNaN ratio.Real))
        let mutable force = Complex(0.0, 0.0)
        for partner = 0 to rp.Degree - 1 do
            if partner <> index then
                let dif = approxRoots.[index] - approxRoots.[partner]
                assert (not(Double.IsNaN dif.Real || Double.IsNaN dif.Imaginary ))
                force <- force + Complex(1.0, 0.0) /  dif

        let denom = ratio * force - Complex(1.0, 0.0)
        assert( denom <> Complex(0.0, 0.0))
        assert( not(Double.IsInfinity denom.Real))
        ratio / denom

    /// <summary>
    /// Returns adjustment to the current value of a root towards the true root.
    /// </summary>
    /// <param name="rp">A real polynomial.</param>
    /// <param name="approxRoots">Current approximations to the roots.</param>
    /// <param name="index">Index of the root for which Aberth change is computed.</param>
    /// <remark>
    /// Ehrlich, Comm. ACM 10 (2), 107-108 (1967)
    /// Aberth, Mathematics of Computation 27 (122), 339-344 (1973)
    /// </remark>
    static member AberthChange (rp:RealPolynomial) (approxRoots:MPComplex[]) (index: int) =
        let ratio = approxRoots.[index] |> rp.LogDerivativeInverse // 2n + O(n^0.5) real multiplications.
        let mutable force = MPComplex.Zero
        for partner = 0 to rp.Degree - 1 do
            if partner <> index then
                let pairForce = (approxRoots.[index] - approxRoots.[partner]).Inverse
                force <- force + pairForce

        ratio / (ratio * force - MPComplex.One)

    static member WeierstrassChange (rp: RealPolynomial) (approxRoots: MPComplex[]) (index: int) =
        let mutable denominator = MPComplex.CreateReal (rp.Coefficient rp.Degree)
        for pair = 0 to rp.Degree - 1 do
            if pair <> index then
                denominator <- denominator * (approxRoots.[index] - approxRoots.[pair])
        let ftn = approxRoots.[index] |> rp.Evaluate
        -ftn / denominator

    static member NewtonChange (rp: RealPolynomial) (approxRoots: MPComplex[]) (index: int) =
        let ratio = approxRoots.[index] |> rp.LogDerivativeInverse
        -ratio

    member rr.FindByFloatAberth () =
        // For polynomials of a large degree, this method may become completely useless due to overflow and underflow.
        let mutable counter = poly.Degree
        let bChange = Array.create poly.Degree true
        let apr = approximateRoots |> Array.map (fun z -> Complex(z.Real.Float, z.Imag.Float))
        while counter > 0 do
            let mutable worstErr = 1.0 / float (1 <<< 20)
            for idx = 0 to poly.Degree - 1 do
                if bChange.[idx] then
                    let deltaRoot = RootRefinery.AberthChangeFloat poly apr idx
                    apr.[idx] <- apr.[idx] + deltaRoot // Updating roots
                    let errmag = Complex.Abs deltaRoot // Error estimation

                    if errmag > worstErr then  worstErr <- errmag
                    if errmag * float poly.Degree < 1.0 then
                        bChange.[idx] <- false
                        counter <- counter - 1
            printfn "Float Aberth: %A to refine, log2(err)=%A" counter (Math.Log(worstErr, 2.0))
        apr |> Array.map MPComplex.Create

    member private rr.Refine (iterationAlgorithm: RealPolynomial -> MPComplex[] -> int -> MPComplex) (targetBits:uint32) =
        let mutable counter = 0
        let bChange = Array.create currentPrecision.Length false
        for idx = 0 to currentPrecision.Length - 1 do
            if currentPrecision.[idx] < targetBits then
                counter <- counter + 1 // the number of roots to refine.
                bChange.[idx] <- true
                approximateRoots.[idx] <- MPComplex.ChangePrecision (targetBits + 30u) approximateRoots.[idx]

        let mutable nRepeat = 0
        let mutable previousCounter = counter
        let rnd = System.Random(1234)
        while counter > 0 do

            #if DEBUG
            let mutable worstErr = -(int32 targetBits)
            // This is not necessary for the algorithm, but can be useful for debugging
            #endif

            for idx = 0 to approximateRoots.Length - 1 do
                if bChange.[idx] then
                    let deltaRoot = iterationAlgorithm poly approximateRoots idx
                    approximateRoots.[idx] <- approximateRoots.[idx] + deltaRoot // Updating roots
                    let logmag = deltaRoot.RoughLog2Modulus // Error estimation

                    #if DEBUG
                    if logmag > worstErr then  worstErr <- logmag
                    #endif

                    if logmag < (-(int32 targetBits)) / 2 then
                        // The exit condition here assumes that the iteration increases
                        // the number of correct significant bits by a factor of at least 2.
                        // The Aberth iteration has a cubic convergence at nondegenerate roots.
                        bChange.[idx] <- false
                        counter <- counter - 1

            #if DEBUG
            printfn "%A to refine, log2(err)=%A" counter worstErr
            #endif

            if previousCounter > counter then
                previousCounter <- counter
                nRepeat <- 0
            else
                nRepeat <- nRepeat + 1

            if nRepeat > nAllowedIterations then
                #if DEBUG
                printfn "Resetting roots"
                #endif
                for idx = 0 to approximateRoots.Length - 1 do
                    if bChange.[idx] then
                        let t =  rnd.NextDouble() * 2.0 * Math.PI //0.0123456 + 2.0 * Math.PI * (float idx / float counter)
                        approximateRoots.[idx] <- MPComplex.Create(Math.Cos t, Math.Sin t) // resets the approximate roots.
                nRepeat <- 0
                nAllowedIterations <- nAllowedIterations + 5 // become a little more lenient.

        for idx = 0 to currentPrecision.Length - 1 do
            currentPrecision.[idx] <- targetBits

    member rr.CurrentPrecision = Array.min currentPrecision

    /// <summary>
    /// Computes roots to a given absolute/additive accuracy, and internally stores the result.
    /// Nothing happens if the cached roots are more accurate than the given precision.
    /// Computed roots is accessed by the member Roots.
    /// </summary>
    /// <param name="precision"></param>
    member rr.ComputeRoots (precision: uint32) =
        if precision > Array.min currentPrecision then
            rr.Refine RootRefinery.AberthChange precision

    member rr.Roots = approximateRoots

    member rr.RootsInsideUnitDisk =
        rr.Roots
        |> Array.filter (fun z -> z.InsideUnitCircle)

/// <summary>
/// Returns roots of poly to targetPrecision bits of precision.
/// This employs companion matrix method; the degree of the polynomial should be less than a few thousand.
/// </summary>
/// <param name="poly"></param>
/// <param name="targetPrecision"></param>
let FindRoots (poly:RealPolynomial) (targetPrecision:uint32) =
    let refinery = RootRefinery(poly)
    poly |> ObtainRootsByCompanionMatrix |> refinery.SetApproximateRoots
    refinery.ComputeRoots targetPrecision
    refinery.Roots

/// <summary>
/// Represents a univariate Laurent polynomial of integer coefficients.
/// The basic constructor is provided with list of coefficients starting with that of the least exponent,
/// and the least exponent. The greatest exponent is inferred from the least exponent
/// and the length of the list of the coefficients.
/// </summary>
type IntegralLaurentPolynomial( coefficients: MPZ[], leastExponent: int ) =

    // Cache
    member private lp.BitLength =
        let bitLength = coefficients |> Array.map (fun (k:MPZ) -> k.BitLength) |> Array.max
        fun () -> bitLength

    // Cache
    member private lp.GetRealPolynomial =
        let realPoly =
            coefficients
            |> Array.map (fun (j:MPZ) -> j.ToMPF |> MPF.Multiply2Power (-lp.BitLength()) )
                  // so that every coefficient in the output real polynomial has magnitude at most 1.
            |> RealPolynomial // The least exponent is forgotten.
        fun () -> realPoly

    /// <summary>
    /// Returns a real polynomial whose coefficient list is the same as the current Laurent polynomials.
    /// The constant term of the returned real polynomial is the coefficient of the least exponent.
    /// </summary>
    member lp.ToRealPolynomial = lp.GetRealPolynomial()

    /// <summary>
    /// Instantiate the zero polynomial.
    /// </summary>
    static member Zero = IntegralLaurentPolynomial( [| MPZ.Zero |], 0 )

    member lp.LeastExponent     = leastExponent
    member lp.GreatestExponent  = leastExponent + coefficients.Length - 1
    member private lp.Coefficients = coefficients

    /// <summary>
    /// The degree is the largest absolute value of the exponent of any nonzero term.
    /// </summary>
    member lp.Degree = max lp.GreatestExponent (- leastExponent)

    member lp.CoefficientSum = Array.sum coefficients

    member lp.Evaluate (z: MPComplex) =
        let den =
            if leastExponent < 0 then
                MPComplex.Power (uint32 (- leastExponent)) z
            else
                MPComplex.Power (uint32 leastExponent) z
                |> MPComplex.ComputeInverse
        // realPoly has scaled coefficients. See the definition of realPoly.
        // This rescaling removes the artificial scaling factor.
        let num = lp.ToRealPolynomial.Evaluate z
        num / (MPComplex.Multiply2Power (lp.BitLength()) den)

    /// <summary>
    /// Returns the coefficient of the term x^deg, which may be zero.
    /// Here, deg is an arbitrary integer.
    /// </summary>
    member lp.Coefficient (deg: int) =
        if (deg < leastExponent) || (deg > lp.GreatestExponent) then
            MPZ.Zero
        else
            coefficients.[deg - leastExponent]

    static member (+) (a: IntegralLaurentPolynomial, b: IntegralLaurentPolynomial) =
        let lex = min a.LeastExponent       b.LeastExponent
        let gex = max a.GreatestExponent    b.GreatestExponent
        let coe = Array.init (gex - lex + 1) (fun jj ->
                        let deg = jj + lex
                        (a.Coefficient deg) + (b.Coefficient deg)
                    )
        IntegralLaurentPolynomial(coe,lex)

    static member (-) (a: IntegralLaurentPolynomial, b: IntegralLaurentPolynomial) =
        let lex = min a.LeastExponent       b.LeastExponent
        let gex = max a.GreatestExponent    b.GreatestExponent
        let coe = Array.init (gex - lex + 1) (fun jj ->
                        let deg = jj + lex
                        (a.Coefficient deg) - (b.Coefficient deg)
                    )
        IntegralLaurentPolynomial(coe,lex)

    static member Square (poly: IntegralLaurentPolynomial) =
        let lex    = 2 * poly.LeastExponent
        let greatest = 2 * poly.GreatestExponent
        let ans = [| for i in 0..(greatest - lex) -> new MPZ(0) |]
        // TODO: The outer loop may be parallelized.
        for deg = lex to greatest do
            for subdeg = max poly.LeastExponent (deg - poly.GreatestExponent)
              to min poly.GreatestExponent (deg - poly.LeastExponent)
              do
                ans.[deg - lex] <- ans.[deg - lex] + (poly.Coefficient subdeg) * (poly.Coefficient (deg - subdeg))
        IntegralLaurentPolynomial(ans,lex)

    static member (*) (poly: IntegralLaurentPolynomial, scalar: MPZ) =
        if scalar.IsZero then IntegralLaurentPolynomial.Zero
        else IntegralLaurentPolynomial(poly.Coefficients |> Array.map (fun x -> x * scalar), poly.LeastExponent)

    override lp.ToString() =
        let mutable str = ""
        for deg in leastExponent .. (lp.GreatestExponent - 1) do
            if not (lp.Coefficient deg).IsZero then
                str <- str + (lp.Coefficient deg).ToString() + "z^" + deg.ToString() + " + "
        str + (lp.Coefficient lp.GreatestExponent).ToString() + "z^" + lp.GreatestExponent.ToString()

    member lp.Roots(precision: uint32) =
        FindRoots lp.ToRealPolynomial precision

    /// <summary>
    /// Tests whether the current Laurent polynomial has only even exponents.
    /// If not, it returns itself with a boolean flag that says it has some nonzero term of odd exponents.
    /// If true, it returns a Laurent polynomial of half the current degree,
    /// by discarding zero terms of odd degree.
    /// For example, if the current polynomial is z^{-2} + 1 + z^4, then the result is z^{-1} + 1 + z^2.
    /// </summary>
    member lp.ToPolyOfSquaredVariable() =
        let everyExponentIsEven =
            let mutable maybe = true
            let mutable idx = leastExponent
            while maybe && (idx <= lp.GreatestExponent) do
                if (not (lp.Coefficient idx).IsZero) && (idx % 2 <> 0) then
                    maybe <- false
                idx <- idx + 1
            maybe

        if everyExponentIsEven then
            #if DEBUG
            printfn "yes, every exponent is even"
            #endif
            // throwing away zero coefficients and return a polynomial g such that g(z^2) = f(z).
            let newCoef = Array.init (coefficients.Length / 2 + 1) (fun idx -> coefficients.[2 * idx])
            (everyExponentIsEven, IntegralLaurentPolynomial(newCoef, leastExponent / 2) )
        else
            (everyExponentIsEven, lp)

    member lp.IsReciprocal =
        if -leastExponent <> lp.GreatestExponent then
            false
        else
            let mutable maybeReciprocal = true
            for idx = 1 to lp.GreatestExponent do
                if lp.Coefficient idx <> lp.Coefficient (-idx) then
                    maybeReciprocal <- false
            maybeReciprocal


/// <summary>
/// Represents a Laruent polynomial with rational coefficients
/// which is either reciprocal (partiy even, f(z) = f(1/z)) or anti-reciprocal (parity odd, f(z) = - f(1/z)).
/// The input "parity" is redundant but included.
/// The behavior is undefined if the polynomials is not of definite parity.
/// </summary>
type RationalPureLaurentPolynomial =
    {
        Denominator: MPZ;
        Numerator: IntegralLaurentPolynomial;
        Parity: int
    }

    member p.Degree = p.Numerator.Degree

    member p.Evaluate =
        let denomReal = MPF.fromBigInteger(p.Denominator)
        fun (z: MPComplex) ->
            let num = p.Numerator.Evaluate z
            let ans = num / denomReal
            if p.Parity % 2 = 0 then
                ans
            else // multiply by sqrt{-1} so that it is real-on-circle.
                { Real = -ans.Imag; Imag = ans.Real }

    member p.ConvertToComplexCoefficientList (precision: uint32) (targetDegree:int)=
        let denInv =
            if p.Parity % 2 = 0 then
                MPComplex.CreateReal (MPF.fromBigInteger  p.Denominator |> curry MPF.ChangePrecision precision)
            else
                MPComplex.CreateImag (MPF.fromBigInteger -p.Denominator |> curry MPF.ChangePrecision precision)
            |> MPComplex.ComputeInverse
        let num =
            Array.init (2 * targetDegree + 1) (fun idx ->
                let d = idx - targetDegree
                let z = p.Numerator.Coefficient d
                MPComplex.CreateReal (MPF.fromBigInteger z  |> curry MPF.ChangePrecision precision)
            )
        Array.map (fun z -> z * denInv) num
