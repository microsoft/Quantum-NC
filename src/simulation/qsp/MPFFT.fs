// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.


/// <summary>
///     Fourier transform on multiprecision complex vectors of length a power
///     of 2 by the Cooley-Tukey algorithm.
/// </summary>
module Microsoft.Quantum.Research.QuantumSignalProcessing.MPFFT

open MPNumber

let private ones32 (n:uint32) =
    let mutable x = n
    x <- x - ((x >>> 1) &&& 0x55555555u);
    x <- (((x >>> 2) &&& 0x33333333u) + (x &&& 0x33333333u));
    x <- (((x >>> 4) + x) &&& 0x0f0f0f0fu);
    x <- x + (x >>> 8);
    x <- x + (x >>> 16);
    int (x &&& 0x0000003fu)

let private ones64 (n:uint64) =
    ones32 (uint32 (n &&& 0xffffffffUL)) + ones32 (uint32 (n>>>32))

/// <summary>
///     Precomputes $\exp(2 \pi i / 2^b)$ of for b = 0,1,...,log2N that are
///     accurate to <c>precision</c> bits.
/// </summary>
type TwiddleFactor( log2N:int, precision:uint32 ) =
    let basisFactors =
        let factors = Array.zeroCreate (log2N + 1)
        factors.[0] <- MPComplex.Create(1.0, 0.0, precision) // index 0, 2^0 -> 1
        if log2N >= 1 then
            factors.[1] <- MPComplex.Create(-1.0, 0.0, precision) // index 1, 2^1 -> -1
        if log2N >= 2 then
            factors.[2] <- MPComplex.Create(0.0, -1.0, precision) // index 2, 2^2 -> sqrt{-1}
        for i in 3..log2N do
            factors.[i] <- MPComplex.Sqrt factors.[i-1]
        factors

    let nN = 1 <<< log2N

    let factorTable =
        let factors = Array.create (nN / 2) (MPComplex.Create(0.0, 0.0))
        let mutable fa = basisFactors.[0] // basisFacotrs.[0] = 1 + 0I
        let mutable nowidx = 0
        factors.[nowidx] <- fa
        for idx in 1..(nN / 2 - 1) do
            let grayUpdate = int (ones32 (uint32 ((idx - 1) ^^^ idx))) - 1 // the changing bit in the Gray code
            let next = nowidx ^^^ (1 <<< grayUpdate)
            if next > nowidx then
                fa <- fa * basisFactors.[log2N - grayUpdate]
            else
                fa <- fa * basisFactors.[log2N - grayUpdate].Conjugate
            nowidx <- next
            factors.[nowidx] <- fa
        factors

    /// <summary>
    /// Returns the number of bits of precision.
    /// </summary>
    member tf.Precision = basisFactors.[0].Precision

    /// <summary>
    /// Returns $\exp(- 2 \pi i / 2^b )$.
    /// </summary>
    member tf.Factor b = factorTable.[b]

    /// <summary>
    /// Returns $\exp(2 \pi i k / N )$.
    /// </summary>
    member tf.Compute (k:int32) =
        let kk =
            // The negative of kk is used, since the FFT here is the "inverse FFT" in the usual sense.
            if k > 0 then
                ((- k) % nN) + nN
            else
                (- k) % nN
        let mutable factor = basisFactors.[0]
        for i in 0..(log2N - 1) do
            if (1 &&& (kk >>> i)) = 1 then
                factor <- factor * basisFactors.[log2N - i]
        factor

let private Separate (a:MPComplex[]) =
    let half_n = a.Length / 2
    let b = Array.create half_n (MPComplex.ZeroP 32u)
    for i in 0..(half_n - 1) do
        b.[i] <- a.[2*i + 1]
        // The fact that this "set" routine keeps the precision of a
        // implies that the substitution is happening purely in F#
    for i in 0..(half_n - 1) do
        a.[i] <- a.[2*i]
    for i in 0..(half_n - 1) do
        a.[i + half_n] <- b.[i]

/// <summary>
///     Computes inverse Fourier transform given by the formula
///     $\sum_{j=0}^{N-1} \exp( - 2 \pi i j / N) vector.[j]$
/// </summary>
/// <param name="vector">
///     Signal complex vector. Must have a length 2^log2N where log2N is a
///     positive integer.
/// </param>
/// <param name="tf">
///     Precomputed trigonometric factors provided TwiddleFactor.
///     It must be instantiated as TwiddleFactor( log2N, precision ) where
///     2^log2N is the length of vector.
/// </param>
let FFTWithGivenTwiddleFactor (vector:MPComplex[]) (tf:TwiddleFactor) =
    let rec rfft (x:MPComplex[]) (s:int) =
        if x.Length <= 1 then
            x
        else
            let half_n = x.Length / 2
            Separate x // even and odd indices
            x.[..(half_n - 1)]   <- rfft x.[..(half_n - 1)] (s + 1)
            x.[half_n..]         <- rfft x.[half_n..]       (s + 1) // recursion; it is not tail recursive, but the tree is always balanced.
            for k in 0..(half_n - 1) do // butterfly
                let wo = (tf.Factor (k <<< s)) * x.[k + half_n]
                x.[k + half_n]  <- x.[k] - wo // Do not change the order here.
                x.[k]           <- x.[k] + wo
            x
    rfft (Array.copy vector) 0

/// <summary>
///     Modifies the given vector by dividing by the length of the vector.
///     Assumes that the length of the vector is a power of 2.
/// </summary>
let DivideByLength (vector:MPComplex[]) =
    let log2N = System.Math.Log (float vector.Length, 2.0) |> int
    vector |> Array.map (MPComplex.Multiply2Power -log2N)

/// <summary>
///     Returns the inverse Fourier transform given by
///     $(1/N) \sum_{j=0}^{N-1} \exp( - 2 \pi i j / N) vector.[j]$
///     The length N is assumed to be a power of 2.
///     The input is left unchanged.
/// </summary>
let FFT (x:MPComplex[]) =
    let xcopy = Array.copy x
    let log2N = int(System.Math.Log( float xcopy.Length, 2.0 ))
    let prec = xcopy.[0].Precision + uint32 log2N
    let tf = TwiddleFactor( log2N, prec )
    let fou = FFTWithGivenTwiddleFactor xcopy tf
    DivideByLength fou
