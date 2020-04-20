// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.

module Microsoft.Quantum.Research.QuantumSignalProcessing.MPNumber

open System
open System.Numerics

let private powerOf2 (n: uint32) =
    let quo = int(n / 8u)
    let rem = int(n % 8u) // this is at most 7.
    let ba = Array.create (if rem < 7 then quo + 1 else quo + 2) 0uy
    ba.[quo] <- ( 1uy <<< rem )
    BigInteger(ba)

let private MaxPrecision = UInt32.MaxValue / 4u
let private MinPrecision = 64u

let uncurry (f : 'x -> 'y -> 'r) =
    (fun (x, y) -> f x y)

let curry (f : ('x * 'y) -> 'r) =
    (fun x y -> f (x, y))

type BigFloat = // The number is Digits * 2^{Exponent}.
    {
        Digits      : BigInteger
        Exponent    : int32
        Precision   : uint32
    }

    static member private Construct (digits, exponent, precision) =
        { Digits = digits; Exponent = exponent; Precision = precision }

    /// <summary>
    /// Represents 0 with Precision of 64 bits.
    /// </summary>
    static member Zero  = { Digits = 0I; Exponent = 0; Precision = MinPrecision }

    /// <summary>
    /// Represents 0 with given precision.
    /// </summary>
    static member ZeroP (precision: uint32) = { Digits = 0I; Exponent = 0; Precision = precision }

    /// <summary>
    /// Represents 1 with Precision of 64 bits.
    /// </summary>
    static member One   = { Digits = 1I; Exponent = 0; Precision = MinPrecision }

    member bf.IsZero    = bf.Digits.IsZero
    member bf.Sign      = bf.Digits.Sign

    member bf.RoughLog2Modulus =
        if bf.Digits.IsZero then
            Int32.MinValue
        else
            int32(BigInteger.Log(BigInteger.Abs bf.Digits, 2.0)) + bf.Exponent

    member internal bf.FloatWithSeparateExponent =
        let arr = bf.Digits.ToByteArray()
        let headbytes = if arr.Length > 8 then arr.[(arr.Length - 8)..] else arr // collecting 64bits of most significant bit.
        let change = 8 * (arr.Length - headbytes.Length) |> int32
        let exp = bf.Exponent + change
        if exp % 2 = 0 then
            (float(BigInteger(headbytes)), exp)
        else
            (float(2I * BigInteger(headbytes)), exp - 1)

    member bf.AsString =
        lazy
            if bf.Digits.IsZero then
                "+.0e0"
            else
                let abs = BigInteger.Abs bf.Digits
                let flo, ex = bf.FloatWithSeparateExponent
                let computeExponent (f : float) =
                    Math.Log10(Math.Abs f) + (float ex) * Math.Log10(2.0) |> Math.Ceiling |> int
                let roughOrder = computeExponent flo
                let nDigits = max roughOrder (BigInteger.Log10(abs) |> int)
                let digits =
                    if bf.Exponent > 0 then
                        abs * powerOf2(uint32 bf.Exponent) / BigInteger.Pow(10I, roughOrder - nDigits)
                    else
                        abs * BigInteger.Pow(10I, nDigits - roughOrder) / powerOf2 (uint32 -bf.Exponent)
                let sign = if bf.Digits.Sign >= 0 then "+" else "-"
                let dig = digits.ToString()
                let order =
                    if digits.ToString().Chars 0 = '1' then // avoiding subtle conversion
                        computeExponent (2. * flo)
                    else roughOrder
                sprintf "%s.%se%O" sign (dig.Substring(0, min nDigits dig.Length)) order

    override bf.ToString() =
        bf.AsString.Value

    /// <summary>
    /// Controls the precision. Precision field of bf is ignored.
    /// </summary>
    static member ChangePrecision (targetPrecision: uint32, bf: BigFloat) =
        let nBytesTarget = float targetPrecision / 8.0 |> Math.Ceiling |> int
        let ba = bf.Digits.ToByteArray()
        let nBytesNow = ba.Length
        if nBytesNow > nBytesTarget then
            {
                Digits      = BigInteger(ba.[(nBytesNow - nBytesTarget)..])
                Exponent    = bf.Exponent + 8 * (nBytesNow - nBytesTarget)
                Precision   = nBytesTarget * 8 |> uint32
            }
        else
            { bf with Precision = nBytesTarget * 8 |> uint32 }

    // Converters
    static member fromFloat(x : float, ?precision : uint32) =
        let fromFloat' x =
            let bitsInFloat = 53
            let expo =
                if x = 0.0 then
                    0.0
                else
                    Math.Log(Math.Abs x, 2.0)
                |> int
            let digits =
                x * Math.Pow(2.0, float (bitsInFloat - expo))
                |> BigInteger
            { Digits = digits; Exponent = expo - bitsInFloat; Precision = MinPrecision }

        match precision with
        | None -> fromFloat' x
        | Some prec -> fromFloat' x |> curry BigFloat.ChangePrecision prec

    static member fromInt (x: int) = BigFloat.fromFloat(float x)

    static member fromBigInteger (x: BigInteger) =
        let nBytes = x.ToByteArray().Length
        { Digits = x; Exponent = 0; Precision = 8 * nBytes |> uint32 }

    member bf.Float =
        let f, ex = bf.FloatWithSeparateExponent
        f * Math.Pow(2.0, float ex)

    // Operators
    static member (+) (x: BigFloat, y: BigFloat) =
        if x.IsZero then y
        elif y.IsZero then x
        else
            let prec = max x.Precision y.Precision |> int
            let dig, expo =
                let xdigits, xexp = x.Digits, x.Exponent
                let ydigits, yexp = y.Digits, y.Exponent
                if (yexp > 64 + xexp + prec) then
                    (ydigits, yexp)
                elif (xexp > 64 + yexp + prec) then
                    (xdigits, xexp)
                else
                    if xexp > yexp then
                        let shift = powerOf2 (xexp - yexp |> uint32)
                        ((xdigits * shift) + ydigits, yexp)
                    elif xexp < yexp then
                        let shift = powerOf2 (yexp - xexp |> uint32)
                        (xdigits + (ydigits * shift), xexp)
                    else
                        (xdigits + ydigits, xexp)
            { Digits = dig; Exponent = expo ; Precision = MaxPrecision } // vacuously high precision
            |> curry BigFloat.ChangePrecision (uint32 prec)

    static member (-) (x: BigFloat, y: BigFloat) =
        let prec = max x.Precision y.Precision |> int
        if x.IsZero then { y with Digits = -y.Digits; Precision = uint32 prec }
        elif y.IsZero then { x with Precision = uint32 prec }
        else
            let dig, expo =
                let xdigits, xexp = x.Digits, x.Exponent
                let ydigits, yexp = y.Digits, y.Exponent
                if (yexp > 64 + xexp + prec) then
                    (-ydigits, yexp)
                elif (xexp > 64 + yexp + prec) then
                    (xdigits, xexp)
                else
                    if xexp > yexp then
                        let shift = powerOf2 (xexp - yexp |> uint32)
                        ((xdigits * shift) - ydigits, yexp)
                    elif xexp < yexp then
                        let shift = powerOf2 (yexp - xexp |> uint32)
                        (xdigits - (ydigits * shift), xexp)
                    else
                        (xdigits - ydigits, xexp)
            { Digits = dig; Exponent = expo; Precision = MaxPrecision } // vacuously high precision
            |> curry BigFloat.ChangePrecision (uint32 prec)

    static member (~-) (bf: BigFloat) = { bf with Digits = - bf.Digits }

    static member (*) (x: BigFloat, y: BigFloat) =
        let prec = max x.Precision y.Precision
        if x.Digits.IsZero || y.Digits.IsZero then BigFloat.ZeroP prec
        else
            {
                Digits = x.Digits * y.Digits
                Exponent = x.Exponent + y.Exponent
                Precision = MaxPrecision
            } // vacuously high precision
            |> curry BigFloat.ChangePrecision prec

    static member (*) (x: BigFloat, y:int) =
        if x.IsZero || y = 0 then BigFloat.ZeroP x.Precision
        else
            { Digits = x.Digits * BigInteger(y); Exponent = x.Exponent; Precision = MaxPrecision }
            // vacuously high precision
            |> curry BigFloat.ChangePrecision x.Precision

    static member inline (*) (x: int, y: BigFloat) = y * x
    static member inline (*) (x: BigFloat, y: float) = x * BigFloat.fromFloat(y)
    static member inline (*) (y: float, x: BigFloat) = x * BigFloat.fromFloat(y)

    member inline bf.Inverse = bf.ComputeInverse bf.Precision

    /// <summary>
    /// Inverse of itself to the given bits of precision
    /// </summary>
    member bf.ComputeInverse (precision: uint32) =
        let init =
            let f, expo = bf.FloatWithSeparateExponent
            let finv = BigFloat.fromFloat(1.0 / f)
            { Digits = finv.Digits; Exponent = finv.Exponent - expo; Precision = MaxPrecision }
            |> curry BigFloat.ChangePrecision precision
        let two = BigFloat.fromFloat(2.0, precision)
        let next now _ = now * (two - bf * now)
        let nIter = Math.Log(float precision, 2.0) - 5.0 |> Math.Ceiling |> int
        List.fold next init [1..nIter]

    static member (/) (x: BigFloat, y: BigFloat) =
        x * y.ComputeInverse (max x.Precision y.Precision)

    /// <summary>
    /// Multiplies by a power of 2.
    /// </summary>
    /// <param name="expo">The exponent of 2 to be multiplied. It may be either positive or negative.</param>
    static member inline Multiply2Power (expo: int) (bf: BigFloat) =
        if  bf.Digits.IsZero then { bf with Exponent = 0 }
        else
            { bf with Exponent = bf.Exponent + expo }

    member bf.Half =
        if bf.IsZero then bf
        else { bf with Exponent = bf.Exponent - 1}

    member bf.TimesTwo =
        if bf.IsZero then bf
        else { bf with Exponent = bf.Exponent + 1}


    /// <summary>
    /// Truncates its fractional part.
    /// </summary>
    static member Truncate (bf: BigFloat) =
        if bf.Exponent >= 0 then bf
        else { Digits = bf.Digits / powerOf2 (uint32 -bf.Exponent); Exponent = 0; Precision = bf.Precision }

    static member RoundDown (place: int) (bf: BigFloat) =
        let trc = { bf with Exponent = bf.Exponent - place } |> BigFloat.Truncate
        { trc with Exponent = trc.Exponent + place }

    static member Sqrt(x: BigFloat) =
        if x.Digits.IsZero then x
        else
            let f, expo = x.FloatWithSeparateExponent
            let shiftedX = { x with Exponent = x.Exponent - expo }
            let init = BigFloat.fromFloat(Math.Sqrt(f), x.Precision) // initial approximation.
            let nIter = Math.Log(x.Precision |> float, 2.0) |> int
            let next now _ = (now + shiftedX / now).Half
            List.fold next init [1..nIter]
            |> BigFloat.Multiply2Power (expo/2)

type MPZ = BigInteger
type MPF = BigFloat

let LeastCommonMultiple (x:MPZ[]) =
    let lcm (a:MPZ) (b:MPZ) =
        let gcd = MPZ.GreatestCommonDivisor(a,b)
        a * b / gcd
    let signedlcm = Array.fold lcm 1I x
    MPZ.Abs signedlcm

type private MPQ =
    {   Denominator: MPZ
        Numerator: MPZ  }

/// <summary>
/// Each floating point number is converted to a rational number with a common denominator,
/// and then the tuple of the common denominator and the numerator array is returned.
/// </summary>
let MPFArrayToMPZArray (input:MPF[]) =
    let RationalFromMPF (x:MPF) =
        if x.Exponent < 0 then
            let denom = MPZ.Pow(2I, int -x.Exponent)
            let num = x.Digits
            let gcd = MPZ.GreatestCommonDivisor (denom, num)
            {Denominator = denom / gcd; Numerator = num / gcd}
        else
            {Denominator = 1I; Numerator = x.Digits * BigInteger.Pow(2I, int x.Exponent)}
    let rational = Array.map RationalFromMPF input
    let den = rational
            |> Array.map (fun (q:MPQ) -> q.Denominator)
            |> LeastCommonMultiple
    let numlist = rational |> Array.map (fun (q:MPQ) -> q.Numerator * den / q.Denominator)
    (den, numlist)

/// <summary>
/// Integral power of the input x.
/// </summary>
let MPFPower (exponent: uint32) (x: MPF) =
    let mutable ans =
        if (exponent &&& 1u) = 1u then x
        else MPF.fromFloat(1.0, x.Precision)
    if exponent > 0u then
        let mutable zPower = x
        for idx = 1 to Math.Log(float exponent, 2.0) |> int do
            zPower <- zPower * zPower
            if ((exponent >>> idx) &&& 1u) = 1u then
                ans <- ans * zPower
    ans

type BigInteger with
    member z.Float      = MPF.fromBigInteger(z).Float
    member z.ToMPF      = MPF.fromBigInteger(z)
    member z.BitLength  = 8 * z.ToByteArray().Length

/// <summary>
/// Computes sin(x) using the naive Taylor series.
/// Not the best, but simple.
/// </summary>
let private MPSinByTaylor (x: MPF) =
    if x.IsZero then MPF.ZeroP x.Precision
    else
        let xsq = -x * x
        let mutable ans     = x
        let mutable factor  = x
        let mutable k = 1
        let finalPrec = - (int x.Precision) + x.RoughLog2Modulus
        while factor.RoughLog2Modulus > finalPrec do
            factor  <- factor * xsq / MPF.fromInt( (2 * k + 1) * 2 * k)
            ans     <- ans + factor
            k       <- k + 1
        ans

/// <summary>
/// Computes cos(x) using the naive Taylor series.
/// Not the best, but simple.
/// </summary>
let private MPCosByTaylor (x: MPF) =
    let xsq = -x * x
    let mutable ans     = MPF.fromInt(1)
    let mutable factor  = MPF.fromInt(1)
    let mutable k       = 1
    let finalPrec = - (int x.Precision) + x.RoughLog2Modulus
    while factor.RoughLog2Modulus > finalPrec do
        factor <- factor * xsq / MPF.fromInt( (2 * k - 1) * 2 * k)
        ans <- ans + factor
        k <- k + 1
    ans

/// <summary>
/// Returns pi=3.1415... to a given number of bits of precision.
/// </summary>
let MPPi =
    let mutable internalPi = MPF.fromFloat(Math.PI) // cache
    fun (precision: uint32) ->
        let oldPrec = internalPi.Precision
        if oldPrec = precision then
            internalPi
        elif oldPrec > precision then
            internalPi |> curry BigFloat.ChangePrecision precision
        else
        // The algorithm is Newton's iteration towards to root of f(x) = tan x - 1.
        // The correction term is f(x)/f'(x) =  (sin(x) - cos(x))cos(x).
        // I'm sure there are better methods, but this part is used only when
        // one has to compute input values using some trigonometric functions, and hence is not repeated many times.

            // increase the number of bits.
            // starting value of the iteration is close to pi/4.
            let initialPiOver4 =
                internalPi
                |> curry BigFloat.ChangePrecision precision
                |> BigFloat.Multiply2Power -2

            // The number of iterations; it is a bit overkill but safe.
            let nIter = Math.Log(float (precision - oldPrec), 2.0 ) |> int

            let NewtonIterate approxPi4 (_: int) =
                let sin = MPSinByTaylor approxPi4
                let cos = MPCosByTaylor approxPi4
                approxPi4 - (sin - cos) * cos

            internalPi <-
                List.fold NewtonIterate initialPiOver4 [0..nIter]
                |> BigFloat.Multiply2Power 2

            internalPi

let private trigNormalize (x: MPF) =
    let n = x.Float / Math.PI |> int
    if Math.Abs n = 0 then
        (x, +1)
    else
        let mpi = MPPi x.Precision
        let xx = x - mpi * MPF.fromInt(n)
        if n % 2 <> 0 then
            (xx, -1)
        else
            (xx, +1)

let MPSin (x: MPF) =
    let xx, sign = trigNormalize x
    let ans = MPSinByTaylor xx
    if sign < 0 then -ans else ans

let MPCos (x: MPF) =
    let xx, sign = trigNormalize x
    let ans = MPCosByTaylor xx
    if sign < 0 then -ans else ans

let private _MPExpm1 (x: MPF) =
    if x.IsZero then MPF.ZeroP x.Precision
    else
        let mutable ans     = x
        let mutable factor  = x // If these initial values were 1 then it computes exp(x), rather than exp(x)-1.
        let mutable k = 2
        let finalPrec = - (int x.Precision) + x.RoughLog2Modulus
        while factor.RoughLog2Modulus > finalPrec do
            factor <- factor * x / MPF.fromInt(k)
            ans <- ans + factor
            k <- k + 1
        ans

let MPExp =
    let oneRough = MPF.fromFloat(1.0, 64u)
    let mutable mathE = _MPExpm1 oneRough + oneRough
    fun (x:MPF) ->
        let one = MPF.fromFloat(1.0, x.Precision)
        let r = x.Float |> int32
        let majPart, xx =
            if r = 0 then
                one, x
            else
                let exp1 =
                    if x.Precision > mathE.Precision then
                        mathE <- _MPExpm1 one + one
                        mathE
                    else
                        mathE |> curry BigFloat.ChangePrecision x.Precision
                if r < 0 then // x is negative
                    MPFPower (uint32 (-r)) exp1.Inverse, x + MPF.fromInt(-r)
                else // r > 0
                    MPFPower (uint32 r) exp1, x - MPF.fromInt(r)
        let remainder = one + _MPExpm1 xx
        majPart * remainder

type MPComplex =
    {
        Real: MPF
        Imag: MPF
    }

    static member Zero = { Real = MPF.Zero; Imag = MPF.Zero }
    static member One  = { Real = MPF.One ; Imag = MPF.Zero }

    static member ZeroP (precision: uint32) = { Real = MPF.ZeroP precision; Imag = MPF.ZeroP precision }

    static member Create(x: float, y: float) =
        { Real = MPF.fromFloat(x); Imag = MPF.fromFloat(y) }
    static member Create(x: float, y: float, prec: uint32) =
        { Real = MPF.fromFloat(x, prec); Imag = MPF.fromFloat(y, prec) }
    static member Create(z: Complex) =
        MPComplex.Create(z.Real, z.Imaginary)

    static member CreateReal(r: MPF) = { Real = r; Imag = MPF.ZeroP r.Precision }
    static member CreateImag(r: MPF) = { Real = MPF.ZeroP r.Precision; Imag = r }

    member inline z.Precision          = min z.Real.Precision z.Imag.Precision
    member inline z.Conjugate          = { Real = z.Real; Imag = - z.Imag }
    member inline z.AbsSq              = z.Real * z.Real + z.Imag * z.Imag
    member inline z.Modulus            = z.AbsSq |> BigFloat.Sqrt
    member inline z.InsideUnitCircle   = (z.AbsSq - MPF.fromFloat(1.0)).Sign < 0

    static member inline (+) (a: MPComplex, b: MPComplex)  = { Real = a.Real + b.Real; Imag = a.Imag + b.Imag  }
    static member inline (+) (a: MPComplex, b: MPF)        = { a with Real = a.Real + b }
    static member inline (-) (a: MPComplex, b: MPComplex)  = { Real = a.Real - b.Real; Imag = a.Imag - b.Imag  }
    static member inline (~-)(a: MPComplex)                = { Real = -a.Real        ; Imag = -a.Imag          }
    static member inline (-) (a: MPComplex, b: MPF)        = { a with Real = a.Real - b }

    static member inline (*) (a: MPComplex, b: MPComplex) =
        {
            Real = a.Real * b.Real - a.Imag * b.Imag
            Imag = a.Real * b.Imag + a.Imag * b.Real
        }

    static member inline (*) (a: MPComplex, b: MPF) = { Real = a.Real * b; Imag = a.Imag * b }
    static member inline (*) (a: MPF, b: MPComplex) = b * a

    static member (/) (numerator: MPComplex, denominator: MPComplex) =
        let denomAbsSq = denominator.AbsSq
        let re = numerator.Real * denominator.Real + numerator.Imag * denominator.Imag
        let im = numerator.Imag * denominator.Real - numerator.Real * denominator.Imag
        { Real = re / denomAbsSq; Imag = im / denomAbsSq }

    static member (/) (numerator: MPComplex, denominator: MPF) =
        { Real = numerator.Real / denominator; Imag = numerator.Imag / denominator }

    /// <summary>
    /// Returns the larger of the exponent (base 2) of real and imaginary parts in their floating point representation.
    /// </summary>
    member z.RoughLog2Modulus =
        max (z.Real.RoughLog2Modulus) (z.Imag.RoughLog2Modulus)

    /// <summary>
    /// Returns 1 / z.
    /// </summary>
    member inline z.Inverse = z.AbsSq.Inverse * z.Conjugate

    static member inline ComputeInverse (z: MPComplex) = z.Inverse

    /// <summary>
    /// Returns z / 2.
    /// </summary>
    member z.Half = { Real = z.Real.Half; Imag = z.Imag.Half }

    /// <summary>
    /// Returns 2 z.
    /// </summary>
    member z.TimesTwo = { Real = z.Real.TimesTwo; Imag = z.Imag.TimesTwo }


    /// <summary>
    /// Computes the square root of z up to the precision of z using Newton's method.
    /// The branch cut is the same as that of Complex.Sqrt
    /// </summary>
    static member Sqrt (z: MPComplex) =
        let exp = 2 * int(z.RoughLog2Modulus / 2) // exponent rounded to an even number.
        let approx =
            Complex( (z.Real |> BigFloat.Multiply2Power -exp).Float,
                     (z.Imag |> BigFloat.Multiply2Power -exp).Float)
            |> Complex.Sqrt
        // This shift is to embrace the possibility that a number is too large or small for float variable, but is still okay for MPF
        let prec = z.Precision
        let makeModest (r:float) = if Math.Abs(r) < 1.0e-6 then 0.0 else r
        let init =
            {
                Real = MPF.fromFloat(makeModest approx.Real,      prec) |> BigFloat.Multiply2Power (exp / 2)
                Imag = MPF.fromFloat(makeModest approx.Imaginary, prec) |> BigFloat.Multiply2Power (exp / 2)
            }
        let nIter = Math.Log( float prec, 2.0 )  |> int
        let next (now:MPComplex) _ = (now + z / now).Half
        List.fold next init [1..nIter]

    /// <summary>
    /// Computes z^exp where exp >= 0.
    /// </summary>
    static member Power (exp: uint32) (z: MPComplex) =
        let mutable ans =
            if (exp &&& 1u) = 1u then
                z
            else
                MPComplex.Create(1.0, 0.0, z.Precision)
        if exp > 0u then
            let mutable zPower = z
            for idx = 1 to Math.Log(float exp, 2.0) |> int do
                zPower <- zPower * zPower
                if ((exp >>> idx) &&& 1u) = 1u then
                    ans <- ans * zPower
        ans

    static member inline Multiply2Power (exp: int) (z: MPComplex) =
        { Real = z.Real |> BigFloat.Multiply2Power exp; Imag = z.Imag |> BigFloat.Multiply2Power exp }

    override z.ToString() =
        "(" + z.Real.ToString() + "," + z.Imag.ToString() + ")"

    static member ChangePrecision (precision: uint32) (z: MPComplex) =
        {
            Real = z.Real |> curry BigFloat.ChangePrecision precision
            Imag = z.Imag |> curry BigFloat.ChangePrecision precision
        }


type MPMatrix2x2 =
    {
        A: MPComplex // (1,1)
        B: MPComplex // (1,2)
        C: MPComplex // (2,1)
        D: MPComplex // (2,2)
    }

    static member Zero =
         {
            A = MPComplex.Zero
            B = MPComplex.Zero
            C = MPComplex.Zero
            D = MPComplex.Zero
         }

    member m.Precision =
        let x = min m.A.Precision m.B.Precision
        let y = min m.C.Precision m.D.Precision
        min x y

    static member ChangePrecision (precision: uint32) (m: MPMatrix2x2) =
        {
            A = m.A |> MPComplex.ChangePrecision precision
            B = m.B |> MPComplex.ChangePrecision precision
            C = m.C |> MPComplex.ChangePrecision precision
            D = m.D |> MPComplex.ChangePrecision precision
        }

    static member Identity (precision: uint32) =
        {
            A = MPComplex.Create(1.0, 0.0, precision)
            B = MPComplex.Create(0.0, 0.0, precision)
            C = MPComplex.Create(1.0, 0.0, precision)
            D = MPComplex.Create(1.0, 0.0, precision)
         }


    static member SigmaX =
         {
            A = MPComplex.Zero
            B = MPComplex.One
            C = MPComplex.One
            D = MPComplex.Zero
         }
    static member SigmaY =
         {
            A = MPComplex.Zero
            B = MPComplex.Create(0.0, -1.0)
            C = MPComplex.Create(0.0, +1.0)
            D = MPComplex.Zero
         }
    static member SigmaZ =
         {
            A = MPComplex.Create(+1.0, 0.0)
            B = MPComplex.Zero
            C = MPComplex.Zero
            D = MPComplex.Create(-1.0, 0.0)
         }

    static member (*) (mat1: MPMatrix2x2, mat2: MPMatrix2x2) =
        {
            A = mat1.A * mat2.A + mat1.B * mat2.C
            B = mat1.A * mat2.B + mat1.B * mat2.D
            C = mat1.C * mat2.A + mat1.D * mat2.C
            D = mat1.C * mat2.B + mat1.D * mat2.D
        }

    /// <summary>
    /// Scalar multiplication.
    /// </summary>
    static member (*) (z: MPComplex, mat: MPMatrix2x2) =
        {
            A = z * mat.A
            B = z * mat.B
            C = z * mat.C
            D = z * mat.D
        }

    static member (+) (mat1: MPMatrix2x2, mat2: MPMatrix2x2) =
        {
            A = mat1.A + mat2.A
            B = mat1.B + mat2.B
            C = mat1.C + mat2.C
            D = mat1.D + mat2.D
        }

    static member (-) (mat1: MPMatrix2x2, mat2: MPMatrix2x2) =
        {
            A = mat1.A - mat2.A
            B = mat1.B - mat2.B
            C = mat1.C - mat2.C
            D = mat1.D - mat2.D
        }

    static member (~-) (mat: MPMatrix2x2) =
        { A = -mat.A; B = -mat.B; C = -mat.C; D = -mat.D }

    static member Tr(mat: MPMatrix2x2) =
        mat.A + mat.D

    member m.ConjugateTranspose =
        {
            A = m.A.Conjugate
            B = m.C.Conjugate
            C = m.B.Conjugate
            D = m.D.Conjugate
        }
    member m.Dagger = m.ConjugateTranspose

    /// <summary>
    /// Returns $\sqrt{ |a|^2 + |b|^2 + |c|^2 + |d|^2 }$.
    /// </summary>
    static member FrobeniusNorm (m: MPMatrix2x2) =
        m.A.AbsSq + m.B.AbsSq + m.C.AbsSq + m.D.AbsSq
        |> MPF.Sqrt

    /// <summary>
    /// Returns the Frobenius norm of m.Dagger * m - Id.
    /// </summary>
    member m.DeviationFromUnitary =
        let maybeId = m.Dagger * m
        { maybeId with
            A = maybeId.A - MPComplex.One
            D = maybeId.D - MPComplex.One
        }
        |> MPMatrix2x2.FrobeniusNorm

    /// <summary>
    /// Returns mat.Dagger * mat / Tr(mat.Dagger * mat)
    /// </summary>
    static member ComputeRightProjector (mat: MPMatrix2x2) =
        // Assumes mat has rank 1.
        // TODO: Consider implementing the singular value decomposition.
        let ans = mat.Dagger * mat
        let invtr = (MPMatrix2x2.Tr(ans)).Inverse
        invtr * ans

    /// <summary>
    /// Multiplies itself by a factor of 0.5.
    /// </summary>
    member m.Half =
        {
            A = m.A.Half
            B = m.B.Half
            C = m.C.Half
            D = m.D.Half
        }

    override m.ToString() =
        "[ " + m.A.ToString() + " & " + m.B.ToString() + " \\\\ " + m.C.ToString() + " & " + m.D.ToString() + " ]"

    /// <summary>
    /// Returns $\bra + \text{(matrix)} \ket +$.
    /// </summary>
    static member HalfSumOfAllEntries (m: MPMatrix2x2) = (m.A + m.B + m.C + m.D).Half


type MPVector =
    {
        X: MPF;
        Y: MPF;
        Z: MPF
    }

    static member Zero = { X = MPF.Zero; Y = MPF.Zero; Z = MPF.Zero }
    override v.ToString() =
        "(" + v.X.ToString() + ", " + v.Y.ToString() + ", " + v.Z.ToString() + ")"

    static member (*) (scalar: MPF, v: MPVector) =
        { X = scalar * v.X; Y = scalar * v.Y; Z = scalar * v.Z }

    static member (-) (a: MPVector, b: MPVector) =
        {
            X = a.X - b.X
            Y = a.Y - b.Y
            Z = a.Z - b.Z
        }

    member v.Norm = v.X * v.X + v.Y * v.Y + v.Z * v.Z |> MPF.Sqrt

    static member Normalize (v: MPVector) = v.Norm.Inverse * v