// Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the
// Microsoft Software License Terms for Microsoft Quantum Development Kit Libraries
// and Samples. See LICENSE in the project root for license information.

module Microsoft.Quantum.Research.QuantumSignalProcessing.Tests

open System
open Xunit
open Xunit.Abstractions
open System.Numerics
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra

open Microsoft.Quantum.Research.QuantumSignalProcessing.MPNumber
open Microsoft.Quantum.Research.QuantumSignalProcessing.MPFFT
open Microsoft.Quantum.Research.QuantumSignalProcessing.MPPolynomials
open Microsoft.Quantum.Research.QuantumSignalProcessing.QSPPrimitives
open Microsoft.Quantum.Research.QuantumSignalProcessing.QSP

type BigFloat = MPNumber.BigFloat

let BigFloatTest () =

    BigFloat.fromFloat(0.1) |> printfn "0.1 = %O"
    BigFloat.fromFloat(0.2) |> printfn "0.2 = %A"
    BigFloat.fromFloat(0.3) |> printfn "0.3 = %A"
    BigFloat.fromFloat(0.4) |> printfn "0.4 = %A"
    BigFloat.fromFloat(0.5) |> printfn "0.5 = %A"
    BigFloat.fromFloat(0.9) |> printfn "0.9 = %A"
    BigFloat.fromFloat(1.0) |> printfn "1.0 = %A"
    BigFloat.fromFloat(1.1) |> printfn "1.1 = %A"
    BigFloat.fromFloat(1.2) |> printfn "1.2 = %A"

    printfn "-----------------------------------------"
    let two = BigFloat.fromFloat(2.0, 1000u)
    //two.Set 2.0
    //two.SetPrecision 1000u

    printfn "2 = %O" two
    printfn "inverse of 2 = %O" two.Inverse

    printfn "2 + 2 = %O" (two + two)
    let zero = (two - two)
    printfn "2 - 2 = %O, %O" zero zero.Exponent
    printfn "2^2 = %O" (two * two)
    printfn "1 = %O" (two / two)

    printfn "-----------Small number addition test----------------------"
    let a = MPF.fromFloat(1.0, 100u)
    let b = MPF.fromFloat(1.0, 100u) |> MPF.Multiply2Power -88

    printfn "a = %O" a
    printfn "b = %O" b
    printfn "a + b = %O"  (a + b)
    printfn "--------------------------------"
    let bf = BigFloat.fromFloat(12.123, 64u)
    printfn "%O" bf
    printfn "%O" bf.Float
    printfn "%O" (MPF.Truncate bf).Float

    printfn "-------------------------"
    let one = MPF.fromFloat(10.0, 1000u)
    let two = MPF.fromFloat(20.0, 1000u)
    let three = MPF.fromFloat(30.0, 1000u)
    let e1 = MPExp one
    let e2 = MPExp two
    let e3 = MPExp three
    let e6 = MPF.fromFloat(60.0, 1000u) |> MPExp
    printfn "%O\n%O\n%O" e1 e2 e3
    printfn "1 = %O" (e1 * e2 * e3 / e6)
    printfn "-------------------------"
    printfn "BigFloat Test end ---------------------"


let BlochProjectorTest() =

    let v = {
              X = MPF.fromFloat( +1.3241, 200u )
              Y = MPF.fromFloat( -4.3241, 200u )
              Z = MPF.fromFloat( +0.7241, 200u )
            } |> MPVector.Normalize
    printfn "00Normalized vector is %A" v

    let proj = BlochVectorToProjector v
    let projsq = proj * proj
    let postV = ProjectorToBlochVector projsq
    printfn "ReNormalized vector is %A" postV

let MatrixArithmeticTest() =
    let x = {
                A = MPComplex.Create(1.0,1.0)
                B = MPComplex.Create(2.0,1.0)
                C = MPComplex.Create(3.0,2.0)
                D = MPComplex.Create(4.0,0.0)
            }
    let y = x * x
    printfn "x * x = %A" y
    printfn "x = %A" x
    printfn "xd= %A" x.Dagger
    printfn "x = %A" x
    printfn "xdx = %A" (x.Dagger * x)

let IntegralLaurentPolynomialRootsTest() =
    let (den, nums) =
        [| 0.23; 0.123; -1.134; -3.9; 2.1 |]
        |> Array.map (fun x -> MPF.fromFloat(x))
        |> MPFArrayToMPZArray
    let ilp = IntegralLaurentPolynomial(nums, -2)
    ilp.Roots 100u
    |> printfn "%A"


let SquareIntegralLaurentPolynomialTest () =
    let coe = [| 1I ; 1I ; -123I |]
    let f = IntegralLaurentPolynomial(coe, -1)
    let sq = f |> IntegralLaurentPolynomial.Square
    printfn "original = %A" f
    printfn "square   = %A" sq

let SqrtTest() =
    let prec = 10000u
    let sqrt = { MPComplex.Zero with Imag = MPF.One |> curry MPF.ChangePrecision prec } |> MPComplex.Sqrt
    printfn "%O" sqrt
    let half = MPF.fromFloat (0.5, prec)
    let rt2 = MPF.Sqrt half
    printfn "%O" rt2

let ComplexMultiplyTest() =
    let prec = 6000u
    let mutable z = MPComplex.One |> MPComplex.ChangePrecision prec
    let w = MPComplex.Sqrt( MPComplex.Create( 0.0, 1.0, prec ) )
    printfn "sqrt{i} = %O" w
    let stopWatch = System.Diagnostics.Stopwatch.StartNew()
    for i in 1..1000 do
        z <- z * w
    stopWatch.Stop()
    printfn "This is close to zero: %O" (z - MPComplex.One)
    printfn "F# complexMul took (ms) %O" stopWatch.Elapsed.TotalMilliseconds

let RealMuliplyTest() =
    let stopWatch = System.Diagnostics.Stopwatch.StartNew()
    let mutable z = MPF.fromFloat(1.0, 100000u)
    let w = MPF.fromFloat( 1.0003, 100000u )
    let f = MPF.fromFloat( 0.9997, 100000u )
    for i in 1..100 do
        z <- z * w
        z <- z * f
    stopWatch.Stop()
    printfn "%O" z
    printfn "F# realMul took (ms) %A" stopWatch.Elapsed.TotalMilliseconds



let TestMathNetFFT log2N omega =
    let n = 1 <<< log2N
    let x = Array.create n (Complex(0.0, 0.0))
    let tf = TwiddleFactor(log2N, 64u)
    for idx in 0..(n - 1) do
        //let ti = 2.0 * Math.PI * float omega * float i / float n
        let c = tf.Compute (omega * idx)
        x.[idx] <- Complex( c.Real.Float, c.Imag.Float )
    MathNet.Numerics.IntegralTransforms.Fourier.Forward x
    //printfn "y.[%A] is %A" omega x.[omega]
    for i in 0..(n - 1) do
        printfn "y.[%A] is %A" i x.[i]


let RootFindingTest() =
    let coefList = [|
            93; 93; -28; -77; 17; 53;
            18; 81; -77; -48; -22; -2; 32; 26; 77; -36; 100; 47; 62; -59; 63; -93; 2; 93; -32; 98; 65; 21; 1; 52; 84; 8; -48;
            -6; 43; -52; -34; -53; 74; 1; 69; -70; -45; 48; -54; 47; 56; -9; 64; 9; 81; -88; -65; -35; 44; 7; -54; 10; -28; 3; 42; 3; 18; 23; -94;
            -60; 30; -49; -8; 55; -31; -58; 80; 85; -96; 2; 29; -95; -45; -69; 99; -8; 64; -83; -27; 9; -77; 62; 42; 69; 94; -30; -78; 56; -9; -91;
            17; -49; -52; -100; 52; 43; -21; 51; 26; -5; 53; -38; 97; 80; 35; 23;
            -27; -36; 19; 81; 71; 37; -75; 3; 89; 85; 25; 67; 91; -98; -71; 66;
            -83; 60; -2; -76; -32; -20; 75; 48; 90; -9; 15; -64; -82; 67; -69;
            -5; 65; -9; -99; 65; 12; 94; 89; -58; 98; -1; 57; -29; -91; -17; 95;
            -4; -90; -64; -81; 95; -39; 52; 72; -59; 86; 10; -98; -25; 92; 97;
            -97; -44; -85; -44; 25; 41; 90; 98; 15; -43; 48; 8; -60; -21; -60;
            -16; -87; -53; -47; 73; -53; 65; 32; -16; 11; 73; 0; 14; 76; -45; -8;
            -17; 2; -6; 18; 45; -5; -70; 23; 81; -22; 79; 95; -46; 51; 67; -50;
            -61; 21; -61; 58; 92; 62; 76; -22; -4; -51; 21; 24; -89; 90; 49; 46;
            -54; 50; -34; -65; -62; -89; -52; 69; 57; 82; 62; 71; -56; 90; -90;
            -4; 52; 36; 69; -7; 49; -12; 55; 4; 80; -38; 21; 37; 20; -56; -95;
            -33; -4; -100; -56; 26; 5; 71; 63; -52; 28; -68; -35; -21; -22; -13;
            -91; 98; 49; -7; 25; 20; 49; -6; 9; 21; 70; 47; 59; 54; -17; -90; 62;
            2; 46; 79; -64; 27; 27; -28; -75; 96; -2; 15; -35; -45; -75; -84;
            -74; 25; -64; 81; 6; 67; 71; 67; 9; 11; 22; 23; -8; 11; -10; 5; -40;
            -83; 88; -78; 75; 80; 45; -9; 55; -59; 3; 1; 21; 51; -18; -84; -55;
            81; 37; -26; -14; -16; 8; -32; -39; -39; -43; 39; -83; -30; -74; -89;
            -78; -14; 94; 60; 81; 96; 9; -78; 22; -48; 61; -55; -97; -87; 37;
            -65; -38; -92; -25; -93; 26; -49; -4; 88; -87; 2; 40; -53; -20; 69;
            -49; 12; -18; 94; 88; 91; -66
          |]
    let roots =
        coefList
        |> Array.map float
        |> MathNet.Numerics.FindRoots.Polynomial
        |> Array.map MPComplex.Create

    let rr = RootRefinery(RealPolynomial(coefList))
    rr.SetApproximateRoots roots
    rr.ComputeRoots 100u
    rr.ComputeRoots 200u
    rr.ComputeRoots 600u

let JacobiAngerTest() =
    JacobiAngerExpansion 1.0e-70 1.32 |> ignore
    printfn "Done"


let OneOverSineTest() =
    OneOverSine 1.0e-2 30.0 |> ignore
    OneOverSine 1.0e-10 5.0 |> ignore
    printfn "Done"

type QSPTests() =

    [<Fact>]
    member __.ComplexModulusPrecisionTest() =
        let z = MPComplex.Create(1.0, 1.0, 64u)
        Assert.True( z.Precision = z.Modulus.Precision )

    [<Fact>]
    member __.LeastCommonMultipleTest() =
        let ans = [| 1I; 3I; 12I; 15I; -30I |] |> LeastCommonMultiple
        Assert.True( (ans - 60I).IsZero )


    member __.SampleRootFinder() =
        let zeros =
            [|-1024I; -1465303I; -1769348806I; -1775719827921I; -1453104891602280I;
            -946969418618304255I; -477034578533815870524I; -178676776246659516080499I;
            -1024I
            |]
          |> Array.map float
          |> MathNet.Numerics.FindRoots.Polynomial
          |> Array.filter (fun r -> r = Complex(0.0,0.0) )
        // The coefficient list starts with a nonzero number, hence 0 cannot be a root.
        // In fact, the roots
        // -1.7e20, -1.3e-3, -9.7e-4 +- 8.8e-4 I, 1.6e-4 +- 1.3e-3 I,  4.5e-4 +- 1.1e-3 I
        // are far from zero with double precision.
        // Therefore, the following line must be true.
        zeros.Length = 0

    [<Fact>]
    member __.RootFindingUsingMKLTest() =
        Control.UseNativeMKL()
        Assert.True( __.SampleRootFinder() )

    [<Fact>]
    member __.RootFindingUsingManagedTest() =
        Control.UseManaged()
        // The following demonstrates the unstability of the root finding routine using the managed code.
        Assert.False( __.SampleRootFinder() )

    [<Fact>]
    member __.RealSqrtPrecisionTest() =
        let x = MPF.fromFloat(0.5, 10000u) |> MPF.Sqrt
        let re = x.ToString().ToCharArray()
        let goal =
            "7071067811865475244008443621048490392848359376884740365883398689953662392310535194251937671638207"
            |> fun x -> x.ToCharArray()
        let len_goal = goal.Length
        let strRe = String(  re, 2, len_goal )
        let strGo = String(goal, 0, len_goal )
        Assert.True( strGo = strRe && true )

    [<Fact>]
    member __.ComplexSqrtPrecisionTest() =
        let x = MPComplex.Create(0.0, 1.0, 10000u) |> MPComplex.Sqrt
        let re = x.Real.ToString().ToCharArray()
        let im = x.Imag.ToString().ToCharArray()
        let goal =
            "7071067811865475244008443621048490392848359376884740365883398689953662392310535194251937671638207"
            |> fun x -> x.ToCharArray()
        let len_goal = goal.Length
        let strRe = String( re, 2, len_goal )
        let strIm = String( im, 2, len_goal )
        let strGo = String(goal, 0, len_goal )
        Assert.True(strGo = strRe && strRe = strIm)

    [<Fact>]
    member __.BigFloatTruncateTest() =
        Assert.True( (MPF.fromFloat(1234.0) - (MPF.fromFloat( 1234.5678 ) |> MPF.Truncate)).RoughLog2Modulus < -20 )

    [<Fact>]
    member __.RealPolynomialEvaluationTest() =
        let prec = 64u
        let c = [| 4; 2; 15; -13; 4 |]
        let f = RealPolynomial( c )
        printfn "polynomial is %O" f
        let z = MPComplex.Create( 0.71, 0.83, prec )
        let fe = f.Evaluate z
        printfn "   Evaluate f%O = %O" z fe
        printfn "ST Evaluate f%O = %O" z (f.STEvaluate z)
        printfn "EvaluateDerivativeSlow f'%O = %O" z (f.EvaluateDerivativeSlow z)
        printfn "ST   EvaluateDerivative f%O = %O" z (f.STEvaluateDerivative z)
        let ste= f.STEvaluate z
        let dif = fe - ste
        printfn "Evaluation difference = %O with precision %A" dif z.Precision
        let zz = z |> MPComplex.ChangePrecision (prec * 20u)
        let tinyDiff = (f.Evaluate zz) - (f.STEvaluate zz)
        printfn "Evaluation difference = %O with precision %A" tinyDiff zz.Precision
        Assert.True( tinyDiff.RoughLog2Modulus < -1000 )


    member __.TestRandomSignalPoly (expansionPrecision: uint32, decomposePrecision: uint32) =
        let randomReals = [|
                -0.405664; 0.119356; 0.40769; 0.932002; -0.630361; 0.890098;
                0.418777; 0.71085; -0.859934; 0.00710487; -0.164678; -0.708004;
                0.272825; -0.256874; -0.909851; -0.401272; 0.715576; 0.710236;
                0.155831; -0.702643; -0.0256289; 0.768353; 0.819868; 0.461453;
                -0.120919; -0.643399; 0.170688; -0.824982; -0.729558; -0.562011;
                -0.973768; 0.087969; 0.779457; -0.604807; 0.517294; 0.36157;
                0.699766; -0.451361; -0.109543; 0.129403; 0.656827; -0.415813;
                0.610716; -0.238136; -0.492792; -0.880376; -0.718022; 0.660816;
                0.782106; 0.120093; -0.994196; 0.643039; -0.1643; -0.343824;
                -0.623534; -0.191356; 0.770227; -0.954637; 0.011434; 0.464503;
                -0.522941; 0.34735; 0.906663; 0.976689; -0.385281; 0.458595;
                -0.99699; -0.665977; 0.683913; -0.850725; 0.249503; 0.288307;
                0.220773; 0.877152; 0.50972; 0.329643; 0.00515425; -0.319956;
                -0.0717718; 0.209307; -0.015322; -0.419631; -0.981251; 0.0895187;
                0.695572; -0.085672; 0.494495; 0.404528; -0.00802474; -0.881274;
                -0.0976171; -0.88891; 0.00261358; 0.0705922; -0.952396; 0.55413;
                0.121073; 0.418138; 0.181401; 0.557835; -0.73337; -0.0407727;
                0.0391607; 0.0836149; 0.0834104; 0.135884; 0.738987; -0.539464;
                -0.60892; 0.0520106; 0.799318; 0.911141; -0.335802; -0.955107;
                0.624563; 0.324544; -0.244182; 0.462294; -0.801054; 0.528899;
                -0.287998; -0.0123166; 0.942507; 0.545773; 0.469021; -0.568561;
                -0.369375; -0.249994; 0.20115; -0.709574; 0.017186; -0.954556;
                0.175469; -0.520542; -0.0476345; 0.59685; 0.184741; -0.634058;
                0.594973; 0.575866; 0.297902; -0.0387713; 0.462094; -0.307965;
                0.993225; 0.468726; -0.83091; -0.705645; -0.258965; 0.427937;
                -0.927865; -0.942206; 0.588605; -0.928195; -0.664154; -0.0897801;
                -0.120965; 0.41634; -0.17765; 0.636119; 0.791766; -0.449427; 0.77885;
                0.647438; -0.641807; -0.152998; 0.0710088; 0.725964; 0.0889895;
                0.886137; 0.749495; -0.682047; 0.709477; 0.416508; -0.194376;
                0.482668; -0.334299; -0.395313; -0.227001; 0.479975; -0.0176547;
                0.81001; 0.382476; 0.175623; 0.721971; 0.836463; -0.237081; 0.484274;
                0.00982969; 0.407085; -0.407666; -0.819864; 0.832394; 0.822526;
                -0.474557; -0.115004; 0.236141; 0.55216; 0.271581; 0.356214;
                0.136246; 0.999926; 0.429373; 0.888483; -0.679497; 0.969758;
                0.759467; -0.408382; 0.0761139; 0.784971; 0.810074; 0.931359;
                -0.571669; 0.86394; 0.215178; -0.234207; -0.0672123; -0.428078;
                0.16333; -0.519086; -0.998058; -0.928655; -0.530291; 0.716775;
                0.0152761; 0.466872; -0.543348; 0.313972; 0.912811; 0.172302;
                0.405574; -0.666843; -0.729693; 0.715637; 0.323181; -0.222532;
                -0.899442; 0.865823; -0.184503; 0.487481; 0.482364; -0.00516659;
                0.429142; -0.00849896; -0.48836; 0.958542; 0.550859; -0.45936;
                0.941645; 0.419334; 0.41675; -0.212983; -0.646168; 0.0575724;
                -0.9944; 0.337345; -0.022231; -0.40717; -0.8435; -0.22159; -0.62825;
                0.950664; 0.285732; 0.224187; -0.0364356; 0.931666; -0.0624503;
                -0.309158; -0.028818; 0.985131; 0.790838; -0.731322; -0.589722;
                -0.424275; -0.12624; -0.284455; -0.53926; 0.49227; 0.868742;
                0.570012; 0.0241643; -0.134076; 0.755199; -0.946951; 0.479172;
                -0.514089; -0.844554; 0.905628; -0.985748; -0.926074; 0.866171;
                0.917426; -0.129296; 0.478631; 0.53768; 0.928849; 0.653021; 0.864742;
                -0.906273; -0.647982 |] // length = 300

        let n = randomReals.Length / 3
        printfn "%A factors." n
        printfn "expansion precision %A, decompositin precision %A" expansionPrecision decomposePrecision

        let initDir = Array.init n (fun idx ->
                        {
                            X = MPF.fromFloat(randomReals.[3 * idx + 0], expansionPrecision)
                            Y = MPF.fromFloat(randomReals.[3 * idx + 1], expansionPrecision)
                            Z = MPF.fromFloat(randomReals.[3 * idx + 2], expansionPrecision)
                        } |> MPVector.Normalize
            )

        let signalPoly =
            {
                FrontUnitaryCorrection = MPMatrix2x2.Identity expansionPrecision
                PointsOnBlochSphere = initDir
            }
            |> ReconstructSignalPolynomial

        for idx in 0..(signalPoly.Length - 1) do
            signalPoly.[idx] <- signalPoly.[idx] |> MPMatrix2x2.ChangePrecision decomposePrecision

        let qsp = signalPoly |> FullSignalPolynomialToPrimitiveDirections

        printfn "Correcting unitary %O" qsp.FrontUnitaryCorrection
        printfn "initDir.[0] = %O" (initDir.[0])
        printfn "postDir.[0] = %O" (qsp.PointsOnBlochSphere.[0])
        initDir.[0] - qsp.PointsOnBlochSphere.[0]

    [<Fact>]
    member __.RandomSignalPolynomialTest() =
        __.TestRandomSignalPoly (64u, 64u) |> ignore
        printfn ""
        __.TestRandomSignalPoly (128u, 128u) |> ignore
        printfn ""
        let dif = __.TestRandomSignalPoly (256u, 256u)
        Assert.True( dif.Norm.RoughLog2Modulus < -80 )

    member __.TestFFT log2N omega =
        let n = 1 <<< log2N
        let prec = 1000u
        let x = Array.create n MPComplex.Zero
        printfn "precision of empty x is %A" x.[omega].Precision
        let tf = TwiddleFactor(log2N, prec)
        for idx in 0..(n - 1) do
            //let ti = 2.0 * Math.PI * float omega * float i / float n
            x.[idx] <- tf.Compute (omega * idx)
            //printfn "x.[%A] is %A" idx x.[idx].ToStringFull
        printfn "precision of x.[%A] is %A" omega x.[omega].Precision
        let y = FFT x
        printfn "y.[%A] is %A with precision %A" omega (y.[omega].ToString()) y.[omega].Precision
        y

    [<Fact>]
    member __.TestMPFFT() =
        let freq = 351
        let y = __.TestFFT 12 freq
        let dif = y.[freq] - MPF.One
        Assert.True( dif.RoughLog2Modulus < -900 )

    [<Fact>]
    member __.``Sum of squares of sin and cos is 1``() =
        let theta = MPF.fromFloat(-12.423, 1000u)
        let sin = MPSin theta
        let cos = MPCos theta
        let maybeZero = sin * sin + cos * cos - MPF.One
        Assert.True( maybeZero.RoughLog2Modulus < -900 )

[<EntryPoint>]
let main argv =
    //BigFloatTest ()
    //RealMuliplyTest()
    //ComplexMultiplyTest()

    //RootFindingTest()
    JacobiAngerTest()
    //OneOverSineTest()

    //let qsptest = QSPTests()
    //qsptest.RandomSignalPolynomialTest()


    Console.Read() |> ignore
    0 // return an integer exit code
