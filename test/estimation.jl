
(priorparams, hyperparams,
    uyLS, xyLS, tyLS, yScale, yNoise,
    U, X, realT, binaryT, Y, obj) = getEstimationTestParams()

@testset "conditionalITE equals 0 if intervention equals treatment" begin

    doT = true
    @testset "No U, no X, binaryT" begin
        meanITE, covITE = conditionalITE(
            nothing, nothing, tyLS, yNoise, yScale,
            nothing, nothing, binaryT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "No U, binaryT" begin
        meanITE, covITE = conditionalITE(
            nothing, xyLS, tyLS, yNoise, yScale,
            nothing, X, binaryT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "No X, binaryT" begin
        meanITE, covITE = conditionalITE(
            uyLS, nothing, tyLS, yNoise, yScale,
            U, nothing, binaryT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "Full model, binaryT" begin
        meanITE, covITE = conditionalITE(
            uyLS, xyLS, tyLS, yNoise, yScale,
            U, X, binaryT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end

    doT = 1.0
    @testset "No U, no X, realT" begin
        meanITE, covITE = conditionalITE(
            nothing, nothing, tyLS, yNoise, yScale,
            nothing, nothing, realT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "No U, realT" begin
        meanITE, covITE = conditionalITE(
            nothing, xyLS, tyLS, yNoise, yScale,
            nothing, X, realT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "No X, realT" begin
        meanITE, covITE = conditionalITE(
            uyLS, nothing, tyLS, yNoise, yScale,
            U, nothing, realT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "Full model, realT" begin
        meanITE, covITE = conditionalITE(
            uyLS, xyLS, tyLS, yNoise, yScale,
            U, X, realT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
end

@testset "conditionalSATE" begin
    doT = true
    @testset "No U, no X, binaryT" begin
        meanITE, covITE = conditionalITE(
            nothing, nothing, tyLS, yNoise, yScale,
            nothing, nothing, binaryT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "No U, binaryT" begin
        meanITE, covITE = conditionalITE(
            nothing, xyLS, tyLS, yNoise, yScale,
            nothing, X, binaryT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "No X, binaryT" begin
        meanITE, covITE = conditionalITE(
            uyLS, nothing, tyLS, yNoise, yScale,
            U, nothing, binaryT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "Full model, binaryT" begin
        meanITE, covITE = conditionalITE(
            uyLS, xyLS, tyLS, yNoise, yScale,
            U, X, binaryT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end

    doT = 1.0
    @testset "No U, no X, realT" begin
        meanITE, covITE = conditionalITE(
            nothing, nothing, tyLS, yNoise, yScale,
            nothing, nothing, realT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "No U, realT" begin
        meanITE, covITE = conditionalITE(
            nothing, xyLS, tyLS, yNoise, yScale,
            nothing, X, realT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "No X, realT" begin
        meanITE, covITE = conditionalITE(
            uyLS, nothing, tyLS, yNoise, yScale,
            U, nothing, realT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "Full model, realT" begin
        meanITE, covITE = conditionalITE(
            uyLS, xyLS, tyLS, yNoise, yScale,
            U, X, realT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
end

@testset "ITEDistributions" begin
    doT = true
    @testset "No U, no X, binaryT" begin
        g = gpslc(nothing, nothing, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "No U, binaryT" begin
        g = gpslc(nothing, X, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "No X, binaryT" begin
        g = gpslc(obj, nothing, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "Full model, binaryT" begin
        g = gpslc(obj, X, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.predictionCovarianceNoise
    end

    doT = 1.0
    @testset "No U, no X, realT" begin
        g = gpslc(nothing, nothing, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "No U, realT" begin
        g = gpslc(nothing, X, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "No X, realT" begin
        g = gpslc(obj, nothing, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "Full model, realT" begin
        g = gpslc(obj, X, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.predictionCovarianceNoise
    end
end


@testset "SATEDistributions" begin
    doT = true
    @testset "No U, no X, binaryT" begin
        g = gpslc(nothing, nothing, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "No U, binaryT" begin
        g = gpslc(nothing, X, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "No X, binaryT" begin
        g = gpslc(obj, nothing, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "Full model, binaryT" begin
        g = gpslc(obj, X, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.predictionCovarianceNoise
    end


    doT = 1.0
    @testset "No U, no X, realT" begin
        g = gpslc(nothing, nothing, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "No U, realT" begin
        g = gpslc(nothing, X, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "No X, realT" begin
        g = gpslc(obj, nothing, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.predictionCovarianceNoise
    end
    @testset "Full model, realT" begin
        g = gpslc(obj, X, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.predictionCovarianceNoise
    end
end



@testset "ITEsamples: toy variance should match prediction covariance noise" begin
    nSamplesPerMixture = 5
    doT = true
    @testset "No U, no X, binaryT" begin
        g = gpslc(nothing, nothing, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "No U, binaryT" begin
        g = gpslc(nothing, X, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "No X, binaryT" begin
        g = gpslc(obj, nothing, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "Full model, binaryT" begin
        g = gpslc(obj, X, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end

    doT = 1.0
    @testset "No U, no X, realT" begin
        g = gpslc(nothing, nothing, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "No U, realT" begin
        g = gpslc(nothing, X, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "No X, realT" begin
        g = gpslc(obj, nothing, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "Full model, realT" begin
        g = gpslc(obj, X, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, doT)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
end


@testset "SATEsamples: toy variance should match prediction covariance noise" begin
    nSamplesPerMixture = 5
    doT = true
    @testset "No U, no X, binaryT" begin
        g = gpslc(nothing, nothing, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "No U, binaryT" begin
        g = gpslc(nothing, X, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "No X, binaryT" begin
        g = gpslc(obj, nothing, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "Full model, binaryT" begin
        g = gpslc(obj, X, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end


    doT = 1.0
    @testset "No U, no X, realT" begin
        g = gpslc(nothing, nothing, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "No U, realT" begin
        g = gpslc(nothing, X, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "No X, realT" begin
        g = gpslc(obj, nothing, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
    @testset "Full model, realT" begin
        g = gpslc(obj, X, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, doT)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test isapprox(Statistics.var(samples), hyperparams.predictionCovarianceNoise, atol=1e-9)
    end
end