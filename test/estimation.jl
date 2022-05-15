
priorparams = getPriorParameters()
hyperparams = GPSLC.getHyperParameters()
nSamplesPerMixture = 30
uyLS = [1.0]
xyLS = [1.0]
tyLS = 1.0
yScale = 1.0
yNoise = 1.0
U = [[1.0]]
X = ones(1, 1)
realT = [1.0]
binaryT = [true]
Y = [rand()]
obj = [1]

@testset "conditionalITE" begin

    @testset "No U, no X, binaryT" begin
        doT = true
        meanITE, covITE = conditionalITE(
            nothing, nothing, tyLS, yNoise, yScale,
            nothing, nothing, binaryT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "No U, binaryT" begin
        doT = true
        meanITE, covITE = conditionalITE(
            nothing, xyLS, tyLS, yNoise, yScale,
            nothing, X, binaryT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "No X, binaryT" begin
        doT = true
        meanITE, covITE = conditionalITE(
            uyLS, nothing, tyLS, yNoise, yScale,
            U, nothing, binaryT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "Full model, binaryT" begin
        doT = true
        meanITE, covITE = conditionalITE(
            uyLS, xyLS, tyLS, yNoise, yScale,
            U, X, binaryT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end


    @testset "No U, no X, realT" begin
        doT = 1.0
        meanITE, covITE = conditionalITE(
            nothing, nothing, tyLS, yNoise, yScale,
            nothing, nothing, realT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "No U, realT" begin
        doT = 1.0
        meanITE, covITE = conditionalITE(
            nothing, xyLS, tyLS, yNoise, yScale,
            nothing, X, realT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "No X, realT" begin
        doT = true
        meanITE, covITE = conditionalITE(
            uyLS, nothing, tyLS, yNoise, yScale,
            U, nothing, realT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "Full model, realT" begin
        doT = 1.0
        meanITE, covITE = conditionalITE(
            uyLS, xyLS, tyLS, yNoise, yScale,
            U, X, realT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
end

@testset "conditionalSATE" begin
    @testset "No U, no X, binaryT" begin
        doT = true
        meanITE, covITE = conditionalITE(
            nothing, nothing, tyLS, yNoise, yScale,
            nothing, nothing, binaryT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "No U, binaryT" begin
        doT = true
        meanITE, covITE = conditionalITE(
            nothing, xyLS, tyLS, yNoise, yScale,
            nothing, X, binaryT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "No X, binaryT" begin
        doT = true
        meanITE, covITE = conditionalITE(
            uyLS, nothing, tyLS, yNoise, yScale,
            U, nothing, binaryT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "Full model, binaryT" begin
        doT = true
        meanITE, covITE = conditionalITE(
            uyLS, xyLS, tyLS, yNoise, yScale,
            U, X, binaryT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end


    @testset "No U, no X, realT" begin
        doT = 1.0
        meanITE, covITE = conditionalITE(
            nothing, nothing, tyLS, yNoise, yScale,
            nothing, nothing, realT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "No U, realT" begin
        doT = 1.0
        meanITE, covITE = conditionalITE(
            nothing, xyLS, tyLS, yNoise, yScale,
            nothing, X, realT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "No X, realT" begin
        doT = 1.0
        meanITE, covITE = conditionalITE(
            uyLS, nothing, tyLS, yNoise, yScale,
            U, nothing, realT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
    @testset "Full model, realT" begin
        doT = 1.0
        meanITE, covITE = conditionalITE(
            uyLS, xyLS, tyLS, yNoise, yScale,
            U, X, realT, Y, doT)
        meanSATE, varSATE = conditionalSATE(meanITE, covITE)
        @test meanSATE == 0.0
        @test varSATE == 0.0
    end
end

@testset "ITEDistributions" begin
    @testset "No U, no X, binaryT" begin
        g = gpslc(nothing, nothing, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "No U, binaryT" begin
        g = gpslc(nothing, X, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "No X, binaryT" begin
        g = gpslc(obj, nothing, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "Full model, binaryT" begin
        g = gpslc(obj, X, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.iteCovarianceNoise
    end


    @testset "No U, no X, realT" begin
        g = gpslc(nothing, nothing, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "No U, realT" begin
        g = gpslc(nothing, X, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "No X, realT" begin
        g = gpslc(obj, nothing, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "Full model, realT" begin
        g = gpslc(obj, X, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        @test mean(MeanITEs) ≈ 0.0
        @test mean(CovITEs) ≈ hyperparams.iteCovarianceNoise
    end
end


@testset "SATEDistributions" begin
    @testset "No U, no X, binaryT" begin
        g = gpslc(nothing, nothing, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "No U, binaryT" begin
        g = gpslc(nothing, X, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "No X, binaryT" begin
        g = gpslc(obj, nothing, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "Full model, binaryT" begin
        g = gpslc(obj, X, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.iteCovarianceNoise
    end


    @testset "No U, no X, realT" begin
        g = gpslc(nothing, nothing, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "No U, realT" begin
        g = gpslc(nothing, X, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "No X, realT" begin
        g = gpslc(obj, nothing, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.iteCovarianceNoise
    end
    @testset "Full model, realT" begin
        g = gpslc(obj, X, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        @test mean(MeanSATEs) ≈ 0.0
        @test mean(CovSATEs) ≈ hyperparams.iteCovarianceNoise
    end
end



@testset "ITEsamples" begin
    @testset "No U, no X, binaryT" begin
        g = gpslc(nothing, nothing, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovITEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "No U, binaryT" begin
        g = gpslc(nothing, X, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovITEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "No X, binaryT" begin
        g = gpslc(obj, nothing, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovITEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "Full model, binaryT" begin
        g = gpslc(obj, X, binaryT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovITEs) <= hyperparams.iteCovarianceNoise
    end


    @testset "No U, no X, realT" begin
        g = gpslc(nothing, nothing, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovITEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "No U, realT" begin
        g = gpslc(nothing, X, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovITEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "No X, realT" begin
        g = gpslc(obj, nothing, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovITEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "Full model, realT" begin
        g = gpslc(obj, X, realT, Y)
        MeanITEs, CovITEs = ITEDistributions(g, true)
        samples = ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovITEs) <= hyperparams.iteCovarianceNoise
    end
end


@testset "SATEsamples" begin
    @testset "No U, no X, binaryT" begin
        g = gpslc(nothing, nothing, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovSATEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "No U, binaryT" begin
        g = gpslc(nothing, X, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovSATEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "No X, binaryT" begin
        g = gpslc(obj, nothing, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovSATEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "Full model, binaryT" begin
        g = gpslc(obj, X, binaryT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovSATEs) <= hyperparams.iteCovarianceNoise
    end


    @testset "No U, no X, realT" begin
        g = gpslc(nothing, nothing, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovSATEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "No U, realT" begin
        g = gpslc(nothing, X, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovSATEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "No X, realT" begin
        g = gpslc(obj, nothing, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovSATEs) <= hyperparams.iteCovarianceNoise
    end
    @testset "Full model, realT" begin
        g = gpslc(obj, X, realT, Y)
        MeanSATEs, CovSATEs = SATEDistributions(g, true)
        samples = SATEsamples(MeanSATEs, CovSATEs, nSamplesPerMixture)
        @test -sqrt(hyperparams.iteCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.iteCovarianceNoise)
        @test Statistics.var(CovSATEs) <= hyperparams.iteCovarianceNoise
    end
end