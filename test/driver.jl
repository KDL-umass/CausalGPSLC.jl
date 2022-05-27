@testset "gpslc meta information tests" begin
    g = gpslc("$(prefix)test_data/minimal.csv")
    n = getN(g)
    nU = getNU(g)
    nX = getNX(g)
    @test true
end

@testset "Sample Treatment Effects for Toy Examples" begin
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
    Y = [rand()]
    obj = [1]

    doT = 1.0

    g = gpslc(obj, X, realT, Y)
    n = getN(g)
    nU = getNU(g)
    nX = getNX(g)
    @testset "sampleITE" begin
        samples = sampleITE(g, doT)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test Statistics.var(samples) <= hyperparams.predictionCovarianceNoise
    end

    @testset "sampleSATE" begin
        samples = sampleSATE(g, doT)
        @test -sqrt(hyperparams.predictionCovarianceNoise) <= mean(samples)
        @test mean(samples) <= sqrt(hyperparams.predictionCovarianceNoise)
        @test Statistics.var(samples) <= hyperparams.predictionCovarianceNoise
    end
end

@testset "summarizeEstimates" begin
    @testset "NEEC using gpslc" begin
        expected = CSV.read("$(prefix)test_results/NEEC_sampled_0.6.csv", DataFrame)
        g = gpslc("$(prefix)test_data/NEEC_sampled.csv")
        ITEsamples = sampleITE(g, 0.6)
        actual = summarizeEstimates(ITEsamples; savetofile="tmp.csv")
        @test countCloseEnough(expected, actual) >= 0.50
    end

    @testset "credible interval" begin
        @testset "90%" begin
            interval = 0.9
            sample = 0:100
            samples = toMatrix([collect(sample)], 1, 101)
            summary = summarizeEstimates(samples; credible_interval=interval)
            @test summary[1, "LowerBound"] ≈ 5.0
            @test summary[1, "UpperBound"] ≈ 95.0
        end
        @testset "80%" begin
            interval = 0.8
            sample = 0:100
            samples = toMatrix([collect(sample)], 1, 101)
            summary = summarizeEstimates(samples; credible_interval=interval)
            @test summary[1, "LowerBound"] ≈ 10.0
            @test summary[1, "UpperBound"] ≈ 90.0
        end
    end
end