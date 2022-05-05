@testset "conditionalITE" begin
    hyperparams = getHyperParameters()

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
        doT = true
        meanITE, covITE = conditionalITE(
            nothing, nothing, tyLS, yNoise, yScale,
            nothing, nothing, realT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
    @testset "No U, realT" begin
        doT = true
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
        doT = true
        meanITE, covITE = conditionalITE(
            uyLS, xyLS, tyLS, yNoise, yScale,
            U, X, realT, Y, doT)
        @test all(meanITE .== 0.0)
        @test all(covITE .== 0.0)
    end
end

@testset "conditionalSATE" begin

end

@testset "SATEsamples" begin

end

@testset "ITEsamples" begin

end

@testset "sampleITE" begin

end

@testset "SummarizeITE" begin

end