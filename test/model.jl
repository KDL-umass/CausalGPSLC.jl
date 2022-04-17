function loadData()
    CSV.read("$(prefix)test_data/additive_linear.csv", DataFrame)
end

@testset "Model Generation" begin
    hyperparams = getHyperParameters()
    n = 10
    nX = 5
    X = rand(n, nX)
    binaryT::Array{Bool,1} = collect(rand(n) .< 0.5)
    realT::Array{Float64,1} = rand(n)
    Y = rand(n)
    nU = 2
    objectCounts = [5, 5]
    hyperparams["SigmaU"] = generateSigmaU(objectCounts)

    obs = Gen.choicemap()
    obs[:Y] = Y

    @testset "Binary" begin
        # Treatment
        for i in 1:n
            obs[:T=>i=>:T] = binaryT[i]
        end

        @testset "GPSLCNoUNoCovBinaryT" begin
            (trace, _) = generate(GPSLCNoUNoCovBinaryT, (hyperparams, binaryT), obs)
            @test true
        end
        @testset "GPSLCNoCovBinaryT" begin
            (trace, _) = generate(GPSLCNoCovBinaryT, (hyperparams, nU), obs)
            @test true
        end

        # Covariates
        for k in 1:nX
            obs[:X=>k=>:X] = X[:, k]
        end
        @testset "GPSLCNoUBinaryT" begin
            (trace, _) = generate(GPSLCNoUBinaryT, (hyperparams, X), obs)
            @test true
        end
        @testset "GPSLCBinaryT" begin
            (trace, _) = generate(GPSLCBinaryT, (hyperparams, nU, nX), obs)
            @test true
        end
    end
    @testset "Real" begin
        for i in 1:n
            obs[:T=>i=>:T] = realT[i]
        end

    end
end