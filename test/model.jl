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

    obs = Gen.choicemap()
    obs[:Y] = Y

    @testset "Binary" begin
        for i in 1:n
            obs[:T=>i=>:T] = binaryT[i]
        end

        @testset "GPSLCNoUNoCovBinaryT" begin
            (trace, _) = generate(GPSLCNoUNoCovBinaryT, (hyperparams, binaryT), obs)
            @test true
        end
        @testset "GPSLCNoUBinaryT" begin
            (trace, _) = generate(GPSLCNoUBinaryT, (hyperparams, X), obs)
            @test true
        end
    end
    @testset "Real" begin
        for i in 1:n
            obs[:T=>i=>:T] = realT[i]
        end

    end
end