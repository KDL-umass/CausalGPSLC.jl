@testset "Model Generation" begin
    priorparams, n, nU, nX, X, binaryT, realT = getToyData()

    @testset "Binary" begin
        # Treatment
        obs, _ = getToyObservations(n)
        for i in 1:n
            obs[:T=>i=>:T] = binaryT[i]
        end

        @testset "CausalGPSLCNoUNoCovBinaryT" begin
            (trace, _) = generate(CausalGPSLCNoUNoCovBinaryT, (priorparams, n, nothing, nothing), obs)
            @test true
        end
        @testset "CausalGPSLCNoCovBinaryT" begin
            (trace, _) = generate(CausalGPSLCNoCovBinaryT, (priorparams, n, nU, nothing), obs)
            @test true
        end

        # Covariates
        for k in 1:nX
            obs[:X=>k=>:X] = X[:, k]
        end
        @testset "CausalGPSLCNoUBinaryT" begin
            (trace, _) = generate(CausalGPSLCNoUBinaryT, (priorparams, n, nothing, nX), obs)
            @test true
        end
        @testset "CausalGPSLCBinaryT" begin
            (trace, _) = generate(CausalGPSLCBinaryT, (priorparams, n, nU, nX), obs)
            @test true
        end
    end

    @testset "Real" begin
        obs, _ = getToyObservations(n)
        obs[:T] = realT

        @testset "CausalGPSLCNoUNoCovRealT" begin
            (trace, _) = generate(CausalGPSLCNoUNoCovRealT, (priorparams, n, nothing, nothing), obs)
            @test true
        end

        obs[:T] = realT # whole symbol because no logit Map
        @testset "CausalGPSLCNoCovRealT" begin
            (trace, _) = generate(CausalGPSLCNoCovRealT, (priorparams, n, nU, nothing), obs)
            @test true
        end

        # Covariates
        for k in 1:nX
            obs[:X=>k=>:X] = X[:, k]
        end
        @testset "CausalGPSLCNoURealT" begin
            (trace, _) = generate(CausalGPSLCNoURealT, (priorparams, n, nothing, nU), obs)
            @test true
        end

        @testset "CausalGPSLCRealT" begin
            (trace, _) = generate(CausalGPSLCRealT, (priorparams, n, nU, nX), obs)
            @test true
        end

    end
end