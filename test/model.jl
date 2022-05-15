@testset "Model Generation" begin
    priorparams, n, nU, nX, X, binaryT, realT = getToyData()

    @testset "Binary" begin
        # Treatment
        obs, _ = getToyObservations(n)
        for i in 1:n
            obs[:T=>i=>:T] = binaryT[i]
        end

        @testset "GPSLCNoUNoCovBinaryT" begin
            (trace, _) = generate(GPSLCNoUNoCovBinaryT, (priorparams, n, nothing, nothing), obs)
            @test true
        end
        @testset "GPSLCNoCovBinaryT" begin
            (trace, _) = generate(GPSLCNoCovBinaryT, (priorparams, n, nU, nothing), obs)
            @test true
        end

        # Covariates
        for k in 1:nX
            obs[:X=>k=>:X] = X[:, k]
        end
        @testset "GPSLCNoUBinaryT" begin
            (trace, _) = generate(GPSLCNoUBinaryT, (priorparams, n, nothing, nX), obs)
            @test true
        end
        @testset "GPSLCBinaryT" begin
            (trace, _) = generate(GPSLCBinaryT, (priorparams, n, nU, nX), obs)
            @test true
        end
    end

    @testset "Real" begin
        obs, _ = getToyObservations(n)
        obs[:T] = realT

        @testset "GPSLCNoUNoCovRealT" begin
            (trace, _) = generate(GPSLCNoUNoCovRealT, (priorparams, n, nothing, nothing), obs)
            @test true
        end

        obs[:T] = realT # whole symbol because no logit Map
        @testset "GPSLCNoCovRealT" begin
            (trace, _) = generate(GPSLCNoCovRealT, (priorparams, n, nU, nothing), obs)
            @test true
        end

        # Covariates
        for k in 1:nX
            obs[:X=>k=>:X] = X[:, k]
        end
        @testset "GPSLCNoURealT" begin
            (trace, _) = generate(GPSLCNoURealT, (priorparams, n, nothing, nU), obs)
            @test true
        end

        @testset "GPSLCRealT" begin
            (trace, _) = generate(GPSLCRealT, (priorparams, n, nU, nX), obs)
            @test true
        end

    end
end