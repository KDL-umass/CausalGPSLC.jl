@testset "I/O" begin
    (priorparams, hyperparams,
        uyLS, xyLS, tyLS, yScale, yNoise,
        U, X, realT, binaryT, Y, obj) = getEstimationTestParams()
    savedG = nothing
    @testset "saveGPSLCObject" begin
        savedG = gpslc(obj, X, realT, Y)
        saveGPSLCObject(savedG, "tmp")
        @test true
    end
    @testset "loadGPSLCObject" begin
        loadedG = loadGPSLCObject("tmp")
        @test loadedG.X == X
        @test loadedG.obj == obj
        @test loadedG.T == realT
        @test loadedG.Y == Y
        @test loadedG.posteriorSamples == savedG.posteriorSamples
        @test loadedG.priorparams == savedG.priorparams
        @test isequal(loadedG.hyperparams, savedG.hyperparams)
    end
end