@testset "I/O" begin
    (priorparams, hyperparams,
        uyLS, xyLS, tyLS, yScale, yNoise,
        U, X, realT, binaryT, Y, obj) = getEstimationTestParams()
    savedG = nothing
    @testset "saveGPSLCObject" begin
        savedG = gpslc(obj, X, realT, Y)
        saveGPSLCObject(savedG, "tmp")
        saveGPSLCObject(savedG, "tmp.gpslc")
        @test true
    end
    @testset "loadGPSLCObject" begin
        loadedG = loadGPSLCObject("tmp")
        loadedG = loadGPSLCObject("tmp.gpslc")
        @test loadedG.X == X
        @test loadedG.obj == obj
        @test loadedG.T == realT
        @test loadedG.Y == Y
        @test loadedG.posteriorSamples == savedG.posteriorSamples
        @test loadedG.priorparams == savedG.priorparams
        @test loadedG.hyperparams == savedG.hyperparams
    end
end