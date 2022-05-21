@testset "predictCounterfactualEffects" begin
    (priorparams, hyperparams,
        uyLS, xyLS, tyLS, yScale, yNoise,
        U, X, realT, binaryT, Y, obj) = getEstimationTestParams()

    nSamplesPerMixture = 30
    doT = 1.0
    g = gpslc(obj, X, realT, Y)

    ite, _ = predictCounterfactualEffects(g, 15, minDoT=0.0, maxDoT=1.0)
    actual = mean(ite)
    @test -1.0 <= actual && actual <= 1.0
end