@testset "predictCounterfactualEffects" begin
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

    ite, _ = predictCounterfactualEffects(g, 15, minDoT=0.0, maxDoT=1.0)
    actual = mean(ite)
    @test -1.0 <= actual && actual <= 1.0
end