"""
Returns `obs` with Toy output data samples set and those outputs `Y`

This facilitates different tests adding covariate and treatment 
observations when getting dataset values from `getToyData`
"""
function getToyObservations(n)
    Y = rand(n)
    obs = Gen.choicemap()
    obs[:Y] = Y
    return obs, Y
end

"""Returns `priorparams, n, nU, nX, X, binaryT, realT` for a toy dataset"""
function getToyData(n=10, nU=2, nX=5)
    priorparams = getPriorParameters()
    X = rand(n, nX)
    binaryT::Array{Bool,1} = collect(rand(n) .< 0.5)
    realT::Array{Float64,1} = rand(n)

    objectCounts = [floor(Int64, n / nU) for _ in 1:nU]
    if n % nU > 0
        objectCounts = vcat(objectCounts, [n % nU])
    end
    priorparams["SigmaU"] = generateSigmaU(objectCounts)
    return priorparams, n, nU, nX, X, binaryT, realT
end

""" 
    (priorparams, hyperparams, uyLS, xyLS, tyLS, yScale, yNoise,
    U, X, realT, binaryT, Y, obj) = getEstimationTestParams()

Returns simple dataset and params for estimation testing.
"""
function getEstimationTestParams()
    priorparams = getPriorParameters()
    hyperparams = GPSLC.getHyperParameters()
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
    obj = [1]
    return (priorparams, hyperparams,
        uyLS, xyLS, tyLS, yScale, yNoise,
        U, X, realT, binaryT, Y, obj)
end