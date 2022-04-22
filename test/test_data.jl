"""Load additive linear synthetic data from paper"""
function loadData()
    CSV.read("$(prefix)test_data/additive_linear.csv", DataFrame)
end

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

"""Returns `hyperparams, n, nU, nX, X, binaryT, realT` for a toy dataset"""
function getToyData(n=10, nU=2, nX=5)
    hyperparams = getHyperParameters()
    X = rand(n, nX)
    binaryT::Array{Bool,1} = collect(rand(n) .< 0.5)
    realT::Array{Float64,1} = rand(n)

    objectCounts = [floor(Int64, n / nU) for _ in 1:nU]
    if n % nU > 0
        objectCounts = vcat(objectCounts, [n % nU])
    end
    hyperparams["SigmaU"] = generateSigmaU(objectCounts)
    return hyperparams, n, nU, nX, X, binaryT, realT
end