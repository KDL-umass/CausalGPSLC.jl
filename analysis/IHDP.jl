# +
using Gen
using LinearAlgebra
using PyPlot
using TOML
using JLD
using CSV
using DataFrames
using Statistics
using Distributions
using Random

include("../src/model.jl")
include("../src/estimation.jl")
include("../src/inference.jl")
include("../data/processing_IHDP.jl")

using .Model
using .Estimation
using .Inference
using .ProcessingIHDP
logmeanexp(x) = logsumexp(x)-log(length(x))

# +
experiment = 3
experiments = [i for i in experiment*10-9:experiment*10]

config_paths = ["../experiments/config/IHDP/$(experiment).toml" for experiment in experiments]
configs = [TOML.parsefile(config_path) for config_path in config_paths]
config = configs[1]
println(config["model"]["type"])

# +
# Regenerate fixed data


Random.seed!(config["data_params"]["seed"])

data = DataFrame(CSV.File(config["paths"]["data"]))[1:config["data_params"]["nData"], :]
pairs = generatePairs(data, config["data_params"]["pPair"])
nData = size(data)[1]
n = nData + length(pairs)

SigmaU = generateSigmaU(pairs, nData)
T_ = generateT(data, pairs)
T = [Bool(t) for t in T_]
doT = [Bool(1-t) for t in T_]
X_ = generateX(data, pairs)
nX = size(X_)[2]
X = [X_[!, i] for i in 1:nX]
U = generateU(data, pairs)
BetaX, BetaU = generateWeights(config["data_params"]["weights"], config["data_params"]["weightsP"])
Y_, Ycf = generateOutcomes(X_, U, T_, BetaX, BetaU, config["data_params"]["CATT"], n)
Y = [Float64(y) for y in Y_]
println()
# -

# Test that all loaded outcomes are the same as generated.
Ys = [load("../experiments/" * config["paths"]["posterior_dir"] * "/Y.jld")["Y"] for config in configs]
println()

# +
nOuter = 500
burnIn = 100
stepSize = 10

nChains = 10
nPostPerChain = 500

# Treatment effect on the treated

MeanITEs = Dict()
CovITEs = Dict()

MeanITEs[true] = []
MeanITEs[false] = []
CovITEs[true] = []
CovITEs[false] = []

for i in 1:nChains
    for j in burnIn:stepSize:nOuter
        post = load("../experiments/" * configs[i]["paths"]["posterior_dir"] * "Posterior$(j).jld")
        xyLS = convert(Array{Float64,1}, post["xyLS"])
        
        if config["model"]["type"] == "no_confounding"
            uyLS = nothing
            U = nothing
        else
            uyLS = convert(Array{Float64,1}, post["uyLS"])
            U = post["U"]
        end
        
        for doT in [true, false]
            MeanITE, CovITE = conditionalITE(uyLS, 
                                              post["tyLS"],
                                              xyLS,
                                              post["yNoise"],
                                              post["yScale"],
                                              U,
                                              X,
                                              T,
                                              Y,
                                              doT)
            push!(MeanITEs[doT], MeanITE)
            push!(CovITEs[doT], CovITE)
        end
    end
end

# +
# Compute precision in heterogenous treatment effects
pehe = Dict()

for doT in [true, false]
    mask = T .!= doT
    YcfEst = mean([MeanITE[mask] + Y[mask] for MeanITE in MeanITEs[doT]])
    error = Ycf[mask] - YcfEst
    println(mean(error.^2))
    pehe[string(doT)] = (mean(error.^2)).^0.5
end
save("IHDP_results/pehe$(experiment).jld", pehe)

# +
# Compute likelihood of true counterfactual outcome. 
# ind is the likelihood of all individuals. 
# agg is computed as the likelihood of the conditional average treatment effect.

truthLogLikelihoods = Dict()
scores = Dict()

for doT in [true, false]
    truthLogLikelihoods[doT] = []
    mask = T .!= doT
    for (i, MeanITE) in enumerate(MeanITEs[doT])
        CovITE = CovITEs[doT][i]
        truthLogLikelihood = Distributions.logpdf(MvNormal(MeanITE[mask] + Y[mask], Symmetric(CovITE[mask, mask])), Ycf[mask])
        push!(truthLogLikelihoods[doT], truthLogLikelihood)
    end
#   Normalize the log density by the number of instances within each group
    scores[string(doT)] = logmeanexp([Real(llh/sum(mask)) for llh in truthLogLikelihoods[doT]])
end
# scores
save("IHDP_results/llh$(experiment).jld", scores)
# -

println(load("IHDP_results/pehe1.jld"))
println(load("IHDP_results/pehe2.jld"))
println(load("IHDP_results/pehe3.jld"))

println(load("IHDP_results/llh1.jld"))
println(load("IHDP_results/llh2.jld"))
println(load("IHDP_results/llh3.jld"))


