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
include("../data/synthetic.jl")

using .Model
using .Estimation
using .Inference
using .Synthetic
# -

experiment = 1

config_path = "../experiments/config/collider/$(experiment).toml"
config = TOML.parsefile(config_path)

# Generate synthetic data
Random.seed!(1234)
data_config_path = config["paths"]["data"]
SigmaU, U_, T_, X, Y_, epsY, ftx = generate_synthetic_collider(data_config_path)
n = length(T_)
println()

# Test that loaded outcomes are the same as generated
Y = load("../experiments/" * config["paths"]["posterior_dir"] * "/Y.jld")["Y"]
mean(Y_ - Y)

data_config = TOML.parsefile(data_config_path)
obj_size = data_config["data"]["obj_size"]
nObjects = Int(n/obj_size)
objectIndeces = [[j for j in ((i-1)*obj_size + 1):(i * obj_size)] for i in 1:nObjects]
println()

# +
nOuter = 5000
burnIn = 1000
stepSize = 100

MeanITEs = Dict()
CovITEs = Dict()
MeanITEsNoU = Dict()
CovITEsNoU = Dict()

mask = Dict()

T = T_
doTnSteps = 20
lower = minimum(T) * 1.05
upper = maximum(T) * 0.95

doTstepSize = (upper - lower)/doTnSteps

doTs = [doT for doT in lower:doTstepSize:upper]


for doT in doTs
    MeanITEs[doT] = []
    CovITEs[doT] = []
    MeanITEsNoU[doT] = []
    CovITEsNoU[doT] = []
    mask[doT] = T .!= doT
end
# -

# Run Estimation
for i in burnIn:stepSize:nOuter
    post = load("../experiments/" * config["paths"]["posterior_dir"] * "Posterior$(i).jld")

    uyLS = convert(Array{Float64,1}, post["uyLS"])
    
#     println(uyLS)

    for doT in doTs
        MeanITE, CovITE = conditionalITE(uyLS, 
                                          post["tyLS"],
                                          nothing,
                                          post["yNoise"],
                                          post["yScale"],
                                          post["U"],
                                          nothing,
                                          T,
                                          Y,
                                          doT)
        
        
        push!(MeanITEs[doT], MeanITE[mask[doT]])
        push!(CovITEs[doT], CovITE[mask[doT], mask[doT]])
    end
end

for i in burnIn:stepSize:nOuter
    post = load("../experiments/" * config["paths"]["posterior_dir"] * "Posterior$(i).jld")


    for doT in doTs
        MeanITE, CovITE = conditionalITE(nothing, 
                                          post["tyLS"],
                                          nothing,
                                          post["yNoise"],
                                          post["yScale"],
                                          nothing,
                                          nothing,
                                          T,
                                          Y,
                                          doT)
        
        
        push!(MeanITEsNoU[doT], MeanITE[mask[doT]])
        push!(CovITEsNoU[doT], CovITE[mask[doT], mask[doT]])
    end
end

# +
# Generate Samples

function generateSamples(MeanITEs, CovITEs, doTs)
    estIntSamples = Dict()

    lower_bound = 0.025
    upper_bound = 0.975
    eps = 1e-10
    nSamplesPerPost = 1000

    for doT in doTs
        estIntSamples[doT] = []

        for (i, MeanITE) in enumerate(MeanITEs[doT])
            CovITE = Symmetric(CovITEs[doT][i])
            MeanSATE = mean(MeanITE)
            VarSATE = mean(CovITE)
            for j in 1:nSamplesPerPost
                push!(estIntSamples[doT], normal(MeanSATE + mean(Y[mask[doT]]), VarSATE))
            end
        end
    end

    estIntMean  = [mean(estIntSamples[doT]) for doT in doTs]
    estIntLower = [quantile(estIntSamples[doT], lower_bound) for doT in doTs]
    estIntUpper = [quantile(estIntSamples[doT], upper_bound) for doT in doTs]
    return estIntMean, estIntLower, estIntUpper
end
# -

estIntMean, estIntLower, estIntUpper = generateSamples(MeanITEs, CovITEs, doTs)
estIntMeanNoU, estIntLowerNoU, estIntUpperNoU = generateSamples(MeanITEsNoU, CovITEsNoU, doTs)
println()

# +
# OLS with a collider
# Added noise to stabilize the matrix inverse
eps = 1e-13
D = cat(T, U_, dims=2)
BetaCollider = (((D' * D) + eps*I)\D') * Y
SATEsCollider = [BetaCollider[1] * doT for doT in doTs]

# OLS without the collider
BetaNoCollider = (((T' * T) + eps*I) \ T') * Y
SATEsNoCollider = [BetaNoCollider * doT for doT in doTs]

truths = [mean(ftx([doT], X, epsY)) for doT in doTs]
println()

# +
scatter(T, Y, c=U_, alpha = 0.5)
plot(doTs, SATEsCollider .+ mean(Y), c="red", label="OLS w/ Collider")
plot(doTs, SATEsNoCollider .+ mean(Y), c="blue", label="OLS w/out Collider")
plot(doTs, truths, c="grey", label="Truth")
plot(doTs, estIntMean, c="black", label="GPROC w/ U")
plot(doTs, estIntLower, c="black", linestyle="--")
plot(doTs, estIntUpper, c="black", linestyle="--")

plot(doTs, estIntMeanNoU, c="purple", label="GPROC w/out U")
plot(doTs, estIntLowerNoU, c="purple", linestyle="--")
plot(doTs, estIntUpperNoU, c="purple", linestyle="--")
legend()
xlim(-3, 3)
# ylim(-2.5, 2.5)
xlabel("T")
ylabel("Y")
savefig("collider_results/collider.png")
# -



