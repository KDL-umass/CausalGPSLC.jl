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

logmeanexp(x) = logsumexp(x)-log(length(x))
# -

Random.seed!(1234)
experiment = 48
# experiment = ARGS[1]

config_path = "../experiments/config/synthetic/$(experiment).toml"
config = TOML.parsefile(config_path)

data_config_path = config["paths"]["data"]
SigmaU, U_, T_, X_, Y_, epsY, ftxu = generate_synthetic_confounder(data_config_path)
nX = size(X_)[2]
n = length(T_)
X = [X_[:, i] for i in 1:nX]
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
Ycfs = Dict()

mask = Dict()

if maximum(T_) == 1.0
    T = [Bool(t) for t in T_]
    doTs = [true, false]
    binary = true
else
    T = T_
    doTnSteps = 20
    lower = minimum(T) * 1.05
    upper = maximum(T) * 0.95
    
    doTstepSize = (upper - lower)/doTnSteps

    doTs = [doT for doT in lower:doTstepSize:upper]
    binary = false
end

for doT in doTs
    MeanITEs[doT] = []
    CovITEs[doT] = []
    mask[doT] = T .!= doT
    Ycfs[doT] = ftxu(fill(Float64(doT), length(T)), X_, U_, epsY)[mask[doT]]
end
# -

# Run Estimation
if config["model"]["type"] == "GP_per_object"
    for i in burnIn:stepSize:nOuter
        for doT in doTs
            MeanITE = zeros(n)
            CovITE = zeros(n, n)
            for (j, objectIndex) in enumerate(objectIndeces)
                post = load("../experiments/" * config["paths"]["posterior_dir"] * "Object$(j)Posterior$(i).jld")
                xyLS = convert(Array{Float64,1}, post["xyLS"])

                MeanITE[objectIndex], CovITE[objectIndex, objectIndex] = conditionalITE(nothing, 
                                                                                      post["tyLS"],
                                                                                      xyLS,
                                                                                      post["yNoise"],
                                                                                      post["yScale"],
                                                                                      nothing,
                                                                                      [x[objectIndex] for x in X],
                                                                                      T[objectIndex],
                                                                                      Y[objectIndex],
                                                                                      doT)
            end
            push!(MeanITEs[doT], MeanITE[mask[doT]])
            push!(CovITEs[doT], CovITE[mask[doT], mask[doT]])
        end
    end        
else
    for i in burnIn:stepSize:nOuter
        post = load("../experiments/" * config["paths"]["posterior_dir"] * "Posterior$(i).jld")
        xyLS = convert(Array{Float64,1}, post["xyLS"])

        if config["model"]["type"] == "no_confounding"
            uyLS = nothing
            U = nothing
        else
            uyLS = convert(Array{Float64,1}, post["uyLS"])
            U = post["U"]
        end


        for doT in doTs
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
            push!(MeanITEs[doT], MeanITE[mask[doT]])
            push!(CovITEs[doT], CovITE[mask[doT], mask[doT]])
        end
    end
end
# + {}
# Generate Samples and compute log likelihoods
estIntSamples = Dict()
truthLogLikelihoods = Dict()
llhScores = Dict()

lower_bound = 0.025
upper_bound = 0.975
eps = 1e-10
nSamplesPerPost = 100


for doT in doTs
    estIntSamples[doT] = []

    truthLogLikelihoods[doT] = []
    
    for (i, MeanITE) in enumerate(MeanITEs[doT])
        CovITE = Symmetric(CovITEs[doT][i])
        MeanSATE = mean(MeanITE)
        VarSATE = mean(CovITE)
        for j in 1:nSamplesPerPost
            push!(estIntSamples[doT], normal(MeanSATE + mean(Y[mask[doT]]), VarSATE))
        end
        
        truthLogLikelihood = Distributions.logpdf(MvNormal(MeanITE + Y[mask[doT]], CovITE + I*eps), Ycfs[doT])/length(Ycfs[doT])
        push!(truthLogLikelihoods[doT], truthLogLikelihood)
    end
    llhScores[doT] = logmeanexp([llh for llh in truthLogLikelihoods[doT]])
    
end

estIntMean  = [mean(estIntSamples[doT]) for doT in doTs]
estIntLower = [quantile(estIntSamples[doT], lower_bound) for doT in doTs]
estIntUpper = [quantile(estIntSamples[doT], upper_bound) for doT in doTs]
println()
# -

# Generate Dose-Response Curve. Continuous Treatment Only
if !binary
    YcfMean = [mean(Ycfs[doT]) for doT in doTs]
    
    scatter(T, Y, c=U_[:, 1], label="Observational Data")
    estColor = "black"
    truthColor = "grey"
    plot(doTs, YcfMean, c=truthColor, label="Truth")
    plot(doTs, estIntMean, c=estColor, label="Estimated")
    plot(doTs, estIntUpper, linestyle="--", c=estColor)
    plot(doTs, estIntLower, linestyle="--", c=estColor)
    legend()
    xlabel("T")
    ylabel("Y")
    savefig("synthetic_results/$(experiment).png")
end
println(config["model"]["type"])

# +
# Compute precision in heterogenous treatment effects
pehe = Dict()

if binary
    for doT in doTs
        YcfEst = mean([MeanITE + Y[mask[doT]] for MeanITE in MeanITEs[doT]])
        error = Ycfs[doT] - YcfEst
        println(mean(error.^2))
        pehe[string(doT)] = (mean(error.^2)).^0.5
    end
    save("synthetic_results/pehe$(experiment).jld", pehe)
else
    for doT in doTs
        YcfEst = mean(MeanITE + Y for MeanITE in MeanITEs[doT])
        error = Ycfs[doT] - YcfEst
        pehe[doT] = (mean(error.^2)).^0.5
    end
    save("synthetic_results/pehe$(experiment).jld", "pehe", mean(values(pehe)))
    mean(values(pehe))
end

# +
# Report Log Likelihood score

save("synthetic_results/llh$(experiment).jld", "llhScore", mean(values(llhScores)))
mean(values(llhScores))
# + {}
# Summarize scores for table.


set = 12


for experiment in (4*set-3):(4*set)
# for experiment in 1:44
    config_path = "../experiments/config/synthetic/$(experiment).toml"
    config = TOML.parsefile(config_path)
    data_config_path = config["paths"]["data"]
    data_config = TOML.parsefile(data_config_path)
    Random.seed!(1234)
    SigmaU, U_, T_, X_, Y_, epsY, ftxu = generate_synthetic_confounder(data_config_path)
    
    if maximum(T_) == 1.0
        treatmentType = "Binary"
        pehe_ = load("synthetic_results/pehe$(experiment).jld")
        pehe = pehe_["true"] * mean(T_ .== false) + pehe_["false"] * mean(T_ .== true)
    else
        treatmentType = "Continuous"
        pehe = load("synthetic_results/pehe$(experiment).jld")["pehe"]
    end
    
    if length(data_config["data"]["TYAssignment"]) == 1
        if length(data_config["data"]["TYparams"]["poly"]) == 2
            functionType = "Linear"
        else
            functionType = "Poly"
        end
    else
        functionType = "Poly_Sin"
    end
    
    aggOpp = data_config["data"]["YaggOp"]
    modelType = config["model"]["type"]

    
    llhScore = load("synthetic_results/llh$(experiment).jld")["llhScore"]
    pehe = round(pehe, digits=1)
    llhScore = round(llhScore, digits=1)
    

    
    println("$(modelType), $(aggOpp), $(functionType), $(treatmentType), $(pehe), $(llhScore)")
end
# -



