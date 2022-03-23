import TOML
using Gen
using Serialization
using LinearAlgebra
using JLD
using Random
using CSV
using DataFrames
using Statistics
include("../src/model.jl")
include("../src/inference.jl")
include("../data/processing_IHDP.jl")

using .Model
using .Inference
using .ProcessingIHDP

experiment = ARGS[1]
# experiment = 1

config_path = "../experiments/config/IHDP/$(experiment).toml"
config = TOML.parsefile(config_path)

# +
# Set data generation seed.
Random.seed!(config["data_params"]["seed"])

data = DataFrame(CSV.File(config["paths"]["data"]))[1:config["data_params"]["nData"], :]
pairs = generatePairs(data, config["data_params"]["pPair"])
nData = size(data)[1]
n = nData + length(pairs)

SigmaU = generateSigmaU(pairs, nData)
T_ = generateT(data, pairs)
T = [Bool(t) for t in T_]
X_ = generateX(data, pairs)
nX = size(X_)[2]
X = [X_[!, i] for i in 1:nX]
U = generateU(data, pairs)
BetaX, BetaU = generateWeights(config["data_params"]["weights"], config["data_params"]["weightsP"])
Y_, Ycf = generateOutcomes(X_, U, T_, BetaX, BetaU, config["data_params"]["CATT"], n)
Y = [Float64(y) for y in Y_]

objectIndeces = [[i] for i in 1:nData]

for (i, pair) in enumerate(pairs)
    push!(objectIndeces[pair], i + nData)
end
# -

# Load inference hyperparameters
nOuter   = config["inference"]["nOuter"]
nMHInner = config["inference"]["nMHInner"]
nESInner = config["inference"]["nESInner"]

# +
# Run Inference

Random.seed!(config["inference"]["seed"])
hyperparameters = config["model_hyperparameters"]

nU = config["model"]["nU"]
n = length(T)

# Full GPROC Model
if config["model"]["type"] == "correct"
    hyperparameters["SigmaU"] = SigmaU
    PosteriorSamples, _ = Posterior(hyperparameters, X, T, Y, nU, nOuter, nMHInner, nESInner)
end

# No latent confounding
if config["model"]["type"] == "no_confounding"
    PosteriorSamples, _ = Posterior(hyperparameters, X, T, Y, nothing, nOuter, nMHInner, nESInner)
end

# No object information
if config["model"]["type"] == "no_objects"
    hyperparameters["SigmaU"] = Matrix{Float64}(I, n, n)
    PosteriorSamples, _ = Posterior(hyperparameters, X, T, Y, nU, nOuter, nMHInner, nESInner)
end

# GP model per group
if config["model"]["type"] == "GP_per_object"
    PosteriorSamplesDict = Dict()
    for (i, objectIndex) in enumerate(objectIndeces) 
        PosteriorSamplesDict[i], _ = Posterior(hyperparameters, [X[j][objectIndex] for j in 1:nX], T[objectIndex], 
                                               Y[objectIndex], nothing, nOuter, nMHInner, nESInner)
    end
end

# +
# Save Inference Results
if config["model"]["type"] in ["correct", "no_objects"]
    for i in 1:nOuter
        uyLS = []
        xyLS = []
        U = []

        tyLS = PosteriorSamples[i][:tyLS]
        yNoise = PosteriorSamples[i][:yNoise]
        yScale = PosteriorSamples[i][:yScale]
        
        for x in 1:nX
            push!(xyLS, PosteriorSamples[i][:xyLS => x => :LS])
        end
        
        for u in 1:nU
            push!(uyLS, PosteriorSamples[i][:uyLS => u => :LS])
            push!(U, PosteriorSamples[i][:U => u => :U])
        end

        save(config["paths"]["posterior_dir"] * "Posterior$(i).jld", "U", U, 
                                                                     "uyLS", uyLS,
                                                                     "xyLS", xyLS,
                                                                     "tyLS", tyLS, 
                                                                     "yNoise", yNoise,
                                                                     "yScale", yScale)
    end
end

if config["model"]["type"] == "no_confounding"
    for i in 1:nOuter
        xyLS = []
        
        tyLS = PosteriorSamples[i][:tyLS]
        yNoise = PosteriorSamples[i][:yNoise]
        yScale = PosteriorSamples[i][:yScale]
        
        for x in 1:nX
            push!(xyLS, PosteriorSamples[i][:xyLS => x => :LS])
        end
        
        save(config["paths"]["posterior_dir"] * "Posterior$(i).jld", "xyLS", xyLS,
                                                                     "tyLS", tyLS, 
                                                                     "yNoise", yNoise,
                                                                     "yScale", yScale)
    end
end

if config["model"]["type"] == "GP_per_object"
    for (i, objectIndex) in enumerate(objectIndeces)
        for j in 1:nOuter
            xyLS = []
            
            for x in 1:nX
                push!(xyLS, PosteriorSamplesDict[i][j][:xyLS => x => :LS])
            end
            
            tyLS = PosteriorSamplesDict[i][j][:tyLS]
            yNoise = PosteriorSamplesDict[i][j][:yNoise]
            yScale = PosteriorSamplesDict[i][j][:yScale]

            save(config["paths"]["posterior_dir"] * "Object$(i)Posterior$(j).jld", "xyLS", xyLS,
                                                                                  "tyLS", tyLS, 
                                                                                  "yNoise", yNoise,
                                                                                  "yScale", yScale)
        end
    end
end


save(config["paths"]["posterior_dir"] * "T.jld", "T", T_)
save(config["paths"]["posterior_dir"] * "Y.jld", "Y", Y)
save(config["paths"]["posterior_dir"] * "Ycf.jld", "Ycf", Ycf)
save(config["paths"]["posterior_dir"] * "X.jld", "X", X)
save(config["paths"]["posterior_dir"] * "SigmaU.jld", "SigmaU", SigmaU)
save(config["paths"]["posterior_dir"] * "objectIndeces.jld", "objectIndeces", objectIndeces)

# -


