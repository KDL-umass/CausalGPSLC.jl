import TOML
using Gen
using Serialization
using JLD
using Random
using LinearAlgebra
using Statistics
include("../src/model.jl")
include("../src/inference.jl")
include("../data/synthetic.jl")

using .Model
using .Inference
using .Synthetic

# experiment = ARGS[1]
experiment = 1

config_path = "../experiments/config/collider/$(experiment).toml"
config = TOML.parsefile(config_path)

# +
# Generate synthetic data
Random.seed!(1234)
data_config_path = config["paths"]["data"]
SigmaU, U, T, X, Y, epsY, ftx = generate_synthetic_collider(data_config_path)
n = length(T)

println()
# -

if maximum(X) == 0.0
    X = nothing
    nX = 0
else
    save(config["paths"]["posterior_dir"] * "X.jld", "X", X)
end

# Load inference hyperparameters
nOuter = config["inference"]["nOuter"]
nMHInner = config["inference"]["nMHInner"]
nESInner = config["inference"]["nESInner"]

data_config = TOML.parsefile(data_config_path)
obj_size = data_config["data"]["obj_size"]
nObjects = Int(length(T)/obj_size)
objectIndeces = [[j for j in ((i-1)*obj_size + 1):(i * obj_size)] for i in 1:nObjects]
println()

# +
# Run Inference
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
    if !isBinary
        nMHInner = nothing
        nESInner = nothing
    end
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

    if !isBinary
        nMHInner = nothing
        nESInner = nothing
    end

    for (object, objectIndex) in enumerate(objectIndeces)
        PosteriorSamplesDict[object], _ = Posterior(hyperparameters, [x[objectIndex] for x in X], T[objectIndex], Y[objectIndex],
                                                 nothing, nOuter, nMHInner, nESInner)
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


save(config["paths"]["posterior_dir"] * "T.jld", "T", T)
save(config["paths"]["posterior_dir"] * "Y.jld", "Y", Y)
save(config["paths"]["posterior_dir"] * "SigmaU.jld", "SigmaU", SigmaU)
save(config["paths"]["posterior_dir"] * "objectIndeces.jld", "objectIndeces", objectIndeces)
# -


