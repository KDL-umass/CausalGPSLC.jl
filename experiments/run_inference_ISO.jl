import TOML
using Gen
using Serialization
using LinearAlgebra
using JLD
using Random
using CSV
using DataFrames
Random.seed!(1234)
include("../src/model.jl")
include("../src/inference.jl")
include("../data/processing_iso.jl")

using .Model
using .Inference
using .ProcessingISO

experiment = ARGS[1]

config_path = "../experiments/config/ISO/$(experiment).toml"
config = TOML.parsefile(config_path)

# +
# Load and process data
df = DataFrame(CSV.File(config["paths"]["data"]))
weekday_df = df[df[!, :IsWeekday] .== "TRUE", :]

importanceWeights = generateImportanceWeights(config["new_means"], config["new_vars"], weekday_df)
T, Y, SigmaU, regions_key = resampleData(config["subsample_params"]["nSamplesPerState"], importanceWeights, weekday_df)
# -

# Scale T and Y
T /= 100
Y /= 1000

# Load inference hyperparameters
nOuter = config["inference"]["nOuter"]
nMHInner = config["inference"]["nMHInner"]
nESInner = config["inference"]["nESInner"]

# +
# Run Inference
hyperparameters = config["model_hyperparameters"]

nU = config["model"]["nU"]
n = length(T)

# Full GPROC Model
if config["model"]["type"] == "correct"
    hyperparameters["SigmaU"] = SigmaU
    PosteriorSamples, _ = Posterior(hyperparameters, nothing, T, Y, nU, nOuter, nMHInner, nESInner)
end

# No latent confounding
if config["model"]["type"] == "no_confounding"
    PosteriorSamples, _ = Posterior(hyperparameters, nothing, T, Y, nothing, nOuter, nothing, nothing)
end

# No object information
if config["model"]["type"] == "no_objects"
    hyperparameters["SigmaU"] = Matrix{Float64}(I, n, n)
    PosteriorSamples, _ = Posterior(hyperparameters, nothing, T, Y, nU, nOuter, nMHInner, nESInner)
end

# GP model per group
if config["model"]["type"] == "GP_per_object"
    PosteriorSamplesDict = Dict()
    
    regions = [region for region in Set(regions_key)]
    for region in regions
        in_region = (regions_key .== region)
        
        PosteriorSamplesDict[region], _ = Posterior(hyperparameters, nothing, T[in_region], Y[in_region], 
                                                 nothing, nOuter, nothing, nothing)
    end
end

# +
# Save Inference Results
if config["model"]["type"] in ["correct", "no_objects"]
    for i in 1:nOuter
        uyLS = []
        U = []

        tyLS = PosteriorSamples[i][:tyLS]
        yNoise = PosteriorSamples[i][:yNoise]
        yScale = PosteriorSamples[i][:yScale]

        for u in 1:nU
            push!(uyLS, PosteriorSamples[i][:uyLS => u => :LS])
            push!(U, PosteriorSamples[i][:U => u => :U])
        end

        save(config["paths"]["posterior_dir"] * "Posterior$(i).jld", "U", U, 
                                                                     "uyLS", uyLS, 
                                                                     "tyLS", tyLS, 
                                                                     "yNoise", yNoise,
                                                                     "yScale", yScale)
    end
end

if config["model"]["type"] == "no_confounding"
    for i in 1:nOuter
        tyLS = PosteriorSamples[i][:tyLS]
        yNoise = PosteriorSamples[i][:yNoise]
        yScale = PosteriorSamples[i][:yScale]
        
        save(config["paths"]["posterior_dir"] * "Posterior$(i).jld", "tyLS", tyLS, 
                                                                     "yNoise", yNoise,
                                                                     "yScale", yScale)
    end
end

if config["model"]["type"] == "GP_per_object"
    
    for region in regions
        for i in 1:nOuter
            tyLS = PosteriorSamplesDict[region][i][:tyLS]
            yNoise = PosteriorSamplesDict[region][i][:yNoise]
            yScale = PosteriorSamplesDict[region][i][:yScale]

            save(config["paths"]["posterior_dir"] * "$(region)Posterior$(i).jld", "tyLS", tyLS, 
                                                                                  "yNoise", yNoise,
                                                                                  "yScale", yScale)
        end
    end
end


save(config["paths"]["posterior_dir"] * "T.jld", "T", T)
save(config["paths"]["posterior_dir"] * "Y.jld", "Y", Y)
save(config["paths"]["posterior_dir"] * "regions_key.jld", "regions_key", regions_key)
# -


