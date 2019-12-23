import TOML
using Gen
using Serialization
using Random
Random.seed!(1234)
using CSV
using DataFrames
include("../src/model.jl")
include("../src/inference.jl")
include("../data/semi_synthetic.jl")

using .Model
using .Inference
using .SemiSynthetic

experiment = ARGS[1]

config_path = "../experiments/config/semi_synthetic/$(experiment).toml"
config = TOML.parsefile(config_path)

# +
# Load and process data
df = DataFrame(CSV.File(config["paths"]["data"]))
weekday_df = df[df[!, :IsWeekday] .== "TRUE", :]

importanceWeights = generateImportanceWeights(config["new_means"], config["new_vars"], weekday_df)
T, Y, SigmaU, regions = resampleData(config["subsample_params"]["nSamplesPerState"], importanceWeights, weekday_df)

# +
# Load model hyperparameters
hyperparameters = config["model_hyperparameters"]

nU = config["model"]["nU"]

if config["model"]["SigmaU"] == "correct"
    hyperparameters["SigmaU"] = SigmaU
end
# TODO add more options here (for incorrect sigmaU)
# -

# Load inference hyperparameters
nOuter = config["inference"]["nOuter"]
nMHInner = config["inference"]["nMHInner"]
nESInner = config["inference"]["nESInner"]

PosteriorSamples, _ = Posterior(hyperparameters, nothing, T, Y, nU, nOuter, nMHInner, nESInner)

open(config["paths"]["posterior_output"], "w") do io
    serialize(io, PosteriorSamples)
end
