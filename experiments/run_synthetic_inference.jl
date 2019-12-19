import TOML
using Gen
using Serialization

include("../src/model.jl")
include("../src/inference.jl")
include("../data/synthetic.jl")

using .Model
using .Inference
using .Synthetic

experiment = ARGS[1]

config_path = "../experiments/config/synthetic/$(experiment).toml"
config = TOML.parsefile(config_path)

# Generate synthetic data
data_config_path = config["paths"]["data_config"]
SigmaU, U, T, X, Y, epsY, ftxu = generate_synthetic_confounder(data_config_path)

# Process X
X_ = [X[i, :] for i in 1:size(X, 1)]
Xcol = collect(Iterators.product(X_, X_))
nX = length(X[1])

data_config = TOML.parsefile(data_config_path)
if data_config["data"]["Ttype"] == "continuous"
    inference = ContinuousPosterior
end

if data_config["data"]["Ttype"] == "binary"
    inference = BinaryPosterior
end

# Specify hyperparameters
hyperparameters = config["model_hyperparameters"]

if config["model"]["SigmaU"] == "correct"
    hyperparameters["SigmaU"] = SigmaU
end

# TODO Add more options here (for incorrect sigmaU).

# Specify kernels
kernel_key = Dict()
kernel_key["rbf"] = rbfKernel
kernel_key["lin"] = linearKernel

for kernel in ["utKernel", "xtKernel", "uyKernel", "xyKernel", "tyKernel"]
    hyperparameters[kernel] = kernel_key[config["model_hyperparameters"][kernel]]
end

nOuter = config["inference"]["nOuter"]
nMHInner = config["inference"]["nMHInner"]
nESInner = config["inference"]["nESInner"]

PosteriorSamples, trace = inference(hyperparameters, T, Y, Xcol, nX, nOuter, nMHInner, nESInner)

open(config["paths"]["posterior_output"], "w") do io
    serialize(io, PosteriorSamples)
end