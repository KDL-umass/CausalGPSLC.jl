module GPSLC

using Gen
using DataFrames
using Distributions
using LinearAlgebra
using ProgressBars
using Mocking
using Statistics
import CSV
import FunctionalCollections

include("types.jl")
include("utils.jl")
include("data.jl")
include("hyperparameters.jl")

include("kernel.jl")
include("model_prior.jl")
include("model_likelihood.jl")
include("model.jl")
include("proposal.jl")

include("inference.jl")
include("likelihood.jl")
include("prediction.jl")
include("estimation.jl")
include("driver.jl")

end