module GPSLCTests

using Revise
using Mocking
using Test
using GPSLC
using ProgressBars
using DataFrames
using FunctionalCollections
import CSV

import Random
Random.seed!(0)

# Adjust file paths
prefix = ""
if pwd()[end-3:end] != "test"
    prefix = "test/"
end

# Mock ProgressBars
Mocking.activate()
patch = @patch function ProgressBars.tqdm(x)
    println("Mocking out progress bars")
    return x
end

apply(patch) do
    include("comparison.jl")
    include("kernel.jl")
    include("latent.jl")
    include("sbc.jl")
    include("utils.jl")
end

# Bayesian Workflow -> A guide on writing Bayes code + tests
# https://arxiv.org/pdf/2011.01808.pdf
end