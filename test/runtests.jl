module GPSLCTests

using Revise
using Mocking
using Gen
using Test
using ProgressBars
using DataFrames
using FunctionalCollections
using Distributions
using HypothesisTests
using GPSLC
import CSV

import Random
rng = Random.seed!(0)

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

Mocking.apply(patch) do
    include("kernel.jl")
    include("latent.jl")
    include("model.jl")
    include("utils.jl")
    include("comparison.jl")
    # Bayesian Workflow -> A guide on writing Bayes code + tests
    # https://arxiv.org/pdf/2011.01808.pdf
    include("sbc.jl")
end

end