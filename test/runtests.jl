module GPSLCTests

using Revise
using Mocking
using Gen
using Test
using LinearAlgebra
using ProgressBars
using DataFrames
using FunctionalCollections
using Distributions
using HypothesisTests
using GPSLC
import CSV

import Random
rng = Random.seed!(0)

# utility functions for getting test data shared between various tests
include("test_data.jl")
include("test_model.jl")

# Adjust file paths
prefix = ""
if pwd()[end-3:end] != "test"
    prefix = "test/"
end

# Mock ProgressBars
Mocking.activate()
patch = @patch function ProgressBars.tqdm(x)
    return x
end

Mocking.apply(patch) do
    include("sbc.jl")
    include("utils.jl")
    include("kernel.jl")
    include("model.jl")
    include("inference.jl")
    include("comparison.jl")
    # Bayesian Workflow -> A guide on writing Bayes code + tests
    # https://arxiv.org/pdf/2011.01808.pdf
end

end