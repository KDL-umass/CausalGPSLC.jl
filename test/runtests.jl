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
rng = Random.seed!(1234)

# utility functions for getting test data shared between various tests
include("test_data.jl")
include("test_model.jl")

# Adjust file paths for CI
prefix = ""
notInCI = pwd()[end-3:end] != "test"
if notInCI
    prefix = "test/"
end

# Mock ProgressBars
Mocking.activate()
patch = @patch function ProgressBars.tqdm(x)
    return x
end

Mocking.apply(patch) do
    # include("estimation.jl")
    # include("utils.jl")
    # include("kernel.jl")
    # include("model.jl")
    # include("inference.jl")

    if notInCI # leaving intense tests out of ci pipeline
        # Bayesian Workflow -> A guide on writing Bayes code + tests
        # https://arxiv.org/pdf/2011.01808.pdf
        # include("sbc.jl")
        include("comparison.jl")
    end
end

end