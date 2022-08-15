module CausalGPSLCTests

using Revise
using Mocking
using Gen
using Test
using LinearAlgebra
using ProgressBars
using DataFrames
using FunctionalCollections
using Distributions
using Statistics
using HypothesisTests
using CausalGPSLC
import CSV

import Random
rng = Random.seed!(1234)

# utility functions for getting test data shared between various tests
include("test_data.jl")
include("test_model.jl")
include("test_utils.jl")

# Long running intense inference tests inappropriate for CI load
runIntenseTests = false

# Adjust file paths for CI
prefix = ""
inCI = pwd()[end-3:end] == "test" # leaving intense tests out of ci pipeline
if !inCI
    prefix = "test/"
end

# Mock ProgressBars
Mocking.activate()
patch = @patch function ProgressBars.tqdm(x)
    return x
end

Mocking.apply(patch) do
    # Bayesian Workflow -> A guide on writing Bayes code + tests
    # https://arxiv.org/pdf/2011.01808.pdf
    if runIntenseTests
        include("posterior.jl")
        include("sbc.jl")
    end

    include("utils.jl")
    include("data.jl")
    include("estimation.jl")
    include("kernel.jl")
    include("model.jl")
    include("inference.jl")
    include("driver.jl")
    include("gpslc.jl")
    include("io.jl")
    include("prediction.jl")
end

end