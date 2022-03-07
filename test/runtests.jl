module GPSLCTests

using Mocking
using Test
using GPSLC
using ProgressBars
using DataFrames
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
end