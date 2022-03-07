module GPSLCTests

using GPSLC
using Test
using Mocking
using ProgressBars

import Random
Random.seed!(0)

# Adjust file paths
prefix = ""
if pwd()[end-3:end] != "test"
    prefix = "test/"
end

# Mock ProgressBars
Mocking.activate()
patch = @patch tqdm(x::UnitRange{Int64}) = x

apply(patch) do
    include("comparison.jl")
    include("kernel.jl")
    include("latent.jl")
    include("sbc.jl")
    include("utils.jl")
end
end