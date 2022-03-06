module GPSLCTests

using GPSLC
using Test

import Random
Random.seed!(0)

include("comparison.jl")
include("kernel.jl")
include("latent.jl")
include("sbc.jl")
include("utils.jl")``

end