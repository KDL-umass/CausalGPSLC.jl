@show LOAD_PATH
push!(LOAD_PATH, "@stdlib")
import Pkg
Pkg.add("Documenter")

using Documenter, GPSLC

makedocs(sitename = "GPSLC")

deploydocs(
    repo = "github.com/KDL-umass/GPSLC.jl.git",
    devbranch = "main"
)
