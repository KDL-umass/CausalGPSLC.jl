# Run from `docs/` directory

@show LOAD_PATH
push!(LOAD_PATH, "@stdlib")
push!(LOAD_PATH, "../src/")
push!(LOAD_PATH, "..")
import Pkg
Pkg.add("Documenter")

using Documenter, GPSLC

makedocs(sitename="GPSLC")

deploydocs(
    repo="github.com/KDL-umass/GPSLC.jl.git",
    devbranch="main"
)
