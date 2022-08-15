# Run from `docs/` directory

@show LOAD_PATH
push!(LOAD_PATH, "@stdlib")
push!(LOAD_PATH, "../src/")
push!(LOAD_PATH, "..")
import Pkg
Pkg.add("Documenter");

using Documenter, CausalGPSLC

makedocs(
    sitename="CausalGPSLC",
    pages=[
        "Documentation" => "index.md",
        "Contributing" => "contributing.md"
    ],
    format=[
        # Documenter.LaTeX(), # uncomment to generate PDF
        Documenter.HTML()
    ]
)

deploydocs(
    repo="github.com/KDL-umass/CausalGPSLC",
    devbranch="main"
)
