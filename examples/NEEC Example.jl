using Revise
using Random
Random.seed!(1234)

using GPSLC
using Plots

export NEEC_Example

"""
    NEEC_Example

Creates an example plot of the NEEC treatment vs outcome data.

Plots the original and the intervened data together.
"""
function NEEC_Example(dataFile="examples/data/NEEC_sampled.csv")
    g = gpslc(dataFile)

    println("Estimating ITE")
    ite = sampleITE(g, doT=0.6) GPS

    s = summarizeITE(ite; savetofile="examples/results/NEEC_sampled_0.6.csv")

    plot(legend=:outertopright, size=(750, 400))
    for o in unique(g.obj)
        idx = vec(g.obj .== o)
        scatter!(g.T[idx], g.Y[idx], label="$(o) original", markershape=:circle)
        scatter!(g.T[idx], s[!, "Mean"][idx], label="$(o) do(T=.6)", markershape=:diamond)
    end
    xlabel!("Treatment")
    ylabel!("Outcome")
    title!("NEEC Energy consumption data")
end

NEEC_Example()