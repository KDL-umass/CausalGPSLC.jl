import Random # hide
Random.seed!(1234) # hide
using GPSLC # hide
using Plots # hide
using Statistics # hide

# run inference
dataFile = "docs/example_data/NEEC_sampled.csv"
g = gpslc(dataFile)

# collect counterfactual outcomes
nsamples = 30
numPosteriorSamples = getNumPosteriorSamples(g)
o = "MA"
idx = vec(g.obj .== o)
ite, doT = predictCounterfactualEffects(g, nsamples; fidelity=10)
YcfMA = mean(g.Y[idx]) .+ ite[:, idx, :]

# get credible interval on counterfactual outcomes
sate = mean(YcfMA, dims=2)[:, 1, :]
s = summarizeEstimates(sate)

# plot outcomes and credible interval
order = sortperm(doT)
plot(legend=:outertopright, size=(750, 400))
scatter!(g.T[idx] .* 100, g.Y[idx] .* 10, label="$(o) obs", markershape=:circle)
plot!(doT[order] .* 100, s[!, "Mean"][order] .* 10,
    ribbon=(s[!, "LowerBound"][order], s[!, "UpperBound"][order]),
    label="$(o) cf", color=:green)

xlabel!("Temperature °F")
ylabel!("Energy Consumption")
title!("Energy Consumption for Massachusetts")