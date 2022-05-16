import Random # hide
Random.seed!(1234) # hide

using GPSLC
using Plots
using Statistics

dataFile = "docs/example_data/NEEC_sampled.csv"
g = gpslc(dataFile)



nsamples = 30
numPosteriorSamples = getNumPosteriorSamples(g)

cfs = zeros(getN(g), getN(g), nsamples * numPosteriorSamples)

for (i, doT) in enumerate(g.T)
    cfs[i, :, :] = predictCounterfactualOutcomes(g, doT, nsamples)
end

ite = mean(cfs, dims=[2])[:, 1, :]
s = summarizeITE(ite)

plot(legend=:outertopright, size=(750, 400))
o = "MA"
agg = CSV.read("test/test_data/NEEC_aggregated.csv", DataFrame)
idx = vec(agg[!, "obj"] .== o)
scatter!(agg[idx, "T"], agg[idx, "Y"] .* 10, label="agg", makershape=:triangle)
idx = vec(g.obj .== o)
scatter!(g.T[idx] .* 100, g.Y[idx] .* 10, label="$(o) original", markershape=:circle)
scatter!(g.T[idx] .* 100, s[!, "Mean"][idx] .* 10, label="$(o) doT", markershape=:diamond)

xlabel!("Temperature °F")
ylabel!("Energy Consumption")
title!("Energy Consumption for Massachusetts")