import Random # hide
Random.seed!(1234) # hide

using GPSLC
using Plots
using Statistics

dataFile = "docs/example_data/NEEC_sampled.csv"
g = gpslc(dataFile)

println("Estimating ITE")
range = 0.0:0.1:1.0

nsamples = 30
numPosteriorSamples = getNumPosteriorSamples(g)

ites = zeros(length(range), getN(g), nsamples * numPosteriorSamples)
for (i, doT) in enumerate(range)
    ites[i, :, :] = sampleITE(g, doT; samplesPerPosterior=nsamples)
end

ite = mean(ites, dims=1)[1, :, :]
s = summarizeITE(ite)

plot(legend=:outertopright, size=(750, 400))
o = "MA"
idx = vec(g.obj .== o)
scatter!(g.T[idx] .* 100, g.Y[idx] .* 10000, label="$(o) original", markershape=:circle)
scatter!(g.T[idx] .* 100, s[!, "Mean"][idx] .* 10000, label="$(o) doT", markershape=:diamond)

xlabel!("Temperature °F")
ylabel!("Energy Consumption")
title!("Energy Consumption for Massachusetts")