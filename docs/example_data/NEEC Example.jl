import Random # hide
Random.seed!(1234) # hide
using GPSLC # hide
using Plots # hide
using Statistics # hide

# run inference
dataFile = "docs/example_data/NEEC_sampled.csv"
g = gpslc(dataFile)

# collect counterfactual outcomes
idx = vec(g.obj .== "MA")

ite, doT = predictCounterfactualEffects(g, 30)
Ycf = mean(g.Y[idx]) .+ ite[:, idx, :]

# get credible interval on counterfactual outcomes
sate = mean(Ycf, dims=2)[:, 1, :]
interval = summarizeEstimates(sate)

# plot outcomes and credible interval
treatmentScale = 100
outcomeScale = 10

# observed data
plot(legend=:outertopright, size=(750, 400))
T = g.T[idx] .* treatmentScale
Y = g.Y[idx] .* outcomeScale
scatter!(T, Y, label="MA obs", markershape=:circle)

# counterfactual
T = doT .* treatmentScale
Y = interval[!, "Mean"] .* outcomeScale
plot!(T, Y, label="MA cf", color=:green,
    ribbon=(interval[!, "LowerBound"], interval[!, "UpperBound"]))

xlabel!("Temperature °F")
ylabel!("Energy Consumption (GWh)")
title!("Energy Consumption for Massachusetts")