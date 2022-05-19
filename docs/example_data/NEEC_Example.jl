import Random # hide
Random.seed!(1234) # hide
using GPSLC # hide
using Plots # hide
using Statistics # hide

# run inference
dataFile = "docs/example_data/NEEC_sampled.csv"
hyperparams = getHyperParameters()
hyperparams.nOuter = 25
hyperparams.nU = 3
hyperparams.nMHInner = 3
hyperparams.nESInner = 5

g = gpslc(dataFile; hyperparams=hyperparams)

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
plot(legend=:outertopright, size=(750, 400), margin=0.5Plots.cm)
T = g.T[idx] .* treatmentScale
Y = g.Y[idx] .* outcomeScale
scatter!(T, Y, label="MA obs", markershape=:circle)

# counterfactual
T = doT .* treatmentScale
Y = interval[!, "Mean"] .* outcomeScale
plot!(T, Y, label="MA cf", color=:green,
    ribbon=(interval[!, "LowerBound"] .* outcomeScale,
        interval[!, "UpperBound"] .* outcomeScale
    ))

xlabel!("Temperature °F")
ylabel!("Energy Consumption (GWh)")
title!("Energy Consumption for Massachusetts")
# savefig("neec25.png")