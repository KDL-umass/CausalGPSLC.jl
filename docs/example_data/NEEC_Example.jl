import Random # hide
Random.seed!(1234) # hide
using CausalGPSLC # hide
using Plots # hide
using Statistics # hide

hyperparams = getHyperParameters()
hyperparams.nOuter = 100
hyperparams.nU = 2
hyperparams.nMHInner = 3
hyperparams.nESInner = 5

# run inference
dataFile = "docs/example_data/NEEC_sampled.csv"
g = gpslc(dataFile; hyperparams=hyperparams)
saveGPSLCObject(g, "example$(hyperparams.nOuter)-$(hyperparams.nU)")

# collect counterfactual outcomes
nsamples = 100
ite, doT = predictCounterfactualEffects(g, nsamples; fidelity=100)

idx = vec(g.obj .== "MA")
maITE = ite[:, idx, :]

# get credible interval on counterfactual outcomes
sate = mean(maITE, dims=2)[:, 1, :]
interval = summarizeEstimates(sate)
lowerSATE = interval[!, "LowerBound"]
meanSATE = interval[!, "Mean"]
upperSATE = interval[!, "UpperBound"]

# plot outcomes and credible interval
treatmentScale = 100
outcomeScale = 10


# observed data
plot(legend=:outertopright, size=(750, 400), margin=0.5Plots.cm, dpi=600)
T = g.T[idx] .* treatmentScale
Y = g.Y[idx] .* outcomeScale
scatter!(T, Y, label="MA obs", markershape=:circle)

# counterfactual
T = doT .* treatmentScale
meanOutcome = mean(g.Y[idx])
Y = (meanOutcome .+ meanSATE) .* outcomeScale
upper = (upperSATE .- meanSATE) .* outcomeScale
lower = (meanSATE .- lowerSATE) .* outcomeScale

plot!(T, Y, label="MA cf", color=:green,
    ribbon=(lower, upper))

xlabel!("Temperature °F")
ylabel!("Energy Consumption (GWh)")
title!("Energy Consumption for Massachusetts")
savefig("neec$(hyperparams.nOuter)-$(hyperparams.nU).png")