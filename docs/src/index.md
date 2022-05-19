# GPSLC.jl

```@meta
CurrentModule = GPSLC
```

```@contents
Pages=["index.md"]
Depth = 3
```

```@docs
gpslc
```

The primary struct that we provide interfaces for is the
[`GPSLCObject`](@ref), which most of the high-level functions like the 
[treatment effect](#treatment-effects) functions take as input 
in addition to their other arguments.

# Treatment Effects

## Individual Treatment Effect (ITE)

A contribution of the original GPSLC paper is to produce accurate individual treatment effect conditioned on observed data, using inferred values of latent confounders determined by given structure.

```@docs
sampleITE
```

## Sample Average Treatment Effect (SATE)

Another popular and useful treatment effect estimate is SATE.

```@docs
sampleSATE
```

## Counterfactual Effects

It can be helpful to calculate treatment effect estimates for the whole
domain of treatment values in the data, as in the [example](#examples)
below. For this we can use `predictCounterfactualEffects`, which also
tracks the values of the `doT` intervention values for comparison.

```@docs
predictCounterfactualEffects
```

## Summarizing

It can be helpful to summarize the inferred individual treatment effects
and sample average treatment effects
into mean and credible intervals.

```@docs
summarizeEstimates
```

# Examples

The example below is similar to Figure 3 in the original GPSLC paper. 

## New England Energy Consumption

This example creates an example plot of the NEEC treatment vs outcome data. Plots the original and the intervened data together.

```@example
import Random # hide
Random.seed!(1234) # hide
using GPSLC # hide
using Plots # hide
using Statistics # hide

# run inference
dataFile = "../example_data/NEEC_sampled.csv"
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
```

Above we can see the Gaussian Process using individual treatment effect
estimates to predict the energy consumption (outcome) from the temperature (treatment) for Massachusetts. The shaded region is a 90% credible interval from the samples taken by [`predictCounterfactualEffects`](@ref) and processed by [`summarizeEstimates`](@ref) which calculates the credible intervals by computing the 5_th_ and 95_th_ percentiles of the samples.

The data can be found [here](../example_data/NEEC_sampled.csv).

# Types

## External Types

Relevant types for using GPSLC.jl in a high-level way, where 
inference and prediction are automatically performed are described below.

```@docs
HyperParameters
```

The default values for [`HyperParameters`](@ref) are provided by

```@docs
getHyperParameters
```

```@docs
PriorParameters
```

The default values for [`PriorParameters`](@ref) are provided by

```@docs
getPriorParameters
```

### GPSLCObject

```@docs
GPSLCObject
```

Helpful functions for retrieving meta values from a `GPSLCObject`

```@docs
getN
```
```@docs
getNX
```
```@docs
getNU
```
```@docs
getNumPosteriorSamples
```

## Internal Types

These types are are used in internal inference procedures,
so if you need to fine tune the inference, or access the posterior functions directly,
these will be relevant to your work.

```@docs
Confounders
```

```@docs
Covariates
```

```@docs
Treatment
```

```@docs
Outcome
```