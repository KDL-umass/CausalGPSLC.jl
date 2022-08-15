```@meta
CurrentModule = CausalGPSLC
```
Gaussian Processes with Structured Latent Confounders
=====================================================

`CausalGPSLC.jl` is a Julia package for semi-parametric causal effect estimation with structured latent confounding. It provides interfaces for performing causal inference over the latent variables and Gaussian process parameters to produce accurate causal effect estimates.

The original GP-SLC paper can be found here: [http://proceedings.mlr.press/v119/witty20a/witty20a.pdf](http://proceedings.mlr.press/v119/witty20a/witty20a.pdf).

```@contents
Pages=["index.md"]
Depth = 3
```

# Inference

```@docs
gpslc
```

The primary struct that we provide interfaces for is the
[`GPSLCObject`](@ref), which most of the high-level functions like the 
[treatment effect](#Treatment-Effects) functions take as input 
in addition to their other arguments.

# Treatment Effects

## Individual Treatment Effect (ITE)

A contribution of the original GP-SLC paper is to produce accurate individual treatment effect estimates, conditioned on the observed data and using the inferred values of the latent confounders as determined by the provided structure.

```@docs
sampleITE
```

## Sample Average Treatment Effect (SATE)

Another popular and useful treatment effect estimate is SATE, 
which averages individual treatment effects over the individuals in the sample.

```@docs
sampleSATE
```

## Counterfactual Effects

It can be helpful to calculate treatment effect estimates for the whole
domain of treatment values in the data, or some subset, as in the [example](#Examples)
below. For this we can use `predictCounterfactualEffects`, which also
tracks the values of the `doT` intervention values for comparison.

```@docs
predictCounterfactualEffects
```

## Summarizing

It can be helpful to summarize the inferred individual treatment effects
and sample average treatment effects into mean and credible intervals.
A use case for this is demonstrated in the [examples](#Examples) section.

```@docs
summarizeEstimates
```

# Examples

The examples below illustrate use cases for
* setting hyperparameters, 
* performing inference,
* saving inference results,
* predicting counterfactual effects,
* calculating sample average treatment effect (SATE),
* computing credible intervals for SATE,
* and plotting those intervals relative to the original data

## New England Energy Consumption

This example creates an example plot of the NEEC treatment vs outcome data. Plots the original and the intervened data together. The example below is similar to Figure 3 in the original GP-SLC paper.

```@example
import Random # hide
Random.seed!(1234) # hide
using CausalGPSLC # hide
using Plots # hide
using Statistics # hide

# set hyperparameters
hyperparams = getHyperParameters()
hyperparams.nOuter = 25
hyperparams.nU = 2
hyperparams.nMHInner = 3
hyperparams.nESInner = 5

# run inference
dataFile = "../example_data/NEEC_sampled.csv"
g = gpslc(dataFile; hyperparams=hyperparams)
saveGPSLCObject(g, "exampleGPSLCObject")

# collect counterfactual outcomes
maIdx = vec(g.obj .== "MA")
nSamples = 100
ite, doT = predictCounterfactualEffects(g, nSamples)
maITE = ite[:, maIdx, :]

# get credible interval on counterfactual outcomes
sate = mean(maITE, dims=2)[:, 1, :]
interval = summarizeEstimates(sate)

meanOutcome = mean(g.Y[maIdx])
lowerSATE = interval[!, "LowerBound"]
meanSATE = interval[!, "Mean"]
upperSATE = interval[!, "UpperBound"]

# plot outcomes and credible interval
treatmentScale = 100
outcomeScale = 10

# observed data
plot(legend=:outertopright, size=(750, 400), margin=0.5Plots.cm, dpi=600)
obsT = g.T[maIdx] .* treatmentScale
obsY = g.Y[maIdx] .* outcomeScale
scatter!(obsT, obsY, label="MA obs", markershape=:circle)

# counterfactual
Tcf = doT .* treatmentScale
Ycf = (meanOutcome .+ meanSATE) .* outcomeScale
upper = (upperSATE .- meanSATE) .* outcomeScale
lower = (meanSATE .- lowerSATE) .* outcomeScale

plot!(Tcf, Ycf, label="MA cf", color=:green,
    ribbon=(lower, upper))

xlabel!("Temperature °F")
ylabel!("Energy Consumption (GWh)")
title!("Energy Consumption for Massachusetts")
```

Above we can see the Gaussian Process using individual treatment effect
estimates to predict the energy consumption (outcome) from the temperature (treatment) for Massachusetts. The shaded region is a 90% credible interval from the samples taken by [`predictCounterfactualEffects`](@ref) and processed by [`summarizeEstimates`](@ref) which calculates the credible intervals by computing the 5_th_ and 95_th_ percentiles of the samples.

The data used in this example can be found [here](../example_data/NEEC_sampled.csv).

# Types

## External Types

External types are those relevant for using `CausalGPSLC.jl` in a high-level way, 
where inference and prediction are automatically performed.

### HyperParameters

```@docs
HyperParameters
```

The default values for [`HyperParameters`](@ref) are provided by

```@docs
getHyperParameters
```

### PriorParameters

```@docs
PriorParameters
```

The default values for [`PriorParameters`](@ref) are provided by

```@docs
getPriorParameters
```

### GPSLCObject

The `GPSLCObject` is the high-level Julia struct that most of the externally facing
interfaces rely on to perform their operations. Since it contains inference samples,
the hyperparameters, and the observed data, it is at the center of all post-inference
interfaces that manipulate the posterior samples according to the observed data. 

This also means that the `GPSLCObject` contains the result of a large portion 
of compute time, as well as contains all the relevant data for a given workflow. 
For this reason, all the fields and functions that utilize it are externally available,
and described below, to provide users with a simple way to extend the functionality 
of `CausalGPSLC.jl` and estimate other quantities of interest.

```@docs
GPSLCObject
```

#### Retrieving meta-values from a `GPSLCObject`

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

#### Saving and loading `GPSLCObject`s

`GPSLCObject`s contain all the posterior samples, which can be intensive to calculate and can be reused for various estimations, we provide a pair of interfaces to save and load the `GPSLCObjects` from the filesystem.

```@docs
saveGPSLCObject
```

```@docs
loadGPSLCObject
```

## Internal Types

These types are are used in internal inference procedures,
so if users need to fine tune or modify the inference procedure, 
or access the model directly, these will be relevant.

### Confounders

```@docs
Confounders
```

### Covariates

```@docs
Covariates
```

### Treatment

```@docs
Treatment
```

### Outcome

```@docs
Outcome
```