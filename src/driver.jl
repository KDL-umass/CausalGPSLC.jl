using Mocking
export gpslc, samplePosterior, sampleITE, sampleSATE, summarizeEstimates

"""
    gpslc(filename * ".csv")
    gpslc(DataFrame(X1=...,X2=...,T=...,Y=...,obj=...))

Run posterior inference on the input data.

Datatypes of DataFrame or CSV must follow these standards:
    
- `T` (Boolean/Float64)
- `Y` (Float64)
- `X1...XN` (Float64...Float64)
- `obj` (Any)

Optional parameters
- `hyperparams::`[`HyperParameters`](@ref)=[`getHyperParameters`](@ref)(): Hyper parameters primarily define the high level amount of inference to perform.
- `priorparams::`[`PriorParameters`](@ref)=[`getPriorParameters`](@ref)(): Prior parameters define the high level priors to draw from when constructing kernel functions and latent confounder structure.

Returns a [`GPSLCObject`](@ref) which stores the 
hyperparameters, prior parameters, data, and posterior samples.
"""
function gpslc(data::Union{DataFrame,String};
    hyperparams::HyperParameters=getHyperParameters(),
    priorparams::PriorParameters=getPriorParameters()
)::GPSLCObject
    SigmaU, obj, X, T, Y = prepareData(data)
    GPSLCObject(hyperparams, priorparams, SigmaU, obj, X, T, Y)
end

function gpslc(obj::Union{ObjectLabels,Nothing}, X::Union{Covariates,Nothing}, T::Treatment, Y::Outcome; hyperparams::HyperParameters=getHyperParameters(),
    priorparams::PriorParameters=getPriorParameters()
)::GPSLCObject
    if obj !== nothing
        SigmaU = generateSigmaU(obj, priorparams["sigmaUNoise"], priorparams["sigmaUCov"])
    else
        SigmaU = nothing
    end
    GPSLCObject(hyperparams, priorparams, SigmaU, obj, X, T, Y)
end


"""
    samplePosterior(hyperparameters, priorparameters, SigmaU, X, T, Y)

Draw samples from the posterior given the observed data.

Params: 
- `hyperparams::`[`HyperParameters`](@ref): 
- `priorparams::`[`PriorParameters`](@ref): 
- `SigmaU::`[`ConfounderStructure`](@ref): 
- `X::`[`Covariates`](@ref): 
- `T::`[`Treatment`](@ref): 
- `Y::`[`Outcome`](@ref): 

Posterior samples are returned as a Vector of Gen choicemaps.
"""
function samplePosterior(hyperparams, priorparams, SigmaU, X, T, Y)
    # avoid storing SigmaU twice in g
    priorparams["SigmaU"] = SigmaU # databased priorparameter 
    posteriorSamples, _ = Posterior(
        priorparams, X, T, Y,
        hyperparams.nU,
        hyperparams.nOuter,
        hyperparams.nMHInner,
        hyperparams.nESInner)
    return posteriorSamples
end


"""
    sampleITE(g, doT)
    sampleITE(g, doT; samplesPerPosterior=10)
Estimate Individual Treatment Effect with GPSLC model

Params:
- `g::`[`GPSLCObject`](@ref): Contains data and hyperparameters
- `doT`: The requested intervention (e.g. set all treatments to 1.0)
- `samplesPerPosterior`: How many ITE samples to draw per posterior sample in `g`.

Returns:

`ITEsamples`: `n x m` matrix where `n` is the number of individuals, and `m` is the number of samples.
"""
function sampleITE(g::GPSLCObject, doT::Intervention; samplesPerPosterior::Int64=10)
    MeanITEs, CovITEs = ITEDistributions(g, doT)
    ITEsamples(MeanITEs, CovITEs, samplesPerPosterior)
end


"""
    sampleSATE(g, doT)
    sampleSATE(g, doT; samplesPerPosterior=10)
Estimate Sample Average Treatment Effect with GPSLC model

Using [`sampleITE`](@ref), samples can be drawn for the sample average treatment effect

Params:
- `g::`[`GPSLCObject`](@ref): Contains data and hyperparameters
- `doT`: The requested intervention (e.g. set all treatments to 1.0)
- `samplesPerPosterior`: How many samples to draw per posterior sample in `g`.

Returns:

`SATEsamples`: `n x m` matrix where `n` is the number of individuals, and `m` is the number of samples.
"""
function sampleSATE(g::GPSLCObject, doT::Intervention; samplesPerPosterior::Int64=10)
    MeanSATEs, CovSATEs = SATEDistributions(g, doT)
    SATEsamples(MeanSATEs, CovSATEs, samplesPerPosterior)
end


"""
    summarizeEstimates(samples)
    summarizeEstimates(samples; savetofile="ite_samples.csv")
Summarize Predicted Estimates (Counterfactual Outcomes or Individual Treatment Effects)

Create dataframe of mean, lower and upper quantiles of the samples from [`sampleITE`](@ref) or [`predictCounterfactualEffects`](@ref).

Params:
- `samples`: `n x m` array of samples
- `savetofile`: Optionally save the resultant dataframe as CSV to the filename passed.

Returns:
- `df`: Dataframe of Individual, Mean, LowerBound, and UpperBound values for the credible intervals around the sample.
"""
function summarizeEstimates(samples; savetofile::String="")
    Mean = mean(samples, dims=2)[:, 1]
    LowerBound = broadcast(quantile,
        [samples[i, :] for i in 1:size(samples)[1]], 0.05)
    UpperBound = broadcast(quantile,
        [samples[i, :] for i in 1:size(samples)[1]], 0.95)
    df = DataFrame(Individual=1:size(Mean)[1], Mean=Mean, LowerBound=LowerBound, UpperBound=UpperBound)
    if savetofile != ""
        CSV.write(savetofile, df)
        println("Saved ITE mean and 90% credible intervals to " * savetofile)
    end
    return df
end
