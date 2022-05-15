using Mocking
export gpslc, samplePosterior, sampleITE, ITEsamples, SATEsamples, summarizeITE

"""
    gpslc

Run posterior inference on the input data.

Datatypes of DataFrame or CSV must follow these standards:
    
- `T` (Boolean/Float64)
- `Y` (Float64)
- `X1...XN` (Float64...Float64)
- `obj` (Any)

Returns a [`GPSLCObject`](@ref) which stores the 
hyperparameters, priorparameters, data, and posterior samples.
"""
function gpslc(data::Union{DataFrame,String};
    hyperparams::HyperParameters=getHyperParameters(),
    priorparams::PriorParameters=getPriorParameters()
)::GPSLCObject
    SigmaU, obj, X, T, Y = prepareData(data)
    GPSLCObject(hyperparams, priorparams, SigmaU, obj, X, T, Y)
end


"""
    samplePosterior

Draw samples from the posterior given the observed data.

Params: 
- [`g::GPSLCObject`](@ref): The GPSLCObject that contains the data and hyperparameters.

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
    Estimate Individual Treatment Effect with GPSLC model

Params:
- `g::`[`GPSLCObject`](@ref): Contains data and hyperparameters
- `doT`: The recommended intervention (e.g. set all treatments to 1.0)
- `samplesPerPosterior`: How many ITE samples to draw per posterior sample from `g`.

Returns:

`ITEsamples`: `n x m` matrix where `n` is the number of individuals, and `m` is the number of samples.
"""
function sampleITE(g::GPSLCObject; doT::Intervention=0.6, samplesPerPosterior::Int64=10)
    MeanITEs, CovITEs = ITEDistributions(g, doT)
    ITEsamples(MeanITEs, CovITEs, samplesPerPosterior)
end


"""
    Estimate Sample Average Treatment Effect with GPSLC model

Params:
- `g::`[`GPSLCObject`](@ref): Contains data and hyperparameters
- `doT`: The recommended intervention (e.g. set all treatments to 1.0)
- `samplesPerPosterior`: How many ITE samples to draw per posterior sample from `g`.

Returns:

`SATEsamples`: `n x m` matrix where `n` is the number of individuals, and `m` is the number of samples.
"""
function sampleSATE(g::GPSLCObject; doT::Intervention=0.6, samplesPerPosterior::Int64=10)
    MeanSATEs, CovSATEs = conditionalSATE(g, doT)
    SATEsamples(MeanSATEs, CovSATEs, samplesPerPosterior)
end


"""
    Summarize Individual Treatment Estimates

Create dataframe of mean, lower and upper quantiles of the ITE samples.

Params:
- `ITEsamples`: `n x m` array of ITE samples
- `savetofile`: Optionally save the resultant dataframe as CSV to the filename passed.

Returns:
- `df`: Dataframe of Individual, Mean, LowerBound, and UpperBound values for the samples.
"""
function summarizeITE(ITEsamples; savetofile::String="")
    meanITE = mean(ITEsamples, dims=2)[:, 1]
    lowerITE = broadcast(quantile,
        [ITEsamples[i, :] for i in 1:size(ITEsamples)[1]], 0.05)
    upperITE = broadcast(quantile,
        [ITEsamples[i, :] for i in 1:size(ITEsamples)[1]], 0.95)
    df = DataFrame(Individual=1:size(meanITE)[1], Mean=meanITE, LowerBound=lowerITE, UpperBound=upperITE)
    if savetofile != ""
        CSV.write(savetofile, df)
        println("Saved ITE mean and 90% credible intervals to " * savetofile)
    end
    return df
end


