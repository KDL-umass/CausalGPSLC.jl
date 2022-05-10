using Mocking
export gpslc, samplePosterior, sampleITE, ITEsamples, SATEsamples, summarizeITE

"""
    gpslc

Run posterior inference on the input data

Returns a [`GPSLCObject`](@ref) which stores the 
hyperparameters, priorparameters, data, and posterior samples.
"""
function gpslc(data::Union{DataFrame,String};
    hyperparams::HyperParameters=getHyperParameters(),
    priorparams::PriorParameters=getPriorParameters()
)::GPSLCObject
    X, T, Y, SigmaU = prepareData(data)
    GPSLCObject(hyperparams, priorparams, SigmaU, X, T, Y)
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

`ITEsamples`: `n x m` matrix where `n` is the number of data, and `m` is the number of samples
"""
function sampleITE(g::GPSLCObject; doT::Intervention=0.6, samplesPerPosterior::Int64=10)
    n = getN(g)
    nU = getNU(g)
    burnIn = g.hyperparams.nBurnIn
    stepSize = g.hyperparams.stepSize
    nOuter = g.hyperparams.nOuter
    # output in Algorithm 3
    ITEsamples = zeros(n, samplesPerPosterior * length(burnIn:stepSize:nOuter))

    idx = 1
    for i in @mock tqdm(burnIn:stepSize:nOuter)
        uyLS = zeros(nU)
        U = zeros(n, nU)
        for u in 1:nU
            uyLS[u] = g.posteriorSamples[i][:uyLS=>u=>:LS]
            U[:, u] = g.posteriorSamples[i][:U=>u=>:U]
        end
        U = toMatrix(U, n, nU)
        @assert size(U) == (n, nU)

        if g.X === nothing
            xyLS = nothing
        else
            nX = getNX(g)
            xyLS = zeros(nX)
            for k in 1:nX
                xyLS[k] = g.posteriorSamples[i][:xyLS=>k=>:LS]
            end
        end

        MeanITE, CovITE = conditionalITE(
            uyLS,
            xyLS,
            g.posteriorSamples[i][:tyLS],
            g.posteriorSamples[i][:yNoise],
            g.posteriorSamples[i][:yScale],
            U, g.X, g.T, g.Y, doT)

        for _ in 1:samplesPerPosterior
            dist = Distributions.MvNormal(
                MeanITE,
                LinearAlgebra.Symmetric(CovITE) + I * (1e-10)
            )
            samples = rand(dist)
            ITEsamples[:, idx] = samples
            idx += 1
        end
    end
    return ITEsamples
end


"""Individual Treatment Effect Samples"""
function ITEsamples(MeanITEs, CovITEs, nSamplesPerMixture)
    nMixtures = length(MeanITEs[:, 1])
    n = length(MeanITEs[1, :])

    samples = zeros(nMixtures * nSamplesPerMixture, n)
    i = 0
    for j in 1:nMixtures
        mean = MeanITEs[j]
        cov = CovITEs[j]
        for _ in 1:nSamplesPerMixture
            i += 1
            samples[i, :] = mvnormal(mean, cov)
        end
    end
    return samples
end

"""Sample Average Treatment Effect samples"""
function SATEsamples(MeanSATEs, VarSATEs, nSamplesPerMixture)
    nMixtures = length(MeanSATEs)

    samples = zeros(nMixtures * nSamplesPerMixture)

    i = 0
    for j in 1:nMixtures
        mean = MeanSATEs[j]
        var = VarSATEs[j]
        for _ in 1:nSamplesPerMixture
            i += 1
            samples[i] = normal(mean, var)
        end
    end
    return samples
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


