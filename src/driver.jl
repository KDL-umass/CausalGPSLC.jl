using Mocking
export samplePosterior, sampleITE, summarizeITE

"""
    Draw samples from the posterior given the observed data `X,T,Y`.

Params:
- `X`: Input covariates
- `T`: Input treatment
- `Y`: Output
- `SigmaU`: Object structure
- `hyperParams`=[`getHyperParameters`](@ref)`()`: Hyperparameters for posterior sampling

Returns:

`posteriorSample`: samples from posterior determined by hyperparameters
"""
function samplePosterior(X, T, Y, SigmaU; hyperparams::Dict{String,Any}=getHyperParameters(), nU::Int64=1, nOuter::Int64=25, nMHInner::Int64=1, nESInner::Int64=1)
    hyperparams["SigmaU"] = SigmaU # databased hyperparameter
    posteriorSample, _ = Posterior(hyperparams, X, T, Y, nU, nOuter, nMHInner, nESInner)
    return posteriorSample
end

"""
    Estimate Individual Treatment Effect with GPSLC model

Params:
- `X`: Input covariates
- `T`: Input treatment
- `Y`: Output
- `SigmaU`: Object structure
- `posteriorSample`=[`samplePosterior`](@ref)`(X,T,Y,SigmaU)`: Samples of parameters from posterior

Returns:

`ITEsamples`: `n x m` matrix where `n` is the number of data, and `m` is the number of samples
"""
function sampleITE(X::Union{Covariates,Nothing}, T::Treatment, Y::Outcome, SigmaU;
    posteriorSample=samplePosterior(X, T, Y, SigmaU),
    doT::Intervention=0.6, nU::Int64=1, nOuter::Int64=25,
    burnIn::Int64=10, stepSize::Int64=1, samplesPerPost::Int64=10)

    n = size(T, 1)
    # output in Algorithm 3
    ITEsamples = zeros(n, samplesPerPost * length(burnIn:stepSize:nOuter))

    idx = 1
    for i in @mock tqdm(burnIn:stepSize:nOuter)
        uyLS = zeros(nU)
        U = zeros(n, nU)
        for u in 1:nU
            uyLS[u] = posteriorSample[i][:uyLS=>u=>:LS]
            U[:, u] = posteriorSample[i][:U=>u=>:U]
        end
        U = toMatrix(U, n, nU)
        @assert size(U) == (n, nU)

        if X === nothing
            xyLS = nothing
        else
            nX = size(X, 2)
            xyLS = zeros(nX)
            for k in 1:nX
                xyLS[k] = posteriorSample[i][:xyLS=>k=>:LS]
            end
        end

        MeanITE, CovITE = conditionalITE(
            uyLS,
            xyLS,
            posteriorSample[i][:tyLS],
            posteriorSample[i][:yNoise],
            posteriorSample[i][:yScale],
            U, X, T, Y, doT)

        for _ in 1:samplesPerPost
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
