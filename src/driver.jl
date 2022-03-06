using GPSLC

using CSV
using DataFrames
using LinearAlgebra
using ProgressBars
using Statistics
using Distributions

export getHyperParameters, sampleITE, samplePosterior, summarizeITE

"""
*Hyperparameters*

Defaults are those used in original paper, listed here for modification

- `uNoiseShape::Float64=4.0`: shape parameter of the InvGamma prior over the noise of U
- `uNoiseScale::Float64=4.0`: scale parameter of the InvGamma prior over the noise of U
- `xNoiseShape::Float64=4.0`: shape parameter of the InvGamma prior over the noise of X
- `xNoiseScale::Float64=4.0`: scale parameter of the InvGamma prior over the noise of X
- `tNoiseShape::Float64=4.0`: shape parameter of the InvGamma prior over the noise of T
- `tNoiseScale::Float64=4.0`: scale parameter of the InvGamma prior over the noise of T
- `yNoiseShape::Float64=4.0`: shape parameter of the InvGamma prior over the noise of Y
- `yNoiseScale::Float64=4.0`: scale parameter of the InvGamma prior over the noise of Y
- `xScaleShape::Float64=4.0`: shape parameter of the InvGamma prior over kernel scale of X
- `xScaleScale::Float64=4.0`: scale parameter of the InvGamma prior over kernel scale of X
- `tScaleShape::Float64=4.0`: shape parameter of the InvGamma prior over kernel scale of T
- `tScaleScale::Float64=4.0`: scale parameter of the InvGamma prior over kernel scale of T
- `yScaleShape::Float64=4.0`: shape parameter of the InvGamma prior over kernel scale of Y
- `yScaleScale::Float64=4.0`: scale parameter of the InvGamma prior over kernel scale of Y
- `uxLSShape::Float64=4.0`: shape parameter of the InvGamma prior over kernel lengthscale of U and X
- `uxLSScale::Float64=4.0`: scale parameter of the InvGamma prior over kernel lengthscale of U and X
- `utLSShape::Float64=4.0`: shape parameter of the InvGamma prior over kernel lengthscale of U and T
- `utLSScale::Float64=4.0`: scale parameter of the InvGamma prior over kernel lengthscale of U and T
- `xtLSShape::Float64=4.0`: shape parameter of the InvGamma prior over kernel lengthscale of X and T
- `xtLSScale::Float64=4.0`: scale parameter of the InvGamma prior over kernel lengthscale of X and T
- `uyLSShape::Float64=4.0`: shape parameter of the InvGamma prior over kernel lengthscale of U and Y
- `uyLSScale::Float64=4.0`: scale parameter of the InvGamma prior over kernel lengthscale of U and Y
- `xyLSShape::Float64=4.0`: shape parameter of the InvGamma prior over kernel lengthscale of X and Y
- `xyLSScale::Float64=4.0`: scale parameter of the InvGamma prior over kernel lengthscale of X and Y
- `tyLSShape::Float64=4.0`: shape parameter of the InvGamma prior over kernel lengthscale of T and Y
- `tyLSScale::Float64=4.0`: scale parameter of the InvGamma prior over kernel lengthscale of T and Y
"""
function getHyperParameters()
    Dict{String,Any}(
        "uNoiseShape" => 4.0,
        "uNoiseScale" => 4.0,
        "xNoiseShape" => 4.0,
        "xNoiseScale" => 4.0,
        "tNoiseShape" => 4.0,
        "tNoiseScale" => 4.0,
        "yNoiseShape" => 4.0,
        "yNoiseScale" => 4.0,
        "xScaleShape" => 4.0,
        "xScaleScale" => 4.0,
        "tScaleShape" => 4.0,
        "tScaleScale" => 4.0,
        "yScaleShape" => 4.0,
        "yScaleScale" => 4.0,
        "uxLSShape" => 4.0,
        "uxLSScale" => 4.0,
        "utLSShape" => 4.0,
        "utLSScale" => 4.0,
        "xtLSShape" => 4.0,
        "xtLSScale" => 4.0,
        "uyLSShape" => 4.0,
        "uyLSScale" => 4.0,
        "xyLSShape" => 4.0,
        "xyLSScale" => 4.0,
        "tyLSShape" => 4.0,
        "tyLSScale" => 4.0,
    )
end


"""
Estimate Individual Treatment Effect using GPSLC model

Params:
- `X`: Input covariates
- `T`: Input treatment
- `Y`: Output
- `SigmaU`: Object structure
- `posteriorSample`=[`samplePosterior`](@ref)`(X,T,Y,SigmaU)`: Samples of parameters from posterior

Returns:

`ITEsamples`: `n x m` matrix where `n` is the number of data, and `m` is the number of samples
"""
function sampleITE(X, T, Y, SigmaU; posteriorSample = samplePosterior(X, T, Y, SigmaU),
    doT::Float64 = 0.6, nU::Int = 1, nOuter::Int = 25,
    burnIn::Int = 10, stepSize::Int = 1, samplesPerPost::Int = 10; verbose = true
)
    if !verbose
        tqdm = loop(x) = x
    end
    ITEsamples = zeros(length(T), samplesPerPost * length(burnIn:stepSize:nOuter)) # output in Algorithm 3
    idx = 1
    for i in tqdm(burnIn:stepSize:nOuter)
        uyLS = []
        U = []
        for u in 1:nU
            push!(uyLS, posteriorSample[i][:uyLS=>u=>:LS])
            push!(U, posteriorSample[i][:U=>u=>:U])
        end

        if X === nothing
            xyLS = nothing
        else
            xyLS = convert(Array{Float64,1}, posteriorSample[i][:xyLS])
        end
        uyLS = convert(Array{Float64,1}, uyLS)

        MeanITE, CovITE = conditionalITE(uyLS,
            posteriorSample[i][:tyLS],
            xyLS,
            posteriorSample[i][:yNoise],
            posteriorSample[i][:yScale],
            U,
            X,
            T,
            Y,
            doT)

        for j in 1:samplesPerPost
            samples = rand(MvNormal(MeanITE, Symmetric(CovITE) + I * (1e-10)))
            ITEsamples[:, idx] = samples
            idx += 1
        end
    end
    return ITEsamples
end

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
function samplePosterior(X, T, Y, SigmaU; hyperparams::Dict{String,Any} = getHyperParameters(),
    nU::Int = 1, nOuter::Int = 25, nMHInner::Int = 1, nESInner::Int = 1, verbose = true
)
    hyperparams["SigmaU"] = SigmaU # databased hyperparameter
    posteriorSample, _ = Posterior(hyperparams, X, T, Y, nU, nOuter, nMHInner, nESInner; verbose = verbose)
    return posteriorSample
end

"""
Create dataframe of mean, lower and upper quantiles of the ITE samples.

Params:
- `ITEsamples`: `n x m` array of ITE samples
- `savetofile`: Optionally save the resultant dataframe as CSV to the filename passed.

Returns:
- `df`: Dataframe of Individual, Mean, LowerBound, and UpperBound values for the samples.
"""
function summarizeITE(ITEsamples; savetofile::String = "")
    meanITE = mean(ITEsamples, dims = 2)[:, 1]
    lowerITE = broadcast(quantile, [ITEsamples[i, :] for i in 1:size(ITEsamples)[1]], 0.05)
    upperITE = broadcast(quantile, [ITEsamples[i, :] for i in 1:size(ITEsamples)[1]], 0.95)
    df = DataFrame(Individual = 1:size(meanITE)[1], Mean = meanITE, LowerBound = lowerITE, UpperBound = upperITE)
    if savetofile != ""
        CSV.write(savetofile, df)
        println("Saved ITE mean and 90% credible intervals to " * savetofile)
    end
    return df
end
