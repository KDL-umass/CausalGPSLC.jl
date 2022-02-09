module GPSLC

using CSV
using DataFrames
using LinearAlgebra
using ProgressBars
using Statistics
using Distributions

println("Loading Inference")
include("inference.jl")
using .Inference

println("Loading Estimation")
include("estimation.jl")
using .Estimation

println("Loading Utilities")
include("utils.jl")
using .Utils

export prepareData, getHyperParameters, sampleITE, samplePosterior, summarizeITE

function prepareData(csv_path)
    df = CSV.read(csv_path, DataFrame)

    # build a list of object size
    # [a, a, a, b, c, c] -> [3, 1, 2]
    counts = Dict()
    for o in df[!, :obj]
        if o in keys(counts)
            counts[o] += 1
        else
            counts[o] = 1
        end
    end
    obj_count = [counts[o] for o in uniq(df[!, :obj])]

    # generate a block matrix based on object counts.
    # SigmaU is shorthand for the object structure of the latent confounder.
    SigmaU = generateSigmaU(obj_count)

    # prepare inputs
    T = Array(df[!, :T])
    Y = Array(df[!, :Y])

    cols = names(df)
    cols = deleteat!(cols, cols .== "T")
    cols = deleteat!(cols, cols .== "Y")
    cols = deleteat!(cols, cols .== "obj")
    if length(cols) == 0
        X = nothing
    else
        X_ = df[!, cols]
        nX = size(X_)[2]
        X = [Array(X_[!, i]) for i in 1:nX]
    end

    X, T, Y, SigmaU
end

"""
*Hyperparameters*

Defaults are those used in original paper, listed here for modification

- `uNoiseShape::Float64=4.0`: the shape parameter of the prior inv gamma over the noise of U
- `uNoiseScale::Float64=4.0`: the scale parameter of the prior inv gamma over the noise of U
- `xNoiseShape::Float64=4.0`: the shape parameter of the prior inv gamma over the noise of X
- `xNoiseScale::Float64=4.0`: the scale parameter of the prior inv gamma over the noise of X
- `tNoiseShape::Float64=4.0`: the shape parameter of the prior inv gamma over the noise of T
- `tNoiseScale::Float64=4.0`: the scale parameter of the prior inv gamma over the noise of T
- `yNoiseShape::Float64=4.0`: the shape parameter of the prior inv gamma over the noise of Y
- `yNoiseScale::Float64=4.0`: the scale parameter of the prior inv gamma over the noise of Y
- `xScaleShape::Float64=4.0`: the shape parameter of the prior inv gamma over kernel scale of X
- `xScaleScale::Float64=4.0`: the scale parameter of the prior inv gamma over kernel scale of X
- `tScaleShape::Float64=4.0`: the shape parameter of the prior inv gamma over kernel scale of T
- `tScaleScale::Float64=4.0`: the scale parameter of the prior inv gamma over kernel scale of T
- `yScaleShape::Float64=4.0`: the shape parameter of the prior inv gamma over kernel scale of Y
- `yScaleScale::Float64=4.0`: the scale parameter of the prior inv gamma over kernel scale of Y
- `uxLSShape::Float64=4.0`: the shape parameter of the prior inv gamma over kernel lengthscale of U and X
- `uxLSScale::Float64=4.0`: the scale parameter of the prior inv gamma over kernel lengthscale of U and X
- `utLSShape::Float64=4.0`: the shape parameter of the prior inv gamma over kernel lengthscale of U and T
- `utLSScale::Float64=4.0`: the scale parameter of the prior inv gamma over kernel lengthscale of U and T
- `xtLSShape::Float64=4.0`: the shape parameter of the prior inv gamma over kernel lengthscale of X and T
- `xtLSScale::Float64=4.0`: the scale parameter of the prior inv gamma over kernel lengthscale of X and T
- `uyLSShape::Float64=4.0`: the shape parameter of the prior inv gamma over kernel lengthscale of U and Y
- `uyLSScale::Float64=4.0`: the scale parameter of the prior inv gamma over kernel lengthscale of U and Y
- `xyLSShape::Float64=4.0`: the shape parameter of the prior inv gamma over kernel lengthscale of X and Y
- `xyLSScale::Float64=4.0`: the scale parameter of the prior inv gamma over kernel lengthscale of X and Y
- `tyLSShape::Float64=4.0`: the shape parameter of the prior inv gamma over kernel lengthscale of T and Y
- `tyLSScale::Float64=4.0`: the scale parameter of the prior inv gamma over kernel lengthscale of T and Y
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

Returns:

`ITEsamples`: `n x m` matrix where `n` is the number of data, and `m` is the number of samples

"""
function sampleITE(X, T, Y, SigmaU; posteriorsample = samplePosterior(X, T, Y, SigmaU),
    doT::Float64 = 0.6, nU::Int = 1, nOuter::Int = 25,
    burnIn::Int = 10, stepSize::Int = 1, samplesPerPost::Int = 10
)
    ITEsamples = zeros(length(T), samplesPerPost * length(burnIn:stepSize:nOuter)) # output in Algorithm 3
    idx = 1
    for i in tqdm(burnIn:stepSize:nOuter)
        uyLS = []
        U = []
        for u in 1:nU
            push!(uyLS, posteriorsample[i][:uyLS=>u=>:LS])
            push!(U, posteriorsample[i][:U=>u=>:U])
        end

        if X === nothing
            xyLS = nothing
        else
            xyLS = convert(Array{Float64,1}, posteriorsample[i][:xyLS])
        end
        uyLS = convert(Array{Float64,1}, uyLS)

        MeanITE, CovITE = conditionalITE(uyLS,
            posteriorsample[i][:tyLS],
            xyLS,
            posteriorsample[i][:yNoise],
            posteriorsample[i][:yScale],
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
"""
function samplePosterior(X, T, Y, SigmaU; hyperparams::Dict{String,Any} = getHyperParameters(),
    nU::Int = 1, nOuter::Int = 25, nMHInner::Int = 1, nESInner::Int = 1
)
    hyperparams["SigmaU"] = SigmaU # databased hyperparameter
    posteriorsample, _ = Posterior(hyperparams, X, T, Y, nU, nOuter, nMHInner, nESInner)
    return posteriorsample
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

end
