using Pkg

Pkg.activate("GPSLCenv")

println("Adding packages")
using Random
using CSV
using DataFrames
using ArgParse
using DataFrames
using LinearAlgebra
using ProgressBars
using Statistics
using Distributions
Random.seed!(1234)

println("Loading Inference")
include("src/inference.jl")
using .Inference

println("Loading Estimation")
include("src/estimation.jl")
using .Estimation

println("Loading Utilities")
include("utils.jl")
using .Utils

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--datapath"
        help = "a path to the data"
        default = "data/NEEC_sampled.csv"
        arg_type = String
        "--output_filepath"
        help = "filepath for inference results"
        default = "results/NEEC_sampled_80.csv"
        arg_type = String
        "--doT"
        help = "treatment value to intervene"
        default = 0.8
        arg_type = Float64
        # posterior updates
        "--nOuter"
        help = "the number of posterior steps"
        default = 25
        arg_type = Int
        "--nMHInner"
        help = "the number of metropolis hastings sampling steps"
        default = 1
        arg_type = Int
        "--nESInner"
        help = "the number of elliptical slice sampling steps"
        default = 1
        arg_type = Int
        "--nU"
        help = "the dimension of latent confounders to model"
        default = 1
        arg_type = Int
        # inference
        "--burnIn"
        help = "the number of posterior samples for burn-in"
        default = 10
        arg_type = Int
        "--stepSize"
        help = "the step size during the inference step"
        default = 1
        arg_type = Int
        "--samplesPerPost"
        help = "the number of samples from each posterior for treatment effect approximation"
        default = 10
        arg_type = Int
        # parameters for priors
        "--uNoiseShape"
        help = "the shape parameter of the prior inv gamma over the noise of U"
        default = 4.0
        arg_type = Float64
        "--uNoiseScale"
        help = "the scale parameter of the prior inv gamma over the noise of U"
        default = 4.0
        arg_type = Float64
        "--xNoiseShape"
        help = "the shape parameter of the prior inv gamma over the noise of X"
        default = 4.0
        arg_type = Float64
        "--xNoiseScale"
        help = "the scale parameter of the prior inv gamma over the noise of X"
        default = 4.0
        arg_type = Float64
        "--tNoiseShape"
        help = "the shape parameter of the prior inv gamma over the noise of T"
        default = 4.0
        arg_type = Float64
        "--tNoiseScale"
        help = "the scale parameter of the prior inv gamma over the noise of T"
        default = 4.0
        arg_type = Float64
        "--yNoiseShape"
        help = "the shape parameter of the prior inv gamma over the noise of Y"
        default = 4.0
        arg_type = Float64
        "--yNoiseScale"
        help = "the scale parameter of the prior inv gamma over the noise of Y"
        default = 4.0
        arg_type = Float64
        "--xScaleShape"
        help = "the shape parameter of the prior inv gamma over kernel scale of X"
        default = 4.0
        arg_type = Float64
        "--xScaleScale"
        help = "the scale parameter of the prior inv gamma over kernel scale of X"
        default = 4.0
        arg_type = Float64
        "--tScaleShape"
        help = "the shape parameter of the prior inv gamma over kernel scale of T"
        default = 4.0
        arg_type = Float64
        "--tScaleScale"
        help = "the scale parameter of the prior inv gamma over kernel scale of T"
        default = 4.0
        arg_type = Float64
        "--yScaleShape"
        help = "the shape parameter of the prior inv gamma over kernel scale of Y"
        default = 4.0
        arg_type = Float64
        "--yScaleScale"
        help = "the scale parameter of the prior inv gamma over kernel scale of Y"
        default = 4.0
        arg_type = Float64
        "--uxLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of U and X"
        default = 4.0
        arg_type = Float64
        "--uxLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of U and X"
        default = 4.0
        arg_type = Float64
        "--utLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of U and T"
        default = 4.0
        arg_type = Float64
        "--utLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of U and T"
        default = 4.0
        arg_type = Float64
        "--xtLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of X and T"
        default = 4.0
        arg_type = Float64
        "--xtLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of X and T"
        default = 4.0
        arg_type = Float64
        "--uyLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of U and Y"
        default = 4.0
        arg_type = Float64
        "--uyLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of U and Y"
        default = 4.0
        arg_type = Float64
        "--xyLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of X and Y"
        default = 4.0
        arg_type = Float64
        "--xyLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of X and Y"
        default = 4.0
        arg_type = Float64
        "--tyLSShape"
        help = "the shape parameter of the prior inv gamma over kernel lengthscale of T and Y"
        default = 4.0
        arg_type = Float64
        "--tyLSScale"
        help = "the scale parameter of the prior inv gamma over kernel lengthscale of T and Y"
        default = 4.0
        arg_type = Float64
    end
    return parse_args(s)
end


function main()
    # parse argument
    println("Parsing Arguments")
    parsed_args = parse_commandline()
    csv_path = parsed_args["datapath"]
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
    parsed_args["SigmaU"] = SigmaU

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


    # running GPSLC
    nOuter = parsed_args["nOuter"]
    nMHInner = parsed_args["nMHInner"]
    nESInner = parsed_args["nESInner"]
    nU = parsed_args["nU"]

    println("Running Inference on U and Kernel Hyperparameters")
    posteriorsample, _ = Posterior(parsed_args, X, T, Y, nU, nOuter, nMHInner, nESInner)

    # inference of treatment effects
    burnIn = parsed_args["burnIn"]
    stepSize = parsed_args["stepSize"]
    samplesPerPost = parsed_args["samplesPerPost"]

    println("Estimating ITE")
    ITEsamples = zeros(length(T), samplesPerPost * length(burnIn:stepSize:nOuter)) # output in Algorithm 3
    idx = 1
    for i in tqdm(burnIn:stepSize:nOuter)
        uyLS = []
        U = []
        for u in 1:nU
            push!(uyLS, posteriorsample[i][:uyLS=>u=>:LS])
            push!(U, posteriorsample[i][:U=>u=>:U])
        end

        doT = parsed_args["doT"]

        if X == nothing
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
    # n x m matrix where n is the number of data,
    # and m is the number of samples

    meanITE = mean(ITEsamples, dims = 2)[:, 1]
    lowerITE = broadcast(quantile, [ITEsamples[i, :] for i in 1:size(ITEsamples)[1]], 0.05)
    upperITE = broadcast(quantile, [ITEsamples[i, :] for i in 1:size(ITEsamples)[1]], 0.95)


    df = DataFrame(Individual = 1:size(meanITE)[1], Mean = meanITE, LowerBound = lowerITE, UpperBound = upperITE)
    CSV.write(parsed_args["output_filepath"], df)
    println("Saved ITE mean and 90% credible intervals to " * parsed_args["output_filepath"])
end

main()
