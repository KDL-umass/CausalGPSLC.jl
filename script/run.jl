using Random
using CSV
using ArgParse
using DataFrames
using LinearAlgebra
using ProgressBars
using Statistics
using Distributions
Random.seed!(1234)

include("../src/inference.jl")
using .Inference

include("../src/estimation.jl")
using .Estimation

function generateSigmaU(nIndividualsArray::Array{Int}, eps::Float64=1e-13, cov::Float64=1.0)
    """
    generate covariance matrix for U given object config
    """
    n = sum(nIndividualsArray)
    SigmaU = Matrix{Float64}(I, n, n)
    i = 1
    for nIndividuals in nIndividualsArray
        SigmaU[i:i+nIndividuals-1,i:i+nIndividuals-1] = ones(nIndividuals, nIndividuals) * cov
        i += nIndividuals
    end

    SigmaU[diagind(SigmaU)] .= 1 + eps
    return SigmaU
end

function uniq(v)
  v1 = Vector{eltype(v)}()
  if length(v)>0
    laste = v[1]
    push!(v1,laste)
    for e in v
      if e != laste
        laste = e
        push!(v1,laste)
      end
    end
  end
  return v1
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "datapath"
            help = "a path to the data"
            arg_type = String
            required = true
        # posterior updates
        "--nOuter"
            help = "the number of posterior steps"
            default = 5000
            arg_type = Int
        "--nMHInner"
            help = "the number of metropolis hastings sampling steps"
            default = 3
            arg_type = Int
        "--nESInner"
            help = "the number of elliptical slice sampling steps"
            default = 2
            arg_type = Int
        "--nU"
            help = "the dimension of latent confounders to model"
            default = 3
            arg_type = Int
        # inference
        "--burnIn"
            help = "the number of posterior samples for burn-in"
            default = 10
            arg_type = Int
        "--stepSize"
            help = "the step size during the inference step"
            default = 10
            arg_type = Int
        "--samplesPerPost"
            help = "the number of samples from each posterior for treatment effect approximation"
            default = 100
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
    parsed_args = parse_commandline()
    csv_path = parsed_args["datapath"]
    df = CSV.read(csv_path)

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

    # generate a block matrix based on object counts
    sigU = generateSigmaU(obj_count)
    parsed_args["SigmaU"] = sigU

    # prepare inputs
    T = Array(df[!, :T])
    Y = Array(df[!, :Y])

    cols = names(df)
    cols = deleteat!(cols, cols .== :T)
    cols = deleteat!(cols, cols .== :Y)
    cols = deleteat!(cols, cols .== :obj)
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

    println("posterior sampling...")
    posteriorsample, _ = Posterior(parsed_args, X, T, Y, nU, nOuter, nMHInner, nESInner)

    # inference of treatment effects
    burnIn = parsed_args["burnIn"]
    stepSize = parsed_args["stepSize"]
    samplesPerPost = parsed_args["samplesPerPost"]

    println("inference...")
    ITEsamples = zeros(length(T), samplesPerPost*length(burnIn:stepSize:nOuter)) # output in Algorithm 3
    idx = 1
    for i in tqdm(burnIn:stepSize:nOuter)
        uyLS = []
        U = []
        for u in 1:nU
            push!(uyLS, posteriorsample[i][:uyLS => u => :LS])
            push!(U, posteriorsample[i][:U => u => :U])
        end

        # Intervention assignment in algorithm 3. Use the mean of observed T as demo
        doT = mean(T)

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
        m = MeanITE .+ Y
        v = Symmetric(CovITE[:, :]) + I*(1e-10)
        for j in 1:samplesPerPost
            samples = rand(MvNormal(m, v)) # Yobs+ITE for all data when intervened to be T = doT
            ITEsamples[:, idx] = samples
            idx += 1
        end
    end
    # n x m matrix where n is the number of data,
    # and m is the number of samples
    ITEsamples
end

main()