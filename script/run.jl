using Random
using CSV
using ArgParse
using DataFrames
using LinearAlgebra
Random.seed!(1234)

include("../src/inference.jl")
using .Inference

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
        "--nOuter"
            help = "the number of posterior steps"
            default = 5000
            arg_type = Int
        "--nMHInner"
            help = "the number of metropolis hastings sampling steps"
            default = 3
            arg_type = Int
        "--nU"
            help = "the number of metropolis hastings sampling steps"
            default = 3
            arg_type = Int
        "--nESInner"
            help = "the number of elliptical slice sampling steps"
            default = 2
            arg_type = Int
        "--uNoiseShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--uNoiseScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--xNoiseShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--xNoiseScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--tNoiseShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--tNoiseScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--yNoiseShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--yNoiseScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--xScaleShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--xScaleScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--tScaleShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--tScaleScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--yScaleShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--yScaleScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--uxLSShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--uxLSScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--utLSShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--utLSScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--xtLSShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--xtLSScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--uyLSShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--uyLSScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--xyLSShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--xyLSScale"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--tyLSShape"
            help = "the number of elliptical slice sampling steps"
            default = 4.0
            arg_type = Float64
        "--tyLSScale"
            help = "the number of elliptical slice sampling steps"
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


    nOuter = parsed_args["nOuter"]
    nMHInner = parsed_args["nMHInner"]
    nESInner = parsed_args["nESInner"]
    nU = parsed_args["nU"]

    Posterior(parsed_args, X, T, Y, nU, nOuter, nMHInner, nESInner)
end

main()