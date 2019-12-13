module GenerateData

import TOML
using StatsBase
using LinearAlgebra
using Base
using Gen

include("synthetic.jl")
using .Synthetic
export generate_synthetic

function generate_synthetic(config_path::String)
    """
    return SigmaU, T, X, Y, epsY
    """
    config = TOML.parsefile(config_path)
    n = config["data"]["n"]
    obj_size = config["data"]["obj_size"]
    eps = config["data"]["eps"]
    ucov = config["data"]["ucov"]
    xvar = config["data"]["xvar"]

    # variance for data (used in additive noise model)
    tNoise = config["data"]["tNoise"]
    xNoise = config["data"]["xNoise"]
    uNoise = config["data"]["uNoise"]
    yNoise = config["data"]["yNoise"]

    SigmaU = generateSigmaU(n, [obj_size for i in 1:n/obj_size], eps, ucov)
    SigmaX = generateSigmaX(n, xvar, eps)

    n = size(SigmaU)[1]
    X = mvnormal(zeros(n), SigmaX * xNoise)
    U = mvnormal(zeros(n), SigmaU * uNoise)

    T, Y, epsY = NaN, NaN, NaN
    ft, ftx, ftxu = NaN, NaN, NaN  # causal queries

    # assignment for T
    dtypex = config["data"]["XTAssignment"]
    xtparams = config["data"]["XTparams"]
    dtypeu = config["data"]["UTAssignment"]
    utparams = config["data"]["UTparams"]

    T = generateT(X, U, dtypex, dtypeu, xtparams, utparams, xNoise)

    dtypex = config["data"]["XYAssignment"]
    xyparams = config["data"]["XYparams"]
    dtypeu = config["data"]["UYAssignment"]
    uyparams = config["data"]["UYparams"]
    dtypet = config["data"]["TYAssignment"]
    typarams = config["data"]["TYparams"]

    Y, epsY = generateY(X, U, T, dtypex, dtypeu, dtypet, xyparams, uyparams, typarams, yNoise)

    ft = generate_ft(dtypet, typarams)
    ftx = generate_ftx(dtypet, dtypex, typarams, xyparams)
    ftxu = generate_ftxu(dtypet, dtypex, dtypeu, typarams, xyparams, uyparams)

    causal_query = collect((ft, ftx, ftxu))
    return SigmaU, U, T, X, Y, epsY, causal_query
end

end