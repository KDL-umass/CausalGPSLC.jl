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
    return SigmaU, U, T, X, Y, epsY, causal_query
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

    # generate X and U
    X = mvnormal(zeros(size(SigmaU)[1]), SigmaX * xNoise)
    U = mvnormal(zeros(size(SigmaU)[1]), SigmaU * uNoise)

    # assignment for T
    dtypex = config["data"]["XTAssignment"]
    xtparams = config["data"]["XTparams"]
    dtypeu = config["data"]["UTAssignment"]
    utparams = config["data"]["UTparams"]
    T = generateT(X, U, dtypex, dtypeu, xtparams, utparams, xNoise)

    # assignment for Y
    dtypex = config["data"]["XYAssignment"]
    xyparams = config["data"]["XYparams"]
    dtypeu = config["data"]["UYAssignment"]
    uyparams = config["data"]["UYparams"]
    dtypet = config["data"]["TYAssignment"]
    typarams = config["data"]["TYparams"]
    Y, epsY = generateY(X, U, T, dtypex, dtypeu, dtypet, xyparams, uyparams, typarams, yNoise)

    # recover true causal assignment
    ft = generate_ft(dtypet, typarams) # function of T
    ftx = generate_ftx(dtypet, dtypex, typarams, xyparams) # function of T and X
    ftxu = generate_ftxu(dtypet, dtypex, dtypeu, typarams, xyparams, uyparams) # function of T and X and U

    causal_query = collect((ft, ftx, ftxu))
    return SigmaU, U, T, X, Y, epsY, causal_query
end

end