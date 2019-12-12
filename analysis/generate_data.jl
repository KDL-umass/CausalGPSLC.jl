module GenerateData

import TOML

include("synthetic.jl")
using .Synthetic

function generate_synthetic(config_path::String)
    """
    return SigmaU, T, X, Y, epsY
    """
    config = TOML.parsefile(config_path)
    n = config["data"]["n"]
    num_obj = config["data"]["num_obj"]
    eps = config["data"]["eps"]
    ucov = config["data"]["ucov"]
    dtype = config["data"]["dataType"]

    # variance for data
    tNoise = config["data"]["tNoise"]
    xNoise = config["data"]["xNoise"]
    uNoise = config["data"]["uNoise"]
    yNoise = config["data"]["yNoise"]

    SigmaU = generateSigmaU(n, [num_obj for i in 1:n/num_obj], eps, ucov)
    U, T, X, Y, epsY = Nothing, Nothing, Nothing, Nothing, Nothing

    if dtype == "linear"
        # parameter for linear data
        UTslope = config["data"]["linear"]["UTslope"]
        UYslope = config["data"]["linear"]["UYslope"]
        TYslope = config["data"]["linear"]["TYslope"]
        U, T, Y, epsY = simLinearData(SigmaU, tNoise, yNoise, uNoise, UTslope, UYslope, TYslope)
    end
    return SigmaU, U, T, X, Y, epsY
end

end