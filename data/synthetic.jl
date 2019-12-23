module Synthetic

import TOML
using Random
Random.seed!(1234)
using Gen
using LinearAlgebra
using StatsBase
using StatsFuns
using Statistics
using SparseArrays

export generate_synthetic_confounder, generate_synthetic_collider, generateSigmaU


function generateSigmaU(n::Int, nIndividualsArray::Array{Int}, eps::Float64, cov::Float64)
    """
    generate covariance matrix for U given object config
    """
    SigmaU = Matrix{Float64}(I, n, n)
    i = 1
    for nIndividuals in nIndividualsArray
        SigmaU[i:i+nIndividuals-1,i:i+nIndividuals-1] = ones(nIndividuals, nIndividuals) * cov
        i += nIndividuals
    end

    SigmaU[diagind(SigmaU)] .= 1 + eps
    return SigmaU
end

function generateSigmaX(n::Int, sigma, eps::Float64)
    """
    generate covariance matrix for X
    """
    SigmaX = Matrix{Float64}(I, n, n)
    SigmaX[diagind(SigmaX)] .= sigma .+ eps
    return SigmaX
end

# transform input
function PolyTransform(X, beta::Array{Float64})
    """
    X <- beta[1] * X + beta[2] * X^2 + ... + beta[p] * X^3
    """
    n = size(X)[1]
    p = size(beta)[1]
    Y = zeros(n)
    isMat = (length(size(X)) > 1)
    for i in 1:n
        for d in 1:p
            if isMat
                Y[i] += sum(beta[d].* X[i, :].^(d-1))
            else
                Y[i] += beta[d]*X[i]^(d-1)
            end
        end
    end
    return Y
end

function ExpTransform(X, beta::Array{Float64})
    """
    X <- beta1 * exp(beta2 * X)
    """
    n = size(X)[1]
    Y = zeros(n)
    isMat = (length(size(X)) > 1)
    for i in 1:n
        if isMat
            Y[i] = sum(beta[1] .+ beta[2] .* exp.(beta[3] .* X[i, :]))
        else
            Y[i] = beta[1] + beta[2] * exp(beta[3] * X[i])
        end
    end
    return Y
end

function SinTransform(X, beta::Array{Float64})
    """
    X <- beta1 * sin(beta2 * X)
    """
    n = size(X)[1]
    Y = zeros(n)
    isMat = (length(size(X)) > 1)
    for i in 1:n
        if isMat
            Y[i] = sum(beta[1] .+ beta[2] .* sin.(beta[3] .* X[i, :]))
        else
            Y[i] = beta[1] + beta[2] * sin(beta[3] * X[i])
        end
    end
    return Y
end

function AggregateTransform(X, dtype::Array{String}, param)
    """
    transform r.v input X via f(X) based on config
    """
    X_ = zeros(size(X)[1])
    if sum(X) == 0
        return X
    else
        for (i, value) in enumerate(dtype)
            if value == "polynomial"
                beta = float.(param["poly"])
                if (param["aggOp"] == "*") && (i != 1)
                    X_ = X_ .* PolyTransform(X, beta)
                else
                    X_ = X_ .+ PolyTransform(X, beta)
                end
            elseif value == "exponential"
                beta = float.(param["exp"])
                if (param["aggOp"] == "*") && (i != 1)
                    X_ = X_ .* ExpTransform(X, beta)
                else
                    X_ = X_ .+ ExpTransform(X, beta)
                end
            else
                beta = float.(param["sin"])
                if (param["aggOp"] == "*") && (i != 1)
                    X_ = X_ .* SinTransform(X, beta) .+ 1
                else
                    X_ = X_ .+ SinTransform(X, beta)
                end
            end
        end
        return X_
    end
end

function generateT(X::Array{Float64}, U::Array{Float64}, dtypeX::Array{String}, dtypeU::Array{String},
    Xparam, Uparam, tNoise::Float64, aggOp::String, Ttype::String)
    """
    generate T with additive noise based on config. T = f(X) + g(U) + eps
    """
    n = size(X)[1]
    T = zeros(n)
    X_ = AggregateTransform(X, dtypeX, Xparam)
    U_ = AggregateTransform(U, dtypeU, Uparam)
    for i in 1:n
        if aggOp == "+"
            T[i] = normal(X_[i] + U_[i], tNoise)
        else
            T[i] = normal(X_[i] * U_[i], tNoise)
        end
    end
    if Ttype == "binary"
        T = Float64.(bernoulli.(logistic.(T)))
    else
        T = (T .- mean(T)) ./ std(T)
    end
    return T
end

function generateT(X::Array{Float64}, dtypeX::Array{String},
    Xparam, tNoise::Float64, Ttype::String)
    """
    generate T with additive noise based on config. T = f(X) + g(U) + eps
    """
    n = size(X)[1]
    T = zeros(n)
    X_ = AggregateTransform(X, dtypeX, Xparam)
    for i in 1:n
        T[i] = normal(X_[i], tNoise)
    end
    if Ttype == "binary"
        T = Float64.(bernoulli.(logistic.(T)))
    else
        T = (T .- mean(T)) ./ std(T)
    end
    return T
end

function generateY(X::Array{Float64}, U::Array{Float64}, T::Array{Float64},
    dtypeX::Array{String}, dtypeU::Array{String}, dtypeT::Array{String},
    Xparam, Uparam, Tparam, yNoise::Float64, aggOp::String)
    """
    generate Y with additive noise based on config. Y = f(X) + g(U) + h(T) + eps
    """
    n = size(X)[1]
    Y = zeros(n)
    epsY = zeros(n)
    X_ = AggregateTransform(X, dtypeX, Xparam)
    U_ = AggregateTransform(U, dtypeU, Uparam)
    T_ = AggregateTransform(T, dtypeT, Tparam)

    for i in 1:n
        epsY[i] = normal(0, yNoise)
        if aggOp == "+"
            Y[i] = T_[i] + U_[i] + X_[i] + epsY[i]
        else
            Y[i] = T_[i] * U_[i] * X_[i] + epsY[i]
        end
    end
    return Y, epsY
end

function generateY(X::Array{Float64}, T::Array{Float64},
    dtypeX::Array{String}, dtypeT::Array{String},
    Xparam, Tparam, yNoise::Float64, aggOp::String)
    """
    generate Y with additive noise based on config. Y = f(X) + g(U) + h(T) + eps
    """
    n = size(X)[1]
    Y = zeros(n)
    epsY = zeros(n)
    X_ = AggregateTransform(X, dtypeX, Xparam)
    T_ = AggregateTransform(T, dtypeT, Tparam)

    for i in 1:n
        epsY[i] = normal(0, yNoise)
        if aggOp == "+"
            Y[i] = T_[i] + X_[i] + epsY[i]
        else
            Y[i] = T_[i] * X_[i] + epsY[i]
        end
    end
    return Y, epsY
end


# generate causal queries based on config. Recover h(T), h(T)+f(X), h(T)+f(X)+g(U)
function generate_ftxu(dtypeT::Array{String}, dtypeX::Array{String}, dtypeU::Array{String},
    Tparam, Xparam, Uparam, aggOp::String)

    function ftxu(T::Array{Float64}, X, U::Array{Float64}, epsY::Array{Float64})
        T_ = AggregateTransform(T, dtypeT, Tparam)
        X_ = AggregateTransform(X, dtypeX, Xparam)
        U_ = AggregateTransform(U, dtypeU, Uparam)
        if aggOp == "+"
            Y = T_ .+ X_ .+ U_
        else
            Y = T_ .* X_ .* U_
        end
        return Y .+ epsY
    end
    return ftxu
end

function generate_ftxu(dtypeT::Array{String}, dtypeX::Array{String},
    Tparam, Xparam, aggOp::String)

    function ftxu(T::Array{Float64}, X, U::Array{Float64}, epsY::Array{Float64})
        T_ = AggregateTransform(T, dtypeT, Tparam)
        X_ = AggregateTransform(X, dtypeX, Xparam)
        if aggOp == "+"
            Y = T_ .+ X_ 
        else
            Y = T_ .* X_
        end
        return Y .+ epsY
    end
    return ftxu
end

function generate_synthetic_confounder(config_path::String)
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

    # generate X and U
    # assume indep X
    xdim = config["data"]["xdim"]
    SigmaX = generateSigmaX(n, xvar, eps)

    udim = config["data"]["udim"]
    SigmaU = generateSigmaU(n, [obj_size for i in 1:n/obj_size], eps, ucov)
    U = zeros(size(SigmaU)[1], udim)    # N x dim
    for i in 1:udim
        U[:, i] .= mvnormal(zeros(size(SigmaU)[1]), SigmaU * uNoise)
    end
    if xdim == 0
        X = zeros(n, 1)
    else
        X = zeros(n, xdim)
        W = Array(sprandn(udim, xdim, 0.5))  # random sparse weight matrix
        for i in 1:n
            for j in 1:xdim
                X[i, j] = normal(sum(U[i, :] .* W[:, j]), xNoise)
            end
        end
    end

    # assignment for T
    dtypex = string.(config["data"]["XTAssignment"])
    xtparams = config["data"]["XTparams"]
    dtypeu = string.(config["data"]["UTAssignment"])
    utparams = config["data"]["UTparams"]
    aggOp = config["data"]["TaggOp"]
    Ttype = config["data"]["Ttype"]
    T = generateT(X, U, dtypex, dtypeu, xtparams, utparams, xNoise, aggOp, Ttype)

    # assignment for Y
    dtypex = string.(config["data"]["XYAssignment"])
    xyparams = config["data"]["XYparams"]
    dtypeu = string.(config["data"]["UYAssignment"])
    uyparams = config["data"]["UYparams"]
    dtypet = string.(config["data"]["TYAssignment"])
    typarams = config["data"]["TYparams"]
    aggOp = config["data"]["YaggOp"]
    Y, epsY = generateY(X, U, T, dtypex, dtypeu, dtypet, xyparams, uyparams, typarams, yNoise, aggOp)

    # recover true causal assignment
    ftxu = generate_ftxu(dtypet, dtypex, dtypeu, typarams, xyparams, uyparams, aggOp) # function of T and X and U

    return SigmaU, U, T, X, Y, epsY, ftxu
end

function generate_synthetic_collider(config_path::String)
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

    SigmaX = generateSigmaX(n, xvar, eps)

    # generate X
    # assume indep X
    xdim = config["data"]["xdim"]
    if xdim == 0
        X = zeros(size(SigmaX)[1], 1)
    else
        X = zeros(size(SigmaX)[1], xdim)    # N x dim
        for i in 1:xdim
            X[:, i] .= mvnormal(zeros(size(SigmaX)[1]), SigmaX * xNoise)
        end
    end

    # assignment for T
    dtypex = string.(config["data"]["XTAssignment"])
    xtparams = config["data"]["XTparams"]
    dtypeu = string.(config["data"]["UTAssignment"])
    utparams = config["data"]["UTparams"]
    aggOp = config["data"]["TaggOp"]
    Ttype = config["data"]["Ttype"]
    T = generateT(X, dtypex, xtparams, xNoise, Ttype)

    # assignment for Y
    dtypex = string.(config["data"]["XYAssignment"])
    xyparams = config["data"]["XYparams"]
    dtypeu = string.(config["data"]["UYAssignment"])
    uyparams = config["data"]["UYparams"]
    dtypet = string.(config["data"]["TYAssignment"])
    typarams = config["data"]["TYparams"]
    aggOp = config["data"]["YaggOp"]
    Y, epsY = generateY(X, T, dtypex, dtypet, xyparams, typarams, yNoise, aggOp)

    # recover true causal assignment
    ftxu = generate_ftxu(dtypet, dtypex, typarams, xyparams, aggOp) # function of T and X and U

    return SigmaU, U, T, X, Y, epsY, ftxu
end

end
