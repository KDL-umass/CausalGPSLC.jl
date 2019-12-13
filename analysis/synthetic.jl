module Synthetic

using Gen
using LinearAlgebra
using PyPlot
using Seaborn
using StatsBase

include("../src/inference.jl")
include("../src/estimation.jl")
include("../src/model.jl")
using .Inference
using .Estimation
using .Model
export generateSigmaU, generateSigmaX, generateT, generateY, generate_ft, generate_ftx, generate_ftxu


function generateSigmaU(n::Int, nIndividualsArray::Array{Int}, eps::Float64, cov::Float64)
    """
    generate covariance matrix for U given object config
    """
    SigmaU = Matrix{Float64}(I, n, n)
    i = 1
    for nIndividuals in nIndividualsArray
        SigmaU[i:i+nIndividuals-1,i:i+nIndividuals-1] = ones(nIndividuals, nIndividuals) * cov
        i+= nIndividuals
    end
    
    SigmaU[diagind(SigmaU)] .= 1 + eps
    return SigmaU
end


function generateSigmaX(n::Int, sigma, eps::Float64)
    """
    generate covariance matrix for X
    """
    SigmaU = Matrix{Float64}(I, n, n)
    SigmaU[diagind(SigmaU)] .= sigma .+ eps
    return SigmaU
end


# transform input
function PolyTransform(X::Array{Float64}, beta::Array{Float64})
    """
    X <- beta[1] * X + beta[2] * X^2 + ... + beta[p] * X^3
    """
    n = size(X)[1]
    p = size(beta)[1]
    Y = zeros(n)
    for i in 1:n
        for d in 1:p
            Y[i] += beta[d]*X[i]^d
        end
    end
    return Y
end

function ExpTransform(X::Array{Float64}, beta1::Float64, beta2::Float64)
    """
    X <- beta1 * exp(beta2 * X)
    """
    n = size(X)[1]
    Y = zeros(n)
    for i in 1:n
        Y[i] = beta1 * exp(beta2 * X[i])
    end
    return Y
end

function SinTransform(X::Array{Float64}, beta1::Float64, beta2::Float64)
    """
    X <- beta1 * sin(beta2 * X)
    """
    n = size(X)[1]
    Y = zeros(n)
    for i in 1:n
        Y[i] = beta1 * sin(beta2 * X[i])
    end
    return Y
end


function AggregateTransoform(X::Array{Float64}, dtype::Array{String}, param)
    """
    transform r.v input X via f(X) based on config
    """
    X_ = zeros(size(X))
    for value in dtype
        if value == "polynomial"
            beta = param["poly"]
            X_ = X_ .+ PolyTransform(X, beta)
        elseif value == "exponential"
            beta = param["exp"]
            X_ = X_ .+ ExpTransform(X, beta[1], beta[2])
        else
            beta = param["sin"]
            X_ = X_ .+ SinTransform(X, beta[1], beta[2])
        end
    end
    return X_
end

function generateT(X::Array{Float64}, U::Array{Float64}, dtypeX::Array{String}, dtypeU::Array{String},
    Xparam, Uparam, tNoise)
    """
    generate T with additive noise based on config. T = f(X) + g(U) + eps
    """
    n = size(X)[1]
    T = zeros(n)
    X_ = AggregateTransoform(X, dtypeX, Xparam)
    U_ = AggregateTransoform(U, dtypeU, Uparam)
    for i in 1:n
        T[i] = normal(X_[i] + U_[i], tNoise)
    end
    return T
end


function generateY(X::Array{Float64}, U::Array{Float64}, T::Array{Float64},
    dtypeX::Array{String}, dtypeU::Array{String}, dtypeT::Array{String},
    Xparam, Uparam, Tparam, yNoise)
    """
    generate Y with additive noise based on config. Y = f(X) + g(U) + h(T) + eps
    """
    n = size(X)[1]
    Y = zeros(n)
    epsY = zeros(n)
    X_ = AggregateTransoform(X, dtypeX, Xparam)
    U_ = AggregateTransoform(U, dtypeU, Uparam)
    T_ = AggregateTransoform(T, dtypeT, Tparam)

    for i in 1:n
        epsY[i] = normal(0, yNoise)
        Y[i] = X_[i] + U_[i] + T_[i] + epsY[i]
    end
    return Y, epsY
end


# generate causal queries based on config. Recover h(T), h(T)+f(X), h(T)+f(X)+g(U)

function generate_ft(dtypeT::Array{String}, Tparam)
    function ft(T::Array{Float64})
        T_ = AggregateTransoform(T, dtypeT, Tparam)
        return T_
    end
    return ft
end

function generate_ftx(dtypeT::Array{String}, dtypeX::Array{String}, Tparam, Xparam)
    function ftx(T::Array{Float64}, X::Array{Float64})
        T_ = AggregateTransoform(T, dtypeT, Tparam)
        X_ = AggregateTransoform(X, dtypeX, Xparam)
        return T_ .+ X_
    end
    return ftx
end

function generate_ftxu(dtypeT::Array{String}, dtypeX::Array{String}, dtypeU::Array{String},
    Tparam, Xparam, Uparam)

    function ftxu(T::Array{Float64}, X::Array{Float64}, U::Array{Float64})
        T_ = AggregateTransoform(T, dtypeT, Tparam)
        X_ = AggregateTransoform(X, dtypeX, Xparam)
        U_ = AggregateTransoform(U, dtypeU, Uparam)
        return T_ .+ X_ .+ U_
    end
    return ftxu
end


end