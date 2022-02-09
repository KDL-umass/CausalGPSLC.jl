module Kernel

using LinearAlgebra

import FunctionalCollections

export rbfKernelLog, processCov, expit

"""
Radial Basis Function Kernel applied element-wise to two vectors `X1` and `X2` passed

Params:
- `X1`: First array of values
- `X2`: Second array of values
- `LS`: Lengthscale array

Output normalized by `LS` squared
"""
function rbfKernelLog(X1::Array{Float64,1}, X2::Array{Float64,1},
    LS::Union{
        Array{Float64,1},
        FunctionalCollections.PersistentVector{Float64}
    })
    return -broadcast(/, ((X1 .- X2') .^ 2,), LS .^ 2)
end

function rbfKernelLog(X1::Array{Float64,1}, X2::Array{Float64,1}, LS::Float64)
    return -((X1 .- X2') / LS) .^ 2
end

function rbfKernelLog(X1::Array{Bool,1}, X2::Array{Bool,1}, LS::Float64)
    return -((X1 .- X2') / LS) .^ 2
end

function rbfKernelLog(X1::FunctionalCollections.PersistentVector{Bool},
    X2::FunctionalCollections.PersistentVector{Bool}, LS::Float64)
    return -((X1 .- X2') / LS) .^ 2
end


"""Logit maps a [0,1] onto the Real values"""
logit(prob)::Real = log(prob / (1 - prob))

"""Expit is the inverse of the logit function, mapping a Real to [0,1]"""
expit(x::Real) = exp(x) / (1.0 + exp(x))


"""Exponentiate and scale a log covariance matrix; add noise if passed"""
function processCov(logCov::Array{Float64}, scale::Float64,
    noise::Union{Float64,Nothing})
    if noise === nothing
        return exp.(logCov) * scale
    else
        return exp.(logCov) * scale + 1I * noise
    end
end

end