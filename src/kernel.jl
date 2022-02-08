module Kernel

using LinearAlgebra

import FunctionalCollections

export rbfKernelLog, processCov, expit

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

expit(x::Real) = exp(x) / (1.0 + exp(x))

function processCov(logCov::Array{Float64}, scale::Float64, noise::Float64)
    return exp.(logCov) * scale + 1I * noise
end

function processCov(logCov::Array{Float64}, scale::Float64, noise::Nothing)
    return exp.(logCov) * scale
end

end