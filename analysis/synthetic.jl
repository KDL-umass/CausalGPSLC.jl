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
export generateSigmaU, simLinearData
# -

# # Synthetic Data Generators
function generateSigmaU(n::Int, nIndividualsArray::Array{Int}, eps::Float64, cov::Float64)
    SigmaU = Matrix{Float64}(I, n, n)

    i = 1
    for nIndividuals in nIndividualsArray
        SigmaU[i:i+nIndividuals-1,i:i+nIndividuals-1] = ones(nIndividuals, nIndividuals) * cov
        i+= nIndividuals
    end
    
    SigmaU[diagind(SigmaU)] .= 1 + eps
    
    return SigmaU + Matrix{Float64}(I, n, n) * eps
end

# +
function simLinearData(SigmaU, tNoise, yNoise, uNoise, UTslope, UYslope, TYslope)
    n = size(SigmaU)[1]
    U = zeros(n)
    T = zeros(n)
    Y = zeros(n)
    epsY = zeros(n)

    U = mvnormal(zeros(n), SigmaU * uNoise)

    for i in 1:n
        T[i] = normal(U[i] * UTslope, tNoise)
        Ymean = (T[i] * TYslope + U[i] * UYslope)
        epsY[i] = normal(0, yNoise)
        Y[i] = Ymean + epsY[i]
    end
    return U, T, Y, epsY
end

function simLinearIntData(U, epsY, doT, TYslope, UYslope)
    n = length(epsY)
    
    Yint = zeros(n)
    
    for i in 1:n
        Ymean = (doT * TYslope + U[i] * UYslope)
        Yint[i] = Ymean + epsY[i]
    end
    return Yint
end

end
