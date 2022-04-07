export rbfKernelLog, logit, expit, processCov

"""
Radial Basis Function Kernel applied element-wise to two vectors `X1` and `X2` passed

Params:
- `X1`: First array of values
- `X2`: Second array of values
- `LS`: Lengthscale array

Output normalized by `LS` squared
"""
function rbfKernelLog(X1::SupportedRBFVector, X2::SupportedRBFVector, LS::SupportedRBFLengthscale)
    return -broadcast(/, ((X1 .- X2') .^ 2,), LS .^ 2)
end


function rbfKernelLog(X1::SupportedRBFVector, X2::SupportedRBFVector, LS::Float64)
    return -((X1 .- X2') / LS) .^ 2
end

"""
2D rbfKernelLog
"""
function rbfKernelLog(X1::SupportedRBFMatrix, X2::SupportedRBFMatrix, LS::Union{SupportedRBFLengthscale,Float64})
    sum([rbfKernelLog(X1[i, :], X2[i, :], LS) for i in 1:size(X1, 1)])
end

function rbfKernelLog(X1::SupportedRBFData, X2::SupportedRBFData, LS::Union{SupportedRBFLengthscale,Float64})
    sum(broadcast(rbfKernelLog, X1, X2, LS))
end


"""Logit maps a [0,1] onto the Real values"""
logit(prob)::Real = log(prob / (1 - prob))

"""Expit is the inverse of the logit function, mapping a Real to [0,1]"""
expit(x::Real) = exp(x) / (1.0 + exp(x))


"""
Convert covariance matrix back from log-space, scale and add noise (if passed)
"""
function processCov(logCov::Union{Float64,Array{Float64}}, scale::Float64, noise::Float64)
    return exp.(logCov) * scale + 1I * noise
end

function processCov(logCov::Union{Float64,Array{Float64}}, scale::Float64)
    return exp.(logCov) * scale
end
