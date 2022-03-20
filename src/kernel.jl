export rbfKernelLog, logit, expit, processCov

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

SupportedRBFMatrix = Union{
    Array{Float64,1},Array{Int64,1},Array{Bool,1},FunctionalCollections.PersistentVector{Bool}
}

function rbfKernelLog(X1::SupportedRBFMatrix, X2::SupportedRBFMatrix, LS::Float64)
    return -((X1 .- X2') / LS) .^ 2
end


"""Logit maps a [0,1] onto the Real values"""
logit(prob)::Real = log(prob / (1 - prob))

"""Expit is the inverse of the logit function, mapping a Real to [0,1]"""
expit(x::Real) = exp(x) / (1.0 + exp(x))


"""
Convert covariance matrix back from log-space, scale and add noise (if passed)
"""
function processCov(logCov::Array{Float64}, scale::Float64, noise::Float64)
    return exp.(logCov) * scale + 1I * noise
end

function processCov(logCov::Array{Float64}, scale::Float64)
    return exp.(logCov) * scale
end
