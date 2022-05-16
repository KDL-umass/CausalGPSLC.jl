export rbfKernelLog, rbfKernelLogScalar, logit, expit, processCov

"""
Radial Basis Function Kernel applied element-wise to two vectors `X1` and `X2` passed

Params:
- `X1`: First array of values
- `X2`: Second array of values
- `LS`: Lengthscale array

Output normalized by `LS` squared
"""
function rbfKernelLogScalar(Xi::SupportedRBFVector, Xiprime::SupportedRBFVector, LS::SupportedRBFLengthscale)
    @assert (size(LS, 1) == size(Xi, 1)
             ||
             size(LS) == ()) "vector lengthscale doesn't match individual"
    return -sum((Xi .- Xiprime) .^ 2 ./ LS .^ 2) # denom of exp vs sqrt of denom of exp
    # TODO: try running without LS.^2
end

"""
2D rbfKernelLog
"""
function rbfKernelLog(X1::SupportedRBFMatrix, X2::SupportedRBFMatrix, LS::SupportedRBFLengthscale)
    @assert size(X1) == size(X2) "X1 and X2 are different sizes!"
    n = size(X1, 1)
    cov = zeros(n, n)
    for i = 1:n, ip = 1:n
        cov[i, ip] = rbfKernelLogScalar(X1[i, :], X2[ip, :], LS)
    end
    return cov
end

function rbfKernelLog(X1::SupportedRBFData, X2::SupportedRBFData, LS::SupportedRBFLengthscale)
    n = size(X1, 1)
    @assert size(X1) == size(X2) "X1 and X2 are different sizes!"
    cov = zeros(n, n)
    for i = 1:n, ip = 1:n
        cov[i, ip] = rbfKernelLogScalar(X1[i], X2[ip], LS)
    end
    return cov
end


"""Logit maps a [0,1] onto the Real values"""
logit(prob)::Real = log(prob / (1 - prob))

"""Expit is the inverse of the logit function, mapping a Real to [0,1]"""
expit(x::Real) = exp(x) / (1.0 + exp(x))


"""Convert covariance matrix back from log-space, scale and add noise (if passed)"""
function processCov(logCov::Union{Float64,Array{Float64}}, scale::Union{Float64,Array{Float64}}, noise::Float64)
    return exp.(logCov) * scale + 1I * noise
end

function processCov(logCov::Union{Float64,Array{Float64}}, scale::Float64)
    return exp.(logCov) * scale
end
