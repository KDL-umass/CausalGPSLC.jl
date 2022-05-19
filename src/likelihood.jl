export likelihoodDistribution

"""
    likelihoodDistribution

A utility that uses multiple dispatch to take in one set of parameters for GPSLC and the observed data, plus an intervention `doT`, and outputs the necessary matrices to compute MeanITE and CovITE, as well as other predictions, like directly predicting ``Y_cf``.
"""
function likelihoodDistribution(
    uyLS::Vector{Float64}, xyLS::Array{Float64}, tyLS::Float64,
    yNoise::Float64, yScale::Float64,
    U::Confounders, X::Covariates, T::Treatment,
    Y::Outcome, doT::Intervention)

    n = size(Y, 1)
    @assert size(U, 1) == n
    @assert size(X, 1) == n
    @assert size(T, 1) == n
    nU = ndims(U) == 2 ? size(U, 2) : 1
    @assert size(uyLS, 1) == nU
    nX = ndims(X) == 2 ? size(X, 2) : 1
    @assert size(xyLS, 1) == nX

    # e.g. U's contribution to log cov mat of Y
    uyCovLog = rbfKernelLog(U, U, uyLS)
    xyCovLog = rbfKernelLog(X, X, xyLS)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    tyCovLogS = rbfKernelLog(T, fill(doT, n), tyLS)
    tyCovLogSS = rbfKernelLog(fill(doT, n), fill(doT, n), tyLS)

    CovWW = processCov(uyCovLog + xyCovLog + tyCovLog, yScale, 0.0)
    CovWW = Symmetric(CovWW)
    CovWWp = Symmetric(CovWW + (yNoise * 1I))

    #   K(W, W_*) in the paper. The cross covariance matrix is not in general symettric.
    CovWWs = processCov(uyCovLog + xyCovLog + tyCovLogS, yScale, 0.0)

    #   K(W_*, W_*) in the paper.
    CovWsWs = processCov(uyCovLog + xyCovLog + tyCovLogSS, yScale, 0.0)
    CovWsWs = Symmetric(CovWsWs)

    #   Intermediate inverse products to avoid repeated computation.
    CovWWpInvCovWW = CovWWp \ CovWW
    CovWWpInvCovWWs = CovWWp \ CovWWs

    #   Covariance of P([f, f_*]|Y)
    CovC11 = CovWW - (CovWW * CovWWpInvCovWW)
    CovC12 = CovWWs - (CovWW * CovWWpInvCovWWs)
    CovC21 = CovWWs' - (CovWWs' * CovWWpInvCovWW)
    CovC22 = CovWsWs - (CovWWs' * CovWWpInvCovWWs)

    return Y, CovWW, CovWWs, CovWWp, CovC11, CovC12, CovC21, CovC22
end

"""No Covariates Continuous/Binary"""
function likelihoodDistribution(
    uyLS::Vector{Float64}, xyLS::Nothing, tyLS::Float64,
    yNoise::Float64, yScale::Float64,
    U::Confounders, X::Nothing, T::Treatment,
    Y::Outcome, doT::Intervention)

    n = size(Y, 1)
    @assert size(U, 1) == n
    @assert size(T, 1) == n
    nU = ndims(U) == 2 ? size(U, 2) : 1
    @assert size(uyLS, 1) == nU

    uyCovLog = rbfKernelLog(U, U, uyLS)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    tyCovLogS = rbfKernelLog(T, fill(doT, n), tyLS)
    tyCovLogSS = rbfKernelLog(fill(doT, n), fill(doT, n), tyLS)

    CovWW = processCov(uyCovLog .+ tyCovLog, yScale, 0.0)
    CovWW = Symmetric(CovWW)
    CovWWp = Symmetric(CovWW + (yNoise * 1I))

    #   K(W, W_*) in the paper. The cross covariance matrix is not in general symettric.
    CovWWs = processCov(uyCovLog .+ tyCovLogS, yScale, 0.0)

    #   K(W_*, W_*) in the paper.
    CovWsWs = processCov(uyCovLog .+ tyCovLogSS, yScale, 0.0)
    CovWsWs = Symmetric(CovWsWs)

    #   Intermediate inverse products to avoid repeated computation.
    CovWWpInvCovWW = CovWWp \ CovWW
    CovWWpInvCovWWs = CovWWp \ CovWWs

    #   Covariance of P([f, f_*]|Y)
    CovC11 = CovWW - (CovWW * CovWWpInvCovWW)
    CovC12 = CovWWs - (CovWW * CovWWpInvCovWWs)
    CovC21 = CovWWs' - (CovWWs' * CovWWpInvCovWW)
    CovC22 = CovWsWs - (CovWWs' * CovWWpInvCovWWs)

    return Y, CovWW, CovWWs, CovWWp, CovC11, CovC12, CovC21, CovC22
end

"""No Confounders Continuous/Binary"""
function likelihoodDistribution(
    uyLS::Nothing, xyLS::Vector{Float64}, tyLS::Float64,
    yNoise::Float64, yScale::Float64,
    U::Nothing, X::Covariates, T::Treatment,
    Y::Outcome, doT::Intervention)

    n = size(Y, 1)
    @assert size(X, 1) == n
    @assert size(T, 1) == n
    nX = ndims(X) == 2 ? size(X, 2) : 1
    @assert size(xyLS, 1) == nX

    xyCovLog = rbfKernelLog(X, X, xyLS)
    tyCovLog = rbfKernelLog(T, T, tyLS)
    tyCovLogS = rbfKernelLog(T, fill(doT, n), tyLS)
    tyCovLogSS = rbfKernelLog(fill(doT, n), fill(doT, n), tyLS)

    CovWW = processCov(xyCovLog + tyCovLog, yScale, 0.0)
    CovWW = Symmetric(CovWW)
    CovWWp = Symmetric(CovWW + (yNoise * 1I))

    #   K(W, W_*) in the paper. The cross covariance matrix is not in general symettric.
    CovWWs = processCov(xyCovLog + tyCovLogS, yScale, 0.0)

    #   K(W_*, W_*) in the paper.
    CovWsWs = processCov(xyCovLog + tyCovLogSS, yScale, 0.0)
    CovWsWs = Symmetric(CovWsWs)

    #   Intermediate inverse products to avoid repeated computation.
    CovWWpInvCovWW = CovWWp \ CovWW
    CovWWpInvCovWWs = CovWWp \ CovWWs

    #   Covariance of P([f, f_*]|Y)
    CovC11 = CovWW - (CovWW * CovWWpInvCovWW)
    CovC12 = CovWWs - (CovWW * CovWWpInvCovWWs)
    CovC21 = CovWWs' - (CovWWs' * CovWWpInvCovWW)
    CovC22 = CovWsWs - (CovWWs' * CovWWpInvCovWWs)

    return Y, CovWW, CovWWs, CovWWp, CovC11, CovC12, CovC21, CovC22
end

"""No Confounders No Covariates Continuous/Binary"""
function likelihoodDistribution(
    uyLS::Nothing, xyLS::Nothing, tyLS::Float64,
    yNoise::Float64, yScale::Float64,
    U::Nothing, X::Nothing, T::Treatment,
    Y::Outcome, doT::Intervention)

    n = size(Y, 1)
    @assert size(T, 1) == n

    tyCovLog = rbfKernelLog(T, T, tyLS)
    tyCovLogS = rbfKernelLog(T, fill(doT, n), tyLS)
    tyCovLogSS = rbfKernelLog(fill(doT, n), fill(doT, n), tyLS)

    CovWW = processCov(tyCovLog, yScale, 0.0)
    CovWW = Symmetric(CovWW)
    CovWWp = Symmetric(CovWW + (yNoise * 1I))

    #   K(W, W_*) in the paper. The cross covariance matrix is not in general symettric.
    CovWWs = processCov(tyCovLogS, yScale, 0.0)

    #   K(W_*, W_*) in paper.
    CovWsWs = processCov(tyCovLogSS, yScale, 0.0)
    CovWsWs = Symmetric(CovWsWs)

    #   Intermediate inverse products to avoid repeated computation.
    CovWWpInvCovWW = CovWWp \ CovWW
    CovWWpInvCovWWs = CovWWp \ CovWWs

    #   Covariance of P([f, f_*]|Y)
    CovC11 = CovWW - (CovWW * CovWWpInvCovWW)
    CovC12 = CovWWs - (CovWW * CovWWpInvCovWWs)
    CovC21 = CovWWs' - (CovWWs' * CovWWpInvCovWW)
    CovC22 = CovWsWs - (CovWWs' * CovWWpInvCovWWs)

    return Y, CovWW, CovWWs, CovWWp, CovC11, CovC12, CovC21, CovC22
end